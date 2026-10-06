import * as ort from 'onnxruntime-web/webgpu';
import type { Backend, Manifest, ModelSpec, WorkerReply } from './types';

type Role = keyof Manifest['models'];
type Report = (event: WorkerReply) => void;
const roles: Role[] = ['gate', 'orient', 'dewarp'];

export async function readManifest(base: string): Promise<Manifest> {
  const response = await fetch(new URL('models/manifest.json', base));
  if (!response.ok)
    throw new Error(
      'Scanner models are not installed on this site. Ask the site owner to install the model bundle.',
    );
  let manifest: Manifest;
  try {
    manifest = await response.json();
  } catch {
    throw new Error('The model bundle is missing or invalid. Ask the site owner to install it.');
  }
  if (
    manifest.schema !== 'flattener-web-bundle-1' ||
    typeof manifest.version !== 'string' ||
    !manifest.version ||
    manifest.models?.dewarp?.grid_convention !== 'flattener'
  ) {
    throw new Error('This model bundle is not compatible with the web scanner.');
  }
  for (const role of roles) {
    const model = manifest.models[role];
    if (
      !model ||
      !/^[\w.-]+\.onnx$/.test(model.file) ||
      !Number.isInteger(model.bytes) ||
      model.bytes < 1 ||
      (model.parts !== undefined &&
        (!Array.isArray(model.parts) ||
          !model.parts.length ||
          !model.parts.every((part) => typeof part === 'string' && /^[\w.-]+$/.test(part))))
    ) {
      throw new Error('The model bundle is incomplete. Ask the site owner to export it again.');
    }
  }
  return manifest;
}

async function readBody(
  response: Response,
  model: ModelSpec,
  chunks: Uint8Array[],
  loaded: number,
  report: Report,
): Promise<number> {
  if (response.body) {
    const reader = response.body.getReader();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (loaded > model.bytes) {
        await reader.cancel();
        throw new Error('A model download has an unexpected size. Please reload and try again.');
      }
      report({
        type: 'progress',
        stage: 'Downloading scanner',
        detail: `${model.file} · ${(loaded / 1e6).toFixed(1)} / ${(model.bytes / 1e6).toFixed(1)} MB`,
        fraction: loaded / model.bytes,
      });
    }
  } else {
    const value = new Uint8Array(await response.arrayBuffer());
    chunks.push(value);
    loaded += value.length;
    if (loaded > model.bytes)
      throw new Error('A model download has an unexpected size. Please reload and try again.');
  }
  return loaded;
}

async function modelBytes(
  base: string,
  model: ModelSpec,
  version: string,
  report: Report,
): Promise<ArrayBuffer> {
  const url = new URL(`models/${model.file}`, base);
  url.searchParams.set('v', version);
  let cache: Cache | undefined;
  try {
    cache = await caches.open('flattener-models-v1');
  } catch {
    /* Private mode may disable persistent cache. */
  }
  const cached = await cache?.match(url);
  if (cached) {
    const data = await cached.arrayBuffer();
    if (data.byteLength === model.bytes) return data;
    await cache?.delete(url);
  }
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (const file of model.parts ?? [model.file]) {
    const part = new URL(`models/${file}`, base);
    part.searchParams.set('v', version);
    const response = await fetch(part);
    if (!response.ok)
      throw new Error(
        'A scanner model could not be downloaded. Check your connection and try again.',
      );
    loaded = await readBody(response, model, chunks, loaded, report);
  }
  const data = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    data.set(chunk, offset);
    offset += chunk.length;
  }
  if (data.byteLength !== model.bytes)
    throw new Error('A model download has an unexpected size. Please reload and try again.');
  try {
    await cache?.put(
      url,
      new Response(data, { headers: { 'Content-Type': 'application/octet-stream' } }),
    );
  } catch {
    /* Scanning still works when storage is full. */
  }
  return data.buffer;
}

export class Models {
  private sessions: Partial<Record<Role, ort.InferenceSession>> = {};
  private bytes: Partial<Record<Role, ArrayBuffer>> = {};
  private activeRoles: Role[] = roles;
  constructor(
    readonly manifest: Manifest,
    private base: string,
    private report: Report,
    public backend: Backend = 'wasm',
  ) {}

  async load(preferred?: Backend, selected: Role[] = roles) {
    this.activeRoles = selected;
    ort.env.wasm.numThreads = 0; // Let the runtime choose up to four threads when supported.
    ort.env.wasm.proxy = false; // This code is already in a dedicated worker.
    ort.env.wasm.wasmPaths = new URL('runtime/', this.base).href;
    const gpu = (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu;
    let useGPU = false;
    if (preferred !== 'wasm' && gpu) {
      try {
        useGPU = !!(await gpu.requestAdapter());
      } catch {
        /* WASM below. */
      }
    }
    this.backend = useGPU ? 'webgpu' : 'wasm';
    for (const role of this.activeRoles)
      this.bytes[role] = await modelBytes(
        this.base,
        this.manifest.models[role],
        this.manifest.version,
        this.report,
      );
    try {
      await this.createSessions();
    } catch (error) {
      if (this.backend === 'wasm') throw error;
      await this.fallback();
    }
  }

  private async createSessions() {
    for (const role of this.activeRoles) {
      this.report({
        type: 'progress',
        stage: 'Preparing scanner',
        detail: `Loading ${role === 'gate' ? 'page detection' : role === 'orient' ? 'text orientation' : 'page correction'}…`,
      });
      this.sessions[role] = await ort.InferenceSession.create(this.bytes[role]!, {
        executionProviders: [this.backend],
        graphOptimizationLevel: 'all',
      });
    }
  }

  async fallback() {
    await this.dispose();
    this.backend = 'wasm';
    this.report({
      type: 'progress',
      stage: 'Switching to CPU',
      detail: 'Your graphics device could not run the scanner. Processing will continue locally.',
    });
    await this.createSessions();
  }

  async warmup() {
    for (const role of this.activeRoles) {
      const inputs = Object.fromEntries(
        Object.entries(this.manifest.models[role].inputs).map(([name, shape]) => [
          name,
          new Float32Array(shape.reduce((size, dimension) => size * dimension, 1)).fill(0.5),
        ]),
      );
      await this.run(role, inputs);
    }
  }

  async run(
    role: Role,
    inputs: Record<string, Float32Array>,
  ): Promise<Record<string, Float32Array>> {
    const feed: Record<string, ort.Tensor> = {};
    for (const [name, data] of Object.entries(inputs))
      feed[name] = new ort.Tensor('float32', data, this.manifest.models[role].inputs[name]);
    let outputs: ort.InferenceSession.ReturnType | undefined;
    try {
      outputs = await this.sessions[role]!.run(feed);
      const result: Record<string, Float32Array> = {};
      for (const [key, value] of Object.entries(outputs))
        result[key] = new Float32Array(value.data as Float32Array);
      return result;
    } finally {
      for (const value of Object.values(feed)) value.dispose();
      if (outputs) for (const value of Object.values(outputs)) value.dispose();
    }
  }

  async dispose() {
    for (const session of Object.values(this.sessions)) {
      try {
        await session.release();
      } catch {
        /* Device may already be lost. */
      }
    }
    this.sessions = {};
  }
}
