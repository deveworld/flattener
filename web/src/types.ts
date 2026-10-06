export type Pixels = { width: number; height: number; data: Uint8ClampedArray };
export type Mask = { width: number; height: number; data: Uint8Array };
export type PaperField = { width: number; height: number; data: Float32Array };
export type EditSettings = {
  cleanColor: boolean;
  mask: Mask;
  rotation: number;
  reviewedMask?: boolean;
  /** Fractions of the unrotated scan to keep; absent keeps the whole page. */
  crop?: { left: number; top: number; right: number; bottom: number };
};
export type ProcessingStep = { id: string; title: string; detail: string; blob: Blob };
export type Grid = { rows: number; cols: number; data: Float64Array };
export type Placement = {
  matrix: [number, number, number, number];
  offset: [number, number];
  width: number;
  height: number;
};
export type Options = {
  quality: 'fast' | 'refined';
  orient: boolean;
  maxSide: number;
  /** Unsharp masking in Original color, scaled by how much the page was reduced. */
  sharpen: boolean;
  /** The color of the returned scan: Clean color, or the original color. */
  cleanColor: boolean;
};
export type Backend = 'webgpu' | 'wasm';
export type Point = [number, number];
export type PageOutline = { points: Point[]; area: number; clipped: boolean };
/** How a camera photo was taken: a still photo or a preview frame, its size and how long
 * the shutter took (seconds). */
export type CaptureInfo = {
  method: 'photo' | 'frame';
  width: number;
  height: number;
  /** The live preview's video size, to compare its framing with the photo's. */
  preview: { width: number; height: number };
  seconds: number;
};
export type CameraRequest =
  { type: 'load'; base: string } | { type: 'frame'; image: Pixels; width: number; height: number };
export type CameraReply =
  | { type: 'ready' }
  | { type: 'progress'; message: string }
  | { type: 'outline'; outline: PageOutline | null; width: number; height: number }
  | { type: 'error'; message: string };
export type ModelSpec = {
  file: string;
  bytes: number;
  /** Files whose concatenation is ``file``, for hosts with a per-file size limit. */
  parts?: string[];
  inputs: Record<string, number[]>;
  outputs: string[];
};
export type OrientConfig = {
  global_size: number;
  patch_size: number;
  patches: number;
  page_long_side: number;
  energy_grid: number;
  pad_value: number;
};
export type Manifest = {
  schema: 'flattener-web-bundle-1';
  version: string;
  precision: 'float32';
  models: {
    dewarp: ModelSpec & {
      input_size: [number, number];
      grid_size: [number, number];
      grid_convention: string;
      tta: string;
    };
    gate: ModelSpec & { input_size: [number, number] };
    orient: ModelSpec & { config: OrientConfig; threshold: number };
  };
};
export type ScanMetadata = {
  backend: Backend;
  seconds: number;
  width: number;
  height: number;
  warnings: string[];
  passes: number;
  rotation: number;
  source: { width: number; height: number };
  model: string;
  /** Sharpening strength for Original color; 0 when off. */
  sharpen: number;
};
export type PreparationState = 'idle' | 'loading' | 'ready' | 'error';
export type WorkerRequest =
  | { type: 'prepare'; base: string; backend?: Backend }
  | {
      type: 'scan';
      file: File;
      options: Options;
      base: string;
      backend?: Backend;
    };
export type WorkerReply =
  | { type: 'preparation'; state: PreparationState }
  | { type: 'progress'; stage: string; detail: string; fraction?: number }
  | { type: 'reset-steps' }
  | ({ type: 'step' } & ProcessingStep)
  | { type: 'backend'; backend: Backend }
  | { type: 'result'; blob: Blob; raw: Blob; suggestion: Mask; metadata: ScanMetadata }
  | { type: 'error'; message: string };
export type EditRequest = {
  id: number;
  raw: Blob;
  settings: EditSettings;
  suggestion: Mask;
  sharpen: number;
};
export type EditReply =
  | { type: 'edited'; id: number; blob: Blob; width: number; height: number }
  | { type: 'error'; id: number; message: string };
