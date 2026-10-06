import { pageOutline } from './camera-outline';
import { gateInput } from './geometry';
import { Models, readManifest } from './models';
import type { CameraReply, CameraRequest, Pixels } from './types';

let models: Models | undefined;
let busy = false;
const reply = (message: CameraReply) => self.postMessage(message);

async function detect(image: Pixels) {
  const [height, width] = models!.manifest.models.gate.input_size;
  const input = gateInput(image, height, width);
  let output;
  try {
    output = await models!.run('gate', { image: input });
  } catch (error) {
    if (models!.backend !== 'webgpu') throw error;
    await models!.fallback();
    output = await models!.run('gate', { image: input });
  }
  return pageOutline(output.probability, width, height);
}

self.onmessage = async ({ data }: MessageEvent<CameraRequest>) => {
  if (busy) return;
  busy = true;
  try {
    if (data.type === 'load') {
      const candidate = new Models(await readManifest(data.base), data.base, (event) => {
        if (event.type === 'progress') reply({ type: 'progress', message: event.stage });
      });
      try {
        await candidate.load(undefined, ['gate']);
        models = candidate;
      } catch (error) {
        await candidate.dispose();
        throw error;
      }
      reply({ type: 'ready' });
    } else if (models) {
      reply({
        type: 'outline',
        outline: await detect(data.image),
        width: data.width,
        height: data.height,
      });
    }
  } catch {
    reply({ type: 'error', message: 'Live outline unavailable. You can still take a photo.' });
  } finally {
    busy = false;
  }
};
