import {
  blendFlip,
  boxPlacement,
  checkGrid,
  compose,
  fromCHW,
  gateInput,
  gateSplit,
  mirroredInput,
  modelInput,
  outlinePolygon,
  outputSize,
  pageAngle,
  pageBox,
  pageComponents,
  quarterTurn,
  render,
  splitPageBox,
  tightPlacement,
  turnPlacement,
  whole,
} from './geometry';
import { orientationViews } from './orientation';
import { createSteps } from './previews';
import { sharpenAmount } from './sharpen';
import type { Models } from './models';
import type { Grid, Options, Pixels, Placement, WorkerReply } from './types';

export async function scan(
  image: Pixels,
  models: Models,
  options: Options,
  report: (event: WorkerReply) => void,
) {
  const warnings: string[] = [];
  const steps = createSteps(report);
  const status = (stage: string, detail: string) => report({ type: 'progress', stage, detail });
  const manifest = models.manifest;
  status('Finding the page', 'Locating the document in your photo.');
  const [gh, gw] = manifest.models.gate.input_size;
  const { probability } = await models.run('gate', { image: gateInput(image, gh, gw) });
  if (!probability.every(Number.isFinite))
    throw new Error('Page detection failed. Try a different photo.');
  const [ih, iw] = manifest.models.dewarp.input_size,
    [rows, cols] = manifest.models.dewarp.grid_size;
  const refined = options.quality === 'refined';
  async function predict(view: Placement): Promise<Grid> {
    const input = modelInput(image, view, ih, iw);
    const out = await models.run('dewarp', { image: input });
    let grid = fromCHW(out.grid, rows, cols);
    if (refined && manifest.models.dewarp.tta === 'flip') {
      const mirror = await models.run('dewarp', { image: mirroredInput(input, ih, iw) });
      grid = blendFlip(grid, fromCHW(mirror.grid, rows, cols));
    }
    return compose(grid, view);
  }

  let box = pageBox(probability, gh, gw, image);
  const components = pageComponents(probability, gh, gw);
  if (gateSplit(components)) {
    // Printed covers and dark bands split the gate's page into pieces; the page outline
    // of a dewarp pass over the whole photo then decides the crop.
    const whole0 = await predict(whole(image));
    if (!checkGrid(whole0)) box = splitPageBox(whole0, components, gh, gw, image) ?? box;
  }
  if (!box)
    throw new Error(
      'No clear page was found. Try a well-lit photo with the full document visible.',
    );
  await steps.page(image, probability, gw, gh);
  let placement = boxPlacement(box, image),
    rotation = 0;

  async function orientation(pixels: Pixels, view: Placement) {
    const { global, patches } = orientationViews(pixels, view, manifest.models.orient.config);
    const { probabilities } = await models.run('orient', { global_view: global, patches });
    if (!probabilities.every(Number.isFinite))
      throw new Error(
        'Text orientation could not be read. Try again with automatic orientation turned off.',
      );
    const k = Array.from(probabilities).indexOf(Math.max(...probabilities));
    return { k, accepted: probabilities[k] >= manifest.models.orient.threshold };
  }
  if (options.orient) {
    status('Reading the direction', 'Finding the upright direction of the text.');
    const result = await orientation(image, placement);
    if (result.accepted) {
      rotation = result.k;
      placement = turnPlacement(placement, result.k);
    }
  }
  status(
    'Flattening the page',
    refined
      ? 'Estimating the paper shape and checking a mirrored view.'
      : 'Estimating the paper shape.',
  );
  let grid = await predict(placement),
    passes = 1;
  const firstError = checkGrid(grid);
  if (firstError) throw new Error(firstError);
  await steps.mesh(image, grid, passes);
  if (refined)
    for (let i = 0; i < 2; i++) {
      status(
        'Refining the edges',
        `Crop refinement ${i + 1} of 2. Sampling from your original photo.`,
      );
      const angle = pageAngle(grid, image.width, image.height);
      placement = tightPlacement(grid, image, angle ?? rotation * 90);
      const next = await predict(placement);
      passes++;
      if (checkGrid(next)) {
        warnings.push('An edge refinement was unstable; the previous valid correction was kept.');
        break;
      }
      grid = next;
      await steps.mesh(image, grid, passes);
    }
  if (options.orient) {
    status('Checking the result', 'Checking text direction on the flattened page.');
    const [ph, pw] = outputSize(grid, image, 1024);
    const preview = render(image, grid, ph, pw);
    const result = await orientation(preview, whole(preview));
    if (result.accepted) {
      grid = quarterTurn(grid, result.k);
      rotation = (rotation + result.k) % 4;
    } else warnings.push('Text direction is uncertain. Check the result and use Rotate if needed.');
  }
  status('Rendering your scan', 'Writing the corrected image at full output resolution.');
  const [height, width] = outputSize(grid, image, options.maxSide);
  const result = render(image, grid, height, width);
  // Sharpening for Original color, applied by the editor: Clean color already darkens
  // text, and sharpening on top of it lowered OCR accuracy on the KR-Real pages.
  let sharpen = 0;
  if (options.sharpen) {
    // Source pixels per output pixel over the page, by area: reduction sharpens more.
    const polygon = outlinePolygon(grid, image.width, image.height);
    let area = 0;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++)
      area += (polygon[j][0] + polygon[i][0]) * (polygon[j][1] - polygon[i][1]);
    sharpen = sharpenAmount(Math.sqrt(Math.abs(area / 2) / (width * height)));
  }
  await steps.image(
    result,
    'flat',
    'Page flattened',
    'The original photo sampled through the final page map, before color or finger edits.',
  );
  return { result, warnings, rotation, passes, grid, probability, gw, gh, sharpen };
}
