import {
  blendFlip,
  boxPlacement,
  checkGrid,
  compose,
  cornerGaps,
  fillCorners,
  fromCHW,
  gateInput,
  gateSplit,
  gridDistortion,
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
import type { Grid, Options, Pixels, Placement, ScanTrace, WorkerReply } from './types';

/** What ``scan`` needs from the models: the manifest and inference. */
export type ScanModels = Pick<Models, 'manifest' | 'run'>;

/** Optional diagnostics: ``trace`` is filled with every decision; ``steps`` replaces the
 * preview images (for runs outside the browser). For experiments, ``box`` fixes the crop box
 * (photo pixels: left, top, right, bottom) and ``turn`` fixes the quarter turns applied
 * before flattening in place of the orientation model's choice. */
export type ScanDebug = {
  trace?: ScanTrace;
  steps?: ReturnType<typeof createSteps>;
  box?: number[];
  turn?: number;
};

const traced = (grid: Grid) => ({
  rows: grid.rows,
  cols: grid.cols,
  data: Array.from(grid.data, (v) => Math.round(v * 1e6) / 1e6),
});

export async function scan(
  image: Pixels,
  models: ScanModels,
  options: Options,
  report: (event: WorkerReply) => void,
  { trace, steps = createSteps(report), box: fixedBox, turn }: ScanDebug = {},
) {
  const warnings: string[] = [];
  const status = (stage: string, detail: string) => report({ type: 'progress', stage, detail });
  const manifest = models.manifest;
  status('Finding the page', 'Locating the document in your photo.');
  const [gh, gw] = manifest.models.gate.input_size;
  const { probability } = await models.run('gate', { image: gateInput(image, gh, gw) });
  if (!probability.every(Number.isFinite))
    throw new Error('Page detection failed. Try a different photo.');
  if (trace) trace.gate = { width: gw, height: gh, probability: Array.from(probability) };
  const [ih, iw] = manifest.models.dewarp.input_size,
    [rows, cols] = manifest.models.dewarp.grid_size;
  const refined = options.quality === 'refined';
  let flipGap: number | undefined, flipGaps: number[] | undefined;
  async function predict(view: Placement): Promise<Grid> {
    const input = modelInput(image, view, ih, iw);
    const out = await models.run('dewarp', { image: input });
    let grid = fromCHW(out.grid, rows, cols);
    flipGap = flipGaps = undefined;
    if (refined && manifest.models.dewarp.tta === 'flip') {
      const mirror = await models.run('dewarp', { image: mirroredInput(input, ih, iw) });
      const info: { gap?: number; gaps?: number[] } = {};
      grid = blendFlip(grid, fromCHW(mirror.grid, rows, cols), info);
      flipGap = info.gap;
      flipGaps = info.gaps?.map((v) => Math.round(v * 1e5) / 1e5);
    }
    return compose(grid, view);
  }
  const record = (name: string, grid: Grid, error: string | null, used: boolean) =>
    trace?.passes.push({ name, grid: traced(grid), flipGap, flipGaps, error, used });
  const place = (name: string, placement: Placement) => trace?.placements.push({ name, placement });

  let box = fixedBox ?? pageBox(probability, gh, gw, image);
  const components = pageComponents(probability, gh, gw);
  const split = !fixedBox && gateSplit(components);
  let fromWholePass = false;
  if (split) {
    // Printed covers and dark bands split the gate's page into pieces; the page outline
    // of a dewarp pass over the whole photo then decides the crop.
    const whole0 = await predict(whole(image));
    const error = checkGrid(whole0);
    const outlineBox = error ? null : splitPageBox(whole0, components, gh, gw, image);
    record('whole photo', whole0, error, !!outlineBox);
    if (outlineBox) {
      box = outlineBox;
      fromWholePass = true;
    }
  }
  if (trace) trace.box = { value: box, pieces: components.length, split, fromWholePass };
  if (!box)
    throw new Error(
      'No clear page was found. Try a well-lit photo with the full document visible.',
    );
  await steps.page(image, probability, gw, gh);
  const boxView = boxPlacement(box, image);
  let placement = boxView,
    rotation = 0;
  place('page box', placement);

  async function orientation(pixels: Pixels, view: Placement) {
    const { global, patches } = orientationViews(pixels, view, manifest.models.orient.config);
    const { probabilities } = await models.run('orient', { global_view: global, patches });
    if (!probabilities.every(Number.isFinite))
      throw new Error(
        'Text orientation could not be read. Try again with automatic orientation turned off.',
      );
    const k = Array.from(probabilities).indexOf(Math.max(...probabilities));
    return { k, accepted: probabilities[k] >= manifest.models.orient.threshold, probabilities };
  }
  if (turn !== undefined) {
    rotation = ((turn % 4) + 4) % 4;
    placement = turnPlacement(placement, rotation);
    place('page box, turned as fixed', placement);
  } else if (options.orient) {
    status('Reading the direction', 'Finding the upright direction of the text.');
    const result = await orientation(image, placement);
    trace?.orientation.push({
      stage: 'before',
      probabilities: Array.from(result.probabilities),
      k: result.k,
      accepted: result.accepted,
    });
    if (result.accepted) {
      rotation = result.k;
      placement = turnPlacement(placement, result.k);
      place('page box, turned upright', placement);
    }
  }

  // Dewarp passes behind the result, a confirming try (below) included.
  let passes = 0;
  // Flattens the page from ``placement``, turned ``quarter`` times: the first pass, a corner
  // fill and in Refined mode two edge refinements. ``restore`` undoes a kept corner fill.
  async function flatten(placement: Placement, quarter: number, label = '') {
    const notes: string[] = [];
    let restore = () => {};
    let grid = await predict(placement);
    passes++;
    const firstError = checkGrid(grid);
    record(`${label}first pass`, grid, firstError, !firstError);
    if (firstError) return { error: firstError } as const;
    // A page corner missing from the photo (a dog-ear, a corner under another sheet or a
    // hand) makes the model bend the whole page toward the cut. When the page comes out bent
    // and the gate's page has a corner gap, the gap is painted with the paper color (in
    // ``image`` itself) and the page predicted again; the painted photo is kept only when the
    // page comes out clearly less bent (by a fifth), and restored otherwise.
    const bend = gridDistortion(grid, image.width, image.height)?.bend ?? 0;
    const gaps = bend >= 0.02 ? cornerGaps(probability, gw, gh, image.width, image.height) : [];
    if (gaps.length) {
      const unpaint = fillCorners(image, gaps, probability, gw, gh);
      const filled = await predict(placement);
      passes++;
      const error = checkGrid(filled);
      const after = error
        ? Infinity
        : (gridDistortion(filled, image.width, image.height)?.bend ?? Infinity);
      const better = after < 0.8 * bend;
      record(`${label}corners filled`, filled, error, better);
      if (trace)
        trace.corners = { gaps, bend, after: Number.isFinite(after) ? after : null, used: better };
      if (better) {
        grid = filled;
        restore = unpaint;
      } else unpaint();
    }
    await steps.mesh(image, grid, passes);
    if (refined)
      for (let i = 0; i < 2; i++) {
        status(
          'Refining the edges',
          `Crop refinement ${i + 1} of 2. Sampling from your original photo.`,
        );
        const angle = pageAngle(grid, image.width, image.height);
        placement = tightPlacement(grid, image, angle ?? quarter * 90);
        place(`${label}refinement ${i + 1}`, placement);
        const next = await predict(placement);
        passes++;
        const error = checkGrid(next);
        // Every valid refinement is kept. Rejecting one that bends the page more (by
        // gridDistortion) threw away tighter, straighter crops on KR-Real: a tighter crop
        // raises the bend, and a bent first refinement was set right by the second.
        record(`${label}refinement ${i + 1}`, next, error, !error);
        if (error) {
          notes.push('An edge refinement was unstable; the previous valid correction was kept.');
          break;
        }
        grid = next;
        await steps.mesh(image, grid, passes);
      }
    return { grid, notes, restore };
  }

  status(
    'Flattening the page',
    refined
      ? 'Estimating the paper shape and checking a mirrored view.'
      : 'Estimating the paper shape.',
  );
  const flat = await flatten(placement, rotation);
  if ('error' in flat) throw new Error(flat.error);
  let { grid, notes } = flat,
    uncertain = false;
  if (options.orient) {
    status('Checking the result', 'Checking text direction on the flattened page.');
    const check = async (grid: Grid, stage: 'after' | 'confirm') => {
      const [ph, pw] = outputSize(grid, image, 1024);
      const preview = render(image, grid, ph, pw);
      const result = await orientation(preview, whole(preview));
      trace?.orientation.push({
        stage,
        probabilities: Array.from(result.probabilities),
        k: result.k,
        accepted: result.accepted,
      });
      return result;
    };
    const result = await check(grid, 'after');
    uncertain = !result.accepted;
    if (result.accepted) {
      grid = quarterTurn(grid, result.k);
      rotation = (rotation + result.k) % 4;
    } else if (result.k !== 0) {
      // An uncertain turn is tried: the page is flattened again turned that way and kept when
      // the check then reads it as upright. On DocUNet photos in four turns this set right 5 of
      // 7 wrong turns; the tries that were wrong (a cut-off KR-Real page) did not read upright.
      status('Checking the result', 'Flattening the page again in the suggested direction.');
      const quarter = (rotation + result.k) % 4;
      const view = turnPlacement(boxView, quarter);
      place('page box, turned to confirm', view);
      const again = await flatten(view, quarter, 'confirm ');
      if (!('error' in again)) {
        const confirmed = await check(again.grid, 'confirm');
        if (confirmed.accepted && confirmed.k === 0) {
          ({ grid, notes } = again);
          rotation = quarter;
          uncertain = false;
        } else again.restore();
      }
    }
  }
  warnings.push(...notes);
  if (uncertain)
    warnings.push('Text direction is uncertain. Check the result and use Rotate if needed.');
  // Bends of 0.05 and more were visible on every KR-Real page that showed them (an occluded
  // corner, a page held in the hand); the scan is kept but the user is told.
  if ((gridDistortion(grid, image.width, image.height)?.bend ?? 0) >= 0.05)
    warnings.push(
      'The page may be bent near an edge or a corner. If text looks warped, take the photo again with the whole page flat and visible.',
    );
  status('Rendering your scan', 'Writing the corrected image at full output resolution.');
  const [height, width] = outputSize(grid, image, options.maxSide);
  if (trace) {
    flipGap = flipGaps = undefined;
    trace.rotation = rotation;
    trace.output = { height, width };
    record('final', grid, null, true);
  }
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
