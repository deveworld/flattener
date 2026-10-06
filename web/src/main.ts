import './style.css';
import { Camera } from './camera';
import { emptyMask, maskCount } from './cleanup';
import { isFullCrop } from './crop';
import { CropEditor } from './crop-editor';
import { MaskEditor } from './mask-editor';
import { pdfFromJpegs, type PdfPage } from './pdf';
import type {
  EditReply,
  EditRequest,
  EditSettings,
  Mask,
  Options,
  ProcessingStep,
  PreparationState,
  ScanMetadata,
  ScanTrace,
  WorkerReply,
  WorkerRequest,
} from './types';
import { localize } from './i18n';

localize();

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const fileInput = $<HTMLInputElement>('file-input'),
  scanButton = $<HTMLButtonElement>('scan');
const originalImage = $<HTMLImageElement>('original-image'),
  scanImage = $<HTMLImageElement>('scan-image');
let worker: Worker | undefined, selected: File | undefined;
let originalURL = '',
  resultURL = '',
  resultBlob: Blob | undefined,
  metadata: ScanMetadata | undefined;
let busy = false,
  editing = false,
  photoReady = false,
  retryable = false;
let focusAction = false;
let refining = false;
let activeOptions: Options | undefined, resultOptions: Options | undefined;
/** The color mode last chosen; new scans start in it. */
let colorChoice = true;
/** The result shows its edit tools (color, fingers, crop, rotate, 100% view). */
let editMode = false;
/** ``?debug``: each scan records its decisions for a diagnostics download. */
const debug = new URLSearchParams(location.search).has('debug');
let lastTrace: ScanTrace | undefined;
let preparationState: PreparationState = 'idle';
let rawBlob: Blob | undefined, suggestion: Mask | undefined, settings: EditSettings | undefined;
let editWorker: Worker | undefined,
  editId = 0,
  editorOpen = false;
let pendingEdit: { settings: EditSettings; history: 'push' | 'pop' | 'reset' } | undefined;
const editHistory: EditSettings[] = [];
const steps = new Map<string, ProcessingStep & { url: string; button: HTMLButtonElement }>();
const maskEditor = new MaskEditor();
const cropEditor = new CropEditor();
const camera = new Camera(
  (file) => chooseFile(file),
  (dismissed) => {
    if (dismissed && !selected && !followingHistory) leaveAway();
  },
);
if (debug) camera.diagnostics = { frames: [], captures: [] };

// History: the camera and an open photo share one entry above the start screen, so the
// browser's back gesture returns to the start screen instead of leaving the app.
let followingHistory = false;
if (history.state?.flattener === 'away') history.replaceState(null, '');
function enterAway() {
  if (history.state?.flattener !== 'away') history.pushState({ flattener: 'away' }, '');
}
function leaveAway() {
  if (history.state?.flattener === 'away') history.back();
}
window.addEventListener('popstate', () => {
  if (history.state?.flattener === 'away') return;
  followingHistory = true;
  try {
    if (camera.isOpen) camera.close();
    if (selected) showStart();
  } finally {
    followingHistory = false;
  }
});
const cloneSettings = (s: EditSettings): EditSettings => ({
  ...s,
  mask: { ...s.mask, data: s.mask.data.slice() },
});

function preparationStatus(state: PreparationState) {
  preparationState = state;
  const labels = {
    idle: 'Scanner will prepare when needed.',
    loading: 'Preparing scanner…',
    ready: 'Ready to flatten',
    error: 'Scanner could not load.',
  };
  for (const element of document.querySelectorAll<HTMLElement>('.scanner-preparation')) {
    element.dataset.state = state;
    element.querySelector('[data-scanner-status]')!.textContent = labels[state];
    element.querySelector<HTMLButtonElement>('[data-prepare-retry]')!.hidden = state !== 'error';
  }
}

function prepareScanner() {
  if (preparationState === 'loading' || preparationState === 'ready' || busy) return;
  preparationStatus('loading');
  try {
    scannerWorker().postMessage({
      type: 'prepare',
      base: new URL('.', document.baseURI).href,
    } satisfies WorkerRequest);
  } catch {
    worker?.terminate();
    worker = undefined;
    preparationStatus('error');
  }
}

for (const button of document.querySelectorAll<HTMLButtonElement>('[data-prepare-retry]'))
  button.onclick = prepareScanner;
window.setTimeout(prepareScanner, 250);

function showError(message: string) {
  $('error').textContent = message;
  $('error').hidden = false;
}
function message(title: string, detail = '') {
  $('stage-title').textContent = title;
  $('stage-detail').textContent = detail;
  $('stage-detail').hidden = !detail;
}
function busyState(value: boolean) {
  busy = value;
  const locked = value || editing || editorOpen;
  scanButton.disabled = locked || !selected || !photoReady;
  // Photos flatten as soon as they open; the button only offers a retry.
  scanButton.hidden = value || !!resultBlob || !retryable;
  $('cancel').hidden = !value && !editing;
  $('cancel').textContent = editing
    ? 'Cancel edit'
    : refining
      ? 'Cancel refinement'
      : 'Cancel scan';
  for (const id of ['refine', 'crop', 'rotate', 'zoom', 'remove-fingers', 'add-page', 'edit-open'])
    $<HTMLButtonElement>(id).disabled = locked || (id === 'add-page' && pageAdded);
  $<HTMLButtonElement>('edit-done').disabled = locked;
  $<HTMLSelectElement>('color-mode').disabled = locked;
  $<HTMLButtonElement>('undo-edit').disabled = locked || !editHistory.length;
  $<HTMLButtonElement>('reset-edits').disabled = locked || !edited();
  $('undo-edit').hidden = !editHistory.length;
  $('reset-edits').hidden = !edited();
  $('progress').hidden = !value && !editing;
  $('work-controls').hidden = !selected || editorOpen;
  $('result-actions').hidden = !resultBlob || editing || editorOpen || editMode;
  $('refine-controls').hidden =
    !resultBlob || resultOptions?.quality !== 'fast' || locked || editMode;
  const tools = !resultBlob || busy || editorOpen || !editMode;
  $('edit-controls').hidden = tools;
  $('adjustments').hidden = tools;
  $('edit-done').hidden = tools;
  $('view-controls').hidden = !resultBlob || locked || !$('process-view').hidden;
  for (const step of steps.values()) step.button.disabled = locked;
  $('workspace').setAttribute('aria-busy', String(value || editing));
}
function edited() {
  return (
    !!settings &&
    (settings.cleanColor !== (resultOptions?.cleanColor ?? true) ||
      !!settings.rotation ||
      !!maskCount(settings.mask) ||
      !isFullCrop(settings.crop))
  );
}
function clearSteps() {
  for (const step of steps.values()) URL.revokeObjectURL(step.url);
  steps.clear();
  $('process-steps').replaceChildren();
  $('process-history').hidden = true;
  $<HTMLDetailsElement>('process-history').open = false;
  $('process-view').hidden = true;
  $<HTMLImageElement>('process-image').removeAttribute('src');
  $('process-image').hidden = true;
}
function displayStep(id: string) {
  const step = steps.get(id);
  if (!step) return;
  $('images').hidden = true;
  $('process-view').hidden = false;
  $('view-controls').hidden = true;
  const img = $<HTMLImageElement>('process-image');
  img.src = step.url;
  img.alt = step.title;
  img.hidden = false;
  $('process-title').textContent = step.title;
  $('process-detail').textContent = step.detail;
  $('process-back').hidden = busy;
  $('process-back').textContent = resultBlob ? 'Back to scan' : 'Back to photo';
  for (const [key, entry] of steps) entry.button.setAttribute('aria-pressed', String(key === id));
}
function addStep(step: ProcessingStep) {
  const previous = steps.get(step.id);
  if (previous) URL.revokeObjectURL(previous.url);
  const url = URL.createObjectURL(step.blob),
    button = previous?.button ?? document.createElement('button');
  button.type = 'button';
  button.disabled = busy || editing || editorOpen;
  const image = document.createElement('img');
  image.src = url;
  image.alt = '';
  image.loading = 'lazy';
  const label = document.createElement('span');
  label.textContent = step.title;
  button.replaceChildren(image, label);
  button.onclick = () => {
    displayStep(step.id);
    scrollTo($('drop-area'));
  };
  if (!previous) {
    const item = document.createElement('li');
    item.append(button);
    $('process-steps').append(item);
  }
  steps.set(step.id, { ...step, url, button });
  $('process-history').hidden = false;
  if (!refining) displayStep(step.id);
}
function clearEdits() {
  editWorker?.terminate();
  editWorker = undefined;
  ++editId;
  pendingEdit = undefined;
  editing = false;
  editorOpen = false;
  maskEditor.close();
  cropEditor.close();
  $('finger-editor').hidden = true;
  $('crop-editor').hidden = true;
  rawBlob = undefined;
  suggestion = undefined;
  settings = undefined;
  editHistory.length = 0;
}
function clearResult() {
  clearEdits();
  resultOptions = undefined;
  refining = false;
  clearSteps();
  if (resultURL) URL.revokeObjectURL(resultURL);
  resultURL = '';
  resultBlob = undefined;
  metadata = undefined;
  scanImage.hidden = true;
  scanImage.removeAttribute('src');
  ++downloadVersion;
  if (downloadURL) URL.revokeObjectURL(downloadURL);
  downloadURL = '';
  $('result-actions').hidden = true;
  $('refine-controls').hidden = true;
  $('edit-controls').hidden = true;
  $('edit-done').hidden = true;
  $('diagnostics').hidden = true;
  editMode = false;
  $('view-controls').hidden = true;
  $('warnings').hidden = true;
  $('adjustments').hidden = true;
  $<HTMLButtonElement>('show-scan').disabled = true;
  $<HTMLButtonElement>('show-compare').disabled = true;
  setZoom(false);
}
function scrollTo(element: HTMLElement, block: ScrollLogicalPosition = 'start') {
  element.scrollIntoView({
    behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
    block,
  });
}
/** The start screen; collected PDF pages stay. */
function showStart() {
  if (editorOpen) closeEditor();
  if (editing) {
    editWorker?.terminate();
    editWorker = undefined;
    ++editId;
    pendingEdit = undefined;
    editing = false;
  }
  if (busy) {
    worker?.terminate();
    worker = undefined;
    preparationStatus('idle');
  }
  clearResult();
  selected = undefined;
  document.body.classList.remove('has-photo');
  photoReady = false;
  retryable = false;
  originalImage.hidden = true;
  originalImage.removeAttribute('src');
  URL.revokeObjectURL(originalURL);
  originalURL = '';
  $('empty').hidden = false;
  $('images').hidden = true;
  $('file-toolbar').hidden = true;
  $('work-controls').hidden = true;
  $('view-controls').hidden = true;
  $('error').hidden = true;
  $('filename').textContent = '';
  message('');
  busyState(false);
  prepareScanner();
}
function rejectPhoto(detail: string) {
  showStart();
  showError(detail);
}
function scanFailed(detail: string) {
  worker?.terminate();
  worker = undefined;
  preparationStatus('idle');
  const keepResult = refining && !!resultBlob;
  refining = false;
  busyState(false);
  retryable = !keepResult;
  busyState(false);
  view(keepResult ? 'scan' : 'original');
  showError(detail);
  message(
    keepResult ? 'Refinement could not be completed.' : 'The scan could not be completed.',
    keepResult
      ? 'Your previous scan is still available. You can retry refinement.'
      : 'Adjust the settings or try a different photo.',
  );
  scanButton.textContent = 'Try again';
  if (focusAction) (keepResult ? $('refine') : scanButton).focus({ preventScroll: true });
}
function view(mode: 'original' | 'scan' | 'compare') {
  $('images').hidden = false;
  $('process-view').hidden = true;
  $('view-controls').hidden = !resultBlob || busy || editing || editorOpen;
  for (const step of steps.values()) step.button.setAttribute('aria-pressed', 'false');
  $('original-pane').hidden = mode === 'scan';
  $('scan-pane').hidden = mode === 'original';
  $('images').classList.toggle('compare', mode === 'compare');
  for (const name of ['original', 'scan', 'compare'])
    $(`show-${name}`).setAttribute('aria-pressed', String(name === mode));
}
function chooseFile(file: File) {
  if (busy || editing || editorOpen || camera.isOpen) return;
  if (file.size > 40 * 1024 * 1024) {
    showError('This photo is larger than 40 MB. Choose a smaller image.');
    return;
  }
  if (file.type && !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    showError('Choose a JPEG, PNG, or WebP photo. HEIC and PDF files are not supported yet.');
    return;
  }
  prepareScanner();
  enterAway();
  selected = file;
  document.body.classList.add('has-photo');
  photoReady = false;
  retryable = false;
  $('error').hidden = true;
  clearResult();
  if (originalURL) URL.revokeObjectURL(originalURL);
  originalImage.hidden = true;
  originalURL = URL.createObjectURL(file);
  originalImage.src = originalURL;
  $('empty').hidden = true;
  $('images').hidden = false;
  $('file-toolbar').hidden = false;
  $('work-controls').hidden = false;
  $('filename').textContent = file.name;
  $('filename').title = file.name;
  view('original');
  busyState(false);
  message('Opening your photo…');
  scanButton.textContent = 'Flatten photo';
}
originalImage.onload = () => {
  const width = originalImage.naturalWidth,
    height = originalImage.naturalHeight;
  if (width * height > 20_000_000) {
    rejectPhoto('This photo exceeds 20 megapixels. Resize it before scanning.');
    return;
  }
  if (Math.min(width, height) < 64) {
    rejectPhoto('This photo is too small. Choose an image at least 64 pixels on each side.');
    return;
  }
  photoReady = true;
  originalImage.hidden = false;
  startScan();
};
originalImage.onerror = () =>
  rejectPhoto('This image could not be opened. Save it as JPEG or PNG and try again.');
$('drop-choose').onclick = () => fileInput.click();
$('open-camera').onclick = () => {
  if (busy || editing || editorOpen) return;
  prepareScanner();
  enterAway();
  camera.open($('open-camera'));
};
$('back').onclick = () => {
  if (history.state?.flattener === 'away') history.back();
  else showStart();
  $('open-camera').focus({ preventScroll: true });
  scrollTo($('workspace'));
};
fileInput.onchange = () => {
  if (fileInput.files?.[0]) chooseFile(fileInput.files[0]);
  fileInput.value = '';
};

let dragDepth = 0;
const stage = $('drop-area');
stage.addEventListener('dragenter', (event) => {
  event.preventDefault();
  if (!busy && !editing && !editorOpen) {
    dragDepth++;
    stage.classList.add('dragover');
  }
});
stage.addEventListener('dragover', (event) => event.preventDefault());
stage.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    stage.classList.remove('dragover');
  }
});
stage.addEventListener('drop', (event) => {
  event.preventDefault();
  dragDepth = 0;
  stage.classList.remove('dragover');
  if (busy || editing || editorOpen) return;
  const files = event.dataTransfer?.files;
  if (files?.length === 1) chooseFile(files[0]);
  else if (files?.length) showError('Choose one photo at a time.');
});
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => e.preventDefault());
document.addEventListener('paste', (e) => {
  const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith('image/'));
  if (file && !busy && !editing && !editorOpen) {
    e.preventDefault();
    chooseFile(file);
  }
});

function installResult(blob: Blob) {
  if (resultURL) URL.revokeObjectURL(resultURL);
  resultBlob = blob;
  resultURL = URL.createObjectURL(blob);
  scanImage.src = resultURL;
  scanImage.hidden = false;
  pageAdded = false;
  $('add-page-label').textContent = 'Add to PDF';
  void prepareDownload();
}

// Saving: the link is prepared ahead for the chosen format, so a tap saves without waiting.
type SaveFormat = 'png' | 'jpg' | 'pdf';
const formatSelect = $<HTMLSelectElement>('save-format');
let downloadURL = '',
  downloadVersion = 0;
try {
  const stored = localStorage.getItem('flattener-format');
  if (stored === 'png' || stored === 'jpg' || stored === 'pdf') formatSelect.value = stored;
} catch {
  /* Storage can be unavailable; PNG stays the default. */
}
function saveFormat(): SaveFormat {
  return formatSelect.value as SaveFormat;
}
async function jpegPage(blob: Blob): Promise<PdfPage & { blob: Blob }> {
  const bitmap = await createImageBitmap(blob);
  try {
    const canvas = document.createElement('canvas');
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bitmap, 0, 0);
    const jpeg = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('encode'))), 'image/jpeg', 0.92),
    );
    return {
      blob: jpeg,
      jpeg: new Uint8Array(await jpeg.arrayBuffer()),
      width: bitmap.width,
      height: bitmap.height,
    };
  } finally {
    bitmap.close();
  }
}
async function prepareDownload() {
  const version = ++downloadVersion;
  const link = $<HTMLAnchorElement>('download');
  const format = saveFormat();
  if (!resultBlob) return;
  link.removeAttribute('href');
  link.setAttribute('aria-disabled', 'true');
  const source = resultBlob;
  let blob: Blob;
  try {
    blob =
      format === 'png'
        ? source
        : format === 'jpg'
          ? (await jpegPage(source)).blob
          : pdfFromJpegs([await jpegPage(source)]);
  } catch {
    if (version === downloadVersion) showError('The file could not be prepared. Try again.');
    return;
  }
  if (version !== downloadVersion) return;
  if (downloadURL) URL.revokeObjectURL(downloadURL);
  downloadURL = URL.createObjectURL(blob);
  link.href = downloadURL;
  link.download = `${(selected?.name ?? 'document').replace(/\.[^.]*$/, '')}-scan.${format}`;
  link.removeAttribute('aria-disabled');
}
formatSelect.onchange = () => {
  try {
    localStorage.setItem('flattener-format', saveFormat());
  } catch {
    /* The choice still applies to this visit. */
  }
  void prepareDownload();
};
$('download').addEventListener('click', (event) => {
  if ($('download').getAttribute('aria-disabled') === 'true') event.preventDefault();
});

// Pages collected for one PDF; they stay while new photos are scanned.
type CollectedPage = PdfPage & { url: string };
const pages: CollectedPage[] = [];
let pageAdded = false,
  pagesURL = '';
function renderPages() {
  const list = $('pages-list');
  list.replaceChildren(
    ...pages.map((page, index) => {
      const item = document.createElement('li');
      const image = document.createElement('img');
      image.src = page.url;
      image.alt = '';
      const number = document.createElement('span');
      number.className = 'page-number';
      number.textContent = String(index + 1);
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'page-remove';
      remove.setAttribute('aria-label', `Remove page ${index + 1}`);
      remove.innerHTML =
        '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7 7 17" /></svg>';
      remove.onclick = () => {
        URL.revokeObjectURL(page.url);
        pages.splice(index, 1);
        renderPages();
      };
      item.append(image, number, remove);
      return item;
    }),
  );
  $('pages').hidden = !pages.length;
  $('pages-count').textContent = pages.length === 1 ? '1 page' : `${pages.length} pages`;
  if (pagesURL) URL.revokeObjectURL(pagesURL);
  pagesURL = pages.length ? URL.createObjectURL(pdfFromJpegs(pages)) : '';
  const save = $<HTMLAnchorElement>('pages-save');
  save.hidden = !pages.length;
  if (pagesURL) save.href = pagesURL;
  else save.removeAttribute('href');
  save.download = `scans-${pages.length}p.pdf`;
}
$('add-page').onclick = async () => {
  if (!resultBlob || pageAdded || busy || editing || editorOpen) return;
  const button = $<HTMLButtonElement>('add-page');
  button.disabled = true;
  try {
    const page = await jpegPage(resultBlob);
    pages.push({ ...page, url: URL.createObjectURL(page.blob) });
    pageAdded = true;
    renderPages();
    $('add-page-label').textContent = 'Added to PDF';
    message(`Added as page ${pages.length}.`, 'Go back to scan the next page.');
  } catch {
    showError('The page could not be added. Try again.');
  } finally {
    busyState(busy);
  }
};
$('pages-clear').onclick = () => {
  for (const page of pages) URL.revokeObjectURL(page.url);
  pages.length = 0;
  pageAdded = false;
  $('add-page-label').textContent = 'Add to PDF';
  renderPages();
  busyState(busy);
};

function scannerWorker(): Worker {
  if (!worker) {
    worker = new Worker(new URL('./scan.worker.ts', import.meta.url), { type: 'module' });
    const active = worker;
    worker.onmessage = (event: MessageEvent<WorkerReply>) => {
      if (worker !== active) return;
      const data = event.data;
      if (data.type === 'preparation') {
        preparationStatus(data.state);
      } else if (data.type === 'progress') {
        message(data.stage, data.detail);
        const progress = $<HTMLProgressElement>('progress');
        if (data.fraction === undefined) progress.removeAttribute('value');
        else {
          progress.max = 1;
          progress.value = data.fraction;
        }
      } else if (data.type === 'step') {
        addStep(data);
      } else if (data.type === 'reset-steps') {
        clearSteps();
        view(refining ? 'scan' : 'original');
      } else if (data.type === 'error') {
        scanFailed(data.message);
      } else if (data.type === 'result') {
        clearEdits();
        resultOptions = activeOptions;
        refining = false;
        metadata = data.metadata;
        rawBlob = data.raw;
        lastTrace = data.trace;
        $('diagnostics').hidden = !lastTrace;
        suggestion = data.suggestion;
        settings = {
          cleanColor: resultOptions?.cleanColor ?? true,
          mask: emptyMask(metadata.width, metadata.height),
          rotation: 0,
        };
        $<HTMLSelectElement>('color-mode').value = settings.cleanColor ? 'clean' : 'original';
        installResult(data.blob);
        busyState(false);
        $<HTMLButtonElement>('show-scan').disabled = false;
        $<HTMLButtonElement>('show-compare').disabled = false;
        $('warnings').textContent = metadata.warnings.join(' ');
        $('warnings').hidden = metadata.warnings.length === 0;
        $('view-controls').hidden = false;
        scanButton.textContent = 'Scan again';
        view('scan');
        message(
          resultOptions?.quality === 'refined'
            ? 'Your refined scan is ready.'
            : 'Your scan is ready.',
          maskCount(suggestion) ? 'Possible fingers found. Review them under Edit.' : '',
        );
        if (focusAction) $('download').focus({ preventScroll: true });
        if (matchMedia('(max-width: 700px)').matches) scrollTo($('work-controls'), 'end');
      }
    };
    worker.onerror = () => {
      if (worker !== active) return;
      if (!busy) {
        active.terminate();
        worker = undefined;
        preparationStatus('error');
        return;
      }
      scanFailed(
        'The scanner stopped unexpectedly. Try again with a smaller image, or reload this page.',
      );
    };
  }
  return worker;
}

function startScan(quality: Options['quality'] = 'fast') {
  if (!selected || !photoReady || busy || editing || editorOpen) return;
  if (quality === 'refined' && (!resultBlob || !resultOptions)) return;
  if (!window.isSecureContext) {
    showError('Open this app over HTTPS or localhost so the scanner can load securely.');
    return;
  }
  if (!window.Worker || !window.OffscreenCanvas || !window.createImageBitmap) {
    showError(
      'This browser does not support the scanner. Try a current version of Chrome, Edge, Firefox, or Safari.',
    );
    return;
  }
  focusAction = [scanButton, $('refine')].some((button) => button.matches(':focus-visible'));
  $('error').hidden = true;
  const options: Options =
    quality === 'refined'
      ? { ...resultOptions!, quality, cleanColor: colorChoice }
      : {
          quality,
          orient: true,
          maxSide: 2560,
          sharpen: true,
          cleanColor: colorChoice,
        };
  if (quality === 'refined') {
    refining = true;
    clearSteps();
    view('scan');
  } else {
    clearResult();
    view('original');
  }
  activeOptions = options;
  busyState(true);
  if (focusAction) $('cancel').focus({ preventScroll: true });
  message(refining ? 'Refining your scan…' : 'Starting your scan…');
  $<HTMLProgressElement>('progress').removeAttribute('value');
  const request: WorkerRequest = {
    type: 'scan',
    file: selected,
    options,
    base: new URL('.', document.baseURI).href,
    trace: debug || undefined,
  };
  try {
    scannerWorker().postMessage(request);
  } catch {
    scanFailed(
      'The photo could not be opened by the scanner. Choose it again or reload this page.',
    );
  }
}
scanButton.onclick = () => startScan();
$('refine').onclick = () => startScan('refined');
$('cancel').onclick = () => {
  const restoreFocus = $('cancel').matches(':focus-visible');
  if (editing) {
    editWorker?.terminate();
    editWorker = undefined;
    ++editId;
    pendingEdit = undefined;
    editing = false;
    busyState(false);
    view('scan');
    if (settings)
      $<HTMLSelectElement>('color-mode').value = settings.cleanColor ? 'clean' : 'original';
    message('Edit cancelled.', 'Your previous scan is unchanged.');
    if (restoreFocus) $('download').focus({ preventScroll: true });
    return;
  }
  worker?.terminate();
  worker = undefined;
  preparationStatus('idle');
  const keepResult = refining && !!resultBlob;
  refining = false;
  retryable = !keepResult;
  busyState(false);
  view(keepResult ? 'scan' : 'original');
  message(
    keepResult ? 'Refinement cancelled.' : 'Scan cancelled.',
    keepResult ? 'Your previous scan is still available.' : 'Your photo is ready to try again.',
  );
  if (restoreFocus) (keepResult ? $('refine') : scanButton).focus({ preventScroll: true });
};
for (const mode of ['original', 'scan', 'compare'] as const)
  $(`show-${mode}`).onclick = () => view(mode);
$('process-back').onclick = () => view(resultBlob ? 'scan' : 'original');
$('diagnostics').onclick = () => {
  const link = $<HTMLAnchorElement>('diagnostics');
  URL.revokeObjectURL(link.href);
  const report = {
    userAgent: navigator.userAgent,
    file: selected && { name: selected.name, size: selected.size, type: selected.type },
    options: resultOptions,
    metadata,
    trace: lastTrace,
    camera: camera.diagnostics,
  };
  link.href = URL.createObjectURL(new Blob([JSON.stringify(report)], { type: 'application/json' }));
};
function setZoom(active: boolean) {
  $('images').classList.toggle('zoomed', active);
  $('zoom').setAttribute('aria-pressed', String(active));
  $('zoom').textContent = active ? 'Fit to view' : '100% view';
}
$('zoom').onclick = () => setZoom(!$('images').classList.contains('zoomed'));
$('edit-open').onclick = () => {
  const keyboard = $('edit-open').matches(':focus-visible');
  editMode = true;
  busyState(busy);
  if (keyboard) $('color-mode').focus({ preventScroll: true });
  if (matchMedia('(max-width: 700px)').matches) scrollTo($('adjustments'));
};
$('edit-done').onclick = () => {
  const keyboard = $('edit-done').matches(':focus-visible');
  editMode = false;
  setZoom(false);
  busyState(busy);
  if (keyboard) $('edit-open').focus({ preventScroll: true });
};
function editFailed(detail: string) {
  editWorker?.terminate();
  editWorker = undefined;
  pendingEdit = undefined;
  editing = false;
  busyState(false);
  view('scan');
  if (settings)
    $<HTMLSelectElement>('color-mode').value = settings.cleanColor ? 'clean' : 'original';
  showError(detail);
  message('The edit could not be applied.', 'Your previous scan is still available.');
  if (focusAction) $('download').focus({ preventScroll: true });
}
function applyEdit(next: EditSettings, history: 'push' | 'pop' | 'reset' = 'push') {
  if (!rawBlob || !settings || editing || busy || editorOpen) return;
  focusAction = !!document.activeElement?.matches(':focus-visible');
  $('error').hidden = true;
  editing = true;
  pendingEdit = { settings: next, history };
  view('scan');
  busyState(false);
  $<HTMLProgressElement>('progress').removeAttribute('value');
  message('Applying your changes…', 'Recalculating from the unedited scan.');
  if (focusAction) $('cancel').focus({ preventScroll: true });
  try {
    if (!editWorker) {
      editWorker = new Worker(new URL('./edit.worker.ts', import.meta.url), { type: 'module' });
      editWorker.onmessage = ({ data }: MessageEvent<EditReply>) => {
        if (data.id !== editId || !pendingEdit || !settings) return;
        if (data.type === 'error') {
          editFailed(data.message);
          return;
        }
        if (pendingEdit.history === 'pop') editHistory.pop();
        else {
          editHistory.push(cloneSettings(settings));
          if (editHistory.length > 20) editHistory.shift();
        }
        settings = pendingEdit.settings;
        pendingEdit = undefined;
        installResult(data.blob);
        editing = false;
        $<HTMLSelectElement>('color-mode').value = settings.cleanColor ? 'clean' : 'original';
        if (metadata) {
          metadata.width = data.width;
          metadata.height = data.height;
        }
        busyState(false);
        view('scan');
        message('');
        if (focusAction) $('download').focus({ preventScroll: true });
      };
      editWorker.onerror = () => editFailed('The image editor stopped. Try the edit again.');
    }
    const request: EditRequest = {
      id: ++editId,
      raw: rawBlob,
      settings: next,
      suggestion: suggestion!,
      sharpen: metadata?.sharpen ?? 0,
    };
    editWorker.postMessage(request);
  } catch (error) {
    editFailed(error instanceof Error ? error.message : 'The edit could not start. Try again.');
  }
}
$('rotate').onclick = () => {
  if (settings) applyEdit({ ...cloneSettings(settings), rotation: (settings.rotation + 1) % 4 });
};
$<HTMLSelectElement>('color-mode').onchange = () => {
  colorChoice = $<HTMLSelectElement>('color-mode').value === 'clean';
  if (settings) applyEdit({ ...cloneSettings(settings), cleanColor: colorChoice });
};
$('undo-edit').onclick = () => {
  if (editHistory.length) applyEdit(cloneSettings(editHistory[editHistory.length - 1]), 'pop');
};
$('reset-edits').onclick = () => {
  if (settings)
    applyEdit(
      {
        cleanColor: resultOptions?.cleanColor ?? true,
        mask: { ...settings.mask, data: new Uint8Array(settings.mask.data.length) },
        rotation: 0,
      },
      'reset',
    );
};
function closeEditor() {
  editorOpen = false;
  maskEditor.close();
  cropEditor.close();
  $('finger-editor').hidden = true;
  $('crop-editor').hidden = true;
  $('process-history').hidden = !steps.size;
  view('scan');
  busyState(false);
}
$('remove-fingers').onclick = async () => {
  if (!rawBlob || !settings || !suggestion || busy || editing || editorOpen) return;
  const keyboard = $('remove-fingers').matches(':focus-visible');
  editorOpen = true;
  busyState(false);
  $('error').hidden = true;
  try {
    await maskEditor.open(
      rawBlob,
      settings.reviewedMask ? settings.mask : suggestion,
      suggestion,
      settings.rotation,
    );
    if (!editorOpen) return;
    $('images').hidden = true;
    $('process-view').hidden = true;
    $('finger-editor').hidden = false;
    $('process-history').hidden = true;
    if (keyboard) $('brush-add').focus({ preventScroll: true });
    scrollTo($('finger-editor'));
  } catch {
    closeEditor();
    showError('The removal editor could not open. Try again.');
  }
};
$('mask-cancel').onclick = () => {
  const keyboard = $('mask-cancel').matches(':focus-visible');
  closeEditor();
  if (keyboard) $('remove-fingers').focus({ preventScroll: true });
};
$('mask-apply').onclick = () => {
  if (!settings || !editorOpen) return;
  const keyboard = $('mask-apply').matches(':focus-visible');
  const mask = maskEditor.selection();
  closeEditor();
  if (keyboard) $('remove-fingers').focus({ preventScroll: true });
  applyEdit({ ...cloneSettings(settings), mask, reviewedMask: true });
};
$('crop').onclick = async () => {
  if (!rawBlob || !settings || busy || editing || editorOpen) return;
  const keyboard = $('crop').matches(':focus-visible');
  editorOpen = true;
  busyState(false);
  $('error').hidden = true;
  try {
    await cropEditor.open(rawBlob, settings.rotation, settings.crop);
    if (!editorOpen) return;
    $('images').hidden = true;
    $('process-view').hidden = true;
    $('crop-editor').hidden = false;
    $('process-history').hidden = true;
    if (keyboard) document.querySelector<HTMLElement>('.crop-handle')?.focus();
    scrollTo($('crop-editor'));
  } catch {
    closeEditor();
    showError('The crop editor could not open. Try again.');
  }
};
$('crop-reset').onclick = () => cropEditor.reset();
$('crop-cancel').onclick = () => {
  const keyboard = $('crop-cancel').matches(':focus-visible');
  closeEditor();
  if (keyboard) $('crop').focus({ preventScroll: true });
};
$('crop-apply').onclick = () => {
  if (!settings || !editorOpen) return;
  const keyboard = $('crop-apply').matches(':focus-visible');
  const crop = cropEditor.selection();
  closeEditor();
  if (keyboard) $('crop').focus({ preventScroll: true });
  const next = cloneSettings(settings);
  next.crop = isFullCrop(crop) ? undefined : crop;
  applyEdit(next);
};
window.addEventListener('pagehide', () => {
  worker?.terminate();
  worker = undefined;
  editWorker?.terminate();
  editWorker = undefined;
  if (editorOpen) closeEditor();
});
window.addEventListener('pageshow', () => {
  if (editing && !editWorker) editFailed('The edit was interrupted. Try applying it again.');
  if (busy && !worker) {
    const keepResult = refining && !!resultBlob;
    refining = false;
    busyState(false);
    view(keepResult ? 'scan' : 'original');
    message(
      keepResult ? 'Refinement interrupted.' : 'Scan interrupted.',
      keepResult
        ? 'Your previous scan is still available.'
        : 'The page was left during processing. Start the scan again to continue.',
    );
  }
});
// Keeping only the selected photo and latest scan bounds memory across repeated scans.
window.addEventListener('beforeunload', () => {
  URL.revokeObjectURL(originalURL);
  URL.revokeObjectURL(resultURL);
  for (const step of steps.values()) URL.revokeObjectURL(step.url);
  for (const page of pages) URL.revokeObjectURL(page.url);
  URL.revokeObjectURL(downloadURL);
  URL.revokeObjectURL(pagesURL);
});
