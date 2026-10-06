import {
  AutoCapture,
  containVideo,
  guidance,
  previewInPhoto,
  smoothOutline,
  type Gray,
  type Guidance,
} from './camera-outline';
import type { CameraReply, CameraRequest, CaptureInfo, PageOutline } from './types';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const MAX_PIXELS = 20_000_000; // the scanner's input limit

type StillCapture = {
  getPhotoCapabilities(): Promise<{
    imageWidth?: { max?: number };
    imageHeight?: { max?: number };
  }>;
  takePhoto(settings?: { imageWidth?: number; imageHeight?: number }): Promise<Blob>;
};
const StillCaptureClass = (
  globalThis as unknown as { ImageCapture?: new (track: MediaStreamTrack) => StillCapture }
).ImageCapture;

function within<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => window.setTimeout(() => reject(new Error('timeout')), ms)),
  ]);
}

/** A grayscale copy of ``source`` with ``long`` pixels on its long side. */
function grayOf(source: CanvasImageSource, width: number, height: number, long: number): Gray {
  const scale = long / Math.max(width, height),
    canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext('2d', { willReadFrequently: true })!;
  context.imageSmoothingQuality = 'high';
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data,
    data = new Float32Array(canvas.width * canvas.height);
  for (let i = 0; i < data.length; i++)
    data[i] = 0.299 * pixels[i * 4] + 0.587 * pixels[i * 4 + 1] + 0.114 * pixels[i * 4 + 2];
  return { width: canvas.width, height: canvas.height, data };
}

/** ``blob`` as a capture within the scanner's pixel limit (re-encoded only when larger), with
 * a small grayscale copy to compare its view with the preview's. */
async function fitShot(blob: Blob, method: CaptureInfo['method']) {
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  try {
    const { width, height } = bitmap;
    const gray = grayOf(bitmap, width, height, 256);
    if (width * height <= MAX_PIXELS) return { blob, method, width, height, gray };
    const scale = Math.sqrt(MAX_PIXELS / (width * height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.floor(width * scale);
    canvas.height = Math.floor(height * scale);
    canvas.getContext('2d')!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const smaller = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.95),
    );
    if (!smaller) throw new Error('encode');
    return { blob: smaller, method, width: canvas.width, height: canvas.height, gray };
  } finally {
    bitmap.close();
  }
}

/** Still photos per camera (device id) while the page is open: at the largest size, at the
 * browser's default size, or none, after photos showed another view than the preview; and
 * the view of the last photo that was not used. */
const photoModes = new Map<string, { mode: 'largest' | 'default' | 'off'; view?: number }>();

const AUTO_UNAVAILABLE =
  'Auto capture needs the live outline, which is unavailable. Take photos with the shutter.';

const HINTS: Record<Guidance, string> = {
  none: 'Point the camera at a page',
  shape: 'Point the camera at one page',
  clipped: 'Move back to fit the page',
  close: 'Move back a little',
  small: 'Move closer to the page',
  tilted: 'Hold the camera straight over the page',
  ok: 'Page found',
};

export class Camera {
  private dialog = $<HTMLDialogElement>('camera');
  private video = $<HTMLVideoElement>('camera-video');
  private overlay = document.getElementById('camera-outline') as unknown as SVGSVGElement;
  private path = document.getElementById('camera-path') as unknown as SVGPathElement;
  private shutter = $<HTMLButtonElement>('camera-shutter');
  private flip = $<HTMLButtonElement>('camera-flip');
  private frame = document.createElement('canvas');
  private context = this.frame.getContext('2d', { willReadFrequently: true })!;
  private stream?: MediaStream;
  private detector?: Worker;
  private timer = 0;
  private expiry = 0;
  private watchdog = 0;
  private generation = 0;
  private facing: 'environment' | 'user' = 'environment';
  private outline: PageOutline | null = null;
  private trigger?: HTMLElement;
  private capturing = false;
  private capturedClose = false;
  private permission?: PermissionStatus;
  private nativeCapture = $<HTMLInputElement>('camera-capture');
  private autoButton = $<HTMLButtonElement>('camera-auto');
  private auto = false;
  private autoCapture = new AutoCapture();
  private unavailable: () => void = () => {};
  /** A frame is with the detector; one frame at a time keeps a single loop. */
  private inFlight = false;
  /** The live outline failed for this camera session. */
  private outlineFailed = false;

  constructor(
    /** A taken photo; ``continuous`` keeps the camera open for the next page. */
    private captured: (file: File, info?: CaptureInfo, continuous?: boolean) => void,
    private choosePhoto: () => void,
    /** After the camera closes; ``dismissed`` is false when it closed for a taken photo. */
    private closed: (dismissed: boolean) => void = () => {},
  ) {
    $('camera-close').onclick = () => this.close();
    $('camera-retry').onclick = () => void this.start();
    // The phone's camera app needs no site permission prompt, so it works when the browser
    // cannot ask (an app drawing over the screen) or remembers a refusal.
    $('camera-native').onclick = () => {
      this.close();
      this.nativeCapture.click();
    };
    this.nativeCapture.onchange = () => {
      const file = this.nativeCapture.files?.[0];
      this.nativeCapture.value = '';
      if (file) this.captured(file);
    };
    $('camera-gallery').onclick = () => {
      this.close();
      this.choosePhoto();
    };
    this.shutter.onclick = () => void this.capture();
    try {
      this.auto = localStorage.getItem('flattener-auto') === 'on';
    } catch {
      /* Storage can be unavailable; auto capture starts off. */
    }
    this.showAuto();
    this.autoButton.onclick = () => {
      this.auto = !this.auto;
      try {
        localStorage.setItem('flattener-auto', this.auto ? 'on' : 'off');
      } catch {
        /* The choice still applies to this visit. */
      }
      this.autoCapture.reset();
      this.showAuto();
      this.setCountdown(0);
    };
    this.flip.onclick = () => {
      this.facing = this.facing === 'environment' ? 'user' : 'environment';
      void this.start();
    };
    this.dialog.addEventListener('cancel', (event) => {
      event.preventDefault();
      this.close();
    });
    this.dialog.addEventListener('close', () => {
      if (this.dialog.open) return;
      this.stop();
      document.body.classList.remove('camera-open');
    });
    this.video.addEventListener('resize', () => {
      this.clearOutline();
      this.resize();
    });
    new ResizeObserver(() => this.resize()).observe($('camera-viewfinder'));
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this.dialog.open) this.close();
    });
    window.addEventListener('pagehide', () => this.close());
  }

  get isOpen() {
    return this.dialog.open;
  }

  /** Progress of pages taken in auto mode, shown under the viewfinder. */
  setPages(text: string) {
    $('camera-batch').textContent = text;
    $('camera-batch').hidden = !text;
  }

  private showAuto() {
    this.autoButton.setAttribute('aria-pressed', String(this.auto));
    this.dialog.classList.toggle('auto', this.auto);
    $('camera-hint').textContent = !this.auto
      ? 'Keep the whole page in view.'
      : this.outlineFailed
        ? AUTO_UNAVAILABLE
        : 'Auto capture takes each page when it is steady, with quick processing.';
  }

  private setCountdown(progress: number) {
    this.shutter.style.setProperty('--countdown', String(progress));
    this.shutter.classList.toggle('counting', progress > 0);
  }

  open(trigger: HTMLElement) {
    if (this.dialog.open) return;
    this.trigger = trigger;
    this.dialog.showModal();
    document.body.classList.add('camera-open');
    void this.start();
  }

  close() {
    const wasOpen = this.dialog.open;
    this.unwatchPermission();
    this.stop();
    if (this.dialog.open) this.dialog.close();
    document.body.classList.remove('camera-open');
    if (wasOpen) {
      this.trigger?.focus({ preventScroll: true });
      const dismissed = !this.capturedClose;
      this.capturedClose = false;
      this.closed(dismissed);
    }
  }

  private stop() {
    ++this.generation;
    this.inFlight = false;
    this.outlineFailed = false;
    window.clearTimeout(this.timer);
    window.clearTimeout(this.watchdog);
    this.detector?.terminate();
    this.detector = undefined;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = undefined;
    this.video.pause();
    this.video.srcObject = null;
    this.shutter.disabled = true;
    this.capturing = false;
    this.clearOutline();
  }

  private status(text: string) {
    if ($('camera-status').textContent !== text) $('camera-status').textContent = text;
  }

  private unwatchPermission() {
    if (this.permission) this.permission.onchange = null;
    this.permission = undefined;
  }

  /** After a refusal: name a remembered block, and reopen the camera once it is allowed. */
  private async watchPermission(generation: number) {
    let status: PermissionStatus;
    try {
      status = await navigator.permissions.query({ name: 'camera' as PermissionName });
    } catch {
      return; // Browsers without a camera permission query keep the general message.
    }
    if (generation !== this.generation || !this.dialog.open) return;
    this.unwatchPermission();
    this.permission = status;
    const explain = () => {
      if (status.state === 'denied')
        $('camera-error-message').textContent =
          'Camera access is blocked for this site. Allow it in the site settings next to the ' +
          'address bar, and the camera opens here. Or use the camera app.';
    };
    explain();
    status.onchange = () => {
      if (status.state === 'granted' && this.dialog.open && !this.stream) void this.start();
      else explain();
    };
  }

  private async start() {
    this.unwatchPermission();
    this.stop();
    const generation = this.generation;
    $('camera-error').hidden = true;
    this.video.hidden = true;
    this.flip.disabled = true;
    this.status('Opening camera…');
    $('camera-hint').textContent = 'Allow camera access to frame your page.';
    try {
      if (!window.isSecureContext)
        throw new Error('Open this page over HTTPS to use the camera, or choose a photo.');
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error('This browser cannot open the camera. Choose a photo to continue.');
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: this.facing },
          width: { ideal: 2560 },
          height: { ideal: 1920 },
          frameRate: { ideal: 24, max: 30 },
        },
      });
      if (generation !== this.generation) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      this.stream = stream;
      this.video.srcObject = stream;
      await this.video.play();
      if (generation !== this.generation) return;
      this.video.hidden = false;
      this.shutter.disabled = false;
      this.resize();
      this.status('Finding the page…');
      this.autoCapture.reset();
      this.showAuto();
      stream.getVideoTracks()[0].addEventListener('ended', () => {
        if (generation === this.generation)
          this.fail('The camera disconnected. Try opening it again.');
      });
      void navigator.mediaDevices
        .enumerateDevices()
        .then((devices) => {
          if (generation !== this.generation) return;
          this.flip.hidden = devices.filter((device) => device.kind === 'videoinput').length < 2;
          this.flip.disabled = false;
        })
        .catch(() => {
          /* Camera switching is optional. */
        });
      this.loadDetector(generation);
    } catch (error) {
      if (generation !== this.generation) return;
      const name = error instanceof DOMException ? error.name : '';
      this.fail(
        name === 'NotAllowedError'
          ? 'The camera was not allowed. If no permission prompt appeared, close apps that ' +
              'draw over the screen, such as chat bubbles or screen filters, and try again. ' +
              'Or use the camera app.'
          : name === 'NotFoundError'
            ? 'No camera was found. Connect one or choose a photo.'
            : name === 'NotReadableError'
              ? 'The camera is in use. Close other camera apps and try again.'
              : error instanceof Error
                ? error.message
                : 'The camera could not open. Try again.',
      );
      if (name === 'NotAllowedError') void this.watchPermission(this.generation);
    }
  }

  private fail(message: string) {
    this.stop();
    this.video.hidden = true;
    this.flip.disabled = true;
    $('camera-error').hidden = false;
    $('camera-error-message').textContent = message;
    this.status('Camera unavailable');
    $('camera-hint').textContent = 'You can also choose a photo from your library.';
  }

  private loadDetector(generation: number) {
    this.inFlight = false;
    this.outlineFailed = false;
    const unavailable = () => {
      if (generation !== this.generation) return;
      this.outlineFailed = true;
      window.clearTimeout(this.watchdog);
      window.clearTimeout(this.timer);
      this.detector?.terminate();
      this.detector = undefined;
      this.clearOutline();
      this.status('Camera ready');
      $('camera-hint').textContent = this.auto
        ? AUTO_UNAVAILABLE
        : 'Live outline unavailable. You can still take a photo.';
    };
    this.unavailable = unavailable;
    try {
      this.detector = new Worker(new URL('./camera.worker.ts', import.meta.url), {
        type: 'module',
      });
      this.detector.onerror = unavailable;
      this.watchdog = window.setTimeout(unavailable, 60000);
      this.detector.onmessage = ({ data }: MessageEvent<CameraReply>) => {
        if (generation !== this.generation) return;
        if (data.type === 'outline') this.inFlight = false;
        if (this.capturing && (data.type === 'outline' || data.type === 'progress')) {
          // The loop pauses while a photo is taken; a frame from just before is dropped, as
          // the preview can freeze or go dark meanwhile. The capture restarts the loop.
          return;
        }
        if (data.type === 'progress') {
          this.status('Loading page detection…');
        } else if (data.type === 'ready') {
          window.clearTimeout(this.watchdog);
          this.sendFrame(generation, unavailable);
        } else if (data.type === 'outline') {
          window.clearTimeout(this.watchdog);
          if (data.width === this.video.videoWidth && data.height === this.video.videoHeight)
            this.drawOutline(data.outline);
          this.timer = window.setTimeout(() => this.sendFrame(generation, unavailable), 160);
        } else unavailable();
      };
      this.detector.postMessage({
        type: 'load',
        base: new URL('.', document.baseURI).href,
      } satisfies CameraRequest);
    } catch {
      unavailable();
    }
  }

  private sendFrame(generation: number, unavailable: () => void) {
    if (generation !== this.generation || !this.detector || this.capturing || this.inFlight) return;
    if (this.video.readyState < 2 || !this.video.videoWidth) {
      this.timer = window.setTimeout(() => this.sendFrame(generation, unavailable), 100);
      return;
    }
    try {
      const width = this.video.videoWidth,
        height = this.video.videoHeight;
      const scale = Math.min(1, 480 / Math.max(width, height));
      this.frame.width = Math.max(1, Math.round(width * scale));
      this.frame.height = Math.max(1, Math.round(height * scale));
      this.context.drawImage(this.video, 0, 0, this.frame.width, this.frame.height);
      const image = this.context.getImageData(0, 0, this.frame.width, this.frame.height);
      this.detector.postMessage({ type: 'frame', image, width, height } satisfies CameraRequest, [
        image.data.buffer,
      ]);
      this.inFlight = true;
      window.clearTimeout(this.watchdog);
      this.watchdog = window.setTimeout(unavailable, 15000);
    } catch {
      unavailable();
    }
  }

  private clearOutline() {
    window.clearTimeout(this.expiry);
    this.outline = null;
    this.setCountdown(0);
    this.path.setAttribute('d', '');
    this.dialog.classList.remove('page-found', 'page-clipped');
  }

  private drawOutline(outline: PageOutline | null) {
    window.clearTimeout(this.expiry);
    const hint = guidance(outline, this.video.videoWidth / Math.max(1, this.video.videoHeight));
    if (!outline) this.clearOutline();
    else {
      this.outline = smoothOutline(this.outline, outline);
      this.path.setAttribute(
        'd',
        this.outline.points.map(([x, y], i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' ') + 'Z',
      );
      this.dialog.classList.toggle('page-found', hint === 'ok');
      this.dialog.classList.toggle('page-clipped', hint !== 'ok');
      this.expiry = window.setTimeout(() => this.expire(), 1200);
    }
    let text = HINTS[hint];
    if (this.auto) {
      const progress = this.autoCapture.frame(outline, hint === 'ok', performance.now());
      this.setCountdown(progress);
      if (hint === 'ok') text = this.autoCapture.waiting ? 'Turn to the next page' : 'Hold still';
      if (progress >= 1) void this.capture();
    }
    this.status(text);
  }

  /** No reply for a while (a slow device): hide the outline without calling it a change. */
  private expire() {
    this.clearOutline();
    this.autoCapture.stale();
    this.status('Finding the page…');
  }

  private resize() {
    if (!this.video.videoWidth || !this.video.videoHeight) return;
    const frame = $('camera-viewfinder');
    const box = containVideo(
      frame.clientWidth,
      frame.clientHeight,
      this.video.videoWidth,
      this.video.videoHeight,
    );
    Object.assign(this.overlay.style, {
      width: `${box.width}px`,
      height: `${box.height}px`,
      left: `${box.left}px`,
      top: `${box.top}px`,
    });
  }

  /** A still photo at the sensor's resolution where the browser offers one. */
  private async takeStill(largest: boolean) {
    const track = this.stream?.getVideoTracks()[0];
    if (!track || !StillCaptureClass) return null;
    try {
      const capture = new StillCaptureClass(track);
      let settings = {};
      if (largest) {
        const capabilities = await within(capture.getPhotoCapabilities(), 3000);
        const width = capabilities.imageWidth?.max,
          height = capabilities.imageHeight?.max;
        if (width && height) settings = { imageWidth: width, imageHeight: height };
      }
      return await fitShot(await within(capture.takePhoto(settings), 8000), 'photo');
    } catch {
      return null;
    }
  }

  /**
   * A still photo that shows what the preview shows. Some phones take the photo with another
   * lens (an ultra-wide one at 0.6x while the preview is at 1x). Such a photo is not used:
   * the next try asks for the browser's default photo size, and after that this camera uses
   * preview frames. ``view`` is the preview's share of the last rejected photo's width.
   */
  private async takeMatchingStill() {
    const preview = this.video.videoWidth
      ? grayOf(this.video, this.video.videoWidth, this.video.videoHeight, 64)
      : null;
    const camera = this.stream?.getVideoTracks()[0]?.getSettings().deviceId ?? '';
    let { mode, view } = photoModes.get(camera) ?? { mode: 'largest' };
    while (mode !== 'off') {
      const photo = await this.takeStill(mode === 'largest');
      if (!photo) return { shot: null, view };
      const match = preview ? previewInPhoto(preview, photo.gray) : null;
      // A poor match (motion, a rotated photo) cannot tell; the photo is kept then.
      if (!match || match.score < 0.5 || Math.abs(match.scale - 1) <= 0.1)
        return { shot: photo, view };
      view = match.scale;
      mode = mode === 'largest' ? 'default' : 'off';
      photoModes.set(camera, { mode, view });
    }
    return { shot: null, view };
  }

  /** The current preview frame, the fallback where still photos are not offered. */
  private async grabFrame() {
    const canvas = document.createElement('canvas');
    const scale = Math.min(
      1,
      Math.sqrt(MAX_PIXELS / (this.video.videoWidth * this.video.videoHeight)),
    );
    canvas.width = Math.floor(this.video.videoWidth * scale);
    canvas.height = Math.floor(this.video.videoHeight * scale);
    canvas.getContext('2d')!.drawImage(this.video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.96),
    );
    if (!blob) throw new Error('The photo could not be saved. Try taking it again.');
    return { blob, method: 'frame' as const, width: canvas.width, height: canvas.height };
  }

  private async capture() {
    if (!this.stream || this.shutter.disabled || this.capturing) return;
    const generation = this.generation;
    const started = performance.now();
    const preview = { width: this.video.videoWidth, height: this.video.videoHeight };
    this.capturing = true;
    this.shutter.disabled = true;
    this.flip.disabled = true;
    this.setCountdown(0);
    this.status('Taking photo…');
    window.clearTimeout(this.expiry);
    window.clearTimeout(this.timer);
    window.clearTimeout(this.watchdog);
    try {
      const still = await this.takeMatchingStill();
      if (generation !== this.generation) return;
      const shot = still.shot ?? (await this.grabFrame());
      if (generation !== this.generation) return;
      const info: CaptureInfo = {
        method: shot.method,
        width: shot.width,
        height: shot.height,
        preview,
        seconds: (performance.now() - started) / 1000,
      };
      if (still.view !== undefined) info.view = still.view;
      const file = new File([shot.blob], `document-${Date.now()}.jpg`, { type: 'image/jpeg' });
      if (!this.auto) {
        this.capturedClose = true;
        this.close();
        this.captured(file, info);
        return;
      }
      // Auto mode keeps the camera open; the next page needs a page change first.
      this.captured(file, info, true);
      this.autoCapture.taken();
      this.capturing = false;
      this.shutter.disabled = false;
      this.flip.disabled = this.flip.hidden;
      const viewfinder = $('camera-viewfinder');
      viewfinder.classList.remove('flash');
      void viewfinder.offsetWidth;
      viewfinder.classList.add('flash');
      this.status('Page taken');
      if (this.outlineFailed) this.showAuto();
      else if (this.inFlight)
        // A frame sent before the photo is still out; its reply restarts the loop.
        this.watchdog = window.setTimeout(this.unavailable, 15000);
      else this.timer = window.setTimeout(() => this.sendFrame(generation, this.unavailable), 160);
    } catch {
      if (generation !== this.generation) return;
      this.capturing = false;
      this.shutter.disabled = false;
      this.flip.disabled = this.flip.hidden;
      this.status('Could not take photo. Try again.');
      this.detector?.terminate();
      this.detector = undefined;
      this.clearOutline();
      this.loadDetector(generation);
    }
  }
}
