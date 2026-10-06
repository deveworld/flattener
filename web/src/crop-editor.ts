import { FULL_CROP, MIN_CROP, turnCrop, type Crop } from './crop';

type Handle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w' | 'move';
const MAX_SIDE = 1600;

/**
 * Crop box over the flat scan, shown as it is turned. Corners and edges resize the box,
 * dragging inside moves it, and arrow keys move a focused handle. The selection comes
 * back in the unrotated scan's fractions, so later quarter turns keep it.
 */
export class CropEditor {
  private surface = document.getElementById('crop-surface') as HTMLDivElement;
  private canvas = document.getElementById('crop-canvas') as HTMLCanvasElement;
  private box = document.getElementById('crop-box') as HTMLDivElement;
  private window = document.getElementById('crop-window') as HTMLDivElement;
  private rotation = 0;
  /** The box in the shown (turned) image, as fractions. */
  private rect: Crop = { ...FULL_CROP };
  private drag?: { handle: Handle; pointer: number; start: Crop; x: number; y: number };
  private version = 0;

  constructor() {
    this.box.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || this.drag) return;
      const target = (event.target as HTMLElement).closest<HTMLElement>('[data-handle]');
      const handle = (target?.dataset.handle as Handle | undefined) ?? 'move';
      const [x, y] = this.position(event);
      event.preventDefault();
      this.box.setPointerCapture(event.pointerId);
      this.drag = { handle, pointer: event.pointerId, start: { ...this.rect }, x, y };
      this.surface.classList.add('dragging');
    });
    this.box.addEventListener('pointermove', (event) => {
      if (!this.drag || event.pointerId !== this.drag.pointer) return;
      const [x, y] = this.position(event);
      this.rect = resize(this.drag.start, this.drag.handle, x - this.drag.x, y - this.drag.y);
      this.layout();
    });
    const end = (event: PointerEvent) => {
      if (!this.drag || event.pointerId !== this.drag.pointer) return;
      this.drag = undefined;
      this.surface.classList.remove('dragging');
    };
    this.box.addEventListener('pointerup', end);
    this.box.addEventListener('pointercancel', end);
    this.box.addEventListener('keydown', (event) => {
      const target = (event.target as HTMLElement).closest<HTMLElement>('[data-handle]');
      const step = event.shiftKey ? 0.05 : 0.01;
      const moves: Record<string, [number, number]> = {
        ArrowLeft: [-step, 0],
        ArrowRight: [step, 0],
        ArrowUp: [0, -step],
        ArrowDown: [0, step],
      };
      const move = moves[event.key];
      if (!target || !move) return;
      event.preventDefault();
      this.rect = resize(this.rect, target.dataset.handle as Handle, move[0], move[1]);
      this.layout();
    });
  }

  async open(raw: Blob, rotation: number, crop: Crop | undefined) {
    const version = ++this.version;
    this.rotation = ((rotation % 4) + 4) % 4;
    const bitmap = await createImageBitmap(raw);
    try {
      if (version !== this.version) return;
      const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
      const width = Math.max(1, Math.round(bitmap.width * scale)),
        height = Math.max(1, Math.round(bitmap.height * scale));
      const turned = this.rotation % 2 === 1;
      this.canvas.width = turned ? height : width;
      this.canvas.height = turned ? width : height;
      const ctx = this.canvas.getContext('2d')!;
      ctx.save();
      ctx.translate(this.canvas.width / 2, this.canvas.height / 2);
      ctx.rotate((this.rotation * Math.PI) / 2);
      ctx.drawImage(bitmap, -width / 2, -height / 2, width, height);
      ctx.restore();
    } finally {
      bitmap.close();
    }
    this.rect = turnCrop(crop ?? FULL_CROP, this.rotation);
    this.layout();
  }

  /** The crop in the unrotated scan's fractions. */
  selection(): Crop {
    return turnCrop(this.rect, 4 - this.rotation);
  }

  reset() {
    this.rect = { ...FULL_CROP };
    this.layout();
  }

  close() {
    ++this.version;
    this.drag = undefined;
    this.surface.classList.remove('dragging');
  }

  private position(event: PointerEvent): [number, number] {
    const bounds = this.surface.getBoundingClientRect();
    return [
      (event.clientX - bounds.left) / Math.max(1, bounds.width),
      (event.clientY - bounds.top) / Math.max(1, bounds.height),
    ];
  }

  private layout() {
    const { left, top, right, bottom } = this.rect;
    const style = {
      left: `${left * 100}%`,
      top: `${top * 100}%`,
      width: `${(right - left) * 100}%`,
      height: `${(bottom - top) * 100}%`,
    };
    Object.assign(this.box.style, style);
    Object.assign(this.window.style, style);
  }
}

/** ``rect`` with ``handle`` moved by (dx, dy), kept inside the image and above the minimum. */
function resize(rect: Crop, handle: Handle, dx: number, dy: number): Crop {
  let { left, top, right, bottom } = rect;
  if (handle === 'move') {
    const x = Math.min(1 - right, Math.max(-left, dx)),
      y = Math.min(1 - bottom, Math.max(-top, dy));
    return { left: left + x, top: top + y, right: right + x, bottom: bottom + y };
  }
  if (handle.includes('w')) left = Math.min(right - MIN_CROP, Math.max(0, left + dx));
  if (handle.includes('e')) right = Math.max(left + MIN_CROP, Math.min(1, right + dx));
  if (handle.includes('n')) top = Math.min(bottom - MIN_CROP, Math.max(0, top + dy));
  if (handle.includes('s')) bottom = Math.max(top + MIN_CROP, Math.min(1, bottom + dy));
  return { left, top, right, bottom };
}
