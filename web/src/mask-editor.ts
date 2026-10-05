import { maskCount, rotateMask } from './cleanup';
import type { Mask } from './types';

/** A bounded-resolution editable mask; the original image never changes here. */
export class MaskEditor {
  private canvas = document.getElementById('mask-canvas') as HTMLCanvasElement;
  private ctx = this.canvas.getContext('2d')!;
  private paper = document.createElement('canvas');
  private overlay = document.createElement('canvas');
  private mask!: Mask;
  private automatic!: Mask;
  private undo: Uint8Array[] = [];
  private rotation = 0;
  private erase = false;
  private painting = false;
  private pointer: number | undefined;
  private cursor: [number, number] | undefined;
  private previous: [number, number] | undefined;
  private version = 0;
  private active = false;
  private $<T extends HTMLElement>(id: string) {
    return document.getElementById(id) as T;
  }

  constructor() {
    this.$('brush-add').onclick = () => this.setTool(false);
    this.$('brush-erase').onclick = () => this.setTool(true);
    this.$('mask-undo').onclick = () => {
      const data = this.undo.pop();
      if (data) {
        this.mask.data = data;
        this.refresh();
      }
    };
    this.$('mask-clear').onclick = () => {
      this.checkpoint();
      this.mask.data.fill(0);
      this.refresh();
    };
    this.$('mask-auto').onclick = () => {
      this.checkpoint();
      this.mask.data = this.automatic.data.slice();
      this.refresh();
    };
    this.$<HTMLInputElement>('brush-size').oninput = () => this.draw();
    this.canvas.onpointerdown = (e) => {
      if (!this.active || this.pointer !== undefined || e.button !== 0) return;
      e.preventDefault();
      this.canvas.focus({ preventScroll: true });
      this.pointer = e.pointerId;
      this.canvas.setPointerCapture(e.pointerId);
      this.cursor = this.position(e);
      this.begin();
      this.paint(this.cursor);
    };
    this.canvas.onpointermove = (e) => {
      if (!this.active || (this.pointer !== undefined && e.pointerId !== this.pointer)) return;
      this.cursor = this.position(e);
      if (this.painting) this.paint(this.cursor);
      else this.draw();
    };
    const finish = () => {
      this.painting = false;
      this.previous = undefined;
      this.pointer = undefined;
    };
    this.canvas.onpointerup = finish;
    this.canvas.onpointercancel = finish;
    this.canvas.onlostpointercapture = finish;
    this.canvas.onpointerleave = () => {
      if (!this.painting) {
        this.cursor = undefined;
        this.draw();
      }
    };
    this.canvas.onkeydown = (e) => {
      if (!this.active) return;
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', ' '].includes(e.key)) return;
      e.preventDefault();
      this.cursor ??= [this.mask.width / 2, this.mask.height / 2];
      if (e.key === ' ') {
        if (!e.repeat) this.begin();
        this.paint(this.cursor);
        return;
      }
      const step = e.shiftKey ? 1 : Math.max(1, this.radius() / 3);
      this.cursor[0] = Math.max(
        0,
        Math.min(
          this.mask.width - 1,
          this.cursor[0] + (e.key === 'ArrowRight' ? step : e.key === 'ArrowLeft' ? -step : 0),
        ),
      );
      this.cursor[1] = Math.max(
        0,
        Math.min(
          this.mask.height - 1,
          this.cursor[1] + (e.key === 'ArrowDown' ? step : e.key === 'ArrowUp' ? -step : 0),
        ),
      );
      if (this.painting) this.paint(this.cursor);
      else this.draw();
    };
    this.canvas.onkeyup = (e) => {
      if (e.key === ' ') finish();
    };
    this.canvas.onblur = () => {
      finish();
      this.cursor = undefined;
      this.draw();
    };
  }

  async open(raw: Blob, mask: Mask, automatic: Mask, rotation: number) {
    const version = ++this.version;
    const bitmap = await createImageBitmap(raw);
    try {
      if (version !== this.version) return;
      this.rotation = rotation;
      this.mask = rotateMask(mask, rotation);
      this.automatic = rotateMask(automatic, rotation);
      const rotated = rotation % 2 !== 0;
      const w = rotated ? bitmap.height : bitmap.width,
        h = rotated ? bitmap.width : bitmap.height;
      const scale = Math.min(1, 1200 / Math.max(w, h));
      this.paper.width = this.canvas.width = Math.round(w * scale);
      this.paper.height = this.canvas.height = Math.round(h * scale);
      const ctx = this.paper.getContext('2d')!;
      ctx.translate(this.paper.width / 2, this.paper.height / 2);
      ctx.rotate((rotation * Math.PI) / 2);
      ctx.scale(scale, scale);
      ctx.drawImage(bitmap, -bitmap.width / 2, -bitmap.height / 2);
      this.undo = [];
      this.cursor = undefined;
      this.previous = undefined;
      this.active = true;
      this.painting = false;
      this.pointer = undefined;
      this.setTool(false);
      this.refresh();
    } finally {
      bitmap.close();
    }
  }
  close() {
    this.active = false;
    this.painting = false;
    this.pointer = undefined;
    ++this.version;
  }
  selection() {
    return rotateMask(this.mask, -this.rotation);
  }
  private setTool(erase: boolean) {
    this.erase = erase;
    this.$('brush-add').setAttribute('aria-pressed', String(!erase));
    this.$('brush-erase').setAttribute('aria-pressed', String(erase));
    this.draw();
  }
  private radius() {
    return (
      (Number(this.$<HTMLInputElement>('brush-size').value) *
        Math.min(this.mask.width, this.mask.height)) /
      200
    );
  }
  private position(e: PointerEvent): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    return [
      ((e.clientX - rect.left) * this.mask.width) / rect.width,
      ((e.clientY - rect.top) * this.mask.height) / rect.height,
    ];
  }
  private checkpoint() {
    this.undo.push(this.mask.data.slice());
    if (this.undo.length > 20) this.undo.shift();
  }
  private begin() {
    this.checkpoint();
    this.painting = true;
    this.previous = undefined;
  }
  private paint(point: [number, number]) {
    const from = this.previous ?? point,
      r = this.radius();
    const steps = Math.max(
      1,
      Math.ceil(Math.hypot(point[0] - from[0], point[1] - from[1]) / Math.max(1, r / 3)),
    );
    for (let step = 0; step <= steps; step++) {
      const x = from[0] + ((point[0] - from[0]) * step) / steps,
        y = from[1] + ((point[1] - from[1]) * step) / steps;
      for (
        let yy = Math.max(0, Math.floor(y - r));
        yy < Math.min(this.mask.height, Math.ceil(y + r));
        yy++
      )
        for (
          let xx = Math.max(0, Math.floor(x - r));
          xx < Math.min(this.mask.width, Math.ceil(x + r));
          xx++
        )
          if ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r)
            this.mask.data[yy * this.mask.width + xx] = this.erase ? 0 : 1;
    }
    this.previous = point;
    this.refresh();
  }
  private refresh() {
    this.overlay.width = this.mask.width;
    this.overlay.height = this.mask.height;
    const ctx = this.overlay.getContext('2d')!,
      overlay = ctx.createImageData(this.mask.width, this.mask.height);
    for (let i = 0; i < this.mask.data.length; i++)
      if (this.mask.data[i]) overlay.data.set([211, 65, 48, 140], i * 4);
    ctx.putImageData(overlay, 0, 0);
    const count = maskCount(this.mask),
      percent = (100 * count) / this.mask.data.length;
    const tooLarge = percent > 20;
    this.$('mask-status').textContent = tooLarge
      ? 'The selection is too large. Keep it below 20% of the page.'
      : count
        ? `${percent.toFixed(1)}% of the page selected. Red areas will be filled with paper color.`
        : 'No area selected. Paint over a finger to remove it.';
    this.$<HTMLButtonElement>('mask-apply').disabled = tooLarge;
    this.$<HTMLButtonElement>('mask-undo').disabled = !this.undo.length;
    this.$<HTMLButtonElement>('mask-clear').disabled = !count;
    this.$<HTMLButtonElement>('mask-auto').disabled = !maskCount(this.automatic);
    this.draw();
  }
  private draw() {
    if (!this.active) return;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.drawImage(this.paper, 0, 0);
    this.ctx.drawImage(this.overlay, 0, 0, this.canvas.width, this.canvas.height);
    if (this.cursor) {
      const scale = this.canvas.width / this.mask.width;
      this.ctx.beginPath();
      this.ctx.arc(
        this.cursor[0] * scale,
        this.cursor[1] * scale,
        this.radius() * scale,
        0,
        Math.PI * 2,
      );
      this.ctx.lineWidth = 3;
      this.ctx.strokeStyle = '#fff';
      this.ctx.stroke();
      this.ctx.lineWidth = 1.5;
      this.ctx.strokeStyle = this.erase ? '#245e46' : '#8a291e';
      this.ctx.stroke();
    }
  }
}
