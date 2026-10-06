// A minimal PDF writer: one JPEG per page, each page sized to its image with the long side
// of an A4 sheet (841.89 pt), so pages print at a document's usual size.

export type PdfPage = { jpeg: Uint8Array; width: number; height: number };

const LONG_SIDE = 841.89;
const encoder = new TextEncoder();

export function pdfFromJpegs(pages: PdfPage[]): Blob {
  const parts: Uint8Array[] = [];
  const offsets: number[] = [];
  let length = 0;
  const push = (part: string | Uint8Array) => {
    const bytes = typeof part === 'string' ? encoder.encode(part) : part;
    parts.push(bytes);
    length += bytes.length;
  };
  const object = (id: number, ...body: (string | Uint8Array)[]) => {
    offsets[id] = length;
    push(`${id} 0 obj\n`);
    for (const part of body) push(part);
    push('\nendobj\n');
  };

  push('%PDF-1.4\n');
  push(new Uint8Array([0x25, 0xe2, 0xe3, 0xcf, 0xd3, 0x0a])); // binary marker comment
  const kids = pages.map((_, index) => `${3 + 3 * index} 0 R`).join(' ');
  object(1, '<< /Type /Catalog /Pages 2 0 R >>');
  object(2, `<< /Type /Pages /Kids [${kids}] /Count ${pages.length} >>`);
  pages.forEach((page, index) => {
    const id = 3 + 3 * index;
    const scale = LONG_SIDE / Math.max(page.width, page.height);
    const width = (page.width * scale).toFixed(2),
      height = (page.height * scale).toFixed(2);
    object(
      id,
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${width} ${height}] ` +
        `/Resources << /XObject << /Im${index} ${id + 2} 0 R >> >> /Contents ${id + 1} 0 R >>`,
    );
    const content = `q ${width} 0 0 ${height} 0 0 cm /Im${index} Do Q`;
    object(id + 1, `<< /Length ${content.length} >>\nstream\n${content}\nendstream`);
    object(
      id + 2,
      `<< /Type /XObject /Subtype /Image /Width ${page.width} /Height ${page.height} ` +
        `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${page.jpeg.length} >>\nstream\n`,
      page.jpeg,
      '\nendstream',
    );
  });
  const count = 2 + 3 * pages.length;
  const xref = length;
  push(`xref\n0 ${count + 1}\n0000000000 65535 f \n`);
  for (let id = 1; id <= count; id++) push(`${String(offsets[id]).padStart(10, '0')} 00000 n \n`);
  push(`trailer\n<< /Size ${count + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
  return new Blob(parts as BlobPart[], { type: 'application/pdf' });
}
