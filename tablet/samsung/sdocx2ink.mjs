// Samsung Notes .sdocx -> one JSON file per page with the pen strokes, for tablet/app_line.py:
//   {"format": "ink-v1", "source": "sdocx", "page": n, "width", "height", "strokes": [{"points": [[x, y, t_ms, pressure], ...]}]}
// The parser is the npm package @twango/sdocx (GPL-3.0, https://github.com/twangodev/sdocx); `npm ci` in this folder installs it.
//   node sdocx2ink.mjs <out_dir> <file.sdocx> [...]
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { basename, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import init, { parse } from '@twango/sdocx';
const here = dirname(fileURLToPath(import.meta.url));
await init(readFileSync(join(here, 'node_modules/@twango/sdocx/sdocx_bg.wasm')));
const [, , outDir, ...files] = process.argv;
if (!outDir || !files.length) { console.log('usage: node sdocx2ink.mjs <out_dir> <file.sdocx> [...]'); process.exit(2); }
mkdirSync(outDir, { recursive: true });
for (const f of files) {
  let doc;
  try { doc = parse(new Uint8Array(readFileSync(f))); } catch (e) { console.log('FAIL', basename(f), String(e).slice(0, 80)); continue; }
  const stem = basename(f).replace(/\.sdocx$/, '');
  doc.pages.forEach((pg, pi) => {
    const strokes = [];
    for (const ob of pg.objects || []) {
      const s = ob.content?.Stroke;
      if (!s || !s.points?.length) continue;
      strokes.push({ pen: s.rendering?.pen_name?.split('.').pop() ?? null, points: s.points.map((p, i) => [+p.x.toFixed(2), +p.y.toFixed(2), s.timestamps?.[i] ?? i, +(s.pressures?.[i] ?? 1).toFixed(3)]) });
    }
    const name = `${stem}_p${String(pi).padStart(2, '0')}.json`;
    writeFileSync(join(outDir, name), JSON.stringify({ format: 'ink-v1', source: 'sdocx', doc: stem, page: pi, width: pg.width, height: pg.height, strokes }));
    console.log(`${name}: ${strokes.length} strokes`);
  });
}
