"""Huawei Notes exports (.hinote) -> pen strokes of every page.

The file is only read as data. The format has no public description; these notes are reverse-engineered from real notebooks.
A truncated zip is read through its local file headers.
The notes below come from a fuller converter: they also describe fields, flags and options that this reader does not use.

Layout (zip): <note-uuid>.jhinote (gzip JSON: customNoteContent.noteTitle, pageRatio = width/height), pages/<page-uuid>.jhinote
(gzip JSON customNotePageContent: pageNumber = order in notebook, pageRatio, attachment[] whose filePath .../hwFile/<name>.bin
is the page's ink), custom_md.jhinote (hashes), files/<page>_<ts>_<n>.bin (ink) + files/<page>_<ts>_<n>.jpg (rendered preview,
763x1080 for A4). Pages without a .bin attachment carry no ink and are skipped.
.bin "PENCILENGINE", all big-endian: magic[12], u32 header_len=76, u32[4] (0x10001,0,1,0), zero pad to 76. Then nested chunks,
each = 48-byte header [u32 hdr_len=48, u32 type, u32 version, u32 payload_len, uuid[16], u32 x4] + payload.
 type 2 = layer (one per page): ex = [pre_len=12, n_strokes, strokes_bytes, 1]; payload = u32[3] (0, 1000, 1000) then
  n_strokes stroke chunks, then a trailing type-3 chunk (stroke-uuid index, ignored).
 type 0 = stroke (version 0x10000 or 0x10002, same layout): ex = [68, 48, 116, data_len]; payload = 116-byte header + data.
  header u32 slots: 0-1 i64 t_modified (epoch ms; == t_start unless the stroke was moved/copied later), 3 f32 last point
  index, 4 0x01000000, 5 0xffffffff for free ink (shape id otherwise), 6-7 i64 t_start (pen down, epoch ms), 8-9 i64 t_end,
  10-16 extra params (non-zero only on tool-11 strokes), 17 u32 tool, 19-21 f32 b,g,r in 0..1 (BGR order!), 22-23 f32 (1.0),
  24 f32 brush width (page units), rest 0.
  tool: 1 = pen (99.8% of strokes), 11 = a second pen type (same real pressure/time, e.g. rotated red notes),
  4 = translucent highlighter, 2 = shape tool (lines/rects/circles: every point pressure 1.0 and t 0).
  data: u32 0|2 (2 = shape), u32 n_points, u32 rec_size=36, n_points x [u32 flag (0 down, 4/6 move), f32 index, f32 x,
  f32 y, u32 t_ms since t_start, f32 pressure 0..1, f32 orientation (rad), f32 tilt (rad), f32 0.2], then u32 5 + f32 last
  index (pen-up marker, no coordinates). Point 0 (flag 0) is the pen-down sample at t=0, usually at/next to point 1; kept.
Coordinates: page units, page width = 1000, height = 1000/pageRatio (1414.5 for A4, 1196.6 for the 0.8357 PDF pages),
origin top-left, y down; the preview jpg is the same page scaled by jpg_width/1000. Pressure and per-point time are real
(stylus, ~8-16 ms steps). Strokes that were moved/copied after writing (~1.4%) keep pressure but every point has t=0: for
them t_ms = point index (header t_start/t_end still real). Erased ink is not stored (no eraser records; renders of the
stored strokes match the previews), so every stored stroke is visible. Written by default: tools 1 and 11 with slot 5 ==
0xffffffff and data flag 0 (pen='pen' / 'pen11'), in file order (= drawing order; t_start decreases only at 0.04% of
neighbouring pairs, moved strokes), t0_ms = t_start. Shapes (tool 2, or
slot 5 != 0xffffffff, or data flag 2) only with --shapes as pen='shape'; highlighter (tool 4) only with --highlighter as
pen='highlighter'. Inserted pictures/PDF backgrounds live in the page JSON (pageElement, bkgAttachmentId), not in .bin,
and are never written. Colour is kept as stored: 3 pages with dark paper (background base6) are written in white ink.
"""
import sys, os, json, zipfile, gzip, struct, zlib, io, re

TOOLS = {1: 'pen', 11: 'pen11', 4: 'highlighter', 2: 'shape'}
U = lambda d, o: struct.unpack_from('>I', d, o)[0]


def read_entries(path):
    """name -> bytes. Normal zip via zipfile (in memory); broken/truncated zip by scanning local headers."""
    raw = open(path, 'rb').read()
    try:
        z = zipfile.ZipFile(io.BytesIO(raw)); return {n: z.read(n) for n in z.namelist() if not n.endswith('/')}, None
    except zipfile.BadZipFile:
        pass
    out, o = {}, 0
    while True:
        o = raw.find(b'PK\x03\x04', o)
        if o < 0 or o + 30 > len(raw): break
        ver, flg, meth, _, _, crc, csz, usz, nl, xl = struct.unpack_from('<HHHHHIIIHH', raw, o + 4)
        name = raw[o + 30:o + 30 + nl].decode('utf8', 'replace'); b = o + 30 + nl + xl
        try:
            if meth == 8:
                dec = zlib.decompressobj(-15); data = dec.decompress(raw[b:b + csz] if csz and not flg & 8 else raw[b:])
                if not dec.eof: raise ValueError('truncated')
                end = len(raw) - len(dec.unused_data) if not (csz and not flg & 8) else b + csz
            elif meth == 0 and csz and not flg & 8:
                data = raw[b:b + csz]; end = b + csz
                if len(data) < csz: raise ValueError('truncated')
            else:
                raise ValueError('method')
            if crc and zlib.crc32(data) & 0xFFFFFFFF != crc: raise ValueError('crc')
            if not name.endswith('/'): out[name] = data
            o = max(end, o + 4)
        except Exception:
            o += 4
    return out, 'recovered %d entries from truncated zip' % len(out)


def jload(b):
    try: b = gzip.decompress(b)
    except OSError: pass
    return json.loads(b.decode('utf8'))


def decode_bin(d, shapes=False, highlighter=False):
    """PENCILENGINE bytes -> list of ink-v1 strokes (file = drawing order)."""
    if d[:12] != b'PENCILENGINE': raise ValueError('bad magic')
    o, strokes = U(d, 12), []
    while o + 48 <= len(d):
        hl, typ, ver, L = U(d, o), U(d, o + 4), U(d, o + 8), U(d, o + 12)
        if typ == 2:
            pre, ns = U(d, o + 32), U(d, o + 36); q = o + hl + pre; e = min(o + hl + L, len(d))
            for _ in range(ns):
                if q + 48 > e: break
                shl, styp, SL = U(d, q), U(d, q + 4), U(d, q + 12); h = q + shl; hlen = U(d, q + 40); nq = h + SL
                if styp != 0 or nq > e: q = nq; continue
                q = nq
                tmod, = struct.unpack_from('>q', d, h); tstart, = struct.unpack_from('>q', d, h + 24)
                slot5 = U(d, h + 20); tool = U(d, h + 68)
                b, g, r, _a1, _a2, w = struct.unpack_from('>6f', d, h + 76)
                D = h + hlen; fmt, n, rs = U(d, D), U(d, D + 4), U(d, D + 8)
                pen = 'shape' if (tool == 2 or slot5 != 0xFFFFFFFF or fmt != 0) else TOOLS.get(tool, 'tool%d' % tool)
                if (pen == 'shape' and not shapes) or (pen == 'highlighter' and not highlighter): continue
                if n == 0 or rs < 24 or D + 12 + n * rs > nq: continue
                raw = [struct.unpack_from('>IfffIf', d, D + 12 + rs * i) for i in range(n)]
                notime = n > 1 and all(p[4] == 0 for p in raw)
                P = [[round(x, 3), round(y, 3), i if notime else int(t), round(p, 4)] for i, (_f, _k, x, y, t, p) in enumerate(raw)]
                strokes.append(dict(t0_ms=tstart or tmod, pen=pen, color=[round(255 * c) for c in (r, g, b)], width=round(w, 4), points=P))
        o += hl + L
    return strokes


def read(path):
    """-> the pages in notebook order: [{"width", "height", "strokes": [[[x, y], ...], ...]}], strokes in writing order."""
    ents, _ = read_entries(path)
    head = [n for n in ents if n.endswith('.jhinote') and '/' not in n and n != 'custom_md.jhinote']
    ratio = None
    if head:
        try: ratio = jload(ents[head[0]])['customNoteContent'].get('pageRatio')
        except Exception: pass
    pages = []
    for n, b in ents.items():
        if not (n.startswith('pages/') and n.endswith('.jhinote')): continue
        try: pc = jload(b)['customNotePageContent']
        except Exception: continue
        if pc.get('isDelete'): continue
        bins = [os.path.basename(a.get('filePath', '')) for a in pc.get('attachment', []) if a.get('filePath', '').endswith('.bin') and not a.get('isDelete')]
        pages.append((pc.get('pageNumber', 0), pc.get('createTime', 0), pc['id'], pc.get('pageRatio') or ratio, bins))
    pages.sort()
    out = []
    for _, _, _, pr, bins in pages:
        strokes = []
        for bn in bins:
            if 'files/' + bn in ents: strokes += decode_bin(ents['files/' + bn])
        out.append(dict(width=1000.0, height=round(1000.0 / pr, 3) if pr else 1414.5, strokes=[[p[:2] for p in s['points']] for s in strokes]))
    return out
