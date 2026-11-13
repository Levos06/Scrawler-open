"""GoodNotes 5 and 6 notebooks (.goodnotes) -> pen strokes of every page.

The file is only read as data: zipfile, a hand-written protobuf wire decoder and an LZ4 block decoder. Nothing is extracted to
disk. The format has no public description; these notes are reverse-engineered from real notebooks and may not cover every
version of the app.
The notes below come from a fuller converter: they also describe fields, flags and options that this reader does not use.

- zip: notes/<uuid> page content, search/<uuid> recognition index, attachments/<uuid> PDF/JPEG/MP4,
  index.events.pb document/page records, index.*.pb file lists.
- every file is a stream of varint-length-delimited protobuf records.
- index.events.pb: record field 54 = page {2: page id, 3: {1: template id}, 4: {1: fractional order key}};
  field 2 = page template {2: id, 4: background attachment, 8: {1: w, 2: h} page size in points}.
  notes/ and search/ file uuid = page uuid + 1 (as a 128-bit integer).
- notes/<uuid>: pairs of records keyed by element uuid: a header {1: uuid, 2: {1: version}, 3: 1 if deleted,
  9: Lamport clock} and a body. Body field 7 = stroke, field 1 = placed image, field 20 = text box.
  Stroke body: 2 = Apple "bv41" LZ4 frame holding a "tpl" record with signature
  vuA(v)A(S(uu))A(S(uuuu))vA(f) = [2, pen width, segment types, [start point], [(cx,cy,ex,ey) quad segments],
  1, extra floats]; 4 = RGBA floats 0..1; 5 = 1 for highlighter; 6 = {1: dx, 2: dy} translation (lasso move);
  7 = {1,2: z-order key}; 9 = recognised shape (line/ellipse/polygon) with empty point list; 14 = 1 deleted.
- The quad segments are midpoint smoothing of the raw samples: end_i == (c_i + c_{i+1}) / 2 exactly, so the
  raw (already distance-decimated) pen samples are start, c_1..c_n, e_n. Those are what we write.
- search/<uuid>: one protobuf message, field 6 repeated = recognition entries in reading order:
  {1: bbox {1: origin, 2: size}, 3: top-3 alternatives, 8: line number, 11: locale,
   7: items {3: stroke uuids of the group, 4: the word's points}}. Entries with one item are words, entries with two
  items are separators (' ' or '\\n') between them. Words are mapped to strokes by exact point match; a cursive
  stroke can be split between two words, so a stroke may be listed under both.
- GoodNotes 6 (2025+, seen in notebooks of October 2026) writes a second stroke record, signature
  vA(v)A(u)A(u)A(v)A(v)A(u)A(u)A(u)A(u)A(v) = [2, u16 flags per segment, [x, y, w] start sample, [x, y, w]*N raw
  samples, u16 per segment, u16 ops, [x, y]*M, [], [x, y]*K outline, [x, y, w, tx, ty]*N, u16*N]. When the first
  segment flag has bit 4 set (flags 4, 5, ...; seen in 2 of 23 notebooks, mixed with the other kind), the start record is
  [x, y, w, e] and the sample array is per segment [x, y, w, x, y, w, e1, e2, e3] (two samples + an extra triple that
  is skipped; reading it as samples draws fans of lines across the page). Only the samples
  are used: x, y and w = rendered width per point (pressure-driven, 0.6..2.2 pt, no nominal width in the record).
  The width is kept as "widths"; pressure = w / nominal, nominal = 99th percentile of w over the notebook
  (written as width_nominal on the page), page tagged pressure:"width". The old record keeps pressure 1.0.
- Otherwise there is NO per-point pressure and NO per-point or per-stroke clock time in the file: pressure is written as
  1.0, t_ms as the point index inside the stroke, t0_ms as null, and the page is tagged time:"ordinal".
  Strokes are ordered by the z-order key (= drawing order unless re-arranged), ties by the Lamport clock.
"""
import sys
import os
import json
import struct
import zipfile
import collections
import uuid as uuidlib


def varint(b, i):
    r = 0
    s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def fields(b):
    # generic protobuf wire decoder -> list of (field, wiretype, value)
    i = 0
    out = []
    n = len(b)
    while i < n:
        k, i = varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = varint(b, i)
        elif w == 1:
            v = b[i:i + 8]
            i += 8
        elif w == 2:
            ln, i = varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif w == 5:
            v = b[i:i + 4]
            i += 4
        else:
            raise ValueError('wire type %d' % w)
        if i > n:
            raise ValueError('field overrun')
        out.append((f, w, v))
    return out


def fdict(b):
    # field -> list of values
    d = {}
    for f, w, v in fields(b):
        d.setdefault(f, []).append(v)
    return d


def first(d, k, default=None):
    v = d.get(k)
    return v[0] if v else default


def records(b):
    i = 0
    out = []
    while i < len(b):
        ln, i = varint(b, i)
        out.append(b[i:i + ln])
        i += ln
    return out


def f32(v):
    return struct.unpack('<f', v)[0]


def vec2(b):
    d = fdict(b)
    x = first(d, 1)
    y = first(d, 2)
    return (f32(x) if x is not None else 0.0, f32(y) if y is not None else 0.0)


def lz4_block(src):
    dst = bytearray()
    i = 0
    n = len(src)
    while i < n:
        tok = src[i]
        i += 1
        ll = tok >> 4
        if ll == 15:
            while True:
                c = src[i]
                i += 1
                ll += c
                if c != 255:
                    break
        dst += src[i:i + ll]
        i += ll
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        ml = tok & 15
        if ml == 15:
            while True:
                c = src[i]
                i += 1
                ml += c
                if c != 255:
                    break
        ml += 4
        st = len(dst) - off
        if st < 0:
            raise ValueError('lz4 offset')
        for k in range(ml):
            dst.append(dst[st + k])
    return bytes(dst)


def bvx(b):
    # Apple libcompression LZ4 container: blocks "bv41" <raw u32> <packed u32> data | "bv4-" <raw u32> data | "bv4$"
    out = bytearray()
    i = 0
    while i < len(b):
        m = b[i:i + 4]
        if m == b'bv41':
            dl, cl = struct.unpack('<II', b[i + 4:i + 12])
            blk = lz4_block(b[i + 12:i + 12 + cl])
            if len(blk) != dl:
                raise ValueError('lz4 size')
            out += blk
            i += 12 + cl
        elif m == b'bv4-':
            dl, = struct.unpack('<I', b[i + 4:i + 8])
            out += b[i + 8:i + 8 + dl]
            i += 8 + dl
        elif m == b'bv4$':
            break
        else:
            raise ValueError('bvx magic %r' % m)
    return bytes(out)


def tpl_sig(s, i=0):
    out = []
    while i < len(s):
        c = s[i]
        if c in 'vuf':
            out.append(c)
            i += 1
        elif c in 'AS':
            sub, i = tpl_sig(s, i + 2)
            out.append((c, sub))
        elif c == ')':
            return out, i + 1
        else:
            raise ValueError('tpl char %r' % c)
    return out, i


def tpl_node(node, b, o):
    # v = u16, u = 4-byte value (float in practice), f = float, A(..) = u32 count + items, S(..) = struct
    if isinstance(node, str):
        if node == 'v':
            return struct.unpack('<H', b[o:o + 2])[0], o + 2
        if o + 4 > len(b):
            raise ValueError('tpl short')
        return struct.unpack('<f', b[o:o + 4])[0], o + 4
    kind, sub = node
    if kind == 'S':
        vals = []
        for s in sub:
            v, o = tpl_node(s, b, o)
            vals.append(v)
        return tuple(vals), o
    cnt = struct.unpack('<I', b[o:o + 4])[0]
    o += 4
    if cnt * 2 > len(b):
        raise ValueError('tpl count')
    item = sub[0] if len(sub) == 1 else ('S', sub)
    vals = []
    for _ in range(cnt):
        v, o = tpl_node(item, b, o)
        vals.append(v)
    return vals, o


def parse_tpl(b):
    if b[:4] != b'tpl\x00':
        raise ValueError('not tpl')
    e = b.index(b'\x00', 8)
    sig = b[8:e].decode('ascii')
    nodes, _ = tpl_sig(sig)
    o = e + 1
    vals = []
    for n in nodes:
        v, o = tpl_node(n, b, o)
        vals.append(v)
    return sig, vals


STROKE_SIG = 'vuA(v)A(S(uu))A(S(uuuu))vA(f)'
STROKE_SIG2 = 'vA(v)A(u)A(u)A(v)A(v)A(u)A(u)A(u)A(u)A(v)'


def uuid_plus(u, k):
    return str(uuidlib.UUID(int=(uuidlib.UUID(u).int + k) % (1 << 128))).upper()


def read_pages(z):
    """-> list of dicts {page_id, notes, search, width, height, bg} in notebook order."""
    names = set(z.namelist())
    tmpl = {}
    pages = []
    if 'index.events.pb' in names:
        for r in records(z.read('index.events.pb')):
            fs = fields(r)
            if len(fs) < 2 or fs[1][1] != 2:
                continue
            kind, body = fs[1][0], fs[1][2]
            if kind == 2:
                d = fdict(body)
                tid = first(d, 2, b'').decode('utf-8', 'replace')
                size = vec2(first(d, 8)) if first(d, 8) else None
                tmpl[tid] = dict(size=size, bg=first(d, 4, b'').decode('utf-8', 'replace') or None)
            elif kind == 54:
                d = fdict(body)
                pid = first(d, 2, b'').decode('utf-8', 'replace')
                key = first(fdict(first(d, 4)), 1, b'') if first(d, 4) else b''
                tid = first(fdict(first(d, 3)), 1, b'').decode('utf-8', 'replace') if first(d, 3) else None
                pages.append((key, pid, tid))
    # fractional-index keys sort as plain bytes
    pages.sort()
    out = []
    used = set()
    for key, pid, tid in pages:
        nf = None
        for k in (1, 0):
            try:
                cand = 'notes/' + uuid_plus(pid, k)
            except ValueError:
                continue
            if cand in names:
                nf = cand
                break
        if nf is None:
            continue
        used.add(nf)
        t = tmpl.get(tid, {})
        sz = t.get('size')
        sf = 'search/' + nf[6:]
        out.append(dict(page_id=pid, notes=nf, search=sf if sf in names else None,
                        width=sz[0] if sz else None, height=sz[1] if sz else None, bg=t.get('bg')))
    # notes files without a page record go last, in index.notes.pb order
    rest = []
    if 'index.notes.pb' in names:
        for r in records(z.read('index.notes.pb')):
            p = first(fdict(r), 2, b'').decode('utf-8', 'replace')
            if p in names and p not in used:
                rest.append(p)
    rest += sorted(n for n in names if n.startswith('notes/') and n not in used and n not in rest)
    for nf in rest:
        sf = 'search/' + nf[6:]
        out.append(dict(page_id=None, notes=nf, search=sf if sf in names else None, width=None, height=None, bg=None))
    return out


def read_strokes(blob, stats, shapes=False):
    heads = {}
    bodies = {}
    for r in records(blob):
        try:
            fs = fields(r)
        except ValueError:
            stats['bad_record'] += 1
            continue
        if not fs:
            continue
        f0, w0, v0 = fs[0]
        if f0 == 1 and len(fs) > 1 and w0 == 2:
            d = fdict(r)
            u = first(d, 1).decode('utf-8', 'replace')
            ver = first(fdict(first(d, 2, b'')), 1, 0) if first(d, 2) else 0
            if u not in heads or ver >= heads[u]['ver']:
                heads[u] = dict(ver=ver, deleted=first(d, 3, 0) == 1, clock=first(d, 9, 0))
        elif f0 == 7 and w0 == 2:
            sd = fdict(v0)
            u = first(sd, 1, b'').decode('utf-8', 'replace')
            ver = first(fdict(first(sd, 15)), 1, 0) if first(sd, 15) else 0
            if u not in bodies or ver >= bodies[u][0]:
                bodies[u] = (ver, sd)
        elif f0 == 1:
            stats['image'] += 1
        elif f0 == 20:
            stats['textbox'] += 1
        else:
            stats['other_record'] += 1
    out = []
    for u, (ver, sd) in bodies.items():
        h = heads.get(u, {})
        if first(sd, 14, 0) == 1 or h.get('deleted'):
            stats['deleted'] += 1
            continue
        try:
            sig, vals = parse_tpl(bvx(first(sd, 2, b'')))
        except (ValueError, IndexError, struct.error):
            stats['bad_geometry'] += 1
            continue
        rgba = [f32(x) for x in (first(fdict(first(sd, 4)), k) for k in (1, 2, 3, 4)) if x is not None] if first(sd, 4) else []
        color = [int(round(c * 255)) for c in rgba[:3]] if len(rgba) >= 3 else None
        off = vec2(first(sd, 6)) if first(sd, 6) else (0.0, 0.0)
        zk = fdict(first(sd, 7)) if first(sd, 7) else {}
        zk = fdict(first(zk, 1)) if first(zk, 1) else {}
        order = (first(zk, 1, 0), first(zk, 2, 0), first(zk, 3, 0), h.get('clock', 0))
        pen = 'highlighter' if first(sd, 5, 0) == 1 else 'pen'
        if sig == STROKE_SIG2:
            if pen == 'highlighter':
                stats['highlighter'] += 1
                continue
            raw = []
            if len(vals[2]) >= 3:
                raw.append(vals[2][:3])
            v3 = vals[3]
            flags = vals[1]
            if flags and (flags[0] & 4):
                # variant B: start record [x, y, w, e]; each segment = [x, y, w, x, y, w, e1, e2, e3]
                for i in range(0, len(v3) - 8, 9):
                    raw += [v3[i:i + 3], v3[i + 3:i + 6]]
                stats['strokes_v2b'] += 1
            else:
                raw += [v3[i:i + 3] for i in range(0, len(v3) - 2, 3)]
            if not raw:
                stats['empty'] += 1
                continue
            pts = [[round(x + off[0], 2), round(y + off[1], 2), i, 1.0] for i, (x, y, w) in enumerate(raw)]
            if off != (0.0, 0.0):
                stats['moved'] += 1
            stats['strokes_v2'] += 1
            out.append((order, dict(t0_ms=None, pen=pen, color=color, width=None, points=pts,
                                    widths=[round(w, 4) for _, _, w in raw], pressure='width')))
            continue
        if sig != STROKE_SIG:
            stats['other_geometry'] += 1
            continue
        segs = vals[4]
        if not segs or not vals[3]:
            shape = fdict(first(sd, 9)) if first(sd, 9) else {}
            if shape and first(shape, 1) is not None:
                # straight-line shape: {1: {1: {1:x,2:y}, 1: {1:x,2:y}}}
                pts = [vec2(p) for p in fdict(first(shape, 1)).get(1, [])]
                if shapes and len(pts) >= 2:
                    stats['shape_line_kept'] += 1
                    out.append((order, dict(t0_ms=None, pen='shape-line', color=color, width=vals[1],
                                            points=[[round(x + off[0], 2), round(y + off[1], 2), i, 1.0] for i, (x, y) in enumerate(pts)])))
                else:
                    stats['shape_line'] += 1
            elif shape:
                stats['shape_other'] += 1
            else:
                stats['empty'] += 1
            continue
        if pen == 'highlighter':
            stats['highlighter'] += 1
            continue
        sx, sy = vals[3][0]
        raw = [(sx, sy)] + [(s[0], s[1]) for s in segs] + [(segs[-1][2], segs[-1][3])]
        pts = [[round(x + off[0], 2), round(y + off[1], 2), i, 1.0] for i, (x, y) in enumerate(raw)]
        if off != (0.0, 0.0):
            stats['moved'] += 1
        if vals[6]:
            stats['extra_floats'] += 1
        out.append((order, dict(t0_ms=None, pen=pen, color=color, width=round(vals[1], 4), points=pts)))
    out.sort(key=lambda t: t[0])
    return [s for _, s in out]


def read(path):
    """-> the pages in notebook order: [{"width", "height", "strokes": [[[x, y], ...], ...]}], strokes in writing order."""
    z = zipfile.ZipFile(path)
    out = []
    for pg in read_pages(z):
        strokes = read_strokes(z.read(pg['notes']), collections.Counter())
        out.append(dict(width=pg['width'], height=pg['height'], strokes=[[p[:2] for p in s['points']] for s in strokes if s['pen'] == 'pen']))
    return out
