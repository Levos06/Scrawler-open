"""Notewise notebooks (.notewise, iPad) -> pen strokes of every page.

The file is only read as data: zipfile in memory and the protobuf wire decoder of goodnotes.py. The format has no public
description; these notes are reverse-engineered from real notebooks.
The notes below come from a fuller converter: they also describe fields, flags and options that this reader does not use.

- zip with two kinds of entries, each the base64 text of ONE plain protobuf message (no compression inside):
  note = {1: notebook id, 2: title, 4 repeated: page ids in notebook order, 7: ?},
  page/<page id> = {1: page id, 2: page index (absent = 0), 4 repeated: element, 6: paper {1: {1: ?, 3: width,
  4: height} in pixels (2480 x 3508 = A4 at 300 dpi; the ink coordinates live in that space), 2: {1: '#RRGGBB',
  2: alpha} background, 7, 8: template}, 7: double sort key (1024 * (index + 1)), 9, 10: ?}.
- element = {1: id, 2: unix ms, 3: affine transform as a row-major 3x3 float matrix with fields 1..9
  ([1 2 3; 4 5 6; 7 8 9], absent = identity entry; 3/6 = tx/ty of a lasso move, only translations seen),
  4: stroke | 5: highlighter | 6: shape | 15: other tool, 10: running index inside a pasted group}.
  stroke = {1: pen kind (1 = pressure pen with per-point width, 2 = constant-width pen), 2: nominal width,
  3: colour {1: '#RRGGBB', 2: alpha}, 4: packed float32 x, 5: packed float32 y, 6: packed float32 rendered width
  per point, 7: 1, 9: ? (12 strokes)}.
  highlighter = {1: width, 2: colour (alpha 0.25), 3/4: packed x/y, 5: 1} (dropped unless --highlighter).
  shape = {1: line colour, 2: width, 4: {4: straight line {1: {1: x, 2: y}, 2: {1: x, 2: y}}}, 5, 6: fill colour}:
  no hand trajectory, dropped unless --shapes (then written as a 2-point pen='shape-line').
  field 15 (one element, kind 12, 13 points): unknown tool, dropped.
- There is NO per-point clock. Element field 2 is the time the stroke was committed (pen-up), not pen-down:
  consecutive strokes are often closer in time than the earlier one could have been drawn, never than the later
  one. The lower envelope of the gaps grows by about 1.8-2 ms per point of the later stroke, so the samples are
  roughly uniform in time, but that rate is an estimate, so t_ms = point index and the page is tagged
  time:"ordinal"; t0_ms = the commit time (tag t0:"pen_up").
- There is no raw force either. The per-point rendered width of pen kind 1 tapers at both ends of a stroke and does
  not thin with speed, i.e. it is pressure-driven. pressure = rendered width / nominal width (mostly 0.2..1.0,
  rarely up to about 1.5), the widths are kept as "widths", and the page is tagged pressure:"width".
  Pen kind 2 has a constant width: pressure 1.0 and the stroke is tagged pressure:"none".
- Copy-paste: the pasted elements get fresh, consecutive commit times (1 ms apart) and a transform; the copied
  raw points are identical to the original's. Elements in such runs (>= 3 elements, steps <= 1 ms) are tagged
  pasted:true (their t0_ms is the paste time); a stroke whose raw points repeat an earlier stroke of the notebook
  is dropped unless --keep-copies (the ink would otherwise be in the data twice).
- Erasing removes elements: no deletion flag was found, every element in the file is live.
- Elements are stored in commit order; strokes are ordered by (commit time, file position).
"""
import sys
import os
import json
import struct
import base64
import hashlib
import zipfile
import collections

from goodnotes import fdict, first


def f32(v, default=0.0):
    return struct.unpack('<f', v)[0] if v is not None and len(v) == 4 else default


def dbl(v, default=0.0):
    return struct.unpack('<d', v)[0] if v is not None and len(v) == 8 else default


def floats(b):
    n = len(b) // 4
    return list(struct.unpack('<%df' % n, b[:4 * n])) if n else []


def s(v):
    return v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else v


def colour(b):
    """{1: '#RRGGBB', 2: alpha} -> ([r, g, b], alpha 0..1)."""
    d = fdict(b) if b else {}
    hx = s(first(d, 1, b'#000000')).lstrip('#')
    try:
        rgb = [int(hx[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        rgb = None
    return rgb, round(f32(first(d, 2), 1.0), 4)


def b64msg(z, name):
    return fdict(base64.b64decode(z.read(name)))


def read_note(z):
    """-> (title, [page entry names] in notebook order)."""
    names = set(z.namelist())
    note = b64msg(z, 'note') if 'note' in names else {}
    order = ['page/' + s(p) for p in note.get(4, [])]
    order = [n for n in order if n in names]
    # pages missing from the list go last, by their own index
    rest = sorted(n for n in names if n.startswith('page/') and n not in order)
    if rest:
        keyed = []
        for n in rest:
            pg = b64msg(z, n)
            keyed.append((dbl(first(pg, 7)), first(pg, 2, 0), n))
        order += [n for _, _, n in sorted(keyed)]
    return s(first(note, 2, b'')), order


IDENTITY = {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0, 6: 0.0}


def transform(b):
    d = fdict(b) if b else {}
    return tuple(f32(first(d, k), IDENTITY[k]) for k in (1, 2, 3, 4, 5, 6))


def apply_tf(tf, x, y):
    a, b, c, d, e, f = tf
    return a * x + b * y + c, d * x + e * y + f


def pasted_ids(els):
    """element positions committed in runs of >= 3 with steps <= 1 ms (copy-paste, not handwriting)."""
    ts = sorted((t, i) for i, t in enumerate(e['t'] for e in els))
    out = set()
    run = []
    for t, i in ts:
        if run and t - run[-1][0] > 1:
            if len(run) >= 3:
                out.update(j for _, j in run)
            run = []
        run.append((t, i))
    if len(run) >= 3:
        out.update(j for _, j in run)
    return out


def read_page(pg, stats, seen, shapes=False, highlighter=False, keep_copies=False):
    els = []
    for k, v in enumerate(pg.get(4, [])):
        e = fdict(v)
        els.append(dict(k=k, e=e, t=first(e, 2, 0)))
    pasted = pasted_ids(els)
    out = []
    for i, el in enumerate(els):
        e = el['e']
        tf = transform(first(e, 3))
        moved = tf != (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        if 4 in e:
            st = fdict(first(e, 4))
            kind = first(st, 1, 0)
            nominal = f32(first(st, 2), 0.0) if isinstance(first(st, 2), bytes) else float(first(st, 2, 0))
            rgb, alpha = colour(first(st, 3))
            xb, yb = first(st, 4, b''), first(st, 5, b'')
            xs, ys, ws = floats(xb), floats(yb), floats(first(st, 6, b''))
            if not xs or len(xs) != len(ys):
                stats['bad_points'] += 1
                continue
            if len(ws) != len(xs):
                ws = [nominal] * len(xs)
            h = hashlib.md5(xb + b'|' + yb).hexdigest()
            if h in seen:
                stats['copy_of_earlier_stroke'] += 1
                if not keep_copies:
                    continue
            seen.add(h)
            if kind not in (1, 2):
                stats['pen_kind_%d' % kind] += 1
            if 9 in st:
                stats['stroke_field9'] += 1
            const = kind == 2 or nominal <= 0
            pts = []
            for j, (x, y, w) in enumerate(zip(xs, ys, ws)):
                X, Y = apply_tf(tf, x, y)
                p = 1.0 if const else w / nominal
                pts.append([round(X, 2), round(Y, 2), j, round(p, 4)])
            sk = dict(t0_ms=el['t'], pen='pen', color=rgb, width=round(nominal, 3), points=pts,
                      widths=[round(w, 3) for w in ws], brush=kind, alpha=alpha)
            if const:
                sk['pressure'] = 'none'
                stats['strokes_const_width'] += 1
                stats['points_const_width'] += len(pts)
            else:
                stats['strokes_width_pressure'] += 1
                stats['points_width_pressure'] += len(pts)
            if i in pasted:
                sk['pasted'] = True
                stats['pasted'] += 1
            if moved:
                stats['moved'] += 1
            out.append((el['t'], el['k'], sk))
        elif 5 in e:
            hl = fdict(first(e, 5))
            xs, ys = floats(first(hl, 3, b'')), floats(first(hl, 4, b''))
            if not highlighter or not xs:
                stats['highlighter'] += 1
                continue
            rgb, alpha = colour(first(hl, 2))
            pts = [[round(X, 2), round(Y, 2), j, 1.0] for j, (X, Y) in enumerate(apply_tf(tf, x, y) for x, y in zip(xs, ys))]
            out.append((el['t'], el['k'], dict(t0_ms=el['t'], pen='highlighter', color=rgb, width=first(hl, 1),
                                               points=pts, alpha=alpha, pressure='none')))
            stats['highlighter_kept'] += 1
        elif 6 in e:
            sh = fdict(first(e, 6))
            geo = fdict(first(sh, 4, b''))
            if 4 not in geo:
                stats['shape_other'] += 1
                continue
            ln = fdict(first(geo, 4))
            a = fdict(first(ln, 1, b''))
            b = fdict(first(ln, 2, b''))
            if not shapes:
                stats['shape_line'] += 1
                continue
            rgb, alpha = colour(first(sh, 1))
            p1 = apply_tf(tf, f32(first(a, 1)), f32(first(a, 2)))
            p2 = apply_tf(tf, f32(first(b, 1)), f32(first(b, 2)))
            out.append((el['t'], el['k'], dict(t0_ms=el['t'], pen='shape-line', color=rgb, width=first(sh, 2),
                                               points=[[round(p1[0], 2), round(p1[1], 2), 0, 1.0],
                                                       [round(p2[0], 2), round(p2[1], 2), 1, 1.0]],
                                               alpha=alpha, pressure='none')))
            stats['shape_line_kept'] += 1
        else:
            stats['other_element_' + '_'.join(str(k) for k in sorted(e) if k not in (1, 2, 3, 10))] += 1
    out.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in out]


def read(path):
    """-> the pages in notebook order: [{"width", "height", "strokes": [[[x, y], ...], ...]}], strokes in writing order."""
    z = zipfile.ZipFile(path)
    _, order = read_note(z)
    seen = set()
    out = []
    for name in order:
        pg = b64msg(z, name)
        strokes = read_page(pg, collections.Counter(), seen)
        size = fdict(first(fdict(first(pg, 6, b'')), 1, b''))
        out.append(dict(width=first(size, 3), height=first(size, 4), strokes=[[p[:2] for p in k['points']] for k in strokes if k['pen'] == 'pen']))
    return out
