# -*- coding: utf-8 -*-
"""Your own reference line for write_pages.py: pen strokes of ONE handwritten line -> the [n,3] array the models take as the style
(dx, dy, pen_up; line height = 1, points every 0.05 of it along the pen path, as in training).
  input : JSON, a list of strokes, each a list of points [x, y] in any units, y pointing down, in writing order;
          or the JSON written by trace/scrawler_trace.py (a photo of your line)
  python make_ref.py my_line.json my_line.npy
then python write_pages.py notes.txt out --ref my_line.npy
Write a line of ordinary prose of 30-50 characters: the style is taken from this one line only."""
import sys, json, numpy as np
STEP = 0.05
def to_ref(strokes):
    S = [np.asarray(s, np.float64)[:, :2] for s in strokes if len(s)]
    if not S: raise SystemExit('no strokes in the input')
    A = np.concatenate(S); o = A.min(0); h = float(A[:, 1].max() - o[1])
    if h <= 0: raise SystemExit('the line has zero height')
    out = []
    for p in S:
        p = (p - o) / h
        if len(p) == 1: out.append(np.array([[p[0, 0], p[0, 1], 1.0]])); continue
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        u = np.r_[np.arange(0, cum[-1], STEP), cum[-1]] if cum[-1] > 1e-6 else np.array([0.0, cum[-1]])
        r = np.stack([np.interp(u, cum, p[:, 0]), np.interp(u, cum, p[:, 1]), np.zeros(len(u))], 1); r[-1, 2] = 1; out.append(r)
    a = np.concatenate(out); d = a.copy(); d[1:, :2] = a[1:, :2] - a[:-1, :2]; d[0, :2] = 0
    return d.astype(np.float32)
if __name__ == '__main__':
    J = json.load(open(sys.argv[1]))
    if isinstance(J, dict):   # the output of trace/scrawler_trace.py: points [x, y, pen_up, pressure]
        J_, cur = [], []
        for x, y, pen, _ in J['points']:
            cur.append([x, y])
            if pen: J_.append(cur); cur = []
        J = J_ + ([cur] if cur else [])
    q = to_ref(J); np.save(sys.argv[2], q); print(f'{sys.argv[2]}: {len(q)} points, {int(q[:, 2].sum())} strokes')
