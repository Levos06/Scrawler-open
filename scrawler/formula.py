"""Formulas: MathJax says where every symbol stands, the glyph model draws the symbols, and thin marks are drawn as pen lines.

Coordinates inside this module: MathJax units x LAYOUT_SCALE, y down.
"""
import collections
import json
import math
import os
import subprocess
import unicodedata

import numpy as np
import torch

from .common import MATHJAX_DIR
from .glyph_model import GlyphModel, GlyphPicker

LAYOUT_SCALE = 0.1
X_HEIGHT = 450 * LAYOUT_SCALE
X_HEIGHT_EM = 0.442  # x-height of a composed formula in the units compose() returns
VARIANTS = 4
RETRIES = 16
POINT_STEP = 0.05

# fractions, radicals and arrows are pen lines over their layout box
LINE_MARKS = {"BAR", "√", "→", "←", "↑", "↓", "⇒", "⇐", "⇔", "↔", "VEC"}
# accents, slashes and the like, which the glyph model does not draw well
OWN_MARKS = {"/", "↔", "↦", "ˆ", "^", "˜", "~", " ̄", "¯", " ̃", " ̂", "‾", "̄", "̂", "̃", "̅"}
SUBSTITUTE_CODES = {"2A7E": "2265", "2A7D": "2264"}  # slanted >= and <= are drawn as the plain ones
# classes the glyph model has no row for, drawn as a similar shape
DONORS = {
    "O": "0", "κ": "k", "Σ": "∑",
    "≃": "≈", "∙": "⋅", "*": "∗", "⋯": "…", "⇓": "↓", "⟸": "←", "⇉": "→", "⎧": "{", "⎩": "{", "⎛": "(",
    "∩": "u", "Θ": "θ",
}  # fmt: skip
KEEP_SHAPE = set("=+−-<>≠±")  # a sheared '=' reads as '>'
VERTICAL_BARS = ("|", "∥", "‖", "∣")
DOTS = ("⋅", "·", "…", "...", "⋯", ".", ":")
DOT_ACCENTS = (" ̇", "˙", "̇")


def class_of(code):
    if code == "BAR":
        return "BAR"
    ch = chr(int(code, 16))
    if ch == "⃗":
        return "VEC"
    normal = unicodedata.normalize("NFKC", ch)
    return normal if normal.strip() else ch


def layout(formulas):
    """LaTeX -> [{ok, glyphs: [{c, bbox, node}], ...}] with MathJax under node."""
    lines = "\n".join(json.dumps(dict(id=i, tex=t), ensure_ascii=False) for i, t in enumerate(formulas)) + "\n"
    result = subprocess.run(
        ["node", "layout.mjs"], input=lines, capture_output=True, text=True, cwd=MATHJAX_DIR, env=dict(os.environ, LAYOUT_TEXT="1")
    )
    out = {}
    for line in result.stdout.splitlines():
        try:
            o = json.loads(line)
            out[o["id"]] = o
        except ValueError:
            pass
    return [out.get(i) for i in range(len(formulas))]


def _merge_radicals(glyphs):
    """A radical sign and its overbar -> one box."""
    out = []
    i = 0
    while i < len(glyphs):
        g = glyphs[i]
        if g["c"] == "221A" and i + 1 < len(glyphs) and glyphs[i + 1]["c"] == "BAR" and glyphs[i + 1].get("node") in ("msqrt", "mroot"):
            b = glyphs[i + 1]["bbox"]
            g = dict(c="221A", bbox=[min(g["bbox"][0], b[0]), min(g["bbox"][1], b[1]), max(g["bbox"][2], b[2]), max(g["bbox"][3], b[3])])
            i += 2
        else:
            i += 1
        out.append(g)
    return out


def _fix_limits(glyphs):
    """MathJax puts the lower limit of an integral under its slanted foot; handwriting puts it to the right."""
    for i, g in enumerate(glyphs):
        if g["c"] not in ("222B", "222E"):
            continue
        x0, y0, x1, y1 = g["bbox"]
        middle = (y0 + y1) / 2
        group = [h for h in glyphs[i + 1 : i + 4] if h["bbox"][0] < x1 and h["bbox"][1] > middle]
        if group:
            dx = x1 + 0.05 * (y1 - y0) - min(h["bbox"][0] for h in group)
            for h in group:
                h["bbox"] = [h["bbox"][0] + dx, h["bbox"][1], h["bbox"][2] + dx, h["bbox"][3]]
    return glyphs


def _segment(a, b, n=8):
    return [[a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t] for t in np.linspace(0, 1, n)]


def _polyline(points, rng, jitter, n=6):
    out = []
    for i in range(len(points) - 1):
        q = _segment(points[i], points[i + 1], n)
        out += q if not out else q[1:]
    return [[x + rng.normal(0, jitter), y + rng.normal(0, jitter), 0.5] for x, y in out]


def _line_mark(c, box, rng):
    """Pen-line strokes for a fraction bar, a radical or an arrow over a layout box."""
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = max(y1 - y0, 1e-6)
    ym = (y0 + y1) / 2
    xm = (x0 + x1) / 2
    j = 0.02 * X_HEIGHT
    tilt = rng.normal(0, 0.01) * w
    if c == "BAR":
        return [_polyline([[x0 - 0.03 * w, ym - tilt], [x1 + 0.03 * w, ym + tilt]], rng, j, 12)]
    if c == "√":
        tw = min(0.5 * h, 0.4 * w)
        return [_polyline([[x0, y0 + 0.6 * h], [x0 + 0.18 * tw, y0 + 0.52 * h], [x0 + 0.55 * tw, y1], [x0 + tw, y0], [x1, y0 + tilt]], rng, j, 6)]
    head = min(0.35 * w, max(0.9 * h, 0.25 * X_HEIGHT)) if c not in ("↑", "↓") else min(0.35 * h, max(0.9 * w, 0.25 * X_HEIGHT))
    if c == "VEC":
        return [_polyline([[x0, ym], [x1, ym]], rng, j, 8), _polyline([[x1 - head, ym - 0.5 * head], [x1, ym], [x1 - head, ym + 0.5 * head]], rng, j, 4)]
    if c in ("↑", "↓"):
        tip, sign = (y1, -1) if c == "↓" else (y0, 1)
        return [
            _polyline([[xm, y0], [xm, y1]], rng, j, 10),
            _polyline([[xm - 0.5 * head, tip + sign * head], [xm, tip], [xm + 0.5 * head, tip + sign * head]], rng, j, 4),
        ]
    double = c in ("⇒", "⇐", "⇔")
    d = 0.22 * h if double else 0.0
    strokes = []
    for yy in [ym - d, ym + d] if double else [ym]:
        strokes.append(_polyline([[x0 + (0.15 * head if c in "⇐⇔" else 0), yy], [x1 - (0.15 * head if c in "⇒⇔" else 0), yy]], rng, j, 10))
    if c in ("→", "⇒", "⇔", "↔"):
        strokes.append(_polyline([[x1 - head, ym - 0.55 * head - d], [x1, ym], [x1 - head, ym + 0.55 * head + d]], rng, j, 4))
    if c in ("←", "⇐", "⇔", "↔"):
        strokes.append(_polyline([[x0 + head, ym - 0.55 * head - d], [x0, ym], [x0 + head, ym + 0.55 * head + d]], rng, j, 4))
    return strokes


def _transform_glyph(strokes, mx, my, size, rng):
    """Small random shear, bend, proportions and pen speed, so that two instances of a symbol are not the same shape."""
    shear_x = rng.normal(0, 0.06)
    shear_y = rng.normal(0, 0.05)
    bend_v = rng.normal(0, 0.10)
    bend_h = rng.normal(0, 0.08)
    aspect = rng.normal(0, 0.04)
    ascender = rng.normal(0, 0.04)
    descender = rng.normal(0, 0.04)
    px, py = rng.normal(0, 0.02, 2)
    speed = 0.5
    pressure = math.exp(rng.normal(0, 0.27))
    all_points = np.concatenate([(p[:, :2] - [mx, my]) / size for p in strokes])
    mean_x2 = float(np.mean(all_points[:, 0] ** 2))
    mean_y2 = float(np.mean(all_points[:, 1] ** 2))
    out = []
    for p in strokes:
        q = p.copy()
        x = (p[:, 0] - mx) / size
        y = (p[:, 1] - my) / size
        if len(p) > 3:  # uneven pen speed: the points move along the same path
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.stack([x, y], 1), axis=0), axis=1))]
            if d[-1] > 0:
                u = np.linspace(0, 1, len(p))
                phase = rng.uniform(0, 6.283)
                w = (
                    u
                    + speed * np.sin(2 * np.pi * u + phase) / (2 * np.pi)
                    - speed * np.sin(phase) / (2 * np.pi) * (1 - u)
                    - speed * np.sin(2 * np.pi + phase) / (2 * np.pi) * u
                )
                w = np.clip(np.maximum.accumulate(w), 0, 1) * d[-1]
                x = np.interp(w, d, x)
                y = np.interp(w, d, y)
                if q.shape[1] > 3:
                    q[:, 3] = np.interp(w, d, p[:, 3])
        x = x + shear_x * y
        y = y + shear_y * x
        y = y + bend_v * (x**2 - mean_x2)
        x = x + bend_h * (y**2 - mean_y2)
        x = x * math.exp(aspect / 2)
        y = y * math.exp(-aspect / 2)
        y = np.where(y < -0.25, -0.25 + (y + 0.25) * (1 + ascender), np.where(y > 0.25, 0.25 + (y - 0.25) * (1 + descender), y))
        denominator = 1 + px * x + py * y
        x = x / denominator
        y = y / denominator
        q[:, 0] = x * size + mx
        q[:, 1] = y * size + my
        if q.shape[1] > 3:
            q[:, 3] = np.clip(q[:, 3] * pressure, 0, 1)
        out.append(q)
    return out


def _bounds(strokes):
    a = np.array([p[:2] for t in strokes for p in t], np.float32)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


class FormulaWriter:
    def __init__(self, text_model, reference, semb):
        self.text = text_model
        self.reference = np.asarray(reference, np.float32)
        self.semb = semb
        self.style_row = semb[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
        self.stats = collections.Counter()
        self._glyphs = None
        self._picker = None
        self._base_heights = {}

    @property
    def glyphs(self):
        if self._glyphs is None:
            self._glyphs = GlyphModel(self.stats)
        return self._glyphs

    @property
    def picker(self):
        if self._picker is None:
            self._picker = GlyphPicker(self.stats)
        return self._picker

    def _variants(self, classes, rng, k=1):
        n = len(classes) // max(k, 1)
        out = []
        for j in range(max(k, 1)):
            out += self.glyphs.sample(classes[j * n : (j + 1) * n], self.style_row, rng)
        return out

    def _write_words(self, words):
        """Words inside \\text{} -> {word: strokes}, written by the text model."""
        words = [w for w in dict.fromkeys(words) if w.strip()]
        if not words:
            return {}
        lines = [list(w) for w in words]
        style = torch.cat([self.text.style(self.semb)] * len(words), 0)
        semb = torch.cat([self.semb] * len(words), 0)
        durations = self.text.durations(lines, semb)
        out = {}
        for word, q in zip(words, self.text.sample(lines, durations, style)):
            q = np.asarray(q, np.float32)
            xy = np.cumsum(q[:, :2], 0) * 100.0
            strokes, current = [], []
            for k in range(len(q)):
                current.append([float(xy[k, 0]), float(xy[k, 1]), 0.5])
                if q[k, 2] > 0.5:
                    strokes.append(current)
                    current = []
            if current:
                strokes.append(current)
            out[word] = strokes
        return out

    def _compose_text(self, boxes):
        """Places words of \\text{} in their layout boxes, fitted by x-height."""
        s = LAYOUT_SCALE
        words, current = [], []
        for g in sorted(boxes, key=lambda g: g["bbox"][0]):
            if not g.get("ch", "").strip():
                if current:
                    words.append(current)
                    current = []
                continue
            if current and g["bbox"][0] - current[-1]["bbox"][2] > 50:
                words.append(current)
                current = []
            current.append(g)
        if current:
            words.append(current)
        written = self._write_words(["".join(g["ch"] for g in w) for w in words])
        out = []
        for word in words:
            strokes = written.get("".join(g["ch"] for g in word))
            if not strokes:
                continue
            a = np.array([p[:2] for t in strokes for p in t], np.float32)
            bx0 = min(g["bbox"][0] for g in word) * s
            bx1 = max(g["bbox"][2] for g in word) * s
            baseline = float(np.median([g["bbox"][3] for g in word])) * s
            box_height = float(np.median([g["bbox"][3] - g["bbox"][1] for g in word])) * s
            word_height = max(np.percentile(a[:, 1], 85) - np.percentile(a[:, 1], 15), 1e-6)
            k = min(1.1 * (bx1 - bx0) / max(np.ptp(a[:, 0]), 1e-6), box_height / word_height)
            bottom = np.percentile(a[:, 1], 85)
            for t in strokes:
                out.append([[float((p[0] - a[:, 0].min()) * k + bx0), float((p[1] - bottom) * k + baseline), float(p[2])] for p in t])
        return out

    def _own_marks(self, glyphs, rng):
        """Marks the glyph model does not draw -> pen lines. Returns the remaining glyphs and the strokes of the marks."""
        self.glyphs  # loaded here so that the order of random draws does not depend on the input
        s = LAYOUT_SCALE
        keep, out = [], []
        for g in glyphs:
            c = class_of(g["c"]) if g["c"] not in ("BAR", "TEXT") else g["c"]
            if g["c"] in SUBSTITUTE_CODES:
                keep.append(dict(g, c=SUBSTITUTE_CODES[g["c"]]))
                continue
            if c not in OWN_MARKS:
                keep.append(g)
                continue
            x0, y0, x1, y1 = [v * s for v in g["bbox"]]
            w = max(x1 - x0, 1e-6)
            h = max(y1 - y0, 1e-6)
            j = 0.015 * X_HEIGHT
            ym = (y0 + y1) / 2
            if c == "↔":
                out += _line_mark("↔", [x0, y0, x1, y1], rng)
            elif c == "↦":
                out += _line_mark("→", [x0, y0, x1, y1], rng) + [_polyline([[x0, ym - 0.4 * h], [x0, ym + 0.4 * h]], rng, j, 5)]
            elif c == "/":
                out.append(_polyline([[x0 + 0.15 * w, y1], [x1 - 0.15 * w, y0]], rng, j, 10))
            elif c in ("ˆ", "^", " ̂", "̂"):
                out.append(_polyline([[x0 + 0.1 * w, y1], [(x0 + x1) / 2, y0], [x1 - 0.1 * w, y1]], rng, j, 5))
            elif c in ("˜", "~", " ̃", "̃"):
                out.append([[float(x0 + t * w), float(ym - 0.35 * h * np.sin(2 * np.pi * t) + rng.normal(0, j)), 0.5] for t in np.linspace(0, 1, 14)])
            else:  # overline, macron
                out.append(_polyline([[x0, ym], [x1, ym]], rng, j, 10))
            self.stats["own_mark_" + c] += 1
        return keep, out

    def _load_base_heights(self, codes):
        new = [c for c in set(codes) if c not in self._base_heights and c not in ("BAR", "TEXT")]
        if new:
            for c, o in zip(new, layout([chr(int(c, 16)) for c in new])):
                self._base_heights[c] = (o["glyphs"][0]["bbox"][3] - o["glyphs"][0]["bbox"][1]) if o and o.get("ok") and o["glyphs"] else None

    def _script_factor(self, g):
        """Subscripts: MathJax shrinks them by 0.71, handwriting shrinks digits as much but letters less."""
        if g["c"] in ("BAR", "TEXT"):
            return 1.0
        height = self._base_heights.get(g["c"])
        if not height or (g["bbox"][3] - g["bbox"][1]) >= 0.85 * height:
            return 1.0
        ch = unicodedata.normalize("NFKC", chr(int(g["c"], 16)))
        if ch.isdigit():
            return 0.71
        if ch.isascii() and ch.isalpha():
            return 1.0
        return 0.92

    def _push_right(self, strokes, placed):
        """Moves a glyph right if it collides with a neighbour on the same line."""
        if not strokes or not placed:
            return
        a = np.concatenate([np.asarray(q)[:, :2] for q in strokes])
        bx0, by0, bx1, by1 = a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()
        shift = 0.0
        for px0, py0, px1, py1 in placed:
            overlap = min(by1, py1) - max(by0, py0)
            smaller = min(by1 - by0, py1 - py0)
            same_line = abs((py0 + py1) / 2 - (by0 + by1) / 2) < 0.3 * max(by1 - by0, py1 - py0)  # a subscript is not a collision
            if (py1 - py0) > 1.8 * (by1 - by0) and py0 <= by0 and py1 >= by1:  # a radical or a tall bracket encloses, it is no neighbour
                continue
            if same_line and smaller > 0.12 * X_HEIGHT and overlap > 0.5 * smaller and (px0 + px1) / 2 < (bx0 + bx1) / 2 and px1 + 0.35 * X_HEIGHT > bx0:
                shift = max(shift, px1 + 0.35 * X_HEIGHT - bx0)
        if 0 < shift < 2.0 * X_HEIGHT:
            for q in strokes:
                q[:, 0] += shift
            self.stats["pushed"] += 1

    def _compose_layout(self, glyphs, rng):
        s = LAYOUT_SCALE
        xh = X_HEIGHT
        strokes, missing = [], []
        merged = _merge_radicals(glyphs)
        texts = [g for g in merged if g["c"] == "TEXT"]
        glyphs = []
        for g in merged:
            if g["c"] == "TEXT":
                continue
            if g["c"] == "226A":  # << = two <
                b = g["bbox"]
                m = (b[0] + b[2]) / 2
                w = b[2] - b[0]
                glyphs += [dict(g, c="3C", bbox=[b[0], b[1], m + 0.08 * w, b[3]]), dict(g, c="3C", bbox=[m - 0.08 * w, b[1], b[2], b[3]])]
            else:
                glyphs.append(g)
        self._load_base_heights([g["c"] for g in glyphs])
        if texts:
            strokes += self._compose_text(texts)
        if not glyphs:
            return strokes, missing

        class_for_net = lambda c: DONORS.get(c, c)
        x_min = min(g["bbox"][0] for g in glyphs)
        n = len(glyphs)
        top = min(g["bbox"][1] for g in glyphs)
        height = max(max(g["bbox"][3] for g in glyphs) - top, 1e-6)
        variants = self._variants([class_for_net(class_of(g["c"])) for g in glyphs] * VARIANTS, rng, VARIANTS)
        shape_rng = np.random.default_rng(int(rng.integers(1 << 31)))
        gap = base = 0.0
        placed = []
        last = 0

        for gi, g in enumerate(glyphs):
            if len(strokes) > last:
                placed.append(_bounds(strokes[last:]))
                last = len(strokes)
            upside_down = class_of(g["c"]) == "∩"  # ∩ is a u turned over
            c = class_of(g["c"])
            if c in DONORS:
                self.stats["donor_" + c] += 1
                c = DONORS[c]

            if c in DOTS:  # the net draws a dot as a ring
                b = [v * s for v in g["bbox"]]
                r = 0.05 * xh
                if c == ":":
                    dots = [((b[0] + b[2]) / 2, b[1] + 0.15 * (b[3] - b[1])), ((b[0] + b[2]) / 2, b[3] - 0.15 * (b[3] - b[1]))]
                else:
                    count = 1 if c in ("⋅", "·", ".") else 3
                    cy = (b[1] + b[3]) / 2 if c != "." else b[3] - r
                    dots = [(b[0] + (k + 0.5) * (b[2] - b[0]) / count, cy) for k in range(count)]
                for cx, cy in dots:
                    strokes.append([[float(cx - r), float(cy), 0.5], [float(cx), float(cy + r * 0.6), 0.5], [float(cx + r), float(cy), 0.5]])
                self.stats["dot_mark"] += 1
                continue
            if c in DOT_ACCENTS:
                b = [v * s for v in g["bbox"]]
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                r = 0.05 * xh
                strokes.append([[float(cx - r), float(cy), 0.5], [float(cx), float(cy + r * 0.6), 0.5], [float(cx + r), float(cy), 0.5]])
                self.stats["dot_mark"] += 1
                continue
            if c in VERTICAL_BARS:  # the net draws | as / ( ) ? 7
                b = [v * s for v in g["bbox"]]
                xm = (b[0] + b[2]) / 2
                offsets = [0.0] if c in ("|", "∣") else [-0.22 * (b[2] - b[0]), 0.22 * (b[2] - b[0])]
                bars = [np.asarray(_polyline([[xm + d, b[1]], [xm + d, b[3]]], rng, 0.015 * xh, 10), np.float64) for d in offsets]
                self._push_right(bars, placed)
                strokes += [t.tolist() for t in bars]
                self.stats["vbar_pen"] += 1
                continue
            if c in LINE_MARKS:
                strokes += _line_mark(c, [v * s for v in g["bbox"]], rng)
                self.stats["line_mark"] += 1
                continue

            relative_height = (g["bbox"][3] - g["bbox"][1]) / height
            q = None
            if c not in ("[", "]"):  # square brackets always go through the retry below
                k = self.picker.pick([variants[k * n + gi] for k in range(VARIANTS)], c, relative_height)
                q = variants[k * n + gi] if k is not None else None
                if q is not None and not q["strokes"]:
                    q = None
            if q is None:
                if not variants[gi]["strokes"]:  # the net does not know this class
                    missing.append(c)
                    continue
                more = self._variants([class_for_net(class_of(g["c"]))] * RETRIES, rng, 1)
                k = self.picker.pick(more, c, relative_height)
                if k is not None:
                    q = more[k]
                    self.stats["retry_ok"] += 1
                else:
                    everything = more + [variants[j * n + gi] for j in range(VARIANTS)]
                    k = self.picker.pick(everything, c, relative_height, force=True)
                    q = everything[k if k is not None else 0]
                    self.stats["retry_forced"] += 1

            b = [v * s for v in g["bbox"]]
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            points = [np.asarray(t, np.float32) for t in q["strokes"]]
            all_points = np.concatenate([p[:, :2] for p in points])
            x0, y0 = all_points.min(0)
            x1, y1 = all_points.max(0)
            qw = max(x1 - x0, 1e-3)
            qh = max(y1 - y0, 1e-3)
            if bw > 2.5 * bh:
                kx = bw / qw
                ky = min(kx, bh / qh * 1.5)
            elif bh > 2.2 * bw and c in "()[]{}|∫∮∑∏":
                ky = bh / qh
                kx = min(ky, 1.6 * bw / qw)
            else:
                ky = bh / qh
                kx = ky
            f = 1.0 + rng.normal(0, 0.06)
            kx *= f
            ky *= f
            angle = math.radians(rng.normal(0, 2.5))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            drift = 0.02 * xh * math.sin((b[0] - x_min * s) / (6 * xh) + rng.uniform(0, 6.28))
            cx = (b[0] + b[2]) / 2 + rng.normal(0, 0.06 * xh)
            cy = (b[1] + b[3]) / 2 + rng.normal(0, 0.06 * xh) + drift
            if c not in ("(", ")", "[", "]", "{", "}", "|") and qw * kx > 1.3 * bw and bw > 0:  # much wider than its box: crowds the neighbours
                f2 = 1.3 * bw / (qw * kx)
                kx *= f2
                ky *= f2
                self.stats["narrowed"] += 1
            if class_of(g["c"]) == "Θ":  # capital Theta from a lowercase one
                kx *= 1.35
            fs = self._script_factor(g)
            if fs != 1.0:
                kx *= fs
                ky *= fs
                cx = b[0] + (cx - b[0]) * fs
                self.stats["script_scaled"] += 1
            mx = (x0 + x1) / 2
            my = (y0 + y1) / 2
            transformed = _transform_glyph(points, mx, my, max(qw, qh), shape_rng)
            if c not in KEEP_SHAPE:  # the random numbers are drawn either way
                points = transformed
            gap = 0.7 * gap + math.sqrt(1 - 0.49) * shape_rng.normal(0, 0.1)
            base = 0.8 * base + 0.6 * shape_rng.normal(0, 0.08)
            upright = int(g["c"], 16) < 128 and chr(int(g["c"], 16)).isalpha()  # letters of cos, sin: no gaps inside the word
            cx += (0 if upright else gap) * min(bh, 1.5 * xh)
            cy += base * min(bh, 1.5 * xh)
            if upside_down:
                points = [np.concatenate([2 * mx - p[:, :1], 2 * my - p[:, 1:2], p[:, 2:]], 1) for p in points]
            placed_strokes = []
            for p in points:
                u = (p[:, 0] - mx) * kx
                v = (p[:, 1] - my) * ky
                placed_strokes.append(np.stack([cx + u * cos_a - v * sin_a, cy + u * sin_a + v * cos_a, p[:, 3]], 1))
            self._push_right(placed_strokes, placed)
            strokes += [t.tolist() for t in placed_strokes]
        return strokes, missing

    def compose(self, tex, rng):
        """LaTeX -> (strokes, things it could not write). Strokes are None if the formula failed."""
        tex = tex.strip().strip("$").strip()
        try:
            o = layout([tex])[0]
            if not o or not o.get("ok"):
                return None, ["LAYOUT"]
            glyphs, extra = self._own_marks(_fix_limits(o["glyphs"]), rng)
            strokes, missing = self._compose_layout(glyphs, rng)
            strokes += extra
        except Exception as e:
            return None, ["ERROR " + repr(e)[:80]]
        if not strokes:
            return None, missing or ["EMPTY"]
        scale = LAYOUT_SCALE * 1000.0
        return [[[p[0] / scale, p[1] / scale] for p in stroke] for stroke in strokes], missing

    def mixed_line(self, seq, tokens, durations, rng):
        """The text model wrote the whole line, `$...$` included. Its points of each formula are replaced by the composed formula,
        and the text after it moves right. -> (line [n, 3], things it could not write, formulas replaced); line None on failure."""
        q = np.asarray(seq, np.float32)
        char_of_point = np.repeat(np.arange(len(tokens)), [max(1, int(x)) for x in durations])
        if len(char_of_point) != len(q):
            return None, ["PLAN"], 0
        a_ = np.cumsum(q[:, :2], 0)
        pen = q[:, 2].copy()
        spans = _dollar_spans(tokens)
        if not spans:
            return q, [], 0
        in_formula = np.zeros(len(q), bool)
        for a, b in spans:
            in_formula |= (char_of_point >= a) & (char_of_point < b)
        is_text = ~in_formula & np.array([not tokens[c].isspace() for c in char_of_point])
        if is_text.sum() < 10:
            return None, ["NOTEXT"], 0
        y = a_[is_text, 1]
        baseline = float(np.percentile(y, 85))
        xh = max(float(np.percentile(y, 85) - np.percentile(y, 15)), 1e-3)
        gap = 0.6 * xh
        k = xh / X_HEIGHT_EM
        pieces = []
        shift = 0.0
        missing = []
        prev = 0
        replaced = 0
        for a, b in spans:
            ix = np.where((char_of_point >= a) & (char_of_point < b))[0]
            before = np.arange(prev, ix[0])
            pieces.append((a_[before] + [shift, 0], pen[before]))
            x_left = (a_[before, 0].max() + shift) if len(before) else a_[ix, 0].min() + shift
            strokes, ms = self.compose("".join(tokens[a:b]).strip("$"), rng)
            missing += ms
            if not strokes:  # keep the text model's own points of the span
                pieces.append((a_[ix] + [shift, 0], pen[ix]))
                prev = ix[-1] + 1
                continue
            f = to_points(strokes, k)
            f[:, 0] += x_left + gap - f[:, 0].min()
            f[:, 1] += baseline
            if len(before):
                pieces[-1][1][-1] = 1.0  # lift the pen before the formula
            pieces.append((f[:, :2], f[:, 2]))
            replaced += 1
            after = np.arange(ix[-1] + 1, len(q))
            x_right = a_[after, 0].min() if len(after) else None
            if x_right is not None:
                shift = f[:, 0].max() + gap - x_right
            prev = ix[-1] + 1
        rest = np.arange(prev, len(q))
        pieces.append((a_[rest] + [shift, 0], pen[rest]))
        points = np.concatenate([p for p, _ in pieces if len(p)])
        pens = np.concatenate([p for _, p in pieces if len(p)])
        pens[-1] = 1.0
        d = np.zeros((len(points), 3), np.float32)
        d[1:, :2] = points[1:] - points[:-1]
        d[0, :2] = 0
        d[:, 2] = pens
        return d, missing, replaced


def _dollar_spans(tokens):
    """Character spans of $...$ and $$...$$."""
    out, start, k = [], None, 0
    while k < len(tokens):
        if tokens[k] == "$":
            n = 2 if k + 1 < len(tokens) and tokens[k + 1] == "$" else 1
            if start is None:
                start = k
            else:
                out.append((start, k + n))
                start = None
            k += n
            continue
        k += 1
    return out


def to_points(strokes, k):
    """Composed strokes, scaled by k -> absolute points [n, 3]: x, y, pen_up, every POINT_STEP along the path."""
    out = []
    for stroke in strokes:
        p = np.asarray(stroke, np.float32)[:, :2] * k
        if len(p) == 1:
            out.append(np.array([[p[0, 0], p[0, 1], 1.0]], np.float32))
            continue
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        s = np.r_[np.arange(0, cum[-1], POINT_STEP), cum[-1]] if cum[-1] > 1e-6 else np.array([0.0, cum[-1]])
        r = np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1]), np.zeros(len(s))], 1).astype(np.float32)
        r[-1, 2] = 1
        out.append(r)
    return np.concatenate(out)
