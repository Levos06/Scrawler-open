"""Strokes -> vector PDF, SVG and a 300 dpi PNG.

Every stroke is a smooth curve: Catmull-Rom through the model's points, written as cubic Beziers. The PDF is written by hand, so
no extra library is needed. Strokes are lightly smoothed first ([1 2 1]/4 once): the models' points carry a small jitter that a
vector line would show.

A document: {"W": px, "H": px, "dpi": 150, "ruling": [y px, ...], "pages": [[stroke, ...], ...]}, stroke = [[x, y], ...], y down.
"""
import os
import zlib

import numpy as np
from PIL import Image, ImageDraw


def _smooth(stroke, enabled):
    s = np.asarray(stroke, np.float64)
    if not enabled or len(s) < 5:
        return s
    t = s.copy()
    t[1:-1] = 0.25 * s[:-2] + 0.5 * s[1:-1] + 0.25 * s[2:]
    return t


def _bezier(s):
    """Points [n, 2] -> cubic segments (p0, c1, c2, p1)."""
    n = len(s)
    if n < 2:
        return []
    P = np.vstack([s[:1], s, s[-1:]])
    out = []
    for i in range(1, n):
        p0, p1 = P[i], P[i + 1]
        out.append((p0, p0 + (P[i + 1] - P[i - 1]) / 6, p1 - (P[i + 2] - P[i]) / 6, p1))
    return out


def _sample(segment, k=8):
    p0, c1, c2, p1 = segment
    t = np.linspace(0, 1, k + 1)[1:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * p1


def _write_pdf(pages, ruling, W, H, dpi, path, pen, smooth):
    k = 72.0 / dpi
    PW, PH = W * k, H * k
    f = lambda x, y: f"{x * k:.2f} {PH - y * k:.2f}"
    objects = {1: b"<< /Type /Catalog /Pages 2 0 R >>"}
    kids = []
    for i, strokes in enumerate(pages):
        content = ["0.86 G 0.5 w"] + [f"{f(0, y)} m {f(W, y)} l S" for y in ruling]
        content.append(f"0.08 G {pen:.2f} w 1 J 1 j")
        for stroke in strokes:
            s = _smooth(stroke, smooth)
            if len(s) == 1:
                content.append(f"{f(*s[0])} m {f(*s[0])} l S")
                continue
            content.append(f"{f(*s[0])} m " + " ".join(f"{f(*a)} {f(*b)} {f(*d)} c" for _, a, b, d in _bezier(s)) + " S")
        data = zlib.compress("\n".join(content).encode())
        page, stream = 3 + 2 * i, 4 + 2 * i
        kids.append(page)
        objects[page] = f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PW:.2f} {PH:.2f}] /Contents {stream} 0 R >>".encode()
        objects[stream] = f"<< /Length {len(data)} /Filter /FlateDecode >>\nstream\n".encode() + data + b"\nendstream"
    objects[2] = f'<< /Type /Pages /Kids [{" ".join(f"{p} 0 R" for p in kids)}] /Count {len(kids)} >>'.encode()
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for n in sorted(objects):
        offsets[n] = len(out)
        out += f"{n} 0 obj\n".encode() + objects[n] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offsets[n]:010d} 00000 n \n".encode() for n in sorted(objects))
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as fh:
        fh.write(out)


def _write_svg(strokes, ruling, W, H, dpi, path, pen, smooth):
    width = pen * dpi / 72.0
    g = lambda p: f"{p[0]:.2f},{p[1]:.2f}"
    parts = []
    for stroke in strokes:
        s = _smooth(stroke, smooth)
        if len(s) == 1:
            parts.append(f"M{g(s[0])} L{g(s[0])}")
        else:
            parts.append(f"M{g(s[0])} " + " ".join(f"C{g(a)} {g(b)} {g(d)}" for _, a, b, d in _bezier(s)))
    lines = "".join(f'<line x1="0" y1="{y:.1f}" x2="{W}" y2="{y:.1f}" stroke="#dcdcdc" stroke-width="1"/>' for y in ruling)
    with open(path, "w") as fh:
        fh.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="210mm" height="297mm"><rect width="100%" height="100%" fill="#fff"/>'
            + lines
            + f'<path d="{" ".join(parts)}" fill="none" stroke="#141414" stroke-width="{width:.2f}" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )


def _write_png(strokes, ruling, W, H, dpi, path, pen, out_dpi, smooth):
    oversample = 4
    k = out_dpi / dpi * oversample
    image = Image.new("L", (int(W * k), int(H * k)), 255)
    draw = ImageDraw.Draw(image)
    width = max(1, int(round(pen / 72.0 * out_dpi * oversample)))
    for y in ruling:
        draw.line([(0, y * k), (W * k, y * k)], fill=222, width=oversample)
    for stroke in strokes:
        s = _smooth(stroke, smooth)
        points = np.vstack([s[:1]] + [_sample(seg) for seg in _bezier(s)]) * k if len(s) > 1 else s * k
        if len(points) > 1:
            draw.line([tuple(p) for p in points], fill=20, width=width, joint="curve")
        for p in (points[0], points[-1]):
            draw.ellipse([p[0] - width / 2, p[1] - width / 2, p[0] + width / 2, p[1] + width / 2], fill=20)
    image.resize((int(W * out_dpi / dpi), int(H * out_dpi / dpi)), Image.LANCZOS).save(path, dpi=(out_dpi, out_dpi))


def render(doc, out, pen=1.1, png_dpi=300, smooth=True):
    W, H, dpi, ruling = doc["W"], doc["H"], doc.get("dpi", 150), doc["ruling"]
    _write_pdf(doc["pages"], ruling, W, H, dpi, os.path.join(out, "pages.pdf"), pen, smooth)
    for i, strokes in enumerate(doc["pages"]):
        _write_svg(strokes, ruling, W, H, dpi, os.path.join(out, f"page_{i + 1:02d}.svg"), pen, smooth)
        _write_png(strokes, ruling, W, H, dpi, os.path.join(out, f"page_{i + 1:02d}.png"), pen, png_dpi, smooth)
