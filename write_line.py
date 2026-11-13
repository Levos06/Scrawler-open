"""Writes a line of text in the hand of one reference line, the way synthink.dev does it.

python write_line.py "Мороз и солнце; день чудесный!" out_line --ref my_line.npy --ref-text "what the reference line says"

The first argument is the text of one line, or a .txt file with one line of text per row. Formulas are not written here: for
pages with LaTeX use write_pages.py. Needs the extra weights: python download_weights.py --site

In the folder `out` (it must not exist yet):

- line.svg, line.png (300 dpi) and line.pdf: the chosen attempt of every line, one under another
- attempts.png: all attempts, the chosen one marked with a bar on the left
- strokes.json: every attempt as strokes, with its character error rate and its closeness to the reference

The recipe is in scrawler/site_model.py. The style comes from --ref (an .npy from make_ref.py). --ref-text is what is written
in that line; it sets how wide the writer writes. Without it the recogniser reads the reference, which is right in most
letters but not all, so give the text when you know it.
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from scrawler.common import DEFAULT_REF, DEVICE
from scrawler.render import _bezier, _sample, _smooth, _write_pdf
from scrawler.site_model import ATTEMPTS, SiteModel, to_strokes

PX = 90.0  # page px per line height at 150 dpi
GAP = 1.5  # distance between lines, in line heights
MARGIN = 0.6


def place(line, row):
    """A line [n, 3] -> strokes in page px, the top of the line at `row` line heights."""
    strokes = to_strokes(line)
    top = min(s[:, 1].min() for s in strokes)
    left = min(s[:, 0].min() for s in strokes)
    return [((s - [left, top]) + [MARGIN, MARGIN + row * GAP]) * PX for s in strokes]


def page_size(placed):
    points = np.concatenate([s for strokes in placed for s in strokes])
    return int(points[:, 0].max() + MARGIN * PX) + 1, int(points[:, 1].max() + MARGIN * PX) + 1


def write_svg(strokes, W, H, path, pen):
    width = pen * 150 / 72.0
    g = lambda p: f"{p[0]:.2f},{p[1]:.2f}"
    parts = []
    for stroke in strokes:
        s = _smooth(stroke, True)
        if len(s) == 1:
            parts.append(f"M{g(s[0])} L{g(s[0])}")
        else:
            parts.append(f"M{g(s[0])} " + " ".join(f"C{g(a)} {g(b)} {g(d)}" for _, a, b, d in _bezier(s)))
    with open(path, "w") as fh:
        fh.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W / 150 * 25.4:.1f}mm" height="{H / 150 * 25.4:.1f}mm">'
            + f'<path d="{" ".join(parts)}" fill="none" stroke="#141414" stroke-width="{width:.2f}" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )


def write_png(strokes, W, H, path, pen, marks=()):
    over, k = 4, 300 / 150 * 4
    image = Image.new("L", (int(W * k), int(H * k)), 255)
    draw = ImageDraw.Draw(image)
    width = max(1, int(round(pen / 72.0 * 300 * over)))
    for y0, y1 in marks:
        draw.rectangle([0.15 * PX * k, y0 * k, 0.25 * PX * k, y1 * k], fill=120)
    for stroke in strokes:
        s = _smooth(stroke, True)
        points = np.vstack([s[:1]] + [_sample(seg) for seg in _bezier(s)]) * k if len(s) > 1 else s * k
        if len(points) > 1:
            draw.line([tuple(p) for p in points], fill=20, width=width, joint="curve")
        for p in (points[0], points[-1]):
            draw.ellipse([p[0] - width / 2, p[1] - width / 2, p[0] + width / 2, p[1] + width / 2], fill=20)
    image.resize((int(W * 2), int(H * 2)), Image.LANCZOS).save(path, dpi=(300, 300))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", help="one line of text, or a .txt file with a line per row")
    ap.add_argument("out", help="output folder, must not exist")
    ap.add_argument("--ref", default=DEFAULT_REF, help="the style: one line of handwriting, [n, 3] .npy from make_ref.py")
    ap.add_argument("--ref-text", help="what is written in the reference line; read by the recogniser if not given")
    ap.add_argument("--attempts", type=int, default=ATTEMPTS, help="samples per line; the best one is kept")
    ap.add_argument("--seed", type=int, default=0, help="another seed gives another writing of the same text")
    ap.add_argument("--pen", type=float, default=1.1, help="pen width in points")
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists")
    if os.path.isfile(args.text):
        with open(args.text, encoding="utf-8") as fh:
            texts = [row.strip() for row in fh if row.strip()]
    else:
        texts = [args.text.strip()]

    model = SiteModel()
    reference = np.load(args.ref).astype(np.float32)
    ref_text = args.ref_text
    if ref_text is None and os.path.abspath(args.ref) == os.path.abspath(DEFAULT_REF):
        with open(os.path.splitext(DEFAULT_REF)[0] + ".json", encoding="utf-8") as fh:
            ref_text = json.load(fh)["text"]
    if ref_text is None:
        ref_text = model.read(reference)
        print(f'the reference reads as "{ref_text}"; pass --ref-text if that is wrong')
    ref = model.reference(reference, ref_text)
    print(f"device {DEVICE}, writer's spread {ref['spread']:.2f}")

    report, chosen, everything, marks = [], [], [], []
    for n, text in enumerate(texts):
        unknown = sorted(set(text) - set(model.clean(text)))
        if unknown:
            print(f"line {n + 1}: the model does not know {' '.join(unknown)}, dropped")
        started = time.time()
        try:
            result = model.write(text, ref, attempts=args.attempts, seed=args.seed + n)
        except ValueError as error:
            raise SystemExit(f"line {n + 1}: {error}")
        pick = result["pick"]
        print(
            f"line {n + 1}: {time.time() - started:.0f} s, error rate {' '.join(f'{c:.2f}' for c in result['cer'])}, "
            f"closeness {' '.join(f'{c:.2f}' for c in result['cos'])}, chosen {pick + 1}"
        )
        chosen.append(place(result["lines"][pick], n))
        for k, line in enumerate(result["lines"]):
            row = len(everything) + 0.4 * n
            everything.append(place(line, row))
            if k == pick:
                marks.append(((MARGIN + row * GAP) * PX, (MARGIN + row * GAP + 1) * PX))
        report.append(
            {
                "text": result["text"],
                "pick": pick,
                "cer": [round(float(c), 4) for c in result["cer"]],
                "cos": [round(float(c), 4) for c in result["cos"]],
                "attempts": [[[[round(float(x), 4), round(float(y), 4)] for x, y in s] for s in to_strokes(line)] for line in result["lines"]],
            }
        )

    os.makedirs(args.out)
    W, H = page_size(chosen)
    strokes = [s for line in chosen for s in line]
    write_svg(strokes, W, H, os.path.join(args.out, "line.svg"), args.pen)
    write_png(strokes, W, H, os.path.join(args.out, "line.png"), args.pen)
    _write_pdf([strokes], [], W, H, 150, os.path.join(args.out, "line.pdf"), args.pen, True)
    W, H = page_size(everything)
    write_png([s for line in everything for s in line], W, H, os.path.join(args.out, "attempts.png"), args.pen, marks)
    with open(os.path.join(args.out, "strokes.json"), "w", encoding="utf-8") as fh:
        json.dump({"reference_text": ref_text, "spread": round(ref["spread"], 4), "lines": report}, fh, ensure_ascii=False)
    print("written to", args.out)


if __name__ == "__main__":
    main()
