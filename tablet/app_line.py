"""One handwritten line out of a notebook of a tablet app -> the strokes JSON that make_ref.py takes.

python tablet/app_line.py notes.goodnotes                                   # what is in the file: pages and the lines found
python tablet/app_line.py notes.goodnotes my_line.json --page 3 --line 2 --svg check.svg
python make_ref.py my_line.json my_line.npy

Reads GoodNotes (.goodnotes), Notewise (.notewise), Huawei Notes (.hinote), Kilonotes (.zip export) and the page JSON that
tablet/samsung/sdocx2ink.mjs writes from a Samsung Notes .sdocx. See tablet/README.md for how to get the file out of each app.

A page is split into lines by where the ink is densest from top to bottom. That works on ordinary rows of text and can fail on
crowded pages, formulas and drawings, so look at the --svg picture: the page with the lines numbered and the chosen one in red.
The surest way is a new page with one line on it; then no --line is needed.
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_pages(path):
    """-> (app name, pages); a page = {"width", "height", "strokes": [[[x, y], ...], ...]}, y pointing down."""
    low = path.lower()
    if low.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            page = json.load(fh)
        if not isinstance(page, dict) or "strokes" not in page:
            raise SystemExit("this JSON is not a page written by tablet/samsung/sdocx2ink.mjs")
        strokes = [[p[:2] for p in s["points"]] for s in page["strokes"] if s.get("points")]
        return {"sdocx": "Samsung Notes"}.get(page.get("source"), "page JSON"), [dict(width=page.get("width"), height=page.get("height"), strokes=strokes)]
    if low.endswith(".sdocx"):
        raise SystemExit("Samsung Notes: first run  node tablet/samsung/sdocx2ink.mjs pages notes.sdocx  and give this script a page JSON")
    if low.endswith(".goodnotes"):
        import goodnotes

        return "GoodNotes", goodnotes.read(path)
    if low.endswith(".notewise"):
        import notewise

        return "Notewise", notewise.read(path)
    if low.endswith(".hinote"):
        import huawei

        return "Huawei Notes", huawei.read(path)
    if zipfile.is_zipfile(path):
        names = zipfile.ZipFile(path).namelist()
        if any(n.endswith("/info") and "/pages/" in n for n in names):
            import kilonotes

            return "Kilonotes", kilonotes.read(path)
        if any(n.startswith("notes/") for n in names):
            import goodnotes

            return "GoodNotes", goodnotes.read(path)
        if any(n.startswith("page/") for n in names):
            import notewise

            return "Notewise", notewise.read(path)
        if any(n.endswith(".jhinote") for n in names):
            import huawei

            return "Huawei Notes", huawei.read(path)
    raise SystemExit("unknown file: expected .goodnotes, .notewise, .hinote, a Kilonotes .zip or a page .json from samsung/sdocx2ink.mjs")


def split_lines(strokes):
    """Strokes of a page -> lists of stroke indices, one per text line, top to bottom; writing order is kept inside a line."""
    strokes = [np.asarray(s, np.float64) for s in strokes]
    if not strokes:
        return []
    heights = np.array([np.ptp(s[:, 1]) for s in strokes])
    widths = np.array([np.ptp(s[:, 0]) for s in strokes])
    letters = heights[(heights > 0) & (np.maximum(heights, widths) > 0.2 * np.median(np.maximum(heights, widths)))]
    h = float(np.median(letters)) if len(letters) else float(max(heights.max(), 1.0))
    if h <= 0:
        return [list(range(len(strokes)))]
    # ink along y: every stroke spreads its length evenly over its points
    ys, ws = [], []
    for s in strokes:
        length = float(np.hypot(*np.diff(s, axis=0).T).sum()) if len(s) > 1 else 0.0
        ys.append(s[:, 1])
        ws.append(np.full(len(s), (length + 0.1 * h) / len(s)))
    ys, ws = np.concatenate(ys), np.concatenate(ws)
    step = h / 8
    edges = np.arange(ys.min() - 2 * h, ys.max() + 2 * h + step, step)
    density, _ = np.histogram(ys, edges, weights=ws)
    kernel = np.hanning(2 * 8 + 1)  # about one letter height wide
    density = np.convolve(density, kernel / kernel.sum(), mode="same")
    floor = 0.12 * density.max()
    peaks = [i for i in range(1, len(density) - 1) if density[i] > floor and density[i] >= density[i - 1] and density[i] > density[i + 1]]
    merged = []
    for i in peaks:  # two maxima closer than a letter height with no real dip between them are one line
        if merged and (i - merged[-1]) * step < 1.1 * h and density[merged[-1] : i + 1].min() > 0.6 * min(density[i], density[merged[-1]]):
            if density[i] > density[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    centres = np.array([edges[i] + step / 2 for i in merged])
    if len(centres) == 0:
        return [list(range(len(strokes)))]
    lines = [[] for _ in centres]
    for k, s in enumerate(strokes):
        lines[int(np.argmin(np.abs(centres - np.median(s[:, 1]))))].append(k)
    return [line for line in lines if line]


def write_svg(path, page, lines, chosen):
    strokes = [np.asarray(s, np.float64) for s in page["strokes"]]
    points = np.concatenate(strokes)
    x0, y0 = points.min(0)
    x1, y1 = points.max(0)
    pad = 0.04 * max(x1 - x0, y1 - y0, 1.0)
    W, H = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    k = 1200.0 / W
    size = max(10.0, 0.012 * 1200)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {H * k:.0f}" width="1200" height="{H * k:.0f}"><rect width="100%" height="100%" fill="#fff"/>']
    for n, line in enumerate(lines):
        colour = "#c33f2e" if n == chosen else "#141414"
        d = " ".join("M" + " L".join(f"{(x - x0 + pad) * k:.1f},{(y - y0 + pad) * k:.1f}" for x, y in strokes[i]) for i in line)
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>')
        top = min(strokes[i][:, 1].min() for i in line)
        left = min(strokes[i][:, 0].min() for i in line)
        out.append(f'<text x="{max(2.0, (left - x0 + pad) * k - 2.2 * size):.1f}" y="{(top - y0 + pad) * k + size:.1f}" font-family="sans-serif" font-size="{size:.0f}" fill="#2a6fb0">{n + 1}</text>')
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("".join(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", help="the file exported from the app")
    ap.add_argument("out", nargs="?", help="where to write the strokes of the line (JSON); without it the file is only listed")
    ap.add_argument("--page", type=int, help="page number, from 1; default: the only page with ink")
    ap.add_argument("--line", type=int, help="line number on the page, from 1; default: the whole page is the line")
    ap.add_argument("--svg", help="also draw the page with the lines numbered")
    args = ap.parse_args()

    app, pages = read_pages(args.notebook)
    inked = [n for n, p in enumerate(pages) if p["strokes"]]
    if not inked:
        raise SystemExit(f"{app}: no pen strokes found in this file")
    if args.out is None and args.page is None:
        print(f"{app}: {len(pages)} pages, {len(inked)} with ink")
        for n in inked:
            print(f"  page {n + 1}: {len(pages[n]['strokes'])} strokes, {len(split_lines(pages[n]['strokes']))} lines")
        print("pick one: --page N [--line K], and look at it with --svg check.svg")
        return
    if args.page is None:
        if len(inked) > 1:
            raise SystemExit(f"{len(inked)} pages have ink; choose one with --page (run without the output name to list them)")
        args.page = inked[0] + 1
    if not 1 <= args.page <= len(pages) or not pages[args.page - 1]["strokes"]:
        raise SystemExit(f"page {args.page} has no ink")
    page = pages[args.page - 1]
    lines = split_lines(page["strokes"])
    if args.line is None:
        chosen, picked = None, list(range(len(page["strokes"])))
        if len(lines) > 1:
            print(f"the page seems to hold {len(lines)} lines and all of them are taken as one; choose with --line if that is wrong")
    else:
        if not 1 <= args.line <= len(lines):
            raise SystemExit(f"the page has {len(lines)} lines")
        chosen, picked = args.line - 1, lines[args.line - 1]
    for n, line in enumerate(lines):
        xs = np.concatenate([np.asarray(page["strokes"][i], np.float64)[:, 0] for i in line])
        print(f"  line {n + 1}: {len(line)} strokes, width {np.ptp(xs):.0f}" + ("  <- taken" if n == chosen else ""))
    if args.svg:
        write_svg(args.svg, page, lines if chosen is not None else [picked], chosen if chosen is not None else 0)
        print("picture:", args.svg)
    if args.out:
        if os.path.exists(args.out):
            raise SystemExit(f"{args.out} already exists")
        strokes = [[[round(float(x), 3), round(float(y), 3)] for x, y in page["strokes"][i]] for i in picked]
        with open(args.out, "w") as fh:
            json.dump(strokes, fh)
        print(f"{args.out}: {len(strokes)} strokes from {app}. Next: python make_ref.py {args.out} my_line.npy")


if __name__ == "__main__":
    main()
