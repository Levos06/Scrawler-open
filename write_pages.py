"""Turns a text file with LaTeX formulas into handwritten notebook pages.

python write_pages.py notes.txt out
python write_pages.py notes.txt out --ref my_line.npy

Input: UTF-8 text, one line of the notes per line. $...$ is a formula, an empty line is a paragraph gap, and a line starting
with "# " is a heading. Long lines are wrapped, never inside $...$.

Output (the folder must not exist yet): pages.pdf (vector), page_NN.svg, page_NN.png (300 dpi), strokes.json, report.json.
"""
import argparse
import collections
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import torch

from scrawler.common import DEFAULT_REF, DEVICE, MATHJAX_DIR
from scrawler.formula import X_HEIGHT_EM, FormulaWriter, to_points
from scrawler.render import render
from scrawler.style_encoder import load_style_net, style_vector
from scrawler.text_model import TextModel

PAGE_WIDTH, PAGE_HEIGHT = 1240, 1754  # A4 at 150 dpi
MARGIN_LEFT, MARGIN_TOP = 90, 110
BATCH = 16


def wrap(line, width):
    out, current = [], ""
    for part in re.split(r"(\$[^$]*\$)", line):
        for word in [part] if part.startswith("$") else re.split(r"(?<= )", part):
            if not word:
                continue
            if len(current) + len(word) > width and current.strip():
                out.append(current.rstrip())
                current = word.lstrip()
            else:
                current += word
    if current.strip():
        out.append(current.rstrip())
    return out


def read_items(path, width):
    """-> list of lines of the notes; None is a gap."""
    items = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh.read().splitlines():
            s = raw.strip()
            if not s:
                items.append(None)
                continue
            if s.startswith("# "):
                s = s[2:].strip()
                items.append(None)
            for line in wrap(s, width):
                # a wrapped tail like «$...$:» is a formula with its punctuation inside
                m = re.fullmatch(r"\s*\$([^$]+)\$\s*([.,:;!?]*)\s*", line)
                items.append("$" + m.group(1) + m.group(2) + "$" if m else line)
    return items


def kind_of(line):
    if "$" not in line:
        return "prose"
    s = line.strip()
    return "formula" if s.startswith("$") and s.endswith("$") and s.strip("$").count("$") == 0 else "mixed"


def absolute(q):
    q = np.asarray(q, np.float32)
    return np.cumsum(q[:, :2], 0), q[:, 2]


def x_height(q):
    """The shortest y-interval holding 60% of the inked points, x 1.29. -> (x-height, baseline). The text model normalises every
    line by its own height, so this is measured line by line."""
    a, pen = absolute(q)
    y = np.sort(a[pen < 0.5, 1] if (pen < 0.5).sum() > 10 else a[:, 1])
    n = len(y)
    k = max(1, int(0.6 * n))
    if n - k < 1:
        return max(float(y[-1] - y[0]), 1e-3), float(y[-1])
    widths = y[k:] - y[: n - k]
    j = int(np.argmin(widths))
    return max(1.29 * float(widths[j]), 1e-3), float(y[j + k])


def in_x_heights(q):
    """A line [n, 3] -> points with x-height 1 and the baseline at y = 0, and the pen column."""
    xh, baseline = x_height(q)
    a, pen = absolute(q)
    return (a - [0, baseline]) / xh, pen


def lay_out_pages(items, written, kinds, xh, lines_per_page):
    pitch = 3.2 * xh
    on_grid = lambda v: MARGIN_TOP + np.ceil((v - MARGIN_TOP) / pitch - 1e-6) * pitch  # the next ruling line at or below v
    ruling = [float(v) for v in np.arange(MARGIN_TOP, PAGE_HEIGHT - 60, pitch)]
    pages = [[]]
    y = MARGIN_TOP - pitch
    previous_bottom = float(MARGIN_TOP - 2 * xh)
    count = 0
    for k, text in enumerate(items):
        if text is None:
            y += pitch
            previous_bottom += pitch
            continue
        a, pen = written[k]
        top, bottom = a[:, 1].min(), a[:, 1].max()
        s = xh
        width = (a[:, 0].max() - a[:, 0].min()) * s
        if width > PAGE_WIDTH - MARGIN_LEFT - 50:
            s *= (PAGE_WIDTH - MARGIN_LEFT - 50) / width
        baseline = on_grid(max(y + pitch, previous_bottom + 0.6 * xh - top * s))
        if baseline + bottom * s > PAGE_HEIGHT - 60 or count >= lines_per_page:
            pages.append([])
            y = MARGIN_TOP - pitch
            previous_bottom = float(MARGIN_TOP - 2 * xh)
            count = 0
            baseline = on_grid(max(y + pitch, previous_bottom + 0.6 * xh - top * s))
        x0 = a[:, 0].min()
        indent = 0 if kinds.get(k) != "formula" else 40
        current = []
        for (px, py), p in zip(a, pen):
            current.append([round(float(MARGIN_LEFT + indent + (px - x0) * s), 2), round(float(baseline + py * s), 2)])
            if p > 0.5:
                pages[-1].append(current)
                current = []
        if current:
            pages[-1].append(current)
        y = baseline
        previous_bottom = baseline + bottom * s
        count += 1
    return dict(W=PAGE_WIDTH, H=PAGE_HEIGHT, dpi=150, ruling=ruling, pages=pages)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="the text file")
    ap.add_argument("out", help="output folder, must not exist")
    ap.add_argument("--ref", default=DEFAULT_REF, help="the style: one line of handwriting, [n, 3] .npy from make_ref.py")
    ap.add_argument("--seed", type=int, default=0, help="another seed gives another writing of the same text")
    ap.add_argument("--wrap", type=int, default=42, help="line length in characters")
    ap.add_argument("--x-height", type=float, default=15.0, help="height of small letters in page px; the ruling follows it")
    ap.add_argument("--lines-per-page", type=int, default=99)
    ap.add_argument("--pen", type=float, default=1.1, help="pen width in points")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists")

    items = read_items(args.input, args.wrap)
    lines = [(k, t) for k, t in enumerate(items) if t]
    kinds = {k: kind_of(t) for k, t in lines}
    if any(v != "prose" for v in kinds.values()) and not (shutil.which("node") and os.path.isdir(os.path.join(MATHJAX_DIR, "node_modules"))):
        sys.exit("Formulas need Node.js and MathJax: install Node, then run  cd tools/mathjax && npm ci")

    text = TextModel()
    unknown = sorted({c for _, t in lines for c in t if c not in text.stoi})
    if unknown:
        print("characters the text model does not know (dropped from prose):", "".join(unknown), flush=True)
    reference = np.load(args.ref).astype(np.float32)
    print(f"device {DEVICE}, style from {args.ref} ({len(reference)} points)", flush=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    semb = style_vector(load_style_net(), reference)
    style = text.style(semb)
    formulas = FormulaWriter(text, reference, semb)

    tokens = {k: [c for c in t if c in text.stoi] for k, t in lines}
    by_text_model = [k for k, _ in lines if kinds[k] != "formula"]
    sampled, durations = {}, {}
    for start in range(0, len(by_text_model), BATCH):
        chunk = by_text_model[start : start + BATCH]
        chars = [tokens[k] for k in chunk]
        planned = text.durations(chars, torch.cat([semb] * len(chunk), 0))
        for k, d, q in zip(chunk, planned, text.sample(chars, planned, torch.cat([style] * len(chunk), 0))):
            sampled[k] = np.asarray(q, np.float32)
            durations[k] = d
        print(f"  text: {min(start + BATCH, len(by_text_model))}/{len(by_text_model)} lines", flush=True)
    text_seconds = time.time() - started

    written, report = {}, []
    for number, (k, t) in enumerate(lines):
        t0 = time.time()
        entry = dict(line=k, kind=kinds[k], text=t)
        if kinds[k] == "formula":
            strokes, entry["missing"] = formulas.compose(t, rng)
            if strokes:
                a = to_points(strokes, 1 / X_HEIGHT_EM)
                written[k] = (a[:, :2], a[:, 2])
        elif kinds[k] == "mixed":
            q, entry["missing"], _ = formulas.mixed_line(sampled[k], tokens[k], durations[k], rng)
            if q is not None:
                xh, baseline = x_height(sampled[k])  # by the line before the formulas went in
                a, pen = absolute(q)
                written[k] = ((a - [0, baseline]) / xh, pen)
        else:
            written[k] = in_x_heights(sampled[k])
        if k not in written:  # the formula module failed: the text model writes the line itself
            entry["fallback_text_model"] = True
            planned = text.durations([tokens[k]], semb)
            written[k] = in_x_heights(np.asarray(text.sample([tokens[k]], planned, style)[0], np.float32))
        entry["sec"] = round(time.time() - t0, 2)
        report.append(entry)
        if kinds[k] != "prose":
            print(f"  [{number + 1}/{len(lines)}] {kinds[k]} {entry['sec']:.1f} s{' FALLBACK' if entry.get('fallback_text_model') else ''}: {t[:60]}", flush=True)

    doc = lay_out_pages(items, written, kinds, args.x_height, args.lines_per_page)
    os.makedirs(args.out)
    with open(os.path.join(args.out, "strokes.json"), "w") as fh:
        json.dump(doc, fh)
    render(doc, args.out, pen=args.pen)
    summary = dict(
        pages=len(doc["pages"]),
        lines=len(lines),
        kinds=dict(collections.Counter(kinds.values())),
        sec_total=round(time.time() - started, 1),
        sec_text_model=round(text_seconds, 1),
        fallbacks=sum(1 for r in report if r.get("fallback_text_model")),
        glyph_sources=dict(formulas.stats),
        per_line=report,
    )
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"{len(doc['pages'])} pages, {len(lines)} lines in {time.time() - started:.0f} s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
