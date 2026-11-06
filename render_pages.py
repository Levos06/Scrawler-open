"""Re-render strokes.json without the models, for example with another pen width.

python render_pages.py out/strokes.json out_thick --pen 1.4
"""
import argparse
import json
import os
import sys

from scrawler.render import render


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strokes", help="strokes.json written by write_pages.py")
    ap.add_argument("out", help="output folder, must not exist")
    ap.add_argument("--pen", type=float, default=1.1, help="pen width in points")
    ap.add_argument("--png-dpi", type=int, default=300)
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()
    if os.path.exists(args.out):
        sys.exit(f"{args.out} exists")
    os.makedirs(args.out)
    render(json.load(open(args.strokes)), args.out, pen=args.pen, png_dpi=args.png_dpi, smooth=not args.no_smooth)
    print("rendered ->", args.out)


if __name__ == "__main__":
    main()
