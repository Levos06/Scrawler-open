"""Downloads the model weights from Hugging Face into this folder.

python download_weights.py            # the models that write pages, about 80 MB
python download_weights.py --tracer   # also the tracer that reads your line from a photo, about 60 MB more
"""
import argparse
import os

from huggingface_hub import snapshot_download

REPO = "levossadtchi/Scrawler-open"
WRITER = [
    "ckpt_dl/b4styc/fp16/step003000_fp16.pt",
    "ckpt_dl/planner7/best.pt",
    "ckpt_semb3_all/best.pt",
    "ckpt_glyph_dn_u2s/fp16/step050000.pt",
    "ckpt_dl/glyph_cls2.pt",
]
TRACER = ["trace/weights/*"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    snapshot_download(REPO, allow_patterns=WRITER + (TRACER if args.tracer else []), local_dir=root)
    print("weights are in", root)


if __name__ == "__main__":
    main()
