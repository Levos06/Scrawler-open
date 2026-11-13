"""Downloads the model weights from Hugging Face into this folder.

python download_weights.py            # the models that write pages, about 80 MB
python download_weights.py --tracer   # also the tracer that reads your line from a photo, about 60 MB more
python download_weights.py --site     # also the models of synthink.dev for write_line.py, about 75 MB more
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
SITE = [
    "ckpt_dl/u3ref/fp16/step060000_fp16.pt",
    "ckpt_ctc4/ctc_fp16.pt",
    "ckpt_stylecls/embed_fp16.pt",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", action="store_true")
    ap.add_argument("--site", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    snapshot_download(REPO, allow_patterns=WRITER + (TRACER if args.tracer else []) + (SITE if args.site else []), local_dir=root)
    print("weights are in", root)


if __name__ == "__main__":
    main()
