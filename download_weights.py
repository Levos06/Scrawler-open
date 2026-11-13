"""Downloads the model weights from Hugging Face into this folder.

python download_weights.py            # the models that write pages, about 80 MB
python download_weights.py --tracer   # also the tracer that reads your line from a photo, about 60 MB more
python download_weights.py --site     # also the models of synthink.dev for write_line.py, about 75 MB more
python download_weights.py --bilingual  # also the English and Russian model for write_line.py --model bilingual, about 100 MB more
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
BILINGUAL = [  # trained on BRUSH as well: non-commercial research only, see README
    "ckpt_dl/u4s2/fp16/step020000_fp16.pt",
    "ckpt_semb4/best.pt",
    "ckpt_ctc6/ctc_fp16.pt",
    "ckpt_ctc4/ctc_fp16.pt",
    "ckpt_stylecls/embed_fp16.pt",
]
SITE = [
    "ckpt_dl/u3ref/fp16/step060000_fp16.pt",
    "ckpt_ctc4/ctc_fp16.pt",
    "ckpt_stylecls/embed_fp16.pt",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracer", action="store_true")
    ap.add_argument("--site", action="store_true")
    ap.add_argument("--bilingual", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    snapshot_download(REPO, allow_patterns=WRITER + (TRACER if args.tracer else []) + (SITE if args.site else []) + (BILINGUAL if args.bilingual else []), local_dir=root)
    print("weights are in", root)


if __name__ == "__main__":
    main()
