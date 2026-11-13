"""Paths and the device, shared by all modules."""
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pick_device():
    forced = os.environ.get("SCRAWLER_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
AMP = DEVICE == "cuda"
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

CKPT_TEXT = os.path.join(ROOT, "ckpt_dl", "b4styc", "fp16", "step003000_fp16.pt")
CKPT_PLANNER = os.path.join(ROOT, "ckpt_dl", "planner7", "best.pt")
CKPT_SITE_TEXT = os.path.join(ROOT, "ckpt_dl", "u3ref", "fp16", "step060000_fp16.pt")
CKPT_RECOGNISER = os.path.join(ROOT, "ckpt_ctc4", "ctc_fp16.pt")
CKPT_CLASSIFIER = os.path.join(ROOT, "ckpt_stylecls", "embed_fp16.pt")
CKPT_BILINGUAL_TEXT = os.path.join(ROOT, "ckpt_dl", "u4s2", "fp16", "step020000_fp16.pt")
CKPT_BILINGUAL_STYLE = os.path.join(ROOT, "ckpt_semb4", "best.pt")
CKPT_BILINGUAL_RECOGNISER = os.path.join(ROOT, "ckpt_ctc6", "ctc_fp16.pt")
CKPT_STYLE = os.path.join(ROOT, "ckpt_semb3_all", "best.pt")
CKPT_GLYPH = os.path.join(ROOT, "ckpt_glyph_dn_u2s", "fp16", "step050000.pt")
CKPT_GLYPH_CLASSIFIER = os.path.join(ROOT, "ckpt_dl", "glyph_cls2.pt")
STROKE_COUNTS = os.path.join(ROOT, "data", "glyph_stroke_counts.json")
DEFAULT_REF = os.path.join(ROOT, "examples", "ref_hwr200.npy")
MATHJAX_DIR = os.path.join(ROOT, "tools", "mathjax")


def load_checkpoint(path):
    if not os.path.exists(path):
        hint = " --site" if path in (CKPT_SITE_TEXT, CKPT_RECOGNISER, CKPT_CLASSIFIER) else ""
        hint = " --bilingual" if path in (CKPT_BILINGUAL_TEXT, CKPT_BILINGUAL_STYLE, CKPT_BILINGUAL_RECOGNISER) else hint
        raise SystemExit(f"{path} not found. Run: python download_weights.py{hint}")
    return torch.load(path, map_location=DEVICE, weights_only=True)
