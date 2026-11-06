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
CKPT_STYLE = os.path.join(ROOT, "ckpt_semb3_all", "best.pt")
CKPT_GLYPH = os.path.join(ROOT, "ckpt_glyph_dn_u2s", "fp16", "step050000.pt")
CKPT_GLYPH_CLASSIFIER = os.path.join(ROOT, "ckpt_dl", "glyph_cls2.pt")
STROKE_COUNTS = os.path.join(ROOT, "data", "glyph_stroke_counts.json")
DEFAULT_REF = os.path.join(ROOT, "examples", "ref_hwr200.npy")
MATHJAX_DIR = os.path.join(ROOT, "tools", "mathjax")


def load_checkpoint(path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run: python download_weights.py")
    return torch.load(path, map_location=DEVICE, weights_only=True)
