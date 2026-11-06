"""The glyph model and its picker.

The glyph model is a diffusion denoiser that draws one formula symbol of a given class in the style of a given writer. Several
variants are drawn for each symbol; a small image classifier keeps the most recognisable one.
"""
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .common import AMP, CKPT_GLYPH, CKPT_GLYPH_CLASSIFIER, DEVICE, STROKE_COUNTS, load_checkpoint
from .text_model import ConvBlock, FiLM, alpha_bar, time_embedding

STYLE_DIM = 256
HIDDEN = 256
GRU_HIDDEN = 256
COND_DIM = 192
BLOCKS_BEFORE = 4
BLOCKS_AFTER = 3
NOISE_STEPS = 100
SAMPLE_STEPS = 10
GUIDANCE = 1.2
SCALE = 0.15
PICK_MARGIN = 0.25  # the right class must beat the best other class by this much
MIN_STROKE_SHARE = 0.4  # a variant's stroke count must be this common among real glyphs of the class

ALPHA_BAR = torch.tensor(alpha_bar(NOISE_STEPS), device=DEVICE)


class GlyphDenoiser(nn.Module):
    def __init__(self, classes):
        super().__init__()
        D, G = HIDDEN, COND_DIM
        self.ncls = classes
        self.ec = nn.Embedding(classes + 1, D)  # the last row = "dropped" for guidance
        self.gc = nn.Embedding(classes + 1, G)
        self.ew = nn.Linear(STYLE_DIM + 1, D)  # style = [embedding, null flag]
        self.gw = nn.Linear(STYLE_DIM + 1, G)
        self.inp = nn.Linear(4 + 4 + 5, D)
        self.tm = nn.Sequential(nn.Linear(128, G), nn.SiLU(), nn.Linear(G, G))
        dilations = [1, 2, 4, 1, 2, 4, 8]
        self.pre = nn.ModuleList([ConvBlock(D, G, dilations[k]) for k in range(BLOCKS_BEFORE)])
        self.gln = nn.LayerNorm(D)
        self.gfilm = FiLM(G, D)
        self.gru = nn.GRU(D, GRU_HIDDEN, num_layers=2, batch_first=True, bidirectional=True)
        self.gout = nn.Linear(2 * GRU_HIDDEN, D)
        self.post = nn.ModuleList([ConvBlock(D, G, dilations[k]) for k in range(BLOCKS_AFTER)])
        self.oln = nn.LayerNorm(D)
        self.head = nn.Linear(D, 4)

    def forward(self, z, t, classes, style, mask_, self_cond):
        mask = mask_[..., None]
        B, length = mask_.shape
        lengths = mask_.sum(1).clamp(min=1)
        i = torch.arange(length, device=z.device, dtype=torch.float32)[None].expand(B, -1)
        phase = (i / lengths[:, None])[..., None]
        feats = torch.cat(
            [z, self_cond, phase, torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase), torch.log(lengths)[:, None, None].expand(-1, length, -1) / 4.0, i[..., None] / 100.0],
            -1,
        )
        x = (self.inp(feats) + self.ec(classes)[:, None] + self.ew(style)[:, None]) * mask
        g = F.silu(self.tm(time_embedding(t)) + self.gc(classes) + self.gw(style))
        for block in self.pre:
            x = block(x, mask, g)
        h = self.gfilm(self.gln(x), g)
        with torch.autocast(DEVICE if DEVICE != "mps" else "cpu", enabled=False):
            packed = nn.utils.rnn.pack_padded_sequence(h.float(), lengths.long().cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=length)
        x = (x + self.gout(out.to(x.dtype))) * mask
        for block in self.post:
            x = block(x, mask, g)
        out = self.head(self.oln(x)).float()
        return out[..., :3] * mask, out[..., 3]


@torch.no_grad()
def _denoise(net, classes, style, lengths, generator):
    B = len(classes)
    length = int(max(lengths))
    mask_ = torch.zeros(B, length, device=DEVICE)
    for b, n in enumerate(lengths):
        mask_[b, : int(n)] = 1
    mask = mask_[..., None]
    classes = torch.tensor(classes, device=DEVICE, dtype=torch.long)
    style = torch.tensor(np.asarray(style), device=DEVICE, dtype=torch.float32)
    z = torch.randn(B, length, 4, generator=generator).to(DEVICE) * mask
    self_cond = torch.zeros_like(z)
    steps = np.linspace(NOISE_STEPS - 1, 0, SAMPLE_STEPS).round().astype(int)
    classes2 = torch.cat([classes, torch.full_like(classes, net.ncls)])
    style2 = torch.cat([style, style])
    mask2 = torch.cat([mask_, mask_])
    for k, t in enumerate(steps):
        tt = torch.full((B,), int(t), device=DEVICE, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            xy, pen = net(torch.cat([z, z]), torch.cat([tt, tt]), classes2, style2, mask2, torch.cat([self_cond, self_cond]))
        xy = xy[B:] + GUIDANCE * (xy[:B] - xy[B:])
        pen = pen[B:] + GUIDANCE * (pen[:B] - pen[B:])
        x0 = torch.cat([xy[..., :2].float(), torch.tanh(pen.float() / 2)[..., None], xy[..., 2:3].float()], -1) * mask
        self_cond = x0
        if k == len(steps) - 1:
            break
        a, a_next = ALPHA_BAR[t], ALPHA_BAR[steps[k + 1]]
        eps = (z - a.sqrt() * x0) / (1 - a).sqrt()
        z = (a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps) * mask
    x0 = x0.cpu().numpy()
    pen_up = (pen.float() > 0.0).cpu().numpy()
    result = []
    for b in range(B):
        n = int(lengths[b])
        q = np.zeros((n, 4), np.float32)
        q[:, 0] = x0[b, :n, 0] * SCALE
        q[:, 1] = x0[b, :n, 1] * SCALE
        q[:, 2] = pen_up[b, :n]
        q[-1, 2] = 1
        q[:, 3] = np.clip((x0[b, :n, 3] + 1) / 2, 0, 1)
        result.append(q)
    return result


def _to_strokes(q):
    strokes, current = [], []
    for k in range(len(q)):
        current.append([float(q[k, 0]), float(q[k, 1]), 0.0, float(q[k, 3])])
        if q[k, 2] > 0.5 or k == len(q) - 1:
            strokes.append(current)
            current = []
    return [s for s in strokes if s]


class GlyphModel:
    def __init__(self, stats):
        state = load_checkpoint(CKPT_GLYPH)
        self.stats = stats
        self.classes = state["classes"]
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        self.net = GlyphDenoiser(len(self.classes)).to(DEVICE)
        self.net.load_state_dict({k: v.float() for k, v in state["e"].items()})
        self.net.eval()
        # how many pen points real glyphs of each class have: a variant's length is drawn from this
        self.lengths = {int(k.split("|")[1]): np.array(v, np.int32) for k, v in state["lens"].items() if k.startswith("-1|")}

    def _draw_length(self, class_id, rng):
        a = self.lengths.get(class_id)
        if a is None or len(a) == 0:
            return 30
        return int(a[int(rng.integers(len(a)))])

    def sample(self, classes, style, rng):
        """One batch: class names -> {"strokes": ...}; classes the net does not know come back with no strokes."""
        labels = ["—" if c == "BAR" else "\u20d7" if c == "VEC" else c for c in classes]
        known = [i for i, label in enumerate(labels) if label in self.class_index]
        result = [dict(strokes=[]) for _ in labels]
        self.stats["glyph_net_net"] += len(known)
        self.stats["glyph_net_unknown"] += len(labels) - len(known)
        if not known:
            return result
        ids = [self.class_index[labels[i]] for i in known]
        lengths = [self._draw_length(c, rng) for c in ids]
        rows = np.zeros((len(known), STYLE_DIM + 1), np.float32)
        rows[:, :STYLE_DIM] = style
        generator = torch.Generator(device="cpu").manual_seed(int(rng.integers(1 << 31)))
        out = _denoise(self.net, ids, rows, lengths, generator)
        for j, i in enumerate(known):
            result[i] = dict(strokes=_to_strokes(out[j]))
        return result


class ClassifierNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),
        )
        self.f = nn.Sequential(nn.Linear(128 * 4 + 2, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, image, extra):
        return self.f(torch.cat([self.c(image).flatten(1), extra], 1))


def _raster(strokes, size=32):
    points = np.array([p[:2] for s in strokes for p in s], np.float32)
    x0, y0 = points.min(0)
    x1, y1 = points.max(0)
    scale = (size - 4) / max(x1 - x0, y1 - y0, 1e-6)
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    ox = (size - (x1 - x0) * scale) / 2
    oy = (size - (y1 - y0) * scale) / 2
    for s in strokes:
        q = [((p[0] - x0) * scale + ox, (p[1] - y0) * scale + oy) for p in s]
        if len(q) > 1:
            draw.line(q, fill=255, width=2)
        elif q:
            draw.point(q[0], fill=255)
    return np.asarray(image, np.float32) / 255.0


def _shape_features(strokes, relative_height):
    points = np.array([p[:2] for s in strokes for p in s], np.float32)
    w = np.ptp(points[:, 0])
    h = np.ptp(points[:, 1])
    return np.array([relative_height, np.log((w + 1e-3) / (h + 1e-3))], np.float32)


class GlyphPicker:
    """Chooses the most recognisable of several drawn variants of one symbol."""

    def __init__(self, stats):
        self.stats = stats
        state = load_checkpoint(CKPT_GLYPH_CLASSIFIER)
        self.net = ClassifierNet(len(state["classes"])).to(DEVICE)
        self.net.load_state_dict(state["state"])
        self.net.eval()
        self.index = {c: i for i, c in enumerate(state["classes"])}
        counts = json.load(open(STROKE_COUNTS, encoding="utf-8"))
        self.stroke_counts = {c: {int(k): v for k, v in d.items()} for c, d in counts.items()}

    def _stroke_share(self, cls, n):
        counts = self.stroke_counts.get(cls)
        total = sum(counts.values()) if counts else 0
        return counts.get(n, 0) / total if total >= 20 else 1.0

    def pick(self, variants, cls, relative_height, force=False):
        """Index of the best variant, or None if none is recognised as `cls`. force: the best of what there is."""
        if cls not in self.index:
            self.stats["unscored"] += 1
            return 0
        drawn = lambda q: q["strokes"] and sum(len(t) for t in q["strokes"]) >= 2
        ok = [k for k, q in enumerate(variants) if drawn(q) and self._stroke_share(cls, len(q["strokes"])) >= MIN_STROKE_SHARE]
        if cls in ("0", "O", "o"):  # no slashed zeros
            ok = [k for k in ok if len(variants[k]["strokes"]) == 1]
        if not ok and force:
            ok = [k for k, q in enumerate(variants) if drawn(q)]
        if not ok:
            self.stats["fallback_nstrokes"] += 1
            return None
        images = np.stack([_raster(variants[k]["strokes"]) for k in ok])
        extras = np.stack([_shape_features(variants[k]["strokes"], relative_height) for k in ok])
        with torch.no_grad():
            logits = self.net(torch.from_numpy(images)[:, None].to(DEVICE), torch.from_numpy(extras).to(DEVICE))
            probs = torch.softmax(logits, -1).cpu().numpy()
        own = probs[:, self.index[cls]]
        recognised = probs.argmax(1) == self.index[cls]
        others = probs.copy()
        others[:, self.index[cls]] = -1
        own = own - others.max(1)
        recognised = recognised & (own >= PICK_MARGIN)
        if not recognised.any() and force:
            self.stats["forced"] += 1
            return ok[int(np.argmax(own))]
        if not recognised.any():
            self.stats["fallback"] += 1
            return None
        self.stats["selected"] += 1
        self.stats["first_ok"] += int(recognised[0])
        return ok[int(np.argmax(own))]
