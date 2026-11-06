"""The text model: a diffusion denoiser over the pen points of a whole line, and its duration planner.

The planner says how many pen points each character gets. The denoiser then turns noise into points, knowing for every point
which character it belongs to and how far through that character it is. It is not autoregressive: all points are denoised at once.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import AMP, CKPT_PLANNER, CKPT_TEXT, DEVICE, load_checkpoint

MAX_POINTS = 1200
NOISE_STEPS = 100
SAMPLE_STEPS = 20
GUIDANCE = 1.5
PHASES = 8
HIDDEN = 384
GRU_HIDDEN = 512
BLOCKS_BEFORE = 6
BLOCKS_AFTER = 4
ADVANCE = 0.017  # mean x advance per point; the net predicts the residual
SCALE_X = 0.6
SCALE_Y = 0.25


def alpha_bar(steps=NOISE_STEPS, s=0.008):
    t = np.arange(1, steps + 1) / steps
    f = np.cos((t + s) / (1 + s) * math.pi / 2) ** 2 / math.cos(s / (1 + s) * math.pi / 2) ** 2
    return np.clip(f, 1e-4, 0.9999).astype(np.float32)


ALPHA_BAR = torch.tensor(alpha_bar(), device=DEVICE)


def time_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None] * (1000.0 / NOISE_STEPS)
    return torch.cat([a.sin(), a.cos()], -1)


class FiLM(nn.Module):
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.l = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.l.weight)
        nn.init.zeros_(self.l.bias)

    def forward(self, x, g):
        a, b = self.l(g)[:, None].chunk(2, -1)
        return x * (1 + a) + b


class ConvBlock(nn.Module):
    """A ConvNeXt-style 1D block."""

    def __init__(self, dim, cond_dim, dilation):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, 7, padding=3 * dilation, dilation=dilation, groups=dim)
        self.ln = nn.LayerNorm(dim)
        self.film = FiLM(cond_dim, dim)
        self.f1 = nn.Linear(dim, 4 * dim)
        self.f2 = nn.Linear(4 * dim, dim)
        self.gam = nn.Parameter(torch.full((dim,), 0.1))

    def forward(self, x, mask, g):
        h = self.dw((x * mask).transpose(1, 2)).transpose(1, 2)
        h = self.film(self.ln(h), g)
        h = self.f2(F.gelu(self.f1(h)))
        return (x + self.gam * h) * mask


class Denoiser(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        D, G = HIDDEN, 256
        self.ec = nn.Embedding(vocab_size + 1, D)  # current, previous and next character; the last row = "dropped" for guidance
        self.ep = nn.Embedding(vocab_size + 1, D)
        self.en = nn.Embedding(vocab_size + 1, D)
        self.eph = nn.Embedding(PHASES, D)
        self.inp = nn.Linear(3 + 3 + 3 + 2, D)
        self.tm = nn.Sequential(nn.Linear(128, G), nn.SiLU(), nn.Linear(G, G))
        self.sp = nn.Linear(256, G)
        self.null_style = nn.Parameter(torch.zeros(256))
        self.semb_proj = nn.Linear(256, 256)
        self.sin = nn.Linear(256, D)
        dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
        self.pre = nn.ModuleList([ConvBlock(D, G, dilations[k]) for k in range(BLOCKS_BEFORE)])
        self.gln = nn.LayerNorm(D)
        self.gfilm = FiLM(G, D)
        self.gru = nn.GRU(D, GRU_HIDDEN, num_layers=2, batch_first=True, bidirectional=True)
        self.gout = nn.Linear(2 * GRU_HIDDEN, D)
        self.post = nn.ModuleList([ConvBlock(D, G, dilations[k]) for k in range(BLOCKS_AFTER)])
        self.oln = nn.LayerNorm(D)
        self.head = nn.Linear(D, 3)

    def forward(self, z, t, cond, style, self_cond):
        mask_ = cond["mask"]
        mask = mask_[..., None]
        length = z.size(1)
        pos = torch.arange(length, device=z.device, dtype=torch.float32)[None, :, None].expand(z.size(0), -1, -1) / 600.0
        phase = cond["progress"][..., None]
        feats = torch.cat(
            [z, self_cond, phase, torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase), torch.log1p(cond["duration"])[..., None] / 3.0, pos],
            -1,
        )
        x = (self.inp(feats) + self.ec(cond["char"]) + self.ep(cond["prev"]) + self.en(cond["next"]) + self.eph(cond["phase"]) + self.sin(style)[:, None]) * mask
        g = F.silu(self.tm(time_embedding(t)) + self.sp(style))
        for block in self.pre:
            x = block(x, mask, g)
        h = self.gfilm(self.gln(x), g)
        lengths = mask_.sum(1).long().clamp(min=1)
        with torch.autocast("cuda", enabled=False):
            packed = nn.utils.rnn.pack_padded_sequence(h.float(), lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=length)
        x = (x + self.gout(out.to(x.dtype))) * mask
        for block in self.post:
            x = block(x, mask, g)
        out = self.head(self.oln(x)).float()
        return out[..., :2] * mask, out[..., 2]


class Planner(nn.Module):
    """Characters + style -> log1p of the number of pen points of each character."""

    MAX_TOKENS = 200

    def __init__(self, vocab_size, d=256, layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(self.MAX_TOKENS + 2, d)
        self.sty = nn.Linear(256, d)
        self.null = nn.Parameter(torch.zeros(256))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, tokens, token_mask, style):
        h = self.emb(tokens) + self.pos(torch.arange(tokens.shape[1], device=tokens.device))[None] + self.sty(style)[:, None]
        return self.out(self.enc(h, src_key_padding_mask=token_mask < 0.5))[..., 0]


def _plan(char_ids, durations):
    """Per-character plan -> per-point conditioning."""
    char, prev, next_, phase, progress, duration = [], [], [], [], [], []
    for k, (c, d) in enumerate(zip(char_ids, durations)):
        d = max(1, int(d))
        before = char_ids[k - 1] if k > 0 else 0
        after = char_ids[k + 1] if k + 1 < len(char_ids) else 0
        for q in range(d):
            char.append(c)
            prev.append(before)
            next_.append(after)
            phase.append(min(PHASES - 1, (q * PHASES) // d))
            progress.append(q / d)
            duration.append(d)
    return char, prev, next_, phase, progress, duration


class TextModel:
    def __init__(self):
        state = load_checkpoint(CKPT_TEXT)
        self.itos = state["itos"]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.null_token = len(self.itos)
        self.net = Denoiser(len(self.itos)).to(DEVICE)
        self.net.load_state_dict({k: v.float() for k, v in state["e"].items()})
        self.net.eval()
        state = load_checkpoint(CKPT_PLANNER)
        self.planner = Planner(len(state["itos"])).to(DEVICE)
        self.planner.load_state_dict(state["m"])
        self.planner.eval()

    @torch.no_grad()
    def style(self, semb):
        """The writer's style vector -> the vector the denoiser takes."""
        return self.net.semb_proj(semb)

    @torch.no_grad()
    def durations(self, lines, semb):
        """lines: lists of characters; semb [B, 256]. -> points per character for each line."""
        B = len(lines)
        U = min(Planner.MAX_TOKENS + 2, -(-(max(len(t) for t in lines) + 2) // 16) * 16)
        tokens = torch.zeros(B, U, dtype=torch.long)
        token_mask = torch.zeros(B, U)
        for b, line in enumerate(lines):
            ids = [1] + [self.stoi.get(c, 0) for c in line] + [2]  # <bos> ... <eos>
            tokens[b, : len(ids)] = torch.tensor(ids)
            token_mask[b, : len(ids)] = 1
        out = self.planner(tokens.to(DEVICE), token_mask.to(DEVICE), semb.to(DEVICE).float())
        d = torch.expm1(out).clamp(min=1).round().long().cpu().numpy()
        return [list(d[b, 1 : 1 + len(t)]) for b, t in enumerate(lines)]

    @torch.no_grad()
    def sample(self, lines, durations, style):
        """-> for each line an array [n, 3]: dx, dy, pen_up. Uses the global torch random generator."""
        B = len(lines)
        plans = [_plan([self.stoi.get(c, 0) for c in t], d) for t, d in zip(lines, durations)]
        length = min(MAX_POINTS, max(len(p[0]) for p in plans))
        lengths = [min(length, len(p[0])) for p in plans]
        cond = {k: torch.zeros(B, length, dtype=torch.long, device=DEVICE) for k in ("char", "prev", "next", "phase")}
        cond.update(
            progress=torch.zeros(B, length, device=DEVICE),
            duration=torch.zeros(B, length, device=DEVICE),
            mask=torch.zeros(B, length, device=DEVICE),
        )
        for b, plan in enumerate(plans):
            n = lengths[b]
            for key, values in zip(("char", "prev", "next", "phase", "progress", "duration"), plan):
                cond[key][b, :n] = torch.tensor(values[:n], dtype=cond[key].dtype)
            cond["mask"][b, :n] = 1
        style = style.expand(B, -1).clone()

        # classifier-free guidance: the same batch twice, the second time with the characters dropped
        dropped = {
            k: (torch.where(cond["mask"] > 0, torch.full_like(v, self.null_token), v) if k in ("char", "prev", "next") else v)
            for k, v in cond.items()
        }
        cond2 = {k: torch.cat([cond[k], dropped[k]], 0) for k in cond}
        style2 = torch.cat([style, style], 0)

        mask = cond["mask"][..., None]
        z = torch.randn(B, length, 3, device=DEVICE) * mask
        self_cond = torch.zeros_like(z)
        steps = np.linspace(NOISE_STEPS - 1, 0, SAMPLE_STEPS).round().astype(int)
        for k, t in enumerate(steps):
            tt = torch.full((B,), int(t), device=DEVICE, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                xy, pen = self.net(torch.cat([z, z], 0), torch.cat([tt, tt], 0), cond2, style2, torch.cat([self_cond, self_cond], 0))
            xy = xy[B:] + GUIDANCE * (xy[:B] - xy[B:])
            pen = pen[B:] + GUIDANCE * (pen[:B] - pen[B:])
            x0 = torch.cat([xy.float(), torch.tanh(pen.float() / 2)[..., None]], -1) * mask
            self_cond = x0
            if k == len(steps) - 1:
                break
            a, a_next = ALPHA_BAR[t], ALPHA_BAR[steps[k + 1]]
            eps = (z - a.sqrt() * x0) / (1 - a).sqrt()
            z = (a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps) * mask  # DDIM

        index = torch.arange(length, device=DEVICE, dtype=torch.float32)[None]
        points = torch.stack([x0[..., 0] * SCALE_X + ADVANCE * index, x0[..., 1] * SCALE_Y], -1).cpu().numpy()
        pen_up = (pen.float() > 0.0).float().cpu().numpy()
        result = []
        for b in range(B):
            n = lengths[b]
            d = np.zeros((n, 3), np.float32)
            d[1:, :2] = np.diff(points[b, :n], axis=0)
            d[:, 2] = pen_up[b, :n]
            d[-1, 2] = 1
            result.append(d)
        return result
