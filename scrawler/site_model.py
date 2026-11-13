"""The text model of synthink.dev and its recipe: one reference line and a text -> the text as a line in that hand.

Three things differ from scrawler/text_model.py:

- The denoiser also attends to the pen points of the reference line (two cross-attention layers), so it reads the line itself
  and not only its style vector.
- The planner's point counts are multiplied by the writer's spread: the length of the reference in points over the length the
  planner gives for the reference text, clipped to [0.6, 1.4]. This needs the text of the reference line.
- Every line is sampled four times. A recogniser keeps the attempts that read best (character error rate within 0.02 of the
  lowest), and of those a writer classifier picks the one closest to the reference.

Lines are arrays [n, 3] of (dx, dy, pen_up) with line height 1, as everywhere in this package.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import AMP, CKPT_CLASSIFIER, CKPT_PLANNER, CKPT_RECOGNISER, CKPT_SITE_TEXT, DEVICE, load_checkpoint
from .style_encoder import SCALE, features, load_style_net, resample, style_vector
from .text_model import ADVANCE, ALPHA_BAR, GUIDANCE, MAX_POINTS, NOISE_STEPS, SAMPLE_STEPS, SCALE_X, SCALE_Y
from .text_model import Denoiser, Planner, _plan, time_embedding

ATTEMPTS = 4
SPREAD_MIN, SPREAD_MAX = 0.6, 1.4
CER_SLACK = 0.02
REF_STEP = 0.05


def to_strokes(line):
    """A line [n, 3] -> a list of strokes, each an array [k, 2] of absolute points."""
    line = np.asarray(line, np.float32)
    points = np.cumsum(line[:, :2], 0)
    strokes, start = [], 0
    for i in np.flatnonzero(line[:, 2] > 0.5):
        strokes.append(points[start : i + 1])
        start = i + 1
    if start < len(points):
        strokes.append(points[start:])
    return strokes


def to_reference(strokes):
    """Strokes in any units -> a reference line: height 1, a point every 0.05 along the pen path. The same as make_ref.py."""
    strokes = [np.asarray(s, np.float64)[:, :2] for s in strokes if len(s)]
    everything = np.concatenate(strokes)
    origin = everything.min(0)
    height = float(everything[:, 1].max() - origin[1])
    out = []
    for p in strokes:
        p = (p - origin) / height
        if len(p) == 1:
            out.append(np.array([[p[0, 0], p[0, 1], 1.0]]))
            continue
        cum = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        u = np.r_[np.arange(0, cum[-1], REF_STEP), cum[-1]] if cum[-1] > 1e-6 else np.array([0.0, cum[-1]])
        r = np.stack([np.interp(u, cum, p[:, 0]), np.interp(u, cum, p[:, 1]), np.zeros(len(u))], 1)
        r[-1, 2] = 1
        out.append(r)
    absolute = np.concatenate(out)
    line = absolute.copy()
    line[1:, :2] = absolute[1:, :2] - absolute[:-1, :2]
    line[0, :2] = 0
    return line.astype(np.float32)


class ReferenceEncoder(nn.Module):
    """A reference line -> the keys the denoiser attends to: one learned "sink" key and one key for every 4 pen points."""

    def __init__(self, d=192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(4, 128, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, 192, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(192, 256, 5, stride=2, padding=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(256, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.rkv = nn.Sequential(nn.Linear(2 * d, 256), nn.LayerNorm(256))
        self.sink = nn.Parameter(torch.zeros(1, 256))

    def forward(self, line):
        """line [n, 3] -> keys [1, 1 + ceil(n / 4), 256]."""
        y = torch.cumsum(line[:, 1], 0)
        middle = torch.sort(y).values[(len(y) - 1) // 2]  # the lower median
        x = torch.cat([line[:, :2] * SCALE, line[:, 2:3], (y - middle)[:, None]], -1)[None]
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(h)
        return torch.cat([self.sink[None], self.rkv(out)], 1)


class SiteDenoiser(Denoiser):
    """The denoiser of text_model.py with cross-attention to the reference keys after the GRU and before the head."""

    def __init__(self, vocab_size):
        super().__init__(vocab_size)
        D = self.head.in_features
        self.enc = ReferenceEncoder()
        self.src_emb = nn.Parameter(torch.zeros(3, 256))  # where the ink came from: 0 tablet, 1 traced scan, 2 unknown
        self.xln = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        self.xatt = nn.ModuleList([nn.MultiheadAttention(D, 4, kdim=256, vdim=256, batch_first=True) for _ in range(2)])

    def forward(self, z, t, cond, style, self_cond, keys):
        mask_ = cond["mask"]
        mask = mask_[..., None]
        length = z.size(1)
        pos = torch.arange(length, device=z.device, dtype=torch.float32)[None, :, None].expand(z.size(0), -1, -1) / 600.0
        phase = cond["progress"][..., None]
        feats = torch.cat(
            [z, self_cond, phase, torch.sin(2 * np.pi * phase), torch.cos(2 * np.pi * phase), torch.log1p(cond["duration"])[..., None] / 3.0, pos],
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
        keys = keys.to(x.dtype)
        x = (x + self.xatt[0](self.xln[0](x), keys, keys, need_weights=False)[0]) * mask
        for block in self.post:
            x = block(x, mask, g)
        x = (x + self.xatt[1](self.xln[1](x), keys, keys, need_weights=False)[0]) * mask
        out = self.head(self.oln(x)).float()
        return out[..., :2] * mask, out[..., 2]


class Recogniser(nn.Module):
    """Pen points -> per-point character scores (CTC). Reads generated lines to tell how legible they are."""

    def __init__(self, vocab_size, hidden=384):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.rnn = nn.LSTM(hidden, hidden, num_layers=3, bidirectional=True, batch_first=True)
        self.out = nn.Linear(2 * hidden, vocab_size)

    def forward(self, x):
        out, _ = self.rnn(self.inp(x))
        return self.out(out)


class WriterEmbedder(nn.Module):
    """A writer classifier without its class layer: a line -> a vector that is close for lines of the same hand."""

    def __init__(self, d=192):
        super().__init__()
        self.stem = ReferenceEncoder(d).stem
        self.gru = nn.GRU(256, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(4 * d, 256), nn.GELU())

    def forward(self, x):
        out, _ = self.gru(self.stem(x.transpose(1, 2)).transpose(1, 2))
        return self.head(torch.cat([out.mean(1), out.max(1).values], -1))


def _edit_rate(target, hypothesis):
    """Edit distance between two id lists over the length of the target."""
    if not target and not hypothesis:
        return 0.0
    row = list(range(len(hypothesis) + 1))
    for i, a in enumerate(target, 1):
        prev, row[0] = row[0], i
        for j, b in enumerate(hypothesis, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (a != b))
    return row[len(hypothesis)] / max(1, len(target))


class SiteModel:
    def __init__(self):
        state = load_checkpoint(CKPT_SITE_TEXT)
        self.itos = state["itos"]
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.null_token = len(self.itos)
        self.net = SiteDenoiser(len(self.itos)).to(DEVICE)
        unused = ("enc.proj.", "enc.cls.")  # the classifier head the reference encoder was started from
        self.net.load_state_dict({k: v.float() for k, v in state["e"].items() if not k.startswith(unused)})
        self.net.eval()
        state = load_checkpoint(CKPT_PLANNER)
        self.planner = Planner(len(state["itos"])).to(DEVICE)
        self.planner.load_state_dict(state["m"])
        self.planner.eval()
        self.style_net = load_style_net()
        state = load_checkpoint(CKPT_RECOGNISER)
        self.recogniser = Recogniser(len(state["itos"])).to(DEVICE)
        self.recogniser.load_state_dict({k: v.float() for k, v in state["m"].items()})
        self.recogniser.eval()
        state = load_checkpoint(CKPT_CLASSIFIER)
        self.embedder = WriterEmbedder().to(DEVICE)
        self.embedder.load_state_dict({k: v.float() for k, v in state["m"].items()})
        self.embedder.eval()

    def clean(self, text):
        """Drops the characters the model does not know."""
        return "".join(c for c in text if c in self.stoi)

    @torch.no_grad()
    def _durations(self, text, semb):
        ids = [1] + [self.stoi[c] for c in text] + [2]  # <bos> ... <eos>
        tokens = torch.tensor([ids], device=DEVICE)
        out = self.planner(tokens, torch.ones(1, len(ids), device=DEVICE), semb.float())
        d = torch.expm1(out).clamp(min=1).round().long().cpu().numpy()
        return [int(v) for v in d[0, 1:-1]]

    @torch.no_grad()
    def _read(self, lines):
        """Equal-length lines -> the recognised character ids of each."""
        x = np.stack([np.stack([q[:, 0] * 20.0, q[:, 1] * 20.0, q[:, 2], np.cumsum(q[:, 1])], 1) for q in lines]).astype(np.float32)
        best = self.recogniser(torch.from_numpy(x).to(DEVICE)).argmax(-1).cpu().numpy()
        out = []
        for row in best:
            ids, prev = [], 0
            for c in row:
                if c != 0 and c != prev:
                    ids.append(int(c))
                prev = c
            out.append(ids)
        return out

    def read(self, line):
        """What the recogniser reads in a line. Use it to check or fill in the text of your reference."""
        return "".join(s for s in (self.itos[i] for i in self._read([np.asarray(line, np.float32)])[0]) if len(s) == 1)

    @torch.no_grad()
    def _embed(self, line):
        """The unit writer-classifier vector of a line, after bringing the line to the reference form."""
        x = features(resample(to_reference(to_strokes(line))))
        e = self.embedder(torch.from_numpy(x[None]).to(DEVICE)).float()[0]
        return F.normalize(e, dim=-1)

    @torch.no_grad()
    def reference(self, line, text):
        """A reference line and what is written in it -> everything the recipe needs from the writer."""
        line = np.asarray(line, np.float32)[:, :3]
        semb = style_vector(self.style_net, line)
        text = self.clean(text)
        spread = 1.0
        if text:
            spread = float(np.clip(len(line) / max(1, sum(self._durations(text, semb))), SPREAD_MIN, SPREAD_MAX))
        return {
            "semb": semb,
            "style": self.net.semb_proj(semb) + self.net.src_emb[0][None],  # written as tablet ink
            "keys": self.net.enc(torch.from_numpy(line[:MAX_POINTS]).to(DEVICE)),
            "embedding": self._embed(line),
            "spread": spread,
        }

    @torch.no_grad()
    def write(self, text, ref, attempts=ATTEMPTS, seed=0):
        """-> {"lines": the attempts, "pick": the index of the chosen one, "cer", "cos", "text": the text after cleaning}."""
        text = self.clean(text)
        if not text:
            raise ValueError("no characters the model knows")
        ids = [self.stoi[c] for c in text]
        durations = [max(1, int(round(d * ref["spread"]))) for d in self._durations(text, ref["semb"])]
        length = sum(durations)
        if length > MAX_POINTS:
            raise ValueError(f"the line is too long: {length} pen points, the model takes {MAX_POINTS}. Shorten the text.")
        B = attempts
        plan = _plan(ids, durations)
        cond = {
            k: torch.tensor(v, dtype=torch.float32 if k in ("progress", "duration") else torch.long, device=DEVICE)[None].expand(B, -1)
            for k, v in zip(("char", "prev", "next", "phase", "progress", "duration"), plan)
        }
        cond["mask"] = torch.ones(B, length, device=DEVICE)
        # classifier-free guidance: the same batch twice, the second time with the characters dropped; style and reference stay
        dropped = {k: (torch.full_like(v, self.null_token) if k in ("char", "prev", "next") else v) for k, v in cond.items()}
        cond2 = {k: torch.cat([cond[k], dropped[k]], 0) for k in cond}
        style2 = ref["style"].expand(2 * B, -1)
        keys2 = ref["keys"].expand(2 * B, -1, -1)

        generator = torch.Generator().manual_seed(seed)
        z = torch.randn(B, length, 3, generator=generator).to(DEVICE)
        self_cond = torch.zeros_like(z)
        steps = np.linspace(NOISE_STEPS - 1, 0, SAMPLE_STEPS).round().astype(int)
        for k, t in enumerate(steps):
            tt = torch.full((2 * B,), int(t), device=DEVICE, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
                xy, pen = self.net(torch.cat([z, z], 0), tt, cond2, style2, torch.cat([self_cond, self_cond], 0), keys2)
            xy = xy[B:] + GUIDANCE * (xy[:B] - xy[B:])
            pen = pen[B:] + GUIDANCE * (pen[:B] - pen[B:])
            x0 = torch.cat([xy.float(), torch.tanh(pen.float() / 2)[..., None]], -1)
            self_cond = x0
            if k == len(steps) - 1:
                break
            a, a_next = ALPHA_BAR[t], ALPHA_BAR[steps[k + 1]]
            eps = (z - a.sqrt() * x0) / (1 - a).sqrt()
            z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps  # DDIM

        index = torch.arange(length, device=DEVICE, dtype=torch.float32)[None]
        points = torch.stack([x0[..., 0] * SCALE_X + ADVANCE * index, x0[..., 1] * SCALE_Y], -1).cpu().numpy()
        pen_up = (pen.float() > 0.0).float().cpu().numpy()
        lines = []
        for b in range(B):
            d = np.zeros((length, 3), np.float32)
            d[1:, :2] = np.diff(points[b], axis=0)
            d[:, 2] = pen_up[b]
            d[-1, 2] = 1
            lines.append(d)

        cer = np.array([_edit_rate(ids, h) for h in self._read(lines)], np.float32)
        cos = np.array([float(self._embed(q) @ ref["embedding"]) for q in lines], np.float32)
        legible = np.flatnonzero(cer <= cer.min() + CER_SLACK)
        pick = int(legible[np.argmax(cos[legible])])
        return {"lines": lines, "pick": pick, "cer": cer, "cos": cos, "text": text, "durations": durations}
