"""Writer style encoder: one line of handwriting -> a unit style vector [1, 256].

A line is an array [n, 3] of (dx, dy, pen_up) with line height 1.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CKPT_STYLE, DEVICE, load_checkpoint

SCALE = 50.0
STEP = 0.05
MAX_POINTS = 1200


def resample(seq, step=STEP):
    """Places points every `step` along the pen path, stroke by stroke."""
    seq = np.asarray(seq, np.float32)
    if len(seq) == 0:
        return np.zeros((1, 3), np.float32)
    points = np.cumsum(seq[:, :2], 0)
    ends = np.flatnonzero(seq[:, 2] > 0.5)
    starts = np.concatenate([[0], ends[:-1] + 1]) if len(ends) else np.array([0])
    if len(ends) == 0 or ends[-1] != len(seq) - 1:
        ends = np.append(ends, len(seq) - 1)
        starts = np.concatenate([[0], ends[:-1] + 1])
    out_points, out_pen = [], []
    for a, b in zip(starts, ends):
        if b < a:
            continue
        stroke = points[a : b + 1]
        if len(stroke) > 1:
            cum = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(stroke, axis=0).T))])
            total = cum[-1]
            if total > step * 0.5:
                t = np.arange(0, total, step)
                if total - t[-1] > step * 0.25:
                    t = np.append(t, total)
                stroke = np.stack([np.interp(t, cum, stroke[:, 0]), np.interp(t, cum, stroke[:, 1])], 1)
            else:
                stroke = stroke[[0]]
        out_points.append(stroke)
        pen = np.zeros(len(stroke), np.float32)
        pen[-1] = 1
        out_pen.append(pen)
    points = np.concatenate(out_points)
    deltas = np.zeros((len(points), 3), np.float32)
    deltas[1:, :2] = np.diff(points, axis=0)
    deltas[:, 2] = np.concatenate(out_pen)
    return deltas


def features(seq):
    seq = np.asarray(seq, np.float32)[:MAX_POINTS]
    y = np.cumsum(seq[:, 1])
    return np.stack([seq[:, 0] * SCALE, seq[:, 1] * SCALE, seq[:, 2], y - np.median(y)], 1).astype(np.float32)


class StyleNet(nn.Module):
    def __init__(self, out_dim=256, d=192):
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
        self.head = nn.Sequential(nn.Linear(4 * d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, out_dim))

    def forward(self, x, lengths):
        h = self.stem(x.transpose(1, 2)).transpose(1, 2)
        lengths = torch.clamp((lengths + 3) // 4, min=1)  # the stem downsamples by 4
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(out.size(1), device=out.device)[None] < lengths.to(out.device)[:, None]).float()[..., None]
        mean = (out * mask).sum(1) / mask.sum(1)
        top = (out + (mask - 1) * 1e4).max(1).values
        return self.head(torch.cat([mean, top], -1))


def load_style_net(path=CKPT_STYLE):
    state = load_checkpoint(path)
    net = StyleNet(out_dim=256, d=state["D"]).to(DEVICE)
    net.load_state_dict(state["m"])
    net.eval()
    return net


@torch.no_grad()
def style_vector(net, line):
    x = features(resample(np.asarray(line, np.float32)[:, :3]))
    batch = torch.from_numpy(x[None]).to(DEVICE)
    vector = F.normalize(net(batch, torch.tensor([len(x)])).float(), dim=-1).mean(0, keepdim=True)
    return F.normalize(vector, dim=-1).to(DEVICE)
