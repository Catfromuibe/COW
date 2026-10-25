"""Equal error rate. Higher score means stronger evidence of manipulation."""

from __future__ import annotations

import torch


def eer(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """labels: 0 genuine, 1 manipulated. Returns a fraction in [0, 1]."""
    scores = scores.detach().float().cpu().view(-1)
    labels = labels.detach().float().cpu().view(-1)
    if scores.numel() == 0 or labels.min() == labels.max():
        return float("nan")
    order = torch.argsort(scores)
    y = labels[order]
    genuine = (y == 0).float()
    spoof = (y == 1).float()
    n_gen = genuine.sum().clamp_min(1)
    n_spo = spoof.sum().clamp_min(1)
    # Sorted ascending. Index i predicts manipulated when score >= sorted[i].
    tp = torch.flip(torch.cumsum(torch.flip(spoof, (0,)), 0), (0,))
    fp = torch.flip(torch.cumsum(torch.flip(genuine, (0,)), 0), (0,))
    fpr = torch.cat([fp / n_gen, torch.zeros(1)])
    fnr = torch.cat([(n_spo - tp) / n_spo, torch.ones(1)])
    point = (fpr - fnr).abs().argmin()
    return float((fpr[point] + fnr[point]) * 0.5)
