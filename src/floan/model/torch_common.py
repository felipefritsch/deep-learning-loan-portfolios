"""Shared PyTorch loss + streaming-NLL evaluator (``02_LOAN_LEVEL §5``/``§6``).

Every Phase-2 torch model — the M7 multinomial logit (a 0-hidden-layer net) and the
M8+ deep nets — trains and is scored through the **same** importance-weighted
cross-entropy and the **same** NLL accumulator, so out-of-sample NLL differences are
attributable to architecture alone (``§5``). This module owns only those shared pieces
(loss, eval loop, device/seed); each model supplies a ``predict(model, batch, device)``
callback that maps a loader/minibatch dict to ``[N, 7]`` class logits, which is where the
architectures actually differ (one-hot linear for the logit, embeddings+MLP for the NN).

The weighted mean is the right estimator because the train pool is thinned + weighted
(``weight = 1/p_keep``, ``00_OVERVIEW §6.4``): scoring as ``Σ w·CE / Σ w`` keeps the loss
an unbiased estimate of the true conditional NLL. On the never-thinned eval pool every
weight is 1.0, so the weighted mean collapses to the plain mean −log p̂ — identical to the
empirical-matrix benchmark's metric (``benchmarks.py``), which is what makes the two
out-of-sample NLLs directly comparable.

Pure torch + numpy; no SSD, no project imports — importable from the unit tests.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device + determinism
# ---------------------------------------------------------------------------
def resolve_device(name: str = "cpu") -> torch.device:
    """``'cpu'`` (default — deterministic, the M7 choice) / ``'mps'`` / ``'auto'``
    (mps if available else cpu). CUDA is never present on the mac dev box; the
    cloud-GPU path (M8+) passes an explicit device."""
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch so a run (init + minibatch order) is reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ---------------------------------------------------------------------------
# Loss — importance-weighted cross-entropy (= mean NLL)
# ---------------------------------------------------------------------------
def weighted_ce(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weight-averaged cross-entropy ``Σ w·CE / Σ w`` (the §6.4 unbiased training loss).
    ``F.cross_entropy(reduction='none')`` is the per-row −log softmax[y] (log-sum-exp
    stable)."""
    ce = F.cross_entropy(logits, y, reduction="none")
    return (w * ce).sum() / w.sum()


def _as_long(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)


def _as_f32(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Streaming NLL evaluator — float64 accumulation for reproducibility
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_nll(model, batches, predict, device) -> dict:
    """Mean negative log-likelihood of ``model`` over an iterable of batch dicts.

    ``batches`` yields dicts with at least ``y`` (int class index) and ``w`` (weight);
    ``predict(model, batch, device) -> logits`` builds the model input and forwards it.
    Sums are kept in float64 so the result does not drift with batch size / order.
    Returns ``weighted_nll`` (Σ w·CE / Σ w — the headline, unbiased on the train pool),
    ``unweighted_nll`` (plain mean — equals weighted_nll on the unthinned eval pool),
    ``n_rows`` and ``sum_w``.
    """
    model.eval()
    swce = sw = sce = 0.0
    n = 0
    for b in batches:
        y = _as_long(b["y"], device)
        w = _as_f32(b["w"], device)
        ce = F.cross_entropy(predict(model, b, device), y, reduction="none").double()
        swce += float((w.double() * ce).sum())
        sw += float(w.double().sum())
        sce += float(ce.sum())
        n += int(y.numel())
    return {"weighted_nll": swce / sw, "unweighted_nll": sce / n,
            "n_rows": n, "sum_w": sw}
