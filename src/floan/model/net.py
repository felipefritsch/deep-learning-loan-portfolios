"""M8 — Deep neural net (Model C, ``02_LOAN_LEVEL §6``): MLP with categorical embeddings.

The nonlinear model of the headline comparison. Same data, importance-weighted loss and
NLL evaluator as the M7 logit (``torch_common.py``) — the only thing that changes is the
``predict`` body: categoricals enter as **learned embeddings** (vs the logit's one-hot,
``§2``) and the design flows through a deep ReLU MLP with dropout (``§6``), so every
out-of-sample NLL difference is attributable to architecture alone (``§5``).

Architecture (``§6``, the paper's optimum): ``[standardized continuous ‖ embedded
categoricals ‖ binary] → ReLU MLP (200 then 140×4) → 7 class logits``, dropout on each
hidden layer. ``net.py`` is a pure library (no SSD, no project config) — depth/dropout
are constructor args so the M9 depth grid ``{1,3,5,7} × dropout {0,0.2,0.5}`` reuses it
unchanged; M8 trains the single paper config ``depth=5, dropout=0.2``.

Index 0 of every embedding is the reserved ``UNK`` row (out-of-vocab / null, ``features
§2``); it is a normal **learnable** parameter (no ``padding_idx``), so missing values and
unseen test levels share one bucket the net can fit, rather than a frozen zero vector.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from floan.model import features as F  # N_CLASSES; the vocab/scaler contract the batch dicts follow

# Paper's depth→width rule (§6: "5 layers: 200 then 140×4"). First hidden layer is wide
# (200), the rest a constant 140; depth is the number of hidden layers.
PAPER_FIRST_WIDTH = 200
PAPER_REST_WIDTH = 140
# Embedding-size cap. The fast.ai heuristic min(cap, round(1.6·card^0.56)) keeps the two
# high-cardinality geos (MSA ~407, zip3 ~920) at ~30/~50 dims instead of the logit's ~1325
# one-hot columns — the compression that motivates embeddings over one-hot (§2).
MAX_EMB_DIM = 64


def depth_to_hidden(depth: int, width_mult: float = 1.0) -> list[int]:
    """Hidden-layer widths for a given depth under the paper's rule (§6).

    ``depth`` = number of hidden layers. ``depth 5 → [200, 140, 140, 140, 140]`` (the M8
    config); ``1 → [200]``; the M9 grid ``{1,3,5,7}`` indexes straight in. ``depth 0`` is
    the multinomial logit (``logit.py``), so it returns an empty list (a bare output layer).

    ``width_mult`` scales both the wide first layer and the constant rest (M12 width sweep:
    the appendix completeness check that the paper-inherited 200/140 widths aren't leaving
    accuracy on the table). It defaults to 1.0, where ``round(200·1.0)=200`` /
    ``round(140·1.0)=140`` reproduce the paper widths exactly — so every pre-M12 run is
    bit-identical. Half → 100/70, double → 400/280.
    """
    if depth < 1:
        return []
    first = int(round(PAPER_FIRST_WIDTH * width_mult))
    rest = int(round(PAPER_REST_WIDTH * width_mult))
    return [first] + [rest] * (depth - 1)


def emb_dim(cardinality: int, cap: int = MAX_EMB_DIM) -> int:
    """fast.ai embedding width ``min(cap, round(1.6·card^0.56))``. ``cardinality`` is the
    full ``Vocab.vocab_size`` (levels **plus** the UNK slot), so it matches the embedding's
    row count exactly."""
    return int(min(cap, max(1, round(1.6 * cardinality ** 0.56))))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MortgageMLP(nn.Module):
    """``[cont ‖ embed(cat) ‖ bin] → (Linear→ReLU→Dropout)×L → Linear→7 logits``.

    ``cont`` ``[N, n_continuous]`` standardized floats, ``cat`` ``[N, n_categorical]``
    int64 vocab indices (one embedding per column, concatenated), ``bin`` ``[N, n_binary]``
    0/1 floats. Dropout (rate ``dropout``) sits after every hidden ReLU (``§6``); the output
    layer is unregularized by dropout. With ``hidden_dims=[]`` the model degenerates to a
    single affine map — numerically the logit, kept so the depth grid is total.
    """

    def __init__(self, n_continuous: int, n_binary: int, vocab_sizes: list[int],
                 hidden_dims: list[int], n_classes: int = F.N_CLASSES,
                 dropout: float = 0.2, emb_dims: list[int] | None = None):
        super().__init__()
        if emb_dims is None:
            emb_dims = [emb_dim(v) for v in vocab_sizes]
        assert len(emb_dims) == len(vocab_sizes), "emb_dims must align with vocab_sizes"
        self.embeddings = nn.ModuleList(
            [nn.Embedding(v, d) for v, d in zip(vocab_sizes, emb_dims)])
        in_dim = n_continuous + n_binary + sum(emb_dims)

        layers: list[nn.Module] = []
        d_prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d_prev, h), nn.ReLU(), nn.Dropout(dropout)]
            d_prev = h
        layers.append(nn.Linear(d_prev, n_classes))
        self.mlp = nn.Sequential(*layers)

        # Record the constructor inputs so a checkpoint can rebuild this net identically.
        self.n_continuous = n_continuous
        self.n_binary = n_binary
        self.vocab_sizes = list(vocab_sizes)
        self.emb_dims = list(emb_dims)
        self.hidden_dims = list(hidden_dims)
        self.n_classes = n_classes
        self.dropout = dropout
        self.in_dim = in_dim

    def forward(self, cont: torch.Tensor, cat: torch.Tensor,
                binb: torch.Tensor) -> torch.Tensor:
        feats = [cont]
        for j, emb in enumerate(self.embeddings):
            feats.append(emb(cat[:, j]))
        feats.append(binb)
        return self.mlp(torch.cat(feats, dim=1))

    def arch(self) -> dict:
        """Constructor-argument dict — everything needed to rebuild this net for a resume."""
        return {"n_continuous": self.n_continuous, "n_binary": self.n_binary,
                "vocab_sizes": self.vocab_sizes, "emb_dims": self.emb_dims,
                "hidden_dims": self.hidden_dims, "n_classes": self.n_classes,
                "dropout": self.dropout}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def from_arch(arch: dict) -> "MortgageMLP":
    """Rebuild a :class:`MortgageMLP` from the dict :meth:`MortgageMLP.arch` produced —
    the resume path (architecture restored from the checkpoint, weights then loaded)."""
    return MortgageMLP(
        n_continuous=arch["n_continuous"], n_binary=arch["n_binary"],
        vocab_sizes=arch["vocab_sizes"], hidden_dims=arch["hidden_dims"],
        n_classes=arch["n_classes"], dropout=arch["dropout"],
        emb_dims=arch["emb_dims"])


def build(scaler: F.Scaler, vocab: F.Vocab, depth: int = 5,
          dropout: float = 0.2, width_mult: float = 1.0,
          n_binary: int | None = None) -> "MortgageMLP":
    """Construct the net for a fitted ``(scaler, vocab)`` at the paper's depth/width (§6).
    ``n_continuous`` from the scaler, ``n_binary`` from the (fixed) binary block, embedding
    cardinalities from the vocab. ``width_mult`` (default 1.0 = paper widths) scales the
    hidden layers for the M12 width sweep. ``n_binary`` defaults to ``len(F.BINARY)``; the
    M26a augmented net overrides it to ``len(F.BINARY) + len(history.HIST_BINARY)`` (the
    continuous + embedding counts grow automatically via the larger scaler/vocab)."""
    return MortgageMLP(
        n_continuous=len(scaler.cols),
        n_binary=len(F.BINARY) if n_binary is None else n_binary,
        vocab_sizes=list(vocab.vocab_sizes),
        hidden_dims=depth_to_hidden(depth, width_mult), dropout=dropout)


# ---------------------------------------------------------------------------
# predict callback — the torch_common.{weighted_ce,evaluate_nll} contract
# ---------------------------------------------------------------------------
def make_nn_predict():
    """Return ``predict(model, batch, device) -> [N, 7] logits`` for an index-encoded
    batch dict (``features.encode_frame(..., encode='index')``): standardized ``cont`` and
    0/1 ``bin`` go to float32, the ``cat`` vocab indices to int64 for the embedding lookup."""
    def predict(model, batch, device):
        cont = torch.as_tensor(np.asarray(batch["cont"]), dtype=torch.float32, device=device)
        cat = torch.as_tensor(np.asarray(batch["cat"]), dtype=torch.long, device=device)
        binb = torch.as_tensor(np.asarray(batch["bin"]), dtype=torch.float32, device=device)
        return model(cont, cat, binb)
    return predict
