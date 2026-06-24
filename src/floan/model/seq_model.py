"""M26 (step 2a) — minimal sequence model: a 1-layer GRU (``04 §M26``).

The *discriminating* (not competitive) seq model for the value-of-memory probe. Same
per-timestep feature contract as the FF net (``features``: scaled continuous + embedded
categoricals incl. state + binaries), so a GRU here vs the FF net in M26a differ ONLY in
architecture: the FF net sees the row at t; the GRU consumes the trailing-T rows.

``[B, T, cont ‖ embed(cat) ‖ bin] → pack_padded(lengths) → GRU(1 layer, hidden) → h_n →
Linear → 7 logits``. RIGHT-padded sequences (real steps first) let ``pack_padded_sequence``
ignore the pads, so the final hidden state ``h_n`` is the prediction point (t) cleanly — no
leading-pad perturbation. The embedding/width heuristics are reused from ``net.py`` so the
embeddings match the FF net's.

Pure torch (no SSD/config); the ``predict(model, batch, device)`` callback matches the
``torch_common`` loss/eval contract (so ``weighted_ce`` / ``evaluate_nll`` work unchanged).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from floan.model import features as F
from floan.model import net as N  # reuse emb_dim heuristic so embeddings match the FF net


class SeqGRU(nn.Module):
    """1-layer GRU over per-timestep [cont ‖ embed(cat) ‖ bin]; last real hidden → 7 logits."""

    def __init__(self, n_continuous: int, n_binary: int, vocab_sizes: list[int],
                 hidden: int = 48, n_classes: int = F.N_CLASSES,
                 emb_dims: list[int] | None = None):
        super().__init__()
        if emb_dims is None:
            emb_dims = [N.emb_dim(v) for v in vocab_sizes]
        self.embeddings = nn.ModuleList([nn.Embedding(v, d)
                                         for v, d in zip(vocab_sizes, emb_dims)])
        self.input_dim = n_continuous + n_binary + sum(emb_dims)
        self.gru = nn.GRU(self.input_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)
        # provenance for a checkpoint rebuild
        self.n_continuous, self.n_binary = n_continuous, n_binary
        self.vocab_sizes, self.emb_dims = list(vocab_sizes), list(emb_dims)
        self.hidden, self.n_classes = hidden, n_classes

    def forward(self, cont: torch.Tensor, cat: torch.Tensor, binb: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        # cont [B,T,nc]  cat [B,T,ncat] (long)  binb [B,T,nb]  lengths [B]
        embs = [emb(cat[:, :, j]) for j, emb in enumerate(self.embeddings)]
        x = torch.cat([cont, *embs, binb], dim=-1)                       # [B,T,input_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)                                        # h_n [1,B,hidden]
        return self.head(h_n[-1])                                        # [B,7]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(scaler: F.Scaler, vocab: F.Vocab, hidden: int = 48) -> SeqGRU:
    """Construct the GRU for a fitted ``(scaler, vocab)`` — continuous count from the scaler,
    binary from the fixed block, embedding cardinalities from the vocab (FF-matched)."""
    return SeqGRU(n_continuous=len(scaler.cols), n_binary=len(F.BINARY),
                  vocab_sizes=list(vocab.vocab_sizes), hidden=hidden)


def make_seq_predict():
    """``predict(model, batch, device) -> [N, 7]`` for a sequence batch dict
    (``cont``/``cat``/``bin`` ``[B,T,·]`` + ``lengths``) — the torch_common contract."""
    def predict(model, batch, device):
        cont = torch.as_tensor(np.asarray(batch["cont"]), dtype=torch.float32, device=device)
        cat = torch.as_tensor(np.asarray(batch["cat"]), dtype=torch.long, device=device)
        binb = torch.as_tensor(np.asarray(batch["bin"]), dtype=torch.float32, device=device)
        lengths = torch.as_tensor(np.asarray(batch["lengths"]), dtype=torch.long)  # CPU for pack
        return model(cont, cat, binb, lengths)
    return predict
