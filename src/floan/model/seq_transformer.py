"""W2 (``final_week_plan.md``) — minimal sequence transformer: a self-attention encoder
over the trailing-T monthly sequence. The learned-memory counterpart to the GRU
(``seq_model.SeqGRU``), added as a fourth arm to the M27a memory comparison.

Drop-in for the GRU arm: identical per-timestep feature contract as ``net.py`` /
``seq_model.py`` (scaled continuous + embedded categoricals incl. state + binaries) and the
same ``predict(model, batch, device) -> [N, 7]`` callback (the ``torch_common`` loss/eval
contract), so a transformer here vs the GRU differ ONLY in architecture (self-attention vs
recurrent). Pure torch (no SSD/config); embedding heuristic reused from ``net.py`` so the
embeddings match the FF net and the GRU.

``[B, T, cont ‖ embed(cat) ‖ bin] -> Linear d_model + sinusoidal positional encoding ->
TransformerEncoder(n_layers, n_head) with a key-padding mask from lengths -> take the encoder
output at the last REAL step (t) -> Linear -> 7 logits``. Sequences are RIGHT-padded (real
steps first, ``sequence.py``), so the prediction point is index ``lengths-1`` — the
self-attention analogue of the GRU's last real hidden state, keeping the two arms' readout
semantics identical.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from floan.model import features as F
from floan.model import net as N  # reuse emb_dim heuristic so embeddings match the FF net / GRU

# Small frozen architecture (the GRU is hidden=48, 1 layer; this is the matched-capacity
# attention counterpart, a *discriminating* arm, not a separately-tuned competitor).
D_MODEL, N_HEAD, N_LAYERS, FFN, DROPOUT = 64, 4, 2, 128, 0.1


class _PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding, added to the projected inputs."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x [B, T, d_model]
        return x + self.pe[:, : x.size(1)]


class SeqTransformer(nn.Module):
    """Self-attention encoder over per-timestep [cont ‖ embed(cat) ‖ bin]; the encoder
    output at the last real step (t) → 7 logits. Same I/O as ``seq_model.SeqGRU``."""

    def __init__(self, n_continuous: int, n_binary: int, vocab_sizes: list[int],
                 d_model: int = D_MODEL, n_head: int = N_HEAD, n_layers: int = N_LAYERS,
                 ffn: int = FFN, dropout: float = DROPOUT, max_len: int = 12,
                 n_classes: int = F.N_CLASSES, emb_dims: list[int] | None = None):
        super().__init__()
        if emb_dims is None:
            emb_dims = [N.emb_dim(v) for v in vocab_sizes]
        self.embeddings = nn.ModuleList([nn.Embedding(v, d)
                                         for v, d in zip(vocab_sizes, emb_dims)])
        self.input_dim = n_continuous + n_binary + sum(emb_dims)
        self.proj = nn.Linear(self.input_dim, d_model)
        self.posenc = _PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=ffn, dropout=dropout,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)
        # provenance for a checkpoint rebuild (mirrors SeqGRU)
        self.n_continuous, self.n_binary = n_continuous, n_binary
        self.vocab_sizes, self.emb_dims = list(vocab_sizes), list(emb_dims)
        self.d_model, self.n_head, self.n_layers = d_model, n_head, n_layers
        self.ffn, self.dropout_p, self.max_len = ffn, dropout, max_len
        self.n_classes = n_classes

    def forward(self, cont: torch.Tensor, cat: torch.Tensor, binb: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        # cont [B,T,nc]  cat [B,T,ncat] (long)  binb [B,T,nb]  lengths [B]
        embs = [emb(cat[:, :, j]) for j, emb in enumerate(self.embeddings)]
        x = torch.cat([cont, *embs, binb], dim=-1)                  # [B,T,input_dim]
        x = self.posenc(self.proj(x))                              # [B,T,d_model]
        B, T = x.shape[0], x.shape[1]
        pos = torch.arange(T, device=x.device).unsqueeze(0)        # [1,T]
        pad_mask = pos >= lengths.to(x.device).unsqueeze(1)        # [B,T] True = padded → ignored
        h = self.encoder(x, src_key_padding_mask=pad_mask)         # [B,T,d_model]
        last = (lengths.to(x.device) - 1).clamp(min=0)             # index of step t (real)
        h_t = h[torch.arange(B, device=x.device), last]            # [B,d_model]
        return self.head(h_t)                                      # [B,7]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(scaler: F.Scaler, vocab: F.Vocab, **kw) -> SeqTransformer:
    """Construct the transformer for a fitted ``(scaler, vocab)`` — same signature shape as
    ``seq_model.build`` so the M27a/seq drivers swap arms by import alone. ``max_len`` defaults
    to the sequence length; override architecture via ``**kw`` (d_model, n_head, n_layers, …)."""
    kw.setdefault("max_len", 12)
    return SeqTransformer(n_continuous=len(scaler.cols), n_binary=len(F.BINARY),
                          vocab_sizes=list(vocab.vocab_sizes), **kw)


def make_seq_predict():
    """``predict(model, batch, device) -> [N, 7]`` for a sequence batch dict
    (``cont``/``cat``/``bin`` ``[B,T,·]`` + ``lengths``) — identical to ``seq_model``'s, so the
    shared ``torch_common.weighted_ce`` / ``evaluate._nll`` path is reused unchanged."""
    def predict(model, batch, device):
        cont = torch.as_tensor(np.asarray(batch["cont"]), dtype=torch.float32, device=device)
        cat = torch.as_tensor(np.asarray(batch["cat"]), dtype=torch.long, device=device)
        binb = torch.as_tensor(np.asarray(batch["bin"]), dtype=torch.float32, device=device)
        lengths = torch.as_tensor(np.asarray(batch["lengths"]), dtype=torch.long)
        return model(cont, cat, binb, lengths)
    return predict
