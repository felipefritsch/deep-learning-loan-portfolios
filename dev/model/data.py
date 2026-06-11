"""M6 — Shard-streaming loader (``02_LOAN_LEVEL §3`` item 4, the May-8 scheme).

Every loan-month changes role across the 11 windows (test in k → train in k+2), so the
export wrote ONE shared train pool + ONE eval pool and the window split is applied here,
at load time, by a strict ``period_ym`` mask (``config.{train,val,test}_bounds``). The
loader then realises the paper's randomization:

    per epoch:  shuffle part(=shard) order
                → load one shard (masked, feature columns only)
                → shuffle rows in memory
                → yield minibatches

Because each part is a whole-loan shard, its rows span *all* vintages for that loan
subset, so a within-shard shuffle already mixes calendar time — a minibatch is never
drawn from a single vintage (§3). Train split streams ``train_pool``; val/test stream the
never-thinned ``eval_pool`` (identical frozen rows for every model). Batches are NumPy
(the feature pipeline in ``features.py``); M7+ wraps them in torch tensors.

``fit_window`` derives the per-window scaler + vocab from that window's train slice only
(no leakage), the same objects the loader then applies to every split. For the dev
variant the masked train slice's feature columns fit in RAM; the full variant (M10)
should swap the fit to streaming moments + DuckDB quantiles (the loader API is unchanged).

Run the smoke check (needs the SSD + the dev export):
    .venv/bin/python dev/model/data.py --variant dev --k 2015
Companion tests (hermetic, synthetic parquet): ``test_features.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

import config
import features as F

# Raw pool columns the loader must read (derived cats are computed by prepare_raw from
# period_ym/orig_ym; weight + keys ride along for the loss and eval alignment).
RAW_COLS: list[str] = sorted(set(
    F.CONTINUOUS + F.RAW_CATEGORICAL + F.BINARY
    + [F.TARGET_COL, F.WEIGHT_COL, "period_ym", "orig_ym"] + F.KEY_COLS
))


# ---------------------------------------------------------------------------
# Window resolution — which pool + which period_ym mask
# ---------------------------------------------------------------------------
def window_spec(variant: str, k: int, split: str) -> tuple[Path, tuple[int, int]]:
    """``(pool_dir, (lo, hi))`` for window ``k``'s ``split`` ∈ {train,val,test}. Train
    reads ``train_pool``; val/test read the frozen ``eval_pool`` (§3)."""
    design = config.TRAINING_DIR / variant
    bounds = {"train": config.train_bounds, "val": config.val_bounds,
              "test": config.test_bounds}[split](k)
    pool = "train_pool" if split == "train" else "eval_pool"
    return design / pool, bounds


def _masked_scan(pool_dir: Path, bounds: tuple[int, int]) -> pl.LazyFrame:
    lo, hi = bounds
    return (pl.scan_parquet(str(pool_dir / "part-*.parquet"))
              .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
              .select(RAW_COLS))


# ---------------------------------------------------------------------------
# Per-window fit — scaler + vocab from the train slice ONLY (§2, §6.4)
# ---------------------------------------------------------------------------
def fit_window(variant: str, k: int, min_count: int = 1) -> tuple[F.Scaler, F.Vocab]:
    """Fit the scaler + vocab on window ``k``'s train slice (``train_pool`` ∩ train
    mask). Dev-scale: the masked feature columns are collected once into RAM."""
    pool_dir, bounds = window_spec(variant, k, "train")
    df = F.prepare_raw(_masked_scan(pool_dir, bounds).collect())
    return F.Scaler.fit(df), F.Vocab.fit(df, min_count=min_count)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class WindowLoader:
    """Stream minibatches for one window/split. Iterating ``epoch`` reshuffles part order
    and in-shard rows deterministically from ``(seed, epoch)``."""

    def __init__(self, pool_dir: Path, bounds: tuple[int, int], scaler: F.Scaler,
                 vocab: F.Vocab, batch_size: int = 8192, encode: str = "index",
                 seed: int = 0, drop_last: bool = False, shuffle: bool = True):
        self.pool_dir = Path(pool_dir)
        self.bounds = bounds
        self.scaler, self.vocab = scaler, vocab
        self.batch_size = batch_size
        self.encode = encode
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.parts = sorted(self.pool_dir.glob("part-*.parquet"))

    def _load_part(self, part: Path) -> pl.DataFrame:
        lo, hi = self.bounds
        lf = (pl.scan_parquet(str(part))
                .filter((pl.col("period_ym") >= lo) & (pl.col("period_ym") < hi))
                .select(RAW_COLS))
        return F.prepare_raw(lf.collect())

    def iter_batches(self, epoch: int = 0):
        rng = np.random.default_rng([self.seed, epoch])
        order = rng.permutation(len(self.parts)) if self.shuffle else np.arange(len(self.parts))
        for pi in order:
            df = self._load_part(self.parts[pi])
            if df.height == 0:
                continue
            if self.shuffle:
                df = df[rng.permutation(df.height)]   # in-shard shuffle (mixes vintages)
            batch = F.encode_frame(df, self.scaler, self.vocab, self.encode)
            n = df.height
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                if self.drop_last and end > n:
                    break
                yield {kk: vv[start:end] for kk, vv in batch.items()}

    def __iter__(self):
        return self.iter_batches(0)


def loaders(variant: str, k: int, scaler: F.Scaler, vocab: F.Vocab, **kw):
    """Convenience: ``(train, val, test)`` WindowLoaders sharing one scaler/vocab. Val/
    test default to ``shuffle=False`` (deterministic, order-stable evaluation)."""
    def mk(split, **over):
        pool_dir, bounds = window_spec(variant, k, split)
        return WindowLoader(pool_dir, bounds, scaler, vocab, **{**kw, **over})
    return mk("train"), mk("val", shuffle=False), mk("test", shuffle=False)


# ---------------------------------------------------------------------------
# Smoke check (needs the SSD + dev export) — one masked shard through the pipeline
# ---------------------------------------------------------------------------
def _smoke(variant: str, k: int) -> None:
    config.require_drive()
    print(f"=== data.py smoke  variant={variant}  k={k} ===")
    scaler, vocab = fit_window(variant, k)
    print(f"scaler: {len(scaler.cols)} continuous cols "
          f"(log={scaler.log_cols}, robust={scaler.robust_cols})")
    print(f"vocab : {len(vocab.cols)} categoricals, sizes={dict(zip(vocab.cols, vocab.vocab_sizes))}")
    tr, va, te = loaders(variant, k, scaler, vocab, batch_size=8192)
    for name, ld in (("train", tr), ("val", va), ("test", te)):
        nb = nrow = wsum = 0
        ymin, ymax = 10**9, -1
        for b in ld.iter_batches(0):
            nb += 1
            nrow += b["y"].shape[0]
            wsum += float(b["w"].sum())
            ymin, ymax = min(ymin, b["period_ym"].min()), max(ymax, b["period_ym"].max())
            if nb == 1:
                print(f"  [{name}] first batch: cont{b['cont'].shape} cat{b['cat'].shape} "
                      f"bin{b['bin'].shape} y{b['y'].shape}  y∈[{b['y'].min()},{b['y'].max()}]")
        lo, hi = ld.bounds
        print(f"  [{name}] {nb} batches  {nrow:,} rows  Σw={wsum:,.0f}  "
              f"period_ym∈[{ymin},{ymax}] ⊂ [{lo},{hi})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke-check the shard-streaming loader (M6).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--k", type=int, default=config.TUNING_YEAR)
    args = ap.parse_args()
    _smoke(args.variant, args.k)


if __name__ == "__main__":
    main()
