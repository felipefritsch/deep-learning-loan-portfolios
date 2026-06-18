"""Stage 5 — DuckDB views, sampling & training helpers  (s5_sample.py).

Query the panel lake out-of-core and draw RAM-sized, model-ready samples — never
loading the whole 3.31 B-row panel. Also provides the training-time pieces from
``01_SCHEMA.md`` §6: shard iteration, leakage-safe time masking, and train-only
feature scalers (the lake itself is never standardised).

The base training predicate bakes in the Stage-6 QA findings — a usable
supervised example needs a known current state AND next state, and must not be a
dirty "reported before origination" row::

    state IS NOT NULL AND state_next IS NOT NULL AND orig_ym <= period_ym

Key helpers
-----------
* ``transition_matrix(where)``      — 7×7 counts over any slice
* ``balanced_sample(per_class, cutoff_ym)`` — class-balanced sample on
  ``state_next`` (keeps ALL rare foreclosure/REO transitions, downsamples
  ``current``) via one-pass per-class Bernoulli sampling
* ``slice_quarter(q)`` / ``iter_shards(shards)`` — quarter / shard readers
  (``WHERE shard = k`` skips row groups — no full scan)
* ``training_mask(cutoff_ym)``      — strict ``period_ym < cutoff_ym`` (the label
  is one month ahead, so the cut must exclude the labelled month too)
* ``fit_scaler(cutoff_ym)`` / ``apply_scaler(df, scaler)`` — train-slice-only
  center/scale (robust median/IQR on log1p for skewed dollars, else mean/std),
  persisted to ``models/scaler_<cutoff>.json``
* ``materialize(df, name)``         — cache a sample on the fast internal disk

Read-only over the lake; never writes to ``raw/``.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import duckdb

from floan.pipeline import config
from floan.pipeline import schema

# Internal-disk working-set cache (fast NVMe) — keep training samples here so the
# model loop never streams the USB lake every epoch.
SAMPLE_CACHE = config.DUCKDB_PATH.parent / "samples"

# Stage-6 QA filters: a valid supervised example.
BASE_TRAIN = ("state IS NOT NULL AND state_next IS NOT NULL "
              "AND orig_ym <= period_ym")

STATES = ("current", "dpd_30", "dpd_60", "dpd_90plus", "foreclosure", "REO", "prepaid")


def _qcol(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def connect() -> duckdb.DuckDBPyConnection:
    """Open the disposable internal-disk catalog and expose the panel as ``panel``."""
    config.require_drive()
    con = duckdb.connect(str(config.DUCKDB_PATH))
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    g = f"{config.PANEL_DIR}/acq_quarter=*/part.parquet"
    con.execute(f"CREATE OR REPLACE VIEW panel AS "
                f"SELECT * FROM read_parquet('{g}', hive_partitioning=true)")
    return con


def training_mask(cutoff_ym: "int | None") -> str:
    """Leakage-safe filter: include an example only if its LABELLED month
    (period+1) is still ≤ cutoff, i.e. ``period_ym < cutoff_ym`` (strict)."""
    return f"period_ym < {int(cutoff_ym)}" if cutoff_ym else "TRUE"


def _where(cutoff_ym=None, extra=None) -> str:
    parts = [BASE_TRAIN, training_mask(cutoff_ym)]
    if extra:
        parts.append(f"({extra})")
    return " AND ".join(p for p in parts if p and p != "TRUE")


def transition_matrix(con, cutoff_ym=None, extra=None) -> pl.DataFrame:
    """7-state × 7-state transition counts over the (filtered) panel."""
    df = con.execute(
        f"SELECT state, state_next, count(*) AS n FROM panel "
        f"WHERE {_where(cutoff_ym, extra)} GROUP BY 1, 2"
    ).pl()
    return df


def balanced_sample(con, per_class: int = 1_000_000, cutoff_ym=None,
                    extra=None, seed: float = 0.42) -> pl.DataFrame:
    """Class-balanced sample on ``state_next``: up to ``per_class`` rows per target
    class (ALL rows of rarer classes), via one-pass per-class Bernoulli sampling.

    Preserves every foreclosure/REO transition; downsamples ``current``. One scan,
    no sort. Returns a Polars frame sized to ~ ``per_class × #classes``.
    """
    where = _where(cutoff_ym, extra)
    counts = dict(con.execute(
        f"SELECT state_next, count(*) FROM panel WHERE {where} GROUP BY 1").fetchall())
    fracs = {s: min(1.0, per_class / c) for s, c in counts.items() if c}
    case = " ".join(f"WHEN '{s}' THEN {f}" for s, f in fracs.items())
    con.execute(f"SELECT setseed({seed})")
    return con.execute(
        f"SELECT * FROM panel WHERE {where} "
        f"AND random() < (CASE state_next {case} ELSE 0 END)"
    ).pl()


def slice_quarter(con, quarter: str, cutoff_ym=None, extra=None) -> pl.DataFrame:
    return con.execute(
        f"SELECT * FROM panel WHERE acq_quarter = '{quarter}' "
        f"AND {_where(cutoff_ym, extra)}"
    ).pl()


def iter_shards(con, shards=None, cutoff_ym=None, extra=None):
    """Yield (shard, frame) one shard at a time. ``WHERE shard = k`` skips row
    groups (the panel is sorted by shard) — no full-panel scan per shard."""
    if shards is None:
        shards = range(config.N_SHARDS)
    for k in shards:
        df = con.execute(
            f"SELECT * FROM panel WHERE shard = {int(k)} AND {_where(cutoff_ym, extra)}"
        ).pl()
        yield k, df


# ---------------------------------------------------------------------------
# Train-only feature scaler (never standardise the lake; §6.4)
# ---------------------------------------------------------------------------
def fit_scaler(con, cutoff_ym=None, extra=None, save: bool = True) -> dict:
    """Center/scale for each continuous feature, computed over the TRAIN slice
    only (one DuckDB pass). Robust median/IQR on ``log1p`` for skewed dollar
    columns; mean/std otherwise. Persisted to ``models/scaler_<cutoff>.json``."""
    where = _where(cutoff_ym, extra)
    cont = schema.FEATURE_SPEC["continuous_standardize"]
    parts = []
    for i, c in enumerate(cont):
        x = f"ln(1 + {_qcol(c)}::DOUBLE)" if c in schema.LOG_TRANSFORM else f"{_qcol(c)}::DOUBLE"
        parts += [f"avg({x}) AS mean_{i}", f"stddev_samp({x}) AS std_{i}",
                  f"approx_quantile({x},0.25) AS q25_{i}",
                  f"approx_quantile({x},0.50) AS q50_{i}",
                  f"approx_quantile({x},0.75) AS q75_{i}"]
    res = con.execute(f"SELECT {', '.join(parts)} FROM panel WHERE {where}")
    row = dict(zip([d[0] for d in res.description], res.fetchone()))

    scaler = {"cutoff_ym": cutoff_ym, "fitted_at": datetime.now().isoformat(timespec="seconds"),
              "where": where, "columns": {}}
    for i, c in enumerate(cont):
        log = c in schema.LOG_TRANSFORM
        if log:
            center, scale, method = row[f"q50_{i}"], (row[f"q75_{i}"] - row[f"q25_{i}"]), "log1p+median/IQR"
        else:
            center, scale, method = row[f"mean_{i}"], row[f"std_{i}"], "mean/std"
        scale = scale if scale and abs(scale) > 1e-9 else 1.0
        scaler["columns"][c] = {"method": method, "log": log,
                                "center": float(center) if center is not None else 0.0,
                                "scale": float(scale)}
    if save:
        config.MODELS.mkdir(parents=True, exist_ok=True)
        tag = cutoff_ym if cutoff_ym else "full"
        p = config.MODELS / f"scaler_{tag}.json"
        p.write_text(json.dumps(scaler, indent=2))
        scaler["_path"] = str(p)
    return scaler


def apply_scaler(df: pl.DataFrame, scaler: dict, fill_null: bool = True) -> pl.DataFrame:
    """Standardise the continuous columns of ``df`` with a fitted scaler
    (``log1p`` then (x−center)/scale). Categorical/binary/target untouched.
    Standardised nulls → 0 (the mean) when ``fill_null`` — pair with the
    ``*_missing`` flags so absence stays informative."""
    exprs = []
    for c, p in scaler["columns"].items():
        if c not in df.columns:
            continue
        x = pl.col(c).cast(pl.Float64)
        if p["log"]:
            x = (1 + x).log()
        z = (x - p["center"]) / p["scale"]
        if fill_null:
            z = z.fill_null(0.0)
        exprs.append(z.alias(c))
    return df.with_columns(exprs)


def materialize(df: pl.DataFrame, name: str, to_ssd: bool = False) -> Path:
    """Cache a sample as Parquet — on the fast internal disk by default (so the
    training loop never streams USB), or on the SSD (``processed/samples``)."""
    out_dir = config.SAMPLE_DIR if to_ssd else SAMPLE_CACHE
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.parquet"
    df.write_parquet(p, compression="zstd")
    return p


def _print_matrix(df: pl.DataFrame, title: str) -> None:
    m = {(r["state"], r["state_next"]): r["n"] for r in df.to_dicts()}
    print(f"\n{title}")
    print("            " + "".join(f"{s[:8]:>11}" for s in STATES) + f"{'(end)':>11}")
    for a in STATES:
        line = f"{a:>10}" + "".join(f"{m.get((a, b), 0):>11,}" for b in STATES)
        line += f"{m.get((a, None), 0):>11,}"
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 5 — sampling & training helpers (demo)")
    ap.add_argument("--per-class", type=int, default=1_000_000,
                    help="target rows per state_next class in the balanced sample")
    ap.add_argument("--cutoff-ym", type=int, default=None,
                    help="rolling-backtest cut-off YYYYMM (train on period_ym < cutoff)")
    ap.add_argument("--quarter", help="quick test: build the sample within one quarter only")
    args = ap.parse_args()

    con = connect()
    extra = f"acq_quarter = '{args.quarter}'" if args.quarter else None

    t0 = time.time()
    samp = balanced_sample(con, per_class=args.per_class, cutoff_ym=args.cutoff_ym, extra=extra)
    dt = time.time() - t0
    print(f"balanced_sample: {samp.height:,} rows in {dt:.1f}s "
          f"(per_class={args.per_class:,}, cutoff={args.cutoff_ym})")
    by = samp.group_by("state_next").len().sort("len", descending=True)
    print("state_next balance:", {r["state_next"]: r["len"] for r in by.to_dicts()})
    _print_matrix(transition_matrix(con, cutoff_ym=args.cutoff_ym, extra=extra),
                  "transition matrix (sampled slice source):")

    t1 = time.time()
    scaler = fit_scaler(con, cutoff_ym=args.cutoff_ym, extra=extra)
    print(f"\nfit_scaler: {len(scaler['columns'])} continuous cols in {time.time()-t1:.1f}s "
          f"-> {scaler.get('_path')}")
    scaled = apply_scaler(samp, scaler)
    # show a couple of standardised columns are ~0 mean / unit-ish scale on the sample
    chk = scaled.select([
        pl.col("Original Interest Rate").mean().alias("rate_mean"),
        pl.col("Original UPB").std().alias("logupb_std"),
    ]).to_dicts()[0]
    print("apply_scaler sanity (on sample):", {k: round(v, 3) for k, v in chk.items()})

    p = materialize(scaled, "demo_balanced_sample")
    print(f"materialized scaled sample -> {p}")
    con.close()
