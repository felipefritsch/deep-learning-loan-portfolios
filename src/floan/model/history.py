"""M26a — history-summary features for the Markov-assumption probe (``04 §M26a``).

The cheapest test of path-dependence: feed four *causal* history summaries into the
**existing** feed-forward net and ask whether val-NLL improves beyond seed noise. No
sequence model, no new model class. The summaries are derived **per loan-month** from the
panel (the only complete monthly state series — ``train_pool`` thins current→current rows
and ``eval_pool`` floors the label month, so neither can reconstruct a loan's history):

  * ``prev_state``                 — state at the immediately preceding record (S_{t−1})
  * ``months_since_last_delinq``   — months back to the most recent delinquency before t
  * ``ever_delinquent``            — 1 if the loan was ever delinquent before t
  * ``n_prior_delinq_episodes``    — # delinquency episodes that began before t

**Leakage guard (the whole point).** Every summary is a function of rows with
``period < t`` ONLY, and never reads ``state_next`` (the target):

  1. the aggregates run over the window frame ``ROWS BETWEEN UNBOUNDED PRECEDING AND 1
     PRECEDING`` — a *structural* end-before-t, not a hand-applied filter, so row t's own
     state can never enter its own features; ``prev_state = LAG(state)`` is strictly t−1;
  2. the SQL references only ``state`` (last month's *reported* status, known at t) and
     ``period`` — ``state_next`` appears nowhere, so no outcome contamination;
  3. the features attach to model rows by exact ``(Loan Identifier, period_ym)``, so the
     existing window masks still bound every row to history through t−1 (mask-invariant).

``DELINQ_STATES`` is the DPD block only (the spec's "delinquency episodes"); ``prev_state``
still records the *full* prior state (incl. foreclosure). The derivation is written once as
DuckDB window functions (out-of-core over the ~78 M-row dev block) and cached to a compact
``(loan, period_ym)``-keyed parquet that the augmented loader left-joins at load time.

Run:
    .venv/bin/python -m floan.model.history build       --variant dev --k 2015
    .venv/bin/python -m floan.model.history spot-check   --variant dev --k 2015
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import polars as pl

from floan.model import config
from floan.model import data as D
from floan.model import export  # reuse connect(): panel view + memory limit + temp spill
from floan.model import features as F

# The delinquency states that define an "episode" / recency / ever-flag (DPD block only —
# 04 §M26a "delinquency episodes"; foreclosure/REO are terminal credit events kept out so
# the signal stays delinquency-specific). prev_state records the full prior state regardless.
DELINQ_STATES: tuple[str, ...] = ("dpd_30", "dpd_60", "dpd_90plus")

# History feature membership, by model block (the augmented additions to features.py's
# CONTINUOUS / CATEGORICAL / BINARY). prev_state embeds like every other categorical;
# the two counts standardize (mean/std); ever_delinquent passes through 0/1.
HIST_CATEGORICAL: list[str] = ["prev_state"]
HIST_CONTINUOUS: list[str] = ["months_since_last_delinq", "n_prior_delinq_episodes"]
HIST_BINARY: list[str] = ["ever_delinquent"]
HIST_COLS: list[str] = HIST_CATEGORICAL + HIST_BINARY + HIST_CONTINUOUS

# Augmented feature blocks (current-state-only block ‖ history block) — the augmented net's
# Scaler/Vocab fit over these; n_binary = len(F.BINARY) + len(HIST_BINARY).
AUG_CONTINUOUS: list[str] = F.CONTINUOUS + HIST_CONTINUOUS
AUG_CATEGORICAL: list[str] = F.CATEGORICAL + HIST_CATEGORICAL
AUG_BINARY: list[str] = F.BINARY + HIST_BINARY

_CACHE: dict[tuple[str, int], pl.DataFrame] = {}


# ---------------------------------------------------------------------------
# The causal derivation — one window-function SELECT, reused for build + flip-test
# ---------------------------------------------------------------------------
def _history_select_sql(source: str, shard_pred: str, period_hi: int | None) -> str:
    """The leakage-safe history SELECT over ``source`` (a table/view with the panel
    columns). ``shard_pred`` restricts the loan block; ``period_hi`` (or ``None``) trims the
    OUTPUT period_ym (applied AFTER the windows, so the per-loan history stays complete).

    The two windows: ``w`` orders a loan by period (for LAG / episode entry); ``wpre`` is the
    SAME order but its frame ENDS AT ``1 PRECEDING`` — the structural "period < t" guard for
    every aggregate. ``state_next`` is referenced nowhere."""
    delinq = "state IN " + str(DELINQ_STATES)
    where_hi = "" if period_hi is None else f"WHERE period_ym < {period_hi}"
    return f"""
    WITH base AS (
        SELECT "Loan Identifier" AS loan, period, period_ym, state,
               (year(period) * 12 + month(period) - 1) AS mi,
               CASE WHEN {delinq} THEN 1 ELSE 0 END AS is_delinq
        FROM {source}
        WHERE {shard_pred}
    ),
    seq AS (
        SELECT *,
               LAG(state) OVER w AS prev_state,
               CASE WHEN is_delinq = 1 THEN mi END AS delinq_mi,
               CASE WHEN is_delinq = 1 AND COALESCE(LAG(is_delinq) OVER w, 0) = 0
                    THEN 1 ELSE 0 END AS ep_start
        FROM base
        WINDOW w AS (PARTITION BY loan ORDER BY period)
    ),
    hist AS (
        SELECT loan AS "Loan Identifier", period_ym, prev_state,
               CAST(COALESCE(MAX(is_delinq) OVER wpre, 0) AS TINYINT) AS ever_delinquent,
               (mi - MAX(delinq_mi) OVER wpre) AS months_since_last_delinq,
               CAST(COALESCE(SUM(ep_start) OVER wpre, 0) AS INTEGER) AS n_prior_delinq_episodes
        FROM seq
        WINDOW wpre AS (PARTITION BY loan ORDER BY period
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    )
    SELECT "Loan Identifier", period_ym, prev_state, ever_delinquent,
           months_since_last_delinq, n_prior_delinq_episodes
    FROM hist {where_hi}
    """


def _hist_dir(variant: str, k: int) -> Path:
    return config.TRAINING_DIR / variant / "history" / f"k{k}"


def _esc(p: Path) -> str:
    return str(p).replace("'", "''")


# ---------------------------------------------------------------------------
# Build / load / join
# ---------------------------------------------------------------------------
def build(variant: str = "dev", k: int = config.TUNING_YEAR, rebuild: bool = False) -> Path:
    """Derive the history cache for window ``k``'s loan block, **one shard at a time**, into
    ``(loan, period_ym)`` part files. Per-shard is exact (shard = hash(loan) % N_SHARDS is
    loan-keyed, so a loan's full history lives in a single shard ⇒ ``PARTITION BY loan`` is
    complete within the shard) AND memory-bounded (the all-shards windowed COPY OOMs the
    DuckDB limit). Output trimmed to ``period_ym < test_bounds(k)[1]`` (k's train/val/test
    feature months); the windows still see each loan's full history."""
    out_dir = _hist_dir(variant, k)
    if list(out_dir.glob("part-*.parquet")) and not rebuild:
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    shard_lt = config.VARIANTS[variant]["train_shard_lt"]
    period_hi = config.test_bounds(k)[1]
    con = export.connect()      # panel view + memory_limit + temp_directory (out-of-core)
    con.execute("SET threads=4")                          # bound peak memory for the window
    t0, total = time.perf_counter(), 0
    print(f"=== history.build  variant={variant} k={k}  shards 0..{shard_lt - 1}  "
          f"period_ym<{period_hi} ===")
    for s in range(shard_lt):
        part = out_dir / f"part-{s:04d}.parquet"
        sql = _history_select_sql("panel", f"shard = {s}", period_hi)
        con.execute(f"COPY ({sql}) TO '{_esc(part)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = con.execute(f"SELECT count(*) FROM read_parquet('{_esc(part)}')").fetchone()[0]
        total += n
        print(f"  shard {s:>2}: {n:>10,} rows -> {part.name}")
    con.close()
    print(f"wrote {out_dir}  ({total:,} loan-month rows, {time.perf_counter() - t0:.0f}s)")
    return out_dir


def load(variant: str = "dev", k: int = config.TUNING_YEAR) -> pl.DataFrame:
    """Read (and memoize) the history cache; build it first via ``history build``."""
    key = (variant, k)
    if key not in _CACHE:
        parts = sorted(_hist_dir(variant, k).glob("part-*.parquet"))
        if not parts:
            raise FileNotFoundError(
                f"history cache missing under {_hist_dir(variant, k)} — run `python -m "
                f"floan.model.history build --variant {variant} --k {k}` first")
        _CACHE[key] = pl.read_parquet([str(p) for p in parts])
    return _CACHE[key]


def join_history(df: pl.DataFrame, variant: str, k: int) -> pl.DataFrame:
    """Left-join the four history columns onto a prepared model frame by the exact
    ``(Loan Identifier, period_ym)`` key (1:1 — one panel record per loan-month). Any
    unmatched row keeps nulls, which the downstream encoders map safely (prev_state → UNK,
    ever_delinquent → 0, the two counts → scaler centre).

    The cache (right side, ~38 M rows) is **scanned lazily and joined under the streaming
    engine**, so no full history frame is held resident — peak memory stays bounded across
    the probe's repeated fits/scores (the all-in-RAM join swaps the Mac by the later runs)."""
    parts = sorted(_hist_dir(variant, k).glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"history cache missing under {_hist_dir(variant, k)} — run `python -m "
            f"floan.model.history build --variant {variant} --k {k}` first")
    hist = pl.scan_parquet([str(p) for p in parts])
    # The streaming hash-join does NOT preserve left-row order, so carry an explicit row
    # index and sort back — callers that pair the joined features with a separately-ordered
    # target/origin array (e.g. seq_spike's full-val scoring) would otherwise misalign.
    return (df.lazy().with_row_index("__ord")
              .join(hist, on=["Loan Identifier", "period_ym"], how="left")
              .collect(engine="streaming")
              .sort("__ord").drop("__ord"))


# ---------------------------------------------------------------------------
# Augmented per-window fit — scaler + vocab over the extended blocks (train slice only)
# ---------------------------------------------------------------------------
def fit_window_aug(variant: str, k: int, min_count: int = 1) -> tuple[F.Scaler, F.Vocab]:
    """The ``data.fit_window`` analogue for the augmented net: fit the Scaler over
    ``AUG_CONTINUOUS`` and the Vocab over ``AUG_CATEGORICAL`` on window ``k``'s train slice
    (with history joined) — train-only, no leakage, same objects the loader then applies."""
    pool_dir, bounds = D.window_spec(variant, k, "train")
    df = join_history(F.prepare_raw(D._masked_scan(pool_dir, bounds).collect()), variant, k)
    return (F.Scaler.fit(df, cols=AUG_CONTINUOUS),
            F.Vocab.fit(df, cols=AUG_CATEGORICAL, min_count=min_count))


# ---------------------------------------------------------------------------
# Leakage spot-check — one concrete loan-month + the flip-state_next proof
# ---------------------------------------------------------------------------
def _reference(loan_rows: pl.DataFrame, t_ym: int) -> dict:
    """Readable Polars oracle for the four features at feature-month ``t_ym`` of one loan,
    using ONLY records strictly before t (the independent cross-check of the SQL)."""
    d = loan_rows.sort("period_ym")
    pys = d.get_column("period_ym").to_list()
    states = d.get_column("state").to_list()
    i = pys.index(t_ym)
    prior_states, prior_pys = states[:i], pys[:i]
    mi = lambda ym: (ym // 100) * 12 + (ym % 100 - 1)
    last = [mi(p) for p, s in zip(prior_pys, prior_states) if s in DELINQ_STATES]
    episodes = was = 0
    for s in prior_states:
        now = s in DELINQ_STATES
        episodes += int(now and not was)
        was = now
    return {"prev_state": states[i - 1] if i > 0 else None,
            "ever_delinquent": int(any(s in DELINQ_STATES for s in prior_states)),
            "months_since_last_delinq": (mi(t_ym) - max(last)) if last else None,
            "n_prior_delinq_episodes": episodes}


def spot_check(variant: str = "dev", k: int = config.TUNING_YEAR,
               loan: str | None = None, period_ym: int | None = None) -> dict:
    """Print + assert the leakage guard on one concrete loan-month:
      (a) show the loan's pre-t (period_ym, state) records — the only allowed inputs;
      (b) the four cached features equal the independent pre-t oracle;
      (c) FLIP every ``state_next`` for the loan and re-derive through the SAME SQL —
          the four features are byte-identical (empirical proof state_next is never read).
    """
    config.require_drive()
    hist = load(variant, k)
    if loan is None or period_ym is None:
        pick = (hist.filter((pl.col("ever_delinquent") == 1)
                            & (pl.col("n_prior_delinq_episodes") >= 1)
                            & (pl.col("months_since_last_delinq") >= 2))
                    .head(1))
        if pick.height == 0:
            raise RuntimeError("no delinquent-history loan-month found in the cache")
        loan = pick.get_column("Loan Identifier").item()
        period_ym = int(pick.get_column("period_ym").item())

    # Pull the loan's full panel series (the inputs); state_next carried only for the flip.
    con = export.connect()
    loan_rows = con.execute(
        'SELECT "Loan Identifier", period, period_ym, state, state_next '
        'FROM panel WHERE "Loan Identifier" = ? ORDER BY period', [loan]).pl()
    con.close()

    print(f"=== leakage spot-check  variant={variant} k={k}  loan={loan}  t={period_ym} ===")
    pre = loan_rows.filter(pl.col("period_ym") <= period_ym).select(
        "period_ym", "state", "state_next")
    print("loan records up to and including t (only period < t may feed the features):")
    print(pre)

    cached = (hist.filter((pl.col("Loan Identifier") == loan)
                          & (pl.col("period_ym") == period_ym))
                  .select(HIST_COLS).row(0, named=True))
    expect = _reference(loan_rows, period_ym)
    print(f"cached   : {cached}")
    print(f"oracle   : {expect}")
    assert cached["prev_state"] == expect["prev_state"], "prev_state mismatch"
    assert int(cached["ever_delinquent"]) == expect["ever_delinquent"], "ever_delinquent mismatch"
    assert (cached["months_since_last_delinq"] == expect["months_since_last_delinq"]), \
        "months_since_last_delinq mismatch"
    assert int(cached["n_prior_delinq_episodes"]) == expect["n_prior_delinq_episodes"], \
        "n_prior_delinq_episodes mismatch"

    # (c) Flip ALL state_next for this loan; re-derive via the SAME SQL; assert unchanged.
    src = loan_rows.select("Loan Identifier", "period", "period_ym", "state")
    orig = src.with_columns(pl.col("state").alias("state_next"))           # arbitrary value
    flip = src.with_columns(pl.lit("prepaid").alias("state_next"))         # all flipped
    c = duckdb.connect()
    c.register("orig", orig.to_arrow())
    c.register("flip", flip.to_arrow())
    r_orig = c.execute(_history_select_sql("orig", "TRUE", None)).pl().sort("period_ym")
    r_flip = c.execute(_history_select_sql("flip", "TRUE", None)).pl().sort("period_ym")
    c.close()
    flip_ok = r_orig.equals(r_flip)
    assert flip_ok, "FLIP TEST FAILED — features changed when state_next changed (leakage!)"
    print(f"flip-state_next test: PASS — all {r_orig.height} rows' features identical "
          f"under flipped state_next (state_next is never read)")
    print("LEAKAGE SPOT-CHECK PASS")
    return {"loan": loan, "period_ym": period_ym, "cached": cached, "oracle": expect,
            "flip_state_next_unchanged": bool(flip_ok)}


def main() -> None:
    ap = argparse.ArgumentParser(description="M26a history-summary features (build / spot-check).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="derive + cache the history features")
    b.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    b.add_argument("--k", type=int, default=config.TUNING_YEAR)
    b.add_argument("--rebuild", action="store_true")
    s = sub.add_parser("spot-check", help="leakage spot-check on one loan-month")
    s.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    s.add_argument("--k", type=int, default=config.TUNING_YEAR)
    s.add_argument("--loan", default=None)
    s.add_argument("--period-ym", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.variant, args.k, rebuild=args.rebuild)
    else:
        spot_check(args.variant, args.k, loan=args.loan, period_ym=args.period_ym)


if __name__ == "__main__":
    main()
