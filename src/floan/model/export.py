"""M4 — Build the shared training export (``02_LOAN_LEVEL §3``).

One DuckDB pass per shard turns the Stage-4 panel into two Parquet pools under
``processed/training/<variant>/`` that the rolling backtest (M5+) masks by window:

  train_pool/  thinned + weighted, ALL label years. Keeps every row with
               state_next≠current OR state≠current; keeps current→current rows with
               probability ``P_KEEP_CURRENT`` (deterministic hash of loan+period) and
               attaches ``weight = 1/p_keep`` so the loss is an unbiased estimate of
               the true conditional probabilities (00_OVERVIEW §6.4).
  eval_pool/   NEVER thinned; a fixed, loan-disjoint shard block (shard < eval_shard_lt)
               of every loan-month with label month ≥ EVAL_LABEL_YM_MIN. Each window's
               val/test slice is cut from this pool by period_ym mask, identical across
               all models.

Both pools carry the raw (unstandardized) panel features under their schema names,
the macro features joined with per-variable lags (``macro_features.py``), and the
``mkt_rate`` proxy — scalers/vocabularies stay train-only at model time (M6), never
baked into the lake. The panel is sorted by ``(shard, loan, period)``, so the per-shard
WHERE skips all non-matching row groups: the dev export reads ~3% of the lake in
seconds. Processing one shard at a time bounds memory for the macro join.

Run:
    .venv/bin/python dev/model/export.py                 # dev variant (default)
    .venv/bin/python dev/model/export.py --variant dev
    .venv/bin/python dev/model/export.py --verify-only   # re-run Accept checks on existing output
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import polars as pl

import config  # dev/model/config.py — also puts dev/pipeline on sys.path
import macro_features as mf
import schema  # pipeline schema — FEATURE_SPEC column names (single source of truth)

_REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Export schema — built from the pipeline FEATURE_SPEC so it can't drift.
# ---------------------------------------------------------------------------
_FS = schema.FEATURE_SPEC
# Panel model features (raw values); 'excluded'/'target_driver'/extra date columns
# are intentionally left out (Unscheduled Principal Current is 100% null, etc.).
PANEL_FEATURE_COLS: list[str] = (
    _FS["continuous_standardize"] + _FS["categorical"] + _FS["binary"]
)
KEY_COLS = ["Loan Identifier", "shard", "period", "period_ym", "orig_ym"]
TARGET_COLS = ["state", "state_next", "censored"]
SELECT_COLS = KEY_COLS + PANEL_FEATURE_COLS + TARGET_COLS

# Final on-disk column order (identical for every part): keys, label/weight, target,
# panel features, macro features, mkt_rate proxy.
OUTPUT_COLS = (
    KEY_COLS + ["label_ym", "weight"] + TARGET_COLS
    + PANEL_FEATURE_COLS + mf.MACRO_FEATURE_COLS + ["mkt_rate"]
)

# Pure-macro columns that MUST be null-free (full source coverage). The derived
# incentive/ltv_mtm may inherit panel missingness (current_rate/current_upb) and are
# reported, not asserted (00_OVERVIEW: handled by the missingness indicators).
MACRO_NONNULL = ["pmms30", "dgs10", "slope_10y2y", "unrate_nat", "unrate_state",
                 "hpi_chg_12m"]

_THR = int(config.P_KEEP_CURRENT * 1_000_000)           # thinning hash threshold
_ORIGIN_SQL = "state IN " + str(config.ORIGIN_STATES)   # ('current','dpd_30',...)
_CUR2CUR = "state = 'current' AND state_next = 'current'"
_LABEL_YM = config.LABEL_YM_SQL.format(c="period_ym")   # period_ym + 1 calendar month


def _q(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DuckDB
# ---------------------------------------------------------------------------
def connect() -> duckdb.DuckDBPyConnection:
    config.require_drive()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    tmp = _REPO / "reports" / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(tmp).replace(chr(39), chr(39) * 2)}'")
    con.execute("CREATE OR REPLACE VIEW panel AS "
                f"SELECT * FROM read_parquet('{config.PANEL_GLOB}', hive_partitioning=true)")
    return con


def _shard_sql(pool: str, shard: int) -> str:
    cols = ", ".join(_q(c) for c in SELECT_COLS)
    if pool == "train":
        extra = f"({_LABEL_YM}) AS label_ym, " \
                f"CAST(CASE WHEN {_CUR2CUR} THEN {1.0 / config.P_KEEP_CURRENT} ELSE 1.0 END AS DOUBLE) AS weight"
        keep = (f"NOT({_CUR2CUR}) OR "
                f"(hash(\"Loan Identifier\" || '|' || period_ym::VARCHAR) % 1000000) < {_THR}")
        where = f"shard = {shard} AND {_ORIGIN_SQL} AND state_next IS NOT NULL AND ({keep})"
    else:  # eval — never thinned, label-month floor
        extra = f"({_LABEL_YM}) AS label_ym, CAST(1.0 AS DOUBLE) AS weight"
        where = (f"shard = {shard} AND {_ORIGIN_SQL} AND state_next IS NOT NULL "
                 f"AND ({_LABEL_YM}) >= {config.EVAL_LABEL_YM_MIN}")
    return (f"SELECT {cols}, {extra} FROM panel WHERE {where} "
            f'ORDER BY "Loan Identifier", period')


# ---------------------------------------------------------------------------
# Macro attachment (transient rename to macro_features' canonical names + back)
# ---------------------------------------------------------------------------
def attach_macro(df: pl.DataFrame, nat: pl.DataFrame, state: pl.DataFrame,
                 mkt: pl.DataFrame) -> pl.DataFrame:
    canon = mf.PANEL_ALIASES                        # {verbose panel name: canonical}
    df = df.rename(canon)
    df = mf.attach_macro(df, nat, state).drop(mf.MACRO_FEATURE_SPEC["intermediate"])
    df = df.rename({v: k for k, v in canon.items()})
    df = df.join(mkt.select("month_ym", "mkt_rate"),
                 left_on="period_ym", right_on="month_ym", how="left")
    return df.select(OUTPUT_COLS)


def _load_macro() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    nat = pl.read_parquet(config.MACRO_DIR / "macro_national.parquet")
    state = pl.read_parquet(config.MACRO_DIR / "macro_state.parquet")
    mkt = pl.read_parquet(config.MACRO_DIR / "mkt_rate.parquet")
    return nat, state, mkt


def _write_pool(con, nat, state, mkt, pool: str, shards, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):      # idempotent: clear stale parts
        old.unlink()
    for s in shards:
        df = con.execute(_shard_sql(pool, s)).pl()
        df = attach_macro(df, nat, state, mkt)
        df.write_parquet(out_dir / f"part-{s:04d}.parquet", compression="zstd",
                         compression_level=config.ZSTD_LEVEL, row_group_size=300_000)
        print(f"  {pool}_pool shard {s:>3}: {df.height:>10,} rows -> part-{s:04d}.parquet")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def _pool_glob(out_dir: Path) -> str:
    return f"{out_dir}/part-*.parquet"


def _label_year_counts(con, out_dir: Path) -> dict[str, int]:
    rows = con.execute(
        f"SELECT label_ym // 100 AS ly, count(*) AS n "
        f"FROM read_parquet('{_pool_glob(out_dir)}') GROUP BY 1 ORDER BY 1"
    ).fetchall()
    return {str(int(ly)): int(n) for ly, n in rows}


def _content_hash(con, out_dir: Path) -> str:
    h = con.execute(
        "SELECT bit_xor(hash(\"Loan Identifier\" || '|' || period_ym::VARCHAR "
        "|| '|' || state_next)) FROM read_parquet('" + _pool_glob(out_dir) + "')"
    ).fetchone()[0]
    return str(h)


def _build_manifest(con, variant: str, train_lt: int, eval_lt: int, design_dir: Path) -> dict:
    train_dir, eval_dir = design_dir / "train_pool", design_dir / "eval_pool"
    train_n = con.execute(
        f"SELECT count(*), sum(weight) FROM read_parquet('{_pool_glob(train_dir)}')").fetchone()
    cur_kept = con.execute(
        f"SELECT count(*) FROM read_parquet('{_pool_glob(train_dir)}') "
        "WHERE state='current' AND state_next='current'").fetchone()[0]
    eval_n = con.execute(
        f"SELECT count(*) FROM read_parquet('{_pool_glob(eval_dir)}')").fetchone()[0]
    return {
        "design": variant,
        "variant": variant,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_panel_glob": config.PANEL_GLOB,
        "macro_dir": str(config.MACRO_DIR),
        "params": {
            "p_keep_current": config.P_KEEP_CURRENT,
            "train_shard_lt": train_lt,
            "eval_shard_lt": eval_lt,
            "eval_label_ym_min": config.EVAL_LABEL_YM_MIN,
            "n_shards": config.N_SHARDS,
            "test_years": config.TEST_YEARS,
        },
        "train_pool": {
            "rows": int(train_n[0]),
            "weighted_rows": float(train_n[1]),
            "cur_to_cur_kept": int(cur_kept),
            "rows_by_label_year": _label_year_counts(con, train_dir),
            "content_hash": _content_hash(con, train_dir),
        },
        "eval_pool": {
            "rows": int(eval_n),
            "rows_by_label_year": _label_year_counts(con, eval_dir),
            "content_hash": _content_hash(con, eval_dir),
        },
        "output_columns": OUTPUT_COLS,
        "macro_feature_cols": mf.MACRO_FEATURE_COLS,
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(variant: str) -> Path:
    v = config.VARIANTS[variant]
    train_lt, eval_lt = v["train_shard_lt"], v["eval_shard_lt"]
    assert eval_lt <= train_lt, (
        "eval block must be ⊆ the train block (eval loans live in the train pool, "
        "separated only by the period_ym mask)")
    train_shards = range(train_lt)
    design_dir = config.TRAINING_DIR / variant

    print(f"=== export variant={variant}  design_dir={design_dir} ===")
    print(f"train shards: 0..{train_lt - 1}  "
          f"eval shards: 0..{eval_lt - 1}  p_keep={config.P_KEEP_CURRENT}")
    nat, state, mkt = _load_macro()
    con = connect()
    _write_pool(con, nat, state, mkt, "train", train_shards, design_dir / "train_pool")
    _write_pool(con, nat, state, mkt, "eval", range(eval_lt), design_dir / "eval_pool")

    manifest = _build_manifest(con, variant, train_lt, eval_lt, design_dir)
    (design_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    con.close()
    print(f"\nwrote {design_dir / 'manifest.json'}")
    print(f"  train_pool: {manifest['train_pool']['rows']:,} rows "
          f"(weighted {manifest['train_pool']['weighted_rows']:,.0f})")
    print(f"  eval_pool : {manifest['eval_pool']['rows']:,} rows")
    return design_dir


# ---------------------------------------------------------------------------
# Verify — every M4 Accept criterion, with evidence
# ---------------------------------------------------------------------------
def verify(variant: str) -> None:
    design_dir = config.TRAINING_DIR / variant
    manifest = json.loads((design_dir / "manifest.json").read_text())
    params = manifest["params"]
    # `train_shard_lt` is the current key; fall back to the legacy `dev_shard_lt`
    # (None ⇒ full population) so an older manifest still verifies.
    train_lt = params.get("train_shard_lt", params.get("dev_shard_lt"))
    eval_lt = params["eval_shard_lt"]
    train_glob = _pool_glob(design_dir / "train_pool")
    eval_glob = _pool_glob(design_dir / "eval_pool")
    con = connect()
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    # --- Accept 1: per-label-year counts reconcile with direct panel counts -----
    print("\n[1] manifest row counts reconcile with direct DuckDB panel counts")
    shard_train = "TRUE" if train_lt is None else f"shard < {train_lt}"
    keep = (f"NOT({_CUR2CUR}) OR "
            f"(hash(\"Loan Identifier\" || '|' || period_ym::VARCHAR) % 1000000) < {_THR}")
    panel_train = dict(con.execute(
        f"SELECT ({_LABEL_YM})//100 AS ly, count(*) FROM panel "
        f"WHERE {shard_train} AND {_ORIGIN_SQL} AND state_next IS NOT NULL AND ({keep}) "
        "GROUP BY 1").fetchall())
    panel_eval = dict(con.execute(
        f"SELECT ({_LABEL_YM})//100 AS ly, count(*) FROM panel "
        f"WHERE shard < {eval_lt} AND {_ORIGIN_SQL} AND state_next IS NOT NULL "
        f"AND ({_LABEL_YM}) >= {config.EVAL_LABEL_YM_MIN} GROUP BY 1").fetchall())
    for name, panel_counts, mkey in (("train", panel_train, "train_pool"),
                                     ("eval", panel_eval, "eval_pool")):
        man = {int(k): v for k, v in manifest[mkey]["rows_by_label_year"].items()}
        match = {int(k): v for k, v in panel_counts.items()} == man
        check(f"{name}_pool label-year counts == panel ({len(man)} years, "
              f"{sum(man.values()):,} rows)", match)
        if not match:
            print("    manifest:", man)
            print("    panel   :", {int(k): v for k, v in panel_counts.items()})

    # --- Accept 2: eval pool loan-disjoint-by-shard + zero thinned rows ----------
    print("\n[2] eval pool is loan-disjoint-by-shard and unthinned")
    shards = [r[0] for r in con.execute(
        f"SELECT DISTINCT shard FROM read_parquet('{eval_glob}') ORDER BY 1").fetchall()]
    check(f"eval shards ⊆ [0,{eval_lt}) (whole loans by shard)",
          max(shards) < eval_lt, f"shards present: {shards}")
    wmin, wmax = con.execute(
        f"SELECT min(weight), max(weight) FROM read_parquet('{eval_glob}')").fetchone()
    check("eval weights all = 1.0 (no importance-weighting ⇒ unthinned)",
          wmin == 1.0 and wmax == 1.0, f"weight∈[{wmin},{wmax}]")
    eval_c2c = con.execute(
        f"SELECT count(*) FROM read_parquet('{eval_glob}') "
        "WHERE state='current' AND state_next='current'").fetchone()[0]
    panel_c2c = con.execute(
        f"SELECT count(*) FROM panel WHERE shard < {eval_lt} AND {_CUR2CUR} "
        f"AND state_next IS NOT NULL AND ({_LABEL_YM}) >= {config.EVAL_LABEL_YM_MIN}").fetchone()[0]
    check("eval keeps every current→current row (zero dropped)",
          eval_c2c == panel_c2c, f"eval={eval_c2c:,} panel={panel_c2c:,}")

    # --- Accept 3: window masks from config.py slice both pools correctly --------
    print("\n[3] config.py window masks slice both pools (spot-check k=2015, k=2025)")
    for k in (2015, 2025):
        for pool, glob in (("train", train_glob), ("eval", eval_glob)):
            for split, bounds in (("train", config.train_bounds(k)),
                                  ("val", config.val_bounds(k)),
                                  ("test", config.test_bounds(k))):
                if pool == "eval" and split == "train":
                    continue   # eval pool starts at 2014; earlier train years not present
                n, lys = con.execute(
                    f"SELECT count(*), list_sort(array_agg(DISTINCT label_ym//100)) "
                    f"FROM read_parquet('{glob}') "
                    f"WHERE {config.sql_period_mask('period_ym', bounds)}").fetchone()
                exp = {"train": list(range(2000, k - 1)), "val": [k - 1], "test": [k]}[split]
                exp = [y for y in exp if y >= (2014 if pool == "eval" else 2000)]
                good = (n == 0 and not exp) or (n > 0 and list(lys) == exp)
                check(f"k={k} {pool}.{split}: {n:>10,} rows, label-years {list(lys)}",
                      good, "" if good else f"expected {exp}")

    # --- Accept 4: macro lags + fallback indicator + null check ------------------
    print("\n[4] macro join: per-variable lags, fallback indicator, null-free")
    nat, state, mkt = _load_macro()
    row = con.execute(
        f"SELECT \"Loan Identifier\", period_ym, orig_ym, \"Property State\" ps, "
        f"\"Current Interest Rate\" cr, \"Current Actual UPB\" cu, \"Original UPB\" ou, "
        f"\"Original Loan-to-Value (LTV)\" lv, pmms30, unrate_nat, unrate_state, "
        f"incentive, ltv_mtm, hpi_chg_12m, unrate_fallback, hpi_fallback "
        f"FROM read_parquet('{train_glob}') WHERE \"Property State\" NOT IN ('PR','VI','GU') "
        f"AND \"Current Interest Rate\" IS NOT NULL LIMIT 1").pl().row(0, named=True)
    p, o, st = row["period_ym"], row["orig_ym"], row["ps"]

    def nat_at(m, col):
        return nat.filter(pl.col("month_ym") == m)[col].item()

    def hpi_at(m):  # state HPI with US fallback (mirrors the join)
        r = state.filter((pl.col("state") == st) & (pl.col("month_ym") == m))
        if r.height == 0:
            r = state.filter((pl.col("state") == "US") & (pl.col("month_ym") == m))
        return r["hpi_state"].item()

    src_pmms = nat_at(mf.sub_months(p, mf.LAG["pmms30"]), "pmms30")
    src_unr_nat = nat_at(mf.sub_months(p, mf.LAG["unrate"]), "unrate_nat")
    src_unr_st = state.filter((pl.col("state") == st)
                              & (pl.col("month_ym") == mf.sub_months(p, mf.LAG["unrate"])))["unrate_state"].item()
    exp_ltv = (row["cu"] / row["ou"]) * row["lv"] * hpi_at(mf.sub_months(o, mf.LAG["hpi"])) \
        / hpi_at(mf.sub_months(p, mf.LAG["hpi"]))
    print(f"    sample loan={row['Loan Identifier']} period_ym={p} orig_ym={o} state={st}")
    check(f"pmms30 lag0: joined {row['pmms30']:.4f} == source(month {mf.sub_months(p,0)}) {src_pmms:.4f}",
          abs(row["pmms30"] - src_pmms) < 1e-4)
    check(f"unrate_nat lag1: joined {row['unrate_nat']:.4f} == source(month {mf.sub_months(p,1)}) {src_unr_nat:.4f}",
          abs(row["unrate_nat"] - src_unr_nat) < 1e-4)
    check(f"unrate_state lag1: joined {row['unrate_state']:.4f} == source(state {st}, month {mf.sub_months(p,1)}) {src_unr_st:.4f}",
          abs(row["unrate_state"] - src_unr_st) < 1e-4)
    check(f"incentive == current_rate − pmms30 ({row['cr']:.3f} − {row['pmms30']:.3f})",
          abs(row["incentive"] - (row["cr"] - row["pmms30"])) < 1e-4)
    check(f"ltv_mtm lag2 (HPI orig {mf.sub_months(o,2)} / period {mf.sub_months(p,2)}): "
          f"joined {row['ltv_mtm']:.4f} == recomputed {exp_ltv:.4f}",
          abs(row["ltv_mtm"] - exp_ltv) < 1e-3)

    fb = con.execute(
        f"SELECT sum(unrate_fallback), sum(hpi_fallback), count(*) "
        f"FROM read_parquet('{train_glob}')").fetchone()
    check(f"unrate_fallback / hpi_fallback present (territory rows: "
          f"{int(fb[0]):,} / {int(fb[1]):,} of {int(fb[2]):,})", fb[0] > 0 and fb[1] > 0)
    nullcond = " OR ".join(f"{c} IS NULL" for c in MACRO_NONNULL)
    for glob, name in ((train_glob, "train"), (eval_glob, "eval")):
        nnull = con.execute(
            f"SELECT count(*) FROM read_parquet('{glob}') WHERE {nullcond}").fetchone()[0]
        check(f"{name}_pool: zero null pure-macro cols {MACRO_NONNULL}", nnull == 0,
              f"nulls={nnull}")
    inc_null, ltv_null, mkt_null = con.execute(
        f"SELECT count(*) FILTER (WHERE incentive IS NULL), "
        f"count(*) FILTER (WHERE ltv_mtm IS NULL), count(*) FILTER (WHERE mkt_rate IS NULL) "
        f"FROM read_parquet('{train_glob}')").fetchone()
    print(f"    (derived-feature nulls, panel-inherited: incentive={inc_null:,} "
          f"ltv_mtm={ltv_null:,}; mkt_rate={mkt_null:,} — expect 0)")
    check("mkt_rate proxy null-free", mkt_null == 0)

    # --- Accept 5: idempotency (content hash recorded; re-run reproduces it) -----
    print("\n[5] idempotency: written content hash matches manifest (re-run reproduces)")
    for mkey, glob in (("train_pool", train_glob), ("eval_pool", eval_glob)):
        h = _content_hash(con, design_dir / ("train_pool" if mkey == "train_pool" else "eval_pool"))
        check(f"{mkey} content_hash stable", h == manifest[mkey]["content_hash"],
              f"hash={h}")
    con.close()
    print(f"\n{'ALL ACCEPT CHECKS PASS' if ok else 'SOME CHECKS FAILED'} (variant={variant})")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/verify the training export (M4).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--verify-only", action="store_true",
                    help="run the Accept checks against existing output (no rebuild)")
    args = ap.parse_args()
    if not args.verify_only:
        build(args.variant)
    verify(args.variant)


if __name__ == "__main__":
    main()
