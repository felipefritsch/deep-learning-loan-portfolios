"""M5 — Empirical transition-matrix benchmark over the 11 rolling windows (``02 §4``).

Model A, the no-covariate floor of the headline comparison: for each window k the
4×7 row-normalised one-month transition matrix estimated on that window's *training*
slice (labels ≤ Dec(k−2), the **full unthinned panel** — ``02 §4``), Laplace-smoothed
(α = 0.5) so no test transition has probability 0, scored by out-of-sample mean
negative log-likelihood on the window's **frozen test slice** (label year k, cut from
the never-thinned ``eval_pool`` — the identical rows every later model is scored on,
``02 §3.2``/``§6``). Plus the optional "bucketed matrix" (``02 §4``): the same idea
conditioned on state × FICO-tercile × incentive-tercile, showing how far cheap
nonparametric conditioning gets before any fitting.

How it stays cheap on a 3.3-billion-row panel
---------------------------------------------
A loan-month's role changes across windows (test in k → train in k+2), so we never
re-scan per window. The training counts come from **one** panel pass grouped by
``(period_ym, state, state_next, fico_t, inc_t)`` (≤ ~140 k rows out); the pooled
matrix, all 11 expanding-window plain matrices (sum periods < Dec(k−2), marginalise
the terciles) and the bucketed matrices are then derived in memory. The test counts
are **one** ``eval_pool`` pass grouped by ``(label_year, state, state_next, fico_t,
inc_t)``. Tercile **edges are frozen from the tuning-window (k=2015) train slice**
(``period_ym < 2013-12``, weighted by 1/p_keep, read from ``train_pool``): historical-
only (no look-ahead), and they are feature boundaries — not outcomes — so they leak
nothing about the target. Incentive = ``current_rate − pmms30(t)`` (lag 0, the macro
join's definition), identical on the train (panel⋈macro) and test (eval column) sides.

Reproducibility: every quantity is an integer ``GROUP BY`` count or a deterministic
edge, summed in a fixed cell order, so the NLLs are bit-for-bit stable across runs.

Run:
    .venv/bin/python -m floan.model.benchmarks                 # dev variant (default)
    .venv/bin/python -m floan.model.benchmarks --variant dev
    .venv/bin/python -m floan.model.benchmarks --verify-only   # re-run Accept checks on existing output
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from floan.model import config
from floan.pipeline import schema  # pipeline schema — canonical STATES order (single source of truth)

_REPO = Path(__file__).resolve().parents[2]

ALPHA = 0.5                                   # Laplace smoothing (02 §4)
EDGE_SLICE_HI = config._dec(config.TUNING_YEAR - 2)   # period_ym < Dec(2013): tuning train slice
ORIGIN = list(config.ORIGIN_STATES)           # current, dpd_30, dpd_60, dpd_90plus
DEST = list(schema.STATES)                     # 7 destinations, canonical order
_OI = {s: i for i, s in enumerate(ORIGIN)}
_DI = {s: i for i, s in enumerate(DEST)}
_ORIGIN_SQL = "state IN " + str(config.ORIGIN_STATES)


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO,
                           capture_output=True, text=True)
        return r.stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DuckDB — panel + macro_national + the export pools as views
# ---------------------------------------------------------------------------
def connect(variant: str) -> duckdb.DuckDBPyConnection:
    config.require_drive()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    tmp = _REPO / "reports" / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(tmp).replace(chr(39), chr(39) * 2)}'")
    design = config.TRAINING_DIR / variant
    con.execute("CREATE OR REPLACE VIEW panel AS "
                f"SELECT * FROM read_parquet('{config.PANEL_GLOB}', hive_partitioning=true)")
    con.execute("CREATE OR REPLACE VIEW macro_nat AS "
                f"SELECT * FROM read_parquet('{config.MACRO_DIR / 'macro_national.parquet'}')")
    con.execute("CREATE OR REPLACE VIEW train_pool AS "
                f"SELECT * FROM read_parquet('{design / 'train_pool' / 'part-*.parquet'}')")
    con.execute("CREATE OR REPLACE VIEW eval_pool AS "
                f"SELECT * FROM read_parquet('{design / 'eval_pool' / 'part-*.parquet'}')")
    return con


# ---------------------------------------------------------------------------
# Tercile edges — weighted, from the tuning-window training slice (frozen)
# ---------------------------------------------------------------------------
def _weighted_terciles(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Lower-weighted 1/3 and 2/3 quantiles (deterministic; ties broken by stable sort)."""
    order = np.argsort(values, kind="stable")
    v, cw = values[order], np.cumsum(weights[order])
    total = cw[-1]
    e1 = float(v[int(np.searchsorted(cw, total / 3.0, side="left"))])
    e2 = float(v[int(np.searchsorted(cw, 2.0 * total / 3.0, side="left"))])
    return e1, e2


def tercile_edges(con) -> dict[str, list[float]]:
    df = con.execute(
        "SELECT fico_orig, incentive, weight FROM train_pool "
        f"WHERE period_ym < {EDGE_SLICE_HI} AND {_ORIGIN_SQL} "
        "AND fico_orig IS NOT NULL AND incentive IS NOT NULL"
    ).pl()
    w = df["weight"].to_numpy()
    fe1, fe2 = _weighted_terciles(df["fico_orig"].to_numpy().astype(float), w)
    ie1, ie2 = _weighted_terciles(df["incentive"].to_numpy().astype(float), w)
    return {"fico": [fe1, fe2], "incentive": [ie1, ie2]}


def _bin_sql(col: str, e1: float, e2: float) -> str:
    """0 / 1 / 2 tercile bin for ``col`` (null ⇒ 3, the back-off bucket)."""
    return (f"CASE WHEN {col} IS NULL THEN 3 WHEN {col} < {e1!r} THEN 0 "
            f"WHEN {col} < {e2!r} THEN 1 ELSE 2 END")


# ---------------------------------------------------------------------------
# Counts — one panel pass (train) + one eval pass (test)
# ---------------------------------------------------------------------------
def train_counts(con, edges: dict) -> pl.DataFrame:
    """(period_ym, state, state_next, fico_t, inc_t, n) on the full unthinned panel.

    incentive = current_rate − pmms30(period_ym) (lag-0 macro join, the export's
    definition). The LEFT JOIN is on macro_nat's unique month key, so it neither
    drops nor duplicates panel rows — the tercile marginal recovers the plain counts.
    """
    fe1, fe2 = edges["fico"]
    ie1, ie2 = edges["incentive"]
    return con.execute(
        f"""
        SELECT period_ym, state, state_next,
               {_bin_sql('fico_orig', fe1, fe2)} AS fico_t,
               {_bin_sql('inc', ie1, ie2)} AS inc_t,
               count(*) AS n
        FROM (
          SELECT p.period_ym, p.state, p.state_next, p.fico_orig,
                 (p."Current Interest Rate" - mn.pmms30) AS inc
          FROM panel p LEFT JOIN macro_nat mn ON p.period_ym = mn.month_ym
          WHERE {_ORIGIN_SQL} AND p.state_next IS NOT NULL
        )
        GROUP BY 1, 2, 3, 4, 5
        """
    ).pl()


def test_counts(con, edges: dict) -> pl.DataFrame:
    """(label_year, state, state_next, fico_t, inc_t, n) on the frozen eval pool.

    eval_pool already carries fico_orig and the incentive column (same lag-0
    definition), so no macro join is needed; the same frozen edges bin both sides.
    """
    fe1, fe2 = edges["fico"]
    ie1, ie2 = edges["incentive"]
    return con.execute(
        f"""
        SELECT label_ym // 100 AS ly, state, state_next,
               {_bin_sql('fico_orig', fe1, fe2)} AS fico_t,
               {_bin_sql('incentive', ie1, ie2)} AS inc_t,
               count(*) AS n
        FROM eval_pool
        WHERE {_ORIGIN_SQL} AND state_next IS NOT NULL
        GROUP BY 1, 2, 3, 4, 5
        """
    ).pl()


# ---------------------------------------------------------------------------
# Matrices (Laplace) from the count frames
# ---------------------------------------------------------------------------
def _laplace(counts: np.ndarray) -> np.ndarray:
    """Row-normalise a 4×7 count matrix with Laplace α (rows then sum to 1)."""
    sm = counts + ALPHA
    return sm / sm.sum(axis=1, keepdims=True)


def _plain_counts(df: pl.DataFrame) -> np.ndarray:
    """4×7 origin→destination counts from a (state, state_next, n[, ...]) frame."""
    c = np.zeros((len(ORIGIN), len(DEST)))
    for s, sn, n in df.group_by("state", "state_next").agg(pl.col("n").sum()).iter_rows():
        c[_OI[s], _DI[sn]] += n
    return c


def _bucket_counts(df: pl.DataFrame) -> dict[tuple, np.ndarray]:
    """{(origin_i, fico_t, inc_t): 7-vector of dest counts} for terciles 0..2 only."""
    cells: dict[tuple, np.ndarray] = {}
    g = df.filter((pl.col("fico_t") < 3) & (pl.col("inc_t") < 3)) \
          .group_by("state", "fico_t", "inc_t", "state_next").agg(pl.col("n").sum())
    for s, ft, it, sn, n in g.iter_rows():
        cells.setdefault((_OI[s], ft, it), np.zeros(len(DEST)))[_DI[sn]] += n
    return cells


def build_matrices(train: pl.DataFrame) -> dict:
    """Pooled + per-window plain matrices and per-window bucketed cells (probabilities)."""
    pooled_c = _plain_counts(train)
    out: dict = {
        "pooled_counts": pooled_c,
        "pooled_plain": _laplace(pooled_c),
        "window_plain": {},      # k -> 4×7 probs
        "window_bucket": {},     # k -> {(oi,ft,it): 7 probs}
    }
    for k in config.TEST_YEARS:
        tr = train.filter(pl.col("period_ym") < config._dec(k - 2))   # labels ≤ Dec(k−2)
        out["window_plain"][k] = _laplace(_plain_counts(tr))
        out["window_bucket"][k] = {key: _laplace(v[None, :])[0]
                                   for key, v in _bucket_counts(tr).items()}
    return out


# ---------------------------------------------------------------------------
# NLL on the frozen test slices (cell-weighted: Σ n·(−log p̂) / N)
# ---------------------------------------------------------------------------
def window_nll(test_k: pl.DataFrame, plain: np.ndarray, bucket: dict) -> dict:
    """Empirical + bucketed test NLL for one window from its test count frame."""
    nll_e = nll_b = n_tot = backoff = 0.0
    for s, sn, ft, it, n in test_k.select(
            "state", "state_next", "fico_t", "inc_t", "n").iter_rows():
        oi, di = _OI[s], _DI[sn]
        n_tot += n
        nll_e += n * -np.log(plain[oi, di])
        cell = bucket.get((oi, ft, it)) if ft < 3 and it < 3 else None
        if cell is None:
            backoff += n
            nll_b += n * -np.log(plain[oi, di])
        else:
            nll_b += n * -np.log(cell[di])
    return {"n_test": int(n_tot), "nll_empirical": nll_e / n_tot,
            "nll_bucketed": nll_b / n_tot, "bucket_backoff_frac": backoff / n_tot}


def evaluate(test: pl.DataFrame, mats: dict) -> dict:
    """Per-window + pooled (all-years, each row scored once) empirical & bucketed NLL."""
    # DuckDB returns GROUP BY rows in a non-deterministic order; fix a canonical order
    # so the floating-point NLL summation is bit-for-bit reproducible across runs.
    test = test.sort("ly", "state", "state_next", "fico_t", "inc_t")
    per_window, pe = {}, {"se": 0.0, "sb": 0.0, "n": 0}
    for k in config.TEST_YEARS:
        tk = test.filter(pl.col("ly") == k)
        if tk.height == 0:
            continue
        r = window_nll(tk, mats["window_plain"][k], mats["window_bucket"][k])
        per_window[k] = r
        pe["se"] += r["nll_empirical"] * r["n_test"]
        pe["sb"] += r["nll_bucketed"] * r["n_test"]
        pe["n"] += r["n_test"]
    pooled = {"n_test": pe["n"], "nll_empirical": pe["se"] / pe["n"],
              "nll_bucketed": pe["sb"] / pe["n"]}
    return {"per_window": per_window, "pooled": pooled}


# ---------------------------------------------------------------------------
# Persist — run folder (metrics + matrices) and writeup tables
# ---------------------------------------------------------------------------
def _mat_to_rows(probs: np.ndarray) -> list[dict]:
    return [{"origin": ORIGIN[i], **{DEST[j]: float(probs[i, j]) for j in range(len(DEST))}}
            for i in range(len(ORIGIN))]


def _to_latex(df: pl.DataFrame, caption: str) -> str:
    align = "".join("r" if df[c].dtype.is_numeric() else "l" for c in df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    fmt = lambda v: ("" if v is None else f"{v:,}" if isinstance(v, int)
                     else f"{v:.6g}" if isinstance(v, float) else esc(v))
    body = [" & ".join(fmt(v) for v in row) + r" \\" for row in df.iter_rows()]
    return "\n".join([r"\begin{table}[htbp]", r"\centering", rf"\caption{{{esc(caption)}}}",
                      rf"\begin{{tabular}}{{{align}}}", r"\toprule",
                      " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule",
                      *body, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def _save_table(df: pl.DataFrame, name: str, caption: str) -> None:
    d = config.OUTPUTS / "tables" / "benchmarks"
    d.mkdir(parents=True, exist_ok=True)
    df.write_csv(d / f"{name}.csv")
    (d / f"{name}.tex").write_text(_to_latex(df, caption))


def write_outputs(variant: str, edges: dict, mats: dict, ev: dict) -> Path:
    design = config.TRAINING_DIR / variant
    manifest = json.loads((design / "manifest.json").read_text())
    run = config.MODELS / "benchmarks" / variant
    run.mkdir(parents=True, exist_ok=True)

    metrics = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # --- everything below is the reproducible block (created_utc excluded) ----
        "model": "empirical_matrix_benchmark",
        "variant": variant,
        "git_commit": _git_commit(),
        "alpha": ALPHA,
        "tercile_edges": {**edges,
                          "source": f"train_pool tuning slice period_ym<{EDGE_SLICE_HI}, "
                                    "weighted by 1/p_keep"},
        "source": {
            "panel_glob": config.PANEL_GLOB,
            "macro_national": str(config.MACRO_DIR / "macro_national.parquet"),
            "train_pool_content_hash": manifest["train_pool"]["content_hash"],
            "eval_pool_content_hash": manifest["eval_pool"]["content_hash"],
        },
        "test_years": config.TEST_YEARS,
        "per_window": {str(k): ev["per_window"][k] for k in sorted(ev["per_window"])},
        "pooled": ev["pooled"],
    }
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))

    matrices = {
        "states": {"origin": ORIGIN, "dest": DEST},
        "alpha": ALPHA,
        "tercile_edges": edges,
        "pooled_counts": {ORIGIN[i]: [int(x) for x in mats["pooled_counts"][i]]
                          for i in range(len(ORIGIN))},
        "pooled_plain": {ORIGIN[i]: [float(x) for x in mats["pooled_plain"][i]]
                         for i in range(len(ORIGIN))},
        "window_plain": {str(k): {ORIGIN[i]: [float(x) for x in mats["window_plain"][k][i]]
                                  for i in range(len(ORIGIN))} for k in config.TEST_YEARS},
        "window_bucket": {str(k): {f"{oi}|{ft}|{it}": [float(x) for x in v]
                                   for (oi, ft, it), v in mats["window_bucket"][k].items()}
                          for k in config.TEST_YEARS},
    }
    (run / "matrices.json").write_text(json.dumps(matrices, indent=2))

    # Writeup tables -------------------------------------------------------------
    rows = [{"window_k": k, "n_test": ev["per_window"][k]["n_test"],
             "nll_empirical": round(ev["per_window"][k]["nll_empirical"], 6),
             "nll_bucketed": round(ev["per_window"][k]["nll_bucketed"], 6),
             "bucket_backoff_pct": round(100 * ev["per_window"][k]["bucket_backoff_frac"], 3)}
            for k in sorted(ev["per_window"])]
    rows.append({"window_k": "pooled", "n_test": ev["pooled"]["n_test"],
                 "nll_empirical": round(ev["pooled"]["nll_empirical"], 6),
                 "nll_bucketed": round(ev["pooled"]["nll_bucketed"], 6),
                 "bucket_backoff_pct": None})
    _save_table(pl.DataFrame(rows), "M5_nll_by_window",
                "Empirical & bucketed transition-matrix out-of-sample NLL by test year "
                "(M5; pooled = all-years, each test row scored once).")
    _save_table(pl.DataFrame(_mat_to_rows(mats["pooled_plain"])), "M5_pooled_matrix",
                "Pooled empirical one-month transition matrix, Laplace-smoothed (M5); "
                "rows (transient origins) sum to 1.")
    return run


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def compute(con) -> tuple[dict, dict, dict]:
    edges = tercile_edges(con)
    mats = build_matrices(train_counts(con, edges))
    ev = evaluate(test_counts(con, edges), mats)
    return edges, mats, ev


def build(variant: str) -> Path:
    print(f"=== M5 empirical-matrix benchmark  variant={variant} ===")
    con = connect(variant)
    edges, mats, ev = compute(con)
    con.close()
    print(f"tercile edges (frozen, tuning slice): fico={edges['fico']}  "
          f"incentive={edges['incentive']}")
    run = write_outputs(variant, edges, mats, ev)
    print(f"\nper-window out-of-sample NLL (empirical | bucketed):")
    for k in sorted(ev["per_window"]):
        r = ev["per_window"][k]
        print(f"  k={k}  n_test={r['n_test']:>10,}  "
              f"emp={r['nll_empirical']:.5f}  buck={r['nll_bucketed']:.5f}  "
              f"(backoff {100 * r['bucket_backoff_frac']:.2f}%)")
    p = ev["pooled"]
    print(f"  pooled n_test={p['n_test']:,}  emp={p['nll_empirical']:.5f}  "
          f"buck={p['nll_bucketed']:.5f}")
    print(f"\nwrote {run / 'metrics.json'}\n      {run / 'matrices.json'}")
    return run


# ---------------------------------------------------------------------------
# Verify — every M5 Accept criterion, with evidence
# ---------------------------------------------------------------------------
def _t21_counts() -> np.ndarray | None:
    p = config.OUTPUTS / "tables" / "eda" / "T2.1_transition_counts.csv"
    if not p.exists():
        return None
    df = pl.read_csv(p)
    c = np.zeros((len(ORIGIN), len(DEST)))
    for row in df.iter_rows(named=True):
        for j, d in enumerate(DEST):
            c[_OI[row["origin"]], j] = row[d]
    return c


def verify(variant: str) -> None:
    run = config.MODELS / "benchmarks" / variant
    metrics = json.loads((run / "metrics.json").read_text())
    matrices = json.loads((run / "matrices.json").read_text())
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    # --- Accept 1: all 11 test NLLs finite (smoothing works) --------------------
    print("\n[1] all 11 window test NLLs finite (Laplace smoothing ⇒ no log 0)")
    pw = metrics["per_window"]
    check(f"all {len(config.TEST_YEARS)} windows present", len(pw) == len(config.TEST_YEARS),
          f"windows={sorted(int(k) for k in pw)}")
    for kind in ("nll_empirical", "nll_bucketed"):
        vals = [pw[k][kind] for k in pw]
        check(f"{kind}: all finite", all(np.isfinite(v) and v > 0 for v in vals),
              f"range [{min(vals):.4f}, {max(vals):.4f}]")
    check("pooled empirical & bucketed NLL finite",
          np.isfinite(metrics["pooled"]["nll_empirical"])
          and np.isfinite(metrics["pooled"]["nll_bucketed"]),
          f"pooled emp={metrics['pooled']['nll_empirical']:.5f} "
          f"buck={metrics['pooled']['nll_bucketed']:.5f}")

    # --- Accept 2: pooled matrix ≈ M2's (T2.1) in shape -------------------------
    print("\n[2] pooled matrix ≈ M2 (T2.1) in shape")
    pooled_counts = np.array([matrices["pooled_counts"][o] for o in ORIGIN])
    t21 = _t21_counts()
    if t21 is None:
        check("T2.1 counts table present for comparison", False, "missing T2.1 CSV")
    else:
        check("pooled raw counts == T2.1 exactly (same population, before smoothing)",
              np.array_equal(pooled_counts, t21),
              f"max|Δcount|={int(np.abs(pooled_counts - t21).max())}")
        # smoothed probs vs T2.1's unsmoothed probs: differ only at the α/row-total level
        pooled_probs = np.array([matrices["pooled_plain"][o] for o in ORIGIN])
        t21_probs = t21 / t21.sum(axis=1, keepdims=True)
        check("pooled smoothed probs ≈ T2.1 probs (max|Δ| < 1e-6)",
              np.abs(pooled_probs - t21_probs).max() < 1e-6,
              f"max|Δprob|={np.abs(pooled_probs - t21_probs).max():.2e}")
        rs = pooled_probs.sum(axis=1)
        check("4×7 shape, rows sum to 1, current-dominant diagonal",
              pooled_probs.shape == (4, 7) and np.abs(rs - 1).max() < 1e-12
              and pooled_probs[0, 0] > 0.97,
              f"current→current={pooled_probs[0, 0]:.4f}")

    # --- Accept 3: NLLs reproducible bit-for-bit across two runs -----------------
    print("\n[3] NLLs reproducible bit-for-bit (recompute == stored metrics.json)")
    con = connect(variant)
    _, _, ev2 = compute(con)
    con.close()
    same = True
    for k in config.TEST_YEARS:
        a, b = metrics["per_window"][str(k)], ev2["per_window"][k]
        same = same and a["nll_empirical"] == b["nll_empirical"] \
            and a["nll_bucketed"] == b["nll_bucketed"] and a["n_test"] == b["n_test"]
    same = same and metrics["pooled"]["nll_empirical"] == ev2["pooled"]["nll_empirical"] \
        and metrics["pooled"]["nll_bucketed"] == ev2["pooled"]["nll_bucketed"]
    check("recomputed per-window + pooled NLLs identical to stored (==, no tolerance)", same)

    print(f"\n{'ALL ACCEPT CHECKS PASS' if ok else 'SOME CHECKS FAILED'} (variant={variant})")
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build/verify the empirical-matrix benchmark (M5).")
    ap.add_argument("--variant", default="dev", choices=list(config.VARIANTS))
    ap.add_argument("--verify-only", action="store_true",
                    help="run the Accept checks against existing output (no rebuild)")
    args = ap.parse_args()
    if not args.verify_only:
        build(args.variant)
    verify(args.variant)


if __name__ == "__main__":
    main()
