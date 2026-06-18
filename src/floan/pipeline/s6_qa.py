"""Stage 6 — QA & reconciliation  (s6_qa.py).

Prove each stage preserved what it should and surface data-quality issues, over
the full lake. Reads metadata (instant) for row reconciliation and runs a handful
of bounded DuckDB aggregations over the panel for the rest. Emits:

  * ``outputs/qa_report.md``    — human-readable summary (the gate before deletion)
  * ``outputs/null_audit.csv``  — null rate per column per quarter
  * ``outputs/feature_stats.csv`` — scale stats for the continuous features

Checks (02_PIPELINE_STAGES.md §Stage 6): row reconciliation (manifest = perf =
clean = panel), null/missingness audit (+ schema-drift flags), target 7x7
distribution + marginals, ``(loan, period)`` uniqueness, shard balance, calendar
sanity, and continuous-feature scale stats. Read-only; never writes to ``raw/``.

Run:  ``python s6_qa.py``
"""

from __future__ import annotations

import csv as csvmod
from datetime import datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

import config
import schema

REPORT_PATH = config.OUTPUTS / "qa_report.md"
NULL_AUDIT_PATH = config.OUTPUTS / "null_audit.csv"
FEATURE_STATS_PATH = config.OUTPUTS / "feature_stats.csv"
MANIFEST_PATH = config.OUTPUTS / "inventory_manifest.csv"

STATES = ["current", "dpd_30", "dpd_60", "dpd_90plus", "foreclosure", "REO", "prepaid"]


def _quarter_key(q: str) -> tuple[int, int]:
    return (int(q[:4]), int(q[-1]))


def _qcol(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _rows(p: Path) -> int:
    return pq.ParquetFile(str(p)).metadata.num_rows if p.exists() else -1


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    tmp = config.DUCKDB_PATH.parent / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(tmp)}'")
    return con


def _panel_view(con: duckdb.DuckDBPyConnection) -> str:
    g = f"{config.PANEL_DIR}/acq_quarter=*/part.parquet"
    con.execute(f"CREATE OR REPLACE VIEW panel AS "
                f"SELECT * FROM read_parquet('{g}', hive_partitioning=true)")
    return "panel"


def reconcile() -> tuple[list[dict], bool]:
    """manifest = perf = clean = panel rows, per quarter (metadata only)."""
    man = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, newline="") as fh:
            man = {r["acq_quarter"]: r for r in csvmod.DictReader(fh)}
    quarters = sorted(
        (p.name.split("=", 1)[1] for p in config.PERF_DIR.glob("acq_quarter=*")),
        key=_quarter_key,
    )
    out, ok = [], True
    for q in quarters:
        rm = int(man[q]["rows"]) if q in man and man[q]["rows"] not in ("", "0") else -1
        rp = _rows(config.PERF_DIR / f"acq_quarter={q}" / "part.parquet")
        rc = _rows(config.CLEAN_DIR / f"acq_quarter={q}" / "part.parquet")
        rpn = _rows(config.PANEL_DIR / f"acq_quarter={q}" / "part.parquet")
        match = (rm == rp == rc == rpn)
        ok &= match
        out.append({"q": q, "manifest": rm, "perf": rp, "clean": rc, "panel": rpn, "ok": match})
    return out, ok


def null_audit(con: duckdb.DuckDBPyConnection, cols: list[str]) -> tuple[list[dict], list[str]]:
    """Per-quarter null rate for every column (one grouped full scan, no distinct)."""
    sel = ["acq_quarter", "count(*) AS n"]
    for c in cols:
        sel.append(f"count({_qcol(c)}) AS {_qcol('nn_' + c)}")
    rows = con.execute(
        f"SELECT {', '.join(sel)} FROM panel GROUP BY acq_quarter"
    ).fetchall()
    desc = [d[0] for d in con.description]
    recs = [dict(zip(desc, r)) for r in rows]
    recs.sort(key=lambda r: _quarter_key(r["acq_quarter"]))

    audit = []
    for r in recs:
        n = r["n"]
        row = {"acq_quarter": r["acq_quarter"], "rows": n}
        for c in cols:
            nn = r["nn_" + c]
            row[c] = round(1 - nn / n, 6) if n else None
        audit.append(row)

    # schema-drift flag: a column whose null rate crosses 0.05<->0.95 across quarters
    drift = []
    for c in cols:
        rates = [a[c] for a in audit if a[c] is not None]
        if rates and min(rates) < 0.05 and max(rates) > 0.95:
            drift.append(c)
    return audit, drift


def dup_check() -> dict:
    """(loan, period) duplicates per quarter — EXACT, cheap.

    Each panel file is sorted by (shard, loan, period) and shard is a function of
    the loan, so any duplicate (loan, period) rows are ADJACENT. A single
    ordered ``lag`` pass reading just two columns finds them — no global distinct,
    no re-sort. Single-threaded + preserve_insertion_order so the scan follows
    file order.
    """
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=true")
    dups = {}
    files = sorted(config.PANEL_DIR.glob("acq_quarter=*/part.parquet"),
                   key=lambda p: _quarter_key(p.parent.name.split("=", 1)[1]))
    for p in files:
        q = p.parent.name.split("=", 1)[1]
        pp = str(p).replace("'", "''")
        n = con.execute(f"""
            SELECT count(*) FROM (
              SELECT ("Loan Identifier" = lag("Loan Identifier") OVER ()
                      AND period = lag(period) OVER ()) AS dup
              FROM read_parquet('{pp}', hive_partitioning=false)
            ) WHERE dup
        """).fetchone()[0]
        dups[q] = n
    con.close()
    return dups


def target_dist(con: duckdb.DuckDBPyConnection) -> tuple[dict, dict, int]:
    marg = dict(con.execute(
        "SELECT coalesce(state,'<null>'), count(*) FROM panel GROUP BY 1").fetchall())
    mat = {(a, b): c for a, b, c in con.execute(
        "SELECT state, state_next, count(*) FROM panel WHERE state IS NOT NULL GROUP BY 1,2"
    ).fetchall()}
    nulls = marg.get("<null>", 0)
    return marg, mat, nulls


def shard_calendar(con: duckdb.DuckDBPyConnection) -> dict:
    nd, mn, mx, av = con.execute(
        "SELECT count(*), min(c), max(c), avg(c) FROM "
        "(SELECT shard, count(*) c FROM panel GROUP BY shard)").fetchone()
    smin, smax = con.execute("SELECT min(shard), max(shard) FROM panel").fetchone()
    pmin, pmax, omin, omax, bad = con.execute(
        "SELECT min(period_ym), max(period_ym), min(orig_ym), max(orig_ym), "
        "count(*) FILTER (WHERE orig_ym > period_ym) FROM panel").fetchone()
    return {"nshard": nd, "smin": smin, "smax": smax, "cmin": mn, "cmax": mx, "cavg": av,
            "spread": 100 * (mx - mn) / av if av else 0,
            "period_min": pmin, "period_max": pmax, "orig_min": omin, "orig_max": omax,
            "orig_after_period": bad}


def feature_stats(con: duckdb.DuckDBPyConnection) -> list[dict]:
    """Scale stats for every continuous feature.

    Exact mean/std/min/max in one pass (cheap). Quantiles are diagnostic only, so
    they're estimated on a 2% sample — 45 simultaneous ``approx_quantile``
    t-digests over the full 3.31 B rows thrash the CPU and stall; on a sample they
    are fast and plenty accurate for picking transforms.
    """
    cols = schema.FEATURE_SPEC["continuous_standardize"]

    # Pass 1 — exact basics (no t-digests).
    parts = ["count(*) AS n"]
    for i, c in enumerate(cols):
        q = _qcol(c)
        parts += [f"count({q}) AS nn_{i}", f"avg({q}::DOUBLE) AS mean_{i}",
                  f"stddev_samp({q}::DOUBLE) AS std_{i}",
                  f"min({q}) AS min_{i}", f"max({q}) AS max_{i}"]
    res = con.execute(f"SELECT {', '.join(parts)} FROM panel")
    d = dict(zip([x[0] for x in res.description], res.fetchone()))

    # Pass 2 — quantiles on a 2% sample.
    qparts = []
    for i, c in enumerate(cols):
        q = _qcol(c)
        for p in (1, 25, 50, 75, 99):
            qparts.append(f"approx_quantile({q}::DOUBLE,{p/100}) AS p{p:02d}_{i}")
    res2 = con.execute(f"SELECT {', '.join(qparts)} FROM panel USING SAMPLE 2% (system)")
    qd = dict(zip([x[0] for x in res2.description], res2.fetchone()))

    n = d["n"]
    stats = []
    for i, c in enumerate(cols):
        nn = d[f"nn_{i}"]
        stats.append({
            "column": c, "log_flagged": c in schema.LOG_TRANSFORM,
            "null_rate": round(1 - nn / n, 6) if n else None,
            "mean": d[f"mean_{i}"], "std": d[f"std_{i}"],
            "min": d[f"min_{i}"], "max": d[f"max_{i}"],
            "p01": qd[f"p01_{i}"], "p25": qd[f"p25_{i}"], "median": qd[f"p50_{i}"],
            "p75": qd[f"p75_{i}"], "p99": qd[f"p99_{i}"],
        })
    return stats


def _fmt(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:,.4g}"
    return f"{x:,}" if isinstance(x, int) else str(x)


def main() -> None:
    config.ensure_dirs()
    config.require_drive()
    con = _connect()
    _panel_view(con)
    cols = pq.ParquetFile(
        str(next(config.PANEL_DIR.glob("acq_quarter=*/part.parquet")))).schema_arrow.names

    print("QA: reconciliation …", flush=True)
    recon, recon_ok = reconcile()
    print("QA: null/missingness audit (full scan) …", flush=True)
    audit, drift = null_audit(con, cols)
    print("QA: target distribution …", flush=True)
    marg, mat, nulls = target_dist(con)
    print("QA: shard + calendar sanity …", flush=True)
    sc = shard_calendar(con)
    print("QA: feature scale stats …", flush=True)
    fstats = feature_stats(con)
    con.close()
    print("QA: (loan, period) duplicate check …", flush=True)
    dups = dup_check()

    total_dups = sum(dups.values())
    total_rows = sum(a["rows"] for a in audit)
    for a in audit:
        a["dup_loan_period"] = dups.get(a["acq_quarter"], 0)

    # ---- null_audit.csv ----
    with open(NULL_AUDIT_PATH, "w", newline="") as fh:
        w = csvmod.DictWriter(fh, fieldnames=["acq_quarter", "rows", "dup_loan_period"] + cols)
        w.writeheader()
        for a in audit:
            w.writerow(a)

    # ---- feature_stats.csv ----
    with open(FEATURE_STATS_PATH, "w", newline="") as fh:
        w = csvmod.DictWriter(fh, fieldnames=list(fstats[0].keys()))
        w.writeheader()
        w.writerows(fstats)

    # ---- qa_report.md ----
    L = ["# Stage 6 — QA & Reconciliation Report\n",
         f"_Generated {datetime.now().isoformat(timespec='seconds')}._\n",
         f"- Total loan-months: **{total_rows:,}** across **{len(recon)}** vintages",
         f"- Row reconciliation (manifest = perf = clean = panel): "
         f"**{'PASS ✓' if recon_ok else 'FAIL ✗'}**",
         f"- Duplicate (loan, period) rows: **{total_dups:,}** "
         f"({'PASS ✓' if total_dups == 0 else 'FAIL ✗'})",
         f"- Null `state` (unusable target) rows: **{nulls:,}** "
         f"({100*nulls/total_rows:.5f}%)\n"]

    L.append("## Row reconciliation\n")
    L.append("| Quarter | manifest | perf | clean | panel | OK |")
    L.append("|---|--:|--:|--:|--:|:--:|")
    for r in recon:
        L.append(f"| {r['q']} | {r['manifest']:,} | {r['perf']:,} | {r['clean']:,} | "
                 f"{r['panel']:,} | {'✓' if r['ok'] else '✗'} |")
    L.append("")

    L.append("## Target distribution\n")
    L.append("**State marginals:**\n")
    L.append("| state | count | share |")
    L.append("|---|--:|--:|")
    for s in STATES + ["<null>"]:
        c = marg.get(s, 0)
        L.append(f"| {s} | {c:,} | {100*c/total_rows:.4f}% |")
    L.append("\n**7×7 transition matrix** (row=state → col=state_next; (end)=terminal):\n")
    L.append("| from \\ to | " + " | ".join(s for s in STATES) + " | (end) |")
    L.append("|---" + "|--:" * (len(STATES) + 1) + "|")
    for a in STATES:
        cells = " | ".join(f"{mat.get((a, b), 0):,}" for b in STATES)
        L.append(f"| **{a}** | {cells} | {mat.get((a, None), 0):,} |")
    L.append("")

    L.append("## Shard & calendar sanity\n")
    L.append(f"- Shards: **{sc['nshard']}** (range [{sc['smin']}, {sc['smax']}], expect "
             f"[0, {config.N_SHARDS - 1}]); per-shard rows {sc['cmin']:,}–{sc['cmax']:,} "
             f"(avg {sc['cavg']:,.0f}, spread {sc['spread']:.1f}%)")
    L.append("- Loans spanning >1 shard: **0 by construction** "
             "(`shard = hash(loan_id) % N_SHARDS` is a pure function of the loan)")
    L.append(f"- `period_ym` range: {sc['period_min']}–{sc['period_max']}; "
             f"`orig_ym` range: {sc['orig_min']}–{sc['orig_max']}")
    L.append(f"- Rows with `orig_ym > period_ym` (loan reported before origination — "
             f"source dirt, see Data-quality findings): **{sc['orig_after_period']:,}**")
    L.append(f"- Derived right-censoring cut-off (max `period_ym`): **{sc['period_max']}**\n")

    L.append("## Feature scale stats (continuous)\n")
    L.append("Diagnostics only — scalers are fit train-only at model time (`models/`). "
             "`log` = right-skewed, apply `log1p` before standardizing.\n")
    L.append("| column | log | null% | mean | std | min | p01 | median | p99 | max |")
    L.append("|---|:--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for s in fstats:
        L.append(f"| {s['column']} | {'✓' if s['log_flagged'] else ''} | "
                 f"{100*(s['null_rate'] or 0):.2f}% | {_fmt(s['mean'])} | {_fmt(s['std'])} | "
                 f"{_fmt(s['min'])} | {_fmt(s['p01'])} | {_fmt(s['median'])} | "
                 f"{_fmt(s['p99'])} | {_fmt(s['max'])} |")
    L.append("")

    L.append("## Null / missingness audit\n")
    L.append(f"Full per-column-per-quarter null rates → `null_audit.csv`. "
             f"Schema-drift flags (null rate crosses 5%↔95% across quarters): "
             f"**{', '.join(drift) if drift else 'none'}**.\n")

    L.append("## Acceptance — pipeline integrity (blocking)\n")
    integrity = [
        ("Row reconciliation (manifest = perf = clean = panel)", recon_ok),
        ("(loan, period) uniqueness", total_dups == 0),
        ("shard ∈ [0, N_SHARDS) and count == N_SHARDS",
         sc["nshard"] == config.N_SHARDS and sc["smin"] == 0 and sc["smax"] == config.N_SHARDS - 1),
    ]
    for name, passed in integrity:
        L.append(f"- {name}: {'PASS ✓' if passed else 'FAIL ✗'}")
    integ_ok = all(p for _, p in integrity)
    L.append(f"\n**Pipeline integrity: "
             f"{'CERTIFIED ✓' if integ_ok else 'FAIL ✗ — investigate before proceeding.'}**\n")

    L.append("## Data-quality findings (non-blocking — handle at sampling/model time)\n")
    empty = [s["column"] for s in fstats if (s["null_rate"] or 0) >= 0.999]
    highnull = [(s["column"], s["null_rate"]) for s in fstats if 0.5 <= (s["null_rate"] or 0) < 0.999]
    L.append(f"- `orig_ym > period_ym` (loan reported before origination — source dirt): "
             f"**{sc['orig_after_period']} rows** (negligible); exclude via `orig_ym <= period_ym` at sampling.")
    L.append(f"- Null `state` (XX/blank — unusable label): **{nulls:,}** "
             f"({100*nulls/total_rows:.5f}%); excluded by `state IS NOT NULL`.")
    if empty:
        L.append(f"- Empty columns (≥99.9% null — drop from the feature set): **{', '.join(empty)}**.")
    if highnull:
        L.append("- High-null columns (informative absence, captured by a `*_missing` flag): "
                 + ", ".join(f"{c} ({100*(r or 0):.0f}%)" for c, r in highnull) + ".")
    L.append(f"- Schema-drift columns (null rate crosses 5%↔95% across vintages — feature "
             f"availability changed over time): **{', '.join(drift) if drift else 'none'}**.")
    L.append(f"\n**Overall: {'lake CERTIFIED ✓ for modelling (apply the sampling filters above).' if integ_ok else 'integrity FAILURE — do not proceed.'}**\n")

    REPORT_PATH.write_text("\n".join(L))
    print(f"\nWrote {REPORT_PATH}")
    print(f"      {NULL_AUDIT_PATH}")
    print(f"      {FEATURE_STATS_PATH}")
    print(f"Reconciliation: {'PASS' if recon_ok else 'FAIL'}; duplicates: {total_dups}; "
          f"null-state: {nulls}; overall: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
