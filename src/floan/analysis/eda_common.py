"""Shared helpers for Phase-1 EDA scripts (``dev/model_plan/01_EDA.md``).

One read-only DuckDB connection to the Stage-4 transition panel plus the table /
figure save conventions, so every EDA script regenerates its artifact with a
single command, in bounded memory.

Conventions
-----------
* ``connect()``           — in-memory DuckDB over the panel lake, ``memory_limit``
  capped and sorts/aggregations spilled to the fast internal disk (never the USB
  lake). Exposes the panel as the view ``panel`` (``acq_quarter`` injected by Hive
  partitioning). Read-only; never writes to ``raw/``.
* ``save_table(df, name)``  — write a RAM-sized result to ``outputs/tables/eda/``
  as ``<name>.csv`` **and** ``<name>.tex`` (a small booktabs table; no pandas dep).
* ``save_figure(fig, name)`` — write a matplotlib figure to ``outputs/figures/eda/``
  as ``<name>.png`` (300 dpi) and ``<name>.pdf`` (used from M2 onwards).

Paths and the schema/feature spec come from the pipeline's single source of truth
(``dev/pipeline/{config,schema}.py``), re-exported here so EDA scripts import once.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl

# The pipeline package is the single source of truth for paths (ROOT, OUTPUTS,
# PANEL_DIR, require_drive) and the schema / feature spec — import it directly.
from floan.pipeline import config, schema

TABLES_DIR = config.OUTPUTS / "tables" / "eda"
FIGURES_DIR = config.OUTPUTS / "figures" / "eda"
PANEL_GLOB = f"{config.PANEL_DIR}/acq_quarter=*/part.parquet"


def connect() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB exposing the panel lake as the view ``panel``.

    Bounded memory: ``memory_limit`` from ``config`` and sorts/aggregations spill
    to ``reports/duckdb_tmp`` on the internal disk. Fails fast if the SSD is not
    mounted (``require_drive``).
    """
    config.require_drive()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    # The progress bar writes \r-updates that overwrite captured stdout in batch /
    # background runs, corrupting the printed evidence; off here (results unaffected).
    con.execute("SET enable_progress_bar=false")
    tmp = config.DUCKDB_PATH.parent / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(tmp).replace(chr(39), chr(39) * 2)}'")
    con.execute(
        "CREATE OR REPLACE VIEW panel AS "
        f"SELECT * FROM read_parquet('{PANEL_GLOB}', hive_partitioning=true)"
    )
    return con


def _latex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.6g}"
    return _latex_escape(str(v))


def _to_latex(df: pl.DataFrame, caption: str | None) -> str:
    align = "".join("r" if df[c].dtype.is_numeric() else "l" for c in df.columns)
    out = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        out.append(rf"\caption{{{_latex_escape(caption)}}}")
    out += [rf"\begin{{tabular}}{{{align}}}", r"\toprule",
            " & ".join(_latex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    out += [" & ".join(_fmt(v) for v in row) + r" \\" for row in df.iter_rows()]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def save_table(df: pl.DataFrame, name: str, caption: str | None = None) -> tuple[Path, Path]:
    """Write ``df`` to ``outputs/tables/eda/<name>.{csv,tex}``; return both paths."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / f"{name}.csv"
    tex_path = TABLES_DIR / f"{name}.tex"
    df.write_csv(csv_path)
    tex_path.write_text(_to_latex(df, caption))
    return csv_path, tex_path


def save_figure(fig, name: str) -> tuple[Path, Path]:
    """Write a matplotlib ``fig`` to ``outputs/figures/eda/<name>.{png,pdf}``."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path, pdf_path


# ---------------------------------------------------------------------------
# Empirical-hazard helpers (M3 nonlinearity figures, ``01_EDA.md`` §3)
# ---------------------------------------------------------------------------
# Each hazard figure is one DuckDB pass that emits numerator / denominator counts
# per *fine* covariate bin (an integer value or a rounded float). These helpers
# then collapse the fine bins into equal-population buckets (the spec's default,
# §3: "equal-population buckets ... not equal-width") and attach Wilson 95% CIs.
# Exact (no sampling), single-scan, and the bucketing lives in one place.


def wilson_ci(k, n, z: float = 1.96):
    """Wilson score interval for a binomial proportion ``k/n`` (vectorized).

    Returns ``(lo, hi)`` numpy arrays; degrades gracefully when ``n == 0``.
    Preferred over the normal approximation for the small per-bucket rates and
    the rare-event tails (foreclosure, deep delinquency) in these figures.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        denom = 1.0 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    return lo, hi


def equal_pop_edges(values, weights, n: int) -> list[float]:
    """Interior cut points splitting a weighted distribution into ``n`` equal-mass
    buckets. ``values`` are the fine-bin covariate values, ``weights`` their
    populations (denominators). Returns up to ``n-1`` strictly increasing edges
    (ties collapse, so fewer buckets where the covariate is degenerate)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v, kind="stable")
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    total = cum[-1]
    edges: list[float] = []
    for i in range(1, n):
        idx = int(np.searchsorted(cum, total * i / n, side="left"))
        e = float(v[min(idx, len(v) - 1)])
        if not edges or e > edges[-1]:
            edges.append(e)
    return edges


def hazard_curve(fine: pl.DataFrame, value_col: str, num_col: str, den_col: str,
                 n: int) -> pl.DataFrame:
    """Collapse a fine histogram into ``n`` equal-population buckets and rate them.

    ``fine`` has one row per fine covariate bin with a numerator (``num_col``,
    target-transition count) and denominator (``den_col``, at-risk count). Buckets
    carry ~equal denominator mass (by cumulative weight). Returns one row per
    bucket: denominator-weighted mean covariate ``value``, ``rate = num/den``,
    Wilson ``lo``/``hi``, and the raw ``num``/``den``.
    """
    f = (fine.filter(pl.col(value_col).is_not_null() & (pl.col(den_col) > 0))
         .sort(value_col))
    total = float(f[den_col].sum())
    f = f.with_columns(
        (((pl.col(den_col).cum_sum() - pl.col(den_col)) / total * n)
         .floor().cast(pl.Int64).clip(0, n - 1)).alias("_b")
    )
    g = (f.group_by("_b").agg(
            (pl.col(value_col).cast(pl.Float64) * pl.col(den_col)).sum().alias("_vw"),
            pl.col(num_col).sum().alias("num"),
            pl.col(den_col).sum().alias("den"),
         ).sort("_b")
         .with_columns((pl.col("_vw") / pl.col("den")).alias("value"))
         .drop("_vw", "_b"))
    lo, hi = wilson_ci(g["num"].to_numpy(), g["den"].to_numpy())
    return g.with_columns(
        (pl.col("num") / pl.col("den")).alias("rate"),
        pl.Series("lo", lo), pl.Series("hi", hi),
    ).select("value", "rate", "lo", "hi", "num", "den")
