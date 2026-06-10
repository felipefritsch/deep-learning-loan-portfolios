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

import sys
from pathlib import Path

import duckdb
import polars as pl

# The pipeline package is the single source of truth for paths (ROOT, OUTPUTS,
# PANEL_DIR, require_drive) and the schema / feature spec — import it directly.
_PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

import config  # noqa: E402
import schema  # noqa: E402

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
