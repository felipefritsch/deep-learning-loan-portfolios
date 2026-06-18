"""M6 — Feature pipeline (``02_LOAN_LEVEL §2``).

The model-time feature contract shared by every Phase-2 model (logit, NN, ensemble)
and consumed by the streaming loader in ``data.py``. Three concerns:

  * **Scaler** — standardize the continuous block on the *training slice only*
    (``01_SCHEMA §6.4``): log1p the heavily-skewed dollar columns, robust median/IQR
    for the right-skewed ratios, mean/std otherwise. Nulls impute to the column centre
    (→ 0 after standardizing). Fit per backtest window, persisted to the run folder.
  * **Vocab** — embedding vocabulary per categorical, built on the train slice. Index
    ``0`` is a reserved ``UNK``; known levels map to ``1..V``; **unseen test levels and
    nulls map to 0** (the leakage-safe out-of-vocab bucket).
  * **One-hot** — the logit benchmark's encoding (``§2``): drop-first dummies over the
    vocab, with the ``UNK`` column dropped as the reference (so unseen/missing = base).

Feature membership is *sourced* from the pipeline ``schema.FEATURE_SPEC`` (the binding
contract ``export.py`` already wrote the pools against) plus the macro additions in
``macro_features.MACRO_FEATURE_SPEC`` — never hand-typed here, so it cannot drift.
Origin ``state`` enters as a categorical feature (the paper's single conditioned model,
``§1``); ``period_ym``/``orig_ym`` enter only via derived ``month_of_year`` /
``vintage_year`` categoricals (``§2`` — raw calendar time is never a continuous feature).

Pure-Polars/NumPy and hermetic: ``Scaler.fit`` / ``Vocab.fit`` take an in-memory frame
(the masked train slice), so the unit tests need no SSD and no torch. Companion tests:
``test_features.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

import config  # noqa: F401 — side effect: puts dev/pipeline on sys.path for `schema`
import macro_features as mf
import schema  # pipeline feature spec + canonical STATES order (single source of truth)

# ---------------------------------------------------------------------------
# Feature membership — sourced from the pipeline + macro specs (no drift).
# ---------------------------------------------------------------------------
# Continuous (standardized): panel continuous + macro continuous. mkt_rate is left out
# of the default feature block — it is the PMMS proxy rate behind the robustness
# incentive (01_EDA §4), near-collinear with pmms30; it stays in the pool for that
# variant but is not a standalone model feature.
CONTINUOUS: list[str] = (
    schema.FEATURE_SPEC["continuous_standardize"]
    + mf.MACRO_FEATURE_SPEC["continuous_standardize"]
)
# Heavily right-skewed dollar columns → log1p before standardizing (schema.LOG_TRANSFORM).
LOG_COLS: frozenset[str] = frozenset(schema.LOG_TRANSFORM)
# Right-skewed bounded-below ratios → robust median/IQR centre/scale (§6.4 "robust where
# skewed"). Guarded: a (near-)zero IQR falls back to std, then to 1.0, so a mostly-constant
# column (e.g. MI%, ~0 for most loans) degrades gracefully instead of dividing by zero.
ROBUST_COLS: frozenset[str] = frozenset(
    {"Debt-to-Income (DTI)", "Mortgage Insurance Percentage"}
)

# Categorical (embed / one-hot): origin state (the conditioning feature, §1) + panel
# categoricals + derived time categoricals (§2). The two derived columns are added by
# ``prepare_raw`` from period_ym / orig_ym.
RAW_CATEGORICAL: list[str] = ["state"] + schema.FEATURE_SPEC["categorical"]
DERIVED_CATEGORICAL: list[str] = ["month_of_year", "vintage_year"]
CATEGORICAL: list[str] = RAW_CATEGORICAL + DERIVED_CATEGORICAL

# Binary (0/1 passthrough): panel binaries + macro fallback indicators.
BINARY: list[str] = schema.FEATURE_SPEC["binary"] + mf.MACRO_FEATURE_SPEC["binary"]

# Target: state_next → index in the canonical 7-state space. Origin-state filtering and
# state_next-non-null are already applied by the export, but we keep the map total.
TARGET_COL = "state_next"
WEIGHT_COL = "weight"
STATES: list[str] = list(schema.STATES)
STATE_INDEX: dict[str, int] = {s: i for i, s in enumerate(STATES)}
N_CLASSES = len(STATES)

# Key columns carried through the loader for eval-time row alignment (M11) and tests.
KEY_COLS: list[str] = ["Loan Identifier", "period_ym", "label_ym"]

UNK = 0  # reserved out-of-vocabulary / null index in every categorical vocab.


# ---------------------------------------------------------------------------
# Raw-frame preparation — derive the model-time columns the spec adds (§2).
# ---------------------------------------------------------------------------
def prepare_raw(df: pl.DataFrame) -> pl.DataFrame:
    """Add the derived categoricals (``month_of_year``, ``vintage_year``) to a raw pool
    frame. Used by BOTH the fit (vocab/scaler) and the loader, so the columns seen at
    fit time and transform time are identical. Calendar month-of-year captures prepay
    seasonality; vintage-year the origination cohort (§2). Stored as strings so the
    vocab treats them like every other categorical level."""
    return df.with_columns(
        (pl.col("period_ym") % 100).cast(pl.Utf8).alias("month_of_year"),
        (pl.col("orig_ym") // 100).cast(pl.Utf8).alias("vintage_year"),
    )


def derive_macro(loans: pl.DataFrame, nat: pl.DataFrame, state: pl.DataFrame) -> pl.DataFrame:
    """Model-side entry for the macro-derived features (incentive, ``ltv_mtm``,
    ``hpi_chg_12m``) — wraps ``macro_features.attach_macro`` so the §4 formula lives in
    one place. The pools already carry these columns (``export.py``), and ``ltv_mtm``
    cannot be rebuilt from pool columns alone (the intermediate HPI levels are dropped),
    so this is the canonical recomputation path used for validation and for any
    window-local re-derivation (M10)."""
    return mf.attach_macro(loans, nat, state)


# ---------------------------------------------------------------------------
# Scaler — train-slice standardization (§6.4)
# ---------------------------------------------------------------------------
@dataclass
class Scaler:
    """Per-column centre/scale fitted on a train slice. ``transform`` log1p's the
    flagged columns, applies ``(x − centre) / scale``, and imputes nulls to 0."""

    cols: list[str]
    center: dict[str, float]
    scale: dict[str, float]
    log_cols: list[str] = field(default_factory=list)
    robust_cols: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, df: pl.DataFrame, cols: list[str] = None,
            log_cols: frozenset[str] = LOG_COLS,
            robust_cols: frozenset[str] = ROBUST_COLS) -> "Scaler":
        """Fit centre/scale from the (already train-masked) frame ``df``.

        For ``robust_cols``: centre = median, scale = IQR/1.349 (≈ σ for a normal),
        falling back to std then 1.0 when the IQR (or std) collapses. Otherwise:
        centre = mean, scale = std (→ 1.0 when std collapses). log1p is applied *before*
        computing the statistics for ``log_cols`` (so the stats live in log space)."""
        cols = cols if cols is not None else CONTINUOUS
        center, scale = {}, {}
        for c in cols:
            x = df.get_column(c).cast(pl.Float64)
            if c in log_cols:
                x = x.clip(lower_bound=-1.0 + 1e-9).log1p()
            mean = x.mean()
            std = x.std(ddof=0)
            if c in robust_cols:
                med = x.median()
                q25, q75 = x.quantile(0.25), x.quantile(0.75)
                iqr = None if (q25 is None or q75 is None) else (q75 - q25)
                ctr = med if med is not None else (mean or 0.0)
                if iqr is not None and iqr > 1e-12:
                    scl = iqr / 1.349
                elif std is not None and std > 1e-12:
                    scl = std
                else:
                    scl = 1.0
            else:
                ctr = mean if mean is not None else 0.0
                scl = std if (std is not None and std > 1e-12) else 1.0
            center[c], scale[c] = float(ctr), float(scl)
        return cls(cols=list(cols), center=center, scale=scale,
                   log_cols=[c for c in cols if c in log_cols],
                   robust_cols=[c for c in cols if c in robust_cols])

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """``[N, len(cols)]`` float32 standardized matrix; nulls/NaN → 0 (the centre)."""
        exprs = []
        for c in self.cols:
            x = pl.col(c).cast(pl.Float64)
            if c in self.log_cols:
                x = x.clip(lower_bound=-1.0 + 1e-9).log1p()
            z = (x - self.center[c]) / self.scale[c]
            exprs.append(z.fill_null(0.0).fill_nan(0.0).cast(pl.Float32).alias(c))
        out = df.select(exprs)
        return out.to_numpy() if out.width else np.empty((df.height, 0), np.float32)

    def to_dict(self) -> dict:
        return {"cols": self.cols, "center": self.center, "scale": self.scale,
                "log_cols": self.log_cols, "robust_cols": self.robust_cols}

    @classmethod
    def from_dict(cls, d: dict) -> "Scaler":
        return cls(cols=d["cols"], center=d["center"], scale=d["scale"],
                   log_cols=d.get("log_cols", []), robust_cols=d.get("robust_cols", []))


# ---------------------------------------------------------------------------
# Vocab — embedding vocabularies with a reserved UNK (§2)
# ---------------------------------------------------------------------------
@dataclass
class Vocab:
    """Per-categorical ``{level: index}`` map. Index 0 is ``UNK`` (out-of-vocab/null);
    known train levels occupy ``1..V``. ``transform`` codes a frame to integer indices;
    unseen levels and nulls fall to ``UNK``."""

    cols: list[str]
    maps: dict[str, dict[str, int]]   # col -> {level(str): idx in 1..V}

    @classmethod
    def fit(cls, df: pl.DataFrame, cols: list[str] = CATEGORICAL,
            min_count: int = 1) -> "Vocab":
        """Build vocabularies from the (train-masked) frame. Levels are sorted for a
        deterministic index assignment. ``min_count`` prunes rare levels into ``UNK``
        (default 1 = keep all observed levels)."""
        maps: dict[str, dict[str, int]] = {}
        for c in cols:
            vc = (df.select(pl.col(c).cast(pl.Utf8))
                    .drop_nulls()
                    .group_by(c).len()
                    .filter(pl.col("len") >= min_count)
                    .sort(c))
            levels = vc.get_column(c).to_list()
            maps[c] = {lvl: i + 1 for i, lvl in enumerate(levels)}   # 0 reserved for UNK
        return cls(cols=list(cols), maps=maps)

    def vocab_size(self, col: str) -> int:
        """Embedding cardinality including the UNK slot (= |levels| + 1)."""
        return len(self.maps[col]) + 1

    @property
    def vocab_sizes(self) -> list[int]:
        return [self.vocab_size(c) for c in self.cols]

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """``[N, len(cols)]`` int64 index matrix; unseen levels and nulls → ``UNK`` (0)."""
        exprs = []
        for c in self.cols:
            m = self.maps[c]
            col = pl.col(c).cast(pl.Utf8).fill_null("\x00__null__")
            if m:
                expr = col.replace_strict(list(m.keys()), list(m.values()),
                                          default=UNK, return_dtype=pl.Int64)
            else:
                expr = pl.lit(UNK, dtype=pl.Int64)
            exprs.append(expr.alias(c))
        out = df.select(exprs)
        return out.to_numpy() if out.width else np.empty((df.height, 0), np.int64)

    def to_dict(self) -> dict:
        return {"cols": self.cols, "maps": self.maps}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        return cls(cols=d["cols"], maps={k: dict(v) for k, v in d["maps"].items()})


# ---------------------------------------------------------------------------
# One-hot path for the logit benchmark (§2 — drop-first dummies)
# ---------------------------------------------------------------------------
def one_hot(codes: np.ndarray, vocab_size: int, drop_first: bool = True) -> np.ndarray:
    """One-hot a single integer-coded column. With ``drop_first`` the ``UNK`` column
    (index 0) is dropped as the reference, so an unseen/missing level encodes as the
    all-zero baseline and known levels ``1..V`` each get a dummy (``V`` columns)."""
    codes = np.asarray(codes).reshape(-1)
    full = np.zeros((codes.shape[0], vocab_size), dtype=np.float32)
    full[np.arange(codes.shape[0]), codes] = 1.0
    return full[:, 1:] if drop_first else full


def one_hot_block(code_matrix: np.ndarray, vocab: Vocab,
                  drop_first: bool = True) -> np.ndarray:
    """Horizontally concatenate ``one_hot`` over every categorical column (the logit's
    design block). ``code_matrix`` is the ``Vocab.transform`` output, columns in
    ``vocab.cols`` order."""
    if code_matrix.shape[1] == 0:
        return np.empty((code_matrix.shape[0], 0), np.float32)
    blocks = [one_hot(code_matrix[:, j], vocab.vocab_size(c), drop_first)
              for j, c in enumerate(vocab.cols)]
    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# Binary block + target + frame encoding
# ---------------------------------------------------------------------------
def binary_matrix(df: pl.DataFrame, cols: list[str] = BINARY) -> np.ndarray:
    """``[N, len(cols)]`` float32 0/1 block (Bool/Int → float; nulls → 0)."""
    out = df.select([pl.col(c).cast(pl.Float32).fill_null(0.0).alias(c) for c in cols])
    return out.to_numpy() if out.width else np.empty((df.height, 0), np.float32)


def target_indices(df: pl.DataFrame, col: str = TARGET_COL) -> np.ndarray:
    """``state_next`` → ``[N]`` int64 class index in the canonical STATES order."""
    return (df.select(pl.col(col).replace_strict(
        list(STATE_INDEX.keys()), list(STATE_INDEX.values()),
        default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1))


def encode_frame(df: pl.DataFrame, scaler: Scaler, vocab: Vocab,
                 encode: str = "index") -> dict:
    """Turn a prepared (``prepare_raw``) frame into model-ready NumPy arrays.

    Returns a batch dict: ``cont`` (standardized continuous), ``cat`` (int vocab indices
    for the embedding path, or the one-hot design block when ``encode='onehot'``),
    ``bin`` (0/1), ``y`` (target index), ``w`` (importance weight), plus the ``loan`` /
    ``period_ym`` keys for eval-time row alignment."""
    codes = vocab.transform(df)
    cat = one_hot_block(codes, vocab) if encode == "onehot" else codes
    return {
        "cont": scaler.transform(df),
        "cat": cat,
        "bin": binary_matrix(df),
        "y": target_indices(df),
        "w": df.get_column(WEIGHT_COL).cast(pl.Float32).to_numpy(),
        "loan": df.get_column("Loan Identifier").to_numpy(),
        "period_ym": df.get_column("period_ym").to_numpy(),
    }


# ---------------------------------------------------------------------------
# Persistence — scalers/vocab live in the run folder (§2, §6.4)
# ---------------------------------------------------------------------------
def save_pipeline(out_dir: Path, scaler: Scaler, vocab: Vocab) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scaler.json").write_text(json.dumps(scaler.to_dict(), indent=2))
    (out_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), indent=2))


def load_pipeline(out_dir: Path) -> tuple[Scaler, Vocab]:
    out_dir = Path(out_dir)
    scaler = Scaler.from_dict(json.loads((out_dir / "scaler.json").read_text()))
    vocab = Vocab.from_dict(json.loads((out_dir / "vocab.json").read_text()))
    return scaler, vocab
