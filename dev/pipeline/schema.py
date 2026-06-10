"""Schema, dtypes, code maps and the Sirignano seven-state target.

Single source of truth in code for the positional 113-column Fannie Mae
loan-performance layout (see ``dev/pipeline_plan/01_SCHEMA.md``). The CSVs have
**no header** — column identity is positional, so the order of ``COLUMNS`` *is*
the contract. Positions 0-109 are stable in the modern layout; 110-112 vary by
release and are carried as opaque strings (``extra_11x``) until reconciled
against the official Fannie Mae File Layout & Glossary.

Everything downstream imports names/types/maps from here so there is exactly
one definition of the schema.
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# 1. Positional column layout (01_SCHEMA.md §1)
#    The tuple order IS the contract — do not reorder. dtype codes:
#      str  = free string/id            cat  = dictionary-encoded categorical
#      date = parsed from MMYYYY        f32/f64 = float
#      i32/i16/i8 = nullable integer (smallest that fits)
# ---------------------------------------------------------------------------
# (position, name, dtype-code)
_LAYOUT: list[tuple[int, str, str]] = [
    (0,   "Reference Pool ID", "str"),
    (1,   "Loan Identifier", "str"),
    (2,   "Monthly Reporting Period", "date"),
    (3,   "Channel", "cat"),
    (4,   "Seller Name", "cat"),
    (5,   "Servicer Name", "cat"),
    (6,   "Master Servicer", "cat"),
    (7,   "Original Interest Rate", "f32"),
    (8,   "Current Interest Rate", "f32"),
    (9,   "Original UPB", "f32"),
    (10,  "UPB at Issuance", "f32"),
    (11,  "Current Actual UPB", "f32"),
    (12,  "Original Loan Term", "i16"),
    (13,  "Origination Date", "date"),
    (14,  "First Payment Date", "date"),
    (15,  "Loan Age", "i16"),
    (16,  "Remaining Months to Legal Maturity", "i16"),
    (17,  "Remaining Months to Maturity", "i16"),
    (18,  "Maturity Date", "date"),
    (19,  "Original Loan-to-Value (LTV)", "f32"),
    (20,  "Original Combined LTV (CLTV)", "f32"),
    (21,  "Number of Borrowers", "i8"),
    (22,  "Debt-to-Income (DTI)", "f32"),
    (23,  "Borrower Credit Score at Origination", "i16"),
    (24,  "Co-Borrower Credit Score at Origination", "i16"),
    (25,  "First-Time Home Buyer Indicator", "cat"),
    (26,  "Loan Purpose", "cat"),
    (27,  "Property Type", "cat"),
    (28,  "Number of Units", "i8"),
    (29,  "Occupancy Status", "cat"),
    (30,  "Property State", "cat"),
    (31,  "Metropolitan Statistical Area (MSA)", "cat"),
    (32,  "Zip Code Short", "cat"),
    (33,  "Mortgage Insurance Percentage", "f32"),
    (34,  "Amortization Type", "cat"),
    (35,  "Prepayment Penalty Indicator", "cat"),
    (36,  "Interest-Only Loan Indicator", "cat"),
    (37,  "Interest-Only First P&I Payment Date", "date"),
    (38,  "Months to Amortization", "i16"),
    (39,  "Current Loan Delinquency Status", "str"),
    (40,  "Loan Payment History", "str"),
    (41,  "Modification Flag", "cat"),
    (42,  "Mortgage Insurance Cancellation Indicator", "cat"),
    (43,  "Zero Balance Code", "cat"),
    (44,  "Zero Balance Effective Date", "date"),
    (45,  "UPB at the Time of Removal", "f32"),
    (46,  "Repurchase Date", "date"),
    (47,  "Scheduled Principal Current", "f32"),
    (48,  "Total Principal Current", "f32"),
    (49,  "Unscheduled Principal Current", "f32"),
    (50,  "Last Paid Installment Date", "date"),
    (51,  "Foreclosure Date", "date"),
    (52,  "Disposition Date", "date"),
    (53,  "Foreclosure Costs", "f32"),
    (54,  "Property Preservation & Repair Costs", "f32"),
    (55,  "Asset Recovery Costs", "f32"),
    (56,  "Misc Holding Expenses & Credits", "f32"),
    (57,  "Associated Taxes for Holding Property", "f32"),
    (58,  "Net Sales Proceeds", "f32"),
    (59,  "Credit Enhancement Proceeds", "f32"),
    (60,  "Repurchase Make-Whole Proceeds", "f32"),
    (61,  "Other Foreclosure Proceeds", "f32"),
    (62,  "Modification Non-Interest-Bearing UPB", "f32"),
    (63,  "Principal Forgiveness Amount", "f32"),
    (64,  "Original List Start Date", "date"),
    (65,  "Original List Price", "f32"),
    (66,  "Current List Start Date", "date"),
    (67,  "Current List Price", "f32"),
    (68,  "Borrower Credit Score at Issuance", "i16"),
    (69,  "Co-Borrower Credit Score at Issuance", "i16"),
    (70,  "Borrower Credit Score Current", "i16"),
    (71,  "Co-Borrower Credit Score Current", "i16"),
    (72,  "Mortgage Insurance Type", "cat"),
    (73,  "Servicing Activity Indicator", "cat"),
    (74,  "Current Period Modification Loss Amount", "f32"),
    (75,  "Cumulative Modification Loss Amount", "f32"),
    (76,  "Current Period Credit Event Net Gain/Loss", "f32"),
    (77,  "Cumulative Credit Event Net Gain/Loss", "f32"),
    (78,  "Special Eligibility Program", "cat"),   # glossary fld 79; replaced the old HomeReady Indicator (Jan 2023 release)
    (79,  "Foreclosure Principal Write-off Amount", "f32"),
    (80,  "Relocation Mortgage Indicator", "cat"),
    (81,  "Zero Balance Code Change Date", "date"),
    (82,  "Loan Holdback Indicator", "cat"),
    (83,  "Loan Holdback Effective Date", "date"),
    (84,  "Delinquent Accrued Interest", "f32"),
    (85,  "Property Valuation Method", "cat"),
    (86,  "High Balance Loan Indicator", "cat"),
    (87,  "ARM Initial Fixed-Rate Period <=5yr Indicator", "cat"),
    (88,  "ARM Product Type", "cat"),
    (89,  "Initial Fixed-Rate Period", "i16"),
    (90,  "Interest Rate Adjustment Frequency", "i16"),
    (91,  "Next Interest Rate Adjustment Date", "date"),
    (92,  "Next Payment Change Date", "date"),
    (93,  "Index", "cat"),
    (94,  "ARM Cap Structure", "cat"),
    (95,  "Initial Interest Rate Cap Up Percent", "f32"),
    (96,  "Periodic Interest Rate Cap Up Percent", "f32"),
    (97,  "Lifetime Interest Rate Cap Up Percent", "f32"),
    (98,  "Mortgage Margin", "f32"),
    (99,  "ARM Balloon Indicator", "cat"),
    (100, "ARM Plan Number", "cat"),
    (101, "Borrower Assistance Plan", "cat"),
    (102, "High LTV Refinance Option Indicator", "cat"),
    (103, "Deal Name", "cat"),
    (104, "Repurchase Make-Whole Proceeds Flag", "cat"),
    (105, "Alternative Delinquency Resolution", "cat"),
    (106, "Alternative Delinquency Resolution Count", "i16"),
    (107, "Total Deferral Amount", "f32"),
    (108, "Payment Deferral Modification Event Indicator", "cat"),
    (109, "Interest-Bearing UPB", "f32"),
    # 110-112: CONFIRMED against the official Fannie Mae File Layout & Glossary
    # (glossary fields 111-113). These Classic FICO scores are populated from the
    # December 2025 activity period and REPLACE the deprecated Borrower/Co-Borrower
    # Credit Score at Origination (cols 23/24), which stop populating March 2026.
    (110, "Origination Classic FICO", "i16"),   # glossary fld 111
    (111, "Issuance Classic FICO", "i16"),       # glossary fld 112
    (112, "Current Classic FICO", "i16"),        # glossary fld 113
]

COLUMNS: list[str] = [name for _, name, _ in _LAYOUT]
DTYPES: dict[str, str] = {name: dt for _, name, dt in _LAYOUT}

assert len(COLUMNS) == 113, f"COLUMNS must be 113, got {len(COLUMNS)}"
assert len(set(COLUMNS)) == 113, "COLUMNS names must be unique"

# Mapping from the dtype-code vocabulary to concrete Polars dtypes. ``date`` is
# intentionally absent: dates are produced by parse_mmyyyy(), not a plain cast.
POLARS_DTYPE: dict[str, pl.DataType] = {
    "str": pl.Utf8,
    "cat": pl.Categorical,
    "f32": pl.Float32,
    "f64": pl.Float64,
    "i32": pl.Int32,
    "i16": pl.Int16,
    "i8": pl.Int8,
}

# ---------------------------------------------------------------------------
# 2. Categorical code maps (01_SCHEMA.md §2)
#    Applied after loading; keep the raw code too for joins/audit.
# ---------------------------------------------------------------------------
CHANNEL = {"R": "Retail", "B": "Broker", "C": "Correspondent"}
LOAN_PURPOSE = {
    "P": "Purchase",
    "C": "Cash-out refi",
    "R": "No-cash-out refi",
    "U": "Refi-unspecified",
}
OCCUPANCY = {"P": "Principal", "S": "Second", "I": "Investor", "U": "Unknown"}
PROPERTY = {
    "SF": "Single-family",
    "CO": "Condo",
    "CP": "Co-op",
    "MH": "Manufactured",
    "PU": "PUD",
}
AMORT_TYPE = {"FRM": "Fixed", "ARM": "Adjustable"}
YESNO = {"Y": True, "N": False}  # also "7"/"9"/"" → null where they appear

# ---------------------------------------------------------------------------
# 4. Seven-state target maps (01_SCHEMA.md §4)
# ---------------------------------------------------------------------------
# Zero Balance Code → terminal state. Present only in the loan's final month.
ZERO_BAL_STATE: dict[str, str] = {
    "01": "prepaid",       # prepaid / matured (paid in full)
    "02": "foreclosure",   # third-party sale (terminal credit event)
    "03": "foreclosure",   # short sale
    "06": "prepaid",       # repurchased (non-credit exit; flag separately)
    "09": "REO",           # REO disposition / deed-in-lieu
    "15": "foreclosure",   # note sale
    "16": "prepaid",       # reperforming loan sale (flag)
    "96": "prepaid",       # removal (non-credit) (flag)
    "97": "foreclosure",   # delinquency/credit-event repurchase
    "98": "foreclosure",   # delinquency/credit-event repurchase
}

# The seven-label space.
STATES: tuple[str, ...] = (
    "current", "dpd_30", "dpd_60", "dpd_90plus", "foreclosure", "REO", "prepaid",
)

# Codes that should be treated as "flagged" non-credit exits (06/16/96). Carried
# for audit; the state itself is still ``prepaid`` per the map above.
ZERO_BAL_FLAGGED: frozenset[str] = frozenset({"06", "16", "96"})

# ---------------------------------------------------------------------------
# 5. Model keep-list (01_SCHEMA.md §5) — exact COLUMNS names.
#    Shorthand in the spec is resolved to the canonical positional names here.
# ---------------------------------------------------------------------------
KEEP_COLS: list[str] = [
    "Loan Identifier",
    "Monthly Reporting Period",
    "Channel",
    "Original Interest Rate",
    "Current Interest Rate",
    "Original UPB",
    "Current Actual UPB",
    "Original Loan Term",
    "Origination Date",
    "First Payment Date",
    "Loan Age",
    "Remaining Months to Maturity",
    "Original Loan-to-Value (LTV)",
    "Original Combined LTV (CLTV)",
    "Number of Borrowers",
    "Debt-to-Income (DTI)",
    "Borrower Credit Score at Origination",
    "Co-Borrower Credit Score at Origination",
    "Origination Classic FICO",   # new source; coalesced with the above (FICO migration, Dec 2025+)
    "First-Time Home Buyer Indicator",
    "Loan Purpose",
    "Property Type",
    "Number of Units",
    "Occupancy Status",
    "Property State",
    "Metropolitan Statistical Area (MSA)",
    "Zip Code Short",
    "Mortgage Insurance Percentage",
    "Amortization Type",
    "Interest-Only Loan Indicator",
    "Current Loan Delinquency Status",
    "Modification Flag",
    "Zero Balance Code",
    "Zero Balance Effective Date",
    "Unscheduled Principal Current",
]

# Derived columns added by Stage 4 (not present in the raw CSV).
DERIVED_COLS: list[str] = ["state", "state_next", "censored"]

# Every keep-list source column must be a real positional column.
_missing = [c for c in KEEP_COLS if c not in COLUMNS]
assert not _missing, f"KEEP_COLS not in COLUMNS: {_missing}"


# ---------------------------------------------------------------------------
# 6.4. Feature roles (01_SCHEMA.md §6.4) — over the CLEAN/PANEL column names.
#   Source of truth for Stage 5 (train-only scalers) and Stage 6 (scale stats).
#   The lake is never standardised; these only classify how each column is used.
# ---------------------------------------------------------------------------
FEATURE_SPEC: dict[str, list[str]] = {
    # standardize at train time (train-slice stats only); LOG_TRANSFORM first
    "continuous_standardize": [
        "Original Interest Rate", "Current Interest Rate",
        "Original UPB", "Current Actual UPB",
        "Original Loan Term", "Loan Age", "Remaining Months to Maturity",
        "Original Loan-to-Value (LTV)", "Original Combined LTV (CLTV)",
        "Debt-to-Income (DTI)", "fico_orig", "fico_co",
        "Mortgage Insurance Percentage",
        "Number of Borrowers", "Number of Units",
    ],
    # Present in the layout but EMPTY in SF Loan Performance (100% null per Stage 6
    # QA — it is a CAS/CIRT-only field). Kept in the lake for faithfulness, but
    # excluded from the model feature set.
    "excluded": ["Unscheduled Principal Current"],
    "categorical": [  # embed / one-hot
        "Channel", "Loan Purpose", "Property Type", "Occupancy Status",
        "Property State", "Metropolitan Statistical Area (MSA)", "Zip Code Short",
        "Amortization Type",
    ],
    "binary": [  # 0/1 (incl. missingness indicators)
        "First-Time Home Buyer Indicator", "Interest-Only Loan Indicator",
        "Modification Flag", "fico_orig_missing", "fico_co_missing",
        "dti_missing", "cltv_missing", "mi_pct_missing",
    ],
    "time": [  # masking / cyclical encoding
        "period", "period_ym", "orig_date", "orig_ym",
        "First Payment Date", "Zero Balance Effective Date",
    ],
    "target": ["state", "state_next", "censored"],
    "split": ["shard"],
    "identifier": ["Loan Identifier"],
    # raw target drivers kept for derivation/audit; redundant with `state` as
    # features, so excluded from the model feature set.
    "target_driver": ["Current Loan Delinquency Status", "Zero Balance Code"],
}

# Heavily right-skewed dollar columns → apply log1p before standardizing.
LOG_TRANSFORM: list[str] = [
    "Original UPB", "Current Actual UPB",
]

# Self-consistency: no column appears in two roles.
_all_roles = [c for cols in FEATURE_SPEC.values() for c in cols]
assert len(_all_roles) == len(set(_all_roles)), "FEATURE_SPEC: a column is in two roles"


# ---------------------------------------------------------------------------
# 3. Type-casting & sentinel helpers (01_SCHEMA.md §3)
#    All return Polars expressions so they compose inside lazy scans.
# ---------------------------------------------------------------------------
def _as_expr(col: str | pl.Expr) -> pl.Expr:
    return pl.col(col) if isinstance(col, str) else col


def parse_mmyyyy(col: str | pl.Expr) -> pl.Expr:
    """Parse a ``MMYYYY`` (6-char) string to a first-of-month ``date``.

    ``MM`` = chars 0-1, ``YYYY`` = chars 2-5 → ``date(YYYY, MM, 1)``.
    Empty / malformed (not exactly 6 chars, or non-numeric) → null.
    """
    s = _as_expr(col).cast(pl.Utf8, strict=False).str.strip_chars()
    s = pl.when(s.str.len_chars() == 6).then(s).otherwise(None)
    month = s.str.slice(0, 2).cast(pl.Int32, strict=False)
    year = s.str.slice(2, 4).cast(pl.Int32, strict=False)
    return pl.date(year, month, 1)


def scrub_credit_score(col: str | pl.Expr) -> pl.Expr:
    """Credit score → Int16, with sentinels (9999, 0, <300, >850) → null."""
    c = _as_expr(col).cast(pl.Int32, strict=False)
    return (
        pl.when(c.is_between(300, 850))  # inclusive
        .then(c)
        .otherwise(None)
        .cast(pl.Int16)
    )


def fico_orig_expr(
    old_col: str = "Borrower Credit Score at Origination",
    new_col: str = "Origination Classic FICO",
) -> pl.Expr:
    """Origination FICO, coalesced across the field migration → ``fico_orig``.

    The old `Borrower Credit Score at Origination` (col 23) stops populating from
    the March 2026 activity period; the new `Origination Classic FICO` (col 110)
    populates from December 2025. Our data straddles that switch, so prefer the
    old field where present and fall back to the new one (both scrubbed). Both are
    Classic FICO at origination; semantics differ only slightly (primary-borrower
    vs lowest-representative), acceptable for modelling.
    """
    return (
        scrub_credit_score(old_col)
        .fill_null(scrub_credit_score(new_col))
        .alias("fico_orig")
    )


def blank_to_null(col: str | pl.Expr) -> pl.Expr:
    """Empty-string (after trim) → null; otherwise pass through unchanged.

    Numeric blanks must become null, not 0 — many monetary fields are blank
    until an event occurs.
    """
    c = _as_expr(col)
    trimmed = c.cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.when(trimmed.str.len_chars() == 0).then(None).otherwise(c)


# ---------------------------------------------------------------------------
# 4. Seven-state derivation (01_SCHEMA.md §4)
# ---------------------------------------------------------------------------
def derive_state(
    delinq_col: str = "Current Loan Delinquency Status",
    zb_col: str = "Zero Balance Code",
) -> pl.Expr:
    """Return a Polars expr giving the per-loan-month seven-state ``state``.

    Logic (Zero Balance Code takes precedence, as it appears only in the
    terminating month):

        if Zero Balance Code is a known terminal code:
            state = ZERO_BAL_STATE[zb]          # prepaid / foreclosure / REO
        elif delinquency status == 0:  current
        elif == 1:                     dpd_30
        elif == 2:                     dpd_60
        elif >= 3:                     dpd_90plus
        else:                          null     # XX / blank / unknown

    Delinquency status is cast to int so non-numeric ``XX`` → null naturally
    (avoids the lexical-comparison trap where ``"XX" >= "03"``). An unknown /
    unmapped Zero Balance Code falls through to the delinquency logic rather
    than guessing a terminal state.
    """
    zb = pl.col(zb_col).cast(pl.Utf8, strict=False).str.strip_chars()
    dq = (
        pl.col(delinq_col)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .cast(pl.Int32, strict=False)
    )
    zb_keys = list(ZERO_BAL_STATE.keys())
    return (
        pl.when(zb.is_in(zb_keys))
        .then(zb.replace_strict(ZERO_BAL_STATE, default=None))
        .when(dq == 0).then(pl.lit("current"))
        .when(dq == 1).then(pl.lit("dpd_30"))
        .when(dq == 2).then(pl.lit("dpd_60"))
        .when(dq >= 3).then(pl.lit("dpd_90plus"))
        .otherwise(None)
        .alias("state")
    )
