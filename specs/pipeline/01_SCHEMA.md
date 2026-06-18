# 01 — Schema, Dtypes & Modelling Target

> Single source of truth for the column layout, types, categorical code maps, date parsing, and the Sirignano seven-state target. Claude Code must translate this into `dev/pipeline/schema.py`.

---

## 0. Critical facts

- The CSVs have **no header**. Column identity is **positional** — the order below *is* the contract.
- Measured field count: **113** (split on `|`). Field 0 is `Reference Pool ID` and is usually blank; many trailing fields are blank by design.
- Fannie Mae periodically adds columns. Acquisition quarters run to ~2025 with reporting periods to `2025-12`, so the newest 113-field layout applies (payment-deferral, alternative-delinquency-resolution, and the Dec-2025 Classic FICO fields).
- **Verification done.** Reconciled position-for-position against the official Fannie Mae *"Single-Family Loan Performance Dataset and Credit Risk Transfer — Glossary and File Layout"* (`docs/crt-file-layout-and-glossary.pdf`, 113 fields). All names confirmed (positions 110–112 = the Classic FICO trio; position 78 = `Special Eligibility Program`). Stage 1 still asserts `measured_field_count == 113` per file and quarantines any file whose count differs rather than guessing.

## 0a. Data model — vintage vs reporting period vs release (read this first)

Three different things are all called "quarter" in this dataset. Conflating them is the most common modelling error, so pin them down:

1. **Acquisition (vintage) quarter — identifies the *file*.** Each `YYYYQn.csv` is one origination cohort: the loans Fannie Mae *acquired* in that quarter, carrying *every* monthly performance record those specific loans ever produce. A loan appears in **exactly one** file. There is **no loan-level overlap** between files, so concatenating all ~100 vintage files is the correct way to assemble the full sample — dropping a file deletes that entire cohort (missing data, **not** de-duplication).
2. **Monthly reporting period — identifies the *row*.** One row per loan per month (`Monthly Reporting Period`, col 2). A 30-year loan still alive contributes up to ~360 rows. This time dimension — not duplication — is why the dataset is ~800 GB. Note the static origination fields (rate, LTV, FICO, …) are **denormalised**: repeated on every monthly row of a loan. That repetition is expected and compresses away under Parquet dictionary/RLE encoding.
3. **Release / refresh — Fannie Mae republishes the *whole* dataset ~quarterly**, each time extending every vintage's performance history by another quarter. "Download all files every quarter" (per the official tutorial) means *re-pull the whole set on a new release to get fresher months* — it does **not** mean stacking releases.

**The only genuine redundancy risk is mixing releases.** If different vintage files came from different releases you get inconsistent right-censoring cut-offs (and duplicate vintages if the same quarter was pulled twice). Mitigation: all files must come from **one release**. Stage 1 records each file's `max(Monthly Reporting Period)` and flags any vintage whose cut-off diverges from the modal one. Within a single consistent release there is **zero** cross-file redundancy.

**Scope:** this public dataset is a *subset* — 30-year fixed-rate, fully amortizing, fully documented, conventional, LTV ≤ 97%; it excludes ARMs, balloons, interest-only, HARP/Refi Plus, reduced-doc, and government-insured loans. (So some ARM/IO columns in §1 are near-empty here — present for layout compatibility with the CRT remittance files.)

## 1. Column layout (positional)

The list below is the canonical ordered layout. Use it to build `COLUMNS = [...]`. Positions are 0-indexed. Dtype column gives the **target Parquet dtype** (read everything as string first, then cast — see §3). `cat` = dictionary-encoded categorical, `f32`/`f64` = float, `i32`/`i16`/`i8` = nullable integer, `date` = parsed from `MMYYYY`, `str` = free string/id.

| # | Column | Dtype | Notes |
|---|---|---|---|
| 0 | Reference Pool ID | str | usually blank |
| 1 | Loan Identifier | str | **key**; stable across months for one loan |
| 2 | Monthly Reporting Period | date | `MMYYYY`; the panel time index |
| 3 | Channel | cat | R/B/C (see §2) |
| 4 | Seller Name | cat | high-cardinality; dictionary-encode |
| 5 | Servicer Name | cat | |
| 6 | Master Servicer | cat | |
| 7 | Original Interest Rate | f32 | |
| 8 | Current Interest Rate | f32 | |
| 9 | Original UPB | f32 | |
| 10 | UPB at Issuance | f32 | |
| 11 | Current Actual UPB | f32 | drives prepay/curtailment |
| 12 | Original Loan Term | i16 | months |
| 13 | Origination Date | date | `MMYYYY` |
| 14 | First Payment Date | date | `MMYYYY` |
| 15 | Loan Age | i16 | months since origination |
| 16 | Remaining Months to Legal Maturity | i16 | |
| 17 | Remaining Months to Maturity | i16 | |
| 18 | Maturity Date | date | `MMYYYY` |
| 19 | Original Loan-to-Value (LTV) | f32 | |
| 20 | Original Combined LTV (CLTV) | f32 | |
| 21 | Number of Borrowers | i8 | |
| 22 | Debt-to-Income (DTI) | f32 | |
| 23 | Borrower Credit Score at Origination | i16 | sentinel 9999 → null |
| 24 | Co-Borrower Credit Score at Origination | i16 | sentinel 9999 → null |
| 25 | First-Time Home Buyer Indicator | cat | Y/N |
| 26 | Loan Purpose | cat | P/C/R/U (see §2) |
| 27 | Property Type | cat | SF/CO/CP/MH/PU |
| 28 | Number of Units | i8 | 1–4 |
| 29 | Occupancy Status | cat | P/S/I |
| 30 | Property State | cat | 2-letter |
| 31 | Metropolitan Statistical Area (MSA) | cat | 5-digit code |
| 32 | Zip Code Short | cat | 3-digit (privacy-truncated) |
| 33 | Mortgage Insurance Percentage | f32 | |
| 34 | Amortization Type | cat | FRM/ARM |
| 35 | Prepayment Penalty Indicator | cat | Y/N |
| 36 | Interest-Only Loan Indicator | cat | Y/N |
| 37 | Interest-Only First P&I Payment Date | date | |
| 38 | Months to Amortization | i16 | |
| 39 | **Current Loan Delinquency Status** | str | **target driver**; `00`,`01`,`02`,…,`XX` (see §4) |
| 40 | Loan Payment History | str | 24-month delinquency string |
| 41 | Modification Flag | cat | Y/N |
| 42 | Mortgage Insurance Cancellation Indicator | cat | |
| 43 | **Zero Balance Code** | cat | **target driver**; 01/02/03/06/09/15/16/96/97 (see §4) |
| 44 | Zero Balance Effective Date | date | |
| 45 | UPB at the Time of Removal | f32 | |
| 46 | Repurchase Date | date | |
| 47 | Scheduled Principal Current | f32 | |
| 48 | Total Principal Current | f32 | |
| 49 | Unscheduled Principal Current | f32 | curtailment signal |
| 50 | Last Paid Installment Date | date | |
| 51 | Foreclosure Date | date | |
| 52 | Disposition Date | date | |
| 53 | Foreclosure Costs | f32 | |
| 54 | Property Preservation & Repair Costs | f32 | |
| 55 | Asset Recovery Costs | f32 | |
| 56 | Misc Holding Expenses & Credits | f32 | |
| 57 | Associated Taxes for Holding Property | f32 | |
| 58 | Net Sales Proceeds | f32 | |
| 59 | Credit Enhancement Proceeds | f32 | |
| 60 | Repurchase Make-Whole Proceeds | f32 | |
| 61 | Other Foreclosure Proceeds | f32 | |
| 62 | Modification Non-Interest-Bearing UPB | f32 | |
| 63 | Principal Forgiveness Amount | f32 | |
| 64 | Original List Start Date | date | |
| 65 | Original List Price | f32 | |
| 66 | Current List Start Date | date | |
| 67 | Current List Price | f32 | |
| 68 | Borrower Credit Score at Issuance | i16 | |
| 69 | Co-Borrower Credit Score at Issuance | i16 | |
| 70 | Borrower Credit Score Current | i16 | |
| 71 | Co-Borrower Credit Score Current | i16 | |
| 72 | Mortgage Insurance Type | cat | 1/2/3 |
| 73 | Servicing Activity Indicator | cat | |
| 74 | Current Period Modification Loss Amount | f32 | |
| 75 | Cumulative Modification Loss Amount | f32 | |
| 76 | Current Period Credit Event Net Gain/Loss | f32 | |
| 77 | Cumulative Credit Event Net Gain/Loss | f32 | |
| 78 | Special Eligibility Program | cat | glossary fld 79; replaced old HomeReady Indicator (Jan 2023) |
| 79 | Foreclosure Principal Write-off Amount | f32 | |
| 80 | Relocation Mortgage Indicator | cat | Y/N |
| 81 | Zero Balance Code Change Date | date | |
| 82 | Loan Holdback Indicator | cat | |
| 83 | Loan Holdback Effective Date | date | |
| 84 | Delinquent Accrued Interest | f32 | |
| 85 | Property Valuation Method | cat | |
| 86 | High Balance Loan Indicator | cat | Y/N |
| 87 | ARM Initial Fixed-Rate Period ≤5yr Indicator | cat | |
| 88 | ARM Product Type | cat | |
| 89 | Initial Fixed-Rate Period | i16 | |
| 90 | Interest Rate Adjustment Frequency | i16 | |
| 91 | Next Interest Rate Adjustment Date | date | |
| 92 | Next Payment Change Date | date | |
| 93 | Index | cat | |
| 94 | ARM Cap Structure | cat | |
| 95 | Initial Interest Rate Cap Up Percent | f32 | |
| 96 | Periodic Interest Rate Cap Up Percent | f32 | |
| 97 | Lifetime Interest Rate Cap Up Percent | f32 | |
| 98 | Mortgage Margin | f32 | |
| 99 | ARM Balloon Indicator | cat | |
| 100 | ARM Plan Number | cat | |
| 101 | Borrower Assistance Plan | cat | |
| 102 | High LTV Refinance Option Indicator | cat | |
| 103 | Deal Name | cat | |
| 104 | Repurchase Make-Whole Proceeds Flag | cat | |
| 105 | Alternative Delinquency Resolution | cat | newer field |
| 106 | Alternative Delinquency Resolution Count | i16 | newer field |
| 107 | Total Deferral Amount | f32 | newer field |
| 108 | Payment Deferral Modification Event Indicator | cat | newer field |
| 109 | Interest-Bearing UPB | f32 | newer field |
| 110 | Origination Classic FICO | i16 | glossary fld 111; populated Dec 2025+ |
| 111 | Issuance Classic FICO | i16 | glossary fld 112; populated Dec 2025+ |
| 112 | Current Classic FICO | i16 | glossary fld 113; populated Dec 2025+ |

> **Reconciled against the official glossary (`docs/crt-file-layout-and-glossary.pdf`, 113 fields).** Positions 110–112 are **confirmed** as the new Classic FICO trio (glossary fields 111–113), not reserved — they carry origination/issuance/current Classic FICO scores that begin populating with the **December 2025** activity period. Position 78 is **`Special Eligibility Program`** (glossary fld 79), which replaced the legacy HomeReady Indicator in the Jan 2023 release. The assertion `len(COLUMNS) == 113` holds and the layout matches the glossary position-for-position.

> **FICO field migration (important).** The legacy `Borrower/Co-Borrower Credit Score at Origination` (cols 23/24) **stop populating from the March 2026 activity period**, while `Origination Classic FICO` (col 110) **starts from December 2025**. Because this dataset's reporting horizon reaches `2025-12`, it straddles the switch. Stage 3 must therefore derive `fico_orig = coalesce(Borrower Credit Score at Origination, Origination Classic FICO)` (both sentinel-scrubbed) so origination FICO stays populated across the transition — see `schema.fico_orig_expr()`. `Issuance`/`Current Classic FICO` remain available in the full perf lake for an optional dynamic-FICO feature later.

## 2. Categorical code maps (model-relevant)

Apply these as label mappings *after* loading (keep raw code too, for joins/audit):

```python
CHANNEL      = {"R": "Retail", "B": "Broker", "C": "Correspondent"}
LOAN_PURPOSE = {"P": "Purchase", "C": "Cash-out refi", "R": "No-cash-out refi", "U": "Refi-unspecified"}
OCCUPANCY    = {"P": "Principal", "S": "Second", "I": "Investor"}
PROPERTY     = {"SF": "Single-family", "CO": "Condo", "CP": "Co-op", "MH": "Manufactured", "PU": "PUD"}
AMORT_TYPE   = {"FRM": "Fixed", "ARM": "Adjustable"}
YESNO        = {"Y": True, "N": False}   # also handle "7"/"9"/"" → null where it appears
```

## 3. Type-casting & sentinel rules

1. **Read everything as string first** (PyArrow `column_types=string`, or DuckDB `all_varchar=true`), then cast. This avoids type-inference crashes on dirty fields and guarantees stable schema across quarters.
2. **Dates** are `MMYYYY` (6 chars). Parse to first-of-month: `MM` = chars 0–1, `YYYY` = chars 2–5 → `date(YYYY, MM, 1)`. Empty string → null. Build one reusable parser; apply to every `date` column.
3. **Credit scores**: values `9999`, `0`, or `< 300`/`> 850` → null (sentinels for missing).
4. **Numeric blanks** (`""`) → null, not `0`. Many monetary fields are blank until an event occurs (e.g. foreclosure costs).
5. **Downcast** floats to `f32` and integers to the smallest nullable int that fits — this is most of the memory/disk saving. Use nullable integer types (Arrow/Polars handle this natively; in pandas use `Int16` etc.).
6. **Dictionary-encode** all `cat` columns in Parquet — large repeated strings like `Seller Name` shrink dramatically.

## 4. Target: the Sirignano seven states

Derive a per-loan-month categorical `state` from `Current Loan Delinquency Status` (col 39) and `Zero Balance Code` (col 43). The Zero Balance Code only appears in the loan's final (terminating) month; in all prior months it is blank and the delinquency status governs.

**`Current Loan Delinquency Status` (col 39):**

| Code | Meaning |
|---|---|
| `00` | Current |
| `01` | 30 days delinquent (1 missed payment) |
| `02` | 60 days delinquent |
| `03` | 90 days delinquent |
| `04`+ | 120+ days delinquent |
| `XX` | unknown / not reported → null |

**`Zero Balance Code` (col 43)** — present only in the terminating month:

| Code | Meaning | Mapped state |
|---|---|---|
| `01` | Prepaid / matured (paid in full) | `prepaid` |
| `02` | Third-party sale | `foreclosure` (terminal credit event) |
| `03` | Short sale | `foreclosure` |
| `06` | Repurchased | `prepaid` (treat as non-credit exit; flag separately) |
| `09` | REO disposition / deed-in-lieu | `REO` |
| `15` | Note sale | `foreclosure` |
| `16` | Reperforming loan sale | `prepaid` (flag) |
| `96` | Removal (non-credit) | `prepaid` (flag) |
| `97`/`98` | Delinquency/credit-event repurchase | `foreclosure` |

**State derivation logic (per loan-month):**

```text
if Zero Balance Code is present:
    state = map_zero_balance(zb_code)          # prepaid / foreclosure / REO
elif delinquency_status == "00":   state = "current"
elif delinquency_status == "01":   state = "dpd_30"
elif delinquency_status == "02":   state = "dpd_60"
elif delinquency_status >= "03":   state = "dpd_90plus"
else:                              state = None   # XX / blank
```

Seven label space: `{current, dpd_30, dpd_60, dpd_90plus, foreclosure, REO, prepaid}`.

**`state_next`** = the same `Loan Identifier`'s `state` in the next `Monthly Reporting Period`, obtained with a window function ordered by date partitioned by loan (see `02_PIPELINE_STAGES.md §4`). The supervised target is the transition `state → state_next`. The last observed month of each loan has `state_next = absorbing terminal` (prepaid/foreclosure/REO) or null if the loan is simply right-censored at the data cut-off — keep a `censored` flag to distinguish.

## 5. Model keep-list (column projection for the cleaned lake)

To stay inside the disk budget, the **clean** and **panel** lakes carry only these (drop the rest, which are mostly post-default servicing/expense fields not used as predictors):

`Loan Identifier, Monthly Reporting Period, Channel, Original/Current Interest Rate, Original UPB, Current Actual UPB, Original Loan Term, Origination Date, First Payment Date, Loan Age, Remaining Months to Maturity, Original LTV, Original CLTV, Number of Borrowers, DTI, Borrower/Co-Borrower Credit Score at Origination, Origination Classic FICO, First-Time Home Buyer Indicator, Loan Purpose, Property Type, Number of Units, Occupancy Status, Property State, MSA, Zip Code Short, Mortgage Insurance Percentage, Amortization Type, Interest-Only Indicator, Current Loan Delinquency Status, Modification Flag, Zero Balance Code, Zero Balance Effective Date, Unscheduled Principal Current` (the three origination-FICO sources are coalesced into one `fico_orig` in Stage 3 — `01_SCHEMA.md` FICO-migration note) + the **derived training columns** of §6: `state`, `state_next`, `censored`, `period_ym`, `orig_ym`, `shard`, and the missingness indicators.

Keep the full 113-column **perf** lake too (cheap once compressed) so you can revisit dropped fields without re-parsing CSV.

## 6. Derived training columns, sharding & feature roles

These columns are added on top of the keep-list so the panel is ready for deep-learning training (minibatch SGD, rolling backtests) without reprocessing. They are **derived**, not raw — Stage 3 adds the calendar/missingness columns, Stage 4 adds the shard and the seven-state target.

### 6.1 Calendar columns (Stage 3) — for rolling backtests

| Column | Dtype | Definition |
|---|---|---|
| `period_ym` | i32 | Current reporting month as integer `YYYYMM`, from `Monthly Reporting Period`. The training-mask time index. |
| `orig_ym`   | i32 | Origination month as integer `YYYYMM`, from `Origination Date`. Loan vintage; also a feature. |

Keep the parsed `date` versions (`period`, `orig_date`) too; the `YYYYMM` ints exist for fast integer masking.

> **Rolling-backtest leakage rule.** Because `state_next` looks **one month ahead**, a backtest that trains "on data up to month `T`" must mask on the **target** month, not the feature month: include a loan-month example only if `period_ym < T` (so the labelled next month is ≤ `T`). Masking on `period_ym ≤ T` would leak the post-cutoff transition. `period_ym` makes this a one-line filter.

### 6.2 Shard column (Stage 4) — for minibatch randomization & splits

| Column | Dtype | Definition |
|---|---|---|
| `shard` | i16 | `deterministic_hash(Loan Identifier) % N_SHARDS`. A stable bucket in `[0, N_SHARDS)`. |

Design decisions:

- **Hash of the loan id, not a stored random number, and keyed on the loan, not the observation.** Deterministic → reproducible with no RNG seed to persist; **vintage-stable** → a re-released quarter keeps each loan's shard; **leakage-safe** → every month of a loan lands in the *same* shard (so a loan never straddles train/test, and sequence models stay intact).
- **Use a fixed-seed 64-bit hash of the `Loan Identifier` string**, then `% N_SHARDS`. Pin the seed/engine in `schema.py`; changing it reshuffles all shards (a one-time decision — document it).
- **`N_SHARDS` is a `config.py` knob (default 256)**, chosen for ~3–5M loan-months per shard at full scale so a shard comfortably fits in RAM. Adjust if the final row count differs materially.
- **Stage 4 writes the panel sorted by `(shard, Loan Identifier, period)`** so Parquet row-group min/max stats let `WHERE shard = k` **skip** non-matching row groups — cheap shard reads without a physical repartition. Within-loan month order is preserved (a loan's rows stay contiguous).
- **Training loop:** each epoch shuffle the shard order → load one shard (RAM-sized) → shuffle rows within → yield minibatches. Shard-order × within-shard shuffling gives good SGD randomness while staying streaming.
- **Split key (optional):** because shards are loan-disjoint, a fixed block (e.g. `shard < 0.1 · N_SHARDS`) can serve as a permanent held-out test set, orthogonal to the `period_ym` time axis. Not reserved by default — opt in at model time.

### 6.3 Missingness indicators (Stage 3) — informative nulls

Several fields are sentinel-scrubbed to null in §3; for credit models the *fact* of missingness is predictive, so carry a boolean alongside the scrubbed value:

`fico_orig_missing`, `fico_co_missing`, `dti_missing`, `cltv_missing`, `mi_pct_missing` (each `True` where the source was sentinel/blank).

### 6.4 Feature roles & standardization policy (`feature_spec`)

`schema.py` exposes a `feature_spec` classifying every kept/derived column by its modelling role, so the training code (and Stage 6 QA) treat each correctly:

| Role | Columns (indicative) | Handling |
|---|---|---|
| `continuous_standardize` | interest rates, `Original/Current Actual UPB` (**log-transform** — heavily skewed), loan term, loan age, remaining months, LTV, CLTV, DTI, FICO (borrower/co-borrower), MI %, unscheduled principal | standardize at train time |
| `categorical` | channel, loan purpose, property type, occupancy, property state, MSA, zip3, amort type, IO indicator, FTHB, modification flag, number of units/borrowers | embed / one-hot |
| `binary` | Y/N-derived flags, missingness indicators | pass through (0/1) |
| `identifier` | `Loan Identifier` | not a feature |
| `time` | `period_ym`, `orig_ym` (+ `period`, `orig_date`) | masking / optional cyclical encoding |
| `target` | `state`, `state_next`, `censored` | label |
| `split` | `shard` | batching / split, never a feature |

> **Standardization is NOT baked into the lake.** Writing standardized values to `processed/` would (a) leak test/future statistics into training and (b) be wrong for every rolling window, which needs its *own* train-only stats. Instead: flag the columns here; **compute mean/std (robust median/IQR where skewed) on the training slice only**, per backtest window, at model time; persist scalers to `models/scaler_<window>.json` for reproducible inference. Stage 6 emits per-column scale stats (mean/std/quantiles/null-rate) to `outputs/` to inform transform choices (e.g. confirm the log on UPB).

---

### Recap

The file is positional with 113 columns, no header; the order in §1 is the contract and Stage 1 must assert the count and reconcile against Fannie Mae's official glossary (positions 110–112 are version-dependent — name them generically, don't guess). Read all columns as strings, then cast: `MMYYYY` dates to first-of-month, credit-score and numeric sentinels to null, floats→f32, ints→smallest nullable, categoricals dictionary-encoded. The modelling target is the seven-state monthly transition derived from `Current Loan Delinquency Status` + `Zero Balance Code` per §4, with `state_next` via a per-loan window and a `censored` flag. The cleaned lake projects only the §5 keep-list to respect the disk budget. §6 adds the derived training columns: `period_ym`/`orig_ym` (integer year-months for rolling-backtest masking, with a one-month leakage rule on the target), a `shard` = `hash(loan_id) % N_SHARDS` for reproducible, leakage-safe minibatch randomization (panel sorted by `(shard, loan, period)` for row-group skipping), missingness indicators, and a `feature_spec` of ML roles — standardization is flagged here but fitted train-only at model time, never baked into the lake.
