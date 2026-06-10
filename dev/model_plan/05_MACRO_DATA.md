# 05 — Macro Data: Sources, Download & Integration Spec (standalone)

> **How to use this file.** Self-contained and executable: hand it to Claude Code with *"Read `dev/model_plan/05_MACRO_DATA.md` and execute its tasks MD1–MD4 in order."* It does not require reading the rest of the plan. §1–§2 are for Felipe (where the data comes from, what to click/run); §3–§6 are the build/join/QA contract for Claude Code.
>
> **Design (from `00_OVERVIEW §3.4`):** macro enters as two tiny time-keyed Parquet tables joined at **export time** — the 800 GB lakes and the pipeline stages are never touched or re-run.

---

## 1. The variable set

| Variable | Series / source | Native freq | Geography | → Monthly value | Join lag* |
|---|---|---|---|---|---|
| 30-yr mortgage rate `pmms30` | FRED `MORTGAGE30US` (Freddie Mac PMMS) | weekly | national | calendar-month mean | 0 m |
| 10-yr Treasury `dgs10` | FRED `DGS10` | daily | national | calendar-month mean | 0 m |
| Curve slope `slope_10y2y` | FRED `DGS10` − `DGS2` | daily | national | month mean of spread | 0 m |
| Unemployment (nat.) `unrate_nat` | FRED `UNRATE` (BLS CPS, SA) | monthly | national | as-is | 1 m |
| Unemployment (state) `unrate_state` | FRED `{POSTAL}UR` (BLS LAUS, SA), e.g. `CAUR`, `TXUR`, `DCUR` | monthly | 50 states + DC | as-is | 1 m |
| House-price index `hpi_nat`, `hpi_state` | **Freddie Mac FMHPI master file** (SA index) | monthly | national + states | as-is | 2 m |

\* **Join lag** = publication realism: month-*t* unemployment is released early in *t+1*; FMHPI for month *t* about two months later; rates are real-time. A loan-month with `period_ym = t` is joined to the macro value of month `t − lag`. Lags live in one config dict — easy to change, documented in the writeup.

**Conventions:** final-revision series (not real-time vintages) — stated limitation. Seasonally adjusted versions throughout. HPI *levels* are never model features; only derived ratios are (`02_LOAN_LEVEL §2`).

## 2. Where to download (Felipe — ~15 minutes, no API keys)

Create a dated snapshot directory on the SSD first: `/Volumes/SSD Felipe/dissertation/raw/macro/<YYYY-MM-DD>/`.

> **Invariant amendment (deliberate, one-time):** `raw/` immutability now reads "written once at download time, never modified after." `raw/Performance_All/` remains untouched. The fetch script (MD1) may write *only* into a new dated snapshot folder.

1. **FRED series** (PMMS, DGS10, DGS2, UNRATE, 51 state UR series) — keyless CSV endpoint, one URL per series:
   `https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US` (etc.)
   Don't click 55 links — task MD1 gives Claude Code a polite loop. If you prefer manual: each series page on fred.stlouisfed.org → *Download → CSV*.
2. **FMHPI** — https://www.freddiemac.com/research/indices/house-price-index → download the **master file** CSV (one file: US + all states + MSAs, monthly, NSA & SA columns). Save into the snapshot folder. (Site occasionally requires a browser download — that's why this one is listed as manual.)
3. That's all. Everything else is Claude Code's job (MD1–MD4).

## 3. Build spec — two output tables

Built by `dev/model/build_macro.py` (idempotent; reads the newest snapshot under `raw/macro/`, or a `--snapshot` argument) into:

**`processed/macro/macro_national.parquet`** — one row per month, 2000-01 → snapshot end:

| col | def |
|---|---|
| `month_ym` (i32, key) | observation month `YYYYMM` |
| `pmms30`, `dgs10`, `slope_10y2y`, `unrate_nat` (f32) | per §1 |

**`processed/macro/macro_state.parquet`** — one row per (state, month):

| col | def |
|---|---|
| `state` (cat, key) | 2-letter postal code |
| `month_ym` (i32, key) | observation month |
| `unrate_state`, `hpi_state` (f32) | per §1 |
| also: national rows | a pseudo-state `US` row per month carrying `unrate_nat` / `hpi_nat`, used as the fallback target |

Rules: store values **by observation month** (lags are applied at join time, not baked in); parse FRED CSVs defensively (`.` = missing → null); month-mean aggregation for weekly/daily series uses only observations within the calendar month; assert monotone, gap-free `month_ym` per series (forward-fill single-month gaps, fail on longer ones).

## 4. Join spec (consumed by `export.py` — see `02_LOAN_LEVEL §3`)

```sql
-- lags from config: LAG = {pmms30:0, dgs10:0, slope_10y2y:0, unrate:1, hpi:2}
-- national: join macro_national ON month_ym = sub_months(period_ym, LAG[var])  (one join per lag group)
-- state:    join macro_state    ON (state = property_state, month_ym = sub_months(period_ym, LAG[var]))
```

- **Fallback:** if `(property_state, month)` is missing (territories: PR/VI/GU; series start gaps), use the `US` pseudo-state row and set `unrate_fallback = 1` (resp. `hpi_fallback = 1`). Indicators are model features (binary role).
- **Derived features** (computed in `features.py`, need loan columns): `incentive = current_rate − pmms30`; `ltv_mtm = (current_upb / orig_upb) × orig_ltv × hpi_state(orig_ym) / hpi_state(period_ym)` (HPI at both ends from the same lagged join convention; if orig_ym predates the HPI series, fall back to `hpi_nat` at both ends); `hpi_chg_12m = hpi_state(t)/hpi_state(t−12) − 1`.
- `feature_spec` additions: `pmms30, dgs10, slope_10y2y, unrate_nat, unrate_state, incentive, ltv_mtm, hpi_chg_12m` → `continuous_standardize`; `unrate_fallback, hpi_fallback` → `binary`. HPI levels → role `intermediate` (never a feature).

## 5. Tasks for Claude Code (execute in order)

### MD1 — Fetch script + snapshot
`dev/model/fetch_macro.py`: loop the FRED CSV endpoint over `[MORTGAGE30US, DGS10, DGS2, UNRATE] + [f"{p}UR" for p in 50 states + DC]` with a ~1 s sleep, writing raw CSVs to `raw/macro/<today>/fred/`; verify the FMHPI master file is already present in the snapshot (manual step — fail with a clear message if absent). Record a `snapshot_manifest.json` (file list, sizes, sha256, fetch date).
**Accept:** 55 FRED CSVs present, each parsing to ≥ 300 monthly rows (≥ 26 yrs) after aggregation; manifest written; re-run into the same dated folder is a no-op.

### MD2 — `build_macro.py` → the two Parquet tables
Per §3. Defensive parsing; FMHPI schema asserted on load (expect Year/Month, GEO_Type ∈ {US, State, CBSA}, GEO_Name, SA index column — names may drift, resolve by inspection not assumption).
**Accept:** `macro_national` has one gap-free row per month 2000-01→snapshot end; `macro_state` covers 50 states + DC + `US` rows with both series, gap-free per state; spot-check 3 known values against the sources (e.g. UNRATE 2009-10 = 10.0, PMMS ~2.65–2.7 around 2021-01, NV HPI drawdown ≈ −50%+ peak-to-trough 2006→2012).

### MD3 — Wire into export + features
Implement §4 in `export.py` / `features.py` / `feature_spec`, including fallback indicators and derived features, with unit tests (synthetic loan: known UPBs/LTV/HPI path → exact `ltv_mtm`; lag correctness: a loan-month at `period_ym = 202001` gets `unrate` of 2019-12 and `hpi` of 2019-11).
**Accept:** unit tests pass; a 100k-row dev export has zero null macro columns (fallbacks engaged where needed) and join-rate stats logged.

### MD4 — EDA validation figures
F4.2 macro small-multiples and F4.3 state-dispersion (`01_EDA §4`), saved to `outputs/figures/eda/`.
**Accept:** figures regenerate from `dev/analysis/` scripts; series shapes match known history (2008–09 unemployment spike, 2012 & 2021 rate troughs, 2006→12 HPI bust).

## 6. Acceptance block (the whole of M2b in `04_TASKS.md`)

MD1–MD4 all pass; `backup_ssd.sh` already covers `processed/macro/` (verify the two new tables appear in `ssd_mirror/` after a run); `00_OVERVIEW`/`02_LOAN_LEVEL` feature lists and this file agree on column names (single source of truth for lags: the config dict).
