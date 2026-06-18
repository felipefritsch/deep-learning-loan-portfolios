"""M6 unit tests — the feature pipeline (features.py) + shard-streaming loader (data.py).

Covers every M6 Accept criterion, hermetically (synthetic frames + synthetic parquet in
a temp dir; no SSD, no torch):

  * scaler is fit on the TRAIN slice only — val statistics differ;
  * UNK mapping for a synthetic unseen categorical level (and nulls);
  * ltv_mtm / hpi_chg_12m formulas on a synthetic loan (via features.derive_macro);
  * the loader yields every (masked) row exactly once per epoch, in shuffled order,
    with the window's period_ym mask honoured;
  * weights present and correct (current→current carries 1/p_keep).

Run:  .venv/bin/python -m floan.model.test_features     (or python -m pytest)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from floan.model import features as F
from floan.model import macro_features as mf
from floan.model import data as D


# ===========================================================================
# Scaler
# ===========================================================================
def test_scaler_train_only():
    """Fit on a train slice; the scaler must carry TRAIN stats, so a differently-
    distributed val slice does NOT standardize to mean 0 / std 1."""
    col = "Loan Age"
    train = pl.DataFrame({col: [10.0, 20.0, 30.0, 40.0, 50.0]})        # mean 30, std 14.14
    val = pl.DataFrame({col: [100.0, 110.0, 120.0]})                   # mean 110 — far off
    sc = F.Scaler.fit(train, cols=[col])

    assert abs(sc.center[col] - 30.0) < 1e-9                            # train mean, not val
    assert abs(sc.scale[col] - np.std([10, 20, 30, 40, 50])) < 1e-9    # population std (ddof=0)

    z_train = sc.transform(train)[:, 0]
    z_val = sc.transform(val)[:, 0]
    assert abs(z_train.mean()) < 1e-6 and abs(z_train.std() - 1.0) < 1e-6   # train ~N(0,1)
    assert z_val.mean() > 5.0                                           # val shifted (train-fit)
    # Re-fitting on val would have centred it — proving the loader must never do that.
    assert abs(F.Scaler.fit(val, cols=[col]).center[col] - 110.0) < 1e-9


def test_scaler_log_and_robust_and_null():
    """log1p on the dollar columns, robust median/IQR on the skewed ratios, null→0."""
    upb = "Original UPB"                                                # in F.LOG_COLS
    dti = "Debt-to-Income (DTI)"                                        # in F.ROBUST_COLS
    df = pl.DataFrame({
        upb: [0.0, np.expm1(1.0), np.expm1(2.0), np.expm1(3.0), np.expm1(4.0)],  # log1p → 0..4
        dti: [10.0, 20.0, 30.0, 40.0, 1000.0],                         # right-skewed
    })
    sc = F.Scaler.fit(df, cols=[upb, dti])
    # log applied before stats: centre == mean of {0,1,2,3,4} == 2.0.
    assert upb in sc.log_cols and abs(sc.center[upb] - 2.0) < 1e-9
    # robust: centre == median (30), scale == IQR/1.349, NOT mean/std (the outlier moves
    # the mean to ~220 but the median is unmoved → robust to the 1000 tail).
    assert dti in sc.robust_cols and abs(sc.center[dti] - 30.0) < 1e-9
    q25, q75 = df[dti].quantile(0.25), df[dti].quantile(0.75)
    assert abs(sc.scale[dti] - (q75 - q25) / 1.349) < 1e-9

    # null imputes to the centre → standardized 0; no NaN/Inf in the output matrix.
    z = sc.transform(pl.DataFrame({upb: [None, np.expm1(1.5)], dti: [None, 30.0]}))
    assert z[0, 0] == 0.0 and z[0, 1] == 0.0
    assert np.isfinite(z).all()


def test_scaler_zero_scale_guard():
    """A constant column has zero spread → scale falls back to 1.0 (no divide-by-zero)."""
    df = pl.DataFrame({"Loan Age": [7.0, 7.0, 7.0]})
    sc = F.Scaler.fit(df, cols=["Loan Age"])
    assert sc.scale["Loan Age"] == 1.0
    assert np.isfinite(sc.transform(df)).all()


# ===========================================================================
# Vocab + one-hot
# ===========================================================================
def test_vocab_unk_mapping():
    """Known train levels map to 1..V (sorted); unseen test levels and nulls → UNK (0)."""
    train = pl.DataFrame({"Channel": ["R", "B", "C", "R", "B"]})
    vocab = F.Vocab.fit(train, cols=["Channel"])
    assert vocab.maps["Channel"] == {"B": 1, "C": 2, "R": 3}            # sorted, 0 reserved
    assert vocab.vocab_size("Channel") == 4                             # |levels| + UNK

    test = pl.DataFrame({"Channel": ["R", "ZZZ", None, "B"]})           # ZZZ unseen, None null
    codes = vocab.transform(test)[:, 0]
    assert codes.tolist() == [3, F.UNK, F.UNK, 1]                       # unseen & null → 0


def test_vocab_min_count_prunes_to_unk():
    """A rare level below min_count is pruned into UNK."""
    train = pl.DataFrame({"Loan Purpose": ["P"] * 5 + ["C"] * 5 + ["RARE"]})
    vocab = F.Vocab.fit(train, cols=["Loan Purpose"], min_count=2)
    assert "RARE" not in vocab.maps["Loan Purpose"]
    assert vocab.transform(pl.DataFrame({"Loan Purpose": ["RARE"]}))[0, 0] == F.UNK


def test_one_hot_drop_first():
    """drop-first one-hot: UNK (0) is the all-zero reference; known levels 1..V get dummies."""
    codes = np.array([0, 1, 2, 3])                                     # UNK, B, C, R
    oh = F.one_hot(codes, vocab_size=4, drop_first=True)
    assert oh.shape == (4, 3)                                          # 4 cats → 3 dummies
    assert oh[0].tolist() == [0, 0, 0]                                # UNK = reference
    assert oh[1].tolist() == [1, 0, 0] and oh[3].tolist() == [0, 0, 1]
    # without drop_first the UNK column is retained.
    assert F.one_hot(codes, 4, drop_first=False).shape == (4, 4)


# ===========================================================================
# Derived macro features (ltv_mtm / hpi_chg_12m) on a synthetic loan
# ===========================================================================
def _month_range(a: int, b: int) -> list[int]:
    out, y, m = [], a // 100, a % 100
    while y * 100 + m <= b:
        out.append(y * 100 + m)
        m = m + 1 if m < 12 else 1
        y = y if m != 1 else y + 1
    return out


def test_derived_macro_formulas():
    """incentive = current_rate − pmms30; ltv_mtm = (cur/orig UPB)·orig_ltv·HPIo/HPIt;
    hpi_chg_12m = HPIt/HPI₋12 − 1 — checked through features.derive_macro on a known path."""
    months = _month_range(201401, 202012)
    hpi = {201411: 100.0, 201811: 110.0, 201911: 125.0}               # orig / 12m-ago / period
    nat = pl.DataFrame({
        "month_ym": months,
        "pmms30": [4.0 if m == 202001 else 9.0 for m in months],
        "dgs10": [2.0] * len(months), "slope_10y2y": [0.5] * len(months),
        "unrate_nat": [5.0] * len(months),
    }).with_columns(pl.col("month_ym").cast(pl.Int32))
    state = pl.DataFrame(
        [("CA", m, 5.0, hpi.get(m, 1.0)) for m in months],
        schema=["state", "month_ym", "unrate_state", "hpi_state"], orient="row",
    ).with_columns(pl.col("month_ym").cast(pl.Int32))
    loan = pl.DataFrame([dict(
        period_ym=202001, orig_ym=201501, property_state="CA",
        current_rate=6.0, current_upb=180000.0, orig_upb=200000.0, orig_ltv=80.0,
    )]).with_columns(pl.col("period_ym").cast(pl.Int32), pl.col("orig_ym").cast(pl.Int32))

    out = F.derive_macro(loan, nat, state).row(0, named=True)
    assert abs(out["incentive"] - 2.0) < 1e-9                          # 6.0 − 4.0
    assert abs(out["ltv_mtm"] - 57.6) < 1e-9                           # 0.9·80·(100/125)
    assert abs(out["hpi_chg_12m"] - (125 / 110 - 1)) < 1e-12           # 125/110 − 1


# ===========================================================================
# Loader (data.py) — synthetic parquet shards in a temp dir
# ===========================================================================
def _synthetic_pool(dirpath: Path, rows_per_part: list[list[int]]) -> dict[str, float]:
    """Write one parquet part per inner list of period_ym values. Every row gets a unique
    Loan Identifier; weight follows the current→current = 1/p_keep rule (p_keep=0.05 ⇒ 20).
    Returns {loan: expected_weight} for every row written."""
    P_KEEP = 0.05
    expected_w: dict[str, float] = {}
    lid = 0
    for pi, periods in enumerate(rows_per_part):
        n = len(periods)
        # alternate transitions so some rows are current→current (weight 20), some not.
        states = ["current" if i % 2 == 0 else "dpd_30" for i in range(n)]
        nexts = ["current" if i % 2 == 0 else "prepaid" for i in range(n)]
        loans = [f"L{pi}_{i:04d}" for i in range(n)]
        for i in range(n):
            w = 1.0 / P_KEEP if (states[i] == "current" and nexts[i] == "current") else 1.0
            expected_w[loans[i]] = w
        data: dict[str, list] = {
            "Loan Identifier": loans,
            "period_ym": periods,
            "orig_ym": [201001 + (i % 5) for i in range(n)],
            "label_ym": [mf.sub_months(p, -1) for p in periods],       # +1 month
            "weight": [1.0 / P_KEEP if states[i] == "current" and nexts[i] == "current"
                       else 1.0 for i in range(n)],
            "state": states, "state_next": nexts,
        }
        for j, c in enumerate(F.CONTINUOUS):
            data[c] = [float((i + j) % 17) for i in range(n)]
        for c in F.RAW_CATEGORICAL:
            if c == "state":
                continue
            data[c] = [("A" if i % 3 else "B") for i in range(n)]
        for c in F.BINARY:
            data[c] = [i % 2 for i in range(n)]
        pl.DataFrame(data).select(D.RAW_COLS).write_parquet(dirpath / f"part-{pi:04d}.parquet")
        lid += n
    return expected_w


def _fit_on_window(pool_dir: Path, bounds) -> tuple[F.Scaler, F.Vocab]:
    df = F.prepare_raw(
        pl.scan_parquet(str(pool_dir / "part-*.parquet"))
        .filter((pl.col("period_ym") >= bounds[0]) & (pl.col("period_ym") < bounds[1]))
        .select(D.RAW_COLS).collect())
    return F.Scaler.fit(df), F.Vocab.fit(df)


def test_loader_every_row_once_shuffled_and_masked():
    """One epoch yields every IN-WINDOW row exactly once, in shuffled (not on-disk) order;
    out-of-window rows (period_ym outside the mask) are excluded."""
    bounds = (200001, 201312)                                          # window k=2015 train mask
    in_win = [201201, 201205, 201301, 201306, 201311]
    out_win = [201312, 201400, 201500]                                 # ≥ Dec(2013): excluded
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _synthetic_pool(d, [in_win + out_win, in_win, in_win[:3] + out_win])
        scaler, vocab = _fit_on_window(d, bounds)
        ld = D.WindowLoader(d, bounds, scaler, vocab, batch_size=4, seed=7)

        # expected in-window (loan, period) multiset, computed directly from disk.
        disk = (pl.scan_parquet(str(d / "part-*.parquet"))
                .filter((pl.col("period_ym") >= bounds[0]) & (pl.col("period_ym") < bounds[1]))
                .select("Loan Identifier", "period_ym").collect())
        expected = sorted(zip(disk["Loan Identifier"].to_list(), disk["period_ym"].to_list()))

        seen, order = [], []
        for b in ld.iter_batches(0):
            for loan, p in zip(b["loan"].tolist(), b["period_ym"].tolist()):
                seen.append((loan, p))
                order.append(loan)
        assert sorted(seen) == expected                               # every row exactly once
        assert len(seen) == len(set(seen)) == len(expected)           # no dupes, no drops
        assert all(bounds[0] <= p < bounds[1] for _, p in seen)       # mask honoured
        # shuffled: the yielded loan order differs from the on-disk (filename, row) order.
        disk_order = disk["Loan Identifier"].to_list()
        assert order != disk_order


def test_loader_weights_and_reproducible_and_onehot():
    """weights present and == 1/p_keep on current→current; (seed,epoch) reproducible;
    epoch advance reshuffles; one-hot encode path returns a wider dense block."""
    bounds = (200001, 201312)
    periods = [201201, 201202, 201203, 201204, 201205, 201206]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        expected_w = _synthetic_pool(d, [periods, periods])
        scaler, vocab = _fit_on_window(d, bounds)
        ld = D.WindowLoader(d, bounds, scaler, vocab, batch_size=5, seed=1)

        # weights: every yielded row's weight equals the current→current rule (20 or 1).
        for b in ld.iter_batches(0):
            for loan, w in zip(b["loan"].tolist(), b["w"].tolist()):
                assert abs(w - expected_w[loan]) < 1e-9
        wvals = {expected_w[l] for l in expected_w}
        assert wvals == {1.0, 1.0 / 0.05}                             # both 1.0 and 20.0 present

        # reproducibility: same (seed, epoch) → identical row order; next epoch differs.
        ord0 = [l for b in ld.iter_batches(0) for l in b["loan"].tolist()]
        ord0b = [l for b in ld.iter_batches(0) for l in b["loan"].tolist()]
        ord1 = [l for b in ld.iter_batches(1) for l in b["loan"].tolist()]
        assert ord0 == ord0b and ord0 != ord1

        # one-hot path: cat block widens to the drop-first dummy design (Σ(V_c − 1) cols).
        oh = D.WindowLoader(d, bounds, scaler, vocab, batch_size=64, encode="onehot",
                            shuffle=False)
        b = next(oh.iter_batches(0))
        exp_cols = sum(vocab.vocab_size(c) - 1 for c in vocab.cols)
        assert b["cat"].shape[1] == exp_cols
        assert set(np.unique(b["cat"])).issubset({0.0, 1.0})


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
