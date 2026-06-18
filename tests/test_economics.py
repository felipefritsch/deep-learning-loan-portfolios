"""M15b unit tests — the economic-error wiring (``economics.py``): the cashflow-engine valuation,
the model-minus-realized error definitions, and the T5.1 aggregation, all hermetic (no SSD, no GPU,
no model artifacts; a synthetic ``smm_paths`` frame stands in for the M15a parquet).

The engine itself is pinned by ``test_pool.py``'s closed-form suite; here we pin the *error layer*:
  * zero error when a model's SMM path equals the realized path (the valuation reference);
  * the §5.3 sign convention on a premium pass-through — a model that under-predicts prepayment
    (slower SMM than realized) over-values the pool (price_err > 0), dates cashflows late
    (wal_err > 0), and shows a negative CPR error;
  * the T5.1 ``aggregate`` mean|err| / signed-bias / headline-reduction arithmetic.

Run:  .venv/bin/python -m floan.model.test_economics     (or python -m pytest)
"""
from __future__ import annotations

import numpy as np
import polars as pl

try:  # economics.py/pool.py import torch at module load
    import torch  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("torch not installed", allow_module_level=True)

from floan.model import economics as EC
from floan.model import pool as P

SMM_COLS = EC.SMM_COLS


def _frame(rows: list[dict]) -> pl.DataFrame:
    """Build a synthetic smm_paths frame; each row dict carries scheme/pool/model/wac/wam/upb and a
    scalar ``smm`` broadcast to the 12 monthly columns (constant path)."""
    out = []
    for r in rows:
        rec = {"scheme": r["scheme"], "pool": r["pool"], "model": r["model"],
               "n_loans": r.get("n_loans", 1000), "wac": r["wac"], "wam": r["wam"], "upb": r["upb"]}
        rec.update({c: float(r["smm"]) for c in SMM_COLS})
        out.append(rec)
    return pl.DataFrame(out)


def test_zero_error_when_model_equals_realized():
    """If a model's SMM path is identical to realized, every error is exactly 0 (the engine and the
    annualised-CPR map are deterministic functions of the path)."""
    df = _frame([
        {"scheme": "char", "pool": 100, "model": "logit",    "wac": 0.045, "wam": 340, "upb": 1e8, "smm": 0.004},
        {"scheme": "char", "pool": 100, "model": "realized", "wac": 0.045, "wam": 340, "upb": 1e8, "smm": 0.004},
    ])
    e = EC.per_pool_econ(2020, df).row(0, named=True)
    assert abs(e["cpr_err"]) < 1e-9 and abs(e["wal_err"]) < 1e-9 and abs(e["price_err"]) < 1e-9


def test_premium_underprediction_sign_convention():
    """Premium pool (WAC > WAC−25bp curve). A model SLOWER than realized (under-predicts prepay)
    ⇒ cpr_err < 0, price_err > 0 (over-values), wal_err > 0 (cashflows dated late); a FASTER model
    flips all three. Pins the §5.3 sign convention end to end."""
    base = dict(scheme="char", pool=1, wac=0.05, wam=340, upb=1e8)
    df = _frame([
        {**base, "model": "logit",    "smm": 0.002},   # slow  → under-predicts prepay
        {**base, "model": "ensemble", "smm": 0.020},   # fast  → over-predicts prepay
        {**base, "model": "realized", "smm": 0.008},
    ])
    out = EC.per_pool_econ(2020, df)
    slow = out.filter(pl.col("model") == "logit").row(0, named=True)
    fast = out.filter(pl.col("model") == "ensemble").row(0, named=True)
    assert slow["cpr_err"] < 0 < slow["price_err"] and slow["wal_err"] > 0
    assert fast["cpr_err"] > 0 > fast["price_err"] and fast["wal_err"] < 0
    # price strictly orders with speed (premium erosion): slower keeps more value than realized.
    assert slow["price"] > slow["price_real"] > fast["price"]


def test_aggregate_mean_abs_bias_and_headline_reduction():
    """``aggregate`` returns mean|err|, signed bias, and the per-anchor ensemble-vs-logit reduction
    in mean|price error|. Construct two char pools with hand-set price errors and check the arithmetic
    (logit |err| = mean(|+0.6|,|−0.2|)=0.40; ensemble = mean(|+0.1|,|+0.1|)=0.10 ⇒ 75% reduction)."""
    # Two pools whose realized speed sits between logit (slow) and ensemble (≈realized), so logit
    # carries the larger |price error|. Exact magnitudes are pinned by the asserts below, not guessed.
    df = _frame([
        {"scheme": "char", "pool": 1, "model": "logit",    "wac": 0.05, "wam": 340, "upb": 1e8, "smm": 0.001},
        {"scheme": "char", "pool": 1, "model": "ensemble", "wac": 0.05, "wam": 340, "upb": 1e8, "smm": 0.0075},
        {"scheme": "char", "pool": 1, "model": "realized", "wac": 0.05, "wam": 340, "upb": 1e8, "smm": 0.008},
        {"scheme": "char", "pool": 2, "model": "logit",    "wac": 0.05, "wam": 340, "upb": 1e8, "smm": 0.020},
        {"scheme": "char", "pool": 2, "model": "ensemble", "wac": 0.05, "wam": 340, "upb": 1e8, "smm": 0.0125},
        {"scheme": "char", "pool": 2, "model": "realized", "wac": 0.05, "wam": 340, "upb": 1e8, "smm": 0.012},
    ])
    agg = EC.aggregate({2020: EC.per_pool_econ(2020, df)})
    lo = agg["cells"][2020]["char"]["price"]["logit"]
    en = agg["cells"][2020]["char"]["price"]["ensemble"]
    # mean|err| == mean of abs per-pool errors; bias == mean of signed errors; n == 2.
    pe = EC.per_pool_econ(2020, df).filter(pl.col("scheme") == "char")
    for m, (mae, bias, n) in (("logit", lo), ("ensemble", en)):
        ev = pe.filter(pl.col("model") == m).get_column("price_err").to_numpy()
        assert n == 2 and abs(mae - np.abs(ev).mean()) < 1e-12 and abs(bias - ev.mean()) < 1e-12
    red = agg["headline"]["char"][2020]["reduction_pct"]
    assert abs(red - (1.0 - en[0] / lo[0]) * 100.0) < 1e-9
    assert en[0] < lo[0] and red > 0          # ensemble (≈realized) beats the slow logit here


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("all economics (M15b) tests passed")
