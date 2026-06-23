"""M23 — calibrator unit tests (synthetic; no SSD / GPU / model artifacts).

Covers the two fitted calibrators (temperature, isotonic) and the engine-seam contract:
  * temperature ``T == 1`` is a genuine bit-identity (the no-op regression check);
  * the fit recovers a known injected miscalibration;
  * temperature preserves ranking/argmax (AUC-neutral, ``06 §4``);
  * isotonic rows stay valid distributions;
  * both calibrators drop into ``pool.roll_forward_predictor`` and yield valid distributions;
  * a fitted-T=1 calibrator through the engine is bit-identical to no calibrator (the M23
    disabled-path regression check, distinct from M21's ``identity_calibrator`` test).
"""

from __future__ import annotations

import numpy as np

from floan.model import calibration as C
from floan.model import pool as P


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _softmax_rows(rng, n, s=None):
    r = rng.random((n, s or P.N_CLASSES))
    return r / r.sum(axis=1, keepdims=True)


class _StubPredictor:
    """A non-torch Predictor returning fixed per-origin rows — exercises the engine seam with no
    model artifacts (mirrors tests/test_pool.py)."""

    def __init__(self, rows):
        self.rows = rows

    def origin_scores(self, frame_h):
        return [r.copy() for r in self.rows]


def _toy_base(n, origin="current", t0=201912):
    import polars as pl
    return pl.DataFrame({
        "Loan Identifier": np.arange(n, dtype=np.int64),
        "state": [origin] * n,
        "Loan Age": np.full(n, 10, dtype=np.int64),
        "Remaining Months to Maturity": np.full(n, 350, dtype=np.int64),
        "period_ym": np.full(n, t0, dtype=np.int64),
    })


# --------------------------------------------------------------------------- temperature
def test_temperature_T1_is_bit_identity():
    """``TemperatureCalibrator(1.0)`` returns the input array unchanged — the short-circuit the
    user required (``softmax(log p / 1)`` is NOT bit-exact)."""
    rng = np.random.default_rng(0)
    p = _softmax_rows(rng, 50)
    cal = C.TemperatureCalibrator(1.0)
    for o in range(len(C.ORIGIN_STATES)):
        out = cal(p, o)
        assert out is p or np.array_equal(out, p)


def test_temperature_recovers_known_miscalibration():
    """If the model is over-confident by a factor ``a`` (``p_model = softmax(a·z)``) while labels
    come from the true ``softmax(z)``, the val-NLL-optimal temperature is ``≈ a`` (it divides the
    sharpened logits back). Fit should recover it within a few %."""
    rng = np.random.default_rng(1)
    n = 200_000
    z = rng.normal(size=(n, P.N_CLASSES)) * 1.5
    p_true = _softmax(z)
    # vectorised inverse-CDF sampling of the true labels
    cdf = np.cumsum(p_true, axis=1)
    u = rng.random(n)
    y = (u[:, None] < cdf).argmax(axis=1)
    a = 1.8
    p_model = _softmax(a * z)
    ft = C.fit_temperature(p_model, y)
    assert abs(ft["T"] - a) / a < 0.05                       # recovers a within 5%
    assert ft["nll_cal"] <= ft["nll_raw"] + 1e-9             # calibration never worsens val NLL


def test_temperature_preserves_argmax_ranking():
    """Temperature scaling is monotone per row, so it preserves the argmax and the within-row
    ranking (the ``06 §4`` AUC-preserving property)."""
    rng = np.random.default_rng(2)
    p = _softmax_rows(rng, 500)
    q = C._temper(p, 2.5)
    assert np.array_equal(p.argmax(1), q.argmax(1))
    assert np.allclose(q.sum(1), 1.0)
    # ranking within each row is preserved
    assert np.array_equal(p.argsort(1), q.argsort(1))


# --------------------------------------------------------------------------- isotonic
def test_isotonic_rows_are_valid_distributions():
    """``IsotonicCalibrator`` output rows sum to 1 and stay in [0, 1]."""
    rng = np.random.default_rng(3)
    n = 5_000
    origin_idx = rng.integers(0, len(C.ORIGIN_STATES), size=n)
    p = _softmax_rows(rng, n)
    # synthetic labels correlated with the predicted argmax so isotonic has signal
    y = np.where(rng.random(n) < 0.7, p.argmax(1), rng.integers(0, P.N_CLASSES, n))
    maps = C.fit_isotonic(p, y, origin_idx)
    cal = C.IsotonicCalibrator(maps)
    for o in range(len(C.ORIGIN_STATES)):
        out = cal(p, o)
        assert out.shape == p.shape
        assert np.all(out >= -1e-9) and np.all(out <= 1 + 1e-9)
        assert np.allclose(out.sum(1), 1.0)


# --------------------------------------------------------------------------- engine seam
def test_fitted_calibrators_flow_through_engine():
    """Both fitted calibrators conform to ``pool.Calibrator`` and drop into
    ``roll_forward_predictor`` (calibrate the RAW per-origin scores before assembly), producing
    valid composed distributions at every snapshot."""
    rng = np.random.default_rng(4)
    n = 16
    rows = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    base = _toy_base(n)
    origin_idx = rng.integers(0, len(C.ORIGIN_STATES), size=n)
    maps = C.fit_isotonic(np.vstack(rows)[:n], rng.integers(0, P.N_CLASSES, n), origin_idx)
    for cal in (C.TemperatureCalibrator(1.7), C.IsotonicCalibrator(maps)):
        out = P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=6,
                                       snapshots=(1, 6), calibrator=cal,
                                       zero_impossible=False, chunk=n + 1)
        for h in (1, 6):
            s = out["snapshots"][h]
            assert s.shape == (n, P.N_CLASSES)
            assert np.all(s >= -1e-9) and np.allclose(s.sum(1), 1.0)


def test_fitted_T1_calibrator_is_bit_identical_through_engine():
    """The M23 no-op regression check: a FITTED ``TemperatureCalibrator(1.0)`` through the engine
    is bit-identical to running with no calibrator (distinct from M21's ``identity_calibrator``
    test — this guards the calibrator class's short-circuit specifically)."""
    rng = np.random.default_rng(5)
    n = 12
    rows = [_softmax_rows(rng, n) for _ in P.ORIGIN_STATES]
    base = _toy_base(n)
    a = P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=6, snapshots=(1, 6),
                                 zero_impossible=False, chunk=n + 1)
    b = P.roll_forward_predictor(_StubPredictor(rows), base, 201912, horizon=6, snapshots=(1, 6),
                                 calibrator=C.TemperatureCalibrator(1.0),
                                 zero_impossible=False, chunk=n + 1)
    for h in (1, 6):
        assert np.array_equal(a["snapshots"][h], b["snapshots"][h])
