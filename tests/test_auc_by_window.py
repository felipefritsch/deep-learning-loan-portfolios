"""M25 — per-window AUC driver unit tests (synthetic; no SSD / GPU / model artifacts).

The driver's whole contract is that every AUC it reports is the number ``evaluate.py`` would
produce on the identical frozen rows. These tests pin that down without the SSD:
  * each cell equals a direct ``evaluate._auc_one_vs_rest`` call on the same sliced column;
  * each cell matches ``evaluate.build_auc`` (the pooled M11 AUC machinery) run on a single-window
    pooled dict — the strong "traces to evaluate.py" anchor;
  * the ensemble series exists only on key windows (gaps elsewhere) and is ``None`` otherwise;
  * a degenerate cell (no positives) yields ``None`` rather than crashing.
"""

from __future__ import annotations

import numpy as np

try:  # evaluate.py / auc_by_window.py import torch at module load; skip cleanly when absent
    import torch  # noqa: F401
except ImportError:
    import pytest
    pytest.skip("torch not installed", allow_module_level=True)

from floan.model import auc_by_window as A
from floan.model import evaluate as E


def _probs(rng, n):
    r = rng.random((n, E.F.N_CLASSES)).astype(np.float32)
    return r / r.sum(axis=1, keepdims=True)


def _make_window(rng, n, with_ensemble):
    """A synthetic score_window() result: origins as 0..3 codes (evaluate.OI order), random y,
    row-stochastic float32 prob blocks per model. ``ensemble`` present only when asked."""
    origin = rng.integers(0, len(E.ORIGIN_STATES), size=n).astype(np.int8)
    y = rng.integers(0, E.F.N_CLASSES, size=n).astype(np.int8)
    probs = {"empirical": _probs(rng, n), "logit": _probs(rng, n), "nn": _probs(rng, n)}
    if with_ensemble:
        probs["ensemble"] = _probs(rng, n)
    return {"k": 0, "n": n, "y": y, "origin": origin, "test_key_hash": 12345, "probs": probs}


def test_each_cell_equals_evaluate_auc_fn():
    rng = np.random.default_rng(0)
    pw = {2015: _make_window(rng, 6000, True), 2016: _make_window(rng, 6000, False)}
    table = A.build_auc_by_window(pw)
    for origin, dest in A.TRANSITIONS:
        per_year = table[A._key(origin, dest)]
        for k, w in pw.items():
            mask = w["origin"] == E.OI[origin]
            di = E.SI[dest]
            pos = w["y"][mask].astype(np.int64) == di
            for m in A.MODELS:
                got = per_year[k]["auc"][m]
                if m not in w["probs"]:
                    assert got is None
                    continue
                exp = E._auc_one_vs_rest(w["probs"][m][mask][:, di], pos)
                # identical function, identical inputs -> bit-identical (None passes through)
                assert got == exp or (got is None and exp is None)
            assert per_year[k]["n"] == int(mask.sum())
            assert per_year[k]["n_pos"] == int(pos.sum())


def test_matches_pooled_build_auc():
    """The per-window cells equal evaluate.build_auc run on a single-window pooled dict — i.e. the
    same AUC the M11 suite reports, just sliced per window instead of pooled."""
    rng = np.random.default_rng(3)
    w = _make_window(rng, 9000, True)
    mine = A.build_auc_by_window({2015: w})

    pooled = {"y": w["y"], "origin": w["origin"],
              "probs": {"logit": w["probs"]["logit"], "nn": w["probs"]["nn"]},
              "probs_ens": {"ensemble": w["probs"]["ensemble"]},
              "ens_mask": np.ones(w["n"], dtype=bool)}
    ref = E.build_auc(pooled)
    ref_full = {(r["origin"], r["destination"]): r for r in ref["full_pooled"]}
    ref_key = {(r["origin"], r["destination"]): r for r in ref["key_window_pooled"]}

    for origin, dest in A.TRANSITIONS:
        cell = mine[A._key(origin, dest)][2015]["auc"]
        for m in ("logit", "nn"):
            a, b = cell[m], ref_full[(origin, dest)].get(m)
            assert a == b or (a is None and b is None)
        ae, be = cell["ensemble"], ref_key[(origin, dest)].get("ensemble")
        assert ae == be or (ae is None and be is None)


def test_ensemble_only_on_key_window():
    rng = np.random.default_rng(1)
    pw = {2015: _make_window(rng, 4000, True), 2016: _make_window(rng, 4000, False)}
    table = A.build_auc_by_window(pw)
    cur_prepaid = table[A._key("current", "prepaid")]
    assert cur_prepaid[2015]["auc"]["ensemble"] is not None
    assert cur_prepaid[2016]["auc"]["ensemble"] is None
    # logit / nn are present on both windows
    for k in (2015, 2016):
        assert cur_prepaid[k]["auc"]["logit"] is not None
        assert cur_prepaid[k]["auc"]["nn"] is not None


def test_degenerate_cell_returns_none():
    rng = np.random.default_rng(2)
    w = _make_window(rng, 3000, True)
    w["y"][:] = E.SI["current"]          # no row transitions to prepaid/dpd_30 anywhere
    table = A.build_auc_by_window({2015: w})
    cell = table[A._key("current", "prepaid")][2015]
    assert cell["n_pos"] == 0
    for m in A.MODELS:
        assert cell["auc"][m] is None


def test_structure_covers_all_transitions_and_windows():
    rng = np.random.default_rng(4)
    pw = {y: _make_window(rng, 1500, y in E.KEY_WINDOWS) for y in (2015, 2016, 2017)}
    table = A.build_auc_by_window(pw)
    assert set(table) == {A._key(o, d) for o, d in A.TRANSITIONS}
    for key in table:
        assert set(table[key]) == {2015, 2016, 2017}
        for k in table[key]:
            assert set(table[key][k]["auc"]) == set(A.MODELS)


def test_writers_emit_files_with_note(tmp_path, monkeypatch):
    """Exercise the table + figure writers offline: all four table formats + both figure files,
    with the 'AUC illustrates, NLL decides' note in the markdown (Accept #3)."""
    monkeypatch.setattr(A.config, "OUTPUTS", tmp_path)
    rng = np.random.default_rng(5)
    pw = {y: _make_window(rng, 2000, y in E.KEY_WINDOWS) for y in (2015, 2016)}
    table = A.build_auc_by_window(pw)

    A.write_auc_by_window(table)
    d = tmp_path / "tables" / "loan_level"
    for ext in ("json", "csv", "md", "tex"):
        assert (d / f"auc_by_window.{ext}").exists()
    md = (d / "auc_by_window.md").read_text()
    assert "AUC illustrates, NLL decides" in md
    for origin, dest in A.TRANSITIONS:
        assert A._disp(origin, dest) in md

    figp = A.fig_auc_by_window(table)
    assert figp.with_suffix(".png").exists() and figp.with_suffix(".pdf").exists()
