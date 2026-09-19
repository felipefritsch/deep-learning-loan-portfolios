"""W3b (``ADR-002``) — Monte-Carlo H>1 pricing of the sequence models (GRU + transformer).

M27b (:mod:`seq_predict`) priced the sequence GRU at **H=1** — its *weakest* horizon (the
value-of-memory edge is multi-month) — via matrix composition, exact at one step. This module
carries the GRU **and** the transformer to **H ∈ {1,3,6,12}** the only valid way for a
path-dependent model: ``seq_pricing.simulate_paths`` (ADR-002 Monte-Carlo roll-forward) instead
of composition. It is purely **additive** — the cashflow engine, ``compose``/``absorb_rows``,
and the ``smm_paths``/``economics`` aggregation are reused **unchanged**; only the per-loan
state-distribution *generator* differs (simulation vs composition), which is the genuine
modelling difference (ADR-002 Option A — preserve the controlled comparison).

The driver :func:`price_anchor_mc` mirrors :func:`seq_predict.price_anchor_h1` but (a) rolls the
full ``[N, 12]`` SMM path, (b) assembles each loan's **trailing-T observed history** so the MC
window matches M27b's, and (c) is **arch-agnostic** (GRU or transformer, via the
:mod:`seq_predict` architecture seam). Two hard guards gate every anchor:

  * **H=1 identity** (:func:`h1_identity`) — the MC mean at H=1 reproduces the model's direct
    one-step ``evaluate`` prediction (:func:`seq_predict.gru_eval_probs`) to Monte-Carlo
    tolerance ``O(1/√N)``. This is the integration check that the **history assembly is
    byte-right** (history rows < t0 ++ the anchor base row at t0 == M27b's ``build_split``
    trailing-T window).
  * **Aggregation identity** — Σ per-loan dollar value == pool value to ~1e-6 (the ADR-001 §M22
    linear-in-UPB check, now fed by the sequence simulation).

A third, run-once check — :func:`paths_convergence` — sweeps ``n_paths`` at the COVID anchor and
picks the smallest ``N`` whose pool price is stable, reused for the other anchors (ADR-002
paths-convergence requirement; seed the sampler for reproducibility).

The economic translation is **identical** to every other model: ``pool_smm_paths`` →
``paths_to_frame`` → ``economics.per_pool_econ(horizons=(1,3,6,12))`` through the SAME
``cashflow_engine`` and ``WAC−25bp`` curve, so a GRU/transformer-vs-logit price difference is
attributable to the learner, not the engine (the dissertation's controlled comparison).

Run (GPU pod; COVID-first, both arms — retrains the 10M anchor weights if absent, then prices)::

    python -m floan.model.seq_price_mc --device cuda                       # COVID-first key set
    python -m floan.model.seq_price_mc --device cuda --anchors 2020 2023   # subset
    python -m floan.model.seq_price_mc --device cuda --archs gru --n-paths 2000

Hermetic tests (no SSD/GPU/network) live in ``tests/test_seq_price_mc.py`` — they exercise the
pure pieces (:func:`price_simulation`, :func:`h1_identity`) on a synthetic untrained model.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from floan.model import config
from floan.model import economics as EC
from floan.model import export
from floan.model import pool as PL
from floan.model import pools as PP                  # char_cell_ids — stratification key for the subsample
from floan.model import seq_predict as SQ           # anchor_*, train_or_load, gru_eval_probs, arch seam
from floan.model import seq_pricing as MC           # SeqPredictor, simulate_paths, price_paths, _apply_post
from floan.model import sequence as S               # _window_sql / _PANEL_PULL_COLS / _macro_tables / SEQ_LEN
from floan.model import smm_paths as SP             # pool_smm_paths / paths_to_frame / _realized_steps
from floan.model import torch_common as tc
from floan.model import train as T                  # _git_commit (provenance)

VARIANT = "full"
ANCHORS = list(EC.ANCHORS)                          # [2015, 2019, 2020, 2023, 2025] → Dec2014/18/19/22/24
COVID_ANCHOR = 2020                                 # the paths-convergence anchor (largest path dependence)
HORIZONS = tuple(config.HORIZONS)                   # (1, 3, 6, 12)

# Paths-convergence (ADR-002): sweep N, take the smallest where the pool price is stable.
PATHS_GRID = (500, 1000, 2000, 3200)
PRICE_TOL = 0.05                                    # per 100 face — "pool price moves < 0.05"

# H=1 identity Monte-Carlo tolerance FLOORS (the gate is O(1/√N), floored at these so the band
# never collapses below the engine/float residual). A broken history assembly biases EVERY loan,
# so mean|Δ| (not just the max) is the load-bearing statistic — it concentrates at the MC floor
# when the window is byte-right and blows up (~0.1) when it is not.
H1_TOL_MEAN = 1e-2
H1_TOL_MAX = 0.15

MC_OUT = config.REPO_ROOT / "artifacts" / "results" / "m27b_mc"
# The W3b weights are retrained at the 10M sample — a DIFFERENT sample from M27b's 1.5M H=1 weights in
# seq_predict.WEIGHTS_ROOT (m27b_gpu_runs/) — so they land in their own SSD dir, leaving M27b untouched.
MC_WEIGHTS_ROOT = config.OUTPUTS / "m27b_mc"           # 10M GRU/transformer weights (k{k}[_xf]_s{seed}.pt)

# Loans-per-anchor cap (ADR-002 §Mitigations: "simulate on a subsample of loans per anchor … seed
# the sampler for reproducibility"). The per-loan MC roll-forward (seq_pricing.simulate_paths) costs
# ~L·H GPU forwards; the full anchor pops are ~0.6–0.72M loans, so the swept full-pop run is ~2 days.
# We instead price a seeded, FICO×rate×LTV-stratified subsample of this size (stratified on
# pools.char_cell_ids so every characteristic cell — hence every char pool — stays populated). The
# H=1 identity + aggregation guards re-verify on the subsample; the size/seed are recorded in each
# anchor JSON for Ch.5 disclosure. seq_pricing.simulate_paths is left untouched.
# Size chosen within ADR-002's cap: the char pools (pools.char_cell_ids, MIN_CELL=2000) thin with
# the sample, so ~0.6–0.72M-loan anchors retain ~15 of their 39 full-pop char cells at 100k (≈11 at
# 75k, ≈21 at 150k) — 100k balances pool coverage against the ~9–10h overnight budget.
# ``0`` is the "no cap / full pop" sentinel (the vectorized full-population run).
SUBSAMPLE_N = 100_000
# Paths-convergence is a Monte-Carlo-variance property (how many paths stabilise the pool price), not
# a function of pop size, so the COVID N-sweep runs on a seeded stratified subsample of this many
# loans (ADR-002) — picking N cheaply — while pricing still runs on the full pop. ``0`` => sweep on
# the (already-capped) pricing pop itself.
CONV_SUBSAMPLE_N = 20_000


# ===========================================================================
# History assembly — each pop loan's observed trailing rows BEFORE t0 (the MC window prefix)
# ===========================================================================
def anchor_history(k: int, pop: pl.DataFrame, *, T: int = S.SEQ_LEN) -> pl.DataFrame:
    """Each pop loan's **observed** trailing rows for the months strictly **before** ``t0 =
    Dec(k−1)`` — long format (one row per loan-month), macro-reattached, carrying ``Loan
    Identifier`` + ``period_ym`` + the model feature columns, period-ordered per loan. Pulled from
    the SAME panel+macro path :func:`sequence.build_split` uses (``_window_sql`` →
    ``export.attach_macro``), so when :class:`seq_pricing.SeqPredictor` appends the anchor **base**
    row at t0 (step h=1), the reconstructed window is ``build_split``'s trailing-T window
    byte-for-byte — which the H=1 identity guard verifies.

    History-window length (**stated assumption**): we pull the ``T−1`` trailing months before t0.
    With the base row at t0 appended by ``SeqPredictor`` at h=1, the H=1 window is exactly the
    trailing ``T`` months ending at t0 — i.e. M27b's ``build_split`` window
    (``[t_mi−(T−1), t_mi]``). Loans with a shorter panel history yield a shorter (right-padded)
    window, exactly as ``build_split`` does. The frame is NOT ``prepare_raw``-ed here
    (``SeqPredictor.encode_rows`` applies ``prepare_raw`` + Scaler/Vocab), matching the
    ``build_split`` pre-``prepare_raw`` frame so a history row encodes identically to the cached
    training tensors."""
    t0 = config._dec(k - 1)
    pts = SQ.anchor_points(k, pop)                              # point_id/loan/t_ym/t_mi at t0
    shard_lt = config.VARIANTS[VARIANT]["train_shard_lt"]
    con = export.connect()
    con.register("points", pts.select("point_id", "loan", "t_ym", "t_mi").to_arrow())
    win = con.execute(S._window_sql(S._PANEL_PULL_COLS, T, shard_lt)).pl()   # trailing ≤T rows ending at t0
    con.close()
    nat, st, mkt = S._macro_tables()
    pu = (win.filter(pl.col("period_ym") < t0)                 # drop the t0 row (it is the base)
             .unique(subset=["Loan Identifier", "period_ym"])
             .select(S._PANEL_PULL_COLS)
             .with_columns(pl.col("period_ym").alias("label_ym"), pl.lit(1.0).alias("weight")))
    return export.attach_macro(pu, nat, st, mkt).sort(["Loan Identifier", "period_ym"])


def _origin_classes(base: pl.DataFrame) -> np.ndarray:
    """Per-loan origin **class index** (0..6) from the base frame's ``state`` column."""
    return base.select(pl.col("state").replace_strict(
        list(MC.SI), list(MC.SI.values()), default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1)


def subsample_pop(pop: pl.DataFrame, k: int, *, target_n: int = SUBSAMPLE_N, seed: int = 0) -> pl.DataFrame:
    """A seeded, **stratified** subsample of ``anchor_pop(k)`` to ~``target_n`` loans (ADR-002
    loans-per-anchor cap). Strata are the FICO×rate×LTV characteristic cells :func:`pools.char_cell_ids`
    uses to form the ``char`` pools: within each cell we keep ``round(f·cell_n)`` loans (≥1 so no cell
    — hence no char pool — drops out), ``f = target_n / pop.height``, drawn without replacement from a
    generator seeded on ``(seed, k)``. The result is re-sorted by ``Loan Identifier`` to preserve the
    membership/order invariant the downstream ``smm_paths``/``anchor_base``/``anchor_points`` rely on.
    Returns ``pop`` unchanged if it is already ≤ ``target_n``, or if ``target_n <= 0`` (the
    "no cap / full pop" sentinel — used for the vectorized full-population run)."""
    n = pop.height
    if target_n <= 0 or n <= target_n:
        return pop
    cell, _ = PP.char_cell_ids(pop)
    rng = np.random.default_rng([seed, k])
    f = target_n / n
    keep = np.zeros(n, dtype=bool)
    for c in np.unique(cell):
        idx = np.where(cell == c)[0]
        n_keep = min(len(idx), max(1, int(round(len(idx) * f))))
        keep[rng.choice(idx, size=n_keep, replace=False)] = True
    return pop.filter(pl.Series(keep)).sort("Loan Identifier")


# ===========================================================================
# Guard 1 — H=1 identity (history assembly is byte-right)
# ===========================================================================
def h1_identity(sim: dict, direct: np.ndarray, origin_cls: np.ndarray, *,
                calibrator: PL.Calibrator | None = None,
                tol_mean: float | None = None, tol_max: float | None = None) -> dict:
    """The MC mean at H=1 (``sim["h1_dist"]``) must reproduce the model's **direct one-step**
    prediction ``direct`` ``[N, 7]`` (from :func:`seq_predict.gru_eval_probs`) to Monte-Carlo
    tolerance. Both sides get the SAME per-origin post-processing the roll-forward applies *before*
    assembly (the ``calibrator``, then the M20 ``current→fc/REO`` zeroing, via
    :func:`seq_pricing._apply_post`), so the residual is pure MC noise ``O(1/√N)`` — keyed by each
    loan's ``origin_cls``. Tolerances default to ``max(floor, c/√N)`` (the MC band, floored).
    Returns the receipt; ``pass`` is the hard gate."""
    n_paths = int(sim["n_paths"])
    if tol_mean is None:
        tol_mean = max(H1_TOL_MEAN, 0.6 / np.sqrt(max(1, n_paths)))
    if tol_max is None:
        tol_max = max(H1_TOL_MAX, 6.0 / np.sqrt(max(1, n_paths)))
    ref = MC._apply_post(direct, origin_cls, calibrator, zero_impossible=True)
    d = np.abs(sim["h1_dist"] - ref)
    mean_abs, max_abs = float(d.mean()), float(d.max())
    return {"mean_abs": mean_abs, "max_abs": max_abs, "tol_mean": float(tol_mean),
            "tol_max": float(tol_max), "n_paths": n_paths,
            "pass": bool(mean_abs < tol_mean and max_abs < tol_max)}


# ===========================================================================
# Economic translation — UNCHANGED smm_paths → economics, fed by the MC simulation
# ===========================================================================
def _smm_tbl(pop: pl.DataFrame, sim: dict, realized: tuple, model_name: str) -> dict:
    """The ``smm_paths``-shaped ``tbl`` :func:`smm_paths.pool_smm_paths` consumes, from ONE MC
    simulation. ``cum``/``alive`` are the MC cumulative-prepaid / alive-before masses (the exact
    composition-path analogues ``per_loan_smm`` reads). The 60+ first-passage column (T4.2 counts
    only) is **out of scope** for MC pricing — ``simulate_paths`` rolls the prepay chain, and
    ``pool_smm_paths``/``paths_to_frame``/``per_pool_econ`` read only ``cum`` + ``alive`` for the
    price grid. Row order is ``pop``'s, which ``simulate_paths`` preserves (sim rows ⟷ base ⟷ pop)."""
    alive_real, prepay_real, dpd60_real, dpd90_real = realized
    return {"df": pop, "model_names": [model_name],
            "smm": {model_name: {"cum": sim["cum_prepaid"], "alive": sim["alive_before"]}},
            "alive_real": alive_real, "prepay_real": prepay_real,
            "dpd60_real": dpd60_real, "dpd90_real": dpd90_real,
            "upb": pop.get_column("upb").to_numpy(),
            "wac": pop.get_column("wac").to_numpy(),
            "wam": pop.get_column("wam").to_numpy(),
            "pop_ids": pop.get_column("Loan Identifier").to_numpy()}


def price_simulation(sim: dict, base: pl.DataFrame, pop: pl.DataFrame, k: int, realized: tuple, *,
                     model_name: str = "gru", horizons: tuple[int, ...] = HORIZONS,
                     servicing_spread: float = PL.SERVICING_SPREAD) -> dict:
    """Pure (SSD-free given ``realized``): one MC simulation → the per-(scheme,pool) econ frame at
    ``H ∈ horizons`` via the **UNCHANGED** :func:`smm_paths.pool_smm_paths` →
    :func:`smm_paths.paths_to_frame` → :func:`economics.per_pool_econ`, **plus** the ADR-001 §M22
    aggregation-identity check (Σ per-loan dollar value == pool value, ~1e-6) on the same MC SMM
    paths through :func:`seq_pricing.price_paths`. Returns ``{"econ", "smm_frame", "aggregation",
    "pool_price", "pool_wal"}`` (``pool_*`` are the full-12-month whole-population aggregates)."""
    tbl = _smm_tbl(pop, sim, realized, model_name)
    paths = SP.pool_smm_paths(tbl, k)
    smm_frame = SP.paths_to_frame(paths, k)
    econ = EC.per_pool_econ(k, smm_frame, horizons=tuple(horizons))

    res = MC.price_paths(sim, base, servicing_spread=servicing_spread)
    value_sum = float(np.nansum(res["value"]))                 # non-priceable loans carry 0 value
    abs_diff = abs(res["pool_value"] - value_sum)
    agg = {"pool_value": float(res["pool_value"]), "value_sum": value_sum,
           "pool_price": float(res["pool_price"]), "abs_diff": float(abs_diff), "tol": 1e-6,
           "pass": bool(abs_diff <= 1e-6 * max(1.0, abs(res["pool_value"])))}
    return {"econ": econ, "smm_frame": smm_frame, "aggregation": agg,
            "pool_price": float(res["pool_price"]), "pool_wal": float(res["pool_wal"])}


# ===========================================================================
# The MC pricing driver (ADR-002 Decision item 2) — one model, one anchor
# ===========================================================================
def price_anchor_mc(model, scaler, vocab, device, k: int, pop: pl.DataFrame, *,
                    horizon: int = PL.HORIZON, n_paths: int, calibrator: PL.Calibrator | None = None,
                    seed: int = 0, arch: str = "gru",
                    servicing_spread: float = PL.SERVICING_SPREAD) -> dict:
    """Monte-Carlo H>1 pricing of ONE sequence model (``arch`` ``"gru"``/``"xf"``) at anchor ``k``
    on the M24 pricing pop (``seq_predict.anchor_pop``). Builds ``base = anchor_base(k, pop)`` and
    the loan trailing-T :func:`anchor_history`, wraps them in :class:`seq_pricing.SeqPredictor`,
    rolls ``seq_pricing.simulate_paths`` to ``horizon`` with ``n_paths`` trajectories, then prices
    via :func:`price_simulation` (the unchanged ``smm_paths``/``economics`` grid at H∈{1,3,6,12}).

    Hard gates (**raise** on failure, aborting the anchor): the H=1 identity (:func:`h1_identity` —
    the history-assembly integration check) and the §M22 aggregation identity.

    ``calibrator`` defaults to ``None``: per M23 calibration is a no-op for the torch/sequence path
    (T≈1, on-diagonal), so the MC conditions identically to the M24/M27b **raw** composition path.
    Pass a fitted :class:`pool.Calibrator` to enable it — ``simulate_paths`` applies it to each
    month's raw scores before sampling, the same seam point as the composition path (ADR-001 §M23 /
    ADR-002 item 6). Returns ``{"econ", "smm_frame", "meta", "guards"}``."""
    t0 = config._dec(k - 1)
    model_name = "gru" if arch == "gru" else "xf"
    base = SQ.anchor_base(k, pop)
    if base.height != pop.height:
        raise AssertionError(f"base/pop misalignment: {base.height} vs {pop.height}")
    history = anchor_history(k, pop)
    pred = MC.SeqPredictor(model, scaler, vocab, device, history=history)
    sim = MC.simulate_paths(pred, base, t0, horizon=horizon, n_paths=int(n_paths),
                            calibrator=calibrator, seed=seed)

    # ---- Guard 1: H=1 identity (MC H=1 window == build_split window ⇒ direct one-step prediction) ----
    pts = SQ.anchor_points(k, pop)
    direct = SQ.gru_eval_probs([model], k, scaler, vocab, pts, device, arch=arch)
    h1 = h1_identity(sim, direct, _origin_classes(base), calibrator=calibrator)
    if not h1["pass"]:
        raise AssertionError(
            f"H=1 identity FAILED k{k} {arch}: mean|Δ|={h1['mean_abs']:.3e} (tol {h1['tol_mean']:.3e}) "
            f"max|Δ|={h1['max_abs']:.3e} (tol {h1['tol_max']:.3e}) — the MC H=1 window does not match "
            f"the model's direct one-step prediction (history assembly is not byte-right)")

    # ---- Pricing (unchanged smm_paths → economics) + Guard 3: aggregation identity ----
    realized = SP._realized_steps(k, pop.get_column("Loan Identifier").to_numpy())
    priced = price_simulation(sim, base, pop, k, realized, model_name=model_name,
                              servicing_spread=servicing_spread)
    if not priced["aggregation"]["pass"]:
        a = priced["aggregation"]
        raise AssertionError(
            f"aggregation identity FAILED k{k} {arch}: |pool_value − Σ per-loan|={a['abs_diff']:.3e} "
            f"> {a['tol']:g}·max(1,|value|)")

    meta = {"k": k, "t0": t0, "arch": arch, "model_name": model_name, "n_pop": int(pop.height),
            "horizon": int(horizon), "n_paths": int(n_paths), "seed": seed,
            "calibrated": calibrator is not None, "servicing_spread": servicing_spread,
            "pool_price_h12": priced["pool_price"], "pool_wal_h12": priced["pool_wal"],
            "E_cum_prepaid_h12": float(np.nanmean(sim["cum_prepaid"][:, -1])),
            "E_alive_before_h1": float(np.nanmean(sim["alive_before"][:, 0]))}
    return {"econ": priced["econ"], "smm_frame": priced["smm_frame"], "meta": meta,
            "guards": {"h1_identity": h1, "aggregation": priced["aggregation"]}}


# ===========================================================================
# Guard 2 — paths-convergence sweep (run once at the COVID anchor; reuse N)
# ===========================================================================
def paths_convergence(model, scaler, vocab, device, k: int, pop: pl.DataFrame, *, arch: str = "gru",
                      horizon: int = PL.HORIZON, grid=PATHS_GRID, tol: float = PRICE_TOL,
                      calibrator: PL.Calibrator | None = None, seed: int = 0) -> dict:
    """Sweep ``n_paths`` over ``grid`` at anchor ``k`` (run at the COVID anchor, where the path
    dependence — hence the MC variance — is largest) and pick the smallest ``N`` whose pool price
    (per 100 face) moves ``< tol`` from the previous sweep; reuse it for the other anchors (ADR-002
    paths-convergence guard). The seed is held fixed so the sweep isolates path count from sampling
    variance. The convergence scalar is the whole-population pool price
    (:func:`seq_pricing.price_paths`). Falls back to the largest ``grid`` entry if it never settles
    (flagged ``converged=False``)."""
    t0 = config._dec(k - 1)
    base = SQ.anchor_base(k, pop)
    history = anchor_history(k, pop)
    pred = MC.SeqPredictor(model, scaler, vocab, device, history=history)
    prices: list[float] = []
    chosen: int | None = None
    for N in grid:
        sim = MC.simulate_paths(pred, base, t0, horizon=horizon, n_paths=int(N),
                                calibrator=calibrator, seed=seed)
        price = float(MC.price_paths(sim, base)["pool_price"])
        if prices and chosen is None and abs(price - prices[-1]) < tol:
            chosen = int(N)
        delta = abs(price - prices[-1]) if prices else float("nan")
        prices.append(price)
        print(f"  [conv k{k} {arch}] n_paths={N:>5} pool_price={price:.4f} Δ={delta:.4f}"
              f"{'  *converged' if chosen == int(N) else ''}", flush=True)
    converged = chosen is not None
    if chosen is None:
        chosen = int(grid[-1])
    return {"k": k, "arch": arch, "grid": list(grid), "prices": prices, "tol": tol,
            "n_paths": chosen, "converged": converged}


# ===========================================================================
# Driver — per-anchor (train/load both arms → convergence → price → persist)
# ===========================================================================
def _econ_headline(econ: pl.DataFrame, model_name: str) -> dict:
    """Per-H mean |price error| + mean price (char scheme) for the model — the anchor headline
    (the multi-month value-of-memory edge should surface at H≥3, invisible at H=1 in M27b)."""
    char = econ.filter((pl.col("scheme") == "char") & (pl.col("model") == model_name))
    out: dict = {}
    for h in HORIZONS:
        g = char.filter(pl.col("h") == h)
        if g.height:
            out[str(h)] = {"mean_abs_price_err": float(g.get_column("price_err").abs().mean()),
                           "mean_price": float(g.get_column("price").mean()), "n_pools": int(g.height)}
    return out


def run_anchor(k: int, device, *, archs: tuple[str, ...] = SQ.ARCHS, seed: int = 0,
               n_paths: int | None = None, calibrator: PL.Calibrator | None = None,
               horizon: int = PL.HORIZON, retrain: bool = False, conv_grid=PATHS_GRID,
               subsample_n: int = SUBSAMPLE_N, conv_subsample_n: int = CONV_SUBSAMPLE_N) -> dict:
    """Price every ``arch`` at anchor ``k`` and persist per-arm artifacts under :data:`MC_OUT`
    (``k{k}_{arch}_h.json`` summary + ``_econ.parquet`` + ``_smm_paths.parquet``). If ``n_paths`` is
    ``None`` the first arm runs the :func:`paths_convergence` sweep and the chosen ``N`` is reused
    for the remaining arm(s) at this anchor. ``subsample_n`` caps the priced loans-per-anchor
    (ADR-002) via :func:`subsample_pop`; the same subsample feeds the sweep, the pricing, and the
    guards. Returns the chosen ``N`` + per-arm payloads."""
    full_pop = SQ.anchor_pop(k)
    pop = subsample_pop(full_pop, k, target_n=subsample_n, seed=seed)
    sub_meta = {"target_n": int(subsample_n), "actual_n": int(pop.height),
                "full_pop_n": int(full_pop.height), "seed": int(seed),
                "stratify": "char_cell_ids(FICO×rate×LTV)"}
    print(f"  [k{k}] subsample {pop.height:,}/{full_pop.height:,} loans "
          f"(target {subsample_n:,}, seed {seed}, char-cell stratified)", flush=True)
    MC_OUT.mkdir(parents=True, exist_ok=True)
    chosen_n, conv, arms = n_paths, None, {}
    for arch in archs:
        arm_json = MC_OUT / f"k{k}_{arch}_h.json"
        if arm_json.exists() and not retrain:                  # per-arm resume: this arm already priced
            arms[arch] = json.loads(arm_json.read_text())
            if chosen_n is None:                               # recover the anchor's N from the done arm
                chosen_n = int(arms[arch]["meta"]["n_paths"])
            print(f"  [k{k} {arch}] already priced (N={chosen_n}) — skip (resume)", flush=True)
            continue
        t0 = time.perf_counter()
        model, scaler, vocab = SQ.train_or_load(k, seed, device, arch=arch,
                                                weights_root=MC_WEIGHTS_ROOT, retrain=retrain)
        if chosen_n is None:                                   # convergence sweep (once per anchor)
            conv_pop = subsample_pop(pop, k, target_n=conv_subsample_n, seed=seed)   # sweep N on a subsample
            conv = paths_convergence(model, scaler, vocab, device, k, conv_pop, arch=arch,
                                     horizon=horizon, grid=conv_grid, calibrator=calibrator, seed=seed)
            conv["conv_n_loans"] = int(conv_pop.height)        # N picked on this many loans (priced on full pop)
            chosen_n = conv["n_paths"]
            print(f"  [k{k}] convergence N={chosen_n} picked on {conv_pop.height:,} loans "
                  f"(converged={conv['converged']}); pricing on {pop.height:,}", flush=True)
        res = price_anchor_mc(model, scaler, vocab, device, k, pop, horizon=horizon,
                              n_paths=chosen_n, calibrator=calibrator, seed=seed, arch=arch)
        res["econ"].write_parquet(MC_OUT / f"k{k}_{arch}_econ.parquet")
        res["smm_frame"].write_parquet(MC_OUT / f"k{k}_{arch}_smm_paths.parquet")
        payload = {"meta": res["meta"], "guards": res["guards"],
                   "subsample": sub_meta,                       # ADR-002 loans-per-anchor cap (Ch.5 disclosure)
                   "convergence": conv,                        # only the sweeping arm carries it
                   "headline": _econ_headline(res["econ"], res["meta"]["model_name"]),
                   "econ_parquet": f"k{k}_{arch}_econ.parquet",
                   "smm_parquet": f"k{k}_{arch}_smm_paths.parquet",
                   "git_commit": T._git_commit(),
                   "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "wall_sec": round(time.perf_counter() - t0, 1)}
        (MC_OUT / f"k{k}_{arch}_h.json").write_text(json.dumps(payload, indent=2, default=float))
        arms[arch] = payload
        print(f"  [k{k} {arch}] priced N={chosen_n} pool_price(H12)={res['meta']['pool_price_h12']:.4f} "
              f"H=1 mean|Δ|={res['guards']['h1_identity']['mean_abs']:.2e} "
              f"agg|Δ|={res['guards']['aggregation']['abs_diff']:.1e} -> k{k}_{arch}_h.json "
              f"[{payload['wall_sec']:.0f}s]", flush=True)
    return {"k": k, "n_paths": chosen_n, "convergence": conv, "arms": arms}


def run(anchors: list[int], device, *, archs: tuple[str, ...] = SQ.ARCHS, seed: int = 0,
        n_paths: int | None = None, calibrator: PL.Calibrator | None = None,
        horizon: int = PL.HORIZON, retrain: bool = False, subsample_n: int = SUBSAMPLE_N,
        conv_subsample_n: int = CONV_SUBSAMPLE_N) -> dict:
    """Price ``anchors`` **COVID-first** (so the headline anchor + the paths-convergence sweep land
    first), reusing the COVID-chosen ``N`` for the rest. One anchor's failure does not sink the
    others (each is caught + reported)."""
    ordered = ([COVID_ANCHOR] if COVID_ANCHOR in anchors else []) + [a for a in anchors if a != COVID_ANCHOR]
    print(f"=== W3b MC sequence pricing · anchors={ordered} (COVID-first) · archs={list(archs)} · "
          f"device={device}{'  n_paths=' + str(n_paths) if n_paths else '  (convergence sweep)'} · "
          f"subsample_n={subsample_n:,} ===", flush=True)
    chosen_n, out = n_paths, {}
    for k in ordered:
        try:
            res = run_anchor(k, device, archs=archs, seed=seed, n_paths=chosen_n,
                             calibrator=calibrator, horizon=horizon, retrain=retrain,
                             subsample_n=subsample_n, conv_subsample_n=conv_subsample_n)
            chosen_n = res["n_paths"]                          # adopt the COVID-anchor convergence N
            out[k] = res
        except Exception as e:                                 # one anchor must not sink the rest
            print(f"!!! W3b k{k} FAILED: {type(e).__name__}: {e}", flush=True)
            out[k] = {"k": k, "error": f"{type(e).__name__}: {e}"}
    return {"anchors": ordered, "n_paths": chosen_n, "results": out}


def main() -> None:
    ap = argparse.ArgumentParser(description="W3b — Monte-Carlo H>1 sequence-model pricing (ADR-002).")
    ap.add_argument("--anchors", type=int, nargs="*", default=None, help="default: key set (COVID-first)")
    ap.add_argument("--archs", nargs="+", default=list(SQ.ARCHS), choices=list(SQ.ARCHS))
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--n-paths", type=int, default=None, help="fix N (skip the convergence sweep)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retrain", action="store_true", help="force retrain even if weights exist")
    ap.add_argument("--subsample-n", type=int, default=SUBSAMPLE_N,
                    help=f"loans-per-anchor pricing cap; 0 = full pop (default {SUBSAMPLE_N:,})")
    ap.add_argument("--conv-subsample-n", type=int, default=CONV_SUBSAMPLE_N,
                    help=f"loans used for the COVID N-sweep; 0 = use the pricing pop (default {CONV_SUBSAMPLE_N:,})")
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda" else tc.resolve_device(args.device))
    anchors = args.anchors if args.anchors else ANCHORS
    run(anchors, device, archs=tuple(args.archs), seed=args.seed, n_paths=args.n_paths,
        retrain=args.retrain, subsample_n=args.subsample_n, conv_subsample_n=args.conv_subsample_n)


if __name__ == "__main__":
    main()
