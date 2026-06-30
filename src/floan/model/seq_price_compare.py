"""W3b (``ADR-002``) — matched-comparator pricing for the sequence-model exhibit.

:mod:`seq_price_mc` prices the path-dependent sequence models (GRU + transformer) at
``H ∈ {1,3,6,12}`` by Monte-Carlo path simulation, on a seeded **150k char-cell-stratified
subsample** of each anchor pop (ADR-002 loans-per-anchor cap). This module prices the
**first-order / memoryless comparators** — the empirical transition matrix, the logit, the **FF
current-state net** (the key apples-to-apples baseline to the GRU: same point-in-time covariates,
no recurrence), and the 8-net ensemble — at the **same** horizons, on the **identical** subsample,
via the **unchanged composition path**:

    pool.roll_forward_predictor (matrix composition, exact for a first-order Markov chain)
      -> smm_paths.pool_smm_paths -> smm_paths.paths_to_frame -> economics.per_pool_econ

so the GRU/transformer MC prices and the comparator composition prices share one population, one
pool structure (``pools.char_cell_ids`` / ``random_pool_index``), one realized reference, and one
``cashflow_engine`` — the *only* thing that differs is the learner (and, for the sequence models,
simulation vs composition). This is the controlled comparison ADR-001/002 are built around: a
price gap is attributable to the model, not the machinery.

**Why these comparators and not the others.** Composition is exact only for a model whose month-h
distribution is a function of the *point-in-time* covariates at month h (ADR-002 §Context). The
empirical matrix, logit, FF current-state net and the ensemble-of-those all satisfy that, so they
compose to H>1 with no Monte-Carlo error. The **FF-history** arm and the **sequence models**
consume a trajectory, so they are MC-only (priced by :mod:`seq_price_mc`) and are deliberately
absent here. The FF current-state net is the headline comparator — it isolates the value of the
GRU's memory by holding everything else (architecture family, training points, covariates) fixed.

**Identical population — the non-negotiable (ADR-002 / M27b align-gate).** The sequence arm did not
persist its subsample loan-ids, so this module **regenerates** them with the identical seed +
stratification (:func:`seq_price_mc.subsample_pop`, deterministic on ``np.random.default_rng([seed,
k])``), then *proves* the population is byte-identical via two guards: (1) the regenerated pop size
equals the size the sequence arm recorded in ``k{k}_{gru,xf}_h.json``; (2) the per-(scheme,pool)
``n_loans`` are **exactly** equal and the pool ``wac/wam/upb`` and the *realized* valuation
(``cpr_real/wal_real/price_real``) match the committed ``k{k}_gru_econ.parquet`` to ~1e-9 — same
loans ⇒ same pools ⇒ same realized cashflows. The subsample ids + seed/stratification are persisted
and committed for Ch.5 disclosure. A third guard re-checks the H=1 composition identity per
comparator (composed one step == the model's direct one-step prediction), the regression tie-back
to the validated engine.

Run (GPU pod; key set, both schemes, H∈{1,3,6,12} on the matched 150k subsample)::

    python -m floan.model.seq_price_compare --device cuda                  # 5 key anchors
    python -m floan.model.seq_price_compare --device cuda --anchors 2020   # one anchor

Hermetic tests (no SSD/GPU/network) live in ``tests/test_seq_price_compare.py``; they exercise the
SSD-free cores — the composition+economics translation, the H=1 identity guard, the pool-alignment
guard, the subsample determinism — on a synthetic predictor + synthetic econ frames.
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
from floan.model import pool as PL
from floan.model import pools as PP                  # char_cell_ids — stratification key (Ch.5 disclosure)
from floan.model import seq_predict as SQ            # anchor_pop / anchor_base
from floan.model import seq_price_mc as MX           # subsample_pop / MC_OUT (matched population + artifact dir)
from floan.model import smm_paths as SP              # pool_smm_paths / paths_to_frame / _realized_steps
from floan.model import torch_common as tc
from floan.model import train as T                   # _git_commit (provenance)

VARIANT = "full"
ANCHORS = list(EC.ANCHORS)                           # [2015, 2019, 2020, 2023, 2025]
HORIZONS = tuple(config.HORIZONS)                    # (1, 3, 6, 12)
MC_OUT = MX.MC_OUT                                   # src/floan/model/m27b_mc — next to the GRU/xf artifacts

# Composable comparators, headline-first. ``empirical`` is checkpoint-free (panel-derived); ``logit``
# / ``nn`` (the FF current-state net) / ``ensemble`` load frozen window-k checkpoints via the SAME
# loader evaluate/M24 use. ``nn`` is the key baseline (the FF analogue of the sequence GRU).
COMPARATORS = ("empirical", "logit", "nn", "ensemble")
COMPARATOR_LABELS = {"empirical": "Empirical", "logit": "Logit",
                     "nn": "FF current-state", "ensemble": "Ensemble ×8"}
SUBSAMPLE_N = 150_000                                 # the loans-per-anchor cap the sequence arm priced
SEED = 0
ALIGN_TOL = 1e-9                                      # pool wac/wam/upb + realized valuation match tolerance
H1_TOL = 1e-5                                         # composed-H1 vs direct one-step (one chunk ⇒ ~0)


# ===========================================================================
# Comparator predictors — the SAME frozen window-k models M24 prices, behind the ADR-001 seam
# ===========================================================================
def load_predictors(k: int, device, want: tuple[str, ...] = ("logit", "nn", "ensemble")
                    ) -> tuple[dict, dict]:
    """Build the composable comparator predictors for anchor ``k`` via the **unchanged**
    :func:`pool._load_models` + :func:`pool._build_predictors` (the exact loader evaluate/M24 use):
    the checkpoint-free :class:`pool.EmpiricalPredictor` plus a :class:`pool.TorchPredictor` per
    requested torch model (an :class:`pool.EnsemblePredictor` over the 8 members for ``ensemble``).

    Resilient + self-reporting (ADR-002 "report which comparators priced and which checkpoints were
    missing"): the happy path loads the whole set at once; if any checkpoint is absent it falls back
    to loading each model independently so the present ones still price, and returns a ``missing``
    map naming what could not be loaded (and why). Returns ``(predictors, missing)``."""
    try:                                                   # happy path — one load for the full set
        scaler, vocab, emp, tms = PL._load_models(k, device, which=set(want))
        preds = PL._build_predictors(scaler, vocab, emp, tms, device)
        missing = {n: "checkpoint absent / not a key window"
                   for n in want if n not in preds}        # e.g. ensemble off a non-key window
        return preds, missing
    except Exception:                                      # granular fallback — price what loads
        preds, missing = {}, {}
        try:                                               # empirical needs only the panel matrix
            scaler, vocab, emp, _ = PL._load_models(k, device, which=set())
            preds.update(PL._build_predictors(scaler, vocab, emp, {}, device))
        except Exception as e:
            missing["empirical"] = f"{type(e).__name__}: {e}"
        for name in want:
            try:
                s, v, e2, tms = PL._load_models(k, device, which={name})
                built = PL._build_predictors(s, v, e2, tms, device)
                if name in built:
                    preds[name] = built[name]
                else:
                    missing[name] = "checkpoint absent / not a key window"
            except Exception as ex:
                missing[name] = f"{type(ex).__name__}: {ex}"
        return preds, missing


# ===========================================================================
# Composition — one comparator, H∈{1,3,6,12}, on the matched subsample base (UNCHANGED engine)
# ===========================================================================
def compose_comparator(predictor, base: pl.DataFrame, t0: int, *, horizon: int = PL.HORIZON,
                       chunk: int = PL.DEFAULT_CHUNK) -> dict:
    """Roll ONE comparator predictor forward ``horizon`` months over ``base`` via the unchanged
    :func:`pool.roll_forward_predictor` (production ``zero_impossible=True``, ``capture_smm``), and
    return the SMM masses the pricing grid consumes: ``{"cum": [N,horizon], "alive": [N,horizon]}``
    (the exact composition analogues of the MC arm's ``cum_prepaid`` / ``alive_before``). No new
    math — the same roll the M24/M27b composition path runs, just read off at every horizon."""
    rf = PL.roll_forward_predictor(predictor, base, t0, horizon=horizon, snapshots=HORIZONS,
                                   zero_impossible=True, capture_smm=True, chunk=chunk)
    return {"cum": rf["smm"]["cum_prepaid"].astype(np.float64),
            "alive": rf["smm"]["alive_before"].astype(np.float64)}


def h1_identity(predictor, base: pl.DataFrame, t0: int, *, tol: float = H1_TOL) -> dict:
    """Guard: the comparator's H=1 **composition** reproduces its **direct one-step** prediction
    (M13 Accept-#2 / :func:`seq_predict.identity_check` stage (c), per comparator on the matched
    pop). Composes one step raw (``zero_impossible=False``, single chunk ⇒ bit-exact) and compares
    to the predictor's own per-origin scores gathered at each loan's true origin. Returns the
    receipt; ``pass`` is the gate."""
    rf = PL.roll_forward_predictor(predictor, base, t0, horizon=1, snapshots=(1,),
                                   zero_impossible=False, capture_h1=True, chunk=base.height + 1)
    composed = rf["h1"]
    scores = np.stack(predictor.origin_scores(PL.advance_frame(base, t0, 1)))   # [4, N, 7]
    oi = base.select(pl.col("state").replace_strict(
        list(PL.SI), list(PL.SI.values()), default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1)
    direct = scores[oi, np.arange(base.height)]                                 # [N, 7]
    max_abs = float(np.abs(composed - direct).max()) if composed.shape == direct.shape else float("inf")
    return {"max_abs": max_abs, "tol": float(tol), "n": int(base.height),
            "pass": bool(composed.shape == direct.shape and max_abs <= tol)}


# ===========================================================================
# Economic translation — UNCHANGED smm_paths -> economics, fed by the comparator compositions
# ===========================================================================
def price_compare(pop: pl.DataFrame, smm: dict, k: int, realized: tuple, *,
                  horizons: tuple[int, ...] = HORIZONS) -> pl.DataFrame:
    """Pure (SSD-free given ``smm`` + ``realized``): the per-(scheme,pool,model,h) econ frame for a
    set of comparators, via the **UNCHANGED** :func:`smm_paths.pool_smm_paths` ->
    :func:`smm_paths.paths_to_frame` -> :func:`economics.per_pool_econ`. ``smm`` is
    ``{model_name: {"cum": [N,12], "alive": [N,12]}}`` (one entry per priced comparator); ``realized``
    is the ``(alive, prepay, dpd60, dpd90)`` tuple from :func:`smm_paths._realized_steps`. Row order
    is ``pop``'s throughout. Returns the long econ frame (comparator rows + the realized reference's
    error columns), tagged with the ``h`` horizon column."""
    tbl = {"df": pop, "model_names": list(smm),
           "smm": {m: {"cum": smm[m]["cum"], "alive": smm[m]["alive"]} for m in smm},
           "alive_real": realized[0], "prepay_real": realized[1],
           "dpd60_real": realized[2], "dpd90_real": realized[3],
           "upb": pop.get_column("upb").to_numpy(),
           "wac": pop.get_column("wac").to_numpy(),
           "wam": pop.get_column("wam").to_numpy(),
           "pop_ids": pop.get_column("Loan Identifier").to_numpy()}
    paths = SP.pool_smm_paths(tbl, k)
    smm_frame = SP.paths_to_frame(paths, k)
    return EC.per_pool_econ(k, smm_frame, horizons=tuple(horizons))


# ===========================================================================
# Population guards — identical pop vs the sequence arm (byte-identical pools + realized reference)
# ===========================================================================
def _seq_submeta(k: int) -> dict | None:
    """The subsample meta the sequence arm recorded for anchor ``k`` (``target_n / actual_n /
    full_pop_n / seed / stratify``), read from the committed ``k{k}_gru_h.json`` (or the ``xf``
    arm). ``None`` if neither has landed yet."""
    for arm in ("gru", "xf"):
        p = MC_OUT / f"k{k}_{arm}_h.json"
        if p.exists():
            sub = json.loads(p.read_text()).get("subsample")
            if sub:
                return sub
    return None


def assert_population(k: int, pop: pl.DataFrame, *, seed: int, target_n: int) -> dict:
    """Guard 1 — the regenerated subsample matches what the sequence arm priced: same ``seed`` /
    ``target_n``, and the regenerated size equals the sequence arm's recorded ``actual_n``. Raises on
    a mismatch (a divergent population would invalidate the whole controlled comparison)."""
    sub = _seq_submeta(k)
    rec: dict = {"checked": sub is not None, "actual_n": int(pop.height),
                 "n_char_cells": int(np.unique(PP.char_cell_ids(pop)[0]).size)}
    if sub is None:
        rec["reason"] = "no committed sequence subsample meta (k{k}_{gru,xf}_h.json) — cannot cross-check size"
        return rec
    rec.update({"seq_actual_n": int(sub["actual_n"]), "seq_seed": int(sub["seed"]),
                "seq_target_n": int(sub["target_n"]), "seq_stratify": sub.get("stratify")})
    if int(sub["seed"]) != int(seed) or int(sub["target_n"]) != int(target_n):
        raise AssertionError(
            f"population guard FAILED k{k}: seed/target_n differ from the sequence arm "
            f"(seq seed={sub['seed']} target_n={sub['target_n']} vs ours seed={seed} target_n={target_n})")
    if int(sub["actual_n"]) != int(pop.height):
        raise AssertionError(
            f"population guard FAILED k{k}: regenerated subsample size {pop.height:,} != the sequence "
            f"arm's actual_n {sub['actual_n']:,} (seed/stratification drift)")
    rec["pass"] = True
    return rec


def assert_pool_alignment(econ: pl.DataFrame, k: int, *, ref_arm: str = "gru",
                          tol: float = ALIGN_TOL) -> dict:
    """Guard 2 — the matched populations produce **byte-identical pools** and an identical realized
    reference. Compares this run's per-(scheme,pool,h) ``n_loans`` (exact ==), pool ``wac/wam/upb``,
    and the realized valuation ``cpr_real/wal_real/price_real`` against the committed
    ``k{k}_{ref_arm}_econ.parquet``: same loans ⇒ same ``char_cell_ids`` / ``random_pool_index``
    pools ⇒ same realized cashflows. This is the M27b align-gate, now proving the sequence and
    comparator arms priced one and the same population. Raises on any mismatch."""
    ref_path = MC_OUT / f"k{k}_{ref_arm}_econ.parquet"
    if not ref_path.exists():
        return {"checked": False, "reason": f"{ref_path.name} absent — sequence arm not committed yet"}
    keys = ["scheme", "pool", "h"]
    cols = ["n_loans", "wac", "wam", "upb", "cpr_real", "wal_real", "price_real"]
    ref = pl.read_parquet(ref_path).select([*keys, *cols]).unique(subset=keys)
    mine = econ.select([*keys, *cols]).unique(subset=keys)
    if mine.height != ref.height:
        raise AssertionError(
            f"pool alignment FAILED k{k}: {mine.height} (scheme,pool,h) rows vs {ref.height} in "
            f"{ref_path.name} — pool sets differ (population not identical)")
    j = mine.join(ref, on=keys, how="inner", suffix="_ref")
    if j.height != mine.height:
        raise AssertionError(f"pool alignment FAILED k{k}: {mine.height - j.height} pools have no "
                             f"match in {ref_path.name} (population not identical)")
    n_mismatch = int((j.get_column("n_loans") != j.get_column("n_loans_ref")).sum())
    if n_mismatch:
        raise AssertionError(f"pool alignment FAILED k{k}: {n_mismatch} pools differ in n_loans "
                             f"(population not identical)")
    worst = {}
    for c in ("wac", "wam", "upb", "cpr_real", "wal_real", "price_real"):
        d = float((j.get_column(c) - j.get_column(f"{c}_ref")).abs().max())
        worst[c] = d
        if d > tol:
            raise AssertionError(f"pool alignment FAILED k{k}: max|Δ{c}|={d:.3e} > {tol:g} vs "
                                 f"{ref_path.name} (population/realized reference not identical)")
    return {"checked": True, "ref": ref_path.name, "n_pools": int(j.height),
            "n_loans_exact": True, "max_abs": worst, "tol": float(tol), "pass": True}


# ===========================================================================
# Persisted subsample (Ch.5 disclosure) + the matched exhibit table
# ===========================================================================
def write_subsample(k: int, pop: pl.DataFrame, *, seed: int, target_n: int) -> Path:
    """Persist the anchor's subsample loan-id list (sorted, with its char-cell id) + seed /
    stratification — the reproducibility record ADR-002 asks for. Returns the parquet path."""
    cell = PP.char_cell_ids(pop)[0]
    out = pl.DataFrame({"Loan Identifier": pop.get_column("Loan Identifier").to_numpy(),
                        "char_cell": cell.astype(np.int64)}).sort("Loan Identifier")
    p = MC_OUT / f"k{k}_subsample.parquet"
    out.write_parquet(p)
    return p


def build_matched_table(k: int, econ: pl.DataFrame) -> dict:
    """The matched exhibit: the comparators' mean |price error| beside the GRU and transformer, at
    H∈{1,3,6,12} on the char scheme (the headline cross-pool metric). Reads the committed
    ``k{k}_{gru,xf}_econ.parquet`` and concatenates them with this run's comparator econ so every
    column is the same population, same engine. Writes ``k{k}_matched.{md,parquet}`` next to the
    per-arm artifacts; returns the in-memory table. Missing sequence arms simply leave blank columns."""
    parts = [econ.select(["scheme", "k", "model", "h", "price_err"])]
    for arm in ("gru", "xf"):
        p = MC_OUT / f"k{k}_{arm}_econ.parquet"
        if p.exists():
            parts.append(pl.read_parquet(p).select(["scheme", "k", "model", "h", "price_err"]))
    allc = pl.concat(parts, how="vertical_relaxed").filter(pl.col("scheme") == "char")
    mae = (allc.group_by(["model", "h"])
                .agg(pl.col("price_err").abs().mean().alias("mae_price"),
                     pl.col("price_err").mean().alias("bias_price"),
                     pl.len().alias("n_pools"))
                .sort(["model", "h"]))
    MC_OUT.mkdir(parents=True, exist_ok=True)
    mae.write_parquet(MC_OUT / f"k{k}_matched.parquet")

    order = [m for m in (*COMPARATORS, "gru", "xf") if m in mae.get_column("model").unique().to_list()]
    labels = {**COMPARATOR_LABELS, "gru": "GRU (MC)", "xf": "Transformer (MC)"}
    lines = [f"# W3b matched comparison — anchor Dec{k-1} (k={k}), char scheme",
             "",
             "Mean |price error| per 100 face (signed bias in parentheses), by model × horizon, on "
             f"the **identical {SUBSAMPLE_N:,}-loan subsample**. Comparators priced by exact matrix "
             "composition; GRU / transformer by Monte-Carlo path simulation. Same population, pools, "
             "realized reference and cashflow engine — the gap is the learner.",
             "",
             "| H | " + " | ".join(labels.get(m, m) for m in order) + " |",
             "|" + "---|" * (len(order) + 1)]
    for h in HORIZONS:
        cells = []
        for m in order:
            r = mae.filter((pl.col("model") == m) & (pl.col("h") == h))
            cells.append(f"{r['mae_price'][0]:.3f} ({r['bias_price'][0]:+.3f})" if r.height else "—")
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    (MC_OUT / f"k{k}_matched.md").write_text("\n".join(lines) + "\n")
    return {"table_parquet": f"k{k}_matched.parquet", "table_md": f"k{k}_matched.md",
            "models": order, "rows": mae.to_dicts()}


# ===========================================================================
# Driver — per anchor (subsample -> compose comparators -> price -> guard -> persist)
# ===========================================================================
def run_anchor(k: int, device, *, seed: int = SEED, target_n: int = SUBSAMPLE_N,
               comparators: tuple[str, ...] = COMPARATORS, chunk: int = PL.DEFAULT_CHUNK) -> dict:
    """Price every available composable comparator at anchor ``k`` on the matched subsample and
    persist the econ frame + subsample ids + the matched table under :data:`MC_OUT`. Returns the
    payload (priced list, missing checkpoints, the three guards)."""
    t_start = time.perf_counter()
    t0 = config._dec(k - 1)
    full_pop = SQ.anchor_pop(k)
    pop = MX.subsample_pop(full_pop, k, target_n=target_n, seed=seed)     # the EXACT sequence-arm subsample
    pop_guard = assert_population(k, pop, seed=seed, target_n=target_n)
    print(f"  [k{k}] subsample {pop.height:,}/{full_pop.height:,} loans (seed {seed}, "
          f"char-cell stratified) — population guard {'OK' if pop_guard.get('pass') else pop_guard.get('reason','?')}",
          flush=True)

    base = SQ.anchor_base(k, pop)
    if base.height != pop.height:
        raise AssertionError(f"base/pop misalignment k{k}: {base.height} vs {pop.height}")
    realized = SP._realized_steps(k, pop.get_column("Loan Identifier").to_numpy())

    want = tuple(c for c in comparators if c != "empirical")
    preds, missing = load_predictors(k, device, want=want)
    if "empirical" in comparators and "empirical" not in preds:
        missing.setdefault("empirical", "loader returned no empirical stub")
    ordered = [c for c in comparators if c in preds]                      # headline-first, only the present ones

    smm, h1 = {}, {}
    for name in ordered:
        smm[name] = compose_comparator(preds[name], base, t0, chunk=chunk)
        h1[name] = h1_identity(preds[name], base, t0)
        if not h1[name]["pass"]:
            raise AssertionError(
                f"H=1 identity FAILED k{k} {name}: composed-H1 vs direct one-step max|Δ|="
                f"{h1[name]['max_abs']:.3e} > {h1[name]['tol']:g} (composition does not reproduce the "
                f"model's direct prediction)")
        print(f"    [k{k} {name}] composed H∈{list(HORIZONS)}  H=1 identity max|Δ|={h1[name]['max_abs']:.2e}",
              flush=True)
    if not smm:
        raise AssertionError(f"k{k}: no comparator priced (missing={missing}) — nothing to write")

    econ = price_compare(pop, smm, k, realized)
    align = assert_pool_alignment(econ, k)
    MC_OUT.mkdir(parents=True, exist_ok=True)
    econ.write_parquet(MC_OUT / f"k{k}_compare_econ.parquet")
    sub_path = write_subsample(k, pop, seed=seed, target_n=target_n)
    matched = build_matched_table(k, econ)

    payload = {
        "meta": {"k": k, "t0": t0, "n_pop": int(pop.height), "full_pop_n": int(full_pop.height),
                 "seed": int(seed), "target_n": int(target_n),
                 "stratify": "char_cell_ids(FICO×rate×LTV)", "horizons": list(HORIZONS)},
        "priced": ordered, "missing": missing,
        "guards": {"population": pop_guard, "pool_alignment": align,
                   "h1_identity": {m: h1[m] for m in ordered}},
        "subsample_parquet": sub_path.name, "econ_parquet": f"k{k}_compare_econ.parquet",
        "matched_table": matched,
        "git_commit": T._git_commit(),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "wall_sec": round(time.perf_counter() - t_start, 1),
    }
    (MC_OUT / f"k{k}_compare.json").write_text(json.dumps(payload, indent=2, default=float))
    align_str = ("OK" if align.get("pass") else align.get("reason", "?"))
    print(f"  [k{k}] priced {ordered}  missing={list(missing) or 'none'}  pool-alignment={align_str}  "
          f"-> k{k}_compare_econ.parquet + k{k}_matched.md  [{payload['wall_sec']:.0f}s]", flush=True)
    return payload


def run(anchors: list[int], device, *, seed: int = SEED, target_n: int = SUBSAMPLE_N,
        comparators: tuple[str, ...] = COMPARATORS, chunk: int = PL.DEFAULT_CHUNK) -> dict:
    """Price the matched comparators across ``anchors``. One anchor's failure is caught + reported
    (the guards raise loudly so a population/identity break is never silently shipped), so a single
    bad anchor does not sink the others."""
    print(f"=== W3b matched-comparator pricing · anchors={anchors} · comparators={list(comparators)} · "
          f"device={device} · subsample_n={target_n:,} (seed {seed}) ===", flush=True)
    out: dict = {}
    for k in anchors:
        try:
            out[k] = run_anchor(k, device, seed=seed, target_n=target_n,
                                 comparators=comparators, chunk=chunk)
        except Exception as e:
            print(f"!!! W3b-compare k{k} FAILED: {type(e).__name__}: {e}", flush=True)
            out[k] = {"k": k, "error": f"{type(e).__name__}: {e}"}
    return {"anchors": anchors, "results": out}


def main() -> None:
    ap = argparse.ArgumentParser(description="W3b — matched-comparator H∈{1,3,6,12} composition pricing (ADR-002).")
    ap.add_argument("--anchors", type=int, nargs="*", default=None, help="default: 5 key anchors")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--target-n", type=int, default=SUBSAMPLE_N,
                    help=f"loans-per-anchor subsample (must match the sequence arm; default {SUBSAMPLE_N:,})")
    ap.add_argument("--comparators", nargs="+", default=list(COMPARATORS), choices=list(COMPARATORS))
    ap.add_argument("--chunk", type=int, default=PL.DEFAULT_CHUNK)
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda" else tc.resolve_device(args.device))
    anchors = args.anchors if args.anchors else ANCHORS
    run(anchors, device, seed=args.seed, target_n=args.target_n,
        comparators=tuple(args.comparators), chunk=args.chunk)


if __name__ == "__main__":
    main()
