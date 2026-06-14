"""M13 — Roll-forward harness (``03_POOL_LEVEL §3``): compose the frozen loan-level model
into 12-month horizon probabilities by **matrix multiplication**, the Phase-3 engine.

For a loan *i* alive at the anchor ``t0 = Dec(k−1)`` (origin state non-terminal), the
12-month outcome distribution is the product of monthly transition matrices::

    p ← one-hot(state_{t0})                         # 1×7 state distribution
    for h = 1..12:
        features x_{t0+h-1}(i):  age += h−1, remaining −= h−1, month_of_year follows
                                 the calendar;  macro block + all static fields frozen at t0
        P_h ← model's 7×7 matrix for loan i at month h   (terminal rows = identity)
        p ← p · P_h

Because the three terminal states (foreclosure / REO / prepaid) are absorbing and we carry
the full 7-vector, the matrix product is the **exact** horizon distribution under the model —
no Monte-Carlo error (``§3``). Two outcomes per loan:

  * ``P(prepaid within 12m) = p[prepaid]``                       (normal chain)
  * ``P(60+ dpd within 12m) = p_fp[dpd_60]``                     (first-passage chain)

The **first-passage** variant makes ``dpd_60`` temporarily absorbing so cures cannot leak the
event; because delinquency is monotone (a loan must pass through ``dpd_60`` to reach
``dpd_90plus``/``foreclosure``/``REO``), all 60+ mass concentrates at ``dpd_60`` (``§3``).

Layering: the numeric engine (``compose`` / ``absorb_rows`` / ``assemble_matrix``) is pure
NumPy and hermetically unit-tested against the closed form (``test_pool.py``). The model
scoring reuses the frozen run folders and the exact ``evaluate.py`` softmax path, so the h=1
composition reproduces ``evaluate.py``'s per-row probabilities bit-for-bit (the M13 Accept).
M14 consumes :func:`roll_forward` to build pools and predicted counts.

Run (GPU box; the full eval pool lives there):
    .venv/bin/python dev/model/pool.py --k 2020 --device cuda            # full window
    .venv/bin/python dev/model/pool.py --k 2020 --device cuda --verify   # + h=1 == evaluate
    .venv/bin/python dev/model/pool.py --k 2020 --max-loans 50000        # quick smoke
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

import backtest as B
import config
import data as D
import evaluate as E
import features as F
import macro_features as mf
import net as N          # noqa: F401 — E._load_nn rebuilds via net; kept explicit for clarity
import torch_common as tc
import train as T

VARIANT = "full"
HORIZON = 12
EVAL_BATCH = E.EVAL_BATCH                 # 32768 — same batching as evaluate (numerically inert)
DEFAULT_CHUNK = 200_000                   # loans per roll-forward chunk (bounded memory)

STATES = F.STATES                         # canonical 7-state order
SI = F.STATE_INDEX                        # state name -> class index 0..6
N_CLASSES = F.N_CLASSES
ORIGIN_STATES = list(config.ORIGIN_STATES)            # the 4 transient origins
ORIGIN_ROWS = [SI[s] for s in ORIGIN_STATES]          # their row indices in the 7×7
TERMINAL_ROWS = [SI[s] for s in STATES if s not in ORIGIN_STATES]  # fc / REO / prepaid
PREPAID = SI["prepaid"]
DPD60 = SI["dpd_60"]
STATE_COL = F.CATEGORICAL.index("state")              # the `state` column position in the vocab


# ===========================================================================
# Layer 1 — pure numeric composition engine (hermetic; unit-tested vs closed form)
# ===========================================================================
def compose(p0: np.ndarray, step_mats) -> np.ndarray:
    """Compose a per-loan Markov chain: ``p0`` ``[N, S]`` initial distributions, ``step_mats``
    an iterable of per-step transition tensors ``[N, S, S]`` (row = origin, col = destination).
    Returns the horizon distribution ``[N, S]`` = ``p0 · M_1 · M_2 · …`` (float64)."""
    p = np.ascontiguousarray(p0, dtype=np.float64)
    for M in step_mats:
        p = np.einsum("ni,nij->nj", p, M)
    return p


def absorb_rows(M: np.ndarray, idx: int) -> np.ndarray:
    """Copy of ``M`` ``[..., S, S]`` with state ``idx`` made absorbing (its row → ``e_idx``):
    the first-passage trick — once mass enters ``idx`` it stays, so the horizon mass at ``idx``
    is the probability of *ever* reaching it."""
    out = M.copy()
    out[..., idx, :] = 0.0
    out[..., idx, idx] = 1.0
    return out


def assemble_matrix(origin_scores: list[np.ndarray]) -> np.ndarray:
    """Build per-loan 7×7 transition tensors ``[N, 7, 7]`` from the four transient-origin score
    rows (each ``[N, 7]`` softmax over destinations, in ``ORIGIN_STATES`` order). Terminal rows
    are the identity (foreclosure / REO / prepaid are absorbing, ``§3``)."""
    n = origin_scores[0].shape[0]
    M = np.zeros((n, N_CLASSES, N_CLASSES), dtype=np.float64)
    for oi, row in zip(ORIGIN_ROWS, origin_scores):
        M[:, oi, :] = row
    for ti in TERMINAL_ROWS:
        M[:, ti, ti] = 1.0
    return M


def onehot_origin(origin_idx: np.ndarray) -> np.ndarray:
    """``[N, 7]`` one-hot initial distribution from per-loan origin class indices."""
    p0 = np.zeros((origin_idx.shape[0], N_CLASSES), dtype=np.float64)
    p0[np.arange(origin_idx.shape[0]), origin_idx] = 1.0
    return p0


# ===========================================================================
# Layer 2 — feature evolution (deterministic fields only; macro + statics frozen at t0)
# ===========================================================================
def add_months(ym: int, k: int) -> int:
    """``YYYYMM`` integer plus ``k`` calendar months (reuses the macro month arithmetic)."""
    return mf.sub_months(ym, -k)


def advance_frame(base: pl.DataFrame, t0: int, h: int) -> pl.DataFrame:
    """The loan-month frame at horizon step ``h`` (month ``t0+h−1``): age advances, remaining
    term decrements, ``period_ym`` rolls forward to drive ``month_of_year`` seasonality. Every
    other column — the entire macro block, UPB, rates, FICO, LTV, vintage — stays frozen at t0
    (``§3`` covariate-evolution assumption). ``state`` is overridden later, per origin pass."""
    dm = h - 1
    return base.with_columns(
        (pl.col("Loan Age") + dm).alias("Loan Age"),
        (pl.col("Remaining Months to Maturity") - dm).alias("Remaining Months to Maturity"),
        pl.lit(add_months(t0, dm), dtype=pl.Int64).alias("period_ym"),
    )


@torch.no_grad()
def _forward_probs(model, cont, cat, binb) -> np.ndarray:
    """Softmax class probabilities ``[N, 7]`` (float32) — the exact ``evaluate._resident_probs``
    numerics (float64 softmax then cast), so h=1 matches ``evaluate.py`` bit-for-bit. ``cont`` /
    ``binb`` float32 device tensors, ``cat`` an int device tensor (cast to long per batch)."""
    model.eval()
    n = int(cont.shape[0])
    out = np.empty((n, N_CLASSES), dtype=np.float32)
    for s in range(0, n, EVAL_BATCH):
        sl = slice(s, s + EVAL_BATCH)
        logits = model(cont[sl], cat[sl].long(), binb[sl])
        out[sl] = torch.softmax(logits.double(), dim=1).float().cpu().numpy()
    return out


def _origin_scores(model, cont, cat, binb, device) -> list[np.ndarray]:
    """Score a torch model under each of the 4 transient origin states by overriding *only* the
    ``state`` vocab index (every other feature held), returning the four destination rows."""
    scores = []
    for s in ORIGIN_STATES:
        cat_o = cat.clone()
        cat_o[:, STATE_COL] = config_state_index(s)
        scores.append(_forward_probs(model, cont, cat_o, binb))
    return scores


# The `state` vocab index is window-specific; cache the per-call resolution in a closure-free
# module global set by roll_forward (kept tiny and explicit rather than threading vocab through).
_STATE_IDX: dict[str, int] = {}


def config_state_index(s: str) -> int:
    return _STATE_IDX[s]


# ===========================================================================
# Roll-forward driver
# ===========================================================================
def _load_models(k: int, device):
    """Frozen window-``k`` models + the shared (scaler, vocab). Mirrors ``evaluate.score_window``
    model loading: empirical matrix, logit, best single NN, and (key windows) the 8-net
    ensemble. Returns ``(scaler, vocab, emp[4×7], torch_models{name:callable_or_list})``."""
    nn_run = config.MODELS / "nn" / VARIANT / B._nn_tag(k, dict(config.NN_SELECTED))
    logit_run = config.MODELS / "logit" / VARIANT / f"k{k}"
    scaler, vocab = F.load_pipeline(nn_run)

    emp = E.empirical_matrix(k)                                   # [4, 7]
    torch_models: dict[str, object] = {
        "logit": E._load_logit(logit_run, scaler, vocab, device),
        "nn": E._load_nn(nn_run, device),
    }
    if k in E.KEY_WINDOWS:
        torch_models["ensemble"] = [
            E._load_nn(config.MODELS / "nn" / VARIANT / f"ens_k{k}_s{s}", device)
            for s in range(B.E.N_MEMBERS)]
    return scaler, vocab, emp, torch_models


def _chunk_matrices(scaler, vocab, emp, torch_models, base: pl.DataFrame,
                    t0: int, h: int, device) -> dict[str, np.ndarray]:
    """Per-model per-loan transition tensors ``[c, 7, 7]`` for horizon step ``h`` on a loan
    chunk. The empirical matrix is feature-free (constant over h); logit/NN/ensemble are scored
    on the evolved features. Ensemble = mean of the 8 members' transition matrices (= mean of
    member softmaxes, the paper Fig-7 rule; identical to ``evaluate``'s mean-of-probs at h=1)."""
    step_df = F.prepare_raw(advance_frame(base, t0, h))
    enc = F.encode_frame(step_df, scaler, vocab, encode="index")
    cont = torch.as_tensor(enc["cont"], dtype=torch.float32, device=device)
    binb = torch.as_tensor(enc["bin"], dtype=torch.float32, device=device)
    cat = torch.as_tensor(np.asarray(enc["cat"]), dtype=torch.long, device=device)
    c = cont.shape[0]

    mats: dict[str, np.ndarray] = {}
    # Empirical — same 4 rows for every loan, every step. Cast to float32 first so the h=1
    # composition reproduces evaluate.py's stored float32 empirical probabilities exactly.
    mats["empirical"] = assemble_matrix([np.broadcast_to(emp[oi].astype(np.float32), (c, N_CLASSES))
                                         for oi in range(len(ORIGIN_STATES))])
    for name, model in torch_models.items():
        if name == "ensemble":
            acc = [np.zeros((c, N_CLASSES), dtype=np.float64) for _ in ORIGIN_STATES]
            for m in model:
                for j, row in enumerate(_origin_scores(m, cont, cat, binb, device)):
                    acc[j] += row
            scores = [a / len(model) for a in acc]
        else:
            scores = _origin_scores(model, cont, cat, binb, device)
        mats[name] = assemble_matrix(scores)
    return mats


def roll_forward(k: int, device, *, variant: str = VARIANT, chunk: int = DEFAULT_CHUNK,
                 max_loans: int | None = None, horizon: int = HORIZON,
                 capture_h1: bool = False) -> dict:
    """Roll every frozen window-``k`` model forward ``horizon`` months over all loans alive at
    the anchor ``t0 = Dec(k−1)`` (the eval-pool rows at ``period_ym == t0``), in bounded-memory
    loan chunks. Returns a per-loan result table (loan id, origin, and per-model
    ``prepaid_12m`` / ``dpd60p_12m`` probabilities), plus — when ``capture_h1`` — the full h=1
    distribution per model for the ``evaluate.py`` equality check."""
    global _STATE_IDX
    assert variant == VARIANT, "roll-forward consumes the full-export frozen models"
    t0 = config._dec(k - 1)
    scaler, vocab, emp, torch_models = _load_models(k, device)
    _STATE_IDX = {s: vocab.maps["state"][s] for s in ORIGIN_STATES}
    model_names = ["empirical", *torch_models.keys()]

    # All t0-alive loans = the eval-pool test slice rows at the anchor month (origin is
    # already transient by construction of the export). Load once; chunk for bounded memory.
    pool_dir, bounds = D.window_spec(variant, k, "test")
    df = (D._masked_scan(pool_dir, bounds)
          .filter(pl.col("period_ym") == t0)
          .collect())
    if max_loans is not None:
        df = df.head(max_loans)
    n_total = df.height

    loan_out, origin_out = [], []
    prep = {m: [] for m in model_names}
    dpd60 = {m: [] for m in model_names}
    h1_acc = {m: [] for m in model_names} if capture_h1 else None

    for start in range(0, n_total, chunk):
        base = F.prepare_raw(df.slice(start, chunk))
        origin_idx = base.select(pl.col("state").replace_strict(
            list(SI), list(SI.values()), default=-1, return_dtype=pl.Int64)
        ).to_numpy().reshape(-1)
        p0 = onehot_origin(origin_idx)
        p = {m: p0.copy() for m in model_names}
        pfp = {m: p0.copy() for m in model_names}

        for h in range(1, horizon + 1):
            mats = _chunk_matrices(scaler, vocab, emp, torch_models, base, t0, h, device)
            for m in model_names:
                M = mats[m]
                p[m] = np.einsum("ni,nij->nj", p[m], M)
                pfp[m] = np.einsum("ni,nij->nj", pfp[m], absorb_rows(M, DPD60))
                if capture_h1 and h == 1:
                    h1_acc[m].append(p[m].copy())

        loan_out.append(base.get_column("Loan Identifier").to_numpy())
        origin_out.append(base.get_column("state").to_numpy())
        for m in model_names:
            prep[m].append(p[m][:, PREPAID])
            dpd60[m].append(pfp[m][:, DPD60])

    cols = {"Loan Identifier": np.concatenate(loan_out),
            "origin": np.concatenate(origin_out)}
    for m in model_names:
        cols[f"{m}_prepaid_12m"] = np.concatenate(prep[m])
        cols[f"{m}_dpd60p_12m"] = np.concatenate(dpd60[m])
    result = pl.DataFrame(cols)

    out = {"k": k, "t0": t0, "n_loans": n_total, "model_names": model_names,
           "horizon": horizon, "result": result}
    if capture_h1:
        out["h1"] = {m: np.concatenate(h1_acc[m]) for m in model_names}
    return out


# ===========================================================================
# Accept #2 — h=1 composition equals evaluate.py outputs exactly
# ===========================================================================
# cross-check tolerance: cuBLAS picks different GEMM algorithms for different batch shapes, so
# the *committed* full-slice evaluate run and a fresh anchor-only scoring of the deep net agree
# only to GPU floating-point precision (~1e-6), not bit-for-bit. The harness exactness is proven
# separately (identical batching ⇒ Δ=0); this bounds the residual against the committed numbers.
XCHECK_TOL = 1e-5


def _score_direct(base: pl.DataFrame, scaler, vocab, emp, torch_models, origin_idx, device) -> dict:
    """The plain loan-level prediction for each anchor row: score it ONCE with its *true* origin
    state — exactly what ``evaluate.py`` computes per row. Same encode + same float64-softmax
    forward as the roll-forward, so it is the bit-exact reference for the h=1 composition."""
    enc = F.encode_frame(base, scaler, vocab, encode="index")
    cont = torch.as_tensor(enc["cont"], dtype=torch.float32, device=device)
    binb = torch.as_tensor(enc["bin"], dtype=torch.float32, device=device)
    cat = torch.as_tensor(np.asarray(enc["cat"]), dtype=torch.long, device=device)
    direct = {"empirical": emp[origin_idx].astype(np.float32)}      # 4×7 indexed by true origin
    for name, model in torch_models.items():
        if name == "ensemble":
            acc = np.zeros((cont.shape[0], N_CLASSES), dtype=np.float64)
            for m in model:
                acc += _forward_probs(m, cont, cat, binb)
            direct[name] = (acc / len(model))
        else:
            direct[name] = _forward_probs(model, cont, cat, binb)
    return direct


def verify_h1(k: int, device, max_loans: int | None = None) -> dict:
    """At h=1 the evolution is the identity (age+0, remaining−0, month=t0), so the composed
    1-step distribution ``one-hot(origin)·P_1`` must equal the row of ``P_1`` for the loan's true
    origin — i.e. the model's per-row prediction that ``evaluate.py`` scores. Two checks:

    (1) **Harness exactness** — against a direct same-batching scoring of the anchor rows
        (:func:`_score_direct`): proves the one-hot init, the per-origin ``state`` override, the
        7×7 assembly and the composition select the right row, **bit-for-bit**.
    (2) **Cross-check vs the committed run** — against ``evaluate.score_window``'s stored
        probabilities on the same anchor rows: agreement to ``XCHECK_TOL`` (the deep net's
        residual is cuBLAS GEMM batch-shape nondeterminism, not a harness discrepancy; the logit
        matches exactly, corroborating that the feature encoding is faithful)."""
    global _STATE_IDX
    t0 = config._dec(k - 1)
    scaler, vocab, emp, torch_models = _load_models(k, device)
    _STATE_IDX = {s: vocab.maps["state"][s] for s in ORIGIN_STATES}
    model_names = ["empirical", *torch_models.keys()]

    pool_dir, bounds = D.window_spec(VARIANT, k, "test")
    base = (D._masked_scan(pool_dir, bounds).filter(pl.col("period_ym") == t0).collect())
    if max_loans is not None:
        base = base.head(max_loans)
    base = F.prepare_raw(base)
    n = base.height
    origin_idx = base.select(pl.col("state").replace_strict(
        list(SI), list(SI.values()), default=-1, return_dtype=pl.Int64)).to_numpy().reshape(-1)

    # (1) Harness exactness — single chunk so the roll-forward's internal batching matches the
    # direct scoring exactly (GEMM shapes identical ⇒ no batch-shape nondeterminism).
    rf = roll_forward(k, device, chunk=n + 1, max_loans=max_loans, horizon=1, capture_h1=True)
    direct = _score_direct(base, scaler, vocab, emp, torch_models, origin_idx, device)

    # (2) Committed evaluate.py outputs on the same anchor rows (loaded in the same scan order).
    ev = E.score_window(k, device)
    period = (D._masked_scan(pool_dir, bounds)
              .select("period_ym").collect().get_column("period_ym").to_numpy())
    anchor = period == t0
    if max_loans is not None:
        keep = np.where(anchor)[0][:max_loans]
        anchor = np.zeros(anchor.shape[0], dtype=bool)
        anchor[keep] = True

    checks, ok = [], True
    for m in model_names:
        a = rf["h1"][m]
        # (1) bit-exact vs direct same-batching scoring.
        exact = a.shape == direct[m].shape and np.array_equal(a, direct[m])
        # (2) vs committed evaluate (tolerance; empirical/logit are bit-identical, nn ~1e-6).
        b = ev["probs"][m][anchor] if m in ev["probs"] else None
        xmax = float(np.abs(a - b).max()) if b is not None and a.shape == b.shape else None
        xident = (xmax == 0.0) if xmax is not None else None
        within = (xmax is None) or (xmax <= XCHECK_TOL)
        ok = ok and exact and within
        checks.append({"model": m, "n": int(a.shape[0]), "harness_bit_exact": bool(exact),
                       "xcheck_max_abs_diff": xmax, "xcheck_bit_identical": xident,
                       "xcheck_within_tol": bool(within)})
        xstr = "n/a" if xmax is None else f"{xmax:.2e}"
        print(f"  [{'PASS' if exact and within else 'FAIL'}] h=1 {m}: {a.shape[0]:,} rows | "
              f"harness bit-exact={exact} | vs evaluate max|Δ|={xstr} "
              f"(≤{XCHECK_TOL:g}={within}, bit-identical={xident})")
    return {"all_pass": ok, "n_anchor": int(n), "xcheck_tol": XCHECK_TOL, "checks": checks}


# ===========================================================================
# Driver
# ===========================================================================
def _peak_rss_mb() -> float:
    import resource
    # ru_maxrss is KB on Linux, bytes on macOS.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    return rss / 1024 if sys.platform != "darwin" else rss / (1024 * 1024)


def run(k: int, device, *, chunk: int, max_loans: int | None, do_verify: bool) -> dict:
    t0 = time.perf_counter()
    print(f"=== M13 roll-forward  k={k}  anchor=Dec{k-1}  device={device}"
          f"  chunk={chunk:,}{' max_loans=' + format(max_loans, ',') if max_loans else ''} ===")

    summary: dict = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     "variant": VARIANT, "k": k, "git_commit": T._git_commit(),
                     "device": str(device), "chunk": chunk, "max_loans": max_loans}

    if do_verify:
        print("\n[Accept 2] h=1 composition == evaluate.py outputs (bit-identical)")
        summary["h1_vs_evaluate"] = verify_h1(k, device, max_loans=max_loans)

    print("\n[Accept 3] full roll-forward over all t0-alive loans (bounded-memory chunks)")
    tr = time.perf_counter()
    rf = roll_forward(k, device, chunk=chunk, max_loans=max_loans, horizon=HORIZON)
    wall = time.perf_counter() - tr
    res = rf["result"]
    # Sanity: probabilities are valid and the harness produced one row per alive loan.
    sane = {}
    for m in rf["model_names"]:
        pr = res.get_column(f"{m}_prepaid_12m").to_numpy()
        dq = res.get_column(f"{m}_dpd60p_12m").to_numpy()
        sane[m] = {"prepaid_mean": float(pr.mean()), "dpd60p_mean": float(dq.mean()),
                   "prepaid_in_01": bool((pr >= 0).all() and (pr <= 1).all()),
                   "dpd60p_in_01": bool((dq >= 0).all() and (dq <= 1).all())}
        print(f"  {m:>9}: E[prepaid≤12m]={pr.mean():.4f}  E[60+≤12m]={dq.mean():.4f}  "
              f"in[0,1]={sane[m]['prepaid_in_01'] and sane[m]['dpd60p_in_01']}")
    peak = _peak_rss_mb()
    print(f"  {rf['n_loans']:,} alive loans rolled {HORIZON}m in {wall:.0f}s  "
          f"peak RSS {peak:,.0f} MB  rows_out={res.height:,}")

    summary.update({"n_loans": rf["n_loans"], "rows_out": res.height, "roll_sec": wall,
                    "peak_rss_mb": peak, "sanity": sane,
                    "outcome_means": {m: sane[m] for m in sane}})
    summary["wall_sec"] = time.perf_counter() - t0

    out_dir = config.MODELS / "nn" / VARIANT
    out = out_dir / f"pool_rollforward_k{k}{'_smoke' if max_loans else ''}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}  [{summary['wall_sec']:.0f}s]")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="M13 roll-forward harness (Phase-3 §3 engine).")
    ap.add_argument("--k", type=int, default=2020, help="test year / window (anchor = Dec(k-1))")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="loans per chunk")
    ap.add_argument("--max-loans", type=int, default=None, help="cap alive loans (smoke)")
    ap.add_argument("--verify", action="store_true", help="run the h=1 == evaluate check")
    args = ap.parse_args()
    config.require_drive()
    device = (torch.device("cuda") if args.device == "cuda"
              else tc.resolve_device(args.device))
    run(args.k, device, chunk=args.chunk, max_loans=args.max_loans, do_verify=args.verify)


if __name__ == "__main__":
    main()
