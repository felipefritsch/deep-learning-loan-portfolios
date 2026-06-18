# GBT — Tomorrow's Run Guide
### Pod sweep (finish M17) → M18 (exhibits + calibration) → M19 (pool + memo)

You are resuming a paused GBT sweep. **2 of 11 windows are banked** (k2015, k2019), the config is
frozen at `msh=1000`, and the remaining **9 windows must run on a high-RAM box** because the 16 GB
laptop swap-died on the big windows. The pod does *only* the nine fits; M18 and M19 run locally.

**The shape of the day:** push the branch → start the pod → run the 9 fits → sync results back →
tear the pod down → M18 → M19. The COVID window (k2020) runs **first** and is the analytically
important one.

> **Why a pod at all:** feasibility, not speed. k2025 (67 M rows) needs ~35–40 GB resident; on
> 16 GB it swaps to death. A ≥64 GB box runs every window RAM-resident at the fast ~1–2 h pace.

---

## PART 0 — On the Mac, before the pod (~5 min)

The pod clones your **code** from GitHub and reads your **data** from the retained volume. So the
`main` branch (with the `msh=1000` config) **must be pushed first**, or the pod fits the
wrong hyperparameters and your nine new windows won't match the two banked ones.

```bash
# 1. Reconnect the SSD, then from the repo root on the Mac:
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"

# 2. Confirm you're on the right commit and push it
git status                      # expect: on branch main, clean
git log --oneline -1            # expect: 062f361 (or later) — M16+M17+M17_NOTES, msh=1000
git push origin main

# 3. Grab the clone URL (you'll paste it on the pod) and confirm the two banked windows
git remote get-url origin
ls "/Volumes/SSD Felipe/dissertation/models/gbt/full/"   # expect: k2015  k2019
```

✓ **Checkpoint:** branch pushed; clone URL copied; `k2015` and `k2019` present.

---

## PART 1 — Start the pod + attach the volume (RunPod console, ~5 min)

Manual, in the RunPod web console (same flow as your NN runs):

1. **Deploy a new CPU pod** — choose one with **≥ 64 GB RAM** and a decent core count (16–32 cores
   is plenty; GBT is CPU-bound, no GPU needed). A CPU pod is far cheaper than the GPU box.
2. **Attach your retained network volume** (the one from the NN runs, with the export on it). It
   typically mounts at `/workspace`.
3. **Set a spending limit** if RunPod offers one, or just note the hourly rate — the sweep is
   ~6–13 h sequential, so a cheap CPU pod is a few dollars. You'll tear it down today.
4. **Connect → copy the SSH command** (gives you `<IP>`, `<PORT>`, and the key path).

✓ **Checkpoint:** pod running, volume attached, SSH details in hand.

---

## PART 2 — Prime the pod (SSH + bash, ~10 min)

SSH in, then paste the blocks below **in order**. Fill the two placeholders first.

```bash
# --- SSH in (from the Mac terminal) ---
ssh root@<IP> -p <PORT> -i <KEY_PATH>
```

```bash
# === ON THE POD ===
# A) Get the code at the exact committed state (msh=1000)
cd ~
git clone <YOUR_REPO_URL> diss-code
cd ~/diss-code
git checkout main
git log --oneline -1            # ✓ must show 062f361 (or later) — same as the Mac
```

```bash
# B) Find the export on the volume, then point the code's ROOT at it via a symlink
#    (keeps the committed code unmodified — no diff, exact reproducibility)
ls /workspace
EXPORT_DIR=$(find /workspace -maxdepth 5 -type d -name train_pool 2>/dev/null | head -1)
echo "found train_pool at: $EXPORT_DIR"
# Fallback: empty find = the volume has no export yet. Push it up FROM THE MAC (~2.7 GB total —
# 2.0G train_pool + 0.75G eval_pool, i.e. a few minutes, NOT tens of GB), then re-run this block.
# The trailing-slash source copies both train_pool/ and eval_pool/.
if [ -z "$EXPORT_DIR" ]; then
  STAGE_ROOT=/workspace/dissertation     # one var drives BOTH the upload target and the symlink — they can't diverge
  echo "!! no train_pool on the volume — ON THE MAC, rsync the export up first, then re-run this block:"
  echo "   rsync -avz --progress -e \"ssh -p <PORT> -i <KEY>\" \\"
  echo "     \"/Volumes/SSD Felipe/dissertation/processed/training/full/\" \\"
  echo "     \"root@<IP>:$STAGE_ROOT/processed/training/full/\""
  echo "   then re-run this block so the find resolves EXPORT_ROOT to $STAGE_ROOT (or set EXPORT_ROOT=$STAGE_ROOT by hand)."
  # STOP until the upload finishes. EXPORT_ROOT must equal $STAGE_ROOT for the symlink step below.
fi
# EXPORT_ROOT = the folder ABOVE processed/  (e.g. /workspace/dissertation)
EXPORT_ROOT=$(echo "$EXPORT_DIR" | sed 's#/processed/.*##')
echo "EXPORT_ROOT = $EXPORT_ROOT"      # <-- note this; you'll reuse it in PART 4
# also confirm the eval pool exists (needed for scoring)
ls "$EXPORT_ROOT"/processed/training/full/   # ✓ expect: train_pool  eval_pool

# point the code's expected ROOT at the volume (require_drive() will then pass)
mkdir -p "/Volumes/SSD Felipe"
ln -sfn "$EXPORT_ROOT" "/Volumes/SSD Felipe/dissertation"
```

```bash
# C) Python env + dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
pip install "lightgbm==4.6.0" polars pyarrow duckdb numpy
pip install -e .                # installs the floan package + its pyproject.toml-pinned deps, so `python -m floan.…` resolves
```

```bash
# D) Verify the frozen config BEFORE spending any compute
grep -A3 -i "GBT_SELECTED" src/floan/model/config.py    # ✓ must show min_sum_hessian... = 1000
```

```bash
# E) SMOKE TEST — tiny fit, validates env + data + config without burning real time
#    (standing-rule-3 --smoke path, <=100k rows)
.venv/bin/python -m floan.model.gbt --mode frozen --variant full --k 2020 --max-rows 100000

# smoke passed — delete the capped-fit folder so PART 4's count stays 11 and it isn't rsynced back
rm -rf "/Volumes/SSD Felipe/dissertation/models/gbt/full/k2020_smoke"
```

✓ **Checkpoint:** clone at the right commit; `train_pool`+`eval_pool` found; `GBT_SELECTED` shows
`msh=1000`; smoke test runs clean and exits.

> **If the smoke test errors on a `torch` import** (backtest.py may import it at module top even
> under `--gbt-only`): `pip install torch --index-url https://download.pytorch.org/whl/cpu` and
> re-run the smoke test.

---

## PART 3 — Launch the 9-window sweep (bash, then wait ~6–13 h)

Sequential, regime windows first, RAM-resident (no swap on a 64 GB box). The COVID window lands
first. Set `THREADS` to roughly the pod's core count.

```bash
# === ON THE POD ===
cd ~/diss-code && source .venv/bin/activate
mkdir -p "/Volumes/SSD Felipe/dissertation/logs"

THREADS=16    # ≈ the pod's core count

nohup .venv/bin/python -u -m floan.model.backtest --device cpu --gbt-only \
  --windows 2020 2023 2025 2016 2017 2018 2021 2022 2024 --gbt-threads $THREADS \
  >> "/Volumes/SSD Felipe/dissertation/logs/gbt_sweep.log" 2>&1 &

echo "PID = $!"
```

Monitor live (re-attachable; the `nohup` job survives SSH drops):

```bash
tail -f "/Volumes/SSD Felipe/dissertation/logs/gbt_sweep.log"
# Ctrl-C just stops watching, not the job.

# Check completed windows at any time:
ls "/Volumes/SSD Felipe/dissertation/models/gbt/full/"
```

✓ **Checkpoint (first one, ~1–2 h in):** `k2020/metrics.json` appears — that's the COVID window,
your key result. You can start reading it before the rest finish.

> **Idempotent + resumable:** if the pod hiccups, just re-run the exact launch command — completed
> windows are skipped. The atomic writes mean no half-written folder is ever mistaken for done.

> **Optional speed-up (only if you want it):** windows run one-at-a-time here. A 64 GB box could run
> ~4 concurrently — but that needs `backtest.py` confirmed safe to run as parallel processes (shared
> summary file / DuckDB connections). Don't hand-roll it; if you want the ~3× speed, ask Claude Code
> first: *"Is backtest.py safe to run as N concurrent single-window processes, or does it share
> mutable state (summary file, caches, DB connections)? If safe, give me a bounded-concurrency
> launcher; if not, leave it sequential."* Sequential finishes well before Friday regardless.

---

## PART 4 — Sync results back, then tear down (on the Mac, ~5 min)

**Do not tear down before syncing** — never leave the only copy of the run folders on a rented box.
Run these **on the Mac**, with the SSD connected. Use the `EXPORT_ROOT` you noted in PART 2-B.

```bash
# === ON THE MAC ===
PORT=<PORT>; IP=<IP>; KEY=<KEY_PATH>
REMOTE_ROOT=<EXPORT_ROOT>          # e.g. /workspace/dissertation  (from PART 2-B)
SSD="/Volumes/SSD Felipe/dissertation"

# Pull the new GBT run folders from the pod into the SSD
rsync -avz --progress -e "ssh -p $PORT -i $KEY" \
  "root@$IP:$REMOTE_ROOT/models/gbt/full/" \
  "$SSD/models/gbt/full/"

# Verify all 11 windows are now present locally
ls "$SSD/models/gbt/full/"
# ✓ expect 11: k2015 k2016 k2017 k2018 k2019 k2020 k2021 k2022 k2023 k2024 k2025
ls "$SSD/models/gbt/full/" | wc -l    # ✓ expect 11
```

```bash
# Mirror to the second copy + commit the run records, THEN tear down
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
scripts/backup_ssd.sh
git add -A && git commit -m "M17: 9 remaining GBT windows fitted on pod (msh=1000), all 11 banked"
git push origin main
```

✓ **Checkpoint:** 11 windows on the SSD; mirrored; committed + pushed.
**Now tear down the pod** in the RunPod console (Stop/Terminate) — the meter stops.

---

## PART 5 — Finish locally with Claude Code: M18, then M19

Everything below runs on the Mac (local CPU), one task per turn, stop at each gate — same discipline
as M16/M17. Paste each prompt into Claude Code in order.

### → M18 — GBT into the loan-level exhibits + calibration decision

```
All 11 GBT windows are now banked on the SSD (k2015/k2019 from the Mac, the other 9 from the pod,
all at msh=1000 — deterministic, so thread count doesn't change them). Proceed to M18.

Work on `main` — all GBT work stays on this branch. Do not create a new branch; commit directly to `main`.

1. Fold GBT in as the fifth model column on the identical frozen test rows: Table A (tuning window),
   Table B (rolling + pooled), NLL-by-origin-state, and the one-vs-rest AUC table. Assert row-count +
   hash identity so GBT is scored on exactly the same rows as logit/NN/ensemble; each pooled test row
   counted once. (The NLL/AUC math in evaluate.py is model-agnostic; score_window + MODEL_ORDER/
   MODEL_LABELS need a gbt entry — that's expected, surface it, don't code around it.)

2. Read me three things before anything else:
   (a) the k2020 COVID window result — GBT vs logit vs NN — it's the regime where the inductive
       biases are most likely to diverge and it carries the Ch.5 story;
   (b) the pooled all-windows GBT NLL vs logit and NN, and the §7 read (tie / net-edges-GBT / GBT-wins);
   (c) the impossible-cell-mass QA pattern across all 11 windows against best_iter — confirm whether
       ONLY the low-best_iter (early-stopped) windows breach 1e-3 and the long-fit ones stay clean
       (the fit-maturity hypothesis from M17_NOTES). A long-fit window breaching is a different signal
       — flag it.

3. Calibration (conditional): run the existing reliability diagrams on raw GBT softmax. If on-diagonal,
   document and do nothing. If off-diagonal beyond the nets' level, fit per-window temperature scaling
   on the val slice (per-class isotonic + renormalise as fallback), persist the transform to each run
   folder, and re-check — keep it only where it improves val ECE without worsening test NLL.

Accept: GBT column in Table A/B (+pooled), NLL-by-origin, and AUC on asserted-identical frozen rows;
calibration decision documented (applied-with-evidence or skipped-with-evidence); memo 2b / chapter-4
tables regenerate. Stop and show me the three reads + the calibration decision.
```

### → M19 — GBT into the pool roll-forward + economic translation + memo (the GBT-baseline gate)

```
M18 looks right. Proceed to M19 — the final GBT task.

1. Add ONE model-agnostic predictor seam to pool.py: a callable returning the per-loan 7-vector at the
   evolved covariates, so the composition (03 §3) and cashflow (03 §5) engines score the net OR the GBT
   booster with the engine logic otherwise unchanged. (The current path is torch-specific; the seam is
   real new code — the minimal refactor, nothing else in the engine touched.) If M18 fitted a GBT
   calibrator, apply it inside this predictor — pool valuation consumes levels.

2. Produce GBT columns in T4.2 (count R²/RMSE), T5.1 (CPR/WAL/price errors) and F5.2 (price-error
   buckets) at the regime anchors, through the same cashflow engine and fixed discount curve as the
   other models.

3. Update writeup/memos/03_pool with the GBT row in the relevant tables, the CPU/GPU deviation note,
   and the §7 pre-registered framing (flexibility-wins vs smooth-target-rewards-the-net), read against
   the actual full-sweep numbers.

Accept (GBT-baseline gate): h=1 GBT composition equals the GBT loan-level prediction on the same anchor
rows (the M13 Accept #2 analogue); engine sign-coherence (price-error sign vs WAL-error sign) holds for
GBT; GBT columns in T4.2/T5.1/F5.2 at ≥3 regime-spanning anchors via the UNCHANGED engine; the cashflow
engine's closed-form unit tests still pass; memo updated. Stop and show me the gate + the headline
GBT-vs-net price-error comparison.
```

### Close out

```bash
# === ON THE MAC, after the M19 gate passes ===
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
scripts/backup_ssd.sh
git add -A && git commit -m "M19: GBT pool-level translation + memo — GBT-baseline gate"
git push origin main
```

Push to `main`; refresh the brief's GBT column from the full sweep.

---

## One-screen checklist

| # | Where | Do | ✓ when |
|---|-------|-----|--------|
| 0 | Mac | push `main`; copy clone URL | branch on GitHub; k2015/k2019 present |
| 1 | RunPod | deploy ≥64 GB CPU pod + attach volume | SSH details in hand |
| 2 | Pod | clone+checkout, symlink ROOT, env, **smoke** | smoke clean; `GBT_SELECTED`=msh=1000 |
| 3 | Pod | launch 9-window sweep (regime-first) | k2020 lands ~1–2 h in |
| 4 | Mac | rsync back, verify 11, mirror, **tear down** | 11 windows on SSD; pod stopped |
| 5 | Mac/Claude Code | M18 (exhibits+calibration) → M19 (pool+memo) | GBT-baseline gate passes |

## Things that bite if skipped
- **Didn't push `main` in PART 0** → pod fits the old `msh=100` and the 9 windows are
  inconsistent with the 2 banked ones. (PART 2-D's `GBT_SELECTED` check is the catch.)
- **Tore down the pod before rsync** → the 9 run folders are gone. Sync, verify 11, *then* terminate.
- **Skipped the smoke test** → you discover a broken env/path after an hour of compute, not before.
- **Read the pooled number as the verdict** → it's the regime windows (esp. k2020) that carry the
  story; read those, not just the average.
