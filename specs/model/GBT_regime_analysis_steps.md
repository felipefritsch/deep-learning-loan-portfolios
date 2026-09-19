# GBT — Regime-First Analysis Steps
### Read the 5 regime windows early (for Friday), finish the rest after

**The idea:** the 5 regime windows (2015, 2019, 2020, 2023, 2025) *are* the story — calm,
late-cycle, COVID, rate-shock, latest. Two (2015, 2019) are already on your Mac; three (2020,
2023, 2025) are wave 1 on the pod. The moment wave 1 finishes you have the complete regime spine,
so you can pull just those three to the Mac and run the analysis **tonight**, without waiting for
the six fillers or the pooled-all-11 statistic. The fillers + pooled NLL + pool translation come
after.

**Where things live (so the comparison works):** GBT regime results need to sit on the Mac SSD
**next to** the logit / NN / empirical results (`models/{logit,nn,benchmarks}/full/`), because the
analysis compares GBT vs those *within each window*. That's why Step 2 is a partial rsync — it
brings the three pod windows home so all five can be read against the baselines locally.

| Step | When | Where |
|------|------|-------|
| 1 | wave 1 (k2020/k2023/k2025) all banked | pod — confirm |
| 2 | immediately after | **Mac terminal** — partial rsync |
| 3 | after rsync verifies 5 | **Codex on Mac** (fresh chat) — regime read |
| 4 | all 9 pod windows banked (later tonight) | pod — verify-only |
| 5 | after step 4 | **Mac terminal** — full rsync + mirror + **tear down pod** |
| 6 | after full sync | **Codex on Mac** — full M18 + M19 |

---

## STEP 1 — Confirm wave 1 is complete  *(pod, informational)*

Codex's event watcher fires on "all wave-1 banked." To check yourself, in the **pod terminal**:

```bash
ls /workspace/dissertation/models/gbt/full/   # expect: k2020  k2023  k2025  (+ any fillers started)
for k in k2020 k2023 k2025; do
  [ -f "/workspace/dissertation/models/gbt/full/$k/metrics.json" ] && echo "$k ✓ banked" || echo "$k … training"
done
```

✓ **Proceed when all three show `✓ banked`.** (k2023 is already done; you're waiting on k2020, k2025.)

---

## STEP 2 — Pull the 3 regime windows to the Mac  *(Mac terminal)*

**First reconnect + mount the SSD** if you ejected it. Then this pulls **only** k2020/k2023/k2025
(the fillers keep running on the pod, untouched):

```bash
# === ON THE MAC ===
rsync -avz -e "ssh -p 13761 -i ~/.ssh/id_ed25519" \
  --include='/k2020/***' --include='/k2023/***' --include='/k2025/***' \
  --exclude='*' \
  "root@213.173.111.104:/workspace/dissertation/models/gbt/full/" \
  "/Volumes/SSD Felipe/dissertation/models/gbt/full/"

# verify all 5 regime windows are now on the Mac
ls "/Volumes/SSD Felipe/dissertation/models/gbt/full/"   # expect at least: k2015 k2019 k2020 k2023 k2025
```

✓ **Proceed when k2015, k2019, k2020, k2023, k2025 all show.** (This rsync is accumulate-only and
safe to re-run; pulling the full set later in Step 5 re-pulls these identically — no conflict.)

> Don't tear down the pod here — the fillers are still running on it.

---

## STEP 3 — Regime-only read  *(Codex on the Mac — open a FRESH chat in the repo)*

Paste this. It's M18 restricted to the 5 regime windows: the per-window GBT-vs-net-vs-logit
comparison, the COVID headline, the §7 lean, and the fit-maturity check — enough to write the
Friday draft. It does **not** compute the pooled-all-11 NLL or touch the pool engine.

```
The 5 REGIME GBT windows are now on the SSD: k2015 + k2019 (fitted locally) and k2020 + k2023 +
k2025 (just rsync'd from the pod) — all at frozen msh=1000, deterministic. The 6 fillers are still
running on the pod, so this is a REGIME-ONLY read, not the full pooled M18. Do NOT compute the
pooled-all-11 NLL (needs every window) and do NOT touch the pool/cashflow engine (that's M19).
Work on main, no new branch.

Read GBT from models/gbt/full/k{2015,2019,2020,2023,2025}/ and compare against the existing logit
(models/logit/full/k*), best single NN + ensemble (models/nn/full/k*), and empirical/bucketed floor
(models/benchmarks/full/) for the SAME windows. Reuse evaluate.py's NLL/AUC machinery; assert
row-count + hash identity so GBT is scored on exactly the frozen test rows the other models used.
If a baseline is missing for a window, note it and proceed.

Produce, and STOP for me to read:

1. Per-window comparison table (5 regime windows): test NLL for empirical / logit / best single NN /
   ensemble / GBT + each window's empirical floor. NLL is NOT comparable across windows (each test
   year is a different difficulty — k2023's floor ~0.079 vs k2015's ~0.113). Compare models only
   WITHIN a window. Give GBT−logit, GBT−single-NN, and GBT−ensemble gaps per window.

2. The COVID read (k2020). Known result to verify and frame: GBT 0.18412 beats logit (0.19195) by
   −0.00783 and beats the single net (0.18487) by −0.00075; the 8-net ensemble (0.18341) edges GBT
   by +0.00071. Set the GBT−single-NN gaps against the calm references (k2015 ≈ +0.00047, k2019 ≈
   +0.00290, both net-ahead). State explicitly that k2020 INVERTS this — GBT beats the single net
   and only the ensemble noses ahead — which CONTRADICTS the "smooth bias helps most under
   distribution shift" prior: on the regime-shift window GBT is most competitive, not least. Give
   the economic reading as a hypothesis (COVID signal dominated by sharp covariate effects —
   forbearance-driven delinquency transitions, the rate-collapse prepay surge — that trees capture
   well, where the net's smoothness edge matters less). This is the single most important read.

3. §7 lean per window: tie (flexibility wins, architecture secondary) / net-ahead (smooth target
   rewards the smooth learner) / GBT-ahead. Summarise the pattern across regimes.

4. IMPOSSIBLE-CELL FORENSICS. The breach is now 2/3 across the wave-1 regime windows: k2020
   (mean 1.56e-3) and k2025 (1.43e-3) FAIL the 1e-3 mean gate; k2023 (9.1e-4) passes; all three sit
   at ~0.95–0.9998 max-row mass. The earlier fit-maturity explanation is FALSIFIED — k2020 is the
   longest fit in the sweep (best_iter 1,919) and it breaches hardest, so the breach is not an
   early-stop artifact. For each regime window, break the impossible-cell mass down BY ORIGIN STATE
   and BY DESTINATION CELL: which origin(s) and which destination cell(s) carry the mass? Then
   decide, per flagged cell, whether it is TRULY structurally forbidden (e.g. an absorbing-state
   violation like prepaid→current or REO→current) or RARE-BUT-LEGAL (a cell the impossible-mask
   flags too aggressively). Report which, and recommend ONE of: (a) zero-and-renormalize those cells
   before the M19 pool roll, if truly forbidden — a real fix; or (b) recalibrate / relax the 1e-3
   mean gate, if the flagged cells are legal and the gate is just too tight. Note that k2023 passes
   while doing the SAME thing (~0.95 max-row mass) as the failing windows — that points at the mean
   threshold, not the model; confirm or refute with the breakdown.

5. Light calibration peek (NOT the full decision): reliability diagrams on RAW GBT softmax for
   current→prepaid and current→dpd_30 over the 5 regime windows (pooled + the 2020-21 split).
   On- or off-diagonal only; per-window temperature scaling deferred to full M18.

Then write a ~1-page PRELIMINARY memo to writeup/memos/gbt_regime_preliminary.md: the §7 read, the
COVID inversion (state the overturned prior honestly), the impossible-cell forensic finding + the
(a)/(b) recommendation — framed as PRELIMINARY (5 of 11 windows; pooled NLL, fillers, final
calibration pending). Friday-meeting draft material. 
```

✓ **Output:** a per-window comparison table, the COVID verdict, the impossible-cell forensic finding,
and a preliminary memo for Friday. **The COVID inversion is the headline** — GBT being *most*
competitive on the regime-shift window (not least) overturns the smooth-bias-helps-under-shift prior,
which is a more interesting and more honest finding than the tidy one. The forensics (task 4) converts
the 2/3 gate failure from "flagged, worth revisiting" into "here's the cell and here's the fix."

---

## STEP 4 — Finish the fillers + build the summary  *(pod, later tonight)*

When all 9 pod windows are banked (the watcher's "all 9" event), build the cross-window summary +
QA gate **on the pod** via the verify-only entrypoint (race-free, no refits):

```bash
# === ON THE POD ===
cd ~/diss-code && source .venv/bin/activate
ls /workspace/dissertation/models/gbt/full/   # expect 9 pod windows: k2016..k2018, k2020..k2025
.venv/bin/python -m floan.model.backtest --device cpu --gbt-only --verify-only
```

✓ **Proceed when all 9 are banked and verify-only passes** (no stray `.tmp`, every window's QA logged).

---

## STEP 5 — Full sync-back, mirror, tear down  *(Mac terminal)*

```bash
# === ON THE MAC (SSD mounted) ===
rsync -avz --progress -e "ssh -p 13761 -i ~/.ssh/id_ed25519" \
  "root@213.173.111.104:/workspace/dissertation/models/gbt/full/" \
  "/Volumes/SSD Felipe/dissertation/models/gbt/full/"

ls "/Volumes/SSD Felipe/dissertation/models/gbt/full/" | wc -l   # ✓ expect 11

cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
scripts/backup_ssd.sh
git add -A && git commit -m "M17: all 11 GBT windows banked (msh=1000) — regime read done"
git push origin main
```

✓ **Then tear down the pod** in the RunPod console (Stop/Terminate — the $1.04/hr meter stops).
**Only after the 11-window verify + mirror.** Never the other order.

---

## STEP 6 — Full M18 + M19  *(Codex on the Mac)*

Now run the complete versions from the original run guide's **PART 5** — they're the authoritative
finish:

- **Full M18:** folds GBT into Table A/B + AUC + NLL-by-origin across **all 11** windows, computes
  the **pooled all-windows NLL** (every test row predicted once), and makes the **final per-window
  calibration decision**. (The regime memo from Step 3 becomes the regime section of this.)
- **M19:** the pool-level roll-forward seam + economic translation (T4.2 / T5.1 / F5.2) + the memo —
  the GBT-baseline gate.

Paste the M18 and M19 prompts from `GBT_tomorrow_run_guide.md` PART 5. The only delta: tell M18 it can
**reuse the regime comparison from `writeup/memos/gbt_regime_preliminary.md`** and just extend it with
the fillers + pooled NLL, rather than redoing the regime windows.

---

## What you have at each checkpoint

- **After Step 3 (tonight, ~1h after wave 1):** the full regime evidence — GBT vs net vs logit across
  calm/late-cycle/COVID/rate-shock/latest, the COVID headline, a preliminary memo. **This is everything
  the Friday meeting needs.**
- **After Step 5:** all 11 windows banked, mirrored, committed; pod down.
- **After Step 6:** pooled NLL, full exhibits, calibration decision, pool-level translation — the
  complete GBT extension.

## Don't-trip list
- **Read NLLs within a window, never across** — k2023's 0.0697 is lower than k2015's 0.1036 because
  2023 was an easier year, not because GBT improved. Only GBT-vs-net-vs-logit *inside* a window means
  anything.
- **The COVID inversion is the headline** — k2020 GBT beats logit and the single net and ties the
  ensemble, so the net's lead does NOT widen under the regime shift; it vanishes. Write the overturned
  prior honestly. The gap must be read against the *authoritative* Mac-SSD NN numbers (Step 3), not
  the pod's `dissertation-data/` staging dir.
- **The impossible-cell breach is a 2/3 pattern, not a fluke** — k2020 + k2025 fail, k2023 passes,
  all at ~0.95 max-row mass; fit-maturity is falsified (longest fit breaches hardest). Step 3's task 4
  forensics resolves it (which cell, forbidden vs rare-but-legal, fix vs gate-recalibration) — don't
  ship the regime memo without it.
- **Don't tear down the pod before the full 11-window sync** (Step 5), and don't tear it down at Step 2
  (fillers still running).
- **SSD mounted** for Steps 2, 5, 6 (it's the destination + where the baselines live).
