# Run Schedule — re-anchored to M13 (evening start)

> Tags per step: ⏱ duration · 🌐 internet (continuous = live session; momentary = one command) · 💾 SSD plugged in.
> Prompts come from `PROMPTS.md`. Practical rule: leave the SSD in; Wi-Fi only matters while you're typing.
> **Pod now pushes to GitHub natively** (SSH key registered) — no more Mac-bridge patches. Code travels by `git push` from the pod; only `outputs/` + `models/` (gitignored) still come home by rsync.

## Status as of now

- ✅ **M1–M12** complete, committed, pushed. M12 reviewed (Accept criteria pass; k=2015 grid intact). See `robustness.{json,md}`.
- ✅ **M11b** complete — §4.1 numbers verified, chapters 3–4 placeholders swapped, §4.2 ensemble comparison confirmed per-window (no pooled all-years ensemble figure), clean 47-page recompile.
- 📝 **Memo 02c** (`writeup/memos/02c_robustness_caveats.md`) written, **pending its own commit** (left unstaged by M11b — correct).
- 🔄 **M13** — roll-forward harness, **in progress on the pod** (interactive). Watch the two gates: toy-chain closed-form **PASS**, and h=1 probabilities **equal `evaluate.py` exactly**.

---

## TONIGHT

### 1. Commit the 02c memo — ⏱ 1 min · 💾 — (do whenever)

```bash
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
git add writeup/memos/02c_robustness_caveats.md
git commit -m "memo 02c: M12 robustness caveats" && git push
```

### 2. Finish M13 — ⏱ ~1 h remaining (interactive) · 🌐 continuous · 💾 no

Stay with it through the two Accept gates (toy-chain PASS, h=1 == evaluator exactly) — exact-match checks are where the time goes. It commits; then from the **pod**:

```bash
cd /workspace/repo && git pull --rebase && git push
```

### 3. Launch M14 GPU half (anchor scoring), then bed — ⏱ launch 5 min, runs 1–2 h unattended · 🌐 launch only · 💾 no

Only the GPU scoring runs tonight (Mac-side realized counts wait for tomorrow). `/clear`, paste the **M14 prompt**, let it queue the scoring as a tmux/nohup batch, then verify with your own eyes before sleeping:

```bash
tmux ls && nvidia-smi
```

Want: a session listed (or a python proc under `nvidia-smi`) and GPU-Util above idle. Then **leave the pod running overnight** (there's a live job on it) and sleep. Disconnect-safe.

> **Too tired after M13?** Don't pay for an idle pod. Either launch M14 now (5 min) and let it work overnight, or **terminate the pod after M13** and run M14 GPU half first thing tomorrow on a fresh pod (15 min redeploy — schedule below). Don't leave it idling unused.

---

## TOMORROW — mostly Mac

### 4. Bring M14 scoring home + terminate pod — ⏱ 20 min · 🌐 momentary · 💾 YES

**Pod** (if M14 committed code):

```bash
cd /workspace/repo && git pull --rebase && git push
```

**Mac:**

```bash
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
git pull
rsync -av dissertation_pod:/workspace/dissertation-data/models/nn "/Volumes/SSD Felipe/dissertation/models/"
rsync -av dissertation_pod:/workspace/dissertation-data/outputs/ "/Volumes/SSD Felipe/dissertation/outputs/"
./dev/tools/backup_ssd.sh
```

Verify the M14 scoring artifacts landed on the SSD, **then** RunPod console: **Stop → Terminate**. Close the remote VS Code window. *(GPU chapter closed.)*

*(If you terminated last night instead: redeploy a pod, set `~/.ssh/config` to the new IP, `ssh dissertation_pod`, then on the pod `ln -s /workspace/dissertation-data "/Volumes/SSD Felipe/dissertation"`, `cd /workspace/repo && git pull && source /workspace/.venv/bin/activate`, confirm `torch.cuda.is_available()` True, run M14 GPU half, then do step 4.)*

### 5. M14 second half — realized counts + Accept — ⏱ 2–4 h · 🌐 no · 💾 YES

Local Claude Code: finish M14 per its session plan (realized-outcome queries against the panel, predicted-vs-realized artifacts). When it commits:

```bash
git push
```

### 6. M15 — econ translation + memo + §4.3 swaps — ⏱ 2–3 h · 🌐 final push only · 💾 YES

Paste the **M15 prompt** (local). Watch the two cashflow closed-form unit tests pass; review T5.1's headline price-error reduction; confirm no Phase-3 placeholders remain in chapter 4. Then the Phase-3 gate:

```bash
./dev/tools/backup_ssd.sh
git push
```

### 7. Wrap — ⏱ 15 min · 🌐 momentary · 💾 no

Mark M11–M15 ✅ in `PROMPTS.md`:

```bash
git add dev/model_plan/PROMPTS.md && git commit -m "PROMPTS: Phase 2-3 complete" && git push
```

One-line status email to supervisor. Done — remaining work is prose.
