# Two-Day Run Schedule — M11 → M15 (with all commands)

> Tags per step: ⏱ duration · 🌐 internet (continuous = live session; momentary = one command) · 💾 SSD plugged in.
> Prompts come from `PROMPTS.md`. Practical rule: leave the SSD in both days; Wi-Fi only matters while you're typing.

---

## DAY 1

### 1. Pod setup + M11 — ⏱ setup 15 min, M11 1–3 h · 🌐 continuous (blips OK) · 💾 no

Deploy pod (A4500; PRO 4500 if scarce). Copy IP/port from the Connect panel into `~/.ssh/config`, then test from the **Mac**:

```bash
ssh dissertation_pod
```

**Pod terminal** (remote VS Code window, open `/workspace/repo`):

```bash
mkdir -p "/Volumes/SSD Felipe" && ln -s /workspace/dissertation-data "/Volumes/SSD Felipe/dissertation"
cd /workspace/repo && git pull
source /workspace/.venv/bin/activate
python -c "import torch; print(torch.cuda.is_available())"   # True
ls /workspace/dissertation-data/models/nn                      # M10b run folders present
```

Paste the **M11 prompt**. Review when done: Table B (per-year + pooled), by-origin breakdown, AUC, calibration, memos 2a/2b.

### 2. M11 hand-off — ⏱ 10 min · 🌐 momentary · 💾 YES

**Pod:**

```bash
cd /workspace/repo && git pull --rebase && git push
```

**Mac:**

```bash
rsync -av dissertation_pod:/workspace/dissertation-data/outputs/ "/Volumes/SSD Felipe/dissertation/outputs/"
```

### 3. Parallel block — M12 ⏱ 2–4 h (unattended) ∥ M11b ⏱ 30–60 min

**Pod:** `/clear`, paste the **M12 prompt**. 🌐 launch only — the fits run in tmux; verify then disconnect freely:

```bash
tmux ls && nvidia-smi
```

**Mac (simultaneously, local VS Code Claude Code):** paste the **M11b prompt**. 🌐 no · 💾 YES.
First make sure the Mac repo has M11's memo commits:

```bash
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk" && git pull
```

### 4. M12 review + Phase-2 gate — ⏱ 20 min · 🌐 yes · 💾 YES

Reconnect; let the M12 session analyze and verify Accept. Then:

**Pod:**

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
git push   # if M11b/✅ commits are pending
```

### 5. M13 on pod — ⏱ 1–2 h · 🌐 continuous (interactive) · 💾 no

`/clear`, paste the **M13 prompt**. Watch for: toy-chain closed-form test PASS; h=1 probabilities match the evaluator. It commits; push:

```bash
cd /workspace/repo && git pull --rebase && git push
```

### 6. M14 GPU half (anchor scoring) — ⏱ 1–2 h unattended · 🌐 launch only · 💾 no

`/clear`, paste the **M14 prompt** (Claude Code will split GPU scoring vs Mac-side realized counts; tonight only the scoring runs). Verify it's tmux-resident, then walk away:

```bash
tmux ls && nvidia-smi
```

(If still running at day's end: leave the pod up overnight — there's a job on it.)

### 7. Sync home + terminate — ⏱ 20 min · 🌐 momentary · 💾 YES

**Pod:**

```bash
cd /workspace/repo && git pull --rebase && git push
```

**Mac:**

```bash
rsync -av dissertation_pod:/workspace/dissertation-data/models/nn "/Volumes/SSD Felipe/dissertation/models/"
rsync -av dissertation_pod:/workspace/dissertation-data/outputs/ "/Volumes/SSD Felipe/dissertation/outputs/"
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
git pull && ./dev/tools/backup_ssd.sh
```

RunPod console: **Stop → Terminate** the pod. Close the remote VS Code window. *(GPU chapter closed.)*

---

## DAY 2 — all Mac, no pod

### 8. M14 second half (realized counts + Accept) — ⏱ 2–4 h · 🌐 no · 💾 YES

Local Claude Code: continue/finish M14 per its session plan (realized-outcome queries against the panel, predicted-vs-realized artifacts). When it commits:

```bash
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk" && git push
```

### 9. M15 — econ translation + memo + §4.3 swaps — ⏱ 2–3 h · 🌐 final push only · 💾 YES

Paste the **M15 prompt** (local). Watch the two cashflow closed-form unit tests pass; review T5.1's headline price-error reduction; confirm no placeholders remain in chapter 4. Then the Phase-3 gate:

```bash
./dev/tools/backup_ssd.sh
git push
```

### 10. Wrap — ⏱ 15 min · 🌐 momentary · 💾 no

Mark M11–M15 ✅ in `PROMPTS.md`:

```bash
git add dev/model_plan/PROMPTS.md && git commit -m "PROMPTS: Phase 2-3 complete" && git push
```

One-line status email to supervisor. Done — remaining work is prose.
