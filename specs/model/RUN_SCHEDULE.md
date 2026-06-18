# Run Schedule — what's left (M15 → wrap)

> Tags per step: ⏱ duration · 🌐 internet (momentary = one git command; the work itself is offline) · 💾 SSD.
> Prompts come from `PROMPTS.md` (use the split **M15a / M15b / M15c** blocks). All Mac, no pod — the GPU chapter is closed.

## Status

- ✅ **M1–M14** complete, committed, pushed; `models/` + `outputs/` synced to the SSD and mirrored. **Pod terminated.**
- ▶️ **Remaining: M15a → M15b → M15c → wrap.** All on the Mac, no GPU.

## SSD rule (read this first)

**The SSD must be connected for every M15 session.** `config.require_drive()` fails fast without it, and each sub-step reads or writes SSD-resident data:

- **M15a** reads the frozen model checkpoints (`models/nn/…`) and the eval-pool panel, reuses `outputs/tables/pool_level/pools_random_k*.parquet`, and writes `smm_paths_k*.parquet` — all on the SSD.
- **M15b** reads the SMM paths + pools and writes `T5.1` / `F5.2` to `outputs/` — SSD.
- **M15c** copies the real figure/table artifacts off the SSD into `writeup/latex/figs/`, and `backup_ssd.sh` mirrors the SSD.

The only purely-repo bits (the engine's hermetic unit tests, the `latexmk` recompile itself) happen *inside* sessions that already need the SSD, so the simple rule is: **SSD plugged in the whole time.** Internet is only needed momentarily for `git push`/`pull`.

---

## 1. M15a — cashflow engine + monthly SMM paths — ⏱ ~1.5–2.5 h · 🌐 momentary (commit) · 💾 YES (essential)

`/clear`, paste the **M15a** prompt. It runs an internal parallel block:

- **(A) SMM-path regeneration** launches first as a background CPU job (`nohup`, logs to `logs/m15/smm_paths.log`). This is the slow piece (~1–2 h). Confirm it's actually running before moving on:

```bash
tail -f "/Volumes/SSD Felipe/dissertation/logs/m15/smm_paths.log"   # should be growing; Ctrl-C to stop watching
```

- **(B) cashflow engine** is built and unit-tested while (A) runs.

**Gates to watch:** the two closed-form tests must pass to **~1e-8** (zero-prepay = annuity; constant-SMM = survival schedule) — this is the time-variable part, like M13's toy-chain. And the SMM run must finish with sane CPR ranges. Then it commits `M15a`.

```bash
git push
```

> **If you background (A) and step away:** keep the Mac awake — `caffeinate -i` (or lid open / on power). A sleeping laptop *pauses* the job (unlike the pod). It's only ~1–2 h, so this isn't really an overnight job.
> **Fallback if CPU is painfully slow:** a ~20-min pod round-trip (redeploy → `ssh` → symlink → `git pull` → run just the SMM scoring on GPU → rsync `smm_paths*` home → terminate) is the only step that would benefit from GPU.

## 2. M15b — CPR/WAL/price errors + T5.1 + F5.2 — ⏱ ~30–60 min · 🌐 momentary (commit) · 💾 YES (essential)

`/clear`, paste the **M15b** prompt. Light compute over ~550 pools/anchor.

**Watch:** the headline ensemble-vs-logit |price-error| reduction is reported **per anchor** (not pooled), and the regime signature matches the counts — ensemble beats logit off-COVID, inverts at Dec-2019 under frozen-t0 macro. T5.1 + F5.2 exist for ensemble vs logit at ≥3 regime-spanning anchors. Then it commits `M15b`.

```bash
git push
```

## 3. M15c — pool memo + §4.3 LaTeX swap (Phase-3 gate) — ⏱ ~1–1.5 h · 🌐 momentary (commit) · 💾 YES (essential)

`/clear`, paste the **M15c** prompt. Writes `writeup/memos/03_pool.md`, then swaps the remaining §4.3 placeholders for the real artifacts (copied off the SSD) and recompiles.

**Gates to watch:** `latexmk` recompiles clean — **zero undefined references**, no Phase-3 placeholders left in chapter 4. Every M15 Accept criterion verified with evidence. Then it commits `M15c`.

**Phase-3 gate (you run these):**

```bash
./scripts/backup_ssd.sh        # 💾 SSD essential — mirrors the SSD dirs
git push                         # 🌐
```

## 4. Wrap — ⏱ 15 min · 🌐 momentary · 💾 no

Mark M11–M15 ✅ in `PROMPTS.md`, commit, push:

```bash
git add specs/model/PROMPTS.md && git commit -m "PROMPTS: Phase 2-3 complete" && git push
```

One-line status email to supervisor. Done — remaining work is prose.

---

### Order at a glance

`M15a` (SSD) → push → `M15b` (SSD) → push → `M15c` (SSD) → `backup_ssd.sh` + push → wrap. SSD connected for 1–3; only the final wrap commit needs nothing but the repo. Splittable across sittings — each sub-prompt restates "prior sub-tasks complete & committed," so you can stop after any push and resume later.
