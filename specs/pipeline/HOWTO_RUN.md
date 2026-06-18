# HOWTO — Running this plan with Claude Code

Everything Claude Code needs is in `dev/`. The workflow is: open it in the project, give it the kickoff prompt, then work through `03_CLAUDE_CODE_TASKS.md` one task at a time.

---

## 1. Open Claude Code in the project

```bash
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
claude
```

Launch from the project root so Claude Code sees both `dev/pipeline_plan/` (specs) and `dev/pipeline/CLAUDE.md` (conventions). It auto-loads a `CLAUDE.md` from the working directory; since yours is in `dev/pipeline/`, either `cd dev/pipeline` before launching or just reference it in the kickoff prompt — both work.

## 2. Connect the SSD first

Make sure `SSD Felipe` is mounted at `/Volumes/SSD Felipe/dissertation/` before starting, or Task 0's `require_drive()` guard will refuse to run.

## 3. Kickoff prompt (paste first)

```
Read these files in full before doing anything:
- dev/pipeline/CLAUDE.md   (the hard rules — follow them exactly)
- dev/pipeline_plan/00_OVERVIEW.md
- dev/pipeline_plan/01_SCHEMA.md
- dev/pipeline_plan/02_PIPELINE_STAGES.md
- dev/pipeline_plan/03_CLAUDE_CODE_TASKS.md

Then summarise the plan back to me in ~8 bullets and confirm you understand
the no-full-load rule and the immutable-raw layout. Do NOT write any code yet.
We'll work through 03_CLAUDE_CODE_TASKS.md one task at a time, and I'll approve
each before you move on. Start by proposing Task 0 only.
```

## 4. Working rhythm

One task at a time, in order. After each, check the acceptance criteria in `03` before saying "proceed." Two checkpoints matter most:

- **Task 2 (inventory):** make it show you `outputs/inventory_summary.md` and confirm the vintage sequence is contiguous, all files share one release cut-off, and any truncated/corrupt file is quarantined — *before* any conversion.
- **Task 3 (conversion):** insist it smoke-tests on a small healthy vintage (seconds) and reports the round-trip row count before running `--all` on the full ~800 GB.

---

## 5. Model selection & rate limits  ⚠️

A build this long burns through usage fast if every step runs on the most capable (most expensive) model. Most of this pipeline is mechanical and does **not** need a frontier model. Match the model to the task and you'll go further before hitting limits.

**Switch models with the `/model` command** inside Claude Code (it sets the model for the session; run it again any time to change). Rough mapping:

| Use a **cheaper/faster** model (e.g. Haiku/Sonnet) for | Use the **most capable** model only for |
|---|---|
| Task 0 scaffold, `requirements.txt`, boilerplate | Task 1 schema reconciliation against the official glossary (positions 110–112) |
| Stage 2 streaming conversion code (mechanical) | Task 5 seven-state target logic + window/censoring edge cases |
| Stage 3 casts, Stage 6 row-count reconciliation | Debugging a non-obvious failure or unexpected data shape |
| Re-running stages, file edits, log inspection | Design decisions where the spec is ambiguous |

Default to the cheaper model and **escalate only when a task is genuinely subtle.** Drop back down as soon as you're past it.

**Watching rate limits:**

- Tell Claude Code up front: *"Prefer the cheapest model that can do each task; only ask me to switch up for the schema-mapping and target-logic steps."*
- Keep context lean — run **`/clear`** between stages so it isn't re-carrying earlier conversation; have it re-read the relevant spec section instead.
- Don't paste large data samples into chat (a few rows is plenty); let the code stream files, not the conversation.
- The heavy ~800 GB conversion is **compute on your machine, not model tokens** — once Stage 2 code is approved, running it costs no model usage, so kick off `--all` and step away.
- If you hit a limit mid-build, the stages are idempotent and per-quarter, so you can resume later exactly where you left off.

---

### Recap

Open Claude Code in the project root, mount the SSD, paste the kickoff prompt, and approve tasks one at a time — pausing at the inventory and the `2014Q1` smoke test. To stay inside rate limits, run the mechanical steps on a cheaper model via `/model` and reserve the capable model for the schema reconciliation, the seven-state target logic, and real debugging; use `/clear` between stages and let code (not chat) handle the data.
