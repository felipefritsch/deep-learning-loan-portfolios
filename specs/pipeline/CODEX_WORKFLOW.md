# Codex workflow for the data pipeline

Everything Codex needs is in `specs/pipeline/` (the specifications),
`src/floan/pipeline/` (the implementation), and the layered `AGENTS.md` files. Work through
`03_CODEX_TASKS.md` one bounded task at a time.

---

## 1. Open Codex at the repository root

```bash
codex
```

Launch from the project root so Codex automatically reads the root `AGENTS.md`. The kickoff prompt
below explicitly loads `src/floan/pipeline/AGENTS.md`; alternatively, launch Codex with that
directory as the working directory to have the nested instructions loaded automatically.

## 2. Connect the SSD first

Make sure `SSD Felipe` is mounted at `/Volumes/SSD Felipe/dissertation/` before starting, or Task 0's `require_drive()` guard will refuse to run.

## 3. Kickoff prompt (paste first)

```
Read these files in full before doing anything:
- src/floan/pipeline/AGENTS.md   (the hard rules — follow them exactly)
- specs/pipeline/00_OVERVIEW.md
- specs/pipeline/01_SCHEMA.md
- specs/pipeline/02_PIPELINE_STAGES.md
- specs/pipeline/03_CODEX_TASKS.md

Then summarise the plan back to me in ~8 bullets and confirm you understand
the no-full-load rule and the immutable-raw layout. Do NOT write any code yet.
We'll work through 03_CODEX_TASKS.md one task at a time, and I'll approve
each before you move on. Start by proposing Task 0 only.
```

## 4. Working rhythm

One task at a time, in order. After each, check the acceptance criteria in `03` before saying "proceed." Two checkpoints matter most:

- **Task 2 (inventory):** make it show you `outputs/inventory_summary.md` and confirm the vintage sequence is contiguous, all files share one release cut-off, and any truncated/corrupt file is quarantined — *before* any conversion.
- **Task 3 (conversion):** insist it smoke-tests on a small healthy vintage (seconds) and reports the round-trip row count before running `--all` on the full ~800 GB.

---

## 5. Model selection and context management

A build this long burns through usage fast if every step runs on the most capable (most expensive) model. Most of this pipeline is mechanical and does **not** need a frontier model. Match the model to the task and you'll go further before hitting limits.

Use `/model` in Codex to match reasoning effort to the task:

| Use a **faster/lower-cost** option for | Use the **most capable** option for |
|---|---|
| Task 0 scaffold, `requirements.txt`, boilerplate | Task 1 schema reconciliation against the official glossary (positions 110–112) |
| Stage 2 streaming conversion code (mechanical) | Task 5 seven-state target logic + window/censoring edge cases |
| Stage 3 casts, Stage 6 row-count reconciliation | Debugging a non-obvious failure or unexpected data shape |
| Re-running stages, file edits, log inspection | Design decisions where the spec is ambiguous |

Default to the cheaper model and **escalate only when a task is genuinely subtle.** Drop back down as soon as you're past it.

**Watching rate limits:**

- Tell Codex up front: *"Prefer the cheapest model that can do each task; only ask me to switch up for the schema-mapping and target-logic steps."*
- Keep context lean by starting a fresh task between stages and pointing it to the relevant spec section.
- Don't paste large data samples into chat (a few rows is plenty); let the code stream files, not the conversation.
- The heavy ~800 GB conversion is **compute on your machine, not model tokens** — once Stage 2 code is approved, running it costs no model usage, so kick off `--all` and step away.
- If you hit a limit mid-build, the stages are idempotent and per-quarter, so you can resume later exactly where you left off.

---

### Recap

Open Codex in the project root, mount the SSD, paste the kickoff prompt, and approve tasks one at a
time—pausing at inventory and the first conversion smoke test. Use `/model` to reserve the most
capable option for schema reconciliation, target logic, and non-obvious debugging. Let the code
stream the data; do not paste large samples into the conversation.
