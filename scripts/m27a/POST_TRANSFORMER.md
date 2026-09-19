# Post-transformer-roll handoff (W2 → W2c)

`finish_xf.sh` self-harvests: on completion it commits + pushes the transformer results
(`artifacts/results/m27a_results_10m/k*_xf_s*.json`) to `origin/main` and writes
`/workspace/dissertation/outputs/XF_DONE.txt` whose last line is `SAFE TO STOP`.

## 1. On the pod (Codex) — confirm it finished and pushed
Paste:

> Check `/workspace/dissertation/outputs/XF_DONE.txt`. If its last line is `SAFE TO STOP`, confirm the
> transformer results are on origin/main: `git -C /workspace/repo rev-list --left-right --count
> origin/main...HEAD` must be `0 0`, and `ls /workspace/repo/artifacts/results/m27a_results_10m/k*_xf_s0.json
> | wc -l` should be 11. Report both numbers. If `XF_DONE.txt` is absent or the roll didn't finish, run
> `bash scripts/m27a/finish_xf.sh` (resumable — it skips done runs, then harvests + pushes), then re-check.

Then **Stop** the pod (not Terminate — the network volume + results persist either way, but Stop halts billing).

## 2. On the Mac — pull, then continue in Codex
```
cd "/Users/felipefritsch/Documents/Masters MCF Oxford/Dissertation/Dissertation - Asset Loans Default Risk"
git pull --ff-only origin main
ls artifacts/results/m27a_results_10m/k*_xf_s0.json | wc -l    # want 11
```
Then tell Codex **"transformer done"** → it folds the transformer in as the 4th arm of `tab:seqrolling`
(adds the column + a sentence, recompiles, shows the diff).
