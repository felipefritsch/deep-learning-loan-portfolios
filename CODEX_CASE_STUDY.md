# Codex case study: removing a six-hour pricing bottleneck

> **Status:** this is a sample interview write-up reconstructed from a real repository change,
> commit `bb847b5`. The code change, performance figures, and validation outcome come from the
> repository history. The prompt below is a faithful reconstruction because the original chat
> transcript was not retained; it should not be presented as verbatim.

## Context

The sequence-pricing pipeline builds a trailing monthly history for every loan before running
Monte-Carlo paths through a GRU or transformer. At a full-population anchor, initialization appeared
to hang: roughly 600,000 loans and millions of history rows took about six hours before simulation
even began.

The problem was algorithmic rather than GPU-related. For every distinct loan, the implementation
evaluated `loans == loan` across the entire history array. With `L` loans and `N` rows, history
assembly was therefore O(`L × N`).

## Sample prompt to Codex

> Investigate the full-population stall in `SeqPredictor.__init__`. Profile the history-assembly
> path before changing it. Preserve each loan's incoming period order and the exact encoded arrays;
> do not change model inputs, sampling, or pricing logic. Propose the smallest replacement for the
> per-loan boolean-mask loop, explain its complexity, and show the diff before applying it. Validate
> the new grouping against the existing implementation on a representative slice, including loans
> with one and many history rows. The outputs must be byte-identical. Add lightweight progress
> reporting to the long simulation loop, then run the relevant tests and report timings.

This prompt is useful because it fixes the success criteria before implementation: semantic
identity, stable within-loan ordering, improved asymptotic complexity, a surgical diff, and measured
runtime evidence.

## Proposed and accepted diff

Codex identified the repeated boolean mask and proposed a stable sort followed by a single split at
loan boundaries:

```python
order = np.argsort(loans, kind="stable")
loans_sorted = loans[order]
bounds = np.flatnonzero(loans_sorted[1:] != loans_sorted[:-1]) + 1

for group in np.split(order, bounds):
    loan = loans[group[0]]
    self._hist[loan] = (continuous[group], categorical[group], binary[group])
```

The stable sort groups a loan's rows while preserving their original period order. The change also
added progress output approximately every 10% of the Monte-Carlo loop, including observed loans per
second. No model or cashflow mathematics changed.

## Review decisions

| Decision | Outcome | Reason |
|---|---|---|
| Stable sort plus boundary split | Accepted | Removes the repeated full-array scans while preserving within-loan order. |
| Plain/default sort | Rejected | An unstable sort could silently reorder equal-loan rows and therefore corrupt the sequence. |
| Polars group-by rewrite | Rejected for this fix | Larger semantic surface and unnecessary conversion after features were already encoded as arrays. |
| Change batching or reduce the population | Rejected | Would hide the bottleneck or change the experiment rather than correct the algorithm. |
| Progress logging | Accepted | Makes a multi-hour GPU job observable without changing results. |

## Validation evidence

- The optimized histories were compared with the original implementation and were byte-identical.
- Full-population history assembly fell from approximately **six hours to four seconds**.
- The committed implementation is a **17-line addition / 4-line removal** in
  [`src/floan/model/seq_pricing.py`](src/floan/model/seq_pricing.py).
- The current hermetic suite passes **166 tests**, including sequence-pricing identity,
  path-dependence, aggregation, and vectorized-versus-reference checks.

## What this demonstrates

The value of Codex here was not merely generating code. It helped isolate the real complexity
failure, propose a narrow alternative, and make the long-running workflow observable. My role was
to specify the invariants, reject changes that altered the experiment, review the diff, and demand
equivalence and timing evidence before accepting it.

For an application or interview, I would use this structure but replace the reconstructed prompt
with the original transcript if it becomes available, and bring the actual commit diff rather than
claiming that the repository alone proves who authored each line.
