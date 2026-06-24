"""M26a unit tests — the history-summary derivation (history.py).

Hermetic (a synthetic in-memory panel run through the ACTUAL production DuckDB SQL — no
SSD, no torch). The point of the probe is its leakage guard, so the tests pin exactly that:

  * every feature is a function of records with ``period < t`` ONLY (a delinquent row never
    counts itself — the ``ROWS … 1 PRECEDING`` frame), with the four values asserted against
    a hand-computed trajectory;
  * ``PARTITION BY loan`` isolation (one loan's history never bleeds into another's);
  * the FLIP-``state_next`` invariant — mutating every ``state_next`` leaves all four
    features byte-identical (empirical proof the target is never read);
  * the augmented block encoders map nulls safely (prev_state → UNK, ever → 0, counts →
    scaler centre), so the augmented net's input is well-formed.
"""

from __future__ import annotations

from datetime import date

import duckdb
import polars as pl

from floan.model import features as F
from floan.model import history as H


# L1 traces a full delinquency arc (clean → episode 1 (30→60) → cure → episode 2); L2 is a
# second loan present only to prove cross-loan isolation. state_next is arbitrary (the SQL
# must never read it).
def _synthetic_panel() -> pl.DataFrame:
    rows = [
        ("L1", 201401, "current"), ("L1", 201402, "current"), ("L1", 201403, "dpd_30"),
        ("L1", 201404, "dpd_60"), ("L1", 201405, "current"), ("L1", 201406, "current"),
        ("L1", 201407, "dpd_30"), ("L1", 201408, "current"),
        ("L2", 201401, "current"), ("L2", 201402, "dpd_30"), ("L2", 201403, "current"),
    ]
    loan = [r[0] for r in rows]
    ym = [r[1] for r in rows]
    st = [r[2] for r in rows]
    return pl.DataFrame({
        "Loan Identifier": loan,
        "period": [date(y // 100, y % 100, 1) for y in ym],
        "period_ym": ym,
        "state": st,
        "state_next": st[1:] + ["prepaid"],          # arbitrary; must not affect output
    })


def _derive(panel: pl.DataFrame) -> pl.DataFrame:
    """Run the production history SELECT over a registered Arrow table (no SSD)."""
    con = duckdb.connect()
    con.register("t", panel.to_arrow())
    out = con.execute(H._history_select_sql("t", "TRUE", None)).pl()
    con.close()
    return out


# Expected (prev_state, ever_delinquent, months_since_last_delinq, n_prior_episodes) for L1,
# computed by hand using ONLY records strictly before each feature month.
_EXPECT_L1 = {
    201401: (None, 0, None, 0),        # first record: no prior anything
    201402: ("current", 0, None, 0),
    201403: ("current", 0, None, 0),   # the row IS dpd_30 but it must NOT count itself
    201404: ("dpd_30", 1, 1, 1),       # episode 1 began at 201403 (1 month ago)
    201405: ("dpd_60", 1, 1, 1),       # still 1 month since last delinq (201404)
    201406: ("current", 1, 2, 1),
    201407: ("current", 1, 3, 1),      # episode 2 begins AT 201407 → not yet counted
    201408: ("dpd_30", 1, 1, 2),       # now two episodes began before t (201403, 201407)
}


def test_history_causal_features_exact():
    out = _derive(_synthetic_panel())
    got = {int(r["period_ym"]): (r["prev_state"], int(r["ever_delinquent"]),
                                 r["months_since_last_delinq"],
                                 int(r["n_prior_delinq_episodes"]))
           for r in out.filter(pl.col("Loan Identifier") == "L1").iter_rows(named=True)}
    assert got == _EXPECT_L1


def test_history_strict_precedence_no_self_count():
    """The first delinquency month (201403) must read ever=0, episodes=0, months=None —
    proof the ``1 PRECEDING`` frame excludes the row's own (delinquent) state."""
    out = _derive(_synthetic_panel())
    r = out.filter((pl.col("Loan Identifier") == "L1")
                   & (pl.col("period_ym") == 201403)).row(0, named=True)
    assert int(r["ever_delinquent"]) == 0
    assert int(r["n_prior_delinq_episodes"]) == 0
    assert r["months_since_last_delinq"] is None


def test_history_partition_isolation():
    """L2's short history is independent of L1 (PARTITION BY loan)."""
    out = _derive(_synthetic_panel())
    r = out.filter((pl.col("Loan Identifier") == "L2")
                   & (pl.col("period_ym") == 201403)).row(0, named=True)
    assert r["prev_state"] == "dpd_30"
    assert int(r["ever_delinquent"]) == 1
    assert r["months_since_last_delinq"] == 1
    assert int(r["n_prior_delinq_episodes"]) == 1


def test_history_flip_state_next_invariant():
    """Mutating EVERY state_next leaves all four features byte-identical — the target is
    never read (the leakage guard's empirical proof)."""
    panel = _synthetic_panel()
    flipped = panel.with_columns(pl.lit("foreclosure").alias("state_next"))
    assert _derive(panel).sort(["Loan Identifier", "period_ym"]).equals(
        _derive(flipped).sort(["Loan Identifier", "period_ym"]))


def test_augmented_block_encoders_handle_nulls():
    """The history block encodes through the existing pipeline: prev_state null → UNK(0),
    ever_delinquent null → 0, the two counts' nulls → scaler centre (→ 0 standardized)."""
    df = pl.DataFrame({
        "prev_state": ["current", "dpd_30", None],
        "ever_delinquent": [0, 1, None],
        "months_since_last_delinq": [None, 3, 12],
        "n_prior_delinq_episodes": [0, 1, None],
    })
    codes = F.Vocab.fit(df, cols=H.HIST_CATEGORICAL).transform(df)
    assert codes[2, 0] == F.UNK                                   # null prev_state → UNK

    binb = F.binary_matrix(df, H.HIST_BINARY)
    assert binb[0, 0] == 0.0 and binb[1, 0] == 1.0 and binb[2, 0] == 0.0   # null ever → 0

    z = F.Scaler.fit(df, cols=H.HIST_CONTINUOUS).transform(df)
    j = H.HIST_CONTINUOUS.index("months_since_last_delinq")
    assert abs(float(z[0, j])) < 1e-9                             # null count → centre (0)
    assert z.shape[1] == len(H.HIST_CONTINUOUS)
