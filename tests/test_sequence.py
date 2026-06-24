"""M26 unit tests — the sequence builder's leakage-guarded windowing + right-pad scatter.

Hermetic (synthetic panel/points through the ACTUAL window SQL; pure-numpy scatter) — the
M26a-test analogue, lifted to seq-to-one:

  * **ordering** — each point's window is its trailing ≤T months, recency-ranked (pos/cnt);
  * **no future** — the newest real timestep is exactly t (never past t);
  * **per-loan reset** — a point's window contains only that loan's rows;
  * **flip-state_next** — flipping every state_next leaves the window STRUCTURE (which rows,
    pos, cnt) byte-identical (selection never depends on the target), and state_next is not
    a model feature at all;
  * **right-pad scatter** — real steps land at [0..len-1] oldest→newest, mask + lengths right.
"""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import polars as pl

from floan.model import features as F
from floan.model import sequence as S

_SEL = ["Loan Identifier", "period_ym", "state", "state_next"]


def _panel() -> pl.DataFrame:
    # loan A: 6 months with a delinquency arc; loan B: 3 months (later start) — proves reset.
    rows = [
        ("A", 201401, "current"), ("A", 201402, "dpd_30"), ("A", 201403, "dpd_60"),
        ("A", 201404, "current"), ("A", 201405, "current"), ("A", 201406, "dpd_30"),
        ("B", 201403, "current"), ("B", 201404, "current"), ("B", 201405, "dpd_30"),
    ]
    loan = [r[0] for r in rows]
    ym = [r[1] for r in rows]
    st = [r[2] for r in rows]
    return pl.DataFrame({
        "Loan Identifier": loan, "shard": [0] * len(rows),
        "period": [date(y // 100, y % 100, 1) for y in ym], "period_ym": ym,
        "state": st, "state_next": st[1:] + ["prepaid"],   # arbitrary; must not drive windows
    })


def _points() -> pl.DataFrame:
    # 3 prediction points: A@201406 (full T window), A@201403 (short), B@201405 (short, reset).
    pts = [(0, "A", 201406), (1, "A", 201403), (2, "B", 201405)]
    return pl.DataFrame({
        "point_id": [p[0] for p in pts], "loan": [p[1] for p in pts],
        "t_ym": [p[2] for p in pts],
        "t_mi": [(p[2] // 100) * 12 + (p[2] % 100) - 1 for p in pts],
    })


def _run(panel: pl.DataFrame, T: int = 4) -> pl.DataFrame:
    con = duckdb.connect()
    con.register("panel", panel.to_arrow())
    con.register("points", _points().to_arrow())
    out = con.execute(S._window_sql(_SEL, T, shard_lt=99)).pl()
    con.close()
    return out


def test_window_trailing_and_ordering():
    out = _run(_panel(), T=4)
    by = {pid: g.sort("period_ym") for (pid,), g in out.group_by(["point_id"], maintain_order=True)}

    # A@201406: trailing 4 months 201403..201406; newest (pos=1) == t; cnt=4.
    a6 = by[0]
    assert a6.get_column("period_ym").to_list() == [201403, 201404, 201405, 201406]
    assert a6.get_column("cnt").to_list() == [4, 4, 4, 4]
    assert a6.filter(pl.col("pos") == 1).get_column("period_ym").item() == 201406  # newest=t

    # A@201403: only 3 months exist ≤ t within the window (201401..201403); cnt=3.
    assert by[1].get_column("period_ym").to_list() == [201401, 201402, 201403]
    assert by[1].get_column("cnt").to_list() == [3, 3, 3]

    # No future: every point's max real period_ym == its t.
    for pid, tym in ((0, 201406), (1, 201403), (2, 201405)):
        assert int(by[pid].get_column("period_ym").max()) == tym


def test_window_per_loan_reset():
    out = _run(_panel(), T=4)
    # B@201405's window is loan B only (201403..201405) — no bleed from loan A.
    b = out.filter(pl.col("point_id") == 2)
    assert b.get_column("Loan Identifier").unique().to_list() == ["B"]
    assert sorted(b.get_column("period_ym").to_list()) == [201403, 201404, 201405]


def test_window_flip_state_next_invariant():
    """Flipping every state_next leaves the window STRUCTURE byte-identical (selection is
    independent of the target); only the carried state_next column changes."""
    base = _run(_panel(), T=4).sort(["point_id", "period_ym"])
    flip = _run(_panel().with_columns(pl.lit("foreclosure").alias("state_next")), T=4) \
        .sort(["point_id", "period_ym"])
    struct = ["point_id", "Loan Identifier", "period_ym", "pos", "cnt", "state"]
    assert base.select(struct).equals(flip.select(struct))
    assert not base.select("state_next").equals(flip.select("state_next"))  # positive control


def test_state_next_not_a_feature():
    """state_next (the target) is in none of the model feature blocks."""
    assert "state_next" not in (F.CONTINUOUS + F.CATEGORICAL + F.BINARY)


def test_scatter_right_pad():
    """Right-pad scatter: real steps at [0..cnt-1] oldest→newest, pads after, mask/lengths."""
    # point 0: cnt=3 (ti 2,1,0 for pos 1,2,3); point 1: cnt=4 (full). enc rows carry their id.
    pid = np.array([0, 0, 0, 1, 1, 1, 1])
    pos = np.array([1, 2, 3, 1, 2, 3, 4])
    cnt = np.array([3, 3, 3, 4, 4, 4, 4])
    ei = np.array([102, 101, 100, 203, 202, 201, 200])   # newest→oldest per point
    ti = cnt - pos                                        # = [2,1,0, 3,2,1,0]
    cont = np.arange(300, dtype=np.float32).reshape(300, 1)
    cat = np.zeros((300, 1), np.int64)
    binb = np.zeros((300, 1), np.float32)

    Xc, Xk, Xb, mask, lengths = S._scatter(pid, ti, ei, cnt, N=2, T=4,
                                           cont=cont, cat=cat, binb=binb)
    # point 0: real at [0,1,2] = oldest→newest = enc 100,101,102; pad at [3].
    assert Xc[0, :, 0].tolist() == [100.0, 101.0, 102.0, 0.0]
    assert mask[0].tolist() == [1, 1, 1, 0]
    assert lengths.tolist() == [3, 4]
    # point 1: full length, oldest→newest = enc 200..203, no pad.
    assert Xc[1, :, 0].tolist() == [200.0, 201.0, 202.0, 203.0]
    assert mask[1].tolist() == [1, 1, 1, 1]
