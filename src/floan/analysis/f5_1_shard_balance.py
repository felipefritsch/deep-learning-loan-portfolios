"""F5.1 — Shard balance (``01_EDA.md`` §5; carry-over from the May-8 notes).

``shard = hash(loan_id) % N_SHARDS`` (``config.N_SHARDS`` = 256, pinned
``SHARD_SEED``) is the loan-keyed, vintage-stable bucket used for leakage-safe splits
and minibatch randomization in Phase 2. This sanity check confirms the shards are
near-uniform in both rows and loans, and loan-disjoint by construction (each loan in
exactly one shard). One DuckDB pass: exact ``count(*)`` rows and exact
``count(DISTINCT loan)`` per shard. (Exact, not HyperLogLog: ``approx_count_distinct``
grouped over the 256 shards proved unreliable — it reported a large spurious spread in
loans/shard, contradicting the uniform exact count, so this figure uses the exact
distinct despite the heavier scan.)

Run:  python dev/analysis/f5_1_shard_balance.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

import eda_common as eda

config = eda.config


def build(con) -> pl.DataFrame:
    return con.execute(
        """
        SELECT shard,
               count(*)                          AS rows,
               count(DISTINCT "Loan Identifier") AS loans
        FROM panel
        GROUP BY shard
        ORDER BY shard
        """
    ).pl()


def figure(df: pl.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(df["shard"], df["rows"], width=1.0, color="tab:blue")
    axes[0].axhline(df["rows"].mean(), color="black", ls="--", lw=0.8, label="mean")
    axes[0].set_ylabel("loan-months")
    axes[0].set_title("F5.1  Shard balance — rows and loans per shard (256 shards)",
                      fontsize=12, loc="left")
    axes[0].legend(fontsize=8)
    axes[1].bar(df["shard"], df["loans"], width=1.0, color="tab:green")
    axes[1].axhline(df["loans"].mean(), color="black", ls="--", lw=0.8)
    axes[1].set_ylabel("loans")
    axes[1].set_xlabel("shard")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    config.require_drive()
    con = eda.connect()
    df = build(con)
    con.close()

    s_csv, s_tex = eda.save_table(
        df, "F5.1_shard_balance",
        caption="Rows and (approx) loans per shard (F5.1).",
    )
    fig = figure(df)
    png, pdf = eda.save_figure(fig, "F5.1_shard_balance")

    rows_cv = df["rows"].std() / df["rows"].mean()
    loans_cv = df["loans"].std() / df["loans"].mean()
    print(f"F5.1 shard balance: {df.height} shards (expected {config.N_SHARDS})")
    print(f"  rows/shard : mean {df['rows'].mean():,.0f}  "
          f"min {df['rows'].min():,}  max {df['rows'].max():,}  CV {rows_cv:.4f}")
    print(f"  loans/shard: mean {df['loans'].mean():,.0f}  "
          f"min {df['loans'].min():,}  max {df['loans'].max():,}  CV {loans_cv:.4f}")
    ok = df.height == config.N_SHARDS and rows_cv < 0.02 and loans_cv < 0.02
    print(f"  near-uniform (all 256 shards present, CV < 2%): {ok}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
