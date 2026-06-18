"""F3.6 — Vintage effects: default hazard by origination cohort (``01_EDA.md`` §3).

Empirical ``current → dpd_30`` (fresh-delinquency onset) rate at a *fixed early loan
age* — the 12–36-month window where defaults concentrate — by origination year.
Holding loan age fixed strips out the seasoning effect (F3.1) so what remains is the
cohort effect: the 2006–07 vintages, underwritten at the top of the bubble, stand out
sharply against neighbouring years. Vintage is therefore a feature in its own right,
not reducible to the contemporaneous covariates.

One DuckDB pass emits dpd_30/at-risk counts per origination year over ``current``
loan-months aged 12–36; Wilson 95% CIs come from ``eda.wilson_ci``.

Run:  python -m floan.analysis.f3_6_vintage
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from floan.analysis import eda_common as eda

config = eda.config
AGE_LO, AGE_HI = 12, 36  # fixed early-life window (months), inclusive


def build(con) -> pl.DataFrame:
    df = con.execute(
        f"""
        SELECT orig_ym // 100 AS orig_year,
               count(*) FILTER (WHERE state_next IS NOT NULL) AS den,
               count(*) FILTER (WHERE state_next = 'dpd_30') AS num
        FROM panel
        WHERE state = 'current' AND "Loan Age" BETWEEN {AGE_LO} AND {AGE_HI}
        GROUP BY 1
        HAVING count(*) FILTER (WHERE state_next IS NOT NULL) > 0
        ORDER BY 1
        """
    ).pl()
    lo, hi = eda.wilson_ci(df["num"].to_numpy(), df["den"].to_numpy())
    return df.with_columns(
        (pl.col("num") / pl.col("den")).alias("rate"),
        pl.Series("lo", lo), pl.Series("hi", hi),
    )


def figure(df: pl.DataFrame):
    x = df["orig_year"].to_list()
    rate = [v * 100 for v in df["rate"].to_list()]
    yerr = [[(r - l) * 100 for r, l in zip(df["rate"], df["lo"])],
            [(h - r) * 100 for h, r in zip(df["hi"], df["rate"])]]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, rate, color="tab:red", alpha=0.8, yerr=yerr, capsize=2,
                  ecolor="grey", error_kw={"lw": 0.8})
    for yr in (2006, 2007):
        if yr in x:
            bars[x.index(yr)].set_color("darkred")
    ax.set_xlabel("origination year")
    ax.set_ylabel(f"fresh-delinquency rate at age {AGE_LO}–{AGE_HI} mo (% / month)")
    ax.set_title("F3.6  Vintage effects — current→dpd_30 hazard at fixed early age",
                 fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    config.require_drive()
    con = eda.connect()
    df = build(con)
    con.close()

    s_csv, s_tex = eda.save_table(
        df.select("orig_year", "rate", "lo", "hi", "num", "den"), "F3.6_vintage_hazard",
        caption=f"current→dpd_30 rate at loan age {AGE_LO}–{AGE_HI} mo by origination "
                "year (F3.6).",
    )
    fig = figure(df)
    png, pdf = eda.save_figure(fig, "F3.6_vintage")

    # Accept evidence: the worst cohort is a bubble vintage (2005–2008).
    worst = df.sort("rate", descending=True).head(1)
    wy = int(worst["orig_year"].item())
    calm = df.filter(pl.col("orig_year").is_in([2003, 2004, 2012, 2013]))["rate"].mean()
    print(f"F3.6 vintage hazard (current→dpd_30 at age {AGE_LO}–{AGE_HI}mo): "
          f"{df.height} cohorts {df['orig_year'].min()}–{df['orig_year'].max()}")
    print(f"  worst cohort: {wy}  {worst['rate'].item()*100:.3f}%/mo")
    print(f"  calm cohorts (2003/04/12/13 mean): {calm*100:.3f}%/mo  "
          f"-> bubble-vintage multiple ×{worst['rate'].item()/calm:.1f}")
    print(f"  worst vintage in bubble window 2005–2008: {2005 <= wy <= 2008}")
    print(f"\nwrote {png}\n      {pdf}\n      {s_csv}\n      {s_tex}")


if __name__ == "__main__":
    main()
