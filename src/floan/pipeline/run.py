"""run.py — thin per-stage, per-quarter orchestrator for the Fannie Mae pipeline.

Dispatches to the stage modules so the whole dataset can be processed
incrementally and resumed after interruption. Every stage is idempotent and
per-quarter; re-running skips work already done.

Examples
--------
    python run.py inventory                      # Stage 1 (content-aware resume)
    python run.py inventory --force              #   full rescan
    python run.py convert  --quarter 2017Q2      # Stage 2, one vintage
    python run.py convert  --all                 # Stage 2, all non-quarantined
    python run.py clean    --all                 # Stage 3
    python run.py panel    --all                 # Stage 4 (seven-state target)
    python run.py qa                             # Stage 6 QA report
    python run.py sample   --cutoff-ym 201501    # Stage 5 balanced sample + scaler
    python run.py pipeline --quarter 2020Q4      # Stages 2→3→4 end-to-end, one vintage

Stages run independently; a crash mid-run is recoverable (completed quarters are
skipped). The external SSD must be mounted — every stage calls require_drive().
For long runs, wrap with `caffeinate -ims` so the Mac doesn't sleep.
"""

from __future__ import annotations

import argparse

from floan.pipeline import config

STAGES = ["inventory", "convert", "clean", "panel", "qa", "sample", "pipeline"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fannie Mae pipeline orchestrator (per-stage, per-quarter, idempotent)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--quarter", help="process a single vintage, e.g. 2017Q2")
    sel.add_argument("--all", action="store_true", help="process all vintages")
    ap.add_argument("--force", action="store_true", help="reprocess even if outputs exist")
    ap.add_argument("--include-quarantined", action="store_true",
                    help="convert (don't skip) quarantined vintages [convert/pipeline]")
    ap.add_argument("--reprobe", action="store_true",
                    help="reuse cached stats, recompute integrity probes [inventory]")
    ap.add_argument("--per-class", type=int, default=1_000_000,
                    help="rows per state_next class in the balanced sample [sample]")
    ap.add_argument("--cutoff-ym", type=int,
                    help="rolling-backtest cut-off YYYYMM (train on period_ym < cutoff) [sample]")
    args = ap.parse_args()

    config.require_drive()
    quarters = None if args.all else ([args.quarter] if args.quarter else None)

    if args.stage == "inventory":
        import s1_inventory
        s1_inventory.run_inventory(force=args.force, reprobe=args.reprobe)

    elif args.stage == "convert":
        import s2_to_parquet
        s2_to_parquet.run(quarters, include_quarantined=args.include_quarantined,
                          force=args.force)

    elif args.stage == "clean":
        import s3_clean
        s3_clean.run(quarters, force=args.force)

    elif args.stage == "panel":
        import s4_panel
        s4_panel.run(quarters, force=args.force)

    elif args.stage == "qa":
        import s6_qa
        s6_qa.main()

    elif args.stage == "sample":
        import s5_sample
        con = s5_sample.connect()
        try:
            samp = s5_sample.balanced_sample(con, per_class=args.per_class,
                                             cutoff_ym=args.cutoff_ym)
            scaler = s5_sample.fit_scaler(con, cutoff_ym=args.cutoff_ym)
            scaled = s5_sample.apply_scaler(samp, scaler)
            p = s5_sample.materialize(scaled, f"sample_{args.cutoff_ym or 'full'}")
            print(f"sample: {samp.height:,} rows -> {p}")
            print(f"scaler: {scaler.get('_path')}")
        finally:
            con.close()

    elif args.stage == "pipeline":
        # Per-quarter end-to-end: Stage 2 → 3 → 4 (each idempotent/resumable).
        import s2_to_parquet, s3_clean, s4_panel
        s2_to_parquet.run(quarters, include_quarantined=args.include_quarantined,
                          force=args.force)
        s3_clean.run(quarters, force=args.force)
        s4_panel.run(quarters, force=args.force)


if __name__ == "__main__":
    main()
