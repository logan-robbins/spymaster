# Technical Commands (IMPLEMENT.md Companion)

- `cd backend` — run all python commands from the backend workspace using the uv environment.
- `nohup uv run python scripts/batch_download_options.py --start 2025-12-22 --end 2026-01-21 --schemas mbo,statistics,definition --pause-seconds 60 --log-file logs/batch_options_jobs_2025-12-22_2026-01-21.log > /tmp/batch_submit_only_2025-12-22.log 2>&1 &` — submit Databento batch requests day-by-day with a 1-minute pause; logs job IDs for UI size validation.
- `tail -n 50 logs/batch_options_jobs_2025-12-22_2026-01-21.log` — verify submitted job IDs and parent counts per date/schema.
- `nohup uv run python scripts/rebuild_bronze_option_mbo.py --symbol ES --dt 2026-01-06 > /tmp/bronze_mbo_rebuild_2026-01-06.log 2>&1 &` — rebuild 0DTE option MBO bronze for the target replay date after raw file updates.
- `nohup uv run python scripts/run_pipeline_2026_01_06.py --dt 2026-01-06 > /tmp/pipeline_2026-01-06.log 2>&1 &` — recompute silver+gold outputs (book/wall/vacuum/radar/physics/gex/calibration) for UI replay.
- `tail -n 200 /tmp/pipeline_2026-01-06.log` — track stage completion or failures during the 2026-01-06 pipeline run.
- `ls lake/silver/product_type=future_mbo/symbol=ESH6/table=book_snapshot_1s/dt=2026-01-06` — confirm futures snapshot output exists for the replay date.
- `ls lake/silver/product_type=future_option_mbo/symbol=ESH6/table=gex_surface_1s/dt=2026-01-06` — confirm options GEX surface output exists for the replay date.
- `ls lake/gold/hud/symbol=ESH6/table=physics_norm_calibration/dt=2026-01-06` — confirm HUD normalization calibration output exists for the replay date.
