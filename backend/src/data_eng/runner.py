from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import ProductConfig, extract_root, load_config
from .io import partition_ref
from .pipeline import build_pipeline
from .utils import expand_date_range


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the data pipeline for a specific product_type, symbol, and dt partition(s)"
    )
    p.add_argument(
        "--product-type",
        required=True,
        choices=["future_mbo", "future_option_mbo", "equity_mbo", "equity_option_cmbp_1"],
        help="Product type",
    )
    p.add_argument(
        "--layer",
        required=True,
        choices=["bronze", "silver", "gold", "all"],
        help="Pipeline layer to run (bronze, silver, gold, or all)",
    )
    p.add_argument(
        "--symbol",
        required=True,
        help="Symbol to process (ES base for future_mbo silver/gold, contracts like ESM6)",
    )
    p.add_argument(
        "--dt",
        help="Single partition date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--dates",
        help="Comma-separated (2025-06-01,2025-06-02) or range (2025-06-01:2025-06-10)",
    )
    p.add_argument(
        "--start-date",
        help="Start date for range (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end-date",
        help="End date for range (YYYY-MM-DD)",
    )
    p.add_argument(
        "--config",
        default="src/data_eng/config/datasets.yaml",
        help="Path to dataset config YAML (default: src/data_eng/config/datasets.yaml)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for cross-date processing (default: 8)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear output partitions before running (silver/gold only)",
    )
    return p.parse_args()


def _build_tasks(
    product_type: str,
    layer: str,
    symbol: str,
    dates: list[str],
    repo_root: Path,
) -> list[tuple[str, str]]:
    return [(symbol, d) for d in dates]


def _clear_stage_output(cfg, stage, symbol: str, dt: str) -> None:
    dataset_key = stage.io.output
    if isinstance(dataset_key, list):
        dataset_keys = dataset_key
    else:
        dataset_keys = [dataset_key]

    for key in dataset_keys:
        if key.startswith("bronze."):
            raise ValueError(f"Refusing to clear bronze output: {key}")
        out_ref = partition_ref(cfg, key, symbol, dt)
        if out_ref.dir.exists():
            shutil.rmtree(out_ref.dir)


def _resolve_product(cfg, product_type: str, symbol: str) -> ProductConfig | None:
    """Resolve product config for futures types, None for equities."""
    if product_type not in {"future_mbo", "future_option_mbo"}:
        return None
    if not cfg.products:
        return None
    try:
        return cfg.product_for_symbol(symbol)
    except (ValueError, KeyError):
        return None


def run_single_date(
    product_type: str,
    layer: str,
    symbol: str,
    dt: str,
    config_path: Path,
    overwrite: bool,
) -> dict:
    """Run pipeline for one symbol/date pair."""
    try:
        repo_root = Path.cwd()
        cfg = load_config(repo_root=repo_root, config_path=config_path)
        product = _resolve_product(cfg, product_type, symbol)

        stages = build_pipeline(product_type, layer)

        for stage in stages:
            if overwrite:
                _clear_stage_output(cfg, stage, symbol, dt)
            stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=dt, product=product)

        return {'dt': dt, 'symbol': symbol, 'status': 'success'}
    except Exception as e:
        return {'dt': dt, 'symbol': symbol, 'status': 'error', 'error': str(e)}


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    config_path = repo_root / args.config
    
    if args.dt:
        dates = [args.dt]
    else:
        dates = expand_date_range(
            dates=args.dates,
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    if not dates:
        raise ValueError("Must provide --dt, --dates, or --start-date/--end-date")

    if args.overwrite and args.layer not in {"silver", "gold"}:
        raise ValueError("Overwrite is supported for silver and gold layers only")

    tasks = _build_tasks(
        product_type=args.product_type,
        layer=args.layer,
        symbol=args.symbol,
        dates=dates,
        repo_root=repo_root,
    )
    if not tasks:
        raise ValueError("No tasks to run")

    task_symbols = sorted({symbol for symbol, _ in tasks})
    
    print(f"Product Type: {args.product_type}")
    print(f"Layer:        {args.layer}")
    print(f"Symbol:       {args.symbol}")
    print(f"Dates:        {len(dates)}")
    print(f"Contracts:    {len(task_symbols)}")
    print(f"Tasks:        {len(tasks)}")
    print(f"Workers:      {args.workers}")
    print()
    
    if args.workers > 1 and len(tasks) > 1:
        print(f"Processing {len(tasks)} tasks with {args.workers} workers\n")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_single_date,
                    args.product_type,
                    args.layer,
                    symbol,
                    dt,
                    config_path,
                    args.overwrite,
                ): (symbol, dt)
                for symbol, dt in tasks
            }
            
            for future in as_completed(futures):
                result = future.result()
                if result['status'] == 'success':
                    print(f"✅ {result['symbol']} {result['dt']}")
                else:
                    print(f"❌ {result['symbol']} {result['dt']}: {result['error']}")
    else:
        cfg = load_config(repo_root=repo_root, config_path=config_path)
        stages = build_pipeline(args.product_type, args.layer)

        for symbol, dt in tasks:
            print(f"Processing {symbol} {dt}...")
            product = _resolve_product(cfg, args.product_type, symbol)

            for stage in stages:
                if args.overwrite:
                    _clear_stage_output(cfg, stage, symbol, dt)
                stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=dt, product=product)

            print(f"✅ {symbol} {dt}\n")
    
    print(f"DONE")
    print(f"Lake Root: {repo_root / 'lake'}")


if __name__ == "__main__":
    main()
