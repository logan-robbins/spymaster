"""
Validate all processed dates with both validation scripts.

Usage:
    cd backend
    uv run python scripts/validate_processed_dates.py --workers 4
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def get_processed_dates(version: str) -> list:
    """Get dates that have been processed."""
    silver_dir = Path(__file__).parent.parent / 'data' / 'silver' / 'features' / 'es_pipeline' / f'version={version}'
    
    if not silver_dir.exists():
        return []
    
    dates = []
    for date_dir in silver_dir.glob('date=*'):
        date_str = date_dir.name.replace('date=', '')
        dates.append(date_str)
    
    return sorted(dates)


def validate_date(date: str, script_name: str) -> dict:
    """Run validation script for a single date."""
    try:
        result = subprocess.run(
            ['uv', 'run', 'python', '-m', f'scripts.{script_name}', '--date', date],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        return {
            'date': date,
            'script': script_name,
            'success': result.returncode == 0,
            'stdout': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
            'stderr': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
        }
    except Exception as e:
        return {
            'date': date,
            'script': script_name,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Validate processed dates')
    parser.add_argument('--version', default='4.0.0', help='Canonical version')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    
    args = parser.parse_args()
    
    dates = get_processed_dates(args.version)
    
    if not dates:
        print(f"No processed dates found for version {args.version}")
        return 1
    
    print(f"Found {len(dates)} processed dates")
    print(f"Running 2 validations × {len(dates)} dates = {len(dates) * 2} total checks")
    print(f"Workers: {args.workers}\n")
    
    # Create validation jobs
    jobs = []
    for date in dates:
        jobs.append((date, 'validate_stage_14_label_outcomes'))
        jobs.append((date, 'validate_es_pipeline'))
    
    # Run in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(validate_date, date, script): (date, script)
            for date, script in jobs
        }
        
        for future in as_completed(futures):
            date, script = futures[future]
            result = future.result()
            results.append(result)
            
            status = "✅" if result['success'] else "❌"
            script_short = script.replace('validate_', '').replace('_', ' ')
            print(f"{status} {result['date']}: {script_short}")
            
            if not result['success']:
                error_msg = result.get('error') or result.get('stderr', '')[:200]
                print(f"     Error: {error_msg}")
    
    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    total = len(results)
    passed = sum(1 for r in results if r['success'])
    failed = total - passed
    
    print(f"Total: {total} validations")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed validations:")
        for r in results:
            if not r['success']:
                print(f"  ❌ {r['date']}: {r['script']}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

