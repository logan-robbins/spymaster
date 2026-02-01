#!/usr/bin/env python3
"""
Filter ES options MBO files to 0DTE only and save as parquet.
Replaces large DBN files with much smaller parquet files.
"""

import databento as db
from datetime import datetime, date
from pathlib import Path
import pandas as pd

def get_0dte_instruments(definition_file: str, target_date: date) -> set:
    """Extract instrument IDs for contracts expiring on target_date."""
    store = db.DBNStore.from_file(definition_file)
    df = store.to_df()
    
    # Filter for options with strike price
    options_df = df[df['strike_price'].notna()].copy()
    options_df['exp_date'] = options_df['expiration'].dt.date
    
    # Get 0DTE contracts
    dte_contracts = options_df[options_df['exp_date'] == target_date]
    instrument_ids = set(dte_contracts['instrument_id'].tolist())
    
    return instrument_ids

def filter_and_save_parquet(mbo_file: Path, dte_instruments: set, trade_date: date) -> dict:
    """Filter MBO file to 0DTE and save as parquet."""
    print(f"\n  Processing: {mbo_file.name}")
    
    # Get original size
    orig_size_gb = mbo_file.stat().st_size / (1024**3)
    print(f"    Original size: {orig_size_gb:.2f} GB")
    
    # Load and filter
    print(f"    Loading DBN data...")
    store = db.DBNStore.from_file(str(mbo_file))
    df = store.to_df()
    
    total_records = len(df)
    print(f"    Total records: {total_records:,}")
    
    print(f"    Filtering to 0DTE...")
    df_filtered = df[df['instrument_id'].isin(dte_instruments)]
    filtered_records = len(df_filtered)
    
    pct = (filtered_records / total_records * 100) if total_records > 0 else 0
    print(f"    Kept: {filtered_records:,} / {total_records:,} ({pct:.1f}%)")
    
    # Create parquet filename
    parquet_file = mbo_file.with_suffix('.parquet')
    
    # Save as parquet
    print(f"    Writing parquet...")
    df_filtered.to_parquet(parquet_file, compression='snappy', index=False)
    
    # Get new size
    new_size_gb = parquet_file.stat().st_size / (1024**3)
    saved_gb = orig_size_gb - new_size_gb
    
    print(f"    Parquet size: {new_size_gb:.2f} GB")
    print(f"    Saved: {saved_gb:.2f} GB ({(saved_gb/orig_size_gb)*100:.0f}% reduction)")
    
    # Delete original DBN file
    print(f"    Deleting original DBN file...")
    mbo_file.unlink()
    print(f"    ✓ Complete: {parquet_file.name}")
    
    return {
        'date': trade_date,
        'orig_size_gb': orig_size_gb,
        'new_size_gb': new_size_gb,
        'saved_gb': saved_gb,
        'total_records': total_records,
        'filtered_records': filtered_records,
        'percentage': pct
    }

def main():
    base_path = Path(__file__).parent / 'lake' / 'raw' / 'source=databento'
    mbo_base = base_path / 'product_type=future_option_mbo' / 'symbol=ES' / 'table=market_by_order_dbn'
    def_base = base_path / 'dataset=definition'
    
    # Find all MBO files
    mbo_files = list(mbo_base.glob('**/*.mbo.dbn'))
    
    print("="*70)
    print("FILTER ES OPTIONS TO 0DTE AND CONVERT TO PARQUET")
    print("="*70)
    print(f"\nFound {len(mbo_files)} ES options MBO files")
    
    results = []
    total_orig = 0
    total_new = 0
    
    for mbo_file in sorted(mbo_files):
        try:
            # Extract date from filename
            filename = mbo_file.name
            date_str = filename.split('-')[2].replace('.mbo.dbn', '')
            
            if len(date_str) != 8:
                print(f"\n  ⚠️  Skipping {filename}: Can't parse date")
                continue
            
            trade_date = datetime.strptime(date_str, '%Y%m%d').date()
            print(f"\n{'='*70}")
            print(f"Date: {trade_date}")
            print(f"{'='*70}")
            
            # Find definition file
            def_file = def_base / f'glbx-mdp3-{date_str}.definition.dbn'
            if not def_file.exists():
                # Try in subdirectories
                def_candidates = list(def_base.glob(f'**/glbx-mdp3-{date_str}.definition.dbn'))
                if def_candidates:
                    def_file = def_candidates[0]
                else:
                    print(f"  ⚠️  Definition file not found, skipping")
                    continue
            
            print(f"  Loading 0DTE instruments from {def_file.name}...")
            dte_instruments = get_0dte_instruments(str(def_file), trade_date)
            print(f"  Found {len(dte_instruments)} 0DTE contracts")
            
            # Filter and save
            result = filter_and_save_parquet(mbo_file, dte_instruments, trade_date)
            results.append(result)
            
            total_orig += result['orig_size_gb']
            total_new += result['new_size_gb']
            
        except Exception as e:
            print(f"\n  ❌ Error processing {mbo_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nProcessed {len(results)} files successfully")
    print()
    
    for r in results:
        print(f"  {r['date']}: {r['orig_size_gb']:.2f} GB → {r['new_size_gb']:.2f} GB ({r['percentage']:.1f}% kept)")
    
    print()
    print(f"Total original size: {total_orig:.2f} GB")
    print(f"Total new size:      {total_new:.2f} GB")
    print(f"Total saved:         {total_orig - total_new:.2f} GB")
    print(f"Reduction:           {((total_orig - total_new) / total_orig * 100):.1f}%")
    print()
    print("✓ All files converted to parquet format!")
    print("✓ Raw layer now contains filtered 0DTE data only")

if __name__ == '__main__':
    main()
