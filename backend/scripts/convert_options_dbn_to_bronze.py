"""
Convert ES Options DBN files to Bronze Parquet.

Converts trades, NBBO (mbp-1), and statistics schemas from DBN to Bronze Parquet.

Usage:
    cd backend
    uv run python scripts/convert_options_dbn_to_bronze.py --all
    uv run python scripts/convert_options_dbn_to_bronze.py --date 2025-06-05
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import databento as db

_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))

def convert_dbn_to_bronze(dbn_file: Path, output_dir: Path, schema: str, date_str: str) -> int:
    """Convert single DBN file to Bronze Parquet."""
    
    # Read DBN
    store = db.DBNStore.from_file(str(dbn_file))
    df = store.to_df().reset_index()
    
    if df.empty:
        return 0
    
    # Transform to Bronze schema (simplified - just essentials)
    if schema == 'trades':
        df_out = pd.DataFrame({
            'ts_event_ns': df['ts_event'].astype('int64'),
            'ts_recv_ns': df['ts_recv'].astype('int64'),
            'source': 'DATABENTO_CME',
            'underlying': 'ES',
            'option_symbol': df['symbol'].astype(str),
            'price': df['price'].astype('float64'),
            'size': df['size'].astype('int64'),
            'seq': df.get('sequence', 0).astype('int64'),
        })
    elif schema == 'mbp-1':
        df_out = pd.DataFrame({
            'ts_event_ns': df['ts_event'].astype('int64'),
            'ts_recv_ns': df['ts_recv'].astype('int64'),
            'source': 'DATABENTO_CME',
            'underlying': 'ES',
            'option_symbol': df['symbol'].astype(str),
            'bid_px': df.get('bid_px_00', 0).astype('float64'),
            'ask_px': df.get('ask_px_00', 0).astype('float64'),
            'bid_sz': df.get('bid_sz_00', 0).astype('int64'),
            'ask_sz': df.get('ask_sz_00', 0).astype('int64'),
            'seq': df.get('sequence', 0).astype('int64'),
        })
    elif schema == 'statistics':
        # Statistics schema uses 'quantity' for open interest
        # Parse strike and right from symbol (format: "ESZ6 P3650")
        def parse_symbol(sym):
            """Extract strike and right from option symbol."""
            try:
                parts = str(sym).split()
                if len(parts) >= 2:
                    right_strike = parts[1]  # e.g., "P3650"
                    right = right_strike[0]  # "P" or "C"
                    strike = float(right_strike[1:])  # "3650" -> 3650.0
                    return strike, right
            except:
                pass
            return 0.0, ''
        
        parsed = df['symbol'].apply(parse_symbol)
        strikes = [p[0] for p in parsed]
        rights = [p[1] for p in parsed]
        
        df_out = pd.DataFrame({
            'ts_event_ns': df['ts_event'].astype('int64'),
            'ts_recv_ns': df['ts_recv'].astype('int64'),
            'source': 'DATABENTO_CME',
            'underlying': 'ES',
            'option_symbol': df['symbol'].astype(str),
            'exp_date': pd.NaT,  # Would need contract calendar to resolve
            'strike': pd.Series(strikes, dtype='float64'),
            'right': pd.Series(rights, dtype='str'),
            'open_interest': df['quantity'].astype('float64'),
        })
    else:
        raise ValueError(f"Unknown schema: {schema}")
    
    # Partition by hour
    df_out['hour'] = pd.to_datetime(df_out['ts_event_ns'], unit='ns', utc=True).dt.hour
    
    for hour, hour_df in df_out.groupby('hour'):
        hour_dir = output_dir / f'hour={hour:02d}'
        hour_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = hour_dir / f'{schema}_{date_str}_h{hour:02d}.parquet'
        hour_df_out = hour_df.drop(columns=['hour'])
        
        table = pa.Table.from_pandas(hour_df_out, preserve_index=False)
        pq.write_table(table, output_path, compression='zstd')
    
    return len(df_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--date', type=str)
    
    args = parser.parse_args()
    
    raw_dir = _backend_dir / 'data' / 'raw' / 'es_options'
    bronze_root = _backend_dir / 'data' / 'bronze' / 'options'
    
    # Find DBN files to convert
    if args.all:
        dbn_files = list(raw_dir.glob('*.dbn'))
    elif args.date:
        date_compact = args.date.replace('-', '')
        dbn_files = list(raw_dir.glob(f'*{date_compact}*.dbn'))
    else:
        print("Provide --all or --date")
        return 1
    
    print(f"Converting {len(dbn_files)} DBN files to Bronze Parquet...")
    
    for dbn_file in dbn_files:
        # Parse filename: es-opt-20251029.mbp-1.dbn
        parts = dbn_file.stem.split('.')
        if len(parts) < 2:
            continue
        
        date_compact = parts[0].split('-')[-1]  # 20251029
        schema = parts[1]  # trades or mbp-1
        
        # Format date
        date_str = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}"
        
        # Output directory
        if schema == 'trades':
            schema_name = 'trades'
        elif schema == 'mbp-1':
            schema_name = 'nbbo'
        elif schema == 'statistics':
            schema_name = 'statistics'
        else:
            print(f"  ⏭️  Skipping unknown schema: {schema}")
            continue
        
        output_dir = bronze_root / schema_name / 'underlying=ES' / f'date={date_str}'
        
        print(f"\n{date_str} {schema_name}:")
        print(f"  Reading {dbn_file.name}...")
        
        count = convert_dbn_to_bronze(dbn_file, output_dir, schema, date_str)
        print(f"  ✅ Wrote {count:,} records to {output_dir}")
    
    print("\n✅ Conversion complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

