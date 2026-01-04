#!/usr/bin/env python3
"""
Migrate existing bronze options data to lake/ structure.

Source: data/bronze/options/{trades,nbbo,statistics}/underlying=ES/date=YYYY-MM-DD/hour=HH/
Target: lake/bronze/source=databento/product_type=future_option/symbol=ES/table={trades,nbbo,statistics}/dt=YYYY-MM-DD/hour=HH/
"""

from pathlib import Path
import shutil

def migrate_options_bronze():
    backend_dir = Path(__file__).parent.parent
    source_base = backend_dir / "data" / "bronze" / "options"
    target_base = backend_dir / "lake" / "bronze" / "source=databento" / "product_type=future_option"
    
    tables = ["trades", "nbbo", "statistics"]
    
    for table in tables:
        source_table_dir = source_base / table
        if not source_table_dir.exists():
            print(f"Skipping {table} - source does not exist")
            continue
        
        print(f"\nProcessing {table}...")
        
        for underlying_dir in source_table_dir.glob("underlying=*"):
            underlying = underlying_dir.name.split("=")[1]
            symbol = underlying
            
            for date_dir in underlying_dir.glob("date=*"):
                date = date_dir.name.split("=")[1]
                dt = date
                
                for hour_dir in date_dir.glob("hour=*"):
                    hour = hour_dir.name.split("=")[1]
                    
                    target_dir = (
                        target_base / 
                        f"symbol={symbol}" / 
                        f"table={table}" / 
                        f"dt={dt}" / 
                        f"hour={hour}"
                    )
                    
                    target_dir.parent.parent.mkdir(parents=True, exist_ok=True)
                    
                    if target_dir.exists():
                        print(f"  Skip (exists): {symbol}/{table}/dt={dt}/hour={hour}")
                    else:
                        shutil.copytree(hour_dir, target_dir)
                        print(f"  Migrated: {symbol}/{table}/dt={dt}/hour={hour}")

if __name__ == "__main__":
    print("Migrating bronze options data to lake structure...")
    print("=" * 80)
    migrate_options_bronze()
    print("\n" + "=" * 80)
    print("Migration complete!")

