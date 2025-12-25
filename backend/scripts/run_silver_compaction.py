
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lake.silver_compactor import SilverCompactor

def main():
    print("Initializing Silver Compactor...")
    compactor = SilverCompactor()
    
    # We will iterate over known schemas and compact whatever is available
    schemas = [
        ('futures.trades', 'ES'),
        ('futures.mbp10', 'ES'),
        ('options.trades', 'SPY'),
        # Add others if needed: stocks.trades, stocks.quotes
    ]
    
    for schema, partition_val in schemas:
        print(f"\nScanning for {schema} ({partition_val})...")
        try:
            dates = compactor.get_available_bronze_dates(schema, partition_val)
            print(f"Found dates: {dates}")
            
            for date in dates:
                print(f"Compacting {date}...")
                result = compactor.compact_date(date, schema, partition_val, force=True) # forcing to ensure fresh analysis
                print(f"Result: {result['status']} - Written: {result.get('rows_written', 0)}, Removed Dups: {result.get('duplicates_removed', 0)}")
        except Exception as e:
            print(f"Error processing {schema}: {e}")

if __name__ == "__main__":
    main()

