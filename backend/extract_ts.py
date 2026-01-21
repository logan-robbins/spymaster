
import pandas as pd
import re
from pathlib import Path

def extract_ts():
    log_path = Path("hunt_output.log")
    if not log_path.exists():
        print("Log not found")
        return

    content = log_path.read_text()
    
    # Look for Trigger Row
    # The output format of named tuple matches:
    # Trigger Row: Pandas(Index=..., window_start_ts_ns=..., ...)
    
    match = re.search(r"window_end_ts_ns=(\d+)", content)
    if match:
        ts_ns = int(match.group(1))
        # Convert to NY time
        ts = pd.Timestamp(ts_ns, unit="ns", tz="UTC").tz_convert("America/New_York")
        print(f"Timestamp detected: {ts}")
    else:
        print("Timestamp not found in log")
        # Print tail to see what happened
        print("Log tail:")
        print(content[-500:])

if __name__ == "__main__":
    extract_ts()
