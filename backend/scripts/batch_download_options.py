
import os
import sys
import time
from pathlib import Path
import databento as db
from dotenv import load_dotenv

# Setup environment
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
load_dotenv(backend_dir / '.env')

def batch_download():
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        print("Missing DATABENTO_API_KEY")
        return

    client = db.Historical(key=api_key)
    
    date_str = "2026-01-06"
    
    # Comprehensive list of parents for 0DTE coverage
    parents = ["ES.OPT", "EW.OPT", "EW1.OPT", "EW2.OPT", "EW3.OPT", "EW4.OPT"]
    for i in range(1, 6):
        parents.append(f"E{i}.OPT")
        for char in ['A', 'B', 'C', 'D', 'E']:
            parents.append(f"E{i}{char}.OPT")
            
    print(f"Submitting Batch Job for {date_str} with {len(parents)} parents...")
    
    try:
        job = client.batch.submit_job(
            dataset="GLBX.MDP3",
            symbols=parents,
            schema="mbo",
            start=date_str,
            end="2026-01-07",
            stype_in="parent",
            delivery="download", # We will download manually via client, usually 'download' means keep on server? 
            # encoding="dbn" default
        )
        job_id = job["id"]
        print(f"Job IDs: {job_id}. Waiting for completion...")
        
        # Poll
        while True:
            updated_job = client.batch.list_jobs(since=date_str) # Get recent jobs
            # Filter for our job
            my_job = next((j for j in updated_job if j["id"] == job_id), None)
            
            if not my_job:
                print("Job not found in list?")
                time.sleep(5)
                continue
            
            print(f"Job Object: {my_job}") # Debug
            status = my_job.get("state") # Try 'state' if 'status' missing
            if not status: status = my_job.get("status")
            
            print(f"Status: {status}")
            
            if status == "done":
                break
            elif status == "error":
                print("Job failed.")
                return
                
            time.sleep(10)
            
        print("Job Done. Downloading...")
        
        target_dir = backend_dir / "lake/raw/source=databento/product_type=future_option_mbo/symbol=ES/table=market_by_order_dbn"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"glbx-mdp3-20260106.mbo.dbn"
        
        # Download
        # client.batch.download returns list of files
        downloaded_files = client.batch.download(
            job_id=job_id,
            output_dir=target_dir
        )
        
        # Rename or verify
        # Batch download names files like <job_id>/<file>.dbn or similar.
        # We need to coalesce or rename.
        print(f"Downloaded: {downloaded_files}")
        
        if len(downloaded_files) == 1:
            # Move to standard name
            downloaded = downloaded_files[0] # Path object
            downloaded.rename(target_file)
            print(f"Renamed to {target_file}")
            
            # Cleanup job directory if empty?
            # client.batch.download creates the file directly in output_dir usually
            pass
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    batch_download()
