# Spymaster Backend

Python 3.12 backend using `uv` for dependency and command execution.

## Environment

- Python: `>=3.12,<3.13`
- Package/tool runner: `uv`
- Virtual environment location: `.venv`

## Setup

```bash
uv sync
```

## Common Commands

```bash
# Run tests
uv run pytest

# Inspect futures downloader options
uv run scripts/batch_download_futures.py --help

# Inspect equities downloader options
uv run scripts/batch_download_equities.py --help

# Run vacuum-pressure replay pipeline
uv run scripts/run_vacuum_pressure.py --help

# Warm cache artifacts
uv run scripts/warm_cache.py --help
```

## Lake Layout

Lake root: `lake/`

- Raw data: `lake/raw/source=databento/...`
- Bronze data: `lake/bronze/source=databento/...`
- Cache data: `lake/cache/...`

## GCS Transfer Status (Completed)

- Project: `transformer-478002`
- Bucket: `gs://spylake-478002-260215-47a822d7`
- Prefix: `gs://spylake-478002-260215-47a822d7/lake`
- Transfer scope: full recursive copy of local `lake/` structure and files
- Final parity:
- Local files: `27`
- Remote files: `27`
- Local bytes: `17220317736`
- Remote bytes: `17220317736`

## GCS Public Access Status (Enabled)

- Bucket: `gs://spylake-478002-260215-47a822d7`
- Public principal: `allUsers`
- Public roles applied:
- `roles/storage.objectViewer` (public object reads)
- `roles/storage.legacyBucketReader` (public bucket object listing)
- Anonymous verification:
- Object fetch HTTP status: `200`
- Bucket listing HTTP status: `200`

### Public Access Commands Used

```bash
gcloud storage buckets add-iam-policy-binding \
  gs://spylake-478002-260215-47a822d7 \
  --member=allUsers \
  --role=roles/storage.objectViewer

gcloud storage buckets add-iam-policy-binding \
  gs://spylake-478002-260215-47a822d7 \
  --member=allUsers \
  --role=roles/storage.legacyBucketReader
```

### Public Access Verification

```bash
# Anonymous object read
curl -sS -o /dev/null -w '%{http_code}\n' \
  'https://storage.googleapis.com/spylake-478002-260215-47a822d7/lake/raw/source=databento/product_type=equity_mbo/symbol=QQQ/table=market_by_order_dbn/symbology.json'

# Anonymous bucket object listing
curl -sS -o /dev/null -w '%{http_code}\n' \
  'https://storage.googleapis.com/storage/v1/b/spylake-478002-260215-47a822d7/o?prefix=lake/&maxResults=5'
```

### Command Used

```bash
gsutil -m -q \
  -o GSUtil:parallel_process_count=8 \
  -o GSUtil:parallel_thread_count=32 \
  -o GSUtil:parallel_composite_upload_threshold=150M \
  rsync -r "/Users/logan.robbins/research/spymaster/backend/lake" \
  "gs://spylake-478002-260215-47a822d7/lake"
```

### Verify Command

```bash
echo "LOCAL_FILES=$(find lake -type f | wc -l | tr -d ' ')"
echo "LOCAL_BYTES=$(find lake -type f -exec stat -f %z {} + | awk '{s+=$1} END {printf(\"%.0f\", s)}')"

echo "REMOTE_FILES=$(gcloud storage ls -r 'gs://spylake-478002-260215-47a822d7/lake/**' | wc -l | tr -d ' ')"
echo "REMOTE_BYTES=$(gcloud storage du -s 'gs://spylake-478002-260215-47a822d7/lake' | awk '{print $1}')"
```

## Transfer Debugging

- Transfer logs from this run are under `logs/`:
- `logs/gsutil_rsync_*.log`
- `logs/gsutil_rsync_fast_*.log`
- `logs/gsutil_rsync_quiet*.log`
- Listing command:

```bash
gcloud storage ls -r 'gs://spylake-478002-260215-47a822d7/lake/**' | sort
```
