#!/usr/bin/env bash
set -euo pipefail

SELECTION_PATH="lake/selection/mbo_contract_day_selection.parquet"
INDEX_DIR="lake/indexes/mbo_pm_high"
WORKERS=8
LEVEL_ID=pm_high
export LEVEL_ID
TS_FORMAT="+%Y-%m-%d %H:%M:%S"

if [ ! -f "${SELECTION_PATH}" ]; then
  echo "Missing selection map: ${SELECTION_PATH}" >&2
  exit 1
fi

DATE_RANGE=$(uv run python - <<'PY'
import pandas as pd

df = pd.read_parquet("lake/selection/mbo_contract_day_selection.parquet")
df = df.loc[df["selected_symbol"] != ""]
if df.empty:
    raise ValueError("No included sessions in selection map")
dates = df["session_date"].astype(str)
print(f"{dates.min()}:{dates.max()}")
PY
)

echo "[$(date "${TS_FORMAT}")] CLEAR silver/gold/indexes"
rm -rf lake/silver/product_type=future_mbo
rm -rf lake/gold/product_type=future_mbo
rm -rf "${INDEX_DIR}"

echo "[$(date "${TS_FORMAT}")] START silver rebuild"
uv run python -m src.data_eng.runner \
  --product-type future_mbo \
  --layer silver \
  --symbol ES \
  --dates "${DATE_RANGE}" \
  --workers "${WORKERS}" \
  --overwrite
echo "[$(date "${TS_FORMAT}")] DONE silver rebuild"

echo "[$(date "${TS_FORMAT}")] START trigger vectors rebuild"
uv run python - <<'PY'
from pathlib import Path
import shutil

import pandas as pd

from src.data_eng.config import load_config
from src.data_eng.io import partition_ref
from src.data_eng.retrieval.mbo_contract_day_selector import load_selection
from src.data_eng.stages.gold.future_mbo.build_trigger_vectors import GoldBuildMboTriggerVectors

repo_root = Path.cwd()
cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
selection = load_selection(Path("lake/selection/mbo_contract_day_selection.parquet"))
selection = selection.loc[selection["selected_symbol"] != ""]
if len(selection) == 0:
    raise ValueError("No included sessions in selection map")

stage = GoldBuildMboTriggerVectors()
feature_key = "gold.future_mbo.mbo_trigger_vector_features"

for row in selection.itertuples(index=False):
    symbol = str(getattr(row, "selected_symbol"))
    session_date = str(getattr(row, "session_date"))
    out_ref = partition_ref(cfg, stage.io.output, symbol, session_date)
    if out_ref.dir.exists():
        shutil.rmtree(out_ref.dir)
    feature_ref = partition_ref(cfg, feature_key, symbol, session_date)
    if feature_ref.dir.exists():
        shutil.rmtree(feature_ref.dir)
    stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=session_date)
    print(f"vectors {symbol} {session_date}")
PY
echo "[$(date "${TS_FORMAT}")] DONE trigger vectors rebuild"

echo "[$(date "${TS_FORMAT}")] START seed stats rebuild"
uv run python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.data_eng.config import load_config
from src.data_eng.contracts import load_avro_contract, enforce_contract
from src.data_eng.io import partition_ref, is_partition_complete, read_partition
from src.data_eng.retrieval.normalization import fit_robust_stats

repo_root = Path.cwd()
cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
selection = pd.read_parquet("lake/selection/mbo_contract_day_selection.parquet")
selection = selection.loc[selection["selected_symbol"] != ""]
if selection.empty:
    raise ValueError("No included sessions in selection map")

key = "gold.future_mbo.mbo_trigger_vectors"
contract = load_avro_contract(repo_root / cfg.dataset(key).contract)

vectors = []
for row in selection.itertuples(index=False):
    session_date = str(getattr(row, "session_date"))
    symbol = str(getattr(row, "selected_symbol"))
    ref = partition_ref(cfg, key, symbol, session_date)
    if not is_partition_complete(ref):
        raise FileNotFoundError(f"Missing vectors {symbol} {session_date}")
    df = read_partition(ref)
    if len(df) == 0:
        continue
    df = enforce_contract(df, contract)
    vectors.append(np.array(df["vector"].tolist(), dtype=np.float64))

if not vectors:
    raise ValueError("No vectors for stats")

stacked = np.vstack(vectors)
stats = fit_robust_stats(stacked)
out_dir = Path("lake/indexes/mbo_pm_high")
out_dir.mkdir(parents=True, exist_ok=True)
seed_path = out_dir / "norm_stats_seed.json"
seed_path.write_text(json.dumps({
    "median": stats.median.tolist(),
    "mad": stats.mad.tolist(),
    "vector_dim": int(stats.median.shape[0]),
}))
print(seed_path)
PY
echo "[$(date "${TS_FORMAT}")] DONE seed stats rebuild"

echo "[$(date "${TS_FORMAT}")] START index rebuild"
uv run python -m src.data_eng.retrieval.index_builder \
  --dates "${DATE_RANGE}" \
  --output-dir "${INDEX_DIR}" \
  --norm-stats-path "${INDEX_DIR}/norm_stats_seed.json" \
  --overwrite
echo "[$(date "${TS_FORMAT}")] DONE index rebuild"

echo "[$(date "${TS_FORMAT}")] START gold signals/pressure rebuild"
MBO_INDEX_DIR="${INDEX_DIR}" uv run python -m src.data_eng.runner \
  --product-type future_mbo \
  --layer gold \
  --symbol ES \
  --dates "${DATE_RANGE}" \
  --workers "${WORKERS}"
echo "[$(date "${TS_FORMAT}")] DONE gold signals/pressure rebuild"
echo "[$(date "${TS_FORMAT}")] DONE full rebuild"
