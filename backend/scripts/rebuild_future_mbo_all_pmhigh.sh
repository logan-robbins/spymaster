#!/usr/bin/env bash
set -euo pipefail

DATE_RANGE="2025-10-01:2026-01-09"
SELECTION_PATH="lake/selection/mbo_contract_day_selection.parquet"
INDEX_DIR="lake/indexes/mbo_pm_high"

rm -rf lake/silver/product_type=future_mbo lake/gold/product_type=future_mbo lake/indexes

uv run python -m src.data_eng.retrieval.mbo_contract_day_selector \
  --dates "${DATE_RANGE}" \
  --output-path "${SELECTION_PATH}"

BRONZE_SYMBOLS=$(ls -d lake/bronze/source=databento/product_type=future_mbo/symbol=* | sed 's/.*symbol=//')

for symbol in ${BRONZE_SYMBOLS}; do
  uv run python -m src.data_eng.runner \
    --product-type future_mbo \
    --layer silver \
    --symbol "${symbol}" \
    --dates "${DATE_RANGE}" \
    --workers 8
done

SYMBOL_DATES=$(uv run python - <<'PY'
from pathlib import Path

from src.data_eng.retrieval.mbo_contract_day_selector import load_selection

selection = load_selection(Path("lake/selection/mbo_contract_day_selection.parquet"))
selection = selection.loc[selection["selected_symbol"] != ""]
if len(selection) == 0:
    raise ValueError("No included sessions in selection map")
groups = selection.groupby("selected_symbol")["session_date"].apply(list)
for symbol, dates in groups.items():
    print(f"{symbol}|{','.join(dates)}")
PY
)

MBO_SELECTION_PATH="${SELECTION_PATH}" LEVEL_ID=pm_high uv run python - <<'PY'
from pathlib import Path

from src.data_eng.config import load_config
from src.data_eng.retrieval.mbo_contract_day_selector import load_selection
from src.data_eng.stages.gold.future_mbo.build_trigger_vectors import GoldBuildMboTriggerVectors

repo_root = Path.cwd()
cfg = load_config(repo_root=repo_root, config_path=repo_root / "src/data_eng/config/datasets.yaml")
selection = load_selection(Path("lake/selection/mbo_contract_day_selection.parquet"))
selection = selection.loc[selection["selected_symbol"] != ""]

stage = GoldBuildMboTriggerVectors()
for row in selection.itertuples(index=False):
    symbol = str(getattr(row, "selected_symbol"))
    session_date = str(getattr(row, "session_date"))
    stage.run(cfg=cfg, repo_root=repo_root, symbol=symbol, dt=session_date)
    print(f"vectors {symbol} {session_date}")
PY

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

uv run python -m src.data_eng.retrieval.index_builder \
  --dates "${DATE_RANGE}" \
  --selection-path "${SELECTION_PATH}" \
  --output-dir "${INDEX_DIR}" \
  --norm-stats-path "${INDEX_DIR}/norm_stats_seed.json"

while IFS= read -r line; do
  symbol="${line%%|*}"
  dates="${line#*|}"
  if [ -z "${symbol}" ] || [ -z "${dates}" ]; then
    echo "Missing symbol or dates in selection map" >&2
    exit 1
  fi
  LEVEL_ID=pm_high MBO_SELECTION_PATH="${SELECTION_PATH}" MBO_INDEX_DIR="${INDEX_DIR}" uv run python -m src.data_eng.runner \
    --product-type future_mbo \
    --layer gold \
    --symbol "${symbol}" \
    --dates "${dates}" \
    --workers 8
done <<< "${SYMBOL_DATES}"
