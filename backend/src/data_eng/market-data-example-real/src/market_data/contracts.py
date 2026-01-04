from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class AvroContract:
    """A minimal Avro contract representation (field order matters)."""

    name: str
    fields: List[str]


def load_avro_contract(contract_path: Path) -> AvroContract:
    obj = json.loads(contract_path.read_text())
    fields = [f["name"] for f in obj["fields"]]
    return AvroContract(name=obj.get("name", ""), fields=fields)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def contract_hash(contract_path: Path) -> str:
    """Hash the exact bytes of the contract file (used in manifests)."""
    return sha256_text(contract_path.read_text())


def enforce_contract(df: pd.DataFrame, contract: AvroContract) -> pd.DataFrame:
    """Validate + reorder columns to match the contract."""
    missing = [c for c in contract.fields if c not in df.columns]
    extra = [c for c in df.columns if c not in contract.fields]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected extra columns not in contract: {extra}")

    return df.loc[:, contract.fields]
