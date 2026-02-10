from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import yaml

PRICE_SCALE = 1e-9


@dataclass(frozen=True)
class ProductConfig:
    """Per-product constants for pipeline stages."""

    root: str
    tick_size: float
    grid_max_ticks: int
    strike_step_points: float
    max_strike_offsets: int
    contract_multiplier: float

    @property
    def tick_int(self) -> int:
        return int(round(self.tick_size / PRICE_SCALE))

    @property
    def strike_step_int(self) -> int:
        return int(round(self.strike_step_points / PRICE_SCALE))

    @property
    def strike_ticks(self) -> int:
        return int(round(self.strike_step_points / self.tick_size))


@dataclass(frozen=True)
class DatasetSpec:
    """A logical dataset mapped to a physical location and a schema contract."""

    key: str
    path: str  # relative to lake_root
    format: str  # csv (demo) / parquet (prod)
    partition_keys: List[str]
    contract: str  # relative path to contract file


@dataclass(frozen=True)
class AppConfig:
    lake_root: Path
    datasets: Dict[str, DatasetSpec]
    products: Dict[str, ProductConfig]

    def dataset(self, key: str) -> DatasetSpec:
        try:
            return self.datasets[key]
        except KeyError as e:
            raise KeyError(f"Unknown dataset key: {key}") from e

    def product(self, root: str) -> ProductConfig:
        try:
            return self.products[root]
        except KeyError as e:
            raise KeyError(
                f"Unknown product root: {root}. "
                f"Known roots: {sorted(self.products.keys())}"
            ) from e

    def product_for_symbol(self, symbol: str) -> ProductConfig:
        root = extract_root(symbol, self.products.keys())
        return self.product(root)


def extract_root(symbol: str, known_roots: Iterable[str]) -> str:
    """Extract the product root from a contract symbol.

    Matches the longest known root that is a prefix of the symbol.
    Examples: ESH6 → ES, MNQH6 → MNQ, SIH6 → SI, 6EH6 → 6E
    """
    best = ""
    for root in known_roots:
        if symbol.startswith(root) and len(root) > len(best):
            best = root
    if not best:
        raise ValueError(
            f"Cannot extract product root from symbol '{symbol}'. "
            f"Known roots: {sorted(known_roots)}"
        )
    return best


def load_config(repo_root: Path, config_path: Path) -> AppConfig:
    """Load config/datasets.yaml and config/products.yaml."""

    raw = yaml.safe_load(config_path.read_text())
    lake_root = repo_root / raw["lake_root"]

    ds: Dict[str, DatasetSpec] = {}
    for key, spec in raw["datasets"].items():
        ds[key] = DatasetSpec(
            key=key,
            path=spec["path"],
            format=spec["format"],
            partition_keys=list(spec["partition_keys"]),
            contract=spec["contract"],
        )

    products_path = config_path.parent / "products.yaml"
    products: Dict[str, ProductConfig] = {}
    if products_path.exists():
        products_raw = yaml.safe_load(products_path.read_text())
        for root, spec in products_raw.get("products", {}).items():
            products[root] = ProductConfig(
                root=root,
                tick_size=float(spec["tick_size"]),
                grid_max_ticks=int(spec["grid_max_ticks"]),
                strike_step_points=float(spec["strike_step_points"]),
                max_strike_offsets=int(spec["max_strike_offsets"]),
                contract_multiplier=float(spec["contract_multiplier"]),
            )

    return AppConfig(lake_root=lake_root, datasets=ds, products=products)
