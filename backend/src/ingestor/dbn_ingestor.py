"""
Databento DBN file ingestor for ES futures data.

Reads DBN files from dbn-data/ directory and converts to our event types:
- trades schema -> FuturesTrade events
- mbp-10 schema -> MBP10 events

Per PLAN.md §11.3 (Databento GLBX.MDP3 contract).

Agent I deliverable per §12 of PLAN.md.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any, Tuple
import re
import json

import databento as db
from databento_dbn import TradeMsg, MBP10Msg, SType

from src.common.event_types import (
    FuturesTrade, MBP10, BidAskLevel, EventSource, Aggressor
)


def _aggressor_from_side(side: str) -> Aggressor:
    """
    Convert Databento side to Aggressor enum.

    Per PLAN.md §11.3:
    - 'A' (ask=sell aggressor) -> SELL
    - 'B' (bid=buy aggressor) -> BUY
    - 'N' (none) -> MID
    """
    if side == 'A':
        return Aggressor.SELL
    elif side == 'B':
        return Aggressor.BUY
    return Aggressor.MID


@dataclass
class DBNFileInfo:
    """Metadata about a DBN file."""
    path: str
    date: str  # YYYY-MM-DD
    schema: str  # 'trades' or 'mbp-10'
    dataset: str  # e.g., 'glbx-mdp3'


class DBNIngestor:
    """
    Ingestor for Databento DBN files.

    Reads DBN files from a directory and yields normalized events.
    Supports streaming iteration to handle large files efficiently.
    """

    def __init__(self, dbn_data_root: Optional[str] = None):
        """
        Initialize DBN ingestor.

        Args:
            dbn_data_root: Root directory containing DBN files
                          (defaults to dbn-data/ in project root)
        """
        if dbn_data_root:
            self.dbn_root = Path(dbn_data_root)
        else:
            # Default to dbn-data/ in project root
            # backend/src/ingestor/dbn_ingestor.py -> project_root/dbn-data
            self.dbn_root = Path(__file__).parent.parent.parent.parent / 'dbn-data'

        self._symbology_cache: Dict[str, Dict[int, str]] = {}

    def discover_files(self, schema: Optional[str] = None) -> List[DBNFileInfo]:
        """
        Discover available DBN files.

        Args:
            schema: Filter by schema ('trades' or 'mbp-10'), or None for all

        Returns:
            List of DBNFileInfo for available files
        """
        files = []

        # Check both schema directories
        schemas_to_check = ['trades', 'MBP-10'] if schema is None else [schema]

        for schema_dir in schemas_to_check:
            schema_path = self.dbn_root / schema_dir
            if not schema_path.exists():
                continue

            for file_path in schema_path.glob('*.dbn'):
                # Parse filename: glbx-mdp3-20251216.trades.dbn
                parts = file_path.stem.split('.')
                if len(parts) >= 2:
                    dataset_date = parts[0]  # glbx-mdp3-20251216
                    file_schema = parts[1]   # trades or mbp-10

                    # Extract date from dataset-date string
                    date_str = dataset_date.split('-')[-1]  # 20251216
                    if len(date_str) == 8:
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

                        files.append(DBNFileInfo(
                            path=str(file_path),
                            date=formatted_date,
                            schema=file_schema,
                            dataset='-'.join(dataset_date.split('-')[:-1])
                        ))

        return sorted(files, key=lambda f: (f.date, f.schema))

    def get_available_dates(self, schema: str = 'trades') -> List[str]:
        """Get list of available dates for a schema."""
        files = self.discover_files(schema)
        return sorted(set(f.date for f in files))

    def _load_symbology(self, schema_dir: str) -> Dict[int, str]:
        """Load instrument ID to symbol mapping."""
        if schema_dir in self._symbology_cache:
            return self._symbology_cache[schema_dir]

        symbology_path = self.dbn_root / schema_dir / 'symbology.json'
        if symbology_path.exists():
            with open(symbology_path) as f:
                data = json.load(f)
                # Map instrument_id -> symbol
                mapping = {}
                for symbol, entries in data.get('result', {}).items():
                    for entry in entries:
                        # Databento symbology uses "s" as the instrument id (string).
                        iid = entry.get('i', entry.get('s'))
                        if iid is None:
                            continue
                        try:
                            mapping[int(iid)] = symbol
                        except Exception:
                            continue
                self._symbology_cache[schema_dir] = mapping
                return mapping

        return {}

    @staticmethod
    def _is_outright_symbol(symbol: str, prefix: str) -> bool:
        """
        Return True if symbol is an outright contract for the given prefix.

        Example:
          prefix="ES" accepts: ESZ5, ESH6, ESU6
          rejects spreads like: ESZ6-ESH7
        """
        if not symbol or not symbol.startswith(prefix):
            return False
        if "-" in symbol:
            return False
        # Typical CME futures contract code is: PREFIX + MonthLetter + YearDigit
        # Keep this strict to avoid pulling in spread/strategy instruments that
        # can have very different price scales (e.g. ~58.2).
        pattern = re.compile(rf"^{re.escape(prefix)}[A-Z]\d$")
        return bool(pattern.match(symbol))

    def read_trades(
        self,
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        symbol_prefix: Optional[str] = None
    ) -> Iterator[FuturesTrade]:
        """
        Read trades from DBN files.

        Args:
            date: Specific date to read (YYYY-MM-DD), or None for all
            start_ns: Filter by ts_event_ns >= start_ns
            end_ns: Filter by ts_event_ns <= end_ns

        Yields:
            FuturesTrade events
        """
        files = self.discover_files('trades')
        if date:
            files = [f for f in files if f.date == date]

        symbology = self._load_symbology('trades')

        for file_info in files:
            print(f"  DBN: Reading trades from {file_info.path}")

            try:
                store = db.DBNStore.from_file(file_info.path)

                for record in store:
                    if not isinstance(record, TradeMsg):
                        continue

                    ts_event_ns = record.ts_event

                    # Apply time filters
                    if start_ns and ts_event_ns < start_ns:
                        continue
                    if end_ns and ts_event_ns > end_ns:
                        continue

                    # Get symbol from instrument ID
                    symbol = symbology.get(record.instrument_id, f'ES_{record.instrument_id}')

                    # Filter to the intended futures contract (avoid spreads / misc instruments)
                    if symbol_prefix and not self._is_outright_symbol(symbol, symbol_prefix):
                        continue

                    yield FuturesTrade(
                        ts_event_ns=ts_event_ns,
                        ts_recv_ns=record.ts_recv if hasattr(record, 'ts_recv') else ts_event_ns,
                        source=EventSource.DIRECT_FEED,
                        symbol=symbol,
                        price=record.price / 1e9,  # Fixed-point to float
                        size=record.size,
                        aggressor=_aggressor_from_side(record.side),
                        exchange='CME',
                        conditions=None,
                        seq=record.sequence if hasattr(record, 'sequence') else None
                    )

            except Exception as e:
                print(f"  DBN ERROR reading {file_info.path}: {e}")

    def read_mbp10(
        self,
        date: Optional[str] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        symbol_prefix: Optional[str] = None
    ) -> Iterator[MBP10]:
        """
        Read MBP-10 data from DBN files.

        Args:
            date: Specific date to read (YYYY-MM-DD), or None for all
            start_ns: Filter by ts_event_ns >= start_ns
            end_ns: Filter by ts_event_ns <= end_ns

        Yields:
            MBP10 events
        """
        # Handle the directory name difference (MBP-10 vs mbp-10)
        files = self.discover_files('MBP-10') or self.discover_files('mbp-10')
        if date:
            files = [f for f in files if f.date == date]

        symbology = self._load_symbology('MBP-10')

        for file_info in files:
            print(f"  DBN: Reading MBP-10 from {file_info.path}")

            try:
                store = db.DBNStore.from_file(file_info.path)

                for record in store:
                    if not isinstance(record, MBP10Msg):
                        continue

                    ts_event_ns = record.ts_event

                    # Apply time filters
                    if start_ns and ts_event_ns < start_ns:
                        continue
                    if end_ns and ts_event_ns > end_ns:
                        continue

                    # Get symbol from instrument ID
                    symbol = symbology.get(record.instrument_id, f'ES_{record.instrument_id}')

                    # Filter to the intended futures contract (avoid spreads / misc instruments)
                    if symbol_prefix and not self._is_outright_symbol(symbol, symbol_prefix):
                        continue

                    # Convert levels
                    levels = []
                    for level in record.levels[:10]:  # Top 10 levels
                        levels.append(BidAskLevel(
                            bid_px=level.bid_px / 1e9,  # Fixed-point to float
                            bid_sz=level.bid_sz,
                            ask_px=level.ask_px / 1e9,
                            ask_sz=level.ask_sz
                        ))

                    yield MBP10(
                        ts_event_ns=ts_event_ns,
                        ts_recv_ns=record.ts_recv if hasattr(record, 'ts_recv') else ts_event_ns,
                        source=EventSource.DIRECT_FEED,
                        symbol=symbol,
                        levels=levels,
                        is_snapshot=False,  # Databento sends all as updates
                        seq=record.sequence if hasattr(record, 'sequence') else None
                    )

            except Exception as e:
                print(f"  DBN ERROR reading {file_info.path}: {e}")

    def get_time_range(self, date: str, schema: str = 'trades') -> Tuple[Optional[int], Optional[int]]:
        """
        Get the time range (min/max ts_event_ns) for a specific date.

        Returns:
            Tuple of (start_ns, end_ns) or (None, None) if no data
        """
        files = self.discover_files(schema)
        files = [f for f in files if f.date == date]

        if not files:
            return None, None

        min_ts = None
        max_ts = None

        for file_info in files:
            try:
                store = db.DBNStore.from_file(file_info.path)

                # Just sample first and last records
                first_record = None
                last_record = None

                for record in store:
                    if first_record is None:
                        first_record = record
                    last_record = record

                if first_record:
                    ts = first_record.ts_event
                    if min_ts is None or ts < min_ts:
                        min_ts = ts

                if last_record:
                    ts = last_record.ts_event
                    if max_ts is None or ts > max_ts:
                        max_ts = ts

            except Exception as e:
                print(f"  DBN ERROR getting time range from {file_info.path}: {e}")

        return min_ts, max_ts
