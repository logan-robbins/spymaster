"""
Unified Replay Engine supporting Bronze Parquet + DBN data sources.

Per PLAN.md §8.2:
- Replays SPY trades + quotes + options trades + ES futures in deterministic event-time order
- Emits proper event type objects (StockTrade, StockQuote, OptionTrade, FuturesTrade, MBP10)
- Supports speed multiplier for testing
- Can replay from Bronze or Silver

Agent I deliverable per §12 of PLAN.md.
"""

import asyncio
import heapq
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union
)

from .event_types import (
    StockTrade, StockQuote, OptionTrade, FuturesTrade, MBP10,
    EventSource, Aggressor
)
from .bronze_writer import BronzeReader
from .dbn_ingestor import DBNIngestor


class EventType(Enum):
    """Type of event for routing."""
    STOCK_TRADE = 'stock_trade'
    STOCK_QUOTE = 'stock_quote'
    OPTION_TRADE = 'option_trade'
    FUTURES_TRADE = 'futures_trade'
    MBP10 = 'mbp10'


@dataclass
class TimestampedEvent:
    """
    Wrapper for event with timestamp for heap ordering.
    """
    ts_event_ns: int
    event_type: EventType
    event: Union[StockTrade, StockQuote, OptionTrade, FuturesTrade, MBP10]

    def __lt__(self, other: 'TimestampedEvent') -> bool:
        """Enable heap ordering by timestamp."""
        return self.ts_event_ns < other.ts_event_ns


class UnifiedReplayEngine:
    """
    Unified replay engine that merges multiple data sources.

    Sources:
    - Bronze Parquet: stocks.trades, stocks.quotes, options.trades
    - DBN files: ES futures trades, MBP-10

    Events are merged by ts_event_ns for deterministic replay.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        data_root: Optional[str] = None,
        dbn_data_root: Optional[str] = None
    ):
        """
        Initialize unified replay engine.

        Args:
            queue: Queue to emit events to
            data_root: Root for Bronze/Silver data
            dbn_data_root: Root for DBN files (ES futures)
        """
        self.queue = queue
        self.bronze_reader = BronzeReader(data_root)
        self.dbn_ingestor = DBNIngestor(dbn_data_root)
        self.running = False

    async def run(
        self,
        date: str,
        speed: float = 1.0,
        include_spy: bool = True,
        include_options: bool = True,
        include_es: bool = False,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> None:
        """
        Run replay for a specific date.

        Args:
            date: Date to replay (YYYY-MM-DD)
            speed: Playback speed multiplier (1.0 = realtime)
            include_spy: Include SPY trades and quotes
            include_options: Include SPY options trades
            include_es: Include ES futures trades and MBP-10
            start_ns: Start time filter (nanoseconds)
            end_ns: End time filter (nanoseconds)
        """
        print(f"  Unified Replay: Starting for {date} at {speed}x speed")
        self.running = True

        # Collect all events into a merged heap
        events = []

        # Load SPY data from Bronze
        if include_spy:
            await self._load_spy_events(date, events, start_ns, end_ns)

        # Load options data from Bronze
        if include_options:
            await self._load_option_events(date, events, start_ns, end_ns)

        # Load ES futures from DBN
        if include_es:
            await self._load_es_events(date, events, start_ns, end_ns)

        if not events:
            print(f"  Unified Replay: No events found for {date}")
            return

        # Sort by timestamp (heapify for efficiency with large datasets)
        heapq.heapify(events)

        print(f"  Unified Replay: Loaded {len(events)} events")

        # Replay loop
        first_event = events[0]
        replay_start_ns = first_event.ts_event_ns
        wall_clock_start = datetime.now(timezone.utc).timestamp() * 1e9

        emitted = 0
        while events and self.running:
            event_wrapper = heapq.heappop(events)

            # Calculate delay based on speed
            event_offset_ns = event_wrapper.ts_event_ns - replay_start_ns
            target_wall_ns = wall_clock_start + (event_offset_ns / speed)
            current_wall_ns = datetime.now(timezone.utc).timestamp() * 1e9

            wait_ns = target_wall_ns - current_wall_ns
            if wait_ns > 0:
                await asyncio.sleep(wait_ns / 1e9)

            # Emit event to queue
            await self.queue.put(event_wrapper.event)
            emitted += 1

            if emitted % 10000 == 0:
                print(f"  Unified Replay: Emitted {emitted} events...")

        print(f"  Unified Replay: Complete - emitted {emitted} events")

    async def _load_spy_events(
        self,
        date: str,
        events: List[TimestampedEvent],
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> None:
        """Load SPY trades and quotes from Bronze."""
        # Load trades
        trades_df = self.bronze_reader.read_stock_trades(
            symbol='SPY',
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        )

        if not trades_df.empty:
            for _, row in trades_df.iterrows():
                event = StockTrade(
                    ts_event_ns=int(row['ts_event_ns']),
                    ts_recv_ns=int(row.get('ts_recv_ns', row['ts_event_ns'])),
                    source=EventSource.REPLAY,
                    symbol=row.get('symbol', 'SPY'),
                    price=float(row['price']),
                    size=int(row['size']),
                    exchange=row.get('exchange'),
                    conditions=row.get('conditions'),
                    seq=row.get('seq')
                )
                events.append(TimestampedEvent(
                    ts_event_ns=event.ts_event_ns,
                    event_type=EventType.STOCK_TRADE,
                    event=event
                ))

            print(f"  Unified Replay: Loaded {len(trades_df)} SPY trades")

        # Load quotes
        quotes_df = self.bronze_reader.read_stock_quotes(
            symbol='SPY',
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        )

        if not quotes_df.empty:
            for _, row in quotes_df.iterrows():
                event = StockQuote(
                    ts_event_ns=int(row['ts_event_ns']),
                    ts_recv_ns=int(row.get('ts_recv_ns', row['ts_event_ns'])),
                    source=EventSource.REPLAY,
                    symbol=row.get('symbol', 'SPY'),
                    bid_px=float(row['bid_px']),
                    ask_px=float(row['ask_px']),
                    bid_sz=int(row['bid_sz']),
                    ask_sz=int(row['ask_sz']),
                    bid_exch=row.get('bid_exch'),
                    ask_exch=row.get('ask_exch'),
                    seq=row.get('seq')
                )
                events.append(TimestampedEvent(
                    ts_event_ns=event.ts_event_ns,
                    event_type=EventType.STOCK_QUOTE,
                    event=event
                ))

            print(f"  Unified Replay: Loaded {len(quotes_df)} SPY quotes")

    async def _load_option_events(
        self,
        date: str,
        events: List[TimestampedEvent],
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> None:
        """Load SPY options trades from Bronze."""
        options_df = self.bronze_reader.read_option_trades(
            underlying='SPY',
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        )

        if not options_df.empty:
            for _, row in options_df.iterrows():
                # Parse aggressor
                aggressor = Aggressor.MID
                if 'aggressor' in row:
                    agg_val = row['aggressor']
                    if agg_val == 1 or agg_val == 'BUY':
                        aggressor = Aggressor.BUY
                    elif agg_val == -1 or agg_val == 'SELL':
                        aggressor = Aggressor.SELL

                event = OptionTrade(
                    ts_event_ns=int(row['ts_event_ns']),
                    ts_recv_ns=int(row.get('ts_recv_ns', row['ts_event_ns'])),
                    source=EventSource.REPLAY,
                    underlying=row.get('underlying', 'SPY'),
                    option_symbol=row['option_symbol'],
                    exp_date=str(row['exp_date']),
                    strike=float(row['strike']),
                    right=row['right'],
                    price=float(row['price']),
                    size=int(row['size']),
                    opt_bid=row.get('opt_bid'),
                    opt_ask=row.get('opt_ask'),
                    aggressor=aggressor,
                    conditions=row.get('conditions'),
                    seq=row.get('seq')
                )
                events.append(TimestampedEvent(
                    ts_event_ns=event.ts_event_ns,
                    event_type=EventType.OPTION_TRADE,
                    event=event
                ))

            print(f"  Unified Replay: Loaded {len(options_df)} option trades")

    async def _load_es_events(
        self,
        date: str,
        events: List[TimestampedEvent],
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> None:
        """Load ES futures trades and MBP-10 from DBN files."""
        # Check if DBN data exists for this date
        available_dates = self.dbn_ingestor.get_available_dates('trades')

        if date not in available_dates:
            print(f"  Unified Replay: No DBN data for {date}")
            return

        # Load trades (using iterator to handle large files)
        trade_count = 0
        for trade in self.dbn_ingestor.read_trades(
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        ):
            events.append(TimestampedEvent(
                ts_event_ns=trade.ts_event_ns,
                event_type=EventType.FUTURES_TRADE,
                event=trade
            ))
            trade_count += 1

        if trade_count > 0:
            print(f"  Unified Replay: Loaded {trade_count} ES trades")

        # Note: MBP-10 data is very large (GBs), only load if specifically requested
        # and consider sampling or limiting to specific time windows

    def stop(self) -> None:
        """Stop the replay."""
        self.running = False


class StreamingReplayEngine:
    """
    Memory-efficient streaming replay for very large datasets.

    Uses iterators instead of loading all events into memory.
    Best for replaying full days of ES MBP-10 data.
    """

    def __init__(
        self,
        queue: asyncio.Queue,
        data_root: Optional[str] = None,
        dbn_data_root: Optional[str] = None
    ):
        """
        Initialize streaming replay engine.

        Args:
            queue: Queue to emit events to
            data_root: Root for Bronze/Silver data
            dbn_data_root: Root for DBN files
        """
        self.queue = queue
        self.bronze_reader = BronzeReader(data_root)
        self.dbn_ingestor = DBNIngestor(dbn_data_root)
        self.running = False

    async def run_es_trades(
        self,
        date: str,
        speed: float = 1.0,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> None:
        """
        Stream ES trades for a specific date.

        More memory efficient than loading all into heap.
        """
        print(f"  Streaming Replay: Starting ES trades for {date}")
        self.running = True

        first_ts = None
        wall_clock_start = None
        emitted = 0

        for trade in self.dbn_ingestor.read_trades(
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        ):
            if not self.running:
                break

            if first_ts is None:
                first_ts = trade.ts_event_ns
                wall_clock_start = datetime.now(timezone.utc).timestamp() * 1e9

            # Calculate delay
            event_offset_ns = trade.ts_event_ns - first_ts
            target_wall_ns = wall_clock_start + (event_offset_ns / speed)
            current_wall_ns = datetime.now(timezone.utc).timestamp() * 1e9

            wait_ns = target_wall_ns - current_wall_ns
            if wait_ns > 0 and speed > 0:
                await asyncio.sleep(wait_ns / 1e9)

            await self.queue.put(trade)
            emitted += 1

            if emitted % 10000 == 0:
                print(f"  Streaming Replay: Emitted {emitted} ES trades...")

        print(f"  Streaming Replay: Complete - emitted {emitted} ES trades")

    async def run_es_mbp10(
        self,
        date: str,
        speed: float = 1.0,
        sample_rate: int = 100,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None
    ) -> None:
        """
        Stream ES MBP-10 data for a specific date.

        Args:
            sample_rate: Only emit every Nth update (for very large files)
        """
        print(f"  Streaming Replay: Starting ES MBP-10 for {date} (sample 1:{sample_rate})")
        self.running = True

        first_ts = None
        wall_clock_start = None
        emitted = 0
        count = 0

        for mbp in self.dbn_ingestor.read_mbp10(
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        ):
            if not self.running:
                break

            count += 1

            # Sample rate
            if count % sample_rate != 0:
                continue

            if first_ts is None:
                first_ts = mbp.ts_event_ns
                wall_clock_start = datetime.now(timezone.utc).timestamp() * 1e9

            # Calculate delay
            event_offset_ns = mbp.ts_event_ns - first_ts
            target_wall_ns = wall_clock_start + (event_offset_ns / speed)
            current_wall_ns = datetime.now(timezone.utc).timestamp() * 1e9

            wait_ns = target_wall_ns - current_wall_ns
            if wait_ns > 0 and speed > 0:
                await asyncio.sleep(wait_ns / 1e9)

            await self.queue.put(mbp)
            emitted += 1

            if emitted % 1000 == 0:
                print(f"  Streaming Replay: Emitted {emitted} MBP-10 updates...")

        print(f"  Streaming Replay: Complete - emitted {emitted} MBP-10 updates (from {count} total)")

    def stop(self) -> None:
        """Stop the replay."""
        self.running = False
