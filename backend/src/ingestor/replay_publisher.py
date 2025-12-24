"""
Replay Publisher for DBN Files (+ Bronze Options)

Reads Databento DBN files and optional Bronze option trades, then publishes to NATS at configurable speed.
This allows "Replay Mode" to just be a NATS publisher, so other services
don't know the difference between live and replay.

NATS Subjects Published:
- market.futures.trades (FuturesTrade)
- market.futures.mbp10 (MBP10)
- market.options.trades (OptionTrade, optional)

Per AGENT A tasks in NEXT.md.
"""

import asyncio
import time
from typing import Iterator, Optional, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from src.ingestor.dbn_ingestor import DBNIngestor
from src.lake.bronze_writer import BronzeReader
from src.common.bus import NATSBus
from src.common.event_types import FuturesTrade, MBP10, OptionTrade, EventSource, Aggressor


@dataclass
class ReplayStats:
    """Statistics for replay session."""
    events_published: int = 0
    trades_published: int = 0
    mbp10_published: int = 0
    options_published: int = 0
    start_time: Optional[float] = None
    first_event_ts: Optional[int] = None
    last_event_ts: Optional[int] = None
    
    def elapsed_wall_time(self) -> float:
        """Wall clock time elapsed."""
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def elapsed_event_time_sec(self) -> float:
        """Event time elapsed in seconds."""
        if self.first_event_ts and self.last_event_ts:
            return (self.last_event_ts - self.first_event_ts) / 1e9
        return 0.0
    
    def actual_speed(self) -> float:
        """Actual replay speed (event_time / wall_time)."""
        wall = self.elapsed_wall_time()
        event = self.elapsed_event_time_sec()
        if wall > 0 and event > 0:
            return event / wall
        return 0.0


class ReplayPublisher:
    """
    Publishes DBN file contents (and optional Bronze options) to NATS at configurable speed.
    
    Replay Speed:
    - 0.0 = as fast as possible (no delays)
    - 1.0 = realtime (1 second of event time = 1 second of wall time)
    - 2.0 = 2x speed
    - 0.5 = half speed
    """
    
    def __init__(
        self,
        bus: NATSBus,
        dbn_ingestor: DBNIngestor,
        replay_speed: float = 1.0,
        bronze_reader: Optional[BronzeReader] = None
    ):
        self.bus = bus
        self.dbn_ingestor = dbn_ingestor
        self.replay_speed = replay_speed
        self.bronze_reader = bronze_reader or BronzeReader()
        self.stats = ReplayStats()
    
    async def replay_date(
        self,
        date: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        include_trades: bool = True,
        include_mbp10: bool = True,
        include_options: bool = False
    ):
        """
        Replay all data for a specific date.
        
        Args:
            date: Date to replay (YYYY-MM-DD)
            start_ns: Optional start time filter (nanoseconds)
            end_ns: Optional end time filter (nanoseconds)
            include_trades: Whether to replay trades
            include_mbp10: Whether to replay MBP-10
            include_options: Whether to replay option trades from Bronze
        """
        print(f"🎬 Starting replay for {date}")
        print(f"   Speed: {self.replay_speed}x")
        print(f"   Trades: {include_trades}, MBP-10: {include_mbp10}, Options: {include_options}")
        
        self.stats = ReplayStats()
        self.stats.start_time = time.time()
        
        # Stream-merge both iterators by event time to avoid materializing whole day in memory.
        trade_iter = iter(self.dbn_ingestor.read_trades(date, start_ns, end_ns)) if include_trades else iter(())
        mbp_iter = iter(self.dbn_ingestor.read_mbp10(date, start_ns, end_ns)) if include_mbp10 else iter(())
        option_iter = self._iter_option_trades(date, start_ns, end_ns) if include_options else iter(())

        next_trade = next(trade_iter, None)
        next_mbp = next(mbp_iter, None)
        next_option = next(option_iter, None)

        if next_trade is None and next_mbp is None and next_option is None:
            print("  ⚠️  No events found for this date")
            return

        print("  🎯 Publishing to NATS (streaming merge)...")

        prev_event_ns = None

        while next_trade is not None or next_mbp is not None or next_option is not None:
            candidates: List[Tuple[int, int, str, Any]] = []
            if next_trade is not None:
                candidates.append((next_trade.ts_event_ns, 0, "trade", next_trade))
            if next_option is not None:
                candidates.append((next_option.ts_event_ns, 1, "option", next_option))
            if next_mbp is not None:
                candidates.append((next_mbp.ts_event_ns, 2, "mbp10", next_mbp))

            _, _, event_type, event = min(candidates, key=lambda x: (x[0], x[1]))
            if event_type == "trade":
                next_trade = next(trade_iter, None)
            elif event_type == "option":
                next_option = next(option_iter, None)
            else:
                next_mbp = next(mbp_iter, None)

            # Update stats
            if self.stats.first_event_ts is None:
                self.stats.first_event_ts = event.ts_event_ns
            self.stats.last_event_ts = event.ts_event_ns
            self.stats.events_published += 1
            
            # Apply replay speed delays
            if self.replay_speed > 0 and prev_event_ns is not None:
                event_delta_ns = event.ts_event_ns - prev_event_ns
                wall_delay_sec = (event_delta_ns / 1e9) / self.replay_speed
                
                # Cap maximum delay to avoid huge gaps
                if wall_delay_sec > 5.0:
                    wall_delay_sec = 5.0
                
                if wall_delay_sec > 0:
                    await asyncio.sleep(wall_delay_sec)
            
            # Publish to NATS
            if event_type == "trade":
                await self.bus.publish("market.futures.trades", event)
                self.stats.trades_published += 1
            elif event_type == "option":
                await self.bus.publish("market.options.trades", event)
                self.stats.options_published += 1
            elif event_type == "mbp10":
                await self.bus.publish("market.futures.mbp10", event)
                self.stats.mbp10_published += 1
            
            prev_event_ns = event.ts_event_ns
            
            # Progress update every 10k events
            if self.stats.events_published % 10000 == 0:
                speed = self.stats.actual_speed()
                print(f"    Progress: {self.stats.events_published:,} events "
                      f"({self.stats.trades_published:,} trades, "
                      f"{self.stats.options_published:,} options, "
                      f"{self.stats.mbp10_published:,} mbp10) @ {speed:.2f}x")
        
        # Final stats
        self._print_stats()
    
    async def replay_continuous(
        self,
        dates: Optional[List[str]] = None,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        include_options: bool = False
    ):
        """
        Replay multiple dates continuously.
        
        Args:
            dates: List of dates to replay, or None for all available
            start_ns: Optional start time filter
            end_ns: Optional end time filter
        """
        if dates is None:
            dates = self.dbn_ingestor.get_available_dates('trades')
        
        print(f"🎬 Continuous replay mode")
        print(f"   Dates: {len(dates)} days")
        print(f"   Range: {dates[0]} to {dates[-1]}")
        
        for date in dates:
            await self.replay_date(date, start_ns, end_ns, include_options=include_options)
            print(f"  ✅ Completed {date}\n")

    def _iter_option_trades(
        self,
        date: str,
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> Iterator[OptionTrade]:
        options_df = self.bronze_reader.read_option_trades(
            underlying="SPY",
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        )
        if options_df.empty:
            raise ValueError(f"No Bronze option trades found for {date}")

        for row in options_df.itertuples(index=False):
            aggressor = Aggressor.MID
            agg_val = getattr(row, "aggressor", 0)
            if agg_val in (1, "BUY", "buy"):
                aggressor = Aggressor.BUY
            elif agg_val in (-1, "SELL", "sell"):
                aggressor = Aggressor.SELL

            yield OptionTrade(
                ts_event_ns=int(row.ts_event_ns),
                ts_recv_ns=int(getattr(row, "ts_recv_ns", row.ts_event_ns)),
                source=EventSource.REPLAY,
                underlying=getattr(row, "underlying", "SPY"),
                option_symbol=row.option_symbol,
                exp_date=str(row.exp_date),
                strike=float(row.strike),
                right=row.right,
                price=float(row.price),
                size=int(row.size),
                opt_bid=getattr(row, "opt_bid", None),
                opt_ask=getattr(row, "opt_ask", None),
                aggressor=aggressor,
                conditions=getattr(row, "conditions", None),
                seq=getattr(row, "seq", None)
            )
    
    def _print_stats(self):
        """Print replay statistics."""
        wall = self.stats.elapsed_wall_time()
        event = self.stats.elapsed_event_time_sec()
        speed = self.stats.actual_speed()
        
        print("\n" + "=" * 60)
        print("📊 REPLAY STATISTICS")
        print("=" * 60)
        print(f"  Events Published: {self.stats.events_published:,}")
        print(f"    - Trades: {self.stats.trades_published:,}")
        print(f"    - Options: {self.stats.options_published:,}")
        print(f"    - MBP-10: {self.stats.mbp10_published:,}")
        print(f"  Wall Time: {wall:.2f}s")
        print(f"  Event Time: {event:.2f}s")
        print(f"  Actual Speed: {speed:.2f}x (target: {self.replay_speed}x)")
        
        if self.stats.events_published > 0:
            throughput = self.stats.events_published / wall if wall > 0 else 0
            print(f"  Throughput: {throughput:.0f} events/sec")
        
        print("=" * 60)


async def main():
    """
    Standalone entry point for replay publisher.
    
    Usage:
        export REPLAY_SPEED=1.0  # 1x realtime
        export REPLAY_DATE=2025-12-16
        export REPLAY_INCLUDE_OPTIONS=true
        uv run python -m src.ingestor.replay_publisher
    """
    import os
    import sys
    from src.common.config import CONFIG
    
    print("=" * 60)
    print("🎬 REPLAY PUBLISHER")
    print("=" * 60)
    
    # Get configuration
    replay_speed = CONFIG.REPLAY_SPEED
    replay_date = os.getenv("REPLAY_DATE")  # Single date or None for all
    include_options = os.getenv("REPLAY_INCLUDE_OPTIONS", "false").lower() == "true"
    print(f"   Options replay: {include_options}")
    
    # Initialize NATS
    bus = NATSBus(servers=[CONFIG.NATS_URL])
    await bus.connect()
    
    # Initialize DBN Ingestor
    dbn_ingestor = DBNIngestor()
    
    # Check available data
    available_dates = dbn_ingestor.get_available_dates('trades')
    if not available_dates:
        print("❌ No DBN data found in dbn-data/ directory")
        sys.exit(1)
    
    print(f"📁 Found {len(available_dates)} days of data:")
    for date in available_dates:
        print(f"   - {date}")
    
    # Initialize Replay Publisher
    publisher = ReplayPublisher(
        bus=bus,
        dbn_ingestor=dbn_ingestor,
        replay_speed=replay_speed
    )
    
    try:
        if replay_date:
            # Replay specific date
            if replay_date not in available_dates:
                print(f"❌ Date {replay_date} not found in available data")
                sys.exit(1)
            await publisher.replay_date(replay_date, include_options=include_options)
        else:
            # Replay all dates
            await publisher.replay_continuous(dates=available_dates, include_options=include_options)
        
        print("\n✅ Replay complete")
        
    except KeyboardInterrupt:
        print("\n⏹ Replay interrupted")
    except Exception as e:
        print(f"❌ Replay error: {e}")
        raise
    finally:
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
