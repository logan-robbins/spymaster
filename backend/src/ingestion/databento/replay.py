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
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

from src.ingestion.databento.dbn_reader import DBNReader


# US Eastern timezone for market hours
ET = ZoneInfo("America/New_York")

# Pre-market starts at 4:00 AM ET
PREMARKET_START_HOUR = 4
PREMARKET_START_MINUTE = 0

# Regular Trading Hours (RTH) start at 9:30 AM ET
RTH_START_HOUR = 9
RTH_START_MINUTE = 30


def get_premarket_start_ns(date_str: str) -> int:
    """
    Get the start timestamp (nanoseconds) for pre-market on a given date.

    Pre-market starts at 4:00 AM ET.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        Unix nanoseconds for 4:00 AM ET on that date
    """
    year, month, day = map(int, date_str.split('-'))

    # Create datetime at 4:00 AM ET
    dt_et = datetime(year, month, day, PREMARKET_START_HOUR, PREMARKET_START_MINUTE, 0, tzinfo=ET)

    # Convert to Unix timestamp (seconds) then to nanoseconds
    return int(dt_et.timestamp() * 1_000_000_000)


def get_rth_start_ns(date_str: str) -> int:
    """
    Get the start timestamp (nanoseconds) for RTH on a given date.

    RTH (Regular Trading Hours) starts at 9:30 AM ET.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        Unix nanoseconds for 9:30 AM ET on that date
    """
    year, month, day = map(int, date_str.split('-'))

    # Create datetime at 9:30 AM ET
    dt_et = datetime(year, month, day, RTH_START_HOUR, RTH_START_MINUTE, 0, tzinfo=ET)

    # Convert to Unix timestamp (seconds) then to nanoseconds
    return int(dt_et.timestamp() * 1_000_000_000)
from src.io.bronze import BronzeReader
from src.common.bus import NATSBus
from src.common.event_types import FuturesTrade, MBP10, OptionTrade, EventSource, Aggressor, BidAskLevel


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
        dbn_ingestor: DBNReader,
        replay_speed: float = 1.0,
        bronze_reader: Optional[BronzeReader] = None,
        use_bronze_futures: bool = False,
        futures_symbol: str = "ES"
    ):
        self.bus = bus
        self.dbn_ingestor = dbn_ingestor
        self.replay_speed = replay_speed
        self.bronze_reader = bronze_reader or BronzeReader()
        self.use_bronze_futures = use_bronze_futures
        self.futures_symbol = futures_symbol
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
        if include_trades:
            if self.use_bronze_futures:
                trade_iter = iter(self._iter_futures_trades_bronze(date, start_ns, end_ns))
            else:
                trade_iter = iter(self.dbn_ingestor.read_trades(
                    date,
                    start_ns,
                    end_ns,
                    symbol_prefix=self.futures_symbol
                ))
        else:
            trade_iter = iter(())

        if include_mbp10:
            if self.use_bronze_futures:
                mbp_iter = iter(self._iter_futures_mbp10_bronze(date, start_ns, end_ns))
            else:
                mbp_iter = iter(self.dbn_ingestor.read_mbp10(
                    date,
                    start_ns,
                    end_ns,
                    symbol_prefix=self.futures_symbol
                ))
        else:
            mbp_iter = iter(())
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
        end_ns: Optional[int] = None,
        include_options: bool = True
    ):
        """
        Replay multiple dates continuously, each starting from RTH (9:30 AM ET).

        Args:
            dates: List of dates to replay, or None for all available
            end_ns: Optional end time filter
            include_options: Whether to include options (default True)
        """
        if dates is None:
            dates = self.dbn_ingestor.get_available_dates('trades')

        print(f"🎬 Continuous replay mode")
        print(f"   Dates: {len(dates)} days")
        print(f"   Range: {dates[0]} to {dates[-1]}")

        for date in dates:
            # Each date starts from RTH (9:30 AM ET)
            start_ns = get_rth_start_ns(date)
            await self.replay_date(date, start_ns, end_ns, include_options=include_options)
            print(f"  ✅ Completed {date}\n")

    def _iter_option_trades(
        self,
        date: str,
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> Iterator[OptionTrade]:
        options_df = self.bronze_reader.read_option_trades(
            underlying="ES",
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        )
        if options_df.empty:
            raise ValueError(f"No Bronze option trades found for {date}")
        required = ["ts_event_ns", "option_symbol", "exp_date", "strike", "right", "price", "size"]
        missing = [col for col in required if col not in options_df.columns]
        if missing:
            raise ValueError(f"Bronze option trades missing columns: {missing}")

        ts_event = options_df["ts_event_ns"].to_numpy()
        ts_recv = options_df["ts_recv_ns"].to_numpy() if "ts_recv_ns" in options_df.columns else ts_event
        underlying = options_df["underlying"].to_numpy() if "underlying" in options_df.columns else None
        option_symbol = options_df["option_symbol"].to_numpy()
        exp_date = options_df["exp_date"].to_numpy()
        strike = options_df["strike"].to_numpy()
        right = options_df["right"].to_numpy()
        price = options_df["price"].to_numpy()
        size = options_df["size"].to_numpy()
        opt_bid = options_df["opt_bid"].to_numpy() if "opt_bid" in options_df.columns else None
        opt_ask = options_df["opt_ask"].to_numpy() if "opt_ask" in options_df.columns else None
        aggressor = options_df["aggressor"].to_numpy() if "aggressor" in options_df.columns else None
        conditions = options_df["conditions"].to_numpy() if "conditions" in options_df.columns else None
        seq = options_df["seq"].to_numpy() if "seq" in options_df.columns else None

        for idx in range(len(options_df)):
            agg_val = aggressor[idx] if aggressor is not None else 0
            if agg_val in (1, "BUY", "buy"):
                agg_enum = Aggressor.BUY
            elif agg_val in (-1, "SELL", "sell"):
                agg_enum = Aggressor.SELL
            else:
                agg_enum = Aggressor.MID

            yield OptionTrade(
                ts_event_ns=int(ts_event[idx]),
                ts_recv_ns=int(ts_recv[idx]),
                source=EventSource.REPLAY,
                underlying=underlying[idx] if underlying is not None else "ES",
                option_symbol=option_symbol[idx],
                exp_date=str(exp_date[idx]),
                strike=float(strike[idx]),
                right=right[idx],
                price=float(price[idx]),
                size=int(size[idx]),
                opt_bid=opt_bid[idx] if opt_bid is not None else None,
                opt_ask=opt_ask[idx] if opt_ask is not None else None,
                aggressor=agg_enum,
                conditions=conditions[idx] if conditions is not None else None,
                seq=seq[idx] if seq is not None else None
            )

    def _iter_futures_trades_bronze(
        self,
        date: str,
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> Iterator[FuturesTrade]:
        trades_df = self.bronze_reader.read_futures_trades(
            symbol=self.futures_symbol,
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        )
        if trades_df.empty:
            raise ValueError(f"No Bronze futures trades found for {date} (symbol={self.futures_symbol})")
        required = ["ts_event_ns", "ts_recv_ns", "source", "symbol", "price", "size"]
        missing = [col for col in required if col not in trades_df.columns]
        if missing:
            raise ValueError(f"Bronze futures trades missing columns: {missing}")

        ts_event = trades_df["ts_event_ns"].to_numpy()
        ts_recv = trades_df["ts_recv_ns"].to_numpy()
        source = trades_df["source"].to_numpy()
        symbol = trades_df["symbol"].to_numpy()
        price = trades_df["price"].to_numpy()
        size = trades_df["size"].to_numpy()
        aggressor = trades_df["aggressor"].to_numpy() if "aggressor" in trades_df.columns else None
        exchange = trades_df["exchange"].to_numpy() if "exchange" in trades_df.columns else None
        conditions = trades_df["conditions"].to_numpy() if "conditions" in trades_df.columns else None
        seq = trades_df["seq"].to_numpy() if "seq" in trades_df.columns else None

        for idx in range(len(trades_df)):
            yield FuturesTrade(
                ts_event_ns=int(ts_event[idx]),
                ts_recv_ns=int(ts_recv[idx]),
                source=EventSource(source[idx]),
                symbol=symbol[idx],
                price=float(price[idx]),
                size=int(size[idx]),
                aggressor=Aggressor(aggressor[idx]) if aggressor is not None else Aggressor.MID,
                exchange=exchange[idx] if exchange is not None else None,
                conditions=conditions[idx] if conditions is not None else None,
                seq=seq[idx] if seq is not None else None
            )

    def _iter_futures_mbp10_bronze(
        self,
        date: str,
        start_ns: Optional[int],
        end_ns: Optional[int]
    ) -> Iterator[MBP10]:
        mbp_df = self.bronze_reader.read_futures_mbp10(
            symbol=self.futures_symbol,
            date=date,
            start_ns=start_ns,
            end_ns=end_ns
        )
        if mbp_df.empty:
            raise ValueError(f"No Bronze futures MBP-10 found for {date} (symbol={self.futures_symbol})")
        required = ["ts_event_ns", "ts_recv_ns", "source", "symbol"]
        missing = [col for col in required if col not in mbp_df.columns]
        if missing:
            raise ValueError(f"Bronze futures MBP-10 missing columns: {missing}")

        level_cols = []
        for idx in range(1, 11):
            level_cols.extend([
                f"bid_px_{idx}",
                f"bid_sz_{idx}",
                f"ask_px_{idx}",
                f"ask_sz_{idx}",
            ])
        missing_levels = [col for col in level_cols if col not in mbp_df.columns]
        if missing_levels:
            raise ValueError(f"Bronze futures MBP-10 missing level columns: {missing_levels}")

        ts_event = mbp_df["ts_event_ns"].to_numpy()
        ts_recv = mbp_df["ts_recv_ns"].to_numpy()
        source = mbp_df["source"].to_numpy()
        symbol = mbp_df["symbol"].to_numpy()
        is_snapshot = mbp_df["is_snapshot"].to_numpy() if "is_snapshot" in mbp_df.columns else None
        seq = mbp_df["seq"].to_numpy() if "seq" in mbp_df.columns else None

        bid_px = [mbp_df[f"bid_px_{idx}"].to_numpy() for idx in range(1, 11)]
        bid_sz = [mbp_df[f"bid_sz_{idx}"].to_numpy() for idx in range(1, 11)]
        ask_px = [mbp_df[f"ask_px_{idx}"].to_numpy() for idx in range(1, 11)]
        ask_sz = [mbp_df[f"ask_sz_{idx}"].to_numpy() for idx in range(1, 11)]

        for row_idx in range(len(mbp_df)):
            levels = [
                BidAskLevel(
                    bid_px=float(bid_px[level_idx][row_idx]),
                    bid_sz=int(bid_sz[level_idx][row_idx]),
                    ask_px=float(ask_px[level_idx][row_idx]),
                    ask_sz=int(ask_sz[level_idx][row_idx])
                )
                for level_idx in range(10)
            ]

            yield MBP10(
                ts_event_ns=int(ts_event[row_idx]),
                ts_recv_ns=int(ts_recv[row_idx]),
                source=EventSource(source[row_idx]),
                symbol=symbol[row_idx],
                levels=levels,
                is_snapshot=bool(is_snapshot[row_idx]) if is_snapshot is not None else False,
                seq=seq[row_idx] if seq is not None else None
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
        uv run python -m src.ingestion.databento.replay
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
    include_options = os.getenv("REPLAY_INCLUDE_OPTIONS", "true").lower() == "true"
    use_bronze_futures = os.getenv("REPLAY_USE_BRONZE_FUTURES", "false").lower() == "true"
    futures_symbol = os.getenv("REPLAY_FUTURES_SYMBOL", "ES")
    print(f"   Options replay: {include_options}")
    print(f"   Bronze futures replay: {use_bronze_futures} (symbol={futures_symbol})")
    
    # Initialize NATS
    bus = NATSBus(servers=[CONFIG.NATS_URL])
    await bus.connect()
    
    # Initialize DBN Ingestor
    dbn_ingestor = DBNReader()
    
    # Check available data
    available_dates = dbn_ingestor.get_available_dates('trades')
    if not available_dates:
        print("❌ No DBN data found in data/raw/ directory")
        sys.exit(1)
    
    print(f"📁 Found {len(available_dates)} days of data:")
    for date in available_dates:
        print(f"   - {date}")
    
    # Initialize Replay Publisher
    publisher = ReplayPublisher(
        bus=bus,
        dbn_ingestor=dbn_ingestor,
        replay_speed=replay_speed,
        use_bronze_futures=use_bronze_futures,
        futures_symbol=futures_symbol
    )
    
    try:
        if replay_date:
            # Replay specific date
            if replay_date not in available_dates:
                print(f"❌ Date {replay_date} not found in available data")
                sys.exit(1)

            # Start from pre-market (04:00 AM ET) so PM_HIGH/PM_LOW and SMA warmup bars exist.
            start_ns = get_premarket_start_ns(replay_date)
            start_dt = datetime.fromtimestamp(start_ns / 1e9, tz=ET)
            print(f"   Start time: {start_dt.strftime('%H:%M:%S %Z')} (premarket open)")

            await publisher.replay_date(replay_date, start_ns=start_ns, include_options=include_options)
        else:
            # Replay all dates (each starting from pre-market)
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
