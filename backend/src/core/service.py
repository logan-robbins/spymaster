"""
Core Service: The Brain of Spymaster.

Agent C deliverable per NEXT.md.

This service:
- Subscribes to market data streams from NATS (market.*)
- Updates MarketState on every message
- Runs a periodic snap loop (100-250ms) to compute level signals
- Publishes level signals to NATS (levels.signals)

The Core Service is the physics engine orchestrator.
"""

import asyncio
import json
import os
from typing import Optional
from dataclasses import asdict

from src.common.bus import NATSBus
from src.common.config import CONFIG
from src.common.event_types import (
    FuturesTrade, MBP10, OptionTrade, 
    EventSource, Aggressor
)
from src.core.market_state import MarketState
from src.core.level_signal_service import LevelSignalService
from src.core.greek_enricher import GreekEnricher


class CoreService:
    """
    The Brain: orchestrates market state updates and level signal computation.
    
    Usage:
        service = CoreService(bus)
        await service.start()
        # Service runs until stopped
    """
    
    def __init__(
        self, 
        bus: NATSBus,
        greek_enricher: Optional[GreekEnricher] = None,
        config=None,
        user_hotzones: Optional[list] = None
    ):
        """
        Initialize Core Service.
        
        Args:
            bus: NATSBus instance (must be connected)
            greek_enricher: Optional GreekEnricher instance (will create if not provided)
            config: Config object (defaults to global CONFIG)
            user_hotzones: Optional user-defined levels
        """
        self.bus = bus
        self.config = config or CONFIG
        self.user_hotzones = user_hotzones
        
        # Initialize market state
        buffer_seconds = max(self.config.W_b, self.config.CONFIRMATION_WINDOW_SECONDS)
        self.market_state = MarketState(
            max_buffer_window_seconds=buffer_seconds * 2  # 2x confirmation window for safety
        )
        
        # Initialize greek enricher (for option trades)
        if greek_enricher is None:
            # Try to get API key from environment
            api_key = os.environ.get("POLYGON_API_KEY")
            if api_key:
                self.greek_enricher = GreekEnricher(api_key=api_key)
                print("✅ GreekEnricher initialized with API key from environment")
            else:
                print("⚠️  No POLYGON_API_KEY found - greek enrichment disabled")
                self.greek_enricher = None
        else:
            self.greek_enricher = greek_enricher
        
        # Initialize level signal service
        # Use REPLAY_DATE if set, otherwise use current date
        trading_date = os.getenv("REPLAY_DATE", None)
        self.level_signal_service = LevelSignalService(
            market_state=self.market_state,
            user_hotzones=user_hotzones,
            config=self.config,
            trading_date=trading_date
        )

        # Optional viewport scoring (Phase 3)
        self.viewport_scoring_service = None
        if os.getenv("VIEWPORT_SCORING_ENABLED", "false").lower() == "true":
            from pathlib import Path
            import joblib

            from src.core.viewport_feature_builder import ViewportFeatureBuilder
            from src.core.viewport_manager import ViewportManager
            from src.core.viewport_scoring_service import ViewportScoringService
            from src.core.inference_engine import ViewportInferenceEngine
            from src.ml.tree_inference import TreeModelBundle
            from src.ml.retrieval_engine import RetrievalIndex

            model_dir = Path(os.getenv("VIEWPORT_MODEL_DIR", "data/ml/boosted_trees"))
            retrieval_path = Path(os.getenv("VIEWPORT_RETRIEVAL_INDEX", "data/ml/retrieval_index.joblib"))
            ablation = os.getenv("VIEWPORT_ABLATION", "full")
            horizons_env = os.getenv("VIEWPORT_HORIZONS", "")
            horizons = [int(h) for h in horizons_env.split(",") if h.strip()] or [60, 120, 180, 300]
            timeframe = os.getenv("VIEWPORT_TIMEFRAME", "").strip() or None

            if not model_dir.exists():
                raise FileNotFoundError(f"Viewport model dir missing: {model_dir}")
            if not retrieval_path.exists():
                raise FileNotFoundError(f"Viewport retrieval index missing: {retrieval_path}")

            retrieval_index = joblib.load(retrieval_path)
            if not isinstance(retrieval_index, RetrievalIndex):
                raise ValueError("Viewport retrieval index is not a RetrievalIndex instance.")

            stage_a_bundle = TreeModelBundle(
                model_dir=model_dir,
                stage="stage_a",
                ablation=ablation,
                horizons=horizons,
                timeframe=timeframe
            )
            stage_b_bundle = TreeModelBundle(
                model_dir=model_dir,
                stage="stage_b",
                ablation=ablation,
                horizons=horizons,
                timeframe=timeframe
            )
            stage_a_engine = ViewportInferenceEngine(stage_a_bundle, retrieval_index)
            stage_b_engine = ViewportInferenceEngine(stage_b_bundle, retrieval_index)

            viewport_manager = ViewportManager(
                fuel_engine=self.level_signal_service.fuel_engine,
                trading_date=self.level_signal_service._trading_date
            )
            feature_builder = ViewportFeatureBuilder(
                barrier_engine=self.level_signal_service.barrier_engine,
                tape_engine=self.level_signal_service.tape_engine,
                fuel_engine=self.level_signal_service.fuel_engine
            )
            self.viewport_scoring_service = ViewportScoringService(
                market_state=self.market_state,
                level_universe=self.level_signal_service.level_universe,
                viewport_manager=viewport_manager,
                feature_builder=feature_builder,
                stage_a_engine=stage_a_engine,
                stage_b_engine=stage_b_engine,
                trading_date=self.level_signal_service._trading_date
            )
        
        # Snap loop control
        self.snap_task: Optional[asyncio.Task] = None
        self.running = False
        
        print("✅ CoreService initialized")
    
    async def start(self):
        """
        Start the Core Service.
        
        This will:
        1. Subscribe to market data streams
        2. Start the snap loop
        3. Run until stopped
        """
        if self.running:
            print("⚠️  CoreService already running")
            return
        
        self.running = True
        print("🚀 CoreService starting...")
        
        # Subscribe to market data streams
        await self._subscribe_to_market_data()
        
        # Start snap loop
        self.snap_task = asyncio.create_task(self._snap_loop())
        
        print("✅ CoreService running")
        
        # Keep running
        try:
            await self.snap_task
        except asyncio.CancelledError:
            print("🛑 CoreService stopped")
    
    async def stop(self):
        """Stop the Core Service."""
        print("🛑 Stopping CoreService...")
        self.running = False
        
        if self.snap_task:
            self.snap_task.cancel()
            try:
                await self.snap_task
            except asyncio.CancelledError:
                pass
        
        print("✅ CoreService stopped")
    
    async def _subscribe_to_market_data(self):
        """Subscribe to all market data streams from NATS."""
        print("🎧 Subscribing to market data streams...")
        
        # Subscribe to futures trades (ES)
        await self.bus.subscribe(
            subject="market.futures.trades",
            callback=self._handle_futures_trade,
            durable_name="core-futures-trades"
        )
        
        # Subscribe to futures MBP-10 (ES)
        await self.bus.subscribe(
            subject="market.futures.mbp10",
            callback=self._handle_futures_mbp10,
            durable_name="core-futures-mbp10"
        )
        
        # Subscribe to options trades (SPY options)
        await self.bus.subscribe(
            subject="market.options.trades",
            callback=self._handle_option_trade,
            durable_name="core-options-trades"
        )
        
        print("✅ Subscribed to market data streams")
    
    async def _handle_futures_trade(self, msg: dict):
        """
        Handle incoming ES futures trade from NATS.
        
        Args:
            msg: Deserialized FuturesTrade dict
        """
        try:
            # Reconstruct FuturesTrade from dict
            trade = FuturesTrade(
                ts_event_ns=msg["ts_event_ns"],
                ts_recv_ns=msg["ts_recv_ns"],
                source=EventSource(msg["source"]),
                symbol=msg["symbol"],
                price=msg["price"],
                size=msg["size"],
                aggressor=Aggressor(msg["aggressor"]),
                exchange=msg.get("exchange"),
                conditions=msg.get("conditions"),
                seq=msg.get("seq")
            )
            
            # Update market state
            self.market_state.update_es_trade(trade)
            
        except Exception as e:
            print(f"❌ Error handling futures trade: {e}")
    
    async def _handle_futures_mbp10(self, msg: dict):
        """
        Handle incoming ES MBP-10 snapshot from NATS.
        
        Args:
            msg: Deserialized MBP10 dict
        """
        try:
            # Reconstruct MBP10 from dict
            # MBP10 has nested BidAskLevel structures, need careful reconstruction
            from src.common.event_types import BidAskLevel
            
            levels = []
            for level_dict in msg.get("levels", []):
                level = BidAskLevel(
                    bid_px=level_dict["bid_px"],
                    bid_sz=level_dict["bid_sz"],
                    ask_px=level_dict["ask_px"],
                    ask_sz=level_dict["ask_sz"]
                )
                levels.append(level)
            
            mbp = MBP10(
                ts_event_ns=msg["ts_event_ns"],
                ts_recv_ns=msg["ts_recv_ns"],
                source=EventSource(msg["source"]),
                symbol=msg["symbol"],
                levels=levels,
                is_snapshot=msg.get("is_snapshot", False),
                seq=msg.get("seq")
            )
            
            # Update market state
            self.market_state.update_es_mbp10(mbp)
            
        except Exception as e:
            print(f"❌ Error handling MBP-10: {e}")
    
    async def _handle_option_trade(self, msg: dict):
        """
        Handle incoming option trade from NATS.
        
        Args:
            msg: Deserialized OptionTrade dict
        """
        try:
            # Reconstruct OptionTrade from dict
            trade = OptionTrade(
                ts_event_ns=msg["ts_event_ns"],
                ts_recv_ns=msg["ts_recv_ns"],
                source=EventSource(msg["source"]),
                underlying=msg["underlying"],
                option_symbol=msg["option_symbol"],
                exp_date=msg["exp_date"],
                strike=msg["strike"],
                right=msg["right"],
                price=msg["price"],
                size=msg["size"],
                opt_bid=msg.get("opt_bid"),
                opt_ask=msg.get("opt_ask"),
                aggressor=Aggressor(msg["aggressor"]),
                conditions=msg.get("conditions"),
                seq=msg.get("seq")
            )
            
            # Enrich with greeks (from cache)
            delta = 0.0
            gamma = 0.0
            
            if self.greek_enricher:
                greeks = self.greek_enricher.get_greeks(trade.option_symbol)

                if greeks:
                    delta = greeks.delta
                    gamma = greeks.gamma
            
            # Update market state
            self.market_state.update_option_trade(
                trade=trade,
                delta=delta,
                gamma=gamma
            )
            
        except Exception as e:
            print(f"❌ Error handling option trade: {e}")
    
    async def _snap_loop(self):
        """
        Periodic snap loop: compute level signals and publish to NATS.
        
        Runs every SNAP_INTERVAL_MS (default 250ms).
        """
        interval_s = self.config.SNAP_INTERVAL_MS / 1000.0
        print(f"🔄 Snap loop starting (interval={interval_s}s)")
        
        while self.running:
            try:
                # Compute level signals
                payload = self.level_signal_service.compute_level_signals()
                if self.viewport_scoring_service is not None:
                    viewport_targets = self.viewport_scoring_service.score_viewport()
                    payload["viewport"] = {
                        "ts": payload.get("ts"),
                        "targets": viewport_targets
                    }

                # Build flow snapshot for frontend strike grid (0DTE only)
                spot = payload.get("spy", {}).get("spot")
                flow_snapshot = self.market_state.get_option_flow_snapshot(
                    spot=spot,
                    strike_range=self.config.STRIKE_RANGE,
                    exp_date_filter=self.level_signal_service.trading_date
                )
                
                # Publish to NATS
                await self.bus.publish(
                    subject="levels.signals",
                    payload=payload
                )

                # Publish flow snapshot (market.* stream)
                await self.bus.publish(
                    subject="market.flow",
                    payload=flow_snapshot
                )
                
                # Log summary (optional, can be removed for performance)
                num_levels = len(payload.get("levels", []))
                spot = payload.get("spy", {}).get("spot")
                spot_str = f"{spot:.2f}" if spot is not None else "N/A"
                print(f"📊 Published {num_levels} level signals (spot={spot_str})")
                if num_levels > 0:
                    first_level = payload["levels"][0]
                    print(f"   First level: {first_level.get('kind')} @ {first_level.get('price', 0):.2f}")
                
            except Exception as e:
                print(f"❌ Error in snap loop: {e}")
            
            # Sleep until next snap
            await asyncio.sleep(interval_s)
