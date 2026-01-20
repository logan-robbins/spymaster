# Databricks notebook source
# MAGIC %md
# MAGIC # rt__bronze_to_silver
# MAGIC Silver streaming job: stateful orderbook reconstruction + rollups.
# MAGIC Uses `applyInPandasWithState` for stateful processing keyed by contract_id.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, window, sum as spark_sum, avg, min as spark_min, max as spark_max,
    first, last, count, lit, current_timestamp, to_date, pandas_udf
)
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, IntegerType,
    TimestampType, ArrayType
)
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout
import pandas as pd
from typing import Iterator, Tuple
from dataclasses import dataclass, asdict
import json

# COMMAND ----------

# Configuration
BRONZE_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/bronze/stream"
CHECKPOINT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/checkpoints/rt__silver"
BAR_5S_OUTPUT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/silver/bar_5s_stream"
BOOK_STATE_OUTPUT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/silver/orderbook_state_stream"

# COMMAND ----------

# State schema for orderbook
@dataclass
class OrderbookState:
    contract_id: str
    bids: dict  # price -> size
    asks: dict  # price -> size
    last_update_ns: int
    last_trade_price: float
    last_trade_size: int
    total_volume: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OrderbookState":
        return cls(**d)

# COMMAND ----------

# Bar 5s output schema
bar_5s_schema = StructType([
    StructField("contract_id", StringType(), True),
    StructField("bar_ts", LongType(), True),
    StructField("open_price", DoubleType(), True),
    StructField("high_price", DoubleType(), True),
    StructField("low_price", DoubleType(), True),
    StructField("close_price", DoubleType(), True),
    StructField("volume", LongType(), True),
    StructField("trade_count", IntegerType(), True),
    StructField("bid_price", DoubleType(), True),
    StructField("ask_price", DoubleType(), True),
    StructField("spread", DoubleType(), True),
    StructField("mid_price", DoubleType(), True),
    StructField("session_date", StringType(), True),
])

# State output schema
state_output_schema = StructType([
    StructField("contract_id", StringType()),
    StructField("state_json", StringType()),
])

# COMMAND ----------

def process_orderbook_updates(
    key: Tuple[str],
    pdf_iter: Iterator[pd.DataFrame],
    state: GroupState
) -> Iterator[pd.DataFrame]:
    """
    Stateful processing function for orderbook reconstruction.
    Keyed by contract_id, maintains book state across micro-batches.
    """
    contract_id = key[0]

    # Load or initialize state
    if state.exists:
        book_state = OrderbookState.from_dict(json.loads(state.get))
    else:
        book_state = OrderbookState(
            contract_id=contract_id,
            bids={},
            asks={},
            last_update_ns=0,
            last_trade_price=0.0,
            last_trade_size=0,
            total_volume=0,
        )

    bars = []
    BAR_DURATION_NS = 5_000_000_000  # 5 seconds

    for pdf in pdf_iter:
        if pdf.empty:
            continue

        # Sort by event time
        pdf = pdf.sort_values("event_time")

        for _, row in pdf.iterrows():
            action = row["action"]
            side = row["side"]
            price = row["price"]
            size = row["size"]
            event_time = row["event_time"]

            # Update orderbook state based on action
            if action in ("A", "a"):  # Add
                book = book_state.bids if side == "B" else book_state.asks
                book[price] = book.get(price, 0) + size
            elif action in ("C", "c"):  # Cancel
                book = book_state.bids if side == "B" else book_state.asks
                if price in book:
                    book[price] = max(0, book.get(price, 0) - size)
                    if book[price] == 0:
                        del book[price]
            elif action in ("T", "t"):  # Trade
                book_state.last_trade_price = price
                book_state.last_trade_size = size
                book_state.total_volume += size

            book_state.last_update_ns = event_time

        # Compute bar metrics from current state
        best_bid = max(book_state.bids.keys()) if book_state.bids else 0.0
        best_ask = min(book_state.asks.keys()) if book_state.asks else 0.0
        spread = best_ask - best_bid if best_bid and best_ask else 0.0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0

        # Get the bar timestamp (floor to 5s boundary)
        bar_ts = (book_state.last_update_ns // BAR_DURATION_NS) * BAR_DURATION_NS

        # Create bar output
        bar = {
            "contract_id": contract_id,
            "bar_ts": bar_ts,
            "open_price": book_state.last_trade_price,
            "high_price": book_state.last_trade_price,
            "low_price": book_state.last_trade_price,
            "close_price": book_state.last_trade_price,
            "volume": book_state.total_volume,
            "trade_count": 1,
            "bid_price": best_bid,
            "ask_price": best_ask,
            "spread": spread,
            "mid_price": mid_price,
            "session_date": str(pd.Timestamp(book_state.last_update_ns, unit='ns').date()),
        }
        bars.append(bar)

    # Update state
    state.update(json.dumps(book_state.to_dict()))

    if bars:
        yield pd.DataFrame(bars)
    else:
        yield pd.DataFrame(columns=bar_5s_schema.fieldNames())

# COMMAND ----------

# Read from Bronze Delta stream
df_bronze = (
    spark.readStream
    .format("delta")
    .load(BRONZE_PATH)
)

# COMMAND ----------

# Apply stateful processing
df_silver = (
    df_bronze
    .groupBy("contract_id")
    .applyInPandasWithState(
        process_orderbook_updates,
        outputStructType=bar_5s_schema,
        stateStructType=state_output_schema,
        outputMode="append",
        timeoutConf=GroupStateTimeout.NoTimeout,
    )
)

# COMMAND ----------

# Write bar_5s stream to Delta
query = (
    df_silver.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .partitionBy("session_date")
    .trigger(processingTime="10 seconds")
    .start(BAR_5S_OUTPUT_PATH)
)

# COMMAND ----------

query.awaitTermination()
