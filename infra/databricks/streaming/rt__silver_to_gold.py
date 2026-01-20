# Databricks notebook source
# MAGIC %md
# MAGIC # rt__silver_to_gold
# MAGIC Gold streaming job: computes feature vectors from Silver bar_5s stream.
# MAGIC Outputs feature time-series and model input vectors.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, window, lag, avg, stddev, sum as spark_sum,
    array, lit, current_timestamp, to_json, struct
)
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType,
    IntegerType, ArrayType, TimestampType
)

# COMMAND ----------

# Configuration
SILVER_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/silver/bar_5s_stream"
CHECKPOINT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/checkpoints/rt__gold"
VECTORS_OUTPUT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/gold/setup_vectors_stream"
FEATURES_OUTPUT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/gold/feature_series_stream"

# Event Hubs configuration for publishing
EVENTHUB_NAMESPACE = "ehnspymasterdevoxxrlojskvxey"
EVENTHUB_NAME = "features_gold"
EVENTHUB_CONNECTION_STRING = dbutils.secrets.get(scope="spymaster", key="eventhub-connection-string")

# COMMAND ----------

# Read from Silver Delta stream
df_silver = (
    spark.readStream
    .format("delta")
    .load(SILVER_PATH)
)

# COMMAND ----------

# Compute feature primitives using window functions
# This creates a streaming window over the bar_5s data

# Define window specs for lookback calculations
window_spec_5 = Window.partitionBy("contract_id").orderBy("bar_ts").rowsBetween(-5, 0)
window_spec_10 = Window.partitionBy("contract_id").orderBy("bar_ts").rowsBetween(-10, 0)
window_spec_20 = Window.partitionBy("contract_id").orderBy("bar_ts").rowsBetween(-20, 0)

df_features = (
    df_silver
    # Price returns
    .withColumn("prev_close", lag("close_price", 1).over(Window.partitionBy("contract_id").orderBy("bar_ts")))
    .withColumn("return_1", (col("close_price") - col("prev_close")) / col("prev_close"))

    # Moving averages
    .withColumn("ma_5", avg("close_price").over(window_spec_5))
    .withColumn("ma_10", avg("close_price").over(window_spec_10))
    .withColumn("ma_20", avg("close_price").over(window_spec_20))

    # Volatility
    .withColumn("vol_5", stddev("close_price").over(window_spec_5))
    .withColumn("vol_10", stddev("close_price").over(window_spec_10))

    # Volume features
    .withColumn("vol_ma_5", avg("volume").over(window_spec_5))
    .withColumn("vol_ratio", col("volume") / col("vol_ma_5"))

    # Spread features
    .withColumn("spread_ma_5", avg("spread").over(window_spec_5))
    .withColumn("spread_ratio", col("spread") / col("spread_ma_5"))

    # Price position relative to MAs
    .withColumn("price_vs_ma5", (col("close_price") - col("ma_5")) / col("ma_5"))
    .withColumn("price_vs_ma10", (col("close_price") - col("ma_10")) / col("ma_10"))

    # Momentum
    .withColumn("momentum_5", col("close_price") - lag("close_price", 5).over(Window.partitionBy("contract_id").orderBy("bar_ts")))
    .withColumn("momentum_10", col("close_price") - lag("close_price", 10).over(Window.partitionBy("contract_id").orderBy("bar_ts")))

    # Bid-ask imbalance
    .withColumn("bid_ask_imbalance", (col("bid_price") - col("ask_price")) / col("spread"))
)

# COMMAND ----------

# Create feature vector array
df_vectors = (
    df_features
    .select(
        col("contract_id"),
        col("bar_ts").alias("vector_time"),
        col("session_date"),
        col("close_price"),
        col("volume"),
        array(
            col("return_1"),
            col("price_vs_ma5"),
            col("price_vs_ma10"),
            col("vol_ratio"),
            col("spread_ratio"),
        ).alias("feature_vector"),
        # Also keep individual features for dashboard
        col("return_1"),
        col("ma_5"),
        col("ma_10"),
        col("vol_5"),
        col("vol_ratio"),
        col("spread"),
        col("spread_ratio"),
        col("momentum_5"),
        col("momentum_10"),
        col("bid_ask_imbalance"),
    )
    .withColumn("model_id", lit("ES_MODEL"))
    .withColumn("gold_ingest_time", current_timestamp())
    .na.fill(0.0)  # Handle nulls from window functions at stream start
)

# COMMAND ----------

# Write vectors to Delta for inference job
query_vectors = (
    df_vectors.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/vectors")
    .partitionBy("session_date")
    .trigger(processingTime="10 seconds")
    .start(VECTORS_OUTPUT_PATH)
)

# COMMAND ----------

# Publish to Event Hubs for Fabric ingestion
ehConf = {
    "eventhubs.connectionString": sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(
        f"{EVENTHUB_CONNECTION_STRING};EntityPath={EVENTHUB_NAME}"
    ),
}

df_to_eventhub = (
    df_vectors
    .select(
        to_json(struct(
            col("contract_id"),
            col("vector_time"),
            col("model_id"),
            col("close_price"),
            col("volume"),
            col("return_1"),
            col("ma_5"),
            col("vol_ratio"),
            col("spread"),
            col("momentum_5"),
            col("bid_ask_imbalance"),
        )).alias("body")
    )
)

query_eventhub = (
    df_to_eventhub.writeStream
    .format("eventhubs")
    .options(**ehConf)
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/eventhub")
    .trigger(processingTime="10 seconds")
    .start()
)

# COMMAND ----------

# Wait for both queries
spark.streams.awaitAnyTermination()
