# Databricks notebook source
# MAGIC %md
# MAGIC # rt__mbo_raw_to_bronze
# MAGIC Bronze streaming job: reads from Event Hubs `mbo_raw` and writes to Delta lake.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, current_timestamp, to_date
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType, IntegerType
)

# COMMAND ----------

# Configuration
EVENTHUB_NAMESPACE = "ehnspymasterdevoxxrlojskvxey"
EVENTHUB_NAME = "mbo_raw"
CONSUMER_GROUP = "databricks_bronze"
CHECKPOINT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/checkpoints/rt__bronze"
OUTPUT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/bronze/stream"

# Get connection string from Key Vault (via Databricks secret scope)
# Set up secret scope: databricks secrets create-scope --scope spymaster
EVENTHUB_CONNECTION_STRING = dbutils.secrets.get(scope="spymaster", key="eventhub-connection-string")

# COMMAND ----------

# Bronze envelope schema (matches data contract in WORK.md)
bronze_schema = StructType([
    StructField("event_time", LongType(), True),
    StructField("ingest_time", LongType(), True),
    StructField("venue", StringType(), True),
    StructField("symbol", IntegerType(), True),
    StructField("instrument_type", StringType(), True),
    StructField("underlier", StringType(), True),
    StructField("contract_id", StringType(), True),
    StructField("action", StringType(), True),
    StructField("order_id", LongType(), True),
    StructField("side", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("size", IntegerType(), True),
    StructField("sequence", LongType(), True),
    StructField("payload", StringType(), True),
])

# COMMAND ----------

# Event Hubs configuration
ehConf = {
    "eventhubs.connectionString": sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(
        f"{EVENTHUB_CONNECTION_STRING};EntityPath={EVENTHUB_NAME}"
    ),
    "eventhubs.consumerGroup": CONSUMER_GROUP,
    "eventhubs.startingPosition": '{"offset": "-1", "seqNo": -1, "enqueuedTime": null, "isInclusive": true}',
}

# COMMAND ----------

# Read from Event Hubs
df_raw = (
    spark.readStream
    .format("eventhubs")
    .options(**ehConf)
    .load()
)

# Parse the body as JSON
df_parsed = (
    df_raw
    .select(
        from_json(col("body").cast("string"), bronze_schema).alias("data"),
        col("enqueuedTime").alias("eventhub_enqueued_time"),
        col("offset").alias("eventhub_offset"),
        col("sequenceNumber").alias("eventhub_sequence"),
    )
    .select("data.*", "eventhub_enqueued_time", "eventhub_offset", "eventhub_sequence")
    .withColumn("bronze_ingest_time", current_timestamp())
    .withColumn("session_date", to_date(col("eventhub_enqueued_time")))
)

# COMMAND ----------

# Write to Delta with partitioning
query = (
    df_parsed.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .partitionBy("session_date", "underlier", "instrument_type")
    .trigger(processingTime="10 seconds")
    .start(OUTPUT_PATH)
)

# COMMAND ----------

# Wait for termination (runs indefinitely)
query.awaitTermination()
