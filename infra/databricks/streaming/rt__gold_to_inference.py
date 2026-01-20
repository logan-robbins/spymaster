# Databricks notebook source
# MAGIC %md
# MAGIC # rt__gold_to_inference
# MAGIC Inference streaming job: calls Azure ML endpoint with feature vectors.
# MAGIC Publishes scores to Event Hub for Fabric dashboard consumption.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, pandas_udf, struct, to_json, current_timestamp, lit
)
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, DoubleType,
    IntegerType, ArrayType
)
import pandas as pd
import requests
import json
from typing import Iterator

# COMMAND ----------

# Configuration
GOLD_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/gold/setup_vectors_stream"
CHECKPOINT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/checkpoints/rt__inference"
SCORES_OUTPUT_PATH = "abfss://lake@spymasterdevlakeoxxrlojs.dfs.core.windows.net/gold/inference_scores_stream"

# Azure ML endpoint configuration
AML_ENDPOINT_URI = "https://es-model-endpoint.westus.inference.ml.azure.com/score"
AML_API_KEY = dbutils.secrets.get(scope="spymaster", key="aml-endpoint-key")

# Event Hubs configuration
EVENTHUB_NAMESPACE = "ehnspymasterdevoxxrlojskvxey"
EVENTHUB_NAME = "inference_scores"
EVENTHUB_CONNECTION_STRING = dbutils.secrets.get(scope="spymaster", key="eventhub-connection-string")

# COMMAND ----------

# Define inference UDF that calls Azure ML endpoint
@pandas_udf("struct<prediction:int, prob_0:double, prob_1:double>")
def call_aml_endpoint(feature_vectors: pd.Series) -> pd.DataFrame:
    """
    Batch inference UDF that calls Azure ML managed online endpoint.
    """
    results = []

    # Convert Series of arrays to list of lists
    vectors = [list(v) if v is not None else [0.0] * 5 for v in feature_vectors]

    # Batch the requests (Azure ML supports batch inference)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]

        try:
            response = requests.post(
                AML_ENDPOINT_URI,
                headers={
                    "Authorization": f"Bearer {AML_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"features": batch},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()

            # Parse response
            if isinstance(result, str):
                result = json.loads(result)

            predictions = result.get("predictions", [])
            probabilities = result.get("probabilities", [])

            for pred, probs in zip(predictions, probabilities):
                results.append({
                    "prediction": int(pred),
                    "prob_0": float(probs[0]) if len(probs) > 0 else 0.0,
                    "prob_1": float(probs[1]) if len(probs) > 1 else 0.0,
                })

        except Exception as e:
            # On error, append nulls for this batch
            for _ in batch:
                results.append({
                    "prediction": -1,
                    "prob_0": 0.0,
                    "prob_1": 0.0,
                })
            print(f"Inference error: {e}")

    return pd.DataFrame(results)

# COMMAND ----------

# Read from Gold vectors stream
df_gold = (
    spark.readStream
    .format("delta")
    .load(GOLD_PATH)
)

# COMMAND ----------

# Apply inference
df_with_scores = (
    df_gold
    .withColumn("inference_result", call_aml_endpoint(col("feature_vector")))
    .select(
        col("contract_id"),
        col("vector_time"),
        col("session_date"),
        col("model_id"),
        col("close_price"),
        col("feature_vector"),
        col("inference_result.prediction").alias("prediction"),
        col("inference_result.prob_0").alias("prob_0"),
        col("inference_result.prob_1").alias("prob_1"),
    )
    .withColumn("inference_time", current_timestamp())
    .withColumn("model_version", lit("1"))
)

# COMMAND ----------

# Write scores to Delta
query_delta = (
    df_with_scores.writeStream
    .format("delta")
    .outputMode("append")
    .option("checkpointLocation", f"{CHECKPOINT_PATH}/delta")
    .partitionBy("session_date")
    .trigger(processingTime="10 seconds")
    .start(SCORES_OUTPUT_PATH)
)

# COMMAND ----------

# Publish to Event Hubs for Fabric dashboard
ehConf = {
    "eventhubs.connectionString": sc._jvm.org.apache.spark.eventhubs.EventHubsUtils.encrypt(
        f"{EVENTHUB_CONNECTION_STRING};EntityPath={EVENTHUB_NAME}"
    ),
}

df_to_eventhub = (
    df_with_scores
    .select(
        to_json(struct(
            col("contract_id"),
            col("vector_time"),
            col("model_id"),
            col("model_version"),
            col("close_price"),
            col("prediction"),
            col("prob_0"),
            col("prob_1"),
            col("inference_time"),
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
