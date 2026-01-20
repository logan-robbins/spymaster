import pytest
import json
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
from pyspark.sql import functions as F


def test_mbo_schema_validation(spark_session):
    mbo_schema = StructType([
        StructField("action", StringType(), nullable=False),
        StructField("price", DoubleType(), nullable=False),
        StructField("size", LongType(), nullable=False),
        StructField("side", StringType(), nullable=False),
        StructField("order_id", LongType(), nullable=True),
        StructField("contract_id", StringType(), nullable=False),
        StructField("event_time", LongType(), nullable=False),
    ])
    
    valid_data = [
        {
            "action": "A",
            "price": 6050.25,
            "size": 10,
            "side": "B",
            "order_id": 12345,
            "contract_id": "ESZ5",
            "event_time": 1700000000000000000
        }
    ]
    
    df = spark_session.createDataFrame(valid_data, schema=mbo_schema)
    assert df.count() == 1
    assert df.filter(F.col("price") > 0).count() == 1


def test_json_parsing_with_dlq(spark_session):
    mbo_schema = StructType([
        StructField("action", StringType(), nullable=False),
        StructField("price", DoubleType(), nullable=False),
        StructField("size", LongType(), nullable=False),
        StructField("side", StringType(), nullable=False),
        StructField("contract_id", StringType(), nullable=False),
        StructField("event_time", LongType(), nullable=False),
    ])
    
    raw_data = [
        ('{"action":"A","price":6050.25,"size":10,"side":"B","contract_id":"ESZ5","event_time":1700000000000000000}',),
        ('{"action":"A","price":"INVALID","size":10,"side":"B","contract_id":"ESZ5","event_time":1700000000000000000}',),
        ('INVALID_JSON',),
    ]
    
    df_raw = spark_session.createDataFrame(raw_data, ["body_str"])
    
    df_parsed = df_raw.withColumn("parsed", F.from_json(F.col("body_str"), mbo_schema))
    
    df_valid = df_parsed.filter(F.col("parsed").isNotNull())
    df_invalid = df_parsed.filter(F.col("parsed").isNull())
    
    assert df_valid.count() == 1
    assert df_invalid.count() == 2


def test_watermark_deduplication(spark_session, tmp_path):
    from pyspark.sql.types import TimestampType
    from datetime import datetime, timedelta
    
    base_time = datetime(2024, 1, 1, 10, 0, 0)
    
    data = [
        ("ESZ5", 1001, base_time, 6050.0),
        ("ESZ5", 1001, base_time, 6050.0),
        ("ESZ5", 1002, base_time + timedelta(seconds=5), 6051.0),
        ("ESZ5", 1002, base_time + timedelta(seconds=5), 6051.0),
        ("ESZ5", 1003, base_time + timedelta(seconds=10), 6052.0),
    ]
    
    df = spark_session.createDataFrame(
        data,
        ["contract_id", "order_id", "event_time_ts", "price"]
    )
    
    df_dedup = (
        df
        .withWatermark("event_time_ts", "5 seconds")
        .dropDuplicatesWithinWatermark(["contract_id", "order_id"])
    )
    
    result = df_dedup.collect()
    assert len(result) == 3


def test_side_sign_calculation(spark_session):
    data = [
        ("B",),
        ("BUY",),
        ("BID",),
        ("A",),
        ("ASK",),
        ("SELL",),
        ("S",),
        ("UNKNOWN",),
    ]
    
    df = spark_session.createDataFrame(data, ["side"])
    
    df_with_sign = df.withColumn(
        "side_sign",
        F.when(F.upper(F.col("side")).isin("B", "BUY", "BID"), F.lit(1))
        .when(F.upper(F.col("side")).isin("S", "SELL", "ASK", "A"), F.lit(-1))
        .otherwise(F.lit(0))
    )
    
    buy_count = df_with_sign.filter(F.col("side_sign") == 1).count()
    sell_count = df_with_sign.filter(F.col("side_sign") == -1).count()
    unknown_count = df_with_sign.filter(F.col("side_sign") == 0).count()
    
    assert buy_count == 3
    assert sell_count == 4
    assert unknown_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
