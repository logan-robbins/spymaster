import pytest
from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timedelta


def test_ofi_calculation(spark_session):
    data = [
        (100.0, 50.0),
        (80.0, 120.0),
        (200.0, 200.0),
    ]
    
    df = spark_session.createDataFrame(data, ["buy_size", "sell_size"])
    
    df_with_ofi = df.withColumn("ofi", F.col("buy_size") - F.col("sell_size"))
    
    results = df_with_ofi.collect()
    
    assert results[0]["ofi"] == 50.0
    assert results[1]["ofi"] == -40.0
    assert results[2]["ofi"] == 0.0


def test_ofi_ratio_calculation(spark_session):
    data = [
        (100.0, 50.0),
        (80.0, 120.0),
        (0.0, 0.0),
    ]
    
    df = spark_session.createDataFrame(data, ["buy_size", "sell_size"])
    
    df_with_ofi = df.withColumn("ofi", F.col("buy_size") - F.col("sell_size"))
    df_with_ofi = df_with_ofi.withColumn(
        "ofi_ratio",
        F.when(
            (F.col("buy_size") + F.col("sell_size")) > 0,
            F.col("ofi") / (F.col("buy_size") + F.col("sell_size"))
        ).otherwise(0.0)
    )
    
    results = df_with_ofi.collect()
    
    assert round(results[0]["ofi_ratio"], 4) == round(50.0 / 150.0, 4)
    assert round(results[1]["ofi_ratio"], 4) == round(-40.0 / 200.0, 4)
    assert results[2]["ofi_ratio"] == 0.0


def test_pressure_calculation(spark_session):
    data = [
        (100.0, 50.0),
        (80.0, 120.0),
        (100.0, 100.0),
    ]
    
    df = spark_session.createDataFrame(data, ["bid_size", "ask_size"])
    
    df_with_pressure = df.withColumn("pressure", F.col("bid_size") - F.col("ask_size"))
    
    results = df_with_pressure.collect()
    
    assert results[0]["pressure"] == 50.0
    assert results[1]["pressure"] == -40.0
    assert results[2]["pressure"] == 0.0


def test_rolling_window_features(spark_session):
    data = [
        ("ESZ5", 1, 6050.0, 0.25),
        ("ESZ5", 2, 6050.5, 0.50),
        ("ESZ5", 3, 6051.0, 0.25),
        ("ESZ5", 4, 6051.5, 0.75),
        ("ESZ5", 5, 6052.0, 0.50),
    ]
    
    df = spark_session.createDataFrame(data, ["contract_id", "bucket_ns", "mid", "spread"])
    
    window_spec = Window.partitionBy("contract_id").orderBy("bucket_ns").rowsBetween(-2, 0)
    
    df_with_windows = df.withColumn("spread_mean_3", F.avg("spread").over(window_spec))
    
    results = df_with_windows.orderBy("bucket_ns").collect()
    
    assert results[0]["spread_mean_3"] == 0.25
    assert round(results[1]["spread_mean_3"], 4) == round((0.25 + 0.50) / 2, 4)
    assert round(results[2]["spread_mean_3"], 4) == round((0.25 + 0.50 + 0.25) / 3, 4)


def test_mid_return_calculation(spark_session):
    data = [
        ("ESZ5", 1, 6050.0),
        ("ESZ5", 2, 6051.0),
        ("ESZ5", 3, 6050.5),
        ("ESZ5", 4, 6052.0),
    ]
    
    df = spark_session.createDataFrame(data, ["contract_id", "bucket_ns", "mid"])
    
    df_with_lag = df.withColumn(
        "mid_lag1",
        F.lag("mid", 1).over(Window.partitionBy("contract_id").orderBy("bucket_ns"))
    )
    
    df_with_ret = df_with_lag.withColumn(
        "mid_ret",
        (F.col("mid") - F.col("mid_lag1")) / F.col("mid_lag1")
    )
    
    results = df_with_ret.orderBy("bucket_ns").collect()
    
    assert results[0]["mid_ret"] is None
    assert round(results[1]["mid_ret"], 6) == round((6051.0 - 6050.0) / 6050.0, 6)
    assert round(results[2]["mid_ret"], 6) == round((6050.5 - 6051.0) / 6051.0, 6)


def test_feature_vector_creation(spark_session):
    data = [
        (6050.0, 0.25, 100.0, 50.0),
    ]
    
    df = spark_session.createDataFrame(data, ["mid", "spread", "bid_size", "ask_size"])
    
    feature_cols = ["mid", "spread", "bid_size", "ask_size"]
    
    df_with_vector = df.withColumn(
        "feature_vector",
        F.array(*[F.col(c).cast("double") for c in feature_cols])
    )
    
    result = df_with_vector.collect()[0]
    
    assert len(result["feature_vector"]) == 4
    assert result["feature_vector"][0] == 6050.0
    assert result["feature_vector"][1] == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
