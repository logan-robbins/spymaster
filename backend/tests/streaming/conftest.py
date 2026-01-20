import pytest
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip


@pytest.fixture(scope="session")
def spark_session():
    builder = (
        SparkSession.builder
        .appName("TestSpymaster")
        .master("local[2]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.streaming.stateStore.providerClass",
                "org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )
    
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    yield spark
    
    spark.stop()


@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture(scope="function")
def checkpoint_dir(tmp_path):
    checkpoint_path = tmp_path / "checkpoints"
    checkpoint_path.mkdir()
    return str(checkpoint_path)


@pytest.fixture(scope="function")
def delta_path(tmp_path):
    delta_path = tmp_path / "delta"
    delta_path.mkdir()
    return str(delta_path)
