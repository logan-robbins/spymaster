"""
Tests for SilverCompactor.

Verifies:
- Module initialization
- Hash expression building
- Deduplication logic
- Partitioned output
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.silver_compactor import SilverCompactor, SilverReader


class TestSilverCompactorInit:
    """Tests for initialization."""

    def test_init_creates_silver_dir(self):
        """Silver directory should be created on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = os.path.join(tmpdir, 'lake')
            sc = SilverCompactor(data_root=data_root)

            assert os.path.exists(sc.silver_root)

    def test_schema_config(self):
        """All expected schemas should be configured."""
        sc = SilverCompactor()

        expected_schemas = [
            'stocks.trades',
            'stocks.quotes',
            'futures.trades',
            'futures.mbp10',
            'options.trades',
            'options.greeks_snapshots',
        ]

        for schema in expected_schemas:
            assert schema in sc.SCHEMA_CONFIG


class TestHashExpression:
    """Tests for event_id hash building."""

    def test_build_hash_expr_single_col(self):
        """Hash expression with single column."""
        sc = SilverCompactor()
        expr = sc._build_hash_expr(['col1'])

        assert 'md5' in expr
        assert 'col1' in expr

    def test_build_hash_expr_multiple_cols(self):
        """Hash expression with multiple columns."""
        sc = SilverCompactor()
        expr = sc._build_hash_expr(['col1', 'col2', 'col3'])

        assert 'md5' in expr
        assert 'col1' in expr
        assert 'col2' in expr
        assert 'col3' in expr
        assert "|| '|' ||" in expr  # Separator


class TestCompaction:
    """Tests for compaction logic with synthetic data."""

    @pytest.fixture
    def temp_lake(self):
        """Create temporary data lake with Bronze data."""
        tmpdir = tempfile.mkdtemp()
        data_root = os.path.join(tmpdir, 'lake')

        # Create Bronze directory structure
        bronze_path = os.path.join(
            data_root,
            'bronze',
            'futures',
            'trades',
            'symbol=ES',
            'date=2025-12-16',
            'hour=14'
        )
        os.makedirs(bronze_path)

        # Create synthetic Bronze data with duplicates
        df = pd.DataFrame({
            'ts_event_ns': [
                1734357600_000_000_000,  # Duplicate
                1734357600_000_000_000,  # Duplicate
                1734357601_000_000_000,
                1734357602_000_000_000,
            ],
            'ts_recv_ns': [
                1734357600_001_000_000,
                1734357600_002_000_000,  # Later recv -> should be deduped
                1734357601_001_000_000,
                1734357602_001_000_000,
            ],
            'source': ['sim', 'sim', 'sim', 'sim'],
            'symbol': ['ES', 'ES', 'ES', 'ES'],
            'price': [6870.0, 6870.0, 6870.25, 6870.50],
            'size': [10, 10, 15, 20],
            'aggressor': [1, 1, -1, 1],
            'exchange': [None, None, None, None],
            'seq': [1, 1, 2, 3],  # Same seq for duplicates
        })

        # Write Bronze parquet
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            os.path.join(bronze_path, 'part-0001.parquet'),
            compression='zstd'
        )

        yield data_root

        # Cleanup
        shutil.rmtree(tmpdir)

    def test_compact_removes_duplicates(self, temp_lake):
        """Compaction should remove duplicate events."""
        sc = SilverCompactor(data_root=temp_lake)

        result = sc.compact_date(
            date='2025-12-16',
            schema='futures.trades',
            partition_value='ES'
        )

        assert result['status'] == 'success'
        # After dedup, we should have 3 unique rows (original 4 with 1 duplicate)
        assert result['rows_written'] == 3

    def test_compact_creates_silver_output(self, temp_lake):
        """Compaction should create Silver parquet files."""
        sc = SilverCompactor(data_root=temp_lake)

        sc.compact_date(
            date='2025-12-16',
            schema='futures.trades',
            partition_value='ES'
        )

        silver_path = os.path.join(
            temp_lake,
            'silver',
            'futures',
            'trades',
            'symbol=ES',
            'date=2025-12-16',
            'hour=14'
        )

        assert os.path.exists(silver_path)
        assert len(os.listdir(silver_path)) > 0

    def test_compact_sorts_by_ts_event(self, temp_lake):
        """Output should be sorted by ts_event_ns."""
        sc = SilverCompactor(data_root=temp_lake)

        sc.compact_date(
            date='2025-12-16',
            schema='futures.trades',
            partition_value='ES'
        )

        # Read Silver output
        sr = SilverReader(data_root=temp_lake)
        df = sr.read_futures_trades(symbol='ES', date='2025-12-16')

        assert len(df) == 3
        assert df['ts_event_ns'].is_monotonic_increasing

    def test_skip_if_no_bronze(self):
        """Should skip gracefully if no Bronze data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = os.path.join(tmpdir, 'lake')
            sc = SilverCompactor(data_root=data_root)

            result = sc.compact_date(
                date='2025-12-16',
                schema='futures.trades',
                partition_value='ES'
            )

            assert result['status'] == 'skipped'
            assert 'not found' in result['reason']

    def test_skip_if_silver_exists(self, temp_lake):
        """Should skip if Silver already exists (unless force=True)."""
        sc = SilverCompactor(data_root=temp_lake)

        # First compaction
        result1 = sc.compact_date(
            date='2025-12-16',
            schema='futures.trades',
            partition_value='ES'
        )
        assert result1['status'] == 'success'

        # Second compaction without force
        result2 = sc.compact_date(
            date='2025-12-16',
            schema='futures.trades',
            partition_value='ES',
            force=False
        )
        assert result2['status'] == 'skipped'
        assert 'already exists' in result2['reason']

        # Third compaction with force
        result3 = sc.compact_date(
            date='2025-12-16',
            schema='futures.trades',
            partition_value='ES',
            force=True
        )
        assert result3['status'] == 'success'


class TestSilverReader:
    """Tests for SilverReader."""

    @pytest.fixture
    def silver_with_data(self):
        """Create Silver directory with test data."""
        tmpdir = tempfile.mkdtemp()
        data_root = os.path.join(tmpdir, 'lake')

        silver_path = os.path.join(
            data_root,
            'silver',
            'futures',
            'trades',
            'symbol=ES',
            'date=2025-12-16',
            'hour=14'
        )
        os.makedirs(silver_path)

        df = pd.DataFrame({
            'ts_event_ns': [
                1734357600_000_000_000,
                1734357601_000_000_000,
                1734357602_000_000_000,
            ],
            'ts_recv_ns': [
                1734357600_001_000_000,
                1734357601_001_000_000,
                1734357602_001_000_000,
            ],
            'source': ['sim', 'sim', 'sim'],
            'symbol': ['ES', 'ES', 'ES'],
            'price': [6870.0, 6870.25, 6870.50],
            'size': [10, 15, 20],
        })

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            os.path.join(silver_path, 'part-0000.parquet'),
            compression='zstd'
        )

        yield data_root

        shutil.rmtree(tmpdir)

    def test_read_futures_trades(self, silver_with_data):
        """Should read futures trades from Silver."""
        sr = SilverReader(data_root=silver_with_data)
        df = sr.read_futures_trades(symbol='ES', date='2025-12-16')

        assert len(df) == 3
        assert 'ts_event_ns' in df.columns
        assert 'price' in df.columns

    def test_read_with_time_filter(self, silver_with_data):
        """Should filter by ts_event_ns range."""
        sr = SilverReader(data_root=silver_with_data)

        # Read only first two rows
        df = sr.read_futures_trades(
            symbol='ES',
            date='2025-12-16',
            end_ns=1734357601_500_000_000
        )

        assert len(df) == 2

    def test_get_available_dates(self, silver_with_data):
        """Should list available dates."""
        sr = SilverReader(data_root=silver_with_data)
        dates = sr.get_available_dates('futures/trades', 'symbol=ES')

        assert '2025-12-16' in dates
