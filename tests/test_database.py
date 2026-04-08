"""Tests for database module with indicators-only schema."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.database import DatabaseManager


@pytest.fixture
def db_path(tmp_path):
    """Provide temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def test_db(db_path):
    """Create a test database manager."""
    db = DatabaseManager(str(db_path))
    yield db
    db.close()


@pytest.fixture
def sample_stock_df():
    """Provide sample stock data DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=5, freq='D')
    df = pd.DataFrame({
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'High': [102.0, 103.0, 104.0, 105.0, 106.0],
        'Low': [99.0, 100.0, 101.0, 102.0, 103.0],
        'Close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'Volume': [1000, 2000, 1500, 3000, 2500]
    }, index=dates)
    return df


def test_database_initialization(db_path):
    """Test database creates required tables."""
    db = DatabaseManager(str(db_path))
    
    # Check tables exist
    cursor = db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row['name'] for row in cursor.fetchall()}
    
    assert 'stock_daily' in tables
    assert 'indicators' in tables
    db.close()


def test_save_and_get_stock_data(test_db, sample_stock_df):
    """Test saving and retrieving stock data."""
    ticker = 'AAPL'
    
    # Save data
    test_db.save_stock_data(ticker, sample_stock_df)
    
    # Retrieve data
    retrieved = test_db.get_stock_data(ticker)
    
    assert len(retrieved) == len(sample_stock_df)
    assert 'Open' in retrieved.columns
    assert 'Close' in retrieved.columns


def test_get_latest_stock_date(test_db, sample_stock_df):
    """Test retrieving latest date for ticker."""
    ticker = 'AAPL'
    
    # Initially no data
    assert test_db.get_latest_stock_date(ticker) is None
    
    # Save data
    test_db.save_stock_data(ticker, sample_stock_df)
    
    # Check latest date
    latest = test_db.get_latest_stock_date(ticker)
    expected_date = sample_stock_df.index.max().strftime('%Y-%m-%d')
    assert latest == expected_date


def test_truncate_indicators(test_db):
    """Test truncating indicators table."""
    ticker = 'AAPL'
    date = '2024-01-15'
    indicators = {'RSI_14': 65.5, 'MACD': 0.5}
    
    # Insert indicator
    test_db.insert_latest_indicator(ticker, date, indicators)
    assert test_db.count_indicators() == 1
    
    # Truncate
    deleted = test_db.truncate_indicators()
    assert deleted == 1
    assert test_db.count_indicators() == 0


def test_insert_and_retrieve_indicators(test_db):
    """Test inserting and retrieving indicators."""
    ticker = 'AAPL'
    date = '2024-01-15'
    indicators = {
        'RSI_14': 65.5,
        'MACD': 0.5,
        'MACD_Hist': 0.3,
        'SMA_20': 150.0,
        'Current_Price': 155.0
    }
    
    test_db.insert_latest_indicator(ticker, date, indicators)
    
    # Retrieve via get_all_latest_indicators
    all_latest = test_db.get_all_latest_indicators()
    assert len(all_latest) == 1
    assert all_latest[0][0] == ticker
    assert all_latest[0][1] == date


def test_data_upsert(test_db, sample_stock_df):
    """Test that saving duplicates updates rather than duplicates."""
    ticker = 'AAPL'
    
    # Save initial data
    test_db.save_stock_data(ticker, sample_stock_df)
    
    # Modify and re-save
    modified_df = sample_stock_df.copy()
    modified_df.loc[modified_df.index[0], 'Close'] = 999.0
    test_db.save_stock_data(ticker, modified_df)
    
    # Should still have same number of rows
    retrieved = test_db.get_stock_data(ticker)
    assert len(retrieved) == len(sample_stock_df)


def test_is_data_fresh(test_db, sample_stock_df):
    """Test data freshness check."""
    ticker = 'AAPL'
    
    # Initially not fresh
    assert not test_db.is_data_fresh('stock_daily', ticker, ttl_days=1)
    
    # Save recent data
    recent_df = pd.DataFrame({
        'Open': [100.0],
        'High': [102.0],
        'Low': [99.0],
        'Close': [101.0],
        'Volume': [1000]
    }, index=[pd.Timestamp.today()])
    test_db.save_stock_data(ticker, recent_df)
    
    # Should be fresh now
    assert test_db.is_data_fresh('stock_daily', ticker, ttl_days=1)


def test_clear_old_stock_data(test_db, sample_stock_df):
    """Test clearing old stock data."""
    ticker = 'AAPL'
    test_db.save_stock_data(ticker, sample_stock_df)
    
    # Clear data older than 1 day (should delete all)
    deleted = test_db.clear_old_stock_data(days=1)
    assert deleted == len(sample_stock_df)
    
    # Verify cleared
    data = test_db.get_stock_data(ticker)
    assert len(data) == 0


def test_database_migration_drops_deprecated_tables(test_db):
    """Test that deprecated tables are dropped during initialization."""
    cursor = test_db.conn.cursor()
    
    # Create deprecated tables
    cursor.execute('CREATE TABLE IF NOT EXISTS news (id INTEGER PRIMARY KEY)')
    cursor.execute('CREATE TABLE IF NOT EXISTS sentiment (id INTEGER PRIMARY KEY)')
    test_db.conn.commit()
    
    # Re-initialize to trigger migration
    db2 = DatabaseManager(test_db.db_path)
    
    # Verify tables are dropped
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row['name'] for row in cursor.fetchall()}
    
    assert 'news' not in tables
    assert 'sentiment' not in tables
    db2.close()
