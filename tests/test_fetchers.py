"""Tests for data fetchers with caching."""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from src.fetchers import StockDataFetcher
from src.database import DatabaseManager


@pytest.fixture
def sample_df():
    """Create sample stock data DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    data = {
        'Open': [100.0 + i for i in range(30)],
        'High': [101.0 + i for i in range(30)],
        'Low': [99.0 + i for i in range(30)],
        'Close': [100.5 + i for i in range(30)],
        'Volume': [1000000] * 30
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    db = DatabaseManager(db_path)
    yield db

    db.close()
    os.unlink(db_path)


def test_stock_fetcher_fetch_data_success(mocker):
    """Test successful data fetch."""
    mock_ticker = Mock()
    mock_history = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0],
        'Volume': [1000000, 1100000, 1200000],
        'High': [101.0, 102.0, 103.0],
        'Low': [99.0, 100.0, 101.0],
        'Open': [99.5, 100.5, 101.5]
    })
    mock_ticker.history.return_value = mock_history
    mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.fetch_data("AAPL")

    assert result is not None
    assert len(result) == 3
    assert 'Close' in result.columns


def test_stock_fetcher_fetch_data_failure(mocker):
    """Test fetch data failure."""
    mocker.patch('src.fetchers.yf.Ticker', side_effect=Exception("Network error"))

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.fetch_data("AAPL")

    assert result is None


def test_stock_fetcher_calculate_indicators(sample_df):
    """Test indicator calculation."""
    fetcher = StockDataFetcher(db_manager=None)
    indicators = fetcher.calculate_indicators(sample_df)

    assert 'Current_Price' in indicators
    assert 'Change' in indicators
    assert 'Change_Pct' in indicators


def test_stock_fetcher_calculate_indicators_insufficient_data():
    """Test indicator calculation with insufficient data."""
    small_df = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0, 103.0]  # Only 4 rows
    })

    fetcher = StockDataFetcher(db_manager=None)
    indicators = fetcher.calculate_indicators(small_df)

    # Should still have some indicators but not all
    assert 'Current_Price' in indicators


def test_stock_fetcher_get_indicators(mocker, sample_df):
    """Test get_indicators method."""
    mocker.patch('src.fetchers.yf.Ticker')
    mock_ticker = mocker.patch('src.fetchers.yf.Ticker').return_value
    mock_ticker.history.return_value = sample_df

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.get_indicators("AAPL")

    assert "error" not in result
    assert 'Current_Price' in result


def test_stock_fetcher_backfill_1year(mocker):
    """Test backfill functionality."""
    mock_ticker = Mock()
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')  # ~1 year
    mock_history = pd.DataFrame({
        'Close': [100.0 + i for i in range(252)],
        'Volume': [1000000 for _ in range(252)],
        'High': [101.0 + i for i in range(252)],
        'Low': [99.0 + i for i in range(252)],
        'Open': [99.5 + i for i in range(252)]
    }, index=dates)
    mock_ticker.history.return_value = mock_history
    mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.backfill_1year("AAPL")

    assert result is not None
    assert len(result) == 252


def test_stock_fetcher_uses_cache(mocker, temp_db, sample_df):
    """Test that fetcher uses cached data when available."""
    # Pre-populate cache
    temp_db.save_stock_data("AAPL", sample_df)

    # Mock yfinance to ensure it's not called
    mocker.patch('src.fetchers.yf.Ticker', side_effect=Exception("Should not call API"))

    fetcher = StockDataFetcher(db_manager=temp_db)
    result = fetcher.fetch_data("AAPL", force_refresh=False)

    assert result is not None
    assert not result.empty


def test_stock_fetcher_force_refresh_bypasses_cache(mocker, temp_db, sample_df):
    """Test that force_refresh bypasses cache."""
    # Pre-populate cache
    temp_db.save_stock_data("AAPL", sample_df)

    # Mock fresh data from API
    fresh_df = sample_df.copy()
    fresh_df['Close'] = fresh_df['Close'] * 2  # Different data

    mock_ticker = Mock()
    mock_ticker.history.return_value = fresh_df.head(3)
    mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher(db_manager=temp_db)
    result = fetcher.fetch_data("AAPL", force_refresh=True)

    assert result is not None
    assert len(result) == 3  # Fresh data, not cached
    # Verify API was called
    mock_ticker.history.assert_called_once()


def test_stock_fetcher_fallback_to_cache_on_failure(mocker, temp_db, sample_df):
    """Test fallback to stale cache when API fails."""
    # Pre-populate cache
    temp_db.save_stock_data("AAPL", sample_df)

    # Simulate API failure
    mocker.patch('src.fetchers.yf.Ticker', side_effect=Exception("API down"))

    fetcher = StockDataFetcher(db_manager=temp_db)
    result = fetcher.fetch_data("AAPL", force_refresh=True)

    # Should fall back to cached data
    assert result is not None
    assert not result.empty
    assert len(result) == len(sample_df)


def test_stock_fetcher_delta_fetch_with_cache(mocker, sample_df):
    """Test delta fetch when cache is current."""
    # Use real DB for this test
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(db_path)
        db.save_stock_data("AAPL", sample_df)

        # Mock returning same data
        mock_ticker = Mock()
        mock_ticker.history.return_value = sample_df.tail(5)
        mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

        fetcher = StockDataFetcher(db_manager=db)
        result = fetcher.fetch_delta("AAPL")

        assert result is not None
        assert len(result) == len(sample_df)
        db.close()
    finally:
        os.unlink(db_path)


def test_stock_fetcher_delta_fetch_no_cache(mocker):
    """Test delta fetch when no cache exists."""
    mock_df = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0],
        'Volume': [1000000, 1100000, 1200000],
        'High': [101.0, 102.0, 103.0],
        'Low': [99.0, 100.0, 101.0],
        'Open': [99.5, 100.5, 101.5]
    })
    mock_ticker = Mock()
    mock_ticker.history.return_value = mock_df
    mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.fetch_delta("AAPL")

    assert result is not None
    assert len(result) == 3
