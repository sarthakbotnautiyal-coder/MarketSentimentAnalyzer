"""Tests for data fetchers with caching."""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
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
    dates = pd.date_range('2024-01-01', periods=3)
    mock_history = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0],
        'Volume': [1000000, 1100000, 1200000],
        'High': [101.0, 102.0, 103.0],
        'Low': [99.0, 100.0, 101.0],
        'Open': [99.5, 100.5, 101.5]
    }, index=dates)
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
    df = pd.DataFrame({'Close': [100, 101], 'Volume': [1000, 2000]})
    fetcher = StockDataFetcher(db_manager=None)

    indicators = fetcher.calculate_indicators(df)

    assert 'error' in indicators or 'Current_Price' in indicators


def test_stock_fetcher_backfill_1year(mocker):
    """Test 1-year backfill."""
    dates = pd.date_range('2024-01-01', periods=252)
    mock_data = pd.DataFrame({
        'Close': [100.0] * 252,
        'Volume': [1000] * 252
    }, index=dates)
    mock_ticker = Mock()
    mock_ticker.history.return_value = mock_data
    mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.backfill_1year("AAPL")

    assert result is not None


def test_stock_fetcher_uses_cache(mocker, sample_df):
    """Test that fetcher uses cached data when fresh."""
    mock_db = MagicMock()
    mock_db.get_latest_stock_date.return_value = '2024-12-31'  # Recent date
    mock_db.get_stock_data.return_value = sample_df

    fetcher = StockDataFetcher(db_manager=mock_db)
    result = fetcher.fetch_data("AAPL")

    assert result is not None
    mock_db.get_stock_data.assert_called_once_with("AAPL")


def test_stock_fetcher_force_refresh_bypasses_cache(mocker):
    """Test that force_refresh bypasses cache."""
    dates = pd.date_range('2024-01-01', periods=30)
    mock_data = pd.DataFrame({
        'Close': [100.0] * 30,
        'Volume': [1000] * 30
    }, index=dates)
    mock_ticker = Mock()
    mock_ticker.history.return_value = mock_data
    mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

    mock_db = MagicMock()
    mock_db.get_latest_stock_date.return_value = '2024-12-31'

    fetcher = StockDataFetcher(db_manager=mock_db)
    result = fetcher.fetch_data("AAPL", force_refresh=True)

    assert result is not None
    mock_db.get_stock_data.assert_not_called()


def test_stock_fetcher_delta_fetch_no_cache(mocker):
    """Test delta fetching when no cached data exists."""
    dates = pd.date_range('2024-01-01', periods=252)
    mock_data = pd.DataFrame({
        'Close': [100.0] * 252,
        'Volume': [1000] * 252
    }, index=dates)
    mock_ticker = Mock()
    mock_ticker.history.return_value = mock_data
    mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.fetch_delta("AAPL")

    assert result is not None
