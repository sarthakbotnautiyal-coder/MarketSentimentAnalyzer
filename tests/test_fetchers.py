"""Tests for data fetchers."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.fetchers import StockDataFetcher, NewsFetcher


@pytest.fixture
def sample_df():
    """Create sample stock data DataFrame."""
    data = {
        'Open': [100.0, 101.0, 102.0, 103.0, 104.0] * 40,
        'High': [101.0, 102.0, 103.0, 104.0, 105.0] * 40,
        'Low': [99.0, 100.0, 101.0, 102.0, 103.0] * 40,
        'Close': [100.5, 101.5, 102.5, 103.5, 104.5] * 40,
        'Volume': [1000000] * 200
    }
    df = pd.DataFrame(data)
    return df


def test_stock_fetcher_fetch_data_success(mocker):
    """Test successful data fetch."""
    mock_ticker = Mock()
    mock_history = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0],
        'Volume': [1000000, 1100000, 1200000]
    })
    mock_ticker.history.return_value = mock_history
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher()
    result = fetcher.fetch_data("AAPL")

    assert result is not None
    assert len(result) == 3
    assert 'Close' in result.columns


def test_stock_fetcher_fetch_data_failure(mocker):
    """Test fetch data failure."""
    mocker.patch('yfinance.Ticker', side_effect=Exception("Network error"))

    fetcher = StockDataFetcher()
    result = fetcher.fetch_data("AAPL")

    assert result is None


def test_stock_fetcher_calculate_indicators(sample_df):
    """Test indicator calculation."""
    fetcher = StockDataFetcher()
    indicators = fetcher.calculate_indicators(sample_df)

    assert 'Current_Price' in indicators
    assert 'RSI_14' in indicators
    assert 'MACD' in indicators
    assert 'SMA_5' in indicators
    assert 'EMA_5' in indicators
    assert 'SMA_200' in indicators
    assert 'EMA_200' in indicators

    # Verify values are not None (with enough data)
    assert indicators['Current_Price'] is not None
    assert indicators['RSI_14'] is not None


def test_stock_fetcher_calculate_indicators_insufficient_data():
    """Test indicator calculation with insufficient data."""
    small_df = pd.DataFrame({
        'Close': [100.0, 101.0, 102.0, 103.0]  # Only 4 rows, not enough for SMA_200
    })

    fetcher = StockDataFetcher()
    indicators = fetcher.calculate_indicators(small_df)

    # Should still have some indicators but not all
    assert 'Current_Price' in indicators


def test_stock_fetcher_get_indicators(mocker, sample_df):
    """Test get_indicators method."""
    mocker.patch('yfinance.Ticker')  # Return Mock
    mock_ticker = mocker.patch('yfinance.Ticker').return_value
    mock_ticker.history.return_value = sample_df

    fetcher = StockDataFetcher()
    result = fetcher.get_indicators("AAPL")

    assert "error" not in result
    assert 'Current_Price' in result


def test_news_fetcher_fetch_success(mocker):
    """Test successful news fetch."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "web": {
            "results": [
                {
                    "title": "Test News 1",
                    "url": "http://example.com/1",
                    "snippet": "Summary 1",
                    "published_time": "2024-01-01"
                },
                {
                    "title": "Test News 2",
                    "url": "http://example.com/2",
                    "snippet": "Summary 2",
                    "published_time": "2024-01-02"
                }
            ]
        }
    }
    mock_response.raise_for_status = Mock()
    mocker.patch('requests.get', return_value=mock_response)

    fetcher = NewsFetcher("test_api_key")
    articles = fetcher.fetch_news("AAPL stock news")

    assert len(articles) == 2
    assert articles[0]['title'] == "Test News 1"
    assert articles[0]['snippet'] == "Summary 1"


def test_news_fetcher_fetch_failure(mocker):
    """Test news fetch failure."""
    mocker.patch('requests.get', side_effect=Exception("Network error"))

    fetcher = NewsFetcher("test_api_key")
    articles = fetcher.fetch_news("AAPL stock news")

    assert articles == []


def test_news_fetcher_empty_results(mocker):
    """Test news fetch with empty results."""
    mock_response = Mock()
    mock_response.json.return_value = {"web": {"results": []}}
    mock_response.raise_for_status = Mock()
    mocker.patch('requests.get', return_value=mock_response)

    fetcher = NewsFetcher("test_api_key")
    articles = fetcher.fetch_news("AAPL stock news")

    assert len(articles) == 0
