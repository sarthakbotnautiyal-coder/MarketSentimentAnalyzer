"""Tests for data fetchers with caching."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.fetchers import StockDataFetcher, NewsFetcher
from src.database import DatabaseManager


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
        'Volume': [1000000, 1100000, 1200000]
    })
    mock_ticker.history.return_value = mock_history
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.fetch_data("AAPL")

    assert result is not None
    assert len(result) == 3
    assert 'Close' in result.columns


def test_stock_fetcher_fetch_data_failure(mocker):
    """Test fetch data failure."""
    mocker.patch('yfinance.Ticker', side_effect=Exception("Network error"))

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.fetch_data("AAPL")

    assert result is None


def test_stock_fetcher_calculate_indicators(sample_df):
    """Test indicator calculation."""
    fetcher = StockDataFetcher(db_manager=None)
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

    fetcher = StockDataFetcher(db_manager=None)
    indicators = fetcher.calculate_indicators(small_df)

    # Should still have some indicators but not all
    assert 'Current_Price' in indicators


def test_stock_fetcher_get_indicators(mocker, sample_df):
    """Test get_indicators method."""
    mocker.patch('yfinance.Ticker')  # Return Mock
    mock_ticker = mocker.patch('yfinance.Ticker').return_value
    mock_ticker.history.return_value = sample_df

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.get_indicators("AAPL")

    assert "error" not in result
    assert 'Current_Price' in result


def test_stock_fetcher_backfill_1year(mocker):
    """Test backfill functionality."""
    mock_ticker = Mock()
    mock_history = pd.DataFrame({
        'Close': [100.0 + i for i in range(252)],  # ~1 year of data
        'Volume': [1000000 for _ in range(252)]
    })
    mock_ticker.history.return_value = mock_history
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)

    fetcher = StockDataFetcher(db_manager=None)
    result = fetcher.backfill_1year("AAPL")

    assert result is not None
    assert len(result) == 252


def test_stock_fetcher_uses_cache(mocker, temp_db, sample_df):
    """Test that fetcher uses cached data when available."""
    # Pre-populate cache
    temp_db.save_stock_data("AAPL", sample_df)
    
    # Mock yfinance to ensure it's not called
    mocker.patch('yfinance.Ticker', side_effect=Exception("Should not call API"))
    
    fetcher = StockDataFetcher(db_manager=temp_db)
    result = fetcher.fetch_data("AAPL", force_refresh=False)
    
    assert result is not None
    assert not result.empty
    # Should use cached data, so yfinance should not be called
    # The mock will raise if called


def test_stock_fetcher_force_refresh_bypasses_cache(mocker, temp_db, sample_df):
    """Test that force_refresh bypasses cache."""
    # Pre-populate cache with old data
    temp_db.save_stock_data("AAPL", sample_df)
    
    # Mock fresh data from API
    fresh_df = pd.DataFrame({
        'Close': [200.0, 201.0, 202.0],
        'Volume': [2000000, 2100000, 2200000]
    })
    mock_ticker = Mock()
    mock_ticker.history.return_value = fresh_df
    mocker.patch('yfinance.Ticker', return_value=mock_ticker)
    
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
    mocker.patch('yfinance.Ticker', side_effect=Exception("API down"))
    
    fetcher = StockDataFetcher(db_manager=temp_db)
    result = fetcher.fetch_data("AAPL", force_refresh=True)
    
    # Should fall back to cached data
    assert result is not None
    assert not result.empty
    assert len(result) == len(sample_df)


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

    fetcher = NewsFetcher("test_api_key", db_manager=None)
    articles = fetcher.fetch_news("AAPL stock news")

    assert len(articles) == 2
    assert articles[0]['title'] == "Test News 1"
    assert articles[0]['snippet'] == "Summary 1"


def test_news_fetcher_fetch_failure(mocker):
    """Test news fetch failure."""
    mocker.patch('requests.get', side_effect=Exception("Network error"))

    fetcher = NewsFetcher("test_api_key", db_manager=None)
    articles = fetcher.fetch_news("AAPL stock news")

    assert articles == []


def test_news_fetcher_empty_results(mocker):
    """Test news fetch with empty results."""
    mock_response = Mock()
    mock_response.json.return_value = {"web": {"results": []}}
    mock_response.raise_for_status = Mock()
    mocker.patch('requests.get', return_value=mock_response)

    fetcher = NewsFetcher("test_api_key", db_manager=None)
    articles = fetcher.fetch_news("AAPL stock news")

    assert len(articles) == 0


def test_news_fetcher_uses_cache(mocker, temp_db):
    """Test that news fetcher uses cached data."""
    ticker = "AAPL"
    articles = [
        {'title': 'Cached', 'url': 'http://cached.com', 'snippet': 'From cache'}
    ]
    temp_db.save_news(ticker, articles)
    
    # Mock API to ensure not called
    mocker.patch('requests.get', side_effect=Exception("Should not call API"))
    
    fetcher = NewsFetcher("test_api_key", db_manager=temp_db, news_ttl_days=7)
    result = fetcher.fetch_news("AAPL news", ticker=ticker, force_refresh=False)
    
    assert len(result) >= 1
    assert result[0]['title'] == 'Cached'


def test_news_fetcher_force_refresh_bypasses_cache(mocker, temp_db):
    """Test force_refresh bypasses news cache."""
    ticker = "AAPL"
    old_articles = [{'title': 'Old', 'url': 'http://old.com', 'snippet': 'Old'}]
    temp_db.save_news(ticker, old_articles)
    
    fresh_articles = [
        {'title': 'Fresh', 'url': 'http://fresh.com', 'snippet': 'Fresh'}
    ]
    mock_response = Mock()
    mock_response.json.return_value = {"web": {"results": [
        {'title': 'Fresh', 'url': 'http://fresh.com', 'snippet': 'Fresh', 'published_time': '2024-01-01'}
    ]}}
    mock_response.raise_for_status = Mock()
    mocker.patch('requests.get', return_value=mock_response)
    
    fetcher = NewsFetcher("test_api_key", db_manager=temp_db, news_ttl_days=7)
    result = fetcher.fetch_news("AAPL news", ticker=ticker, force_refresh=True)
    
    # Should include fresh data (plus maybe old if fresh API returned multiple)
    assert any(a['title'] == 'Fresh' for a in result)
    mock_response.assert_called_once()


def test_sentiment_analyzer_caching(mocker, temp_db):
    """Test that sentiment analyzer uses cache."""
    ticker = "AAPL"
    # Pre-populate cache with fresh enough data
    temp_db.save_sentiment(
        ticker,
        datetime.now().strftime('%Y-%m-%d'),
        'bullish',
        0.85,
        'From cache'
    )
    
    # Mock Ollama to ensure not called
    mocker.patch('requests.post', side_effect=Exception("Should not call Ollama"))
    
    analyzer = SentimentAnalyzer(db_manager=temp_db, sentiment_ttl_days=7)
    result = analyzer.analyze_batch(['test'], ticker=ticker, force_refresh=False)
    
    assert result['sentiment'] == 'bullish'
    assert result['explanation'] == 'From cache'


def test_sentiment_analyzer_force_refresh_bypasses_cache(mocker, temp_db):
    """Test force_refresh fetches new sentiment."""
    ticker = "AAPL"
    # Pre-populate cache
    temp_db.save_sentiment(
        ticker,
        datetime.now().strftime('%Y-%m-%d'),
        'bearish',
        0.5,
        'Old'
    )
    
    # Mock Ollama response
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'bullish Fresh analysis'}
    mock_response.raise_for_status = Mock()
    mocker.patch('requests.post', return_value=mock_response)
    
    analyzer = SentimentAnalyzer(db_manager=temp_db, sentiment_ttl_days=7)
    result = analyzer.analyze_batch(['test'], ticker=ticker, force_refresh=True)
    
    assert result['sentiment'] == 'bullish'
    assert 'Fresh' in result['explanation']
    # Verify Ollama was called
    mock_response.assert_called_once()


def test_sentiment_analyzer_saves_to_cache(mocker, temp_db):
    """Test that sentiment results are saved to cache."""
    ticker = "AAPL"
    
    # Mock Ollama response
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'neutral Test explanation'}
    mock_response.raise_for_status = Mock()
    mocker.patch('requests.post', return_value=mock_response)
    
    analyzer = SentimentAnalyzer(db_manager=temp_db, sentiment_ttl_days=7)
    result = analyzer.analyze_batch(['test'], ticker=ticker)
    
    # Check it's saved to database
    cached = temp_db.get_cached_sentiment(ticker, days_back=30)
    assert cached is not None
    assert cached['sentiment'] == 'neutral'
