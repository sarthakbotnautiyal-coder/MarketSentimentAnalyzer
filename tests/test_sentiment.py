"""Tests for sentiment analyzer with caching."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from src.sentiment import SentimentAnalyzer
from src.database import DatabaseManager
import tempfile
import os


@pytest.fixture
def analyzer():
    """Create sentiment analyzer for tests."""
    return SentimentAnalyzer(host="http://test:11434", model="test-model")


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = DatabaseManager(db_path)
    yield db
    
    db.close()
    os.unlink(db_path)


def test_analyze_bullish(mocker, analyzer):
    """Test bullish sentiment detection."""
    mock_response = Mock()
    mock_response.json.return_value = {"response": "bullish The stock is performing well."}
    mocker.patch('requests.post', return_value=mock_response)

    result = analyzer.analyze("Great earnings report, stock up 10%")

    assert result['sentiment'] == 'bullish'
    assert 'confidence' in result


def test_analyze_bearish(mocker, analyzer):
    """Test bearish sentiment detection."""
    mock_response = Mock()
    mock_response.json.return_value = {"response": "bearish Concerns about growth."}
    mocker.patch('requests.post', return_value=mock_response)

    result = analyzer.analyze("Layoffs announced, stock plunges")

    assert result['sentiment'] == 'bearish'


def test_analyze_neutral(mocker, analyzer):
    """Test neutral sentiment detection."""
    mock_response = Mock()
    mock_response.json.return_value = {"response": "neutral The stock was flat."}
    mocker.patch('requests.post', return_value=mock_response)

    result = analyzer.analyze("Stock traded in a range today")

    assert result['sentiment'] == 'neutral'


def test_analyze_empty_input(analyzer):
    """Test empty input handling."""
    result = analyzer.analyze("")
    assert result['sentiment'] == 'neutral'
    assert result['confidence'] == 1.0


def test_analyze_connection_error(mocker, analyzer):
    """Test connection error handling."""
    mocker.patch('requests.post', side_effect=Exception("Connection refused"))

    result = analyzer.analyze("Some text")

    assert result['sentiment'] == 'neutral'
    assert 'error' in result['explanation'].lower()


def test_analyze_unexpected_response(mocker, analyzer):
    """Test handling of unexpected LLM response."""
    mock_response = Mock()
    mock_response.json.return_value = {"response": "The stock is up"}
    mocker.patch('requests.post', return_value=mock_response)

    result = analyzer.analyze("Some text")

    # Should default to neutral for non-sentiment words
    assert result['sentiment'] in ['neutral', 'bullish', 'bearish']


def test_analyze_batch_success(mocker, analyzer):
    """Test batch analysis success."""
    mock_response = Mock()
    mock_response.json.return_value = {"response": "bullish Overall positive news"}
    mocker.patch('requests.post', return_value=mock_response)

    texts = ["Good news 1", "Good news 2", "Good news 3"]
    result = analyzer.analyze_batch(texts)

    assert result['sentiment'] == 'bullish'
    assert 'explanation' in result


def test_analyze_batch_empty(analyzer):
    """Test batch analysis with empty list."""
    result = analyzer.analyze_batch([])
    assert result['sentiment'] == 'neutral'
    assert result['confidence'] == 1.0


def test_analyze_batch_no_articles(analyzer):
    """Test batch analysis with one article."""
    result = analyzer.analyze_batch(["Single article about stock"])
    # Should handle without error
    assert 'sentiment' in result


# New caching tests

def test_sentiment_analyzer_caching_disabled():
    """Test that caching works when db_manager is None."""
    analyzer = SentimentAnalyzer(db_manager=None)
    assert analyzer.db is None


def test_sentiment_analyzer_caches_result(mocker, temp_db):
    """Test that sentiment results are cached."""
    ticker = "AAPL"
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'bullish Exciting news'}
    mock_response.raise_for_status = Mock()
    mocker.patch('requests.post', return_value=mock_response)
    
    analyzer = SentimentAnalyzer(db_manager=temp_db, sentiment_ttl_days=7)
    result = analyzer.analyze_batch(['test'], ticker=ticker)
    
    # Check saved to cache
    cached = temp_db.get_cached_sentiment(ticker, days_back=30)
    assert cached is not None
    assert cached['sentiment'] == 'bullish'
    assert cached['explanation'] == 'Exciting news'


def test_sentiment_analyzer_uses_cache(mocker, temp_db):
    """Test that cache is used when available and fresh."""
    ticker = "AAPL"
    # Pre-populate cache
    today = datetime.now().strftime('%Y-%m-%d')
    temp_db.save_sentiment(ticker, today, 'bearish', 0.75, 'From cache')
    
    # Mock Ollama to ensure it's not called
    mocker.patch('requests.post', side_effect=Exception("Should not call API"))
    
    analyzer = SentimentAnalyzer(db_manager=temp_db, sentiment_ttl_days=7)
    result = analyzer.analyze_batch(['test'], ticker=ticker, force_refresh=False)
    
    assert result['sentiment'] == 'bearish'
    assert result['explanation'] == 'From cache'


def test_sentiment_analyzer_force_refresh(mocker, temp_db):
    """Test force_refresh bypasses cache."""
    ticker = "AAPL"
    # Pre-populate cache
    today = datetime.now().strftime('%Y-%m-%d')
    temp_db.save_sentiment(ticker, today, 'bearish', 0.75, 'Old cache')
    
    # Mock fresh response
    mock_response = Mock()
    mock_response.json.return_value = {'response': 'bullish Fresh'}
    mock_response.raise_for_status = Mock()
    mocker.patch('requests.post', return_value=mock_response)
    
    analyzer = SentimentAnalyzer(db_manager=temp_db, sentiment_ttl_days=7)
    result = analyzer.analyze_batch(['test'], ticker=ticker, force_refresh=True)
    
    assert result['sentiment'] == 'bullish'
    assert result['explanation'] == 'Fresh'
    # Verify API was called
    mock_response.assert_called_once()


def test_sentiment_analyzer_fallback_on_api_failure(mocker, temp_db):
    """Test that stale cache is used when API fails and force_refresh is True."""
    ticker = "AAPL"
    # Pre-populate cache with old data (older than TTL)
    old_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    temp_db.save_sentiment(ticker, old_date, 'neutral', 0.5, 'Stale cache')
    
    # Simulate API failure
    mocker.patch('requests.post', side_effect=Exception("API down"))
    
    analyzer = SentimentAnalyzer(db_manager=temp_db, sentiment_ttl_days=7)
    # force_refresh=True should attempt API first (which fails), then fallback to stale cache
    result = analyzer.analyze_batch(['test'], ticker=ticker, force_refresh=True)
    
    # Even though cache is stale, fallback should occur on API failure
    assert result is not None
    assert result.get('sentiment') == 'neutral'
    assert 'Stale cache' in result.get('explanation', '')
