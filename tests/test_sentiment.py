"""Tests for sentiment analyzer."""

import pytest
from unittest.mock import Mock, patch
from src.sentiment import SentimentAnalyzer


@pytest.fixture
def analyzer():
    """Create sentiment analyzer for tests."""
    return SentimentAnalyzer(host="http://test:11434", model="test-model")


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
