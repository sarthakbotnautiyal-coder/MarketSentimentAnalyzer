"""Tests for display module."""

import pytest
from io import StringIO
from src.display import Display


def test_print_indicators_with_data(capsys):
    """Test printing indicators table."""
    indicators = {
        'Current_Price': 150.0,
        'RSI_14': 65.5,
        'SMA_50': 148.0,
        'EMA_50': 149.0
    }
    Display.print_indicators("AAPL", indicators)

    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "Current_Price" in captured.out
    assert "RSI_14" in captured.out


def test_print_indicators_with_error(capsys):
    """Test printing indicators with error."""
    indicators = {"error": "Failed to fetch data"}
    Display.print_indicators("AAPL", indicators)

    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "Failed to fetch data" in captured.out


def test_print_news_with_articles(capsys):
    """Test printing news articles."""
    articles = [
        {
            'title': 'Test Article 1',
            'url': 'http://example.com/1',
            'snippet': 'This is a test summary that is long enough to be truncated',
            'published': '2024-01-01'
        },
        {
            'title': 'Test Article 2',
            'url': 'http://example.com/2',
            'snippet': 'Another test summary',
            'published': '2024-01-02'
        }
    ]
    Display.print_news("AAPL", articles)

    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "Test Article 1" in captured.out
    assert "Test Article 2" in captured.out
    assert "http://example.com/1" in captured.out


def test_print_news_empty(capsys):
    """Test printing empty news."""
    Display.print_news("AAPL", [])

    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "No news" in captured.out


def test_print_sentiment(capsys):
    """Test printing sentiment."""
    sentiment = {
        'sentiment': 'bullish',
        'confidence': 0.85,
        'explanation': 'Positive earnings report'
    }
    Display.print_sentiment("AAPL", sentiment)

    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    assert "BULLISH" in captured.out or "bullish" in captured.out.lower()
    assert "0.85" in captured.out


def test_print_summary(capsys):
    """Test summary printing."""
    results = {
        "AAPL": {
            "indicators": {
                "Current_Price": 150.0,
                "Change": 2.5,
                "RSI_14": 65.5
            },
            "sentiment": {"sentiment": "bullish"}
        },
        "MSFT": {
            "indicators": {
                "Current_Price": 300.0,
                "Change": -1.0,
                "RSI_14": 45.0
            },
            "sentiment": {"sentiment": "neutral"}
        }
    }
    Display.print_summary(results)

    captured = capsys.readouterr()
    assert "SUMMARY" in captured.out
    assert "AAPL" in captured.out
    assert "MSFT" in captured.out
    assert "BULLISH" in captured.out or "bullish" in captured.out.lower()
