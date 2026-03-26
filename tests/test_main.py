"""Integration tests for main module."""

import pytest
from unittest.mock import patch, Mock
from src.config import Config
from src.main import MarketSentimentAnalyzer, main


def test_market_sentiment_analyzer_initialization():
    """Test analyzer initialization."""
    config = Config(
        tickers=["AAPL"],
        brave_api_key="test_key",
        ollama_host="http://test:11434",
        ollama_model="test-model"
    )
    analyzer = MarketSentimentAnalyzer(config)

    assert analyzer.config == config
    assert analyzer.stock_fetcher is not None
    assert analyzer.news_fetcher is not None
    assert analyzer.sentiment_analyzer is not None
    assert analyzer.display is not None


def test_run_single_ticker(mocker, tmp_path):
    """Test running analysis for single ticker."""
    # Mock config
    config = Config(
        tickers=["AAPL"],
        brave_api_key="test_key",
        ollama_host="http://test:11434",
        ollama_model="test-model"
    )

    # Mock stock data
    mock_df = Mock()
    mock_df.__len__ = Mock(return_value=200)
    mock_df.columns = ['Close']
    mock_close = [100.0 + i for i in range(200)]
    mock_df['Close'] = mock_close
    mock_df['Close'].iloc[-1] = 299.0

    # Mock indicators calculation
    mock_indicators = {
        'Current_Price': 299.0,
        'RSI_14': 55.0,
        'MACD': 1.0,
        'MACD_Signal': 0.5,
        'MACD_Hist': 0.5,
        'SMA_5': 295.0,
        'EMA_5': 296.0,
        'Change': 1.5
    }

    analyzer = MarketSentimentAnalyzer(config)
    mocker.patch.object(analyzer.stock_fetcher, 'get_indicators', return_value=mock_indicators)
    mocker.patch.object(analyzer.news_fetcher, 'fetch_news', return_value=[
        {'title': 'Test News', 'snippet': 'Test summary', 'url': 'http://test.com'}
    ])
    mocker.patch.object(analyzer.sentiment_analyzer, 'analyze_batch', return_value={
        'sentiment': 'bullish',
        'confidence': 0.9,
        'explanation': 'Positive news'
    })

    results = analyzer.run()

    assert "AAPL" in results
    assert results["AAPL"]["indicators"] == mock_indicators
    assert len(results["AAPL"]["news"]) == 1
    assert results["AAPL"]["sentiment"]["sentiment"] == "bullish"


def test_run_multiple_tickers(mocker):
    """Test running analysis for multiple tickers."""
    config = Config(
        tickers=["AAPL", "MSFT"],
        brave_api_key="test_key",
        ollama_host="http://test:11434",
        ollama_model="test-model"
    )

    analyzer = MarketSentimentAnalyzer(config)

    def mock_get_indicators(ticker):
        return {'Current_Price': 100.0, 'RSI_14': 50.0}

    def mock_fetch_news(query):
        return [{'title': 'News for ' + query, 'snippet': 'Summary', 'url': ''}]

    mocker.patch.object(analyzer.stock_fetcher, 'get_indicators', side_effect=mock_get_indicators)
    mocker.patch.object(analyzer.news_fetcher, 'fetch_news', side_effect=mock_fetch_news)
    mocker.patch.object(analyzer.sentiment_analyzer, 'analyze_batch', return_value={
        'sentiment': 'neutral',
        'confidence': 0.5,
        'explanation': 'Meh'
    })

    results = analyzer.run()

    assert len(results) == 2
    assert "AAPL" in results
    assert "MSFT" in results


def test_main_success(mocker):
    """Test main entry point success."""
    mock_config = Config(
        tickers=["AAPL"],
        brave_api_key="test_key",
        ollama_host="http://test:11434"
    )

    mocker.patch('src.main.Config.load', return_value=mock_config)
    mocker.patch('src.main.MarketSentimentAnalyzer', return_value=Mock(run=Mock(return_value={})))

    exit_code = main()

    assert exit_code == 0


def test_main_configuration_error(mocker):
    """Test main entry point with configuration error."""
    mocker.patch('src.main.Config.load', side_effect=ValueError("Missing BRAVE_API_KEY"))

    exit_code = main()

    assert exit_code == 1


def test_main_keyboard_interrupt(mocker):
    """Test main entry point with keyboard interrupt."""
    mocker.patch('src.main.Config.load', side_effect=KeyboardInterrupt)

    exit_code = main()

    assert exit_code == 130


def test_main_unexpected_error(mocker):
    """Test main entry point with unexpected error."""
    mocker.patch('src.main.Config.load', side_effect=Exception("Unexpected"))

    exit_code = main()

    assert exit_code == 1
