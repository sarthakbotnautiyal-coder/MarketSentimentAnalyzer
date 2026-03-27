"""Integration tests for main module with database caching."""

import pytest
from unittest.mock import patch, Mock, MagicMock
from src.config import Config, DatabaseConfig
from src.main import MarketSentimentAnalyzer, main, parse_args
from pathlib import Path


class TestParseArgs:
    """Tests for argument parsing."""

    def test_parse_no_args(self):
        """Test parsing with no arguments."""
        with patch('sys.argv', ['main.py']):
            args = parse_args()
        assert args.backfill is None
        assert args.force_refresh is False

    def test_parse_backfill_1y(self):
        """Test parsing --backfill 1y."""
        with patch('sys.argv', ['main.py', '--backfill', '1y']):
            args = parse_args()
        assert args.backfill == '1y'
        assert args.force_refresh is False

    def test_parse_force_refresh(self):
        """Test parsing --force-refresh."""
        with patch('sys.argv', ['main.py', '--force-refresh']):
            args = parse_args()
        assert args.force_refresh is True

    def test_parse_both(self):
        """Test parsing both arguments."""
        with patch('sys.argv', ['main.py', '--backfill', '1y', '--force-refresh']):
            args = parse_args()
        assert args.backfill == '1y'
        assert args.force_refresh is True


class TestMarketSentimentAnalyzer:
    """Tests for main application class."""

    def test_initialization_without_db(self):
        """Test analyzer initialization without database."""
        config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            ollama_host="http://test:11434",
            ollama_model="test-model"
        )
        analyzer = MarketSentimentAnalyzer(config, backfill=False, force_refresh=False)

        assert analyzer.config == config
        assert analyzer.backfill is False
        assert analyzer.force_refresh is False
        assert analyzer.db is None  # No database manager created

    def test_initialization_with_db(self, mocker, tmp_path):
        """Test analyzer initialization with database."""
        db_path = tmp_path / "test.db"
        db_config = DatabaseConfig(path=str(db_path))
        config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            database=db_config
        )

        # Mock DatabaseManager to avoid actual file creation in test
        mock_db = MagicMock()
        mocker.patch('src.main.DatabaseManager', return_value=mock_db)

        analyzer = MarketSentimentAnalyzer(config, backfill=False, force_refresh=False)

        assert analyzer.db is not None
        # Check database path was used
        from src.main import DatabaseManager
        DatabaseManager.assert_called_once_with(str(db_path))

    def test_backfill_mode(self, mocker, tmp_path):
        """Test backfill mode calls backfill_1year."""
        db_path = tmp_path / "test.db"
        db_config = DatabaseConfig(path=str(db_path))
        config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            database=db_config
        )

        mock_db = MagicMock()
        mocker.patch('src.main.DatabaseManager', return_value=mock_db)

        analyzer = MarketSentimentAnalyzer(config, backfill=True, force_refresh=False)
        mocker.patch.object(analyzer.stock_fetcher, 'backfill_1year', return_value=MagicMock(empty=False))

        results = analyzer.run()

        # Verify backfill was called
        analyzer.stock_fetcher.backfill_1year.assert_called_once_with("AAPL")

    def test_normal_mode_uses_cache(self, mocker, tmp_path):
        """Test normal (non-backfill) mode uses cache."""
        db_path = tmp_path / "test.db"
        db_config = DatabaseConfig(path=str(db_path))
        config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            database=db_config
        )

        mock_db = MagicMock()
        mock_db.get_latest_stock_date.return_value = "2024-01-15"
        mock_db.get_stock_data.return_value = MagicMock(empty=False)
        mocker.patch('src.main.DatabaseManager', return_value=mock_db)

        analyzer = MarketSentimentAnalyzer(config, backfill=False, force_refresh=False)
        
        # Mock necessary methods
        mock_indicators = {'Current_Price': 100.0}
        mocker.patch.object(analyzer.stock_fetcher, 'calculate_indicators', return_value=mock_indicators)
        mocker.patch.object(analyzer.news_fetcher, 'fetch_news', return_value=[])
        mocker.patch.object(analyzer.sentiment_analyzer, 'analyze_batch', return_value={
            'sentiment': 'neutral', 'confidence': 0.5, 'explanation': 'Test'
        })

        results = analyzer.run()

        # Verify it used the cached data path (not fetch_data directly)
        # Instead, it should have called get_stock_data on the database
        mock_db.get_stock_data.assert_called()

    def test_force_refresh_bypasses_cache(self, mocker, tmp_path):
        """Test force_refresh bypasses cache."""
        db_path = tmp_path / "test.db"
        db_config = DatabaseConfig(path=str(db_path))
        config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            database=db_config
        )

        mock_db = MagicMock()
        mock_db.get_latest_stock_date.return_value = None  # No cache
        mocker.patch('src.main.DatabaseManager', return_value=mock_db)

        analyzer = MarketSentimentAnalyzer(config, backfill=False, force_refresh=True)
        
        # Mock get_indicators to ensure it's called with force_refresh=True
        mock_indicators = {'Current_Price': 100.0}
        mocker.patch.object(analyzer.stock_fetcher, 'get_indicators', return_value=mock_indicators)
        mocker.patch.object(analyzer.news_fetcher, 'fetch_news', return_value=[])
        mocker.patch.object(analyzer.sentiment_analyzer, 'analyze_batch', return_value={
            'sentiment': 'neutral', 'confidence': 0.5, 'explanation': 'Test'
        })

        results = analyzer.run()

        # Verify get_indicators was called with force_refresh=True
        analyzer.stock_fetcher.get_indicators.assert_called_with("AAPL", force_refresh=True)

    def test_cleanup(self, mocker, tmp_path):
        """Test cleanup closes database."""
        db_path = tmp_path / "test.db"
        db_config = DatabaseConfig(path=str(db_path))
        config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            database=db_config
        )

        mock_db = MagicMock()
        mocker.patch('src.main.DatabaseManager', return_value=mock_db)

        analyzer = MarketSentimentAnalyzer(config)
        analyzer.cleanup()

        mock_db.close.assert_called_once()

    def test_run_handles_error(self, mocker, tmp_path):
        """Test that run handles errors gracefully."""
        db_path = tmp_path / "test.db"
        db_config = DatabaseConfig(path=str(db_path))
        config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            database=db_config
        )

        mock_db = MagicMock()
        mocker.patch('src.main.DatabaseManager', return_value=mock_db)

        analyzer = MarketSentimentAnalyzer(config)
        # Make stock_fetcher.get_indicators raise an exception
        mocker.patch.object(analyzer.stock_fetcher, 'get_indicators', side_effect=Exception("Test error"))

        results = analyzer.run()

        # Should continue and record error
        assert "AAPL" in results
        assert "error" in results["AAPL"]

    def test_multiple_tickers(self, mocker, tmp_path):
        """Test processing multiple tickers."""
        db_path = tmp_path / "test.db"
        db_config = DatabaseConfig(path=str(db_path))
        config = Config(
            tickers=["AAPL", "MSFT"],
            brave_api_key="test_key",
            database=db_config
        )

        mock_db = MagicMock()
        mocker.patch('src.main.DatabaseManager', return_value=mock_db)

        analyzer = MarketSentimentAnalyzer(config)

        mock_indicators = {'Current_Price': 100.0}
        def mock_get_indicators(ticker, force_refresh=False):
            return mock_indicators

        mocker.patch.object(analyzer.stock_fetcher, 'get_indicators', side_effect=mock_get_indicators)
        mocker.patch.object(analyzer.news_fetcher, 'fetch_news', return_value=[])
        mocker.patch.object(analyzer.sentiment_analyzer, 'analyze_batch', return_value={
            'sentiment': 'neutral', 'confidence': 0.5, 'explanation': 'Test'
        })

        results = analyzer.run()

        assert len(results) == 2
        assert "AAPL" in results and "MSFT" in results


class TestMainFunction:
    """Tests for main entry point."""

    def test_main_success(self, mocker):
        """Test main success path."""
        mock_config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            ollama_host="http://test:11434"
        )

        mocker.patch('src.main.Config.load', return_value=mock_config)
        mock_analyzer = Mock(run=Mock(return_value={}), cleanup=Mock())
        mocker.patch('src.main.MarketSentimentAnalyzer', return_value=mock_analyzer)

        exit_code = main()

        assert exit_code == 0
        mock_analyzer.run.assert_called_once()
        mock_analyzer.cleanup.assert_called_once()

    def test_main_configuration_error(self, mocker):
        """Test main with configuration error."""
        mocker.patch('src.main.Config.load', side_effect=ValueError("Missing BRAVE_API_KEY"))

        exit_code = main()

        assert exit_code == 1

    def test_main_keyboard_interrupt(self, mocker):
        """Test main with keyboard interrupt."""
        mocker.patch('src.main.Config.load', side_effect=KeyboardInterrupt)

        exit_code = main()

        assert exit_code == 130

    def test_main_unexpected_error(self, mocker):
        """Test main with unexpected error."""
        mocker.patch('src.main.Config.load', side_effect=Exception("Unexpected"))

        exit_code = main()

        assert exit_code == 1

    def test_main_with_args(self, mocker):
        """Test main with CLI arguments."""
        mock_config = Config(
            tickers=["AAPL"],
            brave_api_key="test_key",
            database=DatabaseConfig()
        )
        mocker.patch('src.main.Config.load', return_value=mock_config)
        mock_analyzer = Mock(run=Mock(return_value={}), cleanup=Mock())
        mocker.patch('src.main.MarketSentimentAnalyzer', return_value=mock_analyzer)

        with patch('sys.argv', ['main.py', '--backfill', '1y']):
            exit_code = main()

        assert exit_code == 0
        # Check that analyzer was created with backfill=True
        call_args = mock_analyzer.call_args
        assert call_args[1]['backfill'] is True
