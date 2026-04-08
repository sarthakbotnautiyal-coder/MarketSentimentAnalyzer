"""Tests for main module."""

import pytest
from unittest.mock import patch, Mock, MagicMock
from src.config import Config, DatabaseConfig
from src.main import parse_args, main, run_normal, process_ticker


def test_parse_args_no_args():
    """Test parse args with no arguments."""
    with patch('sys.argv', ['main.py']):
        args = parse_args()
    assert args.backfill is None
    assert args.force_refresh is False


def test_parse_args_backfill():
    """Test parse args with backfill."""
    with patch('sys.argv', ['main.py', '--backfill', '1y']):
        args = parse_args()
    assert args.backfill == '1y'


def test_parse_args_force_refresh():
    """Test parse args with force refresh."""
    with patch('sys.argv', ['main.py', '--force-refresh']):
        args = parse_args()
    assert args.force_refresh is True


class TestProcessTicker:
    """Tests for process_ticker function."""

    def test_process_ticker_success(self, mocker):
        """Test successful ticker processing."""
        mock_db = Mock()
        mock_fetcher = Mock()
        mock_df = Mock()
        mock_df.__getitem__ = Mock(return_value=Mock())
        mock_df.empty = False
        mock_df.index = [MagicMock(strftime=lambda fmt: '2024-01-15')]

        mock_fetcher.fetch_delta.return_value = mock_df
        mock_indicators = {'Current_Price': 150.0}
        mock_fetcher.calculate_indicators.return_value = mock_indicators

        mocker.patch('src.main.Display.print_indicators')

        result = process_ticker('AAPL', mock_db, mock_fetcher)

        assert result['ticker'] == 'AAPL'
        assert result['indicators']['Current_Price'] == 150.0
        mock_db.save_indicators.assert_called_once()

    def test_process_ticker_no_data(self, mocker):
        """Test processing ticker with no data."""
        mock_db = Mock()
        mock_fetcher = Mock()
        mock_fetcher.fetch_delta.return_value = None

        mocker.patch('src.main.logger')

        result = process_ticker('AAPL', mock_db, mock_fetcher)

        assert 'error' in result['indicators']


class TestRunNormal:
    """Tests for run_normal function."""

    def test_run_normal_success(self, mocker):
        """Test normal run mode."""
        mock_db = Mock()
        mock_fetcher = Mock()
        mock_df = Mock()
        mock_df.__getitem__ = Mock(return_value=Mock())
        mock_df.empty = False
        mock_df.index = [MagicMock(strftime=lambda fmt: '2024-01-15')]

        mock_fetcher.fetch_delta.return_value = mock_df
        mock_indicators = {'Current_Price': 150.0}
        mock_fetcher.calculate_indicators.return_value = mock_indicators

        mocker.patch('src.main.Display.print_indicators')
        mocker.patch('src.main.Display.print_summary')
        mocker.patch('src.main.StockDataFetcher', return_value=mock_fetcher)

        config = Config()
        tickers = ['AAPL', 'MSFT']

        results = run_normal(config, mock_db, tickers)

        assert 'AAPL' in results
        assert 'MSFT' in results
        mock_db.truncate_indicators.assert_called_once()


class TestMain:
    """Tests for main entry point."""

    def test_main_success(self, mocker):
        """Test main success path."""
        mock_config = Config(tickers=['AAPL'])
        mocker.patch('src.main.Config.load', return_value=mock_config)

        mock_db = Mock()
        mocker.patch('src.main.DatabaseManager', return_value=mock_db)

        mocker.patch('src.main.run_normal', return_value={'AAPL': {}})
        mocker.patch('sys.argv', ['main.py'])

        result = main()

        assert result == 0

    def test_main_keyboard_interrupt(self, mocker):
        """Test main with keyboard interrupt."""
        mocker.patch('src.main.Config.load', side_effect=KeyboardInterrupt)
        mocker.patch('sys.argv', ['main.py'])

        result = main()

        assert result == 130

    def test_main_error(self, mocker):
        """Test main with error."""
        mocker.patch('src.main.Config.load', side_effect=Exception("Test error"))
        mocker.patch('sys.argv', ['main.py'])

        result = main()

        assert result == 1

    def test_main_no_tickers(self, mocker):
        """Test main with no tickers configured."""
        mock_config = Config(tickers=[])
        mocker.patch('src.main.Config.load', return_value=mock_config)
        mocker.patch('sys.argv', ['main.py'])

        result = main()

        assert result == 1
