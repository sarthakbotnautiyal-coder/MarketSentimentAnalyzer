"""Tests for technical indicators calculation and storage."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.fetchers import StockDataFetcher
from src.database import DatabaseManager
import tempfile
import os


@pytest.fixture
def sample_stock_data():
    """Create a sample DataFrame with OHLCV data for testing indicators."""
    # Create 250 days of sample data (enough for all indicators)
    dates = pd.date_range(start='2025-01-01', periods=250, freq='D')
    
    # Generate realistic price movements
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 250)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some volatility for ATR
    high = prices * (1 + np.abs(np.random.normal(0, 0.01, 250)))
    low = prices * (1 - np.abs(np.random.normal(0, 0.01, 250)))
    close = prices + np.random.normal(0, 0.005, 250)
    volume = np.random.randint(1000000, 10000000, 250)
    
    df = pd.DataFrame({
        'Open': prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df


class TestIndicatorCalculations:
    """Test suite for technical indicator calculations."""

    def test_calculate_indicators_basic(self, sample_stock_data):
        """Test that basic indicators are computed."""
        fetcher = StockDataFetcher()
        result = fetcher.calculate_indicators(sample_stock_data)
        
        # Core indicators should exist
        expected_keys = ['Current_Price', 'Change', 'Change_Pct']
        for key in expected_keys:
            assert key in result, f"Missing indicator: {key}"

    def test_price_indicators_valid(self, sample_stock_data):
        """Test that price-based indicators have valid values."""
        fetcher = StockDataFetcher()
        result = fetcher.calculate_indicators(sample_stock_data)
        
        price = result['Current_Price']
        assert price is not None
        assert price > 0
        
        # Verify it's the last close price
        expected_price = sample_stock_data['Close'].iloc[-1]
        assert abs(price - expected_price) < 0.01

    def test_change_computed(self, sample_stock_data):
        """Test that change is computed."""
        fetcher = StockDataFetcher()
        result = fetcher.calculate_indicators(sample_stock_data)
        
        # Change and Change_Pct should exist
        assert 'Change' in result
        assert 'Change_Pct' in result
        
        # Change should be close - previous close
        expected_change = sample_stock_data['Close'].iloc[-1] - sample_stock_data['Close'].iloc[-2]
        assert abs(result['Change'] - expected_change) < 0.01

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Only 5 days of data
        dates = pd.date_range(start='2025-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000000] * 5
        }, index=dates)
        
        fetcher = StockDataFetcher()
        result = fetcher.calculate_indicators(df)
        
        # Should still return basic indicators
        assert result['Current_Price'] is not None
        assert 'Change' in result


class TestDatabaseIndicators:
    """Test suite for database indicator storage and retrieval."""

    def test_save_and_retrieve_indicators(self):
        """Test saving and retrieving indicators from database."""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = DatabaseManager(db_path)
            
            ticker = "TEST"
            date = "2025-03-15"
            indicators = {
                'RSI_14': 65.5,
                'MACD': 1.234,
                'MACD_Hist': 0.567,
                'SMA_20': 150.25,
                'SMA_50': 148.50,
                'SMA_200': 145.0,
                'BB_Upper': 155.0,
                'BB_Middle': 150.0,
                'BB_Lower': 145.0,
                'ATR_14': 3.5,
                'Volume_10d_Avg': 2000000,
                'Volume_30d_Avg': 1800000,
                'Volume_Ratio': 1.111,
                'High_20d': 160.0,
                'Low_20d': 140.0,
                'Current_Price': 150.0,
            }
            
            # Save indicators
            db.save_indicators(ticker, date, indicators)
            
            # Retrieve indicators
            retrieved = db.get_indicators(ticker, date)
            
            assert retrieved is not None
            assert retrieved['ticker'] == ticker
            assert retrieved['date'] == date
            assert retrieved['rsi'] == indicators['RSI_14']
            assert retrieved['macd'] == indicators['MACD']
            assert retrieved['current_price'] == indicators['Current_Price']
            
            db.close()
        finally:
            os.unlink(db_path)

    def test_get_latest_indicators(self):
        """Test retrieving the most recent indicators for a ticker."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = DatabaseManager(db_path)
            
            ticker = "TEST"
            # Save multiple dates
            dates = ["2025-03-10", "2025-03-12", "2025-03-15"]
            for date in dates:
                indicators = {
                    'RSI_14': 50.0,
                    'MACD': 0.0,
                    'MACD_Hist': 0.0,
                    'SMA_20': 100.0,
                    'Current_Price': 150.0,
                }
                db.save_indicators(ticker, date, indicators)
            
            # Get latest
            latest = db.get_indicators(ticker, "2025-03-15")
            
            assert latest is not None
            assert latest['date'] == "2025-03-15"
            
            db.close()
        finally:
            os.unlink(db_path)

    def test_missing_indicators_returns_none(self):
        """Test that querying non-existent ticker/date returns None."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = DatabaseManager(db_path)
            
            result = db.get_indicators("NONEXISTENT", "2025-03-15")
            assert result is None
            
            result = db.get_indicators("TEST", "2099-01-01")
            assert result is None
            
            db.close()
        finally:
            os.unlink(db_path)


class TestIndicatorStorageWithTruncation:
    """Test truncation behavior with indicators."""

    def test_truncate_and_save_single_row(self):
        """Test that truncation and single-row save works correctly."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db = DatabaseManager(db_path)
            
            # Save indicators for multiple tickers
            tickers = ["A", "B", "C", "D", "E"]
            for ticker in tickers:
                indicators = {
                    'Current_Price': 100.0,
                    'RSI_14': 50.0,
                }
                db.save_indicators(ticker, "2025-04-01", indicators)
            
            # Verify count
            assert db.count_indicators() == 5
            
            # Truncate and resave
            db.truncate_indicators()
            assert db.count_indicators() == 0
            
            # Save new single row
            db.save_indicators("X", "2025-04-02", {'Current_Price': 200.0})
            assert db.count_indicators() == 1
            
            retrieved = db.get_indicators("X", "2025-04-02")
            assert retrieved['current_price'] == 200.0
            
            db.close()
        finally:
            os.unlink(db_path)


class TestVolatilityIndicators:
    """Test suite for new volatility and earnings indicators."""

    def test_historical_volatility_20d(self, sample_stock_data):
        """Test 20-day historical volatility calculation."""
        fetcher = StockDataFetcher()
        result = fetcher.calculate_indicators(sample_stock_data)

        assert 'Historical_Volatility_20d' in result
        hv = result['Historical_Volatility_20d']
        assert hv is not None
        assert hv > 0
        # HV should be reasonable (0-200% annualized for most stocks)
        assert 0 < hv < 200

    def test_historical_volatility_30d(self, sample_stock_data):
        """Test 30-day historical volatility calculation."""
        fetcher = StockDataFetcher()
        result = fetcher.calculate_indicators(sample_stock_data)

        assert 'Historical_Volatility_30d' in result
        hv = result['Historical_Volatility_30d']
        assert hv is not None
        assert hv > 0
        assert 0 < hv < 200

    def test_hv_30d_less_than_hv_20d_typically(self):
        """Test that 30d HV is often similar to or less than 20d HV for smooth data."""
        # Create data with a volatility spike in the last 20 days
        np.random.seed(99)
        dates = pd.date_range(start='2025-01-01', periods=250, freq='D')

        # Low vol for first 230 days, high vol for last 20
        low_vol_returns = np.random.normal(0.001, 0.005, 230)
        high_vol_returns = np.random.normal(0.001, 0.03, 20)
        all_returns = np.concatenate([low_vol_returns, high_vol_returns])
        prices = 100 * np.exp(np.cumsum(all_returns))

        df = pd.DataFrame({
            'Open': prices,
            'High': prices * 1.01,
            'Low': prices * 0.99,
            'Close': prices,
            'Volume': [1000000] * 250
        }, index=dates)

        fetcher = StockDataFetcher()
        result = fetcher.calculate_indicators(df)

        assert 'Historical_Volatility_20d' in result
        assert 'Historical_Volatility_30d' in result
        # 20d HV should be higher than 30d because the spike is in last 20 days
        assert result['Historical_Volatility_20d'] > result['Historical_Volatility_30d']

    def test_insufficient_data_for_hv(self):
        """Test HV returns None when insufficient data."""
        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'Open': [100] * 10,
            'High': [101] * 10,
            'Low': [99] * 10,
            'Close': [100.5] * 10,
            'Volume': [1000000] * 10
        }, index=dates)

        fetcher = StockDataFetcher()
        hv = fetcher.calculate_historical_volatility(df, period=20)
        assert hv is None

    def test_calculate_historical_volatility_standalone(self, sample_stock_data):
        """Test standalone HV calculation method."""
        fetcher = StockDataFetcher()

        hv_20 = fetcher.calculate_historical_volatility(sample_stock_data, period=20)
        assert hv_20 is not None
        assert hv_20 > 0

        hv_30 = fetcher.calculate_historical_volatility(sample_stock_data, period=30)
        assert hv_30 is not None
        assert hv_30 > 0

    def test_legacy_hv_key_exists(self, sample_stock_data):
        """Test that legacy Hist_Volatility_20d key is still present."""
        fetcher = StockDataFetcher()
        result = fetcher.calculate_indicators(sample_stock_data)

        assert 'Hist_Volatility_20d' in result
        # Should match the new key
        assert result['Hist_Volatility_20d'] == result['Historical_Volatility_20d']


class TestImpliedVolatility:
    """Test suite for implied volatility from options chain."""

    def test_get_implied_volatility_success(self, mocker):
        """Test successful IV fetch from options chain."""
        # Mock yf.Ticker
        mock_ticker = mocker.MagicMock()

        # Mock history for current price
        dates = pd.date_range('2025-01-01', periods=5)
        mock_history = pd.DataFrame({
            'Close': [150.0, 151.0, 152.0, 153.0, 154.0],
            'Open': [149.0, 150.0, 151.0, 152.0, 153.0],
            'High': [151.0, 152.0, 153.0, 154.0, 155.0],
            'Low': [148.0, 149.0, 150.0, 151.0, 152.0],
            'Volume': [1000000] * 5
        }, index=dates)
        mock_ticker.history.return_value = mock_history

        # Mock options expirations
        mock_ticker.options = ['2025-02-01', '2025-03-01', '2025-04-01']

        # Mock option chain
        mock_calls = pd.DataFrame({
            'strike': [145.0, 150.0, 155.0, 160.0],
            'impliedVolatility': [0.25, 0.22, 0.20, 0.18],
        })
        mock_opt = mocker.MagicMock()
        mock_opt.calls = mock_calls
        mock_ticker.option_chain.return_value = mock_opt

        mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

        fetcher = StockDataFetcher()
        iv = fetcher.get_implied_volatility("AAPL")

        assert iv is not None
        # ATM call at strike=150 (closest to 154) has IV=0.22 -> 22.0%
        assert iv == 22.0

    def test_get_implied_volatility_no_options(self, mocker):
        """Test IV returns None when no options data available."""
        mock_ticker = mocker.MagicMock()
        dates = pd.date_range('2025-01-01', periods=5)
        mock_history = pd.DataFrame({
            'Close': [150.0] * 5,
            'Open': [149.0] * 5,
            'High': [151.0] * 5,
            'Low': [148.0] * 5,
            'Volume': [1000000] * 5
        }, index=dates)
        mock_ticker.history.return_value = mock_history
        mock_ticker.options = []
        mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

        fetcher = StockDataFetcher()
        iv = fetcher.get_implied_volatility("AAPL")
        assert iv is None

    def test_get_implied_volatility_network_error(self, mocker):
        """Test IV returns None on network error."""
        mocker.patch('src.fetchers.yf.Ticker', side_effect=Exception("Network error"))

        fetcher = StockDataFetcher()
        iv = fetcher.get_implied_volatility("AAPL")
        assert iv is None


class TestIVRankAndPercentile:
    """Test suite for IV Rank and IV Percentile calculations."""

    def test_iv_rank_calculation(self):
        """Test IV Rank calculation."""
        fetcher = StockDataFetcher()

        # IV at the top of the range -> 100%
        rank = fetcher.calculate_iv_rank(15.0, 35.0, 35.0)
        assert rank == 100.0

        # IV at the bottom of the range -> 0%
        rank = fetcher.calculate_iv_rank(15.0, 35.0, 15.0)
        assert rank == 0.0

        # IV in the middle -> 50%
        rank = fetcher.calculate_iv_rank(15.0, 35.0, 25.0)
        assert rank == 50.0

    def test_iv_rank_zero_range(self):
        """Test IV Rank returns None when range is zero."""
        fetcher = StockDataFetcher()
        rank = fetcher.calculate_iv_rank(25.0, 25.0, 25.0)
        assert rank is None

    def test_iv_rank_clamped(self):
        """Test IV Rank is clamped to 0-100 range."""
        fetcher = StockDataFetcher()

        # IV above max -> clamped to 100
        rank = fetcher.calculate_iv_rank(15.0, 35.0, 40.0)
        assert rank == 100.0

        # IV below min -> clamped to 0
        rank = fetcher.calculate_iv_rank(15.0, 35.0, 10.0)
        assert rank == 0.0

    def test_iv_percentile_calculation(self):
        """Test IV Percentile calculation."""
        fetcher = StockDataFetcher()

        # Series of 100 HV values from 10 to 30
        hv_series = pd.Series(np.linspace(10, 30, 100))

        # Current IV of 20 should be ~50th percentile
        pct = fetcher.calculate_iv_percentile(hv_series, 20.0)
        assert pct is not None
        assert 45 < pct < 55  # approximately 50%

    def test_iv_percentile_empty_series(self):
        """Test IV Percentile returns None for empty series."""
        fetcher = StockDataFetcher()
        empty_series = pd.Series([], dtype=float)
        pct = fetcher.calculate_iv_percentile(empty_series, 25.0)
        assert pct is None


class TestEarningsDate:
    """Test suite for next earnings date fetching."""

    def test_get_next_earnings_date_dict(self, mocker):
        """Test earnings date from dict-style calendar."""
        mock_ticker = mocker.MagicMock()
        mock_ticker.calendar = {
            'Earnings Date': [pd.Timestamp('2025-04-25')],
            'Earnings Average': [1.50],
        }
        mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

        fetcher = StockDataFetcher()
        result = fetcher.get_next_earnings_date("AAPL")

        assert result == '2025-04-25'

    def test_get_next_earnings_date_no_calendar(self, mocker):
        """Test earnings date returns None when no calendar."""
        mock_ticker = mocker.MagicMock()
        mock_ticker.calendar = None
        mocker.patch('src.fetchers.yf.Ticker', return_value=mock_ticker)

        fetcher = StockDataFetcher()
        result = fetcher.get_next_earnings_date("AAPL")
        assert result is None

    def test_get_next_earnings_date_error(self, mocker):
        """Test earnings date returns None on error."""
        mocker.patch('src.fetchers.yf.Ticker', side_effect=Exception("API error"))

        fetcher = StockDataFetcher()
        result = fetcher.get_next_earnings_date("AAPL")
        assert result is None


class TestDatabaseVolatilityColumns:
    """Test that new volatility/earnings columns are stored and retrieved."""

    def test_save_and_retrieve_volatility_indicators(self):
        """Test saving and retrieving new volatility indicators."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        try:
            db = DatabaseManager(db_path)

            indicators = {
                'Current_Price': 150.0,
                'RSI_14': 65.5,
                'Implied_Volatility': 22.5,
                'Historical_Volatility_20d': 18.3,
                'Historical_Volatility_30d': 17.1,
                'IV_Rank': 55.0,
                'IV_Percentile': 62.5,
                'Next_Earnings_Date': '2025-04-25',
            }

            db.save_indicators("AAPL", "2025-04-09", indicators)
            retrieved = db.get_indicators("AAPL", "2025-04-09")

            assert retrieved is not None
            assert retrieved['implied_volatility'] == 22.5
            assert retrieved['historical_volatility_20d'] == 18.3
            assert retrieved['historical_volatility_30d'] == 17.1
            assert retrieved['iv_rank'] == 55.0
            assert retrieved['iv_percentile'] == 62.5
            assert retrieved['next_earnings_date'] == '2025-04-25'

            db.close()
        finally:
            os.unlink(db_path)

    def test_migration_adds_new_columns(self):
        """Test that migration adds new columns to existing database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name

        try:
            # Create DB and save old-style indicators (no new columns)
            db = DatabaseManager(db_path)
            old_indicators = {
                'Current_Price': 100.0,
                'RSI_14': 50.0,
            }
            db.save_indicators("TEST", "2025-04-01", old_indicators)
            db.close()

            # Reopen - migration should run
            db2 = DatabaseManager(db_path)

            # Save with new columns
            new_indicators = {
                'Current_Price': 110.0,
                'RSI_14': 55.0,
                'Implied_Volatility': 20.0,
                'Historical_Volatility_20d': 15.0,
                'Historical_Volatility_30d': 14.0,
                'IV_Rank': 40.0,
                'IV_Percentile': 55.0,
                'Next_Earnings_Date': '2025-05-01',
            }
            db2.save_indicators("TEST2", "2025-04-09", new_indicators)

            retrieved = db2.get_indicators("TEST2", "2025-04-09")
            assert retrieved['implied_volatility'] == 20.0
            assert retrieved['iv_rank'] == 40.0
            assert retrieved['next_earnings_date'] == '2025-05-01'

            db2.close()
        finally:
            os.unlink(db_path)
