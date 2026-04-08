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
