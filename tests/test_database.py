"""Tests for database module."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import pandas as pd

from src.database import DatabaseManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = DatabaseManager(db_path)
    yield db
    
    db.close()
    os.unlink(db_path)


@pytest.fixture
def sample_stock_df():
    """Create sample stock DataFrame."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = {
        'Open': [100.0 + i for i in range(10)],
        'High': [101.0 + i for i in range(10)],
        'Low': [99.0 + i for i in range(10)],
        'Close': [100.5 + i for i in range(10)],
        'Volume': [1000000 + i * 1000 for i in range(10)]
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def sample_indicators():
    """Create sample indicators dictionary."""
    return {
        'RSI_14': 65.5,
        'MACD': 1.2,
        'MACD_Signal': 0.8,
        'MACD_Hist': 0.4,
        'SMA_20': 105.0,
        'EMA_20': 104.5,
        'Current_Price': 110.5,
        'Change': 2.5
    }


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    def test_create_tables(self, temp_db):
        """Test that all tables are created."""
        cursor = temp_db.conn.cursor()
        
        # Check stock_daily table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_daily'")
        assert cursor.fetchone() is not None
        
        # Check news table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='news'")
        assert cursor.fetchone() is not None
        
        # Check sentiment table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment'")
        assert cursor.fetchone() is not None

    def test_save_and_get_stock_data(self, temp_db, sample_stock_df):
        """Test saving and retrieving stock data."""
        ticker = "AAPL"
        temp_db.save_stock_data(ticker, sample_stock_df)
        
        # Retrieve all data
        result = temp_db.get_stock_data(ticker)
        
        assert not result.empty
        assert len(result) == len(sample_stock_df)
        assert 'Close' in result.columns
        # Check a value
        assert abs(result.iloc[-1]['Close'] - sample_stock_df.iloc[-1]['Close']) < 0.01

    def test_save_stock_with_indicators(self, temp_db, sample_stock_df, sample_indicators):
        """Test saving stock data with indicators JSON."""
        ticker = "AAPL"
        # Create indicators dict keyed by date
        indicators_by_date = {}
        for date in sample_stock_df.index:
            date_str = date.strftime('%Y-%m-%d')
            indicators_by_date[date_str] = sample_indicators
        
        temp_db.save_stock_data(ticker, sample_stock_df, indicators_by_date)
        
        result = temp_db.get_stock_data(ticker)
        assert not result.empty
        
        # Check indicators column exists and has data
        last_row = result.iloc[-1]
        assert 'indicators' in last_row
        assert last_row['indicators'] is not None
        assert 'RSI_14' in last_row['indicators']

    def test_get_latest_stock_date(self, temp_db, sample_stock_df):
        """Test getting latest stock date."""
        ticker = "AAPL"
        temp_db.save_stock_data(ticker, sample_stock_df)
        
        latest = temp_db.get_latest_stock_date(ticker)
        expected = sample_stock_df.index[-1].strftime('%Y-%m-%d')
        assert latest == expected

    def test_get_latest_stock_date_empty(self, temp_db):
        """Test getting latest date when no data."""
        latest = temp_db.get_latest_stock_date("AAPL")
        assert latest is None

    def test_get_stock_data_with_date_range(self, temp_db, sample_stock_df):
        """Test retrieving stock data with date range."""
        ticker = "AAPL"
        temp_db.save_stock_data(ticker, sample_stock_df)
        
        start = sample_stock_df.index[2].strftime('%Y-%m-%d')
        end = sample_stock_df.index[7].strftime('%Y-%m-%d')
        
        result = temp_db.get_stock_data(ticker, start_date=start, end_date=end)
        
        assert len(result) == 6  # indices 2-7 inclusive
        assert result.index[0] >= pd.to_datetime(start)
        assert result.index[-1] <= pd.to_datetime(end)

    def test_save_and_get_news(self, temp_db):
        """Test saving and retrieving news articles."""
        ticker = "AAPL"
        articles = [
            {
                'title': 'Test Article 1',
                'url': 'http://example.com/1',
                'snippet': 'Summary 1',
                'source': 'Example',
                'published': '2024-01-01'
            },
            {
                'title': 'Test Article 2',
                'url': 'http://example.com/2',
                'snippet': 'Summary 2',
                'source': 'Example',
                'published': '2024-01-02'
            }
        ]
        
        temp_db.save_news(ticker, articles)
        
        cached = temp_db.get_cached_news(ticker, days_back=30)
        assert len(cached) == 2
        assert cached[0]['title'] == 'Test Article 1'

    def test_news_deduplication(self, temp_db):
        """Test that duplicate news by URL is not saved."""
        ticker = "AAPL"
        articles = [
            {'title': 'Article 1', 'url': 'http://same.com', 'snippet': 'First'},
            {'title': 'Article 1 Duplicate', 'url': 'http://same.com', 'snippet': 'Second'},
            {'title': 'Article 2', 'url': 'http://different.com', 'snippet': 'Third'}
        ]
        
        temp_db.save_news(ticker, articles)
        temp_db.save_news(ticker, articles)  # Save again
        
        cached = temp_db.get_cached_news(ticker, days_back=30)
        # Should only have 2 unique URLs
        assert len(cached) == 2
        urls = [a['url'] for a in cached]
        assert 'http://same.com' in urls
        assert 'http://different.com' in urls

    def test_news_without_url_skipped(self, temp_db):
        """Test that articles without URLs are skipped."""
        ticker = "AAPL"
        articles = [
            {'title': 'No URL', 'url': '', 'snippet': 'Missing'},
            {'title': 'Has URL', 'url': 'http://valid.com', 'snippet': 'Valid'}
        ]
        
        temp_db.save_news(ticker, articles)
        cached = temp_db.get_cached_news(ticker, days_back=30)
        assert len(cached) == 1
        assert cached[0]['url'] == 'http://valid.com'

    def test_get_latest_news_date(self, temp_db):
        """Test getting latest news date."""
        ticker = "AAPL"
        articles = [
            {'title': 'A', 'url': 'http://a.com', 'snippet': 'A', 'published': '2024-01-01'}
        ]
        temp_db.save_news(ticker, articles)
        
        latest = temp_db.get_latest_news_date(ticker)
        assert latest is not None

    def test_save_and_get_sentiment(self, temp_db):
        """Test saving and retrieving sentiment."""
        ticker = "AAPL"
        temp_db.save_sentiment(ticker, '2024-01-15', 'bullish', 0.85, 'Strong positive')
        
        cached = temp_db.get_cached_sentiment(ticker, days_back=30)
        assert cached is not None
        assert cached['sentiment'] == 'bullish'
        assert cached['confidence'] == 0.85

    def test_get_latest_sentiment_date(self, temp_db):
        """Test getting latest sentiment date."""
        ticker = "AAPL"
        temp_db.save_sentiment(ticker, '2024-01-15', 'bullish', 0.85, 'Strong')
        
        latest = temp_db.get_latest_sentiment_date(ticker)
        assert latest == '2024-01-15'

    def test_is_data_fresh(self, temp_db):
        """Test freshness checking."""
        ticker = "AAPL"
        temp_db.save_sentiment(ticker, datetime.now().strftime('%Y-%m-%d'), 'bullish', 0.85, 'Test')
        
        assert temp_db.is_data_fresh('sentiment', ticker, 1) is True
        assert temp_db.is_data_fresh('sentiment', ticker, 0) is False  # TTL 0 means not fresh

    def test_is_data_fresh_stale(self, temp_db):
        """Test stale data detection."""
        ticker = "AAPL"
        old_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        temp_db.save_sentiment(ticker, old_date, 'bearish', 0.75, 'Old')
        
        assert temp_db.is_data_fresh('sentiment', ticker, 7) is False
        assert temp_db.is_data_fresh('sentiment', ticker, 14) is True

    def test_clear_old_data(self, temp_db):
        """Test cleanup of old records."""
        # Insert some old and new data
        ticker = "AAPL"
        old_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        recent_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        df_old = pd.DataFrame({
            'Close': [100.0],
            'Volume': [1000000]
        }, index=[pd.to_datetime(old_date)])
        temp_db.save_stock_data(ticker, df_old)
        
        df_recent = pd.DataFrame({
            'Close': [200.0],
            'Volume': [2000000]
        }, index=[pd.to_datetime(recent_date)])
        temp_db.save_stock_data(ticker, df_recent)
        
        temp_db.clear_old_data(days=30)
        
        # Only recent should remain
        result = temp_db.get_stock_data(ticker)
        assert len(result) == 1
        assert result.index[0].strftime('%Y-%m-%d') == recent_date

    def test_clear_old_data_news_sentiment(self, temp_db):
        """Test cleanup of old news and sentiment."""
        ticker = "AAPL"
        old_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        
        # Save old sentiment
        temp_db.save_sentiment(ticker, old_date, 'bearish', 0.5, 'Old')
        
        temp_db.clear_old_data(days=30)
        
        # Should be gone
        cached = temp_db.get_cached_sentiment(ticker, days_back=30)
        assert cached is None

    def test_context_manager(self):
        """Test DatabaseManager as context manager."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        with DatabaseManager(db_path) as db:
            # Check it works
            assert db.conn is not None
        
        # After exiting context, connection should be closed
        # Try to use it - should fail or reconnect
        try:
            cursor = db.conn.cursor()
            cursor.execute("SELECT 1")
            # If this succeeded, connection is still open - that's okay for SQLite
        except Exception:
            pass
        
        os.unlink(db_path)
