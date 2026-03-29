"""Database caching layer for MarketSentimentAnalyzer."""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import structlog

logger = structlog.get_logger()


class DatabaseManager:
    """Manage SQLite database for caching market data."""

    def __init__(self, db_path: str | Path):
        """Initialize database connection and create tables if needed."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()

        # Stock daily data table (simplified: only price/volume)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_daily (
                date TEXT,
                ticker TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (date, ticker)
            )
        ''')

        # Indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicators (
                date TEXT,
                ticker TEXT,
                rsi REAL,
                macd REAL,
                macd_hist REAL,
                sma20 REAL,
                sma50 REAL,
                sma200 REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                atr REAL,
                vol_10d REAL,
                vol_30d REAL,
                vol_ratio REAL,
                high_20d REAL,
                low_20d REAL,
                PRIMARY KEY (date, ticker)
            )
        ''')

        # Indexes for stock_daily
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stock_daily(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_daily(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_ticker ON indicators(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date)')

        self.conn.commit()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Stock data methods
    def save_stock_data(self, ticker: str, df: pd.DataFrame):
        """Save stock DataFrame to database."""
        cursor = self.conn.cursor()

        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)

            cursor.execute('''
                INSERT OR REPLACE INTO stock_daily
                (date, ticker, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                date_str,
                ticker,
                float(row.get('Open', 0)) if pd.notna(row.get('Open')) else None,
                float(row.get('High', 0)) if pd.notna(row.get('High')) else None,
                float(row.get('Low', 0)) if pd.notna(row.get('Low')) else None,
                float(row.get('Close', 0)) if pd.notna(row.get('Close')) else None,
                int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None
            ))

        self.conn.commit()
        logger.info(f"Saved {len(df)} stock records for {ticker}")

    def get_latest_stock_date(self, ticker: str) -> Optional[str]:
        """Get the latest date in cache for a ticker."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(date) as latest_date
            FROM stock_daily
            WHERE ticker = ?
        ''', (ticker,))
        result = cursor.fetchone()
        return result['latest_date'] if result and result['latest_date'] else None

    def get_stock_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Retrieve stock data from database.

        Returns a DataFrame with columns matching yfinance format:
        Open, High, Low, Close, Volume with datetime index.
        """
        query = '''SELECT date, open, high, low, close, volume 
                   FROM stock_daily WHERE ticker = ?'''
        params = [ticker]

        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)

        query += ' ORDER BY date'

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        # Build DataFrame
        data = []
        dates = []
        for row in rows:
            dates.append(row['date'])
            data.append({
                'Open': float(row['open']) if row['open'] is not None else None,
                'High': float(row['high']) if row['high'] is not None else None,
                'Low': float(row['low']) if row['low'] is not None else None,
                'Close': float(row['close']) if row['close'] is not None else None,
                'Volume': int(row['volume']) if row['volume'] is not None else None
            })

        df = pd.DataFrame(data, index=pd.to_datetime(dates))
        return df

    # News methods
    def save_news(self, ticker: str, articles: List[Dict[str, str]]):
        """Save news articles to database, skipping duplicates by URL."""
        cursor = self.conn.cursor()
        saved_count = 0

        for article in articles:
            url = article.get('url', '')
            if not url:
                continue  # Skip articles without URL

            # Check if already exists
            cursor.execute('''
                SELECT 1 FROM news WHERE url = ? AND ticker = ?
            ''', (url, ticker))

            if cursor.fetchone():
                continue  # Skip duplicate

            cursor.execute('''
                INSERT OR IGNORE INTO news
                (date, ticker, title, url, snippet, source, published)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d'),
                ticker,
                article.get('title', ''),
                url,
                article.get('snippet', ''),
                article.get('source', ''),
                article.get('published', '')
            ))
            saved_count += 1

        self.conn.commit()
        if saved_count > 0:
            logger.info(f"Saved {saved_count} new news articles for {ticker}")

    def get_latest_news_date(self, ticker: str) -> Optional[str]:
        """Get the latest news date in cache for a ticker."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(date) as latest_date
            FROM news
            WHERE ticker = ?
        ''', (ticker,))
        result = cursor.fetchone()
        return result['latest_date'] if result and result['latest_date'] else None

    def get_cached_news(self, ticker: str, days_back: int = 7) -> List[Dict[str, str]]:
        """Get cached news from the last N days."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT title, url, snippet, source, published, date
            FROM news
            WHERE ticker = ? AND date >= ?
            ORDER BY date DESC
        ''', (ticker, cutoff_date))
        rows = cursor.fetchall()

        articles = []
        for row in rows:
            articles.append({
                'title': row['title'],
                'url': row['url'],
                'snippet': row['snippet'],
                'source': row['source'],
                'published': row['published']
            })

        return articles

    # Sentiment methods
    def save_sentiment(self, ticker: str, date: str, sentiment: str, confidence: float, explanation: str):
        """Save sentiment analysis to database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sentiment
            (date, ticker, sentiment, confidence, explanation)
            VALUES (?, ?, ?, ?, ?)
        ''', (date, ticker, sentiment, confidence, explanation))
        self.conn.commit()
        logger.debug(f"Saved sentiment for {ticker} on {date}")

    def get_latest_sentiment_date(self, ticker: str) -> Optional[str]:
        """Get the latest sentiment date in cache for a ticker."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(date) as latest_date
            FROM sentiment
            WHERE ticker = ?
        ''', (ticker,))
        result = cursor.fetchone()
        return result['latest_date'] if result and result['latest_date'] else None

    def get_cached_sentiment(self, ticker: str, days_back: int = 7) -> Optional[Dict[str, Any]]:
        """Get most recent sentiment from cache within N days."""
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT sentiment, confidence, explanation, date
            FROM sentiment
            WHERE ticker = ? AND date >= ?
            ORDER BY date DESC
            LIMIT 1
        ''', (ticker, cutoff_date))
        row = cursor.fetchone()

        if row:
            return {
                'sentiment': row['sentiment'],
                'confidence': row['confidence'],
                'explanation': row['explanation'],
                'date': row['date']
            }
        return None

    # Freshness checking
    def is_data_fresh(self, table: str, ticker: str, ttl_days: int) -> bool:
        """Check if data for a ticker is fresh based on TTL."""
        if table == 'stock_daily':
            latest = self.get_latest_stock_date(ticker)
        elif table == 'news':
            latest = self.get_latest_news_date(ticker)
        elif table == 'sentiment':
            latest = self.get_latest_sentiment_date(ticker)
        else:
            return False

        if not latest:
            return False

        try:
            latest_date = datetime.strptime(latest, '%Y-%m-%d')
            cutoff = datetime.now() - timedelta(days=ttl_days)
            return latest_date >= cutoff
        except (ValueError, TypeError):
            return False

    # Indicators methods
    def save_indicators(self, ticker: str, date: str, indicators: Dict[str, Any]):
        """Save or replace indicators row."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO indicators
            (date, ticker, rsi, macd, macd_hist, sma20, sma50, sma200,
             bb_upper, bb_middle, bb_lower, atr,
             vol_10d, vol_30d, vol_ratio, high_20d, low_20d)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date,
            ticker,
            indicators.get('RSI_14'),
            indicators.get('MACD'),
            indicators.get('MACD_Hist'),
            indicators.get('SMA_20'),
            indicators.get('SMA_50'),
            indicators.get('SMA_200'),
            indicators.get('BB_Upper'),
            indicators.get('BB_Middle'),
            indicators.get('BB_Lower'),
            indicators.get('ATR_14'),
            indicators.get('Volume_10d_Avg'),
            indicators.get('Volume_30d_Avg'),
            indicators.get('Volume_Ratio'),
            indicators.get('High_20d'),
            indicators.get('Low_20d')
        ))
        self.conn.commit()
        logger.debug(f"Saved indicators for {ticker} on {date}")

    def get_indicators(self, ticker: str, date: str) -> Optional[Dict[str, Any]]:
        """Retrieve indicators for a ticker on a given date."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM indicators WHERE ticker = ? AND date = ?', (ticker, date))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)

    def clear_old_data(self, days: int = 30):
        """Delete data older than N days to keep database size manageable."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor = self.conn.cursor()

        # Clear old stock data
        cursor.execute('DELETE FROM stock_daily WHERE date < ?', (cutoff,))
        stock_deleted = cursor.rowcount

        # Clear old news
        cursor.execute('DELETE FROM news WHERE date < ?', (cutoff,))
        news_deleted = cursor.rowcount

        # Clear old sentiment (keep longer maybe)
        sentiment_cutoff = (datetime.now() - timedelta(days=days*3)).strftime('%Y-%m-%d')
        cursor.execute('DELETE FROM sentiment WHERE date < ?', (sentiment_cutoff,))
        sentiment_deleted = cursor.rowcount

        self.conn.commit()
        logger.info(f"Cleared old data: {stock_deleted} stock, {news_deleted} news, {sentiment_deleted} sentiment records")
