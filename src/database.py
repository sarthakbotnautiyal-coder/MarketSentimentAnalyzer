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

        # Stock daily data table
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

        # Ticker metadata table - tracks last fetched date per ticker
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ticker_metadata (
                ticker TEXT PRIMARY KEY,
                last_fetched_date TEXT
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
                current_price REAL,
                implied_volatility REAL,
                historical_volatility_20d REAL,
                historical_volatility_30d REAL,
                iv_rank REAL,
                iv_percentile REAL,
                next_earnings_date TEXT,
                PRIMARY KEY (date, ticker)
            )
        ''')

        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stock_daily(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_daily(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_ticker ON indicators(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_date ON indicators(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ticker_metadata_ticker ON ticker_metadata(ticker)')

        self.conn.commit()
        self._migrate_tables()

    def _migrate_tables(self):
        """Auto-migrate from old schema to new schema."""
        cursor = self.conn.cursor()

        # Check if ticker_metadata table exists and has data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ticker_metadata'")
        if not cursor.fetchone():
            logger.info("ticker_metadata table not found - will be created")
            return

        # Check if we need to populate from existing stock_daily data
        cursor.execute("SELECT COUNT(*) as count FROM ticker_metadata")
        metadata_count = cursor.fetchone()['count']

        if metadata_count == 0:
            logger.info("Populating ticker_metadata from existing stock_daily data")
            cursor.execute('''
                INSERT OR REPLACE INTO ticker_metadata (ticker, last_fetched_date)
                SELECT ticker, MAX(date) as last_fetched_date
                FROM stock_daily
                GROUP BY ticker
            ''')
            self.conn.commit()
            logger.info(f"Populated metadata for {cursor.rowcount} tickers")

        # Migrate new indicator columns (added for volatility/earnings indicators)
        cursor.execute("PRAGMA table_info(indicators)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        new_columns = [
            ('implied_volatility', 'REAL'),
            ('historical_volatility_20d', 'REAL'),
            ('historical_volatility_30d', 'REAL'),
            ('iv_rank', 'REAL'),
            ('iv_percentile', 'REAL'),
            ('next_earnings_date', 'TEXT'),
        ]

        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                cursor.execute(f'ALTER TABLE indicators ADD COLUMN {col_name} {col_type}')
                logger.info(f"Added column {col_name} to indicators table")

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

    def get_last_fetched_date(self, ticker: str) -> Optional[str]:
        """Get the last fetched date for a ticker from ticker_metadata.

        If no entry exists but stock_daily has data for the ticker,
        auto-migrate by inserting the MAX(date) from stock_daily.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Last fetched date as YYYY-MM-DD string or None if not tracked
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT last_fetched_date FROM ticker_metadata WHERE ticker = ?
        ''', (ticker,))
        result = cursor.fetchone()
        if result and result['last_fetched_date']:
            return result['last_fetched_date']

        # Auto-migrate: infer from stock_daily if available
        cursor.execute('''
            SELECT MAX(date) as latest_date FROM stock_daily WHERE ticker = ?
        ''', (ticker,))
        stock_result = cursor.fetchone()
        if stock_result and stock_result['latest_date']:
            # Insert into ticker_metadata for future use
            self.update_last_fetched_date(ticker, stock_result['latest_date'])
            logger.info(f"Auto-migrated last_fetched_date for {ticker}: {stock_result['latest_date']}")
            return stock_result['latest_date']

        return None

    def update_last_fetched_date(self, ticker: str, date: str):
        """Update the last fetched date for a ticker.

        Args:
            ticker: Stock ticker symbol
            date: Date as YYYY-MM-DD string
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO ticker_metadata (ticker, last_fetched_date)
            VALUES (?, ?)
        ''', (ticker, date))
        self.conn.commit()
        logger.debug(f"Updated last_fetched_date for {ticker} to {date}")

    def get_latest_stock_date(self, ticker: str) -> Optional[str]:
        """Get the latest date in cache for a ticker (from stock_daily)."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT MAX(date) as latest_date
            FROM stock_daily
            WHERE ticker = ?
        ''', (ticker,))
        result = cursor.fetchone()
        return result['latest_date'] if result and result['latest_date'] else None

    def get_stock_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Retrieve stock data from database."""
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

    def is_data_fresh(self, table_name: str, ticker: str, ttl_days: int = 1) -> bool:
        """Check if data is fresh within TTL."""
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now() - timedelta(days=ttl_days)).strftime('%Y-%m-%d')

        if table_name == 'stock_daily':
            cursor.execute('''
                SELECT MAX(date) as latest_date
                FROM stock_daily WHERE ticker = ?
            ''', (ticker,))
        else:
            return False

        result = cursor.fetchone()
        if not result or not result['latest_date']:
            return False

        return result['latest_date'] >= cutoff_date

    # Indicators methods
    def truncate_indicators(self):
        """Truncate all data from indicators table."""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM indicators')
        deleted_count = cursor.rowcount
        self.conn.commit()
        logger.info(f"Truncated indicators table: {deleted_count} rows deleted")
        return deleted_count

    def insert_latest_indicator(self, ticker: str, date: str, indicators: Dict[str, Any]):
        """Insert or replace indicator row for latest day's data."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO indicators
            (date, ticker, rsi, macd, macd_hist, sma20, sma50, sma200,
             bb_upper, bb_middle, bb_lower, atr,
             vol_10d, vol_30d, vol_ratio, high_20d, low_20d, current_price,
             implied_volatility, historical_volatility_20d, historical_volatility_30d,
             iv_rank, iv_percentile, next_earnings_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            indicators.get('Low_20d'),
            indicators.get('Current_Price'),
            indicators.get('Implied_Volatility'),
            indicators.get('Historical_Volatility_20d'),
            indicators.get('Historical_Volatility_30d'),
            indicators.get('IV_Rank'),
            indicators.get('IV_Percentile'),
            indicators.get('Next_Earnings_Date'),
        ))
        self.conn.commit()
        logger.debug(f"Inserted latest indicators for {ticker} on {date}")

    def get_indicator_rows(self, ticker: str, date: str) -> Optional[Dict[str, Any]]:
        """Retrieve an indicator row by ticker and date."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM indicators WHERE ticker = ? AND date = ?', (ticker, date))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)

    def get_all_latest_indicators(self) -> List[Tuple[str, str]]:
        """Get all tickers with their latest indicator date."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT ticker, MAX(date) as latest_date
            FROM indicators
            GROUP BY ticker
            ORDER BY ticker
        ''')
        return [(row['ticker'], row['latest_date']) for row in cursor.fetchall()]

    def save_indicators(self, ticker: str, date: str, indicators: Dict[str, Any]):
        """Save or replace indicators row (alias for insert_latest_indicator)."""
        self.insert_latest_indicator(ticker, date, indicators)

    def get_indicators(self, ticker: str, date: str) -> Optional[Dict[str, Any]]:
        """Retrieve indicators for a ticker on a given date."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM indicators WHERE ticker = ? AND date = ?', (ticker, date))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)

    def count_indicators(self) -> int:
        """Count total rows in indicators table."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM indicators')
        result = cursor.fetchone()
        return result['count'] if result else 0

    def clear_old_stock_data(self, days: int = 365):
        """Delete stock data older than N days to keep database size manageable."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM stock_daily WHERE date < ?', (cutoff,))
        stock_deleted = cursor.rowcount

        self.conn.commit()
        logger.info(f"Cleared old stock data: {stock_deleted} records")
        return stock_deleted
