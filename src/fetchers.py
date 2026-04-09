"""
Stock data fetcher using yfinance.

Handles:
- Fetching historical stock data via yfinance
- Incremental/delta updates with database caching
- Technical indicator calculations
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import structlog
import yfinance as yf

logger = structlog.get_logger()


class StockDataFetcher:
    """Fetches stock data from Yahoo Finance and calculates technical indicators."""

    def __init__(self, period: str = "1y", interval: str = "1d",
                 db_manager=None):
        """Initialize fetcher.

        Args:
            period: Default period for fetching data (1y, 6mo, 3mo, 1mo, 5d)
            interval: Data interval (1d, 1wk, 1mo)
            db_manager: Database manager for caching (optional)
        """
        self.period = period
        self.interval = interval
        self.db = db_manager

    def fetch_data(self, ticker: str, period: str = None,
                   force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Fetch historical data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            period: Time period (overrides default if provided)
            force_refresh: If True, bypass database cache and fetch fresh

        Returns:
            DataFrame with stock data or None on failure
        """
        fetch_period = period or self.period

        # Check database cache first (unless forcing refresh)
        if not force_refresh and self.db:
            cached = self.db.get_stock_data(ticker)
            if cached is not None and not cached.empty:
                return cached

        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period=fetch_period, interval=self.interval)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            # Clean up index
            df.index = df.index.tz_localize(None) if df.index.tz else df.index

            # Cache to database
            if self.db:
                self.db.save_stock_data(ticker, df)
                latest_date = df.index.max().strftime('%Y-%m-%d')
                self.db.update_last_fetched_date(ticker, latest_date)

            logger.info(f"Fetched {len(df)} rows for {ticker} ({fetch_period})")
            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def fetch_delta(self, ticker: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Fetch only new data since last fetch (delta/incremental).

        Uses the ticker_metadata table to determine last fetched date.
        Combines cached data with newly fetched data.

        Args:
            ticker: Stock ticker symbol
            force_refresh: If True, fetch full history instead of delta

        Returns:
            DataFrame with stock data (combined from cache + new) or None
        """
        if force_refresh or not self.db:
            return self.fetch_data(ticker, period="1y", force_refresh=force_refresh)

        # Get the last fetched date from ticker_metadata
        last_fetched_date = self.db.get_last_fetched_date(ticker)

        if last_fetched_date is None:
            # No metadata entry - first run, fetch full year
            logger.info(f"No ticker_metadata entry for {ticker}, fetching full year")
            df = self.fetch_data(ticker, period="1y", force_refresh=True)
            if df is not None and not df.empty:
                # Update metadata with the latest date from fetched data
                latest_date = df.index.max().strftime('%Y-%m-%d')
                self.db.update_last_fetched_date(ticker, latest_date)
            return df

        # Parse last fetched date
        try:
            last_date = datetime.strptime(last_fetched_date, '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"Invalid date format in ticker_metadata for {ticker}: {last_fetched_date}")
            return self.fetch_data(ticker, period="1y", force_refresh=True)

        today = datetime.now().date()

        if last_date >= today - timedelta(days=1):
            # Cache is up to date, just return cached data
            logger.info(f"Cache is current for {ticker} (last: {last_fetched_date})")
            return self.db.get_stock_data(ticker)

        # Need to fetch delta from last_fetched_date + 1 day to today
        # Add one day to avoid re-fetching the last known date
        start_date = last_date + timedelta(days=1)
        end_date = today

        if start_date > end_date:
            logger.info(f"Start date {start_date} is after end date {end_date}, nothing to fetch")
            return self.db.get_stock_data(ticker)

        logger.info(f"Fetching delta for {ticker}: {start_date} to {end_date}")

        try:
            yf_ticker = yf.Ticker(ticker)

            # Fetch data for the specific date range
            # yfinance supports start/end through history() parameters
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            df = yf_ticker.history(start=start_str, end=end_str, interval=self.interval)

            if df.empty:
                logger.warning(f"No new data returned for {ticker} in range {start_str} to {end_str}")
                # Update metadata anyway? No, keep old date
                return self.db.get_stock_data(ticker)

            # Clean up index
            df.index = df.index.tz_localize(None) if df.index.tz else df.index

            logger.info(f"Delta fetch returned {len(df)} new rows for {ticker}")

            # Get existing cached data
            cached_df = self.db.get_stock_data(ticker)

            if cached_df is not None and not cached_df.empty:
                # Combine cached data with new data
                combined = pd.concat([cached_df, df])
                combined = combined.sort_index()
                # Remove duplicates, keeping the later (delta) values if overlap
                combined = combined[~combined.index.duplicated(keep='last')]
                logger.info(f"Combined cache ({len(cached_df)}) with delta ({len(df)}) = {len(combined)} rows")
            else:
                combined = df
                logger.info(f"No cache found, using {len(df)} fetched rows")

            # Save combined data to database
            if self.db:
                self.db.save_stock_data(ticker, combined)
                # Update metadata with latest date
                latest_date = combined.index.max().strftime('%Y-%m-%d')
                self.db.update_last_fetched_date(ticker, latest_date)

            return combined

        except Exception as e:
            logger.error(f"Error fetching delta for {ticker}: {e}")
            return self.db.get_stock_data(ticker) if self.db else None

    def backfill_1year(self, ticker: str) -> Optional[pd.DataFrame]:
        """Backfill 1 year of historical data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with 1 year of stock data
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period="1y", interval="1d")
            if df.empty:
                logger.warning(f"No data for backfill of {ticker}")
                return None

            # Clean up index
            df.index = df.index.tz_localize(None) if df.index.tz else df.index

            logger.info(f"Backfilled {len(df)} rows for {ticker}")

            # Cache to database
            if self.db:
                self.db.save_stock_data(ticker, df)
                # Update last_fetched_date to latest
                latest_date = df.index.max().strftime('%Y-%m-%d')
                self.db.update_last_fetched_date(ticker, latest_date)

            return df
        except Exception as e:
            logger.error(f"Error backfilling {ticker}: {e}")
            return None

    def backfill_with_delta(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch data using delta strategy - only new data since last date.

        This is used for normal runs (not --backfill mode).
        Per ticker: fetch only new data since last date in stock_daily.

        Args:
            ticker: Stock ticker symbol

        Returns:
            DataFrame with all available data (cached + new)
        """
        # For the first run, there's no data - do a full 1y fetch
        # For subsequent runs, only fetch data since last known date
        return self.fetch_delta(ticker)

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for latest data point.

        Pure pandas implementation (no pandas_ta dependency).

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary with calculated indicators
        """
        if df.empty:
            return {"error": "No data available"}

        try:
            indicators = {}

            # Basic price info
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[0]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0

            indicators['Current_Price'] = round(float(current_price), 2)
            indicators['Change'] = round(float(price_change), 2)
            indicators['Change_Pct'] = round(float(price_change_pct), 2)

            # Volume
            indicators['Current_Volume'] = int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0

            data = df.copy()
            close = data['Close'].astype(float)

            # --- RSI (14-period) ---
            if len(data) >= 14:
                delta = close.diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14, min_periods=14).mean()
                avg_loss = loss.rolling(window=14, min_periods=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                val = rsi.iloc[-1]
                if not np.isnan(val):
                    indicators['RSI_14'] = round(float(val), 2)

            # --- MACD (12, 26, 9) ---
            if len(data) >= 26:
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                macd_hist = macd_line - signal_line
                indicators['MACD'] = round(float(macd_line.iloc[-1]), 4)
                indicators['MACD_Signal'] = round(float(signal_line.iloc[-1]), 4)
                indicators['MACD_Hist'] = round(float(macd_hist.iloc[-1]), 4)

            # --- Simple Moving Averages ---
            if len(data) >= 20:
                indicators['SMA_20'] = round(float(close.tail(20).mean()), 2)
            if len(data) >= 50:
                indicators['SMA_50'] = round(float(close.tail(50).mean()), 2)
            if len(data) >= 200:
                indicators['SMA_200'] = round(float(close.tail(200).mean()), 2)

            # --- EMA 5 ---
            if len(data) >= 5:
                ema5 = close.ewm(span=5, adjust=False).mean()
                indicators['EMA_5'] = round(float(ema5.iloc[-1]), 2)

            # --- Bollinger Bands (20-period, 2 std) ---
            if len(data) >= 20:
                sma20 = close.rolling(window=20).mean()
                std20 = close.rolling(window=20).std()
                indicators['BB_Upper'] = round(float((sma20 + 2 * std20).iloc[-1]), 2)
                indicators['BB_Middle'] = round(float(sma20.iloc[-1]), 2)
                indicators['BB_Lower'] = round(float((sma20 - 2 * std20).iloc[-1]), 2)

            # --- ATR (14-period) ---
            if len(data) >= 14 and all(c in data.columns for c in ['High', 'Low', 'Close']):
                high = data['High'].astype(float)
                low = data['Low'].astype(float)
                prev_close = close.shift(1)
                tr = pd.concat([
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()
                ], axis=1).max(axis=1)
                atr = tr.rolling(window=14, min_periods=14).mean()
                val = atr.iloc[-1]
                if not np.isnan(val):
                    indicators['ATR_14'] = round(float(val), 4)

            # --- Volume averages ---
            if 'Volume' in data.columns and len(data) >= 10:
                vol = data['Volume'].astype(float)
                vol_10d = vol.tail(10).mean()
                indicators['Volume_10d_Avg'] = int(vol_10d)
                if len(data) >= 30:
                    vol_30d = vol.tail(30).mean()
                    indicators['Volume_30d_Avg'] = int(vol_30d)
                    if vol_30d > 0:
                        indicators['Volume_Ratio'] = round(float(vol.iloc[-1] / vol_30d), 2)

            # --- High/Low over last 20 days ---
            if len(data) >= 20:
                indicators['High_20d'] = round(float(data['High'].tail(20).max()), 2)
                indicators['Low_20d'] = round(float(data['Low'].tail(20).min()), 2)

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {"error": str(e)}

    def get_indicators(self, ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch data and calculate indicators.

        Args:
            ticker: Stock ticker symbol
            force_refresh: Bypass cache if True

        Returns:
            Dictionary with indicators and error info
        """
        # Use delta fetch to get only new data
        df = self.fetch_delta(ticker, force_refresh=force_refresh)
        if df is None or df.empty:
            return {"error": f"No data available for {ticker}"}

        indicators = self.calculate_indicators(df)
        indicators['Ticker'] = ticker
        indicators['Date'] = df.index[-1].strftime('%Y-%m-%d')
        return indicators
