"""Data fetchers with caching for stock data."""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import structlog
import pandas as pd
import yfinance as yf

logger = structlog.get_logger()


class StockDataFetcher:
    """Fetch and cache stock market data."""

    def __init__(self, period: str = "5d", interval: str = "1d",
                 db_manager=None, ttl_days: int = 1):
        """Initialize with caching database manager.

        Args:
            period: Default period to fetch (e.g., "5d", "1y")
            interval: Data interval (e.g., "1d", "1h")
            db_manager: Optional DatabaseManager instance for caching
            ttl_days: Cache TTL for stock data
        """
        self.period = period
        self.interval = interval
        self.db = db_manager
        self.ttl_days = ttl_days

    def _is_cache_usable(self, ticker: str, force_refresh: bool = False) -> bool:
        """Check if we should use cached data."""
        if force_refresh or not self.db:
            return False
        latest_date = self.db.get_latest_stock_date(ticker)
        if latest_date is None:
            return False
        # Check if cache is fresh
        cache_date = datetime.strptime(latest_date, '%Y-%m-%d').date()
        today = datetime.now().date()
        return (today - cache_date).days < self.ttl_days

    def fetch_data(self, ticker: str, period: str = None,
                   force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Fetch stock data using yfinance with caching.

        Args:
            ticker: Stock ticker symbol
            period: Period to fetch (e.g., "5d", "1y")
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            DataFrame with stock data or None on error
        """
        period = period or self.period

        # Try to use cached data if valid and not forcing refresh
        if self._is_cache_usable(ticker, force_refresh):
            logger.info(f"Using cached stock data for {ticker}")
            return self.db.get_stock_data(ticker)

        # Fetch fresh data
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period=period, interval=self.interval)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            # Clean up index to be plain dates
            df.index = df.index.tz_localize(None) if df.index.tz else df.index

            logger.info(f"Fetched {len(df)} rows of stock data for {ticker}")

            # Cache to database
            if self.db:
                self.db.save_stock_data(ticker, df)

            return df
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            # Fallback to stale cache if available
            if self.db:
                logger.info(f"Falling back to cached data for {ticker}")
                return self.db.get_stock_data(ticker)
            return None

    def fetch_delta(self, ticker: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """Fetch only new data since last cached date.

        Args:
            ticker: Stock ticker symbol
            force_refresh: If True, fetch full history instead of delta

        Returns:
            DataFrame with stock data (combined from cache + new) or None
        """
        if force_refresh or not self.db:
            return self.fetch_data(ticker, force_refresh=force_refresh)

        latest_date = self.db.get_latest_stock_date(ticker)
        if latest_date is None:
            # No cache - do full fetch
            logger.info(f"No cache for {ticker}, fetching full history")
            return self.fetch_data(ticker, force_refresh=True)

        # Calculate delta start date
        latest = datetime.strptime(latest_date, '%Y-%m-%d').date()
        today = datetime.now().date()

        if latest >= today - timedelta(days=1):
            # Cache is up to date, just return it
            logger.info(f"Cache is current for {ticker}")
            return self.db.get_stock_data(ticker)

        # Need delta fetch - use full fetch but let yfinance handle it
        # yfinance will return full period, we'll merge with cache
        try:
            yf_ticker = yf.Ticker(ticker)
            # Fetch minimal period to get latest data
            df = yf_ticker.history(period="5d", interval=self.interval)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return self.db.get_stock_data(ticker)

            # Clean up index
            df.index = df.index.tz_localize(None) if df.index.tz else df.index

            # Get cached data
            cached_df = self.db.get_stock_data(ticker)

            # Combine: use cached data plus any new rows from fetch
            if cached_df is not None and not cached_df.empty:
                # Filter out dates we already have
                new_dates = df.index.difference(cached_df.index)
                if len(new_dates) > 0:
                    new_df = df.loc[new_dates]
                    combined = pd.concat([cached_df, new_df])
                    combined = combined.sort_index()
                    logger.info(f"Delta fetch added {len(new_df)} rows for {ticker}")
                else:
                    combined = cached_df
                    logger.info(f"No new data for {ticker}")
            else:
                combined = df
                logger.info(f"Fetched {len(df)} rows for {ticker}")

            # Save combined data
            if self.db:
                self.db.save_stock_data(ticker, combined)

            return combined
        except Exception as e:
            logger.error(f"Error in delta fetch for {ticker}: {e}")
            # Fallback to cache
            return self.db.get_stock_data(ticker)

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
            DataFrame with stock data (combined from cache + new) or None
        """
        return self.fetch_delta(ticker, force_refresh=False)

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for latest price data.

        Args:
            df: DataFrame with stock data

        Returns:
            Dictionary of calculated indicators for the latest day
        """
        if df.empty:
            return {"error": "No data available"}

        # Get latest price data
        try:
            indicators = {}

            # Basic price info
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[0]
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0

            indicators['Current_Price'] = round(current_price, 2)
            indicators['Change'] = round(price_change, 2)
            indicators['Change_Pct'] = round(price_change_pct, 2)

            # Volume
            indicators['Current_Volume'] = int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0

            # Calculate technical indicators if pandas_ta is available
            try:
                import pandas_ta as ta

                # Make a copy to avoid SettingWithCopy warnings
                data = df.copy()

                # RSI
                if len(data) >= 14:
                    rsi = ta.rsi(data['Close'], length=14)
                    if rsi is not None and len(rsi) > 0:
                        indicators['RSI_14'] = float(rsi.iloc[-1])

                # MACD
                if len(data) >= 26:
                    macd = ta.macd(data['Close'])
                    if macd is not None:
                        indicators['MACD'] = float(macd['MACD_12_26_9'].iloc[-1])
                        indicators['MACD_Signal'] = float(macd['MACDs_12_26_9'].iloc[-1])
                        indicators['MACD_Hist'] = float(macd['MACDh_12_26_9'].iloc[-1])

                # Moving Averages
                if len(data) >= 20:
                    indicators['SMA_20'] = float(ta.sma(data['Close'], length=20).iloc[-1])
                if len(data) >= 50:
                    indicators['SMA_50'] = float(ta.sma(data['Close'], length=50).iloc[-1])
                if len(data) >= 200:
                    indicators['SMA_200'] = float(ta.sma(data['Close'], length=200).iloc[-1])

                if len(data) >= 5:
                    indicators['EMA_5'] = float(ta.ema(data['Close'], length=5).iloc[-1])

                # Bollinger Bands
                if len(data) >= 20:
                    bb = ta.bbands(data['Close'], length=20, std=2.0)
                    if bb is not None:
                        indicators['BB_Upper'] = float(bb['BBU_20_2.0'].iloc[-1])
                        indicators['BB_Middle'] = float(bb['BBM_20_2.0'].iloc[-1])
                        indicators['BB_Lower'] = float(bb['BBL_20_2.0'].iloc[-1])

                # ATR
                if len(data) >= 14:
                    atr = ta.atr(data['High'], data['Low'], data['Close'], length=14)
                    if atr is not None and len(atr) > 0:
                        indicators['ATR_14'] = float(atr.iloc[-1])

                # Volume averages
                if len(data) >= 10:
                    vol_10d = data['Volume'].tail(10).mean()
                    indicators['Volume_10d_Avg'] = int(vol_10d)
                if len(data) >= 30:
                    vol_30d = data['Volume'].tail(30).mean()
                    indicators['Volume_30d_Avg'] = int(vol_30d)
                    if vol_30d > 0:
                        indicators['Volume_Ratio'] = round(data['Volume'].iloc[-1] / vol_30d, 2)

                # High/Low over last 20 days
                if len(data) >= 20:
                    high_20d = data['High'].tail(20).max()
                    low_20d = data['Low'].tail(20).min()
                    indicators['High_20d'] = float(high_20d)
                    indicators['Low_20d'] = float(low_20d)

            except ImportError:
                logger.debug("pandas_ta not available, using basic indicators")
            except Exception as e:
                logger.warning(f"Error calculating indicators: {e}")

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
        return indicators
