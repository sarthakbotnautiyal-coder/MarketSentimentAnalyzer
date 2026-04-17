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

    def get_implied_volatility(self, ticker: str) -> Optional[float]:
        """Fetch implied volatility from ATM option in the options chain.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Implied volatility as a percentage (e.g., 25.5 for 25.5%) or None on failure
        """
        try:
            yf_ticker = yf.Ticker(ticker)

            # Get current price from recent history
            hist = yf_ticker.history(period="5d")
            if hist.empty:
                logger.warning(f"No price data for IV calculation of {ticker}")
                return None
            current_price = hist['Close'].iloc[-1]

            # Get available option expirations
            expirations = yf_ticker.options
            if not expirations or len(expirations) < 2:
                logger.warning(f"No options data available for {ticker}")
                return None

            # Use second expiry (first may be too near-term / illiquid)
            expiry = expirations[1]
            opt_chain = yf_ticker.option_chain(expiry)

            if opt_chain.calls.empty:
                logger.warning(f"No call options for {ticker} at expiry {expiry}")
                return None

            # Find ATM call option (strike closest to current price)
            calls = opt_chain.calls
            atm_idx = (calls['strike'] - current_price).abs().idxmin()
            atm_call = calls.loc[atm_idx]
            iv = atm_call['impliedVolatility']

            if pd.isna(iv):
                logger.warning(f"IV is NaN for ATM call of {ticker}")
                return None

            # Return as percentage (e.g., 0.25 -> 25.0)
            return round(float(iv) * 100, 2)

        except Exception as e:
            logger.warning(f"Error fetching implied volatility for {ticker}: {e}")
            return None

    def calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20) -> Optional[float]:
        """Calculate annualized historical volatility over a given period.

        Args:
            df: DataFrame with OHLCV data
            period: Lookback period in trading days (default: 20)

        Returns:
            Annualized historical volatility as percentage, or None if insufficient data
        """
        if df.empty or len(df) < period + 1:
            return None

        close = df['Close'].astype(float)
        log_returns = np.log(close / close.shift(1)).dropna()

        if len(log_returns) < period:
            return None

        hv = log_returns.tail(period).std() * np.sqrt(252) * 100
        return round(float(hv), 2)

    def calculate_iv_rank(self, hv_52w_min: float, hv_52w_max: float,
                          current_iv: float) -> Optional[float]:
        """Calculate IV Rank - where current IV sits in the 52-week range.

        IV Rank = ((Current IV - 52w Low) / (52w High - 52w Low)) * 100

        Uses HV as a proxy when historical IV data is unavailable.

        Args:
            hv_52w_min: 52-week minimum historical volatility
            hv_52w_max: 52-week maximum historical volatility
            current_iv: Current implied volatility

        Returns:
            IV Rank as percentage (0-100), or None if range is zero
        """
        if hv_52w_max == hv_52w_min:
            return None
        iv_rank = ((current_iv - hv_52w_min) / (hv_52w_max - hv_52w_min)) * 100
        return round(max(0.0, min(100.0, iv_rank)), 2)

    def calculate_iv_percentile(self, hv_series: pd.Series, current_iv: float) -> Optional[float]:
        """Calculate IV Percentile - percentage of days IV was below current level.

        IV Percentile = (Number of days with IV < Current IV / Total days) * 100

        Uses HV series as proxy when historical IV data is unavailable.

        Args:
            hv_series: Series of daily historical volatility values (252 trading days)
            current_iv: Current implied volatility

        Returns:
            IV Percentile as percentage (0-100), or None if insufficient data
        """
        if hv_series.empty or hv_series.isna().all():
            return None
        valid = hv_series.dropna()
        if len(valid) == 0:
            return None
        percentile = (valid < current_iv).sum() / len(valid) * 100
        return round(float(percentile), 2)

    def get_next_earnings_date(self, ticker: str) -> Optional[str]:
        """Fetch next earnings date for a ticker from yfinance calendar.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Earnings date as YYYY-MM-DD string, or None if unavailable
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            cal = yf_ticker.calendar

            if cal is None:
                logger.debug(f"No calendar data for {ticker}")
                return None

            # yfinance calendar can be a dict or DataFrame depending on version
            if isinstance(cal, dict):
                earnings_date = cal.get('Earnings Date')
                if earnings_date is not None:
                    if isinstance(earnings_date, (list, np.ndarray)) and len(earnings_date) > 0:
                        earnings_date = earnings_date[0]
                    if isinstance(earnings_date, (datetime, pd.Timestamp)):
                        return earnings_date.strftime('%Y-%m-%d')
                    if pd.notna(earnings_date):
                        return str(earnings_date)
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for idx, row in cal.iterrows():
                    if 'Earnings' in str(idx):
                        next_earnings = row.iloc[0]
                        if pd.notna(next_earnings):
                            if isinstance(next_earnings, (datetime, pd.Timestamp)):
                                return next_earnings.strftime('%Y-%m-%d')
                            return str(next_earnings)

            return None
        except Exception as e:
            logger.warning(f"Error fetching earnings date for {ticker}: {e}")
            return None

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
                avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
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

            # --- Historical Volatility (20-day and 30-day, annualized) ---
            log_returns = np.log(close / close.shift(1)).dropna()

            if len(log_returns) >= 20:
                hv_20d = log_returns.tail(20).std() * np.sqrt(252) * 100
                indicators['Historical_Volatility_20d'] = round(float(hv_20d), 2)
                # Keep legacy key for backward compatibility
                indicators['Hist_Volatility_20d'] = round(float(hv_20d), 2)

            if len(log_returns) >= 30:
                hv_30d = log_returns.tail(30).std() * np.sqrt(252) * 100
                indicators['Historical_Volatility_30d'] = round(float(hv_30d), 2)

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {"error": str(e)}

    def get_indicators(self, ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetch data and calculate indicators including volatility and earnings.

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

        # --- Implied Volatility from options chain ---
        iv = self.get_implied_volatility(ticker)
        if iv is not None:
            indicators['Implied_Volatility'] = iv

        # --- IV Rank and IV Percentile (using HV as proxy) ---
        # Calculate daily HV series for the full dataset
        close = df['Close'].astype(float)
        log_returns = np.log(close / close.shift(1)).dropna()

        # Rolling 20-day HV for the entire series
        if len(log_returns) >= 20:
            hv_series = log_returns.rolling(20).std() * np.sqrt(252) * 100
            hv_series = hv_series.dropna()

            if len(hv_series) > 0:
                # Use current IV if available, else current 20d HV
                reference_vol = iv if iv is not None else indicators.get('Historical_Volatility_20d')

                if reference_vol is not None:
                    # 52-week range (last ~252 trading days)
                    hv_52w = hv_series.tail(252)
                    if len(hv_52w) > 1:
                        hv_min = float(hv_52w.min())
                        hv_max = float(hv_52w.max())
                        iv_rank = self.calculate_iv_rank(hv_min, hv_max, reference_vol)
                        if iv_rank is not None:
                            indicators['IV_Rank'] = iv_rank

                    # IV Percentile over available history
                    iv_pct = self.calculate_iv_percentile(hv_series, reference_vol)
                    if iv_pct is not None:
                        indicators['IV_Percentile'] = iv_pct

        # --- Next Earnings Date ---
        earnings_date = self.get_next_earnings_date(ticker)
        if earnings_date is not None:
            indicators['Next_Earnings_Date'] = earnings_date

        return indicators
