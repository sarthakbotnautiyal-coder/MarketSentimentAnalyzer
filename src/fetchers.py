"""Data fetchers for stock indicators and news with caching."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import requests
import structlog

# Conditionally import pandas_ta - optional due to numba/Python compatibility
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    ta = None
    structlog.get_logger(__name__).warning("pandas_ta not available - using mock indicators")

logger = structlog.get_logger()


class StockDataFetcher:
    """Fetch stock data and calculate technical indicators with caching."""

    def __init__(self, period: str = "1y", interval: str = "1d", db_manager: Any = None, ttl_days: int = 1):
        """Initialize fetcher with period, interval, optional database cache and TTL in days."""
        self.period = period
        self.interval = interval
        self.db = db_manager
        self.ttl_days = ttl_days

    def fetch_data(self, ticker: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data for a ticker using cache if available and fresh.
        Implements incremental updates: if cache is stale, fetch only new data.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame with stock data or None if failed
        """
        # If database available and not forcing refresh, check cache freshness
        if self.db and not force_refresh:
            try:
                if self.db.is_data_fresh('stock_daily', ticker, self.ttl_days):
                    # Cache is fresh, return it
                    logger.info(f"Using fresh cached data for {ticker}")
                    return self.db.get_stock_data(ticker)
                
                # Cache is stale or empty - need to fetch new data
                latest_date_str = self.db.get_latest_stock_date(ticker)
                if latest_date_str:
                    # Incremental fetch: get data from day after latest cached date
                    latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
                    start_date = latest_date + timedelta(days=1)
                    if start_date > datetime.now():
                        # Already have up-to-date data (latest date is today or future)
                        logger.info(f"Cache for {ticker} already up-to-date")
                        return self.db.get_stock_data(ticker)
                    
                    logger.info(f"Fetching incremental data for {ticker} from {start_date.date()}")
                    try:
                        stock = yf.Ticker(ticker)
                        df = stock.history(start=start_date, interval=self.interval)
                    except Exception as e:
                        logger.error(f"Error during incremental fetch for {ticker}: {e}")
                        # On failure, fall back to stale cache
                        logger.warning(f"Using stale cache for {ticker} due to fetch failure")
                        return self.db.get_stock_data(ticker)
                    
                    if df.empty:
                        logger.info(f"No new data for {ticker}")
                        return self.db.get_stock_data(ticker)
                    
                    # Save new data to database (will upsert)
                    self.db.save_stock_data(ticker, df)
                    # Return combined data (cached + new)
                    logger.info(f"Fetched {len(df)} new rows for {ticker}")
                    return self.db.get_stock_data(ticker)
            except Exception as e:
                logger.warning(f"Failed to read from cache for {ticker}: {e}")
                # Will fall through to fresh fetch below

        # Full fetch (no db, force_refresh, or cache miss/stale error)
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=self.period, interval=self.interval)
            if df.empty:
                logger.warning(f"No data fetched for {ticker}")
                return None
            logger.info(f"Fetched {len(df)} rows for {ticker} from yfinance")
            
            # Save to database if available
            if self.db:
                try:
                    self.db.save_stock_data(ticker, df)
                except Exception as e:
                    logger.error(f"Failed to save to database for {ticker}: {e}")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            # Try to return cached data on fetch failure even if force_refresh was set
            if self.db:
                try:
                    cached_df = self.db.get_stock_data(ticker)
                    if not cached_df.empty:
                        logger.warning(f"API failed, using stale cached data for {ticker}")
                        return cached_df
                except Exception:
                    pass
            return None

    def backfill_1year(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch 1 year of historical data for initial backfill.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with 1 year of stock data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period='1y', interval=self.interval)
            if df.empty:
                logger.warning(f"No data fetched for {ticker} during backfill")
                return None
            
            logger.info(f"Backfilled {len(df)} rows for {ticker}")
            
            if self.db:
                try:
                    self.db.save_stock_data(ticker, df)
                    logger.info(f"Saved backfill data to database for {ticker}")
                except Exception as e:
                    logger.error(f"Failed to save backfill to database for {ticker}: {e}")
            
            return df
        except Exception as e:
            logger.error(f"Error during backfill for {ticker}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators from price data."""
        indicators = {}

        # Log if data is limited for some indicators
        if len(df) < 200:
            logger.warning(f"Limited data: {len(df)} rows; some indicators may be unavailable")

        if PANDAS_TA_AVAILABLE:
            # RSI (14 period)
            if len(df) >= 14:
                rsi = ta.rsi(df['Close'], length=14)
                indicators['RSI_14'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None
            else:
                indicators['RSI_14'] = None

            # MACD
            try:
                macd = ta.macd(df['Close'])
                if macd is not None and not macd.empty:
                    indicators['MACD'] = macd['MACD_12_26_9'].iloc[-1]
                    indicators['MACD_Signal'] = macd['MACDs_12_26_9'].iloc[-1]
                    indicators['MACD_Hist'] = macd['MACDh_12_26_9'].iloc[-1]
                else:
                    indicators['MACD'] = indicators['MACD_Signal'] = indicators['MACD_Hist'] = None
            except Exception:
                indicators['MACD'] = indicators['MACD_Signal'] = indicators['MACD_Hist'] = None

            # SMA periods
            for period in [5, 10, 20, 50, 100, 200]:
                if len(df) >= period:
                    sma = ta.sma(df['Close'], length=period)
                    if not sma.empty:
                        indicators[f'SMA_{period}'] = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else None
                    else:
                        indicators[f'SMA_{period}'] = None
                else:
                    indicators[f'SMA_{period}'] = None

            # EMA periods
            for period in [5, 10, 20, 50, 100, 200]:
                if len(df) >= period:
                    ema = ta.ema(df['Close'], length=period)
                    if not ema.empty:
                        indicators[f'EMA_{period}'] = ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else None
                    else:
                        indicators[f'EMA_{period}'] = None
                else:
                    indicators[f'EMA_{period}'] = None
        else:
            # Mock indicators when pandas_ta is unavailable
            if len(df) > 0:
                current_price = df['Close'].iloc[-1]
                indicators['RSI_14'] = 50.0  # Neutral RSI
                indicators['MACD'] = 0.0
                indicators['MACD_Signal'] = 0.0
                indicators['MACD_Hist'] = 0.0
                for period in [5, 10, 20, 50, 100, 200]:
                    indicators[f'SMA_{period}'] = current_price if len(df) >= period else None
                    indicators[f'EMA_{period}'] = current_price if len(df) >= period else None
            else:
                indicators['RSI_14'] = None
                indicators['MACD'] = None
                indicators['MACD_Signal'] = None
                indicators['MACD_Hist'] = None
                for period in [5, 10, 20, 50, 100, 200]:
                    indicators[f'SMA_{period}'] = None
                    indicators[f'EMA_{period}'] = None

        # Current price
        if len(df) > 0:
            indicators['Current_Price'] = df['Close'].iloc[-1]
        else:
            indicators['Current_Price'] = None

        # Previous close for change calculation
        if len(df) > 1 and indicators['Current_Price'] is not None:
            prev_close = df['Close'].iloc[-2]
            indicators['Change'] = ((indicators['Current_Price'] - prev_close) / prev_close) * 100
        else:
            indicators['Change'] = None

        return indicators

    def get_indicators(self, ticker: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch data and calculate indicators for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Dictionary of indicators or error
        """
        df = self.fetch_data(ticker, force_refresh=force_refresh)
        if df is None or df.empty:
            return {"error": f"Failed to fetch data for {ticker}"}

        return self.calculate_indicators(df)


class NewsFetcher:
    """Fetch latest news using Brave Search API with caching."""

    def __init__(self, api_key: str, count: int = 5, db_manager: Any = None, news_ttl_days: int = 7):
        """Initialize with API key, number of results, optional database cache and TTL."""
        self.api_key = api_key
        self.count = count
        self.db = db_manager
        self.news_ttl_days = news_ttl_days
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def _fetch_fresh_news(self, query: str) -> List[Dict[str, str]]:
        """Fetch fresh news from Brave API."""
        params = {
            "q": query,
            "count": self.count * 2,  # Fetch more to filter
            "freshness": "week",  # Prefer recent results
        }

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            results = response.json()

            articles = []
            for item in results.get("web", {}).get("results", [])[:self.count]:
                articles.append({
                    "title": item.get("title", "No title"),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", "No summary"),
                    "published": item.get("published_time", "Unknown")
                })

            return articles
        except requests.RequestException as e:
            logger.error(f"Error fetching news for '{query}': {e}")
            return []
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing news response: {e}")
            return []

    def fetch_news(self, query: str, ticker: str = None, force_refresh: bool = False) -> List[Dict[str, str]]:
        """
        Fetch news articles for a query with caching support.
        
        Args:
            query: Search query
            ticker: Stock ticker (for cache key, if None uses query as identifier)
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of news articles
        """
        ticker = ticker or query.split()[0]  # Use first word as ticker if not provided
        
        # Check cache first if database available and not forcing refresh
        if self.db and not force_refresh:
            try:
                if self.db.is_data_fresh('news', ticker, self.news_ttl_days):
                    cached_news = self.db.get_cached_news(ticker, days_back=self.news_ttl_days)
                    if cached_news:
                        logger.info(f"Using cached news for {ticker} ({len(cached_news)} articles)")
                        return cached_news[:self.count]
            except Exception as e:
                logger.warning(f"Failed to read news cache for {ticker}: {e}")

        # Fetch fresh news
        articles = self._fetch_fresh_news(query)
        
        if articles:
            # Save to cache
            if self.db:
                try:
                    self.db.save_news(ticker, articles)
                except Exception as e:
                    logger.error(f"Failed to save news to database for {ticker}: {e}")
        
        return articles[:self.count]
