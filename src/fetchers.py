"""Data fetchers for stock indicators and news."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import requests

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetch stock data and calculate technical indicators."""

    def __init__(self, period: str = "6mo", interval: str = "1d"):
        """Initialize fetcher with period and interval."""
        self.period = period
        self.interval = interval

    def fetch_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch historical stock data for a ticker."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=self.period, interval=self.interval)
            if df.empty:
                logger.warning(f"No data fetched for {ticker}")
                return None
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators from price data."""
        indicators = {}

        # Ensure we have enough data
        if len(df) < 200:
            logger.warning(f"Insufficient data for some indicators: {len(df)} rows")
            return indicators

        # RSI (14 period)
        rsi = ta.rsi(df['Close'], length=14)
        indicators['RSI_14'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None

        # MACD
        macd = ta.macd(df['Close'])
        if macd is not None:
            indicators['MACD'] = macd['MACD_12_26_9'].iloc[-1]
            indicators['MACD_Signal'] = macd['MACDs_12_26_9'].iloc[-1]
            indicators['MACD_Hist'] = macd['MACDh_12_26_9'].iloc[-1]

        # SMA periods
        for period in [5, 10, 20, 50, 100, 200]:
            sma = ta.sma(df['Close'], length=period)
            if not sma.empty:
                indicators[f'SMA_{period}'] = sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else None

        # EMA periods
        for period in [5, 10, 20, 50, 100, 200]:
            ema = ta.ema(df['Close'], length=period)
            if not ema.empty:
                indicators[f'EMA_{period}'] = ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else None

        # Current price
        indicators['Current_Price'] = df['Close'].iloc[-1]

        # Previous close for change calculation
        if len(df) > 1:
            prev_close = df['Close'].iloc[-2]
            indicators['Change'] = ((indicators['Current_Price'] - prev_close) / prev_close) * 100
        else:
            indicators['Change'] = None

        return indicators

    def get_indicators(self, ticker: str) -> Dict[str, Any]:
        """Fetch data and calculate indicators for a ticker."""
        df = self.fetch_data(ticker)
        if df is None:
            return {"error": f"Failed to fetch data for {ticker}"}

        return self.calculate_indicators(df)


class NewsFetcher:
    """Fetch latest news using Brave Search API."""

    def __init__(self, api_key: str, count: int = 5):
        """Initialize with API key and number of results."""
        self.api_key = api_key
        self.count = count
        self.base_url = "https://api.search.brave.com/res/v1/web/search"

    def fetch_news(self, query: str) -> List[Dict[str, str]]:
        """Fetch news articles for a query."""
        # Build query with news focus and date constraint
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
