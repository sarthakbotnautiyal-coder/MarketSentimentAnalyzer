"""Console display using tabulate."""

import logging
from typing import Dict, Any, List
from tabulate import tabulate

logger = logging.getLogger(__name__)


class Display:
    """Handle console output formatting."""

    @staticmethod
    def print_indicators(ticker: str, indicators: Dict[str, Any]) -> None:
        """Print indicators table for a ticker."""
        if "error" in indicators:
            print(f"\n{ticker}: {indicators['error']}")
            return

        print(f"\n{'='*60}")
        print(f"Ticker: {ticker}")
        print(f"{'='*60}")

        # Prepare indicators table
        table_data = []
        for key, value in indicators.items():
            if value is not None:
                if isinstance(value, float):
                    formatted = f"{value:.2f}"
                else:
                    formatted = str(value)
                table_data.append([key, formatted])

        if table_data:
            print(tabulate(table_data, headers=["Indicator", "Value"], tablefmt="grid"))
        else:
            print("No indicators available")

    @staticmethod
    def print_news(ticker: str, articles: List[Dict[str, str]]) -> None:
        """Print news articles for a ticker."""
        print(f"\n{'='*60}")
        print(f"Latest News for {ticker}")
        print(f"{'='*60}")

        if not articles:
            print("No news articles found.")
            return

        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Published: {article.get('published', 'Unknown')}")
            print(f"   Summary: {article['snippet'][:200]}...")
            if article.get('url'):
                print(f"   URL: {article['url']}")

    @staticmethod
    def print_sentiment(ticker: str, sentiment: Dict[str, Any]) -> None:
        """Print sentiment analysis for a ticker."""
        print(f"\n{'='*60}")
        print(f"Market Sentiment for {ticker}")
        print(f"{'='*60}")

        sentiment_value = sentiment.get("sentiment", "N/A").upper()
        confidence = sentiment.get("confidence", 0.0)

        # Colorize based on sentiment
        if sentiment_value == "BULLISH":
            sentiment_display = f"\033[92m{sentiment_value}\033[0m"  # Green
        elif sentiment_value == "BEARISH":
            sentiment_display = f"\033[91m{sentiment_value}\033[0m"  # Red
        else:
            sentiment_display = f"\033[93m{sentiment_value}\033[0m"  # Yellow

        print(f"Sentiment: {sentiment_display}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Explanation: {sentiment.get('explanation', 'N/A')}")

    @staticmethod
    def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
        """Print summary across all tickers."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        summary_data = []
        for ticker, data in results.items():
            indicators = data.get("indicators", {})
            sentiment = data.get("sentiment", {})

            # Extract key metrics
            price = indicators.get("Current_Price", "N/A")
            change = indicators.get("Change", "N/A")
            rsi = indicators.get("RSI_14", "N/A")
            sentiment_value = sentiment.get("sentiment", "N/A").upper()

            summary_data.append([
                ticker,
                f"${price:.2f}" if isinstance(price, (int, float)) else price,
                f"{change:.2f}%" if isinstance(change, (int, float)) else change,
                f"{rsi:.1f}" if isinstance(rsi, (int, float)) else rsi,
                sentiment_value
            ])

        if summary_data:
            headers = ["Ticker", "Price", "Change", "RSI", "Sentiment"]
            print(tabulate(summary_data, headers=headers, tablefmt="grid"))
