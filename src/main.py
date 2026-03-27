"""Main entry point for MarketSentimentAnalyzer."""

import logging
import sys
from pathlib import Path
from typing import Dict, Any
import structlog

from src.config import Config
from src.fetchers import StockDataFetcher, NewsFetcher
from src.sentiment import SentimentAnalyzer
from src.display import Display

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class MarketSentimentAnalyzer:
    """Main application class."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.stock_fetcher = StockDataFetcher()
        self.news_fetcher = NewsFetcher(config.brave_api_key, config.news_count)
        self.sentiment_analyzer = SentimentAnalyzer(
            config.ollama_host,
            config.ollama_model
        )
        self.display = Display()
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """Run analysis for all tickers."""
        logger.info("Starting market sentiment analysis", tickers=self.config.tickers)

        for ticker in self.config.tickers:
            logger.info("Processing ticker", ticker=ticker)
            try:
                # Fetch indicators
                indicators = self.stock_fetcher.get_indicators(ticker)

                # Fetch news
                query = f"{ticker} stock news today"
                articles = self.news_fetcher.fetch_news(query)

                # Analyze sentiment
                if articles:
                    summaries = [a['snippet'] for a in articles if a.get('snippet')]
                    sentiment = self.sentiment_analyzer.analyze_batch(summaries)
                else:
                    sentiment = {"sentiment": "neutral", "confidence": 0.0, "explanation": "No news available"}

                # Store results
                self.results[ticker] = {
                    "indicators": indicators,
                    "news": articles,
                    "sentiment": sentiment
                }

                # Display results
                self.display.print_indicators(ticker, indicators)
                self.display.print_news(ticker, articles)
                self.display.print_sentiment(ticker, sentiment)

            except Exception as e:
                logger.error("Error processing ticker", ticker=ticker, error=str(e), exc_info=True)
                self.results[ticker] = {"error": str(e)}

        # Print summary
        self.display.print_summary(self.results)

        logger.info("Analysis complete", tickers_processed=len(self.results))
        return self.results


def main():
    """Main entry point."""
    try:
        config = Config.load()
        analyzer = MarketSentimentAnalyzer(config)
        results = analyzer.run()
        return 0
    except ValueError as e:
        logger.error("Configuration error", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        print("\nInterrupted by user.")
        return 130
    except Exception as e:
        logger.error("Unexpected error", error=str(e), exc_info=True)
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
