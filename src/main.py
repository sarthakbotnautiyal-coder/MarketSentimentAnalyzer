"""Main entry point for MarketSentimentAnalyzer."""

import argparse
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Dict, Any
import structlog
from structlog.stdlib import ProcessorFormatter, filter_by_level, add_logger_name, add_log_level, PositionalArgumentsFormatter
from structlog.processors import TimeStamper, StackInfoRenderer, format_exc_info, UnicodeDecoder, JSONRenderer
from structlog.dev import ConsoleRenderer

from src.config import Config, DatabaseConfig
from src.database import DatabaseManager
from src.fetchers import StockDataFetcher, NewsFetcher
from src.sentiment import SentimentAnalyzer
from src.display import Display


def setup_logging(config: Config):
    """Configure structured logging with file rotation and console output."""
    # Determine log level from environment
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Create log directory if it doesn't exist
    log_dir = Path(config.database.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing handlers from root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(log_level)

    # Shared processors for structlog
    shared_processors = [
        filter_by_level,
        add_logger_name,
        add_log_level,
        PositionalArgumentsFormatter(),
        TimeStamper(fmt="iso"),
        StackInfoRenderer(),
        format_exc_info,
        UnicodeDecoder(),
        ProcessorFormatter.wrap_for_formatter,
    ]

    # Configure structlog to use these processors
    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # File handler with JSON formatting
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / "market_sentiment.log",
        when="midnight",
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_formatter = ProcessorFormatter(
        processor=JSONRenderer(),
        foreign_pre_chain=shared_processors[:-1],  # exclude wrap_for_formatter
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler with colored output if terminal
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_processor = ConsoleRenderer(colors=sys.stderr.isatty())
    console_formatter = ProcessorFormatter(
        processor=console_processor,
        foreign_pre_chain=shared_processors[:-1],
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)


logger = structlog.get_logger()


class MarketSentimentAnalyzer:
    """Main application class."""

    def __init__(self, config: Config, backfill: bool = False, force_refresh: bool = False):
        """Initialize with configuration and options."""
        self.config = config
        self.backfill = backfill
        self.force_refresh = force_refresh

        # Initialize database if caching is enabled
        self.db = None
        if config.database:
            try:
                self.db = DatabaseManager(config.database.path)
                logger.info("Database initialized", path=config.database.path)
            except Exception as e:
                logger.error("Failed to initialize database", error=str(e))
                self.db = None

        # Initialize components with database if available
        stock_period = config.database.stock_ttl if hasattr(config.database, 'stock_ttl') else "1d"
        self.stock_fetcher = StockDataFetcher(
            period="6mo",
            interval="1d",
            db_manager=self.db
        )
        self.news_fetcher = NewsFetcher(
            config.brave_api_key,
            config.news_count,
            db_manager=self.db,
            news_ttl_days=config.database.news_ttl_days
        )
        self.sentiment_analyzer = SentimentAnalyzer(
            config.ollama_host,
            config.ollama_model,
            db_manager=self.db,
            sentiment_ttl_days=config.database.sentiment_ttl_days
        )
        self.display = Display()
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """Run analysis for all tickers."""
        logger.info("Starting market sentiment analysis",
                   tickers=self.config.tickers,
                   backfill=self.backfill,
                   force_refresh=self.force_refresh)

        for ticker in self.config.tickers:
            logger.info("Processing ticker", ticker=ticker)
            try:
                # Handle backfill if requested
                if self.backfill:
                    logger.info(f"Backfilling 1 year data for {ticker}")
                    df = self.stock_fetcher.backfill_1year(ticker)
                    if df is None or df.empty:
                        logger.error(f"Backfill failed for {ticker}")
                        self.results[ticker] = {"error": "Backfill failed"}
                        continue
                    # After backfill, calculate indicators from the backfilled data
                    indicators = self.stock_fetcher.calculate_indicators(df)
                else:
                    # Normal operation: fetch indicators (with caching)
                    indicators = self.stock_fetcher.get_indicators(
                        ticker,
                        force_refresh=self.force_refresh
                    )

                    if "error" in indicators:
                        logger.error(f"Failed to get indicators for {ticker}", error=indicators["error"])
                        self.results[ticker] = {"error": indicators["error"]}
                        continue

                # Fetch news (with caching)
                query = f"{ticker} stock news today"
                articles = self.news_fetcher.fetch_news(
                    query,
                    ticker=ticker,
                    force_refresh=self.force_refresh
                )

                # Analyze sentiment (with caching)
                if articles:
                    summaries = [a['snippet'] for a in articles if a.get('snippet')]
                    sentiment = self.sentiment_analyzer.analyze_batch(
                        summaries,
                        ticker=ticker,
                        force_refresh=self.force_refresh
                    )
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

    def cleanup(self):
        """Cleanup resources."""
        if self.db:
            try:
                self.db.close()
            except Exception:
                pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Market Sentiment Analyzer")
    parser.add_argument(
        "--backfill",
        choices=["1y"],
        help="Backfill historical data (1y = 1 year)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh all data, bypassing cache"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        # Parse CLI args
        args = parse_args()

        # Load config
        config = Config.load()

        # Setup logging after config is loaded
        setup_logging(config)

        # Create analyzer
        analyzer = MarketSentimentAnalyzer(
            config=config,
            backfill=args.backfill is not None,
            force_refresh=args.force_refresh
        )

        # Run analysis
        results = analyzer.run()

        # Cleanup
        analyzer.cleanup()

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
