#!/usr/bin/env python3

Fetches stock data with incremental/delta updates, calculates technical indicators,
and stores only the latest day's indicators in the database for each ticker.

Usage:
    python -m src.main [--backfill 1y] [--force-refresh]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import structlog

# Setup logger early
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

from src.config import Config, DatabaseConfig
from src.database import DatabaseManager
from src.fetchers import StockDataFetcher
from src.display import Display

logger = structlog.get_logger()


def process_ticker(ticker: str, db: DatabaseManager, fetcher: StockDataFetcher,
                   force_refresh: bool = False) -> dict:
    """Process a single ticker: fetch data, calculate indicators, store in DB.

    Args:
        ticker: Stock ticker symbol
        db: Database manager instance
        fetcher: Stock data fetcher instance
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        Dictionary with ticker info and indicators
    """
    logger.info("Processing ticker", ticker=ticker)

    try:
        # Use delta fetch (incremental) unless forcing refresh
        if force_refresh or db is None:
            df = fetcher.fetch_data(ticker, force_refresh=True)
        else:
            df = fetcher.fetch_delta(ticker)

        if df is None or df.empty:
            logger.warning("No data available for ticker", ticker=ticker)
            return {
                "ticker": ticker,
                "indicators": {"error": "No data available"}
            }

        # Calculate indicators for latest day
        indicators = fetcher.calculate_indicators(df)

        # Add date from latest data
        latest_date = df.index[-1]
        date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        indicators['Date'] = date_str

        # Save indicators to DB (overwrites previous entry for this ticker due to PRIMARY KEY)
        if db and "error" not in indicators:
            db.save_indicators(ticker, date_str, indicators)

        # Display
        Display.print_indicators(ticker, indicators)

        return {
            "ticker": ticker,
            "indicators": indicators
        }

    except Exception as e:
        logger.error("Error processing ticker", ticker=ticker, error=str(e))
        error_result = {"ticker": ticker, "indicators": {"error": str(e)}}
        Display.print_indicators(ticker, error_result['indicators'])
        return error_result


def run_backfill(config: Config, db: DatabaseManager, tickers: list,
                 force_refresh: bool = False) -> dict:
    """Run backfill mode: fetch 1 year of data for all tickers.

    Args:
        config: Application configuration
        db: Database manager
        tickers: List of ticker symbols
        force_refresh: If True, bypass cache

    Returns:
        Dictionary of results by ticker
    """
    results = {}
    fetcher = StockDataFetcher(period="1y", db_manager=db)

    # Truncate indicators before backfill (implements "latest day only" logic)
    if db:
        db.truncate_indicators()

    for ticker in tickers:
        try:
            # Fetch 1 year of data
            df = fetcher.backfill_1year(ticker)

            if df is None or df.empty:
                logger.warning("Failed to backfill ticker", ticker=ticker)
                results[ticker] = {
                    "ticker": ticker,
                    "indicators": {"error": "Failed to backfill"}
                }
                continue

            # Calculate indicators for latest day
            indicators = fetcher.calculate_indicators(df)

            # Add date from latest data
            latest_date = df.index[-1]
            date_str = latest_date.strftime('%Y-%m-%d')
            indicators['Date'] = date_str

            # Save indicators (overwrites previous due to PRIMARY KEY on ticker)
            if db and "error" not in indicators:
                db.save_indicators(ticker, date_str, indicators)

            Display.print_indicators(ticker, indicators)
            results[ticker] = {"ticker": ticker, "indicators": indicators}

        except Exception as e:
            logger.error("Error backfilling ticker", ticker=ticker, error=str(e))
            error_result = {"ticker": ticker, "indicators": {"error": str(e)}}
            Display.print_indicators(ticker, error_result['indicators'])
            results[ticker] = error_result

    return results


def run_normal(config: Config, db: DatabaseManager, tickers: list,
               force_refresh: bool = False) -> dict:
    """Run normal mode: incremental fetch for all tickers.

    Args:
        config: Application configuration
        db: Database manager
        tickers: List of ticker symbols
        force_refresh: If True, bypass cache and fetch all fresh data

    Returns:
        Dictionary of results by ticker
    """
    results = {}
    fetcher = StockDataFetcher(period="5d", db_manager=db)

    # Truncate indicators first (keep only latest day)
    if db:
        db.truncate_indicators()

    for ticker in tickers:
        result = process_ticker(ticker, db, fetcher, force_refresh)
        results[ticker] = result

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--backfill',
        type=str,
        choices=['1y'],
        help='Backfill 1 year of historical data'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Bypass cache and fetch fresh data'
    )
    return parser.parse_args()


    """Main application class for stock technical analysis.

    This class provides backward compatibility and structure similar
    to the original implementation.
    """

    def __init__(self, config: Config, backfill: bool = False,
                 force_refresh: bool = False):
        """Initialize with configuration.

        Args:
            config: Application configuration
            backfill: Whether to backfill 1 year of data
            force_refresh: Whether to bypass cache
        """
        self.config = config
        self.backfill = backfill
        self.force_refresh = force_refresh

        # Initialize database connection
        self.db = None
        if config.database:
            self.db = DatabaseManager(config.database.path)

        logger.info(
            "Analyzer initialized",
            tickers_count=len(config.tickers),
            backfill=backfill,
            force_refresh=force_refresh
        )

    def run(self) -> dict:
        """Run analysis for all configured tickers.

        Returns:
            Dictionary mapping tickers to their results
        """
        tickers = self.config.tickers

        if self.backfill:
            results = run_backfill(
                self.config, self.db, tickers, self.force_refresh
            )
        else:
            results = run_normal(
                self.config, self.db, tickers, self.force_refresh
            )

        # Display summary
        Display.print_summary(results)

        # Clean up old stock data (keep last 30 days to manage DB size)
        if self.db:
            self.db.clear_old_stock_data(days=30)

        return results

    def cleanup(self):
        """Clean up resources."""
        if self.db:
            self.db.close()
            logger.info("Database connection closed")


def main():
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()

    try:
        # Load configuration
        config = Config.load()

        if not config.tickers:
            logger.error("No tickers configured")
            print("Error: No tickers configured. Check config/tickers.json")
            return 1

        # Initialize and run analyzer
            config,
            backfill=args.backfill is not None,
            force_refresh=args.force_refresh
        )

        try:
            analyzer.run()
        finally:
            analyzer.cleanup()

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        print(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())