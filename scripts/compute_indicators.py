#!/usr/bin/env python3
"""
Compute and store technical indicators from stock_daily data.

Reads stock price data from the database, calculates indicators,
and saves them to the indicators table.

Usage:
    python compute_indicators.py [--date YYYY-MM-DD] [--ticker TICKER]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import structlog

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.database import DatabaseManager
from src.fetchers import StockDataFetcher

# Setup logger
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
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()


def compute_and_store_indicators(db: DatabaseManager, ticker: str, as_of_date: str = None):
    """
    Compute indicators for a ticker and store them in the database.
    Stores only the latest day's indicators after truncation.

    Args:
        db: Database manager instance
        ticker: Stock ticker symbol
        as_of_date: Compute indicators as of this date (YYYY-MM-DD).
                    If None, uses latest available date.
    """
    end_date = as_of_date if as_of_date else datetime.now().strftime('%Y-%m-%d')

    # Fetch sufficient historical data from database
    start_date_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=400)
    start_date = start_date_dt.strftime('%Y-%m-%d')

    df = db.get_stock_data(ticker, start_date=start_date, end_date=end_date)

    if df.empty:
        logger.error(f"No stock data found for {ticker} in date range {start_date} to {end_date}")
        return False

    # Check if we have enough data
    MIN_DATA_REQUIRED = 200  # For SMA200
    if len(df) < MIN_DATA_REQUIRED:
        logger.warning(f"Insufficient data for {ticker}: have {len(df)} rows, need at least {MIN_DATA_REQUIRED}")

    # Calculate indicators using StockDataFetcher method
    fetcher = StockDataFetcher()
    indicators = fetcher.calculate_indicators(df)

    if indicators.get('Current_Price') is None:
        logger.error(f"Failed to compute indicators for {ticker}: no price data")
        return False

    # Determine the date to store: use the latest date in the data that is <= as_of_date
    latest_date_in_data = df.index.max().strftime('%Y-%m-%d')
    if as_of_date is None:
        store_date = latest_date_in_data
    else:
        store_date = as_of_date if as_of_date <= latest_date_in_data else latest_date_in_data

    # Save to database (latest day only)
    db.insert_latest_indicator(ticker, store_date, indicators)
    logger.info(f"Saved indicators for {ticker} as of {store_date}",
                price=indicators.get('Current_Price'),
                rsi=indicators.get('RSI_14'))

    return True


def main():
    parser = argparse.ArgumentParser(description="Compute technical indicators from stock data")
    parser.add_argument(
        "--date",
        dest="as_of_date",
        help="Compute indicators as of this date (YYYY-MM-DD). Default: latest available",
        type=str
    )
    parser.add_argument(
        "--ticker",
        help="Specific ticker to compute. If not provided, computes for all tickers in config",
        type=str
    )
    args = parser.parse_args()

    # Validate date format if provided
    if args.as_of_date:
        try:
            datetime.strptime(args.as_of_date, '%Y-%m-%d')
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            return 1

    try:
        # Load config
        config = Config.load()

        # Initialize database
        db = DatabaseManager(config.database.path)

        # Truncate indicators table before recalculating
        logger.info("Truncating indicators table for fresh calculation")
        db.truncate_indicators()

        # Determine which tickers to process
        tickers = [args.ticker] if args.ticker else config.tickers
        logger.info("Computing indicators", tickers=tickers, as_of_date=args.as_of_date or "latest")

        # Process each ticker
        success_count = 0
        fail_count = 0

        for ticker in tickers:
            try:
                if compute_and_store_indicators(db, ticker, args.as_of_date):
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"Failed to compute indicators for {ticker}", error=str(e), exc_info=True)
                fail_count += 1

        db.close()

        logger.info("Computation complete", success=success_count, failed=fail_count)
        print(f"✓ Completed: {success_count} succeeded, {fail_count} failed")
        return 0 if fail_count == 0 else 1

    except Exception as e:
        logger.error("Fatal error", error=str(e), exc_info=True)
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
