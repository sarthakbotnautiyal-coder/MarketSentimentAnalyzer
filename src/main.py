#!/usr/bin/env python3
"""
Fetches stock data, calculates technical indicators,
and filters tickers based on Stage 1 hard filter conditions.

Usage:
    python -m src.main [--backfill 1y] [--force-refresh]
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import structlog

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

from src.config import Config
from src.database import DatabaseManager
from src.fetchers import StockDataFetcher
from src.options_engine import Stage1HardFilters

logger = structlog.get_logger()


def process_ticker(ticker: str, db: DatabaseManager, fetcher: StockDataFetcher,
                   force_refresh: bool = False) -> dict:
    """Fetch data and calculate indicators for a single ticker."""
    logger.info("Processing ticker", ticker=ticker)

    try:
        if force_refresh or db is None:
            df = fetcher.fetch_data(ticker, force_refresh=True)
        else:
            df = fetcher.fetch_delta(ticker)

        if df is None or df.empty:
            logger.warning("No data available for ticker", ticker=ticker)
            return {"ticker": ticker, "indicators": {"error": "No data available"}}

        indicators = fetcher.get_indicators(ticker)
        latest_date = df.index[-1]
        date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        indicators['Date'] = date_str

        if db and "error" not in indicators:
            db.save_indicators(ticker, date_str, indicators)

        return {"ticker": ticker, "indicators": indicators}

    except Exception as e:
        logger.error("Error processing ticker", ticker=ticker, error=str(e))
        return {"ticker": ticker, "indicators": {"error": str(e)}}


def _compute_indicators_from_db(db: DatabaseManager, tickers: list) -> dict:
    """Compute indicators from full DB history for all tickers."""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    calc_fetcher = StockDataFetcher()

    results = {}
    for ticker in tickers:
        try:
            df = db.get_stock_data(ticker, start_date=start_date, end_date=end_date)
            if df.empty:
                logger.warning("No DB data for indicator computation", ticker=ticker)
                results[ticker] = {"ticker": ticker, "indicators": {"error": "No data in DB"}}
                continue

            indicators = calc_fetcher.get_indicators(ticker)
            if indicators.get('Current_Price') is None:
                results[ticker] = {"ticker": ticker, "indicators": {"error": "Computation failed"}}
                continue

            latest_date_in_data = df.index.max().strftime('%Y-%m-%d')
            indicators['Date'] = latest_date_in_data

            db.insert_latest_indicator(ticker, latest_date_in_data, indicators)
            results[ticker] = {"ticker": ticker, "indicators": indicators}

        except Exception as e:
            logger.error("Error computing indicators from DB", ticker=ticker, error=str(e))
            results[ticker] = {"ticker": ticker, "indicators": {"error": str(e)}}

    return results


def run_backfill(config: Config, db: DatabaseManager, tickers: list,
                 force_refresh: bool = False) -> dict:
    """Run backfill mode: fetch 1 year of data for all tickers."""
    results = {}
    fetcher = StockDataFetcher(period="1y", db_manager=db)

    if db:
        db.truncate_indicators()

    for ticker in tickers:
        try:
            df = fetcher.backfill_1year(ticker)
            if df is None or df.empty:
                results[ticker] = {"ticker": ticker, "indicators": {"error": "Failed to backfill"}}
                continue

            indicators = fetcher.get_indicators(ticker)
            latest_date = df.index[-1]
            date_str = latest_date.strftime('%Y-%m-%d')

            if db and "error" not in indicators:
                db.save_indicators(ticker, date_str, indicators)

            results[ticker] = {"ticker": ticker, "indicators": indicators}

        except Exception as e:
            logger.error("Error backfilling ticker", ticker=ticker, error=str(e))
            results[ticker] = {"ticker": ticker, "indicators": {"error": str(e)}}

    if db:
        db.truncate_indicators()
        results = _compute_indicators_from_db(db, tickers)

    return results


def run_normal(config: Config, db: DatabaseManager, tickers: list,
               force_refresh: bool = False) -> dict:
    """Run normal mode: incremental fetch for all tickers."""
    results = {}
    fetcher = StockDataFetcher(period="5d", db_manager=db)

    if db:
        db.truncate_indicators()

    for ticker in tickers:
        result = process_ticker(ticker, db, fetcher, force_refresh)
        results[ticker] = result

    if db:
        db.truncate_indicators()
        results = _compute_indicators_from_db(db, tickers)

    return results


def filter_favorable_tickers(results: dict, stage1: Stage1HardFilters) -> list:
    """Return list of tickers that pass Stage 1 hard filters."""
    favorable = []
    for ticker, result in results.items():
        indicators = result.get("indicators", {})
        if "error" in indicators:
            continue

        earnings_date = indicators.get("Next_Earnings_Date")
        candidates, _ = stage1.evaluate_all_with_warning(indicators, earnings_date)

        if candidates:
            favorable.append(ticker)

    return favorable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backfill', type=str, choices=['1y'], help='Backfill 1 year of historical data')
    parser.add_argument('--force-refresh', action='store_true', help='Bypass cache and fetch fresh data')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        config = Config.load()

        if not config.tickers:
            logger.error("No tickers configured")
            print("Error: No tickers configured. Check config/tickers.json")
            return 1

        db = None
        if config.database:
            db = DatabaseManager(config.database.path)

        date_str = datetime.now().strftime('%Y-%m-%d')

        if args.backfill:
            results = run_backfill(config, db, config.tickers, args.force_refresh)
        else:
            results = run_normal(config, db, config.tickers, args.force_refresh)

        stage1 = Stage1HardFilters()
        favorable = filter_favorable_tickers(results, stage1)

        print(f"\n{'='*60}")
        print(f"Favorable Tickers — {date_str}")
        print(f"{'='*60}")
        if favorable:
            for ticker in favorable:
                print(f"  ✓ {ticker}")
        else:
            print("  No tickers passed Stage 1 filters.")
        print(f"{'='*60}\n")

        if db:
            db.close()

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