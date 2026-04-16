#!/usr/bin/env python3
"""
Fetches stock data with incremental/delta updates, calculates technical indicators,
and stores only the latest day's indicators in the database for each ticker.

Usage:
    python -m src.main [--backfill 1y] [--force-refresh]
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
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
from src.options_engine import OptionsSignalEngine, SignalType

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
        indicators = fetcher.get_indicators(ticker)

        # Add date from latest data
        latest_date = df.index[-1]
        date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
        indicators['Date'] = date_str

        # Fetch earnings date
        earnings_date = fetcher.get_next_earnings_date(ticker)
        if earnings_date:
            indicators['Earnings_Date'] = earnings_date

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


def _compute_indicators_from_db(db: DatabaseManager, tickers: list) -> dict:
    """Compute indicators from full DB history for all tickers.

    Reads up to 400 days of stock data from the database to ensure sufficient
    history for indicators like SMA_200, MACD, etc. Updates the indicators
    table and the results dictionary.

    Args:
        db: Database manager instance
        tickers: List of ticker symbols

    Returns:
        Dictionary mapping tickers to updated indicator results
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    calc_fetcher = StockDataFetcher()  # No db_manager needed for calculate_indicators

    results = {}

    for ticker in tickers:
        try:
            df = db.get_stock_data(ticker, start_date=start_date, end_date=end_date)
            if df.empty:
                logger.warning("No DB data for indicator computation", ticker=ticker)
                results[ticker] = {"ticker": ticker, "indicators": {"error": "No data in DB"}}
                continue

            if len(df) < 200:
                logger.warning("Insufficient data for full indicators", ticker=ticker,
                               rows=len(df), need=200)

            indicators = calc_fetcher.get_indicators(ticker)

            if indicators.get('Current_Price') is None:
                logger.error("Failed to compute indicators", ticker=ticker)
                results[ticker] = {"ticker": ticker, "indicators": {"error": "Computation failed"}}
                continue

            latest_date_in_data = df.index.max().strftime('%Y-%m-%d')
            indicators['Date'] = latest_date_in_data

            db.insert_latest_indicator(ticker, latest_date_in_data, indicators)
            logger.info("Computed indicators from DB", ticker=ticker,
                        price=indicators.get('Current_Price'),
                        rsi=indicators.get('RSI_14'))

            results[ticker] = {"ticker": ticker, "indicators": indicators}

        except Exception as e:
            logger.error("Error computing indicators from DB", ticker=ticker, error=str(e))
            results[ticker] = {"ticker": ticker, "indicators": {"error": str(e)}}

    return results


def _save_signals_to_json(signals_obj, date_str: str, base_dir: Path = None) -> Path:
    """Save ticker signals to JSON file.

    Args:
        signals_obj: TickerSignals object
        date_str: Date string for directory structure
        base_dir: Base directory for signals (defaults to data/signals)

    Returns:
        Path to the saved JSON file
    """
    if base_dir is None:
        base_dir = Path("data/signals")
    signals_dir = base_dir / date_str
    signals_dir.mkdir(parents=True, exist_ok=True)

    output_path = signals_dir / f"{signals_obj.ticker}.json"
    with open(output_path, "w") as f:
        json.dump(signals_obj.to_dict(), f, indent=2)

    logger.info("Saved signals", ticker=signals_obj.ticker, path=str(output_path))
    return output_path


def _print_signal_summary(signals_obj) -> None:
    """Print a concise signal summary to stdout.

    Args:
        signals_obj: TickerSignals object
    """
    ticker = signals_obj.ticker
    price = signals_obj.current_price
    print(f"\n📊 {ticker} Options Signals (${price:.2f})")

    for signal in signals_obj.signals:
        emoji = {
            "SELL_PUTS": "🟢",
            "SELL_CALLS": "🔴",
            "BUY_LEAPS": "🟡",
            "HOLD": "⚪",
            "NEUTRAL": "⚪",
        }.get(signal.signal_type.value, "⚪")

        conf_emoji = {
            "HIGH": "■■■",
            "MEDIUM": "■■○",
            "LOW": "■○○",
            "NONE": "○○○",
        }.get(signal.confidence.value, "○○○")

        print(f"  {emoji} {signal.signal_type.value} [{conf_emoji}]")

        # Print key reasoning points (first 2)
        for reason in signal.reasoning[:2]:
            print(f"      • {reason}")

        if signal.target_price:
            print(f"      • Target: ${signal.target_price:.2f} | Stop: ${signal.stop_loss:.2f}")


def _generate_and_save_signals(results: dict, date_str: str, engine: OptionsSignalEngine) -> dict:
    """Generate signals for all tickers with valid indicators and save to JSON.

    Args:
        results: Dictionary of {ticker: {"ticker": ..., "indicators": ...}}
        date_str: Date string for directory structure
        engine: OptionsSignalEngine instance

    Returns:
        Dictionary of {ticker: TickerSignals}
    """
    signals_results = {}
    all_signals = []

    for ticker, result in results.items():
        indicators = result.get("indicators", {})
        if "error" in indicators:
            logger.debug("Skipping signal generation for error ticker", ticker=ticker)
            continue

        try:
            ticker_signals = engine.generate_signals_for_ticker(ticker, indicators)
            signals_results[ticker] = ticker_signals
            all_signals.append(ticker_signals.to_dict())

            # Save per-ticker JSON
            _save_signals_to_json(ticker_signals, date_str)

            # Print signal summary
            _print_signal_summary(ticker_signals)

        except Exception as e:
            logger.error("Error generating signals", ticker=ticker, error=str(e))

    # Save summary JSON
    if all_signals:
        summary_path = Path("data/signals") / date_str / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({
                "date": date_str,
                "generated_at": datetime.now().isoformat(),
                "tickers": all_signals
            }, f, indent=2)
        logger.info("Saved signals summary", path=str(summary_path))

    return signals_results


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
            indicators = fetcher.get_indicators(ticker)

            # Add date from latest data
            latest_date = df.index[-1]
            date_str = latest_date.strftime('%Y-%m-%d')
            # Fetch earnings date
            earnings_date = fetcher.get_next_earnings_date(ticker)
            if earnings_date:
                indicators['Earnings_Date'] = earnings_date
    
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

    # Compute indicators from full DB history (ensures SMA_200, MACD, etc. have enough data)
    if db:
        logger.info("Computing indicators from full DB history after backfill")
        db.truncate_indicators()
        results = _compute_indicators_from_db(db, tickers)
        for ticker, result in results.items():
            if "error" not in result.get("indicators", {}):
                Display.print_indicators(ticker, result["indicators"])

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

    # Compute indicators from full DB history (ensures SMA_200, MACD, etc. have enough data)
    if db:
        logger.info("Computing indicators from full DB history")
        db.truncate_indicators()
        results = _compute_indicators_from_db(db, tickers)
        for ticker, result in results.items():
            if "error" not in result.get("indicators", {}):
                Display.print_indicators(ticker, result["indicators"])

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


class Analyzer:
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

        # Initialize signal engine
        self.signal_engine = OptionsSignalEngine()

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
        date_str = datetime.now().strftime('%Y-%m-%d')

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

        # Generate and save signals for all tickers
        print("\n" + "="*60)
        print("OPTIONS SIGNALS")
        print("="*60)
        _generate_and_save_signals(results, date_str, self.signal_engine)
        print("="*60 + "\n")

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
        analyzer = Analyzer(
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
