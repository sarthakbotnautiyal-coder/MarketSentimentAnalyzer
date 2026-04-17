#!/usr/bin/env python3
"""
CLI script to generate options trading signals for a ticker using stored indicators.
Usage: python scripts/generate_signals.py --ticker AAPL [--date 2026-03-29]
Outputs structured JSON.

Supports two modes:
  --legacy   : Use original OptionsSignalEngine (backward compatible)
  --hybrid   : Use new HybridSignalPipeline (Stage1 + Stage2) [default]
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
import structlog

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from options_engine import OptionsSignalEngine, HybridSignalPipeline

logger = structlog.get_logger()


def get_indicators(db_path: str, ticker: str, target_date: str = None) -> dict:
    """
    Query indicators and stock_daily for the ticker/date.
    Maps DB columns to engine expected keys.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if target_date:
        cursor.execute(
            """
            SELECT i.*, s.close as current_price
            FROM indicators i
            JOIN stock_daily s ON i.date = s.date AND i.ticker = s.ticker
            WHERE i.ticker = ? AND i.date = ?
            """,
            (ticker, target_date)
        )
    else:
        # Latest available
        cursor.execute(
            """
            SELECT i.*, s.close as current_price
            FROM indicators i
            JOIN stock_daily s ON i.date = s.date AND i.ticker = s.ticker
            WHERE i.ticker = ?
            ORDER BY i.date DESC
            LIMIT 1
            """,
            (ticker,)
        )

    row = cursor.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"No indicators data found for {ticker} on {target_date or 'latest date'}")

    # Map DB columns to engine keys
    macd_signal = row['macd'] - row['macd_hist'] if row['macd'] is not None and row['macd_hist'] is not None else None

    indicators = {
        'Ticker': ticker,
        'RSI_14': row['rsi'],
        'MACD': row['macd'],
        'MACD_Signal': macd_signal,
        'SMA_20': row['sma20'],
        'SMA_50': row['sma50'],
        'SMA_200': row['sma200'],
        'BB_Upper': row['bb_upper'],
        'BB_Middle': row['bb_middle'],
        'BB_Lower': row['bb_lower'],
        'ATR_14': row['atr'],
        'Volume_10d_Avg': row['vol_10d'],
        'Volume_30d_Avg': row['vol_30d'],
        'Volume_Ratio': row['vol_ratio'],
        'Recent_High': row['high_20d'],
        'Recent_Low': row['low_20d'],
        'Current_Price': row['current_price'],
        # IV Rank not in DB yet — will be added in future
        # 'IV_Rank': row.get('iv_rank'),
    }

    return dict(indicators)


def get_earnings_date(db_path: str, ticker: str) -> str | None:
    """Query the next earnings date for the ticker if available."""
    # Future: join with earnings table
    # For now, return None (no earnings exclusion)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate options signals for a ticker.")
    parser.add_argument("--ticker", required=True, help="Stock ticker (e.g., AAPL)")
    parser.add_argument("--date", help="Analysis date (YYYY-MM-DD), default: latest")
    parser.add_argument("--db-path", default="data/market_data.db", help="Path to SQLite DB")
    parser.add_argument(
        "--mode",
        choices=["legacy", "hybrid"],
        default="hybrid",
        help="Pipeline mode: 'legacy' uses OptionsSignalEngine, 'hybrid' uses HybridSignalPipeline (default)"
    )
    args = parser.parse_args()

    try:
        indicators = get_indicators(args.db_path, args.ticker, args.date)
        earnings_date = get_earnings_date(args.db_path, args.ticker)

        if args.mode == "legacy":
            logger.info("Using legacy engine", ticker=args.ticker)
            engine = OptionsSignalEngine()
            signals = engine.generate_signals_for_ticker(args.ticker, indicators)
            output = signals.to_dict()
            output["pipeline_mode"] = "legacy"
        else:
            logger.info("Using hybrid pipeline", ticker=args.ticker)
            pipeline = HybridSignalPipeline()
            output = pipeline.generate_signals(args.ticker, indicators, earnings_date)
            output["pipeline_mode"] = "hybrid"

        print(json.dumps(output, indent=2))

    except Exception as e:
        logger.error("Error generating signals", error=str(e), ticker=args.ticker)
        sys.exit(1)


if __name__ == "__main__":
    main()
