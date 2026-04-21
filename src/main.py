#!/usr/bin/env python3
"""
Fetches stock data, calculates technical indicators,
runs Stage 1 hard filters, then uses LLM to produce signal decisions.

Usage:
    python -m src.main [--backfill 1y] [--force-refresh]
"""

import argparse
import json
import os
import sys
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import structlog

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
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
from src.llm_signal_advisor import LLMSignalAdvisor

logger = structlog.get_logger()

# ── Telegram config ────────────────────────────────────────────────────────────

_TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_TELEGRAM_SIGNALS_CHAT_ID = os.environ.get("TELEGRAM_SIGNALS_CHAT_ID", "-5257920178")


def _send_telegram_alert(text: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if not _TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — skipping Telegram alert")
        return False
    url = f"https://api.telegram.org/bot{_TELEGRAM_BOT_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": _TELEGRAM_SIGNALS_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            if result.get("ok"):
                logger.info("Telegram alert sent", chat_id=_TELEGRAM_SIGNALS_CHAT_ID)
                return True
            else:
                logger.error("Telegram API error", error=result)
                return False
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        logger.error(
            "Telegram HTTP 400 error",
            status=e.code,
            response=body[:500],
            text_preview=text[:200],
        )
        return False
    except Exception as e:
        logger.error("Failed to send Telegram alert", error=str(e))
        return False


def _build_telegram_message(decisions: list, date_str: str) -> str:
    """Build a Telegram message from HIGH confidence decisions."""
    if not decisions:
        return None

    conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    sig_emoji = {"SELL_PUTS": "📈", "SELL_CALLS": "📉", "NO_TRADE": "➖"}

    header = f"📊 <b>MSA Signals — {date_str}</b>\n<i>{len(decisions)} HIGH confidence signal(s)</i>\n"
    parts = [header]

    for item in decisions:
        ticker = item["ticker"]
        price = item["price"]
        advice = item["advice"]

        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
        sig = advice.signal_decision
        conf = advice.confidence

        lines = [
            f"<b>{ticker}</b> | {price_str}",
            f"{sig_emoji.get(sig, '📊')} {sig}  {conf_emoji.get(conf, '')} {conf}",
        ]

        if advice.top_3_reasons:
            for reason in advice.top_3_reasons:
                lines.append(f"  • {reason}")

        if advice.strike_recommendation:
            lines.append(f"  🎯 Strike: {advice.strike_recommendation}")
        if advice.expiry_recommendation:
            lines.append(f"  📅 Expiry: {advice.expiry_recommendation}")
        if advice.stop_loss:
            lines.append(f"  🛑 Stop: {advice.stop_loss}")
        if advice.premium_estimate:
            lines.append(f"  💰 Premium: {advice.premium_estimate}")

        parts.append("\n".join(lines))
        parts.append("\n" + "─" * 30)

    message = "\n".join(parts)
    if len(message) > 4000:
        message = message[:4000] + "\n\n <i>(truncated)</i>"
    return message


def _maybe_send_telegram(decisions: list, date_str: str) -> None:
    """Send Telegram alert if token is configured. Silent failure."""
    if not _TELEGRAM_BOT_TOKEN:
        logger.info("TELEGRAM_BOT_TOKEN not set — skipping alert")
        return

    # Only send HIGH confidence decisions
    high_conf = [d for d in decisions if d["advice"].confidence == "HIGH"]
    if not high_conf:
        pass  # silent skip
        return

    message = _build_telegram_message(high_conf, date_str)
    if message:
        _send_telegram_alert(message)


# ── Ticker processing ────────────────────────────────────────────────────────


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


# ── Run modes ────────────────────────────────────────────────────────────────


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


# ── Filter + LLM pipeline ────────────────────────────────────────────────────


def filter_and_advise(results: dict, stage1: Stage1HardFilters,
                      advisor: LLMSignalAdvisor) -> tuple[list, list]:
    """
    Run Stage 1 hard filters on all tickers, then call LLM for each candidate.

    Returns:
        (favorable_tickers, decisions)
    """
    favorable_tickers = []
    decisions = []

    for ticker, result in results.items():
        indicators = result.get("indicators", {})
        if "error" in indicators:
            continue

        earnings_date = indicators.get("Next_Earnings_Date")
        candidates, _ = stage1.evaluate_all_with_warning(indicators, earnings_date)

        if not candidates:
            logger.debug("Ticker failed Stage 1", ticker=ticker)
            continue

        favorable_tickers.append(ticker)
        logger.info(f"Stage 1 passed for {ticker} — requesting LLM advice")

        advice = advisor.advise(
            ticker=ticker,
            indicators=indicators,
            stage1_passed=True,
        )

        decisions.append({
            "ticker": ticker,
            "price": indicators.get("Current_Price", 0),
            "indicators": indicators,
            "advice": advice,
        })

    return favorable_tickers, decisions


def _print_favorable_tickers(tickers: list, date_str: str) -> None:
    """Print the list of tickers that passed Stage 1 filters."""
    print(f"\n{'='*60}")
    print(f"Favorable Tickers — {date_str}")
    print(f"{'='*60}")
    if tickers:
        for ticker in tickers:
            print(f"  ✓ {ticker}")
    else:
        print("  No tickers passed Stage 1 filters.")
    print(f"{'='*60}\n")


def _print_llm_results(decisions: list, date_str: str) -> None:
    """Print LLM signal decisions to console — HIGH confidence only."""
    conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    sig_emoji = {"SELL_PUTS": "📈", "SELL_CALLS": "📉", "NO_TRADE": "➖"}

    high_conf = [d for d in decisions if d["advice"].confidence == "HIGH"]

    print(f"\n{'='*60}")
    print(f"LLM Signal Advisor — {date_str}  [HIGH confidence only]")
    print(f"{'='*60}")

    if not high_conf:
        print("  No HIGH confidence signals.")
        print(f"{'='*60}\n")
        return

    for item in high_conf:
        ticker = item["ticker"]
        price = item["price"]
        advice = item["advice"]

        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
        sig = advice.signal_decision
        conf = advice.confidence

        print(f"\n{sig_emoji.get(sig, '📊')} *{ticker}*  |  {price_str}")
        print(f"   Signal: {sig}  {conf_emoji.get(conf, '')} {conf}")

        if advice.top_3_reasons:
            print("   Reasons:")
            for reason in advice.top_3_reasons:
                print(f"     • {reason}")

        if advice.strike_recommendation:
            print(f"   Strike: {advice.strike_recommendation}")
        if advice.expiry_recommendation:
            print(f"   Expiry: {advice.expiry_recommendation}")
        if advice.stop_loss:
            print(f"   Stop:   {advice.stop_loss}")
        if advice.premium_estimate:
            print(f"   Premium: {advice.premium_estimate}")
        if advice.risk_flags:
            print(f"   Risk:   {' | '.join(advice.risk_flags)}")

        print()

    print(f"{'='*60}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────


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
        advisor = LLMSignalAdvisor()

        favorable, decisions = filter_and_advise(results, stage1, advisor)

        # Always print favorable tickers list
        _print_favorable_tickers(favorable, date_str)

        # LLM results — HIGH confidence only (console + Telegram)
        _print_llm_results(decisions, date_str)

        # Telegram — HIGH confidence signals only
        _maybe_send_telegram(decisions, date_str)

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