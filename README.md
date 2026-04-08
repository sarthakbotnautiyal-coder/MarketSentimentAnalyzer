# MarketSentimentAnalyzer

A Python application that fetches market data and calculates technical indicators for stock tickers. Uses SQLite for persistent caching and performs delta fetching to minimize API calls.

## Features

- **Technical Indicators**: Fetches stock data and calculates RSI, MACD, SMA, EMA across multiple timeframes
- **Persistent Caching**: SQLite database stores all data to avoid rate limits and speed up repeated runs
- **Incremental/Delta Updates**: Only fetches new data since last stored date; backfill 1-year historical for new tickers
- **Console Tables**: Beautiful formatted output with tabulate
- **Structured Logging**: Uses structlog for clear, structured logs
- **Robust Error Handling**: Graceful degradation when services are unavailable; falls back to cached data
- **Comprehensive Tests**: Pytest suite with mocks for external services

## Project Structure

```
MarketSentimentAnalyzer/
├── config/
│   ├── tickers.json        # List of tickers to analyze
│   └── database.yaml       # Database configuration (caching TTLs)
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration handling
│   ├── database.py         # SQLite database manager
│   ├── display.py          # Console output formatting
│   ├── fetchers.py         # Stock data fetching with caching
│   └── main.py             # Application entry point
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_database.py   # Database operations tests
│   ├── test_fetchers.py   # Fetcher caching tests
│   ├── test_display.py
│   └── test_main.py       # Integration tests
├── data/                   # (Created on first run)
│   └── market_data.db     # SQLite database with cached data
├── requirements.txt        # Pip dependencies
├── pyproject.toml          # Poetry configuration
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- Poetry (optional, for dependency management)

### Installation

**Using Poetry (recommended):**
```bash
poetry install
```

**Using pip:**
```bash
pip install -r requirements.txt
```

### Configuration

1. Tickers to analyze are configured in `config/tickers.json`.

2. Database settings are in `config/database.yaml`:
   ```yaml
   path: data/market_data.db
   stock_ttl: 30              # Stock data retention (days)
   ```

## Running

### Basic Usage

```bash
# Using Poetry
poetry run python -m src.main

# Using pip
python3 -m src.main
```

### Initial Backfill (First Run)

On first run, backfill 1 year of historical stock data:

```bash
python3 -m src.main --backfill 1y
```

This downloads a full year of historical data for all tickers and stores it in the database. Indicators are calculated and only the latest day's indicators are stored (one row per ticker).

### Force Refresh

Bypass cache and fetch fresh data from all sources:

```bash
python3 -m src.main --force-refresh
```

Combine with backfill for a full refresh:

```bash
python3 -m src.main --backfill 1y --force-refresh
```

### How Caching Works

- **Stock Data**: Cached in `stock_daily` table with date+ticker primary key. Daily runs use delta fetching - only new data since the last stored date is fetched.
- **Indicators**: The `indicators` table is truncated on each run, and only the latest day's indicators are stored (one row per ticker).

On API failures, the tool falls back to the latest cached data (if available) and logs a warning.

## Output

The tool prints:

1. **Indicators Table** for each ticker:
   - RSI (14)
   - MACD, Signal, Histogram
   - SMA (20, 50, 200)
   - EMA (5)
   - Bollinger Bands (Upper, Middle, Lower)
   - ATR (14)
   - Volume averages (10d, 30d) and ratio
   - 20-day High/Low
   - Current Price, Change %

2. **Summary Table**: Combined view across all tickers

## Database Schema

### stock_daily
```sql
CREATE TABLE stock_daily (
    date TEXT,
    ticker TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (date, ticker)
);
```

### indicators
```sql
CREATE TABLE indicators (
    date TEXT,
    ticker TEXT,
    rsi REAL,
    macd REAL,
    macd_hist REAL,
    sma20 REAL,
    sma50 REAL,
    sma200 REAL,
    bb_upper REAL,
    bb_middle REAL,
    bb_lower REAL,
    atr REAL,
    vol_10d REAL,
    vol_30d REAL,
    vol_ratio REAL,
    high_20d REAL,
    low_20d REAL,
    PRIMARY KEY (date, ticker)
);
```

Indexes are automatically created on ticker and date columns for performance.

## Testing

```bash
# Using Poetry
poetry run pytest -v

# Using pip
pytest -v
```

Tests mock external dependencies:
- `yfinance` (stock data)

Database tests use a temporary SQLite database.

## Safety Notes

- **No trading**: This tool is for analysis only, not for executing trades
- **Database**: SQLite file (`data/market_data.db`) is ignored by git; local cache only

## Requirements

- Python 3.9+
- See `pyproject.toml` / `requirements.txt` for full list

## Architecture

1. **Config**: Loads tickers and database configuration
2. **Database**: SQLite with two tables:
   - `stock_daily`: Historical OHLCV data per ticker per date
   - `indicators`: Latest technical indicators per ticker (truncated each run)
3. **Fetchers**:
   - `StockDataFetcher`: Uses yfinance + pandas_ta; caches in database; supports backfill and incremental/delta updates
4. **Display**: tabulate for console tables
5. **Main**: Orchestrates workflow, handles CLI args, manages database lifecycle, handles errors

## Extension Ideas

- Export to CSV/JSON
- Email reports
- Web dashboard with Flask/FastAPI
- Additional indicators (more timeframes, custom calculations)
- Discord/Telegram/Slack notifications
- Webhook triggers for significant changes

## License

MIT

## Author

Venkat - Senior Python Developer
