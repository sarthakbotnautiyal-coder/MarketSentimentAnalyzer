# MarketSentimentAnalyzer


## Features

- **Technical Indicators**: Fetches stock data and calculates RSI, MACD, SMA, and EMA across multiple timeframes
- **Persistent Caching**: SQLite database stores all data to avoid rate limits and speed up repeated runs
- **Incremental Updates**: Only fetches new data; backfill 1-year historical for new tickers
- **Console Tables**: Beautiful formatted output with tabulate
- **Structured Logging**: Uses structlog for clear, structured logs
- **Robust Error Handling**: Graceful degradation when services are unavailable; falls back to cached data
- **Comprehensive Tests**: Pytest suite with mocks for all external services

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
├── .env.example            # Environment template
├── requirements.txt        # Pip dependencies
├── pyproject.toml          # Poetry configuration
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- Poetry (optional, for dependency management)
- [Ollama](https://ollama.ai) running locally with `qwen2.5:7b` model:
  ```bash
  ollama pull qwen2.5:7b
  ```
- Brave Search API key (free tier available at [brave.com/search/api/](https://brave.com/search/api/))

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

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Brave API key:
   ```env
   BRAVE_API_KEY=your_brave_api_key_here
   ```

3. (Optional) Configure Ollama location if not on localhost:
   ```env
   OLLAMA_HOST=http://localhost:11434
   ```

4. (Optional) Change ticker list in `config/tickers.json`.

5. (Optional) Adjust caching TTLs in `config/database.yaml`:
   ```yaml
   path: data/market_data.db
   stock_ttl: '1d'      # Stock data considered fresh for 1 day
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

This downloads a full year of historical data for all tickers and stores it in the database.

### Force Refresh

Bypass cache and fetch fresh data from all APIs (useful for testing or manual updates):

```bash
python3 -m src.main --force-refresh
```

Combine with backfill for a full refresh:

```bash
python3 -m src.main --backfill 1y --force-refresh
```

### How Caching Works

- **Stock Data**: Cached in `stock_daily` table with date+ticker primary key. Daily runs will only fetch fresh data if the latest cached date is older than `stock_ttl` (default 1 day).

On API failures, the tool falls back to the latest cached data (if available) and logs a warning.

## Output

The tool prints:

1. **Indicators Table** for each ticker:
   - RSI (14)
   - MACD, Signal, Histogram
   - SMA (5, 10, 20, 50, 100, 200)
   - EMA (5, 10, 20, 50, 100, 200)
   - Current Price, Change %

2. **News Articles**: Top 5 latest headlines with summaries


4. **Summary Table**: Combined view across all tickers

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
    indicators TEXT,  -- JSON with calculated indicators
    PRIMARY KEY (date, ticker)
);
```

```sql
    date TEXT,
    ticker TEXT,
    title TEXT,
    url TEXT,
    snippet TEXT,
    source TEXT,
    published TEXT,
    PRIMARY KEY (url, ticker)
);
```

```sql
    date TEXT,
    ticker TEXT,
    confidence REAL,
    explanation TEXT,
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

Tests mock all external dependencies:
- `yfinance` (stock data)
- Brave Search API
- Ollama API

Database tests use a temporary SQLite database.

## Safety Notes

- **No trading**: This tool is for analysis only, not for executing trades
- **Rate limits**: Respect Brave Search API rate limits (free tier: 2,000 queries/month)
- **API key security**: Never commit `.env` file; use `.env.example` as template
- **Database**: SQLite file (`data/market_data.db`) is ignored by git; local cache only

## Requirements

- Python 3.9+
- See `pyproject.toml` / `requirements.txt` for full list

## Architecture

1. **Config**: Loads tickers, environment settings, and database configuration
3. **Fetchers**:
   - `StockDataFetcher`: Uses yfinance + pandas_ta; caches in database; supports backfill and incremental updates
   - `NewsFetcher`: Brave Search API with deduplication and TTL caching
4. **SentimentAnalyzer**: Ollama API with TTL caching to avoid spamming LLM
5. **Display**: tabulate for console tables
6. **Main**: Orchestrates workflow, handles CLI args, manages database lifecycle, handles errors

## Extension Ideas

- Export to CSV/JSON
- Email reports
- Web dashboard with Flask/FastAPI
- Additional indicators (Bollinger Bands, ATR, etc.)
- Discord/Telegram/Slack notifications
- Webhook triggers for significant changes

## License

MIT

## Author

Venkat - Senior Python Developer
