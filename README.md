# MarketSentimentAnalyzer

Analyze market sentiment for stock tickers by combining technical indicators, latest news, and AI-powered sentiment analysis.

## Features

- **Technical Indicators**: Fetches stock data and calculates RSI, MACD, SMA, and EMA across multiple timeframes
- **News Integration**: Retrieves latest news using Brave Search API
- **AI Sentiment**: Uses local Ollama (qwen2.5:7b) to analyze news sentiment
- **Console Tables**: Beautiful formatted output with tabulate
- **Structured Logging**: Uses structlog for clear, structured logs
- **Robust Error Handling**: Graceful degradation when services are unavailable
- **Comprehensive Tests**: Pytest suite with mocks for all external services

## Project Structure

```
MarketSentimentAnalyzer/
├── config/
│   └── tickers.json        # List of tickers to analyze
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration handling
│   ├── fetchers.py         # Stock & news data fetchers
│   ├── sentiment.py        # LLM sentiment analysis
│   ├── display.py          # Console output formatting
│   └── main.py             # Application entry point
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_fetchers.py
│   ├── test_sentiment.py
│   ├── test_display.py
│   └── test_main.py
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

## Running

```bash
# Using Poetry
poetry run python -m src.main

# Using pip
python3 -m src.main
```

## Output

The tool prints:

1. **Indicators Table** for each ticker:
   - RSI (14)
   - MACD, Signal, Histogram
   - SMA (5, 10, 20, 50, 100, 200)
   - EMA (5, 10, 20, 50, 100, 200)
   - Current Price, Change %

2. **News Articles**: Top 5 latest headlines with summaries

3. **Sentiment Analysis**: LLM-determined sentiment (bullish/bearish/neutral) with confidence and explanation

4. **Summary Table**: Combined view across all tickers

## Testing

```bash
# Using Poetry
poetry run pytest

# Using pip
pytest
```

Tests mock all external dependencies:
- `yfinance` (stock data)
- Brave Search API
- Ollama API

## Safety Notes

- **No trading**: This tool is for analysis only, not for executing trades
- **Rate limits**: Respect Brave Search API rate limits (free tier: 2,000 queries/month)
- **Local model**: Ollama runs locally; no data sent to cloud for sentiment analysis
- **API key security**: Never commit `.env` file; use `.env.example` as template

## Requirements

- Python 3.9+
- See `pyproject.toml` / `requirements.txt` for full list

## Architecture

1. **Config**: Loads tickers and environment settings
2. **Fetchers**:
   - `StockDataFetcher`: Uses yfinance + pandas_ta for indicators
   - `NewsFetcher`: Brave Search API for news
3. **SentimentAnalyzer**: Ollama API for LLM analysis
4. **Display**: tabulate for console tables
5. **Main**: Orchestrates workflow, handles errors

## Extension Ideas

- Export to CSV/JSON
- Email reports
- Web dashboard with Flask/FastAPI
- Additional indicators (Bollinger Bands, ATR, etc.)
- Multiple news sources
- Historical sentiment tracking
- Discord/Telegram notifications

## License

MIT

## Author

Venkat - Senior Python Developer
