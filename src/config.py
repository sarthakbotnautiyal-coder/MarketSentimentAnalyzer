"""Configuration handling for MarketSentimentAnalyzer."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
import structlog

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

logger = structlog.get_logger()


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str = "data/market_data.db"
    stock_ttl: str = "1d"
    news_ttl: str = "7d"
    sentiment_ttl: str = "7d"
    log_dir: str = "logs"

    @property
    def stock_ttl_days(self) -> int:
        """Convert TTL string to days."""
        if self.stock_ttl.endswith('d'):
            return int(self.stock_ttl.rstrip('d'))
        return 1  # Default

    @property
    def news_ttl_days(self) -> int:
        """Convert TTL string to days."""
        if self.news_ttl.endswith('d'):
            return int(self.news_ttl.rstrip('d'))
        return 7  # Default

    @property
    def sentiment_ttl_days(self) -> int:
        """Convert TTL string to days."""
        if self.sentiment_ttl.endswith('d'):
            return int(self.sentiment_ttl.rstrip('d'))
        return 7  # Default


@dataclass
class Config:
    """Application configuration."""
    tickers: List[str]
    brave_api_key: str
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    news_count: int = 5
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    backfill_period: str = "1y"
    force_refresh: bool = False

    @classmethod
    def load(cls, config_path: Path = None, env_path: Path = None, db_config_path: Path = None) -> "Config":
        """Load configuration from files and environment."""
        # Load tickers from config/tickers.json
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tickers.json"

        with open(config_path) as f:
            config_data = json.load(f)
            # Support both {"tickers": [...]} and bare [...]
            if isinstance(config_data, list):
                tickers = config_data
            else:
                tickers = config_data.get("tickers", [])

        # Load database config
        if db_config_path is None:
            db_config_path = Path(__file__).parent.parent / "config" / "database.yaml"

        db_config = DatabaseConfig()
        if db_config_path.exists():
            if YAML_AVAILABLE:
                try:
                    with open(db_config_path) as f:
                        db_data = yaml.safe_load(f) or {}
                        db_config = DatabaseConfig(**db_data)
                except Exception as e:
                    logger.warning(f"Failed to parse database config: {e}, using defaults")
            else:
                logger.warning("PyYAML not installed, using default database config")
        else:
            logger.warning(f"Database config not found at {db_config_path}, using defaults")

        # Load environment variables
        if env_path is None:
            env_path = Path(__file__).parent.parent / ".env"

        if env_path.exists():
            from dotenv import load_dotenv
            load_dotenv(env_path)
        else:
            logger.warning(f".env file not found at {env_path}")

        brave_api_key = os.getenv("BRAVE_API_KEY", "")
        if not brave_api_key:
            raise ValueError("BRAVE_API_KEY environment variable is required")

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        news_count = int(os.getenv("NEWS_COUNT", "5"))
        backfill_period = os.getenv("BACKFILL_PERIOD", "1y")
        force_refresh = os.getenv("FORCE_REFRESH", "false").lower() == "true"

        return cls(
            tickers=tickers,
            brave_api_key=brave_api_key,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            news_count=news_count,
            database=db_config,
            backfill_period=backfill_period,
            force_refresh=force_refresh
        )
