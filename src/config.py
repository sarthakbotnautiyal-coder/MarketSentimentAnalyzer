"""Configuration handling for MarketSentimentAnalyzer."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Application configuration."""
    tickers: List[str]
    brave_api_key: str
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    news_count: int = 5

    @classmethod
    def load(cls, config_path: Path = None, env_path: Path = None) -> "Config":
        """Load configuration from files and environment."""
        # Load tickers from config/tickers.json
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tickers.json"

        with open(config_path) as f:
            config_data = json.load(f)
            tickers = config_data.get("tickers", [])

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

        return cls(
            tickers=tickers,
            brave_api_key=brave_api_key,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            news_count=news_count
        )
