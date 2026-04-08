"""Configuration handling for MarketSentimentAnalyzer."""

import json
import os
from pathlib import Path
from typing import List, Optional
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
    log_dir: str = "logs"


@dataclass
class Config:
    """Application configuration."""
    tickers: List[str] = field(default_factory=list)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    @classmethod
    def load(cls, config_dir: Path = None) -> "Config":
        """Load configuration from JSON config files.

        Args:
            config_dir: Directory containing config files (defaults to project config/)

        Returns:
            Config instance with loaded settings
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"

        # Load tickers
        tickers_path = config_dir / "tickers.json"
        tickers = []
        if tickers_path.exists():
            with open(tickers_path) as f:
                data = json.load(f)
                # Support both list and dict format
                tickers = data if isinstance(data, list) else data.get("tickers", [])
            logger.info(f"Loaded {len(tickers)} tickers from {tickers_path}")
        else:
            logger.warning(f"Tickers file not found: {tickers_path}")

        # Load database config
        db_config = DatabaseConfig()
        db_config_path = config_dir / "database.yaml"
        if db_config_path.exists() and YAML_AVAILABLE:
            try:
                with open(db_config_path) as f:
                    db_data = yaml.safe_load(f) or {}
                db_config.path = db_data.get('path', db_config.path)
                db_config.stock_ttl = db_data.get('stock_ttl', db_config.stock_ttl)
                db_config.log_dir = db_data.get('log_dir', db_config.log_dir)
                logger.info(f"Loaded database config from {db_config_path}")
            except Exception as e:
                logger.warning(f"Failed to parse database.yaml: {e}")
        elif not YAML_AVAILABLE:
            logger.warning("PyYAML not installed, using default database config")

        config = cls(
            tickers=tickers,
            database=db_config
        )

        logger.info("Configuration loaded", tickers_count=len(tickers))
        return config
