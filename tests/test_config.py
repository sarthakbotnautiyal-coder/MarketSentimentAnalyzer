"""Tests for configuration module."""

import json
import tempfile
from pathlib import Path
from src.config import Config, DatabaseConfig


def test_config_default_values():
    """Test default configuration values."""
    db_config = DatabaseConfig()
    assert db_config.path == "data/market_data.db"
    assert db_config.stock_ttl == "1d"


def test_config_load_success():
    """Test successful config loading from directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        # Create tickers.json
        tickers_file = config_dir / "tickers.json"
        tickers_file.write_text(json.dumps(["AAPL", "MSFT"]))

        # Create database.yaml
        db_file = config_dir / "database.yaml"
        db_file.write_text("path: data/test.db\nstock_ttl: '2d'\nlog_dir: test_logs\n")

        config = Config.load(config_dir)

        assert config.tickers == ["AAPL", "MSFT"]
        assert config.database.path == "data/test.db"
        assert config.database.stock_ttl == "2d"
        assert config.database.log_dir == "test_logs"


def test_config_load_tickers_dict_format():
    """Test loading tickers from dict format (legacy)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        # Create tickers.json with dict format
        tickers_file = config_dir / "tickers.json"
        tickers_file.write_text(json.dumps({"tickers": ["GOOGL", "AMZN"]}))

        config = Config.load(config_dir)
        assert config.tickers == ["GOOGL", "AMZN"]


def test_config_load_missing_db_config():
    """Test loading config without database.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "config"
        config_dir.mkdir()

        tickers_file = config_dir / "tickers.json"
        tickers_file.write_text(json.dumps(["TSLA"]))

        config = Config.load(config_dir)

        assert config.tickers == ["TSLA"]
        assert config.database.path == "data/market_data.db"  # Default
