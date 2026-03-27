"""Tests for configuration module."""

import json
import os
import tempfile
from pathlib import Path
from src.config import Config


def test_config_load_success(tmp_path):
    """Test successful config loading."""
    # Create tickers.json
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    tickers_file = config_dir / "tickers.json"
    tickers_file.write_text(json.dumps({"tickers": ["AAPL", "MSFT"]}))

    # Create .env
    env_file = tmp_path / ".env"
    env_file.write_text("BRAVE_API_KEY=test_key\nOLLAMA_HOST=http://test:11434\n")

    # Patch paths
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to temp directory
        dest_config = Path(tmpdir) / "config" / "tickers.json"
        dest_config.parent.mkdir()
        dest_config.write_text(tickers_file.read_text())
        dest_env = Path(tmpdir) / ".env"
        dest_env.write_text(env_file.read_text())

        # Override load method paths
        original_load = Config.load
        def patched_load(config_path=None, env_path=None):
            if config_path is None:
                config_path = dest_config
            if env_path is None:
                env_path = dest_env
            return original_load(config_path, env_path)

        Config.load = staticmethod(patched_load)

        try:
            config = Config.load()
            assert config.tickers == ["AAPL", "MSFT"]
            assert config.brave_api_key == "test_key"
            assert config.ollama_host == "http://test:11434"
            assert config.news_count == 5
        finally:
            Config.load = original_load


def test_config_missing_brave_api_key(tmp_path):
    """Test error when BRAVE_API_KEY is missing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    tickers_file = config_dir / "tickers.json"
    tickers_file.write_text(json.dumps({"tickers": ["AAPL"]}))

    env_file = tmp_path / ".env"
    env_file.write_text("")  # No API key

    with tempfile.TemporaryDirectory() as tmpdir:
        dest_config = Path(tmpdir) / "config" / "tickers.json"
        dest_config.parent.mkdir()
        dest_config.write_text(tickers_file.read_text())
        dest_env = Path(tmpdir) / ".env"
        dest_env.write_text("")

        original_load = Config.load
        def patched_load(config_path=None, env_path=None):
            if config_path is None:
                config_path = dest_config
            if env_path is None:
                env_path = dest_env
            return original_load(config_path, env_path)

        Config.load = staticmethod(patched_load)

        try:
            Config.load()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "BRAVE_API_KEY" in str(e)
        finally:
            Config.load = original_load


def test_config_default_values():
    """Test default configuration values."""
    # Test that Config dataclass has sensible defaults
    from src.config import Config
    assert Config.ollama_model == "qwen2.5:7b"
    assert Config.news_count == 5
