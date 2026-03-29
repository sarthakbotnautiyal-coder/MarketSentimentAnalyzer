"""Tests for generate_signals.py script."""

import pytest
import sqlite3
import json
import tempfile
import shutil
import os
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from options_engine import OptionsSignalEngine
from scripts.generate_signals import get_indicators  # Note: need to extract if not module


@pytest.fixture
def sample_db(tmp_path: Path):
    """Create temp DB with sample AAPL data."""
    db_path = tmp_path / "test.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE stock_daily (
            date TEXT, ticker TEXT, close REAL, PRIMARY KEY(date, ticker)
        )
    ''')
    cursor.execute('''
        CREATE TABLE indicators (
            date TEXT, ticker TEXT, rsi REAL, macd REAL, macd_hist REAL,
            sma20 REAL, sma50 REAL, sma200 REAL, bb_upper REAL, bb_middle REAL,
            bb_lower REAL, atr REAL, vol_10d REAL, vol_30d REAL, vol_ratio REAL,
            high_20d REAL, low_20d REAL, PRIMARY KEY(date, ticker)
        )
    ''')
    
    # Insert sample data matching earlier AAPL
    date = '2026-03-27'
    cursor.execute('''
        INSERT INTO stock_daily VALUES (?, 'AAPL', ?)
    ''', (date, 248.80))
    
    cursor.execute('''
        INSERT INTO indicators VALUES (?, 'AAPL', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        date,
        37.97, -3.6413, -0.1943, 255.38, 260.21, 247.70,
        266.27, 255.38, 244.48, 5.61, 42720320, 41852050, 1.02,
        266.53, 246.0
    ))
    
    conn.commit()
    conn.close()
    return db_path


def test_get_indicators(sample_db):
    """Test indicator fetching."""
    indicators = get_indicators(str(sample_db), 'AAPL', '2026-03-27')
    
    assert indicators['RSI_14'] == 37.97
    assert indicators['Current_Price'] == 248.80
    assert indicators['MACD_Signal'] is not None  # Computed


def test_engine_integration(sample_db):
    """Test full pipeline."""
    indicators = get_indicators(str(sample_db), 'AAPL', '2026-03-27')
    engine = OptionsSignalEngine()
    signals = engine.generate_signals_for_ticker('AAPL', indicators)
    
    assert signals.ticker == 'AAPL'
    assert len(signals.signals) > 0
    assert 'SELL_CALLS' in [s.signal_type.value for s in signals.signals]  # Expected from data


@pytest.mark.parametrize('ticker,expected', [
    ('AAPL', True),
    ('INVALID', False),
])
def test_cli_simulation(sample_db, ticker, expected):
    """Simulate CLI call."""
    from scripts.generate_signals import main, get_indicators as gi
    
    with patch('sys.argv', ['script.py', '--ticker', ticker, '--db-path', str(sample_db)]):
        if expected:
            try:
                main()
            except SystemExit:
                pytest.fail("Should not exit")
        else:
            import pytest
            with pytest.raises(ValueError):
                gi(str(sample_db), ticker)


# Note: Full CLI subprocess test requires venv; skipped for unit focus
