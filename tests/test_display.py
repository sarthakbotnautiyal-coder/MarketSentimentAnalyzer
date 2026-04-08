"""Tests for display module (indicators-only output)."""

from src.display import Display
import pytest


def test_print_indicators(capsys):
    """Test printing indicators for a ticker."""
    indicators = {
        'Current_Price': 150.00,
        'RSI_14': 65.5,
        'MACD': 0.5
    }

    Display.print_indicators("AAPL", indicators)

    captured = capsys.readouterr()
    assert "AAPL" in captured.out
    # Price formatted as int (150) or float (150.00) depending on rounding
    assert ("150" in captured.out or "150.00" in captured.out)
    assert "65.5" in captured.out
    assert "0.5" in captured.out


def test_print_indicators_with_error(capsys):
    """Test printing indicators with error."""
    indicators = {"error": "Failed to fetch"}

    Display.print_indicators("AAPL", indicators)

    captured = capsys.readouterr()
    assert "Failed to fetch" in captured.out


def test_print_indicators_empty(capsys):
    """Test printing empty indicators."""
    indicators = {}

    Display.print_indicators("AAPL", indicators)

    captured = capsys.readouterr()
    assert "AAPL" in captured.out


def test_print_summary(capsys):
    """Test printing summary table."""
    results = {
        "AAPL": {
            "indicators": {
                "Current_Price": 150.00,
                "Change": 2.5,
                "RSI_14": 65.5
            }
        },
        "MSFT": {
            "indicators": {
                "Current_Price": 300.00,
                "Change": -1.2,
                "RSI_14": 45.0
            }
        }
    }

    Display.print_summary(results)

    captured = capsys.readouterr()
    assert "SUMMARY" in captured.out
    assert "AAPL" in captured.out
    assert "MSFT" in captured.out
    assert "$150.00" in captured.out
