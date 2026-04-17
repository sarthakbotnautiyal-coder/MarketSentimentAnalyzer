"""Tests for Stage1HardFilters, Stage2LLM, and HybridSignalPipeline."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from options_engine import (
    Stage1HardFilters,
    Stage2LLM,
    HybridSignalPipeline,
    OptionsSignalEngine,
    CandidateSignal,
    NoCandidate,
    SignalType,
    ConfidenceLevel,
    FilterResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_file(tmp_path: Path):
    """Create a temp threshold config."""
    config = {
        "sell_puts": {
            "rsi_max": 40,
            "rsi_ideal_min": 25,
            "rsi_ideal_max": 40,
            "iv_rank_min": 60,
            "price_above_sma20": True,
            "bb_lower_proximity": True,
            "vol_ratio_min": 1.0,
            "earnings_exclude_days": 14,
        },
        "sell_calls": {
            "rsi_min": 65,
            "iv_rank_min": 60,
            "price_below_sma20": True,
            "bb_upper_proximity": True,
            "vol_ratio_min": 1.0,
            "earnings_exclude_days": 14,
        },
        "buy_leaps": {
            "price_above_sma200": True,
            "rsi_min": 50,
            "macd_positive": True,
            "vol_ratio_min": 1.0,
            "iv_rank_max": 40,
            "earnings_exclude_days": 14,
        },
    }
    path = tmp_path / "signal_thresholds.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f)
    return str(path)


@pytest.fixture
def filters(config_file):
    """Stage1HardFilters instance."""
    return Stage1HardFilters(config_path=config_file)


@pytest.fixture
def sell_puts_indicators():
    """Indicators that should pass SELL_PUTS filters (all 5 conditions pass)."""
    return {
        "RSI_14": 35.0,
        "IV_Rank": 65.0,
        "Current_Price": 150.0,
        "SMA_20": 145.0,
        "BB_Lower": 148.0,  # Near price means near lower band
        "Volume_Ratio": 1.3,
    }


@pytest.fixture
def sell_calls_indicators():
    """Indicators that should pass SELL_CALLS filters (all 5 conditions pass)."""
    return {
        "RSI_14": 72.0,
        "IV_Rank": 70.0,
        "Current_Price": 140.0,
        "SMA_20": 145.0,
        "BB_Upper": 143.0,  # Near price means near upper band
        "Volume_Ratio": 1.2,
    }


@pytest.fixture
def buy_leaps_indicators():
    """Indicators that should pass BUY_LEAPS filters (all 5 conditions pass)."""
    return {
        "RSI_14": 58.0,
        "IV_Rank": 30.0,
        "Current_Price": 200.0,
        "SMA_200": 180.0,
        "MACD": 2.5,
        "Volume_Ratio": 1.1,
    }


@pytest.fixture
def neutral_indicators():
    """Indicators that should fail all filters."""
    return {
        "RSI_14": 50.0,
        "IV_Rank": 50.0,
        "Current_Price": 100.0,
        "SMA_20": 100.0,
        "SMA_200": 110.0,
        "BB_Upper": 105.0,
        "BB_Lower": 95.0,
        "Volume_Ratio": 0.9,
        "MACD": -0.5,
    }


# ---------------------------------------------------------------------------
# Stage1HardFilters: SELL_PUTS
# ---------------------------------------------------------------------------

class TestSellPutsFilter:
    def test_sell_puts_passes_all_conditions(self, filters, sell_puts_indicators):
        """All 5 conditions met → CandidateSignal returned."""
        result = filters.evaluate_sell_puts(sell_puts_indicators, None)
        assert isinstance(result, CandidateSignal)
        assert result.signal_type == SignalType.SELL_PUTS
        assert result.confidence in (ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)

    def test_sell_puts_high_rsi_fails(self, filters, sell_puts_indicators):
        """RSI too high + BB not near lower band + low volume → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 55.0,  # FAIL
            "IV_Rank": 65.0,  # pass
            "Current_Price": 150.0,
            "SMA_20": 145.0,  # pass
            "BB_Lower": 100.0,  # FAIL (price not near lower band)
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_sell_puts(indicators, None)
        assert isinstance(result, NoCandidate)
        passed = [r for r in result.filter_results if r.passed]
        assert len(passed) < 4  # Not enough passes

    def test_sell_puts_low_iv_rank_fails(self, filters, sell_puts_indicators):
        """IV Rank too low + BB not near lower band + low volume → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 45.0,  # FAIL (not oversold)
            "IV_Rank": 45.0,  # FAIL
            "Current_Price": 150.0,
            "SMA_20": 145.0,  # pass
            "BB_Lower": 100.0,  # FAIL
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_sell_puts(indicators, None)
        assert isinstance(result, NoCandidate)

    def test_sell_puts_price_below_sma20_fails(self, filters, sell_puts_indicators):
        """Price below SMA20 + BB not near lower band + low volume → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 45.0,  # FAIL (not oversold)
            "IV_Rank": 65.0,  # pass
            "Current_Price": 140.0,  # < SMA20 = FAIL
            "SMA_20": 145.0,
            "BB_Lower": 100.0,  # FAIL
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_sell_puts(indicators, None)
        assert isinstance(result, NoCandidate)

    def test_sell_puts_low_volume_fails(self, filters, sell_puts_indicators):
        """Volume ratio too low + BB not near lower band + RSI marginal → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 45.0,  # FAIL (not oversold)
            "IV_Rank": 65.0,  # pass
            "Current_Price": 150.0,
            "SMA_20": 145.0,  # pass
            "BB_Lower": 100.0,  # FAIL
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_sell_puts(indicators, None)
        assert isinstance(result, NoCandidate)

    def test_sell_puts_missing_indicator_skips_gracefully(self, filters, sell_puts_indicators):
        """Missing indicator → skipped with warning, still returns result."""
        indicators = {k: v for k, v in sell_puts_indicators.items() if k != "RSI_14"}
        result = filters.evaluate_sell_puts(indicators, None)
        assert isinstance(result, (CandidateSignal, NoCandidate))

    def test_sell_puts_earnings_too_close_fails(self, filters, sell_puts_indicators):
        """Earnings within 14 days → NoCandidate."""
        tomorrow = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
        result = filters.evaluate_sell_puts(sell_puts_indicators, earnings_date=tomorrow)
        assert isinstance(result, NoCandidate)
        earnings_filter = next(
            (r for r in result.filter_results if r.condition == "earnings"), None
        )
        assert earnings_filter is not None
        assert earnings_filter.passed is False

    def test_sell_puts_earnings_safe(self, filters, sell_puts_indicators):
        """Earnings > 14 days away → passes earnings check."""
        far_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        result = filters.evaluate_sell_puts(sell_puts_indicators, earnings_date=far_date)
        assert isinstance(result, CandidateSignal)


# ---------------------------------------------------------------------------
# Stage1HardFilters: SELL_CALLS
# ---------------------------------------------------------------------------

class TestSellCallsFilter:
    def test_sell_calls_passes_all_conditions(self, filters, sell_calls_indicators):
        """All conditions met → CandidateSignal."""
        result = filters.evaluate_sell_calls(sell_calls_indicators, None)
        assert isinstance(result, CandidateSignal)
        assert result.signal_type == SignalType.SELL_CALLS

    def test_sell_calls_low_rsi_fails(self, filters, sell_calls_indicators):
        """RSI too low + BB not near upper band + low volume → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 50.0,  # FAIL (need > 65)
            "IV_Rank": 70.0,  # pass
            "Current_Price": 140.0,
            "SMA_20": 145.0,  # pass (price < sma20)
            "BB_Upper": 200.0,  # FAIL (price not near upper band)
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_sell_calls(indicators, None)
        assert isinstance(result, NoCandidate)

    def test_sell_calls_price_above_sma20_fails(self, filters, sell_calls_indicators):
        """Price above SMA20 + BB not near upper band + low volume → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 72.0,  # pass
            "IV_Rank": 70.0,  # pass
            "Current_Price": 150.0,  # > SMA20 = FAIL
            "SMA_20": 145.0,
            "BB_Upper": 200.0,  # FAIL
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_sell_calls(indicators, None)
        assert isinstance(result, NoCandidate)


# ---------------------------------------------------------------------------
# Stage1HardFilters: BUY_LEAPS
# ---------------------------------------------------------------------------

class TestBuyLeapsFilter:
    def test_buy_leaps_passes_all_conditions(self, filters, buy_leaps_indicators):
        """All conditions met → CandidateSignal."""
        result = filters.evaluate_buy_leaps(buy_leaps_indicators, None)
        assert isinstance(result, CandidateSignal)
        assert result.signal_type == SignalType.BUY_LEAPS

    def test_buy_leaps_price_below_sma200_fails(self, filters, buy_leaps_indicators):
        """Price below SMA200 + high IV + low volume → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 58.0,  # pass
            "IV_Rank": 50.0,  # FAIL (> 40 means high IV)
            "Current_Price": 160.0,  # < SMA200 = FAIL
            "SMA_200": 180.0,
            "MACD": 2.5,  # pass
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_buy_leaps(indicators, None)
        assert isinstance(result, NoCandidate)

    def test_buy_leaps_negative_macd_fails(self, filters, buy_leaps_indicators):
        """MACD negative + high IV + low volume → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 58.0,  # pass
            "IV_Rank": 50.0,  # FAIL
            "Current_Price": 200.0,
            "SMA_200": 180.0,  # pass
            "MACD": -1.5,  # FAIL
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_buy_leaps(indicators, None)
        assert isinstance(result, NoCandidate)

    def test_buy_leaps_high_iv_rank_fails(self, filters, buy_leaps_indicators):
        """IV Rank too high + negative MACD + low volume → only 2 conditions pass → NoCandidate."""
        indicators = {
            "RSI_14": 58.0,  # pass
            "IV_Rank": 55.0,  # FAIL (must be < 40)
            "Current_Price": 200.0,
            "SMA_200": 180.0,  # pass
            "MACD": -1.0,  # FAIL
            "Volume_Ratio": 0.5,  # FAIL
        }
        result = filters.evaluate_buy_leaps(indicators, None)
        assert isinstance(result, NoCandidate)


# ---------------------------------------------------------------------------
# Stage1HardFilters: evaluate() returns first candidate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_returns_first_candidate(self, filters, sell_puts_indicators):
        """evaluate() tries SELL_PUTS first, returns it if passes."""
        result = filters.evaluate(sell_puts_indicators)
        assert isinstance(result, CandidateSignal)
        assert result.signal_type == SignalType.SELL_PUTS

    def test_evaluate_falls_through_to_sell_calls(self, filters, sell_calls_indicators):
        """SELL_PUTS fails, SELL_CALLS passes → returns SELL_CALLS."""
        result = filters.evaluate(sell_calls_indicators)
        assert isinstance(result, CandidateSignal)
        assert result.signal_type == SignalType.SELL_CALLS

    def test_evaluate_falls_through_to_buy_leaps(self, filters, buy_leaps_indicators):
        """SELL_PUTS and SELL_CALLS fail, BUY_LEAPS passes → returns BUY_LEAPS."""
        result = filters.evaluate(buy_leaps_indicators)
        assert isinstance(result, CandidateSignal)
        assert result.signal_type == SignalType.BUY_LEAPS

    def test_evaluate_all_fail(self, filters, neutral_indicators):
        """No filters pass → NoCandidate."""
        result = filters.evaluate(neutral_indicators)
        assert isinstance(result, NoCandidate)


# ---------------------------------------------------------------------------
# Stage2LLM Stub
# ---------------------------------------------------------------------------

class TestStage2LLM:
    def test_process_candidate_returns_no_candidate(self, sell_puts_indicators):
        """Stub always returns NO_CANDIDATE."""
        llm = Stage2LLM()
        candidate = CandidateSignal(
            signal_type=SignalType.SELL_PUTS,
            confidence=ConfidenceLevel.HIGH,
            reasons=["RSI oversold", "High IV Rank"],
            current_price=150.0,
        )
        result = llm.process_candidate("AAPL", sell_puts_indicators, candidate)
        assert result["signal_decision"] == "NO_CANDIDATE"
        assert "Stage 2 not yet implemented" in result["reason"]


# ---------------------------------------------------------------------------
# HybridSignalPipeline
# ---------------------------------------------------------------------------

class TestHybridSignalPipeline:
    def test_pipeline_sell_puts_candidate(self, config_file, sell_puts_indicators):
        """Pipeline with SELL_PUTS candidate → Stage 1 pass, Stage 2 reject."""
        pipeline = HybridSignalPipeline(config_path=config_file)
        output = pipeline.generate_signals("AAPL", sell_puts_indicators)

        assert output["ticker"] == "AAPL"
        assert output["pipeline"] == "hybrid_v1"
        assert output["stage1"]["passed"] is True
        assert output["stage1"]["signal_type"] == "SELL_PUTS"
        assert output["stage2"]["skipped"] is False
        assert output["stage2"]["decision"] == "NO_CANDIDATE"
        assert output["final_signal"] is None  # Stage 2 rejected

    def test_pipeline_no_candidate(self, config_file, neutral_indicators):
        """Pipeline with no Stage 1 candidate → early exit."""
        pipeline = HybridSignalPipeline(config_path=config_file)
        output = pipeline.generate_signals("AAPL", neutral_indicators)

        assert output["stage1"]["passed"] is False
        assert output["stage2"]["skipped"] is True
        assert output["final_signal"] is None

    def test_pipeline_earnings_exclusion(self, config_file, sell_puts_indicators):
        """Earnings within exclusion period → Stage 1 fails."""
        tomorrow = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
        pipeline = HybridSignalPipeline(config_path=config_file)
        output = pipeline.generate_signals("AAPL", sell_puts_indicators, earnings_date=tomorrow)

        assert output["stage1"]["passed"] is False
        # Check filter_results contains the failed earnings condition
        fr = output["stage1"].get("filter_results", [])
        assert len(fr) > 0, "filter_results should not be empty"
        earnings_filter = next((r for r in fr if r.get("condition") == "earnings"), None)
        assert earnings_filter is not None
        assert earnings_filter["passed"] is False

    def test_pipeline_indicators_summary(self, config_file, sell_puts_indicators):
        """Output includes indicators_summary."""
        pipeline = HybridSignalPipeline(config_path=config_file)
        output = pipeline.generate_signals("AAPL", sell_puts_indicators)

        assert "indicators_summary" in output
        assert "RSI_14" in output["indicators_summary"]
        assert "IV_Rank" in output["indicators_summary"]
        assert "Current_Price" in output["indicators_summary"]

    def test_pipeline_earnings_date_none(self, config_file, sell_puts_indicators):
        """No earnings date provided → passes earnings check."""
        pipeline = HybridSignalPipeline(config_path=config_file)
        output = pipeline.generate_signals("AAPL", sell_puts_indicators, earnings_date=None)
        assert output["stage1"]["passed"] is True


# ---------------------------------------------------------------------------
# Backward compatibility: OptionsSignalEngine still works
# ---------------------------------------------------------------------------

class TestOptionsSignalEngine:
    def test_legacy_engine_still_works(self):
        """OptionsSignalEngine.generate_signals_for_ticker still functions."""
        engine = OptionsSignalEngine()
        indicators = {
            "RSI_14": 35.0,
            "MACD": 1.5,
            "MACD_Signal": 0.8,
            "Current_Price": 150.0,
            "SMA_20": 145.0,
            "SMA_200": 130.0,
            "Volume_10d_Avg": 45000000,
            "Volume_30d_Avg": 40000000,
        }
        result = engine.generate_signals_for_ticker("AAPL", indicators)
        assert result.ticker == "AAPL"
        assert len(result.signals) > 0


# ---------------------------------------------------------------------------
# FilterResult dataclass
# ---------------------------------------------------------------------------

class TestFilterResult:
    def test_filter_result_passed(self):
        r = FilterResult(passed=True, condition="rsi", reason="RSI=35.0 < 40")
        assert r.passed is True
        assert r.condition == "rsi"

    def test_filter_result_failed(self):
        r = FilterResult(passed=False, condition="volume", reason="Vol ratio=0.8 < 1.0")
        assert r.passed is False
