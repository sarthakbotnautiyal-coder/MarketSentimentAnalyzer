"""Options signal engine for generating trading recommendations."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import structlog

logger = structlog.get_logger()


class SignalType(Enum):
    """Types of options signals."""
    SELL_PUTS = "SELL_PUTS"
    SELL_CALLS = "SELL_CALLS"
    BUY_LEAPS = "BUY_LEAPS"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"
    NO_CANDIDATE = "NO_CANDIDATE"


class ConfidenceLevel(Enum):
    """Confidence levels for signals."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


@dataclass
class Signal:
    """Represents an options trading signal."""
    signal_type: SignalType
    confidence: ConfidenceLevel
    reasoning: List[str]
    # Additional optional fields
    current_price: float = None
    target_price: float = None
    stop_loss: float = None
    expiry: str = None  # e.g., "2 weeks"

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for JSON serialization."""
        result = asdict(self)
        result['signal_type'] = self.signal_type.value
        result['confidence'] = self.confidence.value
        return result


@dataclass
class TickerSignals:
    """Complete signal output for a ticker."""
    ticker: str
    timestamp: str
    current_price: float
    signals: List[Signal]
    # Include key indicators used in decision
    indicators_summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "current_price": self.current_price,
            "signals": [s.to_dict() for s in self.signals],
            "indicators_summary": self.indicators_summary
        }


# ============================================================================
# Stage 1: Hard Filters — Rule-based fast filtering
# ============================================================================


@dataclass
class FilterResult:
    """Result of a single filter condition."""
    passed: bool
    condition: str
    reason: str


@dataclass
class CandidateSignal:
    """A candidate that passed Stage 1 filters."""
    signal_type: SignalType
    confidence: ConfidenceLevel
    reasons: List[str]
    filter_results: List[FilterResult] = field(default_factory=list)
    current_price: float = None
    target_price: float = None
    stop_loss: float = None
    expiry: str = None


@dataclass
class NoCandidate:
    """No candidate passed Stage 1 filters."""
    reason: str
    filter_results: List[FilterResult] = field(default_factory=list)


class Stage1HardFilters:
    """
    Fast rule-based filter layer for the hybrid signal pipeline.

    Reads all thresholds from config/signal_thresholds.yaml — no hardcoded values.
    Handles missing indicators gracefully (skip condition, log warning).
    """

    def __init__(self, config_path: str = None):
        """Initialize with threshold config."""
        self.logger = logger.bind(component="Stage1HardFilters")

        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "signal_thresholds.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.sell_puts = self.config["sell_puts"]
        self.sell_calls = self.config["sell_calls"]
        self.buy_leaps = self.config["buy_leaps"]

    def _check_earnings_exclusion(
        self, earnings_date: Optional[str], exclude_days: int
    ) -> FilterResult:
        """Check if earnings date is too close."""
        if earnings_date is None:
            return FilterResult(passed=True, condition="earnings", reason="No earnings date")

        try:
            earnings_dt = datetime.strptime(earnings_date, "%Y-%m-%d")
            days_until = (earnings_dt - datetime.now()).days

            if days_until < 0:
                return FilterResult(
                    passed=True,  # Past earnings are fine
                    condition="earnings",
                    reason=f"Earnings already passed ({days_until} days ago)"
                )
            elif days_until <= exclude_days:
                return FilterResult(
                    passed=False,
                    condition="earnings",
                    reason=f"Earnings in {days_until} days — within {exclude_days} day exclusion"
                )
            else:
                return FilterResult(
                    passed=True,
                    condition="earnings",
                    reason=f"Earnings in {days_until} days — safe"
                )
        except ValueError as e:
            self.logger.warning("Could not parse earnings date", earnings_date=earnings_date, error=str(e))
            return FilterResult(passed=True, condition="earnings", reason="Could not parse earnings date")

    def _get_indicator(self, indicators: dict, key: str, default: float = None) -> float:
        """Safely get an indicator, logging a warning if missing."""
        value = indicators.get(key, default)
        if value is None:
            self.logger.warning("Missing indicator", key=key)
        return value

    def evaluate_sell_puts(self, indicators: dict, earnings_date: Optional[str]) -> CandidateSignal | NoCandidate:
        """Evaluate SELL_PUTS conditions (oversold entry → bounce play)."""
        cfg = self.sell_puts
        results: List[FilterResult] = []
        reasons: List[str] = []

        # Check earnings first
        earnings_result = self._check_earnings_exclusion(earnings_date, cfg["earnings_exclude_days"])
        results.append(earnings_result)
        if not earnings_result.passed:
            return NoCandidate(reason="Earnings exclusion", filter_results=results)

        # RSI < rsi_max
        rsi = self._get_indicator(indicators, "RSI_14")
        if rsi is not None:
            rsi_pass = rsi < cfg["rsi_max"]
            results.append(FilterResult(
                passed=rsi_pass,
                condition="rsi",
                reason=f"RSI={rsi:.1f}, required < {cfg['rsi_max']}" +
                       (f" (ideal: {cfg['rsi_ideal_min']}-{cfg['rsi_ideal_max']})" if rsi_pass else "")
            ))
            if rsi_pass:
                reasons.append(f"RSI in oversold zone: {rsi:.1f}")
        else:
            results.append(FilterResult(passed=False, condition="rsi", reason="RSI missing"))

        # IV Rank > iv_rank_min
        iv_rank = self._get_indicator(indicators, "IV_Rank")
        if iv_rank is not None:
            iv_pass = iv_rank > cfg["iv_rank_min"]
            results.append(FilterResult(
                passed=iv_pass,
                condition="iv_rank",
                reason=f"IV Rank={iv_rank:.1f}, required > {cfg['iv_rank_min']}"
            ))
            if iv_pass:
                reasons.append(f"High IV Rank: {iv_rank:.1f}")
        else:
            results.append(FilterResult(passed=False, condition="iv_rank", reason="IV Rank missing"))

        # Price > SMA20
        price = self._get_indicator(indicators, "Current_Price")
        sma20 = self._get_indicator(indicators, "SMA_20")
        if price is not None and sma20 is not None:
            price_pass = price > sma20
            results.append(FilterResult(
                passed=price_pass,
                condition="price_sma20",
                reason=f"Price=${price:.2f}, SMA20=${sma20:.2f}"
            ))
            if price_pass:
                reasons.append(f"Price above SMA20 (${price:.2f} > ${sma20:.2f})")
        else:
            results.append(FilterResult(passed=False, condition="price_sma20", reason="Price or SMA20 missing"))

        # Near lower Bollinger Band
        bb_lower = self._get_indicator(indicators, "BB_Lower")
        if price is not None and bb_lower is not None:
            bb_pass = price <= bb_lower * 1.05  # Within 5% of lower band
            results.append(FilterResult(
                passed=bb_pass,
                condition="bb_lower",
                reason=f"Price near lower BB: ${price:.2f} vs BB_lower=${bb_lower:.2f}"
            ))
            if bb_pass:
                reasons.append(f"Near lower Bollinger Band")
        else:
            results.append(FilterResult(passed=False, condition="bb_lower", reason="BB_Lower missing"))

        # Volume ratio > vol_ratio_min
        vol_ratio = self._get_indicator(indicators, "Volume_Ratio")
        if vol_ratio is not None:
            vol_pass = vol_ratio > cfg["vol_ratio_min"]
            results.append(FilterResult(
                passed=vol_pass,
                condition="volume",
                reason=f"Vol ratio={vol_ratio:.2f}, required > {cfg['vol_ratio_min']}"
            ))
            if vol_pass:
                reasons.append(f"Volume confirmation: {vol_ratio:.2f}x")
        else:
            results.append(FilterResult(passed=False, condition="volume", reason="Volume_Ratio missing"))

        # Count passes — need at least 4 conditions to be a candidate
        passed = [r for r in results if r.passed]
        if len(passed) >= 4:
            confidence = ConfidenceLevel.HIGH if len(passed) >= 5 else ConfidenceLevel.MEDIUM
            return CandidateSignal(
                signal_type=SignalType.SELL_PUTS,
                confidence=confidence,
                reasons=reasons,
                filter_results=results,
                current_price=price,
                target_price=round(price * 0.95, 2) if price else None,
                stop_loss=round(price * 0.90, 2) if price else None,
                expiry="2 weeks"
            )

        return NoCandidate(
            reason=f"Only {len(passed)}/5 conditions passed for SELL_PUTS",
            filter_results=results
        )

    def evaluate_sell_calls(self, indicators: dict, earnings_date: Optional[str]) -> CandidateSignal | NoCandidate:
        """Evaluate SELL_CALLS conditions (overbought entry → mean reversion)."""
        cfg = self.sell_calls
        results: List[FilterResult] = []
        reasons: List[str] = []

        # Check earnings first
        earnings_result = self._check_earnings_exclusion(earnings_date, cfg["earnings_exclude_days"])
        results.append(earnings_result)
        if not earnings_result.passed:
            return NoCandidate(reason="Earnings exclusion", filter_results=results)

        # RSI > rsi_min
        rsi = self._get_indicator(indicators, "RSI_14")
        if rsi is not None:
            rsi_pass = rsi > cfg["rsi_min"]
            results.append(FilterResult(
                passed=rsi_pass,
                condition="rsi",
                reason=f"RSI={rsi:.1f}, required > {cfg['rsi_min']}"
            ))
            if rsi_pass:
                reasons.append(f"RSI in overbought zone: {rsi:.1f}")
        else:
            results.append(FilterResult(passed=False, condition="rsi", reason="RSI missing"))

        # IV Rank > iv_rank_min
        iv_rank = self._get_indicator(indicators, "IV_Rank")
        if iv_rank is not None:
            iv_pass = iv_rank > cfg["iv_rank_min"]
            results.append(FilterResult(
                passed=iv_pass,
                condition="iv_rank",
                reason=f"IV Rank={iv_rank:.1f}, required > {cfg['iv_rank_min']}"
            ))
            if iv_pass:
                reasons.append(f"High IV Rank: {iv_rank:.1f}")
        else:
            results.append(FilterResult(passed=False, condition="iv_rank", reason="IV Rank missing"))

        # Price < SMA20
        price = self._get_indicator(indicators, "Current_Price")
        sma20 = self._get_indicator(indicators, "SMA_20")
        if price is not None and sma20 is not None:
            price_pass = price < sma20
            results.append(FilterResult(
                passed=price_pass,
                condition="price_sma20",
                reason=f"Price=${price:.2f}, SMA20=${sma20:.2f}"
            ))
            if price_pass:
                reasons.append(f"Price below SMA20 (${price:.2f} < ${sma20:.2f})")
        else:
            results.append(FilterResult(passed=False, condition="price_sma20", reason="Price or SMA20 missing"))

        # Near upper Bollinger Band
        bb_upper = self._get_indicator(indicators, "BB_Upper")
        if price is not None and bb_upper is not None:
            bb_pass = price >= bb_upper * 0.95  # Within 5% of upper band
            results.append(FilterResult(
                passed=bb_pass,
                condition="bb_upper",
                reason=f"Price near upper BB: ${price:.2f} vs BB_upper=${bb_upper:.2f}"
            ))
            if bb_pass:
                reasons.append(f"Near upper Bollinger Band")
        else:
            results.append(FilterResult(passed=False, condition="bb_upper", reason="BB_Upper missing"))

        # Volume ratio > vol_ratio_min
        vol_ratio = self._get_indicator(indicators, "Volume_Ratio")
        if vol_ratio is not None:
            vol_pass = vol_ratio > cfg["vol_ratio_min"]
            results.append(FilterResult(
                passed=vol_pass,
                condition="volume",
                reason=f"Vol ratio={vol_ratio:.2f}, required > {cfg['vol_ratio_min']}"
            ))
            if vol_pass:
                reasons.append(f"Volume confirmation: {vol_ratio:.2f}x")
        else:
            results.append(FilterResult(passed=False, condition="volume", reason="Volume_Ratio missing"))

        # Count passes
        passed = [r for r in results if r.passed]
        if len(passed) >= 4:
            confidence = ConfidenceLevel.HIGH if len(passed) >= 5 else ConfidenceLevel.MEDIUM
            return CandidateSignal(
                signal_type=SignalType.SELL_CALLS,
                confidence=confidence,
                reasons=reasons,
                filter_results=results,
                current_price=price,
                target_price=round(price * 1.05, 2) if price else None,
                stop_loss=round(price * 1.10, 2) if price else None,
                expiry="2 weeks"
            )

        return NoCandidate(
            reason=f"Only {len(passed)}/5 conditions passed for SELL_CALLS",
            filter_results=results
        )

    def evaluate_buy_leaps(self, indicators: dict, earnings_date: Optional[str]) -> CandidateSignal | NoCandidate:
        """Evaluate BUY_LEAPS conditions (long-term bullish)."""
        cfg = self.buy_leaps
        results: List[FilterResult] = []
        reasons: List[str] = []

        # Check earnings first
        earnings_result = self._check_earnings_exclusion(earnings_date, cfg["earnings_exclude_days"])
        results.append(earnings_result)
        if not earnings_result.passed:
            return NoCandidate(reason="Earnings exclusion", filter_results=results)

        # Price > SMA200
        price = self._get_indicator(indicators, "Current_Price")
        sma200 = self._get_indicator(indicators, "SMA_200")
        if price is not None and sma200 is not None:
            price_pass = price > sma200
            results.append(FilterResult(
                passed=price_pass,
                condition="price_sma200",
                reason=f"Price=${price:.2f}, SMA200=${sma200:.2f}"
            ))
            if price_pass:
                reasons.append(f"Price above SMA200 (${price:.2f} > ${sma200:.2f})")
        else:
            results.append(FilterResult(passed=False, condition="price_sma200", reason="Price or SMA200 missing"))

        # RSI > rsi_min
        rsi = self._get_indicator(indicators, "RSI_14")
        if rsi is not None:
            rsi_pass = rsi > cfg["rsi_min"]
            results.append(FilterResult(
                passed=rsi_pass,
                condition="rsi",
                reason=f"RSI={rsi:.1f}, required > {cfg['rsi_min']}"
            ))
            if rsi_pass:
                reasons.append(f"RSI bullish: {rsi:.1f}")
        else:
            results.append(FilterResult(passed=False, condition="rsi", reason="RSI missing"))

        # MACD > 0
        macd = self._get_indicator(indicators, "MACD")
        if macd is not None:
            macd_pass = macd > 0
            results.append(FilterResult(
                passed=macd_pass,
                condition="macd",
                reason=f"MACD={macd:.4f}, required > 0"
            ))
            if macd_pass:
                reasons.append(f"MACD positive: {macd:.4f}")
        else:
            results.append(FilterResult(passed=False, condition="macd", reason="MACD missing"))

        # Volume ratio > vol_ratio_min
        vol_ratio = self._get_indicator(indicators, "Volume_Ratio")
        if vol_ratio is not None:
            vol_pass = vol_ratio > cfg["vol_ratio_min"]
            results.append(FilterResult(
                passed=vol_pass,
                condition="volume",
                reason=f"Vol ratio={vol_ratio:.2f}, required > {cfg['vol_ratio_min']}"
            ))
            if vol_pass:
                reasons.append(f"Volume confirmation: {vol_ratio:.2f}x")
        else:
            results.append(FilterResult(passed=False, condition="volume", reason="Volume_Ratio missing"))

        # IV Rank < iv_rank_max (low IV preferred for LEAPS)
        iv_rank = self._get_indicator(indicators, "IV_Rank")
        if iv_rank is not None:
            iv_pass = iv_rank < cfg["iv_rank_max"]
            results.append(FilterResult(
                passed=iv_pass,
                condition="iv_rank",
                reason=f"IV Rank={iv_rank:.1f}, required < {cfg['iv_rank_max']}"
            ))
            if iv_pass:
                reasons.append(f"Low IV Rank (good for LEAPS): {iv_rank:.1f}")
        else:
            results.append(FilterResult(passed=False, condition="iv_rank", reason="IV Rank missing"))

        # Count passes — need at least 4 conditions
        passed = [r for r in results if r.passed]
        if len(passed) >= 4:
            confidence = ConfidenceLevel.HIGH if len(passed) >= 5 else ConfidenceLevel.MEDIUM
            return CandidateSignal(
                signal_type=SignalType.BUY_LEAPS,
                confidence=confidence,
                reasons=reasons,
                filter_results=results,
                current_price=price,
                target_price=round(price * 1.20, 2) if price else None,
                stop_loss=round(price * 0.85, 2) if price else None,
                expiry="3-6 months"
            )

        return NoCandidate(
            reason=f"Only {len(passed)}/5 conditions passed for BUY_LEAPS",
            filter_results=results
        )

    def evaluate(self, indicators: dict, earnings_date: Optional[str] = None) -> CandidateSignal | NoCandidate:
        """
        Evaluate all three signal types and return the first candidate.
        Tries in order: SELL_PUTS, SELL_CALLS, BUY_LEAPS.
        """
        self.logger.info("Evaluating Stage 1 hard filters", ticker=indicators.get("Ticker", "unknown"))

        # Try SELL_PUTS
        result = self.evaluate_sell_puts(indicators, earnings_date)
        if isinstance(result, CandidateSignal):
            self.logger.info("SELL_PUTS candidate found", confidence=result.confidence.value)
            return result

        # Try SELL_CALLS
        result = self.evaluate_sell_calls(indicators, earnings_date)
        if isinstance(result, CandidateSignal):
            self.logger.info("SELL_CALLS candidate found", confidence=result.confidence.value)
            return result

        # Try BUY_LEAPS
        result = self.evaluate_buy_leaps(indicators, earnings_date)
        if isinstance(result, CandidateSignal):
            self.logger.info("BUY_LEAPS candidate found", confidence=result.confidence.value)
            return result

        # All failed — preserve filter_results from the last NoCandidate
        return NoCandidate(
            reason=result.reason if isinstance(result, NoCandidate) else "No signal type passed Stage 1 filters",
            filter_results=result.filter_results if isinstance(result, NoCandidate) else []
        )

# Stage 2: LLM Stub — placeholder for future LLM-based decision making
# ============================================================================


class Stage2LLM:
    """
    Stage 2 LLM-based signal refinement (stub implementation).

    In production, this would call an LLM API to:
    - Validate the Stage 1 candidate against news/sentiment
    - Check for upcoming catalysts
    - Make a final GO/NO_GO decision

    For now: always returns NO_CANDIDATE to keep pipeline intact.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.logger = logger.bind(component="Stage2LLM")
        self.api_key = api_key
        self.model = model

    def process_candidate(
        self, ticker: str, indicators: dict, stage1_result: CandidateSignal
    ) -> dict:
        """
        Process a Stage 1 candidate and return Stage 2 decision.

        Returns dict with:
        - signal_decision: "NO_CANDIDATE" | "GO" | "NO_GO"
        - reason: str
        - confidence_override: ConfidenceLevel (optional)
        """
        self.logger.info(
            "Stage 2 LLM called (stub)",
            ticker=ticker,
            stage1_signal=stage1_result.signal_type.value
        )

        # Stub: reject all candidates until Stage 2 is implemented
        return {
            "signal_decision": "NO_CANDIDATE",
            "reason": "Stage 2 not yet implemented — Stage 1 candidates are informational only",
            "stage1_candidate": stage1_result.signal_type.value,
        }


# ============================================================================
# Hybrid Signal Pipeline — Orchestrates Stage 1 + Stage 2
# ============================================================================


class HybridSignalPipeline:
    """
    Hybrid pipeline: Stage1HardFilters → [Stage2LLM] → Output.

    Single interface: generate_signals(ticker, indicators) -> dict

    Stage 2 is stubbed — candidates pass through but are flagged as pending.
    """

    def __init__(self, config_path: str = None):
        self.logger = logger.bind(component="HybridSignalPipeline")
        self.stage1 = Stage1HardFilters(config_path=config_path)
        self.stage2 = Stage2LLM()

    def generate_signals(
        self, ticker: str, indicators: dict, earnings_date: Optional[str] = None
    ) -> dict:
        """
        Generate signals using the hybrid pipeline.

        Args:
            ticker: Stock ticker symbol
            indicators: Dict of technical indicators
            earnings_date: Optional earnings date string (YYYY-MM-DD)

        Returns:
            dict with pipeline output including stage results
        """
        self.logger.info("Pipeline invoked", ticker=ticker)

        # Stage 1: Hard Filters
        stage1_result = self.stage1.evaluate(indicators, earnings_date)

        # If no candidate, return early
        if isinstance(stage1_result, NoCandidate):
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "pipeline": "hybrid_v1",
                "stage1": {
                    "passed": False,
                    "reason": stage1_result.reason,
                    "filter_results": [
                        {"condition": r.condition, "passed": r.passed, "reason": r.reason}
                        for r in stage1_result.filter_results
                    ]
                },
                "stage2": {
                    "skipped": True,
                    "reason": "No Stage 1 candidate"
                },
                "final_signal": None,
                "indicators_summary": self._summarize_indicators(indicators)
            }

        # Stage 1 passed — send to Stage 2 stub
        stage2_result = self.stage2.process_candidate(ticker, indicators, stage1_result)

        # Build final signal
        if stage2_result.get("signal_decision") == "NO_CANDIDATE":
            # Stage 2 rejected or stub rejected
            final_signal = None
        else:
            # Stage 2 approved — build full signal
            final_signal = {
                "signal_type": stage1_result.signal_type.value,
                "confidence": stage2_result.get("confidence_override", stage1_result.confidence.value),
                "reasoning": stage1_result.reasons,
                "current_price": stage1_result.current_price,
                "target_price": stage1_result.target_price,
                "stop_loss": stage1_result.stop_loss,
                "expiry": stage1_result.expiry
            }

        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "pipeline": "hybrid_v1",
            "stage1": {
                "passed": True,
                "signal_type": stage1_result.signal_type.value,
                "confidence": stage1_result.confidence.value,
                "reasons": stage1_result.reasons,
                "filter_results": [
                    {"condition": r.condition, "passed": r.passed, "reason": r.reason}
                    for r in stage1_result.filter_results
                ]
            },
            "stage2": {
                "skipped": False,
                "decision": stage2_result.get("signal_decision"),
                "reason": stage2_result.get("reason"),
                "stage1_candidate": stage2_result.get("stage1_candidate")
            },
            "final_signal": final_signal,
            "indicators_summary": self._summarize_indicators(indicators)
        }

    def _summarize_indicators(self, indicators: dict) -> dict:
        """Extract key indicators for summary."""
        keys = [
            "RSI_14", "MACD", "MACD_Signal", "SMA_20", "SMA_200",
            "BB_Upper", "BB_Lower", "ATR_14", "Volume_Ratio", "IV_Rank",
            "Current_Price"
        ]
        return {k: indicators.get(k) for k in keys if k in indicators}


# ============================================================================
# Legacy OptionsSignalEngine — kept for backward compatibility
# ============================================================================

class OptionsSignalEngine:
    """
    Generates options trading signals based on technical indicators.

    Signal Rules:
    - Sell puts (bullish/neutral, ~2-week expiry):
        * price > SMA20
        * RSI > 40
        * MACD bullish (MACD > Signal)
        * Strong volume (Volume_10d_Avg > Volume_30d_Avg)

    - Sell calls (bearish/neutral, ~2-week expiry):
        * price < SMA20
        * RSI < 60
        * MACD bearish (MACD < Signal)

    - Buy leaps (long-term bullish, 3-6 months):
        * price > SMA200
        * RSI > 50
        * MACD positive (MACD > 0)
        * Growing volume (Volume_10d_Avg > Volume_30d_Avg)
    """

    def __init__(self):
        """Initialize the signal engine."""
        self.logger = logger.bind(component="OptionsSignalEngine")

    def evaluate_signal_confidence(self, conditions_met: int, total_conditions: int) -> ConfidenceLevel:
        """Determine confidence level based on percentage of conditions met."""
        if total_conditions == 0:
            return ConfidenceLevel.NONE
        percentage = (conditions_met / total_conditions) * 100
        if percentage >= 90:
            return ConfidenceLevel.HIGH
        elif percentage >= 70:
            return ConfidenceLevel.MEDIUM
        elif percentage >= 50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.NONE

    def generate_sell_put_signal(self, indicators: Dict[str, Any]) -> Signal:
        """Generate signal for selling puts."""
        conditions = []
        reasoning = []

        # Condition 1: price > SMA20
        sma20 = indicators.get('SMA_20')
        price = indicators.get('Current_Price')
        if price is not None and sma20 is not None and price > sma20:
            conditions.append(True)
            reasoning.append(f"Price (${price:.2f}) > SMA20 (${sma20:.2f})")
        elif price is not None and sma20 is not None:
            conditions.append(False)
            reasoning.append(f"Price (${price:.2f}) <= SMA20 (${sma20:.2f})")

        # Condition 2: RSI > 40
        rsi = indicators.get('RSI_14')
        if rsi is not None and rsi > 40:
            conditions.append(True)
            reasoning.append(f"RSI ({rsi:.1f}) > 40")
        elif rsi is not None:
            conditions.append(False)
            reasoning.append(f"RSI ({rsi:.1f}) <= 40")

        # Condition 3: MACD bullish (MACD > Signal)
        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_Signal')
        if macd is not None and macd_signal is not None and macd > macd_signal:
            conditions.append(True)
            reasoning.append(f"MACD ({macd:.4f}) > Signal ({macd_signal:.4f})")
        elif macd is not None and macd_signal is not None:
            conditions.append(False)
            reasoning.append(f"MACD ({macd:.4f}) <= Signal ({macd_signal:.4f})")

        # Condition 4: Strong volume (10d avg > 30d avg)
        vol_10d = indicators.get('Volume_10d_Avg')
        vol_30d = indicators.get('Volume_30d_Avg')
        if vol_10d is not None and vol_30d is not None and vol_10d > vol_30d:
            conditions.append(True)
            reasoning.append(f"10-day volume ({vol_10d:,.0f}) > 30-day average ({vol_30d:,.0f})")
        elif vol_10d is not None and vol_30d is not None:
            conditions.append(False)
            reasoning.append(f"10-day volume ({vol_10d:,.0f}) <= 30-day average ({vol_30d:,.0f})")

        confidence = self.evaluate_signal_confidence(len(conditions), 4)

        if confidence == ConfidenceLevel.NONE:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=ConfidenceLevel.NONE,
                reasoning=["Not enough bullish conditions to sell puts"],
                current_price=price
            )

        return Signal(
            signal_type=SignalType.SELL_PUTS,
            confidence=confidence,
            reasoning=reasoning,
            current_price=price,
            expiry="2 weeks",
            # For selling puts, strike price typically below current price
            target_price=round(price * 0.95, 2) if price else None,
            stop_loss=round(price * 0.90, 2) if price else None
        )

    def generate_sell_call_signal(self, indicators: Dict[str, Any]) -> Signal:
        """Generate signal for selling calls."""
        conditions = []
        reasoning = []

        # Condition 1: price < SMA20
        sma20 = indicators.get('SMA_20')
        price = indicators.get('Current_Price')
        if price is not None and sma20 is not None and price < sma20:
            conditions.append(True)
            reasoning.append(f"Price (${price:.2f}) < SMA20 (${sma20:.2f})")
        elif price is not None and sma20 is not None:
            conditions.append(False)
            reasoning.append(f"Price (${price:.2f}) >= SMA20 (${sma20:.2f})")

        # Condition 2: RSI < 60
        rsi = indicators.get('RSI_14')
        if rsi is not None and rsi < 60:
            conditions.append(True)
            reasoning.append(f"RSI ({rsi:.1f}) < 60")
        elif rsi is not None:
            conditions.append(False)
            reasoning.append(f"RSI ({rsi:.1f}) >= 60")

        # Condition 3: MACD bearish (MACD < Signal)
        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_Signal')
        if macd is not None and macd_signal is not None and macd < macd_signal:
            conditions.append(True)
            reasoning.append(f"MACD ({macd:.4f}) < Signal ({macd_signal:.4f})")
        elif macd is not None and macd_signal is not None:
            conditions.append(False)
            reasoning.append(f"MACD ({macd:.4f}) >= Signal ({macd_signal:.4f})")

        conditions_met = sum(1 for c in conditions if c)
        confidence = self.evaluate_signal_confidence(conditions_met, 3)

        if confidence == ConfidenceLevel.NONE:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=ConfidenceLevel.NONE,
                reasoning=["Not enough bearish conditions to sell calls"],
                current_price=price
            )

        return Signal(
            signal_type=SignalType.SELL_CALLS,
            confidence=confidence,
            reasoning=reasoning,
            current_price=price,
            expiry="2 weeks",
            # For selling calls, strike price typically above current price
            target_price=round(price * 1.05, 2) if price else None,
            stop_loss=round(price * 1.10, 2) if price else None
        )

    def generate_buy_leaps_signal(self, indicators: Dict[str, Any]) -> Signal:
        """Generate signal for buying LEAPS (long-term bullish)."""
        conditions = []
        reasoning = []

        # Condition 1: price > SMA200 (long-term uptrend)
        sma200 = indicators.get('SMA_200')
        price = indicators.get('Current_Price')
        if price is not None and sma200 is not None and price > sma200:
            conditions.append(True)
            reasoning.append(f"Price (${price:.2f}) > SMA200 (${sma200:.2f})")
        elif price is not None and sma200 is not None:
            conditions.append(False)
            reasoning.append(f"Price (${price:.2f}) <= SMA200 (${sma200:.2f})")

        # Condition 2: RSI > 50 (momentum)
        rsi = indicators.get('RSI_14')
        if rsi is not None and rsi > 50:
            conditions.append(True)
            reasoning.append(f"RSI ({rsi:.1f}) > 50")
        elif rsi is not None:
            conditions.append(False)
            reasoning.append(f"RSI ({rsi:.1f}) <= 50")

        # Condition 3: MACD positive (above zero)
        macd = indicators.get('MACD')
        if macd is not None and macd > 0:
            conditions.append(True)
            reasoning.append(f"MACD ({macd:.4f}) > 0")
        elif macd is not None:
            conditions.append(False)
            reasoning.append(f"MACD ({macd:.4f}) <= 0")

        # Condition 4: Growing volume (10d avg > 30d avg)
        vol_10d = indicators.get('Volume_10d_Avg')
        vol_30d = indicators.get('Volume_30d_Avg')
        if vol_10d is not None and vol_30d is not None and vol_10d > vol_30d:
            conditions.append(True)
            reasoning.append(f"Volume trend positive: 10-day ({vol_10d:,.0f}) > 30-day ({vol_30d:,.0f})")
        elif vol_10d is not None and vol_30d is not None:
            conditions.append(False)
            reasoning.append(f"Volume trend negative: 10-day ({vol_10d:,.0f}) <= 30-day ({vol_30d:,.0f})")

        confidence = self.evaluate_signal_confidence(len(conditions), 4)

        if confidence == ConfidenceLevel.NONE:
            return Signal(
                signal_type=SignalType.HOLD,
                confidence=ConfidenceLevel.NONE,
                reasoning=["Not enough long-term bullish conditions for LEAPS"],
                current_price=price
            )

        return Signal(
            signal_type=SignalType.BUY_LEAPS,
            confidence=confidence,
            reasoning=reasoning,
            current_price=price,
            expiry="3-6 months",
            # For LEAPS, target is higher price appreciation
            target_price=round(price * 1.20, 2) if price else None,
            stop_loss=round(price * 0.85, 2) if price else None
        )

    def generate_signals_for_ticker(self, ticker: str, indicators: Dict[str, Any]) -> TickerSignals:
        """
        Generate all applicable signals for a ticker.

        Args:
            ticker: Stock ticker symbol
            indicators: Dictionary with calculated indicators

        Returns:
            TickerSignals object with all signals
        """
        self.logger.info("Generating signals", ticker=ticker, indicators_count=len(indicators))

        # Get current price
        current_price = indicators.get('Current_Price')

        # Generate individual signals
        sell_put_signal = self.generate_sell_put_signal(indicators)
        sell_call_signal = self.generate_sell_call_signal(indicators)
        buy_leaps_signal = self.generate_buy_leaps_signal(indicators)

        # Collect non-hold signals
        signals = []
        for signal in [sell_put_signal, sell_call_signal, buy_leaps_signal]:
            if signal.signal_type != SignalType.HOLD:
                signals.append(signal)

        # If no strong signals, add a neutral recommendation
        if not signals:
            # Determine overall trend direction
            trend = "neutral"
            rsi = indicators.get('RSI_14', 50)
            if rsi and rsi > 55:
                trend = "bullish-leaning"
            elif rsi and rsi < 45:
                trend = "bearish-leaning"

            neutral_signal = Signal(
                signal_type=SignalType.NEUTRAL,
                confidence=ConfidenceLevel.NONE,
                reasoning=[f"No clear options signal. Current trend: {trend}. RSI: {rsi:.1f}" if rsi else "Insufficient data for signal"],
                current_price=current_price
            )
            signals.append(neutral_signal)

        # Create summary of key indicators
        indicators_summary = {
            "RSI_14": indicators.get('RSI_14'),
            "MACD": indicators.get('MACD'),
            "MACD_Signal": indicators.get('MACD_Signal'),
            "SMA_20": indicators.get('SMA_20'),
            "SMA_200": indicators.get('SMA_200'),
            "BB_Upper": indicators.get('BB_Upper'),
            "BB_Lower": indicators.get('BB_Lower'),
            "ATR_14": indicators.get('ATR_14'),
            "Volume_10d_Avg": indicators.get('Volume_10d_Avg'),
            "Volume_30d_Avg": indicators.get('Volume_30d_Avg'),
            "Volume_Ratio": indicators.get('Volume_Ratio'),
            "Recent_High": indicators.get('Recent_High'),
            "Recent_Low": indicators.get('Recent_Low'),
        }

        return TickerSignals(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            current_price=current_price,
            signals=signals,
            indicators_summary=indicators_summary
        )
