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
    NO_TRADE = "NO_TRADE"


class ConfidenceLevel(Enum):
    """Confidence levels for signals."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


# JSON output schema required by Stage 2 LLM — gemma4:e4b via Ollama
LLM_OUTPUT_SCHEMA = {
    "signal_decision": "SELL_PUTS | SELL_CALLS | BUY_LEAPS | NO_TRADE",
    "confidence": 0.87,
    "confidence_level": "HIGH | MEDIUM | LOW",
    "reasoning_summary": "Brief 1-2 sentence explanation of why this signal was chosen or why no trade is recommended.",
    "top_3_reasons": ["reason1", "reason2", "reason3"],
    "strike_recommendation": {
        "strike": 5100,
        "delta_estimate": 0.25,
        "distance_pct": 2.2
    },
    "expiry_recommendation": {
        "target_expiry": "2026-04-30",
        "dte": 14
    },
    "stop_loss": {
        "level": 4998.00,
        "distance_pct": 4.23,
        "distance_atr": 1.0
    },
    "risk_flags": ["flag1", "flag2"]
}


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

    def evaluate_all(self, indicators: dict, earnings_date: Optional[str] = None) -> List[CandidateSignal]:
        """
        Evaluate ALL signal types and return all passing candidates.
        Used by Stage 2 to make one LLM call per ticker with all candidates.
        """
        self.logger.info("Evaluating all Stage 1 signal types", ticker=indicators.get("Ticker", "unknown"))
        candidates: List[CandidateSignal] = []

        for eval_fn, label in [
            (self.evaluate_sell_puts, "SELL_PUTS"),
            (self.evaluate_sell_calls, "SELL_CALLS"),
            (self.evaluate_buy_leaps, "BUY_LEAPS"),
        ]:
            result = eval_fn(indicators, earnings_date)
            if isinstance(result, CandidateSignal):
                self.logger.info(f"{label} candidate found", confidence=result.confidence.value)
                candidates.append(result)

        if not candidates:
            self.logger.warning(
                "Stage 1 rejected ticker — no signal type passed filters",
                ticker=indicators.get("Ticker", "unknown"),
            )
        return candidates


# ============================================================================
# Stage 2: LLM-based signal decision — gemma4:e4b via Ollama
# ============================================================================


class Stage2LLM:
    """
    Stage 2 LLM-based signal decision using gemma4:e4b via Ollama.

    One LLM call per ticker (not per signal type) — aggregates all Stage 1
    passing candidates and produces a final GO/NO_TRADE decision with full
    strike, expiry, and stop-loss recommendations.

    Graceful degradation: if Ollama is unavailable or times out, emits
    NO_TRADE for that ticker and logs a warning. The pipeline never crashes.
    """

    def __init__(self, config_path: str = "config/llm_config.yaml"):
        self.logger = logger.bind(component="Stage2LLM")

        # Load LLM config
        config_file = Path(__file__).parent.parent / config_path
        with open(config_file) as f:
            self.cfg = yaml.safe_load(f)

        # Load prompt templates
        prompts_file = Path(__file__).parent.parent / "config" / "llm_prompts.yaml"
        with open(prompts_file) as f:
            self.prompts = yaml.safe_load(f)

        # Lazily import LLMClient to avoid circular imports
        from src.llm_client import LLMClient
        self.llm = LLMClient(config_path=config_path, logger=self.logger)

    def _format_candidates(self, candidates: List[CandidateSignal], indicators: dict) -> str:
        """Format Stage 1 candidates for the user prompt template."""
        def _fmt(v, fmt):
            return format(v, fmt) if v is not None else "N/A"

        lines = []
        price = indicators.get("Current_Price", 0)

        for c in candidates:
            filter_detail = "; ".join(
                f"{r.condition}={'PASS' if r.passed else 'FAIL'}({r.reason})"
                for r in c.filter_results
            )
            lines.append(
                f"- [{c.signal_type.value}] confidence={c.confidence.value}\n"
                f"  Reasons: {', '.join(c.reasons)}\n"
                f"  Filters: {filter_detail}\n"
                f"  Price=${_fmt(c.current_price, '.2f')}, Target=${_fmt(c.target_price, '.2f')}, "
                f"Stop=${_fmt(c.stop_loss, '.2f')}, Expiry={c.expiry or 'N/A'}"
            )

        return "\n".join(lines) if lines else "(none)"

    def _format_indicators(self, indicators: dict) -> dict:
        """Format indicator values for the prompt template."""
        return {
            "ticker": indicators.get("Ticker", "UNKNOWN"),
            "price": indicators.get("Current_Price", 0),
            "rsi": f"{indicators.get('RSI_14', 'N/A'):.1f}" if isinstance(indicators.get("RSI_14"), (int, float)) else "N/A",
            "iv_rank": f"{indicators.get('IV_Rank', 'N/A'):.1f}" if isinstance(indicators.get("IV_Rank"), (int, float)) else "N/A",
            "macd": f"{indicators.get('MACD', 'N/A'):.4f}" if isinstance(indicators.get("MACD"), (int, float)) else "N/A",
            "macd_signal": f"{indicators.get('MACD_Signal', 'N/A'):.4f}" if isinstance(indicators.get("MACD_Signal"), (int, float)) else "N/A",
            "sma20": f"{indicators.get('SMA_20', 'N/A'):.2f}" if isinstance(indicators.get("SMA_20"), (int, float)) else "N/A",
            "sma200": f"{indicators.get('SMA_200', 'N/A'):.2f}" if isinstance(indicators.get("SMA_200"), (int, float)) else "N/A",
            "bb_upper": f"{indicators.get('BB_Upper', 'N/A'):.2f}" if isinstance(indicators.get("BB_Upper"), (int, float)) else "N/A",
            "bb_lower": f"{indicators.get('BB_Lower', 'N/A'):.2f}" if isinstance(indicators.get("BB_Lower"), (int, float)) else "N/A",
            "atr": f"{indicators.get('ATR_14', 'N/A'):.2f}" if isinstance(indicators.get("ATR_14"), (int, float)) else "N/A",
            "vol_ratio": f"{indicators.get('Volume_Ratio', 'N/A'):.2f}" if isinstance(indicators.get("Volume_Ratio"), (int, float)) else "N/A",
        }

    def _build_user_prompt(self, ticker: str, indicators: dict, candidates: List[CandidateSignal]) -> str:
        """Build the user prompt from the template and runtime data."""
        tmpl = self.prompts["user_prompt_template"]
        ind = self._format_indicators(indicators)
        cand_block = self._format_candidates(candidates, indicators)

        return tmpl.format(
            ticker=ticker,
            price=ind["price"],
            rsi=ind["rsi"],
            iv_rank=ind["iv_rank"],
            macd=ind["macd"],
            macd_signal=ind["macd_signal"],
            sma20=ind["sma20"],
            sma200=ind["sma200"],
            bb_upper=ind["bb_upper"],
            bb_lower=ind["bb_lower"],
            atr=ind["atr"],
            vol_ratio=ind["vol_ratio"],
            stage1_candidates=cand_block,
        )

    def _parse_llm_output(self, raw: dict, price: float, atr: Optional[float]) -> dict:
        """
        Parse LLM JSON output into a clean stage2 result dict.
        Applies ATR-adjusted stop-loss and maps signal_decision to SignalType.
        """
        signal_decision = raw.get("signal_decision", "NO_TRADE")

        # Map to SignalType
        decision_map = {
            "SELL_PUTS": SignalType.SELL_PUTS,
            "SELL_CALLS": SignalType.SELL_CALLS,
            "BUY_LEAPS": SignalType.BUY_LEAPS,
            "NO_TRADE": SignalType.NO_TRADE,
        }
        signal_type = decision_map.get(signal_decision, SignalType.NO_TRADE)

        # ATR-adjusted stop loss
        stop_loss_raw = raw.get("stop_loss", {})
        stop_level = stop_loss_raw.get("level")
        if stop_level is None and atr and price:
            stop_level = round(price - (2 * atr), 2)  # default: 2x ATR below price

        # Confidence level
        conf_level_map = {"HIGH": ConfidenceLevel.HIGH, "MEDIUM": ConfidenceLevel.MEDIUM, "LOW": ConfidenceLevel.LOW}
        conf_level = conf_level_map.get(raw.get("confidence_level", ""), ConfidenceLevel.MEDIUM)

        # Strike recommendation
        strike_rec = raw.get("strike_recommendation", {})
        strike = strike_rec.get("strike") or round(price)
        delta_est = strike_rec.get("delta_estimate", 0.25)
        distance_pct = strike_rec.get("distance_pct", 0.0)

        # Expiry recommendation
        expiry_rec = raw.get("expiry_recommendation", {})
        target_expiry = expiry_rec.get("target_expiry")
        dte = expiry_rec.get("dte", 14)

        return {
            "signal_decision": signal_decision,
            "signal_type": signal_type.value,
            "confidence": raw.get("confidence", 0.5),
            "confidence_level": conf_level.value,
            "reasoning_summary": raw.get("reasoning_summary", ""),
            "top_3_reasons": raw.get("top_3_reasons", [])[:3],
            "strike_recommendation": {
                "strike": strike,
                "delta_estimate": delta_est,
                "distance_pct": round(distance_pct, 2),
            },
            "expiry_recommendation": {
                "target_expiry": target_expiry,
                "dte": dte,
            },
            "stop_loss": {
                "level": stop_level,
                "distance_pct": stop_loss_raw.get("distance_pct", 0.0),
                "distance_atr": stop_loss_raw.get("distance_atr", 0.0),
            },
            "risk_flags": raw.get("risk_flags", []),
        }

    def process_candidates(
        self, ticker: str, indicators: dict, candidates: List[CandidateSignal]
    ) -> dict:
        """
        Process all Stage 1 candidates for a ticker with one LLM call.

        Args:
            ticker: Stock ticker symbol.
            indicators: Dict of technical indicators.
            candidates: List of CandidateSignal objects that passed Stage 1.

        Returns:
            Stage 2 result dict with final signal decision, or NO_TRADE on failure.
            Never raises — graceful degradation always returns a valid dict.
        """
        self.logger.info(
            "Stage 2 LLM called",
            ticker=ticker,
            candidate_count=len(candidates),
            signal_types=[c.signal_type.value for c in candidates],
        )

        if not candidates:
            return {
                "signal_decision": "NO_TRADE",
                "reason": "No Stage 1 candidates to evaluate",
                "stage1_candidates": [],
                "llm_raw": None,
            }

        # Build prompts
        system_prompt = self.prompts["system_prompt"]
        user_prompt = self._build_user_prompt(ticker, indicators, candidates)

        # Call LLM
        price = indicators.get("Current_Price", 0)
        atr = indicators.get("ATR_14")

        raw = self.llm.generate_json(
            prompt=user_prompt,
            system_prompt=system_prompt,
            schema=LLM_OUTPUT_SCHEMA,
        )

        if raw is None:
            # Graceful degradation — Ollama unavailable/timeout
            self.logger.warning(
                "LLM call failed — emitting NO_TRADE, pipeline continues",
                ticker=ticker,
            )
            return {
                "signal_decision": "NO_TRADE",
                "reason": "LLM unavailable or timeout — graceful degradation",
                "stage1_candidates": [c.signal_type.value for c in candidates],
                "llm_raw": None,
            }

        # Parse and return
        result = self._parse_llm_output(raw, price, atr)
        result["stage1_candidates"] = [c.signal_type.value for c in candidates]
        result["llm_raw"] = raw

        self.logger.info(
            "Stage 2 LLM decision",
            ticker=ticker,
            signal_decision=result["signal_decision"],
            confidence=result["confidence"],
        )
        return result


# ============================================================================
# Hybrid Signal Pipeline — Orchestrates Stage 1 + Stage 2
# ============================================================================


class HybridSignalPipeline:
    """
    Hybrid pipeline: Stage1HardFilters → Stage2LLM → Output.

    Single interface: generate_signals(ticker, indicators) -> dict

    Stage 2 aggregates ALL Stage 1 passing candidates (one LLM call per ticker)
    and produces a definitive final signal.
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

        # Stage 1: collect ALL passing candidates
        stage1_candidates = self.stage1.evaluate_all(indicators, earnings_date)

        # If no candidates, return early
        if not stage1_candidates:
            # Run evaluate() once more to get NoCandidate info
            single_result = self.stage1.evaluate(indicators, earnings_date)
            return {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "pipeline": "hybrid_v2",
                "stage1": {
                    "passed": False,
                    "reason": single_result.reason,
                    "filter_results": [
                        {"condition": r.condition, "passed": r.passed, "reason": r.reason}
                        for r in single_result.filter_results
                    ],
                    "all_candidates": [],
                },
                "stage2": {
                    "skipped": True,
                    "reason": "No Stage 1 candidate"
                },
                "final_signal": None,
                "indicators_summary": self._summarize_indicators(indicators)
            }

        # Stage 2: one LLM call with ALL candidates
        stage2_result = self.stage2.process_candidates(ticker, indicators, stage1_candidates)

        # Build final signal from Stage 2 output
        if stage2_result.get("signal_decision") in ("NO_TRADE", "NO_CANDIDATE"):
            final_signal = None
        else:
            sig = stage2_result
            final_signal = {
                "signal_type": sig["signal_type"],
                "confidence": sig["confidence_level"],
                "confidence_score": sig["confidence"],
                "reasoning_summary": sig.get("reasoning_summary", ""),
                "reasoning": sig["top_3_reasons"],
                "current_price": indicators.get("Current_Price"),
                "target_price": sig["strike_recommendation"]["strike"],
                "stop_loss": sig["stop_loss"]["level"],
                "expiry": sig["expiry_recommendation"]["target_expiry"],
                "risk_flags": sig["risk_flags"],
            }

        return {
            "ticker": ticker,
            "timestamp": datetime.now().isoformat(),
            "pipeline": "hybrid_v2",
            "stage1": {
                "passed": True,
                "all_candidates": [
                    {
                        "signal_type": c.signal_type.value,
                        "confidence": c.confidence.value,
                        "reasons": c.reasons,
                    }
                    for c in stage1_candidates
                ],
                "filter_results": [
                    {"condition": r.condition, "passed": r.passed, "reason": r.reason}
                    for c in stage1_candidates
                    for r in c.filter_results
                ]
            },
            "stage2": {
                "skipped": False,
                "decision": stage2_result.get("signal_decision"),
                "reason": stage2_result.get("reason"),
                "confidence": stage2_result.get("confidence"),
                "confidence_level": stage2_result.get("confidence_level"),
                "strike_recommendation": stage2_result.get("strike_recommendation"),
                "expiry_recommendation": stage2_result.get("expiry_recommendation"),
                "stop_loss": stage2_result.get("stop_loss"),
                "risk_flags": stage2_result.get("risk_flags"),
                "stage1_candidates": stage2_result.get("stage1_candidates"),
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
        self.stage1 = Stage1HardFilters()

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
            target_price=round(price * 0.95, 2) if price else None,
            stop_loss=round(price * 0.90, 2) if price else None
        )

    def generate_sell_call_signal(self, indicators: Dict[str, Any]) -> Signal:
        """Generate signal for selling calls."""
        conditions = []
        reasoning = []

        sma20 = indicators.get('SMA_20')
        price = indicators.get('Current_Price')
        if price is not None and sma20 is not None and price < sma20:
            conditions.append(True)
            reasoning.append(f"Price (${price:.2f}) < SMA20 (${sma20:.2f})")
        elif price is not None and sma20 is not None:
            conditions.append(False)
            reasoning.append(f"Price (${price:.2f}) >= SMA20 (${sma20:.2f})")

        rsi = indicators.get('RSI_14')
        if rsi is not None and rsi < 60:
            conditions.append(True)
            reasoning.append(f"RSI ({rsi:.1f}) < 60")
        elif rsi is not None:
            conditions.append(False)
            reasoning.append(f"RSI ({rsi:.1f}) >= 60")

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
            target_price=round(price * 1.05, 2) if price else None,
            stop_loss=round(price * 1.10, 2) if price else None
        )

    def generate_buy_leaps_signal(self, indicators: Dict[str, Any]) -> Signal:
        """Generate signal for buying LEAPS (long-term bullish)."""
        conditions = []
        reasoning = []

        sma200 = indicators.get('SMA_200')
        price = indicators.get('Current_Price')
        if price is not None and sma200 is not None and price > sma200:
            conditions.append(True)
            reasoning.append(f"Price (${price:.2f}) > SMA200 (${sma200:.2f})")
        elif price is not None and sma200 is not None:
            conditions.append(False)
            reasoning.append(f"Price (${price:.2f}) <= SMA200 (${sma200:.2f})")

        rsi = indicators.get('RSI_14')
        if rsi is not None and rsi > 50:
            conditions.append(True)
            reasoning.append(f"RSI ({rsi:.1f}) > 50")
        elif rsi is not None:
            conditions.append(False)
            reasoning.append(f"RSI ({rsi:.1f}) <= 50")

        macd = indicators.get('MACD')
        if macd is not None and macd > 0:
            conditions.append(True)
            reasoning.append(f"MACD ({macd:.4f}) > 0")
        elif macd is not None:
            conditions.append(False)
            reasoning.append(f"MACD ({macd:.4f}) <= 0")

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
            target_price=round(price * 1.20, 2) if price else None,
            stop_loss=round(price * 0.85, 2) if price else None
        )

    def generate_signals_for_ticker(self, ticker: str, indicators: Dict[str, Any]) -> TickerSignals:
        """
        Generate all applicable signals for a ticker.

        Args:
            ticker: Stock ticker symbol
            indicators: Dictionary with calculated indicators (may include Earnings_Date)

        Returns:
            TickerSignals object with all signals
        """
        self.logger.info("Generating signals", ticker=ticker, indicators_count=len(indicators))

        current_price = indicators.get('Current_Price')
        earnings_date = indicators.get('Earnings_Date')

        # Use Stage1HardFilters.evaluate_all() — respects all threshold configs
        # and earnings exclusion. The standalone generate_*_signal() methods
        # bypass these checks and produce incorrect signals.
        stage1_candidates = self.stage1.evaluate_all(indicators, earnings_date)

        signals = []
        for candidate in stage1_candidates:
            sig = Signal(
                signal_type=candidate.signal_type,
                confidence=candidate.confidence,
                reasoning=candidate.reasons,
                current_price=current_price,
                expiry="2 weeks" if candidate.signal_type == SignalType.SELL_PUTS else
                        "2 weeks" if candidate.signal_type == SignalType.SELL_CALLS else
                        "3-6 months",
                target_price=candidate.target_price,
                stop_loss=candidate.stop_loss,
            )
            signals.append(sig)

        if not signals:
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
