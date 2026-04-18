"""Options signal engine for generating trading recommendations."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import structlog

logger = structlog.get_logger()


class SignalType(Enum):
    """Types of options signals."""
    SELL_PUTS = "SELL_PUTS"
    SELL_CALLS = "SELL_CALLS"
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
    "type": "object",
    "properties": {
        "signal_decision": {
            "type": "string",
            "enum": ["SELL_PUTS", "SELL_CALLS", "NO_TRADE"]
        },
        "confidence": {"type": "number"},
        "confidence_level": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW"]
        },
        "reasoning_summary": {"type": "string"},
        "top_3_reasons": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3
        },
        "strike_recommendation": {
            "type": "object",
            "properties": {
                "strike": {"type": "number"},
                "delta_estimate": {"type": "number"},
                "distance_pct": {"type": "number"}
            }
        },
        "expiry_recommendation": {
            "type": "object",
            "properties": {
                "target_expiry": {"type": "string"},
                "dte": {"type": "integer"}
            }
        },
        "stop_loss": {
            "type": "object",
            "properties": {
                "level": {"type": "number"},
                "distance_pct": {"type": "number"},
                "distance_atr": {"type": "number"}
            }
        },
        "risk_flags": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["signal_decision", "confidence", "confidence_level", "top_3_reasons"]
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
    # Near-boundary warning: True if any condition was within 10% of threshold
    filter_warning: bool = False


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

    Public methods maintain backward-compatible signatures.
    Internal _*_with_warning() methods return filter_warning for pipeline use.
    """

    # Threshold for flagging near-boundary cases (within 10% of threshold)
    NEAR_BOUNDARY_THRESHOLD = 0.10

    def __init__(self, config_path: str = None):
        """Initialize with threshold config."""
        self.logger = logger.bind(component="Stage1HardFilters")

        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "signal_thresholds.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.sell_puts = self.config["sell_puts"]
        self.sell_calls = self.config["sell_calls"]

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

    def _is_near_boundary(self, value: float, threshold: float) -> bool:
        """
        Check if a value is within 10% of a threshold boundary.
        Used to flag borderline cases for filter_warning.
        """
        if value is None or threshold is None or threshold == 0:
            return False
        distance = abs(value - threshold)
        return distance <= threshold * self.NEAR_BOUNDARY_THRESHOLD

    # ------------------------------------------------------------------
    # Internal helpers — return (result, filter_warning) tuples
    # ------------------------------------------------------------------

    def _eval_sell_puts_with_warning(
        self, indicators: dict, earnings_date: Optional[str]
    ) -> Tuple[CandidateSignal | NoCandidate, bool]:
        """
        Internal: evaluate SELL_PUTS + compute filter_warning.
        Returns (result, filter_warning) for pipeline use.
        """
        cfg = self.sell_puts
        results: List[FilterResult] = []
        reasons: List[str] = []
        filter_warning = False

        # Check earnings first
        earnings_result = self._check_earnings_exclusion(earnings_date, cfg["earnings_exclude_days"])
        results.append(earnings_result)
        if not earnings_result.passed:
            return NoCandidate(reason="Earnings exclusion", filter_results=results), False

        # RSI < rsi_max
        rsi = self._get_indicator(indicators, "RSI_14")
        if rsi is not None:
            rsi_pass = rsi < cfg["rsi_max"]
            if not rsi_pass and self._is_near_boundary(rsi, cfg["rsi_max"]):
                filter_warning = True
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
            if not iv_pass and self._is_near_boundary(iv_rank, cfg["iv_rank_min"]):
                filter_warning = True
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
            # Near boundary: price above SMA20 but within 10% of it
            if price > sma20 and (price - sma20) / sma20 <= self.NEAR_BOUNDARY_THRESHOLD:
                filter_warning = True
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
        if price is not None and bb_lower is not None and bb_lower > 0:
            bb_pass = price <= bb_lower * 1.05  # Within 5% of lower band
            if not bb_pass and (price - bb_lower) / bb_lower <= self.NEAR_BOUNDARY_THRESHOLD:
                filter_warning = True
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
            if not vol_pass and self._is_near_boundary(vol_ratio, cfg["vol_ratio_min"]):
                filter_warning = True
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
            cand = CandidateSignal(
                signal_type=SignalType.SELL_PUTS,
                confidence=confidence,
                reasons=reasons,
                filter_results=results,
                current_price=price,
                target_price=round(price * 0.95, 2) if price else None,
                stop_loss=round(price * 0.90, 2) if price else None,
                expiry="2 weeks",
                filter_warning=filter_warning,
            )
            return cand, filter_warning

        return NoCandidate(
            reason=f"Only {len(passed)}/5 conditions passed for SELL_PUTS",
            filter_results=results
        ), filter_warning

    def _eval_sell_calls_with_warning(
        self, indicators: dict, earnings_date: Optional[str]
    ) -> Tuple[CandidateSignal | NoCandidate, bool]:
        """
        Internal: evaluate SELL_CALLS + compute filter_warning.
        Returns (result, filter_warning) for pipeline use.
        """
        cfg = self.sell_calls
        results: List[FilterResult] = []
        reasons: List[str] = []
        filter_warning = False

        # Check earnings first
        earnings_result = self._check_earnings_exclusion(earnings_date, cfg["earnings_exclude_days"])
        results.append(earnings_result)
        if not earnings_result.passed:
            return NoCandidate(reason="Earnings exclusion", filter_results=results), False

        # RSI > rsi_min
        rsi = self._get_indicator(indicators, "RSI_14")
        if rsi is not None:
            rsi_pass = rsi > cfg["rsi_min"]
            if not rsi_pass and self._is_near_boundary(rsi, cfg["rsi_min"]):
                filter_warning = True
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
            if not iv_pass and self._is_near_boundary(iv_rank, cfg["iv_rank_min"]):
                filter_warning = True
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
            # Near boundary: price below SMA20 but within 10% of it
            if price < sma20 and (sma20 - price) / sma20 <= self.NEAR_BOUNDARY_THRESHOLD:
                filter_warning = True
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
        if price is not None and bb_upper is not None and bb_upper > 0:
            bb_pass = price >= bb_upper * 0.95  # Within 5% of upper band
            if not bb_pass and (bb_upper - price) / bb_upper <= self.NEAR_BOUNDARY_THRESHOLD:
                filter_warning = True
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
            if not vol_pass and self._is_near_boundary(vol_ratio, cfg["vol_ratio_min"]):
                filter_warning = True
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
            cand = CandidateSignal(
                signal_type=SignalType.SELL_CALLS,
                confidence=confidence,
                reasons=reasons,
                filter_results=results,
                current_price=price,
                target_price=round(price * 1.05, 2) if price else None,
                stop_loss=round(price * 1.10, 2) if price else None,
                expiry="2 weeks",
                filter_warning=filter_warning,
            )
            return cand, filter_warning

        return NoCandidate(
            reason=f"Only {len(passed)}/5 conditions passed for SELL_CALLS",
            filter_results=results
        ), filter_warning

    # ------------------------------------------------------------------
    # Public API — backward-compatible signatures (no filter_warning in return)
    # ------------------------------------------------------------------

    def evaluate(
        self, indicators: dict, earnings_date: Optional[str] = None
    ) -> CandidateSignal | NoCandidate:
        """
        Evaluate both signal types and return the first candidate.
        Tries in order: SELL_PUTS, SELL_CALLS.

        Backward-compatible: returns CandidateSignal | NoCandidate directly.
        Use evaluate_all() for the full candidate list.
        """
        self.logger.info("Evaluating Stage 1 hard filters", ticker=indicators.get("Ticker", "unknown"))

        # Try SELL_PUTS
        result, _ = self._eval_sell_puts_with_warning(indicators, earnings_date)
        if isinstance(result, CandidateSignal):
            self.logger.info("SELL_PUTS candidate found", confidence=result.confidence.value)
            return result

        # Try SELL_CALLS
        result, _ = self._eval_sell_calls_with_warning(indicators, earnings_date)
        if isinstance(result, CandidateSignal):
            self.logger.info("SELL_CALLS candidate found", confidence=result.confidence.value)
            return result

        # All failed
        return result

    def evaluate_all(self, indicators: dict, earnings_date: Optional[str] = None) -> List[CandidateSignal]:
        """
        Evaluate ALL signal types and return all passing candidates.
        Backward-compatible: returns List[CandidateSignal] directly.
        """
        self.logger.info("Evaluating all Stage 1 signal types", ticker=indicators.get("Ticker", "unknown"))
        candidates: List[CandidateSignal] = []

        for eval_fn, label in [
            (self._eval_sell_puts_with_warning, "SELL_PUTS"),
            (self._eval_sell_calls_with_warning, "SELL_CALLS"),
        ]:
            result, _ = eval_fn(indicators, earnings_date)
            if isinstance(result, CandidateSignal):
                self.logger.info(f"{label} candidate found", confidence=result.confidence.value)
                candidates.append(result)

        if not candidates:
            self.logger.warning(
                "Stage 1 rejected ticker — no signal type passed filters",
                ticker=indicators.get("Ticker", "unknown"),
            )
        return candidates

    def evaluate_with_warning(
        self, indicators: dict, earnings_date: Optional[str] = None
    ) -> Tuple[CandidateSignal | NoCandidate, bool]:
        """
        Backward-compatible evaluate + filter_warning for pipeline use.
        Returns (result, filter_warning).
        """
        result, fw = self._eval_sell_puts_with_warning(indicators, earnings_date)
        if isinstance(result, CandidateSignal):
            return result, fw

        result, fw = self._eval_sell_calls_with_warning(indicators, earnings_date)
        if isinstance(result, CandidateSignal):
            return result, fw

        return result, fw

    def evaluate_all_with_warning(
        self, indicators: dict, earnings_date: Optional[str] = None
    ) -> Tuple[List[CandidateSignal], bool]:
        """
        Backward-compatible evaluate_all + filter_warning for pipeline use.
        Returns (candidates, any_filter_warning).
        """
        candidates: List[CandidateSignal] = []
        any_warning = False

        for eval_fn, label in [
            (self._eval_sell_puts_with_warning, "SELL_PUTS"),
            (self._eval_sell_calls_with_warning, "SELL_CALLS"),
        ]:
            result, fw = eval_fn(indicators, earnings_date)
            if isinstance(result, CandidateSignal):
                candidates.append(result)
                if fw:
                    any_warning = True

        return candidates, any_warning


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

    Config: config/llm_config.yaml (Ollama endpoint, model, timeout, retry)
    Prompts: config/llm_prompt.yaml (system + user templates, editable without code)
    """

    # URL path for Ollama API (matches config/llm_config.yaml default)
    OLLAMA_API_PATH = "/api/generate"

    def __init__(self, config_path: str = "config/llm_config.yaml"):
        self.logger = logger.bind(component="Stage2LLM")

        # Load LLM config
        config_file = Path(__file__).parent.parent / config_path
        with open(config_file) as f:
            self.cfg = yaml.safe_load(f)

        # Load NEW prompt templates from llm_prompt.yaml
        prompts_file = Path(__file__).parent.parent / "config" / "llm_prompt.yaml"
        with open(prompts_file) as f:
            self.prompts = yaml.safe_load(f)

        # Lazily import LLMClient to avoid circular imports
        from src.llm_client import LLMClient
        self.llm = LLMClient(config_path=config_path, logger=self.logger)

    def _build_indicators_json(self, indicators: dict) -> str:
        """Build a clean JSON string of key indicators for the user prompt."""
        keys = [
            "Ticker", "Current_Price", "RSI_14", "IV_Rank", "MACD",
            "MACD_Signal", "SMA_20", "SMA_200", "BB_Upper", "BB_Lower",
            "ATR_14", "Volume_Ratio"
        ]
        filtered = {k: indicators.get(k) for k in keys if k in indicators}
        return json.dumps(filtered, indent=2)

    def _build_stage1_pass_details(self, candidates: List[CandidateSignal]) -> str:
        """Build stage1_passed detail string for the user prompt."""
        lines = []
        for c in candidates:
            filter_lines = []
            for r in c.filter_results:
                status = "PASS" if r.passed else "FAIL"
                filter_lines.append(f"  {r.condition}: {status} — {r.reason}")
            lines.append(
                f"[{c.signal_type.value}] confidence={c.confidence.value}\n"
                + "\n".join(filter_lines)
            )
        return "\n\n".join(lines) if lines else "(none)"

    def _build_user_prompt(
        self,
        ticker: str,
        indicators: dict,
        candidates: List[CandidateSignal],
        filter_warning: bool = False
    ) -> str:
        """
        Build the user prompt from llm_prompt.yaml template and runtime data.
        Uses the task-specified schema: {stage1_passed}, {signal_type}, {indicators_json}.
        """
        tmpl = self.prompts["user_prompt_template"]
        stage1_passed = self._build_stage1_pass_details(candidates)

        # Primary signal type (first candidate determines direction)
        signal_type = candidates[0].signal_type.value if candidates else "UNKNOWN"

        # Append filter warning note if near-boundary conditions exist
        warning_note = ""
        if filter_warning:
            warning_note = (
                "\n\n⚠️  FILTER WARNING: This candidate is near a threshold boundary "
                "(within 10%). Reduce confidence accordingly."
            )

        return tmpl.format(
            stage1_passed=stage1_passed + warning_note,
            signal_type=signal_type,
            indicators_json=self._build_indicators_json(indicators),
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

    def process_candidate(
        self,
        ticker: str,
        indicators: dict,
        stage1_pass_details: dict,
    ) -> dict:
        """
        Primary Stage 2 entry point called by HybridSignalPipeline.

        Args:
            ticker: Stock ticker symbol.
            indicators: Dict of full technical indicators.
            stage1_pass_details: Dict containing:
                - candidates: List[CandidateSignal] that passed Stage 1
                - filter_warning: bool — True if any filter was near-boundary

        Returns:
            Stage 2 result dict with final signal decision, or NO_TRADE on failure.
            Never raises — graceful degradation always returns a valid dict.
        """
        candidates = stage1_pass_details.get("candidates", [])
        filter_warning = stage1_pass_details.get("filter_warning", False)

        self.logger.info(
            "Stage 2 LLM called",
            ticker=ticker,
            candidate_count=len(candidates),
            signal_types=[c.signal_type.value for c in candidates],
            filter_warning=filter_warning,
        )

        # Stage 1 fail: log NO_CANDIDATE, no LLM call
        if not candidates:
            self.logger.info(
                "Stage 1 failed — logging NO_CANDIDATE, no LLM call",
                ticker=ticker,
            )
            return {
                "signal_decision": "NO_CANDIDATE",
                "reason": "Stage 1 filter failed — no candidate",
                "stage1_pass_details": stage1_pass_details,
                "llm_raw": None,
                "filter_warning": filter_warning,
            }

        # Build prompts from llm_prompt.yaml
        system_prompt = self.prompts["system_prompt"]
        user_prompt = self._build_user_prompt(ticker, indicators, candidates, filter_warning)

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
                "stage1_pass_details": {
                    "candidates": [
                        {"signal_type": c.signal_type.value, "confidence": c.confidence.value}
                        for c in candidates
                    ],
                    "filter_warning": filter_warning,
                },
                "llm_raw": None,
                "filter_warning": filter_warning,
            }

        # Parse and return
        result = self._parse_llm_output(raw, price, atr)
        result["stage1_pass_details"] = {
            "candidates": [
                {"signal_type": c.signal_type.value, "confidence": c.confidence.value}
                for c in candidates
            ],
            "filter_warning": filter_warning,
        }
        result["llm_raw"] = raw
        result["filter_warning"] = filter_warning

        self.logger.info(
            "Stage 2 LLM decision",
            ticker=ticker,
            signal_decision=result["signal_decision"],
            confidence=result["confidence"],
            filter_warning=filter_warning,
        )
        return result

    def process_candidates(
        self,
        ticker: str,
        indicators: dict,
        candidates: List[CandidateSignal],
        filter_warning: bool = False,
    ) -> dict:
        """
        Process all Stage 1 candidates for a ticker with one LLM call.
        (Maintained for backward compatibility.)

        Args:
            ticker: Stock ticker symbol.
            indicators: Dict of technical indicators.
            candidates: List of CandidateSignal objects that passed Stage 1.
            filter_warning: True if any filter was near-boundary.

        Returns:
            Stage 2 result dict with final signal decision, or NO_TRADE on failure.
        """
        stage1_pass_details = {
            "candidates": candidates,
            "filter_warning": filter_warning,
        }
        return self.process_candidate(ticker, indicators, stage1_pass_details)


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

        # Stage 1: collect ALL passing candidates + filter_warning
        stage1_candidates, filter_warning = self.stage1.evaluate_all_with_warning(indicators, earnings_date)

        stage1_pass_details = {
            "candidates": stage1_candidates,
            "filter_warning": filter_warning,
        }

        # If no candidates, return early with NO_CANDIDATE
        if not stage1_candidates:
            single_result, _ = self.stage1.evaluate_with_warning(indicators, earnings_date)
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
                    "filter_warning": filter_warning,
                },
                "stage2": {
                    "skipped": True,
                    "reason": "No Stage 1 candidate",
                    "signal_decision": "NO_CANDIDATE",
                },
                "final_signal": None,
                "indicators_summary": self._summarize_indicators(indicators)
            }

        # Stage 2: one LLM call with ALL candidates
        stage2_result = self.stage2.process_candidate(ticker, indicators, stage1_pass_details)

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
                ],
                "filter_warning": filter_warning,
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
                "stage1_pass_details": stage2_result.get("stage1_pass_details"),
                "filter_warning": stage2_result.get("filter_warning", filter_warning),
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

    def generate_signals_for_ticker(self, ticker: str, indicators: Dict[str, Any]) -> TickerSignals:
        """
        Generate all applicable signals for a ticker.
        """
        self.logger.info("Generating signals", ticker=ticker, indicators_count=len(indicators))

        current_price = indicators.get('Current_Price')
        earnings_date = indicators.get('Earnings_Date')

        stage1_candidates = self.stage1.evaluate_all(indicators, earnings_date)

        signals = []
        for candidate in stage1_candidates:
            sig = Signal(
                signal_type=candidate.signal_type,
                confidence=candidate.confidence,
                reasoning=candidate.reasons,
                current_price=current_price,
                expiry="2 weeks",
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
