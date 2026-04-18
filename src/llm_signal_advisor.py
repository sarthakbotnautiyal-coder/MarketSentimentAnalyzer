"""
LLM-powered signal advisor for options trading decisions.

Uses a local Ollama model to evaluate technical indicators and produce
structured signal decisions with confidence and reasoning.

Pydantic schema ensures uniform output regardless of LLM provider.
"""

import json
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger(__name__)


# ── Output Schema ────────────────────────────────────────────────────────────


class LLMSignalAdvice(BaseModel):
    """
    Structured output from the LLM signal advisor.

    All field values are flat strings or simple arrays — no nested dicts.
    Strict mode disabled for LLM tolerance; extra fields stripped silently.

    Optional fields default to "" or None so missing LLM output doesn't
    cause validation errors (e.g. NO_TRADE outputs often omit strike/expiry/stop).
    """

    signal_decision: str = Field(
        default="NO_TRADE",
        description="SELL_PUTS | SELL_CALLS | NO_TRADE",
    )
    confidence: str = Field(
        default="LOW",
        description="HIGH | MEDIUM | LOW",
    )
    top_3_reasons: list[str] = Field(
        default_factory=list,
        description="Up to 3 short reasons (under 10 words each)",
    )
    strike_recommendation: Optional[str] = Field(
        default="",
        description="e.g. $78.50 or ATM / -2% OTM",
    )
    expiry_recommendation: Optional[str] = Field(
        default="",
        description="e.g. 14 DTE or next expiration",
    )
    stop_loss: Optional[str] = Field(
        default="",
        description="e.g. $76.00 (-2.5%) or 1.5×ATR",
    )
    premium_estimate: Optional[str] = Field(
        default=None,
        description="Approximate premium per contract",
    )
    risk_flags: Optional[list[str]] = Field(
        default=None,
        description="e.g. earnings within 30 days, low liquidity",
    )
    llm_notes: Optional[str] = Field(
        default=None,
        description="Any additional observations from the LLM",
    )


# ── Prompt Builder ────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are an expert options trading analyst.

You will receive a JSON payload containing full technical indicators for a single stock ticker. Your job is to decide whether this ticker is a GOOD CANDIDATE for either SELL_PUTS or SELL_CALLS based on the current market conditions.

## Your Task

Evaluate the indicators carefully and output a STRICT JSON object ONLY — no markdown fences, no prose, no explanation outside the JSON.

## Decision Criteria

**SELL_PUTS candidates:**
- RSI < 35 strongly preferred (oversold)
- MACD bullish crossover (MACD just crossed above signal) confirms reversal
- Price near lower Bollinger Band (bounce setup)
- High IV Rank (>30) = good premium
- Price above SMA20 = short-term uptrend confirmation
- Volume above average

**SELL_CALLS candidates:**
- RSI > 65 strongly preferred (overbought)
- MACD bearish crossover (MACD just crossed below signal) confirms reversal
- Price near upper Bollinger Band (reversal setup)
- High IV Rank (>30) = good premium
- Price below SMA20 = short-term downtrend confirmation
- Volume above average

## Confidence Levels

- **HIGH**: RSI and MACD both strongly aligned + IV Rank > 40 + price near target band
- **MEDIUM**: At least 3 conditions met cleanly
- **LOW**: Borderline — only just clearing thresholds

## Important Rules

1. Output ONLY a raw JSON object starting with `{` and ending with `}`
2. All field values must be flat strings or simple arrays — NEVER nested dicts
3. `top_3_reasons`: each reason must be under 10 words
4. `strike_recommendation`: use format like "$78.50" or "ATM" or "-5% OTM". If NO_TRADE, omit this field or use empty string.
5. `expiry_recommendation`: use format like "14 DTE" or "next expiration". If NO_TRADE, omit or use empty string.
6. `stop_loss`: use ATR multiplier if available, e.g. "$76.00 (-2.5%)" or "$76.00 (1.5×ATR)". If NO_TRADE, omit or use empty string.
7. If conditions are poor for both SELL_PUTS and SELL_CALLS → output NO_TRADE
8. `premium_estimate` and `risk_flags` are optional — omit if not confident
9. Never output null for any field — use empty string "" instead of null"""


def _build_user_prompt(ticker: str, indicators: dict, stage1_passed: bool) -> str:
    """Build the user prompt with all indicator values."""
    key_fields = {
        "Current_Price": indicators.get("Current_Price"),
        "RSI_14": indicators.get("RSI_14"),
        "IV_Rank": indicators.get("IV_Rank"),
        "IV_Percentile": indicators.get("IV_Percentile"),
        "Implied_Volatility": indicators.get("Implied_Volatility"),
        "MACD": indicators.get("MACD"),
        "MACD_Signal": indicators.get("MACD_Signal"),
        "MACD_Hist": indicators.get("MACD_Hist"),
        "MACD_Prev": indicators.get("MACD_Prev"),
        "MACD_Signal_Prev": indicators.get("MACD_Signal_Prev"),
        "SMA_20": indicators.get("SMA_20"),
        "SMA_50": indicators.get("SMA_50"),
        "SMA_200": indicators.get("SMA_200"),
        "BB_Upper": indicators.get("BB_Upper"),
        "BB_Middle": indicators.get("BB_Middle"),
        "BB_Lower": indicators.get("BB_Lower"),
        "ATR_14": indicators.get("ATR_14"),
        "Volume_Ratio": indicators.get("Volume_Ratio"),
        "Volume_10d_Avg": indicators.get("Volume_10d_Avg"),
        "Historical_Volatility_20d": indicators.get("Historical_Volatility_20d"),
        "Historical_Volatility_30d": indicators.get("Historical_Volatility_30d"),
        "High_20d": indicators.get("High_20d"),
        "Low_20d": indicators.get("Low_20d"),
        "EMA_5": indicators.get("EMA_5"),
        "Next_Earnings_Date": indicators.get("Next_Earnings_Date"),
        "Change": indicators.get("Change"),
    }

    # Filter out None values for cleaner prompt
    key_fields = {k: v for k, v in key_fields.items() if v is not None}

    indicators_json = json.dumps(key_fields, indent=2)
    stage1_status = "PASSED" if stage1_passed else "NOT PASSED"

    return f"""Ticker: {ticker}
Stage 1 Filter Status: {stage1_status}

Technical Indicators:
{indicators_json}

Output JSON only."""


# ── Advisor Class ─────────────────────────────────────────────────────────────


class LLMSignalAdvisor:
    """
    LLM-powered signal advisor.

    Uses the configured Ollama model (from config/llm_config.yaml) to
    evaluate ticker indicators and produce structured signal decisions.

    Falls back gracefully if Ollama is unavailable — emits NO_TRADE with
    low confidence rather than crashing.
    """

    def __init__(self, llm_client=None):
        """
        Initialise the advisor.

        Args:
            llm_client: Optional pre-initialized LLM client instance.
                        If None, creates one from config.
        """
        self.logger = logger.bind(component="LLMSignalAdvisor")

        if llm_client is None:
            from src.llm_client import LLMClient
            self.llm = LLMClient()
        else:
            self.llm = llm_client

        self._schema = self._pydantic_to_schema(LLMSignalAdvice)

    @staticmethod
    def _pydantic_to_schema(model_cls) -> dict:
        """Convert a Pydantic model to a JSON schema dict for prompt injection."""
        return {
            "type": "object",
            "properties": {
                "signal_decision": {
                    "type": "string",
                    "enum": ["SELL_PUTS", "SELL_CALLS", "NO_TRADE"]
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"]
                },
                "top_3_reasons": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "strike_recommendation": {"type": "string"},
                "expiry_recommendation": {"type": "string"},
                "stop_loss": {"type": "string"},
                "premium_estimate": {"type": "string"},
                "risk_flags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "llm_notes": {"type": "string"},
            },
            "required": ["signal_decision", "confidence", "top_3_reasons"]
        }

    @staticmethod
    def _sanitise_raw(raw: dict) -> dict:
        """
        Sanitise raw LLM output before Pydantic validation.

        Handles:
        - null values → replaced with "" for string fields
        - Missing optional fields → filled with ""
        - Nested dicts in string fields → flattened to string
        """
        out = dict(raw)

        # Null → "" for all string fields
        for field in ["strike_recommendation", "expiry_recommendation",
                      "stop_loss", "premium_estimate", "llm_notes"]:
            val = out.get(field)
            if val is None:
                out[field] = ""
            elif isinstance(val, (dict, list)):
                out[field] = str(val)

        # top_3_reasons: ensure it's a list
        top3 = out.get("top_3_reasons")
        if top3 is None:
            out["top_3_reasons"] = []
        elif isinstance(top3, str):
            # Split by newlines or bullet points
            parts = [p.strip() for p in top3.split("\n") if p.strip()]
            out["top_3_reasons"] = parts[:3] if parts else []

        # risk_flags: ensure it's a list
        rf = out.get("risk_flags")
        if rf is None:
            out["risk_flags"] = None
        elif isinstance(rf, str):
            out["risk_flags"] = [rf]

        return out

    def advise(self, ticker: str, indicators: dict, stage1_passed: bool = True) -> LLMSignalAdvice:
        """
        Get LLM signal advice for a single ticker.

        Args:
            ticker: Stock ticker symbol
            indicators: Full indicators dict from fetcher
            stage1_passed: Whether Stage 1 hard filters passed

        Returns:
            LLMSignalAdvice object with decision, confidence, and reasoning
        """
        user_prompt = _build_user_prompt(ticker, indicators, stage1_passed)

        self.logger.info(
            "Requesting LLM signal advice",
            ticker=ticker,
            stage1_passed=stage1_passed,
            indicator_count=len([k for k, v in indicators.items() if v is not None]),
        )

        raw = self.llm.generate_json(
            prompt=user_prompt,
            system_prompt=SYSTEM_PROMPT,
            schema=self._schema,
        )

        if raw is None:
            self.logger.warning("LLM returned no output — emitting NO_TRADE", ticker=ticker)
            return LLMSignalAdvice(
                signal_decision="NO_TRADE",
                confidence="LOW",
                top_3_reasons=["LLM unavailable or returned no response"],
            )

        # Sanitise nulls and missing fields before validation
        sanitised = self._sanitise_raw(raw)

        try:
            advice = LLMSignalAdvice.model_validate(sanitised, strict=False)
            self.logger.info(
                "LLM advice received",
                ticker=ticker,
                decision=advice.signal_decision,
                confidence=advice.confidence,
            )
            return advice
        except Exception as exc:
            self.logger.warning(
                "Schema validation failed on LLM output — emitting NO_TRADE",
                ticker=ticker,
                error=str(exc),
                raw_preview=str(raw)[:200],
            )
            return LLMSignalAdvice(
                signal_decision="NO_TRADE",
                confidence="LOW",
                top_3_reasons=[f"LLM output validation failed: {str(exc)[:50]}"],
            )


# ── Convenience function ──────────────────────────────────────────────────────


def get_advice(ticker: str, indicators: dict, stage1_passed: bool = True) -> LLMSignalAdvice:
    """
    Convenience wrapper around LLMSignalAdvisor.advise().

    Creates a transient advisor and returns advice for one ticker.
    For batch usage, create an LLMSignalAdvisor instance once and reuse it.
    """
    advisor = LLMSignalAdvisor()
    return advisor.advise(ticker, indicators, stage1_passed)