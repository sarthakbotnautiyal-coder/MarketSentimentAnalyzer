"""Options signal engine for generating trading recommendations."""

from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import structlog

logger = structlog.get_logger()


class SignalType(Enum):
    """Types of options signals."""
    SELL_PUTS = "SELL_PUTS"
    SELL_CALLS = "SELL_CALLS"
    BUY_LEAPS = "BUY_LEAPS"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"


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
            expiry建议="2 weeks",
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
            expiry建议="2 weeks",
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
        from datetime import datetime
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