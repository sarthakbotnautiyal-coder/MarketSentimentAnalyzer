#!/usr/bin/env python3
"""
Options Selling Signal Generator.

Reads indicators from market_data.db, applies CSP/CC strategy rules from
MT-2026-049 (Solanki), and outputs ranked sell signals for all 34 tickers.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    ticker: str
    direction: str             # "CSP" or "CC"
    strength: str               # "STRONG" or "MODERATE" or "SKIP"
    iv_rank: float
    iv_percentile: float
    rsi: float
    bband_position: float       # 0.0=lower band, 1.0=upper band
    macd_hist: float
    days_to_earnings: int
    atr_14: float
    score: int                  # 1–7
    entry_price: float
    suggested_strike: float
    suggested_expiry: str        # YYYY-MM-DD (T+21)
    premium_estimate: float
    risk_amount: float
    contracts_recommended: int
    skip_reason: Optional[str]
    timestamp: str

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class OptionSignalGenerator:
    """Applies CSP/CC strategy rules to indicator data stored in SQLite."""

    def __init__(
        self,
        db_path: str,
        tickers_path: str,
        portfolio_size: float = 50_000.0,
        max_risk_pct: float = 0.02,
    ) -> None:
        self.db_path = db_path
        self.tickers_path = tickers_path
        self.portfolio_size = portfolio_size
        self.max_risk_pct = max_risk_pct
        self._tickers: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_tickers(self) -> list[str]:
        """Load ticker list from JSON config."""
        with open(self.tickers_path) as f:
            raw = json.load(f)
        # Handle nested {"tickers": [...]} or bare [...]
        if isinstance(raw, dict):
            raw = raw.get("tickers", raw.get("tickers_list", []))
        return raw

    def get_latest_indicators(self, ticker: str) -> Optional[dict]:
        """Fetch the most recent indicator row for a ticker."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT * FROM indicators
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (ticker,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def calc_bband_position(price: float, bb_upper: float, bb_lower: float) -> float:
        """Position within Bollinger Band range: 0.0 = lower band, 1.0 = upper band."""
        band_range = bb_upper - bb_lower
        if band_range == 0:
            return 0.5
        pos = (price - bb_lower) / band_range
        return max(0.0, min(1.0, pos))

    @staticmethod
    def days_until_earnings(earnings_date_str: str | None) -> int:
        """Calendar days from today to the next earnings date."""
        if not earnings_date_str:
            return 999
        try:
            ed = date.fromisoformat(earnings_date_str)
            return (ed - date.today()).days
        except (ValueError, TypeError):
            return 999

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_ticker(self, row: dict, ticker: str) -> list[Signal]:
        """
        Evaluate all strategy rules and return 0–2 Signal objects (CSP + CC).

        Priority order:
          1. Earnings filter  (≤28 days → SKIP)
          2. IV Rank filter   (<20 → SKIP)
          3. CSP rules
          4. CC rules
        """
        today_str = date.today().isoformat()
        expiry = (date.today() + timedelta(days=21)).isoformat()

        price       = row.get("current_price") or 0.0
        sma_50      = row.get("sma50")         or 0.0
        sma_200     = row.get("sma200")        or 0.0
        rsi         = row.get("rsi")           or 50.0
        iv_rank     = row.get("iv_rank")       or 0.0
        iv_pct      = row.get("iv_percentile") or 0.0
        bb_upper    = row.get("bb_upper")      or 0.0
        bb_lower    = row.get("bb_lower")      or 0.0
        macd_hist   = row.get("macd_hist")     or 0.0
        vol_ratio   = row.get("vol_ratio")     or 0.0
        atr         = row.get("atr")           or 0.0
        earnings_str = row.get("next_earnings_date")

        days_earn = self.days_until_earnings(earnings_str)
        bb_pos = self.calc_bband_position(price, bb_upper, bb_lower)

        signals: list[Signal] = []

        # ---- Priority 1: Earnings ------------------------------------------------
        if days_earn <= 28:
            signals.append(
                Signal(
                    ticker=ticker, direction="CSP", strength="SKIP",
                    iv_rank=iv_rank, iv_percentile=iv_pct, rsi=rsi,
                    bband_position=bb_pos, macd_hist=macd_hist,
                    days_to_earnings=days_earn, atr_14=atr,
                    score=0, entry_price=price,
                    suggested_strike=0.0, suggested_expiry=expiry,
                    premium_estimate=0.0, risk_amount=0.0,
                    contracts_recommended=0,
                    skip_reason=f"Earnings in {days_earn}d",
                    timestamp=today_str,
                )
            )
            signals.append(
                Signal(
                    ticker=ticker, direction="CC", strength="SKIP",
                    iv_rank=iv_rank, iv_percentile=iv_pct, rsi=rsi,
                    bband_position=bb_pos, macd_hist=macd_hist,
                    days_to_earnings=days_earn, atr_14=atr,
                    score=0, entry_price=price,
                    suggested_strike=0.0, suggested_expiry=expiry,
                    premium_estimate=0.0, risk_amount=0.0,
                    contracts_recommended=0,
                    skip_reason=f"Earnings in {days_earn}d",
                    timestamp=today_str,
                )
            )
            return signals

        # ---- Priority 2: IV Rank < 20 ------------------------------------------
        if iv_rank < 20:
            skip_reason = f"IV Rank {iv_rank:.1f}% < 20%"
            signals.append(
                Signal(
                    ticker=ticker, direction="CSP", strength="SKIP",
                    iv_rank=iv_rank, iv_percentile=iv_pct, rsi=rsi,
                    bband_position=bb_pos, macd_hist=macd_hist,
                    days_to_earnings=days_earn, atr_14=atr,
                    score=0, entry_price=price,
                    suggested_strike=0.0, suggested_expiry=expiry,
                    premium_estimate=0.0, risk_amount=0.0,
                    contracts_recommended=0,
                    skip_reason=skip_reason,
                    timestamp=today_str,
                )
            )
            signals.append(
                Signal(
                    ticker=ticker, direction="CC", strength="SKIP",
                    iv_rank=iv_rank, iv_percentile=iv_pct, rsi=rsi,
                    bband_position=bb_pos, macd_hist=macd_hist,
                    days_to_earnings=days_earn, atr_14=atr,
                    score=0, entry_price=price,
                    suggested_strike=0.0, suggested_expiry=expiry,
                    premium_estimate=0.0, risk_amount=0.0,
                    contracts_recommended=0,
                    skip_reason=skip_reason,
                    timestamp=today_str,
                )
            )
            return signals

        # ---- CSP evaluation -------------------------------------------------------
        csp_signal = self._eval_csp(
            ticker, price, sma_50, sma_200, rsi, iv_rank, iv_pct,
            bb_pos, macd_hist, vol_ratio, days_earn, atr, today_str, expiry,
        )
        signals.append(csp_signal)

        # ---- CC evaluation --------------------------------------------------------
        cc_signal = self._eval_cc(
            ticker, price, sma_50, sma_200, rsi, iv_rank, iv_pct,
            bb_pos, macd_hist, vol_ratio, days_earn, atr, today_str, expiry,
        )
        signals.append(cc_signal)

        return signals

    def _eval_csp(
        self, ticker: str, price: float, sma_50: float, sma_200: float,
        rsi: float, iv_rank: float, iv_pct: float,
        bb_pos: float, macd_hist: float, vol_ratio: float,
        days_earn: int, atr: float, today_str: str, expiry: str,
    ) -> Signal:
        """
        CSP rules (all must be True):
          - price > sma_200   (uptrend intact)
          - price < sma_50    (at discount to short-term avg)
          - 30 <= rsi <= 50
          - iv_rank > 40
          - bband_position <= 0.3  (near lower band)
          - macd_hist <= 0         (not steepening positive)
          - vol_ratio > 0.7
        """
        pass_rules = (
            price > sma_200
            and price < sma_50
            and 30.0 <= rsi <= 50.0
            and iv_rank > 40.0
            and bb_pos <= 0.3
            and macd_hist <= 0.0
            and vol_ratio > 0.7
        )

        if not pass_rules:
            reason = self._csp_skip_reason(
                price, sma_50, sma_200, rsi, iv_rank, bb_pos, macd_hist, vol_ratio,
            )
            return self._build_signal(
                ticker=ticker, direction="CSP",
                strength="SKIP", score=0,
                iv_rank=iv_rank, iv_pct=iv_pct, rsi=rsi,
                bb_pos=bb_pos, macd_hist=macd_hist,
                days_earn=days_earn, atr=atr,
                price=price, expiry=expiry,
                skip_reason=reason, today_str=today_str,
            )

        # Scoring (1 point each)
        score = 0
        if iv_rank > 50:                         score += 1
        if 30.0 <= rsi <= 50.0:                  score += 1
        if bb_pos <= 0.3:                        score += 1
        if macd_hist <= 0.0:                     score += 1
        if price > sma_200 and price < sma_50:   score += 1
        if days_earn > 28:                       score += 1
        if vol_ratio > 0.7:                      score += 1

        strength = "STRONG" if score >= 5 else "MODERATE" if score >= 3 else "SKIP"

        return self._build_signal(
            ticker=ticker, direction="CSP",
            strength=strength, score=score,
            iv_rank=iv_rank, iv_pct=iv_pct, rsi=rsi,
            bb_pos=bb_pos, macd_hist=macd_hist,
            days_earn=days_earn, atr=atr,
            price=price, expiry=expiry,
            skip_reason=None, today_str=today_str,
        )

    def _eval_cc(
        self, ticker: str, price: float, sma_50: float, sma_200: float,
        rsi: float, iv_rank: float, iv_pct: float,
        bb_pos: float, macd_hist: float, vol_ratio: float,
        days_earn: int, atr: float, today_str: str, expiry: str,
    ) -> Signal:
        """
        CC rules (all must be True):
          - price > sma_50 AND price > sma_200
          - rsi > 55
          - iv_rank > 50
          - bband_position >= 0.7  (near upper band)
          - macd_hist >= 0         (positive but flattening)
          - vol_ratio > 0.6
        """
        pass_rules = (
            price > sma_50
            and price > sma_200
            and rsi > 55.0
            and iv_rank > 50.0
            and bb_pos >= 0.7
            and macd_hist >= 0.0
            and vol_ratio > 0.6
        )

        if not pass_rules:
            reason = self._cc_skip_reason(
                price, sma_50, sma_200, rsi, iv_rank, bb_pos, macd_hist, vol_ratio,
            )
            return self._build_signal(
                ticker=ticker, direction="CC",
                strength="SKIP", score=0,
                iv_rank=iv_rank, iv_pct=iv_pct, rsi=rsi,
                bb_pos=bb_pos, macd_hist=macd_hist,
                days_earn=days_earn, atr=atr,
                price=price, expiry=expiry,
                skip_reason=reason, today_str=today_str,
            )

        # Scoring (1 point each)
        score = 0
        if iv_rank > 50:                          score += 1
        if rsi > 55.0:                            score += 1
        if bb_pos >= 0.7:                         score += 1
        if macd_hist >= 0.0:                      score += 1
        if price > sma_50 and price > sma_200:    score += 1
        if days_earn > 28:                        score += 1
        if vol_ratio > 0.6:                       score += 1

        strength = "STRONG" if score >= 5 else "MODERATE" if score >= 3 else "SKIP"

        return self._build_signal(
            ticker=ticker, direction="CC",
            strength=strength, score=score,
            iv_rank=iv_rank, iv_pct=iv_pct, rsi=rsi,
            bb_pos=bb_pos, macd_hist=macd_hist,
            days_earn=days_earn, atr=atr,
            price=price, expiry=expiry,
            skip_reason=None, today_str=today_str,
        )

    # ------------------------------------------------------------------
    # Skip reasons (human-readable)
    # ------------------------------------------------------------------

    @staticmethod
    def _csp_skip_reason(
        price, sma_50, sma_200, rsi, iv_rank, bb_pos, macd_hist, vol_ratio,
    ) -> str:
        reasons = []
        if not (price > sma_200):
            reasons.append(f"price ≤ SMA200 ({price:.2f} ≤ {sma_200:.2f})")
        if not (price < sma_50):
            reasons.append(f"price ≥ SMA50 ({price:.2f} ≥ {sma_50:.2f})")
        if not (30.0 <= rsi <= 50.0):
            reasons.append(f"RSI {rsi:.1f} not in [30–50]")
        if not (iv_rank > 40.0):
            reasons.append(f"IV Rank {iv_rank:.1f}% ≤ 40%")
        if not (bb_pos <= 0.3):
            reasons.append(f"BBand pos {bb_pos:.2f} > 0.3")
        if not (macd_hist <= 0.0):
            reasons.append(f"MACD hist {macd_hist:.4f} > 0 (positive)")
        if not (vol_ratio > 0.7):
            reasons.append(f"Vol ratio {vol_ratio:.2f} ≤ 0.7")
        return "; ".join(reasons) if reasons else "CSP rules not met"

    @staticmethod
    def _cc_skip_reason(
        price, sma_50, sma_200, rsi, iv_rank, bb_pos, macd_hist, vol_ratio,
    ) -> str:
        reasons = []
        if not (price > sma_50):
            reasons.append(f"price ≤ SMA50 ({price:.2f} ≤ {sma_50:.2f})")
        if not (price > sma_200):
            reasons.append(f"price ≤ SMA200 ({price:.2f} ≤ {sma_200:.2f})")
        if not (rsi > 55.0):
            reasons.append(f"RSI {rsi:.1f} ≤ 55")
        if not (iv_rank > 50.0):
            reasons.append(f"IV Rank {iv_rank:.1f}% ≤ 50%")
        if not (bb_pos >= 0.7):
            reasons.append(f"BBand pos {bb_pos:.2f} < 0.7")
        if not (macd_hist >= 0.0):
            reasons.append(f"MACD hist {macd_hist:.4f} < 0")
        if not (vol_ratio > 0.6):
            reasons.append(f"Vol ratio {vol_ratio:.2f} ≤ 0.6")
        return "; ".join(reasons) if reasons else "CC rules not met"

    # ------------------------------------------------------------------
    # Signal construction helpers
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        ticker: str,
        direction: str,
        strength: str,
        score: int,
        iv_rank: float,
        iv_pct: float,
        rsi: float,
        bb_pos: float,
        macd_hist: float,
        days_earn: int,
        atr: float,
        price: float,
        expiry: str,
        skip_reason: Optional[str],
        today_str: str,
    ) -> Signal:
        """Shared signal builder — handles strike, premium, contracts."""
        if skip_reason or strength == "SKIP":
            return Signal(
                ticker=ticker, direction=direction, strength=strength,
                iv_rank=iv_rank, iv_percentile=iv_pct, rsi=rsi,
                bband_position=bb_pos, macd_hist=macd_hist,
                days_to_earnings=days_earn, atr_14=atr,
                score=score, entry_price=price,
                suggested_strike=0.0, suggested_expiry=expiry,
                premium_estimate=0.0, risk_amount=0.0,
                contracts_recommended=0,
                skip_reason=skip_reason or f"Rules not met ({strength})",
                timestamp=today_str,
            )

        # Strike selection
        if direction == "CSP":
            # OTM put — strike below current price by ~2%
            strike = round(price * 0.98, -1)
        else:
            # OTM call — strike above current price by ~2%
            strike = round(price * 1.02, -1)
        strike = round(strike / 5) * 5  # round to nearest $5

        # Premium estimate: rough annualised approximation
        premium_estimate = round((iv_rank / 100.0) * price * 0.3, 2)

        # Risk & contracts
        risk_amount = 2.0 * atr * 100.0
        max_risk_dollars = self.portfolio_size * self.max_risk_pct
        contracts = int(max_risk_dollars / risk_amount) if risk_amount > 0 else 0
        contracts = max(1, min(contracts, 5))

        return Signal(
            ticker=ticker, direction=direction, strength=strength,
            iv_rank=iv_rank, iv_percentile=iv_pct, rsi=rsi,
            bband_position=bb_pos, macd_hist=macd_hist,
            days_to_earnings=days_earn, atr_14=atr,
            score=score, entry_price=price,
            suggested_strike=strike, suggested_expiry=expiry,
            premium_estimate=premium_estimate, risk_amount=risk_amount,
            contracts_recommended=contracts,
            skip_reason=None,
            timestamp=today_str,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(self) -> list[Signal]:
        """Scan all tickers, return a flat list of Signals (CSP + CC each)."""
        tickers = self._load_tickers()
        all_signals: list[Signal] = []

        for ticker in tickers:
            row = self.get_latest_indicators(ticker)
            if row is None:
                # Emit two SKIP signals for missing tickers
                today_str = date.today().isoformat()
                expiry = (date.today() + timedelta(days=21)).isoformat()
                all_signals.extend([
                    Signal(
                        ticker=ticker, direction=d, strength="SKIP",
                        iv_rank=0.0, iv_percentile=0.0, rsi=0.0,
                        bband_position=0.0, macd_hist=0.0,
                        days_to_earnings=999, atr_14=0.0,
                        score=0, entry_price=0.0,
                        suggested_strike=0.0, suggested_expiry=expiry,
                        premium_estimate=0.0, risk_amount=0.0,
                        contracts_recommended=0,
                        skip_reason="No data in DB",
                        timestamp=today_str,
                    )
                    for d in ("CSP", "CC")
                ])
                continue

            signals = self.score_ticker(row, ticker)
            all_signals.extend(signals)

        return all_signals

    @staticmethod
    def save_json(signals: list[Signal], output_path: str) -> None:
        """Serialise Signal list to a JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump([s.to_dict() for s in signals], f, indent=2)
        print(f"✅  Saved {len(signals)} signals → {out}")

    @staticmethod
    def print_table(signals: list[Signal]) -> None:
        """Print a readable signal table to stdout."""
        print("\n" + "=" * 130)
        print(f"{'Ticker':<8} {'Dir':<5} {'Strength':<10} {'Score':<6} {'IV Rank':<9} {'RSI':<6} "
              f"{'BB Pos':<8} {'MACD H':<9} {'d→Earn':<8} {'Price':<8} {'Strike':<8} "
              f"{'Expiry':<12} {'Premium':<9} {'Risk':<8} {'Contracts':<10} {'Skip Reason'}")
        print("-" * 130)

        for s in signals:
            if s.strength == "SKIP":
                print(
                    f"{s.ticker:<8} {s.direction:<5} {s.strength:<10} {s.score:<6} "
                    f"{s.iv_rank:<9.1f} {s.rsi:<6.1f} {s.bband_position:<8.2f} "
                    f"{s.macd_hist:<9.4f} {s.days_to_earnings:<8} {s.entry_price:<8.2f} "
                    f"{'—':<8} {'—':<12} {'—':<9} {'—':<8} {'—':<10} {s.skip_reason or ''}"
                )
            else:
                print(
                    f"{s.ticker:<8} {s.direction:<5} {s.strength:<10} {s.score:<6} "
                    f"{s.iv_rank:<9.1f} {s.rsi:<6.1f} {s.bband_position:<8.2f} "
                    f"{s.macd_hist:<9.4f} {s.days_to_earnings:<8} {s.entry_price:<8.2f} "
                    f"{s.suggested_strike:<8.2f} {s.suggested_expiry:<12} "
                    f"{s.premium_estimate:<9.2f} {s.risk_amount:<8.2f} "
                    f"{s.contracts_recommended:<10} {'—'}"
                )

        print("=" * 130)
        actionable = [s for s in signals if s.strength in ("STRONG", "MODERATE")]
        print(f"\n📊  Total signals: {len(signals)}  |  Actionable: {len(actionable)}  "
              f"|  SKIP: {len(signals) - len(actionable)}")
        csp_actionable = [s for s in actionable if s.direction == "CSP"]
        cc_actionable = [s for s in actionable if s.direction == "CC"]
        print(f"   CSP actionable: {len(csp_actionable)}  |  CC actionable: {len(cc_actionable)}")


# ---------------------------------------------------------------------------
# CLI entry point (also importable via `python -m src.signals`)
# ---------------------------------------------------------------------------

def run_signals() -> None:
    """Run the signal generator and print results."""
    base = Path(__file__).parent.parent
    db_path = base / "data" / "market_data.db"
    tickers_path = base / "data" / "tickers.json"
    today = date.today().isoformat()
    output_path = base / "data" / f"option_signals_{today}.json"

    gen = OptionSignalGenerator(
        db_path=str(db_path),
        tickers_path=str(tickers_path),
        portfolio_size=50_000.0,
        max_risk_pct=0.02,
    )

    print(f"🔍  Scanning {gen._load_tickers().__len__()} tickers …")
    signals = gen.generate_signals()
    gen.print_table(signals)
    gen.save_json(signals, str(output_path))


if __name__ == "__main__":
    run_signals()
