"""Console display using tabulate."""

from typing import Dict, Any, List
from tabulate import tabulate
import structlog

logger = structlog.get_logger()


class Display:
    """Handle console output formatting."""

    @staticmethod
    def print_indicators(ticker: str, indicators: Dict[str, Any]) -> None:
        """Print indicators table for a ticker."""
        if "error" in indicators:
            print(f"\n{ticker}: {indicators['error']}")
            return

        print(f"\n{'='*60}")
        print(f"Ticker: {ticker}")
        print(f"{'='*60}")

        # Prepare indicators table
        table_data = []
        for key, value in indicators.items():
            if value is not None:
                if isinstance(value, float):
                    formatted = f"{value:.2f}"
                else:
                    formatted = str(value)
                table_data.append([key, formatted])

        if table_data:
            print(tabulate(table_data, headers=["Indicator", "Value"], tablefmt="grid"))
        else:
            print("No indicators available")

    @staticmethod
    def print_summary(results: Dict[str, Dict[str, Any]]) -> None:
        """Print summary across all tickers."""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        summary_data = []
        for ticker, data in results.items():
            indicators = data.get("indicators", {})

            # Extract key metrics
            price = indicators.get("Current_Price", "N/A")
            change = indicators.get("Change", "N/A")
            rsi = indicators.get("RSI_14", "N/A")

            summary_data.append([
                ticker,
                f"${price:.2f}" if isinstance(price, (int, float)) else price,
                f"{change:.2f}%" if isinstance(change, (int, float)) else change,
                f"{rsi:.1f}" if isinstance(rsi, (int, float)) else rsi,
            ])

        if summary_data:
            headers = ["Ticker", "Price", "Change", "RSI"]
            print(tabulate(summary_data, headers=headers, tablefmt="grid"))
