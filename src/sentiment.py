"""Sentiment analysis using Ollama LLM."""

import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment of news using Ollama LLM."""

    VALID_SENTIMENTS = {"bullish", "bearish", "neutral"}

    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        """Initialize with Ollama host and model."""
        self.host = host.rstrip("/")
        self.model = model
        self.generate_url = f"{self.host}/api/generate"

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of given text."""
        if not text or text.strip() == "":
            return {"sentiment": "neutral", "confidence": 1.0, "explanation": "Empty input"}

        prompt = self._build_prompt(text)

        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("response", "").strip()

            return self._parse_response(generated_text)
        except requests.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": f"Connection error: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error in sentiment analysis: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": f"Unexpected error: {e}"}

    def analyze_batch(self, texts: list[str]) -> Dict[str, Any]:
        """Analyze multiple news items and aggregate sentiment."""
        if not texts:
            return {"sentiment": "neutral", "confidence": 1.0, "explanation": "No news provided"}

        # Combine texts with separator
        combined = "\n\n---\n\n".join(texts[:5])  # Limit to 5 articles

        prompt = f"""Analyze the overall market sentiment based on these news summaries:

{combined}

What is the overall sentiment? Respond with exactly one word: bullish, bearish, or neutral, followed by a brief explanation (one paragraph)."""

        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("response", "").strip()

            # Try to extract sentiment from first word
            first_word = generated_text.split()[0].lower() if generated_text else "neutral"
            if first_word not in self.VALID_SENTIMENTS:
                first_word = "neutral"

            return {
                "sentiment": first_word,
                "confidence": 0.8,  # Default for batch
                "explanation": generated_text[:200] if len(generated_text) > 200 else generated_text
            }
        except requests.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": f"Connection error: {e}"}

    def _build_prompt(self, text: str) -> str:
        """Build prompt for single text analysis."""
        return f"""Analyze the sentiment of this financial news summary:

{text}

Respond with exactly one word (bullish, bearish, or neutral) followed by a brief explanation."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured sentiment."""
        if not response:
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": "Empty response"}

        words = response.lower().split()
        first_word = words[0] if words else ""

        if first_word in self.VALID_SENTIMENTS:
            sentiment = first_word
            explanation = " ".join(words[1:]) if len(words) > 1 else response
        else:
            # Try to find a sentiment word in the response
            for word in words:
                if word in self.VALID_SENTIMENTS:
                    sentiment = word
                    explanation = response
                    break
            else:
                sentiment = "neutral"
                explanation = response

        return {
            "sentiment": sentiment,
            "confidence": 0.9 if sentiment in words[:3] else 0.5,
            "explanation": explanation[:200]  # Limit length
        }
