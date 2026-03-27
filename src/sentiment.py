"""Sentiment analysis using Ollama LLM with caching."""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import requests
import structlog

logger = structlog.get_logger()


class SentimentAnalyzer:
    """Analyze sentiment of news using Ollama LLM with caching."""

    VALID_SENTIMENTS = {"bullish", "bearish", "neutral"}

    def __init__(self, host: str = "http://localhost:11434", model: str = "qwen2.5:7b", db_manager: Any = None, sentiment_ttl_days: int = 7):
        """Initialize with Ollama host, model, optional database cache and TTL."""
        self.host = host.rstrip("/")
        self.model = model
        self.generate_url = f"{self.host}/api/generate"
        self.db = db_manager
        self.sentiment_ttl_days = sentiment_ttl_days

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

    def analyze_batch(self, texts: list[str], ticker: str = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze multiple news items and aggregate sentiment with caching.
        
        Args:
            texts: List of text summaries to analyze
            ticker: Stock ticker for cache key
            force_refresh: If True, bypass cache
            
        Returns:
            Aggregated sentiment analysis
        """
        if not ticker:
            ticker = "DEFAULT"  # For caching when no ticker specified

        # Check cache first if database available and not forcing refresh
        if self.db and not force_refresh:
            try:
                if self.db.is_data_fresh('sentiment', ticker, self.sentiment_ttl_days):
                    cached = self.db.get_cached_sentiment(ticker, self.sentiment_ttl_days)
                    if cached:
                        logger.info(f"Using cached sentiment for {ticker} from {cached.get('date')}")
                        return {
                            "sentiment": cached['sentiment'],
                            "confidence": cached['confidence'],
                            "explanation": cached['explanation']
                        }
            except Exception as e:
                logger.warning(f"Failed to read sentiment cache for {ticker}: {e}")

        if not texts:
            result = {"sentiment": "neutral", "confidence": 1.0, "explanation": "No news provided"}
            if self.db:
                try:
                    today = datetime.now().strftime('%Y-%m-%d')
                    self.db.save_sentiment(ticker, today, result['sentiment'], result['confidence'], result['explanation'])
                except Exception as e:
                    logger.error(f"Failed to save sentiment to database for {ticker}: {e}")
            return result

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

            analysis = {
                "sentiment": first_word,
                "confidence": 0.8,  # Default for batch
                "explanation": generated_text[:200] if len(generated_text) > 200 else generated_text
            }

            # Cache the result
            if self.db:
                try:
                    today = datetime.now().strftime('%Y-%m-%d')
                    self.db.save_sentiment(
                        ticker,
                        today,
                        analysis['sentiment'],
                        analysis['confidence'],
                        analysis['explanation']
                    )
                except Exception as e:
                    logger.error(f"Failed to save sentiment to database for {ticker}: {e}")

            return analysis
        except requests.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            fallback = {"sentiment": "neutral", "confidence": 0.0, "explanation": f"Connection error: {e}"}
            # Don't cache failures
            return fallback
        except Exception as e:
            logger.error(f"Unexpected error in sentiment analysis: {e}")
            fallback = {"sentiment": "neutral", "confidence": 0.0, "explanation": f"Unexpected error: {e}"}
            return fallback

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
