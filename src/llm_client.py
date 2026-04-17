"""LLM client for Stage 2 signal generation via Ollama (gemma4:e4b)."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

# Use structlog so .bind() is available in all environments
import structlog

_log = structlog.get_logger(__name__)


class LLMClient:
    """
    LLM client backed by Ollama /api/generate.

    Supports plain text generation and JSON-mode generation with schema
    enforcement and retry-on-parse-failure.
    """

    def __init__(
        self,
        config_path: str = "config/llm_config.yaml",
        logger=None,
    ):
        """
        Load LLM config and initialise the Ollama session.

        Args:
            config_path: Path to llm_config.yaml (relative to repo root).
            logger: Optional bound structlog logger. If omitted, uses module logger.
        """
        self.logger = logger or _log.bind(component="LLMClient")

        config_file = Path(__file__).parent.parent / config_path
        with open(config_file) as f:
            self.cfg = yaml.safe_load(f)

        self.model = self.cfg["model"]
        self.provider = self.cfg["provider"]
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.timeout = self.cfg.get("timeout_seconds", 120)
        self.retry_attempts = self.cfg.get("retry_attempts", 1)
        self.json_mode = self.cfg.get("json_mode", True)

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _post(self, payload: dict) -> Optional[dict]:
        """
        POST to the Ollama /api/generate endpoint.

        Returns parsed JSON dict on success, None on timeout or HTTP error.
        """
        url = f"{self.base_url}/api/generate"
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            self.logger.warning("Ollama request timed out", url=url, timeout=self.timeout)
            return None
        except requests.exceptions.ConnectionError:
            self.logger.warning("Ollama connection failed — is Ollama running?", url=url)
            return None
        except requests.exceptions.HTTPError as exc:
            self.logger.error("Ollama HTTP error", status=exc.response.status_code, body=exc.response.text[:500])
            return None
        except Exception as exc:
            self.logger.error("Ollama unexpected error", error=str(exc))
            return None

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """
        Call the LLM and return raw text response.

        Args:
            prompt: User prompt string.
            system_prompt: Optional system-level instruction.

        Returns:
            Generated text string, or None on failure.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        result = self._post(payload)
        if result is None:
            return None

        return result.get("response", "").strip() or None

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        schema: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Call the LLM enforcing JSON output, retrying once on parse failure.

        Args:
            prompt: User prompt string.
            system_prompt: Optional system-level instruction.
            schema: Optional JSON schema hint to embed in the prompt.

        Returns:
            Parsed JSON dict, or None on failure.
        """
        # Build JSON-enforcing system prompt
        json_instruction = (
            "You MUST respond with valid JSON only — no markdown fences, "
            "no explanation outside the JSON object."
        )
        if schema:
            json_instruction += f"\n\nOutput schema:\n{json.dumps(schema, indent=2)}"

        full_system = (system_prompt + "\n\n" + json_instruction).strip()

        for attempt in range(self.retry_attempts + 1):
            raw = self.generate(prompt, full_system)
            if raw is None:
                return None

            try:
                # Strip markdown fences if model wraps in ```json ... ```
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    lines = cleaned.splitlines()
                    cleaned = "\n".join(lines[1:])  # drop first fence line
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                return json.loads(cleaned.strip())
            except json.JSONDecodeError as exc:
                self.logger.warning(
                    "LLM JSON parse failed",
                    attempt=attempt + 1,
                    raw_preview=raw[:200],
                    error=str(exc),
                )
                if attempt == self.retry_attempts:
                    self.logger.error("All JSON parse retries exhausted")
                    return None
                # Retry with stricter instruction on next attempt
                full_system = (
                    "IMPORTANT: You must output ONLY a raw JSON object. "
                    "No markdown, no text before or after. Start with { and end with }.\n\n"
                    + system_prompt
                    + "\n\n"
                    + json_instruction
                )

        return None
