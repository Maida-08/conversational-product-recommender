"""
Provider-agnostic LLM client.

Talks to any OpenAI-compatible endpoint (Groq by default). Swapping providers means
changing LLM_BASE_URL / LLM_MODEL / the API key in .env, not code.

Free-tier protections:
  - token-per-minute throttle (sliding 60s window) so we stay under Groq's 8K TPM
  - on 429: honour retry-after if short, otherwise switch to the fallback model
  - never sleeps longer than llm_max_retry_wait_s (a daily-quota 429 can say "wait hours")
  - callers get a single LLMError and fall back to deterministic rules
"""
import json
import re
import time
from collections import deque
from typing import Any, Optional

import openai
from openai import OpenAI

from app.config import Settings, get_settings


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a usable answer. Callers should fall back to rules."""


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and stray prose."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    return json.loads(text[start:end + 1])


def _retry_after_seconds(err: openai.RateLimitError) -> Optional[float]:
    try:
        value = err.response.headers.get("retry-after")
        return float(value) if value is not None else None
    except Exception:
        return None


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        if not self.settings.llm_enabled:
            raise LLMError("GROQ_API_KEY is not set")
        self._client = OpenAI(
            base_url=self.settings.llm_base_url,
            api_key=self.settings.groq_api_key,
            max_retries=0,   # we do our own retry/fallback logic
            timeout=60,
        )
        self._window: deque[tuple[float, int]] = deque()   # (timestamp, tokens)
        self._send_reasoning_effort = True

    # ------------------------------------------------------------ throttling
    def _tokens_used_last_minute(self) -> int:
        now = time.time()
        while self._window and now - self._window[0][0] > 60:
            self._window.popleft()
        return sum(t for _, t in self._window)

    def _throttle(self, estimated_tokens: int) -> None:
        budget = self.settings.llm_tokens_per_minute_budget
        while self._tokens_used_last_minute() + estimated_tokens > budget and self._window:
            wait = 60 - (time.time() - self._window[0][0]) + 0.5
            time.sleep(max(wait, 0.5))

    # ------------------------------------------------------------------ call
    def _call(self, model: str, system: str, user: str, max_tokens: int):
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_completion_tokens=max_tokens,
        )
        if self._send_reasoning_effort and self.settings.llm_reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.settings.llm_reasoning_effort}
        return self._client.chat.completions.create(**kwargs)

    def complete_json(self, system: str, user: str, max_tokens: int = 3500) -> tuple[dict, dict]:
        """Return (parsed_json, meta). meta = {"model": str, "total_tokens": int}."""
        models = [self.settings.llm_model]
        if self.settings.llm_fallback_model and self.settings.llm_fallback_model != models[0]:
            models.append(self.settings.llm_fallback_model)

        estimate = int((len(system) + len(user)) / 3) + 1500
        last_error = "unknown error"

        for model in models:
            for attempt in range(2):
                self._throttle(estimate)
                try:
                    resp = self._call(model, system, user, max_tokens)
                    used = getattr(resp.usage, "total_tokens", None) or estimate
                    self._window.append((time.time(), int(used)))
                    content = resp.choices[0].message.content
                    return _extract_json(content), {"model": model, "total_tokens": int(used)}

                except openai.RateLimitError as e:
                    wait = _retry_after_seconds(e)
                    last_error = f"rate limited on {model}"
                    if wait is None or wait > self.settings.llm_max_retry_wait_s:
                        break   # long wait (likely daily quota) -> try the fallback model
                    time.sleep(wait + 1)

                except openai.BadRequestError as e:
                    # Some models/providers reject reasoning_effort; drop it once and retry.
                    if self._send_reasoning_effort and "reasoning" in str(e).lower():
                        self._send_reasoning_effort = False
                        continue
                    raise LLMError(f"bad request to {model}: {e}") from e

                except (openai.APIConnectionError, openai.APITimeoutError) as e:
                    last_error = f"connection problem on {model}: {e}"
                    time.sleep(2 * (attempt + 1))

                except openai.APIStatusError as e:
                    raise LLMError(f"API error from {model}: {e.status_code} {e}") from e

                except (ValueError, json.JSONDecodeError) as e:
                    last_error = f"unparseable JSON from {model}: {e}"

        raise LLMError(last_error)