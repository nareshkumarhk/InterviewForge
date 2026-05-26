from openai import OpenAI, RateLimitError, APIStatusError
import os
import time
from typing import Optional
from dataclasses import dataclass
import json


class AIClientError(Exception):
    """Raised when an AI API call fails permanently."""


class RateLimitExhausted(AIClientError):
    """Raised when rate limit persists after all retries. Carries current limit state."""

    def __init__(self, message: str, rate_limit: Optional["RateLimitInfo"] = None):
        super().__init__(message)
        self.rate_limit = rate_limit


@dataclass
class RateLimitInfo:
    limit_requests: Optional[int]
    remaining_requests: Optional[int]
    limit_tokens: Optional[int]
    remaining_tokens: Optional[int]
    reset_requests: Optional[str]   # e.g. "1s", "6m0s", "1h0m0s"
    reset_tokens: Optional[str]

    @classmethod
    def from_headers(cls, headers) -> "RateLimitInfo":
        def _int(key: str) -> Optional[int]:
            v = headers.get(key)
            return int(v) if v is not None else None

        return cls(
            limit_requests=_int("x-ratelimit-limit-requests"),
            remaining_requests=_int("x-ratelimit-remaining-requests"),
            limit_tokens=_int("x-ratelimit-limit-tokens"),
            remaining_tokens=_int("x-ratelimit-remaining-tokens"),
            reset_requests=headers.get("x-ratelimit-reset-requests"),
            reset_tokens=headers.get("x-ratelimit-reset-tokens"),
        )

    def __str__(self) -> str:
        return (
            f"Requests: {self.remaining_requests}/{self.limit_requests} remaining"
            f" (resets {self.reset_requests})"
            f" | Tokens: {self.remaining_tokens}/{self.limit_tokens} remaining"
            f" (resets {self.reset_tokens})"
        )


class AIClient:
    """OpenAI API client wrapper with retry logic, error handling, and rate limit visibility."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_retries = 5
        self.base_delay = 5
        self.last_rate_limit: Optional[RateLimitInfo] = None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        response_format: Optional[dict] = None,
    ) -> str:
        """Call OpenAI API with exponential backoff on rate limits."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs = {"model": self.model, "messages": messages, "temperature": temperature}
        if response_format:
            kwargs["response_format"] = response_format

        for attempt in range(self.max_retries):
            try:
                raw = self.client.chat.completions.with_raw_response.create(**kwargs)
                self.last_rate_limit = RateLimitInfo.from_headers(raw.headers)
                return raw.parse().choices[0].message.content
            except RateLimitError as e:
                self._handle_rate_limit(e, attempt)
            except APIStatusError as e:
                raise AIClientError(f"OpenAI API error {e.status_code}: {e.message}") from e
            except OSError as e:
                self._handle_transient_error(e, attempt)

        raise AIClientError(f"API call failed after {self.max_retries} attempts")

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> dict:
        """Call OpenAI API expecting a JSON response."""
        response = self.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise AIClientError(f"Failed to parse JSON response: {e}\nResponse: {response}") from e

    def check_rate_limits(self) -> RateLimitInfo:
        """Make a minimal API call (1 token) to fetch current rate limit state from headers."""
        raw = self.client.chat.completions.with_raw_response.create(
            model=self.model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            temperature=0,
        )
        self.last_rate_limit = RateLimitInfo.from_headers(raw.headers)
        return self.last_rate_limit

    def _handle_rate_limit(self, error: RateLimitError, attempt: int) -> None:
        # Print the raw API error body so the caller can see the exact code + message
        raw_body = getattr(error, "body", None) or getattr(error, "response", None)
        if raw_body is not None and hasattr(raw_body, "json"):
            try:
                body_json = raw_body.json()
                err = body_json.get("error", body_json)
                print(
                    f"[OpenAI 429] code={err.get('code')} "
                    f"type={err.get('type')} "
                    f"message={err.get('message')}"
                )
            except Exception:
                print(f"[OpenAI 429] raw error: {error}")
        else:
            print(f"[OpenAI 429] {error}")

        # Capture rate-limit headers from the 429 response
        if hasattr(error, "response") and error.response is not None:
            self.last_rate_limit = RateLimitInfo.from_headers(error.response.headers)

        if "insufficient_quota" in str(error):
            raise AIClientError(
                "OpenAI quota exceeded. Add credits at https://platform.openai.com/settings/billing"
            ) from error

        if attempt >= self.max_retries - 1:
            raise RateLimitExhausted(
                f"Rate limit persists after {self.max_retries} attempts",
                rate_limit=self.last_rate_limit,
            ) from error

        delay = self.base_delay * (2 ** attempt)  # 5, 10, 20, 40s
        print(f"Rate limited (attempt {attempt + 1}/{self.max_retries}). Retrying in {delay}s...")
        time.sleep(delay)

    def _handle_transient_error(self, error: OSError, attempt: int) -> None:
        if attempt >= self.max_retries - 1:
            raise AIClientError(f"API call failed after {self.max_retries} attempts: {error}") from error
        delay = 2 * (attempt + 1)
        print(f"Network error (attempt {attempt + 1}/{self.max_retries}): {error}. Retrying in {delay}s...")
        time.sleep(delay)
