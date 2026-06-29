"""Gateway seragam untuk Google.

Satu SlidingWindowRateLimiter GLOBAL dibagi oleh semua gateway sehingga
total panggilan ke semua provider (termasuk judge) tidak pernah melebihi RPM_LIMIT.
"""
from __future__ import annotations

from collections import deque
import os
import random
import re
import threading
import time
from typing import Callable, Optional
from dotenv import load_dotenv

load_dotenv(override=False)

import requests

from .configs import OLLAMA_API_BASE_URL, OLLAMA_LOCAL_BASE_URL, RPM_LIMIT

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

class ModelConfigurationError(RuntimeError):
    """Konfigurasi provider/model belum lengkap."""


# ---------------------------------------------------------------------------
# Rate limiter bersama (singleton per proses)
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Rate limiter rolling-window yang aman dipakai bersama beberapa gateway.

    Satu instance dipakai bersama oleh model yang diuji DAN judge sehingga
    total panggilan ke semua API tidak pernah melebihi `rpm` per menit.
    """

    def __init__(
        self,
        rpm: int = RPM_LIMIT,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpm < 1:
            raise ValueError("rpm harus minimal 1")
        self.rpm = rpm
        self.window_seconds = window_seconds
        self._clock = clock
        self._sleep = sleeper
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Tunggu sampai slot tersedia; kembalikan total waktu tunggu (detik)."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                cutoff = now - self.window_seconds
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()

                if len(self._calls) < self.rpm:
                    self._calls.append(now)
                    return waited

                delay = max(0.0, self.window_seconds - (now - self._calls[0]))

            # Tambahan kecil mencegah busy-loop akibat pembulatan clock sistem.
            delay = delay if delay > 0 else 0.001
            self._sleep(delay)
            waited += delay


# Singleton global — dibuat sekali saat modul diimpor.
_SHARED_LIMITER: SlidingWindowRateLimiter | None = None
_LIMITER_LOCK = threading.Lock()


def get_shared_limiter() -> SlidingWindowRateLimiter:
    """Kembalikan (atau buat) rate limiter tunggal untuk seluruh proses."""
    global _SHARED_LIMITER
    with _LIMITER_LOCK:
        if _SHARED_LIMITER is None:
            _SHARED_LIMITER = SlidingWindowRateLimiter(rpm=RPM_LIMIT)
    return _SHARED_LIMITER


def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass

    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if value is not None:
        return int(value)

    match = re.search(r"\b(429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def _retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    try:
        if retry_after is not None:
            return max(0.0, float(retry_after))
    except (TypeError, ValueError):
        pass

    match = re.search(r"(?:retry in|retryDelay[^0-9]*)(\d+(?:\.\d+)?)s", str(exc), re.I)
    if match:
        return float(match.group(1))
    return min(60.0, (2**attempt) * 2.0 + random.uniform(0, 0.5))


def _ollama_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/chat" if base.endswith("/api") else f"{base}/api/chat"


# ---------------------------------------------------------------------------
# ModelGateway
# ---------------------------------------------------------------------------

class ModelGateway:
    """Satu antarmuka generate untuk seluruh provider di registry.

    Menggunakan rate limiter bersama yang sama dengan gateway lain agar
    model yang diuji + judge tidak melebihi batas RPM global.
    """

    def __init__(
        self,
        spec: dict,
        limiter: Optional[SlidingWindowRateLimiter] = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 2,
    ) -> None:
        self.spec = spec
        self.limiter = limiter or get_shared_limiter()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate teks; setiap attempt (termasuk retry) dihitung oleh limiter."""
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                text = self._dispatch(prompt=prompt, system_prompt=system_prompt)
                if not text or not text.strip():
                    raise RuntimeError("respons model kosong")
                return text.strip()
            except Exception as exc:
                code = _status_code(exc)
                if code not in RETRYABLE_STATUS_CODES or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"{self.spec['display_name']} gagal: {exc}"
                    ) from exc
                time.sleep(_retry_delay(exc, attempt))
        raise RuntimeError("generate gagal tanpa exception")  # pragma: no cover

    def _dispatch(self, prompt: str, system_prompt: str) -> str:
        provider = self.spec["provider"]
        if provider == "google":
            return self._generate_google(prompt, system_prompt)
        if provider in {"ollama_local", "ollama_api"}:
            return self._generate_ollama(prompt, system_prompt)
        raise ModelConfigurationError(f"Provider tidak didukung: {provider}")


    def _generate_google(self, prompt: str, system_prompt: str) -> str:
        try:
            from google import genai
        except ImportError as exc:
            raise ModelConfigurationError("Pasang dependensi: pip install google-genai") from exc

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ModelConfigurationError("Set env var GOOGLE_API_KEY atau GEMINI_API_KEY")

        client = genai.Client(api_key=api_key)
        contents = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = client.models.generate_content(
            model=self.spec["model"],
            contents=contents,
        )
        text = getattr(response, "text", None)
        if text:
            return text
        raise RuntimeError("Respons Google tidak memiliki field text")


    def _generate_ollama(self, prompt: str, system_prompt: str) -> str:
        is_local = self.spec["provider"] == "ollama_local"
        base_url = OLLAMA_LOCAL_BASE_URL if is_local else OLLAMA_API_BASE_URL
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if not is_local and os.getenv("OLLAMA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OLLAMA_API_KEY']}"

        response = requests.post(
            _ollama_chat_url(base_url),
            headers=headers,
            json={"model": self.spec["model"], "messages": messages, "stream": False},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message") or {}
        text = message.get("content") or payload.get("response")
        if text:
            return str(text)
        raise RuntimeError("Respons Ollama tidak memiliki message.content")


def build_gateway(model_key: str) -> ModelGateway:
    """Buat ModelGateway untuk model yang terdaftar di configs.MODEL_SPECS."""
    from .configs import get_model_spec
    spec = get_model_spec(model_key)
    return ModelGateway(spec=spec, limiter=get_shared_limiter())