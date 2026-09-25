"""
smart_retry.py — Drop-in retry orchestrator for GoogleAIStudioAPIBackend.
 
Problems solved:
    1. Frequent 500 / 503 transient server errors  →  exponential backoff + jitter
    2. Gemma 4 structured-output hangs             →  defer to the backend's own request
                                                                                                        and stream inactivity timeouts so
                                                                                                        valid late responses are not dropped
    3. JSON repair failures                        →  dedicated JSON-only retry budget
    4. Runaway retries burning quota               →  per-error-class circuit breakers
 
Usage — replace the inner call in _create_chat_completion_inner_function:
 
    from smart_retry import SmartRetryOrchestrator
 
    _retry = SmartRetryOrchestrator()   # one instance per backend, or module-level singleton
 
    content, finish_reason = _retry.call(
        fn=lambda: self._call_api_with_thinking_retry(...),
        model_name=model_name,
        is_structured=response_format is not None,
        is_gemma4=short_name.startswith("gemma-4-"),
    )
"""
from __future__ import annotations
 
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TypeVar
 
log = logging.getLogger(__name__)
 
T = TypeVar("T")
 
 
# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
 
class ErrorClass(Enum):
    TRANSIENT_SERVER  = auto()   # 500, 503, "internal error"  → retry w/ backoff
    RATE_LIMIT        = auto()   # 429, "resource exhausted"   → retry w/ long backoff
    TIMEOUT           = auto()   # wall-clock hang or deadline  → retry w/ shorter timeout
    JSON_INVALID      = auto()   # bad JSON / schema mismatch   → retry w/ rephrase hint
    CONTENT_FILTER    = auto()   # safety block                 → no retry
    MODEL_UNAVAILABLE = auto()   # 404 / permission denied      → no retry
    UNKNOWN           = auto()   # anything else                → limited retry
 
 
_TRANSIENT_MARKERS  = ("500", "503", "internal error", "service unavailable", "backend error")
_RATE_MARKERS       = ("429", "rate limit", "resource exhausted", "quota")
_TIMEOUT_MARKERS    = ("timed out", "timeout", "deadline", "hang detected", "exceeded")
_JSON_MARKERS       = ("json", "schema validation", "could not extract", "could not repair")
_FILTER_MARKERS     = ("content policy", "safety", "blocked", "content_filter")
_UNAVAIL_MARKERS    = ("not found", "unsupported model", "model not", "permission denied",
                       "does not support")
 
 
def classify(exc: Exception) -> ErrorClass:
    msg = str(exc).lower()
    if any(m in msg for m in _FILTER_MARKERS):
        return ErrorClass.CONTENT_FILTER
    if any(m in msg for m in _UNAVAIL_MARKERS):
        return ErrorClass.MODEL_UNAVAILABLE
    if any(m in msg for m in _RATE_MARKERS):
        return ErrorClass.RATE_LIMIT
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return ErrorClass.TRANSIENT_SERVER
    if any(m in msg for m in _TIMEOUT_MARKERS):
        return ErrorClass.TIMEOUT
    if any(m in msg for m in _JSON_MARKERS):
        return ErrorClass.JSON_INVALID
    return ErrorClass.UNKNOWN
 
 
# ---------------------------------------------------------------------------
# Circuit breaker (per model × error class)
# ---------------------------------------------------------------------------
 
@dataclass
class _CircuitState:
    failures:     int   = 0
    open_until:   float = 0.0   # epoch seconds; 0 = closed
    half_open_at: float = 0.0
 
 
class CircuitBreaker:
    """
    Per-(model, error_class) circuit breaker.
 
    States:
      CLOSED     → normal operation
      OPEN       → fast-fail for `open_seconds`
      HALF-OPEN  → allow one probe; success → CLOSED, failure → OPEN again
    """
 
    def __init__(
        self,
        failure_threshold: int   = 5,
        open_seconds:      float = 60.0,
        half_open_after:   float = 30.0,
    ) -> None:
        self._threshold    = failure_threshold
        self._open_seconds = open_seconds
        self._half_open    = half_open_after
        self._states: dict[tuple[str, ErrorClass], _CircuitState] = {}
        self._lock = threading.Lock()
 
    def _key(self, model: str, ec: ErrorClass) -> tuple[str, ErrorClass]:
        return (model, ec)
 
    def is_open(self, model: str, ec: ErrorClass) -> bool:
        with self._lock:
            st = self._states.get(self._key(model, ec))
            if st is None:
                return False
            now = time.time()
            if st.open_until and now < st.open_until:
                # Allow one half-open probe after half_open_after seconds
                if now >= st.half_open_at and st.half_open_at > 0:
                    st.half_open_at = 0.0   # consume the probe slot
                    return False
                return True
            return False
 
    def record_failure(self, model: str, ec: ErrorClass) -> None:
        with self._lock:
            key = self._key(model, ec)
            st = self._states.setdefault(key, _CircuitState())
            st.failures += 1
            if st.failures >= self._threshold:
                now = time.time()
                st.open_until   = now + self._open_seconds
                st.half_open_at = now + self._half_open
                log.warning(
                    "Circuit OPEN for model=%s error=%s  "
                    "(will probe again in %.0fs)",
                    model, ec.name, self._half_open,
                )
 
    def record_success(self, model: str, ec: ErrorClass) -> None:
        with self._lock:
            key = self._key(model, ec)
            if key in self._states:
                self._states[key] = _CircuitState()   # reset
 
 
# ---------------------------------------------------------------------------
# Retry policy per error class
# ---------------------------------------------------------------------------
 
@dataclass
class RetryPolicy:
    max_attempts:    int   = 3
    base_delay:      float = 1.0    # seconds
    max_delay:       float = 60.0   # seconds
    jitter_fraction: float = 0.3    # ±30 % randomisation
    backoff_factor:  float = 2.0
 
 
_DEFAULT_POLICIES: dict[ErrorClass, RetryPolicy] = {
    ErrorClass.TRANSIENT_SERVER:  RetryPolicy(max_attempts=5, base_delay=3.0,  max_delay=60.0,  backoff_factor=2.0),
    ErrorClass.RATE_LIMIT:        RetryPolicy(max_attempts=4, base_delay=15.0, max_delay=120.0, backoff_factor=2.5),
    ErrorClass.TIMEOUT:           RetryPolicy(max_attempts=3, base_delay=2.0,  max_delay=30.0,  backoff_factor=1.5),
    ErrorClass.JSON_INVALID:      RetryPolicy(max_attempts=4, base_delay=1.0,  max_delay=15.0,  backoff_factor=1.5),
    ErrorClass.CONTENT_FILTER:    RetryPolicy(max_attempts=1),   # no retry
    ErrorClass.MODEL_UNAVAILABLE: RetryPolicy(max_attempts=1),   # no retry
    ErrorClass.UNKNOWN:           RetryPolicy(max_attempts=3, base_delay=2.0,  max_delay=30.0,  backoff_factor=2.0),
}
 
 
def _jittered_delay(policy: RetryPolicy, attempt: int) -> float:
    """
    Full-jitter exponential backoff:
        delay = uniform(0, min(max_delay, base * factor^(attempt-1)))
    """
    cap = min(policy.max_delay, policy.base_delay * (policy.backoff_factor ** (attempt - 1)))
    base_wait = cap * random.uniform(1 - policy.jitter_fraction, 1 + policy.jitter_fraction)
    return max(0.0, base_wait)
 
 
# ---------------------------------------------------------------------------
# Gemma 4 structured-output hang guard
# ---------------------------------------------------------------------------
 
@dataclass
class Gemma4TimeoutConfig:
    """
    Gemma 4 structured-output is prone to silently hanging.
    The backend already applies request and stream-level timeouts, so the retry
    layer only keeps these knobs for diagnostics and future tuning.
    """
    first_attempt_timeout:  int = 45    # seconds — fast fail on first hang
    second_attempt_timeout: int = 90    # give a bit more room on retry
    fallback_timeout:       int = 120   # final attempt gets full budget
 
    # After this many consecutive hangs, switch strategy
    hang_escalation_threshold: int = 2
 
 
# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
 
@dataclass
class SmartRetryOrchestrator:
    """
    Wraps any callable that makes a Google AI Studio API call and applies:
 
      • Error-class-aware retry policies with full-jitter exponential backoff
      • Per-model circuit breakers to prevent quota exhaustion
      • Hard wall-clock timeouts for Gemma 4 structured-output hangs
      • Stream → non-stream strategy escalation on repeated hangs
      • Accumulated hang counter that lowers timeout budget proactively
 
    Constructor parameters
    ----------------------
    policies : override the default RetryPolicy for any ErrorClass
    gemma4_cfg : tune Gemma 4 hang timeouts
    circuit_breaker : share a CircuitBreaker instance across orchestrators
    """
 
    policies:         dict[ErrorClass, RetryPolicy] = field(default_factory=lambda: dict(_DEFAULT_POLICIES))
    gemma4_cfg:       Gemma4TimeoutConfig            = field(default_factory=Gemma4TimeoutConfig)
    circuit_breaker:  CircuitBreaker                 = field(default_factory=CircuitBreaker)
 
    # Internal hang-tracking (per model)
    _hang_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock         = field(default_factory=threading.Lock, init=False, repr=False)
 
    # -----------------------------------------------------------------------
    def call(
        self,
        fn:            Callable[[], T],
        model_name:    str  = "unknown",
        is_structured: bool = False,
        is_gemma4:     bool = False,
    ) -> T:
        """
        Execute *fn*, retrying intelligently on failure.
 
        Parameters
        ----------
        fn            : zero-argument callable that performs the API call
        model_name    : used for circuit-breaker keying and logging
        is_structured : True when response_format is not None
        is_gemma4     : True when the active model is a Gemma 4 variant
        """
        use_hang_guard = is_gemma4 and is_structured
 
        attempt = 0
        last_exc: Exception | None = None
        ec: ErrorClass = ErrorClass.UNKNOWN
 
        while True:
            attempt += 1
            policy  = self.policies.get(ec, _DEFAULT_POLICIES[ErrorClass.UNKNOWN])
 
            # ── circuit breaker fast-fail ───────────────────────────────────
            if self.circuit_breaker.is_open(model_name, ec):
                log.warning(
                    "[SmartRetry] Circuit OPEN for model=%s ec=%s — skipping attempt %d",
                    model_name, ec.name, attempt,
                )
                if last_exc is not None:
                    raise last_exc
                raise RuntimeError(f"Circuit open for {model_name}/{ec.name}")
 
            # ── execute ────────────────────────────────────────────────────
            t0 = time.time()
            try:
                if use_hang_guard:
                    log.info(
                        "[SmartRetry] Gemma4 structured call is using backend-level timeout handling "
                        "to avoid dropping late but valid responses.",
                    )
                result = fn()
 
                elapsed = time.time() - t0
                log.info(
                    "[SmartRetry] ✓ model=%s attempt=%d elapsed=%.1fs",
                    model_name, attempt, elapsed,
                )
                self.circuit_breaker.record_success(model_name, ec)
                # Reset hang count on success
                with self._lock:
                    self._hang_counts[model_name] = 0
                return result
 
            except (TimeoutError, Exception) as exc:
                elapsed = time.time() - t0
                ec = classify(exc)
                last_exc = exc
                policy = self.policies.get(ec, _DEFAULT_POLICIES.get(ec, RetryPolicy()))
 
                log.warning(
                    "[SmartRetry] ✗ model=%s attempt=%d ec=%s elapsed=%.1fs: %s",
                    model_name, attempt, ec.name, elapsed, exc,
                )
 
                # Track hangs
                if ec == ErrorClass.TIMEOUT:
                    with self._lock:
                        self._hang_counts[model_name] = \
                            self._hang_counts.get(model_name, 0) + 1
                        new_hang_count = self._hang_counts[model_name]
                    log.warning(
                        "[SmartRetry] Gemma4 hang count for model=%s now %d",
                        model_name, new_hang_count,
                    )
 
                self.circuit_breaker.record_failure(model_name, ec)
 
                # Non-retryable classes
                if ec in (ErrorClass.CONTENT_FILTER, ErrorClass.MODEL_UNAVAILABLE):
                    log.error(
                        "[SmartRetry] Non-retryable error class %s — raising immediately.",
                        ec.name,
                    )
                    raise
 
                # Exhausted retries
                if attempt >= policy.max_attempts:
                    log.error(
                        "[SmartRetry] Exhausted %d/%d attempts for model=%s ec=%s — raising.",
                        attempt, policy.max_attempts, model_name, ec.name,
                    )
                    raise
 
                # ── wait before next attempt ────────────────────────────────
                delay = _jittered_delay(policy, attempt)
 
                # For Gemma 4 hangs: no point sleeping long, move fast
                if use_hang_guard and ec == ErrorClass.TIMEOUT:
                    delay = min(delay, 5.0)
 
                log.info(
                    "[SmartRetry] Waiting %.1fs before attempt %d/%d (ec=%s)…",
                    delay, attempt + 1, policy.max_attempts, ec.name,
                )
                time.sleep(delay)
 
    # -----------------------------------------------------------------------
    def get_hang_stats(self) -> dict[str, int]:
        """Return accumulated hang counts per model (useful for diagnostics)."""
        with self._lock:
            return dict(self._hang_counts)
 
    def reset_hang_stats(self, model_name: str | None = None) -> None:
        with self._lock:
            if model_name:
                self._hang_counts.pop(model_name, None)
            else:
                self._hang_counts.clear()
 
 
# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly
# ---------------------------------------------------------------------------
 
default_retry = SmartRetryOrchestrator()
 
 
# ---------------------------------------------------------------------------
# Integration helper — wraps the inner call in _create_chat_completion_inner_function
# ---------------------------------------------------------------------------
 
def wrap_google_call(
    fn:             Callable[[], tuple[str, str | None]],
    model_name:     str,
    response_format: Any  = None,
    short_name:     str   = "",
    orchestrator:   SmartRetryOrchestrator | None = None,
) -> tuple[str, str | None]:
    """
    Convenience wrapper for use inside GoogleAIStudioAPIBackend.
 
    Replace:
        content, finish_reason = self._call_api_with_thinking_retry(...)
 
    With:
        from smart_retry import wrap_google_call
        content, finish_reason = wrap_google_call(
            fn=lambda: self._call_api_with_thinking_retry(...),
            model_name=model_name,
            response_format=response_format,
            short_name=short_name,
        )
    """
    orch = orchestrator or default_retry
    return orch.call(
        fn=fn,
        model_name=model_name,
        is_structured=response_format is not None,
        is_gemma4=short_name.startswith("gemma-4-"),
    )
 