from __future__ import annotations

import concurrent.futures
import json
import os
import re
import textwrap
import threading
import time
from typing import Any, Dict, List, Literal, Optional, Type, TypedDict, Union, cast

from google import genai
from google.genai import types as genai_types
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.oai.backend.base import APIBackend
from rdagent.oai.llm_conf import LLMSettings


from rdagent.oai.backend.smart_retry import wrap_google_call
from rdagent.oai.backend.stream_hang_guard import guarded_stream, StreamInactivityTimeout
from rdagent.oai.backend.dynamic_tokens import estimate_structured_max_tokens


class GoogleAPIError(Exception):
    """Wrapper that gives Google exceptions a .message attribute compatible
    with base.py retry pattern-matching."""

    def __init__(self, original: Exception, message: str | None = None) -> None:
        self.message = message or str(original)
        self.original = original
        super().__init__(self.message)


class _ModelHealthStats:
    """Per-model statistics: response times, timeouts, success rates."""

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name
        self.response_times: list[float] = []
        self.timeout_count: int = 0
        self.success_count: int = 0
        self.total_attempts: int = 0
        self._lock = threading.Lock()

    def record_success(self, elapsed_seconds: float) -> None:
        with self._lock:
            self.response_times.append(elapsed_seconds)
            self.success_count += 1
            self.total_attempts += 1
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]

    def record_timeout(self) -> None:
        with self._lock:
            self.timeout_count += 1
            self.total_attempts += 1

    def get_avg_response_time(self) -> float:
        with self._lock:
            if not self.response_times:
                return 0.0
            return sum(self.response_times) / len(self.response_times)

    def get_timeout_rate(self) -> float:
        with self._lock:
            if self.total_attempts == 0:
                return 0.0
            return self.timeout_count / self.total_attempts

    def is_unhealthy(self) -> bool:
        return self.get_timeout_rate() > 0.5 and self.total_attempts >= 3

    def get_health_status(self) -> str:
        avg_time = self.get_avg_response_time()
        if self.total_attempts == 0:
            return "untested"
        if avg_time < 15:
            return "responsive"
        if avg_time < 60:
            return "slow"
        return "hanging"


class ModelHealthRegistry:
    """Global registry tracking model health across all calls."""

    def __init__(self) -> None:
        self._stats: Dict[str, _ModelHealthStats] = {}
        self._lock = threading.Lock()

    def _get_stats(self, model_name: str) -> _ModelHealthStats:
        stats = self._stats.get(model_name)
        if stats is None:
            stats = _ModelHealthStats(model_name)
            self._stats[model_name] = stats
        return stats

    def record_success(self, model_name: str, elapsed_seconds: float) -> None:
        with self._lock:
            self._get_stats(model_name).record_success(elapsed_seconds)

    def record_timeout(self, model_name: str) -> None:
        with self._lock:
            self._get_stats(model_name).record_timeout()

    def get_adaptive_timeout(self, model_name: str, attempt_number: int = 1, default_timeout: int = 60) -> int:
        with self._lock:
            stats = self._stats.get(model_name)
            if stats is None:
                return default_timeout
            avg_time = stats.get_avg_response_time()
            if stats.is_unhealthy():
                return 30
            if attempt_number == 1:
                if avg_time < 15:
                    return 30
                elif avg_time < 60:
                    return 45
                return 60
            else:
                return min(30 * (2 ** (attempt_number - 1)), 120)

    def get_ranked_fallback_models(self, primary_model: str, candidates: list[str]) -> list[str]:
        with self._lock:
            result = [primary_model] if primary_model in candidates else []
            others = [m for m in candidates if m != primary_model]
            ranked = sorted(
                others,
                key=lambda m: (
                    self._get_stats(m).is_unhealthy(),
                    self._get_stats(m).get_timeout_rate(),
                    self._get_stats(m).get_avg_response_time(),
                ),
            )
            result.extend(ranked)
            return result

    def get_status_summary(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                model: {
                    "status": stats.get_health_status(),
                    "avg_time_s": round(stats.get_avg_response_time(), 2),
                    "timeout_rate": round(stats.get_timeout_rate(), 2),
                    "total_attempts": stats.total_attempts,
                }
                for model, stats in self._stats.items()
                if stats.total_attempts > 0
            }


_model_health_registry = ModelHealthRegistry()


def _llm_status_banner(status: str, detail: str, color: str) -> str:
    bg_by_status = {
        "waiting": "\033[105m",
        "start": "\033[104m",
        "success": "\033[102m",
        "error": "\033[101m",
    }
    bg = bg_by_status.get(status, "\033[106m")
    return f"\033[1;30m{bg}=== LLM STATUS: {status} | {detail} ===\033[0m"


def _wrap_google_exception(e: Exception) -> Exception:
    msg = str(e).lower()
    if any(k in msg for k in ("too long", "token limit", "exceeds", "context length")):
        return GoogleAPIError(e, "maximum context length exceeded")
    if "429" in msg or "rate limit" in msg or "resource exhausted" in msg:
        return GoogleAPIError(e, f"Rate limit exceeded: {e}")
    if "timeout" in msg or "deadline" in msg:
        return GoogleAPIError(e, f"Request timed out: {e}")
    if "safety" in msg or "blocked" in msg or "policy" in msg:
        return GoogleAPIError(e, f"Content policy violation: {e}")
    if "500" in msg or "internal error" in msg or "503" in msg or "service unavailable" in msg:
        return GoogleAPIError(e, f"Transient server error (retryable): {e}")
    return GoogleAPIError(e)


class _ModelSpec(TypedDict):
    input_price:  float
    output_price: float
    max_input:    int
    max_output:   int
    schema:       bool
    thinking:     bool


_MODEL_REGISTRY: Dict[str, _ModelSpec] = {
    "gemini-2.5-pro":   dict(input_price=1.25,  output_price=10.00, max_input=1_048_576, max_output=65_536, schema=True,  thinking=True),
    "gemini-2.5-flash": dict(input_price=0.15,  output_price=0.60,  max_input=1_048_576, max_output=65_536, schema=True,  thinking=True),
    "gemini-2.0-flash": dict(input_price=0.10,  output_price=0.40,  max_input=1_048_576, max_output=8_192,  schema=True,  thinking=False),
    "gemini-1.5-pro":   dict(input_price=1.25,  output_price=5.00,  max_input=2_097_152, max_output=8_192,  schema=True,  thinking=False),
    "gemini-1.5-flash": dict(input_price=0.075, output_price=0.30,  max_input=1_048_576, max_output=8_192,  schema=True,  thinking=False),
    "gemma-4-31b-it":   dict(input_price=0.0,   output_price=0.0,   max_input=256_000,   max_output=8_192,  schema=False, thinking=False),   
    "gemma-4-26b-it":   dict(input_price=0.0,   output_price=0.0,   max_input=131_072,   max_output=8_192,  schema=False, thinking=False),
    "gemma-3-27b-it":   dict(input_price=0.0,   output_price=0.0,   max_input=131_072,   max_output=8_192,  schema=False, thinking=False),
    "gemma-3-12b-it":   dict(input_price=0.0,   output_price=0.0,   max_input=131_072,   max_output=8_192,  schema=False, thinking=False),
    "gemma-3-4b-it":    dict(input_price=0.0,   output_price=0.0,   max_input=131_072,   max_output=8_192,  schema=False, thinking=False),
    "gemma-3-1b-it":    dict(input_price=0.0,   output_price=0.0,   max_input=32_768,    max_output=8_192,  schema=False, thinking=False),
    "gemma-2-27b-it":   dict(input_price=0.0,   output_price=0.0,   max_input=8_192,     max_output=8_192,  schema=False, thinking=False),
}

MODELS_WITH_RESPONSE_SCHEMA: frozenset[str] = frozenset(
    k for k, v in _MODEL_REGISTRY.items() if v["schema"]
)

MODELS_WITH_THINKING: frozenset[str] = frozenset(
    k for k, v in _MODEL_REGISTRY.items() if v["thinking"]
)

_IGNORED_KWARGS = frozenset({
    "stream", "max_retries", "retries", "model", "temperature",
    "max_tokens", "reasoning_effort", "seed", "n", "stop",
    "presence_penalty", "frequency_penalty", "top_p", "logit_bias",
    "user", "functions", "function_call", "tools", "tool_choice",
})

_JSON_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)
_JSON_INLINE_FENCE_RE = re.compile(r"```(?:json)?(.+?)```", re.DOTALL | re.IGNORECASE)
_BAD_ESCAPE_RE = re.compile(r'\\([^"\\\\/bfnrtu])')

_THINKING_CHANNEL_TAG_RE = re.compile(
    r"<\|channel\|?>.*?<\|?channel\|>",
    re.DOTALL | re.IGNORECASE,
)
_THINKING_CHANNEL_BLOCK_RE = re.compile(
    r"<\|channel\s*>.*?<\s*channel\|>",
    re.DOTALL | re.IGNORECASE,
)

_JSON_PROMPT_TEMPLATE = textwrap.dedent("""    Respond with a single, complete, valid JSON object that strictly follows
    this JSON Schema. Output ONLY the JSON object — no explanation, no markdown
    fences, no trailing text.

    CRITICAL JSON RULES — violation will cause a parse error:
    - All backslashes inside string values MUST be doubled: \\\\ not \\
    - File/directory paths must escape every backslash: "C:\\\\Users\\\\file.txt"
    - Only these escape sequences are valid in JSON strings:
      \\\\ \\" \\/ \\b \\f \\n \\r \\t \\uXXXX
    - Never use raw backslashes before normal letters: \\p \\_ \\: \\w are INVALID
    - No trailing commas. No comments. No extra keys outside the schema.

    Schema:
    {schema}
""")

_JSON_PROMPT_TEMPLATE_GEMMA4 = textwrap.dedent("""    OUTPUT FORMAT: Respond with ONLY a valid JSON object matching this schema.
    No markdown, no explanation, no extra text.
    Rules: double all backslashes (\\\\), no trailing commas, no comments.

    Schema: {schema}
""")


class GoogleAIStudioSettings(LLMSettings):
    model_config = ConfigDict(env_prefix="GOOGLE_AI_STUDIO_")

    api_key: str = Field(default="", description="Google AI Studio API key.")
    chat_top_p: float | None = Field(default=None, description="Nucleus sampling threshold.")
    chat_top_k: int | None = Field(default=None, description="Top-k sampling cutoff.")
    enable_thinking: bool = Field(default=False, description="Enable model thinking / extended-reasoning mode when supported.")


GOOGLE_SETTINGS = GoogleAIStudioSettings()
ACC_COST: float = 0.0
_GENAI_CLIENT: genai.Client | None = None


def _ensure_configured() -> None:
    if not getattr(_ensure_configured, "_done", False):
        api_key = GOOGLE_SETTINGS.api_key or os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GOOGLE_AI_STUDIO_API_KEY must be set as an env var or in settings. "
                "Get one at https://aistudio.google.com/apikey"
            )
        _ensure_configured._done = True


def _get_client() -> genai.Client:
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        api_key = GOOGLE_SETTINGS.api_key or os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "")
        _GENAI_CLIENT = genai.Client(api_key=api_key)
    return _GENAI_CLIENT


def _model_short_name(model: str) -> str:
    return model.split("/")[-1] if "/" in model else model


def _convert_messages_to_google_format(
    messages: List[Dict[str, Any]],
) -> tuple[str | None, List[Dict[str, Any]]]:
    """
    Convert OpenAI-style messages to Google Generative AI format.
    Returns (system_instruction, contents).
    """
    system_parts: List[str] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "") or ""

        if role in ("system", "developer"):
            system_parts.append(content)
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": f"[{role}] {content}"}]})

    system_instruction = "\n\n".join(system_parts) if system_parts else None

    if contents and contents[0]["role"] == "model":
        contents.insert(0, {"role": "user", "parts": [{"text": "Continue."}]})

    merged: List[Dict[str, Any]] = []
    for c in contents:
        if merged and merged[-1]["role"] == c["role"]:
            merged[-1]["parts"].extend(c["parts"])
        else:
            merged.append(c)

    if not merged:
        merged.append({"role": "user", "parts": [{"text": "Hello."}]})

    return system_instruction, merged


def _pydantic_to_google_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """
    Convert a Pydantic v2 BaseModel to a Google-compatible JSON schema dict.
    Resolves $refs inline, preserves anyOf/oneOf, removes unsupported keys.
    """
    schema = model_cls.model_json_schema()
    _UNSUPPORTED_KEYS = frozenset({"title", "$defs", "definitions", "$schema", "default"})

    defs = schema.get("$defs") or schema.get("definitions") or {}
    _cache: Dict[str, Any] = {}
    _in_progress: set[str] = set()

    def _resolve_ref(ref: str) -> Any:
        if ref in _cache:
            return _cache[ref]
        if ref in _in_progress:
            return {"type": "object"}
        key = ref.split("/")[-1]
        if key not in defs:
            return {"type": "object"}
        _in_progress.add(ref)
        resolved = _clean(defs[key])
        _in_progress.discard(ref)
        _cache[ref] = resolved
        return resolved

    def _clean(value: Any) -> Any:
        if isinstance(value, list):
            return [_clean(item) for item in value]
        if not isinstance(value, dict):
            return value
        if "$ref" in value:
            resolved = _resolve_ref(str(value["$ref"]))
            extras = {k: _clean(v) for k, v in value.items() if k != "$ref"}
            if isinstance(resolved, dict):
                return {**resolved, **extras}
            return extras or {"type": "object"}
        result = {}
        for k, v in value.items():
            if k in _UNSUPPORTED_KEYS:
                continue
            result[k] = _clean(v)
        return result

    return _clean(schema)


def _strip_thinking_channel_tags(text: str) -> str:
    """
    v3 tweak: Strip thinking channel tags from model output.
    Even with thinking "off", the 31B model emits empty
    <|channel>thought\n<channel|> tags before output.
    This function removes them so they don't interfere with JSON extraction.
    """
    if "<" not in text:
        return text
    cleaned = _THINKING_CHANNEL_BLOCK_RE.sub("", text)
    cleaned = _THINKING_CHANNEL_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _sanitize_json_string_escapes(text: str) -> str:
    """
    Fix invalid JSON escape sequences that Gemma commonly emits inside string
    values — e.g. \\p, \\_, \\:, \\w, and bare Windows path separators.

    Strategy: locate every JSON string literal via a state machine, then within
    each string replace any backslash that is not followed by a valid JSON
    escape character with a doubled backslash.

    This is safer than a global regex because it only touches content inside
    quoted strings, leaving JSON structural characters untouched.
    """
    if "\\" not in text:
        return text

    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch != '"':
            result.append(ch)
            i += 1
            continue
        result.append('"')
        i += 1
        while i < n:
            c = text[i]
            if c == '"':
                result.append('"')
                i += 1
                break
            elif c == '\\':
                if i + 1 < n:
                    nxt = text[i + 1]
                    if nxt in ('"', '/', '\\', 'b', 'f', 'n', 'r', 't', 'u'):
                        result.append('\\')
                        result.append(nxt)
                        i += 2
                        if nxt == 'u':
                            hex_digits = text[i:i + 4]
                            result.append(hex_digits)
                            i += 4
                    else:
                        result.append('\\\\')
                        i += 1
                else:
                    result.append('\\\\')
                    i += 1
            else:
                result.append(c)
                i += 1

    return "".join(result)


def _fix_escape_at_position(text: str, exc: json.JSONDecodeError) -> str:
    """
    Targeted positional fix: given a JSONDecodeError for an invalid escape,
    double the offending backslash at exc.pos - 1.
    Returns the patched string, or the original if the position doesn't look
    like a backslash (so the caller can detect no-op and break the retry loop).
    """
    if "escape" not in str(exc).lower():
        return text
    pos = exc.pos
    if pos > 0 and pos <= len(text) and text[pos - 1] == "\\":
        return text[: pos - 1] + "\\\\" + text[pos:]
    return text


def _extract_json_from_text(text: str) -> str:
    """
    Best-effort extraction of a JSON object or array from free-form model output.

    v3 changes vs. v2:
    - Strips thinking channel tags FIRST, before any parse attempt.

    v2 changes vs. original:
    - Runs _sanitize_json_string_escapes FIRST, before any parse attempt.
    - Carries the sanitized form forward through all fallback strategies.
    - Falls back to positional escape fixing as a last resort before giving up.

    Strategy (in order):
    1. Strip thinking channel tags.
    2. Sanitize bad escapes in string literals.
    3. If the sanitized text parses as JSON, return it.
    4. Strip outer markdown fences (```json ... ```).
    5. Try inline fence extraction.
    6. Find the first '{' or '[' and use raw_decode to extract the longest
       valid JSON prefix — tolerates trailing prose after the JSON.
    7. Return the sanitized text unchanged and let the caller decide.
    """
    stripped = text.strip()

    stripped = _strip_thinking_channel_tags(stripped)

    sanitized = _sanitize_json_string_escapes(stripped)

    try:
        json.loads(sanitized)
        return sanitized
    except json.JSONDecodeError:
        pass

    for candidate_text in (sanitized, stripped):
        m = _JSON_FENCE_RE.match(candidate_text)
        if m:
            candidate = _sanitize_json_string_escapes(m.group(1).strip())
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                sanitized = candidate
                break

    m2 = _JSON_INLINE_FENCE_RE.search(sanitized)
    if m2:
        candidate = _sanitize_json_string_escapes(m2.group(1).strip())
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    for start_char in ('{', '['):
        idx = sanitized.find(start_char)
        if idx == -1:
            continue
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(sanitized, idx)
            return json.dumps(obj)
        except json.JSONDecodeError:
            pass

    return sanitized


def _compress_schema_for_small_model(schema: dict) -> dict:
    """Strip descriptions, titles, defaults, and examples from a JSON schema
    to reduce token count for prompt-based JSON enforcement on small models."""
    STRIP_KEYS = {"title", "description", "examples", "default", "$comment"}

    def _strip(obj):
        if isinstance(obj, dict):
            return {
                k: _strip(v)
                for k, v in obj.items()
                if k not in STRIP_KEYS
            }
        if isinstance(obj, list):
            return [_strip(item) for item in obj]
        return obj

    return _strip(schema)


def _repair_json_for_schema(
    text: str,
    schema_cls: Type[BaseModel] | None,
) -> str:
    """
    Extract JSON from text, iteratively fix remaining escape errors, then
    optionally validate against a Pydantic schema.

    v2 changes vs. original:
    - After _extract_json_from_text, runs up to MAX_ESCAPE_FIXES positional
      escape fixes before giving up, so that single-bad-escape responses
      (very common with Gemma) are repaired in-place rather than retried.
    - Only raises ValueError for errors that positional fixing cannot resolve,
      preserving base.py's retry budget for genuinely unrecoverable outputs.

    Returns a JSON string or raises ValueError.
    """
    MAX_ESCAPE_FIXES = 10
    extracted = _extract_json_from_text(text)

    for attempt in range(MAX_ESCAPE_FIXES):
        try:
            obj = json.loads(extracted)
            break
        except json.JSONDecodeError as exc:
            exc_str = str(exc).lower()
            if "escape" in exc_str or "invalid \\escape" in exc_str:
                fixed = _fix_escape_at_position(extracted, exc)
                if fixed == extracted:
                    raise ValueError(
                        f"Could not fix escape error at pos {exc.pos}: {exc}\n"
                        f"Raw text (first 500 chars): {text[:500]}"
                    ) from exc
                extracted = fixed
                logger.warning(
                    f"Fixed bad JSON escape (attempt {attempt + 1}/{MAX_ESCAPE_FIXES}): {exc}",
                    tag="llm_messages",
                )
            else:
                raise ValueError(
                    f"Could not extract valid JSON from model output: {exc}"
                ) from exc
    else:
        raise ValueError(
            f"Could not repair JSON after {MAX_ESCAPE_FIXES} escape fixes.\n"
            f"Raw text (first 500 chars): {text[:500]}"
        )

    if schema_cls is None:
        return json.dumps(obj)

    try:
        validated = schema_cls.model_validate(obj)
        return validated.model_dump_json()
    except Exception as exc:
        raise ValueError(f"Schema validation failed: {exc}") from exc


def _build_json_prompt_suffix(
    response_format: Any,
    compress: bool = False,
    model_family: str = "",
) -> str:
    """
    Build a prompt suffix that instructs small models to return valid JSON.
    Uses the v2 template with explicit escape rules.
    Supports schema compression and model-specific templates.
    """
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        schema = response_format.model_json_schema()
    elif isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            return (
                "\n\nRespond with a single, complete, valid JSON object. "
                "Output ONLY the JSON — no prose, no fences. "
                "All backslashes in string values must be doubled (\\\\)."
            )
        schema = response_format
    else:
        return ""

    if compress:
        schema = _compress_schema_for_small_model(schema)

    schema_str = json.dumps(
        schema,
        indent=None if compress else 2,
        separators=(',', ':') if compress else (', ', ': '),
    )

    if model_family.startswith("gemma-4-"):
        template = _JSON_PROMPT_TEMPLATE_GEMMA4
    else:
        template = _JSON_PROMPT_TEMPLATE

    return "\n\n" + template.format(schema=schema_str)


def _estimate_schema_output_tokens(response_format: Any) -> int:
    """Estimate how many output tokens a schema response needs."""
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        schema_str = json.dumps(response_format.model_json_schema())
    elif isinstance(response_format, dict):
        schema_str = json.dumps(response_format)
    else:
        return 2048

    num_props = schema_str.count('"type"')
    has_arrays = '"array"' in schema_str
    schema_len = len(schema_str)

    base = 512
    base += num_props * 150
    if has_arrays:
        base += 1024
    if schema_len > 2000:
        base += 512

    return min(max(base, 1024), 8192)


def _compress_messages_for_structured_output(
    messages: list[dict[str, Any]],
    max_content_tokens: int = 60_000,
) -> list[dict[str, Any]]:
    """Compress message history when total content is too long for reliable
    structured output on small models.

    Strategy:
    - Always preserve system messages and the last user message in full.
    - Truncate middle conversation turns (keep first + last 2 turns).
    - Add a separator indicating content was compressed.
    """
    total_chars = sum(len(m.get("content", "") or "") for m in messages)
    estimated_tokens = total_chars // 4
    if estimated_tokens <= max_content_tokens:
        return messages

    system_msgs = [m for m in messages if m["role"] in ("system", "developer")]
    non_system = [m for m in messages if m["role"] not in ("system", "developer")]

    if len(non_system) <= 4:
        compressed = []
        for m in messages:
            m = dict(m)
            content = m.get("content", "") or ""
            if len(content) > 8000 and m["role"] not in ("system", "developer"):
                m["content"] = (
                    content[:3000]
                    + "\n\n[... content truncated for brevity ...]\n\n"
                    + content[-3000:]
                )
            compressed.append(m)
        return compressed

    keep_first = non_system[:1]
    keep_last = non_system[-2:]
    middle = non_system[1:-2]

    middle_summary = {
        "role": "user",
        "content": (
            f"[Previous conversation: {len(middle)} messages exchanged "
            f"covering the ongoing discussion. Key context preserved above and below.]"
        ),
    }
    return system_msgs + keep_first + [middle_summary] + keep_last


def _inject_json_schema_prompt(
    messages: list[dict[str, Any]],
    response_format: Any,
    compress: bool = False,
    inject_in_system: bool = False,
    model_family: str = "",
) -> list[dict[str, Any]]:
    """
    Append a JSON schema instruction to messages.

    When inject_in_system=True (recommended for Gemma 4), places the schema
    in the system message for persistent attention. Otherwise appends to the
    last user message (original behavior).
    """
    suffix = _build_json_prompt_suffix(response_format, compress=compress, model_family=model_family)
    if not suffix:
        return messages

    messages = list(messages)

    if inject_in_system:
        for i, msg in enumerate(messages):
            if msg["role"] in ("system", "developer"):
                msg = dict(msg)
                msg["content"] = (msg.get("content") or "") + suffix
                messages[i] = msg
                return messages
        messages.insert(0, {"role": "system", "content": suffix.lstrip()})
        return messages

    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            msg = dict(messages[i])
            msg["content"] = (msg.get("content") or "") + suffix
            messages[i] = msg
            return messages

    messages.append({"role": "user", "content": suffix.lstrip()})
    return messages


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    spec = _MODEL_REGISTRY.get(_model_short_name(model))
    in_price  = spec["input_price"]  if spec else 0.0
    out_price = spec["output_price"] if spec else 0.0
    return (prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price


def _looks_like_degenerate_repetition(text: str) -> bool:
    tail = text[-8_000:]
    if len(tail) < 600:
        return False
    words = tail.lower().split()
    if len(words) < 120:
        return False
    max_run = run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run >= 35


def _apply_output_integrity_guard(
    content: str,
    response_format: Optional[Union[dict, Type[BaseModel]]],
) -> str:
    """Reduce pathological prompt-recap loops in plain-text outputs."""
    if not content:
        return content

    enabled = os.environ.get("GOOGLE_AI_STUDIO_ENABLE_OUTPUT_GUARD", "true").lower() == "true"
    if not enabled or response_format is not None:
        return content

    strip_recap = os.environ.get("GOOGLE_AI_STUDIO_STRIP_PROMPT_RECAP", "true").lower() == "true"
    if not strip_recap:
        return content

    recap_re = re.compile(r"\bwait\b[\s*_`'\".,:;!?-]*the\s+prompt\s+says", re.IGNORECASE)
    confirmed_re = re.compile(r"^\s*(?:[-*]\s*)?confirmed\.?\s*$", re.IGNORECASE)

    lines = content.splitlines()
    recap_lines = [ln for ln in lines if recap_re.search(ln)]
    lines = [ln for ln in lines if not recap_re.search(ln)]

    if len(recap_lines) < 3:
        return "\n".join(lines).strip() or content

    filtered: list[str] = []
    for line in lines:
        if recap_re.search(line):
            continue
        if confirmed_re.match(line):
            continue
        filtered.append(line)

    if not any(ln.strip() for ln in filtered):
        recap_only_mode = os.environ.get("GOOGLE_AI_STUDIO_PROMPT_RECAP_ONLY_MODE", "suppress").lower()
        if recap_only_mode in {"suppress", "empty"}:
            return ""
        unique_recap: list[str] = []
        seen: set[str] = set()
        for line in recap_lines:
            key = line.strip()
            if key and key not in seen:
                seen.add(key)
                unique_recap.append(line)
            if len(unique_recap) >= 5:
                break
        if unique_recap:
            return "\n".join(unique_recap)
        return content

    deduped: list[str] = []
    last_norm = None
    for line in filtered:
        norm = line.strip()
        if norm and norm == last_norm:
            continue
        deduped.append(line)
        last_norm = norm if norm else last_norm

    return "\n".join(deduped).strip() or content


_BLOCKED_HARM_CATEGORIES: tuple[genai_types.HarmCategory, ...] = (
    genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
)


def _permissive_safety_settings() -> list[genai_types.SafetySetting]:
    return [
        genai_types.SafetySetting(category=cat, threshold=genai_types.HarmBlockThreshold.BLOCK_NONE)
        for cat in _BLOCKED_HARM_CATEGORIES
    ]


class GoogleAIStudioAPIBackend(APIBackend):
    """
    Google AI Studio (Gemini/Gemma) implementation of APIBackend.

    v2 improvements:
    - JSON repair pipeline overhauled: pre-sanitizes escape sequences before
      any parse attempt (state-machine based, only touches string interiors),
      then iteratively fixes remaining bad-escape positions reported by the
      parser. Most Gemma escape errors are now fixed without a retry.
    - JSON prompt template v2: explicit backslash/escape rules reduce bad
      output frequency at the source.
    - _repair_json_for_schema no longer raises on fixable escape errors,
      preserving base.py's retry budget for genuinely unrecoverable responses.
    - Gemma 4 structured output optimizations: message compression, schema
      compression, system message injection, forced thinking, temperature=0.1.

    v3 tweaks:
    - Strip thinking channel tags from JSON extraction.
    - Bumped gemma-4-31b-it max_input to 256_000.
    - Gemma 4 structured output: temperature=0.1, top_k=4, top_p=0.95.
    """

    _has_logged_settings: bool = False

    @staticmethod
    def _masked_google_settings() -> dict[str, Any]:
        data = GOOGLE_SETTINGS.model_dump()
        api_key = data.get("api_key")
        if isinstance(api_key, str) and api_key:
            data["api_key"] = "***"
        return data

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ensure_configured()
        if not self.__class__._has_logged_settings:
            masked = self._masked_google_settings()
            logger.info(f"Google AI Studio settings loaded: {masked}")
            logger.log_object(masked, tag="GOOGLE_AI_STUDIO_SETTINGS")
            self.__class__._has_logged_settings = True
        self._thinking_config_unsupported: bool = False
        super().__init__(*args, **kwargs)

    @staticmethod
    def _request_timeout_seconds() -> int:
        raw = os.environ.get("GOOGLE_AI_STUDIO_REQUEST_TIMEOUT_SECONDS", "120")
        try:
            val = max(1, int(raw))
        except ValueError:
            val = 120
        return val

    @staticmethod
    def _chat_fallback_models(primary_model: str) -> list[str]:
        raw = os.environ.get("GOOGLE_AI_STUDIO_CHAT_FALLBACK_MODELS", "")
        configured = [m.strip() for m in raw.split(",") if m.strip()]

        candidates: list[str] = []
        for model in [primary_model, *configured]:
            if model and model not in candidates:
                candidates.append(model)
        return candidates

    @staticmethod
    def _is_model_unavailable_error(message: str) -> bool:
        lowered = message.lower()
        return (
            "not found" in lowered
            or "unsupported model" in lowered
            or "model not" in lowered
            or "does not support" in lowered
            or "permission denied" in lowered
        )

    @staticmethod
    def _run_with_timeout(
        func: Any,
        timeout_seconds: int,
        label: str,
        model_name: str = "",
        attempt_number: int = 1,
    ) -> Any:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(func)
        start_time = time.time()

        if model_name:
            adaptive_timeout = _model_health_registry.get_adaptive_timeout(
                model_name, attempt_number, timeout_seconds,
            )
            effective_timeout = max(timeout_seconds, adaptive_timeout)
        else:
            effective_timeout = timeout_seconds

        try:
            return future.result(timeout=effective_timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            elapsed = time.time() - start_time
            if model_name:
                _model_health_registry.record_timeout(model_name)
            timeout_msg = (
                f"{label} exceeded {effective_timeout}s timeout "
                f"(waited {elapsed:.1f}s, attempt #{attempt_number}). "
                f"Hang detected at: {label}. "
                f"This indicates the Google API is unresponsive at the network level. "
                f"Try: 1) Increase GOOGLE_AI_STUDIO_REQUEST_TIMEOUT_SECONDS, "
                f"2) Use a different model, 3) Check network connectivity."
            )
            logger.error(
                f"{LogColors.RED}{timeout_msg}{LogColors.END}",
                tag="llm_messages",
            )
            raise GoogleAPIError(Exception("timeout"), timeout_msg)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        try:
            client = _get_client()
            model_name = GOOGLE_SETTINGS.chat_model
            _, contents = _convert_messages_to_google_format(messages)
            resp = client.models.count_tokens(model=model_name, contents=contents)
            num_tokens = resp.total_tokens
        except Exception as e:
            logger.warning(
                f"Native token counting failed ({e}). Falling back to char estimate.",
                tag="debug_google_token",
            )
            text = " ".join(m.get("content", "") or "" for m in messages)
            num_tokens = max(1, len(text) // 4)
        logger.info(f"Token count: {num_tokens}", tag="debug_google_token")
        return num_tokens

    def _create_embedding_inner_function(
        self,
        input_content_list: list[str],
    ) -> list[list[float]]:
        configured_model = _model_short_name(GOOGLE_SETTINGS.embedding_model)
        fallback_models = [configured_model]

        for candidate in ("gemini-embedding-001", "text-embedding-004"):
            if candidate not in fallback_models:
                fallback_models.append(candidate)

        if GOOGLE_SETTINGS.log_llm_chat_content:
            logger.info(
                f"Creating embedding for {len(input_content_list)} item(s).",
                tag="debug_google_emb",
            )

        client = _get_client()
        last_error: Exception | None = None
        result = None
        used_model = configured_model

        for model_name in fallback_models:
            logger.info(f"Using emb model {model_name}", tag="debug_google_emb")
            try:
                result = client.models.embed_content(
                    model=model_name,
                    contents=input_content_list,
                    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                used_model = model_name
                break
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                model_missing = "not found" in msg or "not supported for embedcontent" in msg
                if model_missing and model_name != fallback_models[-1]:
                    logger.warning(
                        f"Embedding model '{model_name}' unavailable, trying next fallback.",
                        tag="debug_google_emb",
                    )
                    continue
                raise _wrap_google_exception(e) from e

        if result is None:
            assert last_error is not None
            raise _wrap_google_exception(last_error)

        if used_model != configured_model:
            logger.warning(
                f"{LogColors.YELLOW}Embedding model '{configured_model}' unavailable; "
                f"fell back to '{used_model}'.{LogColors.END}",
                tag="debug_google_emb",
            )
        return [list(emb.values or []) for emb in (result.embeddings or [])]

    class CompleteKwargs(TypedDict):
        model: str
        temperature: float
        top_p: float | None
        top_k: int | None
        enable_thinking: bool
        max_tokens: int | None
        reasoning_effort: Literal["low", "medium", "high"] | None

    def get_complete_kwargs(self) -> CompleteKwargs:
        model = GOOGLE_SETTINGS.chat_model
        temperature = GOOGLE_SETTINGS.chat_temperature
        top_p = GOOGLE_SETTINGS.chat_top_p
        top_k = GOOGLE_SETTINGS.chat_top_k
        enable_thinking = GOOGLE_SETTINGS.enable_thinking
        max_tokens = GOOGLE_SETTINGS.chat_max_tokens
        reasoning_effort = GOOGLE_SETTINGS.reasoning_effort

        if GOOGLE_SETTINGS.chat_model_map:
            current_tag = getattr(logger, "tag", "") or ""
            for tag_key, mc in GOOGLE_SETTINGS.chat_model_map.items():
                if tag_key in current_tag:
                    model = mc.get("model", model)
                    if "temperature" in mc:
                        temperature = float(mc["temperature"])
                    if "top_p" in mc:
                        top_p = float(mc["top_p"])
                    if "top_k" in mc:
                        top_k = int(mc["top_k"])
                    if "enable_thinking" in mc:
                        enable_thinking = bool(mc["enable_thinking"])
                    if "max_tokens" in mc:
                        max_tokens = int(mc["max_tokens"])
                    if mc.get("reasoning_effort") in ("low", "medium", "high"):
                        reasoning_effort = cast(
                            Literal["low", "medium", "high"], mc["reasoning_effort"]
                        )
                    else:
                        reasoning_effort = None
                    break

        return self.CompleteKwargs(
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

    def _create_chat_completion_inner_function(
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, str | None]:
        """
        Core chat completion. Called by base class auto-continue logic.

        For models without native schema support (Gemma 3/4, etc.):
        1. Appends a JSON schema prompt (v2: includes escape rules) to the last user message.
        2. Calls the API with plain text output.
        3. Extracts / repairs / validates JSON from the raw response.
           - Pre-sanitizes escape sequences via state machine (catches ~95% of Gemma issues).
           - Strips thinking channel tags (v3 tweak).
           - Iteratively fixes remaining positional escape errors (catches the rest).
           - Only re-raises ValueError for errors it cannot fix in-place.

        Returns (content_text, finish_reason).
        """
        for k in list(kwargs.keys()):
            if k in _IGNORED_KWARGS:
                kwargs.pop(k)

        ck = self.get_complete_kwargs()
        model_name = ck["model"]
        short_name = _model_short_name(model_name)

        use_native_schema = (
            response_format is not None
            and short_name in MODELS_WITH_RESPONSE_SCHEMA
            and GOOGLE_SETTINGS.enable_response_schema
        )
        use_prompt_json = (
            response_format is not None
            and not use_native_schema
        )

        if use_prompt_json:
            logger.info(
                f"Model {model_name} does not support native "
                f"response_schema. Falling back to prompt-based JSON enforcement.",
                tag="llm_messages",
            )

        effective_messages = list(messages)

        if use_prompt_json:
            is_gemma = short_name.startswith("gemma-")
            if is_gemma and short_name.startswith("gemma-4-"):
                effective_messages = _compress_messages_for_structured_output(
                    effective_messages,
                    max_content_tokens=50_000,
                )
            effective_messages = _inject_json_schema_prompt(
                effective_messages,
                response_format,
                compress=is_gemma,
                inject_in_system=is_gemma,
                model_family=short_name,
            )

        if GOOGLE_SETTINGS.log_llm_chat_content:
            if GOOGLE_SETTINGS.show_llm_prompts:
                logger.info(self._build_log_messages(effective_messages), tag="llm_messages")

        system_instruction, contents = _convert_messages_to_google_format(effective_messages)

        if response_format is not None:
            if short_name.startswith("gemma-4-"):
                effective_temp = 1
                gen_kwargs: Dict[str, Any] = {"temperature": effective_temp, "top_k": 64, "top_p": 0.95}
            else:
                effective_temp = min(float(ck["temperature"]), 0.2)
                gen_kwargs: Dict[str, Any] = {"temperature": effective_temp}
        else:
            effective_temp = float(ck["temperature"])
            gen_kwargs: Dict[str, Any] = {"temperature": effective_temp}

        if ck["top_p"] is not None and "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = float(ck["top_p"])
        if ck["top_k"] is not None and "top_k" not in gen_kwargs:
            gen_kwargs["top_k"] = int(ck["top_k"])

        force_thinking_for_structured = (
            response_format is not None
            and short_name.startswith("gemma-4-")
            and short_name in MODELS_WITH_THINKING
        )

        if not self._thinking_config_unsupported:
            if force_thinking_for_structured:
                gen_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_budget=1024
                )
            elif not ck["enable_thinking"] and short_name in MODELS_WITH_THINKING:
                gen_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_budget=0
                )

        if ck["max_tokens"]:
            gen_kwargs["max_output_tokens"] = ck["max_tokens"]
        elif response_format is not None:
            gen_kwargs["max_output_tokens"] = estimate_structured_max_tokens(
                response_format=response_format,
                messages=effective_messages,
            )

        if use_native_schema:
            rf = response_format
            if isinstance(rf, type) and issubclass(rf, BaseModel):
                gen_kwargs["response_mime_type"] = "application/json"
                gen_kwargs["response_schema"] = _pydantic_to_google_schema(rf)
            elif isinstance(rf, dict):
                gen_kwargs["response_mime_type"] = "application/json"
                if rf.get("type") != "json_object":
                    gen_kwargs["response_schema"] = rf

        if system_instruction:
            gen_kwargs["system_instruction"] = system_instruction

        gen_kwargs["safety_settings"] = _permissive_safety_settings()

        generation_config = self._build_generation_config(gen_kwargs)

        client = _get_client()

        explicit_disable_structured_stream = (
            os.environ.get("GOOGLE_AI_STUDIO_DISABLE_STREAM_FOR_STRUCTURED_OUTPUT", "").lower() == "true"
        )
        force_structured_stream = (
            os.environ.get("GOOGLE_AI_STUDIO_FORCE_STREAM_FOR_STRUCTURED_OUTPUT", "false").lower() == "true"
        )
        prefer_non_stream_structured = (
            response_format is not None
            and short_name.startswith("gemma-")
            and not force_structured_stream
        )

        stream_enabled = GOOGLE_SETTINGS.chat_stream and (
            response_format is None
            or (not explicit_disable_structured_stream and not prefer_non_stream_structured)
            or force_structured_stream
        )

        structured_stream_fallback_enabled = (
            GOOGLE_SETTINGS.chat_stream
            and response_format is not None
            and not explicit_disable_structured_stream
            and not stream_enabled
        )

        if (
            prefer_non_stream_structured
            and not GOOGLE_SETTINGS.chat_stream
            and response_format is not None
        ):
            logger.warning(
                f"{LogColors.YELLOW}Structured stream fallback is disabled because "
                f"chat_stream=False. Gemma models may hang on non-stream structured "
                f"output. Consider enabling chat_stream.{LogColors.END}",
                tag="llm_messages",
            )

        allow_model_failover = (
            os.environ.get("GOOGLE_AI_STUDIO_ENABLE_MODEL_FAILOVER", "false").lower() == "true"
        )
        model_candidates = (
            self._chat_fallback_models(model_name)
            if allow_model_failover and response_format is None
            else [model_name]
        )

        smart_fallback = (
            os.environ.get("GOOGLE_AI_STUDIO_SMART_MODEL_FALLBACK", "false").lower() == "true"
        )
        if smart_fallback and len(model_candidates) > 1:
            model_candidates = _model_health_registry.get_ranked_fallback_models(model_name, model_candidates)
            logger.info(
                f"{LogColors.YELLOW}Smart fallback enabled. Model ranking by health: "
                f"{model_candidates}{LogColors.END}",
                tag="llm_messages",
            )

        last_error: Exception | None = None
        used_model = model_name
        content = ""
        finish_reason: str | None = None

        if GOOGLE_SETTINGS.log_llm_chat_content and not GOOGLE_SETTINGS.show_llm_prompts:
            logger.info(
                _llm_status_banner(
                    "waiting",
                    f"Model {model_name} is generating a response",
                    LogColors.YELLOW,
                ),
                tag="llm_messages",
            )

        for idx, candidate_model in enumerate(model_candidates):
            logger.info(f"Using chat model {candidate_model}", tag="llm_messages")
            try:
                content, finish_reason = wrap_google_call(
                    fn=lambda: self._call_api_with_thinking_retry(
                        client, candidate_model, contents, generation_config,
                        gen_kwargs, stream_enabled, response_format is not None,
                        structured_stream_fallback_enabled, idx + 1,
                    ),
                    model_name=candidate_model,
                    response_format=response_format,
                    short_name=short_name,
                )
                used_model = candidate_model
                break
            except Exception as e:
                last_error = e
                can_failover = idx < len(model_candidates) - 1
                is_timeout = "timed out" in str(e).lower() or "timeout" in str(e).lower()
                if can_failover and (is_timeout or self._is_model_unavailable_error(str(e))):
                    logger.warning(
                        f"{LogColors.YELLOW}Model {candidate_model} failed ({e}); "
                        f"trying fallback {model_candidates[idx + 1]}.{LogColors.END}",
                        tag="llm_messages",
                    )
                    continue
                raise
        else:
            if last_error is not None:
                raise last_error

        if response_format is not None and finish_reason == "content_filter":
            raise GoogleAPIError(
                Exception("content filtered during structured output"),
                "Content policy violation during structured output generation",
            )

        if response_format is not None and content:
            schema_cls = (
                response_format
                if isinstance(response_format, type) and issubclass(response_format, BaseModel)
                else None
            )
            try:
                content = _repair_json_for_schema(content, schema_cls)
            except ValueError as exc:
                logger.warning(
                    f"{LogColors.YELLOW}JSON repair/validation failed: {exc}. "
                    f"Re-raising so base.py can retry.{LogColors.END}",
                    tag="llm_messages",
                )
                raise

        content = _apply_output_integrity_guard(content, response_format)
        self._log_cost_and_tokens(used_model, messages, content, finish_reason)
        return content, finish_reason

    def supports_response_schema(self) -> bool:
        short = _model_short_name(GOOGLE_SETTINGS.chat_model)
        return short in MODELS_WITH_RESPONSE_SCHEMA and GOOGLE_SETTINGS.enable_response_schema

    @property
    def chat_token_limit(self) -> int:
        spec = _MODEL_REGISTRY.get(_model_short_name(GOOGLE_SETTINGS.chat_model))
        if spec:
            return spec["max_input"] - spec["max_output"]
        return super().chat_token_limit

    def _build_generation_config(
        self, gen_kwargs: Dict[str, Any]
    ) -> genai_types.GenerateContentConfig:
        try:
            return genai_types.GenerateContentConfig(**gen_kwargs)
        except TypeError as exc:
            if "thinking_config" in gen_kwargs:
                logger.warning(
                    f"{LogColors.YELLOW}GenerateContentConfig rejected thinking_config "
                    f"({exc}); proceeding without it.{LogColors.END}",
                    tag="llm_messages",
                )
                gen_kwargs.pop("thinking_config")
                self._thinking_config_unsupported = True
                return genai_types.GenerateContentConfig(**gen_kwargs)
            raise

    def _call_api_with_thinking_retry(
            self,
            client: genai.Client,
            model_name: str,
            contents: list[Dict[str, Any]],
            generation_config: genai_types.GenerateContentConfig,
            gen_kwargs: Dict[str, Any],
            stream_enabled: bool,
            expect_structured_output: bool,
            structured_stream_fallback_enabled: bool,
            attempt_number: int = 1,
        ) -> tuple[str, str | None]:
            try:
                if stream_enabled:
                    return self._stream_response(
                        client, model_name, contents, generation_config,
                        expect_structured_output, attempt_number,
                    )
                return self._non_stream_response(client, model_name, contents, generation_config, attempt_number)
            except Exception as e:
                msg = str(e).lower()
                if "thinking budget is not supported" in msg and "thinking_config" in gen_kwargs:
                    logger.info(
                        f"Model {model_name} rejected thinking budget at runtime; "
                        f"retrying without thinking_config.",
                        tag="llm_messages",
                    )
                    self._thinking_config_unsupported = True
                    gen_kwargs.pop("thinking_config")
                    generation_config = genai_types.GenerateContentConfig(**gen_kwargs)
                    try:
                        if stream_enabled:
                            return self._stream_response(
                                client, model_name, contents, generation_config,
                                expect_structured_output, attempt_number,
                            )
                        return self._non_stream_response(client, model_name, contents, generation_config, attempt_number)
                    except Exception as fallback_e:
                        logger.warning(f"RAW ERROR TYPE: {type(fallback_e).__name__}")
                        logger.warning(f"RAW ERROR ARGS: {fallback_e.args}")
                        if hasattr(fallback_e, '__dict__'):
                            logger.warning(f"RAW ERROR DICT: {fallback_e.__dict__}")
                        raise _wrap_google_exception(fallback_e) from fallback_e

                if (
                    not stream_enabled
                    and structured_stream_fallback_enabled
                    and expect_structured_output
                ):
                    logger.warning(
                        f"{LogColors.YELLOW}Non-stream structured call failed for {model_name}; "
                        f"retrying once with stream mode.{LogColors.END}",
                        tag="llm_messages",
                    )
                    try:
                        return self._stream_response(
                            client, model_name, contents, generation_config,
                            expect_structured_output, attempt_number,
                        )
                    except Exception as fallback_e:
                        logger.warning(f"RAW ERROR TYPE: {type(fallback_e).__name__}")
                        logger.warning(f"RAW ERROR ARGS: {fallback_e.args}")
                        if hasattr(fallback_e, '__dict__'):
                            logger.warning(f"RAW ERROR DICT: {fallback_e.__dict__}")
                        raise _wrap_google_exception(fallback_e) from fallback_e
                    
                if stream_enabled and ("500" in msg or "internal error" in msg or "503" in msg):
                    logger.warning(
                        f"{LogColors.YELLOW}Streaming call hit server error (500/503) for {model_name}; "
                        f"retrying once with non-stream mode.{LogColors.END}",
                        tag="llm_messages",
                    )
                    time.sleep(3)  # give Google's server a moment before retry
                    try:
                        return self._non_stream_response(client, model_name, contents, generation_config, attempt_number)
                    except Exception as fallback_e:
                        logger.warning(f"RAW ERROR TYPE: {type(fallback_e).__name__}")
                        logger.warning(f"RAW ERROR ARGS: {fallback_e.args}")
                        if hasattr(fallback_e, '__dict__'):
                            logger.warning(f"RAW ERROR DICT: {fallback_e.__dict__}")
                        raise _wrap_google_exception(fallback_e) from fallback_e

                if stream_enabled and ("timed out" in msg or "timeout" in msg or "deadline" in msg):
                    logger.warning(
                        f"{LogColors.YELLOW}Streaming call timed out for {model_name}; "
                        f"retrying once with non-stream mode.{LogColors.END}",
                        tag="llm_messages",
                    )
                    try:
                        return self._non_stream_response(client, model_name, contents, generation_config, attempt_number)
                    except Exception as fallback_e:
                        logger.warning(f"RAW ERROR TYPE: {type(fallback_e).__name__}")
                        logger.warning(f"RAW ERROR ARGS: {fallback_e.args}")
                        if hasattr(fallback_e, '__dict__'):
                            logger.warning(f"RAW ERROR DICT: {fallback_e.__dict__}")
                        raise _wrap_google_exception(fallback_e) from fallback_e

                logger.warning(f"RAW ERROR TYPE: {type(e).__name__}")
                logger.warning(f"RAW ERROR ARGS: {e.args}")
                if hasattr(e, '__dict__'):
                    logger.warning(f"RAW ERROR DICT: {e.__dict__}")
                raise _wrap_google_exception(e) from e

    def _stream_response(
            self,
            client: genai.Client,
            model_name: str,
            contents: list[Dict[str, Any]],
            generation_config: genai_types.GenerateContentConfig,
            expect_structured_output: bool = False,
            attempt_number: int = 1,
        ) -> tuple[str, str | None]:

            def _consume_stream(self_ref) -> tuple[str, str | None]:
                if GOOGLE_SETTINGS.log_llm_chat_content and GOOGLE_SETTINGS.show_llm_prompts:
                    logger.info("assistant (stream):", tag="llm_messages")

                start_time = time.time()
                start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
                logger.info(
                    f"[{start_time_str}] Initiating streaming API call to {model_name}...",
                    tag="llm_messages",
                )

                response = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )

                elapsed = time.time() - start_time
                now_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                logger.info(
                    f"[{now_time_str}] API streaming connection established in {elapsed:.2f}s, consuming chunks...",
                    tag="llm_messages",
                )

                content = ""
                finish_reason: str | None = None
                max_chars = int(os.environ.get("GOOGLE_AI_STUDIO_STREAM_MAX_CHARS", "200000"))
                chunk_count = 0

                inactivity_s = int(os.environ.get("GOOGLE_AI_STUDIO_STREAM_INACTIVITY_TIMEOUT_S", "30"))
                warn_s       = int(os.environ.get("GOOGLE_AI_STUDIO_STREAM_CHUNK_WARN_THRESHOLD_S", "10"))

                try:
                    chunk_iter = guarded_stream(response, inactivity_timeout_s=inactivity_s, warn_threshold_s=warn_s)

                    for chunk in chunk_iter:
                        chunk_count += 1
                        chunk_text = self_ref._extract_text_from_response(chunk)

                        if chunk_count == 1:
                            logger.info(
                                f"Received first chunk ({len(chunk_text)} chars)",
                                tag="llm_messages",
                            )

                        if chunk_text:
                            if chunk_text.startswith(content) and len(chunk_text) > len(content):
                                chunk_text = chunk_text[len(content):]
                            elif content.endswith(chunk_text):
                                chunk_text = ""

                        if chunk_text:
                            content += chunk_text
                            if GOOGLE_SETTINGS.log_llm_chat_content and GOOGLE_SETTINGS.show_llm_prompts:
                                logger.info(chunk_text, raw=True, tag="llm_messages")

                            if len(content) > max_chars:
                                raise GoogleAPIError(
                                    Exception("stream exceeded safety size limit"),
                                    "Request timed out: stream output exceeded safety size limit",
                                )

                            if _looks_like_degenerate_repetition(content):
                                raise GoogleAPIError(
                                    Exception("degenerate repetition detected"),
                                    "Request timed out: degenerate repetitive generation detected",
                                )

                            if expect_structured_output and ("{" in content or "[" in content):
                                if (
                                    chunk_text.endswith("}")
                                    or chunk_text.endswith("]")
                                    or "thought:" in content.lower()
                                ):
                                    try:
                                        repaired = _extract_json_from_text(content)
                                        json.loads(repaired)
                                        content = repaired
                                        finish_reason = finish_reason or "stop"
                                        break
                                    except Exception:
                                        pass

                        if chunk.candidates and chunk.candidates[0].finish_reason:
                            finish_reason = self_ref._map_finish_reason(chunk.candidates[0].finish_reason)

                except StreamInactivityTimeout as exc:
                    logger.warning(
                        f"{LogColors.YELLOW}[StreamGuard] {exc}{LogColors.END}",
                        tag="llm_messages",
                    )
                    raise GoogleAPIError(Exception("stream inactivity hang"), str(exc)) from exc

                if GOOGLE_SETTINGS.log_llm_chat_content and GOOGLE_SETTINGS.show_llm_prompts:
                    logger.info("\n", raw=True, tag="llm_messages")

                return content, finish_reason

            return self._run_with_timeout(
                lambda: _consume_stream(self),
                self._request_timeout_seconds(),
                "stream generation",
                model_name,
                attempt_number,
            )

    def _non_stream_response(
        self,
        client: genai.Client,
        model_name: str,
        contents: list[Dict[str, Any]],
        generation_config: genai_types.GenerateContentConfig,
        attempt_number: int = 1,
    ) -> tuple[str, str | None]:

        def _call_once() -> tuple[str, str | None]:
            start_time = time.time()
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
            logger.info(
                _llm_status_banner(
                    "start",
                    f"{start_time_str} | non-stream call to {model_name}",
                    LogColors.CYAN,
                ),
                tag="llm_messages",
            )

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
            )

            elapsed = time.time() - start_time
            now_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger.info(
                f"[{now_time_str}] API response received in {elapsed:.2f}s "
                f"({len(self._extract_text_from_response(response))} chars)",
                tag="llm_messages",
            )

            content = ""
            finish_reason: str | None = None

            if response.candidates:
                candidate = response.candidates[0]
                content = self._extract_text_from_response(response)
                if candidate.finish_reason:
                    finish_reason = self._map_finish_reason(candidate.finish_reason)
            else:
                content = ""
                finish_reason = "content_filter"
                logger.warning(
                    f"{LogColors.RED}Response blocked or empty. "
                    f"Prompt feedback: {getattr(response, 'prompt_feedback', 'N/A')}{LogColors.END}",
                    tag="llm_messages",
                )

            if GOOGLE_SETTINGS.log_llm_chat_content and GOOGLE_SETTINGS.show_llm_prompts:
                fr_str = (
                    f"{LogColors.RED}finish={finish_reason}{LogColors.END} "
                    if finish_reason and finish_reason != "stop"
                    else ""
                )
                logger.info(
                    f"assistant: {fr_str}\n{content}",
                    tag="llm_messages",
                )

            return content, finish_reason

        return self._run_with_timeout(
            _call_once,
            self._request_timeout_seconds(),
            "non-stream generation",
            model_name,
            attempt_number,
        )

    def _log_cost_and_tokens(
        self,
        model_name: str,
        messages: list[dict[str, Any]],
        content: str,
        finish_reason: str | None,
    ) -> None:
        global ACC_COST

        prompt_tokens = 0
        completion_tokens = 0
        cost: float | None = None
        try:
            prompt_tokens = self._calculate_token_from_messages(messages)
            try:
                client = _get_client()
                completion_tokens = client.models.count_tokens(
                    model=model_name, contents=content
                ).total_tokens
            except Exception:
                completion_tokens = max(1, len(content) // 4)

            cost = _calculate_cost(model_name, prompt_tokens, completion_tokens)
            if cost is not None and np.isfinite(cost):
                ACC_COST += cost
                if GOOGLE_SETTINGS.log_llm_chat_content:
                    logger.info(
                        f"Cost: ${cost:.10f}  Accumulated: ${ACC_COST:.10f}  "
                        f"finish={finish_reason}",
                    )
        except Exception as exc:
            logger.warning(f"Cost calculation failed for {model_name}: {exc}. Skipping.")

        logger.log_object(
            {
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost if cost is not None else float("nan"),
                "accumulated_cost": ACC_COST,
            },
            tag="token_cost",
        )

    @staticmethod
    def _map_finish_reason(google_reason: Any) -> str:
        reason_str = str(google_reason).strip().lower()
        if not reason_str:
            return "stop"
        if "." in reason_str:
            reason_str = reason_str.split(".")[-1]
        if reason_str in {"stop", "unspecified", "finish_reason_unspecified"}:
            return "stop"
        if "max_tokens" in reason_str or reason_str in {"length", "token_limit", "max_tokens"}:
            return "length"
        if "safety" in reason_str or "recitation" in reason_str:
            return "content_filter"
        if reason_str == "other":
            return "stop"
        return reason_str

    @staticmethod
    def _extract_text_from_response(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            return text
        if getattr(response, "candidates", None):
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                parts: list[str] = []
                for part in candidate.content.parts:
                    ptxt = getattr(part, "text", None)
                    if isinstance(ptxt, str) and ptxt:
                        parts.append(ptxt)
                return "".join(parts)
        return ""