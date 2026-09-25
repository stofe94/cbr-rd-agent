"""
dynamic_tokens.py — Per-call max_output_tokens estimation for structured output.
 
Problem
-------
A single STRUCTURED_MAX_TOKENS env var is wrong for a DS pipeline that mixes:
  - Pure JSON metadata / configs     →  ~500–1500 tokens
  - JSON with embedded Python code   →  ~3000–7000 tokens
 
Setting it high (8192) increases hang probability on small calls.
Setting it low (2048) truncates code-heavy responses.
 
Solution
--------
Inspect the Pydantic schema and the last user message to estimate the
appropriate ceiling per call, then clamp to a safe maximum.
 
Usage
-----
Replace the static token assignment in _create_chat_completion_inner_function:
 
    # OLD
    gen_kwargs["max_output_tokens"] = int(os.environ.get(
        "GOOGLE_AI_STUDIO_STRUCTURED_MAX_TOKENS", "2048"
    ))
 
    # NEW
    from dynamic_tokens import estimate_structured_max_tokens
    gen_kwargs["max_output_tokens"] = estimate_structured_max_tokens(
        response_format=response_format,
        messages=effective_messages,
    )
"""
from __future__ import annotations
 
import json
import logging
import os
import re
from typing import Any, Type
 
from pydantic import BaseModel
 
log = logging.getLogger(__name__)
 
# ── Hard limits ────────────────────────────────────────────────────────────────
# Never go below this even for trivial schemas — schema overhead, thinking tags, etc.
_FLOOR_TOKENS = 512
 
# Never exceed this regardless of estimation — Gemma 4 hang probability rises
# sharply above this point. 6144 = 6K, covers even large code generations.
_CEILING_TOKENS = int(os.environ.get("GOOGLE_AI_STUDIO_STRUCTURED_MAX_TOKENS_CEILING", "6144"))
 
# Minimum for calls that look like they contain code
_CODE_FLOOR_TOKENS = int(os.environ.get("GOOGLE_AI_STUDIO_STRUCTURED_CODE_FLOOR_TOKENS", "3072"))
 
# Schema analysis ──────────────────────────────────────────────────────────────
_CODE_FIELD_NAMES = re.compile(
    r"\b(code|script|source|implementation|solution|program|function|method|"
    r"snippet|pipeline|module|file_content|content|body|text)\b",
    re.IGNORECASE,
)
 
_CODE_DESCRIPTIONS = re.compile(
    r"(python|code|script|implementation|source code|program)",
    re.IGNORECASE,
)
 
# Message context analysis ─────────────────────────────────────────────────────
_CODE_REQUEST_MARKERS = re.compile(
    r"\b(implement|write|generate|create|code|script|function|class|"
    r"pipeline|solution|program|refactor|fix|debug|modify)\b",
    re.IGNORECASE,
)
 
 
def _schema_has_code_fields(schema: dict) -> bool:
    """Return True if the schema has any field that looks like it holds code."""
    schema_str = json.dumps(schema)
 
    # Check field names
    if _CODE_FIELD_NAMES.search(schema_str):
        return True
 
    # Check descriptions
    if _CODE_DESCRIPTIONS.search(schema_str):
        return True
 
    return False
 
 
def _schema_field_count(schema: dict) -> int:
    """Count total number of properties across the whole schema (including nested)."""
    count = 0
    stack = [schema]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if "properties" in node:
            count += len(node["properties"])
            stack.extend(node["properties"].values())
        for key in ("items", "additionalProperties"):
            if key in node:
                stack.append(node[key])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in node:
                stack.extend(node[key])
    return count
 
 
def _schema_has_arrays(schema: dict) -> bool:
    return '"array"' in json.dumps(schema)
 
 
def _last_user_message_looks_like_code_request(messages: list[dict]) -> bool:
    """Check if the most recent user message is asking for code generation."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "") or ""
            return bool(_CODE_REQUEST_MARKERS.search(content[:2000]))
    return False
 
 
def _estimate_from_schema(schema: dict) -> int:
    """
    Base estimate purely from schema structure.
 
    Heuristics:
    - Each field → ~150 tokens average JSON key+value overhead
    - Arrays → multiply by an assumed 3-item average
    - Code fields → jump to code floor immediately
    - Nested objects → add 20% overhead
    """
    has_code   = _schema_has_code_fields(schema)
    has_arrays = _schema_has_arrays(schema)
    field_count = _schema_field_count(schema)
 
    if has_code:
        # Code fields dominate; base is the code floor, add per-field overhead
        base = _CODE_FLOOR_TOKENS
    else:
        base = _FLOOR_TOKENS + field_count * 150
 
    if has_arrays and not has_code:
        base = int(base * 2.0)   # arrays can hold multiple items
 
    return base
 
 
def estimate_structured_max_tokens(
    response_format: Any,
    messages:        list[dict] | None = None,
) -> int:
    """
    Estimate max_output_tokens for a structured output call.
 
    Parameters
    ----------
    response_format : Pydantic class, dict schema, or {"type": "json_object"}
    messages        : full message list (used to detect code-generation requests)
 
    Returns
    -------
    int — token ceiling for this specific call, clamped to [FLOOR, CEILING]
    """
    schema: dict | None = None
 
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        try:
            schema = response_format.model_json_schema()
        except Exception:
            schema = None
    elif isinstance(response_format, dict):
        if response_format.get("type") == "json_object":
            # Generic JSON object — unknown shape, use a safe middle ground
            schema = None
        else:
            schema = response_format
 
    # ── Schema-based estimate ─────────────────────────────────────────────────
    if schema is not None:
        schema_estimate = _estimate_from_schema(schema)
    else:
        schema_estimate = 2048   # unknown schema → conservative default
 
    # ── Context-based adjustment ──────────────────────────────────────────────
    context_looks_like_code = (
        messages is not None
        and _last_user_message_looks_like_code_request(messages)
    )
 
    if context_looks_like_code and schema_estimate < _CODE_FLOOR_TOKENS:
        log.info(
            "[DynTokens] User message looks like code request; "
            "bumping estimate from %d → %d",
            schema_estimate, _CODE_FLOOR_TOKENS,
        )
        schema_estimate = _CODE_FLOOR_TOKENS
 
    # ── Apply ceiling and floor ───────────────────────────────────────────────
    result = max(_FLOOR_TOKENS, min(_CEILING_TOKENS, schema_estimate))
 
    log.info(
        "[DynTokens] response_format=%s  schema_estimate=%d  "
        "code_context=%s  final=%d  (ceiling=%d)",
        getattr(response_format, "__name__", type(response_format).__name__),
        schema_estimate,
        context_looks_like_code,
        result,
        _CEILING_TOKENS,
    )
 
    return result
 
 
# ── Convenience: patch the env-var fallback ───────────────────────────────────
 
def get_structured_max_tokens_env_fallback() -> int:
    """
    Use this as the fallback when response_format is not available.
    Reads GOOGLE_AI_STUDIO_STRUCTURED_MAX_TOKENS but clamps to ceiling.
    """
    raw = int(os.environ.get("GOOGLE_AI_STUDIO_STRUCTURED_MAX_TOKENS", "4096"))
    return max(_FLOOR_TOKENS, min(_CEILING_TOKENS, raw))