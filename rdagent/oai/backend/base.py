from __future__ import annotations

import io
import json
import re
import sqlite3
import time
import tokenize
import uuid
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, List, Optional, Tuple, Type, Union, cast

from pydantic import BaseModel, TypeAdapter

from rdagent.core.exception import CodeBlockParseError, PolicyError
from rdagent.core.utils import LLM_CACHE_SEED_GEN, SingletonBaseClass
from rdagent.log import LogColors
from rdagent.log import rdagent_logger as logger
from rdagent.log.timer import RD_Agent_TIMER_wrapper
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.utils.embedding import truncate_content_list
from rdagent.utils import md5_hash

try:
    import litellm
    import openai

    openai_imported = True
except ImportError:
    openai_imported = False


class JSONParser:
    """JSON parser supporting multiple strategies"""

    def __init__(self, add_json_in_prompt: bool = False) -> None:
        self.strategies: List[Callable[[str], str]] = [
            self._direct_parse,
            self._extract_from_code_block,
            self._extract_from_any_code_block,
            self._extract_json_from_anywhere,
            self._fix_python_syntax,
            self._extract_with_fix_combined,
        ]
        self.add_json_in_prompt = add_json_in_prompt

    def parse(self, content: str) -> str:
        """Parse JSON content, automatically trying multiple strategies"""
        original_content = content

        for strategy in self.strategies:
            try:
                return strategy(original_content)
            except json.JSONDecodeError:
                continue

        # All strategies failed
        if not self.add_json_in_prompt:
            error = json.JSONDecodeError(
                "Failed to parse JSON after all attempts, maybe because 'messages' must contain the word 'json' in some form",
                original_content,
                0,
            )
            error.message = "Failed to parse JSON after all attempts, maybe because 'messages' must contain the word 'json' in some form"  # type: ignore[attr-defined]
            raise error
        else:
            raise json.JSONDecodeError("Failed to parse JSON after all attempts", original_content, 0)

    def _direct_parse(self, content: str) -> str:
        """Strategy 1: Direct parsing (including handling extra data)"""
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                return self._extract_first_json(content)
            raise

    def _extract_from_code_block(self, content: str) -> str:
        """Strategy 2: Extract JSON from code block"""
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON code block found", content, 0)

        json_content = match.group(1).strip()
        return self._direct_parse(json_content)

    def _extract_from_any_code_block(self, content: str) -> str:
        """Strategy 3: Extract JSON from any fenced markdown code block."""
        match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", content, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No code block found", content, 0)

        block_content = match.group(1).strip()
        return self._direct_parse(block_content)

    def _fix_python_syntax(self, content: str) -> str:
        """Strategy 4: Fix Python syntax before parsing"""
        fixed = self._fix_python_booleans(content)
        return self._direct_parse(fixed)

    def _extract_with_fix_combined(self, content: str) -> str:
        """Strategy 5: Combined strategy - fix Python syntax first, then extract the first JSON object"""
        fixed = self._fix_python_booleans(content)

        # Try to extract code block from the fixed content
        match = re.search(r"```json\s*(.*?)\s*```", fixed, re.DOTALL)
        if match:
            fixed = match.group(1).strip()

        return self._direct_parse(fixed)

    @staticmethod
    def _fix_python_booleans(json_str: str) -> str:
        """Safely fix Python-style booleans to JSON standard format using tokenize"""
        replacements = {"True": "true", "False": "false", "None": "null"}

        try:
            out = []
            io_string = io.StringIO(json_str)
            tokens = tokenize.generate_tokens(io_string.readline)

            for toknum, tokval, _, _, _ in tokens:
                if toknum == tokenize.NAME and tokval in replacements:
                    out.append(replacements[tokval])
                else:
                    out.append(tokval)

            result = "".join(out)
            return result

        except (tokenize.TokenError, json.JSONDecodeError):
            # If tokenize fails, fallback to regex method
            for python_val, json_val in replacements.items():
                json_str = re.sub(rf"\b{python_val}\b", json_val, json_str)
            return json_str

    @staticmethod
    def _extract_first_json(response: str) -> str:
        """Extract complete JSON payload from mixed output.

        If multiple JSON values are concatenated, return the last one because
        providers often emit a corrected JSON object after an initial draft.
        """
        decoder = json.JSONDecoder()
        idx = 0
        last_obj: Any = None
        length = len(response)

        while idx < length:
            # Skip non-JSON leading characters before each decode attempt.
            while idx < length and response[idx] not in "[{\"-0123456789tfn":
                idx += 1
            if idx >= length:
                break
            try:
                obj, end = decoder.raw_decode(response[idx:])
            except json.JSONDecodeError:
                idx += 1
                continue

            last_obj = obj
            idx += end

        if last_obj is None:
            raise json.JSONDecodeError("No JSON object found", response, 0)

        return json.dumps(last_obj)

    @staticmethod
    def _extract_json_from_anywhere(content: str) -> str:
        """Extract a JSON object or array from free-form text."""
        decoder = json.JSONDecoder()
        for index, char in enumerate(content):
            if char not in "[{":
                continue
            try:
                obj, _ = decoder.raw_decode(content[index:])
                return json.dumps(obj)
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("No JSON object found", content, 0)


class CodeBlockParser:
    """
    Generic code block extractor supporting multiple languages.
    Raises CodeBlockParseError on extraction failure to trigger retry.
    """

    SUPPORTED_LANGUAGES = {
        "python": ["python", "py", "python3", "Python", "Py"],
        "yaml": ["yaml", "yml"],
    }

    def __init__(self, language: str = "python", fallback_to_raw: bool = False) -> None:
        """
        Args:
            language: Target language type (python, yaml, etc.)
            fallback_to_raw: If True, return raw content when extraction fails.
                           If False (default), raise CodeBlockParseError to trigger retry.
        """
        self.language = language.lower()
        self.fallback_to_raw = fallback_to_raw
        self._lang_aliases = self._get_language_aliases(self.language)

    def _get_language_aliases(self, language: str) -> List[str]:
        """Get all possible aliases for the language."""
        for lang, aliases in self.SUPPORTED_LANGUAGES.items():
            if language in [lang] + aliases:
                return [lang] + aliases
        return [language]

    def parse(self, content: str) -> str:
        """
        Parse content and extract code block with exact language tag.

        Returns:
            Extracted code string.

        Raises:
            CodeBlockParseError: When extraction fails and fallback_to_raw=False.
        """
        # Match code block with exact language tag (```python, ```yaml, etc.)
        for alias in self._lang_aliases:
            pattern = rf"```{alias}\s*\n(.*?)\n```"
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if self.fallback_to_raw:
            return content.strip()

        raise CodeBlockParseError(
            message=f"Failed to extract {self.language} code block",
            content=content,
            language=self.language,
        )


CONTINUE_PROMPT = (
    "Continue exactly from where you stopped. Do not repeat, restate, or summarize any previous content. "
    "Output only the remaining continuation."
)


class SQliteLazyCache(SingletonBaseClass):
    def __init__(self, cache_location: str) -> None:
        super().__init__()
        self.cache_location = cache_location
        db_file_exist = Path(cache_location).exists()
        # TODO: sqlite3 does not support multiprocessing.
        self.conn = sqlite3.connect(cache_location, timeout=20)
        self.c = self.conn.cursor()
        if not db_file_exist:
            self.c.execute(
                """
                CREATE TABLE chat_cache (
                    md5_key TEXT PRIMARY KEY,
                    chat TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE embedding_cache (
                    md5_key TEXT PRIMARY KEY,
                    embedding TEXT
                )
                """,
            )
            self.c.execute(
                """
                CREATE TABLE message_cache (
                    conversation_id TEXT PRIMARY KEY,
                    message TEXT
                )
                """,
            )
            self.conn.commit()

    def chat_get(self, key: str) -> str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT chat FROM chat_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        return None if result is None else result[0]

    def embedding_get(self, key: str) -> list | dict | str | None:
        md5_key = md5_hash(key)
        self.c.execute("SELECT embedding FROM embedding_cache WHERE md5_key=?", (md5_key,))
        result = self.c.fetchone()
        return None if result is None else json.loads(result[0])

    def chat_set(self, key: str, value: str) -> None:
        md5_key = md5_hash(key)
        self.c.execute(
            "INSERT OR REPLACE INTO chat_cache (md5_key, chat) VALUES (?, ?)",
            (md5_key, value),
        )
        self.conn.commit()
        return None

    def embedding_set(self, content_to_embedding_dict: dict) -> None:
        for key, value in content_to_embedding_dict.items():
            md5_key = md5_hash(key)
            self.c.execute(
                "INSERT OR REPLACE INTO embedding_cache (md5_key, embedding) VALUES (?, ?)",
                (md5_key, json.dumps(value)),
            )
        self.conn.commit()

    def message_get(self, conversation_id: str) -> list[dict[str, Any]]:
        self.c.execute("SELECT message FROM message_cache WHERE conversation_id=?", (conversation_id,))
        result = self.c.fetchone()
        return [] if result is None else cast(list[dict[str, Any]], json.loads(result[0]))

    def message_set(self, conversation_id: str, message_value: list[dict[str, Any]]) -> None:
        self.c.execute(
            "INSERT OR REPLACE INTO message_cache (conversation_id, message) VALUES (?, ?)",
            (conversation_id, json.dumps(message_value)),
        )
        self.conn.commit()
        return None


class SessionChatHistoryCache(SingletonBaseClass):
    def __init__(self) -> None:
        """load all history conversation json file from self.session_cache_location"""
        self.cache = SQliteLazyCache(cache_location=LLM_SETTINGS.prompt_cache_path)

    def message_get(self, conversation_id: str) -> list[dict[str, Any]]:
        return self.cache.message_get(conversation_id)

    def message_set(self, conversation_id: str, message_value: list[dict[str, Any]]) -> None:
        self.cache.message_set(conversation_id, message_value)


class ChatSession:
    def __init__(self, api_backend: Any, conversation_id: str | None = None, system_prompt: str | None = None) -> None:
        self.conversation_id = str(uuid.uuid4()) if conversation_id is None else conversation_id
        self.system_prompt = system_prompt if system_prompt is not None else LLM_SETTINGS.default_system_prompt
        self.api_backend = api_backend

    def build_chat_completion_message(self, user_prompt: str) -> list[dict[str, Any]]:
        history_message = SessionChatHistoryCache().message_get(self.conversation_id)
        messages = history_message
        if not messages:
            messages.append({"role": LLM_SETTINGS.system_prompt_role, "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def build_chat_completion_message_and_calculate_token(self, user_prompt: str) -> Any:
        messages = self.build_chat_completion_message(user_prompt)
        return self.api_backend._calculate_token_from_messages(messages)

    def build_chat_completion(self, user_prompt: str, *args, **kwargs) -> str:  # type: ignore[no-untyped-def]
        """
        this function is to build the session messages
        user prompt should always be provided
        """
        messages = self.build_chat_completion_message(user_prompt)

        with logger.tag(f"session_{self.conversation_id}"):
            start_time = datetime.now().astimezone()
            response: str = self.api_backend._try_create_chat_completion_or_embedding(  # noqa: SLF001
                *args,
                messages=messages,
                chat_completion=True,
                **kwargs,
            )
            end_time = datetime.now().astimezone()
            logger.log_object(
                {
                    "system": self.system_prompt,
                    "user": user_prompt,
                    "resp": response,
                    "start": start_time,
                    "end": end_time,
                },
                tag="debug_llm",
            )

        messages.append(
            {
                "role": "assistant",
                "content": response,
            },
        )
        SessionChatHistoryCache().message_set(self.conversation_id, messages)
        return response

    def get_conversation_id(self) -> str:
        return self.conversation_id

    def display_history(self) -> None:
        # TODO: Realize a beautiful presentation format for history messages
        pass


class APIBackend(ABC):
    """
    Abstract base class for LLM API backends
    supporting auto retry, cache and auto continue
    Inner api call should be implemented in the subclass
    """

    _tpm_lock: Lock = Lock()
    _tpm_usage_window: deque[tuple[float, int]] = deque()

    def __init__(
        self,
        use_chat_cache: bool | None = None,
        dump_chat_cache: bool | None = None,
        use_embedding_cache: bool | None = None,
        dump_embedding_cache: bool | None = None,
    ):
        self.dump_chat_cache = LLM_SETTINGS.dump_chat_cache if dump_chat_cache is None else dump_chat_cache
        self.use_chat_cache = LLM_SETTINGS.use_chat_cache if use_chat_cache is None else use_chat_cache
        self.dump_embedding_cache = (
            LLM_SETTINGS.dump_embedding_cache if dump_embedding_cache is None else dump_embedding_cache
        )
        self.use_embedding_cache = (
            LLM_SETTINGS.use_embedding_cache if use_embedding_cache is None else use_embedding_cache
        )
        if self.dump_chat_cache or self.use_chat_cache or self.dump_embedding_cache or self.use_embedding_cache:
            self.cache_file_location = LLM_SETTINGS.prompt_cache_path
            self.cache = SQliteLazyCache(cache_location=self.cache_file_location)

        self.retry_wait_seconds = LLM_SETTINGS.retry_wait_seconds

    def build_chat_session(
        self,
        conversation_id: str | None = None,
        session_system_prompt: str | None = None,
    ) -> ChatSession:
        """
        conversation_id is a 256-bit string created by uuid.uuid4() and is also
        the file name under session_cache_folder/ for each conversation
        """
        return ChatSession(self, conversation_id, session_system_prompt)

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list[dict[str, Any]] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> list[dict[str, Any]]:
        """
        build the messages to avoid implementing several redundant lines of code

        """
        if former_messages is None:
            former_messages = []
        # shrink multiple break will recursively remove multiple breaks(more than 2)
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            if system_prompt is not None:
                while "\n\n\n" in system_prompt:
                    system_prompt = system_prompt.replace("\n\n\n", "\n\n")
        system_prompt = LLM_SETTINGS.default_system_prompt if system_prompt is None else system_prompt
        messages = [
            {
                "role": LLM_SETTINGS.system_prompt_role,
                "content": system_prompt,
            },
        ]
        messages.extend(former_messages[-1 * LLM_SETTINGS.max_past_message_include :])
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        return messages

    def _build_log_messages(self, messages: list[dict[str, Any]]) -> str:
        log_messages = ""
        for m in messages:
            log_messages += (
                f"\n{LogColors.MAGENTA}{LogColors.BOLD}Role:{LogColors.END}"
                f"{LogColors.CYAN}{m['role']}{LogColors.END}\n"
                f"{LogColors.MAGENTA}{LogColors.BOLD}Content:{LogColors.END} "
                f"{LogColors.CYAN}{m['content']}{LogColors.END}\n"
            )
        return log_messages

    def build_messages_and_create_chat_completion(  # type: ignore[no-untyped-def]
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        former_messages: list | None = None,
        chat_cache_prefix: str = "",
        shrink_multiple_break: bool = False,
        *args,
        **kwargs,
    ) -> str:
        """
        Responseible for building messages and logging messages

        TODO: What is weird is that the function is called before we seperate embeddings and chat completion.

        Parameters
        ----------
        user_prompt : str
        system_prompt : str | None
        former_messages : list | None
        response_format : BaseModel | dict
            A BaseModel based on pydantic or a dict
        **kwargs
        Returns
        -------
        str
        """
        if former_messages is None:
            former_messages = []
        messages = self._build_messages(
            user_prompt,
            system_prompt,
            former_messages,
            shrink_multiple_break=shrink_multiple_break,
        )

        start_time = datetime.now().astimezone()
        resp = self._try_create_chat_completion_or_embedding(  # type: ignore[misc]
            *args,
            messages=messages,
            chat_completion=True,
            chat_cache_prefix=chat_cache_prefix,
            **kwargs,
        )
        end_time = datetime.now().astimezone()
        if isinstance(resp, list):
            raise ValueError("The response of _try_create_chat_completion_or_embedding should be a string.")
        logger.log_object(
            {"system": system_prompt, "user": user_prompt, "resp": resp, "start": start_time, "end": end_time},
            tag="debug_llm",
        )
        return resp

    def create_embedding(self, input_content: str | list[str], *args, **kwargs) -> list[float] | list[list[float]]:  # type: ignore[no-untyped-def]
        input_content_list = [input_content] if isinstance(input_content, str) else input_content
        resp = self._try_create_chat_completion_or_embedding(  # type: ignore[misc]
            input_content_list=input_content_list,
            embedding=True,
            *args,
            **kwargs,
        )
        if isinstance(input_content, str):
            return resp[0]  # type: ignore[return-value]
        return resp  # type: ignore[return-value]

    def build_messages_and_calculate_token(
        self,
        user_prompt: str,
        system_prompt: str | None,
        former_messages: list[dict[str, Any]] | None = None,
        *,
        shrink_multiple_break: bool = False,
    ) -> int:
        if former_messages is None:
            former_messages = []
        messages = self._build_messages(
            user_prompt, system_prompt, former_messages, shrink_multiple_break=shrink_multiple_break
        )
        return self._calculate_token_from_messages(messages)

    def _try_create_chat_completion_or_embedding(  # type: ignore[no-untyped-def]
        self,
        max_retry: int = 10,
        chat_completion: bool = False,
        embedding: bool = False,
        *args,
        **kwargs,
    ) -> str | list[list[float]]:
        """This function to share operation between embedding and chat completion"""

        def _extract_retry_delay_seconds(raw_error: str) -> float:
            lowered = raw_error.lower()
            patterns = [
                r"please retry (?:after|in)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:seconds?|s)",
                r'"retrydelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"',
            ]
            for pattern in patterns:
                match = re.search(pattern, lowered)
                if match:
                    return float(match.group(1))
            return float(self.retry_wait_seconds)

        def _shrink_messages_to_tpm_budget(
            message_list: list[dict[str, Any]],
            target_tokens: int,
        ) -> tuple[list[dict[str, Any]], bool]:
            updated = deepcopy(message_list)
            try:
                current_tokens = self._calculate_token_from_messages(updated)
            except Exception:
                return updated, False

            if current_tokens <= target_tokens:
                return updated, False

            # Prefer dropping older context first, preserving system + latest user ask.
            while current_tokens > target_tokens and len(updated) > 2:
                drop_idx = next(
                    (
                        idx
                        for idx, msg in enumerate(updated[:-1])
                        if msg.get("role") != LLM_SETTINGS.system_prompt_role
                    ),
                    None,
                )
                if drop_idx is None:
                    break
                del updated[drop_idx]
                try:
                    current_tokens = self._calculate_token_from_messages(updated)
                except Exception:
                    break

            if current_tokens <= target_tokens:
                return updated, True

            # If still too large, truncate the latest textual message.
            for idx in range(len(updated) - 1, -1, -1):
                content = updated[idx].get("content")
                if isinstance(content, str) and content:
                    ratio = max(0.1, min(0.95, target_tokens / max(current_tokens, 1)))
                    keep_len = max(512, int(len(content) * ratio))
                    if keep_len < len(content):
                        updated[idx]["content"] = content[:keep_len] + "\n\n[Prompt truncated due to TPM budget]"
                        return updated, True
            return updated, False

        def _apply_proactive_tpm_shrink() -> None:
            if not chat_completion or not LLM_SETTINGS.chat_auto_shrink_for_tpm:
                return
            if not isinstance(kwargs.get("messages"), list):
                return

            tpm_limit = LLM_SETTINGS.chat_tpm_limit
            if tpm_limit is None or tpm_limit <= 0:
                return

            strategy = LLM_SETTINGS.chat_tpm_strategy
            request_tokens: int | None = None
            try:
                request_tokens = self._calculate_token_from_messages(kwargs["messages"])
            except Exception:
                request_tokens = None

            if strategy == "wait" and request_tokens is not None:
                window_seconds = max(1, int(LLM_SETTINGS.chat_tpm_window_seconds))

                # A single oversized request cannot be fixed by waiting.
                if request_tokens > tpm_limit:
                    if not LLM_SETTINGS.chat_allow_tpm_shrink:
                        raise RuntimeError(
                            "Single request token count exceeds configured TPM limit and prompt shrinking is disabled. "
                            "Increase CHAT_TPM_LIMIT or allow shrinking."
                        )
                    logger.warning(
                        "Single request token count exceeds configured TPM limit. "
                        "Switching to shrink strategy for this request."
                    )
                    strategy = "shrink"
                else:
                    while True:
                        now_ts = time.time()
                        with self._tpm_lock:
                            cutoff = now_ts - window_seconds
                            while self._tpm_usage_window and self._tpm_usage_window[0][0] < cutoff:
                                self._tpm_usage_window.popleft()

                            used_tokens = sum(tokens for _, tokens in self._tpm_usage_window)
                            if used_tokens + request_tokens <= tpm_limit:
                                self._tpm_usage_window.append((now_ts, request_tokens))
                                logger.info(
                                    "TPM pacing accepted request "
                                    f"({used_tokens}+{request_tokens}<={tpm_limit} tokens/{window_seconds}s)."
                                )
                                return

                            wait_seconds = self._tpm_usage_window[0][0] + window_seconds - now_ts

                        wait_seconds = max(wait_seconds, 0.2)
                        logger.warning(
                            f"TPM budget full ({used_tokens}/{tpm_limit}). "
                            f"Sleeping {wait_seconds:.1f}s to preserve full prompt."
                        )
                        time.sleep(wait_seconds)

            if strategy != "shrink" or not LLM_SETTINGS.chat_allow_tpm_shrink:
                return

            ratio = min(max(LLM_SETTINGS.chat_tpm_target_ratio, 0.1), 0.98)
            target_budget = max(1000, int(tpm_limit * ratio))
            shrunk_messages, shrunk = _shrink_messages_to_tpm_budget(kwargs["messages"], target_budget)
            if shrunk:
                kwargs["messages"] = shrunk_messages
                logger.warning(
                    f"Proactively shrank chat messages to fit TPM budget (target <= {target_budget} tokens)."
                )

        assert not (chat_completion and embedding), "chat_completion and embedding cannot be True at the same time"
        max_retry = LLM_SETTINGS.max_retry if LLM_SETTINGS.max_retry is not None else max_retry
        timeout_count = 0
        violation_count = 0
        embedding_truncated = False  # Track if we've already tried truncation
        timeout_warning_logged = False  # Track if we've already logged the timeout warning
        last_exception: Exception | None = None
        last_error_message: str | None = None
        _apply_proactive_tpm_shrink()
        for i in range(max_retry):
            API_start_time = datetime.now()
            try:
                if embedding:
                    return self._create_embedding_with_cache(*args, **kwargs)
                if chat_completion:
                    return self._create_chat_completion_auto_continue(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                last_exception = e
                error_message = str(getattr(e, "message", "") or e)
                last_error_message = error_message
                lowered_error_message = error_message.lower()
                did_inline_rate_limit_retry = False

                if chat_completion and "json mode is not enabled" in error_message.lower():
                    logger.warning(
                        "JSON mode is not supported by the current model/provider. "
                        "Retrying with prompt-based JSON guidance instead."
                    )
                    kwargs["response_format"] = None
                    kwargs["json_mode"] = False
                    kwargs["add_json_in_prompt"] = True
                    try:
                        return self._create_chat_completion_auto_continue(*args, **kwargs)
                    except Exception as fallback_error:  # noqa: BLE001
                        e = fallback_error
                        error_message = str(getattr(e, "message", "") or e)

                if hasattr(e, "message") and (
                    "'messages' must contain the word 'json' in some form" in e.message
                    or "\\'messages\\' must contain the word \\'json\\' in some form" in e.message
                ):
                    kwargs["add_json_in_prompt"] = True

                is_rate_limit_error = (
                    (openai_imported and isinstance(e, openai.RateLimitError))
                    or "rate limit" in lowered_error_message
                    or "resource_exhausted" in lowered_error_message
                    or "quota exceeded" in lowered_error_message
                    or '"code": 429' in lowered_error_message
                )

                if chat_completion and is_rate_limit_error:
                    quota_match = re.search(r"input_token_count[^\n]*limit:\s*(\d+)", lowered_error_message)
                    if quota_match and isinstance(kwargs.get("messages"), list) and LLM_SETTINGS.chat_allow_tpm_shrink:
                        quota_limit = int(quota_match.group(1))
                        target_budget = max(1000, int(quota_limit * 0.85))
                        shrunk_messages, shrunk = _shrink_messages_to_tpm_budget(kwargs["messages"], target_budget)
                        if shrunk:
                            kwargs["messages"] = shrunk_messages
                            logger.warning(
                                f"Shrank chat messages to fit provider TPM budget (target <= {target_budget} tokens)."
                            )

                    for inline_retry_index in range(3):
                        retry_delay_seconds = _extract_retry_delay_seconds(error_message)
                        # Add a tiny buffer to reduce immediate re-throttle risk.
                        retry_delay_seconds = max(retry_delay_seconds, 1.0) + 0.2
                        logger.warning(
                            f"Rate limit/quota reached. Sleeping {retry_delay_seconds:.1f}s "
                            f"before inline retry {inline_retry_index + 1}/3."
                        )
                        time.sleep(retry_delay_seconds)
                        try:
                            did_inline_rate_limit_retry = True
                            return self._create_chat_completion_auto_continue(*args, **kwargs)
                        except Exception as rate_limit_retry_error:  # noqa: BLE001
                            e = rate_limit_retry_error
                            error_message = str(getattr(e, "message", "") or e)
                            lowered_error_message = error_message.lower()
                            if not (
                                "rate limit" in lowered_error_message
                                or "resource_exhausted" in lowered_error_message
                                or "quota exceeded" in lowered_error_message
                                or '"code": 429' in lowered_error_message
                            ):
                                break

                too_long_error_message = hasattr(e, "message") and (
                    "maximum context length" in e.message or "input must have less than" in e.message
                )

                if embedding and too_long_error_message:
                    if not embedding_truncated:
                        # Handle embedding text too long error - truncate once and retry
                        model_name = LLM_SETTINGS.embedding_model
                        logger.warning(f"Embedding text too long for model {model_name}, truncating content")

                        # Apply truncation to content list and continue to retry
                        original_content_list = kwargs.get("input_content_list", [])
                        kwargs["input_content_list"] = truncate_content_list(original_content_list, model_name)
                        embedding_truncated = True  # Mark that we've tried truncation
                        # Continue to next iteration to retry embedding with truncated content
                    else:
                        # Already tried truncation, raise error with guidance
                        raise RuntimeError(
                            f"Embedding failed even after truncation. "
                            f"Please set LLM_SETTINGS.embedding_max_length to a smaller value."
                        ) from e
                else:
                    RD_Agent_TIMER_wrapper.api_fail_count += 1
                    RD_Agent_TIMER_wrapper.latest_api_fail_time = datetime.now().astimezone()

                    if (
                        openai_imported
                        and isinstance(e, litellm.BadRequestError)
                        and (
                            isinstance(e.__cause__, litellm.ContentPolicyViolationError)
                            or "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"
                            in str(e)
                        )
                    ):
                        violation_count += 1
                        if violation_count >= LLM_SETTINGS.violation_fail_limit:
                            logger.warning("Content policy violation detected.")
                            raise PolicyError(e)

                    if (
                        openai_imported
                        and isinstance(e, openai.APITimeoutError)
                        or (
                            isinstance(e, openai.APIError)
                            and hasattr(e, "message")
                            and "Your resource has been temporarily blocked because we detected behavior that may violate our content policy."
                            in e.message
                        )
                    ):
                        timeout_count += 1
                        if timeout_count >= LLM_SETTINGS.timeout_fail_limit:
                            logger.warning("Timeout error, please check your network connection.")
                            raise e

                    recommended_wait_seconds = self.retry_wait_seconds
                    if openai_imported and isinstance(e, openai.RateLimitError) and hasattr(e, "message"):
                        match = re.search(r"Please retry after (\d+) seconds\.", e.message)
                        if match:
                            recommended_wait_seconds = int(match.group(1))
                    if not did_inline_rate_limit_retry:
                        time.sleep(recommended_wait_seconds)
                    if RD_Agent_TIMER_wrapper.timer.started and not isinstance(e, json.decoder.JSONDecodeError):
                        RD_Agent_TIMER_wrapper.timer.add_duration(datetime.now() - API_start_time)
                
                # Only log exception message if it wasn't already logged (e.g., timeout errors logged in _run_with_timeout)
                if not (hasattr(e, "message") and "exceeded" in str(e) and "timeout" in str(e).lower()):
                    logger.warning(str(e))
                    logger.warning(f"Retrying {i+1}th time...")
        error_message = f"Failed to create chat completion after {max_retry} retries."
        if last_exception is not None:
            last_type = type(last_exception).__name__
            last_msg = (last_error_message or str(last_exception) or repr(last_exception)).strip()
            if last_msg:
                error_message += f" Last error ({last_type}): {last_msg}"
            else:
                error_message += f" Last error type: {last_type}"
        raise RuntimeError(error_message)

    def _add_json_in_prompt(self, messages: list[dict[str, Any]]) -> None:
        """
        add json related content in the prompt if add_json_in_prompt is True
        """
        for message in messages[::-1]:
            message["content"] = message["content"] + "\nPlease respond in json format."
            if message["role"] == LLM_SETTINGS.system_prompt_role:
                # NOTE: assumption: systemprompt is always the first message
                break

    def _create_chat_completion_auto_continue(
        self,
        messages: list[dict[str, Any]],
        json_mode: bool = False,
        chat_cache_prefix: str = "",
        seed: Optional[int] = None,
        json_target_type: Optional[str] = None,
        add_json_in_prompt: bool = False,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        code_block_language: Optional[str] = None,
        code_block_fallback: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Call the chat completion function and automatically continue the conversation if the finish_reason is length.
        """

        if response_format is None and json_mode:
            response_format = {"type": "json_object"}

        # 0) return directly if cache is hit
        if seed is None and LLM_SETTINGS.use_auto_chat_cache_seed_gen:
            seed = LLM_CACHE_SEED_GEN.get_next_seed()
        input_content_json = json.dumps(messages)
        input_content_json = (
            chat_cache_prefix + input_content_json + f"<seed={seed}/>"
        )  # FIXME this is a hack to make sure the cache represents the round index
        if self.use_chat_cache:
            cache_result = self.cache.chat_get(input_content_json)
            if cache_result is not None:
                if LLM_SETTINGS.log_llm_chat_content:
                    logger.info(self._build_log_messages(messages), tag="llm_messages")
                    logger.info(f"{LogColors.CYAN}Response:{cache_result}{LogColors.END}", tag="llm_messages")
                return cache_result

        # 1) get a full response
        all_response = ""
        new_messages = deepcopy(messages)
        # Loop to get a full response
        try_n = 6
        # Before retry loop, initialize the flag
        json_added = False
        for _ in range(try_n):  # for some long code, 3 times may not enough for reasoning models
            if response_format == {"type": "json_object"} and add_json_in_prompt and not json_added:
                self._add_json_in_prompt(new_messages)
                json_added = True
            response, finish_reason = self._create_chat_completion_inner_function(
                messages=new_messages,
                response_format=response_format,
                **kwargs,
            )
            all_response += response

            # Handle litellm bug: finish_reason='stop' but code block not closed
            # TODO: this is a temporary solution, and should be removed when litellm is fixed.
            if finish_reason == "stop" and code_block_language:
                if all_response.count("```") % 2 == 1:  # Odd count = unclosed code block
                    logger.warning("Detected unclosed code block with finish_reason='stop', treating as truncated")
                    finish_reason = "length"

            if finish_reason is None or finish_reason != "length":
                break  # we get a full response now.
            new_messages.append({"role": "assistant", "content": response})
            new_messages.append({"role": "user", "content": CONTINUE_PROMPT})
        else:
            raise RuntimeError(f"Failed to continue the conversation after {try_n} retries.")

        # 2) refine the response and return
        if LLM_SETTINGS.reasoning_think_rm:
            # Only remove <think>...</think> if it appears at the beginning of the response
            # Strategy 1: Try to match complete <think>...</think> pattern at the start
            match = re.match(r"\s*<think>(.*?)</think>(.*)", all_response, re.DOTALL)
            if match:
                _, all_response = match.groups()
            else:
                # Strategy 2: If no complete match, try to match only </think> at the start
                match = re.match(r"\s*</think>(.*)", all_response, re.DOTALL)
                if match:
                    all_response = match.group(1)
                # If no match at all, keep original content

        # 3) format checking
        if response_format == {"type": "json_object"} or json_target_type:
            parser = JSONParser(add_json_in_prompt=add_json_in_prompt)
            all_response = parser.parse(all_response)
            if json_target_type:
                # deepseek will enter this branch
                TypeAdapter(json_target_type).validate_json(all_response)
        elif add_json_in_prompt:
            # Best-effort normalization for providers without strict JSON mode.
            # If parsing fails, keep raw text and let caller-side handling decide.
            try:
                all_response = JSONParser(add_json_in_prompt=True).parse(all_response)
            except json.JSONDecodeError:
                pass

        # 4) code block extraction
        if code_block_language:
            code_parser = CodeBlockParser(
                language=code_block_language,
                fallback_to_raw=code_block_fallback,
            )
            all_response = code_parser.parse(all_response)

        if response_format is not None:
            if not isinstance(response_format, dict) and issubclass(response_format, BaseModel):
                # It may raise TypeError if initialization fails
                response_format(**json.loads(all_response))
            elif response_format == {"type": "json_object"}:
                logger.info(f"Using OpenAI response format: {response_format}")
            else:
                logger.warning(f"Unknown response_format: {response_format}, skipping validation.")
        if self.dump_chat_cache:
            self.cache.chat_set(input_content_json, all_response)
        return all_response

    def _create_embedding_with_cache(
        self, input_content_list: list[str], *args: Any, **kwargs: Any
    ) -> list[list[float]]:
        content_to_embedding_dict = {}
        filtered_input_content_list = []
        if self.use_embedding_cache:
            for content in input_content_list:
                cache_result = self.cache.embedding_get(content)
                if cache_result is not None:
                    content_to_embedding_dict[content] = cache_result
                else:
                    filtered_input_content_list.append(content)
        else:
            filtered_input_content_list = input_content_list

        if len(filtered_input_content_list) > 0:
            resp = self._create_embedding_inner_function(input_content_list=filtered_input_content_list)
            for index, data in enumerate(resp):
                content_to_embedding_dict[filtered_input_content_list[index]] = data
            if self.dump_embedding_cache:
                self.cache.embedding_set(content_to_embedding_dict)
        return [content_to_embedding_dict[content] for content in input_content_list]  # type: ignore[misc]

    @abstractmethod
    def supports_response_schema(self) -> bool:
        """
        Check if the backend supports function calling
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _calculate_token_from_messages(self, messages: list[dict[str, Any]]) -> int:
        """
        Calculate the token count from messages
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        """
        Call the embedding function
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def _create_chat_completion_inner_function(  # type: ignore[no-untyped-def] # noqa: C901, PLR0912, PLR0915
        self,
        messages: list[dict[str, Any]],
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        *args,
        **kwargs,
    ) -> tuple[str, str | None]:
        """
        Call the chat completion function
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def chat_token_limit(self) -> int:
        return LLM_SETTINGS.chat_token_limit
