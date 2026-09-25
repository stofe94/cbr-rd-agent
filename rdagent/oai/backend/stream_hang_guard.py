"""
stream_hang_guard.py — Inactivity-based hang detection for Google AI Studio streams.

The problem
-----------
Gemma 4-31b on long DS tasks legitimately streams for 3+ minutes — so a
wall-clock timeout kills valid requests.  A *hang* has a different signature:
chunks stop arriving entirely.  This module wraps the chunk iterator and raises
if no new chunk arrives within `inactivity_timeout_s` seconds.

The stream itself is consumed in a background thread; the main thread polls a
shared queue.  If the queue stays empty for `inactivity_timeout_s` seconds the
stream is considered hung and we raise StreamInactivityTimeout.

Usage (inside _stream_response)
--------------------------------
Replace:
    for chunk in response:
        ...

With:
    from stream_hang_guard import guarded_stream, StreamInactivityTimeout
    try:
        for chunk in guarded_stream(response, inactivity_timeout_s=30):
            ...
    except StreamInactivityTimeout as exc:
        raise GoogleAPIError(Exception("hang"), str(exc)) from exc

Tuning
------
GOOGLE_AI_STUDIO_STREAM_INACTIVITY_TIMEOUT_S  (default 30)
    Seconds of silence before declaring a hang.  30s is conservative —
    Gemma 4 almost always emits a chunk within 10s when healthy.

GOOGLE_AI_STUDIO_STREAM_CHUNK_WARN_THRESHOLD_S  (default 10)
    Log a WARNING when a single inter-chunk gap exceeds this value so you
    can tune the inactivity threshold from real logs.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Generator, Iterable, Iterator, TypeVar

_SENTINEL = object()   # marks end-of-stream
_ERROR    = object()   # marks an exception from the producer thread

T = TypeVar("T")

_DEFAULT_INACTIVITY_S    = int(os.environ.get("GOOGLE_AI_STUDIO_STREAM_INACTIVITY_TIMEOUT_S",  "30"))
_DEFAULT_WARN_THRESHOLD_S = int(os.environ.get("GOOGLE_AI_STUDIO_STREAM_CHUNK_WARN_THRESHOLD_S", "10"))


class StreamInactivityTimeout(Exception):
    """Raised when no chunk arrives within the inactivity window."""
    def __init__(self, inactivity_s: float, total_elapsed_s: float, chunks_received: int) -> None:
        self.inactivity_s    = inactivity_s
        self.total_elapsed_s = total_elapsed_s
        self.chunks_received = chunks_received
        super().__init__(
            f"Stream hang detected: no chunk for {inactivity_s:.0f}s "
            f"(total elapsed {total_elapsed_s:.0f}s, {chunks_received} chunks received). "
            f"This is a Gemma 4 inactivity hang, not a slow-but-healthy long stream."
        )


def guarded_stream(
    iterable:            Iterable[T],
    inactivity_timeout_s: int = _DEFAULT_INACTIVITY_S,
    warn_threshold_s:    int  = _DEFAULT_WARN_THRESHOLD_S,
) -> Generator[T, None, None]:
    """
    Wrap any iterable (e.g. a Google streaming response) and yield its items
    while enforcing an *inactivity* timeout between consecutive items.

    Total stream duration is unbounded — a 10-minute healthy stream is fine.
    Only silence (no new items) for `inactivity_timeout_s` seconds triggers.

    Parameters
    ----------
    iterable             : the upstream chunk iterator
    inactivity_timeout_s : raise StreamInactivityTimeout if silent for this long
    warn_threshold_s     : emit a warning log for gaps longer than this
    """
    import logging
    log = logging.getLogger(__name__)

    buf: queue.Queue = queue.Queue(maxsize=64)
    start_time = time.time()

    def _producer() -> None:
        try:
            for item in iterable:
                buf.put(("item", item))
            buf.put(("end", None))
        except Exception as exc:
            buf.put(("error", exc))

    thread = threading.Thread(target=_producer, daemon=True)
    thread.start()

    chunks_received = 0
    last_chunk_time = time.time()

    while True:
        now = time.time()
        silence_s = now - last_chunk_time

        # ── warn on slow-but-not-hung gaps ─────────────────────────────────
        if silence_s > warn_threshold_s and chunks_received > 0:
            log.warning(
                "[StreamGuard] No chunk for %.1fs (total %.1fs, %d chunks so far). "
                "Still within inactivity limit of %ds.",
                silence_s,
                now - start_time,
                chunks_received,
                inactivity_timeout_s,
            )

        # ── check inactivity timeout ────────────────────────────────────────
        if silence_s >= inactivity_timeout_s:
            thread.join(timeout=0.1)   # don't block; thread may be stuck in C
            raise StreamInactivityTimeout(
                inactivity_s    = silence_s,
                total_elapsed_s = now - start_time,
                chunks_received = chunks_received,
            )

        # ── poll queue with a short deadline so we can re-check silence ────
        time_until_timeout = max(0.1, inactivity_timeout_s - silence_s)
        poll_interval      = min(1.0, time_until_timeout)

        try:
            kind, payload = buf.get(timeout=poll_interval)
        except queue.Empty:
            continue   # loop back; silence check at top will catch a real hang

        if kind == "end":
            break
        if kind == "error":
            raise payload   # re-raise original exception from producer thread
        if kind == "item":
            chunks_received += 1
            elapsed = time.time() - start_time
            gap     = time.time() - last_chunk_time

            if gap > warn_threshold_s:
                log.warning(
                    "[StreamGuard] Chunk #%d arrived after %.1fs gap (total %.1fs).",
                    chunks_received, gap, elapsed,
                )

            last_chunk_time = time.time()
            yield payload   # type: ignore[misc]

    log.info(
        "[StreamGuard] Stream completed: %d chunks, %.1fs total.",
        chunks_received,
        time.time() - start_time,
    )
    thread.join(timeout=5.0)