"""Tracks failure patterns to prevent repeating known bad approaches.

Stores failures with embeddings so the CBR loop can query:
'What approaches have failed for problems like mine?'

Caching:
    get_relevant() results are cached by (query, top_k, min_similarity)
    hash.  The cache is fully invalidated on every log() call, since new
    failures change what's relevant.  This eliminates the duplicate
    embedding call that happens when direct_exp_gen() and quality_gate
    both query failures for the same hypothesis text.
"""
import hashlib
import json
import os
import time
from typing import List, Optional

from rdagent.log import rdagent_logger as logger
from rdagent.components.knowledge_management.vector_base import (
    PDVectorBase, KnowledgeMetaData,
)


class FailureTracker:
    """Stores and retrieves failure patterns using embedding-based search.

    Unlike simple keyword matching, this uses PDVectorBase so the agent
    can find semantically similar failures even with different wording.
    """

    MAX_FAILURES = 200  # Rolling window

    def __init__(self, kb_dir: str = "./cbr_knowledge_base"):
        self.failures_file = os.path.join(kb_dir, "failure_patterns.json")
        self.vb_path = os.path.join(kb_dir, "failure_vectors.pkl")

        os.makedirs(kb_dir, exist_ok=True)

        # Load
        self.failures: List[dict] = []
        if os.path.exists(self.failures_file):
            with open(self.failures_file) as f:
                self.failures = json.load(f)

        # Normalize IDs for backward compatibility and keep a monotonic counter.
        max_id = 0
        for i, failure in enumerate(self.failures, 1):
            if "id" not in failure:
                failure["id"] = i
            max_id = max(max_id, int(failure["id"]))
        self.next_failure_id = max_id + 1

        if os.path.exists(self.vb_path):
            import pickle
            with open(self.vb_path, "rb") as f:
                self.vb: PDVectorBase = pickle.load(f)
        else:
            self.vb = PDVectorBase()

        # ── get_relevant cache ─────────────────────────
        # Maps cache_key (str) -> List[dict]
        # Fully invalidated whenever a new failure is logged.
        self._relevant_cache: dict[str, List[dict]] = {}

        logger.info(f"FailureTracker loaded: {len(self.failures)} patterns")

    # ─── Cache helpers ────────────────────────────────

    @staticmethod
    def _cache_key(query: str, top_k: int, min_similarity: float) -> str:
        raw = f"{query}|{top_k}|{min_similarity:.4f}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _invalidate_cache(self) -> None:
        """Clear get_relevant cache. Called whenever new failures are logged."""
        self._relevant_cache.clear()

    # ─── Public API ───────────────────────────────────

    def log(
        self,
        problem_description: str,
        error_message: str,
        approach_description: str = "",
        component: str = "",
        competition: str = "",
    ):
        """Record a new failure pattern."""
        entry = {
            "id": self.next_failure_id,
            "problem": problem_description,
            "error": error_message,
            "approach": approach_description,
            "component": component,
            "competition": competition,
            "timestamp": time.time(),
        }
        self.next_failure_id += 1

        self.failures.append(entry)

        # Rolling window
        if len(self.failures) > self.MAX_FAILURES:
            self.failures = self.failures[-self.MAX_FAILURES:]

        # Embed for semantic retrieval
        embed_text = f"{entry['problem']} | {entry['approach']} | {entry['error']}"
        self.vb.add(KnowledgeMetaData(
            content=embed_text,
            label=f"failure_{entry['id']}",
        ))

        # Invalidate cache — new failure changes relevance rankings.
        self._invalidate_cache()

        self._save()
        logger.info(f"Failure logged: {error_message}...")

    def get_relevant(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.3,
    ) -> List[dict]:
        """Find failure patterns relevant to current problem via embedding search.

        Results are cached so repeated calls with the same query (e.g. from
        direct_exp_gen and quality_gate in the same loop iteration) only
        incur one embedding call.
        """
        if not self.failures:
            return []

        key = self._cache_key(query, top_k, min_similarity)
        if key in self._relevant_cache:
            logger.debug("FailureTracker: cache hit for get_relevant")
            return self._relevant_cache[key]

        docs, scores = self.vb.search(content=query, topk_k=top_k)
        results = list(zip(docs, scores))
        id_to_failure = {int(f["id"]): f for f in self.failures if "id" in f}
        relevant = []
        for metadata, score in results:
            if score >= min_similarity:
                idx_str = metadata.label.replace("failure_", "")
                try:
                    failure_id = int(idx_str)
                    entry = id_to_failure.get(failure_id)
                    if entry is None:
                        # Backward compatibility for old vector labels that used position.
                        idx = failure_id - 1
                        if 0 <= idx < len(self.failures):
                            entry = self.failures[idx]
                    if entry is not None:
                        out = entry.copy()
                        out["similarity"] = round(score, 3)
                        relevant.append(out)
                except (ValueError, IndexError):
                    continue

        self._relevant_cache[key] = relevant
        return relevant

    def format_for_prompt(self, failures: List[dict]) -> str:
        """Format failure patterns as context for LLM prompts."""
        if not failures:
            return "No known failure patterns for this problem type."
        lines = ["## Known Failure Patterns (AVOID these approaches):"]
        for i, f in enumerate(failures, 1):
            lines.append(
                f"{i}. Approach: {f['approach']}\n"
                f"   Error: {f['error']}\n"
                f"   Similarity: {f.get('similarity', 'N/A')}"
            )
        return "\n".join(lines)

    def _save(self):
        with open(self.failures_file, "w") as f:
            json.dump(self.failures, f, indent=2)
        import pickle
        with open(self.vb_path, "wb") as f:
            pickle.dump(self.vb, f)
