"""Quality Gate for CBR case retention.

5-gate pipeline deciding whether to retain an experiment as a Case:
  Gate 1: Execution Success (no runtime errors)
  Gate 2: Metric Extraction (numeric result available)
  Gate 3: Improvement Check (beats baseline or first result)
  Gate 4: Novelty Check (not a near-duplicate of existing case)
  Gate 5: Generalization (LLM extracts transferable knowledge)

Embedding call reduction:
  Gate 4 (_gate_novelty) accepts an optional `cached_query` parameter.
  When supplied (by CBRDataScienceRDLoop.feedback), it uses that query
  for the cbr_kb.retrieve() call instead of re-embedding the case text,
  so if direct_exp_gen already warmed the cache with the same query the
  novelty check costs zero embedding calls.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from .case_schema import Case, CaseOutcome, CaseMetrics


class GateResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # gate not applicable


@dataclass
class RetentionDecision:
    retained: bool
    case_id: Optional[str] = None
    gate_results: dict = None  # {gate_name: (GateResult, reason)}
    summary: str = ""
    generalized_techniques: List[str] = None
    generalized_model_type: str = "unknown"

    def __post_init__(self):
        if self.gate_results is None:
            self.gate_results = {}
        if self.generalized_techniques is None:
            self.generalized_techniques = []


class QualityGate:
    """5-gate retention pipeline for CBR cases."""

    METRIC_ALIASES = {
        "score": {"score", "metric", "result"},
        "accuracy": {"accuracy", "acc"},
        "auc": {"auc", "auroc", "rocauc", "roc_auc"},
        "rmse": {"rmse", "rootmeansquarederror"},
        "mae": {"mae", "meanabsoluteerror"},
        "mse": {"mse", "meansquarederror"},
        "f1": {"f1", "f1score"},
        "precision": {"precision"},
        "recall": {"recall"},
        "r2": {"r2", "r2score"},
        "logloss": {"logloss", "log_loss", "crossentropy", "cross_entropy", "nll", "loss"},
    }

    KNOWN_METRIC_TOKENS = {
        alias
        for aliases in METRIC_ALIASES.values()
        for alias in aliases
    } | {"score", "metric", "result"}

    # Error patterns in execution logs
    ERROR_PATTERNS = [
        "Traceback (most recent call last)",
        "RuntimeError:",
        "FAILED",
        "MemoryError",
        "CUDA out of memory",
        "ModuleNotFoundError",
        "FileNotFoundError",
    ]

    def __init__(
        self,
        min_improvement_pct: float = 0.0,
        dedup_threshold: float = 0.92,
        require_improvement: bool = False,
    ):
        self.min_improvement_pct = min_improvement_pct
        self.dedup_threshold = dedup_threshold
        self.require_improvement = require_improvement

    def evaluate(
        self,
        case: Case,
        execution_log: str,
        cbr_kb,  # CBRKnowledgeBase
        cached_query: Optional[str] = None,
    ) -> RetentionDecision:
        """Run all 5 gates. Returns RetentionDecision.

        Args:
            case: The Case being evaluated.
            execution_log: Raw stdout/stderr from experiment execution.
            cbr_kb: The CBRKnowledgeBase instance.
            cached_query: Optional pre-built query string from the propose
                phase.  When provided, Gate 4 uses it for the novelty
                retrieve() call so the result can be served from cache
                (zero extra embedding calls if the cache is warm).
        """

        gates = {}

        # ── Gate 1: Execution Success ──────────────────
        g1_result, g1_reason = self._gate_execution(execution_log)
        gates["1_execution"] = (g1_result, g1_reason)
        if g1_result == GateResult.FAIL:
            case.outcome = CaseOutcome.FAILURE
            case.metrics.error_message = g1_reason
            return RetentionDecision(
                retained=False, gate_results=gates,
                summary=f"REJECTED at Gate 1: {g1_reason}",
            )

        # ── Gate 2: Metric Extraction ─────────────────
        g2_result, g2_reason = self._gate_metric(case, execution_log)
        gates["2_metric"] = (g2_result, g2_reason)
        if g2_result == GateResult.FAIL:
            return RetentionDecision(
                retained=False, gate_results=gates,
                summary=f"REJECTED at Gate 2: {g2_reason}",
            )

        # ── Gate 3: Improvement Check ────────────────
        g3_result, g3_reason = self._gate_improvement(case)
        gates["3_improvement"] = (g3_result, g3_reason)
        if g3_result == GateResult.FAIL and self.require_improvement:
            case.outcome = CaseOutcome.PARTIAL
            return RetentionDecision(
                retained=False, gate_results=gates,
                summary=f"REJECTED at Gate 3: {g3_reason}",
            )

        # ── Gate 4: Novelty / Dedup ──────────────────
        g4_result, g4_reason = self._gate_novelty(case, cbr_kb, cached_query=cached_query)
        gates["4_novelty"] = (g4_result, g4_reason)
        if g4_result == GateResult.FAIL:
            return RetentionDecision(
                retained=False, gate_results=gates,
                summary=f"REJECTED at Gate 4: {g4_reason}",
            )

        # ── Gate 5: Generalization (LLM) ─────────────
        g5_result, g5_reason, techniques, model_type = self._gate_generalize(case, cbr_kb=cbr_kb)
        gates["5_generalize"] = (g5_result, g5_reason)

        # All gates passed → retain
        case.outcome = CaseOutcome.SUCCESS
        return RetentionDecision(
            retained=True,
            case_id=case.case_id,
            gate_results=gates,
            summary=f"RETAINED: {case.case_id} | all 5 gates passed",
            generalized_techniques=techniques,
            generalized_model_type=model_type,
        )

    # ─── Individual Gates ─────────────────────────────

    def _gate_execution(self, log: str) -> Tuple[GateResult, str]:
        """Gate 1: Check for runtime errors in execution log."""
        for pattern in self.ERROR_PATTERNS:
            if pattern in log:
                # Extract the specific error line
                for line in reversed(log.split("\n")):
                    if "Error" in line or "Exception" in line:
                        return GateResult.FAIL, line.strip()
                return GateResult.FAIL, f"Error pattern found: {pattern}"
        return GateResult.PASS, "No errors detected"

    def _gate_metric(self, case: Case, log: str) -> Tuple[GateResult, str]:
        """Gate 2: Extract and validate metric from execution log.

        Tries task-aware extraction first (metric aliases + ambiguity checks),
        then falls back to generic metric parsing.
        """
        if case.metrics.primary_metric_value is not None:
            return GateResult.PASS, f"Metric pre-set: {case.metrics.primary_metric_value}"

        metric_name = case.metrics.primary_metric_name
        metric_aliases = self._resolve_metric_aliases(metric_name, case.problem.eval_metric)
        candidates = self._extract_metric_candidates(log)

        if not candidates:
            return GateResult.FAIL, "Could not extract numeric metric from execution log"

        target_candidates = [c for c in candidates if c[0] in metric_aliases]

        if target_candidates:
            picked_alias, val = target_candidates[-1]
            case.metrics.primary_metric_value = val
            case.metrics.compute_improvement()
            return GateResult.PASS, f"Extracted {picked_alias}: {val}"

        unique_metrics = {metric for metric, _ in candidates}
        if len(unique_metrics) > 1:
            metrics_str = ", ".join(sorted(unique_metrics))
            return GateResult.FAIL, f"Ambiguous metric in log; found multiple metrics: {metrics_str}"

        picked_alias, val = candidates[-1]
        case.metrics.primary_metric_value = val
        case.metrics.compute_improvement()
        return GateResult.PASS, f"Extracted {picked_alias}: {val}"

    @classmethod
    def _normalize_metric_token(cls, token: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", token.lower().strip())

    @classmethod
    def _resolve_metric_aliases(cls, *metric_names: Optional[str]) -> set[str]:
        aliases: set[str] = set()
        for name in metric_names:
            if not name:
                continue
            norm = cls._normalize_metric_token(name)
            if not norm or norm == "unknown":
                continue
            aliases.add(norm)
            for alias_set in cls.METRIC_ALIASES.values():
                normalized_alias_set = {cls._normalize_metric_token(a) for a in alias_set}
                if norm in normalized_alias_set:
                    aliases |= normalized_alias_set
        # Always allow generic score-like aliases as a weak fallback.
        aliases |= {"score", "metric", "result"}
        return aliases

    @classmethod
    def _extract_metric_candidates(cls, log: str) -> List[Tuple[str, float]]:
        """Extract (metric_alias, numeric_value) pairs from log text."""
        float_pat = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        metric_token_pat = r"[a-zA-Z][a-zA-Z0-9_\-]{1,30}"
        pattern = re.compile(
            rf"(?i)\b(?P<metric>{metric_token_pat})\b[^\n\r]*?"
            rf"(?:\s*(?::|=|is|was|of)\s*|\s+)"
            rf"(?P<value>{float_pat})"
        )

        candidates: List[Tuple[str, float]] = []
        for m in pattern.finditer(log):
            metric_raw = m.group("metric")
            metric_norm = cls._normalize_metric_token(metric_raw)
            if metric_norm not in cls.KNOWN_METRIC_TOKENS:
                continue
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            candidates.append((metric_norm, value))

        # Additional pass for common contextual lines like
        # "validation RMSE is 0.566899" where metric token is not at line start.
        contextual = re.compile(
            rf"(?i)\b(?:val|validation|test|final|ensemble)\b[^\n\r]*?"
            rf"\b(?P<metric>{metric_token_pat})\b\s*(?::|=|is|was|of)?\s*(?P<value>{float_pat})"
        )
        for m in contextual.finditer(log):
            metric_norm = cls._normalize_metric_token(m.group("metric"))
            if metric_norm not in cls.KNOWN_METRIC_TOKENS:
                continue
            try:
                value = float(m.group("value"))
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            candidates.append((metric_norm, value))

        return candidates

    def _gate_improvement(self, case: Case) -> Tuple[GateResult, str]:
        """Gate 3: Check if result improves over baseline."""
        m = case.metrics
        if m.baseline_value is None:
            # First result — no baseline to compare against
            return GateResult.SKIP, "No baseline available (first run)"

        m.compute_improvement()

        if m.improvement_pct is None:
            return GateResult.SKIP, "Cannot compute improvement"

        # Account for metric direction
        direction = case.problem.eval_direction
        if direction == "minimize":
            # For minimize metrics: improvement means LOWER value
            effective_improvement = -m.improvement_pct
        else:
            effective_improvement = m.improvement_pct

        if effective_improvement >= self.min_improvement_pct:
            return GateResult.PASS, f"Improvement: {effective_improvement:+.2f}%"
        else:
            return GateResult.FAIL, (f"Improvement {effective_improvement:+.2f}% "
                                     f"below threshold {self.min_improvement_pct}%")

    def _gate_novelty(
        self,
        case: Case,
        cbr_kb,
        cached_query: Optional[str] = None,
    ) -> Tuple[GateResult, str]:
        """Gate 4: Check if case is a near-duplicate of existing KB entry.

        Args:
            case: Candidate case being evaluated.
            cbr_kb: Knowledge base to search.
            cached_query: If provided, use this query string for the
                retrieve() call instead of case.to_embedding_text().
                When the cache is warm (same query used in direct_exp_gen)
                this saves one embedding API call entirely.
        """
        if not cbr_kb.cases:
            return GateResult.PASS, "KB empty — novel by definition"

        # Prefer the pre-built query from the propose phase (cache-friendly).
        # Fall back to the case's own embedding text when not available.
        query = cached_query if cached_query else case.to_embedding_text()

        results = cbr_kb.retrieve(
            query=query,
            top_k=1,
            exclude_failures=False,
            mmr_lambda=1.0,  # pure relevance for dedup check
            update_reuse_stats=False,  # always False for novelty checks
        )

        if results:
            nearest_case, sim_score = results[0]
            if sim_score >= self.dedup_threshold:
                # Is it actually better?
                direction = case.problem.eval_direction
                if (case.metrics.primary_metric_value is not None
                        and nearest_case.metrics.primary_metric_value is not None
                        and ((direction == "minimize" and case.metrics.primary_metric_value < nearest_case.metrics.primary_metric_value)
                             or (direction != "minimize" and case.metrics.primary_metric_value > nearest_case.metrics.primary_metric_value))):
                    return GateResult.PASS, (f"Near-duplicate {nearest_case.case_id} "
                                              f"(sim={sim_score:.2f}) but NEW is BETTER")
                return GateResult.FAIL, (f"Near-duplicate of {nearest_case.case_id} "
                                          f"(sim={sim_score:.2f} ≥ {self.dedup_threshold})")

        return GateResult.PASS, "No near-duplicates found"

    def _gate_generalize(self, case: Case, cbr_kb=None) -> Tuple[GateResult, str, List[str], str]:
        """Gate 5: LLM-based extraction of generalizable knowledge.

        Uses R&D-Agent's APIBackend to call the configured LLM.
        Full code is passed without truncation — at 300–500 lines per file
        the context cost is acceptable and truncation risks losing key details.
        Set CBR_GATE5_CODE_CHARS in env to a positive integer to re-enable
        truncation if your workspace files are unusually large.

        Args:
            case: The Case being generalized.
            cbr_kb: Optional CBRKnowledgeBase; when provided, loads the full
                code snapshot from the .py file instead of relying on the
                in-memory code_snippet (which may be truncated for reloaded cases).
        """
        if cbr_kb is not None:
            code_for_prompt = cbr_kb.get_full_code(case.case_id) or case.solution.code_snippet
        else:
            code_for_prompt = case.solution.code_snippet
        code_for_prompt = code_for_prompt or ""

        # Truncate only if explicitly configured via env var
        gate5_limit = int(os.environ.get("CBR_GATE5_CODE_CHARS", "0"))
        if gate5_limit > 0 and len(code_for_prompt) > gate5_limit:
            code_for_prompt = code_for_prompt[:gate5_limit]

        prompt = f"""Analyze this successful ML experiment and extract transferable knowledge.

Problem: {case.problem.task_type} / {case.problem.data_type}
Metric: {case.metrics.primary_metric_name} = {case.metrics.primary_metric_value}
Hypothesis: {case.solution.hypothesis_text}
Code excerpt:
{code_for_prompt}

Respond in EXACTLY this format (one line each):
MODEL_TYPE: 
TECHNIQUES: 
INSIGHT: 
"""

        try:
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=prompt,
                system_prompt="You are a senior ML engineer analyzing experiments.",
                json_mode=False,
            )
        except Exception as e:
            logger.warning(f"Gate 5 LLM call failed: {e}")
            return GateResult.SKIP, str(e), [], "unknown"

        # Parse response
        model_type = "unknown"
        techniques = []
        for line in response.strip().split("\n"):
            if line.upper().startswith("MODEL_TYPE:"):
                model_type = line.split(":", 1)[-1].strip().lower()
            elif line.upper().startswith("TECHNIQUES:"):
                raw = line.split(":", 1)[-1]
                techniques = [t.strip().lower() for t in raw.split(",") if t.strip()]

        # Apply to case
        case.solution.model_type = model_type
        case.solution.key_techniques = techniques

        return GateResult.PASS, f"Generalized: {model_type}, {techniques}", techniques, model_type
