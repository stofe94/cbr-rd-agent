"""CBR-enhanced DataScience R&D Loop.

Injects Case-Based Reasoning into R&D-Agent's standard loop:
  propose()  → Enriches hypothesis with retrieved cases + failure avoidance
  feedback() → Evaluates result through QualityGate and retains successful cases

Embedding call reduction:
  - direct_exp_gen() builds ONE query and passes it to both cbr_kb.retrieve()
    and failure_tracker.get_relevant() — both caches are warm for that query
    for the rest of the loop iteration.
  - quality_gate._gate_novelty() calls cbr_kb.retrieve(update_reuse_stats=False)
    which hits the cache populated during direct_exp_gen because retrieve() now
    uses _save_no_invalidate() for reuse-stat persistence — cache stays warm
    until the next iteration's _flush_loop_caches() call.
  - At the start of each new loop iteration the caches are explicitly flushed
    so stale entries don't accumulate across iterations.
  - PDVectorBase.search() is called with constraint_labels=list(cases.keys())
    to prevent KaggleExperienceBase trunk rows from polluting CBR retrieval.

CBR context injection:
  - CBR context is written to hypothesis.appendix (not hypothesis.reason) so
    case_schema._build_plan_summary() reads clean concise_* fields and
    plan_summary stored in the KB contains no injected code or context.

Embedding task_type fix:
  - PDVectorBase.search() embeds the query with task_type=RETRIEVAL_QUERY
    (patched via _begin_token_usage_run) for correct asymmetric similarity.
  - PDVectorBase.add() keeps task_type=RETRIEVAL_DOCUMENT (default).

Token minimization:
  - _format_cbr_context(): truncates plan_summary to PLAN_SUMMARY_CHARS;
    to PLAN_SUMMARY_CHARS; uses compact single-line case headers.
  - _build_cbr_query(): description truncated to QUERY_DESC_CHARS.
  - No truncation at store time — all KB fields persisted at full length.
  - Gate-5 prompt: code capped at GATE5_CODE_CHARS.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.loop import DataScienceRDLoop
from rdagent.scenarios.data_science.proposal.exp_gen.base import DSTrace
from rdagent.oai.llm_utils import APIBackend

from .case_schema import Case
from .case_knowledge import CBRKnowledgeBase
from .quality_gate import QualityGate
from .failure_tracker import FailureTracker

# ─── Configuration (from env vars or defaults) ──────────
CBR_KB_DIR = os.environ.get("CBR_KB_DIR", "./knowledge_base")
CBR_TOP_K = int(os.environ.get("CBR_TOP_K", "3"))
CBR_CASE_SIMILARITY_THRESHOLD = float(os.environ.get("CBR_CASE_SIMILARITY_THRESHOLD", "0.5"))
CBR_MIN_IMPROVEMENT = float(os.environ.get("CBR_MIN_IMPROVEMENT", "0.0"))
CBR_DEDUP_THRESHOLD = float(os.environ.get("CBR_DEDUP_THRESHOLD", "0.92"))
CBR_REQUIRE_IMPROVEMENT = os.environ.get("CBR_REQUIRE_IMPROVEMENT", "false").lower() == "true"
CBR_EXCLUDE_COMPETITION = os.environ.get("CBR_EXCLUDE_COMPETITION", "").strip() or None

# ─── A/B testing switch ──────────────────────────────────
# When CBR_DISABLE_RETRIEVAL=1 the loop runs without any CBR influence on
# the LLM: no case retrieval, no failure injection, no CBR context in the
# hypothesis or coding prompts. Token tracking, quality gates and case
# retention all remain active so the run is otherwise identical and can
# be compared 1:1 against a CBR-enabled run with the same seed/budget.
#
# Typical usage:
#   CBR_DISABLE_RETRIEVAL=1 rdagent ...   # baseline run, no CBR help
#   CBR_DISABLE_RETRIEVAL=0 rdagent ...   # CBR-enabled run
# Then compare best metric, iterations-to-best, token totals, and failure
# rate across multiple repetitions on the same competition.
CBR_DISABLE_RETRIEVAL = os.environ.get("CBR_DISABLE_RETRIEVAL", "0").lower() in ("1", "true", "yes")

# ─── Token budget constants ──────────────────────────────
# plan_summary shown in the hypothesis-enrichment prompt (per case).
# Only display-time truncation — nothing is truncated at store time.
PLAN_SUMMARY_CHARS = int(os.environ.get("CBR_PLAN_SUMMARY_CHARS", "300"))
# scenario.description included in the embedding query
QUERY_DESC_CHARS = int(os.environ.get("CBR_QUERY_DESC_CHARS", "400"))
# Code in hypothesis appendix (planning phase): 0 = no code shown,
# keeping the planning prompt lean. Code is fully available in coding phase.
HYPOTHESIS_CODE_CHARS = int(os.environ.get("CBR_HYPOTHESIS_CODE_CHARS", "0"))
# Code in coding phase prompt: 0 = no truncation (full code always shown).
# At 300–500 lines per file (~4000–6000 chars) and TOP_K=3, total is well
# within modern LLM context windows.
CODING_CODE_CHARS = int(os.environ.get("CBR_CODING_CODE_CHARS", "0"))
# Code sent to Gate-5 LLM: 0 = no truncation.
GATE5_CODE_CHARS = int(os.environ.get("CBR_GATE5_CODE_CHARS", "0"))


def _trunc(text: str, max_chars: int, suffix: str = "…") -> str:
    """Truncate text to max_chars, appending suffix if cut."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


class CBRDataScienceRDLoop(DataScienceRDLoop):
    """R&D Loop + CBR: learns from past experiments across sessions.

    Inherits the full DataScienceRDLoop and injects CBR at two points:
    1. propose() — Retrieves similar cases from KB to enrich hypothesis context
    2. feedback() — Evaluates and retains successful experiments as new cases
    """

    def __init__(self, PROP_SETTING):
        super().__init__(PROP_SETTING)

        # ─── CBR Components ──────────────────────────
        self.cbr_kb = CBRKnowledgeBase(kb_dir=CBR_KB_DIR)
        self.quality_gate = QualityGate(
            min_improvement_pct=CBR_MIN_IMPROVEMENT,
            dedup_threshold=CBR_DEDUP_THRESHOLD,
            require_improvement=CBR_REQUIRE_IMPROVEMENT,
        )
        self.failure_tracker = FailureTracker(kb_dir=CBR_KB_DIR)

        # ─── State ───────────────────────────────────
        self._current_retrieved_cases = []
        self._current_relevant_cases = []
        self._current_source_case_id = None
        self._baseline_metric = None
        self._current_cbr_query: str = ""

        # ─── Token Usage Tracking ────────────────────
        self._token_usage_phase = "idle"
        self._token_usage_active = False
        self._token_usage_state = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "embedding_input_tokens": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "models": {},
            "embedding_models": {},
            "phase_breakdown": {
                "propose": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "embedding_input_tokens": 0,
                    "llm_calls": 0,
                    "embedding_calls": 0,
                },
                "feedback": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "embedding_input_tokens": 0,
                    "llm_calls": 0,
                    "embedding_calls": 0,
                },
            },
            "started_at": None,
        }
        self._orig_chat_completion_inner = None
        self._orig_embedding_inner = None

        logger.info(
            f"CBR Loop initialized | mode: {'DISABLED (A/B baseline)' if CBR_DISABLE_RETRIEVAL else 'ENABLED'} | "
            f"KB: {self.cbr_kb.stats()} | Failures: {len(self.failure_tracker.failures)}"
        )

    # ═══════════════════════════════════════════════
    # Phase 1 Override: DIRECT_EXP_GEN → CBR RETRIEVE + ENRICH
    # ═══════════════════════════════════════════════

    async def direct_exp_gen(self, prev_out: Dict[str, Any]):
        """Wrap standard direct_exp_gen() with CBR retrieval and enrichment.

        Embedding-call budget for this method: 2 (one for cbr_kb.retrieve,
        one for failure_tracker.get_relevant).  Both results are cached so
        any downstream code that calls them with the same query — including
        quality_gate._gate_novelty() — pays zero extra embedding calls.
        """
        self._flush_loop_caches()
        self._begin_token_usage_run()
        self._token_usage_phase = "propose"

        try:
            scenario = self.trace.scen if hasattr(self.trace, "scen") else None
            exclude_competition = self._resolve_exclude_competition(scenario)
            pre_query = self._build_cbr_query(None, scenario)

            # A/B testing: skip all retrieval when disabled to get a clean
            # baseline run. Token tracking, gates, and add_case still run
            # so the runs are otherwise identical and comparable.
            if CBR_DISABLE_RETRIEVAL:
                pre_results = []
                self._current_cbr_query = pre_query
                self._current_retrieved_cases = []
                exp = await super().direct_exp_gen(prev_out)
                logger.info("CBR DISABLED: skipping retrieval and enrichment")
                return exp

            try:
                pre_results = self.cbr_kb.retrieve(
                    query=pre_query,
                    top_k=CBR_TOP_K,
                    exclude_competition=exclude_competition,
                    min_similarity=CBR_CASE_SIMILARITY_THRESHOLD,
                    update_reuse_stats=False,
                )
            except Exception:
                pre_results = []

            self._current_cbr_query = pre_query
            self._current_retrieved_cases = pre_results

            try:
                pre_cbr_context = (
                    self._format_cbr_context(pre_results, [], include_code=False)
                    if pre_results else ""
                )
                if hasattr(self, "trace") and self.trace is not None:
                    setattr(self.trace, "cbr_retrieved_for_planning", pre_cbr_context)
            except Exception:
                pass

            if pre_results:
                logger.info(f"CBR PRE-RETRIEVE: warmed cache with {len(pre_results)} cases for planning")

            exp = await super().direct_exp_gen(prev_out)
            hypothesis = getattr(exp, "hypothesis", None)

            if hypothesis is None:
                logger.warning("CBR RETRIEVE: No hypothesis found in direct_exp_gen output")
                return exp

            query = self._build_cbr_query(hypothesis, scenario)
            self._current_cbr_query = query

            self._current_retrieved_cases = self.cbr_kb.retrieve(
                query=query,
                top_k=CBR_TOP_K,
                exclude_competition=exclude_competition,
                min_similarity=CBR_CASE_SIMILARITY_THRESHOLD,
            )

            # Cache is now warm — zero extra embeddings for same query.
            failures = self.failure_tracker.get_relevant(query, top_k=3)

            if self._current_retrieved_cases or failures:
                cbr_context = self._format_cbr_context(self._current_retrieved_cases, failures)
                # Inject CBR context into appendix — not reason — so that
                # case_schema._build_plan_summary() can read a clean reason/
                # concise_* without CBR noise polluting plan_summary.
                if hasattr(hypothesis, "appendix"):
                    hypothesis.appendix = cbr_context
                elif hasattr(hypothesis, "reason"):
                    hypothesis.reason = cbr_context + "\n\n" + (hypothesis.reason or "")
                elif hasattr(hypothesis, "context"):
                    hypothesis.context = cbr_context

                try:
                    if hasattr(self, "trace") and self.trace is not None:
                        setattr(self.trace, "cbr_retrieved_for_planning", cbr_context)
                except Exception:
                    pass

                scenario_task_type = getattr(scenario, "task_type", None) if scenario else None
                scenario_data_type = getattr(scenario, "data_type", None) if scenario else None
                self._current_relevant_cases = [
                    (case, score)
                    for case, score in self._current_retrieved_cases
                    if self._is_case_relevant(case, score, scenario_task_type, scenario_data_type)
                ]
                if self._current_relevant_cases:
                    logger.info(
                        f"CBR RELEVANCE: {len(self._current_relevant_cases)}/"
                        f"{len(self._current_retrieved_cases)} cases relevant for coding"
                    )
                else:
                    logger.info("CBR RELEVANCE: no cases passed relevance filter")

                if self._current_relevant_cases:
                    self._current_source_case_id = self._current_relevant_cases[0][0].case_id
                else:
                    self._current_source_case_id = None

                logger.info(
                    f"CBR RETRIEVE: {len(self._current_retrieved_cases)} cases, "
                    f"{len(failures)} failure patterns injected"
                )
            else:
                self._current_source_case_id = None
                logger.info("CBR RETRIEVE: Cold start — no relevant cases or failures")

            return exp
        finally:
            self._finalize_token_usage_run()

    # ═══════════════════════════════════════════════
    # Phase 2 Override: CODING → inject CBR code into task descriptions
    # ═══════════════════════════════════════════════

    # ── Type-normalization tables ─────────────────────────────────────────
    # Maps any variant spelling → canonical group label.
    # Matching is group-level: two types match iff they share a canonical group.
    _TASK_TYPE_GROUPS: dict[str, str] = {
        # classification family
        "classification":              "classification",
        "binary_classification":       "classification",
        "binary":                      "classification",
        "multiclass_classification":   "classification",
        "multiclass":                  "classification",
        "tabular_classification":      "classification",
        "multi_class":                 "classification",
        "multi-class":                 "classification",
        # regression family
        "regression":                  "regression",
        "tabular_regression":          "regression",
        "numeric_prediction":          "regression",
        # time series
        "time_series":                 "time_series",
        "time_series_forecasting":     "time_series",
        "forecasting":                 "time_series",
        "timeseries":                  "time_series",
        # NLP
        "nlp":                         "nlp",
        "text_classification":         "nlp",
        "natural_language_processing": "nlp",
        "text":                        "nlp",
        "text_regression":             "nlp",
        # CV
        "cv":                          "cv",
        "computer_vision":             "cv",
        "image_classification":        "cv",
        "object_detection":            "cv",
        "image":                       "cv",
        "image_regression":            "cv",
        # recommendation
        "recommendation":              "recommendation",
        "collaborative_filtering":     "recommendation",
        "ranking":                     "recommendation",
    }

    _DATA_TYPE_GROUPS: dict[str, str] = {
        "tabular":          "tabular",
        "tabular_data":     "tabular",
        "structured":       "tabular",
        "csv":              "tabular",
        "text":             "text",
        "nlp":              "text",
        "natural_language": "text",
        "image":            "image",
        "images":           "image",
        "vision":           "image",
        "time_series":      "time_series",
        "timeseries":       "time_series",
        "sequential":       "time_series",
        "graph":            "graph",
        "network":          "graph",
        "multimodal":       "multimodal",
        "mixed":            "multimodal",
    }

    @classmethod
    def _normalize_task_type(cls, raw: str) -> str:
        """Return the canonical group for a task_type string, or the lowercased raw value."""
        return cls._TASK_TYPE_GROUPS.get(raw.lower().strip(), raw.lower().strip())

    @classmethod
    def _normalize_data_type(cls, raw: str) -> str:
        """Return the canonical group for a data_type string, or the lowercased raw value."""
        return cls._DATA_TYPE_GROUPS.get(raw.lower().strip(), raw.lower().strip())

    # ═══════════════════════════════════════════════

    def _is_case_relevant(
        self,
        case,
        score: float,
        scenario_task_type: Optional[str],
        scenario_data_type: Optional[str],
        min_similarity: float = CBR_CASE_SIMILARITY_THRESHOLD,
    ) -> bool:
        """Determine if a retrieved case is relevant enough for code injection.

        Uses normalised type-group matching instead of exact string equality so
        that e.g. "binary_classification" (scenario) matches "classification"
        (stored case) and "tabular_data" matches "tabular".

        Rules:
        - Score must meet min_similarity threshold.
        - If both scenario types are absent/None: wildcard → always relevant.
        - task_type and data_type are each normalised to a canonical group label.
          "unknown" on either side is treated as a wildcard.
        - Both dimensions must match; if one dimension is absent it is skipped.
        """
        if score < min_similarity:
            return False
        if not scenario_task_type and not scenario_data_type:
            return True

        def _type_match(scenario_raw: Optional[str], case_raw: str, norm_fn) -> bool:
            if not scenario_raw:
                return True  # dimension absent → skip
            s = norm_fn(scenario_raw)
            c = norm_fn(case_raw)
            return s in ("unknown", "") or c in ("unknown", "") or s == c

        task_match = _type_match(
            scenario_task_type,
            getattr(case.problem, "task_type", "unknown") or "unknown",
            self._normalize_task_type,
        )
        data_match = _type_match(
            scenario_data_type,
            getattr(case.problem, "data_type", "unknown") or "unknown",
            self._normalize_data_type,
        )
        return task_match and data_match

    def _format_cbr_code_section(self, cases) -> str:
        """Format retrieved cases as code reference for the coding prompt.

        Full code is shown without truncation (CODING_CODE_CHARS=0 default).
        At 300–500 lines per main.py and TOP_K=3, total injected code stays
        well within modern LLM context windows. Set CBR_CODING_CODE_CHARS
        in env to a positive integer to re-enable truncation if needed.
        """
        if not cases:
            return ""

        lines = [
            "## CBR Reference Code (Similar Past Cases)",
            "> Adapt what is useful, ignore what does not fit. Do NOT copy blindly.",
        ]

        any_code_added = False
        for i, (case, score) in enumerate(cases, 1):
            full_code = self.cbr_kb.get_full_code(case.case_id) or case.solution.code_snippet or ""
            if not full_code:
                continue
            code = self._extract_main_py(full_code)
            if not code.strip():
                continue

            # Truncate only if explicitly configured — default is no truncation
            if CODING_CODE_CHARS > 0:
                code = _trunc(code, CODING_CODE_CHARS)

            metric_str = (
                f"{case.metrics.primary_metric_name}={case.metrics.primary_metric_value:.4f}"
                if case.metrics.primary_metric_value is not None
                else f"{case.metrics.primary_metric_name}=N/A"
            )
            techniques_str = (
                f" techs=[{', '.join(case.solution.key_techniques[:5])}]"
                if case.solution.key_techniques else ""
            )

            lines.append(
                f"\n### Ref {i} (sim={score:.2f} {case.problem.task_type}/{case.problem.data_type}"
                f" model={case.solution.model_type} {metric_str}{techniques_str})"
            )
            lines.append("```python")
            lines.append(code)
            lines.append("```")
            any_code_added = True

        if not any_code_added:
            return ""
        return "\n".join(lines)

    def _extract_main_py(self, full_code: str) -> str:
        """Extract main.py content from the multi-file snapshot format."""
        main_marker = "# ===== FILE: main.py ====="
        if main_marker not in full_code:
            return full_code.strip()
        start = full_code.index(main_marker)
        next_marker = full_code.find("# ===== FILE:", start + len(main_marker))
        code = full_code[start:next_marker].strip() if next_marker != -1 else full_code[start:].strip()
        lines = code.split("\n", 1)
        return lines[1].strip() if len(lines) > 1 else ""

    def coding(self, prev_out):
        # A/B testing: bypass all CBR code injection when disabled.
        # Standard rdagent coding path runs untouched.
        if CBR_DISABLE_RETRIEVAL:
            return super().coding(prev_out)

        logger.info(f"CBR CODING ENTER: relevant={len(self._current_relevant_cases)}")
        experiment = prev_out.get("direct_exp_gen")
        if experiment is None:
            return super().coding(prev_out)

        attrs = [a for a in dir(experiment) if not a.startswith('_')]
        logger.info(f"CBR DEBUG: experiment attrs = {attrs}")
        for attr_name in ["task", "sub_tasks", "pending_tasks_list", "tasks"]:
            val = getattr(experiment, attr_name, "NOT_PRESENT")
            logger.info(f"CBR DEBUG: experiment.{attr_name} = {type(val).__name__}: {val if val == 'NOT_PRESENT' else str(val)[:200]}")

        ws = getattr(experiment, "experiment_workspace", None)
        if ws and hasattr(ws, "file_dict"):
            logger.info(f"CBR DEBUG: workspace files = {list(ws.file_dict.keys())}")

        if self._current_relevant_cases:
            cbr_code_section = self._format_cbr_code_section(self._current_relevant_cases)
            if cbr_code_section:
                injected_anywhere = False

                sub_tasks = getattr(experiment, "sub_tasks", None) or []
                for task in sub_tasks:
                    if hasattr(task, "description") and task.description:
                        task.description += "\n\n" + cbr_code_section
                        logger.info(f"CBR CODING: ✓ injected into sub_task '{getattr(task, 'name', '?')}'")
                        injected_anywhere = True

                pending = getattr(experiment, "pending_tasks_list", None)
                if pending:
                    for task_group in pending:
                        if isinstance(task_group, list):
                            for task in task_group:
                                if hasattr(task, "description") and task.description:
                                    task.description += "\n\n" + cbr_code_section
                                    logger.info(f"CBR CODING: ✓ injected into pending task '{getattr(task, 'name', '?')}'")
                                    injected_anywhere = True
                        elif hasattr(task_group, "description") and task_group.description:
                            task_group.description += "\n\n" + cbr_code_section
                            injected_anywhere = True

                pipeline_task = getattr(experiment, "task", None)
                if pipeline_task and hasattr(pipeline_task, "description") and pipeline_task.description:
                    pipeline_task.description += "\n\n" + cbr_code_section
                    logger.info("CBR CODING: ✓ injected into pipeline task")
                    injected_anywhere = True

                if ws and hasattr(ws, "file_dict"):
                    ws.file_dict["cbr_reference.md"] = cbr_code_section
                    logger.info("CBR CODING: ✓ wrote cbr_reference.md to workspace")
                    injected_anywhere = True

                if not injected_anywhere:
                    logger.error("CBR CODING: ✗✗✗ NO INJECTION PATH WORKED")

        return super().coding(prev_out)

    # ═══════════════════════════════════════════════
    # Phase 5 Override: FEEDBACK → CBR RETAIN
    # ═══════════════════════════════════════════════

    def feedback(self, prev_out: Dict[str, Any]):
        self._begin_token_usage_run()
        self._token_usage_phase = "feedback"

        try:
            fb = super().feedback(prev_out)

            experiment = prev_out.get("running")
            hypothesis = getattr(experiment, "hypothesis", None) if experiment is not None else None
            if not hypothesis or experiment is None:
                logger.warning("CBR RETAIN: Could not extract current hypothesis/experiment")
                return fb

            trace: DSTrace = self.trace
            scenario = trace.scen if hasattr(trace, "scen") else None
            execution_log = str(fb) if fb else ""
            metric_value = self._extract_metric_from_experiment(experiment)

            case = Case.from_rd_agent(
                hypothesis=hypothesis,
                experiment=experiment,
                scenario=scenario,
                metric_value=metric_value,
                baseline_value=self._baseline_metric,
                execution_feedback=execution_log,
                loop_idx=self.loop_idx if hasattr(self, "loop_idx") else None,
                node_idx=None,
                source_case_id=self._current_source_case_id,
            )

            decision = self.quality_gate.evaluate(
                case=case,
                execution_log=execution_log,
                cbr_kb=self.cbr_kb,
                cached_query=self._current_cbr_query or None,
            )

            if decision.generalized_techniques:
                case.solution.key_techniques = decision.generalized_techniques
            if decision.generalized_model_type != "unknown":
                case.solution.model_type = decision.generalized_model_type

            if decision.retained:
                self.cbr_kb.add_case(case)
                if self._baseline_metric is None and case.metrics.primary_metric_value is not None:
                    self._baseline_metric = case.metrics.primary_metric_value

                # ── Kausalitätsprüfung ────────────────────────────────────
                # Reuse wird nur gesetzt wenn der neue Code nachweislich aus
                # einem retrieved Case entwickelt wurde — gemessen über
                # Embedding-Similarity (gleicher Ansatz) UND Code-Fingerprint-
                # Similarity (gleiche Struktur). Reine LLM-Eigenentwicklungen
                # ohne Bezug zu retrieved Cases bekommen kein reuse-Signal.
                #
                # WICHTIG: Wir prüfen nur gegen _current_relevant_cases, also
                # die Cases die tatsächlich per inject in den Coding-Prompt
                # geflossen sind. Wenn _is_case_relevant() 0/N Cases durchließ,
                # hat das LLM keinen CBR-Code gesehen → Causality ist strukturell
                # unmöglich → kein reuse-Signal, egal wie ähnlich die Embeddings sind.
                if not self._current_relevant_cases:
                    logger.info(
                        "CBR REUSE: kein reuse-Signal — keine Cases wurden in Coding injiziert "
                        f"(retrieved={len(self._current_retrieved_cases)}, relevant=0)"
                    )
                    causal_cases = []
                else:
                    causal_cases = self.cbr_kb.check_reuse_causality(
                        new_case=case,
                        retrieved_cases=self._current_relevant_cases,
                    )
                if causal_cases:
                    for src_case, emb_sim, fp_sim in causal_cases:
                        source = self.cbr_kb.get_case(src_case.case_id)
                        if source:
                            source.record_reuse(was_successful=True)
                            logger.info(
                                f"CBR REUSE: ✔ {source.case_id} "
                                f"(emb={emb_sim:.3f} fp={fp_sim:.3f})"
                            )
                    self.cbr_kb.save()
                else:
                    logger.info(
                        "CBR REUSE: kein kausaler Source-Case gefunden — "
                        "LLM hat eigenständig entwickelt, kein reuse-Signal"
                    )
                logger.info(f"CBR RETAIN: ✔ {decision.summary}")
            else:
                self.failure_tracker.log(
                    problem_description=case.problem.raw_description,
                    error_message=case.metrics.error_message or decision.summary,
                    approach_description=case.solution.hypothesis_text,
                    component=case.problem.target_component,
                    competition=case.problem.competition_name,
                )
                # record_reuse(False) nur wenn:
                # 1. Kein Runtime-Error (Gate 1/2) — der war nicht am Source-Case schuld
                # 2. Ein kausaler Source-Case existiert — nur dann ist das Signal sinnvoll
                failed_gate = next(iter(decision.gate_results), "")
                if failed_gate not in ("1_execution", "2_metric") and self._current_relevant_cases:
                    causal_cases = self.cbr_kb.check_reuse_causality(
                        new_case=case,
                        retrieved_cases=self._current_relevant_cases,
                    )
                    if causal_cases:
                        for src_case, emb_sim, fp_sim in causal_cases:
                            source = self.cbr_kb.get_case(src_case.case_id)
                            if source:
                                source.record_reuse(was_successful=False)
                                logger.info(
                                    f"CBR REUSE: ✗ {source.case_id} "
                                    f"(emb={emb_sim:.3f} fp={fp_sim:.3f})"
                                )
                        self.cbr_kb.save()
                logger.info(f"CBR RETAIN: ✗ {decision.summary}")

            for gate_name, (result, reason) in decision.gate_results.items():
                logger.debug(f"  Gate {gate_name}: {result.value} — {reason}")

            return fb
        finally:
            self._finalize_token_usage_run()

    def _extract_metric_from_experiment(self, experiment) -> Optional[float]:
        result = getattr(experiment, "result", None)
        if result is None:
            return None
        try:
            if hasattr(result, "empty") and result.empty:
                return None
            if hasattr(result, "iloc"):
                for idx in range(min(len(result.index), 3)):
                    for col in result.columns:
                        val = result.iloc[idx][col]
                        if isinstance(val, (int, float)):
                            return float(val)
        except Exception:
            pass
        if isinstance(result, (int, float)):
            return float(result)
        return None

    # ─── Cache management ────────────────────────────

    def _flush_loop_caches(self) -> None:
        self.cbr_kb._invalidate_cache()
        self.failure_tracker._invalidate_cache()
        self._current_relevant_cases = []
        logger.debug("CBR caches flushed for new loop iteration")

    # ─── Token usage tracking ────────────────────────

    def _begin_token_usage_run(self) -> None:
        if self._token_usage_active:
            self._finalize_token_usage_run()

        self._token_usage_active = True
        self._token_usage_state = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "embedding_input_tokens": 0,
            "llm_calls": 0,
            "embedding_calls": 0,
            "models": {},
            "embedding_models": {},
            "phase_breakdown": {
                "propose": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "embedding_input_tokens": 0,
                    "llm_calls": 0,
                    "embedding_calls": 0,
                },
                "feedback": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "embedding_input_tokens": 0,
                    "llm_calls": 0,
                    "embedding_calls": 0,
                },
            },
            "started_at": time.time(),
        }

        backend_cls = APIBackend().__class__
        if self._orig_chat_completion_inner is not None or self._orig_embedding_inner is not None:
            return

        self._orig_chat_completion_inner = backend_cls._create_chat_completion_inner_function
        self._orig_embedding_inner = backend_cls._create_embedding_inner_function
        loop_ref = self

        def _wrapped_chat_completion_inner(backend_self, messages, response_format=None, *args, **kwargs):
            content, finish_reason = loop_ref._orig_chat_completion_inner(
                backend_self, messages, response_format=response_format, *args, **kwargs,
            )
            model_name = "unknown"
            prompt_tokens = 0
            completion_tokens = 0
            try:
                if hasattr(backend_self, "get_complete_kwargs"):
                    model_name = str(backend_self.get_complete_kwargs().get("model", "unknown"))
                from litellm import token_counter
                prompt_tokens = int(token_counter(model=model_name, messages=messages))
                completion_tokens = int(token_counter(model=model_name, text=content or ""))
            except Exception:
                try:
                    prompt_tokens = int(backend_self._calculate_token_from_messages(messages))
                except Exception:
                    prompt_tokens = 0

            if loop_ref._token_usage_active:
                phase = loop_ref._token_usage_phase if loop_ref._token_usage_phase in ("propose", "feedback") else "propose"
                loop_ref._token_usage_state["prompt_tokens"] += prompt_tokens
                loop_ref._token_usage_state["completion_tokens"] += completion_tokens
                loop_ref._token_usage_state["llm_calls"] += 1
                loop_ref._token_usage_state["models"][model_name] = (
                    loop_ref._token_usage_state["models"].get(model_name, 0) + 1
                )
                phase_stat = loop_ref._token_usage_state["phase_breakdown"][phase]
                phase_stat["prompt_tokens"] += prompt_tokens
                phase_stat["completion_tokens"] += completion_tokens
                phase_stat["llm_calls"] += 1
            return content, finish_reason

        def _wrapped_embedding_inner(backend_self, input_content_list, *args, **kwargs):
            # ── task_type fix ────────────────────────────────────────────────
            # The backend's _create_embedding_inner_function hardcodes
            # task_type="RETRIEVAL_DOCUMENT" inside its own client.models.
            # embed_content() call. Its signature accepts no kwargs, so we
            # cannot pass config through.
            #
            # Instead we patch client.models.embed_content on the global
            # genai.Client singleton (returned by google_ai_studio._get_client)
            # for the duration of one inner call. The patch wraps the original
            # method and overrides task_type to RETRIEVAL_QUERY when the input
            # looks like a search query (1 short string).
            #
            # Heuristic: 1 short string = search query; longer or list of
            # multiple strings = document indexing (add_case / trunks).
            _QUERY_CHAR_THRESHOLD = 2000
            _is_query = (
                len(input_content_list) == 1
                and len(input_content_list[0]) < _QUERY_CHAR_THRESHOLD
            )

            _patched = False
            _orig_embed_content = None
            _client = None
            if _is_query:
                try:
                    from rdagent.oai.backend import google_ai_studio as _gas
                    from google.genai import types as genai_types
                    _client = _gas._get_client()
                    _orig_embed_content = _client.models.embed_content

                    def _patched_embed_content(*pargs, **pkwargs):
                        # Override task_type regardless of what the backend passed
                        pkwargs["config"] = genai_types.EmbedContentConfig(
                            task_type="RETRIEVAL_QUERY"
                        )
                        return _orig_embed_content(*pargs, **pkwargs)

                    _client.models.embed_content = _patched_embed_content
                    _patched = True
                    logger.debug(
                        "CBR embedding: overriding task_type to RETRIEVAL_QUERY",
                        tag="debug_google_emb",
                    )
                except Exception:
                    # Non-Google backend, missing genai, or import failure.
                    # Proceed with original RETRIEVAL_DOCUMENT default.
                    pass

            try:
                embeddings = loop_ref._orig_embedding_inner(
                    backend_self, input_content_list, *args, **kwargs
                )
            finally:
                # Always restore the original method, even on exception, so
                # subsequent document-indexing calls use the correct task_type.
                if _patched and _orig_embed_content is not None and _client is not None:
                    try:
                        _client.models.embed_content = _orig_embed_content
                    except Exception:
                        pass
            embedding_model = getattr(LLM_SETTINGS, "embedding_model", "unknown")
            embedding_input_tokens = 0
            try:
                from litellm import token_counter
                for txt in input_content_list:
                    embedding_input_tokens += int(token_counter(model=embedding_model, text=txt or ""))
            except Exception:
                embedding_input_tokens = 0

            if loop_ref._token_usage_active:
                phase = loop_ref._token_usage_phase if loop_ref._token_usage_phase in ("propose", "feedback") else "propose"
                loop_ref._token_usage_state["embedding_input_tokens"] += embedding_input_tokens
                loop_ref._token_usage_state["embedding_calls"] += 1
                loop_ref._token_usage_state["embedding_models"][embedding_model] = (
                    loop_ref._token_usage_state["embedding_models"].get(embedding_model, 0) + 1
                )
                phase_stat = loop_ref._token_usage_state["phase_breakdown"][phase]
                phase_stat["embedding_input_tokens"] += embedding_input_tokens
                phase_stat["embedding_calls"] += 1
            return embeddings

        backend_cls._create_chat_completion_inner_function = _wrapped_chat_completion_inner
        backend_cls._create_embedding_inner_function = _wrapped_embedding_inner

    def _finalize_token_usage_run(self) -> None:
        if not self._token_usage_active:
            return

        prompt_tokens = int(self._token_usage_state.get("prompt_tokens", 0))
        completion_tokens = int(self._token_usage_state.get("completion_tokens", 0))
        embedding_input_tokens = int(self._token_usage_state.get("embedding_input_tokens", 0))
        record = {
            "timestamp": time.time(),
            "loop_idx": getattr(self, "loop_idx", None),
            "run_started_at": self._token_usage_state.get("started_at"),
            "run_finished_at": time.time(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "embedding_input_tokens": embedding_input_tokens,
            "total_tokens": prompt_tokens + completion_tokens + embedding_input_tokens,
            "llm_calls": int(self._token_usage_state.get("llm_calls", 0)),
            "embedding_calls": int(self._token_usage_state.get("embedding_calls", 0)),
            "models": self._token_usage_state.get("models", {}),
            "embedding_models": self._token_usage_state.get("embedding_models", {}),
            "phase_breakdown": self._token_usage_state.get("phase_breakdown", {}),
        }

        logger.info(
            "CBR TOKENS: "
            f"loop={record['loop_idx']} cbr={'OFF' if CBR_DISABLE_RETRIEVAL else 'ON'} "
            f"prompt={record['prompt_tokens']} "
            f"completion={record['completion_tokens']} embedding_in={record['embedding_input_tokens']} "
            f"total={record['total_tokens']} calls={record['llm_calls']} emb_calls={record['embedding_calls']}"
        )

        backend_cls = APIBackend().__class__
        if self._orig_chat_completion_inner is not None:
            backend_cls._create_chat_completion_inner_function = self._orig_chat_completion_inner
            self._orig_chat_completion_inner = None
        if self._orig_embedding_inner is not None:
            backend_cls._create_embedding_inner_function = self._orig_embedding_inner
            self._orig_embedding_inner = None

        self._token_usage_active = False
        self._token_usage_phase = "idle"

    # ─── Helper Methods ───────────────────────────────

    def _resolve_exclude_competition(self, scenario) -> Optional[str]:
        raw = CBR_EXCLUDE_COMPETITION
        if raw is None:
            return None
        lowered = raw.strip().lower()
        if lowered in {"false", "0", "no", "off", "none"}:
            return None
        if lowered in {"true", "1", "yes", "on", "current"}:
            current_comp = getattr(scenario, "competition", None) if scenario is not None else None
            if current_comp:
                return str(current_comp)
            logger.warning("CBR_EXCLUDE_COMPETITION set to current alias but competition unavailable")
            return None
        return raw

    def _build_cbr_query(self, hypothesis, scenario) -> str:
        """Build a compact query text for KB retrieval.

        Description is truncated to QUERY_DESC_CHARS to keep embedding tokens low.
        """
        parts = []
        if hasattr(hypothesis, "component"):
            parts.append(f"Component: {hypothesis.component}")
        if hasattr(hypothesis, "hypothesis"):
            parts.append(f"Hypothesis: {hypothesis.hypothesis}")
        if hasattr(hypothesis, "problem_desc"):
            parts.append(f"Problem: {hypothesis.problem_desc}")
        if scenario:
            if hasattr(scenario, "task_type"):
                parts.append(f"Task: {scenario.task_type}")
            if hasattr(scenario, "data_type"):
                parts.append(f"Data: {scenario.data_type}")
            if hasattr(scenario, "description"):
                desc = _trunc(str(scenario.description), QUERY_DESC_CHARS)
                parts.append(f"Description: {desc}")
        return " | ".join(parts) if parts else "general ML problem"

    def _format_cbr_context(self, cases, failures, include_code: bool = True) -> str:
        """Format retrieved cases + failures as prompt context.

        Planning phase (include_code=False, used for hypothesis.appendix):
          Shows header + approach + techniques + plan_summary only.
          No code — the planning LLM doesn't need implementation details,
          and keeping the planning prompt lean saves significant tokens.

        Coding phase (include_code=True, used in task.description):
          Shows full code without truncation. At 300–500 lines per file
          and TOP_K=3 cases, total code is well within context limits.

        plan_summary is truncated to PLAN_SUMMARY_CHARS for display only —
        the full value is always stored in cases.json.
        """
        sections = []

        if cases:
            sections.append("## Retrieved Cases from CBR Knowledge Base")
            for i, (case, score) in enumerate(cases, 1):
                metric_val = (
                    f"{case.metrics.primary_metric_value:.4f}"
                    if case.metrics.primary_metric_value is not None else "N/A"
                )
                header = (
                    f"\n### Case {i} | sim={score:.2f} | {case.problem.task_type}/{case.problem.data_type}"
                    f" | model={case.solution.model_type} | {case.metrics.primary_metric_name}={metric_val}"
                    f" | reuse_ok={case.reuse_success_rate:.0%} | comp={case.source_competition}"
                )
                lines = [header]

                if case.solution.hypothesis_text:
                    lines.append(f"**Approach**: {case.solution.hypothesis_text}")

                if case.solution.key_techniques:
                    lines.append(f"**Techniques**: {', '.join(case.solution.key_techniques[:8])}")

                if case.solution.plan_summary:
                    summary = _trunc(case.solution.plan_summary, PLAN_SUMMARY_CHARS)
                    lines.append(f"**Plan**: {summary}")

                if include_code and HYPOTHESIS_CODE_CHARS != 0:
                    code = self.cbr_kb.get_full_code(case.case_id) or case.solution.code_snippet or ""
                    if code:
                        if HYPOTHESIS_CODE_CHARS > 0:
                            code = _trunc(code, HYPOTHESIS_CODE_CHARS)
                        lines.append(f"**Code**:\n```python\n{code}\n```")

                sections.append("\n".join(lines))

        if failures:
            sections.append(self.failure_tracker.format_for_prompt(failures))

        return "\n".join(sections)