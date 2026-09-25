"""Structured CBR Case schema integrated with R&D-Agent's experiment model.

A Case wraps an R&D-Agent DSExperiment + DSHypothesis with structured
metadata for retrieval, quality assessment, and cross-session persistence.

Key design decisions:
- Cases are JSON-serializable (no pickle) for transparency
- Embedding text is derived from problem + solution features
- Code snapshots are stored separately (file-based) to keep JSON lean
- Provenance tracking: every case knows which parent it was adapted from
- No truncation at store time — all fields persisted at full length
- plan_summary is built from structured concise_* fields of DSHypothesis,
  not from hypothesis.reason (which contains injected CBR context via appendix)
- Display-time truncation only happens in cbr_loop.py prompt formatting
- to_embedding_text() uses only the compact structured fields; no
  free-form blobs are included
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

def _trunc(text: str, max_chars: int, suffix: str = "…") -> str:
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + suffix


def _build_plan_summary(hypothesis) -> str:
    """Build compact plan_summary from structured DSHypothesis fields.

    Uses the concise_* fields which are purpose-built summaries generated
    by the LLM — no code, no injected CBR context (that lives in appendix).
    Falls back to a cleaned hypothesis.reason only when all concise fields
    are absent, stripping the appendix block in that case.
    """
    parts = []
    if getattr(hypothesis, "concise_observation", None):
        parts.append(f"Obs: {hypothesis.concise_observation}")
    if getattr(hypothesis, "concise_justification", None):
        parts.append(f"Why: {hypothesis.concise_justification}")
    if getattr(hypothesis, "concise_knowledge", None):
        parts.append(f"Knowledge: {hypothesis.concise_knowledge}")
    if getattr(hypothesis, "problem_desc", None):
        parts.append(f"Problem: {hypothesis.problem_desc}")
    if parts:
        return " | ".join(parts)

    # Fallback: reason without appendix block
    raw = getattr(hypothesis, "reason", "") or ""
    appendix = getattr(hypothesis, "appendix", None)
    if appendix and appendix in raw:
        raw = raw.replace(appendix, "").strip()
    return raw


# ─── Enums ────────────────────────────────────────────────────────

class CaseOutcome(str, Enum):
    """Uses str mixin so json.dumps works without custom encoder."""
    SUCCESS  = "success"
    PARTIAL  = "partial"   # ran but didn't beat baseline
    FAILURE  = "failure"   # runtime error
    UNTESTED = "untested"  # retained from external source, not yet validated


class ComponentType(str, Enum):
    """Maps to R&D-Agent's COMPLETE_ORDER in DSTrace."""
    DATA_LOADER = "DataLoadSpec"
    FEATURE_ENG = "FeatureEng"
    MODEL       = "Model"
    ENSEMBLE    = "Ensemble"
    WORKFLOW    = "Workflow"
    PIPELINE    = "Pipeline"
    UNKNOWN     = "unknown"


# ─── Sub-structures ───────────────────────────────────────────────

@dataclass
class ProblemSignature:
    """Structured problem description for similarity matching."""
    task_type: str = "unknown"
    data_type: str = "unknown"
    eval_metric: str = "unknown"
    eval_direction: str = "maximize"
    target_component: str = "Pipeline"
    data_description: str = ""
    competition_name: str = ""
    raw_description: str = ""          # full text, no truncation at store time
    constraints: List[str] = field(default_factory=list)


@dataclass
class SolutionSignature:
    """What was actually done — extracted from code + hypothesis."""
    hypothesis_text: str = ""
    plan_summary: str = ""             # from concise_* fields, no truncation at store time
    model_type: str = "unknown"
    key_techniques: List[str] = field(default_factory=list)
    code_hash: str = ""
    code_path: str = ""
    code_snippet: str = ""             # full code stored in separate .py file
    workspace_files: List[str] = field(default_factory=list)


@dataclass
class CaseMetrics:
    """Quantitative outcome tracking."""
    primary_metric_name: str = "unknown"
    primary_metric_value: Optional[float] = None
    baseline_value: Optional[float] = None
    improvement_pct: Optional[float] = None
    execution_time_seconds: Optional[float] = None
    n_costeer_iterations: int = 0
    n_rd_loops: int = 0
    error_message: Optional[str] = None

    def compute_improvement(self):
        if (self.primary_metric_value is not None
                and self.baseline_value is not None
                and self.baseline_value != 0):
            self.improvement_pct = (
                (self.primary_metric_value - self.baseline_value)
                / abs(self.baseline_value)
            ) * 100.0


# ─── Main Case class ──────────────────────────────────────────────

@dataclass
class Case:
    """A complete CBR case for the R&D-Agent knowledge base."""

    # Identity
    case_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: int = 1

    # Core CBR triad
    problem: ProblemSignature = field(default_factory=ProblemSignature)
    solution: SolutionSignature = field(default_factory=SolutionSignature)
    outcome: CaseOutcome = CaseOutcome.UNTESTED
    metrics: CaseMetrics = field(default_factory=CaseMetrics)

    # Provenance & reuse tracking
    source_case_id: Optional[str] = None
    source_competition: str = ""
    adaptation_delta: str = ""
    reuse_count: int = 0
    reuse_success_count: int = 0

    # R&D-Agent trace reference
    trace_loop_idx: Optional[int] = None
    trace_node_idx: Optional[int] = None

    # ─── Serialization ────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        d["created_at"] = self._format_timestamp(self.created_at)
        d["updated_at"] = self._format_timestamp(self.updated_at)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Case:
        d["created_at"] = cls._parse_timestamp(d.get("created_at", time.time()))
        d["updated_at"] = cls._parse_timestamp(d.get("updated_at", d["created_at"]))
        d["outcome"] = CaseOutcome(d["outcome"])
        d["problem"] = ProblemSignature(**d["problem"])
        d["solution"] = SolutionSignature(**d["solution"])
        d["metrics"] = CaseMetrics(**d["metrics"])

        if (
            (not d["problem"].eval_metric or d["problem"].eval_metric.lower() == "unknown")
            and d["metrics"].primary_metric_name
            and d["metrics"].primary_metric_name.lower() != "unknown"
        ):
            d["problem"].eval_metric = d["metrics"].primary_metric_name

        if not d["solution"].model_type or d["solution"].model_type.lower() == "unknown":
            inferred_model = cls._infer_model_type(
                d["solution"].hypothesis_text,
                d["solution"].plan_summary,
                d["solution"].code_snippet,
            )
            if inferred_model != "unknown":
                d["solution"].model_type = inferred_model

        return cls(**d)

    # ─── Embedding text ───────────────────────────────

    def to_embedding_text(self) -> str:
        """Compact structured text for embedding — no large blobs.

        Deliberately excludes plan_summary and raw code to keep
        the embedding input focused on retrieval-relevant signals.
        raw_description is included at full length (stored without truncation).
        """
        parts = [
            f"Task: {self.problem.task_type}",
            f"Data: {self.problem.data_type}",
            f"Metric: {self.problem.eval_metric}",
            f"Component: {self.problem.target_component}",
            f"Competition: {self.problem.competition_name}",
        ]
        if self.problem.raw_description:
            parts.append(f"Problem: {self.problem.raw_description}")

        if self.solution.key_techniques:
            parts.append(f"Techniques: {', '.join(self.solution.key_techniques)}")
        if self.solution.hypothesis_text:
            parts.append(f"Approach: {self.solution.hypothesis_text}")

        if self.metrics.primary_metric_value is not None:
            parts.append(
                f"Result: {self.metrics.primary_metric_name}"
                f"={self.metrics.primary_metric_value:.4f}"
            )

        return " | ".join(parts)

    # ─── Factory: build from R&D-Agent objects ────────

    @classmethod
    def from_rd_agent(
        cls,
        hypothesis,
        experiment,
        scenario,
        metric_value: Optional[float] = None,
        baseline_value: Optional[float] = None,
        execution_feedback: str = "",
        loop_idx: Optional[int] = None,
        node_idx: Optional[int] = None,
        source_case_id: Optional[str] = None,
    ) -> Case:
        """Factory: creates a Case from R&D-Agent's native objects.

        No truncation at store time — all fields are persisted at full length.
        Display-time truncation happens in cbr_loop.py prompt formatting only.
        CBR context is written to hypothesis.appendix (not reason) so
        plan_summary stays clean.
        """
        legacy_metric_direction = getattr(scenario, "evaluation_metric_direction", None)
        scenario_metric_name = getattr(scenario, "metric_name", None)
        if (
            (not scenario_metric_name or str(scenario_metric_name).strip().lower() == "unknown")
            and isinstance(legacy_metric_direction, (list, tuple))
            and len(legacy_metric_direction) > 0
        ):
            scenario_metric_name = legacy_metric_direction[0]

        scenario_metric_direction = getattr(scenario, "metric_direction", None)
        if scenario_metric_direction is None and isinstance(legacy_metric_direction, (list, tuple)):
            if len(legacy_metric_direction) > 1:
                scenario_metric_direction = legacy_metric_direction[1]

        if isinstance(scenario_metric_direction, str):
            direction_value = scenario_metric_direction.strip().lower()
            direction_is_maximize = direction_value in {"maximize", "max", "higher", "larger", "true", "1"}
        elif isinstance(scenario_metric_direction, bool):
            direction_is_maximize = scenario_metric_direction
        else:
            direction_is_maximize = True

        # Full raw_description — no truncation at store time
        raw_desc = getattr(scenario, "description", "") or ""

        problem = ProblemSignature(
            task_type=getattr(scenario, "task_type", "unknown"),
            data_type=getattr(scenario, "data_type", "tabular"),
            eval_metric=str(scenario_metric_name).strip() if scenario_metric_name else "unknown",
            eval_direction="maximize" if direction_is_maximize else "minimize",
            target_component=getattr(hypothesis, "component", "Pipeline"),
            competition_name=getattr(scenario, "competition", ""),
            raw_description=raw_desc,
            data_description=getattr(scenario, "data_folder_description", ""),
        )

        # Build code snapshot
        code_snapshot = ""
        workspace_files = []
        if hasattr(experiment, "experiment_workspace") and experiment.experiment_workspace:
            ws = experiment.experiment_workspace
            if hasattr(ws, "file_dict"):
                workspace_files = list(ws.file_dict.keys())
                py_files = {k: v for k, v in ws.file_dict.items() if k.endswith(".py")}
                if py_files:
                    preferred = ["main.py", "model_01.py", "feature.py"]
                    ordered_files = [f for f in preferred if f in py_files]
                    ordered_files.extend(sorted(f for f in py_files if f not in ordered_files))
                    chunks = []
                    for fname in ordered_files:
                        content = py_files.get(fname, "")
                        if content:
                            chunks.append(f"# ===== FILE: {fname} =====\n{content.rstrip()}\n")
                    code_snapshot = "\n\n".join(chunks).strip()

        code_hash = hashlib.sha256(code_snapshot.encode()).hexdigest()[:16] if code_snapshot else ""
        inferred_model_type = cls._infer_model_type(
            getattr(hypothesis, "hypothesis", ""),
            getattr(hypothesis, "reason", ""),
            code_snapshot,
        )

        # plan_summary from structured concise_* fields — no truncation at store time.
        # CBR context lives in hypothesis.appendix and is excluded here.
        plan_summary = _build_plan_summary(hypothesis)

        solution = SolutionSignature(
            hypothesis_text=getattr(hypothesis, "hypothesis", ""),
            plan_summary=plan_summary,
            model_type=inferred_model_type,
            code_hash=code_hash,
            code_snippet=code_snapshot,
            workspace_files=workspace_files,
        )

        metric_name = problem.eval_metric
        if not metric_name or metric_name.lower() == "unknown":
            inferred_metric = cls._infer_metric_name_from_experiment(experiment)
            if inferred_metric:
                metric_name = inferred_metric

        metrics = CaseMetrics(
            primary_metric_name=metric_name,
            primary_metric_value=metric_value,
            baseline_value=baseline_value,
        )
        metrics.compute_improvement()

        if metric_value is None:
            outcome = CaseOutcome.FAILURE
        elif baseline_value is not None and metric_value > baseline_value:
            outcome = CaseOutcome.SUCCESS
        elif metric_value is not None:
            outcome = CaseOutcome.PARTIAL
        else:
            outcome = CaseOutcome.UNTESTED

        return cls(
            problem=problem,
            solution=solution,
            outcome=outcome,
            metrics=metrics,
            source_case_id=source_case_id,
            source_competition=problem.competition_name,
            trace_loop_idx=loop_idx,
            trace_node_idx=node_idx,
        )

    @staticmethod
    def _infer_metric_name_from_experiment(experiment) -> str:
        result = getattr(experiment, "result", None)
        if result is None or not hasattr(result, "columns"):
            return "unknown"
        for col in result.columns:
            if col is None:
                continue
            name = str(col).strip()
            if not name:
                continue
            lowered = name.lower()
            if lowered in {"unknown", "score", "value", "metric", "unnamed: 0"}:
                continue
            return name
        return "unknown"

    @staticmethod
    def _infer_model_type(*texts: Optional[str]) -> str:
        """Infer model type from code and/or text descriptions.

        Strategy:
          1. Code (last argument, if it looks like Python) is searched with
             import/instantiation regexes — precise, strips comments first.
          2. Text fields (hypothesis, plan_summary) are searched with looser
             keyword matching.
          3. All matching model families are collected.
             - Single family  → "xgboost"
             - Multiple       → "ensemble (lightgbm + xgboost)"
             - None           → "unknown"
        """
        import re as _re

        # ── Helpers ────────────────────────────────────────────
        def _strip_docstrings_and_comments(src: str) -> str:
            src = _re.sub(r'"""[\s\S]*?"""', '', src)
            src = _re.sub(r"'''[\s\S]*?'''", '', src)
            lines = []
            for line in src.split('\n'):
                s = line.lstrip()
                if s.startswith('#'):
                    continue
                if '#' in line:
                    line = line[:line.index('#')]
                lines.append(line)
            return '\n'.join(lines)

        # ── Separate code from descriptive text ────────────────
        # Heuristic: the last non-empty text is the code snapshot when
        # it contains common Python keywords (import, def, class, =).
        all_texts = [t for t in texts if isinstance(t, str) and t.strip()]
        code_text = ""
        prose_texts = []
        for t in all_texts:
            if _re.search(r'\b(?:import|def |class |if |for |return )\b', t):
                code_text += "\n" + t
            else:
                prose_texts.append(t)

        clean_code = _strip_docstrings_and_comments(code_text).lower()
        prose = "\n".join(prose_texts).lower()

        # ── Detection rules ────────────────────────────────────
        # (family, code_patterns, prose_keywords)
        # code_patterns: matched against clean executable code only
        # prose_keywords: matched against hypothesis/plan text too
        rules = [
            ("xgboost",
             [r'import\s+xgboost', r'from\s+xgboost\s+import',
              r'xgb\.xgb', r'xgbregressor\s*\(', r'xgbclassifier\s*\('],
             ["xgboost", "xgbregressor", "xgbclassifier"]),
            ("lightgbm",
             [r'import\s+lightgbm', r'from\s+lightgbm\s+import',
              r'lgb\.lgbm', r'lgbmregressor\s*\(', r'lgbmclassifier\s*\('],
             ["lightgbm", "lgbmregressor", "lgbmclassifier"]),
            ("catboost",
             [r'import\s+catboost', r'from\s+catboost\s+import',
              r'catboostregressor\s*\(', r'catboostclassifier\s*\('],
             ["catboost"]),
            ("random_forest",
             [r'randomforestregressor\s*\(', r'randomforestclassifier\s*\(',
              r'extratreesregressor\s*\(', r'extratreesclassifier\s*\('],
             ["randomforestclassifier", "randomforestregressor", "random forest"]),
            ("gradient_boosting",
             [r'gradientboostingregressor\s*\(', r'gradientboostingclassifier\s*\(',
              r'histgradientboostingregressor\s*\(', r'histgradientboostingclassifier\s*\('],
             ["gradientboosting", "histgradientboosting"]),
            ("adaboost",
             [r'adaboostregressor\s*\(', r'adaboostclassifier\s*\('],
             ["adaboost"]),
            ("linear_model",
             [r'linearregression\s*\(', r'logisticregression\s*\(',
              r'\bridge\s*\(', r'\blasso\s*\(', r'elasticnet\s*\('],
             ["linearregression", "logisticregression", "elasticnet", "linear model"]),
            ("svm",
             [r'(?:linear)?svc\s*\(', r'(?:linear)?svr\s*\(',
              r'from\s+sklearn\.svm\s+import'],
             ["support vector", "svc(", "svr("]),
            ("knn",
             [r'kneighborsregressor\s*\(', r'kneighborsclassifier\s*\('],
             ["kneighbors", "k-nearest"]),
            ("transformer",
             [r'from\s+transformers\s+import', r'bertmodel\s*\(', r'roberta'],
             ["transformer", "bert", "roberta", "deberta", "distilbert", "gpt"]),
            ("neural_network",
             [r'import\s+torch', r'from\s+torch\s+import',
              r'import\s+tensorflow', r'from\s+tensorflow',
              r'import\s+keras', r'from\s+keras',
              r'nn\.module', r'nn\.linear', r'nn\.sequential',
              r'mlpclassifier\s*\(', r'mlpregressor\s*\('],
             ["neural network", "deep learning", "mlpclassifier", "mlpregressor"]),
        ]

        found: list = []
        seen: set = set()
        for family, code_pats, prose_keys in rules:
            if family in seen:
                continue
            hit = (
                any(_re.search(p, clean_code) for p in code_pats)
                or any(k in prose for k in prose_keys)
            )
            if hit:
                found.append(family)
                seen.add(family)

        # Explicit ensemble keywords always override / supplement
        ensemble_keywords = ["stacking", "blending", "votingclassifier",
                             "votingregressor", "stackingclassifier"]
        if any(k in clean_code or k in prose for k in ensemble_keywords):
            if len(found) < 2:
                return "ensemble"
            # fall through to multi-family ensemble label below

        if not found:
            return "unknown"
        if len(found) == 1:
            return found[0]
        return f"ensemble ({' + '.join(sorted(found))})"

    @staticmethod
    def _format_timestamp(ts: float) -> str:
        return datetime.fromtimestamp(float(ts)).astimezone().isoformat(timespec="seconds")

    @staticmethod
    def _parse_timestamp(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return time.time()
            try:
                return float(raw)
            except ValueError:
                pass
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return time.time()
        return time.time()

    def record_reuse(self, was_successful: bool):
        self.reuse_count += 1
        if was_successful:
            self.reuse_success_count += 1
        self.updated_at = time.time()

    @property
    def reuse_success_rate(self) -> float:
        if self.reuse_count == 0:
            return 0.0
        return self.reuse_success_count / self.reuse_count

    def __repr__(self):
        return (
            f"Case({self.case_id} | {self.problem.task_type}/{self.problem.data_type} | "
            f"{self.outcome.value} | {self.metrics.primary_metric_name}"
            f"={self.metrics.primary_metric_value})"
        )