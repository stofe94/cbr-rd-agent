"""CBR (Case-Based Reasoning) extension for R&D-Agent Data Science scenario."""
from .case_schema import Case, CaseOutcome, ProblemSignature, SolutionSignature, CaseMetrics
from .case_knowledge import CBRKnowledgeBase
from .quality_gate import QualityGate, RetentionDecision
from .failure_tracker import FailureTracker
from .cbr_loop import CBRDataScienceRDLoop

__all__ = [
    "Case", "CaseOutcome", "ProblemSignature", "SolutionSignature", "CaseMetrics",
    "CBRKnowledgeBase",
    "QualityGate", "RetentionDecision",
    "FailureTracker",
    "CBRDataScienceRDLoop",
]