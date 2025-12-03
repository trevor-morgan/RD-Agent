"""Pandas Pitfall Detection System.

This module provides tools to detect, diagnose, and correct common pandas
coding pitfalls that cause silent failures in factor implementations.

Key Components:
- PitfallLinter: AST-based pre-execution code analysis
- RuntimeDiagnostics: Post-execution DataFrame issue diagnosis
- PitfallCorrector: Automated code correction suggestions
- PitfallKnowledgeBridge: Integration with CoSTEER knowledge graph

Example usage:
    from rdagent.components.coder.pitfall_detector import (
        PitfallLinter,
        lint_factor_code,
        has_critical_pitfalls,
    )

    # Quick check
    if has_critical_pitfalls(code):
        print("Code has critical issues!")

    # Detailed linting
    linter = PitfallLinter()
    results = linter.lint_code(code)
    for result in results:
        print(f"[{result.pitfall.id}] {result.pitfall.name}: {result.suggested_fix}")
"""

from rdagent.components.coder.pitfall_detector.corrector import (
    CodeCorrection,
    PitfallCorrector,
    auto_correct_factor_code,
    suggest_factor_corrections,
)
from rdagent.components.coder.pitfall_detector.diagnostics import (
    DataFrameAnalysis,
    RuntimeDiagnostics,
    diagnose_factor_result,
)
from rdagent.components.coder.pitfall_detector.knowledge_bridge import (
    PitfallKnowledgeBridge,
    PitfallKnowledgeEntry,
    get_pitfall_knowledge_bridge,
    get_pitfall_prompt_context,
)
from rdagent.components.coder.pitfall_detector.linter import (
    PitfallLinter,
    has_critical_pitfalls,
    lint_factor_code,
)
from rdagent.components.coder.pitfall_detector.patterns import (
    PANDAS_PITFALLS,
    Diagnosis,
    LintResult,
    PitfallPattern,
    get_critical_pitfalls,
    get_pitfall_by_id,
)

__all__ = [
    "CodeCorrection",
    "DataFrameAnalysis",
    "Diagnosis",
    "LintResult",
    "PANDAS_PITFALLS",
    "PitfallCorrector",
    "PitfallKnowledgeBridge",
    "PitfallKnowledgeEntry",
    "PitfallLinter",
    "PitfallPattern",
    "RuntimeDiagnostics",
    "auto_correct_factor_code",
    "diagnose_factor_result",
    "get_critical_pitfalls",
    "get_pitfall_by_id",
    "get_pitfall_knowledge_bridge",
    "get_pitfall_prompt_context",
    "has_critical_pitfalls",
    "lint_factor_code",
    "suggest_factor_corrections",
]
