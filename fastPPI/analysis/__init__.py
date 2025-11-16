"""
Analysis tools for preprocessing code.
"""

try:
    from .unified_tracer import (
        trace_unified_execution,
        categorize_operations,
        generate_operation_summary,
        UnifiedTracer
    )
    from .extended_codegen import (
        generate_operation_report,
        ExtendedCCodeGenerator
    )
    from .analyze import analyze_preprocessing_code
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

__all__ = []

if ANALYSIS_AVAILABLE:
    __all__.extend([
        "trace_unified_execution",
        "categorize_operations",
        "generate_operation_summary",
        "UnifiedTracer",
        "generate_operation_report",
        "ExtendedCCodeGenerator",
        "analyze_preprocessing_code",
    ])

