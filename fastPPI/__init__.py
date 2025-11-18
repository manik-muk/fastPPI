"""
FastPPI: Fast Preprocessing Pipeline Interpreter
Converts Python preprocessing code to optimized C/C++ binaries
Supports NumPy and pandas operations
"""

__version__ = "0.2.0"

# Core functionality
from .core import compile_to_binary, generate_c_code, build_computational_graph
from .tracers import trace_execution

# Extended functionality for pandas
try:
    from .analysis import (
        trace_unified_execution,
        categorize_operations,
        generate_operation_report
    )
    from .tracers import trace_pandas_execution
    EXTENDED_SUPPORT = True
except ImportError:
    EXTENDED_SUPPORT = False

__all__ = [
    "compile_to_binary",
    "trace_execution",
    "generate_c_code",
    "build_computational_graph",
]

if EXTENDED_SUPPORT:
    __all__.extend([
        "trace_unified_execution",
        "categorize_operations",
        "trace_pandas_execution",
        "generate_operation_report",
    ])

