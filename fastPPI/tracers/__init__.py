"""
Operation tracers for NumPy, pandas, and string operations.
"""

from .tracer import trace_execution, ExecutionTracer, Operation

try:
    from .pandas_tracer import trace_pandas_execution, PandasTracer, PandasOperation
    EXTENDED_TRACERS = True
except ImportError:
    EXTENDED_TRACERS = False

try:
    from .string_tracer import trace_string_execution, StringTracer, StringOperation
    STRING_TRACERS = True
except ImportError:
    STRING_TRACERS = False

__all__ = [
    "trace_execution",
    "ExecutionTracer",
    "Operation",
]

if EXTENDED_TRACERS:
    __all__.extend([
        "trace_pandas_execution",
        "PandasTracer",
        "PandasOperation",
    ])

if STRING_TRACERS:
    __all__.extend([
        "trace_string_execution",
        "StringTracer",
        "StringOperation",
    ])

