"""
Unified tracer that combines NumPy, pandas, and string tracing.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..tracers.tracer import ExecutionTracer, Operation
from ..tracers.pandas_tracer import PandasTracer, PandasOperation, PANDAS_AVAILABLE as PD_AVAIL

try:
    from ..tracers.string_tracer import StringTracer, StringOperation
    STRING_TRACER_AVAILABLE = True
except ImportError:
    STRING_TRACER_AVAILABLE = False
    StringTracer = None
    StringOperation = Operation


class UnifiedTracer:
    """Unified tracer for NumPy, pandas, and string operations."""
    
    def __init__(self):
        self.numpy_tracer = ExecutionTracer()
        self.pandas_tracer = PandasTracer() if PANDAS_AVAILABLE else None
        self.string_tracer = StringTracer() if STRING_TRACER_AVAILABLE else None
        self.all_operations: List[Union[Operation, PandasOperation, StringOperation]] = []
        
    def start_tracing(self):
        """Enable all tracers."""
        self.numpy_tracer.start_tracing()
        if self.pandas_tracer:
            self.pandas_tracer.start_tracing()
        if self.string_tracer:
            self.string_tracer.start_tracing()
        
    def stop_tracing(self):
        """Disable all tracers."""
        self.numpy_tracer.stop_tracing()
        if self.pandas_tracer:
            self.pandas_tracer.stop_tracing()
        if self.string_tracer:
            self.string_tracer.stop_tracing()
        
    def collect_operations(self):
        """Collect all operations from all tracers."""
        self.all_operations = []
        
        # Collect NumPy operations
        for op in self.numpy_tracer.operations:
            self.all_operations.append(op)
        
        # Collect pandas operations
        if self.pandas_tracer:
            for op in self.pandas_tracer.operations:
                self.all_operations.append(op)
        
        # Collect string operations
        if self.string_tracer:
            for op in self.string_tracer.operations:
                self.all_operations.append(op)
        
        # Renumber op_ids to ensure uniqueness (maintain execution order)
        # First, sort by original op_id to preserve order
        self.all_operations.sort(key=lambda op: (op.op_id, type(op).__name__))
        
        # Renumber with unique sequential IDs
        for new_id, op in enumerate(self.all_operations):
            op.op_id = new_id
        
        return self.all_operations


def trace_unified_execution(code: str, example_inputs: Dict[str, Any], 
                           globals_dict: Optional[Dict] = None) -> List:
    """
    Execute code with unified tracing (NumPy, pandas, string operations).
    
    Args:
        code: Python code string to execute
        example_inputs: Dictionary of input variable names to example values
        globals_dict: Optional globals dictionary for execution context
        
    Returns:
        List of all captured operations
    """
    tracer = UnifiedTracer()
    
    # Prepare execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        'np': np,
        'numpy': np,
    }
    
    if PANDAS_AVAILABLE:
        exec_globals['pd'] = pd
        exec_globals['pandas'] = pd
    
    # Add requests module if available (for http_get_json support)
    try:
        import requests
        exec_globals['requests'] = requests
    except ImportError:
        pass
    
    # Add string operation modules
    if STRING_TRACER_AVAILABLE:
        import re
        import unicodedata
        exec_globals['re'] = re
        exec_globals['unicodedata'] = unicodedata
    
    if globals_dict:
        exec_globals.update(globals_dict)
    
    # Add example inputs
    exec_globals.update(example_inputs)
    
    # Start tracing and execute
    tracer.start_tracing()
    try:
        exec(code, exec_globals)
    finally:
        tracer.stop_tracing()
    
    return tracer.collect_operations()


def categorize_operations(operations: List) -> Dict[str, List]:
    """
    Categorize operations by type.
    
    Returns:
        Dictionary with keys: 'numpy', 'pandas', 'string', 'compilable', 'non_compilable'
    """
    categorized = {
        'numpy': [],
        'pandas': [],
        'string': [],
        'compilable': [],
        'non_compilable': []
    }
    
    for op in operations:
        if isinstance(op, Operation) and not isinstance(op, (PandasOperation, StringOperation)):
            categorized['numpy'].append(op)
            categorized['compilable'].append(op)  # NumPy ops are compilable
        elif isinstance(op, PandasOperation):
            categorized['pandas'].append(op)
            # Determine if pandas op is compilable
            if op.op_name in ['df_mean', 'df_median', 'series_mean', 'series_median', 'df_fillna', 'series_fillna']:
                categorized['compilable'].append(op)
            else:
                categorized['non_compilable'].append(op)
        elif isinstance(op, StringOperation):
            categorized['string'].append(op)
            # String operations are generally compilable
            categorized['compilable'].append(op)
    
    return categorized


def generate_operation_summary(operations: List) -> str:
    """Generate a summary of captured operations."""
    categorized = categorize_operations(operations)
    
    summary = []
    summary.append(f"Total operations captured: {len(operations)}")
    summary.append(f"  - NumPy operations: {len(categorized['numpy'])}")
    summary.append(f"  - Pandas operations: {len(categorized['pandas'])}")
    summary.append(f"  - String operations: {len(categorized['string'])}")
    summary.append(f"")
    summary.append(f"Compilable operations: {len(categorized['compilable'])}")
    summary.append(f"Non-compilable operations: {len(categorized['non_compilable'])}")
    
    if categorized['non_compilable']:
        summary.append(f"")
        summary.append("Non-compilable operations:")
        for op in categorized['non_compilable']:
            if isinstance(op, PandasOperation):
                summary.append(f"  - Pandas: {op.op_name}")
    
    return "\n".join(summary)

