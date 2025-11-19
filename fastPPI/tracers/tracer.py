"""
Execution tracer that captures operations during Python code execution.
Tracks NumPy operations, array operations, and mathematical computations.
"""

import sys
import types
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import inspect


class Operation:
    """Represents a single operation in the computational graph."""
    
    def __init__(self, op_name: str, func: Callable, args: tuple, kwargs: dict, result: Any, op_id: int):
        self.op_name = op_name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = result
        self.op_id = op_id
        self.input_shapes = []
        self.output_shape = None
        self.dtype = None
        
        # Extract shape and dtype information
        if isinstance(result, np.ndarray):
            self.output_shape = result.shape
            self.dtype = result.dtype
            for arg in args:
                if isinstance(arg, np.ndarray):
                    self.input_shapes.append(arg.shape)
                elif isinstance(arg, (int, float)):
                    self.input_shapes.append(None)
        elif isinstance(result, (int, float, bool)):
            self.output_shape = ()
            self.dtype = type(result)
            
    def __repr__(self):
        return f"Operation(op_name='{self.op_name}', op_id={self.op_id}, shape={self.output_shape})"


class ExecutionTracer:
    """Traces Python execution to capture operations."""
    
    def __init__(self):
        self.operations: List[Operation] = []
        self.op_counter = 0
        self.array_registry: Dict[int, np.ndarray] = {}  # id(obj) -> array
        self.enabled = False
        self.original_functions = {}
        
    def start_tracing(self):
        """Enable tracing."""
        self.enabled = True
        self._patch_numpy()
        
    def stop_tracing(self):
        """Disable tracing and restore original functions."""
        self.enabled = False
        self._unpatch_numpy()
        
    def _patch_numpy(self):
        """Patch NumPy functions to capture calls."""
        # Store original functions
        self.original_functions = {
            'add': np.add,
            'subtract': np.subtract,
            'multiply': np.multiply,
            'divide': np.divide,
            'dot': np.dot,
            'matmul': np.matmul,
            'sum': np.sum,
            'mean': np.mean,
            'std': np.std,
            'max': np.max,
            'min': np.min,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'reshape': np.reshape,
            'transpose': np.transpose,
            'concatenate': np.concatenate,
            'stack': np.stack,
            'zeros': np.zeros,
            'ones': np.ones,
            'array': np.array,
            'arange': np.arange,
            'where': np.where,
            'clip': np.clip,
            'round': np.round,
            'floor': np.floor,
            'ceil': np.ceil,
        }
        
        # Wrap functions
        for name, original_func in self.original_functions.items():
            wrapped = self._wrap_function(name, original_func)
            setattr(np, name, wrapped)
            # Note: np.ndarray is immutable in newer NumPy versions, so we can't patch methods directly
            # Module-level functions are sufficient for tracing
    
    def _unpatch_numpy(self):
        """Restore original NumPy functions."""
        for name, original_func in self.original_functions.items():
            setattr(np, name, original_func)
    
    def _wrap_function(self, name: str, original_func: Callable) -> Callable:
        """Wrap a NumPy function to capture its execution."""
        # Check if this is a ufunc (has .reduce, .accumulate, etc.)
        is_ufunc = hasattr(original_func, 'reduce') or hasattr(original_func, '__call__')
        
        # Create a wrapper class that preserves ufunc interface
        class FunctionWrapper:
            def __init__(self, tracer_instance, original, func_name):
                self._tracer = tracer_instance
                self._original = original
                self._name = func_name
            
            def __call__(self, *args, **kwargs):
                if not self._tracer.enabled:
                    return self._original(*args, **kwargs)
                
                # Execute the original function
                result = self._original(*args, **kwargs)
                
                # Record the operation
                self._tracer.op_counter += 1
                op = Operation(self._name, self._original, args, kwargs, result, self._tracer.op_counter)
                self._tracer.operations.append(op)
                
                # Register arrays in the registry
                if isinstance(result, np.ndarray):
                    self._tracer.array_registry[id(result)] = result
                for arg in args:
                    if isinstance(arg, np.ndarray):
                        self._tracer.array_registry[id(arg)] = arg
                        
                return result
            
            def __getattr__(self, attr):
                # Forward attribute access to original function (for .reduce, .accumulate, etc.)
                return getattr(self._original, attr)
        
        wrapped = FunctionWrapper(self, original_func, name)
        return wrapped
    
    def capture_operation(self, op_name: str, func: Callable, *args, **kwargs) -> Any:
        """Manually capture an operation."""
        if not self.enabled:
            return func(*args, **kwargs)
        
        result = func(*args, **kwargs)
        self.op_counter += 1
        op = Operation(op_name, func, args, kwargs, result, self.op_counter)
        self.operations.append(op)
        return result


# Global tracer instance
_global_tracer = ExecutionTracer()


def trace_execution(user_code: str, example_inputs: Dict[str, Any], 
                   globals_dict: Optional[Dict] = None) -> List[Operation]:
    """
    Execute user code with example inputs and trace all operations.
    
    Args:
        user_code: Python code string to execute
        example_inputs: Dictionary of input variable names to example values
        globals_dict: Optional globals dictionary for execution context
        
    Returns:
        List of captured operations
    """
    # Validate and convert user_code to string if needed
    if not isinstance(user_code, (str, bytes)):
        raise TypeError(
            f"trace_execution() expects user_code to be a string or bytes, "
            f"got {type(user_code).__name__}. "
            f"If you're passing a function, convert it to a code string first."
        )
    
    # Convert bytes to string if needed
    if isinstance(user_code, bytes):
        user_code = user_code.decode('utf-8')
    
    tracer = ExecutionTracer()
    
    # Prepare execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        'np': np,
        'numpy': np,
    }
    
    # Add pandas if available
    try:
        import pandas as pd
        exec_globals['pd'] = pd
        exec_globals['pandas'] = pd
    except ImportError:
        pass
    
    # Add argparse to handle scripts that import it
    import argparse
    exec_globals['argparse'] = argparse
    
    if globals_dict:
        exec_globals.update(globals_dict)
    
    # Add example inputs to execution environment
    exec_globals.update(example_inputs)
    
    # Start tracing and execute
    tracer.start_tracing()
    try:
        exec(user_code, exec_globals)
    finally:
        tracer.stop_tracing()
    
    return tracer.operations

