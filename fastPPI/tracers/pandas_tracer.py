"""
Pandas operation tracer for FastPPI.
Captures pandas DataFrame and Series operations.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable
import inspect

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .tracer import Operation
from .lambda_analyzer import analyze_lambda


class PandasOperation(Operation):
    """Extended operation class for pandas operations."""
    
    def __init__(self, op_name: str, func: Callable, args: tuple, kwargs: dict, 
                 result: Any, op_id: int, obj_type: str = "unknown"):
        super().__init__(op_name, func, args, kwargs, result, op_id)
        self.obj_type = obj_type  # "DataFrame", "Series", etc.
        self.column_name = None  # For column-specific operations
        self.lambda_info = None  # For apply() operations with lambda functions
        
    def __repr__(self):
        return f"PandasOperation(op_name='{self.op_name}', obj_type='{self.obj_type}', id={self.op_id})"


class PandasTracer:
    """Traces pandas operations during execution."""
    
    def __init__(self):
        self.operations: List[PandasOperation] = []
        self.op_counter = 0
        self.enabled = False
        self.original_functions = {}
        self.dataframe_registry: Dict[int, pd.DataFrame] = {}
        self.series_registry: Dict[int, pd.Series] = {}
        
    def start_tracing(self):
        """Enable tracing of pandas operations."""
        self.enabled = True
        self._patch_pandas()
        
    def stop_tracing(self):
        """Disable tracing and restore original functions."""
        self.enabled = False
        self._unpatch_pandas()
        
    def _patch_pandas(self):
        """Patch pandas functions and methods to capture calls."""
        # Import StringMethods
        from pandas.core.strings.accessor import StringMethods
        
        # Store original DataFrame methods
        self.original_functions = {
            # DataFrame methods
            'df_mean': pd.DataFrame.mean,
            'df_median': pd.DataFrame.median,
            'df_fillna': pd.DataFrame.fillna,
            'df_dropna': pd.DataFrame.dropna,
            'df_apply': pd.DataFrame.apply,
            'df_astype': pd.DataFrame.astype,
            'df_getitem': pd.DataFrame.__getitem__,  # Column access
            
            # Series methods
            'series_mean': pd.Series.mean,
            'series_median': pd.Series.median,
            'series_fillna': pd.Series.fillna,
            'series_apply': pd.Series.apply,
            'series_astype': pd.Series.astype,
            
            # StringMethods (for .str operations)
            'str_lower': StringMethods.lower,
            'str_upper': StringMethods.upper,
            'str_strip': StringMethods.strip,
            'str_replace': StringMethods.replace,
            'str_contains': StringMethods.contains,
            
            # Top-level functions
            'read_csv': pd.read_csv,
            'DataFrame': pd.DataFrame,  # For patching DataFrame constructor
        }
        
        # Track requests.get calls for http_get_json pattern
        self.pending_http_requests = {}  # response_id -> url
        self.json_responses = {}  # json_data_id -> response_id (to track which response produced which JSON)
        
        # Patch methods
        pd.DataFrame.mean = self._wrap_method('df_mean', self.original_functions['df_mean'], 'DataFrame')
        pd.DataFrame.median = self._wrap_method('df_median', self.original_functions['df_median'], 'DataFrame')
        pd.DataFrame.fillna = self._wrap_method('df_fillna', self.original_functions['df_fillna'], 'DataFrame')
        pd.DataFrame.dropna = self._wrap_method('df_dropna', self.original_functions['df_dropna'], 'DataFrame')
        pd.DataFrame.apply = self._wrap_method('df_apply', self.original_functions['df_apply'], 'DataFrame')
        pd.DataFrame.astype = self._wrap_method('df_astype', self.original_functions['df_astype'], 'DataFrame')
        pd.DataFrame.__getitem__ = self._wrap_getitem('df_getitem', self.original_functions['df_getitem'])
        
        pd.Series.mean = self._wrap_method('series_mean', self.original_functions['series_mean'], 'Series')
        pd.Series.median = self._wrap_method('series_median', self.original_functions['series_median'], 'Series')
        pd.Series.fillna = self._wrap_method('series_fillna', self.original_functions['series_fillna'], 'Series')
        pd.Series.apply = self._wrap_method('series_apply', self.original_functions['series_apply'], 'Series')
        pd.Series.astype = self._wrap_method('series_astype', self.original_functions['series_astype'], 'Series')
        
        # Patch StringMethods directly
        StringMethods.lower = self._wrap_str_method('str_lower', self.original_functions['str_lower'])
        StringMethods.upper = self._wrap_str_method('str_upper', self.original_functions['str_upper'])
        StringMethods.strip = self._wrap_str_method('str_strip', self.original_functions['str_strip'])
        StringMethods.replace = self._wrap_str_method('str_replace', self.original_functions['str_replace'])
        StringMethods.contains = self._wrap_str_method('str_contains', self.original_functions['str_contains'])
        
        pd.read_csv = self._wrap_function('read_csv', self.original_functions['read_csv'])
        
        # Patch DataFrame constructor to detect http_get_json pattern
        pd.DataFrame = self._wrap_dataframe_constructor('DataFrame', self.original_functions['DataFrame'])
        
        # Patch requests.get if available
        try:
            import requests
            self.original_functions['requests_get'] = requests.get
            requests.get = self._wrap_requests_get('requests_get', self.original_functions['requests_get'])
            # Also patch response.json() method
            if hasattr(requests.Response, 'json'):
                self.original_functions['response_json'] = requests.Response.json
                requests.Response.json = self._wrap_response_json('response_json', self.original_functions['response_json'])
        except ImportError:
            pass
        
    def _unpatch_pandas(self):
        """Restore original pandas functions."""
        from pandas.core.strings.accessor import StringMethods
        
        pd.DataFrame.mean = self.original_functions['df_mean']
        pd.DataFrame.median = self.original_functions['df_median']
        pd.DataFrame.fillna = self.original_functions['df_fillna']
        pd.DataFrame.dropna = self.original_functions['df_dropna']
        pd.DataFrame.apply = self.original_functions['df_apply']
        pd.DataFrame.astype = self.original_functions['df_astype']
        pd.DataFrame.__getitem__ = self.original_functions['df_getitem']
        
        pd.Series.mean = self.original_functions['series_mean']
        pd.Series.median = self.original_functions['series_median']
        pd.Series.fillna = self.original_functions['series_fillna']
        pd.Series.apply = self.original_functions['series_apply']
        pd.Series.astype = self.original_functions['series_astype']
        
        # Restore StringMethods
        StringMethods.lower = self.original_functions['str_lower']
        StringMethods.upper = self.original_functions['str_upper']
        StringMethods.strip = self.original_functions['str_strip']
        StringMethods.replace = self.original_functions['str_replace']
        StringMethods.contains = self.original_functions['str_contains']
        
        pd.read_csv = self.original_functions['read_csv']
        pd.DataFrame = self.original_functions['DataFrame']
        
        # Restore requests.get if it was patched
        try:
            import requests
            if 'requests_get' in self.original_functions:
                requests.get = self.original_functions['requests_get']
            if 'response_json' in self.original_functions:
                requests.Response.json = self.original_functions['response_json']
        except ImportError:
            pass
        
        # Clear pending requests
        self.pending_http_requests = {}
        self.json_responses = {}
    
    def _wrap_method(self, name: str, original_method: Callable, obj_type: str) -> Callable:
        """Wrap a pandas method to capture its execution."""
        def wrapped(self_obj, *args, **kwargs):
            if not self.enabled:
                return original_method(self_obj, *args, **kwargs)
            
            # Execute the original method
            result = original_method(self_obj, *args, **kwargs)
            
            # Record the operation
            self.op_counter += 1
            op = PandasOperation(name, original_method, (self_obj,) + args, kwargs, 
                               result, self.op_counter, obj_type)
            
            # Register results safely
            if PANDAS_AVAILABLE and pd is not None:
                try:
                    if isinstance(result, pd.DataFrame):
                        self.dataframe_registry[id(result)] = result
                    elif isinstance(result, pd.Series):
                        self.series_registry[id(result)] = result
                except (TypeError, AttributeError):
                    pass
            
            # Special handling for apply() with lambda functions
            if name in ('series_apply', 'df_apply') and len(args) > 0:
                func_arg = args[0]
                if callable(func_arg) and (inspect.isfunction(func_arg) or inspect.ismethod(func_arg)):
                    # Try to analyze the lambda/function
                    try:
                        # Get external variables from the closure
                        external_vars = {}
                        if hasattr(func_arg, '__closure__') and func_arg.__closure__:
                            if hasattr(func_arg, '__code__'):
                                freevars = func_arg.__code__.co_freevars
                                for i, var_name in enumerate(freevars):
                                    if i < len(func_arg.__closure__):
                                        cell = func_arg.__closure__[i]
                                        external_vars[var_name] = cell.cell_contents
                        
                        lambda_info = analyze_lambda(func_arg, external_vars)
                        if lambda_info:
                            op.lambda_info = lambda_info
                    except Exception as e:
                        # If lambda analysis fails, that's okay - we'll skip it
                        pass
            
            self.operations.append(op)
            
            # Register results safely (already done above)
                
            return result
        return wrapped
    
    def _wrap_getitem(self, name: str, original_method: Callable) -> Callable:
        """Wrap DataFrame.__getitem__ to capture column access."""
        def wrapped(self_obj, key):
            if not self.enabled:
                return original_method(self_obj, key)
            
            # Execute the original getitem
            result = original_method(self_obj, key)
            
            # Only trace if the result is a Series (single column access)
            # Skip if it's a DataFrame (multi-column access or boolean indexing)
            if isinstance(result, pd.Series):
                # Record the operation
                self.op_counter += 1
                op = PandasOperation('df_getitem', original_method, (self_obj, key), {}, 
                                   result, self.op_counter, 'DataFrame')
                op.column_name = key  # Store the column name
                self.operations.append(op)
                
                # Register the resulting Series
                self.series_registry[id(result)] = result
                
            return result
        return wrapped
    
    def _wrap_str_method(self, name: str, original_method: Callable) -> Callable:
        """Wrap a StringMethods method to capture its execution."""
        def wrapped(self_obj, *args, **kwargs):
            if not self.enabled:
                return original_method(self_obj, *args, **kwargs)
            
            # The StringMethods object has ._orig which is the Series
            series = self_obj._orig if hasattr(self_obj, '_orig') else None
            
            # Execute the original method
            result = original_method(self_obj, *args, **kwargs)
            
            # Record the operation
            self.op_counter += 1
            op = PandasOperation(name, original_method, (series,) + args, kwargs, 
                               result, self.op_counter, 'Series')
            # Store the source series for code generation
            op.source_series = series
            self.operations.append(op)
            
            # Register result if it's a Series
            if isinstance(result, pd.Series):
                self.series_registry[id(result)] = result
                
            return result
        return wrapped
    
    def _wrap_function(self, name: str, original_func: Callable) -> Callable:
        """Wrap a pandas function to capture its execution."""
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original_func(*args, **kwargs)
            
            # Execute the original function
            result = original_func(*args, **kwargs)
            
            # Record the operation
            self.op_counter += 1
            op = PandasOperation(name, original_func, args, kwargs, result, self.op_counter)
            self.operations.append(op)
            
            # Register results
            if isinstance(result, pd.DataFrame):
                self.dataframe_registry[id(result)] = result
                
            return result
        return wrapped
    
    def _wrap_requests_get(self, name: str, original_func: Callable) -> Callable:
        """Wrap requests.get to track HTTP requests for http_get_json pattern."""
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original_func(*args, **kwargs)
            
            # Execute the original function
            response = original_func(*args, **kwargs)
            
            # Store the URL for this response
            if len(args) > 0 and isinstance(args[0], str):
                url = args[0]
                self.pending_http_requests[id(response)] = url
            
            return response
        return wrapped
    
    def _wrap_response_json(self, name: str, original_func: Callable) -> Callable:
        """Wrap response.json() to track JSON data from HTTP responses."""
        def wrapped(self_obj, *args, **kwargs):
            if not self.enabled:
                return original_func(self_obj, *args, **kwargs)
            
            # Execute the original function
            json_data = original_func(self_obj, *args, **kwargs)
            
            # Track which response produced this JSON data
            response_id = id(self_obj)
            json_id = id(json_data)
            self.json_responses[json_id] = response_id
            
            return json_data
        return wrapped
    
    def _wrap_dataframe_constructor(self, name: str, original_func: Callable) -> Callable:
        """Wrap DataFrame constructor to detect http_get_json pattern."""
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original_func(*args, **kwargs)
            
            # Check if this is the pattern: pd.DataFrame(response.json())
            # We detect this by checking if args[0] is a list/dict and if we can
            # trace it back to a requests.get() call
            url = None
            if len(args) > 0:
                data = args[0]
                # Check if data looks like JSON (list or dict)
                if isinstance(data, (list, dict)):
                    # Check if this JSON data came from a response.json() call
                    json_id = id(data)
                    if json_id in self.json_responses:
                        response_id = self.json_responses[json_id]
                        if response_id in self.pending_http_requests:
                            url = self.pending_http_requests[response_id]
                    # Also check if there's a URL in kwargs (for explicit passing)
                    elif 'url' in kwargs:
                        url = kwargs['url']
            
            # Execute the original function
            result = original_func(*args, **kwargs)
            
            # If we detected an HTTP request pattern, create http_get_json operation
            if url or (len(args) > 0 and isinstance(args[0], (list, dict)) and 
                      isinstance(args[0], list) and len(args[0]) > 0 and 
                      isinstance(args[0][0], dict)):
                self.op_counter += 1
                # Try to extract URL from kwargs or use placeholder
                extracted_url = url if url else (kwargs.get('url', '') if 'url' in kwargs else None)
                if not extracted_url:
                    # Try to find URL from pending requests by checking if this looks like JSON response
                    # For now, we'll use a placeholder and let codegen extract it from the source
                    extracted_url = None  # Will be extracted during codegen
                
                op = PandasOperation('http_get_json', original_func, args, kwargs, result, self.op_counter)
                op.url = extracted_url  # Store URL if we found it
                op.obj_type = 'DataFrame'
                self.operations.append(op)
                
                if PANDAS_AVAILABLE and pd is not None:
                    try:
                        if isinstance(result, pd.DataFrame):
                            self.dataframe_registry[id(result)] = result
                    except (TypeError, AttributeError):
                        pass
                
                return result
            
            # Normal DataFrame construction - record as regular operation
            # (We skip tracing normal DataFrame construction to avoid noise)
            # Only trace if it's a known pattern like read_csv result
            
            return result
        return wrapped


def trace_pandas_execution(code: str, globals_dict: Optional[Dict] = None) -> List[PandasOperation]:
    """
    Execute code with pandas tracing enabled.
    
    Args:
        code: Python code string to execute
        globals_dict: Optional globals dictionary for execution context
        
    Returns:
        List of captured pandas operations
    """
    tracer = PandasTracer()
    
    # Prepare execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        'pd': pd,
        'pandas': pd,
        'np': np,
        'numpy': np,
    }
    if globals_dict:
        exec_globals.update(globals_dict)
    
    # Start tracing and execute
    tracer.start_tracing()
    try:
        exec(code, exec_globals)
    finally:
        tracer.stop_tracing()
    
    return tracer.operations

