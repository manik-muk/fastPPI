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
from .http_tracer import HTTPTracer


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
        # Use HTTP tracer for tracking HTTP requests
        self.http_tracer = HTTPTracer()
        
    def start_tracing(self):
        """Enable tracing of pandas operations."""
        self.enabled = True
        self._patch_pandas()
        
    def stop_tracing(self):
        """Disable tracing and restore original functions."""
        self.enabled = False
        self._unpatch_pandas()
        # Stop HTTP tracing
        self.http_tracer.stop_tracing()
        
    def _patch_pandas(self):
        """Patch pandas functions and methods to capture calls."""
        # Import StringMethods and DatetimeProperties
        from pandas.core.strings.accessor import StringMethods
        try:
            from pandas.core.indexes.accessors import DatetimeProperties
        except ImportError:
            # Fallback for older pandas versions
            try:
                from pandas.core.accessor import DatetimeProperties
            except ImportError:
                DatetimeProperties = None
        
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
            'df_sort_values': pd.DataFrame.sort_values,
            'df_groupby': pd.DataFrame.groupby,
            
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
            
            # DatetimeProperties (for .dt operations)
            'dt_day': DatetimeProperties.day if DatetimeProperties else None,
            'dt_month': DatetimeProperties.month if DatetimeProperties else None,
            'dt_year': DatetimeProperties.year if DatetimeProperties else None,
            
            # Top-level functions
            'read_csv': pd.read_csv,
            'concat': pd.concat,
            'get_dummies': pd.get_dummies,
            'to_datetime': pd.to_datetime,
            'DataFrame': pd.DataFrame,  # For patching DataFrame constructor
            'Series': pd.Series,  # For patching Series constructor
        }
        
        # HTTP tracking is now handled by HTTPTracer
        
        # Patch DataFrame class methods FIRST (before wrapping the constructor)
        # This ensures all DataFrames created will have the patched methods
        pd.DataFrame.__getitem__ = self._wrap_getitem('df_getitem', self.original_functions['df_getitem'])
        pd.DataFrame.mean = self._wrap_method('df_mean', self.original_functions['df_mean'], 'DataFrame')
        pd.DataFrame.median = self._wrap_method('df_median', self.original_functions['df_median'], 'DataFrame')
        pd.DataFrame.fillna = self._wrap_method('df_fillna', self.original_functions['df_fillna'], 'DataFrame')
        pd.DataFrame.dropna = self._wrap_method('df_dropna', self.original_functions['df_dropna'], 'DataFrame')
        pd.DataFrame.apply = self._wrap_method('df_apply', self.original_functions['df_apply'], 'DataFrame')
        pd.DataFrame.sort_values = self._wrap_method('df_sort_values', self.original_functions['df_sort_values'], 'DataFrame')
        pd.DataFrame.groupby = self._wrap_method('df_groupby', self.original_functions['df_groupby'], 'DataFrame')
        pd.DataFrame.astype = self._wrap_method('df_astype', self.original_functions['df_astype'], 'DataFrame')
        
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
        
        # Patch DatetimeProperties directly (if available)
        if DatetimeProperties and self.original_functions['dt_day']:
            DatetimeProperties.day = self._wrap_dt_method('dt_day', self.original_functions['dt_day'])
            DatetimeProperties.month = self._wrap_dt_method('dt_month', self.original_functions['dt_month'])
            DatetimeProperties.year = self._wrap_dt_method('dt_year', self.original_functions['dt_year'])
        
        pd.read_csv = self._wrap_function('read_csv', self.original_functions['read_csv'])
        pd.concat = self._wrap_function('concat', self.original_functions['concat'])
        pd.get_dummies = self._wrap_function('get_dummies', self.original_functions['get_dummies'])
        pd.to_datetime = self._wrap_function('to_datetime', self.original_functions['to_datetime'])
        
        # Patch DataFrame constructor to detect http_get_json pattern
        pd.DataFrame = self._wrap_dataframe_constructor('DataFrame', self.original_functions['DataFrame'])
        
        # Patch Series constructor to trace Series creation
        pd.Series = self._wrap_series_constructor('Series', self.original_functions['Series'])
        
        # Start HTTP tracing (for detecting http_get_json pattern)
        self.http_tracer.start_tracing()
        
    def _unpatch_pandas(self):
        """Restore original pandas functions."""
        from pandas.core.strings.accessor import StringMethods
        try:
            from pandas.core.indexes.accessors import DatetimeProperties
        except ImportError:
            try:
                from pandas.core.accessor import DatetimeProperties
            except ImportError:
                DatetimeProperties = None
        
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
        
        # Restore DatetimeProperties (if available)
        if DatetimeProperties and self.original_functions.get('dt_day'):
            DatetimeProperties.day = self.original_functions['dt_day']
            DatetimeProperties.month = self.original_functions['dt_month']
            DatetimeProperties.year = self.original_functions['dt_year']
        
        pd.read_csv = self.original_functions['read_csv']
        pd.to_datetime = self.original_functions['to_datetime']
        pd.DataFrame = self.original_functions['DataFrame']
        
        # HTTP tracing cleanup is handled by HTTPTracer
    
    def _wrap_method(self, name: str, original_method: Callable, obj_type: str) -> Callable:
        """Wrap a pandas method to capture its execution."""
        def wrapped(self_obj, *args, **kwargs):
            if not self.enabled:
                return original_method(self_obj, *args, **kwargs)
            
            # For apply operations, temporarily disable NumPy tracing
            # to avoid capturing individual element operations (e.g., np.log for each element)
            # We only want to capture the apply operation itself, not the operations inside the lambda
            numpy_tracer_enabled_state = None
            if name in ('series_apply', 'df_apply'):
                # Temporarily disable NumPy tracing during apply execution
                # All NumPy functions share the same ExecutionTracer instance
                try:
                    import numpy as np
                    # Get the tracer instance from any numpy function (they all share the same one)
                    func = getattr(np, 'log', None)
                    if func is not None and hasattr(func, '_tracer'):
                        # func._tracer is the ExecutionTracer instance
                        tracer_instance = func._tracer
                        if hasattr(tracer_instance, 'enabled'):
                            numpy_tracer_enabled_state = tracer_instance.enabled
                            tracer_instance.enabled = False
                except (AttributeError, ImportError, TypeError):
                    # If we can't disable numpy tracing, continue anyway
                    numpy_tracer_enabled_state = None
            
            # Execute the original method
            result = original_method(self_obj, *args, **kwargs)
            
            # Re-enable NumPy tracing if we disabled it
            if numpy_tracer_enabled_state is not None:
                try:
                    import numpy as np
                    func = getattr(np, 'log', None)
                    if func is not None and hasattr(func, '_tracer'):
                        tracer_instance = func._tracer
                        if hasattr(tracer_instance, 'enabled'):
                            tracer_instance.enabled = numpy_tracer_enabled_state
                except (AttributeError, ImportError, TypeError):
                    pass
            
            # Record the operation
            self.op_counter += 1
            op = PandasOperation(name, original_method, (self_obj,) + args, kwargs, 
                               result, self.op_counter, obj_type)
            
            # Register results safely
            if PANDAS_AVAILABLE and pd is not None:
                try:
                    # Use original class for isinstance check
                    if isinstance(result, self.original_functions['DataFrame']):
                        self.dataframe_registry[id(result)] = result
                    elif hasattr(result, 'name') and hasattr(result, 'index') and not hasattr(result, 'columns'):
                        # Looks like a Series
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
        tracer_self = self  # Capture self reference for the closure
        original_dataframe_class = pd.DataFrame if PANDAS_AVAILABLE else None  # Store original DataFrame class
        
        def wrapped(self_obj, key):
            # Always execute the original getitem first
            result = original_method(self_obj, key)
            
            # Only trace if tracing is enabled
            if not tracer_self.enabled:
                return result
            
            # Only trace if the result is a Series (single column access)
            # Skip if it's a DataFrame (multi-column access or boolean indexing)
            if PANDAS_AVAILABLE and pd is not None:
                try:
                    # Use the original DataFrame class for isinstance check
                    # because pd.DataFrame might be wrapped
                    if isinstance(result, tracer_self.original_functions['DataFrame']):
                        # This is a DataFrame - don't trace
                        return result
                    elif hasattr(result, 'name') and hasattr(result, 'index') and not hasattr(result, 'columns'):
                        # This looks like a Series (has name and index, but not columns)
                        # Record the operation
                        tracer_self.op_counter += 1
                        op = PandasOperation('df_getitem', original_method, (self_obj, key), {}, 
                                           result, tracer_self.op_counter, 'DataFrame')
                        op.column_name = key  # Store the column name
                        tracer_self.operations.append(op)
                        
                        # Register the resulting Series
                        tracer_self.series_registry[id(result)] = result
                except (TypeError, AttributeError) as e:
                    # Silently ignore errors
                    pass
                
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
            if PANDAS_AVAILABLE and pd is not None:
                try:
                    # Check if it looks like a Series
                    if hasattr(result, 'name') and hasattr(result, 'index') and not hasattr(result, 'columns'):
                        self.series_registry[id(result)] = result
                except (TypeError, AttributeError):
                    pass
                
            return result
        return wrapped
    
    def _wrap_dt_method(self, name: str, original_method: Callable) -> Callable:
        """Wrap a DatetimeProperties method to capture its execution."""
        def wrapped(self_obj, *args, **kwargs):
            if not self.enabled:
                return original_method(self_obj, *args, **kwargs)
            
            # The DatetimeProperties object has ._orig which is the Series
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
            if PANDAS_AVAILABLE and pd is not None:
                try:
                    # Check if it looks like a Series
                    if hasattr(result, 'name') and hasattr(result, 'index') and not hasattr(result, 'columns'):
                        self.series_registry[id(result)] = result
                except (TypeError, AttributeError):
                    pass
                
            return result
        return wrapped
    
    def _wrap_function(self, name: str, original_func: Callable) -> Callable:
        """Wrap a pandas function to capture its execution."""
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original_func(*args, **kwargs)
            
            # Execute the original function
            result = original_func(*args, **kwargs)
            
            # Determine object type for the operation
            obj_type = "unknown"
            if PANDAS_AVAILABLE and pd is not None:
                try:
                    # Check DataFrame first using original class
                    if isinstance(result, self.original_functions['DataFrame']):
                        obj_type = "DataFrame"
                    elif hasattr(result, 'name') and hasattr(result, 'index') and not hasattr(result, 'columns'):
                        # Looks like a Series (has name and index, but not columns)
                        obj_type = "Series"
                    elif hasattr(result, 'columns') and hasattr(result, 'index'):
                        # Has columns - probably a DataFrame
                        obj_type = "DataFrame"
                except (TypeError, AttributeError):
                    pass
            
            # Record the operation
            self.op_counter += 1
            op = PandasOperation(name, original_func, args, kwargs, result, self.op_counter, obj_type)
            self.operations.append(op)
            
            # Register results safely
            if PANDAS_AVAILABLE and pd is not None:
                try:
                    # Use original class for isinstance check
                    if isinstance(result, self.original_functions['DataFrame']):
                        self.dataframe_registry[id(result)] = result
                    elif hasattr(result, 'name') and hasattr(result, 'index') and not hasattr(result, 'columns'):
                        # Looks like a Series
                        self.series_registry[id(result)] = result
                except (TypeError, AttributeError):
                    pass
            
            return result
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
                    # Use HTTP tracer to get URL for this JSON data
                    url = self.http_tracer.get_url_for_json(data)
                    # Also check if there's a URL in kwargs (for explicit passing)
                    if not url and 'url' in kwargs:
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
                        # Use original class for isinstance check
                        if isinstance(result, self.original_functions['DataFrame']):
                            self.dataframe_registry[id(result)] = result
                    except (TypeError, AttributeError):
                        pass
                
                return result
            
            # Normal DataFrame construction - trace it so we can generate C code
            # Check if it's a dict or list of dicts (common DataFrame creation patterns)
            should_trace = False
            if len(args) > 0:
                if isinstance(args[0], dict):
                    should_trace = True
                elif isinstance(args[0], list) and len(args[0]) > 0:
                    # List of dicts or list of lists
                    should_trace = True
            
            if should_trace:
                self.op_counter += 1
                op = PandasOperation('DataFrame', original_func, args, kwargs, result, self.op_counter, 'DataFrame')
                self.operations.append(op)
                
                if PANDAS_AVAILABLE and pd is not None:
                    try:
                        # Use original class for isinstance check
                        if isinstance(result, self.original_functions['DataFrame']):
                            self.dataframe_registry[id(result)] = result
                    except (TypeError, AttributeError):
                        pass
            
            return result
        return wrapped
    
    def _wrap_series_constructor(self, name: str, original_func: Callable) -> Callable:
        """Wrap Series constructor to trace Series creation."""
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original_func(*args, **kwargs)
            
            # Execute the original function
            result = original_func(*args, **kwargs)
            
            # Trace Series construction from lists/arrays
            should_trace = False
            if len(args) > 0:
                if isinstance(args[0], (list, np.ndarray)):
                    should_trace = True
            
            if should_trace:
                self.op_counter += 1
                op = PandasOperation('Series', original_func, args, kwargs, result, self.op_counter, 'Series')
                self.operations.append(op)
                
                if PANDAS_AVAILABLE and pd is not None:
                    try:
                        # Check if it looks like a Series
                        if hasattr(result, 'name') and hasattr(result, 'index') and not hasattr(result, 'columns'):
                            self.series_registry[id(result)] = result
                    except (TypeError, AttributeError):
                        pass
            
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

