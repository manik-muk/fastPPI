"""
String operations tracer for FastPPI.
Captures string operations including regex, unicode normalization, formatting, and sanitization.
"""

import re
import unicodedata
from typing import Dict, List, Any, Optional, Callable
from .tracer import Operation


class StringOperation(Operation):
    """Extended operation class for string operations."""
    
    def __init__(self, op_name: str, func: Callable, args: tuple, kwargs: dict, 
                 result: Any, op_id: int, operation_type: str = "unknown"):
        super().__init__(op_name, func, args, kwargs, result, op_id)
        self.operation_type = operation_type  # "regex", "unicode", "format", "sanitize", etc.
        self.input_string = None  # The input string
        self.pattern = None  # For regex operations
        self.replacement = None  # For regex sub operations
        
    def __repr__(self):
        return f"StringOperation(op_name='{self.op_name}', type='{self.operation_type}', id={self.op_id})"


class StringTracer:
    """Traces string operations during execution."""
    
    def __init__(self):
        self.operations: List[StringOperation] = []
        self.op_counter = 0
        self.enabled = False
        self.original_functions = {}
        self.string_registry: Dict[int, str] = {}  # id(obj) -> string
        
    def start_tracing(self):
        """Enable tracing of string operations."""
        self.enabled = True
        self._patch_string_operations()
        
    def stop_tracing(self):
        """Disable tracing and restore original functions."""
        self.enabled = False
        self._unpatch_string_operations()
        
    def _patch_string_operations(self):
        """Patch string-related functions to capture calls."""
        # Store original functions
        self.original_functions = {
            # Regex operations
            're_search': re.search,
            're_match': re.match,
            're_findall': re.findall,
            're_sub': re.sub,
            're_finditer': re.finditer,
            
            # Unicode normalization
            'unicodedata_normalize': unicodedata.normalize,
            
            # String methods (we'll patch str class methods)
            'str_format': str.format,
            'str_contains': None,  # Will be handled via 'in' operator
            'str_find': str.find,
            'str_replace': str.replace,
        }
        
        # Patch regex module
        re.search = self._wrap_re_function('re_search', re.search)
        re.match = self._wrap_re_function('re_match', re.match)
        re.findall = self._wrap_re_function('re_findall', re.findall)
        re.sub = self._wrap_re_function('re_sub', re.sub)
        re.finditer = self._wrap_re_function('re_finditer', re.finditer)
        
        # Patch unicodedata
        unicodedata.normalize = self._wrap_unicode_function('unicodedata_normalize', unicodedata.normalize)
        
        # Patch str.format (we'll handle f-strings separately via AST)
        # Note: str.format is a method, so we need to patch it differently
        # We'll handle this in the wrapper
        
    def _unpatch_string_operations(self):
        """Restore original string operation functions."""
        # Restore regex
        re.search = self.original_functions['re_search']
        re.match = self.original_functions['re_match']
        re.findall = self.original_functions['re_findall']
        re.sub = self.original_functions['re_sub']
        re.finditer = self.original_functions['re_finditer']
        
        # Restore unicodedata
        unicodedata.normalize = self.original_functions['unicodedata_normalize']
        
    def _wrap_re_function(self, name: str, original_func: Callable) -> Callable:
        """Wrap a regex function to capture its execution."""
        def wrapped(pattern, string, *args, **kwargs):
            if not self.enabled:
                return original_func(pattern, string, *args, **kwargs)
            
            # Execute the original function
            result = original_func(pattern, string, *args, **kwargs)
            
            # Record the operation
            self.op_counter += 1
            op_type = "regex_search" if name == "re_search" else \
                     "regex_match" if name == "re_match" else \
                     "regex_findall" if name == "re_findall" else \
                     "regex_sub" if name == "re_sub" else "regex"
            
            op = StringOperation(name, original_func, (pattern, string) + args, kwargs, 
                               result, self.op_counter, op_type)
            op.pattern = pattern if isinstance(pattern, str) else getattr(pattern, 'pattern', None)
            op.input_string = string if isinstance(string, str) else None
            if name == "re_sub" and len(args) > 0:
                op.replacement = args[0] if isinstance(args[0], str) else None
            
            self.operations.append(op)
            
            # Register string if it's a string result
            if isinstance(result, str):
                self.string_registry[id(result)] = result
            elif hasattr(result, 'group'):  # Match object
                if result.group(0):
                    self.string_registry[id(result.group(0))] = result.group(0)
            
            return result
        return wrapped
    
    def _wrap_unicode_function(self, name: str, original_func: Callable) -> Callable:
        """Wrap a unicode function to capture its execution."""
        def wrapped(form, unistr, *args, **kwargs):
            if not self.enabled:
                return original_func(form, unistr, *args, **kwargs)
            
            # Execute the original function
            result = original_func(form, unistr, *args, **kwargs)
            
            # Record the operation
            self.op_counter += 1
            op = StringOperation(name, original_func, (form, unistr) + args, kwargs, 
                               result, self.op_counter, "unicode_normalize")
            op.input_string = unistr if isinstance(unistr, str) else None
            
            self.operations.append(op)
            
            # Register result string
            if isinstance(result, str):
                self.string_registry[id(result)] = result
            
            return result
        return wrapped
    
    def trace_string_contains(self, string: str, substring: str) -> bool:
        """Trace string containment check (for prompt sanitization)."""
        if not self.enabled:
            return substring in string
        
        result = substring in string
        
        self.op_counter += 1
        op = StringOperation('str_contains', None, (string, substring), {}, 
                           result, self.op_counter, "sanitize")
        op.input_string = string
        op.pattern = substring  # Reuse pattern field for substring
        self.operations.append(op)
        
        return result
    
    def trace_string_format(self, template: str, *args, **kwargs) -> str:
        """Trace string formatting."""
        if not self.enabled:
            return template.format(*args, **kwargs)
        
        result = template.format(*args, **kwargs)
        
        self.op_counter += 1
        op = StringOperation('str_format', str.format, (template,) + args, kwargs, 
                           result, self.op_counter, "format")
        op.input_string = template
        self.operations.append(op)
        
        return result


def trace_string_execution(code: str, globals_dict: Optional[Dict] = None) -> List[StringOperation]:
    """
    Execute code with string operation tracing enabled.
    
    Args:
        code: Python code string to execute
        globals_dict: Optional globals dictionary for execution context
        
    Returns:
        List of captured string operations
    """
    tracer = StringTracer()
    
    # Prepare execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        're': re,
        'unicodedata': unicodedata,
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

