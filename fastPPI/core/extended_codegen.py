"""
Extended C code generator with support for pandas operations.
Requires C implementations of these operations to be linked.
"""

from typing import List, Dict, Set, Optional, Tuple
from .codegen import CCodeGenerator
from .graph import ComputationalGraph, GraphNode
from ..tracers.tracer import Operation
import numpy as np

# Try to import pandas and string operation types
try:
    from ..tracers.pandas_tracer import PandasOperation
    EXTENDED_OPS_AVAILABLE = True
except ImportError:
    EXTENDED_OPS_AVAILABLE = False
    PandasOperation = Operation

try:
    from ..tracers.string_tracer import StringOperation
    STRING_OPS_AVAILABLE = True
except ImportError:
    STRING_OPS_AVAILABLE = False
    StringOperation = Operation


class CFunctionRegistry:
    """
    Registry mapping Python operations to C function calls.
    This is where you register your C implementations.
    """
    
    # Map: (operation_name, obj_type) -> (c_function_name, includes, dependencies)
    FUNCTION_MAP: Dict[Tuple[str, str], Dict[str, str]] = {
        # Pandas operations
        ('mean', 'DataFrame'): {
            'c_function': 'pandas_df_mean',
            'include': '"pandas_c.h"',
            'return_type': 'double*',
            'description': 'Compute mean of DataFrame columns'
        },
        ('mean', 'Series'): {
            'c_function': 'pandas_series_mean',
            'include': '"pandas_c.h"',
            'return_type': 'double',
            'description': 'Compute mean of Series'
        },
        ('fillna', 'DataFrame'): {
            'c_function': 'pandas_df_fillna',
            'include': '"pandas_c.h"',
            'return_type': 'DataFrame*',
            'description': 'Fill missing values in DataFrame'
        },
        ('fillna', 'Series'): {
            'c_function': 'pandas_series_fillna',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Fill missing values in Series'
        },
        ('apply', 'Series'): {
            'c_function': 'pandas_series_apply',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Apply function to Series elements'
        },
        ('apply_lambda', 'Series'): {
            'c_function': 'pandas_series_apply_lambda',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Apply lambda function to Series elements (inline)'
        },
        ('astype', 'Series'): {
            'c_function': 'pandas_series_astype',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Convert Series to different dtype'
        },
        ('str.lower', 'Series'): {
            'c_function': 'pandas_series_str_lower',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Convert Series strings to lowercase'
        },
        ('str_lower', 'Series'): {  # Alternative name without dot
            'c_function': 'pandas_series_str_lower',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Convert Series strings to lowercase'
        },
        ('str_upper', 'Series'): {
            'c_function': 'pandas_series_str_upper',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Convert Series strings to uppercase'
        },
        ('str_strip', 'Series'): {
            'c_function': 'pandas_series_str_strip',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Strip whitespace from Series strings'
        },
        ('isna', 'Series'): {
            'c_function': 'pandas_series_isna',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Check for null values in Series'
        },
        ('isnull', 'Series'): {  # Alias for isna
            'c_function': 'pandas_series_isna',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Check for null values in Series'
        },
        ('read_csv', None): {
            'c_function': 'pandas_read_csv',
            'include': '"pandas_c.h"',
            'return_type': 'DataFrame*',
            'description': 'Read CSV file into DataFrame'
        },
        ('http_get_json', None): {
            'c_function': 'pandas_http_get_json',
            'include': '"pandas_c.h"',
            'return_type': 'DataFrame*',
            'description': 'Make HTTP GET request and parse JSON array into DataFrame'
        },
        ('getitem', 'DataFrame'): {
            'c_function': 'pandas_df_getitem',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Get column from DataFrame'
        },
        ('df_getitem', 'DataFrame'): {  # Alternative name
            'c_function': 'pandas_df_getitem',
            'include': '"pandas_c.h"',
            'return_type': 'Series*',
            'description': 'Get column from DataFrame'
        },
        ('concat', None): {
            'c_function': 'pandas_concat',
            'include': '"pandas_c.h"',
            'return_type': 'DataFrame*',
            'description': 'Concatenate DataFrames'
        },
        ('sort_values', 'DataFrame'): {
            'c_function': 'pandas_df_sort_values',
            'include': '"pandas_c.h"',
            'return_type': 'DataFrame*',
            'description': 'Sort DataFrame by column values'
        },
        ('groupby', 'DataFrame'): {
            'c_function': 'pandas_df_groupby',
            'include': '"pandas_c.h"',
            'return_type': 'DataFrame*',
            'description': 'Group DataFrame by column (simplified)'
        },
        
        # String operations
        ('re_search', None): {
            'c_function': 'regex_search',
            'include': '"string_c.h"',
            'return_type': 'bool',
            'description': 'Search for regex pattern in string'
        },
        ('re_match', None): {
            'c_function': 'regex_match',
            'include': '"string_c.h"',
            'return_type': 'bool',
            'description': 'Match regex pattern at start of string'
        },
        ('re_findall', None): {
            'c_function': 'regex_findall',
            'include': '"string_c.h"',
            'return_type': 'int',  # Returns int, outputs matches via pointer
            'description': 'Find all regex matches in string'
        },
        ('re_sub', None): {
            'c_function': 'regex_sub',
            'include': '"string_c.h"',
            'return_type': 'char*',
            'description': 'Substitute regex matches with replacement'
        },
        ('unicodedata_normalize', None): {
            'c_function': 'unicode_normalize',
            'include': '"string_c.h"',
            'return_type': 'char*',
            'description': 'Normalize Unicode string'
        },
        ('str_format', None): {
            'c_function': 'string_format',
            'include': '"string_c.h"',
            'return_type': 'char*',
            'description': 'Format string with placeholders'
        },
        ('str_contains', None): {
            'c_function': 'string_contains',
            'include': '"string_c.h"',
            'return_type': 'bool',
            'description': 'Check if string contains substring (prompt sanitization)'
        },
        ('string_contains_any_word', None): {
            'c_function': 'string_contains_any_word',
            'include': '"string_c.h"',
            'return_type': 'bool',
            'description': 'Check if string contains any word from list'
        },
        ('string_contains_all_words', None): {
            'c_function': 'string_contains_all_words',
            'include': '"string_c.h"',
            'return_type': 'bool',
            'description': 'Check if string contains all words from list'
        },
    }
    
    @classmethod
    def get_c_function(cls, op_name: str, obj_type: Optional[str] = None) -> Optional[Dict[str, str]]:
        """Get C function mapping for an operation."""
        # Strip common prefixes from pandas operations
        clean_op_name = op_name
        if op_name.startswith('series_'):
            clean_op_name = op_name[7:]  # Remove 'series_' prefix
        elif op_name.startswith('df_'):
            clean_op_name = op_name[3:]  # Remove 'df_' prefix
        
        key = (clean_op_name, obj_type)
        if key in cls.FUNCTION_MAP:
            return cls.FUNCTION_MAP[key]
        # Try without obj_type
        key = (clean_op_name, None)
        return cls.FUNCTION_MAP.get(key)
    
    @classmethod
    def register_function(cls, op_name: str, obj_type: Optional[str], 
                         c_function: str, include: str, return_type: str, description: str = ""):
        """Register a new C function mapping."""
        cls.FUNCTION_MAP[(op_name, obj_type)] = {
            'c_function': c_function,
            'include': include,
            'return_type': return_type,
            'description': description
        }


class ExtendedCCodeGenerator(CCodeGenerator):
    """
    Extended code generator that supports pandas operations.
    
    To use:
    1. Implement C functions for pandas operations
    2. Register them using CFunctionRegistry.register_function()
    3. Compile with your C library linked (e.g., -lpandas_c)
    """
    
    def __init__(self, graph: ComputationalGraph, input_vars: List[str], 
                 input_arrays: Dict[str, np.ndarray] = None,
                 link_libraries: List[str] = None):
        super().__init__(graph, input_vars, input_arrays)
        self.link_libraries = link_libraries or []
        self.dataframe_vars: Dict[int, str] = {}  # op_id -> C variable name for DataFrame
        self.series_vars: Dict[int, str] = {}     # op_id -> C variable name for Series
        
        # Track which includes we need
        self.pandas_include = False
        self.string_include = False
        self.string_inputs: Set[str] = set()  # Track which input vars are strings
        self.string_conversions: Dict[str, str] = {}  # Track converted string variables to avoid duplicates
        self.batch_string_inputs: Set[str] = set()  # Track which inputs are lists of strings
    
    def _get_c_type_for_result(self, op: Operation) -> str:
        """Get C type for operation result, handling pandas types."""
        if EXTENDED_OPS_AVAILABLE and isinstance(op, PandasOperation):
            if op.obj_type == "DataFrame":
                return "DataFrame*"
            elif op.obj_type == "Series":
                return "Series*"
        
        # Fall back to parent implementation
        if isinstance(op.result, np.ndarray):
            return "double*"
        elif isinstance(op.result, (int, float)):
            return "double"
        else:
            return "void*"
    
    def _generate_lambda_apply_code(self, node: GraphNode,
                                   allocated_vars: Set[str],
                                   declared_scalars: Set[str]) -> List[str]:
        """Generate inline C code for Series.apply() with a lambda function."""
        op = node.operation
        lambda_info = op.lambda_info
        lines = []
        result_var = self._get_var_name(op.op_id)
        
        # Find the input Series
        input_series_var = None
        if len(op.args) > 0:
            series_obj = op.args[0]
            for prev_node in reversed(list(self.graph.nodes)):
                if prev_node.operation.op_id >= op.op_id:
                    continue
                if isinstance(prev_node.operation, PandasOperation):
                    prev_op = prev_node.operation
                    if id(prev_op.result) == id(series_obj):
                        input_series_var = self._get_var_name(prev_op.op_id)
                        break
        
        if not input_series_var:
            lines.append(f"    // ERROR: Could not find input Series for lambda apply")
            return lines
        
        self.pandas_include = True
        param_name = lambda_info['param_name']
        c_expr = lambda_info['c_code']
        requires_null_check = lambda_info.get('requires_null_check', False)
        external_vars = lambda_info.get('external_vars', {})
        
        # Extract external variables and declare them as constants in C
        # These are variables captured from the closure (e.g., age_mean)
        external_var_declarations = []
        for var_name, var_value in external_vars.items():
            if isinstance(var_value, (int, float)):
                external_var_declarations.append(f"    const double {var_name} = {var_value};")
            elif isinstance(var_value, bool):
                external_var_declarations.append(f"    const int {var_name} = {1 if var_value else 0};")
            elif isinstance(var_value, str):
                escaped_val = var_value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                external_var_declarations.append(f'    const char* {var_name} = "{escaped_val}";')
            else:
                # For other types, try to convert to numeric
                try:
                    numeric_val = float(var_value)
                    external_var_declarations.append(f"    const double {var_name} = {numeric_val};")
                except:
                    # Skip if we can't convert
                    pass
        
        # Search for variables in C expression that aren't in external_vars
        # These might be computed earlier in the graph (e.g., age_mean = df["age"].mean())
        import re
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        potential_vars = set(re.findall(var_pattern, c_expr))
        # Exclude known C keywords and functions, and the lambda parameter
        c_keywords = {'if', 'else', 'isnan', 'is_null', 'true', 'false', 'NULL', 'NAN', 'INFINITY', 'abs', 'fabs'}
        potential_vars = {v for v in potential_vars if v not in c_keywords and v != param_name}
        
        # Search graph for scalar operations (like mean) that might match variable names
        for var_name in potential_vars:
            if var_name not in external_vars:
                # Look for scalar operations (mean, sum, etc.) before this operation
                for prev_node in self.graph.nodes:
                    if prev_node.operation.op_id >= op.op_id:
                        continue
                    prev_op = prev_node.operation
                    # Check if this is a scalar result (mean operation typically returns scalar)
                    if isinstance(prev_op.result, (int, float, np.number)):
                        # This is a scalar - check if operation name matches variable name pattern
                        # For age_mean, look for operations with 'mean' in name
                        op_name = getattr(prev_op, 'op_name', '')
                        if 'mean' in op_name.lower() or 'sum' in op_name.lower():
                            # Use the scalar value
                            var_value = float(prev_op.result)
                            external_var_declarations.append(f"    const double {var_name} = {var_value};")
                            external_vars[var_name] = var_value  # Add for reference
                            break
                    # Also check for pandas Series.mean() results
                    try:
                        import pandas as pd
                        if isinstance(prev_op.result, (pd.Series, pd.DataFrame)):
                            # Check if this is a mean operation result
                            op_name = getattr(prev_op, 'op_name', '')
                            if 'mean' in op_name.lower():
                                # Extract scalar value from Series/DataFrame
                                try:
                                    if isinstance(prev_op.result, pd.Series):
                                        var_value = float(prev_op.result)
                                    elif isinstance(prev_op.result, pd.DataFrame):
                                        # Take first scalar value
                                        var_value = float(prev_op.result.values.flat[0])
                                    else:
                                        continue
                                    external_var_declarations.append(f"    const double {var_name} = {var_value};")
                                    external_vars[var_name] = var_value
                                    break
                                except:
                                    pass
                    except ImportError:
                        pass
        
        # Generate inline loop to apply lambda to each element
        lines.append(f"    // Apply lambda: {param_name} -> {c_expr}")
        
        # Declare external variables if any
        if external_var_declarations:
            lines.append("    // External variables from closure:")
            lines.extend(external_var_declarations)
        
        lines.append(f"    Series* {result_var} = series_create({input_series_var}->length, {input_series_var}->dtype);")
        allocated_vars.add(result_var)
        self.series_vars[op.op_id] = result_var
        
        # Copy name from input series
        lines.append(f"    if ({input_series_var}->name) {{")
        lines.append(f"        {result_var}->name = strdup({input_series_var}->name);")
        lines.append(f"    }}")
        
        # Apply lambda to each element
        lines.append(f"    for (int64_t i = 0; i < {input_series_var}->length; i++) {{")
        
        # Check if the C expression itself handles null checking (e.g., uses isnan())
        handles_nulls_explicitly = 'isnan' in c_expr
        
        if requires_null_check and not handles_nulls_explicitly:
            # Lambda uses the parameter but doesn't explicitly check for nulls
            # Add null checking wrapper
            lines.append(f"        if (is_null({input_series_var}, i)) {{")
            lines.append(f"            set_null({result_var}, i, true);")
            lines.append(f"        }} else {{")
            lines.append(f"            double {param_name} = {input_series_var}->data[i];")
            lines.append(f"            {result_var}->data[i] = {c_expr};")
            lines.append(f"        }}")
        else:
            # Either no null check needed or lambda handles nulls explicitly (pd.notnull/pd.isna)
            # Just apply the expression directly
            lines.append(f"        double {param_name} = {input_series_var}->data[i];")
            lines.append(f"        {result_var}->data[i] = {c_expr};")
        
        lines.append(f"    }}")
        lines.append("")
        
        return lines
    
    def _generate_pandas_operation_code(self, node: GraphNode, 
                                       allocated_vars: Set[str], 
                                       declared_scalars: Set[str]) -> List[str]:
        """Generate C code for a pandas operation."""
        if not EXTENDED_OPS_AVAILABLE:
            return []
        
        op = node.operation
        if not isinstance(op, PandasOperation):
            return []
        
        lines = []
        result_var = self._get_var_name(op.op_id)
        
        # Special handling for apply() with lambda functions
        if op.op_name in ('series_apply', 'df_apply') and hasattr(op, 'lambda_info') and op.lambda_info:
            return self._generate_lambda_apply_code(node, allocated_vars, declared_scalars)
        
        # Special handling for DataFrame constructor from dict
        if op.op_name == 'DataFrame' and op.obj_type == 'DataFrame' and len(op.args) > 0:
            # Create DataFrame from dict or list
            if isinstance(op.args[0], dict):
                # Create DataFrame from dictionary
                # Extract column names and data
                df_dict = op.args[0]
                num_rows = 0
                columns = []
                column_data = {}
                
                # Find max length to determine num_rows
                for col_name, col_data in df_dict.items():
                    columns.append(col_name)
                    if isinstance(col_data, (list, np.ndarray)):
                        col_array = np.array(col_data)
                        num_rows = max(num_rows, len(col_array))
                        column_data[col_name] = col_array
                    else:
                        # Scalar - convert to array
                        column_data[col_name] = np.array([col_data])
                        num_rows = max(num_rows, 1)
                
                # Create DataFrame structure
                lines.append(f"    // Create DataFrame from dict with {len(columns)} columns, {num_rows} rows")
                lines.append(f"    DataFrame* {result_var} = dataframe_create({num_rows}, {len(columns)});")
                
                # Add columns
                for i, col_name in enumerate(columns):
                    col_data = column_data[col_name]
                    lines.append(f"    {result_var}->column_names[{i}] = strdup(\"{col_name}\");")
                    
                    # Create Series for this column
                    series_var = f"{result_var}_col_{i}"
                    lines.append(f"    Series* {series_var} = series_create({num_rows}, 'f');")
                    lines.append(f"    {series_var}->name = strdup(\"{col_name}\");")
                    
                    # Fill data
                    if len(col_data) > 0:
                        for j, val in enumerate(col_data[:num_rows]):
                            if np.isnan(val):
                                lines.append(f"    {series_var}->data[{j}] = NAN;")
                            elif np.isinf(val):
                                lines.append(f"    {series_var}->data[{j}] = {'INFINITY' if val > 0 else '-INFINITY'};")
                            else:
                                lines.append(f"    {series_var}->data[{j}] = {val};")
                    
                    lines.append(f"    {result_var}->columns[{i}] = {series_var};")
                
                self.dataframe_vars[op.op_id] = result_var
                allocated_vars.add(result_var)
                return lines
        
        # Get C function mapping
        func_info = CFunctionRegistry.get_c_function(op.op_name, op.obj_type)
        if not func_info:
            # No C implementation available - generate error comment
            lines.append(f"    // ERROR: No C implementation for {op.op_name} on {op.obj_type}")
            lines.append(f"    // TODO: Implement {op.op_name} in C or register with CFunctionRegistry")
            return lines
        
        self.pandas_include = True
        c_func = func_info['c_function']
        return_type = func_info['return_type']
        
        # Special handling for fillna - detect if fill value is string
        if op.op_name in ('fillna', 'series_fillna') and len(op.args) > 1:
            fill_value = op.args[1] if len(op.args) > 1 else op.kwargs.get('value', None)
            if isinstance(fill_value, str):
                # Use string variant of fillna (only takes 2 args: series, fill_value)
                c_func = 'pandas_series_fillna_str'
            else:
                # Numeric fillna requires 3 args: series, fill_value, strategy
                # The function signature is: pandas_series_fillna(series, fill_value, strategy)
                # We default to "constant" strategy if not specified
                c_func = 'pandas_series_fillna'
                # Add strategy argument if not already present
                strategy = op.kwargs.get('method', 'constant')  # Default to constant
                if strategy not in ('constant', 'mean', 'median', 'forward', 'backward'):
                    strategy = 'constant'  # Default to constant for unknown strategies
                # Note: We'll add the strategy to input_vars after building the list
        
        # Special handling for str operations - they store the source series
        if hasattr(op, 'source_series') and op.source_series is not None:
            # For str operations, find the Series that was used
            source_var = None
            if EXTENDED_OPS_AVAILABLE:
                for prev_node in reversed(list(self.graph.nodes)):
                    if prev_node.operation.op_id >= op.op_id:
                        continue
                    if isinstance(prev_node.operation, PandasOperation):
                        prev_op = prev_node.operation
                        if id(prev_op.result) == id(op.source_series):
                            source_var = self._get_var_name(prev_op.op_id)
                            break
            
            if source_var:
                input_vars = [source_var]
            else:
                input_vars = []
        # Special handling for read_csv - csv_path is a string input variable
        elif op.op_name == 'read_csv':
            # For read_csv, the first argument is csv_path which is a string input
            input_vars = []
            if len(op.args) > 0:
                csv_path_arg = op.args[0]
                # Check if this is an input variable by matching string value to input_arrays
                # During tracing, op.args[0] contains the actual string value, not the variable name
                input_var_name = None
                if isinstance(csv_path_arg, str) and self.input_arrays:
                    # Find which input variable has this string value
                    for var_name, var_value in self.input_arrays.items():
                        if isinstance(var_value, str) and var_value == csv_path_arg:
                            input_var_name = var_name
                            break
                
                if input_var_name:
                    # It's an input variable - need to convert from double* to char*
                    self.string_inputs.add(input_var_name)
                    # Generate optimized code to convert double* array to char* string
                    # The double* array contains character codes (from ord() encoding)
                    csv_str_var = f"{input_var_name}_str"
                    lines.append(f"    // Convert string input {input_var_name} from double* to char*")
                    lines.append(f"    char {csv_str_var}[256];  // Max path length")
                    lines.append(f"    int {input_var_name}_len = 0;")
                    lines.append(f"    double val;")
                    lines.append(f"    // Optimized conversion - unrolled loop for better performance")
                    lines.append(f"    while ({input_var_name}_len < 255 && (val = {input_var_name}[{input_var_name}_len]) > 0 && val < 256) {{")
                    lines.append(f"        {csv_str_var}[{input_var_name}_len++] = (char)(int)val;")
                    lines.append(f"    }}")
                    lines.append(f"    {csv_str_var}[{input_var_name}_len] = '\\0';")
                    input_vars.append(csv_str_var)
                elif isinstance(csv_path_arg, str):
                    # String literal - use directly
                    escaped_arg = csv_path_arg.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    input_vars.append(f'"{escaped_arg}"')
                else:
                    input_vars.append("NULL")
        # Special handling for http_get_json - url is a string input variable
        elif op.op_name == 'http_get_json':
            # For http_get_json, the URL comes from the operation's url attribute or args
            input_vars = []
            url_arg = None
            
            # Try to get URL from operation attribute (set by tracer)
            if hasattr(op, 'url') and op.url:
                url_arg = op.url
            # Otherwise try to extract from args (if URL was passed explicitly)
            elif len(op.args) > 0:
                # Check if first arg is a string (URL)
                if isinstance(op.args[0], str):
                    url_arg = op.args[0]
                # Or check if it's a dict/list that came from response.json()
                # In that case, we need to extract URL from the trace
                # For now, we'll look for URL in the operation context
            
            if url_arg:
                # Check if this is an input variable
                input_var_name = None
                if isinstance(url_arg, str) and self.input_arrays:
                    # Find which input variable has this string value
                    for var_name, var_value in self.input_arrays.items():
                        if isinstance(var_value, str) and var_value == url_arg:
                            input_var_name = var_name
                            break
                
                if input_var_name:
                    # It's an input variable - need to convert from double* to char*
                    self.string_inputs.add(input_var_name)
                    url_str_var = f"{input_var_name}_str"
                    lines.append(f"    // Convert string input {input_var_name} from double* to char*")
                    lines.append(f"    char {url_str_var}[512];  // Max URL length")
                    lines.append(f"    int {input_var_name}_len = 0;")
                    lines.append(f"    double val;")
                    lines.append(f"    while ({input_var_name}_len < 511 && (val = {input_var_name}[{input_var_name}_len]) > 0 && val < 256) {{")
                    lines.append(f"        {url_str_var}[{input_var_name}_len++] = (char)(int)val;")
                    lines.append(f"    }}")
                    lines.append(f"    {url_str_var}[{input_var_name}_len] = '\\0';")
                    input_vars.append(url_str_var)
                elif isinstance(url_arg, str):
                    # String literal - use directly
                    escaped_arg = url_arg.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                    input_vars.append(f'"{escaped_arg}"')
                else:
                    input_vars.append("NULL")
            else:
                # URL not found - this shouldn't happen if tracing worked correctly
                input_vars.append('"http://localhost:3000/users"')  # Default fallback
        # Special handling for df_getitem - need to find the source DataFrame
        elif op.op_name == 'df_getitem':
            # For column access, args[0] is the DataFrame, args[1] is the column name
            input_vars = []
            if len(op.args) >= 2:
                df_obj = op.args[0]  # The DataFrame being indexed
                column_name = op.args[1]  # The column name
                
                # Find the most recent DataFrame operation before this one
                df_var = None
                if EXTENDED_OPS_AVAILABLE:
                    for prev_node in reversed(list(self.graph.nodes)):
                        if prev_node.operation.op_id >= op.op_id:
                            continue
                        if isinstance(prev_node.operation, PandasOperation):
                            prev_op = prev_node.operation
                            # Check if this is the DataFrame we're looking for
                            if id(prev_op.result) == id(df_obj):
                                df_var = self._get_var_name(prev_op.op_id)
                                break
                            # Also check if it's a DataFrame by type
                            try:
                                import pandas as pd
                                if isinstance(prev_op.result, pd.DataFrame):
                                    # Use the most recent DataFrame operation
                                    df_var = self._get_var_name(prev_op.op_id)
                                    break
                            except:
                                pass
                
                if df_var:
                    input_vars.append(df_var)
                else:
                    input_vars.append("NULL  /* Could not find source DataFrame */")
                
                # Add column name
                if isinstance(column_name, str):
                    input_vars.append(f'"{column_name}"')
                else:
                    input_vars.append(str(column_name))
        # Special handling for concat - takes list of DataFrames
        elif op.op_name == 'concat':
            input_vars = []
            # Extract DataFrames from args[0] (which is a list)
            if len(op.args) > 0 and isinstance(op.args[0], list):
                df_list = op.args[0]
                # Find DataFrame variables for each DataFrame in the list
                df_vars = []
                for df_obj in df_list:
                    df_var = None
                    if EXTENDED_OPS_AVAILABLE:
                        # First, try to find by object ID
                        for prev_node in reversed(list(self.graph.nodes)):
                            if prev_node.operation.op_id >= op.op_id:
                                continue
                            if isinstance(prev_node.operation, PandasOperation):
                                prev_op = prev_node.operation
                                try:
                                    if id(prev_op.result) == id(df_obj):
                                        # Check if this variable is in dataframe_vars (was created)
                                        if prev_op.op_id in self.dataframe_vars:
                                            df_var = self.dataframe_vars[prev_op.op_id]
                                        else:
                                            df_var = self._get_var_name(prev_op.op_id)
                                        break
                                except:
                                    pass
                        
                        # If not found by ID, try to find by position (for inline DataFrame creation)
                        # Look for DataFrame operations before this concat
                        if not df_var:
                            dataframe_ops = []
                            for prev_node in self.graph.nodes:
                                if prev_node.operation.op_id >= op.op_id:
                                    break
                                if isinstance(prev_node.operation, PandasOperation):
                                    prev_op = prev_node.operation
                                    if prev_op.op_name == 'DataFrame' and prev_op.obj_type == 'DataFrame':
                                        dataframe_ops.append(prev_op)
                            
                            # Use DataFrame operations in order they appear
                            df_idx = len(df_vars)  # Which DataFrame in the list we're looking for
                            if df_idx < len(dataframe_ops):
                                prev_op = dataframe_ops[df_idx]
                                if prev_op.op_id in self.dataframe_vars:
                                    df_var = self.dataframe_vars[prev_op.op_id]
                                else:
                                    df_var = self._get_var_name(prev_op.op_id)
                    
                    if df_var:
                        df_vars.append(df_var)
                    else:
                        df_vars.append("NULL")
                
                # Generate array of DataFrame pointers
                if df_vars:
                    df_array_var = f"{result_var}_dfs"
                    lines.append(f"    DataFrame* {df_array_var}[{len(df_vars)}];")
                    for i, df_var in enumerate(df_vars):
                        lines.append(f"    {df_array_var}[{i}] = {df_var};")
                    input_vars.append(df_array_var)
                    input_vars.append(str(len(df_vars)))
                    
                    # Get axis from kwargs (default 0)
                    axis = op.kwargs.get('axis', 0)
                    input_vars.append(str(axis))
                else:
                    input_vars.append("NULL")
                    input_vars.append("0")
                    input_vars.append("0")
            else:
                input_vars.append("NULL")
                input_vars.append("0")
                input_vars.append("0")
        # Special handling for sort_values - takes DataFrame, column name, and ascending
        elif op.op_name == 'sort_values' or op.op_name == 'df_sort_values':
            input_vars = []
            # Find the source DataFrame
            df_obj = op.args[0] if len(op.args) > 0 else None
            df_var = None
            if df_obj is not None and EXTENDED_OPS_AVAILABLE:
                for prev_node in reversed(list(self.graph.nodes)):
                    if prev_node.operation.op_id >= op.op_id:
                        continue
                    if isinstance(prev_node.operation, PandasOperation):
                        prev_op = prev_node.operation
                        if id(prev_op.result) == id(df_obj):
                            df_var = self._get_var_name(prev_op.op_id)
                            break
                        try:
                            import pandas as pd
                            if isinstance(prev_op.result, pd.DataFrame) and id(prev_op.result) == id(df_obj):
                                df_var = self._get_var_name(prev_op.op_id)
                                break
                        except:
                            pass
            
            if df_var:
                input_vars.append(df_var)
            else:
                input_vars.append("NULL")
            
            # Get column name from args or kwargs
            column_name = None
            if len(op.args) > 1:
                column_name = op.args[1]
            elif 'by' in op.kwargs:
                column_name = op.kwargs['by']
            
            if isinstance(column_name, str):
                input_vars.append(f'"{column_name}"')
            elif isinstance(column_name, list) and len(column_name) > 0:
                # For multiple columns, use first one (simplified)
                input_vars.append(f'"{column_name[0]}"')
            else:
                input_vars.append('""')
            
            # Get ascending flag (default True)
            ascending = op.kwargs.get('ascending', True)
            if isinstance(ascending, bool):
                input_vars.append("1" if ascending else "0")
            elif isinstance(ascending, list):
                input_vars.append("1" if ascending[0] else "0")
            else:
                input_vars.append("1")
        # Special handling for groupby - takes DataFrame and column name
        elif op.op_name == 'groupby' or op.op_name == 'df_groupby':
            input_vars = []
            # Find the source DataFrame
            df_obj = op.args[0] if len(op.args) > 0 else None
            df_var = None
            if df_obj is not None and EXTENDED_OPS_AVAILABLE:
                for prev_node in reversed(list(self.graph.nodes)):
                    if prev_node.operation.op_id >= op.op_id:
                        continue
                    if isinstance(prev_node.operation, PandasOperation):
                        prev_op = prev_node.operation
                        if id(prev_op.result) == id(df_obj):
                            df_var = self._get_var_name(prev_op.op_id)
                            break
                        try:
                            import pandas as pd
                            if isinstance(prev_op.result, pd.DataFrame) and id(prev_op.result) == id(df_obj):
                                df_var = self._get_var_name(prev_op.op_id)
                                break
                        except:
                            pass
            
            if df_var:
                input_vars.append(df_var)
            else:
                input_vars.append("NULL")
            
            # Get column name from args or kwargs
            column_name = None
            if len(op.args) > 1:
                column_name = op.args[1]
            elif 'by' in op.kwargs:
                column_name = op.kwargs['by']
            
            if isinstance(column_name, str):
                input_vars.append(f'"{column_name}"')
            elif isinstance(column_name, list) and len(column_name) > 0:
                # For multiple columns, use first one (simplified)
                input_vars.append(f'"{column_name[0]}"')
            else:
                input_vars.append('""')
        else:
            # Normal argument processing for other operations
            input_vars = []
            for arg in op.args:
                if isinstance(arg, np.ndarray):
                    if id(arg) in self.array_to_var_map:
                        input_vars.append(self.array_to_var_map[id(arg)])
                    else:
                        # Find which node created this array
                        found = False
                        for prev_node in self.graph.nodes:
                            if (isinstance(prev_node.operation.result, np.ndarray) and
                                id(prev_node.operation.result) == id(arg)):
                                input_vars.append(self._get_var_name(prev_node.operation.op_id))
                                found = True
                                break
                        if not found:
                            input_vars.append("NULL")
                else:
                    # First check if it's a scalar (int, float, bool) - these should be handled directly
                    # This is important for operations like fillna where fill_value might be a scalar
                    if isinstance(arg, (int, float, bool, type(None))):
                        # For fillna operations, check if this is the fill_value argument
                        # and if so, try to find the scalar result from a previous operation
                        if op.op_name in ('fillna', 'series_fillna') and len(op.args) > 1 and op.args[1] == arg:
                            # This is the fill_value argument - try to find it as a scalar result
                            found_scalar = False
                            if EXTENDED_OPS_AVAILABLE:
                                for prev_node in self.graph.nodes:
                                    if prev_node.operation.op_id >= op.op_id:
                                        continue
                                    prev_op = prev_node.operation
                                    # Check if this operation produced a scalar that matches
                                    if isinstance(prev_op.result, (int, float, bool)):
                                        try:
                                            # Check if the values match (with tolerance for floats)
                                            if isinstance(arg, float) and isinstance(prev_op.result, float):
                                                if abs(arg - prev_op.result) < 1e-10:
                                                    prev_var = self._get_var_name(prev_op.op_id)
                                                    input_vars.append(prev_var)
                                                    found_scalar = True
                                                    break
                                            elif arg == prev_op.result:
                                                prev_var = self._get_var_name(prev_op.op_id)
                                                input_vars.append(prev_var)
                                                found_scalar = True
                                                break
                                        except:
                                            pass
                            
                            if not found_scalar:
                                # Use the literal value
                                if arg is None:
                                    input_vars.append("NAN")
                                else:
                                    input_vars.append(str(arg))
                        else:
                            # Regular scalar - use directly
                            if arg is None:
                                input_vars.append("NAN")
                            else:
                                input_vars.append(str(arg))
                    else:
                        # Check if this is a DataFrame or Series from a previous pandas operation
                        found_pandas_input = False
                        if EXTENDED_OPS_AVAILABLE:
                            # Look for previous pandas operation that created this object
                            # Try multiple strategies to find the source
                            for prev_node in self.graph.nodes:
                                if prev_node.operation.op_id >= op.op_id:
                                    continue  # Skip current and future operations
                                if isinstance(prev_node.operation, PandasOperation):
                                    prev_op = prev_node.operation
                                    # Strategy 1: Check by object identity
                                    try:
                                        if id(prev_op.result) == id(arg):
                                            prev_var = self._get_var_name(prev_op.op_id)
                                            input_vars.append(prev_var)
                                            found_pandas_input = True
                                            break
                                    except:
                                        pass
                                    
                                    # Strategy 2: Check if it's the same type and appears in sequence
                                    # (This is a heuristic for when object identity changes)
                                    try:
                                        if (type(prev_op.result).__name__ == type(arg).__name__ and
                                            hasattr(arg, 'name') and hasattr(prev_op.result, 'name') and
                                            prev_op.result.name == arg.name):
                                            prev_var = self._get_var_name(prev_op.op_id)
                                            input_vars.append(prev_var)
                                            found_pandas_input = True
                                            break
                                    except:
                                        pass
                        
                        if not found_pandas_input:
                            # Handle other types
                            if isinstance(arg, str):
                                # Escape newlines and quotes in strings
                                escaped_arg = arg.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                                input_vars.append(f'"{escaped_arg}"')
                            elif isinstance(arg, type):
                                # Handle Python type objects (str, int, float, etc.)
                                if arg == str:
                                    input_vars.append("'s'")  # String dtype
                                elif arg == int:
                                    input_vars.append("'i'")  # Integer dtype
                                elif arg == float:
                                    input_vars.append("'f'")  # Float dtype
                                elif arg == bool:
                                    input_vars.append("'b'")  # Boolean dtype
                                else:
                                    input_vars.append("'s'")  # Default to string
                            else:
                                # For unknown pandas objects that we couldn't trace
                                # This indicates the object was modified or transformed
                                # We should try to find the most recent pandas operation of the same type
                                if EXTENDED_OPS_AVAILABLE:
                                    # Last resort: use the most recent pandas operation output
                                    last_pandas_var = None
                                    for prev_node in self.graph.nodes:
                                        if prev_node.operation.op_id >= op.op_id:
                                            break
                                        if isinstance(prev_node.operation, PandasOperation):
                                            last_pandas_var = self._get_var_name(prev_node.operation.op_id)
                                    
                                    if last_pandas_var:
                                        input_vars.append(f"/* WARNING: Guessed input */ {last_pandas_var}")
                                        found_pandas_input = True
                                
                                if not found_pandas_input:
                                    input_vars.append("NULL  /* Could not trace pandas input */")
        
        # Add strategy argument for numeric fillna operations
        if op.op_name in ('fillna', 'series_fillna') and c_func == 'pandas_series_fillna' and len(input_vars) == 2:
            # Numeric fillna needs strategy argument - add it now
            fill_value = op.args[1] if len(op.args) > 1 else op.kwargs.get('value', None)
            if not isinstance(fill_value, str):
                strategy = op.kwargs.get('method', 'constant')  # Default to constant
                if strategy not in ('constant', 'mean', 'median', 'forward', 'backward'):
                    strategy = 'constant'  # Default to constant for unknown strategies
                input_vars.append(f'"{strategy}"')  # Add strategy as string argument
        
        # Generate function call - make sure variable is declared correctly
        if return_type.endswith('*'):
            # Pointer return type - declare if not already declared
            if result_var not in allocated_vars and result_var not in declared_scalars:
                if return_type == "DataFrame*":
                    lines.append(f"    DataFrame* {result_var} = {c_func}({', '.join(input_vars)});")
                    self.dataframe_vars[op.op_id] = result_var
                    allocated_vars.add(result_var)
                elif return_type == "Series*":
                    lines.append(f"    Series* {result_var} = {c_func}({', '.join(input_vars)});")
                    self.series_vars[op.op_id] = result_var
                    allocated_vars.add(result_var)
                else:
                    lines.append(f"    {return_type} {result_var} = {c_func}({', '.join(input_vars)});")
                    allocated_vars.add(result_var)
            else:
                # Variable already declared, just assign
                lines.append(f"    {result_var} = {c_func}({', '.join(input_vars)});")
        else:
            # Scalar return type
            if result_var not in declared_scalars and result_var not in allocated_vars:
                lines.append(f"    {return_type} {result_var};")
                declared_scalars.add(result_var)
            lines.append(f"    {result_var} = {c_func}({', '.join(input_vars)});")
        
        return lines
    
    def _generate_string_operation_code(self, node: GraphNode,
                                       allocated_vars: Set[str],
                                       declared_scalars: Set[str]) -> List[str]:
        """Generate C code for a string operation."""
        if not STRING_OPS_AVAILABLE:
            return []
        
        op = node.operation
        if not isinstance(op, StringOperation):
            return []
        
        lines = []
        result_var = self._get_var_name(op.op_id)
        
        # Get C function mapping
        func_info = CFunctionRegistry.get_c_function(op.op_name, None)
        if not func_info:
            lines.append(f"    // ERROR: No C implementation for {op.op_name}")
            return lines
        
        self.string_include = True
        c_func = func_info['c_function']
        return_type = func_info['return_type']
        
        # Check if we're in batch processing mode and this operation uses batch input
        is_batch_mode = len(self.batch_string_inputs) > 0
        uses_batch_input = False
        batch_input_var_name = None
        
        for arg in op.args:
            if isinstance(arg, str) and self.input_arrays:
                for var_name, var_value in self.input_arrays.items():
                    if var_name in self.batch_string_inputs and isinstance(var_value, (list, tuple)):
                        uses_batch_input = True
                        batch_input_var_name = var_name
                        break
        
        # Get input variables - handle string inputs from function args or input variables
        input_vars = []
        arg_index = 0
        
        for arg in op.args:
            # Check if this is an input variable (string input)
            is_input_var = False
            if isinstance(arg, str) and self.input_arrays:
                for var_name, var_value in self.input_arrays.items():
                    # Check if it's a batch input (list of strings)
                    if var_name in self.batch_string_inputs and isinstance(var_value, (list, tuple)):
                        # In batch mode, use current_text instead of parsing
                        if is_batch_mode and uses_batch_input:
                            input_vars.append("current_text")
                            is_input_var = True
                            break
                        # Not in batch mode yet - fall through to regular handling
                    elif isinstance(var_value, str) and var_value == arg:
                        # It's a single string input variable - need to convert from double* to char*
                        self.string_inputs.add(var_name)
                        # Use op_id to make variable name unique per operation
                        conversion_key = f"{var_name}_{op.op_id}"
                        if conversion_key not in self.string_conversions:
                            str_var = f"{var_name}_str_{op.op_id}"
                            # Use different indentation for batch mode
                            indent = "        " if is_batch_mode else "    "
                            lines.append(f"{indent}// Convert string input {var_name} from double* to char*")
                            lines.append(f"{indent}char {str_var}[256];")
                            lines.append(f"{indent}int {var_name}_len_{op.op_id} = 0;")
                            lines.append(f"{indent}double val_{op.op_id};")
                            lines.append(f"{indent}while ({var_name}_len_{op.op_id} < 255 && (val_{op.op_id} = {var_name}[{var_name}_len_{op.op_id}]) > 0 && val_{op.op_id} < 256) {{")
                            lines.append(f"{indent}    {str_var}[{var_name}_len_{op.op_id}++] = (char)(int)val_{op.op_id};")
                            lines.append(f"{indent}}}")
                            lines.append(f"{indent}{str_var}[{var_name}_len_{op.op_id}] = '\\0';")
                            self.string_conversions[conversion_key] = str_var
                        else:
                            str_var = self.string_conversions[conversion_key]
                        input_vars.append(str_var)
                        is_input_var = True
                        break
            
            if not is_input_var:
                # Check if this is a result from a previous string operation
                found_prev = False
                if isinstance(arg, str):
                    # Check if this string came from a previous operation
                    for prev_node in self.graph.nodes:
                        if prev_node.operation.op_id >= op.op_id:
                            break
                        prev_op = prev_node.operation
                        if isinstance(prev_op, StringOperation):
                            try:
                                if isinstance(prev_op.result, str) and prev_op.result == arg:
                                    # Use previous result variable
                                    prev_var = self._get_var_name(prev_op.op_id)
                                    input_vars.append(prev_var)
                                    found_prev = True
                                    break
                            except:
                                pass
                
                if not found_prev:
                    # Literal string or other value
                    if isinstance(arg, str):
                        escaped = arg.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                        input_vars.append(f'"{escaped}"')
                    elif isinstance(arg, (int, float, bool)):
                        input_vars.append(str(arg))
                    else:
                        input_vars.append("NULL")
            
            arg_index += 1
        
        # Generate function call with appropriate indentation
        indent = "        " if is_batch_mode else "    "
        if return_type == 'char*':
            # String return - allocate
            lines.append(f"{indent}{return_type} {result_var} = {c_func}({', '.join(input_vars)});")
            allocated_vars.add(result_var)
        elif return_type == 'bool':
            # Boolean return - declare as int
            if result_var not in declared_scalars:
                lines.append(f"{indent}int {result_var};")
                declared_scalars.add(result_var)
            lines.append(f"{indent}{result_var} = {c_func}({', '.join(input_vars)});")
        else:
            # Other types
            if result_var not in allocated_vars and result_var not in declared_scalars:
                lines.append(f"{indent}{return_type} {result_var};")
                declared_scalars.add(result_var)
            lines.append(f"{indent}{result_var} = {c_func}({', '.join(input_vars)});")
        
        return lines
    
    def _generate_operation_code(self, node: GraphNode, allocated_vars: Set[str], 
                                declared_scalars: Set[str]) -> List[str]:
        """Override to handle extended operations."""
        op = node.operation
        
        # Try pandas operations first
        if EXTENDED_OPS_AVAILABLE and isinstance(op, PandasOperation):
            result = self._generate_pandas_operation_code(node, allocated_vars, declared_scalars)
            if result:
                return result
        
        # Try string operations
        if STRING_OPS_AVAILABLE and isinstance(op, StringOperation):
            result = self._generate_string_operation_code(node, allocated_vars, declared_scalars)
            if result:
                return result
        
        # Fall back to parent implementation for NumPy operations
        return super()._generate_operation_code(node, allocated_vars, declared_scalars)
    
    def generate(self, output_var_names: List[str] = None) -> str:
        """Generate complete C code with extended includes."""
        lines = []
        
        # Pre-scan for pandas and string operations to set include flags
        if EXTENDED_OPS_AVAILABLE:
            for node in self.graph.nodes:
                op = node.operation
                if isinstance(op, PandasOperation):
                    self.pandas_include = True
                elif STRING_OPS_AVAILABLE and isinstance(op, StringOperation):
                    self.string_include = True
        
        # Add extended includes if needed
        if self.pandas_include:
            self.includes.add('#include "pandas_c.h"')
            self.includes.add('#include "data_structures.h"')
        if self.string_include:
            self.includes.add('#include "string_c.h"')
            # Add string.h for strlen
            self.includes.add('#include <string.h>')
        
        # Add float.h for INFINITY and NAN
        if '#include <float.h>' not in self.includes:
            self.includes.add('#include <float.h>')
        
        # Header includes
        for include in sorted(self.includes):
            lines.append(include)
        lines.append("")
        
        # Function signature
        input_mapping = {}
        for i, var_name in enumerate(self.input_vars):
            input_mapping[var_name] = f"inputs[{i}]"
        
        lines.append("void preprocess(double** inputs, int num_inputs, double** outputs, int num_outputs) {")
        lines.append("")
        
        # Map input arrays
        # First, identify which inputs are strings (single or list) by checking operations and input_arrays
        batch_string_inputs = set()  # Track which inputs are lists of strings
        for var_name, var_value in (self.input_arrays.items() if self.input_arrays else {}):
            # Check if this is a list of strings
            if isinstance(var_value, (list, tuple)) and len(var_value) > 0 and isinstance(var_value[0], str):
                batch_string_inputs.add(var_name)
                self.string_inputs.add(var_name)
        
        for node in self.graph.nodes:
            op = node.operation
            # Check pandas operations for string inputs (e.g., read_csv)
            if EXTENDED_OPS_AVAILABLE and isinstance(op, PandasOperation):
                if op.op_name == 'read_csv' and len(op.args) > 0:
                    csv_path_arg = op.args[0]
                    # Find which input variable has this string value
                    if isinstance(csv_path_arg, str) and self.input_arrays:
                        for var_name, var_value in self.input_arrays.items():
                            if isinstance(var_value, str) and var_value == csv_path_arg:
                                self.string_inputs.add(var_name)
                                break
            # Check string operations for string inputs
            elif STRING_OPS_AVAILABLE and isinstance(op, StringOperation):
                for arg in op.args:
                    if isinstance(arg, str) and self.input_arrays:
                        for var_name, var_value in self.input_arrays.items():
                            if isinstance(var_value, str) and var_value == arg:
                                self.string_inputs.add(var_name)
                                break
        
        # Map inputs - strings will be converted later
        for var_name in self.input_vars:
            if var_name in input_mapping:
                if var_name in self.string_inputs:
                    # String input - declare as double* for now, convert later when used
                    lines.append(f"    double* {var_name} = {input_mapping[var_name]};")
                else:
                    lines.append(f"    double* {var_name} = {input_mapping[var_name]};")
        
        # Store batch string inputs info for later use
        self.batch_string_inputs = batch_string_inputs
        
        # Check if we have batch string processing - if so, wrap everything in a loop
        has_batch_processing = len(batch_string_inputs) > 0
        
        if has_batch_processing:
            # For batch processing, we need to:
            # 1. Parse the batch format to get number of strings
            # 2. Allocate result arrays (one per output, sized to num_strings)
            # 3. Loop over each string
            # 4. Process each string and store results in arrays
            
            # Get the batch input variable name (should be first/only batch input)
            batch_input_var = list(batch_string_inputs)[0]
            
            # Parse batch format: [num_strings, len1, chars1..., len2, chars2..., ...]
            lines.append(f"    // Parse batch string input: {batch_input_var}")
            lines.append(f"    int num_strings = (int){batch_input_var}[0];")
            lines.append(f"    int {batch_input_var}_offset = 1;  // Start after num_strings")
            lines.append("")
            
            # Allocate result arrays based on output operations
            # We'll determine the outputs after processing nodes, but allocate space now
            lines.append("    // Allocate result arrays (will be filled in loop below)")
            
        lines.append("")
        
        # Process nodes in topological order
        sorted_nodes = self.graph.topological_sort()
        
        # Track allocated variables to avoid redefinition
        allocated_vars: Set[str] = set()
        declared_scalars: Set[str] = set()
        
        # For batch processing, we need to track which outputs should be arrays
        # and pre-allocate them
        batch_output_vars = {}  # op_id -> array_var_name
        if has_batch_processing:
            # Identify output operations that should become arrays
            # In batch processing mode, ALL outputs should be arrays (one per string)
            for node in self.graph.output_nodes:
                op = node.operation
                op_id = op.op_id
                var_name = self._get_var_name(op.op_id)
                
                # For batch processing, all outputs should be arrays
                # Determine result type
                result_type = 'double'  # Default
                if isinstance(op.result, (list, tuple)):
                    # It's already a list/tuple - needs to be an array
                    if len(op.result) > 0:
                        result_type = type(op.result[0]).__name__
                elif STRING_OPS_AVAILABLE and isinstance(op, StringOperation):
                    # String operation outputs should be arrays in batch mode
                    func_info = CFunctionRegistry.get_c_function(op.op_name, None)
                    if func_info:
                        return_type = func_info.get('return_type')
                        if return_type == 'bool':
                            # Bool output becomes array of ints
                            result_type = 'int'
                        elif return_type == 'char*':
                            # String length output becomes array of ints
                            result_type = 'int'  # We'll store lengths
                elif isinstance(op.result, (int, float, bool)):
                    # Scalar output becomes array
                    if isinstance(op.result, bool):
                        result_type = 'int'
                    else:
                        result_type = 'double'
                
                # Add to batch_output_vars
                batch_output_vars[op_id] = {
                    'var_name': var_name,
                    'array_var': f"{var_name}_batch",
                    'result_type': result_type
                }
            
            # Pre-allocate result arrays
            if batch_output_vars:
                lines.append(f"    // Pre-allocate result arrays for batch processing")
                batch_input_var = list(batch_string_inputs)[0]
                for op_id, info in batch_output_vars.items():
                    array_var = info['array_var']
                    result_type = info['result_type']
                    if result_type == 'int':
                        lines.append(f"    int* {array_var} = (int*)malloc(num_strings * sizeof(int));")
                    else:
                        lines.append(f"    double* {array_var} = (double*)malloc(num_strings * sizeof(double));")
                    allocated_vars.add(array_var)
                lines.append("")
        
        # Allocate variables for intermediate results - but skip pandas operations
        # They will be handled by their specific code generators
        for node in sorted_nodes:
            op = node.operation
            
            # Skip pandas and string operations - they'll be handled separately
            if EXTENDED_OPS_AVAILABLE:
                if isinstance(op, PandasOperation):
                    continue  # Skip upfront allocation for these
            if STRING_OPS_AVAILABLE:
                if isinstance(op, StringOperation):
                    continue  # Skip upfront allocation for these
            
            var_name = self._get_var_name(op.op_id)
            
            # Skip if already allocated or declared
            if var_name in allocated_vars or var_name in declared_scalars:
                continue
            
            if isinstance(op.result, np.ndarray):
                # Check if this is not an input array
                if id(op.result) not in self.array_to_var_map:
                    shape = op.result.shape
                    ndim = op.result.ndim if hasattr(op.result, 'ndim') else len(shape) if shape else 0
                    # Check if it's a 0-d array (scalar) - shape should be () or ndim == 0
                    # 0-d arrays (ndim == 0 or shape == ()) should be treated as scalars
                    if ndim == 0 or (shape and len(shape) == 0):
                        # 0-d array - treat as scalar
                        if var_name not in allocated_vars and var_name not in declared_scalars:
                            lines.append(f"    double {var_name};")
                            declared_scalars.add(var_name)
                    elif shape and len(shape) > 0:
                        # Regular array (including empty arrays with shape like (0,))
                        # Allocate as array even if np.prod(shape) == 0 (empty array)
                        lines.append(self._generate_array_allocation(var_name, shape, op.result.dtype))
                        allocated_vars.add(var_name)
                    else:
                        # Invalid shape - treat as scalar
                        if var_name not in allocated_vars and var_name not in declared_scalars:
                            lines.append(f"    double {var_name};")
                            declared_scalars.add(var_name)
            else:
                # Declare scalar variables upfront (only if not already declared)
                if var_name not in allocated_vars and var_name not in declared_scalars:
                    lines.append(f"    double {var_name};")
                    declared_scalars.add(var_name)
        
        lines.append("")
        
        # Generate batch processing loop if needed
        if has_batch_processing:
            batch_input_var = list(batch_string_inputs)[0]
            lines.append(f"    // Batch processing loop - process each string")
            lines.append(f"    for (int batch_idx = 0; batch_idx < num_strings; batch_idx++) {{")
            lines.append(f"        // Extract current string from batch format")
            lines.append(f"        // Format: [num_strings, len1, chars1..., len2, chars2..., ...]")
            lines.append(f"        int current_str_len = (int){batch_input_var}[{batch_input_var}_offset++];")
            lines.append(f"        // Allocate buffer for current string")
            lines.append(f"        char* current_text = (char*)malloc((current_str_len + 1) * sizeof(char));")
            lines.append(f"        // Copy characters from encoded format")
            lines.append(f"        for (int i = 0; i < current_str_len; i++) {{")
            lines.append(f"            current_text[i] = (char)(int){batch_input_var}[{batch_input_var}_offset++];")
            lines.append(f"        }}")
            lines.append(f"        current_text[current_str_len] = '\\0';")
            lines.append("")
        
        # Generate operation code - only process each operation once
        processed_ops: Set[int] = set()
        for node in sorted_nodes:
            op_id = node.operation.op_id
            if op_id in processed_ops:
                continue
            processed_ops.add(op_id)
            
            # For batch processing, we need to modify how we generate code
            # String operations should use current_text instead of parsing from input
            op_lines = self._generate_operation_code(node, allocated_vars, declared_scalars)
            
            # If batch processing and this is a string operation that uses the batch input,
            # we need to replace the input conversion with current_text
            if has_batch_processing and STRING_OPS_AVAILABLE and isinstance(node.operation, StringOperation):
                op = node.operation
                # Check if this operation uses the batch input variable
                for arg in op.args:
                    if isinstance(arg, str) and self.input_arrays:
                        for var_name, var_value in self.input_arrays.items():
                            if var_name in batch_string_inputs and isinstance(var_value, (list, tuple)) and len(var_value) > 0:
                                # This operation uses the batch input - replace input conversion
                                # The string conversion code will use current_text instead
                                modified_lines = []
                                for line in op_lines:
                                    # Replace string input conversion with current_text usage
                                    if f"{var_name}_str_" in line and "Convert string input" in line:
                                        # Skip the conversion code - we already have current_text
                                        continue
                                    elif f"{var_name}_str_" in line:
                                        # Replace with current_text
                                        modified_line = line.replace(f"{var_name}_str_{op.op_id}", "current_text")
                                        modified_lines.append(modified_line)
                                    else:
                                        modified_lines.append(line)
                                op_lines = modified_lines
                                break
            
            lines.extend(op_lines)
            if op_lines:
                lines.append("")
            
            # After generating operation code, if batch processing and this is an output,
            # store result in the batch array
            if has_batch_processing and op_id in batch_output_vars:
                info = batch_output_vars[op_id]
                array_var = info['array_var']
                var_name = info['var_name']
                result_type = info['result_type']
                
                # Store result in array at current index
                if result_type == 'int':
                    lines.append(f"        // Store result in batch array")
                    lines.append(f"        {array_var}[batch_idx] = (int){var_name};")
                elif STRING_OPS_AVAILABLE and isinstance(node.operation, StringOperation):
                    # Check if result is a string (char*) by checking if var_name is declared as char*
                    # or by checking the function return type
                    op = node.operation
                    op_name = getattr(op, 'op_name', '')
                    func_info = CFunctionRegistry.get_c_function(op_name, None)
                    # Also check if result is a string type
                    result_is_string = isinstance(op.result, str) if hasattr(op, 'result') else False
                    operation_type = getattr(op, 'operation_type', '')
                    
                    # Check if this operation returns a string (char*)
                    is_string_result = (func_info and func_info.get('return_type') == 'char*') or \
                                      result_is_string or \
                                      operation_type in ('unicode_normalize', 'str_format', 're_sub') or \
                                      op_name in ('unicodedata_normalize', 'str_format', 're_sub')
                    
                    if is_string_result:
                        # Store string length (not pointer address!)
                        lines.append(f"        // Store string length in batch array")
                        lines.append(f"        {array_var}[batch_idx] = (int)({var_name} ? strlen({var_name}) : 0);")
                        lines.append(f"        // Free string after storing length")
                        lines.append(f"        if ({var_name}) free({var_name});")
                    elif func_info and func_info.get('return_type') == 'bool':
                        # Bool result
                        lines.append(f"        {array_var}[batch_idx] = (int){var_name};")
                    else:
                        lines.append(f"        {array_var}[batch_idx] = (double){var_name};")
                else:
                    lines.append(f"        {array_var}[batch_idx] = (double){var_name};")
                lines.append("")
        
        # Close batch processing loop
        if has_batch_processing:
            lines.append("        // Free current string buffer")
            lines.append("        if (current_text) free(current_text);")
            lines.append("    }")
            lines.append("")
        
        # Copy outputs - handle arrays, scalars, and pandas objects
        # For batch processing, outputs are already in arrays
        output_index = 0
        seen_outputs: Set[int] = set()
        for node in self.graph.output_nodes:
            op = node.operation
            op_id = op.op_id
            
            if op_id in seen_outputs:
                continue
            seen_outputs.add(op_id)
            
            output_var = self._get_var_name(op.op_id)
            
            # For batch processing, if this output is in batch_output_vars, use the batch array
            if has_batch_processing and op_id in batch_output_vars:
                info = batch_output_vars[op_id]
                array_var = info['array_var']
                # Convert to double* array for output
                batch_output_array = f"{array_var}_output"
                lines.append(f"    // Convert batch result array to output format")
                lines.append(f"    double* {batch_output_array} = (double*)malloc(num_strings * sizeof(double));")
                lines.append(f"    for (int i = 0; i < num_strings; i++) {{")
                lines.append(f"        {batch_output_array}[i] = (double){array_var}[i];")
                lines.append(f"    }}")
                lines.append(f"    outputs[{output_index}] = {batch_output_array};")
                allocated_vars.add(batch_output_array)
                output_index += 1
                continue
            
            # Check if this is a string operation with string output
            if STRING_OPS_AVAILABLE and isinstance(op, StringOperation):
                # String operations return char* or bool
                # For char* outputs, convert to double* array for return
                func_info = CFunctionRegistry.get_c_function(op.op_name, None)
                if func_info and func_info.get('return_type') == 'char*':
                    # Convert char* to double* array (character codes)
                    char_to_array_var = f"{output_var}_array"
                    lines.append(f"    // Convert char* result to double* array")
                    lines.append(f"    int {char_to_array_var}_len = 0;")
                    lines.append(f"    if ({output_var}) {{")
                    lines.append(f"        {char_to_array_var}_len = strlen({output_var}) + 1;  // Include null terminator")
                    lines.append(f"    }} else {{")
                    lines.append(f"        {char_to_array_var}_len = 1;  // Just null terminator")
                    lines.append(f"    }}")
                    lines.append(f"    double* {char_to_array_var} = (double*)malloc({char_to_array_var}_len * sizeof(double));")
                    lines.append(f"    if ({output_var}) {{")
                    lines.append(f"        for (int i = 0; i < {char_to_array_var}_len - 1; i++) {{")
                    lines.append(f"            {char_to_array_var}[i] = (double)(unsigned char){output_var}[i];")
                    lines.append(f"        }}")
                    lines.append(f"    }}")
                    lines.append(f"    {char_to_array_var}[{char_to_array_var}_len - 1] = 0.0;  // Null terminator")
                    lines.append(f"    outputs[{output_index}] = {char_to_array_var};")
                    allocated_vars.add(char_to_array_var)
                    output_index += 1
                    continue
                # For bool outputs, handle as scalar
                elif func_info and func_info.get('return_type') == 'bool':
                    # Bool is already stored as int, output as scalar
                    scalar_var = f"{output_var}_output"
                    lines.append(f"    double* {scalar_var} = (double*)malloc(sizeof(double));")
                    lines.append(f"    {scalar_var}[0] = (double){output_var};")
                    lines.append(f"    outputs[{output_index}] = {scalar_var};")
                    output_index += 1
                    continue
            
            # Check if this is a pandas operation with non-numeric output
            if EXTENDED_OPS_AVAILABLE and isinstance(op, PandasOperation):
                # Check if result is DataFrame or Series - extract numeric data
                # Use dataframe_vars/series_vars dicts which are set during code generation
                if op.op_id in self.dataframe_vars:
                    # Extract numeric data from DataFrame
                    df_var = self.dataframe_vars[op.op_id]
                    
                    # Generate extraction code with NULL check
                    extracted_var = f"{df_var}_extracted"
                    lines.append(f"    // Extract numeric data from DataFrame {df_var}")
                    lines.append(f"    int64_t {extracted_var}_length = 0;")
                    lines.append(f"    double* {extracted_var} = NULL;")
                    lines.append(f"    if ({df_var} != NULL) {{")
                    lines.append(f"        {extracted_var} = dataframe_to_array({df_var}, &{extracted_var}_length);")
                    lines.append(f"    }}")
                    lines.append(f"    if (!{extracted_var}) {{")
                    lines.append(f"        {extracted_var} = (double*)malloc(sizeof(double));")
                    lines.append(f"        {extracted_var}[0] = NAN;")
                    lines.append(f"        {extracted_var}_length = 1;")
                    lines.append(f"    }}")
                    lines.append(f"    outputs[{output_index}] = {extracted_var};")
                    allocated_vars.add(extracted_var)
                    output_index += 1
                    continue
                
                # Check if this operation created a Series*
                if op.op_id in self.series_vars:
                    series_var = self.series_vars[op.op_id]
                    
                    # Generate extraction code
                    extracted_var = f"{series_var}_extracted"
                    lines.append(f"    // Extract numeric data from Series {series_var}")
                    lines.append(f"    int64_t {extracted_var}_length;")
                    lines.append(f"    double* {extracted_var} = series_to_array({series_var}, &{extracted_var}_length);")
                    lines.append(f"    if (!{extracted_var}) {{")
                    lines.append(f"        {extracted_var} = (double*)malloc(sizeof(double));")
                    lines.append(f"        {extracted_var}[0] = NAN;")
                    lines.append(f"        {extracted_var}_length = 1;")
                    lines.append(f"    }}")
                    lines.append(f"    outputs[{output_index}] = {extracted_var};")
                    allocated_vars.add(extracted_var)
                    output_index += 1
                    continue
                
                # If it's a PandasOperation but not in dataframe_vars/series_vars,
                # it's likely a scalar (e.g., mean())
                # Fall through to handle it as a scalar
            
            # Check if variable was allocated as an array or declared as a scalar
            # This handles 0-d numpy arrays which are isinstance(np.ndarray) but declared as scalars
            # Check declared_scalars first since some vars might be in both sets erroneously
            if output_var in declared_scalars:
                # Variable is a scalar (double) - allocate a single-element array
                scalar_var = f"{output_var}_output"
                lines.append(f"    double* {scalar_var} = (double*)malloc(sizeof(double));")
                lines.append(f"    {scalar_var}[0] = {output_var};")
                lines.append(f"    outputs[{output_index}] = {scalar_var};")
            elif output_var in allocated_vars:
                # Variable is an array (double*)
                lines.append(f"    outputs[{output_index}] = {output_var};")
            else:
                # Unknown variable - treat as scalar
                scalar_var = f"{output_var}_output"
                lines.append(f"    double* {scalar_var} = (double*)malloc(sizeof(double));")
                lines.append(f"    {scalar_var}[0] = {output_var};")
                lines.append(f"    outputs[{output_index}] = {scalar_var};")
            output_index += 1
        
        lines.append("}")
        
        return "\n".join(lines)


def generate_extended_c_code(graph: ComputationalGraph, input_vars: List[str],
                            output_vars: List[str] = None,
                            input_arrays: Dict[str, np.ndarray] = None,
                            link_libraries: List[str] = None) -> str:
    """
    Generate C code with support for pandas operations.
    
    Args:
        graph: Computational graph
        input_vars: List of input variable names
        output_vars: Optional list of output variable names
        input_arrays: Dictionary mapping input variable names to their arrays
        link_libraries: List of library names to link (e.g., ['pandas_c'])
        
    Returns:
        C code as string
    """
    generator = ExtendedCCodeGenerator(graph, input_vars, input_arrays, link_libraries)
    return generator.generate(output_vars)

