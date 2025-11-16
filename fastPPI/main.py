"""
Main entry point for FastPPI framework.
Command-line interface for converting Python preprocessing to C binaries.
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from .tracers import trace_execution
from .core import build_computational_graph, generate_c_code, compile_to_binary
from .core.extended_codegen import generate_extended_c_code, EXTENDED_OPS_AVAILABLE

# Try to import pandas for metadata generation
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Try to import unified tracer for pandas support
try:
    from .analysis.unified_tracer import trace_unified_execution
    UNIFIED_TRACER_AVAILABLE = True
except ImportError:
    UNIFIED_TRACER_AVAILABLE = False

if EXTENDED_OPS_AVAILABLE:
    try:
        from .tracers.pandas_tracer import PandasOperation
    except ImportError:
        PandasOperation = None
else:
    PandasOperation = None


def load_python_file(filepath: str, example_inputs: Dict[str, Any] = None) -> str:
    """Load Python code from file and prepare it for execution."""
    with open(filepath, 'r') as f:
        code = f.read()
    
    # If code has if __name__ == "__main__" block, try to extract and call main function
    # This handles scripts that use argparse
    if 'if __name__ == "__main__":' in code and example_inputs:
        # Try to find a function that takes the input variables
        # For feature_engineering.py, we need to call preprocess_data(csv_path)
        lines = code.split('\n')
        new_lines = []
        skip_main_block = False
        
        for i, line in enumerate(lines):
            if 'if __name__ == "__main__":' in line:
                skip_main_block = True
                # Instead of argparse, call the function directly
                # Look for functions that match input variables
                for func_name in ['preprocess_data', 'main', 'process']:
                    if f'def {func_name}(' in code:
                        # Add call to function with example inputs
                        if func_name == 'preprocess_data' and 'csv_path' in example_inputs:
                            # Keep X_processed and y as final outputs - don't reassign
                            new_lines.append(f'X_processed, y = {func_name}(csv_path)')
                        elif len(example_inputs) == 1:
                            var_name = list(example_inputs.keys())[0]
                            new_lines.append(f'result = {func_name}({var_name})')
                        break
                continue
            elif skip_main_block:
                # Skip lines in the main block
                if line.strip() and not line[0:len(line) - len(line.lstrip())]:
                    # Non-indented line means we're out of the block
                    if not (line.strip().startswith('#') or 'parse_args' in line or 'parser' in line):
                        skip_main_block = False
                        if not line.strip().startswith('X_processed') and not line.strip().startswith('result'):
                            new_lines.append(line)
                continue
            else:
                new_lines.append(line)
        
        if skip_main_block or 'if __name__ == "__main__":' in code:
            # Fallback: just remove the main block and call function directly
            if 'def preprocess_data(' in code and 'csv_path' in example_inputs:
                code = code.split('if __name__ == "__main__":')[0]
                # Call the function and ensure results are captured as outputs
                # The outputs need to be in the top-level scope to be traced
                code += f'\n# Call preprocessing function\n'
                code += f'X_processed, y = preprocess_data(csv_path)\n'
                code += f'# Ensure outputs are accessible (for tracing)\n'
                code += f'result_X = X_processed\n'
                code += f'result_y = y\n'
            elif example_inputs and len(example_inputs) == 1:
                var_name = list(example_inputs.keys())[0]
                # Try to find any function that takes one argument
                import re
                func_matches = re.findall(r'def\s+(\w+)\s*\([^)]*\):', code)
                if func_matches:
                    func_name = func_matches[0]
                    code = code.split('if __name__ == "__main__":')[0]
                    # Check if function returns tuple by inspecting return statement
                    # For now, try unpacking to multiple variables for better tracing
                    # Common patterns: return a, b, c or return (a, b, c)
                    # We'll use a pattern that works for both single and multiple returns
                    # If it's a tuple, unpack it; otherwise, assign directly
                    # Try to detect tuple return by checking return statement
                    return_pattern = rf'return\s+.*{func_name}'
                    has_multiple_returns = len(re.findall(r'return\s+[^,\n]+\s*,\s*[^,\n]+', code)) > 0
                    if has_multiple_returns:
                        # Try unpacking to named variables
                        # Use generic names that can be matched later
                        code += f'\n# Unpack function result\n'
                        code += f'_temp_result = {func_name}({var_name})\n'
                        code += f'if isinstance(_temp_result, tuple):\n'
                        code += f'    result = _temp_result  # Keep tuple for now\n'
                        code += f'else:\n'
                        code += f'    result = _temp_result\n'
                    else:
                        code += f'\nresult = {func_name}({var_name})\n'
        else:
            code = '\n'.join(new_lines)
            
        # Ensure we're not adding duplicate assignments
    
    return code


def parse_example_inputs(input_str: str) -> Dict[str, Any]:
    """
    Parse example inputs from string.
    Format: "var1=[[1,2],[3,4]],var2=5", JSON string, or JSON file path
    """
    # Try loading as JSON file first
    if os.path.exists(input_str):
        with open(input_str, 'r') as f:
            data = json.load(f)
            # Convert lists to numpy arrays (for numeric data only)
            result = {}
            for k, v in data.items():
                if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    result[k] = np.array(v, dtype=np.float64)
                else:
                    result[k] = v
            return result
    
    # Try parsing as JSON string
    input_str_stripped = input_str.strip()
    if input_str_stripped.startswith('{') and input_str_stripped.endswith('}'):
        try:
            data = json.loads(input_str)
            # Convert lists to numpy arrays (for numeric data only)
            result = {}
            for k, v in data.items():
                if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    result[k] = np.array(v, dtype=np.float64)
                else:
                    result[k] = v
            return result
        except json.JSONDecodeError:
            pass  # Fall through to key=value parsing
    
    # Parse from key=value string format
    # Need to handle commas inside list literals properly
    result = {}
    i = 0
    while i < len(input_str):
        # Skip whitespace
        while i < len(input_str) and input_str[i].isspace():
            i += 1
        if i >= len(input_str):
            break
        
        # Find the key (until '=')
        key_start = i
        while i < len(input_str) and input_str[i] != '=':
            i += 1
        if i >= len(input_str):
            break
        
        key = input_str[key_start:i].strip()
        i += 1  # Skip '='
        
        # Find the value (handle nested brackets)
        value_start = i
        bracket_count = 0
        in_string = False
        string_char = None
        
        while i < len(input_str):
            char = input_str[i]
            
            # Handle strings
            if char in ("'", '"') and (i == 0 or input_str[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            
            # Handle brackets (only when not in string)
            elif not in_string:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == ',' and bracket_count == 0:
                    # This comma is a separator, not part of a list
                    break
            
            i += 1
        
        value_str = input_str[value_start:i].strip()
        
        # Try to evaluate as Python literal
        try:
            value = eval(value_str)
            if isinstance(value, list):
                # Ensure numeric arrays have proper dtype
                result[key] = np.array(value, dtype=np.float64)
            elif isinstance(value, (int, float)):
                result[key] = float(value)
            elif isinstance(value, np.ndarray):
                # Ensure it's float64
                if value.dtype != np.float64:
                    result[key] = value.astype(np.float64)
                else:
                    result[key] = value
            else:
                result[key] = value
        except Exception:
            # If eval fails, try parsing as array directly
            try:
                if value_str.startswith('[') and value_str.endswith(']'):
                    # Remove brackets and parse
                    inner = value_str[1:-1].strip()
                    if inner:
                        numbers = [float(x.strip()) for x in inner.split(',') if x.strip()]
                        result[key] = np.array(numbers, dtype=np.float64)
                    else:
                        result[key] = np.array([], dtype=np.float64)
                else:
                    # Try to convert to float
                    result[key] = float(value_str)
            except:
                result[key] = value_str
        
        # Skip comma if we stopped at one
        if i < len(input_str) and input_str[i] == ',':
            i += 1
    
    return result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FastPPI: Convert Python preprocessing code to optimized C binaries"
    )
    parser.add_argument(
        "python_file",
        type=str,
        help="Path to Python file containing preprocessing code"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        required=True,
        help="Example inputs as 'var1=value1,var2=value2' or path to JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="preprocess_binary",
        help="Output path for compiled binary (default: preprocess_binary)"
    )
    parser.add_argument(
        "--save-c",
        type=str,
        default=None,
        help="Save generated C code to file (optional)"
    )
    parser.add_argument(
        "--optimization",
        type=str,
        default="-O3",
        help="Compiler optimization flag (default: -O3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse example inputs first (needed for code modification)
        if args.verbose:
            print(f"Parsing example inputs: {args.inputs}")
        example_inputs = parse_example_inputs(args.inputs)
        
        # Load Python code (may modify it based on inputs)
        if args.verbose:
            print(f"Loading Python file: {args.python_file}")
        python_code = load_python_file(args.python_file, example_inputs)
        if args.verbose:
            print(f"Example inputs: {list(example_inputs.keys())}")
        
        # Trace execution - use unified tracer if available for pandas
        if args.verbose:
            print("Tracing execution...")
        if UNIFIED_TRACER_AVAILABLE:
            if args.verbose:
                print("Using unified tracer (supports NumPy, pandas)")
            operations = trace_unified_execution(python_code, example_inputs)
        else:
            operations = trace_execution(python_code, example_inputs)
        if args.verbose:
            print(f"Captured {len(operations)} operations")
        
        if len(operations) == 0:
            print("Warning: No operations were captured. Make sure your code uses NumPy operations.")
        
        # Build computational graph
        if args.verbose:
            print("Building computational graph...")
        graph = build_computational_graph(operations)
        graph.build()
        
        # Filter output nodes to only include final results (X_processed, y, etc.)
        # Re-execute code to get the final namespace and identify which operations
        # produced the final output variables
        if UNIFIED_TRACER_AVAILABLE:
            exec_globals = {
                '__builtins__': __builtins__,
                'np': np, 'numpy': np,
            }
            try:
                import pandas as pd
                exec_globals['pd'] = pd
                exec_globals['pandas'] = pd
            except ImportError:
                pass
            # Add string operation modules
            try:
                import re
                import unicodedata
                exec_globals['re'] = re
                exec_globals['unicodedata'] = unicodedata
            except ImportError:
                pass
            exec_globals.update(example_inputs)
            exec_locals = {}
            exec(python_code, exec_globals, exec_locals)
            final_namespace = {**exec_globals, **exec_locals}
            
            # Find final output variables - check for X_processed and y (the actual return values)
            final_output_objects = []
            output_var_names = []
            
            # Check for X_processed and y (the function return values)
            if 'X_processed' in final_namespace:
                final_output_objects.append(final_namespace['X_processed'])
                output_var_names.append('X_processed')
            if 'y' in final_namespace:
                final_output_objects.append(final_namespace['y'])
                output_var_names.append('y')
            
            # Check for tuple result - unpack it
            if 'result' in final_namespace:
                result = final_namespace['result']
                if isinstance(result, tuple):
                    # Unpack tuple into individual outputs
                    for i, item in enumerate(result):
                        final_output_objects.append(item)
                        output_var_names.append(f'output_{i}')
                else:
                    final_output_objects.append(result)
                    output_var_names.append('result')
            
            # Fallback to output if nothing else found
            if not final_output_objects:
                for var_name in ['output']:
                    if var_name in final_namespace:
                        output_val = final_namespace[var_name]
                        if isinstance(output_val, tuple):
                            for i, item in enumerate(output_val):
                                final_output_objects.append(item)
                                output_var_names.append(f'output_{i}')
                        else:
                            final_output_objects.append(output_val)
                            output_var_names.append(var_name)
                        break
            
            # Find nodes that produce these final outputs
            if final_output_objects and output_var_names:
                if args.verbose:
                    print(f"Found final output variables: {output_var_names}")
                
                final_output_nodes = []
                for i, output_obj in enumerate(final_output_objects):
                    var_name = output_var_names[i]
                    # Find which operation produced this object
                    # Check in reverse to get the most recent operation that produced it
                    found = False
                    for node in reversed(graph.nodes):
                        op = node.operation
                        try:
                            if hasattr(op, 'result'):
                                # Check object identity first
                                if id(op.result) == id(output_obj):
                                    if node not in final_output_nodes:
                                        final_output_nodes.append(node)
                                        if args.verbose:
                                            print(f"  Matched {var_name} to operation {op.op_name} (op_id={op.op_id})")
                                    found = True
                                    break
                                # For numeric types (int, float), check value equality
                                elif isinstance(output_obj, (int, float)) and isinstance(op.result, type(output_obj)):
                                    if output_obj == op.result:
                                        if node not in final_output_nodes:
                                            final_output_nodes.append(node)
                                            if args.verbose:
                                                print(f"  Matched {var_name} (value) to operation {op.op_name} (op_id={op.op_id})")
                                        found = True
                                        break
                                # For string types, check string equality
                                elif isinstance(output_obj, str) and isinstance(op.result, str):
                                    if output_obj == op.result:
                                        if node not in final_output_nodes:
                                            final_output_nodes.append(node)
                                            if args.verbose:
                                                print(f"  Matched {var_name} (string) to operation {op.op_name} (op_id={op.op_id})")
                                        found = True
                                        break
                                # For bool types, check boolean equality
                                elif isinstance(output_obj, bool) and isinstance(op.result, bool):
                                    if output_obj == op.result:
                                        if node not in final_output_nodes:
                                            final_output_nodes.append(node)
                                            if args.verbose:
                                                print(f"  Matched {var_name} (bool) to operation {op.op_name} (op_id={op.op_id})")
                                        found = True
                                        break
                        except:
                            pass
                    
                    # If object identity didn't match, try comparing pandas objects by content
                    if not found:
                        try:
                            import pandas as pd
                            if isinstance(output_obj, (pd.DataFrame, pd.Series)):
                                for node in reversed(graph.nodes):
                                    op = node.operation
                                    try:
                                        if isinstance(op.result, type(output_obj)):
                                            # Compare by shape/name for Series
                                            if isinstance(output_obj, pd.Series):
                                                if len(op.result) == len(output_obj):
                                                    # Match by name if available, or just length
                                                    result_name = getattr(op.result, 'name', None)
                                                    obj_name = getattr(output_obj, 'name', None)
                                                    if result_name == obj_name:
                                                        if node not in final_output_nodes:
                                                            final_output_nodes.append(node)
                                                            if args.verbose:
                                                                print(f"  Matched {var_name} (Series) to operation {op.op_name} (op_id={op.op_id})")
                                                        found = True
                                                        break
                                            # Compare by shape/columns for DataFrame
                                            elif isinstance(output_obj, pd.DataFrame):
                                                # For DataFrame, check shape and column match
                                                if op.result.shape == output_obj.shape:
                                                    op_cols = list(op.result.columns) if hasattr(op.result, 'columns') else []
                                                    obj_cols = list(output_obj.columns) if hasattr(output_obj, 'columns') else []
                                                    if op_cols == obj_cols:
                                                        if node not in final_output_nodes:
                                                            final_output_nodes.append(node)
                                                            if args.verbose:
                                                                print(f"  Matched {var_name} (DataFrame) to operation {op.op_name} (op_id={op.op_id})")
                                                        found = True
                                                        break
                                    except:
                                        pass
                        except ImportError:
                            pass
                    
                    # Last resort: if still not found and it's a DataFrame, find the most recent DataFrame operation
                    if not found:
                        try:
                            import pandas as pd
                            if isinstance(output_obj, pd.DataFrame):
                                # Find the most recent DataFrame operation
                                for node in reversed(graph.nodes):
                                    op = node.operation
                                    try:
                                        if isinstance(op.result, pd.DataFrame):
                                            # Use the most recent DataFrame operation as the output
                                            if node not in final_output_nodes:
                                                final_output_nodes.append(node)
                                                if args.verbose:
                                                    print(f"  Matched {var_name} (DataFrame) to most recent DataFrame operation {op.op_name} (op_id={op.op_id})")
                                            found = True
                                            break
                                    except:
                                        pass
                        except ImportError:
                            pass
                    
                    # If still not found, use the most recent matching type operation
                    if not found:
                        # Try to match by type and use most recent operation of that type
                        output_type = type(output_obj).__name__
                        for node in reversed(graph.nodes):
                            op = node.operation
                            try:
                                if hasattr(op, 'result'):
                                    op_type = type(op.result).__name__
                                    if op_type == output_type:
                                        # Check if numeric types match
                                        if isinstance(output_obj, (int, float)) and isinstance(op.result, (int, float, bool)):
                                            if node not in final_output_nodes:
                                                final_output_nodes.append(node)
                                                if args.verbose:
                                                    print(f"  Matched {var_name} (type-based, {output_type}) to operation {op.op_name} (op_id={op.op_id})")
                                                found = True
                                                break
                                        # Check if string types match
                                        elif isinstance(output_obj, str) and isinstance(op.result, str):
                                            if node not in final_output_nodes:
                                                final_output_nodes.append(node)
                                                if args.verbose:
                                                    print(f"  Matched {var_name} (type-based, string) to operation {op.op_name} (op_id={op.op_id})")
                                                found = True
                                                break
                                        # Check if bool types match
                                        elif isinstance(output_obj, bool):
                                            # Bool might come from int operations (0/1)
                                            if isinstance(op.result, (bool, int)) and bool(op.result) == output_obj:
                                                if node not in final_output_nodes:
                                                    final_output_nodes.append(node)
                                                    if args.verbose:
                                                        print(f"  Matched {var_name} (type-based, bool) to operation {op.op_name} (op_id={op.op_id})")
                                                    found = True
                                                    break
                            except:
                                pass
                    
                    if not found and args.verbose:
                        print(f"  Warning: Could not match {var_name} to any operation")
                        # Last resort: use the most recent operation as output
                        if graph.nodes:
                            last_node = graph.nodes[-1]
                            if last_node not in final_output_nodes:
                                final_output_nodes.append(last_node)
                                if args.verbose:
                                    print(f"  Using last operation {last_node.operation.op_name} (op_id={last_node.operation.op_id}) as fallback")
                
                # Check if we have batch string processing - if so, use all leaf nodes
                has_batch_processing = False
                if example_inputs:
                    for var_name, var_value in example_inputs.items():
                        if isinstance(var_value, (list, tuple)) and len(var_value) > 0 and isinstance(var_value[0], str):
                            has_batch_processing = True
                            break
                
                # For batch processing, use all leaf nodes as outputs
                # They'll be collected into batch arrays in the C code
                if has_batch_processing:
                    leaf_nodes = [n for n in graph.nodes if len(n.outputs) == 0]
                    if leaf_nodes:
                        graph.output_nodes = leaf_nodes
                        if args.verbose:
                            print(f"Batch processing: Using {len(leaf_nodes)} leaf nodes as outputs")
                # For non-batch processing, only keep final output nodes (filter out intermediate operations)
                elif final_output_nodes:
                    # Update output_nodes to only final outputs
                    graph.output_nodes = final_output_nodes
                    if args.verbose:
                        original_output_count = len([n for n in graph.nodes if len(n.outputs) == 0])
                        print(f"Filtered to {len(final_output_nodes)} final output nodes (from {original_output_count} leaf nodes)")
        
        # Generate C code
        if args.verbose:
            print("Generating C code...")
        input_vars = list(example_inputs.keys())
        
        # Check if we have pandas or string operations
        has_extended_ops = False
        has_string_ops = False
        if EXTENDED_OPS_AVAILABLE:
            if PandasOperation is not None:
                for op in operations:
                    if isinstance(op, PandasOperation):
                        has_extended_ops = True
                        if args.verbose:
                            print(f"Found {type(op).__name__}: {op.op_name}")
                        break
            # Check for string operations
            try:
                from .tracers.string_tracer import StringOperation as StrOp
                for op in operations:
                    if isinstance(op, StrOp):
                        has_string_ops = True
                        has_extended_ops = True  # String ops use extended codegen too
                        if args.verbose:
                            print(f"Found {type(op).__name__}: {op.op_name}")
                        break
            except ImportError as e:
                if args.verbose:
                    print(f"Could not import StringOperation: {e}")
                pass
        
        # Use extended code generator if we have pandas or string operations
        if has_extended_ops:
            if args.verbose:
                if has_string_ops:
                    print("Detected string operations - using extended code generator")
                elif has_extended_ops:
                    print("Detected pandas operations - using extended code generator")
            c_code = generate_extended_c_code(graph, input_vars, input_arrays=example_inputs)
            if args.verbose:
                print(f"Generated C code: {len(c_code.split(chr(10)))} lines")
        else:
            c_code = generate_c_code(graph, input_vars, input_arrays=example_inputs)
            if args.verbose:
                print(f"Generated C code (base): {len(c_code.split(chr(10)))} lines")
        
        # Save C code if requested
        if args.save_c:
            with open(args.save_c, 'w') as f:
                f.write(c_code)
            if args.verbose:
                print(f"C code saved to: {args.save_c}")
        
        # Compile to binary
        if args.verbose:
            print(f"Compiling with clang {args.optimization}...")
        
        # Determine actual output path (compiler adds extension)
        output_path_obj = Path(args.output)
        if not output_path_obj.suffix:
            if sys.platform == "darwin":
                actual_output = str(output_path_obj) + ".dylib"
            elif sys.platform.startswith("linux"):
                actual_output = str(output_path_obj) + ".so"
            elif sys.platform == "win32":
                actual_output = str(output_path_obj) + ".dll"
            else:
                actual_output = args.output
        else:
            actual_output = args.output
        
        success, error_msg = compile_to_binary(c_code, args.output, args.optimization)
        
        if success:
            print(f"✓ Successfully compiled binary: {actual_output}")
            # Make binary executable/readable
            if os.path.exists(actual_output):
                os.chmod(actual_output, 0o755)
            
            # Save metadata for binary execution
            metadata_path = str(output_path_obj) + "_metadata.json"
            metadata = {
                "input_vars": input_vars,
                "num_outputs": len(graph.output_nodes),
                "output_names": [f"output_{i}" for i in range(len(graph.output_nodes))],
                "output_types": [],
                "output_lengths": [],
                "output_shapes": []
            }
            
            # Determine output types and shapes
            output_shapes_list = []
            output_columns_list = []
            
            # Check if we have batch string processing
            has_batch_processing = False
            batch_input_var = None
            num_strings = None
            if example_inputs:
                for var_name, var_value in example_inputs.items():
                    if isinstance(var_value, (list, tuple)) and len(var_value) > 0 and isinstance(var_value[0], str):
                        has_batch_processing = True
                        batch_input_var = var_name
                        num_strings = len(var_value)
                        break
            
            for i, node in enumerate(graph.output_nodes):
                op = node.operation
                result = op.result
                
                # For batch processing, all outputs should be arrays (one per string)
                # The C code stores results in batch arrays, so outputs are arrays
                if has_batch_processing:
                    metadata["output_types"].append("array")
                    metadata["output_lengths"].append(num_strings)  # Array length = num_strings
                    output_shapes_list.append([num_strings])  # 1D array shape
                # For batch processing, if result is a list/tuple, it should be an array
                elif isinstance(result, (list, tuple)):
                    metadata["output_types"].append("array")
                    metadata["output_lengths"].append(len(result))  # Array length
                    output_shapes_list.append([len(result)])  # 1D array shape
                elif isinstance(result, np.ndarray):
                    metadata["output_types"].append("array")
                    shape = list(result.shape) if result.shape else []
                    metadata["output_lengths"].append(int(np.prod(shape)) if shape else 1)
                    output_shapes_list.append(shape)
                elif PANDAS_AVAILABLE and hasattr(pd, 'DataFrame') and isinstance(result, pd.DataFrame):
                    metadata["output_types"].append("dataframe")
                    shape = list(result.shape)
                    # For DataFrame outputs, extract numeric columns only
                    # The C code will extract numeric data row-major
                    numeric_df = result.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        # Calculate length as num_rows * num_numeric_cols
                        num_numeric_cols = len(numeric_df.columns)
                        output_length = int(shape[0] * num_numeric_cols) if shape and shape[0] > 0 else 1
                        metadata["output_lengths"].append(output_length)
                        # Store both full shape and numeric columns info
                        output_shapes_list.append(shape)
                        output_columns_list.append(list(numeric_df.columns) if hasattr(result, 'columns') else [])
                    else:
                        # No numeric columns - empty output
                        metadata["output_lengths"].append(1)
                        output_shapes_list.append([1])
                        output_columns_list.append([])
                elif PANDAS_AVAILABLE and hasattr(pd, 'Series') and isinstance(result, pd.Series):
                    metadata["output_types"].append("series")
                    length = len(result)
                    metadata["output_lengths"].append(length)
                    output_shapes_list.append([length])
                    if i == 0:  # Store name only for first series output
                        metadata["output_name"] = result.name if hasattr(result, 'name') else None
                else:
                    metadata["output_types"].append("scalar")
                    metadata["output_lengths"].append(1)
                    output_shapes_list.append([])
            
            metadata["output_shapes"] = output_shapes_list
            if output_columns_list:
                metadata["output_columns"] = output_columns_list if len(output_columns_list) == 1 else output_columns_list
            
            # Save metadata
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            if args.verbose:
                print(f"Binary is ready for deployment!")
                print(f"Metadata saved to: {metadata_path}")
        else:
            print(f"✗ Compilation failed: {error_msg}")
            if args.save_c:
                print(f"C code saved to {args.save_c} for debugging")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

