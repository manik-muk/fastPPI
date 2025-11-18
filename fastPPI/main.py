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

# Try to import pandas for extended operations
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
    
    # If code has if __name__ == "__main__" block, extract it for execution
    # This handles scripts that use argparse or have main blocks
    if 'if __name__ == "__main__":' in code and example_inputs:
        lines = code.split('\n')
        new_lines = []
        skip_main_block = False
        
        for i, line in enumerate(lines):
            if 'if __name__ == "__main__":' in line:
                skip_main_block = True
                # Look for functions that might be called in main
                # Try to find a function that matches the input variables
                import re
                func_matches = re.findall(r'def\s+(\w+)\s*\([^)]*\):', code)
                
                # Try to call the first function found with the inputs
                if func_matches and example_inputs:
                    func_name = func_matches[0]
                    # Build function call with input variables
                    input_args = ', '.join(example_inputs.keys())
                    new_lines.append(f'result = {func_name}({input_args})')
                continue
            elif skip_main_block:
                # Skip lines in the main block (argparse, etc.)
                if line.strip() and not line[0:len(line) - len(line.lstrip())]:
                    # Non-indented line means we're out of the block
                    if not (line.strip().startswith('#') or 'parse_args' in line or 'parser' in line):
                        skip_main_block = False
                continue
            else:
                new_lines.append(line)
        
        if skip_main_block:
            # Fallback: remove main block and try to call function directly
            code = code.split('if __name__ == "__main__":')[0]
            if func_matches and example_inputs:
                func_name = func_matches[0]
                input_args = ', '.join(example_inputs.keys())
                code += f'\n# Call function with example inputs\n'
                code += f'result = {func_name}({input_args})\n'
        else:
            code = '\n'.join(new_lines)
    
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
        
        # Use all leaf nodes (operations with no outputs) as the final outputs
        # This captures all final results from the traced operations
        leaf_nodes = [n for n in graph.nodes if len(n.outputs) == 0]
        
        # Perform dead code elimination
        if args.verbose:
            print("Performing dead code elimination...")
        removed_count = graph.eliminate_dead_code()
        if args.verbose and removed_count > 0:
            print(f"Removed {removed_count} unused operations")
        
        # Check if we have batch string processing
        has_batch_processing = False
        if example_inputs:
            for var_name, var_value in example_inputs.items():
                if isinstance(var_value, (list, tuple)) and len(var_value) > 0 and isinstance(var_value[0], str):
                    has_batch_processing = True
                    break
        
        # Set output nodes to all leaf nodes
        graph.output_nodes = leaf_nodes
        if args.verbose:
            print(f"Using {len(leaf_nodes)} leaf nodes as outputs (all final operations)")
        
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
            
            if args.verbose:
                print(f"Binary is ready for deployment!")
                print(f"Output binary: {actual_output}")
                print(f"Note: Execute the binary manually to use it. All leaf operations are included as outputs.")
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

