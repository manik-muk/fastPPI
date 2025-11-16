"""
Analysis tool for pandas preprocessing code.
Identifies which operations can be compiled and generates recommendations.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from .unified_tracer import trace_unified_execution, generate_operation_summary, categorize_operations
from .extended_codegen import generate_operation_report


def load_python_file(filepath: str) -> str:
    """Load Python code from file."""
    with open(filepath, 'r') as f:
        return f.read()


def analyze_preprocessing_code(python_file: str, example_inputs: Dict[str, Any] = None):
    """
    Analyze preprocessing code and report on compilability.
    
    Args:
        python_file: Path to Python file
        example_inputs: Optional dictionary of example inputs
    """
    print("=" * 70)
    print("FastPPI: Preprocessing Code Analysis")
    print("=" * 70)
    print()
    
    # Load code
    code = load_python_file(python_file)
    print(f"Analyzing: {python_file}")
    print()
    
    # If no example inputs, try to create dummy inputs
    if example_inputs is None:
        example_inputs = {}
        print("Note: No example inputs provided. Using empty context.")
        print("For better analysis, provide example inputs.")
        print()
    
    # Trace execution
    print("Tracing execution...")
    try:
        operations = trace_unified_execution(code, example_inputs)
        print(f"Captured {len(operations)} operations")
        print()
    except Exception as e:
        print(f"Error during tracing: {e}")
        print()
        print("This may be because:")
        print("  1. The code requires inputs that weren't provided")
        print("  2. The code has dependencies that aren't installed")
        print("  3. The code uses features not yet supported by FastPPI")
        return 1
    
    # Generate analysis
    print(generate_operation_summary(operations))
    print()
    print(generate_operation_report(operations))
    print()
    
    # Categorize operations
    categorized = categorize_operations(operations)
    
    # Provide recommendations
    print("=" * 70)
    print("Recommendations")
    print("=" * 70)
    print()
    
    if len(categorized['compilable']) == len(operations):
        print("✓ All operations can be compiled to C!")
        print("  Run: fastppi <file> --inputs <inputs> --output <output>")
    elif len(categorized['compilable']) > 0:
        pct = 100 * len(categorized['compilable']) / len(operations)
        print(f"⚠ {pct:.1f}% of operations can be compiled to C")
        print()
        print("Options:")
        print("  1. Compile only the NumPy operations (current FastPPI support)")
        print("  2. Refactor code to use more NumPy operations")
        print("  3. Use hybrid approach (C for numeric ops, Python for others)")
    else:
        print("✗ No operations can be compiled (all are pandas specific)")
        print()
        print("To use FastPPI, consider:")
        print("  1. Extracting numeric operations to separate functions")
        print("  2. Rewriting with explicit NumPy operations")
        print("  3. Using FastPPI for the numeric preprocessing parts only")
    
    print()
    return 0


def main():
    """CLI entry point for analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze preprocessing code for FastPPI compilation"
    )
    parser.add_argument(
        "python_file",
        type=str,
        help="Path to Python file containing preprocessing code"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed analysis"
    )
    
    args = parser.parse_args()
    
    try:
        return analyze_preprocessing_code(args.python_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

