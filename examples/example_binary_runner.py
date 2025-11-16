"""
Example demonstrating how to use the binary runner to execute compiled FastPPI binaries.

This example shows:
1. Compiling a simple preprocessing script
2. Running the compiled binary with inputs
3. Getting the output as a Python variable
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fastPPI import run_binary

def example_simple_operation():
    """Example with simple numeric operation."""
    print("=" * 80)
    print("Example: Simple Numeric Operation")
    print("=" * 80)
    print()
    
    print("Step 1: Compile the binary (if not already done)")
    print("  fastppi examples/simple_preprocess.py --inputs 'input_data=[1.0,2.0,3.0,4.0,5.0]' --output simple_preprocess_binary")
    print()
    
    print("Step 2: Run the binary")
    print("  output = run_binary('simple_preprocess_binary.dylib', input_data=[1.0, 2.0, 3.0, 4.0, 5.0])")
    print()
    
    try:
        # Try to run the binary (if it exists)
        output = run_binary(
            "simple_preprocess_binary.dylib",
            input_data=np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        )
        print(f"✓ Output: {output}")
        print(f"  Type: {type(output)}")
        return True
    except FileNotFoundError as e:
        print(f"ℹ Binary not found: {e}")
        print("  This is expected if you haven't compiled the binary yet.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def example_with_metadata():
    """Example with metadata for better type handling."""
    print("=" * 80)
    print("Example: Using Metadata")
    print("=" * 80)
    print()
    
    print("When you compile a binary, FastPPI automatically generates")
    print("a metadata JSON file (e.g., 'binary_metadata.json') that")
    print("contains information about inputs and outputs.")
    print()
    
    try:
        output = run_binary(
            "simple_preprocess_binary.dylib",
            metadata_path="simple_preprocess_binary_metadata.json",  # Auto-detected if not provided
            input_data=np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        )
        print(f"✓ Output: {output}")
        return True
    except FileNotFoundError:
        print("ℹ Binary or metadata not found.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def example_multiple_inputs():
    """Example with multiple input variables."""
    print("=" * 80)
    print("Example: Multiple Inputs")
    print("=" * 80)
    print()
    
    print("You can pass multiple inputs as keyword arguments:")
    print("  output = run_binary('binary.dylib',")
    print("                      array1=[1.0, 2.0, 3.0],")
    print("                      array2=[4.0, 5.0, 6.0],")
    print("                      scale_factor=2.0)")
    print()
    
    try:
        output = run_binary(
            "matrix_ops_binary.dylib",
            input_array1=np.array([1.0, 2.0, 3.0]),
            input_array2=np.array([4.0, 5.0, 6.0]),
            scale_factor=2.0
        )
        print(f"✓ Output: {output}")
        return True
    except FileNotFoundError:
        print("ℹ Binary not found.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def example_dataframe_output():
    """Example with DataFrame output (converted to dictionary)."""
    print("=" * 80)
    print("Example: DataFrame Output")
    print("=" * 80)
    print()
    
    print("When a preprocessing script returns a DataFrame, the binary")
    print("runner will convert it to a dictionary (list of records):")
    print("  output = run_binary('feature_engineering_binary.dylib', csv_path='data.csv')")
    print("  # output will be a dict or list of dicts")
    print()
    
    try:
        output = run_binary(
            "feature_engineering_binary.dylib",
            csv_path="examples/example_data.csv"
        )
        print(f"✓ Output type: {type(output)}")
        if isinstance(output, dict):
            print(f"  Keys: {list(output.keys())}")
            for key, value in output.items():
                print(f"  {key}: {type(value).__name__}")
        elif isinstance(output, list):
            print(f"  Length: {len(output)}")
            if len(output) > 0:
                print(f"  First item type: {type(output[0])}")
        return True
    except FileNotFoundError:
        print("ℹ Binary not found.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all examples."""
    print("=" * 80)
    print("FastPPI Binary Runner Examples")
    print("=" * 80)
    print()
    
    examples = [
        ("Simple Operation", example_simple_operation),
        ("With Metadata", example_with_metadata),
        ("Multiple Inputs", example_multiple_inputs),
        ("DataFrame Output", example_dataframe_output),
    ]
    
    results = []
    for name, func in examples:
        try:
            result = func()
            results.append((name, result))
            print()
        except Exception as e:
            print(f"✗ Example '{name}' failed: {e}")
            results.append((name, False))
            print()
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    for name, result in results:
        status = "✓" if result else "ℹ (skipped - binary not found)"
        print(f"  {status} {name}")
    print()
    
    print("Note: To test these examples, first compile the binaries using fastppi:")
    print("  fastppi examples/simple_preprocess.py --inputs '...' --output simple_preprocess_binary")
    print()


if __name__ == "__main__":
    main()

