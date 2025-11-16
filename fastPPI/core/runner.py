"""
Runtime execution module for running FastPPI compiled binaries.
Provides Python interface for calling compiled C code.
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import json

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


class BinaryRunner:
    """Runner for executing compiled FastPPI binaries."""
    
    def __init__(self, binary_path: str, metadata_path: Optional[str] = None):
        """
        Initialize binary runner.
        
        Args:
            binary_path: Path to compiled binary (.dylib, .so, or .dll)
            metadata_path: Optional path to metadata JSON file containing
                          input/output variable information.
                          If None, tries to auto-detect metadata file by appending "_metadata.json"
        """
        self.binary_path = self._resolve_binary_path(binary_path)
        
        # Auto-detect metadata file if not provided
        if metadata_path is None:
            # Metadata file is named: {binary_name}_metadata.json
            # e.g., string_ops_test_binary.dylib -> string_ops_test_binary.dylib_metadata.json
            metadata_path = str(self.binary_path) + "_metadata.json"
            if not Path(metadata_path).exists():
                # Try without extension as fallback
                metadata_path = str(Path(self.binary_path).with_suffix('')) + "_metadata.json"
                if not Path(metadata_path).exists():
                    metadata_path = None
        
        self.metadata = self._load_metadata(metadata_path) if metadata_path else {}
        self.lib = None
        self._current_output_index = 0
        self._load_library()
    
    def _resolve_binary_path(self, binary_path: str) -> str:
        """Resolve binary path with proper extension."""
        path_obj = Path(binary_path)
        
        # If file exists, return as-is
        if path_obj.exists():
            return str(path_obj)
        
        # Try adding extensions
        if sys.platform == "darwin":
            extensions = [".dylib"]
        elif sys.platform.startswith("linux"):
            extensions = [".so"]
        elif sys.platform == "win32":
            extensions = [".dll"]
        else:
            extensions = [".dylib", ".so", ".dll"]
        
        for ext in extensions:
            candidate = str(path_obj) + ext
            if Path(candidate).exists():
                return candidate
        
        raise FileNotFoundError(f"Binary not found: {binary_path}")
    
    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Load metadata JSON file."""
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            return {}
    
    def _load_library(self):
        """Load the compiled binary library."""
        # Set library path for dynamic linking
        # Try multiple possible locations for the library
        possible_lib_dirs = [
            Path(self.binary_path).parent / "lib",  # Same directory as binary
            Path(self.binary_path).parent.parent / "c_implementations" / "lib",  # Relative to examples
            Path(__file__).parent.parent.parent / "c_implementations" / "lib",  # Relative to fastPPI root
        ]
        
        lib_dir = None
        for possible_dir in possible_lib_dirs:
            if Path(possible_dir).exists():
                lib_dir = str(possible_dir)
                break
        
        if lib_dir:
            if sys.platform == "darwin":
                # Prepend to existing DYLD_LIBRARY_PATH if it exists
                current_path = os.environ.get("DYLD_LIBRARY_PATH", "")
                if current_path:
                    os.environ["DYLD_LIBRARY_PATH"] = f"{lib_dir}:{current_path}"
                else:
                    os.environ["DYLD_LIBRARY_PATH"] = lib_dir
            elif sys.platform.startswith("linux"):
                # Prepend to existing LD_LIBRARY_PATH if it exists
                current_path = os.environ.get("LD_LIBRARY_PATH", "")
                if current_path:
                    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current_path}"
                else:
                    os.environ["LD_LIBRARY_PATH"] = lib_dir
        
        # Load the library
        self.lib = ctypes.CDLL(self.binary_path)
        
        # Define function signature
        self.lib.preprocess.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # inputs
            ctypes.c_int,  # num_inputs
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # outputs
            ctypes.c_int   # num_outputs
        ]
        self.lib.preprocess.restype = None
    
    def _convert_input_to_c(self, value: Any, var_name: str) -> Tuple[ctypes.POINTER(ctypes.c_double), int]:
        """
        Convert Python input to C-compatible format.
        
        Returns:
            (ctypes pointer, length)
        """
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            if value.dtype != np.float64:
                value = value.astype(np.float64)
            flat = value.flatten()
            array = (ctypes.c_double * len(flat))(*flat)
            return ctypes.cast(array, ctypes.POINTER(ctypes.c_double)), len(flat)
        
        # Handle scalars (int, float)
        elif isinstance(value, (int, float)):
            array = (ctypes.c_double * 1)(float(value))
            return ctypes.cast(array, ctypes.POINTER(ctypes.c_double)), 1
        
        # Handle strings - convert to numeric array (character encoding)
        elif isinstance(value, str):
            # Encode string as array of character codes
            # Limit to 255 characters to fit in single bytes (C char is 0-255)
            # Add null terminator (0) at the end
            max_len = min(len(value), 255)
            encoded = np.array([ord(c) for c in value[:max_len]], dtype=np.float64)
            # Ensure null terminator
            if len(encoded) < 256:
                encoded = np.append(encoded, 0.0)  # Null terminator
            array = (ctypes.c_double * len(encoded))(*encoded)
            return ctypes.cast(array, ctypes.POINTER(ctypes.c_double)), len(encoded)
        
        # Handle lists/tuples
        elif isinstance(value, (list, tuple)):
            # Check if it's a list of strings (for string batch processing)
            if len(value) > 0 and isinstance(value[0], str):
                # Encode list of strings: format is [num_strings, str1_len, str1_chars..., str2_len, str2_chars..., ...]
                # This makes it easy to parse in C
                encoded = [float(len(value))]  # First element is number of strings
                max_str_len = 10000  # Maximum string length per string
                for s in value:
                    # Encode string as character codes
                    str_chars = [ord(c) for c in s[:max_str_len]]
                    encoded.append(float(len(str_chars)))  # Length of this string
                    encoded.extend([float(c) for c in str_chars])  # Character codes
                arr = np.array(encoded, dtype=np.float64)
                array = (ctypes.c_double * len(arr))(*arr)
                return ctypes.cast(array, ctypes.POINTER(ctypes.c_double)), len(arr)
            else:
                # Regular numeric list
                arr = np.array(value, dtype=np.float64).flatten()
                array = (ctypes.c_double * len(arr))(*arr)
                return ctypes.cast(array, ctypes.POINTER(ctypes.c_double)), len(arr)
        
        # Handle pandas Series - extract numeric values
        elif PANDAS_AVAILABLE and isinstance(value, pd.Series):
            if value.dtype in [np.float64, np.float32, np.int64, np.int32]:
                arr = value.values.astype(np.float64)
                array = (ctypes.c_double * len(arr))(*arr)
                return ctypes.cast(array, ctypes.POINTER(ctypes.c_double)), len(arr)
            else:
                raise TypeError(f"Unsupported Series dtype for {var_name}: {value.dtype}")
        
        # Handle pandas DataFrame - extract as flat array
        elif PANDAS_AVAILABLE and isinstance(value, pd.DataFrame):
            # Flatten DataFrame to single array
            numeric_df = value.select_dtypes(include=[np.number])
            if numeric_df.empty:
                raise ValueError(f"DataFrame {var_name} has no numeric columns")
            arr = numeric_df.values.flatten().astype(np.float64)
            array = (ctypes.c_double * len(arr))(*arr)
            return ctypes.cast(array, ctypes.POINTER(ctypes.c_double)), len(arr)
        
        else:
            raise TypeError(f"Unsupported input type for {var_name}: {type(value)}")
    
    def _convert_output_from_c(self, output_ptr: ctypes.POINTER(ctypes.c_double), 
                               length: int, expected_type: Optional[str] = None) -> Any:
        """
        Convert C output back to Python format.
        
        Args:
            output_ptr: Pointer to C double array
            length: Length of array
            expected_type: Expected Python type (optional, from metadata)
        
        Returns:
            Python value
        """
        if output_ptr is None or length == 0:
            return None
        
        # Extract array
        array = np.array([output_ptr[i] for i in range(length)], dtype=np.float64)
        
        # Convert based on expected type
        if expected_type == "scalar" or length == 1:
            return float(array[0])
        
        elif expected_type == "array":
            return array
        
        elif expected_type == "dataframe" and PANDAS_AVAILABLE:
            # Try to reconstruct DataFrame from metadata
            output_shapes = self.metadata.get("output_shapes", [])
            output_idx = getattr(self, '_current_output_index', 0)
            if output_shapes and output_idx < len(output_shapes):
                shape = tuple(output_shapes[output_idx])
                if len(shape) == 2:
                    # Get numeric columns info
                    output_columns = self.metadata.get("output_columns", [])
                    if output_columns:
                        if output_idx < len(output_columns):
                            if isinstance(output_columns[output_idx], list):
                                numeric_columns = output_columns[output_idx]
                            elif isinstance(output_columns[0], list) and len(output_columns) == 1:
                                numeric_columns = output_columns[0]
                            elif isinstance(output_columns[0], list):
                                numeric_columns = output_columns[0]  # Fallback to first
                            else:
                                numeric_columns = output_columns
                        else:
                            # Use first entry if available
                            if output_columns and isinstance(output_columns[0], list):
                                numeric_columns = output_columns[0]
                            else:
                                numeric_columns = output_columns
                    else:
                        # No column info - use default names
                        # The output length tells us num_numeric_cols = length / num_rows
                        num_rows = shape[0]
                        num_numeric_cols = length // num_rows if num_rows > 0 else 1
                        numeric_columns = [f"col_{i}" for i in range(num_numeric_cols)]
                    
                    # Reshape the flat array into (num_rows, num_numeric_cols)
                    num_rows = shape[0]
                    num_numeric_cols = len(numeric_columns) if numeric_columns else (length // num_rows if num_rows > 0 else 1)
                    
                    if num_rows > 0 and num_numeric_cols > 0 and length == num_rows * num_numeric_cols:
                        arr_2d = array.reshape((num_rows, num_numeric_cols))
                        return pd.DataFrame(arr_2d, columns=numeric_columns)
                    else:
                        # Fallback: return as flat array
                        return array
            return array  # Fallback to array
        
        elif expected_type == "series" and PANDAS_AVAILABLE:
            return pd.Series(array, name=self.metadata.get("output_name", "series"))
        
        else:
            # Default: return as numpy array
            return array
    
    def run(self, **kwargs) -> Union[Any, Dict[str, Any]]:
        """
        Run the compiled binary with given inputs.
        
        Args:
            **kwargs: Input variables as keyword arguments
            
        Returns:
            Output value(s). If multiple outputs, returns a dictionary.
            If single output, returns the value directly.
            DataFrames/Series are converted to dictionaries if possible.
        """
        # Get input variable names and values
        input_names = list(kwargs.keys())
        input_values = list(kwargs.values())
        
        # Convert inputs to C format
        c_inputs = []
        input_lengths = []
        for var_name, value in zip(input_names, input_values):
            ptr, length = self._convert_input_to_c(value, var_name)
            c_inputs.append(ptr)
            input_lengths.append(length)
        
        # Create inputs array
        inputs_array = (ctypes.POINTER(ctypes.c_double) * len(c_inputs))(*c_inputs)
        
        # Determine number of outputs from metadata or default to 1
        num_outputs = self.metadata.get("num_outputs", 1)
        
        # Prepare outputs array
        outputs_array = (ctypes.POINTER(ctypes.c_double) * num_outputs)()
        
        # Call the C function
        self.lib.preprocess(inputs_array, len(c_inputs), outputs_array, num_outputs)
        
        # Convert outputs back to Python
        output_types = self.metadata.get("output_types", ["scalar"] * num_outputs)
        output_names = self.metadata.get("output_names", [f"output_{i}" for i in range(num_outputs)])
        output_lengths = self.metadata.get("output_lengths", [1] * num_outputs)
        
        results = []
        for i in range(num_outputs):
            output_ptr = outputs_array[i]
            if output_ptr is None:
                # NULL pointer - try to handle gracefully
                print(f"Warning: Output {i} is NULL, skipping")
                results.append(None)
                continue
            
            length = output_lengths[i] if i < len(output_lengths) else 1
            expected_type = output_types[i] if i < len(output_types) else None
            
            # Store current output index for metadata access
            self._current_output_index = i
            
            result = self._convert_output_from_c(output_ptr, length, expected_type)
            results.append(result)
        
        # Reset output index
        self._current_output_index = 0
        
        # Skip dict conversion for performance (can be re-enabled if needed)
        # The DataFrame/Series objects are already reconstructed from metadata
        # Dict conversion is expensive and only needed for JSON serialization
        
        # Return single value or dictionary
        if num_outputs == 1:
            return results[0]
        else:
            return {name: val for name, val in zip(output_names, results)}


def run_binary(binary_path: str, metadata_path: Optional[str] = None, **kwargs) -> Union[Any, Dict[str, Any]]:
    """
    Convenience function to run a compiled binary.
    
    Args:
        binary_path: Path to compiled binary
        metadata_path: Optional path to metadata JSON file
        **kwargs: Input variables as keyword arguments
        
    Returns:
        Output value(s) from the binary
        
    Example:
        ```python
        output = run_binary(
            "preprocess_binary.dylib",
            input_data=[1.0, 2.0, 3.0],
            scale_factor=2.0
        )
        ```
    """
    runner = BinaryRunner(binary_path, metadata_path)
    return runner.run(**kwargs)

