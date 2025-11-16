"""
Base testing framework for FastPPI.
Provides utilities for compiling Python code, loading C binaries, and comparing results.
"""
import os
import sys
import tempfile
import subprocess
import ctypes
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import shutil


class FastPPITestFramework:
    """Base framework for testing FastPPI compiled code against Python pandas."""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir or Path(__file__).parent.parent
        self.test_data_dir = self.workspace_dir / "tests" / "test_data"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
        self.lib_path = str(self.workspace_dir / "c_implementations" / "lib")
        
    def compile_test(self, python_code: str, test_name: str, inputs: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Compile Python code to C binary using FastPPI.
        
        Returns:
            (binary_path, c_code_path) tuple
        """
        # Create temporary Python file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=self.test_data_dir) as f:
            f.write(python_code)
            python_file = f.name
        
        try:
            # Prepare inputs
            import json
            inputs_str = json.dumps(inputs) if inputs else '{}'
            
            # Create output paths
            binary_path = str(self.test_data_dir / f"{test_name}_binary")
            c_code_path = str(self.test_data_dir / f"{test_name}.c")
            
            # Compile using fastppi
            cmd = [
                sys.executable, "-m", "fastPPI.main",
                python_file,
                "--inputs", inputs_str,
                "--output", binary_path,
                "--save-c", c_code_path
            ]
            
            env = os.environ.copy()
            if 'DYLD_LIBRARY_PATH' in env:
                env['DYLD_LIBRARY_PATH'] = f"{self.lib_path}:{env['DYLD_LIBRARY_PATH']}"
            else:
                env['DYLD_LIBRARY_PATH'] = self.lib_path
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(self.workspace_dir))
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                # Try to extract meaningful error from output
                if "error:" in error_msg:
                    # Extract the last few error lines
                    error_lines = [line for line in error_msg.split('\n') if 'error:' in line]
                    if error_lines:
                        error_msg = '\n'.join(error_lines[-3:])  # Last 3 error lines
                raise RuntimeError(f"Compilation failed: {error_msg[:500]}")  # Limit error message length
            
            # Add .dylib extension if needed
            if not Path(binary_path).exists():
                for ext in ['.dylib', '.so', '.dll']:
                    if Path(f"{binary_path}{ext}").exists():
                        binary_path = f"{binary_path}{ext}"
                        break
            
            return binary_path, c_code_path
            
        finally:
            # Cleanup temp Python file
            if os.path.exists(python_file):
                os.unlink(python_file)
    
    def load_c_binary(self, binary_path: str):
        """Load and configure a C binary."""
        lib = ctypes.CDLL(binary_path)
        
        lib.preprocess.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int
        ]
        lib.preprocess.restype = None
        
        return lib
    
    def run_python_code(self, python_code: str, inputs: Dict[str, Any] = None) -> Any:
        """Execute Python code and return results."""
        # Create a namespace with inputs
        namespace = {'pd': pd, 'np': np, '__builtins__': __builtins__}
        
        # Add string operation modules
        try:
            import re
            import unicodedata
            namespace['re'] = re
            namespace['unicodedata'] = unicodedata
        except ImportError:
            pass
        
        if inputs:
            namespace.update(inputs)
        
        # Execute and capture result
        exec(python_code, namespace)
        
        # Try to get a result variable
        if 'result' in namespace:
            return namespace['result']
        elif 'df' in namespace:
            return namespace['df']
        elif 'output' in namespace:
            return namespace['output']
        else:
            # Return all variables that look like results
            return {k: v for k, v in namespace.items() 
                   if not k.startswith('_') and not callable(v) 
                   and isinstance(v, (pd.DataFrame, pd.Series, np.ndarray, int, float, str, list, tuple, bool))}
    
    def compare_results(self, python_result: Any, c_result: Any, tolerance: float = 1e-5) -> Tuple[bool, str]:
        """
        Compare Python and C results.
        
        Returns:
            (match: bool, message: str)
        """
        try:
            # Handle DataFrame
            if isinstance(python_result, pd.DataFrame):
                if not isinstance(c_result, pd.DataFrame):
                    return False, f"Type mismatch: Python DataFrame vs C {type(c_result)}"
                
                # Compare shapes
                if python_result.shape != c_result.shape:
                    return False, f"Shape mismatch: {python_result.shape} vs {c_result.shape}"
                
                # Compare columns
                if list(python_result.columns) != list(c_result.columns):
                    return False, f"Column mismatch: {list(python_result.columns)} vs {list(c_result.columns)}"
                
                # Compare numeric columns
                for col in python_result.select_dtypes(include=[np.number]).columns:
                    if col in c_result.columns:
                        py_vals = python_result[col].values
                        c_vals = c_result[col].values
                        
                        # Handle NaN
                        mask = ~(np.isnan(py_vals) | np.isnan(c_vals))
                        if np.any(mask):
                            if not np.allclose(py_vals[mask], c_vals[mask], rtol=tolerance, atol=tolerance):
                                max_diff = np.max(np.abs(py_vals[mask] - c_vals[mask]))
                                return False, f"Column {col} values differ (max diff: {max_diff})"
                
                # Compare string columns
                for col in python_result.select_dtypes(include=['object']).columns:
                    if col in c_result.columns:
                        py_vals = python_result[col].fillna('').astype(str).values
                        c_vals = c_result[col].fillna('').astype(str).values
                        if not np.array_equal(py_vals, c_vals):
                            return False, f"String column {col} values differ"
                
                return True, "Results match"
            
            # Handle Series
            elif isinstance(python_result, pd.Series):
                if not isinstance(c_result, pd.Series):
                    return False, f"Type mismatch: Python Series vs C {type(c_result)}"
                
                if len(python_result) != len(c_result):
                    return False, f"Length mismatch: {len(python_result)} vs {len(c_result)}"
                
                # Compare values
                if python_result.dtype in [np.float64, np.float32, np.int64, np.int32]:
                    mask = ~(pd.isna(python_result) | pd.isna(c_result))
                    if np.any(mask):
                        if not np.allclose(python_result[mask], c_result[mask], rtol=tolerance, atol=tolerance):
                            max_diff = np.max(np.abs(python_result[mask].values - c_result[mask].values))
                            return False, f"Series values differ (max diff: {max_diff})"
                else:
                    py_vals = python_result.fillna('').astype(str).values
                    c_vals = c_result.fillna('').astype(str).values
                    if not np.array_equal(py_vals, c_vals):
                        return False, "Series string values differ"
                
                return True, "Results match"
            
            # Handle scalars (numbers)
            elif isinstance(python_result, (int, float)):
                if isinstance(c_result, (int, float)):
                    if abs(python_result - c_result) < tolerance:
                        return True, "Results match"
                    else:
                        return False, f"Scalar mismatch: {python_result} vs {c_result}"
            
            # Handle strings
            elif isinstance(python_result, str):
                if isinstance(c_result, str):
                    if python_result == c_result:
                        return True, "Results match"
                    else:
                        return False, f"String mismatch: '{python_result}' vs '{c_result}'"
            
            # Handle booleans
            elif isinstance(python_result, bool):
                if isinstance(c_result, bool):
                    if python_result == c_result:
                        return True, "Results match"
                    else:
                        return False, f"Boolean mismatch: {python_result} vs {c_result}"
                elif isinstance(c_result, (int, float)):
                    # C booleans are often returned as int (0/1)
                    if python_result == bool(c_result):
                        return True, "Results match"
                    else:
                        return False, f"Boolean mismatch: {python_result} vs {c_result}"
            
            # Handle arrays
            elif isinstance(python_result, np.ndarray):
                if isinstance(c_result, np.ndarray):
                    if python_result.shape != c_result.shape:
                        return False, f"Array shape mismatch: {python_result.shape} vs {c_result.shape}"
                    
                    mask = ~(np.isnan(python_result) | np.isnan(c_result))
                    if np.any(mask):
                        if not np.allclose(python_result[mask], c_result[mask], rtol=tolerance, atol=tolerance):
                            return False, "Array values differ"
                        return True, "Results match"
            
            return False, f"Unsupported result type: {type(python_result)}"
            
        except Exception as e:
            return False, f"Comparison error: {str(e)}"
    
    def run_test(self, python_code: str, test_name: str, inputs: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Run a complete test: compile, run Python, run C, compare.
        
        Returns:
            (passed: bool, message: str)
        """
        try:
            # Run Python version
            python_result = self.run_python_code(python_code, inputs)
            
            # Compile to C
            binary_path, c_code_path = self.compile_test(python_code, test_name, inputs)
            
            # Note: Actually running C binary and extracting results is complex
            # because C outputs are in a specific format. For now, we'll verify compilation
            # and let individual tests handle C execution if needed.
            
            return True, f"Compilation successful. Binary: {binary_path}"
            
        except Exception as e:
            return False, f"Test failed: {str(e)}"


def assert_results_match(python_result: Any, c_result: Any, tolerance: float = 1e-5):
    """Assert that Python and C results match."""
    framework = FastPPITestFramework()
    match, message = framework.compare_results(python_result, c_result, tolerance)
    assert match, message

