"""
Simple tests for NumPy operations - tests one by one.
"""
import sys
import os
import tempfile
import subprocess
import ctypes
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_operation(name, python_code, inputs_dict=None):
    """Test a single NumPy operation."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Create temp Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(python_code)
        temp_file = f.name
    
    try:
        # Compile
        inputs_str = "{}" if inputs_dict is None else str(inputs_dict).replace("'", '"')
        binary_path = f"tests/test_data/test_{name}_binary"
        c_path = f"tests/test_data/test_{name}.c"
        
        cmd = [
            sys.executable, "-m", "fastPPI.main",
            temp_file,
            "--inputs", inputs_str,
            "--output", binary_path,
            "--save-c", c_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
        
        if result.returncode != 0:
            print(f"❌ Compilation failed: {result.stderr[:500]}")
            return False
        
        # Find binary
        for ext in ['.dylib', '.so', '.dll']:
            if os.path.exists(f"{binary_path}{ext}"):
                binary_path = f"{binary_path}{ext}"
                break
        
        if not os.path.exists(binary_path):
            print(f"❌ Binary not found: {binary_path}")
            return False
        
        # Run Python version
        namespace = {'np': np, '__builtins__': __builtins__}
        if inputs_dict:
            namespace.update(inputs_dict)
        exec(python_code, namespace)
        python_result = namespace.get('output')
        if python_result is None:
            python_result = namespace.get('result')
        
        # Run C version
        lib = ctypes.CDLL(binary_path)
        lib.preprocess.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int
        ]
        lib.preprocess.restype = None
        
        # Prepare inputs
        c_inputs = None
        num_inputs = 0
        if inputs_dict:
            c_inputs_list = []
            for key, value in inputs_dict.items():
                if isinstance(value, np.ndarray):
                    # Flatten array for ctypes
                    flat_value = value.flatten()
                    arr = (ctypes.c_double * len(flat_value))(*flat_value)
                    ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))
                    c_inputs_list.append(ptr)
                    num_inputs += 1
            if c_inputs_list:
                c_inputs = (ctypes.POINTER(ctypes.c_double) * len(c_inputs_list))(*c_inputs_list)
        
        # Prepare outputs
        outputs = (ctypes.POINTER(ctypes.c_double) * 1)()
        
        # Call
        lib.preprocess(c_inputs, num_inputs, outputs, 1)
        
        # Extract result
        if not outputs[0]:
            print(f"❌ Output is NULL")
            return False
        
        # Try to determine length from Python result
        if isinstance(python_result, np.ndarray):
            length = python_result.size  # Use size for multi-dimensional arrays
        else:
            length = 1
        
        c_result_flat = np.array([outputs[0][i] for i in range(length)], dtype=np.float64)
        
        # Reshape if needed
        if isinstance(python_result, np.ndarray) and python_result.ndim > 1:
            c_result = c_result_flat.reshape(python_result.shape)
        else:
            c_result = c_result_flat
        
        if isinstance(python_result, np.ndarray):
            if c_result.size != python_result.size:
                print(f"❌ Size mismatch: {c_result.size} vs {python_result.size}")
                return False
            match = np.allclose(c_result.flatten(), python_result.flatten(), rtol=1e-5)
        else:
            match = abs(c_result_flat[0] - python_result) < 1e-5
        
        if match:
            print(f"✅ {name} passed")
            print(f"   Python: {python_result}")
            print(f"   C:      {c_result if isinstance(python_result, np.ndarray) else c_result[0]}")
            return True
        else:
            print(f"❌ {name} failed")
            print(f"   Python: {python_result}")
            print(f"   C:      {c_result if isinstance(python_result, np.ndarray) else c_result[0]}")
            return False
            
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


if __name__ == "__main__":
    # Test arange
    test_operation("arange", """
import numpy as np
result = np.arange(5)
output = result
""")
    
    # Test abs
    test_operation("abs", """
import numpy as np
input_data = np.array([-1.5, -2.0, 0.0, 2.0, 3.5])
result = np.abs(input_data)
output = result
""", {"input_data": np.array([-1.5, -2.0, 0.0, 2.0, 3.5])})
    
    # Test max
    test_operation("max", """
import numpy as np
input_data = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
result = np.max(input_data)
output = result
""", {"input_data": np.array([1.0, 5.0, 3.0, 9.0, 2.0])})
    
    # Test min
    test_operation("min", """
import numpy as np
input_data = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
result = np.min(input_data)
output = result
""", {"input_data": np.array([1.0, 5.0, 3.0, 9.0, 2.0])})
    
    # Test clip
    test_operation("clip", """
import numpy as np
input_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
result = np.clip(input_data, 0.0, 2.0)
output = result
""", {"input_data": np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])})
    
    # Test transpose
    test_operation("transpose", """
import numpy as np
input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
result = np.transpose(input_data)
output = result
""", {"input_data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})
    
    # Test arange with start, stop
    test_operation("arange_start_stop", """
import numpy as np
result = np.arange(2, 7)
output = result
""")
    
    # Test arange with start, stop, step
    test_operation("arange_start_stop_step", """
import numpy as np
result = np.arange(0, 10, 2)
output = result
""")
    
    # Test add
    test_operation("add", """
import numpy as np
input_data = np.array([1.0, 2.0, 3.0, 4.0])
result = np.add(input_data, 2.0)
output = result
""", {"input_data": np.array([1.0, 2.0, 3.0, 4.0])})
    
    # Test subtract
    test_operation("subtract", """
import numpy as np
input_data = np.array([5.0, 4.0, 3.0, 2.0])
result = np.subtract(input_data, 1.0)
output = result
""", {"input_data": np.array([5.0, 4.0, 3.0, 2.0])})
    
    # Test divide
    test_operation("divide", """
import numpy as np
input_data = np.array([4.0, 6.0, 8.0, 10.0])
result = np.divide(input_data, 2.0)
output = result
""", {"input_data": np.array([4.0, 6.0, 8.0, 10.0])})
    
    # Test log
    test_operation("log", """
import numpy as np
input_data = np.array([1.0, 2.71828, 7.389, 20.085])
result = np.log(input_data)
output = result
""", {"input_data": np.array([1.0, 2.71828, 7.389, 20.085])})
    
    # Test sqrt
    test_operation("sqrt", """
import numpy as np
input_data = np.array([1.0, 4.0, 9.0, 16.0])
result = np.sqrt(input_data)
output = result
""", {"input_data": np.array([1.0, 4.0, 9.0, 16.0])})
    
    # Test exp
    test_operation("exp", """
import numpy as np
input_data = np.array([0.0, 1.0, 2.0, 3.0])
result = np.exp(input_data)
output = result
""", {"input_data": np.array([0.0, 1.0, 2.0, 3.0])})
    
    # Test std
    test_operation("std", """
import numpy as np
input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = np.std(input_data)
output = result
""", {"input_data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})
    
    # Test sum
    test_operation("sum", """
import numpy as np
input_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = np.sum(input_data)
output = result
""", {"input_data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})
    
    # Test round
    test_operation("round", """
import numpy as np
input_data = np.array([1.4, 2.6, 3.5, 4.1, 5.9])
result = np.round(input_data)
output = result
""", {"input_data": np.array([1.4, 2.6, 3.5, 4.1, 5.9])})
    
    # Test zeros
    test_operation("zeros", """
import numpy as np
result = np.zeros(5)
output = result
""")
    
    # Test ones
    test_operation("ones", """
import numpy as np
result = np.ones(5)
output = result
""")
    
    # Test array
    test_operation("array", """
import numpy as np
result = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
output = result
""")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

