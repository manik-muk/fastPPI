"""
Tests for NumPy operations: array creation, arithmetic, and math functions.
Tests verify Python NumPy output matches compiled C output.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tests.test_framework import FastPPITestFramework


class TestNumPyOperations:
    """Test suite for NumPy operations."""
    
    def __init__(self):
        self.framework = FastPPITestFramework()
    
    def test_arange_simple(self):
        """Test np.arange(stop)."""
        python_code = """
import numpy as np

result = np.arange(5)
output = result
"""
        passed, msg = self.framework.run_test(python_code, "test_arange_simple", {})
        assert passed, f"arange simple test failed: {msg}"
        print("✅ test_arange_simple passed")
    
    def test_arange_start_stop(self):
        """Test np.arange(start, stop)."""
        python_code = """
import numpy as np

result = np.arange(2, 7)
output = result
"""
        passed, msg = self.framework.run_test(python_code, "test_arange_start_stop", {})
        assert passed, f"arange start_stop test failed: {msg}"
        print("✅ test_arange_start_stop passed")
    
    def test_arange_start_stop_step(self):
        """Test np.arange(start, stop, step)."""
        python_code = """
import numpy as np

result = np.arange(0, 10, 2)
output = result
"""
        passed, msg = self.framework.run_test(python_code, "test_arange_start_stop_step", {})
        assert passed, f"arange start_stop_step test failed: {msg}"
        print("✅ test_arange_start_stop_step passed")
    
    def test_abs(self):
        """Test np.abs()."""
        python_code = """
import numpy as np

input_data = np.array([-1.5, -2.0, 0.0, 2.0, 3.5])
result = np.abs(input_data)
output = result
"""
        passed, msg = self.framework.run_test(python_code, "test_abs", 
                                             {"input_data": np.array([-1.5, -2.0, 0.0, 2.0, 3.5])})
        assert passed, f"abs test failed: {msg}"
        print("✅ test_abs passed")
    
    def test_max(self):
        """Test np.max()."""
        python_code = """
import numpy as np

input_data = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
result = np.max(input_data)
output = result
"""
        passed, msg = self.framework.run_test(python_code, "test_max",
                                             {"input_data": np.array([1.0, 5.0, 3.0, 9.0, 2.0])})
        assert passed, f"max test failed: {msg}"
        print("✅ test_max passed")
    
    def test_min(self):
        """Test np.min()."""
        python_code = """
import numpy as np

input_data = np.array([1.0, 5.0, 3.0, 9.0, 2.0])
result = np.min(input_data)
output = result
"""
        passed, msg = self.framework.run_test(python_code, "test_min",
                                             {"input_data": np.array([1.0, 5.0, 3.0, 9.0, 2.0])})
        assert passed, f"min test failed: {msg}"
        print("✅ test_min passed")
    
    def test_transpose_2d(self):
        """Test np.transpose() on 2D array."""
        python_code = """
import numpy as np

input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
result = np.transpose(input_data)
output = result
"""
        passed, msg = self.framework.run_test(python_code, "test_transpose_2d",
                                             {"input_data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})
        assert passed, f"transpose 2d test failed: {msg}"
        print("✅ test_transpose_2d passed")
    
    def test_clip(self):
        """Test np.clip()."""
        python_code = """
import numpy as np

input_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
result = np.clip(input_data, 0.0, 2.0)
output = result
"""
        passed, msg = self.framework.run_test(python_code, "test_clip",
                                             {"input_data": np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])})
        assert passed, f"clip test failed: {msg}"
        print("✅ test_clip passed")
    
    def run_all_tests(self):
        """Run all tests."""
        tests = [
            self.test_arange_simple,
            self.test_arange_start_stop,
            self.test_arange_start_stop_step,
            self.test_abs,
            self.test_max,
            self.test_min,
            self.test_transpose_2d,
            self.test_clip,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
        
        print("\n" + "=" * 80)
        print(f"TEST RESULTS: {passed} passed, {failed} failed")
        print("=" * 80)
        
        return failed == 0


if __name__ == "__main__":
    tester = TestNumPyOperations()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

