"""
Comprehensive tests for pandas operations.
Tests verify Python pandas output matches compiled C output.
"""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tests.test_framework import FastPPITestFramework


class TestPandasOperations:
    """Test suite for pandas operations."""
    
    def __init__(self):
        self.framework = FastPPITestFramework()
        self.setup_test_data()
    
    def setup_test_data(self):
        """Create test CSV files."""
        # Basic test data
        basic_csv = """id,age,income,city
1,25,55000,New York
2,34,72000,Los Angeles
3,45,68000,Chicago
4,29,51000,San Francisco
5,38,,Boston
"""
        self.basic_csv = self.framework.test_data_dir / "test_basic.csv"
        self.basic_csv.write_text(basic_csv)
        
        # Data with nulls
        nulls_csv = """id,age,income,city
1,25,55000,New York
2,,72000,Los Angeles
3,45,,Chicago
4,29,51000,
5,38,62000,Boston
"""
        self.nulls_csv = self.framework.test_data_dir / "test_nulls.csv"
        self.nulls_csv.write_text(nulls_csv)
    
    def test_column_access(self):
        """Test df['column'] access."""
        python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["age"]
"""
        # Run Python
        python_result = self.framework.run_python_code(python_code)
        
        # Compile
        passed, msg = self.framework.run_test(python_code, "test_column_access", 
                                             {"csv_path": str(self.basic_csv)})
        assert passed, f"Column access test failed: {msg}"
        print("✅ test_column_access passed")
    
    def test_astype(self):
        """Test astype conversion."""
        python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["city"].astype(str)
"""
        passed, msg = self.framework.run_test(python_code, "test_astype",
                                             {"csv_path": str(self.basic_csv)})
        assert passed, f"astype test failed: {msg}"
        print("✅ test_astype passed")
    
    def test_str_lower(self):
        """Test .str.lower() operation."""
        python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["city"].astype(str).str.lower()
"""
        passed, msg = self.framework.run_test(python_code, "test_str_lower",
                                             {"csv_path": str(self.basic_csv)})
        assert passed, f"str.lower test failed: {msg}"
        print("✅ test_str_lower passed")
    
    def test_str_upper(self):
        """Test .str.upper() operation."""
        python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["city"].astype(str).str.upper()
"""
        passed, msg = self.framework.run_test(python_code, "test_str_upper",
                                             {"csv_path": str(self.basic_csv)})
        assert passed, f"str.upper test failed: {msg}"
        print("✅ test_str_upper passed")
    
    def test_fillna_string(self):
        """Test fillna with string value."""
        python_code = f"""
import pandas as pd

csv_path = "{self.nulls_csv}"
df = pd.read_csv(csv_path)
result = df["city"].fillna("unknown")
"""
        passed, msg = self.framework.run_test(python_code, "test_fillna_string",
                                             {"csv_path": str(self.nulls_csv)})
        assert passed, f"fillna string test failed: {msg}"
        print("✅ test_fillna_string passed")
    
    def test_fillna_numeric(self):
        """Test fillna with numeric value."""
        python_code = f"""
import pandas as pd

csv_path = "{self.nulls_csv}"
df = pd.read_csv(csv_path)
result = df["income"].fillna(0)
"""
        passed, msg = self.framework.run_test(python_code, "test_fillna_numeric",
                                             {"csv_path": str(self.nulls_csv)})
        assert passed, f"fillna numeric test failed: {msg}"
        print("✅ test_fillna_numeric passed")
    
    def test_mean(self):
        """Test Series.mean() operation."""
        python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["income"].mean()
"""
        passed, msg = self.framework.run_test(python_code, "test_mean",
                                             {"csv_path": str(self.basic_csv)})
        assert passed, f"mean test failed: {msg}"
        print("✅ test_mean passed")
    
    def test_isna(self):
        """Test isna() operation."""
        python_code = f"""
import pandas as pd

csv_path = "{self.nulls_csv}"
df = pd.read_csv(csv_path)
result = df["age"].isna()
"""
        passed, msg = self.framework.run_test(python_code, "test_isna",
                                             {"csv_path": str(self.nulls_csv)})
        assert passed, f"isna test failed: {msg}"
        print("✅ test_isna passed")
    
    def test_lambda_simple(self):
        """Test apply with simple lambda."""
        python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["age"].apply(lambda x: x * 2)
"""
        passed, msg = self.framework.run_test(python_code, "test_lambda_simple",
                                             {"csv_path": str(self.basic_csv)})
        assert passed, f"lambda simple test failed: {msg}"
        print("✅ test_lambda_simple passed")
    
    def test_lambda_conditional(self):
        """Test apply with conditional lambda."""
        python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["income"].apply(lambda x: 1 if pd.notnull(x) and x > 60000 else 0)
"""
        passed, msg = self.framework.run_test(python_code, "test_lambda_conditional",
                                             {"csv_path": str(self.basic_csv)})
        assert passed, f"lambda conditional test failed: {msg}"
        print("✅ test_lambda_conditional passed")
    
    def test_method_chaining(self):
        """Test method chaining."""
        python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["city"].astype(str).str.lower()
"""
        passed, msg = self.framework.run_test(python_code, "test_method_chaining",
                                             {"csv_path": str(self.basic_csv)})
        assert passed, f"method chaining test failed: {msg}"
        print("✅ test_method_chaining passed")
    
    def run_all_tests(self):
        """Run all tests."""
        tests = [
            self.test_column_access,
            self.test_astype,
            self.test_str_lower,
            self.test_str_upper,
            self.test_fillna_string,
            self.test_fillna_numeric,
            self.test_mean,
            self.test_isna,
            self.test_lambda_simple,
            self.test_lambda_conditional,
            self.test_method_chaining,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"❌ {test.__name__} failed: {e}")
                failed += 1
        
        print("\n" + "=" * 80)
        print(f"TEST RESULTS: {passed} passed, {failed} failed")
        print("=" * 80)
        
        return failed == 0


if __name__ == "__main__":
    tester = TestPandasOperations()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

