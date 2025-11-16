"""
Test pandas read_csv functionality.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from tests.test_framework import FastPPITestFramework, assert_results_match


def test_read_csv_basic():
    """Test basic CSV reading."""
    framework = FastPPITestFramework()
    
    # Create test CSV
    csv_content = """id,age,income,city
1,25,55000,New York
2,34,72000,Los Angeles
3,45,68000,Chicago
"""
    csv_path = framework.test_data_dir / "test_basic.csv"
    csv_path.write_text(csv_content)
    
    python_code = f"""
import pandas as pd

csv_path = "{csv_path}"
df = pd.read_csv(csv_path)
result = df
"""
    
    # Run Python version
    python_result = framework.run_python_code(python_code)
    
    # Compile and verify
    passed, message = framework.run_test(python_code, "test_read_csv_basic", {"csv_path": str(csv_path)})
    assert passed, message
    
    print("✅ test_read_csv_basic passed")


def test_read_csv_with_nulls():
    """Test CSV reading with null values."""
    framework = FastPPITestFramework()
    
    csv_content = """id,age,income,city
1,25,55000,New York
2,,72000,Los Angeles
3,45,,Chicago
4,29,51000,
"""
    csv_path = framework.test_data_dir / "test_nulls.csv"
    csv_path.write_text(csv_content)
    
    python_code = f"""
import pandas as pd

csv_path = "{csv_path}"
df = pd.read_csv(csv_path)
result = df
"""
    
    passed, message = framework.run_test(python_code, "test_read_csv_nulls", {"csv_path": str(csv_path)})
    assert passed, message
    
    print("✅ test_read_csv_with_nulls passed")


if __name__ == "__main__":
    test_read_csv_basic()
    test_read_csv_with_nulls()
    print("\n✅ All read_csv tests passed!")

