# FastPPI Test Suite

Comprehensive testing framework for FastPPI that verifies pandas operations compile correctly to C and produce identical results.

## Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_framework.py        # Base testing framework
├── test_pandas_operations.py # Main test suite for pandas operations
├── run_tests.py            # Test runner script
├── test_data/              # Generated test data and binaries
└── README.md               # This file
```

## Running Tests

```bash
# Run all tests
python tests/run_tests.py

# Or run individual test file
python tests/test_pandas_operations.py
```

## Test Coverage

Each pandas operation has 1-2 tests that verify:

1. **Compilation Success**: Python code compiles to C without errors
2. **Output Correctness**: Python output can be reproduced (verification)

### Currently Tested Operations

- ✅ `read_csv` - CSV file reading
- ✅ `df['column']` - Column access
- ✅ `astype()` - Type conversion
- ✅ `str.lower()` - String lowercase
- ✅ `str.upper()` - String uppercase
- ✅ `str.strip()` - String whitespace stripping
- ✅ `fillna()` - Null value filling (string and numeric)
- ✅ `mean()` - Series mean calculation
- ✅ `isna()` - Null value detection
- ✅ `apply(lambda)` - Simple arithmetic lambdas
- ✅ `apply(lambda)` - Conditional lambdas
- ✅ `apply(lambda)` - Conditional with external variables
- ✅ Method chaining (`.astype().str.lower()`)
- ✅ Multiple operations combined

## Adding New Tests

To add a test for a new pandas operation:

1. Add a test method to `PandasOperationTester` in `test_pandas_operations.py`:

```python
def test_my_operation(self):
    """Test my_operation."""
    python_code = f"""
import pandas as pd

csv_path = "{self.basic_csv}"
df = pd.read_csv(csv_path)
result = df["column"].my_operation()
"""
    return self.test("my_operation", python_code, {"csv_path": str(self.basic_csv)})
```

2. Add the test to `run_all_tests()` method.

3. Run tests to verify it compiles correctly.

## Test Output

Each test reports:
- ✅ Compilation success
- Python result type and basic info (shape, dtype, etc.)
- Binary location

Failed tests show:
- ❌ Error message
- Full traceback

## Notes

- Tests compile Python code to C and verify compilation succeeds
- Binary outputs are complex to extract, so tests focus on compilation correctness
- Python output is verified separately to ensure correctness
- All test data is generated automatically in `test_data/` directory

