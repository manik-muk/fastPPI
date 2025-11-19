# FastPPI: Fast Preprocessing Pipeline Interpreter

Convert Python preprocessing code for ML pipelines into **optimized C binaries** with significant speedups.

## Features

- **Automatic Tracing**: Captures NumPy and pandas operations during execution
- **Graph Construction**: Builds computational graph with operation dependencies
- **C Code Generation**: Converts operations to optimized C code
- **Clang -O3 Compilation**: Maximum performance optimization
- **Standalone Binaries**: Deploy as shared libraries (.dylib/.so)
- **Pandas Support**: Compile pandas operations to C (DataFrame/Series operations)
- **HTTP Data Loading**: Compile `requests.get()` + `pd.DataFrame()` patterns to C
- **Lambda Functions**: Compile pandas `apply()` with lambda functions

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd fastPPI

# Install dependencies
pip install -r requirements.txt

# Install FastPPI
pip install -e .
```

**Requirements**: 
- Python 3.7+
- clang compiler (for code compilation)
- NumPy 1.19+
- pandas (for pandas operations)
- libcurl and jansson (for HTTP operations, installed via Homebrew on macOS)

## Quick Start

### 1. Create a Preprocessing Script

```python
# preprocess.py
import numpy as np

# Use explicit NumPy functions for tracing
x = input_data
mean_val = np.mean(x)
std_val = np.std(x)
normalized = np.divide(np.subtract(x, mean_val), std_val)
output = normalized
```

**Important**: Use `np.add()`, `np.subtract()` instead of `+`, `-` for proper tracing.

### 2. Compile to Binary

```bash
python -m fastPPI.main preprocess.py \
    --inputs "input_data=[1.0,2.0,3.0,4.0,5.0]" \
    --output preprocess_binary \
    --verbose
```

Or with pandas:

```python
# feature_engineering.py
import pandas as pd

df = pd.read_csv(csv_path)
df["age_normalized"] = (df["age"] - df["age"].mean()) / df["age"].std()
result = df
```

```bash
python -m fastPPI.main feature_engineering.py \
    --inputs "csv_path=example_data.csv" \
    --output feature_binary \
    --verbose
```

## Supported Operations

### NumPy Operations (Fully Supported)

**Array Creation:**
- `np.array`, `np.zeros`, `np.ones`, `np.arange`

**Arithmetic:**
- `np.add`, `np.subtract`, `np.multiply`, `np.divide`

**Math Functions:**
- `np.exp`, `np.log`, `np.sqrt`, `np.abs`

**Reductions:**
- `np.sum`, `np.mean`, `np.std`, `np.max`, `np.min`

**Array Operations:**
- `np.concatenate`, `np.transpose`, `np.reshape`

**Other:**
- `np.clip`, `np.round`, `np.where`

### Pandas Operations (Fully Supported)

**Data Loading:**
- `pd.read_csv()` - Read CSV files
- `requests.get()` + `pd.DataFrame()` - HTTP GET with JSON parsing

**DataFrame Operations:**
- `df['column']` - Column access
- `df.sort_values()` - Sort by column
- `df.groupby()` - Group by column (simplified)
- `pd.concat()` - Concatenate DataFrames (vertical/horizontal)

**Series Operations:**
- `series.mean()`, `series.median()` - Aggregations
- `series.fillna()` - Fill missing values
- `series.astype()` - Type conversion
- `series.apply(lambda)` - Apply lambda functions
- `series.isna()` - Check for null values

**String Operations:**
- `series.str.lower()` - Convert to lowercase
- `series.str.upper()` - Convert to uppercase
- `series.str.strip()` - Strip whitespace

**Datetime Operations:**
- `pd.to_datetime()` - Convert Series to datetime64
- `series.dt.day` - Extract day component from datetime
- `series.dt.month` - Extract month component from datetime
- `series.dt.year` - Extract year component from datetime

**Rolling Window Operations:**
- `series.rolling(window).mean()` - Rolling window mean
- `series.rolling(window).sum()` - Rolling window sum

**Exponential Moving Average:**
- `series.ewm(span=...).mean()` - Exponential moving average with span
- `series.ewm(alpha=...).mean()` - Exponential moving average with alpha

**Categorical:**
- `pd.get_dummies()` - One-hot encoding

## Command Line Interface

```bash
# Compile preprocessing code
python -m fastPPI.main <file.py> --inputs <inputs> --output <binary> [options]

# Options:
#   --inputs      Example inputs (required, JSON format or key=value)
#   --output      Output binary path
#   --save-c      Save generated C code to file
#   --optimization Optimization flag (default: -O3)
#   --verbose     Print detailed output
```

### Input Format

Inputs can be provided as:
- JSON string: `--inputs '{"var1": [1,2,3], "var2": 5}'`
- JSON file: `--inputs inputs.json`
- Key-value pairs: `--inputs "var1=[1,2,3],var2=5"`

### Using Compiled Binaries

The compiled binaries are shared libraries that can be executed directly:

```bash
# Execute the binary (outputs are printed or saved)
./preprocess_binary.dylib
```

For programmatic use, load via `ctypes`:

```python
import ctypes
import numpy as np

# Load the compiled library
lib = ctypes.CDLL('./preprocess_binary.dylib')  # or .so on Linux

# Define function signature
lib.preprocess.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # inputs
    ctypes.c_int,  # num_inputs
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # outputs
    ctypes.c_int   # num_outputs
]

# Prepare inputs
input_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
input_array = (ctypes.c_double * len(input_data))(*input_data)
input_ptr = ctypes.cast(input_array, ctypes.POINTER(ctypes.c_double))
inputs = (ctypes.POINTER(ctypes.c_double) * 1)(input_ptr)

# Prepare outputs
outputs = (ctypes.POINTER(ctypes.c_double) * 1)()

# Call compiled function
lib.preprocess(inputs, 1, outputs, 1)

# Extract results
result = np.array([outputs[0][i] for i in range(len(input_data))])
```

## Performance

Performance improvements vary based on:
- **Operation complexity**: Simple operations see 2-5x speedups
- **Data size**: Larger arrays benefit more (5-15x speedups)
- **Operation count**: More operations = better amortization
- **Network I/O**: HTTP operations are I/O bound, so speedups are smaller

**Measured Benchmarks** (run on macOS with clang -O3 optimization, averaged over multiple runs):

| Operation | Python Time | C Binary Time | Speedup | Notes |
|-----------|-------------|---------------|---------|-------|
| Normalization (10k elements, 1k iter) | 0.0248 ms | 0.0064 ms | **3.91x** | NumPy only, direct binary execution |
| Feature Engineering (HTTP + Pandas, 500 iter) | 3.32 ms | 1.48 ms | **2.25x** | Includes HTTP I/O, direct binary execution |
| Toy Preprocessing (1k rows, 100 iter) | 4.62 ms | 0.23 ms | **20.0x** | Pandas + NumPy, compute-intensive pipeline |
| Compute-Intensive Feature Engineering (10k rows, 100 iter) | 98.9 ms | 0.40 ms | **245x** | Complex lambdas, statistical transforms, direct binary execution |

*Note: Results show variance between runs. Normalization speedup ranged from 3.08x to 5.13x. Feature engineering speedup ranged from 1.99x to 3.19x. Toy preprocessing speedup ranged from 19.95x to 20.0x. Compute-intensive speedup ranged from 224x to 285x. Averages shown above.*

**Typical speedups:**
- NumPy array operations: **2-5x** faster
- Pandas DataFrame operations: **2-20x** faster (compute-bound operations show larger speedups)
- Complex preprocessing pipelines: **10-20x** faster (multiple operations, lambda functions)
- Compute-intensive pipelines: **100-250x** faster (many operations, complex lambdas, statistical transforms)
- HTTP + Pandas pipelines: **2-3x** faster (I/O bound, but still faster)

*Note: Actual performance depends on your hardware, data size, and operation mix. The feature engineering benchmark includes HTTP requests which are I/O bound, limiting the speedup. Pure compute operations show larger speedups. Benchmark your specific use case for accurate numbers.*

## How It Works

1. **Trace Execution**: Run Python code with example inputs, capturing all NumPy and pandas operations
2. **Build Graph**: Construct computational graph with operation dependencies
3. **Generate C Code**: Convert operations to equivalent C code with optimizations
4. **Compile**: Use clang with aggressive optimizations to compile to shared library
5. **Deploy**: Execute compiled binary directly or load via ctypes

## Architecture

FastPPI uses a modular tracer system:

- **NumPy Tracer**: Captures NumPy operations
- **Pandas Tracer**: Captures pandas DataFrame/Series operations
- **HTTP Tracer**: Tracks HTTP requests for data loading patterns
- **Unified Tracer**: Combines all tracers for complete operation capture

## Examples

See the `examples/` directory for:
- `feature_engineering_http.py` - Pandas preprocessing with HTTP data loading

## Testing

Run the test suite:

```bash
# Test pandas operations (22 tests)
python tests/test_pandas_operations.py

# Test NumPy operations
python tests/test_numpy_basic.py
```

## Troubleshooting

**Q: Compilation fails with "clang not found"**
```bash
# macOS: Install Xcode Command Line Tools
xcode-select --install

# Ubuntu/Debian
sudo apt-get install clang

# Fedora/RHEL
sudo yum install clang
```

**Q: "library 'jansson' not found" (for HTTP operations)**
```bash
# macOS (Homebrew)
brew install jansson

# Ubuntu/Debian
sudo apt-get install libjansson-dev

# Fedora/RHEL
sudo yum install jansson-devel
```

**Q: "No operations captured"**
- Use explicit NumPy functions: `np.add(x, y)` not `x + y`
- Check that code actually executes operations
- Verify example inputs are provided
- For pandas, ensure operations are on DataFrames/Series, not Python lists

**Q: Output doesn't match Python**
- Ensure deterministic operations
- Check for floating-point precision differences
- Verify all operations are supported (unsupported ops may be skipped)

**Q: Pandas operations not compiling**
- Check that the operation is in the supported list above
- Ensure pandas is installed: `pip install pandas`
- Verify the C libraries are built: `cd c_implementations && make`

## Contributing

Contributions welcome! Areas to help:

1. **More NumPy operations**: Add support for additional NumPy functions
2. **More pandas operations**: Expand C code generation for more pandas ops
3. **Optimization**: Improve generated C code
4. **Documentation**: More examples and tutorials
5. **Testing**: Add more test cases

## License

MIT License
