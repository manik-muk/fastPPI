# FastPPI: Fast Preprocessing Pipeline Interpreter

Convert Python preprocessing code for ML pipelines into **optimized C binaries** with **8-25x speedups**.

## Features

- ✅ **Automatic Tracing**: Captures NumPy operations during execution
- ✅ **Graph Construction**: Builds computational graph with operation dependencies
- ✅ **C Code Generation**: Converts NumPy operations to optimized C code
- ✅ **Clang -O3 Compilation**: Maximum performance optimization
- ✅ **Standalone Binaries**: Deploy as shared libraries (.dylib/.so)
- ⚠️ **Pandas Analysis**: Identify compilable operations (experimental)

## Installation

```bash
# Basic installation (NumPy support only)
pip install -r requirements.txt
pip install -e .

# Full installation (with pandas analysis)
pip install -e ".[full]"
```

**Requirements**: 
- Python 3.7+
- clang compiler (for code compilation)
- NumPy 1.19+
- Optional: pandas (for analysis features)

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
fastppi preprocess.py \
    --inputs "input_data=[1.0,2.0,3.0,4.0,5.0]" \
    --output preprocess_binary \
    --verbose
```

### 3. Benchmark Performance

```bash
python benchmark.py
```

**Result**: ~25x speedup! 🚀

## Supported Operations

### Fully Supported (Compiled to C)

**NumPy Operations:**
- Array creation: `np.array`, `np.zeros`, `np.ones`, `np.arange`
- Arithmetic: `np.add`, `np.subtract`, `np.multiply`, `np.divide`
- Math functions: `np.exp`, `np.log`, `np.sqrt`, `np.abs`
- Reductions: `np.sum`, `np.mean`, `np.std`, `np.max`, `np.min`
- Array ops: `np.reshape`, `np.transpose`, `np.concatenate`
- Other: `np.where`, `np.clip`, `np.round`

### Analysis Only (Experimental)

**Pandas Operations:**
- Can be traced and analyzed
- Use `fastppi-analyze` to identify compilable parts
- See "Advanced Usage" section below

## Command Line Interface

```bash
# Compile preprocessing code
fastppi <file.py> --inputs <inputs> --output <binary> [options]

# Analyze pandas code (experimental)
fastppi-analyze <file.py>

# Options:
#   --inputs      Example inputs (required)
#   --output      Output binary path
#   --save-c      Save generated C code
#   --optimization Optimization flag (default: -O3)
#   --verbose     Print detailed output
```

## Examples

### Example 1: Simple Normalization

```python
# examples/simple_preprocess.py
import numpy as np

x = input_data
mean_val = np.mean(x)
std_val = np.std(x)
normalized = np.divide(np.subtract(x, mean_val), std_val)
output = normalized
```

```bash
fastppi examples/simple_preprocess.py \
    --inputs "input_data=[1.0,2.0,3.0,4.0,5.0]" \
    --output preprocess_binary

python benchmark.py
# Result: 25x speedup (0.096ms → 0.004ms)
```

### Example 2: Matrix Operations

```python
# examples/matrix_operations.py
import numpy as np

x = input_data
x_squared = np.multiply(x, x)
x_shifted = np.add(x_squared, 1.5)
output = np.sqrt(x_shifted)
```

```bash
fastppi examples/matrix_operations.py \
    --inputs "input_data=[1.0,2.0,3.0,4.0,5.0]" \
    --output matrix_ops_binary

python benchmark_matrix_ops.py
# Result: 8x speedup (0.058ms → 0.007ms)
```

## Advanced Usage

### Pandas Code Analysis

FastPPI includes experimental support for analyzing pandas code:

```bash
# Analyze which operations can be compiled
fastppi-analyze examples/feature_engineering.py
```

**Output shows:**
- Which operations were captured
- What can be compiled to C
- What requires Python fallback
- Optimization recommendations

### Hybrid Approach (Recommended)

For complex pipelines, use a hybrid approach:

1. **Keep pandas** for: data loading, string ops, categorical encoding
2. **Compile with FastPPI**: numeric array operations, math-heavy preprocessing
3. **Combine**: Load with pandas → extract arrays → process with compiled binary

**Example:**

```python
# Step 1: Pandas preprocessing (Python)
import pandas as pd
df = pd.read_csv('data.csv')
df = df.dropna()
df['city'] = df['city'].str.lower()

# Step 2: Extract numeric arrays
age = df['age'].values
income = df['income'].values

# Step 3: Use compiled binary for numeric preprocessing (C)
import ctypes
lib = ctypes.CDLL('./preprocess_binary.dylib')
# ... call compiled function (see benchmark.py for example)
```

### Using Compiled Binaries

The compiled binaries are shared libraries that can be called from Python:

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

See `benchmark.py` for complete examples.

## Project Structure

```
fastPPI/
├── fastPPI/                  # Core package
│   ├── core/                # Core compilation functionality
│   │   ├── compiler.py      # C code compilation
│   │   ├── codegen.py       # C code generation
│   │   └── graph.py         # Computational graph
│   ├── tracers/             # Operation tracers
│   │   ├── tracer.py        # NumPy tracer
│   │   ├── pandas_tracer.py # Pandas tracer (experimental)
│   ├── analysis/            # Analysis tools
│   │   ├── unified_tracer.py# Unified tracing
│   │   ├── extended_codegen.py # Extended code generation
│   │   └── analyze.py       # Analysis CLI
│   ├── __init__.py
│   ├── main.py              # Main CLI
│   └── cli.py               # CLI wrapper
├── examples/                # Example scripts
│   ├── simple_preprocess.py
│   ├── matrix_operations.py
│   └── feature_engineering.py
├── benchmark.py             # Benchmarking tools
├── benchmark_matrix_ops.py
├── README.md               # This file
├── requirements.txt        # Dependencies
└── setup.py               # Package setup
```

## Performance

Benchmark results on simple preprocessing tasks:

| Example | Python Time | Binary Time | Speedup |
|---------|------------|-------------|---------|
| Normalization | 0.096 ms | 0.004 ms | **25x** |
| Matrix Ops | 0.058 ms | 0.007 ms | **8x** |

Performance gains increase with:
- Larger arrays
- More operations
- Repeated calls
- Batch processing

## How It Works

1. **Trace Execution**: Run Python code with example inputs, capturing all NumPy operations
2. **Build Graph**: Construct computational graph with operation dependencies
3. **Generate C Code**: Convert operations to equivalent C code with optimizations
4. **Compile**: Use clang -O3 to compile to shared library
5. **Deploy**: Use compiled binary in production for fast preprocessing

## Limitations

### Current Limitations

- **NumPy only**: Full support for NumPy operations only
- **Static shapes**: Input/output shapes determined from examples
- **No control flow**: Limited support for if/else, loops
- **No Python operators**: Must use `np.add()` not `+`

### Pandas Limitations

- **Compilation**: Many pandas operations are now compilable to C
- **Simple ops**: Numeric operations and lambdas are supported
- **String ops**: Basic string operations (lower, upper, strip) are supported

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

**Q: "No operations captured"**
- Use explicit NumPy functions: `np.add(x, y)` not `x + y`
- Check that code actually executes operations
- Verify example inputs are provided

**Q: Output doesn't match Python**
- Ensure deterministic operations
- Check for floating-point precision differences
- Use `benchmark.py` to compare outputs

**Q: Want to compile pandas code**
- Use `fastppi-analyze` to see what's compilable
- Extract numeric operations to separate function
- Use hybrid approach (pandas + compiled binary)

## Contributing

Contributions welcome! Areas to help:

1. **More NumPy operations**: Add support for additional NumPy functions
2. **Pandas compilation**: Expand C code generation for more pandas ops
3. **Optimization**: Improve generated C code
4. **Documentation**: More examples and tutorials

## License

MIT License

## Citation

If you use FastPPI in your research, please cite:

```bibtex
@software{fastppi2024,
  title={FastPPI: Fast Preprocessing Pipeline Interpreter},
  author={FastPPI Contributors},
  year={2024},
  url={https://github.com/yourusername/fastPPI}
}
```

## Changelog

### v0.2.0 (Current)
- Added pandas analysis tools
- Unified tracing system
- `fastppi-analyze` command
- Extended code generation framework

### v0.1.0
- Initial release
- NumPy operation support
- C code generation and compilation
- Basic benchmarking tools
