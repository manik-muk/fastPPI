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
