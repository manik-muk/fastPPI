# C Implementations for FastPPI

This directory contains C implementations of pandas operations that FastPPI can compile to.

## Structure

```
c_implementations/
├── include/              # Header files
│   ├── data_structures.h     # DataFrame and Series definitions
│   ├── data_structures.c     # Data structure implementations
│   └── pandas_c.h            # Pandas operation interfaces
├── pandas/               # Pandas implementations
│   └── pandas_operations.c
├── Makefile              # Build script
└── README.md             # This file
```

## Building

```bash
cd c_implementations
make
```

This creates:
- `lib/libpandas_c.a` - Static library for pandas operations
- `lib/libpandas_c.dylib` (or `.so`) - Shared library

## Implemented Operations

### Pandas Operations

- ✅ `pandas_df_mean` - Compute mean of DataFrame columns
- ✅ `pandas_series_mean` - Compute mean of Series
- ✅ `pandas_df_fillna` - Fill missing values in DataFrame
- ✅ `pandas_series_fillna` - Fill missing values in Series
- ✅ `pandas_series_apply` - Apply function to Series (basic ops)
- ✅ `pandas_series_astype` - Convert Series dtype
- ✅ `pandas_series_str_lower` - Convert strings to lowercase
- ✅ `pandas_series_str_upper` - Convert strings to uppercase
- ✅ `pandas_series_str_strip` - Strip whitespace from strings
- ✅ `pandas_series_isna` - Check for null values
- ✅ `pandas_read_csv` - Read CSV file into DataFrame
- ✅ `pandas_df_getitem` - Get column from DataFrame

## Usage with FastPPI

1. **Build the libraries:**
   ```bash
   cd c_implementations
   make
   ```

2. **Set environment variables:**
   ```bash
   export LIBRARY_PATH=$LIBRARY_PATH:$(pwd)/lib
   export C_INCLUDE_PATH=$C_INCLUDE_PATH:$(pwd)/include
   ```

3. **Compile with FastPPI:**
   FastPPI will automatically link these libraries when it detects pandas operations.

## Extending

To add new operations:

1. **Add function to header file** (`pandas_c.h`)

2. **Implement in C file** (`pandas_operations.c`)

3. **Register in FastPPI:**
   ```python
   from fastPPI.core.extended_codegen import CFunctionRegistry
   
   CFunctionRegistry.register_function(
       op_name="your_op",
       obj_type="DataFrame",  # or None
       c_function="your_c_function",
       include='"pandas_c.h"',
       return_type="double*",
       description="Your operation description"
   )
   ```

## Data Structures

### Series
- 1D array with optional null mask
- Supports: float64, int64, string, bool
- Memory managed automatically

### DataFrame
- Collection of named Series (columns)
- All columns must have same length
- Row-major storage

## Notes

- These implementations are **simplified** versions for demonstration
- Production use should include:
  - Better error handling
  - Memory optimization
  - More complete feature sets
  - CSV parsing library (for `read_csv`)
  - Proper sorting algorithms (for median)
  - Thread safety (if needed)

## Testing

Example test program:
```c
#include "include/pandas_c.h"
#include <stdio.h>

int main() {
    // Create a simple Series
    Series* s = series_create(5, 'f');
    s->data[0] = 1.0;
    s->data[1] = 2.0;
    s->data[2] = NAN;  // Missing value
    s->data[3] = 4.0;
    s->data[4] = 5.0;
    
    // Compute mean
    double mean_val = pandas_series_mean(s);
    printf("Mean: %f\n", mean_val);
    
    // Fill missing values
    Series* filled = pandas_series_fillna(s, 0.0, "mean");
    printf("Filled values:\n");
    for (int i = 0; i < filled->length; i++) {
        printf("  [%d] = %f\n", i, filled->data[i]);
    }
    
    series_free(s);
    series_free(filled);
    return 0;
}
```

Compile and run:
```bash
clang test.c -Iinclude -Llib -lpandas_c -o test
./test
```

