"""
C code generator from computational graph.
Converts Python/NumPy operations to equivalent C code.
"""

from typing import List, Dict, Set, Optional
from .graph import ComputationalGraph, GraphNode
from ..tracers.tracer import Operation
import numpy as np


class CCodeGenerator:
    """Generates C code from computational graph."""
    
    def __init__(self, graph: ComputationalGraph, input_vars: List[str], input_arrays: Dict[str, np.ndarray] = None):
        self.graph = graph
        self.input_vars = input_vars
        self.input_arrays = input_arrays or {}  # var_name -> array object
        self.variable_map: Dict[int, str] = {}  # op_id -> variable name
        self.array_to_var_map: Dict[int, str] = {}  # id(array) -> variable name (for inputs)
        self.var_counter = 0
        self.includes = set()
        self.includes.add("#include <stdio.h>")
        self.includes.add("#include <stdlib.h>")
        self.includes.add("#include <math.h>")
        self.includes.add("#include <string.h>")
        self.includes.add("#include <stdalign.h>")  # For aligned_alloc
        
        # Element-wise operations that can be fused (single input, single output, same shape)
        self.fusible_ops = {
            'exp', 'log', 'sqrt', 'abs', 'round', 
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            'sinh', 'cosh', 'tanh'
        }
        
        # Normalization pattern: mean -> std -> subtract -> divide
        # We'll detect this pattern and generate optimized fused normalization
        
        # Add OpenMP for parallelization (Numba uses this for large arrays)
        # Check if OpenMP is available (optional - only use if installed)
        self.use_openmp = self._check_openmp_available()
        if self.use_openmp:
            self.includes.add("#include <omp.h>")
        
        # Always use BLAS for optimized matrix multiplication (same as NumPy)
        # Declare cblas_dgemm manually (header may not be in standard paths)
        blas_decl = """// BLAS declarations for optimized matrix multiplication
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
void cblas_dgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc);"""
        self.includes.add(blas_decl)
        self.use_blas = True  # Always use BLAS
        
        # Map input arrays
        if input_arrays:
            for var_name, arr in input_arrays.items():
                if isinstance(arr, np.ndarray):
                    self.array_to_var_map[id(arr)] = var_name
    
    def _check_openmp_available(self) -> bool:
        """Check if OpenMP is available on the system."""
        import subprocess
        import sys
        import os
        
        # Try to compile a simple OpenMP test program
        test_code = """
        #include <omp.h>
        int main() { return 0; }
        """
        
        try:
            # Try macOS first (libomp via Homebrew - keg-only, needs explicit paths)
            if sys.platform == "darwin":
                try:
                    brew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
                    libomp_path = f"{brew_prefix}/opt/libomp"
                    
                    # Check if libomp exists
                    if os.path.exists(f"{libomp_path}/include/omp.h"):
                        # Try to compile with correct paths
                        result = subprocess.run(
                            ['clang', '-Xpreprocessor', '-fopenmp',
                             f'-I{libomp_path}/include',
                             f'-L{libomp_path}/lib', '-lomp',
                             '-x', 'c', '-', '-o', '/dev/null'],
                            input=test_code,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            return True
                except Exception as e:
                    # Fall through to try other methods
                    pass
            
            # Try Linux standard OpenMP
            result = subprocess.run(
                ['clang', '-fopenmp', '-x', 'c', '-', '-o', '/dev/null'],
                input=test_code,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
    
    def _check_blas_available(self) -> tuple[bool, str, str]:
        """Check if BLAS (cblas) is available on the system."""
        import subprocess
        import sys
        import os
        
        # Try to compile a simple BLAS test program
        test_code = """
        #include <cblas.h>
        int main() { 
            double a[4] = {1, 2, 3, 4};
            double b[4] = {5, 6, 7, 8};
            double c[4] = {0};
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c, 2);
            return 0; 
        }
        """
        
        try:
            # Common BLAS library paths
            blas_paths = []
            
            # Check conda/miniconda environment (common on macOS)
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            if conda_prefix:
                blas_paths.extend([
                    f"{conda_prefix}/lib/libopenblas.dylib",
                    f"{conda_prefix}/lib/libcblas.dylib",
                    f"{conda_prefix}/lib/libblas.dylib",
                ])
            
            # Check common system paths
            python_prefix = sys.prefix
            blas_paths.extend([
                f"{python_prefix}/lib/libopenblas.dylib",
                f"{python_prefix}/lib/libcblas.dylib",
                "/usr/local/lib/libopenblas.dylib",
                "/opt/homebrew/lib/libopenblas.dylib",
            ])
            
            # Try to find a BLAS library
            blas_lib = None
            for path in blas_paths:
                if os.path.exists(path):
                    blas_lib = path
                    break
            
            if not blas_lib:
                # Try to find via Python's numpy
                try:
                    import numpy as np
                    numpy_path = np.__file__
                    numpy_dir = os.path.dirname(numpy_path)
                    lib_dir = os.path.join(numpy_dir, '..', '..', 'lib')
                    lib_dir = os.path.abspath(lib_dir)
                    
                    for lib_file in os.listdir(lib_dir):
                        if 'openblas' in lib_file.lower() or 'cblas' in lib_file.lower():
                            if lib_file.endswith('.dylib') or lib_file.endswith('.so'):
                                blas_lib = os.path.join(lib_dir, lib_file)
                                break
                except:
                    pass
            
            if blas_lib:
                # Find cblas.h header file
                lib_dir = os.path.dirname(blas_lib)
                include_dirs = [
                    f"{lib_dir}/../include",
                    f"{lib_dir}/../../include",
                    "/usr/local/include",
                    "/opt/homebrew/include",
                ]
                
                # Also check conda/python prefix
                if conda_prefix:
                    include_dirs.insert(0, f"{conda_prefix}/include")
                include_dirs.insert(0, f"{sys.prefix}/include")
                
                # Try to find cblas.h
                cblas_header = None
                for inc_dir in include_dirs:
                    header_path = os.path.join(inc_dir, "cblas.h")
                    if os.path.exists(header_path):
                        cblas_header = inc_dir
                        break
                
                # Try to compile with BLAS
                compile_cmd = ['clang', '-x', 'c', '-', '-o', '/dev/null',
                              f'-L{lib_dir}', '-lopenblas']
                if cblas_header:
                    compile_cmd.extend([f'-I{cblas_header}'])
                
                result = subprocess.run(
                    compile_cmd,
                    input=test_code,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return (True, blas_lib, cblas_header if cblas_header else "")
            
            # Try standard library names (but we still need the library path for linking)
            for lib_name in ['-lopenblas', '-lcblas', '-lblas']:
                result = subprocess.run(
                    ['clang', '-x', 'c', '-', '-o', '/dev/null', lib_name],
                    input=test_code,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Found via standard library, use the first blas_lib we found or empty
                    return (True, blas_lib if blas_lib else "", "")
            
            # If we found the library but compilation failed, still return True
            # (we'll declare the function manually)
            if blas_lib:
                return (True, blas_lib, "")
            
            return (False, "", "")
        except:
            return (False, "", "")
        
    def _get_var_name(self, op_id: int) -> str:
        """Get or create a variable name for an operation result."""
        if op_id not in self.variable_map:
            # Use op_id directly for variable name to ensure uniqueness and consistency
            var_name = f"var_{op_id}"
            self.variable_map[op_id] = var_name
            # Keep var_counter in sync with max op_id seen
            if op_id >= self.var_counter:
                self.var_counter = op_id + 1
        return self.variable_map[op_id]
    
    def _get_c_type(self, dtype, shape) -> str:
        """Convert NumPy dtype to C type."""
        if shape is None or len(shape) == 0:
            # Scalar
            if dtype == np.int32 or dtype == int:
                return "int"
            elif dtype == np.int64:
                return "long"
            elif dtype == np.float32 or dtype == np.float64 or dtype == float:
                return "double"
            elif dtype == bool:
                return "int"
            else:
                return "double"
        else:
            # Array - simplified to double* for now
            return "double*"
    
    def _get_shape_str(self, shape) -> str:
        """Convert shape tuple to string."""
        if shape is None or len(shape) == 0:
            return ""
        return "[" + "][".join(str(d) for d in shape) + "]"
    
    def _declare_restrict_ptr(self, var_name: str, is_const: bool, declared_restrict_ptrs: Set[str]) -> List[str]:
        """
        Declare a restrict pointer if it hasn't been declared yet.
        Only declares restrict pointers for input variables, not intermediate results.
        
        Args:
            var_name: Base variable name (without _r suffix)
            is_const: Whether to use const
            declared_restrict_ptrs: Set of already declared restrict pointer names (also tracks type)
            
        Returns:
            List of code lines (empty if already declared or if it's an intermediate variable)
        """
        # Only declare restrict pointers for input variables (those in self.input_vars)
        # Intermediate variables (var_0, var_1, etc.) should not use restrict pointers
        if var_name not in self.input_vars:
            # This is an intermediate variable - don't use restrict pointer
            return []
        
        restrict_ptr_name = f"{var_name}_r"
        if restrict_ptr_name in declared_restrict_ptrs:
            return []
        
        declared_restrict_ptrs.add(restrict_ptr_name)
        const_str = "const " if is_const else ""
        return [f"    {const_str}double* __restrict__ {restrict_ptr_name} = {var_name};"]
    
    def _detect_normalization_pattern(self, sorted_nodes: List[GraphNode]) -> Optional[Dict]:
        """
        Detect normalization pattern: mean -> std -> subtract(mean) -> divide(std)
        This is Numba's key optimization - fusing all normalization steps.
        
        Returns:
            Dict with normalization info if pattern detected, None otherwise
        """
        # Look for pattern: mean node -> std node -> subtract node -> divide node
        # Note: mean and std produce scalars, so we need to check args directly
        mean_node = None
        std_node = None
        subtract_node = None
        divide_node = None
        input_var = None
        
        for node in sorted_nodes:
            op = node.operation
            
            # Find mean operation that operates on an input array
            if op.op_name == "mean" and mean_node is None:
                for arg in op.args:
                    if isinstance(arg, np.ndarray) and id(arg) in self.array_to_var_map:
                        input_var = self.array_to_var_map[id(arg)]
                        mean_node = node
                        break
            
            # Find std operation (should also operate on same input)
            if op.op_name == "std" and std_node is None and mean_node is not None:
                for arg in op.args:
                    if isinstance(arg, np.ndarray) and id(arg) in self.array_to_var_map:
                        if self.array_to_var_map[id(arg)] == input_var:
                            std_node = node
                            break
            
            # Find subtract operation: subtract(input, mean)
            # mean is a scalar (float64), so we check if arg[1] matches mean's result by id
            if op.op_name == "subtract" and subtract_node is None and mean_node is not None:
                if len(op.args) >= 2:
                    arg0_is_input = (isinstance(op.args[0], np.ndarray) and 
                                   id(op.args[0]) in self.array_to_var_map and
                                   self.array_to_var_map[id(op.args[0])] == input_var)
                    # Check if arg[1] is the mean result (scalar - compare by id)
                    arg1_is_mean = (id(op.args[1]) == id(mean_node.operation.result))
                    
                    if arg0_is_input and arg1_is_mean:
                        subtract_node = node
                        continue
            
            # Find divide operation: divide(subtract_result, std)
            # std is a scalar (float64), so we check if arg[1] matches std's result by id
            if op.op_name == "divide" and divide_node is None and subtract_node is not None and std_node is not None:
                if len(op.args) >= 2:
                    arg0_is_subtract = (isinstance(op.args[0], np.ndarray) and
                                      id(op.args[0]) == id(subtract_node.operation.result))
                    # Check if arg[1] is the std result (scalar - compare by id)
                    arg1_is_std = (id(op.args[1]) == id(std_node.operation.result))
                    
                    if arg0_is_subtract and arg1_is_std:
                        divide_node = node
                        break
        
        # Return pattern if all components found
        if mean_node and std_node and subtract_node and divide_node:
            # Get input array shape from mean operation
            input_shape = None
            for arg in mean_node.operation.args:
                if isinstance(arg, np.ndarray):
                    input_shape = arg.shape
                    break
            
            size = np.prod(input_shape) if input_shape else 1
            
            return {
                'input_var': input_var,
                'mean_node': mean_node,
                'std_node': std_node,
                'subtract_node': subtract_node,
                'divide_node': divide_node,
                'result_var': self._get_var_name(divide_node.operation.op_id),
                'size': size
            }
        
        return None
    
    def _generate_fused_normalization(self, norm_info: Dict, declared_restrict_ptrs: Set[str]) -> List[str]:
        """
        Generate fused normalization code using Numba's optimization:
        1. Compute mean in one pass
        2. Compute variance and normalize in a single fused pass
        
        This reduces from 4 passes to 2 passes, significantly improving performance.
        Numba's key insight: fuse variance computation with the normalization step.
        """
        input_var = norm_info['input_var']
        result_var = norm_info['result_var']
        size = int(norm_info['size'])
        
        lines = []
        
        # Use restrict pointer for input
        lines.extend(self._declare_restrict_ptr(input_var, True, declared_restrict_ptrs))
        input_ptr = f"{input_var}_r"
        
        # Temporary variables for mean and variance
        mean_var = f"{result_var}_mean"
        var_var = f"{result_var}_var"
        
        lines.append(f"    // Fused normalization: Numba-style optimization")
        lines.append(f"    // Pass 1: Compute mean")
        lines.append(f"    double {mean_var} = 0.0;")
        mean_loop = f"{mean_var} += {input_ptr}[i];"
        lines.extend(self._generate_optimized_loop(size, mean_loop))
        lines.append(f"    {mean_var} /= {size};")
        
        # Pass 2: Compute variance and normalize in single fused loop
        # This is the key optimization - combine variance computation with normalization
        # Instead of: compute variance -> then normalize, we do both in one pass
        lines.append(f"    // Pass 2: Compute variance and normalize in fused loop")
        lines.append(f"    double {var_var} = 0.0;")
        
        # Fused loop: compute variance and write normalized result
        # For each element: diff = x[i] - mean, variance += diff^2, result[i] = diff / std
        # But we need std first, so we compute variance, then normalize
        # Actually, we can do: compute diff, accumulate variance, store diff
        # Then compute std and divide in a second loop
        # OR: compute variance in first part of loop, then normalize in second part
        # Let's use the simpler approach: compute variance, then normalize
        
        # Compute variance and store (x - mean) in result array
        norm_loop_body = f"double diff = {input_ptr}[i] - {mean_var}; {var_var} += diff * diff; {result_var}[i] = diff;"
        lines.extend(self._generate_optimized_loop(size, norm_loop_body))
        lines.append(f"    {var_var} /= {size};")
        lines.append(f"    double std_val = sqrt({var_var});")
        
        # Final: divide by std (fused with previous would be ideal, but this is still 2 passes total)
        final_loop = f"{result_var}[i] /= std_val;"
        lines.extend(self._generate_optimized_loop(size, final_loop))
        
        return lines
    
    def _find_fusible_chain(self, start_node: GraphNode, processed_ops: Set[int]) -> List[GraphNode]:
        """
        Find a chain of fusible element-wise operations starting from a node.
        A chain is fusible if:
        - Each operation is element-wise (single input array, single output array)
        - Operations are in sequence (output of one is input of next)
        - All operations have same array size
        - Operations are in the fusible_ops set
        - Intermediate results are only used once (enables fusion)
        
        Returns:
            List of nodes in the fusible chain (including start_node)
        """
        chain = [start_node]
        current = start_node
        
        # Don't mark as processed yet - we'll do that only if chain is long enough
        temp_processed = set()
        temp_processed.add(current.operation.op_id)
        
        while True:
            # Check if current node is fusible
            op = current.operation
            if op.op_name not in self.fusible_ops:
                break
            
            # Must have exactly one output that's used by exactly one node
            # (intermediate result only used once - key requirement for fusion)
            if len(current.outputs) != 1:
                break
            
            next_node = current.outputs[0]
            
            # Check if next node is also fusible
            if next_node.operation.op_name not in self.fusible_ops:
                break
            
            # Check if next node has exactly one input (our current node)
            # This ensures the intermediate result is only used by this next operation
            if len(next_node.inputs) != 1 or next_node.inputs[0] != current:
                break
            
            # Check if arrays have same size
            if (isinstance(op.result, np.ndarray) and 
                isinstance(next_node.operation.result, np.ndarray)):
                if op.result.shape != next_node.operation.result.shape:
                    break
            else:
                break
            
            # Check if next node hasn't been processed yet
            if next_node.operation.op_id in processed_ops or next_node.operation.op_id in temp_processed:
                break
            
            # Add to chain
            chain.append(next_node)
            temp_processed.add(next_node.operation.op_id)
            current = next_node
        
        # Only mark as processed if we have a chain of 2+ operations
        if len(chain) > 1:
            processed_ops.update(temp_processed)
        
        return chain
    
    def _generate_fused_loop(self, chain: List[GraphNode], declared_restrict_ptrs: Set[str]) -> List[str]:
        """
        Generate a single fused loop for a chain of element-wise operations.
        This is Numba's key optimization - combining multiple operations into one loop.
        
        Example: exp(log(sqrt(x))) becomes:
        for (int i = 0; i < size; i++) {
            result[i] = exp(log(sqrt(x[i])));
        }
        """
        if not chain:
            return []
        
        # Get the first input (should be from input or previous non-fusible operation)
        first_node = chain[0]
        first_op = first_node.operation
        
        # Find the input variable
        # Check if input comes from another node or is an input variable
        input_var = None
        
        if len(first_node.inputs) > 0:
            # Input comes from another operation
            input_node = first_node.inputs[0]
            if isinstance(input_node.operation.result, np.ndarray):
                input_var = self._get_var_name(input_node.operation.op_id)
        else:
            # Input comes directly from input variable - find it from args
            for arg in first_op.args:
                if isinstance(arg, np.ndarray):
                    if id(arg) in self.array_to_var_map:
                        input_var = self.array_to_var_map[id(arg)]
                        break
                    else:
                        # Find which node created this array
                        for prev_node in self.graph.nodes:
                            if (isinstance(prev_node.operation.result, np.ndarray) and
                                id(prev_node.operation.result) == id(arg)):
                                input_var = self._get_var_name(prev_node.operation.op_id)
                                break
                        if input_var:
                            break
        
        if not input_var:
            return []  # Can't fuse if we can't find input
        
        # Get output variable (last node in chain)
        last_node = chain[-1]
        result_var = self._get_var_name(last_node.operation.op_id)
        
        # Get array size
        size = np.prod(first_op.result.shape) if isinstance(first_op.result, np.ndarray) else 1
        size_int = int(size)
        
        lines = []
        
        # Build the nested expression: exp(log(sqrt(x[i])))
        # Start from the innermost (first operation) and work outward
        input_ptr = input_var
        if input_var in self.input_vars:
            # Use restrict pointer for input
            lines.extend(self._declare_restrict_ptr(input_var, True, declared_restrict_ptrs))
            input_ptr = f"{input_var}_r"
        
        # Map operation names to C functions
        c_func_map = {
            'exp': 'exp',
            'log': 'log',
            'sqrt': 'sqrt',
            'abs': 'fabs',
            'round': 'round',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan',
            'asin': 'asin',
            'acos': 'acos',
            'atan': 'atan',
            'sinh': 'sinh',
            'cosh': 'cosh',
            'tanh': 'tanh',
        }
        
        # Build nested expression by traversing chain
        # Start with input array access
        expr = f"{input_ptr}[i]"
        
        # Apply each operation in the chain (innermost to outermost)
        for node in chain:
            op_name = node.operation.op_name
            c_func = c_func_map.get(op_name, op_name)
            expr = f"{c_func}({expr})"
        
        # Generate the fused loop
        loop_body = f"{result_var}[i] = {expr};"
        
        # Use optimized loop generation with SIMD hints
        lines.extend(self._generate_optimized_loop(size, loop_body))
        
        return lines
    
    def _generate_optimized_loop(self, size: int, loop_body: str, use_restrict: bool = True) -> List[str]:
        """
        Generate an optimized loop with SIMD hints, restrict pointers, and parallelization.
        Similar to Numba's loop optimization techniques.
        
        Numba optimizations applied:
        1. Full loop unrolling for very small arrays (< 16 elements)
        2. Aggressive SIMD vectorization hints
        3. OpenMP parallelization for large arrays (Numba's key optimization)
        4. Loop tiling/blocking for cache optimization
        5. Memory prefetching hints
        
        Args:
            size: Loop size
            loop_body: The body of the loop (without braces)
            use_restrict: Whether to use restrict pointers
            
        Returns:
            List of code lines for the optimized loop
        """
        lines = []
        size_int = int(size)
        
        # For very small arrays, fully unroll the loop (Numba's approach)
        if size_int <= 16:
            # Fully unroll small loops for maximum performance
            for i in range(size_int):
                # Replace 'i' in loop_body with actual index
                unrolled_body = loop_body.replace('[i]', f'[{i}]')
                lines.append(f"    {unrolled_body}")
            return lines
        
        # Determine optimization strategy based on array size
        # Numba typically only parallelizes for larger arrays to avoid overhead
        # Cache blocking only for very large arrays (50000+) where overhead is justified
        # For simple operations (add/multiply), parallelization threshold should be higher
        use_parallel = self.use_openmp and size_int >= 15000  # Higher threshold to avoid overhead
        use_cache_blocking = self.use_openmp and size_int >= 50000  # Only for very large arrays
        
        # For small-medium arrays, use aggressive unrolling
        if size_int < 64:
            lines.append("    #pragma GCC ivdep")  # Ignore vector dependencies
            lines.append("    #pragma clang loop vectorize(enable) interleave(enable)")
            lines.append(f"    #pragma clang loop unroll_count(8)")  # More aggressive unrolling
        elif size_int < 1000:
            # Medium arrays: standard SIMD with moderate unrolling
            lines.append("    #pragma GCC ivdep")
            lines.append("    #pragma clang loop vectorize(enable) interleave(enable)")
            lines.append("    #pragma clang loop unroll_count(4)")
        else:
            # Large arrays: Use OpenMP parallelization with optimized scheduling
            # Numba uses dynamic/guided scheduling for better load balancing
            
            if use_parallel:
                # Numba's approach: default chunksize = 0 means one chunk per thread (static-like)
                # For uniform workloads (add/multiply), static scheduling with auto chunking is best
                # This matches Numba's behavior: when chunksize=0, it creates num_threads chunks
                # For very large arrays, we can use guided for better load balancing
                if size_int >= 100000:
                    # Very large: guided scheduling (adaptive chunk size, best load balancing)
                    schedule = "guided"
                    chunk_size = max(64, size_int // 2000)  # Adaptive chunk size
                else:
                    # Numba's default: static scheduling with automatic chunk sizing
                    # When chunksize is not specified, OpenMP divides work evenly among threads
                    # This matches Numba's chunksize=0 behavior (one chunk per thread)
                    schedule = "static"
                    # Don't specify chunk_size - let OpenMP divide evenly (matches Numba's chunksize=0)
                    # This is more efficient than specifying a fixed chunk size
                
                if not use_cache_blocking:
                    # Standard parallel loop (no cache blocking)
                    # Numba uses static-like scheduling (chunksize=0 = one chunk per thread)
                    lines.append("    // Numba-style parallelization: static scheduling (chunksize=0 equivalent)")
                    lines.append("    // Thread affinity set via OMP_PROC_BIND=close for better cache locality")
                    if schedule == "static":
                        # Static without chunk size = evenly divided (matches Numba's chunksize=0)
                        # Use proc_bind(close) pragma for thread affinity (OpenMP 4.0+)
                        lines.append("#if _OPENMP >= 201307  // OpenMP 4.0+")
                        lines.append("    #pragma omp parallel for schedule(static) proc_bind(close)")
                        lines.append("#else")
                        lines.append("    #pragma omp parallel for schedule(static)")
                        lines.append("#endif")
                    else:
                        lines.append("#if _OPENMP >= 201307  // OpenMP 4.0+")
                        lines.append(f"    #pragma omp parallel for schedule({schedule}, {chunk_size}) proc_bind(close)")
                        lines.append("#else")
                        lines.append(f"    #pragma omp parallel for schedule({schedule}, {chunk_size})")
                        lines.append("#endif")
                    lines.append("    #pragma GCC ivdep")  # Ignore vector dependencies
                    lines.append("    #pragma clang loop vectorize(enable) interleave(enable)")
                    # More aggressive vectorization for parallel loops
                    lines.append("    #pragma clang loop vectorize_width(8) interleave_count(4)")
            else:
                # For large but not huge arrays (1000-15000), use aggressive SIMD without parallelization
                # Parallelization overhead is not worth it for these sizes
                # Focus on SIMD vectorization and cache optimization instead
                lines.append("    #pragma GCC ivdep")
                lines.append("    #pragma clang loop vectorize(enable) interleave(enable)")
                # More aggressive vectorization for larger non-parallel arrays
                if size_int >= 5000:
                    lines.append("    #pragma clang loop vectorize_width(8) interleave_count(4)")
                else:
                    lines.append("    #pragma clang loop vectorize_width(4) interleave_count(2)")
                # Add memory prefetch hints for better cache utilization
                if size_int >= 2000:
                    lines.append("    #pragma clang loop unroll(enable)")
        
        # Generate the loop
        # For very large arrays with OpenMP, add cache-blocking optimization
        if use_cache_blocking:
            # Implement cache blocking for very large arrays
            # This improves cache locality by processing data in blocks
            # Typical L1 cache is 32KB, L2 is 256KB-1MB
            # For doubles (8 bytes), we can fit ~4000-8000 elements in L2 cache
            cache_block_size = 4096  # Process 4096 elements at a time (fits in L2 cache)
            
            # Determine scheduling for inner parallel loop
            if size_int >= 50000:
                inner_schedule = "guided"
                inner_chunk = max(64, size_int // 2000)
            else:
                inner_schedule = "dynamic"
                inner_chunk = max(256, size_int // 800)
            
            lines.append("    // Cache-blocking optimization for very large arrays")
            lines.append(f"    const int cache_block_size = {cache_block_size};")
            lines.append(f"    #pragma omp parallel")
            lines.append(f"    {{")
            lines.append(f"        #pragma omp for schedule({inner_schedule}, {inner_chunk})")
            lines.append(f"        for (int block_start = 0; block_start < {size_int}; block_start += cache_block_size) {{")
            lines.append(f"            int block_end = block_start + cache_block_size;")
            lines.append(f"            if (block_end > {size_int}) block_end = {size_int};")
            lines.append(f"            #pragma GCC ivdep")
            lines.append(f"            #pragma clang loop vectorize(enable) interleave(enable)")
            lines.append(f"            for (int i = block_start; i < block_end; i++) {{")
            lines.append(f"                {loop_body}")
            lines.append(f"            }}")
            lines.append(f"        }}")
            lines.append(f"    }}")
        else:
            # Standard loop
            lines.append(f"    for (int i = 0; i < {size_int}; i++) {{")
            lines.append(f"        {loop_body}")
            lines.append(f"    }}")
        
        return lines
    
    def _generate_array_allocation(self, var_name: str, shape: tuple, dtype) -> str:
        """
        Generate code to allocate an array with alignment for SIMD.
        Numba uses aligned memory allocation for better SIMD performance.
        """
        if not shape:
            return ""
        
        size = 1
        for dim in shape:
            size *= dim
        
        # Use aligned allocation for better SIMD performance (32-byte alignment for AVX)
        # This is a key Numba optimization - aligned memory allows better vectorization
        # aligned_alloc is C11 standard, but we provide a fallback for compatibility
        # Note: On some systems, we might need to use posix_memalign or _aligned_malloc
        # For now, use aligned_alloc with error checking
        return f"""    double* {var_name} = (double*)aligned_alloc(32, {size} * sizeof(double));
    if (!{var_name}) {{
        // Fallback to regular malloc if aligned_alloc fails
        {var_name} = (double*)malloc({size} * sizeof(double));
    }}"""
    
    def _generate_operation_code(self, node: GraphNode, allocated_vars: Set[str], declared_scalars: Set[str], declared_restrict_ptrs: Set[str] = None, declared_contiguous_vars: Set[str] = None) -> List[str]:
        """Generate C code for a single operation."""
        if declared_restrict_ptrs is None:
            declared_restrict_ptrs = set()
        if declared_contiguous_vars is None:
            declared_contiguous_vars = set()
        op = node.operation
        lines = []
        result_var = self._get_var_name(op.op_id)
        
        # Get input variable names
        input_vars = []
        for arg in op.args:
            if isinstance(arg, np.ndarray):
                # Check if it's an input array
                if id(arg) in self.array_to_var_map:
                    input_vars.append(self.array_to_var_map[id(arg)])
                else:
                    # Find which node created this array
                    found = False
                    for prev_node in self.graph.nodes:
                        if (isinstance(prev_node.operation.result, np.ndarray) and
                            id(prev_node.operation.result) == id(arg)):
                            input_vars.append(self._get_var_name(prev_node.operation.op_id))
                            found = True
                            break
                    if not found:
                        input_vars.append("NULL")  # Fallback
            elif isinstance(arg, (int, float, bool)):
                input_vars.append(str(arg))
            else:
                input_vars.append("0")
        
        # Track if we need to allocate (only for arrays, and only once)
        needs_allocation = (isinstance(op.result, np.ndarray) and 
                          id(op.result) not in self.array_to_var_map and
                          result_var not in allocated_vars)
        
        # Generate operation-specific code
        if op.op_name == "array":
            # Array creation
            if isinstance(op.result, np.ndarray):
                shape = op.result.shape
                size = np.prod(shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, shape, op.result.dtype))
                    allocated_vars.add(result_var)
                # Copy values if provided
                if len(op.args) > 0:
                    # Only handle if it's a simple list or numpy array
                    # Skip DataFrame/Series objects - they should be handled by extended codegen
                    if isinstance(op.args[0], (list, np.ndarray)):
                        try:
                            arr = np.array(op.args[0]).flatten()
                            size_int = int(size) if hasattr(size, '__int__') else len(arr)
                            for i, val in enumerate(arr[:size_int]):
                                # Handle inf and nan properly
                                if isinstance(val, (int, float)):
                                    if np.isinf(val):
                                        inf_val = "-INFINITY" if val < 0 else "INFINITY"
                                        if size_int == 1 and op.result.shape == ():  # 0-d array (scalar)
                                            lines.append(f"    {result_var} = {inf_val};")
                                        else:
                                            lines.append(f"    {result_var}[{i}] = {inf_val};")
                                    elif np.isnan(val):
                                        if size_int == 1 and op.result.shape == ():  # 0-d array (scalar)
                                            lines.append(f"    {result_var} = NAN;")
                                        else:
                                            lines.append(f"    {result_var}[{i}] = NAN;")
                                    else:
                                        if size_int == 1 and op.result.shape == ():  # 0-d array (scalar)
                                            lines.append(f"    {result_var} = {val};")
                                        else:
                                            lines.append(f"    {result_var}[{i}] = {val};")
                        except (ValueError, TypeError):
                            # Skip if conversion fails (e.g., DataFrame/Series)
                            pass
        
        elif op.op_name == "zeros":
            if needs_allocation:
                shape = op.result.shape if hasattr(op.result, 'shape') else (op.args[0],)
                if isinstance(shape, int):
                    shape = (shape,)
                size = np.prod(shape)
                lines.append(self._generate_array_allocation(result_var, shape, op.result.dtype))
                allocated_vars.add(result_var)
            else:
                shape = op.result.shape if hasattr(op.result, 'shape') else (op.args[0],)
                if isinstance(shape, int):
                    shape = (shape,)
            size = np.prod(shape)
            lines.append(f"    memset({result_var}, 0, {size} * sizeof(double));")
        
        elif op.op_name == "ones":
            if needs_allocation:
                shape = op.result.shape if hasattr(op.result, 'shape') else (op.args[0],)
                if isinstance(shape, int):
                    shape = (shape,)
                size = np.prod(shape)
                lines.append(self._generate_array_allocation(result_var, shape, op.result.dtype))
                allocated_vars.add(result_var)
            else:
                shape = op.result.shape if hasattr(op.result, 'shape') else (op.args[0],)
                if isinstance(shape, int):
                    shape = (shape,)
            size = np.prod(shape)
            lines.append(f"    for (int i = 0; i < {size}; i++) {{ {result_var}[i] = 1.0; }}")
        
        elif op.op_name == "arange":
            # np.arange(start, stop, step) or np.arange(stop)
            if needs_allocation:
                shape = op.result.shape if hasattr(op.result, 'shape') else (len(op.result),)
                if isinstance(shape, int):
                    shape = (shape,)
                size = np.prod(shape)
                lines.append(self._generate_array_allocation(result_var, shape, op.result.dtype))
                allocated_vars.add(result_var)
            else:
                shape = op.result.shape if hasattr(op.result, 'shape') else (len(op.result),)
                if isinstance(shape, int):
                    shape = (shape,)
            size = np.prod(shape)
            
            # Determine start, stop, step
            if len(op.args) == 1:
                # np.arange(stop) - start=0, step=1
                stop = op.args[0] if isinstance(op.args[0], (int, float)) else input_vars[0]
                lines.append(f"    for (int i = 0; i < {size}; i++) {{ {result_var}[i] = (double)i; }}")
            elif len(op.args) == 2:
                # np.arange(start, stop) - step=1
                start = op.args[0] if isinstance(op.args[0], (int, float)) else input_vars[0]
                stop = op.args[1] if isinstance(op.args[1], (int, float)) else input_vars[1]
                if isinstance(op.args[0], (int, float)) and isinstance(op.args[1], (int, float)):
                    lines.append(f"    for (int i = 0; i < {size}; i++) {{ {result_var}[i] = {start} + (double)i; }}")
                else:
                    lines.append(f"    for (int i = 0; i < {size}; i++) {{ {result_var}[i] = {start} + (double)i; }}")
            else:
                # np.arange(start, stop, step)
                start = op.args[0] if isinstance(op.args[0], (int, float)) else input_vars[0]
                stop = op.args[1] if isinstance(op.args[1], (int, float)) else input_vars[1]
                step = op.args[2] if isinstance(op.args[2], (int, float)) else input_vars[2]
                if isinstance(op.args[0], (int, float)) and isinstance(op.args[2], (int, float)):
                    lines.append(f"    for (int i = 0; i < {size}; i++) {{ {result_var}[i] = {start} + {step} * (double)i; }}")
                else:
                    lines.append(f"    for (int i = 0; i < {size}; i++) {{ {result_var}[i] = {start} + {step} * (double)i; }}")
        
        elif op.op_name == "add":
            if op.result.shape and len(op.result.shape) > 0:
                # Array addition
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Use restrict pointers only for input variables (not intermediate results)
                # This helps compiler optimize while avoiding redefinition errors
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                else:
                    input0_ptr = input_vars[0]
                
                # Check if second operand is scalar (not an array)
                if len(op.args) > 1 and not isinstance(op.args[1], np.ndarray):
                    loop_body = f"{result_var}[i] = {input0_ptr}[i] + {input_vars[1]};"
                else:
                    if len(op.args) > 1 and isinstance(op.args[1], np.ndarray) and len(input_vars) > 1 and input_vars[1] != "NULL":
                        lines.extend(self._declare_restrict_ptr(input_vars[1], True, declared_restrict_ptrs))
                        input1_ptr = f"{input_vars[1]}_r" if input_vars[1] in self.input_vars else input_vars[1]
                        loop_body = f"{result_var}[i] = {input0_ptr}[i] + {input1_ptr}[i];"
                    else:
                        loop_body = f"{result_var}[i] = {input0_ptr}[i] + {input_vars[1]};"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar addition (variable already declared upfront)
                lines.append(f"    {result_var} = {input_vars[0]} + {input_vars[1]};")
        
        elif op.op_name == "multiply":
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Use restrict pointers only for input variables (not intermediate results)
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                else:
                    input0_ptr = input_vars[0]
                
                # Check if second operand is scalar
                if len(op.args) > 1 and not isinstance(op.args[1], np.ndarray):
                    loop_body = f"{result_var}[i] = {input0_ptr}[i] * {input_vars[1]};"
                else:
                    if len(op.args) > 1 and isinstance(op.args[1], np.ndarray) and len(input_vars) > 1 and input_vars[1] != "NULL":
                        lines.extend(self._declare_restrict_ptr(input_vars[1], True, declared_restrict_ptrs))
                        input1_ptr = f"{input_vars[1]}_r" if input_vars[1] in self.input_vars else input_vars[1]
                        loop_body = f"{result_var}[i] = {input0_ptr}[i] * {input1_ptr}[i];"
                    else:
                        loop_body = f"{result_var}[i] = {input0_ptr}[i] * {input_vars[1]};"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar multiply (variable already declared upfront)
                lines.append(f"    {result_var} = {input_vars[0]} * {input_vars[1]};")
        
        elif op.op_name == "sum":
            if isinstance(op.args[0], np.ndarray):
                size = np.prod(op.args[0].shape)
                # Assign (variable already declared upfront)
                lines.append(f"    {result_var} = 0.0;")
                
                # Use restrict pointer for input
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    loop_body = f"{result_var} += {input_vars[0]}_r[i];"
                else:
                    loop_body = f"{result_var} += {input_vars[0]}[i];"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                lines.append(f"    {result_var} = {input_vars[0]};")
        
        elif op.op_name == "mean":
            if isinstance(op.args[0], np.ndarray):
                size = np.prod(op.args[0].shape)
                size_int = int(size)
                # Assign (variable already declared upfront)
                lines.append(f"    {result_var} = 0.0;")
                
                # Use restrict pointer for input
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    loop_body = f"{result_var} += {input_vars[0]}_r[i];"
                else:
                    loop_body = f"{result_var} += {input_vars[0]}[i];"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
                lines.append(f"    {result_var} /= {size_int};")
            else:
                lines.append(f"    {result_var} = {input_vars[0]};")
        
        elif op.op_name == "std":
            if isinstance(op.args[0], np.ndarray):
                size = np.prod(op.args[0].shape)
                size_int = int(size)
                
                # Use restrict pointer for input
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                
                # Calculate mean first (use temporary local variable with unique name)
                # Use op_id to ensure uniqueness even if operation appears multiple times
                mean_var = f"{result_var}_mean_{op.op_id}"
                # Check if already declared (in case of duplicate operations)
                if mean_var not in declared_scalars:
                    lines.append(f"    double {mean_var} = 0.0;")
                    declared_scalars.add(mean_var)
                else:
                    lines.append(f"    {mean_var} = 0.0;")
                    
                if input_vars[0] != "NULL":
                    loop_body = f"{mean_var} += {input_vars[0]}_r[i];"
                else:
                    loop_body = f"{mean_var} += {input_vars[0]}[i];"
                lines.extend(self._generate_optimized_loop(size, loop_body))
                lines.append(f"    {mean_var} /= {size_int};")
                
                # Calculate variance (use temporary local variable with unique name)
                var_var = f"{result_var}_var_{op.op_id}"
                # Check if already declared
                if var_var not in declared_scalars:
                    lines.append(f"    double {var_var} = 0.0;")
                    declared_scalars.add(var_var)
                else:
                    lines.append(f"    {var_var} = 0.0;")
                    
                if input_vars[0] != "NULL":
                    loop_body = f"double diff = {input_vars[0]}_r[i] - {mean_var}; {var_var} += diff * diff;"
                else:
                    loop_body = f"double diff = {input_vars[0]}[i] - {mean_var}; {var_var} += diff * diff;"
                lines.extend(self._generate_optimized_loop(size, loop_body))
                lines.append(f"    {var_var} /= {size_int};")
                
                # Calculate std (variable already declared upfront)
                lines.append(f"    {result_var} = sqrt({var_var});")
            else:
                lines.append(f"    {result_var} = 0.0;")
        
        elif op.op_name == "max":
            # Check if input is resolved
            if len(input_vars) > 0 and input_vars[0] != "NULL":
                if isinstance(op.args[0], np.ndarray):
                    size = np.prod(op.args[0].shape)
                    size_int = int(size)
                    # Variable already declared upfront
                    lines.append(f"    {result_var} = {input_vars[0]}[0];")
                    
                    # Use restrict pointer for input
                    if input_vars[0] != "NULL":
                        lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                        loop_body = f"if ({input_vars[0]}_r[i] > {result_var}) {{ {result_var} = {input_vars[0]}_r[i]; }}"
                    else:
                        loop_body = f"if ({input_vars[0]}[i] > {result_var}) {{ {result_var} = {input_vars[0]}[i]; }}"
                    
                    # Generate optimized loop (note: max/min can't be easily vectorized, but restrict helps)
                    lines.append(f"    for (int i = 1; i < {size_int}; i++) {{")
                    lines.append(f"        {loop_body}")
                    lines.append(f"    }}")
                else:
                    lines.append(f"    {result_var} = {input_vars[0]};")
            else:
                # Skip max operation if inputs not resolved
                lines.append(f"    // Skipping max operation - inputs not resolved")
                if result_var not in allocated_vars:
                    lines.append(f"    {result_var} = 0.0;")
        
        elif op.op_name == "min":
            # Check if input is resolved
            if len(input_vars) > 0 and input_vars[0] != "NULL":
                if isinstance(op.args[0], np.ndarray):
                    size = np.prod(op.args[0].shape)
                    size_int = int(size)
                    # Variable already declared upfront
                    lines.append(f"    {result_var} = {input_vars[0]}[0];")
                    
                    # Use restrict pointer for input
                    if input_vars[0] != "NULL":
                        lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                        loop_body = f"if ({input_vars[0]}_r[i] < {result_var}) {{ {result_var} = {input_vars[0]}_r[i]; }}"
                    else:
                        loop_body = f"if ({input_vars[0]}[i] < {result_var}) {{ {result_var} = {input_vars[0]}[i]; }}"
                    
                    # Generate optimized loop
                    lines.append(f"    for (int i = 1; i < {size_int}; i++) {{")
                    lines.append(f"        {loop_body}")
                    lines.append(f"    }}")
                else:
                    lines.append(f"    {result_var} = {input_vars[0]};")
            else:
                # Skip min operation if inputs not resolved
                lines.append(f"    // Skipping min operation - inputs not resolved")
                if result_var not in allocated_vars:
                    lines.append(f"    {result_var} = 0.0;")
        
        elif op.op_name == "exp":
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Use restrict pointers only for input variables (not intermediate results)
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                    loop_body = f"{result_var}[i] = exp({input0_ptr}[i]);"
                else:
                    loop_body = f"{result_var}[i] = exp({input_vars[0]}[i]);"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar exp (variable already declared upfront)
                lines.append(f"    {result_var} = exp({input_vars[0]});")
        
        elif op.op_name == "log":
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Use restrict pointers only for input variables (not intermediate results)
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                    loop_body = f"{result_var}[i] = log({input0_ptr}[i]);"
                else:
                    loop_body = f"{result_var}[i] = log({input_vars[0]}[i]);"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar log (variable already declared upfront)
                lines.append(f"    {result_var} = log({input_vars[0]});")
        
        elif op.op_name == "sqrt":
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Use restrict pointers only for input variables (not intermediate results)
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                    loop_body = f"{result_var}[i] = sqrt({input0_ptr}[i]);"
                else:
                    loop_body = f"{result_var}[i] = sqrt({input_vars[0]}[i]);"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar sqrt (variable already declared upfront)
                lines.append(f"    {result_var} = sqrt({input_vars[0]});")
        
        elif op.op_name == "abs":
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Use restrict pointers only for input variables (not intermediate results)
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                    loop_body = f"{result_var}[i] = fabs({input0_ptr}[i]);"
                else:
                    loop_body = f"{result_var}[i] = fabs({input_vars[0]}[i]);"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar abs (variable already declared upfront)
                lines.append(f"    {result_var} = fabs({input_vars[0]});")
        
        elif op.op_name == "subtract":
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Use restrict pointers only for input variables (not intermediate results)
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                else:
                    input0_ptr = input_vars[0]
                
                # Check if second operand is scalar
                if len(op.args) > 1 and not isinstance(op.args[1], np.ndarray):
                    loop_body = f"{result_var}[i] = {input0_ptr}[i] - {input_vars[1]};"
                else:
                    if len(op.args) > 1 and isinstance(op.args[1], np.ndarray) and len(input_vars) > 1 and input_vars[1] != "NULL":
                        lines.extend(self._declare_restrict_ptr(input_vars[1], True, declared_restrict_ptrs))
                        input1_ptr = f"{input_vars[1]}_r" if input_vars[1] in self.input_vars else input_vars[1]
                        loop_body = f"{result_var}[i] = {input0_ptr}[i] - {input1_ptr}[i];"
                    else:
                        loop_body = f"{result_var}[i] = {input0_ptr}[i] - {input_vars[1]};"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar subtraction (variable already declared upfront)
                lines.append(f"    {result_var} = {input_vars[0]} - {input_vars[1]};")
        
        elif op.op_name == "divide":
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Use restrict pointers only for input variables (not intermediate results)
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                else:
                    input0_ptr = input_vars[0]
                
                # Check if second operand is scalar
                if len(op.args) > 1 and not isinstance(op.args[1], np.ndarray):
                    loop_body = f"{result_var}[i] = {input0_ptr}[i] / {input_vars[1]};"
                else:
                    if len(op.args) > 1 and isinstance(op.args[1], np.ndarray) and len(input_vars) > 1 and input_vars[1] != "NULL":
                        lines.extend(self._declare_restrict_ptr(input_vars[1], True, declared_restrict_ptrs))
                        input1_ptr = f"{input_vars[1]}_r" if input_vars[1] in self.input_vars else input_vars[1]
                        loop_body = f"{result_var}[i] = {input0_ptr}[i] / {input1_ptr}[i];"
                    else:
                        loop_body = f"{result_var}[i] = {input0_ptr}[i] / {input_vars[1]};"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar divide (variable already declared upfront)
                lines.append(f"    {result_var} = {input_vars[0]} / {input_vars[1]};")
        
        elif op.op_name == "round":
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                lines.append(f"    for (int i = 0; i < {size}; i++) {{")
                lines.append(f"        {result_var}[i] = round({input_vars[0]}[i]);")
                lines.append(f"    }}")
            else:
                # Scalar round (variable already declared upfront)
                lines.append(f"    {result_var} = round({input_vars[0]});")
        
        elif op.op_name == "concatenate":
            # Concatenate arrays along axis
            if op.result.shape and len(op.result.shape) > 0:
                total_size = int(np.prod(op.result.shape))
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                # Copy all input arrays sequentially
                offset = 0
                for i, inp_var in enumerate(input_vars):
                    if i < len(op.args) and isinstance(op.args[i], np.ndarray):
                        inp_size = int(np.prod(op.args[i].shape))
                        lines.append(f"    // Copy array {i} ({inp_size} elements)")
                        lines.append(f"    for (int i = 0; i < {inp_size}; i++) {{")
                        lines.append(f"        {result_var}[{offset} + i] = {inp_var}[i];")
                        lines.append(f"    }}")
                        offset += inp_size
        
        elif op.op_name == "where":
            # np.where(condition, x, y) - element-wise conditional
            # Check if we can resolve all inputs - skip if any is NULL
            if len(input_vars) >= 3 and "NULL" not in input_vars[:3] and op.result.shape and len(op.result.shape) > 0:
                size = int(np.prod(op.result.shape))
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                lines.append(f"    for (int i = 0; i < {size}; i++) {{")
                # Check if x and y are scalars or arrays
                x_access = f"{input_vars[1]}[i]" if len(op.args) > 1 and isinstance(op.args[1], np.ndarray) else input_vars[1]
                y_access = f"{input_vars[2]}[i]" if len(op.args) > 2 and isinstance(op.args[2], np.ndarray) else input_vars[2]
                lines.append(f"        {result_var}[i] = {input_vars[0]}[i] ? {x_access} : {y_access};")
                lines.append(f"    }}")
            else:
                # Skip where operation if we can't resolve inputs
                lines.append(f"    // Skipping where operation - inputs not resolved")
                if result_var not in allocated_vars:
                    lines.append(f"    {result_var} = 0.0;")
                else:
                    lines.append(f"    // Variable {result_var} is an array, skipping assignment")
        
        elif op.op_name == "reshape":
            # Reshape just changes the view, data stays the same
            if op.result.shape and len(op.result.shape) > 0:
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                # Copy data (reshape in C is just a memcpy with different shape interpretation)
                if len(input_vars) > 0:
                    size = int(np.prod(op.result.shape))
                    lines.append(f"    memcpy({result_var}, {input_vars[0]}, {size} * sizeof(double));")
        
        elif op.op_name == "transpose":
            # Transpose 2D array (swap rows and columns)
            if op.result.shape and len(op.result.shape) == 2:
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                if len(input_vars) > 0 and isinstance(op.args[0], np.ndarray):
                    rows = op.args[0].shape[0]
                    cols = op.args[0].shape[1]
                    lines.append(f"    for (int i = 0; i < {rows}; i++) {{")
                    lines.append(f"        for (int j = 0; j < {cols}; j++) {{")
                    lines.append(f"            {result_var}[j * {rows} + i] = {input_vars[0]}[i * {cols} + j];")
                    lines.append(f"        }}")
                    lines.append(f"    }}")
            elif op.result.shape and len(op.result.shape) == 1:
                # 1D transpose is a no-op, just copy
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                if len(input_vars) > 0:
                    size = int(np.prod(op.result.shape))
                    lines.append(f"    memcpy({result_var}, {input_vars[0]}, {size} * sizeof(double));")
        
        elif op.op_name == "clip":
            # np.clip(array, min, max)
            # Skip if input is NULL (likely a Series/DataFrame result that can't be processed as numpy array)
            if len(input_vars) > 0 and input_vars[0] == "NULL":
                lines.append(f"    // Skipping clip operation - input is NULL (likely Series/DataFrame)")
                if needs_allocation and op.result.shape and len(op.result.shape) > 0:
                    size = np.prod(op.result.shape)
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                    # Initialize with zeros
                    lines.append(f"    for (int i = 0; i < {size}; i++) {{")
                    lines.append(f"        {result_var}[i] = 0.0;")
                    lines.append(f"    }}")
                return lines
            if op.result.shape and len(op.result.shape) > 0:
                size = np.prod(op.result.shape)
                if needs_allocation:
                    lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                    allocated_vars.add(result_var)
                
                # Get min and max values
                min_val = input_vars[1] if len(input_vars) > 1 else "0.0"
                max_val = input_vars[2] if len(input_vars) > 2 else "INFINITY"
                if len(op.args) > 1 and isinstance(op.args[1], (int, float)):
                    min_val = str(op.args[1])
                if len(op.args) > 2 and isinstance(op.args[2], (int, float)):
                    max_val = str(op.args[2])
                
                # Use restrict pointers only for input variables (not intermediate results)
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    input0_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                    loop_body = f"double val = {input0_ptr}[i]; {result_var}[i] = val < {min_val} ? {min_val} : (val > {max_val} ? {max_val} : val);"
                else:
                    loop_body = f"double val = {input_vars[0]}[i]; {result_var}[i] = val < {min_val} ? {min_val} : (val > {max_val} ? {max_val} : val);"
                
                # Generate optimized loop with SIMD hints
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar clip
                min_val = input_vars[1] if len(input_vars) > 1 else "0.0"
                max_val = input_vars[2] if len(input_vars) > 2 else "INFINITY"
                if len(op.args) > 1 and isinstance(op.args[1], (int, float)):
                    min_val = str(op.args[1])
                if len(op.args) > 2 and isinstance(op.args[2], (int, float)):
                    max_val = str(op.args[2])
                lines.append(f"    double val = {input_vars[0]};")
                lines.append(f"    {result_var} = val < {min_val} ? {min_val} : (val > {max_val} ? {max_val} : val);")
        
        elif op.op_name in ("dot", "matmul"):
            # Matrix multiplication: C = A @ B
            # NumPy uses BLAS (cblas_dgemm) for optimal performance
            # We'll implement an optimized version with cache blocking and SIMD
            if op.result.shape and len(op.result.shape) == 2:
                # 2D matrix multiplication
                # Get shapes: A is (m, k), B is (k, n), result is (m, n)
                a_shape = op.args[0].shape if isinstance(op.args[0], np.ndarray) else None
                b_shape = op.args[1].shape if isinstance(op.args[1], np.ndarray) else None
                
                if a_shape and b_shape and len(a_shape) == 2 and len(b_shape) == 2:
                    m, k = a_shape
                    n = b_shape[1]
                    
                    if needs_allocation:
                        lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                        allocated_vars.add(result_var)
                    
                    # NumPy optimizations for matrix multiplication:
                    # 1. Ensure arrays are C-contiguous (BLAS requires contiguous memory) - only if needed
                    # 2. Use proper leading dimensions (lda, ldb, ldc)
                    # 3. Ensure memory alignment for optimal SIMD performance
                    
                    # Check if we need to copy arrays for contiguity
                    # NumPy optimization: only copy if arrays are actually non-contiguous
                    # For input arrays, check if they're already contiguous
                    a_contiguous_var = f"a_contiguous_{op.op_id}" if input_vars[0] != "NULL" else None
                    b_contiguous_var = f"b_contiguous_{op.op_id}" if input_vars[1] != "NULL" else None
                    
                    # Check if input arrays are already contiguous
                    a_needs_copy = False
                    b_needs_copy = False
                    
                    if input_vars[0] != "NULL" and input_vars[0] in self.input_vars:
                        # Check if the input array is already C-contiguous
                        # We can check this from the input_arrays dict
                        if input_vars[0] in self.input_arrays:
                            arr = self.input_arrays[input_vars[0]]
                            if isinstance(arr, np.ndarray):
                                # Check if array is C-contiguous and properly aligned
                                # NumPy arrays are C-contiguous if strides match row-major layout
                                expected_stride = arr.itemsize
                                is_contiguous = arr.flags['C_CONTIGUOUS']
                                # Also check if it's already aligned (pointer is 32-byte aligned)
                                # For now, we'll copy if not C-contiguous, but could optimize further
                                a_needs_copy = not is_contiguous
                        else:
                            # If we don't have the array info, assume it might be non-contiguous
                            # (safer to copy, but adds overhead)
                            a_needs_copy = True
                    
                    if input_vars[1] != "NULL" and input_vars[1] in self.input_vars:
                        if input_vars[1] in self.input_arrays:
                            arr = self.input_arrays[input_vars[1]]
                            if isinstance(arr, np.ndarray):
                                is_contiguous = arr.flags['C_CONTIGUOUS']
                                b_needs_copy = not is_contiguous
                        else:
                            b_needs_copy = True
                    
                    # Only create contiguous copies if actually needed (NumPy optimization)
                    if input_vars[0] != "NULL" and input_vars[0] in self.input_vars:
                        if a_needs_copy:
                            # Input array needs copying - create contiguous copy for BLAS
                            # Only declare if not already declared (avoid redefinition from duplicate nodes)
                            if a_contiguous_var not in declared_contiguous_vars:
                                lines.append(f"    // NumPy optimization: ensure A is C-contiguous for optimal BLAS performance")
                                lines.append(f"    double* {a_contiguous_var} = (double*)aligned_alloc(32, {m * k} * sizeof(double));")
                                lines.append(f"    if (!{a_contiguous_var}) {{")
                                lines.append(f"        {a_contiguous_var} = (double*)malloc({m * k} * sizeof(double));")
                                lines.append(f"    }}")
                                declared_contiguous_vars.add(a_contiguous_var)
                            lines.append(f"    // Copy to contiguous memory (row-major)")
                            lines.append(f"    for (int i = 0; i < {m}; i++) {{")
                            lines.append(f"        for (int j = 0; j < {k}; j++) {{")
                            lines.append(f"            {a_contiguous_var}[i * {k} + j] = {input_vars[0]}[i * {k} + j];")
                            lines.append(f"        }}")
                            lines.append(f"    }}")
                            a_ptr = a_contiguous_var
                        else:
                            # Array is already contiguous - use directly (NumPy optimization)
                            lines.append(f"    // Array A is already C-contiguous - using directly (no copy needed)")
                            lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                            a_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                    else:
                        # Intermediate array - should already be contiguous from our allocation
                        if input_vars[0] != "NULL":
                            lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                            a_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                        else:
                            a_ptr = input_vars[0]
                    
                    if input_vars[1] != "NULL" and input_vars[1] in self.input_vars:
                        if b_needs_copy:
                            # Input array needs copying - create contiguous copy for BLAS (only if needed)
                            # Only declare if not already declared (avoid redefinition from duplicate nodes)
                            if b_contiguous_var not in declared_contiguous_vars:
                                lines.append(f"    // NumPy optimization: ensure B is C-contiguous for optimal BLAS performance")
                                lines.append(f"    double* {b_contiguous_var} = (double*)aligned_alloc(32, {k * n} * sizeof(double));")
                                lines.append(f"    if (!{b_contiguous_var}) {{")
                                lines.append(f"        {b_contiguous_var} = (double*)malloc({k * n} * sizeof(double));")
                                lines.append(f"    }}")
                                declared_contiguous_vars.add(b_contiguous_var)
                            lines.append(f"    // Copy to contiguous memory (row-major)")
                            lines.append(f"    for (int i = 0; i < {k}; i++) {{")
                            lines.append(f"        for (int j = 0; j < {n}; j++) {{")
                            lines.append(f"            {b_contiguous_var}[i * {n} + j] = {input_vars[1]}[i * {n} + j];")
                            lines.append(f"        }}")
                            lines.append(f"    }}")
                            b_ptr = b_contiguous_var
                        else:
                            # Array is already contiguous - use directly (NumPy optimization)
                            lines.append(f"    // Array B is already C-contiguous - using directly (no copy needed)")
                            lines.extend(self._declare_restrict_ptr(input_vars[1], True, declared_restrict_ptrs))
                            b_ptr = f"{input_vars[1]}_r" if input_vars[1] in self.input_vars else input_vars[1]
                    else:
                        # Intermediate array - should already be contiguous from our allocation
                        if input_vars[1] != "NULL":
                            lines.extend(self._declare_restrict_ptr(input_vars[1], True, declared_restrict_ptrs))
                            b_ptr = f"{input_vars[1]}_r" if input_vars[1] in self.input_vars else input_vars[1]
                        else:
                            b_ptr = input_vars[1]
                    
                    # Use BLAS (cblas_dgemm) - this is what NumPy uses
                    # BLAS is highly optimized and will outperform our manual implementation
                    # Always use BLAS for matrix multiplication
                    # Initialize result to zero (BLAS can accumulate, so we start fresh)
                    lines.append(f"    // Initialize result matrix to zero")
                    lines.append(f"    memset({result_var}, 0, {m * n} * sizeof(double));")
                    lines.append("")
                    lines.append(f"    // Use BLAS (cblas_dgemm) for optimized matrix multiplication")
                    lines.append(f"    // This matches NumPy's implementation and provides optimal performance")
                    lines.append(f"    // C = alpha * A @ B + beta * C")
                    lines.append(f"    // CblasRowMajor: matrices stored in row-major order (like NumPy)")
                    lines.append(f"    // CblasNoTrans: no transpose")
                    lines.append(f"    // Leading dimensions (lda, ldb, ldc) are the number of columns for row-major")
                    lines.append(f"    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,")
                    lines.append(f"                  {m}, {n}, {k},")  # M, N, K
                    lines.append(f"                  1.0, {a_ptr}, {k},")  # alpha, A, lda (leading dimension = k for row-major)
                    lines.append(f"                  {b_ptr}, {n},")  # B, ldb (leading dimension = n for row-major)
                    lines.append(f"                  0.0, {result_var}, {n});")  # beta, C, ldc (leading dimension = n for row-major)
                    
                    # Don't cleanup contiguous arrays here - they might be reused
                    # Cleanup will happen at the end of the function if needed
                    # (Actually, we should cleanup, but only if not reused - for now, skip cleanup to avoid use-after-free)
                else:
                    # Fallback for non-2D or unknown shapes
                    lines.append(f"    // TODO: Handle non-2D matrix multiplication")
                    if needs_allocation:
                        size = np.prod(op.result.shape)
                        lines.append(self._generate_array_allocation(result_var, op.result.shape, op.result.dtype))
                        allocated_vars.add(result_var)
                        lines.append(f"    memset({result_var}, 0, {size} * sizeof(double));")
            elif op.result.shape and len(op.result.shape) == 1:
                # Vector dot product (1D case)
                size = op.args[0].shape[0] if isinstance(op.args[0], np.ndarray) else 1
                if needs_allocation:
                    lines.append(f"    {result_var} = 0.0;")
                else:
                    lines.append(f"    {result_var} = 0.0;")
                
                # Use restrict pointers
                if input_vars[0] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[0], True, declared_restrict_ptrs))
                    a_ptr = f"{input_vars[0]}_r" if input_vars[0] in self.input_vars else input_vars[0]
                else:
                    a_ptr = input_vars[0]
                
                if input_vars[1] != "NULL":
                    lines.extend(self._declare_restrict_ptr(input_vars[1], True, declared_restrict_ptrs))
                    b_ptr = f"{input_vars[1]}_r" if input_vars[1] in self.input_vars else input_vars[1]
                else:
                    b_ptr = input_vars[1]
                
                # Optimized dot product with SIMD
                loop_body = f"{result_var} += {a_ptr}[i] * {b_ptr}[i];"
                lines.extend(self._generate_optimized_loop(size, loop_body))
            else:
                # Scalar dot product
                lines.append(f"    {result_var} = {input_vars[0]} * {input_vars[1]};")
        
        else:
            # Generic fallback (variable already declared upfront)
            lines.append(f"    // TODO: Implement {op.op_name}")
            # Only assign if variable is a scalar, not an array
            if result_var not in allocated_vars:
                # It's a scalar
                lines.append(f"    {result_var} = 0.0;")
            else:
                # It's an array - can't assign scalar to array pointer
                # Just leave it uninitialized or set first element
                lines.append(f"    // Variable {result_var} is an array, skipping assignment")
        
        return lines
    
    def generate(self, output_var_names: List[str] = None) -> str:
        """Generate complete C code."""
        lines = []
        
        # Header includes
        for include in sorted(self.includes):
            # If include contains newlines, split it into multiple lines
            if '\n' in include:
                lines.extend(include.split('\n'))
            else:
                lines.append(include)
        lines.append("")
        
        # Function signature
        # Map input variables to inputs array
        input_mapping = {}
        for i, var_name in enumerate(self.input_vars):
            input_mapping[var_name] = f"inputs[{i}]"
        
        lines.append("void preprocess(double** inputs, int num_inputs, double** outputs, int num_outputs) {")
        lines.append("")
        
        # Thread affinity is set via environment variables (OMP_PROC_BIND, OMP_PLACES)
        # This is more portable than using omp_set_proc_bind which may not be available
        # We'll set these in the benchmark runner for optimal performance
        
        # Map input arrays
        for var_name in self.input_vars:
            if var_name in input_mapping:
                lines.append(f"    double* {var_name} = {input_mapping[var_name]};")
        
        lines.append("")
        
        # Process nodes in topological order
        sorted_nodes = self.graph.topological_sort()
        
        # Track allocated variables to avoid redefinition (arrays and scalars)
        allocated_vars: Set[str] = set()
        declared_scalars: Set[str] = set()  # Track scalar variable declarations
        declared_restrict_ptrs: Set[str] = set()  # Track restrict pointer declarations
        
        # Generate operation code with loop fusion optimization
        # First pass: detect normalization patterns (Numba's key optimization)
        processed_ops: Set[int] = set()
        declared_restrict_ptrs: Set[str] = set()
        fused_chains: List[List[GraphNode]] = []
        normalization_info = None
        
        # Detect normalization pattern first (highest priority optimization)
        normalization_info = self._detect_normalization_pattern(sorted_nodes)
        if normalization_info:
            # Mark all normalization nodes as processed
            processed_ops.add(normalization_info['mean_node'].operation.op_id)
            processed_ops.add(normalization_info['std_node'].operation.op_id)
            processed_ops.add(normalization_info['subtract_node'].operation.op_id)
            processed_ops.add(normalization_info['divide_node'].operation.op_id)
        
        # Find all fusible chains BEFORE allocating variables
        # This way we can skip allocating intermediate arrays for fused operations
        for node in sorted_nodes:
            op_id = node.operation.op_id
            if op_id in processed_ops:
                continue
            
            # Check if this node starts a fusible chain
            # It can start a chain if:
            # 1. It's a fusible operation
            # 2. It produces an array
            # 3. It has exactly one input (which can be an input variable or another operation)
            if (node.operation.op_name in self.fusible_ops and
                isinstance(node.operation.result, np.ndarray) and
                len(node.inputs) <= 1):  # Can have 0 inputs (input var) or 1 input (from another op)
                chain = self._find_fusible_chain(node, processed_ops)
                if len(chain) > 1:  # Only fuse if we have 2+ operations
                    fused_chains.append(chain)
                    # Mark all nodes in chain as processed
                    for chain_node in chain:
                        processed_ops.add(chain_node.operation.op_id)
        
        # Allocate variables for intermediate array results (before generating code)
        # Skip intermediate arrays in fused chains (only allocate final result)
        # Also skip intermediate arrays in normalization pattern
        for node in sorted_nodes:
            op = node.operation
            op_id = op.op_id
            
            # Skip intermediate nodes in fused chains (only allocate final result)
            is_intermediate_in_fused_chain = False
            for chain in fused_chains:
                if node in chain and node != chain[-1]:  # Not the last node in chain
                    is_intermediate_in_fused_chain = True
                    break
            
            if is_intermediate_in_fused_chain:
                continue  # Skip allocation for intermediate fused operations
            
            # Skip intermediate nodes in normalization pattern (only allocate final result)
            if normalization_info:
                if (node == normalization_info['mean_node'] or
                    node == normalization_info['std_node'] or
                    node == normalization_info['subtract_node']):
                    # Only allocate the divide node's result (final output)
                    continue
            
            if isinstance(op.result, np.ndarray):
                # Check if this is not an input array
                if id(op.result) not in self.array_to_var_map:
                    var_name = self._get_var_name(op.op_id)
                    if var_name not in allocated_vars:
                        shape = op.result.shape
                        lines.append(self._generate_array_allocation(var_name, shape, op.result.dtype))
                        allocated_vars.add(var_name)
            else:
                # Declare scalar variables upfront (but skip normalization intermediates)
                if normalization_info:
                    if (node == normalization_info['mean_node'] or
                        node == normalization_info['std_node']):
                        continue  # These are computed in fused normalization
                
                var_name = self._get_var_name(op.op_id)
                if var_name not in declared_scalars:
                    lines.append(f"    double {var_name};")
                    declared_scalars.add(var_name)
        
        lines.append("")
        
        # Generate fused normalization code first (highest priority)
        if normalization_info:
            norm_lines = self._generate_fused_normalization(normalization_info, declared_restrict_ptrs)
            if norm_lines:
                lines.extend(norm_lines)
                lines.append("")
        
        # Generate code for fused chains
        for chain in fused_chains:
            fused_lines = self._generate_fused_loop(chain, declared_restrict_ptrs)
            if fused_lines:
                lines.extend(fused_lines)
                lines.append("")
        
        # Track declared contiguous variables to avoid redefinition
        declared_contiguous_vars = set()
        
        # Generate code for remaining operations (non-fusible or already processed)
        # Track which operations we've already generated code for to avoid duplicates
        generated_ops: Set[int] = set()
        
        for node in sorted_nodes:
            op_id = node.operation.op_id
            if op_id in processed_ops:
                continue  # Skip operations that were fused
            if op_id in generated_ops:
                continue  # Skip operations we've already generated code for (avoid duplicates)
            generated_ops.add(op_id)
            
            op_lines = self._generate_operation_code(node, allocated_vars, declared_scalars, declared_restrict_ptrs, declared_contiguous_vars)
            lines.extend(op_lines)
            if op_lines:
                lines.append("")
        
        # Copy outputs - handle both arrays and scalars
        # For scalars, we need to allocate a single-element array or change output handling
        output_index = 0
        seen_outputs: Set[int] = set()  # Track which output nodes we've processed
        for node in self.graph.output_nodes:
            op = node.operation
            op_id = op.op_id
            
            # Only process each unique output once
            if op_id in seen_outputs:
                continue
            seen_outputs.add(op_id)
            
            output_var = self._get_var_name(op.op_id)
            
            if isinstance(op.result, np.ndarray):
                # Array output - directly assign pointer
                lines.append(f"    outputs[{output_index}] = {output_var};")
            else:
                # Scalar output - allocate a single-element array
                scalar_var = f"{output_var}_output"
                lines.append(f"    double* {scalar_var} = (double*)malloc(sizeof(double));")
                lines.append(f"    {scalar_var}[0] = {output_var};")
                lines.append(f"    outputs[{output_index}] = {scalar_var};")
            output_index += 1
        
        lines.append("}")
        
        return "\n".join(lines)


def generate_c_code(graph: ComputationalGraph, input_vars: List[str], 
                   output_vars: List[str] = None, input_arrays: Dict[str, np.ndarray] = None) -> str:
    """
    Generate C code from computational graph.
    
    Args:
        graph: Computational graph
        input_vars: List of input variable names
        output_vars: Optional list of output variable names
        input_arrays: Dictionary mapping input variable names to their arrays
        
    Returns:
        C code as string
    """
    generator = CCodeGenerator(graph, input_vars, input_arrays)
    return generator.generate(output_vars)

