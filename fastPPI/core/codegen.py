"""
C code generator from computational graph.
Converts Python/NumPy operations to equivalent C code.
"""

from typing import List, Dict, Set
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
        
        # Add OpenMP for parallelization (optional, can be enabled via compiler flag)
        self.use_openmp = False  # Can be enabled if OpenMP is available
        if self.use_openmp:
            self.includes.add("#include <omp.h>")
        
        # Map input arrays
        if input_arrays:
            for var_name, arr in input_arrays.items():
                if isinstance(arr, np.ndarray):
                    self.array_to_var_map[id(arr)] = var_name
        
        # Element-wise operations that can be fused (single input, single output, same shape)
        self.fusible_ops = {
            'exp', 'log', 'sqrt', 'abs', 'round', 
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            'sinh', 'cosh', 'tanh'
        }
        
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
        Generate an optimized loop with SIMD hints and restrict pointers.
        Similar to Numba's loop optimization techniques.
        
        Args:
            size: Loop size
            loop_body: The body of the loop (without braces)
            use_restrict: Whether to use restrict pointers
            
        Returns:
            List of code lines for the optimized loop
        """
        lines = []
        size_int = int(size)
        
        # Add SIMD vectorization hints for larger loops (like Numba does)
        if size_int > 4:
            lines.append("    #pragma GCC ivdep")  # Ignore vector dependencies (enables vectorization)
            lines.append("    #pragma clang loop vectorize(enable) interleave(enable)")
            # Unroll hints for small loops
            if size_int < 100:
                lines.append("    #pragma clang loop unroll_count(4)")
        
        # Generate the loop
        lines.append(f"    for (int i = 0; i < {size_int}; i++) {{")
        lines.append(f"        {loop_body}")
        lines.append(f"    }}")
        
        return lines
    
    def _generate_array_allocation(self, var_name: str, shape: tuple, dtype) -> str:
        """Generate code to allocate an array."""
        if not shape:
            return ""
        
        size = 1
        for dim in shape:
            size *= dim
        
        return f"    double* {var_name} = (double*)malloc({size} * sizeof(double));"
    
    def _generate_operation_code(self, node: GraphNode, allocated_vars: Set[str], declared_scalars: Set[str], declared_restrict_ptrs: Set[str] = None) -> List[str]:
        """Generate C code for a single operation."""
        if declared_restrict_ptrs is None:
            declared_restrict_ptrs = set()
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
            lines.append(include)
        lines.append("")
        
        # Function signature
        # Map input variables to inputs array
        input_mapping = {}
        for i, var_name in enumerate(self.input_vars):
            input_mapping[var_name] = f"inputs[{i}]"
        
        lines.append("void preprocess(double** inputs, int num_inputs, double** outputs, int num_outputs) {")
        lines.append("")
        
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
        # First pass: identify and fuse element-wise operation chains
        processed_ops: Set[int] = set()
        declared_restrict_ptrs: Set[str] = set()
        fused_chains: List[List[GraphNode]] = []
        
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
            
            if isinstance(op.result, np.ndarray):
                # Check if this is not an input array
                if id(op.result) not in self.array_to_var_map:
                    var_name = self._get_var_name(op.op_id)
                    if var_name not in allocated_vars:
                        shape = op.result.shape
                        lines.append(self._generate_array_allocation(var_name, shape, op.result.dtype))
                        allocated_vars.add(var_name)
            else:
                # Declare scalar variables upfront
                var_name = self._get_var_name(op.op_id)
                if var_name not in declared_scalars:
                    lines.append(f"    double {var_name};")
                    declared_scalars.add(var_name)
        
        lines.append("")
        
        # Generate code for fused chains
        for chain in fused_chains:
            fused_lines = self._generate_fused_loop(chain, declared_restrict_ptrs)
            if fused_lines:
                lines.extend(fused_lines)
                lines.append("")
        
        # Generate code for remaining operations (non-fusible or already processed)
        for node in sorted_nodes:
            op_id = node.operation.op_id
            if op_id in processed_ops:
                continue  # Skip operations that were fused
            
            op_lines = self._generate_operation_code(node, allocated_vars, declared_scalars, declared_restrict_ptrs)
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

