"""
Lambda function analyzer for FastPPI.
Analyzes simple lambda functions and generates equivalent C code.
"""

import ast
import inspect
from typing import Optional, Dict, Any, List


class LambdaAnalyzer:
    """Analyzes lambda functions to generate C code."""
    
    def __init__(self):
        self.external_vars = {}  # Variables captured from outer scope
    
    def analyze(self, lambda_func, external_vars: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Analyze a lambda function and extract information for C code generation.
        
        Returns:
            Dict with keys: 'param_name', 'c_code', 'requires_null_check'
            None if lambda is too complex to translate
        """
        if external_vars:
            self.external_vars = external_vars
        
        try:
            # Try to get the source code of the lambda
            try:
                source = inspect.getsource(lambda_func).strip()
            except (OSError, TypeError):
                # If we can't get source (e.g., lambda defined in exec'd code),
                # try to decompile from bytecode
                return self._analyze_from_bytecode(lambda_func, external_vars)
            
            # Parse the lambda expression
            tree = ast.parse(source)
            
            # Find the Lambda node
            lambda_node = self._find_lambda_node(tree)
            if not lambda_node:
                return None
            
            # Extract parameter name (assuming single parameter)
            if len(lambda_node.args.args) != 1:
                return None  # Only support single-argument lambdas for now
            
            param_name = lambda_node.args.args[0].arg
            
            # Generate C code from the body
            c_code, requires_null_check = self._generate_c_from_ast(lambda_node.body, param_name)
            
            if c_code is None:
                return None
            
            return {
                'param_name': param_name,
                'c_code': c_code,
                'requires_null_check': requires_null_check,
                'external_vars': self.external_vars
            }
        except Exception as e:
            # If we can't analyze the lambda, return None
            print(f"Lambda analysis failed: {e}")
            return None
    
    def _analyze_from_bytecode(self, lambda_func, external_vars: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Analyze lambda from bytecode (comprehensive fallback method).
        Supports conditionals, comparisons, boolean operations, and function calls.
        """
        try:
            import dis
            
            # Get the bytecode
            code = lambda_func.__code__
            
            # Check if it's a single parameter lambda
            if code.co_argcount != 1:
                return None
            
            param_name = code.co_varnames[0]
            
            # Get instructions
            instructions = list(dis.get_instructions(code))
            
            # Build a stack-based interpreter to reconstruct the expression
            c_expr, requires_null = self._reconstruct_from_instructions(
                instructions, param_name, code.co_consts, code.co_names, external_vars or {}
            )
            
            if c_expr:
                return {
                    'param_name': param_name,
                    'c_code': c_expr,
                    'requires_null_check': requires_null,
                    'external_vars': external_vars or {}
                }
            
            return None
        except Exception as e:
            # Silently fail and return None - AST will be tried first anyway
            return None
    
    def _reconstruct_from_instructions(self, instructions, param_name: str, 
                                      constants: tuple, names: tuple, 
                                      external_vars: Dict[str, Any]) -> tuple:
        """
        Reconstruct C expression from Python bytecode instructions using stack-based evaluation.
        For ternary expressions, we need to track both branches.
        Returns: (c_expression, requires_null_check)
        """
        stack = []
        requires_null_check = False
        
        # Check if this is a ternary by looking for conditional jumps
        has_conditional = any(i.opname in ('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE') for i in instructions)
        
        if has_conditional:
            # Handle ternary expressions by simulating both branches
            return self._handle_ternary(instructions, param_name, constants, names, external_vars)
        
        for idx, instr in enumerate(instructions):
            opname = instr.opname
            arg = instr.argval
            
            # Load operations
            if opname == 'LOAD_FAST':
                if arg == param_name:
                    stack.append(param_name)
                    requires_null_check = True
                else:
                    stack.append(str(arg))
            
            elif opname == 'LOAD_CONST':
                if isinstance(arg, (int, float)):
                    stack.append(str(arg))
                elif isinstance(arg, bool):
                    stack.append("true" if arg else "false")
                elif arg is None:
                    stack.append("NAN")
                else:
                    stack.append(f"/* const:{arg} */")
            
            elif opname == 'LOAD_GLOBAL':
                if arg in external_vars:
                    val = external_vars[arg]
                    if isinstance(val, (int, float)):
                        stack.append(str(val))
                    else:
                        stack.append(arg)
                else:
                    stack.append(arg)
            
            elif opname == 'LOAD_ATTR':
                if stack:
                    obj_expr = stack.pop()
                    stack.append(f"{obj_expr}.{arg}")
            
            # Binary operations
            elif opname in ('BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_MULTIPLY', 
                          'BINARY_TRUE_DIVIDE', 'BINARY_FLOOR_DIVIDE', 'BINARY_MODULO'):
                if len(stack) >= 2:
                    right = stack.pop()
                    left = stack.pop()
                    op_map = {
                        'BINARY_ADD': '+', 'BINARY_SUBTRACT': '-',
                        'BINARY_MULTIPLY': '*', 'BINARY_TRUE_DIVIDE': '/',
                        'BINARY_FLOOR_DIVIDE': '/', 'BINARY_MODULO': '%'
                    }
                    op = op_map[opname]
                    stack.append(f"({left} {op} {right})")
            
            elif opname == 'BINARY_OP':  # Python 3.11+
                if len(stack) >= 2 and isinstance(arg, str):
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(f"({left} {arg} {right})")
            
            # Comparison operations
            elif opname == 'COMPARE_OP':
                if len(stack) >= 2:
                    right = stack.pop()
                    left = stack.pop()
                    op_map = {
                        '<': '<', '>': '>', '<=': '<=', '>=': '>=',
                        '==': '==', '!=': '!=',
                        'is': '==', 'is not': '!='
                    }
                    c_op = op_map.get(arg, arg)
                    stack.append(f"({left} {c_op} {right})")
            
            # Unary operations
            elif opname == 'UNARY_NOT':
                if stack:
                    operand = stack.pop()
                    stack.append(f"(!{operand})")
            
            elif opname == 'UNARY_NEGATIVE':
                if stack:
                    operand = stack.pop()
                    stack.append(f"(-{operand})")
            
            # Function calls
            elif opname in ('CALL_FUNCTION', 'CALL'):
                argc = arg if isinstance(arg, int) else 1
                if argc > len(stack):
                    return None, False
                
                args = []
                for _ in range(argc):
                    args.insert(0, stack.pop())
                
                if not stack:
                    return None, False
                func_expr = stack.pop()
                
                # Handle specific functions
                if 'pd.notnull' in func_expr or 'pd.notna' in func_expr:
                    if len(args) == 1:
                        stack.append(f"(!isnan({args[0]}))")
                        requires_null_check = True
                elif 'pd.isnull' in func_expr or 'pd.isna' in func_expr:
                    if len(args) == 1:
                        stack.append(f"isnan({args[0]})")
                        requires_null_check = True
                elif func_expr == 'abs' or func_expr.endswith('.abs'):
                    if len(args) == 1:
                        stack.append(f"fabs({args[0]})")
                else:
                    args_str = ', '.join(args)
                    stack.append(f"{func_expr}({args_str})")
            
            # Conditional jumps - reconstruct ternary
            elif opname in ('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE'):
                # This is a conditional - we'll handle it by continuing execution
                # and reconstructing the ternary later
                pass
            
            elif opname == 'RETURN_VALUE':
                if stack:
                    # Check if we have a conditional structure
                    result_expr = stack.pop()
                    
                    # If this is a ternary, try to reconstruct it from the full execution
                    # For now, return what we have
                    return result_expr, requires_null_check
                return None, False
        
        if stack:
            return stack.pop(), requires_null_check
        return None, False
    
    def _handle_ternary(self, instructions, param_name: str, constants: tuple, 
                       names: tuple, external_vars: Dict[str, Any]) -> tuple:
        """
        Handle ternary expressions (a if condition else b) from bytecode.
        Pattern: <condition_ops> POP_JUMP_IF_FALSE <true_branch> JUMP_FORWARD <false_branch> RETURN_VALUE
        """
        try:
            # Find the conditional jump
            cond_jump_idx = None
            for i, instr in enumerate(instructions):
                if instr.opname in ('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE'):
                    cond_jump_idx = i
                    break
            
            if cond_jump_idx is None:
                return None, False
            
            # Check for multiple POP_JUMP (short-circuit AND/OR)
            all_jumps = [(i, instr) for i, instr in enumerate(instructions) 
                        if instr.opname in ('POP_JUMP_IF_FALSE', 'POP_JUMP_IF_TRUE')]
            
            if len(all_jumps) > 1:
                # Handle short-circuit AND: multiple jumps to same false branch
                # Build condition parts
                cond_parts = []
                last_jump_idx = 0
                for jump_idx, jump_instr in all_jumps:
                    # Execute from last jump to this jump to get condition part
                    part_stack = []
                    for i in range(last_jump_idx, jump_idx):
                        part_stack, _ = self._execute_instruction(
                            instructions[i], part_stack, param_name, constants, names, external_vars
                        )
                    if part_stack:
                        cond_parts.append(part_stack.pop())
                    last_jump_idx = jump_idx + 1
                
                # Combine with AND
                condition = ' && '.join(f"({p})" for p in cond_parts if p)
                requires_null = any('isnan' in p or param_name in p for p in cond_parts)
                cond_jump_idx = all_jumps[-1][0]  # Use last jump
            else:
                # Single conditional - execute up to the conditional to get the condition expression
                cond_stack = []
                requires_null = False
                for i in range(cond_jump_idx):
                    instr = instructions[i]
                    cond_stack, requires_null = self._execute_instruction(
                        instr, cond_stack, param_name, constants, names, external_vars
                    )
                
                if not cond_stack:
                    return None, False
                condition = cond_stack.pop()
            
            # Find where true branch ends (JUMP_FORWARD or RETURN_VALUE)
            true_branch_end = None
            false_branch_start = instructions[cond_jump_idx].arg if hasattr(instructions[cond_jump_idx], 'arg') else None
            
            for i in range(cond_jump_idx + 1, len(instructions)):
                if instructions[i].opname in ('JUMP_FORWARD', 'JUMP_ABSOLUTE'):
                    true_branch_end = i
                    break
                elif instructions[i].opname == 'RETURN_VALUE':
                    true_branch_end = i
                    break
            
            if true_branch_end is None:
                return None, False
            
            # Execute true branch
            true_stack = []
            for i in range(cond_jump_idx + 1, true_branch_end):
                instr = instructions[i]
                true_stack, _ = self._execute_instruction(
                    instr, true_stack, param_name, constants, names, external_vars
                )
            
            true_val = true_stack.pop() if true_stack else "0"
            
            # Execute false branch
            false_stack = []
            false_start_offset = instructions[cond_jump_idx].arg if hasattr(instructions[cond_jump_idx], 'arg') else None
            
            # Find the instruction with the matching offset (the false branch target)
            false_start_idx = true_branch_end + 1
            if false_start_offset is not None:
                # Find the instruction with the matching offset
                for i, instr in enumerate(instructions):
                    if instr.offset == false_start_offset:
                        false_start_idx = i
                        break
                # If we couldn't find it by offset, use the instruction after true_branch_end
                if false_start_idx == true_branch_end + 1 and false_start_idx >= len(instructions):
                    false_start_idx = true_branch_end + 1
            
            # Execute false branch instructions until RETURN_VALUE
            for i in range(false_start_idx, len(instructions)):
                instr = instructions[i]
                if instr.opname == 'RETURN_VALUE':
                    break
                false_stack, _ = self._execute_instruction(
                    instr, false_stack, param_name, constants, names, external_vars
                )
            
            # Get false branch value - if stack is empty, default to "0"
            false_val = false_stack.pop() if false_stack else "0"
            
            # Safety check: if false_val is somehow the same as true_val, something went wrong
            # This can happen if the false branch execution didn't work correctly
            if false_val == true_val and true_val != "0":
                # Try to find the false branch value by looking at the instruction after true_branch_end
                if true_branch_end + 1 < len(instructions):
                    next_instr = instructions[true_branch_end + 1]
                    if next_instr.opname == 'LOAD_CONST':
                        false_val = str(next_instr.argval) if next_instr.argval is not None else "0"
            
            # Construct ternary
            result = f"({condition} ? {true_val} : {false_val})"
            return result, requires_null
            
        except Exception as e:
            return None, False
    
    def _execute_instruction(self, instr, stack, param_name, constants, names, external_vars):
        """Execute a single bytecode instruction and return updated stack."""
        requires_null = False
        opname = instr.opname
        arg = instr.argval
        
        if opname == 'LOAD_FAST':
            if arg == param_name:
                stack.append(param_name)
                requires_null = True
            else:
                stack.append(str(arg))
        
        elif opname == 'LOAD_CONST':
            if isinstance(arg, (int, float)):
                stack.append(str(arg))
            elif isinstance(arg, bool):
                stack.append("true" if arg else "false")
            elif arg is None:
                stack.append("NAN")
        
        elif opname == 'LOAD_GLOBAL':
            if arg in external_vars:
                val = external_vars[arg]
                if isinstance(val, (int, float)):
                    stack.append(str(val))
                else:
                    stack.append(arg)
            else:
                stack.append(arg)
        
        elif opname == 'LOAD_ATTR':
            if stack:
                obj_expr = stack.pop()
                stack.append(f"{obj_expr}.{arg}")
        
        elif opname in ('BINARY_ADD', 'BINARY_SUBTRACT', 'BINARY_MULTIPLY', 
                       'BINARY_TRUE_DIVIDE', 'BINARY_FLOOR_DIVIDE'):
            if len(stack) >= 2:
                right = stack.pop()
                left = stack.pop()
                op_map = {
                    'BINARY_ADD': '+', 'BINARY_SUBTRACT': '-',
                    'BINARY_MULTIPLY': '*', 'BINARY_TRUE_DIVIDE': '/',
                    'BINARY_FLOOR_DIVIDE': '/'
                }
                op = op_map.get(opname, '+')
                stack.append(f"({left} {op} {right})")
        
        elif opname == 'BINARY_OP':  # Python 3.11+
            if len(stack) >= 2:
                right = stack.pop()
                left = stack.pop()
                # Python 3.12 uses numeric codes for binary operations
                op_map = {
                    0: '+', 1: '&', 2: '//', 3: '<<', 4: '@', 5: '*',
                    6: '%', 7: '|', 8: '**', 9: '>>', 10: '-', 11: '/',
                    12: '^'
                }
                op = op_map.get(arg, str(arg)) if isinstance(arg, int) else arg
                stack.append(f"({left} {op} {right})")
        
        elif opname == 'COMPARE_OP':
            if len(stack) >= 2:
                right = stack.pop()
                left = stack.pop()
                op_map = {
                    '<': '<', '>': '>', '<=': '<=', '>=': '>=',
                    '==': '==', '!=': '!='
                }
                c_op = op_map.get(arg, arg)
                stack.append(f"({left} {c_op} {right})")
        
        elif opname == 'UNARY_NOT':
            if stack:
                operand = stack.pop()
                stack.append(f"(!{operand})")
        
        elif opname in ('CALL_FUNCTION', 'CALL'):
            argc = arg if isinstance(arg, int) else 1
            if argc <= len(stack):
                args = [stack.pop() for _ in range(argc)]
                args.reverse()
                
                if stack:
                    func_expr = stack.pop()
                    
                    # Handle NumPy functions (np.log, np.sqrt, etc.)
                    if isinstance(func_expr, str):
                        if func_expr == 'np.log' or func_expr.endswith('.log'):
                            if len(args) == 1:
                                stack.append(f"log({args[0]})")
                        elif func_expr == 'np.sqrt' or func_expr.endswith('.sqrt'):
                            if len(args) == 1:
                                stack.append(f"sqrt({args[0]})")
                        elif func_expr == 'np.exp' or func_expr.endswith('.exp'):
                            if len(args) == 1:
                                stack.append(f"exp({args[0]})")
                        elif func_expr == 'np.abs' or func_expr.endswith('.abs'):
                            if len(args) == 1:
                                stack.append(f"fabs({args[0]})")
                        elif 'pd.notnull' in func_expr or 'pd.notna' in func_expr:
                            if len(args) == 1:
                                stack.append(f"(!isnan({args[0]}))")
                                requires_null = True
                        elif 'pd.isnull' in func_expr or 'pd.isna' in func_expr:
                            if len(args) == 1:
                                stack.append(f"isnan({args[0]})")
                                requires_null = True
                        elif func_expr == 'abs':
                            if len(args) == 1:
                                stack.append(f"fabs({args[0]})")
                        else:
                            args_str = ', '.join(args)
                            stack.append(f"{func_expr}({args_str})")
                    else:
                        args_str = ', '.join(args)
                        stack.append(f"{func_expr}({args_str})")
        
        return stack, requires_null
    
    def _find_lambda_node(self, tree):
        """Find the Lambda node in the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda):
                return node
        return None
    
    def _generate_c_from_ast(self, node, param_name: str) -> tuple:
        """
        Generate C code from an AST node.
        Returns: (c_code, requires_null_check)
        """
        requires_null_check = False
        
        if isinstance(node, ast.Name):
            # Variable reference
            if node.id == param_name:
                return param_name, False
            elif node.id in self.external_vars:
                # External variable
                return str(self.external_vars[node.id]), False
            else:
                return node.id, False
        
        elif isinstance(node, ast.Constant) or isinstance(node, ast.Num):
            # Constant value
            value = node.value if isinstance(node, ast.Constant) else node.n
            return str(value), False
        
        elif isinstance(node, ast.BinOp):
            # Binary operation: x + y, x - y, etc.
            left, left_null = self._generate_c_from_ast(node.left, param_name)
            right, right_null = self._generate_c_from_ast(node.right, param_name)
            requires_null_check = left_null or right_null
            
            op_map = {
                ast.Add: '+',
                ast.Sub: '-',
                ast.Mult: '*',
                ast.Div: '/',
                ast.Mod: '%',
                ast.Pow: '**',  # Will need pow() function
            }
            
            op = op_map.get(type(node.op))
            if op == '**':
                return f"pow({left}, {right})", requires_null_check
            elif op:
                return f"({left} {op} {right})", requires_null_check
            else:
                return None, False
        
        elif isinstance(node, ast.Compare):
            # Comparison: x > 5, x == 0, etc.
            left, left_null = self._generate_c_from_ast(node.left, param_name)
            requires_null_check = left_null
            
            # Handle single comparison (most common case)
            if len(node.ops) == 1 and len(node.comparators) == 1:
                right, right_null = self._generate_c_from_ast(node.comparators[0], param_name)
                requires_null_check = requires_null_check or right_null
                
                op_map = {
                    ast.Gt: '>',
                    ast.Lt: '<',
                    ast.GtE: '>=',
                    ast.LtE: '<=',
                    ast.Eq: '==',
                    ast.NotEq: '!=',
                }
                
                op = op_map.get(type(node.ops[0]))
                if op:
                    return f"({left} {op} {right})", requires_null_check
            
            return None, False
        
        elif isinstance(node, ast.IfExp):
            # Ternary expression: a if condition else b
            test, test_null = self._generate_c_from_ast(node.test, param_name)
            body, body_null = self._generate_c_from_ast(node.body, param_name)
            orelse, orelse_null = self._generate_c_from_ast(node.orelse, param_name)
            
            requires_null_check = test_null or body_null or orelse_null
            
            if test and body and orelse:
                return f"({test} ? {body} : {orelse})", requires_null_check
            
            return None, False
        
        elif isinstance(node, ast.UnaryOp):
            # Unary operation: -x, not x
            operand, operand_null = self._generate_c_from_ast(node.operand, param_name)
            requires_null_check = operand_null
            
            if isinstance(node.op, ast.USub):
                return f"(-{operand})", requires_null_check
            elif isinstance(node.op, ast.Not):
                return f"(!{operand})", requires_null_check
            
            return None, False
        
        elif isinstance(node, ast.Call):
            # Function call
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # Handle pd.notnull(x) or similar
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'pd':
                    func_name = f"pd_{node.func.attr}"
            
            if func_name == 'abs':
                if len(node.args) == 1:
                    arg, arg_null = self._generate_c_from_ast(node.args[0], param_name)
                    return f"fabs({arg})", arg_null
            
            elif func_name == 'pd_notnull' or func_name == 'pd_notna':
                # pd.notnull(x) -> !isnan(x) for numeric values
                if len(node.args) == 1:
                    arg, arg_null = self._generate_c_from_ast(node.args[0], param_name)
                    if arg == param_name:
                        requires_null_check = True
                        return f"(!isnan({arg}))", True
                    else:
                        return f"(!isnan({arg}))", arg_null
            
            elif func_name == 'pd_isnull' or func_name == 'pd_isna':
                # pd.isnull(x) -> isnan(x) for numeric values
                if len(node.args) == 1:
                    arg, arg_null = self._generate_c_from_ast(node.args[0], param_name)
                    if arg == param_name:
                        requires_null_check = True
                        return f"isnan({arg})", True
                    else:
                        return f"isnan({arg})", arg_null
            
            return None, False
        
        elif isinstance(node, ast.BoolOp):
            # Boolean operation: x and y, x or y
            values = []
            for value in node.values:
                val, val_null = self._generate_c_from_ast(value, param_name)
                if val is None:
                    return None, False
                values.append(val)
                requires_null_check = requires_null_check or val_null
            
            if isinstance(node.op, ast.And):
                return f"({' && '.join(values)})", requires_null_check
            elif isinstance(node.op, ast.Or):
                return f"({' || '.join(values)})", requires_null_check
            
            return None, False
        
        return None, False


def analyze_lambda(lambda_func, external_vars: Dict[str, Any] = None) -> Optional[Dict]:
    """
    Convenience function to analyze a lambda.
    
    Args:
        lambda_func: The lambda function to analyze
        external_vars: Dictionary of external variables the lambda uses
    
    Returns:
        Analysis result dict or None if lambda is too complex
    """
    analyzer = LambdaAnalyzer()
    return analyzer.analyze(lambda_func, external_vars)

