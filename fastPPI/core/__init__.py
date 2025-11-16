"""
Core compilation functionality for FastPPI.
"""

from .compiler import compile_to_binary, compile_with_clang
from .codegen import generate_c_code, CCodeGenerator
from .graph import build_computational_graph, ComputationalGraph

__all__ = [
    "compile_to_binary",
    "compile_with_clang",
    "generate_c_code",
    "CCodeGenerator",
    "build_computational_graph",
    "ComputationalGraph",
]

