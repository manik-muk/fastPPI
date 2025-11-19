"""
Compilation utilities for converting C code to optimized binaries.
Uses clang with -O3 optimization.
"""

import subprocess
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def compile_with_clang(c_code: str, output_path: str, 
                      optimization: str = "-O3",
                      additional_flags: Optional[list] = None) -> Tuple[bool, str]:
    """
    Compile C code to binary using clang.
    
    Args:
        c_code: C source code as string
        output_path: Path for output binary
        optimization: Optimization flag (default: -O3)
        additional_flags: Additional compiler flags
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    # Add proper extension for shared library if not present
    output_path_obj = Path(output_path)
    if not output_path_obj.suffix:
        if sys.platform == "darwin":
            output_path = str(output_path_obj) + ".dylib"
        elif sys.platform.startswith("linux"):
            output_path = str(output_path_obj) + ".so"
        elif sys.platform == "win32":
            output_path = str(output_path_obj) + ".dll"
    
    # Create temporary C file
    temp_dir = tempfile.mkdtemp()
    c_file = os.path.join(temp_dir, "preprocess.c")
    
    try:
        # Write C code to file
        with open(c_file, 'w') as f:
            f.write(c_code)
        
        # Build clang command
        # Compile as shared library (.so/.dylib/.dll)
        cmd = ["clang", optimization]
        
        # Add CPU-specific optimizations for better performance
        # -march=native enables all CPU-specific optimizations (SIMD, instruction sets)
        cmd.append("-march=native")
        
        # Check for BLAS and OpenMP in the code
        with open(c_file, 'r') as f:
            c_content = f.read()
            
            # Always link BLAS (we always use it for matrix multiplication)
            if 'cblas_dgemm' in c_content:
                # Add BLAS library linking
                # Try to find BLAS library (same logic as codegen)
                # sys and os are already imported at module level
                
                blas_lib = None
                # Check conda/miniconda environment
                conda_prefix = os.environ.get('CONDA_PREFIX', '')
                if conda_prefix:
                    for lib_name in ['libopenblas.dylib', 'libcblas.dylib', 'libblas.dylib']:
                        lib_path = f"{conda_prefix}/lib/{lib_name}"
                        if os.path.exists(lib_path):
                            blas_lib = lib_path
                            break
                
                if not blas_lib:
                    # Try Python prefix
                    python_prefix = sys.prefix
                    for lib_name in ['libopenblas.dylib', 'libcblas.dylib']:
                        lib_path = f"{python_prefix}/lib/{lib_name}"
                        if os.path.exists(lib_path):
                            blas_lib = lib_path
                            break
                
                if not blas_lib:
                    # Try to find via numpy
                    try:
                        import numpy as np
                        numpy_path = np.__file__
                        numpy_dir = os.path.dirname(numpy_path)
                        lib_dir = os.path.join(numpy_dir, '..', '..', 'lib')
                        lib_dir = os.path.abspath(lib_dir)
                        
                        for lib_file in os.listdir(lib_dir):
                            if ('openblas' in lib_file.lower() or 'cblas' in lib_file.lower()) and \
                               (lib_file.endswith('.dylib') or lib_file.endswith('.so')):
                                blas_lib = os.path.join(lib_dir, lib_file)
                                break
                    except:
                        pass
                
                if blas_lib:
                    # Add library path and link against BLAS
                    lib_dir = os.path.dirname(blas_lib)
                    lib_name = os.path.basename(blas_lib)
                    # Extract library name without extension and 'lib' prefix
                    if lib_name.startswith('lib') and (lib_name.endswith('.dylib') or lib_name.endswith('.so')):
                        lib_base = lib_name[3:].split('.')[0]  # Remove 'lib' prefix and extension
                        cmd.extend(["-L", lib_dir])
                        cmd.append(f"-l{lib_base}")
                        # Add rpath on macOS so library can be found at runtime
                        if sys.platform == "darwin":
                            cmd.extend(["-Wl,-rpath", lib_dir])
                    else:
                        # Link directly to the library file
                        cmd.append(blas_lib)
                        if sys.platform == "darwin":
                            cmd.extend(["-Wl,-rpath", lib_dir])
                else:
                    # Try standard library names
                    cmd.append("-lopenblas")
                    cmd.append("-lcblas")
            
            # Check if OpenMP is used
            if '#include <omp.h>' in c_content:
                # Add OpenMP flags
                if sys.platform == "darwin":
                    # macOS: use libomp (keg-only, needs explicit paths)
                    try:
                        import subprocess as sp
                        brew_prefix = sp.check_output(['brew', '--prefix'], text=True).strip()
                        libomp_path = f"{brew_prefix}/opt/libomp"
                        
                        # Check if libomp exists and add correct paths
                        if os.path.exists(f"{libomp_path}/include/omp.h"):
                            cmd.append("-Xpreprocessor")
                            cmd.append("-fopenmp")
                            cmd.extend(["-I", f"{libomp_path}/include"])
                            cmd.extend(["-L", f"{libomp_path}/lib"])
                            cmd.append("-lomp")
                        else:
                            # Fallback: try old location
                            cmd.append("-Xpreprocessor")
                            cmd.append("-fopenmp")
                            cmd.append("-lomp")
                            if brew_prefix:
                                cmd.extend(["-L", f"{brew_prefix}/lib"])
                    except (sp.CalledProcessError, FileNotFoundError, Exception):
                        # If we can't find libomp, skip OpenMP
                        pass
                else:
                    # Linux: standard OpenMP
                    cmd.append("-fopenmp")
        
        # Numba-style optimizations for maximum performance
        # -ffast-math: Enable aggressive floating-point optimizations (like Numba's fastmath)
        cmd.append("-ffast-math")
        
        # -funroll-loops: Unroll loops for better performance (similar to Numba's loop unrolling)
        cmd.append("-funroll-loops")
        
        # -ftree-vectorize: Enable automatic vectorization (SIMD)
        cmd.append("-ftree-vectorize")
        
        # -fomit-frame-pointer: Reduce overhead (when safe)
        cmd.append("-fomit-frame-pointer")
        
        # -finline-functions: Inline small functions (like Numba's inlining)
        cmd.append("-finline-functions")
        
        # -flto: Link-time optimization for better cross-module optimization
        # Note: This can slow down compilation but improves runtime performance
        # cmd.append("-flto")  # Commented out as it can cause issues with shared libraries
        
        # Additional optimization flags
        cmd.append("-fno-signed-zeros")  # Allow optimizations that ignore sign of zero
        cmd.append("-fno-trapping-math")  # Assume no floating-point exceptions
        cmd.append("-fassociative-math")  # Allow reassociation of floating-point operations
        
        if additional_flags:
            cmd.extend(additional_flags)
        
        # Check if we need to link pandas libraries
        # Look for includes in the C code
        c_lib_path = None
        c_include_path = None
        link_libs = ["-lm"]  # Math library
        
        # Try to find c_implementations directory relative to FastPPI
        fastppi_root = Path(__file__).parent.parent.parent
        c_impl_dir = fastppi_root / "c_implementations"
        if c_impl_dir.exists():
            c_lib_path = c_impl_dir / "lib"
            c_include_path = c_impl_dir / "include"
            
            # Check if pandas_c or string_c headers are included
            with open(c_file, 'r') as f:
                c_content = f.read()
                if 'pandas_c.h' in c_content:
                    link_libs.append("-lpandas_c")
                    # pandas_c library already links curl and jansson, but we may need them
                    # for direct linking if using static library
                    # Check if http_get_json is used (indicates need for curl/jansson)
                    if 'pandas_http_get_json' in c_content:
                        link_libs.append("-lcurl")
                        link_libs.append("-ljansson")
                        # Add Homebrew library path if available
                        try:
                            import subprocess as sp
                            brew_prefix = sp.check_output(['brew', '--prefix'], text=True).strip()
                            if brew_prefix:
                                cmd.extend(["-L", f"{brew_prefix}/lib"])
                        except (sp.CalledProcessError, FileNotFoundError):
                            pass
                if 'string_c.h' in c_content:
                    link_libs.append("-lstring_c")
            
            if link_libs:
                # Add library and include paths
                if c_lib_path.exists():
                    cmd.extend(["-L", str(c_lib_path)])
                if c_include_path.exists():
                    cmd.extend(["-I", str(c_include_path)])
        
        # For macOS, add rpath to library directory so @rpath can be resolved
        if sys.platform == "darwin" and c_lib_path and c_lib_path.exists():
            cmd.extend(["-Wl,-rpath", str(c_lib_path)])
        
        cmd.extend([
            "-shared",  # Create shared library
            "-fPIC",    # Position independent code
            "-o", output_path,
            c_file,
        ])
        cmd.extend(link_libs)
        
        # Compile
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            return False, result.stderr
        
        return True, ""
    
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"
    except Exception as e:
        return False, str(e)
    finally:
        # Cleanup temporary file
        try:
            os.remove(c_file)
            os.rmdir(temp_dir)
        except:
            pass


def compile_to_binary(c_code: str, output_binary: str, 
                     optimization: str = "-O3") -> Tuple[bool, str]:
    """
    Convenience function to compile C code to binary.
    
    Args:
        c_code: C source code as string
        output_binary: Path for output binary
        optimization: Optimization flag (default: -O3)
        
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    return compile_with_clang(c_code, output_binary, optimization)

