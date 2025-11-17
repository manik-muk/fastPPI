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
                        import subprocess
                        try:
                            brew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
                            if brew_prefix:
                                cmd.extend(["-L", f"{brew_prefix}/lib"])
                        except (subprocess.CalledProcessError, FileNotFoundError):
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

