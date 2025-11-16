"""
Setup script for FastPPI package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="fastPPI",
    version="0.2.0",
    description="Fast Preprocessing Pipeline Interpreter - Convert Python preprocessing to optimized C binaries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FastPPI Contributors",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "pandas": ["pandas>=1.0.0"],
        "full": ["pandas>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "fastppi=fastPPI.main:main",
            "fastppi-analyze=fastPPI.analyze:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

