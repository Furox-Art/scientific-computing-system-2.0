"""Build script: declares the optional compiled accelerators.

Compiled kernels accelerate hot loops in linalg, integrate, signals, ml and
graph. Each is optional (``optional=True``) so installs without a C compiler
still work — cds2 falls back to the NumPy/SciPy equivalent at runtime.

OpenMP / SIMD selection
-----------------------
* Linux + GCC/Clang:  OpenMP enabled unless ``CDS_NO_OPENMP=1``. ARM64 gets
  NEON paths via ``__ARM_NEON``.
* macOS:              serial by default; set ``CDS_WITH_LIBOMP=1`` to link
  Homebrew ``libomp`` and enable OpenMP.
* Windows / MSVC:     serial (legacy OpenMP rejects the loop shapes used).
"""

import os
import sys

from setuptools import Extension, setup

extra_compile_args: list[str] = ["-O3"]
extra_link_args: list[str] = []

if os.environ.get("CDS_NO_OPENMP") != "1":
    if sys.platform == "linux":
        extra_compile_args += ["-fopenmp"]
        extra_link_args += ["-fopenmp"]
        # ARM64 NEON is only available on AArch64.
        if os.uname().machine.startswith("aarch64"):
            extra_compile_args += ["-march=armv8-a+fp+simd"]
    elif sys.platform == "darwin" and os.environ.get("CDS_WITH_LIBOMP") == "1":
        extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
        extra_link_args += ["-lomp"]
        if os.uname().machine.startswith("arm64"):
            extra_compile_args += ["-march=armv8-a+fp+simd"]

extensions: list[Extension] = []
if os.environ.get("CDS_PURE") != "1":
    common = dict(
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        optional=True,
    )
    extensions.extend(
        [
            Extension("cds2._fast_kmeans", sources=["src/cds2/src/_fast_kmeans.c"], **common),
            Extension("cds2._fast_pagerank", sources=["src/cds2/src/_fast_pagerank.c"], **common),
            Extension("cds2._fast_linop", sources=["src/cds2/src/_fast_linop.c"], **common),
            Extension("cds2._fast_integrate", sources=["src/cds2/src/_fast_integrate.c"], **common),
            Extension("cds2._fast_signal", sources=["src/cds2/src/_fast_signal.c"], **common),
        ]
    )

setup(ext_modules=extensions)
