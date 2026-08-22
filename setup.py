"""Build script: declares the optional compiled accelerators."""

import os
import sys

from setuptools import Extension, setup

# OpenMP is enabled only where the toolchain handles our loop shapes:
# GCC on Linux (the industrial server target). MSVC's legacy OpenMP
# rejects Py_ssize_t/int-mixed loops (C3015) and Apple clang needs a
# separate libomp, so Windows and macOS build the serial kernel.
extra_compile_args: list[str] = []
extra_link_args: list[str] = []
if os.environ.get("CDS_NO_OPENMP") != "1" and sys.platform == "linux":
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]

extensions = []
if os.environ.get("CDS_PURE") != "1":
    # optional=True keeps installation working on machines without a C
    # compiler; cds2 then uses its NumPy fallback at runtime.
    extensions.append(
        Extension(
            "cds2._fast_kmeans",
            sources=["src/cds2/src/_fast_kmeans.c"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )
    extensions.append(
        Extension(
            "cds2._fast_pagerank",
            sources=["src/cds2/src/_fast_pagerank.c"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )

setup(ext_modules=extensions)
