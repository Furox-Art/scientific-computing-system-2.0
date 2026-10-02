"""Build script for optional compiled accelerators.

Only the native kernels that are wired into public runtime paths are built:
``cds2._fast_kmeans`` and ``cds2._fast_pagerank``. Keeping unused native
extensions out of wheels reduces maintenance and memory-safety attack surface.
"""

import os
import sys

from setuptools import Extension, setup

# Fail-closed native build: when the compiled accelerators are requested
# (anything but CDS_PURE=1), the C sources must exist. setuptools marks the
# extensions `optional=True` so a missing *compiler* still falls back to
# pure Python — but missing *sources* must never silently produce a
# capability-less wheel.
C_SOURCES = [
    "src/cds2/src/_fast_kmeans.c",
    "src/cds2/src/_fast_pagerank.c",
]

if os.environ.get("CDS_PURE") != "1":
    missing_sources = [s for s in C_SOURCES if not os.path.isfile(s)]
    if missing_sources:
        raise SystemExit(
            "ERROR: missing C accelerator sources: "
            + ", ".join(missing_sources)
            + ". Refusing to build a capability-less wheel; "
            "set CDS_PURE=1 for the pure-Python fallback wheel."
        )

extra_compile_args: list[str] = ["-O3"]
extra_link_args: list[str] = []

if os.environ.get("CDS_NO_OPENMP") != "1":
    if sys.platform == "linux":
        extra_compile_args += ["-fopenmp"]
        extra_link_args += ["-fopenmp"]
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
        ]
    )

setup(ext_modules=extensions)
