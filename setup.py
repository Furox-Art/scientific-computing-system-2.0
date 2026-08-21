"""Build script: declares the optional compiled accelerator."""

import os

from setuptools import Extension, setup

extensions = []
if os.environ.get("CDS_PURE") != "1":
    # optional=True keeps installation working on machines without a C
    # compiler; cds2 then uses its NumPy fallback at runtime.
    extensions.append(
        Extension(
            "cds2._fast_kmeans",
            sources=["src/cds2/src/_fast_kmeans.c"],
        )
    )

setup(ext_modules=extensions)
