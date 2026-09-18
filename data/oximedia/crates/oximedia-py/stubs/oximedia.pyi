"""
[Reference-only — kept for backward compatibility]

This file is no longer the canonical type-stub source for the ``oximedia``
Python extension module.  The official, mypy-verified, PEP 561-compliant
stubs live alongside the Python package at::

    crates/oximedia-py/python/oximedia/__init__.pyi
    crates/oximedia-py/python/oximedia/cv2.pyi
    crates/oximedia-py/python/oximedia/io.pyi
    crates/oximedia-py/python/oximedia/utils.pyi
    crates/oximedia-py/python/oximedia/logging.pyi
    crates/oximedia-py/python/oximedia/test.pyi
    crates/oximedia-py/python/oximedia/benchmark.pyi
    crates/oximedia-py/python/oximedia/py.typed

Those files are bundled with the maturin-built wheel, are picked up
automatically by mypy / pyright / pylance, and are kept in sync with the
Rust source in ``crates/oximedia-py/src/``.

This single-file reference originally tracked a small subset of the
public API; please consult the per-submodule files above for the
authoritative type information.
"""

from __future__ import annotations

# Re-export every public name from the canonical package-level stub so that
# any tooling still pointing at this directory continues to resolve names.
# (mypy treats this as ``from oximedia import *`` resolving against the
# ``python/oximedia/__init__.pyi`` package.)
from oximedia import *  # noqa: F401,F403
