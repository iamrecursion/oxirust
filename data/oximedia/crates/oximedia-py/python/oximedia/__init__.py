"""``oximedia`` — Python bindings for the OxiMedia Sovereign Media Framework.

This is a maturin "mixed" Python/Rust package: the Rust extension module is
built as ``oximedia.oximedia`` (an in-package shared library) and this
``__init__.py`` re-exports its public surface so that ``import oximedia`` and
``from oximedia import PyTranscoder`` continue to work.

Submodules registered as PyO3 modules (``cv2``, ``io``, ``utils``,
``logging``, ``test``, ``benchmark``) are re-bound here so that, for example,
``oximedia.io.probe(...)`` resolves regardless of how the user imports the
package.

The companion ``__main__.py`` provides ``python -m oximedia <subcommand>``.
"""

from __future__ import annotations

# Re-export everything from the compiled Rust extension.
# The extension module is named ``oximedia`` by ``#[pymodule] fn oximedia``
# in ``crates/oximedia-py/src/lib.rs``; under maturin "mixed" mode it lands
# at ``oximedia/oximedia.<so|pyd>`` and is therefore imported relative to
# this package.
from .oximedia import *  # noqa: F401, F403 — re-export Rust extension surface
from .oximedia import (  # noqa: F401 — explicit submodule rebind for IDEs
    cv2,
    io,
    logging,
    test,
    utils,
    benchmark,
)

# Expose a ``__version__`` attribute (preferring the installed distribution
# metadata, falling back gracefully if the package is being run from a source
# checkout without an installed wheel).
try:  # pragma: no cover - exercised at runtime only
    from importlib.metadata import PackageNotFoundError, version as _md_version

    try:
        __version__ = _md_version("oximedia")
    except PackageNotFoundError:
        __version__ = "0.0.0+unknown"
    del _md_version, PackageNotFoundError
except ImportError:  # pragma: no cover - importlib.metadata is std since 3.10
    __version__ = "0.0.0+unknown"
