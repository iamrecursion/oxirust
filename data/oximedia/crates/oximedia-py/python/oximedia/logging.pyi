"""
Type stubs for ``oximedia.logging`` — Rust-to-Python logging bridge.

Source: ``crates/oximedia-py/src/logging_py.rs``
"""

from __future__ import annotations

class PyOxiMediaLogger:
    """A named Python-side logger routed via :mod:`logging`."""

    name: str

    def __init__(self, name: str) -> None: ...
    def debug(self, message: str) -> None: ...
    def info(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def critical(self, message: str) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

def init(level: str = "info") -> None:
    """Initialise the Rust-side ``tracing`` subscriber.

    Acceptable levels: ``"trace"``, ``"debug"``, ``"info"``, ``"warning"``,
    ``"error"``, ``"critical"``.
    """
    ...

def set_level(level: str) -> None:
    """Change the active log level for the OxiMedia logger."""
    ...

def get_level() -> str:
    """Return the currently active log level string."""
    ...

def log(level: str, message: str, logger_name: str = "oximedia") -> None:
    """Emit a single log record on ``logger_name`` at ``level``."""
    ...

def is_initialized() -> bool:
    """Return ``True`` if :func:`init` has been called."""
    ...
