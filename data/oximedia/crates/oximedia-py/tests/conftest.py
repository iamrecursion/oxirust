"""Shared pytest fixtures and helpers for oximedia-py tests.

The tests run against a maturin-built wheel installed via ``maturin develop``.
A pure-Python namespace package shadow exists at ``./oximedia/`` in the workspace
so a bare ``import oximedia`` always succeeds — that means
``pytest.importorskip("oximedia")`` is *not* sufficient to gate on a real wheel.

Use the ``requires_wheel`` marker (or call ``wheel_built()`` directly) to detect
whether the compiled extension is loaded.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Wheel-availability detection
# ---------------------------------------------------------------------------


def wheel_built() -> bool:
    """Return ``True`` only when the compiled ``oximedia`` extension is loaded.

    The repository ships a pure-Python namespace package skeleton; we therefore
    test for a hallmark PyO3-exported class (``VideoFrame``) rather than the
    bare module name.
    """
    try:
        import oximedia  # noqa: F401
    except ImportError:
        return False
    return hasattr(oximedia, "VideoFrame")


requires_wheel = pytest.mark.skipif(
    not wheel_built(),
    reason="compiled oximedia extension not built; run `maturin develop --release`",
)


# ---------------------------------------------------------------------------
# Synthetic video / audio fixtures (in-memory)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_video_frame():
    """Return a deterministic checkerboard YUV420p frame with non-empty planes.

    ``oximedia.test.synthetic_video_frame`` returns a frame *without* allocated
    planes (its inner Rust ``VideoFrame::new`` lazily skips ``allocate``), so
    we use ``checkerboard_frame`` instead — it explicitly fills the Y plane and
    sets neutral chroma.
    """
    if not wheel_built():
        pytest.skip("compiled oximedia extension not built")
    import oximedia
    return oximedia.test.checkerboard_frame(width=128, height=64, square_size=16, pts=0)


@pytest.fixture
def synthetic_audio_frame():
    """Return a 20 ms 48 kHz stereo sine-tone audio frame (f32 samples)."""
    if not wheel_built():
        pytest.skip("compiled oximedia extension not built")
    import oximedia
    return oximedia.test.synthetic_audio_frame(
        sample_rate=48000,
        channels=2,
        duration_ms=20.0,
        frequency_hz=440.0,
        pts=0,
    )


@pytest.fixture
def sample_video_path(tmp_path):
    """Best-effort path to a tiny synthetic video file.

    The current ``oximedia.test`` module emits *in-memory* frames only — there
    is no helper that muxes them through a container.  Tests that need a real
    file should use the in-memory ``synthetic_video_frame`` fixture or the
    ``FrameIterator`` synthetic source instead.
    """
    if not wheel_built():
        pytest.skip("compiled oximedia extension not built")
    # Reserve a path the test can fall back to if it manages to write its own
    # mux output; we don't generate the file here.
    return tmp_path / "sample.mkv"


@pytest.fixture
def sample_audio_path(tmp_path):
    """Reserve a path for a synthetic audio file.

    Same caveat as ``sample_video_path``: the ``oximedia.test`` submodule only
    emits in-memory frames, so most tests should use ``synthetic_audio_frame``
    rather than this fixture.
    """
    if not wheel_built():
        pytest.skip("compiled oximedia extension not built")
    return tmp_path / "sample.wav"


# ---------------------------------------------------------------------------
# Workspace-pathing helpers (used by stub-coverage tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def oximedia_py_root() -> pathlib.Path:
    """Absolute path to the ``crates/oximedia-py`` directory."""
    return pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def python_stubs_dir(oximedia_py_root) -> pathlib.Path:
    """Absolute path to the ``python/oximedia`` stub-shipping directory."""
    return oximedia_py_root / "python" / "oximedia"
