"""Basic import smoke tests — verify the compiled extension loads.

These tests are intentionally minimal: they only check that the module and its
documented submodules can be imported and that a couple of well-known
constants are present.  Heavier API surface is covered by the per-feature
test files alongside this one.
"""
from __future__ import annotations

from .conftest import requires_wheel


@requires_wheel
def test_import_oximedia():
    """The top-level ``oximedia`` package imports cleanly with PyO3 classes."""
    import oximedia

    # The compiled extension exposes a large API surface.  Picking a few
    # hallmark types keeps the assertion robust against churn elsewhere.
    assert hasattr(oximedia, "VideoFrame")
    assert hasattr(oximedia, "AudioFrame")
    assert hasattr(oximedia, "PixelFormat")


@requires_wheel
def test_import_cv2_submodule():
    """The cv2 submodule imports cleanly and exposes basic constants."""
    from oximedia import cv2

    assert hasattr(cv2, "IMREAD_COLOR")
    assert cv2.IMREAD_COLOR == 1
    assert hasattr(cv2, "IMREAD_GRAYSCALE")
    assert cv2.IMREAD_GRAYSCALE == 0
    # Color-conversion code constants
    assert hasattr(cv2, "COLOR_BGR2RGB")
    # Threshold types
    assert hasattr(cv2, "THRESH_BINARY")
    assert cv2.THRESH_BINARY == 0


@requires_wheel
def test_import_io_submodule():
    """The io submodule imports cleanly and is non-empty."""
    from oximedia import io

    # Must expose at least one symbol; the exact API is exercised in io tests.
    assert len(dir(io)) > 0


@requires_wheel
def test_import_test_submodule():
    """The ``oximedia.test`` synthetic-media submodule imports."""
    from oximedia import test as omtest

    # All advertised generators must be present.
    for name in (
        "synthetic_video_frame",
        "synthetic_audio_frame",
        "generate_video_sequence",
        "generate_audio_sequence",
        "solid_color_frame",
        "checkerboard_frame",
        "silence_frame",
    ):
        assert hasattr(omtest, name), f"oximedia.test missing: {name}"


@requires_wheel
def test_import_logging_submodule():
    """The logging submodule imports cleanly."""
    from oximedia import logging as omlogging  # noqa: F401


@requires_wheel
def test_import_utils_submodule():
    """The utils submodule imports cleanly."""
    from oximedia import utils  # noqa: F401


@requires_wheel
def test_import_benchmark_submodule():
    """The benchmark submodule imports cleanly."""
    from oximedia import benchmark  # noqa: F401
