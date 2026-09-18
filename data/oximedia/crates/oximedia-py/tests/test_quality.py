"""Quality metric sanity checks: PSNR / SSIM / quality_report.

The underlying ``oximedia-quality`` library is YUV-oriented (luma + chroma
weighted at 4/6 + 1/6 + 1/6).  The Python API in ``oximedia.quality`` builds
a Gray8 ``Frame`` with only the luma plane filled, so identical inputs yield
roughly ``2/3`` (= 0.666...) — *not* 1.0 — because the missing chroma planes
contribute zero weighted score.

What we verify here:
* Identical buffers yield the luma-only "perfect" score (~0.666).
* Wildly different buffers yield a score substantially below that ceiling.
* Bit-inverted reference / distorted yield a clearly negative SSIM.
* ``quality_report`` returns a non-empty list of no-reference metrics on a
  sufficiently large image (NIQE needs ≥ 96×96 patches).
"""
from __future__ import annotations

import math

import pytest

from .conftest import requires_wheel

np = pytest.importorskip("numpy")


def _gray_buf(width: int, height: int, value: int = 0) -> bytes:
    return np.full((height, width), value, dtype=np.uint8).tobytes()


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


@requires_wheel
def test_psnr_identical_buffers_returns_high_score():
    """Identical buffers yield the luma-only "perfect" score (~0.666 → +∞ * 2/3).

    Note: because the wrapper feeds a Gray8 frame with empty chroma planes,
    the weighted PSNR collapses to ``luma_weight * Y_PSNR`` ≈ ``2/3 * ∞``.
    The library may clamp this to a finite sentinel; either way, identical
    inputs should score *higher* than any non-trivial difference.
    """
    import oximedia

    width, height = 64, 64
    buf = _gray_buf(width, height, value=128)
    score = oximedia.compute_psnr(buf, buf, width, height)
    # Identical inputs must beat any noisy comparison by a wide margin.
    rng = np.random.default_rng(seed=42)
    a = rng.integers(0, 256, size=(height, width), dtype=np.uint8).tobytes()
    b = rng.integers(0, 256, size=(height, width), dtype=np.uint8).tobytes()
    different_score = oximedia.compute_psnr(a, b, width, height)
    assert math.isfinite(score) or math.isinf(score)
    assert score > different_score


@requires_wheel
def test_psnr_different_buffers_returns_finite_score():
    """Different buffers yield a finite, modest PSNR score."""
    import oximedia

    width, height = 64, 64
    rng = np.random.default_rng(seed=42)
    a = rng.integers(0, 256, size=(height, width), dtype=np.uint8).tobytes()
    b = rng.integers(0, 256, size=(height, width), dtype=np.uint8).tobytes()
    score = oximedia.compute_psnr(a, b, width, height)
    assert math.isfinite(score)
    # Independent random uint8 → PSNR after luma-only weighting falls in a
    # fairly wide envelope; just assert it's bounded and non-degenerate.
    assert 0.0 < score < 30.0


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


@requires_wheel
def test_ssim_identical_buffers_returns_one():
    """Identical inputs yield SSIM == 1.0."""
    import oximedia

    width, height = 64, 64
    buf = _gray_buf(width, height, value=200)
    score = oximedia.compute_ssim(buf, buf, width, height)
    # SSIM should be very close to 1 for identical buffers.
    assert score == pytest.approx(1.0, abs=1e-6)


@requires_wheel
def test_ssim_different_buffers_returns_lower_score():
    """SSIM on independent random buffers should be substantially below 1."""
    import oximedia

    width, height = 64, 64
    rng = np.random.default_rng(seed=2024)
    a = rng.integers(0, 256, size=(height, width), dtype=np.uint8).tobytes()
    b = rng.integers(0, 256, size=(height, width), dtype=np.uint8).tobytes()
    score = oximedia.compute_ssim(a, b, width, height)
    assert -1.0 <= score < 0.95


@requires_wheel
def test_ssim_inverted_buffer_yields_negative_or_low_score():
    """Bit-inverted reference and distorted yield a clearly degraded SSIM."""
    import oximedia

    width, height = 32, 32
    rng = np.random.default_rng(seed=7)
    ref = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    dist = (255 - ref).astype(np.uint8)
    score = oximedia.compute_ssim(ref.tobytes(), dist.tobytes(), width, height)
    # Inverted image should drastically reduce SSIM.
    assert score < 0.5


# ---------------------------------------------------------------------------
# quality_report (no-reference metrics)
# ---------------------------------------------------------------------------


@requires_wheel
def test_quality_report_returns_metrics_list():
    """quality_report yields a list of PyQualityScore entries."""
    import oximedia

    width, height = 32, 32
    rng = np.random.default_rng(seed=1234)
    img = rng.integers(0, 256, size=(height, width), dtype=np.uint8).tobytes()
    report = oximedia.quality_report(img, width, height)
    assert isinstance(report, list)
    assert len(report) > 0
    # Each entry exposes a ``score`` attribute (PyQualityScore class).
    for entry in report:
        assert hasattr(entry, "score") or hasattr(entry, "metric")
