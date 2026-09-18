"""Parity sanity-checks against the cv2-compat layer.

These are not full OpenCV bit-for-bit tests; they verify that the most-used
helpers behave sensibly: BGR↔RGB swap, imread/imwrite round-trip,
GaussianBlur smooths, and threshold produces 0/255 output.
"""
from __future__ import annotations

import pytest

from .conftest import requires_wheel

np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# cvtColor: BGR ↔ RGB swap
# ---------------------------------------------------------------------------


@requires_wheel
def test_cvt_color_bgr2rgb_swaps_channels():
    """COLOR_BGR2RGB swaps channel order without altering data."""
    import oximedia
    cv2 = oximedia.cv2

    h, w = 8, 8
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # B
    bgr[..., 1] = 20  # G
    bgr[..., 2] = 30  # R

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_arr = np.asarray(rgb).reshape(h, w, 3)
    # After swap, channel 0 should be the original R (=30), channel 2 the original B (=10).
    assert np.all(rgb_arr[..., 0] == 30)
    assert np.all(rgb_arr[..., 1] == 20)
    assert np.all(rgb_arr[..., 2] == 10)


@requires_wheel
def test_cvt_color_bgr2gray_produces_single_channel():
    """COLOR_BGR2GRAY collapses 3 channels into 1 via BT.601 weights."""
    import oximedia
    cv2 = oximedia.cv2

    h, w = 16, 16
    bgr = np.full((h, w, 3), 200, dtype=np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_arr = np.asarray(gray).reshape(h, w)
    # All pixels are the same colour, so all gray values must be equal.
    assert gray_arr.dtype == np.uint8
    assert np.all(gray_arr == gray_arr[0, 0])
    # BT.601 weights on (B=200,G=200,R=200) → 200 (within rounding).
    assert abs(int(gray_arr[0, 0]) - 200) <= 1


# ---------------------------------------------------------------------------
# imread / imwrite round-trip
# ---------------------------------------------------------------------------


@requires_wheel
def test_imread_imwrite_roundtrip(tmp_path):
    """Writing an image and reading it back yields the same pixel data."""
    import oximedia
    cv2 = oximedia.cv2

    h, w = 32, 32
    rng = np.random.default_rng(seed=0xCAFEBEEF)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    out_path = tmp_path / "roundtrip.png"
    ok = cv2.imwrite(str(out_path), img)
    assert ok is True
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    loaded = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    loaded_arr = np.asarray(loaded).reshape(h, w, 3)
    assert loaded_arr.shape == img.shape
    assert loaded_arr.dtype == np.uint8
    # PNG is lossless → exact match expected.
    assert np.array_equal(loaded_arr, img)


@requires_wheel
def test_imread_grayscale_returns_single_channel(tmp_path):
    """imread with IMREAD_GRAYSCALE returns a 2-D array (single channel)."""
    import oximedia
    cv2 = oximedia.cv2

    h, w = 16, 16
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    out_path = tmp_path / "gray.png"
    cv2.imwrite(str(out_path), img)

    loaded = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
    loaded_arr = np.asarray(loaded)
    # Could be 2-D (h, w) or 3-D (h, w, 1) depending on how the layer reshapes.
    assert loaded_arr.size == h * w


# ---------------------------------------------------------------------------
# GaussianBlur: produces a smoother image
# ---------------------------------------------------------------------------


@requires_wheel
def test_gaussian_blur_reduces_high_frequency_variance():
    """A blurred image has lower local variance than the original."""
    import oximedia
    cv2 = oximedia.cv2

    h, w = 64, 64
    rng = np.random.default_rng(seed=0xDEADBEEF)
    noise = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(noise, (5, 5), 1.4)
    blurred_arr = np.asarray(blurred).reshape(h, w, 3)
    # Compare per-pixel variance of the difference against neighbours; the
    # simplest robust signal is: total variance must drop after blurring.
    assert blurred_arr.var() < noise.var()


@requires_wheel
def test_gaussian_blur_preserves_dimensions():
    """GaussianBlur output shape matches input shape."""
    import oximedia
    cv2 = oximedia.cv2

    h, w = 32, 48
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    blurred = cv2.GaussianBlur(img, (3, 3), 1.0)
    blurred_arr = np.asarray(blurred).reshape(h, w, 3)
    assert blurred_arr.shape == img.shape


@requires_wheel
def test_gaussian_blur_rejects_even_kernel():
    """Even-sized kernels are rejected with ValueError."""
    import oximedia
    cv2 = oximedia.cv2

    img = np.full((16, 16, 3), 100, dtype=np.uint8)
    with pytest.raises(ValueError):
        cv2.GaussianBlur(img, (4, 4), 1.0)


# ---------------------------------------------------------------------------
# threshold: produces 0/255 binary output
# ---------------------------------------------------------------------------


@requires_wheel
def test_threshold_binary_produces_zero_or_max():
    """THRESH_BINARY yields only 0 or maxval pixel values."""
    import oximedia
    cv2 = oximedia.cv2

    h, w = 32, 32
    img = np.tile(np.arange(256, dtype=np.uint8), (h * w + 255) // 256)[: h * w]
    img = img.reshape(h, w)
    # cv2.threshold returns (retval, dst).
    retval, dst = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    dst_arr = np.asarray(dst).reshape(h, w)
    assert dst_arr.dtype == np.uint8
    unique_vals = set(np.unique(dst_arr).tolist())
    assert unique_vals.issubset({0, 255})


@requires_wheel
def test_threshold_returns_retval_and_dst():
    """threshold returns a (retval, dst) tuple."""
    import oximedia
    cv2 = oximedia.cv2

    h, w = 16, 16
    img = np.full((h, w), 100, dtype=np.uint8)
    result = cv2.threshold(img, 50, 200, cv2.THRESH_BINARY)
    assert isinstance(result, tuple)
    assert len(result) == 2
    retval, _dst = result
    assert isinstance(retval, float)
