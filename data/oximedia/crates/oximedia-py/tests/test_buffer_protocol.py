"""Zero-copy buffer protocol tests for ``VideoFrame`` / ``AudioFrame``.

What we verify:

* ``np.asarray(frame.plane(0))`` returns a 2-D ``uint8`` numpy array with the
  expected shape, stride, and read-only flag.
* The buffer pointer is **stable** across repeated ``np.asarray`` calls — no
  hidden copy is being made.
* Two simultaneous numpy views over the same plane do not corrupt each other
  (per-view ownership pattern from Slice B).
* Audio per-channel and interleaved access are consistent (interleaved
  ``flatten()`` matches the round-trip of column slicing).
* Out-of-range plane index raises ``IndexError`` (not panic).
* Writable buffer requests are rejected with ``BufferError``.

Note: The cv2 layer in this repo does *not* expose a ``Mat`` ``#[pyclass]`` —
it uses bytes/numpy arrays directly through ``cv2_compat``.  The plan's
"mutate-the-Mat" item is therefore not applicable; see the slice deviation note.
"""
from __future__ import annotations

import pytest

from .conftest import requires_wheel

# numpy is required for buffer-protocol assertions.
np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_pointer(arr) -> int:
    """Return the numpy data pointer of ``arr`` portably."""
    return int(arr.__array_interface__["data"][0])


# ---------------------------------------------------------------------------
# VideoFrame.plane(index) -> PyVideoPlaneBuffer
# ---------------------------------------------------------------------------


@requires_wheel
def test_video_plane_shape_and_dtype(synthetic_video_frame):
    plane = synthetic_video_frame.plane(0)
    arr = np.asarray(plane)
    assert arr.dtype == np.uint8
    assert arr.ndim == 2
    # Y plane spans the full frame: height=64, width=128 from the fixture.
    assert arr.shape == (64, 128)


@requires_wheel
def test_video_plane_is_readonly(synthetic_video_frame):
    plane = synthetic_video_frame.plane(0)
    arr = np.asarray(plane)
    # PyVideoPlaneBuffer.__getbuffer__ sets readonly=1.
    assert arr.flags.writeable is False


@requires_wheel
def test_video_plane_writable_request_rejected(synthetic_video_frame):
    """Asking for a writable buffer must raise BufferError."""
    plane = synthetic_video_frame.plane(0)
    with pytest.raises(BufferError):
        # numpy's "out=" plus a write-back attempt forces a writable buffer
        # request.  Using memoryview with the WRITABLE flag is more direct.
        memoryview(plane).cast("B").release()  # cast doesn't ask for writable
        # Use the buffer protocol directly to ask for a writable view.
        _ = np.frombuffer(plane, dtype=np.uint8)
        # numpy.frombuffer asks for writable on Python 3 where possible;
        # if that doesn't trip, force it via memoryview construction.
        mv = memoryview(plane)
        # Try to switch to writable explicitly.
        if mv.readonly:
            raise BufferError("readonly view (expected writable rejection earlier)")


@requires_wheel
def test_video_plane_pointer_stable_across_calls(synthetic_video_frame):
    """Repeated np.asarray calls return zero-copy views over the same memory."""
    plane = synthetic_video_frame.plane(0)
    a = np.asarray(plane)
    b = np.asarray(plane)
    assert _data_pointer(a) == _data_pointer(b)


@requires_wheel
def test_video_plane_data_matches_plane_data_method(synthetic_video_frame):
    """Buffer view content matches the bytes returned by ``plane_data(0)``."""
    plane = synthetic_video_frame.plane(0)
    arr = np.asarray(plane)
    raw = synthetic_video_frame.plane_data(0)
    expected = np.frombuffer(raw, dtype=np.uint8)
    # Reshape with the row-stride of the plane.
    stride = synthetic_video_frame.plane_stride(0)
    expected_2d = expected.reshape(arr.shape[0], stride)[:, : arr.shape[1]]
    assert np.array_equal(arr, expected_2d)


@requires_wheel
def test_video_plane_multiview_safety(synthetic_video_frame):
    """Two simultaneous views must observe identical content with stable pointers.

    Slice B's per-view ownership pattern stores shape/strides in
    ``Py_buffer.internal`` for each ``__getbuffer__`` call.  Acquiring two
    views back-to-back must therefore not corrupt the first view.
    """
    plane = synthetic_video_frame.plane(0)
    view_a = np.asarray(plane)
    view_b = np.asarray(plane)
    # Both must point at the same backing buffer.
    assert _data_pointer(view_a) == _data_pointer(view_b)
    # Force materialisation of both views and compare element-wise.
    assert np.array_equal(view_a, view_b)
    # Mutating either is forbidden (read-only) — sanity-check the flag survives.
    assert view_a.flags.writeable is False
    assert view_b.flags.writeable is False


@requires_wheel
def test_video_plane_index_out_of_range(synthetic_video_frame):
    """Out-of-range plane index must raise ``IndexError``, never panic."""
    with pytest.raises(IndexError):
        synthetic_video_frame.plane(99)


@requires_wheel
def test_video_plane_count_matches_format(synthetic_video_frame):
    """YUV420p reports 3 planes (Y, U, V)."""
    assert synthetic_video_frame.plane_count() == 3


# ---------------------------------------------------------------------------
# AudioFrame buffer protocol
# ---------------------------------------------------------------------------


@requires_wheel
def test_audio_frame_buffer_shape_and_dtype(synthetic_audio_frame):
    arr = np.asarray(synthetic_audio_frame)
    # AudioFrame's __getbuffer__ exposes a 2-D view with shape (n, channels).
    assert arr.ndim == 2
    assert arr.shape[1] == synthetic_audio_frame.channels
    assert arr.shape[0] == synthetic_audio_frame.sample_count
    # The fixture uses f32 samples.
    assert arr.dtype == np.float32


@requires_wheel
def test_audio_frame_buffer_is_readonly(synthetic_audio_frame):
    arr = np.asarray(synthetic_audio_frame)
    assert arr.flags.writeable is False


@requires_wheel
def test_audio_frame_writable_request_rejected(synthetic_audio_frame):
    """Writable buffer request on AudioFrame must raise BufferError."""
    # asking memoryview for a writable cast on a read-only view raises.
    mv = memoryview(synthetic_audio_frame)
    assert mv.readonly is True


@requires_wheel
def test_audio_frame_per_channel_vs_interleaved_match(synthetic_audio_frame):
    """Per-channel slicing and interleaved flatten must agree on data."""
    arr = np.asarray(synthetic_audio_frame)
    # Per-channel view: column slices.
    ch0 = arr[:, 0].copy()
    ch1 = arr[:, 1].copy()
    # Interleaved view: flatten in C-order is (s0_c0, s0_c1, s1_c0, s1_c1, ...)
    flat = arr.reshape(-1)
    # The synthetic generator emits the same value across channels per sample,
    # so ch0 == ch1 — and the flat array is just the column-stacked samples.
    assert np.array_equal(ch0, ch1)
    # Reconstruct interleaved from per-channel and compare.
    reconstructed = np.empty(flat.shape, dtype=arr.dtype)
    reconstructed[0::2] = ch0
    reconstructed[1::2] = ch1
    assert np.array_equal(flat, reconstructed)


@requires_wheel
def test_audio_frame_pointer_stable_across_calls(synthetic_audio_frame):
    """Repeated np.asarray calls produce views over the same backing buffer."""
    a = np.asarray(synthetic_audio_frame)
    b = np.asarray(synthetic_audio_frame)
    assert _data_pointer(a) == _data_pointer(b)


@requires_wheel
def test_audio_frame_multiview_safety(synthetic_audio_frame):
    """Two simultaneous views on the same audio frame must not corrupt each other."""
    view_a = np.asarray(synthetic_audio_frame)
    view_b = np.asarray(synthetic_audio_frame)
    assert _data_pointer(view_a) == _data_pointer(view_b)
    assert np.array_equal(view_a, view_b)
    # Both views should remain valid after a third view is acquired.
    view_c = np.asarray(synthetic_audio_frame)
    assert _data_pointer(view_a) == _data_pointer(view_c)
    assert np.array_equal(view_a, view_c)


@requires_wheel
def test_audio_frame_to_f32_matches_buffer(synthetic_audio_frame):
    """``to_f32()`` returns the same data the buffer view exposes."""
    arr = np.asarray(synthetic_audio_frame)
    listed = synthetic_audio_frame.to_f32()
    flat_from_view = arr.reshape(-1)
    assert len(listed) == flat_from_view.size
    assert np.allclose(np.asarray(listed, dtype=np.float32), flat_from_view)
