"""Iterator protocol tests for ``FrameIterator`` / ``AudioFrameIterator``.

We verify:
* ``for frame in iterator:`` yields the expected number of frames.
* ``__next__`` raises ``StopIteration`` at exhaustion.
* ``__length_hint__`` decreases as frames are consumed.
* ``reset`` rewinds to the beginning.
* ``seek`` / ``skip`` advance the iterator without going past the end.
"""
from __future__ import annotations

import pytest

from .conftest import requires_wheel


# ---------------------------------------------------------------------------
# FrameIterator
# ---------------------------------------------------------------------------


@requires_wheel
def test_frame_iterator_yields_expected_count():
    """A FrameIterator with frame_count=N yields exactly N frames."""
    import oximedia

    iterator = oximedia.FrameIterator(width=64, height=64, frame_count=5)
    frames = list(iterator)
    assert len(frames) == 5
    # Each yielded item should expose the standard DecodedFrame fields.
    for idx, frame in enumerate(frames):
        assert frame.width == 64
        assert frame.height == 64
        assert frame.index == idx


@requires_wheel
def test_frame_iterator_stop_iteration_at_eof():
    """After all frames are consumed, ``next()`` raises ``StopIteration``."""
    import oximedia

    iterator = oximedia.FrameIterator(width=32, height=32, frame_count=2)
    next(iterator)
    next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


@requires_wheel
def test_frame_iterator_length_hint_decreases():
    """``__length_hint__`` decreases as frames are consumed."""
    import oximedia

    iterator = oximedia.FrameIterator(width=16, height=16, frame_count=4)
    initial_hint = iterator.__length_hint__()
    assert initial_hint == 4
    next(iterator)
    assert iterator.__length_hint__() == 3
    next(iterator)
    assert iterator.__length_hint__() == 2


@requires_wheel
def test_frame_iterator_reset_rewinds():
    """``reset()`` rewinds the iterator to the beginning."""
    import oximedia

    iterator = oximedia.FrameIterator(width=32, height=32, frame_count=3)
    next(iterator)
    next(iterator)
    iterator.reset()
    assert iterator.position == 0
    frames = list(iterator)
    assert len(frames) == 3


@requires_wheel
def test_frame_iterator_skip_advances_position():
    """``skip(n)`` advances the position by ``n`` (clamped to total)."""
    import oximedia

    iterator = oximedia.FrameIterator(width=32, height=32, frame_count=10)
    iterator.skip(3)
    assert iterator.position == 3
    iterator.skip(100)  # over-skip clamps to the end
    assert iterator.position == 10
    assert iterator.is_exhausted is True


@requires_wheel
def test_frame_iterator_seek_clamps_to_end():
    """``seek(idx)`` clamps the position to ``[0, frame_count]``."""
    import oximedia

    iterator = oximedia.FrameIterator(width=32, height=32, frame_count=5)
    iterator.seek(99)
    assert iterator.position == 5
    iterator.seek(0)
    assert iterator.position == 0


@requires_wheel
def test_frame_iterator_keyframe_pattern():
    """Keyframes appear at the configured interval (every N-th frame)."""
    import oximedia

    iterator = oximedia.FrameIterator(
        width=32,
        height=32,
        frame_count=8,
        fps=30.0,
        keyframe_interval=4,
    )
    frames = list(iterator)
    # Frames 0 and 4 should be keyframes; the rest should not be.
    assert frames[0].is_keyframe is True
    assert frames[4].is_keyframe is True
    assert frames[1].is_keyframe is False
    assert frames[5].is_keyframe is False


@requires_wheel
def test_frame_iterator_returns_self_from_iter():
    """``__iter__`` returns the iterator itself."""
    import oximedia

    iterator = oximedia.FrameIterator(width=32, height=32, frame_count=2)
    assert iter(iterator) is iterator


@requires_wheel
def test_frame_iterator_total_frames_attribute():
    """``total_frames`` getter reflects the constructor argument."""
    import oximedia

    iterator = oximedia.FrameIterator(width=32, height=32, frame_count=7)
    assert iterator.total_frames == 7
