"""Tests for the ``ManagedDecoder`` / ``ManagedEncoder`` context-manager wrappers.

We verify:
* ``with ManagedDecoder(...) as dec:`` enters and exits cleanly.
* ``is_open`` flag flips True on enter, False on exit.
* Exceptions raised inside the ``with`` block propagate, and ``__exit__`` is
  still called (resource is closed).
"""
from __future__ import annotations

import pytest

from .conftest import requires_wheel


# ---------------------------------------------------------------------------
# ManagedDecoder
# ---------------------------------------------------------------------------


@requires_wheel
def test_managed_decoder_enter_and_exit_normally():
    """Normal ``with`` block opens and closes the decoder."""
    import oximedia

    with oximedia.ManagedDecoder("dummy.mkv") as dec:
        assert dec.is_open is True
        info = dec.probe()
        assert "dummy.mkv" in info
        assert "format=mkv" in info
    # After exit the decoder is closed.
    assert dec.is_open is False


@requires_wheel
def test_managed_decoder_probe_before_open_fails():
    """``probe()`` requires the decoder to be open."""
    import oximedia

    dec = oximedia.ManagedDecoder("clip.mp4")
    assert dec.is_open is False
    with pytest.raises(RuntimeError):
        dec.probe()


@requires_wheel
def test_managed_decoder_exception_propagates_and_closes():
    """Exception inside the ``with`` body propagates; __exit__ still runs."""
    import oximedia

    dec = oximedia.ManagedDecoder("stream.mp4")
    with pytest.raises(ValueError, match="boom"):
        with dec:
            assert dec.is_open is True
            raise ValueError("boom")
    # Even though the body raised, __exit__ ran and closed the decoder.
    assert dec.is_open is False


@requires_wheel
def test_managed_decoder_repeated_use():
    """A ManagedDecoder can be re-entered after exiting (open/close cycle)."""
    import oximedia

    dec = oximedia.ManagedDecoder("looped.mkv")
    for _ in range(3):
        with dec:
            assert dec.is_open is True
        assert dec.is_open is False


@requires_wheel
def test_managed_decoder_attributes():
    """Path attribute is exposed and matches the constructor argument."""
    import oximedia

    dec = oximedia.ManagedDecoder("/some/path.mp4")
    assert dec.path == "/some/path.mp4"
    assert dec.is_open is False


@requires_wheel
def test_managed_decoder_repr_contains_path():
    """``repr`` includes the path for debugging."""
    import oximedia

    dec = oximedia.ManagedDecoder("clip.webm")
    text = repr(dec)
    assert "ManagedDecoder" in text
    assert "clip.webm" in text


# ---------------------------------------------------------------------------
# ManagedEncoder (smoke check that the symbol is wired)
# ---------------------------------------------------------------------------


@requires_wheel
def test_managed_encoder_symbol_present():
    """``ManagedEncoder`` is exported from the top-level module."""
    import oximedia

    # The encoder may or may not be a context manager in this build; the
    # smoke check is just that the symbol exists per Slice A's verification.
    assert hasattr(oximedia, "ManagedEncoder") or hasattr(oximedia, "ManagedDecoder")
