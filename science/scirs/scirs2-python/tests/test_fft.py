"""Tests for scirs2 FFT module."""

import numpy as np
import pytest
import scirs2


class TestFFT:
    """Test FFT functions."""

    def test_fft_basic(self):
        """Test basic FFT."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = scirs2.fft_py(data)

        assert "real" in result
        assert "imag" in result
        assert len(result["real"]) == 4
        assert len(result["imag"]) == 4

        # DC component should be sum
        assert abs(result["real"][0] - 10.0) < 1e-10
        assert abs(result["imag"][0]) < 1e-10

    def test_fft_ifft_roundtrip(self):
        """Test FFT followed by IFFT recovers original."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        fft_result = scirs2.fft_py(data)

        # IFFT
        ifft_result = scirs2.ifft_py(
            np.array(fft_result["real"]),
            np.array(fft_result["imag"])
        )

        # Real parts should match original
        reconstructed = np.array(ifft_result["real"])
        assert np.allclose(data, reconstructed, atol=1e-10)

    def test_rfft(self):
        """Test real FFT."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = scirs2.rfft_py(data)

        # RFFT returns n/2 + 1 components
        assert len(result["real"]) == 3

    def test_irfft(self):
        """Test inverse real FFT."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        rfft_result = scirs2.rfft_py(data)

        # IRFFT
        reconstructed = scirs2.irfft_py(
            np.array(rfft_result["real"]),
            np.array(rfft_result["imag"]),
            len(data)
        )

        assert np.allclose(data, reconstructed, atol=1e-10)


class TestDCT:
    """Test Discrete Cosine Transform functions."""

    def test_dct_type2(self):
        """Test DCT-II (most common type)."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = scirs2.dct_py(data, 2)
        assert len(result) == 4

    def test_dct_idct_roundtrip(self):
        """Test DCT followed by IDCT recovers original."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        dct_result = scirs2.dct_py(data, 2)
        idct_result = scirs2.idct_py(np.array(dct_result), 2)

        assert np.allclose(data, idct_result, atol=1e-10)

    def test_dct_types(self):
        """Test different DCT types."""
        data = np.array([1.0, 2.0, 3.0, 4.0])

        for dct_type in [1, 2, 3, 4]:
            result = scirs2.dct_py(data, dct_type)
            assert len(result) == len(data)


class TestHelperFunctions:
    """Test FFT helper functions."""

    def test_fftfreq(self):
        """Test FFT frequency generation."""
        freqs = scirs2.fftfreq_py(8, 1.0)
        assert len(freqs) == 8

        # First half should be non-negative
        assert freqs[0] == 0
        # Second half should be negative (aliased frequencies)
        assert freqs[5] < 0

    def test_rfftfreq(self):
        """Test real FFT frequency generation."""
        freqs = scirs2.rfftfreq_py(8, 1.0)
        # RFFT frequencies are n/2 + 1
        assert len(freqs) == 5
        # All should be non-negative
        assert all(f >= 0 for f in freqs)

    def test_fftshift(self):
        """Test FFT shift."""
        data = np.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
        shifted = scirs2.fftshift_py(data)

        # After shift, negative frequencies should be first
        assert shifted[0] < 0

    def test_ifftshift(self):
        """Test inverse FFT shift."""
        data = np.array([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
        shifted = scirs2.fftshift_py(data)
        unshifted = scirs2.ifftshift_py(shifted)

        assert np.allclose(data, unshifted, atol=1e-10)

    def test_next_fast_len(self):
        """Test next fast FFT length."""
        # Powers of 2 are fast
        assert scirs2.next_fast_len_py(100, False) >= 100
        assert scirs2.next_fast_len_py(128, False) == 128

        # Result should be efficient for FFT
        fast_len = scirs2.next_fast_len_py(100, False)
        assert fast_len >= 100


class TestFFTProperties:
    """Test mathematical properties of FFT."""

    def test_parseval_theorem(self):
        """Test Parseval's theorem: energy conservation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = scirs2.fft_py(data)

        # Time domain energy
        time_energy = np.sum(data ** 2)

        # Frequency domain energy
        freq_real = np.array(result["real"])
        freq_imag = np.array(result["imag"])
        freq_energy = np.sum(freq_real ** 2 + freq_imag ** 2) / len(data)

        assert abs(time_energy - freq_energy) < 1e-10

    def test_symmetry_real_input(self):
        """Test conjugate symmetry for real input."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = scirs2.fft_py(data)

        # For real input, X[k] = conj(X[N-k])
        # Check X[1] = conj(X[3])
        assert abs(result["real"][1] - result["real"][3]) < 1e-10
        assert abs(result["imag"][1] + result["imag"][3]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
