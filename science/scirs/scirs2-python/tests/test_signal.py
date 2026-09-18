"""Tests for scirs2 signal processing module."""

import pytest
import numpy as np
import scirs2


class TestConvolution:
    """Test convolution functions."""

    def test_convolve_basic(self):
        """Test basic convolution."""
        a = np.array([1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 0.5])

        result = scirs2.convolve_py(a, v, mode="full")
        # Full convolution length = len(a) + len(v) - 1 = 5
        assert len(result) == 5

    def test_convolve_same_mode(self):
        """Test convolution with same mode."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.array([1.0, 0.0, -1.0])

        result = scirs2.convolve_py(a, v, mode="same")
        assert len(result) == 5

    def test_convolve_valid_mode(self):
        """Test convolution with valid mode."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.array([1.0, 0.0, -1.0])

        result = scirs2.convolve_py(a, v, mode="valid")
        # Valid length = max(len(a), len(v)) - min(len(a), len(v)) + 1 = 3
        assert len(result) == 3

    def test_convolve_identity(self):
        """Test convolution with delta function."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        delta = np.array([1.0])  # Identity for convolution

        result = scirs2.convolve_py(a, delta, mode="full")
        np.testing.assert_allclose(result, a, rtol=1e-10)


class TestCorrelation:
    """Test correlation functions."""

    def test_correlate_basic(self):
        """Test basic cross-correlation."""
        a = np.array([1.0, 2.0, 3.0])
        v = np.array([0.0, 1.0, 0.5])

        result = scirs2.correlate_py(a, v, mode="full")
        assert len(result) == 5

    def test_correlate_same_mode(self):
        """Test correlation with same mode."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.array([1.0, 2.0, 1.0])

        result = scirs2.correlate_py(a, v, mode="same")
        assert len(result) == 5

    def test_autocorrelation(self):
        """Test autocorrelation."""
        a = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = scirs2.correlate_py(a, a, mode="full")

        # Autocorrelation peak at center
        center = len(result) // 2
        assert result[center] >= max(result[0], result[-1])


class TestHilbert:
    """Test Hilbert transform."""

    def test_hilbert_basic(self):
        """Test basic Hilbert transform."""
        # Sine wave
        t = np.linspace(0, 2*np.pi, 100)
        x = np.sin(t)

        result = scirs2.hilbert_py(x)

        assert "real" in result
        assert "imag" in result
        assert len(result["real"]) == len(x)
        assert len(result["imag"]) == len(x)

    def test_hilbert_analytic_signal(self):
        """Test that real part matches input."""
        x = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
        result = scirs2.hilbert_py(x)

        # Real part should match input
        np.testing.assert_allclose(result["real"], x, rtol=1e-6)


class TestWindowFunctions:
    """Test window functions."""

    def test_hann_window(self):
        """Test Hann window."""
        n = 64
        window = scirs2.hann_py(n)

        assert len(window) == n
        # Endpoints should be near 0
        assert window[0] < 0.01
        assert window[-1] < 0.01
        # Middle should be near 1
        assert abs(window[n//2] - 1.0) < 0.01

    def test_hamming_window(self):
        """Test Hamming window."""
        n = 64
        window = scirs2.hamming_py(n)

        assert len(window) == n
        # Endpoints should be around 0.08
        assert window[0] > 0.05 and window[0] < 0.1
        # Middle should be near 1
        assert window[n//2] > 0.95

    def test_blackman_window(self):
        """Test Blackman window."""
        n = 64
        window = scirs2.blackman_py(n)

        assert len(window) == n
        # Endpoints should be near 0
        assert window[0] < 0.01
        # Middle should be near 1
        assert window[n//2] > 0.9

    def test_bartlett_window(self):
        """Test Bartlett (triangular) window."""
        n = 65
        window = scirs2.bartlett_py(n)

        assert len(window) == n
        # Endpoints should be near 0
        assert window[0] < 0.01
        assert window[-1] < 0.01
        # Middle should be exactly 1
        assert abs(window[n//2] - 1.0) < 0.01

    def test_kaiser_window(self):
        """Test Kaiser window."""
        n = 64

        # Beta = 0 gives rectangular window
        window_rect = scirs2.kaiser_py(n, 0.0)
        assert len(window_rect) == n
        assert all(w > 0.99 for w in window_rect)

        # Higher beta gives more tapering
        window_taper = scirs2.kaiser_py(n, 14.0)
        assert window_taper[0] < window_rect[0]
        assert window_taper[n//2] > 0.9


class TestPeakFinding:
    """Test peak finding functions."""

    def test_find_peaks_basic(self):
        """Test basic peak finding."""
        x = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
        peaks = scirs2.find_peaks_py(x)

        # Should find peaks at indices 1, 3, 5
        expected = [1, 3, 5]
        np.testing.assert_array_equal(peaks, expected)

    def test_find_peaks_with_height(self):
        """Test peak finding with height threshold."""
        x = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
        peaks = scirs2.find_peaks_py(x, height=1.5)

        # Only peaks at 3 and 5 are above 1.5
        assert 1 not in peaks
        assert 3 in peaks
        assert 5 in peaks

    def test_find_peaks_with_distance(self):
        """Test peak finding with minimum distance."""
        x = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 1.0, 0.0])
        peaks = scirs2.find_peaks_py(x, distance=3)

        # Peaks should be at least 3 apart
        for i in range(len(peaks) - 1):
            assert peaks[i+1] - peaks[i] >= 3

    def test_find_peaks_sine(self):
        """Test peak finding on sine wave."""
        t = np.linspace(0, 4*np.pi, 100)
        x = np.sin(t)

        peaks = scirs2.find_peaks_py(x)

        # Should find 2 peaks (at pi/2 and 5*pi/2)
        assert len(peaks) == 2

    def test_find_peaks_no_peaks(self):
        """Test when there are no peaks."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Monotonic
        peaks = scirs2.find_peaks_py(x)

        assert len(peaks) == 0


class TestEdgeCases:
    """Test edge cases."""

    def test_convolve_single_element(self):
        """Test convolution with single element."""
        a = np.array([1.0, 2.0, 3.0])
        v = np.array([2.0])

        result = scirs2.convolve_py(a, v)
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0], rtol=1e-10)

    def test_window_small(self):
        """Test window with small size."""
        window = scirs2.hann_py(3)
        assert len(window) == 3

    def test_find_peaks_short_array(self):
        """Test peak finding on short array."""
        x = np.array([1.0, 2.0])  # Too short for peaks
        peaks = scirs2.find_peaks_py(x)

        assert len(peaks) == 0


class TestFilterDesign:
    """Test filter design functions."""

    def test_butter_lowpass(self):
        """Test Butterworth lowpass filter design."""
        result = scirs2.butter_py(4, 0.3, "lowpass")

        assert "b" in result
        assert "a" in result
        # 4th order filter should have 5 coefficients
        assert len(result["b"]) == 5
        assert len(result["a"]) == 5
        # First denominator coefficient should be 1 (normalized)
        assert abs(result["a"][0] - 1.0) < 0.1

    def test_butter_highpass(self):
        """Test Butterworth highpass filter design."""
        result = scirs2.butter_py(3, 0.5, "highpass")

        assert "b" in result
        assert "a" in result
        # 3rd order filter should have 4 coefficients
        assert len(result["b"]) == 4
        assert len(result["a"]) == 4

    def test_butter_different_orders(self):
        """Test Butterworth filter with different orders."""
        for order in [1, 2, 4, 6]:
            result = scirs2.butter_py(order, 0.3, "lowpass")
            assert len(result["b"]) == order + 1
            assert len(result["a"]) == order + 1

    def test_butter_different_cutoffs(self):
        """Test Butterworth filter with different cutoff frequencies."""
        for cutoff in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = scirs2.butter_py(2, cutoff, "lowpass")
            assert len(result["b"]) == 3
            assert len(result["a"]) == 3

    def test_cheby1_lowpass(self):
        """Test Chebyshev Type I lowpass filter design."""
        result = scirs2.cheby1_py(4, 0.5, 0.3, "lowpass")

        assert "b" in result
        assert "a" in result
        assert len(result["b"]) == 5
        assert len(result["a"]) == 5

    def test_cheby1_highpass(self):
        """Test Chebyshev Type I highpass filter design."""
        result = scirs2.cheby1_py(3, 1.0, 0.4, "highpass")

        assert "b" in result
        assert "a" in result
        assert len(result["b"]) == 4
        assert len(result["a"]) == 4

    def test_cheby1_different_ripple(self):
        """Test Chebyshev filter with different ripple values."""
        for ripple in [0.1, 0.5, 1.0, 3.0]:
            result = scirs2.cheby1_py(2, ripple, 0.3, "lowpass")
            assert len(result["b"]) == 3
            assert len(result["a"]) == 3

    def test_filter_coefficients_finite(self):
        """Test that filter coefficients are all finite."""
        result = scirs2.butter_py(4, 0.3, "lowpass")

        assert all(np.isfinite(result["b"]))
        assert all(np.isfinite(result["a"]))


class TestFirFilter:
    """Test FIR filter design functions."""

    def test_firwin_lowpass(self):
        """Test FIR lowpass filter design."""
        coeffs = scirs2.firwin_py(65, 0.3, "hamming", True)

        assert len(coeffs) == 65
        # Coefficients should be symmetric for linear phase
        for i in range(32):
            assert abs(coeffs[i] - coeffs[64 - i]) < 1e-10

    def test_firwin_highpass(self):
        """Test FIR highpass filter design."""
        coeffs = scirs2.firwin_py(65, 0.5, "hamming", False)

        assert len(coeffs) == 65
        # All coefficients should be finite
        assert all(np.isfinite(coeffs))

    def test_firwin_different_windows(self):
        """Test FIR filter with different window functions."""
        for window in ["hamming", "hann", "blackman"]:
            coeffs = scirs2.firwin_py(33, 0.25, window, True)
            assert len(coeffs) == 33
            assert all(np.isfinite(coeffs))

    def test_firwin_different_taps(self):
        """Test FIR filter with different number of taps."""
        for numtaps in [17, 33, 65, 129]:
            coeffs = scirs2.firwin_py(numtaps, 0.3, "hamming", True)
            assert len(coeffs) == numtaps

    def test_firwin_unity_dc_gain(self):
        """Test that lowpass FIR filter has unity DC gain."""
        coeffs = scirs2.firwin_py(65, 0.3, "hamming", True)

        # Sum of coefficients should be approximately 1 for lowpass
        dc_gain = sum(coeffs)
        assert abs(dc_gain - 1.0) < 0.01

    def test_firwin_different_cutoffs(self):
        """Test FIR filter with different cutoff frequencies."""
        for cutoff in [0.1, 0.3, 0.5, 0.7]:
            coeffs = scirs2.firwin_py(65, cutoff, "hamming", True)
            assert len(coeffs) == 65
            assert all(np.isfinite(coeffs))
