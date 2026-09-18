"""
SciPy Comparison Tests for Signal Processing Module

Compares scirs2 signal processing functions against SciPy.signal
to ensure numerical correctness and API compatibility.
"""

import numpy as np
import pytest
import scipy.signal
import scipy.signal.windows
import scirs2


class TestWindows:
    """Test window functions"""

    def test_hamming_window(self):
        """Hamming window should match NumPy"""
        for n in [10, 50, 100]:
            # NumPy Hamming window
            numpy_window = np.hamming(n)

            # SciRS2 Hamming window
            scirs2_window = scirs2.hamming_py(n)

            assert np.allclose(numpy_window, scirs2_window, rtol=1e-12)

    def test_hann_window(self):
        """Hann window should match NumPy"""
        for n in [10, 50, 100]:
            # NumPy Hann window
            numpy_window = np.hanning(n)

            # SciRS2 Hann window
            scirs2_window = scirs2.hann_py(n)

            assert np.allclose(numpy_window, scirs2_window, rtol=1e-12)

    def test_blackman_window(self):
        """Blackman window should match NumPy"""
        for n in [10, 50, 100]:
            # NumPy Blackman window
            numpy_window = np.blackman(n)

            # SciRS2 Blackman window
            scirs2_window = scirs2.blackman_py(n)

            assert np.allclose(numpy_window, scirs2_window, rtol=1e-12)

    def test_kaiser_window(self):
        """Kaiser window should match NumPy"""
        for n in [10, 50]:
            beta = 5.0
            # NumPy Kaiser window
            numpy_window = np.kaiser(n, beta)

            # SciRS2 Kaiser window
            scirs2_window = scirs2.kaiser_py(n, beta)

            assert np.allclose(numpy_window, scirs2_window, rtol=1e-12)


class TestConvolution:
    """Test convolution operations"""

    def test_convolve_basic(self):
        """Convolution should match SciPy/NumPy"""
        np.random.seed(42)

        signal = np.ascontiguousarray(np.random.randn(50))
        kernel = np.ascontiguousarray(np.array([0.25, 0.5, 0.25]))  # Simple smoothing kernel

        # NumPy convolution
        numpy_result = np.convolve(signal, kernel, mode='same')

        # SciRS2 convolution
        scirs2_result = scirs2.convolve_py(signal, kernel, mode='same')

        assert np.allclose(numpy_result, scirs2_result, rtol=1e-10)

    def test_convolve_same_mode_only(self):
        """Convolution with same mode"""
        np.random.seed(42)

        signal = np.ascontiguousarray(np.random.randn(20))
        kernel = np.ascontiguousarray(np.array([1.0, 2.0, 1.0]))

        # Only test 'same' mode
        numpy_result = np.convolve(signal, kernel, mode='same')
        scirs2_result = scirs2.convolve_py(signal, kernel, mode='same')

        assert np.allclose(numpy_result, scirs2_result, rtol=1e-10)


class TestCorrelation:
    """Test correlation operations"""

    def test_correlate_basic(self):
        """Correlation should match NumPy"""
        np.random.seed(42)

        a = np.ascontiguousarray(np.random.randn(50))
        v = np.ascontiguousarray(np.random.randn(10))

        # NumPy correlation
        numpy_result = np.correlate(a, v, mode='same')

        # SciRS2 correlation
        scirs2_result = scirs2.correlate_py(a, v, mode='same')

        assert np.allclose(numpy_result, scirs2_result, rtol=1e-10)


class TestHilbertTransform:
    """Test Hilbert transform"""

    def test_hilbert_computable(self):
        """Hilbert transform should be computable"""
        np.random.seed(42)

        # Create a signal
        t = np.ascontiguousarray(np.linspace(0, 1, 100))
        signal = np.ascontiguousarray(np.sin(2 * np.pi * 5 * t))

        # SciRS2 Hilbert transform (returns dict with 'real' and 'imag')
        result = scirs2.hilbert_py(signal)

        # Should return dict with real and imaginary parts
        assert isinstance(result, dict)
        assert 'real' in result
        assert 'imag' in result

        # Lengths should match input
        assert len(result['real']) == len(signal)
        assert len(result['imag']) == len(signal)

        # All values should be finite
        assert np.all(np.isfinite(result['real']))
        assert np.all(np.isfinite(result['imag']))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
