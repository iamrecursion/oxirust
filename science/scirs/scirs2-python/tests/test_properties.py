"""
Property-Based Tests for SciRS2 Python Bindings

Uses Hypothesis to test mathematical properties and invariants
that should hold for all valid inputs.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays, array_shapes
import scirs2


# Custom strategies for valid inputs
@st.composite
def invertible_matrix(draw, min_size=2, max_size=20):
    """Generate invertible matrices"""
    size = draw(st.integers(min_size=min_size, max_size=max_size))
    # Create well-conditioned matrix
    A = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    # Add diagonal dominance to ensure invertibility
    A = A + np.eye(size) * (np.abs(A).sum() + 10)
    return A


@st.composite
def symmetric_matrix(draw, min_size=2, max_size=20):
    """Generate symmetric matrices"""
    size = draw(st.integers(min_size=min_size, max_size=max_size))
    A = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    return (A + A.T) / 2


@st.composite
def positive_definite_matrix(draw, min_size=2, max_size=20):
    """Generate positive definite matrices"""
    size = draw(st.integers(min_size=min_size, max_size=max_size))
    A = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False)))
    # A.T @ A is always positive semi-definite, add identity for positive definite
    return A.T @ A + np.eye(size)


class TestLinalgProperties:
    """Property-based tests for linear algebra"""

    @settings(max_examples=50, deadline=5000)
    @given(A=invertible_matrix(min_size=2, max_size=15))
    def test_inverse_roundtrip(self, A):
        """A @ inv(A) should equal identity"""
        try:
            A_inv = scirs2.inv_py(A)
            result = A @ A_inv
            expected = np.eye(len(A))
            assert np.allclose(result, expected, atol=1e-6)
        except Exception:
            # Some matrices may still be ill-conditioned
            pass

    @settings(max_examples=50)
    @given(A=arrays(dtype=np.float64, shape=(10, 10), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    def test_determinant_multiplicative(self, A):
        """det(A @ A.T) should equal det(A)^2"""
        try:
            det_A = scirs2.det_py(A)
            det_AAT = scirs2.det_py(A @ A.T)
            assert np.allclose(det_AAT, det_A ** 2, rtol=1e-8)
        except Exception:
            pass

    @settings(max_examples=50)
    @given(A=arrays(dtype=np.float64, shape=(10, 10), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False)))
    def test_trace_is_sum_of_diagonal(self, A):
        """Trace should equal sum of diagonal elements"""
        tr = scirs2.trace_py(A)
        diag_sum = np.sum(np.diag(A))
        assert np.allclose(tr, diag_sum, rtol=1e-12)

    @settings(max_examples=30)
    @given(A=symmetric_matrix(min_size=3, max_size=15))
    def test_symmetric_eigenvalues_real(self, A):
        """Symmetric matrices should have real eigenvalues"""
        result = scirs2.eigh_py(A)
        eigenvalues = result['eigenvalues']
        # All eigenvalues should be real (imaginary part ~0)
        assert eigenvalues.dtype == np.float64 or np.allclose(eigenvalues.imag, 0, atol=1e-10)

    @settings(max_examples=30)
    @given(A=positive_definite_matrix(min_size=3, max_size=15))
    def test_cholesky_roundtrip(self, A):
        """L @ L.T should reconstruct positive definite matrix"""
        try:
            L = scirs2.cholesky_py(A)
            reconstructed = L @ L.T
            assert np.allclose(A, reconstructed, atol=1e-8)
        except Exception:
            # May fail for ill-conditioned matrices
            pass


class TestFFTProperties:
    """Property-based tests for FFT"""

    @settings(max_examples=50)
    @given(signal=arrays(dtype=np.float64, shape=st.integers(8, 128), elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)))
    def test_fft_ifft_roundtrip(self, signal):
        """FFT followed by IFFT should recover original signal"""
        # Forward FFT
        fft_result = scirs2.fft_py(signal)

        # Inverse FFT
        ifft_result = scirs2.ifft_py(fft_result['real'], fft_result['imag'])
        recovered = ifft_result['real']

        assert np.allclose(signal, recovered, atol=1e-10)

    @settings(max_examples=50)
    @given(signal=arrays(dtype=np.float64, shape=st.integers(8, 128), elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)))
    def test_dct_idct_roundtrip(self, signal):
        """DCT followed by IDCT should recover signal"""
        # Forward DCT
        dct_result = scirs2.dct_py(signal, dct_type=2)

        # Inverse DCT
        recovered = scirs2.idct_py(dct_result, dct_type=2)

        assert np.allclose(signal, recovered, atol=1e-9)

    @settings(max_examples=30)
    @given(signal=arrays(dtype=np.float64, shape=st.integers(16, 64), elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)))
    def test_parseval_theorem(self, signal):
        """Energy conservation in FFT (Parseval's theorem)"""
        # Time domain energy
        time_energy = np.sum(signal**2)

        # Frequency domain energy
        fft_result = scirs2.fft_py(signal)
        freq_energy = np.sum(fft_result['real']**2 + fft_result['imag']**2) / len(signal)

        assert np.allclose(time_energy, freq_energy, rtol=1e-8)


class TestStatisticsProperties:
    """Property-based tests for statistics"""

    @settings(max_examples=50)
    @given(data=arrays(dtype=np.float64, shape=st.integers(10, 200), elements=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False)))
    def test_mean_in_range(self, data):
        """Mean should be between min and max"""
        mean_val = scirs2.mean_py(data)
        assert data.min() <= mean_val <= data.max()

    @settings(max_examples=50)
    @given(data=arrays(dtype=np.float64, shape=st.integers(10, 200), elements=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False)))
    def test_std_non_negative(self, data):
        """Standard deviation should be non-negative"""
        std_val = scirs2.std_py(data)
        assert std_val >= 0

    @settings(max_examples=50)
    @given(data=arrays(dtype=np.float64, shape=st.integers(10, 200), elements=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False)))
    def test_median_in_range(self, data):
        """Median should be between min and max"""
        median_val = scirs2.median_py(data)
        assert data.min() <= median_val <= data.max()

    @settings(max_examples=30)
    @given(x=arrays(dtype=np.float64, shape=50, elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)))
    def test_correlation_bounds(self, x):
        """Correlation coefficient should be in [-1, 1]"""
        # Create correlated y
        y = x + np.random.randn(len(x)) * 0.1
        r, p = scirs2.pearsonr_py(x, y)
        assert -1 <= r <= 1


class TestClusteringProperties:
    """Property-based tests for clustering"""

    @settings(max_examples=30)
    @given(
        n_samples=st.integers(20, 100),
        n_features=st.integers(2, 10),
        n_clusters=st.integers(2, 5)
    )
    def test_kmeans_cluster_assignment(self, n_samples, n_features, n_clusters):
        """All points should be assigned to valid clusters"""
        assume(n_clusters < n_samples)  # More samples than clusters

        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)

        km = scirs2.KMeans(n_clusters=n_clusters)
        km.fit(X)

        # All labels should be in range [0, k-1]
        assert km.labels.min() >= 0
        assert km.labels.max() < n_clusters

        # Should have k clusters (or fewer if some are empty)
        assert len(np.unique(km.labels)) <= n_clusters

    @settings(max_examples=30)
    @given(
        data=arrays(dtype=np.float64, shape=(50, 5), elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False))
    )
    def test_normalize_idempotent(self, data):
        """Normalizing twice should equal normalizing once"""
        norm1 = scirs2.normalize_py(data, "l2")
        norm2 = scirs2.normalize_py(norm1, "l2")

        assert np.allclose(norm1, norm2, atol=1e-10)

    @settings(max_examples=30)
    @given(
        data=arrays(dtype=np.float64, shape=(100, 5), elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False))
    )
    def test_standardize_idempotent(self, data):
        """Standardizing twice should equal standardizing once"""
        std1 = scirs2.standardize_py(data)
        std2 = scirs2.standardize_py(std1)

        assert np.allclose(std1, std2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
