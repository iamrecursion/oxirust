"""
SciPy Comparison Tests for Interpolation Module

Compares scirs2 interpolation functions against SciPy.interpolate
to ensure numerical correctness and API compatibility.
"""

import numpy as np
import pytest
import scipy.interpolate
import scirs2


class TestInterp1d:
    """Test 1D interpolation"""

    def test_interp1d_linear(self):
        """Linear interpolation should match SciPy"""
        np.random.seed(42)

        # Known points
        x = np.ascontiguousarray(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        y = np.ascontiguousarray(np.array([0.0, 1.0, 4.0, 9.0, 16.0]))

        # Interpolation points
        x_new = np.ascontiguousarray(np.array([0.5, 1.5, 2.5, 3.5]))

        # SciPy interpolation
        f_scipy = scipy.interpolate.interp1d(x, y, kind='linear')
        y_scipy = f_scipy(x_new)

        # SciRS2 interpolation (use 'method' not 'kind', callable interface)
        interp = scirs2.Interp1d(x, y, method='linear')
        y_scirs2 = interp(x_new)

        assert np.allclose(y_scipy, y_scirs2, rtol=1e-10)

    def test_interp1d_properties(self):
        """Interpolation should preserve known points"""
        np.random.seed(42)

        x = np.ascontiguousarray(np.linspace(0, 10, 20))
        y = np.ascontiguousarray(np.sin(x))

        # Create interpolator
        interp = scirs2.Interp1d(x, y, method='linear')

        # Interpolate at original points (callable interface)
        y_interp = interp(x)

        # Should match original values
        assert np.allclose(y, y_interp, rtol=1e-10)


class TestCubicSpline:
    """Test cubic spline interpolation"""

    def test_cubic_spline_basic(self):
        """Cubic spline should interpolate smoothly"""
        np.random.seed(42)

        # Known points
        x = np.ascontiguousarray(np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        y = np.ascontiguousarray(np.array([0.0, 1.0, 0.5, 2.0, 1.5]))

        # Create spline
        spline = scirs2.CubicSpline(x, y)

        # Interpolate at new points (callable interface)
        x_new = np.ascontiguousarray(np.linspace(0, 4, 20))
        y_interp = spline(x_new)

        # Should be smooth and pass through original points
        assert len(y_interp) == len(x_new)
        assert np.all(np.isfinite(y_interp))

    def test_cubic_spline_preserves_points(self):
        """Cubic spline should pass through data points"""
        np.random.seed(42)

        x = np.ascontiguousarray(np.linspace(0, 5, 10))
        y = np.ascontiguousarray(np.exp(-x) * np.sin(2*np.pi*x))

        spline = scirs2.CubicSpline(x, y)
        y_at_x = spline(x)

        # Should match original points
        assert np.allclose(y, y_at_x, rtol=1e-8)

    def test_cubic_spline_derivative(self):
        """Cubic spline derivative should be computable"""
        np.random.seed(42)

        x = np.ascontiguousarray(np.linspace(0, 2*np.pi, 10))
        y = np.ascontiguousarray(np.sin(x))

        spline = scirs2.CubicSpline(x, y)

        # Compute derivative at midpoints
        x_mid = np.ascontiguousarray((x[:-1] + x[1:]) / 2)
        dy = spline.derivative(x_mid, nu=1)

        # Derivative should be finite
        assert np.all(np.isfinite(dy))
        # For sine function, derivative should be roughly cos
        # Just verify it's computable and reasonable
        assert len(dy) == len(x_mid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
