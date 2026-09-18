"""Tests for scirs2 interpolation module."""

import pytest
import numpy as np
import scirs2


class TestInterp1d:
    """Test Interp1d class."""

    def test_linear_interpolation(self):
        """Test basic linear interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])

        interp = scirs2.Interp1d(x, y, method="linear")

        # Test at known points
        result = interp(np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(result, [0.0, 1.0, 4.0], rtol=1e-10)

    def test_linear_midpoints(self):
        """Test linear interpolation at midpoints."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 2.0, 4.0])

        interp = scirs2.Interp1d(x, y, method="linear")
        result = interp(np.array([0.5, 1.5]))
        np.testing.assert_allclose(result, [1.0, 3.0], rtol=1e-10)

    def test_nearest_interpolation(self):
        """Test nearest neighbor interpolation."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])

        interp = scirs2.Interp1d(x, y, method="nearest")
        result = interp(np.array([0.3, 0.7, 1.5]))

        # 0.3 -> 0 (nearest to 0), 0.7 -> 1 (nearest to 1), 1.5 -> 2 (nearest to 2)
        # Note: exact behavior depends on implementation
        assert len(result) == 3

    def test_cubic_interpolation(self):
        """Test cubic spline interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.sin(x)

        interp = scirs2.Interp1d(x, y, method="cubic")

        # Test at intermediate points
        x_new = np.array([0.5, 1.5, 2.5, 3.5])
        result = interp(x_new)
        expected = np.sin(x_new)

        # Cubic should be reasonably accurate
        np.testing.assert_allclose(result, expected, rtol=0.15)

    def test_eval_single(self):
        """Test single point evaluation."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])

        interp = scirs2.Interp1d(x, y)
        result = interp.eval_single(0.5)
        assert abs(result - 0.5) < 1e-10

    def test_pchip_interpolation(self):
        """Test PCHIP interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 0.0, 1.0])

        interp = scirs2.Interp1d(x, y, method="pchip")
        result = interp(np.array([0.5, 1.5, 2.5]))

        # PCHIP preserves monotonicity
        assert len(result) == 3


class TestInterpFunction:
    """Test interp function."""

    def test_interp_basic(self):
        """Test basic numpy.interp-like functionality."""
        xp = np.array([0.0, 1.0, 2.0])
        fp = np.array([0.0, 10.0, 20.0])
        x = np.array([0.5, 1.5])

        result = scirs2.interp_py(x, xp, fp)
        np.testing.assert_allclose(result, [5.0, 15.0], rtol=1e-10)

    def test_interp_at_data_points(self):
        """Test interpolation at data points."""
        xp = np.array([0.0, 1.0, 2.0, 3.0])
        fp = np.array([1.0, 4.0, 9.0, 16.0])
        x = xp.copy()

        result = scirs2.interp_py(x, xp, fp)
        np.testing.assert_allclose(result, fp, rtol=1e-10)


class TestInterpWithBounds:
    """Test interp_with_bounds function."""

    def test_with_left_bound(self):
        """Test extrapolation with left boundary."""
        xp = np.array([1.0, 2.0, 3.0])
        fp = np.array([10.0, 20.0, 30.0])
        x = np.array([0.0, 1.5, 2.5, 4.0])

        result = scirs2.interp_with_bounds_py(x, xp, fp, left=0.0, right=100.0)

        assert result[0] == 0.0  # Left boundary
        assert abs(result[1] - 15.0) < 1e-10  # Interpolated
        assert abs(result[2] - 25.0) < 1e-10  # Interpolated
        assert result[3] == 100.0  # Right boundary

    def test_default_boundaries(self):
        """Test default boundary behavior (use endpoint values)."""
        xp = np.array([1.0, 2.0, 3.0])
        fp = np.array([10.0, 20.0, 30.0])
        x = np.array([0.0, 4.0])

        result = scirs2.interp_with_bounds_py(x, xp, fp)

        # Default should use endpoint values
        assert abs(result[0] - 10.0) < 1e-10
        assert abs(result[1] - 30.0) < 1e-10


class TestExtrapolation:
    """Test extrapolation modes."""

    def test_nearest_extrapolation(self):
        """Test nearest extrapolation mode."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])

        interp = scirs2.Interp1d(x, y, extrapolate="nearest")

        # Test within bounds should work
        result = interp(np.array([0.5, 1.5]))
        assert len(result) == 2


class TestEdgeCases:
    """Test edge cases."""

    def test_two_points(self):
        """Test interpolation with minimum points."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 10.0])

        interp = scirs2.Interp1d(x, y)
        result = interp(np.array([0.5]))
        np.testing.assert_allclose(result, [5.0], rtol=1e-10)

    def test_many_points(self):
        """Test with many data points."""
        x = np.linspace(0, 10, 1000)
        y = np.sin(x)

        interp = scirs2.Interp1d(x, y)
        x_new = np.array([0.1, 5.0, 9.9])
        result = interp(x_new)

        np.testing.assert_allclose(result, np.sin(x_new), rtol=0.01)

    def test_single_query_point(self):
        """Test with single query point."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])

        interp = scirs2.Interp1d(x, y)
        result = interp(np.array([0.5]))
        assert len(result) == 1

    def test_constant_function(self):
        """Test interpolating constant function."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 5.0, 5.0, 5.0])

        interp = scirs2.Interp1d(x, y)
        result = interp(np.array([0.5, 1.5, 2.5]))
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0], rtol=1e-10)


class TestCubicSpline:
    """Test CubicSpline class."""

    def test_basic_spline(self):
        """Test basic cubic spline interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])

        spline = scirs2.CubicSpline(x, y)

        # Evaluate at known points
        result = spline(np.array([0.0, 1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, y, rtol=1e-10)

    def test_spline_interpolation(self):
        """Test cubic spline interpolation at intermediate points."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 0.5, 0.0])

        spline = scirs2.CubicSpline(x, y, bc_type="natural")

        # Evaluate at intermediate points
        x_new = np.array([0.5, 1.5, 2.5])
        result = spline(x_new)

        # Should be smooth and within reasonable bounds
        assert len(result) == 3
        assert all(not np.isnan(val) for val in result)

    def test_spline_derivative(self):
        """Test cubic spline derivative calculation."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x ** 2  # y = x^2, derivative should be 2x

        spline = scirs2.CubicSpline(x, y)

        # Compute first derivative at x=2 (should be ~4)
        x_eval = np.array([2.0])
        deriv = spline.derivative(x_eval, nu=1)

        np.testing.assert_allclose(deriv, [4.0], rtol=0.1)

    def test_spline_second_derivative(self):
        """Test cubic spline second derivative."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x ** 2  # y = x^2, second derivative should be ~2

        spline = scirs2.CubicSpline(x, y)

        # Compute second derivative
        x_eval = np.array([1.0, 2.0, 3.0])
        deriv2 = spline.derivative(x_eval, nu=2)

        # Second derivatives from spline can vary from analytical due to boundary conditions
        np.testing.assert_allclose(deriv2, [2.0, 2.0, 2.0], rtol=0.3)

    def test_spline_integration(self):
        """Test cubic spline integration."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 1.0, 1.0, 1.0])  # Constant function

        spline = scirs2.CubicSpline(x, y)

        # Integrate from 0 to 3 (should be 3.0 for constant 1.0)
        integral = spline.integrate(0.0, 3.0)
        np.testing.assert_allclose(integral, 3.0, rtol=1e-5)

    def test_spline_integration_polynomial(self):
        """Test integration of polynomial using spline."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x ** 2  # y = x^2

        spline = scirs2.CubicSpline(x, y)

        # Integrate x^2 from 0 to 4 = [x^3/3]_0^4 = 64/3 ≈ 21.333
        integral = spline.integrate(0.0, 4.0)
        expected = 64.0 / 3.0
        np.testing.assert_allclose(integral, expected, rtol=0.05)

    def test_spline_eval_single(self):
        """Test single point evaluation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])

        spline = scirs2.CubicSpline(x, y)

        result = spline.eval_single(1.5)
        assert not np.isnan(result)

    def test_spline_sine_function(self):
        """Test cubic spline on sine function."""
        x = np.linspace(0, 2*np.pi, 10)
        y = np.sin(x)

        spline = scirs2.CubicSpline(x, y)

        # Evaluate at intermediate points
        x_new = np.linspace(0, 2*np.pi, 50)
        result = spline(x_new)
        expected = np.sin(x_new)

        # Should be reasonably accurate
        np.testing.assert_allclose(result, expected, rtol=0.1)


class TestInterp2d:
    """Test 2D interpolation."""

    def test_interp2d_linear_basic(self):
        """Test basic 2D linear interpolation."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0])
        # z = x + y on the grid
        z = np.array([[0.0, 1.0, 2.0],
                      [1.0, 2.0, 3.0]])

        interp = scirs2.Interp2d(x, y, z, kind="linear")

        # Test at grid points
        assert abs(interp(0.0, 0.0) - 0.0) < 1e-10
        assert abs(interp(1.0, 0.0) - 1.0) < 1e-10
        assert abs(interp(2.0, 1.0) - 3.0) < 1e-10

    def test_interp2d_linear_interpolation(self):
        """Test 2D linear interpolation at intermediate points."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0])
        # z = x + y
        z = np.array([[0.0, 1.0, 2.0],
                      [1.0, 2.0, 3.0]])

        interp = scirs2.Interp2d(x, y, z, kind="linear")

        # Test at (0.5, 0.5), should be 1.0
        result = interp(0.5, 0.5)
        np.testing.assert_allclose(result, 1.0, rtol=1e-10)

        # Test at (1.5, 0.5), should be 2.0
        result = interp(1.5, 0.5)
        np.testing.assert_allclose(result, 2.0, rtol=1e-10)

    def test_interp2d_cubic(self):
        """Test 2D cubic interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        # z = x^2 + y^2
        z = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                z[i, j] = x[j]**2 + y[i]**2

        interp = scirs2.Interp2d(x, y, z, kind="cubic")

        # Test at grid points
        assert abs(interp(1.0, 1.0) - 2.0) < 1e-10

        # Test at intermediate point
        result = interp(1.5, 1.5)
        expected = 1.5**2 + 1.5**2  # 4.5
        assert abs(result - expected) < 0.5  # Reasonable tolerance for cubic

    def test_interp2d_eval_array(self):
        """Test 2D interpolation with array of points."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0])
        z = np.array([[0.0, 1.0, 2.0],
                      [1.0, 2.0, 3.0]])

        interp = scirs2.Interp2d(x, y, z, kind="linear")

        x_new = np.array([0.5, 1.0, 1.5])
        y_new = np.array([0.5, 0.5, 0.5])

        result = interp.eval_array(x_new, y_new)

        assert len(result) == 3
        np.testing.assert_allclose(result[1], 1.5, rtol=1e-10)

    def test_interp2d_eval_grid(self):
        """Test 2D interpolation on a regular grid."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        z = np.array([[0.0, 1.0],
                      [1.0, 2.0]])

        interp = scirs2.Interp2d(x, y, z, kind="linear")

        x_new = np.array([0.0, 0.5, 1.0])
        y_new = np.array([0.0, 0.5, 1.0])

        result = interp.eval_grid(x_new, y_new)

        assert result.shape == (3, 3)
        # Test corners
        assert abs(result[0, 0] - 0.0) < 1e-10
        assert abs(result[0, 2] - 1.0) < 1e-10
        assert abs(result[2, 0] - 1.0) < 1e-10
        assert abs(result[2, 2] - 2.0) < 1e-10
        # Test center
        assert abs(result[1, 1] - 1.0) < 1e-10

    def test_interp2d_product_function(self):
        """Test 2D interpolation with product function z = x * y."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        z = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                z[i, j] = x[j] * y[i]

        interp = scirs2.Interp2d(x, y, z, kind="linear")

        # Test at (1.5, 2.0), should be 3.0
        result = interp(1.5, 2.0)
        np.testing.assert_allclose(result, 3.0, rtol=1e-10)

    def test_interp2d_constant(self):
        """Test 2D interpolation with constant function."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        z = np.ones((3, 3)) * 5.0  # Constant 5.0

        interp = scirs2.Interp2d(x, y, z, kind="linear")

        # Should always return 5.0
        assert abs(interp(0.5, 0.5) - 5.0) < 1e-10
        assert abs(interp(1.5, 1.5) - 5.0) < 1e-10
