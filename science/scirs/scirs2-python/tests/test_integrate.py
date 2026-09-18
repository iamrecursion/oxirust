"""Tests for scirs2 integration module."""

import pytest
import numpy as np
import scirs2


class TestTrapezoid:
    """Test trapezoidal integration."""

    def test_trapezoid_constant(self):
        """Test integrating constant function."""
        # Integral of 2 from 0 to 1 = 2
        y = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        result = scirs2.trapezoid_array_py(y, dx=0.25)
        assert abs(result - 2.0) < 1e-10

    def test_trapezoid_linear(self):
        """Test integrating linear function."""
        # Integral of x from 0 to 1 = 0.5
        x = np.linspace(0, 1, 101)
        y = x.copy()
        result = scirs2.trapezoid_array_py(y, x=x)
        assert abs(result - 0.5) < 0.001

    def test_trapezoid_quadratic(self):
        """Test integrating x^2."""
        # Integral of x^2 from 0 to 1 = 1/3
        x = np.linspace(0, 1, 1001)
        y = x**2
        result = scirs2.trapezoid_array_py(y, x=x)
        assert abs(result - 1/3) < 0.001

    def test_trapezoid_sine(self):
        """Test integrating sin(x)."""
        # Integral of sin(x) from 0 to pi = 2
        x = np.linspace(0, np.pi, 1001)
        y = np.sin(x)
        result = scirs2.trapezoid_array_py(y, x=x)
        assert abs(result - 2.0) < 0.001

    def test_trapezoid_with_dx(self):
        """Test using dx parameter."""
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # dx = 0.5, so x = [0, 0.5, 1.0, 1.5, 2.0]
        # Integral of linear function = 0.5 * (0 + 4) * 2 = 4
        result = scirs2.trapezoid_array_py(y, dx=0.5)
        assert abs(result - 4.0) < 0.001


class TestSimpson:
    """Test Simpson's rule integration."""

    def test_simpson_constant(self):
        """Test integrating constant function."""
        y = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        result = scirs2.simpson_array_py(y, dx=0.25)
        assert abs(result - 3.0) < 1e-10

    def test_simpson_quadratic(self):
        """Test integrating x^2 (Simpson should be exact for polynomials up to degree 3)."""
        x = np.linspace(0, 1, 101)
        y = x**2
        result = scirs2.simpson_array_py(y, x=x)
        assert abs(result - 1/3) < 0.0001

    def test_simpson_cubic(self):
        """Test integrating x^3."""
        # Integral of x^3 from 0 to 1 = 0.25
        x = np.linspace(0, 1, 101)
        y = x**3
        result = scirs2.simpson_array_py(y, x=x)
        assert abs(result - 0.25) < 0.0001

    def test_simpson_sine(self):
        """Test integrating sin(x)."""
        x = np.linspace(0, np.pi, 101)
        y = np.sin(x)
        result = scirs2.simpson_array_py(y, x=x)
        assert abs(result - 2.0) < 0.0001

    def test_simpson_exp(self):
        """Test integrating exp(x)."""
        # Integral of exp(x) from 0 to 1 = e - 1
        x = np.linspace(0, 1, 101)
        y = np.exp(x)
        result = scirs2.simpson_array_py(y, x=x)
        assert abs(result - (np.e - 1)) < 0.001


class TestCumulativeTrapezoid:
    """Test cumulative trapezoidal integration."""

    def test_cumulative_linear(self):
        """Test cumulative integral of linear function."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = x.copy()  # f(x) = x
        result = scirs2.cumulative_trapezoid_py(y, x=x)
        # Cumulative integral of x is x^2/2
        expected = np.array([0.5, 2.0, 4.5, 8.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_cumulative_with_initial(self):
        """Test cumulative integral with initial value."""
        y = np.array([0.0, 1.0, 2.0, 3.0])
        result = scirs2.cumulative_trapezoid_py(y, dx=1.0, initial=0.0)
        # With initial, result has same length as input
        assert len(result) == 4
        assert abs(result[0]) < 1e-10

    def test_cumulative_constant(self):
        """Test cumulative integral of constant."""
        y = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        result = scirs2.cumulative_trapezoid_py(y, dx=1.0)
        expected = np.array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestRomberg:
    """Test Romberg integration."""

    def test_romberg_quadratic(self):
        """Test Romberg on quadratic (should be very accurate)."""
        # Need 2^k + 1 points
        x = np.linspace(0, 1, 17)  # 2^4 + 1 = 17
        y = x**2
        dx = x[1] - x[0]
        result = scirs2.romberg_array_py(y, dx=dx)
        assert abs(result - 1/3) < 0.001

    def test_romberg_sine(self):
        """Test Romberg on sine function."""
        x = np.linspace(0, np.pi, 17)
        y = np.sin(x)
        dx = x[1] - x[0]
        result = scirs2.romberg_array_py(y, dx=dx)
        assert abs(result - 2.0) < 0.01


class TestEdgeCases:
    """Test edge cases."""

    def test_trapezoid_two_points(self):
        """Test with minimum number of points."""
        y = np.array([0.0, 1.0])
        result = scirs2.trapezoid_array_py(y, dx=1.0)
        assert abs(result - 0.5) < 1e-10

    def test_simpson_three_points(self):
        """Test Simpson with minimum points."""
        y = np.array([0.0, 1.0, 0.0])
        result = scirs2.simpson_array_py(y, dx=1.0)
        # Simpson's rule on these 3 points
        assert result > 0

    def test_irregular_spacing(self):
        """Test with non-uniform spacing."""
        x = np.array([0.0, 0.1, 0.5, 1.0])
        y = x**2
        result = scirs2.trapezoid_array_py(y, x=x)
        # Should still work with irregular spacing
        assert abs(result - 1/3) < 0.1


class TestQuad:
    """Test adaptive quadrature integration."""

    def test_quad_quadratic(self):
        """Test integrating x^2 from 0 to 1."""
        result = scirs2.quad_py(lambda x: x**2, 0.0, 1.0)
        assert result["success"]
        assert abs(result["value"] - 1/3) < 1e-6
        assert result["error"] < 1e-6

    def test_quad_sine(self):
        """Test integrating sin(x) from 0 to pi."""
        result = scirs2.quad_py(lambda x: np.sin(x), 0.0, np.pi)
        assert result["success"]
        assert abs(result["value"] - 2.0) < 1e-6

    def test_quad_exp(self):
        """Test integrating exp(x) from 0 to 1."""
        result = scirs2.quad_py(lambda x: np.exp(x), 0.0, 1.0)
        assert result["success"]
        assert abs(result["value"] - (np.e - 1)) < 1e-6

    def test_quad_sqrt(self):
        """Test integrating sqrt(x) from 0.01 to 1 (avoid singularity at 0)."""
        result = scirs2.quad_py(lambda x: np.sqrt(x), 0.01, 1.0)
        assert result["success"]
        # Integral of sqrt(x) from 0.01 to 1
        expected = 2.0/3.0 * (1.0**1.5 - 0.01**1.5)
        assert abs(result["value"] - expected) < 1e-4

    def test_quad_oscillatory(self):
        """Test integrating oscillatory function."""
        result = scirs2.quad_py(lambda x: np.sin(10*x), 0.0, 2*np.pi)
        assert result["success"]
        # Should be close to 0
        assert abs(result["value"]) < 1e-4

    def test_quad_polynomial(self):
        """Test integrating polynomial."""
        # Integral of x^3 + 2x from 0 to 2 = 4 + 4 = 8
        result = scirs2.quad_py(lambda x: x**3 + 2*x, 0.0, 2.0)
        assert result["success"]
        assert abs(result["value"] - 8.0) < 1e-6


class TestSolveIVP:
    """Test ODE solvers."""

    def test_solve_ivp_exponential_decay(self):
        """Test solving dy/dt = -y with y(0) = 1."""
        def f(t, y):
            return [-y[0]]

        result = scirs2.solve_ivp_py(
            f,
            t_span=(0.0, 2.0),
            y0=[1.0],
            method="RK45"
        )

        assert result["success"]
        assert len(result["t"]) > 0
        # y(t) = exp(-t), so y(2) ≈ 0.135
        y_final = result["y"][0, -1]
        assert abs(y_final - np.exp(-2.0)) < 0.01

    def test_solve_ivp_harmonic_oscillator(self):
        """Test solving harmonic oscillator: y'' + y = 0."""
        # Convert to system: dy1/dt = y2, dy2/dt = -y1
        def f(t, y):
            return [y[1], -y[0]]

        result = scirs2.solve_ivp_py(
            f,
            t_span=(0.0, 2*np.pi),
            y0=[1.0, 0.0],  # y(0) = 1, y'(0) = 0
            method="RK45"
        )

        assert result["success"]
        # Should return to initial position after one period
        y_final = result["y"][0, -1]
        assert abs(y_final - 1.0) < 0.1

    def test_solve_ivp_linear_growth(self):
        """Test solving dy/dt = t with y(0) = 0."""
        def f(t, y):
            return [t]

        result = scirs2.solve_ivp_py(
            f,
            t_span=(0.0, 1.0),
            y0=[0.0],
            method="RK45"
        )

        assert result["success"]
        # y(t) = t^2/2, so y(1) = 0.5
        y_final = result["y"][0, -1]
        assert abs(y_final - 0.5) < 0.01

    def test_solve_ivp_system(self):
        """Test solving system of ODEs."""
        # Lotka-Volterra predator-prey model (simplified)
        def f(t, y):
            alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
            prey, predator = y[0], y[1]
            return [
                alpha * prey - beta * prey * predator,
                delta * prey * predator - gamma * predator
            ]

        result = scirs2.solve_ivp_py(
            f,
            t_span=(0.0, 10.0),
            y0=[10.0, 5.0],
            method="RK45"
        )

        assert result["success"]
        assert result["y"].shape[0] == 2  # 2 variables
        # Both populations should remain positive
        assert np.all(result["y"] >= 0)

    def test_solve_ivp_rk23(self):
        """Test with RK23 method."""
        def f(t, y):
            return [-y[0]]

        result = scirs2.solve_ivp_py(
            f,
            t_span=(0.0, 1.0),
            y0=[1.0],
            method="RK23"
        )

        assert result["success"]
        y_final = result["y"][0, -1]
        assert abs(y_final - np.exp(-1.0)) < 0.05

    def test_solve_ivp_dop853(self):
        """Test with high-order DOP853 method."""
        def f(t, y):
            return [-y[0]]

        result = scirs2.solve_ivp_py(
            f,
            t_span=(0.0, 1.0),
            y0=[1.0],
            method="DOP853"
        )

        assert result["success"]
        y_final = result["y"][0, -1]
        # DOP853 should be accurate
        assert abs(y_final - np.exp(-1.0)) < 0.01
