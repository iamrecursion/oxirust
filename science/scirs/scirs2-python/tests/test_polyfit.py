"""
Comprehensive tests for polynomial regression (polyfit)
"""

import numpy as np
import pytest
import scirs2


class TestPolyfitBasics:
    """Basic functionality tests for polyfit"""

    def test_linear_fit(self):
        """Test fitting a linear polynomial (degree 1)"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # y = 2x

        result = scirs2.polyfit_py(x, y, deg=1)

        assert "coefficients" in result
        assert "r_squared" in result
        assert len(result["coefficients"]) == 2

        # Check coefficients (intercept, slope)
        assert abs(result["coefficients"][0] - 0.0) < 0.01  # intercept ≈ 0
        assert abs(result["coefficients"][1] - 2.0) < 0.01  # slope ≈ 2

        # Check R-squared (should be perfect fit)
        assert result["r_squared"] > 0.99

    def test_quadratic_fit(self):
        """Test fitting a quadratic polynomial (degree 2)"""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 3.0, 9.0, 19.0, 33.0])  # y ≈ 1 + 2x + x^2

        result = scirs2.polyfit_py(x, y, deg=2)

        assert len(result["coefficients"]) == 3

        # Check coefficients (c0, c1, c2)
        assert abs(result["coefficients"][0] - 1.0) < 0.01  # constant ≈ 1
        assert abs(result["coefficients"][1] - 2.0) < 0.01  # linear ≈ 2
        assert abs(result["coefficients"][2] - 1.0) < 0.01  # quadratic ≈ 1

        # Check R-squared
        assert result["r_squared"] > 0.99

    def test_cubic_fit(self):
        """Test fitting a cubic polynomial (degree 3)"""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0.0, 1.0, 8.0, 27.0, 64.0, 125.0])  # y = x^3

        result = scirs2.polyfit_py(x, y, deg=3)

        assert len(result["coefficients"]) == 4

        # Check that cubic coefficient is close to 1
        assert abs(result["coefficients"][3] - 1.0) < 0.01

        # Check R-squared
        assert result["r_squared"] > 0.99

    def test_constant_fit(self):
        """Test fitting a constant polynomial (degree 0)"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])  # y = 5

        result = scirs2.polyfit_py(x, y, deg=0)

        assert len(result["coefficients"]) == 1
        assert abs(result["coefficients"][0] - 5.0) < 0.01


class TestPolyfitOutputs:
    """Test all output fields of polyfit"""

    def test_output_fields(self):
        """Test that all expected fields are present"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = scirs2.polyfit_py(x, y, deg=1)

        required_fields = [
            "coefficients",
            "r_squared",
            "adj_r_squared",
            "residuals",
            "fitted_values",
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_residuals(self):
        """Test that residuals are calculated correctly"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 3.9, 6.1, 7.9, 10.1])  # Close to y = 2x

        result = scirs2.polyfit_py(x, y, deg=1)

        residuals = result["residuals"]
        fitted_values = result["fitted_values"]

        # Check that residuals = y - fitted_values
        assert len(residuals) == len(y)
        for i in range(len(y)):
            expected_residual = y[i] - fitted_values[i]
            assert abs(residuals[i] - expected_residual) < 0.01

    def test_fitted_values(self):
        """Test that fitted values match predictions"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = scirs2.polyfit_py(x, y, deg=1)

        fitted_values = result["fitted_values"]
        coefficients = result["coefficients"]

        # Manually calculate fitted values: c0 + c1*x
        for i, xi in enumerate(x):
            expected = coefficients[0] + coefficients[1] * xi
            assert abs(fitted_values[i] - expected) < 0.01

    def test_adjusted_r_squared(self):
        """Test that adjusted R-squared is calculated"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

        result = scirs2.polyfit_py(x, y, deg=1)

        # Adjusted R-squared should be slightly less than R-squared
        assert "adj_r_squared" in result
        assert result["adj_r_squared"] <= result["r_squared"]
        assert result["adj_r_squared"] > 0.99  # Should still be very high


class TestPolyfitEdgeCases:
    """Edge cases and error handling"""

    def test_perfect_fit(self):
        """Test polynomial fit with perfect data"""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        # Generate perfect quadratic: y = 1 + 2x + 3x^2
        y = 1.0 + 2.0 * x + 3.0 * x**2

        result = scirs2.polyfit_py(x, y, deg=2)

        # Should be perfect fit
        assert result["r_squared"] > 0.9999
        assert abs(result["coefficients"][0] - 1.0) < 0.0001
        assert abs(result["coefficients"][1] - 2.0) < 0.0001
        assert abs(result["coefficients"][2] - 3.0) < 0.0001

    def test_noisy_data(self):
        """Test polynomial fit with noisy data"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.0 + 3.0 * x + 0.5 * x**2 + np.random.normal(0, 2.0, len(x))

        result = scirs2.polyfit_py(x, y, deg=2)

        # Should still have reasonable fit despite noise
        assert result["r_squared"] > 0.8

        # Coefficients should be approximately correct
        assert abs(result["coefficients"][0] - 2.0) < 2.0
        assert abs(result["coefficients"][1] - 3.0) < 2.0
        assert abs(result["coefficients"][2] - 0.5) < 1.0

    def test_minimum_data_points(self):
        """Test with minimum number of data points"""
        # For degree d, we need at least d+1 points
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 4.0, 9.0])  # y = x^2

        result = scirs2.polyfit_py(x, y, deg=2)

        assert len(result["coefficients"]) == 3
        assert result["r_squared"] >= 0.0

    def test_large_degree(self):
        """Test with large polynomial degree"""
        x = np.linspace(0, 10, 20)
        y = np.sin(x)

        # Fit high-degree polynomial
        result = scirs2.polyfit_py(x, y, deg=10)

        assert len(result["coefficients"]) == 11
        assert result["r_squared"] >= 0.0

    def test_negative_values(self):
        """Test with negative x and y values"""
        x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        y = x**2  # y = x^2

        result = scirs2.polyfit_py(x, y, deg=2)

        # Should fit parabola correctly
        assert abs(result["coefficients"][0]) < 0.01  # constant ≈ 0
        assert abs(result["coefficients"][1]) < 0.01  # linear ≈ 0
        assert abs(result["coefficients"][2] - 1.0) < 0.01  # quadratic ≈ 1


class TestPolyfitNumericalProperties:
    """Numerical stability and accuracy tests"""

    def test_large_values(self):
        """Test with large x and y values"""
        x = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y = 2.0 * x + 100.0

        result = scirs2.polyfit_py(x, y, deg=1)

        assert result["r_squared"] > 0.99
        assert abs(result["coefficients"][1] - 2.0) < 0.1

    def test_small_values(self):
        """Test with small x and y values"""
        x = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        y = 2000.0 * x + 0.5

        result = scirs2.polyfit_py(x, y, deg=1)

        assert result["r_squared"] > 0.99

    def test_different_scales(self):
        """Test with x and y on very different scales"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x * 1000.0

        result = scirs2.polyfit_py(x, y, deg=1)

        assert result["r_squared"] > 0.99
        assert abs(result["coefficients"][1] - 1000.0) < 10.0


class TestPolyfitRealWorldScenarios:
    """Real-world application scenarios"""

    def test_temperature_conversion(self):
        """Test polynomial fit for temperature conversion (should be linear)"""
        # Celsius to Fahrenheit: F = 1.8*C + 32
        celsius = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 100.0])
        fahrenheit = np.array([32.0, 50.0, 68.0, 86.0, 104.0, 212.0])

        result = scirs2.polyfit_py(celsius, fahrenheit, deg=1)

        assert abs(result["coefficients"][0] - 32.0) < 0.1  # intercept
        assert abs(result["coefficients"][1] - 1.8) < 0.1  # slope
        assert result["r_squared"] > 0.9999

    def test_population_growth(self):
        """Test quadratic fit for population growth"""
        years = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        # Approximate quadratic growth
        population = 100.0 + 10.0 * years + 2.0 * years**2

        result = scirs2.polyfit_py(years, population, deg=2)

        assert result["r_squared"] > 0.99
        assert abs(result["coefficients"][0] - 100.0) < 1.0
        assert abs(result["coefficients"][1] - 10.0) < 1.0
        assert abs(result["coefficients"][2] - 2.0) < 1.0

    def test_trajectory_modeling(self):
        """Test parabolic trajectory (physics application)"""
        # Projectile motion: y = h0 + v0*t - 0.5*g*t^2
        # where h0 = 10, v0 = 20, g = 9.8
        t = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        h = 10.0 + 20.0 * t - 0.5 * 9.8 * t**2

        result = scirs2.polyfit_py(t, h, deg=2)

        assert result["r_squared"] > 0.9999
        assert abs(result["coefficients"][0] - 10.0) < 0.01  # initial height
        assert abs(result["coefficients"][1] - 20.0) < 0.01  # initial velocity
        assert abs(result["coefficients"][2] - (-4.9)) < 0.01  # -0.5*g

    def test_signal_approximation(self):
        """Test approximating a sinusoidal signal"""
        x = np.linspace(0, np.pi, 10)
        y = np.sin(x)

        # Use moderate degree polynomial
        result = scirs2.polyfit_py(x, y, deg=5)

        # Should provide reasonable approximation
        assert result["r_squared"] > 0.9

    def test_data_smoothing(self):
        """Test using polyfit for data smoothing"""
        np.random.seed(42)
        x = np.linspace(0, 10, 30)
        y_true = 5.0 + 2.0 * x
        y_noisy = y_true + np.random.normal(0, 1.0, len(x))

        # Fit linear model to smooth data
        result = scirs2.polyfit_py(x, y_noisy, deg=1)

        # Fitted values should be smoother than noisy data
        fitted = np.array(result["fitted_values"])

        # Check that fitted line is close to true line
        for i, xi in enumerate(x):
            expected = 5.0 + 2.0 * xi
            assert abs(fitted[i] - expected) < 2.0  # Within noise range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
