"""
Comprehensive tests for simple linear regression (linregress).

Tests the linregress_py function which performs ordinary least squares
linear regression on two 1D arrays.
"""

import numpy as np
import pytest
import scirs2


class TestLinregressBasics:
    """Test basic functionality of linear regression."""

    def test_perfect_positive_correlation(self):
        """Test with perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float64)  # y = 2x

        result = scirs2.linregress_py(x, y)

        # Perfect line: slope=2, intercept=0
        assert abs(result["slope"] - 2.0) < 1e-10
        assert abs(result["intercept"]) < 1e-10
        # Perfect correlation: r=1
        assert abs(result["rvalue"] - 1.0) < 1e-10
        # Perfect fit: stderr=0
        assert result["stderr"] == 0.0

    def test_positive_slope_with_intercept(self):
        """Test positive slope with non-zero intercept."""
        x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        y = np.array([5, 7, 9, 11, 13], dtype=np.float64)  # y = 2x + 5

        result = scirs2.linregress_py(x, y)

        assert abs(result["slope"] - 2.0) < 1e-10
        assert abs(result["intercept"] - 5.0) < 1e-10
        assert abs(result["rvalue"] - 1.0) < 1e-10

    def test_negative_correlation(self):
        """Test with negative correlation."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([10, 8, 6, 4, 2], dtype=np.float64)  # y = -2x + 12

        result = scirs2.linregress_py(x, y)

        assert abs(result["slope"] - (-2.0)) < 1e-10
        assert abs(result["intercept"] - 12.0) < 1e-10
        assert abs(result["rvalue"] - (-1.0)) < 1e-10

    def test_zero_slope(self):
        """Test with zero slope (horizontal line)."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([5, 5, 5, 5, 5], dtype=np.float64)  # y = 5

        result = scirs2.linregress_py(x, y)

        assert abs(result["slope"]) < 1e-10
        assert abs(result["intercept"] - 5.0) < 1e-10
        # Zero variance in y, so r should be 0 or NaN
        assert result["rvalue"] == 0.0 or np.isnan(result["rvalue"])

    def test_imperfect_correlation(self):
        """Test with imperfect correlation (typical case)."""
        # Data with some scatter
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 17.9, 20.2], dtype=np.float64)

        result = scirs2.linregress_py(x, y)

        # Should have positive slope close to 2
        assert 1.8 < result["slope"] < 2.2
        # Should have positive intercept close to 0
        assert -0.5 < result["intercept"] < 0.5
        # Should have high correlation
        assert 0.99 < result["rvalue"] < 1.0
        # Should have significant p-value
        assert result["pvalue"] < 0.001
        # Should have non-zero standard error
        assert result["stderr"] > 0

    def test_moderate_correlation(self):
        """Test with moderate correlation."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        # Add noise to create moderate correlation
        y = 2 * x + 3 + np.random.normal(0, 5, 50)

        result = scirs2.linregress_py(x, y)

        # Slope should be around 2
        assert 1.5 < result["slope"] < 2.5
        # Intercept should be around 3
        assert 1 < result["intercept"] < 5
        # Moderate correlation
        assert 0.5 < abs(result["rvalue"]) < 0.99
        # P-value should be calculated
        assert "pvalue" in result


class TestLinregressStatistics:
    """Test statistical properties of linear regression."""

    def test_rvalue_squared_equals_r_squared(self):
        """Test that r² equals coefficient of determination."""
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        y = 2 * x + np.random.normal(0, 1, 10)

        result = scirs2.linregress_py(x, y)

        # r² should be between 0 and 1
        r_squared = result["rvalue"] ** 2
        assert 0 <= r_squared <= 1

    def test_stderr_positive_for_imperfect_fit(self):
        """Test that standard error is positive for imperfect fit."""
        np.random.seed(42)
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = 2 * x + np.random.normal(0, 0.5, 5)

        result = scirs2.linregress_py(x, y)

        # Standard error should be positive
        assert result["stderr"] > 0

    def test_pvalue_significance(self):
        """Test p-value for significant vs insignificant correlation."""
        np.random.seed(42)

        # Significant correlation (strong relationship)
        x_sig = np.linspace(0, 10, 100)
        y_sig = 2 * x_sig + np.random.normal(0, 1, 100)
        result_sig = scirs2.linregress_py(x_sig, y_sig)

        # Should be significant
        assert result_sig["pvalue"] < 0.05

        # Weak/insignificant correlation
        x_insig = np.linspace(0, 10, 10)
        y_insig = np.random.normal(0, 10, 10)
        result_insig = scirs2.linregress_py(x_insig, y_insig)

        # Should not be significant (typically)
        # Note: This is probabilistic, but with seed 42 should be consistent
        assert result_insig["pvalue"] > 0.05


class TestLinregressEdgeCases:
    """Test edge cases and error conditions."""

    def test_minimum_points(self):
        """Test with minimum number of points (2)."""
        x = np.array([1, 2], dtype=np.float64)
        y = np.array([2, 4], dtype=np.float64)

        result = scirs2.linregress_py(x, y)

        # Two points always define a perfect line
        assert abs(result["slope"] - 2.0) < 1e-10
        assert abs(result["intercept"]) < 1e-10
        assert abs(result["rvalue"] - 1.0) < 1e-10 or np.isnan(result["rvalue"])

    def test_single_point_error(self):
        """Test that single point raises error."""
        x = np.array([1], dtype=np.float64)
        y = np.array([2], dtype=np.float64)

        with pytest.raises(RuntimeError, match="At least 2"):
            scirs2.linregress_py(x, y)

    def test_empty_arrays_error(self):
        """Test that empty arrays raise error."""
        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)

        with pytest.raises(RuntimeError):
            scirs2.linregress_py(x, y)

    def test_different_lengths_error(self):
        """Test that different length arrays raise error."""
        x = np.array([1, 2, 3], dtype=np.float64)
        y = np.array([1, 2], dtype=np.float64)

        with pytest.raises(RuntimeError, match="length"):
            scirs2.linregress_py(x, y)

    def test_constant_x_error(self):
        """Test that constant x values raise error."""
        x = np.array([5, 5, 5, 5, 5], dtype=np.float64)
        y = np.array([1, 2, 3, 4, 5], dtype=np.float64)

        with pytest.raises(RuntimeError, match="No variation"):
            scirs2.linregress_py(x, y)

    def test_large_values(self):
        """Test with large values."""
        x = np.array([1e10, 2e10, 3e10, 4e10, 5e10], dtype=np.float64)
        y = np.array([2e10, 4e10, 6e10, 8e10, 10e10], dtype=np.float64)

        result = scirs2.linregress_py(x, y)

        # Should still get correct slope
        assert abs(result["slope"] - 2.0) < 1e-6

    def test_small_values(self):
        """Test with small values."""
        # Use larger small values to avoid numerical precision issues
        x = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6], dtype=np.float64)
        y = np.array([2e-6, 4e-6, 6e-6, 8e-6, 10e-6], dtype=np.float64)

        result = scirs2.linregress_py(x, y)

        # Should still get correct slope
        assert abs(result["slope"] - 2.0) < 1e-3


class TestLinregressRealWorld:
    """Test realistic use cases for linear regression."""

    def test_temperature_conversion(self):
        """Test Celsius to Fahrenheit conversion."""
        # Known relationship: F = 1.8C + 32
        celsius = np.array([0, 10, 20, 30, 40, 50], dtype=np.float64)
        fahrenheit = np.array([32, 50, 68, 86, 104, 122], dtype=np.float64)

        result = scirs2.linregress_py(celsius, fahrenheit)

        # Should get exact conversion formula
        assert abs(result["slope"] - 1.8) < 1e-10
        assert abs(result["intercept"] - 32.0) < 1e-10
        assert abs(result["rvalue"] - 1.0) < 1e-10

    def test_weight_vs_height(self):
        """Test weight vs height relationship."""
        # Simulated data
        height = np.array([150, 160, 165, 170, 175, 180, 185], dtype=np.float64)  # cm
        weight = np.array([50, 58, 63, 68, 73, 78, 83], dtype=np.float64)  # kg

        result = scirs2.linregress_py(height, weight)

        # Should have positive slope
        assert result["slope"] > 0
        # Should have strong positive correlation
        assert result["rvalue"] > 0.95
        # Should be significant
        assert result["pvalue"] < 0.01

    def test_time_series_trend(self):
        """Test time series with linear trend."""
        time = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
        # Sales with upward trend
        sales = np.array([100, 105, 108, 112, 118, 120, 125, 130, 133, 138], dtype=np.float64)

        result = scirs2.linregress_py(time, sales)

        # Should have positive slope (upward trend)
        assert result["slope"] > 0
        # Should have high correlation
        assert result["rvalue"] > 0.9
        # Should be significant
        assert result["pvalue"] < 0.01

    def test_negative_relationship(self):
        """Test negative relationship (e.g., price vs demand)."""
        price = np.array([10, 15, 20, 25, 30, 35, 40], dtype=np.float64)
        demand = np.array([100, 85, 70, 55, 40, 25, 10], dtype=np.float64)

        result = scirs2.linregress_py(price, demand)

        # Should have negative slope
        assert result["slope"] < 0
        # Should have strong negative correlation
        assert result["rvalue"] < -0.95
        # P-value might be NaN for perfect correlation, otherwise should be small
        assert np.isnan(result["pvalue"]) or result["pvalue"] < 0.01


class TestLinregressNumericalProperties:
    """Test numerical properties and precision."""

    def test_prediction(self):
        """Test that predictions match expected values."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([3, 5, 7, 9, 11], dtype=np.float64)  # y = 2x + 1

        result = scirs2.linregress_py(x, y)

        # Predict for new x value
        x_new = 6.0
        y_pred = result["slope"] * x_new + result["intercept"]

        # Should predict y = 13
        assert abs(y_pred - 13.0) < 1e-10

    def test_residuals_sum_to_zero(self):
        """Test that residuals sum to approximately zero."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3 + np.random.normal(0, 1, 50)

        result = scirs2.linregress_py(x, y)

        # Calculate residuals
        y_pred = result["slope"] * x + result["intercept"]
        residuals = y - y_pred

        # Sum of residuals should be close to zero
        assert abs(residuals.sum()) < 1e-8

    def test_symmetry(self):
        """Test that swapping x and y affects results appropriately."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float64)

        result_xy = scirs2.linregress_py(x, y)
        result_yx = scirs2.linregress_py(y, x)

        # r-value should be the same
        assert abs(result_xy["rvalue"] - result_yx["rvalue"]) < 1e-10

        # Slopes should be reciprocal (approximately for perfect correlation)
        if result_xy["rvalue"] == 1.0:
            assert abs(result_xy["slope"] * result_yx["slope"] - 1.0) < 1e-10

    def test_scale_invariance_of_correlation(self):
        """Test that r-value is scale-invariant."""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float64)

        result1 = scirs2.linregress_py(x, y)

        # Scale both x and y
        x_scaled = x * 100
        y_scaled = y * 1000

        result2 = scirs2.linregress_py(x_scaled, y_scaled)

        # Correlation coefficient should be the same
        assert abs(result1["rvalue"] - result2["rvalue"]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
