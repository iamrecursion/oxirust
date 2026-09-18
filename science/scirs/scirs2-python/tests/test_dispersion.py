"""
Comprehensive tests for dispersion and variability measures
"""

import numpy as np
import pytest
import scirs2


class TestDispersionBasics:
    """Basic functionality tests for dispersion measures"""

    def test_mean_abs_deviation_basic(self):
        """Test mean absolute deviation with simple data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # MAD from mean (3.0): |1-3|+|2-3|+|3-3|+|4-3|+|5-3| = 2+1+0+1+2 = 6, avg = 1.2
        result = scirs2.mean_abs_deviation_py(data)
        assert abs(result - 1.2) < 0.001

    def test_mean_abs_deviation_with_center(self):
        """Test mean absolute deviation with custom center"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # MAD from center 2.0: |1-2|+|2-2|+|3-2|+|4-2|+|5-2| = 1+0+1+2+3 = 7, avg = 1.4
        result = scirs2.mean_abs_deviation_py(data, center=2.0)
        assert abs(result - 1.4) < 0.001

    def test_median_abs_deviation_basic(self):
        """Test median absolute deviation with simple data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Median is 3.0, deviations: [2, 1, 0, 1, 2], median of deviations is 1.0
        result = scirs2.median_abs_deviation_py(data)
        assert abs(result - 1.0) < 0.001

    def test_median_abs_deviation_with_center(self):
        """Test median absolute deviation with custom center"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Center 2.0, deviations: [1, 0, 1, 2, 3], median is 1.0
        result = scirs2.median_abs_deviation_py(data, center=2.0)
        assert abs(result - 1.0) < 0.001

    def test_median_abs_deviation_with_scale(self):
        """Test median absolute deviation with scale factor"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Default MAD is 1.0, with scale=2.0 should be 2.0
        result = scirs2.median_abs_deviation_py(data, scale=2.0)
        assert abs(result - 2.0) < 0.001

    def test_data_range_basic(self):
        """Test data range with simple data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Range = max - min = 5.0 - 1.0 = 4.0
        result = scirs2.data_range_py(data)
        assert abs(result - 4.0) < 0.001

    def test_coef_variation_basic(self):
        """Test coefficient of variation with simple data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Mean = 3.0, std (ddof=1) ≈ 1.581, CV ≈ 0.527
        result = scirs2.coef_variation_py(data)
        assert abs(result - 0.527) < 0.01

    def test_coef_variation_with_ddof(self):
        """Test coefficient of variation with different ddof"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # With ddof=0, std should be smaller, CV should be smaller
        result_ddof0 = scirs2.coef_variation_py(data, ddof=0)
        result_ddof1 = scirs2.coef_variation_py(data, ddof=1)

        assert result_ddof0 < result_ddof1

    def test_gini_coefficient_basic(self):
        """Test Gini coefficient with simple data"""
        # Equal distribution
        equal_data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result_equal = scirs2.gini_coefficient_py(equal_data)
        assert abs(result_equal - 0.0) < 0.001  # Perfect equality

        # Unequal distribution
        unequal_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result_unequal = scirs2.gini_coefficient_py(unequal_data)
        assert result_unequal > 0.2  # Should show inequality


class TestDispersionOptionalParameters:
    """Test optional parameters for dispersion measures"""

    def test_mean_abs_deviation_none_center(self):
        """Test MAD with None center (should use mean)"""
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result_none = scirs2.mean_abs_deviation_py(data, center=None)
        result_explicit = scirs2.mean_abs_deviation_py(data, center=6.0)  # Mean is 6.0

        assert abs(result_none - result_explicit) < 0.001

    def test_median_abs_deviation_both_params(self):
        """Test median absolute deviation with both center and scale"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.median_abs_deviation_py(data, center=3.0, scale=1.4826)
        # MAD with center 3.0 is 1.0, scaled by 1.4826 ≈ 1.4826
        assert abs(result - 1.4826) < 0.001

    def test_coef_variation_ddof_values(self):
        """Test coefficient of variation with various ddof values"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        # Test ddof=0, 1, 2
        result_0 = scirs2.coef_variation_py(data, ddof=0)
        result_1 = scirs2.coef_variation_py(data, ddof=1)
        result_2 = scirs2.coef_variation_py(data, ddof=2)

        # Higher ddof should give higher CV (larger std)
        assert result_0 < result_1 < result_2


class TestDispersionEdgeCases:
    """Edge cases and special conditions"""

    def test_single_value(self):
        """Test with single value"""
        data = np.array([5.0])

        # MAD should be 0 (only one value)
        mad = scirs2.mean_abs_deviation_py(data)
        assert abs(mad - 0.0) < 0.001

        # Median MAD should be 0
        median_mad = scirs2.median_abs_deviation_py(data)
        assert abs(median_mad - 0.0) < 0.001

        # Range should be 0
        range_val = scirs2.data_range_py(data)
        assert abs(range_val - 0.0) < 0.001

    def test_identical_values(self):
        """Test with all identical values"""
        data = np.array([7.0, 7.0, 7.0, 7.0, 7.0])

        # MAD should be 0
        mad = scirs2.mean_abs_deviation_py(data)
        assert abs(mad - 0.0) < 0.001

        # Median MAD should be 0
        median_mad = scirs2.median_abs_deviation_py(data)
        assert abs(median_mad - 0.0) < 0.001

        # Range should be 0
        range_val = scirs2.data_range_py(data)
        assert abs(range_val - 0.0) < 0.001

        # CV should be 0 (no variation)
        cv = scirs2.coef_variation_py(data)
        assert abs(cv - 0.0) < 0.001

        # Gini should be 0 (perfect equality)
        gini = scirs2.gini_coefficient_py(data)
        assert abs(gini - 0.0) < 0.001

    def test_two_values(self):
        """Test with exactly two values"""
        data = np.array([1.0, 5.0])

        # Mean is 3.0, MAD = (|1-3| + |5-3|) / 2 = 2.0
        mad = scirs2.mean_abs_deviation_py(data)
        assert abs(mad - 2.0) < 0.001

        # Range = 5.0 - 1.0 = 4.0
        range_val = scirs2.data_range_py(data)
        assert abs(range_val - 4.0) < 0.001

    def test_negative_values(self):
        """Test with negative values"""
        data = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])

        # Should work with negative values
        mad = scirs2.mean_abs_deviation_py(data)
        assert mad >= 0.0  # MAD is always non-negative

        median_mad = scirs2.median_abs_deviation_py(data)
        assert median_mad >= 0.0

        # Range = 5.0 - (-5.0) = 10.0
        range_val = scirs2.data_range_py(data)
        assert abs(range_val - 10.0) < 0.001

    def test_extreme_inequality(self):
        """Test Gini coefficient with extreme inequality"""
        # One person has everything
        data = np.array([0.0, 0.0, 0.0, 0.0, 100.0])

        gini = scirs2.gini_coefficient_py(data)
        # Should be close to 1 (maximum inequality)
        assert gini >= 0.8


class TestDispersionNumericalProperties:
    """Numerical stability and accuracy tests"""

    def test_large_values(self):
        """Test with large values"""
        data = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

        # MAD should scale proportionally
        mad = scirs2.mean_abs_deviation_py(data)
        assert mad > 0.0

        # Range = 5000 - 1000 = 4000
        range_val = scirs2.data_range_py(data)
        assert abs(range_val - 4000.0) < 0.01

    def test_small_values(self):
        """Test with small values"""
        data = np.array([0.001, 0.002, 0.003, 0.004, 0.005])

        # Should handle small values accurately
        mad = scirs2.mean_abs_deviation_py(data)
        assert mad > 0.0
        assert mad < 0.01

        # Range = 0.005 - 0.001 = 0.004
        range_val = scirs2.data_range_py(data)
        assert abs(range_val - 0.004) < 0.0001

    def test_different_scales(self):
        """Test coefficient of variation is scale-invariant"""
        data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # 10x larger

        cv1 = scirs2.coef_variation_py(data1)
        cv2 = scirs2.coef_variation_py(data2)

        # CV should be the same (scale-invariant)
        assert abs(cv1 - cv2) < 0.001

    def test_many_values(self):
        """Test with large number of values"""
        np.random.seed(42)
        data = np.random.randn(1000) * 10 + 50

        # All measures should handle large datasets
        mad = scirs2.mean_abs_deviation_py(data)
        assert mad > 0.0

        median_mad = scirs2.median_abs_deviation_py(data)
        assert median_mad > 0.0

        range_val = scirs2.data_range_py(data)
        assert range_val > 0.0

        cv = scirs2.coef_variation_py(data)
        assert cv > 0.0

        gini = scirs2.gini_coefficient_py(np.abs(data))  # Gini needs non-negative
        assert 0.0 <= gini <= 1.0


class TestDispersionRealWorldScenarios:
    """Real-world application scenarios"""

    def test_income_inequality(self):
        """Test Gini coefficient for income inequality"""
        # Simulated income distribution (in thousands)
        incomes = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0, 120.0, 200.0])

        gini = scirs2.gini_coefficient_py(incomes)

        # Real-world income Gini typically 0.25-0.45
        assert 0.2 < gini < 0.6

    def test_quality_control_range(self):
        """Test range for quality control measurements"""
        # Product dimensions in mm (target: 100mm)
        measurements = np.array([99.8, 100.1, 99.9, 100.2, 100.0, 99.7, 100.3])

        range_val = scirs2.data_range_py(measurements)

        # Range should be small for quality products
        assert range_val < 1.0  # Within 1mm tolerance
        assert abs(range_val - 0.6) < 0.01  # 100.3 - 99.7 = 0.6

    def test_coefficient_variation_consistency(self):
        """Test CV for process consistency measurement"""
        # Two manufacturing processes
        process_a = np.array([100.0, 102.0, 98.0, 101.0, 99.0])  # Low variation
        process_b = np.array([100.0, 110.0, 90.0, 105.0, 95.0])  # High variation

        cv_a = scirs2.coef_variation_py(process_a)
        cv_b = scirs2.coef_variation_py(process_b)

        # Process A should be more consistent (lower CV)
        assert cv_a < cv_b

    def test_median_mad_robustness(self):
        """Test MAD vs Median MAD robustness to outliers"""
        # Data with outlier
        normal_data = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        outlier_data = np.array([10.0, 11.0, 12.0, 13.0, 100.0])  # Extreme outlier

        # MAD is affected by outlier
        mad_normal = scirs2.mean_abs_deviation_py(normal_data)
        mad_outlier = scirs2.mean_abs_deviation_py(outlier_data)

        # Median MAD is more robust
        median_mad_normal = scirs2.median_abs_deviation_py(normal_data)
        median_mad_outlier = scirs2.median_abs_deviation_py(outlier_data)

        # MAD should change more than Median MAD
        mad_change = abs(mad_outlier - mad_normal) / mad_normal
        median_mad_change = abs(median_mad_outlier - median_mad_normal) / median_mad_normal

        # Median MAD should be more stable (smaller relative change)
        assert median_mad_change < mad_change

    def test_temperature_variability(self):
        """Test dispersion measures for temperature data"""
        # Daily temperatures (Celsius) over a week
        temps = np.array([18.5, 20.2, 19.8, 21.5, 19.0, 22.0, 20.5])

        # Range shows temperature swing
        range_val = scirs2.data_range_py(temps)
        assert 3.0 < range_val < 4.0  # Expected: ~3.5°C swing

        # MAD shows average deviation
        mad = scirs2.mean_abs_deviation_py(temps)
        assert mad > 0.5  # Some day-to-day variation

        # CV should be low (temperatures don't vary that much proportionally)
        cv = scirs2.coef_variation_py(temps)
        assert cv < 0.1  # Less than 10% variation

    def test_stock_price_volatility(self):
        """Test CV for stock price volatility"""
        # Stock prices over time
        stable_stock = np.array([100.0, 101.0, 99.0, 100.5, 100.2])
        volatile_stock = np.array([100.0, 115.0, 95.0, 110.0, 90.0])

        cv_stable = scirs2.coef_variation_py(stable_stock)
        cv_volatile = scirs2.coef_variation_py(volatile_stock)

        # Volatile stock should have much higher CV
        assert cv_volatile > 2 * cv_stable

    def test_wealth_distribution(self):
        """Test Gini coefficient for wealth distribution scenarios"""
        # Perfect equality: everyone has same wealth
        equal_wealth = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        gini_equal = scirs2.gini_coefficient_py(equal_wealth)
        assert abs(gini_equal - 0.0) < 0.001

        # Moderate inequality
        moderate_wealth = np.array([50.0, 75.0, 100.0, 125.0, 150.0])
        gini_moderate = scirs2.gini_coefficient_py(moderate_wealth)
        assert 0.1 < gini_moderate < 0.4

        # High inequality
        high_wealth = np.array([10.0, 20.0, 30.0, 40.0, 900.0])
        gini_high = scirs2.gini_coefficient_py(high_wealth)
        assert gini_high > 0.6


class TestDispersionComparisons:
    """Test relationships between different dispersion measures"""

    def test_mad_vs_median_mad(self):
        """Test relationship between MAD and Median MAD"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        mad = scirs2.mean_abs_deviation_py(data)
        median_mad = scirs2.median_abs_deviation_py(data)

        # Both should be positive
        assert mad > 0.0
        assert median_mad > 0.0

        # For symmetric distributions, they should be similar
        assert abs(mad - median_mad) / mad < 0.5

    def test_range_bounds_other_measures(self):
        """Test that range provides upper bound for other measures"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        range_val = scirs2.data_range_py(data)
        mad = scirs2.mean_abs_deviation_py(data)
        median_mad = scirs2.median_abs_deviation_py(data)

        # MAD and Median MAD should be less than range
        assert mad < range_val
        assert median_mad < range_val

    def test_gini_consistency(self):
        """Test Gini coefficient bounds and consistency"""
        # Multiple distributions with known properties
        equal = np.array([1.0, 1.0, 1.0, 1.0])
        slightly_unequal = np.array([1.0, 2.0, 3.0, 4.0])
        very_unequal = np.array([1.0, 1.0, 1.0, 97.0])

        gini_equal = scirs2.gini_coefficient_py(equal)
        gini_slight = scirs2.gini_coefficient_py(slightly_unequal)
        gini_very = scirs2.gini_coefficient_py(very_unequal)

        # Should be ordered
        assert gini_equal < gini_slight < gini_very

        # All should be in [0, 1]
        assert 0.0 <= gini_equal <= 1.0
        assert 0.0 <= gini_slight <= 1.0
        assert 0.0 <= gini_very <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
