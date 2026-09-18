"""
Comprehensive tests for quantile and robust statistics
"""

import numpy as np
import pytest
import scirs2


class TestBoxplotStats:
    """Tests for boxplot statistics (five-number summary)"""

    def test_boxplot_stats_basic(self):
        """Test boxplot stats with simple data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = scirs2.boxplot_stats_py(data)

        # Check all required fields are present
        assert "q1" in result
        assert "median" in result
        assert "q3" in result
        assert "whislo" in result
        assert "whishi" in result
        assert "outliers" in result

        # Check values are reasonable
        assert result["q1"] < result["median"] < result["q3"]
        assert result["whislo"] <= result["q1"]
        assert result["q3"] <= result["whishi"]

    def test_boxplot_stats_with_outliers(self):
        """Test boxplot stats detects outliers"""
        # Data with clear outlier
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        result = scirs2.boxplot_stats_py(data)

        # Should detect 100.0 as outlier
        assert len(result["outliers"]) > 0
        assert 100.0 in result["outliers"]

    def test_boxplot_stats_no_outliers(self):
        """Test boxplot stats with no outliers"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.boxplot_stats_py(data)

        # Should have no outliers
        assert len(result["outliers"]) == 0

    def test_boxplot_stats_custom_whis(self):
        """Test boxplot stats with custom whisker parameter"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0])

        # With larger whis, fewer outliers
        result_large = scirs2.boxplot_stats_py(data, whis=3.0)
        result_small = scirs2.boxplot_stats_py(data, whis=1.0)

        # Larger whis should have fewer or equal outliers
        assert len(result_large["outliers"]) <= len(result_small["outliers"])

    def test_boxplot_stats_identical_values(self):
        """Test boxplot stats with identical values"""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        result = scirs2.boxplot_stats_py(data)

        # All quartiles should be the same
        assert abs(result["q1"] - 5.0) < 0.01
        assert abs(result["median"] - 5.0) < 0.01
        assert abs(result["q3"] - 5.0) < 0.01
        assert len(result["outliers"]) == 0

    def test_boxplot_stats_negative_values(self):
        """Test boxplot stats with negative values"""
        data = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])

        result = scirs2.boxplot_stats_py(data)

        # Should work with negative values
        assert result["median"] == 0.0
        assert result["q1"] < 0.0
        assert result["q3"] > 0.0


class TestQuartiles:
    """Tests for quartiles calculation"""

    def test_quartiles_basic(self):
        """Test quartiles with simple data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = scirs2.quartiles_py(data)

        # Should return array of length 3 [Q1, Q2, Q3]
        assert len(result) == 3
        # Q1 < Q2 < Q3
        assert result[0] < result[1] < result[2]

    def test_quartiles_median(self):
        """Test that Q2 (median) is correct"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.quartiles_py(data)

        # Q2 should be the median (3.0)
        assert abs(result[1] - 3.0) < 0.01

    def test_quartiles_consistency_with_boxplot(self):
        """Test quartiles match boxplot_stats"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        quartiles = scirs2.quartiles_py(data)
        boxplot = scirs2.boxplot_stats_py(data)

        # Quartiles should match boxplot stats
        assert abs(quartiles[0] - boxplot["q1"]) < 0.01
        assert abs(quartiles[1] - boxplot["median"]) < 0.01
        assert abs(quartiles[2] - boxplot["q3"]) < 0.01

    def test_quartiles_odd_length(self):
        """Test quartiles with odd-length array"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

        result = scirs2.quartiles_py(data)

        assert len(result) == 3
        assert result[0] < result[1] < result[2]

    def test_quartiles_even_length(self):
        """Test quartiles with even-length array"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        result = scirs2.quartiles_py(data)

        assert len(result) == 3
        assert result[0] < result[1] < result[2]


class TestWinsorizedMean:
    """Tests for winsorized mean (robust mean)"""

    def test_winsorized_mean_basic(self):
        """Test winsorized mean with simple data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.winsorized_mean_py(data, limits=0.1)

        # Should be close to regular mean for clean data
        assert 2.5 < result < 3.5

    def test_winsorized_mean_with_outlier(self):
        """Test winsorized mean is robust to outliers"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        regular_mean = np.mean(data)
        winsorized_mean = scirs2.winsorized_mean_py(data, limits=0.2)

        # Winsorized mean should be much closer to median
        # than regular mean due to outlier
        median = np.median(data)
        assert abs(winsorized_mean - median) < abs(regular_mean - median)

    def test_winsorized_mean_zero_limits(self):
        """Test winsorized mean with zero limits equals regular mean"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        winsorized = scirs2.winsorized_mean_py(data, limits=0.0)
        regular = np.mean(data)

        # With limits=0, should equal regular mean
        assert abs(winsorized - regular) < 0.01

    def test_winsorized_mean_various_limits(self):
        """Test winsorized mean with various limit values"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0])

        # Higher limits should give more robust estimates
        result_10 = scirs2.winsorized_mean_py(data, limits=0.1)
        result_20 = scirs2.winsorized_mean_py(data, limits=0.2)

        # Both should be less affected by outlier than regular mean
        regular = np.mean(data)
        assert result_10 < regular
        assert result_20 < regular

    def test_winsorized_mean_symmetric_data(self):
        """Test winsorized mean with symmetric data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        result = scirs2.winsorized_mean_py(data, limits=0.1)

        # For symmetric data, should be close to median
        median = np.median(data)
        assert abs(result - median) < 0.5


class TestWinsorizedVariance:
    """Tests for winsorized variance (robust variance)"""

    def test_winsorized_variance_basic(self):
        """Test winsorized variance with simple data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.winsorized_variance_py(data, limits=0.1, ddof=1)

        # Should be positive
        assert result > 0.0

    def test_winsorized_variance_with_outlier(self):
        """Test winsorized variance is robust to outliers"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

        regular_var = np.var(data, ddof=1)
        winsorized_var = scirs2.winsorized_variance_py(data, limits=0.2, ddof=1)

        # Winsorized variance should be much smaller
        assert winsorized_var < regular_var / 2

    def test_winsorized_variance_zero_limits(self):
        """Test winsorized variance with zero limits"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        winsorized = scirs2.winsorized_variance_py(data, limits=0.0, ddof=1)
        regular = np.var(data, ddof=1)

        # With limits=0, should be close to regular variance
        assert abs(winsorized - regular) < 2.0

    def test_winsorized_variance_ddof_effect(self):
        """Test ddof parameter effect on winsorized variance"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        var_ddof0 = scirs2.winsorized_variance_py(data, limits=0.1, ddof=0)
        var_ddof1 = scirs2.winsorized_variance_py(data, limits=0.1, ddof=1)

        # Higher ddof should give slightly larger variance
        assert var_ddof1 > var_ddof0

    def test_winsorized_variance_identical_values(self):
        """Test winsorized variance with identical values"""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        result = scirs2.winsorized_variance_py(data, limits=0.1, ddof=1)

        # Variance should be zero (or very close to zero)
        assert abs(result) < 0.01


class TestRobustStatisticsComparison:
    """Tests comparing robust vs non-robust statistics"""

    def test_robust_vs_regular_with_outliers(self):
        """Compare robust statistics with regular statistics on outlier data"""
        # Create data with outliers
        clean_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        outlier_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])

        # Regular statistics are affected by outliers
        regular_mean_clean = np.mean(clean_data)
        regular_mean_outlier = np.mean(outlier_data)
        mean_diff = abs(regular_mean_outlier - regular_mean_clean)

        # Winsorized statistics are robust
        winsorized_mean_clean = scirs2.winsorized_mean_py(clean_data, limits=0.1)
        winsorized_mean_outlier = scirs2.winsorized_mean_py(outlier_data, limits=0.1)
        winsorized_diff = abs(winsorized_mean_outlier - winsorized_mean_clean)

        # Winsorized should be much less affected
        assert winsorized_diff < mean_diff / 2

    def test_boxplot_outlier_detection(self):
        """Test boxplot correctly identifies outliers"""
        # Data with known outliers (more extreme values)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])

        result = scirs2.boxplot_stats_py(data, whis=1.5)

        # Should detect at least one outlier
        assert len(result["outliers"]) > 0

        # 100.0 should be detected as outlier
        assert 100.0 in result["outliers"]


class TestQuantileEdgeCases:
    """Edge cases for quantile and robust statistics"""

    def test_small_dataset(self):
        """Test with very small dataset"""
        data = np.array([1.0, 2.0, 3.0])

        quartiles = scirs2.quartiles_py(data)
        assert len(quartiles) == 3

        boxplot = scirs2.boxplot_stats_py(data)
        assert "median" in boxplot

    def test_large_dataset(self):
        """Test with large dataset"""
        np.random.seed(42)
        data = np.random.randn(1000) * 10 + 50

        quartiles = scirs2.quartiles_py(data)
        assert len(quartiles) == 3

        winsorized_mean = scirs2.winsorized_mean_py(data, limits=0.05)
        assert winsorized_mean > 0.0

    def test_negative_and_positive_values(self):
        """Test with mix of negative and positive values"""
        data = np.array([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])

        quartiles = scirs2.quartiles_py(data)
        assert quartiles[0] < 0.0
        assert quartiles[2] > 0.0

        winsorized_mean = scirs2.winsorized_mean_py(data, limits=0.1)
        # Should be close to 0 for symmetric data
        assert abs(winsorized_mean) < 2.0


class TestRealWorldScenarios:
    """Real-world application scenarios"""

    def test_salary_data_robustness(self):
        """Test robust statistics on salary data with extreme values"""
        # Simulated salary data (in thousands)
        salaries = np.array([30.0, 35.0, 40.0, 42.0, 45.0, 48.0, 50.0, 52.0, 55.0, 60.0, 500.0])  # CEO outlier

        regular_mean = np.mean(salaries)
        winsorized_mean = scirs2.winsorized_mean_py(salaries, limits=0.1)

        # Winsorized mean should be lower than regular mean (less affected by CEO outlier)
        assert winsorized_mean < regular_mean

    def test_sensor_data_outlier_detection(self):
        """Test outlier detection in sensor readings"""
        # Simulated sensor data with measurement error
        readings = np.array([20.1, 20.2, 20.0, 19.9, 20.3, 50.0, 20.1, 20.2])  # 50.0 is error

        boxplot = scirs2.boxplot_stats_py(readings)

        # Should detect the erroneous reading
        assert 50.0 in boxplot["outliers"]

    def test_grading_with_outliers(self):
        """Test grading scenario where dropping extremes is common"""
        # Test scores (out of 100)
        scores = np.array([85.0, 88.0, 90.0, 92.0, 91.0, 87.0, 95.0, 20.0, 93.0, 89.0])  # One very low score

        regular_mean = np.mean(scores)
        winsorized_mean = scirs2.winsorized_mean_py(scores, limits=0.1)

        # Winsorized mean should be higher (less affected by low score)
        assert winsorized_mean > regular_mean

    def test_financial_returns_robustness(self):
        """Test robust statistics on financial returns"""
        # Daily returns (%) - includes market crash day
        returns = np.array([0.5, 0.3, -0.2, 0.4, 0.1, -0.3, 0.6, -15.0, 0.2, 0.4])

        regular_var = np.var(returns, ddof=1)
        winsorized_var = scirs2.winsorized_variance_py(returns, limits=0.1, ddof=1)

        # Winsorized variance should be much smaller (less affected by crash)
        assert winsorized_var < regular_var / 2

    def test_quartile_ranges_for_data_quality(self):
        """Test using quartiles for data quality assessment"""
        # Manufacturing measurements
        measurements = np.array([99.8, 99.9, 100.0, 100.1, 100.2, 99.7, 100.3, 100.0, 99.9])

        quartiles = scirs2.quartiles_py(measurements)
        iqr = quartiles[2] - quartiles[0]  # Interquartile range

        # IQR should be small for quality manufacturing
        assert iqr < 1.0  # Within 1 unit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
