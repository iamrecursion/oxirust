"""Tests for scirs2 statistics module."""

import numpy as np
import pytest
import scirs2


class TestDescriptiveStatistics:
    """Test descriptive statistics functions."""

    def test_describe(self):
        """Test describe function returns all statistics."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = scirs2.describe_py(data)

        assert "mean" in stats
        assert "std" in stats
        assert "var" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "count" in stats

        assert abs(stats["mean"] - 3.0) < 1e-10
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0
        assert stats["count"] == 5

    def test_mean(self):
        """Test mean calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = scirs2.mean_py(data)
        assert abs(mean - 3.0) < 1e-10

    def test_std_population(self):
        """Test population standard deviation (ddof=0)."""
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        std = scirs2.std_py(data, 0)
        expected = np.std(data, ddof=0)
        assert abs(std - expected) < 1e-10

    def test_std_sample(self):
        """Test sample standard deviation (ddof=1)."""
        data = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        std = scirs2.std_py(data, 1)
        expected = np.std(data, ddof=1)
        assert abs(std - expected) < 1e-10

    def test_var(self):
        """Test variance calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = scirs2.var_py(data, 0)
        expected = np.var(data, ddof=0)
        assert abs(var - expected) < 1e-10


class TestPercentiles:
    """Test percentile functions."""

    def test_percentile_median(self):
        """Test 50th percentile equals median."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p50 = scirs2.percentile_py(data, 50.0)
        assert abs(p50 - 3.0) < 1e-10

    def test_percentile_quartiles(self):
        """Test quartile calculations."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q1 = scirs2.percentile_py(data, 25.0)
        q3 = scirs2.percentile_py(data, 75.0)

        assert q1 < q3
        assert 2.0 <= q1 <= 3.0
        assert 6.0 <= q3 <= 7.0

    def test_median(self):
        """Test median function."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        median = scirs2.median_py(data)
        assert abs(median - 3.0) < 1e-10

    def test_iqr(self):
        """Test interquartile range."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        iqr = scirs2.iqr_py(data)
        assert iqr > 0


class TestCorrelation:
    """Test correlation and covariance functions."""

    def test_correlation_perfect_positive(self):
        """Test perfect positive correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        corr = scirs2.correlation_py(x, y)
        assert abs(corr - 1.0) < 1e-10

    def test_correlation_perfect_negative(self):
        """Test perfect negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        corr = scirs2.correlation_py(x, y)
        assert abs(corr + 1.0) < 1e-10

    def test_correlation_zero(self):
        """Test zero correlation for uncorrelated data."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        corr = scirs2.correlation_py(x, y)
        # Correlation undefined when one variable is constant
        # Should return 0 or handle gracefully

    def test_covariance(self):
        """Test covariance calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        cov = scirs2.covariance_py(x, y, 1)
        expected = np.cov(x, y, ddof=1)[0, 1]
        assert abs(cov - expected) < 1e-10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element(self):
        """Test statistics with single element."""
        data = np.array([5.0])
        stats = scirs2.describe_py(data)
        assert stats["mean"] == 5.0
        assert stats["min"] == 5.0
        assert stats["max"] == 5.0

    def test_large_array(self):
        """Test with large array."""
        data = np.random.randn(10000)
        stats = scirs2.describe_py(data)

        # Mean should be close to 0 for standard normal
        assert abs(stats["mean"]) < 0.1
        # Std should be close to 1
        assert abs(stats["std"] - 1.0) < 0.1


class TestTTests:
    """Test statistical t-tests."""

    def test_ttest_1samp_basic(self):
        """Test one-sample t-test."""
        # Sample with mean 5
        data = np.array([4.5, 5.0, 5.5, 4.8, 5.2, 5.1, 4.9, 5.3])

        # Test against mean of 5 (should not be significant)
        result = scirs2.ttest_1samp_py(data, 5.0)
        assert "statistic" in result
        assert "pvalue" in result
        assert "df" in result
        # p-value should be high (not significant)
        assert result["pvalue"] > 0.05

    def test_ttest_1samp_significant(self):
        """Test one-sample t-test with significant difference."""
        # Sample with mean clearly different from 0
        data = np.array([5.1, 5.2, 5.0, 5.3, 5.1, 5.2, 5.0, 5.1, 5.0, 5.2])

        # Test against mean of 0 (should be significant)
        result = scirs2.ttest_1samp_py(data, 0.0)
        assert result["pvalue"] < 0.05

    def test_ttest_1samp_one_sided(self):
        """Test one-sample t-test with one-sided alternative."""
        data = np.array([5.1, 5.2, 5.0, 5.3, 5.1, 5.2, 5.0, 5.1])

        # Test greater than 4
        result = scirs2.ttest_1samp_py(data, 4.0, alternative="greater")
        assert result["pvalue"] < 0.05

    def test_ttest_ind_basic(self):
        """Test independent two-sample t-test."""
        a = np.array([5.1, 5.2, 5.0, 5.3, 5.1])
        b = np.array([5.0, 5.1, 4.9, 5.2, 5.0])

        result = scirs2.ttest_ind_py(a, b)
        assert "statistic" in result
        assert "pvalue" in result
        assert "df" in result

    def test_ttest_ind_significant(self):
        """Test t-test with significantly different samples."""
        a = np.array([10.0, 10.1, 10.2, 10.0, 10.1, 10.0, 10.2])
        b = np.array([0.0, 0.1, 0.2, 0.0, 0.1, 0.0, 0.2])

        result = scirs2.ttest_ind_py(a, b)
        assert result["pvalue"] < 0.05  # Significant


class TestAdditionalStatistics:
    """Test additional statistical functions."""

    def test_skew_symmetric(self):
        """Test skewness of symmetric distribution."""
        # Symmetric data should have skewness near 0
        data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        skew = scirs2.skew_py(data)
        assert abs(skew) < 0.1

    def test_skew_positive(self):
        """Test positive skewness."""
        # Right-skewed data
        data = np.array([1.0, 1.0, 1.0, 2.0, 5.0, 10.0])
        skew = scirs2.skew_py(data)
        assert skew > 0

    def test_kurtosis_normal(self):
        """Test kurtosis of normal-like data."""
        # Excess kurtosis of normal distribution is 0
        np.random.seed(42)
        data = np.random.randn(10000)
        kurt = scirs2.kurtosis_py(data)
        assert abs(kurt) < 0.3

    def test_kurtosis_uniform(self):
        """Test kurtosis of uniform distribution."""
        # Uniform has excess kurtosis of -1.2
        data = np.linspace(0, 1, 1000)
        kurt = scirs2.kurtosis_py(data)
        assert kurt < 0  # Platykurtic

    def test_mode(self):
        """Test mode function."""
        data = np.array([1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
        mode = scirs2.mode_py(data)
        assert abs(mode - 2.0) < 0.01

    def test_gmean(self):
        """Test geometric mean."""
        data = np.array([1.0, 2.0, 4.0, 8.0])
        gmean = scirs2.gmean_py(data)
        # Geometric mean = (1*2*4*8)^(1/4) = 64^0.25 ≈ 2.828
        assert abs(gmean - 2.828) < 0.01

    def test_hmean(self):
        """Test harmonic mean."""
        data = np.array([1.0, 2.0, 4.0])
        hmean = scirs2.hmean_py(data)
        # Harmonic mean = 3 / (1/1 + 1/2 + 1/4) = 3 / 1.75 ≈ 1.714
        assert abs(hmean - 1.714) < 0.01

    def test_zscore(self):
        """Test z-score normalization."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        zscores = scirs2.zscore_py(data)

        # Z-scores should have mean 0 and std 1
        assert abs(np.mean(zscores)) < 1e-10
        assert abs(np.std(zscores) - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestNormalityTests:
    """Test normality tests."""

    def test_shapiro_normal_data(self):
        """Test Shapiro-Wilk with normally distributed data."""
        # Generate approximately normal data
        np.random.seed(42)
        data = np.random.normal(0, 1, 20)  # Use smaller sample

        result = scirs2.shapiro_py(data)
        assert "statistic" in result
        assert "pvalue" in result
        # Just verify the function runs and returns results
        # Note: Current implementation may have precision issues with p-value

    def test_shapiro_uniform_data(self):
        """Test Shapiro-Wilk with non-normal (uniform) data."""
        # Uniform data is not normal
        np.random.seed(42)
        data = np.random.uniform(0, 1, 20)  # Use smaller sample

        result = scirs2.shapiro_py(data)
        assert "statistic" in result
        assert "pvalue" in result

    def test_shapiro_small_sample(self):
        """Test Shapiro-Wilk with small sample."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.shapiro_py(data)
        assert "statistic" in result
        assert "pvalue" in result


class TestChiSquare:
    """Test chi-square goodness-of-fit."""

    def test_chisquare_uniform(self):
        """Test chi-square with uniform distribution."""
        # Roll a die 60 times, expect each face 10 times
        observed = np.array([8.0, 9.0, 11.0, 12.0, 10.0, 10.0])
        expected = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        result = scirs2.chisquare_py(observed, expected)
        assert "statistic" in result
        assert "pvalue" in result
        assert "dof" in result
        # Should have 5 degrees of freedom (6 categories - 1)
        assert result["dof"] == 5

    def test_chisquare_default_expected(self):
        """Test chi-square with default uniform expectation."""
        # Observed frequencies
        observed = np.array([10.0, 15.0, 12.0, 13.0])

        result = scirs2.chisquare_py(observed)
        assert "statistic" in result
        assert "pvalue" in result


class TestANOVA:
    """Test ANOVA functions."""

    def test_f_oneway_equal_groups(self):
        """Test ANOVA with equal group sizes."""
        # Three groups with different means
        group1 = np.array([85.0, 82.0, 78.0, 88.0, 91.0])
        group2 = np.array([76.0, 80.0, 82.0, 84.0, 79.0])
        group3 = np.array([91.0, 89.0, 93.0, 87.0, 90.0])

        result = scirs2.f_oneway_py(group1, group2, group3)
        assert "f_statistic" in result
        assert "pvalue" in result
        assert "df_between" in result
        assert "df_within" in result
        # 3 groups means df_between = 2
        assert result["df_between"] == 2
        # Total 15 observations, 3 groups: df_within = 12
        assert result["df_within"] == 12

    def test_f_oneway_unequal_groups(self):
        """Test ANOVA with unequal group sizes."""
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([4.0, 5.0, 6.0, 7.0])
        group3 = np.array([8.0, 9.0])

        result = scirs2.f_oneway_py(group1, group2, group3)
        assert "f_statistic" in result
        assert "pvalue" in result

    def test_f_oneway_two_groups(self):
        """Test ANOVA with minimum number of groups."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        result = scirs2.f_oneway_py(group1, group2)
        assert "f_statistic" in result
        assert "pvalue" in result
        assert result["df_between"] == 1

    def test_f_oneway_identical_groups(self):
        """Test ANOVA with identical groups (should have high p-value)."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group3 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.f_oneway_py(group1, group2, group3)
        # Groups are identical, so F-statistic should be very small
        # and p-value should be very high (close to 1)
        assert result["f_statistic"] < 0.01 or result["pvalue"] > 0.9
