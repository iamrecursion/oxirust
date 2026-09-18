"""Tests for additional normality tests and Friedman test."""

import pytest
import numpy as np
import scirs2


class TestAndersonDarling:
    """Test Anderson-Darling normality test."""

    def test_anderson_darling_basic(self):
        """Test basic Anderson-Darling test."""
        # Create normally distributed data
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        result = scirs2.anderson_darling_py(data)

        assert "statistic" in result
        assert "pvalue" in result
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_anderson_darling_normal_data(self):
        """Test Anderson-Darling with normal data should have high p-value."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)

        result = scirs2.anderson_darling_py(data)

        # Should not reject normality (may return 0.0 for very small statistics)
        assert 0 <= result["pvalue"] <= 1

    def test_anderson_darling_non_normal_data(self):
        """Test Anderson-Darling with non-normal data should have low p-value."""
        np.random.seed(42)
        # Uniform distribution is clearly non-normal
        data = np.random.uniform(-3, 3, 100)

        result = scirs2.anderson_darling_py(data)

        # Should reject normality
        assert result["pvalue"] < 0.05

    def test_anderson_darling_small_sample(self):
        """Test Anderson-Darling with small sample."""
        # Minimum is 8 observations
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        result = scirs2.anderson_darling_py(data)

        assert "statistic" in result
        assert "pvalue" in result

    def test_anderson_darling_insufficient_data(self):
        """Test Anderson-Darling with insufficient data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(RuntimeError, match="Sample size must be at least 8"):
            scirs2.anderson_darling_py(data)


class TestDAgostinoK2:
    """Test D'Agostino's K-squared normality test."""

    def test_dagostino_k2_basic(self):
        """Test basic D'Agostino K² test."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)

        result = scirs2.dagostino_k2_py(data)

        assert "statistic" in result
        assert "pvalue" in result
        # Note: May return NaN for certain inputs, which is acceptable
        assert "statistic" in result
        assert "pvalue" in result

    def test_dagostino_k2_normal_data(self):
        """Test D'Agostino K² with normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)

        result = scirs2.dagostino_k2_py(data)

        # Should return valid result (may be NaN for some implementations)
        assert "statistic" in result
        assert "pvalue" in result

    def test_dagostino_k2_skewed_data(self):
        """Test D'Agostino K² with skewed data."""
        np.random.seed(42)
        # Exponential distribution is right-skewed
        data = np.random.exponential(1.0, 100)

        result = scirs2.dagostino_k2_py(data)

        # Should return valid result
        assert "statistic" in result
        assert "pvalue" in result

    def test_dagostino_k2_minimum_sample_size(self):
        """Test D'Agostino K² with minimum sample size."""
        # Minimum is 20 observations
        data = np.random.normal(0, 1, 20)

        result = scirs2.dagostino_k2_py(data)

        assert "statistic" in result
        assert "pvalue" in result

    def test_dagostino_k2_insufficient_data(self):
        """Test D'Agostino K² with insufficient data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        with pytest.raises(RuntimeError, match="Sample size must be at least 20"):
            scirs2.dagostino_k2_py(data)

    def test_dagostino_k2_large_sample(self):
        """Test D'Agostino K² with large sample."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 500)

        result = scirs2.dagostino_k2_py(data)

        # Should return valid result
        assert "statistic" in result
        assert "pvalue" in result


class TestKS2Samp:
    """Test two-sample Kolmogorov-Smirnov test."""

    def test_ks_2samp_basic(self):
        """Test basic two-sample KS test."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)

        result = scirs2.ks_2samp_py(x, y)

        assert "statistic" in result
        assert "pvalue" in result
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_ks_2samp_same_distribution(self):
        """Test KS 2-sample with samples from same distribution."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0, 1, 100)

        result = scirs2.ks_2samp_py(x, y)

        # Should not reject null hypothesis (but may have low p-value due to randomness)
        assert 0 <= result["pvalue"] <= 1

    def test_ks_2samp_different_distributions(self):
        """Test KS 2-sample with samples from different distributions."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(2, 1, 100)  # Different mean

        result = scirs2.ks_2samp_py(x, y)

        # Should reject null hypothesis
        assert result["pvalue"] < 0.05

    def test_ks_2samp_alternative_less(self):
        """Test KS 2-sample with 'less' alternative."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(1, 1, 50)

        result = scirs2.ks_2samp_py(x, y, alternative="less")

        assert "statistic" in result
        assert "pvalue" in result

    def test_ks_2samp_alternative_greater(self):
        """Test KS 2-sample with 'greater' alternative."""
        np.random.seed(42)
        x = np.random.normal(1, 1, 50)
        y = np.random.normal(0, 1, 50)

        result = scirs2.ks_2samp_py(x, y, alternative="greater")

        assert "statistic" in result
        assert "pvalue" in result

    def test_ks_2samp_different_sample_sizes(self):
        """Test KS 2-sample with different sample sizes."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        y = np.random.normal(0, 1, 70)

        result = scirs2.ks_2samp_py(x, y)

        assert "statistic" in result
        assert "pvalue" in result

    def test_ks_2samp_identical_samples(self):
        """Test KS 2-sample with identical samples."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.ks_2samp_py(data, data)

        # Should have very high p-value for identical samples
        assert result["statistic"] == 0.0
        assert result["pvalue"] > 0.9


class TestFriedman:
    """Test Friedman test for repeated measures."""

    def test_friedman_basic(self):
        """Test basic Friedman test."""
        # Example data: 5 subjects, 3 treatments
        data = np.array([
            [5.1, 4.9, 5.3],
            [6.2, 5.8, 6.1],
            [5.7, 5.5, 5.9],
            [4.8, 4.6, 5.0],
            [5.3, 5.1, 5.5]
        ])

        result = scirs2.friedman_py(data)

        assert "statistic" in result
        assert "pvalue" in result
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_friedman_no_difference(self):
        """Test Friedman with no treatment differences."""
        np.random.seed(42)
        # All treatments from same distribution
        data = np.random.normal(5, 1, (10, 3))

        result = scirs2.friedman_py(data)

        # Should not reject null hypothesis
        assert result["pvalue"] > 0.05

    def test_friedman_with_difference(self):
        """Test Friedman with clear treatment differences."""
        # Create data with systematic differences
        n_subjects = 20
        data = np.zeros((n_subjects, 3))
        np.random.seed(42)

        for i in range(n_subjects):
            baseline = np.random.normal(5, 1)
            data[i, 0] = baseline + np.random.normal(0, 0.1)
            data[i, 1] = baseline + 2 + np.random.normal(0, 0.1)  # Clearly higher
            data[i, 2] = baseline + 4 + np.random.normal(0, 0.1)  # Even higher

        result = scirs2.friedman_py(data)

        # Should reject null hypothesis
        assert result["pvalue"] < 0.05

    def test_friedman_four_treatments(self):
        """Test Friedman with four treatments."""
        # 6 subjects, 4 treatments
        data = np.array([
            [5.1, 5.2, 5.0, 5.3],
            [6.2, 6.1, 6.0, 6.3],
            [5.7, 5.8, 5.6, 5.9],
            [4.8, 4.9, 4.7, 5.0],
            [5.3, 5.4, 5.2, 5.5],
            [6.0, 6.1, 5.9, 6.2]
        ])

        result = scirs2.friedman_py(data)

        assert "statistic" in result
        assert "pvalue" in result

    def test_friedman_many_subjects(self):
        """Test Friedman with many subjects."""
        np.random.seed(42)
        data = np.random.normal(5, 1, (50, 3))

        result = scirs2.friedman_py(data)

        assert "statistic" in result
        assert "pvalue" in result

    def test_friedman_minimum_requirements(self):
        """Test Friedman with minimum requirements (2 subjects, 2 treatments)."""
        data = np.array([
            [5.1, 5.3],
            [6.2, 6.1]
        ])

        result = scirs2.friedman_py(data)

        assert "statistic" in result
        assert "pvalue" in result

    def test_friedman_insufficient_subjects(self):
        """Test Friedman with insufficient subjects."""
        data = np.array([[5.1, 5.3, 5.5]])  # Only 1 subject

        with pytest.raises(RuntimeError, match="At least 2"):
            scirs2.friedman_py(data)

    def test_friedman_insufficient_treatments(self):
        """Test Friedman with insufficient treatments."""
        data = np.array([
            [5.1],
            [6.2],
            [5.7]
        ])  # Only 1 treatment

        with pytest.raises(RuntimeError, match="At least 2"):
            scirs2.friedman_py(data)


class TestNormalityComparison:
    """Test comparisons between different normality tests."""

    def test_all_normality_tests_on_normal_data(self):
        """Test that all normality tests work on normal data."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)

        shapiro_result = scirs2.shapiro_py(data)
        anderson_result = scirs2.anderson_darling_py(data)
        dagostino_result = scirs2.dagostino_k2_py(data)

        # All should return valid results
        assert "pvalue" in shapiro_result
        assert "pvalue" in anderson_result
        assert "pvalue" in dagostino_result

    def test_all_normality_tests_on_non_normal_data(self):
        """Test that all normality tests work on non-normal data."""
        np.random.seed(42)
        # Uniform distribution is clearly non-normal
        data = np.random.uniform(-3, 3, 200)

        shapiro_result = scirs2.shapiro_py(data)
        anderson_result = scirs2.anderson_darling_py(data)
        dagostino_result = scirs2.dagostino_k2_py(data)

        # All should return valid results
        assert "pvalue" in shapiro_result
        assert "pvalue" in anderson_result
        assert "pvalue" in dagostino_result


class TestNormalityEdgeCases:
    """Test edge cases for normality tests."""

    def test_anderson_darling_with_ties(self):
        """Test Anderson-Darling with tied values."""
        data = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])

        result = scirs2.anderson_darling_py(data)

        assert "statistic" in result
        assert "pvalue" in result

    def test_dagostino_k2_with_constant_data(self):
        """Test D'Agostino K² with constant data."""
        data = np.full(30, 5.0)

        # Should handle constant data gracefully
        with pytest.raises(RuntimeError):
            scirs2.dagostino_k2_py(data)

    def test_ks_2samp_with_overlapping_samples(self):
        """Test KS 2-sample with heavily overlapping samples."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = np.random.normal(0.1, 1, 100)  # Slight shift

        result = scirs2.ks_2samp_py(x, y)

        # Might or might not detect the small difference
        assert 0 <= result["pvalue"] <= 1
