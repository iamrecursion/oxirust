"""
Tests for correlation coefficient tests with significance.

This module tests the Pearson, Spearman, and Kendall tau correlation tests
which compute correlation coefficients and test for statistical significance.
"""

import pytest
import numpy as np
import scirs2


class TestPearsonCorrelation:
    """Tests for Pearson correlation coefficient with p-value."""

    def test_pearson_perfect_positive(self):
        """Test Pearson correlation with perfect positive correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = scirs2.pearsonr_py(x, y)

        assert "correlation" in result
        assert "pvalue" in result
        assert abs(result["correlation"] - 1.0) < 1e-10  # Perfect correlation
        assert result["pvalue"] < 0.01  # Highly significant

    def test_pearson_perfect_negative(self):
        """Test Pearson correlation with perfect negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        result = scirs2.pearsonr_py(x, y)

        assert abs(result["correlation"] - (-1.0)) < 1e-10
        assert result["pvalue"] < 0.01

    def test_pearson_no_correlation(self):
        """Test Pearson correlation with no correlation."""
        # Create uncorrelated data
        np.random.seed(42)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([5.2, 5.1, 4.9, 5.0, 5.1, 4.8, 5.2, 4.9, 5.0, 5.1])

        result = scirs2.pearsonr_py(x, y)

        # Correlation should be close to 0
        assert abs(result["correlation"]) < 0.5
        # P-value should be high (not significant)
        assert result["pvalue"] > 0.05

    def test_pearson_moderate_correlation(self):
        """Test Pearson correlation with moderate positive correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 11.9, 14.2, 15.8, 18.1, 19.9])

        result = scirs2.pearsonr_py(x, y)

        # Should have strong positive correlation
        assert result["correlation"] > 0.9
        assert result["pvalue"] < 0.01

    def test_pearson_alternative_less(self):
        """Test Pearson correlation with alternative='less'."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])  # Negative correlation

        result = scirs2.pearsonr_py(x, y, alternative="less")

        # Verify the function accepts the alternative parameter
        assert result["correlation"] < 0
        assert "pvalue" in result

    def test_pearson_alternative_greater(self):
        """Test Pearson correlation with alternative='greater'."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # Positive correlation

        result = scirs2.pearsonr_py(x, y, alternative="greater")

        # For positive correlation, greater alternative should give low p-value
        assert result["correlation"] > 0
        assert result["pvalue"] < 0.05

    def test_pearson_two_sided_default(self):
        """Test that Pearson uses two-sided by default."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result_default = scirs2.pearsonr_py(x, y)
        result_two_sided = scirs2.pearsonr_py(x, y, alternative="two-sided")

        assert result_default["correlation"] == result_two_sided["correlation"]
        assert result_default["pvalue"] == result_two_sided["pvalue"]

    def test_pearson_minimum_size(self):
        """Test Pearson correlation with minimum sample size."""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])

        result = scirs2.pearsonr_py(x, y)

        # Should work with n=2
        assert "correlation" in result
        assert "pvalue" in result


class TestSpearmanCorrelation:
    """Tests for Spearman rank correlation coefficient with p-value."""

    def test_spearman_perfect_positive(self):
        """Test Spearman correlation with perfect monotonic positive relationship."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        result = scirs2.spearmanr_py(x, y)

        assert abs(result["correlation"] - 1.0) < 1e-10
        assert result["pvalue"] < 0.01

    def test_spearman_perfect_negative(self):
        """Test Spearman correlation with perfect monotonic negative relationship."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])

        result = scirs2.spearmanr_py(x, y)

        assert abs(result["correlation"] - (-1.0)) < 1e-10
        assert result["pvalue"] < 0.01

    def test_spearman_nonlinear_monotonic(self):
        """Test Spearman with nonlinear but monotonic relationship."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = x ** 2  # Nonlinear but monotonic

        result = scirs2.spearmanr_py(x, y)

        # Spearman should detect monotonic relationship
        assert result["correlation"] > 0.99
        assert result["pvalue"] < 0.01

    def test_spearman_with_ties(self):
        """Test Spearman correlation with tied ranks."""
        x = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        y = np.array([5.0, 6.0, 6.0, 7.0, 8.0])

        result = scirs2.spearmanr_py(x, y)

        # Should handle ties correctly
        assert result["correlation"] > 0.9
        assert result["pvalue"] < 0.05

    def test_spearman_alternative_less(self):
        """Test Spearman correlation with alternative='less'."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        result = scirs2.spearmanr_py(x, y, alternative="less")

        # Verify the function accepts the alternative parameter
        assert result["correlation"] < 0
        assert "pvalue" in result

    def test_spearman_alternative_greater(self):
        """Test Spearman correlation with alternative='greater'."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = scirs2.spearmanr_py(x, y, alternative="greater")

        assert result["correlation"] > 0
        assert result["pvalue"] < 0.05

    def test_spearman_no_correlation(self):
        """Test Spearman with no monotonic relationship."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.array([4.0, 2.0, 6.0, 1.0, 7.0, 3.0, 8.0, 5.0])

        result = scirs2.spearmanr_py(x, y)

        # Weak correlation
        assert abs(result["correlation"]) < 0.6


class TestKendallTauCorrelation:
    """Tests for Kendall tau rank correlation coefficient with p-value."""

    def test_kendall_perfect_concordance(self):
        """Test Kendall tau with perfect concordance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        result = scirs2.kendalltau_py(x, y)

        assert abs(result["correlation"] - 1.0) < 1e-10
        # With small sample, p-value might be higher
        assert result["pvalue"] < 0.1

    def test_kendall_perfect_discordance(self):
        """Test Kendall tau with perfect discordance."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        result = scirs2.kendalltau_py(x, y)

        assert abs(result["correlation"] - (-1.0)) < 1e-10
        assert result["pvalue"] < 0.1

    def test_kendall_method_b(self):
        """Test Kendall tau-b (default method)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])

        result = scirs2.kendalltau_py(x, y, method="b")

        # Should have moderate positive correlation
        assert result["correlation"] > 0.5
        assert "pvalue" in result

    def test_kendall_method_c(self):
        """Test Kendall tau-c method."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])

        result = scirs2.kendalltau_py(x, y, method="c")

        assert "correlation" in result
        assert "pvalue" in result

    def test_kendall_with_ties(self):
        """Test Kendall tau with tied ranks."""
        x = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        y = np.array([5.0, 6.0, 6.0, 7.0, 8.0])

        result = scirs2.kendalltau_py(x, y, method="b")

        # tau-b accounts for ties
        assert result["correlation"] > 0.7

    def test_kendall_alternative_less(self):
        """Test Kendall tau with alternative='less'."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

        result = scirs2.kendalltau_py(x, y, method="b", alternative="less")

        assert result["correlation"] < 0
        assert result["pvalue"] < 0.05

    def test_kendall_alternative_greater(self):
        """Test Kendall tau with alternative='greater'."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        result = scirs2.kendalltau_py(x, y, method="b", alternative="greater")

        assert result["correlation"] > 0
        assert result["pvalue"] < 0.05

    def test_kendall_larger_sample(self):
        """Test Kendall tau with larger sample for better power."""
        np.random.seed(42)
        x = np.arange(20, dtype=float)
        y = x + np.random.normal(0, 2, 20)  # Add some noise

        result = scirs2.kendalltau_py(x, y)

        # Should still detect strong positive association
        assert result["correlation"] > 0.7
        assert result["pvalue"] < 0.01


class TestCorrelationComparisons:
    """Tests comparing different correlation methods."""

    def test_all_methods_same_direction(self):
        """Test that all methods agree on correlation direction."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.array([2.1, 3.9, 6.1, 8.2, 9.9, 12.1, 13.8, 16.2])

        pearson = scirs2.pearsonr_py(x, y)
        spearman = scirs2.spearmanr_py(x, y)
        kendall = scirs2.kendalltau_py(x, y)

        # All should be positive
        assert pearson["correlation"] > 0
        assert spearman["correlation"] > 0
        assert kendall["correlation"] > 0

        # All should be significant
        assert pearson["pvalue"] < 0.01
        assert spearman["pvalue"] < 0.01
        assert kendall["pvalue"] < 0.05

    def test_spearman_vs_pearson_nonlinear(self):
        """Test that Spearman handles nonlinearity better than Pearson."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.exp(x / 4.0)  # Exponential relationship

        pearson = scirs2.pearsonr_py(x, y)
        spearman = scirs2.spearmanr_py(x, y)

        # Spearman should be closer to 1 (perfect monotonic)
        assert spearman["correlation"] > pearson["correlation"]
        assert abs(spearman["correlation"] - 1.0) < 0.01

    def test_kendall_magnitude_smaller(self):
        """Test that Kendall tau typically has smaller magnitude than Spearman."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = np.array([2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5, 14.0, 15.5])

        spearman = scirs2.spearmanr_py(x, y)
        kendall = scirs2.kendalltau_py(x, y)

        # Both positive
        assert spearman["correlation"] > 0
        assert kendall["correlation"] > 0

        # Kendall typically smaller magnitude
        # (not always true, but often for this type of data)
        assert kendall["correlation"] <= spearman["correlation"] + 0.1


class TestCorrelationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_different_lengths(self):
        """Test that different array lengths raise an error."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(RuntimeError):
            scirs2.pearsonr_py(x, y)

        with pytest.raises(RuntimeError):
            scirs2.spearmanr_py(x, y)

        with pytest.raises(RuntimeError):
            scirs2.kendalltau_py(x, y)

    def test_constant_array(self):
        """Test correlation with constant (zero variance) arrays."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])

        # Should raise error (can't compute correlation with zero variance)
        with pytest.raises(RuntimeError):
            scirs2.pearsonr_py(x, y)

    def test_invalid_alternative(self):
        """Test that invalid alternative raises error."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        with pytest.raises(RuntimeError):
            scirs2.pearsonr_py(x, y, alternative="invalid")

        with pytest.raises(RuntimeError):
            scirs2.spearmanr_py(x, y, alternative="invalid")

        with pytest.raises(RuntimeError):
            scirs2.kendalltau_py(x, y, alternative="invalid")

    def test_invalid_kendall_method(self):
        """Test that invalid Kendall method raises error."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.0, 6.0, 8.0])

        with pytest.raises(RuntimeError):
            scirs2.kendalltau_py(x, y, method="invalid")

    def test_small_sample_warning(self):
        """Test correlation with very small samples."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])

        # Should work but p-values may be unreliable
        result = scirs2.pearsonr_py(x, y)
        assert "correlation" in result
        assert "pvalue" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
