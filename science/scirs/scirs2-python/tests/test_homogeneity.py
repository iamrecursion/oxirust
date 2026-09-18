"""Tests for homogeneity of variance tests."""

import pytest
import numpy as np
import scirs2


class TestLeveneTest:
    """Test Levene's test for homogeneity of variance."""

    def test_levene_basic_two_groups(self):
        """Test basic Levene's test with two groups."""
        g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])

        result = scirs2.levene_py(g1, g2)

        assert "statistic" in result
        assert "pvalue" in result
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_levene_three_groups(self):
        """Test Levene's test with three groups."""
        g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])
        g3 = np.array([8.95, 9.12, 8.95, 8.85, 9.03])

        result = scirs2.levene_py(g1, g2, g3)

        assert "statistic" in result
        assert "pvalue" in result

    def test_levene_center_mean(self):
        """Test Levene's test using mean as center."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        result = scirs2.levene_py(g1, g2, center="mean")

        assert "statistic" in result
        assert "pvalue" in result

    def test_levene_center_median(self):
        """Test Levene's test using median as center (default)."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        result = scirs2.levene_py(g1, g2, center="median")

        assert "statistic" in result
        assert "pvalue" in result

    def test_levene_center_trimmed(self):
        """Test Levene's test using trimmed mean as center."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])

        result = scirs2.levene_py(g1, g2, center="trimmed", proportion_to_cut=0.1)

        assert "statistic" in result
        assert "pvalue" in result

    def test_levene_equal_variances(self):
        """Test Levene's test with groups having equal variances."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(1, 1, 50)  # Same variance, different mean

        result = scirs2.levene_py(g1, g2)

        # Should have high p-value for equal variances
        assert result["pvalue"] > 0.05

    def test_levene_different_variances(self):
        """Test Levene's test with groups having different variances."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(0, 5, 50)  # Much larger variance

        result = scirs2.levene_py(g1, g2)

        # Should have low p-value for different variances
        assert result["pvalue"] < 0.05

    def test_levene_four_groups(self):
        """Test Levene's test with four groups."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0])
        g3 = np.array([3.0, 4.0, 5.0, 6.0])
        g4 = np.array([4.0, 5.0, 6.0, 7.0])

        result = scirs2.levene_py(g1, g2, g3, g4)

        assert "statistic" in result
        assert "pvalue" in result


class TestBartlettTest:
    """Test Bartlett's test for homogeneity of variance."""

    def test_bartlett_basic_two_groups(self):
        """Test basic Bartlett's test with two groups."""
        g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])

        result = scirs2.bartlett_test_py(g1, g2)

        assert "statistic" in result
        assert "pvalue" in result
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_bartlett_three_groups(self):
        """Test Bartlett's test with three groups."""
        g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])
        g3 = np.array([8.95, 9.12, 8.95, 8.85, 9.03])

        result = scirs2.bartlett_test_py(g1, g2, g3)

        assert "statistic" in result
        assert "pvalue" in result

    def test_bartlett_equal_variances(self):
        """Test Bartlett's test with groups having equal variances."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(1, 1, 50)  # Same variance, different mean

        result = scirs2.bartlett_test_py(g1, g2)

        # Should have high p-value for equal variances
        assert result["pvalue"] > 0.05

    def test_bartlett_different_variances(self):
        """Test Bartlett's test with groups having different variances."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(0, 5, 50)  # Much larger variance

        result = scirs2.bartlett_test_py(g1, g2)

        # Should have low p-value for different variances
        assert result["pvalue"] < 0.05

    def test_bartlett_multiple_groups(self):
        """Test Bartlett's test with multiple groups."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 30)
        g2 = np.random.normal(0, 1, 30)
        g3 = np.random.normal(0, 1, 30)
        g4 = np.random.normal(0, 1, 30)

        result = scirs2.bartlett_test_py(g1, g2, g3, g4)

        assert "statistic" in result
        assert "pvalue" in result
        # Equal variances should give high p-value
        assert result["pvalue"] > 0.05

    def test_bartlett_different_sample_sizes(self):
        """Test Bartlett's test with different sample sizes."""
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        g3 = np.array([3.0, 4.0])

        result = scirs2.bartlett_test_py(g1, g2, g3)

        assert "statistic" in result
        assert "pvalue" in result


class TestBrownForsytheTest:
    """Test Brown-Forsythe test for homogeneity of variance."""

    def test_brown_forsythe_basic_two_groups(self):
        """Test basic Brown-Forsythe test with two groups."""
        g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])

        result = scirs2.brown_forsythe_py(g1, g2)

        assert "statistic" in result
        assert "pvalue" in result
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_brown_forsythe_three_groups(self):
        """Test Brown-Forsythe test with three groups."""
        g1 = np.array([8.88, 9.12, 9.04, 8.98, 9.00])
        g2 = np.array([8.88, 8.95, 9.29, 9.44, 9.15])
        g3 = np.array([8.95, 9.12, 8.95, 8.85, 9.03])

        result = scirs2.brown_forsythe_py(g1, g2, g3)

        assert "statistic" in result
        assert "pvalue" in result

    def test_brown_forsythe_equal_variances(self):
        """Test Brown-Forsythe test with groups having equal variances."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(1, 1, 50)  # Same variance, different mean

        result = scirs2.brown_forsythe_py(g1, g2)

        # Should have high p-value for equal variances
        assert result["pvalue"] > 0.05

    def test_brown_forsythe_different_variances(self):
        """Test Brown-Forsythe test with groups having different variances."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 50)
        g2 = np.random.normal(0, 5, 50)  # Much larger variance

        result = scirs2.brown_forsythe_py(g1, g2)

        # Should have low p-value for different variances
        assert result["pvalue"] < 0.05

    def test_brown_forsythe_robust_to_outliers(self):
        """Test that Brown-Forsythe is robust to outliers (uses median)."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Outlier

        result = scirs2.brown_forsythe_py(g1, g2)

        assert "statistic" in result
        assert "pvalue" in result


class TestHomogeneityEdgeCases:
    """Test edge cases and error conditions."""

    def test_levene_insufficient_groups(self):
        """Test that Levene's test requires at least 2 groups."""
        g1 = np.array([1.0, 2.0, 3.0])

        with pytest.raises(RuntimeError, match="Need at least 2 groups"):
            scirs2.levene_py(g1)

    def test_bartlett_insufficient_groups(self):
        """Test that Bartlett's test requires at least 2 groups."""
        g1 = np.array([1.0, 2.0, 3.0])

        with pytest.raises(RuntimeError, match="Need at least 2 groups"):
            scirs2.bartlett_test_py(g1)

    def test_brown_forsythe_insufficient_groups(self):
        """Test that Brown-Forsythe test requires at least 2 groups."""
        g1 = np.array([1.0, 2.0, 3.0])

        with pytest.raises(RuntimeError, match="Need at least 2 groups"):
            scirs2.brown_forsythe_py(g1)

    def test_levene_with_small_samples(self):
        """Test Levene's test with small sample sizes."""
        g1 = np.array([1.0, 2.0])
        g2 = np.array([3.0, 4.0])

        result = scirs2.levene_py(g1, g2)

        assert "statistic" in result
        assert "pvalue" in result

    def test_bartlett_with_small_samples(self):
        """Test Bartlett's test with small sample sizes."""
        g1 = np.array([1.0, 2.0])
        g2 = np.array([3.0, 4.0])

        result = scirs2.bartlett_test_py(g1, g2)

        assert "statistic" in result
        assert "pvalue" in result


class TestHomogeneityConsistency:
    """Test consistency across homogeneity tests."""

    def test_brown_forsythe_equals_levene_median(self):
        """Test that Brown-Forsythe equals Levene with median center."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        g3 = np.array([3.0, 4.0, 5.0, 6.0, 7.0])

        bf_result = scirs2.brown_forsythe_py(g1, g2, g3)
        lev_result = scirs2.levene_py(g1, g2, g3, center="median")

        # Brown-Forsythe should be equivalent to Levene with median
        assert abs(bf_result["statistic"] - lev_result["statistic"]) < 1e-6
        assert abs(bf_result["pvalue"] - lev_result["pvalue"]) < 1e-6

    def test_homogeneity_order_invariance(self):
        """Test that homogeneity tests are invariant to group order."""
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        g3 = np.array([7.0, 8.0, 9.0])

        # Levene test
        lev1 = scirs2.levene_py(g1, g2, g3)
        lev2 = scirs2.levene_py(g3, g1, g2)
        lev3 = scirs2.levene_py(g2, g3, g1)

        assert abs(lev1["statistic"] - lev2["statistic"]) < 1e-6
        assert abs(lev1["statistic"] - lev3["statistic"]) < 1e-6
        assert abs(lev1["pvalue"] - lev2["pvalue"]) < 1e-6
        assert abs(lev1["pvalue"] - lev3["pvalue"]) < 1e-6

        # Bartlett test
        bar1 = scirs2.bartlett_test_py(g1, g2, g3)
        bar2 = scirs2.bartlett_test_py(g3, g1, g2)
        bar3 = scirs2.bartlett_test_py(g2, g3, g1)

        assert abs(bar1["statistic"] - bar2["statistic"]) < 1e-6
        assert abs(bar1["statistic"] - bar3["statistic"]) < 1e-6
        assert abs(bar1["pvalue"] - bar2["pvalue"]) < 1e-6
        assert abs(bar1["pvalue"] - bar3["pvalue"]) < 1e-6


class TestHomogeneityComparison:
    """Test comparisons between different homogeneity tests."""

    def test_bartlett_vs_levene_normal_data(self):
        """Test Bartlett vs Levene with normal data."""
        np.random.seed(42)
        g1 = np.random.normal(0, 1, 100)
        g2 = np.random.normal(0, 1.5, 100)

        bartlett_result = scirs2.bartlett_test_py(g1, g2)
        levene_result = scirs2.levene_py(g1, g2)

        # Both should detect the difference in variances
        assert bartlett_result["pvalue"] < 0.05
        assert levene_result["pvalue"] < 0.05

    def test_levene_centers_comparison(self):
        """Test different center methods in Levene's test."""
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        mean_result = scirs2.levene_py(g1, g2, center="mean")
        median_result = scirs2.levene_py(g1, g2, center="median")

        # Results should be similar but not necessarily identical
        assert "statistic" in mean_result
        assert "statistic" in median_result
        assert "pvalue" in mean_result
        assert "pvalue" in median_result
