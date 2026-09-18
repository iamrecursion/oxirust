"""Tests for chi-square tests and post-hoc analyses."""

import pytest
import numpy as np
import scirs2


class TestChi2Independence:
    """Test chi-square test for independence."""

    def test_chi2_independence_basic_2x2(self):
        """Test basic chi-square independence on 2x2 table."""
        # Example: Treatment vs Outcome
        observed = np.array([
            [10, 20],
            [15, 25]
        ], dtype=np.int64)

        result = scirs2.chi2_independence_py(observed)

        assert "statistic" in result
        assert "pvalue" in result
        assert "df" in result
        assert "expected" in result
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1
        assert result["df"] == 1  # (2-1) * (2-1)

    def test_chi2_independence_3x3(self):
        """Test chi-square independence on 3x3 table."""
        # Example: Education level vs Income level
        observed = np.array([
            [20, 30, 10],
            [15, 25, 20],
            [25, 15, 40]
        ], dtype=np.int64)

        result = scirs2.chi2_independence_py(observed)

        assert result["df"] == 4  # (3-1) * (3-1)
        assert "expected" in result
        # Expected frequencies should be positive
        expected = result["expected"]
        assert np.all(expected > 0)

    def test_chi2_independence_3x2(self):
        """Test chi-square independence on 3x2 table."""
        # Example: 3 treatments, 2 outcomes
        observed = np.array([
            [30, 20],
            [40, 10],
            [25, 25]
        ], dtype=np.int64)

        result = scirs2.chi2_independence_py(observed)

        assert result["df"] == 2  # (3-1) * (2-1)
        assert "statistic" in result
        assert "pvalue" in result

    def test_chi2_independence_dependent_variables(self):
        """Test with clearly dependent variables."""
        # Create data with strong dependency
        observed = np.array([
            [50, 5],
            [5, 50]
        ], dtype=np.int64)

        result = scirs2.chi2_independence_py(observed)

        # Should have low p-value (reject independence)
        assert result["pvalue"] < 0.05

    def test_chi2_independence_independent_variables(self):
        """Test with independent variables."""
        # Create data with approximately independent variables
        observed = np.array([
            [25, 25],
            [25, 25]
        ], dtype=np.int64)

        result = scirs2.chi2_independence_py(observed)

        # Should have high p-value (fail to reject independence)
        assert result["pvalue"] > 0.05

    def test_chi2_independence_large_table(self):
        """Test with larger contingency table."""
        # 4x5 table
        observed = np.array([
            [10, 20, 15, 25, 30],
            [15, 18, 20, 22, 25],
            [12, 16, 18, 24, 30],
            [20, 22, 17, 19, 22]
        ], dtype=np.int64)

        result = scirs2.chi2_independence_py(observed)

        assert result["df"] == 12  # (4-1) * (5-1)
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_chi2_independence_expected_frequencies(self):
        """Test that expected frequencies are calculated correctly."""
        observed = np.array([
            [20, 30],
            [40, 10]
        ], dtype=np.int64)

        result = scirs2.chi2_independence_py(observed)

        # Manually calculate expected frequencies
        row_sums = observed.sum(axis=1)
        col_sums = observed.sum(axis=0)
        total = observed.sum()

        expected = np.outer(row_sums, col_sums) / total

        # Check expected frequencies match
        assert np.allclose(result["expected"], expected, rtol=1e-10)

    def test_chi2_independence_insufficient_dimensions(self):
        """Test error with insufficient dimensions."""
        # Only 1 row
        observed = np.array([[10, 20, 30]], dtype=np.int64)

        with pytest.raises(RuntimeError, match="at least 2 rows"):
            scirs2.chi2_independence_py(observed)

    def test_chi2_independence_single_column(self):
        """Test error with single column."""
        # Only 1 column
        observed = np.array([[10], [20], [30]], dtype=np.int64)

        with pytest.raises(RuntimeError, match="at least 2"):
            scirs2.chi2_independence_py(observed)


class TestChi2Yates:
    """Test chi-square test with Yates' correction."""

    def test_chi2_yates_basic(self):
        """Test basic Yates' correction on 2x2 table."""
        observed = np.array([
            [10, 20],
            [15, 25]
        ], dtype=np.int64)

        result = scirs2.chi2_yates_py(observed)

        assert "statistic" in result
        assert "pvalue" in result
        assert "df" in result
        assert "expected" in result
        assert result["df"] == 1
        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_chi2_yates_vs_independence(self):
        """Test that Yates gives different (usually lower) statistic."""
        observed = np.array([
            [8, 12],
            [10, 15]
        ], dtype=np.int64)

        yates_result = scirs2.chi2_yates_py(observed)
        chi2_result = scirs2.chi2_independence_py(observed)

        # Yates should give lower or equal statistic
        assert yates_result["statistic"] <= chi2_result["statistic"]

    def test_chi2_yates_small_sample(self):
        """Test Yates' correction with small sample size."""
        # Small frequencies where Yates' correction is beneficial
        observed = np.array([
            [3, 7],
            [5, 5]
        ], dtype=np.int64)

        result = scirs2.chi2_yates_py(observed)

        assert result["statistic"] >= 0
        assert 0 <= result["pvalue"] <= 1

    def test_chi2_yates_strong_association(self):
        """Test Yates with strong association."""
        observed = np.array([
            [30, 5],
            [5, 30]
        ], dtype=np.int64)

        result = scirs2.chi2_yates_py(observed)

        # Should have low p-value
        assert result["pvalue"] < 0.05

    def test_chi2_yates_no_association(self):
        """Test Yates with no association."""
        observed = np.array([
            [25, 25],
            [25, 25]
        ], dtype=np.int64)

        result = scirs2.chi2_yates_py(observed)

        # Should have high p-value
        assert result["pvalue"] > 0.05
        # Statistic should be 0
        assert result["statistic"] == 0.0

    def test_chi2_yates_invalid_dimensions(self):
        """Test error when not 2x2 table."""
        # 3x2 table
        observed = np.array([
            [10, 20],
            [15, 25],
            [20, 30]
        ], dtype=np.int64)

        with pytest.raises(RuntimeError, match="2x2"):
            scirs2.chi2_yates_py(observed)

    def test_chi2_yates_invalid_dimensions_3x3(self):
        """Test error with 3x3 table."""
        observed = np.array([
            [10, 20, 30],
            [15, 25, 35],
            [20, 30, 40]
        ], dtype=np.int64)

        with pytest.raises(RuntimeError, match="2x2"):
            scirs2.chi2_yates_py(observed)


class TestTukeyHSD:
    """Test Tukey's Honestly Significant Difference post-hoc test."""

    def test_tukey_hsd_basic_three_groups(self):
        """Test basic Tukey HSD with three groups."""
        group1 = np.array([85.0, 82.0, 78.0, 88.0, 91.0])
        group2 = np.array([76.0, 80.0, 82.0, 84.0, 79.0])
        group3 = np.array([91.0, 89.0, 93.0, 87.0, 90.0])

        result = scirs2.tukey_hsd_py(group1, group2, group3)

        # Should return list of 3 comparisons: 0-1, 0-2, 1-2
        assert len(result) == 3

        # Check each comparison
        for comparison in result:
            assert "group1" in comparison
            assert "group2" in comparison
            assert "mean_diff" in comparison
            assert "pvalue" in comparison
            assert "significant" in comparison
            assert isinstance(comparison["significant"], bool)
            assert 0 <= comparison["pvalue"] <= 1

    def test_tukey_hsd_two_groups(self):
        """Test Tukey HSD with two groups."""
        group1 = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        group2 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

        result = scirs2.tukey_hsd_py(group1, group2)

        # Should return 1 comparison
        assert len(result) == 1
        assert result[0]["group1"] == 0
        assert result[0]["group2"] == 1
        # Groups are clearly different
        assert result[0]["significant"] is True

    def test_tukey_hsd_four_groups(self):
        """Test Tukey HSD with four groups."""
        group1 = np.array([5.0, 6.0, 7.0])
        group2 = np.array([8.0, 9.0, 10.0])
        group3 = np.array([11.0, 12.0, 13.0])
        group4 = np.array([14.0, 15.0, 16.0])

        result = scirs2.tukey_hsd_py(group1, group2, group3, group4)

        # Should return 6 comparisons: C(4,2) = 6
        assert len(result) == 6

        # Verify all pairs are present
        pairs = {(r["group1"], r["group2"]) for r in result}
        expected_pairs = {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
        assert pairs == expected_pairs

    def test_tukey_hsd_equal_groups(self):
        """Test Tukey HSD with equal group means."""
        np.random.seed(42)
        group1 = np.random.normal(5, 0.1, 20)
        group2 = np.random.normal(5, 0.1, 20)
        group3 = np.random.normal(5, 0.1, 20)

        result = scirs2.tukey_hsd_py(group1, group2, group3, alpha=0.05)

        # Most comparisons should not be significant
        significant_count = sum(1 for r in result if r["significant"])
        assert significant_count == 0  # Expect no significant differences

    def test_tukey_hsd_different_groups(self):
        """Test Tukey HSD with clearly different groups."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        group3 = np.array([20.0, 21.0, 22.0, 23.0, 24.0])

        result = scirs2.tukey_hsd_py(group1, group2, group3, alpha=0.05)

        # All comparisons should be significant
        assert all(r["significant"] for r in result)

    def test_tukey_hsd_custom_alpha(self):
        """Test Tukey HSD with custom alpha level."""
        group1 = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        group2 = np.array([5.5, 6.5, 7.5, 8.5, 9.5])
        group3 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        # Test with supported alpha levels (0.05 and 0.01)
        result_005 = scirs2.tukey_hsd_py(group1, group2, group3, alpha=0.05)
        result_001 = scirs2.tukey_hsd_py(group1, group2, group3, alpha=0.01)

        # Lower alpha (0.01) should have fewer or equal significant results
        sig_005 = sum(1 for r in result_005 if r["significant"])
        sig_001 = sum(1 for r in result_001 if r["significant"])
        assert sig_001 <= sig_005

    def test_tukey_hsd_mean_differences(self):
        """Test that mean differences are calculated correctly."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        result = scirs2.tukey_hsd_py(group1, group2)

        # Mean of group1 = 3.0, mean of group2 = 8.0
        # Difference = 5.0
        expected_diff = 5.0
        assert abs(abs(result[0]["mean_diff"]) - expected_diff) < 0.01

    def test_tukey_hsd_different_sample_sizes(self):
        """Test Tukey HSD with different sample sizes."""
        group1 = np.array([5.0, 6.0, 7.0])
        group2 = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        group3 = np.array([13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])

        result = scirs2.tukey_hsd_py(group1, group2, group3)

        assert len(result) == 3
        assert all("pvalue" in r for r in result)

    def test_tukey_hsd_insufficient_groups(self):
        """Test error with insufficient groups."""
        group1 = np.array([1.0, 2.0, 3.0])

        with pytest.raises(RuntimeError, match="at least 2 groups"):
            scirs2.tukey_hsd_py(group1)


class TestChi2Comparison:
    """Test comparisons between chi-square tests."""

    def test_yates_vs_independence_correction_effect(self):
        """Test that Yates' correction reduces the statistic."""
        # Use small sample where correction matters
        observed = np.array([
            [5, 10],
            [8, 7]
        ], dtype=np.int64)

        yates_result = scirs2.chi2_yates_py(observed)
        chi2_result = scirs2.chi2_independence_py(observed)

        # Yates should give lower statistic (more conservative)
        assert yates_result["statistic"] < chi2_result["statistic"]
        # Yates should give higher p-value (more conservative)
        assert yates_result["pvalue"] > chi2_result["pvalue"]


class TestTukeyEdgeCases:
    """Test edge cases for Tukey HSD."""

    def test_tukey_hsd_single_observation_groups(self):
        """Test Tukey HSD with very small groups."""
        # Minimum 2 observations per group for variance
        group1 = np.array([1.0, 2.0])
        group2 = np.array([3.0, 4.0])
        group3 = np.array([5.0, 6.0])

        result = scirs2.tukey_hsd_py(group1, group2, group3)

        assert len(result) == 3
        assert all("pvalue" in r for r in result)

    def test_tukey_hsd_large_number_of_groups(self):
        """Test Tukey HSD with many groups."""
        np.random.seed(42)
        groups = [np.random.normal(i, 1, 10) for i in range(6)]

        result = scirs2.tukey_hsd_py(*groups)

        # C(6, 2) = 15 comparisons
        assert len(result) == 15
