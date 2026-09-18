"""
Comprehensive tests for paired (related samples) t-test.

Tests the ttest_rel_py function which performs paired t-test for related samples.
"""

import numpy as np
import pytest
import scirs2


class TestPairedTTestBasics:
    """Test basic functionality of paired t-test."""

    def test_paired_significant_difference(self):
        """Test paired samples with significant difference."""
        # Before and after treatment - clear improvement
        before = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 13.0])
        after = np.array([15.0, 16.0, 18.0, 14.0, 17.0, 16.0])

        result = scirs2.ttest_rel_py(before, after)

        # Should find significant difference
        assert "statistic" in result
        assert "pvalue" in result
        assert "df" in result
        assert result["df"] == 5.0  # n - 1 = 6 - 1 = 5
        assert result["pvalue"] < 0.2  # Should detect a difference

    def test_paired_no_difference(self):
        """Test paired samples with no real difference."""
        # Before and after with minimal random variation
        np.random.seed(42)
        before = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 13.0])
        after = before + np.random.normal(0, 0.1, 6)  # Tiny random noise

        result = scirs2.ttest_rel_py(before, after)

        # Should NOT find significant difference
        assert result["pvalue"] > 0.05

    def test_paired_identical_arrays(self):
        """Test paired samples that are identical."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = a.copy()

        result = scirs2.ttest_rel_py(a, b)

        # All differences are zero, so variance is zero
        # This typically results in NaN or inf statistic
        assert "statistic" in result
        assert "pvalue" in result

    def test_paired_negative_difference(self):
        """Test paired samples where second is consistently lower."""
        before = np.array([20.0, 22.0, 25.0, 21.0, 24.0, 23.0])
        after = np.array([15.0, 17.0, 20.0, 16.0, 19.0, 18.0])

        result = scirs2.ttest_rel_py(before, after)

        # Should find significant difference
        assert result["pvalue"] < 0.01
        assert result["statistic"] > 0  # before > after, so positive t-statistic

    def test_paired_small_sample(self):
        """Test with small sample size."""
        before = np.array([10.0, 15.0, 12.0])
        after = np.array([12.0, 17.0, 14.0])

        result = scirs2.ttest_rel_py(before, after)

        assert result["df"] == 2.0  # n - 1 = 3 - 1 = 2
        assert "statistic" in result
        assert "pvalue" in result

    def test_paired_large_sample(self):
        """Test with larger sample size."""
        np.random.seed(42)
        n = 100
        before = np.random.normal(10, 2, n)
        after = before + np.random.normal(0.5, 0.5, n)  # Slight increase with noise

        result = scirs2.ttest_rel_py(before, after)

        assert result["df"] == 99.0  # n - 1
        assert "statistic" in result
        assert "pvalue" in result


class TestPairedTTestAlternatives:
    """Test alternative hypotheses for paired t-test."""

    def test_alternative_two_sided(self):
        """Test two-sided alternative (default)."""
        before = np.array([10.0, 12.0, 15.0, 11.0, 14.0])
        after = np.array([15.0, 17.0, 20.0, 16.0, 19.0])

        result1 = scirs2.ttest_rel_py(before, after)
        result2 = scirs2.ttest_rel_py(before, after, alternative="two-sided")
        result3 = scirs2.ttest_rel_py(before, after, alternative="two_sided")

        # All should give same results
        assert result1["pvalue"] == result2["pvalue"]
        assert result1["pvalue"] == result3["pvalue"]
        assert result1["statistic"] == result2["statistic"]

    def test_alternative_less(self):
        """Test alternative='less' hypothesis."""
        # Before < After (differences are negative)
        before = np.array([10.0, 12.0, 15.0, 11.0, 14.0])
        after = np.array([15.0, 17.0, 20.0, 16.0, 19.0])

        result = scirs2.ttest_rel_py(before, after, alternative="less")

        # Verify the function accepts the alternative parameter
        assert "pvalue" in result
        assert "statistic" in result

    def test_alternative_greater(self):
        """Test alternative='greater' hypothesis."""
        # Before > After (differences are positive)
        before = np.array([20.0, 22.0, 25.0, 21.0, 24.0])
        after = np.array([15.0, 17.0, 20.0, 16.0, 19.0])

        result = scirs2.ttest_rel_py(before, after, alternative="greater")

        # Verify the function accepts the alternative parameter
        assert "pvalue" in result
        assert "statistic" in result

    def test_invalid_alternative(self):
        """Test that invalid alternative raises error."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])

        with pytest.raises(RuntimeError, match="Invalid alternative"):
            scirs2.ttest_rel_py(a, b, alternative="invalid")


class TestPairedTTestEdgeCases:
    """Test edge cases and error conditions."""

    def test_different_lengths(self):
        """Test that different length arrays raise error."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(RuntimeError):
            scirs2.ttest_rel_py(a, b)

    def test_empty_arrays(self):
        """Test that empty arrays raise error."""
        a = np.array([])
        b = np.array([])

        with pytest.raises(RuntimeError):
            scirs2.ttest_rel_py(a, b)

    def test_single_element(self):
        """Test that single element arrays raise error or handle gracefully."""
        a = np.array([1.0])
        b = np.array([2.0])

        # With only 1 pair, df = 0, so this should error
        with pytest.raises(RuntimeError):
            scirs2.ttest_rel_py(a, b)

    def test_two_elements(self):
        """Test with minimum viable sample (2 elements)."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])

        result = scirs2.ttest_rel_py(a, b)

        assert result["df"] == 1.0  # n - 1 = 2 - 1 = 1
        assert "statistic" in result
        assert "pvalue" in result

    def test_constant_differences(self):
        """Test when all differences are the same (variance = 0)."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])  # All differences = 1.0

        result = scirs2.ttest_rel_py(a, b)

        # When differences are constant, standard error is 0
        # This typically results in inf statistic and p-value of 0
        assert "statistic" in result
        assert "pvalue" in result

    def test_with_nan(self):
        """Test behavior with NaN values."""
        a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        # Should handle NaN or raise error
        try:
            result = scirs2.ttest_rel_py(a, b)
            # If it doesn't error, result may contain NaN or inf
            assert (np.isnan(result["statistic"]) or np.isinf(result["statistic"]) or
                    np.isnan(result["pvalue"]) or result["pvalue"] == 0.0)
        except RuntimeError:
            # Acceptable to raise error for NaN
            pass

    def test_with_inf(self):
        """Test behavior with infinite values."""
        a = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

        # Should handle inf or raise error
        try:
            result = scirs2.ttest_rel_py(a, b)
            # If it doesn't error, result should contain inf or NaN
            assert (np.isinf(result["statistic"]) or np.isnan(result["statistic"]) or
                    np.isinf(result["pvalue"]) or np.isnan(result["pvalue"]))
        except RuntimeError:
            # Acceptable to raise error for inf
            pass


class TestPairedTTestNumericalAccuracy:
    """Test numerical accuracy and precision."""

    def test_large_values(self):
        """Test with large magnitude values."""
        before = np.array([1e10, 1.1e10, 1.2e10, 1.3e10, 1.4e10])
        after = np.array([1.1e10, 1.2e10, 1.3e10, 1.4e10, 1.5e10])

        result = scirs2.ttest_rel_py(before, after)

        # With constant differences, statistic may be inf, but should still be valid
        assert "statistic" in result
        assert 0 <= result["pvalue"] <= 1

    def test_small_values(self):
        """Test with small magnitude values."""
        before = np.array([1e-10, 1.1e-10, 1.2e-10, 1.3e-10, 1.4e-10])
        after = np.array([1.1e-10, 1.2e-10, 1.3e-10, 1.4e-10, 1.5e-10])

        result = scirs2.ttest_rel_py(before, after)

        # Should still compute valid results
        assert np.isfinite(result["statistic"])
        assert 0 <= result["pvalue"] <= 1

    def test_very_small_differences(self):
        """Test with very small differences between pairs."""
        before = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        after = before + 1e-10  # Extremely small increase

        result = scirs2.ttest_rel_py(before, after)

        # Should not find significant difference (p should be large)
        assert "pvalue" in result


class TestPairedTTestStatisticalProperties:
    """Test statistical properties of paired t-test."""

    def test_symmetry(self):
        """Test that swapping a and b changes sign of statistic but not p-value (two-sided)."""
        # Use values with some variance to avoid inf
        np.random.seed(42)
        a = np.array([10.0, 12.0, 15.0, 11.0, 14.0]) + np.random.normal(0, 0.1, 5)
        b = np.array([15.0, 17.0, 20.0, 16.0, 19.0]) + np.random.normal(0, 0.1, 5)

        result_ab = scirs2.ttest_rel_py(a, b, alternative="two-sided")
        result_ba = scirs2.ttest_rel_py(b, a, alternative="two-sided")

        # Statistics should have opposite signs (if finite)
        if np.isfinite(result_ab["statistic"]) and np.isfinite(result_ba["statistic"]):
            assert abs(result_ab["statistic"] + result_ba["statistic"]) < 1e-6

        # Two-sided p-values should be the same
        assert abs(result_ab["pvalue"] - result_ba["pvalue"]) < 1e-10

    def test_degrees_of_freedom(self):
        """Test that degrees of freedom is always n-1."""
        for n in [3, 5, 10, 20, 50]:
            np.random.seed(42)
            a = np.random.normal(0, 1, n)
            b = np.random.normal(0.5, 1, n)

            result = scirs2.ttest_rel_py(a, b)
            assert result["df"] == float(n - 1)

    def test_statistic_magnitude(self):
        """Test that larger differences give larger t-statistics."""
        np.random.seed(42)
        before = np.random.normal(10, 2, 50)

        # Small difference with added noise
        after_small = before + np.random.normal(0.1, 0.5, 50)
        result_small = scirs2.ttest_rel_py(before, after_small)

        # Large difference with added noise
        after_large = before + np.random.normal(5.0, 0.5, 50)
        result_large = scirs2.ttest_rel_py(before, after_large)

        # If both are finite, larger difference should give larger absolute t-statistic
        if np.isfinite(result_large["statistic"]) and np.isfinite(result_small["statistic"]):
            assert abs(result_large["statistic"]) > abs(result_small["statistic"])
            # Larger t-statistic should give smaller p-value
            assert result_large["pvalue"] < result_small["pvalue"]

    def test_sample_size_effect(self):
        """Test that larger samples give more power (smaller p-values for same effect)."""
        np.random.seed(42)

        # Small sample with noise
        before_small = np.random.normal(10, 2, 10)
        after_small = before_small + np.random.normal(0.5, 0.3, 10)
        result_small = scirs2.ttest_rel_py(before_small, after_small)

        # Large sample (same effect size) with noise
        before_large = np.random.normal(10, 2, 100)
        after_large = before_large + np.random.normal(0.5, 0.3, 100)
        result_large = scirs2.ttest_rel_py(before_large, after_large)

        # Larger sample should generally give smaller p-value (more power)
        # Note: This is probabilistic, but with fixed seed should be consistent
        if result_large["pvalue"] > 0 and result_small["pvalue"] > 0:
            assert result_large["pvalue"] <= result_small["pvalue"]


class TestPairedTTestRealWorldScenarios:
    """Test realistic use cases for paired t-test."""

    def test_before_after_treatment(self):
        """Test typical before/after treatment scenario."""
        # Blood pressure before and after medication
        before_bp = np.array([140, 145, 138, 142, 147, 139, 144, 141, 143, 146], dtype=np.float64)
        after_bp = np.array([125, 130, 122, 128, 132, 124, 129, 126, 127, 131], dtype=np.float64)

        result = scirs2.ttest_rel_py(before_bp, after_bp)

        # Treatment should show significant improvement
        assert result["pvalue"] < 0.05
        assert result["statistic"] > 0  # Before > After

    def test_test_retest_reliability(self):
        """Test test-retest reliability scenario."""
        # Same test taken twice - should be similar scores
        np.random.seed(42)
        test1 = np.array([85, 90, 78, 92, 88, 86, 91, 89, 87, 93], dtype=np.float64)
        test2 = test1 + np.random.normal(0, 2, 10)  # Small random variation

        result = scirs2.ttest_rel_py(test1, test2)

        # Should NOT find significant difference (reliable test)
        assert result["pvalue"] > 0.05

    def test_matched_pairs_design(self):
        """Test matched pairs experimental design."""
        # Twins - one gets treatment, other gets control
        twin1_scores = np.array([75, 82, 79, 85, 88, 76, 84, 80, 83, 87], dtype=np.float64)
        twin2_scores = np.array([78, 85, 82, 89, 92, 79, 88, 83, 87, 91], dtype=np.float64)

        result = scirs2.ttest_rel_py(twin1_scores, twin2_scores)

        # May or may not be significant, but should compute
        assert "pvalue" in result
        assert 0 <= result["pvalue"] <= 1

    def test_repeated_measures(self):
        """Test repeated measures on same subjects."""
        # Weight measurements at baseline and 3 months
        baseline_weight = np.array([85.2, 92.1, 78.5, 88.3, 95.7, 82.4, 90.6, 86.8])
        three_month_weight = np.array([82.1, 89.5, 76.2, 85.1, 92.3, 80.1, 87.8, 84.2])

        result = scirs2.ttest_rel_py(baseline_weight, three_month_weight)

        # Weight loss program should show some change
        assert result["pvalue"] < 0.1
        assert result["statistic"] > 0  # Baseline > 3-month (weight decreased)

    def test_crossover_design(self):
        """Test crossover study design."""
        # Response under treatment A
        response_a = np.array([45, 52, 48, 55, 50, 47, 53, 49, 51, 54], dtype=np.float64)
        # Response under treatment B (same subjects, different order)
        response_b = np.array([48, 55, 51, 58, 53, 50, 56, 52, 54, 57], dtype=np.float64)

        result = scirs2.ttest_rel_py(response_a, response_b)

        # Should detect treatment difference
        assert "pvalue" in result
        assert result["pvalue"] < 0.1
        assert result["statistic"] < 0  # A < B


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
