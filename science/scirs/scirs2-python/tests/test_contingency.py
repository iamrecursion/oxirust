"""
Comprehensive tests for contingency table analysis functions.

Tests Fisher's exact test, odds ratio, and relative risk calculations.
"""

import numpy as np
import pytest
import scirs2


class TestFisherExactTest:
    """Test Fisher's exact test for 2x2 contingency tables."""

    def test_fisher_basic_2x2(self):
        """Test Fisher's exact test on a basic 2x2 table."""
        # Example from Fisher's Tea Test
        table = np.array([[8, 2], [1, 5]], dtype=np.float64)
        result = scirs2.fisher_exact_py(table)

        assert "odds_ratio" in result
        assert "pvalue" in result
        assert result["odds_ratio"] > 0
        assert 0 <= result["pvalue"] <= 1

    def test_fisher_strong_association(self):
        """Test Fisher's exact test with strong association."""
        # Strong positive association
        table = np.array([[10, 1], [1, 10]], dtype=np.float64)
        result = scirs2.fisher_exact_py(table)

        # Should have large odds ratio and small p-value
        assert result["odds_ratio"] > 10
        assert result["pvalue"] < 0.05

    def test_fisher_no_association(self):
        """Test Fisher's exact test with no association."""
        # No association (odds ratio close to 1)
        table = np.array([[10, 10], [10, 10]], dtype=np.float64)
        result = scirs2.fisher_exact_py(table)

        # Odds ratio should be close to 1
        assert abs(result["odds_ratio"] - 1.0) < 0.01
        # P-value should be large (not significant)
        assert result["pvalue"] > 0.5

    def test_fisher_alternative_hypotheses(self):
        """Test Fisher's exact test with different alternatives."""
        table = np.array([[15, 5], [3, 12]], dtype=np.float64)

        # Two-sided (default)
        result_two = scirs2.fisher_exact_py(table, alternative="two-sided")
        assert "pvalue" in result_two

        # One-sided: less
        result_less = scirs2.fisher_exact_py(table, alternative="less")
        assert "pvalue" in result_less

        # One-sided: greater
        result_greater = scirs2.fisher_exact_py(table, alternative="greater")
        assert "pvalue" in result_greater

    def test_fisher_perfect_association(self):
        """Test Fisher's exact test with perfect association."""
        # All in diagonal (perfect positive association)
        table = np.array([[10, 0], [0, 10]], dtype=np.float64)
        result = scirs2.fisher_exact_py(table)

        # Odds ratio should be infinite or very large
        assert np.isinf(result["odds_ratio"]) or result["odds_ratio"] > 1000
        # P-value should be small (significant)
        assert result["pvalue"] < 0.05

    def test_fisher_perfect_negative_association(self):
        """Test Fisher's exact test with perfect negative association."""
        # All in off-diagonal
        table = np.array([[0, 10], [10, 0]], dtype=np.float64)
        result = scirs2.fisher_exact_py(table)

        # Odds ratio should be 0 or very small
        assert result["odds_ratio"] == 0.0 or result["odds_ratio"] < 0.01
        # P-value should be small (significant)
        assert result["pvalue"] < 0.05

    def test_fisher_small_sample(self):
        """Test Fisher's exact test with small sample sizes."""
        # Very small samples where chi-square wouldn't be appropriate
        table = np.array([[2, 1], [1, 2]], dtype=np.float64)
        result = scirs2.fisher_exact_py(table)

        # Should still compute valid results
        assert result["odds_ratio"] > 0
        assert 0 <= result["pvalue"] <= 1

    def test_fisher_large_sample(self):
        """Test Fisher's exact test with larger sample sizes."""
        # Larger table
        table = np.array([[100, 50], [30, 120]], dtype=np.float64)
        result = scirs2.fisher_exact_py(table)

        assert result["odds_ratio"] > 0
        assert 0 <= result["pvalue"] <= 1

    def test_fisher_invalid_alternative(self):
        """Test that invalid alternative raises error."""
        table = np.array([[10, 5], [5, 10]], dtype=np.float64)

        with pytest.raises(RuntimeError, match="alternative"):
            scirs2.fisher_exact_py(table, alternative="invalid")

    def test_fisher_wrong_dimensions(self):
        """Test that non-2x2 table raises error."""
        # 3x3 table
        table = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

        with pytest.raises(RuntimeError, match="2x2"):
            scirs2.fisher_exact_py(table)

    def test_fisher_negative_values(self):
        """Test that negative values raise error."""
        table = np.array([[10, -5], [5, 10]], dtype=np.float64)

        with pytest.raises(RuntimeError):
            scirs2.fisher_exact_py(table)


class TestOddsRatio:
    """Test odds ratio calculation."""

    def test_odds_ratio_basic(self):
        """Test basic odds ratio calculation."""
        # Example: Disease and exposure
        # Disease+  Disease-
        # Exposed+    a=10      b=5
        # Exposed-    c=3       d=12
        table = np.array([[10, 5], [3, 12]], dtype=np.float64)
        or_val = scirs2.odds_ratio_py(table)

        # OR = (10*12)/(5*3) = 120/15 = 8.0
        assert abs(or_val - 8.0) < 0.001

    def test_odds_ratio_equals_one(self):
        """Test odds ratio when OR = 1 (no association)."""
        table = np.array([[10, 10], [10, 10]], dtype=np.float64)
        or_val = scirs2.odds_ratio_py(table)

        # OR = (10*10)/(10*10) = 1.0
        assert abs(or_val - 1.0) < 0.001

    def test_odds_ratio_less_than_one(self):
        """Test odds ratio less than 1 (protective effect)."""
        # Exposure associated with lower odds of disease
        table = np.array([[2, 10], [10, 5]], dtype=np.float64)
        or_val = scirs2.odds_ratio_py(table)

        # OR = (2*5)/(10*10) = 10/100 = 0.1
        assert abs(or_val - 0.1) < 0.001
        assert or_val < 1.0

    def test_odds_ratio_greater_than_one(self):
        """Test odds ratio greater than 1 (risk factor)."""
        # Exposure associated with higher odds of disease
        table = np.array([[20, 5], [5, 20]], dtype=np.float64)
        or_val = scirs2.odds_ratio_py(table)

        # OR = (20*20)/(5*5) = 400/25 = 16.0
        assert abs(or_val - 16.0) < 0.001
        assert or_val > 1.0

    def test_odds_ratio_zero_cell(self):
        """Test odds ratio with zero in one cell."""
        # b=0 case
        table1 = np.array([[10, 0], [5, 10]], dtype=np.float64)
        or_val1 = scirs2.odds_ratio_py(table1)
        # OR should be infinite
        assert np.isinf(or_val1) or or_val1 > 1000

        # c=0 case
        table2 = np.array([[10, 5], [0, 10]], dtype=np.float64)
        or_val2 = scirs2.odds_ratio_py(table2)
        # OR should be infinite
        assert np.isinf(or_val2) or or_val2 > 1000

    def test_odds_ratio_diagonal_zeros(self):
        """Test odds ratio with zeros in diagonal."""
        # a=0, d=0 (perfect negative association)
        table = np.array([[0, 10], [10, 0]], dtype=np.float64)
        or_val = scirs2.odds_ratio_py(table)

        # OR should be 0
        assert or_val == 0.0

    def test_odds_ratio_large_values(self):
        """Test odds ratio with large values."""
        table = np.array([[1000, 500], [200, 2000]], dtype=np.float64)
        or_val = scirs2.odds_ratio_py(table)

        # OR = (1000*2000)/(500*200) = 2000000/100000 = 20.0
        assert abs(or_val - 20.0) < 0.001

    def test_odds_ratio_small_values(self):
        """Test odds ratio with small values."""
        table = np.array([[1, 2], [2, 1]], dtype=np.float64)
        or_val = scirs2.odds_ratio_py(table)

        # OR = (1*1)/(2*2) = 1/4 = 0.25
        assert abs(or_val - 0.25) < 0.001

    def test_odds_ratio_wrong_dimensions(self):
        """Test that non-2x2 table raises error."""
        table = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        with pytest.raises(RuntimeError):
            scirs2.odds_ratio_py(table)

    def test_odds_ratio_negative_values(self):
        """Test that negative values raise error."""
        table = np.array([[10, -5], [5, 10]], dtype=np.float64)

        with pytest.raises(RuntimeError):
            scirs2.odds_ratio_py(table)


class TestRelativeRisk:
    """Test relative risk (risk ratio) calculation."""

    def test_relative_risk_basic(self):
        """Test basic relative risk calculation."""
        # Example: Cohort study
        #            Disease+  Disease-
        # Exposed+      20        80       (Risk = 20/100 = 0.2)
        # Exposed-      10        90       (Risk = 10/100 = 0.1)
        table = np.array([[20, 80], [10, 90]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR = (20/100)/(10/100) = 0.2/0.1 = 2.0
        assert abs(rr_val - 2.0) < 0.001

    def test_relative_risk_equals_one(self):
        """Test relative risk when RR = 1 (no association)."""
        # Equal risk in both groups
        table = np.array([[10, 40], [10, 40]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR = (10/50)/(10/50) = 1.0
        assert abs(rr_val - 1.0) < 0.001

    def test_relative_risk_less_than_one(self):
        """Test relative risk less than 1 (protective effect)."""
        # Exposure associated with lower risk
        table = np.array([[5, 45], [20, 30]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR = (5/50)/(20/50) = 0.1/0.4 = 0.25
        assert abs(rr_val - 0.25) < 0.001
        assert rr_val < 1.0

    def test_relative_risk_greater_than_one(self):
        """Test relative risk greater than 1 (risk factor)."""
        # Exposure associated with higher risk
        table = np.array([[30, 20], [10, 40]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR = (30/50)/(10/50) = 0.6/0.2 = 3.0
        assert abs(rr_val - 3.0) < 0.001
        assert rr_val > 1.0

    def test_relative_risk_zero_unexposed(self):
        """Test relative risk when unexposed have zero cases."""
        # c=0 case (undefined or infinite RR)
        table = np.array([[10, 40], [0, 50]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR should be infinite
        assert np.isinf(rr_val) or rr_val > 1000

    def test_relative_risk_zero_exposed(self):
        """Test relative risk when exposed have zero cases."""
        # a=0 case (RR = 0)
        table = np.array([[0, 50], [10, 40]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR should be 0
        assert rr_val == 0.0

    def test_relative_risk_high_incidence(self):
        """Test relative risk with high incidence."""
        # High disease incidence in both groups
        table = np.array([[80, 20], [70, 30]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR = (80/100)/(70/100) = 0.8/0.7 ≈ 1.143
        assert abs(rr_val - 1.142857) < 0.001

    def test_relative_risk_low_incidence(self):
        """Test relative risk with low incidence."""
        # Low disease incidence in both groups
        table = np.array([[2, 98], [1, 99]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR = (2/100)/(1/100) = 0.02/0.01 = 2.0
        assert abs(rr_val - 2.0) < 0.001

    def test_relative_risk_large_sample(self):
        """Test relative risk with large sample sizes."""
        table = np.array([[500, 4500], [200, 4800]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR = (500/5000)/(200/5000) = 0.1/0.04 = 2.5
        assert abs(rr_val - 2.5) < 0.001

    def test_relative_risk_small_sample(self):
        """Test relative risk with small sample sizes."""
        table = np.array([[2, 3], [1, 4]], dtype=np.float64)
        rr_val = scirs2.relative_risk_py(table)

        # RR = (2/5)/(1/5) = 0.4/0.2 = 2.0
        assert abs(rr_val - 2.0) < 0.001

    def test_relative_risk_wrong_dimensions(self):
        """Test that non-2x2 table raises error."""
        table = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        with pytest.raises(RuntimeError):
            scirs2.relative_risk_py(table)

    def test_relative_risk_negative_values(self):
        """Test that negative values raise error."""
        table = np.array([[10, -5], [5, 10]], dtype=np.float64)

        with pytest.raises(RuntimeError):
            scirs2.relative_risk_py(table)


class TestContingencyTableComparisons:
    """Test comparisons between different measures."""

    def test_odds_ratio_vs_relative_risk(self):
        """Test relationship between odds ratio and relative risk."""
        # For rare diseases, OR ≈ RR
        # Rare disease (low incidence)
        table = np.array([[5, 95], [2, 98]], dtype=np.float64)

        or_val = scirs2.odds_ratio_py(table)
        rr_val = scirs2.relative_risk_py(table)

        # OR and RR should be similar for rare diseases
        # OR = (5*98)/(95*2) = 490/190 ≈ 2.58
        # RR = (5/100)/(2/100) = 0.05/0.02 = 2.5
        assert abs(or_val - rr_val) < 0.5

    def test_common_disease_or_rr_divergence(self):
        """Test that OR and RR diverge for common diseases."""
        # Common disease (high incidence)
        table = np.array([[60, 40], [30, 70]], dtype=np.float64)

        or_val = scirs2.odds_ratio_py(table)
        rr_val = scirs2.relative_risk_py(table)

        # OR = (60*70)/(40*30) = 4200/1200 = 3.5
        # RR = (60/100)/(30/100) = 0.6/0.3 = 2.0
        # OR should be larger than RR for common outcomes
        assert or_val > rr_val
        assert abs(or_val - 3.5) < 0.001
        assert abs(rr_val - 2.0) < 0.001

    def test_fisher_vs_chi2_small_sample(self):
        """Test Fisher's exact vs chi-square for small samples."""
        # Small sample where Fisher's is more appropriate
        table = np.array([[3, 1], [1, 3]], dtype=np.float64)

        # Fisher's exact test
        fisher_result = scirs2.fisher_exact_py(table)

        # Should compute successfully
        assert "pvalue" in fisher_result
        assert 0 <= fisher_result["pvalue"] <= 1

    def test_consistency_of_measures(self):
        """Test consistency of all three measures."""
        table = np.array([[40, 10], [10, 40]], dtype=np.float64)

        fisher_result = scirs2.fisher_exact_py(table)
        or_val = scirs2.odds_ratio_py(table)
        rr_val = scirs2.relative_risk_py(table)

        # Fisher's test should use the same odds ratio
        assert abs(fisher_result["odds_ratio"] - or_val) < 0.001

        # OR > RR for this common outcome
        assert or_val > rr_val


class TestContingencyEdgeCases:
    """Test edge cases for contingency table analysis."""

    def test_all_zeros(self):
        """Test behavior with all zeros."""
        table = np.array([[0, 0], [0, 0]], dtype=np.float64)

        # Should raise error or handle gracefully
        try:
            fisher_result = scirs2.fisher_exact_py(table)
            # If it computes, odds ratio should be undefined (0 or NaN)
            assert fisher_result["odds_ratio"] == 0.0 or np.isnan(fisher_result["odds_ratio"])
        except RuntimeError:
            # Acceptable to raise error
            pass

    def test_single_nonzero(self):
        """Test with only one nonzero cell."""
        table = np.array([[10, 0], [0, 0]], dtype=np.float64)

        # Should handle or raise error
        try:
            fisher_result = scirs2.fisher_exact_py(table)
            # If it computes, check validity
            assert "pvalue" in fisher_result
        except RuntimeError:
            # Acceptable to raise error
            pass

    def test_very_large_values(self):
        """Test with very large values."""
        table = np.array([[1e6, 1e5], [1e5, 1e6]], dtype=np.float64)

        or_val = scirs2.odds_ratio_py(table)
        rr_val = scirs2.relative_risk_py(table)

        # Should compute valid results
        assert np.isfinite(or_val)
        assert np.isfinite(rr_val)

    def test_very_small_nonzero_values(self):
        """Test with very small non-zero values."""
        table = np.array([[0.001, 0.002], [0.002, 0.001]], dtype=np.float64)

        or_val = scirs2.odds_ratio_py(table)

        # Should compute valid results
        assert np.isfinite(or_val)
        assert or_val > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
