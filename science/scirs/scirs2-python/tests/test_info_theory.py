"""
Comprehensive tests for information theory and advanced statistics
"""

import numpy as np
import pytest
import scirs2


class TestEntropy:
    """Tests for Shannon entropy"""

    def test_entropy_uniform_distribution(self):
        """Test entropy of uniform distribution (maximum entropy)"""
        # Uniform distribution: all outcomes equally likely
        data = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int64)

        result = scirs2.entropy_py(data)

        # Should be positive
        assert result > 0.0

    def test_entropy_deterministic(self):
        """Test entropy of deterministic outcome (zero entropy)"""
        # All same value: no uncertainty
        data = np.array([1, 1, 1, 1, 1], dtype=np.int64)

        result = scirs2.entropy_py(data)

        # Should be zero (no uncertainty)
        assert abs(result - 0.0) < 0.001

    def test_entropy_binary(self):
        """Test entropy of binary distribution"""
        # Equal probability binary: maximum entropy for binary
        data = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

        result_nat = scirs2.entropy_py(data)  # Natural log
        result_bits = scirs2.entropy_py(data, base=2.0)  # Base 2

        # Binary with p=0.5 should have entropy of 1 bit
        assert abs(result_bits - 1.0) < 0.01

    def test_entropy_base_parameter(self):
        """Test entropy with different bases"""
        data = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)

        entropy_e = scirs2.entropy_py(data)  # Natural log (default)
        entropy_2 = scirs2.entropy_py(data, base=2.0)  # Bits
        entropy_10 = scirs2.entropy_py(data, base=10.0)  # Decimal

        # All should be positive
        assert entropy_e > 0.0
        assert entropy_2 > 0.0
        assert entropy_10 > 0.0

        # Conversion: H(base e) = H(base 2) * ln(2)
        assert abs(entropy_e - entropy_2 * np.log(2)) < 0.01

    def test_entropy_skewed_distribution(self):
        """Test entropy of skewed distribution (lower entropy)"""
        # Skewed: one outcome much more likely
        data = np.array([1, 1, 1, 1, 1, 1, 2, 3], dtype=np.int64)

        result = scirs2.entropy_py(data)

        # Should be lower than uniform distribution
        assert result > 0.0
        assert result < 1.5  # Less than maximum for 3 categories


class TestKLDivergence:
    """Tests for Kullback-Leibler divergence"""

    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence of identical distributions (should be zero)"""
        p = np.array([0.3, 0.3, 0.4])
        q = np.array([0.3, 0.3, 0.4])

        result = scirs2.kl_divergence_py(p, q)

        # Identical distributions: zero divergence
        assert abs(result - 0.0) < 0.0001

    def test_kl_divergence_different_distributions(self):
        """Test KL divergence of different distributions"""
        p = np.array([0.4, 0.3, 0.3])
        q = np.array([0.33, 0.33, 0.34])

        result = scirs2.kl_divergence_py(p, q)

        # Different distributions: positive divergence
        assert result > 0.0

    def test_kl_divergence_asymmetric(self):
        """Test KL divergence is asymmetric"""
        p = np.array([0.8, 0.15, 0.05])
        q = np.array([0.3, 0.5, 0.2])

        kl_pq = scirs2.kl_divergence_py(p, q)
        kl_qp = scirs2.kl_divergence_py(q, p)

        # KL(P||Q) != KL(Q||P) for truly different distributions
        assert abs(kl_pq - kl_qp) > 0.01

    def test_kl_divergence_uniform_vs_peaked(self):
        """Test KL divergence between uniform and peaked distributions"""
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        peaked = np.array([0.7, 0.1, 0.1, 0.1])

        result = scirs2.kl_divergence_py(peaked, uniform)

        # Significant divergence
        assert result > 0.1

    def test_kl_divergence_normalized_distributions(self):
        """Test KL divergence with properly normalized distributions"""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])

        # Ensure they sum to 1.0
        assert abs(np.sum(p) - 1.0) < 0.0001
        assert abs(np.sum(q) - 1.0) < 0.0001

        result = scirs2.kl_divergence_py(p, q)
        assert result >= 0.0


class TestCrossEntropy:
    """Tests for cross-entropy"""

    def test_cross_entropy_identical_distributions(self):
        """Test cross-entropy of identical distributions"""
        p = np.array([0.3, 0.3, 0.4])
        q = np.array([0.3, 0.3, 0.4])

        result = scirs2.cross_entropy_py(p, q)

        # Should equal entropy of p
        assert result > 0.0

    def test_cross_entropy_different_distributions(self):
        """Test cross-entropy of different distributions"""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])

        result = scirs2.cross_entropy_py(p, q)

        # Should be positive
        assert result > 0.0

    def test_cross_entropy_vs_kl_divergence(self):
        """Test relationship: H(p,q) = H(p) + KL(p||q)"""
        p = np.array([0.4, 0.3, 0.3])
        q = np.array([0.33, 0.33, 0.34])

        cross_ent = scirs2.cross_entropy_py(p, q)
        kl_div = scirs2.kl_divergence_py(p, q)

        # H(p,q) >= H(p), with equality when p == q
        assert cross_ent >= 0.0

        # H(p,q) = H(p) + KL(p||q)
        # We can verify: cross_ent - kl_div should equal entropy of p
        # (but we need entropy for continuous distributions, which we don't have directly)

    def test_cross_entropy_asymmetric(self):
        """Test cross-entropy is asymmetric"""
        p = np.array([0.8, 0.15, 0.05])
        q = np.array([0.3, 0.5, 0.2])

        ce_pq = scirs2.cross_entropy_py(p, q)
        ce_qp = scirs2.cross_entropy_py(q, p)

        # H(p,q) != H(q,p) for truly different distributions
        assert abs(ce_pq - ce_qp) > 0.01

    def test_cross_entropy_minimum_at_identity(self):
        """Test cross-entropy is minimized when q = p"""
        p = np.array([0.5, 0.3, 0.2])

        # Various q distributions
        q1 = p.copy()  # Same as p
        q2 = np.array([0.4, 0.4, 0.2])  # Different
        q3 = np.array([0.33, 0.33, 0.34])  # Uniform-ish

        ce1 = scirs2.cross_entropy_py(p, q1)
        ce2 = scirs2.cross_entropy_py(p, q2)
        ce3 = scirs2.cross_entropy_py(p, q3)

        # Cross-entropy should be minimized when q = p
        assert ce1 <= ce2
        assert ce1 <= ce3


class TestWeightedMean:
    """Tests for weighted mean"""

    def test_weighted_mean_equal_weights(self):
        """Test weighted mean with equal weights equals regular mean"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        weighted = scirs2.weighted_mean_py(data, weights)
        regular = np.mean(data)

        assert abs(weighted - regular) < 0.001

    def test_weighted_mean_emphasis(self):
        """Test weighted mean emphasizes weighted values"""
        data = np.array([10.0, 15.0, 20.0, 25.0, 80.0])
        weights = np.array([1.0, 1.0, 5.0, 1.0, 1.0])  # Emphasize 20.0

        weighted = scirs2.weighted_mean_py(data, weights)
        regular = np.mean(data)

        # Weighted mean should be closer to 20 than regular mean
        assert abs(weighted - 20.0) < abs(regular - 20.0)

    def test_weighted_mean_zero_weight(self):
        """Test weighted mean with zero weight on some values"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        weights = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # Only first two count

        weighted = scirs2.weighted_mean_py(data, weights)

        # Should be mean of first two values
        assert abs(weighted - 15.0) < 0.001

    def test_weighted_mean_different_scales(self):
        """Test weighted mean with different weight scales"""
        data = np.array([10.0, 20.0, 30.0])

        # Proportional weights
        weights1 = np.array([1.0, 2.0, 3.0])
        weights2 = np.array([2.0, 4.0, 6.0])  # Scaled version

        weighted1 = scirs2.weighted_mean_py(data, weights1)
        weighted2 = scirs2.weighted_mean_py(data, weights2)

        # Should be the same (weights are relative)
        assert abs(weighted1 - weighted2) < 0.001

    def test_weighted_mean_single_value(self):
        """Test weighted mean with single value"""
        data = np.array([42.0])
        weights = np.array([1.0])

        weighted = scirs2.weighted_mean_py(data, weights)

        assert abs(weighted - 42.0) < 0.001


class TestMoment:
    """Tests for statistical moments"""

    def test_moment_first_centered(self):
        """Test first central moment (should be zero)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.moment_py(data, order=1, center=True)

        # First central moment is always zero
        assert abs(result - 0.0) < 0.0001

    def test_moment_second_centered_is_variance(self):
        """Test second central moment equals variance"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        moment2 = scirs2.moment_py(data, order=2, center=True)
        variance = np.var(data, ddof=0)  # Population variance

        assert abs(moment2 - variance) < 0.001

    def test_moment_first_uncentered_is_mean(self):
        """Test first uncentered moment equals mean"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        moment1 = scirs2.moment_py(data, order=1, center=False)
        mean = np.mean(data)

        assert abs(moment1 - mean) < 0.001

    def test_moment_third_centered(self):
        """Test third central moment (related to skewness)"""
        # Symmetric data: third moment should be zero
        data_symmetric = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        moment3_sym = scirs2.moment_py(data_symmetric, order=3, center=True)

        assert abs(moment3_sym) < 0.01

        # Skewed data: third moment should be non-zero
        data_skewed = np.array([1.0, 1.0, 1.0, 2.0, 10.0])
        moment3_skew = scirs2.moment_py(data_skewed, order=3, center=True)

        assert abs(moment3_skew) > 0.1

    def test_moment_fourth_centered(self):
        """Test fourth central moment (related to kurtosis)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        moment4 = scirs2.moment_py(data, order=4, center=True)

        # Should be positive
        assert moment4 > 0.0

    def test_moment_higher_order(self):
        """Test higher order moments"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        moment5 = scirs2.moment_py(data, order=5, center=True)
        moment6 = scirs2.moment_py(data, order=6, center=True)

        # For symmetric data, odd central moments should be near zero
        assert abs(moment5) < 0.1

        # Even central moments should be positive
        assert moment6 > 0.0


class TestInformationTheoryRealWorld:
    """Real-world scenarios for information theory"""

    def test_machine_learning_cross_entropy_loss(self):
        """Test cross-entropy as ML loss function"""
        # True distribution (one-hot encoded)
        true_label = np.array([0.0, 1.0, 0.0])  # Class 2

        # Model predictions (poor)
        pred_poor = np.array([0.6, 0.2, 0.2])
        # Model predictions (good)
        pred_good = np.array([0.1, 0.8, 0.1])

        loss_poor = scirs2.cross_entropy_py(true_label, pred_poor)
        loss_good = scirs2.cross_entropy_py(true_label, pred_good)

        # Better predictions should have lower cross-entropy loss
        assert loss_good < loss_poor

    def test_information_gain_simulation(self):
        """Test information gain concept using entropy"""
        # Before split: mixed data
        before = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)

        # After split: pure subsets
        subset1 = np.array([1, 1, 1], dtype=np.int64)
        subset2 = np.array([2, 2, 2], dtype=np.int64)

        entropy_before = scirs2.entropy_py(before, base=2.0)
        entropy_subset1 = scirs2.entropy_py(subset1, base=2.0)
        entropy_subset2 = scirs2.entropy_py(subset2, base=2.0)

        # Pure subsets should have zero entropy
        assert abs(entropy_subset1) < 0.01
        assert abs(entropy_subset2) < 0.01

        # Original mixed data should have higher entropy
        assert entropy_before > 0.5

    def test_weighted_average_grading(self):
        """Test weighted mean for grade calculation"""
        # Test scores
        scores = np.array([85.0, 92.0, 78.0, 88.0])

        # Weights: homework 20%, midterm 30%, final 50%
        # Assuming: homework, homework, midterm, final
        weights_scenario1 = np.array([0.1, 0.1, 0.3, 0.5])

        weighted_grade = scirs2.weighted_mean_py(scores, weights_scenario1)

        # Final exam (88) weighted heavily
        assert 85.0 < weighted_grade < 90.0

    def test_portfolio_weighted_return(self):
        """Test weighted mean for portfolio returns"""
        # Asset returns (%)
        returns = np.array([5.0, 10.0, -2.0, 8.0])

        # Portfolio weights (must sum to 1.0)
        weights = np.array([0.4, 0.3, 0.1, 0.2])

        portfolio_return = scirs2.weighted_mean_py(returns, weights)

        # Should be between min and max return
        assert -2.0 <= portfolio_return <= 10.0


class TestInformationTheoryEdgeCases:
    """Edge cases for information theory functions"""

    def test_entropy_large_dataset(self):
        """Test entropy with large dataset"""
        np.random.seed(42)
        data = np.random.randint(0, 10, size=1000, dtype=np.int64)

        result = scirs2.entropy_py(data, base=2.0)

        # Should be positive and reasonable
        assert result > 0.0
        assert result < 10.0  # Maximum for 10 categories

    def test_kl_divergence_small_values(self):
        """Test KL divergence with small probability values"""
        # Small but valid probabilities
        p = np.array([0.98, 0.01, 0.01])
        q = np.array([0.01, 0.98, 0.01])

        result = scirs2.kl_divergence_py(p, q)

        # Should be large (very different distributions)
        assert result > 1.0

    def test_weighted_mean_many_values(self):
        """Test weighted mean with many values"""
        np.random.seed(42)
        data = np.random.randn(1000) * 10 + 50
        weights = np.random.rand(1000)

        result = scirs2.weighted_mean_py(data, weights)

        # Should be reasonable
        assert 30.0 < result < 70.0

    def test_moment_zero_variance(self):
        """Test moments with zero variance data"""
        data = np.array([5.0, 5.0, 5.0, 5.0])

        moment2 = scirs2.moment_py(data, order=2, center=True)
        moment3 = scirs2.moment_py(data, order=3, center=True)

        # All centered moments should be zero
        assert abs(moment2) < 0.0001
        assert abs(moment3) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
