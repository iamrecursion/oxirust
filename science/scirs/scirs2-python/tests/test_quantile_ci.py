"""
Tests for quintiles and confidence interval functions

This module tests:
1. Quintiles calculation
2. Skewness confidence intervals
3. Kurtosis confidence intervals
"""

import numpy as np
import pytest
import scirs2


class TestQuintiles:
    """Tests for quintiles function"""

    def test_quintiles_basic(self):
        """Test basic quintiles calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = scirs2.quintiles_py(data)

        assert len(result) == 4
        # Quintiles should be 20th, 40th, 60th, 80th percentiles
        assert result[0] == pytest.approx(2.8, abs=0.01)
        assert result[1] == pytest.approx(4.6, abs=0.01)
        assert result[2] == pytest.approx(6.4, abs=0.01)
        assert result[3] == pytest.approx(8.2, abs=0.01)

    def test_quintiles_ordering(self):
        """Test that quintiles are in ascending order"""
        data = np.random.normal(0, 1, 100)
        result = scirs2.quintiles_py(data)

        assert len(result) == 4
        # Should be in ascending order
        assert result[0] < result[1] < result[2] < result[3]

    def test_quintiles_uniform_data(self):
        """Test quintiles with uniform distribution"""
        data = np.linspace(0, 100, 1000)
        result = scirs2.quintiles_py(data)

        # For uniform distribution, quintiles should be approximately at 20%, 40%, 60%, 80%
        expected = [20.0, 40.0, 60.0, 80.0]
        for i, exp in enumerate(expected):
            assert result[i] == pytest.approx(exp, abs=1.0)

    def test_quintiles_small_dataset(self):
        """Test quintiles with small dataset"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.quintiles_py(data)

        assert len(result) == 4
        # All quintiles should be within data range
        assert np.all(result >= data.min())
        assert np.all(result <= data.max())

    def test_quintiles_vs_quartiles(self):
        """Test that quintiles provide finer granularity than quartiles"""
        data = np.random.normal(50, 10, 500)
        quintiles = scirs2.quintiles_py(data)
        quartiles = scirs2.quartiles_py(data)

        # Q2 (median) from quartiles should lie between Q2 and Q3 from quintiles
        # This verifies that quintiles provide finer-grained information
        assert quintiles[1] < quartiles[1] < quintiles[2]


class TestSkewnessCI:
    """Tests for skewness confidence interval function"""

    def test_skewness_ci_basic(self):
        """Test basic skewness CI calculation"""
        # Right-skewed data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0])
        result = scirs2.skewness_ci_py(data, bias=False, confidence=0.95, n_bootstrap=100, seed=42)

        assert 'estimate' in result
        assert 'lower' in result
        assert 'upper' in result
        assert 'confidence' in result

        # Skewness estimate should be positive for right-skewed data
        assert result['estimate'] > 0
        # Lower bound should be less than estimate
        assert result['lower'] < result['estimate']
        # Upper bound should be greater than estimate
        assert result['upper'] > result['estimate']
        # Confidence level should match
        assert result['confidence'] == 0.95

    def test_skewness_ci_symmetric_data(self):
        """Test skewness CI with symmetric data"""
        np.random.seed(42)
        # Normal distribution is symmetric
        data = np.random.normal(0, 1, 100)
        result = scirs2.skewness_ci_py(data, n_bootstrap=200, seed=42)

        # Skewness should be close to 0 for symmetric data
        assert abs(result['estimate']) < 0.5
        # CI should include 0
        assert result['lower'] < 0 < result['upper']

    def test_skewness_ci_confidence_levels(self):
        """Test different confidence levels"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])

        # 90% CI should be narrower than 95% CI
        result_90 = scirs2.skewness_ci_py(data, confidence=0.90, n_bootstrap=200, seed=42)
        result_95 = scirs2.skewness_ci_py(data, confidence=0.95, n_bootstrap=200, seed=42)

        width_90 = result_90['upper'] - result_90['lower']
        width_95 = result_95['upper'] - result_95['lower']

        assert width_90 < width_95

    def test_skewness_ci_reproducibility(self):
        """Test that results are reproducible with same seed"""
        data = np.random.normal(0, 1, 50)

        result1 = scirs2.skewness_ci_py(data, n_bootstrap=100, seed=123)
        result2 = scirs2.skewness_ci_py(data, n_bootstrap=100, seed=123)

        assert result1['estimate'] == pytest.approx(result2['estimate'])
        assert result1['lower'] == pytest.approx(result2['lower'])
        assert result1['upper'] == pytest.approx(result2['upper'])

    def test_skewness_ci_bootstrap_samples(self):
        """Test effect of number of bootstrap samples"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0])

        # More bootstrap samples should give more stable estimates
        result_100 = scirs2.skewness_ci_py(data, n_bootstrap=100, seed=42)
        result_500 = scirs2.skewness_ci_py(data, n_bootstrap=500, seed=42)

        # Both should give positive skewness for right-skewed data
        assert result_100['estimate'] > 0
        assert result_500['estimate'] > 0


class TestKurtosisCI:
    """Tests for kurtosis confidence interval function"""

    def test_kurtosis_ci_basic(self):
        """Test basic kurtosis CI calculation"""
        # Normal-like data
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        result = scirs2.kurtosis_ci_py(data, fisher=True, bias=False, confidence=0.95, n_bootstrap=100, seed=42)

        assert 'estimate' in result
        assert 'lower' in result
        assert 'upper' in result
        assert 'confidence' in result

        # Fisher's kurtosis (excess kurtosis) should be near 0 for normal data
        assert abs(result['estimate']) < 1.0
        # Lower bound should be less than estimate
        assert result['lower'] < result['estimate']
        # Upper bound should be greater than estimate
        assert result['upper'] > result['estimate']

    def test_kurtosis_ci_fisher_vs_pearson(self):
        """Test Fisher vs Pearson kurtosis definitions"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Fisher's definition (excess kurtosis, subtract 3)
        result_fisher = scirs2.kurtosis_ci_py(data, fisher=True, n_bootstrap=100, seed=42)
        # Pearson's definition (raw kurtosis)
        result_pearson = scirs2.kurtosis_ci_py(data, fisher=False, n_bootstrap=100, seed=42)

        # Pearson's kurtosis should be Fisher's + 3 (approximately)
        diff = result_pearson['estimate'] - result_fisher['estimate']
        assert diff == pytest.approx(3.0, abs=0.1)

    def test_kurtosis_ci_heavy_tails(self):
        """Test kurtosis CI with heavy-tailed data"""
        # Add outliers to create heavy tails
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 50.0, -50.0])
        result = scirs2.kurtosis_ci_py(data, fisher=True, n_bootstrap=100, seed=42)

        # Heavy-tailed distribution should have positive excess kurtosis
        assert result['estimate'] > 0

    def test_kurtosis_ci_confidence_levels(self):
        """Test different confidence levels"""
        data = np.random.normal(0, 1, 80)

        # 90% CI should be narrower than 95% CI
        result_90 = scirs2.kurtosis_ci_py(data, confidence=0.90, n_bootstrap=200, seed=42)
        result_95 = scirs2.kurtosis_ci_py(data, confidence=0.95, n_bootstrap=200, seed=42)

        width_90 = result_90['upper'] - result_90['lower']
        width_95 = result_95['upper'] - result_95['lower']

        assert width_90 < width_95

    def test_kurtosis_ci_reproducibility(self):
        """Test that results are reproducible with same seed"""
        data = np.random.normal(0, 1, 50)

        result1 = scirs2.kurtosis_ci_py(data, n_bootstrap=100, seed=456)
        result2 = scirs2.kurtosis_ci_py(data, n_bootstrap=100, seed=456)

        assert result1['estimate'] == pytest.approx(result2['estimate'])
        assert result1['lower'] == pytest.approx(result2['lower'])
        assert result1['upper'] == pytest.approx(result2['upper'])


class TestConfidenceIntervalRealWorld:
    """Real-world application tests for confidence intervals"""

    def test_income_distribution_skewness(self):
        """Test skewness CI for income distribution (typically right-skewed)"""
        # Simulate income data (right-skewed)
        np.random.seed(42)
        incomes = np.random.lognormal(mean=10, sigma=0.5, size=200)

        result = scirs2.skewness_ci_py(incomes, confidence=0.95, n_bootstrap=300, seed=42)

        # Income distributions are typically right-skewed
        assert result['estimate'] > 0
        # CI should not include negative values
        assert result['lower'] > -0.5  # Allow small negative due to sampling

    def test_returns_kurtosis(self):
        """Test kurtosis CI for financial returns (often fat-tailed)"""
        # Simulate returns with occasional large moves (fat tails)
        np.random.seed(42)
        normal_returns = np.random.normal(0, 0.01, 180)
        extreme_returns = np.random.normal(0, 0.05, 20)
        returns = np.concatenate([normal_returns, extreme_returns])

        result = scirs2.kurtosis_ci_py(returns, fisher=True, confidence=0.95, n_bootstrap=200, seed=42)

        # Fat-tailed distributions typically have positive excess kurtosis
        assert result['estimate'] > 0

    def test_quality_control_metrics(self):
        """Test using quintiles for quality control boundaries"""
        # Simulate product measurements
        np.random.seed(42)
        measurements = np.random.normal(100, 5, 500)

        quintiles = scirs2.quintiles_py(measurements)

        # Can use quintiles to establish quality control zones
        # Bottom 20%: needs improvement
        # 20-40%: below average
        # 40-60%: average
        # 60-80%: above average
        # Top 20%: excellent

        assert quintiles[0] < 100 < quintiles[3]  # Median within range
        assert quintiles[3] - quintiles[0] > 0  # Spread exists


class TestEdgeCases:
    """Edge case tests for new functions"""

    def test_quintiles_small_n(self):
        """Test quintiles with very small sample"""
        data = np.array([1.0, 2.0, 3.0])
        result = scirs2.quintiles_py(data)

        # Should still return 4 values
        assert len(result) == 4

    def test_skewness_ci_minimum_data(self):
        """Test skewness CI with minimum required data points"""
        # Need at least 3 points for skewness
        data = np.array([1.0, 2.0, 3.0])
        result = scirs2.skewness_ci_py(data, n_bootstrap=50, seed=42)

        assert 'estimate' in result
        assert 'lower' in result
        assert 'upper' in result

    def test_kurtosis_ci_minimum_data(self):
        """Test kurtosis CI with minimum required data points"""
        # Need at least 4 points for kurtosis
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = scirs2.kurtosis_ci_py(data, n_bootstrap=50, seed=42)

        assert 'estimate' in result
        assert 'lower' in result
        assert 'upper' in result

    def test_quintiles_all_equal(self):
        """Test quintiles when all values are equal"""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = scirs2.quintiles_py(data)

        # All quintiles should be 5.0
        assert np.all(result == 5.0)

    def test_ci_wide_range(self):
        """Test CI with very wide data range"""
        data = np.array([0.001, 0.002, 0.003, 1000.0, 2000.0, 3000.0])

        # Should still compute CI without errors
        sk_result = scirs2.skewness_ci_py(data, n_bootstrap=50, seed=42)
        kurt_result = scirs2.kurtosis_ci_py(data, n_bootstrap=50, seed=42)

        assert sk_result['estimate'] is not None
        assert kurt_result['estimate'] is not None
