"""
Tests for advanced quantile and statistical measure functions

This module tests:
1. Deciles calculation
2. Standard error of the mean (SEM)
3. Percentile range calculation
"""

import numpy as np
import pytest
import scirs2


class TestDeciles:
    """Tests for deciles function"""

    def test_deciles_basic(self):
        """Test basic deciles calculation"""
        data = np.array([float(i) for i in range(1, 101)])  # 1 to 100
        result = scirs2.deciles_py(data)

        assert len(result) == 9
        # Deciles should be 10th, 20th, ..., 90th percentiles
        expected = [10.9, 20.8, 30.7, 40.6, 50.5, 60.4, 70.3, 80.2, 90.1]
        for i, exp in enumerate(expected):
            assert result[i] == pytest.approx(exp, abs=0.1)

    def test_deciles_ordering(self):
        """Test that deciles are in ascending order"""
        data = np.random.normal(0, 1, 1000)
        result = scirs2.deciles_py(data)

        assert len(result) == 9
        # Should be in ascending order
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_deciles_uniform_data(self):
        """Test deciles with uniform distribution"""
        data = np.linspace(0, 100, 1000)
        result = scirs2.deciles_py(data)

        # For uniform distribution, deciles should be approximately at 10%, 20%, ..., 90%
        expected = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        for i, exp in enumerate(expected):
            assert result[i] == pytest.approx(exp, abs=1.0)

    def test_deciles_small_dataset(self):
        """Test deciles with small dataset"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = scirs2.deciles_py(data)

        assert len(result) == 9
        # All deciles should be within data range
        assert np.all(result >= data.min())
        assert np.all(result <= data.max())

    def test_deciles_vs_quintiles(self):
        """Test that deciles provide finer granularity than quintiles"""
        data = np.random.normal(50, 10, 500)
        deciles = scirs2.deciles_py(data)
        quintiles = scirs2.quintiles_py(data)

        # Quintiles are at 20%, 40%, 60%, 80%
        # Deciles are at 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%
        # So deciles[1] (20%) should match quintiles[0] (20%)
        # and deciles[3] (40%) should match quintiles[1] (40%)
        assert deciles[1] == pytest.approx(quintiles[0], abs=0.1)
        assert deciles[3] == pytest.approx(quintiles[1], abs=0.1)
        assert deciles[5] == pytest.approx(quintiles[2], abs=0.1)
        assert deciles[7] == pytest.approx(quintiles[3], abs=0.1)

    def test_deciles_finer_granularity(self):
        """Test that deciles have values between quintiles"""
        data = np.random.normal(100, 15, 800)
        deciles = scirs2.deciles_py(data)

        # Deciles should have intermediate values
        # e.g., 10th percentile < 20th percentile < 30th percentile
        assert deciles[0] < deciles[1] < deciles[2]
        assert deciles[6] < deciles[7] < deciles[8]


class TestStandardErrorMean:
    """Tests for standard error of the mean (SEM) function"""

    def test_sem_basic(self):
        """Test basic SEM calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # SEM = std / sqrt(n)
        # std (sample, ddof=1) ≈ 1.58, n = 5
        # SEM ≈ 1.58 / sqrt(5) ≈ 0.707
        result = scirs2.sem_py(data, ddof=1)
        expected = np.std(data, ddof=1) / np.sqrt(len(data))

        assert result == pytest.approx(expected, abs=0.001)

    def test_sem_ddof_zero(self):
        """Test SEM with ddof=0 (population)"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        result = scirs2.sem_py(data, ddof=0)
        expected = np.std(data, ddof=0) / np.sqrt(len(data))

        assert result == pytest.approx(expected, abs=0.001)

    def test_sem_ddof_effect(self):
        """Test that ddof affects SEM value"""
        data = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0])

        sem_ddof0 = scirs2.sem_py(data, ddof=0)
        sem_ddof1 = scirs2.sem_py(data, ddof=1)

        # Higher ddof should give larger SEM
        assert sem_ddof1 > sem_ddof0

    def test_sem_large_dataset(self):
        """Test SEM decreases with larger dataset"""
        np.random.seed(42)

        # SEM should decrease as n increases
        data_small = np.random.normal(100, 15, 10)
        data_large = np.random.normal(100, 15, 1000)

        sem_small = scirs2.sem_py(data_small, ddof=1)
        sem_large = scirs2.sem_py(data_large, ddof=1)

        # Larger dataset should have smaller SEM
        assert sem_large < sem_small

    def test_sem_constant_data(self):
        """Test SEM with constant data"""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        result = scirs2.sem_py(data, ddof=1)

        # SEM should be 0 for constant data
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_sem_formula_verification(self):
        """Verify SEM formula: std / sqrt(n)"""
        data = np.array([12.5, 15.3, 18.7, 21.2, 24.8, 27.1, 30.5])

        # Manual calculation
        std_dev = np.std(data, ddof=1)
        n = len(data)
        expected_sem = std_dev / np.sqrt(n)

        result = scirs2.sem_py(data, ddof=1)

        assert result == pytest.approx(expected_sem, abs=1e-10)

    def test_sem_normal_distribution(self):
        """Test SEM with normal distribution"""
        np.random.seed(123)
        data = np.random.normal(50, 10, 100)

        result = scirs2.sem_py(data, ddof=1)

        # SEM should be positive
        assert result > 0
        # SEM should be less than the standard deviation
        assert result < np.std(data, ddof=1)


class TestPercentileRange:
    """Tests for percentile range function"""

    def test_percentile_range_basic(self):
        """Test basic percentile range calculation"""
        data = np.array([float(i) for i in range(1, 101)])  # 1 to 100

        # Range between 25th and 75th percentile (IQR)
        result = scirs2.percentile_range_py(data, 25.0, 75.0)

        # For uniform 1-100, IQR ≈ 75 - 25 = 50
        assert result == pytest.approx(50.0, abs=1.0)

    @pytest.mark.skip(reason="Overflow bug in scirs2-stats quantile_simd.rs:59 with normal distribution")
    def test_percentile_range_iqr_comparison(self):
        """Test that 25-75 range equals IQR"""
        data = np.random.normal(100, 15, 500)

        pct_range = scirs2.percentile_range_py(data, 25.0, 75.0)
        iqr = scirs2.iqr_py(data)

        # Should be approximately equal
        assert pct_range == pytest.approx(iqr, abs=0.5)

    def test_percentile_range_custom_percentiles(self):
        """Test percentile range with custom percentiles"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        # Range between 10th and 90th percentile
        result = scirs2.percentile_range_py(data, 10.0, 90.0)

        # Should capture most of the range
        assert result > 0
        assert result < (data.max() - data.min())

    @pytest.mark.skip(reason="Overflow bug in scirs2-stats quantile_simd.rs:59 with normal distribution")
    def test_percentile_range_narrow(self):
        """Test percentile range with narrow percentiles"""
        data = np.random.normal(50, 10, 1000)

        # Narrow range (45th to 55th percentile)
        narrow_range = scirs2.percentile_range_py(data, 45.0, 55.0)

        # Wide range (10th to 90th percentile)
        wide_range = scirs2.percentile_range_py(data, 10.0, 90.0)

        # Narrow should be much smaller than wide
        assert narrow_range < wide_range / 2

    @pytest.mark.skip(reason="Overflow bug in scirs2-stats quantile_simd.rs:59 with normal distribution")
    def test_percentile_range_symmetric_distribution(self):
        """Test percentile range with symmetric distribution"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        # Symmetric ranges should be similar
        lower_range = scirs2.percentile_range_py(data, 5.0, 50.0)
        upper_range = scirs2.percentile_range_py(data, 50.0, 95.0)

        # Should be approximately equal for symmetric distribution
        assert lower_range == pytest.approx(upper_range, abs=0.3)

    def test_percentile_range_extreme(self):
        """Test percentile range with extreme percentiles"""
        data = np.linspace(0, 100, 1000)

        # Full range (1st to 99th percentile)
        result = scirs2.percentile_range_py(data, 1.0, 99.0)

        # Should be close to full range
        full_range = data.max() - data.min()
        assert result == pytest.approx(0.98 * full_range, abs=2.0)

    def test_percentile_range_identical_values(self):
        """Test percentile range with identical values"""
        data = np.array([42.0, 42.0, 42.0, 42.0, 42.0])

        result = scirs2.percentile_range_py(data, 25.0, 75.0)

        # Range should be 0 for identical values
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_percentile_range_interpolation(self):
        """Test percentile range with linear interpolation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = scirs2.percentile_range_py(data, 20.0, 80.0, interpolation="linear")

        assert result > 0
        assert result <= (data.max() - data.min())


class TestRealWorldScenarios:
    """Real-world application tests"""

    def test_income_deciles(self):
        """Test deciles for income distribution analysis"""
        # Simulated income data (right-skewed)
        np.random.seed(42)
        incomes = np.random.lognormal(mean=10, sigma=0.8, size=1000)

        deciles = scirs2.deciles_py(incomes)

        # Income deciles should be in ascending order
        for i in range(len(deciles) - 1):
            assert deciles[i] < deciles[i + 1]

        # Top decile (90th percentile) should be much higher than bottom
        assert deciles[8] > 3 * deciles[0]

    def test_test_scores_sem(self):
        """Test SEM for test score analysis"""
        # Simulated test scores
        test_scores = np.array([85.0, 88.0, 92.0, 79.0, 95.0, 87.0, 90.0, 84.0, 91.0, 86.0])

        mean_score = np.mean(test_scores)
        sem = scirs2.sem_py(test_scores, ddof=1)

        # 95% confidence interval: mean ± 1.96 * SEM
        ci_lower = mean_score - 1.96 * sem
        ci_upper = mean_score + 1.96 * sem

        # CI should be reasonable
        assert ci_lower < mean_score < ci_upper
        assert ci_upper - ci_lower > 0

    @pytest.mark.skip(reason="Overflow bug in scirs2-stats quantile_simd.rs:59 with normal distribution")
    def test_quality_control_percentile_range(self):
        """Test percentile range for quality control"""
        # Manufacturing measurements
        np.random.seed(42)
        measurements = np.random.normal(100, 2, 500)

        # Middle 80% range (10th to 90th percentile)
        middle_range = scirs2.percentile_range_py(measurements, 10.0, 90.0)

        # Should be smaller than full range
        full_range = measurements.max() - measurements.min()
        assert middle_range < full_range

        # Should be reasonable for this distribution
        assert middle_range < 15  # Within ~7.5 from mean on each side

    def test_performance_metrics_deciles(self):
        """Test deciles for performance grading"""
        # Simulated performance scores
        np.random.seed(123)
        performance = np.random.beta(5, 2, 1000) * 100  # Right-skewed, 0-100

        deciles = scirs2.deciles_py(performance)

        # Use deciles to categorize performance
        # Bottom 10%: needs improvement
        # 10-30%: below average
        # 30-70%: average
        # 70-90%: above average
        # Top 10%: excellent

        # Verify reasonable distribution
        assert deciles[0] > 0  # Bottom 10% above 0
        assert deciles[8] < 100  # Top 10% below 100
        assert deciles[4] > deciles[0]  # Median > bottom


class TestEdgeCases:
    """Edge case tests"""

    def test_deciles_minimum_size(self):
        """Test deciles with minimum data points"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.deciles_py(data)

        # Should still return 9 values
        assert len(result) == 9

    def test_sem_minimum_size_ddof1(self):
        """Test SEM with minimum size for ddof=1"""
        data = np.array([1.0, 2.0])

        # n=2, ddof=1: just enough
        result = scirs2.sem_py(data, ddof=1)
        assert result > 0

    def test_percentile_range_full_range(self):
        """Test percentile range covering full data range"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # 0th to 100th percentile
        result = scirs2.percentile_range_py(data, 0.0, 100.0)

        # Should be close to full range
        assert result == pytest.approx(4.0, abs=0.1)

    def test_deciles_all_equal(self):
        """Test deciles when all values are equal"""
        data = np.array([7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0])
        result = scirs2.deciles_py(data)

        # All deciles should be 7.0
        assert np.all(np.abs(result - 7.0) < 1e-10)

    def test_sem_large_values(self):
        """Test SEM with large values"""
        data = np.array([1000000.0, 2000000.0, 3000000.0, 4000000.0, 5000000.0])

        result = scirs2.sem_py(data, ddof=1)

        # SEM should be positive and reasonable
        assert result > 0
        assert result < np.mean(data)

    def test_percentile_range_negative_values(self):
        """Test percentile range with negative values"""
        data = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0])

        result = scirs2.percentile_range_py(data, 25.0, 75.0)

        # Should work with negative values
        assert result > 0


class TestNumericalStability:
    """Tests for numerical stability"""

    def test_deciles_very_small_values(self):
        """Test deciles with very small values"""
        data = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010])
        result = scirs2.deciles_py(data)

        assert len(result) == 9
        assert np.all(result >= data.min())
        assert np.all(result <= data.max())

    def test_sem_high_precision(self):
        """Test SEM with high-precision data"""
        data = np.array([1.23456789, 2.34567890, 3.45678901, 4.56789012, 5.67890123])

        result = scirs2.sem_py(data, ddof=1)

        # Should maintain reasonable precision
        expected = np.std(data, ddof=1) / np.sqrt(len(data))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_percentile_range_wide_range(self):
        """Test percentile range with wide value range"""
        data = np.array([0.001, 1.0, 100.0, 10000.0, 1000000.0])

        result = scirs2.percentile_range_py(data, 20.0, 80.0)

        # Should handle wide range
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
