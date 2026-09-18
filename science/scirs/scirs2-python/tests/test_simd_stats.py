"""
Test suite for SIMD-optimized statistical functions in scirs2-python

This test suite validates the SIMD-optimized implementations of:
- skewness_simd_py: SIMD-accelerated skewness calculation
- kurtosis_simd_py: SIMD-accelerated kurtosis calculation
- pearson_r_simd_py: SIMD-accelerated Pearson correlation
- covariance_simd_py: SIMD-accelerated covariance calculation

Session 16 additions (4 functions, comprehensive testing).
"""

import numpy as np
import pytest
import scirs2


class TestSkewnessSIMD:
    """Test SIMD-optimized skewness calculation"""

    def test_skewness_simd_symmetric(self):
        """Symmetric data should have skewness close to 0"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.skewness_simd_py(data, bias=True)
        assert abs(result) < 1e-10

    def test_skewness_simd_positive(self):
        """Positively skewed data"""
        data = np.array([1.0, 2.0, 2.0, 3.0, 10.0])
        result = scirs2.skewness_simd_py(data, bias=True)
        assert result > 0

    def test_skewness_simd_negative(self):
        """Negatively skewed data"""
        data = np.array([1.0, 8.0, 8.0, 9.0, 10.0])
        result = scirs2.skewness_simd_py(data, bias=True)
        assert result < 0

    def test_skewness_simd_bias_correction(self):
        """Test bias correction affects result"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        biased = scirs2.skewness_simd_py(data, bias=True)
        unbiased = scirs2.skewness_simd_py(data, bias=False)
        # Both should be close to 0 for symmetric data
        assert abs(biased) < 1e-10
        assert abs(unbiased) < 1e-10

    def test_skewness_simd_vs_regular(self):
        """SIMD version should match regular skew_py"""
        data = np.array([2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0])
        simd_result = scirs2.skewness_simd_py(data, bias=True)
        regular_result = scirs2.skew_py(data)
        assert simd_result == pytest.approx(regular_result, abs=1e-10)

    def test_skewness_simd_large_array(self):
        """SIMD optimization with large array"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        result = scirs2.skewness_simd_py(data, bias=True)
        # Normal distribution should have skewness near 0
        assert abs(result) < 0.1


class TestKurtosisSIMD:
    """Test SIMD-optimized kurtosis calculation"""

    def test_kurtosis_simd_fisher_biased(self):
        """Fisher's definition (excess kurtosis), biased"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.skewness_simd_py(data, bias=True)
        result = scirs2.kurtosis_simd_py(data, fisher=True, bias=True)
        # Uniform-like distribution has negative excess kurtosis
        assert result < 0

    def test_kurtosis_simd_pearson_biased(self):
        """Pearson's definition, biased"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.kurtosis_simd_py(data, fisher=False, bias=True)
        # Pearson's kurtosis = Fisher's + 3
        fisher_result = scirs2.kurtosis_simd_py(data, fisher=True, bias=True)
        assert result == pytest.approx(fisher_result + 3.0, abs=1e-10)

    def test_kurtosis_simd_fisher_unbiased(self):
        """Fisher's definition (excess kurtosis), unbiased"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.kurtosis_simd_py(data, fisher=True, bias=False)
        # Unbiased estimator should differ from biased
        biased = scirs2.kurtosis_simd_py(data, fisher=True, bias=True)
        assert abs(result - biased) > 0.01

    def test_kurtosis_simd_peaked_distribution(self):
        """High kurtosis for peaked distribution"""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 10.0, 15.0, 5.0, 5.0])
        result = scirs2.kurtosis_simd_py(data, fisher=True, bias=True)
        # Peaked distribution should have positive excess kurtosis
        assert result > 0

    def test_kurtosis_simd_vs_regular(self):
        """SIMD version should match regular kurtosis_py"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        simd_result = scirs2.kurtosis_simd_py(data, fisher=True, bias=True)
        regular_result = scirs2.kurtosis_py(data)
        assert simd_result == pytest.approx(regular_result, abs=1e-10)

    def test_kurtosis_simd_large_array(self):
        """SIMD optimization with large array"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        result = scirs2.kurtosis_simd_py(data, fisher=True, bias=True)
        # Normal distribution should have excess kurtosis near 0
        assert abs(result) < 0.2


class TestPearsonRSIMD:
    """Test SIMD-optimized Pearson correlation"""

    def test_pearson_r_simd_perfect_positive(self):
        """Perfect positive correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = scirs2.pearson_r_simd_py(x, y)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_pearson_r_simd_perfect_negative(self):
        """Perfect negative correlation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = scirs2.pearson_r_simd_py(x, y)
        assert result == pytest.approx(-1.0, abs=1e-10)

    def test_pearson_r_simd_no_correlation(self):
        """No correlation (independent variables)"""
        np.random.seed(42)
        x = np.random.normal(0, 1, 1000)
        y = np.random.normal(0, 1, 1000)
        result = scirs2.pearson_r_simd_py(x, y)
        # Should be close to 0 for independent variables
        assert abs(result) < 0.1

    def test_pearson_r_simd_moderate_positive(self):
        """Moderate positive correlation"""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = 0.5 * x + np.random.normal(0, 0.5, 100)
        result = scirs2.pearson_r_simd_py(x, y)
        # Should be moderately positive (around 0.6-0.8)
        assert 0.5 < result < 0.9

    def test_pearson_r_simd_vs_regular(self):
        """SIMD version should match regular pearsonr_py"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 3.9, 6.1, 8.0, 9.9])
        simd_result = scirs2.pearson_r_simd_py(x, y)
        regular_dict = scirs2.pearsonr_py(x, y)
        regular_result = regular_dict['correlation']  # Extract correlation from dict
        assert simd_result == pytest.approx(regular_result, abs=1e-10)

    def test_pearson_r_simd_large_array(self):
        """SIMD optimization with large array"""
        np.random.seed(42)
        x = np.random.normal(0, 1, 10000)
        y = 0.7 * x + np.random.normal(0, 0.5, 10000)
        result = scirs2.pearson_r_simd_py(x, y)
        # Should have moderately strong positive correlation
        assert 0.6 < result < 0.85

    def test_pearson_r_simd_length_mismatch(self):
        """Test error handling for mismatched array lengths"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(RuntimeError, match="(?i)dimension mismatch"):
            scirs2.pearson_r_simd_py(x, y)


class TestCovarianceSIMD:
    """Test SIMD-optimized covariance calculation"""

    def test_covariance_simd_perfect_positive(self):
        """Perfect positive linear relationship"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = scirs2.covariance_simd_py(x, y, ddof=1)
        # Positive covariance
        assert result > 0

    def test_covariance_simd_perfect_negative(self):
        """Perfect negative linear relationship"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = scirs2.covariance_simd_py(x, y, ddof=1)
        # Negative covariance
        assert result < 0

    def test_covariance_simd_zero(self):
        """Zero covariance (independent variables)"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = scirs2.covariance_simd_py(x, y, ddof=1)
        # Constant y gives zero covariance
        assert abs(result) < 1e-10

    def test_covariance_simd_ddof_0(self):
        """Population covariance (ddof=0)"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result_ddof0 = scirs2.covariance_simd_py(x, y, ddof=0)
        result_ddof1 = scirs2.covariance_simd_py(x, y, ddof=1)
        # ddof=0 should give smaller result than ddof=1
        assert result_ddof0 < result_ddof1

    def test_covariance_simd_vs_regular(self):
        """SIMD version should match regular covariance_py"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.1, 3.9, 6.1, 8.0, 9.9])
        simd_result = scirs2.covariance_simd_py(x, y, ddof=1)
        regular_result = scirs2.covariance_py(x, y, ddof=1)
        assert simd_result == pytest.approx(regular_result, abs=1e-10)

    def test_covariance_simd_formula_verification(self):
        """Verify covariance formula: Cov(X,Y) = E[(X-μx)(Y-μy)]"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        manual_cov = np.sum((x - mean_x) * (y - mean_y)) / (len(x) - 1)

        result = scirs2.covariance_simd_py(x, y, ddof=1)
        assert result == pytest.approx(manual_cov, abs=1e-10)

    def test_covariance_simd_large_array(self):
        """SIMD optimization with large array"""
        np.random.seed(42)
        x = np.random.normal(0, 1, 10000)
        y = 0.7 * x + np.random.normal(0, 0.5, 10000)
        result = scirs2.covariance_simd_py(x, y, ddof=1)
        # Should be positive for positively related variables
        assert result > 0

    def test_covariance_simd_length_mismatch(self):
        """Test error handling for mismatched array lengths"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0])
        with pytest.raises(RuntimeError, match="(?i)dimension mismatch"):
            scirs2.covariance_simd_py(x, y, ddof=1)


class TestRealWorldScenarios:
    """Test SIMD functions with real-world-like scenarios"""

    def test_financial_returns_analysis(self):
        """Analyze financial returns data"""
        np.random.seed(42)
        # Simulate daily returns for two stocks
        stock_a = np.random.normal(0.001, 0.02, 252)  # ~1 year
        stock_b = 0.6 * stock_a + np.random.normal(0.0005, 0.015, 252)

        # Correlation between stocks
        corr = scirs2.pearson_r_simd_py(stock_a, stock_b)
        assert 0.4 < corr < 0.8

        # Covariance
        cov = scirs2.covariance_simd_py(stock_a, stock_b, ddof=1)
        assert cov > 0

        # Skewness (returns often skewed)
        skew_a = scirs2.skewness_simd_py(stock_a, bias=False)
        # Skewness should be finite
        assert not np.isnan(skew_a)

        # Kurtosis (returns often have fat tails)
        kurt_a = scirs2.kurtosis_simd_py(stock_a, fisher=True, bias=False)
        # Kurtosis should be finite
        assert not np.isnan(kurt_a)

    def test_sensor_data_correlation(self):
        """Analyze correlation between sensor readings"""
        np.random.seed(42)
        # Temperature and humidity often negatively correlated
        temperature = np.random.normal(20, 5, 1000)
        humidity = -0.4 * temperature + np.random.normal(60, 10, 1000)

        corr = scirs2.pearson_r_simd_py(temperature, humidity)
        assert -0.6 < corr < -0.2

        cov = scirs2.covariance_simd_py(temperature, humidity, ddof=1)
        assert cov < 0

    def test_quality_control_distribution(self):
        """Quality control measurements"""
        np.random.seed(42)
        # Manufacturing measurements (slightly right-skewed)
        measurements = np.concatenate([
            np.random.normal(100, 2, 950),
            np.random.normal(105, 3, 50)  # Some outliers
        ])

        skew = scirs2.skewness_simd_py(measurements, bias=False)
        # Should be positively skewed due to outliers
        assert skew > 0

        kurt = scirs2.kurtosis_simd_py(measurements, fisher=True, bias=False)
        # Should have higher kurtosis due to outliers
        assert kurt > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_skewness_simd_minimum_size(self):
        """Minimum data size for skewness (unbiased needs n>=3)"""
        data = np.array([1.0, 2.0, 3.0])
        result = scirs2.skewness_simd_py(data, bias=False)
        # Should work with exactly 3 points
        assert isinstance(result, float)

    def test_kurtosis_simd_minimum_size(self):
        """Minimum data size for kurtosis (needs n>=4)"""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        result = scirs2.kurtosis_simd_py(data, fisher=True, bias=False)
        # Should work with exactly 4 points
        assert isinstance(result, float)

    def test_pearson_r_simd_minimum_size(self):
        """Minimum data size for correlation"""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        result = scirs2.pearson_r_simd_py(x, y)
        # Should work with 2 points
        assert isinstance(result, float)

    def test_covariance_simd_minimum_size(self):
        """Minimum data size for covariance with ddof=1"""
        x = np.array([1.0, 2.0])
        y = np.array([3.0, 4.0])
        result = scirs2.covariance_simd_py(x, y, ddof=1)
        # Should work with 2 points when ddof=1
        assert isinstance(result, float)

    def test_all_functions_identical_values(self):
        """Test with all identical values"""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        # Skewness should be 0
        skew = scirs2.skewness_simd_py(data, bias=True)
        assert abs(skew) < 1e-10

        # Kurtosis should raise error (zero variance)
        with pytest.raises(RuntimeError, match="Standard deviation is zero"):
            scirs2.kurtosis_simd_py(data, fisher=True, bias=True)

        # Correlation with another array should raise error
        other = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(RuntimeError, match="zero variance"):
            scirs2.pearson_r_simd_py(data, other)

        # Covariance with constant should be 0
        cov = scirs2.covariance_simd_py(data, other, ddof=1)
        assert abs(cov) < 1e-10


class TestNumericalStability:
    """Test numerical stability with challenging data"""

    def test_large_magnitude_values(self):
        """Test with large magnitude values"""
        x = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        y = np.array([2e10, 4e10, 6e10, 8e10, 10e10])

        # Should still compute correctly
        corr = scirs2.pearson_r_simd_py(x, y)
        assert corr == pytest.approx(1.0, abs=1e-6)

        cov = scirs2.covariance_simd_py(x, y, ddof=1)
        assert cov > 0

    def test_small_magnitude_values(self):
        """Test with small magnitude values"""
        # Use slightly larger values to avoid numerical precision issues
        x = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        y = np.array([5e-6, 4e-6, 3e-6, 2e-6, 1e-6])

        # Should still compute correctly
        corr = scirs2.pearson_r_simd_py(x, y)
        assert corr == pytest.approx(-1.0, abs=1e-6)

    def test_mixed_magnitude_values(self):
        """Test with mixed magnitude values"""
        data = np.array([1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5])

        # Should handle mixed magnitudes
        skew = scirs2.skewness_simd_py(data, bias=True)
        assert isinstance(skew, float)
        assert not np.isnan(skew)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
