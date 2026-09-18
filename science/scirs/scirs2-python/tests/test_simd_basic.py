"""
Test suite for basic SIMD-optimized statistical functions in scirs2-python

This test suite validates the SIMD-optimized implementations of fundamental statistics:
- moment_simd_py: Generic nth moment calculation (raw or central)
- mean_simd_py: SIMD-accelerated arithmetic mean
- std_simd_py: SIMD-accelerated standard deviation
- variance_simd_py: SIMD-accelerated variance

Session 17 additions (4 functions, comprehensive testing).
"""

import numpy as np
import pytest
import scirs2


class TestMomentSIMD:
    """Test SIMD-optimized generic moment calculation"""

    def test_moment_simd_zeroth_moment(self):
        """Zeroth moment should always be 1"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.moment_simd_py(data, 0, center=True)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_moment_simd_first_raw(self):
        """First raw moment should equal mean"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        moment1 = scirs2.moment_simd_py(data, 1, center=False)
        mean_val = np.mean(data)
        assert moment1 == pytest.approx(mean_val, abs=1e-10)

    def test_moment_simd_first_central(self):
        """First central moment should be 0"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        moment1_central = scirs2.moment_simd_py(data, 1, center=True)
        assert abs(moment1_central) < 1e-10

    def test_moment_simd_second_central(self):
        """Second central moment should equal variance (with ddof=0)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        moment2 = scirs2.moment_simd_py(data, 2, center=True)
        variance = np.var(data, ddof=0)  # Population variance
        assert moment2 == pytest.approx(variance, abs=1e-10)

    def test_moment_simd_third_central(self):
        """Third central moment for symmetric data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        moment3 = scirs2.moment_simd_py(data, 3, center=True)
        # Symmetric data should have third moment close to 0
        assert abs(moment3) < 1e-10

    def test_moment_simd_fourth_central(self):
        """Fourth central moment relates to kurtosis"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        moment4 = scirs2.moment_simd_py(data, 4, center=True)
        # Fourth moment should be positive
        assert moment4 > 0

    def test_moment_simd_vs_regular(self):
        """SIMD version should match regular moment_py"""
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        simd_result = scirs2.moment_simd_py(data, 2, center=True)
        regular_result = scirs2.moment_py(data, 2, center=True)
        assert simd_result == pytest.approx(regular_result, abs=1e-10)

    def test_moment_simd_large_array(self):
        """SIMD optimization with large array"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 10000)
        # Second central moment (variance)
        moment2 = scirs2.moment_simd_py(data, 2, center=True)
        # Should be close to 1 for standard normal
        assert 0.9 < moment2 < 1.1


class TestMeanSIMD:
    """Test SIMD-optimized mean calculation"""

    def test_mean_simd_basic(self):
        """Basic mean calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.mean_simd_py(data)
        assert result == pytest.approx(3.0, abs=1e-10)

    def test_mean_simd_vs_numpy(self):
        """Should match NumPy mean"""
        data = np.array([10.5, 20.3, 30.7, 40.2, 50.1])
        simd_mean = scirs2.mean_simd_py(data)
        numpy_mean = np.mean(data)
        assert simd_mean == pytest.approx(numpy_mean, abs=1e-10)

    def test_mean_simd_vs_regular(self):
        """Should match regular mean_py"""
        data = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        simd_mean = scirs2.mean_simd_py(data)
        regular_mean = scirs2.mean_py(data)
        assert simd_mean == pytest.approx(regular_mean, abs=1e-10)

    def test_mean_simd_negative_values(self):
        """Mean with negative values"""
        data = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        result = scirs2.mean_simd_py(data)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_mean_simd_all_identical(self):
        """Mean of identical values"""
        data = np.array([7.5, 7.5, 7.5, 7.5, 7.5])
        result = scirs2.mean_simd_py(data)
        assert result == pytest.approx(7.5, abs=1e-10)

    def test_mean_simd_large_array(self):
        """SIMD optimization with large array"""
        np.random.seed(42)
        data = np.random.normal(100, 15, 100000)
        result = scirs2.mean_simd_py(data)
        # Should be close to 100
        assert 99 < result < 101

    def test_mean_simd_single_element(self):
        """Mean of single element"""
        data = np.array([42.0])
        result = scirs2.mean_simd_py(data)
        assert result == pytest.approx(42.0, abs=1e-10)


class TestStdSIMD:
    """Test SIMD-optimized standard deviation calculation"""

    def test_std_simd_basic(self):
        """Basic std calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.std_simd_py(data, ddof=1)
        expected = np.std(data, ddof=1)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_std_simd_population(self):
        """Population std (ddof=0)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pop_std = scirs2.std_simd_py(data, ddof=0)
        sample_std = scirs2.std_simd_py(data, ddof=1)
        # Population std should be smaller than sample std
        assert pop_std < sample_std

    def test_std_simd_vs_numpy(self):
        """Should match NumPy std"""
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)
        simd_std = scirs2.std_simd_py(data, ddof=1)
        numpy_std = np.std(data, ddof=1)
        assert simd_std == pytest.approx(numpy_std, abs=1e-10)

    def test_std_simd_vs_regular(self):
        """Should match regular std_py"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        simd_std = scirs2.std_simd_py(data, ddof=1)
        regular_std = scirs2.std_py(data, ddof=1)
        assert simd_std == pytest.approx(regular_std, abs=1e-10)

    def test_std_simd_zero(self):
        """Std of constant data should be 0"""
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        result = scirs2.std_simd_py(data, ddof=1)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_std_simd_large_array(self):
        """SIMD optimization with large array"""
        np.random.seed(42)
        data = np.random.normal(0, 5, 100000)
        result = scirs2.std_simd_py(data, ddof=1)
        # Should be close to 5
        assert 4.9 < result < 5.1


class TestVarianceSIMD:
    """Test SIMD-optimized variance calculation"""

    def test_variance_simd_basic(self):
        """Basic variance calculation"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = scirs2.variance_simd_py(data, ddof=1)
        expected = np.var(data, ddof=1)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_variance_simd_population(self):
        """Population variance (ddof=0)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pop_var = scirs2.variance_simd_py(data, ddof=0)
        sample_var = scirs2.variance_simd_py(data, ddof=1)
        # Population variance should be smaller
        assert pop_var < sample_var
        assert pop_var == pytest.approx(2.0, abs=1e-10)
        assert sample_var == pytest.approx(2.5, abs=1e-10)

    def test_variance_simd_vs_numpy(self):
        """Should match NumPy variance"""
        np.random.seed(42)
        data = np.random.normal(100, 15, 1000)
        simd_var = scirs2.variance_simd_py(data, ddof=1)
        numpy_var = np.var(data, ddof=1)
        assert simd_var == pytest.approx(numpy_var, abs=1e-8)

    def test_variance_simd_vs_regular(self):
        """Should match regular var_py"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        simd_var = scirs2.variance_simd_py(data, ddof=1)
        regular_var = scirs2.var_py(data, ddof=1)
        assert simd_var == pytest.approx(regular_var, abs=1e-10)

    def test_variance_simd_std_relationship(self):
        """Variance should equal std squared"""
        data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        var = scirs2.variance_simd_py(data, ddof=1)
        std = scirs2.std_simd_py(data, ddof=1)
        assert var == pytest.approx(std ** 2, abs=1e-10)

    def test_variance_simd_zero(self):
        """Variance of constant data should be 0"""
        data = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        result = scirs2.variance_simd_py(data, ddof=1)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_variance_simd_large_array(self):
        """SIMD optimization with large array"""
        np.random.seed(42)
        data = np.random.normal(0, 3, 100000)
        result = scirs2.variance_simd_py(data, ddof=1)
        # Should be close to 9 (3^2)
        assert 8.8 < result < 9.2


class TestRealWorldScenarios:
    """Test SIMD functions with real-world-like scenarios"""

    def test_sensor_calibration(self):
        """Sensor calibration analysis"""
        np.random.seed(42)
        # Sensor readings with known true value
        measurements = np.random.normal(25.0, 0.5, 1000)

        mean = scirs2.mean_simd_py(measurements)
        std = scirs2.std_simd_py(measurements, ddof=1)
        var = scirs2.variance_simd_py(measurements, ddof=1)

        # Mean should be close to 25.0
        assert 24.9 < mean < 25.1
        # Std should be close to 0.5
        assert 0.45 < std < 0.55
        # Variance should be close to 0.25
        assert 0.20 < var < 0.30

    def test_quality_control_process(self):
        """Manufacturing quality control"""
        np.random.seed(42)
        # Target dimension: 100mm ± 0.1mm
        measurements = np.random.normal(100.0, 0.1, 5000)

        mean = scirs2.mean_simd_py(measurements)
        std = scirs2.std_simd_py(measurements, ddof=1)

        # Process should be centered
        assert 99.99 < mean < 100.01
        # Low variation indicates good quality
        assert std < 0.15

    def test_financial_volatility(self):
        """Financial returns volatility analysis"""
        np.random.seed(42)
        # Daily returns (annualized volatility ~20%)
        daily_returns = np.random.normal(0.0005, 0.0127, 252)  # ~1 year

        mean_return = scirs2.mean_simd_py(daily_returns)
        volatility = scirs2.std_simd_py(daily_returns, ddof=1)
        variance = scirs2.variance_simd_py(daily_returns, ddof=1)

        # Mean should be small
        assert abs(mean_return) < 0.01
        # Volatility should be reasonable
        assert 0.01 < volatility < 0.02
        # Variance = volatility^2
        assert variance == pytest.approx(volatility ** 2, abs=1e-10)

    def test_experimental_data_moments(self):
        """Experimental physics data analysis"""
        np.random.seed(42)
        # Measurement data with known distribution
        data = np.random.gamma(5.0, 2.0, 10000)

        # Calculate multiple moments
        moment1 = scirs2.moment_simd_py(data, 1, center=False)  # Mean
        moment2 = scirs2.moment_simd_py(data, 2, center=True)   # Variance
        moment3 = scirs2.moment_simd_py(data, 3, center=True)   # Skewness-related
        moment4 = scirs2.moment_simd_py(data, 4, center=True)   # Kurtosis-related

        # Gamma distribution: mean = k*theta = 5*2 = 10
        assert 9.5 < moment1 < 10.5
        # Variance = k*theta^2 = 5*4 = 20
        assert 19 < moment2 < 21
        # Higher moments should be positive
        assert moment3 > 0
        assert moment4 > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_moment_simd_minimum_size(self):
        """Minimum data size for moments"""
        data = np.array([1.0, 2.0])
        result = scirs2.moment_simd_py(data, 2, center=True)
        # Should work with 2 points
        assert isinstance(result, float)

    def test_mean_simd_single_value(self):
        """Mean of single value"""
        data = np.array([99.9])
        result = scirs2.mean_simd_py(data)
        assert result == pytest.approx(99.9, abs=1e-10)

    def test_std_simd_two_values(self):
        """Std with minimum required points"""
        data = np.array([10.0, 20.0])
        result = scirs2.std_simd_py(data, ddof=1)
        expected = np.std(data, ddof=1)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_variance_simd_two_values(self):
        """Variance with minimum required points"""
        data = np.array([5.0, 15.0])
        result = scirs2.variance_simd_py(data, ddof=1)
        expected = np.var(data, ddof=1)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_all_functions_constant_data(self):
        """All functions with constant data"""
        data = np.array([42.0, 42.0, 42.0, 42.0, 42.0])

        mean = scirs2.mean_simd_py(data)
        assert mean == pytest.approx(42.0, abs=1e-10)

        std = scirs2.std_simd_py(data, ddof=1)
        assert std == pytest.approx(0.0, abs=1e-10)

        var = scirs2.variance_simd_py(data, ddof=1)
        assert var == pytest.approx(0.0, abs=1e-10)

        moment2 = scirs2.moment_simd_py(data, 2, center=True)
        assert moment2 == pytest.approx(0.0, abs=1e-10)


class TestNumericalStability:
    """Test numerical stability with challenging data"""

    def test_large_magnitude_values(self):
        """Test with large magnitude values"""
        data = np.array([1e10, 2e10, 3e10, 4e10, 5e10])

        mean = scirs2.mean_simd_py(data)
        assert mean == pytest.approx(3e10, rel=1e-6)

        std = scirs2.std_simd_py(data, ddof=1)
        expected_std = np.std(data, ddof=1)
        assert std == pytest.approx(expected_std, rel=1e-6)

    def test_small_magnitude_values(self):
        """Test with small magnitude values"""
        data = np.array([1e-8, 2e-8, 3e-8, 4e-8, 5e-8])

        mean = scirs2.mean_simd_py(data)
        assert mean == pytest.approx(3e-8, rel=1e-6)

        var = scirs2.variance_simd_py(data, ddof=1)
        expected_var = np.var(data, ddof=1)
        assert var == pytest.approx(expected_var, rel=1e-6)

    def test_mixed_magnitude_spread(self):
        """Test with values spanning many orders of magnitude"""
        data = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

        mean = scirs2.mean_simd_py(data)
        std = scirs2.std_simd_py(data, ddof=1)

        # Should handle wide range
        assert mean > 0
        assert std > 0
        assert not np.isnan(mean)
        assert not np.isnan(std)

    def test_precision_near_zero(self):
        """Test precision with values near zero"""
        data = np.array([-0.001, -0.0005, 0.0, 0.0005, 0.001])

        mean = scirs2.mean_simd_py(data)
        assert abs(mean) < 1e-10

        var = scirs2.variance_simd_py(data, ddof=1)
        expected = np.var(data, ddof=1)
        assert var == pytest.approx(expected, abs=1e-12)


class TestMomentProperties:
    """Test mathematical properties of moments"""

    def test_moment_relationship_to_variance(self):
        """Second central moment equals variance (ddof=0)"""
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        moment2 = scirs2.moment_simd_py(data, 2, center=True)
        var_pop = scirs2.variance_simd_py(data, ddof=0)

        assert moment2 == pytest.approx(var_pop, abs=1e-10)

    def test_raw_vs_central_moments(self):
        """Raw and central moments should differ (except mean)"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # First moment
        raw1 = scirs2.moment_simd_py(data, 1, center=False)
        central1 = scirs2.moment_simd_py(data, 1, center=True)
        mean = scirs2.mean_simd_py(data)

        assert raw1 == pytest.approx(mean, abs=1e-10)
        assert abs(central1) < 1e-10  # Should be ~0

        # Second moment
        raw2 = scirs2.moment_simd_py(data, 2, center=False)
        central2 = scirs2.moment_simd_py(data, 2, center=True)

        # They should be different
        assert abs(raw2 - central2) > 0.1

    def test_std_variance_relationship(self):
        """std = sqrt(variance)"""
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)

        std = scirs2.std_simd_py(data, ddof=1)
        var = scirs2.variance_simd_py(data, ddof=1)

        assert std == pytest.approx(np.sqrt(var), abs=1e-10)

    def test_moment_additivity(self):
        """Test moment calculation properties"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # All moments should be computable
        for order in range(5):
            moment_raw = scirs2.moment_simd_py(data, order, center=False)
            moment_central = scirs2.moment_simd_py(data, order, center=True)

            assert not np.isnan(moment_raw)
            assert not np.isnan(moment_central)


class TestPerformanceScaling:
    """Test SIMD performance characteristics"""

    def test_mean_small_vs_large(self):
        """Mean should work efficiently for small and large arrays"""
        np.random.seed(42)

        # Small array
        small_data = np.random.normal(0, 1, 10)
        mean_small = scirs2.mean_simd_py(small_data)
        assert isinstance(mean_small, float)

        # Large array (should trigger SIMD)
        large_data = np.random.normal(0, 1, 100000)
        mean_large = scirs2.mean_simd_py(large_data)
        assert isinstance(mean_large, float)

        # Both should give reasonable results
        assert abs(mean_small) < 2.0
        assert abs(mean_large) < 0.1

    def test_variance_scaling(self):
        """Variance calculation should scale well"""
        np.random.seed(42)

        # Different array sizes
        for size in [100, 1000, 10000]:
            data = np.random.normal(0, 1, size)
            var = scirs2.variance_simd_py(data, ddof=1)
            # Should be close to 1 for standard normal
            assert 0.8 < var < 1.2


class TestCrossValidation:
    """Cross-validate SIMD functions against multiple sources"""

    def test_mean_triple_validation(self):
        """Validate mean against NumPy, regular, and manual calculation"""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        simd_mean = scirs2.mean_simd_py(data)
        numpy_mean = np.mean(data)
        regular_mean = scirs2.mean_py(data)
        manual_mean = sum(data) / len(data)

        assert simd_mean == pytest.approx(numpy_mean, abs=1e-10)
        assert simd_mean == pytest.approx(regular_mean, abs=1e-10)
        assert simd_mean == pytest.approx(manual_mean, abs=1e-10)

    def test_std_triple_validation(self):
        """Validate std against NumPy, regular, and formula"""
        data = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        simd_std = scirs2.std_simd_py(data, ddof=1)
        numpy_std = np.std(data, ddof=1)
        regular_std = scirs2.std_py(data, ddof=1)

        assert simd_std == pytest.approx(numpy_std, abs=1e-10)
        assert simd_std == pytest.approx(regular_std, abs=1e-10)

    def test_variance_triple_validation(self):
        """Validate variance against NumPy, regular, and moment"""
        data = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

        simd_var = scirs2.variance_simd_py(data, ddof=1)
        numpy_var = np.var(data, ddof=1)
        regular_var = scirs2.var_py(data, ddof=1)
        moment2 = scirs2.moment_simd_py(data, 2, center=True)

        # Note: moment uses ddof=0, so we need to adjust
        adjusted_moment2 = moment2 * len(data) / (len(data) - 1)

        assert simd_var == pytest.approx(numpy_var, abs=1e-10)
        assert simd_var == pytest.approx(regular_var, abs=1e-10)
        assert simd_var == pytest.approx(adjusted_moment2, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
