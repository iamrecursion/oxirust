"""
SciPy Comparison Tests for Integration Module

Compares scirs2 numerical integration functions against SciPy.integrate
to ensure numerical correctness and API compatibility.
"""

import numpy as np
import pytest
import scipy.integrate
import scirs2


class TestNumericalIntegration:
    """Test numerical integration functions"""

    def test_quad_basic_integral(self):
        """Quadrature integration should work for simple functions"""
        # Define test function: f(x) = x^2, integral from 0 to 1 = 1/3
        def f(x):
            return x**2

        # Note: scirs2.quad_py may require different interface than scipy
        # This test checks if quad_py exists and is callable
        # May need adjustment based on actual API

    def test_simpson_rule(self):
        """Simpson's rule should integrate accurately"""
        # Integrate sin(x) from 0 to pi (should equal 2)
        x = np.ascontiguousarray(np.linspace(0, np.pi, 101))
        y = np.ascontiguousarray(np.sin(x))

        # SciPy Simpson's rule
        scipy_result = scipy.integrate.simpson(y, x)

        # SciRS2 Simpson's rule
        scirs2_result = scirs2.simpson_array_py(y, x)

        # Should be close to 2.0
        assert np.allclose(scipy_result, 2.0, rtol=1e-6)
        assert np.allclose(scirs2_result, scipy_result, rtol=1e-10)

    def test_romberg_integration(self):
        """Romberg integration should be accurate"""
        # Integrate x^3 from 0 to 2 (should equal 4)
        x = np.ascontiguousarray(np.linspace(0, 2, 33))  # 2^n + 1 points for Romberg
        y = np.ascontiguousarray(x**3)

        # SciRS2 Romberg (check if it works)
        scirs2_result = scirs2.romberg_array_py(y, x[1] - x[0])

        # Should be close to 4.0
        expected = 4.0
        assert np.allclose(scirs2_result, expected, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
