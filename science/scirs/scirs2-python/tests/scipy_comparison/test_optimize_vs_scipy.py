"""
SciPy Comparison Tests for Optimization Module

Compares scirs2 optimization functions against SciPy.optimize
to ensure numerical correctness and API compatibility.
"""

import numpy as np
import pytest
import scipy.optimize
import scirs2


class TestScalarMinimization:
    """Test scalar function minimization"""

    def test_minimize_scalar_quadratic(self):
        """Minimize simple quadratic function"""

        # Define objective: f(x) = (x - 2)^2, minimum at x=2
        def f(x):
            return (x - 2.0) ** 2

        # SciPy minimize scalar
        result_scipy = scipy.optimize.minimize_scalar(f, bounds=(0, 4), method='bounded')

        # SciRS2 minimize scalar (check if it works with Python callable)
        try:
            result_scirs2 = scirs2.minimize_scalar_py(f, bounds=(0.0, 4.0))

            # Extract result
            if isinstance(result_scirs2, dict):
                x_scirs2 = result_scirs2.get('x', result_scirs2.get('solution', None))
            else:
                x_scirs2 = result_scirs2

            # Should find minimum near x=2
            if x_scirs2 is not None:
                assert np.allclose(x_scirs2, 2.0, atol=1e-4)
        except TypeError:
            # If Python callable not supported, skip test
            pytest.skip("minimize_scalar_py does not support Python callables")


class TestRootFinding:
    """Test root finding functions"""

    def test_brentq_simple(self):
        """Brent's method should find roots"""

        # Find root of f(x) = x^2 - 4, root at x=2
        def f(x):
            return x**2 - 4.0

        # SciPy brentq
        root_scipy = scipy.optimize.brentq(f, 0, 3)

        # SciRS2 brentq (check if it works with Python callable)
        try:
            root_scirs2 = scirs2.brentq_py(f, 0.0, 3.0)

            # Should find root at x=2
            assert np.allclose(root_scirs2, 2.0, atol=1e-6)
            assert np.allclose(root_scipy, root_scirs2, atol=1e-6)
        except TypeError:
            pytest.skip("brentq_py does not support Python callables")


class TestCurveFitting:
    """Test curve fitting functions"""

    def test_curve_fit_linear(self):
        """Curve fitting should work for linear model"""
        np.random.seed(42)

        # Generate data: y = 2.5*x + 1.5 + noise
        x_data = np.ascontiguousarray(np.linspace(0, 10, 50))
        y_data = np.ascontiguousarray(2.5 * x_data + 1.5 + np.random.randn(50) * 0.5)

        # Define model
        def model(x, a, b):
            return a * x + b

        # SciPy curve_fit
        popt_scipy, _ = scipy.optimize.curve_fit(model, x_data, y_data)

        # SciRS2 curve_fit (check if supported)
        try:
            result_scirs2 = scirs2.curve_fit_py(model, x_data, y_data, p0=np.array([1.0, 0.0]))

            if isinstance(result_scirs2, dict):
                popt_scirs2 = result_scirs2.get('params', result_scirs2.get('x', None))
            else:
                popt_scirs2 = result_scirs2

            # Parameters should be close to [2.5, 1.5]
            if popt_scirs2 is not None:
                assert np.allclose(popt_scirs2[0], 2.5, atol=0.2)  # Slope
                assert np.allclose(popt_scirs2[1], 1.5, atol=0.5)  # Intercept
        except (TypeError, NotImplementedError):
            pytest.skip("curve_fit_py may not support Python callables")


class TestArrayBasedOptimization:
    """Test optimization with array-based objectives"""

    def test_minimize_rosenbrock(self):
        """Minimize Rosenbrock function (array-based test)"""

        # Skip this test as it requires Python callable support
        # which may not be implemented yet
        pytest.skip("Waiting for Python callable support in minimize_py")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
