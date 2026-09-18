"""Tests for additional statistical distributions."""

import pytest
import numpy as np
import scirs2


class TestLognormalDistribution:
    """Test Lognormal distribution."""

    def test_lognorm_creation(self):
        """Test creating lognormal distribution."""
        dist = scirs2.lognorm()
        assert dist is not None

        dist_custom = scirs2.lognorm(mu=0.5, sigma=0.8, loc=0.0)
        assert dist_custom is not None

    def test_lognorm_pdf(self):
        """Test lognormal PDF."""
        dist = scirs2.lognorm()  # mu=0, sigma=1

        # PDF at 0 should be 0 (lognormal is defined for x > 0)
        assert abs(dist.pdf(0.0)) < 1e-10

        # PDF should be positive for x > 0
        assert dist.pdf(1.0) > 0
        assert dist.pdf(2.0) > 0

    def test_lognorm_cdf(self):
        """Test lognormal CDF."""
        dist = scirs2.lognorm()

        # CDF at 0 should be 0
        assert abs(dist.cdf(0.0)) < 1e-10

        # CDF should be monotonically increasing
        assert dist.cdf(0.5) < dist.cdf(1.0) < dist.cdf(2.0)

    def test_lognorm_ppf(self):
        """Test lognormal PPF."""
        dist = scirs2.lognorm()

        # PPF at 0.5 should give median
        median = dist.ppf(0.5)
        assert abs(median - 1.0) < 0.05  # For mu=0, median = exp(0) = 1

        # PPF should be increasing
        assert dist.ppf(0.1) < dist.ppf(0.5) < dist.ppf(0.9)

    def test_lognorm_rvs(self):
        """Test lognormal random variates."""
        dist = scirs2.lognorm()

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # All samples should be positive
        assert all(s > 0 for s in samples)

        # For mu=0, sigma=1, mean should be around exp(0.5) ≈ 1.65
        mean = np.mean(samples)
        assert 1.0 < mean < 3.0  # Reasonable range given variance


class TestWeibullDistribution:
    """Test Weibull distribution."""

    def test_weibull_creation(self):
        """Test creating Weibull distribution."""
        dist = scirs2.weibull_min(2.0)
        assert dist is not None

        dist_custom = scirs2.weibull_min(shape=1.5, scale=2.0, loc=0.0)
        assert dist_custom is not None

    def test_weibull_pdf(self):
        """Test Weibull PDF."""
        dist = scirs2.weibull_min(2.0, scale=1.0)

        # PDF at 0 should be 0 for shape > 1
        assert abs(dist.pdf(0.0)) < 1e-10

        # PDF should be positive for x > 0
        assert dist.pdf(1.0) > 0
        assert dist.pdf(2.0) > 0

    def test_weibull_cdf(self):
        """Test Weibull CDF."""
        dist = scirs2.weibull_min(2.0)

        # CDF at 0 should be 0
        assert abs(dist.cdf(0.0)) < 1e-10

        # CDF should be monotonically increasing
        assert dist.cdf(0.5) < dist.cdf(1.0) < dist.cdf(2.0)

    def test_weibull_ppf(self):
        """Test Weibull PPF."""
        dist = scirs2.weibull_min(2.0)

        # PPF at 0 should be 0
        assert abs(dist.ppf(0.0)) < 1e-3

        # PPF should be increasing
        assert dist.ppf(0.1) < dist.ppf(0.5) < dist.ppf(0.9)

    def test_weibull_rvs(self):
        """Test Weibull random variates."""
        dist = scirs2.weibull_min(2.0, scale=1.0)

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # All samples should be non-negative
        assert all(s >= 0 for s in samples)


class TestLaplaceDistribution:
    """Test Laplace distribution."""

    def test_laplace_creation(self):
        """Test creating Laplace distribution."""
        dist = scirs2.laplace()
        assert dist is not None

        dist_custom = scirs2.laplace(loc=2.0, scale=3.0)
        assert dist_custom is not None

    def test_laplace_pdf(self):
        """Test Laplace PDF."""
        dist = scirs2.laplace()  # loc=0, scale=1

        # PDF should be symmetric around loc
        assert abs(dist.pdf(1.0) - dist.pdf(-1.0)) < 1e-6

        # PDF at 0 should be 0.5 for standard Laplace
        pdf_at_zero = dist.pdf(0.0)
        assert abs(pdf_at_zero - 0.5) < 1e-6

    def test_laplace_cdf(self):
        """Test Laplace CDF."""
        dist = scirs2.laplace()

        # CDF at 0 should be 0.5 (symmetric)
        cdf_at_zero = dist.cdf(0.0)
        assert abs(cdf_at_zero - 0.5) < 1e-6

        # CDF should be monotonically increasing
        assert dist.cdf(-2.0) < dist.cdf(0.0) < dist.cdf(2.0)

    def test_laplace_ppf(self):
        """Test Laplace PPF."""
        dist = scirs2.laplace()

        # PPF at 0.5 should be 0 (median)
        median = dist.ppf(0.5)
        assert abs(median) < 1e-6

        # PPF should be increasing
        assert dist.ppf(0.1) < dist.ppf(0.5) < dist.ppf(0.9)

    def test_laplace_rvs(self):
        """Test Laplace random variates."""
        dist = scirs2.laplace()

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # Mean should be close to 0
        mean = np.mean(samples)
        assert abs(mean) < 0.2


class TestLogisticDistribution:
    """Test Logistic distribution."""

    def test_logistic_creation(self):
        """Test creating logistic distribution."""
        dist = scirs2.logistic()
        assert dist is not None

        dist_custom = scirs2.logistic(loc=1.0, scale=2.0)
        assert dist_custom is not None

    def test_logistic_pdf(self):
        """Test logistic PDF."""
        dist = scirs2.logistic()  # loc=0, scale=1

        # PDF should be symmetric around loc
        assert abs(dist.pdf(1.0) - dist.pdf(-1.0)) < 1e-6

        # PDF at 0 should be 0.25 for standard logistic
        pdf_at_zero = dist.pdf(0.0)
        assert abs(pdf_at_zero - 0.25) < 1e-6

    def test_logistic_cdf(self):
        """Test logistic CDF."""
        dist = scirs2.logistic()

        # CDF at 0 should be 0.5 (symmetric)
        cdf_at_zero = dist.cdf(0.0)
        assert abs(cdf_at_zero - 0.5) < 1e-6

        # CDF should be monotonically increasing
        assert dist.cdf(-2.0) < dist.cdf(0.0) < dist.cdf(2.0)

    def test_logistic_ppf(self):
        """Test logistic PPF."""
        dist = scirs2.logistic()

        # PPF at 0.5 should be 0 (median)
        median = dist.ppf(0.5)
        assert abs(median) < 1e-6

        # PPF should be increasing
        assert dist.ppf(0.1) < dist.ppf(0.5) < dist.ppf(0.9)

    def test_logistic_rvs(self):
        """Test logistic random variates."""
        dist = scirs2.logistic()

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # Mean should be close to 0
        mean = np.mean(samples)
        assert abs(mean) < 0.2


class TestParetoDistribution:
    """Test Pareto distribution."""

    def test_pareto_creation(self):
        """Test creating Pareto distribution."""
        dist = scirs2.pareto(3.0)
        assert dist is not None

        dist_custom = scirs2.pareto(shape=2.0, scale=2.0, loc=0.0)
        assert dist_custom is not None

    def test_pareto_pdf(self):
        """Test Pareto PDF."""
        dist = scirs2.pareto(3.0, scale=1.0)

        # PDF should be 0 for x <= scale+loc
        assert abs(dist.pdf(0.5)) < 1e-10
        assert abs(dist.pdf(1.0)) < 1e-10  # At boundary

        # PDF should be positive for x > scale+loc
        assert dist.pdf(1.1) > 0
        assert dist.pdf(2.0) > 0

    def test_pareto_cdf(self):
        """Test Pareto CDF."""
        dist = scirs2.pareto(3.0, scale=1.0)

        # CDF at scale should be 0
        assert abs(dist.cdf(1.0)) < 1e-10

        # CDF should be monotonically increasing
        assert dist.cdf(1.0) <= dist.cdf(2.0) < dist.cdf(3.0)

    def test_pareto_ppf(self):
        """Test Pareto PPF."""
        dist = scirs2.pareto(3.0, scale=1.0)

        # PPF at 0 should give scale
        assert abs(dist.ppf(0.0) - 1.0) < 1e-3

        # PPF should be increasing
        assert dist.ppf(0.1) < dist.ppf(0.5) < dist.ppf(0.9)

    def test_pareto_rvs(self):
        """Test Pareto random variates."""
        dist = scirs2.pareto(3.0, scale=1.0)

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # All samples should be >= scale
        assert all(s >= 1.0 for s in samples)


class TestGeometricDistribution:
    """Test Geometric distribution."""

    def test_geom_creation(self):
        """Test creating geometric distribution."""
        dist = scirs2.geom(0.3)
        assert dist is not None

        dist_custom = scirs2.geom(0.5)
        assert dist_custom is not None

    def test_geom_pmf(self):
        """Test geometric PMF."""
        dist = scirs2.geom(0.5)

        # PMF at k=0 should be p
        pmf_0 = dist.pmf(0.0)
        assert abs(pmf_0 - 0.5) < 1e-6

        # PMF should decrease for larger k
        assert dist.pmf(0.0) > dist.pmf(1.0) > dist.pmf(2.0)

    def test_geom_cdf(self):
        """Test geometric CDF."""
        dist = scirs2.geom(0.5)

        # CDF should be monotonically increasing
        assert dist.cdf(0.0) < dist.cdf(1.0) < dist.cdf(2.0)

        # CDF at large k should approach 1
        assert dist.cdf(20.0) > 0.999

    def test_geom_ppf(self):
        """Test geometric PPF."""
        dist = scirs2.geom(0.5)

        # PPF at 0.5 should give median
        median = dist.ppf(0.5)
        assert 0.0 <= median <= 2.0

    def test_geom_rvs(self):
        """Test geometric random variates."""
        dist = scirs2.geom(0.3)

        samples = dist.rvs(1000)
        assert len(samples) == 1000

        # All samples should be non-negative integers
        assert all(s >= 0 for s in samples)

        # Mean should be close to (1-p)/p = 0.7/0.3 ≈ 2.33
        mean = np.mean(samples)
        assert 1.5 < mean < 3.5


class TestDistributionConsistency:
    """Test consistency across additional distributions."""

    def test_lognorm_cdf_ppf_inverse(self):
        """Test that PPF is the inverse of CDF for Lognormal."""
        dist = scirs2.lognorm()

        x = 1.5
        cdf_val = dist.cdf(x)
        ppf_val = dist.ppf(cdf_val)

        assert abs(ppf_val - x) < 1e-2

    def test_weibull_cdf_ppf_inverse(self):
        """Test that PPF is the inverse of CDF for Weibull."""
        dist = scirs2.weibull_min(2.0)

        x = 1.0
        cdf_val = dist.cdf(x)
        ppf_val = dist.ppf(cdf_val)

        assert abs(ppf_val - x) < 1e-2

    def test_laplace_cdf_ppf_inverse(self):
        """Test that PPF is the inverse of CDF for Laplace."""
        dist = scirs2.laplace()

        x = 1.0
        cdf_val = dist.cdf(x)
        ppf_val = dist.ppf(cdf_val)

        assert abs(ppf_val - x) < 1e-2

    def test_logistic_cdf_ppf_inverse(self):
        """Test that PPF is the inverse of CDF for Logistic."""
        dist = scirs2.logistic()

        x = 1.0
        cdf_val = dist.cdf(x)
        ppf_val = dist.ppf(cdf_val)

        assert abs(ppf_val - x) < 1e-2

    def test_pareto_cdf_ppf_inverse(self):
        """Test that PPF is the inverse of CDF for Pareto."""
        dist = scirs2.pareto(3.0)

        x = 2.0
        cdf_val = dist.cdf(x)
        ppf_val = dist.ppf(cdf_val)

        assert abs(ppf_val - x) < 1e-2

    def test_geometric_cdf_ppf_inverse(self):
        """Test that PPF is the inverse of CDF for Geometric."""
        dist = scirs2.geom(0.3)

        k = 2.0
        cdf_val = dist.cdf(k)
        ppf_val = dist.ppf(cdf_val)

        # For discrete distributions, ppf(cdf(k)) should give k or k-1
        assert abs(ppf_val - k) <= 1.0
