"""Tests for scirs2 special functions module."""

import pytest
import numpy as np
import scirs2


class TestGammaFunctions:
    """Test gamma and related functions."""

    def test_gamma(self):
        """Test gamma function."""
        # Γ(1) = 1
        assert abs(scirs2.gamma_py(1.0) - 1.0) < 1e-10
        # Γ(2) = 1
        assert abs(scirs2.gamma_py(2.0) - 1.0) < 1e-10
        # Γ(3) = 2
        assert abs(scirs2.gamma_py(3.0) - 2.0) < 1e-10
        # Γ(4) = 6
        assert abs(scirs2.gamma_py(4.0) - 6.0) < 1e-10
        # Γ(5) = 24
        assert abs(scirs2.gamma_py(5.0) - 24.0) < 1e-10

    def test_gamma_array(self):
        """Test vectorized gamma function."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([1.0, 1.0, 2.0, 6.0, 24.0])
        result = scirs2.gamma_array_py(x)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_lgamma(self):
        """Test log-gamma function."""
        # ln(Γ(5)) = ln(24)
        assert abs(scirs2.lgamma_py(5.0) - np.log(24.0)) < 1e-10

    def test_digamma(self):
        """Test digamma function."""
        # ψ(1) = -γ ≈ -0.5772...
        result = scirs2.digamma_py(1.0)
        assert abs(result + 0.5772156649) < 0.001

    def test_beta(self):
        """Test beta function."""
        # B(1, 1) = 1
        assert abs(scirs2.beta_py(1.0, 1.0) - 1.0) < 1e-10
        # B(2, 3) = Γ(2)Γ(3)/Γ(5) = 1*2/24 = 1/12
        assert abs(scirs2.beta_py(2.0, 3.0) - 1.0/12.0) < 1e-10


class TestBesselFunctions:
    """Test Bessel functions."""

    def test_j0(self):
        """Test J0 Bessel function."""
        # J0(0) = 1
        assert abs(scirs2.j0_py(0.0) - 1.0) < 1e-10
        # J0(1) ≈ 0.7652
        assert abs(scirs2.j0_py(1.0) - 0.7652) < 0.001

    def test_j1(self):
        """Test J1 Bessel function."""
        # J1(0) = 0
        assert abs(scirs2.j1_py(0.0)) < 1e-10
        # J1(1) ≈ 0.4401
        assert abs(scirs2.j1_py(1.0) - 0.4401) < 0.001

    def test_jn(self):
        """Test Jn Bessel function."""
        # Jn(n, 0) = 0 for n > 0
        assert abs(scirs2.jn_py(2, 0.0)) < 1e-10
        assert abs(scirs2.jn_py(3, 0.0)) < 1e-10

    def test_y0(self):
        """Test Y0 Bessel function."""
        # Y0(1) ≈ 0.0883
        assert abs(scirs2.y0_py(1.0) - 0.0883) < 0.001

    def test_y1(self):
        """Test Y1 Bessel function."""
        # Y1(1) ≈ -0.7812
        assert abs(scirs2.y1_py(1.0) + 0.7812) < 0.001

    def test_yn(self):
        """Test Yn Bessel function."""
        result = scirs2.yn_py(2, 1.0)
        assert isinstance(result, float)

    def test_i0(self):
        """Test modified Bessel function I0."""
        # I0(0) = 1
        assert abs(scirs2.i0_py(0.0) - 1.0) < 1e-10
        # I0(1) ≈ 1.266
        assert abs(scirs2.i0_py(1.0) - 1.266) < 0.001

    def test_i1(self):
        """Test modified Bessel function I1."""
        # I1(0) = 0
        assert abs(scirs2.i1_py(0.0)) < 1e-10
        # I1(1) ≈ 0.565
        assert abs(scirs2.i1_py(1.0) - 0.565) < 0.001

    def test_k0(self):
        """Test modified Bessel function K0."""
        # K0(1) ≈ 0.4611
        assert abs(scirs2.k0_py(1.0) - 0.4611) < 0.001

    def test_k1(self):
        """Test modified Bessel function K1."""
        # K1(1) ≈ 0.5187
        assert abs(scirs2.k1_py(1.0) - 0.5187) < 0.001

    def test_bessel_array(self):
        """Test vectorized Bessel function."""
        x = np.array([0.0, 1.0, 2.0])
        result = scirs2.j0_array_py(x)
        assert len(result) == 3
        assert abs(result[0] - 1.0) < 1e-10


class TestErrorFunctions:
    """Test error functions."""

    def test_erf(self):
        """Test error function."""
        # erf(0) = 0
        assert abs(scirs2.erf_py(0.0)) < 1e-10
        # erf(∞) → 1, erf(3) ≈ 0.9999779
        assert abs(scirs2.erf_py(3.0) - 0.9999779) < 0.000001

    def test_erfc(self):
        """Test complementary error function."""
        # erfc(0) = 1
        assert abs(scirs2.erfc_py(0.0) - 1.0) < 1e-10
        # erfc(x) = 1 - erf(x)
        x = 1.5
        assert abs(scirs2.erfc_py(x) - (1.0 - scirs2.erf_py(x))) < 1e-10

    def test_erfinv(self):
        """Test inverse error function."""
        # erfinv(erf(x)) = x
        x = 0.5
        assert abs(scirs2.erfinv_py(scirs2.erf_py(x)) - x) < 0.001

    def test_erfcinv(self):
        """Test inverse complementary error function."""
        # erfcinv(erfc(x)) = x
        x = 0.5
        assert abs(scirs2.erfcinv_py(scirs2.erfc_py(x)) - x) < 0.001

    def test_erfcx(self):
        """Test scaled complementary error function."""
        # erfcx(x) = exp(x²) * erfc(x)
        x = 1.0
        expected = np.exp(x**2) * scirs2.erfc_py(x)
        assert abs(scirs2.erfcx_py(x) - expected) < 0.001

    def test_erfi(self):
        """Test imaginary error function."""
        # erfi(0) = 0
        assert abs(scirs2.erfi_py(0.0)) < 1e-10

    def test_dawsn(self):
        """Test Dawson's integral."""
        # Dawson(0) = 0
        assert abs(scirs2.dawsn_py(0.0)) < 1e-10

    def test_erf_array(self):
        """Test vectorized error function."""
        x = np.array([0.0, 0.5, 1.0])
        result = scirs2.erf_array_py(x)
        assert len(result) == 3
        assert abs(result[0]) < 1e-10


class TestCombinatorial:
    """Test combinatorial functions."""

    def test_factorial(self):
        """Test factorial function."""
        assert scirs2.factorial_py(0) == 1.0
        assert scirs2.factorial_py(1) == 1.0
        assert scirs2.factorial_py(2) == 2.0
        assert scirs2.factorial_py(3) == 6.0
        assert scirs2.factorial_py(4) == 24.0
        assert scirs2.factorial_py(5) == 120.0

    def test_comb(self):
        """Test binomial coefficient."""
        # C(n, 0) = 1
        assert scirs2.comb_py(5, 0) == 1.0
        # C(n, 1) = n
        assert scirs2.comb_py(5, 1) == 5.0
        # C(5, 2) = 10
        assert scirs2.comb_py(5, 2) == 10.0
        # C(5, 3) = 10
        assert scirs2.comb_py(5, 3) == 10.0
        # C(n, n) = 1
        assert scirs2.comb_py(5, 5) == 1.0

    def test_perm(self):
        """Test permutations."""
        # P(n, 0) = 1
        assert scirs2.perm_py(5, 0) == 1.0
        # P(n, 1) = n
        assert scirs2.perm_py(5, 1) == 5.0
        # P(5, 2) = 20
        assert scirs2.perm_py(5, 2) == 20.0
        # P(5, 3) = 60
        assert scirs2.perm_py(5, 3) == 60.0
        # P(n, n) = n!
        assert scirs2.perm_py(5, 5) == 120.0


class TestEllipticIntegrals:
    """Test elliptic integral functions."""

    def test_ellipk(self):
        """Test complete elliptic integral of the first kind."""
        # K(0) = π/2
        result = scirs2.ellipk_py(0.0)
        assert abs(result - np.pi/2) < 0.001

    def test_ellipe(self):
        """Test complete elliptic integral of the second kind."""
        # E(0) = π/2
        result = scirs2.ellipe_py(0.0)
        assert abs(result - np.pi/2) < 0.001

    def test_ellipkinc(self):
        """Test incomplete elliptic integral of the first kind."""
        # F(π/2, 0) = π/2
        result = scirs2.ellipkinc_py(np.pi/2, 0.0)
        assert abs(result - np.pi/2) < 0.001

    def test_ellipeinc(self):
        """Test incomplete elliptic integral of the second kind."""
        # E(π/2, 0) = π/2
        result = scirs2.ellipeinc_py(np.pi/2, 0.0)
        assert abs(result - np.pi/2) < 0.001
