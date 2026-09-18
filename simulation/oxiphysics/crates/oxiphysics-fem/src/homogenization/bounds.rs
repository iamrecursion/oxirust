// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Phase definitions and classical homogenization bounds
//! (Voigt, Reuss, Hill, Hashin-Shtrikman, Mori-Tanaka).

use super::matrix_utils::*;

// ---------------------------------------------------------------------------
// Phase
// ---------------------------------------------------------------------------

/// A material phase in a Representative Volume Element (RVE).
#[derive(Debug, Clone)]
pub struct Phase {
    /// Human-readable name of the phase.
    pub name: String,
    /// Volume fraction of this phase (0 to 1).
    pub volume_fraction: f64,
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Poisson's ratio (dimensionless).
    pub poisson_ratio: f64,
    /// Mass density (kg/m³). Defaults to 0.0 when not set.
    pub density: f64,
}

impl Phase {
    /// Create a new `Phase`.
    pub fn new(
        name: impl Into<String>,
        volume_fraction: f64,
        youngs_modulus: f64,
        poisson_ratio: f64,
    ) -> Self {
        Self {
            name: name.into(),
            volume_fraction,
            youngs_modulus,
            poisson_ratio,
            density: 0.0,
        }
    }

    /// Builder-style setter for mass density (kg/m³).
    pub fn with_density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Bulk modulus K = E / (3(1 - 2ν)).
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))
    }

    /// Shear modulus G = E / (2(1 + ν)).
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))
    }

    /// Lamé first parameter λ = Eν / ((1+ν)(1-2ν)).
    pub fn lame_lambda(&self) -> f64 {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;
        e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// Isotropic 6×6 stiffness matrix in Voigt notation (σ = C ε).
    ///
    /// Ordering: \[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy\].
    pub fn stiffness_voigt(&self) -> [[f64; 6]; 6] {
        let lam = self.lame_lambda();
        let mu = self.shear_modulus();
        let c11 = lam + 2.0 * mu;
        let c12 = lam;
        let c44 = mu;
        let mut c = [[0.0f64; 6]; 6];
        // Normal-normal block
        c[0][0] = c11;
        c[0][1] = c12;
        c[0][2] = c12;
        c[1][0] = c12;
        c[1][1] = c11;
        c[1][2] = c12;
        c[2][0] = c12;
        c[2][1] = c12;
        c[2][2] = c11;
        // Shear block
        c[3][3] = c44;
        c[4][4] = c44;
        c[5][5] = c44;
        c
    }

    /// Isotropic 6×6 compliance matrix (ε = S σ).
    pub fn compliance_voigt(&self) -> [[f64; 6]; 6] {
        let e = self.youngs_modulus;
        let nu = self.poisson_ratio;
        let s11 = 1.0 / e;
        let s12 = -nu / e;
        let s44 = 2.0 * (1.0 + nu) / e; // 1/G
        let mut s = [[0.0f64; 6]; 6];
        s[0][0] = s11;
        s[0][1] = s12;
        s[0][2] = s12;
        s[1][0] = s12;
        s[1][1] = s11;
        s[1][2] = s12;
        s[2][0] = s12;
        s[2][1] = s12;
        s[2][2] = s11;
        s[3][3] = s44;
        s[4][4] = s44;
        s[5][5] = s44;
        s
    }
}

// ---------------------------------------------------------------------------
// Voigt-Reuss-Hill bounds
// ---------------------------------------------------------------------------

/// Voigt upper bound: C_V = Σ V_i C_i (uniform strain assumption).
pub fn voigt_average(phases: &[Phase]) -> [[f64; 6]; 6] {
    let mut cv = [[0.0f64; 6]; 6];
    for ph in phases {
        let ci = ph.stiffness_voigt();
        for i in 0..6 {
            for j in 0..6 {
                cv[i][j] += ph.volume_fraction * ci[i][j];
            }
        }
    }
    cv
}

/// Reuss lower bound: S_R = Σ V_i S_i, C_R = S_R^{-1} (uniform stress assumption).
pub fn reuss_average(phases: &[Phase]) -> [[f64; 6]; 6] {
    let mut sr = [[0.0f64; 6]; 6];
    for ph in phases {
        let ci = ph.stiffness_voigt();
        if let Some(si) = mat6_inv(&ci) {
            for i in 0..6 {
                for j in 0..6 {
                    sr[i][j] += ph.volume_fraction * si[i][j];
                }
            }
        }
    }
    mat6_inv(&sr).unwrap_or([[0.0; 6]; 6])
}

/// Voigt-Reuss-Hill average: C_H = 0.5 (C_V + C_R).
pub fn hill_average(phases: &[Phase]) -> [[f64; 6]; 6] {
    let cv = voigt_average(phases);
    let cr = reuss_average(phases);
    mat6_scale(&mat6_add(&cv, &cr), 0.5)
}

/// Effective Young's modulus from a stiffness tensor (via compliance S = C^{-1}).
///
/// E_eff = 1 / S\[0\]\[0\].
pub fn effective_youngs_modulus(c: &[[f64; 6]; 6]) -> f64 {
    if let Some(s) = mat6_inv(c) {
        1.0 / s[0][0]
    } else {
        0.0
    }
}

/// Effective Poisson's ratio from a stiffness tensor.
///
/// ν = -S\[0\]\[1\] / S\[0\]\[0\].
pub fn effective_poisson_ratio(c: &[[f64; 6]; 6]) -> f64 {
    if let Some(s) = mat6_inv(c) {
        -s[0][1] / s[0][0]
    } else {
        0.0
    }
}

/// Effective shear modulus from a stiffness tensor.
///
/// G_eff = 1 / S\[3\]\[3\].
pub fn effective_shear_modulus(c: &[[f64; 6]; 6]) -> f64 {
    if let Some(s) = mat6_inv(c) {
        1.0 / s[3][3]
    } else {
        0.0
    }
}

/// Effective bulk modulus from a stiffness tensor.
///
/// K_eff = E_eff / (3 * (1 - 2*nu_eff)).
pub fn effective_bulk_modulus(c: &[[f64; 6]; 6]) -> f64 {
    let e = effective_youngs_modulus(c);
    let nu = effective_poisson_ratio(c);
    let denom = 3.0 * (1.0 - 2.0 * nu);
    if denom.abs() < 1e-15 { 0.0 } else { e / denom }
}

// ---------------------------------------------------------------------------
// Hashin-Shtrikman bounds
// ---------------------------------------------------------------------------

/// Hashin-Shtrikman upper and lower bounds for a two-phase composite.
///
/// Returns `(K_lower, K_upper, G_lower, G_upper)`.
pub fn hashin_shtrikman_bounds(phase1: &Phase, phase2: &Phase) -> (f64, f64, f64, f64) {
    let k1 = phase1.bulk_modulus();
    let g1 = phase1.shear_modulus();
    let v1 = phase1.volume_fraction;
    let k2 = phase2.bulk_modulus();
    let g2 = phase2.shear_modulus();
    let v2 = phase2.volume_fraction;

    // Lower bounds: the softer phase plays the matrix, the stiffer one the inclusion.
    let (km, gm, vm, ki, _gi, vi) = if k1 < k2 {
        (k1, g1, v1, k2, g2, v2)
    } else {
        (k2, g2, v2, k1, g1, v1)
    };
    let k_lower = km + vi / (1.0 / (ki - km) + 3.0 * vm / (3.0 * km + 4.0 * gm));

    let (km2, gm2, vm2, _ki2, gi2, vi2) = if g1 < g2 {
        (k1, g1, v1, k2, g2, v2)
    } else {
        (k2, g2, v2, k1, g1, v1)
    };
    let g_lower = gm2
        + vi2
            / (1.0 / (gi2 - gm2)
                + 6.0 * (km2 + 2.0 * gm2) * vm2 / (5.0 * gm2 * (3.0 * km2 + 4.0 * gm2)));

    // Upper bounds (swap roles)
    let (km_u, gm_u, vm_u, ki_u, _gi_u, vi_u) = if k1 > k2 {
        (k1, g1, v1, k2, g2, v2)
    } else {
        (k2, g2, v2, k1, g1, v1)
    };
    let k_upper = km_u + vi_u / (1.0 / (ki_u - km_u) + 3.0 * vm_u / (3.0 * km_u + 4.0 * gm_u));

    let (km_u2, gm_u2, vm_u2, _ki_u2, gi_u2, vi_u2) = if g1 > g2 {
        (k1, g1, v1, k2, g2, v2)
    } else {
        (k2, g2, v2, k1, g1, v1)
    };
    let g_upper = gm_u2
        + vi_u2
            / (1.0 / (gi_u2 - gm_u2)
                + 6.0 * (km_u2 + 2.0 * gm_u2) * vm_u2
                    / (5.0 * gm_u2 * (3.0 * km_u2 + 4.0 * gm_u2)));

    (k_lower, k_upper, g_lower, g_upper)
}

// ---------------------------------------------------------------------------
// Mori-Tanaka scheme
// ---------------------------------------------------------------------------

/// Mori-Tanaka homogenization for particle/fiber reinforced composites.
#[derive(Debug, Clone)]
pub struct MoriTanakaScheme {
    /// Matrix phase.
    pub matrix: Phase,
    /// Inclusion (reinforcement) phase.
    pub inclusion: Phase,
}

impl MoriTanakaScheme {
    /// Create a new Mori-Tanaka scheme.
    pub fn new(matrix: Phase, inclusion: Phase) -> Self {
        Self { matrix, inclusion }
    }

    /// Eshelby tensor for a spherical inclusion in an isotropic matrix.
    ///
    /// Uses the matrix Poisson's ratio ν.  The tensor is returned as a 6×6
    /// matrix in the engineering-shear Voigt notation used by
    /// [`Phase::stiffness_voigt`] (order \[xx, yy, zz, yz, xz, xy\]), i.e. as
    /// the map from the eigenstrain vector to the constrained-strain vector
    /// with *engineering* shear components γ = 2ε.  The tensor components of a
    /// sphere are
    ///
    ///   S_1111 = (7 - 5ν) / (15(1 - ν)),   S_1122 = (5ν - 1) / (15(1 - ν)),
    ///   S_1212 = (4 - 5ν) / (15(1 - ν)),
    ///
    /// The constrained shear strain is ε_ij = 2 S_ijij ε*_ij (the components
    /// S_ijij and S_ijji both contribute).  With engineering strains
    /// γ = 2ε and γ* = 2ε* this reads γ_ij = 2 S_ijij γ*_ij, so the shear
    /// diagonal of the Voigt matrix is `2 S_1212`, not `S_1212`.
    pub fn eshelby_tensor_sphere(&self) -> [[f64; 6]; 6] {
        let nu = self.matrix.poisson_ratio;
        let denom = 15.0 * (1.0 - nu);
        let s11 = (7.0 - 5.0 * nu) / denom;
        let s12 = (5.0 * nu - 1.0) / denom;
        let s44 = (4.0 - 5.0 * nu) / denom;

        let mut s = [[0.0f64; 6]; 6];
        // Diagonal normal-normal entries
        s[0][0] = s11;
        s[1][1] = s11;
        s[2][2] = s11;
        // Off-diagonal normal-normal entries
        s[0][1] = s12;
        s[0][2] = s12;
        s[1][0] = s12;
        s[1][2] = s12;
        s[2][0] = s12;
        s[2][1] = s12;
        // Shear entries: Voigt factor 2 for engineering shear strains.
        s[3][3] = 2.0 * s44;
        s[4][4] = 2.0 * s44;
        s[5][5] = 2.0 * s44;
        s
    }

    /// Eshelby tensor for an aligned spheroidal inclusion in an isotropic
    /// matrix (prolate / fiber-like for `aspect_ratio > 1`).
    ///
    /// The spheroid's axis of revolution is the global x-axis (Voigt index 0);
    /// `aspect_ratio` is the ratio α = a₁/a₂ of the semi-axis along that axis
    /// to the (equal) transverse semi-axes.  `aspect_ratio > 1` is a prolate
    /// (fiber-like) inclusion, `aspect_ratio < 1` an oblate (platelet-like)
    /// one, `aspect_ratio = 1` the sphere and `f64::INFINITY` the infinitely
    /// long cylinder (aspect ratios above `1e8` are evaluated as the cylinder,
    /// which they equal to double precision).  The closed forms are those of
    /// Eshelby (1957) as
    /// tabulated by Mura (*Micromechanics of Defects in Solids*, §11) and by
    /// Tandon & Weng (1984).
    ///
    /// The result uses the same engineering-shear Voigt layout as
    /// [`eshelby_tensor_sphere`](Self::eshelby_tensor_sphere): the normal block
    /// holds S_1111, S_1122 = S_1133, S_2211 = S_3311, S_2222 = S_3333 and
    /// S_2233 = S_3322 (it is *not* symmetric for α ≠ 1), and the shear
    /// diagonal holds `2 S_2323` (yz) and `2 S_1212 = 2 S_1313` (xz, xy).
    ///
    /// The tensor is continuous at α = 1 (it reduces to the sphere tensor);
    /// close to α = 1 the closed forms are 0/0-like, so the shape functions
    /// are evaluated from their series expansion there and no precision is
    /// lost.  For a non-finite or non-positive `aspect_ratio` (other than
    /// `+∞`) there is no inclusion, and a tensor filled with `NaN` is returned
    /// so that the result can never be mistaken for a valid one.
    pub fn eshelby_tensor_prolate(&self, aspect_ratio: f64) -> [[f64; 6]; 6] {
        if aspect_ratio.is_nan() || aspect_ratio <= 0.0 {
            return [[f64::NAN; 6]; 6];
        }
        let nu = self.matrix.poisson_ratio;
        let (g, q) = spheroid_shape_factors(aspect_ratio);
        let two_one_minus_nu = 2.0 * (1.0 - nu);
        let s1111 = ((4.0 - 2.0 * nu) * (1.0 - g) + 2.0 * q) / two_one_minus_nu;
        let s1122 = (-(1.0 - 2.0 * nu) * (1.0 - g) - q) / two_one_minus_nu;
        let s2211 = (-1.0 + (1.0 + nu) * g - q) / two_one_minus_nu;
        let s2222 = (3.0 + 2.0 * (1.0 - 2.0 * nu) * g + 3.0 * q) / (4.0 * two_one_minus_nu);
        let s2233 = (0.5 - (1.0 - 2.0 * nu) * g + 0.5 * q) / (2.0 * two_one_minus_nu);
        let s2323 = (0.5 + (1.0 - 2.0 * nu) * g + 0.5 * q) / (2.0 * two_one_minus_nu);
        let s1212 = (-2.0 * nu + (1.0 + nu) * g - 2.0 * q) / (2.0 * two_one_minus_nu);

        let mut s = [[0.0f64; 6]; 6];
        s[0][0] = s1111;
        s[0][1] = s1122;
        s[0][2] = s1122;
        s[1][0] = s2211;
        s[2][0] = s2211;
        s[1][1] = s2222;
        s[2][2] = s2222;
        s[1][2] = s2233;
        s[2][1] = s2233;
        // Engineering-shear Voigt factor 2 on the shear diagonal.
        s[3][3] = 2.0 * s2323;
        s[4][4] = 2.0 * s1212;
        s[5][5] = 2.0 * s1212;
        s
    }

    /// Dilute concentration tensor A_dilute = (I + S C_m^{-1} (C_i - C_m))^{-1}.
    pub fn concentration_tensor_dilute(&self) -> [[f64; 6]; 6] {
        let cm = self.matrix.stiffness_voigt();
        let ci = self.inclusion.stiffness_voigt();
        let s_esh = self.eshelby_tensor_sphere();

        let cm_inv = match mat6_inv(&cm) {
            Some(v) => v,
            None => return mat6_identity(),
        };

        let delta_c = mat6_sub(&ci, &cm);
        // S * C_m^{-1} * ΔC
        let tmp = mat6_mul(&s_esh, &mat6_mul(&cm_inv, &delta_c));
        let inner = mat6_add(&mat6_identity(), &tmp);
        mat6_inv(&inner).unwrap_or_else(mat6_identity)
    }

    /// Effective stiffness via Mori-Tanaka:
    ///
    /// C_eff = C_m + V_i (C_i - C_m) A_MT (V_m I + V_i A_dilute)^{-1}
    pub fn effective_stiffness(&self) -> [[f64; 6]; 6] {
        let vi = self.inclusion.volume_fraction;
        let vm = self.matrix.volume_fraction;
        let cm = self.matrix.stiffness_voigt();
        let ci = self.inclusion.stiffness_voigt();

        let a_dil = self.concentration_tensor_dilute();
        // Compute (V_m I + V_i A_dil)
        let inner = mat6_add(&mat6_scale(&mat6_identity(), vm), &mat6_scale(&a_dil, vi));
        let inner_inv = match mat6_inv(&inner) {
            Some(v) => v,
            None => return cm,
        };

        let a_mt = mat6_mul(&a_dil, &inner_inv);
        let delta_c = mat6_sub(&ci, &cm);
        // C_eff = C_m + V_i * ΔC * A_MT
        mat6_add(&cm, &mat6_scale(&mat6_mul(&delta_c, &a_mt), vi))
    }
}

// ---------------------------------------------------------------------------
// Spheroid shape functions of Eshelby's solution
// ---------------------------------------------------------------------------

/// Half-width of the interval `|α² - 1| < SERIES_RADIUS` in which
/// [`spheroid_shape_factors`] uses the power series instead of the closed form.
const SERIES_RADIUS: f64 = 0.25;

/// Above this aspect ratio a prolate spheroid is indistinguishable from an
/// infinitely long cylinder in double precision (`1 - g` and `q` are below
/// `1e-15`); `α²` would overflow for `α` beyond `1e154`.
const CYLINDER_ASPECT_RATIO: f64 = 1.0e8;

/// Shape functions `(g, q)` of a spheroid with aspect ratio `alpha`.
///
/// `alpha` is the semi-axis along the axis of revolution divided by the
/// transverse semi-axis and `x = alpha² - 1`.  Eshelby's tensor of the spheroid
/// is a combination of
///
/// * `g`, the function of Tandon & Weng (1984),
///   `α / (α²-1)^{3/2} (α √(α²-1) - cosh⁻¹ α)` for `α > 1` and
///   `α / (1-α²)^{3/2} (cos⁻¹ α - α √(1-α²))` for `α < 1`, which equals
///   `1 - L` with `L` the depolarisation factor along the axis
///   (`g = 2/3` for the sphere, `g → 1` for the cylinder, `g → 0` for the disc);
/// * `q = (1 - 3g/2) / x`, the regular combination in which the `1/x`
///   singularities of the individual closed-form entries cancel
///   (`q = -1/5` for the sphere).
///
/// Writing the tensor through `q` instead of `1/(α²-1)` keeps it accurate to
/// rounding for `α → 1`, where the textbook closed forms lose all precision
/// (they are differences of terms of order `1/x`).
fn spheroid_shape_factors(alpha: f64) -> (f64, f64) {
    if alpha > CYLINDER_ASPECT_RATIO {
        return (1.0, 0.0);
    }
    let x = alpha * alpha - 1.0;
    if x.abs() < SERIES_RADIUS {
        spheroid_shape_factors_series(x)
    } else {
        spheroid_shape_factors_closed(alpha)
    }
}

/// Closed-form `(g, q)` of [`spheroid_shape_factors`], for `|α² - 1|` not small.
///
/// With `F(x) = asinh(√x)/√x` (`x > 0`) or `acos(α)/√(1-α²)` (`x < 0`),
/// `g = α (α - F) / x` and `q = (1 - 3g/2) / x`.
fn spheroid_shape_factors_closed(alpha: f64) -> (f64, f64) {
    let x = alpha * alpha - 1.0;
    let f = if x > 0.0 {
        let root = x.sqrt();
        root.asinh() / root
    } else {
        alpha.acos() / (1.0 - alpha * alpha).sqrt()
    };
    let g = alpha * (alpha - f) / x;
    let q = (1.0 - 1.5 * g) / x;
    (g, q)
}

/// Power-series `(g, q)` of [`spheroid_shape_factors`] in `x = α² - 1`
/// (convergent for `|x| < 1`; used for `|x| < SERIES_RADIUS`).
///
/// With `F(x) = Σ (-1)ⁿ C(2n,n) / (4ⁿ (2n+1)) xⁿ` the expansion of
/// `asinh(√x)/√x` and `√(1+x) = Σ sₘ xᵐ`, the product
/// `P(x) = √(1+x) F(x) = Σ pₖ xᵏ` has `p₀ = 1`, `p₁ = 1/3` and
/// `g = 2/3 - Σ_{k≥2} pₖ x^{k-1}`, `q = (3/2) Σ_{k≥2} pₖ x^{k-2}`.
fn spheroid_shape_factors_series(x: f64) -> (f64, f64) {
    const TERMS: usize = 40;
    let mut f_coef = [0.0_f64; TERMS];
    let mut s_coef = [0.0_f64; TERMS];
    f_coef[0] = 1.0;
    s_coef[0] = 1.0;
    // b = C(2n, n) / 4^n
    let mut b = 1.0_f64;
    for n in 1..TERMS {
        let nf = n as f64;
        b *= (2.0 * nf - 1.0) / (2.0 * nf);
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        f_coef[n] = sign * b / (2.0 * nf + 1.0);
        s_coef[n] = s_coef[n - 1] * (1.5 - nf) / nf;
    }
    let mut p_coef = [0.0_f64; TERMS];
    for (k, p) in p_coef.iter_mut().enumerate() {
        *p = (0..=k).map(|m| s_coef[m] * f_coef[k - m]).sum();
    }
    // q_poly = Σ_{k≥2} p_k x^{k-2} by Horner's rule.
    let q_poly = p_coef[2..]
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &p| acc * x + p);
    (2.0 / 3.0 - x * q_poly, 1.5 * q_poly)
}

#[cfg(test)]
mod eshelby_tests {
    use super::*;
    use std::f64::consts::PI;

    /// Poisson ratios of the matrix exercised by the tests.
    const NUS: [f64; 4] = [0.0, 0.2, 0.35, 0.49];

    fn scheme(nu: f64) -> MoriTanakaScheme {
        MoriTanakaScheme::new(
            Phase::new("matrix", 0.7, 3.5e9, nu),
            Phase::new("fibre", 0.3, 210e9, 0.3),
        )
    }

    /// Sum of the 3x3 normal block of a Voigt Eshelby matrix: S_iikk.
    ///
    /// For *every* ellipsoid this equals (1 + nu) / (1 - nu).
    fn normal_block_sum(s: &[[f64; 6]; 6]) -> f64 {
        s.iter()
            .take(3)
            .map(|row| row.iter().take(3).sum::<f64>())
            .sum()
    }

    /// Mura's (1987, eq. 11.17) elliptic integrals `I_i` and `I_ij` of an
    /// ellipsoid with semi-axes `a`, by composite Simpson quadrature.
    ///
    /// With s = c tan^2(theta) the integrands `1/((a_i^2+s) Delta(s))` and
    /// `1/((a_i^2+s)(a_j^2+s) Delta(s))`, `Delta = sqrt(prod(a_k^2+s))`, become
    /// analytic functions that vanish at both ends of `[0, pi/2]`.
    fn mura_integrals(a: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
        let a2 = a.map(|v| v * v);
        let c = (a2[0] * a2[1] * a2[2]).cbrt();
        let intervals = 20_000_usize;
        let h = 0.5 * PI / intervals as f64;
        let mut i_single = [0.0_f64; 3];
        let mut i_double = [[0.0_f64; 3]; 3];
        for k in 1..intervals {
            let weight = if k % 2 == 1 { 4.0 } else { 2.0 };
            let t = (k as f64 * h).tan();
            let s = c * t * t;
            let jacobian = 2.0 * c * t * (1.0 + t * t);
            let delta = ((a2[0] + s) * (a2[1] + s) * (a2[2] + s)).sqrt();
            for i in 0..3 {
                i_single[i] += weight * jacobian / ((a2[i] + s) * delta);
                for j in 0..3 {
                    i_double[i][j] += weight * jacobian / ((a2[i] + s) * (a2[j] + s) * delta);
                }
            }
        }
        let scale = 2.0 * PI * a[0] * a[1] * a[2] * h / 3.0;
        (
            i_single.map(|v| v * scale),
            i_double.map(|row| row.map(|v| v * scale)),
        )
    }

    /// Eshelby tensor components `[S1111, S1122, S2211, S2222, S2233, S2323,
    /// S1212]` of a spheroid with semi-axes `(alpha, 1, 1)` from Mura's general
    /// ellipsoid formulas (11.18) and the numerical integrals above.
    fn mura_spheroid_components(nu: f64, alpha: f64) -> [f64; 7] {
        let a = [alpha, 1.0, 1.0];
        let a2 = a.map(|v| v * v);
        let (i1, i2) = mura_integrals(a);
        let pref = 1.0 / (8.0 * PI * (1.0 - nu));
        let m = (1.0 - 2.0 * nu) * pref;
        [
            3.0 * pref * a2[0] * i2[0][0] + m * i1[0],
            pref * a2[1] * i2[0][1] - m * i1[0],
            pref * a2[0] * i2[1][0] - m * i1[1],
            3.0 * pref * a2[1] * i2[1][1] + m * i1[1],
            pref * a2[2] * i2[1][2] - m * i1[1],
            0.5 * pref * (a2[1] + a2[2]) * i2[1][2] + 0.5 * m * (i1[1] + i1[2]),
            0.5 * pref * (a2[0] + a2[1]) * i2[0][1] + 0.5 * m * (i1[0] + i1[1]),
        ]
    }

    #[test]
    fn sphere_voigt_shear_diagonal_is_twice_s1212() {
        for &nu in &NUS {
            let s = scheme(nu).eshelby_tensor_sphere();
            let s1212 = (4.0 - 5.0 * nu) / (15.0 * (1.0 - nu));
            for (idx, row) in s.iter().enumerate().skip(3) {
                assert!(
                    (row[idx] - 2.0 * s1212).abs() < 1e-14,
                    "nu = {nu}: S[{idx}][{idx}] = {} but 2*S_1212 = {}",
                    row[idx],
                    2.0 * s1212
                );
            }
            // Isotropy: S_1212 = (S_1111 - S_1122) / 2, i.e. the Voigt entry is S_1111 - S_1122.
            assert!((s[3][3] - (s[0][0] - s[0][1])).abs() < 1e-14);
            assert!((normal_block_sum(&s) - (1.0 + nu) / (1.0 - nu)).abs() < 1e-13);
        }
    }

    #[test]
    fn mori_tanaka_sphere_coincides_with_hashin_shtrikman_lower_bounds() {
        // Soft matrix with a stiff spherical inclusion: Mori-Tanaka is exactly the
        // lower Hashin-Shtrikman bound for both K and G.
        let matrix = Phase::new("epoxy", 0.7, 3.5e9, 0.35);
        let sphere = Phase::new("steel", 0.3, 210e9, 0.3);
        let (k_lower, _, g_lower, _) = hashin_shtrikman_bounds(&matrix, &sphere);
        let eff = MoriTanakaScheme::new(matrix, sphere).effective_stiffness();
        let g_eff = effective_shear_modulus(&eff);
        let k_eff = effective_bulk_modulus(&eff);
        assert!(
            (g_eff - g_lower).abs() <= 1e-8 * g_lower,
            "G_MT = {g_eff:.9e}, G_HS- = {g_lower:.9e}"
        );
        assert!(
            (k_eff - k_lower).abs() <= 1e-8 * k_lower,
            "K_MT = {k_eff:.9e}, K_HS- = {k_lower:.9e}"
        );
    }

    #[test]
    fn dilute_concentration_tensor_agrees_with_the_independent_implementation() {
        // `StrainConcentrationTensor` builds the same dilute Eshelby tensor with
        // the Voigt factor 2; the Mori-Tanaka module used to disagree with it.
        use crate::homogenization::advanced::StrainConcentrationTensor;
        let matrix = Phase::new("matrix", 0.7, 3.5e9, 0.35);
        let inclusion = Phase::new("sphere", 0.3, 210e9, 0.3);
        let a_mt = MoriTanakaScheme::new(matrix, inclusion).concentration_tensor_dilute();
        let a_ref =
            StrainConcentrationTensor::dilute_spherical_inclusion(3.5e9, 0.35, 210e9, 0.3).tensor;
        for (row, ref_row) in a_mt.iter().zip(a_ref.iter()) {
            for (&v, &v_ref) in row.iter().zip(ref_row.iter()) {
                assert!((v - v_ref).abs() < 1e-10, "{v} vs {v_ref}");
            }
        }
    }

    #[test]
    fn mura_reference_reproduces_the_sphere_tensor() {
        // Validates the independent quadrature reference itself.
        let (i1, _) = mura_integrals([1.0, 1.0, 1.0]);
        for value in i1 {
            assert!((value - 4.0 * PI / 3.0).abs() < 1e-10, "I_i = {value}");
        }
        for &nu in &NUS {
            let m = mura_spheroid_components(nu, 1.0);
            let s = scheme(nu).eshelby_tensor_sphere();
            let expected = [
                s[0][0],
                s[0][1],
                s[1][0],
                s[1][1],
                s[1][2],
                0.5 * s[3][3],
                0.5 * s[4][4],
            ];
            for (k, (a, b)) in m.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-10,
                    "nu = {nu}, component {k}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn prolate_tensor_matches_mura_elliptic_integral_reference() {
        let names = [
            "S1111", "S1122", "S2211", "S2222", "S2233", "S2323", "S1212",
        ];
        for &nu in &NUS {
            let mt = scheme(nu);
            for &alpha in &[
                0.05, 0.2, 0.5, 0.9, 0.99, 1.0, 1.01, 1.05, 1.1, 1.5, 2.0, 5.0, 20.0,
            ] {
                let s = mt.eshelby_tensor_prolate(alpha);
                let reference = mura_spheroid_components(nu, alpha);
                let got = [
                    s[0][0],
                    s[0][1],
                    s[1][0],
                    s[1][1],
                    s[1][2],
                    0.5 * s[3][3],
                    0.5 * s[4][4],
                ];
                for ((name, a), b) in names.iter().zip(got.iter()).zip(reference.iter()) {
                    assert!(
                        (a - b).abs() < 1e-9,
                        "nu = {nu}, aspect ratio {alpha}: {name} = {a} but Mura gives {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn prolate_tensor_is_continuous_through_the_sphere() {
        for &nu in &NUS {
            let mt = scheme(nu);
            let sphere = mt.eshelby_tensor_sphere();
            for &eps in &[1e-2, 1e-5, 1e-9, 0.0] {
                for &alpha in &[1.0 + eps, 1.0 - eps] {
                    let s = mt.eshelby_tensor_prolate(alpha);
                    let mut max_diff = 0.0_f64;
                    for (row, sphere_row) in s.iter().zip(sphere.iter()) {
                        for (&v, &v_sphere) in row.iter().zip(sphere_row.iter()) {
                            max_diff = max_diff.max((v - v_sphere).abs());
                        }
                    }
                    assert!(
                        max_diff <= 2.0 * eps + 1e-13,
                        "nu = {nu}, alpha = {alpha}: |S - S_sphere| = {max_diff:e}"
                    );
                    if eps == 1e-2 {
                        // The tensor must actually depend on the aspect ratio.
                        assert!(
                            max_diff > 1e-4,
                            "nu = {nu}, alpha = {alpha}: tensor is the sphere tensor"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn prolate_tensor_differs_from_the_sphere_for_elongated_inclusions() {
        let mt = scheme(0.35);
        let sphere = mt.eshelby_tensor_sphere();
        let fibre = mt.eshelby_tensor_prolate(10.0);
        // A needle hardly constrains axial eigenstrain: S_1111 is far below the sphere's.
        assert!(fibre[0][0] < 0.1 * sphere[0][0], "S1111 = {}", fibre[0][0]);
        // ... but constrains transverse eigenstrain more than the sphere does.
        assert!(fibre[1][1] > sphere[1][1], "S2222 = {}", fibre[1][1]);
        // Transverse shear is stiffer than for the sphere (cylinder limit
        // 2*S_2323 = (3 - 4 nu) / (4 (1 - nu)) = 0.615 versus 0.462 at nu = 0.35).
        assert!(
            fibre[3][3] - sphere[3][3] > 0.1,
            "2*S2323 = {}",
            fibre[3][3]
        );
    }

    #[test]
    fn prolate_tensor_has_transversely_isotropic_voigt_structure() {
        for &nu in &NUS {
            let mt = scheme(nu);
            for &alpha in &[
                0.05,
                0.3,
                0.8,
                1.0,
                1.2,
                2.0,
                7.0,
                100.0,
                1.0e4,
                f64::INFINITY,
            ] {
                let s = mt.eshelby_tensor_prolate(alpha);
                let tol = 1e-13;
                // Symmetry about the x axis.
                assert!((s[1][1] - s[2][2]).abs() < tol);
                assert!((s[0][1] - s[0][2]).abs() < tol);
                assert!((s[1][0] - s[2][0]).abs() < tol);
                assert!((s[1][2] - s[2][1]).abs() < tol);
                assert!((s[4][4] - s[5][5]).abs() < tol);
                // Isotropy in the transverse plane: S2323 = (S2222 - S2233) / 2,
                // i.e. the engineering-shear entry is S2222 - S2233.
                assert!(
                    (s[3][3] - (s[1][1] - s[1][2])).abs() < tol,
                    "nu = {nu}, alpha = {alpha}: S44 = {}, S22 - S23 = {}",
                    s[3][3],
                    s[1][1] - s[1][2]
                );
                // No coupling between normal and shear components, nor among shears.
                for (i, row) in s.iter().enumerate() {
                    for (j, &v) in row.iter().enumerate() {
                        let coupled = (i < 3) != (j < 3) || (i >= 3 && i != j);
                        if coupled {
                            assert_eq!(v, 0.0, "S[{i}][{j}] must vanish");
                        }
                    }
                }
                // Eshelby's shape-independent invariant S_iikk = (1 + nu) / (1 - nu).
                assert!(
                    (normal_block_sum(&s) - (1.0 + nu) / (1.0 - nu)).abs() < 1e-12,
                    "nu = {nu}, alpha = {alpha}: S_iikk = {}",
                    normal_block_sum(&s)
                );
            }
        }
    }

    #[test]
    fn long_fibre_limit_is_the_infinite_cylinder() {
        for &nu in &NUS {
            let mt = scheme(nu);
            let cylinder = mt.eshelby_tensor_prolate(f64::INFINITY);
            let d = 8.0 * (1.0 - nu);
            assert!(cylinder[0][0].abs() < 1e-14, "S1111 = {}", cylinder[0][0]);
            assert!(cylinder[0][1].abs() < 1e-14, "S1122 = {}", cylinder[0][1]);
            assert!((cylinder[1][1] - (5.0 - 4.0 * nu) / d).abs() < 1e-14);
            assert!((cylinder[1][2] - (4.0 * nu - 1.0) / d).abs() < 1e-14);
            assert!((cylinder[1][0] - nu / (2.0 * (1.0 - nu))).abs() < 1e-14);
            assert!((cylinder[3][3] - 2.0 * (3.0 - 4.0 * nu) / d).abs() < 1e-14);
            assert!((cylinder[4][4] - 0.5).abs() < 1e-14);
            // Finite aspect ratios approach the cylinder like ln(alpha) / alpha^2.
            for &(alpha, tol) in &[(100.0, 2e-3), (1.0e4, 1e-6), (1.0e7, 2e-12)] {
                let s = mt.eshelby_tensor_prolate(alpha);
                for (row, cyl_row) in s.iter().zip(cylinder.iter()) {
                    for (&v, &v_cyl) in row.iter().zip(cyl_row.iter()) {
                        assert!(
                            (v - v_cyl).abs() < tol,
                            "nu = {nu}, alpha = {alpha}: {v} vs cylinder {v_cyl}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn thin_disc_limit_is_the_penny_shaped_inclusion() {
        for &nu in &NUS {
            let s = scheme(nu).eshelby_tensor_prolate(1.0e-6);
            assert!((s[0][0] - 1.0).abs() < 1e-5, "S1111 = {}", s[0][0]);
            assert!(
                (s[0][1] - nu / (1.0 - nu)).abs() < 1e-5,
                "S1122 = {}",
                s[0][1]
            );
            assert!(s[1][0].abs() < 1e-5, "S2211 = {}", s[1][0]);
            assert!(s[1][1].abs() < 1e-5, "S2222 = {}", s[1][1]);
            assert!(s[1][2].abs() < 1e-5, "S2233 = {}", s[1][2]);
            assert!(s[3][3].abs() < 1e-5, "2*S2323 = {}", s[3][3]);
            assert!((s[4][4] - 1.0).abs() < 1e-5, "2*S1212 = {}", s[4][4]);
        }
    }

    #[test]
    fn shape_function_series_agrees_with_the_closed_form() {
        // The series is used for |alpha^2 - 1| < 0.25; compare with the closed form
        // on a wider range, where both are accurate to rounding.
        for &alpha in &[0.75, 0.8, 0.9, 0.975, 1.025, 1.1, 1.2] {
            let x = alpha * alpha - 1.0;
            let (g_series, q_series) = spheroid_shape_factors_series(x);
            let (g_closed, q_closed) = spheroid_shape_factors_closed(alpha);
            assert!(
                (g_series - g_closed).abs() < 5e-12,
                "alpha = {alpha}: g = {g_series} (series) vs {g_closed} (closed)"
            );
            assert!(
                (q_series - q_closed).abs() < 5e-12,
                "alpha = {alpha}: q = {q_series} (series) vs {q_closed} (closed)"
            );
        }
        // The two branches of `spheroid_shape_factors` meet continuously.
        for sign in [-1.0_f64, 1.0] {
            let x = sign * SERIES_RADIUS;
            let alpha = (1.0 + x).sqrt();
            let (g_series, q_series) = spheroid_shape_factors_series(x);
            let (g_closed, q_closed) = spheroid_shape_factors_closed(alpha);
            assert!((g_series - g_closed).abs() < 1e-13);
            assert!((q_series - q_closed).abs() < 1e-13);
        }
        // Sphere: g = 2/3, q = -1/5.
        let (g, q) = spheroid_shape_factors(1.0);
        assert!((g - 2.0 / 3.0).abs() < 1e-16 && (q + 0.2).abs() < 1e-16);
    }

    #[test]
    fn invalid_aspect_ratios_give_nan_tensors() {
        let mt = scheme(0.3);
        for &bad in &[0.0, -2.0, f64::NAN, f64::NEG_INFINITY] {
            let s = mt.eshelby_tensor_prolate(bad);
            assert!(
                s.iter().flatten().all(|v| v.is_nan()),
                "aspect ratio {bad} must not give a usable tensor"
            );
        }
    }

    /// Mori-Tanaka effective stiffness of aligned spheroids built from the
    /// public Eshelby tensor (dilute concentration tensor + Benveniste's
    /// Mori-Tanaka closure).
    fn mori_tanaka_aligned(matrix: &Phase, inclusion: &Phase, aspect_ratio: f64) -> [[f64; 6]; 6] {
        let mt = MoriTanakaScheme::new(matrix.clone(), inclusion.clone());
        let s = mt.eshelby_tensor_prolate(aspect_ratio);
        let cm = matrix.stiffness_voigt();
        let ci = inclusion.stiffness_voigt();
        let cm_inv = mat6_inv(&cm).expect("matrix stiffness is invertible");
        let delta_c = mat6_sub(&ci, &cm);
        let dilute_inner = mat6_add(
            &mat6_identity(),
            &mat6_mul(&s, &mat6_mul(&cm_inv, &delta_c)),
        );
        let a_dilute = mat6_inv(&dilute_inner).expect("dilute concentration tensor");
        let closure = mat6_add(
            &mat6_scale(&mat6_identity(), matrix.volume_fraction),
            &mat6_scale(&a_dilute, inclusion.volume_fraction),
        );
        let a_mt = mat6_mul(&a_dilute, &mat6_inv(&closure).expect("Mori-Tanaka closure"));
        mat6_add(
            &cm,
            &mat6_scale(&mat6_mul(&delta_c, &a_mt), inclusion.volume_fraction),
        )
    }

    #[test]
    fn aligned_long_fibres_follow_the_rule_of_mixtures_along_the_axis() {
        let (e_m, nu_m, e_f, nu_f, vf) = (3.5e9, 0.35, 210e9, 0.30, 0.3);
        let matrix = Phase::new("matrix", 1.0 - vf, e_m, nu_m);
        let fibre = Phase::new("fibre", vf, e_f, nu_f);
        let c_eff = mori_tanaka_aligned(&matrix, &fibre, 1.0e6);
        let compliance = mat6_inv(&c_eff).expect("effective stiffness is invertible");
        let e_axial = 1.0 / compliance[0][0];
        let voigt = vf * e_f + (1.0 - vf) * e_m;
        assert!(
            (e_axial - voigt).abs() < 0.01 * voigt,
            "E_axial = {e_axial:.4e}, rule of mixtures {voigt:.4e}"
        );
        // The composite is transversely isotropic about the fibre axis:
        // C22 = C33, C12 = C13, C66 = C55 and C44 = (C22 - C23) / 2.
        let scale = c_eff[1][1];
        assert!((c_eff[1][1] - c_eff[2][2]).abs() < 1e-9 * scale);
        assert!((c_eff[0][1] - c_eff[0][2]).abs() < 1e-9 * scale);
        assert!((c_eff[4][4] - c_eff[5][5]).abs() < 1e-9 * scale);
        assert!(
            (c_eff[3][3] - 0.5 * (c_eff[1][1] - c_eff[1][2])).abs() < 1e-9 * scale,
            "C44 = {}, (C22 - C23)/2 = {}",
            c_eff[3][3],
            0.5 * (c_eff[1][1] - c_eff[1][2])
        );
        // Short spheroids are far less stiff along the axis than long fibres.
        let c_short = mori_tanaka_aligned(&matrix, &fibre, 2.0);
        let e_short = 1.0 / mat6_inv(&c_short).expect("invertible")[0][0];
        assert!(e_short < 0.5 * e_axial, "E(alpha = 2) = {e_short:.4e}");
        assert!(e_short > e_m, "E(alpha = 2) = {e_short:.4e}");
    }

    #[test]
    fn aligned_spheres_reproduce_the_isotropic_mori_tanaka_stiffness() {
        let matrix = Phase::new("matrix", 0.6, 3.5e9, 0.35);
        let sphere = Phase::new("sphere", 0.4, 70e9, 0.2);
        let aligned = mori_tanaka_aligned(&matrix, &sphere, 1.0);
        let isotropic = MoriTanakaScheme::new(matrix, sphere).effective_stiffness();
        let scale = isotropic[0][0];
        for (row, iso_row) in aligned.iter().zip(isotropic.iter()) {
            for (&v, &v_iso) in row.iter().zip(iso_row.iter()) {
                assert!((v - v_iso).abs() < 1e-10 * scale, "{v} vs {v_iso}");
            }
        }
    }
}
