//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
/// Arruda-Boyce hyperelastic material (8-chain rubber elasticity model).
///
/// W = mu * sum_{i=1..5} C_i / lambda_m^{2i-2} * (Ī1^i - 3^i) + K/2*(J-1)^2
///
/// where Ī1 = tr(J^{-2/3} FᵀF) is the first invariant of the isochoric
/// deformation, so the deviatoric response is independent of the volume change
/// and the reference configuration is stress free.  The five-term series
/// (C_1 = 1/2, C_2 = 1/20, C_3 = 11/1050, C_4 = 19/7000, C_5 = 519/673750) is
/// the standard expansion of the inverse Langevin function of the 8-chain
/// network.  The initial shear modulus is
/// mu * (1 + 3/(5 lambda_m²) + 99/(175 lambda_m⁴) + 513/(875 lambda_m⁶) + 42039/(67375 lambda_m⁸)).
pub struct ArrudaBoyce {
    /// Shear modulus mu.
    pub mu: f64,
    /// Bulk modulus K.
    pub bulk_modulus: f64,
    /// Locking stretch lambda_m.
    pub lambda_m: f64,
}
impl ArrudaBoyce {
    /// Strain energy density W(F).
    ///
    /// Uses the 5-term series expansion in the isochoric invariant Ī1.
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let (i1, _, _) = invariants(deviatoric_right_cauchy_green(f));
        let j = mat3_det(f);
        let lm2 = self.lambda_m * self.lambda_m;
        let coeffs = [
            0.5,
            1.0 / 20.0,
            11.0 / 1050.0,
            19.0 / 7000.0,
            519.0 / 673750.0,
        ];
        let mut w_dev = 0.0;
        let mut i1_pow = i1;
        let mut three_pow = 3.0_f64;
        let mut lm_pow = 1.0_f64;
        for (k, &ck) in coeffs.iter().enumerate() {
            w_dev += self.mu * ck / lm_pow * (i1_pow - three_pow);
            i1_pow *= i1;
            three_pow *= 3.0;
            if k < coeffs.len() - 1 {
                lm_pow *= lm2;
            }
        }
        let w_vol = self.bulk_modulus / 2.0 * (j - 1.0).powi(2);
        w_dev + w_vol
    }
    /// First Piola-Kirchhoff stress via numerical differentiation.
    pub fn pk1_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        numerical_pk1(f, |ff| self.strain_energy(ff))
    }
}
/// Mooney-Rivlin hyperelastic material.
///
/// Strain energy: W = c10*(Ī1 - 3) + c01*(Ī2 - 3) + (J-1)²/d1
pub struct MooneyRivlin {
    /// Material constant C10.
    pub c10: f64,
    /// Material constant C01.
    pub c01: f64,
    /// Compressibility parameter D1.
    pub d1: f64,
}
impl MooneyRivlin {
    /// Strain energy density W(F).
    ///
    /// Uses deviatoric invariants Ī1, Ī2 of the isochoric deformation.
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let j = mat3_det(f);
        let j_neg13 = j.powf(-1.0 / 3.0);
        let f_bar = mat3_scale(f, j_neg13);
        let c_bar = right_cauchy_green(f_bar);
        let (i1_bar, i2_bar, _) = invariants(c_bar);
        self.c10 * (i1_bar - 3.0) + self.c01 * (i2_bar - 3.0) + (j - 1.0).powi(2) / self.d1
    }
    /// Second Piola-Kirchhoff stress S = 2 * dW/dC (numerical differentiation).
    ///
    /// Uses central differences with perturbation h = 1e-7.
    pub fn pk2_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let c = right_cauchy_green(f);
        let h = 1e-7_f64;
        let mut s = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j_idx in 0..3 {
                let mut c_plus = c;
                let mut c_minus = c;
                c_plus[i][j_idx] += h;
                c_plus[j_idx][i] += h;
                c_minus[i][j_idx] -= h;
                c_minus[j_idx][i] -= h;
                let w_plus = self.strain_energy_from_c(c_plus);
                let w_minus = self.strain_energy_from_c(c_minus);
                s[i][j_idx] = (w_plus - w_minus) / (2.0 * h);
            }
        }
        s
    }
    /// Compute strain energy from a given C (right Cauchy-Green).
    fn strain_energy_from_c(&self, c: [[f64; 3]; 3]) -> f64 {
        let j2 = mat3_det(c);
        let j = j2.sqrt().max(1e-15);
        let j_neg23 = j.powf(-2.0 / 3.0);
        let (i1, i2, _) = invariants(c);
        let i1_bar = i1 * j_neg23;
        let i2_bar = i2 * j.powf(-4.0 / 3.0);
        self.c10 * (i1_bar - 3.0) + self.c01 * (i2_bar - 3.0) + (j - 1.0).powi(2) / self.d1
    }
}
/// Holzapfel-Ogden anisotropic hyperelastic model for fibrous tissues.
///
/// Used for arterial walls, tendons and ligaments.  The model includes an
/// isotropic matrix contribution (Neo-Hookean like) and a fibre reinforcement:
///
/// W = c10*(Ī1 - 3) + k1/(2k2) * (exp(k2*(I4 - 1)²) - 1) + K/2*(J-1)²
///
/// where I4 = a₀ · C · a₀ is the fibre stretch invariant along direction a₀.
///
/// Reference: Holzapfel, G.A., Gasser, T.C., Ogden, R.W. (2000).
/// "A new constitutive framework for arterial wall mechanics and a
/// comparative study with other models". J. Elasticity 61, 1–48.
pub struct HolzapfelOgden {
    /// Matrix modulus c10 \[Pa\].
    pub c10: f64,
    /// Bulk modulus K \[Pa\].
    pub bulk_modulus: f64,
    /// Fibre stiffness k1 \[Pa\].
    pub k1: f64,
    /// Dimensionless fibre nonlinearity k2.
    pub k2: f64,
    /// Fibre direction unit vector a₀ in reference configuration.
    pub a: [f64; 3],
}
impl HolzapfelOgden {
    /// Strain energy density W(F).
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let j = mat3_det(f);
        let j_neg23 = j.powf(-2.0 / 3.0);
        let c = right_cauchy_green(f);
        let (i1, _, _) = invariants(c);
        let i1_bar = i1 * j_neg23;
        let w_iso = self.c10 * (i1_bar - 3.0);
        let a = self.a;
        let mut ca = [0.0_f64; 3];
        for i in 0..3 {
            for k in 0..3 {
                ca[i] += c[i][k] * a[k];
            }
        }
        let i4_full = a[0] * ca[0] + a[1] * ca[1] + a[2] * ca[2];
        let i4 = i4_full * j_neg23;
        let w_fibre = if i4 > 1.0 {
            let di4 = i4 - 1.0;
            self.k1 / (2.0 * self.k2) * ((self.k2 * di4 * di4).exp() - 1.0)
        } else {
            0.0
        };
        let w_vol = self.bulk_modulus / 2.0 * (j - 1.0).powi(2);
        w_iso + w_fibre + w_vol
    }
    /// First Piola-Kirchhoff stress via numerical differentiation.
    pub fn pk1_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        numerical_pk1(f, |ff| self.strain_energy(ff))
    }
    /// Cauchy stress σ = P Fᵀ / J.
    pub fn cauchy_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        pk1_to_cauchy(f, self.pk1_stress(f))
    }
}
/// Volumetric penalty energy type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyType {
    /// Quadratic penalty: U(J) = κ/2 * (J-1)².
    Quadratic,
    /// Simo-Taylor penalty: U(J) = κ/4 * (J² - 1 - 2 ln J).
    SimoTaylor,
    /// Logarithmic penalty: U(J) = κ/2 * (ln J)².
    Logarithmic,
}
/// Fung-type exponential hyperelastic model (common for biological tissue).
///
/// W = c/2 * (exp(Q) - 1) + K/2*(J-1)^2
/// where Q = b1*(Ī1-3)^2 and Ī1 = tr(J^{-2/3} FᵀF) is the first invariant of
/// the isochoric deformation (a pure volume change does not load the
/// exponential term).
pub struct Fung {
    /// Material constant c.
    pub c: f64,
    /// Material constant b1 (exponent scaling).
    pub b1: f64,
    /// Bulk modulus K.
    pub bulk_modulus: f64,
}
impl Fung {
    /// Strain energy density W(F).
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let (i1, _, _) = invariants(deviatoric_right_cauchy_green(f));
        let j = mat3_det(f);
        let di = i1 - 3.0;
        let q = self.b1 * di * di;
        let w_dev = self.c / 2.0 * (q.exp() - 1.0);
        let w_vol = self.bulk_modulus / 2.0 * (j - 1.0).powi(2);
        w_dev + w_vol
    }
    /// First Piola-Kirchhoff stress via numerical differentiation.
    pub fn pk1_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        numerical_pk1(f, |ff| self.strain_energy(ff))
    }
}
/// Yeoh hyperelastic material (polynomial in the isochoric invariant Ī1).
///
/// W = c1*(Ī1-3) + c2*(Ī1-3)^2 + c3*(Ī1-3)^3 + K/2*(J-1)^2
///
/// with Ī1 = tr(J^{-2/3} FᵀF).  The initial shear modulus is 2*c1 and the
/// reference configuration is stress free.
pub struct Yeoh {
    /// First-order coefficient c1.
    pub c1: f64,
    /// Second-order coefficient c2.
    pub c2: f64,
    /// Third-order coefficient c3.
    pub c3: f64,
    /// Bulk modulus K.
    pub bulk_modulus: f64,
}
impl Yeoh {
    /// Strain energy density W(F).
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let (i1, _, _) = invariants(deviatoric_right_cauchy_green(f));
        let j = mat3_det(f);
        let di = i1 - 3.0;
        let w_dev = self.c1 * di + self.c2 * di * di + self.c3 * di * di * di;
        let w_vol = self.bulk_modulus / 2.0 * (j - 1.0).powi(2);
        w_dev + w_vol
    }
    /// First Piola-Kirchhoff stress via numerical differentiation.
    pub fn pk1_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        numerical_pk1(f, |ff| self.strain_energy(ff))
    }
}
/// Gent hyperelastic material model with finite extensibility.
///
/// W = -mu*J_m/2 * ln(1 - (Ī1-3)/J_m) + K/2*(J-1)^2
///
/// with Ī1 = tr(J^{-2/3} FᵀF) the first invariant of the isochoric deformation.
/// The initial shear modulus is mu and the reference configuration is stress
/// free.
pub struct Gent {
    /// Shear modulus mu.
    pub mu: f64,
    /// Bulk modulus K.
    pub bulk_modulus: f64,
    /// Maximum extensibility parameter J_m.
    pub j_m: f64,
}
impl Gent {
    /// Strain energy density W(F).
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let (i1, _, _) = invariants(deviatoric_right_cauchy_green(f));
        let j = mat3_det(f);
        let arg = 1.0 - (i1 - 3.0) / self.j_m;
        let w_dev = if arg > 1e-15 {
            -self.mu * self.j_m / 2.0 * arg.ln()
        } else {
            f64::INFINITY
        };
        let w_vol = self.bulk_modulus / 2.0 * (j - 1.0).powi(2);
        w_dev + w_vol
    }
    /// First Piola-Kirchhoff stress via numerical differentiation.
    pub fn pk1_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        numerical_pk1(f, |ff| self.strain_energy(ff))
    }
}
/// Nearly-incompressible Neo-Hookean material with pressure variable.
///
/// Strain energy: W = μ/2*(Ī1-3) + p*(J-1) - p²/(2κ)
/// where p is the independent pressure variable.
#[derive(Debug, Clone)]
pub struct NearlyIncompressibleNeoHookean {
    /// Shear modulus.
    pub mu: f64,
    /// Bulk modulus κ (penalty parameter).
    pub kappa: f64,
    /// Current pressure variable p (mixed formulation).
    pub pressure: f64,
}
impl NearlyIncompressibleNeoHookean {
    /// Create from Young's modulus and Poisson's ratio.
    pub fn from_young_poisson(e: f64, nu: f64) -> Self {
        let mu = e / (2.0 * (1.0 + nu));
        let kappa = e / (3.0 * (1.0 - 2.0 * nu));
        Self {
            mu,
            kappa,
            pressure: 0.0,
        }
    }
    /// Strain energy density W(F, p).
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let c_bar = deviatoric_right_cauchy_green(f);
        let (i1_bar, _, _) = invariants(c_bar);
        let j = mat3_det(f);
        let w_dev = self.mu / 2.0 * (i1_bar - 3.0);
        let w_vol = self.pressure * (j - 1.0) - self.pressure * self.pressure / (2.0 * self.kappa);
        w_dev + w_vol
    }
    /// Update the pressure variable from volumetric constraint.
    ///
    /// From stationarity w.r.t. p: p = κ*(J-1).
    pub fn update_pressure(&mut self, f: [[f64; 3]; 3]) {
        let j = mat3_det(f);
        self.pressure = self.kappa * (j - 1.0);
    }
    /// First Piola-Kirchhoff stress P = dW_dev/dF + p * J * F^{-T}.
    pub fn pk1_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let p_dev = neo_hookean_deviatoric_pk1(self.mu, f);
        let j = mat3_det(f);
        let f_inv = mat3_inverse(f).unwrap_or(mat3_identity());
        let f_inv_t = mat3_transpose(f_inv);
        let mut result = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                result[i][k] = p_dev[i][k] + self.pressure * j * f_inv_t[i][k];
            }
        }
        result
    }
    /// Cauchy stress σ = P F^T / J.
    pub fn cauchy_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let p = self.pk1_stress(f);
        pk1_to_cauchy(f, p)
    }
}
/// Compressible Neo-Hookean hyperelastic material.
///
/// Strain energy: W = μ/2*(I1 - 3) - μ*ln(J) + λ/2*(ln(J))²
pub struct NeoHookean {
    /// Shear modulus μ.
    pub mu: f64,
    /// First Lamé parameter λ.
    pub lambda: f64,
}
impl NeoHookean {
    /// Construct directly from shear modulus μ and first Lamé parameter λ.
    pub fn new(mu: f64, lambda: f64) -> Self {
        Self { mu, lambda }
    }
    /// Construct from Young's modulus `E` and Poisson's ratio `nu`.
    pub fn from_young_poisson(e: f64, nu: f64) -> Self {
        let mu = e / (2.0 * (1.0 + nu));
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        Self { mu, lambda }
    }
    /// Strain energy density W(F).
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let c = right_cauchy_green(f);
        let (i1, _, _) = invariants(c);
        let j = mat3_det(f);
        let ln_j = j.ln();
        self.mu / 2.0 * (i1 - 3.0) - self.mu * ln_j + self.lambda / 2.0 * ln_j * ln_j
    }
    /// First Piola-Kirchhoff stress P = dW/dF.
    ///
    /// P = μ*(F - F^{-T}) + λ*ln(J)*F^{-T}
    pub fn pk1_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let j = mat3_det(f);
        let ln_j = j.ln();
        let f_inv = mat3_inverse(f).unwrap_or(mat3_identity());
        let f_inv_t = mat3_transpose(f_inv);
        let mut p = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j_idx in 0..3 {
                p[i][j_idx] = self.mu * (f[i][j_idx] - f_inv_t[i][j_idx])
                    + self.lambda * ln_j * f_inv_t[i][j_idx];
            }
        }
        p
    }
    /// Cauchy stress σ = P * F^T / J.
    pub fn cauchy_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let j = mat3_det(f);
        let p = self.pk1_stress(f);
        let ft = mat3_transpose(f);
        let pft = mat3_mul(p, ft);
        let mut sigma = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for k in 0..3 {
                sigma[i][k] = pft[i][k] / j;
            }
        }
        sigma
    }
}
/// One term of an Ogden hyperelastic model.
pub struct OgdenTerm {
    /// Modulus μ for this term.
    pub mu: f64,
    /// Exponent α for this term.
    pub alpha: f64,
}
impl OgdenTerm {
    /// Uniaxial strain energy for this single Ogden term.
    ///
    /// W_uni = μ/α * (λ1^α + 2*λ1^(-α/2) - 3)
    pub fn strain_energy_uniaxial(&self, lambda1: f64) -> f64 {
        self.mu / self.alpha
            * (lambda1.powf(self.alpha) + 2.0 * lambda1.powf(-self.alpha / 2.0) - 3.0)
    }
}
/// Varga hyperelastic material (linear in the principal isochoric stretches).
///
/// W = 2μ * (λ̄1 + λ̄2 + λ̄3 - 3) + K/2*(J-1)²
///
/// where λ̄a = √(eig_a(C̄)) are the principal stretches of the isochoric
/// deformation, C̄ = J^{-2/3} FᵀF (so λ̄1 λ̄2 λ̄3 = 1).  The factor 2 makes the
/// initial shear modulus equal to μ, in line with the other models of this
/// module; the reference configuration is stress free.  This is the simplest
/// entropic rubber model for moderate deformations.
pub struct Varga {
    /// Initial shear modulus μ.
    pub mu: f64,
    /// Bulk modulus K.
    pub bulk_modulus: f64,
}
impl Varga {
    /// Strain energy density W(F).
    ///
    /// The principal isochoric stretches come from the eigenvalues of the
    /// deviatoric right Cauchy-Green tensor C̄ (closed-form symmetric 3×3
    /// solver, accurate also for repeated stretches such as uniaxial or
    /// equibiaxial states): W_dev = 2μ (λ̄1 + λ̄2 + λ̄3 - 3).
    pub fn strain_energy(&self, f: [[f64; 3]; 3]) -> f64 {
        let j = mat3_det(f);
        let stretch_sum: f64 = deviatoric_principal_stretches(f).iter().sum();
        let w_dev = 2.0 * self.mu * (stretch_sum - 3.0);
        let w_vol = self.bulk_modulus / 2.0 * (j - 1.0).powi(2);
        w_dev + w_vol
    }
    /// First Piola-Kirchhoff stress via numerical differentiation.
    pub fn pk1_stress(&self, f: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        numerical_pk1(f, |ff| self.strain_energy(ff))
    }
}
/// Volumetric penalty function for nearly-incompressible hyperelasticity.
///
/// These penalty functions ensure J ≈ 1 for large bulk modulus κ.
#[derive(Debug, Clone, Copy)]
pub struct VolumetricPenalty {
    /// Bulk modulus κ \[Pa\].
    pub kappa: f64,
    /// Penalty function type.
    pub penalty_type: PenaltyType,
}
impl VolumetricPenalty {
    /// Quadratic volumetric penalty: U = κ/2 * (J-1)².
    pub fn quadratic(kappa: f64) -> Self {
        Self {
            kappa,
            penalty_type: PenaltyType::Quadratic,
        }
    }
    /// Simo-Taylor (Ogden-Hill) penalty: U = κ/4 * (J²-1-2 ln J).
    pub fn simo_taylor(kappa: f64) -> Self {
        Self {
            kappa,
            penalty_type: PenaltyType::SimoTaylor,
        }
    }
    /// Logarithmic penalty: U = κ/2 * (ln J)².
    pub fn logarithmic(kappa: f64) -> Self {
        Self {
            kappa,
            penalty_type: PenaltyType::Logarithmic,
        }
    }
    /// Evaluate the volumetric strain energy U(J).
    pub fn energy(&self, j: f64) -> f64 {
        match self.penalty_type {
            PenaltyType::Quadratic => self.kappa / 2.0 * (j - 1.0).powi(2),
            PenaltyType::SimoTaylor => self.kappa / 4.0 * (j * j - 1.0 - 2.0 * j.ln()),
            PenaltyType::Logarithmic => self.kappa / 2.0 * j.ln().powi(2),
        }
    }
    /// Hydrostatic pressure p = dU/dJ.
    pub fn pressure(&self, j: f64) -> f64 {
        match self.penalty_type {
            PenaltyType::Quadratic => self.kappa * (j - 1.0),
            PenaltyType::SimoTaylor => self.kappa / 2.0 * (j - 1.0 / j),
            PenaltyType::Logarithmic => self.kappa * j.ln() / j,
        }
    }
    /// Bulk modulus contribution κ_vol = d²U/dJ².
    pub fn bulk_contribution(&self, j: f64) -> f64 {
        match self.penalty_type {
            PenaltyType::Quadratic => self.kappa,
            PenaltyType::SimoTaylor => self.kappa / 2.0 * (1.0 + 1.0 / (j * j)),
            PenaltyType::Logarithmic => self.kappa * (1.0 - j.ln()) / (j * j),
        }
    }
}
