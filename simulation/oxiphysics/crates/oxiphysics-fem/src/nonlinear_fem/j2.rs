// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! J2 (von Mises) plasticity with combined isotropic and Armstrong–Frederick
//! kinematic hardening: backward-Euler return mapping and its consistent
//! (algorithmic) tangent.
//!
//! # Conventions
//!
//! Tensors are stored in Voigt form `[11, 22, 33, 12, 23, 13]`.  Shear entries of
//! *stress-like* tensors (stress, deviator, back stress) are tensor components,
//! shear entries of *strain-like* tensors (strain and plastic-strain increments)
//! are engineering shears `γ = 2ε`; the elastic law therefore reads `σ₁₂ = G·γ₁₂`.
//! The norm `‖a‖ = √(3/2 · a:a)` of a stress-like deviator is the von Mises
//! equivalent stress.
//!
//! # Model
//!
//! * yield function `f = ‖s − α‖ − σ_y(ε̄ᵖ)` with the linear isotropic law
//!   `σ_y = σ_y0 + H·ε̄ᵖ`,
//! * associative flow `ε̇ᵖ = λ̇ · (3/2)(s − α)/‖s − α‖`, so that `ε̄̇ᵖ = λ̇`,
//! * Armstrong–Frederick back stress `α̇ = C·λ̇·(s − α)/‖s − α‖ − γ·λ̇·α`.
//!
//! `γ = 0` is linear (Prager) kinematic hardening with modulus `C`; for `γ > 0` the
//! back stress saturates at `‖α‖ = C/γ`.

/// Tensor double contraction `a : b` of two symmetric stress-like tensors in Voigt
/// storage (shear entries are tensor components, so each shear product counts twice).
fn contract(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + 2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5])
}

/// J2 plasticity state variables at a material point.
#[derive(Debug, Clone, Default)]
pub struct PlasticState {
    /// Plastic strain tensor (Voigt: \[ε11, ε22, ε33, γ12, γ23, γ13\])
    pub eps_p: [f64; 6],
    /// Equivalent plastic strain (accumulated)
    pub eps_p_eq: f64,
    /// Back stress tensor (kinematic hardening)
    pub alpha: [f64; 6],
}

impl PlasticState {
    /// Create a new zero-initialized plastic state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the equivalent plastic strain `√(2/3 · Δεᵖ:Δεᵖ)` of a plastic strain
    /// increment (Voigt, engineering shear) to the accumulated `eps_p_eq`.
    pub fn update_eps_p_eq(&mut self, delta_eps_p: &[f64; 6]) {
        let norm2: f64 = delta_eps_p[0].powi(2)
            + delta_eps_p[1].powi(2)
            + delta_eps_p[2].powi(2)
            + 0.5 * (delta_eps_p[3].powi(2) + delta_eps_p[4].powi(2) + delta_eps_p[5].powi(2));
        self.eps_p_eq += (2.0 / 3.0 * norm2).sqrt();
    }
}

/// J2 isotropic + kinematic hardening model (Chaboche / Armstrong–Frederick).
#[derive(Debug, Clone)]
pub struct J2Material {
    /// Young's modulus \[Pa\]
    pub young: f64,
    /// Poisson's ratio
    pub nu: f64,
    /// Initial yield stress \[Pa\]
    pub sigma_y0: f64,
    /// Isotropic hardening modulus H \[Pa\]
    pub h_iso: f64,
    /// Kinematic hardening modulus C \[Pa\]
    pub c_kin: f64,
    /// Dynamic-recovery (saturation) parameter γ of the Armstrong–Frederick back
    /// stress \[1/strain\]; `0` gives linear Prager hardening.
    pub gamma_kin: f64,
}

impl J2Material {
    /// Create a new J2 material.
    pub fn new(young: f64, nu: f64, sigma_y0: f64, h_iso: f64) -> Self {
        Self {
            young,
            nu,
            sigma_y0,
            h_iso,
            c_kin: 0.0,
            gamma_kin: 0.0,
        }
    }

    /// Shear modulus G = E / (2(1+ν)).
    #[inline]
    pub fn shear_modulus(&self) -> f64 {
        self.young / (2.0 * (1.0 + self.nu))
    }

    /// Bulk modulus K = E / (3(1−2ν)).
    #[inline]
    pub fn bulk_modulus(&self) -> f64 {
        self.young / (3.0 * (1.0 - 2.0 * self.nu))
    }

    /// Current yield stress at equivalent plastic strain.
    pub fn yield_stress(&self, eps_p_eq: f64) -> f64 {
        self.sigma_y0 + self.h_iso * eps_p_eq
    }

    /// Deviatoric part of a stress tensor (Voigt).
    pub fn deviatoric(sigma: &[f64; 6]) -> [f64; 6] {
        let p = (sigma[0] + sigma[1] + sigma[2]) / 3.0;
        [
            sigma[0] - p,
            sigma[1] - p,
            sigma[2] - p,
            sigma[3],
            sigma[4],
            sigma[5],
        ]
    }

    /// Von Mises stress from Voigt tensor.
    pub fn von_mises(sigma: &[f64; 6]) -> f64 {
        let s = Self::deviatoric(sigma);
        let j2 = 0.5 * (s[0].powi(2) + s[1].powi(2) + s[2].powi(2))
            + s[3].powi(2)
            + s[4].powi(2)
            + s[5].powi(2);
        (3.0 * j2).sqrt()
    }

    /// Elastic predictor step (return the trial stress).
    pub fn elastic_predictor(&self, sigma_n: &[f64; 6], delta_eps: &[f64; 6]) -> [f64; 6] {
        let g = self.shear_modulus();
        let k = self.bulk_modulus();
        let lam = k - 2.0 / 3.0 * g; // Lamé λ

        let de_vol = delta_eps[0] + delta_eps[1] + delta_eps[2];

        [
            sigma_n[0] + lam * de_vol + 2.0 * g * delta_eps[0],
            sigma_n[1] + lam * de_vol + 2.0 * g * delta_eps[1],
            sigma_n[2] + lam * de_vol + 2.0 * g * delta_eps[2],
            sigma_n[3] + g * delta_eps[3],
            sigma_n[4] + g * delta_eps[4],
            sigma_n[5] + g * delta_eps[5],
        ]
    }

    /// Vector `ζ(Δγ) = s_tr − α_n / (1 + γΔγ)` along which the deviatoric
    /// relative stress `s − α` (and with it the flow direction) lies after the return.
    ///
    /// For `γ = 0`, or a vanishing back stress, `ζ` is the trial relative stress
    /// `s_tr − α_n` and the return is radial.  For `γ > 0` the dynamic-recovery term
    /// rotates the flow direction towards `s_tr` as `Δγ` grows.
    fn relaxed_trial(&self, s_tr: &[f64; 6], alpha_n: &[f64; 6], delta_gamma: f64) -> [f64; 6] {
        let denom = 1.0 + self.gamma_kin.max(0.0) * delta_gamma;
        std::array::from_fn(|i| s_tr[i] - alpha_n[i] / denom)
    }

    /// Plastic consistency parameter `Δγ` of the backward-Euler return.
    ///
    /// Eliminating the stress and the back stress from the discrete flow rule gives
    /// the scalar yield condition
    ///
    /// `g(Δγ) = ‖ζ(Δγ)‖ − Δγ·(3G + C/(1 + γΔγ)) − σ_y(ε̄ᵖ_n + Δγ) = 0`.
    ///
    /// For `γ = 0` it is linear and solved in closed form,
    /// `Δγ = φ_tr / (3G + H + C)`; otherwise it is solved by a Newton iteration
    /// safeguarded with bisection on a bracket `[lo, hi]` with `g(lo) > 0 ≥ g(hi)`.
    fn consistency_parameter(
        &self,
        s_tr: &[f64; 6],
        alpha_n: &[f64; 6],
        eps_p_eq_n: f64,
        phi_tr: f64,
    ) -> f64 {
        let g = self.shear_modulus();
        let (c, h) = (self.c_kin, self.h_iso);
        let gamma = self.gamma_kin.max(0.0);
        let prager = phi_tr / (3.0 * g + h + c);
        if gamma == 0.0 {
            return prager;
        }

        let residual = |dg: f64| -> f64 {
            let zeta = self.relaxed_trial(s_tr, alpha_n, dg);
            (1.5 * contract(&zeta, &zeta)).sqrt()
                - dg * (3.0 * g + c / (1.0 + gamma * dg))
                - self.yield_stress(eps_p_eq_n + dg)
        };
        let slope = |dg: f64| -> f64 {
            let denom = 1.0 + gamma * dg;
            let zeta = self.relaxed_trial(s_tr, alpha_n, dg);
            let zeta_norm = (1.5 * contract(&zeta, &zeta)).sqrt();
            let d_zeta: [f64; 6] = std::array::from_fn(|i| alpha_n[i] * gamma / (denom * denom));
            let d_norm = if zeta_norm > 0.0 {
                1.5 * contract(&zeta, &d_zeta) / zeta_norm
            } else {
                0.0
            };
            d_norm - 3.0 * g - c / (denom * denom) - h
        };

        // Bracket the root: g(0) = φ_tr > 0 and g → −∞ for large Δγ.
        let mut lo = 0.0_f64;
        let mut hi = prager;
        for _ in 0..64 {
            if residual(hi) <= 0.0 {
                break;
            }
            lo = hi;
            hi *= 2.0;
        }

        let scale = (self.yield_stress(eps_p_eq_n).abs() + phi_tr.abs()).max(f64::MIN_POSITIVE);
        let tol = 1e-13 * scale;
        let mut x = if lo > 0.0 { lo } else { prager };
        for _ in 0..100 {
            let r = residual(x);
            if r.abs() <= tol {
                break;
            }
            if r > 0.0 {
                lo = x;
            } else {
                hi = x;
            }
            if hi - lo <= 4.0 * f64::EPSILON * hi {
                x = 0.5 * (lo + hi);
                break;
            }
            let d = slope(x);
            let newton = x - r / d;
            x = if d != 0.0 && newton.is_finite() && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
        }
        x
    }

    /// Backward-Euler radial / Armstrong–Frederick return mapping.
    ///
    /// Given the elastic trial stress `sigma_tr` and the converged state `state_n`
    /// the function checks the yield condition and, if violated, returns the
    /// stress on the updated yield surface:
    ///
    /// * `s_{n+1} = s_tr − 3GΔγ·n̂`,
    /// * `Δεᵖ = (3/2)Δγ·n̂` (engineering shear for the shear entries),
    /// * `ε̄ᵖ_{n+1} = ε̄ᵖ_n + Δγ`,
    /// * `α_{n+1} = (α_n + CΔγ·n̂) / (1 + γΔγ)`,
    ///
    /// with the unit flow direction `n̂ = ζ/‖ζ‖` (see `relaxed_trial`).  For isotropic
    /// hardening and linear Prager hardening (`γ = 0`) the return is radial,
    /// `n̂ = (s_tr − α_n)/‖s_tr − α_n‖`, and `Δγ = φ_tr / (3G + H + C)`.
    ///
    /// Returns `(sigma_n1, state_n1, delta_gamma)`; `delta_gamma` is zero for an
    /// elastic step.
    pub fn return_mapping(
        &self,
        sigma_tr: &[f64; 6],
        state_n: &PlasticState,
    ) -> ([f64; 6], PlasticState, f64) {
        let g = self.shear_modulus();
        let s_tr = Self::deviatoric(sigma_tr);
        // The back stress is a deviator by construction; project defensively.
        let alpha_n = Self::deviatoric(&state_n.alpha);

        let eta_tr: [f64; 6] = std::array::from_fn(|i| s_tr[i] - alpha_n[i]);
        let xi_tr = (1.5 * contract(&eta_tr, &eta_tr)).sqrt();
        let phi_tr = xi_tr - self.yield_stress(state_n.eps_p_eq);
        if phi_tr <= 0.0 {
            // Elastic step
            return (*sigma_tr, state_n.clone(), 0.0);
        }

        let delta_gamma = self.consistency_parameter(&s_tr, &alpha_n, state_n.eps_p_eq, phi_tr);

        // Unit flow direction n̂ = ζ/‖ζ‖ (‖n̂‖ = 1 in the von Mises norm).
        let zeta = self.relaxed_trial(&s_tr, &alpha_n, delta_gamma);
        let zeta_norm = (1.5 * contract(&zeta, &zeta)).sqrt().max(f64::MIN_POSITIVE);
        let n_hat: [f64; 6] = std::array::from_fn(|i| zeta[i] / zeta_norm);

        // Plastic strain: Δεᵖ = (3/2)Δγ n̂, shear entries in engineering form.
        let eps_p: [f64; 6] = std::array::from_fn(|i| {
            let tensor_factor = if i < 3 { 1.5 } else { 3.0 };
            state_n.eps_p[i] + tensor_factor * delta_gamma * n_hat[i]
        });

        // Stress: the plastic flow only changes the deviator.
        let p_tr = (sigma_tr[0] + sigma_tr[1] + sigma_tr[2]) / 3.0;
        let sigma_n1: [f64; 6] = std::array::from_fn(|i| {
            let volumetric = if i < 3 { p_tr } else { 0.0 };
            s_tr[i] - 3.0 * g * delta_gamma * n_hat[i] + volumetric
        });

        // Armstrong–Frederick back stress (backward Euler).
        let recovery = 1.0 + self.gamma_kin.max(0.0) * delta_gamma;
        let alpha_n1: [f64; 6] =
            std::array::from_fn(|i| (alpha_n[i] + self.c_kin * delta_gamma * n_hat[i]) / recovery);

        let new_state = PlasticState {
            eps_p,
            eps_p_eq: state_n.eps_p_eq + delta_gamma,
            alpha: alpha_n1,
        };
        (sigma_n1, new_state, delta_gamma)
    }

    /// Elastic tangent `∂σ/∂ε` in Voigt form (stress shear = tensor, strain shear =
    /// engineering).
    fn elastic_tangent(&self) -> [[f64; 6]; 6] {
        let g = self.shear_modulus();
        let k = self.bulk_modulus();
        let lam = k - 2.0 / 3.0 * g;

        let mut c = [[0.0f64; 6]; 6];
        for (i, row) in c.iter_mut().enumerate().take(3) {
            for cell in row.iter_mut().take(3) {
                *cell = lam;
            }
            row[i] += 2.0 * g;
        }
        for (i, row) in c.iter_mut().enumerate().skip(3) {
            row[i] = g;
        }
        c
    }

    /// Consistent (algorithmic) tangent `∂σ_{n+1}/∂ε_{n+1}` in Voigt notation (6×6).
    ///
    /// `delta_gamma` is the consistency parameter returned by
    /// [`return_mapping`](Self::return_mapping) for the same `sigma_tr` and
    /// `state_n`; for `delta_gamma <= 0` the elastic tangent is returned.  The tangent
    /// is the exact derivative of the discrete return map, including isotropic and
    /// Armstrong–Frederick kinematic hardening:
    ///
    /// `δs_{n+1} = θ·δs_tr + w·δΔγ`, `δΔγ = (3/2)·n̂:δs_tr / D̃`,
    ///
    /// with `θ = 1 − 3GΔγ/‖ζ‖`, `D = 3G + C/(1+γΔγ)² + H`, `a = γα_n/(1+γΔγ)²`,
    /// `D̃ = D − (3/2)·n̂:a` and
    /// `w = −3G·n̂·(1 − Δγ·D/‖ζ‖) − 3GΔγ/‖ζ‖·a`.  It is symmetric for `γ = 0`
    /// (isotropic and Prager hardening) and unsymmetric in general.
    pub fn consistent_tangent(
        &self,
        delta_gamma: f64,
        sigma_tr: &[f64; 6],
        state_n: &PlasticState,
    ) -> [[f64; 6]; 6] {
        let c_el = self.elastic_tangent();
        if delta_gamma <= 0.0 {
            return c_el;
        }

        let g = self.shear_modulus();
        let k = self.bulk_modulus();
        let (c, h) = (self.c_kin, self.h_iso);
        let gamma = self.gamma_kin.max(0.0);

        let s_tr = Self::deviatoric(sigma_tr);
        let alpha_n = Self::deviatoric(&state_n.alpha);
        let denom = 1.0 + gamma * delta_gamma;
        let zeta = self.relaxed_trial(&s_tr, &alpha_n, delta_gamma);
        let zeta_norm = (1.5 * contract(&zeta, &zeta)).sqrt().max(f64::MIN_POSITIVE);
        let n_hat: [f64; 6] = std::array::from_fn(|i| zeta[i] / zeta_norm);
        let a: [f64; 6] = std::array::from_fn(|i| alpha_n[i] * gamma / (denom * denom));

        let d_big = 3.0 * g + c / (denom * denom) + h;
        let d_tilde = (d_big - 1.5 * contract(&n_hat, &a)).max(f64::EPSILON * d_big);
        let theta = 1.0 - 3.0 * g * delta_gamma / zeta_norm;
        let w: [f64; 6] = std::array::from_fn(|i| {
            -3.0 * g * n_hat[i] * (1.0 - delta_gamma * d_big / zeta_norm)
                - 3.0 * g * delta_gamma / zeta_norm * a[i]
        });

        let mut tangent = [[0.0f64; 6]; 6];
        for j in 0..6 {
            // Response of the trial deviator to a unit strain component j.
            let d_p = if j < 3 { k } else { 0.0 };
            let ds_tr: [f64; 6] =
                std::array::from_fn(|i| c_el[i][j] - if i < 3 { d_p } else { 0.0 });
            let d_delta_gamma = 1.5 * contract(&n_hat, &ds_tr) / d_tilde;
            for i in 0..6 {
                let volumetric = if i < 3 { d_p } else { 0.0 };
                tangent[i][j] = theta * ds_tr[i] + w[i] * d_delta_gamma + volumetric;
            }
        }
        tangent
    }
}

#[cfg(test)]
mod tests;
