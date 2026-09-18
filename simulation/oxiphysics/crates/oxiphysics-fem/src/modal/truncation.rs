// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Modal truncation: how many modes to keep, and the static correction of the
//! modes that are dropped.
//!
//! The number of modes is chosen from the **effective modal mass** in a given
//! direction, measured against the full rigid-body mass `dᵀ M d` of that direction
//! (not against the mass of whatever modes happen to have been extracted).  The
//! response of the neglected modes to a load can be recovered quasi-statically with
//! the residual flexibility ([`residual_flexibility`]).

use crate::modal::{ModalError, dot, effective_modal_mass, matvec};
use crate::solvers::{jacobi_preconditioner, pcg_solve};
use crate::sparse::CsrMatrix;

/// A direction mass `dᵀ M d` below this is treated as zero.
const MIN_DIRECTION_MASS: f64 = 1e-60;

/// Slack of the comparison `captured fraction ≥ target`: cumulative sums of effective
/// masses differ from `dᵀ M d` by rounding, which must not turn an exactly reached
/// target (for example `1.0` with a complete mode set) into a miss.
const FRACTION_TOLERANCE: f64 = 1e-12;

/// Relative residual to which the static solve `K u = F` of
/// [`residual_flexibility`] is carried.
const STATIC_SOLVE_TOLERANCE: f64 = 1e-12;

/// Configuration for modal truncation.
#[derive(Debug, Clone)]
pub struct ModalTruncationConfig {
    /// Fraction of the total effective mass `dᵀ M d` that must be captured.
    pub mass_capture_fraction: f64,
    /// Maximum number of modes to retain.
    pub max_modes: usize,
    /// Add the static correction (residual flexibility) of the neglected modes
    /// when the response of the truncated model is evaluated, see
    /// [`truncated_static_response`].
    pub static_correction: bool,
}

impl Default for ModalTruncationConfig {
    fn default() -> Self {
        Self {
            mass_capture_fraction: 0.90,
            max_modes: 50,
            static_correction: false,
        }
    }
}

/// Outcome of [`modal_truncation`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModalTruncation {
    /// Number of modes to retain: the smallest number that reaches the target, or
    /// as many as [`ModalTruncationConfig::max_modes`] and the supplied modes
    /// allow when the target cannot be reached.
    pub n_modes: usize,
    /// Fraction of the total mass `dᵀ M d` captured by the first `n_modes` modes
    /// (`Σ_r Γ_r² / m_r / dᵀ M d`; at most `1` for `M`-orthogonal modes).
    pub captured_fraction: f64,
    /// Whether `captured_fraction` reaches
    /// [`ModalTruncationConfig::mass_capture_fraction`].  `false` means that
    /// `max_modes` or the number of supplied modes (or a direction without mass)
    /// limited the selection.
    pub target_met: bool,
}

/// The rigid-body mass `dᵀ M d` of `direction`.
fn direction_mass(mass: &CsrMatrix, direction: &[f64]) -> f64 {
    dot(direction, &matvec(mass, direction))
}

/// Select the minimum number of modes that captures a fraction of the total
/// effective mass in the given direction.
///
/// The effective mass of mode `r` is `m_eff,r = (φ_rᵀ M d)² / (φ_rᵀ M φ_r)`; for a
/// complete set of `M`-orthogonal modes `Σ_r m_eff,r = dᵀ M d`, the full mass moved
/// in direction `d`.  The captured fraction is measured against that total, so a
/// partial mode set reports what it really captures (`captured_fraction < 1`,
/// `target_met = false`) instead of being normalised to its own sum.
///
/// With a direction that carries no mass (`dᵀ M d ≈ 0`) no fraction is defined: as
/// many modes as allowed are kept and `target_met` is `false`.
///
/// # Panics
///
/// Panics if `direction` does not match the size of `mass` (the sparse matrix-vector
/// product asserts it).  A mode shape of another length is combined with the shorter
/// of the two, so every mode shape must have the size of the system.
pub fn modal_truncation(
    mode_shapes: &[Vec<f64>],
    mass: &CsrMatrix,
    direction: &[f64],
    config: &ModalTruncationConfig,
) -> ModalTruncation {
    let limit = config.max_modes.min(mode_shapes.len());
    let total = direction_mass(mass, direction);
    if total.is_nan() || total <= MIN_DIRECTION_MASS {
        return ModalTruncation {
            n_modes: limit,
            captured_fraction: 0.0,
            target_met: false,
        };
    }
    if config.mass_capture_fraction <= FRACTION_TOLERANCE {
        return ModalTruncation {
            n_modes: 0,
            captured_fraction: 0.0,
            target_met: true,
        };
    }
    let masses = effective_modal_mass(&mode_shapes[..limit], mass, direction);
    let mut cumulative = 0.0;
    for (index, m_eff) in masses.iter().enumerate() {
        cumulative += m_eff;
        let captured = cumulative / total;
        if captured + FRACTION_TOLERANCE >= config.mass_capture_fraction {
            return ModalTruncation {
                n_modes: index + 1,
                captured_fraction: captured,
                target_met: true,
            };
        }
    }
    ModalTruncation {
        n_modes: limit,
        captured_fraction: cumulative / total,
        target_met: false,
    }
}

/// Cumulative effective mass fraction as a function of mode count.
///
/// Entry `r` is `Σ_{s ≤ r} m_eff,s / dᵀ M d`: the fraction of the total mass of
/// `direction` captured by the first `r + 1` modes.  A partial mode set therefore
/// ends below `1` (the values are *not* renormalised to the supplied modes), in
/// agreement with [`modal_truncation`].  A direction without mass gives zeros.
pub fn cumulative_mass_fraction(
    mode_shapes: &[Vec<f64>],
    mass: &CsrMatrix,
    direction: &[f64],
) -> Vec<f64> {
    let masses = effective_modal_mass(mode_shapes, mass, direction);
    let total = direction_mass(mass, direction);
    if total.is_nan() || total <= MIN_DIRECTION_MASS {
        return vec![0.0; masses.len()];
    }
    let mut cumulative = 0.0;
    masses
        .iter()
        .map(|&m_eff| {
            cumulative += m_eff;
            cumulative / total
        })
        .collect()
}

/// Static displacement `Σ_r φ_r (φ_rᵀ F)/ω_r²` carried by the given modes
/// (mass-normalised, `ω` in rad/s); rigid-body modes (`ω ≈ 0`) are skipped.
fn modal_static_sum(mode_shapes: &[Vec<f64>], frequencies: &[f64], force: &[f64]) -> Vec<f64> {
    let mut sum = vec![0.0; force.len()];
    for (mode, &omega) in mode_shapes.iter().zip(frequencies.iter()) {
        let omega_sq = omega * omega;
        if omega_sq.is_nan() || omega_sq <= 1e-60 {
            continue;
        }
        let coefficient = dot(mode, force) / omega_sq;
        for (u, phi) in sum.iter_mut().zip(mode.iter()) {
            *u += coefficient * phi;
        }
    }
    sum
}

/// Validate the sizes shared by the static-correction routines and return `n`.
fn check_static_inputs(
    stiffness: &CsrMatrix,
    mode_shapes: &[Vec<f64>],
    frequencies: &[f64],
    force: &[f64],
) -> Result<usize, ModalError> {
    let n = stiffness.nrows;
    if n == 0 {
        return Err(ModalError::EmptySystem);
    }
    if stiffness.ncols != n {
        return Err(ModalError::DimensionMismatch {
            expected: n,
            got: stiffness.ncols,
        });
    }
    if force.len() != n {
        return Err(ModalError::DimensionMismatch {
            expected: n,
            got: force.len(),
        });
    }
    if frequencies.len() != mode_shapes.len() {
        return Err(ModalError::DimensionMismatch {
            expected: mode_shapes.len(),
            got: frequencies.len(),
        });
    }
    if let Some(mode) = mode_shapes.iter().find(|mode| mode.len() != n) {
        return Err(ModalError::DimensionMismatch {
            expected: n,
            got: mode.len(),
        });
    }
    Ok(n)
}

/// Residual flexibility response of the modes that are **not** in the given set,
///
/// `u_res = K⁻¹ F − Σ_r φ_r (φ_rᵀ F) / ω_r²`,
///
/// the static displacement under `force` minus the static part carried by the
/// retained modes `(ω_r, φ_r)`.  Adding it to the modal response of the retained
/// modes (the mode-acceleration / static-correction method) restores the quasi-static
/// contribution of all the neglected high-frequency modes, which converges much
/// faster than adding modes.
///
/// `mode_shapes` must be **mass-normalised** (`φᵀ M φ = 1`) and `frequencies` the
/// natural angular frequencies `ω` (rad/s) of the same modes; rigid-body modes
/// (`ω ≈ 0`) are skipped.  The static solve `K u = F` uses Jacobi-preconditioned CG
/// and therefore needs a symmetric positive definite `K` (a restrained structure).
/// For modes that were extracted but not retained, the same correction without a
/// solve is [`modal_truncation_correction`](crate::dynamic::modal_truncation_correction).
///
/// # Errors
///
/// * [`ModalError::EmptySystem`] for a system without DOF;
/// * [`ModalError::DimensionMismatch`] if `K` is not square, `force` or a mode shape
///   has the wrong length or `frequencies` and `mode_shapes` differ in length;
/// * [`ModalError::StaticSolveFailed`] if the static solve does not converge
///   (singular or indefinite `K`).
pub fn residual_flexibility(
    stiffness: &CsrMatrix,
    mode_shapes: &[Vec<f64>],
    frequencies: &[f64],
    force: &[f64],
) -> Result<Vec<f64>, ModalError> {
    let n = check_static_inputs(stiffness, mode_shapes, frequencies, force)?;
    let preconditioner = jacobi_preconditioner(stiffness);
    let (static_response, stats) = pcg_solve(
        stiffness,
        force,
        &preconditioner,
        20 * n + 100,
        STATIC_SOLVE_TOLERANCE,
    );
    let force_norm = dot(force, force).sqrt();
    let relative_residual = if force_norm > 0.0 {
        stats.residual / force_norm
    } else {
        0.0
    };
    if !stats.converged || !static_response.iter().all(|u| u.is_finite()) {
        return Err(ModalError::StaticSolveFailed {
            residual: relative_residual,
        });
    }
    let modal = modal_static_sum(mode_shapes, frequencies, force);
    Ok(static_response
        .iter()
        .zip(modal.iter())
        .map(|(u, m)| u - m)
        .collect())
}

/// Static displacement under `force` of the modal model truncated to its first
/// `n_keep` modes.
///
/// The retained modes carry `Σ_{r < n_keep} φ_r (φ_rᵀ F)/ω_r²`.  With
/// [`ModalTruncationConfig::static_correction`] the residual flexibility vector
/// ([`residual_flexibility`]) of the dropped modes is added, which makes the result
/// the exact static solution `K⁻¹ F`; without it the dropped modes contribute
/// nothing and the displacement is underestimated.  `mode_shapes` are
/// mass-normalised, `frequencies` are `ω` in rad/s; `n_keep` is clamped to the
/// number of supplied modes.
///
/// # Errors
///
/// The errors of [`residual_flexibility`] (the sizes are checked in either case).
pub fn truncated_static_response(
    stiffness: &CsrMatrix,
    mode_shapes: &[Vec<f64>],
    frequencies: &[f64],
    force: &[f64],
    n_keep: usize,
    config: &ModalTruncationConfig,
) -> Result<Vec<f64>, ModalError> {
    check_static_inputs(stiffness, mode_shapes, frequencies, force)?;
    let n_keep = n_keep.min(mode_shapes.len());
    let modal = modal_static_sum(&mode_shapes[..n_keep], &frequencies[..n_keep], force);
    if !config.static_correction {
        return Ok(modal);
    }
    let residual = residual_flexibility(
        stiffness,
        &mode_shapes[..n_keep],
        &frequencies[..n_keep],
        force,
    )?;
    Ok(modal
        .iter()
        .zip(residual.iter())
        .map(|(m, r)| m + r)
        .collect())
}
