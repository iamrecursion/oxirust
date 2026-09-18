// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Computational homogenization of a voxelised periodic RVE.
//!
//! The six unit macroscopic strains `ε̄ = e_m` are applied to a periodic
//! representative volume element given as a voxel phase map.  For every load case
//! the periodic Lippmann-Schwinger equation
//!
//! ```text
//! ε(x) = ε̄ − Γ⁰ ∗ [(C(x) − C⁰) : ε](x)
//! ```
//!
//! is solved in Fourier space with an isotropic reference medium `C⁰` — the
//! Moulinec-Suquet basic scheme or the accelerated polarisation scheme of
//! `oxiphysics_materials::homogenization`, both built on OxiFFT — and the volume
//! average of the micro stress, `⟨σ⟩ = ⟨C : ε⟩`, fills column `m` of the effective
//! stiffness.  Periodicity is intrinsic to the FFT, so no explicit boundary
//! conditions are needed.
//!
//! # Conventions
//!
//! Voigt order `[xx, yy, zz, yz, xz, xy]` with engineering shear strain, i.e. that
//! of [`Phase::stiffness_voigt`] (the `multiscale_fem` module uses the different
//! order `[xx, yy, zz, xy, yz, xz]`).  Voxel `(ix, iy, iz)` of an `nx × ny × nz`
//! grid is stored at the flat index `ix·ny·nz + iy·nz + iz`.

use super::bounds::Phase;
use crate::error::{Error, Result};
use oxiphysics_materials::homogenization::{Scheme, compute_c_eff_with};

/// A periodic RVE discretised into `nx × ny × nz` voxels; every voxel carries the
/// index of its phase in the `&[Phase]` list passed to the homogenization driver.
///
/// The voxel map alone determines the volume fractions of the phases: the
/// `volume_fraction` field of [`Phase`] is not used by the voxel homogenization.
///
/// The RVE is a *cube*: its edges have the same length whatever the grid
/// dimensions, as in the FFT kernels of `oxiphysics-materials`, so unequal
/// `nx`, `ny`, `nz` describe voxels of different aspect ratio (a laminate sampled
/// with `4 × 4 × 8` voxels has two layers of half the edge length each).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxelRve {
    dims: [usize; 3],
    phase_index: Vec<usize>,
}

/// Number of voxels of a grid, or an error if the grid is empty or overflows.
fn voxel_count_of(dims: [usize; 3]) -> Result<usize> {
    let [nx, ny, nz] = dims;
    nx.checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .filter(|&v| v > 0)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "the RVE grid {nx} x {ny} x {nz} has no voxels or overflows"
            ))
        })
}

impl VoxelRve {
    /// Create an RVE from a flat phase-index array of length `nx·ny·nz`
    /// (voxel `(ix, iy, iz)` at `ix·ny·nz + iy·nz + iz`).
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the grid is empty or the array has the wrong
    /// length.
    pub fn new(dims: [usize; 3], phase_index: Vec<usize>) -> Result<Self> {
        let expected = voxel_count_of(dims)?;
        if phase_index.len() != expected {
            return Err(Error::InvalidInput(format!(
                "{} phase indices were given for a {} x {} x {} grid of {expected} voxels",
                phase_index.len(),
                dims[0],
                dims[1],
                dims[2]
            )));
        }
        Ok(Self { dims, phase_index })
    }

    /// Create an RVE by evaluating `phase_of(ix, iy, iz)` for every voxel.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if the grid is empty.
    pub fn from_fn(
        dims: [usize; 3],
        phase_of: impl Fn(usize, usize, usize) -> usize,
    ) -> Result<Self> {
        let total = voxel_count_of(dims)?;
        let mut phase_index = Vec::with_capacity(total);
        for ix in 0..dims[0] {
            for iy in 0..dims[1] {
                for iz in 0..dims[2] {
                    phase_index.push(phase_of(ix, iy, iz));
                }
            }
        }
        Self::new(dims, phase_index)
    }

    /// Grid dimensions `[nx, ny, nz]`.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Phase index of every voxel (flat, `ix·ny·nz + iy·nz + iz`).
    pub fn phase_index(&self) -> &[usize] {
        &self.phase_index
    }

    /// Total number of voxels.
    pub fn voxel_count(&self) -> usize {
        self.phase_index.len()
    }

    /// Volume fraction of each of the phases `0..n_phases` (voxels whose index is
    /// not below `n_phases` are not counted).
    pub fn volume_fractions(&self, n_phases: usize) -> Vec<f64> {
        let mut counts = vec![0_usize; n_phases];
        for &phase in &self.phase_index {
            if let Some(count) = counts.get_mut(phase) {
                *count += 1;
            }
        }
        let total = self.phase_index.len() as f64;
        counts.into_iter().map(|c| c as f64 / total).collect()
    }
}

/// Iteration scheme of the spectral solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftScheme {
    /// Moulinec-Suquet basic fixed-point scheme with a reference medium stiffer
    /// than every phase (robust, `O(contrast)` iterations).
    Basic,
    /// Accelerated polarisation scheme with the geometric mean of the extreme
    /// phase moduli as reference medium (`O(√contrast)` iterations).
    Accelerated,
}

/// Options of the spectral homogenization solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FftHomogenizationOptions {
    /// Iteration scheme.
    pub scheme: FftScheme,
    /// Relative tolerance on the equilibrium residual `‖ξ·σ̂‖/‖σ̄‖` of every load
    /// case.
    pub tolerance: f64,
    /// Iteration budget per load case.
    pub max_iterations: usize,
}

impl Default for FftHomogenizationOptions {
    fn default() -> Self {
        Self {
            scheme: FftScheme::Accelerated,
            tolerance: 1.0e-8,
            max_iterations: 2000,
        }
    }
}

/// Check that `phase` is a stable isotropic material (`E > 0`, `−1 < ν < 1/2`).
fn validate_phase(index: usize, phase: &Phase) -> Result<()> {
    let e = phase.youngs_modulus;
    let nu = phase.poisson_ratio;
    if e.is_finite() && e > 0.0 && nu.is_finite() && nu > -1.0 && nu < 0.5 {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!(
            "phase {index} ({}) needs E > 0 and -1 < nu < 0.5, got E = {e:e}, nu = {nu}",
            phase.name
        )))
    }
}

/// Lamé parameters `(λ⁰, μ⁰)` of the isotropic reference medium from the extreme
/// bulk (`k_lo`, `k_hi`) and shear (`g_lo`, `g_hi`) moduli of the phases present.
fn reference_medium(scheme: FftScheme, k_lo: f64, k_hi: f64, g_lo: f64, g_hi: f64) -> (f64, f64) {
    let (bulk0, shear0) = match scheme {
        // Stiffer than every phase in bulk and shear: the fixed-point map contracts.
        FftScheme::Basic => (1.01 * k_hi, 1.01 * g_hi),
        // Geometric mean of the extreme moduli: O(√contrast) convergence.
        FftScheme::Accelerated => ((k_lo * k_hi).sqrt(), (g_lo * g_hi).sqrt()),
    };
    (bulk0 - 2.0 * shear0 / 3.0, shear0)
}

/// Computational homogenization of a voxelised periodic RVE by six virtual
/// unit-strain load cases.
///
/// Each load case `ε̄ = e_m` is solved on the periodic voxel grid with the
/// FFT-based Lippmann-Schwinger solver selected by `options`; the volume average of
/// the resulting stress field is column `m` of the returned effective stiffness
/// (6×6, Voigt order `[xx, yy, zz, yz, xz, xy]`, engineering shear strain).
///
/// A homogeneous RVE returns the stiffness of its phase, and a laminate whose
/// layers coincide with voxel layers returns the exact laminate (Backus) stiffness.
///
/// # Arguments
/// * `rve` – voxel phase map; entry `i` indexes `phases`
/// * `phases` – isotropic constituents (`Phase::volume_fraction` is ignored: the
///   voxel map fixes the fractions)
/// * `options` – scheme, tolerance and iteration budget
///
/// # Errors
/// [`Error::InvalidInput`] for an empty phase list, a voxel whose phase index is
/// out of range, a phase that is not a stable isotropic material or invalid
/// options; [`Error::General`] if a load case does not reach the tolerance within
/// the iteration budget or the solver diverges.
pub fn compute_effective_stiffness_from_unit_strains(
    rve: &VoxelRve,
    phases: &[Phase],
    options: &FftHomogenizationOptions,
) -> Result<[[f64; 6]; 6]> {
    if phases.is_empty() {
        return Err(Error::InvalidInput("at least one phase is required".into()));
    }
    if !(options.tolerance.is_finite() && options.tolerance > 0.0) || options.max_iterations == 0 {
        return Err(Error::InvalidInput(
            "the tolerance must be positive and the iteration budget non-zero".into(),
        ));
    }

    let mut used = vec![false; phases.len()];
    for &index in rve.phase_index() {
        match used.get_mut(index) {
            Some(flag) => *flag = true,
            None => {
                return Err(Error::InvalidInput(format!(
                    "a voxel refers to phase {index} but only {} phases were given",
                    phases.len()
                )));
            }
        }
    }

    let mut stiffness = vec![[[0.0_f64; 6]; 6]; phases.len()];
    let (mut k_lo, mut k_hi) = (f64::INFINITY, 0.0_f64);
    let (mut g_lo, mut g_hi) = (f64::INFINITY, 0.0_f64);
    for (index, (phase, &is_used)) in phases.iter().zip(used.iter()).enumerate() {
        if !is_used {
            continue;
        }
        validate_phase(index, phase)?;
        stiffness[index] = phase.stiffness_voigt();
        k_lo = k_lo.min(phase.bulk_modulus());
        k_hi = k_hi.max(phase.bulk_modulus());
        g_lo = g_lo.min(phase.shear_modulus());
        g_hi = g_hi.max(phase.shear_modulus());
    }
    let c_field: Vec<[[f64; 6]; 6]> = rve
        .phase_index()
        .iter()
        .map(|&index| stiffness[index])
        .collect();

    let reference = reference_medium(options.scheme, k_lo, k_hi, g_lo, g_hi);
    let scheme = match options.scheme {
        FftScheme::Basic => Scheme::Basic,
        FftScheme::Accelerated => Scheme::Accelerated,
    };
    let result = compute_c_eff_with(
        &c_field,
        rve.dims(),
        reference,
        options.tolerance,
        options.max_iterations,
        scheme,
    )
    .map_err(|err| Error::General(format!("FFT homogenization failed: {err}")))?;
    if !result.converged {
        return Err(Error::General(format!(
            "FFT homogenization did not reach the tolerance {:e} within {} iterations per \
             load case (iterations used: {:?})",
            options.tolerance, options.max_iterations, result.iterations
        )));
    }
    Ok(result.c_eff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::homogenization::{
        effective_bulk_modulus, effective_youngs_modulus, hashin_shtrikman_bounds, hill_average,
        mat6_symmetry_error, reuss_average, voigt_average,
    };

    fn soft() -> Phase {
        Phase::new("soft", 0.5, 10.0e9, 0.30)
    }

    fn stiff() -> Phase {
        Phase::new("stiff", 0.5, 60.0e9, 0.25)
    }

    fn tight() -> FftHomogenizationOptions {
        FftHomogenizationOptions {
            scheme: FftScheme::Accelerated,
            tolerance: 1.0e-11,
            max_iterations: 5000,
        }
    }

    fn max_relative_difference(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> f64 {
        let scale = b.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
            / scale
    }

    /// Exact stiffness (Backus average) of a laminate of isotropic layers stacked
    /// along z, in the Voigt order `[xx, yy, zz, yz, xz, xy]`.
    fn backus_laminate(layers: &[(f64, &Phase)]) -> [[f64; 6]; 6] {
        let mean = |f: &dyn Fn(f64, f64) -> f64| -> f64 {
            layers
                .iter()
                .map(|(w, p)| w * f(p.lame_lambda(), p.shear_modulus()))
                .sum()
        };
        let inv_m = mean(&|l, m| 1.0 / (l + 2.0 * m));
        let l_over_m = mean(&|l, m| l / (l + 2.0 * m));
        let mu_mean = mean(&|_, m| m);
        let inv_mu = mean(&|_, m| 1.0 / m);
        let c11 = mean(&|l, m| 4.0 * m * (l + m) / (l + 2.0 * m)) + l_over_m * l_over_m / inv_m;
        let c12 = mean(&|l, m| 2.0 * l * m / (l + 2.0 * m)) + l_over_m * l_over_m / inv_m;
        let c13 = l_over_m / inv_m;
        let c33 = 1.0 / inv_m;
        let c44 = 1.0 / inv_mu;
        let mut c = [[0.0_f64; 6]; 6];
        c[0][0] = c11;
        c[1][1] = c11;
        c[0][1] = c12;
        c[1][0] = c12;
        c[0][2] = c13;
        c[2][0] = c13;
        c[1][2] = c13;
        c[2][1] = c13;
        c[2][2] = c33;
        c[3][3] = c44;
        c[4][4] = c44;
        c[5][5] = mu_mean;
        c
    }

    #[test]
    fn homogeneous_rve_returns_the_phase_stiffness() {
        let phase = Phase::new("steel", 1.0, 210.0e9, 0.3);
        let rve = VoxelRve::from_fn([4, 4, 4], |_, _, _| 0).expect("valid grid");
        let c = compute_effective_stiffness_from_unit_strains(
            &rve,
            std::slice::from_ref(&phase),
            &tight(),
        )
        .expect("homogeneous solve");
        assert!(max_relative_difference(&c, &phase.stiffness_voigt()) < 1e-9);
    }

    #[test]
    fn laminate_matches_the_backus_stiffness_for_both_schemes() {
        // 50/50 laminate, layers normal to z: four soft then four stiff voxel layers.
        let phases = [soft(), stiff()];
        let rve =
            VoxelRve::from_fn([4, 4, 8], |_, _, iz| usize::from(iz >= 4)).expect("valid grid");
        let exact = backus_laminate(&[(0.5, &phases[0]), (0.5, &phases[1])]);
        for scheme in [FftScheme::Accelerated, FftScheme::Basic] {
            let options = FftHomogenizationOptions {
                scheme,
                tolerance: 1.0e-12,
                max_iterations: 20_000,
            };
            let c = compute_effective_stiffness_from_unit_strains(&rve, &phases, &options)
                .expect("laminate solve");
            let err = max_relative_difference(&c, &exact);
            assert!(
                err < 1e-7,
                "{scheme:?}: error {err:e}\n{c:?}\nvs\n{exact:?}"
            );
        }
        // The result is not the Hill average that the old driver returned.
        let hill = hill_average(&phases);
        assert!(max_relative_difference(&exact, &hill) > 0.01);
    }

    #[test]
    fn laminate_normal_to_x_is_the_rotated_backus_stiffness() {
        // The same laminate stacked along x: indices permute (x <-> z) in the Voigt
        // matrix, which catches a mixed-up Voigt convention.
        let phases = [soft(), stiff()];
        let rve =
            VoxelRve::from_fn([8, 4, 4], |ix, _, _| usize::from(ix >= 4)).expect("valid grid");
        let c = compute_effective_stiffness_from_unit_strains(&rve, &phases, &tight())
            .expect("laminate solve");
        let z = backus_laminate(&[(0.5, &phases[0]), (0.5, &phases[1])]);
        // Voigt permutation x <-> z: [xx,yy,zz,yz,xz,xy] -> [zz,yy,xx,xy,xz,yz].
        let perm = [2_usize, 1, 0, 5, 4, 3];
        let mut expected = [[0.0_f64; 6]; 6];
        for (i, row) in expected.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = z[perm[i]][perm[j]];
            }
        }
        let err = max_relative_difference(&c, &expected);
        assert!(err < 1e-7, "error {err:e}");
    }

    #[test]
    fn effective_stiffness_is_symmetric_and_between_reuss_and_voigt() {
        // Spherical stiff inclusion in a soft matrix on a 12^3 grid.
        let (matrix, inclusion) = (
            Phase::new("matrix", 0.0, 3.5e9, 0.35),
            Phase::new("glass", 0.0, 70.0e9, 0.20),
        );
        let n = 12_usize;
        let centre = (n as f64 - 1.0) / 2.0;
        let rve = VoxelRve::from_fn([n, n, n], |ix, iy, iz| {
            let d2 = (ix as f64 - centre).powi(2)
                + (iy as f64 - centre).powi(2)
                + (iz as f64 - centre).powi(2);
            usize::from(d2 <= 16.0)
        })
        .expect("valid grid");
        let fractions = rve.volume_fractions(2);
        assert!((fractions[0] + fractions[1] - 1.0).abs() < 1e-15);
        assert!(fractions[1] > 0.1 && fractions[1] < 0.3, "{fractions:?}");
        let phases = [matrix, inclusion];
        let options = FftHomogenizationOptions {
            tolerance: 1.0e-9,
            ..FftHomogenizationOptions::default()
        };
        let c = compute_effective_stiffness_from_unit_strains(&rve, &phases, &options)
            .expect("inclusion solve");
        let norm = c.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(mat6_symmetry_error(&c) < 1e-6 * norm, "not symmetric");

        let weighted = [
            Phase::new("matrix", fractions[0], 3.5e9, 0.35),
            Phase::new("glass", fractions[1], 70.0e9, 0.20),
        ];
        let e_eff = effective_youngs_modulus(&c);
        let e_voigt = effective_youngs_modulus(&voigt_average(&weighted));
        let e_reuss = effective_youngs_modulus(&reuss_average(&weighted));
        assert!(
            e_reuss < e_eff && e_eff < e_voigt,
            "E_eff = {e_eff:e} outside ({e_reuss:e}, {e_voigt:e})"
        );
        // The bulk modulus of a cubic composite lies within the Hashin-Shtrikman bounds
        // (small slack for the voxelised sphere).
        let (k_lo, k_hi, _, _) = hashin_shtrikman_bounds(&weighted[0], &weighted[1]);
        let k_eff = effective_bulk_modulus(&c);
        assert!(
            k_eff > 0.98 * k_lo && k_eff < 1.02 * k_hi,
            "K_eff = {k_eff:e} outside HS [{k_lo:e}, {k_hi:e}]"
        );
        // The microstructure matters: the answer differs from the Hill average.
        assert!(max_relative_difference(&c, &hill_average(&weighted)) > 0.01);
    }

    #[test]
    fn unused_invalid_phases_are_ignored_and_used_ones_validated() {
        let rve = VoxelRve::from_fn([2, 2, 2], |_, _, _| 0).expect("valid grid");
        let invalid = Phase::new("bad", 0.0, -1.0, 0.7);
        let phases = [soft(), invalid];
        assert!(
            compute_effective_stiffness_from_unit_strains(&rve, &phases, &tight()).is_ok(),
            "an unreferenced invalid phase must not matter"
        );
        let rve_bad = VoxelRve::from_fn([2, 2, 2], |_, _, iz| iz).expect("valid grid");
        let result = compute_effective_stiffness_from_unit_strains(&rve_bad, &phases, &tight());
        assert!(matches!(result, Err(Error::InvalidInput(_))), "{result:?}");
    }

    #[test]
    fn invalid_arguments_are_rejected() {
        let phases = [soft()];
        let rve = VoxelRve::from_fn([2, 2, 2], |_, _, _| 0).expect("valid grid");
        // Empty phase list, out-of-range voxel phase, bad options.
        assert!(matches!(
            compute_effective_stiffness_from_unit_strains(&rve, &[], &tight()),
            Err(Error::InvalidInput(_))
        ));
        let out_of_range = VoxelRve::from_fn([2, 2, 2], |_, _, _| 3).expect("valid grid");
        assert!(matches!(
            compute_effective_stiffness_from_unit_strains(&out_of_range, &phases, &tight()),
            Err(Error::InvalidInput(_))
        ));
        for options in [
            FftHomogenizationOptions {
                tolerance: 0.0,
                ..tight()
            },
            FftHomogenizationOptions {
                tolerance: f64::NAN,
                ..tight()
            },
            FftHomogenizationOptions {
                max_iterations: 0,
                ..tight()
            },
        ] {
            assert!(matches!(
                compute_effective_stiffness_from_unit_strains(&rve, &phases, &options),
                Err(Error::InvalidInput(_))
            ));
        }
        // Grid construction.
        assert!(matches!(
            VoxelRve::new([0, 2, 2], vec![]),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            VoxelRve::new([2, 2, 2], vec![0; 7]),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            VoxelRve::new([usize::MAX, 2, 1], vec![]),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn exhausted_iteration_budget_is_an_error_not_a_silent_result() {
        let phases = [soft(), stiff()];
        let rve =
            VoxelRve::from_fn([4, 4, 8], |_, _, iz| usize::from(iz % 2 == 1)).expect("valid grid");
        let options = FftHomogenizationOptions {
            scheme: FftScheme::Basic,
            tolerance: 1.0e-14,
            max_iterations: 2,
        };
        let result = compute_effective_stiffness_from_unit_strains(&rve, &phases, &options);
        assert!(matches!(result, Err(Error::General(_))), "{result:?}");
    }

    #[test]
    fn voxel_grid_accessors_and_fractions() {
        let rve = VoxelRve::from_fn([2, 3, 4], |ix, iy, iz| (ix + iy + iz) % 3).expect("grid");
        assert_eq!(rve.dims(), [2, 3, 4]);
        assert_eq!(rve.voxel_count(), 24);
        assert_eq!(rve.phase_index().len(), 24);
        // Flat index ix*ny*nz + iy*nz + iz: voxel (1, 2, 3) is the last one.
        assert_eq!(rve.phase_index()[23], (1 + 2 + 3) % 3);
        let fractions = rve.volume_fractions(3);
        assert_eq!(fractions.len(), 3);
        assert!((fractions.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        // Indices beyond n_phases are not counted.
        assert!(rve.volume_fractions(1)[0] < 1.0);
    }
}
