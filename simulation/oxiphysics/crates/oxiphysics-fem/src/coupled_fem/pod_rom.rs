//! Proper Orthogonal Decomposition (POD) reduced-order model for coupled problems.
//!
//! The POD basis is computed with the *method of snapshots*: the leading
//! eigenpairs of the snapshot Gram matrix `C = SᵀS` give the singular values
//! `σ_k = √λ_k` and the energy-optimal modes `φ_k = S v_k / σ_k` of the snapshot
//! matrix `S` (one snapshot per column).  The retained modes minimise the
//! projection error ‖S − ΦΦᵀS‖_F over all bases of the same size.

use crate::eigen::jacobi_eigen_dense;
use crate::error::{Error, Result};

/// Gram-matrix eigenvalues below `RANK_TOLERANCE · λ_max` are treated as rank
/// deficiency of the snapshot set (they are rounding noise of the squared
/// singular values) and produce no mode.
const RANK_TOLERANCE: f64 = 1e-12;

/// Proper Orthogonal Decomposition (POD) reduced-order model for coupling.
///
/// Projects the full-order system onto a low-dimensional subspace spanned by the
/// dominant modes of a snapshot set; see [`PodRom::from_snapshots`].
#[derive(Debug, Clone)]
pub struct PodRom {
    /// POD modes in decreasing order of energy: `basis[k]` is the k-th
    /// orthonormal mode (length `n_full`).
    pub basis: Vec<Vec<f64>>,
    /// Singular values σ_k of the retained modes (decreasing).
    pub singular_values: Vec<f64>,
    /// Total snapshot energy Σ_i ‖s_i‖² = Σ_k σ_k² over *all* snapshot modes
    /// (the denominator of [`Self::energy_fraction`]).
    pub total_energy: f64,
    /// Reduced-order stiffness matrix Φᵀ K Φ (n_modes × n_modes); zero until
    /// [`Self::project_system`] is called.
    pub k_rom: Vec<Vec<f64>>,
    /// Reduced-order mass matrix Φᵀ M Φ (n_modes × n_modes); zero until
    /// [`Self::project_system`] is called.
    pub m_rom: Vec<Vec<f64>>,
    /// Reduced-order force vector Φᵀ f; zero until [`Self::project_system`] is called.
    pub f_rom: Vec<f64>,
    /// Reduced coordinates
    pub q: Vec<f64>,
    /// Full-order DOFs
    pub n_full: usize,
    /// Number of POD modes retained
    pub n_modes: usize,
}

impl PodRom {
    /// Compute the POD basis of a snapshot set with the method of snapshots.
    ///
    /// `snapshots[i]` is the i-th snapshot (length `n_full`); up to `n_modes`
    /// dominant modes are kept.  Fewer modes are returned when the snapshots
    /// have a smaller numerical rank (Gram eigenvalues below
    /// 1e-12·λ_max are dropped), so [`Self::n_modes`] is the number actually
    /// retained.  The modes are orthonormal to rounding level.
    ///
    /// The reduced operators `k_rom`, `m_rom`, `f_rom` are zero until
    /// [`Self::project_system`] projects the full-order system.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for an empty snapshot set or `n_modes == 0`,
    /// snapshots of different lengths, a non-finite entry, or a snapshot set
    /// that is identically zero; [`Error::SingularSystem`] if the Gram matrix
    /// has no positive eigenvalue after the rank cut-off.
    pub fn from_snapshots(snapshots: &[Vec<f64>], n_modes: usize) -> Result<Self> {
        let n_snap = snapshots.len();
        if n_snap == 0 || n_modes == 0 {
            return Err(Error::InvalidInput(format!(
                "POD needs at least one snapshot and one requested mode, got {n_snap} snapshots \
                 and n_modes = {n_modes}"
            )));
        }
        let n_full = snapshots[0].len();
        if n_full == 0 || snapshots.iter().any(|s| s.len() != n_full) {
            return Err(Error::InvalidInput(
                "POD snapshots must be non-empty and all of the same length".into(),
            ));
        }
        if snapshots.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::InvalidInput("POD snapshots must be finite".into()));
        }
        let mut gram = vec![vec![0.0; n_snap]; n_snap];
        for i in 0..n_snap {
            for j in i..n_snap {
                let g = dot(&snapshots[i], &snapshots[j]);
                gram[i][j] = g;
                gram[j][i] = g;
            }
        }
        let total_energy: f64 = (0..n_snap).map(|i| gram[i][i]).sum();
        let scale = gram
            .iter()
            .enumerate()
            .map(|(i, row)| row[i])
            .fold(0.0_f64, f64::max);
        if scale <= 0.0 {
            return Err(Error::InvalidInput(
                "all POD snapshots vanish; there is no mode to extract".into(),
            ));
        }
        // The Jacobi eigen-solver uses absolute tolerances, so work on a Gram
        // matrix of unit scale and rescale the eigenvalues afterwards.
        for row in gram.iter_mut() {
            for g in row.iter_mut() {
                *g /= scale;
            }
        }
        let (scaled_values, vectors) = jacobi_eigen_dense(&gram, n_snap);
        let lambda_max = scaled_values.last().copied().unwrap_or(0.0) * scale;
        if lambda_max <= 0.0 {
            return Err(Error::SingularSystem);
        }
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut singular_values: Vec<f64> = Vec::new();
        for k in 0..n_modes.min(n_snap) {
            let idx = n_snap - 1 - k;
            let lambda = scaled_values[idx] * scale;
            if lambda < RANK_TOLERANCE * lambda_max {
                break;
            }
            let sigma = lambda.sqrt();
            let mut mode = vec![0.0; n_full];
            for (snap, &weight) in snapshots.iter().zip(vectors[idx].iter()) {
                for (m, &s) in mode.iter_mut().zip(snap.iter()) {
                    *m += weight * s / sigma;
                }
            }
            // Re-orthonormalise (modified Gram-Schmidt, applied twice): the modes
            // of the squared problem are only orthogonal to ~ε·λ_max/λ_k.
            for _ in 0..2 {
                for previous in &basis {
                    let overlap = dot(&mode, previous);
                    for (m, &p) in mode.iter_mut().zip(previous.iter()) {
                        *m -= overlap * p;
                    }
                }
            }
            let norm = dot(&mode, &mode).sqrt();
            if norm < 0.5 {
                // Numerically dependent on the modes already kept.
                break;
            }
            for m in mode.iter_mut() {
                *m /= norm;
            }
            basis.push(mode);
            singular_values.push(sigma);
        }
        let retained = basis.len();
        if retained == 0 {
            return Err(Error::SingularSystem);
        }
        Ok(Self {
            basis,
            singular_values,
            total_energy,
            k_rom: vec![vec![0.0; retained]; retained],
            m_rom: vec![vec![0.0; retained]; retained],
            f_rom: vec![0.0; retained],
            q: vec![0.0; retained],
            n_full,
            n_modes: retained,
        })
    }
    /// Fraction of the snapshot energy Σ σ_k² captured by the retained modes
    /// (in \[0, 1\]; 1 when the modes span all snapshots).
    pub fn energy_fraction(&self) -> f64 {
        if self.total_energy <= 0.0 {
            return 0.0;
        }
        let captured: f64 = self.singular_values.iter().map(|s| s * s).sum();
        (captured / self.total_energy).min(1.0)
    }
    /// Project a full-order vector onto the reduced basis.
    pub fn project(&self, v_full: &[f64]) -> Vec<f64> {
        self.basis.iter().map(|mode| dot(mode, v_full)).collect()
    }
    /// Reconstruct the full-order vector from reduced coordinates.
    pub fn reconstruct(&self, q: &[f64]) -> Vec<f64> {
        let mut v = vec![0.0; self.n_full];
        for (mode, &q_m) in self.basis.iter().zip(q.iter()) {
            for (v_i, &b) in v.iter_mut().zip(mode.iter()) {
                *v_i += b * q_m;
            }
        }
        v
    }
    /// Project a full-order matrix into the ROM space.
    ///
    /// K_r = Φᵀ K Φ
    pub fn project_matrix(&self, k_full: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k_phi: Vec<Vec<f64>> = self
            .basis
            .iter()
            .map(|mode| k_full.iter().map(|row| dot(row, mode)).collect())
            .collect();
        self.basis
            .iter()
            .map(|mode_i| k_phi.iter().map(|kp| dot(mode_i, kp)).collect())
            .collect()
    }
    /// Project the full-order system (K, M, f) and store the reduced operators
    /// `k_rom = ΦᵀKΦ`, `m_rom = ΦᵀMΦ`, `f_rom = Φᵀf`.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] if `k_full` / `m_full` are not `n_full × n_full`
    /// or `f_full` does not have `n_full` entries; nothing is modified then.
    pub fn project_system(
        &mut self,
        k_full: &[Vec<f64>],
        m_full: &[Vec<f64>],
        f_full: &[f64],
    ) -> Result<()> {
        let n = self.n_full;
        let square = |a: &[Vec<f64>]| a.len() == n && a.iter().all(|row| row.len() == n);
        if !square(k_full) || !square(m_full) || f_full.len() != n {
            return Err(Error::InvalidInput(format!(
                "reduced-order projection needs {n}×{n} matrices and a vector of length {n}"
            )));
        }
        self.k_rom = self.project_matrix(k_full);
        self.m_rom = self.project_matrix(m_full);
        self.f_rom = self.project(f_full);
        Ok(())
    }
    /// Projection error ‖v − Φ Φᵀ v‖ / ‖v‖.
    pub fn projection_error(&self, v_full: &[f64]) -> f64 {
        let q = self.project(v_full);
        let v_approx = self.reconstruct(&q);
        let err: f64 = v_full
            .iter()
            .zip(v_approx.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm: f64 = v_full.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
        err / norm
    }
}

/// Euclidean inner product of two slices (extra entries of the longer one are ignored).
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
