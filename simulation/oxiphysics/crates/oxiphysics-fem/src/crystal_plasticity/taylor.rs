//! Taylor (full-constraints) model: the minimum-shear Taylor factor of a crystal.
//!
//! Imposing an isochoric strain rate **D** on a single crystal requires slip
//! rates γ̇_α with Σ_α γ̇_α · sym(s_α ⊗ n_α) = **D**.  A symmetric, traceless
//! tensor has five independent components, so for equal critical resolved shear
//! stress on all systems the combination that minimises the total shear
//! Σ|γ̇_α| (principle of minimum shear, equivalent to the Bishop-Hill maximum
//! work principle) is a vertex of this linear programme: it activates five
//! slip systems whose 5×5 constraint matrix is non-singular.
//!
//! [`TaylorSolver`] enumerates and factorises every five-system combination of
//! a crystal once (792 combinations for the 12 FCC systems, 384 of them
//! non-singular) and then evaluates, for each imposed strain rate, the minimum
//! of Σ|γ̇_α| over all combinations.  The Taylor factor is
//!
//! M = min Σ|γ̇_α| / ε̇_eq,  ε̇_eq = √(2/3 D:D),
//!
//! i.e. the ratio of the flow stress of a crystal deforming at the imposed
//! strain rate to its critical resolved shear stress (σ = M·τ_c).  For uniaxial
//! tension of an FCC crystal M = √6 ≈ 2.449 along ⟨100⟩, 3√6/2 ≈ 3.674 along
//! ⟨111⟩ and ⟨110⟩, and the orientation average of a random aggregate is 3.06.

use super::SlipSystem;
use crate::dense_lu::LuFactor;

/// Largest number of five-system combinations that is enumerated (C(28, 5) =
/// 98 280).  The count grows as n⁵; larger slip-system sets must be reduced.
const MAX_COMBINATIONS: u128 = 100_000;

/// A slip-system combination is accepted for a strain rate only if it
/// reproduces the strain rate to this relative accuracy.
const RESIDUAL_TOLERANCE: f64 = 1e-9;

/// One non-singular combination of five slip systems.
#[derive(Debug, Clone)]
struct FiveSystemCombination {
    /// Column `c` holds the deviatoric coordinates of the c-th system's
    /// symmetric Schmid tensor.
    matrix: [[f64; 5]; 5],
    /// LU factors of `matrix` (shared by all strain rates).
    lu: LuFactor,
}

/// Pre-factorised five-system combinations of a crystal for Taylor-factor
/// evaluation; see the documentation of the private `taylor` module.
#[derive(Debug, Clone)]
pub struct TaylorSolver {
    combinations: Vec<FiveSystemCombination>,
}

impl TaylorSolver {
    /// Enumerate and factorise the five-system combinations of `systems`.
    ///
    /// # Errors
    /// [`crate::Error::InvalidInput`] for fewer than five systems, more systems
    /// than the enumeration limit (at most 28), or a non-finite slip system;
    /// [`crate::Error::SingularSystem`] if no combination of five systems is
    /// linearly independent (the crystal cannot accommodate an arbitrary
    /// isochoric strain by slip).
    pub fn new(systems: &[SlipSystem]) -> crate::Result<Self> {
        let n = systems.len();
        if n < 5 {
            return Err(crate::Error::InvalidInput(format!(
                "a Taylor analysis needs at least 5 slip systems, got {n}"
            )));
        }
        if binomial_n_choose_5(n) > MAX_COMBINATIONS {
            return Err(crate::Error::InvalidInput(format!(
                "{n} slip systems give {} five-system combinations; at most {MAX_COMBINATIONS} \
                 are supported",
                binomial_n_choose_5(n)
            )));
        }
        let columns: Vec<[f64; 5]> = systems
            .iter()
            .map(|s| deviatoric_coordinates(&s.schmid_tensor))
            .collect();
        if columns.iter().flatten().any(|x| !x.is_finite()) {
            return Err(crate::Error::InvalidInput(
                "slip systems must have finite Schmid tensors".into(),
            ));
        }
        let mut combinations = Vec::new();
        for subset in five_subsets(n) {
            let mut matrix = [[0.0; 5]; 5];
            for (c, &system) in subset.iter().enumerate() {
                for (r, row) in matrix.iter_mut().enumerate() {
                    row[c] = columns[system][r];
                }
            }
            let rows: Vec<Vec<f64>> = matrix.iter().map(|row| row.to_vec()).collect();
            if let Some(lu) = LuFactor::new(&rows) {
                combinations.push(FiveSystemCombination { matrix, lu });
            }
        }
        if combinations.is_empty() {
            return Err(crate::Error::SingularSystem);
        }
        Ok(Self { combinations })
    }

    /// Number of linearly independent five-system combinations (384 for FCC).
    pub fn n_combinations(&self) -> usize {
        self.combinations.len()
    }

    /// Minimum total shear Σ|γ̇_α| accommodating the isochoric strain rate whose
    /// deviatoric coordinates (see [`deviatoric_coordinates`]) are `d`.
    fn minimum_shear(&self, d: &[f64; 5]) -> crate::Result<f64> {
        let d_norm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut best = f64::INFINITY;
        for combination in &self.combinations {
            let gamma = combination.lu.solve(d);
            let total: f64 = gamma.iter().map(|g| g.abs()).sum();
            if !total.is_finite() || total >= best {
                continue;
            }
            let residual = combination
                .matrix
                .iter()
                .zip(d.iter())
                .map(|(row, &target)| {
                    let achieved: f64 = row.iter().zip(gamma.iter()).map(|(a, g)| a * g).sum();
                    (achieved - target).abs()
                })
                .fold(0.0_f64, f64::max);
            if residual <= RESIDUAL_TOLERANCE * d_norm {
                best = total;
            }
        }
        if best.is_finite() {
            Ok(best)
        } else {
            Err(crate::Error::SingularSystem)
        }
    }

    /// Taylor factor M = min Σ|γ̇_α| / ε̇_eq of the crystal for the imposed
    /// strain rate `strain_rate` (symmetric 3×3 tensor, row-major, in the
    /// crystal frame).  Only the deviatoric part is used and the result does not
    /// depend on the magnitude of the strain rate.
    ///
    /// # Errors
    /// [`crate::Error::InvalidInput`] for a non-finite or (deviatorically) zero
    /// strain rate; [`crate::Error::SingularSystem`] if no slip combination
    /// reproduces it.
    pub fn taylor_factor(&self, strain_rate: &[f64; 9]) -> crate::Result<f64> {
        let d = deviatoric_coordinates(strain_rate);
        let d_norm = d.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !d_norm.is_finite() || d_norm <= 0.0 {
            return Err(crate::Error::InvalidInput(
                "the imposed strain rate must be finite and have a non-zero deviatoric part".into(),
            ));
        }
        let minimum_shear = self.minimum_shear(&d)?;
        // The coordinates are orthonormal, so ‖D‖_F = ‖d‖ and ε̇_eq = √(2/3)·‖d‖.
        Ok(minimum_shear / ((2.0_f64 / 3.0).sqrt() * d_norm))
    }

    /// Taylor factor for uniaxial tension along `loading_axis` (given in the
    /// crystal frame; it need not be normalised).  The imposed strain rate is
    /// D = 3/2·a⊗a − 1/2·I, so ε̇_eq equals the axial strain rate.
    ///
    /// # Errors
    /// [`crate::Error::InvalidInput`] if `loading_axis` is zero or not finite;
    /// otherwise those of [`Self::taylor_factor`].
    pub fn uniaxial_taylor_factor(&self, loading_axis: [f64; 3]) -> crate::Result<f64> {
        let length = loading_axis.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !length.is_finite() || length <= 0.0 {
            return Err(crate::Error::InvalidInput(
                "the loading axis must be a finite, non-zero vector".into(),
            ));
        }
        let a = loading_axis.map(|x| x / length);
        let mut d = [0.0; 9];
        for (k, entry) in d.iter_mut().enumerate() {
            let (i, j) = (k / 3, k % 3);
            let delta = if i == j { 0.5 } else { 0.0 };
            *entry = 1.5 * a[i] * a[j] - delta;
        }
        self.taylor_factor(&d)
    }
}

/// Orthonormal 5-component coordinates of the deviatoric part of a symmetric
/// 3×3 tensor `e` (row-major):
/// `((E₁₁−E₂₂)/√2, (2E₃₃−E₁₁−E₂₂)/√6, √2·E₁₂, √2·E₂₃, √2·E₃₁)`.
///
/// The basis is orthonormal for the Frobenius inner product, so the Euclidean
/// inner product of two coordinate vectors equals the double contraction of the
/// deviatoric tensors.  The trace is projected out and the off-diagonal entries
/// are symmetrised.
pub(super) fn deviatoric_coordinates(e: &[f64; 9]) -> [f64; 5] {
    let sqrt2 = std::f64::consts::SQRT_2;
    let sqrt6 = 6.0_f64.sqrt();
    [
        (e[0] - e[4]) / sqrt2,
        (2.0 * e[8] - e[0] - e[4]) / sqrt6,
        sqrt2 * 0.5 * (e[1] + e[3]),
        sqrt2 * 0.5 * (e[5] + e[7]),
        sqrt2 * 0.5 * (e[6] + e[2]),
    ]
}

/// C(n, 5) as a wide integer (no overflow for any `usize` input).
fn binomial_n_choose_5(n: usize) -> u128 {
    if n < 5 {
        return 0;
    }
    let n = n as u128;
    n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 120
}

/// All 5-element subsets of `0..n` in lexicographic order.
fn five_subsets(n: usize) -> Vec<[usize; 5]> {
    let mut subsets = Vec::new();
    if n < 5 {
        return subsets;
    }
    let mut idx = [0usize, 1, 2, 3, 4];
    loop {
        subsets.push(idx);
        let Some(k) = (0..5).rev().find(|&k| idx[k] < k + n - 5) else {
            return subsets;
        };
        idx[k] += 1;
        for j in (k + 1)..5 {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn five_subsets_enumerates_all_combinations_in_order() {
        let subsets = five_subsets(7);
        assert_eq!(subsets.len(), 21);
        assert_eq!(subsets[0], [0, 1, 2, 3, 4]);
        assert_eq!(subsets[20], [2, 3, 4, 5, 6]);
        assert!(subsets.windows(2).all(|w| w[0] < w[1]), "not lexicographic");
        assert_eq!(five_subsets(12).len(), 792);
        assert!(five_subsets(4).is_empty());
        assert_eq!(binomial_n_choose_5(12), 792);
        assert_eq!(binomial_n_choose_5(28), 98_280);
        assert_eq!(binomial_n_choose_5(4), 0);
    }

    #[test]
    fn deviatoric_coordinates_are_orthonormal_for_the_double_contraction() {
        // Two symmetric traceless tensors: coordinate dot product = E:F.
        let e = [1.0, 0.3, -0.2, 0.3, -0.4, 0.5, -0.2, 0.5, -0.6];
        let f = [0.2, -0.7, 0.1, -0.7, 0.9, 0.4, 0.1, 0.4, -1.1];
        let contraction: f64 = e.iter().zip(f.iter()).map(|(a, b)| a * b).sum();
        let (ce, cf) = (deviatoric_coordinates(&e), deviatoric_coordinates(&f));
        let dot: f64 = ce.iter().zip(cf.iter()).map(|(a, b)| a * b).sum();
        assert!((dot - contraction).abs() < 1e-14, "{dot} vs {contraction}");
        // A pure trace has no deviatoric coordinates.
        let isotropic = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        assert!(
            deviatoric_coordinates(&isotropic)
                .iter()
                .all(|c| c.abs() < 1e-15)
        );
    }
}
