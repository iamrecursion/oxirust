// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! LINCS (LINear Constraint Solver) for bond-length constraints.

use super::BondConstraint;
use oxiphysics_core::math::{Real, Vec3};
use std::collections::HashSet;

/// Reference bonds shorter than this have no direction; their constraint is left unsolved.
const MIN_REFERENCE_LENGTH: Real = 1e-30;

/// LINCS (LINear Constraint Solver) for bond-length constraints.
///
/// Hess, Bekker, Berendsen & Fraaije, J. Comput. Chem. 18, 1463 (1997).
///
/// Let `r_c = r_i - r_j` be the bond of constraint `c`, `e_c` its unit vector at the
/// *reference* positions (the positions before the unconstrained step), `w = 1/m` the inverse
/// masses, `B` the matrix whose row `c` holds `+e_c` at atom `i` and `-e_c` at atom `j`,
/// `S = diag(1 / sqrt(w_i + w_j))` and `d` the target lengths.  The constrained positions
/// follow from the Lagrange-multiplier solution with the constraint directions frozen at the
/// reference geometry:
///
/// ```text
/// r = r_unc - M^-1 B^T S (I - A)^-1 S (B r_unc - d),     A = I - S B M^-1 B^T S
/// ```
///
/// `A` has a zero diagonal and is non-zero only for pairs of constraints that share an atom
/// `k`: `A_cd = - S_c S_d s_c(k) s_d(k) w_k (e_c . e_d)` with `s_c(k) = +1` if `k` is the
/// first and `-1` if it is the second atom of `c`.  Instead of inverting `I - A`, LINCS
/// expands `(I - A)^-1 = I + A + A^2 + ...` and keeps the terms up to
/// [`Lincs::expansion_order`] (`A^n` is applied to a vector as `n` sparse products, so a
/// solve costs `O(order * couplings)`).  The series converges geometrically with the
/// spectral radius of `A`, which is small (about `0.3` for a zigzag chain with 109.5 degree
/// bond angles) for typical molecules and approaches `1` for long runs of nearly collinear
/// bonds or for closed triangles of constraints (about `0.77` for rigid TIP3P water written
/// as three distance constraints) -- use a higher order, [`super::Shake`], or for water
/// [`super::SettleWater`] there.
///
/// The solve has two stages ([`Lincs::apply`]):
///
/// 1. the *projection*: the projections of the new bond vectors on the reference directions
///    `e_c` are set to `d_c` by the expression above;
/// 2. the *correction for rotational lengthening*: a bond that rotated away from `e_c` is
///    longer than its projection, so the projections are re-targeted to
///    `p_c = sqrt(d_c^2 - rho_c^2)` with `rho_c` the component of the bond perpendicular to
///    `e_c` (`p_c = sqrt(2 d_c^2 - l_c^2)` with `l_c` the bond length when stage 1 is exact),
///    which is repeated [`Lincs::correction_passes`] times.
///
/// Both stages displace the atoms along the reference bond vectors by forces internal to each
/// constraint, so the centre of mass (momentum) is conserved exactly.  The result is not
/// identical to the converged [`super::Shake`] solution: LINCS is exact for isolated bonds,
/// while for coupled constraints it differs from SHAKE by the truncation of the series
/// (geometric in the expansion order) plus a residual of the third order in the relative
/// displacement, which further correction passes reduce.
pub struct Lincs {
    /// Bond constraints.
    pub constraints: Vec<BondConstraint>,
    /// Order of the LINCS expansion, i.e. the number of terms `A^1 .. A^order` kept after
    /// the identity (typically 4-8).
    pub expansion_order: usize,
    /// Number of correction passes for rotational lengthening (1 in the original
    /// algorithm; 2 conserves energy better in NVE runs; 0 keeps only the projection).
    pub correction_passes: usize,
}

impl Lincs {
    /// Create a new LINCS solver with expansion order 4 and one correction pass.
    pub fn new(constraints: Vec<BondConstraint>) -> Self {
        Self {
            constraints,
            expansion_order: 4,
            correction_passes: 1,
        }
    }

    /// Create a LINCS solver with custom expansion order (one correction pass).
    pub fn with_order(constraints: Vec<BondConstraint>, order: usize) -> Self {
        Self {
            constraints,
            expansion_order: order,
            correction_passes: 1,
        }
    }

    /// Set the number of correction passes for rotational lengthening.
    pub fn with_correction_passes(mut self, passes: usize) -> Self {
        self.correction_passes = passes;
        self
    }

    /// Apply LINCS to correct positions after an unconstrained update.
    ///
    /// `positions` are the unconstrained positions and are overwritten with the constrained
    /// ones; `old_positions` are the reference positions of the step (they fix the constraint
    /// directions); `inv_masses[i]` is `1/m_i` (0 for an infinitely heavy atom).
    ///
    /// Constraints that cannot be solved are left alone: a reference bond of zero length
    /// (no direction), a non-positive or non-finite target length, a constraint of an atom
    /// with itself, a constraint between two atoms of infinite mass, and a constraint that
    /// repeats the atom pair of an earlier one (a redundant constraint would make `I - A`
    /// singular and the series diverge).
    ///
    /// Returns the maximum relative constraint violation `|r^2 - d^2| / d^2` after the
    /// correction, over all constraints (including those left alone); `NaN` if any bond is
    /// not finite.
    ///
    /// # Panics
    ///
    /// Panics if a constraint refers to an atom outside `positions`, `old_positions` or
    /// `inv_masses`, like indexing those slices does.
    pub fn apply(
        &self,
        positions: &mut [Vec3],
        old_positions: &[Vec3],
        inv_masses: &[f64],
    ) -> Real {
        let system = LincsSystem::new(&self.constraints, old_positions, inv_masses);

        // Stage 1: set the projections of the bonds on the reference directions to d.
        let projection_residuals: Vec<Real> = system
            .bonds
            .iter()
            .map(|bond| {
                let bond_vector = positions[bond.atom_i] - positions[bond.atom_j];
                bond.scale * (bond.direction.dot(&bond_vector) - bond.length)
            })
            .collect();
        let solution = system.expand(&projection_residuals, self.expansion_order);
        system.displace(positions, inv_masses, &solution);

        // Stage 2: shorten the projection by the lengthening the rotation of each bond caused.
        for _ in 0..self.correction_passes {
            let lengthening_residuals: Vec<Real> = system
                .bonds
                .iter()
                .map(|bond| {
                    let bond_vector = positions[bond.atom_i] - positions[bond.atom_j];
                    let along = bond.direction.dot(&bond_vector);
                    let across_sq = (bond_vector - bond.direction * along).norm_squared();
                    let target = (bond.length * bond.length - across_sq).max(0.0).sqrt();
                    bond.scale * (along - target)
                })
                .collect();
            let solution = system.expand(&lengthening_residuals, self.expansion_order);
            system.displace(positions, inv_masses, &solution);
        }

        self.max_relative_violation(positions)
    }

    /// Maximum relative constraint violation `|r^2 - d^2| / d^2` (`NaN` if any is `NaN`).
    fn max_relative_violation(&self, positions: &[Vec3]) -> Real {
        let mut max_violation: Real = 0.0;
        for constraint in &self.constraints {
            let bond_vector = positions[constraint.atom_i] - positions[constraint.atom_j];
            let violation = (bond_vector.norm_squared() - constraint.distance_sq).abs()
                / constraint.distance_sq;
            if violation.is_nan() {
                return Real::NAN;
            }
            max_violation = max_violation.max(violation);
        }
        max_violation
    }
}

/// A constraint of a [`Lincs`] solve whose reference direction is defined.
struct ActiveBond {
    atom_i: usize,
    atom_j: usize,
    /// Unit vector of the reference bond `r_i - r_j`.
    direction: Vec3,
    /// `S_c = 1 / sqrt(w_i + w_j)`.
    scale: Real,
    /// Target bond length `d_c`.
    length: Real,
}

/// The constraint geometry LINCS freezes at the reference positions: the solvable
/// constraints and the sparse coupling matrix `A`.
struct LincsSystem {
    bonds: Vec<ActiveBond>,
    /// `couplings[row_starts[c]..row_starts[c + 1]]` are the non-zero `(d, A_cd)` of row `c`.
    row_starts: Vec<usize>,
    couplings: Vec<(usize, Real)>,
}

impl LincsSystem {
    /// Collect the solvable constraints and build `A` from their shared atoms.
    fn new(constraints: &[BondConstraint], old_positions: &[Vec3], inv_masses: &[f64]) -> Self {
        let mut bonds: Vec<ActiveBond> = Vec::with_capacity(constraints.len());
        let mut seen_pairs: HashSet<(usize, usize)> = HashSet::with_capacity(constraints.len());
        for constraint in constraints {
            let (atom_i, atom_j) = (constraint.atom_i, constraint.atom_j);
            let length = constraint.distance_sq.sqrt();
            let inv_mass_sum = inv_masses[atom_i] + inv_masses[atom_j];
            let reference = old_positions[atom_i] - old_positions[atom_j];
            let reference_length = reference.norm();
            let solvable = atom_i != atom_j
                && length.is_finite()
                && length > 0.0
                && inv_mass_sum.is_finite()
                && inv_mass_sum > 0.0
                && reference_length.is_finite()
                && reference_length > MIN_REFERENCE_LENGTH;
            if solvable && seen_pairs.insert((atom_i.min(atom_j), atom_i.max(atom_j))) {
                bonds.push(ActiveBond {
                    atom_i,
                    atom_j,
                    direction: reference / reference_length,
                    scale: 1.0 / inv_mass_sum.sqrt(),
                    length,
                });
            }
        }

        // Constraints sharing an atom are the neighbours of the atom's group once the
        // (atom, constraint) endpoints are sorted by atom.
        let mut endpoints: Vec<(usize, usize)> = Vec::with_capacity(2 * bonds.len());
        for (index, bond) in bonds.iter().enumerate() {
            endpoints.push((bond.atom_i, index));
            endpoints.push((bond.atom_j, index));
        }
        endpoints.sort_unstable();

        // (c, d, A_cd) for every ordered pair of constraints sharing an atom (two distinct
        // constraints share at most one atom, since repeated atom pairs were dropped above).
        let mut triples: Vec<(usize, usize, Real)> = Vec::new();
        for group in endpoints.chunk_by(|a, b| a.0 == b.0) {
            for &(atom, c) in group {
                for &(_, d) in group {
                    if c == d {
                        continue;
                    }
                    let side_c = if bonds[c].atom_i == atom { 1.0 } else { -1.0 };
                    let side_d = if bonds[d].atom_i == atom { 1.0 } else { -1.0 };
                    let coefficient = -bonds[c].scale
                        * bonds[d].scale
                        * side_c
                        * side_d
                        * inv_masses[atom]
                        * bonds[c].direction.dot(&bonds[d].direction);
                    if coefficient != 0.0 {
                        triples.push((c, d, coefficient));
                    }
                }
            }
        }
        triples.sort_by_key(|&(c, _, _)| c);

        let mut row_starts = vec![0_usize; bonds.len() + 1];
        for &(c, _, _) in &triples {
            row_starts[c + 1] += 1;
        }
        for c in 0..bonds.len() {
            row_starts[c + 1] += row_starts[c];
        }
        let couplings = triples.iter().map(|&(_, d, a)| (d, a)).collect();
        Self {
            bonds,
            row_starts,
            couplings,
        }
    }

    /// `(I + A + A^2 + ... + A^order) rhs`, the truncated expansion of `(I - A)^-1 rhs`.
    fn expand(&self, rhs: &[Real], order: usize) -> Vec<Real> {
        let mut solution = rhs.to_vec();
        let mut term = rhs.to_vec();
        let mut next = vec![0.0; rhs.len()];
        for _ in 0..order {
            for (c, slot) in next.iter_mut().enumerate() {
                let row = &self.couplings[self.row_starts[c]..self.row_starts[c + 1]];
                *slot = row
                    .iter()
                    .map(|&(d, coefficient)| coefficient * term[d])
                    .sum();
            }
            for (total, addend) in solution.iter_mut().zip(next.iter()) {
                *total += *addend;
            }
            std::mem::swap(&mut term, &mut next);
        }
        solution
    }

    /// Move the atoms of every constraint by `-+ w S solution e` along its reference direction:
    /// `r = r - M^-1 B^T S solution`.
    fn displace(&self, positions: &mut [Vec3], inv_masses: &[f64], solution: &[Real]) {
        for (bond, &multiplier) in self.bonds.iter().zip(solution.iter()) {
            let shift = bond.direction * (bond.scale * multiplier);
            positions[bond.atom_i] -= shift * inv_masses[bond.atom_i];
            positions[bond.atom_j] += shift * inv_masses[bond.atom_j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{Shake, chain_constraints, water_shake_constraints};

    /// Deterministic uniform generator on `[-1, 1)` (64-bit LCG, Knuth MMIX constants).
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn symmetric(&mut self) -> Real {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            2.0 * ((self.0 >> 11) as f64 / (1_u64 << 53) as f64) - 1.0
        }
    }

    /// `reference` displaced by up to `amplitude` in every coordinate.
    fn perturbed(reference: &[Vec3], amplitude: Real, rng: &mut Lcg) -> Vec<Vec3> {
        reference
            .iter()
            .map(|p| p + Vec3::new(rng.symmetric(), rng.symmetric(), rng.symmetric()) * amplitude)
            .collect()
    }

    /// Planar zigzag chain with unit bonds and the given interior bond angle (degrees).
    fn zigzag_chain(n_atoms: usize, angle_deg: Real) -> Vec<Vec3> {
        let half = angle_deg.to_radians() / 2.0;
        (0..n_atoms)
            .map(|k| Vec3::new(k as Real * half.sin(), (k % 2) as Real * half.cos(), 0.0))
            .collect()
    }

    /// Rigid TIP3P water `[O, H1, H2]` built from the constraint distances themselves.
    fn tip3p_molecule() -> Vec<Vec3> {
        let (d_oh, d_hh): (Real, Real) = (0.9572, 1.5139);
        let half = 0.5 * d_hh;
        let height = (d_oh * d_oh - half * half).sqrt();
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(half, height, 0.0),
            Vec3::new(-half, height, 0.0),
        ]
    }

    /// Methane-like star: atom 0 bonded to four atoms of a regular tetrahedron (1.09 long).
    fn methane_star() -> (Vec<Vec3>, Vec<BondConstraint>) {
        let t = 1.09 / 3.0_f64.sqrt();
        let positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(t, t, t),
            Vec3::new(t, -t, -t),
            Vec3::new(-t, t, -t),
            Vec3::new(-t, -t, t),
        ];
        let constraints = (1..5).map(|k| BondConstraint::new(0, k, 1.09)).collect();
        (positions, constraints)
    }

    /// The converged SHAKE solution for the same reference and unconstrained positions.
    fn shake_solution(
        constraints: &[BondConstraint],
        old: &[Vec3],
        unconstrained: &[Vec3],
        inv_masses: &[Real],
    ) -> Vec<Vec3> {
        let shake = Shake::with_params(constraints.to_vec(), 100_000, 1e-14);
        let mut solved = unconstrained.to_vec();
        shake.apply(&mut solved, old, inv_masses);
        assert!(
            shake.max_relative_violation(&solved) < 1e-12,
            "the SHAKE reference must be converged"
        );
        solved
    }

    fn max_difference(a: &[Vec3], b: &[Vec3]) -> Real {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).norm())
            .fold(0.0, Real::max)
    }

    fn lincs_solution(
        lincs: &Lincs,
        old: &[Vec3],
        unconstrained: &[Vec3],
        inv_masses: &[Real],
    ) -> (Vec<Vec3>, Real) {
        let mut positions = unconstrained.to_vec();
        let violation = lincs.apply(&mut positions, old, inv_masses);
        (positions, violation)
    }

    #[test]
    fn test_lincs_constructors_and_builder() {
        let constraints = vec![BondConstraint::new(0, 1, 1.0)];
        let default = Lincs::new(constraints.clone());
        assert_eq!(default.expansion_order, 4);
        assert_eq!(default.correction_passes, 1);
        let custom = Lincs::with_order(constraints, 8).with_correction_passes(2);
        assert_eq!(custom.expansion_order, 8);
        assert_eq!(custom.correction_passes, 2);
    }

    #[test]
    fn test_lincs_single_bond_is_exact_and_equals_shake() {
        // A lone constraint has no coupling: both stages are exact, so LINCS equals SHAKE.
        let constraints = vec![BondConstraint::new(0, 1, 1.0)];
        let old = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let inv_masses = vec![1.0 / 12.0, 1.0];
        let unconstrained = vec![Vec3::new(0.05, 0.02, -0.01), Vec3::new(1.1, 0.12, 0.06)];
        let (positions, violation) = lincs_solution(
            &Lincs::new(constraints.clone()),
            &old,
            &unconstrained,
            &inv_masses,
        );
        assert!(violation < 1e-14, "violation {violation:e}");
        let length = (positions[0] - positions[1]).norm();
        assert!((length - 1.0).abs() < 1e-14, "length {length}");
        let reference = shake_solution(&constraints, &old, &unconstrained, &inv_masses);
        assert!(max_difference(&positions, &reference) < 1e-12);
        // The light atom does nearly all of the moving.
        let heavy_shift = (positions[0] - unconstrained[0]).norm();
        let light_shift = (positions[1] - unconstrained[1]).norm();
        assert!(
            light_shift > 10.0 * heavy_shift,
            "{heavy_shift} vs {light_shift}"
        );
    }

    #[test]
    fn test_lincs_matches_shake_on_a_ten_atom_chain() {
        let old = zigzag_chain(10, 109.5);
        let constraints = chain_constraints(0, 9, 1.0);
        let inv_masses = vec![1.0; 10];
        let lincs = Lincs::new(constraints.clone());
        let mut rng = Lcg::new(3);
        for trial in 0..25 {
            let unconstrained = perturbed(&old, 1e-3, &mut rng);
            let reference = shake_solution(&constraints, &old, &unconstrained, &inv_masses);
            let (positions, violation) = lincs_solution(&lincs, &old, &unconstrained, &inv_masses);
            let difference = max_difference(&positions, &reference);
            assert!(
                difference < 1e-6,
                "trial {trial}: |LINCS - SHAKE| = {difference:e}"
            );
            assert!(violation < 1e-6, "trial {trial}: violation {violation:e}");
        }
    }

    #[test]
    fn test_lincs_order_four_with_one_correction_conserves_bond_lengths_to_1e_minus_4() {
        // Displacements of 1 % of the bond length per coordinate.
        let old = zigzag_chain(10, 109.5);
        let constraints = chain_constraints(0, 9, 1.0);
        let inv_masses = vec![1.0; 10];
        let mut rng = Lcg::new(8);
        let default = Lincs::new(constraints.clone());
        let projection_only = Lincs::new(constraints).with_correction_passes(0);
        for trial in 0..25 {
            let unconstrained = perturbed(&old, 1e-2, &mut rng);
            let (_, with_correction) = lincs_solution(&default, &old, &unconstrained, &inv_masses);
            let (_, without) = lincs_solution(&projection_only, &old, &unconstrained, &inv_masses);
            assert!(
                with_correction < 1e-4,
                "trial {trial}: relative violation {with_correction:e} after the correction"
            );
            // The rotational lengthening is what the correction pass removes.
            assert!(
                without > 5.0 * with_correction,
                "trial {trial}: {without:e} without vs {with_correction:e} with the correction"
            );
        }
    }

    #[test]
    fn test_lincs_correction_passes_converge_towards_shake() {
        let old = zigzag_chain(10, 109.5);
        let constraints = chain_constraints(0, 9, 1.0);
        let inv_masses = vec![1.0; 10];
        let mut rng = Lcg::new(14);
        let unconstrained = perturbed(&old, 1e-2, &mut rng);
        let reference = shake_solution(&constraints, &old, &unconstrained, &inv_masses);
        let errors: Vec<Real> = (0..4)
            .map(|passes| {
                let lincs = Lincs::new(constraints.clone()).with_correction_passes(passes);
                let (positions, _) = lincs_solution(&lincs, &old, &unconstrained, &inv_masses);
                max_difference(&positions, &reference)
            })
            .collect();
        assert!(errors[0] > 10.0 * errors[1], "{errors:?}");
        assert!(errors[1] > 10.0 * errors[2], "{errors:?}");
        assert!(errors[2] > errors[3] || errors[3] < 1e-8, "{errors:?}");
    }

    #[test]
    fn test_lincs_error_falls_with_the_expansion_order() {
        // A sequential projection (no coupling terms, order 0) is far less accurate than the
        // series: every extra order removes most of the remaining coupling error.
        let old = zigzag_chain(10, 109.5);
        let constraints = chain_constraints(0, 9, 1.0);
        let inv_masses = vec![1.0; 10];
        let mut rng = Lcg::new(19);
        let unconstrained = perturbed(&old, 1e-3, &mut rng);
        let violations: Vec<Real> = [0, 1, 2, 4, 8]
            .into_iter()
            .map(|order| {
                let lincs = Lincs::with_order(constraints.clone(), order);
                lincs_solution(&lincs, &old, &unconstrained, &inv_masses).1
            })
            .collect();
        assert!(violations[0] > 3.0 * violations[1], "{violations:?}");
        assert!(violations[1] > 3.0 * violations[2], "{violations:?}");
        assert!(violations[2] > 3.0 * violations[3], "{violations:?}");
        assert!(violations[3] > 2.0 * violations[4], "{violations:?}");
        assert!(violations[4] < 1e-7, "{violations:?}");
    }

    #[test]
    fn test_lincs_series_converges_to_the_exact_matrix_inverse() {
        // For a tiny displacement the rotation correction is negligible, so the result has
        // exactly the projection `e_c . r_c = d_c` that the exact `(I - A)^-1` gives.  A wrong
        // coupling coefficient (sign, shared atom, mass) would make the series diverge or
        // converge to something else.
        let (star, star_constraints) = methane_star();
        let star_inv_masses = vec![1.0 / 12.0, 1.0, 1.0, 1.0, 1.0];
        let water = tip3p_molecule();
        let water_constraints = water_shake_constraints(0);
        let water_inv_masses = vec![1.0 / 15.9994, 1.0 / 1.008, 1.0 / 1.008];
        let mut rng = Lcg::new(23);
        for (name, old, constraints, inv_masses) in [
            ("methane star", star, star_constraints, star_inv_masses),
            ("water triangle", water, water_constraints, water_inv_masses),
        ] {
            let unconstrained = perturbed(&old, 1e-7, &mut rng);
            let exact = Lincs::with_order(constraints.clone(), 120);
            let truncated = Lincs::with_order(constraints.clone(), 0);
            let (_, exact_violation) = lincs_solution(&exact, &old, &unconstrained, &inv_masses);
            let (_, uncoupled_violation) =
                lincs_solution(&truncated, &old, &unconstrained, &inv_masses);
            assert!(exact_violation < 1e-12, "{name}: {exact_violation:e}");
            assert!(
                uncoupled_violation > 1e5 * exact_violation,
                "{name}: order 0 leaves {uncoupled_violation:e}, order 120 {exact_violation:e}"
            );
        }
    }

    #[test]
    fn test_lincs_unequal_masses_agree_with_shake() {
        let old = zigzag_chain(10, 109.5);
        let constraints = chain_constraints(0, 9, 1.0);
        // Alternating heavy (12 u) and light (1 u) atoms, like a C-H chain.
        let inv_masses: Vec<Real> = (0..10)
            .map(|k| if k % 2 == 0 { 1.0 / 12.0 } else { 1.0 })
            .collect();
        let mut rng = Lcg::new(29);
        let unconstrained = perturbed(&old, 1e-3, &mut rng);
        let reference = shake_solution(&constraints, &old, &unconstrained, &inv_masses);
        let (positions, _) = lincs_solution(
            &Lincs::with_order(constraints, 8),
            &old,
            &unconstrained,
            &inv_masses,
        );
        let difference = max_difference(&positions, &reference);
        assert!(difference < 1e-6, "|LINCS - SHAKE| = {difference:e}");
    }

    #[test]
    fn test_lincs_branched_and_triangle_topologies_agree_with_shake() {
        let mut rng = Lcg::new(31);

        // Four bonds sharing the central atom.
        let (star, star_constraints) = methane_star();
        let star_inv_masses = vec![1.0 / 12.0, 1.0, 1.0, 1.0, 1.0];
        let unconstrained = perturbed(&star, 1e-3, &mut rng);
        let reference = shake_solution(&star_constraints, &star, &unconstrained, &star_inv_masses);
        let (positions, _) = lincs_solution(
            &Lincs::new(star_constraints),
            &star,
            &unconstrained,
            &star_inv_masses,
        );
        let difference = max_difference(&positions, &reference);
        assert!(difference < 1e-8, "star: |LINCS - SHAKE| = {difference:e}");

        // Rigid water as three coupled distance constraints (a closed triangle: the
        // expansion converges slowly, spectral radius ~0.77, hence the high order).
        let water = tip3p_molecule();
        let water_constraints = water_shake_constraints(0);
        let water_inv_masses = vec![1.0 / 15.9994, 1.0 / 1.008, 1.0 / 1.008];
        let unconstrained = perturbed(&water, 1e-3, &mut rng);
        let reference = shake_solution(
            &water_constraints,
            &water,
            &unconstrained,
            &water_inv_masses,
        );
        let (positions, _) = lincs_solution(
            &Lincs::with_order(water_constraints, 24),
            &water,
            &unconstrained,
            &water_inv_masses,
        );
        let difference = max_difference(&positions, &reference);
        assert!(difference < 1e-6, "water: |LINCS - SHAKE| = {difference:e}");
    }

    #[test]
    fn test_lincs_conserves_the_centre_of_mass_and_the_momentum_of_the_correction() {
        let old = zigzag_chain(8, 120.0);
        let constraints = chain_constraints(0, 7, 1.0);
        let masses = [12.0, 1.0, 14.0, 1.0, 16.0, 1.0, 12.0, 1.0];
        let inv_masses: Vec<Real> = masses.iter().map(|m| 1.0 / m).collect();
        let mut rng = Lcg::new(37);
        let unconstrained = perturbed(&old, 5e-3, &mut rng);
        let (positions, _) =
            lincs_solution(&Lincs::new(constraints), &old, &unconstrained, &inv_masses);
        let mut impulse = Vec3::zeros();
        let mut moved = 0.0;
        for ((mass, after), before) in masses
            .iter()
            .zip(positions.iter())
            .zip(unconstrained.iter())
        {
            impulse += (after - before) * *mass;
            moved += (after - before).norm();
        }
        assert!(moved > 1e-3, "the constraints must act: moved {moved}");
        assert!(impulse.norm() < 1e-14, "net constraint impulse {impulse:?}");
    }

    #[test]
    fn test_lincs_leaves_satisfied_constraints_untouched() {
        let old = zigzag_chain(10, 109.5);
        let constraints = chain_constraints(0, 9, 1.0);
        let inv_masses = vec![1.0; 10];
        // Rigidly translated reference: all constraints hold, nothing may move.
        let shifted: Vec<Vec3> = old.iter().map(|p| p + Vec3::new(0.3, -0.2, 0.1)).collect();
        let (positions, violation) =
            lincs_solution(&Lincs::new(constraints), &old, &shifted, &inv_masses);
        assert!(violation < 1e-15, "violation {violation:e}");
        assert!(max_difference(&positions, &shifted) < 1e-15);
    }

    #[test]
    fn test_lincs_skips_constraints_it_cannot_solve() {
        // Atoms 0 and 1 have infinite mass; atoms 2 and 3 coincide in the reference (no bond
        // direction); constraint 2-4 is fine; 5-5 is a constraint of an atom with itself.
        let constraints = vec![
            BondConstraint::new(0, 1, 1.0),
            BondConstraint::new(2, 3, 1.0),
            BondConstraint::new(2, 4, 1.0),
            BondConstraint::new(5, 5, 1.0),
        ];
        let old = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(9.0, 0.0, 0.0),
        ];
        let inv_masses = vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut positions = old.clone();
        positions[1] = Vec3::new(1.2, 0.0, 0.0);
        positions[4] = Vec3::new(6.3, 0.1, 0.0);
        let lincs = Lincs::new(constraints);
        let violation = lincs.apply(&mut positions, &old, &inv_masses);
        // The healthy bond 2-4 is solved ...
        let length = (positions[2] - positions[4]).norm();
        assert!((length - 1.0).abs() < 1e-12, "length {length}");
        // ... the unsolvable ones are reported through the violation, not corrupted.
        assert_eq!(positions[0], old[0]);
        assert_eq!(positions[1], Vec3::new(1.2, 0.0, 0.0));
        assert!(positions.iter().all(|p| p.iter().all(|x| x.is_finite())));
        let worst = lincs
            .constraints
            .iter()
            .map(|c| {
                let r = positions[c.atom_i] - positions[c.atom_j];
                (r.norm_squared() - c.distance_sq).abs() / c.distance_sq
            })
            .fold(0.0, Real::max);
        assert!((violation - worst).abs() < 1e-14, "{violation} vs {worst}");
        assert!(
            violation >= 1.2_f64 * 1.2 - 1.0 - 1e-12,
            "violation {violation}"
        );
    }

    #[test]
    fn test_lincs_ignores_redundant_repeated_constraints() {
        // A constraint listed twice (also with its atoms swapped) is redundant: taken at face
        // value it makes `I - A` singular (`A_cd = -1`) and the series would diverge.
        let old = zigzag_chain(6, 109.5);
        let constraints = chain_constraints(0, 5, 1.0);
        let mut redundant = constraints.clone();
        redundant.push(BondConstraint::new(2, 3, 1.0));
        redundant.push(BondConstraint::new(1, 0, 1.0));
        redundant.insert(0, BondConstraint::new(4, 5, 1.0));
        let inv_masses = vec![1.0; 6];
        let mut rng = Lcg::new(41);
        let unconstrained = perturbed(&old, 1e-3, &mut rng);
        let (expected, expected_violation) =
            lincs_solution(&Lincs::new(constraints), &old, &unconstrained, &inv_masses);
        let (positions, violation) =
            lincs_solution(&Lincs::new(redundant), &old, &unconstrained, &inv_masses);
        assert!(expected_violation < 1e-6, "{expected_violation:e}");
        assert!(violation < 1e-6, "{violation:e}");
        assert!(max_difference(&positions, &expected) < 1e-12);
    }

    #[test]
    fn test_lincs_survives_a_bond_rotated_by_more_than_ninety_degrees() {
        let constraints = vec![
            BondConstraint::new(0, 1, 1.0),
            BondConstraint::new(1, 2, 1.0),
        ];
        let old = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        // Atom 1 flung far off the axis: |across| > d, so the corrected projection is clamped.
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.2, 1.5, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        let violation = Lincs::new(constraints).apply(&mut positions, &old, &[1.0; 3]);
        assert!(violation.is_finite());
        assert!(positions.iter().all(|p| p.iter().all(|x| x.is_finite())));
    }

    #[test]
    fn test_lincs_reports_nan_for_non_finite_positions() {
        let constraints = vec![BondConstraint::new(0, 1, 1.0)];
        let old = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let mut positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(f64::NAN, 0.0, 0.0)];
        let violation = Lincs::new(constraints).apply(&mut positions, &old, &[1.0, 1.0]);
        assert!(violation.is_nan(), "violation {violation}");
    }
}
