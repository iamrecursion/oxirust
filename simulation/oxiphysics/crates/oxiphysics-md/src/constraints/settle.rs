// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! SETTLE: analytic constraint solver for rigid three-site water.

use super::ConstraintStats;
use crate::MdError;
use oxiphysics_core::math::{Real, Vec3};

/// Relative size below which the reference geometry of a molecule is degenerate (coincident
/// hydrogens, or an oxygen on the H-H line).
const FRAME_TOLERANCE: Real = 1e-9;
/// Smallest squared cosine of the tilt of the molecular plane for which the closed-form
/// rotation is still well conditioned (the tilt must stay clear of 90 degrees).
const MIN_COS_TILT_SQ: Real = 1e-12;
/// Smallest `det / trace²` of the in-plane inertia of the rigid triangle (collinear sites
/// have a vanishing determinant).
const MIN_INERTIA_CONDITION: Real = 1e-12;
/// Relative amount by which `|torque|` may exceed the largest in-plane torque a rotation can
/// produce before the molecule is declared unsolvable (rounding noise is clamped).
const TORQUE_TOLERANCE: Real = 1e-9;
/// Convergence tolerance on `|r - d|` of [`SettleWater::apply_iterative`].
const ITERATIVE_TOLERANCE: Real = 1e-8;
/// Sweep limit of [`SettleWater::apply_iterative`].
const ITERATIVE_MAX_SWEEPS: u32 = 500;

/// SETTLE: analytic constraint solver for rigid three-site water (O, H1, H2).
///
/// Solves the three distance constraints (O-H1, O-H2, H1-H2) in closed form, without
/// iterating: S. Miyamoto & P. A. Kollman, J. Comput. Chem. 13, 952 (1992).
///
/// SETTLE returns exactly the positions the converged [`super::Shake`] iteration would, i.e.
/// the constrained positions `r_i = r_i^unc + sum_c (+-) lambda_c r_ref,c / m_i` with
/// constraint forces along the bonds `r_ref` of the *reference* (start-of-step) molecule.
/// Three consequences of that form give the closed-form solution:
///
/// * the constraint forces are internal, so the centre of mass is unchanged and so is the
///   angular momentum about it, `sum_i m_i (r_ref,i - R) x (r_i - r_i^unc) = 0`;
/// * every displacement lies in the reference plane, so the out-of-plane coordinates of the
///   constrained molecule equal those of the unconstrained one;
/// * the constrained molecule is the rigid triangle (side lengths `d_oh`, `d_oh`, `d_hh`)
///   rotated about its centre of mass.
///
/// In the reference frame (z = plane normal) the rigid triangle is `R c_i`, with `c_i` the
/// canonical positions relative to the centre of mass.  The out-of-plane equations fix the
/// third row of the rotation `R` (two angles, `sin(phi)` and `sin(psi)` in the paper), and
/// the angular-momentum equation `A cos(theta) + B sin(theta) = T` fixes the remaining
/// rotation about the normal in closed form.
///
/// The molecule must not move so far in one step that no rigid rotation can explain the
/// displacement (several tenths of a bond length); [`SettleWater::apply_settle`] then
/// reports an error instead of returning garbage.
pub struct SettleWater {
    /// O-H bond length (Å or nm, same units as positions).
    pub d_oh: Real,
    /// H-H distance (Å or nm).
    pub d_hh: Real,
}

impl SettleWater {
    /// Create a SETTLE solver for TIP3P water (d_OH = 0.9572 Å, angle = 104.52°).
    pub fn tip3p() -> Self {
        let d_oh: Real = 0.9572;
        let angle_rad = 104.52_f64.to_radians();
        let d_hh = (2.0 * d_oh * d_oh * (1.0 - angle_rad.cos())).sqrt();
        Self { d_oh, d_hh }
    }

    /// Create a SETTLE solver for SPC water (d_OH = 1.0 Å, angle = 109.47°).
    pub fn spc() -> Self {
        let d_oh: Real = 1.0;
        let angle_rad = 109.47_f64.to_radians();
        let d_hh = (2.0 * d_oh * d_oh * (1.0 - angle_rad.cos())).sqrt();
        Self { d_oh, d_hh }
    }

    /// Constrain one water molecule `[O, H1, H2]` with the analytic SETTLE solution.
    ///
    /// `old` are the constrained positions at the start of the step (they fix the directions
    /// of the constraint forces and the orientation of the molecular plane), `new` the
    /// unconstrained positions after the step, overwritten with the constrained ones.  `m_o`
    /// and `m_h` are the oxygen and (both) hydrogen masses; use
    /// [`SettleWater::apply_settle`] for unequal hydrogen masses.
    ///
    /// # Errors
    ///
    /// [`MdError::SettleDegenerate`] as for [`SettleWater::apply_settle`]; `new` is left
    /// untouched in that case.
    pub fn apply(
        &self,
        old: &[Vec3; 3],
        new: &mut [Vec3; 3],
        m_o: Real,
        m_h: Real,
    ) -> Result<(), MdError> {
        self.apply_settle(old, new, [m_o, m_h, m_h])
    }

    /// Constrain one water molecule `[O, H1, H2]` with the analytic SETTLE solution for
    /// arbitrary site masses `masses = [m_O, m_H1, m_H2]` (e.g. HDO).
    ///
    /// The constrained positions satisfy all three distances to rounding error (about
    /// `1e-15` relative), keep the centre of mass of `new`, and equal the converged SHAKE
    /// solution computed from the same `old` and `new`.
    ///
    /// # Errors
    ///
    /// [`MdError::SettleDegenerate`] if a mass is not positive and finite, `d_oh` / `d_hh` do
    /// not form a triangle (`0 < d_hh < 2 d_oh`), a coordinate is not finite, the reference
    /// molecule `old` is degenerate (coincident hydrogens, or the oxygen on the H-H line),
    /// or the displacement is so large that no rigid rotation reproduces it.  `new` is left
    /// untouched in that case.
    pub fn apply_settle(
        &self,
        old: &[Vec3; 3],
        new: &mut [Vec3; 3],
        masses: [Real; 3],
    ) -> Result<(), MdError> {
        let constrained = self.solve(old, new, masses)?;
        *new = constrained;
        Ok(())
    }

    /// Constrain many molecules at once: atoms `3m, 3m + 1, 3m + 2` of `old` / `new` are
    /// the `[O, H1, H2]` of molecule `m`; all molecules share the masses `m_o`, `m_h`.
    ///
    /// # Errors
    ///
    /// [`MdError::General`] if `old` and `new` differ in length or their length is not a
    /// multiple of 3; otherwise the first [`MdError::SettleDegenerate`] of
    /// [`SettleWater::apply`], tagged with the molecule index.  Molecules before the failing
    /// one are already constrained; it and the following ones are untouched.
    pub fn apply_all(
        &self,
        old: &[Vec3],
        new: &mut [Vec3],
        m_o: Real,
        m_h: Real,
    ) -> Result<(), MdError> {
        if old.len() != new.len() {
            return Err(MdError::General(format!(
                "SETTLE: {} reference atoms but {} atoms to constrain",
                old.len(),
                new.len()
            )));
        }
        let (old_molecules, old_rest) = old.as_chunks::<3>();
        let (new_molecules, new_rest) = new.as_chunks_mut::<3>();
        if !old_rest.is_empty() || !new_rest.is_empty() {
            return Err(MdError::General(format!(
                "SETTLE: {} atoms are not a whole number of 3-site molecules",
                old.len()
            )));
        }
        for (index, (old_molecule, new_molecule)) in old_molecules
            .iter()
            .zip(new_molecules.iter_mut())
            .enumerate()
        {
            self.apply(old_molecule, new_molecule, m_o, m_h)
                .map_err(|error| match error {
                    MdError::SettleDegenerate(reason) => {
                        MdError::SettleDegenerate(format!("molecule {index}: {reason}"))
                    }
                    other => other,
                })?;
        }
        Ok(())
    }

    /// Iterative constraint of one molecule by projection along the *current* bonds.
    ///
    /// Sweeps over the three constraints, displacing the two atoms of a violated constraint
    /// along their present bond vector in inverse proportion to their masses, until every
    /// distance is within `1e-8` of its target or 500 sweeps have run.  It needs no
    /// reference positions, but is **not** SETTLE and not equivalent to SHAKE either: the
    /// corrections do not act along the bonds of the start of the step, so the result
    /// differs from [`SettleWater::apply`] by an amount of the order of the squared
    /// displacement, and the scheme is not time-reversible.  It conserves the centre of mass.
    ///
    /// `inv_masses` are `1/m` of `[O, H1, H2]`; constraints between two atoms of infinite
    /// mass are skipped.  The returned [`ConstraintStats`] reports the sweeps, the maximum
    /// relative violation afterwards ([`SettleWater::max_violation`]) and whether the
    /// tolerance was reached.
    pub fn apply_iterative(
        &self,
        positions: &mut [Vec3; 3],
        inv_masses: [Real; 3],
    ) -> ConstraintStats {
        let constraints = [
            (0_usize, 1_usize, self.d_oh),
            (0, 2, self.d_oh),
            (1, 2, self.d_hh),
        ];
        let mut sweeps = 0_u32;
        let mut converged = false;
        while sweeps < ITERATIVE_MAX_SWEEPS {
            sweeps += 1;
            let mut all_satisfied = true;
            for &(i, j, target) in &constraints {
                let bond = positions[i] - positions[j];
                let length = bond.norm();
                if length < 1e-15 {
                    continue;
                }
                let error = length - target;
                if error.abs() < ITERATIVE_TOLERANCE {
                    continue;
                }
                all_satisfied = false;
                let inv_mass_sum = inv_masses[i] + inv_masses[j];
                if inv_mass_sum < 1e-30 {
                    continue;
                }
                let correction = bond * (error / (length * inv_mass_sum));
                positions[i] -= correction * inv_masses[i];
                positions[j] += correction * inv_masses[j];
            }
            if all_satisfied {
                converged = true;
                break;
            }
        }
        ConstraintStats {
            iterations: sweeps,
            max_violation: self.max_violation(&positions[..]),
            converged,
        }
    }

    /// Maximum constraint violation for a water molecule.
    pub fn max_violation(&self, positions: &[Vec3]) -> Real {
        let checks = [
            ((0, 1), self.d_oh),
            ((0, 2), self.d_oh),
            ((1, 2), self.d_hh),
        ];
        checks
            .iter()
            .map(|&((i, j), target)| {
                let r = (positions[i] - positions[j]).norm();
                (r - target).abs() / target
            })
            .fold(0.0_f64, f64::max)
    }

    /// The closed-form constrained positions, or an error; `new` is not modified.
    fn solve(
        &self,
        old: &[Vec3; 3],
        new: &[Vec3; 3],
        masses: [Real; 3],
    ) -> Result<[Vec3; 3], MdError> {
        let degenerate = MdError::SettleDegenerate;
        let (d_oh, d_hh) = (self.d_oh, self.d_hh);
        if !(d_oh.is_finite() && d_hh.is_finite() && d_oh > 0.0 && d_hh > 0.0 && d_hh < 2.0 * d_oh)
        {
            return Err(degenerate(format!(
                "invalid water geometry d_OH = {d_oh}, d_HH = {d_hh} (need 0 < d_HH < 2 d_OH)"
            )));
        }
        if !masses.iter().all(|m| m.is_finite() && *m > 0.0) {
            return Err(degenerate(format!(
                "masses must be positive and finite, got {masses:?}"
            )));
        }
        if !old
            .iter()
            .chain(new.iter())
            .all(|p| p.iter().all(|x| x.is_finite()))
        {
            return Err(degenerate("non-finite coordinate".to_string()));
        }

        // Canonical rigid triangle in its own plane (sites O, H1, H2), then centred on the
        // centre of mass: offsets c_i and the in-plane second moments of the mass.
        let half_hh = 0.5 * d_hh;
        let height = ((d_oh - half_hh) * (d_oh + half_hh)).sqrt();
        let total_mass: Real = masses.iter().sum();
        let sites = [[half_hh, height], [0.0, 0.0], [d_hh, 0.0]];
        let mut centre = [0.0, 0.0];
        for (mass, site) in masses.iter().zip(sites.iter()) {
            centre[0] += mass * site[0] / total_mass;
            centre[1] += mass * site[1] / total_mass;
        }
        let offsets = sites.map(|site| [site[0] - centre[0], site[1] - centre[1]]);
        let (mut ixx, mut ixy, mut iyy) = (0.0, 0.0, 0.0);
        for (mass, c) in masses.iter().zip(offsets.iter()) {
            ixx += mass * c[0] * c[0];
            ixy += mass * c[0] * c[1];
            iyy += mass * c[1] * c[1];
        }
        let det = ixx * iyy - ixy * ixy;
        let well_conditioned = det > MIN_INERTIA_CONDITION * (ixx + iyy) * (ixx + iyy);
        if !well_conditioned {
            return Err(degenerate(format!(
                "the rigid triangle (d_OH = {d_oh}, d_HH = {d_hh}) is collinear"
            )));
        }

        // Frame of the reference molecule: x along H1 -> H2, y towards O, z = x × y.
        let hh_vector = old[2] - old[1];
        let hh_length = hh_vector.norm();
        if hh_length <= FRAME_TOLERANCE * d_hh {
            return Err(degenerate(
                "reference hydrogens coincide, the molecular plane is undefined".to_string(),
            ));
        }
        let ex = hh_vector / hh_length;
        let h1_to_o = old[0] - old[1];
        let across = h1_to_o - ex * h1_to_o.dot(&ex);
        let across_length = across.norm();
        if across_length <= FRAME_TOLERANCE * d_oh {
            return Err(degenerate(
                "reference oxygen lies on the H-H line, the molecular plane is undefined"
                    .to_string(),
            ));
        }
        let ey = across / across_length;
        let ez = ex.cross(&ey);

        // Unconstrained molecule about its centre of mass (which SETTLE preserves), in the
        // reference frame.
        let centre_of_mass =
            (new[0] * masses[0] + new[1] * masses[1] + new[2] * masses[2]) / total_mass;
        let local = new.map(|site| {
            let relative = site - centre_of_mass;
            [relative.dot(&ex), relative.dot(&ey), relative.dot(&ez)]
        });

        // Out-of-plane coordinates: z_i = R_zx c_ix + R_zy c_iy fixes the third row of R.
        let (mut bx, mut by) = (0.0, 0.0);
        for ((mass, c), p) in masses.iter().zip(offsets.iter()).zip(local.iter()) {
            bx += mass * p[2] * c[0];
            by += mass * p[2] * c[1];
        }
        let tilt_x = (iyy * bx - ixy * by) / det;
        let tilt_y = (ixx * by - ixy * bx) / det;
        let cos_tilt_sq = 1.0 - tilt_x * tilt_x - tilt_y * tilt_y;
        let tilt_is_solvable = cos_tilt_sq > MIN_COS_TILT_SQ;
        if !tilt_is_solvable {
            return Err(degenerate(format!(
                "the molecule left its reference plane by more than a rigid rotation allows \
                 (cos² tilt = {cos_tilt_sq}); reduce the time step"
            )));
        }
        let normal_row = Vec3::new(tilt_x, tilt_y, cos_tilt_sq.sqrt());
        // Any rotation Q with that third row; the free rotation Rz(theta) about the plane
        // normal is applied on top of it.
        let q_x = Vec3::new(
            1.0 - tilt_x * tilt_x,
            -tilt_x * tilt_y,
            -tilt_x * normal_row.z,
        ) / (1.0 - tilt_x * tilt_x).sqrt();
        let q_y = normal_row.cross(&q_x);

        // In-plane angular momentum about the centre of mass: with R = Rz(theta) Q,
        //   A cos(theta) + B sin(theta) = T,  T = sum_i m_i (c_ix y_i - c_iy x_i).
        let mut torque = 0.0;
        for ((mass, c), p) in masses.iter().zip(offsets.iter()).zip(local.iter()) {
            torque += mass * (c[0] * p[1] - c[1] * p[0]);
        }
        let a = ixx * q_y.x + ixy * (q_y.y - q_x.x) - iyy * q_x.y;
        let b = ixx * q_x.x + ixy * (q_x.y + q_y.x) + iyy * q_y.y;
        let radius_sq = a * a + b * b;
        let discriminant = radius_sq - torque * torque;
        if !(radius_sq > 0.0 && discriminant >= -TORQUE_TOLERANCE * radius_sq) {
            return Err(degenerate(format!(
                "no rotation about the reference normal reproduces the in-plane angular \
                 momentum (T = {torque}, |(A, B)| = {}); reduce the time step",
                radius_sq.sqrt()
            )));
        }
        // Of the two solutions of the equation take the one nearer to no rotation (the
        // larger cos(theta)).
        let root = discriminant.max(0.0).sqrt();
        let branch = if b >= 0.0 { 1.0 } else { -1.0 };
        let cos_theta = (a * torque + branch * b * root) / radius_sq;
        let sin_theta = (b * torque - branch * a * root) / radius_sq;
        let r_x = q_x * cos_theta - q_y * sin_theta;
        let r_y = q_x * sin_theta + q_y * cos_theta;

        let constrained = offsets.map(|c| {
            let along_x = r_x.x * c[0] + r_x.y * c[1];
            let along_y = r_y.x * c[0] + r_y.y * c[1];
            let along_z = normal_row.x * c[0] + normal_row.y * c[1];
            centre_of_mass + ex * along_x + ey * along_y + ez * along_z
        });
        if !constrained.iter().all(|p| p.iter().all(|x| x.is_finite())) {
            return Err(degenerate("non-finite constrained coordinate".to_string()));
        }
        Ok(constrained)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{BondConstraint, Shake, water_shake_constraints};
    use oxiphysics_core::math::Mat3;

    const TIP3P_MASSES: [Real; 3] = [15.9994, 1.008, 1.008];

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

        fn vector(&mut self, scale: Real) -> Vec3 {
            Vec3::new(self.symmetric(), self.symmetric(), self.symmetric()) * scale
        }
    }

    /// Rotation by `angle` about `axis` (Rodrigues' formula).
    fn rotation(axis: Vec3, angle: Real) -> Mat3 {
        let k = axis.normalize();
        let cross = Mat3::new(0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0);
        Mat3::identity() + cross * angle.sin() + cross * cross * (1.0 - angle.cos())
    }

    fn random_orientation(rng: &mut Lcg) -> Mat3 {
        let axis = rng.vector(1.0) + Vec3::new(0.0, 0.0, 0.2);
        rotation(axis, std::f64::consts::PI * rng.symmetric())
    }

    /// Rigid water `[O, H1, H2]` of the solver's geometry, rotated and moved to `origin`.
    fn rigid_molecule(solver: &SettleWater, orientation: &Mat3, origin: Vec3) -> [Vec3; 3] {
        let half_hh = 0.5 * solver.d_hh;
        let height = ((solver.d_oh - half_hh) * (solver.d_oh + half_hh)).sqrt();
        let body = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(half_hh, height, 0.0),
            Vec3::new(-half_hh, height, 0.0),
        ];
        body.map(|p| orientation * p + origin)
    }

    fn displaced(molecule: &[Vec3; 3], amplitude: Real, rng: &mut Lcg) -> [Vec3; 3] {
        molecule.map(|p| p + rng.vector(amplitude))
    }

    fn centre_of_mass(molecule: &[Vec3; 3], masses: [Real; 3]) -> Vec3 {
        let total: Real = masses.iter().sum();
        (molecule[0] * masses[0] + molecule[1] * masses[1] + molecule[2] * masses[2]) / total
    }

    fn shake_for(solver: &SettleWater) -> Shake {
        Shake::with_params(
            vec![
                BondConstraint::new(0, 1, solver.d_oh),
                BondConstraint::new(0, 2, solver.d_oh),
                BondConstraint::new(1, 2, solver.d_hh),
            ],
            100_000,
            1e-14,
        )
    }

    fn max_difference(a: &[Vec3], b: &[Vec3]) -> Real {
        a.iter()
            .zip(b.iter())
            .map(|(p, q)| (p - q).norm())
            .fold(0.0, Real::max)
    }

    #[test]
    fn test_settle_geometries_follow_the_water_models() {
        let tip3p = SettleWater::tip3p();
        assert!((tip3p.d_oh - 0.9572).abs() < 1e-12);
        let expected_hh = 2.0 * 0.9572 * (104.52_f64.to_radians() / 2.0).sin();
        assert!((tip3p.d_hh - expected_hh).abs() < 1e-12, "{}", tip3p.d_hh);
        assert!((tip3p.d_hh - water_shake_constraints(0)[2].target_distance()).abs() < 1e-4);

        let spc = SettleWater::spc();
        assert!((spc.d_oh - 1.0).abs() < 1e-12);
        let expected_hh = 2.0 * (109.47_f64.to_radians() / 2.0).sin();
        assert!((spc.d_hh - expected_hh).abs() < 1e-12, "{}", spc.d_hh);
    }

    #[test]
    fn test_settle_satisfies_all_constraints_to_machine_precision_in_one_call() {
        let mut rng = Lcg::new(1);
        for solver in [SettleWater::tip3p(), SettleWater::spc()] {
            for trial in 0..300 {
                let origin = rng.vector(5.0);
                let old = rigid_molecule(&solver, &random_orientation(&mut rng), origin);
                assert!(
                    solver.max_violation(&old) < 1e-14,
                    "reference must be rigid"
                );
                let amplitude = [0.005, 0.02, 0.05][trial % 3];
                let mut new = displaced(&old, amplitude, &mut rng);
                let unconstrained = new;
                assert!(
                    solver.max_violation(&new) > 1e-4,
                    "the step must break the bonds"
                );
                solver
                    .apply(&old, &mut new, 15.9994, 1.008)
                    .expect("moderate displacements are solvable");
                let violation = solver.max_violation(&new);
                assert!(
                    violation < 1e-12,
                    "trial {trial}: relative violation {violation:e} after one call"
                );
                assert!(max_difference(&new, &unconstrained) > 1e-6);
            }
        }
    }

    #[test]
    fn test_settle_equals_the_converged_shake_solution() {
        let mut rng = Lcg::new(2);
        let cases = [
            (SettleWater::tip3p(), TIP3P_MASSES),
            (SettleWater::spc(), [16.0, 1.0, 1.0]),
            // HDO: unequal hydrogen masses.
            (SettleWater::tip3p(), [15.9994, 1.008, 2.014]),
        ];
        for (solver, masses) in cases {
            let shake = shake_for(&solver);
            let inv_masses: Vec<Real> = masses.iter().map(|m| 1.0 / m).collect();
            for trial in 0..100 {
                let old = rigid_molecule(&solver, &random_orientation(&mut rng), rng.vector(3.0));
                let unconstrained = displaced(&old, [0.01, 0.05][trial % 2], &mut rng);

                let mut settled = unconstrained;
                solver
                    .apply_settle(&old, &mut settled, masses)
                    .expect("solvable");
                let mut shaken = unconstrained.to_vec();
                shake.apply(&mut shaken, &old, &inv_masses);
                assert!(shake.max_relative_violation(&shaken) < 1e-12);

                let difference = max_difference(&settled, &shaken);
                assert!(
                    difference < 1e-10,
                    "masses {masses:?}, trial {trial}: |SETTLE - SHAKE| = {difference:e}"
                );
            }
        }
    }

    #[test]
    fn test_settle_conserves_centre_of_mass_and_angular_momentum() {
        let mut rng = Lcg::new(3);
        let solver = SettleWater::tip3p();
        for trial in 0..50 {
            let old = rigid_molecule(&solver, &random_orientation(&mut rng), rng.vector(4.0));
            let unconstrained = displaced(&old, 0.04, &mut rng);
            let mut settled = unconstrained;
            solver
                .apply(&old, &mut settled, TIP3P_MASSES[0], TIP3P_MASSES[1])
                .expect("solvable");

            // Linear momentum: the constraint impulse sum_i m_i (r_i - r_i^unc) vanishes, and
            // with it the shift of the centre of mass.
            let mut impulse = Vec3::zeros();
            let mut angular = Vec3::zeros();
            let old_com = centre_of_mass(&old, TIP3P_MASSES);
            let normal = (old[1] - old[0]).cross(&(old[2] - old[0])).normalize();
            let mut largest = 0.0_f64;
            for ((mass, reference), (after, before)) in TIP3P_MASSES
                .iter()
                .zip(old.iter())
                .zip(settled.iter().zip(unconstrained.iter()))
            {
                let shift = after - before;
                impulse += shift * *mass;
                angular += (reference - old_com).cross(&shift) * *mass;
                // The constraint forces act along the reference bonds, i.e. in the plane.
                assert!(
                    shift.dot(&normal).abs() < 1e-13,
                    "trial {trial}: out-of-plane shift"
                );
                largest = largest.max(shift.norm());
            }
            assert!(largest > 1e-4, "trial {trial}: constraints must act");
            assert!(impulse.norm() < 1e-13, "trial {trial}: impulse {impulse:?}");
            assert!(
                angular.norm() < 1e-13,
                "trial {trial}: angular momentum {angular:?}"
            );
            let shift_of_com = centre_of_mass(&settled, TIP3P_MASSES)
                - centre_of_mass(&unconstrained, TIP3P_MASSES);
            assert!(
                shift_of_com.norm() < 1e-13,
                "trial {trial}: COM moved {shift_of_com:?}"
            );
        }
    }

    #[test]
    fn test_settle_is_the_identity_on_a_rigid_motion_of_the_reference() {
        let mut rng = Lcg::new(4);
        let solver = SettleWater::spc();
        for _ in 0..20 {
            let old = rigid_molecule(&solver, &random_orientation(&mut rng), rng.vector(2.0));
            // Rotate by up to 20 degrees about the centre of mass and translate.
            let com = centre_of_mass(&old, [16.0, 1.0, 1.0]);
            let turn = rotation(
                rng.vector(1.0) + Vec3::new(0.1, 0.1, 0.1),
                0.35 * rng.symmetric(),
            );
            let shift = rng.vector(0.1);
            let moved = old.map(|p| turn * (p - com) + com + shift);
            let mut settled = moved;
            solver
                .apply(&old, &mut settled, 16.0, 1.0)
                .expect("solvable");
            assert!(
                max_difference(&settled, &moved) < 1e-13,
                "an already rigid molecule must not be changed"
            );
        }
    }

    #[test]
    fn test_settle_apply_equals_apply_settle_with_equal_hydrogen_masses() {
        let mut rng = Lcg::new(5);
        let solver = SettleWater::tip3p();
        let old = rigid_molecule(&solver, &random_orientation(&mut rng), rng.vector(1.0));
        let unconstrained = displaced(&old, 0.03, &mut rng);
        let mut with_apply = unconstrained;
        let mut with_masses = unconstrained;
        solver
            .apply(&old, &mut with_apply, 15.9994, 1.008)
            .expect("solvable");
        solver
            .apply_settle(&old, &mut with_masses, [15.9994, 1.008, 1.008])
            .expect("solvable");
        assert_eq!(with_apply, with_masses);
    }

    #[test]
    fn test_settle_apply_all_constrains_every_molecule() {
        let mut rng = Lcg::new(6);
        let solver = SettleWater::tip3p();
        let mut old = Vec::new();
        for _ in 0..5 {
            old.extend(rigid_molecule(
                &solver,
                &random_orientation(&mut rng),
                rng.vector(6.0),
            ));
        }
        let unconstrained: Vec<Vec3> = old.iter().map(|p| p + rng.vector(0.03)).collect();

        let mut batched = unconstrained.clone();
        solver
            .apply_all(&old, &mut batched, 15.9994, 1.008)
            .expect("solvable");
        for (molecule, (old_chunk, new_chunk)) in old
            .chunks_exact(3)
            .zip(unconstrained.chunks_exact(3))
            .enumerate()
        {
            let old_array = [old_chunk[0], old_chunk[1], old_chunk[2]];
            let mut single = [new_chunk[0], new_chunk[1], new_chunk[2]];
            solver
                .apply(&old_array, &mut single, 15.9994, 1.008)
                .expect("solvable");
            assert_eq!(&batched[3 * molecule..3 * molecule + 3], &single[..]);
            assert!(solver.max_violation(&batched[3 * molecule..3 * molecule + 3]) < 1e-12);
        }
    }

    #[test]
    fn test_settle_apply_all_validates_slices_and_names_the_failing_molecule() {
        let mut rng = Lcg::new(7);
        let solver = SettleWater::tip3p();
        let mut old = Vec::new();
        for _ in 0..4 {
            old.extend(rigid_molecule(
                &solver,
                &random_orientation(&mut rng),
                rng.vector(6.0),
            ));
        }
        let unconstrained: Vec<Vec3> = old.iter().map(|p| p + rng.vector(0.02)).collect();

        // Length problems are API misuse, not geometry.
        let mut short = unconstrained[..11].to_vec();
        assert!(matches!(
            solver.apply_all(&old, &mut short, 15.9994, 1.008),
            Err(MdError::General(_))
        ));
        let mut ragged = unconstrained[..11].to_vec();
        assert!(matches!(
            solver.apply_all(&old[..11], &mut ragged, 15.9994, 1.008),
            Err(MdError::General(_))
        ));

        // Molecule 2 has a collinear reference: molecules 0 and 1 are constrained, 2 and 3
        // are untouched.
        let mut broken_old = old.clone();
        broken_old[6] = Vec3::new(0.0, 0.0, 0.0);
        broken_old[7] = Vec3::new(-1.0, 0.0, 0.0);
        broken_old[8] = Vec3::new(1.0, 0.0, 0.0);
        let mut batched = unconstrained.clone();
        let error = solver
            .apply_all(&broken_old, &mut batched, 15.9994, 1.008)
            .expect_err("molecule 2 is degenerate");
        match error {
            MdError::SettleDegenerate(reason) => {
                assert!(reason.starts_with("molecule 2: "), "{reason}");
            }
            other => panic!("unexpected error {other:?}"),
        }
        assert!(solver.max_violation(&batched[0..3]) < 1e-12);
        assert!(solver.max_violation(&batched[3..6]) < 1e-12);
        assert_eq!(&batched[6..], &unconstrained[6..]);
    }

    #[test]
    fn test_settle_rejects_degenerate_input_without_touching_the_positions() {
        let mut rng = Lcg::new(8);
        let solver = SettleWater::tip3p();
        let old = rigid_molecule(&solver, &random_orientation(&mut rng), rng.vector(1.0));
        let unconstrained = displaced(&old, 0.02, &mut rng);
        let is_degenerate =
            |result: Result<(), MdError>| matches!(result, Err(MdError::SettleDegenerate(_)));

        // Masses.
        for masses in [
            [0.0, 1.008, 1.008],
            [15.9994, -1.0, 1.008],
            [15.9994, 1.008, f64::NAN],
            [f64::INFINITY, 1.008, 1.008],
        ] {
            let mut new = unconstrained;
            assert!(
                is_degenerate(solver.apply_settle(&old, &mut new, masses)),
                "{masses:?}"
            );
            assert_eq!(new, unconstrained);
        }

        // Non-finite coordinates in either molecule.
        let mut bad_new = unconstrained;
        bad_new[1].x = f64::NAN;
        let bad_new_before = bad_new;
        assert!(is_degenerate(solver.apply(
            &old,
            &mut bad_new,
            15.9994,
            1.008
        )));
        assert_eq!(bad_new[0], bad_new_before[0]);
        let mut bad_old = old;
        bad_old[2].y = f64::INFINITY;
        let mut new = unconstrained;
        assert!(is_degenerate(
            solver.apply(&bad_old, &mut new, 15.9994, 1.008)
        ));
        assert_eq!(new, unconstrained);

        // Reference with the oxygen on the H-H line, and with coincident hydrogens.
        let collinear = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
        ];
        let mut new = unconstrained;
        assert!(is_degenerate(
            solver.apply(&collinear, &mut new, 15.9994, 1.008)
        ));
        let coincident = [old[0], old[1], old[1]];
        assert!(is_degenerate(solver.apply(
            &coincident,
            &mut new,
            15.9994,
            1.008
        )));
        assert_eq!(new, unconstrained);

        // Geometries that are not triangles.
        for (d_oh, d_hh) in [
            (1.0, 2.0),
            (1.0, 2.5),
            (0.0, 1.0),
            (1.0, -0.5),
            (f64::NAN, 1.0),
        ] {
            let invalid = SettleWater { d_oh, d_hh };
            let mut new = unconstrained;
            assert!(
                is_degenerate(invalid.apply(&old, &mut new, 15.9994, 1.008)),
                "d_OH = {d_oh}, d_HH = {d_hh}"
            );
        }
    }

    #[test]
    fn test_settle_reports_a_step_too_large_for_any_rigid_rotation() {
        let solver = SettleWater::tip3p();
        let old = rigid_molecule(&solver, &Mat3::identity(), Vec3::zeros());
        // The oxygen is thrown 10 Å out of the molecular plane: no rotation of the rigid
        // triangle reaches that, so SETTLE must say so instead of returning garbage.
        let mut new = old;
        new[0].z += 10.0;
        let before = new;
        let result = solver.apply(&old, &mut new, 15.9994, 1.008);
        assert!(
            matches!(result, Err(MdError::SettleDegenerate(_))),
            "{result:?}"
        );
        assert_eq!(new, before);
    }

    #[test]
    fn test_settle_iterative_projection_converges_but_is_reported_honestly() {
        let mut rng = Lcg::new(9);
        let solver = SettleWater::tip3p();
        let inv_masses = [1.0 / 15.9994, 1.0 / 1.008, 1.0 / 1.008];
        let old = rigid_molecule(&solver, &random_orientation(&mut rng), rng.vector(1.0));
        let mut positions = displaced(&old, 0.02, &mut rng);
        let before = positions;
        let stats = solver.apply_iterative(&mut positions, inv_masses);
        assert!(stats.converged, "{stats:?}");
        assert!(stats.iterations > 1 && stats.iterations < ITERATIVE_MAX_SWEEPS);
        assert!(stats.max_violation < 1e-7, "{stats:?}");
        assert_eq!(stats.max_violation, solver.max_violation(&positions));
        // Each correction moves two atoms in inverse proportion to their masses: the centre
        // of mass stays put.
        let masses = [15.9994, 1.008, 1.008];
        let drift = centre_of_mass(&positions, masses) - centre_of_mass(&before, masses);
        assert!(drift.norm() < 1e-13, "{drift:?}");

        // A molecule that already satisfies the constraints needs one (idle) sweep.
        let mut rigid = old;
        let stats = solver.apply_iterative(&mut rigid, inv_masses);
        assert!(stats.converged);
        assert_eq!(stats.iterations, 1);
        assert_eq!(rigid, old);

        // Atoms of infinite mass cannot be moved: the failure is reported, not hidden.
        let mut frozen = before;
        let stats = solver.apply_iterative(&mut frozen, [0.0; 3]);
        assert!(!stats.converged);
        assert_eq!(stats.iterations, ITERATIVE_MAX_SWEEPS);
        assert_eq!(frozen, before);
    }

    #[test]
    fn test_settle_max_violation_is_the_largest_relative_bond_error() {
        let solver = SettleWater::tip3p();
        let mut molecule = rigid_molecule(&solver, &Mat3::identity(), Vec3::zeros());
        assert!(solver.max_violation(&molecule) < 1e-15);
        // Stretch the H1-H2 distance by 1 %: the only violated constraint.
        let midpoint = (molecule[1] + molecule[2]) * 0.5;
        molecule[1] = midpoint + (molecule[1] - midpoint) * 1.01;
        molecule[2] = midpoint + (molecule[2] - midpoint) * 1.01;
        let violation = solver.max_violation(&molecule);
        assert!((violation - 0.01).abs() < 1e-3, "violation {violation}");
    }
}
