// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for electrostatic embedding of a QM region in external
//! (MM) point charges, and unit conversions used by every QM engine.
//!
//! Positions in a [`QmRegion`] are in Å, charges in units of e, energies in
//! kcal/mol and forces in kcal/(mol·Å).  The QM engines work in atomic units, so
//! the external electrostatic potential at a nucleus is reported in Hartree/e:
//!
//! φ_A = Σ_J Q_J / R_AJ   (R_AJ in Bohr)

use super::{QmError, QmRegion};

/// Ångström → Bohr.
pub(crate) const ANG_TO_BOHR: f64 = 1.0 / 0.529_177_210_903;
/// Hartree → kcal/mol.
pub(crate) const HARTREE_TO_KCAL: f64 = 627.509_474_063_1;
/// Hartree/Bohr → kcal/(mol·Å).
pub(crate) const HARTREE_PER_BOHR_TO_KCAL_PER_ANG: f64 = HARTREE_TO_KCAL * ANG_TO_BOHR;
/// Minimum allowed distance (Å) between an external charge and a QM nucleus.
const MIN_CHARGE_DISTANCE_ANG: f64 = 1e-6;
/// Distance (Å) below which two QM atoms are considered to coincide.
pub(crate) const MIN_ATOM_DISTANCE_ANG: f64 = 1e-3;

/// Reject a geometry in which two atoms (numerically) coincide.
///
/// Every QM engine needs a non-zero distance between any two nuclei; a region that
/// was never populated (all atoms hydrogen at the origin, see [`QmRegion::new`])
/// fails here instead of being computed.
///
/// # Errors
/// [`QmError::CoincidentAtoms`] for the first offending pair.
pub(crate) fn check_no_coincident_atoms(positions_ang: &[[f64; 3]]) -> Result<(), QmError> {
    for (i, pos_i) in positions_ang.iter().enumerate() {
        for (j, pos_j) in positions_ang.iter().enumerate().skip(i + 1) {
            if distance(pos_i, pos_j) < MIN_ATOM_DISTANCE_ANG {
                return Err(QmError::CoincidentAtoms {
                    first: i,
                    second: j,
                });
            }
        }
    }
    Ok(())
}

/// Validate the geometric input of `region` and reset all of its outputs.
///
/// Called at the start of every engine so that a failed calculation never leaves
/// the energy / forces of a previous step behind.  Returns the external
/// electrostatic potential φ_A (Hartree/e) at every QM nucleus.
pub(crate) fn begin_run(region: &mut QmRegion) -> Result<Vec<f64>, QmError> {
    let n = region.atomic_numbers.len();
    region.energy = 0.0;
    region.embedding_energy = 0.0;
    region.forces = vec![[0.0; 3]; n];
    region.partial_charges = vec![0.0; n];
    region.external_charge_forces = vec![[0.0; 3]; region.external_charges.len()];
    if n == 0 {
        return Err(QmError::EmptyRegion);
    }
    if region.positions.len() != n {
        return Err(QmError::DimensionMismatch {
            what: "QM positions vs atomic numbers",
            expected: n,
            found: region.positions.len(),
        });
    }
    check_no_coincident_atoms(&region.positions)?;
    external_potential(&region.positions, &region.external_charges)
}

/// External electrostatic potential φ_A = Σ_J Q_J / R_AJ (Hartree/e) at each QM
/// nucleus.
///
/// Fails if a charge is non-finite or (numerically) coincides with a nucleus.
pub(crate) fn external_potential(
    positions_ang: &[[f64; 3]],
    external_charges: &[([f64; 3], f64)],
) -> Result<Vec<f64>, QmError> {
    let mut phi = vec![0.0_f64; positions_ang.len()];
    for (index, (pos_j, q_j)) in external_charges.iter().enumerate() {
        if !q_j.is_finite() || pos_j.iter().any(|x| !x.is_finite()) {
            return Err(QmError::InvalidExternalCharge { index });
        }
        for (phi_a, pos_a) in phi.iter_mut().zip(positions_ang.iter()) {
            let r_ang = distance(pos_a, pos_j);
            if r_ang < MIN_CHARGE_DISTANCE_ANG {
                return Err(QmError::InvalidExternalCharge { index });
            }
            *phi_a += q_j / (r_ang * ANG_TO_BOHR);
        }
    }
    Ok(phi)
}

/// Forces (kcal/(mol·Å)) on the external charges from the derivatives of the
/// QM energy with respect to the nuclear potentials, dE/dφ_A (Hartree per
/// Hartree/e, i.e. units of e).
///
/// Valid whenever the external charges enter the QM energy only through
/// φ_A (PM3 and SCC-DFTB):
///
/// F_J = −Σ_A (∂E/∂φ_A) ∂φ_A/∂R_J,   ∂φ_A/∂R_J = Q_J (R_A − R_J)/R_AJ³.
pub(crate) fn forces_from_potential_derivatives(
    positions_ang: &[[f64; 3]],
    external_charges: &[([f64; 3], f64)],
    de_dphi: &[f64],
) -> Vec<[f64; 3]> {
    external_charges
        .iter()
        .map(|(pos_j, q_j)| {
            let mut f = [0.0_f64; 3];
            for (pos_a, &dedphi) in positions_ang.iter().zip(de_dphi.iter()) {
                let d_bohr = [
                    (pos_a[0] - pos_j[0]) * ANG_TO_BOHR,
                    (pos_a[1] - pos_j[1]) * ANG_TO_BOHR,
                    (pos_a[2] - pos_j[2]) * ANG_TO_BOHR,
                ];
                let r2 = d_bohr[0] * d_bohr[0] + d_bohr[1] * d_bohr[1] + d_bohr[2] * d_bohr[2];
                let r3 = r2 * r2.sqrt();
                for (f_k, d_k) in f.iter_mut().zip(d_bohr.iter()) {
                    *f_k -= dedphi * q_j * d_k / r3;
                }
            }
            [
                f[0] * HARTREE_PER_BOHR_TO_KCAL_PER_ANG,
                f[1] * HARTREE_PER_BOHR_TO_KCAL_PER_ANG,
                f[2] * HARTREE_PER_BOHR_TO_KCAL_PER_ANG,
            ]
        })
        .collect()
}

/// Euclidean distance between two points.
pub(crate) fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Central finite-difference forces (kcal/(mol·Å)) on the QM nuclei of an energy
/// function `energy(positions) -> Hartree`, with step `delta_ang` (Å).
///
/// Any failure of a displaced single point is propagated instead of being
/// replaced by a default energy.
pub(crate) fn finite_difference_forces<F>(
    positions_ang: &[[f64; 3]],
    delta_ang: f64,
    mut energy: F,
) -> Result<Vec<[f64; 3]>, QmError>
where
    F: FnMut(&[[f64; 3]]) -> Result<f64, QmError>,
{
    let mut forces = vec![[0.0_f64; 3]; positions_ang.len()];
    let mut displaced = positions_ang.to_vec();
    for (i, f_i) in forces.iter_mut().enumerate() {
        for (k, f_ik) in f_i.iter_mut().enumerate() {
            let original = displaced[i][k];
            displaced[i][k] = original + delta_ang;
            let e_plus = energy(&displaced)?;
            displaced[i][k] = original - delta_ang;
            let e_minus = energy(&displaced)?;
            displaced[i][k] = original;
            *f_ik = -(e_plus - e_minus) / (2.0 * delta_ang) * HARTREE_TO_KCAL;
        }
    }
    Ok(forces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_difference_forces_match_a_harmonic_gradient() {
        // E = ½ κ Σ |r_i|² (Hartree, Å) has the force −κ r_i per atom; the result is
        // reported in kcal/(mol·Å).
        let kappa = 0.01;
        let positions = [[0.3, -0.2, 0.1], [1.0, 0.5, -0.4]];
        let forces = finite_difference_forces(&positions, 1e-3, |pos| {
            Ok(0.5
                * kappa
                * pos
                    .iter()
                    .map(|p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
                    .sum::<f64>())
        })
        .expect("harmonic energy never fails");
        for (f, p) in forces.iter().zip(positions.iter()) {
            for k in 0..3 {
                assert!(
                    (f[k] + kappa * p[k] * HARTREE_TO_KCAL).abs() < 1e-9,
                    "force {} vs {}",
                    f[k],
                    -kappa * p[k] * HARTREE_TO_KCAL
                );
            }
        }
    }

    #[test]
    fn a_failed_displaced_single_point_aborts_the_gradient() {
        // 2 atoms x 3 components x 2 displacements = 12 single points; a failure at
        // any of them must be reported (not replaced by a default energy), and the
        // evaluation stops at the first failure.
        let positions = [[0.0; 3], [1.0, 0.0, 0.0]];
        for failing_call in 0..12 {
            let mut calls = 0usize;
            let result = finite_difference_forces(&positions, 1e-3, |_| {
                let this_call = calls;
                calls += 1;
                if this_call == failing_call {
                    Err(QmError::ScfNotConverged)
                } else {
                    Ok(-1.0)
                }
            });
            assert_eq!(
                result,
                Err(QmError::ScfNotConverged),
                "failure injected at call {failing_call}"
            );
            assert_eq!(calls, failing_call + 1, "must stop at the first failure");
        }
        let result =
            finite_difference_forces(&positions, 1e-3, |_| Err(QmError::EigensolverFailed));
        assert_eq!(result, Err(QmError::EigensolverFailed));
    }

    #[test]
    fn external_potential_rejects_charges_on_top_of_a_nucleus() {
        let positions = [[0.0; 3], [0.74, 0.0, 0.0]];
        assert_eq!(
            external_potential(&positions, &[([0.0, 0.0, 0.0], 0.5)]),
            Err(QmError::InvalidExternalCharge { index: 0 })
        );
        assert_eq!(
            external_potential(
                &positions,
                &[([3.0, 0.0, 0.0], 1.0), ([0.0, 9.0, 0.0], f64::NAN)]
            ),
            Err(QmError::InvalidExternalCharge { index: 1 })
        );
        // A charge Q at 1 Å (= ANG_TO_BOHR Bohr) from a nucleus gives φ = Q / R.
        let phi = external_potential(&[[0.0; 3]], &[([1.0, 0.0, 0.0], 0.5)]).expect("valid charge");
        assert!((phi[0] - 0.5 / ANG_TO_BOHR).abs() < 1e-12, "φ = {}", phi[0]);
    }
}
