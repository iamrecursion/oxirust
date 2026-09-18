//! Regression tests for the non-bonded pair search of [`CgSystem`] and the cached spring list of
//! [`ElasticNetworkModel`].
//!
//! Every optimised evaluation is compared with an uncached reference that reproduces the
//! original algorithm: all pairs, a `HashSet` of bonded pairs for the exclusions, and the
//! contact map rescanned on every call.

use super::functions::{add, dot, length, normalize, scale, sub};
use super::types::{
    BondExclusions, CELL_GRID_MIN_BEADS, CgAngle, CgBead, CgBond, CgSystem, ElasticNetworkModel,
    MartiniLj, for_each_pair_within_cutoff,
};
use std::collections::HashSet;
use std::f64::consts::PI;

/// Deterministic uniform `[0, 1)` generator (64-bit LCG with the Knuth MMIX constants).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn unit_vector(&mut self) -> [f64; 3] {
        let z = 2.0 * self.uniform() - 1.0;
        let phi = 2.0 * PI * self.uniform();
        let rho = (1.0 - z * z).sqrt();
        [rho * phi.cos(), rho * phi.sin(), z]
    }

    fn point_in_cube(&mut self, edge: f64) -> [f64; 3] {
        [
            edge * self.uniform(),
            edge * self.uniform(),
            edge * self.uniform(),
        ]
    }
}

/// Self-avoiding random-walk chain with harmonic bonds (0.47 nm), angles, and bead types
/// cycling through the five MARTINI classes.  No two beads end up closer than ~0.35 nm, which
/// keeps the Lennard-Jones forces in a range where relative comparisons are meaningful.
fn random_walk_chain(n_beads: usize, cutoff: f64, seed: u64) -> CgSystem {
    let mut rng = Lcg::new(seed);
    let mut sys = CgSystem::new(cutoff);
    let mut placed: Vec<[f64; 3]> = Vec::with_capacity(n_beads);
    let mut pos = [0.0; 3];
    for idx in 0..n_beads {
        sys.add_bead(CgBead::new(
            format!("B{idx}"),
            (idx % 5) as u8,
            72.0,
            0.0,
            pos,
        ));
        placed.push(pos);
        let mut next = pos;
        for _ in 0..64 {
            let candidate = add(pos, scale(rng.unit_vector(), 0.47));
            let crowded = placed
                .iter()
                .rev()
                .skip(1)
                .any(|other| length(sub(candidate, *other)) < 0.35);
            next = candidate;
            if !crowded {
                break;
            }
        }
        pos = next;
    }
    for idx in 0..n_beads.saturating_sub(1) {
        sys.bonds.push(CgBond::new(idx, idx + 1, 0.47, 1250.0));
    }
    for idx in 0..n_beads.saturating_sub(2) {
        sys.angles
            .push(CgAngle::new(idx, idx + 1, idx + 2, 2.0, 25.0));
    }
    sys
}

/// The original non-bonded algorithm: all pairs, `HashSet` exclusions, a square root for
/// every pair.  Returns the energy and the per-bead non-bonded force.
fn reference_nonbonded(sys: &CgSystem) -> (f64, Vec<[f64; 3]>) {
    let n = sys.beads.len();
    let mut excluded: HashSet<(usize, usize)> = HashSet::new();
    for bond in &sys.bonds {
        let pair = if bond.i < bond.j {
            (bond.i, bond.j)
        } else {
            (bond.j, bond.i)
        };
        excluded.insert(pair);
    }
    let mut energy = 0.0;
    let mut forces = vec![[0.0; 3]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if excluded.contains(&(i, j)) {
                continue;
            }
            let rij = sub(sys.beads[j].pos, sys.beads[i].pos);
            let r = length(rij);
            if r > sys.cutoff {
                continue;
            }
            let (sigma, epsilon) =
                MartiniLj::get_lj_params(sys.beads[i].bead_type, sys.beads[j].bead_type);
            energy += MartiniLj::energy(r, sigma, epsilon);
            if r < 1e-10 {
                continue;
            }
            let fvec = scale(
                normalize(rij),
                MartiniLj::force_magnitude(r, sigma, epsilon),
            );
            forces[i] = sub(forces[i], fvec);
            forces[j] = add(forces[j], fvec);
        }
    }
    (energy, forces)
}

fn assert_forces_close(actual: &[[f64; 3]], expected: &[[f64; 3]], rel_tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let largest = expected
        .iter()
        .flatten()
        .fold(1.0_f64, |m, v| m.max(v.abs()));
    for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
        for axis in 0..3 {
            assert!(
                (a[axis] - e[axis]).abs() <= rel_tol * largest,
                "{what}: bead {idx} axis {axis}: {} vs reference {} (largest force {largest:e})",
                a[axis],
                e[axis]
            );
        }
    }
}

/// `compute_nonbonded_energy` and `compute_forces` of a chain with `n_beads` beads against the
/// reference implementation; `uses_grid` states which pair search the chain size selects.
fn assert_chain_matches_reference(n_beads: usize, uses_grid: bool, seed: u64) {
    assert_eq!(
        n_beads > CELL_GRID_MIN_BEADS,
        uses_grid,
        "{n_beads} beads select the other pair search"
    );
    let mut sys = random_walk_chain(n_beads, 2.0, seed);
    // Same geometry with a zero cutoff: no non-bonded pair is inside it, so the forces
    // are the bonded ones alone.
    let mut bonded_only = random_walk_chain(n_beads, 0.0, seed);
    sys.compute_forces();
    bonded_only.compute_forces();

    let (e_ref, nb_ref) = reference_nonbonded(&sys);
    let largest_nb = nb_ref.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(e_ref.abs() > 1.0, "the chain must have non-bonded energy");
    assert!(largest_nb > 1.0, "the chain must have non-bonded forces");
    assert_eq!(bonded_only.compute_nonbonded_energy(), 0.0);

    let energy = sys.compute_nonbonded_energy();
    assert!(
        (energy - e_ref).abs() <= 1e-12 * e_ref.abs(),
        "{n_beads} beads: non-bonded energy {energy} vs reference {e_ref}"
    );
    let expected: Vec<[f64; 3]> = bonded_only
        .beads
        .iter()
        .zip(nb_ref.iter())
        .map(|(bead, nb)| add(bead.force, *nb))
        .collect();
    let actual: Vec<[f64; 3]> = sys.beads.iter().map(|bead| bead.force).collect();
    assert_forces_close(&actual, &expected, 1e-12, &format!("{n_beads} beads"));
}

#[test]
fn test_cgsystem_200_bead_chain_matches_the_hash_set_reference() {
    assert_chain_matches_reference(200, false, 11);
}

#[test]
fn test_cgsystem_600_bead_chain_on_the_cell_grid_matches_the_hash_set_reference() {
    assert_chain_matches_reference(600, true, 23);
}

#[test]
fn test_cgsystem_bonded_pairs_are_excluded_in_either_bond_order() {
    for (bond_i, bond_j) in [(0, 1), (1, 0)] {
        let mut sys = CgSystem::new(2.0);
        sys.add_bead(CgBead::new("A", 0, 72.0, 0.0, [0.0, 0.0, 0.0]));
        sys.add_bead(CgBead::new("B", 0, 72.0, 0.0, [0.4, 0.0, 0.0]));
        // A bond with kb = 0 exerts no force of its own: it only excludes the pair.
        sys.bonds.push(CgBond::new(bond_i, bond_j, 0.4, 0.0));
        assert_eq!(
            sys.compute_nonbonded_energy(),
            0.0,
            "bond ({bond_i}, {bond_j})"
        );
        sys.compute_forces();
        assert_eq!(sys.beads[0].force, [0.0; 3], "bond ({bond_i}, {bond_j})");
        assert_eq!(sys.beads[1].force, [0.0; 3], "bond ({bond_i}, {bond_j})");

        sys.bonds.clear();
        let free_energy = sys.compute_nonbonded_energy();
        let (sigma, epsilon) = MartiniLj::get_lj_params(0, 0);
        assert_eq!(free_energy, MartiniLj::energy(0.4, sigma, epsilon));
    }
}

#[test]
fn test_cgsystem_exclusions_follow_edits_of_the_public_bond_list() {
    // `bonds` is public: replacing a bond by another pair keeps `bonds.len()` unchanged,
    // which a cache keyed on the length would not notice.
    let mut sys = CgSystem::new(2.0);
    for x in [0.0, 0.4, 0.8] {
        sys.add_bead(CgBead::new("B", 0, 72.0, 0.0, [x, 0.0, 0.0]));
    }
    let (sigma, epsilon) = MartiniLj::get_lj_params(0, 0);
    let pair_energy = |r: f64| MartiniLj::energy(r, sigma, epsilon);

    sys.bonds.push(CgBond::new(0, 1, 0.4, 0.0));
    let with_01 = sys.compute_nonbonded_energy();
    assert_eq!(with_01, pair_energy(0.4) + pair_energy(0.8));
    sys.compute_forces();
    let forces_with_01 = sys.beads[1].force;

    sys.bonds[0] = CgBond::new(1, 2, 0.4, 0.0);
    let with_12 = sys.compute_nonbonded_energy();
    assert_eq!(with_12, pair_energy(0.4) + pair_energy(0.8));
    sys.compute_forces();
    assert_ne!(
        sys.beads[1].force, forces_with_01,
        "the excluded pair changed"
    );

    sys.bonds[0] = CgBond::new(0, 2, 0.8, 0.0);
    assert_eq!(sys.compute_nonbonded_energy(), 2.0 * pair_energy(0.4));
}

#[test]
fn test_cgsystem_includes_pairs_exactly_at_the_cutoff_only() {
    let (sigma, epsilon) = MartiniLj::get_lj_params(0, 0);
    for (separation, interacts) in [(1.0, true), (1.0 + 1e-9, false), (1.0 - 1e-9, true)] {
        let mut sys = CgSystem::new(1.0);
        sys.add_bead(CgBead::new("A", 0, 72.0, 0.0, [0.0, 0.0, 0.0]));
        sys.add_bead(CgBead::new("B", 0, 72.0, 0.0, [separation, 0.0, 0.0]));
        let expected = if interacts {
            MartiniLj::energy(separation, sigma, epsilon)
        } else {
            0.0
        };
        assert_eq!(sys.compute_nonbonded_energy(), expected, "r = {separation}");
        sys.compute_forces();
        assert_eq!(sys.beads[0].force[0] != 0.0, interacts, "r = {separation}");
    }
}

#[test]
fn test_bond_exclusions_are_sorted_symmetric_and_deduplicated() {
    let bonds = vec![
        CgBond::new(3, 1, 0.4, 1.0),
        CgBond::new(1, 3, 0.4, 1.0),
        CgBond::new(0, 1, 0.4, 1.0),
        CgBond::new(2, 2, 0.4, 1.0),
        CgBond::new(1, 9, 0.4, 1.0),
        CgBond::new(4, 1, 0.4, 1.0),
    ];
    let exclusions = BondExclusions::new(5, &bonds);
    assert_eq!(exclusions.partners_of(1), &[0, 3, 4]);
    assert_eq!(exclusions.partners_of(0), &[1]);
    assert_eq!(exclusions.partners_of(3), &[1]);
    assert_eq!(exclusions.partners_of(4), &[1]);
    assert!(
        exclusions.partners_of(2).is_empty(),
        "a self-bond excludes nothing"
    );
    assert!(
        exclusions.partners_of(9).is_empty() && exclusions.partners_of(usize::MAX).is_empty(),
        "unknown beads have no partners"
    );
    for (i, j, bonded) in [
        (3, 1, true),
        (1, 3, true),
        (0, 1, true),
        (1, 4, true),
        (0, 3, false),
        (2, 3, false),
        (1, 9, false),
    ] {
        assert_eq!(exclusions.contains(i, j), bonded, "pair ({i}, {j})");
    }
    assert!(BondExclusions::new(0, &[]).partners_of(0).is_empty());
}

/// Pairs reported by `for_each_pair_within_cutoff` that pass the exact `r <= cutoff` test.
fn grid_pairs(positions: &[[f64; 3]], cutoff: f64) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for_each_pair_within_cutoff(positions, cutoff, |i, j, rij, r2| {
        // Pairs with a non-finite coordinate have a NaN vector, which never compares equal.
        if rij.iter().all(|x| x.is_finite()) {
            assert_eq!(
                rij,
                sub(positions[j], positions[i]),
                "pair ({i}, {j}) vector"
            );
            assert_eq!(r2, dot(rij, rij), "pair ({i}, {j}) squared distance");
        }
        if length(rij) <= cutoff {
            pairs.push((i, j));
        }
    });
    pairs
}

fn brute_force_pairs(positions: &[[f64; 3]], cutoff: f64) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, pos_i) in positions.iter().enumerate() {
        for (j, pos_j) in positions.iter().enumerate().skip(i + 1) {
            if length(sub(*pos_j, *pos_i)) <= cutoff {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

fn assert_pairs_match_brute_force(positions: &[[f64; 3]], cutoff: f64, what: &str) {
    let found = grid_pairs(positions, cutoff);
    let expected = brute_force_pairs(positions, cutoff);
    assert!(
        !expected.is_empty(),
        "{what}: the geometry must contain pairs"
    );
    assert_eq!(
        found, expected,
        "{what}: pairs (ascending order, exact set)"
    );
}

#[test]
fn test_pair_search_on_the_cell_grid_matches_the_all_pairs_loop() {
    let mut rng = Lcg::new(5);
    // Random points (negative coordinates included) in a 6 nm cube.
    let random: Vec<[f64; 3]> = (0..500)
        .map(|_| sub(rng.point_in_cube(6.0), [3.0, 3.0, 3.0]))
        .collect();
    assert_pairs_match_brute_force(&random, 1.0, "random cube");

    // Collinear points: the bounding box has no extent along y and z.
    let line: Vec<[f64; 3]> = (0..400).map(|k| [0.3 * f64::from(k), 0.0, 0.0]).collect();
    assert_pairs_match_brute_force(&line, 1.0, "collinear");

    // Two clusters 10⁴ nm apart: a sparse system must not need a huge grid.
    let clusters: Vec<[f64; 3]> = (0..300)
        .map(|k| {
            let offset = if k % 2 == 0 { 0.0 } else { 1.0e4 };
            add(rng.point_in_cube(3.0), [offset, 0.0, 0.0])
        })
        .collect();
    assert_pairs_match_brute_force(&clusters, 1.0, "two distant clusters");

    // Lattice with spacing exactly equal to the cutoff: nearest neighbours sit on the
    // boundary and must be kept, face diagonals (sqrt 2 away) must not.
    let mut lattice = Vec::new();
    for ix in 0..8 {
        for iy in 0..8 {
            for iz in 0..8 {
                lattice.push([f64::from(ix), f64::from(iy), f64::from(iz)]);
            }
        }
    }
    assert_pairs_match_brute_force(&lattice, 1.0, "unit lattice");
    assert_eq!(grid_pairs(&lattice, 1.0).len(), 3 * 8 * 8 * 7);
}

#[test]
fn test_pair_search_far_pairs_are_not_wrapped_together_by_the_periodic_grid() {
    // Two beads at opposite ends of a long, thin cloud: their periodic images must not make
    // them neighbours in free space.
    let mut rng = Lcg::new(17);
    let mut positions: Vec<[f64; 3]> = (0..300)
        .map(|_| {
            [
                20.0 * rng.uniform(),
                0.1 * rng.uniform(),
                0.1 * rng.uniform(),
            ]
        })
        .collect();
    positions[0] = [0.0, 0.0, 0.0];
    positions[1] = [20.0, 0.0, 0.0];
    let found = grid_pairs(&positions, 1.0);
    assert!(!found.contains(&(0, 1)));
    assert_eq!(found, brute_force_pairs(&positions, 1.0));
}

#[test]
fn test_pair_search_without_a_usable_grid_falls_back_to_all_pairs() {
    let mut rng = Lcg::new(29);
    let mut positions: Vec<[f64; 3]> = (0..300).map(|_| rng.point_in_cube(4.0)).collect();
    let finite = grid_pairs(&positions, 1.0);
    assert_eq!(finite, brute_force_pairs(&positions, 1.0));

    // One non-finite coordinate: no grid can be built, pairs not involving it are unchanged.
    positions[7][1] = f64::NAN;
    let found = grid_pairs(&positions, 1.0);
    let expected: Vec<(usize, usize)> = finite
        .iter()
        .copied()
        .filter(|&(i, j)| i != 7 && j != 7)
        .collect();
    assert_eq!(found, expected);

    // A zero cutoff finds only coincident points; an empty input is a no-op.
    assert!(grid_pairs(&positions, 0.0).is_empty());
    assert!(grid_pairs(&[], 1.0).is_empty());
    assert!(grid_pairs(&[[0.0; 3]], 1.0).is_empty());
}

// ---------------------------------------------------------------------------
// ElasticNetworkModel
// ---------------------------------------------------------------------------

fn reference_contact_map(positions: &[[f64; 3]], cutoff: f64) -> Vec<(usize, usize)> {
    brute_force_pairs(positions, cutoff)
}

/// The original, uncached potential energy: contact map rescanned on every call.
fn reference_enm_energy(enm: &ElasticNetworkModel, current: &[[f64; 3]]) -> f64 {
    reference_contact_map(&enm.positions, enm.cutoff)
        .iter()
        .map(|&(i, j)| {
            let r0 = length(sub(enm.positions[j], enm.positions[i]));
            let r = length(sub(current[j], current[i]));
            let dr = r - r0;
            0.5 * enm.spring_constant * dr * dr
        })
        .sum()
}

/// The original, uncached forces.
fn reference_enm_forces(enm: &ElasticNetworkModel, current: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let mut forces = vec![[0.0_f64; 3]; current.len()];
    for (i, j) in reference_contact_map(&enm.positions, enm.cutoff) {
        let r0 = length(sub(enm.positions[j], enm.positions[i]));
        let rij = sub(current[j], current[i]);
        let r = length(rij);
        if r < 1e-15 {
            continue;
        }
        let mag = enm.spring_constant * (r - r0) / r;
        let fij = scale(rij, mag);
        forces[i] = add(forces[i], fij);
        forces[j] = sub(forces[j], fij);
    }
    forces
}

/// The original, uncached Hessian.
fn reference_enm_hessian(enm: &ElasticNetworkModel) -> Vec<Vec<f64>> {
    let dim = 3 * enm.positions.len();
    let mut h = vec![vec![0.0_f64; dim]; dim];
    for (i, j) in reference_contact_map(&enm.positions, enm.cutoff) {
        let rij = sub(enm.positions[j], enm.positions[i]);
        let r2 = dot(rij, rij);
        if r2 < 1e-15 {
            continue;
        }
        for a in 0..3 {
            for b in 0..3 {
                let val = enm.spring_constant * rij[a] * rij[b] / r2;
                h[3 * i + a][3 * j + b] -= val;
                h[3 * j + b][3 * i + a] -= val;
                h[3 * i + a][3 * i + b] += val;
                h[3 * j + a][3 * j + b] += val;
            }
        }
    }
    h
}

/// Random reference structure and a displaced copy of it.
fn random_structure(n_beads: usize, edge: f64, seed: u64) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut rng = Lcg::new(seed);
    let reference: Vec<[f64; 3]> = (0..n_beads).map(|_| rng.point_in_cube(edge)).collect();
    let current: Vec<[f64; 3]> = reference
        .iter()
        .map(|p| add(*p, scale(rng.unit_vector(), 0.05 * rng.uniform())))
        .collect();
    (reference, current)
}

fn assert_enm_matches_reference(enm: &ElasticNetworkModel, current: &[[f64; 3]], what: &str) {
    assert_eq!(
        enm.contact_map(),
        reference_contact_map(&enm.positions, enm.cutoff),
        "{what}: contact map"
    );
    let e_ref = reference_enm_energy(enm, current);
    let e = enm.potential_energy(current);
    assert!(
        (e - e_ref).abs() <= 1e-12 * e_ref.abs().max(1.0),
        "{what}: energy {e} vs reference {e_ref}"
    );
    assert_forces_close(
        &enm.forces(current),
        &reference_enm_forces(enm, current),
        1e-12,
        what,
    );
    let h = enm.hessian();
    let h_ref = reference_enm_hessian(enm);
    let scale_h = enm.spring_constant.abs().max(1.0);
    for (row, (h_row, ref_row)) in h.iter().zip(h_ref.iter()).enumerate() {
        for (col, (a, b)) in h_row.iter().zip(ref_row.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * scale_h,
                "{what}: Hessian[{row}][{col}] = {a} vs reference {b}"
            );
        }
    }
}

#[test]
fn test_enm_50_random_beads_match_the_uncached_reference() {
    let (reference, current) = random_structure(50, 1.2, 7);
    let enm = ElasticNetworkModel::new(reference, 0.5, 10.0);
    assert!(
        enm.contact_map().len() > 50,
        "the structure must be well connected"
    );
    assert!(enm.potential_energy(&current) > 0.0);
    assert!(enm.contacts_are_current());
    assert_enm_matches_reference(&enm, &current, "50 beads");
}

#[test]
fn test_enm_400_beads_use_the_cell_grid_and_match_the_uncached_reference() {
    let n_beads = 400;
    assert!(n_beads > CELL_GRID_MIN_BEADS);
    let (reference, current) = random_structure(n_beads, 4.0, 13);
    let enm = ElasticNetworkModel::new(reference, 0.6, 7.5);
    assert!(enm.contact_map().len() > 400);
    assert_enm_matches_reference(&enm, &current, "400 beads");
}

#[test]
fn test_enm_contacts_include_pairs_exactly_at_the_cutoff() {
    let positions = vec![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let enm = ElasticNetworkModel::new(positions, 0.5, 10.0);
    assert_eq!(enm.contact_map(), vec![(0, 1), (1, 2)]);
    let zero_energy = enm.potential_energy(&enm.positions);
    assert_eq!(zero_energy, 0.0);
}

#[test]
fn test_enm_stays_consistent_when_public_fields_are_edited() {
    let (reference, current) = random_structure(60, 1.2, 3);
    let mut enm = ElasticNetworkModel::new(reference, 0.5, 10.0);
    let n_contacts = enm.contact_map().len();
    assert!(enm.contacts_are_current());

    // Edited cutoff: the cached springs are stale, the results must not be.
    enm.cutoff = 0.3;
    assert!(!enm.contacts_are_current());
    assert!(enm.contact_map().len() < n_contacts);
    assert_enm_matches_reference(&enm, &current, "cutoff edited");
    enm.rebuild_contacts();
    assert!(enm.contacts_are_current());
    assert_enm_matches_reference(&enm, &current, "cutoff rebuilt");

    // Edited reference position: likewise.
    enm.positions[4] = [5.0, 5.0, 5.0];
    assert!(!enm.contacts_are_current());
    assert!(
        enm.contact_map().iter().all(|&(i, j)| i != 4 && j != 4),
        "bead 4 moved out of reach of every other bead"
    );
    assert_enm_matches_reference(&enm, &current, "position edited");
    enm.rebuild_contacts();
    assert!(enm.contacts_are_current());
    assert_enm_matches_reference(&enm, &current, "position rebuilt");

    // The spring constant is not part of the cache: editing it needs no rebuild.
    enm.spring_constant = 25.0;
    assert!(enm.contacts_are_current());
    assert_enm_matches_reference(&enm, &current, "spring constant edited");
}

#[test]
fn test_enm_with_non_finite_reference_positions_does_not_panic() {
    let mut positions = vec![[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]];
    positions[2][0] = f64::NAN;
    let enm = ElasticNetworkModel::new(positions.clone(), 0.5, 10.0);
    // The cache can never match a NaN reference, so every call rebuilds: still correct.
    assert!(!enm.contacts_are_current());
    assert_eq!(enm.contact_map(), vec![(0, 1)]);
    assert_eq!(enm.potential_energy(&positions), 0.0);
    assert_eq!(enm.forces(&positions), vec![[0.0; 3]; 3]);
}
