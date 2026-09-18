//! Regression tests for the Verlet pair list of [`AlloySystem`] and for the
//! analytic cross-species pair force.
//!
//! The reference for the pair search is the brute-force `O(N²)` loop that the
//! list replaced, re-implemented here from [`SimBox::min_image`] alone; the
//! reference for the cross-species force is a central finite difference of
//! [`cross_pair_potential`].

use super::functions::*;
use super::types::*;

type Pair = (usize, usize, [f64; 3], f64);

/// Deterministic uniform numbers in `[0, 1)`.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The search that [`AlloySystem::build_neighbor_list`] used to be.
fn brute_force_pairs(sys: &AlloySystem) -> Vec<Pair> {
    let rc = sys.max_cutoff();
    let mut pairs = Vec::new();
    for i in 0..sys.atoms.len() {
        for j in (i + 1)..sys.atoms.len() {
            let dr = sys
                .sim_box
                .min_image(vsub(sys.atoms[j].position, sys.atoms[i].position));
            let r = vnorm(dr);
            if r < rc && r > 0.1 {
                pairs.push((i, j, dr, r));
            }
        }
    }
    pairs
}

/// A copper crystal of `n_cells`³ cells with every coordinate displaced by up to
/// `amplitude` Å, in a box with the tilt `xy`.  Atoms are wrapped into the box.
fn perturbed_crystal(n_cells: usize, xy: f64, amplitude: f64, seed: u64) -> AlloySystem {
    let cu = EamParams::copper();
    let a = cu.lattice_a;
    let sim_box = SimBox::cubic(n_cells as f64 * a).with_xy_tilt(xy);
    let mut sys = AlloySystem::new(vec![cu], sim_box, MixingRule::Geometric, 0.001);
    let mut rng = XorShift(seed);
    for position in generate_crystal_positions(a, n_cells, CrystalStructure::FCC) {
        let displaced = [
            position[0] + amplitude * (2.0 * rng.next() - 1.0),
            position[1] + amplitude * (2.0 * rng.next() - 1.0),
            position[2] + amplitude * (2.0 * rng.next() - 1.0),
        ];
        let wrapped = sys.sim_box.wrap(displaced);
        sys.add_atom(wrapped, 0);
    }
    sys
}

/// The same atoms in a fresh system: no cached pair list.
fn without_cache(sys: &AlloySystem) -> AlloySystem {
    let mut fresh = AlloySystem::new(
        sys.species.clone(),
        sys.sim_box.clone(),
        sys.mixing_rule,
        sys.dt,
    );
    fresh.atoms = sys.atoms.clone();
    fresh
}

#[test]
fn cell_search_finds_exactly_the_brute_force_pairs() {
    // (cells per edge, tilt): from a box smaller than the cutoff to boxes with
    // several cells per axis, orthorhombic and tilted either way.
    for (n_cells, xy) in [
        (2, 0.0),
        (3, 0.0),
        (3, 1.5),
        (4, 0.0),
        (4, -1.2),
        (6, 0.0),
        (8, 0.0),
        (8, 3.0),
        (10, -2.5),
    ] {
        let sys = perturbed_crystal(n_cells, xy, 0.25, 7 + n_cells as u64);
        let expected = brute_force_pairs(&sys);
        let found = sys.build_neighbor_list();
        assert!(!expected.is_empty());
        assert_eq!(
            found.len(),
            expected.len(),
            "{n_cells} cells, tilt {xy}: pair count"
        );
        assert_eq!(found, expected, "{n_cells} cells, tilt {xy}");
    }
}

#[test]
fn binary_alloy_pair_search_uses_the_largest_species_cutoff() {
    let mut sys = AlloySystem::new(
        vec![EamParams::copper(), EamParams::aluminum()],
        SimBox::cubic(20.0),
        MixingRule::Johnson,
        0.001,
    );
    let mut rng = XorShift(99);
    for _ in 0..400 {
        let p = [20.0 * rng.next(), 20.0 * rng.next(), 20.0 * rng.next()];
        let species = usize::from(rng.next() > 0.5);
        sys.add_atom(p, species);
    }
    // The largest cutoff of the two species (6.0 Å for aluminium) sets the list.
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));
}

#[test]
fn pair_list_is_reused_until_an_atom_moves_half_a_skin() {
    let mut sys = perturbed_crystal(6, 0.0, 0.2, 3);
    sys.compute_forces();
    assert_eq!(sys.neighbor_list_rebuilds(), 1);

    // 0.1 Å is below half of the 0.3 Å skin: the list is kept.
    sys.atoms[7].position[0] += 0.1;
    sys.compute_forces();
    assert_eq!(sys.neighbor_list_rebuilds(), 1);
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));

    // Another atom moving the same way does not add up: the criterion is per atom.
    sys.atoms[9].position[2] -= 0.1;
    sys.compute_forces();
    assert_eq!(sys.neighbor_list_rebuilds(), 1);
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));

    // 0.2 Å in total exceeds 0.15 Å: rebuilt.
    sys.atoms[7].position[0] += 0.1;
    sys.compute_forces();
    assert_eq!(sys.neighbor_list_rebuilds(), 2);
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));

    // A whole period of the box is no displacement at all.
    let period = sys.sim_box.lengths[1];
    sys.atoms[3].position[1] += period;
    sys.compute_forces();
    assert_eq!(sys.neighbor_list_rebuilds(), 2);
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));
}

#[test]
fn potential_energy_shares_the_list_but_never_trusts_a_stale_one() {
    let mut sys = perturbed_crystal(5, 0.0, 0.15, 11);
    sys.compute_forces();
    let e_shared = sys.potential_energy();
    assert_eq!(sys.neighbor_list_rebuilds(), 1, "energy must not rebuild");
    assert_eq!(e_shared, without_cache(&sys).potential_energy());

    // Moving an atom by a lot makes the cached list stale; the energy is
    // nevertheless exact because it falls back to a temporary list.
    sys.atoms[4].position[1] += 0.9;
    assert_eq!(
        sys.potential_energy(),
        without_cache(&sys).potential_energy()
    );
    assert_eq!(sys.neighbor_list_rebuilds(), 1, "`&self` cannot rebuild");

    // So does a changed box, atom count or cutoff.
    sys.compute_forces();
    sys.sim_box.lengths[0] *= 1.02;
    assert_eq!(
        sys.potential_energy(),
        without_cache(&sys).potential_energy()
    );
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));
    sys.compute_forces();
    sys.add_atom([1.0, 1.0, 1.0], 0);
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));
    sys.compute_forces();
    sys.species[0].cutoff = 3.4;
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));
    assert_eq!(
        sys.potential_energy(),
        without_cache(&sys).potential_energy()
    );
    sys.sim_box.xy = 0.8;
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));
}

#[test]
fn degenerate_boxes_and_non_finite_atoms_neither_hang_nor_pair_up() {
    // A NaN coordinate and an infinite one have no neighbours.
    let mut sys = perturbed_crystal(4, 0.0, 0.1, 5);
    sys.atoms[10].position[0] = f64::NAN;
    sys.atoms[11].position[1] = f64::INFINITY;
    let pairs = sys.build_neighbor_list();
    assert!(
        pairs
            .iter()
            .all(|&(i, j, _, _)| i != 10 && j != 10 && i != 11 && j != 11)
    );
    assert_eq!(pairs, brute_force_pairs(&sys));
    sys.compute_forces();
    assert!(sys.atoms[10].force.iter().all(|c| *c == 0.0));

    // A huge but finite coordinate is wrapped exactly and stays a valid atom.
    let mut sys = perturbed_crystal(4, 0.0, 0.1, 5);
    sys.atoms[20].position = [1.0e300, -1.0e300, 3.0e12];
    let list = sys.build_neighbor_list();
    assert_eq!(list, brute_force_pairs(&sys));

    // Degenerate edges behave as before: the axis is simply not periodic.
    for edges in [
        [0.0, 14.0, 14.0],
        [14.0, f64::NAN, 14.0],
        [14.0, 14.0, -3.0],
    ] {
        let mut sys = perturbed_crystal(3, 0.0, 0.1, 5);
        sys.sim_box.lengths = edges;
        assert_eq!(
            sys.build_neighbor_list(),
            brute_force_pairs(&sys),
            "{edges:?}"
        );
        sys.compute_forces();
        assert!(
            sys.atoms
                .iter()
                .all(|a| a.force.iter().all(|c| c.is_finite()))
        );
    }
    let mut sys = perturbed_crystal(3, 0.0, 0.1, 5);
    sys.sim_box.xy = f64::NAN;
    assert_eq!(sys.build_neighbor_list(), brute_force_pairs(&sys));
}

#[test]
fn candidate_search_scales_linearly_with_the_atom_count() {
    // 4000 and 6912 atoms: a brute-force search examines N/2 candidates per atom
    // (2000 and 3456) and its total grows with N^2 (a factor 3.0 here); the cell
    // search examines a number per atom that does not grow with N, so its total
    // grows like N (a factor 1.73).
    let mut totals = Vec::new();
    for n_cells in [10, 12] {
        let mut sys = perturbed_crystal(n_cells, 0.0, 0.1, 21);
        sys.compute_forces();
        assert_eq!(sys.neighbor_list_rebuilds(), 1);
        let atoms = sys.num_atoms();
        let candidates = sys.neighbor_list_candidates();
        let per_atom = candidates as f64 / atoms as f64;
        assert!(
            per_atom < 0.15 * atoms as f64 / 2.0,
            "{atoms} atoms: {per_atom} candidates per atom is not sub-quadratic"
        );
        assert!(
            per_atom < 400.0,
            "{atoms} atoms: {per_atom} candidates per atom"
        );
        totals.push((atoms, candidates));
    }
    let atom_ratio = totals[1].0 as f64 / totals[0].0 as f64;
    let candidate_ratio = totals[1].1 as f64 / totals[0].1 as f64;
    assert!(
        candidate_ratio < 1.4 * atom_ratio && candidate_ratio < 0.8 * atom_ratio * atom_ratio,
        "atoms x{atom_ratio:.2} but candidates x{candidate_ratio:.2}"
    );
}

#[test]
fn forces_from_the_grid_search_match_the_energy_gradient_and_conserve_momentum() {
    let mut sys = AlloySystem::new(
        vec![EamParams::copper(), EamParams::nickel()],
        SimBox::cubic(6.0 * 3.57).with_xy_tilt(1.0),
        MixingRule::Johnson,
        0.001,
    );
    let mut rng = XorShift(1234);
    let a = 3.57;
    for position in generate_crystal_positions(a, 6, CrystalStructure::FCC) {
        let displaced = [
            position[0] + 0.12 * (2.0 * rng.next() - 1.0),
            position[1] + 0.12 * (2.0 * rng.next() - 1.0),
            position[2] + 0.12 * (2.0 * rng.next() - 1.0),
        ];
        let species = usize::from(rng.next() > 0.5);
        let wrapped = sys.sim_box.wrap(displaced);
        sys.add_atom(wrapped, species);
    }
    sys.compute_forces();
    for axis in 0..3 {
        let total: f64 = sys.atoms.iter().map(|a| a.force[axis]).sum();
        assert!(
            total.abs() < 1.0e-8,
            "net force component {axis} = {total:e}"
        );
    }
    let h = 1.0e-5;
    for index in [0, 17, 300, 511, 863] {
        for axis in 0..3 {
            let mut plus = without_cache(&sys);
            plus.atoms[index].position[axis] += h;
            let mut minus = without_cache(&sys);
            minus.atoms[index].position[axis] -= h;
            let numeric = -(plus.potential_energy() - minus.potential_energy()) / (2.0 * h);
            let analytic = sys.atoms[index].force[axis];
            assert!(
                (analytic - numeric).abs() < 1.0e-5 * (1.0 + numeric.abs()),
                "atom {index} axis {axis}: {analytic} vs {numeric}"
            );
        }
    }
}

#[test]
fn trajectory_with_list_reuse_is_bitwise_identical_to_rebuilding_every_step() {
    let mut cached = perturbed_crystal(4, 0.0, 0.1, 17);
    let mut rng = XorShift(5);
    for atom in &mut cached.atoms {
        atom.velocity = [
            6.0 * (2.0 * rng.next() - 1.0),
            6.0 * (2.0 * rng.next() - 1.0),
            6.0 * (2.0 * rng.next() - 1.0),
        ];
    }
    cached.dt = 0.002;
    cached.compute_forces();
    let mut reference = without_cache(&cached);
    reference.compute_forces();
    for _ in 0..40 {
        cached.step().unwrap();
        // A fresh system rebuilds its list in every evaluation.
        let mut next = without_cache(&reference);
        next.step().unwrap();
        reference = next;
    }
    assert!(
        cached.neighbor_list_rebuilds() >= 2,
        "the atoms move far enough to force a rebuild, got {}",
        cached.neighbor_list_rebuilds()
    );
    assert!(
        cached.neighbor_list_rebuilds() < 40,
        "the list must be reused between rebuilds"
    );
    for (a, b) in cached.atoms.iter().zip(reference.atoms.iter()) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
        assert_eq!(a.force, b.force);
    }
}

/// Sample radii that are not too close to a non-smooth point of the cross
/// potential: the inner cut, either cutoff, and (where `skip_zero_crossings`)
/// a zero of either pair potential.
fn smooth_radii(p1: &EamParams, p2: &EamParams, rule: MixingRule) -> Vec<f64> {
    let mut radii = Vec::new();
    let mut r: f64 = 0.62;
    while r < 6.2 {
        let near_inner = (r - 0.5).abs() < 2.0e-3;
        let near_cutoff = (r - p1.cutoff).abs() < 2.0e-3 || (r - p2.cutoff).abs() < 2.0e-3;
        let (v1, v2) = (p1.pair_potential(r), p2.pair_potential(r));
        let near_branch_switch = match rule {
            MixingRule::Geometric => v1.abs() < 2.0e-2 || v2.abs() < 2.0e-2,
            _ => false,
        };
        if !(near_inner || near_cutoff || near_branch_switch) {
            radii.push(r);
        }
        r += 0.0173;
    }
    radii
}

#[test]
fn analytic_cross_pair_force_matches_the_central_difference_of_the_potential() {
    let species = [
        EamParams::copper(),
        EamParams::nickel(),
        EamParams::aluminum(),
        EamParams::iron(),
    ];
    let h = 1.0e-5;
    let mut checked = 0usize;
    for rule in [
        MixingRule::Arithmetic,
        MixingRule::Geometric,
        MixingRule::Johnson,
    ] {
        for p1 in &species {
            for p2 in &species {
                for r in smooth_radii(p1, p2, rule) {
                    let numeric = (cross_pair_potential(p1, p2, r + h, rule)
                        - cross_pair_potential(p1, p2, r - h, rule))
                        / (2.0 * h);
                    let analytic = cross_pair_force(p1, p2, r, rule);
                    assert!(
                        (analytic - numeric).abs() <= 1.0e-6 * (1.0 + numeric.abs()),
                        "{rule:?} {}-{} at r = {r}: analytic {analytic} vs numeric {numeric}",
                        p1.label,
                        p2.label
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 3000, "only {checked} points were compared");
}

#[test]
fn cross_pair_force_is_symmetric_and_reduces_to_the_pair_force_for_like_species() {
    let cu = EamParams::copper();
    let ni = EamParams::nickel();
    for rule in [
        MixingRule::Arithmetic,
        MixingRule::Geometric,
        MixingRule::Johnson,
    ] {
        for r in [1.5, 2.2, 2.55, 3.1, 4.4, 5.2] {
            let ab = cross_pair_force(&cu, &ni, r, rule);
            let ba = cross_pair_force(&ni, &cu, r, rule);
            assert!(
                (ab - ba).abs() < 1.0e-12 * (1.0 + ab.abs()),
                "{rule:?} r = {r}"
            );
            let like = cross_pair_force(&cu, &cu, r, rule);
            assert!(
                (like - cu.pair_force(r)).abs() < 1.0e-10 * (1.0 + like.abs()),
                "{rule:?} r = {r}: {like} vs {}",
                cu.pair_force(r)
            );
        }
    }
}

#[test]
fn geometric_cross_force_is_zero_not_nan_where_one_species_has_no_interaction() {
    // Between the nickel cutoff (5.3 Å) and the copper cutoff (5.5 Å) the nickel
    // pair potential is identically zero: the geometric mean is zero with zero
    // slope (a 0/0 in the product-rule formula).
    let cu = EamParams::copper();
    let ni = EamParams::nickel();
    for r in [5.31, 5.4, 5.49] {
        let force = cross_pair_force(&cu, &ni, r, MixingRule::Geometric);
        assert_eq!(force, 0.0, "r = {r}");
        assert_eq!(
            cross_pair_potential(&cu, &ni, r, MixingRule::Geometric),
            0.0
        );
        // The arithmetic mean still feels the copper tail.
        let arithmetic = cross_pair_force(&cu, &ni, r, MixingRule::Arithmetic);
        assert!((arithmetic - 0.5 * cu.pair_force(r)).abs() < 1.0e-15);
        // Johnson's rule falls back to the arithmetic mean without nickel density.
        let johnson = cross_pair_force(&cu, &ni, r, MixingRule::Johnson);
        assert!((johnson - 0.5 * cu.pair_force(r)).abs() < 1.0e-15);
    }
    // Beyond both cutoffs and inside the inner cut everything vanishes.
    for rule in [
        MixingRule::Arithmetic,
        MixingRule::Geometric,
        MixingRule::Johnson,
    ] {
        assert_eq!(cross_pair_force(&cu, &ni, 5.6, rule), 0.0, "{rule:?}");
        assert_eq!(cross_pair_force(&cu, &ni, 0.3, rule), 0.0, "{rule:?}");
    }
    // An isolated zero crossing has no finite slope; the arithmetic slope is used.
    let cross_over = {
        let (mut lo, mut hi) = (1.8, 2.2);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if cu.pair_potential(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let slope = cross_pair_force(&cu, &ni, cross_over, MixingRule::Geometric);
    assert!(slope.is_finite(), "slope at the zero crossing: {slope}");
}
