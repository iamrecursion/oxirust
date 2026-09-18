//! Tests for chain maps and [`CellularApproximation::map_degree`]: the degree is
//! read off the action of the chain map on the fundamental cycle, not guessed
//! from Euler characteristics.

use super::chain_map::fundamental_cycle;
use super::test_support::{
    complex_from_simplices, simplicial_chain_map, tetrahedron_surface, torus_surface,
};
use super::{CellularApproximation, CwCell, CwComplex};

/// The 2-sphere with one vertex and one 2-cell (empty attaching map).
fn sphere_one_cell() -> CwComplex {
    let mut cw = CwComplex::new();
    cw.add_cell(CwCell::vertex(0, [0.0; 3]));
    cw.add_cell(CwCell::face(0, vec![], vec![]));
    cw
}

/// The circle with one vertex and one loop edge.
fn circle_one_cell() -> CwComplex {
    let mut cw = CwComplex::new();
    cw.add_cell(CwCell::vertex(0, [0.0; 3]));
    cw.add_cell(CwCell {
        id: 0,
        dim: 1,
        label: "loop".into(),
        coords: None,
        boundary: vec![0, 0],
        boundary_signs: vec![1, -1],
    });
    cw
}

/// The circle with two vertices and two edges `v0 -> v1` and `v1 -> v0`.
fn circle_two_cells() -> CwComplex {
    let mut cw = CwComplex::new();
    cw.add_cell(CwCell::vertex(0, [0.0; 3]));
    cw.add_cell(CwCell::vertex(1, [1.0, 0.0, 0.0]));
    cw.add_cell(CwCell::edge(0, 0, 1));
    cw.add_cell(CwCell::edge(1, 1, 0));
    cw
}

/// `S^2` with the antipodal-equivariant cell structure: two vertices, two
/// edges forming the equator and two hemispheres with the same boundary.
fn sphere_equivariant() -> CwComplex {
    let mut cw = CwComplex::new();
    cw.add_cell(CwCell::vertex(0, [1.0, 0.0, 0.0]));
    cw.add_cell(CwCell::vertex(1, [-1.0, 0.0, 0.0]));
    cw.add_cell(CwCell::edge(0, 0, 1));
    cw.add_cell(CwCell::edge(1, 1, 0));
    cw.add_cell(CwCell::face(0, vec![0, 1], vec![1, 1]));
    cw.add_cell(CwCell::face(1, vec![0, 1], vec![1, 1]));
    cw
}

fn swap() -> Vec<Vec<i64>> {
    vec![vec![0, 1], vec![1, 0]]
}

fn identity2() -> Vec<Vec<i64>> {
    vec![vec![1, 0], vec![0, 1]]
}

/// A projective plane: one vertex, one loop and a disc glued twice along it.
fn projective_plane() -> CwComplex {
    let mut cw = circle_one_cell();
    cw.add_cell(CwCell::face(0, vec![0, 0], vec![1, 1]));
    cw
}

fn degree_of(approximation: &CellularApproximation) -> i64 {
    approximation
        .map_degree()
        .expect("the chain map has a degree")
}

/// All permutations of `0..n`.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    fn extend(prefix: &mut Vec<usize>, used: &mut Vec<bool>, n: usize, out: &mut Vec<Vec<usize>>) {
        if prefix.len() == n {
            out.push(prefix.clone());
            return;
        }
        for v in 0..n {
            if !used[v] {
                used[v] = true;
                prefix.push(v);
                extend(prefix, used, n, out);
                prefix.pop();
                used[v] = false;
            }
        }
    }
    let mut out = Vec::new();
    extend(&mut Vec::new(), &mut vec![false; n], n, &mut out);
    out
}

fn parity(perm: &[usize]) -> i64 {
    let inversions = (0..perm.len())
        .flat_map(|i| (i + 1..perm.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| perm[i] > perm[j])
        .count();
    if inversions % 2 == 0 { 1 } else { -1 }
}

#[test]
fn identity_of_a_sphere_has_degree_one() {
    let identity = CellularApproximation::identity(sphere_one_cell());
    assert_eq!(degree_of(&identity), 1);
    let identity = CellularApproximation::identity(sphere_equivariant());
    assert_eq!(degree_of(&identity), 1);
}

#[test]
fn degree_follows_the_top_dimensional_chain_map_not_the_euler_characteristic() {
    // Every map of S^2 to itself has the same Euler characteristics at both
    // ends (the former "degree" was always 1 here); the chain map decides.
    for degree in [-3, -1, 0, 1, 2, 5] {
        let chain_map = vec![vec![vec![1]], vec![], vec![vec![degree]]];
        let approximation =
            CellularApproximation::with_chain_map(sphere_one_cell(), sphere_one_cell(), chain_map)
                .expect("a map of the one-cell sphere is a chain map in every degree");
        assert_eq!(degree_of(&approximation), degree);
    }
}

#[test]
fn circle_maps_have_the_winding_number_as_degree() {
    for winding in -2..=3 {
        let chain_map = vec![vec![vec![1]], vec![vec![winding]]];
        let approximation =
            CellularApproximation::with_chain_map(circle_one_cell(), circle_one_cell(), chain_map)
                .expect("a chain map of the circle");
        assert_eq!(degree_of(&approximation), winding);
    }
}

#[test]
fn maps_between_different_cell_structures_of_the_circle() {
    // Two-edge circle onto the one-edge circle: both edges run once around.
    let wrap_twice = CellularApproximation::with_chain_map(
        circle_two_cells(),
        circle_one_cell(),
        vec![vec![vec![1, 1]], vec![vec![1, 1]]],
    )
    .expect("a chain map");
    assert_eq!(degree_of(&wrap_twice), 2);
    // Running the second edge backwards cancels the first.
    let cancel = CellularApproximation::with_chain_map(
        circle_two_cells(),
        circle_one_cell(),
        vec![vec![vec![1, 1]], vec![vec![1, -1]]],
    )
    .expect("a chain map");
    assert_eq!(degree_of(&cancel), 0);
    // The loop onto the two-edge circle, once around.
    let inclusion = CellularApproximation::with_chain_map(
        circle_one_cell(),
        circle_two_cells(),
        vec![vec![vec![1], vec![0]], vec![vec![1], vec![1]]],
    )
    .expect("a chain map");
    assert_eq!(degree_of(&inclusion), 1);
    // Sending the loop along only one of the two edges is not a chain map.
    let broken = CellularApproximation::with_chain_map(
        circle_one_cell(),
        circle_two_cells(),
        vec![vec![vec![1], vec![0]], vec![vec![1], vec![0]]],
    );
    assert!(broken.is_err());
}

#[test]
fn antipodal_map_of_the_two_sphere_has_degree_minus_one() {
    let sphere = sphere_equivariant();
    // The antipodal map exchanges the cells of every dimension.
    let antipodal = CellularApproximation::with_chain_map(
        sphere.clone(),
        sphere.clone(),
        vec![swap(), swap(), swap()],
    )
    .expect("the antipodal map is a chain map");
    assert_eq!(degree_of(&antipodal), -1);
    // The reflection in the equatorial plane fixes the equator and exchanges
    // the hemispheres: also degree -1.
    let reflection = CellularApproximation::with_chain_map(
        sphere.clone(),
        sphere.clone(),
        vec![identity2(), identity2(), swap()],
    )
    .expect("the reflection is a chain map");
    assert_eq!(degree_of(&reflection), -1);
    // A rotation by pi about the axis through the two vertices fixes them, sends
    // each equator arc to the reversed opposite arc and exchanges the
    // hemispheres: an orientation-preserving map, degree +1.
    let minus_swap = vec![vec![0, -1], vec![-1, 0]];
    let rotation = CellularApproximation::with_chain_map(
        sphere.clone(),
        sphere.clone(),
        vec![identity2(), minus_swap.clone(), minus_swap],
    )
    .expect("the rotation is a chain map");
    assert_eq!(degree_of(&rotation), 1);
    // The wrong sign on the 2-cells is not a chain map.
    let wrong = CellularApproximation::with_chain_map(
        sphere.clone(),
        sphere,
        vec![swap(), swap(), vec![vec![0, -1], vec![-1, 0]]],
    );
    assert!(wrong.is_err());
}

#[test]
fn fundamental_cycles_are_primitive_with_a_positive_first_coefficient() {
    assert_eq!(
        fundamental_cycle(&sphere_one_cell(), 2, "test").expect("rank one"),
        vec![1]
    );
    assert_eq!(
        fundamental_cycle(&circle_two_cells(), 1, "test").expect("rank one"),
        vec![1, 1]
    );
    // The hemispheres share their boundary: the cycle is their difference.
    assert_eq!(
        fundamental_cycle(&sphere_equivariant(), 2, "test").expect("rank one"),
        vec![1, -1]
    );
    // Boundary of the tetrahedron: ∂[0123] = [123] - [023] + [013] - [012].
    let surface = complex_from_simplices(&tetrahedron_surface());
    assert_eq!(
        fundamental_cycle(&surface, 2, "test").expect("rank one"),
        vec![1, -1, 1, -1]
    );
    // Two discs glued 2 and 3 times around a loop: 2x + 3y = 0 has the
    // primitive solution (3, -2), found by scaling with the denominators.
    for (first, second) in [(2usize, 3usize), (4, 6)] {
        let mut cw = circle_one_cell();
        cw.add_cell(CwCell::face(0, vec![0; first], vec![1; first]));
        cw.add_cell(CwCell::face(1, vec![0; second], vec![1; second]));
        let cycle = fundamental_cycle(&cw, 2, "test").expect("rank one");
        let (a, b) = (first as i64, second as i64);
        let gcd = (1..=a.min(b))
            .rev()
            .find(|d| a % d == 0 && b % d == 0)
            .unwrap_or(1);
        assert_eq!(
            cycle,
            vec![b / gcd, -a / gcd],
            "discs glued {first}/{second} times"
        );
    }
    // A zero-dimensional complex with a single point.
    let mut point = CwComplex::new();
    point.add_cell(CwCell::vertex(0, [0.0; 3]));
    assert_eq!(
        fundamental_cycle(&point, 0, "test").expect("rank one"),
        vec![1]
    );
}

#[test]
fn degree_of_a_symmetry_of_the_tetrahedron_surface_is_the_sign_of_its_permutation() {
    let simplices = tetrahedron_surface();
    let sphere = complex_from_simplices(&simplices);
    let all = permutations(4);
    assert_eq!(all.len(), 24);
    for perm in &all {
        let chain_map = simplicial_chain_map(&simplices, &|v| perm[v]);
        let approximation =
            CellularApproximation::with_chain_map(sphere.clone(), sphere.clone(), chain_map)
                .expect("a simplicial map induces a chain map");
        assert_eq!(
            degree_of(&approximation),
            parity(perm),
            "permutation {perm:?}"
        );
    }
}

#[test]
fn degree_of_symmetries_of_a_triangulated_torus() {
    let n = 5;
    let simplices = torus_surface(n);
    let torus = complex_from_simplices(&simplices);
    let maps: [(&str, Box<dyn Fn(usize) -> usize>, i64); 4] = [
        ("identity", Box::new(|v| v), 1),
        // Translation by one step along the first axis.
        (
            "translation",
            Box::new(move |v| ((v / n + 1) % n) * n + v % n),
            1,
        ),
        // Rotation by pi: (x, y) -> (-x, -y).
        (
            "point reflection",
            Box::new(move |v| ((n - v / n) % n) * n + (n - v % n) % n),
            1,
        ),
        // Exchange of the two axes reverses the orientation.
        ("axis swap", Box::new(move |v| (v % n) * n + v / n), -1),
    ];
    for (name, vertex_map, want) in &maps {
        let chain_map = simplicial_chain_map(&simplices, vertex_map.as_ref());
        let approximation =
            CellularApproximation::with_chain_map(torus.clone(), torus.clone(), chain_map)
                .expect("an automorphism induces a chain map");
        assert_eq!(degree_of(&approximation), *want, "{name}");
    }
    // The torus has one fundamental class: its cycle covers all 50 triangles
    // with coefficients +-1.
    let cycle = fundamental_cycle(&torus, 2, "torus").expect("rank one");
    assert_eq!(cycle.len(), 2 * n * n);
    assert!(cycle.iter().all(|&c| c == 1 || c == -1));
}

#[test]
fn map_degree_reports_why_no_degree_exists() {
    // No map attached.
    let bare = CellularApproximation::new(sphere_one_cell(), sphere_one_cell());
    let message = bare.map_degree().expect_err("no chain map").to_string();
    assert!(message.contains("no chain map"), "{message}");
    assert!(bare.validate_chain_map().is_err());

    // A solid tetrahedron has no top homology.
    let ball = CellularApproximation::identity(CwComplex::standard_tetrahedron());
    let message = ball
        .map_degree()
        .expect_err("a ball has no degree")
        .to_string();
    assert!(
        message.contains("H_3") && message.contains("rank 0"),
        "{message}"
    );

    // The model with two 2-cells on one vertex is a wedge of two spheres.
    let wedge = CellularApproximation::identity(CwComplex::standard_sphere_s2());
    let message = wedge
        .map_degree()
        .expect_err("a wedge has no degree")
        .to_string();
    assert!(
        message.contains("H_2") && message.contains("rank 2"),
        "{message}"
    );

    // The projective plane is not orientable: H_2 = 0.
    let plane = CellularApproximation::identity(projective_plane());
    let message = plane
        .map_degree()
        .expect_err("RP^2 has no degree")
        .to_string();
    assert!(
        message.contains("H_2") && message.contains("rank 0"),
        "{message}"
    );

    // Different top dimensions.
    let mismatch = CellularApproximation::with_chain_map(
        sphere_one_cell(),
        circle_one_cell(),
        vec![vec![vec![1]], vec![vec![]], vec![]],
    );
    let approximation = mismatch.expect("shapes and chain-map condition hold");
    assert!(approximation.map_degree().is_err());
}

#[test]
fn chain_maps_with_wrong_shapes_or_without_the_chain_condition_are_rejected() {
    let sphere = sphere_equivariant();
    // Too few matrices.
    assert!(
        CellularApproximation::with_chain_map(sphere.clone(), sphere.clone(), vec![swap()])
            .is_err()
    );
    // Wrong number of rows in f_0.
    assert!(
        CellularApproximation::with_chain_map(
            sphere.clone(),
            sphere.clone(),
            vec![vec![vec![1, 0]], swap(), swap()],
        )
        .is_err()
    );
    // Wrong number of columns in f_1.
    assert!(
        CellularApproximation::with_chain_map(
            sphere.clone(),
            sphere.clone(),
            vec![swap(), vec![vec![0, 1, 0], vec![1, 0, 0]], swap()],
        )
        .is_err()
    );
    // f_0 swaps the vertices but f_1 fixes the edges: f_0 ∂_1 != ∂_1 f_1.
    assert!(
        CellularApproximation::with_chain_map(
            sphere.clone(),
            sphere.clone(),
            vec![swap(), identity2(), identity2()],
        )
        .is_err()
    );
    // The same defect on an approximation assembled by hand is caught when the
    // degree is requested.
    let by_hand = CellularApproximation {
        source: sphere.clone(),
        target: sphere,
        chain_map: vec![swap(), identity2(), identity2()],
    };
    assert!(by_hand.validate_chain_map().is_err());
    assert!(by_hand.map_degree().is_err());
}

#[test]
fn queries_that_ignore_the_map_still_work() {
    let sphere = CellularApproximation::new(sphere_one_cell(), sphere_one_cell());
    assert!(sphere.is_cellular_by_dim());
    assert!(sphere.homotopy_equivalent_homology());
    assert!(sphere.whitehead_equivalent());
    let ball = CwComplex::standard_tetrahedron();
    let shrink = CellularApproximation::new(ball, circle_one_cell());
    assert!(!shrink.is_cellular_by_dim());
    assert!(!shrink.homotopy_equivalent_homology());
}
