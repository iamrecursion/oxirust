//! Regression tests for the adaptive-FEM kernels: conforming longest-edge
//! bisection of triangle meshes and the iteration estimate of the adaptive driver.

use super::*;

// ---- conforming longest-edge bisection ----------------------------------

/// Twice the signed area of the triangle `tri`.
fn signed_area2(nodes: &[[f64; 3]], tri: &[usize; 3]) -> f64 {
    let a = nodes[tri[0]];
    let b = nodes[tri[1]];
    let c = nodes[tri[2]];
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn total_area(nodes: &[[f64; 3]], elements: &[[usize; 3]]) -> f64 {
    elements
        .iter()
        .map(|tri| 0.5 * signed_area2(nodes, tri))
        .sum()
}

/// The first node that lies strictly inside an edge of some triangle (a hanging
/// node), as `(node, triangle)`.
fn find_hanging_node(nodes: &[[f64; 3]], elements: &[[usize; 3]]) -> Option<(usize, usize)> {
    for (t, tri) in elements.iter().enumerate() {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let p = nodes[tri[i]];
            let q = nodes[tri[j]];
            let pq = [q[0] - p[0], q[1] - p[1]];
            let length_squared = pq[0] * pq[0] + pq[1] * pq[1];
            for (r, node) in nodes.iter().enumerate() {
                if r == tri[i] || r == tri[j] {
                    continue;
                }
                let pr = [node[0] - p[0], node[1] - p[1]];
                let cross = pq[0] * pr[1] - pq[1] * pr[0];
                let along = (pq[0] * pr[0] + pq[1] * pr[1]) / length_squared;
                if cross.abs() <= 1e-12 * length_squared.sqrt()
                    && along > 1e-9
                    && along < 1.0 - 1e-9
                {
                    return Some((r, t));
                }
            }
        }
    }
    None
}

/// Edges that belong to more than two triangles.
fn overloaded_edges(elements: &[[usize; 3]]) -> usize {
    let mut counts: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for tri in elements {
        for (i, j) in [(0, 1), (1, 2), (2, 0)] {
            let key = (tri[i].min(tri[j]), tri[i].max(tri[j]));
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts.values().filter(|&&count| count > 2).count()
}

/// `true` when a triangle with the vertex set of `tri` is among `elements`.
fn contains_triangle(elements: &[[usize; 3]], tri: [usize; 3]) -> bool {
    let mut wanted = tri;
    wanted.sort_unstable();
    elements.iter().any(|candidate| {
        let mut sorted = *candidate;
        sorted.sort_unstable();
        sorted == wanted
    })
}

/// `n x n` unit squares, each split into two right triangles along the same
/// diagonal; all triangles counter-clockwise.  The interior nodes are displaced
/// pseudo-randomly by up to `amplitude` per coordinate, which makes the longest
/// edges of neighbouring triangles differ; the boundary stays fixed.
fn right_triangle_grid(n: usize, amplitude: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let stride = n + 1;
    let mut state = 4242_u64;
    let mut noise = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1_u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut nodes = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let interior = i > 0 && i < n && j > 0 && j < n;
            let (dx, dy) = if interior {
                (amplitude * noise(), amplitude * noise())
            } else {
                (0.0, 0.0)
            };
            nodes.push([i as f64 + dx, j as f64 + dy, 0.0]);
        }
    }
    let mut elements = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let bl = j * stride + i;
            let br = bl + 1;
            let tl = bl + stride;
            let tr = tl + 1;
            elements.push([bl, br, tr]);
            elements.push([bl, tr, tl]);
        }
    }
    (nodes, elements)
}

#[test]
fn test_conformity_checker_detects_a_hanging_node() {
    // Bisect only one of two neighbouring triangles by hand: node 4 hangs on the
    // edge of the unsplit neighbour.  The checker must see that.
    let nodes: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let non_conforming = [[0, 1, 4], [4, 1, 3], [0, 4, 2], [1, 3, 2]];
    assert_eq!(find_hanging_node(&nodes, &non_conforming), Some((4, 3)));
    let conforming = [[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]];
    assert_eq!(find_hanging_node(&nodes, &conforming), None);
}

#[test]
fn test_h_refine_two_triangles_with_a_common_longest_edge() {
    // The shared edge (0, 1) is the longest edge of both triangles: one midpoint,
    // both triangles are bisected through it, four triangles in total.
    let nodes: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.8, 0.0],
        [1.0, -0.8, 0.0],
    ];
    let elements: Vec<[usize; 3]> = vec![[0, 1, 2], [0, 3, 1]];
    let degrees = vec![1, 1];
    let result = h_refine_longest_edge(&nodes, &elements, &degrees, &[0]);
    assert_eq!(result.elements.len(), 4);
    assert_eq!(result.nodes.len(), 5);
    assert_eq!(result.nodes[4], [1.0, 0.0, 0.0]);
    assert_eq!(result.n_refined, 2);
    assert_eq!(result.degrees, vec![1, 1, 1, 1]);
    assert_eq!(find_hanging_node(&result.nodes, &result.elements), None);
    assert_eq!(overloaded_edges(&result.elements), 0);
    assert!((total_area(&result.nodes, &result.elements) - 1.6).abs() < 1e-14);
}

#[test]
fn test_h_refine_marking_one_triangle_closes_the_mesh_around_it() {
    // Two triangles share the edge (1, 2).  It is the longest edge of the marked
    // triangle 0 but not of its neighbour, whose longest edge (2, 3) is
    // bisected first; the neighbour then also needs the shared midpoint.  Bisecting
    // only triangle 0 (the old behaviour) leaves a hanging node.
    let nodes: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [3.0, 3.0, 0.0],
    ];
    let elements: Vec<[usize; 3]> = vec![[0, 1, 2], [1, 3, 2]];
    let degrees = vec![1, 2];
    let result = h_refine_longest_edge(&nodes, &elements, &degrees, &[0]);
    assert!(
        result.elements.len() >= 4,
        "{} triangles",
        result.elements.len()
    );
    assert_eq!(result.elements.len(), 5);
    assert_eq!(result.nodes.len(), 6);
    assert_eq!(result.n_refined, 3);
    assert_eq!(find_hanging_node(&result.nodes, &result.elements), None);
    assert_eq!(overloaded_edges(&result.elements), 0);
    // Areas: 1.0 + 3.5.
    assert!((total_area(&result.nodes, &result.elements) - 4.5).abs() < 1e-13);
    // Both marked and propagated triangles are gone, and every child keeps the
    // counter-clockwise orientation of its parent.
    assert!(!contains_triangle(&result.elements, [0, 1, 2]));
    assert!(!contains_triangle(&result.elements, [1, 3, 2]));
    assert!(
        result
            .elements
            .iter()
            .all(|tri| signed_area2(&result.nodes, tri) > 0.0)
    );
    // The children inherit the degree of their parent: area-weighted degrees add
    // up to 1·1.0 + 2·3.5 = 8.
    let weighted: f64 = result
        .elements
        .iter()
        .zip(&result.degrees)
        .map(|(tri, &degree)| degree as f64 * 0.5 * signed_area2(&result.nodes, tri))
        .sum();
    assert!(
        (weighted - 8.0).abs() < 1e-13,
        "weighted degree sum {weighted}"
    );
}

#[test]
fn test_h_refine_leaves_unrelated_triangles_untouched() {
    // A strip of three triangles; only the first is marked and its refinement
    // does not reach the third.
    let nodes: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
    ];
    let elements: Vec<[usize; 3]> = vec![[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]];
    let degrees = vec![1, 1, 1, 1];
    let result = h_refine_longest_edge(&nodes, &elements, &degrees, &[0]);
    assert_eq!(find_hanging_node(&result.nodes, &result.elements), None);
    // The last triangle sits at its old index with its old connectivity.
    assert_eq!(result.elements[3], [4, 5, 3]);
    assert_eq!(result.degrees.len(), result.elements.len());
    assert!((total_area(&result.nodes, &result.elements) - 2.0).abs() < 1e-14);
}

#[test]
fn test_h_refine_random_marks_on_a_grid_stay_conforming() {
    let (nodes, elements) = right_triangle_grid(6, 0.15);
    let degrees: Vec<usize> = (0..elements.len()).map(|e| 1 + e % 3).collect();
    let weighted_before: f64 = elements
        .iter()
        .zip(&degrees)
        .map(|(tri, &degree)| degree as f64 * 0.5 * signed_area2(&nodes, tri))
        .sum();
    let mut state = 12345_u64;
    let mut marked = Vec::new();
    for e in 0..elements.len() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        if (state >> 33).is_multiple_of(4) {
            marked.push(e);
        }
    }
    assert!(!marked.is_empty());
    let result = h_refine_longest_edge(&nodes, &elements, &degrees, &marked);
    assert_eq!(find_hanging_node(&result.nodes, &result.elements), None);
    assert_eq!(overloaded_edges(&result.elements), 0);
    assert!((total_area(&result.nodes, &result.elements) - 36.0).abs() < 1e-10);
    assert!(
        result
            .elements
            .iter()
            .all(|tri| signed_area2(&result.nodes, tri) > 0.0)
    );
    // Every marked triangle was bisected...
    for &e in &marked {
        assert!(
            !contains_triangle(&result.elements, elements[e]),
            "marked triangle {e} survived"
        );
    }
    // ...at least one per marked triangle was added, and the bookkeeping adds up.
    assert!(result.n_refined >= marked.len());
    assert_eq!(result.elements.len(), elements.len() + result.n_refined);
    assert_eq!(result.degrees.len(), result.elements.len());
    let weighted_after: f64 = result
        .elements
        .iter()
        .zip(&result.degrees)
        .map(|(tri, &degree)| degree as f64 * 0.5 * signed_area2(&result.nodes, tri))
        .sum();
    assert!((weighted_after - weighted_before).abs() < 1e-9);
    // One new node per bisected edge: no two nodes coincide.
    for (i, a) in result.nodes.iter().enumerate() {
        for b in result.nodes.iter().skip(i + 1) {
            assert!(a != b, "duplicate node at {a:?}");
        }
    }
}

#[test]
fn test_h_refine_repeated_passes_stay_conforming() {
    // Feed the refined mesh back into the next pass: nodes, connectivity and degrees
    // of one pass are the input of the next one.
    let (mut nodes, mut elements) = right_triangle_grid(5, 0.15);
    let mut degrees: Vec<usize> = (0..elements.len()).map(|e| 1 + e % 2).collect();
    let mut state = 987_654_321_u64;
    for pass in 0..5 {
        let mut marked = Vec::new();
        for e in 0..elements.len() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if (state >> 33).is_multiple_of(5) {
                marked.push(e);
            }
        }
        let before = elements.clone();
        let result = h_refine_longest_edge(&nodes, &elements, &degrees, &marked);
        assert_eq!(
            find_hanging_node(&result.nodes, &result.elements),
            None,
            "pass {pass}"
        );
        assert_eq!(overloaded_edges(&result.elements), 0, "pass {pass}");
        assert!((total_area(&result.nodes, &result.elements) - 25.0).abs() < 1e-9);
        for &e in &marked {
            assert!(
                !contains_triangle(&result.elements, before[e]),
                "pass {pass}: marked triangle {e} survived"
            );
        }
        assert_eq!(result.degrees.len(), result.elements.len());
        nodes = result.nodes;
        elements = result.elements;
        degrees = result.degrees;
    }
    assert!(elements.len() > 50);
}

#[test]
fn test_h_refine_marking_everything_refines_every_triangle() {
    let (nodes, elements) = right_triangle_grid(4, 0.0);
    let degrees = vec![1; elements.len()];
    let marked: Vec<usize> = (0..elements.len()).collect();
    let result = h_refine_longest_edge(&nodes, &elements, &degrees, &marked);
    assert_eq!(find_hanging_node(&result.nodes, &result.elements), None);
    assert!((total_area(&result.nodes, &result.elements) - 16.0).abs() < 1e-12);
    // The hypotenuse of every right triangle is its longest edge and is shared
    // with its mirror image: each pair is bisected together, so the mesh doubles
    // exactly and there is one new node per hypotenuse.
    assert_eq!(result.elements.len(), 2 * elements.len());
    assert_eq!(result.nodes.len(), nodes.len() + elements.len() / 2);
    assert_eq!(result.n_refined, elements.len());
}

#[test]
fn test_h_refine_ignores_duplicate_and_invalid_marks() {
    let nodes: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let elements: Vec<[usize; 3]> = vec![[0, 1, 2]];
    let degrees = vec![3];
    let once = h_refine_longest_edge(&nodes, &elements, &degrees, &[0]);
    let messy = h_refine_longest_edge(&nodes, &elements, &degrees, &[7, 0, 0, usize::MAX, 0]);
    assert_eq!(once.elements, messy.elements);
    assert_eq!(once.nodes, messy.nodes);
    assert_eq!(messy.n_refined, 1);
    assert_eq!(messy.degrees, vec![3, 3]);
    // Only invalid marks: nothing happens.
    let none = h_refine_longest_edge(&nodes, &elements, &degrees, &[5]);
    assert_eq!(none.elements, elements);
    assert_eq!(none.n_refined, 0);
}

#[test]
fn test_h_refine_defaults_missing_degrees_to_one() {
    let nodes: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let elements: Vec<[usize; 3]> = vec![[0, 1, 3], [0, 3, 2]];
    let result = h_refine_longest_edge(&nodes, &elements, &[4], &[1]);
    // Triangle 0 (degree 4) shares the hypotenuse with triangle 1 (no entry: 1).
    assert_eq!(result.elements.len(), 4);
    let mut degrees = result.degrees.clone();
    degrees.sort_unstable();
    assert_eq!(degrees, vec![1, 1, 4, 4]);
}

#[test]
fn test_h_refine_leaves_degenerate_marked_triangles_unchanged() {
    // Coincident nodes (zero-length edge), a missing node and a NaN coordinate
    // cannot be bisected; they must neither panic nor loop.
    let nodes: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [f64::NAN, 1.0, 0.0],
    ];
    let elements: Vec<[usize; 3]> = vec![[0, 1, 2], [0, 2, 9], [0, 2, 3]];
    let degrees = vec![1, 1, 1];
    let result = h_refine_longest_edge(&nodes, &elements, &degrees, &[0, 1, 2]);
    assert_eq!(result.elements, elements);
    assert_eq!(result.nodes.len(), nodes.len());
    assert_eq!(result.n_refined, 0);
}

#[test]
fn test_remaining_iters_follows_the_halving_model() {
    let driver = AdaptiveFemDriver::new(1.0e-3, 10, 0.5, 0.1, false);
    // 8x the tolerance: three halvings.
    assert_eq!(driver.estimated_remaining_iters(8.0e-3), 3);
    // 1.5x the tolerance: one halving is enough.
    assert_eq!(driver.estimated_remaining_iters(1.5e-3), 1);
    // Just above the tolerance still needs one more iteration.
    assert_eq!(driver.estimated_remaining_iters(1.000_001e-3), 1);
    // 1024x the tolerance: exactly ten halvings, the whole budget.
    assert_eq!(driver.estimated_remaining_iters(1.024), 10);
}

#[test]
fn test_remaining_iters_is_zero_once_converged() {
    let driver = AdaptiveFemDriver::new(1.0e-3, 10, 0.5, 0.1, false);
    assert_eq!(driver.estimated_remaining_iters(1.0e-3), 0);
    assert_eq!(driver.estimated_remaining_iters(5.0e-4), 0);
    assert_eq!(driver.estimated_remaining_iters(0.0), 0);
}

#[test]
fn test_remaining_iters_is_capped_by_the_iteration_budget() {
    let mut driver = AdaptiveFemDriver::new(1.0e-3, 10, 0.5, 0.1, false);
    // 2^1000 times the tolerance: the cast of the huge halving count saturates
    // and must not overflow; the answer is the budget that is left.
    assert_eq!(
        driver.estimated_remaining_iters(1.0e-3 * 2.0_f64.powi(1000)),
        10
    );
    assert_eq!(driver.estimated_remaining_iters(1.0e300), 10);
    for k in 0..8 {
        driver.record_iteration(100 * (k + 1), 1.0e-2);
    }
    // Eight of ten iterations are used: two remain, although 8x needs three.
    assert_eq!(driver.estimated_remaining_iters(8.0e-3), 2);
    driver.record_iteration(900, 1.0e-2);
    driver.record_iteration(1000, 1.0e-2);
    assert_eq!(driver.estimated_remaining_iters(8.0e-3), 0);
    // An iteration counter beyond the maximum does not underflow.
    driver.record_iteration(1100, 1.0e-2);
    assert_eq!(driver.estimated_remaining_iters(8.0e-3), 0);
}

#[test]
fn test_remaining_iters_handles_unusable_numbers_without_panicking() {
    let driver = AdaptiveFemDriver::new(1.0e-3, 10, 0.5, 0.1, false);
    assert_eq!(driver.estimated_remaining_iters(f64::NAN), 10);
    assert_eq!(driver.estimated_remaining_iters(f64::INFINITY), 10);
    assert_eq!(driver.estimated_remaining_iters(f64::NEG_INFINITY), 10);
    // A non-positive or non-finite tolerance leaves no usable error ratio.
    for tol in [0.0, -1.0e-3, f64::NAN, f64::INFINITY] {
        let broken = AdaptiveFemDriver::new(tol, 10, 0.5, 0.1, false);
        assert_eq!(broken.estimated_remaining_iters(1.0), 10);
    }
}
