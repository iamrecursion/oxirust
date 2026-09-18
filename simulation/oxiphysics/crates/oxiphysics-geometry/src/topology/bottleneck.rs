//! Exact bottleneck distance between persistence diagrams.
//!
//! A persistence diagram is a finite multiset of points `(birth, death)` above
//! the diagonal; the diagonal itself is available in unlimited supply because a
//! feature of zero persistence is no feature at all. The bottleneck distance
//! of two diagrams `X` and `Y` is
//!
//! ```text
//! d_B(X, Y) = min over bijections eta : X + diag -> Y + diag
//!             of max over x of || x - eta(x) ||_inf ,
//! ```
//!
//! so a point of `X` may be matched with a point of `Y` or with the diagonal,
//! every point is matched exactly once, and the cost of a point is the largest
//! coordinate difference to its partner. The distance to the diagonal is half
//! the persistence.
//!
//! Classes that never die (`death == f64::MAX`) cannot be matched with the
//! diagonal. They are matched among themselves, per homological dimension, at
//! the cost of the birth difference; if the two diagrams have a different
//! number of essential classes in some dimension the distance is infinite.
//!
//! The finite part is computed exactly. Every candidate value of `d_B` is the
//! cost of a point-to-point or point-to-diagonal edge, so the candidates are
//! sorted and the smallest `eps` for which the threshold graph (all edges of
//! cost at most `eps`) has a perfect matching is found by bisection. The
//! matching problem is the usual reduction of Kerber, Morozov and Nigmetov: the
//! left side holds the points of `X` plus one diagonal copy per point of `Y`,
//! the right side holds the points of `Y` plus one diagonal copy per point of
//! `X`; a point may only reach its own diagonal copy and any two diagonal
//! copies are joined at cost zero. A perfect matching is found with
//! Hopcroft-Karp.

use std::collections::VecDeque;

use super::types::BirthDeathPair;

/// Marker for "unmatched" in the matching arrays.
const NONE: usize = usize::MAX;

/// A persistence point `(birth, death)`.
type Point = (f64, f64);

/// `L_inf` distance between two persistence points.
fn linf(p: Point, q: Point) -> f64 {
    (p.0 - q.0).abs().max((p.1 - q.1).abs())
}

/// `L_inf` distance from a persistence point to the diagonal.
fn to_diagonal(p: Point) -> f64 {
    0.5 * (p.1 - p.0).abs()
}

/// Bottleneck distance between two persistence diagrams.
///
/// Pairs of different homological dimensions are never matched with each other,
/// so the result is the largest per-dimension distance. Pairs of zero
/// persistence are ignored. The result is `f64::INFINITY` when the diagrams
/// hold a different number of essential classes in some dimension, and `NaN`
/// when a birth or death value is `NaN`.
pub(crate) fn diagram_bottleneck_distance(a: &[BirthDeathPair], b: &[BirthDeathPair]) -> f64 {
    if a.iter()
        .chain(b)
        .any(|p| p.birth.is_nan() || p.death.is_nan())
    {
        return f64::NAN;
    }
    let mut dims: Vec<usize> = a.iter().chain(b).map(|p| p.dim).collect();
    dims.sort_unstable();
    dims.dedup();
    let mut distance = 0.0_f64;
    for dim in dims {
        let (finite_a, essential_a) = split_dimension(a, dim);
        let (finite_b, essential_b) = split_dimension(b, dim);
        distance = distance
            .max(finite_bottleneck_distance(&finite_a, &finite_b))
            .max(essential_bottleneck_distance(&essential_a, &essential_b));
    }
    distance
}

/// Finite `(birth, death)` points and the births of the essential classes of
/// the pairs of dimension `dim`.
fn split_dimension(pairs: &[BirthDeathPair], dim: usize) -> (Vec<Point>, Vec<f64>) {
    let mut finite = Vec::new();
    let mut essential = Vec::new();
    for pair in pairs.iter().filter(|p| p.dim == dim) {
        if pair.death >= f64::MAX {
            essential.push(pair.birth);
        } else {
            finite.push((pair.birth, pair.death));
        }
    }
    (finite, essential)
}

/// Bottleneck distance between the births of two sets of essential classes.
///
/// On the real line the optimal matching pairs the sorted births in order, so
/// the distance is the largest difference between corresponding entries. It is
/// infinite when the two sets differ in size.
pub(crate) fn essential_bottleneck_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Exact bottleneck distance between two multisets of finite persistence points
/// of one homological dimension. Points on the diagonal are ignored.
pub(crate) fn finite_bottleneck_distance(a: &[Point], b: &[Point]) -> f64 {
    let a: Vec<Point> = a.iter().copied().filter(|p| p.0 != p.1).collect();
    let b: Vec<Point> = b.iter().copied().filter(|p| p.0 != p.1).collect();
    // Sending every point to the diagonal is always a valid bijection, so this
    // bounds the distance from above (and equals it when nothing is closer).
    let upper = a
        .iter()
        .chain(&b)
        .map(|&p| to_diagonal(p))
        .fold(0.0_f64, f64::max);
    if upper == 0.0 {
        return 0.0;
    }
    let mut candidates = vec![0.0];
    candidates.extend(a.iter().chain(&b).map(|&p| to_diagonal(p)));
    for &p in &a {
        for &q in &b {
            let cost = linf(p, q);
            if cost <= upper {
                candidates.push(cost);
            }
        }
    }
    candidates.sort_by(f64::total_cmp);
    candidates.dedup();
    // The largest candidate is `upper`, for which a perfect matching exists.
    let (mut lo, mut hi) = (0, candidates.len() - 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if has_perfect_matching(&threshold_graph(&a, &b, candidates[mid])) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    candidates[lo]
}

/// Adjacency lists (left vertex -> right vertices) of the graph that joins every
/// pair of vertices whose matching cost is at most `eps`.
///
/// Left vertices `0..a.len()` are the points of `a`, left vertex
/// `a.len() + j` is the diagonal copy of `b[j]`. Right vertices `0..b.len()`
/// are the points of `b`, right vertex `b.len() + i` is the diagonal copy of
/// `a[i]`.
fn threshold_graph(a: &[Point], b: &[Point], eps: f64) -> Vec<Vec<usize>> {
    let (na, nb) = (a.len(), b.len());
    let mut adjacency = Vec::with_capacity(na + nb);
    for (i, &p) in a.iter().enumerate() {
        let mut row: Vec<usize> = b
            .iter()
            .enumerate()
            .filter(|&(_, &q)| linf(p, q) <= eps)
            .map(|(j, _)| j)
            .collect();
        if to_diagonal(p) <= eps {
            row.push(nb + i);
        }
        adjacency.push(row);
    }
    for (j, &q) in b.iter().enumerate() {
        let mut row = Vec::with_capacity(na + 1);
        if to_diagonal(q) <= eps {
            row.push(j);
        }
        row.extend(nb..nb + na);
        adjacency.push(row);
    }
    adjacency
}

/// Whether the bipartite graph with `adjacency.len()` vertices on each side has
/// a perfect matching (Hopcroft-Karp).
fn has_perfect_matching(adjacency: &[Vec<usize>]) -> bool {
    let n = adjacency.len();
    if adjacency.iter().any(Vec::is_empty) {
        return false;
    }
    let mut match_left = vec![NONE; n];
    let mut match_right = vec![NONE; n];
    let mut matched = 0;
    // Greedy start: most vertices find a partner immediately.
    for (u, row) in adjacency.iter().enumerate() {
        if let Some(&v) = row.iter().find(|&&v| match_right[v] == NONE) {
            match_left[u] = v;
            match_right[v] = u;
            matched += 1;
        }
    }
    let mut layer = vec![NONE; n];
    while matched < n {
        let Some(limit) = build_layers(adjacency, &match_left, &match_right, &mut layer) else {
            return false;
        };
        let mut cursor = vec![0usize; n];
        let before = matched;
        for root in 0..n {
            if match_left[root] == NONE
                && augment(
                    root,
                    adjacency,
                    limit,
                    &mut layer,
                    &mut cursor,
                    &mut match_left,
                    &mut match_right,
                )
            {
                matched += 1;
            }
        }
        if matched == before {
            // A layered shortest augmenting path always exists here, so this
            // guards against looping forever should the invariant ever break.
            return false;
        }
    }
    true
}

/// Breadth-first layering from the unmatched left vertices along alternating
/// paths. Returns the layer in which an unmatched right vertex is first
/// reached (the length of the shortest augmenting path), or `None` when there
/// is no augmenting path.
fn build_layers(
    adjacency: &[Vec<usize>],
    match_left: &[usize],
    match_right: &[usize],
    layer: &mut [usize],
) -> Option<usize> {
    let mut queue = VecDeque::new();
    for (u, slot) in layer.iter_mut().enumerate() {
        if match_left[u] == NONE {
            *slot = 0;
            queue.push_back(u);
        } else {
            *slot = NONE;
        }
    }
    let mut limit = NONE;
    while let Some(u) = queue.pop_front() {
        if limit != NONE && layer[u] > limit {
            break;
        }
        for &v in &adjacency[u] {
            let w = match_right[v];
            if w == NONE {
                if limit == NONE {
                    limit = layer[u];
                }
            } else if layer[w] == NONE {
                layer[w] = layer[u] + 1;
                queue.push_back(w);
            }
        }
    }
    (limit != NONE).then_some(limit)
}

/// Depth-first search for a shortest augmenting path from the unmatched left
/// vertex `root` inside the layered graph; flips the path when one is found.
fn augment(
    root: usize,
    adjacency: &[Vec<usize>],
    limit: usize,
    layer: &mut [usize],
    cursor: &mut [usize],
    match_left: &mut [usize],
    match_right: &mut [usize],
) -> bool {
    let mut path = vec![root];
    while let Some(&u) = path.last() {
        let Some(&v) = adjacency[u].get(cursor[u]) else {
            // Dead end: no later search may enter this vertex again.
            layer[u] = NONE;
            path.pop();
            continue;
        };
        cursor[u] += 1;
        let w = match_right[v];
        if w == NONE {
            if layer[u] == limit {
                // Every vertex of `path` left through the neighbour just before
                // its cursor; re-pair them along the path.
                for &x in &path {
                    let target = adjacency[x][cursor[x] - 1];
                    match_left[x] = target;
                    match_right[target] = x;
                }
                return true;
            }
        } else if layer[u] < limit && layer[w] == layer[u] + 1 {
            path.push(w);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random numbers in `[0, 1)`.
    struct Lcg(u64);

    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn diagram(&mut self, count: usize) -> Vec<Point> {
            (0..count)
                .map(|_| {
                    let birth = 10.0 * self.next_f64();
                    (birth, birth + 5.0 * self.next_f64() + 0.01)
                })
                .collect()
        }
    }

    fn pair(dim: usize, birth: f64, death: f64) -> BirthDeathPair {
        BirthDeathPair { dim, birth, death }
    }

    /// Smallest `eps` with a perfect matching, found by scanning every
    /// candidate with a simple augmenting-path matcher instead of bisection.
    fn reference_distance(a: &[Point], b: &[Point]) -> f64 {
        let (na, nb) = (a.len(), b.len());
        let mut costs = vec![0.0];
        costs.extend(a.iter().chain(b).map(|&p| to_diagonal(p)));
        for &p in a {
            for &q in b {
                costs.push(linf(p, q));
            }
        }
        costs.sort_by(f64::total_cmp);
        costs.dedup();
        for &eps in &costs {
            let adjacency = threshold_graph(a, b, eps);
            let mut owner = vec![NONE; na + nb];
            let mut all_matched = true;
            for u in 0..na + nb {
                let mut seen = vec![false; na + nb];
                if !kuhn(u, &adjacency, &mut seen, &mut owner) {
                    all_matched = false;
                    break;
                }
            }
            if all_matched {
                return eps;
            }
        }
        f64::INFINITY
    }

    fn kuhn(u: usize, adjacency: &[Vec<usize>], seen: &mut [bool], owner: &mut [usize]) -> bool {
        for &v in &adjacency[u] {
            if seen[v] {
                continue;
            }
            seen[v] = true;
            if owner[v] == NONE || kuhn(owner[v], adjacency, seen, owner) {
                owner[v] = u;
                return true;
            }
        }
        false
    }

    /// Brute force over every partial matching between the two diagrams.
    fn brute_force(a: &[Point], b: &[Point]) -> f64 {
        fn go(i: usize, a: &[Point], b: &[Point], used: &mut [bool], running: f64) -> f64 {
            if i == a.len() {
                return b
                    .iter()
                    .zip(used.iter())
                    .filter(|&(_, &taken)| !taken)
                    .map(|(&q, _)| to_diagonal(q))
                    .fold(running, f64::max);
            }
            let mut best = go(i + 1, a, b, used, running.max(to_diagonal(a[i])));
            for j in 0..b.len() {
                if !used[j] {
                    used[j] = true;
                    best = best.min(go(i + 1, a, b, used, running.max(linf(a[i], b[j]))));
                    used[j] = false;
                }
            }
            best
        }
        go(0, a, b, &mut vec![false; b.len()], 0.0)
    }

    #[test]
    fn empty_diagrams_are_at_distance_zero() {
        assert_eq!(finite_bottleneck_distance(&[], &[]), 0.0);
        assert_eq!(diagram_bottleneck_distance(&[], &[]), 0.0);
    }

    #[test]
    fn a_point_against_nothing_is_matched_with_the_diagonal() {
        let a = [(0.0, 4.0)];
        assert_eq!(finite_bottleneck_distance(&a, &[]), 2.0);
        assert_eq!(finite_bottleneck_distance(&[], &a), 2.0);
        let pd = [pair(0, 0.0, 4.0)];
        assert_eq!(diagram_bottleneck_distance(&pd, &[]), 2.0);
        assert_eq!(diagram_bottleneck_distance(&[], &pd), 2.0);
    }

    #[test]
    fn matching_is_one_to_one() {
        // Both copies of (0, 10) cannot share the single partner, so one of them
        // goes to the diagonal at cost 5.
        let two = [(0.0, 10.0), (0.0, 10.0)];
        let one = [(0.0, 10.0)];
        assert_eq!(finite_bottleneck_distance(&two, &one), 5.0);
        assert_eq!(finite_bottleneck_distance(&one, &two), 5.0);
    }

    #[test]
    fn nearby_points_are_matched_with_each_other() {
        assert_eq!(
            finite_bottleneck_distance(&[(0.0, 4.0)], &[(1.0, 5.0)]),
            1.0
        );
        // Moving by 1 beats the diagonal (cost 2).
        let a = [(0.0, 4.0), (10.0, 20.0)];
        let b = [(1.0, 4.5), (10.0, 19.0)];
        assert_eq!(finite_bottleneck_distance(&a, &b), 1.0);
    }

    #[test]
    fn identical_diagrams_have_distance_zero() {
        let mut rng = Lcg(7);
        let a = rng.diagram(25);
        assert_eq!(finite_bottleneck_distance(&a, &a), 0.0);
    }

    #[test]
    fn points_on_the_diagonal_are_ignored() {
        let a = [(0.0, 4.0), (3.0, 3.0)];
        let b = [(0.0, 4.0), (7.0, 7.0), (1.0, 1.0)];
        assert_eq!(finite_bottleneck_distance(&a, &b), 0.0);
    }

    #[test]
    fn matches_brute_force_on_small_diagrams() {
        let mut rng = Lcg(42);
        for trial in 0..120 {
            let na = trial % 5;
            let nb = (trial / 5) % 5;
            let a = rng.diagram(na);
            let b = rng.diagram(nb);
            let exact = finite_bottleneck_distance(&a, &b);
            let expected = brute_force(&a, &b);
            assert!(
                (exact - expected).abs() < 1e-12,
                "trial {trial}: {exact} != {expected} for {a:?} / {b:?}"
            );
        }
    }

    #[test]
    fn matches_the_linear_scan_on_larger_diagrams() {
        let mut rng = Lcg(2024);
        for (na, nb) in [(12, 12), (20, 9), (7, 25), (30, 30)] {
            let a = rng.diagram(na);
            let b = rng.diagram(nb);
            let exact = finite_bottleneck_distance(&a, &b);
            let expected = reference_distance(&a, &b);
            assert_eq!(exact, expected, "{na} vs {nb} points");
        }
    }

    #[test]
    fn distance_is_symmetric_and_obeys_the_triangle_inequality() {
        let mut rng = Lcg(99);
        for _ in 0..20 {
            let a = rng.diagram(8);
            let b = rng.diagram(11);
            let c = rng.diagram(6);
            let ab = finite_bottleneck_distance(&a, &b);
            let ba = finite_bottleneck_distance(&b, &a);
            assert_eq!(ab, ba);
            let ac = finite_bottleneck_distance(&a, &c);
            let cb = finite_bottleneck_distance(&c, &b);
            assert!(ab <= ac + cb + 1e-12, "{ab} > {ac} + {cb}");
        }
    }

    #[test]
    fn dimensions_are_matched_separately() {
        // Same coordinates in different dimensions are two unrelated features.
        let a = [pair(0, 0.0, 4.0)];
        let b = [pair(1, 0.0, 4.0)];
        assert_eq!(diagram_bottleneck_distance(&a, &b), 2.0);
        // Identical in each dimension: zero.
        let both = [pair(0, 0.0, 4.0), pair(1, 1.0, 3.0)];
        assert_eq!(diagram_bottleneck_distance(&both, &both), 0.0);
        // The distance is the worst dimension.
        let moved = [pair(0, 0.5, 4.0), pair(1, 1.0, 3.0)];
        assert_eq!(diagram_bottleneck_distance(&both, &moved), 0.5);
        let moved_more = [pair(0, 0.5, 4.0), pair(1, 1.0, 3.75)];
        assert_eq!(diagram_bottleneck_distance(&both, &moved_more), 0.75);
    }

    #[test]
    fn essential_classes_are_matched_by_birth() {
        let a = [pair(0, 0.0, f64::MAX), pair(0, 2.0, f64::MAX)];
        let b = [pair(0, 2.5, f64::MAX), pair(0, 0.25, f64::MAX)];
        assert_eq!(diagram_bottleneck_distance(&a, &b), 0.5);
        // An essential class never matches the diagonal or a finite point.
        let finite = [pair(0, 0.0, 10.0)];
        let essential = [pair(0, 0.0, f64::MAX)];
        assert_eq!(
            diagram_bottleneck_distance(&finite, &essential),
            f64::INFINITY
        );
        // Different counts of essential classes: infinite in either direction.
        let two = [pair(1, 0.0, f64::MAX), pair(1, 1.0, f64::MAX)];
        let one = [pair(1, 0.0, f64::MAX)];
        assert_eq!(diagram_bottleneck_distance(&two, &one), f64::INFINITY);
        assert_eq!(diagram_bottleneck_distance(&one, &two), f64::INFINITY);
        // An infinite death counts as essential, too.
        let inf = [pair(0, 1.0, f64::INFINITY)];
        let max = [pair(0, 3.0, f64::MAX)];
        assert_eq!(diagram_bottleneck_distance(&inf, &max), 2.0);
    }

    #[test]
    fn finite_and_essential_parts_combine() {
        let a = [pair(0, 0.0, f64::MAX), pair(0, 1.0, 5.0)];
        let b = [pair(0, 0.25, f64::MAX)];
        // Essential: 0.25. Finite: (1, 5) goes to the diagonal at cost 2.
        assert_eq!(diagram_bottleneck_distance(&a, &b), 2.0);
    }

    #[test]
    fn nan_values_propagate() {
        let a = [pair(0, f64::NAN, 3.0)];
        let b = [pair(0, 0.0, 3.0)];
        assert!(diagram_bottleneck_distance(&a, &b).is_nan());
        assert!(diagram_bottleneck_distance(&b, &a).is_nan());
    }

    #[test]
    fn hopcroft_karp_finds_perfect_matchings() {
        // A path of alternating assignments that needs an augmentation.
        let adjacency = vec![vec![0, 1], vec![0], vec![1, 2]];
        assert!(has_perfect_matching(&adjacency));
        // Two left vertices competing for one right vertex.
        let adjacency = vec![vec![0], vec![0], vec![1, 2]];
        assert!(!has_perfect_matching(&adjacency));
        assert!(has_perfect_matching(&[]));
    }
}
