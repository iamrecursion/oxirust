//! Test helpers: simplicial complexes as [`CwComplex`] values and the chain maps
//! induced by simplicial maps.

use super::{CwCell, CwComplex};
use std::collections::{BTreeSet, HashMap};

/// Oriented simplices per dimension: `simplices[k][id]` lists the vertices of
/// the `k`-cell with that id, in increasing order.  Every face of a simplex must
/// be present among the simplices of the dimension below.
pub(super) type Simplices = Vec<Vec<Vec<usize>>>;

/// Index of every simplex (by its vertex list) within its dimension.
fn simplex_index(simplices: &Simplices) -> Vec<HashMap<Vec<usize>, usize>> {
    simplices
        .iter()
        .map(|cells| {
            cells
                .iter()
                .enumerate()
                .map(|(id, vertices)| (vertices.clone(), id))
                .collect()
        })
        .collect()
}

/// The cell complex whose `k`-cells are the given simplices (cell id = position
/// in the list), with the simplicial boundary `∂[v_0, ..., v_k] =
/// Σ_i (-1)^i [v_0, ..., v̂_i, ..., v_k]`.
pub(super) fn complex_from_simplices(simplices: &Simplices) -> CwComplex {
    let index = simplex_index(simplices);
    let mut complex = CwComplex::new();
    for (dim, cells) in simplices.iter().enumerate() {
        for (id, vertices) in cells.iter().enumerate() {
            if dim == 0 {
                complex.add_cell(CwCell::vertex(id, [vertices[0] as f64, 0.0, 0.0]));
                continue;
            }
            let faces: Vec<usize> = (0..vertices.len())
                .map(|omitted| {
                    let mut face = vertices.clone();
                    face.remove(omitted);
                    index[dim - 1][&face]
                })
                .collect();
            let signs: Vec<i32> = (0..vertices.len())
                .map(|omitted| if omitted % 2 == 0 { 1 } else { -1 })
                .collect();
            complex.add_cell(CwCell::n_cell(id, dim, faces, signs));
        }
    }
    complex
}

/// Parity of the permutation that sorts `values` (all distinct): `1` for an
/// even and `-1` for an odd number of inversions.
fn permutation_sign(values: &[usize]) -> i64 {
    let inversions = values
        .iter()
        .enumerate()
        .map(|(i, a)| values[i + 1..].iter().filter(|b| a > b).count())
        .sum::<usize>();
    if inversions % 2 == 0 { 1 } else { -1 }
}

/// The chain map `f_k : C_k → C_k` of the simplicial map that sends vertex `v`
/// to `vertex_map(v)`, for a map of the complex to itself.
///
/// A simplex goes to the oriented image simplex (sign `-1` when the image
/// vertices come out in odd order) and to zero when two of its vertices
/// collapse.
pub(super) fn simplicial_chain_map(
    simplices: &Simplices,
    vertex_map: &dyn Fn(usize) -> usize,
) -> Vec<Vec<Vec<i64>>> {
    let index = simplex_index(simplices);
    simplices
        .iter()
        .enumerate()
        .map(|(dim, cells)| {
            let mut matrix = vec![vec![0i64; cells.len()]; cells.len()];
            for (column, vertices) in cells.iter().enumerate() {
                let image: Vec<usize> = vertices.iter().map(|&v| vertex_map(v)).collect();
                let mut sorted = image.clone();
                sorted.sort_unstable();
                if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
                    continue;
                }
                matrix[index[dim][&sorted]][column] = permutation_sign(&image);
            }
            matrix
        })
        .collect()
}

/// The boundary of the tetrahedron on the vertices `0..4`: a triangulated
/// 2-sphere with 4 vertices, 6 edges and 4 triangles.
pub(super) fn tetrahedron_surface() -> Simplices {
    vec![
        (0..4).map(|v| vec![v]).collect(),
        vec![
            vec![0, 1],
            vec![0, 2],
            vec![0, 3],
            vec![1, 2],
            vec![1, 3],
            vec![2, 3],
        ],
        vec![vec![0, 1, 2], vec![0, 1, 3], vec![0, 2, 3], vec![1, 2, 3]],
    ]
}

/// A triangulated torus: the `n x n` grid with wrap-around, every square cut
/// along its `(1, 1)` diagonal.  Vertex `(i, j)` has number `i * n + j`; needs
/// `n >= 3` to be a simplicial complex.
pub(super) fn torus_surface(n: usize) -> Simplices {
    let vertex = |i: usize, j: usize| (i % n) * n + (j % n);
    let mut triangles: Vec<Vec<usize>> = Vec::with_capacity(2 * n * n);
    for i in 0..n {
        for j in 0..n {
            for corners in [
                [vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1)],
                [vertex(i, j), vertex(i + 1, j + 1), vertex(i, j + 1)],
            ] {
                let mut triangle = corners.to_vec();
                triangle.sort_unstable();
                triangles.push(triangle);
            }
        }
    }
    let mut edges: BTreeSet<Vec<usize>> = BTreeSet::new();
    for triangle in &triangles {
        edges.insert(vec![triangle[0], triangle[1]]);
        edges.insert(vec![triangle[0], triangle[2]]);
        edges.insert(vec![triangle[1], triangle[2]]);
    }
    vec![
        (0..n * n).map(|v| vec![v]).collect(),
        edges.into_iter().collect(),
        triangles,
    ]
}
