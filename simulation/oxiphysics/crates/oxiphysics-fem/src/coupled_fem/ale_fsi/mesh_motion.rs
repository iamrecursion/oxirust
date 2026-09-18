//! Harmonic (Laplace) extension of the interface motion into the fluid mesh.
//!
//! The ALE mesh displacement `d` solves ∇²d = 0 on the reference configuration
//! with the prescribed displacement on the mesh boundary (the structural
//! displacement on the interface, zero on all other boundary nodes).  The
//! Galerkin discretisation on the linear triangles gives the stiffness matrix
//! K_ab = Σ_e |A_e| ∇λ_a·∇λ_b; the interior displacement follows from
//! K_ii·d_i = −K_ib·d_b.  K_ii is factorised once per [`HarmonicExtension`], so
//! every further extension (one per coupling iteration) costs two
//! back-substitutions.  The extension reproduces affine displacement fields
//! exactly and obeys the discrete maximum principle on Delaunay-type meshes.

use super::geometry::{Triangle, element_corners};
use crate::dense_lu::LuFactor;
use crate::error::{Error, Result};

/// Largest number of interior nodes handled by the dense mesh-motion solve.
const MAX_INTERIOR_NODES: usize = 4096;

/// Linear operator that maps boundary displacements to interior displacements.
#[derive(Debug, Clone)]
pub(super) struct HarmonicExtension {
    /// Node indices of the interior nodes.
    interior: Vec<usize>,
    /// For each interior node the couplings K_ib to boundary nodes
    /// (boundary node index, stiffness entry).
    coupling: Vec<Vec<(usize, f64)>>,
    /// LU factors of the interior block K_ii (`None` without interior nodes).
    lu: Option<LuFactor>,
}

impl HarmonicExtension {
    /// Build the operator on the reference mesh.  `on_boundary[node]` marks the
    /// nodes whose displacement is prescribed.
    ///
    /// # Errors
    /// [`Error::InvalidInput`] for inconsistent sizes, a degenerate element or
    /// too many interior nodes for the dense solve; [`Error::SingularSystem`]
    /// if an interior node is not connected to the boundary through elements.
    pub(super) fn new(
        reference: &[[f64; 2]],
        elements: &[[usize; 3]],
        on_boundary: &[bool],
    ) -> Result<Self> {
        let n = reference.len();
        if on_boundary.len() != n {
            return Err(Error::InvalidInput(format!(
                "mesh motion needs one boundary flag per node, got {} for {n} nodes",
                on_boundary.len()
            )));
        }
        let mut slot: Vec<Option<usize>> = vec![None; n];
        let mut interior = Vec::new();
        for (node, &is_boundary) in on_boundary.iter().enumerate() {
            if !is_boundary {
                slot[node] = Some(interior.len());
                interior.push(node);
            }
        }
        let n_interior = interior.len();
        if n_interior > MAX_INTERIOR_NODES {
            return Err(Error::InvalidInput(format!(
                "the dense mesh-motion solve supports at most {MAX_INTERIOR_NODES} interior \
                 nodes, got {n_interior}"
            )));
        }
        let mut k_ii = vec![vec![0.0; n_interior]; n_interior];
        let mut coupling: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_interior];
        for (e, element) in elements.iter().enumerate() {
            let corners = element_corners(reference, element).ok_or_else(|| {
                Error::InvalidInput(format!("element {e} references a node outside the mesh"))
            })?;
            let triangle = Triangle::new(&corners).ok_or_else(|| {
                Error::InvalidInput(format!("element {e} is degenerate in the reference mesh"))
            })?;
            for (la, &na) in element.iter().enumerate() {
                let Some(row) = slot[na] else { continue };
                for (lb, &nb) in element.iter().enumerate() {
                    let entry = triangle.area
                        * (triangle.grad[la][0] * triangle.grad[lb][0]
                            + triangle.grad[la][1] * triangle.grad[lb][1]);
                    match slot[nb] {
                        Some(col) => k_ii[row][col] += entry,
                        None => coupling[row].push((nb, entry)),
                    }
                }
            }
        }
        let lu = if n_interior == 0 {
            None
        } else {
            Some(LuFactor::new(&k_ii).ok_or(Error::SingularSystem)?)
        };
        Ok(Self {
            interior,
            coupling,
            lu,
        })
    }

    /// Extend the prescribed displacements.  `prescribed[node]` is read for the
    /// boundary nodes only; the result holds the prescribed value at every
    /// boundary node and the harmonic extension at every interior node.
    pub(super) fn extend(&self, prescribed: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let mut displacement = prescribed.to_vec();
        let Some(lu) = &self.lu else {
            return displacement;
        };
        let value_at = |node: usize| prescribed.get(node).copied().unwrap_or([0.0; 2]);
        let mut rhs_x = Vec::with_capacity(self.interior.len());
        let mut rhs_y = Vec::with_capacity(self.interior.len());
        for couplings in &self.coupling {
            let (mut sx, mut sy) = (0.0, 0.0);
            for &(node, entry) in couplings {
                let d = value_at(node);
                sx -= entry * d[0];
                sy -= entry * d[1];
            }
            rhs_x.push(sx);
            rhs_y.push(sy);
        }
        let ux = lu.solve(&rhs_x);
        let uy = lu.solve(&rhs_y);
        for ((&node, &x), &y) in self.interior.iter().zip(ux.iter()).zip(uy.iter()) {
            if let Some(entry) = displacement.get_mut(node) {
                *entry = [x, y];
            }
        }
        displacement
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Positions, elements and boundary flags of a test mesh.
    type TestMesh = (Vec<[f64; 2]>, Vec<[usize; 3]>, Vec<bool>);

    /// Structured `nx × ny` mesh on the unit square (two triangles per cell),
    /// returns (positions, elements, boundary flags).
    fn grid(nx: usize, ny: usize) -> TestMesh {
        let idx = |i: usize, j: usize| j * (nx + 1) + i;
        let mut positions = Vec::new();
        let mut on_boundary = Vec::new();
        for j in 0..=ny {
            for i in 0..=nx {
                positions.push([i as f64 / nx as f64, j as f64 / ny as f64]);
                on_boundary.push(i == 0 || j == 0 || i == nx || j == ny);
            }
        }
        let mut elements = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                elements.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
                elements.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
            }
        }
        (positions, elements, on_boundary)
    }

    #[test]
    fn affine_boundary_displacements_extend_exactly() {
        let (mut positions, elements, on_boundary) = grid(6, 5);
        // Perturb the interior so the test does not rely on a regular mesh.
        for (k, p) in positions.iter_mut().enumerate() {
            if !on_boundary[k] {
                p[0] += 0.03 * ((k * 7 % 5) as f64 - 2.0) / 2.0;
                p[1] += 0.03 * ((k * 3 % 7) as f64 - 3.0) / 3.0;
            }
        }
        let extension = HarmonicExtension::new(&positions, &elements, &on_boundary).unwrap();
        let affine = |p: &[f64; 2]| {
            [
                0.3 + 0.2 * p[0] - 0.1 * p[1],
                -0.4 + 0.5 * p[0] + 0.7 * p[1],
            ]
        };
        let prescribed: Vec<[f64; 2]> = positions
            .iter()
            .zip(on_boundary.iter())
            .map(|(p, &b)| if b { affine(p) } else { [0.0, 0.0] })
            .collect();
        let displacement = extension.extend(&prescribed);
        for (p, d) in positions.iter().zip(displacement.iter()) {
            let e = affine(p);
            assert!(
                (d[0] - e[0]).abs() < 1e-12 && (d[1] - e[1]).abs() < 1e-12,
                "{p:?}: {d:?} vs {e:?}"
            );
        }
    }

    #[test]
    fn interior_displacement_obeys_the_maximum_principle_and_decays_from_the_wall() {
        let (positions, elements, on_boundary) = grid(8, 6);
        let extension = HarmonicExtension::new(&positions, &elements, &on_boundary).unwrap();
        // Bottom edge displaced by 1, everything else fixed.
        let prescribed: Vec<[f64; 2]> = positions
            .iter()
            .map(|p| if p[1] == 0.0 { [0.0, 1.0] } else { [0.0, 0.0] })
            .collect();
        let displacement = extension.extend(&prescribed);
        for (k, d) in displacement.iter().enumerate() {
            assert!(d[0].abs() < 1e-12, "no x-displacement is driven");
            assert!(d[1] > -1e-12 && d[1] < 1.0 + 1e-12, "node {k}: {}", d[1]);
        }
        // Along the centre column the displacement decreases monotonically.
        let column: Vec<f64> = (0..=6).map(|j| displacement[j * 9 + 4][1]).collect();
        assert!(column.windows(2).all(|w| w[0] > w[1]), "{column:?}");
    }

    #[test]
    fn meshes_without_interior_nodes_pass_the_data_through() {
        let (positions, elements, on_boundary) = grid(3, 1);
        assert!(on_boundary.iter().all(|b| *b));
        let extension = HarmonicExtension::new(&positions, &elements, &on_boundary).unwrap();
        let prescribed: Vec<[f64; 2]> = (0..positions.len())
            .map(|k| [k as f64, -(k as f64)])
            .collect();
        assert_eq!(extension.extend(&prescribed), prescribed);
    }

    #[test]
    fn invalid_meshes_are_reported() {
        let (positions, elements, on_boundary) = grid(3, 3);
        assert!(HarmonicExtension::new(&positions, &elements, &on_boundary[1..]).is_err());
        let mut bad_elements = elements.clone();
        bad_elements[0][1] = 999;
        assert!(matches!(
            HarmonicExtension::new(&positions, &bad_elements, &on_boundary),
            Err(Error::InvalidInput(_))
        ));
        // An interior node that no element touches makes K_ii singular.
        let mut orphan_positions = positions.clone();
        orphan_positions.push([0.5, 0.5]);
        let mut flags = on_boundary.clone();
        flags.push(false);
        assert!(matches!(
            HarmonicExtension::new(&orphan_positions, &elements, &flags),
            Err(Error::SingularSystem)
        ));
        // Degenerate element.
        let mut flat = positions.clone();
        flat[elements[0][1]] = flat[elements[0][0]];
        assert!(HarmonicExtension::new(&flat, &elements, &on_boundary).is_err());
    }
}
