//! Cached neighbour lists of [`SphSim`](super::SphSim).
//!
//! Every pair sum of the array-based mini simulator (density, pressure,
//! viscosity and surface tension) only involves the particles inside the
//! cubic-spline support radius `2h`.  [`SphSimNeighbors`] finds those
//! neighbours once with a spatial hash in `O(N)` expected time and stores them
//! in the CSR layout of [`FlatNeighborList`] (`offsets` + `indices`), so the sums
//! cost `O(N k)` for `k` neighbours per particle instead of `O(N²)`.
//!
//! The lists remember the positions and kernel radius they were built for, which
//! makes it cheap and safe to decide whether they are still valid: positions are
//! public fields of `SphSim` and may be changed by the caller at any time.

use std::collections::HashMap;

use crate::neighbor::{FlatNeighborList, SpatialHash3D};

use super::types_sim::wrap_into_box;

/// Separation `a − b` under the minimum-image convention of the periodic box
/// `periodic`; `None` gives the plain difference.
///
/// Axis `k` is wrapped, `dₖ ← dₖ − Lₖ round(dₖ / Lₖ)`, only when `Lₖ` is a
/// positive finite number, which puts the component into `[−Lₖ/2, Lₖ/2]`; any
/// other axis keeps the plain difference.  This is the convention shared by the
/// neighbour search below and the pair sums of [`SphSim`](super::SphSim), so a
/// pair is a neighbour exactly when the sums see it inside the support.
#[inline]
pub(crate) fn minimum_image_delta(
    a: [f64; 3],
    b: [f64; 3],
    periodic: Option<[f64; 3]>,
) -> [f64; 3] {
    let mut d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    if let Some(box_size) = periodic {
        for (dk, &l) in d.iter_mut().zip(box_size.iter()) {
            if l.is_finite() && l > 0.0 {
                *dk -= l * (*dk / l).round();
            }
        }
    }
    d
}

/// Neighbour lists of a particle cloud for the cubic-spline support `2h`.
///
/// Row `i` holds the indices `j ≠ i` with `|xᵢ − xⱼ| ≤ 2h` in ascending order.
/// In a periodic box (see [`build_with`](Self::build_with)) the distance is the
/// minimum-image distance, so particles on opposite faces of the box are
/// neighbours.  A non-positive or non-finite kernel radius gives empty rows.
#[derive(Debug, Clone)]
pub struct SphSimNeighbors {
    /// Kernel radius `h` the lists were built for.
    kernel_radius: f64,
    /// Periodic box `[Lx, Ly, Lz]` the lists were built for, if any.
    periodic: Option<[f64; 3]>,
    /// Particle positions the lists were built for.
    positions: Vec<[f64; 3]>,
    /// The lists in CSR layout.
    list: FlatNeighborList,
}

impl SphSimNeighbors {
    /// Lists for no particles, which are never current for any kernel radius.
    pub fn empty() -> Self {
        Self {
            kernel_radius: f64::NAN,
            periodic: None,
            positions: Vec::new(),
            list: FlatNeighborList::from_neighbor_lists(&[]),
        }
    }

    /// Build the lists of `positions` for the support radius `2 · kernel_radius`
    /// in an open domain.
    pub fn build(positions: &[[f64; 3]], kernel_radius: f64) -> Self {
        Self::build_with(positions, kernel_radius, None)
    }

    /// Build the lists of `positions` for the support radius `2 · kernel_radius`,
    /// optionally in the periodic box `periodic = Some([Lx, Ly, Lz])`.
    ///
    /// In a periodic box the separation of a pair is the minimum-image
    /// separation `dₖ − Lₖ round(dₖ / Lₖ)` (also used by the pair sums of
    /// [`SphSim`](super::SphSim)): an axis is periodic when its
    /// length is a positive finite number, otherwise it is open.  Positions need
    /// not lie inside the box.  The lists stay duplicate-free and correct for
    /// any box length (also for boxes shorter than the support radius), but the
    /// minimum-image convention itself is exact only if every periodic length is
    /// at least `4 · kernel_radius`.  Particles at non-finite positions have no
    /// neighbours and are nobody's neighbour in the periodic case.
    pub fn build_with(
        positions: &[[f64; 3]],
        kernel_radius: f64,
        periodic: Option<[f64; 3]>,
    ) -> Self {
        let radius = 2.0 * kernel_radius;
        let rows = if radius > 0.0 && radius.is_finite() {
            match periodic {
                Some(box_size) => periodic_neighbor_rows(positions, radius, box_size),
                None => SpatialHash3D::all_neighbors(positions, radius),
            }
        } else {
            vec![Vec::new(); positions.len()]
        };
        Self {
            kernel_radius,
            periodic,
            positions: positions.to_vec(),
            list: FlatNeighborList::from_neighbor_lists(&rows),
        }
    }

    /// Whether the lists were built for exactly these `positions` and this
    /// kernel radius in an open domain, i.e. whether they can be used for them
    /// as they are.
    pub fn is_current(&self, positions: &[[f64; 3]], kernel_radius: f64) -> bool {
        self.is_current_with(positions, kernel_radius, None)
    }

    /// Whether the lists were built for exactly these `positions`, this kernel
    /// radius and this periodic box (`None`: open domain).
    pub fn is_current_with(
        &self,
        positions: &[[f64; 3]],
        kernel_radius: f64,
        periodic: Option<[f64; 3]>,
    ) -> bool {
        self.kernel_radius == kernel_radius
            && self.periodic == periodic
            && self.positions.as_slice() == positions
    }

    /// The periodic box `[Lx, Ly, Lz]` the lists were built for, if any.
    pub fn periodic(&self) -> Option<[f64; 3]> {
        self.periodic
    }

    /// The neighbour indices of particle `i`; empty if `i` is out of range.
    pub fn neighbors_of(&self, i: usize) -> &[usize] {
        let end = i
            .checked_add(1)
            .and_then(|next| self.list.offsets.get(next));
        match (self.list.offsets.get(i), end) {
            (Some(&start), Some(&end)) => self.list.flat_neighbors.get(start..end).unwrap_or(&[]),
            _ => &[],
        }
    }

    /// Number of particles the lists were built for.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Whether the lists were built for no particles.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Total number of stored neighbour entries (each pair is stored twice).
    pub fn entry_count(&self) -> usize {
        self.list.total_pairs()
    }

    /// The kernel radius `h` the lists were built for.
    pub fn kernel_radius(&self) -> f64 {
        self.kernel_radius
    }

    /// The lists in CSR layout (`offsets`, `flat_neighbors`).
    pub fn as_flat(&self) -> &FlatNeighborList {
        &self.list
    }
}

impl Default for SphSimNeighbors {
    fn default() -> Self {
        Self::empty()
    }
}

/// Number of cells along a periodic axis of length `length` for cells at least
/// `radius` wide, and the resulting cell width.  A box shorter than `radius`
/// is a single cell; the count is capped so index arithmetic cannot overflow.
fn periodic_axis_cells(length: f64, radius: f64) -> (i64, f64) {
    const MAX_CELLS: f64 = 1.0e12;
    let count = (length / radius).floor().clamp(1.0, MAX_CELLS) as i64;
    (count, length / count as f64)
}

/// Neighbour rows of `positions` for the support `radius` in the periodic box
/// `box_size` (axes whose length is not a positive finite number are open).
///
/// A hash grid with cells at least `radius` wide is laid over the box (wrapping
/// at the faces); a pair is listed when its minimum-image distance is at most
/// `radius`.  The wrapped neighbour cells of a particle are de-duplicated, which
/// matters when an axis has fewer than three cells and the `±1` stencil would
/// otherwise visit one cell twice, so every neighbour appears exactly once.
/// Rows are sorted; particles at non-finite positions are skipped.  Expected
/// cost is `O(N k)` for `k` neighbours per particle.
fn periodic_neighbor_rows(
    positions: &[[f64; 3]],
    radius: f64,
    box_size: [f64; 3],
) -> Vec<Vec<usize>> {
    let periodic_axis = box_size.map(|l| l.is_finite() && l > 0.0);
    let mut cell_count = [i64::MAX; 3];
    let mut cell_width = [radius; 3];
    for k in 0..3 {
        if periodic_axis[k] {
            (cell_count[k], cell_width[k]) = periodic_axis_cells(box_size[k], radius);
        }
    }
    let cell_of = |p: &[f64; 3]| -> Option<[i64; 3]> {
        let mut cell = [0_i64; 3];
        for k in 0..3 {
            if !p[k].is_finite() {
                return None;
            }
            cell[k] = if periodic_axis[k] {
                let wrapped = wrap_into_box(p[k], box_size[k]);
                ((wrapped / cell_width[k]).floor() as i64).clamp(0, cell_count[k] - 1)
            } else {
                (p[k] / cell_width[k]).floor() as i64
            };
        }
        Some(cell)
    };
    // The distinct cells `c - 1, c, c + 1` of axis `k`, wrapped on a periodic axis.
    let axis_cells = |k: usize, c: i64| -> ([i64; 3], usize) {
        let mut cells = [c.saturating_sub(1), c, c.saturating_add(1)];
        if periodic_axis[k] {
            for v in &mut cells {
                *v = v.rem_euclid(cell_count[k]);
            }
        }
        cells.sort_unstable();
        let mut distinct = [cells[0]; 3];
        let mut len = 1;
        for &v in &cells[1..] {
            if v != distinct[len - 1] {
                distinct[len] = v;
                len += 1;
            }
        }
        (distinct, len)
    };

    let cells: Vec<Option<[i64; 3]>> = positions.iter().map(cell_of).collect();
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, cell) in cells.iter().enumerate() {
        if let Some(key) = cell {
            grid.entry(*key).or_default().push(i);
        }
    }
    let radius2 = radius * radius;
    let periodic = Some(box_size);
    positions
        .iter()
        .zip(cells.iter())
        .enumerate()
        .map(|(i, (pos, cell))| {
            let mut row = Vec::new();
            let Some(c) = cell else {
                return row;
            };
            let (xs, nx) = axis_cells(0, c[0]);
            let (ys, ny) = axis_cells(1, c[1]);
            let (zs, nz) = axis_cells(2, c[2]);
            for &cx in &xs[..nx] {
                for &cy in &ys[..ny] {
                    for &cz in &zs[..nz] {
                        let Some(members) = grid.get(&[cx, cy, cz]) else {
                            continue;
                        };
                        for &j in members {
                            if j == i {
                                continue;
                            }
                            let d = minimum_image_delta(*pos, positions[j], periodic);
                            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= radius2 {
                                row.push(j);
                            }
                        }
                    }
                }
            }
            row.sort_unstable();
            row
        })
        .collect()
}
