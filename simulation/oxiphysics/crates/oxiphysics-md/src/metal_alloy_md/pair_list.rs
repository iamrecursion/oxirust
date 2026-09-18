// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Verlet pair list of an [`AlloySystem`](super::AlloySystem).
//!
//! The list holds every pair `i < j` whose minimum-image distance was below
//! `cutoff + skin` when it was built.  As long as no atom has moved by more than
//! `skin / 2` since then, every pair that is closer than `cutoff` is in the list,
//! so the force and energy evaluations only have to re-measure the listed pairs
//! instead of searching all `N (N - 1) / 2` pairs again.
//!
//! The candidate search bins the atoms into cells of the fractional coordinates
//! of the (possibly tilted) simulation box and compares each atom with the atoms
//! of the neighbouring cells, which is `O(N)` for a homogeneous density.  It works
//! on [`SimBox`], including its `xy` tilt, which is why the orthorhombic
//! [`crate::neighbor::VerletList`] (whose `PeriodicBox` has no tilt) cannot be used
//! here: the sheared boxes of the elastic-constant calculation would silently get
//! the wrong periodic images.
//!
//! Every pair is measured with [`SimBox::min_image`], exactly like the brute-force
//! search this list replaces, and the pairs are stored in lexicographic order, so
//! the summation order (and therefore every floating-point result) of the force
//! and energy loops is the same as before.

use super::functions::{vdot, vsub};
use super::types::{MetalAtom, SimBox};

/// Verlet skin (Å): the list is built for `cutoff + skin` and stays valid while
/// no atom has moved by more than half of it.
pub(super) const PAIR_LIST_SKIN: f64 = 0.3;

/// The cell grid never has more than this many cells per atom (at least 27 in
/// total), which keeps the memory of a sparse system proportional to its atom
/// count.  Cells larger than the interaction reach are always correct.
const MAX_CELLS_PER_ATOM: f64 = 2.0;
/// Upper bound of the cell count along one axis before the total is capped.
const MAX_CELLS_PER_AXIS: f64 = 1.0e6;
/// An atom further than this many box periods from the origin is not binned: the
/// fractional part of such a coordinate has lost its digits to rounding, so its
/// cell would not match its minimum-image position.  These atoms are compared
/// with all others instead.
const MAX_BINNED_PERIODS: f64 = 1_048_576.0;

/// A Verlet list valid for the atoms, box and cutoff it was built from.
#[derive(Debug, Clone)]
pub(super) struct PairList {
    /// Pairs `(i, j)`, `i < j`, in lexicographic order.
    pairs: Vec<(usize, usize)>,
    /// Positions at the time of the build.
    reference: Vec<[f64; 3]>,
    /// Box edge lengths at the time of the build.
    lengths: [f64; 3],
    /// Box tilt at the time of the build.
    xy: f64,
    /// Interaction cutoff the list was built for.
    cutoff: f64,
    /// Skin the list was built with.
    skin: f64,
    /// Whether the list may be reused after the atoms moved (see
    /// [`PairList::build`]).
    reusable: bool,
    /// Number of candidate pair distances evaluated by the build.
    candidates: usize,
}

impl PairList {
    /// Build the list of all pairs closer than `cutoff + skin` (a negative or
    /// non-finite skin counts as zero).
    ///
    /// The list is reusable (see [`PairList::is_valid_for`]) only when the
    /// periodic distance of a pair is a genuine nearest-image distance that the
    /// atoms can change by at most their displacement: always for an
    /// orthorhombic box, and for a tilted box only when every edge is at least
    /// twice the list reach (the sequential minimum-image convention of a tilted
    /// box finds the nearest image of every pair within the reach only then).
    pub(super) fn build(atoms: &[MetalAtom], sim_box: &SimBox, cutoff: f64, skin: f64) -> Self {
        let skin = if skin.is_finite() && skin > 0.0 {
            skin
        } else {
            0.0
        };
        let reach = cutoff + skin;
        let box_ok = sim_box.validate().is_ok();
        let reach_ok = reach.is_finite() && reach > 0.0;
        let reusable = box_ok
            && reach_ok
            && (sim_box.xy == 0.0 || sim_box.lengths.iter().all(|&l| l >= 2.0 * reach));
        let (pairs, candidates) = if !reach_ok {
            (Vec::new(), 0)
        } else {
            match CellGrid::new(atoms, sim_box, reach) {
                Some(grid) => grid.collect_pairs(atoms, sim_box, reach),
                None => brute_force_pairs(atoms, sim_box, reach),
            }
        };
        Self {
            pairs,
            reference: atoms.iter().map(|atom| atom.position).collect(),
            lengths: sim_box.lengths,
            xy: sim_box.xy,
            cutoff,
            skin,
            reusable,
            candidates,
        }
    }

    /// Whether the list still contains every pair closer than `cutoff` for the
    /// given atoms and box: same atom count, box and cutoff as at the build, and
    /// no atom further than `skin / 2` (periodic distance) from its position at
    /// the build.  A non-finite displacement never passes.
    pub(super) fn is_valid_for(&self, atoms: &[MetalAtom], sim_box: &SimBox, cutoff: f64) -> bool {
        if !self.reusable
            || atoms.len() != self.reference.len()
            || self.cutoff != cutoff
            || self.lengths != sim_box.lengths
            || self.xy != sim_box.xy
        {
            return false;
        }
        let limit = 0.5 * self.skin;
        let limit_squared = limit * limit;
        atoms.iter().zip(self.reference.iter()).all(|(atom, old)| {
            let moved = sim_box.min_image(vsub(atom.position, *old));
            vdot(moved, moved) <= limit_squared
        })
    }

    /// The pairs `(i, j)`, `i < j`, in lexicographic order.
    pub(super) fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Number of candidate pair distances the build evaluated.
    pub(super) fn candidates(&self) -> usize {
        self.candidates
    }
}

/// Every pair of `atoms` closer than `reach`, by testing all of them.
fn brute_force_pairs(
    atoms: &[MetalAtom],
    sim_box: &SimBox,
    reach: f64,
) -> (Vec<(usize, usize)>, usize) {
    let reach_squared = reach * reach;
    let mut pairs = Vec::new();
    let mut candidates = 0usize;
    for (i, first) in atoms.iter().enumerate() {
        for (offset, second) in atoms[i + 1..].iter().enumerate() {
            candidates += 1;
            let dr = sim_box.min_image(vsub(second.position, first.position));
            if vdot(dr, dr) < reach_squared {
                pairs.push((i, i + 1 + offset));
            }
        }
    }
    (pairs, candidates)
}

/// Atoms binned into the cells of the fractional coordinates of the box.
struct CellGrid {
    /// Number of cells along the three box vectors.
    cells: [usize; 3],
    /// Cell of every atom, `None` for an atom without finite coordinates and for
    /// a stray atom.
    cell_of_atom: Vec<Option<usize>>,
    /// Atoms with finite coordinates too far from the box to be binned.
    strays: Vec<usize>,
    /// Atoms of cell `c` are `atoms_in_cell[start[c]..start[c + 1]]`, ascending.
    start: Vec<usize>,
    /// Atom indices grouped by cell.
    atoms_in_cell: Vec<usize>,
}

impl CellGrid {
    /// Bin the atoms for the interaction reach `reach`; `None` when the box is
    /// degenerate (a non-positive or non-finite edge or tilt), where no
    /// fractional coordinates exist.
    fn new(atoms: &[MetalAtom], sim_box: &SimBox, reach: f64) -> Option<Self> {
        sim_box.validate().ok()?;
        let cells = cell_counts(sim_box, reach, atoms.len());
        let total = cells[0] * cells[1] * cells[2];
        let mut cell_of_atom = Vec::with_capacity(atoms.len());
        let mut strays = Vec::new();
        let mut counts = vec![0usize; total + 1];
        for (i, atom) in atoms.iter().enumerate() {
            let cell = match fractional_coordinates(atom.position, sim_box) {
                Placement::Cell(f) => {
                    let index = |axis: usize| {
                        ((f[axis] * cells[axis] as f64) as usize).min(cells[axis] - 1)
                    };
                    Some((index(0) * cells[1] + index(1)) * cells[2] + index(2))
                }
                Placement::Stray => {
                    strays.push(i);
                    None
                }
                Placement::Unplaced => None,
            };
            if let Some(cell) = cell {
                counts[cell + 1] += 1;
            }
            cell_of_atom.push(cell);
        }
        for c in 0..total {
            counts[c + 1] += counts[c];
        }
        let start = counts.clone();
        let mut fill = counts;
        let mut atoms_in_cell = vec![0usize; start[total]];
        for (i, cell) in cell_of_atom.iter().enumerate() {
            if let Some(cell) = *cell {
                atoms_in_cell[fill[cell]] = i;
                fill[cell] += 1;
            }
        }
        Some(Self {
            cells,
            cell_of_atom,
            strays,
            start,
            atoms_in_cell,
        })
    }

    /// All pairs `i < j` closer than `reach`, and the number of candidates tested.
    fn collect_pairs(
        &self,
        atoms: &[MetalAtom],
        sim_box: &SimBox,
        reach: f64,
    ) -> (Vec<(usize, usize)>, usize) {
        let reach_squared = reach * reach;
        let [nx, ny, nz] = self.cells;
        let mut pairs = Vec::new();
        let mut partners: Vec<usize> = Vec::new();
        let mut candidates = 0usize;
        for (i, cell) in self.cell_of_atom.iter().enumerate() {
            let Some(cell) = *cell else { continue };
            let cz = cell % nz;
            let cy = (cell / nz) % ny;
            let cx = cell / (ny * nz);
            let (xs, x_count) = stencil(cx, nx);
            let (ys, y_count) = stencil(cy, ny);
            let (zs, z_count) = stencil(cz, nz);
            partners.clear();
            for &ax in &xs[..x_count] {
                for &ay in &ys[..y_count] {
                    for &az in &zs[..z_count] {
                        let neighbour = (ax * ny + ay) * nz + az;
                        let members =
                            &self.atoms_in_cell[self.start[neighbour]..self.start[neighbour + 1]];
                        for &j in members {
                            if j <= i {
                                continue;
                            }
                            candidates += 1;
                            let dr = sim_box.min_image(vsub(atoms[j].position, atoms[i].position));
                            if vdot(dr, dr) < reach_squared {
                                partners.push(j);
                            }
                        }
                    }
                }
            }
            partners.sort_unstable();
            pairs.extend(partners.iter().map(|&j| (i, j)));
        }
        if !self.strays.is_empty() {
            for &stray in &self.strays {
                for j in (0..atoms.len()).filter(|&j| j != stray) {
                    candidates += 1;
                    let dr = sim_box.min_image(vsub(atoms[j].position, atoms[stray].position));
                    if vdot(dr, dr) < reach_squared {
                        pairs.push((stray.min(j), stray.max(j)));
                    }
                }
            }
            pairs.sort_unstable();
            pairs.dedup();
        }
        (pairs, candidates)
    }
}

/// Indices of the cells that neighbour cell `c` of `n` along one axis, with
/// periodic wrap and without repeats (one cell for `n = 1`, both for `n = 2`).
fn stencil(c: usize, n: usize) -> ([usize; 3], usize) {
    match n {
        0 | 1 => ([0, 0, 0], 1),
        2 => ([0, 1, 0], 2),
        _ => ([(c + n - 1) % n, c, (c + 1) % n], 3),
    }
}

/// Number of cells along each box vector: as many as keep the distance between
/// opposite cell faces at least `reach` (so all partners within `reach` lie in
/// the same or an adjacent cell), reduced until the grid has at most
/// `MAX_CELLS_PER_ATOM` cells per atom.  The box must be valid.
fn cell_counts(sim_box: &SimBox, reach: f64, n_atoms: usize) -> [usize; 3] {
    let [lx, ly, lz] = sim_box.lengths;
    // Distance between the faces of the cell spanned by (lx,0,0), (xy,ly,0), (0,0,lz):
    // 1 / |reciprocal vector|, which for the x face is lx ly / sqrt(ly^2 + xy^2).
    let widths = [lx * ly / ly.hypot(sim_box.xy), ly, lz];
    let mut counts = widths.map(|w| (w / reach).floor().clamp(1.0, MAX_CELLS_PER_AXIS));
    let max_total = (MAX_CELLS_PER_ATOM * n_atoms as f64).max(27.0);
    while counts[0] * counts[1] * counts[2] > max_total {
        let (largest, _) =
            counts.iter().enumerate().fold(
                (0, 0.0_f64),
                |best, (axis, &c)| {
                    if c > best.1 { (axis, c) } else { best }
                },
            );
        if counts[largest] <= 1.0 {
            break;
        }
        counts[largest] = (counts[largest] / 2.0).floor().max(1.0);
    }
    counts.map(|c| c as usize)
}

/// Where an atom goes in the cell grid.
enum Placement {
    /// Fractional coordinates in `[0, 1)` along the three box vectors.
    Cell([f64; 3]),
    /// Finite, but too many periods away from the origin to be binned.
    Stray,
    /// A coordinate is not finite: the atom pairs with nothing.
    Unplaced,
}

/// Fractional coordinates of `position` with respect to the box vectors
/// `(lx, 0, 0)`, `(xy, ly, 0)`, `(0, 0, lz)`, each reduced to `[0, 1)`.
fn fractional_coordinates(position: [f64; 3], sim_box: &SimBox) -> Placement {
    let [lx, ly, lz] = sim_box.lengths;
    let fy = position[1] / ly;
    let fx = (position[0] - sim_box.xy * fy) / lx;
    let fz = position[2] / lz;
    if !(fx.is_finite() && fy.is_finite() && fz.is_finite()) {
        return Placement::Unplaced;
    }
    if fx.abs().max(fy.abs()).max(fz.abs()) > MAX_BINNED_PERIODS {
        return Placement::Stray;
    }
    let reduce = |f: f64| {
        let r = f - f.floor();
        // A coordinate a hair below an integer rounds up to exactly 1.
        if r >= 1.0 { 0.0 } else { r }
    };
    Placement::Cell([reduce(fx), reduce(fy), reduce(fz)])
}
