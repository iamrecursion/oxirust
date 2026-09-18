// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Uniform spatial hash grid for fast neighbourhood queries.
//!
//! A [`SpatialGrid`] partitions world space into cubic cells of a fixed size.
//! Every registered entry (identified by a `u32` ID) is stored in the cell that
//! contains its position, and the occupied cells are kept in a hash map keyed
//! by the integer cell coordinate.  A query therefore only looks at the cells
//! it can possibly touch instead of at every entry:
//!
//! | Operation | Cost |
//! |-----------|------|
//! | [`insert`](SpatialGrid::insert), [`update`](SpatialGrid::update), [`remove`](SpatialGrid::remove), [`contains`](SpatialGrid::contains), [`position`](SpatialGrid::position) | O(1) expected |
//! | [`query_radius`](SpatialGrid::query_radius), [`query_aabb`](SpatialGrid::query_aabb) | the cells overlapping the query box plus the entries in them (never more than a scan of all occupied cells) |
//! | [`nearest`](SpatialGrid::nearest), [`k_nearest`](SpatialGrid::k_nearest) | expanding shells of cells around the query point; the search stops as soon as the answer is provably final |
//! | [`pairs_within_radius`](SpatialGrid::pairs_within_radius) | each entry is only compared with the entries of nearby cells |
//!
//! ## Choosing `cell_size`
//!
//! Set `cell_size` ≈ the typical query radius (for
//! [`pairs_within_radius`](SpatialGrid::pairs_within_radius): the pair
//! distance).  Cells much smaller than the query radius make a query visit many
//! empty cells (a query never visits more cells than the grid has occupied, so
//! it degrades to a scan of the occupied cells; `pairs_within_radius` looks at
//! every occupied cell for the neighbours of a cell once the neighbourhood
//! outnumbers the occupied cells); cells much larger than the query radius put
//! many unrelated entries into the same cell, which degenerates towards a
//! linear scan.  The cell size can be changed later with
//! [`set_cell_size`](SpatialGrid::set_cell_size), which re-buckets every entry.
//!
//! ## Exactness
//!
//! [`query_radius`](SpatialGrid::query_radius),
//! [`query_aabb`](SpatialGrid::query_aabb) and
//! [`pairs_within_radius`](SpatialGrid::pairs_within_radius) return exactly
//! what a brute-force scan with the floating-point test stated on the method
//! (`dist_sq <= radius * radius`, boundary inclusive) would return, for any cell
//! size and any coordinates.  The cell of a point and the distance between two
//! points round independently of each other, so two points at (about) the search
//! radius -- neighbours on a lattice with spacing 0.1, say -- can lie one cell
//! further apart than `ceil(radius / cell_size)`.  The cell search of
//! `query_radius` and `pairs_within_radius` is therefore widened by a few ulp of
//! the coordinates.  That costs an extra layer of cells only for entries within
//! the margin of a cell boundary (for example the points of a lattice whose
//! spacing is the cell size).  The nearest-neighbour queries stop searching once
//! no unvisited entry can be closer, keeping a safety margin of 1e-9 cell sizes
//! against the same rounding.
//!
//! ## Determinism
//!
//! The grid uses an unseeded hasher and never exposes hash-map iteration
//! order: [`query_radius`](SpatialGrid::query_radius),
//! [`query_aabb`](SpatialGrid::query_aabb) and
//! [`pairs_within_radius`](SpatialGrid::pairs_within_radius) return their
//! results sorted by ID, and [`nearest`](SpatialGrid::nearest) /
//! [`k_nearest`](SpatialGrid::k_nearest) break distance ties by the smaller ID.
//! [`iter`](SpatialGrid::iter) yields the entries in insertion order (with a
//! removed entry's slot taken over by the most recently inserted one).
//!
//! ## Non-finite coordinates
//!
//! Entries with NaN coordinates are stored but never match a radius, AABB or
//! pair query.  Entries at infinite coordinates are only matched by an
//! unbounded query (an infinite radius or an infinite box).  In
//! nearest-neighbour queries both kinds are the last candidates, at infinite
//! distance.
//!
//! ## Example
//!
//! ```rust
//! use oxiphysics::spatial_grid::SpatialGrid;
//!
//! let mut grid = SpatialGrid::new(5.0); // 5 m cells
//! grid.insert(0, [1.0, 0.0,  2.0]);
//! grid.insert(1, [3.0, 0.0, -1.0]);
//! grid.insert(2, [100.0, 0.0, 100.0]); // far away
//!
//! let hits = grid.query_radius([0.0, 0.0, 0.0], 5.0);
//! assert!(hits.contains(&0));
//! assert!(hits.contains(&1));
//! assert!(!hits.contains(&2));
//! ```

use serde::de::Error as _;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors reported by [`SpatialGrid`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum GridError {
    /// The cell size must be a positive, finite number (the offending value is
    /// carried along).
    InvalidCellSize(f64),
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridError::InvalidCellSize(size) => {
                write!(f, "cell_size must be positive and finite, got {size}")
            }
        }
    }
}

impl std::error::Error for GridError {}

// ---------------------------------------------------------------------------
// Hashing — private
// ---------------------------------------------------------------------------

/// A small, fast, unseeded hasher for the integer keys of the grid.
///
/// The standard library's default hasher is randomly seeded and tuned against
/// hash-flooding; neither helps a spatial grid, which looks up a handful of
/// three-integer keys per entry.  Unseeded hashing also keeps the grid
/// reproducible from run to run.  (It is consequently not hardened against
/// adversarially chosen cell coordinates: validate untrusted positions before
/// inserting them.)
#[derive(Debug, Clone, Copy, Default)]
struct CellHasher {
    state: u64,
}

impl CellHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.state = (self.state.rotate_left(5) ^ word).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

impl Hasher for CellHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0_u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_i32(&mut self, value: i32) {
        self.add(u64::from(value.cast_unsigned()));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // SplitMix64 finaliser: the hash table picks buckets from the low bits,
        // which must depend on every input bit.
        let mut x = self.state;
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<CellHasher>>;

// ---------------------------------------------------------------------------
// Position -> cell mapping and rounding margins — private
// ---------------------------------------------------------------------------

/// Relative margin (2^-50, about 8.9e-16) by which a cell search is widened:
/// more than the rounding errors of the floating-point distance test and of the
/// sums `x ± radius` can add up to (see [`inflated_radius`]).
const SEARCH_MARGIN_REL: f64 = 1.0 / (1_u64 << 50) as f64;

/// Absolute margin (about 1e-159) added to every widened search radius.  Squared
/// distances below about 1e-324 underflow to zero, so for any radius the
/// distance test also accepts points up to `sqrt(2^-1074)` = 2^-537 (about
/// 2.5e-162) away; the margin is a little larger than that and negligible next
/// to the size of any real cell.
const SEARCH_MARGIN_ABS: f64 = 1.0e-159;

/// The index of the cell that contains the coordinate `value`.  Coordinates
/// beyond the `i32` cell range saturate and NaN maps to cell 0.  The mapping is
/// monotonic: a larger coordinate never lands in a smaller cell.
#[inline]
fn cell_index(value: f64, inv_cell: f64) -> i32 {
    (value * inv_cell).floor() as i32
}

/// The radius for the cell search around the coordinate `x` of a query of
/// radius `radius` (`radius` must not be NaN).
///
/// The search must find every point that passes the floating-point test
/// `dist_sq(point, center) <= radius * radius`, whichever way the roundings
/// fall.  Working through the test, a point that passes is at most
/// `radius * (1 + 2^-51) + 2^-536` away from `x` along every axis, and the
/// bounds `x ± radius` that delimit the search are rounded by up to
/// `2^-53 * (|x| + radius)`.  The returned radius exceeds both together with
/// room to spare, so the cells that contain `x - inflated` and `x + inflated`
/// -- cell indices are monotonic in the coordinate -- bracket the cell of every
/// point that passes.
///
/// Infinite when the test accepts every point that is not NaN, which is the
/// case as soon as `radius * radius` overflows.
#[inline]
fn inflated_radius(x: f64, radius: f64) -> f64 {
    if (radius * radius).is_infinite() {
        return f64::INFINITY;
    }
    radius + (x.abs() + radius) * SEARCH_MARGIN_REL + SEARCH_MARGIN_ABS
}

// ---------------------------------------------------------------------------
// CellCoord — private
// ---------------------------------------------------------------------------

/// Integer cell coordinate in 3D space.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct CellCoord {
    x: i32,
    y: i32,
    z: i32,
}

impl CellCoord {
    /// The cell containing `pos` (see [`cell_index`]).  The exact geometric test
    /// of every query still decides membership.
    #[inline]
    fn from_pos(pos: [f64; 3], inv_cell: f64) -> Self {
        CellCoord {
            x: cell_index(pos[0], inv_cell),
            y: cell_index(pos[1], inv_cell),
            z: cell_index(pos[2], inv_cell),
        }
    }

    /// `true` if `self` comes strictly before `other` in x-y-z (lexicographic)
    /// order.
    #[inline]
    fn precedes(self, other: CellCoord) -> bool {
        (self.x, self.y, self.z) < (other.x, other.y, other.z)
    }

    #[inline]
    fn in_range(self, min: CellCoord, max: CellCoord) -> bool {
        self.x >= min.x
            && self.x <= max.x
            && self.y >= min.y
            && self.y <= max.y
            && self.z >= min.z
            && self.z <= max.z
    }

    /// Chebyshev (maximum-axis) distance between two cells, in cells.
    #[inline]
    fn chebyshev_distance(self, other: CellCoord) -> u32 {
        self.x
            .abs_diff(other.x)
            .max(self.y.abs_diff(other.y))
            .max(self.z.abs_diff(other.z))
    }

    /// The cell at the given offset, or `None` if it would leave the `i32`
    /// coordinate range.
    #[inline]
    fn offset(self, dx: i64, dy: i64, dz: i64) -> Option<CellCoord> {
        Some(CellCoord {
            x: i32::try_from(i64::from(self.x) + dx).ok()?,
            y: i32::try_from(i64::from(self.y) + dy).ok()?,
            z: i32::try_from(i64::from(self.z) + dz).ok()?,
        })
    }
}

/// Number of cells in the inclusive box `[min, max]` (zero if it is empty).
fn cell_box_volume(min: CellCoord, max: CellCoord) -> u128 {
    if min.x > max.x || min.y > max.y || min.z > max.z {
        return 0;
    }
    let span = |lo: i32, hi: i32| u128::from(hi.abs_diff(lo)) + 1;
    span(min.x, max.x) * span(min.y, max.y) * span(min.z, max.z)
}

/// Number of cells in the cube of half-width `reach` cells (`(2·reach + 1)³`).
fn cell_cube_volume(reach: u32) -> u128 {
    let side = 2 * u128::from(reach) + 1;
    side * side * side
}

// ---------------------------------------------------------------------------
// GridEntry — private
// ---------------------------------------------------------------------------

/// One registered entry.  The serialised form is `{id, position, cell}`; the
/// cell is derived data (recomputed from the position when a grid is loaded, so
/// it may be omitted from the input) and `bucket_pos` is a pure runtime index.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GridEntry {
    id: u32,
    position: [f64; 3],
    #[serde(default)]
    cell: CellCoord,
    /// Index of this entry inside `cells[cell]`.
    #[serde(skip)]
    bucket_pos: usize,
}

impl GridEntry {
    /// The bucket representation of this entry, stored at `index` of `entries`.
    #[inline]
    fn item(&self, index: usize) -> BucketItem {
        BucketItem {
            id: self.id,
            // IDs are `u32` and unique, so there are at most `u32::MAX + 1`
            // entries and every index fits.
            index: index as u32,
            position: self.position,
        }
    }
}

/// What a bucket stores per entry: everything a query needs, inline, so that
/// scanning a cell reads one contiguous block of memory instead of following an
/// index into `entries` for every candidate.  (32 bytes: `u32` + `u32` + 3 x
/// `f64`, no padding.)
#[derive(Debug, Clone, Copy)]
struct BucketItem {
    id: u32,
    /// Index of the entry in `SpatialGrid::entries`.
    index: u32,
    position: [f64; 3],
}

/// Call `visit` for every item of `bucket`.
#[inline]
fn visit_bucket<F: FnMut(&BucketItem)>(bucket: &[BucketItem], visit: &mut F) {
    for item in bucket {
        visit(item);
    }
}

// ---------------------------------------------------------------------------
// Nearest-neighbour search helpers — private
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two points.
#[inline]
fn dist_sq(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// A nearest-neighbour candidate, ordered by squared distance and then by ID so
/// that every result is deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    /// Bit pattern of the (non-negative) squared distance; for non-negative
    /// floats the bit pattern orders exactly like the value.
    dist_sq_bits: u64,
    id: u32,
}

impl Candidate {
    fn new(id: u32, dist_sq: f64) -> Self {
        // Non-finite distances (NaN or infinite coordinates) sort last.
        let dist_sq = if dist_sq.is_finite() {
            dist_sq
        } else {
            f64::INFINITY
        };
        Self {
            dist_sq_bits: dist_sq.to_bits(),
            id,
        }
    }

    fn dist_sq(self) -> f64 {
        f64::from_bits(self.dist_sq_bits)
    }
}

/// A search that consumes entries in expanding shells of cells (see
/// [`SpatialGrid::search_rings`]).
trait RingSearch {
    /// Consider one entry (given as its bucket item).
    fn visit(&mut self, item: &BucketItem);

    /// `true` once the answer cannot be improved by an entry whose squared
    /// distance to the query point is at least `unvisited_min_dist_sq`.
    fn satisfied(&self, unvisited_min_dist_sq: f64) -> bool;
}

/// Keeps the single closest entry.
struct NearestSearch {
    pos: [f64; 3],
    best: Option<Candidate>,
}

impl RingSearch for NearestSearch {
    fn visit(&mut self, item: &BucketItem) {
        let candidate = Candidate::new(item.id, dist_sq(item.position, self.pos));
        if self.best.is_none_or(|best| candidate < best) {
            self.best = Some(candidate);
        }
    }

    fn satisfied(&self, unvisited_min_dist_sq: f64) -> bool {
        self.best
            .is_some_and(|best| best.dist_sq() <= unvisited_min_dist_sq)
    }
}

/// Keeps the `k` closest entries in a max-heap (the worst kept entry on top).
struct KNearestSearch {
    pos: [f64; 3],
    k: usize,
    heap: BinaryHeap<Candidate>,
}

impl RingSearch for KNearestSearch {
    fn visit(&mut self, item: &BucketItem) {
        let candidate = Candidate::new(item.id, dist_sq(item.position, self.pos));
        if self.heap.len() < self.k {
            self.heap.push(candidate);
        } else if let Some(&worst) = self.heap.peek()
            && candidate < worst
        {
            self.heap.pop();
            self.heap.push(candidate);
        }
    }

    fn satisfied(&self, unvisited_min_dist_sq: f64) -> bool {
        self.heap.len() >= self.k
            && self
                .heap
                .peek()
                .is_some_and(|worst| worst.dist_sq() <= unvisited_min_dist_sq)
    }
}

// ---------------------------------------------------------------------------
// SpatialGrid
// ---------------------------------------------------------------------------

/// A uniform spatial hash grid for fast neighbourhood queries.
///
/// Entries are keyed by `u32` IDs. Each ID may appear at most once; calling
/// [`insert`][Self::insert] with an existing ID replaces the entry.
///
/// See the [module documentation](self) for the cost of each operation and for
/// advice on choosing the cell size.
#[derive(Debug, Clone)]
pub struct SpatialGrid {
    /// Side length of each cubic cell (m).
    cell_size: f64,
    /// `1 / cell_size`, cached.
    inv_cell: f64,
    /// All entries, densely packed (insertion order).
    entries: Vec<GridEntry>,
    /// ID → index into `entries`.
    slot: FastMap<u32, usize>,
    /// Occupied cells → the items of the entries inside them.  A cell is
    /// present only while it holds at least one entry.
    cells: FastMap<CellCoord, Vec<BucketItem>>,
}

impl SpatialGrid {
    /// Create an empty grid with the given cell size.
    ///
    /// # Panics
    ///
    /// Panics if `cell_size` is not a positive, finite number.  Use
    /// [`try_new`](Self::try_new) to handle that case without panicking.
    pub fn new(cell_size: f64) -> Self {
        match Self::try_new(cell_size) {
            Ok(grid) => grid,
            Err(error) => panic!("{error}"),
        }
    }

    /// Create an empty grid with the given cell size.
    ///
    /// # Errors
    ///
    /// Returns [`GridError::InvalidCellSize`] if `cell_size` is not a positive,
    /// finite number (or so small that its reciprocal overflows).
    pub fn try_new(cell_size: f64) -> Result<Self, GridError> {
        let inv_cell = Self::checked_inverse(cell_size)?;
        Ok(Self {
            cell_size,
            inv_cell,
            entries: Vec::new(),
            slot: FastMap::default(),
            cells: FastMap::default(),
        })
    }

    /// `1 / cell_size`, or an error if the cell size is unusable.
    fn checked_inverse(cell_size: f64) -> Result<f64, GridError> {
        let inv_cell = 1.0 / cell_size;
        if cell_size.is_finite() && cell_size > 0.0 && inv_cell.is_finite() {
            Ok(inv_cell)
        } else {
            Err(GridError::InvalidCellSize(cell_size))
        }
    }

    /// Side length of each cubic cell (m).
    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// Change the cell size and re-bucket every entry accordingly, in O(N).
    ///
    /// # Errors
    ///
    /// Returns [`GridError::InvalidCellSize`] (and leaves the grid untouched) if
    /// `cell_size` is not a positive, finite number.
    pub fn set_cell_size(&mut self, cell_size: f64) -> Result<(), GridError> {
        let inv_cell = Self::checked_inverse(cell_size)?;
        self.cell_size = cell_size;
        self.inv_cell = inv_cell;
        self.rebuild_index();
        Ok(())
    }

    /// Recompute every entry's cell and both lookup tables from the dense
    /// `entries` array.
    fn rebuild_index(&mut self) {
        self.slot.clear();
        self.cells.clear();
        for (index, entry) in self.entries.iter_mut().enumerate() {
            entry.cell = CellCoord::from_pos(entry.position, self.inv_cell);
            let bucket = self.cells.entry(entry.cell).or_default();
            entry.bucket_pos = bucket.len();
            bucket.push(entry.item(index));
            self.slot.insert(entry.id, index);
        }
    }

    // -----------------------------------------------------------------------
    // Mutation
    // -----------------------------------------------------------------------

    /// Insert or replace an entry.
    ///
    /// If `id` already exists, its position is updated in place.
    pub fn insert(&mut self, id: u32, position: [f64; 3]) {
        let cell = CellCoord::from_pos(position, self.inv_cell);
        if let Some(&index) = self.slot.get(&id) {
            self.relocate(index, position, cell);
            return;
        }
        let index = self.entries.len();
        let mut entry = GridEntry {
            id,
            position,
            cell,
            bucket_pos: 0,
        };
        let bucket = self.cells.entry(cell).or_default();
        entry.bucket_pos = bucket.len();
        bucket.push(entry.item(index));
        self.entries.push(entry);
        self.slot.insert(id, index);
    }

    /// Update the position of an existing entry.  No-op if `id` is not found.
    pub fn update(&mut self, id: u32, position: [f64; 3]) {
        if let Some(&index) = self.slot.get(&id) {
            let cell = CellCoord::from_pos(position, self.inv_cell);
            self.relocate(index, position, cell);
        }
    }

    /// Remove an entry by ID.  No-op if `id` is not found.
    pub fn remove(&mut self, id: u32) {
        let Some(index) = self.slot.remove(&id) else {
            return;
        };
        self.detach_from_bucket(index);
        if index >= self.entries.len() {
            return;
        }
        // The last entry moves into the hole; fix up everything that points at it.
        self.entries.swap_remove(index);
        if let Some(moved) = self.entries.get(index) {
            let (moved_id, moved_cell, moved_bucket_pos) = (moved.id, moved.cell, moved.bucket_pos);
            self.slot.insert(moved_id, index);
            if let Some(item) = self
                .cells
                .get_mut(&moved_cell)
                .and_then(|bucket| bucket.get_mut(moved_bucket_pos))
            {
                item.index = index as u32;
            }
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.slot.clear();
        self.cells.clear();
    }

    /// Move the entry at `index` to `position` (inside `cell`), keeping the
    /// buckets consistent.
    fn relocate(&mut self, index: usize, position: [f64; 3], cell: CellCoord) {
        let Some((old_cell, bucket_pos)) = self
            .entries
            .get(index)
            .map(|entry| (entry.cell, entry.bucket_pos))
        else {
            return;
        };
        if old_cell == cell {
            // Same cell: only the stored positions change.
            if let Some(item) = self
                .cells
                .get_mut(&cell)
                .and_then(|bucket| bucket.get_mut(bucket_pos))
            {
                item.position = position;
            }
            if let Some(entry) = self.entries.get_mut(index) {
                entry.position = position;
            }
            return;
        }

        self.detach_from_bucket(index);
        let Some(entry) = self.entries.get_mut(index) else {
            return;
        };
        entry.position = position;
        entry.cell = cell;
        let bucket = self.cells.entry(cell).or_default();
        entry.bucket_pos = bucket.len();
        bucket.push(entry.item(index));
    }

    /// Remove the entry at `index` from the bucket of its cell (the entry itself
    /// stays in `entries`; the caller re-attaches or deletes it).
    fn detach_from_bucket(&mut self, index: usize) {
        let Some((cell, bucket_pos)) = self
            .entries
            .get(index)
            .map(|entry| (entry.cell, entry.bucket_pos))
        else {
            return;
        };
        let mut moved = None;
        let mut emptied = false;
        if let Some(bucket) = self.cells.get_mut(&cell)
            && bucket_pos < bucket.len()
        {
            bucket.swap_remove(bucket_pos);
            // The bucket's last item now sits at `bucket_pos`.
            moved = bucket.get(bucket_pos).map(|item| item.index as usize);
            emptied = bucket.is_empty();
        }
        if emptied {
            self.cells.remove(&cell);
        }
        if let Some(entry) = moved.and_then(|moved_index| self.entries.get_mut(moved_index)) {
            entry.bucket_pos = bucket_pos;
        }
    }

    // -----------------------------------------------------------------------
    // Single-entry queries
    // -----------------------------------------------------------------------

    /// Return `true` if `id` is registered in the grid.
    pub fn contains(&self, id: u32) -> bool {
        self.slot.contains_key(&id)
    }

    /// Return the current position of `id`, or `None` if not registered.
    pub fn position(&self, id: u32) -> Option<[f64; 3]> {
        self.slot
            .get(&id)
            .and_then(|&index| self.entries.get(index))
            .map(|entry| entry.position)
    }

    // -----------------------------------------------------------------------
    // Neighbourhood queries
    // -----------------------------------------------------------------------

    /// Return all IDs whose position is within `radius` of `center`
    /// (boundary inclusive), sorted by ID.
    ///
    /// A negative or NaN `radius` matches nothing.
    pub fn query_radius(&self, center: [f64; 3], radius: f64) -> Vec<u32> {
        if radius.is_nan() || radius < 0.0 {
            return Vec::new();
        }
        let (min_cell, max_cell) = self.cell_span(center, radius);
        let r2 = radius * radius;
        let mut hits = Vec::new();
        self.for_each_in_cell_box(min_cell, max_cell, |entry| {
            if dist_sq(entry.position, center) <= r2 {
                hits.push(entry.id);
            }
        });
        hits.sort_unstable();
        hits
    }

    /// Return all IDs whose position lies inside the AABB `[min, max]`
    /// (boundary inclusive), sorted by ID.
    pub fn query_aabb(&self, min: [f64; 3], max: [f64; 3]) -> Vec<u32> {
        let min_cell = CellCoord::from_pos(min, self.inv_cell);
        let max_cell = CellCoord::from_pos(max, self.inv_cell);
        let mut hits = Vec::new();
        self.for_each_in_cell_box(min_cell, max_cell, |entry| {
            let p = entry.position;
            if p[0] >= min[0]
                && p[0] <= max[0]
                && p[1] >= min[1]
                && p[1] <= max[1]
                && p[2] >= min[2]
                && p[2] <= max[2]
            {
                hits.push(entry.id);
            }
        });
        hits.sort_unstable();
        hits
    }

    /// Return the ID and distance of the nearest entry to `pos`.
    ///
    /// Ties are broken by the smaller ID.  Returns `None` if the grid is empty.
    pub fn nearest(&self, pos: [f64; 3]) -> Option<(u32, f64)> {
        let mut search = NearestSearch { pos, best: None };
        self.search_rings(pos, &mut search);
        search.best.map(|best| (best.id, best.dist_sq().sqrt()))
    }

    /// Return the `k` nearest entries to `pos`, sorted by ascending distance
    /// (ties by ascending ID).
    ///
    /// Returns fewer than `k` entries if the grid has fewer.
    pub fn k_nearest(&self, pos: [f64; 3], k: usize) -> Vec<(u32, f64)> {
        if k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let mut search = KNearestSearch {
            pos,
            k,
            heap: BinaryHeap::with_capacity(k.min(self.entries.len()) + 1),
        };
        if k >= self.entries.len() {
            // Every entry qualifies: no point in searching shell by shell.
            for (index, entry) in self.entries.iter().enumerate() {
                search.visit(&entry.item(index));
            }
        } else {
            self.search_rings(pos, &mut search);
        }
        search
            .heap
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| (candidate.id, candidate.dist_sq().sqrt()))
            .collect()
    }

    /// Return all pairs `(a, b)` whose positions are within `radius` of each
    /// other (boundary inclusive).
    ///
    /// Each pair is returned once with `a < b`, and the pairs are sorted.
    /// Useful for broad-phase collision detection; for the best performance use
    /// a cell size close to `radius`.  A negative or NaN `radius` matches
    /// nothing.
    pub fn pairs_within_radius(&self, radius: f64) -> Vec<(u32, u32)> {
        if radius.is_nan() || radius < 0.0 {
            return Vec::new();
        }
        let r2 = radius * radius;
        let mut pairs = Vec::new();
        self.for_each_candidate_pair(radius, |a, b| {
            if dist_sq(a.position, b.position) <= r2 {
                pairs.push(if a.id < b.id {
                    (a.id, b.id)
                } else {
                    (b.id, a.id)
                });
            }
        });
        pairs.sort_unstable();
        pairs
    }

    // -----------------------------------------------------------------------
    // Metadata
    // -----------------------------------------------------------------------

    /// Number of entries in the grid.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the grid contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over `(id, position)` pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, [f64; 3])> + '_ {
        self.entries.iter().map(|e| (e.id, e.position))
    }

    /// Return the IDs (sorted) of all entries whose cell coordinate matches
    /// exactly.
    ///
    /// Useful for debugging grid partitioning.
    pub fn entries_in_cell(&self, cell_x: i32, cell_y: i32, cell_z: i32) -> Vec<u32> {
        let target = CellCoord {
            x: cell_x,
            y: cell_y,
            z: cell_z,
        };
        let mut ids: Vec<u32> = self
            .cells
            .get(&target)
            .into_iter()
            .flatten()
            .map(|item| item.id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Return the cell coordinates for a world-space position.
    pub fn cell_for(&self, pos: [f64; 3]) -> (i32, i32, i32) {
        let c = CellCoord::from_pos(pos, self.inv_cell);
        (c.x, c.y, c.z)
    }

    // -----------------------------------------------------------------------
    // Cell traversal — private
    // -----------------------------------------------------------------------

    /// Visit every entry in the cells of the inclusive cell box `[min, max]`.
    ///
    /// Only the cells of the box are looked up -- unless the box contains more
    /// cells than the grid has occupied cells, in which case the occupied cells
    /// are scanned instead.  Either way the work is bounded by the number of
    /// occupied cells plus the visited entries.
    fn for_each_in_cell_box<F: FnMut(&BucketItem)>(
        &self,
        min: CellCoord,
        max: CellCoord,
        mut visit: F,
    ) {
        let volume = cell_box_volume(min, max);
        if volume == 0 {
            return;
        }
        if volume > self.cells.len() as u128 {
            for (cell, bucket) in &self.cells {
                if cell.in_range(min, max) {
                    visit_bucket(bucket, &mut visit);
                }
            }
            return;
        }
        for x in min.x..=max.x {
            for y in min.y..=max.y {
                for z in min.z..=max.z {
                    if let Some(bucket) = self.cells.get(&CellCoord { x, y, z }) {
                        visit_bucket(bucket, &mut visit);
                    }
                }
            }
        }
    }

    /// The inclusive range of cell indices along one axis that can hold a point
    /// whose coordinate on that axis is within `radius` of `x` (widened by a
    /// rounding margin, see [`inflated_radius`]).
    fn axis_span(&self, x: f64, radius: f64) -> (i32, i32) {
        let reach = inflated_radius(x, radius);
        if reach == f64::INFINITY {
            // Unbounded search, or a radius whose square overflows: every point
            // that is not NaN passes the distance test.
            return (i32::MIN, i32::MAX);
        }
        (
            cell_index(x - reach, self.inv_cell),
            cell_index(x + reach, self.inv_cell),
        )
    }

    /// The inclusive box of cells (`min`, `max` corners) that can hold a point
    /// within `radius` of `center`, rounding margin included.
    fn cell_span(&self, center: [f64; 3], radius: f64) -> (CellCoord, CellCoord) {
        let (x_min, x_max) = self.axis_span(center[0], radius);
        let (y_min, y_max) = self.axis_span(center[1], radius);
        let (z_min, z_max) = self.axis_span(center[2], radius);
        (
            CellCoord {
                x: x_min,
                y: y_min,
                z: z_min,
            },
            CellCoord {
                x: x_max,
                y: y_max,
                z: z_max,
            },
        )
    }

    /// The smallest box of cells that contains the [`cell_span`](Self::cell_span)
    /// of every item of `bucket`.
    fn bucket_span(&self, bucket: &[BucketItem], radius: f64) -> (CellCoord, CellCoord) {
        let mut min = CellCoord {
            x: i32::MAX,
            y: i32::MAX,
            z: i32::MAX,
        };
        let mut max = CellCoord {
            x: i32::MIN,
            y: i32::MIN,
            z: i32::MIN,
        };
        for item in bucket {
            let (low, high) = self.cell_span(item.position, radius);
            min = CellCoord {
                x: min.x.min(low.x),
                y: min.y.min(low.y),
                z: min.z.min(low.z),
            };
            max = CellCoord {
                x: max.x.max(high.x),
                y: max.y.max(high.y),
                z: max.z.max(high.z),
            };
        }
        (min, max)
    }

    /// Call `visit(a, b)` once for every unordered pair of entries that can lie
    /// within `radius` of each other: every pair that passes the distance test
    /// of [`pairs_within_radius`](Self::pairs_within_radius) is among them,
    /// whichever way the floating-point roundings fall.
    ///
    /// These are the pairs inside one cell, plus the pairs between a cell and
    /// the cells that the search span of one of its entries touches (see
    /// [`cell_span`](Self::cell_span)).  A pair of cells is only enumerated from
    /// the lexicographically smaller one, so no pair is visited twice.  If the
    /// span of a cell holds more cells than the grid has occupied cells (a cell
    /// size far below the radius), the occupied cells are scanned instead of
    /// looking up the cells of the span one by one.
    fn for_each_candidate_pair<F: FnMut(&BucketItem, &BucketItem)>(
        &self,
        radius: f64,
        mut visit: F,
    ) {
        let occupied = self.cells.len() as u128;
        for (&cell, bucket) in &self.cells {
            // Pairs inside the cell.
            for (n, a) in bucket.iter().enumerate() {
                for b in bucket.iter().skip(n + 1) {
                    visit(a, b);
                }
            }

            // Pairs with the cells ahead of this one that its entries can reach.
            // Every entry's own cell is part of its span, so the span of the
            // bucket starts at or before `cell` and ends at or after it.
            let (min, max) = self.bucket_span(bucket, radius);
            let mut visit_cell = |other: &[BucketItem]| {
                for a in bucket {
                    for b in other {
                        visit(a, b);
                    }
                }
            };
            if cell_box_volume(min, max) > occupied {
                for (&other_cell, other) in &self.cells {
                    if cell.precedes(other_cell) && other_cell.in_range(min, max) {
                        visit_cell(other);
                    }
                }
                continue;
            }
            for x in cell.x..=max.x {
                // Cells of the same x (and y) that come before `cell` belong to
                // the cells behind it: they are visited from there.
                let y_start = if x == cell.x { cell.y } else { min.y };
                for y in y_start..=max.y {
                    let z_start = if x == cell.x && y == cell.y {
                        match cell.z.checked_add(1) {
                            Some(z) => z,
                            None => continue,
                        }
                    } else {
                        min.z
                    };
                    for z in z_start..=max.z {
                        if let Some(other) = self.cells.get(&CellCoord { x, y, z }) {
                            visit_cell(other);
                        }
                    }
                }
            }
        }
    }

    /// Feed entries to `search` in shells of growing Chebyshev radius around the
    /// cell of `pos` until it is [`satisfied`](RingSearch::satisfied).
    ///
    /// After the shells `0..=ring` have been visited, every other entry lies in
    /// a cell that is more than `ring` cells away along some axis, so it is
    /// farther than `ring · cell_size` from `pos`.  Once enumerating shells
    /// would touch more cells than the grid has occupied cells, the remaining
    /// occupied cells are scanned directly, which ends the search.
    fn search_rings<S: RingSearch>(&self, pos: [f64; 3], search: &mut S) {
        if self.entries.is_empty() {
            return;
        }
        let origin = CellCoord::from_pos(pos, self.inv_cell);
        let occupied = self.cells.len() as u128;
        let mut ring: u32 = 0;
        loop {
            if cell_cube_volume(ring) > occupied {
                for (cell, bucket) in &self.cells {
                    if cell.chebyshev_distance(origin) >= ring {
                        visit_bucket(bucket, &mut |item| search.visit(item));
                    }
                }
                return;
            }
            self.visit_shell(origin, ring, search);

            // A hair below the exact bound, to stay clear of rounding in the
            // position-to-cell mapping.
            let guaranteed = f64::from(ring) * self.cell_size * (1.0 - 1e-9);
            if search.satisfied(guaranteed * guaranteed) {
                return;
            }
            ring += 1;
        }
    }

    /// Visit the cells at Chebyshev distance exactly `ring` from `origin`.
    fn visit_shell<S: RingSearch>(&self, origin: CellCoord, ring: u32, search: &mut S) {
        let r = i64::from(ring);
        let mut visit_cell = |dx: i64, dy: i64, dz: i64| {
            if let Some(bucket) = origin
                .offset(dx, dy, dz)
                .and_then(|cell| self.cells.get(&cell))
            {
                visit_bucket(bucket, &mut |item| search.visit(item));
            }
        };
        for dx in -r..=r {
            for dy in -r..=r {
                if dx.abs() == r || dy.abs() == r {
                    // On the shell's side faces the whole z column belongs to it.
                    for dz in -r..=r {
                        visit_cell(dx, dy, dz);
                    }
                } else {
                    // Inside the faces only the two caps do.
                    visit_cell(dx, dy, -r);
                    visit_cell(dx, dy, r);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------
//
// The wire format is `{ "cell_size": f64, "entries": [{ "id", "position",
// "cell" }, ...] }`.  The lookup tables are runtime data: they are rebuilt on
// load, and the stored `cell` is recomputed from the position and the cell size
// instead of being trusted, so a stale or hand-edited cell can never corrupt
// the index.

impl Serialize for SpatialGrid {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("SpatialGrid", 2)?;
        state.serialize_field("cell_size", &self.cell_size)?;
        state.serialize_field("entries", &self.entries)?;
        state.end()
    }
}

/// The serialised shape of a [`SpatialGrid`].
#[derive(Deserialize)]
struct SerializedGrid {
    cell_size: f64,
    entries: Vec<GridEntry>,
}

impl<'de> Deserialize<'de> for SpatialGrid {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let data = SerializedGrid::deserialize(deserializer)?;
        let mut grid = SpatialGrid::try_new(data.cell_size).map_err(D::Error::custom)?;
        grid.entries.reserve(data.entries.len());
        for entry in data.entries {
            // Duplicate IDs: the last one wins, exactly like repeated `insert`s.
            grid.insert(entry.id, entry.position);
        }
        Ok(grid)
    }
}

#[cfg(test)]
mod tests;
