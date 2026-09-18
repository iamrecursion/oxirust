// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Volume rendering data structures: voxel grids, transfer functions,
//! ray-casting, isosurface extraction, and per-grid statistics.
//!
//! All positions and scalar values use `f64` for precision.
//! No nalgebra dependency is introduced here; vectors are `[f64; 3]`.

// ---------------------------------------------------------------------------
// VoxelGrid
// ---------------------------------------------------------------------------

/// A regular 3-D scalar grid.
///
/// `dimensions` is `[nx, ny, nz]`.  `spacing` is the cell size in each
/// dimension.  `data` is stored in C order: index = `iz * ny * nx + iy * nx + ix`.
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    /// Number of cells along each axis: `[nx, ny, nz]`.
    pub dimensions: [usize; 3],
    /// Physical spacing between adjacent cell centres along each axis.
    pub spacing: [f64; 3],
    /// Flattened scalar data (length must equal `nx * ny * nz`).
    pub data: Vec<f64>,
}

impl VoxelGrid {
    /// Construct a new `VoxelGrid` filled with zeros.
    ///
    /// `dimensions` — `[nx, ny, nz]`.
    /// `spacing` — physical cell size in each direction.
    pub fn new(dimensions: [usize; 3], spacing: [f64; 3]) -> Self {
        let n = dimensions[0] * dimensions[1] * dimensions[2];
        Self {
            dimensions,
            spacing,
            data: vec![0.0; n],
        }
    }

    /// Construct a grid from existing data.
    ///
    /// Panics if `data.len() != nx * ny * nz`.
    pub fn from_data(dimensions: [usize; 3], spacing: [f64; 3], data: Vec<f64>) -> Self {
        let n = dimensions[0] * dimensions[1] * dimensions[2];
        assert_eq!(data.len(), n, "data length mismatch");
        Self {
            dimensions,
            spacing,
            data,
        }
    }

    /// Return the scalar value at integer voxel coordinates `(ix, iy, iz)`.
    ///
    /// Returns `0.0` if any coordinate is out of bounds.
    pub fn get(&self, ix: usize, iy: usize, iz: usize) -> f64 {
        let [nx, ny, nz] = self.dimensions;
        if ix >= nx || iy >= ny || iz >= nz {
            return 0.0;
        }
        self.data[iz * ny * nx + iy * nx + ix]
    }

    /// Set the scalar value at integer voxel coordinates `(ix, iy, iz)`.
    ///
    /// Does nothing if any coordinate is out of bounds.
    pub fn set(&mut self, ix: usize, iy: usize, iz: usize, value: f64) {
        let [nx, ny, nz] = self.dimensions;
        if ix >= nx || iy >= ny || iz >= nz {
            return;
        }
        self.data[iz * ny * nx + iy * nx + ix] = value;
    }

    /// Sample the grid at an arbitrary physical position using **trilinear interpolation**.
    ///
    /// The grid origin is at `(0, 0, 0)` in physical space.  Returns `0.0`
    /// for positions outside the grid extent, for a NaN position (or one that
    /// the spacing maps to NaN) and for an empty grid (a zero dimension).
    pub fn sample(&self, pos: [f64; 3]) -> f64 {
        let [nx, ny, nz] = self.dimensions;
        if nx == 0 || ny == 0 || nz == 0 {
            return 0.0;
        }
        let [sx, sy, sz] = self.spacing;

        // Convert to fractional grid coordinates
        let fx = pos[0] / sx;
        let fy = pos[1] / sy;
        let fz = pos[2] / sz;

        // Valid range is `0..=n - 1` along each axis; written as a positive
        // test so that NaN coordinates are rejected as well.
        let inside = |f: f64, n: usize| f >= 0.0 && f <= (n - 1) as f64;
        if !(inside(fx, nx) && inside(fy, ny) && inside(fz, nz)) {
            return 0.0;
        }

        let x0 = (fx.floor() as usize).min(nx - 1);
        let y0 = (fy.floor() as usize).min(ny - 1);
        let z0 = (fz.floor() as usize).min(nz - 1);
        let x1 = (x0 + 1).min(nx - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let z1 = (z0 + 1).min(nz - 1);

        let tx = fx - x0 as f64;
        let ty = fy - y0 as f64;
        let tz = fz - z0 as f64;

        // Trilinear interpolation
        let c000 = self.get(x0, y0, z0);
        let c100 = self.get(x1, y0, z0);
        let c010 = self.get(x0, y1, z0);
        let c110 = self.get(x1, y1, z0);
        let c001 = self.get(x0, y0, z1);
        let c101 = self.get(x1, y0, z1);
        let c011 = self.get(x0, y1, z1);
        let c111 = self.get(x1, y1, z1);

        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }

    /// Return the physical extent `[width, height, depth]` of the grid.
    ///
    /// Nodes sit at `index * spacing`, so an axis with `n` cells spans
    /// `(n - 1) * spacing`; an axis with fewer than two cells (including an
    /// empty one) has zero extent.
    pub fn extent(&self) -> [f64; 3] {
        std::array::from_fn(|axis| {
            self.dimensions[axis].saturating_sub(1) as f64 * self.spacing[axis]
        })
    }
}

// ---------------------------------------------------------------------------
// TransferFunction
// ---------------------------------------------------------------------------

/// A piece-wise linear transfer function mapping scalar values to opacity + colour.
///
/// Control points are `(scalar_value, opacity, [r, g, b])`.  Values between
/// control points are linearly interpolated.  Outside the range, the nearest
/// endpoint is used.
#[derive(Debug, Clone)]
pub struct TransferFunction {
    /// Sorted list of `(scalar, opacity, [r,g,b])` control points.
    ///
    /// Must be sorted by scalar value in ascending order.
    pub points: Vec<(f64, f64, [f64; 3])>,
}

impl TransferFunction {
    /// Create a new, empty transfer function.
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add a control point `(scalar, opacity, colour)`.
    ///
    /// The point list is kept sorted by scalar value after insertion.
    pub fn add_point(&mut self, scalar: f64, opacity: f64, color: [f64; 3]) {
        self.points.push((scalar, opacity, color));
        self.points
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Evaluate the transfer function at `value`.
    ///
    /// Returns `(opacity, [r, g, b])`.  Returns `(0.0, [0.0; 3])` if there
    /// are no control points, or if `value` is NaN (a missing sample is
    /// fully transparent).  Infinite values clamp to the end points.
    pub fn eval(&self, value: f64) -> (f64, [f64; 3]) {
        if value.is_nan() || self.points.is_empty() {
            return (0.0, [0.0; 3]);
        }
        if self.points.len() == 1 {
            let (_, op, col) = self.points[0];
            return (op, col);
        }
        // Below first point
        if value <= self.points[0].0 {
            let (_, op, col) = self.points[0];
            return (op, col);
        }
        // Above last point
        let last = self.points.len() - 1;
        if value >= self.points[last].0 {
            let (_, op, col) = self.points[last];
            return (op, col);
        }
        // Find the bracketing interval `[upper - 1, upper]`.  `value` lies
        // strictly inside the range here, so the search result is already in
        // `1..=last`; the clamp only keeps unsorted points from indexing out
        // of range or underflowing.
        let upper = self
            .points
            .partition_point(|(s, _, _)| *s <= value)
            .clamp(1, last);
        let (s0, op0, col0) = self.points[upper - 1];
        let (s1, op1, col1) = self.points[upper];
        let t = (value - s0) / (s1 - s0);
        let opacity = op0 + t * (op1 - op0);
        let color = [
            col0[0] + t * (col1[0] - col0[0]),
            col0[1] + t * (col1[1] - col0[1]),
            col0[2] + t * (col1[2] - col0[2]),
        ];
        (opacity, color)
    }
}

impl Default for TransferFunction {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RaycastVolume
// ---------------------------------------------------------------------------

/// Ray-casting volume renderer.
///
/// Integrates opacity and colour along a ray through the [`VoxelGrid`]
/// using front-to-back alpha compositing.
#[derive(Debug, Clone)]
pub struct RaycastVolume {
    /// The voxel grid to render.
    pub voxel_grid: VoxelGrid,
    /// Transfer function mapping scalar values to opacity+colour.
    pub transfer_fn: TransferFunction,
    /// Step size along the ray in physical units.
    pub step_size: f64,
    /// Maximum number of integration steps.
    pub max_steps: usize,
}

impl RaycastVolume {
    /// Construct a new `RaycastVolume`.
    pub fn new(
        voxel_grid: VoxelGrid,
        transfer_fn: TransferFunction,
        step_size: f64,
        max_steps: usize,
    ) -> Self {
        Self {
            voxel_grid,
            transfer_fn,
            step_size,
            max_steps,
        }
    }

    /// Trace a ray from `origin` in direction `dir` and return an RGBA colour.
    ///
    /// `dir` does not need to be normalised; it is normalised internally.
    /// The result `[r, g, b, a]` has components in `[0, 1]`.
    pub fn render_ray(&self, origin: [f64; 3], dir: [f64; 3]) -> [f64; 4] {
        // Normalise direction
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if len < 1e-12 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        let d = [dir[0] / len, dir[1] / len, dir[2] / len];

        let mut pos = origin;
        let mut accumulated_r = 0.0_f64;
        let mut accumulated_g = 0.0_f64;
        let mut accumulated_b = 0.0_f64;
        let mut accumulated_a = 0.0_f64;

        for _ in 0..self.max_steps {
            let scalar = self.voxel_grid.sample(pos);
            let (raw_opacity, color) = self.transfer_fn.eval(scalar);

            // Scale opacity by step size for consistent density
            let opacity = 1.0 - (-raw_opacity * self.step_size).exp();

            // Front-to-back alpha compositing
            let weight = opacity * (1.0 - accumulated_a);
            accumulated_r += color[0] * weight;
            accumulated_g += color[1] * weight;
            accumulated_b += color[2] * weight;
            accumulated_a += weight;

            // Early termination
            if accumulated_a >= 0.99 {
                break;
            }

            pos[0] += d[0] * self.step_size;
            pos[1] += d[1] * self.step_size;
            pos[2] += d[2] * self.step_size;
        }

        [
            accumulated_r.clamp(0.0, 1.0),
            accumulated_g.clamp(0.0, 1.0),
            accumulated_b.clamp(0.0, 1.0),
            accumulated_a.clamp(0.0, 1.0),
        ]
    }
}

// ---------------------------------------------------------------------------
// IsosurfaceExtractor (marching cubes)
// ---------------------------------------------------------------------------

/// Marching-cubes isosurface extractor.
///
/// Every cell is triangulated with the complete 256-case Lorensen & Cline
/// tables shared with [`crate::isosurface`]; the crossing point on each
/// crossed cell edge is computed by linear interpolation.
pub struct IsosurfaceExtractor;

impl IsosurfaceExtractor {
    /// Extract an isosurface from `grid` at `isovalue`.
    ///
    /// Returns a list of triangle vertices (groups of three consecutive entries
    /// form one triangle).  The grid is node-sampled: node `(ix, iy, iz)` sits
    /// at `(ix, iy, iz) * spacing`, so no half-cell offset is applied, and the
    /// position of each vertex is computed by linear interpolation along cube
    /// edges.  The triangles of neighbouring cells share their edge vertices
    /// exactly, so a closed field yields a closed, consistently oriented mesh.
    /// Cells with a non-finite corner value are skipped.
    pub fn extract_isosurface(grid: &VoxelGrid, isovalue: f64) -> Vec<[f64; 3]> {
        crate::isosurface::marching_cubes_sampled(
            grid.dimensions,
            grid.spacing,
            |ix, iy, iz| grid.get(ix, iy, iz),
            isovalue,
        )
        .into_iter()
        .flat_map(|triangle| triangle.vertices)
        .collect()
    }
}

// ---------------------------------------------------------------------------
// VolumeStats
// ---------------------------------------------------------------------------

/// Per-grid scalar statistics including min, max, mean, and a histogram.
#[derive(Debug, Clone)]
pub struct VolumeStats {
    /// Minimum scalar value in the grid.
    pub min: f64,
    /// Maximum scalar value in the grid.
    pub max: f64,
    /// Mean scalar value.
    pub mean: f64,
    /// Histogram bin counts.  `bins[i]` is the number of voxels in the i-th bin.
    pub bins: Vec<u64>,
}

impl VolumeStats {
    /// Compute statistics for a [`VoxelGrid`] with the given number of histogram bins.
    ///
    /// `num_bins` must be at least 1.
    pub fn compute(grid: &VoxelGrid, num_bins: usize) -> Self {
        let num_bins = num_bins.max(1);
        let data = &grid.data;
        if data.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                bins: vec![0; num_bins],
            };
        }

        let mut min = data[0];
        let mut max = data[0];
        let mut sum = 0.0_f64;
        for &v in data {
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
        }
        let mean = sum / data.len() as f64;

        let mut bins = vec![0u64; num_bins];
        let range = max - min;
        for &v in data {
            let idx = if range < 1e-30 {
                0
            } else {
                let t = (v - min) / range;
                ((t * num_bins as f64) as usize).min(num_bins - 1)
            };
            bins[idx] += 1;
        }

        Self {
            min,
            max,
            mean,
            bins,
        }
    }

    /// Return the total number of voxels represented by the histogram.
    pub fn total_count(&self) -> u64 {
        self.bins.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- VoxelGrid ---

    #[test]
    fn voxel_grid_new_is_zero() {
        let g = VoxelGrid::new([4, 4, 4], [1.0, 1.0, 1.0]);
        assert!(g.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn voxel_grid_set_and_get() {
        let mut g = VoxelGrid::new([4, 4, 4], [1.0, 1.0, 1.0]);
        g.set(1, 2, 3, 42.0);
        assert_eq!(g.get(1, 2, 3), 42.0);
    }

    #[test]
    fn voxel_grid_get_out_of_bounds_returns_zero() {
        let g = VoxelGrid::new([4, 4, 4], [1.0, 1.0, 1.0]);
        assert_eq!(g.get(10, 0, 0), 0.0);
    }

    #[test]
    fn voxel_grid_sample_at_corner() {
        let mut g = VoxelGrid::new([4, 4, 4], [1.0, 1.0, 1.0]);
        g.set(0, 0, 0, 5.0);
        // Exactly at the origin corner
        let v = g.sample([0.0, 0.0, 0.0]);
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn voxel_grid_sample_outside_returns_zero() {
        let g = VoxelGrid::new([4, 4, 4], [1.0, 1.0, 1.0]);
        assert_eq!(g.sample([-1.0, 0.0, 0.0]), 0.0);
        assert_eq!(g.sample([100.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn voxel_grid_with_a_zero_dimension_is_empty_and_safe() {
        // `(n - 1)` was evaluated on a `usize` for `n = 0` in `sample` and `extent`.
        let spacing = [1.0, 2.0, 0.5];
        // Outside the grid the sample is 0, which this function maps to no opacity.
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 0.0, [0.0, 0.0, 0.0]);
        tf.add_point(1.0, 5.0, [1.0, 1.0, 1.0]);
        for dims in [[0, 4, 4], [4, 0, 4], [4, 4, 0], [0, 0, 0]] {
            let g = VoxelGrid::new(dims, spacing);
            assert!(g.data.is_empty(), "dims {dims:?}");
            assert_eq!(g.sample([0.0, 0.0, 0.0]), 0.0, "dims {dims:?}");
            assert_eq!(g.sample([1.5, 1.0, 0.25]), 0.0, "dims {dims:?}");
            assert_eq!(g.get(0, 0, 0), 0.0, "dims {dims:?}");
            let expected: [f64; 3] =
                std::array::from_fn(|a| dims[a].saturating_sub(1) as f64 * spacing[a]);
            assert_eq!(g.extent(), expected, "dims {dims:?}");
            assert!(IsosurfaceExtractor::extract_isosurface(&g, 0.5).is_empty());
            assert_eq!(VolumeStats::compute(&g, 4).total_count(), 0);
            // An empty grid has nothing to integrate.
            let vol = RaycastVolume::new(g, tf.clone(), 0.5, 16);
            let rgba = vol.render_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            assert!(rgba.iter().all(|c| *c == 0.0), "dims {dims:?}: {rgba:?}");
        }
    }

    #[test]
    fn voxel_grid_single_cell_axes_have_zero_extent() {
        let mut g = VoxelGrid::new([1, 1, 1], [2.0, 2.0, 2.0]);
        g.set(0, 0, 0, 9.0);
        assert_eq!(g.extent(), [0.0; 3]);
        assert_eq!(g.sample([0.0, 0.0, 0.0]), 9.0);
        // Anything beyond the lone node is outside the grid.
        assert_eq!(g.sample([0.1, 0.0, 0.0]), 0.0);
        assert_eq!(g.sample([0.0, -0.1, 0.0]), 0.0);

        let slab = VoxelGrid::new([5, 1, 3], [0.5, 1.0, 2.0]);
        assert_eq!(slab.extent(), [2.0, 0.0, 4.0]);
    }

    #[test]
    fn voxel_grid_sample_rejects_nan_positions() {
        let g = VoxelGrid::from_data([4, 4, 4], [1.0, 1.0, 1.0], vec![3.0; 64]);
        assert_eq!(g.sample([1.5, 1.5, 1.5]), 3.0);
        assert_eq!(g.sample([f64::NAN, 1.5, 1.5]), 0.0);
        assert_eq!(g.sample([1.5, f64::NAN, 1.5]), 0.0);
        assert_eq!(g.sample([1.5, 1.5, f64::NAN]), 0.0);
        // A zero spacing maps the position 0 to 0 / 0 = NaN.
        let flat = VoxelGrid::from_data([4, 4, 4], [0.0, 1.0, 1.0], vec![1.0; 64]);
        assert_eq!(flat.sample([0.0, 1.5, 1.5]), 0.0);
    }

    #[test]
    fn voxel_grid_trilinear_midpoint_uniform() {
        // Grid filled with constant 3.0: any sample should return 3.0
        let g = VoxelGrid::from_data([3, 3, 3], [1.0, 1.0, 1.0], vec![3.0; 27]);
        let v = g.sample([0.5, 0.5, 0.5]);
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn voxel_grid_trilinear_midpoint_gradient() {
        // Linear ramp along x: v(i,j,k) = i
        let dims = [5, 3, 3];
        let mut data = vec![0.0; 5 * 3 * 3];
        for iz in 0..3 {
            for iy in 0..3 {
                for ix in 0..5 {
                    data[iz * 3 * 5 + iy * 5 + ix] = ix as f64;
                }
            }
        }
        let g = VoxelGrid::from_data(dims, [1.0, 1.0, 1.0], data);
        // At x=1.5, y=0, z=0 → interpolated value should be 1.5
        let v = g.sample([1.5, 0.0, 0.0]);
        assert!((v - 1.5).abs() < 1e-10);
    }

    #[test]
    fn voxel_grid_extent() {
        let g = VoxelGrid::new([5, 3, 2], [2.0, 3.0, 4.0]);
        let ext = g.extent();
        assert!((ext[0] - 8.0).abs() < 1e-10);
        assert!((ext[1] - 6.0).abs() < 1e-10);
        assert!((ext[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn voxel_grid_from_data_panics_on_mismatch() {
        let result = std::panic::catch_unwind(|| {
            VoxelGrid::from_data([2, 2, 2], [1.0, 1.0, 1.0], vec![0.0; 5]);
        });
        assert!(result.is_err());
    }

    // --- TransferFunction ---

    #[test]
    fn transfer_fn_empty_returns_zero() {
        let tf = TransferFunction::new();
        let (op, col) = tf.eval(0.5);
        assert_eq!(op, 0.0);
        assert_eq!(col, [0.0; 3]);
    }

    #[test]
    fn transfer_fn_single_point() {
        let mut tf = TransferFunction::new();
        tf.add_point(0.5, 0.8, [1.0, 0.0, 0.0]);
        let (op, col) = tf.eval(0.5);
        assert!((op - 0.8).abs() < 1e-12);
        assert!((col[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn transfer_fn_below_range_clamps_to_first() {
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 0.1, [0.0, 0.0, 1.0]);
        tf.add_point(1.0, 0.9, [1.0, 0.0, 0.0]);
        let (op, _) = tf.eval(-1.0);
        assert!((op - 0.1).abs() < 1e-12);
    }

    #[test]
    fn transfer_fn_above_range_clamps_to_last() {
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 0.1, [0.0, 0.0, 1.0]);
        tf.add_point(1.0, 0.9, [1.0, 0.0, 0.0]);
        let (op, _) = tf.eval(2.0);
        assert!((op - 0.9).abs() < 1e-12);
    }

    #[test]
    fn transfer_fn_interpolates_opacity() {
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 0.0, [0.0, 0.0, 0.0]);
        tf.add_point(1.0, 1.0, [1.0, 1.0, 1.0]);
        let (op, col) = tf.eval(0.5);
        assert!((op - 0.5).abs() < 1e-12);
        assert!((col[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn transfer_fn_interpolates_color() {
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 0.0, [1.0, 0.0, 0.0]);
        tf.add_point(1.0, 0.0, [0.0, 1.0, 0.0]);
        let (_, col) = tf.eval(0.25);
        assert!((col[0] - 0.75).abs() < 1e-10);
        assert!((col[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn transfer_fn_sorted_after_insertion() {
        let mut tf = TransferFunction::new();
        tf.add_point(1.0, 0.9, [1.0, 0.0, 0.0]);
        tf.add_point(0.0, 0.1, [0.0, 0.0, 1.0]);
        // Even though 1.0 was inserted first, the list should be sorted
        assert!(tf.points[0].0 < tf.points[1].0);
    }

    #[test]
    fn transfer_fn_nan_is_transparent_for_two_points() {
        // NaN fails both range guards; the bracketing search then used to
        // compute `0 - 1` on a `usize`.
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 0.1, [0.0, 0.0, 1.0]);
        tf.add_point(1.0, 0.9, [1.0, 0.0, 0.0]);
        assert_eq!(tf.eval(f64::NAN), (0.0, [0.0; 3]));
    }

    #[test]
    fn transfer_fn_nan_is_transparent_for_many_points() {
        let mut tf = TransferFunction::new();
        for (s, op) in [(0.0, 0.0), (0.25, 0.4), (0.5, 0.6), (1.0, 1.0)] {
            tf.add_point(s, op, [op, op, op]);
        }
        assert_eq!(tf.eval(f64::NAN), (0.0, [0.0; 3]));
        // The neighbouring finite values are unaffected.
        let (op, _) = tf.eval(0.375);
        assert!((op - 0.5).abs() < 1e-12, "opacity {op}");
    }

    #[test]
    fn transfer_fn_nan_is_transparent_for_single_point() {
        // A single opaque control point must not make a missing sample opaque.
        let mut tf = TransferFunction::new();
        tf.add_point(0.5, 0.8, [1.0, 0.0, 0.0]);
        assert_eq!(tf.eval(f64::NAN), (0.0, [0.0; 3]));
        let (op, col) = tf.eval(0.5);
        assert!((op - 0.8).abs() < 1e-12);
        assert_eq!(col, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn transfer_fn_nan_is_transparent_for_empty_function() {
        assert_eq!(TransferFunction::new().eval(f64::NAN), (0.0, [0.0; 3]));
    }

    #[test]
    fn transfer_fn_infinite_values_still_clamp_to_the_end_points() {
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 0.1, [0.0, 0.0, 1.0]);
        tf.add_point(1.0, 0.9, [1.0, 0.0, 0.0]);
        assert_eq!(tf.eval(f64::NEG_INFINITY), (0.1, [0.0, 0.0, 1.0]));
        assert_eq!(tf.eval(f64::INFINITY), (0.9, [1.0, 0.0, 0.0]));
    }

    #[test]
    fn raycast_through_nan_samples_is_finite() {
        // A NaN voxel used to panic the transfer function mid-ray.
        let mut data = vec![1.0; 4 * 4 * 4];
        data[0] = f64::NAN;
        let grid = VoxelGrid::from_data([4, 4, 4], [1.0, 1.0, 1.0], data);
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 0.0, [0.0, 0.0, 0.0]);
        tf.add_point(1.0, 5.0, [1.0, 1.0, 1.0]);
        let vol = RaycastVolume::new(grid, tf, 0.25, 64);
        let rgba = vol.render_ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(rgba.iter().all(|c| c.is_finite()), "rgba {rgba:?}");
        // The NaN samples are skipped as transparent and the ray goes on to
        // accumulate the opaque voxels behind them.
        assert!(rgba[3] > 0.5, "alpha {}", rgba[3]);
    }

    // --- RaycastVolume ---

    #[test]
    fn raycast_zero_opacity_returns_transparent() {
        let grid = VoxelGrid::new([4, 4, 4], [1.0, 1.0, 1.0]);
        let tf = TransferFunction::default(); // no points → opacity 0
        let vol = RaycastVolume::new(grid, tf, 0.1, 50);
        let rgba = vol.render_ray([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]);
        assert!(rgba[3] < 1e-6, "alpha should be ~0 with no opacity");
    }

    #[test]
    fn raycast_opaque_field_accumulates_alpha() {
        // Uniform high-opacity grid
        let grid = VoxelGrid::from_data([4, 4, 4], [1.0, 1.0, 1.0], vec![1.0; 64]);
        let mut tf = TransferFunction::new();
        tf.add_point(0.0, 10.0, [1.0, 0.0, 0.0]);
        tf.add_point(1.0, 10.0, [1.0, 0.0, 0.0]);
        let vol = RaycastVolume::new(grid, tf, 0.1, 200);
        let rgba = vol.render_ray([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
        assert!(
            rgba[3] > 0.5,
            "dense field should accumulate significant alpha"
        );
    }

    #[test]
    fn raycast_zero_dir_returns_transparent() {
        let grid = VoxelGrid::new([4, 4, 4], [1.0, 1.0, 1.0]);
        let tf = TransferFunction::default();
        let vol = RaycastVolume::new(grid, tf, 0.1, 50);
        let rgba = vol.render_ray([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert_eq!(rgba, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn raycast_output_clamped_to_0_1() {
        let grid = VoxelGrid::from_data([4, 4, 4], [1.0, 1.0, 1.0], vec![1.0; 64]);
        let mut tf = TransferFunction::new();
        tf.add_point(1.0, 100.0, [2.0, 2.0, 2.0]); // intentionally > 1
        let vol = RaycastVolume::new(grid, tf, 0.1, 100);
        let rgba = vol.render_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        for &v in &rgba {
            assert!(
                (0.0..=1.0).contains(&v),
                "RGBA components must be in [0,1], got {v}"
            );
        }
    }

    #[test]
    fn raycast_step_outside_grid_terminates() {
        // Ray that stays entirely outside the grid — grid samples 0.0 everywhere.
        // Use a TF that maps value 0.0 to opacity 0.0, so alpha stays at 0.
        let grid = VoxelGrid::new([4, 4, 4], [1.0, 1.0, 1.0]);
        let mut tf = TransferFunction::new();
        // Only map high values (1.0) to high opacity; outside-grid (0.0) is fully transparent.
        tf.add_point(0.5, 0.0, [0.0, 0.0, 0.0]);
        tf.add_point(1.0, 1.0, [1.0, 0.0, 0.0]);
        let vol = RaycastVolume::new(grid, tf, 10.0, 10);
        // Start ray far outside the grid bounds [0..3]^3 with spacing 1.0
        let rgba = vol.render_ray([1000.0, 1000.0, 1000.0], [1.0, 0.0, 0.0]);
        assert!(
            rgba[3] < 1e-6,
            "alpha should be 0 when ray is entirely outside grid"
        );
    }

    // --- IsosurfaceExtractor ---

    #[test]
    fn isosurface_empty_grid_no_vertices() {
        let g = VoxelGrid::new([2, 2, 2], [1.0, 1.0, 1.0]);
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.5);
        // All zeros below isovalue → no crossing
        assert!(verts.is_empty());
    }

    #[test]
    fn isosurface_uniform_above_isovalue_no_surface() {
        let g = VoxelGrid::from_data([2, 2, 2], [1.0, 1.0, 1.0], vec![1.0; 8]);
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.5);
        // All above → no crossing
        assert!(verts.is_empty());
    }

    #[test]
    fn isosurface_produces_vertices_on_crossing() {
        // One corner above, rest below → at least one triangle
        let mut g = VoxelGrid::new([2, 2, 2], [1.0, 1.0, 1.0]);
        g.set(1, 1, 1, 1.0); // one corner above 0.5
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.5);
        assert!(!verts.is_empty(), "should generate at least one vertex");
        assert_eq!(verts.len() % 3, 0, "vertex count must be multiple of 3");
    }

    #[test]
    fn isosurface_vertices_lie_on_edges() {
        // Ramp grid: v(i,j,k) = i; isovalue = 0.5 → surface should be at x=0.5
        let dims = [3, 2, 2];
        let mut data = vec![0.0; 3 * 2 * 2];
        for iz in 0..2 {
            for iy in 0..2 {
                for ix in 0..3 {
                    data[iz * 2 * 3 + iy * 3 + ix] = ix as f64;
                }
            }
        }
        let g = VoxelGrid::from_data(dims, [1.0, 1.0, 1.0], data);
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.5);
        assert!(!verts.is_empty());
        // All x-coordinates of extracted vertices should be ~0.5
        for v in &verts {
            assert!(
                (v[0] - 0.5).abs() < 1e-10,
                "vertex x should be 0.5, got {}",
                v[0]
            );
        }
    }

    #[test]
    fn isosurface_single_cell_grid_no_panic() {
        // 1×1×1 grid has no edges to iterate over
        let g = VoxelGrid::from_data([1, 1, 1], [1.0, 1.0, 1.0], vec![1.0]);
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.5);
        assert!(verts.is_empty());
    }

    /// Node-sampled signed distance to a sphere, `|p - centre| - radius`.
    fn sphere_grid(n: usize, spacing: [f64; 3], centre: [f64; 3], radius: f64) -> VoxelGrid {
        let mut g = VoxelGrid::new([n; 3], spacing);
        for iz in 0..n {
            for iy in 0..n {
                for ix in 0..n {
                    let d = [
                        ix as f64 * spacing[0] - centre[0],
                        iy as f64 * spacing[1] - centre[1],
                        iz as f64 * spacing[2] - centre[2],
                    ];
                    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    g.set(ix, iy, iz, dist - radius);
                }
            }
        }
        g
    }

    #[test]
    fn isosurface_plane_split_cell_is_tiled_without_overlap_or_hole() {
        // v = iy, iso 0.5: corners {0,1,4,5} | {2,3,6,7}.  The four crossed
        // edges are 1, 3, 5, 7.  Fanning them in edge-index order covers the
        // triangle z <= x and the triangle x + z >= 1 of the unit square,
        // overlapping in one region and leaving the left corner uncovered.
        let mut data = vec![0.0; 8];
        for iz in 0..2 {
            for iy in 0..2 {
                for ix in 0..2 {
                    data[iz * 4 + iy * 2 + ix] = iy as f64;
                }
            }
        }
        let g = VoxelGrid::from_data([2, 2, 2], [1.0, 1.0, 1.0], data);
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.5);
        assert_eq!(verts.len(), 6, "one planar quad is two triangles");
        for v in &verts {
            assert!((v[1] - 0.5).abs() < 1e-12, "vertex off the plane: {v:?}");
        }
        let triangles: Vec<[[f64; 3]; 3]> =
            verts.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();

        // Barycentric point-in-triangle test in the (x, z) plane.
        let covers = |tri: &[[f64; 3]; 3], x: f64, z: f64| {
            let [a, b, c] = *tri;
            let det = (b[2] - c[2]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[2] - c[2]);
            let l0 = ((b[2] - c[2]) * (x - c[0]) + (c[0] - b[0]) * (z - c[2])) / det;
            let l1 = ((c[2] - a[2]) * (x - c[0]) + (a[0] - c[0]) * (z - c[2])) / det;
            let l2 = 1.0 - l0 - l1;
            l0 > 0.0 && l1 > 0.0 && l2 > 0.0
        };
        let samples = 17;
        let mut checked = 0;
        for i in 0..samples {
            for j in 0..samples {
                let x = (i as f64 + 0.5) / samples as f64;
                let z = (j as f64 + 0.5) / samples as f64;
                if (x - z).abs() < 1e-9 || (x + z - 1.0).abs() < 1e-9 {
                    continue; // on a possible shared diagonal
                }
                let covering = triangles.iter().filter(|t| covers(t, x, z)).count();
                assert_eq!(covering, 1, "point ({x}, {z}) is covered {covering} times");
                checked += 1;
            }
        }
        assert!(checked > 200);
    }

    #[test]
    fn isosurface_sphere_vertices_lie_on_the_sphere() {
        // Radius 3 in a 12^3 grid; no node has |p - c| = 3 exactly because
        // the squared distances to the centre (5.5, 5.5, 5.5) are never
        // integers.
        let spacing = [1.0, 1.0, 1.0];
        let centre = [5.5; 3];
        let g = sphere_grid(12, spacing, centre, 3.0);
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.0);
        assert!(verts.len() > 60 * 3, "only {} vertices", verts.len());
        assert_eq!(verts.len() % 3, 0);
        for v in &verts {
            let d = [v[0] - centre[0], v[1] - centre[1], v[2] - centre[2]];
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!(
                (dist - 3.0).abs() < spacing[0],
                "vertex {v:?} at distance {dist}"
            );
        }
    }

    #[test]
    fn isosurface_sphere_is_closed_after_welding() {
        use std::collections::BTreeMap;
        let g = sphere_grid(12, [1.0; 3], [5.5; 3], 3.0);
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.0);

        // Weld the vertices, then count how many triangles use every edge.
        let mut ids: BTreeMap<[u64; 3], usize> = BTreeMap::new();
        let mut welded: Vec<usize> = Vec::with_capacity(verts.len());
        for v in &verts {
            let next = ids.len();
            let id = *ids.entry(v.map(f64::to_bits)).or_insert(next);
            welded.push(id);
        }
        let mut edge_uses: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for tri in welded.chunks_exact(3) {
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                assert_ne!(a, b, "degenerate triangle {tri:?}");
                *edge_uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        for (edge, uses) in &edge_uses {
            assert_eq!(*uses, 2, "edge {edge:?} is shared by {uses} triangles");
        }
        // Euler characteristic of a sphere: V - E + F = 2.
        let faces = welded.len() / 3;
        assert_eq!(ids.len() + faces, edge_uses.len() + 2);
    }

    #[test]
    fn isosurface_honours_anisotropic_spacing() {
        let spacing = [0.5, 1.0, 2.0];
        let centre = [2.75, 5.5, 11.0];
        let mut g = VoxelGrid::new([12; 3], spacing);
        for iz in 0..12 {
            for iy in 0..12 {
                for ix in 0..12 {
                    // Unit sphere in normalised coordinates: an axis-aligned
                    // ellipsoid with semi-axes (1.5, 3.0, 6.0) in space.
                    let u = [
                        (ix as f64 * spacing[0] - centre[0]) / 0.5,
                        (iy as f64 * spacing[1] - centre[1]) / 1.0,
                        (iz as f64 * spacing[2] - centre[2]) / 2.0,
                    ];
                    let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                    g.set(ix, iy, iz, r - 3.0);
                }
            }
        }
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.0);
        assert!(!verts.is_empty());
        for v in &verts {
            let u = [
                (v[0] - centre[0]) / 0.5,
                (v[1] - centre[1]) / 1.0,
                (v[2] - centre[2]) / 2.0,
            ];
            let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            assert!(
                (r - 3.0).abs() < 0.5,
                "vertex {v:?} at normalised radius {r}"
            );
        }
    }

    #[test]
    fn isosurface_skips_cells_with_nan_samples() {
        let mut g = sphere_grid(12, [1.0; 3], [5.5; 3], 3.0);
        g.set(5, 5, 2, f64::NAN);
        let verts = IsosurfaceExtractor::extract_isosurface(&g, 0.0);
        assert!(!verts.is_empty());
        assert!(verts.iter().all(|v| v.iter().all(|c| c.is_finite())));
    }

    // --- VolumeStats ---

    #[test]
    fn volume_stats_constant_grid() {
        let g = VoxelGrid::from_data([3, 3, 3], [1.0, 1.0, 1.0], vec![5.0; 27]);
        let stats = VolumeStats::compute(&g, 10);
        assert!((stats.min - 5.0).abs() < 1e-12);
        assert!((stats.max - 5.0).abs() < 1e-12);
        assert!((stats.mean - 5.0).abs() < 1e-12);
    }

    #[test]
    fn volume_stats_range() {
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let g = VoxelGrid::from_data([2, 2, 2], [1.0, 1.0, 1.0], data);
        let stats = VolumeStats::compute(&g, 8);
        assert!((stats.min - 0.0).abs() < 1e-12);
        assert!((stats.max - 7.0).abs() < 1e-12);
        assert!((stats.mean - 3.5).abs() < 1e-12);
    }

    #[test]
    fn volume_stats_histogram_total_count() {
        let data: Vec<f64> = (0..27).map(|i| i as f64).collect();
        let g = VoxelGrid::from_data([3, 3, 3], [1.0, 1.0, 1.0], data);
        let stats = VolumeStats::compute(&g, 10);
        assert_eq!(stats.total_count(), 27);
    }

    #[test]
    fn volume_stats_num_bins_respected() {
        let g = VoxelGrid::from_data([2, 2, 2], [1.0, 1.0, 1.0], vec![1.0; 8]);
        let stats = VolumeStats::compute(&g, 16);
        assert_eq!(stats.bins.len(), 16);
    }

    #[test]
    fn volume_stats_empty_grid() {
        let g = VoxelGrid {
            dimensions: [0, 0, 0],
            spacing: [1.0, 1.0, 1.0],
            data: vec![],
        };
        let stats = VolumeStats::compute(&g, 4);
        assert_eq!(stats.total_count(), 0);
        assert_eq!(stats.bins.len(), 4);
    }

    #[test]
    fn volume_stats_bins_at_least_one() {
        let g = VoxelGrid::from_data([2, 2, 2], [1.0, 1.0, 1.0], vec![1.0; 8]);
        let stats = VolumeStats::compute(&g, 0); // min(0, 1) = 1
        assert_eq!(stats.bins.len(), 1);
        assert_eq!(stats.bins[0], 8);
    }
}
