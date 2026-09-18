//! Compute shader dispatch helpers.
//!
//! Provides workgroup sizing utilities, dispatch grid calculation, and
//! basic barrier / dependency tracking for GPU compute passes.

/// Maximum recommended workgroup size per dimension on most GPUs.
pub const MAX_WORKGROUP_DIM: u32 = 256;

/// A 3-D workgroup size.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupSize {
    /// Create a 1-D workgroup (y=1, z=1).
    #[allow(dead_code)]
    #[must_use]
    pub const fn linear(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a 2-D workgroup (z=1).
    #[allow(dead_code)]
    #[must_use]
    pub const fn planar(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a full 3-D workgroup.
    #[allow(dead_code)]
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total number of threads per workgroup.
    #[allow(dead_code)]
    #[must_use]
    pub const fn thread_count(self) -> u32 {
        self.x * self.y * self.z
    }

    /// Returns `true` if the workgroup size is valid (all dims ≥ 1 and total
    /// threads ≤ `max_threads`).
    #[allow(dead_code)]
    #[must_use]
    pub fn is_valid(self, max_threads: u32) -> bool {
        self.x >= 1 && self.y >= 1 && self.z >= 1 && self.thread_count() <= max_threads
    }
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self::linear(64)
    }
}

/// A 3-D dispatch grid (number of workgroups in each dimension).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchGrid {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl DispatchGrid {
    /// Create a new dispatch grid.
    #[allow(dead_code)]
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total workgroups dispatched.
    #[allow(dead_code)]
    #[must_use]
    pub const fn total_workgroups(self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }

    /// Total threads dispatched (grid × workgroup size).
    #[allow(dead_code)]
    #[must_use]
    pub const fn total_threads(self, wg: WorkgroupSize) -> u64 {
        self.total_workgroups() * wg.thread_count() as u64
    }
}

/// Calculate the dispatch grid needed to cover `count` elements with
/// threads of size `wg_size` in the X dimension.
#[allow(dead_code)]
#[must_use]
pub fn dispatch_1d(count: u32, wg_size: u32) -> DispatchGrid {
    assert!(wg_size > 0, "wg_size must be > 0");
    let x = count.div_ceil(wg_size);
    DispatchGrid::new(x, 1, 1)
}

/// Calculate the dispatch grid needed to cover a `width × height` image with
/// a planar workgroup of size `(wg_x, wg_y)`.
#[allow(dead_code)]
#[must_use]
pub fn dispatch_2d(width: u32, height: u32, wg_x: u32, wg_y: u32) -> DispatchGrid {
    assert!(wg_x > 0 && wg_y > 0, "workgroup dims must be > 0");
    let x = width.div_ceil(wg_x);
    let y = height.div_ceil(wg_y);
    DispatchGrid::new(x, y, 1)
}

/// Calculate the dispatch grid for a 3-D volume.
#[allow(dead_code)]
#[must_use]
pub fn dispatch_3d(
    width: u32,
    height: u32,
    depth: u32,
    wg_x: u32,
    wg_y: u32,
    wg_z: u32,
) -> DispatchGrid {
    assert!(
        wg_x > 0 && wg_y > 0 && wg_z > 0,
        "workgroup dims must be > 0"
    );
    DispatchGrid::new(
        width.div_ceil(wg_x),
        height.div_ceil(wg_y),
        depth.div_ceil(wg_z),
    )
}

/// Recommend a square workgroup size that keeps total threads ≤ `max_threads`
/// and is a power of two.
#[allow(dead_code)]
#[must_use]
pub fn recommend_2d_workgroup(max_threads: u32) -> WorkgroupSize {
    let mut side = 1u32;
    while side * side * 4 <= max_threads {
        side *= 2;
    }
    // side² ≤ max_threads
    while side * side > max_threads {
        side /= 2;
    }
    WorkgroupSize::planar(side.max(1), side.max(1))
}

// ---------------------------------------------------------------------------
// Barrier tracking
// ---------------------------------------------------------------------------

/// Type of pipeline barrier.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierKind {
    /// Ensures all memory writes are visible to subsequent reads.
    MemoryReadAfterWrite,
    /// Ensures all dispatches before the barrier complete before new ones begin.
    ExecutionOnly,
    /// Full pipeline barrier (most restrictive, highest cost).
    Full,
}

/// A recorded pipeline barrier.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BarrierRecord {
    /// Sequential index in the command stream.
    pub index: u32,
    /// Kind of barrier.
    pub kind: BarrierKind,
    /// Optional label for debugging.
    pub label: Option<String>,
}

/// Tracks barriers inserted during a compute pass.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct BarrierTracker {
    records: Vec<BarrierRecord>,
    next_index: u32,
}

impl BarrierTracker {
    /// Create a new tracker.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a barrier with the given kind and optional label.
    #[allow(dead_code)]
    pub fn push(&mut self, kind: BarrierKind, label: Option<&str>) {
        self.records.push(BarrierRecord {
            index: self.next_index,
            kind,
            label: label.map(String::from),
        });
        self.next_index += 1;
    }

    /// Number of barriers recorded.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true if no barriers have been recorded.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// All recorded barriers.
    #[allow(dead_code)]
    #[must_use]
    pub fn records(&self) -> &[BarrierRecord] {
        &self.records
    }

    /// Count barriers of a specific kind.
    #[allow(dead_code)]
    #[must_use]
    pub fn count_of_kind(&self, kind: BarrierKind) -> usize {
        self.records.iter().filter(|r| r.kind == kind).count()
    }

    /// Reset the tracker.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.records.clear();
        self.next_index = 0;
    }
}

// ---------------------------------------------------------------------------
// Dispatch record
// ---------------------------------------------------------------------------

/// A recorded compute dispatch (for replay / inspection).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DispatchRecord {
    /// Sequential index.
    pub index: u32,
    /// The pipeline identifier (e.g. shader name).
    pub pipeline_id: String,
    /// The dispatch grid.
    pub grid: DispatchGrid,
    /// The workgroup size declared by the shader.
    pub workgroup_size: WorkgroupSize,
}

/// Tracks dispatches in a compute pass.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct DispatchTracker {
    records: Vec<DispatchRecord>,
    next_index: u32,
}

impl DispatchTracker {
    /// Create a new tracker.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a dispatch.
    #[allow(dead_code)]
    pub fn push(
        &mut self,
        pipeline_id: impl Into<String>,
        grid: DispatchGrid,
        workgroup_size: WorkgroupSize,
    ) {
        self.records.push(DispatchRecord {
            index: self.next_index,
            pipeline_id: pipeline_id.into(),
            grid,
            workgroup_size,
        });
        self.next_index += 1;
    }

    /// Number of dispatches recorded.
    #[allow(dead_code)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true when no dispatches have been recorded.
    #[allow(dead_code)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Total GPU threads dispatched.
    #[allow(dead_code)]
    #[must_use]
    pub fn total_threads(&self) -> u64 {
        self.records
            .iter()
            .map(|r| r.grid.total_threads(r.workgroup_size))
            .sum()
    }

    /// All dispatch records.
    #[allow(dead_code)]
    #[must_use]
    pub fn records(&self) -> &[DispatchRecord] {
        &self.records
    }

    /// Reset the tracker.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.records.clear();
        self.next_index = 0;
    }
}

// ---------------------------------------------------------------------------
// Data-driven (indirect) dispatch support
// ---------------------------------------------------------------------------

/// Strategy used to derive workgroup counts from a data-dependent element
/// count at dispatch preparation time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataDispatchStrategy {
    /// All elements are processed in a single 1D strip: `(ceil(n/wg), 1, 1)`.
    Linear1D,
    /// Elements are spread over a near-square 2D grid.
    Square2D,
    /// Fixed number of rows; columns derived from `ceil(n / (rows * wg_x))`.
    FixedRowCount {
        /// Number of rows in the Y dimension.
        rows: u32,
    },
}

/// Computes and stores dispatch parameters that depend on the number of data
/// elements only known at dispatch-preparation time (e.g., after a GPU
/// readback or a CPU-side counter).
///
/// In a real GPU pipeline this feeds an *indirect dispatch buffer*; here we
/// compute the [`DispatchGrid`] on the CPU side for portability and testing.
pub struct DataDrivenDispatch {
    /// Workgroup size in X.
    wg_x: u32,
    /// Workgroup size in Y.
    wg_y: u32,
    strategy: DataDispatchStrategy,
    /// Grid computed from the last call to [`Self::prepare`].
    grid: Option<DispatchGrid>,
    /// Element count from the last call to [`Self::prepare`].
    last_element_count: u64,
}

impl DataDrivenDispatch {
    /// Create a new data-driven dispatch helper.
    ///
    /// * `wg_x` / `wg_y` — workgroup size dimensions (must be ≥ 1).
    /// * `strategy` — how to map element counts to workgroup grids.
    #[must_use]
    pub fn new(wg_x: u32, wg_y: u32, strategy: DataDispatchStrategy) -> Self {
        let wg_x = wg_x.max(1);
        let wg_y = wg_y.max(1);
        Self {
            wg_x,
            wg_y,
            strategy,
            grid: None,
            last_element_count: 0,
        }
    }

    /// Convenience constructor for a 1D strip with `wg_size` threads per
    /// workgroup.
    #[must_use]
    pub fn linear(wg_size: u32) -> Self {
        Self::new(wg_size, 1, DataDispatchStrategy::Linear1D)
    }

    /// Convenience constructor for a 2D square grid with `wg_x × wg_y`
    /// threads per workgroup.
    #[must_use]
    pub fn square(wg_x: u32, wg_y: u32) -> Self {
        Self::new(wg_x, wg_y, DataDispatchStrategy::Square2D)
    }

    /// Prepare the dispatch grid for `element_count` data elements.
    ///
    /// Returns the resulting [`DispatchGrid`]; the value is also stored
    /// internally and accessible via [`Self::grid`].
    pub fn prepare(&mut self, element_count: u64) -> DispatchGrid {
        self.last_element_count = element_count;
        let n = element_count as u32;
        let grid = match self.strategy {
            DataDispatchStrategy::Linear1D => {
                let x = n.div_ceil(self.wg_x);
                DispatchGrid::new(x.max(1), 1, 1)
            }
            DataDispatchStrategy::Square2D => {
                let threads_per_wg = self.wg_x * self.wg_y;
                let total_wgs = n.div_ceil(threads_per_wg).max(1);
                let side = (total_wgs as f64).sqrt().ceil() as u32;
                let side = side.max(1);
                DispatchGrid::new(side, side, 1)
            }
            DataDispatchStrategy::FixedRowCount { rows } => {
                let rows = rows.max(1);
                // Each row handles `cols` workgroups; each workgroup covers
                // `wg_x` elements in X and implicitly one row in Y.
                let total_wgs = n.div_ceil(self.wg_x * self.wg_y).max(1);
                let cols = total_wgs.div_ceil(rows);
                DispatchGrid::new(cols, rows, 1)
            }
        };
        self.grid = Some(grid);
        grid
    }

    /// The grid computed by the last [`Self::prepare`] call, or `None` if
    /// [`Self::prepare`] has not yet been called.
    #[must_use]
    pub fn grid(&self) -> Option<DispatchGrid> {
        self.grid
    }

    /// The element count supplied to the last [`Self::prepare`] call.
    #[must_use]
    pub fn last_element_count(&self) -> u64 {
        self.last_element_count
    }

    /// Minimum elements coverable by the last computed grid.
    ///
    /// Returns 0 if [`Self::prepare`] has not been called.
    #[must_use]
    pub fn covered_elements(&self) -> u64 {
        match self.grid {
            None => 0,
            Some(g) => {
                u64::from(g.total_workgroups()) * u64::from(self.wg_x) * u64::from(self.wg_y)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_thread_count() {
        let wg = WorkgroupSize::new(8, 8, 1);
        assert_eq!(wg.thread_count(), 64);
    }

    #[test]
    fn test_workgroup_is_valid() {
        assert!(WorkgroupSize::linear(64).is_valid(1024));
        assert!(!WorkgroupSize::new(33, 33, 1).is_valid(1024));
    }

    #[test]
    fn test_dispatch_1d_exact() {
        let g = dispatch_1d(256, 64);
        assert_eq!(g.x, 4);
        assert_eq!(g.y, 1);
        assert_eq!(g.z, 1);
    }

    #[test]
    fn test_dispatch_1d_rounds_up() {
        let g = dispatch_1d(257, 64);
        assert_eq!(g.x, 5);
    }

    #[test]
    fn test_dispatch_2d() {
        let g = dispatch_2d(1920, 1080, 16, 16);
        assert_eq!(g.x, 120); // 1920 / 16
        assert_eq!(g.y, 68); // ceil(1080 / 16)
    }

    #[test]
    fn test_dispatch_3d() {
        let g = dispatch_3d(8, 8, 8, 4, 4, 4);
        assert_eq!(g.x, 2);
        assert_eq!(g.y, 2);
        assert_eq!(g.z, 2);
    }

    #[test]
    fn test_total_workgroups() {
        let g = DispatchGrid::new(4, 4, 1);
        assert_eq!(g.total_workgroups(), 16);
    }

    #[test]
    fn test_total_threads() {
        let g = DispatchGrid::new(2, 2, 1);
        let wg = WorkgroupSize::planar(8, 8);
        assert_eq!(g.total_threads(wg), 256);
    }

    #[test]
    fn test_recommend_2d_workgroup_within_limit() {
        let wg = recommend_2d_workgroup(256);
        assert!(wg.thread_count() <= 256);
    }

    #[test]
    fn test_recommend_2d_workgroup_square() {
        let wg = recommend_2d_workgroup(1024);
        assert_eq!(wg.x, wg.y);
    }

    #[test]
    fn test_barrier_tracker_push_and_count() {
        let mut bt = BarrierTracker::new();
        bt.push(BarrierKind::MemoryReadAfterWrite, Some("pre-blur"));
        bt.push(BarrierKind::Full, None);
        assert_eq!(bt.len(), 2);
        assert_eq!(bt.count_of_kind(BarrierKind::Full), 1);
    }

    #[test]
    fn test_barrier_tracker_reset() {
        let mut bt = BarrierTracker::new();
        bt.push(BarrierKind::ExecutionOnly, None);
        bt.reset();
        assert!(bt.is_empty());
    }

    #[test]
    fn test_dispatch_tracker_total_threads() {
        let mut dt = DispatchTracker::new();
        dt.push(
            "blur",
            DispatchGrid::new(10, 10, 1),
            WorkgroupSize::planar(8, 8),
        );
        // 100 workgroups × 64 threads = 6400
        assert_eq!(dt.total_threads(), 6400);
    }

    #[test]
    fn test_dispatch_tracker_records_sequential_indices() {
        let mut dt = DispatchTracker::new();
        dt.push("a", DispatchGrid::new(1, 1, 1), WorkgroupSize::linear(64));
        dt.push("b", DispatchGrid::new(1, 1, 1), WorkgroupSize::linear(64));
        assert_eq!(dt.records()[0].index, 0);
        assert_eq!(dt.records()[1].index, 1);
    }

    #[test]
    fn test_dispatch_tracker_reset() {
        let mut dt = DispatchTracker::new();
        dt.push("x", DispatchGrid::new(1, 1, 1), WorkgroupSize::linear(32));
        dt.reset();
        assert!(dt.is_empty());
        assert_eq!(dt.total_threads(), 0);
    }

    // --- DataDrivenDispatch tests ---

    #[test]
    fn test_data_driven_linear_exact() {
        let mut dd = DataDrivenDispatch::linear(64);
        let g = dd.prepare(128);
        assert_eq!(g.x, 2);
        assert_eq!(g.y, 1);
        assert_eq!(g.z, 1);
    }

    #[test]
    fn test_data_driven_linear_rounds_up() {
        let mut dd = DataDrivenDispatch::linear(64);
        let g = dd.prepare(65);
        assert_eq!(g.x, 2);
    }

    #[test]
    fn test_data_driven_linear_zero_elements() {
        let mut dd = DataDrivenDispatch::linear(64);
        let g = dd.prepare(0);
        // Must produce at least 1 workgroup
        assert_eq!(g.x, 1);
    }

    #[test]
    fn test_data_driven_square_covers_all_elements() {
        let mut dd = DataDrivenDispatch::square(8, 8);
        dd.prepare(500);
        // covered_elements ≥ 500
        assert!(dd.covered_elements() >= 500);
    }

    #[test]
    fn test_data_driven_square_grid_is_square() {
        let mut dd = DataDrivenDispatch::square(8, 8);
        let g = dd.prepare(1024);
        assert_eq!(g.x, g.y);
    }

    #[test]
    fn test_data_driven_fixed_row_count() {
        let mut dd = DataDrivenDispatch::new(8, 1, DataDispatchStrategy::FixedRowCount { rows: 4 });
        let g = dd.prepare(256);
        // 256 elements / 8 per wg = 32 workgroups; 32 / 4 rows = 8 cols
        assert_eq!(g.y, 4);
        assert_eq!(g.x, 8);
    }

    #[test]
    fn test_data_driven_grid_none_before_prepare() {
        let dd = DataDrivenDispatch::linear(32);
        assert!(dd.grid().is_none());
        assert_eq!(dd.covered_elements(), 0);
    }

    #[test]
    fn test_data_driven_last_element_count_stored() {
        let mut dd = DataDrivenDispatch::linear(16);
        dd.prepare(999);
        assert_eq!(dd.last_element_count(), 999);
    }

    #[test]
    fn test_data_driven_covered_elements_gte_last_count() {
        let mut dd = DataDrivenDispatch::square(4, 4);
        let count = 137_u64;
        dd.prepare(count);
        assert!(dd.covered_elements() >= count);
    }
}
