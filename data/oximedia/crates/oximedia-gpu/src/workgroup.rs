#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
//! GPU workgroup configuration and dispatch sizing.
//!
//! This module provides utilities for computing optimal workgroup sizes
//! and dispatch dimensions for GPU compute shaders. Proper workgroup sizing
//! is critical for achieving good GPU utilization.

/// Maximum workgroup size per dimension on most GPUs.
const MAX_WORKGROUP_DIM: u32 = 1024;

/// Maximum total invocations per workgroup (typical limit).
const MAX_WORKGROUP_TOTAL: u32 = 1024;

/// Preferred warp/wavefront size for NVIDIA/AMD GPUs.
const WARP_SIZE: u32 = 32;

/// Workgroup size in 3 dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkgroupSize {
    /// Size in X dimension.
    pub x: u32,
    /// Size in Y dimension.
    pub y: u32,
    /// Size in Z dimension.
    pub z: u32,
}

impl WorkgroupSize {
    /// Create a new workgroup size.
    #[must_use]
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Create a 1D workgroup size.
    #[must_use]
    pub fn linear(size: u32) -> Self {
        Self {
            x: size,
            y: 1,
            z: 1,
        }
    }

    /// Create a 2D workgroup size.
    #[must_use]
    pub fn flat(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Total number of invocations in this workgroup.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Check if the workgroup size is valid (within typical limits).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.x > 0
            && self.y > 0
            && self.z > 0
            && self.x <= MAX_WORKGROUP_DIM
            && self.y <= MAX_WORKGROUP_DIM
            && self.z <= MAX_WORKGROUP_DIM
            && self.total() <= MAX_WORKGROUP_TOTAL
    }

    /// Check if the total size is a multiple of the warp size.
    #[must_use]
    pub fn is_warp_aligned(&self) -> bool {
        self.total() % WARP_SIZE == 0
    }
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self { x: 8, y: 8, z: 1 }
    }
}

/// Dispatch dimensions for launching a compute shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DispatchDimensions {
    /// Number of workgroups in X.
    pub groups_x: u32,
    /// Number of workgroups in Y.
    pub groups_y: u32,
    /// Number of workgroups in Z.
    pub groups_z: u32,
}

impl DispatchDimensions {
    /// Create new dispatch dimensions.
    #[must_use]
    pub fn new(groups_x: u32, groups_y: u32, groups_z: u32) -> Self {
        Self {
            groups_x,
            groups_y,
            groups_z,
        }
    }

    /// Create 1D dispatch dimensions.
    #[must_use]
    pub fn linear(groups: u32) -> Self {
        Self {
            groups_x: groups,
            groups_y: 1,
            groups_z: 1,
        }
    }

    /// Total number of workgroups.
    #[must_use]
    pub fn total_groups(&self) -> u64 {
        u64::from(self.groups_x) * u64::from(self.groups_y) * u64::from(self.groups_z)
    }

    /// Total number of invocations given a workgroup size.
    #[must_use]
    pub fn total_invocations(&self, workgroup: &WorkgroupSize) -> u64 {
        self.total_groups() * u64::from(workgroup.total())
    }
}

/// Strategy for choosing workgroup sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkgroupStrategy {
    /// Use a square workgroup (e.g. 16x16 for 2D).
    Square,
    /// Prefer wide workgroups (e.g. 256x1).
    Wide,
    /// Prefer tall workgroups (e.g. 1x256).
    Tall,
    /// Optimize for warp/wavefront alignment.
    WarpAligned,
    /// Use the smallest valid workgroup size.
    Minimal,
}

/// Compute optimal workgroup size and dispatch dimensions.
pub struct WorkgroupPlanner;

impl WorkgroupPlanner {
    /// Compute 1D dispatch dimensions for a linear problem.
    ///
    /// Returns `(workgroup_size, dispatch_dims)`.
    #[must_use]
    pub fn plan_1d(
        total_elements: u32,
        strategy: WorkgroupStrategy,
    ) -> (WorkgroupSize, DispatchDimensions) {
        let wg_size = match strategy {
            WorkgroupStrategy::WarpAligned => 256,
            WorkgroupStrategy::Minimal => 64,
            _ => 128,
        };
        let wg = WorkgroupSize::linear(wg_size);
        let groups = div_ceil(total_elements, wg_size);
        (wg, DispatchDimensions::linear(groups))
    }

    /// Compute 2D dispatch dimensions for an image-like problem.
    ///
    /// Returns `(workgroup_size, dispatch_dims)`.
    #[must_use]
    pub fn plan_2d(
        width: u32,
        height: u32,
        strategy: WorkgroupStrategy,
    ) -> (WorkgroupSize, DispatchDimensions) {
        let (wg_x, wg_y) = match strategy {
            WorkgroupStrategy::Square => (16, 16),
            WorkgroupStrategy::Wide => (32, 8),
            WorkgroupStrategy::Tall => (8, 32),
            WorkgroupStrategy::WarpAligned => (16, 16),
            WorkgroupStrategy::Minimal => (8, 8),
        };
        let wg = WorkgroupSize::flat(wg_x, wg_y);
        let groups_x = div_ceil(width, wg_x);
        let groups_y = div_ceil(height, wg_y);
        (wg, DispatchDimensions::new(groups_x, groups_y, 1))
    }

    /// Compute 3D dispatch dimensions.
    ///
    /// Returns `(workgroup_size, dispatch_dims)`.
    #[must_use]
    pub fn plan_3d(width: u32, height: u32, depth: u32) -> (WorkgroupSize, DispatchDimensions) {
        let wg = WorkgroupSize::new(8, 8, 4);
        let groups_x = div_ceil(width, 8);
        let groups_y = div_ceil(height, 8);
        let groups_z = div_ceil(depth, 4);
        (wg, DispatchDimensions::new(groups_x, groups_y, groups_z))
    }

    /// Estimate efficiency ratio of a dispatch (useful work / total work).
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::manual_checked_ops)]
    #[must_use]
    pub fn efficiency(
        problem_size: (u32, u32),
        workgroup: &WorkgroupSize,
        dispatch: &DispatchDimensions,
    ) -> f64 {
        let useful = u64::from(problem_size.0) * u64::from(problem_size.1);
        let total = dispatch.total_invocations(workgroup);
        if total == 0 {
            return 0.0;
        }
        useful as f64 / total as f64
    }
}

/// Integer ceiling division.
fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

// ============================================================================
// Device-aware auto-tuning
// ============================================================================

/// GPU device limits relevant to workgroup sizing.
#[derive(Debug, Clone, Copy)]
pub struct DeviceLimits {
    /// Maximum workgroup size per dimension.
    pub max_workgroup_size_per_dim: u32,
    /// Maximum total invocations per workgroup.
    pub max_workgroup_total_invocations: u32,
    /// Maximum shared memory per workgroup in bytes.
    pub max_shared_memory_bytes: u32,
    /// Preferred warp/wavefront size (0 = unknown).
    pub subgroup_size: u32,
    /// Maximum dispatch groups per dimension.
    pub max_dispatch_per_dim: u32,
}

impl Default for DeviceLimits {
    fn default() -> Self {
        Self {
            max_workgroup_size_per_dim: MAX_WORKGROUP_DIM,
            max_workgroup_total_invocations: MAX_WORKGROUP_TOTAL,
            max_shared_memory_bytes: 49152, // 48 KB
            subgroup_size: WARP_SIZE,
            max_dispatch_per_dim: 65535,
        }
    }
}

impl DeviceLimits {
    /// Create `DeviceLimits` from `wgpu::Limits`.
    #[must_use]
    pub fn from_wgpu(limits: &wgpu::Limits) -> Self {
        Self {
            max_workgroup_size_per_dim: limits
                .max_compute_workgroup_size_x
                .min(limits.max_compute_workgroup_size_y)
                .min(limits.max_compute_workgroup_size_z),
            max_workgroup_total_invocations: limits.max_compute_invocations_per_workgroup,
            max_shared_memory_bytes: limits.max_compute_workgroup_storage_size,
            subgroup_size: WARP_SIZE, // wgpu doesn't expose this directly
            max_dispatch_per_dim: limits.max_compute_workgroups_per_dimension,
        }
    }
}

/// Auto-tuner that selects optimal workgroup sizes based on device limits.
pub struct WorkgroupAutoTuner {
    limits: DeviceLimits,
}

impl WorkgroupAutoTuner {
    /// Create a new auto-tuner with the given device limits.
    #[must_use]
    pub fn new(limits: DeviceLimits) -> Self {
        Self { limits }
    }

    /// Create a new auto-tuner with default limits.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DeviceLimits::default())
    }

    /// Get the device limits.
    #[must_use]
    pub fn limits(&self) -> &DeviceLimits {
        &self.limits
    }

    /// Auto-tune a 1D workgroup size for a linear problem.
    ///
    /// Picks the largest warp-aligned size that fits within device limits.
    #[must_use]
    pub fn tune_1d(&self, total_elements: u32) -> (WorkgroupSize, DispatchDimensions) {
        let subgroup = self.limits.subgroup_size.max(1);
        let max_total = self.limits.max_workgroup_total_invocations;
        let max_dim = self.limits.max_workgroup_size_per_dim;

        // Start from 256, clamp to device limits, align to subgroup size.
        let mut size = 256u32.min(max_total).min(max_dim);
        // Round down to subgroup alignment.
        if let Some(aligned) = size.checked_div(subgroup) {
            size = aligned * subgroup;
        }
        size = size.max(subgroup).max(1);

        // If problem is small, use smaller workgroups.
        if total_elements < size * 4 {
            let smaller = (total_elements.div_ceil(subgroup.max(1))) * subgroup.max(1);
            size = smaller.max(subgroup.max(1)).min(size);
        }

        let wg = WorkgroupSize::linear(size);
        let groups = div_ceil(total_elements, size).min(self.limits.max_dispatch_per_dim);
        (wg, DispatchDimensions::linear(groups))
    }

    /// Auto-tune a 2D workgroup size for an image-like problem.
    ///
    /// Balances squareness with warp alignment and device limits.
    #[must_use]
    #[allow(clippy::manual_checked_ops)]
    pub fn tune_2d(&self, width: u32, height: u32) -> (WorkgroupSize, DispatchDimensions) {
        let max_total = self.limits.max_workgroup_total_invocations;
        let max_dim = self.limits.max_workgroup_size_per_dim;
        let subgroup = self.limits.subgroup_size.max(1);

        // Candidate workgroup sizes (prefer multiples of subgroup_size).
        let candidates: [(u32, u32); 6] = [
            (16, 16), // 256 threads — good default
            (32, 8),  // 256 threads — wide, good for row-major access
            (8, 32),  // 256 threads — tall
            (16, 8),  // 128 threads — smaller
            (8, 8),   // 64 threads — small
            (32, 16), // 512 threads — large
        ];

        let mut best_wg = WorkgroupSize::flat(8, 8);
        let mut best_efficiency = 0.0_f64;

        for &(wx, wy) in &candidates {
            if wx > max_dim || wy > max_dim || wx * wy > max_total {
                continue;
            }
            // Prefer warp-aligned total.
            let total = wx * wy;
            if total % subgroup != 0 {
                continue;
            }

            let gx = div_ceil(width, wx).min(self.limits.max_dispatch_per_dim);
            let gy = div_ceil(height, wy).min(self.limits.max_dispatch_per_dim);
            let total_invocations = (gx as u64) * (gy as u64) * (total as u64);
            let useful = (width as u64) * (height as u64);
            let eff = if total_invocations > 0 {
                useful as f64 / total_invocations as f64
            } else {
                0.0
            };

            if eff > best_efficiency {
                best_efficiency = eff;
                best_wg = WorkgroupSize::flat(wx, wy);
            }
        }

        let gx = div_ceil(width, best_wg.x).min(self.limits.max_dispatch_per_dim);
        let gy = div_ceil(height, best_wg.y).min(self.limits.max_dispatch_per_dim);
        (best_wg, DispatchDimensions::new(gx, gy, 1))
    }

    /// Auto-tune for a 2D problem with shared memory requirements.
    ///
    /// Takes into account the per-pixel shared memory usage and ensures
    /// the workgroup's shared memory fits within device limits.
    #[must_use]
    pub fn tune_2d_with_shared_memory(
        &self,
        width: u32,
        height: u32,
        shared_bytes_per_pixel: u32,
    ) -> (WorkgroupSize, DispatchDimensions) {
        let max_shared = self.limits.max_shared_memory_bytes;
        let max_total = self.limits.max_workgroup_total_invocations;
        let max_dim = self.limits.max_workgroup_size_per_dim;
        let subgroup = self.limits.subgroup_size.max(1);

        // Find largest square-ish workgroup whose shared mem fits.
        let mut best_side = 8u32;
        for candidate_side in &[32u32, 24, 16, 12, 8] {
            let side = *candidate_side;
            let total = side * side;
            if total > max_total || side > max_dim {
                continue;
            }
            if total % subgroup != 0 {
                continue;
            }
            let shared_needed = total * shared_bytes_per_pixel;
            if shared_needed <= max_shared {
                best_side = side;
                break;
            }
        }

        let wg = WorkgroupSize::flat(best_side, best_side);
        let gx = div_ceil(width, best_side).min(self.limits.max_dispatch_per_dim);
        let gy = div_ceil(height, best_side).min(self.limits.max_dispatch_per_dim);
        (wg, DispatchDimensions::new(gx, gy, 1))
    }

    /// Estimate the efficiency of a given configuration.
    #[must_use]
    pub fn estimate_efficiency(
        &self,
        problem_width: u32,
        problem_height: u32,
        workgroup: &WorkgroupSize,
    ) -> f64 {
        let gx = div_ceil(problem_width, workgroup.x);
        let gy = div_ceil(problem_height, workgroup.y);
        let dispatch = DispatchDimensions::new(gx, gy, 1);
        WorkgroupPlanner::efficiency((problem_width, problem_height), workgroup, &dispatch)
    }
}

/// Shared memory layout descriptor for a workgroup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedMemoryLayout {
    /// Size in bytes per workgroup.
    pub size_bytes: u32,
    /// Alignment requirement in bytes.
    pub alignment: u32,
    /// Number of elements (stride-based).
    pub element_count: u32,
    /// Size per element in bytes.
    pub element_size: u32,
}

impl SharedMemoryLayout {
    /// Create a new shared memory layout.
    #[must_use]
    pub fn new(element_count: u32, element_size: u32, alignment: u32) -> Self {
        let aligned_element = round_up(element_size, alignment);
        Self {
            size_bytes: element_count * aligned_element,
            alignment,
            element_count,
            element_size,
        }
    }

    /// Create a layout for float data.
    #[must_use]
    pub fn floats(count: u32) -> Self {
        Self::new(count, 4, 4)
    }

    /// Create a layout for vec4 data.
    #[must_use]
    pub fn vec4s(count: u32) -> Self {
        Self::new(count, 16, 16)
    }

    /// Check if the layout fits within the typical shared memory limit (48 KB).
    #[must_use]
    pub fn fits_in_shared_memory(&self) -> bool {
        self.size_bytes <= 49152 // 48 * 1024
    }
}

/// Round a value up to a given alignment.
fn round_up(value: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workgroup_size_default() {
        let wg = WorkgroupSize::default();
        assert_eq!(wg.x, 8);
        assert_eq!(wg.y, 8);
        assert_eq!(wg.z, 1);
        assert_eq!(wg.total(), 64);
    }

    #[test]
    fn test_workgroup_size_linear() {
        let wg = WorkgroupSize::linear(256);
        assert_eq!(wg.total(), 256);
        assert!(wg.is_valid());
        assert!(wg.is_warp_aligned());
    }

    #[test]
    fn test_workgroup_size_flat() {
        let wg = WorkgroupSize::flat(16, 16);
        assert_eq!(wg.total(), 256);
        assert!(wg.is_valid());
    }

    #[test]
    fn test_workgroup_size_3d() {
        let wg = WorkgroupSize::new(8, 8, 4);
        assert_eq!(wg.total(), 256);
        assert!(wg.is_valid());
    }

    #[test]
    fn test_workgroup_size_invalid_exceeds_max() {
        let wg = WorkgroupSize::new(1025, 1, 1);
        assert!(!wg.is_valid());
    }

    #[test]
    fn test_workgroup_size_invalid_exceeds_total() {
        let wg = WorkgroupSize::new(32, 64, 1);
        assert_eq!(wg.total(), 2048);
        assert!(!wg.is_valid());
    }

    #[test]
    fn test_dispatch_dimensions_linear() {
        let d = DispatchDimensions::linear(10);
        assert_eq!(d.total_groups(), 10);
    }

    #[test]
    fn test_dispatch_total_invocations() {
        let wg = WorkgroupSize::flat(16, 16);
        let d = DispatchDimensions::new(4, 4, 1);
        assert_eq!(d.total_invocations(&wg), 4096);
    }

    #[test]
    fn test_plan_1d() {
        let (wg, d) = WorkgroupPlanner::plan_1d(1000, WorkgroupStrategy::WarpAligned);
        assert_eq!(wg.x, 256);
        assert!(d.groups_x * wg.x >= 1000);
    }

    #[test]
    fn test_plan_2d_square() {
        let (wg, d) = WorkgroupPlanner::plan_2d(1920, 1080, WorkgroupStrategy::Square);
        assert_eq!(wg.x, 16);
        assert_eq!(wg.y, 16);
        assert!(d.groups_x * wg.x >= 1920);
        assert!(d.groups_y * wg.y >= 1080);
    }

    #[test]
    fn test_plan_2d_wide() {
        let (wg, d) = WorkgroupPlanner::plan_2d(3840, 2160, WorkgroupStrategy::Wide);
        assert_eq!(wg.x, 32);
        assert_eq!(wg.y, 8);
        assert!(d.groups_x * wg.x >= 3840);
        assert!(d.groups_y * wg.y >= 2160);
    }

    #[test]
    fn test_plan_3d() {
        let (wg, d) = WorkgroupPlanner::plan_3d(64, 64, 16);
        assert_eq!(wg.total(), 256);
        assert_eq!(d.groups_x, 8);
        assert_eq!(d.groups_y, 8);
        assert_eq!(d.groups_z, 4);
    }

    #[test]
    fn test_efficiency_perfect() {
        let wg = WorkgroupSize::flat(16, 16);
        let d = DispatchDimensions::new(2, 2, 1);
        let eff = WorkgroupPlanner::efficiency((32, 32), &wg, &d);
        assert!((eff - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_efficiency_partial() {
        let wg = WorkgroupSize::flat(16, 16);
        let d = DispatchDimensions::new(1, 1, 1);
        let eff = WorkgroupPlanner::efficiency((10, 10), &wg, &d);
        assert!(eff < 1.0);
        assert!(eff > 0.0);
    }

    #[test]
    fn test_shared_memory_floats() {
        let layout = SharedMemoryLayout::floats(256);
        assert_eq!(layout.size_bytes, 1024);
        assert!(layout.fits_in_shared_memory());
    }

    #[test]
    fn test_shared_memory_vec4s() {
        let layout = SharedMemoryLayout::vec4s(64);
        assert_eq!(layout.size_bytes, 1024);
        assert!(layout.fits_in_shared_memory());
    }

    #[test]
    fn test_shared_memory_exceeds_limit() {
        let layout = SharedMemoryLayout::new(50000, 4, 4);
        assert!(!layout.fits_in_shared_memory());
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(10, 3), 4);
        assert_eq!(div_ceil(9, 3), 3);
        assert_eq!(div_ceil(1, 256), 1);
    }

    #[test]
    fn test_round_up() {
        assert_eq!(round_up(5, 4), 8);
        assert_eq!(round_up(8, 4), 8);
        assert_eq!(round_up(0, 4), 0);
        assert_eq!(round_up(7, 0), 7);
    }

    #[test]
    fn test_warp_alignment() {
        let wg = WorkgroupSize::linear(64);
        assert!(wg.is_warp_aligned());
        let wg2 = WorkgroupSize::linear(33);
        assert!(!wg2.is_warp_aligned());
    }

    // --- Auto-tuner tests ---

    #[test]
    fn test_auto_tuner_1d_default_limits() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        let (wg, dispatch) = tuner.tune_1d(10000);
        assert!(wg.is_valid(), "workgroup must be valid");
        assert!(wg.is_warp_aligned(), "should be warp-aligned");
        assert!(dispatch.groups_x * wg.x >= 10000, "must cover all elements");
    }

    #[test]
    fn test_auto_tuner_1d_small_problem() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        let (wg, dispatch) = tuner.tune_1d(64);
        assert!(wg.is_valid());
        assert!(dispatch.groups_x * wg.x >= 64);
    }

    #[test]
    fn test_auto_tuner_2d_1080p() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        let (wg, dispatch) = tuner.tune_2d(1920, 1080);
        assert!(wg.is_valid());
        assert!(wg.is_warp_aligned());
        assert!(dispatch.groups_x * wg.x >= 1920);
        assert!(dispatch.groups_y * wg.y >= 1080);
    }

    #[test]
    fn test_auto_tuner_2d_4k() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        let (wg, dispatch) = tuner.tune_2d(3840, 2160);
        assert!(wg.is_valid());
        assert!(dispatch.groups_x * wg.x >= 3840);
        assert!(dispatch.groups_y * wg.y >= 2160);
    }

    #[test]
    fn test_auto_tuner_2d_small_image() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        let (wg, dispatch) = tuner.tune_2d(16, 16);
        assert!(wg.is_valid());
        assert!(dispatch.groups_x * wg.x >= 16);
        assert!(dispatch.groups_y * wg.y >= 16);
    }

    #[test]
    fn test_auto_tuner_2d_non_square() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        let (wg, dispatch) = tuner.tune_2d(4096, 32);
        assert!(wg.is_valid());
        assert!(dispatch.groups_x * wg.x >= 4096);
        assert!(dispatch.groups_y * wg.y >= 32);
    }

    #[test]
    fn test_auto_tuner_with_shared_memory() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        // 64 bytes per pixel shared memory — should pick smaller workgroup.
        let (wg, dispatch) = tuner.tune_2d_with_shared_memory(1920, 1080, 64);
        let shared_used = wg.total() * 64;
        assert!(
            shared_used <= tuner.limits().max_shared_memory_bytes,
            "shared memory {} must fit in {} bytes",
            shared_used,
            tuner.limits().max_shared_memory_bytes
        );
        assert!(dispatch.groups_x * wg.x >= 1920);
        assert!(dispatch.groups_y * wg.y >= 1080);
    }

    #[test]
    fn test_auto_tuner_with_large_shared_memory() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        // Very large per-pixel shared memory — should fall back to small workgroup.
        let (wg, dispatch) = tuner.tune_2d_with_shared_memory(256, 256, 512);
        let shared_used = wg.total() * 512;
        assert!(shared_used <= tuner.limits().max_shared_memory_bytes);
        assert!(dispatch.groups_x * wg.x >= 256);
    }

    #[test]
    fn test_auto_tuner_respects_constrained_limits() {
        let limits = DeviceLimits {
            max_workgroup_size_per_dim: 128,
            max_workgroup_total_invocations: 128,
            max_shared_memory_bytes: 16384,
            subgroup_size: 16,
            max_dispatch_per_dim: 32768,
        };
        let tuner = WorkgroupAutoTuner::new(limits);
        let (wg, _) = tuner.tune_2d(1920, 1080);
        assert!(wg.x <= 128);
        assert!(wg.y <= 128);
        assert!(wg.total() <= 128);
    }

    #[test]
    fn test_auto_tuner_efficiency_estimate() {
        let tuner = WorkgroupAutoTuner::with_defaults();
        let wg = WorkgroupSize::flat(16, 16);
        let eff = tuner.estimate_efficiency(32, 32, &wg);
        assert!(
            (eff - 1.0).abs() < 1e-9,
            "perfect fit should have efficiency 1.0"
        );

        let eff2 = tuner.estimate_efficiency(17, 17, &wg);
        assert!(
            eff2 < 1.0,
            "non-aligned problem should have < 1.0 efficiency"
        );
        assert!(eff2 > 0.0);
    }

    #[test]
    fn test_device_limits_default() {
        let limits = DeviceLimits::default();
        assert_eq!(limits.max_workgroup_size_per_dim, 1024);
        assert_eq!(limits.max_workgroup_total_invocations, 1024);
        assert_eq!(limits.max_shared_memory_bytes, 49152);
        assert_eq!(limits.subgroup_size, 32);
    }
}
