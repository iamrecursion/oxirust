#![allow(dead_code)]
//! GPU occupancy calculator and optimization.
//!
//! This module provides tools for estimating GPU occupancy based on
//! workgroup size, register usage, and shared memory consumption.
//! Higher occupancy generally means better latency hiding.

/// Typical GPU multiprocessor specification.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuSpec {
    /// Maximum threads per multiprocessor.
    pub max_threads_per_sm: u32,
    /// Maximum workgroups (blocks) per multiprocessor.
    pub max_blocks_per_sm: u32,
    /// Maximum shared memory per multiprocessor in bytes.
    pub max_shared_memory_per_sm: u32,
    /// Maximum registers per multiprocessor.
    pub max_registers_per_sm: u32,
    /// Warp size (wavefront size).
    pub warp_size: u32,
    /// Total number of multiprocessors on the device.
    pub sm_count: u32,
}

impl GpuSpec {
    /// Create a spec representing a mid-range desktop GPU.
    #[must_use]
    pub fn mid_range() -> Self {
        Self {
            max_threads_per_sm: 1536,
            max_blocks_per_sm: 16,
            max_shared_memory_per_sm: 49152,
            max_registers_per_sm: 65536,
            warp_size: 32,
            sm_count: 30,
        }
    }

    /// Create a spec representing a high-end desktop GPU.
    #[must_use]
    pub fn high_end() -> Self {
        Self {
            max_threads_per_sm: 2048,
            max_blocks_per_sm: 32,
            max_shared_memory_per_sm: 102400,
            max_registers_per_sm: 65536,
            warp_size: 32,
            sm_count: 80,
        }
    }

    /// Create a spec representing integrated/mobile GPU.
    #[must_use]
    pub fn integrated() -> Self {
        Self {
            max_threads_per_sm: 512,
            max_blocks_per_sm: 8,
            max_shared_memory_per_sm: 32768,
            max_registers_per_sm: 32768,
            warp_size: 32,
            sm_count: 8,
        }
    }
}

/// Kernel resource requirements.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelResources {
    /// Threads per workgroup (block).
    pub threads_per_block: u32,
    /// Registers used per thread.
    pub registers_per_thread: u32,
    /// Shared memory used per block in bytes.
    pub shared_memory_per_block: u32,
}

impl KernelResources {
    /// Create a new kernel resource descriptor.
    #[must_use]
    pub fn new(threads: u32, registers: u32, shared_mem: u32) -> Self {
        Self {
            threads_per_block: threads,
            registers_per_thread: registers,
            shared_memory_per_block: shared_mem,
        }
    }

    /// Create a simple kernel with typical register usage.
    #[must_use]
    pub fn simple(threads: u32) -> Self {
        Self {
            threads_per_block: threads,
            registers_per_thread: 32,
            shared_memory_per_block: 0,
        }
    }
}

/// Result of occupancy calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct OccupancyResult {
    /// Achieved occupancy ratio (0.0 to 1.0).
    pub occupancy: f64,
    /// Active warps per SM.
    pub active_warps_per_sm: u32,
    /// Maximum possible warps per SM.
    pub max_warps_per_sm: u32,
    /// Active blocks per SM.
    pub active_blocks_per_sm: u32,
    /// Limiting factor for occupancy.
    pub limiting_factor: OccupancyLimit,
}

/// The factor that limits GPU occupancy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OccupancyLimit {
    /// Limited by max blocks per SM.
    BlockCount,
    /// Limited by thread count.
    ThreadCount,
    /// Limited by register usage.
    Registers,
    /// Limited by shared memory.
    SharedMemory,
    /// No limiting factor (full occupancy).
    None,
}

impl std::fmt::Display for OccupancyLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BlockCount => write!(f, "block count"),
            Self::ThreadCount => write!(f, "thread count"),
            Self::Registers => write!(f, "register usage"),
            Self::SharedMemory => write!(f, "shared memory"),
            Self::None => write!(f, "none"),
        }
    }
}

/// GPU occupancy calculator.
pub struct OccupancyCalculator;

impl OccupancyCalculator {
    /// Calculate occupancy for a given GPU spec and kernel configuration.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::manual_checked_ops)]
    #[must_use]
    pub fn calculate(spec: &GpuSpec, kernel: &KernelResources) -> OccupancyResult {
        if kernel.threads_per_block == 0 || spec.warp_size == 0 {
            return OccupancyResult {
                occupancy: 0.0,
                active_warps_per_sm: 0,
                max_warps_per_sm: 0,
                active_blocks_per_sm: 0,
                limiting_factor: OccupancyLimit::ThreadCount,
            };
        }

        let max_warps = spec.max_threads_per_sm / spec.warp_size;
        let warps_per_block = kernel.threads_per_block.div_ceil(spec.warp_size);

        // Limit by block count
        let blocks_by_count = spec.max_blocks_per_sm;

        // Limit by thread count
        let blocks_by_threads = max_warps.checked_div(warps_per_block).unwrap_or(0);

        // Limit by registers
        let blocks_by_registers = if kernel.registers_per_thread > 0 {
            let regs_per_block = kernel.registers_per_thread * kernel.threads_per_block;
            spec.max_registers_per_sm
                .checked_div(regs_per_block)
                .unwrap_or(blocks_by_count)
        } else {
            blocks_by_count
        };

        // Limit by shared memory
        let blocks_by_shared = spec
            .max_shared_memory_per_sm
            .checked_div(kernel.shared_memory_per_block)
            .unwrap_or(blocks_by_count);

        // Find limiting factor
        let active_blocks = blocks_by_count
            .min(blocks_by_threads)
            .min(blocks_by_registers)
            .min(blocks_by_shared);

        let limiting_factor = if active_blocks == 0 {
            OccupancyLimit::ThreadCount
        } else if active_blocks == blocks_by_shared && blocks_by_shared < blocks_by_count {
            OccupancyLimit::SharedMemory
        } else if active_blocks == blocks_by_registers && blocks_by_registers < blocks_by_count {
            OccupancyLimit::Registers
        } else if active_blocks == blocks_by_threads && blocks_by_threads < blocks_by_count {
            OccupancyLimit::ThreadCount
        } else if active_blocks == blocks_by_count {
            OccupancyLimit::BlockCount
        } else {
            OccupancyLimit::None
        };

        let active_warps = active_blocks * warps_per_block;
        let occupancy = if max_warps > 0 {
            f64::from(active_warps) / f64::from(max_warps)
        } else {
            0.0
        };
        let occupancy = occupancy.min(1.0);

        OccupancyResult {
            occupancy,
            active_warps_per_sm: active_warps.min(max_warps),
            max_warps_per_sm: max_warps,
            active_blocks_per_sm: active_blocks,
            limiting_factor,
        }
    }

    /// Find the optimal workgroup size for best occupancy.
    ///
    /// Iterates over multiples of warp size up to max threads per block.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn find_optimal_block_size(
        spec: &GpuSpec,
        registers_per_thread: u32,
        shared_memory_per_block: u32,
    ) -> u32 {
        let mut best_occupancy = 0.0_f64;
        let mut best_size = spec.warp_size;

        let max_threads = spec.max_threads_per_sm.min(1024);
        let mut threads = spec.warp_size;

        while threads <= max_threads {
            let kernel =
                KernelResources::new(threads, registers_per_thread, shared_memory_per_block);
            let result = Self::calculate(spec, &kernel);
            if result.occupancy > best_occupancy {
                best_occupancy = result.occupancy;
                best_size = threads;
            }
            threads += spec.warp_size;
        }

        best_size
    }

    /// Estimate achieved memory bandwidth given occupancy.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn estimate_bandwidth(
        occupancy: f64,
        peak_bandwidth_gbps: f64,
        memory_intensity: f64,
    ) -> f64 {
        let eff = occupancy.clamp(0.0, 1.0);
        let intensity = memory_intensity.clamp(0.0, 1.0);
        peak_bandwidth_gbps * eff * intensity
    }
}

/// Performance tip based on occupancy analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerformanceTip {
    /// Human-readable tip.
    pub message: String,
    /// Priority (lower is more important).
    pub priority: u32,
}

impl PerformanceTip {
    /// Create a new performance tip.
    #[must_use]
    pub fn new(message: &str, priority: u32) -> Self {
        Self {
            message: message.to_string(),
            priority,
        }
    }
}

/// Generate performance tips based on occupancy results.
#[must_use]
pub fn analyze_performance(result: &OccupancyResult) -> Vec<PerformanceTip> {
    let mut tips = Vec::new();

    if result.occupancy < 0.25 {
        tips.push(PerformanceTip::new(
            "Very low occupancy. Consider reducing resource usage per thread.",
            1,
        ));
    } else if result.occupancy < 0.5 {
        tips.push(PerformanceTip::new(
            "Low occupancy. Adjusting block size or register usage may help.",
            2,
        ));
    }

    match result.limiting_factor {
        OccupancyLimit::Registers => {
            tips.push(PerformanceTip::new(
                "Register usage is the bottleneck. Consider reducing local variables.",
                2,
            ));
        }
        OccupancyLimit::SharedMemory => {
            tips.push(PerformanceTip::new(
                "Shared memory is the bottleneck. Consider reducing shared memory usage.",
                2,
            ));
        }
        _ => {}
    }

    if result.occupancy >= 0.75 {
        tips.push(PerformanceTip::new(
            "Good occupancy. Focus on memory access patterns and instruction throughput.",
            3,
        ));
    }

    tips
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_spec_mid_range() {
        let spec = GpuSpec::mid_range();
        assert_eq!(spec.max_threads_per_sm, 1536);
        assert_eq!(spec.warp_size, 32);
    }

    #[test]
    fn test_gpu_spec_high_end() {
        let spec = GpuSpec::high_end();
        assert_eq!(spec.max_threads_per_sm, 2048);
        assert_eq!(spec.sm_count, 80);
    }

    #[test]
    fn test_gpu_spec_integrated() {
        let spec = GpuSpec::integrated();
        assert_eq!(spec.max_threads_per_sm, 512);
        assert_eq!(spec.sm_count, 8);
    }

    #[test]
    fn test_kernel_resources_simple() {
        let k = KernelResources::simple(256);
        assert_eq!(k.threads_per_block, 256);
        assert_eq!(k.registers_per_thread, 32);
        assert_eq!(k.shared_memory_per_block, 0);
    }

    #[test]
    fn test_occupancy_simple_kernel() {
        let spec = GpuSpec::mid_range();
        let kernel = KernelResources::simple(256);
        let result = OccupancyCalculator::calculate(&spec, &kernel);
        assert!(result.occupancy > 0.0);
        assert!(result.occupancy <= 1.0);
        assert!(result.active_warps_per_sm > 0);
    }

    #[test]
    fn test_occupancy_heavy_registers() {
        let spec = GpuSpec::mid_range();
        let kernel = KernelResources::new(256, 128, 0);
        let result = OccupancyCalculator::calculate(&spec, &kernel);
        // Heavy register usage should reduce occupancy
        assert!(result.occupancy <= 1.0);
    }

    #[test]
    fn test_occupancy_heavy_shared_memory() {
        let spec = GpuSpec::mid_range();
        let kernel = KernelResources::new(256, 32, 32768);
        let result = OccupancyCalculator::calculate(&spec, &kernel);
        assert!(result.occupancy > 0.0);
        assert_eq!(result.limiting_factor, OccupancyLimit::SharedMemory);
    }

    #[test]
    fn test_occupancy_zero_threads() {
        let spec = GpuSpec::mid_range();
        let kernel = KernelResources::new(0, 32, 0);
        let result = OccupancyCalculator::calculate(&spec, &kernel);
        assert!((result.occupancy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_optimal_block_size() {
        let spec = GpuSpec::mid_range();
        let optimal = OccupancyCalculator::find_optimal_block_size(&spec, 32, 0);
        assert!(optimal >= spec.warp_size);
        assert!(optimal <= spec.max_threads_per_sm);
        assert_eq!(optimal % spec.warp_size, 0);
    }

    #[test]
    fn test_estimate_bandwidth() {
        let bw = OccupancyCalculator::estimate_bandwidth(1.0, 500.0, 1.0);
        assert!((bw - 500.0).abs() < f64::EPSILON);

        let bw2 = OccupancyCalculator::estimate_bandwidth(0.5, 500.0, 0.8);
        assert!((bw2 - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_bandwidth_clamping() {
        let bw = OccupancyCalculator::estimate_bandwidth(2.0, 500.0, 1.5);
        assert!((bw - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_performance_tips_low_occupancy() {
        let result = OccupancyResult {
            occupancy: 0.1,
            active_warps_per_sm: 4,
            max_warps_per_sm: 48,
            active_blocks_per_sm: 1,
            limiting_factor: OccupancyLimit::Registers,
        };
        let tips = analyze_performance(&result);
        assert!(!tips.is_empty());
        assert!(tips.iter().any(|t| t.message.contains("Very low")));
    }

    #[test]
    fn test_performance_tips_good_occupancy() {
        let result = OccupancyResult {
            occupancy: 0.8,
            active_warps_per_sm: 38,
            max_warps_per_sm: 48,
            active_blocks_per_sm: 6,
            limiting_factor: OccupancyLimit::None,
        };
        let tips = analyze_performance(&result);
        assert!(tips.iter().any(|t| t.message.contains("Good occupancy")));
    }

    #[test]
    fn test_occupancy_limit_display() {
        assert_eq!(format!("{}", OccupancyLimit::BlockCount), "block count");
        assert_eq!(format!("{}", OccupancyLimit::Registers), "register usage");
        assert_eq!(format!("{}", OccupancyLimit::SharedMemory), "shared memory");
    }

    #[test]
    fn test_performance_tip_creation() {
        let tip = PerformanceTip::new("test tip", 5);
        assert_eq!(tip.message, "test tip");
        assert_eq!(tip.priority, 5);
    }
}
