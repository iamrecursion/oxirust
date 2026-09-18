//! CPU profiling utilities for benchmarks.
//!
//! Provides types for CPU performance counters, CPU tier classification,
//! and CPU information used during benchmarking.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Hardware performance counters captured during a benchmark run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuCounters {
    /// Number of instructions retired.
    pub instructions: u64,
    /// Number of CPU cycles elapsed.
    pub cycles: u64,
    /// Number of last-level cache misses.
    pub cache_misses: u64,
    /// Number of branch mispredictions.
    pub branch_misses: u64,
}

impl CpuCounters {
    /// Create a new counter set.
    #[must_use]
    pub fn new(instructions: u64, cycles: u64, cache_misses: u64, branch_misses: u64) -> Self {
        Self {
            instructions,
            cycles,
            cache_misses,
            branch_misses,
        }
    }

    /// Instructions per cycle (IPC).
    ///
    /// Returns `0.0` when `cycles` is zero.
    #[must_use]
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 {
            return 0.0;
        }
        self.instructions as f64 / self.cycles as f64
    }

    /// Cache miss rate: `cache_misses / instructions`.
    ///
    /// Returns `0.0` when `instructions` is zero.
    #[must_use]
    pub fn cache_miss_rate(&self) -> f64 {
        if self.instructions == 0 {
            return 0.0;
        }
        self.cache_misses as f64 / self.instructions as f64
    }
}

/// Broad classification of CPU capability for benchmarking context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuTier {
    /// Entry-level single-core or mobile processors.
    SingleCore,
    /// Standard desktop multi-core processors.
    MultiCore,
    /// High-performance desktop or workstation processors.
    HighPerf,
    /// Server-grade many-core processors.
    Server,
}

impl CpuTier {
    /// Typical base clock frequency in GHz for this tier.
    #[must_use]
    pub fn typical_ghz(self) -> f32 {
        match self {
            CpuTier::SingleCore => 1.5,
            CpuTier::MultiCore => 3.0,
            CpuTier::HighPerf => 4.0,
            CpuTier::Server => 2.5,
        }
    }
}

/// Information about the CPU on which the benchmark is running.
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Human-readable CPU name.
    pub name: String,
    /// Number of physical (non-hyperthreaded) cores.
    pub physical_cores: u32,
    /// Number of logical cores (including hyperthreads).
    pub logical_cores: u32,
    /// Base clock frequency in MHz.
    pub base_freq_mhz: u32,
    /// Tier classification.
    pub tier: CpuTier,
}

impl CpuInfo {
    /// Create a new `CpuInfo`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        physical_cores: u32,
        logical_cores: u32,
        base_freq_mhz: u32,
        tier: CpuTier,
    ) -> Self {
        Self {
            name: name.into(),
            physical_cores,
            logical_cores,
            base_freq_mhz,
            tier,
        }
    }

    /// Returns `true` when `logical_cores > physical_cores` (i.e., SMT/HT is active).
    #[must_use]
    pub fn is_hyperthreaded(&self) -> bool {
        self.logical_cores > self.physical_cores
    }

    /// Aggregate throughput in GHz: `physical_cores * (base_freq_mhz / 1000)`.
    #[must_use]
    pub fn total_ghz(&self) -> f32 {
        self.physical_cores as f32 * (self.base_freq_mhz as f32 / 1000.0)
    }
}

/// Return a synthetic CPU with known values suitable for use in unit tests.
#[must_use]
pub fn mock_cpu_info() -> CpuInfo {
    CpuInfo::new(
        "Test CPU (Mock)",
        4,     // physical cores
        8,     // logical cores (hyperthreaded)
        3_200, // base freq MHz
        CpuTier::MultiCore,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- CpuCounters tests -----

    #[test]
    fn test_ipc_normal() {
        let counters = CpuCounters::new(2_000_000, 1_000_000, 500, 100);
        assert!((counters.ipc() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_ipc_zero_cycles() {
        let counters = CpuCounters::new(1_000, 0, 0, 0);
        assert_eq!(counters.ipc(), 0.0);
    }

    #[test]
    fn test_cache_miss_rate_normal() {
        let counters = CpuCounters::new(1_000_000, 500_000, 1_000, 0);
        // 1_000 / 1_000_000 = 0.001
        assert!((counters.cache_miss_rate() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_cache_miss_rate_zero_instructions() {
        let counters = CpuCounters::new(0, 0, 0, 0);
        assert_eq!(counters.cache_miss_rate(), 0.0);
    }

    // ----- CpuTier tests -----

    #[test]
    fn test_cpu_tier_single_core_ghz() {
        assert!((CpuTier::SingleCore.typical_ghz() - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_tier_multi_core_ghz() {
        assert!((CpuTier::MultiCore.typical_ghz() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_tier_high_perf_ghz() {
        assert!((CpuTier::HighPerf.typical_ghz() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_tier_server_ghz() {
        assert!((CpuTier::Server.typical_ghz() - 2.5).abs() < 1e-5);
    }

    // ----- CpuInfo tests -----

    #[test]
    fn test_cpu_info_is_hyperthreaded_true() {
        let cpu = CpuInfo::new("HT CPU", 4, 8, 3000, CpuTier::MultiCore);
        assert!(cpu.is_hyperthreaded());
    }

    #[test]
    fn test_cpu_info_is_hyperthreaded_false() {
        let cpu = CpuInfo::new("No HT CPU", 4, 4, 3000, CpuTier::MultiCore);
        assert!(!cpu.is_hyperthreaded());
    }

    #[test]
    fn test_cpu_info_total_ghz() {
        let cpu = CpuInfo::new("Test", 4, 8, 3_000, CpuTier::MultiCore);
        // 4 cores * 3.0 GHz = 12.0 GHz
        assert!((cpu.total_ghz() - 12.0_f32).abs() < 1e-4);
    }

    #[test]
    fn test_cpu_info_name() {
        let cpu = CpuInfo::new("My CPU", 2, 4, 2_500, CpuTier::MultiCore);
        assert_eq!(cpu.name, "My CPU");
    }

    // ----- mock_cpu_info tests -----

    #[test]
    fn test_mock_cpu_info_hyperthreaded() {
        let cpu = mock_cpu_info();
        assert!(cpu.is_hyperthreaded());
    }

    #[test]
    fn test_mock_cpu_info_tier() {
        let cpu = mock_cpu_info();
        assert_eq!(cpu.tier, CpuTier::MultiCore);
    }

    #[test]
    fn test_mock_cpu_info_total_ghz() {
        let cpu = mock_cpu_info();
        // 4 cores * 3.2 GHz = 12.8 GHz
        assert!((cpu.total_ghz() - 12.8_f32).abs() < 0.01);
    }
}
