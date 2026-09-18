//! Hardware-adaptive optimizer for FFT operations

use super::types::*;
use crate::error::FFTResult;
use std::collections::HashMap;

/// Hardware-adaptive optimizer
#[derive(Debug)]
#[allow(dead_code)]
pub struct HardwareAdaptiveOptimizer {
    /// Detected hardware capabilities
    pub hardware_capabilities: HardwareCapabilities,
    /// Optimization profiles
    pub(crate) optimization_profiles: HashMap<HardwareProfile, OptimizationProfile>,
    /// Current active profile
    pub(crate) active_profile: Option<HardwareProfile>,
}

/// Hardware capabilities detection
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// CPU information
    pub cpu_info: CpuInfo,
    /// GPU information (if available)
    pub gpu_info: Option<GpuInfo>,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// SIMD support level
    pub simd_support: SimdSupport,
}

/// CPU information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Number of cores
    pub core_count: usize,
    /// Cache sizes (L1, L2, L3)
    pub cache_sizes: Vec<usize>,
    /// CPU frequency (MHz)
    pub frequency_mhz: u32,
    /// Architecture type
    pub architecture: String,
}

/// GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU memory (MB)
    pub memory_mb: usize,
    /// Compute capability
    pub compute_capability: String,
    /// Number of streaming multiprocessors
    pub sm_count: usize,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f64,
}

/// Memory information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total system memory (MB)
    pub total_mb: usize,
    /// Available memory (MB)
    pub available_mb: usize,
    /// Memory bandwidth (GB/s)
    pub bandwidth_gbs: f64,
}

/// Optimization profile for hardware
#[derive(Debug, Clone)]
pub struct OptimizationProfile {
    /// Preferred algorithms
    pub preferred_algorithms: Vec<FftAlgorithmType>,
    /// Memory allocation strategy
    pub memory_strategy: MemoryAllocationStrategy,
    /// Parallelism configuration
    pub parallelism_config: ParallelismConfig,
    /// SIMD configuration
    pub simd_config: SimdConfig,
}

impl HardwareAdaptiveOptimizer {
    pub fn new() -> FFTResult<Self> {
        Ok(Self {
            hardware_capabilities: HardwareCapabilities::detect()?,
            optimization_profiles: HashMap::new(),
            active_profile: None,
        })
    }
}

impl HardwareCapabilities {
    pub fn detect() -> FFTResult<Self> {
        Ok(Self {
            cpu_info: CpuInfo::detect()?,
            gpu_info: GpuInfo::detect(),
            memory_info: MemoryInfo::detect()?,
            simd_support: SimdSupport::detect()?,
        })
    }
}

impl CpuInfo {
    pub fn detect() -> FFTResult<Self> {
        Ok(Self {
            core_count: num_cpus::get(),
            cache_sizes: vec![32768, 262144, 8388608], // Default L1, L2, L3
            frequency_mhz: 2400,                       // Default frequency
            architecture: "x86_64".to_string(),
        })
    }
}

impl GpuInfo {
    pub fn detect() -> Option<Self> {
        None
    }
}

impl MemoryInfo {
    pub fn detect() -> FFTResult<Self> {
        Ok(Self {
            total_mb: 8192,      // Default 8GB
            available_mb: 4096,  // Default 4GB available
            bandwidth_gbs: 25.6, // Default bandwidth
        })
    }
}

impl SimdSupport {
    pub fn detect() -> FFTResult<Self> {
        Ok(SimdSupport::AVX2)
    }
}
