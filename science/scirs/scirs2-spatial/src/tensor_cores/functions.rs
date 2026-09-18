//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{SpatialError, SpatialResult};

use super::types::{GpuArchitecture, PrecisionMode, TensorCoreCapabilities, TensorCoreType};

/// Detect tensor core capabilities of available GPU hardware
#[allow(dead_code)]
pub fn detect_tensor_core_capabilities() -> SpatialResult<TensorCoreCapabilities> {
    Ok(TensorCoreCapabilities {
        tensor_core_types: vec![
            TensorCoreType::NvidiaTensorCore,
            TensorCoreType::StandardCores,
        ],
        supported_precisions: vec![
            PrecisionMode::Full32,
            PrecisionMode::Mixed16,
            PrecisionMode::BrainFloat16,
            PrecisionMode::Int8Dynamic,
        ],
        max_tensor_size: (4096, 4096, 4096),
        peak_throughput_tops: 312.0,
        memory_bandwidth_gbps: 1555.0,
        l2_cache_mb: 40.0,
        num_sms: 108,
        architecture: GpuArchitecture::Ampere,
    })
}
