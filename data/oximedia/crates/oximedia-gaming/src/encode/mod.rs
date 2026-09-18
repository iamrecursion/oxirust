//! Ultra-low latency encoding for game streaming.
//!
//! This module provides hardware-accelerated encoding with minimal latency:
//!
//! - Low-latency encoding pipeline
//! - NVIDIA NVENC support
//! - Intel Quick Sync Video support
//! - AMD VCE support

pub mod lowlatency;
pub mod nvenc;
pub mod qsv;
pub mod vce;

pub use lowlatency::{EncoderConfig, LatencyMode, LowLatencyEncoder};
pub use nvenc::{NvencConfig, NvencEncoder, NvencPreset};
pub use qsv::{QsvConfig, QsvEncoder, QsvPreset};
pub use vce::{VceConfig, VceEncoder, VcePreset};
