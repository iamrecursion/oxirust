// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Conversion pipeline for processing media files.
//!
//! This module provides the core conversion pipeline that orchestrates
//! decoding, filtering, encoding, and muxing operations.

pub mod executor;
pub mod job;
pub mod options;

use crate::formats::{AudioCodec, ChannelLayout, VideoCodec};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

pub use executor::PipelineExecutor;
pub use job::{ConversionJob, JobPriority, JobStatus};
pub use options::{AudioOptions, BitrateMode, VideoOptions};

/// Video conversion settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoSettings {
    /// Video codec
    pub codec: VideoCodec,
    /// Target resolution (width, height)
    pub resolution: Option<(u32, u32)>,
    /// Frame rate
    pub frame_rate: Option<f64>,
    /// Bitrate mode and value
    pub bitrate: BitrateMode,
    /// Quality preset (0-63 for VP8/VP9, 0-255 for AV1)
    pub quality: Option<u32>,
    /// Two-pass encoding
    pub two_pass: bool,
    /// Encoding speed preset
    pub speed: EncodingSpeed,
    /// HDR to SDR tone mapping
    pub tone_map: bool,
}

/// Audio conversion settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioSettings {
    /// Audio codec
    pub codec: AudioCodec,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Channel layout
    pub channels: ChannelLayout,
    /// Bitrate in bits per second (for lossy codecs)
    pub bitrate: Option<u64>,
    /// Volume normalization
    pub normalize: bool,
    /// Normalization target in dB/LUFS
    pub normalization_target: f64,
}

/// Encoding speed preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingSpeed {
    /// Fastest encoding
    Fast,
    /// Balanced speed
    Medium,
    /// Slower, better quality
    Slow,
    /// Slowest, best quality
    VerySlow,
}

/// Pipeline statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Input file size in bytes
    pub input_size: u64,
    /// Output file size in bytes
    pub output_size: u64,
    /// Processing duration
    pub duration: Duration,
    /// Encoding speed (frames per second)
    pub encoding_fps: f64,
    /// Number of frames processed
    pub frames_processed: u64,
}

impl PipelineStats {
    /// Calculate compression ratio.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.output_size == 0 {
            0.0
        } else {
            self.input_size as f64 / self.output_size as f64
        }
    }

    /// Calculate space savings as percentage.
    #[must_use]
    pub fn space_savings(&self) -> f64 {
        if self.input_size == 0 {
            0.0
        } else {
            ((self.input_size - self.output_size) as f64 / self.input_size as f64) * 100.0
        }
    }
}

/// Conversion pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of parallel workers
    pub workers: usize,
    /// Buffer size for processing
    pub buffer_size: usize,
    /// Enable hardware acceleration
    pub hardware_accel: bool,
    /// Temporary directory for intermediate files
    pub temp_dir: Option<PathBuf>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            workers: num_cpus(),
            buffer_size: 8 * 1024 * 1024, // 8 MB
            hardware_accel: false,
            temp_dir: None,
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.workers > 0);
        assert_eq!(config.buffer_size, 8 * 1024 * 1024);
        assert!(!config.hardware_accel);
    }

    #[test]
    fn test_pipeline_stats() {
        let stats = PipelineStats {
            input_size: 1000,
            output_size: 500,
            duration: Duration::from_secs(10),
            encoding_fps: 30.0,
            frames_processed: 300,
        };

        assert_eq!(stats.compression_ratio(), 2.0);
        assert_eq!(stats.space_savings(), 50.0);
    }

    #[test]
    fn test_pipeline_stats_zero_output() {
        let stats = PipelineStats {
            input_size: 1000,
            output_size: 0,
            duration: Duration::from_secs(10),
            encoding_fps: 30.0,
            frames_processed: 300,
        };

        assert_eq!(stats.compression_ratio(), 0.0);
    }
}
