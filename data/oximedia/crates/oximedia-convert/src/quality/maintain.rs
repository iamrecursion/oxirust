// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Quality maintenance during conversion.

use crate::{ConversionError, QualityComparison, Result};
use std::path::Path;

/// Quality maintainer for preserving quality during conversion.
#[derive(Debug, Clone)]
pub struct QualityMaintainer {
    target_quality: QualityTarget,
    enable_comparison: bool,
}

impl QualityMaintainer {
    /// Create a new quality maintainer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_quality: QualityTarget::Balanced,
            enable_comparison: false,
        }
    }

    /// Set the target quality level.
    #[must_use]
    pub fn with_target(mut self, target: QualityTarget) -> Self {
        self.target_quality = target;
        self
    }

    /// Enable quality comparison.
    #[must_use]
    pub fn with_comparison(mut self, enabled: bool) -> Self {
        self.enable_comparison = enabled;
        self
    }

    /// Get recommended CRF value for video encoding.
    #[must_use]
    pub fn get_crf_value(&self, codec: &str) -> u32 {
        match (codec, &self.target_quality) {
            ("h264", QualityTarget::Best) => 18,
            ("h264", QualityTarget::Balanced) => 23,
            ("h264", QualityTarget::Fast) => 28,
            ("h265", QualityTarget::Best) => 20,
            ("h265", QualityTarget::Balanced) => 26,
            ("h265", QualityTarget::Fast) => 30,
            ("vp9", QualityTarget::Best) => 30,
            ("vp9", QualityTarget::Balanced) => 35,
            ("vp9", QualityTarget::Fast) => 40,
            ("av1", QualityTarget::Best) => 30,
            ("av1", QualityTarget::Balanced) => 35,
            ("av1", QualityTarget::Fast) => 40,
            _ => 23, // default
        }
    }

    /// Get recommended audio bitrate.
    #[must_use]
    pub fn get_audio_bitrate(&self, codec: &str, channels: u32) -> u64 {
        let base = match codec {
            "aac" => match &self.target_quality {
                QualityTarget::Best => 256_000,
                QualityTarget::Balanced => 192_000,
                QualityTarget::Fast => 128_000,
            },
            "mp3" => match &self.target_quality {
                QualityTarget::Best => 320_000,
                QualityTarget::Balanced => 192_000,
                QualityTarget::Fast => 128_000,
            },
            "opus" => match &self.target_quality {
                QualityTarget::Best => 256_000,
                QualityTarget::Balanced => 128_000,
                QualityTarget::Fast => 96_000,
            },
            _ => 192_000,
        };

        // Adjust for channels
        if channels > 2 {
            (base as f64 * 1.5) as u64
        } else {
            base
        }
    }

    /// Get recommended video bitrate based on resolution.
    #[must_use]
    pub fn get_video_bitrate(&self, width: u32, height: u32, fps: f64) -> u64 {
        let pixels = u64::from(width) * u64::from(height);
        let base_multiplier = match &self.target_quality {
            QualityTarget::Best => 0.15,
            QualityTarget::Balanced => 0.10,
            QualityTarget::Fast => 0.07,
        };

        let fps_factor = if fps > 30.0 { 1.3 } else { 1.0 };
        let bitrate = (pixels as f64 * fps * base_multiplier * fps_factor) as u64;

        // Clamp to reasonable values
        bitrate.max(500_000).min(50_000_000)
    }

    /// Compare quality between two files.
    pub fn compare<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        original: P,
        converted: Q,
    ) -> Result<QualityComparison> {
        let original_path = original.as_ref();
        let converted_path = converted.as_ref();

        if !original_path.exists() {
            return Err(ConversionError::InvalidInput(
                "Original file not found".to_string(),
            ));
        }

        if !converted_path.exists() {
            return Err(ConversionError::InvalidOutput(
                "Converted file not found".to_string(),
            ));
        }

        let original_size = std::fs::metadata(original_path)
            .map_err(ConversionError::Io)?
            .len();

        let converted_size = std::fs::metadata(converted_path)
            .map_err(ConversionError::Io)?
            .len();

        // Placeholder for actual quality metrics
        // In a real implementation, this would use metrics like PSNR, SSIM, VMAF
        Ok(QualityComparison {
            original_size,
            converted_size,
            size_reduction_percent: calculate_size_reduction(original_size, converted_size),
            psnr: None,
            ssim: None,
            vmaf: None,
        })
    }

    /// Validate that quality target is achievable.
    pub fn validate_target(&self, source_bitrate: u64, target_bitrate: u64) -> Result<()> {
        if target_bitrate > source_bitrate * 2 {
            return Err(ConversionError::QualityPreservation(
                "Target bitrate is unreasonably high".to_string(),
            ));
        }

        if target_bitrate < 100_000 {
            return Err(ConversionError::QualityPreservation(
                "Target bitrate is too low for acceptable quality".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for QualityMaintainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Target quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityTarget {
    /// Best quality, larger file size
    Best,
    /// Balanced quality and size
    Balanced,
    /// Lower quality, faster encoding
    Fast,
}

fn calculate_size_reduction(original: u64, converted: u64) -> f64 {
    if original == 0 {
        return 0.0;
    }

    ((original as i64 - converted as i64) as f64 / original as f64) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_maintainer_creation() {
        let maintainer = QualityMaintainer::new();
        assert_eq!(maintainer.target_quality, QualityTarget::Balanced);
    }

    #[test]
    fn test_crf_values() {
        let maintainer = QualityMaintainer::new().with_target(QualityTarget::Best);

        assert_eq!(maintainer.get_crf_value("h264"), 18);
        assert_eq!(maintainer.get_crf_value("h265"), 20);
        assert_eq!(maintainer.get_crf_value("vp9"), 30);
    }

    #[test]
    fn test_audio_bitrate() {
        let maintainer = QualityMaintainer::new();

        assert_eq!(maintainer.get_audio_bitrate("aac", 2), 192_000);
        assert_eq!(maintainer.get_audio_bitrate("mp3", 2), 192_000);

        // Multichannel should be higher
        assert!(maintainer.get_audio_bitrate("aac", 6) > 192_000);
    }

    #[test]
    fn test_video_bitrate() {
        let maintainer = QualityMaintainer::new();

        // 1080p at 30fps
        let bitrate_1080p = maintainer.get_video_bitrate(1920, 1080, 30.0);
        assert!(bitrate_1080p > 2_000_000);
        assert!(bitrate_1080p < 10_000_000);

        // 720p should be lower
        let bitrate_720p = maintainer.get_video_bitrate(1280, 720, 30.0);
        assert!(bitrate_720p < bitrate_1080p);
    }

    #[test]
    fn test_validate_target() {
        let maintainer = QualityMaintainer::new();

        assert!(maintainer.validate_target(5_000_000, 3_000_000).is_ok());
        assert!(maintainer.validate_target(5_000_000, 50_000).is_err());
        assert!(maintainer.validate_target(1_000_000, 5_000_000).is_err());
    }

    #[test]
    fn test_size_reduction() {
        assert_eq!(calculate_size_reduction(1000, 500), 50.0);
        assert_eq!(calculate_size_reduction(1000, 750), 25.0);
        assert_eq!(calculate_size_reduction(1000, 1000), 0.0);
    }
}
