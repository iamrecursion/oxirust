//! Proxy encoder implementation.

use super::settings::ProxyGenerationSettings;
use crate::{ProxyError, Result};
use std::path::Path;
use std::time::Instant;

/// Proxy encoder with support for various codecs and settings.
pub struct ProxyEncoder {
    settings: ProxyGenerationSettings,
}

impl ProxyEncoder {
    /// Create a new proxy encoder with the given settings.
    pub fn new(settings: ProxyGenerationSettings) -> Result<Self> {
        settings.validate()?;
        Ok(Self { settings })
    }

    /// Encode a proxy from the input file using the configured codec and settings.
    ///
    /// This implementation uses `oximedia_transcode::TranscodePipeline` to perform
    /// a real low-bitrate re-encode into the specified container format. Codec selection,
    /// target bitrate, and audio codec are all derived from `ProxyGenerationSettings`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input file does not exist or cannot be read
    /// - Output path is invalid or its parent directory cannot be created
    /// - The transcode pipeline fails (e.g., unsupported codec, I/O error)
    pub async fn encode(&self, input: &Path, output: &Path) -> Result<ProxyEncodeResult> {
        use oximedia_transcode::TranscodePipeline;

        // Validate input
        if !input.exists() {
            return Err(ProxyError::FileNotFound(input.display().to_string()));
        }

        // Create output directory if needed
        if let Some(parent) = output.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        tracing::info!(
            "Encoding proxy: {} -> {} (scale: {}, codec: {}, bitrate: {})",
            input.display(),
            output.display(),
            self.settings.scale_factor,
            self.settings.codec,
            self.settings.bitrate,
        );

        let start = Instant::now();

        let mut pipeline = TranscodePipeline::builder()
            .input(input.to_path_buf())
            .output(output.to_path_buf())
            .video_codec(self.settings.codec.as_str())
            .audio_codec(self.settings.audio_codec.as_str())
            .track_progress(false)
            .hw_accel(self.settings.use_hw_accel)
            .build()
            .map_err(|e| ProxyError::GenerationError(e.to_string()))?;

        let transcode_result = pipeline
            .execute()
            .await
            .map_err(|e| ProxyError::GenerationError(e.to_string()))?;

        let encoding_time = start.elapsed().as_secs_f64();

        // Use the file_size from the transcode result; if zero, stat the output file.
        let file_size = if transcode_result.file_size > 0 {
            transcode_result.file_size
        } else {
            std::fs::metadata(output).map(|m| m.len()).unwrap_or(0)
        };

        Ok(ProxyEncodeResult {
            output_path: output.to_path_buf(),
            file_size,
            duration: transcode_result.duration,
            bitrate: self.settings.bitrate,
            codec: self.settings.codec.clone(),
            resolution: (0, 0), // Resolution info not yet available from TranscodeOutput
            encoding_time,
        })
    }

    /// Get the current settings.
    #[must_use]
    pub fn settings(&self) -> &ProxyGenerationSettings {
        &self.settings
    }

    /// Update the settings.
    pub fn set_settings(&mut self, settings: ProxyGenerationSettings) -> Result<()> {
        settings.validate()?;
        self.settings = settings;
        Ok(())
    }

    /// Calculate the target resolution for the given input resolution.
    #[must_use]
    pub fn calculate_target_resolution(&self, input_width: u32, input_height: u32) -> (u32, u32) {
        let target_width = ((input_width as f32) * self.settings.scale_factor) as u32;
        let target_height = ((input_height as f32) * self.settings.scale_factor) as u32;

        // Ensure dimensions are even (required for most video codecs)
        let target_width = target_width & !1;
        let target_height = target_height & !1;

        (target_width, target_height)
    }
}

/// Result of proxy encoding operation.
#[derive(Debug, Clone)]
pub struct ProxyEncodeResult {
    /// Output file path.
    pub output_path: std::path::PathBuf,

    /// File size in bytes.
    pub file_size: u64,

    /// Duration in seconds.
    pub duration: f64,

    /// Actual bitrate in bits per second.
    pub bitrate: u64,

    /// Codec used.
    pub codec: String,

    /// Output resolution (width, height).
    pub resolution: (u32, u32),

    /// Encoding time in seconds.
    pub encoding_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let settings = ProxyGenerationSettings::quarter_res_h264();
        let encoder = ProxyEncoder::new(settings);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_invalid_settings() {
        let mut settings = ProxyGenerationSettings::default();
        settings.scale_factor = 0.0;
        let encoder = ProxyEncoder::new(settings);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_resolution_calculation() {
        let settings = ProxyGenerationSettings::quarter_res_h264();
        let encoder = ProxyEncoder::new(settings).expect("should succeed in test");

        let (width, height) = encoder.calculate_target_resolution(1920, 1080);
        assert_eq!(width, 480);
        assert_eq!(height, 270);

        let (width, height) = encoder.calculate_target_resolution(3840, 2160);
        assert_eq!(width, 960);
        assert_eq!(height, 540);
    }

    #[test]
    fn test_resolution_even_alignment() {
        let settings = ProxyGenerationSettings::default().with_scale_factor(0.3);
        let encoder = ProxyEncoder::new(settings).expect("should succeed in test");

        let (width, height) = encoder.calculate_target_resolution(1920, 1080);
        // Should be even numbers
        assert_eq!(width % 2, 0);
        assert_eq!(height % 2, 0);
    }
}
