//! Proxy generation optimizer for automatic settings adjustment.

use super::settings::ProxyGenerationSettings;
use crate::Result;

/// Proxy generation optimizer.
pub struct ProxyOptimizer {
    /// Target file size in bytes (optional).
    target_size: Option<u64>,

    /// Target bitrate in bits per second (optional).
    target_bitrate: Option<u64>,

    /// Maximum encoding time in seconds (optional).
    max_encoding_time: Option<f64>,
}

impl ProxyOptimizer {
    /// Create a new proxy optimizer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            target_size: None,
            target_bitrate: None,
            max_encoding_time: None,
        }
    }

    /// Set target file size.
    #[must_use]
    pub const fn with_target_size(mut self, size: u64) -> Self {
        self.target_size = Some(size);
        self
    }

    /// Set target bitrate.
    #[must_use]
    pub const fn with_target_bitrate(mut self, bitrate: u64) -> Self {
        self.target_bitrate = Some(bitrate);
        self
    }

    /// Set maximum encoding time.
    #[must_use]
    pub const fn with_max_encoding_time(mut self, time: f64) -> Self {
        self.max_encoding_time = Some(time);
        self
    }

    /// Optimize settings for the given constraints.
    pub fn optimize(
        &self,
        base_settings: ProxyGenerationSettings,
        input_duration: f64,
    ) -> Result<ProxyGenerationSettings> {
        let mut settings = base_settings;

        // Optimize for target bitrate
        if let Some(target_bitrate) = self.target_bitrate {
            settings.bitrate = target_bitrate;
        }

        // Optimize for target file size
        if let Some(target_size) = self.target_size {
            // Calculate required bitrate: (target_size * 8) / duration
            let required_bitrate = (target_size as f64 * 8.0 / input_duration) as u64;

            // Reserve 10% for audio and container overhead
            settings.bitrate = (required_bitrate as f64 * 0.9) as u64;
            settings.audio_bitrate = (required_bitrate as f64 * 0.1) as u64;
        }

        // Optimize for encoding time
        if let Some(_max_time) = self.max_encoding_time {
            // Use faster encoding presets for time constraints
            settings.quality_preset = "ultrafast".to_string();
            settings.threads = num_cpus();
        }

        settings.validate()?;
        Ok(settings)
    }

    /// Estimate output size for given settings.
    #[must_use]
    pub fn estimate_output_size(&self, settings: &ProxyGenerationSettings, duration: f64) -> u64 {
        // Video size: (bitrate * duration) / 8
        let video_size = (settings.bitrate as f64 * duration / 8.0) as u64;

        // Audio size: (audio_bitrate * duration) / 8
        let audio_size = (settings.audio_bitrate as f64 * duration / 8.0) as u64;

        // Container overhead (approximately 5%)
        let overhead = ((video_size + audio_size) as f64 * 0.05) as u64;

        video_size + audio_size + overhead
    }

    /// Estimate encoding time for given settings.
    #[must_use]
    pub fn estimate_encoding_time(&self, settings: &ProxyGenerationSettings, duration: f64) -> f64 {
        // Base encoding speed (realtime factor)
        let base_speed = match settings.quality_preset.as_str() {
            "ultrafast" => 10.0, // 10x realtime
            "veryfast" => 5.0,   // 5x realtime
            "fast" => 3.0,       // 3x realtime
            "medium" => 1.5,     // 1.5x realtime
            "slow" => 0.5,       // 0.5x realtime
            _ => 1.0,            // 1x realtime
        };

        // Adjust for scale factor (smaller = faster)
        let scale_adjustment = 1.0 / (settings.scale_factor as f64);

        // Adjust for threads
        let thread_adjustment: f64 = if settings.threads == 0 {
            num_cpus::get() as f64
        } else {
            settings.threads as f64
        };

        // Calculate estimated time
        let denominator = base_speed * thread_adjustment / scale_adjustment;
        duration / denominator
    }
}

impl Default for ProxyOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(dead_code)]
fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = ProxyOptimizer::new()
            .with_target_size(100_000_000)
            .with_target_bitrate(5_000_000);

        assert_eq!(optimizer.target_size, Some(100_000_000));
        assert_eq!(optimizer.target_bitrate, Some(5_000_000));
    }

    #[test]
    fn test_optimize_for_size() {
        let optimizer = ProxyOptimizer::new().with_target_size(50_000_000); // 50 MB

        let base_settings = ProxyGenerationSettings::quarter_res_h264();
        let optimized = optimizer
            .optimize(base_settings, 60.0)
            .expect("should succeed in test");

        // Check that bitrate was adjusted for target size
        assert!(optimized.bitrate > 0);
    }

    #[test]
    fn test_estimate_output_size() {
        let optimizer = ProxyOptimizer::new();
        let settings = ProxyGenerationSettings::quarter_res_h264();

        let estimated_size = optimizer.estimate_output_size(&settings, 60.0);
        assert!(estimated_size > 0);

        // For 60 seconds at 2 Mbps video + 128 kbps audio
        // Should be approximately: (2000000 + 128000) * 60 / 8 = ~16 MB
        assert!(estimated_size > 10_000_000);
        assert!(estimated_size < 20_000_000);
    }

    #[test]
    fn test_estimate_encoding_time() {
        let optimizer = ProxyOptimizer::new();
        let settings = ProxyGenerationSettings::quarter_res_h264();

        let estimated_time = optimizer.estimate_encoding_time(&settings, 60.0);
        assert!(estimated_time > 0.0);

        // Medium preset should take less than realtime for quarter res
        assert!(estimated_time < 60.0);
    }

    #[test]
    fn test_optimize_for_time() {
        let optimizer = ProxyOptimizer::new().with_max_encoding_time(10.0);

        let base_settings = ProxyGenerationSettings::quarter_res_h264();
        let optimized = optimizer
            .optimize(base_settings, 60.0)
            .expect("should succeed in test");

        // Should use ultrafast preset for time optimization
        assert_eq!(optimized.quality_preset, "ultrafast");
        assert!(optimized.threads > 0);
    }
}
