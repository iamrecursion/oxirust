//! Content fingerprinting for video and audio.
//!
//! This module provides perceptual fingerprinting capabilities for multimedia content,
//! enabling content identification, duplicate detection, and similarity matching.
//!
//! # Features
//!
//! - **Perceptual hashing**: Multiple hash algorithms (pHash, aHash, dHash, wHash)
//! - **Temporal fingerprinting**: Video-level fingerprints with keyframe extraction
//! - **Audio fingerprinting**: Chromaprint-like audio fingerprinting
//! - **Fast matching**: Sub-second lookups for millions of fingerprints
//! - **Robust**: Resistant to compression, scaling, and minor modifications
//!
//! # Example
//!
//! ```
//! use oximedia_cv::fingerprint::{VideoFingerprinter, FingerprintConfig};
//!
//! let config = FingerprintConfig::default();
//! let fingerprinter = VideoFingerprinter::new(config);
//! ```

pub mod chromaprint;
pub mod matching;
pub mod phash;
pub mod temporal;

use crate::error::{CvError, CvResult};
use serde::{Deserialize, Serialize};

/// Configuration for fingerprinting operations.
///
/// This structure controls various sensitivity and performance parameters
/// for fingerprint extraction and matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintConfig {
    /// Hash size for perceptual hashing (8, 16, or 32).
    pub hash_size: usize,

    /// Minimum frame interval for temporal sampling (in frames).
    pub min_frame_interval: usize,

    /// Maximum number of keyframes to extract per video.
    pub max_keyframes: usize,

    /// Scene change threshold (0.0-1.0).
    pub scene_threshold: f64,

    /// Enable audio fingerprinting.
    pub enable_audio: bool,

    /// Audio sample rate for fingerprinting (Hz).
    pub audio_sample_rate: u32,

    /// Audio frame size for FFT (samples).
    pub audio_frame_size: usize,

    /// Audio hop size between frames (samples).
    pub audio_hop_size: usize,

    /// Similarity threshold for matching (0.0-1.0).
    pub similarity_threshold: f64,

    /// Enable parallel processing.
    pub parallel: bool,
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self {
            hash_size: 16,
            min_frame_interval: 30,
            max_keyframes: 100,
            scene_threshold: 0.3,
            enable_audio: true,
            audio_sample_rate: 11025,
            audio_frame_size: 4096,
            audio_hop_size: 1024,
            similarity_threshold: 0.85,
            parallel: true,
        }
    }
}

impl FingerprintConfig {
    /// Creates a new fingerprint configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the hash size.
    #[must_use]
    pub fn with_hash_size(mut self, size: usize) -> Self {
        self.hash_size = size;
        self
    }

    /// Sets the minimum frame interval.
    #[must_use]
    pub fn with_frame_interval(mut self, interval: usize) -> Self {
        self.min_frame_interval = interval;
        self
    }

    /// Sets the maximum number of keyframes.
    #[must_use]
    pub fn with_max_keyframes(mut self, max: usize) -> Self {
        self.max_keyframes = max;
        self
    }

    /// Sets the scene change threshold.
    #[must_use]
    pub fn with_scene_threshold(mut self, threshold: f64) -> Self {
        self.scene_threshold = threshold;
        self
    }

    /// Enables or disables audio fingerprinting.
    #[must_use]
    pub fn with_audio(mut self, enable: bool) -> Self {
        self.enable_audio = enable;
        self
    }

    /// Sets the similarity threshold for matching.
    #[must_use]
    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Enables or disables parallel processing.
    #[must_use]
    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.parallel = enable;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> CvResult<()> {
        if self.hash_size != 8 && self.hash_size != 16 && self.hash_size != 32 {
            return Err(CvError::invalid_parameter(
                "hash_size",
                format!("{} (must be 8, 16, or 32)", self.hash_size),
            ));
        }

        if self.min_frame_interval == 0 {
            return Err(CvError::invalid_parameter(
                "min_frame_interval",
                "0 (must be > 0)",
            ));
        }

        if self.max_keyframes == 0 {
            return Err(CvError::invalid_parameter(
                "max_keyframes",
                "0 (must be > 0)",
            ));
        }

        if !(0.0..=1.0).contains(&self.scene_threshold) {
            return Err(CvError::invalid_parameter(
                "scene_threshold",
                format!("{} (must be in range 0.0-1.0)", self.scene_threshold),
            ));
        }

        if !(0.0..=1.0).contains(&self.similarity_threshold) {
            return Err(CvError::invalid_parameter(
                "similarity_threshold",
                format!("{} (must be in range 0.0-1.0)", self.similarity_threshold),
            ));
        }

        if self.audio_sample_rate == 0 {
            return Err(CvError::invalid_parameter(
                "audio_sample_rate",
                "0 (must be > 0)",
            ));
        }

        if self.audio_frame_size == 0 || (self.audio_frame_size & (self.audio_frame_size - 1)) != 0
        {
            return Err(CvError::invalid_parameter(
                "audio_frame_size",
                format!("{} (must be power of 2)", self.audio_frame_size),
            ));
        }

        if self.audio_hop_size == 0 || self.audio_hop_size > self.audio_frame_size {
            return Err(CvError::invalid_parameter(
                "audio_hop_size",
                format!("{} (must be > 0 and <= frame_size)", self.audio_hop_size),
            ));
        }

        Ok(())
    }
}

/// Video fingerprint combining visual and temporal features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFingerprint {
    /// Perceptual hash of representative frames.
    pub perceptual_hash: Vec<u64>,

    /// Temporal signature of the video.
    pub temporal_signature: Vec<f32>,

    /// Audio fingerprint (if available).
    pub audio_fingerprint: Option<Vec<u32>>,

    /// Duration in seconds.
    pub duration: f64,

    /// Number of frames.
    pub frame_count: usize,

    /// Video resolution (width, height).
    pub resolution: (u32, u32),
}

impl VideoFingerprint {
    /// Creates a new video fingerprint.
    #[must_use]
    pub fn new(
        perceptual_hash: Vec<u64>,
        temporal_signature: Vec<f32>,
        audio_fingerprint: Option<Vec<u32>>,
        duration: f64,
        frame_count: usize,
        resolution: (u32, u32),
    ) -> Self {
        Self {
            perceptual_hash,
            temporal_signature,
            audio_fingerprint,
            duration,
            frame_count,
            resolution,
        }
    }

    /// Returns the size of the fingerprint in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let phash_size = self.perceptual_hash.len() * std::mem::size_of::<u64>();
        let temporal_size = self.temporal_signature.len() * std::mem::size_of::<f32>();
        let audio_size = self
            .audio_fingerprint
            .as_ref()
            .map_or(0, |f| f.len() * std::mem::size_of::<u32>());

        phash_size + temporal_size + audio_size + std::mem::size_of::<Self>()
    }

    /// Compares this fingerprint with another.
    ///
    /// Returns a similarity score in the range [0.0, 1.0].
    #[must_use]
    pub fn compare(&self, other: &Self) -> f64 {
        matching::compare_fingerprints(self, other)
    }
}

/// Main video fingerprinting engine.
pub struct VideoFingerprinter {
    config: FingerprintConfig,
}

impl VideoFingerprinter {
    /// Creates a new video fingerprinter with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::fingerprint::{VideoFingerprinter, FingerprintConfig};
    ///
    /// let config = FingerprintConfig::default();
    /// let fingerprinter = VideoFingerprinter::new(config);
    /// ```
    #[must_use]
    pub fn new(config: FingerprintConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &FingerprintConfig {
        &self.config
    }

    /// Extracts fingerprint from raw image frames.
    ///
    /// # Arguments
    ///
    /// * `frames` - Iterator of image frames (width, height, RGB data)
    /// * `fps` - Frame rate of the video
    ///
    /// # Errors
    ///
    /// Returns an error if fingerprinting fails.
    pub fn fingerprint_frames<I>(&self, frames: I, fps: f64) -> CvResult<VideoFingerprint>
    where
        I: Iterator<Item = (u32, u32, Vec<u8>)>,
    {
        self.config.validate()?;

        // Collect frames
        let frame_vec: Vec<_> = frames.collect();

        if frame_vec.is_empty() {
            return Err(CvError::invalid_parameter("frames", "empty iterator"));
        }

        let frame_count = frame_vec.len();
        let duration = frame_count as f64 / fps;
        let (width, height, _) = &frame_vec[0];
        let resolution = (*width, *height);

        // Extract keyframes
        let keyframes = temporal::extract_keyframes(
            &frame_vec,
            self.config.min_frame_interval,
            self.config.max_keyframes,
            self.config.scene_threshold,
        )?;

        // Compute perceptual hashes
        let perceptual_hash = if self.config.parallel {
            phash::compute_hashes_parallel(&keyframes, self.config.hash_size)?
        } else {
            phash::compute_hashes(&keyframes, self.config.hash_size)?
        };

        // Compute temporal signature
        let temporal_signature =
            temporal::compute_temporal_signature(&frame_vec, self.config.min_frame_interval)?;

        Ok(VideoFingerprint::new(
            perceptual_hash,
            temporal_signature,
            None,
            duration,
            frame_count,
            resolution,
        ))
    }

    /// Extracts audio fingerprint from audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples (mono, f32)
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Errors
    ///
    /// Returns an error if fingerprinting fails.
    pub fn fingerprint_audio(&self, samples: &[f32], sample_rate: u32) -> CvResult<Vec<u32>> {
        self.config.validate()?;

        if !self.config.enable_audio {
            return Ok(Vec::new());
        }

        chromaprint::extract_fingerprint(
            samples,
            sample_rate,
            self.config.audio_frame_size,
            self.config.audio_hop_size,
        )
    }

    /// Adds audio fingerprint to an existing video fingerprint.
    pub fn add_audio_fingerprint(
        &self,
        mut fingerprint: VideoFingerprint,
        samples: &[f32],
        sample_rate: u32,
    ) -> CvResult<VideoFingerprint> {
        let audio_fp = self.fingerprint_audio(samples, sample_rate)?;
        fingerprint.audio_fingerprint = Some(audio_fp);
        Ok(fingerprint)
    }
}

/// Default implementation for `VideoFingerprinter`.
impl Default for VideoFingerprinter {
    fn default() -> Self {
        Self::new(FingerprintConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = FingerprintConfig::default();
        assert_eq!(config.hash_size, 16);
        assert_eq!(config.min_frame_interval, 30);
        assert_eq!(config.max_keyframes, 100);
        assert!(config.enable_audio);
    }

    #[test]
    fn test_config_builder() {
        let config = FingerprintConfig::default()
            .with_hash_size(32)
            .with_frame_interval(60)
            .with_max_keyframes(50)
            .with_audio(false);

        assert_eq!(config.hash_size, 32);
        assert_eq!(config.min_frame_interval, 60);
        assert_eq!(config.max_keyframes, 50);
        assert!(!config.enable_audio);
    }

    #[test]
    fn test_config_validation() {
        let mut config = FingerprintConfig::default();
        assert!(config.validate().is_ok());

        config.hash_size = 10;
        assert!(config.validate().is_err());

        config.hash_size = 16;
        config.scene_threshold = 1.5;
        assert!(config.validate().is_err());

        config.scene_threshold = 0.5;
        config.min_frame_interval = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_fingerprint_creation() {
        let fp = VideoFingerprint::new(
            vec![0x1234567890ABCDEF],
            vec![0.1, 0.2, 0.3],
            Some(vec![100, 200, 300]),
            120.0,
            3000,
            (1920, 1080),
        );

        assert_eq!(fp.perceptual_hash.len(), 1);
        assert_eq!(fp.temporal_signature.len(), 3);
        assert!(fp.audio_fingerprint.is_some());
        assert_eq!(fp.duration, 120.0);
        assert_eq!(fp.frame_count, 3000);
        assert_eq!(fp.resolution, (1920, 1080));
    }

    #[test]
    fn test_fingerprint_size() {
        let fp = VideoFingerprint::new(
            vec![0x1234567890ABCDEF; 10],
            vec![0.5; 100],
            Some(vec![42; 500]),
            60.0,
            1500,
            (1280, 720),
        );

        let size = fp.size_bytes();
        assert!(size > 0);
    }

    #[test]
    fn test_fingerprinter_creation() {
        let config = FingerprintConfig::default();
        let fingerprinter = VideoFingerprinter::new(config.clone());
        assert_eq!(fingerprinter.config().hash_size, config.hash_size);
    }
}
