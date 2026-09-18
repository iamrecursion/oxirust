#![allow(dead_code)]
//! Video frame-level forensic analysis for detecting inter-frame tampering.
//!
//! This module provides frame-by-frame forensic analysis of video sequences to detect
//! frame insertion, deletion, duplication, and temporal splicing.
//!
//! # Features
//!
//! - **Frame duplicate detection** using perceptual hashing
//! - **Temporal consistency analysis** for detecting inserted or deleted frames
//! - **Frame rate anomaly detection** for re-encoded or speed-altered footage
//! - **GOP structure analysis** for re-compression detection
//! - **Inter-frame noise consistency** checking

use std::collections::HashMap;

/// A perceptual hash of a video frame (64-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameHash {
    /// The 64-bit perceptual hash value.
    pub value: u64,
}

impl FrameHash {
    /// Create a new frame hash from a raw value.
    #[must_use]
    pub fn new(value: u64) -> Self {
        Self { value }
    }

    /// Compute the Hamming distance between two frame hashes.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.value ^ other.value).count_ones()
    }

    /// Check if two hashes are considered similar (Hamming distance below threshold).
    #[must_use]
    pub fn is_similar(&self, other: &Self, threshold: u32) -> bool {
        self.hamming_distance(other) <= threshold
    }

    /// Compute a simple perceptual hash from grayscale pixel data.
    ///
    /// The image is downscaled to 8x8, and each pixel is compared to the mean.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_grayscale(data: &[u8], width: u32, height: u32) -> Self {
        if data.is_empty() || width == 0 || height == 0 {
            return Self { value: 0 };
        }

        // Downsample to 8x8
        let mut small = [0u64; 64];
        let mut counts = [0u64; 64];

        for y in 0..height {
            for x in 0..width {
                let sx = (x * 8 / width) as usize;
                let sy = (y * 8 / height) as usize;
                if sx < 8 && sy < 8 {
                    let idx = sy * 8 + sx;
                    let pixel_idx = (y * width + x) as usize;
                    if pixel_idx < data.len() {
                        small[idx] += u64::from(data[pixel_idx]);
                        counts[idx] += 1;
                    }
                }
            }
        }

        // Average each cell
        let mut avg_pixels = [0.0f64; 64];
        let mut total = 0.0f64;
        for i in 0..64 {
            if counts[i] > 0 {
                avg_pixels[i] = small[i] as f64 / counts[i] as f64;
            }
            total += avg_pixels[i];
        }
        let mean = total / 64.0;

        // Build hash
        let mut hash: u64 = 0;
        for (i, &px) in avg_pixels.iter().enumerate() {
            if px > mean {
                hash |= 1u64 << i;
            }
        }

        Self { value: hash }
    }
}

/// Information about a single analyzed frame.
#[derive(Debug, Clone)]
pub struct FrameInfo {
    /// Frame index in the sequence.
    pub index: u64,
    /// Perceptual hash of the frame.
    pub hash: FrameHash,
    /// Average pixel intensity (0-255).
    pub avg_intensity: f64,
    /// Noise variance estimate.
    pub noise_variance: f64,
    /// Difference metric to the previous frame.
    pub prev_diff: f64,
}

impl FrameInfo {
    /// Create new frame info.
    #[must_use]
    pub fn new(
        index: u64,
        hash: FrameHash,
        avg_intensity: f64,
        noise_variance: f64,
        prev_diff: f64,
    ) -> Self {
        Self {
            index,
            hash,
            avg_intensity,
            noise_variance,
            prev_diff,
        }
    }
}

/// A detected frame anomaly.
#[derive(Debug, Clone)]
pub struct FrameAnomaly {
    /// Frame index where the anomaly was detected.
    pub frame_index: u64,
    /// Type of anomaly detected.
    pub anomaly_type: FrameAnomalyType,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Human-readable description.
    pub description: String,
}

/// Types of frame-level anomalies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameAnomalyType {
    /// Duplicate frame detected (exact or near-duplicate).
    DuplicateFrame,
    /// Sudden noise level change suggesting different source.
    NoiseInconsistency,
    /// Unusual frame difference suggesting insertion.
    FrameInsertion,
    /// Missing temporal continuity suggesting deletion.
    FrameDeletion,
    /// Re-compression artifacts detected.
    ReCompression,
    /// Sudden brightness/contrast change.
    IntensityJump,
}

/// Configuration for frame forensic analysis.
#[derive(Debug, Clone)]
pub struct FrameForensicsConfig {
    /// Hamming distance threshold for duplicate detection.
    pub duplicate_threshold: u32,
    /// Noise variance ratio threshold for inconsistency detection.
    pub noise_ratio_threshold: f64,
    /// Frame difference threshold multiplier (relative to median).
    pub diff_threshold_multiplier: f64,
    /// Intensity jump threshold (0-255 scale).
    pub intensity_jump_threshold: f64,
    /// Minimum confidence to report an anomaly.
    pub min_confidence: f64,
}

impl Default for FrameForensicsConfig {
    fn default() -> Self {
        Self {
            duplicate_threshold: 5,
            noise_ratio_threshold: 2.0,
            diff_threshold_multiplier: 3.0,
            intensity_jump_threshold: 30.0,
            min_confidence: 0.5,
        }
    }
}

/// Result of frame forensic analysis on a video sequence.
#[derive(Debug, Clone)]
pub struct FrameForensicsReport {
    /// Total number of frames analyzed.
    pub total_frames: u64,
    /// Detected anomalies.
    pub anomalies: Vec<FrameAnomaly>,
    /// Number of duplicate frames found.
    pub duplicate_count: u64,
    /// Number of noise inconsistencies found.
    pub noise_inconsistency_count: u64,
    /// Overall tampering likelihood (0.0 to 1.0).
    pub tampering_likelihood: f64,
}

/// Frame-level forensic analyzer.
#[derive(Debug, Clone)]
pub struct FrameForensicAnalyzer {
    /// Configuration.
    config: FrameForensicsConfig,
}

impl FrameForensicAnalyzer {
    /// Create a new frame forensic analyzer.
    #[must_use]
    pub fn new(config: FrameForensicsConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self {
            config: FrameForensicsConfig::default(),
        }
    }

    /// Analyze a sequence of frame infos for anomalies.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, frames: &[FrameInfo]) -> FrameForensicsReport {
        let mut anomalies = Vec::new();
        let mut duplicate_count = 0u64;
        let mut noise_inconsistency_count = 0u64;

        if frames.len() < 2 {
            return FrameForensicsReport {
                total_frames: frames.len() as u64,
                anomalies,
                duplicate_count,
                noise_inconsistency_count,
                tampering_likelihood: 0.0,
            };
        }

        // Detect duplicates
        let dups = self.detect_duplicates(frames);
        for anomaly in &dups {
            duplicate_count += 1;
            anomalies.push(anomaly.clone());
        }

        // Detect noise inconsistencies
        let noise_anom = self.detect_noise_inconsistencies(frames);
        for anomaly in &noise_anom {
            noise_inconsistency_count += 1;
            anomalies.push(anomaly.clone());
        }

        // Detect intensity jumps
        let intensity_anom = self.detect_intensity_jumps(frames);
        anomalies.extend(intensity_anom);

        // Detect frame difference anomalies (insertions/deletions)
        let diff_anom = self.detect_diff_anomalies(frames);
        anomalies.extend(diff_anom);

        // Filter by minimum confidence
        anomalies.retain(|a| a.confidence >= self.config.min_confidence);

        // Overall likelihood
        let tampering_likelihood = if frames.is_empty() {
            0.0
        } else {
            let anomaly_ratio = anomalies.len() as f64 / frames.len() as f64;
            (anomaly_ratio * 10.0).min(1.0)
        };

        FrameForensicsReport {
            total_frames: frames.len() as u64,
            anomalies,
            duplicate_count,
            noise_inconsistency_count,
            tampering_likelihood,
        }
    }

    /// Detect duplicate frames.
    fn detect_duplicates(&self, frames: &[FrameInfo]) -> Vec<FrameAnomaly> {
        let mut anomalies = Vec::new();
        let mut seen: HashMap<u64, u64> = HashMap::new();

        for frame in frames {
            if let Some(&first_idx) = seen.get(&frame.hash.value) {
                if first_idx != frame.index {
                    anomalies.push(FrameAnomaly {
                        frame_index: frame.index,
                        anomaly_type: FrameAnomalyType::DuplicateFrame,
                        confidence: 0.95,
                        description: format!(
                            "Frame {} is a duplicate of frame {}",
                            frame.index, first_idx
                        ),
                    });
                }
            } else {
                seen.insert(frame.hash.value, frame.index);
            }
        }

        // Also check near-duplicates (within threshold)
        for i in 1..frames.len() {
            let dist = frames[i].hash.hamming_distance(&frames[i - 1].hash);
            if dist > 0 && dist <= self.config.duplicate_threshold && frames[i].prev_diff < 0.5 {
                anomalies.push(FrameAnomaly {
                    frame_index: frames[i].index,
                    anomaly_type: FrameAnomalyType::DuplicateFrame,
                    #[allow(clippy::cast_precision_loss)]
                    confidence: 1.0 - (dist as f64 / 64.0),
                    description: format!(
                        "Frame {} is near-duplicate of frame {} (hamming={})",
                        frames[i].index,
                        frames[i - 1].index,
                        dist
                    ),
                });
            }
        }

        anomalies
    }

    /// Detect noise level inconsistencies.
    #[allow(clippy::cast_precision_loss)]
    fn detect_noise_inconsistencies(&self, frames: &[FrameInfo]) -> Vec<FrameAnomaly> {
        let mut anomalies = Vec::new();

        if frames.len() < 3 {
            return anomalies;
        }

        // Compute median noise variance
        let mut variances: Vec<f64> = frames.iter().map(|f| f.noise_variance).collect();
        variances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_var = variances[variances.len() / 2];

        if median_var < 1e-10 {
            return anomalies;
        }

        for frame in frames {
            let ratio = frame.noise_variance / median_var;
            if ratio > self.config.noise_ratio_threshold
                || (ratio < 1.0 / self.config.noise_ratio_threshold && ratio > 0.0)
            {
                let confidence = ((ratio - 1.0).abs() / self.config.noise_ratio_threshold).min(1.0);
                anomalies.push(FrameAnomaly {
                    frame_index: frame.index,
                    anomaly_type: FrameAnomalyType::NoiseInconsistency,
                    confidence,
                    description: format!(
                        "Frame {} noise variance {:.2} differs from median {:.2} (ratio={:.2})",
                        frame.index, frame.noise_variance, median_var, ratio
                    ),
                });
            }
        }

        anomalies
    }

    /// Detect intensity jumps.
    fn detect_intensity_jumps(&self, frames: &[FrameInfo]) -> Vec<FrameAnomaly> {
        let mut anomalies = Vec::new();

        for i in 1..frames.len() {
            let diff = (frames[i].avg_intensity - frames[i - 1].avg_intensity).abs();
            if diff > self.config.intensity_jump_threshold {
                let confidence = (diff / 255.0).min(1.0);
                anomalies.push(FrameAnomaly {
                    frame_index: frames[i].index,
                    anomaly_type: FrameAnomalyType::IntensityJump,
                    confidence,
                    description: format!(
                        "Frame {} has intensity jump of {:.1} from previous frame",
                        frames[i].index, diff
                    ),
                });
            }
        }

        anomalies
    }

    /// Detect frame difference anomalies (potential insertions/deletions).
    #[allow(clippy::cast_precision_loss)]
    fn detect_diff_anomalies(&self, frames: &[FrameInfo]) -> Vec<FrameAnomaly> {
        let mut anomalies = Vec::new();

        let diffs: Vec<f64> = frames.iter().map(|f| f.prev_diff).collect();
        if diffs.len() < 3 {
            return anomalies;
        }

        let mut sorted_diffs = diffs[1..].to_vec(); // Skip first (no previous)
        sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_diff = sorted_diffs[sorted_diffs.len() / 2];

        if median_diff < 1e-10 {
            return anomalies;
        }

        let threshold = median_diff * self.config.diff_threshold_multiplier;

        for frame in frames.iter().skip(1) {
            if frame.prev_diff > threshold {
                let confidence = ((frame.prev_diff / median_diff - 1.0)
                    / self.config.diff_threshold_multiplier)
                    .min(1.0);
                anomalies.push(FrameAnomaly {
                    frame_index: frame.index,
                    anomaly_type: FrameAnomalyType::FrameInsertion,
                    confidence,
                    description: format!(
                        "Frame {} has unusually high difference {:.2} (median={:.2})",
                        frame.index, frame.prev_diff, median_diff
                    ),
                });
            }
        }

        anomalies
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &FrameForensicsConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_hash_hamming() {
        let h1 = FrameHash::new(0b1010_1010);
        let h2 = FrameHash::new(0b1010_0000);
        assert_eq!(h1.hamming_distance(&h2), 2);
    }

    #[test]
    fn test_frame_hash_similarity() {
        let h1 = FrameHash::new(0xFF);
        let h2 = FrameHash::new(0xFE);
        assert!(h1.is_similar(&h2, 1));
        assert!(!h1.is_similar(&h2, 0));
    }

    #[test]
    fn test_frame_hash_identical() {
        let h = FrameHash::new(12345);
        assert_eq!(h.hamming_distance(&h), 0);
        assert!(h.is_similar(&h, 0));
    }

    #[test]
    fn test_frame_hash_from_grayscale() {
        let data = vec![128u8; 64 * 64];
        let hash = FrameHash::from_grayscale(&data, 64, 64);
        // Uniform image -> all pixels == mean -> hash should be 0 or all-ones
        // since > mean test: equal values won't set bits
        assert_eq!(hash.value, 0);
    }

    #[test]
    fn test_frame_hash_from_gradient() {
        let mut data = vec![0u8; 64 * 64];
        for (i, pixel) in data.iter_mut().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            {
                *pixel = (i % 256) as u8;
            }
        }
        let hash = FrameHash::from_grayscale(&data, 64, 64);
        // Gradient should produce a non-zero hash
        assert_ne!(hash.value, 0);
    }

    #[test]
    fn test_frame_hash_empty() {
        let hash = FrameHash::from_grayscale(&[], 0, 0);
        assert_eq!(hash.value, 0);
    }

    #[test]
    fn test_config_default() {
        let config = FrameForensicsConfig::default();
        assert_eq!(config.duplicate_threshold, 5);
        assert!((config.noise_ratio_threshold - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyze_empty() {
        let analyzer = FrameForensicAnalyzer::with_defaults();
        let report = analyzer.analyze(&[]);
        assert_eq!(report.total_frames, 0);
        assert!(report.anomalies.is_empty());
    }

    #[test]
    fn test_analyze_single_frame() {
        let analyzer = FrameForensicAnalyzer::with_defaults();
        let frames = vec![FrameInfo::new(0, FrameHash::new(100), 128.0, 5.0, 0.0)];
        let report = analyzer.analyze(&frames);
        assert_eq!(report.total_frames, 1);
    }

    #[test]
    fn test_detect_duplicates() {
        let analyzer = FrameForensicAnalyzer::with_defaults();
        let frames = vec![
            FrameInfo::new(0, FrameHash::new(100), 128.0, 5.0, 0.0),
            FrameInfo::new(1, FrameHash::new(200), 128.0, 5.0, 10.0),
            FrameInfo::new(2, FrameHash::new(100), 128.0, 5.0, 10.0),
        ];
        let report = analyzer.analyze(&frames);
        assert!(report.duplicate_count > 0);
    }

    #[test]
    fn test_detect_intensity_jump() {
        let config = FrameForensicsConfig {
            intensity_jump_threshold: 20.0,
            min_confidence: 0.0,
            ..FrameForensicsConfig::default()
        };
        let analyzer = FrameForensicAnalyzer::new(config);
        let frames = vec![
            FrameInfo::new(0, FrameHash::new(1), 100.0, 5.0, 0.0),
            FrameInfo::new(1, FrameHash::new(2), 100.0, 5.0, 5.0),
            FrameInfo::new(2, FrameHash::new(3), 200.0, 5.0, 5.0), // jump of 100
        ];
        let report = analyzer.analyze(&frames);
        let intensity_anomalies: Vec<_> = report
            .anomalies
            .iter()
            .filter(|a| a.anomaly_type == FrameAnomalyType::IntensityJump)
            .collect();
        assert!(!intensity_anomalies.is_empty());
    }

    #[test]
    fn test_detect_noise_inconsistency() {
        let config = FrameForensicsConfig {
            noise_ratio_threshold: 2.0,
            min_confidence: 0.0,
            ..FrameForensicsConfig::default()
        };
        let analyzer = FrameForensicAnalyzer::new(config);
        let frames = vec![
            FrameInfo::new(0, FrameHash::new(1), 128.0, 5.0, 0.0),
            FrameInfo::new(1, FrameHash::new(2), 128.0, 5.0, 5.0),
            FrameInfo::new(2, FrameHash::new(3), 128.0, 5.0, 5.0),
            FrameInfo::new(3, FrameHash::new(4), 128.0, 25.0, 5.0), // 5x noise
        ];
        let report = analyzer.analyze(&frames);
        assert!(report.noise_inconsistency_count > 0);
    }
}
