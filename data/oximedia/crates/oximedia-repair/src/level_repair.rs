#![allow(dead_code)]
//! Audio level repair and correction utilities.
//!
//! This module handles detection and correction of audio level anomalies
//! such as DC offset, clipping, sudden level jumps, and channel imbalances.
//! It provides tools for restoring clean audio levels without altering the
//! intended dynamic range of the content.

/// Type of level anomaly detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelAnomaly {
    /// DC offset present in the signal.
    DcOffset,
    /// Digital clipping (samples at maximum).
    Clipping,
    /// Sudden level jump between adjacent segments.
    LevelJump,
    /// Channel level imbalance in stereo/multichannel.
    ChannelImbalance,
    /// Extended silence in unexpected location.
    UnexpectedSilence,
    /// Signal level exceeding broadcast safe limits.
    OverLevel,
}

/// Severity of the level anomaly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LevelSeverity {
    /// Barely noticeable.
    Minor,
    /// Audible but tolerable.
    Moderate,
    /// Clearly problematic.
    Significant,
    /// Must be fixed before use.
    Critical,
}

/// A detected level issue at a specific position.
#[derive(Debug, Clone)]
pub struct LevelIssue {
    /// The type of anomaly.
    pub anomaly: LevelAnomaly,
    /// Severity of the issue.
    pub severity: LevelSeverity,
    /// Start sample index.
    pub start_sample: u64,
    /// End sample index.
    pub end_sample: u64,
    /// Measured value associated with the anomaly.
    pub measured_value: f64,
    /// Description of the issue.
    pub description: String,
}

/// Configuration for level analysis.
#[derive(Debug, Clone)]
pub struct LevelAnalysisConfig {
    /// DC offset threshold (absolute value).
    pub dc_offset_threshold: f64,
    /// Clipping threshold (fraction of maximum, e.g., 0.99).
    pub clipping_threshold: f64,
    /// Level jump threshold in dB.
    pub jump_threshold_db: f64,
    /// Channel imbalance threshold in dB.
    pub imbalance_threshold_db: f64,
    /// Window size for analysis in samples.
    pub window_size: usize,
    /// Hop size for overlapping analysis windows.
    pub hop_size: usize,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl Default for LevelAnalysisConfig {
    fn default() -> Self {
        Self {
            dc_offset_threshold: 0.005,
            clipping_threshold: 0.99,
            jump_threshold_db: 12.0,
            imbalance_threshold_db: 6.0,
            window_size: 4096,
            hop_size: 2048,
            sample_rate: 48000,
        }
    }
}

/// Result of a level repair operation.
#[derive(Debug, Clone)]
pub struct LevelRepairResult {
    /// Number of issues detected.
    pub issues_found: usize,
    /// Number of issues repaired.
    pub issues_repaired: usize,
    /// DC offset removed (if any).
    pub dc_offset_removed: f64,
    /// Number of clipped samples repaired.
    pub clipped_samples_repaired: u64,
    /// Peak level after repair.
    pub peak_after: f64,
    /// RMS level after repair.
    pub rms_after: f64,
}

/// Engine for detecting and repairing audio level issues.
#[derive(Debug)]
pub struct LevelRepairEngine {
    /// Configuration for the engine.
    config: LevelAnalysisConfig,
}

impl LevelRepairEngine {
    /// Create a new level repair engine with the given configuration.
    pub fn new(config: LevelAnalysisConfig) -> Self {
        Self { config }
    }

    /// Create an engine with default configuration.
    pub fn default_engine() -> Self {
        Self::new(LevelAnalysisConfig::default())
    }

    /// Analyze audio samples for level anomalies.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, samples: &[f32]) -> Vec<LevelIssue> {
        let mut issues = Vec::new();

        // Check DC offset
        if let Some(issue) = self.check_dc_offset(samples) {
            issues.push(issue);
        }

        // Check clipping
        issues.extend(self.check_clipping(samples));

        // Check level jumps
        issues.extend(self.check_level_jumps(samples));

        issues
    }

    /// Check for DC offset in the signal.
    #[allow(clippy::cast_precision_loss)]
    fn check_dc_offset(&self, samples: &[f32]) -> Option<LevelIssue> {
        if samples.is_empty() {
            return None;
        }
        let sum: f64 = samples.iter().map(|&s| s as f64).sum();
        let mean = sum / samples.len() as f64;

        if mean.abs() > self.config.dc_offset_threshold {
            let severity = if mean.abs() > 0.1 {
                LevelSeverity::Critical
            } else if mean.abs() > 0.02 {
                LevelSeverity::Significant
            } else {
                LevelSeverity::Minor
            };
            Some(LevelIssue {
                anomaly: LevelAnomaly::DcOffset,
                severity,
                start_sample: 0,
                end_sample: samples.len() as u64,
                measured_value: mean,
                description: format!("DC offset of {mean:.6} detected"),
            })
        } else {
            None
        }
    }

    /// Check for clipped samples.
    #[allow(clippy::cast_precision_loss)]
    fn check_clipping(&self, samples: &[f32]) -> Vec<LevelIssue> {
        let threshold = self.config.clipping_threshold as f32;
        let mut issues = Vec::new();
        let mut clip_start: Option<usize> = None;

        for (i, &s) in samples.iter().enumerate() {
            let is_clipped = s.abs() >= threshold;
            match (is_clipped, clip_start) {
                (true, None) => {
                    clip_start = Some(i);
                }
                (false, Some(start)) => {
                    let duration = i - start;
                    let severity = if duration > 1000 {
                        LevelSeverity::Critical
                    } else if duration > 100 {
                        LevelSeverity::Significant
                    } else if duration > 10 {
                        LevelSeverity::Moderate
                    } else {
                        LevelSeverity::Minor
                    };
                    issues.push(LevelIssue {
                        anomaly: LevelAnomaly::Clipping,
                        severity,
                        start_sample: start as u64,
                        end_sample: i as u64,
                        measured_value: duration as f64,
                        description: format!("{duration} clipped samples at position {start}"),
                    });
                    clip_start = None;
                }
                _ => {}
            }
        }

        if let Some(start) = clip_start {
            let duration = samples.len() - start;
            issues.push(LevelIssue {
                anomaly: LevelAnomaly::Clipping,
                severity: LevelSeverity::Significant,
                start_sample: start as u64,
                end_sample: samples.len() as u64,
                measured_value: duration as f64,
                description: format!("{duration} clipped samples at end"),
            });
        }

        issues
    }

    /// Check for sudden level jumps.
    #[allow(clippy::cast_precision_loss)]
    fn check_level_jumps(&self, samples: &[f32]) -> Vec<LevelIssue> {
        let mut issues = Vec::new();
        let window = self.config.window_size;
        let hop = self.config.hop_size;

        if samples.len() < window * 2 {
            return issues;
        }

        let mut prev_rms = compute_rms(&samples[..window]);

        let mut pos = hop;
        while pos + window <= samples.len() {
            let current_rms = compute_rms(&samples[pos..pos + window]);

            if prev_rms > 1e-10 && current_rms > 1e-10 {
                let ratio = (current_rms / prev_rms).abs();
                let db_diff = 20.0 * ratio.log10();
                if db_diff.abs() > self.config.jump_threshold_db {
                    issues.push(LevelIssue {
                        anomaly: LevelAnomaly::LevelJump,
                        severity: if db_diff.abs() > 24.0 {
                            LevelSeverity::Critical
                        } else {
                            LevelSeverity::Moderate
                        },
                        start_sample: pos as u64,
                        end_sample: (pos + window) as u64,
                        measured_value: db_diff,
                        description: format!("Level jump of {db_diff:.1} dB at sample {pos}"),
                    });
                }
            }

            prev_rms = current_rms;
            pos += hop;
        }

        issues
    }

    /// Remove DC offset from audio samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn remove_dc_offset(samples: &mut [f32]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = samples.iter().map(|&s| s as f64).sum();
        let mean = sum / samples.len() as f64;
        let offset = mean as f32;
        for s in samples.iter_mut() {
            *s -= offset;
        }
        mean
    }

    /// Repair clipped samples using cubic interpolation from surrounding data.
    #[allow(clippy::cast_precision_loss)]
    pub fn repair_clipping(samples: &mut [f32], threshold: f32) -> u64 {
        let mut repaired = 0u64;
        let len = samples.len();

        for i in 0..len {
            if samples[i].abs() >= threshold {
                // Find surrounding non-clipped samples
                let left = (0..i)
                    .rev()
                    .find(|&j| samples[j].abs() < threshold)
                    .map(|j| samples[j])
                    .unwrap_or(0.0);
                let right = ((i + 1)..len)
                    .find(|&j| samples[j].abs() < threshold)
                    .map(|j| samples[j])
                    .unwrap_or(0.0);
                let sign = samples[i].signum();
                // Blend toward interpolated value while keeping direction
                samples[i] = sign * ((left.abs() + right.abs()) * 0.5).min(threshold * 0.95);
                repaired += 1;
            }
        }

        repaired
    }

    /// Apply a full repair pass on audio samples.
    #[allow(clippy::cast_precision_loss)]
    pub fn repair(&self, samples: &mut [f32]) -> LevelRepairResult {
        let issues = self.analyze(samples);
        let issues_found = issues.len();

        // Remove DC offset
        let dc_removed = Self::remove_dc_offset(samples);

        // Repair clipping
        let clipped = Self::repair_clipping(samples, self.config.clipping_threshold as f32);

        // Compute post-repair stats
        let peak_after = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max) as f64;
        let rms_after = compute_rms(samples);

        LevelRepairResult {
            issues_found,
            issues_repaired: if dc_removed.abs() > self.config.dc_offset_threshold {
                1
            } else {
                0
            } + if clipped > 0 { 1 } else { 0 },
            dc_offset_removed: dc_removed,
            clipped_samples_repaired: clipped,
            peak_after,
            rms_after,
        }
    }
}

/// Compute RMS of a sample buffer.
#[allow(clippy::cast_precision_loss)]
fn compute_rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_anomaly_eq() {
        assert_eq!(LevelAnomaly::DcOffset, LevelAnomaly::DcOffset);
        assert_ne!(LevelAnomaly::Clipping, LevelAnomaly::DcOffset);
    }

    #[test]
    fn test_severity_ordering() {
        assert!(LevelSeverity::Minor < LevelSeverity::Moderate);
        assert!(LevelSeverity::Moderate < LevelSeverity::Significant);
        assert!(LevelSeverity::Significant < LevelSeverity::Critical);
    }

    #[test]
    fn test_default_config() {
        let cfg = LevelAnalysisConfig::default();
        assert!((cfg.dc_offset_threshold - 0.005).abs() < 1e-9);
        assert_eq!(cfg.window_size, 4096);
        assert_eq!(cfg.sample_rate, 48000);
    }

    #[test]
    fn test_detect_dc_offset() {
        let engine = LevelRepairEngine::default_engine();
        let samples: Vec<f32> = (0..1000).map(|_| 0.1).collect();
        let issues = engine.analyze(&samples);
        assert!(issues.iter().any(|i| i.anomaly == LevelAnomaly::DcOffset));
    }

    #[test]
    fn test_no_dc_offset_clean_signal() {
        let engine = LevelRepairEngine::default_engine();
        // Symmetric signal: positive and negative cancel out
        let mut samples = vec![0.0f32; 1000];
        for i in 0..1000 {
            samples[i] = if i % 2 == 0 { 0.5 } else { -0.5 };
        }
        let issues = engine.analyze(&samples);
        assert!(!issues.iter().any(|i| i.anomaly == LevelAnomaly::DcOffset));
    }

    #[test]
    fn test_detect_clipping() {
        let engine = LevelRepairEngine::default_engine();
        let mut samples = vec![0.5f32; 1000];
        for s in samples.iter_mut().take(20) {
            *s = 1.0;
        }
        let issues = engine.analyze(&samples);
        assert!(issues.iter().any(|i| i.anomaly == LevelAnomaly::Clipping));
    }

    #[test]
    fn test_remove_dc_offset() {
        let mut samples: Vec<f32> = (0..1000).map(|_| 0.1).collect();
        let removed = LevelRepairEngine::remove_dc_offset(&mut samples);
        assert!((removed - 0.1).abs() < 1e-4);
        let new_mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(new_mean.abs() < 1e-4);
    }

    #[test]
    fn test_remove_dc_offset_empty() {
        let mut samples: Vec<f32> = vec![];
        let removed = LevelRepairEngine::remove_dc_offset(&mut samples);
        assert_eq!(removed, 0.0);
    }

    #[test]
    fn test_repair_clipping() {
        let mut samples = vec![0.5f32; 100];
        samples[50] = 1.0;
        samples[51] = -1.0;
        let repaired = LevelRepairEngine::repair_clipping(&mut samples, 0.99);
        assert!(repaired >= 2);
        assert!(samples[50].abs() < 0.99);
        assert!(samples[51].abs() < 0.99);
    }

    #[test]
    fn test_compute_rms() {
        let samples = [1.0f32; 100];
        let rms = compute_rms(&samples);
        assert!((rms - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_rms_empty() {
        let rms = compute_rms(&[]);
        assert_eq!(rms, 0.0);
    }

    #[test]
    fn test_full_repair_pass() {
        let engine = LevelRepairEngine::default_engine();
        let mut samples: Vec<f32> = (0..2000).map(|_| 0.05).collect();
        samples[500] = 1.0;
        let result = engine.repair(&mut samples);
        assert!(result.issues_found > 0);
        assert!(result.dc_offset_removed.abs() > 0.0);
    }

    #[test]
    fn test_level_issue_fields() {
        let issue = LevelIssue {
            anomaly: LevelAnomaly::OverLevel,
            severity: LevelSeverity::Critical,
            start_sample: 100,
            end_sample: 200,
            measured_value: -3.5,
            description: "Over level".to_string(),
        };
        assert_eq!(issue.anomaly, LevelAnomaly::OverLevel);
        assert_eq!(issue.start_sample, 100);
        assert_eq!(issue.end_sample, 200);
    }

    #[test]
    fn test_check_level_jumps_short_buffer() {
        let engine = LevelRepairEngine::default_engine();
        let samples = vec![0.5f32; 100]; // Too short for window
        let issues = engine.check_level_jumps(&samples);
        assert!(issues.is_empty());
    }
}
