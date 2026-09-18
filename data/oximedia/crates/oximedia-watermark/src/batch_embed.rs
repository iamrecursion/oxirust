#![allow(dead_code)]
//! Batch watermark embedding for processing multiple audio segments.
//!
//! Provides a pipeline for embedding watermarks across a collection of audio
//! segments, tracking per-segment results, and producing aggregate quality
//! reports.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a batch job.
pub type BatchId = u64;

/// Status of an individual embedding task within a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Waiting to be processed.
    Pending,
    /// Currently being embedded.
    InProgress,
    /// Completed successfully.
    Done,
    /// Failed with a recoverable error.
    Failed,
    /// Skipped (e.g. segment too short).
    Skipped,
}

/// Configuration for a batch embedding run.
#[derive(Debug, Clone)]
pub struct BatchEmbedConfig {
    /// Embedding strength (0.0 .. 1.0).
    pub strength: f32,
    /// Secret key for PRNG-based algorithms.
    pub key: u64,
    /// Minimum segment length (samples) required for embedding.
    pub min_segment_len: usize,
    /// Maximum number of concurrent tasks (logical limit).
    pub max_concurrency: usize,
    /// Whether to collect per-segment quality metrics.
    pub collect_metrics: bool,
}

impl Default for BatchEmbedConfig {
    fn default() -> Self {
        Self {
            strength: 0.1,
            key: 0,
            min_segment_len: 2048,
            max_concurrency: 4,
            collect_metrics: true,
        }
    }
}

/// Per-segment quality snapshot.
#[derive(Debug, Clone)]
pub struct SegmentMetrics {
    /// Signal-to-noise ratio in dB.
    pub snr_db: f64,
    /// Peak absolute distortion.
    pub peak_distortion: f64,
    /// Mean absolute distortion.
    pub mean_distortion: f64,
}

/// Result of embedding a single segment.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Zero-based segment index.
    pub segment_index: usize,
    /// Final status.
    pub status: TaskStatus,
    /// Optional quality metrics (present when `collect_metrics` is enabled).
    pub metrics: Option<SegmentMetrics>,
    /// Length of the watermarked output in samples.
    pub output_len: usize,
}

/// Aggregate report for an entire batch run.
#[derive(Debug, Clone)]
pub struct BatchReport {
    /// Batch identifier.
    pub batch_id: BatchId,
    /// Total segments submitted.
    pub total_segments: usize,
    /// Number that completed successfully.
    pub succeeded: usize,
    /// Number that failed.
    pub failed: usize,
    /// Number that were skipped.
    pub skipped: usize,
    /// Average SNR across succeeded segments (NaN if none).
    pub avg_snr_db: f64,
    /// Per-segment results keyed by segment index.
    pub results: HashMap<usize, TaskResult>,
}

// ---------------------------------------------------------------------------
// Batch Embedder
// ---------------------------------------------------------------------------

/// Orchestrates batch watermark embedding across multiple audio segments.
#[derive(Debug)]
pub struct BatchEmbedder {
    config: BatchEmbedConfig,
    next_batch_id: BatchId,
}

impl BatchEmbedder {
    /// Create a new batch embedder with the given configuration.
    #[must_use]
    pub fn new(config: BatchEmbedConfig) -> Self {
        Self {
            config,
            next_batch_id: 1,
        }
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &BatchEmbedConfig {
        &self.config
    }

    /// Embed a watermark into every segment, returning watermarked audio and a
    /// report.
    ///
    /// `segments` is a slice of audio buffers (each a `&[f32]`).
    /// `payload` is the raw watermark payload bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn embed_all(
        &mut self,
        segments: &[&[f32]],
        payload: &[u8],
    ) -> (Vec<Vec<f32>>, BatchReport) {
        let batch_id = self.next_batch_id;
        self.next_batch_id += 1;

        let mut outputs: Vec<Vec<f32>> = Vec::with_capacity(segments.len());
        let mut results: HashMap<usize, TaskResult> = HashMap::new();
        let mut succeeded = 0usize;
        let mut failed = 0usize;
        let mut skipped = 0usize;

        for (idx, segment) in segments.iter().enumerate() {
            if segment.len() < self.config.min_segment_len {
                outputs.push(segment.to_vec());
                results.insert(
                    idx,
                    TaskResult {
                        segment_index: idx,
                        status: TaskStatus::Skipped,
                        metrics: None,
                        output_len: segment.len(),
                    },
                );
                skipped += 1;
                continue;
            }

            // Simple additive spread-spectrum style embedding (demo quality).
            let mut out = segment.to_vec();
            let strength = self.config.strength;
            let key = self.config.key.wrapping_add(idx as u64);
            let mut prng_state = key;
            for (i, sample) in out.iter_mut().enumerate() {
                // xorshift-style PRNG
                prng_state ^= prng_state << 13;
                prng_state ^= prng_state >> 7;
                prng_state ^= prng_state << 17;
                let bit_idx = i % (payload.len() * 8);
                let byte_idx = bit_idx / 8;
                let bit_pos = bit_idx % 8;
                let bit = f32::from((payload[byte_idx] >> bit_pos) & 1);
                let chip = if prng_state % 2 == 0 {
                    1.0_f32
                } else {
                    -1.0_f32
                };
                *sample += strength * chip * (2.0 * bit - 1.0);
            }

            let metrics = if self.config.collect_metrics {
                Some(compute_segment_metrics(segment, &out))
            } else {
                None
            };

            let output_len = out.len();
            outputs.push(out);

            if output_len > 0 {
                succeeded += 1;
                results.insert(
                    idx,
                    TaskResult {
                        segment_index: idx,
                        status: TaskStatus::Done,
                        metrics,
                        output_len,
                    },
                );
            } else {
                failed += 1;
                results.insert(
                    idx,
                    TaskResult {
                        segment_index: idx,
                        status: TaskStatus::Failed,
                        metrics: None,
                        output_len: 0,
                    },
                );
            }
        }

        let avg_snr = if succeeded > 0 {
            let sum: f64 = results
                .values()
                .filter_map(|r| r.metrics.as_ref().map(|m| m.snr_db))
                .sum();
            sum / succeeded as f64
        } else {
            f64::NAN
        };

        let report = BatchReport {
            batch_id,
            total_segments: segments.len(),
            succeeded,
            failed,
            skipped,
            avg_snr_db: avg_snr,
            results,
        };

        (outputs, report)
    }

    /// Check whether a segment meets the minimum length requirement.
    #[must_use]
    pub fn is_eligible(&self, segment: &[f32]) -> bool {
        segment.len() >= self.config.min_segment_len
    }
}

/// Compute quality metrics between original and watermarked segment.
#[allow(clippy::cast_precision_loss)]
fn compute_segment_metrics(original: &[f32], watermarked: &[f32]) -> SegmentMetrics {
    let len = original.len().min(watermarked.len());
    if len == 0 {
        return SegmentMetrics {
            snr_db: 0.0,
            peak_distortion: 0.0,
            mean_distortion: 0.0,
        };
    }

    let mut signal_power = 0.0_f64;
    let mut noise_power = 0.0_f64;
    let mut peak_dist = 0.0_f64;
    let mut sum_dist = 0.0_f64;

    for i in 0..len {
        let s = f64::from(original[i]);
        let w = f64::from(watermarked[i]);
        let d = (w - s).abs();
        signal_power += s * s;
        noise_power += (w - s) * (w - s);
        if d > peak_dist {
            peak_dist = d;
        }
        sum_dist += d;
    }

    let snr_db = if noise_power > 0.0 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f64::INFINITY
    };

    SegmentMetrics {
        snr_db,
        peak_distortion: peak_dist,
        mean_distortion: sum_dist / len as f64,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(len: usize) -> Vec<f32> {
        #[allow(clippy::cast_precision_loss)]
        (0..len).map(|i| (i as f32 * 0.1).sin()).collect()
    }

    #[test]
    fn test_default_config() {
        let cfg = BatchEmbedConfig::default();
        assert!((cfg.strength - 0.1).abs() < f32::EPSILON);
        assert_eq!(cfg.max_concurrency, 4);
        assert!(cfg.collect_metrics);
    }

    #[test]
    fn test_batch_embedder_creation() {
        let e = BatchEmbedder::new(BatchEmbedConfig::default());
        assert_eq!(e.next_batch_id, 1);
    }

    #[test]
    fn test_embed_single_segment() {
        let mut e = BatchEmbedder::new(BatchEmbedConfig::default());
        let seg = make_sine(4096);
        let payload = b"AB";
        let (outputs, report) = e.embed_all(&[seg.as_slice()], payload);
        assert_eq!(outputs.len(), 1);
        assert_eq!(report.succeeded, 1);
        assert_eq!(report.failed, 0);
        assert_eq!(report.skipped, 0);
        assert_eq!(report.batch_id, 1);
    }

    #[test]
    fn test_embed_multiple_segments() {
        let mut e = BatchEmbedder::new(BatchEmbedConfig::default());
        let s1 = make_sine(4096);
        let s2 = make_sine(8192);
        let payload = b"XY";
        let (outputs, report) = e.embed_all(&[s1.as_slice(), s2.as_slice()], payload);
        assert_eq!(outputs.len(), 2);
        assert_eq!(report.succeeded, 2);
        assert_eq!(report.total_segments, 2);
    }

    #[test]
    fn test_skip_short_segment() {
        let cfg = BatchEmbedConfig {
            min_segment_len: 8000,
            ..BatchEmbedConfig::default()
        };
        let mut e = BatchEmbedder::new(cfg);
        let short = make_sine(100);
        let payload = b"Z";
        let (_, report) = e.embed_all(&[short.as_slice()], payload);
        assert_eq!(report.skipped, 1);
        assert_eq!(report.succeeded, 0);
    }

    #[test]
    fn test_batch_id_increments() {
        let mut e = BatchEmbedder::new(BatchEmbedConfig::default());
        let seg = make_sine(4096);
        let payload = b"A";
        let (_, r1) = e.embed_all(&[seg.as_slice()], payload);
        let (_, r2) = e.embed_all(&[seg.as_slice()], payload);
        assert_eq!(r1.batch_id, 1);
        assert_eq!(r2.batch_id, 2);
    }

    #[test]
    fn test_output_same_length_as_input() {
        let mut e = BatchEmbedder::new(BatchEmbedConfig::default());
        let seg = make_sine(4096);
        let payload = b"Q";
        let (outputs, _) = e.embed_all(&[seg.as_slice()], payload);
        assert_eq!(outputs[0].len(), seg.len());
    }

    #[test]
    fn test_metrics_present_when_enabled() {
        let mut e = BatchEmbedder::new(BatchEmbedConfig::default());
        let seg = make_sine(4096);
        let payload = b"M";
        let (_, report) = e.embed_all(&[seg.as_slice()], payload);
        let task = report.results.get(&0).expect("should succeed in test");
        assert!(task.metrics.is_some());
    }

    #[test]
    fn test_metrics_absent_when_disabled() {
        let cfg = BatchEmbedConfig {
            collect_metrics: false,
            ..BatchEmbedConfig::default()
        };
        let mut e = BatchEmbedder::new(cfg);
        let seg = make_sine(4096);
        let payload = b"N";
        let (_, report) = e.embed_all(&[seg.as_slice()], payload);
        let task = report.results.get(&0).expect("should succeed in test");
        assert!(task.metrics.is_none());
    }

    #[test]
    fn test_snr_positive_for_small_strength() {
        let mut e = BatchEmbedder::new(BatchEmbedConfig {
            strength: 0.001,
            ..BatchEmbedConfig::default()
        });
        let seg = make_sine(4096);
        let payload = b"S";
        let (_, report) = e.embed_all(&[seg.as_slice()], payload);
        let task = report.results.get(&0).expect("should succeed in test");
        assert!(
            task.metrics
                .as_ref()
                .expect("should succeed in test")
                .snr_db
                > 0.0
        );
    }

    #[test]
    fn test_is_eligible() {
        let e = BatchEmbedder::new(BatchEmbedConfig::default());
        let short: Vec<f32> = vec![0.0; 100];
        let long: Vec<f32> = vec![0.0; 4096];
        assert!(!e.is_eligible(&short));
        assert!(e.is_eligible(&long));
    }

    #[test]
    fn test_empty_batch() {
        let mut e = BatchEmbedder::new(BatchEmbedConfig::default());
        let payload = b"E";
        let (outputs, report) = e.embed_all(&[], payload);
        assert!(outputs.is_empty());
        assert_eq!(report.total_segments, 0);
        assert_eq!(report.succeeded, 0);
    }

    #[test]
    fn test_compute_segment_metrics_zero_len() {
        let m = compute_segment_metrics(&[], &[]);
        assert!((m.snr_db - 0.0).abs() < f64::EPSILON);
        assert!((m.peak_distortion - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_watermark_modifies_samples() {
        let mut e = BatchEmbedder::new(BatchEmbedConfig {
            strength: 0.5,
            ..BatchEmbedConfig::default()
        });
        let seg = make_sine(4096);
        let payload = b"D";
        let (outputs, _) = e.embed_all(&[seg.as_slice()], payload);
        // At least some samples should differ
        let diffs: usize = seg
            .iter()
            .zip(outputs[0].iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-9)
            .count();
        assert!(diffs > 0);
    }
}
