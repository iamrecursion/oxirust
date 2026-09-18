#![allow(dead_code)]
//! Keyframe alignment enforcement across multi-bitrate variants.
//!
//! In adaptive streaming, all bitrate variants must have their segments
//! start at the same keyframe positions. This module provides tools to:
//!
//! - Define a canonical keyframe schedule from the highest-bitrate variant
//! - Validate that all variants align to the canonical schedule
//! - Detect and report misaligned segments
//! - Compute alignment scores and suggest corrections

use std::time::Duration;

use crate::error::{PackagerError, PackagerResult};

// ---------------------------------------------------------------------------
// KeyframePosition
// ---------------------------------------------------------------------------

/// A keyframe position in a media stream.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyframePosition {
    /// Presentation timestamp of the keyframe.
    pub timestamp: Duration,
    /// Segment index this keyframe belongs to.
    pub segment_index: u64,
    /// Byte offset in the stream (for seeking).
    pub byte_offset: u64,
}

impl KeyframePosition {
    /// Create a new keyframe position.
    #[must_use]
    pub fn new(timestamp: Duration, segment_index: u64, byte_offset: u64) -> Self {
        Self {
            timestamp,
            segment_index,
            byte_offset,
        }
    }
}

// ---------------------------------------------------------------------------
// VariantKeyframes
// ---------------------------------------------------------------------------

/// Keyframe positions for a single bitrate variant.
#[derive(Debug, Clone)]
pub struct VariantKeyframes {
    /// Variant identifier (e.g. "1080p", "720p", or a bitrate label).
    pub variant_id: String,
    /// Bitrate of this variant in bits per second.
    pub bitrate: u32,
    /// Ordered list of keyframe positions.
    pub keyframes: Vec<KeyframePosition>,
}

impl VariantKeyframes {
    /// Create a new variant keyframes entry.
    #[must_use]
    pub fn new(variant_id: impl Into<String>, bitrate: u32) -> Self {
        Self {
            variant_id: variant_id.into(),
            bitrate,
            keyframes: Vec::new(),
        }
    }

    /// Add a keyframe position.
    pub fn add_keyframe(&mut self, kf: KeyframePosition) {
        self.keyframes.push(kf);
    }

    /// Get the number of keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    /// Get segment boundary timestamps (where keyframes mark segment starts).
    #[must_use]
    pub fn segment_boundaries(&self) -> Vec<Duration> {
        self.keyframes.iter().map(|kf| kf.timestamp).collect()
    }
}

// ---------------------------------------------------------------------------
// AlignmentTolerance
// ---------------------------------------------------------------------------

/// Configuration for keyframe alignment checking.
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Maximum allowed timestamp difference between corresponding keyframes
    /// across variants. Segments that differ by more than this are misaligned.
    pub tolerance: Duration,
    /// Whether to require the exact same number of keyframes across variants.
    pub require_same_count: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            tolerance: Duration::from_millis(100),
            require_same_count: true,
        }
    }
}

impl AlignmentConfig {
    /// Create a new alignment config with the given tolerance.
    #[must_use]
    pub fn new(tolerance: Duration) -> Self {
        Self {
            tolerance,
            require_same_count: true,
        }
    }

    /// Set whether same keyframe count is required.
    #[must_use]
    pub fn with_same_count(mut self, required: bool) -> Self {
        self.require_same_count = required;
        self
    }
}

// ---------------------------------------------------------------------------
// AlignmentResult
// ---------------------------------------------------------------------------

/// Result of checking keyframe alignment between two variants.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// The reference variant ID.
    pub reference_id: String,
    /// The checked variant ID.
    pub checked_id: String,
    /// Whether all keyframes are aligned within tolerance.
    pub is_aligned: bool,
    /// Number of misaligned keyframes found.
    pub misaligned_count: usize,
    /// Details of each misalignment.
    pub misalignments: Vec<KeyframeMisalignment>,
    /// Overall alignment score (0.0 = totally misaligned, 1.0 = perfect).
    pub alignment_score: f64,
}

/// A single keyframe misalignment between two variants.
#[derive(Debug, Clone)]
pub struct KeyframeMisalignment {
    /// Index of the keyframe in the reference variant.
    pub keyframe_index: usize,
    /// Timestamp in the reference variant.
    pub reference_timestamp: Duration,
    /// Timestamp in the checked variant.
    pub checked_timestamp: Duration,
    /// Absolute difference.
    pub difference: Duration,
}

// ---------------------------------------------------------------------------
// KeyframeAligner
// ---------------------------------------------------------------------------

/// Enforces keyframe alignment across all bitrate variants.
///
/// The typical workflow is:
/// 1. Set the reference variant (usually highest bitrate)
/// 2. Check each other variant against the reference
/// 3. If misaligned, use `compute_aligned_schedule` to generate a
///    canonical keyframe schedule that all variants should follow
pub struct KeyframeAligner {
    config: AlignmentConfig,
    /// The reference (canonical) variant keyframes.
    reference: Option<VariantKeyframes>,
}

impl KeyframeAligner {
    /// Create a new keyframe aligner with the given configuration.
    #[must_use]
    pub fn new(config: AlignmentConfig) -> Self {
        Self {
            config,
            reference: None,
        }
    }

    /// Set the reference variant (canonical keyframe schedule).
    /// Typically this is the highest-bitrate variant.
    pub fn set_reference(&mut self, variant: VariantKeyframes) {
        self.reference = Some(variant);
    }

    /// Get the reference variant.
    #[must_use]
    pub fn reference(&self) -> Option<&VariantKeyframes> {
        self.reference.as_ref()
    }

    /// Check whether a variant's keyframes are aligned with the reference.
    pub fn check_alignment(&self, variant: &VariantKeyframes) -> PackagerResult<AlignmentResult> {
        let reference = self.reference.as_ref().ok_or_else(|| {
            PackagerError::AlignmentFailed("No reference variant set".to_string())
        })?;

        let ref_kfs = &reference.keyframes;
        let var_kfs = &variant.keyframes;

        // Check count mismatch
        if self.config.require_same_count && ref_kfs.len() != var_kfs.len() {
            return Ok(AlignmentResult {
                reference_id: reference.variant_id.clone(),
                checked_id: variant.variant_id.clone(),
                is_aligned: false,
                misaligned_count: ref_kfs.len().abs_diff(var_kfs.len()),
                misalignments: Vec::new(),
                alignment_score: 0.0,
            });
        }

        let check_count = ref_kfs.len().min(var_kfs.len());
        let mut misalignments = Vec::new();
        let mut aligned_count = 0usize;

        for i in 0..check_count {
            let ref_ts = ref_kfs[i].timestamp;
            let var_ts = var_kfs[i].timestamp;
            let diff = if ref_ts > var_ts {
                ref_ts - var_ts
            } else {
                var_ts - ref_ts
            };

            if diff > self.config.tolerance {
                misalignments.push(KeyframeMisalignment {
                    keyframe_index: i,
                    reference_timestamp: ref_ts,
                    checked_timestamp: var_ts,
                    difference: diff,
                });
            } else {
                aligned_count += 1;
            }
        }

        let score = if check_count > 0 {
            aligned_count as f64 / check_count as f64
        } else {
            1.0
        };

        Ok(AlignmentResult {
            reference_id: reference.variant_id.clone(),
            checked_id: variant.variant_id.clone(),
            is_aligned: misalignments.is_empty(),
            misaligned_count: misalignments.len(),
            misalignments,
            alignment_score: score,
        })
    }

    /// Check alignment for multiple variants against the reference.
    pub fn check_all_variants(
        &self,
        variants: &[VariantKeyframes],
    ) -> PackagerResult<Vec<AlignmentResult>> {
        let mut results = Vec::with_capacity(variants.len());
        for variant in variants {
            results.push(self.check_alignment(variant)?);
        }
        Ok(results)
    }

    /// Compute a canonical aligned keyframe schedule from the reference.
    ///
    /// This produces a list of target timestamps that all variants should
    /// encode their keyframes at.
    pub fn compute_aligned_schedule(
        &self,
        target_segment_duration: Duration,
        total_duration: Duration,
    ) -> PackagerResult<Vec<Duration>> {
        if target_segment_duration.is_zero() {
            return Err(PackagerError::AlignmentFailed(
                "Target segment duration must be non-zero".to_string(),
            ));
        }

        let mut schedule = Vec::new();
        let mut current = Duration::ZERO;

        while current < total_duration {
            schedule.push(current);
            current += target_segment_duration;
        }

        Ok(schedule)
    }

    /// Snap a list of keyframe timestamps to the nearest canonical positions.
    ///
    /// Returns the snapped timestamps. Each keyframe is moved to the nearest
    /// position in the canonical schedule (within tolerance), or left in place
    /// if no nearby canonical position exists.
    #[must_use]
    pub fn snap_to_schedule(
        &self,
        keyframe_timestamps: &[Duration],
        canonical_schedule: &[Duration],
    ) -> Vec<Duration> {
        keyframe_timestamps
            .iter()
            .map(|&kf_ts| {
                let nearest = canonical_schedule.iter().min_by_key(|&&canon_ts| {
                    let diff = if kf_ts > canon_ts {
                        kf_ts - canon_ts
                    } else {
                        canon_ts - kf_ts
                    };
                    diff.as_nanos()
                });
                match nearest {
                    Some(&canon_ts) => {
                        let diff = if kf_ts > canon_ts {
                            kf_ts - canon_ts
                        } else {
                            canon_ts - kf_ts
                        };
                        if diff <= self.config.tolerance {
                            canon_ts
                        } else {
                            kf_ts
                        }
                    }
                    None => kf_ts,
                }
            })
            .collect()
    }

    /// Validate that all provided variants are mutually aligned.
    /// Returns `Ok(())` if all are aligned, or a descriptive error.
    pub fn enforce_alignment(&self, variants: &[VariantKeyframes]) -> PackagerResult<()> {
        let results = self.check_all_variants(variants)?;

        for result in &results {
            if !result.is_aligned {
                return Err(PackagerError::AlignmentFailed(format!(
                    "Variant '{}' has {} misaligned keyframes vs reference '{}' (score: {:.2})",
                    result.checked_id,
                    result.misaligned_count,
                    result.reference_id,
                    result.alignment_score,
                )));
            }
        }

        Ok(())
    }
}

impl Default for KeyframeAligner {
    fn default() -> Self {
        Self::new(AlignmentConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_variant(id: &str, bitrate: u32, timestamps_ms: &[u64]) -> VariantKeyframes {
        let mut v = VariantKeyframes::new(id, bitrate);
        for (i, &ts) in timestamps_ms.iter().enumerate() {
            v.add_keyframe(KeyframePosition::new(
                Duration::from_millis(ts),
                i as u64,
                ts * 1000,
            ));
        }
        v
    }

    #[test]
    fn test_keyframe_position_creation() {
        let kf = KeyframePosition::new(Duration::from_secs(6), 1, 120_000);
        assert_eq!(kf.timestamp, Duration::from_secs(6));
        assert_eq!(kf.segment_index, 1);
        assert_eq!(kf.byte_offset, 120_000);
    }

    #[test]
    fn test_variant_keyframes_creation() {
        let mut v = VariantKeyframes::new("1080p", 5_000_000);
        v.add_keyframe(KeyframePosition::new(Duration::ZERO, 0, 0));
        v.add_keyframe(KeyframePosition::new(Duration::from_secs(6), 1, 500_000));
        assert_eq!(v.keyframe_count(), 2);
    }

    #[test]
    fn test_variant_segment_boundaries() {
        let v = make_variant("720p", 3_000_000, &[0, 6000, 12000, 18000]);
        let bounds = v.segment_boundaries();
        assert_eq!(bounds.len(), 4);
        assert_eq!(bounds[2], Duration::from_secs(12));
    }

    #[test]
    fn test_alignment_config_default() {
        let cfg = AlignmentConfig::default();
        assert_eq!(cfg.tolerance, Duration::from_millis(100));
        assert!(cfg.require_same_count);
    }

    #[test]
    fn test_perfect_alignment() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000]);
        let check_v = make_variant("720p", 3_000_000, &[0, 6000, 12000]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::default());
        aligner.set_reference(ref_v);

        let result = aligner.check_alignment(&check_v).expect("should succeed");
        assert!(result.is_aligned);
        assert_eq!(result.misaligned_count, 0);
        assert!((result.alignment_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_within_tolerance_alignment() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000]);
        // 50ms offset is within 100ms tolerance
        let check_v = make_variant("720p", 3_000_000, &[0, 6050, 12030]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::new(Duration::from_millis(100)));
        aligner.set_reference(ref_v);

        let result = aligner.check_alignment(&check_v).expect("should succeed");
        assert!(result.is_aligned);
    }

    #[test]
    fn test_misaligned_keyframes() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000]);
        // 500ms offset exceeds 100ms tolerance
        let check_v = make_variant("720p", 3_000_000, &[0, 6500, 12000]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::default());
        aligner.set_reference(ref_v);

        let result = aligner.check_alignment(&check_v).expect("should succeed");
        assert!(!result.is_aligned);
        assert_eq!(result.misaligned_count, 1);
        assert_eq!(result.misalignments[0].keyframe_index, 1);
        assert_eq!(
            result.misalignments[0].difference,
            Duration::from_millis(500)
        );
    }

    #[test]
    fn test_count_mismatch() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000]);
        let check_v = make_variant("720p", 3_000_000, &[0, 6000]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::default());
        aligner.set_reference(ref_v);

        let result = aligner.check_alignment(&check_v).expect("should succeed");
        assert!(!result.is_aligned);
    }

    #[test]
    fn test_count_mismatch_allowed() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000]);
        let check_v = make_variant("720p", 3_000_000, &[0, 6000]);

        let config = AlignmentConfig::default().with_same_count(false);
        let mut aligner = KeyframeAligner::new(config);
        aligner.set_reference(ref_v);

        let result = aligner.check_alignment(&check_v).expect("should succeed");
        // 2/2 matched keyframes are aligned
        assert!(result.is_aligned);
        assert!((result.alignment_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_no_reference_error() {
        let aligner = KeyframeAligner::default();
        let check_v = make_variant("720p", 3_000_000, &[0, 6000]);
        let result = aligner.check_alignment(&check_v);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_all_variants() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000]);
        let v1 = make_variant("720p", 3_000_000, &[0, 6000, 12000]);
        let v2 = make_variant("480p", 1_500_000, &[0, 6000, 12000]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::default());
        aligner.set_reference(ref_v);

        let results = aligner
            .check_all_variants(&[v1, v2])
            .expect("should succeed");
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_aligned));
    }

    #[test]
    fn test_compute_aligned_schedule() {
        let aligner = KeyframeAligner::default();
        let schedule = aligner
            .compute_aligned_schedule(Duration::from_secs(6), Duration::from_secs(30))
            .expect("should succeed");
        // 0, 6, 12, 18, 24 (not 30 since 30 >= 30)
        assert_eq!(schedule.len(), 5);
        assert_eq!(schedule[0], Duration::ZERO);
        assert_eq!(schedule[4], Duration::from_secs(24));
    }

    #[test]
    fn test_compute_aligned_schedule_zero_duration_error() {
        let aligner = KeyframeAligner::default();
        let result = aligner.compute_aligned_schedule(Duration::ZERO, Duration::from_secs(30));
        assert!(result.is_err());
    }

    #[test]
    fn test_snap_to_schedule() {
        let aligner = KeyframeAligner::new(AlignmentConfig::new(Duration::from_millis(200)));
        let canonical = vec![
            Duration::ZERO,
            Duration::from_secs(6),
            Duration::from_secs(12),
        ];
        let actual = vec![
            Duration::from_millis(50),    // within 200ms of 0
            Duration::from_millis(6100),  // within 200ms of 6000
            Duration::from_millis(12500), // outside tolerance, keep original
        ];
        let snapped = aligner.snap_to_schedule(&actual, &canonical);
        assert_eq!(snapped[0], Duration::ZERO); // snapped
        assert_eq!(snapped[1], Duration::from_secs(6)); // snapped
        assert_eq!(snapped[2], Duration::from_millis(12500)); // not snapped
    }

    #[test]
    fn test_enforce_alignment_pass() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000]);
        let v1 = make_variant("720p", 3_000_000, &[0, 6000, 12000]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::default());
        aligner.set_reference(ref_v);

        let result = aligner.enforce_alignment(&[v1]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_enforce_alignment_fail() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000]);
        let v1 = make_variant("720p", 3_000_000, &[0, 7000, 12000]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::default());
        aligner.set_reference(ref_v);

        let result = aligner.enforce_alignment(&[v1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_alignment_score_partial() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000, 18000]);
        // 2 out of 4 aligned
        let check_v = make_variant("720p", 3_000_000, &[0, 7000, 12000, 19000]);

        let config = AlignmentConfig::new(Duration::from_millis(100));
        let mut aligner = KeyframeAligner::new(config);
        aligner.set_reference(ref_v);

        let result = aligner.check_alignment(&check_v).expect("should succeed");
        assert!(!result.is_aligned);
        assert!((result.alignment_score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_alignment_empty_variants() {
        let ref_v = make_variant("1080p", 5_000_000, &[]);
        let check_v = make_variant("720p", 3_000_000, &[]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::default());
        aligner.set_reference(ref_v);

        let result = aligner.check_alignment(&check_v).expect("should succeed");
        assert!(result.is_aligned);
        assert!((result.alignment_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_snap_empty_schedule() {
        let aligner = KeyframeAligner::default();
        let snapped = aligner.snap_to_schedule(&[Duration::from_secs(5)], &[]);
        assert_eq!(snapped[0], Duration::from_secs(5));
    }

    #[test]
    fn test_snap_empty_keyframes() {
        let aligner = KeyframeAligner::default();
        let snapped = aligner.snap_to_schedule(&[], &[Duration::ZERO, Duration::from_secs(6)]);
        assert!(snapped.is_empty());
    }

    #[test]
    fn test_multiple_misalignments_reported() {
        let ref_v = make_variant("1080p", 5_000_000, &[0, 6000, 12000, 18000, 24000]);
        let check_v = make_variant("720p", 3_000_000, &[0, 7000, 13000, 19000, 24000]);

        let mut aligner = KeyframeAligner::new(AlignmentConfig::default());
        aligner.set_reference(ref_v);

        let result = aligner.check_alignment(&check_v).expect("should succeed");
        assert_eq!(result.misaligned_count, 3);
        assert_eq!(result.misalignments.len(), 3);
    }

    #[test]
    fn test_reference_accessor() {
        let mut aligner = KeyframeAligner::default();
        assert!(aligner.reference().is_none());
        aligner.set_reference(make_variant("1080p", 5_000_000, &[0, 6000]));
        assert!(aligner.reference().is_some());
        assert_eq!(
            aligner.reference().map(|r| r.variant_id.as_str()),
            Some("1080p")
        );
    }
}
