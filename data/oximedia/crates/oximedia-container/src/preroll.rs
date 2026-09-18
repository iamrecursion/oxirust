//! Pre-roll seeking types for sample-accurate seek operations.
//!
//! Pre-roll decoding is needed for codecs that require decoding from
//! a keyframe before the target can be presented. This module provides
//! types that describe the decode chain from keyframe to target PTS.

use crate::seek::SeekIndexEntry;

/// Action to take for a sample during pre-roll seeking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreRollAction {
    /// Decode this sample but discard it (pre-roll frame).
    Decode,
    /// Decode and present this sample (target or post-target).
    Present,
}

/// A single sample in the pre-roll decode chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreRollSample {
    /// The sample entry from the seek index.
    pub entry: SeekIndexEntry,
    /// What to do with this sample after decoding.
    pub action: PreRollAction,
}

/// A complete pre-roll seek plan describing the decode chain from
/// keyframe to target PTS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreRollSeekPlan {
    /// The keyframe to start decoding from.
    pub keyframe: SeekIndexEntry,
    /// The target PTS that was requested.
    pub target_pts: i64,
    /// Ordered list of samples to decode (from keyframe to target).
    pub samples: Vec<PreRollSample>,
    /// Number of samples to decode-and-discard.
    pub discard_count: u32,
    /// Number of samples to present.
    pub present_count: u32,
    /// File offset to seek to (keyframe position).
    pub file_offset: u64,
}

impl PreRollSeekPlan {
    /// Returns true if no pre-roll is needed (target is on a keyframe).
    #[must_use]
    pub fn is_immediate(&self) -> bool {
        self.discard_count == 0
    }

    /// Returns the total number of samples in the decode chain.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.samples.len()
    }

    /// Returns an iterator over only the samples to discard.
    pub fn discard_samples(&self) -> impl Iterator<Item = &PreRollSample> {
        self.samples
            .iter()
            .filter(|s| matches!(s.action, PreRollAction::Decode))
    }

    /// Returns an iterator over only the samples to present.
    pub fn present_samples(&self) -> impl Iterator<Item = &PreRollSample> {
        self.samples
            .iter()
            .filter(|s| matches!(s.action, PreRollAction::Present))
    }

    /// Returns the PTS of the first sample that will be presented.
    #[must_use]
    pub fn first_present_pts(&self) -> Option<i64> {
        self.present_samples().next().map(|s| s.entry.pts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seek::{SampleAccurateSeeker, SampleIndex, SeekIndexEntry};

    fn build_sample_index() -> SampleIndex {
        let mut idx = SampleIndex::new(90000);
        for i in 0u32..10 {
            let pts = i64::from(i) * 3000;
            let entry = if i % 5 == 0 {
                SeekIndexEntry::keyframe(pts, pts, u64::from(i) * 500, 200, 3000, i)
            } else {
                SeekIndexEntry::non_keyframe(pts, pts, u64::from(i) * 500, 200, 3000, i)
            };
            idx.add_entry(entry);
        }
        idx.finalize();
        idx
    }

    #[test]
    fn test_preroll_action_variants() {
        assert_ne!(PreRollAction::Decode, PreRollAction::Present);
    }

    #[test]
    fn test_preroll_seek_plan_is_immediate() {
        let kf = SeekIndexEntry::keyframe(0, 0, 0, 200, 3000, 0);
        let plan = PreRollSeekPlan {
            keyframe: kf,
            target_pts: 0,
            samples: vec![PreRollSample {
                entry: kf,
                action: PreRollAction::Present,
            }],
            discard_count: 0,
            present_count: 1,
            file_offset: 0,
        };
        assert!(plan.is_immediate());
        assert_eq!(plan.total_samples(), 1);
    }

    #[test]
    fn test_preroll_plan_with_discards() {
        let kf = SeekIndexEntry::keyframe(0, 0, 0, 200, 3000, 0);
        let inter = SeekIndexEntry::non_keyframe(3000, 3000, 200, 200, 3000, 1);
        let target = SeekIndexEntry::non_keyframe(6000, 6000, 400, 200, 3000, 2);

        let plan = PreRollSeekPlan {
            keyframe: kf,
            target_pts: 6000,
            samples: vec![
                PreRollSample {
                    entry: kf,
                    action: PreRollAction::Decode,
                },
                PreRollSample {
                    entry: inter,
                    action: PreRollAction::Decode,
                },
                PreRollSample {
                    entry: target,
                    action: PreRollAction::Present,
                },
            ],
            discard_count: 2,
            present_count: 1,
            file_offset: 0,
        };

        assert!(!plan.is_immediate());
        assert_eq!(plan.total_samples(), 3);
        assert_eq!(plan.discard_samples().count(), 2);
        assert_eq!(plan.present_samples().count(), 1);
        assert_eq!(plan.first_present_pts(), Some(6000));
    }

    // ── PreRoll seek integration tests ──────────────────────────────────

    #[test]
    fn test_preroll_seek_on_keyframe() {
        let mut seeker = SampleAccurateSeeker::new();
        seeker.add_stream(0, build_sample_index());

        let plan = seeker.plan_preroll_seek(0, 0, None).expect("should plan");
        assert!(plan.is_immediate());
        assert_eq!(plan.keyframe.pts, 0);
        assert!(plan.present_count >= 1);
    }

    #[test]
    fn test_preroll_seek_between_keyframes() {
        let mut seeker = SampleAccurateSeeker::new();
        seeker.add_stream(0, build_sample_index());

        let plan = seeker
            .plan_preroll_seek(0, 9000, None)
            .expect("should plan");
        assert_eq!(plan.keyframe.pts, 0);
        assert!(!plan.is_immediate());
        assert!(plan.discard_count > 0);
        assert!(plan.present_count >= 1);
        let first_present = plan.first_present_pts().expect("should have present");
        assert!(first_present <= 9000);
    }

    #[test]
    fn test_preroll_seek_with_max_limit() {
        let mut seeker = SampleAccurateSeeker::new();
        seeker.add_stream(0, build_sample_index());

        let plan = seeker
            .plan_preroll_seek(0, 12000, Some(2))
            .expect("should plan");
        assert!(plan.total_samples() <= 3);
    }

    #[test]
    fn test_preroll_seek_unknown_stream() {
        let seeker = SampleAccurateSeeker::new();
        let plan = seeker.plan_preroll_seek(99, 0, None);
        assert!(plan.is_none());
    }

    #[test]
    fn test_preroll_count() {
        let mut seeker = SampleAccurateSeeker::new();
        seeker.add_stream(0, build_sample_index());

        let count = seeker.preroll_count(0, 9000);
        assert!(count.is_some());
        assert!(count.expect("should have count") > 0);
    }

    #[test]
    fn test_preroll_plan_iterators() {
        let mut seeker = SampleAccurateSeeker::new();
        seeker.add_stream(0, build_sample_index());

        let plan = seeker
            .plan_preroll_seek(0, 9000, None)
            .expect("should plan");

        let discard_count = plan.discard_samples().count();
        let present_count = plan.present_samples().count();

        assert_eq!(discard_count, plan.discard_count as usize);
        assert_eq!(present_count, plan.present_count as usize);
        assert_eq!(discard_count + present_count, plan.total_samples());
    }
}
