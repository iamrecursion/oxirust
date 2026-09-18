//! Cut-point detection for multi-camera editing.
//!
//! Provides heuristic algorithms for automatically identifying candidate edit
//! points based on audio energy, motion energy, and dialogue activity.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ── CutPointKind ──────────────────────────────────────────────────────────────

/// Category of a detected cut point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutPointKind {
    /// Audio energy crosses a threshold (e.g. clap, musical beat).
    AudioEnergy,
    /// Motion in the frame exceeds the threshold (e.g. camera move end).
    MotionCut,
    /// A dialogue exchange creates a natural speech edit point.
    DialogueCut,
}

// ── CutPoint ──────────────────────────────────────────────────────────────────

/// A single candidate cut point in the timeline.
#[derive(Debug, Clone)]
pub struct CutPoint {
    /// Frame index at which the cut is suggested.
    pub frame_idx: u64,
    /// Category of the detected cut.
    pub kind: CutPointKind,
    /// Confidence score (0.0 – 1.0).
    pub confidence: f32,
}

impl CutPoint {
    /// Create a new `CutPoint`, clamping the confidence to \[0.0, 1.0\].
    #[must_use]
    pub fn new(frame_idx: u64, kind: CutPointKind, confidence: f32) -> Self {
        Self {
            frame_idx,
            kind,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

// ── AudioEnergyCutDetector ────────────────────────────────────────────────────

/// Detects cut points by looking for sudden rises in per-frame RMS energy.
#[derive(Debug, Clone)]
pub struct AudioEnergyCutDetector {
    /// Energy rise ratio that triggers a cut (e.g. 2.0 = energy doubles).
    pub threshold_ratio: f32,
    /// Minimum number of quiet frames required before a cut can be detected.
    pub min_quiet_frames: usize,
}

impl AudioEnergyCutDetector {
    /// Create a new detector.
    #[must_use]
    pub fn new(threshold_ratio: f32, min_quiet_frames: usize) -> Self {
        Self {
            threshold_ratio: threshold_ratio.max(1.0),
            min_quiet_frames,
        }
    }

    /// Detect audio-energy cut points in a slice of per-frame RMS values.
    ///
    /// Returns a list of `CutPoint`s sorted by frame index.
    #[must_use]
    pub fn detect(&self, energy: &[f32]) -> Vec<CutPoint> {
        if energy.len() < 2 {
            return Vec::new();
        }
        let mut cuts = Vec::new();
        let mut quiet_count = 0usize;
        let mut prev = energy[0];

        for (i, &e) in energy.iter().enumerate().skip(1) {
            if prev < f32::EPSILON {
                quiet_count += 1;
                prev = e;
                continue;
            }
            let ratio = e / prev;
            if ratio >= self.threshold_ratio && quiet_count >= self.min_quiet_frames {
                // confidence scales with how much the ratio exceeds the threshold
                let confidence = ((ratio - self.threshold_ratio) / self.threshold_ratio)
                    .min(1.0)
                    .max(0.0);
                cuts.push(CutPoint::new(
                    i as u64,
                    CutPointKind::AudioEnergy,
                    confidence,
                ));
                quiet_count = 0;
            } else if e <= prev {
                // Energy is not rising: count as a quiet frame
                quiet_count += 1;
            }
            prev = e;
        }
        cuts
    }
}

// ── MotionCutDetector ─────────────────────────────────────────────────────────

/// Detects cut points when inter-frame motion energy drops below a threshold
/// (end of a camera move).
#[derive(Debug, Clone)]
pub struct MotionCutDetector {
    /// Motion magnitude threshold below which a frame is considered static.
    pub static_threshold: f32,
    /// Number of consecutive static frames needed to register a cut.
    pub static_run_length: usize,
}

impl MotionCutDetector {
    /// Create a new `MotionCutDetector`.
    #[must_use]
    pub fn new(static_threshold: f32, static_run_length: usize) -> Self {
        Self {
            static_threshold: static_threshold.max(0.0),
            static_run_length: static_run_length.max(1),
        }
    }

    /// Detect motion-cut points from a slice of per-frame motion magnitudes.
    #[must_use]
    pub fn detect(&self, motion: &[f32]) -> Vec<CutPoint> {
        let mut cuts = Vec::new();
        let mut static_count = 0usize;
        let mut in_motion = false;

        for (i, &m) in motion.iter().enumerate() {
            if m > self.static_threshold {
                in_motion = true;
                static_count = 0;
            } else {
                static_count += 1;
                if in_motion && static_count == self.static_run_length {
                    // The cut is placed at the first static frame of the run
                    let cut_idx = (i + 1).saturating_sub(self.static_run_length) as u64;
                    cuts.push(CutPoint::new(cut_idx, CutPointKind::MotionCut, 0.75));
                    in_motion = false;
                }
            }
        }
        cuts
    }
}

// ── DialogueActivity ─────────────────────────────────────────────────────────

/// Per-frame dialogue activity descriptor.
#[derive(Debug, Clone, Copy)]
pub struct DialogueActivity {
    /// Frame index.
    pub frame_idx: u64,
    /// `true` when speech is detected in the audio of this frame.
    pub speech_active: bool,
    /// Speaker label index (0 for silence / unknown).
    pub speaker_id: u32,
}

// ── DialogueCutDetector ───────────────────────────────────────────────────────

/// Detects natural cut points at speaker transitions in dialogue.
#[derive(Debug, Clone)]
pub struct DialogueCutDetector {
    /// Minimum silence gap (in frames) between two speech segments to count as
    /// a potential cut point.
    pub min_gap_frames: u64,
}

impl DialogueCutDetector {
    /// Create a new `DialogueCutDetector`.
    #[must_use]
    pub fn new(min_gap_frames: u64) -> Self {
        Self {
            min_gap_frames: min_gap_frames.max(1),
        }
    }

    /// Detect dialogue-based cut points.
    ///
    /// A cut is emitted when:
    /// - A silence gap >= `min_gap_frames` separates two speech segments, **or**
    /// - The speaker changes between consecutive speech frames.
    #[must_use]
    pub fn detect(&self, activity: &[DialogueActivity]) -> Vec<CutPoint> {
        if activity.is_empty() {
            return Vec::new();
        }
        let mut cuts = Vec::new();
        let mut last_speech_frame: Option<u64> = None;
        let mut last_speaker: u32 = 0;

        for a in activity {
            if a.speech_active {
                if let Some(last) = last_speech_frame {
                    let gap = a.frame_idx.saturating_sub(last);
                    if gap >= self.min_gap_frames {
                        cuts.push(CutPoint::new(a.frame_idx, CutPointKind::DialogueCut, 0.85));
                    } else if a.speaker_id != last_speaker && last_speaker != 0 {
                        cuts.push(CutPoint::new(a.frame_idx, CutPointKind::DialogueCut, 0.70));
                    }
                }
                last_speech_frame = Some(a.frame_idx);
                last_speaker = a.speaker_id;
            }
        }
        cuts
    }
}

// ── CutPointList ─────────────────────────────────────────────────────────────

/// An ordered collection of cut points with filtering helpers.
#[derive(Debug, Default)]
pub struct CutPointList {
    points: Vec<CutPoint>,
}

impl CutPointList {
    /// Create an empty list.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a cut point.
    pub fn add(&mut self, cp: CutPoint) {
        self.points.push(cp);
    }

    /// Merge cut points from multiple sources, sorted by frame index.
    #[must_use]
    pub fn from_merged(sources: Vec<Vec<CutPoint>>) -> Self {
        let mut all: Vec<CutPoint> = sources.into_iter().flatten().collect();
        all.sort_by_key(|c| c.frame_idx);
        Self { points: all }
    }

    /// Filter to only cut points at or above `min_confidence`.
    #[must_use]
    pub fn filter_by_confidence(&self, min_confidence: f32) -> Vec<&CutPoint> {
        self.points
            .iter()
            .filter(|c| c.confidence >= min_confidence)
            .collect()
    }

    /// Filter to cut points of a specific `kind`.
    #[must_use]
    pub fn filter_by_kind(&self, kind: CutPointKind) -> Vec<&CutPoint> {
        self.points.iter().filter(|c| c.kind == kind).collect()
    }

    /// Number of cut points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// `true` if there are no cut points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return all cut points as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[CutPoint] {
        &self.points
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CutPoint ────────────────────────────────────────────────────────────

    #[test]
    fn test_cut_point_confidence_clamped_high() {
        let cp = CutPoint::new(10, CutPointKind::AudioEnergy, 2.5);
        assert_eq!(cp.confidence, 1.0);
    }

    #[test]
    fn test_cut_point_confidence_clamped_low() {
        let cp = CutPoint::new(0, CutPointKind::MotionCut, -1.0);
        assert_eq!(cp.confidence, 0.0);
    }

    // ── AudioEnergyCutDetector ───────────────────────────────────────────────

    #[test]
    fn test_audio_energy_no_cuts_flat_signal() {
        let detector = AudioEnergyCutDetector::new(2.0, 2);
        let energy = vec![0.5_f32; 20];
        let cuts = detector.detect(&energy);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_audio_energy_detects_sudden_rise() {
        let detector = AudioEnergyCutDetector::new(2.0, 2);
        // Quiet for 3 frames, then energy spikes
        let mut energy = vec![0.1_f32; 4];
        energy.push(0.5);
        let cuts = detector.detect(&energy);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].kind, CutPointKind::AudioEnergy);
    }

    #[test]
    fn test_audio_energy_below_quiet_threshold_no_cut() {
        let detector = AudioEnergyCutDetector::new(2.0, 5);
        // Only 2 quiet frames before rise – below min_quiet_frames=5
        let mut energy = vec![0.1_f32; 2];
        energy.push(0.5);
        let cuts = detector.detect(&energy);
        assert!(cuts.is_empty());
    }

    // ── MotionCutDetector ────────────────────────────────────────────────────

    #[test]
    fn test_motion_cut_detects_end_of_move() {
        let detector = MotionCutDetector::new(0.3, 3);
        let mut motion = vec![0.8_f32; 5];
        // 3 static frames follow
        motion.extend_from_slice(&[0.1, 0.1, 0.1]);
        let cuts = detector.detect(&motion);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].kind, CutPointKind::MotionCut);
    }

    #[test]
    fn test_motion_cut_no_cut_all_motion() {
        let detector = MotionCutDetector::new(0.3, 3);
        let motion = vec![0.9_f32; 10];
        let cuts = detector.detect(&motion);
        assert!(cuts.is_empty());
    }

    #[test]
    fn test_motion_cut_no_cut_run_too_short() {
        let detector = MotionCutDetector::new(0.3, 5);
        // In motion, then only 2 static frames (< 5 required)
        let mut motion = vec![0.9_f32; 4];
        motion.extend_from_slice(&[0.1, 0.1]);
        let cuts = detector.detect(&motion);
        assert!(cuts.is_empty());
    }

    // ── DialogueCutDetector ──────────────────────────────────────────────────

    #[test]
    fn test_dialogue_cut_speaker_change() {
        let detector = DialogueCutDetector::new(10);
        let activity = vec![
            DialogueActivity {
                frame_idx: 0,
                speech_active: true,
                speaker_id: 1,
            },
            DialogueActivity {
                frame_idx: 1,
                speech_active: true,
                speaker_id: 2,
            },
        ];
        let cuts = detector.detect(&activity);
        // Speaker changed
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0].kind, CutPointKind::DialogueCut);
    }

    #[test]
    fn test_dialogue_cut_gap_between_speakers() {
        let detector = DialogueCutDetector::new(3);
        let activity = vec![
            DialogueActivity {
                frame_idx: 0,
                speech_active: true,
                speaker_id: 1,
            },
            DialogueActivity {
                frame_idx: 5,
                speech_active: true,
                speaker_id: 1,
            },
        ];
        let cuts = detector.detect(&activity);
        // Gap = 5 >= min 3
        assert_eq!(cuts.len(), 1);
    }

    #[test]
    fn test_dialogue_cut_no_cut_same_speaker_short_gap() {
        let detector = DialogueCutDetector::new(10);
        let activity = vec![
            DialogueActivity {
                frame_idx: 0,
                speech_active: true,
                speaker_id: 1,
            },
            DialogueActivity {
                frame_idx: 2,
                speech_active: true,
                speaker_id: 1,
            },
        ];
        let cuts = detector.detect(&activity);
        assert!(cuts.is_empty());
    }

    // ── CutPointList ─────────────────────────────────────────────────────────

    #[test]
    fn test_cut_point_list_filter_confidence() {
        let mut list = CutPointList::new();
        list.add(CutPoint::new(1, CutPointKind::AudioEnergy, 0.9));
        list.add(CutPoint::new(2, CutPointKind::MotionCut, 0.4));
        let high = list.filter_by_confidence(0.8);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].frame_idx, 1);
    }

    #[test]
    fn test_cut_point_list_filter_kind() {
        let mut list = CutPointList::new();
        list.add(CutPoint::new(1, CutPointKind::AudioEnergy, 0.9));
        list.add(CutPoint::new(2, CutPointKind::DialogueCut, 0.7));
        let dialogue = list.filter_by_kind(CutPointKind::DialogueCut);
        assert_eq!(dialogue.len(), 1);
        assert_eq!(dialogue[0].frame_idx, 2);
    }

    #[test]
    fn test_cut_point_list_from_merged_sorted() {
        let a = vec![CutPoint::new(50, CutPointKind::AudioEnergy, 0.8)];
        let b = vec![CutPoint::new(10, CutPointKind::MotionCut, 0.7)];
        let list = CutPointList::from_merged(vec![a, b]);
        assert_eq!(list.len(), 2);
        assert_eq!(list.as_slice()[0].frame_idx, 10);
        assert_eq!(list.as_slice()[1].frame_idx, 50);
    }

    #[test]
    fn test_cut_point_list_empty() {
        let list = CutPointList::new();
        assert!(list.is_empty());
    }
}
