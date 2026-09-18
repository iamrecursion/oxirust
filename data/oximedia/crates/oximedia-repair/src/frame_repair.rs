//! Frame-level damage detection and repair for video media.
//!
//! This module defines types and logic for classifying the kind of damage
//! present in a video frame and selecting a repair approach, from lightweight
//! interpolation to full synthetic reconstruction.

#![allow(dead_code)]

/// Categories of damage that can afflict a video frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameDamageType {
    /// The frame is completely missing from the stream.
    Missing,
    /// Only a rectangular sub-region of the frame is corrupted.
    PartialCorruption {
        /// X offset of the damaged region in pixels.
        x: u32,
        /// Y offset of the damaged region in pixels.
        y: u32,
        /// Width of the damaged region in pixels.
        width: u32,
        /// Height of the damaged region in pixels.
        height: u32,
    },
    /// Codec bitstream errors caused block-level artefacts.
    BitStreamError,
    /// The frame was decoded but the checksum failed.
    ChecksumMismatch,
    /// Colour information is wrong (e.g. wrong chroma plane).
    ColorCorruption,
    /// The frame is present but entirely black (unexpected).
    UnexpectedBlack,
}

/// Quality level of a repaired frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RepairQuality {
    /// No repair was possible; the original damaged data is retained.
    None,
    /// Basic concealment (e.g. repeat previous frame).
    Concealment,
    /// Temporal interpolation between neighbouring frames.
    TemporalInterpolation,
    /// Spatial inpainting within the damaged region.
    SpatialInpainting,
    /// Full synthetic reconstruction using surrounding context.
    FullReconstruction,
}

/// Information about a single damaged frame.
#[derive(Debug, Clone)]
pub struct DamagedFrame {
    /// Frame index in the video stream.
    pub frame_index: u64,
    /// Presentation timestamp (in stream time units).
    pub pts: i64,
    /// Type of damage detected.
    pub damage_type: FrameDamageType,
    /// Estimated percentage of the frame that is damaged (0–100).
    pub damage_pct: f32,
}

/// Result of repairing a single frame.
#[derive(Debug, Clone)]
pub struct FrameRepairResult {
    /// Original frame information.
    pub frame: DamagedFrame,
    /// Quality achieved by the repair.
    pub quality: RepairQuality,
    /// Human-readable description of what was done.
    pub description: String,
}

/// Configuration for frame repair operations.
#[derive(Debug, Clone)]
pub struct FrameRepairConfig {
    /// Maximum damage percentage before we give up and conceal.
    pub max_damage_pct_for_inpaint: f32,
    /// Whether to allow synthetic reconstruction for totally missing frames.
    pub allow_full_reconstruction: bool,
    /// Number of reference frames to use for temporal interpolation.
    pub temporal_window: usize,
}

impl Default for FrameRepairConfig {
    fn default() -> Self {
        Self {
            max_damage_pct_for_inpaint: 25.0,
            allow_full_reconstruction: true,
            temporal_window: 4,
        }
    }
}

/// Engine for detecting and repairing damaged video frames.
#[derive(Debug, Default)]
pub struct FrameRepairer {
    config: FrameRepairConfig,
    repaired_frames: Vec<FrameRepairResult>,
}

impl FrameRepairer {
    /// Create a new `FrameRepairer` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new `FrameRepairer` with custom configuration.
    pub fn with_config(config: FrameRepairConfig) -> Self {
        Self {
            config,
            repaired_frames: Vec::new(),
        }
    }

    /// Choose the repair quality target for a damaged frame.
    pub fn choose_quality(&self, frame: &DamagedFrame) -> RepairQuality {
        match frame.damage_type {
            FrameDamageType::Missing => {
                if self.config.allow_full_reconstruction {
                    RepairQuality::FullReconstruction
                } else {
                    RepairQuality::Concealment
                }
            }
            FrameDamageType::PartialCorruption { .. } | FrameDamageType::BitStreamError => {
                if frame.damage_pct <= self.config.max_damage_pct_for_inpaint {
                    RepairQuality::SpatialInpainting
                } else {
                    RepairQuality::TemporalInterpolation
                }
            }
            FrameDamageType::ChecksumMismatch => RepairQuality::TemporalInterpolation,
            FrameDamageType::ColorCorruption => RepairQuality::SpatialInpainting,
            FrameDamageType::UnexpectedBlack => RepairQuality::Concealment,
        }
    }

    /// Repair a damaged frame (simulation — records the decision).
    pub fn repair(&mut self, frame: DamagedFrame) -> FrameRepairResult {
        let quality = self.choose_quality(&frame);
        let description = format!(
            "Frame {} repaired with {:?} (damage type: {:?}, {:.1}% damaged)",
            frame.frame_index, quality, frame.damage_type, frame.damage_pct
        );
        let result = FrameRepairResult {
            frame,
            quality,
            description,
        };
        self.repaired_frames.push(result.clone());
        result
    }

    /// Repair a batch of damaged frames.
    pub fn repair_batch(&mut self, frames: Vec<DamagedFrame>) -> Vec<FrameRepairResult> {
        frames.into_iter().map(|f| self.repair(f)).collect()
    }

    /// Return all repair results recorded so far.
    pub fn history(&self) -> &[FrameRepairResult] {
        &self.repaired_frames
    }

    /// Average repair quality across all frames (as ordinal value).
    #[allow(clippy::cast_precision_loss)]
    pub fn average_quality_score(&self) -> f64 {
        if self.repaired_frames.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.repaired_frames.iter().map(|r| r.quality as u64).sum();
        sum as f64 / self.repaired_frames.len() as f64
    }

    /// Count of frames that achieved at least the given quality.
    pub fn count_at_least(&self, min_quality: RepairQuality) -> usize {
        self.repaired_frames
            .iter()
            .filter(|r| r.quality >= min_quality)
            .count()
    }

    /// Clear repair history.
    pub fn clear_history(&mut self) {
        self.repaired_frames.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn missing_frame(idx: u64) -> DamagedFrame {
        DamagedFrame {
            frame_index: idx,
            pts: idx as i64 * 1000,
            damage_type: FrameDamageType::Missing,
            damage_pct: 100.0,
        }
    }

    fn partial_frame(idx: u64, pct: f32) -> DamagedFrame {
        DamagedFrame {
            frame_index: idx,
            pts: idx as i64 * 1000,
            damage_type: FrameDamageType::PartialCorruption {
                x: 0,
                y: 0,
                width: 100,
                height: 100,
            },
            damage_pct: pct,
        }
    }

    #[test]
    fn test_missing_frame_full_reconstruction() {
        let r = FrameRepairer::new();
        let f = missing_frame(0);
        assert_eq!(r.choose_quality(&f), RepairQuality::FullReconstruction);
    }

    #[test]
    fn test_missing_frame_concealment_when_disabled() {
        let cfg = FrameRepairConfig {
            allow_full_reconstruction: false,
            ..Default::default()
        };
        let r = FrameRepairer::with_config(cfg);
        let f = missing_frame(0);
        assert_eq!(r.choose_quality(&f), RepairQuality::Concealment);
    }

    #[test]
    fn test_partial_low_damage_inpainting() {
        let r = FrameRepairer::new();
        let f = partial_frame(1, 10.0);
        assert_eq!(r.choose_quality(&f), RepairQuality::SpatialInpainting);
    }

    #[test]
    fn test_partial_high_damage_temporal() {
        let r = FrameRepairer::new();
        let f = partial_frame(2, 80.0);
        assert_eq!(r.choose_quality(&f), RepairQuality::TemporalInterpolation);
    }

    #[test]
    fn test_checksum_mismatch_temporal() {
        let r = FrameRepairer::new();
        let f = DamagedFrame {
            frame_index: 3,
            pts: 3000,
            damage_type: FrameDamageType::ChecksumMismatch,
            damage_pct: 5.0,
        };
        assert_eq!(r.choose_quality(&f), RepairQuality::TemporalInterpolation);
    }

    #[test]
    fn test_color_corruption_inpainting() {
        let r = FrameRepairer::new();
        let f = DamagedFrame {
            frame_index: 4,
            pts: 4000,
            damage_type: FrameDamageType::ColorCorruption,
            damage_pct: 30.0,
        };
        assert_eq!(r.choose_quality(&f), RepairQuality::SpatialInpainting);
    }

    #[test]
    fn test_unexpected_black_concealment() {
        let r = FrameRepairer::new();
        let f = DamagedFrame {
            frame_index: 5,
            pts: 5000,
            damage_type: FrameDamageType::UnexpectedBlack,
            damage_pct: 100.0,
        };
        assert_eq!(r.choose_quality(&f), RepairQuality::Concealment);
    }

    #[test]
    fn test_repair_records_history() {
        let mut r = FrameRepairer::new();
        r.repair(missing_frame(0));
        assert_eq!(r.history().len(), 1);
    }

    #[test]
    fn test_repair_batch() {
        let mut r = FrameRepairer::new();
        let frames = vec![missing_frame(0), missing_frame(1), partial_frame(2, 5.0)];
        let results = r.repair_batch(frames);
        assert_eq!(results.len(), 3);
        assert_eq!(r.history().len(), 3);
    }

    #[test]
    fn test_clear_history() {
        let mut r = FrameRepairer::new();
        r.repair(missing_frame(0));
        r.clear_history();
        assert!(r.history().is_empty());
    }

    #[test]
    fn test_average_quality_empty() {
        let r = FrameRepairer::new();
        assert_eq!(r.average_quality_score(), 0.0);
    }

    #[test]
    fn test_count_at_least() {
        let mut r = FrameRepairer::new();
        r.repair(missing_frame(0)); // FullReconstruction
        r.repair(partial_frame(1, 5.0)); // SpatialInpainting
        r.repair(DamagedFrame {
            frame_index: 2,
            pts: 2000,
            damage_type: FrameDamageType::UnexpectedBlack,
            damage_pct: 100.0,
        }); // Concealment
        assert_eq!(r.count_at_least(RepairQuality::TemporalInterpolation), 2);
        assert_eq!(r.count_at_least(RepairQuality::FullReconstruction), 1);
    }

    #[test]
    fn test_quality_ordering() {
        assert!(RepairQuality::None < RepairQuality::Concealment);
        assert!(RepairQuality::Concealment < RepairQuality::TemporalInterpolation);
        assert!(RepairQuality::TemporalInterpolation < RepairQuality::SpatialInpainting);
        assert!(RepairQuality::SpatialInpainting < RepairQuality::FullReconstruction);
    }

    #[test]
    fn test_description_contains_frame_index() {
        let mut r = FrameRepairer::new();
        let result = r.repair(missing_frame(42));
        assert!(
            result.description.contains("42"),
            "description should contain frame index"
        );
    }

    #[test]
    fn test_average_quality_score_nonzero() {
        let mut r = FrameRepairer::new();
        r.repair(missing_frame(0)); // quality = 4 (FullReconstruction)
        let score = r.average_quality_score();
        assert!(score > 0.0);
    }
}
