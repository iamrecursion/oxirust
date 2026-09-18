//! Dolby Vision level analysis – parses and reports which DV metadata levels
//! are present in an RPU stream and validates their size requirements.

#![allow(dead_code)]

/// Dolby Vision metadata level identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DvLevel {
    /// Level 1 – frame-level brightness metadata (required).
    Level1,
    /// Level 2 – trim passes for individual target displays.
    Level2,
    /// Level 3 – reserved / scene-level.
    Level3,
    /// Level 4 – alternative trim passes.
    Level4,
    /// Level 5 – active area offsets.
    Level5,
    /// Level 6 – fallback HDR10 / SDR metadata.
    Level6,
    /// Level 8 – target display metadata.
    Level8,
    /// Level 9 – source display characteristics.
    Level9,
}

impl DvLevel {
    /// Returns the minimum expected serialised byte size for this level.
    #[must_use]
    pub fn metadata_size(self) -> usize {
        match self {
            Self::Level1 => 3,
            Self::Level2 => 11,
            Self::Level3 => 2,
            Self::Level4 => 3,
            Self::Level5 => 4,
            Self::Level6 => 8,
            Self::Level8 => 10,
            Self::Level9 => 9,
        }
    }

    /// Returns `true` if this level is required for a compliant stream.
    #[must_use]
    pub fn is_required(self) -> bool {
        matches!(self, Self::Level1)
    }

    /// Returns `true` if this level carries per-frame (not per-scene) data.
    #[must_use]
    pub fn is_per_frame(self) -> bool {
        matches!(self, Self::Level1 | Self::Level2)
    }

    /// All known levels in ascending numeric order.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Level1,
            Self::Level2,
            Self::Level3,
            Self::Level4,
            Self::Level5,
            Self::Level6,
            Self::Level8,
            Self::Level9,
        ]
    }
}

// ---------------------------------------------------------------------------

/// Summary report produced by [`LevelAnalyzer`].
#[derive(Debug, Clone, Default)]
pub struct LevelReport {
    /// Levels found in the analysed content.
    pub present: Vec<DvLevel>,
    /// Levels that were expected but absent.
    pub missing_required: Vec<DvLevel>,
    /// Total number of RPU frames analysed.
    pub frame_count: u32,
}

impl LevelReport {
    /// Returns `true` if all required levels are present.
    #[must_use]
    pub fn has_required_levels(&self) -> bool {
        self.missing_required.is_empty()
    }

    /// Returns `true` if the given level appears in the report.
    #[must_use]
    pub fn contains(&self, level: DvLevel) -> bool {
        self.present.contains(&level)
    }

    /// Human-readable one-line summary.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} frames analysed, {} levels present, {} required missing",
            self.frame_count,
            self.present.len(),
            self.missing_required.len(),
        )
    }
}

// ---------------------------------------------------------------------------

/// Analyses a set of level-presence flags across an RPU stream.
#[derive(Debug, Default)]
pub struct LevelAnalyzer {
    seen: Vec<DvLevel>,
    frames: u32,
}

impl LevelAnalyzer {
    /// Create a new analyser.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `level` was observed in the current frame. Call once per
    /// frame per level that is present.
    pub fn observe(&mut self, level: DvLevel) {
        self.frames += 1;
        if !self.seen.contains(&level) {
            self.seen.push(level);
        }
    }

    /// Record a whole set of levels observed in one frame.
    pub fn observe_frame(&mut self, levels: &[DvLevel]) {
        self.frames += 1;
        for &l in levels {
            if !self.seen.contains(&l) {
                self.seen.push(l);
            }
        }
    }

    /// Finalise analysis and return the [`LevelReport`].
    #[must_use]
    pub fn analyze(&self) -> LevelReport {
        let missing_required = DvLevel::all()
            .into_iter()
            .filter(|l| l.is_required() && !self.seen.contains(l))
            .collect();

        LevelReport {
            present: self.seen.clone(),
            missing_required,
            frame_count: self.frames,
        }
    }

    /// Reset the analyser to its initial state.
    pub fn reset(&mut self) {
        self.seen.clear();
        self.frames = 0;
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_size_level1() {
        assert_eq!(DvLevel::Level1.metadata_size(), 3);
    }

    #[test]
    fn test_metadata_size_level2() {
        assert_eq!(DvLevel::Level2.metadata_size(), 11);
    }

    #[test]
    fn test_metadata_size_level8() {
        assert_eq!(DvLevel::Level8.metadata_size(), 10);
    }

    #[test]
    fn test_metadata_size_level9() {
        assert_eq!(DvLevel::Level9.metadata_size(), 9);
    }

    #[test]
    fn test_level_is_required_only_level1() {
        assert!(DvLevel::Level1.is_required());
        assert!(!DvLevel::Level2.is_required());
        assert!(!DvLevel::Level6.is_required());
    }

    #[test]
    fn test_level_is_per_frame() {
        assert!(DvLevel::Level1.is_per_frame());
        assert!(DvLevel::Level2.is_per_frame());
        assert!(!DvLevel::Level6.is_per_frame());
    }

    #[test]
    fn test_all_returns_eight_levels() {
        assert_eq!(DvLevel::all().len(), 8);
    }

    #[test]
    fn test_analyzer_empty_missing_required_when_level1_seen() {
        let mut a = LevelAnalyzer::new();
        a.observe(DvLevel::Level1);
        let r = a.analyze();
        assert!(r.has_required_levels());
    }

    #[test]
    fn test_analyzer_missing_required_when_no_level1() {
        let a = LevelAnalyzer::new();
        let r = a.analyze();
        assert!(!r.has_required_levels());
        assert!(r.missing_required.contains(&DvLevel::Level1));
    }

    #[test]
    fn test_observe_frame_deduplicates() {
        let mut a = LevelAnalyzer::new();
        a.observe_frame(&[DvLevel::Level1, DvLevel::Level2]);
        a.observe_frame(&[DvLevel::Level1, DvLevel::Level2]);
        let r = a.analyze();
        // Should only be 2 unique levels
        assert_eq!(r.present.len(), 2);
    }

    #[test]
    fn test_report_contains() {
        let mut a = LevelAnalyzer::new();
        a.observe(DvLevel::Level6);
        let r = a.analyze();
        assert!(r.contains(DvLevel::Level6));
        assert!(!r.contains(DvLevel::Level8));
    }

    #[test]
    fn test_report_summary_non_empty() {
        let mut a = LevelAnalyzer::new();
        a.observe(DvLevel::Level1);
        let r = a.analyze();
        assert!(!r.summary().is_empty());
    }

    #[test]
    fn test_analyzer_reset() {
        let mut a = LevelAnalyzer::new();
        a.observe(DvLevel::Level1);
        a.reset();
        let r = a.analyze();
        assert_eq!(r.frame_count, 0);
        assert!(r.present.is_empty());
    }

    #[test]
    fn test_frame_count_increments_per_observe() {
        let mut a = LevelAnalyzer::new();
        a.observe(DvLevel::Level1);
        a.observe(DvLevel::Level2);
        // Each observe() increments frames
        assert_eq!(a.analyze().frame_count, 2);
    }
}
