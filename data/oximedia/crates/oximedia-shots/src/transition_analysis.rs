//! Transition style analysis for `oximedia-shots`.
//!
//! Provides [`TransitionStyle`] classification, [`TransitionProfile`] data,
//! and a [`TransitionAnalyzer`] that collects statistics on how consecutive
//! shots are joined (hard cut, dissolve, wipe, fade, etc.).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Transition style
// ---------------------------------------------------------------------------

/// High-level classification of a transition between two shots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionStyle {
    /// Instantaneous hard cut.
    HardCut,
    /// Cross-dissolve (additive overlap).
    Dissolve,
    /// Fade-to-black followed by fade-from-black.
    FadeThrough,
    /// Directional wipe (left, right, diagonal, etc.).
    Wipe,
    /// L-cut or J-cut where audio leads/trails the video edit.
    SplitEdit,
    /// Match cut based on compositional similarity.
    MatchCut,
    /// Jump cut (same angle, minor time skip).
    JumpCut,
    /// No transition detected / unknown.
    Unknown,
}

impl TransitionStyle {
    /// Returns a short human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HardCut => "Hard Cut",
            Self::Dissolve => "Dissolve",
            Self::FadeThrough => "Fade Through",
            Self::Wipe => "Wipe",
            Self::SplitEdit => "Split Edit (L/J)",
            Self::MatchCut => "Match Cut",
            Self::JumpCut => "Jump Cut",
            Self::Unknown => "Unknown",
        }
    }

    /// Returns all defined variants.
    #[must_use]
    pub const fn all() -> &'static [TransitionStyle] {
        &[
            Self::HardCut,
            Self::Dissolve,
            Self::FadeThrough,
            Self::Wipe,
            Self::SplitEdit,
            Self::MatchCut,
            Self::JumpCut,
            Self::Unknown,
        ]
    }

    /// Whether this transition is considered an intentional creative choice
    /// (as opposed to a technical artifact like a jump cut).
    #[must_use]
    pub const fn is_creative(self) -> bool {
        matches!(
            self,
            Self::Dissolve | Self::FadeThrough | Self::Wipe | Self::SplitEdit | Self::MatchCut
        )
    }
}

impl std::fmt::Display for TransitionStyle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Transition profile
// ---------------------------------------------------------------------------

/// Per-transition data gathered at an edit point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionProfile {
    /// Zero-based index of the edit point (between shot `index` and `index+1`).
    pub edit_index: usize,
    /// Classified style of this transition.
    pub style: TransitionStyle,
    /// Duration of the transition in frames (0 for hard cuts).
    pub duration_frames: u32,
    /// Confidence of the classification in `[0.0, 1.0]`.
    pub confidence: f64,
}

impl TransitionProfile {
    /// Creates a new transition profile.
    #[must_use]
    pub fn new(
        edit_index: usize,
        style: TransitionStyle,
        duration_frames: u32,
        confidence: f64,
    ) -> Self {
        Self {
            edit_index,
            style,
            duration_frames,
            confidence,
        }
    }

    /// Returns `true` if the transition is instantaneous (zero duration).
    #[must_use]
    pub fn is_instant(&self) -> bool {
        self.duration_frames == 0
    }
}

// ---------------------------------------------------------------------------
// Transition analyzer
// ---------------------------------------------------------------------------

/// Summary statistics produced by [`TransitionAnalyzer`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionSummary {
    /// Total number of transitions analyzed.
    pub total: usize,
    /// Count per style.
    pub style_counts: HashMap<String, usize>,
    /// Average transition duration in frames (excluding hard cuts).
    pub avg_transition_frames: f64,
    /// Percentage of transitions classified as creative choices.
    pub creative_pct: f64,
}

/// Collects [`TransitionProfile`]s and computes aggregate statistics.
#[derive(Debug, Clone, Default)]
pub struct TransitionAnalyzer {
    profiles: Vec<TransitionProfile>,
}

impl TransitionAnalyzer {
    /// Creates a new empty analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: Vec::new(),
        }
    }

    /// Records a transition profile.
    pub fn add(&mut self, profile: TransitionProfile) {
        self.profiles.push(profile);
    }

    /// Returns the number of recorded transitions.
    #[must_use]
    pub fn count(&self) -> usize {
        self.profiles.len()
    }

    /// Returns a reference to all recorded profiles.
    #[must_use]
    pub fn profiles(&self) -> &[TransitionProfile] {
        &self.profiles
    }

    /// Counts occurrences of each [`TransitionStyle`].
    #[must_use]
    pub fn style_distribution(&self) -> HashMap<TransitionStyle, usize> {
        let mut map = HashMap::new();
        for p in &self.profiles {
            *map.entry(p.style).or_insert(0) += 1;
        }
        map
    }

    /// Computes average transition duration (frames) excluding hard cuts.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_transition_duration(&self) -> f64 {
        let non_cut: Vec<_> = self
            .profiles
            .iter()
            .filter(|p| p.duration_frames > 0)
            .collect();
        if non_cut.is_empty() {
            return 0.0;
        }
        let sum: u64 = non_cut.iter().map(|p| u64::from(p.duration_frames)).sum();
        sum as f64 / non_cut.len() as f64
    }

    /// Computes a full summary.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn summarize(&self) -> TransitionSummary {
        let total = self.profiles.len();
        let style_counts: HashMap<String, usize> = self
            .style_distribution()
            .into_iter()
            .map(|(k, v)| (k.label().to_string(), v))
            .collect();

        let creative_count = self
            .profiles
            .iter()
            .filter(|p| p.style.is_creative())
            .count();
        let creative_pct = if total == 0 {
            0.0
        } else {
            creative_count as f64 / total as f64 * 100.0
        };

        TransitionSummary {
            total,
            style_counts,
            avg_transition_frames: self.avg_transition_duration(),
            creative_pct,
        }
    }

    /// Clears all recorded profiles.
    pub fn reset(&mut self) {
        self.profiles.clear();
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- TransitionStyle ----------------------------------------------------

    #[test]
    fn test_style_label() {
        assert_eq!(TransitionStyle::HardCut.label(), "Hard Cut");
        assert_eq!(TransitionStyle::Dissolve.label(), "Dissolve");
        assert_eq!(TransitionStyle::JumpCut.label(), "Jump Cut");
    }

    #[test]
    fn test_style_display() {
        assert_eq!(format!("{}", TransitionStyle::Wipe), "Wipe");
    }

    #[test]
    fn test_style_all_variants() {
        let all = TransitionStyle::all();
        assert_eq!(all.len(), 8);
    }

    #[test]
    fn test_style_is_creative() {
        assert!(TransitionStyle::Dissolve.is_creative());
        assert!(TransitionStyle::MatchCut.is_creative());
        assert!(!TransitionStyle::HardCut.is_creative());
        assert!(!TransitionStyle::JumpCut.is_creative());
        assert!(!TransitionStyle::Unknown.is_creative());
    }

    // -- TransitionProfile --------------------------------------------------

    #[test]
    fn test_profile_creation() {
        let p = TransitionProfile::new(0, TransitionStyle::HardCut, 0, 0.99);
        assert_eq!(p.edit_index, 0);
        assert!(p.is_instant());
    }

    #[test]
    fn test_profile_dissolve_not_instant() {
        let p = TransitionProfile::new(1, TransitionStyle::Dissolve, 15, 0.85);
        assert!(!p.is_instant());
        assert_eq!(p.duration_frames, 15);
    }

    // -- TransitionAnalyzer -------------------------------------------------

    #[test]
    fn test_analyzer_empty() {
        let a = TransitionAnalyzer::new();
        assert_eq!(a.count(), 0);
        assert_eq!(a.avg_transition_duration(), 0.0);
    }

    #[test]
    fn test_analyzer_add_and_count() {
        let mut a = TransitionAnalyzer::new();
        a.add(TransitionProfile::new(0, TransitionStyle::HardCut, 0, 0.9));
        a.add(TransitionProfile::new(
            1,
            TransitionStyle::Dissolve,
            12,
            0.8,
        ));
        assert_eq!(a.count(), 2);
    }

    #[test]
    fn test_style_distribution() {
        let mut a = TransitionAnalyzer::new();
        a.add(TransitionProfile::new(0, TransitionStyle::HardCut, 0, 0.9));
        a.add(TransitionProfile::new(1, TransitionStyle::HardCut, 0, 0.8));
        a.add(TransitionProfile::new(
            2,
            TransitionStyle::Dissolve,
            10,
            0.7,
        ));
        let dist = a.style_distribution();
        assert_eq!(dist[&TransitionStyle::HardCut], 2);
        assert_eq!(dist[&TransitionStyle::Dissolve], 1);
    }

    #[test]
    fn test_avg_transition_duration_only_cuts() {
        let mut a = TransitionAnalyzer::new();
        a.add(TransitionProfile::new(0, TransitionStyle::HardCut, 0, 0.9));
        a.add(TransitionProfile::new(1, TransitionStyle::HardCut, 0, 0.8));
        assert_eq!(a.avg_transition_duration(), 0.0);
    }

    #[test]
    fn test_avg_transition_duration_mixed() {
        let mut a = TransitionAnalyzer::new();
        a.add(TransitionProfile::new(0, TransitionStyle::HardCut, 0, 0.9));
        a.add(TransitionProfile::new(
            1,
            TransitionStyle::Dissolve,
            10,
            0.8,
        ));
        a.add(TransitionProfile::new(
            2,
            TransitionStyle::Dissolve,
            20,
            0.7,
        ));
        let avg = a.avg_transition_duration();
        assert!((avg - 15.0).abs() < 1e-6, "avg was {avg}");
    }

    #[test]
    fn test_summarize_creative_pct() {
        let mut a = TransitionAnalyzer::new();
        a.add(TransitionProfile::new(0, TransitionStyle::HardCut, 0, 0.9));
        a.add(TransitionProfile::new(
            1,
            TransitionStyle::Dissolve,
            10,
            0.8,
        ));
        a.add(TransitionProfile::new(2, TransitionStyle::Wipe, 8, 0.7));
        a.add(TransitionProfile::new(3, TransitionStyle::HardCut, 0, 0.9));
        let s = a.summarize();
        assert_eq!(s.total, 4);
        // 2 creative out of 4 = 50%
        assert!((s.creative_pct - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_summarize_empty() {
        let a = TransitionAnalyzer::new();
        let s = a.summarize();
        assert_eq!(s.total, 0);
        assert_eq!(s.creative_pct, 0.0);
    }

    #[test]
    fn test_reset() {
        let mut a = TransitionAnalyzer::new();
        a.add(TransitionProfile::new(0, TransitionStyle::HardCut, 0, 0.9));
        assert_eq!(a.count(), 1);
        a.reset();
        assert_eq!(a.count(), 0);
    }

    #[test]
    fn test_profiles_accessor() {
        let mut a = TransitionAnalyzer::new();
        a.add(TransitionProfile::new(0, TransitionStyle::HardCut, 0, 0.9));
        assert_eq!(a.profiles().len(), 1);
        assert_eq!(a.profiles()[0].style, TransitionStyle::HardCut);
    }
}
