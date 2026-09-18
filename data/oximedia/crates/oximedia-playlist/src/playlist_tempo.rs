#![allow(dead_code)]
//! Playlist tempo and pacing analysis.
//!
//! Analyzes the rhythm and pacing of a playlist by examining item durations,
//! genre transitions, and energy levels. Useful for broadcast programming
//! to ensure an engaging viewer experience with appropriate pacing.

use std::collections::HashMap;

/// Energy level of a content item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EnergyLevel {
    /// Very calm, slow-paced content.
    VeryLow,
    /// Below average energy.
    Low,
    /// Average energy level.
    Medium,
    /// Above average energy.
    High,
    /// Very intense, fast-paced content.
    VeryHigh,
}

impl EnergyLevel {
    /// Numeric value from 1 to 5.
    pub fn value(&self) -> u32 {
        match self {
            Self::VeryLow => 1,
            Self::Low => 2,
            Self::Medium => 3,
            Self::High => 4,
            Self::VeryHigh => 5,
        }
    }

    /// Create from a numeric value (clamped to 1-5).
    pub fn from_value(v: u32) -> Self {
        match v {
            0 | 1 => Self::VeryLow,
            2 => Self::Low,
            3 => Self::Medium,
            4 => Self::High,
            _ => Self::VeryHigh,
        }
    }
}

/// A content item with tempo-relevant metadata.
#[derive(Debug, Clone)]
pub struct TempoItem {
    /// Item identifier.
    pub id: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Energy level of the content.
    pub energy: EnergyLevel,
    /// Genre or category tag.
    pub genre: String,
}

impl TempoItem {
    /// Create a new tempo item.
    pub fn new(id: &str, duration_ms: u64, energy: EnergyLevel, genre: &str) -> Self {
        Self {
            id: id.to_string(),
            duration_ms,
            energy,
            genre: genre.to_string(),
        }
    }
}

/// Result of an energy transition between two consecutive items.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Energy stays the same.
    Steady,
    /// Energy increases by 1 level.
    GradualRise,
    /// Energy increases by 2+ levels.
    SharpRise,
    /// Energy decreases by 1 level.
    GradualDrop,
    /// Energy decreases by 2+ levels.
    SharpDrop,
}

impl TransitionType {
    /// Whether this is a sharp (abrupt) transition.
    pub fn is_sharp(&self) -> bool {
        matches!(self, Self::SharpRise | Self::SharpDrop)
    }
}

/// Compute the transition type between two energy levels.
#[allow(clippy::cast_possible_wrap)]
pub fn energy_transition(from: EnergyLevel, to: EnergyLevel) -> TransitionType {
    let diff = to.value() as i32 - from.value() as i32;
    match diff {
        0 => TransitionType::Steady,
        1 => TransitionType::GradualRise,
        -1 => TransitionType::GradualDrop,
        d if d >= 2 => TransitionType::SharpRise,
        _ => TransitionType::SharpDrop,
    }
}

/// Statistics about the tempo of a playlist.
#[derive(Debug, Clone)]
pub struct TempoStats {
    /// Average item duration in milliseconds.
    pub avg_duration_ms: f64,
    /// Standard deviation of durations in milliseconds.
    pub duration_std_ms: f64,
    /// Average energy as a float (1.0 to 5.0).
    pub avg_energy: f64,
    /// Number of sharp transitions.
    pub sharp_transitions: usize,
    /// Total transitions analyzed.
    pub total_transitions: usize,
    /// Genre diversity (unique genres / total items).
    pub genre_diversity: f64,
    /// Pacing score from 0.0 to 100.0 (higher = better paced).
    pub pacing_score: f64,
}

/// Analyzer for playlist tempo and pacing.
pub struct TempoAnalyzer {
    /// Target average energy level.
    target_energy: f64,
    /// Maximum acceptable fraction of sharp transitions.
    max_sharp_ratio: f64,
}

impl TempoAnalyzer {
    /// Create a new tempo analyzer with default settings.
    pub fn new() -> Self {
        Self {
            target_energy: 3.0,
            max_sharp_ratio: 0.3,
        }
    }

    /// Set the target average energy.
    pub fn with_target_energy(mut self, target: f64) -> Self {
        self.target_energy = target;
        self
    }

    /// Set the maximum acceptable sharp transition ratio.
    pub fn with_max_sharp_ratio(mut self, ratio: f64) -> Self {
        self.max_sharp_ratio = ratio;
        self
    }

    /// Analyze tempo statistics for a sequence of items.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self, items: &[TempoItem]) -> TempoStats {
        if items.is_empty() {
            return TempoStats {
                avg_duration_ms: 0.0,
                duration_std_ms: 0.0,
                avg_energy: 0.0,
                sharp_transitions: 0,
                total_transitions: 0,
                genre_diversity: 0.0,
                pacing_score: 0.0,
            };
        }

        let n = items.len() as f64;

        // Duration statistics
        let dur_sum: f64 = items.iter().map(|i| i.duration_ms as f64).sum();
        let avg_dur = dur_sum / n;
        let dur_var: f64 = items
            .iter()
            .map(|i| {
                let d = i.duration_ms as f64 - avg_dur;
                d * d
            })
            .sum::<f64>()
            / n;
        let dur_std = dur_var.sqrt();

        // Energy statistics
        let energy_sum: f64 = items.iter().map(|i| i.energy.value() as f64).sum();
        let avg_energy = energy_sum / n;

        // Transition statistics
        let mut sharp = 0usize;
        let total_trans = if items.len() > 1 { items.len() - 1 } else { 0 };
        for i in 1..items.len() {
            let t = energy_transition(items[i - 1].energy, items[i].energy);
            if t.is_sharp() {
                sharp += 1;
            }
        }

        // Genre diversity
        let mut genres: HashMap<&str, usize> = HashMap::new();
        for item in items {
            *genres.entry(&item.genre).or_insert(0) += 1;
        }
        let genre_diversity = genres.len() as f64 / n;

        // Pacing score
        let pacing = self.compute_pacing_score(avg_energy, sharp, total_trans, genre_diversity);

        TempoStats {
            avg_duration_ms: avg_dur,
            duration_std_ms: dur_std,
            avg_energy,
            sharp_transitions: sharp,
            total_transitions: total_trans,
            genre_diversity,
            pacing_score: pacing,
        }
    }

    /// Compute pacing score (0-100).
    #[allow(clippy::cast_precision_loss)]
    fn compute_pacing_score(
        &self,
        avg_energy: f64,
        sharp: usize,
        total_trans: usize,
        genre_diversity: f64,
    ) -> f64 {
        // Energy balance: how close to target
        let energy_diff = (avg_energy - self.target_energy).abs();
        let energy_score = (1.0 - energy_diff / 4.0).max(0.0) * 40.0;

        // Transition smoothness
        let sharp_ratio = if total_trans > 0 {
            sharp as f64 / total_trans as f64
        } else {
            0.0
        };
        let smooth_score = (1.0 - sharp_ratio / self.max_sharp_ratio.max(0.01))
            .max(0.0)
            .min(1.0)
            * 30.0;

        // Diversity
        let diversity_score = genre_diversity.min(1.0) * 30.0;

        (energy_score + smooth_score + diversity_score).min(100.0)
    }

    /// Suggest reordering to smooth sharp transitions.
    /// Returns indices of items that could be moved.
    pub fn suggest_smoothing(&self, items: &[TempoItem]) -> Vec<usize> {
        let mut problematic = Vec::new();
        for i in 1..items.len() {
            let t = energy_transition(items[i - 1].energy, items[i].energy);
            if t.is_sharp() {
                problematic.push(i);
            }
        }
        problematic
    }
}

impl Default for TempoAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_items() -> Vec<TempoItem> {
        vec![
            TempoItem::new("intro", 5_000, EnergyLevel::Low, "intro"),
            TempoItem::new("seg1", 30_000, EnergyLevel::Medium, "drama"),
            TempoItem::new("seg2", 25_000, EnergyLevel::High, "action"),
            TempoItem::new("seg3", 20_000, EnergyLevel::Medium, "drama"),
            TempoItem::new("outro", 5_000, EnergyLevel::Low, "outro"),
        ]
    }

    #[test]
    fn test_energy_level_value_roundtrip() {
        for v in 1..=5 {
            let level = EnergyLevel::from_value(v);
            assert_eq!(level.value(), v);
        }
    }

    #[test]
    fn test_energy_level_from_value_clamped() {
        assert_eq!(EnergyLevel::from_value(0), EnergyLevel::VeryLow);
        assert_eq!(EnergyLevel::from_value(99), EnergyLevel::VeryHigh);
    }

    #[test]
    fn test_energy_transition_steady() {
        assert_eq!(
            energy_transition(EnergyLevel::Medium, EnergyLevel::Medium),
            TransitionType::Steady
        );
    }

    #[test]
    fn test_energy_transition_gradual_rise() {
        assert_eq!(
            energy_transition(EnergyLevel::Low, EnergyLevel::Medium),
            TransitionType::GradualRise
        );
    }

    #[test]
    fn test_energy_transition_sharp_rise() {
        assert_eq!(
            energy_transition(EnergyLevel::VeryLow, EnergyLevel::High),
            TransitionType::SharpRise
        );
        assert!(TransitionType::SharpRise.is_sharp());
    }

    #[test]
    fn test_energy_transition_gradual_drop() {
        assert_eq!(
            energy_transition(EnergyLevel::High, EnergyLevel::Medium),
            TransitionType::GradualDrop
        );
    }

    #[test]
    fn test_energy_transition_sharp_drop() {
        assert_eq!(
            energy_transition(EnergyLevel::VeryHigh, EnergyLevel::Low),
            TransitionType::SharpDrop
        );
        assert!(TransitionType::SharpDrop.is_sharp());
    }

    #[test]
    fn test_analyze_empty() {
        let analyzer = TempoAnalyzer::new();
        let stats = analyzer.analyze(&[]);
        assert_eq!(stats.avg_duration_ms, 0.0);
        assert_eq!(stats.total_transitions, 0);
    }

    #[test]
    fn test_analyze_single_item() {
        let analyzer = TempoAnalyzer::new();
        let items = vec![TempoItem::new("a", 10_000, EnergyLevel::Medium, "drama")];
        let stats = analyzer.analyze(&items);
        assert!((stats.avg_duration_ms - 10_000.0).abs() < 1e-9);
        assert_eq!(stats.total_transitions, 0);
        assert_eq!(stats.sharp_transitions, 0);
    }

    #[test]
    fn test_analyze_sample_items() {
        let analyzer = TempoAnalyzer::new();
        let items = sample_items();
        let stats = analyzer.analyze(&items);
        assert!(stats.avg_duration_ms > 0.0);
        assert!(stats.avg_energy > 0.0);
        assert_eq!(stats.total_transitions, 4);
        assert!(stats.pacing_score > 0.0);
        assert!(stats.pacing_score <= 100.0);
    }

    #[test]
    fn test_genre_diversity() {
        let analyzer = TempoAnalyzer::new();
        let items = vec![
            TempoItem::new("a", 10_000, EnergyLevel::Medium, "drama"),
            TempoItem::new("b", 10_000, EnergyLevel::Medium, "drama"),
        ];
        let stats = analyzer.analyze(&items);
        assert!((stats.genre_diversity - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_suggest_smoothing_no_sharp() {
        let analyzer = TempoAnalyzer::new();
        let items = vec![
            TempoItem::new("a", 10_000, EnergyLevel::Low, "a"),
            TempoItem::new("b", 10_000, EnergyLevel::Medium, "b"),
            TempoItem::new("c", 10_000, EnergyLevel::High, "c"),
        ];
        let suggestions = analyzer.suggest_smoothing(&items);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggest_smoothing_with_sharp() {
        let analyzer = TempoAnalyzer::new();
        let items = vec![
            TempoItem::new("a", 10_000, EnergyLevel::VeryLow, "a"),
            TempoItem::new("b", 10_000, EnergyLevel::VeryHigh, "b"),
        ];
        let suggestions = analyzer.suggest_smoothing(&items);
        assert_eq!(suggestions, vec![1]);
    }

    #[test]
    fn test_duration_std_uniform() {
        let analyzer = TempoAnalyzer::new();
        let items = vec![
            TempoItem::new("a", 10_000, EnergyLevel::Medium, "x"),
            TempoItem::new("b", 10_000, EnergyLevel::Medium, "x"),
            TempoItem::new("c", 10_000, EnergyLevel::Medium, "x"),
        ];
        let stats = analyzer.analyze(&items);
        assert!((stats.duration_std_ms).abs() < 1e-9);
    }

    #[test]
    fn test_analyzer_custom_config() {
        let analyzer = TempoAnalyzer::new()
            .with_target_energy(4.0)
            .with_max_sharp_ratio(0.5);
        let items = sample_items();
        let stats = analyzer.analyze(&items);
        // Just ensure it runs without panic and produces valid output
        assert!(stats.pacing_score >= 0.0);
        assert!(stats.pacing_score <= 100.0);
    }
}
