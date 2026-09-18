#![allow(dead_code)]
//! Cognitive load assessment for media content complexity.
//!
//! Evaluates media content across multiple dimensions (visual, auditory,
//! textual, temporal) to produce a cognitive load score. This helps ensure
//! content is accessible to users with cognitive disabilities, ADHD,
//! learning difficulties, or those who benefit from simplified presentation.
//!
//! Based on Cognitive Load Theory (Sweller, 1988) and media accessibility
//! guidelines from WCAG 2.1 (1.3.5 Identify Input Purpose, 2.2 Enough Time,
//! 3.1 Readable).

use std::fmt;

/// Overall cognitive load level of media content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CognitiveLoadLevel {
    /// Very low cognitive demand.
    VeryLow,
    /// Low cognitive demand.
    Low,
    /// Moderate cognitive demand.
    Moderate,
    /// High cognitive demand.
    High,
    /// Very high cognitive demand.
    VeryHigh,
}

impl fmt::Display for CognitiveLoadLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VeryLow => write!(f, "Very Low"),
            Self::Low => write!(f, "Low"),
            Self::Moderate => write!(f, "Moderate"),
            Self::High => write!(f, "High"),
            Self::VeryHigh => write!(f, "Very High"),
        }
    }
}

/// Dimension of cognitive load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadDimension {
    /// Visual complexity (scene changes, overlays, motion).
    Visual,
    /// Auditory complexity (speech rate, overlapping audio, music).
    Auditory,
    /// Textual complexity (reading level, caption density).
    Textual,
    /// Temporal complexity (pacing, scene duration, information rate).
    Temporal,
    /// Interactivity complexity (required user actions, decision points).
    Interactivity,
}

impl fmt::Display for LoadDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Visual => write!(f, "Visual"),
            Self::Auditory => write!(f, "Auditory"),
            Self::Textual => write!(f, "Textual"),
            Self::Temporal => write!(f, "Temporal"),
            Self::Interactivity => write!(f, "Interactivity"),
        }
    }
}

/// Metrics for visual complexity assessment.
#[derive(Debug, Clone, Default)]
pub struct VisualMetrics {
    /// Average scene duration in seconds.
    pub avg_scene_duration_sec: f64,
    /// Number of scene changes per minute.
    pub scene_changes_per_minute: f64,
    /// Number of simultaneous visual elements (overlays, PIP, text).
    pub simultaneous_elements: u32,
    /// Whether high-motion content is present.
    pub has_high_motion: bool,
    /// Whether flashing or strobing is present.
    pub has_flashing: bool,
    /// Color complexity (number of distinct color regions).
    pub color_complexity: u32,
}

/// Metrics for auditory complexity assessment.
#[derive(Debug, Clone, Default)]
pub struct AuditoryMetrics {
    /// Words spoken per minute.
    pub speech_rate_wpm: f64,
    /// Number of simultaneous audio tracks.
    pub simultaneous_tracks: u32,
    /// Whether background music is present during speech.
    pub music_over_speech: bool,
    /// Whether multiple speakers overlap.
    pub overlapping_speakers: bool,
    /// Signal-to-noise ratio in dB (higher = clearer).
    pub snr_db: f64,
    /// Number of distinct speakers.
    pub speaker_count: u32,
}

/// Metrics for textual complexity assessment.
#[derive(Debug, Clone, Default)]
pub struct TextualMetrics {
    /// Average reading level (grade level).
    pub reading_grade_level: f64,
    /// Words per caption/subtitle.
    pub avg_words_per_caption: f64,
    /// Caption display rate (words per minute).
    pub caption_wpm: f64,
    /// Percentage of technical/jargon words.
    pub jargon_percentage: f64,
    /// Whether captions include non-speech information.
    pub includes_non_speech_info: bool,
    /// Number of acronyms without expansion.
    pub unexpanded_acronyms: u32,
}

/// Metrics for temporal complexity assessment.
#[derive(Debug, Clone, Default)]
pub struct TemporalMetrics {
    /// Total content duration in seconds.
    pub duration_sec: f64,
    /// Information density (bits of new info per minute, estimated).
    pub info_density_per_minute: f64,
    /// Whether pauses exist between information chunks.
    pub has_breathing_room: bool,
    /// Average segment duration before topic change (seconds).
    pub avg_topic_duration_sec: f64,
    /// Whether the content requires real-time response.
    pub requires_real_time_response: bool,
    /// Number of topic/context switches.
    pub context_switches: u32,
}

/// Metrics for interactivity complexity assessment.
#[derive(Debug, Clone, Default)]
pub struct InteractivityMetrics {
    /// Number of required user decisions.
    pub decision_points: u32,
    /// Whether timed responses are required.
    pub timed_responses: bool,
    /// Complexity of navigation (levels of hierarchy).
    pub navigation_depth: u32,
    /// Whether undo/retry is available.
    pub supports_undo: bool,
    /// Number of simultaneous input channels required.
    pub input_channels: u32,
}

/// A dimensional score with explanation.
#[derive(Debug, Clone)]
pub struct DimensionalScore {
    /// The dimension being scored.
    pub dimension: LoadDimension,
    /// Score from 0.0 (no load) to 1.0 (maximum load).
    pub score: f64,
    /// Load level category.
    pub level: CognitiveLoadLevel,
    /// Human-readable factors contributing to the score.
    pub factors: Vec<String>,
    /// Recommendations for reducing load in this dimension.
    pub recommendations: Vec<String>,
}

/// Complete cognitive load assessment result.
#[derive(Debug, Clone)]
pub struct CognitiveLoadReport {
    /// Overall weighted score (0.0 to 1.0).
    pub overall_score: f64,
    /// Overall load level.
    pub overall_level: CognitiveLoadLevel,
    /// Per-dimension scores.
    pub dimensions: Vec<DimensionalScore>,
    /// Whether the content meets accessibility recommendations.
    pub meets_accessibility_guidelines: bool,
    /// Summary of top recommendations.
    pub top_recommendations: Vec<String>,
}

/// Weights for each cognitive load dimension in the overall score.
#[derive(Debug, Clone)]
pub struct DimensionWeights {
    /// Weight for visual dimension.
    pub visual: f64,
    /// Weight for auditory dimension.
    pub auditory: f64,
    /// Weight for textual dimension.
    pub textual: f64,
    /// Weight for temporal dimension.
    pub temporal: f64,
    /// Weight for interactivity dimension.
    pub interactivity: f64,
}

impl Default for DimensionWeights {
    fn default() -> Self {
        Self {
            visual: 0.25,
            auditory: 0.25,
            textual: 0.20,
            temporal: 0.20,
            interactivity: 0.10,
        }
    }
}

impl DimensionWeights {
    /// Normalize weights so they sum to 1.0.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let total = self.visual + self.auditory + self.textual + self.temporal + self.interactivity;
        if total <= 0.0 {
            return Self::default();
        }
        Self {
            visual: self.visual / total,
            auditory: self.auditory / total,
            textual: self.textual / total,
            temporal: self.temporal / total,
            interactivity: self.interactivity / total,
        }
    }
}

/// Analyzes cognitive load of media content across multiple dimensions.
#[derive(Debug)]
pub struct CognitiveLoadAnalyzer {
    /// Dimension weights for overall score.
    weights: DimensionWeights,
    /// Maximum acceptable overall score for accessibility compliance.
    max_acceptable_score: f64,
}

impl Default for CognitiveLoadAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl CognitiveLoadAnalyzer {
    /// Create a new analyzer with default weights.
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: DimensionWeights::default(),
            max_acceptable_score: 0.6,
        }
    }

    /// Create with custom dimension weights.
    #[must_use]
    pub fn with_weights(weights: DimensionWeights) -> Self {
        Self {
            weights: weights.normalized(),
            max_acceptable_score: 0.6,
        }
    }

    /// Set the maximum acceptable score for accessibility compliance.
    #[must_use]
    pub fn with_max_acceptable(mut self, max_score: f64) -> Self {
        self.max_acceptable_score = max_score.clamp(0.0, 1.0);
        self
    }

    /// Assess visual cognitive load.
    #[must_use]
    pub fn assess_visual(&self, metrics: &VisualMetrics) -> DimensionalScore {
        let mut score = 0.0_f64;
        let mut factors = Vec::new();
        let mut recommendations = Vec::new();

        // Scene change rate: > 30/min is very high
        let scene_rate_score = (metrics.scene_changes_per_minute / 30.0).min(1.0);
        score += scene_rate_score * 0.25;
        if scene_rate_score > 0.5 {
            factors.push(format!(
                "High scene change rate: {:.1}/min",
                metrics.scene_changes_per_minute
            ));
            recommendations
                .push("Reduce scene change frequency or add transition cues".to_string());
        }

        // Short scenes increase load
        if metrics.avg_scene_duration_sec > 0.0 {
            let short_scene_score = (3.0 / metrics.avg_scene_duration_sec).min(1.0);
            score += short_scene_score * 0.2;
            if short_scene_score > 0.5 {
                factors.push(format!(
                    "Short average scene duration: {:.1}s",
                    metrics.avg_scene_duration_sec
                ));
                recommendations.push("Increase average scene duration to > 3 seconds".to_string());
            }
        }

        // Simultaneous elements
        let element_score = (f64::from(metrics.simultaneous_elements) / 5.0).min(1.0);
        score += element_score * 0.2;
        if element_score > 0.5 {
            factors.push(format!(
                "Many simultaneous elements: {}",
                metrics.simultaneous_elements
            ));
            recommendations.push("Reduce overlapping visual elements".to_string());
        }

        // High motion
        if metrics.has_high_motion {
            score += 0.15;
            factors.push("High motion content detected".to_string());
            recommendations.push("Provide option to reduce motion".to_string());
        }

        // Flashing
        if metrics.has_flashing {
            score += 0.2;
            factors.push("Flashing/strobing content detected".to_string());
            recommendations.push("Remove flashing content or provide seizure warning".to_string());
        }

        score = score.clamp(0.0, 1.0);
        let level = score_to_level(score);

        DimensionalScore {
            dimension: LoadDimension::Visual,
            score,
            level,
            factors,
            recommendations,
        }
    }

    /// Assess auditory cognitive load.
    #[must_use]
    pub fn assess_auditory(&self, metrics: &AuditoryMetrics) -> DimensionalScore {
        let mut score = 0.0_f64;
        let mut factors = Vec::new();
        let mut recommendations = Vec::new();

        // Speech rate: > 180 WPM is fast
        let speech_score = (metrics.speech_rate_wpm / 200.0).min(1.0);
        score += speech_score * 0.25;
        if speech_score > 0.5 {
            factors.push(format!(
                "Fast speech rate: {:.0} WPM",
                metrics.speech_rate_wpm
            ));
            recommendations.push("Provide speed control or slower narration option".to_string());
        }

        // Simultaneous audio tracks
        if metrics.simultaneous_tracks > 1 {
            let track_score =
                (f64::from(metrics.simultaneous_tracks.saturating_sub(1)) / 3.0).min(1.0);
            score += track_score * 0.2;
            factors.push(format!(
                "Multiple simultaneous audio tracks: {}",
                metrics.simultaneous_tracks
            ));
            recommendations.push("Allow individual track volume control".to_string());
        }

        // Music over speech
        if metrics.music_over_speech {
            score += 0.15;
            factors.push("Music playing over speech".to_string());
            recommendations
                .push("Reduce background music during speech or provide toggle".to_string());
        }

        // Overlapping speakers
        if metrics.overlapping_speakers {
            score += 0.2;
            factors.push("Overlapping speakers detected".to_string());
            recommendations.push("Separate speaker audio or add speaker labels".to_string());
        }

        // Low SNR
        if metrics.snr_db < 15.0 && metrics.snr_db > 0.0 {
            let snr_score = 1.0 - (metrics.snr_db / 15.0);
            score += snr_score * 0.1;
            factors.push(format!(
                "Low signal-to-noise ratio: {:.1} dB",
                metrics.snr_db
            ));
            recommendations
                .push("Improve audio clarity or provide enhanced audio track".to_string());
        }

        // Many speakers
        if metrics.speaker_count > 3 {
            let speaker_score = (f64::from(metrics.speaker_count.saturating_sub(3)) / 5.0).min(1.0);
            score += speaker_score * 0.1;
            factors.push(format!("Many speakers: {}", metrics.speaker_count));
            recommendations.push("Add speaker identification labels".to_string());
        }

        score = score.clamp(0.0, 1.0);
        let level = score_to_level(score);

        DimensionalScore {
            dimension: LoadDimension::Auditory,
            score,
            level,
            factors,
            recommendations,
        }
    }

    /// Assess textual cognitive load.
    #[must_use]
    pub fn assess_textual(&self, metrics: &TextualMetrics) -> DimensionalScore {
        let mut score = 0.0_f64;
        let mut factors = Vec::new();
        let mut recommendations = Vec::new();

        // Reading grade level: > 12 is complex
        let grade_score = (metrics.reading_grade_level / 16.0).min(1.0);
        score += grade_score * 0.3;
        if grade_score > 0.5 {
            factors.push(format!(
                "High reading level: grade {:.1}",
                metrics.reading_grade_level
            ));
            recommendations.push("Simplify language to grade 8 or below".to_string());
        }

        // Caption display rate: > 200 WPM is fast
        let caption_rate_score = (metrics.caption_wpm / 250.0).min(1.0);
        score += caption_rate_score * 0.25;
        if caption_rate_score > 0.5 {
            factors.push(format!("Fast caption rate: {:.0} WPM", metrics.caption_wpm));
            recommendations.push("Reduce caption display rate or allow pause".to_string());
        }

        // Jargon percentage
        let jargon_score = (metrics.jargon_percentage / 20.0).min(1.0);
        score += jargon_score * 0.2;
        if jargon_score > 0.5 {
            factors.push(format!(
                "High jargon content: {:.1}%",
                metrics.jargon_percentage
            ));
            recommendations.push("Provide glossary or simpler alternatives".to_string());
        }

        // Unexpanded acronyms
        if metrics.unexpanded_acronyms > 0 {
            let acronym_score = (f64::from(metrics.unexpanded_acronyms) / 10.0).min(1.0);
            score += acronym_score * 0.15;
            factors.push(format!(
                "Unexpanded acronyms: {}",
                metrics.unexpanded_acronyms
            ));
            recommendations.push("Expand acronyms on first use".to_string());
        }

        // Words per caption
        if metrics.avg_words_per_caption > 12.0 {
            let caption_len_score = ((metrics.avg_words_per_caption - 12.0) / 20.0).min(1.0);
            score += caption_len_score * 0.1;
            factors.push(format!(
                "Long captions: {:.1} words average",
                metrics.avg_words_per_caption
            ));
            recommendations.push("Break captions into shorter segments".to_string());
        }

        score = score.clamp(0.0, 1.0);
        let level = score_to_level(score);

        DimensionalScore {
            dimension: LoadDimension::Textual,
            score,
            level,
            factors,
            recommendations,
        }
    }

    /// Assess temporal cognitive load.
    #[must_use]
    pub fn assess_temporal(&self, metrics: &TemporalMetrics) -> DimensionalScore {
        let mut score = 0.0_f64;
        let mut factors = Vec::new();
        let mut recommendations = Vec::new();

        // Information density
        let density_score = (metrics.info_density_per_minute / 50.0).min(1.0);
        score += density_score * 0.3;
        if density_score > 0.5 {
            factors.push(format!(
                "High information density: {:.1} units/min",
                metrics.info_density_per_minute
            ));
            recommendations.push("Add summaries or reduce information rate".to_string());
        }

        // No breathing room
        if !metrics.has_breathing_room {
            score += 0.2;
            factors.push("No pauses between information segments".to_string());
            recommendations.push("Add pauses between content sections".to_string());
        }

        // Short topic duration
        if metrics.avg_topic_duration_sec > 0.0 && metrics.avg_topic_duration_sec < 30.0 {
            let topic_score = (30.0 / metrics.avg_topic_duration_sec).min(1.0) * 0.5;
            score += topic_score * 0.2;
            factors.push(format!(
                "Rapid topic changes: {:.1}s average",
                metrics.avg_topic_duration_sec
            ));
            recommendations.push("Extend topic segments or add transition cues".to_string());
        }

        // Real-time response requirement
        if metrics.requires_real_time_response {
            score += 0.2;
            factors.push("Requires real-time user response".to_string());
            recommendations
                .push("Provide extended time options or remove time pressure".to_string());
        }

        // Context switches
        if metrics.context_switches > 5 {
            let switch_score =
                (f64::from(metrics.context_switches.saturating_sub(5)) / 15.0).min(1.0);
            score += switch_score * 0.1;
            factors.push(format!(
                "Many context switches: {}",
                metrics.context_switches
            ));
            recommendations.push("Reduce context switching or provide navigation aids".to_string());
        }

        score = score.clamp(0.0, 1.0);
        let level = score_to_level(score);

        DimensionalScore {
            dimension: LoadDimension::Temporal,
            score,
            level,
            factors,
            recommendations,
        }
    }

    /// Assess interactivity cognitive load.
    #[must_use]
    pub fn assess_interactivity(&self, metrics: &InteractivityMetrics) -> DimensionalScore {
        let mut score = 0.0_f64;
        let mut factors = Vec::new();
        let mut recommendations = Vec::new();

        // Decision points
        let decision_score = (f64::from(metrics.decision_points) / 20.0).min(1.0);
        score += decision_score * 0.3;
        if decision_score > 0.5 {
            factors.push(format!("Many decision points: {}", metrics.decision_points));
            recommendations.push("Reduce required decisions or provide defaults".to_string());
        }

        // Timed responses
        if metrics.timed_responses {
            score += 0.25;
            factors.push("Timed responses required".to_string());
            recommendations.push("Remove or extend time limits".to_string());
        }

        // Navigation depth
        if metrics.navigation_depth > 3 {
            let depth_score =
                (f64::from(metrics.navigation_depth.saturating_sub(3)) / 5.0).min(1.0);
            score += depth_score * 0.15;
            factors.push(format!(
                "Deep navigation: {} levels",
                metrics.navigation_depth
            ));
            recommendations.push("Flatten navigation hierarchy".to_string());
        }

        // No undo
        if !metrics.supports_undo && metrics.decision_points > 0 {
            score += 0.15;
            factors.push("No undo/retry available".to_string());
            recommendations.push("Add undo or retry functionality".to_string());
        }

        // Multiple input channels
        if metrics.input_channels > 1 {
            let channel_score =
                (f64::from(metrics.input_channels.saturating_sub(1)) / 3.0).min(1.0);
            score += channel_score * 0.15;
            factors.push(format!(
                "Multiple input channels: {}",
                metrics.input_channels
            ));
            recommendations.push("Allow single-channel input alternatives".to_string());
        }

        score = score.clamp(0.0, 1.0);
        let level = score_to_level(score);

        DimensionalScore {
            dimension: LoadDimension::Interactivity,
            score,
            level,
            factors,
            recommendations,
        }
    }

    /// Generate a complete cognitive load report from all dimensions.
    #[must_use]
    pub fn generate_report(
        &self,
        visual: &VisualMetrics,
        auditory: &AuditoryMetrics,
        textual: &TextualMetrics,
        temporal: &TemporalMetrics,
        interactivity: &InteractivityMetrics,
    ) -> CognitiveLoadReport {
        let visual_score = self.assess_visual(visual);
        let auditory_score = self.assess_auditory(auditory);
        let textual_score = self.assess_textual(textual);
        let temporal_score = self.assess_temporal(temporal);
        let interactivity_score = self.assess_interactivity(interactivity);

        let weights = self.weights.normalized();
        let overall_score = visual_score.score * weights.visual
            + auditory_score.score * weights.auditory
            + textual_score.score * weights.textual
            + temporal_score.score * weights.temporal
            + interactivity_score.score * weights.interactivity;

        let overall_score = overall_score.clamp(0.0, 1.0);
        let overall_level = score_to_level(overall_score);

        // Collect top recommendations from highest-scoring dimensions
        let mut all_recs: Vec<(f64, &str)> = Vec::new();
        for dim in &[
            &visual_score,
            &auditory_score,
            &textual_score,
            &temporal_score,
            &interactivity_score,
        ] {
            for rec in &dim.recommendations {
                all_recs.push((dim.score, rec.as_str()));
            }
        }
        all_recs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let top_recommendations: Vec<String> = all_recs
            .iter()
            .take(5)
            .map(|(_, rec)| (*rec).to_string())
            .collect();

        let meets_guidelines = overall_score <= self.max_acceptable_score;

        CognitiveLoadReport {
            overall_score,
            overall_level,
            dimensions: vec![
                visual_score,
                auditory_score,
                textual_score,
                temporal_score,
                interactivity_score,
            ],
            meets_accessibility_guidelines: meets_guidelines,
            top_recommendations,
        }
    }

    /// Quick assessment from just visual and auditory metrics.
    #[must_use]
    pub fn quick_assess(
        &self,
        visual: &VisualMetrics,
        auditory: &AuditoryMetrics,
    ) -> CognitiveLoadReport {
        self.generate_report(
            visual,
            auditory,
            &TextualMetrics::default(),
            &TemporalMetrics::default(),
            &InteractivityMetrics::default(),
        )
    }
}

/// Map a score (0.0-1.0) to a cognitive load level.
fn score_to_level(score: f64) -> CognitiveLoadLevel {
    if score < 0.2 {
        CognitiveLoadLevel::VeryLow
    } else if score < 0.4 {
        CognitiveLoadLevel::Low
    } else if score < 0.6 {
        CognitiveLoadLevel::Moderate
    } else if score < 0.8 {
        CognitiveLoadLevel::High
    } else {
        CognitiveLoadLevel::VeryHigh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_load_level_display() {
        assert_eq!(CognitiveLoadLevel::VeryLow.to_string(), "Very Low");
        assert_eq!(CognitiveLoadLevel::Low.to_string(), "Low");
        assert_eq!(CognitiveLoadLevel::Moderate.to_string(), "Moderate");
        assert_eq!(CognitiveLoadLevel::High.to_string(), "High");
        assert_eq!(CognitiveLoadLevel::VeryHigh.to_string(), "Very High");
    }

    #[test]
    fn test_cognitive_load_level_ordering() {
        assert!(CognitiveLoadLevel::VeryLow < CognitiveLoadLevel::Low);
        assert!(CognitiveLoadLevel::Low < CognitiveLoadLevel::Moderate);
        assert!(CognitiveLoadLevel::Moderate < CognitiveLoadLevel::High);
        assert!(CognitiveLoadLevel::High < CognitiveLoadLevel::VeryHigh);
    }

    #[test]
    fn test_load_dimension_display() {
        assert_eq!(LoadDimension::Visual.to_string(), "Visual");
        assert_eq!(LoadDimension::Auditory.to_string(), "Auditory");
        assert_eq!(LoadDimension::Textual.to_string(), "Textual");
        assert_eq!(LoadDimension::Temporal.to_string(), "Temporal");
        assert_eq!(LoadDimension::Interactivity.to_string(), "Interactivity");
    }

    #[test]
    fn test_score_to_level() {
        assert_eq!(score_to_level(0.0), CognitiveLoadLevel::VeryLow);
        assert_eq!(score_to_level(0.1), CognitiveLoadLevel::VeryLow);
        assert_eq!(score_to_level(0.3), CognitiveLoadLevel::Low);
        assert_eq!(score_to_level(0.5), CognitiveLoadLevel::Moderate);
        assert_eq!(score_to_level(0.7), CognitiveLoadLevel::High);
        assert_eq!(score_to_level(0.9), CognitiveLoadLevel::VeryHigh);
    }

    #[test]
    fn test_default_weights_sum_to_one() {
        let w = DimensionWeights::default();
        let sum = w.visual + w.auditory + w.textual + w.temporal + w.interactivity;
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weight_normalization() {
        let w = DimensionWeights {
            visual: 2.0,
            auditory: 2.0,
            textual: 2.0,
            temporal: 2.0,
            interactivity: 2.0,
        };
        let norm = w.normalized();
        let sum = norm.visual + norm.auditory + norm.textual + norm.temporal + norm.interactivity;
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((norm.visual - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_assess_visual_calm_content() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = VisualMetrics {
            avg_scene_duration_sec: 10.0,
            scene_changes_per_minute: 3.0,
            simultaneous_elements: 1,
            has_high_motion: false,
            has_flashing: false,
            color_complexity: 5,
        };
        let score = analyzer.assess_visual(&metrics);
        assert!(
            score.score < 0.3,
            "Calm content should have low visual load, got {}",
            score.score
        );
        assert!(score.level <= CognitiveLoadLevel::Low);
    }

    #[test]
    fn test_assess_visual_intense_content() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = VisualMetrics {
            avg_scene_duration_sec: 1.5,
            scene_changes_per_minute: 40.0,
            simultaneous_elements: 6,
            has_high_motion: true,
            has_flashing: true,
            color_complexity: 50,
        };
        let score = analyzer.assess_visual(&metrics);
        assert!(
            score.score > 0.7,
            "Intense content should have high visual load, got {}",
            score.score
        );
        assert!(score.level >= CognitiveLoadLevel::High);
        assert!(!score.recommendations.is_empty());
    }

    #[test]
    fn test_assess_auditory_clear_speech() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = AuditoryMetrics {
            speech_rate_wpm: 120.0,
            simultaneous_tracks: 1,
            music_over_speech: false,
            overlapping_speakers: false,
            snr_db: 30.0,
            speaker_count: 1,
        };
        let score = analyzer.assess_auditory(&metrics);
        assert!(
            score.score < 0.3,
            "Clear speech should have low auditory load, got {}",
            score.score
        );
    }

    #[test]
    fn test_assess_auditory_complex() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = AuditoryMetrics {
            speech_rate_wpm: 220.0,
            simultaneous_tracks: 4,
            music_over_speech: true,
            overlapping_speakers: true,
            snr_db: 8.0,
            speaker_count: 6,
        };
        let score = analyzer.assess_auditory(&metrics);
        assert!(
            score.score > 0.5,
            "Complex audio should have high load, got {}",
            score.score
        );
    }

    #[test]
    fn test_assess_textual_simple() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = TextualMetrics {
            reading_grade_level: 5.0,
            avg_words_per_caption: 8.0,
            caption_wpm: 120.0,
            jargon_percentage: 2.0,
            includes_non_speech_info: true,
            unexpanded_acronyms: 0,
        };
        let score = analyzer.assess_textual(&metrics);
        assert!(
            score.score < 0.3,
            "Simple text should have low load, got {}",
            score.score
        );
    }

    #[test]
    fn test_assess_textual_complex() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = TextualMetrics {
            reading_grade_level: 14.0,
            avg_words_per_caption: 20.0,
            caption_wpm: 280.0,
            jargon_percentage: 25.0,
            includes_non_speech_info: false,
            unexpanded_acronyms: 8,
        };
        let score = analyzer.assess_textual(&metrics);
        assert!(
            score.score > 0.5,
            "Complex text should have high load, got {}",
            score.score
        );
        assert!(!score.recommendations.is_empty());
    }

    #[test]
    fn test_assess_temporal_relaxed() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = TemporalMetrics {
            duration_sec: 300.0,
            info_density_per_minute: 10.0,
            has_breathing_room: true,
            avg_topic_duration_sec: 60.0,
            requires_real_time_response: false,
            context_switches: 2,
        };
        let score = analyzer.assess_temporal(&metrics);
        assert!(
            score.score < 0.3,
            "Relaxed pacing should have low temporal load, got {}",
            score.score
        );
    }

    #[test]
    fn test_assess_temporal_intense() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = TemporalMetrics {
            duration_sec: 300.0,
            info_density_per_minute: 60.0,
            has_breathing_room: false,
            avg_topic_duration_sec: 10.0,
            requires_real_time_response: true,
            context_switches: 20,
        };
        let score = analyzer.assess_temporal(&metrics);
        assert!(
            score.score > 0.5,
            "Intense pacing should have high temporal load, got {}",
            score.score
        );
    }

    #[test]
    fn test_assess_interactivity_passive() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = InteractivityMetrics {
            decision_points: 0,
            timed_responses: false,
            navigation_depth: 1,
            supports_undo: true,
            input_channels: 1,
        };
        let score = analyzer.assess_interactivity(&metrics);
        assert!(
            score.score < 0.1,
            "Passive content should have very low interactivity load"
        );
    }

    #[test]
    fn test_assess_interactivity_complex() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let metrics = InteractivityMetrics {
            decision_points: 25,
            timed_responses: true,
            navigation_depth: 6,
            supports_undo: false,
            input_channels: 3,
        };
        let score = analyzer.assess_interactivity(&metrics);
        assert!(
            score.score > 0.6,
            "Complex interactivity should have high load, got {}",
            score.score
        );
    }

    #[test]
    fn test_generate_report_low_load() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let report = analyzer.generate_report(
            &VisualMetrics {
                avg_scene_duration_sec: 10.0,
                scene_changes_per_minute: 3.0,
                simultaneous_elements: 1,
                has_high_motion: false,
                has_flashing: false,
                color_complexity: 5,
            },
            &AuditoryMetrics {
                speech_rate_wpm: 120.0,
                simultaneous_tracks: 1,
                music_over_speech: false,
                overlapping_speakers: false,
                snr_db: 30.0,
                speaker_count: 1,
            },
            &TextualMetrics {
                reading_grade_level: 5.0,
                avg_words_per_caption: 8.0,
                caption_wpm: 120.0,
                jargon_percentage: 2.0,
                includes_non_speech_info: true,
                unexpanded_acronyms: 0,
            },
            &TemporalMetrics {
                duration_sec: 300.0,
                info_density_per_minute: 10.0,
                has_breathing_room: true,
                avg_topic_duration_sec: 60.0,
                requires_real_time_response: false,
                context_switches: 2,
            },
            &InteractivityMetrics::default(),
        );

        assert!(report.overall_score < 0.4);
        assert!(report.meets_accessibility_guidelines);
        assert_eq!(report.dimensions.len(), 5);
    }

    #[test]
    fn test_generate_report_high_load() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let report = analyzer.generate_report(
            &VisualMetrics {
                avg_scene_duration_sec: 1.0,
                scene_changes_per_minute: 40.0,
                simultaneous_elements: 8,
                has_high_motion: true,
                has_flashing: true,
                color_complexity: 100,
            },
            &AuditoryMetrics {
                speech_rate_wpm: 250.0,
                simultaneous_tracks: 4,
                music_over_speech: true,
                overlapping_speakers: true,
                snr_db: 5.0,
                speaker_count: 8,
            },
            &TextualMetrics {
                reading_grade_level: 16.0,
                avg_words_per_caption: 25.0,
                caption_wpm: 300.0,
                jargon_percentage: 30.0,
                includes_non_speech_info: false,
                unexpanded_acronyms: 12,
            },
            &TemporalMetrics {
                duration_sec: 300.0,
                info_density_per_minute: 70.0,
                has_breathing_room: false,
                avg_topic_duration_sec: 8.0,
                requires_real_time_response: true,
                context_switches: 25,
            },
            &InteractivityMetrics {
                decision_points: 30,
                timed_responses: true,
                navigation_depth: 8,
                supports_undo: false,
                input_channels: 4,
            },
        );

        assert!(report.overall_score > 0.6);
        assert!(!report.meets_accessibility_guidelines);
        assert!(!report.top_recommendations.is_empty());
    }

    #[test]
    fn test_quick_assess() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let report = analyzer.quick_assess(&VisualMetrics::default(), &AuditoryMetrics::default());
        assert_eq!(report.dimensions.len(), 5);
        assert!(report.overall_score >= 0.0);
    }

    #[test]
    fn test_custom_weights() {
        let weights = DimensionWeights {
            visual: 0.5,
            auditory: 0.3,
            textual: 0.1,
            temporal: 0.05,
            interactivity: 0.05,
        };
        let analyzer = CognitiveLoadAnalyzer::with_weights(weights);
        let report = analyzer.quick_assess(
            &VisualMetrics {
                scene_changes_per_minute: 40.0,
                has_flashing: true,
                ..VisualMetrics::default()
            },
            &AuditoryMetrics::default(),
        );
        // With heavy visual weighting and high visual load, overall should be higher
        assert!(report.overall_score > 0.2);
    }

    #[test]
    fn test_max_acceptable_score() {
        let analyzer = CognitiveLoadAnalyzer::new().with_max_acceptable(0.3);
        let report = analyzer.generate_report(
            &VisualMetrics {
                scene_changes_per_minute: 20.0,
                has_high_motion: true,
                ..VisualMetrics::default()
            },
            &AuditoryMetrics {
                speech_rate_wpm: 180.0,
                ..AuditoryMetrics::default()
            },
            &TextualMetrics::default(),
            &TemporalMetrics::default(),
            &InteractivityMetrics::default(),
        );
        // With stricter threshold, even moderate content may fail
        assert!(!report.meets_accessibility_guidelines || report.overall_score <= 0.3);
    }

    #[test]
    fn test_recommendations_prioritized() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let report = analyzer.generate_report(
            &VisualMetrics {
                has_flashing: true,
                scene_changes_per_minute: 50.0,
                ..VisualMetrics::default()
            },
            &AuditoryMetrics::default(),
            &TextualMetrics {
                reading_grade_level: 15.0,
                ..TextualMetrics::default()
            },
            &TemporalMetrics::default(),
            &InteractivityMetrics::default(),
        );
        // Should have recommendations
        assert!(!report.top_recommendations.is_empty());
        assert!(report.top_recommendations.len() <= 5);
    }

    #[test]
    fn test_default_metrics_give_low_load() {
        let analyzer = CognitiveLoadAnalyzer::new();
        let report = analyzer.generate_report(
            &VisualMetrics::default(),
            &AuditoryMetrics::default(),
            &TextualMetrics::default(),
            &TemporalMetrics::default(),
            &InteractivityMetrics::default(),
        );
        assert!(report.overall_score < 0.3);
        assert_eq!(report.overall_level, CognitiveLoadLevel::VeryLow);
    }
}
