//! Automatic content warning classification for video content.
//!
//! Analyses video frames and audio features to automatically assign content
//! warning labels covering:
//!
//! - **Violence**: Flash detection, motion spikes, high-energy audio bursts
//! - **Language**: Audio energy patterns consistent with loud/aggressive speech
//! - **Flashing**: Rapid luminance changes that may trigger photosensitive responses
//! - **Themes**: Content-type based thematic flags (e.g. intense, disturbing)
//!
//! The classifier is signal-based (no on-device ML model) and is suitable as a
//! conservative first-pass filter; downstream classifiers can refine the results.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::content_warning::{ContentWarningClassifier, ContentWarningConfig};
//!
//! let config = ContentWarningConfig::default();
//! let classifier = ContentWarningClassifier::new(config);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use crate::scoring::ScoredScene;
use oximedia_core::Timestamp;
use std::collections::HashMap;

// ─── Warning Labels ──────────────────────────────────────────────────────────

/// Individual content warning label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WarningLabel {
    /// Depictions of violence or physical harm.
    Violence,
    /// Strong or offensive language.
    StrongLanguage,
    /// Rapidly flashing lights (photosensitive risk).
    FlashingLights,
    /// Intense or disturbing themes.
    IntenseThemes,
    /// Loud or sudden audio (hearing risk or startle response).
    LoudSuddenAudio,
    /// Drug or substance references inferred from content patterns.
    SubstanceUse,
    /// Sexually suggestive content signals.
    SexualContent,
    /// Graphic imagery inferred from visual features.
    GraphicImagery,
}

impl WarningLabel {
    /// Human-readable label string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Violence => "Violence",
            Self::StrongLanguage => "Strong Language",
            Self::FlashingLights => "Flashing Lights",
            Self::IntenseThemes => "Intense Themes",
            Self::LoudSuddenAudio => "Loud/Sudden Audio",
            Self::SubstanceUse => "Substance Use",
            Self::SexualContent => "Sexual Content",
            Self::GraphicImagery => "Graphic Imagery",
        }
    }

    /// MPAA-style rating tier threshold (0.0 – 1.0).
    ///
    /// Signals above this threshold contribute to that label.
    #[must_use]
    pub const fn default_threshold(&self) -> f64 {
        match self {
            Self::Violence => 0.55,
            Self::StrongLanguage => 0.50,
            Self::FlashingLights => 0.40,
            Self::IntenseThemes => 0.60,
            Self::LoudSuddenAudio => 0.65,
            Self::SubstanceUse => 0.70,
            Self::SexualContent => 0.75,
            Self::GraphicImagery => 0.60,
        }
    }
}

// ─── Classification Result ────────────────────────────────────────────────────

/// A single timestamped content warning event.
#[derive(Debug, Clone)]
pub struct WarningEvent {
    /// Start of the offending segment.
    pub start: Timestamp,
    /// End of the offending segment.
    pub end: Timestamp,
    /// Warning label.
    pub label: WarningLabel,
    /// Signal confidence (0.0 – 1.0).
    pub confidence: f64,
    /// Human-readable explanation.
    pub reason: String,
}

impl WarningEvent {
    /// Create a new warning event.
    #[must_use]
    pub fn new(
        start: Timestamp,
        end: Timestamp,
        label: WarningLabel,
        confidence: f64,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            start,
            end,
            label,
            confidence: confidence.clamp(0.0, 1.0),
            reason: reason.into(),
        }
    }

    /// Duration of the event in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }
}

/// Result of a full content classification pass.
#[derive(Debug, Clone, Default)]
pub struct ContentWarningReport {
    /// All detected warning events in chronological order.
    pub events: Vec<WarningEvent>,
    /// Aggregate label → max confidence across the whole clip.
    pub summary: HashMap<WarningLabel, f64>,
    /// Overall risk score (0.0 – 1.0); derived from the strongest signal found.
    pub overall_risk: f64,
}

impl ContentWarningReport {
    /// Check whether any events carry the given label.
    #[must_use]
    pub fn has_warning(&self, label: WarningLabel) -> bool {
        self.summary.contains_key(&label)
    }

    /// Return the active labels sorted by confidence (highest first).
    #[must_use]
    pub fn active_labels(&self) -> Vec<(WarningLabel, f64)> {
        let mut labels: Vec<_> = self.summary.iter().map(|(&l, &c)| (l, c)).collect();
        labels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        labels
    }

    /// Format a short warning string, e.g. "Violence, Strong Language".
    #[must_use]
    pub fn format_labels(&self) -> String {
        self.active_labels()
            .iter()
            .map(|(l, _)| l.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Return events filtered by a minimum confidence.
    #[must_use]
    pub fn events_above_confidence(&self, min_confidence: f64) -> Vec<&WarningEvent> {
        self.events
            .iter()
            .filter(|e| e.confidence >= min_confidence)
            .collect()
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Thresholds for individual warning detectors.
#[derive(Debug, Clone)]
pub struct WarningThresholds {
    /// Motion spike threshold for violence detection (0.0 – 1.0).
    pub violence_motion_spike: f64,
    /// Audio energy burst threshold for strong-language detection.
    pub language_audio_burst: f64,
    /// Luminance change rate for flashing-light detection (Hz equivalent).
    pub flash_luminance_rate: f64,
    /// High-audio-peak-to-scene threshold for sudden audio detection.
    pub sudden_audio_ratio: f64,
    /// Minimum fraction of flagged scenes for intense-themes label.
    pub intense_themes_scene_fraction: f64,
}

impl Default for WarningThresholds {
    fn default() -> Self {
        Self {
            violence_motion_spike: 0.55,
            language_audio_burst: 0.50,
            flash_luminance_rate: 0.40,
            sudden_audio_ratio: 0.65,
            intense_themes_scene_fraction: 0.20,
        }
    }
}

/// Configuration for the content warning classifier.
#[derive(Debug, Clone)]
pub struct ContentWarningConfig {
    /// Per-label signal thresholds.
    pub thresholds: WarningThresholds,
    /// Minimum confidence required before a label appears in the summary.
    pub min_report_confidence: f64,
    /// Merge adjacent events of the same label if within this gap (ms).
    pub merge_gap_ms: i64,
    /// Minimum event duration to report (ms).
    pub min_event_duration_ms: i64,
    /// Enable flashing-light detector.
    pub detect_flashing: bool,
    /// Enable violence detector.
    pub detect_violence: bool,
    /// Enable strong-language heuristics.
    pub detect_language: bool,
    /// Enable intense-themes detector.
    pub detect_intense_themes: bool,
}

impl Default for ContentWarningConfig {
    fn default() -> Self {
        Self {
            thresholds: WarningThresholds::default(),
            min_report_confidence: 0.40,
            merge_gap_ms: 1000,
            min_event_duration_ms: 100,
            detect_flashing: true,
            detect_violence: true,
            detect_language: true,
            detect_intense_themes: true,
        }
    }
}

impl ContentWarningConfig {
    /// Create a permissive config that only flags very obvious content.
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            thresholds: WarningThresholds {
                violence_motion_spike: 0.75,
                language_audio_burst: 0.70,
                flash_luminance_rate: 0.60,
                sudden_audio_ratio: 0.80,
                intense_themes_scene_fraction: 0.35,
            },
            min_report_confidence: 0.60,
            ..Self::default()
        }
    }

    /// Create a conservative config for family-safe content checks.
    #[must_use]
    pub fn conservative() -> Self {
        Self {
            thresholds: WarningThresholds {
                violence_motion_spike: 0.35,
                language_audio_burst: 0.30,
                flash_luminance_rate: 0.25,
                sudden_audio_ratio: 0.45,
                intense_themes_scene_fraction: 0.10,
            },
            min_report_confidence: 0.25,
            ..Self::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if !(0.0..=1.0).contains(&self.min_report_confidence) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_report_confidence,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.merge_gap_ms < 0 {
            return Err(AutoError::InvalidParameter {
                name: "merge_gap_ms".into(),
                value: "must be non-negative".into(),
            });
        }
        if self.min_event_duration_ms < 0 {
            return Err(AutoError::InvalidParameter {
                name: "min_event_duration_ms".into(),
                value: "must be non-negative".into(),
            });
        }
        Ok(())
    }
}

// ─── Classifier ───────────────────────────────────────────────────────────────

/// Signal-based content warning classifier.
///
/// All detectors are heuristic and operate purely on feature metrics already
/// extracted by other pipeline stages (no raw pixel access required at this
/// level).
pub struct ContentWarningClassifier {
    config: ContentWarningConfig,
}

impl ContentWarningClassifier {
    /// Create a new classifier.
    #[must_use]
    pub fn new(config: ContentWarningConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_classifier() -> Self {
        Self::new(ContentWarningConfig::default())
    }

    /// Classify a sequence of scored scenes from the scoring pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`AutoError::ConfigurationError`] if the config is invalid.
    pub fn classify_scenes(&self, scenes: &[ScoredScene]) -> AutoResult<ContentWarningReport> {
        self.config.validate()?;

        if scenes.is_empty() {
            return Ok(ContentWarningReport::default());
        }

        let mut all_events: Vec<WarningEvent> = Vec::new();

        if self.config.detect_violence {
            all_events.extend(self.detect_violence(scenes));
        }
        if self.config.detect_flashing {
            all_events.extend(self.detect_flashing_lights(scenes));
        }
        if self.config.detect_language {
            all_events.extend(self.detect_strong_language(scenes));
        }
        if self.config.detect_intense_themes {
            all_events.extend(self.detect_intense_themes(scenes));
        }
        all_events.extend(self.detect_sudden_audio(scenes));

        // Sort by start PTS
        all_events.sort_by_key(|e| e.start.pts);

        // Merge adjacent same-label events
        let merged = self.merge_events(all_events);

        // Filter by min event duration and confidence
        let filtered: Vec<WarningEvent> = merged
            .into_iter()
            .filter(|e| {
                e.duration_ms() >= self.config.min_event_duration_ms
                    && e.confidence >= self.config.min_report_confidence
            })
            .collect();

        // Build summary
        let mut summary: HashMap<WarningLabel, f64> = HashMap::new();
        for event in &filtered {
            let entry = summary.entry(event.label).or_insert(0.0);
            if event.confidence > *entry {
                *entry = event.confidence;
            }
        }

        let overall_risk = summary.values().copied().fold(0.0_f64, f64::max);

        Ok(ContentWarningReport {
            events: filtered,
            summary,
            overall_risk,
        })
    }

    /// Classify audio-only using raw sample RMS windows.
    ///
    /// Returns a report containing [`WarningLabel::LoudSuddenAudio`] and
    /// [`WarningLabel::StrongLanguage`] events derived from audio energy.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn classify_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> AutoResult<ContentWarningReport> {
        self.config.validate()?;

        if samples.is_empty() || sample_rate == 0 {
            return Ok(ContentWarningReport::default());
        }

        let timebase = oximedia_core::Rational::new(1, 1000);
        let window_ms: i64 = 200;
        let window_samples = (sample_rate as i64 * window_ms / 1000) as usize;
        let window_samples = window_samples.max(1);

        let mut events: Vec<WarningEvent> = Vec::new();

        for (chunk_idx, chunk) in samples.chunks(window_samples).enumerate() {
            let rms =
                (chunk.iter().map(|&s| (s * s) as f64).sum::<f64>() / chunk.len() as f64).sqrt();
            let peak = chunk
                .iter()
                .map(|&s| s.abs() as f64)
                .fold(0.0_f64, f64::max);

            let start_ms = chunk_idx as i64 * window_ms;
            let end_ms = start_ms + window_ms;

            // Sudden loud audio: peak well above RMS suggests transient burst
            let suddenness = if rms > 1e-6 { peak / rms } else { 0.0 };
            if peak > self.config.thresholds.sudden_audio_ratio && suddenness > 3.0 {
                events.push(WarningEvent::new(
                    Timestamp::new(start_ms, timebase),
                    Timestamp::new(end_ms, timebase),
                    WarningLabel::LoudSuddenAudio,
                    (peak * suddenness / 10.0).clamp(0.0, 1.0),
                    "Sudden audio transient",
                ));
            }

            // Sustained high-energy → possible strong language
            if rms > self.config.thresholds.language_audio_burst {
                events.push(WarningEvent::new(
                    Timestamp::new(start_ms, timebase),
                    Timestamp::new(end_ms, timebase),
                    WarningLabel::StrongLanguage,
                    (rms * 0.8).clamp(0.0, 1.0),
                    "Sustained high audio energy",
                ));
            }
        }

        events.sort_by_key(|e| e.start.pts);
        let merged = self.merge_events(events);
        let filtered: Vec<WarningEvent> = merged
            .into_iter()
            .filter(|e| {
                e.duration_ms() >= self.config.min_event_duration_ms
                    && e.confidence >= self.config.min_report_confidence
            })
            .collect();

        let mut summary: HashMap<WarningLabel, f64> = HashMap::new();
        for event in &filtered {
            let entry = summary.entry(event.label).or_insert(0.0);
            if event.confidence > *entry {
                *entry = event.confidence;
            }
        }
        let overall_risk = summary.values().copied().fold(0.0_f64, f64::max);

        Ok(ContentWarningReport {
            events: filtered,
            summary,
            overall_risk,
        })
    }

    // ─── Private detectors ────────────────────────────────────────────────────

    /// Detect violence signals: motion spikes + high audio energy together.
    fn detect_violence(&self, scenes: &[ScoredScene]) -> Vec<WarningEvent> {
        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut events = Vec::new();

        for scene in scenes {
            let motion = scene.features.motion_intensity;
            let audio_energy = scene.features.audio_energy;

            // Violence heuristic: high motion AND elevated audio simultaneously
            let signal = (motion * 0.60 + audio_energy * 0.40).min(1.0);

            if motion > self.config.thresholds.violence_motion_spike && signal > 0.45 {
                let confidence = signal;
                events.push(WarningEvent::new(
                    Timestamp::new(scene.start.pts, timebase),
                    Timestamp::new(scene.end.pts, timebase),
                    WarningLabel::Violence,
                    confidence,
                    format!(
                        "Motion spike {:.2} with audio energy {:.2}",
                        motion, audio_energy
                    ),
                ));
            }
        }

        events
    }

    /// Detect flashing lights via rapid changes in scene brightness.
    fn detect_flashing_lights(&self, scenes: &[ScoredScene]) -> Vec<WarningEvent> {
        if scenes.len() < 2 {
            return Vec::new();
        }

        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut events = Vec::new();
        let window = 5usize; // look at a rolling window of consecutive scenes

        for i in 0..scenes.len().saturating_sub(window) {
            let slice = &scenes[i..i + window];
            // Compute brightness variance across the window
            let brightnesses: Vec<f64> = slice.iter().map(|s| s.features.brightness_mean).collect();
            let mean = brightnesses.iter().sum::<f64>() / brightnesses.len() as f64;
            let variance = brightnesses
                .iter()
                .map(|&b| (b - mean).powi(2))
                .sum::<f64>()
                / brightnesses.len() as f64;
            let std_dev = variance.sqrt();

            // High std_dev relative to the flash threshold → rapid luminance swings
            if std_dev > self.config.thresholds.flash_luminance_rate {
                let confidence = (std_dev / 0.5).clamp(0.0, 1.0);
                let start = slice.first().map_or(scenes[i].start, |s| s.start);
                let end = slice.last().map_or(scenes[i].end, |s| s.end);
                events.push(WarningEvent::new(
                    Timestamp::new(start.pts, timebase),
                    Timestamp::new(end.pts, timebase),
                    WarningLabel::FlashingLights,
                    confidence,
                    format!("Brightness std-dev {std_dev:.3} over {window} scenes"),
                ));
            }
        }

        events
    }

    /// Detect strong language via sustained high audio energy.
    fn detect_strong_language(&self, scenes: &[ScoredScene]) -> Vec<WarningEvent> {
        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut events = Vec::new();

        for scene in scenes {
            // High sustained energy with relatively low motion → likely speech, not action
            let audio = scene.features.audio_energy;
            let motion = scene.features.motion_intensity;

            if audio > self.config.thresholds.language_audio_burst && motion < 0.50 {
                let confidence = (audio * 0.75 + (0.5 - motion) * 0.25).clamp(0.0, 1.0);
                events.push(WarningEvent::new(
                    Timestamp::new(scene.start.pts, timebase),
                    Timestamp::new(scene.end.pts, timebase),
                    WarningLabel::StrongLanguage,
                    confidence,
                    format!("High audio energy {audio:.2} in low-motion scene"),
                ));
            }
        }

        events
    }

    /// Detect intense themes from the overall scene score distribution.
    fn detect_intense_themes(&self, scenes: &[ScoredScene]) -> Vec<WarningEvent> {
        if scenes.is_empty() {
            return Vec::new();
        }

        let timebase = oximedia_core::Rational::new(1, 1000);

        let high_intensity_count = scenes.iter().filter(|s| s.adjusted_score() > 0.70).count();
        let fraction = high_intensity_count as f64 / scenes.len() as f64;

        if fraction < self.config.thresholds.intense_themes_scene_fraction {
            return Vec::new();
        }

        // Flag the entire span as potentially intense
        let start = scenes.first().map_or_else(
            || Timestamp::new(0, timebase),
            |s| Timestamp::new(s.start.pts, timebase),
        );
        let end = scenes.last().map_or_else(
            || Timestamp::new(0, timebase),
            |s| Timestamp::new(s.end.pts, timebase),
        );
        let confidence = fraction.clamp(0.0, 1.0);

        vec![WarningEvent::new(
            start,
            end,
            WarningLabel::IntenseThemes,
            confidence,
            format!(
                "{:.0}% of scenes have high intensity scores",
                fraction * 100.0
            ),
        )]
    }

    /// Detect sudden audio peaks via audio_peak feature.
    fn detect_sudden_audio(&self, scenes: &[ScoredScene]) -> Vec<WarningEvent> {
        let timebase = oximedia_core::Rational::new(1, 1000);
        let mut events = Vec::new();

        for scene in scenes {
            let peak = scene.features.audio_peak;
            let energy = scene.features.audio_energy;

            // Peak significantly above running energy → sudden transient
            if peak > self.config.thresholds.sudden_audio_ratio
                && energy > 0.0
                && peak / energy.max(1e-6) > 2.5
            {
                let confidence = (peak * 0.8).clamp(0.0, 1.0);
                events.push(WarningEvent::new(
                    Timestamp::new(scene.start.pts, timebase),
                    Timestamp::new(scene.end.pts, timebase),
                    WarningLabel::LoudSuddenAudio,
                    confidence,
                    format!(
                        "Audio peak {peak:.2} ({:.1}× above energy)",
                        peak / energy.max(1e-6)
                    ),
                ));
            }
        }

        events
    }

    /// Merge adjacent same-label events within `merge_gap_ms`.
    fn merge_events(&self, events: Vec<WarningEvent>) -> Vec<WarningEvent> {
        if events.is_empty() {
            return events;
        }

        let mut merged: Vec<WarningEvent> = Vec::new();
        let mut current = events[0].clone();

        for next in events.into_iter().skip(1) {
            // Same label and within merge gap?
            let gap = next.start.pts - current.end.pts;
            if next.label == current.label && gap <= self.config.merge_gap_ms {
                // Extend current event and keep highest confidence
                current.end = next.end;
                if next.confidence > current.confidence {
                    current.confidence = next.confidence;
                    current.reason = next.reason;
                }
            } else {
                merged.push(current);
                current = next;
            }
        }
        merged.push(current);
        merged
    }
}

impl Default for ContentWarningClassifier {
    fn default() -> Self {
        Self::default_classifier()
    }
}

// ─── Public helpers ───────────────────────────────────────────────────────────

/// Generate a content warning string suitable for display (e.g. "Rated: Violence, Strong Language").
#[must_use]
pub fn format_content_rating(report: &ContentWarningReport) -> String {
    if report.summary.is_empty() {
        return "No content warnings".to_string();
    }
    format!("Content warnings: {}", report.format_labels())
}

/// Check if a clip is safe for general audiences (no significant warnings).
#[must_use]
pub fn is_general_audience_safe(report: &ContentWarningReport, risk_threshold: f64) -> bool {
    report.overall_risk < risk_threshold
        && !report.has_warning(WarningLabel::Violence)
        && !report.has_warning(WarningLabel::StrongLanguage)
        && !report.has_warning(WarningLabel::GraphicImagery)
}

/// Compute a simplified age-rating string from a report.
#[must_use]
pub fn suggest_age_rating(report: &ContentWarningReport) -> &'static str {
    let risk = report.overall_risk;
    if risk < 0.20 {
        "G"
    } else if risk < 0.40 {
        "PG"
    } else if risk < 0.60 {
        "PG-13"
    } else if risk < 0.80 {
        "R"
    } else {
        "NC-17"
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::{ContentType, SceneFeatures, ScoredScene, Sentiment};
    use oximedia_core::{Rational, Timestamp};

    fn make_timestamp(ms: i64) -> Timestamp {
        Timestamp::new(ms, Rational::new(1, 1000))
    }

    fn make_scene(
        start_ms: i64,
        end_ms: i64,
        motion: f64,
        audio_energy: f64,
        audio_peak: f64,
        brightness: f64,
        score: f64,
    ) -> ScoredScene {
        let mut scene = ScoredScene::new(
            make_timestamp(start_ms),
            make_timestamp(end_ms),
            score,
            ContentType::Action,
            Sentiment::Tense,
        );
        scene.features = SceneFeatures {
            motion_intensity: motion,
            audio_energy,
            audio_peak,
            brightness_mean: brightness,
            ..SceneFeatures::default()
        };
        scene
    }

    #[test]
    fn test_default_config_valid() {
        let config = ContentWarningConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_permissive_config_valid() {
        let config = ContentWarningConfig::permissive();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_conservative_config_valid() {
        let config = ContentWarningConfig::conservative();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_confidence_threshold() {
        let mut config = ContentWarningConfig::default();
        config.min_report_confidence = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_empty_scenes_returns_empty_report() {
        let classifier = ContentWarningClassifier::default();
        let report = classifier
            .classify_scenes(&[])
            .expect("classify scenes should succeed");
        assert!(report.events.is_empty());
        assert!(report.summary.is_empty());
        assert_eq!(report.overall_risk, 0.0);
    }

    #[test]
    fn test_violence_detection_high_motion() {
        let classifier = ContentWarningClassifier::new(ContentWarningConfig {
            thresholds: WarningThresholds {
                violence_motion_spike: 0.50,
                ..WarningThresholds::default()
            },
            min_report_confidence: 0.0, // accept any confidence for testing
            ..ContentWarningConfig::default()
        });

        let scenes = vec![
            make_scene(0, 3000, 0.90, 0.80, 0.10, 0.50, 0.80),
            make_scene(3000, 6000, 0.85, 0.75, 0.10, 0.50, 0.75),
        ];

        let report = classifier
            .classify_scenes(&scenes)
            .expect("classify scenes should succeed");
        assert!(
            report.has_warning(WarningLabel::Violence),
            "Expected Violence warning for high-motion/high-audio scenes"
        );
    }

    #[test]
    fn test_no_violence_in_calm_scene() {
        let classifier = ContentWarningClassifier::default();
        let scenes = vec![make_scene(0, 5000, 0.10, 0.15, 0.05, 0.60, 0.30)];
        let report = classifier
            .classify_scenes(&scenes)
            .expect("classify scenes should succeed");
        assert!(!report.has_warning(WarningLabel::Violence));
    }

    #[test]
    fn test_flashing_light_detection() {
        let classifier = ContentWarningClassifier::new(ContentWarningConfig {
            thresholds: WarningThresholds {
                flash_luminance_rate: 0.20,
                ..WarningThresholds::default()
            },
            min_report_confidence: 0.0,
            ..ContentWarningConfig::default()
        });

        // Alternating bright and dark scenes → high brightness variance
        let scenes: Vec<ScoredScene> = (0..10)
            .map(|i| {
                let brightness = if i % 2 == 0 { 0.90 } else { 0.05 };
                make_scene(i * 500, (i + 1) * 500, 0.20, 0.20, 0.10, brightness, 0.40)
            })
            .collect();

        let report = classifier
            .classify_scenes(&scenes)
            .expect("classify scenes should succeed");
        assert!(
            report.has_warning(WarningLabel::FlashingLights),
            "Expected FlashingLights warning for alternating bright/dark scenes"
        );
    }

    #[test]
    fn test_strong_language_detection() {
        let classifier = ContentWarningClassifier::new(ContentWarningConfig {
            thresholds: WarningThresholds {
                language_audio_burst: 0.30,
                ..WarningThresholds::default()
            },
            min_report_confidence: 0.0,
            ..ContentWarningConfig::default()
        });

        // Low motion, high audio energy → strong language
        let scenes = vec![make_scene(0, 5000, 0.10, 0.75, 0.20, 0.50, 0.50)];
        let report = classifier
            .classify_scenes(&scenes)
            .expect("classify scenes should succeed");
        assert!(report.has_warning(WarningLabel::StrongLanguage));
    }

    #[test]
    fn test_intense_themes_detection() {
        let classifier = ContentWarningClassifier::new(ContentWarningConfig {
            thresholds: WarningThresholds {
                intense_themes_scene_fraction: 0.50,
                ..WarningThresholds::default()
            },
            min_report_confidence: 0.0,
            ..ContentWarningConfig::default()
        });

        // All scenes with high importance scores
        let scenes: Vec<ScoredScene> = (0..5)
            .map(|i| make_scene(i * 2000, (i + 1) * 2000, 0.30, 0.30, 0.10, 0.50, 0.85))
            .collect();

        let report = classifier
            .classify_scenes(&scenes)
            .expect("classify scenes should succeed");
        assert!(report.has_warning(WarningLabel::IntenseThemes));
    }

    #[test]
    fn test_sudden_audio_detection() {
        let classifier = ContentWarningClassifier::new(ContentWarningConfig {
            thresholds: WarningThresholds {
                sudden_audio_ratio: 0.50,
                ..WarningThresholds::default()
            },
            min_report_confidence: 0.0,
            ..ContentWarningConfig::default()
        });

        // Scene with very high peak relative to low energy
        let mut scene = make_scene(0, 2000, 0.20, 0.10, 0.90, 0.50, 0.50);
        scene.features.audio_peak = 0.90;
        scene.features.audio_energy = 0.10;

        let report = classifier
            .classify_scenes(&[scene])
            .expect("classify scenes should succeed");
        assert!(report.has_warning(WarningLabel::LoudSuddenAudio));
    }

    #[test]
    fn test_report_active_labels_sorted() {
        let classifier = ContentWarningClassifier::new(ContentWarningConfig {
            thresholds: WarningThresholds {
                violence_motion_spike: 0.30,
                language_audio_burst: 0.20,
                intense_themes_scene_fraction: 0.10,
                ..WarningThresholds::default()
            },
            min_report_confidence: 0.0,
            ..ContentWarningConfig::default()
        });

        let scenes = vec![
            make_scene(0, 3000, 0.80, 0.75, 0.10, 0.50, 0.85),
            make_scene(3000, 6000, 0.05, 0.70, 0.10, 0.50, 0.85),
        ];

        let report = classifier
            .classify_scenes(&scenes)
            .expect("classify scenes should succeed");
        let labels = report.active_labels();
        // Labels should be sorted descending by confidence
        for window in labels.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Labels not sorted by confidence"
            );
        }
    }

    #[test]
    fn test_format_content_rating_empty() {
        let report = ContentWarningReport::default();
        assert_eq!(format_content_rating(&report), "No content warnings");
    }

    #[test]
    fn test_suggest_age_rating() {
        let mut report = ContentWarningReport::default();

        report.overall_risk = 0.10;
        assert_eq!(suggest_age_rating(&report), "G");

        report.overall_risk = 0.35;
        assert_eq!(suggest_age_rating(&report), "PG");

        report.overall_risk = 0.55;
        assert_eq!(suggest_age_rating(&report), "PG-13");

        report.overall_risk = 0.70;
        assert_eq!(suggest_age_rating(&report), "R");

        report.overall_risk = 0.90;
        assert_eq!(suggest_age_rating(&report), "NC-17");
    }

    #[test]
    fn test_is_general_audience_safe() {
        let report = ContentWarningReport::default();
        assert!(is_general_audience_safe(&report, 0.30));
    }

    #[test]
    fn test_audio_classify_empty() {
        let classifier = ContentWarningClassifier::default();
        let report = classifier
            .classify_audio(&[], 44100)
            .expect("classify audio should succeed");
        assert!(report.events.is_empty());
    }

    #[test]
    fn test_audio_classify_loud_transient() {
        let classifier = ContentWarningClassifier::new(ContentWarningConfig {
            thresholds: WarningThresholds {
                sudden_audio_ratio: 0.30,
                ..WarningThresholds::default()
            },
            min_report_confidence: 0.0,
            ..ContentWarningConfig::default()
        });

        // Build a signal: mostly quiet with a loud transient burst
        let sr = 8000u32;
        let mut samples = vec![0.05f32; sr as usize]; // 1 second quiet
                                                      // Insert a loud spike at the 0.5 s mark
        let spike_start = sr as usize / 2;
        for s in samples.iter_mut().skip(spike_start).take(100) {
            *s = 0.95;
        }

        let report = classifier
            .classify_audio(&samples, sr)
            .expect("classify audio should succeed");
        // The spike should trigger LoudSuddenAudio (or StrongLanguage)
        let has_any = report.has_warning(WarningLabel::LoudSuddenAudio)
            || report.has_warning(WarningLabel::StrongLanguage);
        assert!(has_any, "Expected an audio warning for the loud transient");
    }

    #[test]
    fn test_warning_event_duration() {
        let tb = Rational::new(1, 1000);
        let event = WarningEvent::new(
            Timestamp::new(0, tb),
            Timestamp::new(5000, tb),
            WarningLabel::Violence,
            0.8,
            "test",
        );
        assert_eq!(event.duration_ms(), 5000);
    }

    #[test]
    fn test_merge_adjacent_events() {
        // Use a classifier that detects ONLY violence (not intense themes or others)
        // so we can test the merge logic cleanly without other labels interleaving.
        let classifier = ContentWarningClassifier::new(ContentWarningConfig {
            thresholds: WarningThresholds {
                violence_motion_spike: 0.30,
                ..WarningThresholds::default()
            },
            merge_gap_ms: 5000,
            min_report_confidence: 0.0,
            min_event_duration_ms: 0,
            detect_flashing: false,
            detect_language: false,
            detect_intense_themes: false, // disable so only violence events are present
            ..ContentWarningConfig::default()
        });

        // Two consecutive high-motion scenes very close together
        // audio_peak=0.05, audio_energy=0.05 so sudden-audio detector does not fire.
        let scenes = vec![
            make_scene(0, 1000, 0.80, 0.70, 0.05, 0.50, 0.80),
            make_scene(1100, 2000, 0.80, 0.70, 0.05, 0.50, 0.80),
        ];

        let report = classifier
            .classify_scenes(&scenes)
            .expect("classify scenes should succeed");
        // Because the gap (100ms) is less than merge_gap_ms (5000ms),
        // the two violence events should have been merged into one.
        let violence_events: Vec<_> = report
            .events
            .iter()
            .filter(|e| e.label == WarningLabel::Violence)
            .collect();
        assert_eq!(
            violence_events.len(),
            1,
            "Expected merged violence events; got {violence_events:?}"
        );
    }
}
