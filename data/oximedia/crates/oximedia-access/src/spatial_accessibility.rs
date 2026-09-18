//! Audio spatializer for accessibility: directional audio cues for visually impaired users.
//!
//! Provides spatial audio indicators that convey directional information
//! (left, right, front, behind, up, down) through stereo panning, volume,
//! and frequency cues. This helps visually impaired users understand
//! spatial relationships in media content.

use serde::{Deserialize, Serialize};

/// A spatial direction indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpatialDirection {
    /// Sound from the left.
    Left,
    /// Sound from the right.
    Right,
    /// Sound from ahead / center.
    Front,
    /// Sound from behind.
    Behind,
    /// Sound from above.
    Up,
    /// Sound from below.
    Down,
    /// Sound all around (ambient).
    Ambient,
}

impl SpatialDirection {
    /// Get a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Left => "Left",
            Self::Right => "Right",
            Self::Front => "Front",
            Self::Behind => "Behind",
            Self::Up => "Up",
            Self::Down => "Down",
            Self::Ambient => "Ambient",
        }
    }

    /// Get the stereo pan position (-1.0 = full left, 0.0 = center, 1.0 = full right).
    #[must_use]
    pub fn pan_position(&self) -> f64 {
        match self {
            Self::Left => -0.8,
            Self::Right => 0.8,
            Self::Front => 0.0,
            Self::Behind => 0.0,
            Self::Up => 0.0,
            Self::Down => 0.0,
            Self::Ambient => 0.0,
        }
    }

    /// Get the relative volume adjustment for directional emphasis (0.0 to 1.0).
    #[must_use]
    pub fn volume_factor(&self) -> f64 {
        match self {
            Self::Front => 1.0,
            Self::Left | Self::Right => 0.9,
            Self::Up => 0.8,
            Self::Down => 0.7,
            Self::Behind => 0.6,
            Self::Ambient => 0.5,
        }
    }

    /// Get a frequency shift factor to differentiate front/behind.
    /// Values > 1.0 raise pitch slightly, < 1.0 lower pitch.
    #[must_use]
    pub fn frequency_factor(&self) -> f64 {
        match self {
            Self::Front => 1.0,
            Self::Behind => 0.95, // Slightly muffled
            Self::Up => 1.05,     // Slightly brighter
            Self::Down => 0.92,   // Slightly duller
            Self::Left | Self::Right => 1.0,
            Self::Ambient => 1.0,
        }
    }
}

/// A spatial audio cue to be rendered at a specific time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialCue {
    /// When this cue occurs (milliseconds from start).
    pub time_ms: i64,
    /// Duration of the cue in milliseconds.
    pub duration_ms: i64,
    /// Direction of the cue.
    pub direction: SpatialDirection,
    /// Intensity of the cue (0.0 to 1.0).
    pub intensity: f64,
    /// Optional label describing what the cue represents.
    pub label: Option<String>,
}

impl SpatialCue {
    /// Create a new spatial cue.
    #[must_use]
    pub fn new(time_ms: i64, duration_ms: i64, direction: SpatialDirection) -> Self {
        Self {
            time_ms,
            duration_ms,
            direction,
            intensity: 1.0,
            label: None,
        }
    }

    /// Set intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// End time in milliseconds.
    #[must_use]
    pub fn end_time_ms(&self) -> i64 {
        self.time_ms + self.duration_ms
    }

    /// Compute the left channel gain for this cue.
    #[must_use]
    pub fn left_gain(&self) -> f64 {
        let pan = self.direction.pan_position();
        let vol = self.direction.volume_factor() * self.intensity;
        // Equal-power panning law approximation
        let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
        vol * angle.cos()
    }

    /// Compute the right channel gain for this cue.
    #[must_use]
    pub fn right_gain(&self) -> f64 {
        let pan = self.direction.pan_position();
        let vol = self.direction.volume_factor() * self.intensity;
        let angle = (pan + 1.0) * std::f64::consts::FRAC_PI_4;
        vol * angle.sin()
    }
}

/// Configuration for the spatial accessibility engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialAccessConfig {
    /// Base volume for spatial cues (0.0 to 1.0).
    pub base_volume: f64,
    /// Whether to use frequency cues for front/behind differentiation.
    pub use_frequency_cues: bool,
    /// Whether to duck main audio during cues.
    pub duck_main_audio: bool,
    /// Amount to duck main audio in dB (negative).
    pub duck_amount_db: f64,
    /// Fade-in time for cues in milliseconds.
    pub fade_in_ms: i64,
    /// Fade-out time for cues in milliseconds.
    pub fade_out_ms: i64,
    /// Sample rate for audio processing.
    pub sample_rate: u32,
}

impl Default for SpatialAccessConfig {
    fn default() -> Self {
        Self {
            base_volume: 0.7,
            use_frequency_cues: true,
            duck_main_audio: true,
            duck_amount_db: -6.0,
            fade_in_ms: 50,
            fade_out_ms: 100,
            sample_rate: 48000,
        }
    }
}

/// Applies stereo panning to an audio sample pair based on a spatial cue.
#[derive(Debug, Clone)]
pub struct PanResult {
    /// Left channel sample.
    pub left: f64,
    /// Right channel sample.
    pub right: f64,
}

/// Audio spatializer engine for accessibility.
///
/// Takes spatial cues and applies them to stereo audio, producing
/// directional indicators that visually impaired users can perceive.
pub struct SpatialAccessibilityEngine {
    config: SpatialAccessConfig,
    cues: Vec<SpatialCue>,
}

impl SpatialAccessibilityEngine {
    /// Create a new spatial accessibility engine.
    #[must_use]
    pub fn new(config: SpatialAccessConfig) -> Self {
        Self {
            config,
            cues: Vec::new(),
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SpatialAccessConfig::default())
    }

    /// Add a spatial cue.
    pub fn add_cue(&mut self, cue: SpatialCue) {
        self.cues.push(cue);
    }

    /// Add multiple cues.
    pub fn add_cues(&mut self, cues: Vec<SpatialCue>) {
        self.cues.extend(cues);
    }

    /// Clear all cues.
    pub fn clear_cues(&mut self) {
        self.cues.clear();
    }

    /// Get the number of cues.
    #[must_use]
    pub fn cue_count(&self) -> usize {
        self.cues.len()
    }

    /// Get active cues at a given time.
    #[must_use]
    pub fn active_cues_at(&self, time_ms: i64) -> Vec<&SpatialCue> {
        self.cues
            .iter()
            .filter(|c| time_ms >= c.time_ms && time_ms < c.end_time_ms())
            .collect()
    }

    /// Apply spatial processing to a stereo sample pair at a given time.
    ///
    /// `left` and `right` are the original stereo samples (-1.0 to 1.0).
    /// Returns the processed sample pair with spatial cues mixed in.
    #[must_use]
    pub fn process_sample(&self, left: f64, right: f64, time_ms: i64) -> PanResult {
        let active = self.active_cues_at(time_ms);

        if active.is_empty() {
            return PanResult { left, right };
        }

        let mut out_left = left;
        let mut out_right = right;

        // Apply ducking if enabled
        if self.config.duck_main_audio {
            let duck_factor = 10.0_f64.powf(self.config.duck_amount_db / 20.0);
            out_left *= duck_factor;
            out_right *= duck_factor;
        }

        // Mix in spatial cues
        for cue in &active {
            let fade = self.compute_fade(cue, time_ms);
            let cue_left = cue.left_gain() * self.config.base_volume * fade;
            let cue_right = cue.right_gain() * self.config.base_volume * fade;

            out_left += cue_left * 0.3; // Mix at 30% to not overpower
            out_right += cue_right * 0.3;
        }

        // Soft clip to prevent clipping
        PanResult {
            left: soft_clip(out_left),
            right: soft_clip(out_right),
        }
    }

    /// Compute the fade envelope for a cue at a given time.
    fn compute_fade(&self, cue: &SpatialCue, time_ms: i64) -> f64 {
        let elapsed = time_ms - cue.time_ms;
        let remaining = cue.end_time_ms() - time_ms;

        let fade_in = if self.config.fade_in_ms > 0 && elapsed < self.config.fade_in_ms {
            elapsed as f64 / self.config.fade_in_ms as f64
        } else {
            1.0
        };

        let fade_out = if self.config.fade_out_ms > 0 && remaining < self.config.fade_out_ms {
            remaining as f64 / self.config.fade_out_ms as f64
        } else {
            1.0
        };

        fade_in * fade_out
    }

    /// Sort cues by time.
    pub fn sort_cues(&mut self) {
        self.cues.sort_by_key(|c| c.time_ms);
    }

    /// Get the total duration covered by cues (end of last cue).
    #[must_use]
    pub fn total_duration_ms(&self) -> i64 {
        self.cues.iter().map(|c| c.end_time_ms()).max().unwrap_or(0)
    }

    /// Generate a description of the spatial cue timeline for screen readers.
    #[must_use]
    pub fn describe_timeline(&self) -> Vec<String> {
        let mut sorted = self.cues.clone();
        sorted.sort_by_key(|c| c.time_ms);

        sorted
            .iter()
            .map(|c| {
                let time_s = c.time_ms as f64 / 1000.0;
                let label = c.label.as_deref().unwrap_or("directional cue");
                format!(
                    "{:.1}s: {} from {} (intensity {:.0}%)",
                    time_s,
                    label,
                    c.direction.label(),
                    c.intensity * 100.0
                )
            })
            .collect()
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &SpatialAccessConfig {
        &self.config
    }
}

/// Soft clip function to prevent audio clipping using tanh-based saturation.
fn soft_clip(x: f64) -> f64 {
    if x.abs() <= 0.9 {
        x
    } else {
        x.signum() * (0.9 + 0.1 * ((x.abs() - 0.9) / 0.1).tanh())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_direction_labels() {
        assert_eq!(SpatialDirection::Left.label(), "Left");
        assert_eq!(SpatialDirection::Right.label(), "Right");
        assert_eq!(SpatialDirection::Front.label(), "Front");
        assert_eq!(SpatialDirection::Behind.label(), "Behind");
        assert_eq!(SpatialDirection::Up.label(), "Up");
        assert_eq!(SpatialDirection::Down.label(), "Down");
        assert_eq!(SpatialDirection::Ambient.label(), "Ambient");
    }

    #[test]
    fn test_spatial_direction_pan() {
        assert!(SpatialDirection::Left.pan_position() < 0.0);
        assert!(SpatialDirection::Right.pan_position() > 0.0);
        assert!((SpatialDirection::Front.pan_position()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spatial_direction_volume() {
        assert!(
            SpatialDirection::Front.volume_factor() >= SpatialDirection::Behind.volume_factor()
        );
        assert!(SpatialDirection::Left.volume_factor() > SpatialDirection::Ambient.volume_factor());
    }

    #[test]
    fn test_spatial_direction_frequency() {
        assert!(
            SpatialDirection::Up.frequency_factor() > SpatialDirection::Down.frequency_factor()
        );
        assert!((SpatialDirection::Front.frequency_factor() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spatial_cue_creation() {
        let cue = SpatialCue::new(1000, 500, SpatialDirection::Left)
            .with_intensity(0.8)
            .with_label("approaching car");

        assert_eq!(cue.time_ms, 1000);
        assert_eq!(cue.duration_ms, 500);
        assert_eq!(cue.end_time_ms(), 1500);
        assert!((cue.intensity - 0.8).abs() < f64::EPSILON);
        assert_eq!(cue.label.as_deref(), Some("approaching car"));
    }

    #[test]
    fn test_spatial_cue_gains_left() {
        let cue = SpatialCue::new(0, 1000, SpatialDirection::Left);
        assert!(cue.left_gain() > cue.right_gain());
    }

    #[test]
    fn test_spatial_cue_gains_right() {
        let cue = SpatialCue::new(0, 1000, SpatialDirection::Right);
        assert!(cue.right_gain() > cue.left_gain());
    }

    #[test]
    fn test_spatial_cue_gains_center() {
        let cue = SpatialCue::new(0, 1000, SpatialDirection::Front);
        let diff = (cue.left_gain() - cue.right_gain()).abs();
        assert!(diff < 0.01); // Approximately equal for center
    }

    #[test]
    fn test_spatial_cue_intensity_clamping() {
        let cue = SpatialCue::new(0, 1000, SpatialDirection::Left).with_intensity(2.0);
        assert!((cue.intensity - 1.0).abs() < f64::EPSILON);

        let cue2 = SpatialCue::new(0, 1000, SpatialDirection::Left).with_intensity(-0.5);
        assert!(cue2.intensity.abs() < f64::EPSILON);
    }

    #[test]
    fn test_engine_creation() {
        let engine = SpatialAccessibilityEngine::with_defaults();
        assert_eq!(engine.cue_count(), 0);
    }

    #[test]
    fn test_engine_add_cues() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        engine.add_cue(SpatialCue::new(0, 1000, SpatialDirection::Left));
        engine.add_cue(SpatialCue::new(2000, 500, SpatialDirection::Right));
        assert_eq!(engine.cue_count(), 2);
    }

    #[test]
    fn test_engine_clear_cues() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        engine.add_cue(SpatialCue::new(0, 1000, SpatialDirection::Left));
        engine.clear_cues();
        assert_eq!(engine.cue_count(), 0);
    }

    #[test]
    fn test_engine_active_cues() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        engine.add_cue(SpatialCue::new(1000, 2000, SpatialDirection::Left));
        engine.add_cue(SpatialCue::new(2000, 1000, SpatialDirection::Right));

        let active = engine.active_cues_at(500);
        assert!(active.is_empty());

        let active = engine.active_cues_at(1500);
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].direction, SpatialDirection::Left);

        let active = engine.active_cues_at(2500);
        assert_eq!(active.len(), 2); // Both active at 2500
    }

    #[test]
    fn test_engine_process_no_cues() {
        let engine = SpatialAccessibilityEngine::with_defaults();
        let result = engine.process_sample(0.5, 0.5, 0);
        // No cues: pass-through
        assert!((result.left - 0.5).abs() < f64::EPSILON);
        assert!((result.right - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engine_process_left_cue() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        engine.add_cue(SpatialCue::new(0, 2000, SpatialDirection::Left));

        let result = engine.process_sample(0.0, 0.0, 500);
        // Left cue: left channel should have more signal
        assert!(result.left.abs() > result.right.abs());
    }

    #[test]
    fn test_engine_process_right_cue() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        engine.add_cue(SpatialCue::new(0, 2000, SpatialDirection::Right));

        let result = engine.process_sample(0.0, 0.0, 500);
        // Right cue: right channel should have more signal
        assert!(result.right.abs() > result.left.abs());
    }

    #[test]
    fn test_engine_ducking() {
        let config = SpatialAccessConfig {
            duck_main_audio: true,
            duck_amount_db: -6.0,
            ..SpatialAccessConfig::default()
        };
        let mut engine = SpatialAccessibilityEngine::new(config);
        engine.add_cue(SpatialCue::new(0, 2000, SpatialDirection::Front));

        let result = engine.process_sample(1.0, 1.0, 500);
        // Ducked audio should be less than original
        // Note: cue also adds signal, but duck reduces original by ~6dB (factor ~0.5)
        // The original gets ducked from 1.0 to ~0.5, then cue adds back some
        assert!(result.left < 1.0);
    }

    #[test]
    fn test_engine_total_duration() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        engine.add_cue(SpatialCue::new(0, 1000, SpatialDirection::Left));
        engine.add_cue(SpatialCue::new(5000, 2000, SpatialDirection::Right));
        assert_eq!(engine.total_duration_ms(), 7000);
    }

    #[test]
    fn test_engine_total_duration_empty() {
        let engine = SpatialAccessibilityEngine::with_defaults();
        assert_eq!(engine.total_duration_ms(), 0);
    }

    #[test]
    fn test_engine_describe_timeline() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        engine.add_cue(SpatialCue::new(1000, 500, SpatialDirection::Left).with_label("car horn"));
        engine.add_cue(SpatialCue::new(3000, 1000, SpatialDirection::Right));

        let descriptions = engine.describe_timeline();
        assert_eq!(descriptions.len(), 2);
        assert!(descriptions[0].contains("car horn"));
        assert!(descriptions[0].contains("Left"));
        assert!(descriptions[1].contains("Right"));
    }

    #[test]
    fn test_engine_sort_cues() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        engine.add_cue(SpatialCue::new(5000, 500, SpatialDirection::Right));
        engine.add_cue(SpatialCue::new(1000, 500, SpatialDirection::Left));

        engine.sort_cues();
        let desc = engine.describe_timeline();
        assert!(desc[0].contains("1.0s"));
        assert!(desc[1].contains("5.0s"));
    }

    #[test]
    fn test_soft_clip_within_range() {
        assert!((soft_clip(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((soft_clip(-0.5) - (-0.5)).abs() < f64::EPSILON);
        // At 0.9 it should be identity
        assert!((soft_clip(0.9) - 0.9).abs() < f64::EPSILON);
        // At 1.0 it should be close to 1.0 but slightly less (soft clipped)
        let clipped = soft_clip(1.0);
        assert!(clipped > 0.95 && clipped <= 1.0);
    }

    #[test]
    fn test_soft_clip_above_range() {
        let clipped = soft_clip(2.0);
        assert!(clipped > 0.0 && clipped <= 1.0);
    }

    #[test]
    fn test_soft_clip_below_range() {
        let clipped = soft_clip(-2.0);
        assert!(clipped < 0.0 && clipped >= -1.0);
    }

    #[test]
    fn test_config_defaults() {
        let config = SpatialAccessConfig::default();
        assert!((config.base_volume - 0.7).abs() < f64::EPSILON);
        assert!(config.use_frequency_cues);
        assert!(config.duck_main_audio);
        assert_eq!(config.sample_rate, 48000);
    }

    #[test]
    fn test_engine_fade_in() {
        let config = SpatialAccessConfig {
            fade_in_ms: 100,
            fade_out_ms: 0,
            ..SpatialAccessConfig::default()
        };
        let mut engine = SpatialAccessibilityEngine::new(config);
        engine.add_cue(SpatialCue::new(0, 1000, SpatialDirection::Left));

        // At time 0, fade should be ~0
        let early = engine.process_sample(0.0, 0.0, 0);
        // At time 50 (mid fade), should be partially faded in
        let mid = engine.process_sample(0.0, 0.0, 50);
        // At time 200 (past fade), should be full
        let late = engine.process_sample(0.0, 0.0, 200);

        assert!(early.left.abs() <= mid.left.abs());
        assert!(mid.left.abs() <= late.left.abs());
    }

    #[test]
    fn test_engine_add_multiple_cues() {
        let mut engine = SpatialAccessibilityEngine::with_defaults();
        let cues = vec![
            SpatialCue::new(0, 500, SpatialDirection::Left),
            SpatialCue::new(1000, 500, SpatialDirection::Right),
            SpatialCue::new(2000, 500, SpatialDirection::Front),
        ];
        engine.add_cues(cues);
        assert_eq!(engine.cue_count(), 3);
    }

    #[test]
    fn test_pan_result_structure() {
        let result = PanResult {
            left: 0.5,
            right: 0.3,
        };
        assert!((result.left - 0.5).abs() < f64::EPSILON);
        assert!((result.right - 0.3).abs() < f64::EPSILON);
    }
}
