//! AI scene description generation using rule-based analysis.
//!
//! Derives a structured [`SceneDescription`] from low-level perceptual
//! features (brightness, colour temperature, motion, saturation, contrast)
//! without requiring any ML inference at runtime.  The rules are calibrated
//! against empirical distributions of broadcast and UGC content.

#![allow(dead_code)]

// ─── TimeOfDay ───────────────────────────────────────────────────────────────

/// Estimated time of day inferred from frame luminance and colour temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeOfDay {
    /// Bright, daylight illumination.
    Day,
    /// Dark scene with low overall luminance.
    Night,
    /// Warm, low-angle light consistent with dawn.
    Sunrise,
    /// Warm, low-angle light consistent with dusk.
    Sunset,
    /// Artificial interior lighting detected.
    Indoor,
}

impl TimeOfDay {
    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Night => "night",
            Self::Sunrise => "sunrise",
            Self::Sunset => "sunset",
            Self::Indoor => "indoor",
        }
    }
}

// ─── Mood ────────────────────────────────────────────────────────────────────

/// Affective mood inferred from visual features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mood {
    /// Bright, high-saturation, positive.
    Happy,
    /// Dark, desaturated, low-contrast.
    Sad,
    /// High-contrast, fast motion, high luminance variance.
    Tense,
    /// Low motion, soft colours, moderate brightness.
    Peaceful,
    /// High motion, high saturation, dynamic.
    Energetic,
}

impl Mood {
    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Happy => "happy",
            Self::Sad => "sad",
            Self::Tense => "tense",
            Self::Peaceful => "peaceful",
            Self::Energetic => "energetic",
        }
    }
}

// ─── FrameFeatures ───────────────────────────────────────────────────────────

/// Low-level perceptual features extracted from a video frame.
///
/// All values are normalised to `[0.0, 1.0]` unless documented otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameFeatures {
    /// Mean luminance in [0.0, 1.0], where 0 = black and 1 = white.
    pub brightness: f32,
    /// Estimated colour temperature in Kelvin.
    ///
    /// Typical ranges:
    /// - ~2700 K — tungsten / warm indoor
    /// - ~5500 K — daylight
    /// - ~7000 K — overcast / shade
    pub color_temperature: f32,
    /// Frame-to-frame motion magnitude in [0.0, 1.0], where 1 = extreme motion.
    pub motion_magnitude: f32,
    /// Mean saturation in [0.0, 1.0].
    pub saturation: f32,
    /// RMS contrast in [0.0, 1.0].
    pub contrast: f32,
}

impl FrameFeatures {
    /// Construct a `FrameFeatures` struct with explicit values.
    #[must_use]
    pub const fn new(
        brightness: f32,
        color_temperature: f32,
        motion_magnitude: f32,
        saturation: f32,
        contrast: f32,
    ) -> Self {
        Self {
            brightness,
            color_temperature,
            motion_magnitude,
            saturation,
            contrast,
        }
    }

    /// Returns `true` if the brightness is typical of night / dark scenes.
    #[must_use]
    pub fn is_dark(&self) -> bool {
        self.brightness < 0.25
    }

    /// Returns `true` if the brightness is typical of well-lit day scenes.
    #[must_use]
    pub fn is_bright(&self) -> bool {
        self.brightness > 0.6
    }

    /// Returns `true` if the colour temperature indicates warm light (< 4000 K).
    #[must_use]
    pub fn is_warm(&self) -> bool {
        self.color_temperature < 4_000.0
    }

    /// Returns `true` if the colour temperature is typical of daylight (≥ 5000 K).
    #[must_use]
    pub fn is_daylight(&self) -> bool {
        self.color_temperature >= 5_000.0
    }

    /// Returns `true` if motion is above the "busy" threshold.
    #[must_use]
    pub fn has_high_motion(&self) -> bool {
        self.motion_magnitude > 0.55
    }
}

// ─── SceneDescription ────────────────────────────────────────────────────────

/// Structured description of a video scene.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneDescription {
    /// Short textual description of the apparent filming location.
    pub location: String,
    /// Estimated time of day.
    pub time_of_day: TimeOfDay,
    /// Inferred affective mood.
    pub mood: Mood,
    /// List of inferred activities or elements observed in the scene.
    pub activity: Vec<String>,
}

impl SceneDescription {
    /// Construct a `SceneDescription` explicitly.
    #[must_use]
    pub fn new(
        location: impl Into<String>,
        time_of_day: TimeOfDay,
        mood: Mood,
        activity: Vec<String>,
    ) -> Self {
        Self {
            location: location.into(),
            time_of_day,
            mood,
            activity,
        }
    }
}

// ─── SceneDescriptionGenerator ───────────────────────────────────────────────

/// Rule-based scene description generator.
///
/// All decisions are deterministic and calibrated against typical broadcast
/// content distributions.  The rules form a decision tree:
///
/// 1. **TimeOfDay** — derived primarily from `brightness` and
///    `color_temperature`.
/// 2. **Location** — derived from `TimeOfDay` and secondary cues.
/// 3. **Mood** — derived from `saturation`, `contrast`, and `motion_magnitude`.
/// 4. **Activities** — assembled from all four features.
pub struct SceneDescriptionGenerator;

impl SceneDescriptionGenerator {
    /// Analyse `frame_features` and return a [`SceneDescription`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use oximedia_auto::scene_description::{
    ///     FrameFeatures, SceneDescriptionGenerator, TimeOfDay, Mood,
    /// };
    ///
    /// let f = FrameFeatures::new(0.8, 6000.0, 0.1, 0.7, 0.4);
    /// let desc = SceneDescriptionGenerator::analyze(&f);
    /// assert_eq!(desc.time_of_day, TimeOfDay::Day);
    /// ```
    #[must_use]
    pub fn analyze(features: &FrameFeatures) -> SceneDescription {
        let time_of_day = Self::infer_time_of_day(features);
        let location = Self::infer_location(features, time_of_day);
        let mood = Self::infer_mood(features);
        let activity = Self::infer_activities(features, time_of_day, mood);

        SceneDescription {
            location,
            time_of_day,
            mood,
            activity,
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn infer_time_of_day(f: &FrameFeatures) -> TimeOfDay {
        if f.is_dark() {
            // Very dark: night regardless of colour temperature.
            return TimeOfDay::Night;
        }

        // Indoor: moderate brightness + warm light + low contrast + low-moderate saturation.
        // Typically tungsten/halogen lighting (≤ 3200 K), flat lighting, no sky colour cast.
        if f.is_warm()
            && f.color_temperature < 3_300.0
            && f.brightness > 0.25
            && f.brightness < 0.75
            && f.contrast < 0.35
        {
            return TimeOfDay::Indoor;
        }

        // Sunrise / Sunset: warm light + moderate brightness (0.3–0.65).
        // Higher colour temperature than typical indoor (3300–4000 K) or
        // higher saturation characteristic of golden-hour sky colours.
        if f.is_warm() && f.brightness >= 0.3 && f.brightness <= 0.65 {
            // Distinguish sunrise vs. sunset by saturation:
            // sunsets tend to be more saturated (fiery) vs. softer sunrise.
            if f.saturation > 0.5 {
                return TimeOfDay::Sunset;
            }
            return TimeOfDay::Sunrise;
        }

        // Bright daylight: high brightness + daylight or cool colour temperature.
        if f.is_bright() && f.color_temperature >= 4_500.0 {
            return TimeOfDay::Day;
        }

        // Default to Day if sufficiently bright.
        if f.brightness >= 0.5 {
            TimeOfDay::Day
        } else {
            TimeOfDay::Night
        }
    }

    fn infer_location(f: &FrameFeatures, tod: TimeOfDay) -> String {
        match tod {
            TimeOfDay::Indoor => "indoor".to_owned(),
            TimeOfDay::Night => {
                if f.saturation > 0.4 {
                    "urban nightscape".to_owned()
                } else {
                    "dark exterior".to_owned()
                }
            }
            TimeOfDay::Sunrise => "outdoor – dawn".to_owned(),
            TimeOfDay::Sunset => "outdoor – dusk".to_owned(),
            TimeOfDay::Day => {
                if f.saturation > 0.55 && f.brightness > 0.65 {
                    "outdoor – sunny".to_owned()
                } else if f.saturation < 0.3 {
                    "outdoor – overcast".to_owned()
                } else {
                    "outdoor – daylight".to_owned()
                }
            }
        }
    }

    fn infer_mood(f: &FrameFeatures) -> Mood {
        // Tense: high contrast AND (fast motion OR very low/high brightness).
        if f.contrast > 0.65 && (f.has_high_motion() || f.brightness < 0.2 || f.brightness > 0.9) {
            return Mood::Tense;
        }

        // Energetic: high motion + high saturation.
        if f.has_high_motion() && f.saturation > 0.5 {
            return Mood::Energetic;
        }

        // Sad: dark + desaturated + low motion.
        if f.is_dark() && f.saturation < 0.3 && f.motion_magnitude < 0.3 {
            return Mood::Sad;
        }

        // Happy: bright + high saturation — check before Peaceful so that
        // bright, vibrant, low-motion scenes are correctly classed as Happy.
        if f.is_bright() && f.saturation > 0.5 {
            return Mood::Happy;
        }

        // Peaceful: low motion + moderate saturation + moderate brightness.
        if f.motion_magnitude < 0.25 && f.saturation > 0.2 && f.brightness > 0.3 {
            return Mood::Peaceful;
        }

        // Default.
        Mood::Peaceful
    }

    fn infer_activities(f: &FrameFeatures, tod: TimeOfDay, mood: Mood) -> Vec<String> {
        let mut activities: Vec<String> = Vec::new();

        if f.has_high_motion() {
            activities.push("motion detected".to_owned());
        }

        match mood {
            Mood::Energetic => activities.push("dynamic action".to_owned()),
            Mood::Tense => activities.push("high-tension moment".to_owned()),
            Mood::Happy => activities.push("upbeat activity".to_owned()),
            Mood::Sad => activities.push("subdued scene".to_owned()),
            Mood::Peaceful => activities.push("calm scene".to_owned()),
        }

        match tod {
            TimeOfDay::Sunrise => activities.push("morning light".to_owned()),
            TimeOfDay::Sunset => activities.push("golden hour".to_owned()),
            TimeOfDay::Night => activities.push("night scene".to_owned()),
            TimeOfDay::Indoor => activities.push("interior scene".to_owned()),
            TimeOfDay::Day => {}
        }

        if f.saturation > 0.7 {
            activities.push("vibrant colours".to_owned());
        } else if f.saturation < 0.2 {
            activities.push("muted palette".to_owned());
        }

        if f.contrast > 0.7 {
            activities.push("high-contrast composition".to_owned());
        }

        activities
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TimeOfDay label ───────────────────────────────────────────────────────

    #[test]
    fn time_of_day_labels() {
        assert_eq!(TimeOfDay::Day.label(), "day");
        assert_eq!(TimeOfDay::Night.label(), "night");
        assert_eq!(TimeOfDay::Sunrise.label(), "sunrise");
        assert_eq!(TimeOfDay::Sunset.label(), "sunset");
        assert_eq!(TimeOfDay::Indoor.label(), "indoor");
    }

    // ── Mood label ────────────────────────────────────────────────────────────

    #[test]
    fn mood_labels() {
        assert_eq!(Mood::Happy.label(), "happy");
        assert_eq!(Mood::Sad.label(), "sad");
        assert_eq!(Mood::Tense.label(), "tense");
        assert_eq!(Mood::Peaceful.label(), "peaceful");
        assert_eq!(Mood::Energetic.label(), "energetic");
    }

    // ── FrameFeatures helpers ────────────────────────────────────────────────

    #[test]
    fn features_is_dark() {
        let f = FrameFeatures::new(0.1, 5500.0, 0.0, 0.5, 0.3);
        assert!(f.is_dark());
    }

    #[test]
    fn features_is_bright() {
        let f = FrameFeatures::new(0.85, 5500.0, 0.0, 0.5, 0.3);
        assert!(f.is_bright());
    }

    #[test]
    fn features_is_warm() {
        let f = FrameFeatures::new(0.5, 3200.0, 0.2, 0.4, 0.3);
        assert!(f.is_warm());
    }

    #[test]
    fn features_is_daylight() {
        let f = FrameFeatures::new(0.7, 5500.0, 0.1, 0.6, 0.4);
        assert!(f.is_daylight());
    }

    #[test]
    fn features_has_high_motion() {
        let f = FrameFeatures::new(0.5, 5500.0, 0.8, 0.5, 0.4);
        assert!(f.has_high_motion());
    }

    // ── TimeOfDay inference ───────────────────────────────────────────────────

    #[test]
    fn infer_night() {
        let f = FrameFeatures::new(0.1, 3200.0, 0.1, 0.2, 0.2);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.time_of_day, TimeOfDay::Night);
    }

    #[test]
    fn infer_day_bright_daylight() {
        let f = FrameFeatures::new(0.8, 5500.0, 0.1, 0.6, 0.4);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.time_of_day, TimeOfDay::Day);
    }

    #[test]
    fn infer_indoor_warm_moderate() {
        // Warm (3000 K), moderate brightness, low contrast → Indoor
        let f = FrameFeatures::new(0.5, 3000.0, 0.1, 0.4, 0.3);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.time_of_day, TimeOfDay::Indoor);
    }

    #[test]
    fn infer_sunset() {
        // Warm, moderate brightness, high saturation
        let f = FrameFeatures::new(0.5, 3500.0, 0.05, 0.7, 0.4);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.time_of_day, TimeOfDay::Sunset);
    }

    #[test]
    fn infer_sunrise() {
        // Warm, moderate brightness, low saturation
        let f = FrameFeatures::new(0.45, 3500.0, 0.05, 0.3, 0.35);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.time_of_day, TimeOfDay::Sunrise);
    }

    // ── Mood inference ────────────────────────────────────────────────────────

    #[test]
    fn infer_mood_energetic() {
        let f = FrameFeatures::new(0.6, 5500.0, 0.8, 0.7, 0.5);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.mood, Mood::Energetic);
    }

    #[test]
    fn infer_mood_sad() {
        let f = FrameFeatures::new(0.15, 4500.0, 0.1, 0.15, 0.2);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.mood, Mood::Sad);
    }

    #[test]
    fn infer_mood_peaceful() {
        let f = FrameFeatures::new(0.55, 5500.0, 0.1, 0.4, 0.3);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.mood, Mood::Peaceful);
    }

    #[test]
    fn infer_mood_happy() {
        let f = FrameFeatures::new(0.85, 6000.0, 0.2, 0.75, 0.35);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.mood, Mood::Happy);
    }

    #[test]
    fn infer_mood_tense_high_contrast_fast_motion() {
        let f = FrameFeatures::new(0.6, 5000.0, 0.75, 0.4, 0.8);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert_eq!(desc.mood, Mood::Tense);
    }

    // ── Activity list ─────────────────────────────────────────────────────────

    #[test]
    fn activity_list_not_empty() {
        let f = FrameFeatures::new(0.6, 5500.0, 0.1, 0.5, 0.3);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert!(!desc.activity.is_empty());
    }

    #[test]
    fn activity_motion_detected_when_high_motion() {
        let f = FrameFeatures::new(0.6, 5500.0, 0.9, 0.5, 0.3);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert!(
            desc.activity.iter().any(|a| a.contains("motion")),
            "Expected 'motion detected' in activities: {:?}",
            desc.activity
        );
    }

    #[test]
    fn activity_vibrant_colours_when_high_saturation() {
        let f = FrameFeatures::new(0.7, 6000.0, 0.1, 0.85, 0.3);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert!(
            desc.activity.iter().any(|a| a.contains("vibrant")),
            "Expected 'vibrant colours' in activities: {:?}",
            desc.activity
        );
    }

    #[test]
    fn activity_muted_palette_when_low_saturation() {
        let f = FrameFeatures::new(0.5, 5500.0, 0.1, 0.05, 0.3);
        let desc = SceneDescriptionGenerator::analyze(&f);
        assert!(
            desc.activity.iter().any(|a| a.contains("muted")),
            "Expected 'muted palette' in activities: {:?}",
            desc.activity
        );
    }

    // ── SceneDescription construction ────────────────────────────────────────

    #[test]
    fn scene_description_construction() {
        let desc = SceneDescription::new(
            "studio",
            TimeOfDay::Indoor,
            Mood::Peaceful,
            vec!["recording".to_owned()],
        );
        assert_eq!(desc.location, "studio");
        assert_eq!(desc.time_of_day, TimeOfDay::Indoor);
        assert_eq!(desc.mood, Mood::Peaceful);
        assert_eq!(desc.activity, vec!["recording".to_owned()]);
    }
}
