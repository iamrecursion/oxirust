//! Shot log / shooting log.
//!
//! Records per-shot production metadata: setup details, lens data,
//! lighting notes, and camera settings.  This module is intentionally
//! self-contained and depends only on the standard library so that it
//! can be serialised and shared without pulling in heavy video-frame
//! processing.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::fmt;

// ──────────────────────────────────────────────────────────────────────────────
// Camera settings
// ──────────────────────────────────────────────────────────────────────────────

/// ISO sensitivity setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct IsoValue(pub u32);

impl fmt::Display for IsoValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ISO {}", self.0)
    }
}

/// Shutter speed represented as a fraction (numerator / denominator seconds).
#[derive(Debug, Clone, Copy)]
pub struct ShutterSpeed {
    /// Numerator.
    pub numerator: u32,
    /// Denominator.
    pub denominator: u32,
}

impl ShutterSpeed {
    /// Create a shutter speed.
    #[must_use]
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Return the shutter speed as a floating-point number of seconds.
    #[must_use]
    pub fn as_secs_f64(&self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator.max(1))
    }

    /// Return `true` if this shutter speed satisfies the 180° rule
    /// for the given frames-per-second rate (shutter ≈ 1 / (2 × fps)).
    #[must_use]
    pub fn follows_180_rule(&self, fps: f64) -> bool {
        let target = 1.0 / (2.0 * fps);
        let actual = self.as_secs_f64();
        (actual - target).abs() / target < 0.05 // within 5 %
    }
}

impl fmt::Display for ShutterSpeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.numerator == 1 {
            write!(f, "1/{}", self.denominator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

/// Aperture (f-number).
#[derive(Debug, Clone, Copy)]
pub struct Aperture(pub f32);

impl Aperture {
    /// Return `true` if this aperture is considered a shallow depth-of-field
    /// setting (f/2.8 or wider).
    #[must_use]
    pub fn is_shallow_dof(&self) -> bool {
        self.0 <= 2.8
    }

    /// Return `true` if this aperture falls in the "sweet spot" for sharpness
    /// (typically f/5.6 – f/11).
    #[must_use]
    pub fn is_sweet_spot(&self) -> bool {
        self.0 >= 5.6 && self.0 <= 11.0
    }
}

impl fmt::Display for Aperture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f/{:.1}", self.0)
    }
}

/// White balance preset or custom Kelvin value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhiteBalance {
    /// Automatic white balance.
    Auto,
    /// Daylight (~5600 K).
    Daylight,
    /// Tungsten (~3200 K).
    Tungsten,
    /// Fluorescent (~4000 K).
    Fluorescent,
    /// Custom Kelvin value.
    Custom(u32),
}

impl WhiteBalance {
    /// Return the approximate colour temperature in Kelvin.
    #[must_use]
    pub fn kelvin(&self) -> Option<u32> {
        match self {
            Self::Daylight => Some(5600),
            Self::Tungsten => Some(3200),
            Self::Fluorescent => Some(4000),
            Self::Custom(k) => Some(*k),
            Self::Auto => None,
        }
    }
}

/// Camera settings for a single shot.
#[derive(Debug, Clone)]
pub struct CameraSettings {
    /// ISO value.
    pub iso: IsoValue,
    /// Shutter speed.
    pub shutter: ShutterSpeed,
    /// Aperture (f-number).
    pub aperture: Aperture,
    /// White balance.
    pub white_balance: WhiteBalance,
    /// Frames per second.
    pub fps: f64,
}

impl CameraSettings {
    /// Create default 24 fps cinema settings.
    #[must_use]
    pub fn cinema_default() -> Self {
        Self {
            iso: IsoValue(800),
            shutter: ShutterSpeed::new(1, 48), // 180° rule at 24 fps
            aperture: Aperture(2.8),
            white_balance: WhiteBalance::Daylight,
            fps: 24.0,
        }
    }

    /// Return `true` if the shutter speed follows the 180° rule for
    /// the stored fps value.
    #[must_use]
    pub fn shutter_follows_180_rule(&self) -> bool {
        self.shutter.follows_180_rule(self.fps)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Lens data
// ──────────────────────────────────────────────────────────────────────────────

/// Lens category based on focal length (assuming full-frame equivalent).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensCategory {
    /// Ultra-wide (< 24 mm).
    UltraWide,
    /// Wide (24 – 35 mm).
    Wide,
    /// Normal (35 – 70 mm).
    Normal,
    /// Short telephoto (70 – 135 mm).
    ShortTele,
    /// Telephoto (> 135 mm).
    Telephoto,
}

impl LensCategory {
    /// Classify a focal length (mm, full-frame equivalent).
    #[must_use]
    pub fn from_focal_length(mm: f32) -> Self {
        if mm < 24.0 {
            Self::UltraWide
        } else if mm < 35.0 {
            Self::Wide
        } else if mm < 70.0 {
            Self::Normal
        } else if mm < 135.0 {
            Self::ShortTele
        } else {
            Self::Telephoto
        }
    }
}

/// Lens specification for a single shot.
#[derive(Debug, Clone)]
pub struct LensData {
    /// Manufacturer / model name (e.g. "Canon CN-E 50mm T1.3").
    pub name: String,
    /// Focal length in mm (full-frame equivalent).
    pub focal_length_mm: f32,
    /// Maximum aperture (T-stop or f-stop).
    pub max_aperture: f32,
    /// Serial number or identifier.
    pub serial: Option<String>,
    /// Focus distance in metres.
    pub focus_distance_m: Option<f32>,
}

impl LensData {
    /// Return the lens category.
    #[must_use]
    pub fn category(&self) -> LensCategory {
        LensCategory::from_focal_length(self.focal_length_mm)
    }

    /// Return `true` if this is a prime lens (no zoom range stored).
    #[must_use]
    pub fn is_prime(&self) -> bool {
        !self.name.to_lowercase().contains("zoom") && !self.name.contains('-')
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Lighting notes
// ──────────────────────────────────────────────────────────────────────────────

/// Lighting setup type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightingSetup {
    /// Classic three-point (key, fill, back).
    ThreePoint,
    /// Single key light only.
    SingleKey,
    /// Natural / available light.
    Natural,
    /// High-key (bright, low contrast).
    HighKey,
    /// Low-key (dark, high contrast).
    LowKey,
    /// Rembrandt-style (45° key, triangle shadow).
    Rembrandt,
    /// Custom or unspecified.
    Custom,
}

/// A single lighting note entry.
#[derive(Debug, Clone)]
pub struct LightingNote {
    /// Lighting setup used.
    pub setup: LightingSetup,
    /// Free-text description.
    pub notes: String,
    /// Colour temperature in Kelvin, if measured.
    pub colour_temp_k: Option<u32>,
}

impl LightingNote {
    /// Create a basic lighting note.
    #[must_use]
    pub fn new(setup: LightingSetup, notes: impl Into<String>) -> Self {
        Self {
            setup,
            notes: notes.into(),
            colour_temp_k: None,
        }
    }

    /// Attach a measured colour temperature.
    #[must_use]
    pub fn with_colour_temp(mut self, k: u32) -> Self {
        self.colour_temp_k = Some(k);
        self
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shot setup details
// ──────────────────────────────────────────────────────────────────────────────

/// Production location type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocationType {
    /// Interior (studio or practical location).
    Interior,
    /// Exterior.
    Exterior,
    /// Mixed (e.g. doorway shot).
    Mixed,
}

/// A single setup / take entry in the shooting log.
#[derive(Debug, Clone)]
pub struct ShotSetup {
    /// Scene number (e.g. "12A").
    pub scene: String,
    /// Shot number within the scene.
    pub shot_number: u32,
    /// Take number.
    pub take: u32,
    /// Whether this take is marked as a "circle" (selected) take.
    pub circled: bool,
    /// Location type.
    pub location: LocationType,
    /// Description of the shot.
    pub description: String,
    /// Camera settings.
    pub camera: CameraSettings,
    /// Lens used.
    pub lens: LensData,
    /// Lighting notes.
    pub lighting: Vec<LightingNote>,
    /// Free-text director / operator notes.
    pub notes: String,
}

impl ShotSetup {
    /// Create a minimal shot setup entry.
    #[must_use]
    pub fn new(
        scene: impl Into<String>,
        shot_number: u32,
        take: u32,
        description: impl Into<String>,
    ) -> Self {
        Self {
            scene: scene.into(),
            shot_number,
            take,
            circled: false,
            location: LocationType::Interior,
            description: description.into(),
            camera: CameraSettings::cinema_default(),
            lens: LensData {
                name: String::from("Unknown"),
                focal_length_mm: 50.0,
                max_aperture: 2.0,
                serial: None,
                focus_distance_m: None,
            },
            lighting: Vec::new(),
            notes: String::new(),
        }
    }

    /// Mark this take as a circled (selected) take.
    #[must_use]
    pub fn circle(mut self) -> Self {
        self.circled = true;
        self
    }

    /// Add a lighting note.
    pub fn add_lighting(&mut self, note: LightingNote) {
        self.lighting.push(note);
    }

    /// Return a short identifier string, e.g. `"12A-3T2"`.
    #[must_use]
    pub fn id(&self) -> String {
        format!("{}-{}T{}", self.scene, self.shot_number, self.take)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shooting log
// ──────────────────────────────────────────────────────────────────────────────

/// The complete shooting log for a production day / project.
#[derive(Debug, Clone, Default)]
pub struct ShootingLog {
    entries: Vec<ShotSetup>,
}

impl ShootingLog {
    /// Create an empty shooting log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a setup entry.
    pub fn add(&mut self, setup: ShotSetup) {
        self.entries.push(setup);
    }

    /// Return the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all circled (selected) takes.
    #[must_use]
    pub fn circled_takes(&self) -> Vec<&ShotSetup> {
        self.entries.iter().filter(|e| e.circled).collect()
    }

    /// Return all entries for a given scene identifier.
    #[must_use]
    pub fn scene_entries(&self, scene: &str) -> Vec<&ShotSetup> {
        self.entries.iter().filter(|e| e.scene == scene).collect()
    }

    /// Return the total number of takes across all scenes.
    #[must_use]
    pub fn total_takes(&self) -> usize {
        self.entries.len()
    }

    /// Return the maximum take number recorded for a given scene + shot.
    #[must_use]
    pub fn max_take(&self, scene: &str, shot_number: u32) -> u32 {
        self.entries
            .iter()
            .filter(|e| e.scene == scene && e.shot_number == shot_number)
            .map(|e| e.take)
            .max()
            .unwrap_or(0)
    }

    /// Return an iterator over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &ShotSetup> {
        self.entries.iter()
    }

    /// Return entries where the shutter does NOT follow the 180° rule.
    #[must_use]
    pub fn shutter_violations(&self) -> Vec<&ShotSetup> {
        self.entries
            .iter()
            .filter(|e| !e.camera.shutter_follows_180_rule())
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iso_display() {
        assert_eq!(IsoValue(800).to_string(), "ISO 800");
    }

    #[test]
    fn test_shutter_display_unit_numerator() {
        let s = ShutterSpeed::new(1, 50);
        assert_eq!(s.to_string(), "1/50");
    }

    #[test]
    fn test_shutter_as_secs_f64() {
        let s = ShutterSpeed::new(1, 100);
        assert!((s.as_secs_f64() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_shutter_follows_180_rule_24fps() {
        // 1/48 is exact 180° for 24 fps.
        let s = ShutterSpeed::new(1, 48);
        assert!(s.follows_180_rule(24.0));
    }

    #[test]
    fn test_shutter_violates_180_rule() {
        // 1/250 is far from 1/48.
        let s = ShutterSpeed::new(1, 250);
        assert!(!s.follows_180_rule(24.0));
    }

    #[test]
    fn test_aperture_shallow_dof() {
        assert!(Aperture(1.4).is_shallow_dof());
        assert!(!Aperture(8.0).is_shallow_dof());
    }

    #[test]
    fn test_aperture_sweet_spot() {
        assert!(Aperture(8.0).is_sweet_spot());
        assert!(!Aperture(1.4).is_sweet_spot());
    }

    #[test]
    fn test_white_balance_kelvin() {
        assert_eq!(WhiteBalance::Daylight.kelvin(), Some(5600));
        assert_eq!(WhiteBalance::Tungsten.kelvin(), Some(3200));
        assert!(WhiteBalance::Auto.kelvin().is_none());
    }

    #[test]
    fn test_lens_category() {
        assert_eq!(
            LensCategory::from_focal_length(14.0),
            LensCategory::UltraWide
        );
        assert_eq!(LensCategory::from_focal_length(28.0), LensCategory::Wide);
        assert_eq!(LensCategory::from_focal_length(50.0), LensCategory::Normal);
        assert_eq!(
            LensCategory::from_focal_length(85.0),
            LensCategory::ShortTele
        );
        assert_eq!(
            LensCategory::from_focal_length(200.0),
            LensCategory::Telephoto
        );
    }

    #[test]
    fn test_lens_is_prime() {
        let lens = LensData {
            name: String::from("Canon 50mm prime"),
            focal_length_mm: 50.0,
            max_aperture: 1.4,
            serial: None,
            focus_distance_m: None,
        };
        assert!(lens.is_prime());
    }

    #[test]
    fn test_shot_setup_id() {
        let s = ShotSetup::new("12A", 3, 2, "Wide establishing");
        assert_eq!(s.id(), "12A-3T2");
    }

    #[test]
    fn test_shot_setup_circle() {
        let s = ShotSetup::new("1", 1, 1, "Test").circle();
        assert!(s.circled);
    }

    #[test]
    fn test_shooting_log_empty() {
        let log = ShootingLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_shooting_log_add_and_len() {
        let mut log = ShootingLog::new();
        log.add(ShotSetup::new("1", 1, 1, "Shot A"));
        log.add(ShotSetup::new("1", 1, 2, "Shot A take 2"));
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_circled_takes() {
        let mut log = ShootingLog::new();
        log.add(ShotSetup::new("1", 1, 1, "Take 1"));
        log.add(ShotSetup::new("1", 1, 2, "Take 2").circle());
        let circled = log.circled_takes();
        assert_eq!(circled.len(), 1);
        assert_eq!(circled[0].take, 2);
    }

    #[test]
    fn test_scene_entries_filter() {
        let mut log = ShootingLog::new();
        log.add(ShotSetup::new("1", 1, 1, "Scene 1"));
        log.add(ShotSetup::new("2", 1, 1, "Scene 2"));
        assert_eq!(log.scene_entries("1").len(), 1);
        assert_eq!(log.scene_entries("2").len(), 1);
        assert_eq!(log.scene_entries("99").len(), 0);
    }

    #[test]
    fn test_max_take() {
        let mut log = ShootingLog::new();
        log.add(ShotSetup::new("5", 2, 1, "T1"));
        log.add(ShotSetup::new("5", 2, 2, "T2"));
        log.add(ShotSetup::new("5", 2, 3, "T3"));
        assert_eq!(log.max_take("5", 2), 3);
        assert_eq!(log.max_take("5", 99), 0);
    }

    #[test]
    fn test_camera_cinema_default_shutter_180_rule() {
        let cam = CameraSettings::cinema_default();
        assert!(cam.shutter_follows_180_rule());
    }

    #[test]
    fn test_lighting_note_with_colour_temp() {
        let note =
            LightingNote::new(LightingSetup::ThreePoint, "Standard setup").with_colour_temp(5600);
        assert_eq!(note.colour_temp_k, Some(5600));
    }

    #[test]
    fn test_shutter_violations_none_for_valid_log() {
        let mut log = ShootingLog::new();
        log.add(ShotSetup::new("1", 1, 1, "Valid"));
        // Default camera follows 180° rule
        assert_eq!(log.shutter_violations().len(), 0);
    }
}
