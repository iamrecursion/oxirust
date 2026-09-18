//! Downstream keyer (DSK) management for production switcher.
//!
//! Downstream keyers are applied after the M/E rows and before the final
//! program output, used for persistent graphics such as bugs, channel logos,
//! and lower-thirds that must always appear on top of program video.

#![allow(dead_code)]

/// Source type for a downstream keyer.
#[derive(Debug, Clone, PartialEq)]
pub enum DskSource {
    /// External hardware input
    ExternalInput,
    /// Internal media player
    MediaPlayer,
    /// Internal color generator (bars, matte, etc.)
    ColorGenerator,
    /// Test pattern generator
    TestPattern,
}

impl DskSource {
    /// Returns true if this source is an internal media player.
    pub fn is_media_player(&self) -> bool {
        matches!(self, Self::MediaPlayer)
    }
}

/// Configuration settings for a downstream keyer channel.
#[derive(Debug, Clone)]
pub struct DskSettings {
    /// The source providing the keyed image
    pub source: DskSource,
    /// Fill input index
    pub fill_input: u8,
    /// Key (mask) input index
    pub key_input: u8,
    /// Key gain factor (0.0 = transparent, 1.0 = fully opaque)
    pub gain: f32,
    /// Whether to invert the key signal
    pub invert_key: bool,
}

impl DskSettings {
    /// Return a default DSK settings configuration.
    pub fn default_dsk() -> Self {
        Self {
            source: DskSource::ExternalInput,
            fill_input: 0,
            key_input: 1,
            gain: 1.0,
            invert_key: false,
        }
    }
}

impl Default for DskSettings {
    fn default() -> Self {
        Self::default_dsk()
    }
}

/// A single downstream keyer channel.
#[derive(Debug, Clone)]
pub struct DskChannel {
    /// Channel identifier
    pub id: u8,
    /// Current keyer settings
    pub settings: DskSettings,
    /// Whether this DSK is currently on air (applied to program output)
    pub on_air: bool,
    /// Whether this DSK is visible on the preview bus
    pub preview_enabled: bool,
}

impl DskChannel {
    /// Create a new DSK channel with default settings.
    pub fn new(id: u8) -> Self {
        Self {
            id,
            settings: DskSettings::default_dsk(),
            on_air: false,
            preview_enabled: false,
        }
    }

    /// Returns true if this DSK is currently live (on air).
    pub fn is_live(&self) -> bool {
        self.on_air
    }

    /// Apply new settings to this channel.
    pub fn apply_settings(&mut self, settings: DskSettings) {
        self.settings = settings;
    }
}

/// Controller managing all downstream keyer channels.
#[derive(Debug, Clone)]
pub struct DskController {
    /// All DSK channels managed by this controller
    pub channels: Vec<DskChannel>,
}

impl DskController {
    /// Create a new DSK controller with the given number of channels.
    pub fn new(num_channels: u8) -> Self {
        let channels = (0..num_channels).map(DskChannel::new).collect();
        Self { channels }
    }

    /// Set the on-air state for a channel identified by `id`.
    ///
    /// Does nothing if the channel ID does not exist.
    pub fn set_on_air(&mut self, id: u8, on: bool) {
        if let Some(ch) = self.channels.iter_mut().find(|c| c.id == id) {
            ch.on_air = on;
        }
    }

    /// Set the preview-enabled state for a channel identified by `id`.
    ///
    /// Does nothing if the channel ID does not exist.
    pub fn set_preview(&mut self, id: u8, en: bool) {
        if let Some(ch) = self.channels.iter_mut().find(|c| c.id == id) {
            ch.preview_enabled = en;
        }
    }

    /// Return the IDs of all channels currently on air.
    pub fn live_channels(&self) -> Vec<u8> {
        self.channels
            .iter()
            .filter(|c| c.on_air)
            .map(|c| c.id)
            .collect()
    }

    /// Return the total number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

// ---------------------------------------------------------------------------
// New types: KeyType, DownstreamKey, ChromaKeyConfig, DskStack
// ---------------------------------------------------------------------------

/// Keying method for a downstream keyer.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyType {
    /// Luma key: uses brightness to determine transparency.
    Luma,
    /// Chroma key: uses colour to determine transparency.
    Chroma,
    /// Linear key: uses a separate alpha/key signal.
    Linear,
    /// Pattern key: uses a geometric pattern as the mask.
    Pattern,
}

impl KeyType {
    /// Returns `true` for key types that require a separate alpha/key signal.
    #[must_use]
    pub fn requires_alpha(&self) -> bool {
        matches!(self, Self::Linear)
    }
}

/// A downstream keyer with clip, gain, and invert controls.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DownstreamKey {
    /// Keyer identifier.
    pub id: u8,
    /// Keying method.
    pub key_type: KeyType,
    /// Whether the keyer is currently on air.
    pub on_air: bool,
    /// Clip level in [0.0, 1.0]: signals below this are treated as transparent.
    pub clip: f32,
    /// Gain multiplier (1.0 = no gain).
    pub gain: f32,
    /// Invert the key signal before compositing.
    pub invert: bool,
}

impl DownstreamKey {
    /// Process a luma key signal byte (0–255), applying clip, gain, and invert
    /// to produce an 8-bit alpha value.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn process_luma_key(&self, key_signal: u8) -> u8 {
        let v = key_signal as f32 / 255.0;
        // Clip: values below `clip` map to 0, above map to 1 linearly.
        let clipped = if v < self.clip {
            0.0f32
        } else {
            ((v - self.clip) * self.gain).clamp(0.0, 1.0)
        };
        let result = if self.invert { 1.0 - clipped } else { clipped };
        (result * 255.0).round() as u8
    }

    /// Returns `true` if this keyer is on air and thus composited onto program.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.on_air
    }
}

/// Configuration for a chroma keyer.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ChromaKeyConfig {
    /// Centre hue to key out, in degrees [0.0, 360.0).
    pub hue_center: f32,
    /// Hue tolerance in degrees; hues within this range are keyed.
    pub hue_tolerance: f32,
    /// Minimum saturation for a pixel to be keyed (avoids keying grey pixels).
    pub saturation_min: f32,
    /// Spill suppression strength in [0.0, 1.0].
    pub spill_suppress: f32,
}

impl ChromaKeyConfig {
    /// Compute the key value for a pixel given its hue `h` (degrees) and
    /// saturation `s` (0–1).
    ///
    /// Returns:
    /// - `1.0` if the pixel is fully within the key colour (cut/transparent).
    /// - `0.0` if the pixel should be kept.
    /// - Intermediate values for soft edges.
    #[must_use]
    pub fn key_pixel(&self, h: f32, s: f32) -> f32 {
        if s < self.saturation_min {
            return 0.0; // unsaturated → keep
        }
        // Angular distance to hue centre.
        let mut delta = (h - self.hue_center).abs() % 360.0;
        if delta > 180.0 {
            delta = 360.0 - delta;
        }
        if delta >= self.hue_tolerance {
            0.0 // outside tolerance → keep
        } else {
            1.0 - delta / self.hue_tolerance // linear fall-off inside tolerance
        }
    }
}

/// A stack of downstream keyers applied to program output in order.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct DskStack {
    /// All downstream keyers.
    pub keys: Vec<DownstreamKey>,
}

impl DskStack {
    /// Create an empty stack.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns references to only the keyers that are currently on air.
    #[must_use]
    pub fn active_keys(&self) -> Vec<&DownstreamKey> {
        self.keys.iter().filter(|k| k.on_air).collect()
    }

    /// Set the on-air state for the keyer with the given `id`.
    ///
    /// Returns `true` if a matching keyer was found, `false` otherwise.
    pub fn set_on_air(&mut self, id: u8, on_air: bool) -> bool {
        match self.keys.iter_mut().find(|k| k.id == id) {
            Some(k) => {
                k.on_air = on_air;
                true
            }
            None => false,
        }
    }

    /// Total number of keyers in the stack (active or not).
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- DskSource tests ---

    #[test]
    fn test_dsk_source_is_media_player_true() {
        assert!(DskSource::MediaPlayer.is_media_player());
    }

    #[test]
    fn test_dsk_source_is_media_player_false_external() {
        assert!(!DskSource::ExternalInput.is_media_player());
    }

    #[test]
    fn test_dsk_source_is_media_player_false_color() {
        assert!(!DskSource::ColorGenerator.is_media_player());
    }

    #[test]
    fn test_dsk_source_is_media_player_false_test_pattern() {
        assert!(!DskSource::TestPattern.is_media_player());
    }

    // --- DskSettings tests ---

    #[test]
    fn test_dsk_settings_default_source() {
        let s = DskSettings::default_dsk();
        assert_eq!(s.source, DskSource::ExternalInput);
    }

    #[test]
    fn test_dsk_settings_default_gain() {
        let s = DskSettings::default_dsk();
        assert!((s.gain - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dsk_settings_default_no_invert() {
        let s = DskSettings::default_dsk();
        assert!(!s.invert_key);
    }

    // --- DskChannel tests ---

    #[test]
    fn test_channel_starts_off_air() {
        let ch = DskChannel::new(0);
        assert!(!ch.is_live());
    }

    #[test]
    fn test_channel_is_live_after_on_air() {
        let mut ch = DskChannel::new(1);
        ch.on_air = true;
        assert!(ch.is_live());
    }

    #[test]
    fn test_channel_apply_settings_updates_source() {
        let mut ch = DskChannel::new(0);
        let new_settings = DskSettings {
            source: DskSource::MediaPlayer,
            fill_input: 2,
            key_input: 3,
            gain: 0.8,
            invert_key: true,
        };
        ch.apply_settings(new_settings.clone());
        assert_eq!(ch.settings.source, DskSource::MediaPlayer);
        assert!((ch.settings.gain - 0.8).abs() < f32::EPSILON);
        assert!(ch.settings.invert_key);
    }

    // --- DskController tests ---

    #[test]
    fn test_controller_channel_count() {
        let ctrl = DskController::new(4);
        assert_eq!(ctrl.channel_count(), 4);
    }

    #[test]
    fn test_controller_no_live_channels_initially() {
        let ctrl = DskController::new(2);
        assert!(ctrl.live_channels().is_empty());
    }

    #[test]
    fn test_controller_set_on_air() {
        let mut ctrl = DskController::new(2);
        ctrl.set_on_air(0, true);
        let live = ctrl.live_channels();
        assert_eq!(live, vec![0]);
    }

    #[test]
    fn test_controller_set_on_air_invalid_id_no_panic() {
        let mut ctrl = DskController::new(2);
        ctrl.set_on_air(99, true); // should not panic
        assert!(ctrl.live_channels().is_empty());
    }

    #[test]
    fn test_controller_set_preview() {
        let mut ctrl = DskController::new(2);
        ctrl.set_preview(1, true);
        assert!(ctrl.channels[1].preview_enabled);
    }

    #[test]
    fn test_controller_multiple_live_channels() {
        let mut ctrl = DskController::new(4);
        ctrl.set_on_air(0, true);
        ctrl.set_on_air(2, true);
        let mut live = ctrl.live_channels();
        live.sort_unstable();
        assert_eq!(live, vec![0, 2]);
    }

    // --- KeyType tests ---

    #[test]
    fn test_key_type_luma_does_not_require_alpha() {
        assert!(!KeyType::Luma.requires_alpha());
    }

    #[test]
    fn test_key_type_chroma_does_not_require_alpha() {
        assert!(!KeyType::Chroma.requires_alpha());
    }

    #[test]
    fn test_key_type_linear_requires_alpha() {
        assert!(KeyType::Linear.requires_alpha());
    }

    #[test]
    fn test_key_type_pattern_does_not_require_alpha() {
        assert!(!KeyType::Pattern.requires_alpha());
    }

    // --- DownstreamKey::process_luma_key tests ---

    fn make_dsk(clip: f32, gain: f32, invert: bool) -> DownstreamKey {
        DownstreamKey {
            id: 0,
            key_type: KeyType::Luma,
            on_air: true,
            clip,
            gain,
            invert,
        }
    }

    #[test]
    fn test_process_luma_key_below_clip_is_zero() {
        let dsk = make_dsk(0.5, 1.0, false);
        // Signal 100/255 ≈ 0.39 < clip 0.5 → alpha should be 0.
        assert_eq!(dsk.process_luma_key(100), 0);
    }

    #[test]
    fn test_process_luma_key_full_signal_is_max() {
        let dsk = make_dsk(0.0, 1.0, false);
        assert_eq!(dsk.process_luma_key(255), 255);
    }

    #[test]
    fn test_process_luma_key_inverted() {
        let dsk = make_dsk(0.0, 1.0, true);
        // Full signal inverted → alpha 0.
        assert_eq!(dsk.process_luma_key(255), 0);
        // Zero signal inverted → alpha 255.
        assert_eq!(dsk.process_luma_key(0), 255);
    }

    #[test]
    fn test_downstream_key_is_active_when_on_air() {
        let dsk = make_dsk(0.0, 1.0, false);
        assert!(dsk.is_active());
    }

    #[test]
    fn test_downstream_key_is_not_active_when_off_air() {
        let mut dsk = make_dsk(0.0, 1.0, false);
        dsk.on_air = false;
        assert!(!dsk.is_active());
    }

    // --- ChromaKeyConfig::key_pixel tests ---

    #[test]
    fn test_chroma_key_pixel_in_range_returns_one() {
        let cfg = ChromaKeyConfig {
            hue_center: 120.0,
            hue_tolerance: 30.0,
            saturation_min: 0.2,
            spill_suppress: 0.5,
        };
        // Hue exactly at centre with adequate saturation → fully keyed.
        assert!((cfg.key_pixel(120.0, 0.8) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chroma_key_pixel_outside_range_returns_zero() {
        let cfg = ChromaKeyConfig {
            hue_center: 120.0,
            hue_tolerance: 30.0,
            saturation_min: 0.2,
            spill_suppress: 0.5,
        };
        // Hue 200° is 80° away from centre (> tolerance) → kept.
        assert!((cfg.key_pixel(200.0, 0.8) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chroma_key_pixel_low_saturation_returns_zero() {
        let cfg = ChromaKeyConfig {
            hue_center: 120.0,
            hue_tolerance: 30.0,
            saturation_min: 0.5,
            spill_suppress: 0.0,
        };
        // Low saturation → pixel is kept regardless of hue.
        assert!((cfg.key_pixel(120.0, 0.1) - 0.0).abs() < f32::EPSILON);
    }

    // --- DskStack tests ---

    #[test]
    fn test_dsk_stack_empty_key_count() {
        let stack = DskStack::new();
        assert_eq!(stack.key_count(), 0);
    }

    #[test]
    fn test_dsk_stack_active_keys_empty_when_all_off() {
        let mut stack = DskStack::new();
        stack.keys.push(make_dsk(0.0, 1.0, false));
        stack.keys[0].on_air = false;
        assert!(stack.active_keys().is_empty());
    }

    #[test]
    fn test_dsk_stack_active_keys_returns_on_air_keys() {
        let mut stack = DskStack::new();
        stack.keys.push(DownstreamKey {
            id: 0,
            key_type: KeyType::Luma,
            on_air: true,
            clip: 0.0,
            gain: 1.0,
            invert: false,
        });
        stack.keys.push(DownstreamKey {
            id: 1,
            key_type: KeyType::Luma,
            on_air: false,
            clip: 0.0,
            gain: 1.0,
            invert: false,
        });
        assert_eq!(stack.active_keys().len(), 1);
        assert_eq!(stack.active_keys()[0].id, 0);
    }

    #[test]
    fn test_dsk_stack_set_on_air_returns_true_for_found() {
        let mut stack = DskStack::new();
        stack.keys.push(DownstreamKey {
            id: 2,
            key_type: KeyType::Chroma,
            on_air: false,
            clip: 0.0,
            gain: 1.0,
            invert: false,
        });
        assert!(stack.set_on_air(2, true));
        assert!(stack.keys[0].on_air);
    }

    #[test]
    fn test_dsk_stack_set_on_air_returns_false_for_missing() {
        let mut stack = DskStack::new();
        assert!(!stack.set_on_air(99, true));
    }
}
