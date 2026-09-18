//! Override system for applying partial parameter changes on top of a base preset.

#![allow(dead_code)]

use std::collections::HashMap;

/// A named field that can be overridden.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OverrideField {
    /// Video bitrate in bits/s.
    VideoBitrate,
    /// Audio bitrate in bits/s.
    AudioBitrate,
    /// Output frame width.
    Width,
    /// Output frame height.
    Height,
    /// Frames per second (stored as millihertz to stay integer).
    FrameRateMilliHz,
    /// Video codec identifier string.
    VideoCodec,
    /// Audio codec identifier string.
    AudioCodec,
    /// Container format.
    Container,
    /// Constant-rate factor for quality-based encoding.
    Crf,
    /// Encoder preset speed string (e.g. "fast", "slow").
    EncoderPreset,
    /// Arbitrary named parameter.
    Custom(String),
}

impl OverrideField {
    /// Return the canonical field name used in serialization / display.
    #[must_use]
    pub fn field_name(&self) -> String {
        match self {
            Self::VideoBitrate => "video_bitrate".to_string(),
            Self::AudioBitrate => "audio_bitrate".to_string(),
            Self::Width => "width".to_string(),
            Self::Height => "height".to_string(),
            Self::FrameRateMilliHz => "frame_rate_mhz".to_string(),
            Self::VideoCodec => "video_codec".to_string(),
            Self::AudioCodec => "audio_codec".to_string(),
            Self::Container => "container".to_string(),
            Self::Crf => "crf".to_string(),
            Self::EncoderPreset => "encoder_preset".to_string(),
            Self::Custom(name) => name.clone(),
        }
    }

    /// Return `true` if changing this field may break compatibility with the base preset's
    /// delivery target (container or codec changes are considered breaking).
    #[must_use]
    pub fn is_potentially_breaking(&self) -> bool {
        matches!(self, Self::VideoCodec | Self::AudioCodec | Self::Container)
    }
}

/// Value carried by a single override entry.
#[derive(Debug, Clone, PartialEq)]
pub enum OverrideValue {
    /// Integer value (bitrates, dimensions, CRF …).
    Int(i64),
    /// Floating-point value.
    Float(f64),
    /// String value (codec names, container format …).
    Str(String),
    /// Boolean flag.
    Bool(bool),
}

impl OverrideValue {
    /// Try to extract an integer value.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        if let Self::Int(v) = self {
            Some(*v)
        } else {
            None
        }
    }

    /// Try to extract a string value.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        if let Self::Str(v) = self {
            Some(v.as_str())
        } else {
            None
        }
    }
}

/// A single field override: field + new value.
#[derive(Debug, Clone)]
pub struct PresetOverride {
    /// The field being overridden.
    pub field: OverrideField,
    /// The replacement value.
    pub value: OverrideValue,
    /// Human-readable reason for the override.
    pub reason: String,
}

impl PresetOverride {
    /// Create a new override.
    #[must_use]
    pub fn new(field: OverrideField, value: OverrideValue) -> Self {
        Self {
            field,
            value,
            reason: String::new(),
        }
    }

    /// Attach a reason string.
    #[must_use]
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Return `true` if an override is present for the given field (i.e. this struct always
    /// represents a single defined override).
    #[must_use]
    pub fn has_override(&self) -> bool {
        true
    }

    /// Apply this override to a map of base field values, returning the merged map.
    ///
    /// The base map is keyed by field name strings; the returned map adds / replaces
    /// the field controlled by this override.
    #[must_use]
    pub fn apply_to_base(
        &self,
        base: &HashMap<String, OverrideValue>,
    ) -> HashMap<String, OverrideValue> {
        let mut merged = base.clone();
        merged.insert(self.field.field_name(), self.value.clone());
        merged
    }
}

/// A named collection of overrides that can be applied as a unit.
#[derive(Debug, Clone, Default)]
pub struct OverrideSet {
    /// Name of this override set (e.g. "mobile-low-bandwidth").
    pub name: String,
    overrides: Vec<PresetOverride>,
}

impl OverrideSet {
    /// Create an empty override set.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            overrides: Vec::new(),
        }
    }

    /// Add an override to the set. If an override for the same field already exists,
    /// it is replaced.
    pub fn add(&mut self, ovr: PresetOverride) {
        let field_name = ovr.field.field_name();
        // Remove any previous override for the same field
        self.overrides
            .retain(|o| o.field.field_name() != field_name);
        self.overrides.push(ovr);
    }

    /// Remove an override by field, returning it if present.
    pub fn remove(&mut self, field: &OverrideField) -> Option<PresetOverride> {
        let name = field.field_name();
        if let Some(pos) = self
            .overrides
            .iter()
            .position(|o| o.field.field_name() == name)
        {
            Some(self.overrides.remove(pos))
        } else {
            None
        }
    }

    /// Return all currently registered overrides.
    #[must_use]
    pub fn overrides(&self) -> &[PresetOverride] {
        &self.overrides
    }

    /// Number of overrides in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.overrides.len()
    }

    /// Return `true` if no overrides have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }

    /// Apply all overrides in this set to the given base map, returning the merged result.
    /// Later overrides in the set win over earlier ones for the same field.
    #[must_use]
    pub fn merged_values(
        &self,
        base: &HashMap<String, OverrideValue>,
    ) -> HashMap<String, OverrideValue> {
        let mut result = base.clone();
        for ovr in &self.overrides {
            result.insert(ovr.field.field_name(), ovr.value.clone());
        }
        result
    }

    /// Return `true` if the set contains any potentially-breaking overrides.
    #[must_use]
    pub fn has_breaking_overrides(&self) -> bool {
        self.overrides
            .iter()
            .any(|o| o.field.is_potentially_breaking())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- OverrideField ---

    #[test]
    fn test_field_name_video_bitrate() {
        assert_eq!(OverrideField::VideoBitrate.field_name(), "video_bitrate");
    }

    #[test]
    fn test_field_name_custom() {
        let f = OverrideField::Custom("x_param".to_string());
        assert_eq!(f.field_name(), "x_param");
    }

    #[test]
    fn test_field_potentially_breaking_codec() {
        assert!(OverrideField::VideoCodec.is_potentially_breaking());
        assert!(OverrideField::Container.is_potentially_breaking());
    }

    #[test]
    fn test_field_not_breaking_bitrate() {
        assert!(!OverrideField::VideoBitrate.is_potentially_breaking());
        assert!(!OverrideField::Crf.is_potentially_breaking());
    }

    // --- OverrideValue ---

    #[test]
    fn test_override_value_as_int() {
        let v = OverrideValue::Int(5_000_000);
        assert_eq!(v.as_int(), Some(5_000_000));
    }

    #[test]
    fn test_override_value_as_str() {
        let v = OverrideValue::Str("h264".to_string());
        assert_eq!(v.as_str(), Some("h264"));
    }

    #[test]
    fn test_override_value_wrong_type_returns_none() {
        let v = OverrideValue::Float(3.14);
        assert!(v.as_int().is_none());
        assert!(v.as_str().is_none());
    }

    // --- PresetOverride ---

    #[test]
    fn test_preset_override_has_override() {
        let o = PresetOverride::new(OverrideField::Width, OverrideValue::Int(1920));
        assert!(o.has_override());
    }

    #[test]
    fn test_preset_override_with_reason() {
        let o = PresetOverride::new(OverrideField::Height, OverrideValue::Int(1080))
            .with_reason("Force 1080p");
        assert_eq!(o.reason, "Force 1080p");
    }

    #[test]
    fn test_apply_to_base_adds_field() {
        let mut base = HashMap::new();
        base.insert("width".to_string(), OverrideValue::Int(1280));
        let o = PresetOverride::new(OverrideField::Height, OverrideValue::Int(720));
        let merged = o.apply_to_base(&base);
        assert_eq!(merged.get("height"), Some(&OverrideValue::Int(720)));
        assert_eq!(merged.get("width"), Some(&OverrideValue::Int(1280)));
    }

    #[test]
    fn test_apply_to_base_overwrites_existing() {
        let mut base = HashMap::new();
        base.insert("video_bitrate".to_string(), OverrideValue::Int(4_000_000));
        let o = PresetOverride::new(OverrideField::VideoBitrate, OverrideValue::Int(2_000_000));
        let merged = o.apply_to_base(&base);
        assert_eq!(
            merged.get("video_bitrate"),
            Some(&OverrideValue::Int(2_000_000))
        );
    }

    // --- OverrideSet ---

    #[test]
    fn test_override_set_empty() {
        let set = OverrideSet::new("empty");
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    #[test]
    fn test_override_set_add() {
        let mut set = OverrideSet::new("test");
        set.add(PresetOverride::new(
            OverrideField::Width,
            OverrideValue::Int(1280),
        ));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_override_set_add_replaces_same_field() {
        let mut set = OverrideSet::new("dedup");
        set.add(PresetOverride::new(
            OverrideField::Crf,
            OverrideValue::Int(18),
        ));
        set.add(PresetOverride::new(
            OverrideField::Crf,
            OverrideValue::Int(23),
        ));
        assert_eq!(set.len(), 1);
        assert_eq!(set.overrides()[0].value, OverrideValue::Int(23));
    }

    #[test]
    fn test_override_set_remove() {
        let mut set = OverrideSet::new("remove-test");
        set.add(PresetOverride::new(
            OverrideField::EncoderPreset,
            OverrideValue::Str("fast".to_string()),
        ));
        let removed = set.remove(&OverrideField::EncoderPreset);
        assert!(removed.is_some());
        assert!(set.is_empty());
    }

    #[test]
    fn test_override_set_merged_values() {
        let mut base = HashMap::new();
        base.insert("width".to_string(), OverrideValue::Int(1920));
        base.insert("height".to_string(), OverrideValue::Int(1080));

        let mut set = OverrideSet::new("downscale");
        set.add(PresetOverride::new(
            OverrideField::Width,
            OverrideValue::Int(1280),
        ));
        set.add(PresetOverride::new(
            OverrideField::Height,
            OverrideValue::Int(720),
        ));

        let merged = set.merged_values(&base);
        assert_eq!(merged.get("width"), Some(&OverrideValue::Int(1280)));
        assert_eq!(merged.get("height"), Some(&OverrideValue::Int(720)));
    }

    #[test]
    fn test_override_set_has_breaking_overrides_true() {
        let mut set = OverrideSet::new("codec-change");
        set.add(PresetOverride::new(
            OverrideField::VideoCodec,
            OverrideValue::Str("av1".to_string()),
        ));
        assert!(set.has_breaking_overrides());
    }

    #[test]
    fn test_override_set_has_breaking_overrides_false() {
        let mut set = OverrideSet::new("safe");
        set.add(PresetOverride::new(
            OverrideField::VideoBitrate,
            OverrideValue::Int(3_000_000),
        ));
        assert!(!set.has_breaking_overrides());
    }
}
