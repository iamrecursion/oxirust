//! NMOS IS-11 Stream Compatibility Management.
//!
//! Determines whether senders and receivers can be connected without transcoding.
//! Implements the IS-11 v1.0 Stream Compatibility Management API, providing
//! capability advertisement, active constraint management, and compatibility
//! intersection logic for senders and receivers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Core media capability types
// ============================================================================

/// Interlace mode of a video signal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterlaceMode {
    /// Progressive scan.
    Progressive,
    /// Interlaced scan.
    Interlaced,
    /// Progressive segmented frame.
    Psf,
}

/// A single colour component descriptor (e.g. R, G, B, Y, Cb, Cr).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentInfo {
    /// Component name ("R", "G", "B", "Y", "Cb", "Cr", …).
    pub name: String,
    /// Bit depth of this component.
    pub bit_depth: u32,
}

/// IS-11 media format capabilities for a sender or receiver.
///
/// Fields that are `None` are considered unconstrained — any value is accepted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaCapability {
    /// IANA media type, e.g. `"video/raw"` or `"audio/L24"`.
    pub media_type: String,
    /// Grain rate as `(numerator, denominator)`, e.g. `(25, 1)` for 25 fps.
    pub grain_rate: Option<(u32, u32)>,
    /// Frame width in pixels (video only).
    pub frame_width: Option<u32>,
    /// Frame height in pixels (video only).
    pub frame_height: Option<u32>,
    /// Interlace mode (video only).
    pub interlace_mode: Option<InterlaceMode>,
    /// Colorspace string, e.g. `"BT709"`.
    pub colorspace: Option<String>,
    /// Transfer characteristic, e.g. `"SDR"` or `"HLG"`.
    pub transfer_characteristic: Option<String>,
    /// Component layout (video only).
    pub components: Option<Vec<ComponentInfo>>,
    // ── Audio-specific ────────────────────────────────────────────────────
    /// Sample rate in Hz (audio only).
    pub sample_rate: Option<u32>,
    /// Bit depth per sample (audio only).
    pub bit_depth: Option<u32>,
    /// Number of audio channels (audio only).
    pub channel_count: Option<u32>,
}

impl MediaCapability {
    /// Construct a `video/raw` capability with width, height and grain rate.
    pub fn video_raw(width: u32, height: u32, rate: (u32, u32)) -> Self {
        Self {
            media_type: "video/raw".to_string(),
            grain_rate: Some(rate),
            frame_width: Some(width),
            frame_height: Some(height),
            interlace_mode: Some(InterlaceMode::Progressive),
            colorspace: None,
            transfer_characteristic: None,
            components: None,
            sample_rate: None,
            bit_depth: None,
            channel_count: None,
        }
    }

    /// Construct an `audio/L24` capability with sample rate and channel count.
    pub fn audio_l24(sample_rate: u32, channels: u32) -> Self {
        Self {
            media_type: "audio/L24".to_string(),
            grain_rate: None,
            frame_width: None,
            frame_height: None,
            interlace_mode: None,
            colorspace: None,
            transfer_characteristic: None,
            components: None,
            sample_rate: Some(sample_rate),
            bit_depth: Some(24),
            channel_count: Some(channels),
        }
    }

    /// Check whether `self` is compatible with `other`.
    ///
    /// Two capabilities are compatible when:
    /// 1. Their `media_type` strings match exactly.
    /// 2. For every constrained field present in *both* capabilities, the
    ///    values are equal.  A `None` field on either side is unconstrained and
    ///    never causes a mismatch.
    pub fn is_compatible_with(&self, other: &MediaCapability) -> bool {
        // Media type must always match.
        if self.media_type != other.media_type {
            return false;
        }
        // Helper: compare two Option<T> — None on either side is unconstrained.
        fn opt_compat<T: PartialEq>(a: &Option<T>, b: &Option<T>) -> bool {
            match (a, b) {
                (Some(x), Some(y)) => x == y,
                _ => true,
            }
        }
        if !opt_compat(&self.grain_rate, &other.grain_rate) {
            return false;
        }
        if !opt_compat(&self.frame_width, &other.frame_width) {
            return false;
        }
        if !opt_compat(&self.frame_height, &other.frame_height) {
            return false;
        }
        if !opt_compat(&self.interlace_mode, &other.interlace_mode) {
            return false;
        }
        if !opt_compat(&self.colorspace, &other.colorspace) {
            return false;
        }
        if !opt_compat(
            &self.transfer_characteristic,
            &other.transfer_characteristic,
        ) {
            return false;
        }
        if !opt_compat(&self.sample_rate, &other.sample_rate) {
            return false;
        }
        if !opt_compat(&self.bit_depth, &other.bit_depth) {
            return false;
        }
        if !opt_compat(&self.channel_count, &other.channel_count) {
            return false;
        }
        true
    }
}

// ============================================================================
// Compatibility state
// ============================================================================

/// IS-11 compatibility state for a sender-receiver pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompatibilityState {
    /// The sender and receiver can be connected without transcoding.
    Compatible,
    /// A parameter constraint imposed by the receiver is violated by the sender.
    ParameterConstraintViolation,
    /// The capability sets are fundamentally incompatible (e.g. different media
    /// types or irreconcilable parameters).
    CapabilityViolation,
    /// Compatibility cannot be determined (e.g. one side is not registered).
    Unknown,
}

// ============================================================================
// IS-11 active/effective constraints
// ============================================================================

/// IS-11 constraint set: a map from parameter name to allowed values.
///
/// Keys are IS-11 parameter names (e.g. `"urn:x-nmos:cap:format:grain_rate"`).
/// Values are JSON arrays of allowed values per the IS-11 schema.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Is11ConstraintSet(pub HashMap<String, serde_json::Value>);

impl Is11ConstraintSet {
    /// Create an empty constraint set (unconstrained).
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// Insert a named constraint with an array of allowed values.
    pub fn insert(&mut self, param: impl Into<String>, allowed: serde_json::Value) {
        self.0.insert(param.into(), allowed);
    }

    /// Return `true` if there are no constraints (anything is accepted).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

// ============================================================================
// CompatibilityRegistry
// ============================================================================

/// IS-11 compatibility registry.
///
/// Tracks `MediaCapability` records for senders and receivers and computes
/// pairwise compatibility states.  Also stores IS-11 active constraints per
/// resource (used by the HTTP API).
pub struct CompatibilityRegistry {
    sender_caps: HashMap<String, MediaCapability>,
    receiver_caps: HashMap<String, MediaCapability>,
    /// Active constraints advertised by senders (effective constraints).
    sender_constraints: HashMap<String, Vec<Is11ConstraintSet>>,
    /// Active constraints imposed by receivers.
    receiver_constraints: HashMap<String, Vec<Is11ConstraintSet>>,
}

impl Default for CompatibilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CompatibilityRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            sender_caps: HashMap::new(),
            receiver_caps: HashMap::new(),
            sender_constraints: HashMap::new(),
            receiver_constraints: HashMap::new(),
        }
    }

    /// Register or replace a sender capability.
    pub fn register_sender(&mut self, id: String, cap: MediaCapability) {
        self.sender_caps.insert(id, cap);
    }

    /// Register or replace a receiver capability.
    pub fn register_receiver(&mut self, id: String, cap: MediaCapability) {
        self.receiver_caps.insert(id, cap);
    }

    /// Retrieve the capability for a registered sender.
    pub fn get_sender_cap(&self, id: &str) -> Option<&MediaCapability> {
        self.sender_caps.get(id)
    }

    /// Retrieve the capability for a registered receiver.
    pub fn get_receiver_cap(&self, id: &str) -> Option<&MediaCapability> {
        self.receiver_caps.get(id)
    }

    /// Return all registered sender IDs.
    pub fn all_sender_ids(&self) -> Vec<&str> {
        self.sender_caps.keys().map(String::as_str).collect()
    }

    /// Return all registered receiver IDs.
    pub fn all_receiver_ids(&self) -> Vec<&str> {
        self.receiver_caps.keys().map(String::as_str).collect()
    }

    /// Determine the IS-11 compatibility state for a sender-receiver pair.
    pub fn check_compatibility(&self, sender_id: &str, receiver_id: &str) -> CompatibilityState {
        let sender_cap = match self.sender_caps.get(sender_id) {
            Some(c) => c,
            None => return CompatibilityState::Unknown,
        };
        let receiver_cap = match self.receiver_caps.get(receiver_id) {
            Some(c) => c,
            None => return CompatibilityState::Unknown,
        };

        if sender_cap.is_compatible_with(receiver_cap) {
            CompatibilityState::Compatible
        } else if sender_cap.media_type != receiver_cap.media_type {
            CompatibilityState::CapabilityViolation
        } else {
            CompatibilityState::ParameterConstraintViolation
        }
    }

    /// Return all receiver IDs that are compatible with the given sender.
    pub fn compatible_receivers(&self, sender_id: &str) -> Vec<&str> {
        self.receiver_caps
            .keys()
            .filter(|rid| {
                self.check_compatibility(sender_id, rid.as_str()) == CompatibilityState::Compatible
            })
            .map(String::as_str)
            .collect()
    }

    /// Return all sender IDs that are compatible with the given receiver.
    pub fn compatible_senders(&self, receiver_id: &str) -> Vec<&str> {
        self.sender_caps
            .keys()
            .filter(|sid| {
                self.check_compatibility(sid.as_str(), receiver_id)
                    == CompatibilityState::Compatible
            })
            .map(String::as_str)
            .collect()
    }

    /// Store active constraints for a sender.
    pub fn set_sender_active_constraints(
        &mut self,
        sender_id: &str,
        constraints: Vec<Is11ConstraintSet>,
    ) -> Result<(), CompatibilityError> {
        if !self.sender_caps.contains_key(sender_id) {
            return Err(CompatibilityError::SenderNotFound(sender_id.to_string()));
        }
        self.sender_constraints
            .insert(sender_id.to_string(), constraints);
        Ok(())
    }

    /// Retrieve active constraints for a sender.
    pub fn get_sender_active_constraints(
        &self,
        sender_id: &str,
    ) -> Option<&Vec<Is11ConstraintSet>> {
        self.sender_constraints.get(sender_id)
    }

    /// Store active constraints for a receiver.
    pub fn set_receiver_active_constraints(
        &mut self,
        receiver_id: &str,
        constraints: Vec<Is11ConstraintSet>,
    ) -> Result<(), CompatibilityError> {
        if !self.receiver_caps.contains_key(receiver_id) {
            return Err(CompatibilityError::ReceiverNotFound(
                receiver_id.to_string(),
            ));
        }
        self.receiver_constraints
            .insert(receiver_id.to_string(), constraints);
        Ok(())
    }

    /// Retrieve active constraints for a receiver.
    pub fn get_receiver_active_constraints(
        &self,
        receiver_id: &str,
    ) -> Option<&Vec<Is11ConstraintSet>> {
        self.receiver_constraints.get(receiver_id)
    }

    /// Compute the constraint intersection between a sender and a receiver.
    ///
    /// Returns a merged `Is11ConstraintSet` that contains only the parameters
    /// present in *both* sides, keeping the receiver's value (the stricter
    /// constraint) when both define the same parameter.  Returns `None` if
    /// either side has no constraints recorded.
    pub fn intersect_constraints(
        &self,
        sender_id: &str,
        receiver_id: &str,
    ) -> Option<Is11ConstraintSet> {
        let sender_sets = self.sender_constraints.get(sender_id)?;
        let receiver_sets = self.receiver_constraints.get(receiver_id)?;

        let mut merged = Is11ConstraintSet::new();

        for s_set in sender_sets {
            for r_set in receiver_sets {
                for (key, r_val) in &r_set.0 {
                    if s_set.0.contains_key(key) {
                        // Receiver's constraint wins (more specific).
                        merged.0.insert(key.clone(), r_val.clone());
                    }
                }
            }
        }

        Some(merged)
    }
}

// ============================================================================
// Error type
// ============================================================================

/// Errors produced by IS-11 compatibility operations.
#[derive(Debug, thiserror::Error)]
pub enum CompatibilityError {
    /// The specified sender ID is not registered.
    #[error("sender not found: {0}")]
    SenderNotFound(String),
    /// The specified receiver ID is not registered.
    #[error("receiver not found: {0}")]
    ReceiverNotFound(String),
    /// A JSON serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

// ============================================================================
// IS-11 HTTP response helpers
// ============================================================================

/// Serialize a `CompatibilityState` to the IS-11 JSON status object.
pub fn compatibility_state_to_json(state: &CompatibilityState) -> serde_json::Value {
    let (state_str, description) = match state {
        CompatibilityState::Compatible => (
            "compatible",
            "Sender and receiver are compatible without transcoding.",
        ),
        CompatibilityState::ParameterConstraintViolation => (
            "parameter_constraint_violation",
            "One or more parameter constraints are violated.",
        ),
        CompatibilityState::CapabilityViolation => (
            "capability_violation",
            "Capability sets are fundamentally incompatible.",
        ),
        CompatibilityState::Unknown => ("unknown", "Compatibility cannot be determined."),
    };
    serde_json::json!({
        "state": state_str,
        "description": description,
    })
}

/// Serialize a list of `Is11ConstraintSet` to the IS-11 JSON array format.
pub fn constraints_to_json(sets: &[Is11ConstraintSet]) -> serde_json::Value {
    let arr: Vec<serde_json::Value> = sets
        .iter()
        .map(|s| {
            serde_json::Value::Object(s.0.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        })
        .collect();
    serde_json::Value::Array(arr)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── MediaCapability construction ──────────────────────────────────────

    #[test]
    fn test_video_raw_construction() {
        let cap = MediaCapability::video_raw(1920, 1080, (25, 1));
        assert_eq!(cap.media_type, "video/raw");
        assert_eq!(cap.frame_width, Some(1920));
        assert_eq!(cap.frame_height, Some(1080));
        assert_eq!(cap.grain_rate, Some((25, 1)));
        assert_eq!(cap.interlace_mode, Some(InterlaceMode::Progressive));
        assert!(cap.sample_rate.is_none());
    }

    #[test]
    fn test_audio_l24_construction() {
        let cap = MediaCapability::audio_l24(48000, 8);
        assert_eq!(cap.media_type, "audio/L24");
        assert_eq!(cap.sample_rate, Some(48000));
        assert_eq!(cap.channel_count, Some(8));
        assert_eq!(cap.bit_depth, Some(24));
        assert!(cap.frame_width.is_none());
    }

    // ── MediaCapability::is_compatible_with ───────────────────────────────

    #[test]
    fn test_identical_video_caps_compatible() {
        let a = MediaCapability::video_raw(1920, 1080, (25, 1));
        let b = MediaCapability::video_raw(1920, 1080, (25, 1));
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn test_different_media_type_incompatible() {
        let a = MediaCapability::video_raw(1920, 1080, (25, 1));
        let b = MediaCapability::audio_l24(48000, 2);
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_different_resolution_incompatible() {
        let a = MediaCapability::video_raw(1920, 1080, (25, 1));
        let b = MediaCapability::video_raw(1280, 720, (25, 1));
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_different_grain_rate_incompatible() {
        let a = MediaCapability::video_raw(1920, 1080, (25, 1));
        let b = MediaCapability::video_raw(1920, 1080, (30, 1));
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_none_fields_are_unconstrained() {
        let a = MediaCapability {
            media_type: "video/raw".to_string(),
            grain_rate: None, // unconstrained
            frame_width: Some(1920),
            frame_height: Some(1080),
            interlace_mode: None,
            colorspace: None,
            transfer_characteristic: None,
            components: None,
            sample_rate: None,
            bit_depth: None,
            channel_count: None,
        };
        let b = MediaCapability::video_raw(1920, 1080, (25, 1));
        // a has no grain_rate constraint, so the pair is compatible.
        assert!(a.is_compatible_with(&b));
    }

    #[test]
    fn test_colorspace_mismatch_incompatible() {
        let mut a = MediaCapability::video_raw(1920, 1080, (25, 1));
        a.colorspace = Some("BT709".to_string());
        let mut b = MediaCapability::video_raw(1920, 1080, (25, 1));
        b.colorspace = Some("BT2020".to_string());
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_audio_sample_rate_mismatch() {
        let a = MediaCapability::audio_l24(48000, 2);
        let b = MediaCapability::audio_l24(44100, 2);
        assert!(!a.is_compatible_with(&b));
    }

    #[test]
    fn test_audio_channel_mismatch() {
        let a = MediaCapability::audio_l24(48000, 2);
        let b = MediaCapability::audio_l24(48000, 8);
        assert!(!a.is_compatible_with(&b));
    }

    // ── CompatibilityState serialisation ─────────────────────────────────

    #[test]
    fn test_compatibility_state_serde_roundtrip() {
        let states = [
            CompatibilityState::Compatible,
            CompatibilityState::ParameterConstraintViolation,
            CompatibilityState::CapabilityViolation,
            CompatibilityState::Unknown,
        ];
        for s in &states {
            let serialised = serde_json::to_string(s).expect("serialise");
            let back: CompatibilityState = serde_json::from_str(&serialised).expect("deserialise");
            assert_eq!(*s, back);
        }
    }

    // ── CompatibilityRegistry ─────────────────────────────────────────────

    fn make_registry() -> CompatibilityRegistry {
        let mut reg = CompatibilityRegistry::new();
        reg.register_sender(
            "s1".to_string(),
            MediaCapability::video_raw(1920, 1080, (25, 1)),
        );
        reg.register_sender("s2".to_string(), MediaCapability::audio_l24(48000, 2));
        reg.register_receiver(
            "r1".to_string(),
            MediaCapability::video_raw(1920, 1080, (25, 1)),
        );
        reg.register_receiver(
            "r2".to_string(),
            MediaCapability::video_raw(1280, 720, (25, 1)),
        );
        reg.register_receiver("r3".to_string(), MediaCapability::audio_l24(48000, 2));
        reg
    }

    #[test]
    fn test_check_compatibility_compatible() {
        let reg = make_registry();
        assert_eq!(
            reg.check_compatibility("s1", "r1"),
            CompatibilityState::Compatible
        );
    }

    #[test]
    fn test_check_compatibility_capability_violation() {
        let reg = make_registry();
        // s1 is video, r3 is audio → media type mismatch → CapabilityViolation
        assert_eq!(
            reg.check_compatibility("s1", "r3"),
            CompatibilityState::CapabilityViolation
        );
    }

    #[test]
    fn test_check_compatibility_parameter_violation() {
        let reg = make_registry();
        // s1 is 1920×1080, r2 is 1280×720 → same media type, different resolution
        assert_eq!(
            reg.check_compatibility("s1", "r2"),
            CompatibilityState::ParameterConstraintViolation
        );
    }

    #[test]
    fn test_check_compatibility_unknown_sender() {
        let reg = make_registry();
        assert_eq!(
            reg.check_compatibility("nonexistent", "r1"),
            CompatibilityState::Unknown
        );
    }

    #[test]
    fn test_check_compatibility_unknown_receiver() {
        let reg = make_registry();
        assert_eq!(
            reg.check_compatibility("s1", "nonexistent"),
            CompatibilityState::Unknown
        );
    }

    #[test]
    fn test_compatible_receivers() {
        let reg = make_registry();
        let rcvrs = reg.compatible_receivers("s1");
        // r1 is compatible with s1; r2 has different resolution; r3 is audio
        assert_eq!(rcvrs.len(), 1);
        assert!(rcvrs.contains(&"r1"));
    }

    #[test]
    fn test_compatible_senders() {
        let reg = make_registry();
        let sndrs = reg.compatible_senders("r3");
        // Only s2 (audio/L24 48kHz 2ch) matches r3
        assert_eq!(sndrs.len(), 1);
        assert!(sndrs.contains(&"s2"));
    }

    #[test]
    fn test_all_sender_ids() {
        let reg = make_registry();
        let ids = reg.all_sender_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"s1"));
        assert!(ids.contains(&"s2"));
    }

    #[test]
    fn test_all_receiver_ids() {
        let reg = make_registry();
        let ids = reg.all_receiver_ids();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_get_sender_cap() {
        let reg = make_registry();
        assert!(reg.get_sender_cap("s1").is_some());
        assert!(reg.get_sender_cap("unknown").is_none());
    }

    #[test]
    fn test_get_receiver_cap() {
        let reg = make_registry();
        assert!(reg.get_receiver_cap("r1").is_some());
        assert!(reg.get_receiver_cap("unknown").is_none());
    }

    // ── Active constraints ────────────────────────────────────────────────

    #[test]
    fn test_set_sender_active_constraints_ok() {
        let mut reg = make_registry();
        let mut cs = Is11ConstraintSet::new();
        cs.insert(
            "urn:x-nmos:cap:format:grain_rate",
            json!({"numerator": 25, "denominator": 1}),
        );
        assert!(reg.set_sender_active_constraints("s1", vec![cs]).is_ok());
        assert!(reg.get_sender_active_constraints("s1").is_some());
    }

    #[test]
    fn test_set_sender_active_constraints_not_found() {
        let mut reg = make_registry();
        let err = reg
            .set_sender_active_constraints("missing", vec![])
            .expect_err("should fail");
        assert!(matches!(err, CompatibilityError::SenderNotFound(_)));
    }

    #[test]
    fn test_set_receiver_active_constraints_ok() {
        let mut reg = make_registry();
        assert!(reg.set_receiver_active_constraints("r1", vec![]).is_ok());
    }

    #[test]
    fn test_set_receiver_active_constraints_not_found() {
        let mut reg = make_registry();
        let err = reg
            .set_receiver_active_constraints("missing", vec![])
            .expect_err("should fail");
        assert!(matches!(err, CompatibilityError::ReceiverNotFound(_)));
    }

    #[test]
    fn test_intersect_constraints() {
        let mut reg = make_registry();
        let param = "urn:x-nmos:cap:format:grain_rate";

        let mut s_cs = Is11ConstraintSet::new();
        s_cs.insert(param, json!({"numerator": 25, "denominator": 1}));
        reg.set_sender_active_constraints("s1", vec![s_cs])
            .expect("ok");

        let mut r_cs = Is11ConstraintSet::new();
        r_cs.insert(param, json!({"numerator": 25, "denominator": 1}));
        reg.set_receiver_active_constraints("r1", vec![r_cs])
            .expect("ok");

        let intersection = reg.intersect_constraints("s1", "r1").expect("intersection");
        assert!(intersection.0.contains_key(param));
    }

    #[test]
    fn test_constraints_to_json() {
        let mut cs = Is11ConstraintSet::new();
        cs.insert("urn:x-nmos:cap:format:media_type", json!(["video/raw"]));
        let val = constraints_to_json(&[cs]);
        assert!(val.is_array());
        let arr = val.as_array().expect("array");
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_compatibility_state_to_json() {
        let j = compatibility_state_to_json(&CompatibilityState::Compatible);
        assert_eq!(j["state"], json!("compatible"));
    }

    #[test]
    fn test_interlace_mode_serde() {
        let m = InterlaceMode::Interlaced;
        let s = serde_json::to_string(&m).expect("ok");
        assert_eq!(s, "\"interlaced\"");
        let back: InterlaceMode = serde_json::from_str(&s).expect("ok");
        assert_eq!(back, InterlaceMode::Interlaced);
    }

    #[test]
    fn test_component_info_serde() {
        let c = ComponentInfo {
            name: "Y".to_string(),
            bit_depth: 10,
        };
        let s = serde_json::to_string(&c).expect("ok");
        let back: ComponentInfo = serde_json::from_str(&s).expect("ok");
        assert_eq!(back.name, "Y");
        assert_eq!(back.bit_depth, 10);
    }

    #[test]
    fn test_is11_constraint_set_empty() {
        let cs = Is11ConstraintSet::new();
        assert!(cs.is_empty());
    }

    #[test]
    fn test_is11_constraint_set_insert() {
        let mut cs = Is11ConstraintSet::new();
        cs.insert("param", json!([1920]));
        assert!(!cs.is_empty());
        assert!(cs.0.contains_key("param"));
    }
}
