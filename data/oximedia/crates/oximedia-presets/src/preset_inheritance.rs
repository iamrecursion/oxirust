//! Preset inheritance — derive presets from base presets with field-level overrides.
//!
//! This module implements a single-inheritance chain for presets so that a
//! "child" preset can share the majority of settings from a "base" preset
//! while overriding only the fields that differ.  The resolved (materialised)
//! output is always a flat [`InheritedConfig`] that carries no further
//! inheritance reference, making the result safe to use anywhere a complete
//! configuration is needed.
//!
//! # Design
//!
//! * Every preset that participates in inheritance is registered in an
//!   [`InheritanceRegistry`] by its ID and an optional parent ID.
//! * When [`InheritanceRegistry::resolve`] is called for a preset the registry
//!   walks the ancestor chain from the root down, merging fields at each
//!   level.  Child fields **always** win over parent fields (last-write-wins
//!   through the chain).
//! * Circular inheritance is detected and returned as
//!   [`InheritanceError::CircularInheritance`].
//! * Deep chains are guarded by a configurable `max_depth` limit (default 32)
//!   to prevent runaway loops when cycles are not trivially detectable.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use thiserror::Error;

// ── Errors ─────────────────────────────────────────────────────────────────

/// Errors that can occur during preset inheritance resolution.
#[derive(Debug, Error, Clone)]
pub enum InheritanceError {
    /// The requested preset ID is not registered.
    #[error("Preset not found in inheritance registry: {0}")]
    NotFound(String),

    /// A circular parent chain was detected.
    #[error("Circular inheritance detected involving preset: {0}")]
    CircularInheritance(String),

    /// The ancestor chain exceeds the allowed depth.
    #[error("Inheritance depth limit ({limit}) exceeded for preset: {id}")]
    DepthLimitExceeded {
        /// Preset whose chain triggered the limit.
        id: String,
        /// Configured maximum depth.
        limit: usize,
    },
}

// ── InheritableField ───────────────────────────────────────────────────────

/// A single overridable parameter value for an inherited preset.
///
/// Every variant stores the field in its most natural Rust type so that
/// downstream code does not need to parse strings.
#[derive(Debug, Clone, PartialEq)]
pub enum InheritableField {
    /// Unsigned integer value (bitrates, dimensions, etc.).
    UInt(u64),
    /// Signed integer value.
    Int(i64),
    /// Floating-point value (CRF, quality factors, etc.).
    Float(f64),
    /// Boolean flag.
    Bool(bool),
    /// Free-form text (codec names, container formats, etc.).
    Text(String),
    /// Frame-rate as a `(numerator, denominator)` pair.
    FrameRate(u32, u32),
}

impl InheritableField {
    /// Convenience: return the `UInt` value if this is a `UInt` variant.
    #[must_use]
    pub fn as_uint(&self) -> Option<u64> {
        match self {
            Self::UInt(v) => Some(*v),
            _ => None,
        }
    }

    /// Convenience: return the `Float` value if this is a `Float` variant.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Convenience: return the `Bool` value if this is a `Bool` variant.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Convenience: return the text value if this is a `Text` variant.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Convenience: return the `FrameRate` pair if this is a `FrameRate` variant.
    #[must_use]
    pub fn as_frame_rate(&self) -> Option<(u32, u32)> {
        match self {
            Self::FrameRate(n, d) => Some((*n, *d)),
            _ => None,
        }
    }
}

// ── Well-known field names ──────────────────────────────────────────────────

/// Canonical field names used by [`InheritedConfig`] and its helpers.
pub mod field {
    /// Video codec field name.
    pub const VIDEO_CODEC: &str = "video_codec";
    /// Audio codec field name.
    pub const AUDIO_CODEC: &str = "audio_codec";
    /// Video bitrate field name.
    pub const VIDEO_BITRATE: &str = "video_bitrate";
    /// Audio bitrate field name.
    pub const AUDIO_BITRATE: &str = "audio_bitrate";
    /// Width field name.
    pub const WIDTH: &str = "width";
    /// Height field name.
    pub const HEIGHT: &str = "height";
    /// Frame rate field name.
    pub const FRAME_RATE: &str = "frame_rate";
    /// Container format field name.
    pub const CONTAINER: &str = "container";
    /// CRF (Constant Rate Factor) field name.
    pub const CRF: &str = "crf";
    /// Pixel format field name.
    pub const PIXEL_FORMAT: &str = "pixel_format";
    /// Encoding profile field name.
    pub const PROFILE: &str = "profile";
    /// Encoding level field name.
    pub const LEVEL: &str = "level";
    /// HDR mode field name.
    pub const HDR: &str = "hdr";
    /// Two-pass encoding field name.
    pub const TWO_PASS: &str = "two_pass";
}

// ── InheritedConfig ─────────────────────────────────────────────────────────

/// A fully-resolved (flat) configuration produced by walking an inheritance
/// chain and merging fields from all ancestors.
#[derive(Debug, Clone, Default)]
pub struct InheritedConfig {
    /// Resolved fields keyed by canonical field name.
    pub fields: HashMap<String, InheritableField>,
}

impl InheritedConfig {
    /// Create an empty resolved configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a field value.
    pub fn set(&mut self, key: &str, value: InheritableField) {
        self.fields.insert(key.to_string(), value);
    }

    /// Builder-style field setter.
    #[must_use]
    pub fn with_field(mut self, key: &str, value: InheritableField) -> Self {
        self.set(key, value);
        self
    }

    /// Retrieve a field value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&InheritableField> {
        self.fields.get(key)
    }

    /// Convenience: get video codec name.
    #[must_use]
    pub fn video_codec(&self) -> Option<&str> {
        self.fields.get(field::VIDEO_CODEC)?.as_text()
    }

    /// Convenience: get audio codec name.
    #[must_use]
    pub fn audio_codec(&self) -> Option<&str> {
        self.fields.get(field::AUDIO_CODEC)?.as_text()
    }

    /// Convenience: get video bitrate.
    #[must_use]
    pub fn video_bitrate(&self) -> Option<u64> {
        self.fields.get(field::VIDEO_BITRATE)?.as_uint()
    }

    /// Convenience: get output container format.
    #[must_use]
    pub fn container(&self) -> Option<&str> {
        self.fields.get(field::CONTAINER)?.as_text()
    }

    /// Convenience: get output width.
    #[must_use]
    pub fn width(&self) -> Option<u64> {
        self.fields.get(field::WIDTH)?.as_uint()
    }

    /// Convenience: get output height.
    #[must_use]
    pub fn height(&self) -> Option<u64> {
        self.fields.get(field::HEIGHT)?.as_uint()
    }

    /// Convenience: get frame rate.
    #[must_use]
    pub fn frame_rate(&self) -> Option<(u32, u32)> {
        self.fields.get(field::FRAME_RATE)?.as_frame_rate()
    }

    /// Number of fields set.
    #[must_use]
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Merge `other` into `self`, with `other`'s fields winning on conflict.
    pub fn merge_from(&mut self, other: &Self) {
        for (k, v) in &other.fields {
            self.fields.insert(k.clone(), v.clone());
        }
    }
}

// ── InheritanceNode ─────────────────────────────────────────────────────────

/// A single node in the inheritance registry, combining an ID, an optional
/// parent ID, and the fields that this node introduces or overrides.
#[derive(Debug, Clone)]
struct InheritanceNode {
    id: String,
    parent_id: Option<String>,
    overrides: InheritedConfig,
}

// ── InheritanceRegistry ─────────────────────────────────────────────────────

/// Registry of preset inheritance relationships.
///
/// Register base presets with no parent and derived presets with a parent
/// ID, then call [`resolve`] to obtain a fully-merged [`InheritedConfig`].
///
/// [`resolve`]: InheritanceRegistry::resolve
#[derive(Debug, Default)]
pub struct InheritanceRegistry {
    nodes: HashMap<String, InheritanceNode>,
    /// Maximum allowed inheritance depth (default 32).
    max_depth: usize,
}

impl InheritanceRegistry {
    /// Create a new, empty registry with the default depth limit (32).
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            max_depth: 32,
        }
    }

    /// Create a registry with a custom depth limit.
    #[must_use]
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            max_depth,
        }
    }

    /// Register a base preset (no parent).
    pub fn register_base(&mut self, id: &str, config: InheritedConfig) {
        self.nodes.insert(
            id.to_string(),
            InheritanceNode {
                id: id.to_string(),
                parent_id: None,
                overrides: config,
            },
        );
    }

    /// Register a derived preset that inherits from `parent_id`.
    ///
    /// The `overrides` map contains only the fields that differ from the
    /// parent; all other fields are inherited.
    ///
    /// Returns `false` if `parent_id` is not yet registered (the call is
    /// still recorded, resolution will fail later if the parent is never
    /// added).
    pub fn register_derived(
        &mut self,
        id: &str,
        parent_id: &str,
        overrides: InheritedConfig,
    ) -> bool {
        let parent_exists = self.nodes.contains_key(parent_id);
        self.nodes.insert(
            id.to_string(),
            InheritanceNode {
                id: id.to_string(),
                parent_id: Some(parent_id.to_string()),
                overrides,
            },
        );
        parent_exists
    }

    /// Resolve the full (materialised) configuration for `id` by walking the
    /// ancestor chain from root to `id`, merging fields as we go.
    ///
    /// # Errors
    ///
    /// Returns [`InheritanceError::NotFound`] if `id` is unknown,
    /// [`InheritanceError::CircularInheritance`] if a cycle is detected, or
    /// [`InheritanceError::DepthLimitExceeded`] if the chain is too long.
    pub fn resolve(&self, id: &str) -> Result<InheritedConfig, InheritanceError> {
        // Collect the ancestor chain in bottom-up order (id first, root last).
        let chain = self.collect_chain(id)?;

        // Merge from root (last) down to id (first): child overrides parent.
        let mut result = InheritedConfig::new();
        for ancestor_id in chain.iter().rev() {
            let node = self
                .nodes
                .get(ancestor_id.as_str())
                .ok_or_else(|| InheritanceError::NotFound(ancestor_id.clone()))?;
            result.merge_from(&node.overrides);
        }
        Ok(result)
    }

    /// Build the ancestor chain for a given id, returned bottom-up
    /// (requested id first, root last).
    fn collect_chain(&self, id: &str) -> Result<Vec<String>, InheritanceError> {
        let mut chain: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut current_id = id.to_string();

        loop {
            if seen.contains(&current_id) {
                return Err(InheritanceError::CircularInheritance(current_id));
            }
            if chain.len() >= self.max_depth {
                return Err(InheritanceError::DepthLimitExceeded {
                    id: id.to_string(),
                    limit: self.max_depth,
                });
            }

            let node = self
                .nodes
                .get(&current_id)
                .ok_or_else(|| InheritanceError::NotFound(current_id.clone()))?;

            seen.insert(current_id.clone());
            chain.push(current_id.clone());

            match &node.parent_id {
                Some(parent) => current_id = parent.clone(),
                None => break,
            }
        }

        Ok(chain)
    }

    /// Check whether a preset with the given ID is registered.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    /// Number of registered nodes.
    #[must_use]
    pub fn count(&self) -> usize {
        self.nodes.len()
    }

    /// Return all registered IDs.
    #[must_use]
    pub fn ids(&self) -> Vec<&str> {
        self.nodes.keys().map(String::as_str).collect()
    }

    /// Return the depth of the ancestor chain for `id` (1 = root, no parent).
    ///
    /// Returns `None` if the id is not found or the chain contains an error.
    #[must_use]
    pub fn depth(&self, id: &str) -> Option<usize> {
        self.collect_chain(id).ok().map(|c| c.len())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> InheritedConfig {
        InheritedConfig::new()
            .with_field(field::VIDEO_CODEC, InheritableField::Text("h264".into()))
            .with_field(field::AUDIO_CODEC, InheritableField::Text("aac".into()))
            .with_field(field::VIDEO_BITRATE, InheritableField::UInt(5_000_000))
            .with_field(field::AUDIO_BITRATE, InheritableField::UInt(128_000))
            .with_field(field::WIDTH, InheritableField::UInt(1280))
            .with_field(field::HEIGHT, InheritableField::UInt(720))
            .with_field(field::CONTAINER, InheritableField::Text("mp4".into()))
            .with_field(field::FRAME_RATE, InheritableField::FrameRate(30, 1))
    }

    #[test]
    fn test_base_only_resolution() {
        let mut reg = InheritanceRegistry::new();
        reg.register_base("base-720p", base_config());
        let resolved = reg.resolve("base-720p").expect("resolve should succeed");
        assert_eq!(resolved.video_codec(), Some("h264"));
        assert_eq!(resolved.video_bitrate(), Some(5_000_000));
        assert_eq!(resolved.height(), Some(720));
    }

    #[test]
    fn test_single_level_inheritance_overrides_bitrate() {
        let mut reg = InheritanceRegistry::new();
        reg.register_base("base-720p", base_config());

        let overrides = InheritedConfig::new()
            .with_field(field::VIDEO_BITRATE, InheritableField::UInt(8_000_000))
            .with_field(field::HEIGHT, InheritableField::UInt(1080))
            .with_field(field::WIDTH, InheritableField::UInt(1920));

        reg.register_derived("child-1080p", "base-720p", overrides);

        let resolved = reg.resolve("child-1080p").expect("resolve should succeed");
        // Overridden fields
        assert_eq!(resolved.video_bitrate(), Some(8_000_000));
        assert_eq!(resolved.height(), Some(1080));
        assert_eq!(resolved.width(), Some(1920));
        // Inherited fields
        assert_eq!(resolved.video_codec(), Some("h264"));
        assert_eq!(resolved.audio_codec(), Some("aac"));
        assert_eq!(resolved.container(), Some("mp4"));
    }

    #[test]
    fn test_multi_level_inheritance() {
        let mut reg = InheritanceRegistry::new();
        reg.register_base("root", base_config());

        let level1 = InheritedConfig::new()
            .with_field(field::HEIGHT, InheritableField::UInt(1080))
            .with_field(field::WIDTH, InheritableField::UInt(1920));
        reg.register_derived("level1", "root", level1);

        let level2 = InheritedConfig::new()
            .with_field(field::VIDEO_BITRATE, InheritableField::UInt(12_000_000))
            .with_field(field::HDR, InheritableField::Bool(true));
        reg.register_derived("level2", "level1", level2);

        let resolved = reg.resolve("level2").expect("resolve should succeed");
        // Comes from root
        assert_eq!(resolved.video_codec(), Some("h264"));
        // Comes from level1
        assert_eq!(resolved.height(), Some(1080));
        // Comes from level2
        assert_eq!(resolved.video_bitrate(), Some(12_000_000));
        assert_eq!(
            resolved.get(field::HDR).and_then(|f| f.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn test_child_overrides_take_precedence() {
        let mut reg = InheritanceRegistry::new();
        reg.register_base("base", base_config());

        // Both child and grandchild set the codec, grandchild should win
        let child_overrides = InheritedConfig::new()
            .with_field(field::VIDEO_CODEC, InheritableField::Text("vp9".into()));
        reg.register_derived("child", "base", child_overrides);

        let grandchild_overrides = InheritedConfig::new()
            .with_field(field::VIDEO_CODEC, InheritableField::Text("av1".into()));
        reg.register_derived("grandchild", "child", grandchild_overrides);

        let resolved = reg.resolve("grandchild").expect("resolve should succeed");
        assert_eq!(resolved.video_codec(), Some("av1"));
    }

    #[test]
    fn test_not_found_returns_error() {
        let reg = InheritanceRegistry::new();
        let result = reg.resolve("missing-id");
        assert!(matches!(result, Err(InheritanceError::NotFound(_))));
    }

    #[test]
    fn test_circular_inheritance_detected() {
        let mut reg = InheritanceRegistry::new();
        // Manually insert nodes with a cycle (A -> B -> A)
        reg.nodes.insert(
            "a".to_string(),
            InheritanceNode {
                id: "a".to_string(),
                parent_id: Some("b".to_string()),
                overrides: InheritedConfig::new(),
            },
        );
        reg.nodes.insert(
            "b".to_string(),
            InheritanceNode {
                id: "b".to_string(),
                parent_id: Some("a".to_string()),
                overrides: InheritedConfig::new(),
            },
        );
        let result = reg.resolve("a");
        assert!(matches!(
            result,
            Err(InheritanceError::CircularInheritance(_))
        ));
    }

    #[test]
    fn test_depth_limit_exceeded() {
        let mut reg = InheritanceRegistry::with_max_depth(3);
        reg.register_base("n0", InheritedConfig::new());
        reg.register_derived("n1", "n0", InheritedConfig::new());
        reg.register_derived("n2", "n1", InheritedConfig::new());
        reg.register_derived("n3", "n2", InheritedConfig::new());
        // n3 chain is 4 deep (n3->n2->n1->n0), exceeds limit of 3
        let result = reg.resolve("n3");
        assert!(matches!(
            result,
            Err(InheritanceError::DepthLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_depth_method() {
        let mut reg = InheritanceRegistry::new();
        reg.register_base("root", base_config());
        reg.register_derived("child", "root", InheritedConfig::new());
        reg.register_derived("grandchild", "child", InheritedConfig::new());
        assert_eq!(reg.depth("root"), Some(1));
        assert_eq!(reg.depth("child"), Some(2));
        assert_eq!(reg.depth("grandchild"), Some(3));
        assert_eq!(reg.depth("unknown"), None);
    }

    #[test]
    fn test_contains_and_count() {
        let mut reg = InheritanceRegistry::new();
        assert!(!reg.contains("base"));
        reg.register_base("base", base_config());
        assert!(reg.contains("base"));
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_inheritable_field_accessors() {
        assert_eq!(InheritableField::UInt(42).as_uint(), Some(42));
        assert_eq!(InheritableField::Float(3.14).as_float(), Some(3.14));
        assert_eq!(InheritableField::Bool(true).as_bool(), Some(true));
        assert_eq!(
            InheritableField::Text("opus".into()).as_text(),
            Some("opus")
        );
        assert_eq!(
            InheritableField::FrameRate(60, 1).as_frame_rate(),
            Some((60, 1))
        );
        // Wrong variant returns None
        assert_eq!(InheritableField::UInt(1).as_float(), None);
        assert_eq!(InheritableField::Float(1.0).as_bool(), None);
    }

    #[test]
    fn test_merge_from_overwrites() {
        let mut base = InheritedConfig::new()
            .with_field(field::VIDEO_CODEC, InheritableField::Text("h264".into()))
            .with_field(field::HEIGHT, InheritableField::UInt(720));
        let override_cfg =
            InheritedConfig::new().with_field(field::HEIGHT, InheritableField::UInt(1080));
        base.merge_from(&override_cfg);
        assert_eq!(base.height(), Some(1080));
        // Un-overridden field preserved
        assert_eq!(base.video_codec(), Some("h264"));
    }

    #[test]
    fn test_register_derived_returns_false_for_missing_parent() {
        let mut reg = InheritanceRegistry::new();
        let ok = reg.register_derived("child", "nonexistent-parent", InheritedConfig::new());
        assert!(!ok);
    }

    #[test]
    fn test_field_count() {
        let cfg = base_config();
        assert_eq!(cfg.field_count(), 8);
    }
}
