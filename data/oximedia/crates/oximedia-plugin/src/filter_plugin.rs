//! Filter and transform plugin type.
//!
//! While [`CodecPlugin`](crate::traits::CodecPlugin) covers codec decode/encode
//! operations, media pipelines also need *filter* and *transform* stages:
//! colour grading, audio normalisation, subtitle burn-in, thumbnail scaling, etc.
//!
//! This module defines:
//!
//! - [`FilterFrame`] — a byte buffer with metadata passed through a filter chain.
//! - [`FilterPlugin`] — trait for filter/transform plugins.
//! - [`FilterPluginInfo`] — metadata about a filter plugin.
//! - [`FilterKind`] — whether the plugin operates on video, audio, or both.
//! - [`FilterRegistry`] — registry for filter plugins, analogous to
//!   [`PluginRegistry`](crate::registry::PluginRegistry) for codecs.
//! - [`StaticFilterPlugin`] — convenience builder for in-process filter plugins.

use crate::error::{PluginError, PluginResult};
use crate::traits::PLUGIN_API_VERSION;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ── FilterKind ────────────────────────────────────────────────────────────────

/// The type of media data a filter plugin operates on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    /// Filter operates on raw video frames.
    Video,
    /// Filter operates on raw audio samples.
    Audio,
    /// Filter operates on both video and audio (e.g. a muxer synchroniser).
    AudioVideo,
    /// Generic/binary filter (subtitles, metadata, etc.).
    Generic,
}

impl std::fmt::Display for FilterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterKind::Video => write!(f, "video"),
            FilterKind::Audio => write!(f, "audio"),
            FilterKind::AudioVideo => write!(f, "audio+video"),
            FilterKind::Generic => write!(f, "generic"),
        }
    }
}

// ── FilterFrame ───────────────────────────────────────────────────────────────

/// A unit of media data flowing through a filter chain.
///
/// Intentionally minimal: the raw bytes plus a metadata map for any
/// format-specific properties (e.g. `"width"`, `"sample_rate"`, `"pts"`).
#[derive(Debug, Clone)]
pub struct FilterFrame {
    /// Raw frame data (pixels, PCM samples, or arbitrary bytes).
    pub data: Vec<u8>,
    /// Free-form metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl FilterFrame {
    /// Construct a frame with data and empty metadata.
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            metadata: HashMap::new(),
        }
    }

    /// Construct a frame with both data and metadata.
    pub fn with_metadata(data: Vec<u8>, metadata: HashMap<String, String>) -> Self {
        Self { data, metadata }
    }
}

// ── FilterPluginInfo ──────────────────────────────────────────────────────────

/// Metadata returned by [`FilterPlugin::info`].
#[derive(Debug, Clone)]
pub struct FilterPluginInfo {
    /// Plugin name (e.g. `"color-grade"`, `"audio-normalize"`).
    pub name: String,
    /// Plugin version (semver string).
    pub version: String,
    /// Author / organisation.
    pub author: String,
    /// Human-readable description.
    pub description: String,
    /// API version this plugin targets.
    pub api_version: u32,
    /// Kind of frames this plugin accepts.
    pub kind: FilterKind,
    /// Additional supported filter operations (by name).
    pub filter_names: Vec<String>,
}

// ── FilterPlugin ──────────────────────────────────────────────────────────────

/// Trait that all filter / transform plugins implement.
///
/// A filter plugin processes [`FilterFrame`]s in a named operation and
/// returns the transformed frame.  Multiple named operations can be
/// exposed by a single plugin (e.g. `"scale"`, `"crop"`, `"rotate"` for
/// a geometry plugin).
///
/// # Thread Safety
///
/// Filter plugins must be `Send + Sync` because the filter registry may be
/// shared across threads.
pub trait FilterPlugin: Send + Sync {
    /// Get plugin metadata.
    fn info(&self) -> FilterPluginInfo;

    /// Apply the named filter operation to `frame`.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::NotFound`] if `op_name` is not supported,
    /// or [`PluginError::InitFailed`] on processing failure.
    fn apply(&self, op_name: &str, frame: FilterFrame) -> PluginResult<FilterFrame>;

    /// Return `true` if `op_name` is supported by this plugin.
    fn supports_op(&self, op_name: &str) -> bool {
        self.info().filter_names.iter().any(|n| n == op_name)
    }
}

// ── StaticFilterPlugin ────────────────────────────────────────────────────────

type FilterFn = Box<dyn Fn(&str, FilterFrame) -> PluginResult<FilterFrame> + Send + Sync>;

/// A filter plugin backed by a Rust closure, for in-process use.
pub struct StaticFilterPlugin {
    info: FilterPluginInfo,
    handler: Option<FilterFn>,
}

impl StaticFilterPlugin {
    /// Create a new static filter plugin with the given metadata.
    pub fn new(info: FilterPluginInfo) -> Self {
        Self {
            info,
            handler: None,
        }
    }

    /// Register the filter handler.
    ///
    /// The closure receives `(op_name, frame)` and must return the
    /// transformed frame or an error.
    #[must_use]
    pub fn with_handler<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, FilterFrame) -> PluginResult<FilterFrame> + Send + Sync + 'static,
    {
        self.handler = Some(Box::new(f));
        self
    }
}

impl FilterPlugin for StaticFilterPlugin {
    fn info(&self) -> FilterPluginInfo {
        self.info.clone()
    }

    fn apply(&self, op_name: &str, frame: FilterFrame) -> PluginResult<FilterFrame> {
        match &self.handler {
            Some(f) => f(op_name, frame),
            None => Err(PluginError::NotFound(format!(
                "No handler registered for op '{op_name}'"
            ))),
        }
    }
}

// ── FilterRegistry ────────────────────────────────────────────────────────────

/// Central registry for loaded filter plugins.
///
/// Mirrors the design of [`PluginRegistry`](crate::registry::PluginRegistry).
pub struct FilterRegistry {
    plugins: RwLock<Vec<Arc<dyn FilterPlugin>>>,
}

impl FilterRegistry {
    /// Create a new empty filter registry.
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(Vec::new()),
        }
    }

    /// Register a filter plugin.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::ApiIncompatible`] if the API version is wrong,
    /// or [`PluginError::AlreadyRegistered`] on duplicate name.
    pub fn register(&self, plugin: Arc<dyn FilterPlugin>) -> PluginResult<()> {
        let info = plugin.info();

        if info.api_version != PLUGIN_API_VERSION {
            return Err(PluginError::ApiIncompatible(format!(
                "Filter plugin '{}' has API v{}, host expects v{PLUGIN_API_VERSION}",
                info.name, info.api_version
            )));
        }

        let mut plugins = self
            .plugins
            .write()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        for existing in plugins.iter() {
            if existing.info().name == info.name {
                return Err(PluginError::AlreadyRegistered(info.name));
            }
        }

        plugins.push(plugin);
        Ok(())
    }

    /// Apply `op_name` using the first plugin that supports it.
    ///
    /// # Errors
    ///
    /// Returns [`PluginError::NotFound`] if no plugin handles `op_name`.
    pub fn apply_op(&self, op_name: &str, frame: FilterFrame) -> PluginResult<FilterFrame> {
        let plugins = self
            .plugins
            .read()
            .map_err(|e| PluginError::InitFailed(format!("lock poisoned: {e}")))?;

        for plugin in plugins.iter() {
            if plugin.supports_op(op_name) {
                return plugin.apply(op_name, frame);
            }
        }

        Err(PluginError::NotFound(format!(
            "No filter plugin handles op '{op_name}'"
        )))
    }

    /// List all registered filter plugins.
    pub fn list_plugins(&self) -> Vec<FilterPluginInfo> {
        match self.plugins.read() {
            Ok(p) => p.iter().map(|fp| fp.info()).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Return `true` if any registered plugin handles `op_name`.
    pub fn has_op(&self, op_name: &str) -> bool {
        match self.plugins.read() {
            Ok(p) => p.iter().any(|fp| fp.supports_op(op_name)),
            Err(_) => false,
        }
    }

    /// Unload all filter plugins.
    pub fn clear(&self) {
        if let Ok(mut p) = self.plugins.write() {
            p.clear();
        }
    }

    /// Return the number of registered filter plugins.
    pub fn plugin_count(&self) -> usize {
        match self.plugins.read() {
            Ok(p) => p.len(),
            Err(_) => 0,
        }
    }
}

impl Default for FilterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_info(name: &str, kind: FilterKind, ops: &[&str]) -> FilterPluginInfo {
        FilterPluginInfo {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            author: "Test".to_string(),
            description: "Test filter plugin".to_string(),
            api_version: PLUGIN_API_VERSION,
            kind,
            filter_names: ops.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn passthrough_plugin(name: &str, ops: &[&str]) -> Arc<dyn FilterPlugin> {
        let info = make_info(name, FilterKind::Video, ops);
        let plugin = StaticFilterPlugin::new(info).with_handler(|_op, frame| Ok(frame));
        Arc::new(plugin)
    }

    // 1. FilterKind display.
    #[test]
    fn test_filter_kind_display() {
        assert_eq!(FilterKind::Video.to_string(), "video");
        assert_eq!(FilterKind::Audio.to_string(), "audio");
        assert_eq!(FilterKind::AudioVideo.to_string(), "audio+video");
        assert_eq!(FilterKind::Generic.to_string(), "generic");
    }

    // 2. FilterFrame construction.
    #[test]
    fn test_filter_frame_new() {
        let f = FilterFrame::new(vec![1, 2, 3]);
        assert_eq!(f.data, vec![1, 2, 3]);
        assert!(f.metadata.is_empty());
    }

    // 3. FilterFrame with_metadata.
    #[test]
    fn test_filter_frame_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("pts".to_string(), "42".to_string());
        let f = FilterFrame::with_metadata(vec![0], meta);
        assert_eq!(f.metadata.get("pts"), Some(&"42".to_string()));
    }

    // 4. StaticFilterPlugin::supports_op.
    #[test]
    fn test_supports_op() {
        let p = passthrough_plugin("scale-plugin", &["scale", "crop"]);
        assert!(p.supports_op("scale"));
        assert!(p.supports_op("crop"));
        assert!(!p.supports_op("rotate"));
    }

    // 5. StaticFilterPlugin::apply passthrough.
    #[test]
    fn test_apply_passthrough() {
        let p = passthrough_plugin("pass", &["identity"]);
        let frame = FilterFrame::new(b"pixel data".to_vec());
        let out = p.apply("identity", frame.clone()).expect("apply");
        assert_eq!(out.data, frame.data);
    }

    // 6. StaticFilterPlugin without handler returns error.
    #[test]
    fn test_apply_no_handler() {
        let info = make_info("no-handler", FilterKind::Video, &["scale"]);
        let p = StaticFilterPlugin::new(info);
        assert!(p.apply("scale", FilterFrame::new(vec![])).is_err());
    }

    // 7. FilterRegistry register and list.
    #[test]
    fn test_registry_register_list() {
        let reg = FilterRegistry::new();
        let p = passthrough_plugin("scale", &["scale"]);
        reg.register(p).expect("register");
        assert_eq!(reg.plugin_count(), 1);
        let list = reg.list_plugins();
        assert_eq!(list[0].name, "scale");
    }

    // 8. FilterRegistry duplicate rejected.
    #[test]
    fn test_registry_duplicate_rejected() {
        let reg = FilterRegistry::new();
        reg.register(passthrough_plugin("p", &["op"]))
            .expect("first");
        assert!(reg.register(passthrough_plugin("p", &["op"])).is_err());
    }

    // 9. FilterRegistry wrong API version.
    #[test]
    fn test_registry_wrong_api_version() {
        let reg = FilterRegistry::new();
        let info = FilterPluginInfo {
            name: "bad-api".to_string(),
            version: "1.0.0".to_string(),
            author: "T".to_string(),
            description: "D".to_string(),
            api_version: 999,
            kind: FilterKind::Video,
            filter_names: vec![],
        };
        let p = Arc::new(StaticFilterPlugin::new(info));
        assert!(reg.register(p).is_err());
    }

    // 10. apply_op dispatches to correct plugin.
    #[test]
    fn test_apply_op_dispatch() {
        let reg = FilterRegistry::new();
        let invert_info = make_info("invert", FilterKind::Video, &["invert"]);
        let invert = StaticFilterPlugin::new(invert_info).with_handler(|_op, mut frame| {
            for b in &mut frame.data {
                *b = !*b;
            }
            Ok(frame)
        });
        reg.register(Arc::new(invert)).expect("register");

        let input = FilterFrame::new(vec![0x00, 0xFF]);
        let out = reg.apply_op("invert", input).expect("apply_op");
        assert_eq!(out.data, vec![0xFF, 0x00]);
    }

    // 11. apply_op returns NotFound when no plugin handles the op.
    #[test]
    fn test_apply_op_not_found() {
        let reg = FilterRegistry::new();
        let err = reg.apply_op("nonexistent", FilterFrame::new(vec![]));
        assert!(matches!(err, Err(PluginError::NotFound(_))));
    }

    // 12. has_op.
    #[test]
    fn test_has_op() {
        let reg = FilterRegistry::new();
        reg.register(passthrough_plugin("p", &["scale"]))
            .expect("reg");
        assert!(reg.has_op("scale"));
        assert!(!reg.has_op("crop"));
    }

    // 13. clear empties the registry.
    #[test]
    fn test_clear() {
        let reg = FilterRegistry::new();
        reg.register(passthrough_plugin("p", &["op"])).expect("reg");
        reg.clear();
        assert_eq!(reg.plugin_count(), 0);
    }
}
