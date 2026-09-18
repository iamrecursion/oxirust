//! Offline proxy workflows for OxiMedia proxy system.
//!
//! Provides offline proxy editing workflows including proxy-only editing,
//! reconnection to original high-resolution media, and substitution strategies.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Status of an offline proxy clip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OfflineStatus {
    /// Proxy is available and ready for offline editing.
    Available,
    /// Proxy is missing; needs to be regenerated.
    Missing,
    /// Proxy is being generated.
    Generating,
    /// Original media is reconnected; proxy can be replaced.
    Reconnected,
    /// Proxy substitution is active.
    Substituted,
}

/// A proxy clip record used during offline editing.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OfflineProxyClip {
    /// Unique identifier for this clip.
    pub id: String,
    /// Path to the proxy file.
    pub proxy_path: PathBuf,
    /// Path to the original high-resolution file (may be absent during offline).
    pub original_path: Option<PathBuf>,
    /// Current status of the proxy.
    pub status: OfflineStatus,
    /// Proxy resolution as a fraction of original (e.g., 0.25 = quarter res).
    pub resolution_fraction: f32,
}

impl OfflineProxyClip {
    /// Create a new offline proxy clip.
    #[must_use]
    pub fn new(id: impl Into<String>, proxy_path: impl Into<PathBuf>) -> Self {
        Self {
            id: id.into(),
            proxy_path: proxy_path.into(),
            original_path: None,
            status: OfflineStatus::Available,
            resolution_fraction: 0.25,
        }
    }

    /// Set the original media path.
    #[must_use]
    pub fn with_original(mut self, original: impl Into<PathBuf>) -> Self {
        self.original_path = Some(original.into());
        self
    }

    /// Set the resolution fraction.
    #[must_use]
    pub fn with_resolution_fraction(mut self, fraction: f32) -> Self {
        self.resolution_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Check whether the proxy file is present on disk.
    #[must_use]
    pub fn proxy_exists(&self) -> bool {
        self.proxy_path.exists()
    }

    /// Check whether this clip has been reconnected to its original.
    #[must_use]
    pub fn is_reconnected(&self) -> bool {
        self.status == OfflineStatus::Reconnected && self.original_path.is_some()
    }
}

/// Strategy for handling missing proxies during offline editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstitutionStrategy {
    /// Use a black frame substitute.
    BlackFrame,
    /// Use a placeholder image with clip information.
    Placeholder,
    /// Skip the clip entirely in the edited timeline.
    Skip,
    /// Attempt to regenerate the proxy automatically.
    AutoRegenerate,
}

impl Default for SubstitutionStrategy {
    fn default() -> Self {
        Self::Placeholder
    }
}

/// Manager for offline proxy editing sessions.
#[allow(dead_code)]
pub struct OfflineProxySession {
    /// All clips registered in this session.
    clips: HashMap<String, OfflineProxyClip>,
    /// Strategy used when a proxy is missing.
    substitution_strategy: SubstitutionStrategy,
    /// Whether the session is in strict mode (error on missing proxy).
    strict_mode: bool,
}

impl OfflineProxySession {
    /// Create a new offline proxy session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            clips: HashMap::new(),
            substitution_strategy: SubstitutionStrategy::default(),
            strict_mode: false,
        }
    }

    /// Create a session with a specific substitution strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: SubstitutionStrategy) -> Self {
        self.substitution_strategy = strategy;
        self
    }

    /// Enable strict mode.
    #[must_use]
    pub fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Register a proxy clip.
    pub fn register(&mut self, clip: OfflineProxyClip) {
        self.clips.insert(clip.id.clone(), clip);
    }

    /// Get a clip by id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&OfflineProxyClip> {
        self.clips.get(id)
    }

    /// Get a mutable reference to a clip by id.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut OfflineProxyClip> {
        self.clips.get_mut(id)
    }

    /// Returns the total number of clips.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Count clips by status.
    #[must_use]
    pub fn count_by_status(&self, status: &OfflineStatus) -> usize {
        self.clips.values().filter(|c| &c.status == status).count()
    }

    /// Reconnect a clip to its original media.
    ///
    /// Returns `true` if the clip was found and reconnected.
    pub fn reconnect(&mut self, id: &str, original_path: impl Into<PathBuf>) -> bool {
        if let Some(clip) = self.clips.get_mut(id) {
            clip.original_path = Some(original_path.into());
            clip.status = OfflineStatus::Reconnected;
            true
        } else {
            false
        }
    }

    /// Mark a clip as substituted.
    pub fn substitute(&mut self, id: &str) -> bool {
        if let Some(clip) = self.clips.get_mut(id) {
            clip.status = OfflineStatus::Substituted;
            true
        } else {
            false
        }
    }

    /// Get the current substitution strategy.
    #[must_use]
    pub fn substitution_strategy(&self) -> SubstitutionStrategy {
        self.substitution_strategy
    }

    /// Check if strict mode is enabled.
    #[must_use]
    pub fn is_strict(&self) -> bool {
        self.strict_mode
    }

    /// List all clips that need reconnection.
    #[must_use]
    pub fn clips_needing_reconnection(&self) -> Vec<&OfflineProxyClip> {
        self.clips
            .values()
            .filter(|c| c.original_path.is_none())
            .collect()
    }

    /// List all reconnected clips.
    #[must_use]
    pub fn reconnected_clips(&self) -> Vec<&OfflineProxyClip> {
        self.clips.values().filter(|c| c.is_reconnected()).collect()
    }
}

impl Default for OfflineProxySession {
    fn default() -> Self {
        Self::new()
    }
}

/// Reconnect result after attempting to reconnect proxies to originals.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct ReconnectResult {
    /// Number of clips successfully reconnected.
    pub reconnected: usize,
    /// Number of clips that could not be reconnected.
    pub failed: usize,
    /// Paths of originals that were not found.
    pub missing_originals: Vec<PathBuf>,
}

impl ReconnectResult {
    /// Create an empty reconnect result.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total clips processed.
    #[must_use]
    pub fn total(&self) -> usize {
        self.reconnected + self.failed
    }

    /// Success rate as a fraction [0.0, 1.0].
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            return 1.0;
        }
        self.reconnected as f32 / total as f32
    }
}

/// Attempts to automatically reconnect proxies to originals in a directory.
#[allow(dead_code)]
pub struct AutoReconnector {
    /// Root directory to search for originals.
    search_root: PathBuf,
    /// File extensions to consider as originals.
    extensions: Vec<String>,
}

impl AutoReconnector {
    /// Create a new auto-reconnector.
    #[must_use]
    pub fn new(search_root: impl Into<PathBuf>) -> Self {
        Self {
            search_root: search_root.into(),
            extensions: vec![
                "mov".to_string(),
                "mxf".to_string(),
                "mp4".to_string(),
                "r3d".to_string(),
                "braw".to_string(),
            ],
        }
    }

    /// Set the file extensions to search for.
    #[must_use]
    pub fn with_extensions(mut self, exts: Vec<String>) -> Self {
        self.extensions = exts;
        self
    }

    /// Get the search root directory.
    #[must_use]
    pub fn search_root(&self) -> &Path {
        &self.search_root
    }

    /// Check if the given extension is included in the search.
    #[must_use]
    pub fn includes_extension(&self, ext: &str) -> bool {
        self.extensions.iter().any(|e| e.eq_ignore_ascii_case(ext))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offline_proxy_clip_new() {
        let clip = OfflineProxyClip::new("clip001", "/proxy/clip001.mp4");
        assert_eq!(clip.id, "clip001");
        assert_eq!(clip.status, OfflineStatus::Available);
        assert!(clip.original_path.is_none());
    }

    #[test]
    fn test_offline_proxy_clip_with_original() {
        let clip = OfflineProxyClip::new("clip001", "/proxy/clip001.mp4")
            .with_original("/original/clip001.mov");
        assert!(clip.original_path.is_some());
    }

    #[test]
    fn test_offline_proxy_clip_resolution_fraction_clamp() {
        let clip = OfflineProxyClip::new("c1", "/p.mp4").with_resolution_fraction(2.0);
        assert_eq!(clip.resolution_fraction, 1.0);

        let clip2 = OfflineProxyClip::new("c2", "/p.mp4").with_resolution_fraction(-0.5);
        assert_eq!(clip2.resolution_fraction, 0.0);
    }

    #[test]
    fn test_offline_proxy_clip_is_reconnected_false_without_original() {
        let mut clip = OfflineProxyClip::new("c1", "/p.mp4");
        clip.status = OfflineStatus::Reconnected;
        // No original path → not considered reconnected
        assert!(!clip.is_reconnected());
    }

    #[test]
    fn test_offline_proxy_clip_is_reconnected_true() {
        let mut clip = OfflineProxyClip::new("c1", "/p.mp4").with_original("/o.mov");
        clip.status = OfflineStatus::Reconnected;
        assert!(clip.is_reconnected());
    }

    #[test]
    fn test_session_register_and_get() {
        let mut session = OfflineProxySession::new();
        session.register(OfflineProxyClip::new("clip001", "/proxy/clip001.mp4"));
        assert_eq!(session.clip_count(), 1);
        assert!(session.get("clip001").is_some());
        assert!(session.get("nonexistent").is_none());
    }

    #[test]
    fn test_session_count_by_status() {
        let mut session = OfflineProxySession::new();
        session.register(OfflineProxyClip::new("c1", "/p1.mp4"));
        session.register(OfflineProxyClip::new("c2", "/p2.mp4"));
        let mut c3 = OfflineProxyClip::new("c3", "/p3.mp4");
        c3.status = OfflineStatus::Missing;
        session.register(c3);

        assert_eq!(session.count_by_status(&OfflineStatus::Available), 2);
        assert_eq!(session.count_by_status(&OfflineStatus::Missing), 1);
    }

    #[test]
    fn test_session_reconnect() {
        let mut session = OfflineProxySession::new();
        session.register(OfflineProxyClip::new("c1", "/proxy.mp4"));
        let ok = session.reconnect("c1", "/original.mov");
        assert!(ok);
        let clip = session.get("c1").expect("should succeed in test");
        assert_eq!(clip.status, OfflineStatus::Reconnected);
        assert!(clip.original_path.is_some());
    }

    #[test]
    fn test_session_reconnect_nonexistent() {
        let mut session = OfflineProxySession::new();
        let ok = session.reconnect("nonexistent", "/original.mov");
        assert!(!ok);
    }

    #[test]
    fn test_session_substitute() {
        let mut session = OfflineProxySession::new();
        session.register(OfflineProxyClip::new("c1", "/proxy.mp4"));
        session.substitute("c1");
        assert_eq!(
            session.get("c1").expect("should succeed in test").status,
            OfflineStatus::Substituted
        );
    }

    #[test]
    fn test_session_clips_needing_reconnection() {
        let mut session = OfflineProxySession::new();
        session.register(OfflineProxyClip::new("c1", "/p1.mp4")); // no original
        session.register(OfflineProxyClip::new("c2", "/p2.mp4").with_original("/o2.mov"));
        assert_eq!(session.clips_needing_reconnection().len(), 1);
    }

    #[test]
    fn test_reconnect_result_success_rate() {
        let mut result = ReconnectResult::new();
        result.reconnected = 8;
        result.failed = 2;
        assert!((result.success_rate() - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_reconnect_result_success_rate_empty() {
        let result = ReconnectResult::new();
        assert_eq!(result.success_rate(), 1.0);
    }

    #[test]
    fn test_auto_reconnector_includes_extension() {
        let rc = AutoReconnector::new("/media");
        assert!(rc.includes_extension("mov"));
        assert!(rc.includes_extension("MXF"));
        assert!(!rc.includes_extension("avi"));
    }

    #[test]
    fn test_auto_reconnector_custom_extensions() {
        let rc = AutoReconnector::new("/media")
            .with_extensions(vec!["avi".to_string(), "wmv".to_string()]);
        assert!(rc.includes_extension("avi"));
        assert!(!rc.includes_extension("mov"));
    }

    #[test]
    fn test_substitution_strategy_default() {
        let strategy = SubstitutionStrategy::default();
        assert_eq!(strategy, SubstitutionStrategy::Placeholder);
    }

    #[test]
    fn test_session_with_strategy() {
        let session = OfflineProxySession::new().with_strategy(SubstitutionStrategy::BlackFrame);
        assert_eq!(
            session.substitution_strategy(),
            SubstitutionStrategy::BlackFrame
        );
    }

    #[test]
    fn test_session_strict_mode() {
        let session = OfflineProxySession::new().strict();
        assert!(session.is_strict());

        let session2 = OfflineProxySession::new();
        assert!(!session2.is_strict());
    }
}
