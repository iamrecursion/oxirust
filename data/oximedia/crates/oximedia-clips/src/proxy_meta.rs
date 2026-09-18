//! Proxy generation metadata for clip management.
//!
//! Provides [`ProxySpec`], [`ProxyMetadata`], and [`ProxyManagerSpec`] for
//! tracking proxy media files associated with original clips.  The
//! [`ProxyManagerSpec`] type is intentionally named to avoid collision with the
//! existing `proxy::ProxyManager` which manages legacy `ProxyLink` records.

use std::collections::HashMap;

/// Technical specification of a proxy media file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxySpec {
    /// Proxy video width in pixels.
    pub width: u32,
    /// Proxy video height in pixels.
    pub height: u32,
    /// Codec identifier, e.g. `"h264"` or `"prores_proxy"`.
    pub codec: String,
    /// Target bitrate in kbps.
    pub bitrate: u32,
}

impl ProxySpec {
    /// Create a new proxy specification.
    #[must_use]
    pub fn new(width: u32, height: u32, codec: impl Into<String>, bitrate: u32) -> Self {
        Self {
            width,
            height,
            codec: codec.into(),
            bitrate,
        }
    }
}

/// Metadata record linking an original clip to its proxy file.
#[derive(Debug, Clone)]
pub struct ProxyMetadata {
    /// Absolute path to the original (full-resolution) clip.
    pub original_path: String,
    /// Absolute path to the proxy clip.
    pub proxy_path: String,
    /// Technical specification of the proxy.
    pub spec: ProxySpec,
    /// Unix timestamp (seconds since epoch) when the proxy was created.
    pub created_at: u64,
    /// Hex-encoded checksum of the proxy file (e.g. SHA-256).
    pub checksum: String,
}

impl ProxyMetadata {
    /// Create a new proxy metadata record.
    #[must_use]
    pub fn new(
        original_path: impl Into<String>,
        proxy_path: impl Into<String>,
        spec: ProxySpec,
        created_at: u64,
        checksum: impl Into<String>,
    ) -> Self {
        Self {
            original_path: original_path.into(),
            proxy_path: proxy_path.into(),
            spec,
            created_at,
            checksum: checksum.into(),
        }
    }
}

/// Errors produced by [`ProxyManagerSpec`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProxyValidationError {
    /// The proxy's actual dimensions differ from those in the spec.
    DimensionMismatch {
        /// Dimensions recorded in the spec (width, height).
        expected: (u32, u32),
        /// Dimensions observed at validation time (width, height).
        actual: (u32, u32),
    },
    /// The proxy file's checksum does not match the recorded value.
    ChecksumMismatch,
    /// A required file path was not found on disk.
    PathNotFound(String),
    /// No proxy is registered for the given original path.
    NotRegistered(String),
    /// A provided path string is empty.
    EmptyPath,
}

impl std::fmt::Display for ProxyValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "dimension mismatch: expected {}x{}, got {}x{}",
                expected.0, expected.1, actual.0, actual.1
            ),
            Self::ChecksumMismatch => write!(f, "proxy checksum mismatch"),
            Self::PathNotFound(p) => write!(f, "path not found: {p}"),
            Self::NotRegistered(p) => write!(f, "no proxy registered for: {p}"),
            Self::EmptyPath => write!(f, "path must not be empty"),
        }
    }
}

impl std::error::Error for ProxyValidationError {}

/// Manages proxy metadata records indexed by original clip path.
///
/// This type coexists with the legacy `proxy::ProxyManager` which manages
/// [`crate::proxy::ProxyLink`] records.  [`ProxyManagerSpec`] is the newer,
/// spec-driven variant.
#[derive(Debug, Clone, Default)]
pub struct ProxyManagerSpec {
    records: HashMap<String, ProxyMetadata>,
}

impl ProxyManagerSpec {
    /// Create a new, empty proxy manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Register a proxy metadata record.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyValidationError::EmptyPath`] if either path is empty.
    pub fn register(&mut self, meta: ProxyMetadata) -> Result<(), ProxyValidationError> {
        if meta.original_path.is_empty() || meta.proxy_path.is_empty() {
            return Err(ProxyValidationError::EmptyPath);
        }
        self.records.insert(meta.original_path.clone(), meta);
        Ok(())
    }

    /// Look up the proxy record for `original_path`.
    #[must_use]
    pub fn find_proxy(&self, original_path: &str) -> Option<&ProxyMetadata> {
        self.records.get(original_path)
    }

    /// Check whether the proxy dimensions match the recorded spec.
    ///
    /// Returns `Ok(true)` when valid, `Ok(false)` when dimensions mismatch,
    /// and `Err(ProxyValidationError::NotRegistered)` when no record exists.
    ///
    /// # Errors
    ///
    /// Returns [`ProxyValidationError::NotRegistered`] if no proxy is
    /// registered for `original_path`.
    pub fn is_proxy_valid(
        &self,
        original_path: &str,
        actual_width: u32,
        actual_height: u32,
    ) -> Result<bool, ProxyValidationError> {
        let meta = self
            .records
            .get(original_path)
            .ok_or_else(|| ProxyValidationError::NotRegistered(original_path.to_string()))?;

        if meta.spec.width == actual_width && meta.spec.height == actual_height {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Remove and return the proxy record for `original_path`.
    pub fn remove(&mut self, original_path: &str) -> Option<ProxyMetadata> {
        self.records.remove(original_path)
    }

    /// Return references to all registered proxy records.
    #[must_use]
    pub fn list_all(&self) -> Vec<&ProxyMetadata> {
        self.records.values().collect()
    }

    /// Number of registered proxy records.
    #[must_use]
    pub fn count(&self) -> usize {
        self.records.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec(w: u32, h: u32) -> ProxySpec {
        ProxySpec::new(w, h, "h264", 2000)
    }

    fn make_meta(orig: &str, proxy: &str, w: u32, h: u32) -> ProxyMetadata {
        ProxyMetadata::new(orig, proxy, make_spec(w, h), 1_700_000_000, "deadbeef")
    }

    #[test]
    fn test_register_and_find() {
        let mut mgr = ProxyManagerSpec::new();
        let meta = make_meta("/orig/clip.mov", "/proxy/clip_proxy.mp4", 1280, 720);
        mgr.register(meta).expect("register should succeed");

        let found = mgr.find_proxy("/orig/clip.mov").expect("should find proxy");
        assert_eq!(found.proxy_path, "/proxy/clip_proxy.mp4");
    }

    #[test]
    fn test_find_missing_returns_none() {
        let mgr = ProxyManagerSpec::new();
        assert!(mgr.find_proxy("/nonexistent.mov").is_none());
    }

    #[test]
    fn test_register_empty_original_path_errors() {
        let mut mgr = ProxyManagerSpec::new();
        let meta = make_meta("", "/proxy/clip.mp4", 1280, 720);
        let err = mgr.register(meta).unwrap_err();
        assert_eq!(err, ProxyValidationError::EmptyPath);
    }

    #[test]
    fn test_register_empty_proxy_path_errors() {
        let mut mgr = ProxyManagerSpec::new();
        let meta = make_meta("/orig/clip.mov", "", 1280, 720);
        let err = mgr.register(meta).unwrap_err();
        assert_eq!(err, ProxyValidationError::EmptyPath);
    }

    #[test]
    fn test_is_proxy_valid_dimensions_match() {
        let mut mgr = ProxyManagerSpec::new();
        mgr.register(make_meta("/orig/a.mov", "/proxy/a.mp4", 1280, 720))
            .expect("register");
        assert_eq!(mgr.is_proxy_valid("/orig/a.mov", 1280, 720), Ok(true));
    }

    #[test]
    fn test_is_proxy_valid_dimension_mismatch() {
        let mut mgr = ProxyManagerSpec::new();
        mgr.register(make_meta("/orig/a.mov", "/proxy/a.mp4", 1280, 720))
            .expect("register");
        assert_eq!(mgr.is_proxy_valid("/orig/a.mov", 640, 360), Ok(false));
    }

    #[test]
    fn test_is_proxy_valid_not_registered() {
        let mgr = ProxyManagerSpec::new();
        let err = mgr.is_proxy_valid("/unknown.mov", 1280, 720).unwrap_err();
        assert!(matches!(err, ProxyValidationError::NotRegistered(_)));
    }

    #[test]
    fn test_remove_existing() {
        let mut mgr = ProxyManagerSpec::new();
        mgr.register(make_meta("/orig/b.mov", "/proxy/b.mp4", 1920, 1080))
            .expect("register");
        let removed = mgr.remove("/orig/b.mov");
        assert!(removed.is_some());
        assert!(mgr.find_proxy("/orig/b.mov").is_none());
    }

    #[test]
    fn test_remove_missing_returns_none() {
        let mut mgr = ProxyManagerSpec::new();
        assert!(mgr.remove("/nonexistent.mov").is_none());
    }

    #[test]
    fn test_count() {
        let mut mgr = ProxyManagerSpec::new();
        assert_eq!(mgr.count(), 0);
        mgr.register(make_meta("/orig/c.mov", "/proxy/c.mp4", 1280, 720))
            .expect("register");
        assert_eq!(mgr.count(), 1);
        mgr.register(make_meta("/orig/d.mov", "/proxy/d.mp4", 1920, 1080))
            .expect("register");
        assert_eq!(mgr.count(), 2);
    }

    #[test]
    fn test_list_all() {
        let mut mgr = ProxyManagerSpec::new();
        mgr.register(make_meta("/orig/e.mov", "/proxy/e.mp4", 1280, 720))
            .expect("register");
        mgr.register(make_meta("/orig/f.mov", "/proxy/f.mp4", 1920, 1080))
            .expect("register");
        let list = mgr.list_all();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_register_overwrites_duplicate() {
        let mut mgr = ProxyManagerSpec::new();
        mgr.register(make_meta("/orig/g.mov", "/proxy/g_v1.mp4", 1280, 720))
            .expect("register v1");
        mgr.register(make_meta("/orig/g.mov", "/proxy/g_v2.mp4", 1920, 1080))
            .expect("register v2");
        assert_eq!(mgr.count(), 1);
        let found = mgr.find_proxy("/orig/g.mov").expect("should find");
        assert_eq!(found.proxy_path, "/proxy/g_v2.mp4");
        assert_eq!(found.spec.width, 1920);
    }

    #[test]
    fn test_proxy_spec_fields() {
        let spec = ProxySpec::new(3840, 2160, "prores_proxy", 8000);
        assert_eq!(spec.width, 3840);
        assert_eq!(spec.height, 2160);
        assert_eq!(spec.codec, "prores_proxy");
        assert_eq!(spec.bitrate, 8000);
    }

    #[test]
    fn test_proxy_metadata_fields() {
        let meta = ProxyMetadata::new(
            "/orig/h.mxf",
            "/proxy/h_proxy.mp4",
            ProxySpec::new(960, 540, "h264", 1500),
            1_234_567_890,
            "abc123",
        );
        assert_eq!(meta.original_path, "/orig/h.mxf");
        assert_eq!(meta.proxy_path, "/proxy/h_proxy.mp4");
        assert_eq!(meta.created_at, 1_234_567_890);
        assert_eq!(meta.checksum, "abc123");
    }

    #[test]
    fn test_validation_error_display_dimension_mismatch() {
        let err = ProxyValidationError::DimensionMismatch {
            expected: (1280, 720),
            actual: (640, 360),
        };
        let msg = err.to_string();
        assert!(msg.contains("1280x720"));
        assert!(msg.contains("640x360"));
    }

    #[test]
    fn test_validation_error_display_not_registered() {
        let err = ProxyValidationError::NotRegistered("/missing.mov".to_string());
        assert!(err.to_string().contains("/missing.mov"));
    }
}
