//! Proxy-to-original file linking and reconnection.
//!
//! Maintains an in-memory registry of proxy↔original file associations using
//! a simple FNV-1a checksum for integrity validation, and provides a path-based
//! reconnection heuristic.

#![allow(dead_code)]
#![allow(missing_docs)]

// ---------------------------------------------------------------------------
// FNV-1a helpers
// ---------------------------------------------------------------------------

/// Compute an FNV-1a 64-bit hash over the given byte slice.
#[must_use]
fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// ProxyLink
// ---------------------------------------------------------------------------

/// An association between a proxy file and its high-resolution original.
#[derive(Debug, Clone, PartialEq)]
pub struct ProxyLink {
    /// Path to the proxy file.
    pub proxy_path: String,
    /// Path to the original (high-resolution) file.
    pub original_path: String,
    /// FNV-1a checksum of the proxy content at link creation time.
    pub checksum: u64,
    /// Unix timestamp (milliseconds) when the link was created.
    pub created_ms: u64,
}

impl ProxyLink {
    /// Create a new proxy link.
    #[must_use]
    pub fn new(
        proxy_path: impl Into<String>,
        original_path: impl Into<String>,
        checksum: u64,
        created_ms: u64,
    ) -> Self {
        Self {
            proxy_path: proxy_path.into(),
            original_path: original_path.into(),
            checksum,
            created_ms,
        }
    }

    /// Verify that `data` matches the stored checksum using FNV-1a hashing.
    ///
    /// Returns `true` when the computed hash equals [`Self::checksum`].
    #[must_use]
    pub fn is_valid_checksum(&self, data: &[u8]) -> bool {
        fnv1a_64(data) == self.checksum
    }
}

// ---------------------------------------------------------------------------
// ProxyLinkRegistry
// ---------------------------------------------------------------------------

/// In-memory registry that maps proxy files to their originals.
#[derive(Debug, Default)]
pub struct ProxyLinkRegistry {
    /// All registered links.
    pub links: Vec<ProxyLink>,
}

impl ProxyLinkRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a proxy↔original association.
    ///
    /// If a link for the same proxy path already exists it is replaced.
    pub fn register(&mut self, proxy: &str, original: &str, checksum: u64) {
        // Remove any existing link for this proxy path.
        self.links.retain(|l| l.proxy_path != proxy);
        self.links
            .push(ProxyLink::new(proxy, original, checksum, 0));
    }

    /// Find the original path for a given proxy path.
    ///
    /// Returns `None` if no link exists for the given proxy.
    #[must_use]
    pub fn find_original(&self, proxy: &str) -> Option<&str> {
        self.links
            .iter()
            .find(|l| l.proxy_path == proxy)
            .map(|l| l.original_path.as_str())
    }

    /// Find the proxy path for a given original path.
    ///
    /// Returns `None` if the original is not linked to any proxy.
    #[must_use]
    pub fn find_proxy(&self, original: &str) -> Option<&str> {
        self.links
            .iter()
            .find(|l| l.original_path == original)
            .map(|l| l.proxy_path.as_str())
    }

    /// Remove the link for a given proxy path.
    ///
    /// Returns `true` if a link was actually removed.
    pub fn unlink(&mut self, proxy: &str) -> bool {
        let before = self.links.len();
        self.links.retain(|l| l.proxy_path != proxy);
        self.links.len() < before
    }

    /// Returns `true` if the proxy path has a registered link.
    #[must_use]
    pub fn is_linked(&self, proxy: &str) -> bool {
        self.links.iter().any(|l| l.proxy_path == proxy)
    }
}

// ---------------------------------------------------------------------------
// ReconnectResult
// ---------------------------------------------------------------------------

/// Outcome of a proxy reconnection attempt.
#[derive(Debug, PartialEq)]
pub enum ReconnectResult {
    /// A unique matching file was found at the given path.
    Found(String),
    /// No matching file was found in the supplied search paths.
    NotFound,
    /// Multiple candidate files matched (ambiguous).
    Ambiguous(Vec<String>),
}

// ---------------------------------------------------------------------------
// ProxyReconnector
// ---------------------------------------------------------------------------

/// Reconnects a proxy to its original by scanning a set of search paths.
pub struct ProxyReconnector;

impl ProxyReconnector {
    /// Attempt to reconnect `proxy` to its original by matching filename suffix.
    ///
    /// For each candidate in `search_paths`, the reconnector checks whether the
    /// candidate path ends with the same filename (or sub-path) as `proxy`.
    ///
    /// # Returns
    /// * [`ReconnectResult::Found`] – exactly one match.
    /// * [`ReconnectResult::Ambiguous`] – more than one match.
    /// * [`ReconnectResult::NotFound`] – no match.
    #[must_use]
    pub fn reconnect(proxy: &str, search_paths: &[String]) -> ReconnectResult {
        // Use the file-name portion of the proxy path as the matching key.
        let key = proxy.rsplit('/').next().unwrap_or(proxy);

        let matches: Vec<String> = search_paths
            .iter()
            .filter(|p| p.ends_with(key))
            .cloned()
            .collect();

        match matches.len() {
            0 => ReconnectResult::NotFound,
            1 => ReconnectResult::Found(
                matches
                    .into_iter()
                    .next()
                    .expect("invariant: matches.len() == 1 guarantees a first element"),
            ),
            _ => ReconnectResult::Ambiguous(matches),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_empty() {
        // Empty data should still produce a deterministic hash.
        let h1 = fnv1a_64(b"");
        let h2 = fnv1a_64(b"");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        assert_ne!(fnv1a_64(b"hello"), fnv1a_64(b"world"));
    }

    #[test]
    fn test_proxy_link_is_valid_checksum_pass() {
        let data = b"test content";
        let checksum = fnv1a_64(data);
        let link = ProxyLink::new("proxy.mp4", "original.mov", checksum, 0);
        assert!(link.is_valid_checksum(data));
    }

    #[test]
    fn test_proxy_link_is_valid_checksum_fail() {
        let link = ProxyLink::new("proxy.mp4", "original.mov", 0xdeadbeef, 0);
        assert!(!link.is_valid_checksum(b"some data"));
    }

    #[test]
    fn test_registry_register_and_find_original() {
        let mut reg = ProxyLinkRegistry::new();
        reg.register("p.mp4", "o.mov", 0);
        assert_eq!(reg.find_original("p.mp4"), Some("o.mov"));
    }

    #[test]
    fn test_registry_find_proxy() {
        let mut reg = ProxyLinkRegistry::new();
        reg.register("p.mp4", "o.mov", 0);
        assert_eq!(reg.find_proxy("o.mov"), Some("p.mp4"));
    }

    #[test]
    fn test_registry_find_missing_returns_none() {
        let reg = ProxyLinkRegistry::new();
        assert!(reg.find_original("does_not_exist.mp4").is_none());
    }

    #[test]
    fn test_registry_unlink_existing() {
        let mut reg = ProxyLinkRegistry::new();
        reg.register("p.mp4", "o.mov", 0);
        let removed = reg.unlink("p.mp4");
        assert!(removed);
        assert!(!reg.is_linked("p.mp4"));
    }

    #[test]
    fn test_registry_unlink_missing_returns_false() {
        let mut reg = ProxyLinkRegistry::new();
        assert!(!reg.unlink("no_such.mp4"));
    }

    #[test]
    fn test_registry_is_linked() {
        let mut reg = ProxyLinkRegistry::new();
        reg.register("p.mp4", "o.mov", 0);
        assert!(reg.is_linked("p.mp4"));
    }

    #[test]
    fn test_registry_register_replaces_existing() {
        let mut reg = ProxyLinkRegistry::new();
        reg.register("p.mp4", "original1.mov", 0);
        reg.register("p.mp4", "original2.mov", 0);
        assert_eq!(reg.find_original("p.mp4"), Some("original2.mov"));
        assert_eq!(reg.links.len(), 1);
    }

    #[test]
    fn test_reconnector_found() {
        let paths = vec![
            "/media/archive/clip001.mov".to_string(),
            "/media/archive/clip002.mov".to_string(),
        ];
        let result = ProxyReconnector::reconnect("proxy/clip001.mov", &paths);
        assert_eq!(
            result,
            ReconnectResult::Found("/media/archive/clip001.mov".to_string())
        );
    }

    #[test]
    fn test_reconnector_not_found() {
        let paths = vec!["/media/archive/other.mov".to_string()];
        let result = ProxyReconnector::reconnect("proxy/clip001.mov", &paths);
        assert_eq!(result, ReconnectResult::NotFound);
    }

    #[test]
    fn test_reconnector_ambiguous() {
        let paths = vec![
            "/drive1/clip001.mov".to_string(),
            "/drive2/clip001.mov".to_string(),
        ];
        let result = ProxyReconnector::reconnect("proxy/clip001.mov", &paths);
        assert!(matches!(result, ReconnectResult::Ambiguous(_)));
    }
}
