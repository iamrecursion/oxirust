//! Media linking between original high-resolution files and their proxies.
#![allow(dead_code)]

use std::collections::HashMap;

/// Status of a media link between an original and its proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkStatus {
    /// Both original and proxy exist and are verified.
    Valid,
    /// Link record exists but one or both files are missing.
    Broken,
    /// Link has not been verified since creation.
    Unverified,
    /// Original exists but no proxy has been generated yet.
    MissingProxy,
}

impl LinkStatus {
    /// Return `true` if the link is in a valid, usable state.
    pub fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Return `true` if the link is broken or unverified.
    pub fn needs_attention(self) -> bool {
        matches!(self, Self::Broken | Self::MissingProxy)
    }
}

/// A bidirectional link between an original media file and its proxy.
#[derive(Debug, Clone)]
pub struct MediaLink {
    /// Unique link identifier.
    pub id: u64,
    /// Path to the original high-resolution file.
    pub original_path: String,
    /// Path to the proxy file, if one exists.
    pub proxy_path: Option<String>,
    /// Current link status.
    pub status: LinkStatus,
    /// Duration in seconds.
    pub duration_secs: f64,
}

impl MediaLink {
    /// Create a new unverified link for an original file.
    pub fn new(id: u64, original_path: &str, duration_secs: f64) -> Self {
        Self {
            id,
            original_path: original_path.to_owned(),
            proxy_path: None,
            status: LinkStatus::MissingProxy,
            duration_secs,
        }
    }

    /// Return `true` if a proxy path is stored in this link.
    pub fn has_proxy(&self) -> bool {
        self.proxy_path.is_some()
    }

    /// Attach a proxy path and mark the link as unverified.
    pub fn attach_proxy(&mut self, proxy_path: &str) {
        self.proxy_path = Some(proxy_path.to_owned());
        self.status = LinkStatus::Unverified;
    }

    /// Mark the link as verified and valid.
    pub fn mark_valid(&mut self) {
        self.status = LinkStatus::Valid;
    }

    /// Mark the link as broken.
    pub fn mark_broken(&mut self) {
        self.status = LinkStatus::Broken;
    }
}

/// A store for managing many `MediaLink` records.
#[derive(Debug, Default)]
pub struct MediaLinkStore {
    /// Links indexed by original file path.
    by_original: HashMap<String, MediaLink>,
    /// Reverse index: proxy path -> original path.
    by_proxy: HashMap<String, String>,
    /// Auto-incrementing link ID counter.
    next_id: u64,
}

impl MediaLinkStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or update a link for the given original file.
    /// Returns the id of the inserted/updated link.
    pub fn insert(
        &mut self,
        original_path: &str,
        proxy_path: Option<&str>,
        duration_secs: f64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let mut link = MediaLink::new(id, original_path, duration_secs);
        if let Some(proxy) = proxy_path {
            link.attach_proxy(proxy);
            self.by_proxy
                .insert(proxy.to_owned(), original_path.to_owned());
        }
        self.by_original.insert(original_path.to_owned(), link);
        id
    }

    /// Look up a link by the original file path.
    pub fn find_original(&self, original_path: &str) -> Option<&MediaLink> {
        self.by_original.get(original_path)
    }

    /// Look up the original link given a proxy path.
    pub fn find_proxy(&self, proxy_path: &str) -> Option<&MediaLink> {
        let original = self.by_proxy.get(proxy_path)?;
        self.by_original.get(original)
    }

    /// Return a list of original paths that have no proxy attached.
    pub fn unlinked_originals(&self) -> Vec<&str> {
        self.by_original
            .values()
            .filter(|link| !link.has_proxy())
            .map(|link| link.original_path.as_str())
            .collect()
    }

    /// Return the total number of links in the store.
    pub fn len(&self) -> usize {
        self.by_original.len()
    }

    /// Return `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.by_original.is_empty()
    }

    /// Mark a link as valid by original path. Returns `false` if not found.
    pub fn verify(&mut self, original_path: &str) -> bool {
        if let Some(link) = self.by_original.get_mut(original_path) {
            link.mark_valid();
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_status_is_valid() {
        assert!(LinkStatus::Valid.is_valid());
        assert!(!LinkStatus::Broken.is_valid());
        assert!(!LinkStatus::Unverified.is_valid());
        assert!(!LinkStatus::MissingProxy.is_valid());
    }

    #[test]
    fn test_link_status_needs_attention() {
        assert!(LinkStatus::Broken.needs_attention());
        assert!(LinkStatus::MissingProxy.needs_attention());
        assert!(!LinkStatus::Valid.needs_attention());
        assert!(!LinkStatus::Unverified.needs_attention());
    }

    #[test]
    fn test_media_link_new_no_proxy() {
        let link = MediaLink::new(1, "original.mov", 120.0);
        assert!(!link.has_proxy());
        assert_eq!(link.status, LinkStatus::MissingProxy);
    }

    #[test]
    fn test_media_link_attach_proxy() {
        let mut link = MediaLink::new(1, "original.mov", 120.0);
        link.attach_proxy("proxy.mp4");
        assert!(link.has_proxy());
        assert_eq!(link.status, LinkStatus::Unverified);
    }

    #[test]
    fn test_media_link_mark_valid() {
        let mut link = MediaLink::new(1, "original.mov", 120.0);
        link.attach_proxy("proxy.mp4");
        link.mark_valid();
        assert!(link.status.is_valid());
    }

    #[test]
    fn test_media_link_mark_broken() {
        let mut link = MediaLink::new(1, "original.mov", 120.0);
        link.attach_proxy("proxy.mp4");
        link.mark_broken();
        assert_eq!(link.status, LinkStatus::Broken);
    }

    #[test]
    fn test_store_insert_and_find_original() {
        let mut store = MediaLinkStore::new();
        store.insert("orig.mov", Some("proxy.mp4"), 60.0);
        let link = store
            .find_original("orig.mov")
            .expect("should succeed in test");
        assert_eq!(link.original_path, "orig.mov");
    }

    #[test]
    fn test_store_find_proxy() {
        let mut store = MediaLinkStore::new();
        store.insert("orig.mov", Some("proxy.mp4"), 60.0);
        let link = store
            .find_proxy("proxy.mp4")
            .expect("should succeed in test");
        assert_eq!(link.original_path, "orig.mov");
    }

    #[test]
    fn test_store_find_proxy_not_found() {
        let store = MediaLinkStore::new();
        assert!(store.find_proxy("nonexistent.mp4").is_none());
    }

    #[test]
    fn test_store_unlinked_originals() {
        let mut store = MediaLinkStore::new();
        store.insert("a.mov", None, 10.0);
        store.insert("b.mov", Some("b_proxy.mp4"), 10.0);
        let unlinked = store.unlinked_originals();
        assert_eq!(unlinked.len(), 1);
        assert_eq!(unlinked[0], "a.mov");
    }

    #[test]
    fn test_store_is_empty() {
        let store = MediaLinkStore::new();
        assert!(store.is_empty());
    }

    #[test]
    fn test_store_len() {
        let mut store = MediaLinkStore::new();
        store.insert("a.mov", None, 10.0);
        store.insert("b.mov", None, 20.0);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_store_verify_sets_valid() {
        let mut store = MediaLinkStore::new();
        store.insert("a.mov", Some("a_proxy.mp4"), 10.0);
        let result = store.verify("a.mov");
        assert!(result);
        let link = store
            .find_original("a.mov")
            .expect("should succeed in test");
        assert!(link.status.is_valid());
    }

    #[test]
    fn test_store_verify_missing_returns_false() {
        let mut store = MediaLinkStore::new();
        assert!(!store.verify("nonexistent.mov"));
    }
}
