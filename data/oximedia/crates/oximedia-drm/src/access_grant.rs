//! Access grant management for DRM-controlled content libraries.
//!
//! An [`AccessGrant`] describes what a user is permitted to do with a
//! specific piece of content (stream, download, offline, etc.).
//! The [`AccessGrantStore`] provides an in-memory registry with
//! expiry-aware lookup.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The type of access a user has been granted for a piece of content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GrantType {
    /// Unlimited streaming from the service.
    Stream,
    /// Permanent download for offline playback.
    Download,
    /// Time-limited offline access (rental).
    OfflineRental,
    /// Purchased content — permanent full access.
    Purchase,
    /// Promotional or trial access, potentially limited.
    Trial,
}

impl std::fmt::Display for GrantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GrantType::Stream => "stream",
            GrantType::Download => "download",
            GrantType::OfflineRental => "offline_rental",
            GrantType::Purchase => "purchase",
            GrantType::Trial => "trial",
        };
        write!(f, "{s}")
    }
}

/// A single access grant linking a user, content item, and grant type.
#[derive(Debug, Clone)]
pub struct AccessGrant {
    /// Unique identifier for this grant record.
    pub grant_id: String,
    /// The user or account this grant belongs to.
    pub user_id: String,
    /// The content item the grant applies to.
    pub content_id: String,
    /// The type of access granted.
    pub grant_type: GrantType,
    /// Unix epoch seconds when the grant was created.
    pub issued_at: i64,
    /// Optional Unix epoch seconds when this grant expires.
    /// `None` means the grant never expires.
    pub expires_at: Option<i64>,
}

impl AccessGrant {
    /// Create a non-expiring grant.
    pub fn new(
        grant_id: impl Into<String>,
        user_id: impl Into<String>,
        content_id: impl Into<String>,
        grant_type: GrantType,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        Self {
            grant_id: grant_id.into(),
            user_id: user_id.into(),
            content_id: content_id.into(),
            grant_type,
            issued_at: now,
            expires_at: None,
        }
    }

    /// Create a grant that expires after `ttl` from now.
    pub fn expiring(
        grant_id: impl Into<String>,
        user_id: impl Into<String>,
        content_id: impl Into<String>,
        grant_type: GrantType,
        ttl: Duration,
    ) -> Self {
        let mut g = Self::new(grant_id, user_id, content_id, grant_type);
        g.expires_at = Some(g.issued_at + ttl.as_secs() as i64);
        g
    }

    /// Create a grant with explicit timestamps (for testing).
    pub fn with_timestamps(
        grant_id: impl Into<String>,
        user_id: impl Into<String>,
        content_id: impl Into<String>,
        grant_type: GrantType,
        issued_at: i64,
        expires_at: Option<i64>,
    ) -> Self {
        Self {
            grant_id: grant_id.into(),
            user_id: user_id.into(),
            content_id: content_id.into(),
            grant_type,
            issued_at,
            expires_at,
        }
    }

    /// Return `true` if this grant is currently active at `now_secs`.
    pub fn is_active_at(&self, now_secs: i64) -> bool {
        if now_secs < self.issued_at {
            return false;
        }
        if let Some(exp) = self.expires_at {
            now_secs < exp
        } else {
            true
        }
    }
}

/// In-memory store of [`AccessGrant`]s with expiry-aware access queries.
#[derive(Debug, Default)]
pub struct AccessGrantStore {
    /// Keyed by grant_id.
    grants: HashMap<String, AccessGrant>,
}

impl AccessGrantStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a grant.
    pub fn insert(&mut self, grant: AccessGrant) {
        self.grants.insert(grant.grant_id.clone(), grant);
    }

    /// Remove a grant by its ID. Returns the removed grant if present.
    pub fn remove(&mut self, grant_id: &str) -> Option<AccessGrant> {
        self.grants.remove(grant_id)
    }

    /// Return `true` if `user_id` has an active grant of any type for
    /// `content_id` at `now_secs`.
    pub fn has_access(&self, user_id: &str, content_id: &str, now_secs: i64) -> bool {
        self.grants
            .values()
            .any(|g| g.user_id == user_id && g.content_id == content_id && g.is_active_at(now_secs))
    }

    /// Return `true` if `user_id` has an active grant of the specific
    /// `grant_type` for `content_id` at `now_secs`.
    pub fn has_access_type(
        &self,
        user_id: &str,
        content_id: &str,
        grant_type: GrantType,
        now_secs: i64,
    ) -> bool {
        self.grants.values().any(|g| {
            g.user_id == user_id
                && g.content_id == content_id
                && g.grant_type == grant_type
                && g.is_active_at(now_secs)
        })
    }

    /// Return all active grants for a given user at `now_secs`.
    pub fn active_grants_for_user(&self, user_id: &str, now_secs: i64) -> Vec<&AccessGrant> {
        self.grants
            .values()
            .filter(|g| g.user_id == user_id && g.is_active_at(now_secs))
            .collect()
    }

    /// Purge all grants that have expired before `now_secs`.
    pub fn purge_expired(&mut self, now_secs: i64) {
        self.grants.retain(|_, g| g.is_active_at(now_secs));
    }

    /// Total number of grants in the store (including expired ones).
    pub fn len(&self) -> usize {
        self.grants.len()
    }

    /// Return `true` if the store contains no grants.
    pub fn is_empty(&self) -> bool {
        self.grants.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_700_000_000;

    fn stream_grant(user: &str, content: &str) -> AccessGrant {
        AccessGrant::with_timestamps(
            format!("g-{user}-{content}"),
            user,
            content,
            GrantType::Stream,
            NOW - 100,
            None, // no expiry
        )
    }

    fn rental_grant(user: &str, content: &str, expires_at: i64) -> AccessGrant {
        AccessGrant::with_timestamps(
            format!("r-{user}-{content}"),
            user,
            content,
            GrantType::OfflineRental,
            NOW - 100,
            Some(expires_at),
        )
    }

    #[test]
    fn test_has_access_for_stream_grant() {
        let mut store = AccessGrantStore::new();
        store.insert(stream_grant("alice", "movie-1"));
        assert!(store.has_access("alice", "movie-1", NOW));
    }

    #[test]
    fn test_no_access_for_different_user() {
        let mut store = AccessGrantStore::new();
        store.insert(stream_grant("alice", "movie-1"));
        assert!(!store.has_access("bob", "movie-1", NOW));
    }

    #[test]
    fn test_no_access_for_different_content() {
        let mut store = AccessGrantStore::new();
        store.insert(stream_grant("alice", "movie-1"));
        assert!(!store.has_access("alice", "movie-2", NOW));
    }

    #[test]
    fn test_expired_rental_denied() {
        let mut store = AccessGrantStore::new();
        store.insert(rental_grant("bob", "series-1", NOW - 10)); // expired
        assert!(!store.has_access("bob", "series-1", NOW));
    }

    #[test]
    fn test_active_rental_allowed() {
        let mut store = AccessGrantStore::new();
        store.insert(rental_grant("bob", "series-1", NOW + 3600));
        assert!(store.has_access("bob", "series-1", NOW));
    }

    #[test]
    fn test_has_access_type_correct_type() {
        let mut store = AccessGrantStore::new();
        store.insert(stream_grant("alice", "movie-1"));
        assert!(store.has_access_type("alice", "movie-1", GrantType::Stream, NOW));
    }

    #[test]
    fn test_has_access_type_wrong_type() {
        let mut store = AccessGrantStore::new();
        store.insert(stream_grant("alice", "movie-1"));
        assert!(!store.has_access_type("alice", "movie-1", GrantType::Download, NOW));
    }

    #[test]
    fn test_active_grants_for_user() {
        let mut store = AccessGrantStore::new();
        store.insert(stream_grant("alice", "movie-1"));
        store.insert(stream_grant("alice", "movie-2"));
        store.insert(stream_grant("bob", "movie-1"));
        let grants = store.active_grants_for_user("alice", NOW);
        assert_eq!(grants.len(), 2);
    }

    #[test]
    fn test_purge_expired_removes_stale() {
        let mut store = AccessGrantStore::new();
        store.insert(rental_grant("carol", "doc-1", NOW - 1000)); // expired
        store.insert(stream_grant("carol", "doc-2")); // never expires
        store.purge_expired(NOW);
        assert_eq!(store.len(), 1);
        assert!(store.has_access("carol", "doc-2", NOW));
    }

    #[test]
    fn test_purge_keeps_active_grants() {
        let mut store = AccessGrantStore::new();
        store.insert(rental_grant("dave", "clip-1", NOW + 9999));
        store.purge_expired(NOW);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_remove_grant() {
        let mut store = AccessGrantStore::new();
        let g = stream_grant("eve", "film-5");
        let id = g.grant_id.clone();
        store.insert(g);
        let removed = store.remove(&id);
        assert!(removed.is_some());
        assert!(store.is_empty());
    }

    #[test]
    fn test_is_empty_and_len() {
        let mut store = AccessGrantStore::new();
        assert!(store.is_empty());
        store.insert(stream_grant("user", "content"));
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_grant_type_display() {
        assert_eq!(GrantType::Stream.to_string(), "stream");
        assert_eq!(GrantType::Purchase.to_string(), "purchase");
        assert_eq!(GrantType::Trial.to_string(), "trial");
    }

    #[test]
    fn test_is_active_at_not_yet_issued() {
        let g = AccessGrant::with_timestamps("g", "u", "c", GrantType::Stream, NOW + 1000, None);
        assert!(!g.is_active_at(NOW));
    }

    #[test]
    fn test_is_active_at_expired() {
        let g = AccessGrant::with_timestamps(
            "g",
            "u",
            "c",
            GrantType::OfflineRental,
            NOW - 200,
            Some(NOW - 100),
        );
        assert!(!g.is_active_at(NOW));
    }

    #[test]
    fn test_is_active_at_never_expires() {
        let g = AccessGrant::with_timestamps("g", "u", "c", GrantType::Purchase, NOW - 100, None);
        assert!(g.is_active_at(NOW));
        assert!(g.is_active_at(NOW + 1_000_000));
    }
}
