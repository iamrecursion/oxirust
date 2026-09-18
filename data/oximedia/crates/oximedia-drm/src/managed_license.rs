//! License server simulation with full lifecycle management.
//!
//! Provides an in-memory license server that issues, validates, and manages
//! DRM licenses including region restrictions, play-count tracking, and
//! consumer-level revocation.

use crate::key_lifecycle::{ContentKey, KeyStore};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// PlaybackRights
// ---------------------------------------------------------------------------

/// Rights granted to a consumer for a specific piece of content.
#[derive(Debug, Clone)]
pub struct PlaybackRights {
    /// Whether the consumer may play the content.
    pub can_play: bool,
    /// Whether the consumer may download the content for offline use.
    pub can_download: bool,
    /// Whether the consumer may share the content.
    pub can_share: bool,
    /// Whether the consumer may make copies.
    pub can_copy: bool,
    /// Optional maximum resolution (width, height). `None` means unrestricted.
    pub max_resolution: Option<(u32, u32)>,
    /// Allowed ISO 3166-1 alpha-2 region codes. An empty list means all regions.
    pub allowed_regions: Vec<String>,
}

impl PlaybackRights {
    /// Create a permissive `PlaybackRights` allowing all operations in all regions.
    pub fn unrestricted() -> Self {
        Self {
            can_play: true,
            can_download: true,
            can_share: true,
            can_copy: true,
            max_resolution: None,
            allowed_regions: Vec::new(),
        }
    }

    /// Create rights that only permit playback (no download, no copy, no share).
    pub fn play_only() -> Self {
        Self {
            can_play: true,
            can_download: false,
            can_share: false,
            can_copy: false,
            max_resolution: None,
            allowed_regions: Vec::new(),
        }
    }

    /// Return `true` if the given region is allowed (empty list = all allowed).
    pub fn region_allowed(&self, region: &str) -> bool {
        if self.allowed_regions.is_empty() {
            return true;
        }
        self.allowed_regions
            .iter()
            .any(|r| r.eq_ignore_ascii_case(region))
    }
}

// ---------------------------------------------------------------------------
// License
// ---------------------------------------------------------------------------

/// A DRM license linking a consumer to a content key with associated rights.
#[derive(Debug, Clone)]
pub struct License {
    /// Unique license identifier.
    pub license_id: String,
    /// The content key ID this license grants access to.
    pub key_id: String,
    /// Device or user identifier for the consumer.
    pub consumer_id: String,
    /// Playback rights granted by this license.
    pub rights: PlaybackRights,
    /// When the license was issued.
    pub issued_at: SystemTime,
    /// Optional expiry; `None` means perpetual.
    pub expires_at: Option<SystemTime>,
    /// Remaining play count; `None` means unlimited.
    pub play_count_remaining: Option<u32>,
    /// Whether offline playback is permitted.
    pub offline_allowed: bool,
    /// Whether the license has been explicitly revoked.
    revoked: bool,
}

impl License {
    /// Return `true` if the license is still valid (not expired, not revoked,
    /// and play count not exhausted).
    pub fn is_active(&self) -> bool {
        if self.revoked {
            return false;
        }
        if let Some(exp) = self.expires_at {
            if SystemTime::now() >= exp {
                return false;
            }
        }
        if let Some(remaining) = self.play_count_remaining {
            if remaining == 0 {
                return false;
            }
        }
        true
    }

    /// Return `true` if the license has expired (time-based only).
    pub fn is_expired(&self) -> bool {
        if let Some(exp) = self.expires_at {
            SystemTime::now() >= exp
        } else {
            false
        }
    }

    /// Return `true` if the play count has been exhausted.
    pub fn is_exhausted(&self) -> bool {
        match self.play_count_remaining {
            Some(0) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// LicenseError
// ---------------------------------------------------------------------------

/// Errors that can occur during license operations.
#[derive(Debug, Clone)]
pub enum LicenseError {
    /// The requested content key was not found.
    KeyNotFound(String),
    /// The license has expired.
    LicenseExpired(String),
    /// The play count for this license has been exhausted.
    PlayCountExhausted(String),
    /// The consumer's region is not in the allowed list.
    RegionNotAllowed {
        consumer_region: String,
        allowed: Vec<String>,
    },
    /// The consumer is not authorised to use this license (wrong consumer ID).
    ConsumerNotAuthorized(String),
    /// The requested license was not found.
    LicenseNotFound(String),
    /// The license has been explicitly revoked.
    LicenseRevoked(String),
    /// The content key is no longer valid (expired or usage exhausted).
    KeyInvalid(String),
}

impl std::fmt::Display for LicenseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LicenseError::KeyNotFound(id) => write!(f, "content key not found: {}", id),
            LicenseError::LicenseExpired(id) => write!(f, "license expired: {}", id),
            LicenseError::PlayCountExhausted(id) => {
                write!(f, "play count exhausted for license: {}", id)
            }
            LicenseError::RegionNotAllowed {
                consumer_region,
                allowed,
            } => {
                write!(
                    f,
                    "region '{}' not allowed; permitted: [{}]",
                    consumer_region,
                    allowed.join(", ")
                )
            }
            LicenseError::ConsumerNotAuthorized(id) => {
                write!(f, "consumer not authorized for license: {}", id)
            }
            LicenseError::LicenseNotFound(id) => write!(f, "license not found: {}", id),
            LicenseError::LicenseRevoked(id) => write!(f, "license revoked: {}", id),
            LicenseError::KeyInvalid(id) => write!(f, "content key is invalid: {}", id),
        }
    }
}

impl std::error::Error for LicenseError {}

// ---------------------------------------------------------------------------
// LicenseStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about the licenses managed by a `ManagedLicenseServer`.
#[derive(Debug, Clone)]
pub struct LicenseStats {
    /// Total number of licenses ever issued (including revoked/expired).
    pub total_issued: u64,
    /// Number of currently active (valid) licenses.
    pub active: usize,
    /// Number of time-expired licenses.
    pub expired: usize,
    /// Number of licenses with exhausted play counts.
    pub exhausted: usize,
}

// ---------------------------------------------------------------------------
// ManagedLicenseServer
// ---------------------------------------------------------------------------

/// In-memory DRM license server with full lifecycle management.
///
/// Wraps a [`KeyStore`] and maintains a collection of issued [`License`]s.
pub struct ManagedLicenseServer {
    key_store: KeyStore,
    licenses: HashMap<String, License>,
    issued_count: u64,
}

impl ManagedLicenseServer {
    /// Create a new `ManagedLicenseServer` backed by the given `KeyStore`.
    pub fn new(key_store: KeyStore) -> Self {
        Self {
            key_store,
            licenses: HashMap::new(),
            issued_count: 0,
        }
    }

    /// Issue a new `License` for `consumer_id` to access the content key `key_id`.
    ///
    /// `valid_days` specifies how many days the license is valid; `None` creates a perpetual license.
    pub fn issue_license(
        &mut self,
        key_id: &str,
        consumer_id: &str,
        rights: PlaybackRights,
        valid_days: Option<u64>,
    ) -> Result<License, LicenseError> {
        // Verify the key exists in the store
        if self.key_store.get(key_id).is_none() {
            return Err(LicenseError::KeyNotFound(key_id.to_string()));
        }

        // Verify the key is still valid
        let key = self
            .key_store
            .get(key_id)
            .ok_or_else(|| LicenseError::KeyNotFound(key_id.to_string()))?;
        if !key.is_valid() {
            return Err(LicenseError::KeyInvalid(key_id.to_string()));
        }

        let issued_at = SystemTime::now();
        let expires_at = valid_days.map(|days| issued_at + Duration::from_secs(days * 86400));

        // Generate a license ID: hex of (issued_count XOR consumer_id hash XOR key_id hash)
        let license_id = generate_license_id(self.issued_count, consumer_id, key_id);
        self.issued_count += 1;

        let license = License {
            license_id: license_id.clone(),
            key_id: key_id.to_string(),
            consumer_id: consumer_id.to_string(),
            rights,
            issued_at,
            expires_at,
            play_count_remaining: None,
            offline_allowed: false,
            revoked: false,
        };

        self.licenses.insert(license_id, license.clone());
        Ok(license)
    }

    /// Acquire a content key using a license.
    ///
    /// Validates the license (ownership, region, expiry, play count) and, if successful,
    /// decrements the play count and returns a clone of the `ContentKey`.
    pub fn acquire(
        &mut self,
        license_id: &str,
        consumer_id: &str,
        consumer_region: &str,
    ) -> Result<ContentKey, LicenseError> {
        let license = self
            .licenses
            .get_mut(license_id)
            .ok_or_else(|| LicenseError::LicenseNotFound(license_id.to_string()))?;

        // Ownership check
        if license.consumer_id != consumer_id {
            return Err(LicenseError::ConsumerNotAuthorized(license_id.to_string()));
        }

        // Revocation check
        if license.revoked {
            return Err(LicenseError::LicenseRevoked(license_id.to_string()));
        }

        // Expiry check
        if license.is_expired() {
            return Err(LicenseError::LicenseExpired(license_id.to_string()));
        }

        // Play count check
        if license.is_exhausted() {
            return Err(LicenseError::PlayCountExhausted(license_id.to_string()));
        }

        // Region check
        if !license.rights.region_allowed(consumer_region) {
            return Err(LicenseError::RegionNotAllowed {
                consumer_region: consumer_region.to_string(),
                allowed: license.rights.allowed_regions.clone(),
            });
        }

        // Playback permission check
        if !license.rights.can_play {
            return Err(LicenseError::ConsumerNotAuthorized(license_id.to_string()));
        }

        // Decrement play count if limited
        if let Some(ref mut count) = license.play_count_remaining {
            *count = count.saturating_sub(1);
        }

        let key_id = license.key_id.clone();

        // Retrieve the content key
        let key = self
            .key_store
            .get(&key_id)
            .ok_or_else(|| LicenseError::KeyNotFound(key_id.clone()))?
            .clone();

        if !key.is_valid() {
            return Err(LicenseError::KeyInvalid(key_id));
        }

        Ok(key)
    }

    /// Revoke a specific license by ID. Returns `true` if the license existed.
    pub fn revoke_license(&mut self, license_id: &str) -> bool {
        if let Some(lic) = self.licenses.get_mut(license_id) {
            lic.revoked = true;
            true
        } else {
            false
        }
    }

    /// Revoke all licenses belonging to a consumer. Returns the count revoked.
    pub fn revoke_consumer_licenses(&mut self, consumer_id: &str) -> usize {
        let mut count = 0;
        for lic in self.licenses.values_mut() {
            if lic.consumer_id == consumer_id && !lic.revoked {
                lic.revoked = true;
                count += 1;
            }
        }
        count
    }

    /// Return aggregate statistics about the license collection.
    pub fn stats(&self) -> LicenseStats {
        let mut active = 0usize;
        let mut expired = 0usize;
        let mut exhausted = 0usize;

        for lic in self.licenses.values() {
            if lic.revoked {
                continue;
            }
            if lic.is_exhausted() {
                exhausted += 1;
            } else if lic.is_expired() {
                expired += 1;
            } else {
                active += 1;
            }
        }

        LicenseStats {
            total_issued: self.issued_count,
            active,
            expired,
            exhausted,
        }
    }

    /// Return a reference to the internal `KeyStore`.
    pub fn key_store(&self) -> &KeyStore {
        &self.key_store
    }

    /// Return a mutable reference to the internal `KeyStore`.
    pub fn key_store_mut(&mut self) -> &mut KeyStore {
        &mut self.key_store
    }

    /// Return the number of licenses currently held (active + revoked + expired).
    pub fn license_count(&self) -> usize {
        self.licenses.len()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a unique license ID from a counter and consumer/key IDs using FNV-1a hashing.
fn generate_license_id(counter: u64, consumer_id: &str, key_id: &str) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET ^ counter.wrapping_mul(0x9e3779b97f4a7c15);
    for byte in consumer_id.bytes().chain(key_id.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    // Also mix in SystemTime nanos for uniqueness across time
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(0);
    hash ^= nanos;
    hash = hash.wrapping_mul(FNV_PRIME);

    format!("{:016x}", hash)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_lifecycle::ContentKey;
    use std::collections::HashMap;
    use std::time::Duration;

    fn make_server() -> ManagedLicenseServer {
        let mut store = KeyStore::new(100);
        let key = ContentKey::generate_128(HashMap::new());
        store.store(key).expect("store should succeed");
        ManagedLicenseServer::new(store)
    }

    fn first_key_id(server: &ManagedLicenseServer) -> String {
        server
            .key_store()
            .active_keys()
            .first()
            .map(|k| k.key_id.clone())
            .expect("there should be at least one key")
    }

    #[test]
    fn test_issue_license_basic() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let rights = PlaybackRights::play_only();
        let lic = server
            .issue_license(&kid, "user-001", rights, Some(7))
            .expect("issue_license should succeed");
        assert_eq!(lic.consumer_id, "user-001");
        assert_eq!(lic.key_id, kid);
    }

    #[test]
    fn test_issue_license_key_not_found() {
        let mut server = make_server();
        let result = server.issue_license(
            "nonexistent-key",
            "user-001",
            PlaybackRights::play_only(),
            None,
        );
        assert!(matches!(result, Err(LicenseError::KeyNotFound(_))));
    }

    #[test]
    fn test_acquire_returns_key() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let lic = server
            .issue_license(&kid, "user-002", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");
        let key = server
            .acquire(&lic.license_id, "user-002", "US")
            .expect("acquire should succeed");
        assert_eq!(key.key_id, kid);
    }

    #[test]
    fn test_acquire_wrong_consumer() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let lic = server
            .issue_license(&kid, "user-003", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");
        let result = server.acquire(&lic.license_id, "intruder", "US");
        assert!(matches!(
            result,
            Err(LicenseError::ConsumerNotAuthorized(_))
        ));
    }

    #[test]
    fn test_acquire_expired_license() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let lic = server
            .issue_license(&kid, "user-004", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");

        // Manually expire the license
        let lic_entry = server
            .licenses
            .get_mut(&lic.license_id)
            .expect("license should exist");
        lic_entry.expires_at = Some(UNIX_EPOCH + Duration::from_secs(1));

        let result = server.acquire(&lic.license_id, "user-004", "US");
        assert!(matches!(result, Err(LicenseError::LicenseExpired(_))));
    }

    #[test]
    fn test_acquire_play_count_exhausted() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let lic = server
            .issue_license(&kid, "user-005", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");

        // Set play_count_remaining to 0
        let lic_entry = server
            .licenses
            .get_mut(&lic.license_id)
            .expect("license should exist");
        lic_entry.play_count_remaining = Some(0);

        let result = server.acquire(&lic.license_id, "user-005", "US");
        assert!(matches!(result, Err(LicenseError::PlayCountExhausted(_))));
    }

    #[test]
    fn test_acquire_region_not_allowed() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let mut rights = PlaybackRights::unrestricted();
        rights.allowed_regions = vec!["US".to_string(), "CA".to_string()];
        let lic = server
            .issue_license(&kid, "user-006", rights, None)
            .expect("issue_license should succeed");
        let result = server.acquire(&lic.license_id, "user-006", "DE");
        assert!(matches!(result, Err(LicenseError::RegionNotAllowed { .. })));
    }

    #[test]
    fn test_acquire_region_allowed() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let mut rights = PlaybackRights::unrestricted();
        rights.allowed_regions = vec!["US".to_string()];
        let lic = server
            .issue_license(&kid, "user-007", rights, None)
            .expect("issue_license should succeed");
        let result = server.acquire(&lic.license_id, "user-007", "US");
        assert!(result.is_ok());
    }

    #[test]
    fn test_revoke_license() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let lic = server
            .issue_license(&kid, "user-008", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");
        assert!(server.revoke_license(&lic.license_id));
        let result = server.acquire(&lic.license_id, "user-008", "US");
        assert!(matches!(result, Err(LicenseError::LicenseRevoked(_))));
    }

    #[test]
    fn test_revoke_consumer_licenses() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let _l1 = server
            .issue_license(&kid, "user-batch", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");
        let _l2 = server
            .issue_license(&kid, "user-batch", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");
        let revoked = server.revoke_consumer_licenses("user-batch");
        assert_eq!(revoked, 2);
    }

    #[test]
    fn test_stats_active_count() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        server
            .issue_license(&kid, "user-s1", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");
        server
            .issue_license(&kid, "user-s2", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");
        let stats = server.stats();
        assert_eq!(stats.total_issued, 2);
        assert_eq!(stats.active, 2);
    }

    #[test]
    fn test_play_count_decrements() {
        let mut server = make_server();
        let kid = first_key_id(&server);
        let lic = server
            .issue_license(&kid, "user-c1", PlaybackRights::unrestricted(), None)
            .expect("issue_license should succeed");

        // Give it 2 plays
        let lic_entry = server
            .licenses
            .get_mut(&lic.license_id)
            .expect("license should exist");
        lic_entry.play_count_remaining = Some(2);

        server
            .acquire(&lic.license_id, "user-c1", "US")
            .expect("acquire should succeed");
        server
            .acquire(&lic.license_id, "user-c1", "US")
            .expect("acquire should succeed");

        let result = server.acquire(&lic.license_id, "user-c1", "US");
        assert!(matches!(result, Err(LicenseError::PlayCountExhausted(_))));
    }
}
