//! Content key lifecycle management.
//!
//! Provides full lifecycle management for DRM content keys including
//! generation, storage, expiry, usage tracking, and secure export.
//!
//! # Key Lifecycle State Diagram
//!
//! ```text
//! ┌───────────┐
//! │ Generated │  Key material is produced (random bytes or KDF output).
//! └─────┬─────┘  Stored in ContentKey with created_at timestamp.
//!       │ wrap()
//!       ▼
//! ┌─────────┐
//! │ Wrapped │  Key material is encrypted under a Key Encryption Key (KEK).
//! └─────┬───┘  See key_wrap.rs for AES-KW / RSA-OAEP wrapping.
//!       │ distribute() / deliver_license()
//!       ▼
//! ┌─────────────┐
//! │ Distributed │  License request matched; wrapped key sent to client CDM.
//! └──────┬──────┘  Audit event written (see audit_trail.rs).
//!        │ client_ack() or implicit (license_type = Streaming)
//!        ▼
//! ┌────────┐
//! │ Active │  Key is in use for encrypt / decrypt operations.
//! └───┬────┘  usage_count incremented on each use; max_usage checked.
//!     │
//!     ├──── expires_at reached ──────────────────────┐
//!     │                                              ▼
//!     │ rotate() (see key_rotation.rs)        ┌──────────┐
//!     ▼                                       │ Revoked  │  Key invalidated before
//! ┌─────────┐                                 └────┬─────┘  natural expiry (device
//! │ Rotated │  New key becomes Active; old key     │         revocation, compliance).
//! └────┬────┘  enters overlap window for ongoing   │ destroy()
//!      │       segment decryption.                 ▼
//!      │ overlap_window expires               ┌───────────┐
//!      │                                      │ Destroyed │  Key material zeroed and
//!      └──────────────────────────────────────►            │  removed from all stores.
//!                                             └───────────┘
//! ```
//!
//! ## State Transition Details
//!
//! | Transition          | Trigger                                     | Data structures involved                        |
//! |---------------------|---------------------------------------------|-------------------------------------------------|
//! | Generated → Wrapped | `KeyStore::generate_and_store`              | [`ContentKey`], `key_wrap::wrap_key`            |
//! | Wrapped → Distributed | License server grants entitlement         | `entitlement::Entitlement`, `license_server`    |
//! | Distributed → Active | Client CDM acknowledges license            | `managed_license::ManagedLicense`               |
//! | Active → Rotated    | `key_rotation_schedule::RotationSchedule`   | `key_rotation_schedule::RotationSchedule`       |
//! | Active → Revoked    | Policy engine / device registry revocation  | `policy_engine::PolicyEngine`, `device_registry`|
//! | Rotated → Destroyed | Overlap window expiry                       | `KeyStore::remove_key`                          |
//! | Revoked → Destroyed | `KeyStore::remove_key`                      | `audit_trail::AuditTrail` logs destroy event    |
//!
//! ## Orphan Recovery
//!
//! If a key enters the `Distributed` state but no client acknowledgement arrives
//! within the license `playback_duration`, the `ManagedLicense` expiry timer
//! fires and the key is treated as `Revoked`.  `KeyStore::remove_key` can then
//! be called to reach `Destroyed`.  Audit records are preserved even after
//! key material destruction so compliance reports remain accurate.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Key algorithm enumeration
// ---------------------------------------------------------------------------

/// Cryptographic algorithm associated with a content key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyAlgorithm {
    /// AES-128-CTR
    Aes128Ctr,
    /// AES-256-CTR
    Aes256Ctr,
    /// AES-128-CBC
    Aes128Cbc,
    /// ChaCha20 stream cipher
    ChaCha20,
}

impl KeyAlgorithm {
    /// Return the algorithm name as a static string.
    pub fn name(&self) -> &'static str {
        match self {
            KeyAlgorithm::Aes128Ctr => "AES-128-CTR",
            KeyAlgorithm::Aes256Ctr => "AES-256-CTR",
            KeyAlgorithm::Aes128Cbc => "AES-128-CBC",
            KeyAlgorithm::ChaCha20 => "ChaCha20",
        }
    }

    /// Expected key length in bytes for this algorithm.
    pub fn key_len(&self) -> usize {
        match self {
            KeyAlgorithm::Aes128Ctr | KeyAlgorithm::Aes128Cbc => 16,
            KeyAlgorithm::Aes256Ctr | KeyAlgorithm::ChaCha20 => 32,
        }
    }
}

// ---------------------------------------------------------------------------
// ContentKey
// ---------------------------------------------------------------------------

/// A DRM content key with full lifecycle metadata.
#[derive(Debug, Clone)]
pub struct ContentKey {
    /// UUID-style hex identifier derived from the key material.
    pub key_id: String,
    /// Raw key bytes: 16 bytes for 128-bit algorithms, 32 bytes for 256-bit.
    pub key_data: Vec<u8>,
    /// Cryptographic algorithm this key is intended for.
    pub algorithm: KeyAlgorithm,
    /// When the key was generated.
    pub created_at: SystemTime,
    /// Optional expiry time; `None` means the key never expires.
    pub expires_at: Option<SystemTime>,
    /// How many times this key has been used.
    pub usage_count: u64,
    /// Maximum allowed uses; `None` means unlimited.
    pub max_usage: Option<u64>,
    /// Arbitrary metadata labels attached to this key.
    pub labels: HashMap<String, String>,
}

impl ContentKey {
    /// Generate a new random 128-bit (16-byte) content key.
    ///
    /// Uses a xorshift64 PRNG seeded from `SystemTime` for key material.
    /// The `key_id` is derived from the hex encoding of the first 8 bytes.
    pub fn generate_128(labels: HashMap<String, String>) -> Self {
        let key_data = generate_random_bytes(16);
        let key_id = hex_encode(&key_data[0..8]);
        Self {
            key_id,
            key_data,
            algorithm: KeyAlgorithm::Aes128Ctr,
            created_at: SystemTime::now(),
            expires_at: None,
            usage_count: 0,
            max_usage: None,
            labels,
        }
    }

    /// Generate a new random 256-bit (32-byte) content key.
    ///
    /// Uses a xorshift64 PRNG seeded from `SystemTime` for key material.
    /// The `key_id` is derived from the hex encoding of the first 8 bytes.
    pub fn generate_256(labels: HashMap<String, String>) -> Self {
        let key_data = generate_random_bytes(32);
        let key_id = hex_encode(&key_data[0..8]);
        Self {
            key_id,
            key_data,
            algorithm: KeyAlgorithm::Aes256Ctr,
            created_at: SystemTime::now(),
            expires_at: None,
            usage_count: 0,
            max_usage: None,
            labels,
        }
    }

    /// Return `true` if this key is still valid (not expired, not over usage limit).
    pub fn is_valid(&self) -> bool {
        // Check expiry
        if let Some(exp) = self.expires_at {
            if SystemTime::now() >= exp {
                return false;
            }
        }
        // Check usage cap
        if let Some(max) = self.max_usage {
            if self.usage_count >= max {
                return false;
            }
        }
        true
    }

    /// Increment the usage counter by one.
    pub fn record_usage(&mut self) {
        self.usage_count = self.usage_count.saturating_add(1);
    }

    /// Return the key material as a lowercase hex string.
    pub fn to_hex(&self) -> String {
        hex_encode(&self.key_data)
    }

    /// Set an expiry as a `Duration` from now.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.expires_at = SystemTime::now().checked_add(ttl);
        self
    }

    /// Set an explicit expiry time.
    pub fn with_expires_at(mut self, expires_at: SystemTime) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Set a maximum usage count.
    pub fn with_max_usage(mut self, max: u64) -> Self {
        self.max_usage = Some(max);
        self
    }
}

// ---------------------------------------------------------------------------
// KeyStore
// ---------------------------------------------------------------------------

/// Errors returned by `KeyStore` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyStoreError {
    /// The store has reached its maximum capacity.
    StoreFull,
    /// A key with the given ID was not found.
    KeyNotFound(String),
}

impl std::fmt::Display for KeyStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyStoreError::StoreFull => write!(f, "key store is at maximum capacity"),
            KeyStoreError::KeyNotFound(id) => write!(f, "key not found: {}", id),
        }
    }
}

impl std::error::Error for KeyStoreError {}

/// In-memory content key store with a configurable capacity limit.
#[derive(Debug)]
pub struct KeyStore {
    keys: HashMap<String, ContentKey>,
    max_keys: usize,
}

impl KeyStore {
    /// Create a new `KeyStore` with the given maximum key count.
    pub fn new(max_keys: usize) -> Self {
        Self {
            keys: HashMap::new(),
            max_keys,
        }
    }

    /// Store a `ContentKey`.
    ///
    /// Returns `Err(KeyStoreError::StoreFull)` if the store is already at
    /// capacity and no key with the same ID exists (upsert is allowed).
    pub fn store(&mut self, key: ContentKey) -> Result<(), KeyStoreError> {
        // Allow overwriting an existing key (upsert)
        if self.keys.contains_key(&key.key_id) {
            self.keys.insert(key.key_id.clone(), key);
            return Ok(());
        }
        if self.keys.len() >= self.max_keys {
            return Err(KeyStoreError::StoreFull);
        }
        self.keys.insert(key.key_id.clone(), key);
        Ok(())
    }

    /// Retrieve an immutable reference to a key by its ID.
    pub fn get(&self, key_id: &str) -> Option<&ContentKey> {
        self.keys.get(key_id)
    }

    /// Retrieve a mutable reference to a key by its ID.
    pub fn get_mut(&mut self, key_id: &str) -> Option<&mut ContentKey> {
        self.keys.get_mut(key_id)
    }

    /// Revoke (delete) a key by its ID. Returns `true` if the key existed.
    pub fn revoke(&mut self, key_id: &str) -> bool {
        self.keys.remove(key_id).is_some()
    }

    /// Remove all expired keys. Returns the number of keys removed.
    pub fn prune_expired(&mut self) -> usize {
        let now = SystemTime::now();
        let before = self.keys.len();
        self.keys.retain(|_, k| {
            if let Some(exp) = k.expires_at {
                now < exp
            } else {
                true
            }
        });
        before - self.keys.len()
    }

    /// Return a list of all valid (not expired, not over usage cap) keys.
    pub fn active_keys(&self) -> Vec<&ContentKey> {
        self.keys.values().filter(|k| k.is_valid()).collect()
    }

    /// Return the total number of stored keys (including expired ones).
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Return `true` when no keys are stored.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Export a JSON summary of all keys.
    ///
    /// **Security note**: raw key bytes are NOT included. Only metadata is exported.
    pub fn export_keystore(&self) -> String {
        let mut entries = Vec::with_capacity(self.keys.len());
        let mut sorted_keys: Vec<&ContentKey> = self.keys.values().collect();
        // Sort by key_id for deterministic output
        sorted_keys.sort_by(|a, b| a.key_id.cmp(&b.key_id));

        for k in sorted_keys {
            let created = system_time_to_iso(&k.created_at);
            let expires = k
                .expires_at
                .as_ref()
                .map(system_time_to_iso)
                .unwrap_or_else(|| "null".to_string());

            let expires_json = if expires == "null" {
                "null".to_string()
            } else {
                format!("\"{}\"", expires)
            };

            entries.push(format!(
                "  {{\"key_id\":\"{}\",\"algorithm\":\"{}\",\"created_at\":\"{}\",\"expires_at\":{}}}",
                escape_json(&k.key_id),
                escape_json(k.algorithm.name()),
                created,
                expires_json,
            ));
        }

        format!("[\n{}\n]", entries.join(",\n"))
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Encode bytes as a lowercase hex string.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Escape a string for JSON output (minimal: only backslash and double-quote).
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Generate `n` pseudo-random bytes using a xorshift64 PRNG seeded from SystemTime.
///
/// **NOT cryptographically secure** — use a CSPRNG in production.
fn generate_random_bytes(n: usize) -> Vec<u8> {
    // Seed from nanoseconds since UNIX epoch, with a fixed mixer if time is unavailable.
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 ^ (d.as_secs().wrapping_mul(0x9e3779b97f4a7c15)))
        .unwrap_or(0xdeadbeef_cafebabe);

    let mut state = if seed == 0 { 0x1234567890abcdef } else { seed };
    let mut out = Vec::with_capacity(n);

    while out.len() < n {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // splitmix64 finaliser for better bit distribution
        let mut z = state.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }

    out.truncate(n);
    out
}

/// Format a `SystemTime` as an ISO 8601 date-time string (UTC, second precision).
fn system_time_to_iso(t: &SystemTime) -> String {
    match t.duration_since(UNIX_EPOCH) {
        Ok(dur) => {
            let secs = dur.as_secs();
            // Very minimal ISO 8601: compute date components from Unix timestamp
            let (year, month, day, hour, min, sec) = unix_secs_to_datetime(secs);
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                year, month, day, hour, min, sec
            )
        }
        Err(_) => "1970-01-01T00:00:00Z".to_string(),
    }
}

/// Decompose a Unix timestamp (seconds) into (year, month, day, hour, min, sec).
fn unix_secs_to_datetime(secs: u64) -> (u32, u32, u32, u32, u32, u32) {
    let sec = (secs % 60) as u32;
    let mins_total = secs / 60;
    let min = (mins_total % 60) as u32;
    let hours_total = mins_total / 60;
    let hour = (hours_total % 24) as u32;
    let days_total = (hours_total / 24) as u32;

    // Days since 1970-01-01
    // Using the proleptic Gregorian calendar algorithm
    let z = days_total + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    (y, m, d, hour, min, sec)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_generate_128_key_length() {
        let key = ContentKey::generate_128(HashMap::new());
        assert_eq!(key.key_data.len(), 16);
        assert_eq!(key.algorithm, KeyAlgorithm::Aes128Ctr);
    }

    #[test]
    fn test_generate_256_key_length() {
        let key = ContentKey::generate_256(HashMap::new());
        assert_eq!(key.key_data.len(), 32);
        assert_eq!(key.algorithm, KeyAlgorithm::Aes256Ctr);
    }

    #[test]
    fn test_key_id_is_hex_of_first_8_bytes() {
        let key = ContentKey::generate_128(HashMap::new());
        // key_id must be 16 hex chars (8 bytes)
        assert_eq!(key.key_id.len(), 16);
        assert!(key.key_id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_is_valid_fresh_key() {
        let key = ContentKey::generate_128(HashMap::new());
        assert!(key.is_valid());
    }

    #[test]
    fn test_is_valid_expired_key() {
        let key = ContentKey::generate_128(HashMap::new())
            // expired 1 second before the epoch — always in the past
            .with_expires_at(UNIX_EPOCH + Duration::from_secs(1));
        assert!(!key.is_valid());
    }

    #[test]
    fn test_is_valid_usage_exhausted() {
        let mut key = ContentKey::generate_128(HashMap::new()).with_max_usage(2);
        key.record_usage();
        key.record_usage();
        assert!(!key.is_valid());
    }

    #[test]
    fn test_record_usage_increments() {
        let mut key = ContentKey::generate_128(HashMap::new());
        assert_eq!(key.usage_count, 0);
        key.record_usage();
        key.record_usage();
        assert_eq!(key.usage_count, 2);
    }

    #[test]
    fn test_to_hex_length() {
        let key = ContentKey::generate_128(HashMap::new());
        // 16 bytes → 32 hex chars
        assert_eq!(key.to_hex().len(), 32);
        let key256 = ContentKey::generate_256(HashMap::new());
        // 32 bytes → 64 hex chars
        assert_eq!(key256.to_hex().len(), 64);
    }

    #[test]
    fn test_key_store_new() {
        let store = KeyStore::new(10);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_key_store_store_and_get() {
        let mut store = KeyStore::new(10);
        let key = ContentKey::generate_128(HashMap::new());
        let id = key.key_id.clone();
        store.store(key).expect("store should succeed");
        assert!(store.get(&id).is_some());
    }

    #[test]
    fn test_key_store_full_error() {
        let mut store = KeyStore::new(1);
        let k1 = ContentKey::generate_128(HashMap::new());
        // Force unique IDs by using distinct key_data seeds via brute force
        let mut k2 = ContentKey::generate_256(HashMap::new());
        // Ensure k2 has a different key_id from k1
        while k2.key_id == k1.key_id {
            k2 = ContentKey::generate_256(HashMap::new());
        }
        store.store(k1).expect("first store should succeed");
        let result = store.store(k2);
        assert_eq!(result, Err(KeyStoreError::StoreFull));
    }

    #[test]
    fn test_key_store_revoke() {
        let mut store = KeyStore::new(10);
        let key = ContentKey::generate_128(HashMap::new());
        let id = key.key_id.clone();
        store.store(key).expect("store should succeed");
        assert!(store.revoke(&id));
        assert!(store.get(&id).is_none());
        assert!(!store.revoke(&id)); // second revoke returns false
    }

    #[test]
    fn test_key_store_prune_expired() {
        let mut store = KeyStore::new(10);
        // Already-expired key
        let expired = ContentKey::generate_128(HashMap::new())
            .with_expires_at(UNIX_EPOCH + Duration::from_secs(1));
        // Valid key with TTL in the future
        let valid = ContentKey::generate_256(HashMap::new()).with_ttl(Duration::from_hours(1));

        store.store(expired).expect("store expired should succeed");
        store.store(valid).expect("store valid should succeed");
        assert_eq!(store.len(), 2);

        let pruned = store.prune_expired();
        assert_eq!(pruned, 1);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_key_store_active_keys() {
        let mut store = KeyStore::new(10);
        let expired = ContentKey::generate_128(HashMap::new())
            .with_expires_at(UNIX_EPOCH + Duration::from_secs(1));
        let valid = ContentKey::generate_256(HashMap::new());

        store.store(expired).expect("store should succeed");
        store.store(valid).expect("store should succeed");

        let active = store.active_keys();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn test_export_keystore_is_valid_json_like() {
        let mut store = KeyStore::new(10);
        let key = ContentKey::generate_128(HashMap::new());
        store.store(key).expect("store should succeed");

        let json = store.export_keystore();
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("key_id"));
        assert!(json.contains("algorithm"));
        assert!(json.contains("AES-128-CTR"));
        // Must NOT contain raw key bytes (no "key_data")
        assert!(!json.contains("key_data"));
    }
}
