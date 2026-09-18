//! OxiMedia DRM and Encryption Support
//!
//! This crate provides DRM (Digital Rights Management) and encryption support
//! for OxiMedia streaming, including:
//! - CENC (Common Encryption) implementation
//! - Widevine DRM
//! - PlayReady DRM
//! - FairPlay Streaming
//! - W3C Clear Key
//!
//! # DRM System Comparison
//!
//! | System    | UUID                                   | License Protocol         | Key Container     | Supported Schemes  | Platforms                   | Robustness Levels          | Library Status                  |
//! |-----------|----------------------------------------|--------------------------|-------------------|--------------------|-----------------------------|----------------------------|---------------------------------|
//! | Widevine  | edef8ba9-79d6-4ace-a3c8-27dcd51d21ed | Binary protobuf over HTTP| WidevineCdm keys  | cenc, cbcs         | Android, Chrome, Smart TVs  | L1 (TEE), L2 (SW+DRM), L3 | Full RPC + structural CDM       |
//! | PlayReady | 9a04f079-9840-4286-ab92-e65be0885f95  | WS-Trust 1.3 SOAP/XML   | XMR license chain | cenc, cbcs, cbc1   | Windows, Xbox, Smart TVs    | SL150, SL2000, SL3000      | Full RPC + structural CDM       |
//! | FairPlay  | 94ce86fb-07ff-4f43-adb8-93d2fa968ca2  | JSON/KSM over HTTPS      | CKC binary blob   | cbcs               | Apple (iOS/macOS/tvOS)      | HW-bound (Secure Enclave)  | Full RPC + structural CDM       |
//! | ClearKey  | 1077efec-c0b2-4d02-ace3-3c1e52e2fb4b  | W3C JSON (EME)           | JSON key/ID pairs | cenc               | All browsers (testing only) | None (no-op)               | Full implementation             |
//!
//! ## Feature Flags
//!
//! - `widevine` — enables the `widevine` module and HTTP license transport (`HyperPlainLicenseClient`).
//! - `widevine-network` — extends `widevine` with pure-Rust TLS (`HyperRustlsLicenseClient`).
//! - `playready` — enables the `playready` module and SOAP license transport (`HyperPlainPlayReadyClient`).
//! - `playready-network` — extends `playready` with pure-Rust TLS (`HyperRustlsPlayReadyClient`).
//! - `fairplay` — enables the `fairplay` module and JSON/KSM transport (`HyperPlainFairPlayClient`).
//! - `fairplay-network` — extends `fairplay` with pure-Rust TLS (`HyperRustlsFairPlayClient`).
//! - `clearkey` (default) — enables [`clearkey`] module with W3C ClearKey JSON format support.
//! - `hardware-aes` — documents AES-NI runtime auto-detection (the `aes` crate handles dispatch transparently).

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::OnceLock;
use thiserror::Error;
use uuid::Uuid;

// Hardcoded DRM system UUIDs as compile-time constants (no parsing, no panics).
const WIDEVINE_UUID: Uuid = Uuid::from_bytes([
    0xed, 0xef, 0x8b, 0xa9, 0x79, 0xd6, 0x4a, 0xce, 0xa3, 0xc8, 0x27, 0xdc, 0xd5, 0x1d, 0x21, 0xed,
]);
const PLAYREADY_UUID: Uuid = Uuid::from_bytes([
    0x9a, 0x04, 0xf0, 0x79, 0x98, 0x40, 0x42, 0x86, 0xab, 0x92, 0xe6, 0x5b, 0xe0, 0x88, 0x5f, 0x95,
]);
const FAIRPLAY_UUID: Uuid = Uuid::from_bytes([
    0x94, 0xce, 0x86, 0xfb, 0x07, 0xff, 0x4f, 0x43, 0xad, 0xb8, 0x93, 0xd2, 0xfa, 0x96, 0x8c, 0xa2,
]);
const CLEARKEY_UUID: Uuid = Uuid::from_bytes([
    0x10, 0x77, 0xef, 0xec, 0xc0, 0xb2, 0x4d, 0x02, 0xac, 0xe3, 0x3c, 0x1e, 0x52, 0xe2, 0xfb, 0x4b,
]);

// ---------------------------------------------------------------------------
// Cached UUID accessors (D4)
// ---------------------------------------------------------------------------

/// Returns a reference to the cached Widevine DRM system UUID.
///
/// The UUID `edef8ba9-79d6-4ace-a3c8-27dcd51d21ed` is initialized exactly once
/// using [`OnceLock`] and reused on subsequent calls with zero parse overhead.
pub fn widevine_uuid() -> &'static Uuid {
    static CACHE: OnceLock<Uuid> = OnceLock::new();
    CACHE.get_or_init(|| WIDEVINE_UUID)
}

/// Returns a reference to the cached PlayReady DRM system UUID.
///
/// The UUID `9a04f079-9840-4286-ab92-e65be0885f95` is initialized exactly once.
pub fn playready_uuid() -> &'static Uuid {
    static CACHE: OnceLock<Uuid> = OnceLock::new();
    CACHE.get_or_init(|| PLAYREADY_UUID)
}

/// Returns a reference to the cached FairPlay DRM system UUID.
///
/// The UUID `94ce86fb-07ff-4f43-adb8-93d2fa968ca2` is initialized exactly once.
pub fn fairplay_uuid() -> &'static Uuid {
    static CACHE: OnceLock<Uuid> = OnceLock::new();
    CACHE.get_or_init(|| FAIRPLAY_UUID)
}

/// Returns a reference to the cached W3C ClearKey DRM system UUID.
///
/// The UUID `1077efec-c0b2-4d02-ace3-3c1e52e2fb4b` is initialized exactly once.
pub fn clearkey_uuid() -> &'static Uuid {
    static CACHE: OnceLock<Uuid> = OnceLock::new();
    CACHE.get_or_init(|| CLEARKEY_UUID)
}

pub mod access_grant;
pub mod aes_cbc;
pub mod aes_ctr;
pub mod analytics;
pub mod audit_trail;
pub mod buf_pool;
pub mod cenc;
pub mod cenc_scheme;
pub mod cmaf_encrypt;
pub mod compliance;
pub mod content_key;
pub mod cpix;
pub mod dash_cenc;
pub mod device_auth;
pub mod device_fingerprint;
pub mod device_registry;
pub mod drm_token_verifier;
pub mod entitlement;
pub mod geo_fence;
pub mod hw_key_storage;
pub mod hw_key_store;
pub mod key_derivation;
pub mod key_lifecycle;
pub mod key_management;
pub mod key_rotation;
pub mod key_rotation_schedule;
pub mod key_schedule;
pub mod key_wrap;
pub mod license_audit;
pub mod license_chain;
pub mod license_server;
pub mod license_validator;
pub mod managed_license;
pub mod multi_drm;
pub mod multi_key;
pub mod offline;
pub mod output_control;
pub mod playback_policy;
pub mod playback_rules;
pub mod policy;
pub mod policy_engine;
pub mod pssh;
pub mod pssh_parser;
pub mod rate_limit;
pub mod revocation_list;
pub mod session_token;
pub mod sw_secure_enclave;
pub mod sw_tpm;
pub mod token;
pub mod token_bucket_auth;
pub mod watermark_detect;
pub mod watermark_embed;

#[cfg(feature = "clearkey")]
pub mod clearkey;

#[cfg(feature = "widevine")]
pub mod widevine;

#[cfg(feature = "widevine")]
pub mod widevine_rpc;

#[cfg(feature = "widevine")]
pub use widevine_rpc::{HyperPlainLicenseClient, LicenseClient};

#[cfg(all(feature = "widevine", feature = "widevine-network"))]
pub use widevine_rpc::HyperRustlsLicenseClient;

#[cfg(feature = "playready")]
pub mod playready;

#[cfg(feature = "playready")]
pub mod playready_rpc;

#[cfg(feature = "playready")]
pub use playready_rpc::{HyperPlainPlayReadyClient, PlayReadyLicenseClient};

#[cfg(all(feature = "playready", feature = "playready-network"))]
pub use playready_rpc::HyperRustlsPlayReadyClient;

#[cfg(feature = "fairplay")]
pub mod fairplay;

#[cfg(feature = "fairplay")]
pub mod fairplay_rpc;

#[cfg(feature = "fairplay")]
pub use fairplay_rpc::{FairPlayClientExt, FairPlayKeyClient, HyperPlainFairPlayClient};

#[cfg(all(feature = "fairplay", feature = "fairplay-network"))]
pub use fairplay_rpc::HyperRustlsFairPlayClient;

/// DRM-related errors
#[derive(Error, Debug)]
pub enum DrmError {
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    #[error("Decryption error: {0}")]
    DecryptionError(String),

    #[error("Invalid key: {0}")]
    InvalidKey(String),

    #[error("Invalid initialization vector: {0}")]
    InvalidIv(String),

    #[error("License error: {0}")]
    LicenseError(String),

    #[error("License denied by server (status {status}): {body}")]
    LicenseDenied { status: u16, body: String },

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("PSSH parsing error: {0}")]
    PsshError(String),

    #[error("Unsupported DRM system: {0}")]
    UnsupportedDrmSystem(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Base64 decode error: {0}")]
    Base64Error(#[from] base64::DecodeError),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("XML error: {0}")]
    XmlError(String),
}

pub type Result<T> = std::result::Result<T, DrmError>;

/// DRM system identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrmSystem {
    /// Widevine DRM (Google)
    Widevine,
    /// PlayReady DRM (Microsoft)
    PlayReady,
    /// FairPlay Streaming (Apple)
    FairPlay,
    /// W3C Clear Key (for testing)
    ClearKey,
}

impl DrmSystem {
    /// Get the system ID UUID for this DRM system
    pub fn system_id(&self) -> Uuid {
        match self {
            DrmSystem::Widevine => WIDEVINE_UUID,
            DrmSystem::PlayReady => PLAYREADY_UUID,
            DrmSystem::FairPlay => FAIRPLAY_UUID,
            DrmSystem::ClearKey => CLEARKEY_UUID,
        }
    }

    /// Get DRM system from UUID
    pub fn from_uuid(uuid: &Uuid) -> Option<Self> {
        if *uuid == WIDEVINE_UUID {
            Some(DrmSystem::Widevine)
        } else if *uuid == PLAYREADY_UUID {
            Some(DrmSystem::PlayReady)
        } else if *uuid == FAIRPLAY_UUID {
            Some(DrmSystem::FairPlay)
        } else if *uuid == CLEARKEY_UUID {
            Some(DrmSystem::ClearKey)
        } else {
            None
        }
    }
}

impl fmt::Display for DrmSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrmSystem::Widevine => write!(f, "Widevine"),
            DrmSystem::PlayReady => write!(f, "PlayReady"),
            DrmSystem::FairPlay => write!(f, "FairPlay"),
            DrmSystem::ClearKey => write!(f, "ClearKey"),
        }
    }
}

/// DRM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrmConfig {
    /// DRM system to use
    pub system: DrmSystem,
    /// Content key ID
    pub key_id: Vec<u8>,
    /// Content key
    pub key: Vec<u8>,
    /// License server URL
    pub license_url: Option<String>,
    /// Additional system-specific data
    pub system_data: Option<Vec<u8>>,
}

impl DrmConfig {
    /// Create a new DRM configuration
    pub fn new(system: DrmSystem, key_id: Vec<u8>, key: Vec<u8>) -> Self {
        Self {
            system,
            key_id,
            key,
            license_url: None,
            system_data: None,
        }
    }

    /// Set license server URL
    pub fn with_license_url(mut self, url: String) -> Self {
        self.license_url = Some(url);
        self
    }

    /// Set system-specific data
    pub fn with_system_data(mut self, data: Vec<u8>) -> Self {
        self.system_data = Some(data);
        self
    }
}

/// Key provider trait for obtaining content keys
pub trait KeyProvider: Send + Sync {
    /// Get a content key by key ID
    fn get_key(&self, key_id: &[u8]) -> Result<Vec<u8>>;

    /// Get multiple keys at once
    fn get_keys(&self, key_ids: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        key_ids.iter().map(|id| self.get_key(id)).collect()
    }

    /// Check if a key exists
    fn has_key(&self, key_id: &[u8]) -> bool {
        self.get_key(key_id).is_ok()
    }
}

/// Simple in-memory key provider
#[derive(Debug, Clone)]
pub struct MemoryKeyProvider {
    keys: std::collections::HashMap<Vec<u8>, Vec<u8>>,
}

impl MemoryKeyProvider {
    /// Create a new memory key provider
    pub fn new() -> Self {
        Self {
            keys: std::collections::HashMap::new(),
        }
    }

    /// Add a key to the provider
    pub fn add_key(&mut self, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys.insert(key_id, key);
    }

    /// Add multiple keys at once
    pub fn add_keys(&mut self, keys: Vec<(Vec<u8>, Vec<u8>)>) {
        for (key_id, key) in keys {
            self.keys.insert(key_id, key);
        }
    }

    /// Remove a key
    pub fn remove_key(&mut self, key_id: &[u8]) {
        self.keys.remove(key_id);
    }

    /// Clear all keys
    pub fn clear(&mut self) {
        self.keys.clear();
    }

    /// Get number of keys
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if provider is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl Default for MemoryKeyProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyProvider for MemoryKeyProvider {
    fn get_key(&self, key_id: &[u8]) -> Result<Vec<u8>> {
        self.keys
            .get(key_id)
            .cloned()
            .ok_or_else(|| DrmError::InvalidKey("Key not found".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drm_system_uuid() {
        let widevine_uuid =
            Uuid::parse_str("edef8ba9-79d6-4ace-a3c8-27dcd51d21ed").expect("UUID should parse");
        assert_eq!(DrmSystem::Widevine.system_id(), widevine_uuid);

        let playready_uuid =
            Uuid::parse_str("9a04f079-9840-4286-ab92-e65be0885f95").expect("UUID should parse");
        assert_eq!(DrmSystem::PlayReady.system_id(), playready_uuid);

        let fairplay_uuid =
            Uuid::parse_str("94ce86fb-07ff-4f43-adb8-93d2fa968ca2").expect("UUID should parse");
        assert_eq!(DrmSystem::FairPlay.system_id(), fairplay_uuid);

        let clearkey_uuid =
            Uuid::parse_str("1077efec-c0b2-4d02-ace3-3c1e52e2fb4b").expect("UUID should parse");
        assert_eq!(DrmSystem::ClearKey.system_id(), clearkey_uuid);
    }

    #[test]
    fn test_drm_system_from_uuid() {
        let widevine_uuid =
            Uuid::parse_str("edef8ba9-79d6-4ace-a3c8-27dcd51d21ed").expect("UUID should parse");
        assert_eq!(
            DrmSystem::from_uuid(&widevine_uuid),
            Some(DrmSystem::Widevine)
        );

        let unknown_uuid =
            Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("UUID should parse");
        assert_eq!(DrmSystem::from_uuid(&unknown_uuid), None);
    }

    /// Verifies that the cached UUID accessor functions return stable, correct values.
    ///
    /// Each accessor is called twice to exercise the `OnceLock` fast path, and the
    /// results are compared against a byte-level round-trip through `DrmSystem::from_uuid`.
    #[test]
    fn test_drm_system_uuid_cached_values_match() {
        // Call each accessor twice — second call exercises the OnceLock fast path.
        let wv1 = widevine_uuid();
        let wv2 = widevine_uuid();
        assert_eq!(
            wv1, wv2,
            "widevine_uuid must return the same value on repeat calls"
        );
        assert_eq!(DrmSystem::from_uuid(wv1), Some(DrmSystem::Widevine));

        let pr1 = playready_uuid();
        let pr2 = playready_uuid();
        assert_eq!(
            pr1, pr2,
            "playready_uuid must return the same value on repeat calls"
        );
        assert_eq!(DrmSystem::from_uuid(pr1), Some(DrmSystem::PlayReady));

        let fp1 = fairplay_uuid();
        let fp2 = fairplay_uuid();
        assert_eq!(
            fp1, fp2,
            "fairplay_uuid must return the same value on repeat calls"
        );
        assert_eq!(DrmSystem::from_uuid(fp1), Some(DrmSystem::FairPlay));

        let ck1 = clearkey_uuid();
        let ck2 = clearkey_uuid();
        assert_eq!(
            ck1, ck2,
            "clearkey_uuid must return the same value on repeat calls"
        );
        assert_eq!(DrmSystem::from_uuid(ck1), Some(DrmSystem::ClearKey));
    }

    #[test]
    fn test_memory_key_provider() {
        let mut provider = MemoryKeyProvider::new();
        let key_id = vec![1, 2, 3, 4];
        let key = vec![5, 6, 7, 8];

        provider.add_key(key_id.clone(), key.clone());
        assert!(provider.has_key(&key_id));
        assert_eq!(
            provider.get_key(&key_id).expect("get_key should succeed"),
            key
        );

        provider.remove_key(&key_id);
        assert!(!provider.has_key(&key_id));
    }
}
