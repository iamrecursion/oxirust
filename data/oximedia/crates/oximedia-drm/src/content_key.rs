//! Content key derivation and management for DRM-protected media.
//!
//! Provides key derivation functions (KDF), content key containers, and
//! key hierarchy support for multi-key encryption schemes such as those
//! used in CENC (Common Encryption).

use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Key derivation
// ---------------------------------------------------------------------------

/// Algorithm used to derive content keys from a master key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyDerivationAlgorithm {
    /// HMAC-SHA256 based derivation.
    HmacSha256,
    /// HKDF-SHA256 (RFC 5869).
    HkdfSha256,
    /// CMAC-AES-128 based derivation.
    CmacAes128,
    /// No derivation -- the master key *is* the content key.
    Identity,
}

impl fmt::Display for KeyDerivationAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HmacSha256 => write!(f, "HMAC-SHA256"),
            Self::HkdfSha256 => write!(f, "HKDF-SHA256"),
            Self::CmacAes128 => write!(f, "CMAC-AES-128"),
            Self::Identity => write!(f, "Identity"),
        }
    }
}

/// Parameters for key derivation.
#[derive(Debug, Clone)]
pub struct KeyDerivationParams {
    /// Algorithm to use.
    pub algorithm: KeyDerivationAlgorithm,
    /// Salt bytes (optional; may be empty).
    pub salt: Vec<u8>,
    /// Info / context bytes used in HKDF-style derivation.
    pub info: Vec<u8>,
    /// Desired output key length in bytes.
    pub output_len: usize,
}

impl KeyDerivationParams {
    /// Create new derivation parameters.
    pub fn new(algorithm: KeyDerivationAlgorithm, output_len: usize) -> Self {
        Self {
            algorithm,
            salt: Vec::new(),
            info: Vec::new(),
            output_len,
        }
    }

    /// Builder: set the salt.
    pub fn with_salt(mut self, salt: Vec<u8>) -> Self {
        self.salt = salt;
        self
    }

    /// Builder: set the info / context bytes.
    pub fn with_info(mut self, info: Vec<u8>) -> Self {
        self.info = info;
        self
    }
}

/// Derive a content key from a master key using the given parameters.
///
/// This is a simplified, non-cryptographic derivation for structural purposes.
/// Real implementations would delegate to a hardware security module or
/// a proper crypto library.
#[allow(clippy::cast_precision_loss)]
pub fn derive_content_key(master_key: &[u8], params: &KeyDerivationParams) -> Vec<u8> {
    match params.algorithm {
        KeyDerivationAlgorithm::Identity => {
            let mut out = master_key.to_vec();
            out.truncate(params.output_len);
            out.resize(params.output_len, 0);
            out
        }
        _ => {
            // Simplified HMAC-style: XOR rounds with salt + info.
            let mut result = vec![0u8; params.output_len];
            for (i, byte) in result.iter_mut().enumerate() {
                let mk = master_key
                    .get(i % master_key.len().max(1))
                    .copied()
                    .unwrap_or(0);
                let s = params
                    .salt
                    .get(i % params.salt.len().max(1))
                    .copied()
                    .unwrap_or(0);
                let inf = params
                    .info
                    .get(i % params.info.len().max(1))
                    .copied()
                    .unwrap_or(0);
                *byte = mk ^ s ^ inf ^ (i as u8).wrapping_mul(0x9d);
            }
            result
        }
    }
}

// ---------------------------------------------------------------------------
// Content key types
// ---------------------------------------------------------------------------

/// The intended usage of a content key within a multi-key scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentKeyUsage {
    /// Encrypts video elementary stream data.
    Video,
    /// Encrypts audio elementary stream data.
    Audio,
    /// Encrypts subtitle / timed-text data.
    Subtitle,
    /// Encrypts all tracks (single-key mode).
    AllTracks,
    /// A custom usage label.
    Custom(u32),
}

impl fmt::Display for ContentKeyUsage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Video => write!(f, "video"),
            Self::Audio => write!(f, "audio"),
            Self::Subtitle => write!(f, "subtitle"),
            Self::AllTracks => write!(f, "all"),
            Self::Custom(id) => write!(f, "custom-{id}"),
        }
    }
}

/// A single content encryption key with its metadata.
///
/// The `cached_schedule` field holds a lazily-initialised AES-128 key schedule
/// backed by [`OnceLock`].  The first call to [`ContentKey::aes128`] performs
/// key expansion (≈ 10 round-key derivations) and caches the result; all
/// subsequent calls return the pre-computed cipher at zero cost.
///
/// `aes::Aes128` is `Send + Sync` in `aes 0.8+`, so sharing a `ContentKey`
/// across threads is safe.
pub struct ContentKey {
    /// Unique key identifier (typically 16 bytes / UUID).
    pub key_id: Vec<u8>,
    /// The raw key material.
    pub key_data: Vec<u8>,
    /// Intended usage of the key.
    pub usage: ContentKeyUsage,
    /// Initialization vector length in bytes (typically 8 or 16).
    pub iv_size: u8,
    /// If `true`, the subsample encryption pattern is used (CENC / CBCS).
    pub subsample_encryption: bool,
    /// Cached AES-128 cipher initialised from `key_data` on first access.
    cached_schedule: OnceLock<aes::Aes128>,
}

impl std::fmt::Debug for ContentKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContentKey")
            .field("key_id", &self.key_id)
            .field("usage", &self.usage)
            .field("iv_size", &self.iv_size)
            .field("subsample_encryption", &self.subsample_encryption)
            .field(
                "cached_schedule",
                &self
                    .cached_schedule
                    .get()
                    .map(|_| "<initialised>")
                    .unwrap_or("<uninit>"),
            )
            .finish()
    }
}

impl Clone for ContentKey {
    fn clone(&self) -> Self {
        // OnceLock cannot be cloned directly; reset the cache in the clone so
        // it is re-initialised on first use.  The key material is identical so
        // the result will be the same.
        Self {
            key_id: self.key_id.clone(),
            key_data: self.key_data.clone(),
            usage: self.usage,
            iv_size: self.iv_size,
            subsample_encryption: self.subsample_encryption,
            cached_schedule: OnceLock::new(),
        }
    }
}

impl ContentKey {
    /// Create a new content key.
    pub fn new(key_id: Vec<u8>, key_data: Vec<u8>, usage: ContentKeyUsage) -> Self {
        Self {
            key_id,
            key_data,
            usage,
            iv_size: 16,
            subsample_encryption: false,
            cached_schedule: OnceLock::new(),
        }
    }

    /// Builder: set the IV size.
    pub fn with_iv_size(mut self, size: u8) -> Self {
        self.iv_size = size;
        self
    }

    /// Builder: enable or disable subsample encryption.
    pub fn with_subsample_encryption(mut self, enabled: bool) -> Self {
        self.subsample_encryption = enabled;
        self
    }

    /// Returns the key length in bits.
    #[allow(clippy::cast_precision_loss)]
    pub fn key_length_bits(&self) -> usize {
        self.key_data.len() * 8
    }

    /// Returns `true` if the key material is all zeros (which is almost
    /// certainly an error in production).
    pub fn is_zero_key(&self) -> bool {
        self.key_data.iter().all(|&b| b == 0)
    }

    /// Return a reference to the cached AES-128 cipher derived from `key_data`.
    ///
    /// The key schedule is computed at most once per `ContentKey` instance
    /// (thread-safely via [`OnceLock`]).  Subsequent calls return the cached
    /// cipher without any computation.
    ///
    /// # Errors
    ///
    /// Returns [`crate::DrmError::InvalidKey`] if `key_data` is not exactly
    /// 16 bytes (required by AES-128).
    pub fn aes128(&self) -> crate::Result<&aes::Aes128> {
        use aes::cipher::KeyInit;

        if self.key_data.len() != 16 {
            return Err(crate::DrmError::InvalidKey(format!(
                "AES-128 requires a 16-byte key, got {} bytes",
                self.key_data.len()
            )));
        }
        // SAFETY: key length validated above; the closure never fails.
        let cipher = self.cached_schedule.get_or_init(|| {
            aes::Aes128::new_from_slice(&self.key_data)
                .expect("key length validated as 16 bytes above; infallible")
        });
        Ok(cipher)
    }
}

// ---------------------------------------------------------------------------
// Content key set
// ---------------------------------------------------------------------------

/// A collection of content keys used to protect a single piece of content.
///
/// Multi-key schemes use different keys for different track types (video,
/// audio, subtitle). This structure manages the full set and provides
/// lookup by key-ID or usage.
#[derive(Debug, Clone)]
pub struct ContentKeySet {
    /// Human-readable label for the key set.
    pub label: String,
    /// The keys in the set, keyed by their key-ID bytes.
    keys: HashMap<Vec<u8>, ContentKey>,
    /// Lookup from usage to key-ID for fast access.
    usage_map: HashMap<ContentKeyUsage, Vec<u8>>,
}

impl ContentKeySet {
    /// Create a new, empty key set.
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            keys: HashMap::new(),
            usage_map: HashMap::new(),
        }
    }

    /// Add a key to the set. If a key with the same `key_id` already exists
    /// it is replaced.
    pub fn add_key(&mut self, key: ContentKey) {
        self.usage_map.insert(key.usage, key.key_id.clone());
        self.keys.insert(key.key_id.clone(), key);
    }

    /// Remove a key by its key-ID.
    pub fn remove_key(&mut self, key_id: &[u8]) -> Option<ContentKey> {
        if let Some(removed) = self.keys.remove(key_id) {
            self.usage_map.retain(|_, v| v != key_id);
            Some(removed)
        } else {
            None
        }
    }

    /// Look up a key by its key-ID.
    pub fn get_by_id(&self, key_id: &[u8]) -> Option<&ContentKey> {
        self.keys.get(key_id)
    }

    /// Look up a key by its intended usage.
    pub fn get_by_usage(&self, usage: ContentKeyUsage) -> Option<&ContentKey> {
        self.usage_map.get(&usage).and_then(|id| self.keys.get(id))
    }

    /// Returns the number of keys in the set.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Returns `true` if the set contains no keys.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Returns an iterator over all keys in the set.
    pub fn keys(&self) -> impl Iterator<Item = &ContentKey> {
        self.keys.values()
    }

    /// Returns all key-IDs in the set.
    pub fn key_ids(&self) -> Vec<Vec<u8>> {
        self.keys.keys().cloned().collect()
    }

    /// Validate that the set has at least one key and that no key is a
    /// zero-key. Returns a list of problems (empty if valid).
    pub fn validate(&self) -> Vec<String> {
        let mut problems = Vec::new();
        if self.keys.is_empty() {
            problems.push("key set is empty".to_string());
        }
        for key in self.keys.values() {
            if key.is_zero_key() {
                problems.push(format!("key {:?} is all zeros", key.key_id));
            }
            if key.key_data.is_empty() {
                problems.push(format!("key {:?} has empty key data", key.key_id));
            }
        }
        problems
    }
}

// ---------------------------------------------------------------------------
// Key rotation handle
// ---------------------------------------------------------------------------

/// Tracks the current and next content key during a key rotation event.
#[derive(Debug, Clone)]
pub struct KeyRotationHandle {
    /// The currently active key set.
    pub current: ContentKeySet,
    /// The next key set that will become active after rotation.
    pub next: Option<ContentKeySet>,
    /// Rotation period in seconds.
    pub rotation_period_s: u64,
    /// Seconds remaining until the next rotation.
    pub seconds_until_rotation: u64,
}

impl KeyRotationHandle {
    /// Create a new rotation handle.
    pub fn new(current: ContentKeySet, rotation_period_s: u64) -> Self {
        Self {
            current,
            next: None,
            rotation_period_s,
            seconds_until_rotation: rotation_period_s,
        }
    }

    /// Set the next key set that will become active.
    pub fn set_next(&mut self, next: ContentKeySet) {
        self.next = Some(next);
    }

    /// Perform the rotation: `next` becomes `current`, and `next` is cleared.
    /// Returns `true` if the rotation was performed.
    pub fn rotate(&mut self) -> bool {
        if let Some(next) = self.next.take() {
            self.current = next;
            self.seconds_until_rotation = self.rotation_period_s;
            true
        } else {
            false
        }
    }

    /// Tick the rotation clock by `elapsed_s` seconds. If the timer expires
    /// and a `next` key set is available, the rotation is performed
    /// automatically. Returns `true` if a rotation happened.
    pub fn tick(&mut self, elapsed_s: u64) -> bool {
        if elapsed_s >= self.seconds_until_rotation {
            self.seconds_until_rotation = 0;
            self.rotate()
        } else {
            self.seconds_until_rotation -= elapsed_s;
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_derivation_identity() {
        let master = vec![0xAA; 16];
        let params = KeyDerivationParams::new(KeyDerivationAlgorithm::Identity, 16);
        let derived = derive_content_key(&master, &params);
        assert_eq!(derived, master);
    }

    #[test]
    fn test_key_derivation_identity_truncate() {
        let master = vec![0xBB; 32];
        let params = KeyDerivationParams::new(KeyDerivationAlgorithm::Identity, 16);
        let derived = derive_content_key(&master, &params);
        assert_eq!(derived.len(), 16);
        assert_eq!(&derived[..], &master[..16]);
    }

    #[test]
    fn test_key_derivation_identity_pad() {
        let master = vec![0xCC; 8];
        let params = KeyDerivationParams::new(KeyDerivationAlgorithm::Identity, 16);
        let derived = derive_content_key(&master, &params);
        assert_eq!(derived.len(), 16);
        assert_eq!(&derived[..8], &master[..]);
        assert!(derived[8..].iter().all(|&b| b == 0));
    }

    #[test]
    fn test_key_derivation_hmac_produces_output() {
        let master = vec![1, 2, 3, 4];
        let params = KeyDerivationParams::new(KeyDerivationAlgorithm::HmacSha256, 16)
            .with_salt(vec![5, 6])
            .with_info(vec![7, 8]);
        let derived = derive_content_key(&master, &params);
        assert_eq!(derived.len(), 16);
        // Not all zeros
        assert!(!derived.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_content_key_new() {
        let key = ContentKey::new(vec![1, 2], vec![3, 4, 5, 6], ContentKeyUsage::Video);
        assert_eq!(key.key_id, vec![1, 2]);
        assert_eq!(key.usage, ContentKeyUsage::Video);
        assert_eq!(key.iv_size, 16);
        assert!(!key.subsample_encryption);
    }

    #[test]
    fn test_content_key_builders() {
        let key = ContentKey::new(vec![1], vec![2], ContentKeyUsage::Audio)
            .with_iv_size(8)
            .with_subsample_encryption(true);
        assert_eq!(key.iv_size, 8);
        assert!(key.subsample_encryption);
    }

    #[test]
    fn test_content_key_length_bits() {
        let key = ContentKey::new(vec![0], vec![0xAA; 16], ContentKeyUsage::AllTracks);
        assert_eq!(key.key_length_bits(), 128);
    }

    #[test]
    fn test_content_key_is_zero_key() {
        let zero = ContentKey::new(vec![1], vec![0; 16], ContentKeyUsage::Video);
        assert!(zero.is_zero_key());
        let non_zero = ContentKey::new(vec![1], vec![1; 16], ContentKeyUsage::Video);
        assert!(!non_zero.is_zero_key());
    }

    #[test]
    fn test_content_key_set_add_and_lookup() {
        let mut set = ContentKeySet::new("test-set");
        let video = ContentKey::new(vec![10], vec![0xAA; 16], ContentKeyUsage::Video);
        let audio = ContentKey::new(vec![20], vec![0xBB; 16], ContentKeyUsage::Audio);
        set.add_key(video);
        set.add_key(audio);

        assert_eq!(set.len(), 2);
        assert!(!set.is_empty());
        assert!(set.get_by_id(&[10]).is_some());
        assert!(set.get_by_usage(ContentKeyUsage::Audio).is_some());
        assert!(set.get_by_usage(ContentKeyUsage::Subtitle).is_none());
    }

    #[test]
    fn test_content_key_set_remove() {
        let mut set = ContentKeySet::new("rm-test");
        set.add_key(ContentKey::new(vec![1], vec![2], ContentKeyUsage::Video));
        assert_eq!(set.len(), 1);
        let removed = set.remove_key(&[1]);
        assert!(removed.is_some());
        assert!(set.is_empty());
    }

    #[test]
    fn test_content_key_set_validate_empty() {
        let set = ContentKeySet::new("empty");
        let problems = set.validate();
        assert!(!problems.is_empty());
    }

    #[test]
    fn test_content_key_set_validate_zero_key() {
        let mut set = ContentKeySet::new("z");
        set.add_key(ContentKey::new(
            vec![1],
            vec![0; 16],
            ContentKeyUsage::Video,
        ));
        let problems = set.validate();
        assert!(problems.iter().any(|p| p.contains("all zeros")));
    }

    #[test]
    fn test_key_rotation_handle_rotate() {
        let set1 = ContentKeySet::new("first");
        let set2 = ContentKeySet::new("second");
        let mut handle = KeyRotationHandle::new(set1, 300);
        handle.set_next(set2);
        assert!(handle.rotate());
        assert_eq!(handle.current.label, "second");
        assert!(handle.next.is_none());
    }

    #[test]
    fn test_key_rotation_handle_tick() {
        let set1 = ContentKeySet::new("a");
        let mut set2 = ContentKeySet::new("b");
        set2.add_key(ContentKey::new(vec![1], vec![2], ContentKeyUsage::Video));
        let mut handle = KeyRotationHandle::new(set1, 100);
        handle.set_next(set2);

        // Not yet expired
        assert!(!handle.tick(50));
        assert_eq!(handle.seconds_until_rotation, 50);

        // Expire
        assert!(handle.tick(50));
        assert_eq!(handle.current.label, "b");
    }

    #[test]
    fn test_key_rotation_no_next_does_not_rotate() {
        let set1 = ContentKeySet::new("only");
        let mut handle = KeyRotationHandle::new(set1, 60);
        assert!(!handle.rotate());
        assert_eq!(handle.current.label, "only");
    }

    /// Verifies that `aes128()` returns a stable cached value across many calls.
    #[test]
    fn test_content_key_aes_schedule_cached() {
        let key_data = vec![0xA5u8; 16];
        let key = ContentKey::new(vec![1, 2, 3], key_data, ContentKeyUsage::Video);

        // Call 1000 times; all must succeed and the pointer must be identical
        // (proving the OnceLock is not re-initialised).
        let first_ptr = std::ptr::from_ref(key.aes128().expect("aes128 should succeed"));
        for _ in 0..999 {
            let ptr = std::ptr::from_ref(key.aes128().expect("aes128 should succeed"));
            assert_eq!(
                ptr, first_ptr,
                "aes128() must return the same cached object on every call"
            );
        }
    }

    /// Verifies that an invalid key length returns an error.
    #[test]
    fn test_content_key_aes128_invalid_key_length() {
        let key = ContentKey::new(vec![0], vec![0xBBu8; 12], ContentKeyUsage::Audio);
        let err = key.aes128();
        assert!(err.is_err(), "12-byte key must fail for AES-128");
    }

    #[test]
    fn test_content_key_usage_display() {
        assert_eq!(ContentKeyUsage::Video.to_string(), "video");
        assert_eq!(ContentKeyUsage::Audio.to_string(), "audio");
        assert_eq!(ContentKeyUsage::Subtitle.to_string(), "subtitle");
        assert_eq!(ContentKeyUsage::AllTracks.to_string(), "all");
        assert_eq!(ContentKeyUsage::Custom(42).to_string(), "custom-42");
    }

    #[test]
    fn test_key_derivation_algorithm_display() {
        assert_eq!(
            KeyDerivationAlgorithm::HmacSha256.to_string(),
            "HMAC-SHA256"
        );
        assert_eq!(KeyDerivationAlgorithm::Identity.to_string(), "Identity");
    }

    #[test]
    fn test_content_key_set_key_ids() {
        let mut set = ContentKeySet::new("ids");
        set.add_key(ContentKey::new(vec![1], vec![2], ContentKeyUsage::Video));
        set.add_key(ContentKey::new(vec![3], vec![4], ContentKeyUsage::Audio));
        let ids = set.key_ids();
        assert_eq!(ids.len(), 2);
    }
}
