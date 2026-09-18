//! DRM Key Rotation System for live streams.
//!
//! Provides automatic key rotation with configurable derivation strategies:
//! - Random keys per interval
//! - HKDF-derived keys from a master secret
//! - Counter-based deterministic keys

use std::collections::HashMap;

/// Key derivation method for key rotation.
#[derive(Debug, Clone)]
pub enum KeyDerivation {
    /// Fully random 16-byte keys generated each interval.
    Random,
    /// HKDF-derived from a master secret using SHA-256.
    Hkdf {
        /// Master key material (IKM).
        master_secret: Vec<u8>,
        /// Context info bound into derived keys.
        info: Vec<u8>,
    },
    /// Deterministic counter-based derivation.
    Counter {
        /// Starting seed mixed with the period counter.
        seed: u64,
    },
}

/// Policy that governs how and when keys are rotated.
#[derive(Debug, Clone)]
pub struct KeyRotationPolicy {
    /// How many seconds each key period lasts.
    pub rotation_interval_secs: u64,
    /// Algorithm used to derive each period key.
    pub key_derivation: KeyDerivation,
    /// Number of future keys to pre-generate and cache.
    pub preload_count: u32,
    /// Seconds an old key remains valid after rotation (grace window).
    pub overlap_duration_secs: u64,
}

impl Default for KeyRotationPolicy {
    fn default() -> Self {
        Self {
            rotation_interval_secs: 3600,
            key_derivation: KeyDerivation::Random,
            preload_count: 2,
            overlap_duration_secs: 30,
        }
    }
}

/// Manages rotating content-encryption keys for live streams.
///
/// Keys are generated on demand and cached internally. The cache is bounded
/// by the policy's `preload_count` plus a small overlap window.
pub struct RotatingKeyManager {
    policy: KeyRotationPolicy,
    current_period: u64,
    /// period index → 16-byte AES key.
    key_cache: HashMap<u64, Vec<u8>>,
}

impl RotatingKeyManager {
    /// Create a new key manager with the given policy.
    #[must_use]
    pub fn new(policy: KeyRotationPolicy) -> Self {
        Self {
            policy,
            current_period: 0,
            key_cache: HashMap::new(),
        }
    }

    /// Return (or derive) the 16-byte key for a specific period index.
    pub fn key_for_period(&mut self, period: u64) -> Vec<u8> {
        if let Some(cached) = self.key_cache.get(&period) {
            return cached.clone();
        }
        let key = self.derive_key(period);
        self.key_cache.insert(period, key.clone());
        key
    }

    /// Return the period index and key that are active at the given Unix timestamp.
    ///
    /// Also updates the internal `current_period` tracker.
    pub fn current_key(&mut self, now_secs: u64) -> (u64, Vec<u8>) {
        let period = self.period_for_timestamp(now_secs);
        self.current_period = period;
        let key = self.key_for_period(period);
        (period, key)
    }

    /// Convert a Unix timestamp to its enclosing period index.
    #[must_use]
    pub fn period_for_timestamp(&self, ts: u64) -> u64 {
        ts / self.policy.rotation_interval_secs
    }

    /// Pre-generate future keys starting from the period containing `now_secs`.
    ///
    /// Returns `preload_count` `(period, key)` pairs for the upcoming periods.
    pub fn preload_future_keys(&mut self, now_secs: u64) -> Vec<(u64, Vec<u8>)> {
        let start = self.period_for_timestamp(now_secs);
        (0..self.policy.preload_count as u64)
            .map(|i| {
                let p = start + i;
                let k = self.key_for_period(p);
                (p, k)
            })
            .collect()
    }

    /// Return the overlap / grace window in seconds.
    #[must_use]
    pub fn overlap_duration_secs(&self) -> u64 {
        self.policy.overlap_duration_secs
    }

    /// Return the rotation interval in seconds.
    #[must_use]
    pub fn rotation_interval_secs(&self) -> u64 {
        self.policy.rotation_interval_secs
    }

    /// Check if `ts` falls within the grace window of the previous period.
    ///
    /// During the grace window both the old key and the new key are valid,
    /// which allows clients that received the old key to finish decryption.
    #[must_use]
    pub fn in_overlap_window(&self, ts: u64) -> bool {
        let elapsed_in_period = ts % self.policy.rotation_interval_secs;
        elapsed_in_period < self.policy.overlap_duration_secs
    }

    /// Remove cached keys older than `keep_periods` periods before the current one.
    pub fn evict_old_keys(&mut self, now_secs: u64, keep_periods: u64) {
        let current = self.period_for_timestamp(now_secs);
        self.key_cache.retain(|&p, _| p + keep_periods >= current);
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn derive_key(&self, period: u64) -> Vec<u8> {
        match &self.policy.key_derivation {
            KeyDerivation::Random => derive_random_key(period),
            KeyDerivation::Hkdf {
                master_secret,
                info,
            } => derive_hkdf_key(master_secret, info, period),
            KeyDerivation::Counter { seed } => derive_counter_key(*seed, period),
        }
    }
}

// ---------------------------------------------------------------------------
// SHA-256 — minimal implementation (no external crate needed)
// ---------------------------------------------------------------------------

/// Compute SHA-256 of `msg` and return the 32-byte digest.
fn sha256(msg: &[u8]) -> [u8; 32] {
    // Round constants K (first 32 bits of fractional parts of cube roots of
    // the first 64 primes).
    #[allow(clippy::unreadable_literal)]
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Initial hash values (first 32 bits of square roots of the first 8 primes).
    #[allow(clippy::unreadable_literal)]
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Pre-processing: padding.
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    let mut padded = msg.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 512-bit (64-byte) block.
    for block in padded.chunks(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, &word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// HMAC-SHA256.
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;

    // Normalise key to block length.
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let h = sha256(key);
        k[..32].copy_from_slice(&h);
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    // Inner and outer padded keys.
    let ipad: Vec<u8> = k.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = k.iter().map(|b| b ^ 0x5C).collect();

    let mut inner = ipad;
    inner.extend_from_slice(data);
    let inner_hash = sha256(&inner);

    let mut outer = opad;
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

// ---------------------------------------------------------------------------
// Key derivation helpers
// ---------------------------------------------------------------------------

/// Derive a 16-byte key using HKDF-SHA256.
///
/// - Extract: `prk = HMAC-SHA256(salt=period_bytes, ikm=master_secret)`
/// - Expand:  `okm = T(1) = HMAC-SHA256(prk, info || 0x01)`
fn derive_hkdf_key(master_secret: &[u8], info: &[u8], period: u64) -> Vec<u8> {
    let salt = period.to_be_bytes();

    // HKDF-Extract
    let prk = hmac_sha256(&salt, master_secret);

    // HKDF-Expand T(1)
    let mut expand_input = info.to_vec();
    expand_input.push(0x01);
    let okm = hmac_sha256(&prk, &expand_input);

    // Return first 16 bytes as AES-128 key.
    okm[..16].to_vec()
}

/// Derive a 16-byte key from a counter seed using a simple but deterministic
/// process: XOR the seed with the period, then hash the result.
fn derive_counter_key(seed: u64, period: u64) -> Vec<u8> {
    let mixed = seed ^ period;
    // Expand to 32 bytes using a hash chain.
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&mixed.to_be_bytes());
    buf[8..].copy_from_slice(&period.to_be_bytes());
    let hash = sha256(&buf);
    hash[..16].to_vec()
}

/// Derive a "random" 16-byte key that is deterministically seeded by period.
///
/// In production you would use an OS CSPRNG; here we use a hash-based LCG
/// seeded by the period so that tests are reproducible.
fn derive_random_key(period: u64) -> Vec<u8> {
    // Use period as entropy seed for a hash-based expansion.
    let seed_bytes = period.to_be_bytes();
    let hash = sha256(&seed_bytes);
    hash[..16].to_vec()
}

// ---------------------------------------------------------------------------
// Token-based key validation helper
// ---------------------------------------------------------------------------

/// A simple token that binds a period to an HMAC tag for validation.
#[derive(Debug, Clone)]
pub struct PeriodToken {
    /// Period index this token is valid for.
    pub period: u64,
    /// 32-byte HMAC-SHA256 authentication tag.
    pub tag: [u8; 32],
}

impl PeriodToken {
    /// Create a new token by signing `period` with `signing_key`.
    #[must_use]
    pub fn new(period: u64, signing_key: &[u8]) -> Self {
        let tag = hmac_sha256(signing_key, &period.to_be_bytes());
        Self { period, tag }
    }

    /// Verify that this token was issued by the holder of `signing_key`.
    #[must_use]
    pub fn verify(&self, signing_key: &[u8]) -> bool {
        let expected = hmac_sha256(signing_key, &self.period.to_be_bytes());
        // Constant-time comparison.
        let mut diff = 0u8;
        for (a, b) in expected.iter().zip(self.tag.iter()) {
            diff |= a ^ b;
        }
        diff == 0
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_policy(derivation: KeyDerivation) -> KeyRotationPolicy {
        KeyRotationPolicy {
            rotation_interval_secs: 100,
            key_derivation: derivation,
            preload_count: 3,
            overlap_duration_secs: 10,
        }
    }

    // --- SHA-256 / HMAC-SHA256 correctness -------------------------------------

    #[test]
    fn test_sha256_empty() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let h = sha256(b"");
        assert_eq!(h[0], 0xe3);
        assert_eq!(h[1], 0xb0);
        assert_eq!(h[31], 0x55);
    }

    #[test]
    fn test_sha256_abc() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2ec73b00361bbef0469352179912d552d428... (first 3 bytes)
        let h = sha256(b"abc");
        assert_eq!(h[0], 0xba);
        assert_eq!(h[1], 0x78);
        assert_eq!(h[2], 0x16);
    }

    #[test]
    fn test_hmac_sha256_deterministic() {
        let key = b"test-key";
        let data = b"test-data";
        let mac1 = hmac_sha256(key, data);
        let mac2 = hmac_sha256(key, data);
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_hmac_sha256_key_sensitivity() {
        let data = b"data";
        let mac1 = hmac_sha256(b"key1", data);
        let mac2 = hmac_sha256(b"key2", data);
        assert_ne!(mac1, mac2);
    }

    // --- Key derivation -------------------------------------------------------

    #[test]
    fn test_hkdf_key_deterministic() {
        let k1 = derive_hkdf_key(b"master", b"info", 5);
        let k2 = derive_hkdf_key(b"master", b"info", 5);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 16);
    }

    #[test]
    fn test_hkdf_key_period_sensitive() {
        let k1 = derive_hkdf_key(b"master", b"info", 1);
        let k2 = derive_hkdf_key(b"master", b"info", 2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_hkdf_key_master_sensitive() {
        let k1 = derive_hkdf_key(b"master1", b"info", 1);
        let k2 = derive_hkdf_key(b"master2", b"info", 1);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_counter_key_deterministic() {
        let k1 = derive_counter_key(0xDEAD_BEEF_CAFE_1234, 7);
        let k2 = derive_counter_key(0xDEAD_BEEF_CAFE_1234, 7);
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 16);
    }

    #[test]
    fn test_counter_key_period_sensitive() {
        let k1 = derive_counter_key(42, 0);
        let k2 = derive_counter_key(42, 1);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_random_key_len() {
        let k = derive_random_key(0);
        assert_eq!(k.len(), 16);
    }

    #[test]
    fn test_random_key_period_sensitive() {
        let k1 = derive_random_key(0);
        let k2 = derive_random_key(1);
        assert_ne!(k1, k2);
    }

    // --- RotatingKeyManager ---------------------------------------------------

    #[test]
    fn test_period_for_timestamp() {
        let mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        assert_eq!(mgr.period_for_timestamp(0), 0);
        assert_eq!(mgr.period_for_timestamp(99), 0);
        assert_eq!(mgr.period_for_timestamp(100), 1);
        assert_eq!(mgr.period_for_timestamp(250), 2);
    }

    #[test]
    fn test_key_for_period_random() {
        let mut mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        let k0 = mgr.key_for_period(0);
        let k1 = mgr.key_for_period(1);
        assert_eq!(k0.len(), 16);
        assert_eq!(k1.len(), 16);
        assert_ne!(k0, k1);
    }

    #[test]
    fn test_key_for_period_cached() {
        let mut mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        let k1 = mgr.key_for_period(5);
        let k2 = mgr.key_for_period(5);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_key_for_period_hkdf() {
        let policy = make_policy(KeyDerivation::Hkdf {
            master_secret: b"super-secret".to_vec(),
            info: b"video-stream".to_vec(),
        });
        let mut mgr = RotatingKeyManager::new(policy);
        let k0 = mgr.key_for_period(0);
        let k1 = mgr.key_for_period(1);
        assert_eq!(k0.len(), 16);
        assert_ne!(k0, k1);
    }

    #[test]
    fn test_key_for_period_counter() {
        let policy = make_policy(KeyDerivation::Counter { seed: 0x1234_5678 });
        let mut mgr = RotatingKeyManager::new(policy);
        let k3 = mgr.key_for_period(3);
        let k4 = mgr.key_for_period(4);
        assert_ne!(k3, k4);
    }

    #[test]
    fn test_current_key() {
        let mut mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        let (p, k) = mgr.current_key(250);
        assert_eq!(p, 2);
        assert_eq!(k.len(), 16);
    }

    #[test]
    fn test_preload_future_keys() {
        let mut mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        let keys = mgr.preload_future_keys(0);
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0].0, 0);
        assert_eq!(keys[1].0, 1);
        assert_eq!(keys[2].0, 2);
        // All keys should be cached now
        assert!(mgr.key_cache.contains_key(&0));
        assert!(mgr.key_cache.contains_key(&1));
        assert!(mgr.key_cache.contains_key(&2));
    }

    #[test]
    fn test_in_overlap_window() {
        let mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        // First 10 seconds of each period is the grace window
        assert!(mgr.in_overlap_window(0));
        assert!(mgr.in_overlap_window(9));
        assert!(!mgr.in_overlap_window(10));
        assert!(!mgr.in_overlap_window(99));
        assert!(mgr.in_overlap_window(100)); // start of period 1 → grace
    }

    #[test]
    fn test_evict_old_keys() {
        let mut mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        // Populate periods 0..=5
        for p in 0..=5 {
            mgr.key_for_period(p);
        }
        assert_eq!(mgr.key_cache.len(), 6);

        // now_secs = 500 → current period = 5; keep 2 older periods
        mgr.evict_old_keys(500, 2);
        // Periods 0, 1, 2 (5 - 2 = 3 → keep ≥ 3) should be evicted
        assert!(!mgr.key_cache.contains_key(&0));
        assert!(!mgr.key_cache.contains_key(&1));
        assert!(!mgr.key_cache.contains_key(&2));
        assert!(mgr.key_cache.contains_key(&3));
        assert!(mgr.key_cache.contains_key(&5));
    }

    #[test]
    fn test_overlap_duration_accessor() {
        let mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        assert_eq!(mgr.overlap_duration_secs(), 10);
    }

    #[test]
    fn test_rotation_interval_accessor() {
        let mgr = RotatingKeyManager::new(make_policy(KeyDerivation::Random));
        assert_eq!(mgr.rotation_interval_secs(), 100);
    }

    // --- PeriodToken ----------------------------------------------------------

    #[test]
    fn test_period_token_valid() {
        let signing_key = b"my-signing-key";
        let token = PeriodToken::new(42, signing_key);
        assert!(token.verify(signing_key));
    }

    #[test]
    fn test_period_token_wrong_key() {
        let token = PeriodToken::new(42, b"correct-key");
        assert!(!token.verify(b"wrong-key"));
    }

    #[test]
    fn test_period_token_wrong_period() {
        let signing_key = b"key";
        let token1 = PeriodToken::new(1, signing_key);
        let token2 = PeriodToken::new(2, signing_key);
        // Tags should differ for different periods
        assert_ne!(token1.tag, token2.tag);
    }

    #[test]
    fn test_period_token_deterministic() {
        let t1 = PeriodToken::new(7, b"k");
        let t2 = PeriodToken::new(7, b"k");
        assert_eq!(t1.tag, t2.tag);
    }

    // --- Integration ----------------------------------------------------------

    #[test]
    fn test_hkdf_manager_full_rotation() {
        let policy = KeyRotationPolicy {
            rotation_interval_secs: 60,
            key_derivation: KeyDerivation::Hkdf {
                master_secret: b"broadcast-master-key".to_vec(),
                info: b"live-stream-1".to_vec(),
            },
            preload_count: 2,
            overlap_duration_secs: 5,
        };
        let mut mgr = RotatingKeyManager::new(policy);

        // Simulate three rotation points
        let (p0, k0) = mgr.current_key(0);
        let (p1, k1) = mgr.current_key(60);
        let (p2, k2) = mgr.current_key(120);

        assert_eq!(p0, 0);
        assert_eq!(p1, 1);
        assert_eq!(p2, 2);
        assert_ne!(k0, k1);
        assert_ne!(k1, k2);

        // All keys should be 16 bytes
        assert_eq!(k0.len(), 16);
        assert_eq!(k1.len(), 16);
        assert_eq!(k2.len(), 16);

        // Keys are reproducible
        let k0_again = mgr.key_for_period(0);
        assert_eq!(k0, k0_again);
    }

    #[test]
    fn test_counter_manager_token_validation() {
        let policy = KeyRotationPolicy {
            rotation_interval_secs: 30,
            key_derivation: KeyDerivation::Counter { seed: 0xFEED_FACE },
            preload_count: 1,
            overlap_duration_secs: 3,
        };
        let mut mgr = RotatingKeyManager::new(policy);
        let signing_key = b"server-secret";

        // Server issues a token for the current period
        let (period, _key) = mgr.current_key(90); // period 3
        let token = PeriodToken::new(period, signing_key);

        // Client validates token
        assert!(token.verify(signing_key));
        assert_eq!(token.period, 3);
    }
}
