//! SRT Key Material Exchange (KMX) simulation.
//!
//! Implements the SRT key exchange protocol structure per RFC/SRT specification.
//! Uses AES key wrapping (AESKW) concept without an external AES library —
//! implements the structure and state machine, with XOR-based simulation for
//! testing (production would use a real AES implementation).

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// KwAlgorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Key wrapping algorithm identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KwAlgorithm {
    /// AES-128 in CTR mode (SRT standard).
    Aes128,
    /// AES-192 in CTR mode.
    Aes192,
    /// AES-256 in CTR mode (maximum security).
    Aes256,
}

impl KwAlgorithm {
    /// Returns the raw key size in bytes: 16, 24, or 32.
    pub fn key_size_bytes(&self) -> usize {
        match self {
            Self::Aes128 => 16,
            Self::Aes192 => 24,
            Self::Aes256 => 32,
        }
    }

    /// Returns a short human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Aes128 => "AES-128",
            Self::Aes192 => "AES-192",
            Self::Aes256 => "AES-256",
        }
    }

    /// Encode as the 1-byte cipher field used in `KeyMaterial`.
    fn as_u8(self) -> u8 {
        match self {
            Self::Aes128 => 2,
            Self::Aes192 => 3,
            Self::Aes256 => 4,
        }
    }

    /// Decode from the 1-byte cipher field.
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            2 => Some(Self::Aes128),
            3 => Some(Self::Aes192),
            4 => Some(Self::Aes256),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KeyMaterial
// ─────────────────────────────────────────────────────────────────────────────

/// SRT key material structure (from SRT specification).
///
/// This is sent in the SRT handshake extension to establish shared session keys.
#[derive(Debug, Clone)]
pub struct KeyMaterial {
    /// Must be 1.
    pub version: u8,
    /// Packet Type: 2 = Key Material.
    pub pt: u8,
    /// Signature: 0x2029.
    pub sign: u16,
    /// Key flags: 1 = odd key, 2 = even key, 3 = both.
    pub kk: u8,
    /// Key Encryption Key Index (0 = none).
    pub keki: u32,
    /// Cipher / key-wrapping algorithm.
    pub cipher: KwAlgorithm,
    /// Authentication field (0 = none).
    pub auth: u8,
    /// Stream encryption type (2 = TS-based).
    pub se: u8,
    /// 128-bit salt.
    pub salt: Vec<u8>,
    /// Wrapped Stream Encryption Key (SEK).
    pub wrapped_key: Vec<u8>,
}

/// Simple LCG seeded by a `u64` for deterministic pseudo-random generation.
fn lcg_bytes(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        out.push((state >> 33) as u8);
    }
    out
}

/// XOR-based key wrapping (simulation only; production uses RFC 3394 AES-WRAP).
fn sim_wrap(key: &[u8], plaintext: &[u8]) -> Vec<u8> {
    // 8-byte integrity check value prepended, payload XOR'd with repeating key
    let mut out = Vec::with_capacity(8 + plaintext.len());
    out.extend_from_slice(&[0xA6u8; 8]);
    for (i, &b) in plaintext.iter().enumerate() {
        out.push(b ^ key[i % key.len()]);
    }
    out
}

impl KeyMaterial {
    /// Create new key material with a deterministically-seeded SEK.
    ///
    /// `seed` controls the pseudo-random generation; in production a CSPRNG is used.
    pub fn new(algorithm: KwAlgorithm, seed: u64) -> Self {
        let key_bytes = algorithm.key_size_bytes();
        let salt = lcg_bytes(seed ^ 0xDEAD_BEEF_CAFE_0001, 16);
        // Generate the raw session key and wrap it (using a zero KEK for simulation).
        let sek = lcg_bytes(seed ^ 0x1234_5678_9ABC_DEF0, key_bytes);
        let wrapped_key = sim_wrap(&vec![0u8; key_bytes], &sek);

        Self {
            version: 1,
            pt: 2,
            sign: 0x2029,
            kk: 3, // both keys
            keki: 0,
            cipher: algorithm,
            auth: 0,
            se: 2,
            salt,
            wrapped_key,
        }
    }

    /// Serialize to bytes (SRT KMX extension format).
    ///
    /// Layout (all fields big-endian):
    /// ```text
    /// [0]      version
    /// [1]      pt
    /// [2..3]   sign (u16)
    /// [4]      kk
    /// [5..8]   keki (u32)
    /// [9]      cipher
    /// [10]     auth
    /// [11]     se
    /// [12]     salt_len (number of bytes that follow)
    /// [13..12+salt_len]  salt
    /// [13+salt_len]      wrapped_key_len (u16, big-endian, 2 bytes)
    /// [15+salt_len..]    wrapped_key
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.version);
        out.push(self.pt);
        out.push((self.sign >> 8) as u8);
        out.push((self.sign & 0xFF) as u8);
        out.push(self.kk);
        out.push((self.keki >> 24) as u8);
        out.push((self.keki >> 16) as u8);
        out.push((self.keki >> 8) as u8);
        out.push((self.keki & 0xFF) as u8);
        out.push(self.cipher.as_u8());
        out.push(self.auth);
        out.push(self.se);
        out.push(self.salt.len() as u8);
        out.extend_from_slice(&self.salt);
        let wk_len = self.wrapped_key.len() as u16;
        out.push((wk_len >> 8) as u8);
        out.push((wk_len & 0xFF) as u8);
        out.extend_from_slice(&self.wrapped_key);
        out
    }

    /// Parse from bytes.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the parse error.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 15 {
            return Err("KeyMaterial data too short".to_string());
        }
        let version = data[0];
        let pt = data[1];
        let sign = (u16::from(data[2]) << 8) | u16::from(data[3]);
        let kk = data[4];
        let keki = (u32::from(data[5]) << 24)
            | (u32::from(data[6]) << 16)
            | (u32::from(data[7]) << 8)
            | u32::from(data[8]);
        let cipher_byte = data[9];
        let cipher = KwAlgorithm::from_u8(cipher_byte)
            .ok_or_else(|| format!("Unknown cipher byte: {cipher_byte}"))?;
        let auth = data[10];
        let se = data[11];
        let salt_len = data[12] as usize;

        if data.len() < 13 + salt_len + 2 {
            return Err("KeyMaterial truncated at salt".to_string());
        }
        let salt = data[13..13 + salt_len].to_vec();

        let wk_offset = 13 + salt_len;
        let wk_len = (u16::from(data[wk_offset]) << 8 | u16::from(data[wk_offset + 1])) as usize;
        if data.len() < wk_offset + 2 + wk_len {
            return Err("KeyMaterial truncated at wrapped_key".to_string());
        }
        let wrapped_key = data[wk_offset + 2..wk_offset + 2 + wk_len].to_vec();

        Ok(Self {
            version,
            pt,
            sign,
            kk,
            keki,
            cipher,
            auth,
            se,
            salt,
            wrapped_key,
        })
    }

    /// Returns `true` if the signature and version fields are correct.
    pub fn is_valid(&self) -> bool {
        self.sign == 0x2029 && self.version == 1
    }

    /// Returns the key size in bits (128, 192, or 256).
    pub fn key_size_bits(&self) -> usize {
        self.cipher.key_size_bytes() * 8
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HsExtType
// ─────────────────────────────────────────────────────────────────────────────

/// SRT Handshake Extension Type codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HsExtType {
    /// Handshake Request.
    HsReq = 1,
    /// Handshake Response.
    HsRsp = 2,
    /// Key Material Request.
    KmReq = 3,
    /// Key Material Response.
    KmRsp = 4,
    /// Stream ID.
    Sid = 5,
    /// Group membership.
    Group = 6,
}

impl HsExtType {
    /// Try to convert a `u16` into an `HsExtType`.
    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            1 => Some(Self::HsReq),
            2 => Some(Self::HsRsp),
            3 => Some(Self::KmReq),
            4 => Some(Self::KmRsp),
            5 => Some(Self::Sid),
            6 => Some(Self::Group),
            _ => None,
        }
    }

    /// Return the numeric value of this type.
    pub fn as_u16(&self) -> u16 {
        *self as u16
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HsExtension
// ─────────────────────────────────────────────────────────────────────────────

/// SRT handshake extension block.
#[derive(Debug, Clone)]
pub struct HsExtension {
    /// Extension type code.
    pub ext_type: HsExtType,
    /// Size of `data` expressed in 32-bit words (ceiling division).
    pub ext_size: u16,
    /// Raw payload bytes.
    pub data: Vec<u8>,
}

impl HsExtension {
    /// Build a Key Material extension from a `KeyMaterial` value.
    pub fn key_material(km: &KeyMaterial) -> Self {
        let data = km.to_bytes();
        let words = data.len().div_ceil(4) as u16;
        Self {
            ext_type: HsExtType::KmReq,
            ext_size: words,
            data,
        }
    }

    /// Build a Stream ID extension from a UTF-8 string.
    pub fn stream_id(sid: &str) -> Self {
        let data = sid.as_bytes().to_vec();
        let words = data.len().div_ceil(4) as u16;
        Self {
            ext_type: HsExtType::Sid,
            ext_size: words,
            data,
        }
    }

    /// Serialize to bytes.
    ///
    /// Layout:
    /// ```text
    /// [0..1]  ext_type (u16, big-endian)
    /// [2..3]  ext_size (u16, big-endian)
    /// [4..]   data     (ext_size * 4 bytes, zero-padded)
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let padded_len = self.ext_size as usize * 4;
        let mut out = Vec::with_capacity(4 + padded_len);
        out.push((self.ext_type.as_u16() >> 8) as u8);
        out.push((self.ext_type.as_u16() & 0xFF) as u8);
        out.push((self.ext_size >> 8) as u8);
        out.push((self.ext_size & 0xFF) as u8);
        out.extend_from_slice(&self.data);
        // Zero-pad to the declared word count
        while out.len() < 4 + padded_len {
            out.push(0);
        }
        out
    }

    /// Parse from bytes.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the parse error.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 4 {
            return Err("HsExtension too short".to_string());
        }
        let type_code = (u16::from(data[0]) << 8) | u16::from(data[1]);
        let ext_type = HsExtType::from_u16(type_code)
            .ok_or_else(|| format!("Unknown HsExtType: {type_code}"))?;
        let ext_size = (u16::from(data[2]) << 8) | u16::from(data[3]);
        let payload_len = ext_size as usize * 4;
        if data.len() < 4 + payload_len {
            return Err("HsExtension data truncated".to_string());
        }
        let payload = data[4..4 + payload_len].to_vec();
        Ok(Self {
            ext_type,
            ext_size,
            data: payload,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EncryptionState / EncryptionSession
// ─────────────────────────────────────────────────────────────────────────────

/// State machine for SRT encryption negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionState {
    /// No encryption configured.
    NoEncryption,
    /// Waiting for key material from peer.
    PendingKeyMaterial,
    /// Key material sent, waiting for acknowledgment.
    KeyMaterialSent,
    /// Key exchange complete, encryption active.
    Active,
    /// Key rotation in progress (new KM sent, awaiting ACK).
    Rotating,
    /// Decryption failed (bad or mismatched key).
    Failed,
}

/// Manages the SRT encryption state machine and key lifecycle.
pub struct EncryptionSession {
    state: EncryptionState,
    algorithm: KwAlgorithm,
    current_km: Option<KeyMaterial>,
    pending_km: Option<KeyMaterial>,
    key_rotation_interval_packets: u64,
    packets_since_rotation: u64,
}

impl EncryptionSession {
    /// Create a session with the given algorithm that starts in `PendingKeyMaterial`.
    pub fn new(algorithm: KwAlgorithm) -> Self {
        Self {
            state: EncryptionState::PendingKeyMaterial,
            algorithm,
            current_km: None,
            pending_km: None,
            key_rotation_interval_packets: 8_192,
            packets_since_rotation: 0,
        }
    }

    /// Create a session with no encryption (passthrough mode).
    pub fn no_encryption() -> Self {
        Self {
            state: EncryptionState::NoEncryption,
            algorithm: KwAlgorithm::Aes128,
            current_km: None,
            pending_km: None,
            key_rotation_interval_packets: u64::MAX,
            packets_since_rotation: 0,
        }
    }

    /// Return the current state.
    pub fn state(&self) -> EncryptionState {
        self.state
    }

    /// Return `true` if encryption is fully active.
    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            EncryptionState::Active | EncryptionState::Rotating
        )
    }

    /// Initiate key exchange: generate a `KeyMaterial` and transition to
    /// `KeyMaterialSent`.  The caller must send the returned KM to the peer.
    ///
    /// Returns `None` if encryption is disabled.
    pub fn initiate(&mut self, seed: u64) -> Option<KeyMaterial> {
        if self.state == EncryptionState::NoEncryption {
            return None;
        }
        let km = KeyMaterial::new(self.algorithm, seed);
        self.pending_km = Some(km.clone());
        self.state = EncryptionState::KeyMaterialSent;
        Some(km)
    }

    /// Apply key material received from the peer.
    ///
    /// Transitions to `Active` on success.
    ///
    /// # Errors
    ///
    /// Returns an error string if the KM is invalid.
    pub fn apply_peer_km(&mut self, km: KeyMaterial) -> Result<(), String> {
        if !km.is_valid() {
            self.state = EncryptionState::Failed;
            return Err("Peer key material is invalid (bad signature/version)".to_string());
        }
        self.current_km = Some(km);
        self.packets_since_rotation = 0;
        self.state = EncryptionState::Active;
        Ok(())
    }

    /// Returns `true` if it is time to rotate the session key.
    pub fn should_rotate(&self) -> bool {
        self.is_active() && self.packets_since_rotation >= self.key_rotation_interval_packets
    }

    /// Record that one packet has been processed.
    pub fn record_packet(&mut self) {
        self.packets_since_rotation += 1;
    }

    /// Perform a key rotation: generate new key material and transition to
    /// `Rotating`.  The caller must send the returned KM to the peer.
    ///
    /// Returns `None` if not in an active state.
    pub fn rotate(&mut self, seed: u64) -> Option<KeyMaterial> {
        if !self.is_active() {
            return None;
        }
        let km = KeyMaterial::new(self.algorithm, seed);
        self.pending_km = Some(km.clone());
        self.packets_since_rotation = 0;
        self.state = EncryptionState::Rotating;
        Some(km)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kw_algorithm_key_size() {
        assert_eq!(KwAlgorithm::Aes128.key_size_bytes(), 16);
        assert_eq!(KwAlgorithm::Aes192.key_size_bytes(), 24);
        assert_eq!(KwAlgorithm::Aes256.key_size_bytes(), 32);
    }

    #[test]
    fn test_key_material_new_is_valid() {
        let km = KeyMaterial::new(KwAlgorithm::Aes128, 42);
        assert!(km.is_valid(), "KeyMaterial::new should produce valid KM");
    }

    #[test]
    fn test_key_material_roundtrip() {
        let km = KeyMaterial::new(KwAlgorithm::Aes256, 99);
        let bits_before = km.key_size_bits();
        let bytes = km.to_bytes();
        let km2 = KeyMaterial::from_bytes(&bytes).expect("from_bytes should succeed");
        assert_eq!(km2.key_size_bits(), bits_before);
        assert!(km2.is_valid());
    }

    #[test]
    fn test_hs_ext_type_roundtrip() {
        let types = [
            HsExtType::HsReq,
            HsExtType::HsRsp,
            HsExtType::KmReq,
            HsExtType::KmRsp,
            HsExtType::Sid,
            HsExtType::Group,
        ];
        for t in types {
            assert_eq!(HsExtType::from_u16(t.as_u16()), Some(t));
        }
    }

    #[test]
    fn test_encryption_session_inactive() {
        let session = EncryptionSession::no_encryption();
        assert_eq!(session.state(), EncryptionState::NoEncryption);
        assert!(!session.is_active());
    }

    #[test]
    fn test_encryption_session_initiate() {
        let mut session = EncryptionSession::new(KwAlgorithm::Aes128);
        let km = session.initiate(1234);
        assert!(km.is_some(), "initiate should return Some(km)");
        assert_eq!(session.state(), EncryptionState::KeyMaterialSent);
        assert!(km.expect("should succeed in test").is_valid());
    }

    #[test]
    fn test_encryption_rotation() {
        let mut session = EncryptionSession::new(KwAlgorithm::Aes128);
        // Move to Active
        let km = KeyMaterial::new(KwAlgorithm::Aes128, 7);
        session
            .apply_peer_km(km)
            .expect("apply_peer_km should succeed");
        assert_eq!(session.state(), EncryptionState::Active);

        // Drive packets up to the rotation interval
        let interval = session.key_rotation_interval_packets;
        for _ in 0..interval {
            session.record_packet();
        }
        assert!(
            session.should_rotate(),
            "should_rotate must be true after interval packets"
        );

        // Perform rotation
        let new_km = session.rotate(9999);
        assert!(new_km.is_some());
        assert_eq!(session.state(), EncryptionState::Rotating);
        // After rotation, packet counter resets so should_rotate is false
        assert!(!session.should_rotate());
    }
}
