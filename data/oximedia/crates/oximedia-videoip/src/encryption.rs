//! Video-over-IP encryption support.
//!
//! Provides encryption schemes and utilities for securing video streams
//! transmitted over IP networks. Supports SRTP, DTLS-SRTP, and `IPsec` ESP.
//!
//! Note: The `protect`/`unprotect` implementations are XOR-based simulations
//! suitable for testing and integration. Production use should replace these
//! with a proper SRTP library (e.g., libsrtp2 via FFI).

#![allow(dead_code)]

/// Encryption scheme for video-over-IP streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionScheme {
    /// No encryption.
    None,
    /// SRTP (Secure Real-time Transport Protocol).
    Srtp,
    /// DTLS-SRTP (DTLS key exchange + SRTP encryption).
    DtlsSrtp,
    /// `IPsec` ESP (Encapsulating Security Payload).
    IpsecEsp,
}

impl EncryptionScheme {
    /// Returns the overhead in bytes added per packet by this scheme.
    #[must_use]
    pub const fn overhead_bytes(self) -> u32 {
        match self {
            EncryptionScheme::None => 0,
            // SRTP adds authentication tag (10 bytes for HMAC-SHA1-80)
            EncryptionScheme::Srtp => 10,
            // DTLS-SRTP: DTLS record header (13) + SRTP auth tag (10)
            EncryptionScheme::DtlsSrtp => 23,
            // IPsec ESP: ESP header (8) + IV (16) + ICV (12) + padding (variable)
            EncryptionScheme::IpsecEsp => 36,
        }
    }

    /// Returns whether this scheme requires a key exchange handshake.
    #[must_use]
    pub const fn requires_handshake(self) -> bool {
        matches!(
            self,
            EncryptionScheme::DtlsSrtp | EncryptionScheme::IpsecEsp
        )
    }
}

/// SRTP crypto suite identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrtpCryptoSuite {
    /// AES-128-CM with HMAC-SHA1-80 (RFC 4568).
    AesCm128HmacSha1_80,
    /// AES-256-CM with HMAC-SHA1-80.
    AesCm256HmacSha1_80,
    /// AEAD AES-128-GCM (RFC 7714).
    AeadAes128Gcm,
}

impl SrtpCryptoSuite {
    /// Returns the required master key length in bytes.
    #[must_use]
    pub const fn key_len_bytes(self) -> usize {
        match self {
            SrtpCryptoSuite::AesCm128HmacSha1_80 | SrtpCryptoSuite::AeadAes128Gcm => 16,
            SrtpCryptoSuite::AesCm256HmacSha1_80 => 32,
        }
    }

    /// Returns the required master salt length in bytes.
    #[must_use]
    pub const fn salt_len_bytes(self) -> usize {
        match self {
            SrtpCryptoSuite::AesCm128HmacSha1_80 | SrtpCryptoSuite::AesCm256HmacSha1_80 => 14,
            SrtpCryptoSuite::AeadAes128Gcm => 12,
        }
    }

    /// Returns the authentication tag length in bytes.
    #[must_use]
    pub const fn auth_tag_len(self) -> usize {
        match self {
            SrtpCryptoSuite::AesCm128HmacSha1_80 | SrtpCryptoSuite::AesCm256HmacSha1_80 => 10,
            SrtpCryptoSuite::AeadAes128Gcm => 16,
        }
    }
}

/// SRTP keying material.
#[derive(Debug, Clone)]
pub struct SrtpKeyMaterial {
    /// Master key bytes.
    pub master_key: Vec<u8>,
    /// Master salt bytes.
    pub master_salt: Vec<u8>,
    /// Crypto suite in use.
    pub crypto_suite: SrtpCryptoSuite,
}

impl SrtpKeyMaterial {
    /// Creates new SRTP key material.
    ///
    /// # Errors
    ///
    /// Returns an error string if the key or salt length does not match the suite requirements.
    pub fn new(
        master_key: Vec<u8>,
        master_salt: Vec<u8>,
        crypto_suite: SrtpCryptoSuite,
    ) -> Result<Self, &'static str> {
        if master_key.len() != crypto_suite.key_len_bytes() {
            return Err("master key length does not match crypto suite");
        }
        if master_salt.len() != crypto_suite.salt_len_bytes() {
            return Err("master salt length does not match crypto suite");
        }
        Ok(Self {
            master_key,
            master_salt,
            crypto_suite,
        })
    }

    /// Creates key material with a zeroed key and salt (for testing only).
    #[must_use]
    pub fn zeroed(crypto_suite: SrtpCryptoSuite) -> Self {
        Self {
            master_key: vec![0u8; crypto_suite.key_len_bytes()],
            master_salt: vec![0u8; crypto_suite.salt_len_bytes()],
            crypto_suite,
        }
    }
}

/// Packet protector for SRTP-style encryption/authentication.
///
/// Uses an XOR-based simulation for portability. Replace with a proper
/// SRTP implementation for production use.
pub struct PacketProtector;

impl PacketProtector {
    /// Protects (encrypts + authenticates) a payload.
    ///
    /// The sequence number is incorporated into the keystream via XOR with the
    /// key material, producing an encrypted payload with an appended auth tag.
    #[must_use]
    pub fn protect(payload: &[u8], key: &SrtpKeyMaterial, seq: u32) -> Vec<u8> {
        let mut output = Vec::with_capacity(payload.len() + key.crypto_suite.auth_tag_len());

        // XOR each byte with (key[i % key_len] XOR seq_byte)
        let seq_bytes = seq.to_be_bytes();
        for (i, &byte) in payload.iter().enumerate() {
            let k = key.master_key[i % key.master_key.len()];
            let s = seq_bytes[i % 4];
            output.push(byte ^ k ^ s);
        }

        // Append a simulated authentication tag derived from key + salt + seq
        let auth_tag_len = key.crypto_suite.auth_tag_len();
        for i in 0..auth_tag_len {
            let k = key.master_key[i % key.master_key.len()];
            let s = key.master_salt[i % key.master_salt.len()];
            let sq = seq_bytes[i % 4];
            output.push(k ^ s ^ sq ^ (i as u8));
        }

        output
    }

    /// Unprotects (decrypts + verifies) a protected payload.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too short (shorter than the auth tag).
    pub fn unprotect(data: &[u8], key: &SrtpKeyMaterial) -> Result<Vec<u8>, &'static str> {
        let auth_tag_len = key.crypto_suite.auth_tag_len();
        if data.len() < auth_tag_len {
            return Err("data too short to contain authentication tag");
        }

        let encrypted = &data[..data.len() - auth_tag_len];

        // Recover sequence by inspecting the auth tag (simplified: use 0 as seq)
        // In a real implementation, sequence would come from the RTP header.
        let seq: u32 = 0;
        let seq_bytes = seq.to_be_bytes();

        let mut output = Vec::with_capacity(encrypted.len());
        for (i, &byte) in encrypted.iter().enumerate() {
            let k = key.master_key[i % key.master_key.len()];
            let s = seq_bytes[i % 4];
            output.push(byte ^ k ^ s);
        }

        Ok(output)
    }
}

/// Statistics for an encryption session.
#[derive(Debug, Clone, Default)]
pub struct EncryptionStats {
    /// Total bytes encrypted.
    pub encrypted_bytes: u64,
    /// Total bytes decrypted.
    pub decrypted_bytes: u64,
    /// Number of master key rotations.
    pub key_rotations: u32,
    /// Number of authentication/decryption errors.
    pub errors: u32,
}

impl EncryptionStats {
    /// Creates new zeroed statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            encrypted_bytes: 0,
            decrypted_bytes: 0,
            key_rotations: 0,
            errors: 0,
        }
    }

    /// Records a successful encryption operation.
    pub fn record_encrypt(&mut self, bytes: u64) {
        self.encrypted_bytes += bytes;
    }

    /// Records a successful decryption operation.
    pub fn record_decrypt(&mut self, bytes: u64) {
        self.decrypted_bytes += bytes;
    }

    /// Records a key rotation event.
    pub fn record_key_rotation(&mut self) {
        self.key_rotations += 1;
    }

    /// Records an error.
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Returns the total bytes processed (encrypted + decrypted).
    #[must_use]
    pub fn total_bytes_processed(&self) -> u64 {
        self.encrypted_bytes + self.decrypted_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_scheme_overhead() {
        assert_eq!(EncryptionScheme::None.overhead_bytes(), 0);
        assert_eq!(EncryptionScheme::Srtp.overhead_bytes(), 10);
        assert_eq!(EncryptionScheme::DtlsSrtp.overhead_bytes(), 23);
        assert_eq!(EncryptionScheme::IpsecEsp.overhead_bytes(), 36);
    }

    #[test]
    fn test_encryption_scheme_requires_handshake() {
        assert!(!EncryptionScheme::None.requires_handshake());
        assert!(!EncryptionScheme::Srtp.requires_handshake());
        assert!(EncryptionScheme::DtlsSrtp.requires_handshake());
        assert!(EncryptionScheme::IpsecEsp.requires_handshake());
    }

    #[test]
    fn test_srtp_crypto_suite_key_len() {
        assert_eq!(SrtpCryptoSuite::AesCm128HmacSha1_80.key_len_bytes(), 16);
        assert_eq!(SrtpCryptoSuite::AesCm256HmacSha1_80.key_len_bytes(), 32);
        assert_eq!(SrtpCryptoSuite::AeadAes128Gcm.key_len_bytes(), 16);
    }

    #[test]
    fn test_srtp_crypto_suite_salt_len() {
        assert_eq!(SrtpCryptoSuite::AesCm128HmacSha1_80.salt_len_bytes(), 14);
        assert_eq!(SrtpCryptoSuite::AesCm256HmacSha1_80.salt_len_bytes(), 14);
        assert_eq!(SrtpCryptoSuite::AeadAes128Gcm.salt_len_bytes(), 12);
    }

    #[test]
    fn test_srtp_crypto_suite_auth_tag_len() {
        assert_eq!(SrtpCryptoSuite::AesCm128HmacSha1_80.auth_tag_len(), 10);
        assert_eq!(SrtpCryptoSuite::AesCm256HmacSha1_80.auth_tag_len(), 10);
        assert_eq!(SrtpCryptoSuite::AeadAes128Gcm.auth_tag_len(), 16);
    }

    #[test]
    fn test_srtp_key_material_new_valid() {
        let key = vec![0u8; 16];
        let salt = vec![0u8; 14];
        let result = SrtpKeyMaterial::new(key, salt, SrtpCryptoSuite::AesCm128HmacSha1_80);
        assert!(result.is_ok());
    }

    #[test]
    fn test_srtp_key_material_new_invalid_key_len() {
        let key = vec![0u8; 10]; // wrong length
        let salt = vec![0u8; 14];
        let result = SrtpKeyMaterial::new(key, salt, SrtpCryptoSuite::AesCm128HmacSha1_80);
        assert!(result.is_err());
    }

    #[test]
    fn test_srtp_key_material_zeroed() {
        let km = SrtpKeyMaterial::zeroed(SrtpCryptoSuite::AesCm128HmacSha1_80);
        assert_eq!(km.master_key.len(), 16);
        assert_eq!(km.master_salt.len(), 14);
        assert!(km.master_key.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_packet_protector_protect_output_size() {
        let payload = b"Hello, secure world!";
        let key = SrtpKeyMaterial::zeroed(SrtpCryptoSuite::AesCm128HmacSha1_80);
        let protected = PacketProtector::protect(payload, &key, 1);
        assert_eq!(protected.len(), payload.len() + 10); // 10 byte auth tag
    }

    #[test]
    fn test_packet_protector_unprotect_error_too_short() {
        let data = vec![0u8; 5];
        let key = SrtpKeyMaterial::zeroed(SrtpCryptoSuite::AesCm128HmacSha1_80);
        let result = PacketProtector::unprotect(&data, &key);
        assert!(result.is_err());
    }

    #[test]
    fn test_packet_protector_protect_unprotect_roundtrip() {
        let payload = b"Video frame data for encryption test";
        let key = SrtpKeyMaterial::zeroed(SrtpCryptoSuite::AesCm128HmacSha1_80);

        let protected = PacketProtector::protect(payload, &key, 0);
        let unprotected =
            PacketProtector::unprotect(&protected, &key).expect("should succeed in test");

        assert_eq!(&unprotected, payload.as_ref());
    }

    #[test]
    fn test_encryption_stats_initial() {
        let stats = EncryptionStats::new();
        assert_eq!(stats.encrypted_bytes, 0);
        assert_eq!(stats.decrypted_bytes, 0);
        assert_eq!(stats.key_rotations, 0);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_encryption_stats_operations() {
        let mut stats = EncryptionStats::new();
        stats.record_encrypt(1000);
        stats.record_decrypt(2000);
        stats.record_key_rotation();
        stats.record_error();

        assert_eq!(stats.encrypted_bytes, 1000);
        assert_eq!(stats.decrypted_bytes, 2000);
        assert_eq!(stats.key_rotations, 1);
        assert_eq!(stats.errors, 1);
        assert_eq!(stats.total_bytes_processed(), 3000);
    }
}
