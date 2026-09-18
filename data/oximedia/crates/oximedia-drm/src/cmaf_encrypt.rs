//! CMAF segment encryption using AES-128-CBC, AES-128-CTR, and AES-128-CBCS.
//!
//! Implements the three encryption schemes defined in ISO/IEC 23001-7 (Common
//! Encryption) as they apply to CMAF segments:
//!
//! - **CTR** (`cenc`): full-sample AES-128-CTR, byte-for-byte symmetric.
//! - **CBC** (`cbc1`): full-sample AES-128-CBC with PKCS#7 padding.
//! - **CBCS** (`cbcs`): partial-sample CBC — only complete 16-byte blocks are
//!   encrypted; the trailing partial block (if any) is left in the clear.
//!   This matches the FairPlay Streaming / HLS CMAF pattern.
//!
//! All block cipher work delegates to [`crate::aes_ctr`] and [`crate::aes_cbc`].

#![allow(missing_docs)]

use crate::aes_cbc::AesCbc;
use crate::aes_ctr::AesCtr;
use crate::{DrmError, Result};

// ── Public types ─────────────────────────────────────────────────────────────

/// Supported CMAF/CENC encryption schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncryptionScheme {
    /// AES-128-CTR (ISO CENC `cenc` scheme).  Symmetric; encrypt == decrypt.
    Ctr,
    /// AES-128-CBC with PKCS#7 padding (`cbc1` scheme).
    Cbc,
    /// AES-128-CBC partial-block encryption (`cbcs` scheme).
    ///
    /// Only complete 16-byte blocks are encrypted.  Any trailing bytes that do
    /// not fill a full block are passed through unmodified.  There is no PKCS#7
    /// padding.  Decryption is the mirror image.
    Cbcs,
}

/// Configuration for a CMAF encryptor / decryptor.
#[derive(Debug, Clone)]
pub struct CmafEncryptConfig {
    /// 128-bit AES content key.
    pub key: [u8; 16],
    /// 128-bit initialisation vector.
    ///
    /// - CTR: treated as `nonce[0..8] || counter_start_be[8..16]`.
    /// - CBC / CBCS: used directly as the CBC IV.
    pub iv: [u8; 16],
    /// Encryption scheme to apply.
    pub scheme: EncryptionScheme,
}

impl CmafEncryptConfig {
    /// Create a new configuration.
    pub fn new(key: [u8; 16], iv: [u8; 16], scheme: EncryptionScheme) -> Self {
        Self { key, iv, scheme }
    }
}

/// Applies CMAF encryption and decryption to byte slices.
pub struct CmafEncryptor {
    config: CmafEncryptConfig,
}

impl CmafEncryptor {
    /// Create a new encryptor with the given configuration.
    pub fn new(config: CmafEncryptConfig) -> Self {
        Self { config }
    }

    /// Encrypt `data` using the configured scheme.
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.config.scheme {
            EncryptionScheme::Ctr => Ok(self.apply_ctr(data)),
            EncryptionScheme::Cbc => self.apply_cbc_encrypt(data),
            EncryptionScheme::Cbcs => self.apply_cbcs_encrypt(data),
        }
    }

    /// Decrypt `data` using the configured scheme.
    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self.config.scheme {
            EncryptionScheme::Ctr => Ok(self.apply_ctr(data)),
            EncryptionScheme::Cbc => self.apply_cbc_decrypt(data),
            EncryptionScheme::Cbcs => self.apply_cbcs_decrypt(data),
        }
    }

    // ── CTR ───────────────────────────────────────────────────────────────────

    /// AES-128-CTR is self-inverse: encrypt == decrypt.
    ///
    /// The IV is split as `nonce[0..8] || counter_be[8..16]`.
    fn apply_ctr(&self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }
        let cipher = AesCtr::new_128(&self.config.key);
        let mut nonce = [0u8; 8];
        nonce.copy_from_slice(&self.config.iv[0..8]);
        let counter_start = u64::from_be_bytes([
            self.config.iv[8],
            self.config.iv[9],
            self.config.iv[10],
            self.config.iv[11],
            self.config.iv[12],
            self.config.iv[13],
            self.config.iv[14],
            self.config.iv[15],
        ]);
        cipher.encrypt(data, &nonce, counter_start)
    }

    // ── CBC ───────────────────────────────────────────────────────────────────

    fn apply_cbc_encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        let cipher = AesCbc::new_128(&self.config.key);
        cipher
            .encrypt(data, &self.config.iv)
            .map_err(|e| DrmError::EncryptionError(e.to_string()))
    }

    fn apply_cbc_decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let cipher = AesCbc::new_128(&self.config.key);
        cipher
            .decrypt(data, &self.config.iv)
            .map_err(|e| DrmError::DecryptionError(e.to_string()))
    }

    // ── CBCS ─────────────────────────────────────────────────────────────────

    /// CBCS encryption: encrypt only complete 16-byte blocks; leave the
    /// trailing partial block (< 16 bytes) in the clear.  No padding is added.
    fn apply_cbcs_encrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let full_blocks = data.len() / 16;
        let remainder = data.len() % 16;

        let cipher = AesCbc::new_128(&self.config.key);

        let mut out = Vec::with_capacity(data.len());

        if full_blocks > 0 {
            let aligned = &data[..full_blocks * 16];
            let encrypted = cipher
                .encrypt_no_padding(aligned, &self.config.iv)
                .map_err(|e| DrmError::EncryptionError(e.to_string()))?;
            out.extend_from_slice(&encrypted);
        }

        // Trailing partial block is copied verbatim.
        if remainder > 0 {
            out.extend_from_slice(&data[full_blocks * 16..]);
        }

        Ok(out)
    }

    /// CBCS decryption: decrypt only complete 16-byte blocks; copy the
    /// trailing partial block unchanged.
    fn apply_cbcs_decrypt(&self, data: &[u8]) -> Result<Vec<u8>> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        let full_blocks = data.len() / 16;
        let remainder = data.len() % 16;

        let cipher = AesCbc::new_128(&self.config.key);

        let mut out = Vec::with_capacity(data.len());

        if full_blocks > 0 {
            let aligned = &data[..full_blocks * 16];
            let decrypted = cipher
                .decrypt_no_padding(aligned, &self.config.iv)
                .map_err(|e| DrmError::DecryptionError(e.to_string()))?;
            out.extend_from_slice(&decrypted);
        }

        if remainder > 0 {
            out.extend_from_slice(&data[full_blocks * 16..]);
        }

        Ok(out)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> [u8; 16] {
        [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ]
    }

    fn iv() -> [u8; 16] {
        [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01,
        ]
    }

    // ── CTR roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn test_ctr_encrypt_decrypt_roundtrip() {
        let plaintext = b"Hello CMAF CTR segment data!";
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Ctr);
        let enc = CmafEncryptor::new(cfg.clone());
        let ciphertext = enc.encrypt(plaintext).expect("encrypt ok");
        let decrypted = enc.decrypt(&ciphertext).expect("decrypt ok");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_ctr_empty_input() {
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Ctr);
        let enc = CmafEncryptor::new(cfg);
        assert_eq!(enc.encrypt(&[]).expect("ok"), Vec::<u8>::new());
        assert_eq!(enc.decrypt(&[]).expect("ok"), Vec::<u8>::new());
    }

    #[test]
    fn test_ctr_encrypt_changes_plaintext() {
        let plaintext = vec![0xAAu8; 32];
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Ctr);
        let enc = CmafEncryptor::new(cfg);
        let ciphertext = enc.encrypt(&plaintext).expect("encrypt ok");
        assert_ne!(ciphertext, plaintext);
    }

    #[test]
    fn test_ctr_non_block_aligned_roundtrip() {
        let plaintext = b"non-aligned: 17 bytes!!"; // 23 bytes
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Ctr);
        let enc = CmafEncryptor::new(cfg);
        let ct = enc.encrypt(plaintext).expect("ok");
        let pt = enc.decrypt(&ct).expect("ok");
        assert_eq!(pt.as_slice(), plaintext.as_slice());
    }

    // ── CBC roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn test_cbc_encrypt_decrypt_roundtrip() {
        let plaintext = b"AES-128-CBC CMAF segment payload";
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Cbc);
        let enc = CmafEncryptor::new(cfg);
        let ct = enc.encrypt(plaintext).expect("encrypt ok");
        let pt = enc.decrypt(&ct).expect("decrypt ok");
        assert_eq!(pt.as_slice(), plaintext.as_slice());
    }

    #[test]
    fn test_cbc_block_aligned_roundtrip() {
        let plaintext = vec![0x5Au8; 32]; // exactly 2 blocks
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Cbc);
        let enc = CmafEncryptor::new(cfg);
        let ct = enc.encrypt(&plaintext).expect("ok");
        let pt = enc.decrypt(&ct).expect("ok");
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_cbc_ciphertext_differs_from_plaintext() {
        let plaintext = vec![0xBBu8; 16];
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Cbc);
        let enc = CmafEncryptor::new(cfg);
        let ct = enc.encrypt(&plaintext).expect("ok");
        assert_ne!(ct[..16], plaintext[..]);
    }

    // ── CBCS roundtrip ────────────────────────────────────────────────────────

    #[test]
    fn test_cbcs_block_aligned_roundtrip() {
        let plaintext = vec![0x3Cu8; 32]; // 2 complete blocks, no remainder
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Cbcs);
        let enc = CmafEncryptor::new(cfg);
        let ct = enc.encrypt(&plaintext).expect("ok");
        let pt = enc.decrypt(&ct).expect("ok");
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_cbcs_partial_block_preserved() {
        // 17 bytes: 1 full block + 1 byte remainder
        let plaintext: Vec<u8> = (0u8..17).collect();
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Cbcs);
        let enc = CmafEncryptor::new(cfg);
        let ct = enc.encrypt(&plaintext).expect("ok");
        // Output length must match input (no padding added).
        assert_eq!(ct.len(), 17);
        // Last byte (partial block) must be unchanged.
        assert_eq!(ct[16], plaintext[16]);
        // Full block should differ.
        assert_ne!(&ct[..16], &plaintext[..16]);
    }

    #[test]
    fn test_cbcs_non_aligned_roundtrip() {
        let plaintext: Vec<u8> = (0u8..37).map(|i| i.wrapping_mul(3)).collect();
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Cbcs);
        let enc = CmafEncryptor::new(cfg);
        let ct = enc.encrypt(&plaintext).expect("ok");
        let pt = enc.decrypt(&ct).expect("ok");
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_cbcs_empty_input() {
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Cbcs);
        let enc = CmafEncryptor::new(cfg);
        assert_eq!(enc.encrypt(&[]).expect("ok"), Vec::<u8>::new());
        assert_eq!(enc.decrypt(&[]).expect("ok"), Vec::<u8>::new());
    }

    // ── Different IV produces different ciphertext ────────────────────────────

    #[test]
    fn test_different_iv_produces_different_ciphertext() {
        let plaintext = vec![0xCCu8; 32];
        let cfg1 = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Ctr);
        let mut iv2 = iv();
        iv2[15] ^= 0x01;
        let cfg2 = CmafEncryptConfig::new(key(), iv2, EncryptionScheme::Ctr);
        let ct1 = CmafEncryptor::new(cfg1).encrypt(&plaintext).expect("ok");
        let ct2 = CmafEncryptor::new(cfg2).encrypt(&plaintext).expect("ok");
        assert_ne!(ct1, ct2);
    }

    // ── Pure-partial block (< 16 bytes) for CBCS ─────────────────────────────

    #[test]
    fn test_cbcs_only_partial_block_is_passthrough() {
        let plaintext = b"short"; // 5 bytes — no full block
        let cfg = CmafEncryptConfig::new(key(), iv(), EncryptionScheme::Cbcs);
        let enc = CmafEncryptor::new(cfg);
        let ct = enc.encrypt(plaintext).expect("ok");
        // All bytes are in the partial block — must be unchanged.
        assert_eq!(ct.as_slice(), plaintext.as_slice());
    }
}
