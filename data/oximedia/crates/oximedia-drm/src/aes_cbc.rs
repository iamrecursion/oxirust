//! Pure-Rust AES-128-CBC and AES-256-CBC encryption / decryption.
//!
//! Implements AES-CBC mode backed by the [`aes`] crate which automatically
//! uses AES-NI hardware acceleration at runtime when available.
//! Supports PKCS#7 padding.  Required by the CENC `cbcs` scheme (FairPlay).
//!
//! # Hardware Acceleration
//!
//! The `aes` crate detects AES-NI at runtime on x86/x86_64 via `cpuid` and
//! selects the hardware path transparently.  Enabling the `hardware-aes`
//! feature in `Cargo.toml` documents this capability without adding extra deps.

use aes::cipher::KeyInit;
use aes::{Aes128, Aes128Dec, Aes256, Aes256Dec};
use thiserror::Error;

use crate::aes_ctr::{decrypt_block_128, decrypt_block_256, encrypt_block_128, encrypt_block_256};

// ---------------------------------------------------------------------------
// AES-CBC error type
// ---------------------------------------------------------------------------

/// Errors specific to AES-CBC operations.
#[derive(Error, Debug)]
pub enum AesCbcError {
    #[error("invalid key length: expected 16 or 32 bytes, got {0}")]
    InvalidKeyLength(usize),

    #[error("invalid IV length: expected 16 bytes, got {0}")]
    InvalidIvLength(usize),

    #[error("ciphertext length ({0}) is not a multiple of 16")]
    InvalidCiphertextLength(usize),

    #[error("invalid PKCS#7 padding")]
    InvalidPadding,
}

// ---------------------------------------------------------------------------
// Internal cipher enum
// ---------------------------------------------------------------------------

/// Encryption ciphers.
#[derive(Clone)]
enum EncCipher {
    Aes128(Aes128),
    Aes256(Aes256),
}

impl EncCipher {
    fn encrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        match self {
            EncCipher::Aes128(c) => encrypt_block_128(c, block),
            EncCipher::Aes256(c) => encrypt_block_256(c, block),
        }
    }
}

impl std::fmt::Debug for EncCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncCipher::Aes128(_) => write!(f, "EncCipher::Aes128"),
            EncCipher::Aes256(_) => write!(f, "EncCipher::Aes256"),
        }
    }
}

/// Decryption ciphers.
#[derive(Clone)]
enum DecCipher {
    Aes128(Aes128Dec),
    Aes256(Aes256Dec),
}

impl DecCipher {
    fn decrypt_block(&self, block: &[u8; 16]) -> [u8; 16] {
        match self {
            DecCipher::Aes128(c) => decrypt_block_128(c, block),
            DecCipher::Aes256(c) => decrypt_block_256(c, block),
        }
    }
}

impl std::fmt::Debug for DecCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecCipher::Aes128(_) => write!(f, "DecCipher::Aes128"),
            DecCipher::Aes256(_) => write!(f, "DecCipher::Aes256"),
        }
    }
}

// ---------------------------------------------------------------------------
// AesCbc cipher
// ---------------------------------------------------------------------------

/// AES-CBC cipher for 128-bit or 256-bit keys with PKCS#7 padding.
///
/// Backed by the [`aes`] crate which transparently uses AES-NI hardware
/// acceleration when available on x86/x86_64.
#[derive(Debug, Clone)]
pub struct AesCbc {
    key_size_bits: usize,
    enc: EncCipher,
    dec: DecCipher,
}

impl AesCbc {
    /// Create a new AES-128-CBC cipher from a 16-byte key.
    pub fn new_128(key: &[u8; 16]) -> Self {
        Self {
            key_size_bits: 128,
            enc: EncCipher::Aes128(
                Aes128::new_from_slice(key).expect("AES-128 key is exactly 16 bytes; infallible"),
            ),
            dec: DecCipher::Aes128(
                Aes128Dec::new_from_slice(key)
                    .expect("AES-128 key is exactly 16 bytes; infallible"),
            ),
        }
    }

    /// Create a new AES-256-CBC cipher from a 32-byte key.
    pub fn new_256(key: &[u8; 32]) -> Self {
        Self {
            key_size_bits: 256,
            enc: EncCipher::Aes256(
                Aes256::new_from_slice(key).expect("AES-256 key is exactly 32 bytes; infallible"),
            ),
            dec: DecCipher::Aes256(
                Aes256Dec::new_from_slice(key)
                    .expect("AES-256 key is exactly 32 bytes; infallible"),
            ),
        }
    }

    /// Create from a variable-length key slice.
    pub fn from_key(key: &[u8]) -> Result<Self, AesCbcError> {
        match key.len() {
            16 => {
                let mut k = [0u8; 16];
                k.copy_from_slice(key);
                Ok(Self::new_128(&k))
            }
            32 => {
                let mut k = [0u8; 32];
                k.copy_from_slice(key);
                Ok(Self::new_256(&k))
            }
            other => Err(AesCbcError::InvalidKeyLength(other)),
        }
    }

    /// Return the key size in bits.
    pub fn key_size_bits(&self) -> usize {
        self.key_size_bits
    }

    /// Encrypt `plaintext` in CBC mode with PKCS#7 padding.
    ///
    /// `iv` must be exactly 16 bytes.
    pub fn encrypt(&self, plaintext: &[u8], iv: &[u8]) -> Result<Vec<u8>, AesCbcError> {
        if iv.len() != 16 {
            return Err(AesCbcError::InvalidIvLength(iv.len()));
        }

        // Apply PKCS#7 padding
        let pad_len = 16 - (plaintext.len() % 16);
        let padded_len = plaintext.len() + pad_len;
        let mut padded = Vec::with_capacity(padded_len);
        padded.extend_from_slice(plaintext);
        padded.resize(padded_len, pad_len as u8);

        let mut output = Vec::with_capacity(padded_len);
        let mut prev_block = [0u8; 16];
        prev_block.copy_from_slice(iv);

        for chunk in padded.chunks_exact(16) {
            // XOR with previous ciphertext block (or IV)
            let mut input_block = [0u8; 16];
            for i in 0..16 {
                input_block[i] = chunk[i] ^ prev_block[i];
            }

            let cipher_block = self.enc.encrypt_block(&input_block);
            output.extend_from_slice(&cipher_block);
            prev_block = cipher_block;
        }

        Ok(output)
    }

    /// Decrypt `ciphertext` in CBC mode and remove PKCS#7 padding.
    ///
    /// `iv` must be exactly 16 bytes. `ciphertext` length must be a multiple of 16.
    pub fn decrypt(&self, ciphertext: &[u8], iv: &[u8]) -> Result<Vec<u8>, AesCbcError> {
        if iv.len() != 16 {
            return Err(AesCbcError::InvalidIvLength(iv.len()));
        }
        if ciphertext.is_empty() || ciphertext.len() % 16 != 0 {
            return Err(AesCbcError::InvalidCiphertextLength(ciphertext.len()));
        }

        let mut output = Vec::with_capacity(ciphertext.len());
        let mut prev_block = [0u8; 16];
        prev_block.copy_from_slice(iv);

        for chunk in ciphertext.chunks_exact(16) {
            let mut ct_block = [0u8; 16];
            ct_block.copy_from_slice(chunk);

            let decrypted = self.dec.decrypt_block(&ct_block);

            // XOR with previous ciphertext block (or IV)
            for i in 0..16 {
                output.push(decrypted[i] ^ prev_block[i]);
            }
            prev_block = ct_block;
        }

        // Remove PKCS#7 padding
        let pad_byte = output.last().copied().ok_or(AesCbcError::InvalidPadding)?;
        let pad_len = pad_byte as usize;
        if pad_len == 0 || pad_len > 16 || pad_len > output.len() {
            return Err(AesCbcError::InvalidPadding);
        }
        let pad_start = output.len() - pad_len;
        if !output[pad_start..].iter().all(|&b| b == pad_byte) {
            return Err(AesCbcError::InvalidPadding);
        }
        output.truncate(pad_start);

        Ok(output)
    }

    /// Encrypt without PKCS#7 padding. Caller must ensure `plaintext.len()` is a
    /// multiple of 16.  This is used by the CENC `cbcs` pattern encryption where
    /// each block is exactly 16 bytes and no padding is desired.
    pub fn encrypt_no_padding(&self, plaintext: &[u8], iv: &[u8]) -> Result<Vec<u8>, AesCbcError> {
        if iv.len() != 16 {
            return Err(AesCbcError::InvalidIvLength(iv.len()));
        }
        if plaintext.len() % 16 != 0 {
            return Err(AesCbcError::InvalidCiphertextLength(plaintext.len()));
        }

        let mut output = Vec::with_capacity(plaintext.len());
        let mut prev_block = [0u8; 16];
        prev_block.copy_from_slice(iv);

        for chunk in plaintext.chunks_exact(16) {
            let mut input_block = [0u8; 16];
            for i in 0..16 {
                input_block[i] = chunk[i] ^ prev_block[i];
            }
            let cipher_block = self.enc.encrypt_block(&input_block);
            output.extend_from_slice(&cipher_block);
            prev_block = cipher_block;
        }

        Ok(output)
    }

    /// Decrypt without PKCS#7 unpadding (inverse of `encrypt_no_padding`).
    pub fn decrypt_no_padding(&self, ciphertext: &[u8], iv: &[u8]) -> Result<Vec<u8>, AesCbcError> {
        if iv.len() != 16 {
            return Err(AesCbcError::InvalidIvLength(iv.len()));
        }
        if ciphertext.is_empty() || ciphertext.len() % 16 != 0 {
            return Err(AesCbcError::InvalidCiphertextLength(ciphertext.len()));
        }

        let mut output = Vec::with_capacity(ciphertext.len());
        let mut prev_block = [0u8; 16];
        prev_block.copy_from_slice(iv);

        for chunk in ciphertext.chunks_exact(16) {
            let mut ct_block = [0u8; 16];
            ct_block.copy_from_slice(chunk);
            let decrypted = self.dec.decrypt_block(&ct_block);
            for i in 0..16 {
                output.push(decrypted[i] ^ prev_block[i]);
            }
            prev_block = ct_block;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aes_ctr::{decrypt_block_128, encrypt_block_128};

    // ----- AES-128 ECB block roundtrip ----------------------------------------

    #[test]
    fn test_aes128_encrypt_decrypt_block_roundtrip() {
        let key = [
            0x2bu8, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let plaintext: [u8; 16] = [
            0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37,
            0x07, 0x34,
        ];
        let enc =
            Aes128::new_from_slice(&key).expect("16-byte key is valid for AES-128; infallible");
        let dec = Aes128Dec::new_from_slice(&key)
            .expect("16-byte key is valid for AES-128 decryption; infallible");
        let ciphertext = encrypt_block_128(&enc, &plaintext);
        let recovered = decrypt_block_128(&dec, &ciphertext);
        assert_eq!(recovered, plaintext);
    }

    #[test]
    fn test_aes256_encrypt_decrypt_block_roundtrip() {
        let key: [u8; 32] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
            0x1c, 0x1d, 0x1e, 0x1f,
        ];
        let plaintext: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let cipher = AesCbc::new_256(&key);
        // Encrypt and decrypt with no-padding (single block, already 16 bytes)
        let iv = [0u8; 16];
        let ct = cipher.encrypt_no_padding(&plaintext, &iv).expect("encrypt");
        let pt = cipher.decrypt_no_padding(&ct, &iv).expect("decrypt");
        assert_eq!(pt, plaintext.to_vec());
    }

    // ----- AES-CBC with PKCS#7 padding roundtrips ----------------------------

    #[test]
    fn test_cbc128_roundtrip_single_block() {
        let key = [0x42u8; 16];
        let iv = [0x00u8; 16];
        let plaintext = b"Exactly16Bytes!"; // 15 bytes
        let cipher = AesCbc::new_128(&key);
        let ct = cipher
            .encrypt(plaintext, &iv)
            .expect("encrypt should succeed");
        // With PKCS#7: 15 bytes + 1 byte padding = 16 bytes ciphertext
        assert_eq!(ct.len(), 16);
        let pt = cipher.decrypt(&ct, &iv).expect("decrypt should succeed");
        assert_eq!(pt, plaintext.to_vec());
    }

    #[test]
    fn test_cbc128_roundtrip_multi_block() {
        let key = [0xABu8; 16];
        let iv = [0xCDu8; 16];
        let plaintext = b"Hello AES-128-CBC mode with multiple blocks of data for testing!";
        let cipher = AesCbc::new_128(&key);
        let ct = cipher
            .encrypt(plaintext, &iv)
            .expect("encrypt should succeed");
        let pt = cipher.decrypt(&ct, &iv).expect("decrypt should succeed");
        assert_eq!(pt, plaintext.to_vec());
    }

    #[test]
    fn test_cbc256_roundtrip() {
        let key = [0xDEu8; 32];
        let iv = [0x11u8; 16];
        let plaintext = b"AES-256-CBC encryption round-trip test with PKCS7 padding";
        let cipher = AesCbc::new_256(&key);
        let ct = cipher
            .encrypt(plaintext, &iv)
            .expect("encrypt should succeed");
        let pt = cipher.decrypt(&ct, &iv).expect("decrypt should succeed");
        assert_eq!(pt, plaintext.to_vec());
    }

    #[test]
    fn test_cbc_empty_plaintext() {
        let key = [0x00u8; 16];
        let iv = [0x00u8; 16];
        let cipher = AesCbc::new_128(&key);
        let ct = cipher.encrypt(b"", &iv).expect("encrypt should succeed");
        assert_eq!(ct.len(), 16); // one full padding block
        let pt = cipher.decrypt(&ct, &iv).expect("decrypt should succeed");
        assert!(pt.is_empty());
    }

    #[test]
    fn test_cbc_different_iv_different_ciphertext() {
        let key = [0x77u8; 16];
        let plaintext = b"Same plaintext different IVs";
        let cipher = AesCbc::new_128(&key);
        let ct1 = cipher
            .encrypt(plaintext, &[0x00; 16])
            .expect("encrypt should succeed");
        let ct2 = cipher
            .encrypt(plaintext, &[0x01; 16])
            .expect("encrypt should succeed");
        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_cbc_invalid_iv_length() {
        let key = [0x00u8; 16];
        let cipher = AesCbc::new_128(&key);
        let result = cipher.encrypt(b"data", &[0u8; 8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cbc_invalid_ciphertext_length_for_decrypt() {
        let key = [0x00u8; 16];
        let cipher = AesCbc::new_128(&key);
        let result = cipher.decrypt(&[0u8; 15], &[0u8; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cbc_from_key_128() {
        let key = vec![0xAAu8; 16];
        let cipher = AesCbc::from_key(&key).expect("from_key should succeed");
        assert_eq!(cipher.key_size_bits(), 128);
    }

    #[test]
    fn test_cbc_from_key_256() {
        let key = vec![0xBBu8; 32];
        let cipher = AesCbc::from_key(&key).expect("from_key should succeed");
        assert_eq!(cipher.key_size_bits(), 256);
    }

    #[test]
    fn test_cbc_from_key_invalid() {
        let key = vec![0xCCu8; 24];
        assert!(AesCbc::from_key(&key).is_err());
    }

    // ----- No-padding mode (for CENC cbcs) -----------------------------------

    #[test]
    fn test_cbc_no_padding_roundtrip() {
        let key = [0x55u8; 16];
        let iv = [0x33u8; 16];
        // Must be a multiple of 16
        let plaintext = [0xABu8; 48];
        let cipher = AesCbc::new_128(&key);
        let ct = cipher
            .encrypt_no_padding(&plaintext, &iv)
            .expect("encrypt should succeed");
        assert_eq!(ct.len(), 48);
        let pt = cipher
            .decrypt_no_padding(&ct, &iv)
            .expect("decrypt should succeed");
        assert_eq!(pt, plaintext.to_vec());
    }

    #[test]
    fn test_cbc_no_padding_rejects_non_multiple() {
        let key = [0x00u8; 16];
        let iv = [0x00u8; 16];
        let cipher = AesCbc::new_128(&key);
        let result = cipher.encrypt_no_padding(&[0u8; 17], &iv);
        assert!(result.is_err());
    }

    #[test]
    fn test_cbc128_nist_known_answer() {
        // NIST SP 800-38A F.2.1 CBC-AES128.Encrypt
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let iv: [u8; 16] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d,
            0x0e, 0x0f,
        ];
        let plaintext: [u8; 64] = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93,
            0x17, 0x2a, 0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c, 0x9e, 0xb7, 0x6f, 0xac,
            0x45, 0xaf, 0x8e, 0x51, 0x30, 0xc8, 0x1c, 0x46, 0xa3, 0x5c, 0xe4, 0x11, 0xe5, 0xfb,
            0xc1, 0x19, 0x1a, 0x0a, 0x52, 0xef, 0xf6, 0x9f, 0x24, 0x45, 0xdf, 0x4f, 0x9b, 0x17,
            0xad, 0x2b, 0x41, 0x7b, 0xe6, 0x6c, 0x37, 0x10,
        ];
        let expected_ct: [u8; 64] = [
            0x76, 0x49, 0xab, 0xac, 0x81, 0x19, 0xb2, 0x46, 0xce, 0xe9, 0x8e, 0x9b, 0x12, 0xe9,
            0x19, 0x7d, 0x50, 0x86, 0xcb, 0x9b, 0x50, 0x72, 0x19, 0xee, 0x95, 0xdb, 0x11, 0x3a,
            0x91, 0x76, 0x78, 0xb2, 0x73, 0xbe, 0xd6, 0xb8, 0xe3, 0xc1, 0x74, 0x3b, 0x71, 0x16,
            0xe6, 0x9e, 0x22, 0x22, 0x95, 0x16, 0x3f, 0xf1, 0xca, 0xa1, 0x68, 0x1f, 0xac, 0x09,
            0x12, 0x0e, 0xca, 0x30, 0x75, 0x86, 0xe1, 0xa7,
        ];

        let cipher = AesCbc::new_128(&key);
        let ct = cipher
            .encrypt_no_padding(&plaintext, &iv)
            .expect("encrypt should succeed");
        assert_eq!(ct, expected_ct.to_vec(), "NIST CBC-AES128 encrypt failed");

        let pt = cipher
            .decrypt_no_padding(&ct, &iv)
            .expect("decrypt should succeed");
        assert_eq!(pt, plaintext.to_vec(), "NIST CBC-AES128 decrypt failed");
    }

    #[test]
    fn test_cbc_large_data_roundtrip() {
        let key = [0x5au8; 16];
        let iv = [0x3cu8; 16];
        let plaintext: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let cipher = AesCbc::new_128(&key);
        let ct = cipher
            .encrypt(&plaintext, &iv)
            .expect("encrypt should succeed");
        let pt = cipher.decrypt(&ct, &iv).expect("decrypt should succeed");
        assert_eq!(pt, plaintext);
    }

    #[test]
    fn test_cbc_ciphertext_differs_from_plaintext() {
        let key = [0xFFu8; 16];
        let iv = [0xEEu8; 16];
        let plaintext = b"This should be encrypted not plain!";
        let cipher = AesCbc::new_128(&key);
        let ct = cipher
            .encrypt(plaintext, &iv)
            .expect("encrypt should succeed");
        assert_ne!(ct, plaintext.to_vec());
    }

    #[test]
    fn test_cbc_padding_one_byte_short() {
        // 15 bytes of plaintext -> 1 byte padding
        let key = [0x12u8; 16];
        let iv = [0x34u8; 16];
        let plaintext = [0xAA; 15];
        let cipher = AesCbc::new_128(&key);
        let ct = cipher
            .encrypt(&plaintext, &iv)
            .expect("encrypt should succeed");
        assert_eq!(ct.len(), 16);
        let pt = cipher.decrypt(&ct, &iv).expect("decrypt should succeed");
        assert_eq!(pt, plaintext.to_vec());
    }

    #[test]
    fn test_decrypt_block_inverse_of_encrypt_block() {
        // Verify with all-zero key and all-zero plaintext
        let key = [0u8; 16];
        let enc =
            Aes128::new_from_slice(&key).expect("16-byte key is valid for AES-128; infallible");
        let dec = Aes128Dec::new_from_slice(&key)
            .expect("16-byte key is valid for AES-128 decryption; infallible");
        let pt = [0u8; 16];
        let ct = encrypt_block_128(&enc, &pt);
        // ct should not equal pt (AES of zeros is not zeros)
        assert_ne!(ct, pt);
        let recovered = decrypt_block_128(&dec, &ct);
        assert_eq!(recovered, pt);
    }
}
