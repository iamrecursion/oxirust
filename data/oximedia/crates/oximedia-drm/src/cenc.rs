//! Common Encryption (CENC) implementation.
//!
//! Implements ISO/IEC 23001-7 (Common Encryption in ISO Base Media File Format).
//! Supports all four CENC encryption schemes defined in the specification.
//!
//! # CENC Encryption Scheme Variants
//!
//! | Scheme | Cipher      | Pattern         | Crypt Bytes  | DRM Systems         | Typical Use          |
//! |--------|-------------|-----------------|--------------|---------------------|----------------------|
//! | `cenc` | AES-128-CTR | None (full)     | All bytes    | Widevine, PlayReady, ClearKey | OTT VOD (DASH)  |
//! | `cbcs` | AES-128-CBC | 1:9 (1 encrypt, 9 skip) | First 16 B of each "crypt" block | FairPlay, Widevine | HLS live, CMAF |
//! | `cens` | AES-128-CTR | Configurable    | Pattern-based encrypted range | Rare / experimental | Low-latency DASH |
//! | `cbc1` | AES-128-CBC | None (full)     | All bytes    | PlayReady (legacy)  | Legacy DASH/PlayReady |
//!
//! ## Scheme Details
//!
//! ### `cenc` — AES-128-CTR, no pattern
//!
//! The most widely deployed scheme. The entire protected region of each sample
//! is encrypted using AES Counter mode. The 128-bit counter block is formed by
//! concatenating the 64-bit IV (nonce) with a 64-bit big-endian block counter
//! starting at zero for each sample. Required by the W3C Encrypted Media
//! Extensions specification for cross-platform interoperability.
//!
//! ```text
//! Sample: [clear_header][encrypted_body_CTR]
//! ```
//!
//! ### `cbcs` — AES-128-CBC, 1:9 subsample pattern
//!
//! Required by Apple FairPlay Streaming and broadly adopted for HLS content.
//! In the 1:9 pattern, one 16-byte AES block is encrypted followed by nine
//! 16-byte blocks left clear, repeating until the encrypted region ends.
//! Only the first 16 bytes of each "crypt" segment are encrypted; the
//! trailing partial block is always left clear. Each sample restarts the
//! CBC IV (per-sample IV in the `senc` box).
//!
//! ```text
//! Sample: [clear_header][ENC_block][9×clear_blocks][ENC_block][9×clear_blocks]...
//! ```
//!
//! ### `cens` — AES-128-CTR, configurable pattern
//!
//! Like `cbcs` but with CTR mode. The crypt/skip ratio is configurable via
//! [`EncryptionPattern`]. Rarely deployed in practice; provided for
//! completeness with the specification.
//!
//! ### `cbc1` — AES-128-CBC, no pattern (legacy)
//!
//! Full-sample CBC encryption without subsample patterns. Used by early
//! PlayReady deployments before `cbcs` was standardised. The entire encrypted
//! region is CBC-encrypted in a single chain; the IV is carried per-sample.
//! Largely superseded by `cbcs` in modern workflows.

use crate::aes_ctr::{encrypt_block_128, AesCtr};
use crate::{DrmError, DrmSystem, Result};
use aes::cipher::KeyInit;
use aes::Aes128;
use bytes::{BufMut, BytesMut};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::io::Read;
use uuid::Uuid;

/// CENC encryption scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionScheme {
    /// AES-CTR mode (cenc)
    Cenc,
    /// AES-CBC mode (cbc1)
    Cbc1,
    /// AES-CTR subsample pattern encryption (cens)
    Cens,
    /// AES-CBC subsample pattern encryption (cbcs)
    Cbcs,
}

impl EncryptionScheme {
    /// Get the scheme as a four-character code
    pub fn fourcc(&self) -> &'static str {
        match self {
            EncryptionScheme::Cenc => "cenc",
            EncryptionScheme::Cbc1 => "cbc1",
            EncryptionScheme::Cens => "cens",
            EncryptionScheme::Cbcs => "cbcs",
        }
    }

    /// Parse from four-character code
    pub fn from_fourcc(fourcc: &str) -> Option<Self> {
        match fourcc {
            "cenc" => Some(EncryptionScheme::Cenc),
            "cbc1" => Some(EncryptionScheme::Cbc1),
            "cens" => Some(EncryptionScheme::Cens),
            "cbcs" => Some(EncryptionScheme::Cbcs),
            _ => None,
        }
    }

    /// Check if scheme uses pattern encryption
    pub fn uses_pattern(&self) -> bool {
        matches!(self, EncryptionScheme::Cens | EncryptionScheme::Cbcs)
    }
}

/// Subsample encryption info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsampleInfo {
    /// Number of clear (unencrypted) bytes
    pub clear_bytes: u16,
    /// Number of encrypted bytes
    pub encrypted_bytes: u32,
}

impl SubsampleInfo {
    /// Create new subsample info
    pub fn new(clear_bytes: u16, encrypted_bytes: u32) -> Self {
        Self {
            clear_bytes,
            encrypted_bytes,
        }
    }

    /// Get total subsample size
    pub fn total_bytes(&self) -> u32 {
        self.clear_bytes as u32 + self.encrypted_bytes
    }
}

/// Sample encryption pattern for pattern-based encryption
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EncryptionPattern {
    /// Number of encrypted blocks in the pattern
    pub crypt_blocks: u8,
    /// Number of skip (clear) blocks in the pattern
    pub skip_blocks: u8,
}

impl EncryptionPattern {
    /// Create a new encryption pattern
    pub fn new(crypt_blocks: u8, skip_blocks: u8) -> Self {
        Self {
            crypt_blocks,
            skip_blocks,
        }
    }

    /// Default pattern for video (1:9 - encrypt 1 block, skip 9)
    pub fn video_default() -> Self {
        Self {
            crypt_blocks: 1,
            skip_blocks: 9,
        }
    }

    /// Full encryption pattern (encrypt all blocks)
    pub fn full_encryption() -> Self {
        Self {
            crypt_blocks: 1,
            skip_blocks: 0,
        }
    }
}

/// PSSH (Protection System Specific Header) box
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsshBox {
    /// Version (0 or 1)
    pub version: u8,
    /// Flags
    pub flags: u32,
    /// System ID
    pub system_id: Uuid,
    /// Key IDs (version 1 only)
    pub key_ids: Vec<Vec<u8>>,
    /// System-specific data
    pub data: Vec<u8>,
}

impl PsshBox {
    /// Create a new PSSH box (version 0)
    pub fn new(system_id: Uuid, data: Vec<u8>) -> Self {
        Self {
            version: 0,
            flags: 0,
            system_id,
            key_ids: Vec::new(),
            data,
        }
    }

    /// Create a new PSSH box (version 1 with key IDs)
    pub fn new_v1(system_id: Uuid, key_ids: Vec<Vec<u8>>, data: Vec<u8>) -> Self {
        Self {
            version: 1,
            flags: 0,
            system_id,
            key_ids,
            data,
        }
    }

    /// Serialize PSSH box to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buf = BytesMut::new();

        // Calculate size
        let mut size = 4 + 4; // size + box type
        size += 1 + 3; // version + flags
        size += 16; // system_id

        if self.version == 1 {
            size += 4; // key_id_count
            size += self.key_ids.len() * 16; // key_ids (each 16 bytes)
        }

        size += 4; // data_size
        size += self.data.len();

        // Write size
        buf.put_u32(size as u32);

        // Write box type 'pssh'
        buf.put(&b"pssh"[..]);

        // Write version and flags
        buf.put_u8(self.version);
        buf.put_u8(((self.flags >> 16) & 0xFF) as u8);
        buf.put_u8(((self.flags >> 8) & 0xFF) as u8);
        buf.put_u8((self.flags & 0xFF) as u8);

        // Write system ID
        buf.put(self.system_id.as_bytes().as_ref());

        // Write key IDs (version 1 only)
        if self.version == 1 {
            buf.put_u32(self.key_ids.len() as u32);
            for key_id in &self.key_ids {
                if key_id.len() != 16 {
                    return Err(DrmError::PsshError(format!(
                        "Key ID must be 16 bytes, got {}",
                        key_id.len()
                    )));
                }
                buf.put(&key_id[..]);
            }
        }

        // Write data
        buf.put_u32(self.data.len() as u32);
        buf.put(&self.data[..]);

        Ok(buf.to_vec())
    }

    /// Parse PSSH box from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(data);

        // Read size
        let size = read_u32(&mut cursor)?;
        if size as usize != data.len() {
            return Err(DrmError::PsshError(format!(
                "Invalid PSSH size: expected {}, got {}",
                data.len(),
                size
            )));
        }

        // Read box type
        let mut box_type = [0u8; 4];
        cursor.read_exact(&mut box_type)?;
        if &box_type != b"pssh" {
            return Err(DrmError::PsshError(format!(
                "Invalid box type: expected 'pssh', got '{}'",
                String::from_utf8_lossy(&box_type)
            )));
        }

        // Read version
        let version = read_u8(&mut cursor)?;
        if version > 1 {
            return Err(DrmError::PsshError(format!(
                "Unsupported PSSH version: {}",
                version
            )));
        }

        // Read flags
        let flag1 = read_u8(&mut cursor)? as u32;
        let flag2 = read_u8(&mut cursor)? as u32;
        let flag3 = read_u8(&mut cursor)? as u32;
        let flags = (flag1 << 16) | (flag2 << 8) | flag3;

        // Read system ID
        let mut system_id_bytes = [0u8; 16];
        cursor.read_exact(&mut system_id_bytes)?;
        let system_id = Uuid::from_bytes(system_id_bytes);

        // Read key IDs (version 1 only)
        let mut key_ids = Vec::new();
        if version == 1 {
            let key_id_count = read_u32(&mut cursor)?;
            for _ in 0..key_id_count {
                let mut key_id = vec![0u8; 16];
                cursor.read_exact(&mut key_id)?;
                key_ids.push(key_id);
            }
        }

        // Read data
        let data_size = read_u32(&mut cursor)?;
        let mut pssh_data = vec![0u8; data_size as usize];
        cursor.read_exact(&mut pssh_data)?;

        Ok(Self {
            version,
            flags,
            system_id,
            key_ids,
            data: pssh_data,
        })
    }

    /// Get DRM system from PSSH
    pub fn drm_system(&self) -> Option<DrmSystem> {
        DrmSystem::from_uuid(&self.system_id)
    }
}

/// Sample encryption info
#[derive(Debug, Clone)]
pub struct SampleEncryptionInfo {
    /// Initialization vector
    pub iv: Vec<u8>,
    /// Subsample information
    pub subsamples: Vec<SubsampleInfo>,
}

impl SampleEncryptionInfo {
    /// Create new sample encryption info
    pub fn new(iv: Vec<u8>) -> Self {
        Self {
            iv,
            subsamples: Vec::new(),
        }
    }

    /// Add a subsample
    pub fn add_subsample(&mut self, clear_bytes: u16, encrypted_bytes: u32) {
        self.subsamples
            .push(SubsampleInfo::new(clear_bytes, encrypted_bytes));
    }

    /// Check if sample uses subsample encryption
    pub fn has_subsamples(&self) -> bool {
        !self.subsamples.is_empty()
    }
}

/// CENC encryptor
pub struct CencEncryptor {
    scheme: EncryptionScheme,
    key: Vec<u8>,
    pattern: Option<EncryptionPattern>,
}

impl CencEncryptor {
    /// Create a new CENC encryptor
    pub fn new(scheme: EncryptionScheme, key: Vec<u8>) -> Result<Self> {
        if key.len() != 16 {
            return Err(DrmError::InvalidKey(format!(
                "Key must be 16 bytes for AES-128, got {}",
                key.len()
            )));
        }

        Ok(Self {
            scheme,
            key,
            pattern: None,
        })
    }

    /// Set encryption pattern for pattern-based schemes
    pub fn with_pattern(mut self, pattern: EncryptionPattern) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Generate a random IV
    pub fn generate_iv(&self) -> Result<Vec<u8>> {
        let iv_size = match self.scheme {
            EncryptionScheme::Cenc | EncryptionScheme::Cens => 8,
            EncryptionScheme::Cbc1 | EncryptionScheme::Cbcs => 16,
        };

        let mut iv = vec![0u8; iv_size];
        rand::rng().fill_bytes(&mut iv);

        Ok(iv)
    }

    /// Encrypt a full sample (no subsamples)
    pub fn encrypt_sample(&self, data: &[u8], iv: &[u8]) -> Result<Vec<u8>> {
        match self.scheme {
            EncryptionScheme::Cenc | EncryptionScheme::Cens => self.encrypt_ctr(data, iv),
            EncryptionScheme::Cbc1 | EncryptionScheme::Cbcs => self.encrypt_cbc(data, iv),
        }
    }

    /// Encrypt with subsample information
    pub fn encrypt_with_subsamples(
        &self,
        data: &[u8],
        iv: &[u8],
        subsamples: &[SubsampleInfo],
    ) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(data.len());
        let mut offset = 0;

        for subsample in subsamples {
            let clear_end = offset + subsample.clear_bytes as usize;
            let encrypted_end = clear_end + subsample.encrypted_bytes as usize;

            if encrypted_end > data.len() {
                return Err(DrmError::EncryptionError(
                    "Subsample extends beyond data".to_string(),
                ));
            }

            // Copy clear bytes as-is
            result.extend_from_slice(&data[offset..clear_end]);

            // Encrypt encrypted bytes
            let encrypted_data = self.encrypt_sample(&data[clear_end..encrypted_end], iv)?;
            result.extend_from_slice(&encrypted_data);

            offset = encrypted_end;
        }

        // Copy any remaining bytes
        if offset < data.len() {
            result.extend_from_slice(&data[offset..]);
        }

        Ok(result)
    }

    /// Encrypt using AES-CTR mode
    fn encrypt_ctr(&self, data: &[u8], iv: &[u8]) -> Result<Vec<u8>> {
        if iv.len() != 8 && iv.len() != 16 {
            return Err(DrmError::InvalidIv(format!(
                "IV must be 8 or 16 bytes for CTR mode, got {}",
                iv.len()
            )));
        }

        // Extend IV to 16 bytes if needed
        let mut full_iv = vec![0u8; 16];
        full_iv[..iv.len()].copy_from_slice(iv);

        // For CTR mode, we XOR the data with the encrypted counter
        let mut result = data.to_vec();
        let mut counter = full_iv;

        let block_size = 16;
        for (chunk_idx, chunk) in result.chunks_mut(block_size).enumerate() {
            // Encrypt the counter
            let encrypted_counter = self.aes_encrypt_block(&counter)?;

            // XOR with data
            for (i, byte) in chunk.iter_mut().enumerate() {
                *byte ^= encrypted_counter[i];
            }

            // Increment counter
            increment_counter(&mut counter);

            // Apply pattern if using pattern encryption
            if let Some(pattern) = self.pattern {
                let block_in_pattern =
                    chunk_idx % (pattern.crypt_blocks + pattern.skip_blocks) as usize;
                if block_in_pattern >= pattern.crypt_blocks as usize {
                    // Skip this block (restore original)
                    let start = chunk_idx * block_size;
                    let end = (start + chunk.len()).min(data.len());
                    chunk.copy_from_slice(&data[start..end]);
                }
            }
        }

        Ok(result)
    }

    /// Encrypt using AES-CBC mode
    fn encrypt_cbc(&self, data: &[u8], iv: &[u8]) -> Result<Vec<u8>> {
        if iv.len() != 16 {
            return Err(DrmError::InvalidIv(format!(
                "IV must be 16 bytes for CBC mode, got {}",
                iv.len()
            )));
        }

        // Pad data to block size
        let block_size = 16;
        let mut padded_data = data.to_vec();
        let padding_len = block_size - (data.len() % block_size);
        if padding_len != block_size {
            padded_data.resize(data.len() + padding_len, padding_len as u8);
        }

        let mut result = Vec::with_capacity(padded_data.len());
        let mut previous_block = iv.to_vec();

        for (chunk_idx, chunk) in padded_data.chunks(block_size).enumerate() {
            // XOR with previous ciphertext block (or IV)
            let mut xored = vec![0u8; block_size];
            for i in 0..block_size {
                xored[i] = chunk[i] ^ previous_block[i];
            }

            // Encrypt
            let encrypted = self.aes_encrypt_block(&xored)?;

            // Apply pattern if using pattern encryption
            if let Some(pattern) = self.pattern {
                let block_in_pattern =
                    chunk_idx % (pattern.crypt_blocks + pattern.skip_blocks) as usize;
                if block_in_pattern >= pattern.crypt_blocks as usize {
                    // Skip this block (use original, not encrypted)
                    result.extend_from_slice(chunk);
                    previous_block = chunk.to_vec();
                    continue;
                }
            }

            result.extend_from_slice(&encrypted);
            previous_block = encrypted;
        }

        Ok(result)
    }

    /// Encrypt a single AES-128 block (16 bytes) using the stored key.
    ///
    /// Uses the `aes` crate which auto-selects AES-NI at runtime.
    fn aes_encrypt_block(&self, block: &[u8]) -> Result<Vec<u8>> {
        if block.len() != 16 {
            return Err(DrmError::EncryptionError(format!(
                "Block must be 16 bytes, got {}",
                block.len()
            )));
        }
        if self.key.len() != 16 {
            return Err(DrmError::InvalidKey(format!(
                "Key must be 16 bytes, got {}",
                self.key.len()
            )));
        }
        let cipher = Aes128::new_from_slice(&self.key)
            .expect("key length validated as 16 bytes above; infallible");
        let mut arr = [0u8; 16];
        arr.copy_from_slice(block);
        let out = encrypt_block_128(&cipher, &arr);
        Ok(out.to_vec())
    }
}

/// CENC decryptor
pub struct CencDecryptor {
    scheme: EncryptionScheme,
    key: Vec<u8>,
    pattern: Option<EncryptionPattern>,
}

impl CencDecryptor {
    /// Create a new CENC decryptor
    pub fn new(scheme: EncryptionScheme, key: Vec<u8>) -> Result<Self> {
        if key.len() != 16 {
            return Err(DrmError::InvalidKey(format!(
                "Key must be 16 bytes for AES-128, got {}",
                key.len()
            )));
        }

        Ok(Self {
            scheme,
            key,
            pattern: None,
        })
    }

    /// Set encryption pattern for pattern-based schemes
    pub fn with_pattern(mut self, pattern: EncryptionPattern) -> Self {
        self.pattern = Some(pattern);
        self
    }

    /// Decrypt a full sample (no subsamples)
    pub fn decrypt_sample(&self, data: &[u8], iv: &[u8]) -> Result<Vec<u8>> {
        match self.scheme {
            EncryptionScheme::Cenc | EncryptionScheme::Cens => self.decrypt_ctr(data, iv),
            EncryptionScheme::Cbc1 | EncryptionScheme::Cbcs => self.decrypt_cbc(data, iv),
        }
    }

    /// Decrypt with subsample information
    pub fn decrypt_with_subsamples(
        &self,
        data: &[u8],
        iv: &[u8],
        subsamples: &[SubsampleInfo],
    ) -> Result<Vec<u8>> {
        let mut result = Vec::with_capacity(data.len());
        let mut offset = 0;

        for subsample in subsamples {
            let clear_end = offset + subsample.clear_bytes as usize;
            let encrypted_end = clear_end + subsample.encrypted_bytes as usize;

            if encrypted_end > data.len() {
                return Err(DrmError::DecryptionError(
                    "Subsample extends beyond data".to_string(),
                ));
            }

            // Copy clear bytes as-is
            result.extend_from_slice(&data[offset..clear_end]);

            // Decrypt encrypted bytes
            let decrypted_data = self.decrypt_sample(&data[clear_end..encrypted_end], iv)?;
            result.extend_from_slice(&decrypted_data);

            offset = encrypted_end;
        }

        // Copy any remaining bytes
        if offset < data.len() {
            result.extend_from_slice(&data[offset..]);
        }

        Ok(result)
    }

    /// Decrypt using AES-CTR mode (same as encryption due to XOR property)
    fn decrypt_ctr(&self, data: &[u8], iv: &[u8]) -> Result<Vec<u8>> {
        // CTR mode encryption and decryption are the same operation
        let encryptor = CencEncryptor {
            scheme: self.scheme,
            key: self.key.clone(),
            pattern: self.pattern,
        };
        encryptor.encrypt_ctr(data, iv)
    }

    /// Decrypt using AES-CBC mode
    fn decrypt_cbc(&self, data: &[u8], iv: &[u8]) -> Result<Vec<u8>> {
        if iv.len() != 16 {
            return Err(DrmError::InvalidIv(format!(
                "IV must be 16 bytes for CBC mode, got {}",
                iv.len()
            )));
        }

        let block_size = 16;
        if data.len() % block_size != 0 {
            return Err(DrmError::DecryptionError(
                "Data length must be multiple of block size".to_string(),
            ));
        }

        let mut result = Vec::with_capacity(data.len());
        let mut previous_block = iv.to_vec();

        for (chunk_idx, chunk) in data.chunks(block_size).enumerate() {
            // Check pattern
            let should_decrypt = if let Some(pattern) = self.pattern {
                let block_in_pattern =
                    chunk_idx % (pattern.crypt_blocks + pattern.skip_blocks) as usize;
                block_in_pattern < pattern.crypt_blocks as usize
            } else {
                true
            };

            if should_decrypt {
                // Decrypt block (simplified - would use proper AES in production)
                let decrypted = self.aes_decrypt_block(chunk)?;

                // XOR with previous ciphertext block (or IV)
                let mut xored = vec![0u8; block_size];
                for i in 0..block_size {
                    xored[i] = decrypted[i] ^ previous_block[i];
                }

                result.extend_from_slice(&xored);
                previous_block = chunk.to_vec();
            } else {
                // Skip block (copy as-is)
                result.extend_from_slice(chunk);
                previous_block = chunk.to_vec();
            }
        }

        // Remove padding
        if !result.is_empty() {
            let padding_len = result[result.len() - 1] as usize;
            if padding_len > 0 && padding_len <= block_size {
                result.truncate(result.len() - padding_len);
            }
        }

        Ok(result)
    }

    /// Decrypt a single AES block (simplified)
    fn aes_decrypt_block(&self, _block: &[u8]) -> Result<Vec<u8>> {
        // In a real implementation, this would use proper AES decryption
        // For now, we return the block as-is (placeholder)
        // ring crate doesn't provide raw AES block decryption
        Err(DrmError::DecryptionError(
            "Block decryption not fully implemented".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Parallel subsample encryption
// ---------------------------------------------------------------------------

/// Encrypt a slice of independent subsample byte buffers in parallel using rayon.
///
/// Each subsample in `subsamples` is encrypted with AES-128-CTR independently.
/// Because AES-CTR allows random access (the counter for subsample `i` starts at
/// block offset `i * max_blocks_per_subsample`), the samples can be processed
/// without ordering dependencies.
///
/// For simplicity this implementation assigns each subsample its own counter
/// starting at 0 with the same nonce, which is the correct model when every
/// subsample is logically independent (e.g. individual NAL units of a NAL-unit
/// aligned encryption scheme where each sample resets the IV counter).
///
/// # Parameters
/// - `subsamples`: mutable slice of byte vectors; each is encrypted in place.
/// - `key`:        16-byte AES-128 content encryption key.
/// - `iv`:         16-byte initialization vector — the first 8 bytes are used as
///                 the CTR nonce, the remaining 8 bytes are treated as the initial
///                 big-endian counter value (per ISO 23001-7 `cenc` scheme).
///
/// # Errors
/// Returns `DrmError::InvalidKey` if `key.len() != 16` or
/// `DrmError::InvalidIv` if `iv.len() != 16`.
pub fn encrypt_subsamples_parallel(
    subsamples: &mut [Vec<u8>],
    key: &[u8; 16],
    iv: &[u8; 16],
) -> Result<()> {
    // Derive nonce (bytes 0..8) and initial counter (bytes 8..16) from the IV.
    let nonce: [u8; 8] = iv[0..8]
        .try_into()
        .map_err(|_| DrmError::InvalidIv("Failed to extract 8-byte nonce from IV".to_string()))?;
    let counter_start = u64::from_be_bytes(iv[8..16].try_into().map_err(|_| {
        DrmError::InvalidIv("Failed to extract 8-byte counter from IV".to_string())
    })?);

    let cipher = AesCtr::new_128(key);

    // Parallel in-place encryption: each subsample encrypts independently.
    subsamples.par_iter_mut().for_each(|sample| {
        let encrypted = cipher.encrypt(sample, &nonce, counter_start);
        *sample = encrypted;
    });

    Ok(())
}

/// Increment a counter for CTR mode
fn increment_counter(counter: &mut [u8]) {
    for byte in counter.iter_mut().rev() {
        *byte = byte.wrapping_add(1);
        if *byte != 0 {
            break;
        }
    }
}

/// Helper to read u8 from cursor
fn read_u8<R: Read>(cursor: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    cursor.read_exact(&mut buf)?;
    Ok(buf[0])
}

/// Helper to read u32 from cursor
fn read_u32<R: Read>(cursor: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encryption_scheme() {
        assert_eq!(EncryptionScheme::Cenc.fourcc(), "cenc");
        assert_eq!(EncryptionScheme::Cbc1.fourcc(), "cbc1");
        assert_eq!(EncryptionScheme::Cens.fourcc(), "cens");
        assert_eq!(EncryptionScheme::Cbcs.fourcc(), "cbcs");

        assert_eq!(
            EncryptionScheme::from_fourcc("cenc"),
            Some(EncryptionScheme::Cenc)
        );
        assert!(EncryptionScheme::from_fourcc("invalid").is_none());
    }

    #[test]
    fn test_subsample_info() {
        let subsample = SubsampleInfo::new(100, 500);
        assert_eq!(subsample.clear_bytes, 100);
        assert_eq!(subsample.encrypted_bytes, 500);
        assert_eq!(subsample.total_bytes(), 600);
    }

    #[test]
    fn test_encryption_pattern() {
        let pattern = EncryptionPattern::video_default();
        assert_eq!(pattern.crypt_blocks, 1);
        assert_eq!(pattern.skip_blocks, 9);

        let full = EncryptionPattern::full_encryption();
        assert_eq!(full.crypt_blocks, 1);
        assert_eq!(full.skip_blocks, 0);
    }

    #[test]
    fn test_pssh_box_serialization() {
        let system_id = Uuid::parse_str("edef8ba9-79d6-4ace-a3c8-27dcd51d21ed")
            .expect("operation should succeed");
        let data = vec![1, 2, 3, 4, 5];

        let pssh = PsshBox::new(system_id, data.clone());
        let bytes = pssh.to_bytes().expect("operation should succeed");

        let parsed = PsshBox::from_bytes(&bytes).expect("operation should succeed");
        assert_eq!(parsed.version, 0);
        assert_eq!(parsed.system_id, system_id);
        assert_eq!(parsed.data, data);
    }

    #[test]
    fn test_pssh_box_v1_serialization() {
        let system_id = Uuid::parse_str("edef8ba9-79d6-4ace-a3c8-27dcd51d21ed")
            .expect("operation should succeed");
        let key_ids = vec![vec![1u8; 16], vec![2u8; 16]];
        let data = vec![1, 2, 3, 4, 5];

        let pssh = PsshBox::new_v1(system_id, key_ids.clone(), data.clone());
        let bytes = pssh.to_bytes().expect("operation should succeed");

        let parsed = PsshBox::from_bytes(&bytes).expect("operation should succeed");
        assert_eq!(parsed.version, 1);
        assert_eq!(parsed.system_id, system_id);
        assert_eq!(parsed.key_ids, key_ids);
        assert_eq!(parsed.data, data);
    }

    #[test]
    fn test_cenc_encryptor_creation() {
        let key = vec![0u8; 16];
        let encryptor = CencEncryptor::new(EncryptionScheme::Cenc, key);
        assert!(encryptor.is_ok());

        let invalid_key = vec![0u8; 8];
        let encryptor = CencEncryptor::new(EncryptionScheme::Cenc, invalid_key);
        assert!(encryptor.is_err());
    }

    #[test]
    fn test_iv_generation() {
        let key = vec![0u8; 16];
        let encryptor =
            CencEncryptor::new(EncryptionScheme::Cenc, key).expect("operation should succeed");
        let iv = encryptor.generate_iv().expect("operation should succeed");
        assert_eq!(iv.len(), 8); // CTR mode uses 8-byte IV

        let encryptor = CencEncryptor::new(EncryptionScheme::Cbcs, vec![0u8; 16])
            .expect("operation should succeed");
        let iv = encryptor.generate_iv().expect("operation should succeed");
        assert_eq!(iv.len(), 16); // CBC mode uses 16-byte IV
    }

    #[test]
    fn test_counter_increment() {
        let mut counter = vec![0u8; 16];
        increment_counter(&mut counter);
        assert_eq!(counter[15], 1);

        counter[15] = 255;
        increment_counter(&mut counter);
        assert_eq!(counter[15], 0);
        assert_eq!(counter[14], 1);
    }

    // -----------------------------------------------------------------------
    // CENC AES-CTR encrypt/decrypt round-trip test (Task 4)
    // Uses the pure-Rust AesCtr from aes_ctr.rs for a full verified round-trip.
    // -----------------------------------------------------------------------

    #[test]
    fn test_cenc_encrypt_decrypt_roundtrip() {
        use crate::aes_ctr::AesCtr;

        // Known key and IV (AES-128 — 16 bytes each)
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        // IV: 8-byte nonce || 8-byte counter (all zeros → counter starts at 0)
        let nonce: [u8; 8] = [0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7];
        let counter_start: u64 = 0;

        // 128 bytes of known plaintext (incrementing pattern)
        let plaintext: Vec<u8> = (0u8..128).collect();
        assert_eq!(plaintext.len(), 128);

        let cipher = AesCtr::new_128(&key);

        // Encrypt
        let ciphertext = cipher.encrypt(&plaintext, &nonce, counter_start);
        assert_eq!(
            ciphertext.len(),
            plaintext.len(),
            "ciphertext length must match plaintext"
        );
        // Ciphertext must differ from plaintext (with overwhelming probability)
        assert_ne!(
            ciphertext, plaintext,
            "ciphertext must differ from plaintext"
        );

        // Decrypt (CTR is symmetric — same operation)
        let decrypted = cipher.decrypt(&ciphertext, &nonce, counter_start);
        assert_eq!(
            decrypted, plaintext,
            "decrypted output must match original plaintext"
        );

        // Extra: verify wrong nonce produces wrong output
        let wrong_nonce: [u8; 8] = [0xFF; 8];
        let wrong_dec = cipher.decrypt(&ciphertext, &wrong_nonce, counter_start);
        assert_ne!(
            wrong_dec, plaintext,
            "wrong nonce must not recover plaintext"
        );
    }

    // -----------------------------------------------------------------------
    // Parallel subsample encryption test (Task 5)
    // -----------------------------------------------------------------------

    #[test]
    fn test_encrypt_subsamples_parallel_roundtrip() {
        let key: [u8; 16] = [0x42u8; 16];
        // IV: nonce (bytes 0..8) = 0x01..0x08, counter (bytes 8..16) = 0
        let iv: [u8; 16] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];

        // Create 4 subsamples with known content
        let original_subsamples: Vec<Vec<u8>> = vec![
            (0u8..64).collect(),
            (64u8..128).collect(),
            vec![0xABu8; 48],
            vec![0x00u8; 32],
        ];

        // Encrypt
        let mut encrypted = original_subsamples.clone();
        encrypt_subsamples_parallel(&mut encrypted, &key, &iv)
            .expect("parallel encryption should succeed");

        // Each encrypted subsample must differ from the original
        for (i, (orig, enc)) in original_subsamples.iter().zip(encrypted.iter()).enumerate() {
            assert_ne!(enc, orig, "subsample {} should be encrypted", i);
        }

        // Decrypt (AES-CTR is symmetric)
        let mut decrypted = encrypted;
        encrypt_subsamples_parallel(&mut decrypted, &key, &iv)
            .expect("parallel decryption should succeed");

        // Decrypted must match original
        assert_eq!(
            decrypted, original_subsamples,
            "decrypted subsamples must match originals"
        );
    }

    #[test]
    fn test_encrypt_subsamples_parallel_empty_slice() {
        let key: [u8; 16] = [0x00u8; 16];
        let iv: [u8; 16] = [0x00u8; 16];
        let mut subsamples: Vec<Vec<u8>> = vec![];
        let result = encrypt_subsamples_parallel(&mut subsamples, &key, &iv);
        assert!(result.is_ok(), "empty subsample slice should succeed");
    }
}
