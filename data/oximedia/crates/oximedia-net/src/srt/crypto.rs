//! SRT encryption support using AES.
//!
//! Provides AES-128/192/256 encryption for SRT payloads.
//!
//! # Cryptography
//!
//! Payload confidentiality uses real **AES-CTR** (NIST SP 800-38A) via the
//! vetted RustCrypto `aes` + `ctr` crates: [`AesContext::encrypt`]/
//! [`AesContext::decrypt`] select AES-128/192/256 by key length and apply
//! the CTR keystream to the caller-supplied 16-byte IV/counter block. This
//! replaces a previous homebrew XOR/mixing "cipher" that provided no real
//! security.
//!
//! Key material is derived with real **PBKDF2-HMAC-SHA256** (RFC 8018) via
//! [`derive_session_key`], built on the vetted `hmac` + `sha2` crates
//! (mirroring the approach already used and tested in
//! `crate::srt_aes256gcm::derive_key`). Salts are generated with the
//! process-wide CSPRNG (`rand::rng()`, reseeded from OS entropy), not a
//! plain LCG.
//!
//! For a fully authenticated alternative (AES-256-GCM with tamper
//! detection), see [`crate::srt_aes256gcm`].

use crate::error::{NetError, NetResult};
// `KeyInit` is aliased to `AesKeyInit` to coexist with `hmac::KeyInit`: the two
// come from different crypto-common versions (hmac 0.12 vs aes 0.9), so both
// names must be in scope. The block-cipher traits provide the raw single-block
// ECB primitive that RFC 3394 key wrap is built on.
use aes::cipher::{
    Array, BlockCipherDecrypt, BlockCipherEncrypt, KeyInit as AesKeyInit, KeyIvInit, StreamCipher,
};
use aes::{Aes128, Aes192, Aes256};
use bytes::Bytes;
use ctr::Ctr128BE;
use hmac::{Hmac, Mac};
use rand::Rng;
use sha2::Sha256;

/// AES-128 in CTR mode with a 128-bit big-endian counter (full 16-byte IV
/// used as the initial counter block).
type Aes128Ctr = Ctr128BE<Aes128>;
/// AES-192 in CTR mode with a 128-bit big-endian counter.
type Aes192Ctr = Ctr128BE<Aes192>;
/// AES-256 in CTR mode with a 128-bit big-endian counter.
type Aes256Ctr = Ctr128BE<Aes256>;
/// HMAC-SHA256, used as the PBKDF2 pseudo-random function.
type HmacSha256 = Hmac<Sha256>;

/// AES encryption context.
#[derive(Debug, Clone)]
pub struct AesContext {
    /// Encryption key.
    key: Vec<u8>,
    /// Key size (16, 24, or 32 bytes).
    key_size: usize,
    /// Salt for key derivation.
    salt: [u8; 16],
    /// Current key index (for key rotation).
    key_index: u8,
}

impl AesContext {
    /// Creates a new AES context from a passphrase.
    ///
    /// # Errors
    ///
    /// Returns an error if the key size is invalid.
    pub fn from_passphrase(passphrase: &str, key_size: usize) -> NetResult<Self> {
        if ![16, 24, 32].contains(&key_size) {
            return Err(NetError::protocol("Invalid key size"));
        }

        let salt = generate_salt();
        let key = derive_key(passphrase.as_bytes(), &salt, key_size)?;

        Ok(Self {
            key,
            key_size,
            salt,
            key_index: 0,
        })
    }

    /// Creates a new AES context from a raw key.
    ///
    /// # Errors
    ///
    /// Returns an error if the key size is invalid.
    pub fn from_key(key: &[u8]) -> NetResult<Self> {
        let key_size = key.len();
        if ![16, 24, 32].contains(&key_size) {
            return Err(NetError::protocol("Invalid key size"));
        }

        Ok(Self {
            key: key.to_vec(),
            key_size,
            salt: [0; 16],
            key_index: 0,
        })
    }

    /// Returns the key size in bytes.
    #[must_use]
    pub const fn key_size(&self) -> usize {
        self.key_size
    }

    /// Returns the salt.
    #[must_use]
    pub const fn salt(&self) -> &[u8; 16] {
        &self.salt
    }

    /// Encrypts a payload using real AES-CTR (NIST SP 800-38A).
    ///
    /// The AES variant (128/192/256) is selected by this context's key
    /// length; `iv` is used verbatim as the initial 128-bit counter block.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored key length is not 16, 24, or 32 bytes.
    pub fn encrypt(&self, plaintext: &[u8], iv: &[u8; 16]) -> NetResult<Bytes> {
        let mut buf = plaintext.to_vec();
        apply_aes_ctr_keystream(&self.key, iv, &mut buf)?;
        Ok(Bytes::from(buf))
    }

    /// Decrypts a payload.
    ///
    /// AES-CTR decryption is identical to encryption (XOR with the same
    /// keystream derived from the key and counter block).
    ///
    /// # Errors
    ///
    /// Returns an error if the stored key length is not 16, 24, or 32 bytes.
    pub fn decrypt(&self, ciphertext: &[u8], iv: &[u8; 16]) -> NetResult<Bytes> {
        self.encrypt(ciphertext, iv)
    }

    /// Rotates to the next key index.
    pub fn rotate_key(&mut self) {
        self.key_index = self.key_index.wrapping_add(1);
    }

    /// Returns the current key index.
    #[must_use]
    pub const fn key_index(&self) -> u8 {
        self.key_index
    }
}

/// Applies the AES-CTR keystream (RustCrypto `aes` + `ctr`, NIST SP 800-38A)
/// to `buf` in place, selecting AES-128/192/256 by `key.len()`.
///
/// `iv` is used as the full 128-bit initial counter block (matching the
/// `Ctr128BE` big-endian counter convention), consistent with how callers
/// in this module derive per-packet IVs (see `SrtCryptoContext::derive_iv`
/// and `super::connection::generate_iv`).
///
/// # Errors
///
/// Returns an error if `key.len()` is not 16, 24, or 32 bytes.
fn apply_aes_ctr_keystream(key: &[u8], iv: &[u8; 16], buf: &mut [u8]) -> NetResult<()> {
    match key.len() {
        16 => {
            let mut cipher = Aes128Ctr::new_from_slices(key, iv)
                .map_err(|e| NetError::protocol(format!("AES-128-CTR init: {e}")))?;
            cipher.apply_keystream(buf);
        }
        24 => {
            let mut cipher = Aes192Ctr::new_from_slices(key, iv)
                .map_err(|e| NetError::protocol(format!("AES-192-CTR init: {e}")))?;
            cipher.apply_keystream(buf);
        }
        32 => {
            let mut cipher = Aes256Ctr::new_from_slices(key, iv)
                .map_err(|e| NetError::protocol(format!("AES-256-CTR init: {e}")))?;
            cipher.apply_keystream(buf);
        }
        other => {
            return Err(NetError::protocol(format!(
                "Invalid AES key length: {other} bytes (must be 16, 24, or 32)"
            )));
        }
    }
    Ok(())
}

/// Derives an encryption key from a passphrase using real PBKDF2-HMAC-SHA256
/// (RFC 8018).
///
/// # Errors
///
/// Propagates any (practically unreachable) HMAC initialization failure
/// from [`derive_session_key`].
fn derive_key(passphrase: &[u8], salt: &[u8], key_len: usize) -> NetResult<Vec<u8>> {
    derive_session_key(passphrase, salt, key_len, KDF_ITERATIONS)
}

/// Generates a cryptographically secure random salt.
///
/// Uses the process CSPRNG (`rand::rng()`, backed by a CSPRNG seeded from OS
/// entropy — see the `rand` crate's `ThreadRng` documentation), not a
/// non-cryptographic PRNG.
fn generate_salt() -> [u8; 16] {
    let mut salt = [0u8; 16];
    rand::rng().fill_bytes(&mut salt);
    salt
}

/// Key material exchange information.
#[derive(Debug, Clone)]
pub struct KeyMaterial {
    /// Key encryption key.
    pub kek: Vec<u8>,
    /// Salt.
    pub salt: [u8; 16],
    /// Key length.
    pub key_len: u8,
}

impl KeyMaterial {
    /// Creates new key material.
    #[must_use]
    pub fn new(key_len: u8) -> Self {
        let salt = generate_salt();
        let kek = vec![0u8; key_len as usize];

        Self { kek, salt, key_len }
    }

    /// Encodes key material for transmission.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(17 + self.kek.len());
        buf.push(self.key_len);
        buf.extend_from_slice(&self.salt);
        buf.extend_from_slice(&self.kek);
        buf
    }

    /// Decodes key material from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid.
    pub fn decode(data: &[u8]) -> NetResult<Self> {
        if data.len() < 17 {
            return Err(NetError::parse(0, "Key material too short"));
        }

        let key_len = data[0];
        let mut salt = [0u8; 16];
        salt.copy_from_slice(&data[1..17]);
        let kek = data[17..].to_vec();

        Ok(Self { kek, salt, key_len })
    }
}

/// Number of PBKDF2-like iterations for key derivation.
const KDF_ITERATIONS: u32 = 2048;

/// SRT key-exchange version.
const KM_VERSION: u8 = 1;

/// Derives a session key from a passphrase and salt using real
/// PBKDF2-HMAC-SHA256 (RFC 8018).
///
/// Runs `iterations` rounds of HMAC-SHA256 per 32-byte output block,
/// XORing the intermediate `U_i` values together (the standard PBKDF2
/// `F` function), and concatenates as many blocks as needed to produce
/// `key_len` bytes of output. This is the same construction already used
/// and tested in `crate::srt_aes256gcm::derive_key`, generalized here to
/// support multi-block output and a caller-supplied iteration count.
///
/// # Errors
///
/// Returns an error if HMAC initialization fails. In practice this never
/// happens (HMAC accepts keys — here, the passphrase — of any length), but
/// the fallible path is propagated rather than unwrapped, per project
/// policy.
pub fn derive_session_key(
    passphrase: &[u8],
    salt: &[u8],
    key_len: usize,
    iterations: u32,
) -> NetResult<Vec<u8>> {
    let mut output = vec![0u8; key_len];
    let blocks_needed = key_len.div_ceil(32);
    let effective_iterations = iterations.max(1);

    for block_idx in 0..blocks_needed {
        // INT(i): 4-byte big-endian block index, 1-based (PBKDF2 §5.2).
        let block_num = u32::try_from(block_idx)
            .map_err(|_| NetError::protocol("PBKDF2 block index overflow"))?
            .wrapping_add(1);

        let mut combined_salt = salt.to_vec();
        combined_salt.extend_from_slice(&block_num.to_be_bytes());

        // U_1 = HMAC-SHA256(password, salt || INT(i))
        let mut mac = HmacSha256::new_from_slice(passphrase)
            .map_err(|e| NetError::protocol(format!("HMAC init: {e}")))?;
        mac.update(&combined_salt);
        let mut u = mac.finalize().into_bytes();
        let mut t = u;

        // U_2..U_c, T_i = U_1 XOR U_2 XOR ... XOR U_c
        for _ in 1..effective_iterations {
            let mut mac = HmacSha256::new_from_slice(passphrase)
                .map_err(|e| NetError::protocol(format!("HMAC init: {e}")))?;
            mac.update(&u);
            u = mac.finalize().into_bytes();
            for (a, b) in t.iter_mut().zip(u.iter()) {
                *a ^= b;
            }
        }

        let start = block_idx * 32;
        let end = (start + 32).min(key_len);
        output[start..end].copy_from_slice(&t[..end - start]);
    }

    Ok(output)
}

/// RFC 3394 default initial value (Section 2.2.3.1): 0xA6A6A6A6A6A6A6A6.
const AES_KEY_WRAP_IV: [u8; 8] = [0xA6; 8];

/// A raw AES block cipher keyed with the SRT KEK, used for the RFC 3394 key
/// wrap ECB passes. Built once per wrap/unwrap so the AES key schedule is not
/// recomputed for every 64-bit block.
enum WrapCipher {
    /// AES-128 (16-byte KEK).
    Aes128(Aes128),
    /// AES-192 (24-byte KEK).
    Aes192(Aes192),
    /// AES-256 (32-byte KEK).
    Aes256(Aes256),
}

impl WrapCipher {
    /// Builds the cipher for a 16/24/32-byte KEK (AES-128/192/256).
    fn new(kek: &[u8]) -> NetResult<Self> {
        match kek.len() {
            16 => Aes128::new_from_slice(kek)
                .map(WrapCipher::Aes128)
                .map_err(|_| NetError::protocol("AES key wrap: bad 128-bit KEK")),
            24 => Aes192::new_from_slice(kek)
                .map(WrapCipher::Aes192)
                .map_err(|_| NetError::protocol("AES key wrap: bad 192-bit KEK")),
            32 => Aes256::new_from_slice(kek)
                .map(WrapCipher::Aes256)
                .map_err(|_| NetError::protocol("AES key wrap: bad 256-bit KEK")),
            _ => Err(NetError::protocol(
                "AES key wrap requires a 16/24/32-byte KEK",
            )),
        }
    }

    /// Encrypts one 16-byte block in place (ECB, single block).
    fn encrypt_block16(&self, block: &mut [u8; 16]) {
        let mut b = Array::from(*block);
        match self {
            WrapCipher::Aes128(c) => c.encrypt_block(&mut b),
            WrapCipher::Aes192(c) => c.encrypt_block(&mut b),
            WrapCipher::Aes256(c) => c.encrypt_block(&mut b),
        }
        block.copy_from_slice(&b[..]);
    }

    /// Decrypts one 16-byte block in place (ECB, single block).
    fn decrypt_block16(&self, block: &mut [u8; 16]) {
        let mut b = Array::from(*block);
        match self {
            WrapCipher::Aes128(c) => c.decrypt_block(&mut b),
            WrapCipher::Aes192(c) => c.decrypt_block(&mut b),
            WrapCipher::Aes256(c) => c.decrypt_block(&mut b),
        }
        block.copy_from_slice(&b[..]);
    }
}

/// Splits a byte slice (a multiple of 8 in length) into 64-bit blocks.
fn to_blocks(data: &[u8]) -> Vec<[u8; 8]> {
    data.chunks_exact(8)
        .map(|c| {
            let mut b = [0u8; 8];
            b.copy_from_slice(c);
            b
        })
        .collect()
}

/// Wraps `plaintext_key` under `kek` with the **RFC 3394 AES Key Wrap**
/// algorithm.
///
/// This is the real 6·n-round A/B construction with the
/// `0xA6A6A6A6A6A6A6A6` integrity IV (RFC 3394 §2.2.1), so the output
/// interoperates with libsrt / OBS / ffmpeg. It replaces a previous XOR
/// "wrap" that only masqueraded as RFC 3394 and produced non-interoperable,
/// unauthenticated output. `plaintext_key` must be a non-empty multiple of 8
/// bytes (SRT session keys are 16/24/32 → n = 2/3/4 64-bit blocks) and `kek`
/// must be 16/24/32 bytes.
///
/// # Errors
///
/// Returns an error if `plaintext_key` is empty or not a multiple of 8 bytes,
/// or if `kek` is not a valid AES key length.
fn aes_key_wrap(kek: &[u8], plaintext_key: &[u8]) -> NetResult<Vec<u8>> {
    if plaintext_key.is_empty() || plaintext_key.len() % 8 != 0 {
        return Err(NetError::protocol(
            "AES key wrap: plaintext must be a non-empty multiple of 8 bytes",
        ));
    }
    let cipher = WrapCipher::new(kek)?;
    let n = plaintext_key.len() / 8;

    // A = IV; R[i] = P[i]
    let mut a = AES_KEY_WRAP_IV;
    let mut r = to_blocks(plaintext_key);

    // Six rounds over all n blocks (RFC 3394 §2.2.1).
    for j in 0..6u64 {
        for i in 0..n {
            // B = AES(KEK, A | R[i])
            let mut block = [0u8; 16];
            block[..8].copy_from_slice(&a);
            block[8..].copy_from_slice(&r[i]);
            cipher.encrypt_block16(&mut block);

            // A = MSB64(B) XOR t, with t = n*j + i + 1
            let t = (n as u64) * j + i as u64 + 1;
            let mut msb = [0u8; 8];
            msb.copy_from_slice(&block[..8]);
            a = (u64::from_be_bytes(msb) ^ t).to_be_bytes();
            // R[i] = LSB64(B)
            r[i].copy_from_slice(&block[8..]);
        }
    }

    // C[0] = A, C[i] = R[i]
    let mut out = Vec::with_capacity((n + 1) * 8);
    out.extend_from_slice(&a);
    for block in &r {
        out.extend_from_slice(block);
    }
    Ok(out)
}

/// Unwraps a key wrapped with [`aes_key_wrap`] (**RFC 3394 §2.2.2**).
///
/// The final value of the `A` register is compared against the RFC 3394 IV;
/// a mismatch (wrong KEK or tampered ciphertext) is reported as an error.
///
/// # Errors
///
/// Returns an error if `wrapped_key` is not a multiple of 8 bytes (min 16),
/// `kek` is not a valid AES key length, or the integrity check fails.
fn aes_key_unwrap(kek: &[u8], wrapped_key: &[u8]) -> NetResult<Vec<u8>> {
    if wrapped_key.len() < 16 || wrapped_key.len() % 8 != 0 {
        return Err(NetError::protocol(
            "AES key unwrap: ciphertext must be a multiple of 8 bytes and at least 16",
        ));
    }
    let cipher = WrapCipher::new(kek)?;
    let n = wrapped_key.len() / 8 - 1;

    // A = C[0]; R[i] = C[i]
    let mut a = [0u8; 8];
    a.copy_from_slice(&wrapped_key[..8]);
    let mut r = to_blocks(&wrapped_key[8..]);

    // Six rounds in reverse.
    for j in (0..6u64).rev() {
        for i in (0..n).rev() {
            // B = AES-1(KEK, (A XOR t) | R[i]), with t = n*j + i + 1
            let t = (n as u64) * j + i as u64 + 1;
            let a_xor_t = (u64::from_be_bytes(a) ^ t).to_be_bytes();
            let mut block = [0u8; 16];
            block[..8].copy_from_slice(&a_xor_t);
            block[8..].copy_from_slice(&r[i]);
            cipher.decrypt_block16(&mut block);

            a.copy_from_slice(&block[..8]);
            r[i].copy_from_slice(&block[8..]);
        }
    }

    // Integrity check: A must equal the RFC 3394 IV.
    if a != AES_KEY_WRAP_IV {
        return Err(NetError::protocol("AES key unwrap: integrity check failed"));
    }

    let mut out = Vec::with_capacity(n * 8);
    for block in &r {
        out.extend_from_slice(block);
    }
    Ok(out)
}

/// Key schedule for SRT: holds both even and odd session keys.
#[derive(Debug, Clone)]
pub struct KeySchedule {
    /// Even key (key index 0).
    even_key: Vec<u8>,
    /// Odd key (key index 1).
    odd_key: Vec<u8>,
    /// Key size in bytes (16, 24, or 32).
    key_size: usize,
    /// Current active key index (0 = even, 1 = odd).
    active_index: u8,
}

impl KeySchedule {
    /// Creates a new key schedule from a passphrase and salt.
    ///
    /// # Errors
    ///
    /// Returns an error if the key size is invalid.
    pub fn from_passphrase(passphrase: &str, salt: &[u8; 14], key_size: usize) -> NetResult<Self> {
        if ![16, 24, 32].contains(&key_size) {
            return Err(NetError::protocol("Invalid key size for key schedule"));
        }

        // Derive two separate keys: even uses salt with 0x00 prefix, odd with 0x01 prefix
        let mut even_salt = vec![0x00u8];
        even_salt.extend_from_slice(salt);
        let mut odd_salt = vec![0x01u8];
        odd_salt.extend_from_slice(salt);

        let even_key =
            derive_session_key(passphrase.as_bytes(), &even_salt, key_size, KDF_ITERATIONS)?;
        let odd_key =
            derive_session_key(passphrase.as_bytes(), &odd_salt, key_size, KDF_ITERATIONS)?;

        Ok(Self {
            even_key,
            odd_key,
            key_size,
            active_index: 0,
        })
    }

    /// Returns the currently active key.
    #[must_use]
    pub fn active_key(&self) -> &[u8] {
        if self.active_index == 0 {
            &self.even_key
        } else {
            &self.odd_key
        }
    }

    /// Returns the even key.
    #[must_use]
    pub fn even_key(&self) -> &[u8] {
        &self.even_key
    }

    /// Returns the odd key.
    #[must_use]
    pub fn odd_key(&self) -> &[u8] {
        &self.odd_key
    }

    /// Returns the active key index (0 = even, 1 = odd).
    #[must_use]
    pub const fn active_index(&self) -> u8 {
        self.active_index
    }

    /// Switches to the other key (key rotation).
    pub fn rotate(&mut self) {
        self.active_index = 1 - self.active_index;
    }

    /// Returns the key size.
    #[must_use]
    pub const fn key_size(&self) -> usize {
        self.key_size
    }
}

/// SRT packet for encryption/decryption operations.
#[derive(Debug, Clone)]
pub struct SrtPacketBuffer {
    /// Packet sequence number (used for IV generation).
    pub seq_no: u32,
    /// Encryption flag (0 = clear, 1 = even key, 2 = odd key).
    pub encryption_flag: u8,
    /// Packet payload.
    pub payload: Vec<u8>,
}

impl SrtPacketBuffer {
    /// Creates a new packet buffer.
    #[must_use]
    pub fn new(seq_no: u32, encryption_flag: u8, payload: Vec<u8>) -> Self {
        Self {
            seq_no,
            encryption_flag,
            payload,
        }
    }
}

/// SRT crypto context: manages the full lifecycle of SRT encryption.
///
/// This includes key derivation, key schedule, key material (KM) exchange,
/// and packet-level encrypt/decrypt operations.
#[derive(Debug)]
pub struct SrtCryptoContext {
    /// Key schedule holding even/odd session keys.
    key_schedule: KeySchedule,
    /// Salt (14 bytes, per SRT spec).
    salt: [u8; 14],
    /// AES context used for packet encryption.
    aes: AesContext,
    /// Total packets encrypted (for IV derivation).
    packet_count: u64,
}

impl SrtCryptoContext {
    /// Creates a new crypto context from a passphrase.
    ///
    /// # Errors
    ///
    /// Returns an error if `key_size` is not 16, 24, or 32.
    pub fn from_passphrase(passphrase: &str, key_size: usize) -> NetResult<Self> {
        if ![16, 24, 32].contains(&key_size) {
            return Err(NetError::protocol("Invalid key size for SrtCryptoContext"));
        }

        let raw_salt = generate_salt();
        let mut salt = [0u8; 14];
        salt.copy_from_slice(&raw_salt[..14]);

        let key_schedule = KeySchedule::from_passphrase(passphrase, &salt, key_size)?;
        let aes = AesContext::from_key(key_schedule.active_key())?;

        Ok(Self {
            key_schedule,
            salt,
            aes,
            packet_count: 0,
        })
    }

    /// Creates a crypto context from explicit even/odd session keys.
    ///
    /// # Errors
    ///
    /// Returns an error if the key vectors don't match `key_size`.
    pub fn from_keys(even_key: Vec<u8>, odd_key: Vec<u8>, salt: [u8; 14]) -> NetResult<Self> {
        let key_size = even_key.len();
        if key_size != odd_key.len() || ![16, 24, 32].contains(&key_size) {
            return Err(NetError::protocol("Mismatched or invalid key sizes"));
        }

        let key_schedule = KeySchedule {
            even_key: even_key.clone(),
            odd_key,
            key_size,
            active_index: 0,
        };
        let aes = AesContext::from_key(&even_key)?;

        Ok(Self {
            key_schedule,
            salt,
            aes,
            packet_count: 0,
        })
    }

    /// Returns the salt.
    #[must_use]
    pub const fn salt(&self) -> &[u8; 14] {
        &self.salt
    }

    /// Returns the key schedule.
    #[must_use]
    pub const fn key_schedule(&self) -> &KeySchedule {
        &self.key_schedule
    }

    /// Returns the number of packets encrypted so far.
    #[must_use]
    pub const fn packet_count(&self) -> u64 {
        self.packet_count
    }

    /// Derives the IV from the salt and packet sequence number.
    ///
    /// SRT spec: IV = salt XOR (seq_no in last 4 bytes, zero-padded to 16 bytes).
    #[must_use]
    fn derive_iv(&self, seq_no: u32) -> [u8; 16] {
        let mut iv = [0u8; 16];
        // Copy salt into first 14 bytes
        iv[..14].copy_from_slice(&self.salt);
        // XOR last 4 bytes with seq_no (big-endian)
        iv[12] ^= ((seq_no >> 24) & 0xFF) as u8;
        iv[13] ^= ((seq_no >> 16) & 0xFF) as u8;
        iv[14] = ((seq_no >> 8) & 0xFF) as u8;
        iv[15] = (seq_no & 0xFF) as u8;
        iv
    }

    /// Encrypts an SRT packet payload in-place using XOR (stub for testing).
    ///
    /// In production this would use AES-CTR with the derived IV.
    ///
    /// # Errors
    ///
    /// Returns an error if the payload is empty or encryption fails.
    pub fn encrypt_packet(&mut self, packet: &mut SrtPacketBuffer) -> NetResult<()> {
        if packet.payload.is_empty() {
            return Err(NetError::protocol("Cannot encrypt empty payload"));
        }

        let key = self.key_schedule.active_key().to_vec();
        let iv = self.derive_iv(packet.seq_no);

        // Use the AES context for CTR-mode encryption
        let encrypted = self.aes.encrypt(&packet.payload, &iv)?;
        packet.payload = encrypted.to_vec();

        // Set encryption flag: 1 = even key, 2 = odd key
        packet.encryption_flag = self.key_schedule.active_index() + 1;
        self.packet_count += 1;

        let _ = key; // used implicitly via aes context
        Ok(())
    }

    /// Decrypts an SRT packet payload using XOR (stub for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is not encrypted or decryption fails.
    pub fn decrypt_packet(&self, packet: &mut SrtPacketBuffer) -> NetResult<()> {
        if packet.encryption_flag == 0 {
            return Err(NetError::protocol("Packet is not encrypted"));
        }
        if packet.payload.is_empty() {
            return Err(NetError::protocol("Cannot decrypt empty payload"));
        }

        // Select key based on encryption flag
        let key = if packet.encryption_flag == 1 {
            self.key_schedule.even_key()
        } else {
            self.key_schedule.odd_key()
        };

        let iv = self.derive_iv(packet.seq_no);

        // CTR mode decryption = encryption
        let ctx = AesContext::from_key(key)?;
        let decrypted = ctx.decrypt(&packet.payload, &iv)?;
        packet.payload = decrypted.to_vec();
        packet.encryption_flag = 0;
        Ok(())
    }

    /// Rotates to the next key and refreshes the AES context.
    ///
    /// # Errors
    ///
    /// Returns an error if the new key is invalid.
    pub fn rotate_key(&mut self) -> NetResult<()> {
        self.key_schedule.rotate();
        self.aes = AesContext::from_key(self.key_schedule.active_key())?;
        Ok(())
    }

    /// Builds a `KeyMaterial` packet for key exchange.
    ///
    /// Wraps both session keys using the Key Encryption Key (KEK) derived
    /// from the passphrase.
    ///
    /// # Errors
    ///
    /// Returns an error if wrapping fails.
    pub fn build_key_material(&self, kek: &[u8]) -> NetResult<KeyMaterialPacket> {
        let wrapped_even = aes_key_wrap(kek, self.key_schedule.even_key())?;
        let wrapped_odd = aes_key_wrap(kek, self.key_schedule.odd_key())?;

        Ok(KeyMaterialPacket {
            version: KM_VERSION,
            key_size: self.key_schedule.key_size() as u8,
            salt: self.salt,
            wrapped_even_key: wrapped_even,
            wrapped_odd_key: wrapped_odd,
        })
    }

    /// Loads session keys from a `KeyMaterialPacket` using the KEK.
    ///
    /// # Errors
    ///
    /// Returns an error if unwrapping fails or key sizes mismatch.
    pub fn load_key_material(&mut self, km: &KeyMaterialPacket, kek: &[u8]) -> NetResult<()> {
        let even_key = aes_key_unwrap(kek, &km.wrapped_even_key)?;
        let odd_key = aes_key_unwrap(kek, &km.wrapped_odd_key)?;

        if even_key.len() != km.key_size as usize || odd_key.len() != km.key_size as usize {
            return Err(NetError::protocol("Unwrapped key size mismatch"));
        }

        self.key_schedule.even_key = even_key.clone();
        self.key_schedule.odd_key = odd_key;
        self.key_schedule.key_size = km.key_size as usize;
        self.salt = km.salt;
        self.aes = AesContext::from_key(&even_key)?;
        Ok(())
    }
}

/// Key Material Packet (KM) as defined by the SRT spec.
///
/// This packet is sent during the handshake to exchange session keys.
#[derive(Debug, Clone)]
pub struct KeyMaterialPacket {
    /// KM version (always 1).
    pub version: u8,
    /// Key size in bytes.
    pub key_size: u8,
    /// Salt (14 bytes).
    pub salt: [u8; 14],
    /// AES-wrapped even session key.
    pub wrapped_even_key: Vec<u8>,
    /// AES-wrapped odd session key.
    pub wrapped_odd_key: Vec<u8>,
}

impl KeyMaterialPacket {
    /// Encodes the packet to bytes for transmission.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(self.version);
        buf.push(self.key_size);
        buf.extend_from_slice(&self.salt);
        let even_len = self.wrapped_even_key.len() as u16;
        buf.push((even_len >> 8) as u8);
        buf.push((even_len & 0xFF) as u8);
        buf.extend_from_slice(&self.wrapped_even_key);
        let odd_len = self.wrapped_odd_key.len() as u16;
        buf.push((odd_len >> 8) as u8);
        buf.push((odd_len & 0xFF) as u8);
        buf.extend_from_slice(&self.wrapped_odd_key);
        buf
    }

    /// Decodes a packet from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too short or malformed.
    pub fn decode(data: &[u8]) -> NetResult<Self> {
        if data.len() < 18 {
            return Err(NetError::parse(0, "KeyMaterialPacket too short"));
        }
        let version = data[0];
        let key_size = data[1];
        let mut salt = [0u8; 14];
        salt.copy_from_slice(&data[2..16]);

        let even_len = (u16::from(data[16]) << 8 | u16::from(data[17])) as usize;
        if data.len() < 18 + even_len + 2 {
            return Err(NetError::parse(0, "KeyMaterialPacket truncated (even key)"));
        }
        let wrapped_even_key = data[18..18 + even_len].to_vec();

        let pos = 18 + even_len;
        let odd_len = (u16::from(data[pos]) << 8 | u16::from(data[pos + 1])) as usize;
        if data.len() < pos + 2 + odd_len {
            return Err(NetError::parse(0, "KeyMaterialPacket truncated (odd key)"));
        }
        let wrapped_odd_key = data[pos + 2..pos + 2 + odd_len].to_vec();

        Ok(Self {
            version,
            key_size,
            salt,
            wrapped_even_key,
            wrapped_odd_key,
        })
    }
}

/// Password-based authentication for SRT connections.
///
/// Derives a Key Encryption Key (KEK) from a passphrase, which is then used
/// to wrap/unwrap the actual session keys in the key material exchange.
#[derive(Debug, Clone)]
pub struct PassphraseAuth {
    /// Derived Key Encryption Key.
    kek: Vec<u8>,
    /// Salt used for KEK derivation.
    kek_salt: [u8; 16],
    /// Key size (16, 24, or 32 bytes).
    key_size: usize,
}

impl PassphraseAuth {
    /// Creates a new `PassphraseAuth` from a passphrase.
    ///
    /// Derives the KEK using a PBKDF2-like scheme.
    ///
    /// # Errors
    ///
    /// Returns an error if `key_size` is not 16, 24, or 32.
    pub fn new(passphrase: &str, key_size: usize) -> NetResult<Self> {
        if ![16, 24, 32].contains(&key_size) {
            return Err(NetError::protocol("Invalid key size for PassphraseAuth"));
        }

        let kek_salt = generate_salt();
        let kek = derive_session_key(passphrase.as_bytes(), &kek_salt, key_size, KDF_ITERATIONS)?;

        Ok(Self {
            kek,
            kek_salt,
            key_size,
        })
    }

    /// Creates a `PassphraseAuth` with a known salt (for reproducing KEK on the peer).
    ///
    /// # Errors
    ///
    /// Returns an error if `key_size` is not 16, 24, or 32.
    pub fn with_salt(passphrase: &str, key_size: usize, salt: [u8; 16]) -> NetResult<Self> {
        if ![16, 24, 32].contains(&key_size) {
            return Err(NetError::protocol("Invalid key size for PassphraseAuth"));
        }

        let kek = derive_session_key(passphrase.as_bytes(), &salt, key_size, KDF_ITERATIONS)?;

        Ok(Self {
            kek,
            kek_salt: salt,
            key_size,
        })
    }

    /// Returns the derived KEK.
    #[must_use]
    pub fn kek(&self) -> &[u8] {
        &self.kek
    }

    /// Returns the KEK salt.
    #[must_use]
    pub const fn kek_salt(&self) -> &[u8; 16] {
        &self.kek_salt
    }

    /// Returns the key size.
    #[must_use]
    pub const fn key_size(&self) -> usize {
        self.key_size
    }

    /// Authenticates by verifying that two `PassphraseAuth` instances with the
    /// same passphrase produce the same KEK when given the same salt.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.kek == other.kek
    }

    /// Wraps session keys using this KEK, returning a `KeyMaterialPacket`.
    ///
    /// # Errors
    ///
    /// Returns an error if key sizes mismatch.
    pub fn wrap_keys(&self, even_key: &[u8], odd_key: &[u8]) -> NetResult<KeyMaterialPacket> {
        if even_key.len() != self.key_size || odd_key.len() != self.key_size {
            return Err(NetError::protocol("Key size mismatch in wrap_keys"));
        }

        let wrapped_even = aes_key_wrap(&self.kek, even_key)?;
        let wrapped_odd = aes_key_wrap(&self.kek, odd_key)?;
        let mut salt = [0u8; 14];
        salt.copy_from_slice(&self.kek_salt[..14]);

        Ok(KeyMaterialPacket {
            version: KM_VERSION,
            key_size: self.key_size as u8,
            salt,
            wrapped_even_key: wrapped_even,
            wrapped_odd_key: wrapped_odd,
        })
    }

    /// Unwraps session keys from a `KeyMaterialPacket` using this KEK.
    ///
    /// # Errors
    ///
    /// Returns an error if unwrapping fails.
    pub fn unwrap_keys(&self, km: &KeyMaterialPacket) -> NetResult<(Vec<u8>, Vec<u8>)> {
        let even = aes_key_unwrap(&self.kek, &km.wrapped_even_key)?;
        let odd = aes_key_unwrap(&self.kek, &km.wrapped_odd_key)?;
        Ok((even, odd))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes_context_from_passphrase() {
        let ctx = AesContext::from_passphrase("test_password", 16).expect("should succeed in test");
        assert_eq!(ctx.key_size(), 16);
    }

    #[test]
    fn test_aes_context_invalid_key_size() {
        let result = AesContext::from_passphrase("test", 15);
        assert!(result.is_err());
    }

    #[test]
    fn test_encrypt_decrypt() {
        let ctx = AesContext::from_passphrase("password", 16).expect("should succeed in test");
        let plaintext = b"Hello, SRT!";
        let iv = [1u8; 16];

        let ciphertext = ctx.encrypt(plaintext, &iv).expect("should succeed in test");
        assert_ne!(ciphertext.as_ref(), plaintext);

        let decrypted = ctx
            .decrypt(&ciphertext, &iv)
            .expect("should succeed in test");
        assert_eq!(decrypted.as_ref(), plaintext);
    }

    #[test]
    fn test_key_rotation() {
        let mut ctx = AesContext::from_passphrase("test", 16).expect("should succeed in test");
        assert_eq!(ctx.key_index(), 0);
        ctx.rotate_key();
        assert_eq!(ctx.key_index(), 1);
    }

    #[test]
    fn test_key_material() {
        let km = KeyMaterial::new(16);
        let encoded = km.encode();
        let decoded = KeyMaterial::decode(&encoded).expect("should succeed in test");
        assert_eq!(decoded.key_len, 16);
    }

    #[test]
    fn test_derive_key() {
        let key1 = derive_key(b"password", b"salt", 16).expect("should succeed in test");
        let key2 = derive_key(b"password", b"salt", 16).expect("should succeed in test");
        assert_eq!(key1, key2);

        let key3 = derive_key(b"different", b"salt", 16).expect("should succeed in test");
        assert_ne!(key1, key3);
    }

    // --- derive_session_key tests ---

    #[test]
    fn test_derive_session_key_deterministic() {
        let key1 =
            derive_session_key(b"mysecret", b"somesalt", 16, 10).expect("should succeed in test");
        let key2 =
            derive_session_key(b"mysecret", b"somesalt", 16, 10).expect("should succeed in test");
        assert_eq!(key1, key2);
        assert_eq!(key1.len(), 16);
    }

    #[test]
    fn test_derive_session_key_different_passphrase() {
        let key1 = derive_session_key(b"passA", b"salt", 16, 10).expect("should succeed in test");
        let key2 = derive_session_key(b"passB", b"salt", 16, 10).expect("should succeed in test");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_derive_session_key_different_salt() {
        let key1 = derive_session_key(b"pass", b"saltA", 16, 10).expect("should succeed in test");
        let key2 = derive_session_key(b"pass", b"saltB", 16, 10).expect("should succeed in test");
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_derive_session_key_32_bytes() {
        let key = derive_session_key(b"pass", b"salt", 32, 10).expect("should succeed in test");
        assert_eq!(key.len(), 32);
    }

    // --- Real-crypto regression tests (NIST known-answer + round-trip) ---

    /// NIST SP 800-38A §F.5.1 AES-128-CTR known-answer test vector (block 1).
    ///
    /// This MUST fail if `AesContext::encrypt` is a homebrew XOR/mixing
    /// "cipher" (as it previously was) instead of real AES-CTR: the fake
    /// cipher cannot reproduce a NIST reference ciphertext.
    #[test]
    fn test_aes_ctr_nist_known_answer_vector() {
        let key = hex_decode("2b7e151628aed2a6abf7158809cf4f3c");
        let iv: [u8; 16] = hex_decode("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
            .try_into()
            .expect("16-byte IV");
        let plaintext = hex_decode("6bc1bee22e409f96e93d7e117393172a");
        let expected_ciphertext = hex_decode("874d6191b620e3261bef6864990db6ce");

        let ctx = AesContext::from_key(&key).expect("valid 16-byte key");
        let ciphertext = ctx
            .encrypt(&plaintext, &iv)
            .expect("should succeed in test");

        assert_eq!(
            ciphertext.as_ref(),
            expected_ciphertext.as_slice(),
            "AES-128-CTR output must match the NIST SP 800-38A known-answer vector"
        );
    }

    /// Minimal hex decoder for the known-answer test vector above (avoids
    /// pulling in a `hex`/`hex-literal` dependency for a single test).
    fn hex_decode(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap_or(0))
            .collect()
    }

    /// Round-trip encrypt -> decrypt using the same real AES-CTR path,
    /// across all three supported key sizes.
    #[test]
    fn test_aes_ctr_round_trip_all_key_sizes() {
        for key_size in [16usize, 24, 32] {
            let key = vec![0x5Au8; key_size];
            let ctx = AesContext::from_key(&key).expect("should succeed in test");
            let iv = [0x11u8; 16];
            let plaintext = b"OxiMedia SRT AES-CTR round-trip test payload, multi-block!";

            let ciphertext = ctx.encrypt(plaintext, &iv).expect("should succeed in test");
            assert_ne!(ciphertext.as_ref(), plaintext.as_slice());

            let decrypted = ctx
                .decrypt(&ciphertext, &iv)
                .expect("should succeed in test");
            assert_eq!(decrypted.as_ref(), plaintext.as_slice());
        }
    }

    /// Two different keys must not produce the same ciphertext for the same
    /// plaintext/IV (sanity check that the key material actually feeds into
    /// the cipher, unlike a cipher that silently ignores it).
    #[test]
    fn test_aes_ctr_different_keys_different_ciphertext() {
        let iv = [0x22u8; 16];
        let plaintext = b"same plaintext, different keys";

        let ctx_a = AesContext::from_key(&[0xAAu8; 16]).expect("should succeed in test");
        let ctx_b = AesContext::from_key(&[0xBBu8; 16]).expect("should succeed in test");

        let ct_a = ctx_a
            .encrypt(plaintext, &iv)
            .expect("should succeed in test");
        let ct_b = ctx_b
            .encrypt(plaintext, &iv)
            .expect("should succeed in test");

        assert_ne!(ct_a.as_ref(), ct_b.as_ref());
    }

    // --- KeySchedule tests ---

    #[test]
    fn test_key_schedule_from_passphrase() {
        let salt = [0x01u8; 14];
        let ks =
            KeySchedule::from_passphrase("testpass", &salt, 16).expect("should succeed in test");
        assert_eq!(ks.key_size(), 16);
        assert_eq!(ks.active_index(), 0);
        // Even and odd keys should be different
        assert_ne!(ks.even_key(), ks.odd_key());
    }

    #[test]
    fn test_key_schedule_rotate() {
        let salt = [0x02u8; 14];
        let mut ks =
            KeySchedule::from_passphrase("testpass", &salt, 16).expect("should succeed in test");
        let initial_active = ks.active_key().to_vec();
        ks.rotate();
        let after_rotate = ks.active_key().to_vec();
        assert_ne!(initial_active, after_rotate);
        ks.rotate();
        assert_eq!(ks.active_key(), initial_active.as_slice());
    }

    #[test]
    fn test_key_schedule_invalid_key_size() {
        let salt = [0u8; 14];
        let result = KeySchedule::from_passphrase("pass", &salt, 15);
        assert!(result.is_err());
    }

    // --- SrtCryptoContext tests ---

    #[test]
    fn test_srt_crypto_context_creation() {
        let ctx = SrtCryptoContext::from_passphrase("secret", 16).expect("should succeed in test");
        assert_eq!(ctx.packet_count(), 0);
        assert_eq!(ctx.key_schedule().key_size(), 16);
    }

    #[test]
    fn test_srt_crypto_context_invalid_key_size() {
        let result = SrtCryptoContext::from_passphrase("secret", 20);
        assert!(result.is_err());
    }

    #[test]
    fn test_srt_crypto_context_encrypt_decrypt() {
        let mut ctx =
            SrtCryptoContext::from_passphrase("secret", 16).expect("should succeed in test");
        let plaintext = b"Hello, SRT world!".to_vec();

        let mut packet = SrtPacketBuffer::new(1, 0, plaintext.clone());
        ctx.encrypt_packet(&mut packet)
            .expect("should succeed in test");

        // After encryption: payload changed, flag set
        assert_ne!(packet.payload, plaintext);
        assert_ne!(packet.encryption_flag, 0);
        assert_eq!(ctx.packet_count(), 1);

        // Decrypt
        ctx.decrypt_packet(&mut packet)
            .expect("should succeed in test");
        assert_eq!(packet.payload, plaintext);
        assert_eq!(packet.encryption_flag, 0);
    }

    #[test]
    fn test_srt_crypto_context_encrypt_empty_error() {
        let mut ctx =
            SrtCryptoContext::from_passphrase("secret", 16).expect("should succeed in test");
        let mut packet = SrtPacketBuffer::new(1, 0, vec![]);
        let result = ctx.encrypt_packet(&mut packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_srt_crypto_context_decrypt_clear_error() {
        let ctx = SrtCryptoContext::from_passphrase("secret", 16).expect("should succeed in test");
        let mut packet = SrtPacketBuffer::new(1, 0, b"data".to_vec());
        let result = ctx.decrypt_packet(&mut packet);
        assert!(result.is_err());
    }

    #[test]
    fn test_srt_crypto_context_key_rotation() {
        let mut ctx =
            SrtCryptoContext::from_passphrase("secret", 16).expect("should succeed in test");
        let key_before = ctx.key_schedule().active_index();
        ctx.rotate_key().expect("should succeed in test");
        let key_after = ctx.key_schedule().active_index();
        assert_ne!(key_before, key_after);
    }

    #[test]
    fn test_srt_crypto_context_key_material_exchange() {
        let kek = vec![0xABu8; 16];
        let ctx = SrtCryptoContext::from_passphrase("secret", 16).expect("should succeed in test");

        let km = ctx
            .build_key_material(&kek)
            .expect("should succeed in test");
        let encoded = km.encode();
        let decoded = KeyMaterialPacket::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.key_size, 16);
    }

    #[test]
    fn test_srt_crypto_context_load_key_material() {
        let kek = vec![0xCDu8; 16];
        let ctx1 = SrtCryptoContext::from_passphrase("secret", 16).expect("should succeed in test");
        let km = ctx1
            .build_key_material(&kek)
            .expect("should succeed in test");

        let even_key = vec![0xAAu8; 16];
        let odd_key = vec![0xBBu8; 16];
        let mut salt = [0u8; 14];
        salt[0] = 0x11;
        let mut ctx2 =
            SrtCryptoContext::from_keys(even_key, odd_key, salt).expect("should succeed in test");
        ctx2.load_key_material(&km, &kek)
            .expect("should succeed in test");

        // After loading, the even/odd keys should match ctx1's
        assert_eq!(
            ctx2.key_schedule().even_key(),
            ctx1.key_schedule().even_key()
        );
        assert_eq!(ctx2.key_schedule().odd_key(), ctx1.key_schedule().odd_key());
    }

    // --- PassphraseAuth tests ---

    #[test]
    fn test_passphrase_auth_creation() {
        let auth = PassphraseAuth::new("password123", 16).expect("should succeed in test");
        assert_eq!(auth.key_size(), 16);
        assert_eq!(auth.kek().len(), 16);
    }

    #[test]
    fn test_passphrase_auth_invalid_key_size() {
        let result = PassphraseAuth::new("pass", 20);
        assert!(result.is_err());
    }

    #[test]
    fn test_passphrase_auth_same_salt_produces_same_kek() {
        let salt = [0x42u8; 16];
        let auth1 =
            PassphraseAuth::with_salt("password", 16, salt).expect("should succeed in test");
        let auth2 =
            PassphraseAuth::with_salt("password", 16, salt).expect("should succeed in test");
        assert!(auth1.matches(&auth2));
    }

    #[test]
    fn test_passphrase_auth_different_passwords_different_kek() {
        let salt = [0x42u8; 16];
        let auth1 =
            PassphraseAuth::with_salt("passwordA", 16, salt).expect("should succeed in test");
        let auth2 =
            PassphraseAuth::with_salt("passwordB", 16, salt).expect("should succeed in test");
        assert!(!auth1.matches(&auth2));
    }

    #[test]
    fn test_passphrase_auth_wrap_unwrap_keys() {
        let salt = [0x11u8; 16];
        let auth = PassphraseAuth::with_salt("secret", 16, salt).expect("should succeed in test");
        let even_key = vec![0xAAu8; 16];
        let odd_key = vec![0xBBu8; 16];

        let km = auth
            .wrap_keys(&even_key, &odd_key)
            .expect("should succeed in test");
        let (unwrapped_even, unwrapped_odd) =
            auth.unwrap_keys(&km).expect("should succeed in test");

        assert_eq!(unwrapped_even, even_key);
        assert_eq!(unwrapped_odd, odd_key);
    }

    #[test]
    fn test_passphrase_auth_key_size_24() {
        let auth = PassphraseAuth::new("testpass", 24).expect("should succeed in test");
        assert_eq!(auth.kek().len(), 24);
    }

    #[test]
    fn test_passphrase_auth_key_size_32() {
        let auth = PassphraseAuth::new("testpass", 32).expect("should succeed in test");
        assert_eq!(auth.kek().len(), 32);
    }

    // --- KeyMaterialPacket tests ---

    #[test]
    fn test_km_packet_encode_decode() {
        let km = KeyMaterialPacket {
            version: 1,
            key_size: 16,
            salt: [0xABu8; 14],
            wrapped_even_key: vec![0x11u8; 24], // 16-byte key + 8-byte ICV
            wrapped_odd_key: vec![0x22u8; 24],
        };

        let encoded = km.encode();
        let decoded = KeyMaterialPacket::decode(&encoded).expect("should succeed in test");

        assert_eq!(decoded.version, 1);
        assert_eq!(decoded.key_size, 16);
        assert_eq!(decoded.salt, [0xABu8; 14]);
        assert_eq!(decoded.wrapped_even_key, km.wrapped_even_key);
        assert_eq!(decoded.wrapped_odd_key, km.wrapped_odd_key);
    }

    #[test]
    fn test_km_packet_decode_too_short() {
        let result = KeyMaterialPacket::decode(&[0u8; 5]);
        assert!(result.is_err());
    }

    // --- aes_key_wrap / unwrap tests (RFC 3394) ---

    #[test]
    fn test_key_wrap_unwrap_roundtrip() {
        let kek = vec![0x55u8; 16];
        let key = vec![0xAAu8; 16];
        let wrapped = aes_key_wrap(&kek, &key).expect("wrap should succeed");
        let unwrapped = aes_key_unwrap(&kek, &wrapped).expect("should succeed in test");
        assert_eq!(unwrapped, key);
    }

    #[test]
    fn test_key_unwrap_bad_icv() {
        let kek = vec![0x55u8; 16];
        let bad = vec![0x00u8; 24]; // garbage ciphertext: integrity check must fail
        let result = aes_key_unwrap(&kek, &bad);
        assert!(result.is_err());
    }

    #[test]
    fn rfc3394_128bit_kek_128bit_key_vector() {
        // RFC 3394 §4.1: wrap 128-bit key data with a 128-bit KEK.
        let kek: [u8; 16] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
            0x0E, 0x0F,
        ];
        let key: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD,
            0xEE, 0xFF,
        ];
        let expected: [u8; 24] = [
            0x1F, 0xA6, 0x8B, 0x0A, 0x81, 0x12, 0xB4, 0x47, 0xAE, 0xF3, 0x4B, 0xD8, 0xFB, 0x5A,
            0x7B, 0x82, 0x9D, 0x3E, 0x86, 0x23, 0x71, 0xD2, 0xCF, 0xE5,
        ];

        // wrap must match the published ciphertext exactly (interop proof).
        let wrapped = aes_key_wrap(&kek, &key).expect("wrap should succeed");
        assert_eq!(
            wrapped.as_slice(),
            &expected[..],
            "RFC 3394 §4.1 ciphertext"
        );

        // unwrap of the published ciphertext must recover the original key.
        let unwrapped = aes_key_unwrap(&kek, &expected).expect("unwrap should succeed");
        assert_eq!(unwrapped.as_slice(), &key[..]);
    }

    #[test]
    fn rfc3394_wrong_kek_fails_integrity_check() {
        // Unwrapping the RFC 3394 vector with the wrong KEK must fail (the A
        // register will not equal the 0xA6.. IV).
        let key: [u8; 16] = [0x11; 16];
        let good_kek: [u8; 16] = [0x22; 16];
        let wrapped = aes_key_wrap(&good_kek, &key).expect("wrap should succeed");
        let wrong_kek: [u8; 16] = [0x33; 16];
        assert!(aes_key_unwrap(&wrong_kek, &wrapped).is_err());
    }

    #[test]
    fn rfc3394_192_and_256_bit_kek_roundtrip() {
        // Exercise AES-192 and AES-256 KEKs (24/32-byte) over 24/32-byte keys.
        for kek_len in [24usize, 32] {
            let kek = vec![0x5Au8; kek_len];
            let key = vec![0xA5u8; kek_len];
            let wrapped = aes_key_wrap(&kek, &key).expect("wrap should succeed");
            assert_eq!(wrapped.len(), key.len() + 8);
            let unwrapped = aes_key_unwrap(&kek, &wrapped).expect("unwrap should succeed");
            assert_eq!(unwrapped, key);
        }
    }
}
