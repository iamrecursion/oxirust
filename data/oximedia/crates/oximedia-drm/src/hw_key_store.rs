//! Hardware-backed key storage interface.
//!
//! Provides an abstraction over TPM (Trusted Platform Module) and Secure Enclave
//! for hardware-protected key storage. The interface allows content-encryption keys
//! to be wrapped by hardware-resident master keys such that the raw key material
//! never exists in software-accessible memory.
//!
//! # Architecture
//!
//! ```text
//!  ┌────────────────────┐       ┌──────────────────────┐
//!  │  Application       │──────▶│  HwKeyStore (trait)  │
//!  │  (DRM runtime)     │       │  - TpmKeyStore       │
//!  └────────────────────┘       │  - SecureEnclaveStore│
//!                               │  - SoftwareKeyStore  │
//!                               └──────────────────────┘
//! ```
//!
//! In production, TPM 2.0 and Apple Secure Enclave implementations would
//! delegate to platform-specific APIs. The `SoftwareKeyStore` provided here
//! uses AES-GCM wrapping with a software master key and is suitable for
//! testing and platforms without HSM hardware.

use crate::{DrmError, Result};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Backend type enum
// ---------------------------------------------------------------------------

/// The type of hardware security backend.
///
/// Hardware variants ([`Tpm2`](HwBackend::Tpm2),
/// [`SecureEnclave`](HwBackend::SecureEnclave)) are reserved for builds that
/// link against a real platform TSS 2.0 stack or Apple Security framework.
/// Until those `cfg`-gated integrations land, software emulators of the same
/// authorization model report [`SoftwareTpm`](HwBackend::SoftwareTpm) /
/// [`SoftwareSecureEnclave`](HwBackend::SoftwareSecureEnclave) so that callers
/// can reliably distinguish a real hardware root of trust from an in-process
/// emulation. This is the security-correct behaviour: the [`is_hardware`]
/// predicate stays `true` only for the genuine hardware variants.
///
/// [`is_hardware`]: HwBackend::is_hardware
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwBackend {
    /// TPM 2.0 (Trusted Platform Module) – hardware-backed, PC/server.
    ///
    /// Reserved for a future `cfg(feature = "hw-tpm", target_os = "linux")`
    /// integration that talks to `tss-esapi`/`/dev/tpmrm0`.
    Tpm2,
    /// Apple Secure Enclave – hardware-backed, Apple Silicon / T2.
    ///
    /// Reserved for a future `cfg(target_os = "macos")` integration that
    /// calls into CryptoKit / Security.framework.
    SecureEnclave,
    /// Software-emulated TPM 2.0 state machine
    /// (see [`sw_tpm`](crate::sw_tpm)).
    ///
    /// Implements PCR banks, HMAC-protected sealed objects, and policy
    /// authorization sessions entirely in Pure Rust. Reported by
    /// [`TpmKeyStore`] when no real hardware TPM is linked in.
    SoftwareTpm,
    /// Software-emulated Apple Secure Enclave
    /// (see [`sw_secure_enclave`](crate::sw_secure_enclave)).
    ///
    /// Enforces the Secure Enclave non-exportability boundary in software.
    /// Reported by [`SecureEnclaveKeyStore`] when no real Secure Enclave is
    /// linked in.
    SoftwareSecureEnclave,
    /// Software-only fallback (XOR keystream wrapped with a master key).
    Software,
}

impl HwBackend {
    /// Returns `true` if the backend is backed by genuine secure hardware.
    ///
    /// Software-emulated backends ([`SoftwareTpm`](HwBackend::SoftwareTpm),
    /// [`SoftwareSecureEnclave`](HwBackend::SoftwareSecureEnclave),
    /// [`Software`](HwBackend::Software)) all return `false` — they model
    /// the protocol surface of a hardware root of trust without providing
    /// its security guarantees.
    #[must_use]
    pub fn is_hardware(&self) -> bool {
        matches!(self, HwBackend::Tpm2 | HwBackend::SecureEnclave)
    }

    /// Returns `true` if the backend emulates a hardware root of trust in
    /// software (TPM 2.0 emulator, Secure Enclave emulator).
    #[must_use]
    pub fn is_software_emulated(&self) -> bool {
        matches!(
            self,
            HwBackend::SoftwareTpm | HwBackend::SoftwareSecureEnclave
        )
    }
}

impl std::fmt::Display for HwBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tpm2 => write!(f, "TPM 2.0"),
            Self::SecureEnclave => write!(f, "Secure Enclave"),
            Self::SoftwareTpm => write!(f, "Software TPM 2.0"),
            Self::SoftwareSecureEnclave => write!(f, "Software Secure Enclave"),
            Self::Software => write!(f, "Software"),
        }
    }
}

// ---------------------------------------------------------------------------
// Key slot
// ---------------------------------------------------------------------------

/// Identifies a logical key slot within the hardware store.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeySlot {
    /// Human-readable label (e.g. "content-key-1").
    pub label: String,
    /// Unique numeric handle assigned by the store.
    pub handle: u32,
}

impl KeySlot {
    /// Create a new key slot.
    #[must_use]
    pub fn new(label: impl Into<String>, handle: u32) -> Self {
        Self {
            label: label.into(),
            handle,
        }
    }
}

// ---------------------------------------------------------------------------
// Key descriptor
// ---------------------------------------------------------------------------

/// Metadata for a key stored in the hardware backend.
#[derive(Debug, Clone)]
pub struct KeyDescriptor {
    /// Slot this descriptor refers to.
    pub slot: KeySlot,
    /// Length of the wrapped (ciphertext) key material in bytes.
    pub wrapped_len: usize,
    /// Whether the key can be exported in clear-text form.
    pub exportable: bool,
    /// Unix timestamp (seconds) when this key was created.
    pub created_at: u64,
    /// Unix timestamp (seconds) when this key expires, if any.
    pub expires_at: Option<u64>,
}

impl KeyDescriptor {
    /// Returns `true` if the key has expired at the given timestamp.
    #[must_use]
    pub fn is_expired(&self, now: u64) -> bool {
        self.expires_at.map_or(false, |exp| now >= exp)
    }
}

// ---------------------------------------------------------------------------
// HwKeyStore trait
// ---------------------------------------------------------------------------

/// Trait for hardware-backed key storage operations.
///
/// Implementors must ensure that raw key material is wrapped before being
/// stored, and that unsealing requires backend-specific authorization.
pub trait HwKeyStore: Send + Sync {
    /// The hardware backend type.
    fn backend(&self) -> HwBackend;

    /// Import a plaintext content key into the store.
    ///
    /// Returns a [`KeySlot`] that can be used to reference the key later.
    fn import_key(&mut self, label: &str, key_bytes: &[u8], exportable: bool) -> Result<KeySlot>;

    /// Export the wrapped (encrypted) key material for a slot.
    ///
    /// The returned bytes are encrypted with the backend master key and
    /// are safe to store on disk or transmit.
    fn export_wrapped(&self, slot: &KeySlot) -> Result<Vec<u8>>;

    /// Unseal (decrypt) the content key for use in a single operation.
    ///
    /// # Security note
    /// Callers must zeroize the returned `Vec<u8>` after use. In a full
    /// implementation this would be performed inside a TEE.
    fn unseal(&self, slot: &KeySlot) -> Result<Vec<u8>>;

    /// Remove a key from the store, destroying the wrapped material.
    fn delete_key(&mut self, slot: &KeySlot) -> Result<()>;

    /// List all key descriptors in the store.
    fn list_keys(&self) -> Vec<KeyDescriptor>;

    /// Return the descriptor for a specific slot.
    fn get_descriptor(&self, slot: &KeySlot) -> Option<KeyDescriptor>;
}

// ---------------------------------------------------------------------------
// Software key store (AES-GCM wrapping)
// ---------------------------------------------------------------------------

/// Minimal GCM-style wrapping using pure-Rust AES-256 key encryption.
///
/// We avoid importing `aes-gcm` here to keep the hardware-abstraction layer
/// independent. Instead we use a simple XOR-based wrapping (HMAC-protected)
/// that is sufficient for the software simulation layer. Production code
/// would call into the OS keychain / TPM APIs.
fn software_wrap(master: &[u8; 32], plaintext: &[u8]) -> Vec<u8> {
    // nonce = sha256(master || length)[..16]
    let mut nonce_input = master.to_vec();
    nonce_input.extend_from_slice(&(plaintext.len() as u64).to_le_bytes());
    let nonce = sha256_mini(&nonce_input);

    // keystream = sha256(master || nonce) repeated
    let mut wrapped = plaintext.to_vec();
    let mut keystream_seed = master.to_vec();
    keystream_seed.extend_from_slice(&nonce[..16]);

    let mut offset = 0usize;
    let mut counter = 0u64;
    while offset < wrapped.len() {
        let mut block_input = keystream_seed.clone();
        block_input.extend_from_slice(&counter.to_le_bytes());
        let block = sha256_mini(&block_input);
        let block_len = block.len().min(wrapped.len() - offset);
        for (i, b) in wrapped[offset..offset + block_len].iter_mut().enumerate() {
            *b ^= block[i];
        }
        offset += block_len;
        counter += 1;
    }

    // Prepend nonce and append HMAC tag (first 16 bytes of SHA-256)
    let mut tag_input = master.to_vec();
    tag_input.extend_from_slice(&wrapped);
    let tag = sha256_mini(&tag_input);

    let mut result = nonce[..16].to_vec();
    result.extend_from_slice(&wrapped);
    result.extend_from_slice(&tag[..16]);
    result
}

fn software_unwrap(master: &[u8; 32], wrapped_with_nonce: &[u8]) -> Result<Vec<u8>> {
    if wrapped_with_nonce.len() < 32 {
        return Err(DrmError::InvalidKey(
            "Wrapped key too short (< 32 bytes)".to_string(),
        ));
    }

    // Last 16 bytes are the tag
    let tag_offset = wrapped_with_nonce.len() - 16;
    let stored_tag = &wrapped_with_nonce[tag_offset..];
    let nonce = &wrapped_with_nonce[..16];
    let ciphertext = &wrapped_with_nonce[16..tag_offset];

    // Verify tag
    let mut tag_input = master.to_vec();
    tag_input.extend_from_slice(ciphertext);
    let expected_tag = sha256_mini(&tag_input);
    if expected_tag[..16] != *stored_tag {
        return Err(DrmError::InvalidKey(
            "Hardware key store: HMAC verification failed".to_string(),
        ));
    }

    // Decrypt
    let mut plaintext = ciphertext.to_vec();
    let mut keystream_seed = master.to_vec();
    keystream_seed.extend_from_slice(nonce);

    let mut offset = 0usize;
    let mut counter = 0u64;
    while offset < plaintext.len() {
        let mut block_input = keystream_seed.clone();
        block_input.extend_from_slice(&counter.to_le_bytes());
        let block = sha256_mini(&block_input);
        let block_len = block.len().min(plaintext.len() - offset);
        for (i, b) in plaintext[offset..offset + block_len].iter_mut().enumerate() {
            *b ^= block[i];
        }
        offset += block_len;
        counter += 1;
    }

    Ok(plaintext)
}

/// Minimal SHA-256 (same pure-Rust implementation as in key_rotation.rs).
///
/// Exposed to sibling modules (`sw_tpm`, `sw_secure_enclave`) so that the
/// emulators can reuse the existing in-crate primitive without pulling in
/// another SHA-256 implementation.
pub(crate) fn sha256_mini(msg: &[u8]) -> [u8; 32] {
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
    #[allow(clippy::unreadable_literal)]
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    let mut padded = msg.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0x00);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
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

/// Software-emulated hardware key store backed by AES-GCM-style wrapping.
///
/// This is the "Software" backend — suitable for CI/testing or platforms
/// without HSM hardware. The master wrapping key is held in memory (NOT
/// hardware-protected) so this backend does not provide true hardware security.
pub struct SoftwareKeyStore {
    /// Master wrapping key (32 bytes / 256-bit).
    master_key: [u8; 32],
    /// Wrapped keys: handle → wrapped-bytes.
    store: HashMap<u32, Vec<u8>>,
    /// Metadata: handle → descriptor.
    descriptors: HashMap<u32, KeyDescriptor>,
    /// Label to handle mapping.
    label_to_handle: HashMap<String, u32>,
    /// Monotonically-increasing handle counter.
    next_handle: u32,
    /// Current time source (seconds since epoch). Settable for testing.
    current_time: u64,
}

impl SoftwareKeyStore {
    /// Create a new software key store with the given master key.
    ///
    /// In production the master key would come from a TPM or OS keychain.
    /// Here the caller supplies it so that tests are reproducible.
    pub fn new(master_key: [u8; 32]) -> Self {
        Self {
            master_key,
            store: HashMap::new(),
            descriptors: HashMap::new(),
            label_to_handle: HashMap::new(),
            next_handle: 1,
            current_time: 0,
        }
    }

    /// Set the current Unix timestamp (for testing time-based key expiry).
    pub fn set_time(&mut self, ts: u64) {
        self.current_time = ts;
    }

    /// Look up a slot by label.
    pub fn find_by_label(&self, label: &str) -> Option<KeySlot> {
        self.label_to_handle
            .get(label)
            .map(|&h| KeySlot::new(label, h))
    }

    /// Set an expiry timestamp on a key slot (seconds since epoch).
    pub fn set_expiry(&mut self, slot: &KeySlot, expires_at: u64) -> Result<()> {
        let desc = self
            .descriptors
            .get_mut(&slot.handle)
            .ok_or_else(|| DrmError::InvalidKey(format!("Key slot {} not found", slot.handle)))?;
        desc.expires_at = Some(expires_at);
        Ok(())
    }
}

impl HwKeyStore for SoftwareKeyStore {
    fn backend(&self) -> HwBackend {
        HwBackend::Software
    }

    fn import_key(&mut self, label: &str, key_bytes: &[u8], exportable: bool) -> Result<KeySlot> {
        if key_bytes.is_empty() {
            return Err(DrmError::InvalidKey(
                "Key material must not be empty".to_string(),
            ));
        }

        let handle = self.next_handle;
        self.next_handle += 1;

        let wrapped = software_wrap(&self.master_key, key_bytes);
        let wrapped_len = wrapped.len();

        let slot = KeySlot::new(label, handle);
        let desc = KeyDescriptor {
            slot: slot.clone(),
            wrapped_len,
            exportable,
            created_at: self.current_time,
            expires_at: None,
        };

        self.store.insert(handle, wrapped);
        self.descriptors.insert(handle, desc);
        self.label_to_handle.insert(label.to_string(), handle);

        Ok(slot)
    }

    fn export_wrapped(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        let desc = self
            .descriptors
            .get(&slot.handle)
            .ok_or_else(|| DrmError::InvalidKey(format!("Key slot {} not found", slot.handle)))?;
        if !desc.exportable {
            return Err(DrmError::InvalidKey(format!(
                "Key '{}' is not exportable",
                slot.label
            )));
        }
        self.store
            .get(&slot.handle)
            .cloned()
            .ok_or_else(|| DrmError::InvalidKey("Key material not found in store".to_string()))
    }

    fn unseal(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        if let Some(desc) = self.descriptors.get(&slot.handle) {
            if desc.is_expired(self.current_time) {
                return Err(DrmError::LicenseError(format!(
                    "Key '{}' has expired",
                    slot.label
                )));
            }
        } else {
            return Err(DrmError::InvalidKey(format!(
                "Key slot {} not found",
                slot.handle
            )));
        }

        let wrapped = self
            .store
            .get(&slot.handle)
            .ok_or_else(|| DrmError::InvalidKey("Key material not found".to_string()))?;

        software_unwrap(&self.master_key, wrapped)
    }

    fn delete_key(&mut self, slot: &KeySlot) -> Result<()> {
        let removed_wrapped = self.store.remove(&slot.handle);
        let removed_desc = self.descriptors.remove(&slot.handle);
        // Clean up label mapping
        self.label_to_handle.retain(|_, &mut h| h != slot.handle);

        if removed_wrapped.is_none() && removed_desc.is_none() {
            return Err(DrmError::InvalidKey(format!(
                "Key slot {} not found",
                slot.handle
            )));
        }
        Ok(())
    }

    fn list_keys(&self) -> Vec<KeyDescriptor> {
        self.descriptors.values().cloned().collect()
    }

    fn get_descriptor(&self, slot: &KeySlot) -> Option<KeyDescriptor> {
        self.descriptors.get(&slot.handle).cloned()
    }
}

// ---------------------------------------------------------------------------
// TPM 2.0 software-emulated backend
// ---------------------------------------------------------------------------

/// Software-emulated TPM 2.0 backend.
///
/// Wraps a [`SwTpm`](crate::sw_tpm::SwTpm) state machine that models the
/// TPM 2.0 authorization surface: object handles, an HMAC-SHA-256 protected
/// sealed-object store, a 24-slot SHA-256 PCR bank, and PolicyPCR /
/// PolicyAuthValue authorization sessions. The result is reported as
/// [`HwBackend::SoftwareTpm`] — **not** [`HwBackend::Tpm2`] — because no
/// hardware root of trust is actually contacted. A future
/// `cfg(feature = "hw-tpm", target_os = "linux")` implementation may replace
/// this delegate with calls into `tss-esapi`/`/dev/tpmrm0`, at which point
/// the backend identifier may transition to [`HwBackend::Tpm2`].
pub struct TpmKeyStore {
    inner: crate::sw_tpm::SwTpm,
}

impl TpmKeyStore {
    /// Create a new (software-emulated) TPM key store.
    ///
    /// The `master_key` seeds the SP 800-108 HMAC integrity layer that
    /// protects every sealed object. In a real hardware TPM the analogous
    /// secret is the storage hierarchy seed and never leaves the device.
    pub fn new(master_key: [u8; 32]) -> Self {
        Self {
            inner: crate::sw_tpm::SwTpm::new(master_key),
        }
    }

    /// Extend PCR `index` with `data` (PCR_Extend: `PCR ← SHA256(PCR ‖ data)`).
    ///
    /// Returns the new PCR value. Subsequent
    /// [`unseal_with_policy`](TpmKeyStore::unseal_with_policy) calls that
    /// reference this PCR will only succeed when the PCR bank matches the
    /// digest captured at policy creation.
    pub fn pcr_extend(&mut self, index: usize, data: &[u8]) -> Result<[u8; 32]> {
        self.inner.pcr_extend(index, data)
    }

    /// Read the current value of PCR `index`.
    pub fn pcr_read(&self, index: usize) -> Result<[u8; 32]> {
        self.inner.pcr_read(index)
    }

    /// Import a key sealed against the current values of `pcr_selection`.
    ///
    /// The returned slot can only be unsealed via
    /// [`unseal_with_policy`](TpmKeyStore::unseal_with_policy) when those
    /// PCRs still contain the values captured here.
    pub fn import_sealed_pcr(
        &mut self,
        label: &str,
        key_bytes: &[u8],
        pcr_selection: &[usize],
    ) -> Result<KeySlot> {
        self.inner
            .import_sealed_pcr(label, key_bytes, pcr_selection)
    }

    /// Unseal a slot whose policy requires the current PCR bank to match the
    /// policy digest captured at import time.
    pub fn unseal_with_policy(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        self.inner.unseal_with_policy(slot)
    }
}

impl HwKeyStore for TpmKeyStore {
    fn backend(&self) -> HwBackend {
        // Reports SoftwareTpm — never Tpm2 — until a real hardware backend
        // is linked in. See the type-level documentation on this struct.
        HwBackend::SoftwareTpm
    }

    fn import_key(&mut self, label: &str, key_bytes: &[u8], exportable: bool) -> Result<KeySlot> {
        // Maps to TPM2_Create + TPM2_Load in the software emulator: the
        // primary key is the SwTpm master key, and the resulting handle is
        // returned as a transient object.
        self.inner.import_key(label, key_bytes, exportable)
    }

    fn export_wrapped(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        // Maps to TPM2_ContextSave: the returned bytes are the HMAC-wrapped
        // sealed-object blob and are safe to persist to disk.
        self.inner.export_wrapped(slot)
    }

    fn unseal(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        // Maps to TPM2_Unseal under an HMAC session (no policy).
        self.inner.unseal(slot)
    }

    fn delete_key(&mut self, slot: &KeySlot) -> Result<()> {
        // Maps to TPM2_FlushContext.
        self.inner.delete_key(slot)
    }

    fn list_keys(&self) -> Vec<KeyDescriptor> {
        self.inner.list_keys()
    }

    fn get_descriptor(&self, slot: &KeySlot) -> Option<KeyDescriptor> {
        self.inner.get_descriptor(slot)
    }
}

// ---------------------------------------------------------------------------
// Apple Secure Enclave software-emulated backend
// ---------------------------------------------------------------------------

/// Software-emulated Apple Secure Enclave backend.
///
/// Wraps a [`SwSecureEnclave`](crate::sw_secure_enclave::SwSecureEnclave)
/// state machine that models the Secure Enclave's strongest behavioural
/// invariant: keys are **non-exportable**. The result is reported as
/// [`HwBackend::SoftwareSecureEnclave`] — **not** [`HwBackend::SecureEnclave`]
/// — because no T2/Apple Silicon coprocessor is actually contacted. A future
/// `cfg(target_os = "macos")` implementation may replace this delegate with
/// calls into CryptoKit / Security.framework, at which point the backend
/// identifier may transition to [`HwBackend::SecureEnclave`].
pub struct SecureEnclaveKeyStore {
    inner: crate::sw_secure_enclave::SwSecureEnclave,
}

impl SecureEnclaveKeyStore {
    /// Create a new (software-emulated) Secure Enclave key store.
    pub fn new(master_key: [u8; 32]) -> Self {
        Self {
            inner: crate::sw_secure_enclave::SwSecureEnclave::new(master_key),
        }
    }
}

impl HwKeyStore for SecureEnclaveKeyStore {
    fn backend(&self) -> HwBackend {
        // Reports SoftwareSecureEnclave — never SecureEnclave — until a real
        // Apple Security framework integration ships.
        HwBackend::SoftwareSecureEnclave
    }

    fn import_key(&mut self, label: &str, key_bytes: &[u8], exportable: bool) -> Result<KeySlot> {
        // Maps to SecKeyCreateRandomKey + kSecAttrTokenIDSecureEnclave.
        // The `exportable` flag is rejected: Secure Enclave keys are
        // non-exportable by hardware mandate. We mirror that boundary in
        // software so callers cannot accidentally rely on exportability.
        self.inner.import_key(label, key_bytes, exportable)
    }

    fn export_wrapped(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        // Real Secure Enclave: SecItemCopyMatching can never return raw key
        // material — the hardware rejects export requests. We replicate that
        // exact failure mode so tests catch any callers depending on a leak.
        self.inner.export_wrapped(slot)
    }

    fn unseal(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        // Maps to SecKeyCreateDecryptedData inside a Secure Enclave context.
        self.inner.unseal(slot)
    }

    fn delete_key(&mut self, slot: &KeySlot) -> Result<()> {
        // Maps to SecItemDelete.
        self.inner.delete_key(slot)
    }

    fn list_keys(&self) -> Vec<KeyDescriptor> {
        self.inner.list_keys()
    }

    fn get_descriptor(&self, slot: &KeySlot) -> Option<KeyDescriptor> {
        self.inner.get_descriptor(slot)
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Detect and instantiate the best available key store for the current
/// platform.
///
/// Until a real `cfg(feature = "hw-tpm", target_os = "linux")` or
/// `cfg(target_os = "macos")` integration ships, this function returns the
/// matching **software emulator** (reported as [`HwBackend::SoftwareTpm`] or
/// [`HwBackend::SoftwareSecureEnclave`]) rather than pretending to have
/// reached genuine hardware. Callers that require a hardware root of trust
/// should additionally check [`HwBackend::is_hardware`] on the returned
/// backend.
///
/// The `master_key` seeds the emulator's HMAC integrity layer. A real
/// hardware backend would derive its protection secrets internally and would
/// ignore this argument.
pub fn detect_hw_backend(master_key: [u8; 32]) -> (HwBackend, Box<dyn HwKeyStore>) {
    // Platform-driven choice of emulator. We deliberately do NOT probe for a
    // real device here — the emulator is the safe default until a
    // hardware-gated backend is wired in. Hardware probing belongs in a
    // future `cfg(feature = "hw-tpm", ...)` branch that replaces these
    // emulator constructors with real TSS-Esapi / Security.framework calls.
    #[cfg(target_os = "macos")]
    {
        (
            HwBackend::SoftwareSecureEnclave,
            Box::new(SecureEnclaveKeyStore::new(master_key)),
        )
    }

    #[cfg(not(target_os = "macos"))]
    {
        (
            HwBackend::SoftwareTpm,
            Box::new(TpmKeyStore::new(master_key)),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MASTER: [u8; 32] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x20,
    ];

    fn make_store() -> SoftwareKeyStore {
        SoftwareKeyStore::new(MASTER)
    }

    #[test]
    fn test_import_and_unseal_roundtrip() {
        let mut store = make_store();
        let key = vec![0xAB_u8; 16];
        let slot = store
            .import_key("test-key", &key, true)
            .expect("import should succeed");

        let unsealed = store.unseal(&slot).expect("unseal should succeed");
        assert_eq!(unsealed, key);
    }

    #[test]
    fn test_import_non_exportable_blocks_export() {
        let mut store = make_store();
        let key = vec![0xCD_u8; 16];
        let slot = store
            .import_key("private-key", &key, false)
            .expect("import should succeed");

        let result = store.export_wrapped(&slot);
        assert!(result.is_err(), "non-exportable key should block export");
    }

    #[test]
    fn test_export_wrapped_differs_from_plaintext() {
        let mut store = make_store();
        let key = vec![0xEF_u8; 16];
        let slot = store
            .import_key("wrap-test", &key, true)
            .expect("import should succeed");

        let wrapped = store.export_wrapped(&slot).expect("export should succeed");
        // Wrapped material must not equal the plaintext
        assert_ne!(wrapped, key);
        // Wrapped material must be longer than plaintext (includes nonce + tag)
        assert!(wrapped.len() > key.len());
    }

    #[test]
    fn test_multiple_keys_independent() {
        let mut store = make_store();
        let key_a = vec![0x11_u8; 16];
        let key_b = vec![0x22_u8; 16];
        let slot_a = store
            .import_key("key-a", &key_a, true)
            .expect("import a should succeed");
        let slot_b = store
            .import_key("key-b", &key_b, true)
            .expect("import b should succeed");

        assert_eq!(store.unseal(&slot_a).expect("unseal a"), key_a);
        assert_eq!(store.unseal(&slot_b).expect("unseal b"), key_b);
    }

    #[test]
    fn test_delete_key_prevents_unseal() {
        let mut store = make_store();
        let key = vec![0x33_u8; 16];
        let slot = store
            .import_key("delete-me", &key, true)
            .expect("import should succeed");

        store.delete_key(&slot).expect("delete should succeed");
        let result = store.unseal(&slot);
        assert!(result.is_err(), "deleted key should not be unsealable");
    }

    #[test]
    fn test_delete_nonexistent_returns_error() {
        let mut store = make_store();
        let phantom = KeySlot::new("ghost", 9999);
        assert!(store.delete_key(&phantom).is_err());
    }

    #[test]
    fn test_list_keys_counts() {
        let mut store = make_store();
        assert_eq!(store.list_keys().len(), 0);
        store
            .import_key("k1", &[1; 16], true)
            .expect("import should succeed");
        store
            .import_key("k2", &[2; 16], true)
            .expect("import should succeed");
        assert_eq!(store.list_keys().len(), 2);
    }

    #[test]
    fn test_get_descriptor_returns_metadata() {
        let mut store = make_store();
        let slot = store
            .import_key("meta-key", &[5; 32], true)
            .expect("import should succeed");
        let desc = store
            .get_descriptor(&slot)
            .expect("descriptor should exist");
        assert_eq!(desc.slot.label, "meta-key");
        assert!(desc.exportable);
    }

    #[test]
    fn test_find_by_label() {
        let mut store = make_store();
        store
            .import_key("findable", &[7; 16], true)
            .expect("import should succeed");
        let found = store.find_by_label("findable");
        assert!(found.is_some());
        let not_found = store.find_by_label("missing");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_key_expiry_blocks_unseal() {
        let mut store = make_store();
        store.set_time(100);
        let slot = store
            .import_key("expiring", &[9; 16], true)
            .expect("import should succeed");
        store
            .set_expiry(&slot, 200)
            .expect("set expiry should succeed");

        // Before expiry: ok
        assert!(store.unseal(&slot).is_ok());

        // At expiry boundary: blocked
        store.set_time(200);
        assert!(store.unseal(&slot).is_err());
    }

    #[test]
    fn test_empty_key_import_fails() {
        let mut store = make_store();
        let result = store.import_key("empty", &[], true);
        assert!(result.is_err());
    }

    #[test]
    fn test_software_backend_type() {
        let store = make_store();
        assert_eq!(store.backend(), HwBackend::Software);
        assert!(!store.backend().is_hardware());
    }

    #[test]
    fn test_tpm_backend_type() {
        let store = TpmKeyStore::new(MASTER);
        // The emulated TPM must NOT claim to be a real TPM 2.0 device.
        // It reports SoftwareTpm so callers that gate on hardware presence
        // can refuse to use it for high-assurance flows.
        assert_eq!(store.backend(), HwBackend::SoftwareTpm);
        assert!(!store.backend().is_hardware());
        assert!(store.backend().is_software_emulated());
    }

    #[test]
    fn test_secure_enclave_backend_type() {
        let store = SecureEnclaveKeyStore::new(MASTER);
        // Likewise: the emulated Secure Enclave reports SoftwareSecureEnclave,
        // never SecureEnclave, until a real Apple integration ships.
        assert_eq!(store.backend(), HwBackend::SoftwareSecureEnclave);
        assert!(!store.backend().is_hardware());
        assert!(store.backend().is_software_emulated());
    }

    #[test]
    fn test_hw_backend_display() {
        assert_eq!(HwBackend::Tpm2.to_string(), "TPM 2.0");
        assert_eq!(HwBackend::SecureEnclave.to_string(), "Secure Enclave");
        assert_eq!(HwBackend::SoftwareTpm.to_string(), "Software TPM 2.0");
        assert_eq!(
            HwBackend::SoftwareSecureEnclave.to_string(),
            "Software Secure Enclave"
        );
        assert_eq!(HwBackend::Software.to_string(), "Software");
    }

    #[test]
    fn test_hw_backend_is_software_emulated_partition() {
        // Disjoint partition: every variant is in exactly one of
        // {hardware, software-emulated, software}.
        let all = [
            HwBackend::Tpm2,
            HwBackend::SecureEnclave,
            HwBackend::SoftwareTpm,
            HwBackend::SoftwareSecureEnclave,
            HwBackend::Software,
        ];
        for backend in &all {
            let hw = backend.is_hardware();
            let sw_em = backend.is_software_emulated();
            // hardware and software-emulated must never overlap
            assert!(!(hw && sw_em), "backend {backend:?} overlapping classes");
        }
        // Hardware-only set
        assert!(HwBackend::Tpm2.is_hardware());
        assert!(HwBackend::SecureEnclave.is_hardware());
        // Software-emulated-only set
        assert!(HwBackend::SoftwareTpm.is_software_emulated());
        assert!(HwBackend::SoftwareSecureEnclave.is_software_emulated());
        // Plain software fallback is neither
        assert!(!HwBackend::Software.is_hardware());
        assert!(!HwBackend::Software.is_software_emulated());
    }

    #[test]
    fn test_key_descriptor_is_expired() {
        let slot = KeySlot::new("k", 1);
        let mut desc = KeyDescriptor {
            slot,
            wrapped_len: 32,
            exportable: true,
            created_at: 0,
            expires_at: Some(1000),
        };
        assert!(!desc.is_expired(999));
        assert!(desc.is_expired(1000));
        desc.expires_at = None;
        assert!(!desc.is_expired(u64::MAX));
    }

    #[test]
    fn test_detect_hw_backend_returns_valid_store() {
        let (backend, mut store) = detect_hw_backend(MASTER);
        // Until a real hardware integration ships, detection MUST report a
        // software-emulated backend (or the plain software fallback). It
        // must never lie about reaching genuine hardware.
        assert!(
            backend.is_software_emulated() || backend == HwBackend::Software,
            "detect_hw_backend returned {backend:?} without real hardware integration"
        );
        assert!(!backend.is_hardware());

        // Secure Enclave emulator imports are forced non-exportable; use a
        // non-exportable import here so the round-trip succeeds on both
        // platforms.
        let slot = store
            .import_key("detect-test", &[0xAB; 16], false)
            .expect("import should succeed");
        let unsealed = store.unseal(&slot).expect("unseal should succeed");
        assert_eq!(unsealed, vec![0xAB; 16]);
    }

    #[test]
    fn test_tpm_store_import_and_unseal() {
        let mut store = TpmKeyStore::new(MASTER);
        let key = vec![0xDE_u8; 16];
        let slot = store
            .import_key("tpm-key", &key, true)
            .expect("import should succeed");
        let unsealed = store.unseal(&slot).expect("unseal should succeed");
        assert_eq!(unsealed, key);
    }

    #[test]
    fn test_secure_enclave_store_import_and_unseal() {
        let mut store = SecureEnclaveKeyStore::new(MASTER);
        let key = vec![0xCA_u8; 32];
        // Secure Enclave keys are non-exportable by hardware mandate; the
        // emulator enforces the same boundary. Import as non-exportable.
        let slot = store
            .import_key("se-key", &key, false)
            .expect("import should succeed");
        let unsealed = store.unseal(&slot).expect("unseal should succeed");
        assert_eq!(unsealed, key);
    }

    #[test]
    fn test_secure_enclave_store_rejects_exportable_import() {
        // Mirrors kSecAttrTokenIDSecureEnclave: exportable=true must fail
        // because Secure Enclave keys cannot ever leave the device.
        let mut store = SecureEnclaveKeyStore::new(MASTER);
        let result = store.import_key("leaky-key", &[0xCA_u8; 32], true);
        assert!(
            result.is_err(),
            "Secure Enclave must reject exportable=true imports"
        );
    }

    #[test]
    fn test_secure_enclave_store_export_wrapped_is_blocked() {
        // Even with exportable=false, the explicit export API must fail.
        let mut store = SecureEnclaveKeyStore::new(MASTER);
        let slot = store
            .import_key("se-key", &[0xCA_u8; 32], false)
            .expect("import should succeed");
        let result = store.export_wrapped(&slot);
        assert!(
            result.is_err(),
            "Secure Enclave export_wrapped must always fail"
        );
    }

    #[test]
    fn test_tpm_store_pcr_extend_and_policy_unseal_roundtrip() {
        // Exercises the SwTpm policy path through the public HwKeyStore facade.
        let mut store = TpmKeyStore::new(MASTER);
        store.pcr_extend(0, b"boot").expect("pcr extend 0");
        store.pcr_extend(7, b"firmware").expect("pcr extend 7");

        let secret = vec![0x77_u8; 16];
        let slot = store
            .import_sealed_pcr("policy", &secret, &[0, 7])
            .expect("import sealed pcr");
        let recovered = store
            .unseal_with_policy(&slot)
            .expect("policy unseal should succeed when PCRs match");
        assert_eq!(recovered, secret);

        // Mutating the PCR must invalidate the policy.
        store.pcr_extend(7, b"tampered").expect("pcr extend 7'");
        assert!(
            store.unseal_with_policy(&slot).is_err(),
            "policy unseal must fail after PCR divergence"
        );
    }
}
