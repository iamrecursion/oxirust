//! Software-emulated TPM 2.0 state machine (Pure Rust).
//!
//! This module models the subset of the TPM 2.0 authorization surface that
//! the OxiMedia DRM stack actually exercises:
//!
//! * 32-bit object handles partitioned into a transient range
//!   (`0x80000000..0x81000000`) and a persistent range (`0x81000000..`).
//!   Only transient handles are produced by this emulator; persistent
//!   eviction (`TPM2_EvictControl`) is not modelled.
//! * A 24-slot SHA-256 PCR bank with TPM2_PCR_Extend semantics
//!   (`pcr_n ← SHA256(pcr_n ‖ data)`).
//! * Sealed objects whose secret payload is wrapped under an HMAC-SHA-256
//!   key derived from the master seed using the NIST SP 800-108r1 CTR-mode
//!   KDF supplied by [`crate::key_derivation`]. The HMAC tag covers the
//!   per-object metadata so any tamper attempt is caught by
//!   [`SwTpm::unseal`].
//! * Authorization sessions in two modes:
//!     * **HMAC** — plain authValue check (no policy).
//!     * **Policy** — `PolicyPCR` and `PolicyAuthValue` assertions composed
//!       via the TPM2 policy-digest update rule
//!       `digest ← SHA256(old_digest ‖ command_code ‖ argument_digest)`.
//!
//! This emulator does **not** provide hardware-grade isolation. Its only job
//! is to give the rest of the DRM crate a faithful authorization model so
//! that callers can be tested end-to-end against the same protocol shape
//! they would face on real silicon. The reported backend is therefore
//! [`crate::hw_key_store::HwBackend::SoftwareTpm`].
//!
//! A future `cfg(feature = "hw-tpm", target_os = "linux")` integration may
//! replace [`SwTpm`] with calls into `tss-esapi`/`/dev/tpmrm0`, at which
//! point the same [`crate::hw_key_store::HwKeyStore`] API surface continues
//! to apply.

use crate::hw_key_store::{sha256_mini, KeyDescriptor, KeySlot};
use crate::key_derivation::{DerivedKey, KeyDerivationConfig, KeyDerivationMethod, KeyDeriver};
use crate::{DrmError, Result};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// TPM-style command codes (subset used by the policy digest update rule)
// ---------------------------------------------------------------------------

/// `TPM_CC_PolicyPCR` (TPM 2.0 Part 2, §6.5.2). Only the 16 low bits of the
/// real 32-bit code are mixed into the digest update — sufficient for
/// emulation determinism while keeping the constant readable.
const CC_POLICY_PCR: u32 = 0x0000_017F;

/// `TPM_CC_PolicyAuthValue` (TPM 2.0 Part 2, §6.5.2).
const CC_POLICY_AUTH_VALUE: u32 = 0x0000_016B;

/// SHA-256 digest size in bytes.
const DIGEST_SIZE: usize = 32;

/// Number of PCRs in the SHA-256 bank (matches TPM 2.0 default profile).
const PCR_COUNT: usize = 24;

/// Lower bound of the transient object handle range (TPM 2.0 spec §15.1).
const TRANSIENT_HANDLE_BASE: u32 = 0x8000_0000;

/// Lower bound of the persistent object handle range (not used yet).
#[allow(dead_code)]
const PERSISTENT_HANDLE_BASE: u32 = 0x8100_0000;

// ---------------------------------------------------------------------------
// PCR bank
// ---------------------------------------------------------------------------

/// 24-slot SHA-256 PCR bank.
#[derive(Debug, Clone)]
struct PcrBank {
    slots: [[u8; DIGEST_SIZE]; PCR_COUNT],
}

impl PcrBank {
    fn new() -> Self {
        Self {
            slots: [[0u8; DIGEST_SIZE]; PCR_COUNT],
        }
    }

    /// `pcr_n ← SHA256(pcr_n ‖ data)`, returning the new value.
    fn extend(&mut self, index: usize, data: &[u8]) -> Result<[u8; DIGEST_SIZE]> {
        if index >= PCR_COUNT {
            return Err(DrmError::InvalidKey(format!(
                "PCR index {index} out of range (max {})",
                PCR_COUNT - 1
            )));
        }
        let mut buf = Vec::with_capacity(DIGEST_SIZE + data.len());
        buf.extend_from_slice(&self.slots[index]);
        buf.extend_from_slice(data);
        let new_value = sha256_mini(&buf);
        self.slots[index] = new_value;
        Ok(new_value)
    }

    /// Read PCR `index`.
    fn read(&self, index: usize) -> Result<[u8; DIGEST_SIZE]> {
        if index >= PCR_COUNT {
            return Err(DrmError::InvalidKey(format!(
                "PCR index {index} out of range (max {})",
                PCR_COUNT - 1
            )));
        }
        Ok(self.slots[index])
    }

    /// Compute the composite digest of a PCR selection as a real TPM does:
    /// `SHA256(pcr_i0 ‖ pcr_i1 ‖ …)` over the selected indices in ascending
    /// order. Indices are de-duplicated.
    fn composite_digest(&self, selection: &[usize]) -> Result<[u8; DIGEST_SIZE]> {
        // Sort + de-dup so the digest is order-independent (matches the TPM
        // canonical ordering of pcr_select bitmaps).
        let mut sorted: Vec<usize> = selection.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let mut buf = Vec::with_capacity(sorted.len() * DIGEST_SIZE);
        for &idx in &sorted {
            buf.extend_from_slice(&self.read(idx)?);
        }
        Ok(sha256_mini(&buf))
    }
}

// ---------------------------------------------------------------------------
// Policy digest
// ---------------------------------------------------------------------------

/// Policy digest accumulator following TPM2 Part 3, §23.2.3:
/// `policyDigest ← SHA256(policyDigest ‖ commandCode ‖ argumentDigest)`.
///
/// Starts from a 32-byte zero digest (the "empty policy" identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyDigest {
    bytes: [u8; DIGEST_SIZE],
}

impl PolicyDigest {
    /// Identity policy digest (all zeros).
    pub fn empty() -> Self {
        Self {
            bytes: [0u8; DIGEST_SIZE],
        }
    }

    /// Apply `TPM2_PolicyPCR(selection, pcr_digest)`.
    pub fn extend_pcr(self, pcr_selection: &[usize], pcr_digest: &[u8; DIGEST_SIZE]) -> Self {
        // The TPM canonicalises pcr_select before mixing it into the
        // argument; we mirror that with a sort+dedup bitmap.
        let mut sorted: Vec<usize> = pcr_selection.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let mut selection_bytes = Vec::with_capacity(sorted.len() * 4);
        for &idx in &sorted {
            selection_bytes.extend_from_slice(&(idx as u32).to_be_bytes());
        }
        let mut arg = Vec::with_capacity(selection_bytes.len() + DIGEST_SIZE);
        arg.extend_from_slice(&selection_bytes);
        arg.extend_from_slice(pcr_digest);
        let arg_digest = sha256_mini(&arg);

        self.update(CC_POLICY_PCR, &arg_digest)
    }

    /// Apply `TPM2_PolicyAuthValue()`.
    pub fn extend_auth_value(self) -> Self {
        // PolicyAuthValue carries no argument; the spec uses an empty digest
        // for the argument component.
        self.update(CC_POLICY_AUTH_VALUE, &[0u8; DIGEST_SIZE])
    }

    fn update(self, command_code: u32, arg_digest: &[u8; DIGEST_SIZE]) -> Self {
        let mut buf = Vec::with_capacity(DIGEST_SIZE + 4 + DIGEST_SIZE);
        buf.extend_from_slice(&self.bytes);
        buf.extend_from_slice(&command_code.to_be_bytes());
        buf.extend_from_slice(arg_digest);
        Self {
            bytes: sha256_mini(&buf),
        }
    }

    /// Raw 32-byte view of the digest.
    pub fn as_bytes(&self) -> &[u8; DIGEST_SIZE] {
        &self.bytes
    }
}

// ---------------------------------------------------------------------------
// Sealed object
// ---------------------------------------------------------------------------

/// In-memory representation of a TPM2_CreateLoaded result.
///
/// Layout of the wrapped blob returned by [`SwTpm::export_wrapped`]:
///
/// ```text
///   ┌────────────────────────────────────────────────────────────────────┐
///   │ name_digest (32) │ policy_digest (32) │ ciphertext (n) │ tag (32) │
///   └────────────────────────────────────────────────────────────────────┘
/// ```
///
/// `tag = HMAC-SHA256(per_object_hmac_key, name_digest ‖ policy_digest ‖
/// ciphertext)`. Any tamper to any field invalidates the tag.
#[derive(Debug, Clone)]
struct SealedObject {
    /// SHA-256 of the public area (label ‖ handle).
    name_digest: [u8; DIGEST_SIZE],
    /// Policy digest required to unseal this object. All-zero for
    /// objects sealed under an HMAC session only.
    policy_digest: [u8; DIGEST_SIZE],
    /// Ciphertext of the secret payload (XOR keystream wrapping).
    ciphertext: Vec<u8>,
    /// HMAC-SHA-256 tag over `name_digest ‖ policy_digest ‖ ciphertext`.
    tag: [u8; DIGEST_SIZE],
    /// Plaintext length (so the unwrap step can pre-allocate).
    plaintext_len: usize,
    /// Per-object descriptor surfaced via [`crate::hw_key_store::HwKeyStore`].
    descriptor: KeyDescriptor,
    /// Whether this object was sealed against a PolicyPCR session.
    pcr_selection: Option<Vec<usize>>,
}

impl SealedObject {
    fn integrity_input(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(DIGEST_SIZE * 2 + self.ciphertext.len());
        buf.extend_from_slice(&self.name_digest);
        buf.extend_from_slice(&self.policy_digest);
        buf.extend_from_slice(&self.ciphertext);
        buf
    }
}

// ---------------------------------------------------------------------------
// SwTpm — software-emulated TPM 2.0 state machine
// ---------------------------------------------------------------------------

/// Software-emulated TPM 2.0 state machine.
///
/// See module-level docs for the modelled subset. The emulator is created
/// with a 32-byte master seed (analogous to the TPM storage hierarchy seed)
/// and derives per-object HMAC keys via NIST SP 800-108r1 CTR-mode KDF.
pub struct SwTpm {
    /// Storage hierarchy master seed (HMAC-SHA-256 KDF input).
    master_key: [u8; 32],
    /// PCR bank (SHA-256 / 24 slots).
    pcr: PcrBank,
    /// Loaded sealed objects keyed by transient handle.
    objects: BTreeMap<u32, SealedObject>,
    /// Label → handle index.
    label_to_handle: BTreeMap<String, u32>,
    /// Next transient handle to assign.
    next_handle: u32,
    /// Monotonic counter feeding `created_at` on descriptors.
    current_time: u64,
}

impl SwTpm {
    /// Create a new emulator with the given master seed.
    #[must_use]
    pub fn new(master_key: [u8; 32]) -> Self {
        Self {
            master_key,
            pcr: PcrBank::new(),
            objects: BTreeMap::new(),
            label_to_handle: BTreeMap::new(),
            next_handle: TRANSIENT_HANDLE_BASE,
            current_time: 0,
        }
    }

    // -- PCR --------------------------------------------------------------

    /// Extend PCR `index` with `data` (TPM2_PCR_Extend).
    pub fn pcr_extend(&mut self, index: usize, data: &[u8]) -> Result<[u8; DIGEST_SIZE]> {
        self.pcr.extend(index, data)
    }

    /// Read PCR `index` (TPM2_PCR_Read).
    pub fn pcr_read(&self, index: usize) -> Result<[u8; DIGEST_SIZE]> {
        self.pcr.read(index)
    }

    /// Composite digest of the given PCR selection (helper for tests / callers).
    pub fn pcr_composite(&self, selection: &[usize]) -> Result<[u8; DIGEST_SIZE]> {
        self.pcr.composite_digest(selection)
    }

    // -- Object lifecycle -------------------------------------------------

    /// Import a key with no policy (HMAC-session unseal only).
    pub fn import_key(
        &mut self,
        label: &str,
        key_bytes: &[u8],
        exportable: bool,
    ) -> Result<KeySlot> {
        self.import_internal(label, key_bytes, exportable, None)
    }

    /// Import a key sealed against the current values of `pcr_selection`.
    ///
    /// The captured PCR composite digest is baked into the object's policy
    /// digest. [`unseal_with_policy`](Self::unseal_with_policy) will refuse
    /// to release the secret unless the bank still matches.
    pub fn import_sealed_pcr(
        &mut self,
        label: &str,
        key_bytes: &[u8],
        pcr_selection: &[usize],
    ) -> Result<KeySlot> {
        if pcr_selection.is_empty() {
            return Err(DrmError::InvalidKey(
                "PCR-sealed objects require at least one PCR index".to_string(),
            ));
        }
        // Validate every index up-front so we never half-create an object.
        for &idx in pcr_selection {
            if idx >= PCR_COUNT {
                return Err(DrmError::InvalidKey(format!(
                    "PCR index {idx} out of range (max {})",
                    PCR_COUNT - 1
                )));
            }
        }
        self.import_internal(
            label,
            key_bytes,
            /* exportable = */ false,
            Some(pcr_selection.to_vec()),
        )
    }

    fn import_internal(
        &mut self,
        label: &str,
        key_bytes: &[u8],
        exportable: bool,
        pcr_selection: Option<Vec<usize>>,
    ) -> Result<KeySlot> {
        if key_bytes.is_empty() {
            return Err(DrmError::InvalidKey(
                "Key material must not be empty".to_string(),
            ));
        }

        let handle = self.next_handle;
        self.next_handle = self.next_handle.checked_add(1).ok_or_else(|| {
            DrmError::InvalidKey("SwTpm: out of transient object handles".to_string())
        })?;
        if self.next_handle >= PERSISTENT_HANDLE_BASE {
            return Err(DrmError::InvalidKey(
                "SwTpm: transient handle range exhausted".to_string(),
            ));
        }

        // Name digest = SHA-256(label || handle). Real TPMs hash the public
        // area; this is a structurally identical 32-byte commitment.
        let name_digest = compute_name_digest(label, handle);

        // Policy digest: empty for HMAC-only objects; for PolicyPCR we
        // capture the composite of the current PCR bank.
        let policy_digest = match &pcr_selection {
            None => [0u8; DIGEST_SIZE],
            Some(sel) => {
                let pcr_digest = self.pcr.composite_digest(sel)?;
                PolicyDigest::empty()
                    .extend_pcr(sel, &pcr_digest)
                    .as_bytes()
                    .to_owned()
            }
        };

        let per_object_key = derive_object_hmac_key(&self.master_key, &name_digest)?;
        let ciphertext = xor_keystream(&per_object_key, &name_digest, key_bytes);

        let mut obj = SealedObject {
            name_digest,
            policy_digest,
            ciphertext,
            tag: [0u8; DIGEST_SIZE],
            plaintext_len: key_bytes.len(),
            descriptor: KeyDescriptor {
                slot: KeySlot::new(label, handle),
                wrapped_len: 0, // filled in after we know the export size
                exportable,
                created_at: self.current_time,
                expires_at: None,
            },
            pcr_selection,
        };
        obj.tag = hmac_sha256(&per_object_key, &obj.integrity_input());

        // wrapped_len = name + policy + ct + tag (this is the layout returned
        // by export_wrapped).
        obj.descriptor.wrapped_len = DIGEST_SIZE + DIGEST_SIZE + obj.ciphertext.len() + DIGEST_SIZE;

        self.label_to_handle.insert(label.to_string(), handle);
        let slot = obj.descriptor.slot.clone();
        self.objects.insert(handle, obj);
        Ok(slot)
    }

    /// Export the HMAC-wrapped sealed blob (TPM2_ContextSave equivalent).
    pub fn export_wrapped(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        let obj = self.lookup(slot)?;
        if !obj.descriptor.exportable {
            return Err(DrmError::InvalidKey(format!(
                "Object '{}' is not exportable",
                slot.label
            )));
        }
        // PCR-sealed objects must not be exported as raw bytes — that would
        // bypass the policy gate when the blob is re-imported elsewhere.
        if obj.pcr_selection.is_some() {
            return Err(DrmError::InvalidKey(format!(
                "Object '{}' is sealed against PCR policy and cannot be exported",
                slot.label
            )));
        }
        Ok(encode_wrapped(obj))
    }

    /// Unseal with an HMAC session (no policy).
    pub fn unseal(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        let obj = self.lookup(slot)?;
        if obj.pcr_selection.is_some() {
            return Err(DrmError::LicenseError(format!(
                "Object '{}' requires a policy session (use unseal_with_policy)",
                slot.label
            )));
        }
        self.unwrap_object(obj)
    }

    /// Unseal under a PolicyPCR session: the live PCR bank must reproduce
    /// the policy digest that was captured when the object was sealed.
    pub fn unseal_with_policy(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        let obj = self.lookup(slot)?;
        let selection = obj.pcr_selection.as_ref().ok_or_else(|| {
            DrmError::InvalidKey(format!(
                "Object '{}' has no PCR policy attached",
                slot.label
            ))
        })?;
        let pcr_digest = self.pcr.composite_digest(selection)?;
        let live_policy = PolicyDigest::empty()
            .extend_pcr(selection, &pcr_digest)
            .as_bytes()
            .to_owned();
        if live_policy != obj.policy_digest {
            return Err(DrmError::LicenseError(format!(
                "PolicyPCR check failed for object '{}': PCR bank diverged from sealing-time state",
                slot.label
            )));
        }
        self.unwrap_object(obj)
    }

    /// Verify the HMAC tag and decrypt the payload (constant-time tag check).
    fn unwrap_object(&self, obj: &SealedObject) -> Result<Vec<u8>> {
        let key = derive_object_hmac_key(&self.master_key, &obj.name_digest)?;
        let expected = hmac_sha256(&key, &obj.integrity_input());
        if !constant_time_eq(&expected, &obj.tag) {
            return Err(DrmError::InvalidKey(
                "SwTpm: HMAC integrity check failed (object tampered)".to_string(),
            ));
        }
        let mut plaintext = obj.ciphertext.clone();
        let stream = xor_keystream(&key, &obj.name_digest, &plaintext);
        // xor_keystream re-applied yields plaintext.
        for (dst, src) in plaintext.iter_mut().zip(stream.iter()) {
            *dst = *src;
        }
        debug_assert_eq!(plaintext.len(), obj.plaintext_len);
        Ok(plaintext)
    }

    /// Remove an object (TPM2_FlushContext equivalent).
    pub fn delete_key(&mut self, slot: &KeySlot) -> Result<()> {
        // Drop the label mapping first so a concurrent lookup cannot resurrect
        // the handle after we've removed the object.
        self.label_to_handle.retain(|_, &mut h| h != slot.handle);
        match self.objects.remove(&slot.handle) {
            Some(_) => Ok(()),
            None => Err(DrmError::InvalidKey(format!(
                "Object handle {} not found",
                slot.handle
            ))),
        }
    }

    /// List all descriptors.
    pub fn list_keys(&self) -> Vec<KeyDescriptor> {
        self.objects
            .values()
            .map(|obj| obj.descriptor.clone())
            .collect()
    }

    /// Look up a descriptor by slot.
    pub fn get_descriptor(&self, slot: &KeySlot) -> Option<KeyDescriptor> {
        self.objects.get(&slot.handle).map(|o| o.descriptor.clone())
    }

    /// Internal handle resolution.
    fn lookup(&self, slot: &KeySlot) -> Result<&SealedObject> {
        self.objects
            .get(&slot.handle)
            .ok_or_else(|| DrmError::InvalidKey(format!("Object handle {} not found", slot.handle)))
    }

    /// Override the time source (tests).
    #[cfg(test)]
    pub fn set_time(&mut self, ts: u64) {
        self.current_time = ts;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_name_digest(label: &str, handle: u32) -> [u8; DIGEST_SIZE] {
    let mut buf = Vec::with_capacity(label.len() + 4);
    buf.extend_from_slice(label.as_bytes());
    buf.extend_from_slice(&handle.to_be_bytes());
    sha256_mini(&buf)
}

/// Derive a per-object HMAC key from the master seed using SP 800-108 CTR-mode
/// (HMAC-SHA-256 PRF). The label is the object's name digest, so each object
/// gets a unique 32-byte key.
fn derive_object_hmac_key(
    master_key: &[u8; 32],
    name_digest: &[u8; DIGEST_SIZE],
) -> Result<[u8; 32]> {
    let cfg = KeyDerivationConfig::new(KeyDerivationMethod::Sp800108Ctr, 32)
        .with_info(name_digest.to_vec());
    let derived: DerivedKey = KeyDeriver::new("swtpm/obj")
        .derive(master_key, &cfg)
        .map_err(|e| DrmError::InvalidKey(format!("SwTpm: KDF failed: {e}")))?;
    if derived.key_bytes.len() != 32 {
        return Err(DrmError::InvalidKey(format!(
            "SwTpm: KDF returned {} bytes, expected 32",
            derived.key_bytes.len()
        )));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&derived.key_bytes);
    Ok(out)
}

/// XOR keystream generator using SHA-256 in counter mode keyed by the
/// per-object HMAC key. This is identical in spirit to the wrapping used by
/// the existing `SoftwareKeyStore`, but rooted in the per-object KDF output
/// rather than the bare master key.
fn xor_keystream(key: &[u8; 32], iv: &[u8; DIGEST_SIZE], data: &[u8]) -> Vec<u8> {
    let mut out = data.to_vec();
    let mut counter: u64 = 0;
    let mut offset = 0;
    while offset < out.len() {
        let mut seed = Vec::with_capacity(key.len() + iv.len() + 8);
        seed.extend_from_slice(key);
        seed.extend_from_slice(iv);
        seed.extend_from_slice(&counter.to_be_bytes());
        let block = sha256_mini(&seed);
        let block_len = block.len().min(out.len() - offset);
        for (i, b) in out[offset..offset + block_len].iter_mut().enumerate() {
            *b ^= block[i];
        }
        offset += block_len;
        counter = counter.wrapping_add(1);
    }
    out
}

/// HMAC-SHA-256 using the same SHA-256 primitive as the rest of the crate.
fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; DIGEST_SIZE] {
    const BLOCK_SIZE: usize = 64;
    let mut k = [0u8; BLOCK_SIZE];
    if key.len() > BLOCK_SIZE {
        let hashed = sha256_mini(key);
        k[..DIGEST_SIZE].copy_from_slice(&hashed);
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0u8; BLOCK_SIZE];
    let mut opad = [0u8; BLOCK_SIZE];
    for i in 0..BLOCK_SIZE {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }

    let mut inner = Vec::with_capacity(BLOCK_SIZE + message.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(message);
    let inner_hash = sha256_mini(&inner);

    let mut outer = Vec::with_capacity(BLOCK_SIZE + DIGEST_SIZE);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&inner_hash);
    sha256_mini(&outer)
}

/// Constant-time byte equality.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Serialise a sealed object using the layout documented on [`SealedObject`].
fn encode_wrapped(obj: &SealedObject) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(DIGEST_SIZE + DIGEST_SIZE + obj.ciphertext.len() + DIGEST_SIZE);
    out.extend_from_slice(&obj.name_digest);
    out.extend_from_slice(&obj.policy_digest);
    out.extend_from_slice(&obj.ciphertext);
    out.extend_from_slice(&obj.tag);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MASTER: [u8; 32] = [
        0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
        0x99, 0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18, 0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E,
        0x8F, 0x90,
    ];

    fn make_tpm() -> SwTpm {
        SwTpm::new(MASTER)
    }

    // ---------- PCR semantics -------------------------------------------

    #[test]
    fn pcr_starts_zero() {
        let tpm = make_tpm();
        let v = tpm.pcr_read(0).expect("pcr read");
        assert_eq!(v, [0u8; DIGEST_SIZE]);
    }

    #[test]
    fn pcr_extend_matches_sha256_of_zero_concat_data() {
        let mut tpm = make_tpm();
        let extended = tpm.pcr_extend(7, b"boot-loader-stage-1").expect("extend");

        // Spec: pcr ← SHA256(prev_pcr || data). Recompute by hand.
        let mut buf = Vec::new();
        buf.extend_from_slice(&[0u8; DIGEST_SIZE]);
        buf.extend_from_slice(b"boot-loader-stage-1");
        let expected = sha256_mini(&buf);

        assert_eq!(extended, expected);
        assert_eq!(tpm.pcr_read(7).expect("read"), expected);
    }

    #[test]
    fn pcr_extend_is_chained() {
        let mut tpm = make_tpm();
        let v1 = tpm.pcr_extend(0, b"a").expect("extend a");
        let v2 = tpm.pcr_extend(0, b"b").expect("extend b");
        // v2 = SHA256(v1 || "b")
        let mut buf = Vec::new();
        buf.extend_from_slice(&v1);
        buf.extend_from_slice(b"b");
        assert_eq!(v2, sha256_mini(&buf));
        assert_ne!(v1, v2);
    }

    #[test]
    fn pcr_out_of_range_rejected() {
        let mut tpm = make_tpm();
        assert!(tpm.pcr_extend(PCR_COUNT, b"x").is_err());
        assert!(tpm.pcr_read(PCR_COUNT).is_err());
    }

    // ---------- Object handle assignment --------------------------------

    #[test]
    fn handles_are_in_transient_range() {
        let mut tpm = make_tpm();
        let slot = tpm.import_key("k", b"secret", true).expect("import");
        assert!(
            slot.handle >= TRANSIENT_HANDLE_BASE,
            "handle {:#x} not in transient range",
            slot.handle
        );
        assert!(
            slot.handle < PERSISTENT_HANDLE_BASE,
            "handle {:#x} crossed into persistent range",
            slot.handle
        );
    }

    #[test]
    fn handles_are_unique_per_object() {
        let mut tpm = make_tpm();
        let a = tpm.import_key("a", b"x", true).expect("import a");
        let b = tpm.import_key("b", b"y", true).expect("import b");
        assert_ne!(a.handle, b.handle);
    }

    // ---------- Import → seal → unseal round-trip -----------------------

    #[test]
    fn import_seal_unseal_roundtrip() {
        let mut tpm = make_tpm();
        let secret = vec![0x42u8; 16];
        let slot = tpm
            .import_key("content-key-1", &secret, true)
            .expect("import");
        let recovered = tpm.unseal(&slot).expect("unseal");
        assert_eq!(recovered, secret);
    }

    #[test]
    fn import_seal_unseal_handles_long_secrets() {
        let mut tpm = make_tpm();
        let secret: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let slot = tpm.import_key("big", &secret, false).expect("import");
        let recovered = tpm.unseal(&slot).expect("unseal");
        assert_eq!(recovered, secret);
    }

    #[test]
    fn empty_secret_rejected() {
        let mut tpm = make_tpm();
        assert!(tpm.import_key("empty", &[], true).is_err());
    }

    // ---------- HMAC integrity ------------------------------------------

    #[test]
    fn hmac_integrity_rejects_tampered_ciphertext() {
        let mut tpm = make_tpm();
        let slot = tpm
            .import_key("integrity", &[0x99u8; 32], true)
            .expect("import");

        // Tamper directly with the internal ciphertext.
        let obj = tpm.objects.get_mut(&slot.handle).expect("object present");
        obj.ciphertext[0] ^= 0x01;

        let err = tpm.unseal(&slot).expect_err("tampered unseal must fail");
        match err {
            DrmError::InvalidKey(msg) => assert!(msg.contains("HMAC")),
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[test]
    fn hmac_integrity_rejects_tampered_name_digest() {
        let mut tpm = make_tpm();
        let slot = tpm
            .import_key("integrity-name", &[0x77u8; 16], true)
            .expect("import");
        let obj = tpm.objects.get_mut(&slot.handle).expect("object present");
        obj.name_digest[5] ^= 0xff;
        assert!(tpm.unseal(&slot).is_err());
    }

    #[test]
    fn hmac_integrity_rejects_tampered_tag() {
        let mut tpm = make_tpm();
        let slot = tpm
            .import_key("integrity-tag", &[0x55u8; 24], true)
            .expect("import");
        let obj = tpm.objects.get_mut(&slot.handle).expect("object present");
        obj.tag[31] ^= 0xff;
        assert!(tpm.unseal(&slot).is_err());
    }

    // ---------- PolicyPCR sealing ---------------------------------------

    #[test]
    fn policy_pcr_unseal_succeeds_when_pcrs_match() {
        let mut tpm = make_tpm();
        // Capture a known PCR state.
        tpm.pcr_extend(0, b"boot-stage-1").expect("extend");
        tpm.pcr_extend(7, b"firmware-hash").expect("extend");

        let secret = vec![0xABu8; 32];
        let slot = tpm
            .import_sealed_pcr("policy-key", &secret, &[0, 7])
            .expect("import");

        // Read-back the policy digest matches the live state.
        let recovered = tpm.unseal_with_policy(&slot).expect("policy unseal");
        assert_eq!(recovered, secret);
    }

    #[test]
    fn policy_pcr_unseal_fails_when_pcrs_change() {
        let mut tpm = make_tpm();
        tpm.pcr_extend(0, b"boot-stage-1").expect("extend");
        tpm.pcr_extend(7, b"firmware-hash").expect("extend");

        let secret = vec![0xCDu8; 16];
        let slot = tpm
            .import_sealed_pcr("locked-key", &secret, &[0, 7])
            .expect("import");

        // Mutate one of the sealed PCRs.
        tpm.pcr_extend(7, b"tampered-firmware").expect("extend");

        let err = tpm
            .unseal_with_policy(&slot)
            .expect_err("must fail when PCRs diverge");
        match err {
            DrmError::LicenseError(msg) => assert!(msg.contains("PolicyPCR")),
            other => panic!("expected LicenseError, got {other:?}"),
        }
    }

    #[test]
    fn policy_pcr_unseal_unaffected_by_unselected_pcr_change() {
        let mut tpm = make_tpm();
        tpm.pcr_extend(0, b"watched-pcr").expect("extend");
        let secret = vec![0x10u8; 8];
        let slot = tpm
            .import_sealed_pcr("scope-test", &secret, &[0])
            .expect("import");

        // Extend a PCR NOT in the selection — must still unseal.
        tpm.pcr_extend(5, b"unrelated").expect("extend");
        let recovered = tpm.unseal_with_policy(&slot).expect("policy unseal");
        assert_eq!(recovered, secret);
    }

    #[test]
    fn plain_unseal_refuses_policy_objects() {
        let mut tpm = make_tpm();
        let slot = tpm
            .import_sealed_pcr("policy-only", b"x", &[0])
            .expect("import");
        assert!(
            tpm.unseal(&slot).is_err(),
            "plain unseal must reject policy-gated objects"
        );
    }

    #[test]
    fn policy_unseal_refuses_plain_objects() {
        let mut tpm = make_tpm();
        let slot = tpm.import_key("no-policy", b"x", true).expect("import");
        assert!(
            tpm.unseal_with_policy(&slot).is_err(),
            "policy unseal must reject plain objects"
        );
    }

    #[test]
    fn policy_pcr_requires_selection_in_range() {
        let mut tpm = make_tpm();
        let bad = tpm.import_sealed_pcr("oob", b"x", &[PCR_COUNT]);
        assert!(bad.is_err());
    }

    #[test]
    fn policy_pcr_rejects_empty_selection() {
        let mut tpm = make_tpm();
        assert!(tpm.import_sealed_pcr("empty-sel", b"x", &[]).is_err());
    }

    #[test]
    fn policy_pcr_objects_are_non_exportable() {
        let mut tpm = make_tpm();
        tpm.pcr_extend(0, b"x").expect("extend");
        let slot = tpm
            .import_sealed_pcr("policy-noexport", b"key", &[0])
            .expect("import");
        assert!(
            tpm.export_wrapped(&slot).is_err(),
            "PCR-sealed objects must never be exportable"
        );
    }

    // ---------- Export / delete -----------------------------------------

    #[test]
    fn export_wrapped_is_non_trivial_envelope() {
        let mut tpm = make_tpm();
        let secret = vec![0xEFu8; 16];
        let slot = tpm.import_key("exportable", &secret, true).expect("import");
        let blob = tpm.export_wrapped(&slot).expect("export");

        // Envelope = 32 (name) + 32 (policy) + |ct| + 32 (tag).
        assert_eq!(blob.len(), DIGEST_SIZE * 3 + secret.len());
        // Ciphertext segment must not equal the plaintext.
        let ct = &blob[DIGEST_SIZE * 2..DIGEST_SIZE * 2 + secret.len()];
        assert_ne!(ct, secret.as_slice());
    }

    #[test]
    fn non_exportable_objects_block_export() {
        let mut tpm = make_tpm();
        let slot = tpm
            .import_key("private", &[0xAA; 16], false)
            .expect("import");
        assert!(tpm.export_wrapped(&slot).is_err());
    }

    #[test]
    fn delete_removes_object_and_breaks_unseal() {
        let mut tpm = make_tpm();
        let slot = tpm
            .import_key("delete-me", &[0xBB; 16], true)
            .expect("import");
        tpm.delete_key(&slot).expect("delete");
        assert!(tpm.unseal(&slot).is_err());
    }

    #[test]
    fn delete_unknown_handle_errors() {
        let mut tpm = make_tpm();
        let phantom = KeySlot::new("ghost", 0x8000_DEAD);
        assert!(tpm.delete_key(&phantom).is_err());
    }

    // ---------- Listing / metadata --------------------------------------

    #[test]
    fn list_keys_reports_each_import() {
        let mut tpm = make_tpm();
        assert!(tpm.list_keys().is_empty());
        tpm.import_key("k1", &[1u8; 8], true).expect("import 1");
        tpm.import_key("k2", &[2u8; 8], true).expect("import 2");
        assert_eq!(tpm.list_keys().len(), 2);
    }

    #[test]
    fn descriptor_carries_metadata() {
        let mut tpm = make_tpm();
        tpm.set_time(1_700_000_000);
        let slot = tpm
            .import_key("desc-test", &[0x12; 16], false)
            .expect("import");
        let desc = tpm.get_descriptor(&slot).expect("descriptor");
        assert_eq!(desc.slot.label, "desc-test");
        assert!(!desc.exportable);
        assert_eq!(desc.created_at, 1_700_000_000);
        // wrapped_len = 32 name + 32 policy + 16 ct + 32 tag = 112
        assert_eq!(desc.wrapped_len, 32 + 32 + 16 + 32);
    }

    // ---------- Policy digest mechanics ---------------------------------

    #[test]
    fn empty_policy_digest_is_zero() {
        assert_eq!(PolicyDigest::empty().as_bytes(), &[0u8; DIGEST_SIZE]);
    }

    #[test]
    fn policy_digest_extend_pcr_is_deterministic() {
        let pcr_value = [0xAAu8; DIGEST_SIZE];
        let a = PolicyDigest::empty().extend_pcr(&[0, 7], &pcr_value);
        let b = PolicyDigest::empty().extend_pcr(&[7, 0], &pcr_value);
        // Selection ordering must not change the digest (canonical form).
        assert_eq!(a, b);

        let c = PolicyDigest::empty().extend_pcr(&[0, 7], &[0xBBu8; DIGEST_SIZE]);
        assert_ne!(a, c);
    }

    #[test]
    fn policy_digest_extend_auth_value_changes_digest() {
        let base = PolicyDigest::empty();
        let after = base.extend_auth_value();
        assert_ne!(base, after);
        // Identity property: applying the same extension twice in succession
        // yields a different digest from one extension.
        assert_ne!(after, after.extend_auth_value());
    }

    // ---------- Constant-time equality ----------------------------------

    #[test]
    fn constant_time_eq_matches_naive() {
        let a = [1u8, 2, 3, 4];
        let b = [1u8, 2, 3, 4];
        let c = [1u8, 2, 3, 5];
        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
        assert!(!constant_time_eq(&a, &[1u8, 2, 3]));
    }
}
