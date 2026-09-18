//! Software-emulated Apple Secure Enclave (Pure Rust).
//!
//! This module models the strongest behavioural invariant of Apple's Secure
//! Enclave: **content keys are non-exportable**. A real Secure Enclave will
//! reject any `SecItemCopyMatching` request for the key material itself; the
//! only operations it offers are sign/verify and seal/unseal performed
//! *inside* the coprocessor. We mirror that boundary by:
//!
//! 1. Forcing every imported key to be non-exportable. An attempt to import
//!    with `exportable = true` fails with [`DrmError::InvalidKey`].
//! 2. Refusing every [`SwSecureEnclave::export_wrapped`] call regardless of
//!    the descriptor flag — a real device cannot ever release the key bytes,
//!    so neither do we.
//! 3. Reporting [`crate::hw_key_store::HwBackend::SoftwareSecureEnclave`]
//!    (never `SecureEnclave`) so consumers gating on a hardware root of
//!    trust correctly refuse the emulator for high-assurance flows.
//!
//! Unsealing is still supported — it represents the in-device decrypt API
//! (`SecKeyCreateDecryptedData`) and is what content playback needs. The
//! HMAC-SHA-256 integrity wrapping prevents an attacker who scrapes the
//! emulator's storage from modifying the sealed material undetected.
//!
//! A future `cfg(target_os = "macos")` implementation may replace
//! [`SwSecureEnclave`] with calls into CryptoKit / Security.framework.

use crate::hw_key_store::{sha256_mini, KeyDescriptor, KeySlot};
use crate::key_derivation::{DerivedKey, KeyDerivationConfig, KeyDerivationMethod, KeyDeriver};
use crate::{DrmError, Result};
use std::collections::BTreeMap;

const DIGEST_SIZE: usize = 32;

// ---------------------------------------------------------------------------
// Stored item
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct EnclaveItem {
    /// SHA-256 of (label || handle); analogous to the Secure Enclave's
    /// kSecAttrApplicationTag-derived identifier.
    name_digest: [u8; DIGEST_SIZE],
    /// XOR-keystream ciphertext under the per-item HMAC key.
    ciphertext: Vec<u8>,
    /// HMAC-SHA-256 tag over `name_digest ‖ ciphertext`.
    tag: [u8; DIGEST_SIZE],
    /// Public descriptor surfaced via [`crate::hw_key_store::HwKeyStore`].
    descriptor: KeyDescriptor,
}

impl EnclaveItem {
    fn integrity_input(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(DIGEST_SIZE + self.ciphertext.len());
        buf.extend_from_slice(&self.name_digest);
        buf.extend_from_slice(&self.ciphertext);
        buf
    }
}

// ---------------------------------------------------------------------------
// SwSecureEnclave — software-emulated Apple Secure Enclave
// ---------------------------------------------------------------------------

/// Software emulator of the Apple Secure Enclave key store.
///
/// See module-level docs for the modelled invariants.
pub struct SwSecureEnclave {
    /// Master seed feeding the per-item HMAC-SHA-256 KDF.
    master_key: [u8; 32],
    /// Items keyed by 32-bit handle.
    items: BTreeMap<u32, EnclaveItem>,
    /// Label → handle index.
    label_to_handle: BTreeMap<String, u32>,
    /// Next handle to assign. Secure Enclave handles are opaque — we use a
    /// monotonic 32-bit counter starting above 0.
    next_handle: u32,
    /// Monotonic time source (for `created_at` on descriptors).
    current_time: u64,
}

impl SwSecureEnclave {
    /// Create a new emulator with the given master seed.
    #[must_use]
    pub fn new(master_key: [u8; 32]) -> Self {
        Self {
            master_key,
            items: BTreeMap::new(),
            label_to_handle: BTreeMap::new(),
            next_handle: 1,
            current_time: 0,
        }
    }

    /// Import a key into the enclave.
    ///
    /// `exportable` **must** be `false`. Any caller asking for exportable
    /// Secure Enclave keys is making a mistake: real hardware cannot do
    /// that, and the emulator refuses to pretend otherwise.
    pub fn import_key(
        &mut self,
        label: &str,
        key_bytes: &[u8],
        exportable: bool,
    ) -> Result<KeySlot> {
        if exportable {
            return Err(DrmError::InvalidKey(
                "Secure Enclave keys cannot be exportable (kSecAttrTokenIDSecureEnclave)"
                    .to_string(),
            ));
        }
        if key_bytes.is_empty() {
            return Err(DrmError::InvalidKey(
                "Key material must not be empty".to_string(),
            ));
        }

        let handle = self.next_handle;
        self.next_handle = self.next_handle.checked_add(1).ok_or_else(|| {
            DrmError::InvalidKey("SwSecureEnclave: handle space exhausted".to_string())
        })?;

        let name_digest = compute_name_digest(label, handle);
        let key = derive_item_key(&self.master_key, &name_digest)?;
        let ciphertext = xor_keystream(&key, &name_digest, key_bytes);

        let mut item = EnclaveItem {
            name_digest,
            ciphertext,
            tag: [0u8; DIGEST_SIZE],
            descriptor: KeyDescriptor {
                slot: KeySlot::new(label, handle),
                wrapped_len: 0,
                exportable: false,
                created_at: self.current_time,
                expires_at: None,
            },
        };
        item.tag = hmac_sha256(&key, &item.integrity_input());
        item.descriptor.wrapped_len = DIGEST_SIZE + item.ciphertext.len() + DIGEST_SIZE;

        let slot = item.descriptor.slot.clone();
        self.label_to_handle.insert(label.to_string(), handle);
        self.items.insert(handle, item);
        Ok(slot)
    }

    /// **Always** fails: Secure Enclave keys are non-exportable by hardware
    /// mandate. The emulator mirrors that boundary.
    pub fn export_wrapped(&self, _slot: &KeySlot) -> Result<Vec<u8>> {
        Err(DrmError::InvalidKey(
            "Secure Enclave keys cannot be exported".to_string(),
        ))
    }

    /// In-device unseal (analogous to `SecKeyCreateDecryptedData`).
    pub fn unseal(&self, slot: &KeySlot) -> Result<Vec<u8>> {
        let item = self.lookup(slot)?;
        let key = derive_item_key(&self.master_key, &item.name_digest)?;
        let expected = hmac_sha256(&key, &item.integrity_input());
        if !constant_time_eq(&expected, &item.tag) {
            return Err(DrmError::InvalidKey(
                "SwSecureEnclave: HMAC integrity check failed (item tampered)".to_string(),
            ));
        }
        Ok(xor_keystream(&key, &item.name_digest, &item.ciphertext))
    }

    /// Remove an item (analogous to `SecItemDelete`).
    pub fn delete_key(&mut self, slot: &KeySlot) -> Result<()> {
        self.label_to_handle.retain(|_, &mut h| h != slot.handle);
        match self.items.remove(&slot.handle) {
            Some(_) => Ok(()),
            None => Err(DrmError::InvalidKey(format!(
                "Item handle {} not found",
                slot.handle
            ))),
        }
    }

    /// List all descriptors.
    pub fn list_keys(&self) -> Vec<KeyDescriptor> {
        self.items
            .values()
            .map(|item| item.descriptor.clone())
            .collect()
    }

    /// Look up a descriptor by slot.
    pub fn get_descriptor(&self, slot: &KeySlot) -> Option<KeyDescriptor> {
        self.items.get(&slot.handle).map(|i| i.descriptor.clone())
    }

    fn lookup(&self, slot: &KeySlot) -> Result<&EnclaveItem> {
        self.items
            .get(&slot.handle)
            .ok_or_else(|| DrmError::InvalidKey(format!("Item handle {} not found", slot.handle)))
    }

    /// Test-only time setter.
    #[cfg(test)]
    pub fn set_time(&mut self, ts: u64) {
        self.current_time = ts;
    }
}

// ---------------------------------------------------------------------------
// Helpers (mirror sw_tpm's primitives — kept inline so each emulator stays
// self-contained and we don't introduce a sw_common module just for two
// users).
// ---------------------------------------------------------------------------

fn compute_name_digest(label: &str, handle: u32) -> [u8; DIGEST_SIZE] {
    let mut buf = Vec::with_capacity(label.len() + 4);
    buf.extend_from_slice(label.as_bytes());
    buf.extend_from_slice(&handle.to_be_bytes());
    sha256_mini(&buf)
}

fn derive_item_key(master_key: &[u8; 32], name_digest: &[u8; DIGEST_SIZE]) -> Result<[u8; 32]> {
    let cfg = KeyDerivationConfig::new(KeyDerivationMethod::Sp800108Ctr, 32)
        .with_info(name_digest.to_vec());
    let derived: DerivedKey = KeyDeriver::new("swse/item")
        .derive(master_key, &cfg)
        .map_err(|e| DrmError::InvalidKey(format!("SwSecureEnclave: KDF failed: {e}")))?;
    if derived.key_bytes.len() != 32 {
        return Err(DrmError::InvalidKey(format!(
            "SwSecureEnclave: KDF returned {} bytes, expected 32",
            derived.key_bytes.len()
        )));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&derived.key_bytes);
    Ok(out)
}

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MASTER: [u8; 32] = [
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0,
        0x01, 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0x81, 0x91, 0xA1, 0xB1, 0xC1, 0xD1, 0xE1,
        0xF1, 0x02,
    ];

    fn make_enclave() -> SwSecureEnclave {
        SwSecureEnclave::new(MASTER)
    }

    // ---------- Non-exportability boundary -------------------------------

    #[test]
    fn import_with_exportable_true_is_rejected() {
        let mut enc = make_enclave();
        let result = enc.import_key("leaky", &[0x11u8; 16], true);
        assert!(result.is_err(), "exportable=true must be rejected");
        match result {
            Err(DrmError::InvalidKey(msg)) => assert!(msg.contains("Secure Enclave")),
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[test]
    fn export_wrapped_always_fails_even_after_import() {
        let mut enc = make_enclave();
        let slot = enc
            .import_key("locked", &[0x22u8; 16], false)
            .expect("import");
        let err = enc.export_wrapped(&slot).expect_err("export must fail");
        match err {
            DrmError::InvalidKey(msg) => {
                assert!(
                    msg.contains("cannot be exported"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[test]
    fn export_wrapped_fails_for_unknown_handle_too() {
        // The Secure Enclave should not leak whether a handle exists via
        // export. Both known and unknown handles return the same boundary
        // error.
        let enc = make_enclave();
        let phantom = KeySlot::new("ghost", 0xDEAD_BEEF);
        assert!(enc.export_wrapped(&phantom).is_err());
    }

    // ---------- Import → unseal round-trip ------------------------------

    #[test]
    fn import_and_unseal_roundtrip() {
        let mut enc = make_enclave();
        let secret = vec![0xCAu8; 32];
        let slot = enc
            .import_key("se-content-key", &secret, false)
            .expect("import");
        let recovered = enc.unseal(&slot).expect("unseal");
        assert_eq!(recovered, secret);
    }

    #[test]
    fn unseal_handles_long_secrets() {
        let mut enc = make_enclave();
        let secret: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        let slot = enc.import_key("big", &secret, false).expect("import");
        let recovered = enc.unseal(&slot).expect("unseal");
        assert_eq!(recovered, secret);
    }

    #[test]
    fn empty_secret_rejected() {
        let mut enc = make_enclave();
        assert!(enc.import_key("empty", &[], false).is_err());
    }

    #[test]
    fn unseal_unknown_handle_errors() {
        let enc = make_enclave();
        let phantom = KeySlot::new("ghost", 0xDEAD_BEEF);
        assert!(enc.unseal(&phantom).is_err());
    }

    // ---------- HMAC integrity ------------------------------------------

    #[test]
    fn hmac_integrity_rejects_tampered_ciphertext() {
        let mut enc = make_enclave();
        let slot = enc
            .import_key("integrity", &[0x99u8; 32], false)
            .expect("import");
        let item = enc.items.get_mut(&slot.handle).expect("item present");
        item.ciphertext[0] ^= 0x01;

        let err = enc.unseal(&slot).expect_err("tampered unseal must fail");
        match err {
            DrmError::InvalidKey(msg) => assert!(msg.contains("HMAC")),
            other => panic!("expected InvalidKey, got {other:?}"),
        }
    }

    #[test]
    fn hmac_integrity_rejects_tampered_name_digest() {
        let mut enc = make_enclave();
        let slot = enc
            .import_key("integrity-name", &[0x77u8; 16], false)
            .expect("import");
        let item = enc.items.get_mut(&slot.handle).expect("item present");
        item.name_digest[2] ^= 0xff;
        assert!(enc.unseal(&slot).is_err());
    }

    #[test]
    fn hmac_integrity_rejects_tampered_tag() {
        let mut enc = make_enclave();
        let slot = enc
            .import_key("integrity-tag", &[0x55u8; 24], false)
            .expect("import");
        let item = enc.items.get_mut(&slot.handle).expect("item present");
        item.tag[0] ^= 0xff;
        assert!(enc.unseal(&slot).is_err());
    }

    // ---------- Listing / delete ----------------------------------------

    #[test]
    fn list_keys_reflects_imports() {
        let mut enc = make_enclave();
        assert!(enc.list_keys().is_empty());
        enc.import_key("a", &[1; 8], false).expect("import a");
        enc.import_key("b", &[2; 8], false).expect("import b");
        assert_eq!(enc.list_keys().len(), 2);
    }

    #[test]
    fn delete_removes_item() {
        let mut enc = make_enclave();
        let slot = enc
            .import_key("to-delete", &[0xBB; 16], false)
            .expect("import");
        enc.delete_key(&slot).expect("delete");
        assert!(enc.unseal(&slot).is_err());
    }

    #[test]
    fn delete_unknown_handle_errors() {
        let mut enc = make_enclave();
        let phantom = KeySlot::new("ghost", 0xDEAD_BEEF);
        assert!(enc.delete_key(&phantom).is_err());
    }

    #[test]
    fn descriptor_records_non_exportable_flag() {
        let mut enc = make_enclave();
        enc.set_time(1_700_000_000);
        let slot = enc.import_key("desc", &[0x12; 16], false).expect("import");
        let desc = enc.get_descriptor(&slot).expect("descriptor");
        assert!(!desc.exportable, "Secure Enclave items are non-exportable");
        assert_eq!(desc.slot.label, "desc");
        assert_eq!(desc.created_at, 1_700_000_000);
    }

    // ---------- Item independence ---------------------------------------

    #[test]
    fn items_are_independent() {
        let mut enc = make_enclave();
        let k1 = vec![0x11u8; 16];
        let k2 = vec![0x22u8; 16];
        let s1 = enc.import_key("k1", &k1, false).expect("import 1");
        let s2 = enc.import_key("k2", &k2, false).expect("import 2");
        assert_eq!(enc.unseal(&s1).expect("unseal 1"), k1);
        assert_eq!(enc.unseal(&s2).expect("unseal 2"), k2);
        assert_ne!(s1.handle, s2.handle);
    }
}
