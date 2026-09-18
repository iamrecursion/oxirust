// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! DRM signaling support for the packager.
//!
//! Provides types and helpers for embedding DRM protection information
//! (PSSH boxes, key IDs, licence URLs) into HLS and DASH manifests.

// ---------------------------------------------------------------------------
// DrmSystem
// ---------------------------------------------------------------------------

/// Well-known DRM systems supported by this packager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrmSystem {
    /// Google Widevine (used on Android, Chrome, …).
    Widevine,
    /// Microsoft `PlayReady` (used on Windows, Xbox, …).
    PlayReady,
    /// Apple `FairPlay` Streaming (used on iOS, macOS, tvOS, …).
    FairPlay,
    /// W3C `ClearKey` (open, key-rotation based, no licence server needed).
    ClearKey,
}

impl DrmSystem {
    /// Return the DASH/CENC System ID UUID string for this DRM system.
    ///
    /// UUIDs are the standard identifiers as defined in the DASH-IF IOP spec.
    #[must_use]
    pub fn system_id(&self) -> &'static str {
        match self {
            Self::Widevine => "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
            Self::PlayReady => "9a04f079-9840-4286-ab92-e65be0885f95",
            Self::FairPlay => "94ce86fb-07ff-4f43-adb8-93d2fa968ca2",
            Self::ClearKey => "e2719d58-a985-b3c9-781a-b030af78d30e",
        }
    }

    /// Human-readable display name.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Widevine => "Widevine",
            Self::PlayReady => "PlayReady",
            Self::FairPlay => "FairPlay",
            Self::ClearKey => "ClearKey",
        }
    }
}

// ---------------------------------------------------------------------------
// DrmProtectionInfo
// ---------------------------------------------------------------------------

/// DRM protection metadata for a single DRM system applied to a track.
#[derive(Debug, Clone)]
pub struct DrmProtectionInfo {
    /// The DRM system this entry applies to.
    pub system: DrmSystem,
    /// Binary PSSH box data (may be empty for `FairPlay` which uses EXT-X-KEY).
    pub pssh_box: Vec<u8>,
    /// 16-byte key identifier.
    pub key_id: Vec<u8>,
    /// Optional licence acquisition URL.
    pub la_url: Option<String>,
}

impl DrmProtectionInfo {
    /// Create a new protection info entry.
    #[must_use]
    pub fn new(system: DrmSystem, key_id: Vec<u8>) -> Self {
        Self {
            system,
            pssh_box: Vec::new(),
            key_id,
            la_url: None,
        }
    }

    /// Set the PSSH box data.
    #[must_use]
    pub fn with_pssh(mut self, pssh: Vec<u8>) -> Self {
        self.pssh_box = pssh;
        self
    }

    /// Set the licence acquisition URL.
    #[must_use]
    pub fn with_la_url(mut self, url: impl Into<String>) -> Self {
        self.la_url = Some(url.into());
        self
    }
}

// ---------------------------------------------------------------------------
// ContentKey
// ---------------------------------------------------------------------------

/// A symmetric content encryption key with its identifier and IV.
#[derive(Debug, Clone)]
pub struct ContentKey {
    /// 16-byte key identifier.
    pub key_id: Vec<u8>,
    /// 16-byte AES key.
    pub key: Vec<u8>,
    /// 16-byte initialisation vector.
    pub iv: Vec<u8>,
}

// ---------------------------------------------------------------------------
// generate_clear_key_pssh
// ---------------------------------------------------------------------------

/// Build a minimal `ClearKey` PSSH box payload for the given key IDs.
///
/// The `ClearKey` PSSH schema is defined in the EME specification.  This
/// implementation generates a simple JSON-encoded payload (as used in the
/// "keyids" `ClearKey` sub-type) wrapped in a 4-byte little-endian length
/// prefix for easy identification in tests.
///
/// In a real CMAF/ISOBMFF file the full box header would be prepended; that
/// is outside the scope of this pure-Rust prototype.
#[must_use]
pub fn generate_clear_key_pssh(key_ids: &[Vec<u8>]) -> Vec<u8> {
    // Encode key IDs as hex strings
    let ids: Vec<String> = key_ids.iter().map(|k| hex_encode(k)).collect();
    let json = format!(
        r#"{{"kids":[{}]}}"#,
        ids.iter()
            .map(|id| format!("\"{id}\""))
            .collect::<Vec<_>>()
            .join(",")
    );

    let payload = json.into_bytes();
    // Prepend 4-byte LE length
    let mut out = Vec::with_capacity(4 + payload.len());
    let len = payload.len() as u32;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

// ---------------------------------------------------------------------------
// drm_info_for_manifest
// ---------------------------------------------------------------------------

/// Serialise a list of DRM protection entries to a JSON-like string suitable
/// for embedding in a manifest or a `ContentProtection` element.
///
/// Output format (one object per line, array wrapped in `[…]`):
/// ```text
/// [
///   {"system":"Widevine","system_id":"edef8ba9-...","key_id":"aabbcc...", ...},
///   ...
/// ]
/// ```
#[must_use]
pub fn drm_info_for_manifest(protections: &[DrmProtectionInfo]) -> String {
    let mut items: Vec<String> = Vec::with_capacity(protections.len());
    for p in protections {
        let key_id_hex = hex_encode(&p.key_id);
        let pssh_hex = hex_encode(&p.pssh_box);
        let la = p
            .la_url
            .as_deref()
            .map(|u| format!(r#","la_url":"{u}""#))
            .unwrap_or_default();
        items.push(format!(
            r#"  {{"system":"{name}","system_id":"{sid}","key_id":"{kid}","pssh":"{pssh}"{la}}}"#,
            name = p.system.display_name(),
            sid = p.system.system_id(),
            kid = key_id_hex,
            pssh = pssh_hex,
        ));
    }
    format!("[\n{}\n]", items.join(",\n"))
}

// ---------------------------------------------------------------------------
// generate_content_key
// ---------------------------------------------------------------------------

/// Generate a pseudo-random `ContentKey` using a simple, deterministic LCG.
///
/// **Not cryptographically secure** – suitable for testing and prototyping
/// only. Do not use this in a production DRM pipeline.
#[must_use]
pub fn generate_content_key() -> ContentKey {
    // LCG parameters (Knuth MMIX)
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;

    // Use a fixed seed derived from a compile-time constant so results are
    // stable across runs (making unit tests reliable).
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;

    let mut next_byte = || -> u8 {
        state = state.wrapping_mul(A).wrapping_add(C);
        (state >> 56) as u8
    };

    let key_id: Vec<u8> = (0..16).map(|_| next_byte()).collect();
    let key: Vec<u8> = (0..16).map(|_| next_byte()).collect();
    let iv: Vec<u8> = (0..16).map(|_| next_byte()).collect();

    ContentKey { key_id, key, iv }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Simple hex encoder (avoids pulling in the `hex` crate in tests).
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drm_system_ids_unique() {
        let ids = [
            DrmSystem::Widevine.system_id(),
            DrmSystem::PlayReady.system_id(),
            DrmSystem::FairPlay.system_id(),
            DrmSystem::ClearKey.system_id(),
        ];
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 4, "All system IDs should be unique");
    }

    #[test]
    fn test_drm_system_widevine_id() {
        assert_eq!(
            DrmSystem::Widevine.system_id(),
            "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"
        );
    }

    #[test]
    fn test_drm_system_display_names() {
        assert_eq!(DrmSystem::Widevine.display_name(), "Widevine");
        assert_eq!(DrmSystem::PlayReady.display_name(), "PlayReady");
        assert_eq!(DrmSystem::FairPlay.display_name(), "FairPlay");
        assert_eq!(DrmSystem::ClearKey.display_name(), "ClearKey");
    }

    #[test]
    fn test_drm_protection_info_new() {
        let key_id = vec![0u8; 16];
        let info = DrmProtectionInfo::new(DrmSystem::Widevine, key_id.clone());
        assert_eq!(info.system, DrmSystem::Widevine);
        assert!(info.pssh_box.is_empty());
        assert!(info.la_url.is_none());
    }

    #[test]
    fn test_drm_protection_info_with_la_url() {
        let info = DrmProtectionInfo::new(DrmSystem::PlayReady, vec![0u8; 16])
            .with_la_url("https://license.example.com/pr");
        assert_eq!(
            info.la_url.as_deref(),
            Some("https://license.example.com/pr")
        );
    }

    #[test]
    fn test_generate_clear_key_pssh_has_length_prefix() {
        let key_ids = vec![vec![1u8; 16], vec![2u8; 16]];
        let pssh = generate_clear_key_pssh(&key_ids);
        assert!(
            pssh.len() >= 4,
            "should have at least the 4-byte length prefix"
        );
        let declared_len =
            u32::from_le_bytes(pssh[..4].try_into().expect("should succeed in test")) as usize;
        assert_eq!(declared_len + 4, pssh.len());
    }

    #[test]
    fn test_generate_clear_key_pssh_contains_key_ids() {
        let key_ids = vec![vec![0xABu8; 16]];
        let pssh = generate_clear_key_pssh(&key_ids);
        let payload = std::str::from_utf8(&pssh[4..]).expect("should succeed in test");
        // All bytes are 0xAB → hex is "ab" repeated 16 times
        let expected_hex = "ab".repeat(16);
        assert!(payload.contains(&expected_hex));
    }

    #[test]
    fn test_drm_info_for_manifest_empty() {
        let result = drm_info_for_manifest(&[]);
        assert!(result.starts_with('['));
        assert!(result.ends_with(']'));
    }

    #[test]
    fn test_drm_info_for_manifest_contains_system_name() {
        let info = DrmProtectionInfo::new(DrmSystem::Widevine, vec![0u8; 16]);
        let result = drm_info_for_manifest(&[info]);
        assert!(result.contains("Widevine"));
        assert!(result.contains("edef8ba9"));
    }

    #[test]
    fn test_drm_info_for_manifest_contains_la_url() {
        let info = DrmProtectionInfo::new(DrmSystem::ClearKey, vec![0xFFu8; 16])
            .with_la_url("https://lic.example.com");
        let result = drm_info_for_manifest(&[info]);
        assert!(result.contains("https://lic.example.com"));
    }

    #[test]
    fn test_generate_content_key_lengths() {
        let ck = generate_content_key();
        assert_eq!(ck.key_id.len(), 16);
        assert_eq!(ck.key.len(), 16);
        assert_eq!(ck.iv.len(), 16);
    }

    #[test]
    fn test_generate_content_key_deterministic() {
        // Two calls to the pure function should produce the same result
        let ck1 = generate_content_key();
        let ck2 = generate_content_key();
        assert_eq!(ck1.key_id, ck2.key_id);
        assert_eq!(ck1.key, ck2.key);
        assert_eq!(ck1.iv, ck2.iv);
    }
}
