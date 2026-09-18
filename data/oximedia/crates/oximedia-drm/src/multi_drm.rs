//! Multi-DRM packaging: single encrypted content with multiple PSSH boxes.
//!
//! When content is encrypted once using CENC (Common Encryption), it can be
//! played on any DRM system that supports the encryption scheme.  The content
//! is encrypted once, and a separate PSSH box is generated for each target
//! DRM system (Widevine, PlayReady, FairPlay, ClearKey).
//!
//! This module provides:
//! - [`MultiDrmPackager`]: builds multi-DRM protection info from a single key
//! - [`PackagingResult`]: combined init segment with all PSSH boxes
//! - Helper functions for DASH/HLS manifest snippets

use crate::pssh::{PsshBox, PsshBuilder};
use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Target DRM descriptor
// ---------------------------------------------------------------------------

/// Describes a target DRM system and its configuration for packaging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrmTarget {
    /// DRM system.
    pub system: DrmSystem,
    /// License acquisition URL.
    pub license_url: String,
    /// Extra system-specific init data (will be placed inside PSSH data field).
    pub init_data: Vec<u8>,
}

impl DrmTarget {
    /// Create a new DRM target.
    pub fn new(system: DrmSystem, license_url: impl Into<String>) -> Self {
        Self {
            system,
            license_url: license_url.into(),
            init_data: Vec::new(),
        }
    }

    /// Builder: set custom init data.
    pub fn with_init_data(mut self, data: Vec<u8>) -> Self {
        self.init_data = data;
        self
    }

    /// Return the 16-byte system ID for this target.
    pub fn system_id_bytes(&self) -> [u8; 16] {
        let uuid = self.system.system_id();
        *uuid.as_bytes()
    }
}

// ---------------------------------------------------------------------------
// Packaging result
// ---------------------------------------------------------------------------

/// Result of multi-DRM packaging.
#[derive(Debug, Clone)]
pub struct PackagingResult {
    /// The key ID used for encryption (16 bytes).
    pub key_id: Vec<u8>,
    /// The content encryption key (16 bytes, AES-128).
    pub content_key: Vec<u8>,
    /// Per-system PSSH boxes (serialized).
    pub pssh_boxes: HashMap<DrmSystem, Vec<u8>>,
    /// All PSSH boxes concatenated (for init segment insertion).
    pub combined_pssh: Vec<u8>,
    /// DASH `<ContentProtection>` XML snippets per DRM system.
    pub dash_content_protection: HashMap<DrmSystem, String>,
    /// HLS `#EXT-X-KEY` / `#EXT-X-SESSION-KEY` tags per DRM system.
    pub hls_key_tags: HashMap<DrmSystem, String>,
}

impl PackagingResult {
    /// Return the PSSH box for a specific DRM system, if present.
    pub fn pssh_for(&self, system: DrmSystem) -> Option<&[u8]> {
        self.pssh_boxes.get(&system).map(|v| v.as_slice())
    }

    /// Return the DASH ContentProtection XML for a specific DRM system.
    pub fn dash_cp_for(&self, system: DrmSystem) -> Option<&str> {
        self.dash_content_protection
            .get(&system)
            .map(|s| s.as_str())
    }

    /// Return the HLS key tag for a specific DRM system.
    pub fn hls_tag_for(&self, system: DrmSystem) -> Option<&str> {
        self.hls_key_tags.get(&system).map(|s| s.as_str())
    }

    /// Number of DRM systems in this result.
    pub fn system_count(&self) -> usize {
        self.pssh_boxes.len()
    }
}

// ---------------------------------------------------------------------------
// Multi-DRM packager
// ---------------------------------------------------------------------------

/// Builds multi-DRM protection metadata for a single piece of encrypted content.
pub struct MultiDrmPackager {
    /// Key ID (16 bytes).
    key_id: Vec<u8>,
    /// Content key (16 bytes).
    content_key: Vec<u8>,
    /// Target DRM systems.
    targets: Vec<DrmTarget>,
    /// Whether to generate PSSH v1 boxes (with KID list).
    use_pssh_v1: bool,
}

impl MultiDrmPackager {
    /// Create a new packager with the given key ID and content key.
    pub fn new(key_id: Vec<u8>, content_key: Vec<u8>) -> Result<Self> {
        if key_id.len() != 16 {
            return Err(DrmError::InvalidKey(format!(
                "key_id must be 16 bytes, got {}",
                key_id.len()
            )));
        }
        if content_key.len() != 16 {
            return Err(DrmError::InvalidKey(format!(
                "content_key must be 16 bytes, got {}",
                content_key.len()
            )));
        }
        Ok(Self {
            key_id,
            content_key,
            targets: Vec::new(),
            use_pssh_v1: true,
        })
    }

    /// Add a DRM target.
    pub fn add_target(&mut self, target: DrmTarget) {
        self.targets.push(target);
    }

    /// Builder: set whether to use PSSH v1 boxes.
    pub fn with_pssh_v1(mut self, v1: bool) -> Self {
        self.use_pssh_v1 = v1;
        self
    }

    /// Number of configured targets.
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Generate the complete multi-DRM packaging result.
    pub fn package(&self) -> Result<PackagingResult> {
        if self.targets.is_empty() {
            return Err(DrmError::ConfigError(
                "at least one DRM target is required".to_string(),
            ));
        }

        let mut pssh_boxes: HashMap<DrmSystem, Vec<u8>> = HashMap::new();
        let mut combined_pssh = Vec::new();
        let mut dash_cp: HashMap<DrmSystem, String> = HashMap::new();
        let mut hls_tags: HashMap<DrmSystem, String> = HashMap::new();

        for target in &self.targets {
            let pssh = self.build_pssh(target);
            let serialized = pssh.serialize();

            combined_pssh.extend_from_slice(&serialized);
            pssh_boxes.insert(target.system, serialized.clone());

            // Generate DASH ContentProtection element
            let cp_xml = self.build_dash_content_protection(target, &serialized);
            dash_cp.insert(target.system, cp_xml);

            // Generate HLS key tag
            let hls_tag = self.build_hls_key_tag(target);
            hls_tags.insert(target.system, hls_tag);
        }

        Ok(PackagingResult {
            key_id: self.key_id.clone(),
            content_key: self.content_key.clone(),
            pssh_boxes,
            combined_pssh,
            dash_content_protection: dash_cp,
            hls_key_tags: hls_tags,
        })
    }

    fn build_pssh(&self, target: &DrmTarget) -> PsshBox {
        let system_id = target.system_id_bytes();
        let mut builder = PsshBuilder::new()
            .set_system_id(system_id)
            .set_data(target.init_data.clone());

        if self.use_pssh_v1 {
            builder = builder.add_key_id(self.key_id.clone());
        }

        builder.build()
    }

    fn build_dash_content_protection(&self, target: &DrmTarget, pssh_bytes: &[u8]) -> String {
        let system_id = target.system.system_id();
        let key_id_hex = hex::encode(&self.key_id);
        let pssh_b64 = BASE64_STANDARD.encode(pssh_bytes);

        let scheme_uri = match target.system {
            DrmSystem::Widevine => "urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
            DrmSystem::PlayReady => "urn:uuid:9a04f079-9840-4286-ab92-e65be0885f95",
            DrmSystem::FairPlay => "urn:uuid:94ce86fb-07ff-4f43-adb8-93d2fa968ca2",
            DrmSystem::ClearKey => "urn:uuid:1077efec-c0b2-4d02-ace3-3c1e52e2fb4b",
        };

        format!(
            "<ContentProtection schemeIdUri=\"{}\" value=\"{}\">\n  \
             <cenc:pssh>{}</cenc:pssh>\n  \
             <cenc:default_KID>{}</cenc:default_KID>\n\
             </ContentProtection>",
            scheme_uri,
            system_id.as_hyphenated(),
            pssh_b64,
            key_id_hex,
        )
    }

    fn build_hls_key_tag(&self, target: &DrmTarget) -> String {
        let key_id_hex = hex::encode(&self.key_id);
        match target.system {
            DrmSystem::FairPlay => {
                format!(
                    "#EXT-X-KEY:METHOD=SAMPLE-AES,\
                     URI=\"{}\",\
                     KEYFORMAT=\"com.apple.streamingkeydelivery\",\
                     KEYFORMATVERSIONS=\"1\",\
                     KEYID=0x{}",
                    target.license_url, key_id_hex,
                )
            }
            DrmSystem::Widevine => {
                format!(
                    "#EXT-X-KEY:METHOD=SAMPLE-AES,\
                     URI=\"data:text/plain;base64,{}\",\
                     KEYFORMAT=\"urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed\",\
                     KEYFORMATVERSIONS=\"1\",\
                     KEYID=0x{}",
                    BASE64_STANDARD.encode(&self.key_id),
                    key_id_hex,
                )
            }
            DrmSystem::PlayReady => {
                format!(
                    "#EXT-X-KEY:METHOD=SAMPLE-AES,\
                     URI=\"{}\",\
                     KEYFORMAT=\"com.microsoft.playready\",\
                     KEYFORMATVERSIONS=\"1\",\
                     KEYID=0x{}",
                    target.license_url, key_id_hex,
                )
            }
            DrmSystem::ClearKey => {
                format!(
                    "#EXT-X-KEY:METHOD=SAMPLE-AES,\
                     URI=\"{}\",\
                     KEYFORMAT=\"identity\",\
                     KEYID=0x{}",
                    target.license_url, key_id_hex,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: quick multi-DRM setup
// ---------------------------------------------------------------------------

/// Create a standard multi-DRM packager with Widevine + PlayReady + ClearKey.
pub fn standard_multi_drm(
    key_id: Vec<u8>,
    content_key: Vec<u8>,
    widevine_url: &str,
    playready_url: &str,
) -> Result<MultiDrmPackager> {
    let mut packager = MultiDrmPackager::new(key_id, content_key)?;
    packager.add_target(DrmTarget::new(DrmSystem::Widevine, widevine_url));
    packager.add_target(DrmTarget::new(DrmSystem::PlayReady, playready_url));
    packager.add_target(DrmTarget::new(
        DrmSystem::ClearKey,
        "https://clearkey.example.com/license",
    ));
    Ok(packager)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pssh::{
        CLEARKEY_SYSTEM_ID, FAIRPLAY_SYSTEM_ID, PLAYREADY_SYSTEM_ID, WIDEVINE_SYSTEM_ID,
    };

    fn test_key_id() -> Vec<u8> {
        vec![
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10,
        ]
    }

    fn test_content_key() -> Vec<u8> {
        vec![
            0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE,
            0xAF, 0xB0,
        ]
    }

    #[test]
    fn test_packager_creation() {
        let packager = MultiDrmPackager::new(test_key_id(), test_content_key());
        assert!(packager.is_ok());
    }

    #[test]
    fn test_packager_invalid_key_id_length() {
        let result = MultiDrmPackager::new(vec![1, 2, 3], test_content_key());
        assert!(result.is_err());
    }

    #[test]
    fn test_packager_invalid_content_key_length() {
        let result = MultiDrmPackager::new(test_key_id(), vec![1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_packager_no_targets() {
        let packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        let result = packager.package();
        assert!(result.is_err());
    }

    #[test]
    fn test_single_target_packaging() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::Widevine,
            "https://wv.example.com",
        ));
        let result = packager.package().expect("package should succeed");
        assert_eq!(result.system_count(), 1);
        assert!(result.pssh_for(DrmSystem::Widevine).is_some());
        assert!(result.pssh_for(DrmSystem::PlayReady).is_none());
    }

    #[test]
    fn test_multi_target_packaging() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::Widevine,
            "https://wv.example.com",
        ));
        packager.add_target(DrmTarget::new(
            DrmSystem::PlayReady,
            "https://pr.example.com",
        ));
        packager.add_target(DrmTarget::new(
            DrmSystem::ClearKey,
            "https://ck.example.com",
        ));

        let result = packager.package().expect("package should succeed");
        assert_eq!(result.system_count(), 3);
        assert!(result.pssh_for(DrmSystem::Widevine).is_some());
        assert!(result.pssh_for(DrmSystem::PlayReady).is_some());
        assert!(result.pssh_for(DrmSystem::ClearKey).is_some());
    }

    #[test]
    fn test_combined_pssh_contains_all() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::Widevine,
            "https://wv.example.com",
        ));
        packager.add_target(DrmTarget::new(
            DrmSystem::PlayReady,
            "https://pr.example.com",
        ));

        let result = packager.package().expect("package should succeed");

        // Combined should be the concatenation of individual boxes
        let wv_len = result
            .pssh_for(DrmSystem::Widevine)
            .map(|b| b.len())
            .unwrap_or(0);
        let pr_len = result
            .pssh_for(DrmSystem::PlayReady)
            .map(|b| b.len())
            .unwrap_or(0);
        assert_eq!(result.combined_pssh.len(), wv_len + pr_len);
    }

    #[test]
    fn test_pssh_v1_has_key_id() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::Widevine,
            "https://wv.example.com",
        ));

        let result = packager.package().expect("package should succeed");
        let pssh_bytes = result
            .pssh_for(DrmSystem::Widevine)
            .expect("should have pssh");

        // Parse PSSH and verify key_ids
        let boxes = PsshBox::parse(pssh_bytes).expect("should parse");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].key_ids.len(), 1);
        assert_eq!(boxes[0].key_ids[0], test_key_id());
    }

    #[test]
    fn test_pssh_v0_no_key_id() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed")
            .with_pssh_v1(false);
        packager.add_target(DrmTarget::new(
            DrmSystem::Widevine,
            "https://wv.example.com",
        ));

        let result = packager.package().expect("package should succeed");
        let pssh_bytes = result
            .pssh_for(DrmSystem::Widevine)
            .expect("should have pssh");

        let boxes = PsshBox::parse(pssh_bytes).expect("should parse");
        assert_eq!(boxes.len(), 1);
        assert!(boxes[0].key_ids.is_empty());
    }

    #[test]
    fn test_dash_content_protection_generated() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::Widevine,
            "https://wv.example.com",
        ));

        let result = packager.package().expect("package should succeed");
        let cp = result
            .dash_cp_for(DrmSystem::Widevine)
            .expect("should have CP");
        assert!(cp.contains("ContentProtection"));
        assert!(cp.contains("cenc:pssh"));
        assert!(cp.contains("edef8ba9"));
    }

    #[test]
    fn test_hls_key_tag_fairplay() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::FairPlay,
            "skd://fairplay.example.com",
        ));

        let result = packager.package().expect("package should succeed");
        let tag = result
            .hls_tag_for(DrmSystem::FairPlay)
            .expect("should have tag");
        assert!(tag.contains("#EXT-X-KEY"));
        assert!(tag.contains("com.apple.streamingkeydelivery"));
        assert!(tag.contains("skd://fairplay.example.com"));
    }

    #[test]
    fn test_hls_key_tag_widevine() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::Widevine,
            "https://wv.example.com",
        ));

        let result = packager.package().expect("package should succeed");
        let tag = result
            .hls_tag_for(DrmSystem::Widevine)
            .expect("should have tag");
        assert!(tag.contains("#EXT-X-KEY"));
        assert!(tag.contains("edef8ba9"));
    }

    #[test]
    fn test_drm_target_system_id_bytes() {
        let target = DrmTarget::new(DrmSystem::Widevine, "https://example.com");
        assert_eq!(target.system_id_bytes(), WIDEVINE_SYSTEM_ID);

        let target = DrmTarget::new(DrmSystem::PlayReady, "https://example.com");
        assert_eq!(target.system_id_bytes(), PLAYREADY_SYSTEM_ID);

        let target = DrmTarget::new(DrmSystem::FairPlay, "https://example.com");
        assert_eq!(target.system_id_bytes(), FAIRPLAY_SYSTEM_ID);

        let target = DrmTarget::new(DrmSystem::ClearKey, "https://example.com");
        assert_eq!(target.system_id_bytes(), CLEARKEY_SYSTEM_ID);
    }

    #[test]
    fn test_drm_target_with_init_data() {
        let target = DrmTarget::new(DrmSystem::Widevine, "https://example.com")
            .with_init_data(vec![0x08, 0x01, 0x12, 0x10]);
        assert_eq!(target.init_data, vec![0x08, 0x01, 0x12, 0x10]);
    }

    #[test]
    fn test_init_data_appears_in_pssh() {
        let init_data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(
            DrmTarget::new(DrmSystem::Widevine, "https://wv.example.com")
                .with_init_data(init_data.clone()),
        );

        let result = packager.package().expect("package should succeed");
        let pssh_bytes = result
            .pssh_for(DrmSystem::Widevine)
            .expect("should have pssh");
        let boxes = PsshBox::parse(pssh_bytes).expect("should parse");
        assert_eq!(boxes[0].data, init_data);
    }

    #[test]
    fn test_standard_multi_drm() {
        let packager = standard_multi_drm(
            test_key_id(),
            test_content_key(),
            "https://wv.example.com",
            "https://pr.example.com",
        )
        .expect("should succeed");
        assert_eq!(packager.target_count(), 3);
        let result = packager.package().expect("package should succeed");
        assert_eq!(result.system_count(), 3);
    }

    #[test]
    fn test_packaging_result_key_fields() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::ClearKey,
            "https://ck.example.com",
        ));
        let result = packager.package().expect("package should succeed");
        assert_eq!(result.key_id, test_key_id());
        assert_eq!(result.content_key, test_content_key());
    }

    #[test]
    fn test_hls_clearkey_tag() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::ClearKey,
            "https://ck.example.com",
        ));
        let result = packager.package().expect("package should succeed");
        let tag = result
            .hls_tag_for(DrmSystem::ClearKey)
            .expect("should have tag");
        assert!(tag.contains("identity"));
    }

    #[test]
    fn test_hls_playready_tag() {
        let mut packager = MultiDrmPackager::new(test_key_id(), test_content_key())
            .expect("creation should succeed");
        packager.add_target(DrmTarget::new(
            DrmSystem::PlayReady,
            "https://pr.example.com",
        ));
        let result = packager.package().expect("package should succeed");
        let tag = result
            .hls_tag_for(DrmSystem::PlayReady)
            .expect("should have tag");
        assert!(tag.contains("com.microsoft.playready"));
    }
}
