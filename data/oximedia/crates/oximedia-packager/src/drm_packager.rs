#![allow(dead_code)]
//! DRM integration for the packager output.
//!
//! Wires PSSH (Protection System Specific Header) boxes from the `pssh`
//! module into packager init segments and manifests, supporting Widevine,
//! PlayReady, and FairPlay DRM systems.
//!
//! # Workflow
//!
//! 1. Configure a [`DrmPackagerConfig`] with desired DRM systems and key info.
//! 2. Create a [`DrmPackager`] from the config.
//! 3. Call [`DrmPackager::generate_pssh_boxes`] to produce binary PSSH data.
//! 4. Call [`DrmPackager::inject_into_init_segment`] to embed PSSH in an init segment.
//! 5. Call [`DrmPackager::content_protection_xml`] for DASH MPD `ContentProtection` elements.
//! 6. Call [`DrmPackager::hls_ext_x_key_tags`] for HLS `EXT-X-KEY` / `EXT-X-SESSION-KEY` tags.

use crate::error::{PackagerError, PackagerResult};
use crate::pssh::{
    build_cenc_pssh, build_fairplay_pssh, build_playready_pssh, build_widevine_pssh, DrmSystem,
    PsshBox,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

// ---------------------------------------------------------------------------
// DrmSystemConfig
// ---------------------------------------------------------------------------

/// Configuration for a single DRM system.
#[derive(Debug, Clone)]
pub struct DrmSystemConfig {
    /// Which DRM system to configure.
    pub system: DrmSystem,
    /// 16-byte content key identifier.
    pub key_id: [u8; 16],
    /// Optional content ID (used by Widevine).
    pub content_id: Vec<u8>,
    /// Optional licence acquisition URL.
    pub la_url: Option<String>,
    /// Optional key server URI (used by FairPlay).
    pub key_server_uri: Option<String>,
}

impl DrmSystemConfig {
    /// Create a Widevine DRM configuration.
    #[must_use]
    pub fn widevine(key_id: [u8; 16], content_id: Vec<u8>) -> Self {
        Self {
            system: DrmSystem::Widevine,
            key_id,
            content_id,
            la_url: None,
            key_server_uri: None,
        }
    }

    /// Create a PlayReady DRM configuration.
    #[must_use]
    pub fn playready(key_id: [u8; 16]) -> Self {
        Self {
            system: DrmSystem::PlayReady,
            key_id,
            content_id: Vec::new(),
            la_url: None,
            key_server_uri: None,
        }
    }

    /// Create a FairPlay DRM configuration.
    #[must_use]
    pub fn fairplay(key_id: [u8; 16], key_server_uri: impl Into<String>) -> Self {
        Self {
            system: DrmSystem::FairPlay,
            key_id,
            content_id: Vec::new(),
            la_url: None,
            key_server_uri: Some(key_server_uri.into()),
        }
    }

    /// Set the licence acquisition URL.
    #[must_use]
    pub fn with_la_url(mut self, url: impl Into<String>) -> Self {
        self.la_url = Some(url.into());
        self
    }
}

// ---------------------------------------------------------------------------
// DrmPackagerConfig
// ---------------------------------------------------------------------------

/// Configuration for multi-DRM packaging.
#[derive(Debug, Clone)]
pub struct DrmPackagerConfig {
    /// List of DRM systems to include.
    pub systems: Vec<DrmSystemConfig>,
    /// Whether to include PSSH boxes in the init segment's moov.
    pub embed_pssh_in_init: bool,
    /// Whether to include base64 PSSH in DASH MPD.
    pub include_pssh_in_mpd: bool,
}

impl Default for DrmPackagerConfig {
    fn default() -> Self {
        Self {
            systems: Vec::new(),
            embed_pssh_in_init: true,
            include_pssh_in_mpd: true,
        }
    }
}

impl DrmPackagerConfig {
    /// Create a new DRM packager config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a DRM system configuration.
    #[must_use]
    pub fn with_system(mut self, system: DrmSystemConfig) -> Self {
        self.systems.push(system);
        self
    }

    /// Check if any DRM systems are configured.
    #[must_use]
    pub fn has_drm(&self) -> bool {
        !self.systems.is_empty()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> PackagerResult<()> {
        for sys in &self.systems {
            if sys.system == DrmSystem::FairPlay && sys.key_server_uri.is_none() {
                return Err(PackagerError::DrmFailed(
                    "FairPlay requires a key server URI".to_string(),
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GeneratedPssh
// ---------------------------------------------------------------------------

/// A generated PSSH box with its associated DRM system.
#[derive(Debug, Clone)]
pub struct GeneratedPssh {
    /// The DRM system this PSSH belongs to.
    pub system: DrmSystem,
    /// The fully constructed PSSH box.
    pub pssh_box: PsshBox,
    /// Binary encoding of the PSSH box.
    pub encoded: Vec<u8>,
    /// Base64 encoding of the PSSH box (for DASH MPD).
    pub base64: String,
}

// ---------------------------------------------------------------------------
// DrmPackager
// ---------------------------------------------------------------------------

/// Integrates DRM protection into packager output.
pub struct DrmPackager {
    config: DrmPackagerConfig,
    /// Generated PSSH boxes (populated after `generate_pssh_boxes`).
    generated: Vec<GeneratedPssh>,
}

impl DrmPackager {
    /// Create a new DRM packager.
    pub fn new(config: DrmPackagerConfig) -> PackagerResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            generated: Vec::new(),
        })
    }

    /// Generate PSSH boxes for all configured DRM systems.
    pub fn generate_pssh_boxes(&mut self) -> PackagerResult<&[GeneratedPssh]> {
        self.generated.clear();

        for sys_config in &self.config.systems {
            let pssh_box = match sys_config.system {
                DrmSystem::Widevine => {
                    build_widevine_pssh(&sys_config.key_id, &sys_config.content_id)
                }
                DrmSystem::PlayReady => build_playready_pssh(&sys_config.key_id),
                DrmSystem::FairPlay => {
                    let uri = sys_config
                        .key_server_uri
                        .as_deref()
                        .unwrap_or("skd://default");
                    build_fairplay_pssh(&sys_config.key_id, uri)
                }
                DrmSystem::Marlin | DrmSystem::CommonEncryption => {
                    build_cenc_pssh(&[sys_config.key_id])
                }
            };

            let encoded = pssh_box.encode();
            let base64 = BASE64.encode(&encoded);

            self.generated.push(GeneratedPssh {
                system: sys_config.system,
                pssh_box,
                encoded,
                base64,
            });
        }

        Ok(&self.generated)
    }

    /// Get the generated PSSH boxes (must call `generate_pssh_boxes` first).
    #[must_use]
    pub fn pssh_boxes(&self) -> &[GeneratedPssh] {
        &self.generated
    }

    /// Inject PSSH boxes into an existing init segment.
    ///
    /// The init segment must contain a `moov` box. PSSH boxes are appended
    /// inside the `moov` box (after the existing children). The size of the
    /// `moov` box is updated to account for the injected PSSH data.
    pub fn inject_into_init_segment(&self, init_segment: &[u8]) -> PackagerResult<Vec<u8>> {
        if self.generated.is_empty() {
            return Ok(init_segment.to_vec());
        }

        // Find the moov box
        let moov_offset = find_box_offset(init_segment, b"moov").ok_or_else(|| {
            PackagerError::DrmFailed("No moov box found in init segment".to_string())
        })?;

        let moov_size = u32::from_be_bytes(
            init_segment[moov_offset..moov_offset + 4]
                .try_into()
                .map_err(|_| PackagerError::DrmFailed("Failed to read moov size".to_string()))?,
        ) as usize;

        let moov_end = moov_offset + moov_size;

        // Concatenate all PSSH encoded data
        let mut pssh_data = Vec::new();
        for gen in &self.generated {
            pssh_data.extend_from_slice(&gen.encoded);
        }

        // Build new output: everything before moov end + PSSH data + everything after moov
        let new_moov_size = moov_size + pssh_data.len();
        let mut out = Vec::with_capacity(init_segment.len() + pssh_data.len());

        // Copy everything before moov
        out.extend_from_slice(&init_segment[..moov_offset]);

        // Write new moov size
        out.extend_from_slice(&(new_moov_size as u32).to_be_bytes());

        // Copy moov content (skip old size, keep fourcc + children)
        out.extend_from_slice(&init_segment[moov_offset + 4..moov_end]);

        // Append PSSH boxes inside moov
        out.extend_from_slice(&pssh_data);

        // Copy everything after moov
        if moov_end < init_segment.len() {
            out.extend_from_slice(&init_segment[moov_end..]);
        }

        Ok(out)
    }

    /// Generate DASH MPD `<ContentProtection>` XML elements for all DRM systems.
    #[must_use]
    pub fn content_protection_xml(&self) -> String {
        let mut xml = String::new();

        // Add CENC default KID content protection
        if let Some(first) = self.config.systems.first() {
            let kid_hex = hex::encode(first.key_id);
            let kid_uuid = format!(
                "{}-{}-{}-{}-{}",
                &kid_hex[0..8],
                &kid_hex[8..12],
                &kid_hex[12..16],
                &kid_hex[16..20],
                &kid_hex[20..32],
            );
            xml.push_str(&format!(
                "      <ContentProtection schemeIdUri=\"urn:mpeg:dash:mp4protection:2011\" \
                 value=\"cenc\" cenc:default_KID=\"{kid_uuid}\"/>\n"
            ));
        }

        // Add per-system content protection elements
        for gen in &self.generated {
            let system_uuid = gen.pssh_box.system_id_uuid();
            xml.push_str(&format!(
                "      <ContentProtection schemeIdUri=\"urn:uuid:{system_uuid}\""
            ));

            // Find LA URL if configured
            let la_url = self
                .config
                .systems
                .iter()
                .find(|s| s.system == gen.system)
                .and_then(|s| s.la_url.as_deref());

            if let Some(url) = la_url {
                xml.push_str(&format!(" value=\"{url}\""));
            }

            if self.config.include_pssh_in_mpd {
                xml.push_str(">\n");
                xml.push_str(&format!("        <cenc:pssh>{}</cenc:pssh>\n", gen.base64));
                xml.push_str("      </ContentProtection>\n");
            } else {
                xml.push_str("/>\n");
            }
        }

        xml
    }

    /// Generate HLS `EXT-X-KEY` tags for encrypted playlists.
    #[must_use]
    pub fn hls_ext_x_key_tags(&self) -> String {
        let mut tags = String::new();

        for sys_config in &self.config.systems {
            match sys_config.system {
                DrmSystem::FairPlay => {
                    let uri = sys_config
                        .key_server_uri
                        .as_deref()
                        .unwrap_or("skd://default");
                    let kid_hex = hex::encode(sys_config.key_id);
                    tags.push_str(&format!(
                        "#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"{uri}\",\
                         KEYFORMAT=\"com.apple.streamingkeydelivery\",\
                         KEYFORMATVERSIONS=\"1\",\
                         KEYID=0x{kid_hex}\n"
                    ));
                }
                DrmSystem::Widevine => {
                    // Widevine in HLS uses SAMPLE-AES-CTR with a PSSH-containing URI
                    if let Some(gen) = self
                        .generated
                        .iter()
                        .find(|g| g.system == DrmSystem::Widevine)
                    {
                        let kid_hex = hex::encode(sys_config.key_id);
                        let la_url = sys_config
                            .la_url
                            .as_deref()
                            .unwrap_or("https://proxy.uat.widevine.com/proxy");
                        tags.push_str(&format!(
                            "#EXT-X-KEY:METHOD=SAMPLE-AES-CTR,URI=\"data:text/plain;base64,{}\",\
                             KEYFORMAT=\"urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed\",\
                             KEYFORMATVERSIONS=\"1\",\
                             KEYID=0x{kid_hex}\n",
                            gen.base64
                        ));
                        let _ = la_url; // la_url used in actual key server interaction
                    }
                }
                DrmSystem::PlayReady => {
                    if let Some(gen) = self
                        .generated
                        .iter()
                        .find(|g| g.system == DrmSystem::PlayReady)
                    {
                        let kid_hex = hex::encode(sys_config.key_id);
                        tags.push_str(&format!(
                            "#EXT-X-KEY:METHOD=SAMPLE-AES-CTR,URI=\"data:text/plain;base64,{}\",\
                             KEYFORMAT=\"com.microsoft.playready\",\
                             KEYFORMATVERSIONS=\"1\",\
                             KEYID=0x{kid_hex}\n",
                            gen.base64
                        ));
                    }
                }
                _ => {}
            }
        }

        tags
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &DrmPackagerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helper: find box offset
// ---------------------------------------------------------------------------

/// Find the byte offset of an ISOBMFF box by fourcc within `data`.
fn find_box_offset(data: &[u8], fourcc: &[u8; 4]) -> Option<usize> {
    let mut i = 0;
    while i + 8 <= data.len() {
        let size = u32::from_be_bytes(data[i..i + 4].try_into().ok()?) as usize;
        if size < 8 {
            break;
        }
        if &data[i + 4..i + 8] == fourcc {
            return Some(i);
        }
        // Search inside container boxes
        if i + size <= data.len() {
            if let Some(inner) = find_box_offset(&data[i + 8..i + size], fourcc) {
                return Some(i + 8 + inner);
            }
        }
        i += size;
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isobmff_writer::write_init_segment;
    use crate::pssh;

    fn test_key_id() -> [u8; 16] {
        [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10,
        ]
    }

    // --- DrmSystemConfig ---

    #[test]
    fn test_widevine_config() {
        let cfg = DrmSystemConfig::widevine(test_key_id(), b"content-1".to_vec());
        assert_eq!(cfg.system, DrmSystem::Widevine);
        assert_eq!(cfg.content_id, b"content-1");
    }

    #[test]
    fn test_playready_config() {
        let cfg = DrmSystemConfig::playready(test_key_id());
        assert_eq!(cfg.system, DrmSystem::PlayReady);
    }

    #[test]
    fn test_fairplay_config() {
        let cfg = DrmSystemConfig::fairplay(test_key_id(), "skd://key.example.com");
        assert_eq!(cfg.system, DrmSystem::FairPlay);
        assert_eq!(cfg.key_server_uri.as_deref(), Some("skd://key.example.com"));
    }

    #[test]
    fn test_config_with_la_url() {
        let cfg = DrmSystemConfig::widevine(test_key_id(), Vec::new())
            .with_la_url("https://lic.example.com");
        assert_eq!(cfg.la_url.as_deref(), Some("https://lic.example.com"));
    }

    // --- DrmPackagerConfig ---

    #[test]
    fn test_drm_packager_config_empty() {
        let cfg = DrmPackagerConfig::new();
        assert!(!cfg.has_drm());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_drm_packager_config_with_systems() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()))
            .with_system(DrmSystemConfig::playready(test_key_id()));
        assert!(cfg.has_drm());
        assert_eq!(cfg.systems.len(), 2);
    }

    #[test]
    fn test_drm_packager_config_fairplay_requires_uri() {
        let cfg = DrmPackagerConfig::new().with_system(DrmSystemConfig {
            system: DrmSystem::FairPlay,
            key_id: test_key_id(),
            content_id: Vec::new(),
            la_url: None,
            key_server_uri: None, // missing
        });
        assert!(cfg.validate().is_err());
    }

    // --- DrmPackager generation ---

    #[test]
    fn test_generate_pssh_widevine() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), b"test".to_vec()));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        let boxes = packager.generate_pssh_boxes().expect("should succeed");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].system, DrmSystem::Widevine);
        assert!(!boxes[0].encoded.is_empty());
        assert!(!boxes[0].base64.is_empty());
    }

    #[test]
    fn test_generate_pssh_playready() {
        let cfg = DrmPackagerConfig::new().with_system(DrmSystemConfig::playready(test_key_id()));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        let boxes = packager.generate_pssh_boxes().expect("should succeed");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].system, DrmSystem::PlayReady);
    }

    #[test]
    fn test_generate_pssh_fairplay() {
        let cfg = DrmPackagerConfig::new().with_system(DrmSystemConfig::fairplay(
            test_key_id(),
            "skd://test.com/key",
        ));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        let boxes = packager.generate_pssh_boxes().expect("should succeed");
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].system, DrmSystem::FairPlay);
    }

    #[test]
    fn test_generate_multi_drm() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()))
            .with_system(DrmSystemConfig::playready(test_key_id()))
            .with_system(DrmSystemConfig::fairplay(test_key_id(), "skd://test.com"));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        let boxes = packager.generate_pssh_boxes().expect("should succeed");
        assert_eq!(boxes.len(), 3);
    }

    #[test]
    fn test_pssh_roundtrip_decode() {
        let cfg = DrmPackagerConfig::new().with_system(DrmSystemConfig::widevine(
            test_key_id(),
            b"content".to_vec(),
        ));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        let boxes = packager.generate_pssh_boxes().expect("should succeed");

        let decoded = PsshBox::decode(&boxes[0].encoded).expect("should decode");
        assert_eq!(decoded.system_id, pssh::WIDEVINE_SYSTEM_ID);
    }

    // --- Init segment injection ---

    #[test]
    fn test_inject_into_init_segment() {
        let init_cfg = crate::isobmff_writer::InitConfig::new(1920, 1080, 90_000, *b"av01");
        let init = write_init_segment(&init_cfg);

        let drm_cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        let mut packager = DrmPackager::new(drm_cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let injected = packager
            .inject_into_init_segment(&init)
            .expect("should succeed");

        // Injected segment should be larger
        assert!(injected.len() > init.len());

        // Should still start with ftyp
        assert_eq!(&injected[4..8], b"ftyp");

        // Should contain the PSSH box
        let found_pssh = find_box_offset(&injected, b"pssh");
        assert!(found_pssh.is_some());
    }

    #[test]
    fn test_inject_no_moov_error() {
        let drm_cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        let mut packager = DrmPackager::new(drm_cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        // Empty data has no moov
        let result = packager.inject_into_init_segment(&[0u8; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn test_inject_empty_pssh_passthrough() {
        let init_cfg = crate::isobmff_writer::InitConfig::new(1920, 1080, 90_000, *b"av01");
        let init = write_init_segment(&init_cfg);

        let drm_cfg = DrmPackagerConfig::new(); // no systems
        let packager = DrmPackager::new(drm_cfg).expect("should succeed");

        let result = packager
            .inject_into_init_segment(&init)
            .expect("should succeed");
        assert_eq!(result, init);
    }

    #[test]
    fn test_inject_multi_drm_pssh() {
        let init_cfg = crate::isobmff_writer::InitConfig::new(1280, 720, 90_000, *b"vp09");
        let init = write_init_segment(&init_cfg);

        let drm_cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()))
            .with_system(DrmSystemConfig::playready(test_key_id()));
        let mut packager = DrmPackager::new(drm_cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let injected = packager
            .inject_into_init_segment(&init)
            .expect("should succeed");

        // Should have 2 PSSH boxes (they are children of moov, so find them
        // by scanning inside the moov box)
        let moov_off = find_box_offset(&injected, b"moov").expect("moov should exist");
        let moov_size = u32::from_be_bytes(
            injected[moov_off..moov_off + 4]
                .try_into()
                .expect("4 bytes"),
        ) as usize;
        let moov_content = &injected[moov_off + 8..moov_off + moov_size];
        let pssh_results = PsshBox::scan_all(moov_content);
        let pssh_ok: Vec<_> = pssh_results.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(pssh_ok.len(), 2);
    }

    // --- ContentProtection XML ---

    #[test]
    fn test_content_protection_xml_widevine() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let xml = packager.content_protection_xml();
        assert!(xml.contains("urn:mpeg:dash:mp4protection:2011"));
        assert!(xml.contains("cenc:default_KID"));
        assert!(xml.contains("urn:uuid:"));
        assert!(xml.contains("cenc:pssh"));
    }

    #[test]
    fn test_content_protection_xml_with_la_url() {
        let cfg = DrmPackagerConfig::new().with_system(
            DrmSystemConfig::widevine(test_key_id(), Vec::new())
                .with_la_url("https://lic.example.com"),
        );
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let xml = packager.content_protection_xml();
        assert!(xml.contains("https://lic.example.com"));
    }

    #[test]
    fn test_content_protection_xml_no_pssh_in_mpd() {
        let mut cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        cfg.include_pssh_in_mpd = false;
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let xml = packager.content_protection_xml();
        assert!(!xml.contains("cenc:pssh"));
    }

    // --- HLS EXT-X-KEY tags ---

    #[test]
    fn test_hls_ext_x_key_fairplay() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::fairplay(test_key_id(), "skd://test.com"));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let tags = packager.hls_ext_x_key_tags();
        assert!(tags.contains("#EXT-X-KEY:METHOD=SAMPLE-AES"));
        assert!(tags.contains("com.apple.streamingkeydelivery"));
        assert!(tags.contains("skd://test.com"));
        assert!(tags.contains("KEYID=0x"));
    }

    #[test]
    fn test_hls_ext_x_key_widevine() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let tags = packager.hls_ext_x_key_tags();
        assert!(tags.contains("#EXT-X-KEY:METHOD=SAMPLE-AES-CTR"));
        assert!(tags.contains("urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"));
    }

    #[test]
    fn test_hls_ext_x_key_playready() {
        let cfg = DrmPackagerConfig::new().with_system(DrmSystemConfig::playready(test_key_id()));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let tags = packager.hls_ext_x_key_tags();
        assert!(tags.contains("com.microsoft.playready"));
    }

    #[test]
    fn test_hls_ext_x_key_multi_drm() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::fairplay(test_key_id(), "skd://test.com"))
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let tags = packager.hls_ext_x_key_tags();
        // Should have both FairPlay and Widevine tags
        assert!(tags.contains("com.apple.streamingkeydelivery"));
        assert!(tags.contains("urn:uuid:edef8ba9"));
    }

    #[test]
    fn test_hls_ext_x_key_empty_no_drm() {
        let cfg = DrmPackagerConfig::new();
        let mut packager = DrmPackager::new(cfg).expect("should succeed");
        packager.generate_pssh_boxes().expect("should succeed");

        let tags = packager.hls_ext_x_key_tags();
        assert!(tags.is_empty());
    }

    // --- Config accessor ---

    #[test]
    fn test_drm_packager_config_accessor() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        let packager = DrmPackager::new(cfg).expect("should succeed");
        assert!(packager.config().has_drm());
    }

    #[test]
    fn test_pssh_boxes_before_generate() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        let packager = DrmPackager::new(cfg).expect("should succeed");
        assert!(packager.pssh_boxes().is_empty());
    }

    #[test]
    fn test_generate_regenerates() {
        let cfg = DrmPackagerConfig::new()
            .with_system(DrmSystemConfig::widevine(test_key_id(), Vec::new()));
        let mut packager = DrmPackager::new(cfg).expect("should succeed");

        packager.generate_pssh_boxes().expect("should succeed");
        assert_eq!(packager.pssh_boxes().len(), 1);

        // Regenerate should clear and recreate
        packager.generate_pssh_boxes().expect("should succeed");
        assert_eq!(packager.pssh_boxes().len(), 1);
    }
}
