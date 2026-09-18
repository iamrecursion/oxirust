//! DASH CENC (Common Encryption) signaling helpers for MPD manifest generation.
//!
//! Implements ISO/IEC 23009-1 (MPEG-DASH) ContentProtection descriptor generation
//! with support for Widevine, PlayReady, FairPlay, and ClearKey DRM systems.
//! Provides helpers for generating CENC signaling in MPD AdaptationSet elements.

use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;

/// CENC protection scheme for DASH signaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashCencScheme {
    /// AES-CTR mode (cenc) — most widely supported.
    Cenc,
    /// AES-CBC mode with subsample encryption (cbcs) — required for FairPlay.
    Cbcs,
}

impl DashCencScheme {
    /// Returns the four-character code for the scheme URN.
    pub fn fourcc(&self) -> &'static str {
        match self {
            Self::Cenc => "cenc",
            Self::Cbcs => "cbcs",
        }
    }

    /// Returns the URN scheme identifier as used in MPD `@schemeIdUri`.
    pub fn scheme_urn(&self) -> &'static str {
        match self {
            Self::Cenc => "urn:mpeg:dash:mp4protection:2011",
            Self::Cbcs => "urn:mpeg:dash:mp4protection:2011",
        }
    }
}

/// A single ContentProtection descriptor element for an MPD.
#[derive(Debug, Clone)]
pub struct ContentProtectionDescriptor {
    /// The DRM system.
    pub drm_system: DrmSystem,
    /// Encryption scheme.
    pub scheme: DashCencScheme,
    /// Default Key ID (UUID string format: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
    pub default_kid: Option<String>,
    /// System-specific `cenc:pssh` element content (base64-encoded PSSH bytes).
    pub pssh_base64: Option<String>,
    /// Additional DRM-system-specific XML attributes (name → value pairs).
    pub extra_attributes: Vec<(String, String)>,
}

impl ContentProtectionDescriptor {
    /// Create a new ContentProtection descriptor.
    pub fn new(drm_system: DrmSystem, scheme: DashCencScheme) -> Self {
        Self {
            drm_system,
            scheme,
            default_kid: None,
            pssh_base64: None,
            extra_attributes: Vec::new(),
        }
    }

    /// Set the default Key ID (16-byte KID → UUID string).
    pub fn with_default_kid(mut self, kid_bytes: &[u8; 16]) -> Self {
        self.default_kid = Some(bytes_to_uuid_string(kid_bytes));
        self
    }

    /// Set the PSSH payload (raw bytes; will be base64-encoded).
    pub fn with_pssh(mut self, pssh_bytes: Vec<u8>) -> Self {
        self.pssh_base64 = Some(BASE64_STANDARD.encode(&pssh_bytes));
        self
    }

    /// Add an extra XML attribute.
    pub fn with_attribute(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_attributes.push((name.into(), value.into()));
        self
    }

    /// Render as an MPD `<ContentProtection>` XML element string.
    pub fn to_xml(&self) -> String {
        let system_id = uuid_bytes_to_dash_string(self.drm_system.system_id().as_bytes());
        let mut xml = format!(
            r#"<ContentProtection schemeIdUri="urn:uuid:{}" value="{}"#,
            system_id,
            self.scheme.fourcc()
        );

        if let Some(ref kid) = self.default_kid {
            xml.push_str(&format!(r#"" cenc:default_KID="{}"#, kid));
        }

        for (k, v) in &self.extra_attributes {
            xml.push_str(&format!(r#"" {}="{}"#, k, xml_escape(v)));
        }

        xml.push('"');

        if self.pssh_base64.is_some() {
            xml.push('>');
            if let Some(ref pssh_b64) = self.pssh_base64 {
                xml.push_str(&format!(
                    r#"<cenc:pssh xmlns:cenc="urn:mpeg:cenc:2013">{}</cenc:pssh>"#,
                    pssh_b64
                ));
            }
            xml.push_str("</ContentProtection>");
        } else {
            xml.push_str("/>");
        }

        xml
    }
}

/// A full set of ContentProtection descriptors for a DASH AdaptationSet.
///
/// Typically one descriptor per DRM system plus a `mp4protection` marker.
#[derive(Debug, Clone, Default)]
pub struct CencSignaling {
    descriptors: Vec<ContentProtectionDescriptor>,
    /// Encryption scheme applied to the track.
    scheme: Option<DashCencScheme>,
    /// Default KID (16 bytes) for the `mp4protection` element.
    default_kid: Option<[u8; 16]>,
}

impl CencSignaling {
    /// Create a new, empty signaling set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the track-level encryption scheme.
    pub fn with_scheme(mut self, scheme: DashCencScheme) -> Self {
        self.scheme = Some(scheme);
        self
    }

    /// Set the default KID.
    pub fn with_default_kid(mut self, kid: [u8; 16]) -> Self {
        self.default_kid = Some(kid);
        self
    }

    /// Add a DRM-system ContentProtection descriptor.
    pub fn add_descriptor(&mut self, descriptor: ContentProtectionDescriptor) {
        self.descriptors.push(descriptor);
    }

    /// Generate the full set of `<ContentProtection>` XML elements, including
    /// the mandatory `mp4protection` marker element.
    pub fn to_xml_elements(&self) -> Vec<String> {
        let mut elements = Vec::new();

        // 1. Mandatory mp4protection element (schemeIdUri = urn:mpeg:dash:mp4protection:2011)
        if let (Some(scheme), Some(kid)) = (self.scheme, self.default_kid) {
            let kid_str = bytes_to_uuid_string(&kid);
            elements.push(format!(
                r#"<ContentProtection schemeIdUri="{}" value="{}" cenc:default_KID="{}" xmlns:cenc="urn:mpeg:cenc:2013"/>"#,
                scheme.scheme_urn(),
                scheme.fourcc(),
                kid_str,
            ));
        }

        // 2. DRM-system-specific descriptors
        for desc in &self.descriptors {
            elements.push(desc.to_xml());
        }

        elements
    }

    /// Number of DRM-system descriptors (not counting the mp4protection marker).
    pub fn descriptor_count(&self) -> usize {
        self.descriptors.len()
    }
}

// ---------------------------------------------------------------------------
// Builder helpers
// ---------------------------------------------------------------------------

/// Build a Widevine ContentProtection descriptor.
pub fn widevine_descriptor(
    scheme: DashCencScheme,
    kid_bytes: &[u8; 16],
    pssh_bytes: Option<Vec<u8>>,
) -> ContentProtectionDescriptor {
    let mut desc =
        ContentProtectionDescriptor::new(DrmSystem::Widevine, scheme).with_default_kid(kid_bytes);
    if let Some(pssh) = pssh_bytes {
        desc = desc.with_pssh(pssh);
    }
    desc
}

/// Build a PlayReady ContentProtection descriptor.
///
/// `pro_base64` is the PlayReady Object (PRO) encoded as base64.
pub fn playready_descriptor(
    scheme: DashCencScheme,
    kid_bytes: &[u8; 16],
    pro_base64: Option<String>,
    pssh_bytes: Option<Vec<u8>>,
) -> ContentProtectionDescriptor {
    let mut desc =
        ContentProtectionDescriptor::new(DrmSystem::PlayReady, scheme).with_default_kid(kid_bytes);
    if let Some(pro) = pro_base64 {
        desc = desc.with_attribute("mspr:pro", pro);
    }
    if let Some(pssh) = pssh_bytes {
        desc = desc.with_pssh(pssh);
    }
    desc
}

/// Build a ClearKey ContentProtection descriptor (W3C).
pub fn clearkey_descriptor(
    scheme: DashCencScheme,
    kid_bytes: &[u8; 16],
    license_url: Option<String>,
) -> ContentProtectionDescriptor {
    let mut desc =
        ContentProtectionDescriptor::new(DrmSystem::ClearKey, scheme).with_default_kid(kid_bytes);
    if let Some(url) = license_url {
        desc = desc.with_attribute("clearkey:Laurl", url);
    }
    desc
}

/// Build a complete CencSignaling for multi-DRM with Widevine + PlayReady.
pub fn multi_drm_signaling(
    scheme: DashCencScheme,
    kid_bytes: [u8; 16],
    widevine_pssh: Option<Vec<u8>>,
    playready_pssh: Option<Vec<u8>>,
    playready_pro: Option<String>,
) -> CencSignaling {
    let mut signaling = CencSignaling::new()
        .with_scheme(scheme)
        .with_default_kid(kid_bytes);

    signaling.add_descriptor(widevine_descriptor(scheme, &kid_bytes, widevine_pssh));
    signaling.add_descriptor(playready_descriptor(
        scheme,
        &kid_bytes,
        playready_pro,
        playready_pssh,
    ));

    signaling
}

// ---------------------------------------------------------------------------
// DASH MPD ContentProtection validator
// ---------------------------------------------------------------------------

/// Validate that a set of ContentProtection descriptors is well-formed for DASH CENC.
///
/// Returns a list of validation errors (empty if valid).
pub fn validate_cenc_signaling(signaling: &CencSignaling) -> Vec<String> {
    let mut errors = Vec::new();

    if signaling.scheme.is_none() {
        errors.push("CencSignaling: no encryption scheme set".to_string());
    }

    if signaling.default_kid.is_none() {
        errors.push("CencSignaling: no default KID set".to_string());
    }

    if signaling.descriptors.is_empty() {
        errors.push("CencSignaling: no DRM-system descriptors".to_string());
    }

    for desc in &signaling.descriptors {
        if desc.default_kid.is_none() {
            errors.push(format!(
                "Descriptor for {:?}: missing default_KID",
                desc.drm_system
            ));
        }
    }

    errors
}

/// Parse a `cenc:default_KID` UUID string into 16 bytes.
///
/// Accepts the canonical `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` format.
pub fn parse_kid_string(kid_str: &str) -> Result<[u8; 16]> {
    // Strip dashes and decode hex
    let hex_str: String = kid_str.chars().filter(|c| *c != '-').collect();
    if hex_str.len() != 32 {
        return Err(DrmError::InvalidKey(format!(
            "KID string must be 32 hex chars (got {}): '{}'",
            hex_str.len(),
            kid_str
        )));
    }
    let mut out = [0u8; 16];
    for i in 0..16 {
        let byte_str = &hex_str[i * 2..i * 2 + 2];
        out[i] = u8::from_str_radix(byte_str, 16).map_err(|_| {
            DrmError::InvalidKey(format!("Invalid hex in KID string: '{}'", kid_str))
        })?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert 16 raw bytes to a UUID string (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx).
fn bytes_to_uuid_string(bytes: &[u8; 16]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

/// Convert UUID bytes to DASH system ID string (32 hex chars, no dashes, uppercase-compatible).
fn uuid_bytes_to_dash_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            acc.push_str(&format!("{:02x}", b));
            acc
        })
        .chars()
        .enumerate()
        .fold(String::new(), |mut s, (i, c)| {
            if i == 8 || i == 12 || i == 16 || i == 20 {
                s.push('-');
            }
            s.push(c);
            s
        })
}

/// Minimally escape XML attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KID: [u8; 16] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10,
    ];

    #[test]
    fn test_bytes_to_uuid_string() {
        let s = bytes_to_uuid_string(&TEST_KID);
        assert_eq!(s.len(), 36);
        assert_eq!(s.chars().filter(|c| *c == '-').count(), 4);
    }

    #[test]
    fn test_parse_kid_string_roundtrip() {
        let s = bytes_to_uuid_string(&TEST_KID);
        let parsed = parse_kid_string(&s).expect("parse_kid_string should succeed");
        assert_eq!(parsed, TEST_KID);
    }

    #[test]
    fn test_parse_kid_invalid() {
        assert!(parse_kid_string("not-a-uuid").is_err());
        assert!(parse_kid_string("gggggggg-0000-0000-0000-000000000000").is_err());
    }

    #[test]
    fn test_content_protection_descriptor_selfclose() {
        let desc = ContentProtectionDescriptor::new(DrmSystem::Widevine, DashCencScheme::Cenc)
            .with_default_kid(&TEST_KID);
        let xml = desc.to_xml();
        assert!(xml.starts_with("<ContentProtection"));
        assert!(xml.contains("cenc:default_KID="));
        assert!(xml.ends_with("/>"));
    }

    #[test]
    fn test_content_protection_descriptor_with_pssh() {
        let pssh = vec![0xABu8; 32];
        let desc = ContentProtectionDescriptor::new(DrmSystem::Widevine, DashCencScheme::Cenc)
            .with_default_kid(&TEST_KID)
            .with_pssh(pssh);
        let xml = desc.to_xml();
        assert!(xml.contains("<cenc:pssh"));
        assert!(xml.contains("</ContentProtection>"));
    }

    #[test]
    fn test_cenc_signaling_xml_elements() {
        let mut sig = CencSignaling::new()
            .with_scheme(DashCencScheme::Cenc)
            .with_default_kid(TEST_KID);
        sig.add_descriptor(widevine_descriptor(DashCencScheme::Cenc, &TEST_KID, None));
        sig.add_descriptor(playready_descriptor(
            DashCencScheme::Cenc,
            &TEST_KID,
            None,
            None,
        ));

        let elements = sig.to_xml_elements();
        // mp4protection + widevine + playready = 3
        assert_eq!(elements.len(), 3);
        assert!(elements[0].contains("mp4protection"));
        assert!(
            elements[1].contains(DrmSystem::Widevine.system_id().to_string().as_str())
                || elements[1].contains("edef8ba9")
        );
    }

    #[test]
    fn test_validate_cenc_signaling_valid() {
        let mut sig = CencSignaling::new()
            .with_scheme(DashCencScheme::Cenc)
            .with_default_kid(TEST_KID);
        sig.add_descriptor(widevine_descriptor(DashCencScheme::Cenc, &TEST_KID, None));
        let errors = validate_cenc_signaling(&sig);
        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_validate_cenc_signaling_missing_scheme() {
        let sig = CencSignaling::new().with_default_kid(TEST_KID);
        let errors = validate_cenc_signaling(&sig);
        assert!(errors.iter().any(|e| e.contains("scheme")));
    }

    #[test]
    fn test_validate_cenc_signaling_missing_kid() {
        let sig = CencSignaling::new().with_scheme(DashCencScheme::Cenc);
        let errors = validate_cenc_signaling(&sig);
        assert!(errors.iter().any(|e| e.contains("KID")));
    }

    #[test]
    fn test_validate_cenc_signaling_no_descriptors() {
        let sig = CencSignaling::new()
            .with_scheme(DashCencScheme::Cenc)
            .with_default_kid(TEST_KID);
        let errors = validate_cenc_signaling(&sig);
        assert!(errors.iter().any(|e| e.contains("descriptor")));
    }

    #[test]
    fn test_multi_drm_signaling() {
        let sig = multi_drm_signaling(DashCencScheme::Cenc, TEST_KID, None, None, None);
        assert_eq!(sig.descriptor_count(), 2);
        let elements = sig.to_xml_elements();
        assert_eq!(elements.len(), 3); // mp4protection + widevine + playready
    }

    #[test]
    fn test_clearkey_descriptor() {
        let desc = clearkey_descriptor(
            DashCencScheme::Cenc,
            &TEST_KID,
            Some("https://example.com/clearkey".to_string()),
        );
        let xml = desc.to_xml();
        assert!(xml.contains("clearkey:Laurl"));
        assert!(xml.contains("https://example.com/clearkey"));
    }

    #[test]
    fn test_playready_descriptor_with_pro() {
        let desc = playready_descriptor(
            DashCencScheme::Cenc,
            &TEST_KID,
            Some("AQIDBA==".to_string()),
            None,
        );
        let xml = desc.to_xml();
        assert!(xml.contains("mspr:pro"));
    }

    #[test]
    fn test_scheme_fourcc() {
        assert_eq!(DashCencScheme::Cenc.fourcc(), "cenc");
        assert_eq!(DashCencScheme::Cbcs.fourcc(), "cbcs");
    }

    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("a&b"), "a&amp;b");
        assert_eq!(xml_escape("\"q\""), "&quot;q&quot;");
        assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
    }

    #[test]
    fn test_uuid_bytes_to_dash_string() {
        let wv_bytes: [u8; 16] = [
            0xed, 0xef, 0x8b, 0xa9, 0x79, 0xd6, 0x4a, 0xce, 0xa3, 0xc8, 0x27, 0xdc, 0xd5, 0x1d,
            0x21, 0xed,
        ];
        let s = uuid_bytes_to_dash_string(&wv_bytes);
        assert_eq!(s, "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed");
    }
}
