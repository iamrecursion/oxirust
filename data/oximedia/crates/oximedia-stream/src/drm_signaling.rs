//! DRM signaling for adaptive streaming manifests.
//!
//! Generates `#EXT-X-KEY` tags for HLS and `<ContentProtection>` elements for
//! DASH MPD for Widevine, FairPlay, and PlayReady protected streams.

use crate::StreamError;

// ─── DRM System ───────────────────────────────────────────────────────────────

/// Supported DRM systems.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DrmSystem {
    /// Google Widevine (CENC/CBCS).
    Widevine,
    /// Apple FairPlay Streaming.
    FairPlay,
    /// Microsoft PlayReady.
    PlayReady,
    /// Custom DRM identified by a UUID string.
    Custom(String),
}

impl DrmSystem {
    /// Return the well-known system UUID string.
    pub fn system_id(&self) -> &str {
        match self {
            Self::Widevine => "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
            Self::FairPlay => "94ce86fb-07ff-4f43-adb8-93d2fa968ca2",
            Self::PlayReady => "9a04f079-9840-4286-ab92-e65be0885f95",
            Self::Custom(id) => id.as_str(),
        }
    }

    /// Return the HLS `KEYFORMAT` attribute string.
    pub fn hls_keyformat(&self) -> &str {
        match self {
            Self::Widevine => "urn:uuid:edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
            Self::FairPlay => "com.apple.streamingkeydelivery",
            Self::PlayReady => "com.microsoft.playready",
            Self::Custom(id) => id.as_str(),
        }
    }

    /// Return the HLS encryption `METHOD` value.
    ///
    /// FairPlay uses `SAMPLE-AES`; others use `SAMPLE-AES-CTR`.
    pub fn hls_method(&self) -> &str {
        match self {
            Self::FairPlay => "SAMPLE-AES",
            _ => "SAMPLE-AES-CTR",
        }
    }
}

// ─── DRM Signal ───────────────────────────────────────────────────────────────

/// A DRM signal associating a key with a DRM system.
#[derive(Debug, Clone)]
pub struct DrmSignal {
    /// DRM system this signal targets.
    pub system: DrmSystem,
    /// Content key identifier (128-bit / 16 bytes).
    pub key_id: [u8; 16],
    /// Optional PSSH (Protection System Specific Header) box bytes.
    pub pssh_box: Option<Vec<u8>>,
    /// Optional licence acquisition URL.
    pub la_url: Option<String>,
    /// Optional IV (initialisation vector) for the content key.
    pub iv: Option<[u8; 16]>,
}

impl DrmSignal {
    /// Create a minimal DRM signal.
    pub fn new(system: DrmSystem, key_id: [u8; 16]) -> Self {
        Self {
            system,
            key_id,
            pssh_box: None,
            la_url: None,
            iv: None,
        }
    }

    /// Return the key ID as a lowercase hex string.
    pub fn key_id_hex(&self) -> String {
        self.key_id.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Return the key ID formatted as a UUID string.
    pub fn key_id_uuid(&self) -> String {
        let k = &self.key_id;
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            k[0], k[1], k[2], k[3],
            k[4], k[5],
            k[6], k[7],
            k[8], k[9],
            k[10], k[11], k[12], k[13], k[14], k[15],
        )
    }

    /// Return the PSSH box as a base64-encoded string, or `None`.
    pub fn pssh_base64(&self) -> Option<String> {
        self.pssh_box.as_ref().map(|b| encode_base64(b))
    }

    /// Render the HLS `#EXT-X-KEY` line for this signal.
    pub fn to_hls_key_tag(&self) -> String {
        let method = self.system.hls_method();
        let keyformat = self.system.hls_keyformat();

        let uri = self.la_url.as_deref().unwrap_or("skd://key").to_string();

        let mut tag = format!(
            "#EXT-X-KEY:METHOD={},URI=\"{}\",KEYFORMAT=\"{}\"",
            method, uri, keyformat
        );

        // Key ID
        tag.push_str(&format!(",KEYID=0x{}", self.key_id_hex()));

        // IV
        if let Some(iv) = &self.iv {
            let iv_hex: String = iv.iter().map(|b| format!("{:02x}", b)).collect();
            tag.push_str(&format!(",IV=0x{}", iv_hex));
        }

        tag
    }

    /// Render the DASH `<ContentProtection>` XML element for this signal.
    ///
    /// Follows the DASH-IF IOP guidelines for CENC/CBCS signalling.
    pub fn to_dash_content_protection(&self) -> String {
        let system_id = self.system.system_id();
        let key_id_uuid = self.key_id_uuid();

        let mut xml = format!(
            "<ContentProtection schemeIdUri=\"urn:uuid:{}\" value=\"{}\">\n",
            system_id,
            self.system_display_name(),
        );

        xml.push_str(&format!(
            "  <cenc:default_KID xmlns:cenc=\"urn:mpeg:cenc:2013\">{}</cenc:default_KID>\n",
            key_id_uuid,
        ));

        if let Some(pssh_b64) = self.pssh_base64() {
            xml.push_str(&format!(
                "  <cenc:pssh xmlns:cenc=\"urn:mpeg:cenc:2013\">{}</cenc:pssh>\n",
                pssh_b64,
            ));
        }

        if let Some(la) = &self.la_url {
            xml.push_str(&format!(
                "  <ms:laurl xmlns:ms=\"urn:microsoft:playready\">{}</ms:laurl>\n",
                la
            ));
        }

        xml.push_str("</ContentProtection>");
        xml
    }

    fn system_display_name(&self) -> &str {
        match &self.system {
            DrmSystem::Widevine => "WIDEVINE",
            DrmSystem::FairPlay => "FAIRPLAY",
            DrmSystem::PlayReady => "PLAYREADY",
            DrmSystem::Custom(_) => "CUSTOM",
        }
    }
}

// ─── DRM Manifest Builder ─────────────────────────────────────────────────────

/// Assembles DRM signalling across multiple systems for a single presentation.
#[derive(Debug, Default)]
pub struct DrmManifestBuilder {
    signals: Vec<DrmSignal>,
}

impl DrmManifestBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a DRM signal.
    ///
    /// Returns an error if a signal for the same DRM system is already present.
    pub fn add_signal(&mut self, signal: DrmSignal) -> Result<(), StreamError> {
        let dup = self
            .signals
            .iter()
            .any(|s| std::mem::discriminant(&s.system) == std::mem::discriminant(&signal.system));
        if dup {
            return Err(StreamError::Generic(format!(
                "DRM signal for system '{}' already added",
                signal.system.system_id()
            )));
        }
        self.signals.push(signal);
        Ok(())
    }

    /// Return all signals.
    pub fn signals(&self) -> &[DrmSignal] {
        &self.signals
    }

    /// Generate all HLS `#EXT-X-KEY` tags, one per signal, joined by newlines.
    pub fn to_hls_key_tags(&self) -> String {
        self.signals
            .iter()
            .map(|s| s.to_hls_key_tag())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Generate all DASH `<ContentProtection>` elements, joined by newlines.
    pub fn to_dash_content_protections(&self) -> String {
        self.signals
            .iter()
            .map(|s| s.to_dash_content_protection())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return `true` if any signal includes a PSSH box.
    pub fn has_pssh(&self) -> bool {
        self.signals.iter().any(|s| s.pssh_box.is_some())
    }

    /// Return the number of registered signals.
    pub fn signal_count(&self) -> usize {
        self.signals.len()
    }
}

// ─── Base64 helper (no external dep) ─────────────────────────────────────────

static B64_CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn encode_base64(data: &[u8]) -> String {
    let mut out = Vec::with_capacity((data.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 2 < data.len() {
        let b0 = data[i] as usize;
        let b1 = data[i + 1] as usize;
        let b2 = data[i + 2] as usize;
        out.push(B64_CHARS[(b0 >> 2) & 0x3F]);
        out.push(B64_CHARS[((b0 << 4) | (b1 >> 4)) & 0x3F]);
        out.push(B64_CHARS[((b1 << 2) | (b2 >> 6)) & 0x3F]);
        out.push(B64_CHARS[b2 & 0x3F]);
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let b0 = data[i] as usize;
        out.push(B64_CHARS[(b0 >> 2) & 0x3F]);
        out.push(B64_CHARS[(b0 << 4) & 0x3F]);
        out.push(b'=');
        out.push(b'=');
    } else if rem == 2 {
        let b0 = data[i] as usize;
        let b1 = data[i + 1] as usize;
        out.push(B64_CHARS[(b0 >> 2) & 0x3F]);
        out.push(B64_CHARS[((b0 << 4) | (b1 >> 4)) & 0x3F]);
        out.push(B64_CHARS[(b1 << 2) & 0x3F]);
        out.push(b'=');
    }
    String::from_utf8(out).unwrap_or_default()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_key() -> [u8; 16] {
        [0u8; 16]
    }

    fn test_key() -> [u8; 16] {
        [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E,
            0x0F, 0x10,
        ]
    }

    #[test]
    fn test_drm_system_ids() {
        assert_eq!(
            DrmSystem::Widevine.system_id(),
            "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"
        );
        assert_eq!(
            DrmSystem::FairPlay.system_id(),
            "94ce86fb-07ff-4f43-adb8-93d2fa968ca2"
        );
        assert_eq!(
            DrmSystem::PlayReady.system_id(),
            "9a04f079-9840-4286-ab92-e65be0885f95"
        );
    }

    #[test]
    fn test_key_id_hex() {
        let sig = DrmSignal::new(DrmSystem::Widevine, test_key());
        assert_eq!(sig.key_id_hex(), "0102030405060708090a0b0c0d0e0f10");
    }

    #[test]
    fn test_key_id_uuid_format() {
        let sig = DrmSignal::new(DrmSystem::Widevine, test_key());
        let uuid = sig.key_id_uuid();
        // Should have 4 hyphens
        assert_eq!(uuid.matches('-').count(), 4);
    }

    #[test]
    fn test_hls_key_tag_method_widevine() {
        let sig = DrmSignal::new(DrmSystem::Widevine, zero_key());
        let tag = sig.to_hls_key_tag();
        assert!(tag.starts_with("#EXT-X-KEY:METHOD=SAMPLE-AES-CTR"));
    }

    #[test]
    fn test_hls_key_tag_method_fairplay() {
        let sig = DrmSignal::new(DrmSystem::FairPlay, zero_key());
        let tag = sig.to_hls_key_tag();
        assert!(tag.contains("METHOD=SAMPLE-AES"));
    }

    #[test]
    fn test_hls_key_tag_contains_keyid() {
        let sig = DrmSignal::new(DrmSystem::Widevine, test_key());
        let tag = sig.to_hls_key_tag();
        assert!(tag.contains("KEYID=0x0102030405060708090a0b0c0d0e0f10"));
    }

    #[test]
    fn test_dash_content_protection_contains_system_id() {
        let sig = DrmSignal::new(DrmSystem::Widevine, zero_key());
        let xml = sig.to_dash_content_protection();
        assert!(xml.contains("edef8ba9-79d6-4ace-a3c8-27dcd51d21ed"));
    }

    #[test]
    fn test_dash_content_protection_contains_kid() {
        let sig = DrmSignal::new(DrmSystem::PlayReady, zero_key());
        let xml = sig.to_dash_content_protection();
        assert!(xml.contains("<cenc:default_KID"));
    }

    #[test]
    fn test_pssh_base64_encoding() {
        let mut sig = DrmSignal::new(DrmSystem::Widevine, zero_key());
        sig.pssh_box = Some(vec![0x00, 0x01, 0x02]);
        let b64 = sig.pssh_base64().expect("should have base64");
        assert!(!b64.is_empty());
        // "AAEC" is the base64 for [0x00, 0x01, 0x02]
        assert_eq!(b64, "AAEC");
    }

    #[test]
    fn test_manifest_builder_add_and_count() {
        let mut builder = DrmManifestBuilder::new();
        builder
            .add_signal(DrmSignal::new(DrmSystem::Widevine, zero_key()))
            .expect("add widevine");
        builder
            .add_signal(DrmSignal::new(DrmSystem::FairPlay, zero_key()))
            .expect("add fairplay");
        assert_eq!(builder.signal_count(), 2);
    }

    #[test]
    fn test_manifest_builder_duplicate_rejected() {
        let mut builder = DrmManifestBuilder::new();
        builder
            .add_signal(DrmSignal::new(DrmSystem::Widevine, zero_key()))
            .expect("first add");
        let err = builder.add_signal(DrmSignal::new(DrmSystem::Widevine, zero_key()));
        assert!(err.is_err());
    }

    #[test]
    fn test_to_hls_key_tags_multi_system() {
        let mut builder = DrmManifestBuilder::new();
        builder
            .add_signal(DrmSignal::new(DrmSystem::Widevine, zero_key()))
            .expect("add");
        builder
            .add_signal(DrmSignal::new(DrmSystem::PlayReady, zero_key()))
            .expect("add");
        let tags = builder.to_hls_key_tags();
        assert_eq!(tags.lines().count(), 2);
    }
}
