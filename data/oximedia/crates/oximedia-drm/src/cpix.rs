//! CPIX (Content Protection Information Exchange) document support.
//!
//! Implements the DASH-IF CPIX 2.3 specification for exchanging content
//! protection information between entities in a content distribution chain.
//!
//! A CPIX document contains:
//! - Content key list (encrypted or cleartext)
//! - DRM system signaling (PSSH boxes, license acquisition URLs)
//! - Usage rules that bind keys to content (period, track type, label)
//!
//! This module provides building, serializing (XML), and parsing of CPIX
//! documents in a pure-Rust implementation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// CPIX-specific errors.
#[derive(Error, Debug)]
pub enum CpixError {
    #[error("missing required field: {0}")]
    MissingField(String),

    #[error("duplicate content key ID: {0}")]
    DuplicateKeyId(String),

    #[error("unknown content key ID referenced: {0}")]
    UnknownKeyId(String),

    #[error("XML serialization error: {0}")]
    XmlError(String),

    #[error("base64 decode error: {0}")]
    Base64Error(String),

    #[error("validation error: {0}")]
    ValidationError(String),
}

// ---------------------------------------------------------------------------
// Content key
// ---------------------------------------------------------------------------

/// A content key entry in a CPIX document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpixContentKey {
    /// Key ID (UUID as hyphenated string, e.g. "01020304-0506-0708-090a-0b0c0d0e0f10").
    pub key_id: String,
    /// Base64-encoded key value (128 bits = 16 bytes). `None` when keys are
    /// encrypted or delivered out-of-band.
    pub key_value: Option<String>,
    /// Common encryption scheme this key is intended for.
    pub encryption_scheme: Option<String>,
    /// Human-readable label.
    pub label: Option<String>,
}

impl CpixContentKey {
    /// Create a new content key entry.
    pub fn new(key_id: impl Into<String>) -> Self {
        Self {
            key_id: key_id.into(),
            key_value: None,
            encryption_scheme: None,
            label: None,
        }
    }

    /// Builder: set the cleartext key value (base64-encoded).
    pub fn with_key_value(mut self, value: impl Into<String>) -> Self {
        self.key_value = Some(value.into());
        self
    }

    /// Builder: set the encryption scheme (e.g. "cenc", "cbcs").
    pub fn with_encryption_scheme(mut self, scheme: impl Into<String>) -> Self {
        self.encryption_scheme = Some(scheme.into());
        self
    }

    /// Builder: set a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ---------------------------------------------------------------------------
// DRM system signaling
// ---------------------------------------------------------------------------

/// DRM system signaling entry in a CPIX document.
///
/// Each entry binds a particular DRM system to one or more content keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrmSystemSignaling {
    /// DRM system UUID (e.g. Widevine = "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed").
    pub system_id: String,
    /// The content key ID this signaling applies to.
    pub key_id: String,
    /// Base64-encoded PSSH box.
    pub pssh: Option<String>,
    /// License acquisition URL.
    pub license_url: Option<String>,
    /// Custom content protection data (base64-encoded).
    pub content_protection_data: Option<String>,
    /// Custom HLS signaling data (base64-encoded).
    pub hls_signaling_data: Option<String>,
}

impl DrmSystemSignaling {
    /// Create new DRM system signaling.
    pub fn new(system_id: impl Into<String>, key_id: impl Into<String>) -> Self {
        Self {
            system_id: system_id.into(),
            key_id: key_id.into(),
            pssh: None,
            license_url: None,
            content_protection_data: None,
            hls_signaling_data: None,
        }
    }

    /// Builder: set the PSSH box (base64).
    pub fn with_pssh(mut self, pssh: impl Into<String>) -> Self {
        self.pssh = Some(pssh.into());
        self
    }

    /// Builder: set the license acquisition URL.
    pub fn with_license_url(mut self, url: impl Into<String>) -> Self {
        self.license_url = Some(url.into());
        self
    }

    /// Builder: set content protection data.
    pub fn with_content_protection_data(mut self, data: impl Into<String>) -> Self {
        self.content_protection_data = Some(data.into());
        self
    }

    /// Builder: set HLS signaling data.
    pub fn with_hls_signaling_data(mut self, data: impl Into<String>) -> Self {
        self.hls_signaling_data = Some(data.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Usage rules
// ---------------------------------------------------------------------------

/// Filter that matches content by attribute (period, label, track type, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContentFilter {
    /// Matches a specific period index.
    PeriodIndex(u32),
    /// Matches a label string.
    Label(String),
    /// Matches a video track type.
    VideoTrack,
    /// Matches an audio track type.
    AudioTrack,
    /// Matches a subtitle track type.
    SubtitleTrack,
    /// Matches all content (wildcard).
    All,
}

impl fmt::Display for ContentFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContentFilter::PeriodIndex(idx) => write!(f, "period:{idx}"),
            ContentFilter::Label(label) => write!(f, "label:{label}"),
            ContentFilter::VideoTrack => write!(f, "video"),
            ContentFilter::AudioTrack => write!(f, "audio"),
            ContentFilter::SubtitleTrack => write!(f, "subtitle"),
            ContentFilter::All => write!(f, "*"),
        }
    }
}

/// A usage rule that binds a content key to specific content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRule {
    /// The content key ID this rule applies to.
    pub key_id: String,
    /// Filters that determine which content this key protects.
    pub filters: Vec<ContentFilter>,
}

impl UsageRule {
    /// Create a new usage rule.
    pub fn new(key_id: impl Into<String>) -> Self {
        Self {
            key_id: key_id.into(),
            filters: Vec::new(),
        }
    }

    /// Builder: add a filter.
    pub fn with_filter(mut self, filter: ContentFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Check if this rule matches all content (has an `All` filter).
    pub fn matches_all(&self) -> bool {
        self.filters.iter().any(|f| matches!(f, ContentFilter::All))
    }
}

// ---------------------------------------------------------------------------
// CPIX Document
// ---------------------------------------------------------------------------

/// A complete CPIX document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpixDocument {
    /// Document version (default: "2.3").
    pub version: String,
    /// Content ID that this document describes.
    pub content_id: String,
    /// Content keys.
    pub content_keys: Vec<CpixContentKey>,
    /// DRM system signaling entries.
    pub drm_systems: Vec<DrmSystemSignaling>,
    /// Usage rules.
    pub usage_rules: Vec<UsageRule>,
}

impl CpixDocument {
    /// Create a new empty CPIX document.
    pub fn new(content_id: impl Into<String>) -> Self {
        Self {
            version: "2.3".to_string(),
            content_id: content_id.into(),
            content_keys: Vec::new(),
            drm_systems: Vec::new(),
            usage_rules: Vec::new(),
        }
    }

    /// Add a content key.
    pub fn add_content_key(&mut self, key: CpixContentKey) {
        self.content_keys.push(key);
    }

    /// Add a DRM system signaling entry.
    pub fn add_drm_system(&mut self, signaling: DrmSystemSignaling) {
        self.drm_systems.push(signaling);
    }

    /// Add a usage rule.
    pub fn add_usage_rule(&mut self, rule: UsageRule) {
        self.usage_rules.push(rule);
    }

    /// Look up a content key by its key ID.
    pub fn get_content_key(&self, key_id: &str) -> Option<&CpixContentKey> {
        self.content_keys.iter().find(|k| k.key_id == key_id)
    }

    /// Look up all DRM system signaling entries for a given key ID.
    pub fn get_drm_systems_for_key(&self, key_id: &str) -> Vec<&DrmSystemSignaling> {
        self.drm_systems
            .iter()
            .filter(|s| s.key_id == key_id)
            .collect()
    }

    /// Look up usage rules for a given key ID.
    pub fn get_usage_rules_for_key(&self, key_id: &str) -> Vec<&UsageRule> {
        self.usage_rules
            .iter()
            .filter(|r| r.key_id == key_id)
            .collect()
    }

    /// Build a map from key_id -> list of DRM system IDs.
    pub fn key_drm_map(&self) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for sig in &self.drm_systems {
            map.entry(sig.key_id.clone())
                .or_default()
                .push(sig.system_id.clone());
        }
        map
    }

    /// Validate the document for internal consistency.
    pub fn validate(&self) -> Result<(), CpixError> {
        if self.content_id.is_empty() {
            return Err(CpixError::MissingField("content_id".to_string()));
        }
        if self.content_keys.is_empty() {
            return Err(CpixError::MissingField("content_keys".to_string()));
        }

        // Check for duplicate key IDs
        let mut seen_ids = std::collections::HashSet::new();
        for key in &self.content_keys {
            if !seen_ids.insert(&key.key_id) {
                return Err(CpixError::DuplicateKeyId(key.key_id.clone()));
            }
        }

        // Verify DRM system entries reference valid key IDs
        for sig in &self.drm_systems {
            if !seen_ids.contains(&sig.key_id) {
                return Err(CpixError::UnknownKeyId(sig.key_id.clone()));
            }
        }

        // Verify usage rules reference valid key IDs
        for rule in &self.usage_rules {
            if !seen_ids.contains(&rule.key_id) {
                return Err(CpixError::UnknownKeyId(rule.key_id.clone()));
            }
        }

        Ok(())
    }

    /// Serialize the document to a CPIX XML string.
    pub fn to_xml(&self) -> Result<String, CpixError> {
        let mut xml = String::with_capacity(2048);
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str(&format!(
            "<CPIX xmlns=\"urn:dashif:org:cpix\" contentId=\"{}\" version=\"{}\">\n",
            xml_escape(&self.content_id),
            xml_escape(&self.version),
        ));

        // Content key list
        xml.push_str("  <ContentKeyList>\n");
        for key in &self.content_keys {
            xml.push_str(&format!(
                "    <ContentKey kid=\"{}\"",
                xml_escape(&key.key_id),
            ));
            if let Some(scheme) = &key.encryption_scheme {
                xml.push_str(&format!(
                    " commonEncryptionScheme=\"{}\"",
                    xml_escape(scheme)
                ));
            }
            if let Some(label) = &key.label {
                xml.push_str(&format!(" label=\"{}\"", xml_escape(label)));
            }
            if let Some(val) = &key.key_value {
                xml.push_str(&format!(">\n      <Data><Secret><PlainValue>{}</PlainValue></Secret></Data>\n    </ContentKey>\n", xml_escape(val)));
            } else {
                xml.push_str("/>\n");
            }
        }
        xml.push_str("  </ContentKeyList>\n");

        // DRM system list
        if !self.drm_systems.is_empty() {
            xml.push_str("  <DRMSystemList>\n");
            for sig in &self.drm_systems {
                xml.push_str(&format!(
                    "    <DRMSystem systemId=\"{}\" kid=\"{}\">\n",
                    xml_escape(&sig.system_id),
                    xml_escape(&sig.key_id),
                ));
                if let Some(pssh) = &sig.pssh {
                    xml.push_str(&format!("      <PSSH>{}</PSSH>\n", xml_escape(pssh)));
                }
                if let Some(url) = &sig.license_url {
                    xml.push_str(&format!(
                        "      <URIExtXKey>{}</URIExtXKey>\n",
                        xml_escape(url),
                    ));
                }
                if let Some(data) = &sig.content_protection_data {
                    xml.push_str(&format!(
                        "      <ContentProtectionData>{}</ContentProtectionData>\n",
                        xml_escape(data),
                    ));
                }
                if let Some(hls) = &sig.hls_signaling_data {
                    xml.push_str(&format!(
                        "      <HLSSignalingData>{}</HLSSignalingData>\n",
                        xml_escape(hls),
                    ));
                }
                xml.push_str("    </DRMSystem>\n");
            }
            xml.push_str("  </DRMSystemList>\n");
        }

        // Usage rule list
        if !self.usage_rules.is_empty() {
            xml.push_str("  <ContentKeyUsageRuleList>\n");
            for rule in &self.usage_rules {
                xml.push_str(&format!(
                    "    <ContentKeyUsageRule kid=\"{}\">\n",
                    xml_escape(&rule.key_id),
                ));
                for filter in &rule.filters {
                    match filter {
                        ContentFilter::VideoTrack => {
                            xml.push_str("      <VideoFilter/>\n");
                        }
                        ContentFilter::AudioTrack => {
                            xml.push_str("      <AudioFilter/>\n");
                        }
                        ContentFilter::SubtitleTrack => {
                            xml.push_str("      <SubtitleFilter/>\n");
                        }
                        ContentFilter::PeriodIndex(idx) => {
                            xml.push_str(&format!(
                                "      <KeyPeriodFilter periodId=\"{}\"/>\n",
                                idx
                            ));
                        }
                        ContentFilter::Label(label) => {
                            xml.push_str(&format!(
                                "      <LabelFilter label=\"{}\"/>\n",
                                xml_escape(label),
                            ));
                        }
                        ContentFilter::All => {
                            xml.push_str("      <AllFilter/>\n");
                        }
                    }
                }
                xml.push_str("    </ContentKeyUsageRule>\n");
            }
            xml.push_str("  </ContentKeyUsageRuleList>\n");
        }

        xml.push_str("</CPIX>\n");
        Ok(xml)
    }

    /// Parse a CPIX XML string into a `CpixDocument`.
    ///
    /// This is a lightweight pull-parser that handles the subset of CPIX XML
    /// generated by `to_xml()`.
    pub fn from_xml(xml: &str) -> Result<Self, CpixError> {
        let content_id = extract_attr(xml, "CPIX", "contentId")
            .ok_or_else(|| CpixError::MissingField("contentId".to_string()))?;
        let version = extract_attr(xml, "CPIX", "version").unwrap_or_else(|| "2.3".to_string());

        let mut doc = CpixDocument {
            version,
            content_id,
            content_keys: Vec::new(),
            drm_systems: Vec::new(),
            usage_rules: Vec::new(),
        };

        // Parse content keys
        for key_xml in find_elements(xml, "ContentKey") {
            let kid = extract_attr_from_element(&key_xml, "kid")
                .ok_or_else(|| CpixError::MissingField("ContentKey kid".to_string()))?;
            let scheme = extract_attr_from_element(&key_xml, "commonEncryptionScheme");
            let label = extract_attr_from_element(&key_xml, "label");
            let key_value = extract_inner_text(&key_xml, "PlainValue");
            let mut ck = CpixContentKey::new(kid);
            if let Some(v) = key_value {
                ck = ck.with_key_value(v);
            }
            if let Some(s) = scheme {
                ck = ck.with_encryption_scheme(s);
            }
            if let Some(l) = label {
                ck = ck.with_label(l);
            }
            doc.add_content_key(ck);
        }

        // Parse DRM systems
        for drm_xml in find_elements(xml, "DRMSystem") {
            let system_id = extract_attr_from_element(&drm_xml, "systemId")
                .ok_or_else(|| CpixError::MissingField("DRMSystem systemId".to_string()))?;
            let kid = extract_attr_from_element(&drm_xml, "kid")
                .ok_or_else(|| CpixError::MissingField("DRMSystem kid".to_string()))?;
            let mut sig = DrmSystemSignaling::new(system_id, kid);
            if let Some(pssh) = extract_inner_text(&drm_xml, "PSSH") {
                sig = sig.with_pssh(pssh);
            }
            if let Some(url) = extract_inner_text(&drm_xml, "URIExtXKey") {
                sig = sig.with_license_url(url);
            }
            if let Some(data) = extract_inner_text(&drm_xml, "ContentProtectionData") {
                sig = sig.with_content_protection_data(data);
            }
            if let Some(hls) = extract_inner_text(&drm_xml, "HLSSignalingData") {
                sig = sig.with_hls_signaling_data(hls);
            }
            doc.add_drm_system(sig);
        }

        // Parse usage rules
        for rule_xml in find_elements(xml, "ContentKeyUsageRule") {
            let kid = extract_attr_from_element(&rule_xml, "kid")
                .ok_or_else(|| CpixError::MissingField("ContentKeyUsageRule kid".to_string()))?;
            let mut rule = UsageRule::new(kid);
            if rule_xml.contains("<VideoFilter") {
                rule = rule.with_filter(ContentFilter::VideoTrack);
            }
            if rule_xml.contains("<AudioFilter") {
                rule = rule.with_filter(ContentFilter::AudioTrack);
            }
            if rule_xml.contains("<SubtitleFilter") {
                rule = rule.with_filter(ContentFilter::SubtitleTrack);
            }
            if rule_xml.contains("<AllFilter") {
                rule = rule.with_filter(ContentFilter::All);
            }
            // Parse period filters
            for period_elem in find_elements(&rule_xml, "KeyPeriodFilter") {
                if let Some(period_id) = extract_attr_from_element(&period_elem, "periodId") {
                    if let Ok(idx) = period_id.parse::<u32>() {
                        rule = rule.with_filter(ContentFilter::PeriodIndex(idx));
                    }
                }
            }
            // Parse label filters
            for label_elem in find_elements(&rule_xml, "LabelFilter") {
                if let Some(label) = extract_attr_from_element(&label_elem, "label") {
                    rule = rule.with_filter(ContentFilter::Label(label));
                }
            }
            doc.add_usage_rule(rule);
        }

        Ok(doc)
    }
}

// ---------------------------------------------------------------------------
// XML helpers (minimal, no external XML crate dependency for CPIX)
// ---------------------------------------------------------------------------

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn extract_attr(xml: &str, element: &str, attr: &str) -> Option<String> {
    let tag_start = xml.find(&format!("<{element}"))?;
    let tag_end = xml[tag_start..].find('>')? + tag_start;
    let tag = &xml[tag_start..=tag_end];
    extract_attr_from_element(tag, attr)
}

fn extract_attr_from_element(element: &str, attr: &str) -> Option<String> {
    let pattern = format!("{attr}=\"");
    let start = element.find(&pattern)? + pattern.len();
    let end = element[start..].find('"')? + start;
    Some(xml_unescape(&element[start..end]))
}

fn extract_inner_text(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)? + start;
    Some(xml_unescape(xml[start..end].trim()))
}

fn find_elements(xml: &str, tag: &str) -> Vec<String> {
    let mut results = Vec::new();
    let open_full = format!("<{tag}");
    let close = format!("</{tag}>");
    let self_close = "/>";

    let mut search_from = 0;
    while let Some(start) = xml[search_from..].find(&open_full) {
        let abs_start = search_from + start;
        let after_tag = abs_start + open_full.len();

        // Verify this is an exact tag match: the character after `<tag` must be
        // whitespace, '/', or '>' (not another letter like in `<ContentKeyList>`).
        if let Some(next_char) = xml[after_tag..].chars().next() {
            if next_char.is_alphanumeric() || next_char == '_' || next_char == '-' {
                search_from = after_tag;
                continue;
            }
        }

        // Check for self-closing element
        if let Some(next_gt) = xml[after_tag..].find('>') {
            let gt_pos = after_tag + next_gt;
            let before_gt = &xml[after_tag..=gt_pos];
            if before_gt.trim_end().ends_with(self_close) {
                results.push(xml[abs_start..=gt_pos].to_string());
                search_from = gt_pos + 1;
                continue;
            }
        }
        // Look for closing tag
        if let Some(end_offset) = xml[abs_start..].find(&close) {
            let end = abs_start + end_offset + close.len();
            results.push(xml[abs_start..end].to_string());
            search_from = end;
        } else {
            break;
        }
    }
    results
}

fn xml_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> CpixDocument {
        let mut doc = CpixDocument::new("content-123");

        let key1 = CpixContentKey::new("01020304-0506-0708-090a-0b0c0d0e0f10")
            .with_key_value("AAAAAAAAAAAAAAAAAAAAAA==")
            .with_encryption_scheme("cenc")
            .with_label("video_key");

        let key2 = CpixContentKey::new("11121314-1516-1718-191a-1b1c1d1e1f20")
            .with_key_value("BBBBBBBBBBBBBBBBBBBBBB==")
            .with_encryption_scheme("cbcs")
            .with_label("audio_key");

        doc.add_content_key(key1);
        doc.add_content_key(key2);

        let sig1 = DrmSystemSignaling::new(
            "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
            "01020304-0506-0708-090a-0b0c0d0e0f10",
        )
        .with_pssh("AAAARHBzc2g=")
        .with_license_url("https://license.example.com/widevine");

        let sig2 = DrmSystemSignaling::new(
            "9a04f079-9840-4286-ab92-e65be0885f95",
            "01020304-0506-0708-090a-0b0c0d0e0f10",
        )
        .with_license_url("https://license.example.com/playready");

        doc.add_drm_system(sig1);
        doc.add_drm_system(sig2);

        let rule1 = UsageRule::new("01020304-0506-0708-090a-0b0c0d0e0f10")
            .with_filter(ContentFilter::VideoTrack);
        let rule2 = UsageRule::new("11121314-1516-1718-191a-1b1c1d1e1f20")
            .with_filter(ContentFilter::AudioTrack);

        doc.add_usage_rule(rule1);
        doc.add_usage_rule(rule2);

        doc
    }

    #[test]
    fn test_create_cpix_document() {
        let doc = sample_doc();
        assert_eq!(doc.content_keys.len(), 2);
        assert_eq!(doc.drm_systems.len(), 2);
        assert_eq!(doc.usage_rules.len(), 2);
    }

    #[test]
    fn test_validate_valid_document() {
        let doc = sample_doc();
        assert!(doc.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_content_id() {
        let doc = CpixDocument::new("");
        assert!(doc.validate().is_err());
    }

    #[test]
    fn test_validate_no_keys() {
        let doc = CpixDocument::new("test");
        assert!(doc.validate().is_err());
    }

    #[test]
    fn test_validate_duplicate_key_id() {
        let mut doc = CpixDocument::new("test");
        doc.add_content_key(CpixContentKey::new("same-id"));
        doc.add_content_key(CpixContentKey::new("same-id"));
        let err = doc.validate().unwrap_err();
        assert!(format!("{err}").contains("duplicate"));
    }

    #[test]
    fn test_validate_unknown_key_in_drm() {
        let mut doc = CpixDocument::new("test");
        doc.add_content_key(CpixContentKey::new("key-1"));
        doc.add_drm_system(DrmSystemSignaling::new("sys-1", "unknown-key"));
        let err = doc.validate().unwrap_err();
        assert!(format!("{err}").contains("unknown"));
    }

    #[test]
    fn test_validate_unknown_key_in_rule() {
        let mut doc = CpixDocument::new("test");
        doc.add_content_key(CpixContentKey::new("key-1"));
        doc.add_usage_rule(UsageRule::new("missing-key"));
        let err = doc.validate().unwrap_err();
        assert!(format!("{err}").contains("unknown"));
    }

    #[test]
    fn test_get_content_key() {
        let doc = sample_doc();
        let key = doc.get_content_key("01020304-0506-0708-090a-0b0c0d0e0f10");
        assert!(key.is_some());
        assert_eq!(key.map(|k| k.label.as_deref()), Some(Some("video_key")));
    }

    #[test]
    fn test_get_drm_systems_for_key() {
        let doc = sample_doc();
        let sigs = doc.get_drm_systems_for_key("01020304-0506-0708-090a-0b0c0d0e0f10");
        assert_eq!(sigs.len(), 2);
    }

    #[test]
    fn test_key_drm_map() {
        let doc = sample_doc();
        let map = doc.key_drm_map();
        let systems = map.get("01020304-0506-0708-090a-0b0c0d0e0f10");
        assert!(systems.is_some());
        assert_eq!(systems.map(|s| s.len()), Some(2));
    }

    #[test]
    fn test_usage_rule_matches_all() {
        let rule = UsageRule::new("key-1").with_filter(ContentFilter::All);
        assert!(rule.matches_all());
        let rule2 = UsageRule::new("key-1").with_filter(ContentFilter::VideoTrack);
        assert!(!rule2.matches_all());
    }

    #[test]
    fn test_content_filter_display() {
        assert_eq!(ContentFilter::VideoTrack.to_string(), "video");
        assert_eq!(ContentFilter::AudioTrack.to_string(), "audio");
        assert_eq!(ContentFilter::PeriodIndex(3).to_string(), "period:3");
        assert_eq!(
            ContentFilter::Label("hd".to_string()).to_string(),
            "label:hd"
        );
        assert_eq!(ContentFilter::All.to_string(), "*");
    }

    #[test]
    fn test_to_xml_roundtrip() {
        let doc = sample_doc();
        let xml = doc.to_xml().expect("to_xml should succeed");
        assert!(xml.contains("CPIX"));
        assert!(xml.contains("content-123"));
        assert!(xml.contains("ContentKey"));
        assert!(xml.contains("DRMSystem"));
        assert!(xml.contains("ContentKeyUsageRule"));
        assert!(xml.contains("VideoFilter"));
        assert!(xml.contains("AudioFilter"));

        let parsed = CpixDocument::from_xml(&xml).expect("from_xml should succeed");
        assert_eq!(parsed.content_id, "content-123");
        assert_eq!(parsed.content_keys.len(), 2);
        assert_eq!(parsed.drm_systems.len(), 2);
        assert_eq!(parsed.usage_rules.len(), 2);
    }

    #[test]
    fn test_xml_roundtrip_preserves_key_value() {
        let doc = sample_doc();
        let xml = doc.to_xml().expect("to_xml should succeed");
        let parsed = CpixDocument::from_xml(&xml).expect("from_xml should succeed");
        let key = parsed
            .get_content_key("01020304-0506-0708-090a-0b0c0d0e0f10")
            .expect("key should exist");
        assert_eq!(key.key_value.as_deref(), Some("AAAAAAAAAAAAAAAAAAAAAA=="));
    }

    #[test]
    fn test_xml_roundtrip_preserves_drm_fields() {
        let doc = sample_doc();
        let xml = doc.to_xml().expect("to_xml should succeed");
        let parsed = CpixDocument::from_xml(&xml).expect("from_xml should succeed");
        let sigs = parsed.get_drm_systems_for_key("01020304-0506-0708-090a-0b0c0d0e0f10");
        assert_eq!(sigs.len(), 2);
        let wv = sigs.iter().find(|s| s.system_id.contains("edef8ba9"));
        assert!(wv.is_some());
        let wv = wv.expect("widevine should exist");
        assert_eq!(wv.pssh.as_deref(), Some("AAAARHBzc2g="));
        assert_eq!(
            wv.license_url.as_deref(),
            Some("https://license.example.com/widevine")
        );
    }

    #[test]
    fn test_xml_with_special_characters() {
        let mut doc = CpixDocument::new("test&<>\"content");
        doc.add_content_key(CpixContentKey::new("key-1").with_label("label&special"));
        let xml = doc.to_xml().expect("to_xml should succeed");
        assert!(xml.contains("&amp;"));
        let parsed = CpixDocument::from_xml(&xml).expect("from_xml should succeed");
        assert_eq!(parsed.content_id, "test&<>\"content");
    }

    #[test]
    fn test_xml_with_period_and_label_filters() {
        let mut doc = CpixDocument::new("test");
        doc.add_content_key(CpixContentKey::new("key-1"));
        doc.add_usage_rule(
            UsageRule::new("key-1")
                .with_filter(ContentFilter::PeriodIndex(5))
                .with_filter(ContentFilter::Label("hdr".to_string())),
        );
        let xml = doc.to_xml().expect("to_xml should succeed");
        let parsed = CpixDocument::from_xml(&xml).expect("from_xml should succeed");
        let rules = parsed.get_usage_rules_for_key("key-1");
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].filters.len(), 2);
        assert!(rules[0].filters.contains(&ContentFilter::PeriodIndex(5)));
        assert!(rules[0]
            .filters
            .contains(&ContentFilter::Label("hdr".to_string())));
    }

    #[test]
    fn test_document_without_drm_or_rules() {
        let mut doc = CpixDocument::new("minimal");
        doc.add_content_key(CpixContentKey::new("key-1").with_key_value("dGVzdA=="));
        let xml = doc.to_xml().expect("to_xml should succeed");
        assert!(!xml.contains("DRMSystemList"));
        assert!(!xml.contains("ContentKeyUsageRuleList"));
        let parsed = CpixDocument::from_xml(&xml).expect("from_xml should succeed");
        assert_eq!(parsed.content_keys.len(), 1);
        assert!(parsed.drm_systems.is_empty());
        assert!(parsed.usage_rules.is_empty());
    }

    #[test]
    fn test_cpix_error_display() {
        let e = CpixError::MissingField("foo".to_string());
        assert!(format!("{e}").contains("foo"));
    }

    #[test]
    fn test_key_without_value() {
        let mut doc = CpixDocument::new("test");
        doc.add_content_key(CpixContentKey::new("key-1"));
        let xml = doc.to_xml().expect("to_xml should succeed");
        // Self-closing element
        assert!(xml.contains("ContentKey kid=\"key-1\"/>"));
        let parsed = CpixDocument::from_xml(&xml).expect("from_xml should succeed");
        assert!(parsed.content_keys[0].key_value.is_none());
    }

    #[test]
    fn test_hls_signaling_data_roundtrip() {
        let mut doc = CpixDocument::new("hls-test");
        doc.add_content_key(CpixContentKey::new("key-1"));
        doc.add_drm_system(
            DrmSystemSignaling::new("system-1", "key-1").with_hls_signaling_data("aGxzLWRhdGE="),
        );
        let xml = doc.to_xml().expect("to_xml should succeed");
        let parsed = CpixDocument::from_xml(&xml).expect("from_xml should succeed");
        assert_eq!(
            parsed.drm_systems[0].hls_signaling_data.as_deref(),
            Some("aGxzLWRhdGE=")
        );
    }
}
