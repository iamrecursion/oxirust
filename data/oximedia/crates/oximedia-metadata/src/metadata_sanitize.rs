//! Metadata sanitization and privacy cleaning.
//!
//! Provides tools to strip, redact, or anonymize sensitive metadata fields
//! (GPS coordinates, personal identifiers, device serials) for privacy
//! compliance and safe distribution.
//!
//! # Configurable Rules
//!
//! The [`SanitizeRule`] system allows fine-grained control over which fields
//! are sanitized and how. Rules can match by exact key, substring pattern,
//! or field category. Actions include stripping (removal), redacting (value
//! replacement), and value transformation.
//!
//! # Preset Policies
//!
//! - [`privacy_policy`]: Strips GPS, device info, and common PII fields
//! - [`distribution_policy`]: Strips GPS, redacts author info
//! - [`gdpr_policy`]: GDPR-compliant policy (strips all personal data categories)
//! - [`broadcast_policy`]: Preserves rights/copyright, strips everything else sensitive
//!
//! # Metadata Integration
//!
//! [`sanitize_metadata`] operates directly on [`Metadata`] containers,
//! while [`sanitize`] operates on raw `HashMap<String, String>`.

use crate::{Metadata, MetadataValue};
use std::collections::{HashMap, HashSet};

/// Actions that a sanitization rule can perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SanitizeAction {
    /// Remove the field entirely.
    Strip,
    /// Replace the value with the policy's redaction placeholder.
    Redact,
    /// Replace the value with a specific custom string.
    ReplaceWith(String),
    /// Truncate the text value to at most N characters.
    Truncate(usize),
    /// Hash the value (one-way anonymization using a simple FNV-1a hash).
    HashAnonymize,
}

/// How a rule matches against field keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleMatch {
    /// Match a specific field key exactly.
    Exact(String),
    /// Match if the key contains this substring (case-insensitive).
    Contains(String),
    /// Match if the key starts with this prefix (case-insensitive).
    Prefix(String),
    /// Match if the key belongs to a known field category.
    Category(FieldCategory),
}

/// Categories of metadata fields for bulk matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldCategory {
    /// GPS / geolocation fields.
    Gps,
    /// Device / software identification fields.
    DeviceInfo,
    /// Personal identifiable information (name, email, phone, address).
    PersonalInfo,
    /// Copyright and rights-related fields.
    Rights,
    /// Timestamps and date-related fields.
    Temporal,
    /// Technical / encoding parameters.
    Technical,
}

/// Well-known field patterns for each category.
const GPS_PATTERNS: &[&str] = &[
    "gps",
    "latitude",
    "longitude",
    "altitude",
    "geolocation",
    "location",
    "gpslatitude",
    "gpslongitude",
    "gpsaltitude",
    "gpsposition",
    "gpsdestlatitude",
    "gpsdestlongitude",
    "gpsspeed",
    "gpstrack",
    "gpsimgdirection",
];

const DEVICE_PATTERNS: &[&str] = &[
    "serial",
    "device",
    "software",
    "firmware",
    "camera_model",
    "cameraserialnumber",
    "lensserialnumber",
    "make",
    "model",
    "lensmodel",
    "lensmake",
    "bodyserialnum",
    "internalserialnumber",
    "uniquecameramodel",
    "hostcomputer",
];

const PERSONAL_PATTERNS: &[&str] = &[
    "email",
    "phone",
    "address",
    "author",
    "creator",
    "owner",
    "by-line",
    "contact",
    "artist",
    "photographer",
    "credit",
    "personinimage",
    "ownername",
];

const RIGHTS_PATTERNS: &[&str] = &[
    "copyright",
    "rights",
    "license",
    "usage",
    "cprt",
    "copyrightnotice",
    "webstatement",
];

const TEMPORAL_PATTERNS: &[&str] = &[
    "datetime",
    "datecreated",
    "datetimeoriginal",
    "datetimedigitized",
    "modifydate",
    "createdate",
    "metadatadate",
];

const TECHNICAL_PATTERNS: &[&str] = &[
    "encoder",
    "encoded_by",
    "encoding",
    "bitrate",
    "samplerate",
    "codec",
    "format",
    "resolution",
    "colorspace",
];

/// Returns the patterns for a given field category.
fn patterns_for_category(category: FieldCategory) -> &'static [&'static str] {
    match category {
        FieldCategory::Gps => GPS_PATTERNS,
        FieldCategory::DeviceInfo => DEVICE_PATTERNS,
        FieldCategory::PersonalInfo => PERSONAL_PATTERNS,
        FieldCategory::Rights => RIGHTS_PATTERNS,
        FieldCategory::Temporal => TEMPORAL_PATTERNS,
        FieldCategory::Technical => TECHNICAL_PATTERNS,
    }
}

/// A configurable sanitization rule.
#[derive(Debug, Clone)]
pub struct SanitizeRule {
    /// How this rule matches field keys.
    pub matcher: RuleMatch,
    /// What action to take on matched fields.
    pub action: SanitizeAction,
    /// Optional human-readable description of why this rule exists.
    pub description: Option<String>,
}

impl SanitizeRule {
    /// Create a new rule.
    pub fn new(matcher: RuleMatch, action: SanitizeAction) -> Self {
        Self {
            matcher,
            action,
            description: None,
        }
    }

    /// Set a description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Check if this rule matches the given key.
    pub fn matches(&self, key: &str) -> bool {
        let lower_key = key.to_lowercase();
        match &self.matcher {
            RuleMatch::Exact(exact) => lower_key == exact.to_lowercase(),
            RuleMatch::Contains(substr) => lower_key.contains(&substr.to_lowercase()),
            RuleMatch::Prefix(prefix) => lower_key.starts_with(&prefix.to_lowercase()),
            RuleMatch::Category(cat) => {
                let patterns = patterns_for_category(*cat);
                patterns.iter().any(|p| lower_key.contains(p))
            }
        }
    }
}

/// Policy that drives which fields to sanitize and how.
#[derive(Debug, Clone)]
pub struct SanitizePolicy {
    /// Fields to completely remove.
    strip_fields: HashSet<String>,
    /// Fields whose values should be replaced with a redacted placeholder.
    redact_fields: HashSet<String>,
    /// Custom replacement text for redacted fields.
    redact_placeholder: String,
    /// If true, strip all GPS / geolocation fields automatically.
    strip_gps: bool,
    /// If true, strip device / software identification fields automatically.
    strip_device_info: bool,
    /// Maximum allowed value length; longer values are truncated.
    max_value_length: Option<usize>,
    /// Configurable rule chain (evaluated in order; first match wins).
    rules: Vec<SanitizeRule>,
    /// If true, strip personal info fields automatically.
    strip_personal_info: bool,
    /// Allow-list: if non-empty, only these keys survive (everything else stripped).
    allow_list: HashSet<String>,
}

impl Default for SanitizePolicy {
    fn default() -> Self {
        Self {
            strip_fields: HashSet::new(),
            redact_fields: HashSet::new(),
            redact_placeholder: "[REDACTED]".into(),
            strip_gps: false,
            strip_device_info: false,
            max_value_length: None,
            rules: Vec::new(),
            strip_personal_info: false,
            allow_list: HashSet::new(),
        }
    }
}

impl SanitizePolicy {
    /// Create a new default policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a field key to be completely stripped.
    pub fn add_strip_field(&mut self, key: impl Into<String>) {
        self.strip_fields.insert(key.into());
    }

    /// Add a field key to be redacted (value replaced).
    pub fn add_redact_field(&mut self, key: impl Into<String>) {
        self.redact_fields.insert(key.into());
    }

    /// Set the redaction placeholder text.
    pub fn set_redact_placeholder(&mut self, placeholder: impl Into<String>) {
        self.redact_placeholder = placeholder.into();
    }

    /// Enable automatic GPS field stripping.
    pub fn enable_strip_gps(&mut self) {
        self.strip_gps = true;
    }

    /// Enable automatic device info stripping.
    pub fn enable_strip_device_info(&mut self) {
        self.strip_device_info = true;
    }

    /// Enable automatic personal info stripping.
    pub fn enable_strip_personal_info(&mut self) {
        self.strip_personal_info = true;
    }

    /// Set a maximum value length (truncates longer values).
    pub fn set_max_value_length(&mut self, max_len: usize) {
        self.max_value_length = Some(max_len);
    }

    /// Add a configurable rule to the policy.
    pub fn add_rule(&mut self, rule: SanitizeRule) {
        self.rules.push(rule);
    }

    /// Set the allow-list. When non-empty, only keys in this set are kept.
    pub fn set_allow_list(&mut self, keys: HashSet<String>) {
        self.allow_list = keys;
    }

    /// Add a single key to the allow-list.
    pub fn add_allow_key(&mut self, key: impl Into<String>) {
        self.allow_list.insert(key.into());
    }

    /// Get the redaction placeholder.
    pub fn redact_placeholder(&self) -> &str {
        &self.redact_placeholder
    }

    /// Whether GPS stripping is enabled.
    pub fn strip_gps(&self) -> bool {
        self.strip_gps
    }

    /// Whether device info stripping is enabled.
    pub fn strip_device_info(&self) -> bool {
        self.strip_device_info
    }

    /// Whether personal info stripping is enabled.
    pub fn strip_personal_info(&self) -> bool {
        self.strip_personal_info
    }

    /// Get the max value length setting.
    pub fn max_value_length(&self) -> Option<usize> {
        self.max_value_length
    }

    /// Get the set of fields to strip.
    pub fn strip_fields(&self) -> &HashSet<String> {
        &self.strip_fields
    }

    /// Get the set of fields to redact.
    pub fn redact_fields(&self) -> &HashSet<String> {
        &self.redact_fields
    }

    /// Get the rules.
    pub fn rules(&self) -> &[SanitizeRule] {
        &self.rules
    }

    /// Get the allow-list.
    pub fn allow_list(&self) -> &HashSet<String> {
        &self.allow_list
    }

    /// Find the first matching rule for a given key.
    fn find_matching_rule(&self, key: &str) -> Option<&SanitizeRule> {
        self.rules.iter().find(|r| r.matches(key))
    }
}

/// Well-known GPS-related field keys (legacy API).
const GPS_FIELD_PATTERNS: &[&str] = &[
    "gps",
    "latitude",
    "longitude",
    "altitude",
    "geolocation",
    "location",
    "GPSLatitude",
    "GPSLongitude",
    "GPSAltitude",
    "GPSPosition",
];

/// Well-known device / software identification field keys (legacy API).
const DEVICE_FIELD_PATTERNS: &[&str] = &[
    "serial",
    "device",
    "software",
    "firmware",
    "camera_model",
    "CameraSerialNumber",
    "LensSerialNumber",
    "Software",
    "Make",
    "Model",
];

/// Well-known personal info field keys.
const PERSONAL_FIELD_PATTERNS: &[&str] = &[
    "email",
    "phone",
    "address",
    "author",
    "creator",
    "owner",
    "by-line",
    "contact",
    "PersonInImage",
    "OwnerName",
];

/// Check if a field key matches any pattern in a list (case-insensitive substring).
fn matches_patterns(key: &str, patterns: &[&str]) -> bool {
    let lower = key.to_lowercase();
    patterns.iter().any(|p| lower.contains(&p.to_lowercase()))
}

/// Simple FNV-1a hash for anonymization.
fn fnv1a_hash(data: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x00000100000001B3);
    }
    format!("anon_{hash:016x}")
}

/// Result of a sanitization run.
#[derive(Debug, Clone)]
pub struct SanitizeReport {
    /// Number of fields stripped.
    pub fields_stripped: usize,
    /// Number of fields redacted.
    pub fields_redacted: usize,
    /// Number of fields truncated.
    pub fields_truncated: usize,
    /// Number of fields anonymized (hashed).
    pub fields_anonymized: usize,
    /// Keys that were stripped.
    pub stripped_keys: Vec<String>,
    /// Keys that were redacted.
    pub redacted_keys: Vec<String>,
    /// Keys that were truncated.
    pub truncated_keys: Vec<String>,
    /// Keys that were anonymized.
    pub anonymized_keys: Vec<String>,
    /// Rules that were triggered (rule index, key).
    pub rule_matches: Vec<(usize, String)>,
}

impl SanitizeReport {
    /// Create an empty report.
    fn new() -> Self {
        Self {
            fields_stripped: 0,
            fields_redacted: 0,
            fields_truncated: 0,
            fields_anonymized: 0,
            stripped_keys: Vec::new(),
            redacted_keys: Vec::new(),
            truncated_keys: Vec::new(),
            anonymized_keys: Vec::new(),
            rule_matches: Vec::new(),
        }
    }

    /// Total number of fields affected.
    pub fn total_affected(&self) -> usize {
        self.fields_stripped + self.fields_redacted + self.fields_truncated + self.fields_anonymized
    }
}

/// Apply a sanitization policy to a metadata map in-place.
///
/// Returns a report describing what was changed.
pub fn sanitize(fields: &mut HashMap<String, String>, policy: &SanitizePolicy) -> SanitizeReport {
    let mut report = SanitizeReport::new();

    // Allow-list mode: strip everything not in the list first
    if !policy.allow_list.is_empty() {
        let keys_to_remove: Vec<String> = fields
            .keys()
            .filter(|k| !policy.allow_list.contains(k.as_str()))
            .cloned()
            .collect();
        for key in &keys_to_remove {
            fields.remove(key);
            report.fields_stripped += 1;
            report.stripped_keys.push(key.clone());
        }
        return report;
    }

    // Phase 1: Apply configurable rules (first match wins per key)
    let rule_keys: Vec<String> = fields.keys().cloned().collect();
    for key in &rule_keys {
        if let Some((idx, rule)) = policy
            .rules
            .iter()
            .enumerate()
            .find(|(_, r)| r.matches(key))
        {
            report.rule_matches.push((idx, key.clone()));
            match &rule.action {
                SanitizeAction::Strip => {
                    fields.remove(key);
                    report.fields_stripped += 1;
                    report.stripped_keys.push(key.clone());
                }
                SanitizeAction::Redact => {
                    fields.insert(key.clone(), policy.redact_placeholder.clone());
                    report.fields_redacted += 1;
                    report.redacted_keys.push(key.clone());
                }
                SanitizeAction::ReplaceWith(replacement) => {
                    fields.insert(key.clone(), replacement.clone());
                    report.fields_redacted += 1;
                    report.redacted_keys.push(key.clone());
                }
                SanitizeAction::Truncate(max_len) => {
                    if let Some(val) = fields.get_mut(key) {
                        if val.len() > *max_len {
                            val.truncate(*max_len);
                            report.fields_truncated += 1;
                            report.truncated_keys.push(key.clone());
                        }
                    }
                }
                SanitizeAction::HashAnonymize => {
                    if let Some(val) = fields.get(key) {
                        let hashed = fnv1a_hash(val);
                        fields.insert(key.clone(), hashed);
                        report.fields_anonymized += 1;
                        report.anonymized_keys.push(key.clone());
                    }
                }
            }
            continue; // First rule matched; skip legacy checks for this key
        }
    }

    // Phase 2: Legacy pattern-based strip/redact (for keys not already handled by rules)
    let keys_to_strip: Vec<String> = fields
        .keys()
        .filter(|k| {
            if policy.strip_fields.contains(k.as_str()) {
                return true;
            }
            if policy.strip_gps && matches_patterns(k, GPS_FIELD_PATTERNS) {
                return true;
            }
            if policy.strip_device_info && matches_patterns(k, DEVICE_FIELD_PATTERNS) {
                return true;
            }
            if policy.strip_personal_info && matches_patterns(k, PERSONAL_FIELD_PATTERNS) {
                return true;
            }
            false
        })
        .cloned()
        .collect();

    for key in &keys_to_strip {
        fields.remove(key);
        report.fields_stripped += 1;
        report.stripped_keys.push(key.clone());
    }

    // Redact fields
    for key in &policy.redact_fields {
        if fields.contains_key(key) {
            fields.insert(key.clone(), policy.redact_placeholder.clone());
            report.fields_redacted += 1;
            report.redacted_keys.push(key.clone());
        }
    }

    // Truncate long values
    if let Some(max_len) = policy.max_value_length {
        for (key, value) in fields.iter_mut() {
            if value.len() > max_len {
                value.truncate(max_len);
                report.fields_truncated += 1;
                report.truncated_keys.push(key.clone());
            }
        }
    }

    report
}

/// Apply a sanitization policy directly to a [`Metadata`] container.
///
/// This operates on the `Metadata` type, converting text fields to/from strings
/// for rule evaluation. Non-text fields (Binary, Pictures) are handled by
/// key matching only (they can be stripped but not redacted/truncated).
///
/// Returns a report describing what was changed.
pub fn sanitize_metadata(metadata: &mut Metadata, policy: &SanitizePolicy) -> SanitizeReport {
    let mut report = SanitizeReport::new();

    // Collect all keys
    let all_keys: Vec<String> = metadata.keys().into_iter().cloned().collect();

    // Allow-list mode
    if !policy.allow_list.is_empty() {
        for key in &all_keys {
            if !policy.allow_list.contains(key.as_str()) {
                metadata.remove(key);
                report.fields_stripped += 1;
                report.stripped_keys.push(key.clone());
            }
        }
        return report;
    }

    // Collect keys to process (re-fetch after potential removals)
    let keys: Vec<String> = metadata.keys().into_iter().cloned().collect();

    for key in &keys {
        // Check configurable rules first
        if let Some((idx, rule)) = policy
            .rules
            .iter()
            .enumerate()
            .find(|(_, r)| r.matches(key))
        {
            report.rule_matches.push((idx, key.clone()));
            apply_rule_to_metadata(
                metadata,
                key,
                &rule.action,
                &policy.redact_placeholder,
                &mut report,
            );
            continue;
        }

        // Legacy pattern checks
        let should_strip = policy.strip_fields.contains(key.as_str())
            || (policy.strip_gps && matches_patterns(key, GPS_FIELD_PATTERNS))
            || (policy.strip_device_info && matches_patterns(key, DEVICE_FIELD_PATTERNS))
            || (policy.strip_personal_info && matches_patterns(key, PERSONAL_FIELD_PATTERNS));

        if should_strip {
            metadata.remove(key);
            report.fields_stripped += 1;
            report.stripped_keys.push(key.clone());
            continue;
        }

        // Redact
        if policy.redact_fields.contains(key.as_str()) {
            metadata.insert(
                key.clone(),
                MetadataValue::Text(policy.redact_placeholder.clone()),
            );
            report.fields_redacted += 1;
            report.redacted_keys.push(key.clone());
            continue;
        }

        // Truncate
        if let Some(max_len) = policy.max_value_length {
            if let Some(MetadataValue::Text(text)) = metadata.get(key) {
                if text.len() > max_len {
                    let truncated = text[..max_len].to_string();
                    metadata.insert(key.clone(), MetadataValue::Text(truncated));
                    report.fields_truncated += 1;
                    report.truncated_keys.push(key.clone());
                }
            }
        }
    }

    report
}

/// Apply a rule action to a single metadata field.
fn apply_rule_to_metadata(
    metadata: &mut Metadata,
    key: &str,
    action: &SanitizeAction,
    placeholder: &str,
    report: &mut SanitizeReport,
) {
    match action {
        SanitizeAction::Strip => {
            metadata.remove(key);
            report.fields_stripped += 1;
            report.stripped_keys.push(key.to_string());
        }
        SanitizeAction::Redact => {
            metadata.insert(
                key.to_string(),
                MetadataValue::Text(placeholder.to_string()),
            );
            report.fields_redacted += 1;
            report.redacted_keys.push(key.to_string());
        }
        SanitizeAction::ReplaceWith(replacement) => {
            metadata.insert(key.to_string(), MetadataValue::Text(replacement.clone()));
            report.fields_redacted += 1;
            report.redacted_keys.push(key.to_string());
        }
        SanitizeAction::Truncate(max_len) => {
            if let Some(MetadataValue::Text(text)) = metadata.get(key) {
                if text.len() > *max_len {
                    let truncated = text[..*max_len].to_string();
                    metadata.insert(key.to_string(), MetadataValue::Text(truncated));
                    report.fields_truncated += 1;
                    report.truncated_keys.push(key.to_string());
                }
            }
        }
        SanitizeAction::HashAnonymize => {
            if let Some(MetadataValue::Text(text)) = metadata.get(key) {
                let hashed = fnv1a_hash(text);
                metadata.insert(key.to_string(), MetadataValue::Text(hashed));
                report.fields_anonymized += 1;
                report.anonymized_keys.push(key.to_string());
            }
        }
    }
}

/// Create a privacy-focused policy that strips GPS, device info, and common PII.
pub fn privacy_policy() -> SanitizePolicy {
    let mut p = SanitizePolicy::new();
    p.enable_strip_gps();
    p.enable_strip_device_info();
    p.add_strip_field("email");
    p.add_strip_field("phone");
    p.add_strip_field("address");
    p
}

/// Create a distribution-safe policy (strips GPS + redacts author info).
pub fn distribution_policy() -> SanitizePolicy {
    let mut p = SanitizePolicy::new();
    p.enable_strip_gps();
    p.add_redact_field("author");
    p.add_redact_field("creator");
    p
}

/// Create a GDPR-compliant sanitization policy.
///
/// Strips all personal identifiable information, GPS data, and device
/// identifiers. Redacts author/creator fields. Designed for compliance
/// with EU General Data Protection Regulation.
pub fn gdpr_policy() -> SanitizePolicy {
    let mut p = SanitizePolicy::new();
    p.enable_strip_gps();
    p.enable_strip_device_info();
    p.enable_strip_personal_info();

    // Additional GDPR-specific rules
    p.add_rule(
        SanitizeRule::new(
            RuleMatch::Category(FieldCategory::PersonalInfo),
            SanitizeAction::Strip,
        )
        .with_description("GDPR: Remove all personal identifiable information"),
    );
    p.add_rule(
        SanitizeRule::new(
            RuleMatch::Contains("thumbnail".to_string()),
            SanitizeAction::Strip,
        )
        .with_description("GDPR: Remove embedded thumbnails that may contain faces"),
    );

    p
}

/// Create a broadcast-safe sanitization policy.
///
/// Preserves copyright and rights metadata, strips GPS and device info,
/// redacts personal info. Suitable for broadcast distribution workflows.
pub fn broadcast_policy() -> SanitizePolicy {
    let mut p = SanitizePolicy::new();
    p.enable_strip_gps();
    p.enable_strip_device_info();

    // Redact personal info instead of stripping for audit trail
    p.add_rule(
        SanitizeRule::new(
            RuleMatch::Category(FieldCategory::PersonalInfo),
            SanitizeAction::Redact,
        )
        .with_description("Broadcast: Redact personal info for distribution"),
    );

    p
}

/// Strip all metadata except for an explicit allow-list of keys.
pub fn strip_except(fields: &mut HashMap<String, String>, allow_keys: &HashSet<String>) -> usize {
    let before = fields.len();
    fields.retain(|k, _| allow_keys.contains(k));
    before - fields.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetadataFormat;

    fn sample_fields() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("title".into(), "My Video".into());
        m.insert("GPSLatitude".into(), "40.7128".into());
        m.insert("GPSLongitude".into(), "-74.0060".into());
        m.insert("CameraSerialNumber".into(), "ABC123".into());
        m.insert("author".into(), "John Doe".into());
        m.insert("email".into(), "john@example.com".into());
        m
    }

    fn sample_metadata() -> Metadata {
        let mut m = Metadata::new(MetadataFormat::Xmp);
        m.insert("title".into(), MetadataValue::Text("My Video".into()));
        m.insert("GPSLatitude".into(), MetadataValue::Text("40.7128".into()));
        m.insert(
            "GPSLongitude".into(),
            MetadataValue::Text("-74.0060".into()),
        );
        m.insert(
            "CameraSerialNumber".into(),
            MetadataValue::Text("ABC123".into()),
        );
        m.insert("author".into(), MetadataValue::Text("John Doe".into()));
        m.insert(
            "email".into(),
            MetadataValue::Text("john@example.com".into()),
        );
        m
    }

    // ---- Legacy API tests ----

    #[test]
    fn test_default_policy_no_changes() {
        let policy = SanitizePolicy::new();
        let mut fields = sample_fields();
        let orig_len = fields.len();
        let report = sanitize(&mut fields, &policy);
        assert_eq!(report.total_affected(), 0);
        assert_eq!(fields.len(), orig_len);
    }

    #[test]
    fn test_strip_specific_field() {
        let mut policy = SanitizePolicy::new();
        policy.add_strip_field("email");
        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert_eq!(report.fields_stripped, 1);
        assert!(!fields.contains_key("email"));
    }

    #[test]
    fn test_strip_gps() {
        let mut policy = SanitizePolicy::new();
        policy.enable_strip_gps();
        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert!(report.fields_stripped >= 2);
        assert!(!fields.contains_key("GPSLatitude"));
        assert!(!fields.contains_key("GPSLongitude"));
    }

    #[test]
    fn test_strip_device_info() {
        let mut policy = SanitizePolicy::new();
        policy.enable_strip_device_info();
        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert!(report.fields_stripped >= 1);
        assert!(!fields.contains_key("CameraSerialNumber"));
    }

    #[test]
    fn test_redact_field() {
        let mut policy = SanitizePolicy::new();
        policy.add_redact_field("author");
        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert_eq!(report.fields_redacted, 1);
        assert_eq!(
            fields.get("author").expect("should succeed in test"),
            "[REDACTED]"
        );
    }

    #[test]
    fn test_custom_redact_placeholder() {
        let mut policy = SanitizePolicy::new();
        policy.add_redact_field("author");
        policy.set_redact_placeholder("***");
        let mut fields = sample_fields();
        sanitize(&mut fields, &policy);
        assert_eq!(fields.get("author").expect("should succeed in test"), "***");
    }

    #[test]
    fn test_max_value_length_truncation() {
        let mut policy = SanitizePolicy::new();
        policy.set_max_value_length(5);
        let mut fields = HashMap::new();
        fields.insert("long".into(), "abcdefghij".into());
        fields.insert("short".into(), "ab".into());
        let report = sanitize(&mut fields, &policy);
        assert_eq!(report.fields_truncated, 1);
        assert_eq!(fields.get("long").expect("should succeed in test"), "abcde");
        assert_eq!(fields.get("short").expect("should succeed in test"), "ab");
    }

    #[test]
    fn test_privacy_policy() {
        let policy = privacy_policy();
        assert!(policy.strip_gps());
        assert!(policy.strip_device_info());
        assert!(policy.strip_fields().contains("email"));
    }

    #[test]
    fn test_distribution_policy() {
        let policy = distribution_policy();
        assert!(policy.strip_gps());
        assert!(policy.redact_fields().contains("author"));
    }

    #[test]
    fn test_strip_except() {
        let mut fields = sample_fields();
        let mut allow = HashSet::new();
        allow.insert("title".into());
        let removed = strip_except(&mut fields, &allow);
        assert!(removed >= 4);
        assert_eq!(fields.len(), 1);
        assert!(fields.contains_key("title"));
    }

    #[test]
    fn test_combined_strip_and_redact() {
        let mut policy = SanitizePolicy::new();
        policy.add_strip_field("email");
        policy.add_redact_field("author");
        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert_eq!(report.fields_stripped, 1);
        assert_eq!(report.fields_redacted, 1);
        assert!(!fields.contains_key("email"));
        assert_eq!(
            fields.get("author").expect("should succeed in test"),
            "[REDACTED]"
        );
    }

    #[test]
    fn test_report_total_affected() {
        let mut policy = SanitizePolicy::new();
        policy.enable_strip_gps();
        policy.add_redact_field("author");
        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert!(report.total_affected() >= 3);
    }

    #[test]
    fn test_matches_patterns_case_insensitive() {
        assert!(matches_patterns("myGpsField", GPS_FIELD_PATTERNS));
        assert!(matches_patterns("LATITUDE_INFO", GPS_FIELD_PATTERNS));
        assert!(!matches_patterns("title", GPS_FIELD_PATTERNS));
    }

    #[test]
    fn test_empty_fields_no_panic() {
        let policy = privacy_policy();
        let mut fields = HashMap::new();
        let report = sanitize(&mut fields, &policy);
        assert_eq!(report.total_affected(), 0);
    }

    // ---- Configurable rules tests ----

    #[test]
    fn test_rule_exact_match() {
        let rule = SanitizeRule::new(
            RuleMatch::Exact("GPSLatitude".to_string()),
            SanitizeAction::Strip,
        );
        assert!(rule.matches("GPSLatitude"));
        assert!(rule.matches("gpslatitude")); // case-insensitive
        assert!(!rule.matches("GPSLongitude"));
    }

    #[test]
    fn test_rule_contains_match() {
        let rule = SanitizeRule::new(
            RuleMatch::Contains("serial".to_string()),
            SanitizeAction::Strip,
        );
        assert!(rule.matches("CameraSerialNumber"));
        assert!(rule.matches("LensSerialNumber"));
        assert!(!rule.matches("title"));
    }

    #[test]
    fn test_rule_prefix_match() {
        let rule = SanitizeRule::new(RuleMatch::Prefix("GPS".to_string()), SanitizeAction::Strip);
        assert!(rule.matches("GPSLatitude"));
        assert!(rule.matches("gpsLongitude"));
        assert!(!rule.matches("myGPSField")); // prefix, not contains
    }

    #[test]
    fn test_rule_category_match() {
        let rule = SanitizeRule::new(
            RuleMatch::Category(FieldCategory::Gps),
            SanitizeAction::Strip,
        );
        assert!(rule.matches("GPSLatitude"));
        assert!(rule.matches("my_longitude_field"));
        assert!(!rule.matches("title"));
    }

    #[test]
    fn test_rule_with_description() {
        let rule = SanitizeRule::new(RuleMatch::Exact("email".into()), SanitizeAction::Strip)
            .with_description("Remove email for privacy");
        assert_eq!(
            rule.description.as_deref(),
            Some("Remove email for privacy")
        );
    }

    #[test]
    fn test_rule_hash_anonymize() {
        let mut policy = SanitizePolicy::new();
        policy.add_rule(SanitizeRule::new(
            RuleMatch::Exact("author".into()),
            SanitizeAction::HashAnonymize,
        ));

        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert_eq!(report.fields_anonymized, 1);

        let anonymized = fields.get("author").expect("author should exist");
        assert!(anonymized.starts_with("anon_"));
        assert_ne!(anonymized, "John Doe");
    }

    #[test]
    fn test_rule_replace_with() {
        let mut policy = SanitizePolicy::new();
        policy.add_rule(SanitizeRule::new(
            RuleMatch::Exact("author".into()),
            SanitizeAction::ReplaceWith("Anonymous".into()),
        ));

        let mut fields = sample_fields();
        sanitize(&mut fields, &policy);
        assert_eq!(fields.get("author").map(|s| s.as_str()), Some("Anonymous"));
    }

    #[test]
    fn test_rule_truncate() {
        let mut policy = SanitizePolicy::new();
        policy.add_rule(SanitizeRule::new(
            RuleMatch::Exact("email".into()),
            SanitizeAction::Truncate(4),
        ));

        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert_eq!(report.fields_truncated, 1);
        assert_eq!(fields.get("email").map(|s| s.as_str()), Some("john"));
    }

    #[test]
    fn test_rules_first_match_wins() {
        let mut policy = SanitizePolicy::new();
        // First rule: strip anything containing "GPS"
        policy.add_rule(SanitizeRule::new(
            RuleMatch::Contains("GPS".into()),
            SanitizeAction::Strip,
        ));
        // Second rule: redact anything containing "GPS" (should never fire)
        policy.add_rule(SanitizeRule::new(
            RuleMatch::Contains("GPS".into()),
            SanitizeAction::Redact,
        ));

        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        // GPS fields should be stripped, not redacted
        assert!(!fields.contains_key("GPSLatitude"));
        assert!(
            report.fields_redacted == 0
                || !report.redacted_keys.contains(&"GPSLatitude".to_string())
        );
    }

    #[test]
    fn test_allow_list_mode() {
        let mut policy = SanitizePolicy::new();
        policy.add_allow_key("title");
        policy.add_allow_key("author");

        let mut fields = sample_fields();
        let report = sanitize(&mut fields, &policy);
        assert_eq!(fields.len(), 2);
        assert!(fields.contains_key("title"));
        assert!(fields.contains_key("author"));
        assert!(report.fields_stripped >= 4);
    }

    #[test]
    fn test_strip_personal_info() {
        let mut policy = SanitizePolicy::new();
        policy.enable_strip_personal_info();

        let mut fields = sample_fields();
        sanitize(&mut fields, &policy);
        assert!(!fields.contains_key("email"));
        assert!(!fields.contains_key("author"));
    }

    // ---- Metadata integration tests ----

    #[test]
    fn test_sanitize_metadata_strip_gps() {
        let mut policy = SanitizePolicy::new();
        policy.enable_strip_gps();

        let mut metadata = sample_metadata();
        let report = sanitize_metadata(&mut metadata, &policy);
        assert!(report.fields_stripped >= 2);
        assert!(!metadata.contains("GPSLatitude"));
        assert!(!metadata.contains("GPSLongitude"));
        assert!(metadata.contains("title"));
    }

    #[test]
    fn test_sanitize_metadata_redact() {
        let mut policy = SanitizePolicy::new();
        policy.add_redact_field("author");

        let mut metadata = sample_metadata();
        sanitize_metadata(&mut metadata, &policy);
        assert_eq!(
            metadata.get("author").and_then(|v| v.as_text()),
            Some("[REDACTED]")
        );
    }

    #[test]
    fn test_sanitize_metadata_rule_strip() {
        let mut policy = SanitizePolicy::new();
        policy.add_rule(SanitizeRule::new(
            RuleMatch::Category(FieldCategory::Gps),
            SanitizeAction::Strip,
        ));

        let mut metadata = sample_metadata();
        let report = sanitize_metadata(&mut metadata, &policy);
        assert!(!metadata.contains("GPSLatitude"));
        assert!(!metadata.contains("GPSLongitude"));
        assert!(report.rule_matches.len() >= 2);
    }

    #[test]
    fn test_sanitize_metadata_rule_hash() {
        let mut policy = SanitizePolicy::new();
        policy.add_rule(SanitizeRule::new(
            RuleMatch::Exact("email".into()),
            SanitizeAction::HashAnonymize,
        ));

        let mut metadata = sample_metadata();
        let report = sanitize_metadata(&mut metadata, &policy);
        assert_eq!(report.fields_anonymized, 1);
        let val = metadata
            .get("email")
            .and_then(|v| v.as_text())
            .expect("email should exist");
        assert!(val.starts_with("anon_"));
    }

    #[test]
    fn test_sanitize_metadata_truncate() {
        let mut policy = SanitizePolicy::new();
        policy.set_max_value_length(5);

        let mut metadata = sample_metadata();
        let report = sanitize_metadata(&mut metadata, &policy);
        assert!(report.fields_truncated > 0);
        // "My Video" -> "My Vi" (truncated to 5)
        let title = metadata
            .get("title")
            .and_then(|v| v.as_text())
            .expect("title");
        assert_eq!(title.len(), 5);
    }

    #[test]
    fn test_sanitize_metadata_allow_list() {
        let mut policy = SanitizePolicy::new();
        policy.add_allow_key("title");

        let mut metadata = sample_metadata();
        let report = sanitize_metadata(&mut metadata, &policy);
        assert!(report.fields_stripped >= 5);
        assert!(metadata.contains("title"));
        assert!(!metadata.contains("GPSLatitude"));
        assert!(!metadata.contains("author"));
    }

    #[test]
    fn test_sanitize_metadata_rule_replace_with() {
        let mut policy = SanitizePolicy::new();
        policy.add_rule(SanitizeRule::new(
            RuleMatch::Exact("author".into()),
            SanitizeAction::ReplaceWith("Redacted Author".into()),
        ));

        let mut metadata = sample_metadata();
        sanitize_metadata(&mut metadata, &policy);
        assert_eq!(
            metadata.get("author").and_then(|v| v.as_text()),
            Some("Redacted Author")
        );
    }

    // ---- Preset policy tests ----

    #[test]
    fn test_gdpr_policy() {
        let policy = gdpr_policy();
        assert!(policy.strip_gps());
        assert!(policy.strip_device_info());
        assert!(policy.strip_personal_info());
        assert!(!policy.rules().is_empty());
    }

    #[test]
    fn test_gdpr_policy_strips_all_pii() {
        let policy = gdpr_policy();
        let mut metadata = sample_metadata();
        sanitize_metadata(&mut metadata, &policy);
        assert!(!metadata.contains("GPSLatitude"));
        assert!(!metadata.contains("email"));
        assert!(!metadata.contains("CameraSerialNumber"));
    }

    #[test]
    fn test_broadcast_policy() {
        let policy = broadcast_policy();
        assert!(policy.strip_gps());
        assert!(policy.strip_device_info());
        assert!(!policy.rules().is_empty());
    }

    #[test]
    fn test_broadcast_policy_preserves_rights() {
        let policy = broadcast_policy();
        let mut metadata = Metadata::new(MetadataFormat::Xmp);
        metadata.insert(
            "copyright".into(),
            MetadataValue::Text("(c) 2026 Studio".into()),
        );
        metadata.insert("GPSLatitude".into(), MetadataValue::Text("40.7128".into()));

        sanitize_metadata(&mut metadata, &policy);
        // GPS should be stripped
        assert!(!metadata.contains("GPSLatitude"));
        // Copyright should be preserved (broadcast policy only strips GPS/device)
        assert!(metadata.contains("copyright"));
    }

    // ---- FNV-1a hash tests ----

    #[test]
    fn test_fnv1a_hash_deterministic() {
        let h1 = fnv1a_hash("test");
        let h2 = fnv1a_hash("test");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_hash_different_inputs() {
        let h1 = fnv1a_hash("hello");
        let h2 = fnv1a_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fnv1a_hash_format() {
        let h = fnv1a_hash("test");
        assert!(h.starts_with("anon_"));
        assert_eq!(h.len(), 5 + 16); // "anon_" + 16 hex chars
    }

    // ---- Field category tests ----

    #[test]
    fn test_field_category_gps_patterns() {
        let patterns = patterns_for_category(FieldCategory::Gps);
        assert!(patterns.contains(&"latitude"));
        assert!(patterns.contains(&"longitude"));
    }

    #[test]
    fn test_field_category_personal_patterns() {
        let patterns = patterns_for_category(FieldCategory::PersonalInfo);
        assert!(patterns.contains(&"email"));
        assert!(patterns.contains(&"phone"));
    }

    #[test]
    fn test_field_category_rights_patterns() {
        let patterns = patterns_for_category(FieldCategory::Rights);
        assert!(patterns.contains(&"copyright"));
        assert!(patterns.contains(&"license"));
    }

    #[test]
    fn test_field_category_temporal_patterns() {
        let patterns = patterns_for_category(FieldCategory::Temporal);
        assert!(patterns.contains(&"datetime"));
        assert!(patterns.contains(&"createdate"));
    }

    #[test]
    fn test_field_category_technical_patterns() {
        let patterns = patterns_for_category(FieldCategory::Technical);
        assert!(patterns.contains(&"encoder"));
        assert!(patterns.contains(&"bitrate"));
    }

    #[test]
    fn test_rule_category_device_info() {
        let rule = SanitizeRule::new(
            RuleMatch::Category(FieldCategory::DeviceInfo),
            SanitizeAction::Strip,
        );
        assert!(rule.matches("CameraSerialNumber"));
        assert!(rule.matches("LensMake"));
        assert!(!rule.matches("title"));
    }
}
