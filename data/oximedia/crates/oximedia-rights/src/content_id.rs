//! Content identifier module for linking ISRC, ISAN, EIDR, and other
//! industry-standard IDs to rights records.
//!
//! # Supported identifier types
//!
//! | Type | Example | Authority |
//! |------|---------|-----------|
//! | ISRC | `US-ABC-23-00001` | IFPI / RIAA |
//! | ISAN | `0000-0000-8947-0000-8-0000-0000-D` | ISAN-IA |
//! | EIDR | `10.5240/7791-8534-2C23-9030-8610-5` | EIDR |
//! | UPC  | `012345678905` | GS1 |
//! | ISBN | `978-3-16-148410-0` | ISBN International Agency |
//! | DOI  | `10.1000/xyz123` | International DOI Foundation |
//! | Custom | any string | user-defined |

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── ContentIdKind ────────────────────────────────────────────────────────────

/// The kind / scheme of a content identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentIdKind {
    /// International Standard Recording Code (audio recordings).
    Isrc,
    /// International Standard Audiovisual Number (audiovisual works).
    Isan,
    /// Entertainment Identifier Registry (film & TV).
    Eidr,
    /// Universal Product Code (physical / digital releases).
    Upc,
    /// International Standard Book Number.
    Isbn,
    /// Digital Object Identifier.
    Doi,
    /// User-defined / proprietary identifier scheme.
    Custom(String),
}

impl ContentIdKind {
    /// Human-readable name of the identifier scheme.
    #[must_use]
    pub fn scheme_name(&self) -> &str {
        match self {
            Self::Isrc => "ISRC",
            Self::Isan => "ISAN",
            Self::Eidr => "EIDR",
            Self::Upc => "UPC",
            Self::Isbn => "ISBN",
            Self::Doi => "DOI",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ── ContentId ────────────────────────────────────────────────────────────────

/// A single content identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentId {
    /// Identifier scheme.
    pub kind: ContentIdKind,
    /// The identifier value (normalised to uppercase for ISRC/ISAN/EIDR).
    pub value: String,
    /// Optional human-readable label (e.g. the title of the work).
    pub label: Option<String>,
}

impl ContentId {
    /// Create a new content ID with optional label.
    #[must_use]
    pub fn new(kind: ContentIdKind, value: impl Into<String>) -> Self {
        let raw: String = value.into();
        let normalised = Self::normalise(&kind, &raw);
        Self {
            kind,
            value: normalised,
            label: None,
        }
    }

    /// Builder: attach a human-readable label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Normalise the identifier value according to scheme conventions.
    fn normalise(kind: &ContentIdKind, value: &str) -> String {
        match kind {
            ContentIdKind::Isrc | ContentIdKind::Isan | ContentIdKind::Eidr => value.to_uppercase(),
            _ => value.to_string(),
        }
    }

    /// Perform basic structural validation.
    ///
    /// - **ISRC**: `CC-XXX-YY-NNNNN` (12 chars when dashes removed, no deep
    ///   check-digit verification).
    /// - **ISAN**: 26 hex chars + dashes (simplified, no check-digit).
    /// - **EIDR**: must start with `10.5240/`.
    /// - **UPC**: 12 decimal digits.
    /// - **ISBN**: 13 decimal digits (stripped of dashes/spaces).
    /// - **DOI**: must start with `10.`.
    /// - **Custom**: always valid.
    #[must_use]
    pub fn is_structurally_valid(&self) -> bool {
        match &self.kind {
            ContentIdKind::Isrc => {
                let stripped: String = self.value.chars().filter(|c| *c != '-').collect();
                stripped.len() == 12 && stripped.chars().all(|c| c.is_alphanumeric())
            }
            ContentIdKind::Isan => {
                // ISAN has the format XXXX-XXXX-XXXX-XXXX-X-XXXX-XXXX-X (with optional check)
                let stripped: String = self.value.chars().filter(|c| *c != '-').collect();
                stripped.len() >= 16 && stripped.chars().all(|c| c.is_ascii_hexdigit())
            }
            ContentIdKind::Eidr => self.value.starts_with("10.5240/"),
            ContentIdKind::Upc => {
                self.value.len() == 12 && self.value.chars().all(|c| c.is_ascii_digit())
            }
            ContentIdKind::Isbn => {
                let stripped: String = self.value.chars().filter(|c| c.is_ascii_digit()).collect();
                stripped.len() == 13 && stripped.chars().all(|c| c.is_ascii_digit())
            }
            ContentIdKind::Doi => self.value.starts_with("10."),
            ContentIdKind::Custom(_) => true,
        }
    }
}

// ── RightsRecord binding ─────────────────────────────────────────────────────

/// A binding between a rights record ID and one or more content identifiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentIdBinding {
    /// Internal rights record ID.
    pub record_id: String,
    /// Content identifiers linked to this record.
    pub identifiers: Vec<ContentId>,
    /// Additional free-form metadata (key → value).
    pub metadata: HashMap<String, String>,
}

impl ContentIdBinding {
    /// Create a new binding for a record with no identifiers.
    #[must_use]
    pub fn new(record_id: impl Into<String>) -> Self {
        Self {
            record_id: record_id.into(),
            identifiers: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Attach a content identifier to this binding.
    pub fn add_id(&mut self, id: ContentId) {
        self.identifiers.push(id);
    }

    /// Add metadata key/value.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Find identifiers of a specific kind.
    #[must_use]
    pub fn ids_of_kind(&self, kind: &ContentIdKind) -> Vec<&ContentId> {
        self.identifiers
            .iter()
            .filter(|id| &id.kind == kind)
            .collect()
    }

    /// Whether this binding contains at least one identifier of the given kind.
    #[must_use]
    pub fn has_kind(&self, kind: &ContentIdKind) -> bool {
        self.identifiers.iter().any(|id| &id.kind == kind)
    }

    /// Number of attached identifiers.
    #[must_use]
    pub fn id_count(&self) -> usize {
        self.identifiers.len()
    }
}

// ── ContentIdRegistry ────────────────────────────────────────────────────────

/// In-memory registry that maps content identifiers to rights record IDs and
/// vice-versa.
#[derive(Debug, Default)]
pub struct ContentIdRegistry {
    /// Bindings keyed by record ID.
    bindings: HashMap<String, ContentIdBinding>,
    /// Reverse index: normalised identifier value → list of record IDs.
    reverse: HashMap<String, Vec<String>>,
}

impl ContentIdRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a binding.  Replaces any existing binding for the same record.
    pub fn register(&mut self, binding: ContentIdBinding) {
        // Remove old reverse-index entries if replacing.
        if let Some(old) = self.bindings.get(&binding.record_id) {
            for id in &old.identifiers {
                if let Some(ids) = self.reverse.get_mut(&id.value) {
                    ids.retain(|r| r != &binding.record_id);
                }
            }
        }
        // Insert new reverse-index entries.
        for id in &binding.identifiers {
            self.reverse
                .entry(id.value.clone())
                .or_default()
                .push(binding.record_id.clone());
        }
        self.bindings.insert(binding.record_id.clone(), binding);
    }

    /// Add a single content identifier to an existing binding, creating the
    /// binding if it does not yet exist.
    pub fn add_id_to_record(&mut self, record_id: &str, id: ContentId) {
        let value = id.value.clone();
        let binding = self
            .bindings
            .entry(record_id.to_string())
            .or_insert_with(|| ContentIdBinding::new(record_id));
        binding.add_id(id);
        self.reverse
            .entry(value)
            .or_default()
            .push(record_id.to_string());
    }

    /// Retrieve the binding for a specific rights record.
    #[must_use]
    pub fn get_binding(&self, record_id: &str) -> Option<&ContentIdBinding> {
        self.bindings.get(record_id)
    }

    /// Look up all rights record IDs linked to a specific identifier value.
    #[must_use]
    pub fn lookup_by_id_value(&self, value: &str) -> Vec<&str> {
        let normalised = value.to_uppercase();
        self.reverse
            .get(&normalised)
            .map(|ids| ids.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Find all bindings that have at least one identifier of the given kind.
    #[must_use]
    pub fn find_by_kind(&self, kind: &ContentIdKind) -> Vec<&ContentIdBinding> {
        self.bindings
            .values()
            .filter(|b| b.has_kind(kind))
            .collect()
    }

    /// Total number of registered bindings.
    #[must_use]
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Remove a binding and its reverse-index entries.
    pub fn remove(&mut self, record_id: &str) -> Option<ContentIdBinding> {
        if let Some(binding) = self.bindings.remove(record_id) {
            for id in &binding.identifiers {
                if let Some(ids) = self.reverse.get_mut(&id.value) {
                    ids.retain(|r| r != record_id);
                }
            }
            Some(binding)
        } else {
            None
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn isrc_id() -> ContentId {
        ContentId::new(ContentIdKind::Isrc, "US-ABC-23-00001")
    }

    fn eidr_id() -> ContentId {
        ContentId::new(ContentIdKind::Eidr, "10.5240/7791-8534-2C23-9030-8610-5")
    }

    fn isan_id() -> ContentId {
        ContentId::new(ContentIdKind::Isan, "0000-0000-8947-0000-8-0000-0000-D")
    }

    // ── ContentIdKind ──

    #[test]
    fn test_scheme_names() {
        assert_eq!(ContentIdKind::Isrc.scheme_name(), "ISRC");
        assert_eq!(ContentIdKind::Isan.scheme_name(), "ISAN");
        assert_eq!(ContentIdKind::Eidr.scheme_name(), "EIDR");
        assert_eq!(ContentIdKind::Upc.scheme_name(), "UPC");
        assert_eq!(ContentIdKind::Isbn.scheme_name(), "ISBN");
        assert_eq!(ContentIdKind::Doi.scheme_name(), "DOI");
        assert_eq!(
            ContentIdKind::Custom("MyScheme".into()).scheme_name(),
            "MyScheme"
        );
    }

    // ── ContentId validation ──

    #[test]
    fn test_isrc_valid() {
        // 12 chars without dashes: US + ABC + 23 + 00001 = 2+3+2+5 = 12
        let id = ContentId::new(ContentIdKind::Isrc, "USABC2300001");
        assert!(id.is_structurally_valid());
    }

    #[test]
    fn test_isrc_with_dashes_valid() {
        let id = isrc_id();
        assert!(id.is_structurally_valid());
    }

    #[test]
    fn test_isrc_too_short() {
        let id = ContentId::new(ContentIdKind::Isrc, "US-ABC-23-000");
        assert!(!id.is_structurally_valid());
    }

    #[test]
    fn test_eidr_valid() {
        let id = eidr_id();
        assert!(id.is_structurally_valid());
    }

    #[test]
    fn test_eidr_invalid_prefix() {
        let id = ContentId::new(ContentIdKind::Eidr, "10.9999/abc");
        assert!(!id.is_structurally_valid());
    }

    #[test]
    fn test_upc_valid() {
        let id = ContentId::new(ContentIdKind::Upc, "012345678905");
        assert!(id.is_structurally_valid());
    }

    #[test]
    fn test_upc_invalid_length() {
        let id = ContentId::new(ContentIdKind::Upc, "01234");
        assert!(!id.is_structurally_valid());
    }

    #[test]
    fn test_isbn_valid() {
        // ISBN-13 digits only
        let id = ContentId::new(ContentIdKind::Isbn, "9783161484100");
        assert!(id.is_structurally_valid());
    }

    #[test]
    fn test_isbn_with_dashes_valid() {
        let id = ContentId::new(ContentIdKind::Isbn, "978-3-16-148410-0");
        assert!(id.is_structurally_valid());
    }

    #[test]
    fn test_doi_valid() {
        let id = ContentId::new(ContentIdKind::Doi, "10.1000/xyz123");
        assert!(id.is_structurally_valid());
    }

    #[test]
    fn test_doi_invalid() {
        let id = ContentId::new(ContentIdKind::Doi, "11.1000/xyz");
        assert!(!id.is_structurally_valid());
    }

    #[test]
    fn test_custom_always_valid() {
        let id = ContentId::new(ContentIdKind::Custom("MY".into()), "anything");
        assert!(id.is_structurally_valid());
    }

    #[test]
    fn test_isrc_normalised_uppercase() {
        let id = ContentId::new(ContentIdKind::Isrc, "us-abc-23-00001");
        assert_eq!(id.value, "US-ABC-23-00001");
    }

    // ── ContentIdBinding ──

    #[test]
    fn test_binding_add_and_count() {
        let mut b = ContentIdBinding::new("rec-1");
        b.add_id(isrc_id());
        b.add_id(eidr_id());
        assert_eq!(b.id_count(), 2);
    }

    #[test]
    fn test_binding_ids_of_kind() {
        let mut b = ContentIdBinding::new("rec-2");
        b.add_id(isrc_id());
        b.add_id(eidr_id());
        let isrcs = b.ids_of_kind(&ContentIdKind::Isrc);
        assert_eq!(isrcs.len(), 1);
    }

    #[test]
    fn test_binding_has_kind() {
        let mut b = ContentIdBinding::new("rec-3");
        b.add_id(isrc_id());
        assert!(b.has_kind(&ContentIdKind::Isrc));
        assert!(!b.has_kind(&ContentIdKind::Eidr));
    }

    #[test]
    fn test_binding_metadata() {
        let mut b = ContentIdBinding::new("rec-4");
        b.add_metadata("title", "My Song");
        assert_eq!(b.metadata.get("title").map(String::as_str), Some("My Song"));
    }

    // ── ContentIdRegistry ──

    #[test]
    fn test_registry_register_and_lookup() {
        let mut reg = ContentIdRegistry::new();
        let mut binding = ContentIdBinding::new("rec-A");
        binding.add_id(isrc_id());
        reg.register(binding);

        let found = reg.get_binding("rec-A");
        assert!(found.is_some());
        assert_eq!(found.expect("binding should exist").id_count(), 1);
    }

    #[test]
    fn test_registry_lookup_by_id_value() {
        let mut reg = ContentIdRegistry::new();
        let mut b1 = ContentIdBinding::new("rec-B");
        b1.add_id(isrc_id());
        reg.register(b1);

        let records = reg.lookup_by_id_value("US-ABC-23-00001");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0], "rec-B");
    }

    #[test]
    fn test_registry_lookup_case_insensitive() {
        let mut reg = ContentIdRegistry::new();
        let mut b = ContentIdBinding::new("rec-C");
        b.add_id(isrc_id()); // stored as "US-ABC-23-00001"
        reg.register(b);

        let records = reg.lookup_by_id_value("us-abc-23-00001");
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn test_registry_find_by_kind() {
        let mut reg = ContentIdRegistry::new();
        let mut b1 = ContentIdBinding::new("r1");
        b1.add_id(isrc_id());
        reg.register(b1);

        let mut b2 = ContentIdBinding::new("r2");
        b2.add_id(eidr_id());
        reg.register(b2);

        let isrc_bindings = reg.find_by_kind(&ContentIdKind::Isrc);
        assert_eq!(isrc_bindings.len(), 1);
        assert_eq!(isrc_bindings[0].record_id, "r1");
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = ContentIdRegistry::new();
        let mut b = ContentIdBinding::new("rec-D");
        b.add_id(isrc_id());
        reg.register(b);
        assert_eq!(reg.binding_count(), 1);

        let removed = reg.remove("rec-D");
        assert!(removed.is_some());
        assert_eq!(reg.binding_count(), 0);

        // Reverse index should also be cleaned
        assert!(reg.lookup_by_id_value("US-ABC-23-00001").is_empty());
    }

    #[test]
    fn test_registry_add_id_to_record_creates_binding() {
        let mut reg = ContentIdRegistry::new();
        reg.add_id_to_record("rec-E", isrc_id());
        assert_eq!(reg.binding_count(), 1);
        assert_eq!(
            reg.get_binding("rec-E")
                .expect("binding should exist")
                .id_count(),
            1
        );
    }

    #[test]
    fn test_registry_add_multiple_ids_to_same_record() {
        let mut reg = ContentIdRegistry::new();
        reg.add_id_to_record("rec-F", isrc_id());
        reg.add_id_to_record("rec-F", eidr_id());
        assert_eq!(
            reg.get_binding("rec-F")
                .expect("binding should exist")
                .id_count(),
            2
        );
    }

    #[test]
    fn test_registry_replace_binding_updates_reverse_index() {
        let mut reg = ContentIdRegistry::new();

        // First binding: ISRC
        let mut b1 = ContentIdBinding::new("rec-G");
        b1.add_id(isrc_id());
        reg.register(b1);

        // Replace with binding that has only EIDR
        let mut b2 = ContentIdBinding::new("rec-G");
        b2.add_id(eidr_id());
        reg.register(b2);

        // Old ISRC should no longer resolve
        assert!(reg.lookup_by_id_value("US-ABC-23-00001").is_empty());
        // New EIDR should resolve
        assert_eq!(
            reg.lookup_by_id_value("10.5240/7791-8534-2C23-9030-8610-5")
                .len(),
            1
        );
    }

    #[test]
    fn test_isan_valid() {
        let id = isan_id();
        assert!(id.is_structurally_valid());
    }
}
