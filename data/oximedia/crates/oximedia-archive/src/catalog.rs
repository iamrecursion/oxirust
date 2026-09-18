//! Archive catalog management.
//!
//! Provides catalog entries, access control, full-text search, date-range
//! search, CSV import/export, and OAI-PMH XML export.

#![allow(dead_code)]

/// Access control level for catalog entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessLevel {
    /// Freely available to all.
    Public,
    /// Available to authenticated users.
    Restricted,
    /// Sensitive; accessible only to named individuals.
    Confidential,
    /// Internal staff use only.
    Internal,
}

impl AccessLevel {
    /// Returns `true` if a user with the given `user_role` may access this level.
    ///
    /// Role hierarchy (most privileged first): `"admin"`, `"staff"`, `"user"`, anything else.
    #[must_use]
    pub fn can_access(&self, user_role: &str) -> bool {
        match self {
            Self::Public => true,
            Self::Restricted => matches!(user_role, "admin" | "staff" | "user"),
            Self::Internal => matches!(user_role, "admin" | "staff"),
            Self::Confidential => user_role == "admin",
        }
    }

    /// Short string label.
    #[must_use]
    pub const fn label(&self) -> &str {
        match self {
            Self::Public => "public",
            Self::Restricted => "restricted",
            Self::Confidential => "confidential",
            Self::Internal => "internal",
        }
    }
}

/// A single catalog record describing a media asset.
#[derive(Clone, Debug)]
pub struct CatalogEntry {
    /// Unique identifier.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// Extended description.
    pub description: String,
    /// Creation timestamp (Unix milliseconds).
    pub date_created_ms: u64,
    /// File format identifier (e.g., `"dpx"`, `"mp4"`).
    pub format: String,
    /// Duration in seconds, if applicable.
    pub duration_secs: Option<f64>,
    /// Physical shelf location, if applicable.
    pub physical_location: Option<String>,
    /// Path or URI to the digital file.
    pub digital_path: Option<String>,
    /// Rights statement or licence.
    pub rights: String,
    /// Access control level.
    pub access_level: AccessLevel,
}

impl CatalogEntry {
    /// Create a new catalog entry with mandatory fields.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        format: impl Into<String>,
        date_created_ms: u64,
        rights: impl Into<String>,
        access_level: AccessLevel,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: String::new(),
            date_created_ms,
            format: format.into(),
            duration_secs: None,
            physical_location: None,
            digital_path: None,
            rights: rights.into(),
            access_level,
        }
    }
}

/// In-memory catalog index supporting search operations.
#[derive(Default)]
pub struct CatalogIndex {
    entries: Vec<CatalogEntry>,
}

impl CatalogIndex {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the index.
    pub fn add(&mut self, entry: CatalogEntry) {
        self.entries.push(entry);
    }

    /// Search entries whose `title` contains `query` (case-insensitive).
    #[must_use]
    pub fn search_by_title(&self, query: &str) -> Vec<&CatalogEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.title.to_lowercase().contains(&q))
            .collect()
    }

    /// Return entries created within the given millisecond timestamp range (inclusive).
    #[must_use]
    pub fn search_by_date_range(&self, start_ms: u64, end_ms: u64) -> Vec<&CatalogEntry> {
        self.entries
            .iter()
            .filter(|e| e.date_created_ms >= start_ms && e.date_created_ms <= end_ms)
            .collect()
    }

    /// Total number of entries in the index.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    /// Look up an entry by its exact ID.
    #[must_use]
    pub fn get_by_id(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }
}

/// Catalog export utilities.
pub struct CatalogExport;

impl CatalogExport {
    /// Export entries to CSV format.
    ///
    /// Columns: id, title, format, date_created_ms, duration_secs, rights, access_level
    #[must_use]
    pub fn to_csv(entries: &[CatalogEntry]) -> String {
        let mut out =
            String::from("id,title,format,date_created_ms,duration_secs,rights,access_level\n");
        for e in entries {
            let duration = e.duration_secs.map(|d| d.to_string()).unwrap_or_default();
            out.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                csv_escape(&e.id),
                csv_escape(&e.title),
                csv_escape(&e.format),
                e.date_created_ms,
                duration,
                csv_escape(&e.rights),
                e.access_level.label(),
            ));
        }
        out
    }

    /// Export entries as a minimal OAI-PMH `ListRecords` XML response.
    #[must_use]
    pub fn to_oai_pmh(entries: &[CatalogEntry]) -> String {
        let mut out = String::from(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
             <OAI-PMH xmlns=\"http://www.openarchives.org/OAI/2.0/\">\n\
             <responseDate>2026-01-01T00:00:00Z</responseDate>\n\
             <request verb=\"ListRecords\"/>\n\
             <ListRecords>\n",
        );

        for e in entries {
            out.push_str("  <record>\n    <header>\n");
            out.push_str(&format!(
                "      <identifier>{}</identifier>\n",
                xml_escape(&e.id)
            ));
            out.push_str(&format!(
                "      <datestamp>{}</datestamp>\n",
                ms_to_iso8601(e.date_created_ms)
            ));
            out.push_str("    </header>\n    <metadata>\n      <oai_dc:dc\n");
            out.push_str("        xmlns:oai_dc=\"http://www.openarchives.org/OAI/2.0/oai_dc/\"\n");
            out.push_str("        xmlns:dc=\"http://purl.org/dc/elements/1.1/\">\n");
            out.push_str(&format!(
                "        <dc:title>{}</dc:title>\n",
                xml_escape(&e.title)
            ));
            if !e.description.is_empty() {
                out.push_str(&format!(
                    "        <dc:description>{}</dc:description>\n",
                    xml_escape(&e.description)
                ));
            }
            out.push_str(&format!(
                "        <dc:format>{}</dc:format>\n",
                xml_escape(&e.format)
            ));
            out.push_str(&format!(
                "        <dc:rights>{}</dc:rights>\n",
                xml_escape(&e.rights)
            ));
            out.push_str("      </oai_dc:dc>\n    </metadata>\n  </record>\n");
        }

        out.push_str("</ListRecords>\n</OAI-PMH>");
        out
    }
}

/// Catalog import utilities.
pub struct CatalogImport;

impl CatalogImport {
    /// Parse catalog entries from a CSV string.
    ///
    /// Expects the header row `id,title,format,date_created_ms,duration_secs,rights,access_level`.
    /// Lines that cannot be parsed are silently skipped.
    #[must_use]
    pub fn from_csv(csv: &str) -> Vec<CatalogEntry> {
        let mut entries = Vec::new();
        let mut lines = csv.lines();

        // Skip header
        if lines.next().is_none() {
            return entries;
        }

        for line in lines {
            let cols: Vec<&str> = line.splitn(7, ',').collect();
            if cols.len() < 7 {
                continue;
            }

            let id = csv_unescape(cols[0]);
            let title = csv_unescape(cols[1]);
            let format = csv_unescape(cols[2]);
            let date_created_ms: u64 = cols[3].trim().parse().unwrap_or(0);
            let duration_secs: Option<f64> = cols[4]
                .trim()
                .parse()
                .ok()
                .filter(|_| !cols[4].trim().is_empty());
            let rights = csv_unescape(cols[5]);
            let access_level = match cols[6].trim() {
                "public" => AccessLevel::Public,
                "restricted" => AccessLevel::Restricted,
                "confidential" => AccessLevel::Confidential,
                "internal" => AccessLevel::Internal,
                _ => AccessLevel::Public,
            };

            let mut entry =
                CatalogEntry::new(id, title, format, date_created_ms, rights, access_level);
            entry.duration_secs = duration_secs;
            entries.push(entry);
        }

        entries
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Wrap a CSV field in quotes if it contains a comma, newline, or quote.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('\n') || s.contains('"') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Strip surrounding quotes from a CSV field.
fn csv_unescape(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len() - 1].replace("\"\"", "\"")
    } else {
        s.to_string()
    }
}

/// Convert Unix milliseconds to a simplified ISO 8601 date string.
fn ms_to_iso8601(ms: u64) -> String {
    let secs = ms / 1_000;
    let days = secs / 86_400;
    let year = 1970 + days / 365;
    // Approximate month/day
    let day_of_year = days % 365;
    let month = day_of_year / 30 + 1;
    let day = day_of_year % 30 + 1;
    format!("{year:04}-{month:02}-{day:02}T00:00:00Z")
}

/// Minimal XML character escaping.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(id: &str, title: &str, ms: u64) -> CatalogEntry {
        CatalogEntry::new(id, title, "dpx", ms, "CC0", AccessLevel::Public)
    }

    #[test]
    fn test_access_level_public() {
        assert!(AccessLevel::Public.can_access("anyone"));
        assert!(AccessLevel::Public.can_access("guest"));
    }

    #[test]
    fn test_access_level_restricted() {
        assert!(AccessLevel::Restricted.can_access("user"));
        assert!(!AccessLevel::Restricted.can_access("guest"));
    }

    #[test]
    fn test_access_level_internal() {
        assert!(AccessLevel::Internal.can_access("staff"));
        assert!(!AccessLevel::Internal.can_access("user"));
    }

    #[test]
    fn test_access_level_confidential() {
        assert!(AccessLevel::Confidential.can_access("admin"));
        assert!(!AccessLevel::Confidential.can_access("staff"));
    }

    #[test]
    fn test_catalog_index_add_and_count() {
        let mut idx = CatalogIndex::new();
        idx.add(sample_entry("a1", "Sunset Reel", 1_000_000));
        idx.add(sample_entry("a2", "Night Scene", 2_000_000));
        assert_eq!(idx.total_count(), 2);
    }

    #[test]
    fn test_search_by_title() {
        let mut idx = CatalogIndex::new();
        idx.add(sample_entry("a1", "Sunset Reel", 1_000_000));
        idx.add(sample_entry("a2", "Night Scene", 2_000_000));
        idx.add(sample_entry("a3", "Sunset Beach", 3_000_000));

        let results = idx.search_by_title("sunset");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_date_range() {
        let mut idx = CatalogIndex::new();
        idx.add(sample_entry("a1", "A", 1_000));
        idx.add(sample_entry("a2", "B", 5_000));
        idx.add(sample_entry("a3", "C", 9_000));

        let results = idx.search_by_date_range(2_000, 8_000);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a2");
    }

    #[test]
    fn test_catalog_export_csv_header() {
        let entries = vec![sample_entry("id1", "My Film", 0)];
        let csv = CatalogExport::to_csv(&entries);
        assert!(csv.starts_with("id,title,format,"));
        assert!(csv.contains("My Film"));
    }

    #[test]
    fn test_catalog_export_oai_pmh() {
        let entries = vec![sample_entry("oai:1", "Test", 86_400_000)];
        let xml = CatalogExport::to_oai_pmh(&entries);
        assert!(xml.contains("<OAI-PMH"));
        assert!(xml.contains("<dc:title>Test</dc:title>"));
        assert!(xml.contains("oai:1"));
    }

    #[test]
    fn test_catalog_import_from_csv() {
        let csv = "id,title,format,date_created_ms,duration_secs,rights,access_level\n\
                   film001,My Documentary,mp4,1700000000000,3600.5,CC-BY,public\n";
        let entries = CatalogImport::from_csv(csv);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "film001");
        assert_eq!(entries[0].title, "My Documentary");
        assert_eq!(entries[0].format, "mp4");
        assert!((entries[0].duration_secs.expect("test expectation failed") - 3600.5).abs() < 1e-6);
        assert_eq!(entries[0].access_level, AccessLevel::Public);
    }

    #[test]
    fn test_catalog_csv_roundtrip() {
        let original = vec![
            sample_entry("r1", "Film A", 1_000_000),
            sample_entry("r2", "Film, B", 2_000_000),
        ];
        let csv = CatalogExport::to_csv(&original);
        let imported = CatalogImport::from_csv(&csv);
        assert_eq!(imported.len(), 2);
        assert_eq!(imported[0].id, original[0].id);
    }
}

// ── Hierarchical catalog organization ─────────────────────────────────────────

/// Unique collection identifier.
pub type CollectionId = String;

/// A collection node in the hierarchical catalog tree.
///
/// Collections form a tree: each collection may have zero or more
/// sub-collections and zero or more direct entry IDs.
#[derive(Clone, Debug)]
pub struct CatalogCollection {
    /// Unique identifier for this collection.
    pub id: CollectionId,
    /// Human-readable name.
    pub name: String,
    /// Optional description.
    pub description: String,
    /// Parent collection ID, or `None` for root-level collections.
    pub parent_id: Option<CollectionId>,
    /// IDs of entries directly belonging to this collection.
    pub entry_ids: Vec<String>,
    /// IDs of child sub-collections.
    pub child_ids: Vec<CollectionId>,
    /// Creation timestamp (Unix milliseconds).
    pub created_at_ms: u64,
}

impl CatalogCollection {
    /// Create a new root-level collection.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            parent_id: None,
            entry_ids: Vec::new(),
            child_ids: Vec::new(),
            created_at_ms: 0,
        }
    }

    /// Create a sub-collection under a parent.
    #[must_use]
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the creation timestamp.
    #[must_use]
    pub fn with_created_at(mut self, ms: u64) -> Self {
        self.created_at_ms = ms;
        self
    }

    /// Add an entry ID to this collection.
    pub fn add_entry(&mut self, entry_id: impl Into<String>) {
        self.entry_ids.push(entry_id.into());
    }

    /// Number of direct entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entry_ids.len()
    }

    /// Number of direct child sub-collections.
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.child_ids.len()
    }

    /// Whether this is a root collection (no parent).
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.parent_id.is_none()
    }
}

/// Error type for hierarchical catalog operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HierarchyError {
    /// Collection not found.
    CollectionNotFound(String),
    /// Duplicate collection ID.
    DuplicateCollection(String),
    /// Cycle detected (a collection cannot be its own ancestor).
    CycleDetected(String),
    /// Entry not found in collection.
    EntryNotFound(String),
}

impl std::fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CollectionNotFound(id) => write!(f, "collection not found: {id}"),
            Self::DuplicateCollection(id) => write!(f, "duplicate collection: {id}"),
            Self::CycleDetected(id) => write!(f, "cycle detected at: {id}"),
            Self::EntryNotFound(id) => write!(f, "entry not found: {id}"),
        }
    }
}

/// Hierarchical catalog index supporting collections and sub-collections.
///
/// Provides tree-structured organization, breadcrumb path computation,
/// recursive entry collection, and move/rename operations.
#[derive(Debug, Default)]
pub struct HierarchicalCatalog {
    /// All collections keyed by ID.
    collections: std::collections::HashMap<CollectionId, CatalogCollection>,
    /// All entries keyed by entry ID -> set of collection IDs they belong to.
    entry_memberships: std::collections::HashMap<String, Vec<CollectionId>>,
}

impl HierarchicalCatalog {
    /// Create an empty hierarchical catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a collection to the catalog.
    pub fn add_collection(&mut self, collection: CatalogCollection) -> Result<(), HierarchyError> {
        if self.collections.contains_key(&collection.id) {
            return Err(HierarchyError::DuplicateCollection(collection.id.clone()));
        }
        // If it has a parent, register as child of parent.
        if let Some(ref parent_id) = collection.parent_id {
            let parent = self
                .collections
                .get_mut(parent_id)
                .ok_or_else(|| HierarchyError::CollectionNotFound(parent_id.clone()))?;
            parent.child_ids.push(collection.id.clone());
        }
        self.collections.insert(collection.id.clone(), collection);
        Ok(())
    }

    /// Get a collection by its ID.
    #[must_use]
    pub fn get_collection(&self, id: &str) -> Option<&CatalogCollection> {
        self.collections.get(id)
    }

    /// Get a mutable reference to a collection.
    pub fn get_collection_mut(&mut self, id: &str) -> Option<&mut CatalogCollection> {
        self.collections.get_mut(id)
    }

    /// Total number of collections.
    #[must_use]
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }

    /// Return all root-level collections (those with no parent).
    #[must_use]
    pub fn root_collections(&self) -> Vec<&CatalogCollection> {
        self.collections.values().filter(|c| c.is_root()).collect()
    }

    /// Return the direct children of a collection.
    #[must_use]
    pub fn children_of(&self, collection_id: &str) -> Vec<&CatalogCollection> {
        self.collections
            .get(collection_id)
            .map(|c| {
                c.child_ids
                    .iter()
                    .filter_map(|cid| self.collections.get(cid))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Assign an entry to a collection.
    pub fn assign_entry(
        &mut self,
        entry_id: &str,
        collection_id: &str,
    ) -> Result<(), HierarchyError> {
        let col = self
            .collections
            .get_mut(collection_id)
            .ok_or_else(|| HierarchyError::CollectionNotFound(collection_id.to_string()))?;
        if !col.entry_ids.contains(&entry_id.to_string()) {
            col.entry_ids.push(entry_id.to_string());
        }
        self.entry_memberships
            .entry(entry_id.to_string())
            .or_default()
            .push(collection_id.to_string());
        Ok(())
    }

    /// Remove an entry from a collection.
    pub fn remove_entry_from(
        &mut self,
        entry_id: &str,
        collection_id: &str,
    ) -> Result<(), HierarchyError> {
        let col = self
            .collections
            .get_mut(collection_id)
            .ok_or_else(|| HierarchyError::CollectionNotFound(collection_id.to_string()))?;
        let before = col.entry_ids.len();
        col.entry_ids.retain(|eid| eid != entry_id);
        if col.entry_ids.len() == before {
            return Err(HierarchyError::EntryNotFound(entry_id.to_string()));
        }
        if let Some(memberships) = self.entry_memberships.get_mut(entry_id) {
            memberships.retain(|cid| cid != collection_id);
        }
        Ok(())
    }

    /// Get the breadcrumb path from root to the given collection.
    ///
    /// Returns a vector of `(id, name)` pairs from root to the target.
    #[must_use]
    pub fn breadcrumb_path(&self, collection_id: &str) -> Vec<(String, String)> {
        let mut path = Vec::new();
        let mut current_id = Some(collection_id.to_string());
        let mut seen = std::collections::HashSet::new();

        while let Some(ref cid) = current_id {
            if seen.contains(cid) {
                break; // cycle protection
            }
            seen.insert(cid.clone());
            if let Some(col) = self.collections.get(cid) {
                path.push((col.id.clone(), col.name.clone()));
                current_id = col.parent_id.clone();
            } else {
                break;
            }
        }
        path.reverse();
        path
    }

    /// Recursively collect all entry IDs in a collection and its sub-collections.
    #[must_use]
    pub fn all_entries_recursive(&self, collection_id: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut stack = vec![collection_id.to_string()];
        let mut visited = std::collections::HashSet::new();

        while let Some(cid) = stack.pop() {
            if visited.contains(&cid) {
                continue;
            }
            visited.insert(cid.clone());
            if let Some(col) = self.collections.get(&cid) {
                result.extend(col.entry_ids.iter().cloned());
                for child_id in &col.child_ids {
                    stack.push(child_id.clone());
                }
            }
        }
        result
    }

    /// Compute the depth of a collection in the hierarchy (root = 0).
    #[must_use]
    pub fn depth(&self, collection_id: &str) -> usize {
        let path = self.breadcrumb_path(collection_id);
        if path.is_empty() {
            0
        } else {
            path.len() - 1
        }
    }

    /// Get all collections an entry belongs to.
    #[must_use]
    pub fn entry_collections(&self, entry_id: &str) -> Vec<&CatalogCollection> {
        self.entry_memberships
            .get(entry_id)
            .map(|cids| {
                cids.iter()
                    .filter_map(|cid| self.collections.get(cid))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Remove a leaf collection (must have no children).
    pub fn remove_collection(
        &mut self,
        collection_id: &str,
    ) -> Result<CatalogCollection, HierarchyError> {
        let col = self
            .collections
            .get(collection_id)
            .ok_or_else(|| HierarchyError::CollectionNotFound(collection_id.to_string()))?;
        if !col.child_ids.is_empty() {
            return Err(HierarchyError::CycleDetected(format!(
                "collection '{}' has {} children; remove children first",
                collection_id,
                col.child_ids.len()
            )));
        }
        // Unlink from parent
        let parent_id = col.parent_id.clone();
        let col = self
            .collections
            .remove(collection_id)
            .ok_or_else(|| HierarchyError::CollectionNotFound(collection_id.to_string()))?;
        if let Some(ref pid) = parent_id {
            if let Some(parent) = self.collections.get_mut(pid) {
                parent.child_ids.retain(|cid| cid != collection_id);
            }
        }
        // Remove entry memberships
        for entry_id in &col.entry_ids {
            if let Some(memberships) = self.entry_memberships.get_mut(entry_id) {
                memberships.retain(|cid| cid != collection_id);
            }
        }
        Ok(col)
    }
}

impl HierarchicalCatalog {
    /// Compute the depth from the breadcrumb path length.
    fn _depth_from_breadcrumb(&self, collection_id: &str) -> usize {
        let path = self.breadcrumb_path(collection_id);
        if path.is_empty() {
            0
        } else {
            path.len() - 1
        }
    }
}

#[cfg(test)]
mod hierarchy_tests {
    use super::*;

    fn build_hierarchy() -> HierarchicalCatalog {
        let mut h = HierarchicalCatalog::new();
        h.add_collection(CatalogCollection::new("root", "Root Collection").with_created_at(1000))
            .expect("add root");
        h.add_collection(
            CatalogCollection::new("films", "Films")
                .with_parent("root")
                .with_description("All film assets"),
        )
        .expect("add films");
        h.add_collection(CatalogCollection::new("docs", "Documentaries").with_parent("films"))
            .expect("add docs");
        h.add_collection(CatalogCollection::new("music", "Music Videos").with_parent("films"))
            .expect("add music");
        h.add_collection(CatalogCollection::new("audio", "Audio Collection").with_created_at(2000))
            .expect("add audio");
        h
    }

    #[test]
    fn test_add_and_count() {
        let h = build_hierarchy();
        assert_eq!(h.collection_count(), 5);
    }

    #[test]
    fn test_root_collections() {
        let h = build_hierarchy();
        let roots = h.root_collections();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_children_of() {
        let h = build_hierarchy();
        let children = h.children_of("films");
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_children_of_leaf() {
        let h = build_hierarchy();
        let children = h.children_of("docs");
        assert!(children.is_empty());
    }

    #[test]
    fn test_assign_entry() {
        let mut h = build_hierarchy();
        h.assign_entry("film001", "docs").expect("assign entry");
        let col = h.get_collection("docs").expect("get docs");
        assert_eq!(col.entry_count(), 1);
        assert!(col.entry_ids.contains(&"film001".to_string()));
    }

    #[test]
    fn test_remove_entry() {
        let mut h = build_hierarchy();
        h.assign_entry("film001", "docs").expect("assign");
        h.remove_entry_from("film001", "docs").expect("remove");
        let col = h.get_collection("docs").expect("get docs");
        assert_eq!(col.entry_count(), 0);
    }

    #[test]
    fn test_remove_entry_not_found() {
        let mut h = build_hierarchy();
        let result = h.remove_entry_from("ghost", "docs");
        assert!(matches!(result, Err(HierarchyError::EntryNotFound(_))));
    }

    #[test]
    fn test_breadcrumb_path() {
        let h = build_hierarchy();
        let path = h.breadcrumb_path("docs");
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].0, "root");
        assert_eq!(path[1].0, "films");
        assert_eq!(path[2].0, "docs");
    }

    #[test]
    fn test_breadcrumb_path_root() {
        let h = build_hierarchy();
        let path = h.breadcrumb_path("root");
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].0, "root");
    }

    #[test]
    fn test_all_entries_recursive() {
        let mut h = build_hierarchy();
        h.assign_entry("film001", "docs").expect("assign");
        h.assign_entry("film002", "music").expect("assign");
        h.assign_entry("film003", "films").expect("assign");
        let all = h.all_entries_recursive("films");
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_all_entries_recursive_leaf() {
        let mut h = build_hierarchy();
        h.assign_entry("film001", "docs").expect("assign");
        let all = h.all_entries_recursive("docs");
        assert_eq!(all.len(), 1);
    }

    #[test]
    fn test_duplicate_collection_error() {
        let mut h = build_hierarchy();
        let result = h.add_collection(CatalogCollection::new("root", "Duplicate"));
        assert!(matches!(
            result,
            Err(HierarchyError::DuplicateCollection(_))
        ));
    }

    #[test]
    fn test_parent_not_found_error() {
        let mut h = HierarchicalCatalog::new();
        let result =
            h.add_collection(CatalogCollection::new("child", "Child").with_parent("nonexistent"));
        assert!(matches!(result, Err(HierarchyError::CollectionNotFound(_))));
    }

    #[test]
    fn test_remove_leaf_collection() {
        let mut h = build_hierarchy();
        let removed = h.remove_collection("docs").expect("remove docs");
        assert_eq!(removed.id, "docs");
        assert_eq!(h.collection_count(), 4);
        // Parent should no longer list "docs"
        let films = h.get_collection("films").expect("get films");
        assert!(!films.child_ids.contains(&"docs".to_string()));
    }

    #[test]
    fn test_remove_non_leaf_fails() {
        let mut h = build_hierarchy();
        let result = h.remove_collection("films");
        assert!(result.is_err());
    }

    #[test]
    fn test_entry_collections() {
        let mut h = build_hierarchy();
        h.assign_entry("film001", "docs").expect("assign");
        h.assign_entry("film001", "music").expect("assign");
        let cols = h.entry_collections("film001");
        assert_eq!(cols.len(), 2);
    }

    #[test]
    fn test_collection_is_root() {
        let h = build_hierarchy();
        assert!(h.get_collection("root").expect("root").is_root());
        assert!(!h.get_collection("films").expect("films").is_root());
    }

    #[test]
    fn test_hierarchy_error_display() {
        let e = HierarchyError::CollectionNotFound("test".into());
        assert!(e.to_string().contains("test"));
    }
}

// ── New asset-catalog types ───────────────────────────────────────────────────

/// Classification of a media asset.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetKind {
    /// Motion picture or video clip.
    Video,
    /// Audio-only file.
    Audio,
    /// Still image.
    Image,
    /// Text document or PDF.
    Document,
    /// Subtitle or caption file.
    Subtitle,
    /// Sidecar / companion file.
    Sidecar,
}

impl AssetKind {
    /// Returns `true` for primary media kinds (Video, Audio, Image).
    #[must_use]
    pub const fn is_media(self) -> bool {
        matches!(self, Self::Video | Self::Audio | Self::Image)
    }
}

/// A single entry in the asset inventory.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct AssetCatalogEntry {
    /// Unique numeric identifier.
    pub id: u64,
    /// Relative or absolute path.
    pub path: String,
    /// Asset classification.
    pub kind: AssetKind,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Creation time as Unix epoch seconds.
    pub created_epoch: u64,
    /// Arbitrary string tags for this asset.
    pub tags: Vec<String>,
}

impl AssetCatalogEntry {
    /// Returns `true` if `t` is in the tag list (exact match).
    #[must_use]
    pub fn has_tag(&self, t: &str) -> bool {
        self.tags.iter().any(|tag| tag == t)
    }
}

/// In-memory searchable asset inventory.
#[allow(dead_code)]
#[derive(Default, Debug)]
pub struct ArchiveCatalog {
    /// All catalog entries.
    pub entries: Vec<AssetCatalogEntry>,
    next_id: u64,
}

impl ArchiveCatalog {
    /// Create an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry; assigns and returns a unique id.
    pub fn add(
        &mut self,
        path: impl Into<String>,
        kind: AssetKind,
        size_bytes: u64,
        created_epoch: u64,
        tags: Vec<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push(AssetCatalogEntry {
            id,
            path: path.into(),
            kind,
            size_bytes,
            created_epoch,
            tags,
        });
        id
    }

    /// Find an entry by its numeric id.
    #[must_use]
    pub fn find_by_id(&self, id: u64) -> Option<&AssetCatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Find all entries that carry the given tag.
    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<&AssetCatalogEntry> {
        self.entries.iter().filter(|e| e.has_tag(tag)).collect()
    }

    /// Find entries whose path contains `query` (case-insensitive).
    #[must_use]
    pub fn search_path(&self, query: &str) -> Vec<&AssetCatalogEntry> {
        let q = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| e.path.to_lowercase().contains(&q))
            .collect()
    }

    /// Sum of all entry sizes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Count of entries matching `kind`.
    #[must_use]
    pub fn kind_count(&self, kind: AssetKind) -> usize {
        self.entries.iter().filter(|e| e.kind == kind).count()
    }
}

#[cfg(test)]
mod asset_catalog_tests {
    use super::*;

    fn make_catalog() -> ArchiveCatalog {
        let mut c = ArchiveCatalog::new();
        c.add(
            "videos/intro.mp4",
            AssetKind::Video,
            1_000_000,
            1_000,
            vec!["featured".into()],
        );
        c.add(
            "audio/bg.wav",
            AssetKind::Audio,
            500_000,
            2_000,
            vec!["music".into(), "featured".into()],
        );
        c.add(
            "images/thumb.jpg",
            AssetKind::Image,
            200_000,
            3_000,
            vec!["thumbnail".into()],
        );
        c.add(
            "docs/readme.pdf",
            AssetKind::Document,
            50_000,
            4_000,
            vec![],
        );
        c
    }

    #[test]
    fn test_asset_kind_is_media_video() {
        assert!(AssetKind::Video.is_media());
    }

    #[test]
    fn test_asset_kind_is_media_audio() {
        assert!(AssetKind::Audio.is_media());
    }

    #[test]
    fn test_asset_kind_not_media_document() {
        assert!(!AssetKind::Document.is_media());
    }

    #[test]
    fn test_asset_kind_not_media_sidecar() {
        assert!(!AssetKind::Sidecar.is_media());
    }

    #[test]
    fn test_has_tag_true() {
        let c = make_catalog();
        assert!(c
            .find_by_id(0)
            .expect("find_by_id should succeed")
            .has_tag("featured"));
    }

    #[test]
    fn test_has_tag_false() {
        let c = make_catalog();
        assert!(!c
            .find_by_id(0)
            .expect("find_by_id should succeed")
            .has_tag("music"));
    }

    #[test]
    fn test_find_by_id_present() {
        let c = make_catalog();
        assert!(c.find_by_id(2).is_some());
    }

    #[test]
    fn test_find_by_id_missing() {
        let c = make_catalog();
        assert!(c.find_by_id(999).is_none());
    }

    #[test]
    fn test_find_by_tag() {
        let c = make_catalog();
        let results = c.find_by_tag("featured");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_path_case_insensitive() {
        let c = make_catalog();
        let results = c.search_path("VIDEOS");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "videos/intro.mp4");
    }

    #[test]
    fn test_total_size_bytes() {
        let c = make_catalog();
        assert_eq!(c.total_size_bytes(), 1_750_000);
    }

    #[test]
    fn test_kind_count_video() {
        let c = make_catalog();
        assert_eq!(c.kind_count(AssetKind::Video), 1);
    }

    #[test]
    fn test_kind_count_zero() {
        let c = make_catalog();
        assert_eq!(c.kind_count(AssetKind::Subtitle), 0);
    }
}
