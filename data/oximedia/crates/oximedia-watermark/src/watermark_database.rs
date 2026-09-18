//! Persistent registry of embedded watermarks.
//!
//! Provides [`WatermarkRecord`] for representing a watermark registration
//! and [`WatermarkDatabase`] as an in-memory store with look-up by content
//! hash, owner, and algorithm.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Returns the current Unix timestamp in seconds.
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs()
}

/// A single watermark registration record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatermarkRecord {
    /// Unique identifier assigned at registration time.
    pub id: u64,
    /// SHA-256 (or similar) hex-encoded content hash of the watermarked asset.
    pub content_hash: String,
    /// Name or identifier of the rights owner.
    pub owner: String,
    /// Algorithm used when embedding (e.g. "`SpreadSpectrum`", "Echo").
    pub algorithm: String,
    /// Unix timestamp (seconds) when the record was created.
    pub created_at: u64,
    /// Optional free-form notes (e.g. license tier, project name).
    pub notes: Option<String>,
}

impl WatermarkRecord {
    /// Create a new record.  The `id` is caller-supplied; use [`WatermarkDatabase::register`]
    /// to have the database assign a monotonic ID automatically.
    #[must_use]
    pub fn new(
        id: u64,
        content_hash: impl Into<String>,
        owner: impl Into<String>,
        algorithm: impl Into<String>,
    ) -> Self {
        Self {
            id,
            content_hash: content_hash.into(),
            owner: owner.into(),
            algorithm: algorithm.into(),
            created_at: now_unix(),
            notes: None,
        }
    }

    /// Attach notes to the record.
    #[must_use]
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }
}

/// In-memory database of watermark registration records.
pub struct WatermarkDatabase {
    records: Vec<WatermarkRecord>,
    /// Index: `content_hash` → record IDs.
    hash_index: HashMap<String, Vec<u64>>,
    next_id: u64,
}

impl WatermarkDatabase {
    /// Create an empty database.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            hash_index: HashMap::new(),
            next_id: 1,
        }
    }

    /// Register a new watermark and return the assigned record ID.
    pub fn register(
        &mut self,
        content_hash: impl Into<String>,
        owner: impl Into<String>,
        algorithm: impl Into<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let hash = content_hash.into();
        let record = WatermarkRecord::new(id, hash.clone(), owner, algorithm);

        self.hash_index.entry(hash).or_default().push(id);
        self.records.push(record);
        id
    }

    /// Add a pre-constructed [`WatermarkRecord`] (preserves its `id`).
    ///
    /// Returns `false` if a record with the same ID already exists.
    pub fn insert(&mut self, record: WatermarkRecord) -> bool {
        if self.records.iter().any(|r| r.id == record.id) {
            return false;
        }
        self.hash_index
            .entry(record.content_hash.clone())
            .or_default()
            .push(record.id);
        self.records.push(record);
        true
    }

    /// Find all records whose `content_hash` matches the given hash.
    #[must_use]
    pub fn find_by_hash(&self, hash: &str) -> Vec<&WatermarkRecord> {
        let ids = match self.hash_index.get(hash) {
            Some(ids) => ids,
            None => return Vec::new(),
        };
        ids.iter()
            .filter_map(|&id| self.records.iter().find(|r| r.id == id))
            .collect()
    }

    /// Find all records belonging to a specific owner.
    #[must_use]
    pub fn find_by_owner(&self, owner: &str) -> Vec<&WatermarkRecord> {
        self.records.iter().filter(|r| r.owner == owner).collect()
    }

    /// Find all records that used a specific algorithm.
    #[must_use]
    pub fn find_by_algorithm(&self, algorithm: &str) -> Vec<&WatermarkRecord> {
        self.records
            .iter()
            .filter(|r| r.algorithm == algorithm)
            .collect()
    }

    /// Look up a record by its unique ID.
    #[must_use]
    pub fn get_by_id(&self, id: u64) -> Option<&WatermarkRecord> {
        self.records.iter().find(|r| r.id == id)
    }

    /// Remove a record by ID.  Returns the removed record, or `None`.
    pub fn remove(&mut self, id: u64) -> Option<WatermarkRecord> {
        if let Some(pos) = self.records.iter().position(|r| r.id == id) {
            let record = self.records.remove(pos);
            // Clean up the hash index.
            if let Some(ids) = self.hash_index.get_mut(&record.content_hash) {
                ids.retain(|&i| i != id);
                if ids.is_empty() {
                    self.hash_index.remove(&record.content_hash);
                }
            }
            Some(record)
        } else {
            None
        }
    }

    /// Total number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the database is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Iterate over all records in insertion order.
    #[must_use]
    pub fn all(&self) -> &[WatermarkRecord] {
        &self.records
    }
}

impl Default for WatermarkDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_db() -> WatermarkDatabase {
        let mut db = WatermarkDatabase::new();
        db.register("abc123", "Acme Corp", "SpreadSpectrum");
        db.register("abc123", "Acme Corp", "Echo");
        db.register("def456", "BetaMedia", "SpreadSpectrum");
        db
    }

    // --- WatermarkRecord ---

    #[test]
    fn test_record_fields() {
        let r = WatermarkRecord::new(42, "hash_x", "Owner A", "QIM").with_notes("Promo clip");
        assert_eq!(r.id, 42);
        assert_eq!(r.content_hash, "hash_x");
        assert_eq!(r.owner, "Owner A");
        assert_eq!(r.algorithm, "QIM");
        assert_eq!(r.notes.as_deref(), Some("Promo clip"));
    }

    // --- WatermarkDatabase ---

    #[test]
    fn test_empty_on_new() {
        let db = WatermarkDatabase::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
    }

    #[test]
    fn test_register_increments_len() {
        let db = sample_db();
        assert_eq!(db.len(), 3);
    }

    #[test]
    fn test_register_returns_sequential_ids() {
        let mut db = WatermarkDatabase::new();
        let id1 = db.register("h1", "O1", "SS");
        let id2 = db.register("h2", "O2", "SS");
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_find_by_hash_two_results() {
        let db = sample_db();
        let results = db.find_by_hash("abc123");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_find_by_hash_no_match() {
        let db = sample_db();
        assert!(db.find_by_hash("zzz").is_empty());
    }

    #[test]
    fn test_find_by_owner() {
        let db = sample_db();
        let results = db.find_by_owner("Acme Corp");
        assert_eq!(results.len(), 2);
        assert!(db.find_by_owner("Unknown").is_empty());
    }

    #[test]
    fn test_find_by_algorithm() {
        let db = sample_db();
        let ss = db.find_by_algorithm("SpreadSpectrum");
        assert_eq!(ss.len(), 2);
        let echo = db.find_by_algorithm("Echo");
        assert_eq!(echo.len(), 1);
    }

    #[test]
    fn test_get_by_id() {
        let db = sample_db();
        let r = db.get_by_id(1).expect("id 1 should exist");
        assert_eq!(r.id, 1);
        assert!(db.get_by_id(99).is_none());
    }

    #[test]
    fn test_insert_custom_record() {
        let mut db = WatermarkDatabase::new();
        let record = WatermarkRecord::new(100, "custom_hash", "Custom Owner", "Phase");
        assert!(db.insert(record));
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_insert_duplicate_id_returns_false() {
        let mut db = WatermarkDatabase::new();
        let r1 = WatermarkRecord::new(1, "h", "O", "A");
        let r2 = WatermarkRecord::new(1, "h2", "O2", "B");
        assert!(db.insert(r1));
        assert!(!db.insert(r2));
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn test_remove_existing() {
        let mut db = sample_db();
        let removed = db.remove(1);
        assert!(removed.is_some());
        assert_eq!(db.len(), 2);
        // Hash index should be updated.
        assert_eq!(db.find_by_hash("abc123").len(), 1);
    }

    #[test]
    fn test_remove_nonexistent_returns_none() {
        let mut db = sample_db();
        assert!(db.remove(999).is_none());
    }

    #[test]
    fn test_all_returns_all_records() {
        let db = sample_db();
        assert_eq!(db.all().len(), 3);
    }
}
