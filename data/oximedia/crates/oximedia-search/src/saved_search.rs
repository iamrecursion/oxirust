#![allow(dead_code)]
//! Persisting and re-executing named search queries.
//!
//! `saved_search` lets users save any [`SavedQuery`] under a human-readable
//! name, optionally schedule automatic re-execution, track run history,
//! and be notified when new results appear.
//!
//! # Key types
//!
//! | Type | Purpose |
//! |---|---|
//! | [`SavedQuery`] | A named, serialisable query snapshot |
//! | [`SavedQueryStore`] | In-process registry of saved queries |
//! | [`QueryRunRecord`] | Historical run result metadata |
//! | [`ChangeNotification`] | Describes result-set changes since last run |
//!
//! # Persistence model
//!
//! The store serialises to/from JSON so it can be saved to any `Write`
//! target (file, Redis string, S3 object, etc.).  The store itself does
//! **not** own an I/O layer — callers are responsible for serialising the
//! store when persistence is needed.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{SearchError, SearchResult};

// ---------------------------------------------------------------------------
// Query snapshot types
// ---------------------------------------------------------------------------

/// Serialisable snapshot of a search query for a saved search.
///
/// Uses a simplified representation so that saved searches can be
/// re-executed across versions without tight coupling to the live
/// `SearchQuery` type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedQuery {
    /// Unique identifier for this saved search.
    pub id: Uuid,
    /// Human-readable name chosen by the user.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Text query expression (may be `None` for filter-only searches).
    pub text_query: Option<String>,
    /// JSON-serialised filter parameters (free-form to avoid coupling).
    pub filter_json: Option<String>,
    /// Sort field name.
    pub sort_field: Option<String>,
    /// Sort order: `"asc"` or `"desc"`.
    pub sort_order: Option<String>,
    /// Maximum results per run.
    pub limit: usize,
    /// Creator user identifier.
    pub owner_id: Option<String>,
    /// Tags for organising saved searches.
    pub tags: Vec<String>,
    /// Whether this saved search is shared with other users.
    pub is_shared: bool,
    /// Unix timestamp when the query was saved.
    pub created_at: i64,
    /// Unix timestamp when the query was last modified.
    pub updated_at: i64,
    /// Whether automatic scheduling is enabled.
    pub schedule_enabled: bool,
    /// Interval in seconds between automatic re-runs (`None` = manual only).
    pub schedule_interval_secs: Option<u64>,
}

impl SavedQuery {
    /// Create a new saved query with a text expression.
    pub fn new(name: impl Into<String>, text_query: impl Into<String>, now_secs: i64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            description: None,
            text_query: Some(text_query.into()),
            filter_json: None,
            sort_field: None,
            sort_order: None,
            limit: 50,
            owner_id: None,
            tags: Vec::new(),
            is_shared: false,
            created_at: now_secs,
            updated_at: now_secs,
            schedule_enabled: false,
            schedule_interval_secs: None,
        }
    }

    /// Builder: set description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Builder: set owner.
    #[must_use]
    pub fn with_owner(mut self, owner: impl Into<String>) -> Self {
        self.owner_id = Some(owner.into());
        self
    }

    /// Builder: set tags.
    #[must_use]
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Builder: enable automatic scheduling.
    #[must_use]
    pub fn with_schedule(mut self, interval_secs: u64) -> Self {
        self.schedule_enabled = true;
        self.schedule_interval_secs = Some(interval_secs);
        self
    }

    /// Builder: set result limit.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

// ---------------------------------------------------------------------------
// Run history
// ---------------------------------------------------------------------------

/// Outcome of a single run of a saved search.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunOutcome {
    /// The search completed successfully.
    Success,
    /// The search failed with an error message.
    Failure(String),
    /// The run was cancelled before completion.
    Cancelled,
}

/// Record of a single saved-search execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRunRecord {
    /// Unique run ID.
    pub run_id: Uuid,
    /// ID of the saved query that was executed.
    pub query_id: Uuid,
    /// Unix timestamp when the run started.
    pub started_at: i64,
    /// Duration of the run in milliseconds.
    pub duration_ms: u64,
    /// Number of results returned.
    pub result_count: usize,
    /// IDs of results returned (for change detection).
    pub result_ids: Vec<Uuid>,
    /// Run outcome.
    pub outcome: RunOutcome,
}

impl QueryRunRecord {
    /// Create a successful run record.
    pub fn success(
        query_id: Uuid,
        started_at: i64,
        duration_ms: u64,
        result_ids: Vec<Uuid>,
    ) -> Self {
        let count = result_ids.len();
        Self {
            run_id: Uuid::new_v4(),
            query_id,
            started_at,
            duration_ms,
            result_count: count,
            result_ids,
            outcome: RunOutcome::Success,
        }
    }

    /// Create a failed run record.
    pub fn failure(query_id: Uuid, started_at: i64, duration_ms: u64, error: String) -> Self {
        Self {
            run_id: Uuid::new_v4(),
            query_id,
            started_at,
            duration_ms,
            result_count: 0,
            result_ids: Vec::new(),
            outcome: RunOutcome::Failure(error),
        }
    }
}

// ---------------------------------------------------------------------------
// Change notification
// ---------------------------------------------------------------------------

/// Describes changes to a saved search's result set between two runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeNotification {
    /// ID of the saved query.
    pub query_id: Uuid,
    /// IDs of assets that appeared in the new run but not the previous.
    pub new_results: Vec<Uuid>,
    /// IDs of assets that appeared in the previous run but not the new one.
    pub removed_results: Vec<Uuid>,
    /// Total results in the new run.
    pub total_results: usize,
    /// Whether any change occurred.
    pub has_changes: bool,
}

impl ChangeNotification {
    /// Compare two consecutive runs and produce a change notification.
    #[must_use]
    pub fn diff(query_id: Uuid, prev: &QueryRunRecord, next: &QueryRunRecord) -> Self {
        let prev_set: std::collections::HashSet<Uuid> = prev.result_ids.iter().copied().collect();
        let next_set: std::collections::HashSet<Uuid> = next.result_ids.iter().copied().collect();

        let new_results: Vec<Uuid> = next_set.difference(&prev_set).copied().collect();
        let removed_results: Vec<Uuid> = prev_set.difference(&next_set).copied().collect();
        let has_changes = !new_results.is_empty() || !removed_results.is_empty();

        Self {
            query_id,
            new_results,
            removed_results,
            total_results: next.result_count,
            has_changes,
        }
    }
}

// ---------------------------------------------------------------------------
// SavedQueryStore
// ---------------------------------------------------------------------------

/// In-memory store for saved search queries with run history.
#[derive(Debug, Default)]
pub struct SavedQueryStore {
    /// Saved queries keyed by ID.
    queries: HashMap<Uuid, SavedQuery>,
    /// Run history keyed by query ID → list of runs (newest first).
    history: HashMap<Uuid, Vec<QueryRunRecord>>,
    /// Maximum run records to keep per query.
    max_history_per_query: usize,
}

impl SavedQueryStore {
    /// Create a new store with a 50-record history limit per query.
    #[must_use]
    pub fn new() -> Self {
        Self {
            queries: HashMap::new(),
            history: HashMap::new(),
            max_history_per_query: 50,
        }
    }

    /// Create a store with a custom history limit.
    #[must_use]
    pub fn with_history_limit(limit: usize) -> Self {
        Self {
            queries: HashMap::new(),
            history: HashMap::new(),
            max_history_per_query: limit.max(1),
        }
    }

    /// Save a new query (or replace if the ID already exists).
    pub fn save(&mut self, query: SavedQuery) {
        self.queries.insert(query.id, query);
    }

    /// Update an existing saved query, preserving the ID and created_at.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DocumentNotFound` if the ID is not found.
    pub fn update(&mut self, mut query: SavedQuery, now_secs: i64) -> SearchResult<()> {
        let existing = self
            .queries
            .get(&query.id)
            .ok_or_else(|| SearchError::DocumentNotFound(query.id.to_string()))?;
        query.created_at = existing.created_at;
        query.updated_at = now_secs;
        self.queries.insert(query.id, query);
        Ok(())
    }

    /// Delete a saved query and its run history.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DocumentNotFound` if not found.
    pub fn delete(&mut self, query_id: Uuid) -> SearchResult<()> {
        self.queries
            .remove(&query_id)
            .ok_or_else(|| SearchError::DocumentNotFound(query_id.to_string()))?;
        self.history.remove(&query_id);
        Ok(())
    }

    /// Get a saved query by ID.
    #[must_use]
    pub fn get(&self, query_id: Uuid) -> Option<&SavedQuery> {
        self.queries.get(&query_id)
    }

    /// List all saved queries.
    #[must_use]
    pub fn list(&self) -> Vec<&SavedQuery> {
        let mut v: Vec<&SavedQuery> = self.queries.values().collect();
        v.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        v
    }

    /// List saved queries belonging to a specific owner.
    #[must_use]
    pub fn list_by_owner(&self, owner_id: &str) -> Vec<&SavedQuery> {
        let mut v: Vec<&SavedQuery> = self
            .queries
            .values()
            .filter(|q| q.owner_id.as_deref() == Some(owner_id))
            .collect();
        v.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        v
    }

    /// List shared saved queries.
    #[must_use]
    pub fn list_shared(&self) -> Vec<&SavedQuery> {
        let mut v: Vec<&SavedQuery> = self.queries.values().filter(|q| q.is_shared).collect();
        v.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        v
    }

    /// Find saved queries by tag.
    #[must_use]
    pub fn find_by_tag(&self, tag: &str) -> Vec<&SavedQuery> {
        self.queries
            .values()
            .filter(|q| q.tags.iter().any(|t| t.eq_ignore_ascii_case(tag)))
            .collect()
    }

    /// Record a run for a saved query.
    ///
    /// Generates a [`ChangeNotification`] by diffing against the previous run.
    ///
    /// # Errors
    ///
    /// Returns `SearchError::DocumentNotFound` if the query_id is unknown.
    pub fn record_run(
        &mut self,
        record: QueryRunRecord,
    ) -> SearchResult<Option<ChangeNotification>> {
        if !self.queries.contains_key(&record.query_id) {
            return Err(SearchError::DocumentNotFound(record.query_id.to_string()));
        }

        let history = self.history.entry(record.query_id).or_default();
        let notification = history
            .first()
            .map(|prev| ChangeNotification::diff(record.query_id, prev, &record));

        history.insert(0, record);
        history.truncate(self.max_history_per_query);

        Ok(notification)
    }

    /// Get the run history for a query, newest first.
    #[must_use]
    pub fn run_history(&self, query_id: Uuid) -> &[QueryRunRecord] {
        self.history
            .get(&query_id)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Get the most recent successful run for a query.
    #[must_use]
    pub fn last_successful_run(&self, query_id: Uuid) -> Option<&QueryRunRecord> {
        self.run_history(query_id)
            .iter()
            .find(|r| r.outcome == RunOutcome::Success)
    }

    /// Return queries that are due for automatic re-execution.
    ///
    /// A query is due if `schedule_enabled` is `true` and
    /// `now_secs - last_run_started_at >= schedule_interval_secs`.
    /// Queries that have never been run are always included.
    #[must_use]
    pub fn queries_due_for_run(&self, now_secs: i64) -> Vec<&SavedQuery> {
        self.queries
            .values()
            .filter(|q| {
                if !q.schedule_enabled {
                    return false;
                }
                let interval = match q.schedule_interval_secs {
                    Some(i) => i as i64,
                    None => return false,
                };
                let last_run = self
                    .run_history(q.id)
                    .first()
                    .map(|r| r.started_at)
                    .unwrap_or(i64::MIN / 2);
                now_secs.saturating_sub(last_run) >= interval
            })
            .collect()
    }

    /// Total number of saved queries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Whether the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    /// Serialise the entire store to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation fails.
    pub fn to_json(&self) -> SearchResult<String> {
        let data = StorageSnapshot {
            queries: self.queries.values().cloned().collect(),
            history: self
                .history
                .iter()
                .map(|(&k, v)| (k.to_string(), v.clone()))
                .collect(),
        };
        serde_json::to_string(&data).map_err(SearchError::Serialization)
    }

    /// Restore the store from a JSON snapshot produced by `to_json`.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialisation fails.
    pub fn from_json(json: &str) -> SearchResult<Self> {
        let data: StorageSnapshot =
            serde_json::from_str(json).map_err(SearchError::Serialization)?;
        let mut queries = HashMap::new();
        for q in data.queries {
            queries.insert(q.id, q);
        }
        let mut history: HashMap<Uuid, Vec<QueryRunRecord>> = HashMap::new();
        for (k, v) in data.history {
            let id: Uuid = k
                .parse()
                .map_err(|e: uuid::Error| SearchError::Other(e.to_string()))?;
            history.insert(id, v);
        }
        Ok(Self {
            queries,
            history,
            max_history_per_query: 50,
        })
    }
}

// Serialisation helper — not part of the public API.
#[derive(Serialize, Deserialize)]
struct StorageSnapshot {
    queries: Vec<SavedQuery>,
    history: HashMap<String, Vec<QueryRunRecord>>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: i64 = 1_700_000_000;

    fn make_query(name: &str) -> SavedQuery {
        SavedQuery::new(name, "test query", NOW)
    }

    #[test]
    fn test_saved_query_new() {
        let q = make_query("My Search");
        assert_eq!(q.name, "My Search");
        assert_eq!(q.text_query.as_deref(), Some("test query"));
        assert_eq!(q.limit, 50);
        assert!(!q.is_shared);
        assert!(!q.schedule_enabled);
    }

    #[test]
    fn test_saved_query_builder() {
        let q = SavedQuery::new("tagged", "footage", NOW)
            .with_owner("user-1")
            .with_tags(vec!["production".into(), "2025".into()])
            .with_limit(100)
            .with_schedule(3600);
        assert_eq!(q.owner_id.as_deref(), Some("user-1"));
        assert_eq!(q.tags.len(), 2);
        assert_eq!(q.limit, 100);
        assert!(q.schedule_enabled);
        assert_eq!(q.schedule_interval_secs, Some(3600));
    }

    #[test]
    fn test_store_save_and_get() {
        let mut store = SavedQueryStore::new();
        let q = make_query("Search A");
        let id = q.id;
        store.save(q);
        assert_eq!(store.len(), 1);
        let retrieved = store.get(id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.map(|q| q.name.as_str()), Some("Search A"));
    }

    #[test]
    fn test_store_delete() {
        let mut store = SavedQueryStore::new();
        let q = make_query("Temp");
        let id = q.id;
        store.save(q);
        assert!(store.delete(id).is_ok());
        assert!(store.get(id).is_none());
    }

    #[test]
    fn test_store_delete_nonexistent() {
        let mut store = SavedQueryStore::new();
        assert!(store.delete(Uuid::new_v4()).is_err());
    }

    #[test]
    fn test_store_update() {
        let mut store = SavedQueryStore::new();
        let q = make_query("Original");
        let id = q.id;
        let original_created = q.created_at;
        store.save(q);

        let mut updated = make_query("Updated");
        updated.id = id;
        store.update(updated, NOW + 100).expect("should update");

        let retrieved = store.get(id).expect("exists");
        assert_eq!(retrieved.name, "Updated");
        assert_eq!(retrieved.created_at, original_created);
        assert_eq!(retrieved.updated_at, NOW + 100);
    }

    #[test]
    fn test_store_list_sorted_by_created_desc() {
        let mut store = SavedQueryStore::new();
        let mut q1 = make_query("Old");
        q1.created_at = NOW - 1000;
        let mut q2 = make_query("New");
        q2.created_at = NOW;
        store.save(q1);
        store.save(q2);
        let list = store.list();
        assert_eq!(list[0].name, "New");
    }

    #[test]
    fn test_list_by_owner() {
        let mut store = SavedQueryStore::new();
        let q1 = make_query("A").with_owner("alice");
        let q2 = make_query("B").with_owner("bob");
        let q3 = make_query("C").with_owner("alice");
        store.save(q1);
        store.save(q2);
        store.save(q3);
        let alice_queries = store.list_by_owner("alice");
        assert_eq!(alice_queries.len(), 2);
    }

    #[test]
    fn test_list_shared() {
        let mut store = SavedQueryStore::new();
        let mut q1 = make_query("shared");
        q1.is_shared = true;
        let q2 = make_query("private");
        store.save(q1);
        store.save(q2);
        let shared = store.list_shared();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].name, "shared");
    }

    #[test]
    fn test_find_by_tag() {
        let mut store = SavedQueryStore::new();
        let q1 = make_query("A").with_tags(vec!["nature".into(), "wildlife".into()]);
        let q2 = make_query("B").with_tags(vec!["city".into()]);
        let q3 = make_query("C").with_tags(vec!["Nature".into()]);
        store.save(q1);
        store.save(q2);
        store.save(q3);
        let nature = store.find_by_tag("nature");
        assert_eq!(nature.len(), 2); // case-insensitive
    }

    #[test]
    fn test_record_run_and_history() {
        let mut store = SavedQueryStore::new();
        let q = make_query("Running");
        let qid = q.id;
        store.save(q);

        let ids = vec![Uuid::new_v4(), Uuid::new_v4()];
        let record = QueryRunRecord::success(qid, NOW, 150, ids.clone());
        let notification = store.record_run(record).expect("ok");
        assert!(notification.is_none()); // first run → no previous

        assert_eq!(store.run_history(qid).len(), 1);
        assert_eq!(store.run_history(qid)[0].result_count, 2);
    }

    #[test]
    fn test_record_run_change_notification() {
        let mut store = SavedQueryStore::new();
        let q = make_query("Watch");
        let qid = q.id;
        store.save(q);

        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let id_c = Uuid::new_v4();

        // First run.
        let r1 = QueryRunRecord::success(qid, NOW, 10, vec![id_a, id_b]);
        store.record_run(r1).expect("ok");

        // Second run: id_b gone, id_c new.
        let r2 = QueryRunRecord::success(qid, NOW + 100, 10, vec![id_a, id_c]);
        let note = store.record_run(r2).expect("ok").expect("change notif");
        assert!(note.has_changes);
        assert!(note.new_results.contains(&id_c));
        assert!(note.removed_results.contains(&id_b));
        assert!(!note.new_results.contains(&id_a));
    }

    #[test]
    fn test_last_successful_run() {
        let mut store = SavedQueryStore::new();
        let q = make_query("Reliable");
        let qid = q.id;
        store.save(q);

        let good = QueryRunRecord::success(qid, NOW, 20, vec![Uuid::new_v4()]);
        store.record_run(good).expect("ok");

        let bad = QueryRunRecord::failure(qid, NOW + 50, 5, "timeout".into());
        store.record_run(bad).expect("ok");

        let last_ok = store.last_successful_run(qid);
        assert!(last_ok.is_some());
        assert_eq!(last_ok.map(|r| r.started_at), Some(NOW));
    }

    #[test]
    fn test_queries_due_for_run() {
        let mut store = SavedQueryStore::new();
        let scheduled = SavedQuery::new("Scheduled", "search", NOW).with_schedule(3600);
        let manual = make_query("Manual");
        let sched_id = scheduled.id;
        store.save(scheduled);
        store.save(manual);

        // With now = NOW + 4000 (>3600 past creation), scheduled query is due.
        let due = store.queries_due_for_run(NOW + 4000);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].id, sched_id);
    }

    #[test]
    fn test_queries_due_not_scheduled() {
        let mut store = SavedQueryStore::new();
        let q = make_query("No Schedule");
        store.save(q);
        let due = store.queries_due_for_run(NOW + 99_999);
        assert!(due.is_empty());
    }

    #[test]
    fn test_history_limit() {
        let mut store = SavedQueryStore::with_history_limit(3);
        let q = make_query("Limited");
        let qid = q.id;
        store.save(q);

        for i in 0..5 {
            let r = QueryRunRecord::success(qid, NOW + i, 10, vec![]);
            store.record_run(r).expect("ok");
        }
        assert_eq!(store.run_history(qid).len(), 3);
    }

    #[test]
    fn test_record_run_unknown_query() {
        let mut store = SavedQueryStore::new();
        let r = QueryRunRecord::success(Uuid::new_v4(), NOW, 10, vec![]);
        assert!(store.record_run(r).is_err());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut store = SavedQueryStore::new();
        let q = make_query("Serializable").with_owner("user");
        let qid = q.id;
        store.save(q);
        let r = QueryRunRecord::success(qid, NOW, 100, vec![Uuid::new_v4()]);
        store.record_run(r).expect("ok");

        let json = store.to_json().expect("serialize");
        let restored = SavedQueryStore::from_json(&json).expect("deserialize");

        assert_eq!(restored.len(), 1);
        let restored_q = restored.get(qid).expect("should exist");
        assert_eq!(restored_q.name, "Serializable");
        assert_eq!(restored.run_history(qid).len(), 1);
    }

    #[test]
    fn test_change_notification_diff() {
        let qid = Uuid::new_v4();
        let id_a = Uuid::new_v4();
        let id_b = Uuid::new_v4();
        let id_c = Uuid::new_v4();

        let prev = QueryRunRecord::success(qid, NOW, 10, vec![id_a, id_b]);
        let next = QueryRunRecord::success(qid, NOW + 60, 10, vec![id_b, id_c]);

        let note = ChangeNotification::diff(qid, &prev, &next);
        assert!(note.has_changes);
        assert!(note.new_results.contains(&id_c));
        assert!(note.removed_results.contains(&id_a));
        assert!(!note.new_results.contains(&id_b));
        assert_eq!(note.total_results, 2);
    }

    #[test]
    fn test_change_notification_no_change() {
        let qid = Uuid::new_v4();
        let id_a = Uuid::new_v4();
        let prev = QueryRunRecord::success(qid, NOW, 10, vec![id_a]);
        let next = QueryRunRecord::success(qid, NOW + 60, 10, vec![id_a]);
        let note = ChangeNotification::diff(qid, &prev, &next);
        assert!(!note.has_changes);
        assert!(note.new_results.is_empty());
        assert!(note.removed_results.is_empty());
    }
}
