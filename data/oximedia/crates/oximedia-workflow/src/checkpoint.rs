//! File-based workflow checkpoint/resume support for `oximedia-workflow`.
//!
//! [`CheckpointManager`] persists workflow execution state to disk so that a
//! failed or interrupted workflow can be resumed from the last saved checkpoint
//! rather than re-run from the beginning.
//!
//! # Storage layout
//!
//! Each checkpoint is written as a JSON file:
//! ```text
//! <storage_dir>/<workflow_id>.checkpoint.json
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_workflow::checkpoint::{CheckpointManager, WorkflowCheckpoint};
//! use std::collections::HashMap;
//!
//! let mgr = CheckpointManager::new(std::env::temp_dir());
//!
//! let cp = WorkflowCheckpoint {
//!     workflow_id: "wf-001".to_string(),
//!     completed_steps: vec!["ingest".to_string()],
//!     step_outputs: HashMap::new(),
//!     created_at: 0,
//!     workflow_version: 1,
//! };
//!
//! mgr.save(&cp).expect("save checkpoint");
//! let loaded = mgr.load("wf-001").expect("load checkpoint");
//! assert_eq!(loaded.completed_steps, vec!["ingest"]);
//! ```

use crate::error::{Result, WorkflowError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// WorkflowCheckpoint
// ---------------------------------------------------------------------------

/// A snapshot of workflow execution state persisted to disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkflowCheckpoint {
    /// Unique identifier of the workflow this checkpoint belongs to.
    pub workflow_id: String,
    /// Names of steps that had completed when this checkpoint was taken.
    pub completed_steps: Vec<String>,
    /// Arbitrary JSON outputs produced by each completed step, keyed by step name.
    pub step_outputs: HashMap<String, serde_json::Value>,
    /// Unix timestamp (seconds) when the checkpoint was created.
    pub created_at: u64,
    /// Schema / content version of the workflow definition at checkpoint time.
    pub workflow_version: u32,
}

impl WorkflowCheckpoint {
    /// Creates a new checkpoint with the current wall-clock timestamp.
    #[must_use]
    pub fn new(workflow_id: impl Into<String>, workflow_version: u32) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            workflow_id: workflow_id.into(),
            completed_steps: Vec::new(),
            step_outputs: HashMap::new(),
            created_at,
            workflow_version,
        }
    }

    /// Marks a step as completed, optionally recording its output.
    pub fn mark_step_completed(
        &mut self,
        step_name: impl Into<String>,
        output: Option<serde_json::Value>,
    ) {
        let name = step_name.into();
        if !self.completed_steps.contains(&name) {
            self.completed_steps.push(name.clone());
        }
        if let Some(v) = output {
            self.step_outputs.insert(name, v);
        }
    }

    /// Returns `true` if the given step was already completed.
    #[must_use]
    pub fn is_step_completed(&self, step_name: &str) -> bool {
        self.completed_steps.iter().any(|s| s == step_name)
    }

    /// Returns the output for a completed step, if any was recorded.
    #[must_use]
    pub fn step_output(&self, step_name: &str) -> Option<&serde_json::Value> {
        self.step_outputs.get(step_name)
    }

    /// Returns the number of completed steps.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.completed_steps.len()
    }
}

// ---------------------------------------------------------------------------
// CheckpointManager
// ---------------------------------------------------------------------------

/// Manages file-based checkpoints for one or more workflows.
///
/// Each checkpoint is persisted as a JSON file named
/// `<storage_dir>/<workflow_id>.checkpoint.json`.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    storage_dir: PathBuf,
}

impl CheckpointManager {
    /// Creates a new manager that stores checkpoints under `storage_dir`.
    ///
    /// The directory is created on first use (lazily), so it does not need to
    /// exist at construction time.
    #[must_use]
    pub fn new(storage_dir: PathBuf) -> Self {
        Self { storage_dir }
    }

    /// Returns the storage directory path.
    #[must_use]
    pub fn storage_dir(&self) -> &PathBuf {
        &self.storage_dir
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Returns the path for a given workflow's checkpoint file.
    fn checkpoint_path(&self, workflow_id: &str) -> PathBuf {
        self.storage_dir
            .join(format!("{}.checkpoint.json", workflow_id))
    }

    /// Ensures the storage directory exists.
    fn ensure_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.storage_dir).map_err(WorkflowError::Io)
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Serialises `checkpoint` to JSON and writes it to disk.
    ///
    /// Overwrites any previously saved checkpoint for the same `workflow_id`.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::Io`] if the file cannot be written, or
    /// [`WorkflowError::Serialization`] if JSON serialisation fails.
    pub fn save(&self, checkpoint: &WorkflowCheckpoint) -> Result<()> {
        self.ensure_dir()?;
        let path = self.checkpoint_path(&checkpoint.workflow_id);
        let json = serde_json::to_vec_pretty(checkpoint)?;
        std::fs::write(&path, &json).map_err(WorkflowError::Io)
    }

    /// Reads and deserialises the checkpoint for `workflow_id` from disk.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::FileNotFound`] if no checkpoint exists, or
    /// [`WorkflowError::Io`] / [`WorkflowError::Serialization`] on read/parse
    /// failure.
    pub fn load(&self, workflow_id: &str) -> Result<WorkflowCheckpoint> {
        let path = self.checkpoint_path(workflow_id);
        if !path.exists() {
            return Err(WorkflowError::FileNotFound(path));
        }
        let bytes = std::fs::read(&path).map_err(WorkflowError::Io)?;
        let cp: WorkflowCheckpoint = serde_json::from_slice(&bytes)?;
        Ok(cp)
    }

    /// Deletes the checkpoint file for `workflow_id`.
    ///
    /// Returns `Ok(())` even if no checkpoint file existed.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::Io`] if the file exists but cannot be removed.
    pub fn delete(&self, workflow_id: &str) -> Result<()> {
        let path = self.checkpoint_path(workflow_id);
        if path.exists() {
            std::fs::remove_file(&path).map_err(WorkflowError::Io)?;
        }
        Ok(())
    }

    /// Returns `true` if a checkpoint file exists for `workflow_id`.
    #[must_use]
    pub fn exists(&self, workflow_id: &str) -> bool {
        self.checkpoint_path(workflow_id).exists()
    }

    /// Returns the workflow IDs of all checkpoints currently on disk.
    ///
    /// Scans `storage_dir` for files matching `*.checkpoint.json` and strips
    /// the suffix to produce the workflow ID.  Files that do not parse as
    /// valid checkpoints are silently skipped.
    ///
    /// # Errors
    ///
    /// Returns [`WorkflowError::Io`] if the storage directory cannot be read.
    pub fn list_checkpoints(&self) -> Result<Vec<String>> {
        if !self.storage_dir.exists() {
            return Ok(Vec::new());
        }

        let mut ids = Vec::new();
        let entries = std::fs::read_dir(&self.storage_dir).map_err(WorkflowError::Io)?;

        for entry in entries {
            let entry = entry.map_err(WorkflowError::Io)?;
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if let Some(id) = name.strip_suffix(".checkpoint.json") {
                // Verify the file is a valid checkpoint before advertising it.
                if self.exists(id) {
                    ids.push(id.to_string());
                }
            }
        }

        ids.sort();
        Ok(ids)
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Returns `true` when the given step should be **skipped** because it was
/// already recorded as completed in `checkpoint`.
///
/// Pass `None` when no checkpoint is available (first run), and the function
/// always returns `false`.
///
/// # Examples
///
/// ```rust
/// use oximedia_workflow::checkpoint::{WorkflowCheckpoint, should_skip_step};
///
/// let mut cp = WorkflowCheckpoint::new("wf-001", 1);
/// cp.mark_step_completed("ingest", None);
///
/// assert!(should_skip_step(&Some(cp), "ingest"));
/// assert!(!should_skip_step(&None, "ingest"));
/// ```
#[must_use]
pub fn should_skip_step(checkpoint: &Option<WorkflowCheckpoint>, step_name: &str) -> bool {
    checkpoint
        .as_ref()
        .map(|cp| cp.is_step_completed(step_name))
        .unwrap_or(false)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: unique temp dir per test to avoid collisions.
    fn temp_dir(suffix: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("oximedia_cp_test_{}", suffix));
        // Remove any leftovers from a previous run (ignore errors).
        let _ = std::fs::remove_dir_all(&base);
        base
    }

    // ------------------------------------------------------------------
    // WorkflowCheckpoint unit tests
    // ------------------------------------------------------------------

    #[test]
    fn test_checkpoint_new_fields() {
        let cp = WorkflowCheckpoint::new("wf-999", 2);
        assert_eq!(cp.workflow_id, "wf-999");
        assert_eq!(cp.workflow_version, 2);
        assert!(cp.completed_steps.is_empty());
        assert!(cp.step_outputs.is_empty());
        // created_at should be non-zero (we are running after 1970).
        assert!(cp.created_at > 0);
    }

    #[test]
    fn test_mark_step_completed_no_output() {
        let mut cp = WorkflowCheckpoint::new("wf-001", 1);
        cp.mark_step_completed("ingest", None);
        assert!(cp.is_step_completed("ingest"));
        assert!(cp.step_output("ingest").is_none());
        assert_eq!(cp.completed_count(), 1);
    }

    #[test]
    fn test_mark_step_completed_with_output() {
        let mut cp = WorkflowCheckpoint::new("wf-001", 1);
        let val = serde_json::json!({"frames": 1000});
        cp.mark_step_completed("analyse", Some(val.clone()));
        assert!(cp.is_step_completed("analyse"));
        assert_eq!(cp.step_output("analyse"), Some(&val));
    }

    #[test]
    fn test_mark_step_completed_dedup() {
        let mut cp = WorkflowCheckpoint::new("wf-001", 1);
        cp.mark_step_completed("transcode", None);
        cp.mark_step_completed("transcode", None);
        assert_eq!(cp.completed_count(), 1);
    }

    #[test]
    fn test_is_step_completed_false() {
        let cp = WorkflowCheckpoint::new("wf-001", 1);
        assert!(!cp.is_step_completed("missing_step"));
    }

    // ------------------------------------------------------------------
    // CheckpointManager: save / load round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_save_and_load() {
        let dir = temp_dir("save_load");
        let mgr = CheckpointManager::new(dir.clone());

        let mut cp = WorkflowCheckpoint::new("wf-save", 1);
        cp.mark_step_completed("step-a", Some(serde_json::json!({"ok": true})));
        mgr.save(&cp).expect("save should succeed");

        let loaded = mgr.load("wf-save").expect("load should succeed");
        assert_eq!(loaded.workflow_id, "wf-save");
        assert_eq!(loaded.completed_steps, vec!["step-a"]);
        assert_eq!(
            loaded.step_output("step-a"),
            Some(&serde_json::json!({"ok": true}))
        );
    }

    #[test]
    fn test_load_nonexistent_returns_error() {
        let dir = temp_dir("load_missing");
        let mgr = CheckpointManager::new(dir);
        let result = mgr.load("no-such-workflow");
        assert!(
            result.is_err(),
            "loading a missing checkpoint must return Err"
        );
    }

    #[test]
    fn test_save_overwrites_previous() {
        let dir = temp_dir("overwrite");
        let mgr = CheckpointManager::new(dir);

        let mut cp1 = WorkflowCheckpoint::new("wf-ow", 1);
        cp1.mark_step_completed("step-a", None);
        mgr.save(&cp1).expect("first save");

        let mut cp2 = WorkflowCheckpoint::new("wf-ow", 1);
        cp2.mark_step_completed("step-a", None);
        cp2.mark_step_completed("step-b", None);
        mgr.save(&cp2).expect("second save");

        let loaded = mgr.load("wf-ow").expect("load after overwrite");
        assert_eq!(loaded.completed_count(), 2);
    }

    // ------------------------------------------------------------------
    // CheckpointManager: exists
    // ------------------------------------------------------------------

    #[test]
    fn test_exists_true_after_save() {
        let dir = temp_dir("exists_true");
        let mgr = CheckpointManager::new(dir);
        let cp = WorkflowCheckpoint::new("wf-ex", 1);
        mgr.save(&cp).expect("save");
        assert!(mgr.exists("wf-ex"));
    }

    #[test]
    fn test_exists_false_before_save() {
        let dir = temp_dir("exists_false");
        let mgr = CheckpointManager::new(dir);
        assert!(!mgr.exists("wf-not-saved"));
    }

    #[test]
    fn test_exists_false_after_delete() {
        let dir = temp_dir("exists_after_delete");
        let mgr = CheckpointManager::new(dir);
        let cp = WorkflowCheckpoint::new("wf-del-ex", 1);
        mgr.save(&cp).expect("save");
        mgr.delete("wf-del-ex").expect("delete");
        assert!(!mgr.exists("wf-del-ex"));
    }

    // ------------------------------------------------------------------
    // CheckpointManager: delete
    // ------------------------------------------------------------------

    #[test]
    fn test_delete_existing() {
        let dir = temp_dir("delete_existing");
        let mgr = CheckpointManager::new(dir);
        let cp = WorkflowCheckpoint::new("wf-del", 1);
        mgr.save(&cp).expect("save");
        assert!(mgr.exists("wf-del"));
        mgr.delete("wf-del").expect("delete should succeed");
        assert!(!mgr.exists("wf-del"));
    }

    #[test]
    fn test_delete_nonexistent_ok() {
        let dir = temp_dir("delete_nonexistent");
        let mgr = CheckpointManager::new(dir);
        // Should not error even though nothing was saved.
        assert!(mgr.delete("ghost-workflow").is_ok());
    }

    // ------------------------------------------------------------------
    // CheckpointManager: list_checkpoints
    // ------------------------------------------------------------------

    #[test]
    fn test_list_empty_when_no_dir() {
        let dir = temp_dir("list_nodir");
        // Do NOT create the directory.
        let mgr = CheckpointManager::new(dir);
        let ids = mgr.list_checkpoints().expect("list should succeed");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_list_checkpoints_after_multiple_saves() {
        let dir = temp_dir("list_multiple");
        let mgr = CheckpointManager::new(dir);

        for id in &["wf-alpha", "wf-beta", "wf-gamma"] {
            let cp = WorkflowCheckpoint::new(*id, 1);
            mgr.save(&cp).expect("save");
        }

        let mut ids = mgr.list_checkpoints().expect("list");
        ids.sort();
        assert_eq!(ids, vec!["wf-alpha", "wf-beta", "wf-gamma"]);
    }

    #[test]
    fn test_list_excludes_deleted() {
        let dir = temp_dir("list_after_delete");
        let mgr = CheckpointManager::new(dir);

        mgr.save(&WorkflowCheckpoint::new("wf-keep", 1))
            .expect("save keep");
        mgr.save(&WorkflowCheckpoint::new("wf-gone", 1))
            .expect("save gone");
        mgr.delete("wf-gone").expect("delete gone");

        let ids = mgr.list_checkpoints().expect("list");
        assert!(ids.contains(&"wf-keep".to_string()));
        assert!(!ids.contains(&"wf-gone".to_string()));
    }

    // ------------------------------------------------------------------
    // should_skip_step
    // ------------------------------------------------------------------

    #[test]
    fn test_should_skip_step_with_completed_step() {
        let mut cp = WorkflowCheckpoint::new("wf-skip", 1);
        cp.mark_step_completed("encode", None);
        assert!(should_skip_step(&Some(cp), "encode"));
    }

    #[test]
    fn test_should_skip_step_with_missing_step() {
        let cp = WorkflowCheckpoint::new("wf-skip", 1);
        assert!(!should_skip_step(&Some(cp), "not-yet-done"));
    }

    #[test]
    fn test_should_skip_step_with_no_checkpoint() {
        assert!(!should_skip_step(&None, "any-step"));
    }

    #[test]
    fn test_should_skip_step_multiple_steps() {
        let mut cp = WorkflowCheckpoint::new("wf-multi", 1);
        cp.mark_step_completed("ingest", None);
        cp.mark_step_completed("transcode", None);
        assert!(should_skip_step(&Some(cp.clone()), "ingest"));
        assert!(should_skip_step(&Some(cp.clone()), "transcode"));
        assert!(!should_skip_step(&Some(cp), "upload"));
    }
}
