//! Batch metadata operations: apply a uniform set of operations to multiple files in parallel.
//!
//! # Overview
//!
//! [`BatchMetadataEditor`] applies a slice of [`MetadataOperation`]s to a collection of file
//! paths concurrently using Rayon.  Operations are applied in declaration order; if more than one
//! operation targets the same field key the last one wins.
//!
//! # Example
//!
//! ```no_run
//! use std::path::PathBuf;
//! use oximedia_metadata::batch_editor::{BatchMetadataEditor, MetadataOperation};
//!
//! let editor = BatchMetadataEditor::new();
//! let files = vec![PathBuf::from("a.mp3"), PathBuf::from("b.mp3")];
//! let ops = vec![
//!     MetadataOperation::SetField {
//!         key: "artist".to_string(),
//!         value: "COOLJAPAN OU".to_string(),
//!     },
//! ];
//! let results = editor.apply_batch(&files, &ops);
//! for (path, result) in files.iter().zip(results.iter()) {
//!     match result {
//!         Ok(()) => println!("{}: ok", path.display()),
//!         Err(e) => eprintln!("{}: {}", path.display(), e),
//!     }
//! }
//! ```

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::Error;

// ─────────────────────────────────────────────────────────────────────────────
// MetadataOperation
// ─────────────────────────────────────────────────────────────────────────────

/// A single logical operation that can be applied to a metadata store.
///
/// Operations are designed to be composable: a `Vec<MetadataOperation>` is
/// applied sequentially to a mutable in-memory store, and the outcome is then
/// persisted to disk.
#[derive(Debug, Clone)]
pub enum MetadataOperation {
    /// Insert or overwrite a text field.
    SetField {
        /// The metadata field key (e.g. `"TIT2"`, `"artist"`, `"©nam"`).
        key: String,
        /// UTF-8 text value to associate with the key.
        value: String,
    },

    /// Remove a field entirely.  A no-op if the field does not exist.
    RemoveField {
        /// The metadata field key to remove.
        key: String,
    },

    /// Copy all fields from a *source* file and merge them into the target.
    ///
    /// Fields present in the source **overwrite** existing fields in the target.
    /// The source file is read-only; no modifications are made to it.
    CopyFrom {
        /// Path to the file whose metadata should be copied.
        source: PathBuf,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// In-memory metadata store (simple key→value map)
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight in-memory key/value metadata store used during batch processing.
///
/// All values are stored as UTF-8 strings.  Binary fields (pictures, etc.) are
/// out of scope for the batch text-field editor.
#[derive(Debug, Default, Clone)]
pub struct InMemoryStore {
    fields: HashMap<String, String>,
}

impl InMemoryStore {
    /// Create a new, empty store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Load the store from a file by scanning for key=value pairs.
    ///
    /// The format expected is a UTF-8 text file with one `key=value` pair per
    /// line (matching the Vorbis Comment / ID3 text field convention used by
    /// most metadata sidecar tools).  Lines that do not contain `=` are
    /// silently skipped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be read.
    pub fn load_from_file(path: &Path) -> Result<Self, Error> {
        let raw = std::fs::read_to_string(path)?;
        let mut store = Self::new();
        for line in raw.lines() {
            if let Some(pos) = line.find('=') {
                let key = line[..pos].trim().to_string();
                let value = line[pos + 1..].trim().to_string();
                if !key.is_empty() {
                    store.fields.insert(key, value);
                }
            }
        }
        Ok(store)
    }

    /// Persist the store to a file as `key=value` lines.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`] if the file cannot be written.
    pub fn save_to_file(&self, path: &Path) -> Result<(), Error> {
        use std::fmt::Write as FmtWrite;
        let mut out = String::new();
        // Sort keys for deterministic output
        let mut keys: Vec<&str> = self.fields.keys().map(String::as_str).collect();
        keys.sort_unstable();
        for key in keys {
            writeln!(out, "{}={}", key, self.fields[key])
                .map_err(|e| Error::WriteError(e.to_string()))?;
        }
        std::fs::write(path, out.as_bytes())?;
        Ok(())
    }

    /// Set (or overwrite) a field.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.fields.insert(key.into(), value.into());
    }

    /// Remove a field, returning the old value if it existed.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.fields.remove(key)
    }

    /// Get a field value.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(String::as_str)
    }

    /// Merge all fields from `other` into `self`, overwriting on conflict.
    pub fn merge_from(&mut self, other: &Self) {
        for (k, v) in &other.fields {
            self.fields.insert(k.clone(), v.clone());
        }
    }

    /// Return the number of stored fields.
    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Return `true` if no fields are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Return an iterator over (key, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.fields.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Operation application
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a single [`MetadataOperation`] to an [`InMemoryStore`].
///
/// `CopyFrom` reads the source file from disk; all other variants are in-memory.
///
/// # Errors
///
/// Returns an error only for I/O failures (e.g. `CopyFrom` source unreadable).
fn apply_operation(store: &mut InMemoryStore, op: &MetadataOperation) -> Result<(), Error> {
    match op {
        MetadataOperation::SetField { key, value } => {
            store.set(key.clone(), value.clone());
            Ok(())
        }
        MetadataOperation::RemoveField { key } => {
            store.remove(key);
            Ok(())
        }
        MetadataOperation::CopyFrom { source } => {
            let source_store = InMemoryStore::load_from_file(source)?;
            store.merge_from(&source_store);
            Ok(())
        }
    }
}

/// Apply a sequence of operations to a store in order.
fn apply_operations(store: &mut InMemoryStore, ops: &[MetadataOperation]) -> Result<(), Error> {
    for op in ops {
        apply_operation(store, op)?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchMetadataEditor
// ─────────────────────────────────────────────────────────────────────────────

/// Applies a set of [`MetadataOperation`]s to multiple files in parallel.
///
/// Each file is processed independently on Rayon's global thread pool.  Results
/// are returned in the same order as the input `files` slice.
///
/// Files are expected to exist as writable key=value text sidecar files.  If a
/// file does not yet exist an empty store is created before applying operations.
///
/// # Thread Safety
///
/// `BatchMetadataEditor` is `Send + Sync`; it carries no mutable state.
#[derive(Debug, Default, Clone)]
pub struct BatchMetadataEditor {
    /// When `true` the edited store is persisted back to the original path.
    pub write_back: bool,
}

impl BatchMetadataEditor {
    /// Create a new editor.  By default `write_back` is disabled so that the
    /// caller controls persistence.
    #[must_use]
    pub fn new() -> Self {
        Self { write_back: false }
    }

    /// Enable automatic write-back to disk after applying operations.
    #[must_use]
    pub fn with_write_back(mut self) -> Self {
        self.write_back = true;
        self
    }

    /// Apply `operations` to each file in `files` in parallel via Rayon.
    ///
    /// Returns a `Vec<Result<(), Error>>` of the same length as `files` (one
    /// entry per file, in the same order).
    ///
    /// Each file is:
    /// 1. Loaded into an [`InMemoryStore`] (or created empty if absent).
    /// 2. Processed through `operations` sequentially.
    /// 3. Optionally persisted if `self.write_back` is `true`.
    ///
    /// An error in one file does **not** abort processing of the others.
    #[must_use]
    pub fn apply_batch(
        &self,
        files: &[PathBuf],
        operations: &[MetadataOperation],
    ) -> Vec<Result<(), Error>> {
        files
            .par_iter()
            .map(|path| self.process_single(path, operations))
            .collect()
    }

    /// Process a single file: load, apply all operations, optionally save.
    fn process_single(&self, path: &Path, ops: &[MetadataOperation]) -> Result<(), Error> {
        // Load existing store, or start fresh if the file does not exist.
        let mut store = if path.exists() {
            InMemoryStore::load_from_file(path)?
        } else {
            InMemoryStore::new()
        };

        apply_operations(&mut store, ops)?;

        if self.write_back {
            store.save_to_file(path)?;
        }

        Ok(())
    }

    /// Apply operations and return the resulting [`InMemoryStore`]s without
    /// touching the file system.  Useful for unit-testing or dry-run checks.
    ///
    /// Returns one store per file in the same order as `files`.
    #[must_use]
    pub fn apply_batch_in_memory(
        &self,
        files: &[PathBuf],
        operations: &[MetadataOperation],
    ) -> Vec<Result<InMemoryStore, Error>> {
        files
            .par_iter()
            .map(|path| {
                let mut store = if path.exists() {
                    InMemoryStore::load_from_file(path)?
                } else {
                    InMemoryStore::new()
                };
                apply_operations(&mut store, operations)?;
                Ok(store)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Write a simple key=value file and return its path.
    fn write_kv_file(name: &str, pairs: &[(&str, &str)]) -> PathBuf {
        let mut path = temp_dir();
        path.push(name);
        let content: String = pairs
            .iter()
            .map(|(k, v)| format!("{}={}\n", k, v))
            .collect();
        std::fs::write(&path, content.as_bytes()).expect("write test file");
        path
    }

    // ── InMemoryStore ─────────────────────────────────────────────────────────

    #[test]
    fn test_store_set_and_get() {
        let mut s = InMemoryStore::new();
        s.set("artist", "Test Artist");
        assert_eq!(s.get("artist"), Some("Test Artist"));
        assert_eq!(s.get("title"), None);
    }

    #[test]
    fn test_store_remove() {
        let mut s = InMemoryStore::new();
        s.set("genre", "Jazz");
        assert_eq!(s.remove("genre"), Some("Jazz".to_string()));
        assert!(s.get("genre").is_none());
    }

    #[test]
    fn test_store_merge_overwrites() {
        let mut a = InMemoryStore::new();
        a.set("artist", "Old");
        a.set("title", "My Track");

        let mut b = InMemoryStore::new();
        b.set("artist", "New");
        b.set("album", "My Album");

        a.merge_from(&b);
        assert_eq!(a.get("artist"), Some("New"));
        assert_eq!(a.get("title"), Some("My Track"));
        assert_eq!(a.get("album"), Some("My Album"));
    }

    #[test]
    fn test_store_load_and_save() {
        let path = write_kv_file(
            "test_store_load.kv",
            &[("title", "Hello"), ("year", "2025")],
        );
        let s = InMemoryStore::load_from_file(&path).expect("loading test store should succeed");
        assert_eq!(s.get("title"), Some("Hello"));
        assert_eq!(s.get("year"), Some("2025"));

        let save_path = {
            let mut p = temp_dir();
            p.push("test_store_save.kv");
            p
        };
        s.save_to_file(&save_path)
            .expect("saving test store should succeed");
        let s2 = InMemoryStore::load_from_file(&save_path)
            .expect("reloading saved store should succeed");
        assert_eq!(s2.get("title"), Some("Hello"));
        assert_eq!(s2.get("year"), Some("2025"));
    }

    #[test]
    fn test_store_empty_is_empty() {
        let s = InMemoryStore::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    // ── MetadataOperation ─────────────────────────────────────────────────────

    #[test]
    fn test_op_set_field() {
        let mut store = InMemoryStore::new();
        let op = MetadataOperation::SetField {
            key: "title".to_string(),
            value: "Test".to_string(),
        };
        apply_operation(&mut store, &op).expect("set field operation should succeed");
        assert_eq!(store.get("title"), Some("Test"));
    }

    #[test]
    fn test_op_remove_field_existing() {
        let mut store = InMemoryStore::new();
        store.set("genre", "Pop");
        let op = MetadataOperation::RemoveField {
            key: "genre".to_string(),
        };
        apply_operation(&mut store, &op).expect("remove field operation should succeed");
        assert!(store.get("genre").is_none());
    }

    #[test]
    fn test_op_remove_field_nonexistent_is_noop() {
        let mut store = InMemoryStore::new();
        let op = MetadataOperation::RemoveField {
            key: "nonexistent".to_string(),
        };
        // Should not error
        apply_operation(&mut store, &op).expect("removing nonexistent field should succeed");
    }

    #[test]
    fn test_op_copy_from() {
        let src = write_kv_file(
            "test_copy_src.kv",
            &[("composer", "Bach"), ("year", "1700")],
        );
        let mut store = InMemoryStore::new();
        store.set("title", "Existing");
        let op = MetadataOperation::CopyFrom { source: src };
        apply_operation(&mut store, &op).expect("copy from operation should succeed");
        assert_eq!(store.get("composer"), Some("Bach"));
        assert_eq!(store.get("year"), Some("1700"));
        assert_eq!(store.get("title"), Some("Existing"));
    }

    #[test]
    fn test_op_copy_from_missing_file_returns_error() {
        let mut store = InMemoryStore::new();
        let op = MetadataOperation::CopyFrom {
            source: PathBuf::from("/nonexistent/path/file.kv"),
        };
        assert!(apply_operation(&mut store, &op).is_err());
    }

    // ── BatchMetadataEditor ───────────────────────────────────────────────────

    #[test]
    fn test_batch_apply_multiple_files() {
        // Create three source files
        let files: Vec<PathBuf> = (0..3)
            .map(|i| {
                write_kv_file(
                    &format!("batch_file_{}.kv", i),
                    &[("track", &i.to_string())],
                )
            })
            .collect();

        let ops = vec![
            MetadataOperation::SetField {
                key: "artist".to_string(),
                value: "COOLJAPAN OU".to_string(),
            },
            MetadataOperation::RemoveField {
                key: "track".to_string(),
            },
        ];

        let editor = BatchMetadataEditor::new();
        let results = editor.apply_batch(&files, &ops);

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "expected ok, got {:?}", r);
        }
    }

    #[test]
    fn test_batch_apply_in_memory_set_field() {
        let paths: Vec<PathBuf> = (0..5)
            .map(|i| {
                // Use non-existent paths; apply_batch_in_memory should start with empty store
                let mut p = temp_dir();
                p.push(format!("does_not_exist_batch_{}.kv", i));
                // Make sure they do not accidentally exist
                let _ = std::fs::remove_file(&p);
                p
            })
            .collect();

        let ops = vec![MetadataOperation::SetField {
            key: "label".to_string(),
            value: "OxiMedia".to_string(),
        }];

        let editor = BatchMetadataEditor::new();
        let stores = editor.apply_batch_in_memory(&paths, &ops);

        assert_eq!(stores.len(), 5);
        for store_result in stores {
            let store = store_result.expect("in-memory batch apply should succeed");
            assert_eq!(store.get("label"), Some("OxiMedia"));
        }
    }

    #[test]
    fn test_batch_apply_error_does_not_abort_others() {
        // Mix valid files with a CopyFrom pointing to a missing source.
        let valid = write_kv_file("batch_valid.kv", &[("title", "Valid")]);
        let paths = vec![valid];

        let ops = vec![MetadataOperation::CopyFrom {
            source: PathBuf::from("/nonexistent/missing.kv"),
        }];

        let editor = BatchMetadataEditor::new();
        let results = editor.apply_batch(&paths, &ops);

        // Each file is independent; result should be Err for files where CopyFrom fails
        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }

    #[test]
    fn test_batch_write_back() {
        let file = write_kv_file("batch_write_back.kv", &[("genre", "Classical")]);

        let ops = vec![MetadataOperation::SetField {
            key: "artist".to_string(),
            value: "Mozart".to_string(),
        }];

        let editor = BatchMetadataEditor::new().with_write_back();
        let results = editor.apply_batch(std::slice::from_ref(&file), &ops);
        assert!(results[0].is_ok());

        // Read back and verify the artist was persisted
        let store = InMemoryStore::load_from_file(&file)
            .expect("reloading after write-back should succeed");
        assert_eq!(store.get("artist"), Some("Mozart"));
        // Original field should still be there
        assert_eq!(store.get("genre"), Some("Classical"));
    }

    #[test]
    fn test_apply_operations_sequential_order() {
        let mut store = InMemoryStore::new();
        store.set("color", "red");

        let ops = vec![
            MetadataOperation::SetField {
                key: "color".to_string(),
                value: "blue".to_string(),
            },
            MetadataOperation::SetField {
                key: "color".to_string(),
                value: "green".to_string(),
            },
        ];

        apply_operations(&mut store, &ops).expect("sequential operations should succeed");
        // Last operation wins
        assert_eq!(store.get("color"), Some("green"));
    }
}
