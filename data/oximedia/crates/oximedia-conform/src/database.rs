//! `SQLite` database for media catalog and conform sessions.

use crate::error::ConformResult;
use crate::types::{FrameRate, MediaFile, Timecode};
use chrono::{DateTime, Utc};
use oxisql_core::{ToSqlValue, Value};
use oxisql_sqlite_compat::SqliteConnectionBlocking;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Database schema version.
///
/// Version history:
/// - 1: initial schema
/// - 2: adds `file_mtime INTEGER` and `file_size INTEGER` to `media_files`
const SCHEMA_VERSION: i64 = 2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn map_db(e: impl std::fmt::Display) -> crate::error::ConformError {
    crate::error::ConformError::Database(e.to_string())
}

fn col_text(row: &oxisql_core::Row, idx: usize) -> ConformResult<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Ok(s.clone()),
        Some(other) => Err(crate::error::ConformError::Database(format!(
            "column {idx}: expected text, got {}",
            other.type_name()
        ))),
        None => Err(crate::error::ConformError::Database(format!(
            "column {idx} missing from row"
        ))),
    }
}

fn col_opt_text(row: &oxisql_core::Row, idx: usize) -> Option<String> {
    match row.get_by_index(idx) {
        Some(Value::Text(s)) => Some(s.clone()),
        _ => None,
    }
}

fn col_opt_real(row: &oxisql_core::Row, idx: usize) -> Option<f64> {
    match row.get_by_index(idx) {
        Some(Value::F64(f)) => Some(*f),
        Some(Value::I64(n)) => Some(*n as f64),
        _ => None,
    }
}

fn col_opt_i64(row: &oxisql_core::Row, idx: usize) -> Option<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Some(*n),
        _ => None,
    }
}

fn col_i64(row: &oxisql_core::Row, idx: usize) -> ConformResult<i64> {
    match row.get_by_index(idx) {
        Some(Value::I64(n)) => Ok(*n),
        Some(Value::Null) | None => Ok(0),
        Some(other) => Err(crate::error::ConformError::Database(format!(
            "column {idx}: expected integer, got {}",
            other.type_name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct Inner {
    conn: SqliteConnectionBlocking,
}

impl Inner {
    fn exec(&self, sql: &str, params: &[&dyn ToSqlValue]) -> ConformResult<u64> {
        self.conn.execute(sql, params).map_err(map_db)
    }

    fn query(&self, sql: &str, params: &[&dyn ToSqlValue]) -> ConformResult<Vec<oxisql_core::Row>> {
        self.conn.query(sql, params).map_err(map_db)
    }

    fn execute_batch(&self, sql: &str) -> ConformResult<()> {
        // `execute_batch` returns the number of affected rows; the catalog only
        // issues DDL here so the count is discarded.
        self.conn.execute_batch(sql).map(|_| ()).map_err(map_db)
    }
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// Database manager for media catalog.
#[derive(Clone)]
pub struct Database {
    inner: Arc<Mutex<Inner>>,
}

impl Database {
    fn with_inner<F, T>(&self, f: F) -> ConformResult<T>
    where
        F: FnOnce(&Inner) -> ConformResult<T>,
    {
        let guard = self
            .inner
            .lock()
            .map_err(|_| crate::error::ConformError::Database("mutex poisoned".to_string()))?;
        f(&guard)
    }

    /// Create or open a database at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created or opened.
    pub fn open<P: AsRef<Path>>(path: P) -> ConformResult<Self> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| {
                crate::error::ConformError::Database(
                    "database path contains non-UTF-8 characters".to_string(),
                )
            })?
            .to_string();

        let conn = SqliteConnectionBlocking::open(&path_str).map_err(map_db)?;

        let db = Self {
            inner: Arc::new(Mutex::new(Inner { conn })),
        };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Create an in-memory database (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub fn in_memory() -> ConformResult<Self> {
        let conn = SqliteConnectionBlocking::open_memory().map_err(map_db)?;

        let db = Self {
            inner: Arc::new(Mutex::new(Inner { conn })),
        };
        db.initialize_schema()?;
        Ok(db)
    }

    /// Initialize the database schema.
    fn initialize_schema(&self) -> ConformResult<()> {
        // Create schema_version table first.
        self.with_inner(|inner| {
            inner.execute_batch(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY)",
            )
        })?;

        let version_rows = self
            .with_inner(|inner| inner.query("SELECT version FROM schema_version LIMIT 1", &[]))?;

        let version: Option<i64> = version_rows.first().and_then(|r| col_opt_i64(r, 0));

        match version {
            None => {
                self.create_tables()?;
                let v = SCHEMA_VERSION;
                self.with_inner(|inner| {
                    inner.exec("INSERT INTO schema_version (version) VALUES ($1)", &[&v])
                })?;
            }
            Some(v) if v < 2 => {
                self.migrate_v1_to_v2()?;
                let new_v = SCHEMA_VERSION;
                self.with_inner(|inner| {
                    inner.exec("UPDATE schema_version SET version = $1", &[&new_v])
                })?;
            }
            _ => {
                // Already at current version — nothing to do.
            }
        }

        Ok(())
    }

    /// Check whether `col_name` exists in `media_files` by querying PRAGMA.
    fn column_exists(&self, col_name: &str) -> ConformResult<bool> {
        let rows = self.with_inner(|inner| inner.query("PRAGMA table_info(media_files)", &[]))?;
        // Column 1 of PRAGMA table_info is the column name.
        Ok(rows
            .iter()
            .any(|row| col_opt_text(row, 1).map(|n| n == col_name).unwrap_or(false)))
    }

    /// Migrate from schema version 1 to version 2.
    fn migrate_v1_to_v2(&self) -> ConformResult<()> {
        if !self.column_exists("file_mtime")? {
            self.with_inner(|inner| {
                inner.exec("ALTER TABLE media_files ADD COLUMN file_mtime INTEGER", &[])
            })?;
        }

        if !self.column_exists("file_size")? {
            self.with_inner(|inner| {
                inner.exec("ALTER TABLE media_files ADD COLUMN file_size INTEGER", &[])
            })?;
        }

        Ok(())
    }

    /// Create database tables.
    fn create_tables(&self) -> ConformResult<()> {
        self.with_inner(|inner| {
            inner.execute_batch(
                "CREATE TABLE IF NOT EXISTS media_files (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    duration REAL,
                    timecode_start TEXT,
                    width INTEGER,
                    height INTEGER,
                    fps REAL,
                    size INTEGER,
                    md5 TEXT,
                    xxhash TEXT,
                    metadata TEXT,
                    cataloged_at TEXT NOT NULL,
                    file_mtime INTEGER,
                    file_size INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_media_filename ON media_files(filename);
                CREATE INDEX IF NOT EXISTS idx_media_path ON media_files(path);
                CREATE INDEX IF NOT EXISTS idx_media_md5 ON media_files(md5);
                CREATE TABLE IF NOT EXISTS conform_sessions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    edl_path TEXT,
                    status TEXT NOT NULL,
                    config TEXT
                );
                CREATE TABLE IF NOT EXISTS clip_matches (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    clip_id TEXT NOT NULL,
                    media_file_id TEXT NOT NULL,
                    match_score REAL NOT NULL,
                    match_method TEXT NOT NULL,
                    details TEXT,
                    FOREIGN KEY(session_id) REFERENCES conform_sessions(id),
                    FOREIGN KEY(media_file_id) REFERENCES media_files(id)
                );
                CREATE INDEX IF NOT EXISTS idx_clip_matches_session ON clip_matches(session_id);",
            )
        })
    }

    /// Add a media file to the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub fn add_media_file(&self, media: &MediaFile) -> ConformResult<()> {
        let id_s = media.id.to_string();
        let path_s = media.path.to_string_lossy().to_string();
        let timecode_s: Option<String> = media.timecode_start.map(|tc| tc.to_string());
        let fps_f: Option<f64> = media.fps.map(|fr| fr.as_f64());
        let size_i: Option<i64> = media.size.map(|s| s as i64);
        let width_i: Option<i64> = media.width.map(i64::from);
        let height_i: Option<i64> = media.height.map(i64::from);
        let cataloged_s = media.cataloged_at.to_rfc3339();

        self.with_inner(|inner| {
            inner.exec(
                "INSERT OR REPLACE INTO media_files
                 (id, path, filename, duration, timecode_start, width, height, fps,
                  size, md5, xxhash, metadata, cataloged_at)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
                &[
                    &id_s,
                    &path_s,
                    &media.filename.as_str(),
                    &media.duration as &dyn ToSqlValue,
                    &timecode_s.as_deref() as &dyn ToSqlValue,
                    &width_i as &dyn ToSqlValue,
                    &height_i as &dyn ToSqlValue,
                    &fps_f as &dyn ToSqlValue,
                    &size_i as &dyn ToSqlValue,
                    &media.md5.as_deref() as &dyn ToSqlValue,
                    &media.xxhash.as_deref() as &dyn ToSqlValue,
                    &media.metadata.as_deref() as &dyn ToSqlValue,
                    &cataloged_s,
                ],
            )?;
            Ok(())
        })
    }

    /// Find media files by filename.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn find_by_filename(&self, filename: &str) -> ConformResult<Vec<MediaFile>> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, path, filename, duration, timecode_start, width, height, fps,
                  size, md5, xxhash, metadata, cataloged_at
                 FROM media_files WHERE filename = $1",
                &[&filename],
            )
        })?;

        rows.iter().map(Self::row_to_media_file).collect()
    }

    /// Find media files by path pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn find_by_path_pattern(&self, pattern: &str) -> ConformResult<Vec<MediaFile>> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, path, filename, duration, timecode_start, width, height, fps,
                  size, md5, xxhash, metadata, cataloged_at
                 FROM media_files WHERE path LIKE $1",
                &[&pattern],
            )
        })?;

        rows.iter().map(Self::row_to_media_file).collect()
    }

    /// Find media files by MD5 hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn find_by_md5(&self, md5: &str) -> ConformResult<Vec<MediaFile>> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, path, filename, duration, timecode_start, width, height, fps,
                  size, md5, xxhash, metadata, cataloged_at
                 FROM media_files WHERE md5 = $1",
                &[&md5],
            )
        })?;

        rows.iter().map(Self::row_to_media_file).collect()
    }

    /// Get all media files.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_all_media_files(&self) -> ConformResult<Vec<MediaFile>> {
        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT id, path, filename, duration, timecode_start, width, height, fps,
                  size, md5, xxhash, metadata, cataloged_at
                 FROM media_files ORDER BY cataloged_at DESC",
                &[],
            )
        })?;

        rows.iter().map(Self::row_to_media_file).collect()
    }

    /// Get media file count.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_media_file_count(&self) -> ConformResult<usize> {
        let rows = self.with_inner(|inner| inner.query("SELECT COUNT(*) FROM media_files", &[]))?;
        let count = col_i64(
            rows.first()
                .ok_or_else(|| crate::error::ConformError::Database("no rows".to_string()))?,
            0,
        )?;
        #[allow(clippy::cast_sign_loss)]
        Ok(count as usize)
    }

    /// Remove a media file from the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if the deletion fails.
    pub fn remove_media_file(&self, id: &uuid::Uuid) -> ConformResult<()> {
        let id_s = id.to_string();
        self.with_inner(|inner| {
            inner.exec("DELETE FROM media_files WHERE id = $1", &[&id_s])?;
            Ok(())
        })
    }

    /// Clear all media files from the catalog.
    ///
    /// # Errors
    ///
    /// Returns an error if the deletion fails.
    pub fn clear_media_files(&self) -> ConformResult<()> {
        self.with_inner(|inner| {
            inner.exec("DELETE FROM media_files", &[])?;
            Ok(())
        })
    }

    // ── Incremental scan helpers ─────────────────────────────────────────────

    /// Check whether the catalog already has an up-to-date entry for `path`.
    ///
    /// Returns `true` iff the catalog contains a row whose `path`, `file_size`,
    /// and `file_mtime` all match the supplied values.
    ///
    /// `mtime` is expressed as Unix seconds (u64).
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn has_unchanged(&self, path: &Path, size: u64, mtime: u64) -> ConformResult<bool> {
        let path_s = path.to_string_lossy().to_string();
        let size_i = size as i64;
        let mtime_i = mtime as i64;

        let rows = self.with_inner(|inner| {
            inner.query(
                "SELECT COUNT(*) FROM media_files
                 WHERE path = $1 AND file_size = $2 AND file_mtime = $3",
                &[&path_s, &size_i, &mtime_i],
            )
        })?;

        let count = col_i64(
            rows.first()
                .ok_or_else(|| crate::error::ConformError::Database("no rows".to_string()))?,
            0,
        )?;
        Ok(count > 0)
    }

    /// Record or refresh the `file_mtime` and `file_size` for a path.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_mtime(&self, path: &Path, size: u64, mtime: u64) -> ConformResult<()> {
        let path_s = path.to_string_lossy().to_string();
        let mtime_i = mtime as i64;
        let size_i = size as i64;

        self.with_inner(|inner| {
            inner.exec(
                "UPDATE media_files SET file_mtime = $1, file_size = $2 WHERE path = $3",
                &[&mtime_i, &size_i, &path_s],
            )?;
            Ok(())
        })
    }

    /// Create a new conform session.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub fn create_session(
        &self,
        id: uuid::Uuid,
        name: &str,
        edl_path: Option<&Path>,
        config: &str,
    ) -> ConformResult<()> {
        let id_s = id.to_string();
        let now = Utc::now().to_rfc3339();
        let edl_s: Option<String> = edl_path.map(|p| p.to_string_lossy().to_string());

        self.with_inner(|inner| {
            inner.exec(
                "INSERT INTO conform_sessions
                 (id, name, created_at, updated_at, edl_path, status, config)
                 VALUES ($1, $2, $3, $4, $5, $6, $7)",
                &[
                    &id_s,
                    &name,
                    &now,
                    &now,
                    &edl_s.as_deref() as &dyn ToSqlValue,
                    &"created",
                    &config,
                ],
            )?;
            Ok(())
        })
    }

    /// Update session status.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_session_status(&self, id: &uuid::Uuid, status: &str) -> ConformResult<()> {
        let id_s = id.to_string();
        let now = Utc::now().to_rfc3339();

        self.with_inner(|inner| {
            inner.exec(
                "UPDATE conform_sessions SET status = $1, updated_at = $2 WHERE id = $3",
                &[&status, &now, &id_s],
            )?;
            Ok(())
        })
    }

    /// Convert a database row to a `MediaFile`.
    fn row_to_media_file(row: &oxisql_core::Row) -> ConformResult<MediaFile> {
        let id_s = col_text(row, 0)?;
        let path_s = col_text(row, 1)?;
        let filename = col_text(row, 2)?;
        let duration = col_opt_real(row, 3);
        let timecode_s = col_opt_text(row, 4);
        let width = col_opt_i64(row, 5).map(|n| n as u32);
        let height = col_opt_i64(row, 6).map(|n| n as u32);
        let fps_f = col_opt_real(row, 7);
        let size = col_opt_i64(row, 8).map(|n| n as u64);
        let md5 = col_opt_text(row, 9);
        let xxhash = col_opt_text(row, 10);
        let metadata = col_opt_text(row, 11);
        let cataloged_s = col_text(row, 12)?;

        let id = uuid::Uuid::parse_str(&id_s)
            .map_err(|e| crate::error::ConformError::Database(format!("uuid parse: {e}")))?;

        let cataloged_at = DateTime::parse_from_rfc3339(&cataloged_s)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| crate::error::ConformError::Database(format!("datetime parse: {e}")))?;

        Ok(MediaFile {
            id,
            path: PathBuf::from(path_s),
            filename,
            duration,
            timecode_start: timecode_s.and_then(|s| Timecode::parse(&s).ok()),
            width,
            height,
            fps: fps_f.map(FrameRate::Custom),
            size,
            md5,
            xxhash,
            metadata,
            cataloged_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let db = Database::in_memory().expect("db should be valid");
        assert_eq!(
            db.get_media_file_count()
                .expect("get_media_file_count should succeed"),
            0
        );
    }

    #[test]
    fn test_add_media_file() {
        let db = Database::in_memory().expect("db should be valid");
        let media = MediaFile::new(PathBuf::from("/test/file.mov"));
        db.add_media_file(&media)
            .expect("add_media_file should succeed");
        assert_eq!(
            db.get_media_file_count()
                .expect("get_media_file_count should succeed"),
            1
        );
    }

    #[test]
    fn test_find_by_filename() {
        let db = Database::in_memory().expect("db should be valid");
        let media = MediaFile::new(PathBuf::from("/test/file.mov"));
        db.add_media_file(&media)
            .expect("add_media_file should succeed");

        let found = db
            .find_by_filename("file.mov")
            .expect("found should be valid");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].filename, "file.mov");
    }

    #[test]
    fn test_clear_media_files() {
        let db = Database::in_memory().expect("db should be valid");
        let media1 = MediaFile::new(PathBuf::from("/test/file1.mov"));
        let media2 = MediaFile::new(PathBuf::from("/test/file2.mov"));
        db.add_media_file(&media1)
            .expect("add_media_file should succeed");
        db.add_media_file(&media2)
            .expect("add_media_file should succeed");
        assert_eq!(
            db.get_media_file_count()
                .expect("get_media_file_count should succeed"),
            2
        );

        db.clear_media_files()
            .expect("clear_media_files should succeed");
        assert_eq!(
            db.get_media_file_count()
                .expect("get_media_file_count should succeed"),
            0
        );
    }

    #[test]
    fn test_session_creation() {
        let db = Database::in_memory().expect("db should be valid");
        let session_id = uuid::Uuid::new_v4();
        db.create_session(session_id, "Test Session", None, "{}")
            .expect("create_session should succeed");
        db.update_session_status(&session_id, "completed")
            .expect("update_session_status should succeed");
    }

    // ── Incremental scan tests ────────────────────────────────────────────────

    /// An unchanged file must be recognised as unchanged.
    #[test]
    fn test_incremental_scan_skips_unchanged_file() {
        let db = Database::in_memory().expect("in-memory db must open");
        let path = PathBuf::from("/media/project/A001C001.mov");
        let size: u64 = 1_234_567;
        let mtime: u64 = 1_717_200_000;

        assert!(
            !db.has_unchanged(&path, size, mtime)
                .expect("has_unchanged must not error"),
            "file not yet cataloged must return false"
        );

        let mut media = MediaFile::new(path.clone());
        media.size = Some(size);
        db.add_media_file(&media)
            .expect("add_media_file must succeed");
        db.update_mtime(&path, size, mtime)
            .expect("update_mtime must succeed");

        assert!(
            db.has_unchanged(&path, size, mtime)
                .expect("has_unchanged must not error"),
            "unchanged file must be recognised as unchanged after ingest"
        );
    }

    /// When the mtime is bumped, has_unchanged must return false.
    #[test]
    fn test_incremental_scan_reingests_on_mtime_bump() {
        let db = Database::in_memory().expect("in-memory db must open");
        let path = PathBuf::from("/media/project/B002C002.mxf");
        let size: u64 = 9_876_543;
        let mtime_old: u64 = 1_717_200_000;
        let mtime_new: u64 = mtime_old + 60;

        let mut media = MediaFile::new(path.clone());
        media.size = Some(size);
        db.add_media_file(&media)
            .expect("add_media_file must succeed");
        db.update_mtime(&path, size, mtime_old)
            .expect("update_mtime must succeed");

        assert!(
            db.has_unchanged(&path, size, mtime_old)
                .expect("has_unchanged must not error"),
            "old mtime must be recognised as unchanged"
        );

        assert!(
            !db.has_unchanged(&path, size, mtime_new)
                .expect("has_unchanged must not error"),
            "bumped mtime must NOT be recognised as unchanged"
        );
    }
}
