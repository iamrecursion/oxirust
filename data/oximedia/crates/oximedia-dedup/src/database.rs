//! SQLite-based deduplication database and indexing.
//!
//! # Schema
//!
//! ## Tables
//!
//! ### `files`
//! Primary index of all known media files.
//!
//! | Column       | Type    | Notes                              |
//! |--------------|---------|------------------------------------|
//! | `id`         | INTEGER | Auto-increment primary key         |
//! | `path`       | TEXT    | Absolute file path (UNIQUE)        |
//! | `size`       | INTEGER | File size in bytes                 |
//! | `hash`       | TEXT    | BLAKE3 hex digest                  |
//! | `created_at` | INTEGER | Unix epoch (seconds, via `strftime`) |
//! | `updated_at` | INTEGER | Unix epoch, updated on upsert      |
//!
//! **Indices:** `idx_files_hash` on `hash` for O(log n) duplicate lookup.
//!
//! ### `fingerprints`
//! Perceptual / audio fingerprints keyed by `file_id`.
//!
//! | Column       | Type    | Notes                                   |
//! |--------------|---------|-----------------------------------------|
//! | `id`         | INTEGER | Auto-increment primary key              |
//! | `file_id`    | INTEGER | FK → `files.id` (CASCADE DELETE)        |
//! | `type`       | TEXT    | Fingerprint kind (`phash`, `audio`, …)  |
//! | `data`       | TEXT    | Hex-encoded fingerprint bits            |
//! | `created_at` | INTEGER | Unix epoch                              |
//!
//! **Indices:** `idx_fingerprints_type` on `type`; `idx_fingerprints_data` on `data`.
//!
//! ### `metadata`
//! Optional media metadata for codec/format-based matching.
//!
//! | Column        | Type    | Notes                     |
//! |---------------|---------|---------------------------|
//! | `id`          | INTEGER | Auto-increment primary key |
//! | `file_id`     | INTEGER | FK → `files.id`            |
//! | `duration`    | REAL    | Duration in seconds        |
//! | `width`       | INTEGER | Video width                |
//! | `height`      | INTEGER | Video height               |
//! | `bitrate`     | INTEGER | Bitrate (bits/s)           |
//! | `framerate`   | REAL    | Frames per second          |
//! | `sample_rate` | INTEGER | Audio sample rate (Hz)     |
//! | `channels`    | INTEGER | Audio channel count        |
//! | `video_codec` | TEXT    | Codec name (e.g. `av1`)   |
//! | `audio_codec` | TEXT    | Codec name (e.g. `opus`)  |
//! | `container`   | TEXT    | Container (e.g. `mp4`)    |
//!
//! ### `chunks`
//! Rolling-hash content chunks for sub-file deduplication.
//!
//! | Column    | Type    | Notes                             |
//! |-----------|---------|-----------------------------------|
//! | `id`      | INTEGER | Auto-increment primary key        |
//! | `file_id` | INTEGER | FK → `files.id` (CASCADE DELETE)  |
//! | `offset`  | INTEGER | Byte offset within file           |
//! | `size`    | INTEGER | Chunk size in bytes               |
//! | `hash`    | TEXT    | Rolling-hash hex digest           |
//!
//! **Indices:** `idx_chunks_hash` on `hash`.
//!
//! ## Migration Strategy
//!
//! All `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` statements run
//! on every `open()` / `open_memory()` call (inside the private `initialize` helper).
//! This is an additive-only, forward-only strategy — no DDL `ALTER TABLE` or
//! schema version table is maintained in v0.1.x.  A proper `schema_version` table
//! with up/down migrations will be added before the first stable release.
//!
//! # Backend
//!
//! Backed by [`oxisql_sqlite_compat::SqliteConnection`] — a Pure-Rust,
//! C/C++-free SQLite-compatible engine (wraps `oxisqlite`, a C-free fork of
//! Limbo).  No `libsqlite3-sys` / `rusqlite` / `sqlx-sqlite` dependency is
//! pulled in, keeping the opt-in `sqlite` feature Pure Rust as well.

use crate::{DedupError, DedupResult};
use oxisql_core::Connection;
use std::collections::HashMap;
use std::path::Path;

/// A single entry for bulk insertion via [`DedupDatabase::insert_batch`].
///
/// The file referenced by `path` **must exist on disk** at the time of the call
/// because the implementation reads `std::fs::metadata` to obtain the file size.
#[derive(Debug, Clone)]
pub struct BatchFileEntry {
    /// Absolute path to the media file.
    pub path: String,
    /// Hex-encoded content hash (BLAKE3 recommended).
    pub hash: String,
}

/// SQLite database for deduplication.
pub struct DedupDatabase {
    pool: oxisql_sqlite_compat::SqliteConnection,
}

/// Pure DDL schema — `CREATE TABLE` / `CREATE INDEX` only, no PRAGMA/VACUUM,
/// so it can be issued in a single `execute_batch` call.
const SCHEMA_DDL: &str = r#"
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        size INTEGER NOT NULL,
        hash TEXT NOT NULL,
        created_at INTEGER DEFAULT (strftime('%s', 'now')),
        updated_at INTEGER DEFAULT (strftime('%s', 'now'))
    );
    CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
    CREATE TABLE IF NOT EXISTS fingerprints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at INTEGER DEFAULT (strftime('%s', 'now')),
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_fingerprints_type ON fingerprints(type);
    CREATE INDEX IF NOT EXISTS idx_fingerprints_data ON fingerprints(data);
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        duration REAL,
        width INTEGER,
        height INTEGER,
        bitrate INTEGER,
        framerate REAL,
        sample_rate INTEGER,
        channels INTEGER,
        video_codec TEXT,
        audio_codec TEXT,
        container TEXT,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL,
        offset INTEGER NOT NULL,
        size INTEGER NOT NULL,
        hash TEXT NOT NULL,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);
"#;

impl DedupDatabase {
    /// Open or create a database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub async fn open(path: impl AsRef<Path>) -> DedupResult<Self> {
        let path = path.as_ref();

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let path_str = path.to_string_lossy().to_string();
        let pool = oxisql_sqlite_compat::SqliteConnection::open(&path_str)
            .await
            .map_err(DedupError::Database)?;

        let db = Self { pool };
        db.initialize().await?;

        Ok(db)
    }

    /// Open in-memory database (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub async fn open_memory() -> DedupResult<Self> {
        let pool = oxisql_sqlite_compat::SqliteConnection::open_memory()
            .await
            .map_err(DedupError::Database)?;

        let db = Self { pool };
        db.initialize().await?;

        Ok(db)
    }

    /// Initialize database schema.
    async fn initialize(&self) -> DedupResult<()> {
        self.pool
            .execute_batch(SCHEMA_DDL)
            .await
            .map_err(DedupError::Database)?;
        Ok(())
    }

    /// Insert a file into the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    ///
    /// # Implementation note
    ///
    /// The underlying Pure-Rust `oxisqlite` engine (`oxisql-sqlite-compat`
    /// 0.3.x) parses `RETURNING` but does not yet execute it (a silent
    /// no-op — see upstream `translate/insert.rs`'s unused `_returning`
    /// parameter), and `last_insert_rowid()` is not reliable here because
    /// this is an upsert (`ON CONFLICT … DO UPDATE`): on the update path no
    /// new row is inserted, so it would not reflect the conflicting row's
    /// id. A follow-up `SELECT id … WHERE path = $1` is used instead, which
    /// is correct on both the insert and update-on-conflict paths.
    pub async fn insert_file(&self, path: impl AsRef<Path>, hash: &str) -> DedupResult<i64> {
        let path = path.as_ref().to_string_lossy().to_string();
        let size = std::fs::metadata(path.as_str())?.len() as i64;

        self.pool
            .execute(
                r"
            INSERT INTO files (path, size, hash)
            VALUES ($1, $2, $3)
            ON CONFLICT(path) DO UPDATE SET
                size = excluded.size,
                hash = excluded.hash,
                updated_at = strftime('%s', 'now')
            ",
                &[&path, &size, &hash],
            )
            .await
            .map_err(DedupError::Database)?;

        let rows = self
            .pool
            .query("SELECT id FROM files WHERE path = $1", &[&path])
            .await
            .map_err(DedupError::Database)?;

        rows.first()
            .ok_or_else(|| DedupError::Other("insert_file: row not found after upsert".into()))?
            .try_get_by_index(0)
            .map_err(DedupError::Database)
    }

    /// Get file ID by path.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_file_id(&self, path: impl AsRef<Path>) -> DedupResult<Option<i64>> {
        let path = path.as_ref().to_string_lossy().to_string();

        let rows = self
            .pool
            .query("SELECT id FROM files WHERE path = $1", &[&path])
            .await
            .map_err(DedupError::Database)?;

        rows.first()
            .map(|row| row.try_get_by_index(0).map_err(DedupError::Database))
            .transpose()
    }

    /// Insert a fingerprint.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub async fn insert_fingerprint(
        &self,
        file_id: i64,
        fingerprint_type: &str,
        data: &str,
    ) -> DedupResult<i64> {
        self.pool
            .execute(
                r"
            INSERT INTO fingerprints (file_id, type, data)
            VALUES ($1, $2, $3)
            ",
                &[&file_id, &fingerprint_type, &data],
            )
            .await
            .map_err(DedupError::Database)?;

        self.last_insert_rowid().await
    }

    /// Insert metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_metadata(
        &self,
        file_id: i64,
        duration: Option<f64>,
        width: Option<i32>,
        height: Option<i32>,
        video_codec: Option<&str>,
        audio_codec: Option<&str>,
        container: Option<&str>,
    ) -> DedupResult<i64> {
        self.pool
            .execute(
                r"
            INSERT INTO metadata (file_id, duration, width, height, video_codec, audio_codec, container)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ",
                &[
                    &file_id,
                    &duration,
                    &width,
                    &height,
                    &video_codec,
                    &audio_codec,
                    &container,
                ],
            )
            .await
            .map_err(DedupError::Database)?;

        self.last_insert_rowid().await
    }

    /// Insert a chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if the insertion fails.
    pub async fn insert_chunk(
        &self,
        file_id: i64,
        offset: i64,
        size: i64,
        hash: &str,
    ) -> DedupResult<i64> {
        self.pool
            .execute(
                r"
            INSERT INTO chunks (file_id, offset, size, hash)
            VALUES ($1, $2, $3, $4)
            ",
                &[&file_id, &offset, &size, &hash],
            )
            .await
            .map_err(DedupError::Database)?;

        self.last_insert_rowid().await
    }

    /// Fetch the rowid of the most recently successful `INSERT` on this
    /// connection via SQLite's `last_insert_rowid()` scalar function.
    ///
    /// Must only be called immediately after an `INSERT` that is known to
    /// have inserted a new row (not an `ON CONFLICT … DO UPDATE` that may
    /// have taken the update path — see [`Self::insert_file`]).
    async fn last_insert_rowid(&self) -> DedupResult<i64> {
        let rows = self
            .pool
            .query("SELECT last_insert_rowid()", &[])
            .await
            .map_err(DedupError::Database)?;
        rows.first()
            .ok_or_else(|| DedupError::Other("last_insert_rowid(): no row returned".into()))?
            .try_get_by_index(0)
            .map_err(DedupError::Database)
    }

    /// Find files with duplicate hashes.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_duplicate_hashes(&self) -> DedupResult<HashMap<String, Vec<String>>> {
        let rows = self
            .pool
            .query(
                r"
            SELECT hash, path
            FROM files
            WHERE hash IN (
                SELECT hash
                FROM files
                GROUP BY hash
                HAVING COUNT(*) > 1
            )
            ORDER BY hash, path
            ",
                &[],
            )
            .await
            .map_err(DedupError::Database)?;

        let mut duplicates: HashMap<String, Vec<String>> = HashMap::new();

        for row in rows {
            let hash: String = row.try_get_by_index(0).map_err(DedupError::Database)?;
            let path: String = row.try_get_by_index(1).map_err(DedupError::Database)?;

            duplicates.entry(hash).or_insert_with(Vec::new).push(path);
        }

        Ok(duplicates)
    }

    /// Find files with similar fingerprints.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_similar_fingerprints(
        &self,
        fingerprint_type: &str,
    ) -> DedupResult<HashMap<String, Vec<String>>> {
        let rows = self
            .pool
            .query(
                r"
            SELECT f.data, fi.path
            FROM fingerprints f
            JOIN files fi ON f.file_id = fi.id
            WHERE f.type = $1
            ORDER BY f.data
            ",
                &[&fingerprint_type],
            )
            .await
            .map_err(DedupError::Database)?;

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();

        for row in rows {
            let data: String = row.try_get_by_index(0).map_err(DedupError::Database)?;
            let path: String = row.try_get_by_index(1).map_err(DedupError::Database)?;

            groups.entry(data).or_insert_with(Vec::new).push(path);
        }

        // Filter to only groups with multiple files
        Ok(groups
            .into_iter()
            .filter(|(_, paths)| paths.len() > 1)
            .collect())
    }

    /// Find duplicate chunks.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_duplicate_chunks(&self) -> DedupResult<HashMap<String, Vec<String>>> {
        let rows = self
            .pool
            .query(
                r"
            SELECT c.hash, f.path
            FROM chunks c
            JOIN files f ON c.file_id = f.id
            WHERE c.hash IN (
                SELECT hash
                FROM chunks
                GROUP BY hash
                HAVING COUNT(*) > 1
            )
            ORDER BY c.hash
            ",
                &[],
            )
            .await
            .map_err(DedupError::Database)?;

        let mut duplicates: HashMap<String, Vec<String>> = HashMap::new();

        for row in rows {
            let hash: String = row.try_get_by_index(0).map_err(DedupError::Database)?;
            let path: String = row.try_get_by_index(1).map_err(DedupError::Database)?;

            let paths = duplicates.entry(hash).or_insert_with(Vec::new);
            if !paths.contains(&path) {
                paths.push(path);
            }
        }

        Ok(duplicates)
    }

    /// Get all files.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_all_files(&self) -> DedupResult<Vec<(String, String)>> {
        let rows = self
            .pool
            .query("SELECT path, hash FROM files ORDER BY path", &[])
            .await
            .map_err(DedupError::Database)?;

        rows.iter()
            .map(|row| {
                let path: String = row.try_get_by_index(0).map_err(DedupError::Database)?;
                let hash: String = row.try_get_by_index(1).map_err(DedupError::Database)?;
                Ok((path, hash))
            })
            .collect()
    }

    /// Count total files.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn count_files(&self) -> DedupResult<usize> {
        let rows = self
            .pool
            .query("SELECT COUNT(*) FROM files", &[])
            .await
            .map_err(DedupError::Database)?;

        let count: i64 = rows
            .first()
            .ok_or_else(|| DedupError::Other("count_files: no row returned".into()))?
            .try_get_by_index(0)
            .map_err(DedupError::Database)?;
        Ok(count as usize)
    }

    /// Count unique hashes.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn count_unique_hashes(&self) -> DedupResult<usize> {
        let rows = self
            .pool
            .query("SELECT COUNT(DISTINCT hash) FROM files", &[])
            .await
            .map_err(DedupError::Database)?;

        let count: i64 = rows
            .first()
            .ok_or_else(|| DedupError::Other("count_unique_hashes: no row returned".into()))?
            .try_get_by_index(0)
            .map_err(DedupError::Database)?;
        Ok(count as usize)
    }

    /// Delete file by path.
    ///
    /// # Errors
    ///
    /// Returns an error if the deletion fails.
    pub async fn delete_file(&self, path: impl AsRef<Path>) -> DedupResult<()> {
        let path = path.as_ref().to_string_lossy().to_string();

        self.pool
            .execute("DELETE FROM files WHERE path = $1", &[&path])
            .await
            .map_err(DedupError::Database)?;

        Ok(())
    }

    /// Delete files by hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the deletion fails.
    pub async fn delete_by_hash(&self, hash: &str) -> DedupResult<usize> {
        let affected = self
            .pool
            .execute("DELETE FROM files WHERE hash = $1", &[&hash])
            .await
            .map_err(DedupError::Database)?;

        Ok(affected as usize)
    }

    /// Optimize database (vacuum and analyze).
    ///
    /// `VACUUM` is not yet implemented by the underlying Pure-Rust `oxisqlite`
    /// engine (`oxisql-sqlite-compat` 0.3.x); the call is therefore
    /// best-effort and its error is swallowed so that `ANALYZE` — which *is*
    /// supported and rebuilds query-planner statistics — still runs.
    ///
    /// # Errors
    ///
    /// Returns an error if `ANALYZE` fails.
    pub async fn optimize(&self) -> DedupResult<()> {
        // Best-effort: not yet supported by the Pure-Rust SQLite engine.
        let _ = self.pool.execute("VACUUM", &[]).await;
        self.pool
            .execute("ANALYZE", &[])
            .await
            .map_err(DedupError::Database)?;
        Ok(())
    }

    /// Get database statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if queries fail.
    pub async fn get_stats(&self) -> DedupResult<DatabaseStats> {
        let total_files = self.count_files().await?;
        let unique_hashes = self.count_unique_hashes().await?;

        let rows = self
            .pool
            .query("SELECT COUNT(*) FROM fingerprints", &[])
            .await
            .map_err(DedupError::Database)?;
        let total_fingerprints: i64 = rows
            .first()
            .ok_or_else(|| DedupError::Other("get_stats: no fingerprint count row".into()))?
            .try_get_by_index(0)
            .map_err(DedupError::Database)?;

        let rows = self
            .pool
            .query("SELECT COUNT(*) FROM chunks", &[])
            .await
            .map_err(DedupError::Database)?;
        let total_chunks: i64 = rows
            .first()
            .ok_or_else(|| DedupError::Other("get_stats: no chunk count row".into()))?
            .try_get_by_index(0)
            .map_err(DedupError::Database)?;

        let rows = self
            .pool
            .query("SELECT SUM(size) FROM files", &[])
            .await
            .map_err(DedupError::Database)?;
        let total_size: Option<i64> = rows
            .first()
            .ok_or_else(|| DedupError::Other("get_stats: no size sum row".into()))?
            .try_get_by_index(0)
            .map_err(DedupError::Database)?;

        Ok(DatabaseStats {
            total_files,
            unique_hashes,
            total_fingerprints: total_fingerprints as usize,
            total_chunks: total_chunks as usize,
            total_size: total_size.unwrap_or(0) as u64,
        })
    }

    /// Get all files with their stored metadata.
    ///
    /// Returns a list of `(file_path, duration_secs, width, height, video_codec, audio_codec, container)`.
    /// Fields that were not stored are `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_all_files_with_metadata(
        &self,
    ) -> DedupResult<
        Vec<(
            String,
            Option<f64>,
            Option<i32>,
            Option<i32>,
            Option<String>,
            Option<String>,
            Option<String>,
        )>,
    > {
        let rows = self
            .pool
            .query(
                r"
            SELECT f.path,
                   m.duration,
                   m.width,
                   m.height,
                   m.video_codec,
                   m.audio_codec,
                   m.container
            FROM files f
            LEFT JOIN metadata m ON m.file_id = f.id
            ORDER BY f.path
            ",
                &[],
            )
            .await
            .map_err(DedupError::Database)?;

        let mut result = Vec::with_capacity(rows.len());
        for row in rows {
            let path: String = row.try_get_by_index(0).map_err(DedupError::Database)?;
            let duration: Option<f64> = row.try_get_by_index(1).map_err(DedupError::Database)?;
            let width: Option<i32> = row.try_get_by_index(2).map_err(DedupError::Database)?;
            let height: Option<i32> = row.try_get_by_index(3).map_err(DedupError::Database)?;
            let video_codec: Option<String> =
                row.try_get_by_index(4).map_err(DedupError::Database)?;
            let audio_codec: Option<String> =
                row.try_get_by_index(5).map_err(DedupError::Database)?;
            let container: Option<String> =
                row.try_get_by_index(6).map_err(DedupError::Database)?;
            result.push((
                path,
                duration,
                width,
                height,
                video_codec,
                audio_codec,
                container,
            ));
        }

        Ok(result)
    }

    /// Get all fingerprints of a given type together with the owning file path.
    ///
    /// Returns `(file_path, fingerprint_hex_string)` pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_all_fingerprints_by_type(
        &self,
        fingerprint_type: &str,
    ) -> DedupResult<Vec<(String, String)>> {
        let rows = self
            .pool
            .query(
                r"
            SELECT f.path, fp.data
            FROM fingerprints fp
            JOIN files f ON fp.file_id = f.id
            WHERE fp.type = $1
            ORDER BY f.path
            ",
                &[&fingerprint_type],
            )
            .await
            .map_err(DedupError::Database)?;

        rows.into_iter()
            .map(|row| {
                let path: String = row.try_get_by_index(0).map_err(DedupError::Database)?;
                let data: String = row.try_get_by_index(1).map_err(DedupError::Database)?;
                Ok((path, data))
            })
            .collect()
    }

    /// Insert multiple files in a single atomic transaction for throughput.
    ///
    /// All entries are wrapped in a single transaction. If any row fails the
    /// whole batch is rolled back and the error is returned.
    ///
    /// Each file referenced by [`BatchFileEntry::path`] **must exist on disk** because
    /// the method reads `std::fs::metadata` to determine file size.
    ///
    /// Returns the number of rows that were inserted or updated (upserted).
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be started, if any file's metadata
    /// cannot be read from disk, or if any SQL statement fails.
    pub async fn insert_batch(&self, entries: &[BatchFileEntry]) -> DedupResult<usize> {
        let mut tx = self
            .pool
            .transaction()
            .await
            .map_err(DedupError::Database)?;
        let mut count = 0usize;

        for entry in entries {
            let size = std::fs::metadata(entry.path.as_str())?.len() as i64;

            let result = tx
                .execute(
                    r"
                INSERT INTO files (path, size, hash)
                VALUES ($1, $2, $3)
                ON CONFLICT(path) DO UPDATE SET
                    size = excluded.size,
                    hash = excluded.hash,
                    updated_at = strftime('%s', 'now')
                ",
                    &[&entry.path, &size, &entry.hash],
                )
                .await;

            if let Err(e) = result {
                let _ = tx.rollback().await;
                return Err(DedupError::Database(e));
            }

            count += 1;
        }

        tx.commit().await.map_err(DedupError::Database)?;
        Ok(count)
    }

    /// Begin a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if transaction cannot be started.
    pub async fn begin_transaction(&self) -> DedupResult<Box<dyn oxisql_core::Transaction + '_>> {
        self.pool.transaction().await.map_err(DedupError::Database)
    }

    /// Close the database.
    ///
    /// # Errors
    ///
    /// Never fails; `SqliteConnection` has no explicit close handle — the
    /// connection is released when the last clone is dropped. Kept as an
    /// `async fn` returning `Result` for API compatibility with callers that
    /// treat closing as fallible.
    pub async fn close(self) -> DedupResult<()> {
        drop(self.pool);
        Ok(())
    }
}

/// Database statistics.
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    /// Total number of indexed files
    pub total_files: usize,

    /// Number of unique hashes
    pub unique_hashes: usize,

    /// Total number of fingerprints
    pub total_fingerprints: usize,

    /// Total number of chunks
    pub total_chunks: usize,

    /// Total size of all files in bytes
    pub total_size: u64,
}

impl DatabaseStats {
    /// Calculate duplicate file count.
    #[must_use]
    pub fn duplicate_files(&self) -> usize {
        self.total_files.saturating_sub(self.unique_hashes)
    }

    /// Calculate deduplication ratio.
    #[must_use]
    pub fn dedup_ratio(&self) -> f64 {
        if self.total_files == 0 {
            return 0.0;
        }
        self.duplicate_files() as f64 / self.total_files as f64
    }

    /// Estimate potential storage savings.
    #[must_use]
    pub fn estimated_savings(&self) -> u64 {
        if self.total_files == 0 {
            return 0;
        }
        let avg_size = self.total_size / self.total_files as u64;
        avg_size * self.duplicate_files() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_database_creation() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");
        let stats = db.get_stats().await.expect("operation should succeed");
        assert_eq!(stats.total_files, 0);
    }

    #[tokio::test]
    async fn test_insert_file() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        // Create a temporary file
        let temp_file = std::env::temp_dir().join("test_file.txt");
        std::fs::write(&temp_file, b"test content").expect("operation should succeed");

        let file_id = db
            .insert_file(&temp_file, "abcd1234")
            .await
            .expect("operation should succeed");
        assert!(file_id > 0);

        let count = db.count_files().await.expect("operation should succeed");
        assert_eq!(count, 1);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[tokio::test]
    async fn test_duplicate_detection() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        let temp_dir = std::env::temp_dir();
        let file1 = temp_dir.join("test1.txt");
        let file2 = temp_dir.join("test2.txt");

        std::fs::write(&file1, b"test").expect("operation should succeed");
        std::fs::write(&file2, b"test").expect("operation should succeed");

        let hash = "same_hash";

        db.insert_file(&file1, hash)
            .await
            .expect("operation should succeed");
        db.insert_file(&file2, hash)
            .await
            .expect("operation should succeed");

        let duplicates = db
            .find_duplicate_hashes()
            .await
            .expect("operation should succeed");
        assert_eq!(duplicates.len(), 1);
        assert_eq!(
            duplicates
                .get(hash)
                .expect("operation should succeed")
                .len(),
            2
        );

        // Cleanup
        std::fs::remove_file(&file1).ok();
        std::fs::remove_file(&file2).ok();
    }

    #[tokio::test]
    async fn test_fingerprints() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        let temp_file = std::env::temp_dir().join("test_fp.txt");
        std::fs::write(&temp_file, b"test").expect("operation should succeed");

        let file_id = db
            .insert_file(&temp_file, "hash123")
            .await
            .expect("operation should succeed");
        let fp_id = db
            .insert_fingerprint(file_id, "phash", "abc123")
            .await
            .expect("operation should succeed");

        assert!(fp_id > 0);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[tokio::test]
    async fn test_chunks() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        let temp_file = std::env::temp_dir().join("test_chunk.txt");
        std::fs::write(&temp_file, b"test").expect("operation should succeed");

        let file_id = db
            .insert_file(&temp_file, "hash456")
            .await
            .expect("operation should succeed");
        let chunk_id = db
            .insert_chunk(file_id, 0, 100, "chunk_hash")
            .await
            .expect("operation should succeed");

        assert!(chunk_id > 0);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[tokio::test]
    async fn test_delete_file() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");

        let temp_file = std::env::temp_dir().join("test_delete.txt");
        std::fs::write(&temp_file, b"test").expect("operation should succeed");

        db.insert_file(&temp_file, "hash_del")
            .await
            .expect("operation should succeed");

        let count_before = db.count_files().await.expect("operation should succeed");
        assert_eq!(count_before, 1);

        db.delete_file(&temp_file)
            .await
            .expect("operation should succeed");

        let count_after = db.count_files().await.expect("operation should succeed");
        assert_eq!(count_after, 0);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[tokio::test]
    async fn test_stats() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("operation should succeed");
        let stats = db.get_stats().await.expect("operation should succeed");

        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.unique_hashes, 0);
        assert_eq!(stats.duplicate_files(), 0);
        assert_eq!(stats.dedup_ratio(), 0.0);
    }

    #[tokio::test]
    async fn test_batch_insert_100_entries() {
        let db = DedupDatabase::open_memory()
            .await
            .expect("failed to open in-memory DB");

        let temp_dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);

        // Create 100 temporary files and prepare batch entries
        let mut entries = Vec::with_capacity(100);
        let mut created_paths = Vec::with_capacity(100);
        for i in 0..100usize {
            let p = temp_dir.join(format!("dedup_batch_{nanos}_{i}.tmp"));
            std::fs::write(&p, format!("content_{i}")).expect("write temp file");
            entries.push(BatchFileEntry {
                path: p.to_string_lossy().to_string(),
                hash: format!("hash_{i:04x}"),
            });
            created_paths.push(p);
        }

        let inserted = db
            .insert_batch(&entries)
            .await
            .expect("batch insert should succeed");
        assert_eq!(inserted, 100, "insert_batch must return the inserted count");

        let count = db.count_files().await.expect("count_files should succeed");
        assert_eq!(count, 100, "database must contain exactly 100 files");

        // Cleanup
        for p in &created_paths {
            std::fs::remove_file(p).ok();
        }
    }

    #[tokio::test]
    async fn test_batch_insert_vs_individual_same_results() {
        let temp_dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(1);

        // Build 50 temp files shared between both DBs
        let mut created_paths = Vec::with_capacity(50);
        let mut entries = Vec::with_capacity(50);
        for i in 0..50usize {
            let p = temp_dir.join(format!("dedup_cmp_{nanos}_{i}.tmp"));
            std::fs::write(&p, format!("cmp_content_{i}")).expect("write temp file");
            entries.push(BatchFileEntry {
                path: p.to_string_lossy().to_string(),
                hash: format!("cmp_hash_{i:04x}"),
            });
            created_paths.push(p);
        }

        // DB A — batch insert
        let db_a = DedupDatabase::open_memory().await.expect("open DB A");
        let inserted = db_a.insert_batch(&entries).await.expect("batch insert A");
        assert_eq!(inserted, 50);

        // DB B — individual inserts
        let db_b = DedupDatabase::open_memory().await.expect("open DB B");
        for entry in &entries {
            db_b.insert_file(entry.path.as_str(), &entry.hash)
                .await
                .expect("individual insert B");
        }

        // Both must produce identical file count and hash distribution
        let count_a = db_a.count_files().await.expect("count A");
        let count_b = db_b.count_files().await.expect("count B");
        assert_eq!(count_a, count_b, "file counts must match");

        let files_a = db_a.get_all_files().await.expect("all files A");
        let files_b = db_b.get_all_files().await.expect("all files B");
        assert_eq!(files_a.len(), files_b.len());

        // Compare sorted path lists
        let mut paths_a: Vec<_> = files_a.iter().map(|(p, _)| p.clone()).collect();
        let mut paths_b: Vec<_> = files_b.iter().map(|(p, _)| p.clone()).collect();
        paths_a.sort();
        paths_b.sort();
        assert_eq!(paths_a, paths_b, "path sets must be identical");

        // Cleanup
        for p in &created_paths {
            std::fs::remove_file(p).ok();
        }
    }
}
