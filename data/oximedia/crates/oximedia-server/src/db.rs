//! Database layer — Pure-Rust SQLite via OxiSQL (`oxisql-sqlite-compat`).
//!
//! No C, C++, or Fortran is compiled anywhere in this storage layer
//! (COOLJAPAN Pure Rust Policy — this replaces the previous `sqlx` +
//! `libsqlite3-sys` backend). The on-disk format is standard SQLite, so
//! databases created here remain readable by any SQLite tooling. This
//! mirrors the same migration already applied to `oximedia-rights`
//! (`crates/oximedia-rights/src/database/storage.rs`).
//!
//! This module also provides a small compatibility shim ([`query`],
//! [`query_scalar`], [`Row`], [`SqlitePool`], [`AssertSqlSafe`], …) that
//! mirrors the subset of the `sqlx` 0.9 query-builder API previously used
//! throughout this crate, so the ~30 call sites spread across `admin.rs`,
//! `library.rs`, `upload.rs`, `batch_ops.rs`, and `api/*.rs` did not need to
//! be rewritten query-by-query — only their `sqlx::` path prefix changed to
//! `crate::db::`. Two differences from `sqlx` are worth calling out:
//!
//! - OxiSQL uses `$1, $2, …` positional parameters natively, not SQLite's
//!   `?`. `qmark_to_dollar` rewrites `?` placeholders (quote-aware) into
//!   `$N` form before each statement reaches the engine, so call sites keep
//!   writing SQLite-style `?` unchanged.
//! - [`Row::get`] cannot panic on a decode error the way `sqlx::Row::get`
//!   does (this crate's no-`unwrap`/`expect`-in-production policy forbids
//!   that), so a missing or mismatched column falls back to `T::default()`
//!   instead of panicking. Every column read here comes from a table this
//!   module itself creates in [`Database::migrate`], so in practice this
//!   fallback path is never exercised.

use crate::error::ServerResult;
use oxisql_core::{Connection as _, FromValue, OxiSqlError, ToSqlValue, Value};
use oxisql_sqlite_compat::SqliteConnection;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type produced by the database layer.
pub type Error = OxiSqlError;

// ---------------------------------------------------------------------------
// URL / placeholder helpers
// ---------------------------------------------------------------------------

/// Strips `sqlite://` / `sqlite:` URL prefixes so both plain filesystem paths
/// and connection-URL strings (e.g. `sqlite:oximedia.db`, `sqlite:///tmp/x.db`)
/// resolve to a path OxiSQL understands. An empty result (e.g. bare
/// `sqlite://`) maps to `:memory:`.
fn sqlite_url_to_path(path: &str) -> &str {
    let stripped = path
        .strip_prefix("sqlite://")
        .or_else(|| path.strip_prefix("sqlite:"))
        .unwrap_or(path);
    if stripped.is_empty() {
        ":memory:"
    } else {
        stripped
    }
}

/// Rewrites SQLite-style `?` positional placeholders into OxiSQL's `$1, $2, …`
/// positional syntax, in left-to-right order. Skips over single-quoted string
/// literals (with `''`-escape awareness) and double-quoted identifiers so a
/// literal `?` or quote character inside one is never mistaken for syntax.
fn qmark_to_dollar(sql: &str) -> String {
    let chars: Vec<char> = sql.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(n + 8);
    let mut i = 0;
    let mut count = 0usize;

    while i < n {
        match chars[i] {
            '\'' => {
                out.push('\'');
                i += 1;
                while i < n {
                    let c = chars[i];
                    out.push(c);
                    i += 1;
                    if c == '\'' {
                        if i < n && chars[i] == '\'' {
                            out.push('\'');
                            i += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
            '"' => {
                out.push('"');
                i += 1;
                while i < n && chars[i] != '"' {
                    out.push(chars[i]);
                    i += 1;
                }
                if i < n {
                    out.push('"');
                    i += 1;
                }
            }
            '?' => {
                count += 1;
                out.push('$');
                out.push_str(&count.to_string());
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// AssertSqlSafe — sqlx 0.9 `SqlSafeStr` escape-hatch compatibility marker
// ---------------------------------------------------------------------------

/// Compatibility marker for call sites written against sqlx 0.9's
/// `AssertSqlSafe` escape hatch (dynamic SQL built only from hardcoded
/// literal fragments, e.g. the admin filter/pagination queries in
/// `admin.rs`). This query builder has no compile-time "must be a
/// `&'static str` unless asserted safe" check to satisfy, so this type
/// exists purely to keep those call sites unchanged; it is a transparent
/// pass-through.
pub struct AssertSqlSafe<T>(pub T);

impl<T: Into<String>> From<AssertSqlSafe<T>> for String {
    fn from(value: AssertSqlSafe<T>) -> Self {
        value.0.into()
    }
}

// ---------------------------------------------------------------------------
// Row
// ---------------------------------------------------------------------------

/// A single result row, with named-column access.
#[derive(Debug, Clone)]
pub struct Row(oxisql_core::Row);

impl Row {
    /// Reads column `col`, decoded as `T`.
    ///
    /// Falls back to `T::default()` if the column is missing or fails to
    /// decode as `T` (see the module-level docs for why this does not
    /// panic).
    #[must_use]
    pub fn get<T: FromValue + Default>(&self, col: &str) -> T {
        self.0.try_get(col).unwrap_or_default()
    }

    /// Reads the first column of the row, decoded as `T`. Used by
    /// [`QueryScalar`] for single-column aggregate queries (`COUNT`, `SUM`,
    /// …) whose result column has no stable name to look up by.
    fn get_scalar<T: FromValue + Default>(&self) -> T {
        self.0.try_get_by_index(0).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// QueryResult
// ---------------------------------------------------------------------------

/// Outcome of an [`Query::execute`] call.
pub struct QueryResult {
    rows_affected: u64,
}

impl QueryResult {
    /// Number of rows affected by the statement.
    #[must_use]
    pub fn rows_affected(&self) -> u64 {
        self.rows_affected
    }
}

// ---------------------------------------------------------------------------
// SqlitePool
// ---------------------------------------------------------------------------

/// Database connection handle.
///
/// OxiSQL connections are internally reference-counted and safe to share
/// across async tasks (see [`SqliteConnection`]), so this handle plays the
/// role a connection pool used to: clone it freely and issue queries
/// concurrently.
#[derive(Clone)]
pub struct SqlitePool {
    conn: SqliteConnection,
}

impl SqlitePool {
    /// Opens (creating if missing) the SQLite database at `url`.
    ///
    /// Accepts plain filesystem paths as well as `sqlite:`/`sqlite://`
    /// connection-URL strings; pass `":memory:"` (or an empty path) for an
    /// in-memory database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn connect(url: &str) -> Result<Self, Error> {
        let conn = SqliteConnection::open(sqlite_url_to_path(url)).await?;
        Ok(Self { conn })
    }

    async fn execute_owned(&self, sql: String, params: Vec<Value>) -> Result<QueryResult, Error> {
        let refs: Vec<&dyn ToSqlValue> = params.iter().map(|v| v as &dyn ToSqlValue).collect();
        let rows_affected = self.conn.execute(&sql, &refs).await?;
        Ok(QueryResult { rows_affected })
    }

    async fn query_owned(&self, sql: String, params: Vec<Value>) -> Result<Vec<Row>, Error> {
        let refs: Vec<&dyn ToSqlValue> = params.iter().map(|v| v as &dyn ToSqlValue).collect();
        let rows = self.conn.query(&sql, &refs).await?;
        Ok(rows.into_iter().map(Row).collect())
    }
}

// ---------------------------------------------------------------------------
// Query — mirrors `sqlx::query(...).bind(...).execute()/fetch_*()`
// ---------------------------------------------------------------------------

/// A SQL statement builder, mirroring the subset of `sqlx::query`'s builder
/// API used throughout this crate.
pub struct Query {
    sql: String,
    params: Vec<Value>,
}

impl Query {
    fn new(sql: String) -> Self {
        Self {
            sql,
            params: Vec::new(),
        }
    }

    /// Binds the next `?` positional parameter, in left-to-right order.
    #[must_use]
    pub fn bind<T: ToSqlValue>(mut self, value: T) -> Self {
        self.params.push(value.to_value());
        self
    }

    /// Executes a DML/DDL statement, returning the number of rows affected.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement fails.
    pub async fn execute(self, pool: &SqlitePool) -> Result<QueryResult, Error> {
        pool.execute_owned(qmark_to_dollar(&self.sql), self.params)
            .await
    }

    /// Executes a `SELECT` expected to return exactly one row.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement fails or no row is returned.
    pub async fn fetch_one(self, pool: &SqlitePool) -> Result<Row, Error> {
        let rows = pool
            .query_owned(qmark_to_dollar(&self.sql), self.params)
            .await?;
        rows.into_iter()
            .next()
            .ok_or_else(|| OxiSqlError::Execution("query returned no rows".to_string()))
    }

    /// Executes a `SELECT` expected to return at most one row.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement fails.
    pub async fn fetch_optional(self, pool: &SqlitePool) -> Result<Option<Row>, Error> {
        let rows = pool
            .query_owned(qmark_to_dollar(&self.sql), self.params)
            .await?;
        Ok(rows.into_iter().next())
    }

    /// Executes a `SELECT`, returning all result rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement fails.
    pub async fn fetch_all(self, pool: &SqlitePool) -> Result<Vec<Row>, Error> {
        pool.query_owned(qmark_to_dollar(&self.sql), self.params)
            .await
    }
}

/// Starts building a SQL statement. See [`Query`].
pub fn query<S: Into<String>>(sql: S) -> Query {
    Query::new(sql.into())
}

// ---------------------------------------------------------------------------
// QueryScalar — mirrors `sqlx::query_scalar`
// ---------------------------------------------------------------------------

/// A single-column `SELECT` builder, mirroring `sqlx::query_scalar`.
pub struct QueryScalar<O> {
    inner: Query,
    _marker: std::marker::PhantomData<fn() -> O>,
}

impl<O> QueryScalar<O> {
    /// Binds the next `?` positional parameter, in left-to-right order.
    #[must_use]
    pub fn bind<T: ToSqlValue>(mut self, value: T) -> Self {
        self.inner = self.inner.bind(value);
        self
    }
}

impl<O: FromValue + Default> QueryScalar<O> {
    /// Executes the query, decoding the single result column as `O`.
    ///
    /// # Errors
    ///
    /// Returns an error if the statement fails or no row is returned.
    pub async fn fetch_one(self, pool: &SqlitePool) -> Result<O, Error> {
        let row = self.inner.fetch_one(pool).await?;
        Ok(row.get_scalar::<O>())
    }
}

/// Starts building a single-column `SELECT`. See [`QueryScalar`].
pub fn query_scalar<O, S: Into<String>>(sql: S) -> QueryScalar<O> {
    QueryScalar {
        inner: Query::new(sql.into()),
        _marker: std::marker::PhantomData,
    }
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// Database connection pool.
#[derive(Clone)]
pub struct Database {
    pool: SqlitePool,
}

impl Database {
    /// Creates a new database connection pool.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection fails.
    pub async fn new(url: &str) -> ServerResult<Self> {
        let pool = SqlitePool::connect(url).await?;
        Ok(Self { pool })
    }

    /// Returns a reference to the connection pool.
    #[must_use]
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Runs database migrations.
    ///
    /// # Errors
    ///
    /// Returns an error if migrations fail.
    pub async fn migrate(&self) -> ServerResult<()> {
        // Users table
        query(
            r"
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_login INTEGER
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // API keys table
        query(
            r"
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                expires_at INTEGER,
                last_used INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Media table
        query(
            r"
            CREATE TABLE IF NOT EXISTS media (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                duration REAL,
                width INTEGER,
                height INTEGER,
                codec_video TEXT,
                codec_audio TEXT,
                bitrate INTEGER,
                framerate REAL,
                sample_rate INTEGER,
                channels INTEGER,
                thumbnail_path TEXT,
                sprite_path TEXT,
                preview_path TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Media metadata table (flexible key-value store)
        query(
            r"
            CREATE TABLE IF NOT EXISTS media_metadata (
                media_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (media_id, key),
                FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE CASCADE
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Transcoding jobs table
        query(
            r"
            CREATE TABLE IF NOT EXISTS transcode_jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                media_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                progress REAL NOT NULL DEFAULT 0.0,
                output_format TEXT NOT NULL,
                output_codec_video TEXT,
                output_codec_audio TEXT,
                output_width INTEGER,
                output_height INTEGER,
                output_bitrate INTEGER,
                output_path TEXT,
                error_message TEXT,
                created_at INTEGER NOT NULL,
                started_at INTEGER,
                completed_at INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE CASCADE
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Collections table
        query(
            r"
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                thumbnail_path TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Collection items table
        query(
            r"
            CREATE TABLE IF NOT EXISTS collection_items (
                collection_id TEXT NOT NULL,
                media_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                added_at INTEGER NOT NULL,
                PRIMARY KEY (collection_id, media_id),
                FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE CASCADE
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Multipart uploads table
        query(
            r"
            CREATE TABLE IF NOT EXISTS multipart_uploads (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                total_size INTEGER NOT NULL,
                uploaded_size INTEGER NOT NULL DEFAULT 0,
                chunk_size INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                completed_chunks INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'uploading',
                created_at INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Upload chunks table
        query(
            r"
            CREATE TABLE IF NOT EXISTS upload_chunks (
                upload_id TEXT NOT NULL,
                chunk_number INTEGER NOT NULL,
                chunk_path TEXT NOT NULL,
                chunk_size INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                uploaded_at INTEGER NOT NULL,
                PRIMARY KEY (upload_id, chunk_number),
                FOREIGN KEY (upload_id) REFERENCES multipart_uploads(id) ON DELETE CASCADE
            )
            ",
        )
        .execute(&self.pool)
        .await?;

        // Create indexes
        query("CREATE INDEX IF NOT EXISTS idx_media_user_id ON media(user_id)")
            .execute(&self.pool)
            .await?;
        query("CREATE INDEX IF NOT EXISTS idx_media_status ON media(status)")
            .execute(&self.pool)
            .await?;
        query("CREATE INDEX IF NOT EXISTS idx_media_created_at ON media(created_at)")
            .execute(&self.pool)
            .await?;
        query("CREATE INDEX IF NOT EXISTS idx_transcode_jobs_user_id ON transcode_jobs(user_id)")
            .execute(&self.pool)
            .await?;
        query("CREATE INDEX IF NOT EXISTS idx_transcode_jobs_media_id ON transcode_jobs(media_id)")
            .execute(&self.pool)
            .await?;
        query("CREATE INDEX IF NOT EXISTS idx_transcode_jobs_status ON transcode_jobs(status)")
            .execute(&self.pool)
            .await?;
        query("CREATE INDEX IF NOT EXISTS idx_collections_user_id ON collections(user_id)")
            .execute(&self.pool)
            .await?;
        query("CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id)")
            .execute(&self.pool)
            .await?;

        // NOTE: The previous sqlx-backed schema also created a
        // `media_fts USING fts5(...)` virtual table here. It is intentionally
        // dropped in this Pure-Rust migration: nothing in this crate ever
        // queries it (grep confirms zero references to `media_fts` outside
        // this historical `CREATE VIRTUAL TABLE`), and the embedded OxiSQL
        // engine is not guaranteed to ship the SQLite `fts5` extension. If
        // full-text search on media filenames/metadata is needed later,
        // implement it against a real query path (not dead schema) and
        // re-verify `fts5` support against the pinned OxiSQL version first.

        Ok(())
    }

    /// Checks if the database is healthy.
    ///
    /// # Errors
    ///
    /// Returns an error if the health check fails.
    pub async fn health_check(&self) -> ServerResult<bool> {
        let result: i64 = query_scalar("SELECT 1").fetch_one(&self.pool).await?;
        Ok(result == 1)
    }

    /// Gets storage statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_storage_stats(&self) -> ServerResult<StorageStats> {
        let row = query(
            r"
            SELECT
                COUNT(*) as total_files,
                SUM(file_size) as total_size,
                AVG(file_size) as avg_size,
                MAX(file_size) as max_size,
                MIN(file_size) as min_size
            FROM media
            WHERE status = 'ready'
            ",
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(StorageStats {
            total_files: row.get::<i64>("total_files") as u64,
            total_size: row.get::<Option<i64>>("total_size").unwrap_or(0) as u64,
            avg_size: row.get::<Option<f64>>("avg_size").unwrap_or(0.0),
            max_size: row.get::<Option<i64>>("max_size").unwrap_or(0) as u64,
            min_size: row.get::<Option<i64>>("min_size").unwrap_or(0) as u64,
        })
    }
}

/// Storage statistics.
#[derive(Debug, Clone)]
pub struct StorageStats {
    /// Total number of files
    pub total_files: u64,
    /// Total storage size in bytes
    pub total_size: u64,
    /// Average file size in bytes
    pub avg_size: f64,
    /// Largest file size in bytes
    pub max_size: u64,
    /// Smallest file size in bytes
    pub min_size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── qmark_to_dollar ──────────────────────────────────────────────────────

    #[test]
    fn test_qmark_to_dollar_basic() {
        assert_eq!(
            qmark_to_dollar("SELECT * FROM t WHERE id = ?"),
            "SELECT * FROM t WHERE id = $1"
        );
    }

    #[test]
    fn test_qmark_to_dollar_multiple() {
        assert_eq!(
            qmark_to_dollar("INSERT INTO t (a, b, c) VALUES (?, ?, ?)"),
            "INSERT INTO t (a, b, c) VALUES ($1, $2, $3)"
        );
    }

    #[test]
    fn test_qmark_to_dollar_no_placeholders() {
        assert_eq!(qmark_to_dollar("VACUUM"), "VACUUM");
    }

    #[test]
    fn test_qmark_to_dollar_skips_single_quoted_literal() {
        // A literal `?` inside a string must not be counted or rewritten.
        assert_eq!(
            qmark_to_dollar("SELECT * FROM t WHERE q = 'literal?' AND id = ?"),
            "SELECT * FROM t WHERE q = 'literal?' AND id = $1"
        );
    }

    #[test]
    fn test_qmark_to_dollar_handles_escaped_quote() {
        assert_eq!(
            qmark_to_dollar("SELECT * FROM t WHERE q = 'it''s' AND id = ?"),
            "SELECT * FROM t WHERE q = 'it''s' AND id = $1"
        );
    }

    #[test]
    fn test_qmark_to_dollar_skips_double_quoted_identifier() {
        assert_eq!(
            qmark_to_dollar(r#"SELECT "col?name" FROM t WHERE id = ?"#),
            r#"SELECT "col?name" FROM t WHERE id = $1"#
        );
    }

    // ── sqlite_url_to_path ───────────────────────────────────────────────────

    #[test]
    fn test_sqlite_url_to_path_scheme_prefix() {
        assert_eq!(sqlite_url_to_path("sqlite:oximedia.db"), "oximedia.db");
    }

    #[test]
    fn test_sqlite_url_to_path_double_slash_prefix() {
        assert_eq!(sqlite_url_to_path("sqlite:///tmp/x.db"), "/tmp/x.db");
    }

    #[test]
    fn test_sqlite_url_to_path_bare_path() {
        assert_eq!(sqlite_url_to_path("test.db"), "test.db");
    }

    #[test]
    fn test_sqlite_url_to_path_memory() {
        assert_eq!(sqlite_url_to_path("sqlite::memory:"), ":memory:");
        assert_eq!(sqlite_url_to_path("sqlite://"), ":memory:");
    }

    // ── AssertSqlSafe ────────────────────────────────────────────────────────

    #[test]
    fn test_assert_sql_safe_passthrough() {
        let sql: String = AssertSqlSafe("SELECT 1".to_string()).into();
        assert_eq!(sql, "SELECT 1");
    }

    // ── Database integration (in-memory) ────────────────────────────────────

    #[tokio::test]
    async fn test_database_migrate_and_health_check() {
        let db = Database::new(":memory:").await.expect("open in-memory db");
        db.migrate().await.expect("migrate");
        let healthy = db.health_check().await.expect("health check");
        assert!(healthy);
    }

    #[tokio::test]
    async fn test_database_get_storage_stats_empty() {
        let db = Database::new(":memory:").await.expect("open in-memory db");
        db.migrate().await.expect("migrate");
        let stats = db.get_storage_stats().await.expect("storage stats");
        assert_eq!(stats.total_files, 0);
        assert_eq!(stats.total_size, 0);
    }

    #[tokio::test]
    async fn test_query_builder_roundtrip() {
        let db = Database::new(":memory:").await.expect("open in-memory db");
        db.migrate().await.expect("migrate");

        query("INSERT INTO users (id, username, email, password_hash, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)")
            .bind("u1")
            .bind("alice")
            .bind("alice@example.com")
            .bind("hash")
            .bind("admin")
            .bind(0i64)
            .bind(0i64)
            .execute(db.pool())
            .await
            .expect("insert user");

        let row = query("SELECT id, username FROM users WHERE id = ?")
            .bind("u1")
            .fetch_one(db.pool())
            .await
            .expect("fetch user");
        assert_eq!(row.get::<String>("username"), "alice");

        let count: i64 = query_scalar("SELECT COUNT(*) FROM users")
            .fetch_one(db.pool())
            .await
            .expect("count users");
        assert_eq!(count, 1);
    }
}
