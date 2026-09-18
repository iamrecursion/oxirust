//! Rights database storage implementation.
//!
//! Backed by Pure-Rust SQLite via OxiSQL (`oxisql-sqlite-compat`) — no C, C++,
//! or Fortran is compiled anywhere in this storage layer (COOLJAPAN Pure Rust
//! Policy).  The on-disk format is standard SQLite, so databases created here
//! remain readable by any SQLite tooling.

use crate::{Result, RightsError};
use oxisql_core::{Connection as _, Row, ToSqlValue};
use oxisql_sqlite_compat::SqliteConnection;

/// Cloneable handle to the rights SQLite database.
///
/// OxiSQL connections are internally reference-counted and safe to share
/// across async tasks, so this handle plays the role the connection pool used
/// to: clone it freely and issue queries concurrently.  Statements are cached
/// per connection, giving pool-like prepared-statement reuse.
#[derive(Clone)]
pub struct RightsPool {
    conn: SqliteConnection,
}

impl std::fmt::Debug for RightsPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RightsPool")
            .field("path", &self.conn.path())
            .finish()
    }
}

impl RightsPool {
    /// Open (creating if missing) the SQLite database at `path`.
    ///
    /// Accepts plain filesystem paths as well as `sqlite://…` / `sqlite:…`
    /// connection URLs for drop-in compatibility with the previous pool-based
    /// API.  Pass `":memory:"` (or an empty path) for an in-memory database.
    pub async fn open(path: &str) -> Result<Self> {
        let normalized = normalize_sqlite_path(path);
        let conn = SqliteConnection::open(normalized).await?;
        Ok(Self { conn })
    }

    /// Execute a DML/DDL statement, returning the number of rows affected.
    ///
    /// Positional parameters use `$1`, `$2`, … numbering.
    pub async fn execute(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<u64> {
        Ok(self.conn.execute(sql, params).await?)
    }

    /// Run a `SELECT` and return all result rows.
    pub async fn query(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<Vec<Row>> {
        Ok(self.conn.query(sql, params).await?)
    }

    /// Run a `SELECT` expected to return at most one row.
    pub async fn query_optional(
        &self,
        sql: &str,
        params: &[&dyn ToSqlValue],
    ) -> Result<Option<Row>> {
        let mut rows = self.conn.query(sql, params).await?;
        if rows.len() > 1 {
            return Err(RightsError::Database(format!(
                "expected at most one row, got {}",
                rows.len()
            )));
        }
        Ok(rows.pop())
    }

    /// Run a `SELECT` expected to return exactly one row.
    pub async fn query_one(&self, sql: &str, params: &[&dyn ToSqlValue]) -> Result<Row> {
        self.query_optional(sql, params)
            .await?
            .ok_or_else(|| RightsError::Database("expected one row, got none".to_string()))
    }
}

/// Strip `sqlite://` / `sqlite:` URL prefixes so both plain paths and
/// pool-style connection URLs keep working after the OxiSQL migration.
fn normalize_sqlite_path(path: &str) -> &str {
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

/// Rights database using Pure-Rust SQLite (OxiSQL).
pub struct RightsDatabase {
    pool: RightsPool,
}

impl RightsDatabase {
    /// Create a new rights database with default settings.
    pub async fn new(path: &str) -> Result<Self> {
        Self::new_with_pool(path, 5, 5).await
    }

    /// Create a new rights database with explicit pool configuration.
    ///
    /// # Parameters
    /// - `path` — SQLite path or connection URL (e.g. `sqlite:///tmp/rights.db`)
    /// - `_max_connections` — retained for API compatibility; OxiSQL shares a
    ///   single clone-safe connection handle with an internal statement cache,
    ///   so no fixed-size pool is required
    /// - `_connect_timeout_secs` — retained for API compatibility
    pub async fn new_with_pool(
        path: &str,
        _max_connections: u32,
        _connect_timeout_secs: u64,
    ) -> Result<Self> {
        let pool = RightsPool::open(path).await?;
        let db = Self { pool };
        db.initialize_schema().await?;
        Ok(db)
    }

    /// Get a reference to the database handle ("pool").
    pub fn pool(&self) -> &RightsPool {
        &self.pool
    }

    /// Initialize the database schema
    async fn initialize_schema(&self) -> Result<()> {
        // Rights owners table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS rights_owners (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                contact_info TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ",
                &[],
            )
            .await?;

        // Assets table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            ",
                &[],
            )
            .await?;

        // Rights grants table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS rights_grants (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                license_type TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                is_exclusive INTEGER NOT NULL DEFAULT 0,
                territory_json TEXT,
                usage_restrictions_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id),
                FOREIGN KEY (owner_id) REFERENCES rights_owners(id)
            )
            ",
                &[],
            )
            .await?;

        // License agreements table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS license_agreements (
                id TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL,
                agreement_number TEXT NOT NULL,
                terms_json TEXT NOT NULL,
                status TEXT NOT NULL,
                signed_date TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (grant_id) REFERENCES rights_grants(id)
            )
            ",
                &[],
            )
            .await?;

        // Usage logs table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS usage_logs (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                grant_id TEXT,
                usage_type TEXT NOT NULL,
                usage_date TEXT NOT NULL,
                territory TEXT,
                platform TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id),
                FOREIGN KEY (grant_id) REFERENCES rights_grants(id)
            )
            ",
                &[],
            )
            .await?;

        // Clearances table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS clearances (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                clearance_type TEXT NOT NULL,
                status TEXT NOT NULL,
                requester TEXT,
                approver TEXT,
                requested_date TEXT NOT NULL,
                approved_date TEXT,
                expiry_date TEXT,
                notes TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
            ",
                &[],
            )
            .await?;

        // Royalty payments table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS royalty_payments (
                id TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                amount REAL NOT NULL,
                currency TEXT NOT NULL,
                payment_period_start TEXT NOT NULL,
                payment_period_end TEXT NOT NULL,
                status TEXT NOT NULL,
                payment_date TEXT,
                calculation_data_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (grant_id) REFERENCES rights_grants(id),
                FOREIGN KEY (owner_id) REFERENCES rights_owners(id)
            )
            ",
                &[],
            )
            .await?;

        // Audit trail table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS audit_trail (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action TEXT NOT NULL,
                user_id TEXT,
                changes_json TEXT,
                timestamp TEXT NOT NULL,
                ip_address TEXT
            )
            ",
                &[],
            )
            .await?;

        // Expiration alerts table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS expiration_alerts (
                id TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                alert_date TEXT NOT NULL,
                notification_sent INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (grant_id) REFERENCES rights_grants(id)
            )
            ",
                &[],
            )
            .await?;

        // Watermark configurations table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS watermark_configs (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                watermark_type TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
            ",
                &[],
            )
            .await?;

        // DRM metadata table
        self.pool
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS drm_metadata (
                id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                drm_type TEXT NOT NULL,
                encryption_key_id TEXT,
                content_id TEXT,
                license_url TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (asset_id) REFERENCES assets(id)
            )
            ",
                &[],
            )
            .await?;

        // Create indices for better query performance
        self.pool
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_rights_grants_asset ON rights_grants(asset_id)",
                &[],
            )
            .await?;

        self.pool
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_rights_grants_owner ON rights_grants(owner_id)",
                &[],
            )
            .await?;

        self.pool
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_logs_asset ON usage_logs(asset_id)",
                &[],
            )
            .await?;

        self.pool
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_clearances_asset ON clearances(asset_id)",
                &[],
            )
            .await?;

        self.pool
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_trail_entity ON audit_trail(entity_type, entity_id)",
                &[],
            )
            .await?;

        // Performance: index on territory_json (text prefix) and end_date for
        // faster filtered queries in rights_grants.
        self.pool
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_rights_grants_territory ON rights_grants(territory_json)",
                &[],
            )
            .await?;

        self.pool
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_rights_grants_end_date ON rights_grants(end_date)",
                &[],
            )
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_sqlite_path() {
        assert_eq!(normalize_sqlite_path("/tmp/a.db"), "/tmp/a.db");
        assert_eq!(normalize_sqlite_path("sqlite:///tmp/a.db"), "/tmp/a.db");
        assert_eq!(normalize_sqlite_path("sqlite:/tmp/a.db"), "/tmp/a.db");
        assert_eq!(normalize_sqlite_path(":memory:"), ":memory:");
        assert_eq!(normalize_sqlite_path("sqlite://"), ":memory:");
    }

    #[tokio::test]
    async fn test_database_creation() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path).await;
        assert!(db.is_ok());
    }

    #[tokio::test]
    async fn test_schema_initialization() {
        let temp_dir = tempfile::tempdir().expect("rights test operation should succeed");
        let db_path = format!("sqlite://{}/test.db", temp_dir.path().display());
        let db = RightsDatabase::new(&db_path)
            .await
            .expect("rights test operation should succeed");

        // Verify tables exist by querying sqlite_master
        let result = db
            .pool()
            .query_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='rights_owners'",
                &[],
            )
            .await;
        assert!(result.is_ok());
    }
}
