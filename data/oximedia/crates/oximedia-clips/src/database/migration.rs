//! Database schema migration utilities.
//!
//! Runs against the Pure-Rust OxiSQL engine (`oxisql-sqlite-compat`); no C
//! `libsqlite3-sys` is involved.

use crate::error::ClipResult;
use oxisql_core::Connection;
use oxisql_sqlite_compat::SqliteConnection;

/// Migrates the database to the latest schema version.
///
/// # Errors
///
/// Returns an error if the migration fails.
pub async fn migrate_database(conn: &SqliteConnection) -> ClipResult<()> {
    // Create version table
    conn.execute(
        r"
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        ",
        &[],
    )
    .await?;

    // Get current version
    let current_version = get_current_version(conn).await?;

    // Apply migrations
    if current_version < 1 {
        apply_migration_v1(conn).await?;
    }

    if current_version < 2 {
        apply_migration_v2(conn).await?;
    }

    if current_version < 3 {
        apply_migration_v3(conn).await?;
    }

    Ok(())
}

async fn get_current_version(conn: &SqliteConnection) -> ClipResult<i64> {
    let rows = conn
        .query("SELECT MAX(version) FROM schema_version", &[])
        .await?;

    // MAX(...) over an empty table yields a single NULL row.
    Ok(rows
        .first()
        .map(|row| row.try_get_by_index::<Option<i64>>(0))
        .transpose()?
        .flatten()
        .unwrap_or(0))
}

/// Records the completion of migration *version* with the current timestamp.
///
/// The timestamp is bound as an RFC 3339 string from the host clock instead
/// of relying on the SQL `datetime('now')` function, keeping the statement
/// engine-agnostic.
async fn record_migration(conn: &SqliteConnection, version: i64) -> ClipResult<()> {
    let applied_at = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES ($1, $2)",
        &[&version, &applied_at],
    )
    .await?;
    Ok(())
}

async fn apply_migration_v1(conn: &SqliteConnection) -> ClipResult<()> {
    // Create the core clips table.  This must happen here (and not rely on
    // storage.rs) so that the migration path is self-contained when invoked
    // directly (e.g. in tests using an in-memory database).
    conn.execute(
        r"
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            duration INTEGER,
            frame_rate_num INTEGER,
            frame_rate_den INTEGER,
            in_point INTEGER,
            out_point INTEGER,
            rating INTEGER NOT NULL DEFAULT 0,
            is_favorite INTEGER NOT NULL DEFAULT 0,
            is_rejected INTEGER NOT NULL DEFAULT 0,
            keywords TEXT,
            created_at TEXT NOT NULL,
            modified_at TEXT NOT NULL,
            custom_metadata TEXT
        )
        ",
        &[],
    )
    .await?;

    // Record migration
    record_migration(conn, 1).await
}

async fn apply_migration_v2(conn: &SqliteConnection) -> ClipResult<()> {
    // Create bins table
    conn.execute(
        r"
        CREATE TABLE IF NOT EXISTS bins (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            color TEXT,
            created_at TEXT NOT NULL,
            modified_at TEXT NOT NULL
        )
        ",
        &[],
    )
    .await?;

    // Create clip_bins junction table
    conn.execute(
        r"
        CREATE TABLE IF NOT EXISTS clip_bins (
            clip_id TEXT NOT NULL,
            bin_id TEXT NOT NULL,
            PRIMARY KEY (clip_id, bin_id),
            FOREIGN KEY (clip_id) REFERENCES clips(id) ON DELETE CASCADE,
            FOREIGN KEY (bin_id) REFERENCES bins(id) ON DELETE CASCADE
        )
        ",
        &[],
    )
    .await?;

    // Record migration
    record_migration(conn, 2).await
}

/// Migration v3: Add indexes on `keywords` and `rating` columns for faster
/// filtered queries.
async fn apply_migration_v3(conn: &SqliteConnection) -> ClipResult<()> {
    // Index on the `rating` column – used heavily by rating-based filters.
    conn.execute(
        r"
        CREATE INDEX IF NOT EXISTS idx_clips_rating
            ON clips (rating)
        ",
        &[],
    )
    .await?;

    // Index on the `keywords` column – SQLite FTS5 would be ideal, but a
    // plain B-tree index on the JSON string still speeds up `LIKE '%keyword%'`
    // scans for moderate-sized libraries.
    conn.execute(
        r"
        CREATE INDEX IF NOT EXISTS idx_clips_keywords
            ON clips (keywords)
        ",
        &[],
    )
    .await?;

    // Composite index on (is_favorite, rating) for common "show favorites
    // above a threshold" queries.
    conn.execute(
        r"
        CREATE INDEX IF NOT EXISTS idx_clips_favorite_rating
            ON clips (is_favorite, rating)
        ",
        &[],
    )
    .await?;

    // Record migration
    record_migration(conn, 3).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_migration() {
        let conn = SqliteConnection::open_memory()
            .await
            .expect("open_memory should succeed");

        migrate_database(&conn)
            .await
            .expect("operation should succeed");

        let version = get_current_version(&conn)
            .await
            .expect("get_current_version should succeed");
        assert!(version >= 1);
    }

    #[tokio::test]
    async fn test_migration_is_idempotent() {
        let conn = SqliteConnection::open_memory()
            .await
            .expect("open_memory should succeed");

        migrate_database(&conn)
            .await
            .expect("first migration should succeed");
        migrate_database(&conn)
            .await
            .expect("second migration should succeed");

        let version = get_current_version(&conn)
            .await
            .expect("get_current_version should succeed");
        assert_eq!(version, 3, "re-running migrations must not add versions");
    }
}
