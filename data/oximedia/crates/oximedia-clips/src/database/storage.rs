//! `SQLite` database storage for clips.
//!
//! Backed by the Pure-Rust OxiSQL engine (`oxisql-sqlite-compat`), so the
//! default build compiles zero C code (no `libsqlite3-sys`).

use crate::clip::{Clip, ClipId};
use crate::error::{ClipError, ClipResult};
use oxisql_core::{Connection, Row, ToSqlValue};
use oxisql_sqlite_compat::SqliteConnection;

/// Upsert statement shared by [`ClipDatabase::save_clip`] and
/// [`ClipDatabase::batch_save_clips`].
///
/// `INSERT OR REPLACE` is used instead of `ON CONFLICT ... DO UPDATE` for
/// engine compatibility. Semantics are equivalent here because every column
/// (including `created_at`, which travels with the [`Clip`] struct) is
/// supplied on each save.
const UPSERT_CLIP_SQL: &str = "
    INSERT OR REPLACE INTO clips (
        id, file_path, name, description, duration,
        frame_rate_num, frame_rate_den, in_point, out_point,
        rating, is_favorite, is_rejected, keywords,
        created_at, modified_at, custom_metadata
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
";

/// Owned SQL parameter values derived from a [`Clip`].
///
/// The borrowed fields of the clip (`name`, `description`, `duration`, …) are
/// referenced directly in [`Self::as_params`]; only values that require
/// conversion (JSON, RFC 3339 timestamps, integer flags) are materialized
/// here.
struct ClipSqlParams {
    id: String,
    file_path: String,
    frame_rate_num: Option<i64>,
    frame_rate_den: Option<i64>,
    rating: i64,
    is_favorite: i64,
    is_rejected: i64,
    keywords_json: String,
    created_at: String,
    modified_at: String,
}

impl ClipSqlParams {
    fn from_clip(clip: &Clip) -> ClipResult<Self> {
        let keywords_json = serde_json::to_string(&clip.keywords)
            .map_err(|e| ClipError::Serialization(e.to_string()))?;

        let (frame_rate_num, frame_rate_den) = clip
            .frame_rate
            .map_or((None, None), |fr| (Some(fr.num), Some(fr.den)));

        Ok(Self {
            id: clip.id.to_string(),
            file_path: clip.file_path.to_string_lossy().to_string(),
            frame_rate_num,
            frame_rate_den,
            rating: i64::from(clip.rating.to_value()),
            is_favorite: i64::from(clip.is_favorite),
            is_rejected: i64::from(clip.is_rejected),
            keywords_json,
            created_at: clip.created_at.to_rfc3339(),
            modified_at: clip.modified_at.to_rfc3339(),
        })
    }

    /// Positional parameters matching [`UPSERT_CLIP_SQL`] (`$1`..`$16`).
    fn as_params<'a>(&'a self, clip: &'a Clip) -> [&'a dyn ToSqlValue; 16] {
        [
            &self.id,
            &self.file_path,
            &clip.name,
            &clip.description,
            &clip.duration,
            &self.frame_rate_num,
            &self.frame_rate_den,
            &clip.in_point,
            &clip.out_point,
            &self.rating,
            &self.is_favorite,
            &self.is_rejected,
            &self.keywords_json,
            &self.created_at,
            &self.modified_at,
            &clip.custom_metadata,
        ]
    }
}

/// Normalizes a sqlx-style database URL to a plain path accepted by the
/// OxiSQL engine.
///
/// Accepted forms: `":memory:"`, `"sqlite::memory:"`, `"sqlite://path"`,
/// `"sqlite:path"`, and plain filesystem paths. Missing database files are
/// created automatically by the engine.
fn normalize_database_url(url: &str) -> &str {
    let stripped = url
        .strip_prefix("sqlite://")
        .or_else(|| url.strip_prefix("sqlite:"))
        .unwrap_or(url);
    if stripped.is_empty() {
        ":memory:"
    } else {
        stripped
    }
}

/// `SQLite` database for clip storage.
#[derive(Debug, Clone)]
pub struct ClipDatabase {
    conn: SqliteConnection,
}

impl ClipDatabase {
    /// Creates a new clip database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or migrated.
    pub async fn new(database_url: impl AsRef<str>) -> ClipResult<Self> {
        let path = normalize_database_url(database_url.as_ref());
        let conn = SqliteConnection::open(path).await?;

        let db = Self { conn };
        db.migrate().await?;

        Ok(db)
    }

    async fn migrate(&self) -> ClipResult<()> {
        self.conn
            .execute(
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

        Ok(())
    }

    /// Saves a clip to the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be saved.
    pub async fn save_clip(&self, clip: &Clip) -> ClipResult<()> {
        let params = ClipSqlParams::from_clip(clip)?;
        self.conn
            .execute(UPSERT_CLIP_SQL, &params.as_params(clip))
            .await?;

        Ok(())
    }

    /// Gets a clip by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip is not found or cannot be loaded.
    pub async fn get_clip(&self, clip_id: &ClipId) -> ClipResult<Clip> {
        let id = clip_id.to_string();
        let rows = self
            .conn
            .query("SELECT * FROM clips WHERE id = $1", &[&id])
            .await?;

        let row = rows
            .first()
            .ok_or_else(|| ClipError::ClipNotFound(clip_id.to_string()))?;

        Self::row_to_clip(row)
    }

    /// Gets all clips.
    ///
    /// # Errors
    ///
    /// Returns an error if clips cannot be loaded.
    pub async fn get_all_clips(&self) -> ClipResult<Vec<Clip>> {
        let rows = self
            .conn
            .query("SELECT * FROM clips ORDER BY created_at DESC", &[])
            .await?;

        rows.iter().map(Self::row_to_clip).collect()
    }

    /// Deletes a clip by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the clip cannot be deleted.
    pub async fn delete_clip(&self, clip_id: &ClipId) -> ClipResult<()> {
        let id = clip_id.to_string();
        self.conn
            .execute("DELETE FROM clips WHERE id = $1", &[&id])
            .await?;

        Ok(())
    }

    /// Searches clips by name or keywords.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub async fn search_clips(&self, query: &str) -> ClipResult<Vec<Clip>> {
        let search_pattern = format!("%{query}%");

        let rows = self
            .conn
            .query(
                "SELECT * FROM clips WHERE name LIKE $1 OR keywords LIKE $2 \
                 ORDER BY created_at DESC",
                &[&search_pattern, &search_pattern],
            )
            .await?;

        rows.iter().map(Self::row_to_clip).collect()
    }

    fn row_to_clip(row: &Row) -> ClipResult<Clip> {
        use chrono::DateTime;
        use oximedia_core::types::Rational;
        use std::path::PathBuf;

        let id_str: String = row.try_get("id")?;
        let id = id_str
            .parse()
            .map_err(|e: uuid::Error| ClipError::Serialization(e.to_string()))?;

        let keywords_json: Option<String> = row.try_get("keywords")?;
        let keywords: Vec<String> = match keywords_json {
            Some(ref json) => {
                serde_json::from_str(json).map_err(|e| ClipError::Serialization(e.to_string()))?
            }
            None => Vec::new(),
        };

        let created_at_str: String = row.try_get("created_at")?;
        let created_at = DateTime::parse_from_rfc3339(&created_at_str)
            .map_err(|e| ClipError::Serialization(e.to_string()))?
            .with_timezone(&chrono::Utc);

        let modified_at_str: String = row.try_get("modified_at")?;
        let modified_at = DateTime::parse_from_rfc3339(&modified_at_str)
            .map_err(|e| ClipError::Serialization(e.to_string()))?
            .with_timezone(&chrono::Utc);

        let rating_val: i64 = row.try_get("rating")?;
        let rating = u8::try_from(rating_val)
            .ok()
            .and_then(|v| crate::logging::Rating::from_value(v).ok())
            .unwrap_or(crate::logging::Rating::Unrated);

        let frame_rate = match (
            row.try_get::<Option<i64>>("frame_rate_num")?,
            row.try_get::<Option<i64>>("frame_rate_den")?,
        ) {
            (Some(num), Some(den)) => Some(Rational::new(num, den)),
            _ => None,
        };

        Ok(Clip {
            id,
            file_path: PathBuf::from(row.try_get::<String>("file_path")?),
            name: row.try_get("name")?,
            description: row.try_get("description")?,
            duration: row.try_get("duration")?,
            frame_rate,
            in_point: row.try_get("in_point")?,
            out_point: row.try_get("out_point")?,
            rating,
            is_favorite: row.try_get::<i64>("is_favorite")? != 0,
            is_rejected: row.try_get::<i64>("is_rejected")? != 0,
            keywords,
            markers: Vec::new(), // Would need separate table
            created_at,
            modified_at,
            custom_metadata: row.try_get("custom_metadata")?,
            camera: None,
        })
    }

    /// Returns the number of clips in the database.
    ///
    /// # Errors
    ///
    /// Returns an error if the count query fails.
    pub async fn count_clips(&self) -> ClipResult<i64> {
        let rows = self.conn.query("SELECT COUNT(*) FROM clips", &[]).await?;

        Ok(rows
            .first()
            .map(|row| row.try_get_by_index::<i64>(0))
            .transpose()?
            .unwrap_or(0))
    }

    /// Saves multiple clips in a single database transaction.
    ///
    /// This is significantly faster than calling `save_clip()` in a loop for
    /// large batches because SQLite commits per-transaction rather than
    /// per-statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction cannot be started or any clip fails
    /// to save.
    pub async fn batch_save_clips(&self, clips: &[Clip]) -> ClipResult<()> {
        let mut tx = self.conn.transaction().await?;

        for clip in clips {
            let params = ClipSqlParams::from_clip(clip)?;
            tx.execute(UPSERT_CLIP_SQL, &params.as_params(clip)).await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Returns clips with pagination support.
    ///
    /// `page` is 0-indexed; `page_size` is the number of clips per page.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_clips_page(&self, page: i64, page_size: i64) -> ClipResult<Vec<Clip>> {
        let offset = page * page_size;
        let rows = self
            .conn
            .query(
                "SELECT * FROM clips ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                &[&page_size, &offset],
            )
            .await?;

        rows.iter().map(Self::row_to_clip).collect()
    }

    /// Searches clips with pagination support.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn search_clips_page(
        &self,
        query: &str,
        page: i64,
        page_size: i64,
    ) -> ClipResult<Vec<Clip>> {
        let search_pattern = format!("%{query}%");
        let offset = page * page_size;

        let rows = self
            .conn
            .query(
                "SELECT * FROM clips WHERE name LIKE $1 OR keywords LIKE $2 \
                 ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                &[&search_pattern, &search_pattern, &page_size, &offset],
            )
            .await?;

        rows.iter().map(Self::row_to_clip).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_database_creation() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");
        let count = db.count_clips().await.expect("count_clips should succeed");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_sqlx_style_url_accepted() {
        // Legacy callers may still pass sqlx-style URLs.
        let db = ClipDatabase::new("sqlite::memory:")
            .await
            .expect("sqlite::memory: URL should be accepted");
        let count = db.count_clips().await.expect("count_clips should succeed");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_save_and_get_clip() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clip = Clip::new(PathBuf::from("/test.mov"));
        clip.set_name("Test Clip");
        clip.add_keyword("test");

        db.save_clip(&clip).await.expect("operation should succeed");

        let loaded = db
            .get_clip(&clip.id)
            .await
            .expect("get_clip should succeed");
        assert_eq!(loaded.name, "Test Clip");
        assert_eq!(loaded.keywords.len(), 1);
    }

    #[tokio::test]
    async fn test_save_is_upsert() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clip = Clip::new(PathBuf::from("/test.mov"));
        clip.set_name("Original");
        db.save_clip(&clip)
            .await
            .expect("first save should succeed");

        clip.set_name("Renamed");
        db.save_clip(&clip)
            .await
            .expect("second save should succeed");

        let count = db.count_clips().await.expect("count_clips should succeed");
        assert_eq!(count, 1, "upsert must not create a duplicate row");

        let loaded = db
            .get_clip(&clip.id)
            .await
            .expect("get_clip should succeed");
        assert_eq!(loaded.name, "Renamed");
    }

    #[tokio::test]
    async fn test_delete_clip() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");

        let clip = Clip::new(PathBuf::from("/test.mov"));
        db.save_clip(&clip).await.expect("save should succeed");
        db.delete_clip(&clip.id)
            .await
            .expect("delete should succeed");

        assert!(db.get_clip(&clip.id).await.is_err());
        let count = db.count_clips().await.expect("count_clips should succeed");
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_search_clips() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clip1 = Clip::new(PathBuf::from("/test1.mov"));
        clip1.set_name("Interview");

        let mut clip2 = Clip::new(PathBuf::from("/test2.mov"));
        clip2.set_name("Action Scene");

        db.save_clip(&clip1)
            .await
            .expect("operation should succeed");
        db.save_clip(&clip2)
            .await
            .expect("operation should succeed");

        let results = db
            .search_clips("interview")
            .await
            .expect("search_clips should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Interview");
    }

    #[tokio::test]
    async fn test_batch_save_and_pagination() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clips = Vec::new();
        for i in 0..5 {
            let mut clip = Clip::new(PathBuf::from(format!("/clip{i}.mov")));
            clip.set_name(format!("Clip {i}"));
            clips.push(clip);
        }

        db.batch_save_clips(&clips)
            .await
            .expect("batch_save_clips should succeed");

        let count = db.count_clips().await.expect("count_clips should succeed");
        assert_eq!(count, 5);

        let page0 = db
            .get_clips_page(0, 2)
            .await
            .expect("get_clips_page should succeed");
        assert_eq!(page0.len(), 2);

        let page2 = db
            .get_clips_page(2, 2)
            .await
            .expect("get_clips_page should succeed");
        assert_eq!(page2.len(), 1, "last page holds the remaining clip");

        // All pages combined recover every saved clip exactly once.
        let mut seen = std::collections::HashSet::new();
        for page in 0..3 {
            for clip in db
                .get_clips_page(page, 2)
                .await
                .expect("get_clips_page should succeed")
            {
                assert!(seen.insert(clip.id), "clip must not repeat across pages");
            }
        }
        assert_eq!(seen.len(), 5);
    }

    #[tokio::test]
    async fn test_roundtrip_preserves_fields() {
        let db = ClipDatabase::new(":memory:")
            .await
            .expect("new should succeed");

        let mut clip = Clip::new(PathBuf::from("/full.mov"));
        clip.set_name("Full Clip");
        clip.description = Some("A description".to_string());
        clip.duration = Some(1200);
        clip.in_point = Some(10);
        clip.out_point = Some(110);
        clip.is_favorite = true;
        clip.add_keyword("alpha");
        clip.add_keyword("beta");
        clip.custom_metadata = Some(r#"{"scene":"1A"}"#.to_string());

        db.save_clip(&clip).await.expect("save should succeed");

        let loaded = db
            .get_clip(&clip.id)
            .await
            .expect("get_clip should succeed");
        assert_eq!(loaded.description.as_deref(), Some("A description"));
        assert_eq!(loaded.duration, Some(1200));
        assert_eq!(loaded.in_point, Some(10));
        assert_eq!(loaded.out_point, Some(110));
        assert!(loaded.is_favorite);
        assert!(!loaded.is_rejected);
        assert_eq!(loaded.keywords, vec!["alpha", "beta"]);
        assert_eq!(loaded.custom_metadata.as_deref(), Some(r#"{"scene":"1A"}"#));
    }
}
