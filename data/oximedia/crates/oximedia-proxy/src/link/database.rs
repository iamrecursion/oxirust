//! Proxy link database for storing proxy-original relationships.

use crate::{ProxyError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Proxy link database.
pub struct LinkDatabase {
    /// Database file path.
    db_path: PathBuf,

    /// In-memory link storage.
    links: HashMap<PathBuf, ProxyLinkRecord>,
}

/// A single proxy link record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyLinkRecord {
    /// Proxy file path.
    pub proxy_path: PathBuf,

    /// Original file path.
    pub original_path: PathBuf,

    /// Proxy resolution scale factor.
    pub scale_factor: f32,

    /// Proxy codec.
    pub codec: String,

    /// Original duration in seconds.
    pub duration: f64,

    /// Original timecode (if available).
    pub timecode: Option<String>,

    /// Creation timestamp.
    pub created_at: i64,

    /// Last verified timestamp.
    pub verified_at: Option<i64>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl LinkDatabase {
    /// Create or open a link database at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created or opened.
    pub async fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();

        // Create parent directory if needed
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Load existing database or create new one
        let links = if db_path.exists() {
            Self::load_from_file(&db_path)?
        } else {
            HashMap::new()
        };

        Ok(Self { db_path, links })
    }

    /// Add a proxy link to the database.
    pub fn add_link(&mut self, record: ProxyLinkRecord) -> Result<()> {
        self.links.insert(record.proxy_path.clone(), record);
        self.save()?;
        Ok(())
    }

    /// Get a link by proxy path.
    #[must_use]
    pub fn get_link(&self, proxy_path: &Path) -> Option<&ProxyLinkRecord> {
        self.links.get(proxy_path)
    }

    /// Get a link by original path.
    #[must_use]
    pub fn get_link_by_original(&self, original_path: &Path) -> Option<&ProxyLinkRecord> {
        self.links
            .values()
            .find(|link| link.original_path == original_path)
    }

    /// Remove a link by proxy path.
    pub fn remove_link(&mut self, proxy_path: &Path) -> Result<Option<ProxyLinkRecord>> {
        let result = self.links.remove(proxy_path);
        self.save()?;
        Ok(result)
    }

    /// Update a link's verification timestamp.
    pub fn update_verification(&mut self, proxy_path: &Path, timestamp: i64) -> Result<()> {
        if let Some(link) = self.links.get_mut(proxy_path) {
            link.verified_at = Some(timestamp);
            self.save()?;
            Ok(())
        } else {
            Err(ProxyError::LinkNotFound(proxy_path.display().to_string()))
        }
    }

    /// Get all links in the database.
    #[must_use]
    pub fn all_links(&self) -> Vec<&ProxyLinkRecord> {
        self.links.values().collect()
    }

    /// Get the number of links in the database.
    #[must_use]
    pub fn count(&self) -> usize {
        self.links.len()
    }

    /// Save the database to disk.
    fn save(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.links)
            .map_err(|e| ProxyError::DatabaseError(format!("Failed to serialize database: {e}")))?;

        std::fs::write(&self.db_path, json)?;
        Ok(())
    }

    /// Load the database from disk.
    fn load_from_file(path: &Path) -> Result<HashMap<PathBuf, ProxyLinkRecord>> {
        let content = std::fs::read_to_string(path)?;
        let links = serde_json::from_str(&content).map_err(|e| {
            ProxyError::DatabaseError(format!("Failed to deserialize database: {e}"))
        })?;
        Ok(links)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_database_creation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_links.json");

        let db = LinkDatabase::new(&db_path).await;
        assert!(db.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn test_add_and_get_link() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_links2.json");

        let mut db = LinkDatabase::new(&db_path)
            .await
            .expect("should succeed in test");

        let record = ProxyLinkRecord {
            proxy_path: PathBuf::from("proxy.mp4"),
            original_path: PathBuf::from("original.mov"),
            scale_factor: 0.25,
            codec: "h264".to_string(),
            duration: 10.0,
            timecode: Some("01:00:00:00".to_string()),
            created_at: 123456789,
            verified_at: None,
            metadata: HashMap::new(),
        };

        db.add_link(record.clone()).expect("should succeed in test");

        let retrieved = db.get_link(Path::new("proxy.mp4"));
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.expect("should succeed in test").original_path,
            PathBuf::from("original.mov")
        );

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn test_remove_link() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_links3.json");

        let mut db = LinkDatabase::new(&db_path)
            .await
            .expect("should succeed in test");

        let record = ProxyLinkRecord {
            proxy_path: PathBuf::from("proxy.mp4"),
            original_path: PathBuf::from("original.mov"),
            scale_factor: 0.25,
            codec: "h264".to_string(),
            duration: 10.0,
            timecode: None,
            created_at: 123456789,
            verified_at: None,
            metadata: HashMap::new(),
        };

        db.add_link(record).expect("should succeed in test");
        assert_eq!(db.count(), 1);

        db.remove_link(Path::new("proxy.mp4"))
            .expect("should succeed in test");
        assert_eq!(db.count(), 0);

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }
}
