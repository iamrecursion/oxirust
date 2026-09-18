//! Proxy link manager for managing proxy-original relationships.

use super::database::{LinkDatabase, ProxyLinkRecord};
use crate::{ProxyError, Result};
use std::collections::HashMap;
use std::path::Path;

/// Proxy link manager.
pub struct ProxyLinkManager {
    database: LinkDatabase,
}

impl ProxyLinkManager {
    /// Create a new proxy link manager with the specified database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let database = LinkDatabase::new(db_path).await?;
        Ok(Self { database })
    }

    /// Link a proxy to its original file.
    ///
    /// # Errors
    ///
    /// Returns an error if the link cannot be created.
    pub async fn link_proxy(
        &mut self,
        proxy_path: impl AsRef<Path>,
        original_path: impl AsRef<Path>,
    ) -> Result<()> {
        self.link_proxy_with_metadata(
            proxy_path,
            original_path,
            0.25,
            "h264",
            0.0,
            None,
            HashMap::new(),
        )
        .await
    }

    /// Link a proxy with full metadata.
    #[allow(clippy::too_many_arguments)]
    pub async fn link_proxy_with_metadata(
        &mut self,
        proxy_path: impl AsRef<Path>,
        original_path: impl AsRef<Path>,
        scale_factor: f32,
        codec: impl Into<String>,
        duration: f64,
        timecode: Option<String>,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let record = ProxyLinkRecord {
            proxy_path: proxy_path.as_ref().to_path_buf(),
            original_path: original_path.as_ref().to_path_buf(),
            scale_factor,
            codec: codec.into(),
            duration,
            timecode,
            created_at: current_timestamp(),
            verified_at: None,
            metadata,
        };

        self.database.add_link(record)?;

        tracing::info!(
            "Linked proxy {} to original {}",
            proxy_path.as_ref().display(),
            original_path.as_ref().display()
        );

        Ok(())
    }

    /// Get the original path for a proxy.
    ///
    /// # Errors
    ///
    /// Returns an error if no link exists for the proxy.
    pub fn get_original(&self, proxy_path: impl AsRef<Path>) -> Result<&Path> {
        self.database
            .get_link(proxy_path.as_ref())
            .map(|link| link.original_path.as_path())
            .ok_or_else(|| ProxyError::LinkNotFound(proxy_path.as_ref().display().to_string()))
    }

    /// Get the proxy path for an original file.
    ///
    /// # Errors
    ///
    /// Returns an error if no link exists for the original.
    pub fn get_proxy(&self, original_path: impl AsRef<Path>) -> Result<&Path> {
        self.database
            .get_link_by_original(original_path.as_ref())
            .map(|link| link.proxy_path.as_path())
            .ok_or_else(|| ProxyError::LinkNotFound(original_path.as_ref().display().to_string()))
    }

    /// Check if a proxy link exists.
    #[must_use]
    pub fn has_link(&self, proxy_path: impl AsRef<Path>) -> bool {
        self.database.get_link(proxy_path.as_ref()).is_some()
    }

    /// Remove a proxy link.
    pub fn remove_link(&mut self, proxy_path: impl AsRef<Path>) -> Result<()> {
        self.database.remove_link(proxy_path.as_ref())?;
        Ok(())
    }

    /// Verify a proxy link and update its verification timestamp.
    pub fn verify_link(&mut self, proxy_path: impl AsRef<Path>) -> Result<bool> {
        let link = self
            .database
            .get_link(proxy_path.as_ref())
            .ok_or_else(|| ProxyError::LinkNotFound(proxy_path.as_ref().display().to_string()))?;

        // Check if both files exist
        let proxy_exists = link.proxy_path.exists();
        let original_exists = link.original_path.exists();

        if proxy_exists && original_exists {
            self.database
                .update_verification(proxy_path.as_ref(), current_timestamp())?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get all links in the database.
    #[must_use]
    pub fn all_links(&self) -> Vec<ProxyLink> {
        self.database
            .all_links()
            .iter()
            .map(|record| ProxyLink {
                proxy_path: record.proxy_path.clone(),
                original_path: record.original_path.clone(),
                scale_factor: record.scale_factor,
                codec: record.codec.clone(),
                duration: record.duration,
                timecode: record.timecode.clone(),
            })
            .collect()
    }

    /// Get the number of links in the database.
    #[must_use]
    pub fn count(&self) -> usize {
        self.database.count()
    }
}

/// A proxy link (public API).
#[derive(Debug, Clone)]
pub struct ProxyLink {
    /// Proxy file path.
    pub proxy_path: std::path::PathBuf,

    /// Original file path.
    pub original_path: std::path::PathBuf,

    /// Proxy resolution scale factor.
    pub scale_factor: f32,

    /// Proxy codec.
    pub codec: String,

    /// Duration in seconds.
    pub duration: f64,

    /// Timecode (if available).
    pub timecode: Option<String>,
}

/// Get the current Unix timestamp.
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("infallible: system clock is always after UNIX_EPOCH")
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_link_manager() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_manager.json");

        let mut manager = ProxyLinkManager::new(&db_path)
            .await
            .expect("should succeed in test");

        manager
            .link_proxy("proxy.mp4", "original.mov")
            .await
            .expect("should succeed in test");

        let original = manager
            .get_original("proxy.mp4")
            .expect("should succeed in test");
        assert_eq!(original, Path::new("original.mov"));

        assert_eq!(manager.count(), 1);

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn test_has_link() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_has_link.json");

        let mut manager = ProxyLinkManager::new(&db_path)
            .await
            .expect("should succeed in test");

        assert!(!manager.has_link("proxy.mp4"));

        manager
            .link_proxy("proxy.mp4", "original.mov")
            .await
            .expect("should succeed in test");

        assert!(manager.has_link("proxy.mp4"));

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }
}
