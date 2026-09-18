//! EDL-based conforming.

use super::engine::ConformResult;
use crate::{ProxyError, ProxyLinkManager, Result};
use std::path::Path;

/// EDL conformer for relinking proxy edits to original media.
pub struct EdlConformer<'a> {
    #[allow(dead_code)]
    link_manager: &'a ProxyLinkManager,
}

impl<'a> EdlConformer<'a> {
    /// Create a new EDL conformer.
    #[must_use]
    pub const fn new(link_manager: &'a ProxyLinkManager) -> Self {
        Self { link_manager }
    }

    /// Conform an EDL to original media.
    ///
    /// # Errors
    ///
    /// Returns an error if conforming fails.
    pub async fn conform(
        &self,
        edl_path: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<ConformResult> {
        let edl_path = edl_path.as_ref();
        let output = output.as_ref();

        // Validate EDL exists
        if !edl_path.exists() {
            return Err(ProxyError::FileNotFound(edl_path.display().to_string()));
        }

        // This is a placeholder implementation
        // In a real implementation, this would:
        // 1. Parse the EDL using oximedia-edl
        // 2. For each event, relink proxy references to originals
        // 3. Generate a new EDL or timeline with original references
        // 4. Preserve frame-accurate timecode

        tracing::info!(
            "Conforming EDL {} to {}",
            edl_path.display(),
            output.display()
        );

        Ok(ConformResult {
            output_path: output.to_path_buf(),
            clips_relinked: 0,
            clips_failed: 0,
            total_duration: 0.0,
            frame_accurate: true,
        })
    }

    /// Parse an EDL and return the list of referenced proxy files.
    pub fn extract_proxy_references(&self, edl_path: impl AsRef<Path>) -> Result<Vec<String>> {
        let _edl_path = edl_path.as_ref();

        // Placeholder implementation
        // Would parse EDL and extract all referenced media files
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_edl_conformer() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_edl_conform.json");

        let manager = ProxyLinkManager::new(&db_path)
            .await
            .expect("should succeed in test");
        let conformer = EdlConformer::new(&manager);

        let refs = conformer.extract_proxy_references("test.edl");
        assert!(refs.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }
}
