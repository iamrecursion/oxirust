//! Offline editing workflow.

use crate::{ConformEngine, ProxyGenerator, ProxyLinkManager, ProxyPreset, Result};
use std::path::Path;

/// Offline editing workflow manager.
pub struct OfflineWorkflow {
    link_manager: ProxyLinkManager,
    generator: ProxyGenerator,
}

impl OfflineWorkflow {
    /// Create a new offline workflow with the specified database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let link_manager = ProxyLinkManager::new(db_path).await?;
        let generator = ProxyGenerator::new();

        Ok(Self {
            link_manager,
            generator,
        })
    }

    /// Ingest media and create proxy.
    ///
    /// # Errors
    ///
    /// Returns an error if proxy generation or linking fails.
    pub async fn ingest(
        &mut self,
        original: impl AsRef<Path>,
        proxy_output: impl AsRef<Path>,
        preset: ProxyPreset,
    ) -> Result<()> {
        // Generate proxy
        let result = self
            .generator
            .generate(&original, &proxy_output, preset)
            .await?;

        // Link proxy to original
        self.link_manager
            .link_proxy_with_metadata(
                &proxy_output,
                &original,
                preset.to_settings().scale_factor,
                result.codec,
                result.duration,
                None,
                std::collections::HashMap::new(),
            )
            .await?;

        tracing::info!(
            "Ingested {} -> proxy {}",
            original.as_ref().display(),
            proxy_output.as_ref().display()
        );

        Ok(())
    }

    /// Conform edited proxy to original media.
    ///
    /// # Errors
    ///
    /// Returns an error if conforming fails.
    pub async fn conform(
        &self,
        edl_path: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<crate::ConformResult> {
        let db_path = std::env::temp_dir().join("workflow_conform.json");
        let engine = ConformEngine::new(&db_path).await?;
        engine.conform_from_edl(edl_path, output).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_offline_workflow_creation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_offline.json");

        let workflow = OfflineWorkflow::new(&db_path).await;
        assert!(workflow.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }
}
