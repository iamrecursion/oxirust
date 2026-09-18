//! Round-trip workflow (offline to online to delivery).

use crate::{OfflineWorkflow, ProxyPreset, Result};
use std::path::Path;

/// Round-trip workflow manager for complete offline-online pipeline.
pub struct RoundtripWorkflow {
    offline: OfflineWorkflow,
}

impl RoundtripWorkflow {
    /// Create a new round-trip workflow.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub async fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let offline = OfflineWorkflow::new(db_path).await?;
        Ok(Self { offline })
    }

    /// Phase 1: Ingest originals and create proxies.
    ///
    /// # Errors
    ///
    /// Returns an error if ingest fails.
    pub async fn phase_ingest(
        &mut self,
        originals: &[impl AsRef<Path>],
        proxy_dir: impl AsRef<Path>,
        preset: ProxyPreset,
    ) -> Result<()> {
        for original in originals {
            let filename = original
                .as_ref()
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("proxy.mp4");

            let proxy_path = proxy_dir.as_ref().join(filename);

            self.offline.ingest(original, proxy_path, preset).await?;
        }

        Ok(())
    }

    /// Phase 2: Edit with proxies (external editing application).
    pub fn phase_edit(&self) -> Result<String> {
        Ok("Edit with your NLE using proxy files".to_string())
    }

    /// Phase 3: Conform to originals for finishing.
    ///
    /// # Errors
    ///
    /// Returns an error if conform fails.
    pub async fn phase_conform(
        &self,
        edl_path: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<crate::ConformResult> {
        self.offline.conform(edl_path, output).await
    }

    /// Phase 4: Final delivery (rendering, packaging, etc.).
    pub async fn phase_deliver(&self, _output: impl AsRef<Path>) -> Result<()> {
        // Placeholder: would handle final delivery tasks
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_roundtrip_workflow_creation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_roundtrip.json");

        let workflow = RoundtripWorkflow::new(&db_path).await;
        assert!(workflow.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }
}
