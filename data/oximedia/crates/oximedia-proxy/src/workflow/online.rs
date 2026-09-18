//! Online finishing workflow.

use crate::{ConformEngine, Result};
use std::path::Path;

/// Online finishing workflow manager.
pub struct OnlineWorkflow {
    conform_engine: ConformEngine,
}

impl OnlineWorkflow {
    /// Create a new online workflow with the specified database.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened.
    pub async fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        let conform_engine = ConformEngine::new(db_path).await?;
        Ok(Self { conform_engine })
    }

    /// Relink to high-resolution originals for finishing.
    ///
    /// # Errors
    ///
    /// Returns an error if relinking fails.
    pub async fn relink_to_originals(
        &self,
        edl_path: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<crate::ConformResult> {
        self.conform_engine.conform_from_edl(edl_path, output).await
    }

    /// Verify all original files are available for finishing.
    pub fn verify_originals(&self) -> Result<Vec<String>> {
        // Placeholder: would check all original files exist
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_online_workflow_creation() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_online.json");

        let workflow = OnlineWorkflow::new(&db_path).await;
        assert!(workflow.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }
}
