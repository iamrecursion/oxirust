//! Validation checker for proxy workflows.

use super::report::ValidationReport;
use crate::{ProxyLinkManager, Result};

/// Validation checker for proxy workflows.
pub struct ValidationChecker<'a> {
    link_manager: &'a ProxyLinkManager,
}

impl<'a> ValidationChecker<'a> {
    /// Create a new validation checker.
    #[must_use]
    pub const fn new(link_manager: &'a ProxyLinkManager) -> Self {
        Self { link_manager }
    }

    /// Validate all proxy links.
    pub fn validate(&self) -> Result<ValidationReport> {
        let all_links = self.link_manager.all_links();
        let total = all_links.len();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        for link in &all_links {
            // Check if files exist
            if !link.proxy_path.exists() {
                errors.push(format!(
                    "Proxy file not found: {}",
                    link.proxy_path.display()
                ));
            }

            if !link.original_path.exists() {
                errors.push(format!(
                    "Original file not found: {}",
                    link.original_path.display()
                ));
            }

            // Check for potential issues
            if link.duration == 0.0 {
                warnings.push(format!(
                    "Zero duration for link: {}",
                    link.proxy_path.display()
                ));
            }
        }

        Ok(ValidationReport {
            total_links: total,
            valid_links: total - errors.len() / 2,
            errors,
            warnings,
        })
    }

    /// Quick validation check (just existence).
    pub fn quick_validate(&self) -> Result<bool> {
        let all_links = self.link_manager.all_links();

        for link in &all_links {
            if !link.proxy_path.exists() || !link.original_path.exists() {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_validation_checker() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("test_validation.json");

        let manager = ProxyLinkManager::new(&db_path)
            .await
            .expect("should succeed in test");
        let checker = ValidationChecker::new(&manager);

        let report = checker.validate();
        assert!(report.is_ok());

        // Clean up
        let _ = std::fs::remove_file(db_path);
    }
}
