//! Proxy link verification.

use super::manager::ProxyLinkManager;
use crate::Result;

/// Proxy link verifier.
pub struct ProxyVerifier {
    manager: ProxyLinkManager,
}

impl ProxyVerifier {
    /// Create a new proxy verifier.
    #[must_use]
    pub fn new(manager: ProxyLinkManager) -> Self {
        Self { manager }
    }

    /// Verify all links in the database.
    pub fn verify_all(&mut self) -> Result<VerificationReport> {
        let all_links = self.manager.all_links();
        let total = all_links.len();
        let mut valid = 0;
        let mut invalid = 0;
        let mut missing_proxy = Vec::new();
        let mut missing_original = Vec::new();

        for link in &all_links {
            let proxy_exists = link.proxy_path.exists();
            let original_exists = link.original_path.exists();

            if proxy_exists && original_exists {
                valid += 1;
                let _ = self.manager.verify_link(&link.proxy_path);
            } else {
                invalid += 1;
                if !proxy_exists {
                    missing_proxy.push(link.proxy_path.clone());
                }
                if !original_exists {
                    missing_original.push(link.original_path.clone());
                }
            }
        }

        Ok(VerificationReport {
            total,
            valid,
            invalid,
            missing_proxy,
            missing_original,
        })
    }

    /// Verify a specific link.
    pub fn verify_link(&mut self, proxy_path: &std::path::Path) -> Result<bool> {
        self.manager.verify_link(proxy_path)
    }
}

/// Verification report.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Total number of links checked.
    pub total: usize,

    /// Number of valid links.
    pub valid: usize,

    /// Number of invalid links.
    pub invalid: usize,

    /// Proxy files that are missing.
    pub missing_proxy: Vec<std::path::PathBuf>,

    /// Original files that are missing.
    pub missing_original: Vec<std::path::PathBuf>,
}

impl VerificationReport {
    /// Check if all links are valid.
    #[must_use]
    pub const fn all_valid(&self) -> bool {
        self.invalid == 0
    }

    /// Get the percentage of valid links.
    #[must_use]
    pub fn valid_percentage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.valid as f64 / self.total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_report() {
        let report = VerificationReport {
            total: 10,
            valid: 8,
            invalid: 2,
            missing_proxy: Vec::new(),
            missing_original: Vec::new(),
        };

        assert!(!report.all_valid());
        assert_eq!(report.valid_percentage(), 80.0);
    }

    #[test]
    fn test_all_valid() {
        let report = VerificationReport {
            total: 5,
            valid: 5,
            invalid: 0,
            missing_proxy: Vec::new(),
            missing_original: Vec::new(),
        };

        assert!(report.all_valid());
        assert_eq!(report.valid_percentage(), 100.0);
    }
}
