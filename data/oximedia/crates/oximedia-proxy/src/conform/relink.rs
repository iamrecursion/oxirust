//! Proxy relink utilities.

use crate::{ProxyLinkManager, Result};
use std::path::{Path, PathBuf};

/// Relink proxies to originals.
pub struct Relinker<'a> {
    link_manager: &'a ProxyLinkManager,
}

impl<'a> Relinker<'a> {
    /// Create a new relinker.
    #[must_use]
    pub const fn new(link_manager: &'a ProxyLinkManager) -> Self {
        Self { link_manager }
    }

    /// Relink a single proxy to its original.
    ///
    /// # Errors
    ///
    /// Returns an error if no link exists.
    pub fn relink(&self, proxy_path: &Path) -> Result<PathBuf> {
        self.link_manager
            .get_original(proxy_path)
            .map(|p| p.to_path_buf())
    }

    /// Relink multiple proxies to their originals.
    pub fn relink_batch(&self, proxy_paths: &[PathBuf]) -> Vec<RelinkResult> {
        proxy_paths
            .iter()
            .map(|proxy| match self.relink(proxy) {
                Ok(original) => RelinkResult::Success {
                    proxy: proxy.clone(),
                    original,
                },
                Err(e) => RelinkResult::Failed {
                    proxy: proxy.clone(),
                    error: e.to_string(),
                },
            })
            .collect()
    }
}

/// Result of a relink operation.
#[derive(Debug, Clone)]
pub enum RelinkResult {
    /// Successful relink.
    Success {
        /// Proxy path.
        proxy: PathBuf,
        /// Original path.
        original: PathBuf,
    },
    /// Failed relink.
    Failed {
        /// Proxy path.
        proxy: PathBuf,
        /// Error message.
        error: String,
    },
}

impl RelinkResult {
    /// Check if this result is successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relink_result() {
        let result = RelinkResult::Success {
            proxy: PathBuf::from("proxy.mp4"),
            original: PathBuf::from("original.mov"),
        };

        assert!(result.is_success());
    }
}
