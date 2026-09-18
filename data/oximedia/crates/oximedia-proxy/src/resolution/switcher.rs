//! Resolution switching for dynamic proxy selection.

use super::manager::{ProxyResolution, ResolutionManager};
use crate::Result;
use std::path::{Path, PathBuf};

/// Resolution switcher for changing between proxy resolutions.
pub struct ResolutionSwitcher {
    manager: ResolutionManager,
    current_resolution: ProxyResolution,
}

impl ResolutionSwitcher {
    /// Create a new resolution switcher.
    #[must_use]
    pub fn new(manager: ResolutionManager) -> Self {
        Self {
            manager,
            current_resolution: ProxyResolution::Quarter,
        }
    }

    /// Switch to a different resolution.
    pub fn switch_to(&mut self, resolution: ProxyResolution) {
        self.current_resolution = resolution;
        tracing::info!("Switched to {} resolution", resolution.name());
    }

    /// Get the current proxy for an original file at the current resolution.
    pub fn get_current_proxy(&self, original: &Path) -> Result<PathBuf> {
        let original_buf = original.to_path_buf();
        self.manager
            .get_best_variant(&original_buf, self.current_resolution)
            .map(|v| v.path.clone())
            .ok_or_else(|| crate::ProxyError::LinkNotFound(original.display().to_string()))
    }

    /// Get the current resolution.
    #[must_use]
    pub const fn current_resolution(&self) -> ProxyResolution {
        self.current_resolution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution::manager::ProxyVariant;

    #[test]
    fn test_resolution_switcher() {
        let mut manager = ResolutionManager::new();

        let original = PathBuf::from("original.mov");
        let variant = ProxyVariant {
            resolution: ProxyResolution::Quarter,
            path: PathBuf::from("proxy_quarter.mp4"),
            file_size: 1000,
            codec: "h264".to_string(),
        };

        manager.add_variant(original.clone(), variant);

        let mut switcher = ResolutionSwitcher::new(manager);
        assert_eq!(switcher.current_resolution(), ProxyResolution::Quarter);

        switcher.switch_to(ProxyResolution::Half);
        assert_eq!(switcher.current_resolution(), ProxyResolution::Half);
    }
}
