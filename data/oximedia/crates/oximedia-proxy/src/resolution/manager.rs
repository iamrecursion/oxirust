//! Multi-resolution proxy management.

use std::collections::HashMap;
use std::path::PathBuf;

/// Multi-resolution proxy manager.
pub struct ResolutionManager {
    /// Map of original to resolution variants.
    variants: HashMap<PathBuf, Vec<ProxyVariant>>,
}

impl ResolutionManager {
    /// Create a new resolution manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variants: HashMap::new(),
        }
    }

    /// Add a proxy variant for an original file.
    pub fn add_variant(&mut self, original: PathBuf, variant: ProxyVariant) {
        self.variants
            .entry(original)
            .or_insert_with(Vec::new)
            .push(variant);
    }

    /// Get all variants for an original file.
    #[must_use]
    pub fn get_variants(&self, original: &PathBuf) -> Option<&Vec<ProxyVariant>> {
        self.variants.get(original)
    }

    /// Get the best variant for a target resolution.
    #[must_use]
    pub fn get_best_variant(
        &self,
        original: &PathBuf,
        target_resolution: ProxyResolution,
    ) -> Option<&ProxyVariant> {
        self.variants.get(original).and_then(|variants| {
            variants
                .iter()
                .find(|v| v.resolution == target_resolution)
                .or_else(|| variants.first())
        })
    }
}

impl Default for ResolutionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Proxy resolution levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProxyResolution {
    /// Quarter resolution (25%).
    Quarter,
    /// Half resolution (50%).
    Half,
    /// Full resolution (100%).
    Full,
}

impl ProxyResolution {
    /// Get the scale factor for this resolution.
    #[must_use]
    pub const fn scale_factor(&self) -> f32 {
        match self {
            Self::Quarter => 0.25,
            Self::Half => 0.5,
            Self::Full => 1.0,
        }
    }

    /// Get the name of this resolution.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Quarter => "Quarter",
            Self::Half => "Half",
            Self::Full => "Full",
        }
    }
}

/// A proxy variant at a specific resolution.
#[derive(Debug, Clone)]
pub struct ProxyVariant {
    /// Resolution level.
    pub resolution: ProxyResolution,
    /// Proxy file path.
    pub path: PathBuf,
    /// File size in bytes.
    pub file_size: u64,
    /// Codec used.
    pub codec: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_manager() {
        let mut manager = ResolutionManager::new();

        let original = PathBuf::from("original.mov");
        let variant = ProxyVariant {
            resolution: ProxyResolution::Quarter,
            path: PathBuf::from("proxy_quarter.mp4"),
            file_size: 1000,
            codec: "h264".to_string(),
        };

        manager.add_variant(original.clone(), variant);

        let variants = manager.get_variants(&original);
        assert!(variants.is_some());
        assert_eq!(variants.expect("should succeed in test").len(), 1);
    }

    #[test]
    fn test_proxy_resolution() {
        assert_eq!(ProxyResolution::Quarter.scale_factor(), 0.25);
        assert_eq!(ProxyResolution::Half.scale_factor(), 0.5);
        assert_eq!(ProxyResolution::Full.scale_factor(), 1.0);

        assert_eq!(ProxyResolution::Quarter.name(), "Quarter");
    }
}
