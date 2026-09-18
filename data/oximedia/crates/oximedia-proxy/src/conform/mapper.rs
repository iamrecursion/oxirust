//! Media path mapping utilities for conforming.

use crate::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Media path mapper for relinking proxy paths to original paths.
pub struct PathMapper {
    /// Mapping from proxy paths to original paths.
    mappings: HashMap<PathBuf, PathBuf>,

    /// Base proxy directory.
    proxy_base: Option<PathBuf>,

    /// Base original directory.
    original_base: Option<PathBuf>,

    /// Use case-insensitive matching.
    case_insensitive: bool,
}

impl PathMapper {
    /// Create a new path mapper.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
            proxy_base: None,
            original_base: None,
            case_insensitive: false,
        }
    }

    /// Set the proxy base directory.
    #[must_use]
    pub fn with_proxy_base(mut self, base: PathBuf) -> Self {
        self.proxy_base = Some(base);
        self
    }

    /// Set the original base directory.
    #[must_use]
    pub fn with_original_base(mut self, base: PathBuf) -> Self {
        self.original_base = Some(base);
        self
    }

    /// Enable case-insensitive matching.
    #[must_use]
    pub const fn case_insensitive(mut self, enabled: bool) -> Self {
        self.case_insensitive = enabled;
        self
    }

    /// Add a path mapping.
    pub fn add_mapping(&mut self, proxy: PathBuf, original: PathBuf) {
        self.mappings.insert(proxy, original);
    }

    /// Map a proxy path to an original path.
    #[must_use]
    pub fn map(&self, proxy_path: &Path) -> Option<PathBuf> {
        // Try direct mapping first
        if let Some(original) = self.mappings.get(proxy_path) {
            return Some(original.clone());
        }

        // Try base directory mapping
        if let (Some(proxy_base), Some(original_base)) = (&self.proxy_base, &self.original_base) {
            if let Ok(relative) = proxy_path.strip_prefix(proxy_base) {
                return Some(original_base.join(relative));
            }
        }

        // Try case-insensitive matching
        if self.case_insensitive {
            let proxy_lower = proxy_path.to_string_lossy().to_lowercase();
            for (key, value) in &self.mappings {
                if key.to_string_lossy().to_lowercase() == proxy_lower {
                    return Some(value.clone());
                }
            }
        }

        None
    }

    /// Map multiple paths.
    #[must_use]
    pub fn map_batch(&self, proxy_paths: &[PathBuf]) -> Vec<MappingResult> {
        proxy_paths
            .iter()
            .map(|proxy| {
                if let Some(original) = self.map(proxy) {
                    MappingResult::Success {
                        proxy: proxy.clone(),
                        original,
                    }
                } else {
                    MappingResult::Failed {
                        proxy: proxy.clone(),
                    }
                }
            })
            .collect()
    }

    /// Clear all mappings.
    pub fn clear(&mut self) {
        self.mappings.clear();
    }

    /// Get the number of mappings.
    #[must_use]
    pub fn count(&self) -> usize {
        self.mappings.len()
    }

    /// Get all proxy paths.
    #[must_use]
    pub fn proxy_paths(&self) -> Vec<PathBuf> {
        self.mappings.keys().cloned().collect()
    }

    /// Get all original paths.
    #[must_use]
    pub fn original_paths(&self) -> Vec<PathBuf> {
        self.mappings.values().cloned().collect()
    }
}

impl Default for PathMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Path mapping result.
#[derive(Debug, Clone)]
pub enum MappingResult {
    /// Successful mapping.
    Success {
        /// Proxy path.
        proxy: PathBuf,
        /// Original path.
        original: PathBuf,
    },
    /// Failed mapping.
    Failed {
        /// Proxy path.
        proxy: PathBuf,
    },
}

impl MappingResult {
    /// Check if mapping was successful.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    /// Get the proxy path.
    #[must_use]
    pub fn proxy(&self) -> &Path {
        match self {
            Self::Success { proxy, .. } | Self::Failed { proxy } => proxy,
        }
    }
}

/// Automatic path mapper that tries to infer mappings.
pub struct AutoPathMapper {
    mapper: PathMapper,
}

impl AutoPathMapper {
    /// Create a new automatic path mapper.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mapper: PathMapper::new(),
        }
    }

    /// Auto-detect mappings based on filename matching.
    pub fn auto_detect(&mut self, proxy_dir: &Path, original_dir: &Path) -> Result<usize> {
        let mut count = 0;

        // Scan proxy directory
        if let Ok(entries) = std::fs::read_dir(proxy_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        let proxy_path = entry.path();

                        // Try to find matching original
                        if let Some(original_path) =
                            self.find_matching_original(&proxy_path, original_dir)
                        {
                            self.mapper.add_mapping(proxy_path, original_path);
                            count += 1;
                        }
                    }
                }
            }
        }

        Ok(count)
    }

    /// Find matching original file for a proxy.
    fn find_matching_original(&self, proxy: &Path, original_dir: &Path) -> Option<PathBuf> {
        // Get proxy filename without extension
        let proxy_stem = proxy.file_stem()?.to_str()?;

        // Scan original directory for matches
        if let Ok(entries) = std::fs::read_dir(original_dir) {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    // Simple filename matching
                    if filename.contains(proxy_stem) {
                        return Some(entry.path());
                    }
                }
            }
        }

        None
    }

    /// Get the underlying path mapper.
    #[must_use]
    pub const fn mapper(&self) -> &PathMapper {
        &self.mapper
    }
}

impl Default for AutoPathMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_mapper() {
        let mut mapper = PathMapper::new();

        mapper.add_mapping(PathBuf::from("proxy.mp4"), PathBuf::from("original.mov"));

        let result = mapper.map(Path::new("proxy.mp4"));
        assert!(result.is_some());
        assert_eq!(
            result.expect("should succeed in test"),
            PathBuf::from("original.mov")
        );

        assert_eq!(mapper.count(), 1);
    }

    #[test]
    fn test_path_mapper_with_bases() {
        let mapper = PathMapper::new()
            .with_proxy_base(PathBuf::from("/proxies"))
            .with_original_base(PathBuf::from("/originals"));

        let result = mapper.map(Path::new("/proxies/clip1.mp4"));
        assert!(result.is_some());
        assert_eq!(
            result.expect("should succeed in test"),
            PathBuf::from("/originals/clip1.mp4")
        );
    }

    #[test]
    fn test_mapping_result() {
        let result = MappingResult::Success {
            proxy: PathBuf::from("proxy.mp4"),
            original: PathBuf::from("original.mov"),
        };

        assert!(result.is_success());
        assert_eq!(result.proxy(), Path::new("proxy.mp4"));

        let result = MappingResult::Failed {
            proxy: PathBuf::from("proxy.mp4"),
        };

        assert!(!result.is_success());
    }

    #[test]
    fn test_batch_mapping() {
        let mut mapper = PathMapper::new();
        mapper.add_mapping(PathBuf::from("proxy1.mp4"), PathBuf::from("original1.mov"));
        mapper.add_mapping(PathBuf::from("proxy2.mp4"), PathBuf::from("original2.mov"));

        let proxies = vec![
            PathBuf::from("proxy1.mp4"),
            PathBuf::from("proxy2.mp4"),
            PathBuf::from("proxy3.mp4"),
        ];

        let results = mapper.map_batch(&proxies);
        assert_eq!(results.len(), 3);
        assert!(results[0].is_success());
        assert!(results[1].is_success());
        assert!(!results[2].is_success());
    }

    #[test]
    fn test_auto_path_mapper() {
        let auto_mapper = AutoPathMapper::new();
        assert_eq!(auto_mapper.mapper().count(), 0);
    }
}
