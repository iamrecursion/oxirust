//! Bottleneck classification.

use super::detect::Bottleneck;
use serde::{Deserialize, Serialize};

/// Type of bottleneck.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// CPU-bound bottleneck.
    CPU,

    /// Memory-bound bottleneck.
    Memory,

    /// I/O-bound bottleneck.
    IO,

    /// GPU-bound bottleneck.
    GPU,

    /// Synchronization bottleneck.
    Synchronization,

    /// Algorithm inefficiency.
    Algorithm,

    /// Unknown type.
    Unknown,
}

impl BottleneckType {
    /// Get a description of this bottleneck type.
    pub fn description(&self) -> &str {
        match self {
            Self::CPU => "CPU-bound - high CPU utilization",
            Self::Memory => "Memory-bound - memory allocation/access bottleneck",
            Self::IO => "I/O-bound - disk or network I/O bottleneck",
            Self::GPU => "GPU-bound - GPU processing bottleneck",
            Self::Synchronization => "Synchronization - lock contention or thread synchronization",
            Self::Algorithm => "Algorithm - inefficient algorithm or data structure",
            Self::Unknown => "Unknown bottleneck type",
        }
    }

    /// Get suggestions for this bottleneck type.
    pub fn suggestions(&self) -> Vec<&str> {
        match self {
            Self::CPU => vec![
                "Optimize hot loops",
                "Use SIMD instructions",
                "Parallelize computation",
                "Reduce branching",
            ],
            Self::Memory => vec![
                "Reduce allocations",
                "Use object pooling",
                "Optimize data layout",
                "Use stack allocation where possible",
            ],
            Self::IO => vec![
                "Use async I/O",
                "Batch operations",
                "Add caching",
                "Optimize buffer sizes",
            ],
            Self::GPU => vec![
                "Reduce draw calls",
                "Optimize shaders",
                "Use instancing",
                "Reduce texture switches",
            ],
            Self::Synchronization => vec![
                "Reduce lock contention",
                "Use lock-free data structures",
                "Increase granularity",
                "Avoid unnecessary synchronization",
            ],
            Self::Algorithm => vec![
                "Use better data structures",
                "Optimize algorithm complexity",
                "Cache results",
                "Precompute when possible",
            ],
            Self::Unknown => vec!["Profile more to identify bottleneck type"],
        }
    }
}

/// Bottleneck classifier.
#[derive(Debug)]
pub struct BottleneckClassifier;

impl BottleneckClassifier {
    /// Classify a bottleneck based on its characteristics.
    pub fn classify(bottleneck: &Bottleneck) -> BottleneckType {
        let location = bottleneck.location.to_lowercase();

        // Check in order of specificity
        if location.contains("lock") || location.contains("mutex") || location.contains("sync") {
            BottleneckType::Synchronization
        } else if location.contains("alloc") || location.contains("memory") {
            BottleneckType::Memory
        } else if location.contains("gpu")
            || location.contains("render")
            || location.contains("draw")
        {
            BottleneckType::GPU
        } else if location.contains("compute")
            || location.contains("calculate")
            || location.contains("process")
        {
            BottleneckType::CPU
        } else if location.contains("_read")
            || location.contains("_write")
            || location.contains("file_")
            || location.contains("_io")
            || location.starts_with("read_")
            || location.starts_with("write_")
        {
            BottleneckType::IO
        } else {
            BottleneckType::Unknown
        }
    }

    /// Classify and add suggestions to a bottleneck.
    pub fn classify_with_suggestions(mut bottleneck: Bottleneck) -> Bottleneck {
        let bottleneck_type = Self::classify(&bottleneck);
        let suggestions = bottleneck_type.suggestions();

        if let Some(first_suggestion) = suggestions.first() {
            bottleneck = bottleneck.with_suggestion(first_suggestion.to_string());
        }

        bottleneck
    }

    /// Get all suggestions for a bottleneck.
    pub fn get_all_suggestions(bottleneck: &Bottleneck) -> Vec<String> {
        let bottleneck_type = Self::classify(bottleneck);
        bottleneck_type
            .suggestions()
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_bottleneck_type_description() {
        assert!(!BottleneckType::CPU.description().is_empty());
        assert!(!BottleneckType::Memory.description().is_empty());
    }

    #[test]
    fn test_bottleneck_type_suggestions() {
        let suggestions = BottleneckType::CPU.suggestions();
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_classify_cpu() {
        let bottleneck = Bottleneck::new(
            "test".to_string(),
            "compute_heavy_function".to_string(),
            Duration::from_secs(1),
        );

        assert_eq!(
            BottleneckClassifier::classify(&bottleneck),
            BottleneckType::CPU
        );
    }

    #[test]
    fn test_classify_memory() {
        let bottleneck = Bottleneck::new(
            "test".to_string(),
            "allocate_buffer".to_string(),
            Duration::from_secs(1),
        );

        assert_eq!(
            BottleneckClassifier::classify(&bottleneck),
            BottleneckType::Memory
        );
    }

    #[test]
    fn test_classify_sync() {
        let bottleneck = Bottleneck::new(
            "test".to_string(),
            "mutex_lock".to_string(),
            Duration::from_secs(1),
        );

        assert_eq!(
            BottleneckClassifier::classify(&bottleneck),
            BottleneckType::Synchronization
        );
    }

    #[test]
    fn test_classify_with_suggestions() {
        let bottleneck = Bottleneck::new(
            "test".to_string(),
            "compute_function".to_string(),
            Duration::from_secs(1),
        );

        let classified = BottleneckClassifier::classify_with_suggestions(bottleneck);
        assert!(classified.suggestion.is_some());
    }
}
