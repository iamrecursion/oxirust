#![allow(dead_code)]
//! Memory access pattern analysis and layout optimisation hints.

/// The high-level pattern with which memory is accessed during an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessPattern {
    /// Each element is accessed once in address order (ideal for streaming).
    Sequential,
    /// Every N-th element is accessed (common in image row/column scans).
    Strided {
        /// Number of elements between successive accesses.
        stride: usize,
    },
    /// Accesses follow no predictable pattern.
    Random,
    /// Indirect / scatter-gather indexing (e.g. texture sampling).
    Gather,
}

impl AccessPattern {
    /// Returns a [0.0, 1.0] cache-efficiency score for this pattern.
    ///
    /// 1.0 means perfect linear streaming; 0.0 means worst-case random.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn cache_efficiency(&self) -> f32 {
        match self {
            Self::Sequential => 1.0,
            Self::Strided { stride } => {
                // Efficiency decays as stride grows past a cache line (64 B / element).
                // Approximation: 1 / sqrt(stride) clamped to [0.1, 1.0].
                let s = *stride as f32;
                (1.0 / s.sqrt()).clamp(0.1, 1.0)
            }
            Self::Random => 0.1,
            Self::Gather => 0.2,
        }
    }

    /// Returns a short label suitable for logging.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Strided { .. } => "strided",
            Self::Random => "random",
            Self::Gather => "gather",
        }
    }
}

/// Per-region access statistics collected during a profiling run.
#[derive(Debug, Clone)]
pub struct RegionStat {
    /// Name of the memory region (e.g., "frame buffer", "lut").
    pub region: String,
    /// Access pattern observed for this region.
    pub pattern: AccessPattern,
    /// Total bytes accessed during the measurement window.
    pub bytes_accessed: u64,
}

impl RegionStat {
    /// Creates a new `RegionStat`.
    #[must_use]
    pub fn new(region: impl Into<String>, pattern: AccessPattern, bytes_accessed: u64) -> Self {
        Self {
            region: region.into(),
            pattern,
            bytes_accessed,
        }
    }
}

/// Aggregated memory access profile for an operation or pipeline stage.
#[derive(Debug, Default)]
pub struct MemoryAccessProfile {
    regions: Vec<RegionStat>,
}

impl MemoryAccessProfile {
    /// Creates an empty profile.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records statistics for a memory region.
    pub fn record(&mut self, stat: RegionStat) {
        self.regions.push(stat);
    }

    /// Returns the pattern with the highest total bytes accessed.
    ///
    /// Returns `AccessPattern::Sequential` as a safe default when the profile is empty.
    #[must_use]
    pub fn dominant_pattern(&self) -> AccessPattern {
        if self.regions.is_empty() {
            return AccessPattern::Sequential;
        }
        // Group by pattern label and sum bytes.
        let mut seq: u64 = 0;
        let mut strided: u64 = 0;
        let mut random: u64 = 0;
        let mut gather: u64 = 0;
        for r in &self.regions {
            match &r.pattern {
                AccessPattern::Sequential => seq += r.bytes_accessed,
                AccessPattern::Strided { .. } => strided += r.bytes_accessed,
                AccessPattern::Random => random += r.bytes_accessed,
                AccessPattern::Gather => gather += r.bytes_accessed,
            }
        }
        // Find the weighted-average stride across all strided regions.
        let total_strided_bytes: u64 = self
            .regions
            .iter()
            .filter_map(|r| {
                if let AccessPattern::Strided { stride } = r.pattern {
                    Some(stride as u64 * r.bytes_accessed)
                } else {
                    None
                }
            })
            .sum();
        let dominant_stride = total_strided_bytes
            .checked_div(strided)
            .map(|v| v as usize)
            .unwrap_or(1);

        let max = seq.max(strided).max(random).max(gather);
        if max == seq {
            AccessPattern::Sequential
        } else if max == strided {
            AccessPattern::Strided {
                stride: dominant_stride,
            }
        } else if max == random {
            AccessPattern::Random
        } else {
            AccessPattern::Gather
        }
    }

    /// Returns the weighted cache efficiency across all regions.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn weighted_cache_efficiency(&self) -> f32 {
        let total_bytes: u64 = self.regions.iter().map(|r| r.bytes_accessed).sum();
        if total_bytes == 0 {
            return 1.0;
        }
        let weighted: f64 = self
            .regions
            .iter()
            .map(|r| r.pattern.cache_efficiency() as f64 * r.bytes_accessed as f64)
            .sum();
        (weighted / total_bytes as f64) as f32
    }

    /// Returns the total number of regions recorded.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }
}

/// Suggested memory layout for optimizing a given access profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutSuggestion {
    /// Keep the current layout — it is already optimal.
    NoChange,
    /// Interleave channels for sequential processing (AoS → SoA).
    Interleave,
    /// Pack data into tiles to improve 2-D locality.
    TiledLayout {
        /// Suggested tile width in elements.
        tile_width: u32,
        /// Suggested tile height in elements.
        tile_height: u32,
    },
    /// Prefetch data ahead of consumption.
    Prefetch {
        /// Suggested prefetch distance in cache lines.
        distance: u32,
    },
}

/// Produces layout optimisation suggestions from a memory access profile.
#[derive(Debug, Default)]
pub struct MemoryOptimizer;

impl MemoryOptimizer {
    /// Creates a new `MemoryOptimizer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Analyses the profile and returns a layout suggestion.
    #[must_use]
    pub fn suggest_layout(&self, profile: &MemoryAccessProfile) -> LayoutSuggestion {
        match profile.dominant_pattern() {
            AccessPattern::Sequential => {
                // Already optimal; suggest prefetching to hide latency.
                LayoutSuggestion::Prefetch { distance: 8 }
            }
            AccessPattern::Strided { stride } if stride <= 4 => {
                // Small stride — interleaving can help.
                LayoutSuggestion::Interleave
            }
            AccessPattern::Strided { .. } => {
                // Large stride — tiling improves 2-D locality.
                LayoutSuggestion::TiledLayout {
                    tile_width: 32,
                    tile_height: 32,
                }
            }
            AccessPattern::Random | AccessPattern::Gather => {
                // Random/gather: tiling still helps somewhat.
                LayoutSuggestion::TiledLayout {
                    tile_width: 16,
                    tile_height: 16,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_cache_efficiency() {
        assert!((AccessPattern::Sequential.cache_efficiency() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_random_cache_efficiency() {
        assert!((AccessPattern::Random.cache_efficiency() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_gather_cache_efficiency() {
        assert!((AccessPattern::Gather.cache_efficiency() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_strided_efficiency_decreases_with_stride() {
        let s1 = AccessPattern::Strided { stride: 1 }.cache_efficiency();
        let s16 = AccessPattern::Strided { stride: 16 }.cache_efficiency();
        assert!(s1 > s16);
    }

    #[test]
    fn test_strided_efficiency_floor() {
        // Very large stride should not go below 0.1.
        let eff = AccessPattern::Strided { stride: 10_000 }.cache_efficiency();
        assert!(eff >= 0.1);
    }

    #[test]
    fn test_access_pattern_labels() {
        assert_eq!(AccessPattern::Sequential.label(), "sequential");
        assert_eq!(AccessPattern::Random.label(), "random");
        assert_eq!(AccessPattern::Gather.label(), "gather");
        assert_eq!(AccessPattern::Strided { stride: 4 }.label(), "strided");
    }

    #[test]
    fn test_profile_empty_dominant_pattern() {
        let profile = MemoryAccessProfile::new();
        assert_eq!(profile.dominant_pattern(), AccessPattern::Sequential);
    }

    #[test]
    fn test_profile_dominant_random_when_heaviest() {
        let mut profile = MemoryAccessProfile::new();
        profile.record(RegionStat::new("lut", AccessPattern::Random, 1_000_000));
        profile.record(RegionStat::new("fb", AccessPattern::Sequential, 100));
        assert_eq!(profile.dominant_pattern(), AccessPattern::Random);
    }

    #[test]
    fn test_profile_weighted_efficiency_all_sequential() {
        let mut profile = MemoryAccessProfile::new();
        profile.record(RegionStat::new("fb", AccessPattern::Sequential, 4096));
        assert!((profile.weighted_cache_efficiency() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_profile_region_count() {
        let mut profile = MemoryAccessProfile::new();
        profile.record(RegionStat::new("a", AccessPattern::Sequential, 100));
        profile.record(RegionStat::new("b", AccessPattern::Random, 200));
        assert_eq!(profile.region_count(), 2);
    }

    #[test]
    fn test_optimizer_sequential_suggests_prefetch() {
        let optimizer = MemoryOptimizer::new();
        let mut profile = MemoryAccessProfile::new();
        profile.record(RegionStat::new("fb", AccessPattern::Sequential, 1024));
        assert!(matches!(
            optimizer.suggest_layout(&profile),
            LayoutSuggestion::Prefetch { .. }
        ));
    }

    #[test]
    fn test_optimizer_random_suggests_tiled() {
        let optimizer = MemoryOptimizer::new();
        let mut profile = MemoryAccessProfile::new();
        profile.record(RegionStat::new("lut", AccessPattern::Random, 1024));
        assert!(matches!(
            optimizer.suggest_layout(&profile),
            LayoutSuggestion::TiledLayout { .. }
        ));
    }

    #[test]
    fn test_optimizer_small_stride_suggests_interleave() {
        let optimizer = MemoryOptimizer::new();
        let mut profile = MemoryAccessProfile::new();
        profile.record(RegionStat::new(
            "ch",
            AccessPattern::Strided { stride: 2 },
            1024,
        ));
        assert_eq!(
            optimizer.suggest_layout(&profile),
            LayoutSuggestion::Interleave
        );
    }

    #[test]
    fn test_optimizer_large_stride_suggests_tiled() {
        let optimizer = MemoryOptimizer::new();
        let mut profile = MemoryAccessProfile::new();
        profile.record(RegionStat::new(
            "row",
            AccessPattern::Strided { stride: 1920 },
            1024,
        ));
        assert!(matches!(
            optimizer.suggest_layout(&profile),
            LayoutSuggestion::TiledLayout { .. }
        ));
    }
}
