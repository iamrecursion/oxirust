//! Texture sampler configuration and caching.
//!
//! Provides enumerations for texture filtering and wrapping modes, a
//! configuration struct combining them, and a simple in-memory cache so
//! identical configurations share the same sampler object.

#![allow(dead_code)]

use std::collections::HashMap;

/// How texels are filtered when a texture is sampled at a non-integer
/// coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FilterMode {
    /// Return the nearest texel with no interpolation (pixelated).
    Nearest,
    /// Linearly interpolate between adjacent texels (smooth).
    #[default]
    Linear,
    /// Use mipmaps with nearest selection between levels.
    NearestMipmapNearest,
    /// Use mipmaps with linear interpolation between levels.
    LinearMipmapLinear,
}

impl FilterMode {
    /// Return `true` if this filter mode uses mipmaps.
    #[must_use]
    pub const fn uses_mipmaps(self) -> bool {
        matches!(self, Self::NearestMipmapNearest | Self::LinearMipmapLinear)
    }
}

/// How texture coordinates outside [0, 1] are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum WrapMode {
    /// Texture repeats (tiles) indefinitely.
    Repeat,
    /// Texture tiles but every other repetition is mirrored.
    MirrorRepeat,
    /// Coordinates are clamped to [0, 1]; edge texels are stretched.
    #[default]
    ClampToEdge,
    /// Coordinates outside [0, 1] sample a configured border colour.
    ClampToBorder,
}

/// The level-of-detail bias applied when selecting a mipmap level.
///
/// Positive values blur the texture; negative values sharpen it.
pub type LodBias = f32;

/// Complete configuration for a texture sampler.
///
/// Combines filter and wrap modes together with an optional LOD bias and
/// the maximum anisotropy level.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    /// Filtering applied when the texture is minified.
    pub min_filter: FilterMode,
    /// Filtering applied when the texture is magnified.
    pub mag_filter: FilterMode,
    /// Wrapping applied along the U (horizontal) texture axis.
    pub wrap_u: WrapMode,
    /// Wrapping applied along the V (vertical) texture axis.
    pub wrap_v: WrapMode,
    /// Level-of-detail bias (applied after automatic mip selection).
    pub lod_bias: LodBias,
    /// Maximum anisotropy level (1 = isotropic, 16 = maximum).
    pub max_anisotropy: u8,
}

impl SamplerConfig {
    /// A minimal default: linear filter, clamp-to-edge, no anisotropy.
    #[must_use]
    pub fn linear_clamp() -> Self {
        Self {
            min_filter: FilterMode::Linear,
            mag_filter: FilterMode::Linear,
            wrap_u: WrapMode::ClampToEdge,
            wrap_v: WrapMode::ClampToEdge,
            lod_bias: 0.0,
            max_anisotropy: 1,
        }
    }

    /// Nearest-neighbour filter, repeat wrapping — good for tiling textures.
    #[must_use]
    pub fn nearest_repeat() -> Self {
        Self {
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            wrap_u: WrapMode::Repeat,
            wrap_v: WrapMode::Repeat,
            lod_bias: 0.0,
            max_anisotropy: 1,
        }
    }

    /// Trilinear mipmap filtering with 16× anisotropy — high quality.
    #[must_use]
    pub fn trilinear_anisotropic() -> Self {
        Self {
            min_filter: FilterMode::LinearMipmapLinear,
            mag_filter: FilterMode::Linear,
            wrap_u: WrapMode::Repeat,
            wrap_v: WrapMode::Repeat,
            lod_bias: 0.0,
            max_anisotropy: 16,
        }
    }

    /// Return `true` if this config requires mipmap generation.
    #[must_use]
    pub fn needs_mipmaps(&self) -> bool {
        self.min_filter.uses_mipmaps()
    }

    /// Clamp `max_anisotropy` to the hardware-reported limit.
    pub fn clamp_anisotropy(&mut self, hardware_max: u8) {
        self.max_anisotropy = self.max_anisotropy.min(hardware_max);
    }
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self::linear_clamp()
    }
}

/// A handle to a cached sampler, returned by [`SamplerCache`].
///
/// The `u64` is an opaque identifier generated from the config hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SamplerHandle(u64);

impl SamplerHandle {
    /// The underlying numeric identifier.
    #[must_use]
    pub fn id(self) -> u64 {
        self.0
    }
}

/// In-memory cache that de-duplicates identical sampler configurations.
///
/// When the same [`SamplerConfig`] is requested more than once the same
/// [`SamplerHandle`] is returned, avoiding redundant GPU object creation.
///
/// # Example
///
/// ```
/// use oximedia_gpu::sampler::{SamplerCache, SamplerConfig};
///
/// let mut cache = SamplerCache::new();
/// let h1 = cache.get_or_insert(SamplerConfig::linear_clamp());
/// let h2 = cache.get_or_insert(SamplerConfig::linear_clamp());
/// assert_eq!(h1, h2); // same config → same handle
/// assert_eq!(cache.len(), 1);
/// ```
#[derive(Debug, Default)]
pub struct SamplerCache {
    /// Maps a stable config hash to its handle.
    entries: HashMap<u64, SamplerHandle>,
    /// Counter used to generate monotonically increasing handles.
    next_id: u64,
}

impl SamplerCache {
    /// Create an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute a stable hash key for a [`SamplerConfig`].
    fn config_key(cfg: &SamplerConfig) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        cfg.min_filter.hash(&mut h);
        cfg.mag_filter.hash(&mut h);
        cfg.wrap_u.hash(&mut h);
        cfg.wrap_v.hash(&mut h);
        // Treat lod_bias bits as a stable integer for hashing.
        cfg.lod_bias.to_bits().hash(&mut h);
        cfg.max_anisotropy.hash(&mut h);
        h.finish()
    }

    /// Return an existing handle if `config` is already cached, or allocate a
    /// new one and store it.
    pub fn get_or_insert(&mut self, config: SamplerConfig) -> SamplerHandle {
        let key = Self::config_key(&config);
        if let Some(&handle) = self.entries.get(&key) {
            return handle;
        }
        let handle = SamplerHandle(self.next_id);
        self.next_id += 1;
        self.entries.insert(key, handle);
        handle
    }

    /// Number of unique sampler configurations in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all cached entries and reset the handle counter.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_id = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_mode_linear_no_mipmap() {
        assert!(!FilterMode::Linear.uses_mipmaps());
    }

    #[test]
    fn filter_mode_nearest_no_mipmap() {
        assert!(!FilterMode::Nearest.uses_mipmaps());
    }

    #[test]
    fn filter_mode_mipmap_variants_use_mipmaps() {
        assert!(FilterMode::NearestMipmapNearest.uses_mipmaps());
        assert!(FilterMode::LinearMipmapLinear.uses_mipmaps());
    }

    #[test]
    fn filter_mode_default_is_linear() {
        assert_eq!(FilterMode::default(), FilterMode::Linear);
    }

    #[test]
    fn wrap_mode_default_is_clamp_to_edge() {
        assert_eq!(WrapMode::default(), WrapMode::ClampToEdge);
    }

    #[test]
    fn sampler_config_linear_clamp_no_mipmaps() {
        let cfg = SamplerConfig::linear_clamp();
        assert!(!cfg.needs_mipmaps());
    }

    #[test]
    fn sampler_config_trilinear_needs_mipmaps() {
        let cfg = SamplerConfig::trilinear_anisotropic();
        assert!(cfg.needs_mipmaps());
    }

    #[test]
    fn sampler_config_clamp_anisotropy() {
        let mut cfg = SamplerConfig::trilinear_anisotropic();
        cfg.clamp_anisotropy(4);
        assert_eq!(cfg.max_anisotropy, 4);
    }

    #[test]
    fn sampler_config_clamp_anisotropy_no_increase() {
        let mut cfg = SamplerConfig::linear_clamp();
        cfg.clamp_anisotropy(32);
        assert_eq!(cfg.max_anisotropy, 1); // was 1, not increased
    }

    #[test]
    fn sampler_cache_deduplicate() {
        let mut cache = SamplerCache::new();
        let h1 = cache.get_or_insert(SamplerConfig::linear_clamp());
        let h2 = cache.get_or_insert(SamplerConfig::linear_clamp());
        assert_eq!(h1, h2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn sampler_cache_different_configs_different_handles() {
        let mut cache = SamplerCache::new();
        let h1 = cache.get_or_insert(SamplerConfig::linear_clamp());
        let h2 = cache.get_or_insert(SamplerConfig::nearest_repeat());
        assert_ne!(h1, h2);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn sampler_cache_is_empty_initially() {
        let cache = SamplerCache::new();
        assert!(cache.is_empty());
    }

    #[test]
    fn sampler_cache_clear_resets() {
        let mut cache = SamplerCache::new();
        cache.get_or_insert(SamplerConfig::linear_clamp());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.next_id, 0);
    }

    #[test]
    fn sampler_handle_id() {
        let mut cache = SamplerCache::new();
        let h = cache.get_or_insert(SamplerConfig::linear_clamp());
        assert_eq!(h.id(), 0);
    }

    #[test]
    fn sampler_config_default_is_linear_clamp() {
        let a = SamplerConfig::default();
        let b = SamplerConfig::linear_clamp();
        assert_eq!(a, b);
    }

    #[test]
    fn nearest_repeat_config_values() {
        let cfg = SamplerConfig::nearest_repeat();
        assert_eq!(cfg.min_filter, FilterMode::Nearest);
        assert_eq!(cfg.wrap_u, WrapMode::Repeat);
    }
}
