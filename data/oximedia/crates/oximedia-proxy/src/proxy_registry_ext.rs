//! Extended proxy registry with best-match selection.
//!
//! [`ProxyRegistryExt`] maintains an in-memory list of [`ProxyRecord`]s —
//! lightweight descriptors that associate a source asset with one or more
//! proxy variants — and provides [`ProxyRegistryExt::find_best_proxy`] to
//! return the variant most appropriate for a given resolution target.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Resolution expressed as `(width, height)` in pixels.
pub type Resolution = (u32, u32);

/// Describes a single proxy variant stored in the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyVariantRecord {
    /// Path to the proxy file on disk.
    pub path: PathBuf,
    /// Pixel dimensions of the proxy.
    pub resolution: Resolution,
    /// Codec label (e.g. `"h264"`, `"prores"`).
    pub codec: String,
    /// Video bitrate in kbps.
    pub video_kbps: u32,
}

impl ProxyVariantRecord {
    /// Create a new variant record.
    pub fn new(
        path: impl Into<PathBuf>,
        resolution: Resolution,
        codec: impl Into<String>,
        video_kbps: u32,
    ) -> Self {
        Self {
            path: path.into(),
            resolution,
            codec: codec.into(),
            video_kbps,
        }
    }

    /// Number of pixels in the frame.
    pub fn pixel_count(&self) -> u64 {
        self.resolution.0 as u64 * self.resolution.1 as u64
    }
}

/// Associates a source asset with its registered proxy variants.
#[derive(Debug, Clone)]
pub struct ProxyRecord {
    /// Canonical path to the original (high-resolution) source file.
    pub source_path: PathBuf,
    /// All known proxy variants for this source.
    pub variants: Vec<ProxyVariantRecord>,
}

impl ProxyRecord {
    /// Create an empty record for `source_path`.
    pub fn new(source_path: impl Into<PathBuf>) -> Self {
        Self {
            source_path: source_path.into(),
            variants: Vec::new(),
        }
    }

    /// Add a proxy variant to this record.
    pub fn add_variant(&mut self, variant: ProxyVariantRecord) {
        self.variants.push(variant);
    }

    /// Return `true` if at least one variant has been registered.
    pub fn has_proxies(&self) -> bool {
        !self.variants.is_empty()
    }

    /// Find the variant whose pixel count is closest to `target_resolution`.
    ///
    /// Among variants with equal pixel distance, the one with the higher bitrate
    /// is preferred.
    pub fn best_variant_for(&self, target: Resolution) -> Option<&ProxyVariantRecord> {
        if self.variants.is_empty() {
            return None;
        }
        let target_px = target.0 as i64 * target.1 as i64;
        self.variants.iter().min_by(|a, b| {
            let da = (a.pixel_count() as i64 - target_px).unsigned_abs();
            let db = (b.pixel_count() as i64 - target_px).unsigned_abs();
            da.cmp(&db).then_with(|| b.video_kbps.cmp(&a.video_kbps))
        })
    }
}

/// In-memory registry of [`ProxyRecord`]s indexed by source path.
#[derive(Debug, Default)]
pub struct ProxyRegistryExt {
    records: HashMap<PathBuf, ProxyRecord>,
}

impl ProxyRegistryExt {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register or replace the record for `source_path`.
    pub fn insert(&mut self, record: ProxyRecord) {
        self.records.insert(record.source_path.clone(), record);
    }

    /// Add a single proxy variant for `source_path`.
    ///
    /// Creates a new [`ProxyRecord`] if one does not already exist.
    pub fn add_variant(&mut self, source_path: impl Into<PathBuf>, variant: ProxyVariantRecord) {
        let path = source_path.into();
        self.records
            .entry(path.clone())
            .or_insert_with(|| ProxyRecord::new(path))
            .add_variant(variant);
    }

    /// Return the record for `source_path`, if present.
    pub fn get(&self, source_path: &Path) -> Option<&ProxyRecord> {
        self.records.get(source_path)
    }

    /// Remove the record for `source_path`, returning it if it existed.
    pub fn remove(&mut self, source_path: &Path) -> Option<ProxyRecord> {
        self.records.remove(source_path)
    }

    /// Return the number of source assets registered.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return `true` if no records have been registered.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Find the best proxy variant for `source_path` at `target_resolution`.
    ///
    /// Returns `None` if the source is not registered or has no variants.
    pub fn find_best_proxy(
        &self,
        source_path: &Path,
        target_resolution: Resolution,
    ) -> Option<&ProxyVariantRecord> {
        self.records
            .get(source_path)?
            .best_variant_for(target_resolution)
    }

    /// Return all source paths that have at least one proxy registered.
    pub fn sources_with_proxies(&self) -> Vec<&Path> {
        self.records
            .values()
            .filter(|r| r.has_proxies())
            .map(|r| r.source_path.as_path())
            .collect()
    }

    /// Total number of proxy variants across all records.
    pub fn total_variants(&self) -> usize {
        self.records.values().map(|r| r.variants.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_variant(path: &str, w: u32, h: u32, codec: &str, kbps: u32) -> ProxyVariantRecord {
        ProxyVariantRecord::new(path, (w, h), codec, kbps)
    }

    #[test]
    fn variant_record_pixel_count() {
        let v = make_variant("p.mp4", 1920, 1080, "h264", 8000);
        assert_eq!(v.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn proxy_record_add_variant() {
        let mut rec = ProxyRecord::new("/src/clip.mov");
        assert!(!rec.has_proxies());
        rec.add_variant(make_variant("/proxy/clip_360.mp4", 640, 360, "h264", 500));
        assert!(rec.has_proxies());
        assert_eq!(rec.variants.len(), 1);
    }

    #[test]
    fn proxy_record_best_variant_exact_match() {
        let mut rec = ProxyRecord::new("/src/clip.mov");
        rec.add_variant(make_variant("/proxy/360p.mp4", 640, 360, "h264", 500));
        rec.add_variant(make_variant("/proxy/720p.mp4", 1280, 720, "h264", 2000));
        let best = rec
            .best_variant_for((1280, 720))
            .expect("should succeed in test");
        assert_eq!(best.resolution, (1280, 720));
    }

    #[test]
    fn proxy_record_best_variant_closest_pixel_count() {
        let mut rec = ProxyRecord::new("/src/clip.mov");
        rec.add_variant(make_variant("/proxy/360p.mp4", 640, 360, "h264", 500));
        rec.add_variant(make_variant("/proxy/1080p.mp4", 1920, 1080, "h264", 8000));
        // Target is 720p — 1280×720 = 921 600. 360p is 230 400, 1080p is 2 073 600.
        // Delta 360p = |230400 - 921600| = 691200
        // Delta 1080p = |2073600 - 921600| = 1152000
        // 360p is closer
        let best = rec
            .best_variant_for((1280, 720))
            .expect("should succeed in test");
        assert_eq!(best.resolution, (640, 360));
    }

    #[test]
    fn proxy_record_best_variant_empty_returns_none() {
        let rec = ProxyRecord::new("/src/clip.mov");
        assert!(rec.best_variant_for((1280, 720)).is_none());
    }

    #[test]
    fn registry_ext_insert_and_get() {
        let mut reg = ProxyRegistryExt::new();
        let rec = ProxyRecord::new("/src/a.mov");
        reg.insert(rec);
        assert!(reg.get(Path::new("/src/a.mov")).is_some());
    }

    #[test]
    fn registry_ext_add_variant_creates_record() {
        let mut reg = ProxyRegistryExt::new();
        reg.add_variant(
            "/src/b.mov",
            make_variant("/p/b_360.mp4", 640, 360, "h264", 500),
        );
        assert!(reg.get(Path::new("/src/b.mov")).is_some());
    }

    #[test]
    fn registry_ext_remove() {
        let mut reg = ProxyRegistryExt::new();
        reg.insert(ProxyRecord::new("/src/c.mov"));
        let removed = reg.remove(Path::new("/src/c.mov"));
        assert!(removed.is_some());
        assert!(reg.get(Path::new("/src/c.mov")).is_none());
    }

    #[test]
    fn registry_ext_len_and_empty() {
        let mut reg = ProxyRegistryExt::new();
        assert!(reg.is_empty());
        reg.insert(ProxyRecord::new("/src/d.mov"));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn registry_ext_find_best_proxy() {
        let mut reg = ProxyRegistryExt::new();
        reg.add_variant(
            "/src/e.mov",
            make_variant("/p/e_360.mp4", 640, 360, "h264", 500),
        );
        reg.add_variant(
            "/src/e.mov",
            make_variant("/p/e_1080.mp4", 1920, 1080, "h264", 8000),
        );
        let best = reg.find_best_proxy(Path::new("/src/e.mov"), (1920, 1080));
        assert!(best.is_some());
        assert_eq!(
            best.expect("should succeed in test").resolution,
            (1920, 1080)
        );
    }

    #[test]
    fn registry_ext_find_best_proxy_not_found() {
        let reg = ProxyRegistryExt::new();
        assert!(reg
            .find_best_proxy(Path::new("/nonexistent.mov"), (1280, 720))
            .is_none());
    }

    #[test]
    fn registry_ext_sources_with_proxies() {
        let mut reg = ProxyRegistryExt::new();
        reg.add_variant(
            "/src/f.mov",
            make_variant("/p/f.mp4", 640, 360, "h264", 500),
        );
        reg.insert(ProxyRecord::new("/src/g.mov")); // no variants
        let sources = reg.sources_with_proxies();
        assert_eq!(sources.len(), 1);
    }

    #[test]
    fn registry_ext_total_variants() {
        let mut reg = ProxyRegistryExt::new();
        reg.add_variant(
            "/src/h.mov",
            make_variant("/p/h1.mp4", 640, 360, "h264", 500),
        );
        reg.add_variant(
            "/src/h.mov",
            make_variant("/p/h2.mp4", 1280, 720, "h264", 2000),
        );
        reg.add_variant(
            "/src/i.mov",
            make_variant("/p/i1.mp4", 640, 360, "h264", 500),
        );
        assert_eq!(reg.total_variants(), 3);
    }

    #[test]
    fn registry_ext_prefer_higher_bitrate_on_tie() {
        let mut rec = ProxyRecord::new("/src/j.mov");
        // Two variants at exactly the same resolution
        rec.add_variant(make_variant("/p/j_low.mp4", 1280, 720, "h264", 1000));
        rec.add_variant(make_variant("/p/j_high.mp4", 1280, 720, "h264", 5000));
        let best = rec
            .best_variant_for((1280, 720))
            .expect("should succeed in test");
        assert_eq!(best.video_kbps, 5000);
    }
}
