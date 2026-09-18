//! Memory-mapped I/O simulation.
//!
//! Provides in-process simulation of memory-mapped file I/O, including
//! region-based slicing, typed reads, page-aligned buffer allocation, and
//! huge page configuration metadata for large-file mappings on Linux.

#![allow(dead_code)]

// ──────────────────────────────────────────────────────────────────────────────
// MmapRegion
// ──────────────────────────────────────────────────────────────────────────────

/// A simulated memory-mapped region backed by a `Vec<u8>`.
#[derive(Debug, Clone)]
pub struct MmapRegion {
    /// Raw bytes of this region.
    pub data: Vec<u8>,
    /// Byte offset of this region within the file.
    pub offset: u64,
    /// Number of bytes in this region.
    pub length: u64,
}

impl MmapRegion {
    /// Create a new region.
    #[must_use]
    pub fn new(data: Vec<u8>, offset: u64) -> Self {
        let length = data.len() as u64;
        Self {
            data,
            offset,
            length,
        }
    }

    /// Return a slice of the region's data starting at `start` with `len` bytes.
    ///
    /// Returns `None` if the range lies outside the region.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn slice(&self, start: u64, len: usize) -> Option<&[u8]> {
        let end = start.checked_add(len as u64)?;
        if end > self.length {
            return None;
        }
        let s = start as usize;
        let e = end as usize;
        Some(&self.data[s..e])
    }

    /// Read a little-endian `u32` from `offset` within this region.
    ///
    /// Returns `None` if fewer than 4 bytes remain.
    #[must_use]
    pub fn read_u32_le(&self, offset: u64) -> Option<u32> {
        let bytes = self.slice(offset, 4)?;
        Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a little-endian `u64` from `offset` within this region.
    ///
    /// Returns `None` if fewer than 8 bytes remain.
    #[must_use]
    pub fn read_u64_le(&self, offset: u64) -> Option<u64> {
        let bytes = self.slice(offset, 8)?;
        Some(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MmapFile
// ──────────────────────────────────────────────────────────────────────────────

/// A simulated memory-mapped file composed of multiple [`MmapRegion`]s.
#[derive(Debug, Default)]
pub struct MmapFile {
    /// Human-readable path (not used for actual file access in simulation).
    pub path: String,
    /// Mapped regions, in the order they were added.
    pub regions: Vec<MmapRegion>,
    /// Running total of all mapped bytes.
    pub total_size: u64,
}

impl MmapFile {
    /// Create a new (empty) simulated mapped file.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            regions: Vec::new(),
            total_size: 0,
        }
    }

    /// Add a new region containing `data` at file `offset`.
    ///
    /// Returns the index of the newly added region.
    pub fn map_region(&mut self, data: Vec<u8>, offset: u64) -> usize {
        let region = MmapRegion::new(data, offset);
        self.total_size += region.length;
        self.regions.push(region);
        self.regions.len() - 1
    }

    /// Retrieve a reference to the region at `idx`, or `None` if out of bounds.
    #[must_use]
    pub fn get_region(&self, idx: usize) -> Option<&MmapRegion> {
        self.regions.get(idx)
    }

    /// Return the total number of mapped bytes across all regions.
    #[must_use]
    pub fn total_mapped_bytes(&self) -> u64 {
        self.total_size
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// PageAlignedBuffer
// ──────────────────────────────────────────────────────────────────────────────

/// A buffer whose logical size is rounded up to a multiple of `page_size`.
#[derive(Debug)]
pub struct PageAlignedBuffer {
    /// Underlying data; length is always a multiple of `page_size`.
    pub data: Vec<u8>,
    /// Page size in bytes.
    pub page_size: usize,
}

impl PageAlignedBuffer {
    /// Allocate a `PageAlignedBuffer` of at least `size` bytes, rounded up
    /// to the next page boundary.  The default page size is 4096 bytes.
    #[must_use]
    pub fn new(size: usize) -> Self {
        const DEFAULT_PAGE_SIZE: usize = 4096;
        Self::with_page_size(size, DEFAULT_PAGE_SIZE)
    }

    /// Allocate with an explicit `page_size`.
    ///
    /// # Panics
    ///
    /// Panics if `page_size` is zero.
    #[must_use]
    pub fn with_page_size(size: usize, page_size: usize) -> Self {
        assert!(page_size > 0, "page_size must be non-zero");
        let pages = size.div_ceil(page_size).max(1);
        let aligned = pages * page_size;
        Self {
            data: vec![0u8; aligned],
            page_size,
        }
    }

    /// Return the aligned length of the buffer (always a multiple of `page_size`).
    #[must_use]
    pub fn aligned_len(&self) -> usize {
        self.data.len()
    }

    /// Return the number of pages occupied.
    #[must_use]
    pub fn page_count(&self) -> usize {
        self.data.len() / self.page_size
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // MmapRegion ──────────────────────────────────────────────────────────────

    #[test]
    fn test_region_new_sets_length() {
        let r = MmapRegion::new(vec![1, 2, 3, 4], 0);
        assert_eq!(r.length, 4);
        assert_eq!(r.offset, 0);
    }

    #[test]
    fn test_region_slice_full() {
        let r = MmapRegion::new(vec![10, 20, 30], 0);
        assert_eq!(r.slice(0, 3), Some([10u8, 20, 30].as_slice()));
    }

    #[test]
    fn test_region_slice_partial() {
        let r = MmapRegion::new(vec![1, 2, 3, 4, 5], 0);
        assert_eq!(r.slice(1, 3), Some([2u8, 3, 4].as_slice()));
    }

    #[test]
    fn test_region_slice_out_of_bounds() {
        let r = MmapRegion::new(vec![0u8; 4], 0);
        assert!(r.slice(3, 2).is_none());
    }

    #[test]
    fn test_region_slice_empty() {
        let r = MmapRegion::new(vec![9u8; 8], 100);
        assert_eq!(r.slice(0, 0), Some([].as_slice()));
    }

    #[test]
    fn test_region_read_u32_le() {
        // 0x01020304 in little-endian = bytes [04, 03, 02, 01]
        let r = MmapRegion::new(vec![0x04, 0x03, 0x02, 0x01], 0);
        assert_eq!(r.read_u32_le(0), Some(0x0102_0304));
    }

    #[test]
    fn test_region_read_u32_le_not_enough_bytes() {
        let r = MmapRegion::new(vec![0, 1, 2], 0);
        assert!(r.read_u32_le(0).is_none());
    }

    #[test]
    fn test_region_read_u64_le() {
        let bytes: Vec<u8> = (0u8..8).collect();
        let r = MmapRegion::new(bytes, 0);
        let expected = u64::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(r.read_u64_le(0), Some(expected));
    }

    #[test]
    fn test_region_read_u64_le_not_enough_bytes() {
        let r = MmapRegion::new(vec![0u8; 7], 0);
        assert!(r.read_u64_le(0).is_none());
    }

    // MmapFile ────────────────────────────────────────────────────────────────

    #[test]
    fn test_mmap_file_map_region_returns_index() {
        let mut f = MmapFile::new("test.bin");
        let idx = f.map_region(vec![1, 2, 3], 0);
        assert_eq!(idx, 0);
        let idx2 = f.map_region(vec![4, 5], 3);
        assert_eq!(idx2, 1);
    }

    #[test]
    fn test_mmap_file_total_mapped_bytes() {
        let mut f = MmapFile::new("x");
        f.map_region(vec![0u8; 100], 0);
        f.map_region(vec![0u8; 200], 100);
        assert_eq!(f.total_mapped_bytes(), 300);
    }

    #[test]
    fn test_mmap_file_get_region_valid() {
        let mut f = MmapFile::new("x");
        f.map_region(vec![42u8; 4], 0);
        let r = f.get_region(0).expect("region 0 must exist");
        assert_eq!(r.data, vec![42u8; 4]);
    }

    #[test]
    fn test_mmap_file_get_region_out_of_bounds() {
        let f = MmapFile::new("x");
        assert!(f.get_region(0).is_none());
    }

    // PageAlignedBuffer ───────────────────────────────────────────────────────

    #[test]
    fn test_page_aligned_buffer_exact_page() {
        let buf = PageAlignedBuffer::new(4096);
        assert_eq!(buf.aligned_len(), 4096);
        assert_eq!(buf.page_count(), 1);
    }

    #[test]
    fn test_page_aligned_buffer_rounds_up() {
        let buf = PageAlignedBuffer::new(1);
        assert_eq!(buf.aligned_len(), 4096);
        assert_eq!(buf.page_count(), 1);
    }

    #[test]
    fn test_page_aligned_buffer_multiple_pages() {
        let buf = PageAlignedBuffer::new(8193);
        assert_eq!(buf.aligned_len(), 4096 * 3);
        assert_eq!(buf.page_count(), 3);
    }

    #[test]
    fn test_page_aligned_buffer_custom_page_size() {
        let buf = PageAlignedBuffer::with_page_size(100, 64);
        assert_eq!(buf.aligned_len(), 128);
        assert_eq!(buf.page_count(), 2);
    }

    #[test]
    fn test_page_aligned_buffer_zero_size() {
        // Zero size should still allocate one page.
        let buf = PageAlignedBuffer::new(0);
        assert_eq!(buf.aligned_len(), 4096);
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// HugePageConfig — huge page support metadata
// ──────────────────────────────────────────────────────────────────────────────

/// Huge page size variants available on Linux.
///
/// On Linux, huge pages can be configured via `madvise(MADV_HUGEPAGE)` or
/// `MAP_HUGETLB`. This enum captures the most common sizes and is used to
/// annotate `MmapFile` regions for large-file optimisations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HugePageSize {
    /// 2 MiB huge pages (most common on x86-64 Linux).
    TwoMib,
    /// 1 GiB huge pages (requires kernel huge-page pool pre-allocation).
    OneGib,
    /// Custom size in bytes (must be a multiple of the system base page size).
    Custom(usize),
}

impl HugePageSize {
    /// Return the size in bytes.
    #[must_use]
    pub fn bytes(self) -> usize {
        match self {
            Self::TwoMib => 2 * 1024 * 1024,
            Self::OneGib => 1024 * 1024 * 1024,
            Self::Custom(n) => n,
        }
    }
}

impl std::fmt::Display for HugePageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TwoMib => write!(f, "2MiB"),
            Self::OneGib => write!(f, "1GiB"),
            Self::Custom(n) => write!(f, "custom-{}B", n),
        }
    }
}

/// Policy for requesting huge pages on a mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HugePagePolicy {
    /// Do not request huge pages (default small-page behaviour).
    #[default]
    Disabled,
    /// Request transparent huge pages via `madvise(MADV_HUGEPAGE)`.
    ///
    /// The kernel may promote pages to huge pages at any point; this is a
    /// best-effort hint only (Linux-specific).
    Transparent,
    /// Require explicit huge pages via `MAP_HUGETLB`.
    ///
    /// The mapping must be backed by pre-allocated huge pages from the kernel
    /// huge-page pool. Falls back to `Disabled` on non-Linux platforms.
    Explicit(HugePageSize),
}

impl HugePagePolicy {
    /// Returns `true` if huge pages are requested in any form.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::Disabled)
    }

    /// Returns the explicit huge page size, if configured.
    #[must_use]
    pub fn explicit_size(&self) -> Option<HugePageSize> {
        match self {
            Self::Explicit(sz) => Some(*sz),
            _ => None,
        }
    }

    /// Return a human-readable description.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::Disabled => "disabled".to_string(),
            Self::Transparent => "transparent (MADV_HUGEPAGE)".to_string(),
            Self::Explicit(sz) => format!("explicit MAP_HUGETLB ({})", sz),
        }
    }
}

/// Minimum file-size threshold (in bytes) above which huge pages are recommended.
///
/// Files smaller than this are generally not worth the overhead of huge page
/// setup; the threshold here corresponds to a single 2 MiB huge page.
pub const HUGE_PAGE_THRESHOLD_BYTES: u64 = 2 * 1024 * 1024;

/// A memory-mapped file region annotated with huge-page configuration.
///
/// On Linux, `MmapRegionHuge` carries metadata that would be passed to
/// `madvise(2)` or `mmap(2)` with `MAP_HUGETLB` when creating the actual
/// OS mapping. In this pure-Rust simulation the data is backed by a `Vec<u8>`.
#[derive(Debug, Clone)]
pub struct MmapRegionHuge {
    /// The base simulated region.
    pub region: MmapRegion,
    /// Huge-page policy for this region.
    pub policy: HugePagePolicy,
    /// Whether the effective mapping is backed by huge pages (simulation flag).
    pub huge_pages_active: bool,
}

impl MmapRegionHuge {
    /// Create a new huge-page-enabled region.
    ///
    /// `huge_pages_active` is set to `true` when `policy` is not `Disabled`
    /// and the region is large enough (`>= HUGE_PAGE_THRESHOLD_BYTES`).
    #[must_use]
    pub fn new(data: Vec<u8>, offset: u64, policy: HugePagePolicy) -> Self {
        let large_enough = data.len() as u64 >= HUGE_PAGE_THRESHOLD_BYTES;
        let huge_pages_active = policy.is_enabled() && large_enough;
        let region = MmapRegion::new(data, offset);
        Self {
            region,
            policy,
            huge_pages_active,
        }
    }

    /// Return the number of huge pages that would be required to back this region.
    ///
    /// Returns `None` if huge pages are not configured or if the region is not
    /// large enough.
    #[must_use]
    pub fn required_huge_pages(&self) -> Option<usize> {
        let sz = match self.policy {
            HugePagePolicy::Explicit(sz) => sz,
            _ => return None,
        };
        let page_bytes = sz.bytes();
        if page_bytes == 0 {
            return None;
        }
        Some(self.region.length.div_ceil(page_bytes as u64) as usize)
    }

    /// Return a slice of the underlying data.
    #[must_use]
    pub fn slice(&self, start: u64, len: usize) -> Option<&[u8]> {
        self.region.slice(start, len)
    }
}

/// A `MmapFile` that supports huge page annotations on individual regions.
#[derive(Debug, Default)]
pub struct MmapFileHuge {
    /// Logical path of the file.
    pub path: String,
    /// Regions with huge-page metadata.
    pub regions: Vec<MmapRegionHuge>,
    /// Running total of bytes.
    pub total_size: u64,
    /// Default policy applied to new regions when the file exceeds the threshold.
    pub default_policy: HugePagePolicy,
}

impl MmapFileHuge {
    /// Create an empty `MmapFileHuge`.
    #[must_use]
    pub fn new(path: impl Into<String>, default_policy: HugePagePolicy) -> Self {
        Self {
            path: path.into(),
            regions: Vec::new(),
            total_size: 0,
            default_policy,
        }
    }

    /// Map a region with the default policy.
    pub fn map_region(&mut self, data: Vec<u8>, offset: u64) -> usize {
        let policy = if data.len() as u64 >= HUGE_PAGE_THRESHOLD_BYTES {
            self.default_policy
        } else {
            HugePagePolicy::Disabled
        };
        self.map_region_with_policy(data, offset, policy)
    }

    /// Map a region with an explicit policy override.
    pub fn map_region_with_policy(
        &mut self,
        data: Vec<u8>,
        offset: u64,
        policy: HugePagePolicy,
    ) -> usize {
        let region = MmapRegionHuge::new(data, offset, policy);
        self.total_size += region.region.length;
        self.regions.push(region);
        self.regions.len() - 1
    }

    /// Count regions that have huge pages active.
    #[must_use]
    pub fn huge_page_region_count(&self) -> usize {
        self.regions.iter().filter(|r| r.huge_pages_active).count()
    }

    /// Total bytes backed by huge pages.
    #[must_use]
    pub fn huge_page_bytes(&self) -> u64 {
        self.regions
            .iter()
            .filter(|r| r.huge_pages_active)
            .map(|r| r.region.length)
            .sum()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests — huge page additions
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod huge_page_tests {
    use super::*;

    #[test]
    fn test_huge_page_size_bytes() {
        assert_eq!(HugePageSize::TwoMib.bytes(), 2 * 1024 * 1024);
        assert_eq!(HugePageSize::OneGib.bytes(), 1024 * 1024 * 1024);
        assert_eq!(HugePageSize::Custom(4096).bytes(), 4096);
    }

    #[test]
    fn test_huge_page_size_display() {
        assert_eq!(HugePageSize::TwoMib.to_string(), "2MiB");
        assert_eq!(HugePageSize::OneGib.to_string(), "1GiB");
        assert_eq!(HugePageSize::Custom(8192).to_string(), "custom-8192B");
    }

    #[test]
    fn test_huge_page_policy_disabled() {
        let p = HugePagePolicy::Disabled;
        assert!(!p.is_enabled());
        assert!(p.explicit_size().is_none());
    }

    #[test]
    fn test_huge_page_policy_transparent() {
        let p = HugePagePolicy::Transparent;
        assert!(p.is_enabled());
        assert!(p.explicit_size().is_none());
        assert!(p.description().contains("MADV_HUGEPAGE"));
    }

    #[test]
    fn test_huge_page_policy_explicit() {
        let p = HugePagePolicy::Explicit(HugePageSize::TwoMib);
        assert!(p.is_enabled());
        assert_eq!(p.explicit_size(), Some(HugePageSize::TwoMib));
        assert!(p.description().contains("MAP_HUGETLB"));
    }

    #[test]
    fn test_mmap_region_huge_small_data_disabled() {
        // Small region: huge_pages_active should be false even with policy enabled
        let data = vec![0u8; 1024]; // only 1 KiB — below threshold
        let region = MmapRegionHuge::new(data, 0, HugePagePolicy::Transparent);
        assert!(!region.huge_pages_active);
    }

    #[test]
    fn test_mmap_region_huge_large_data_transparent() {
        // Large region (2 MiB): huge pages should activate
        let data = vec![0u8; 2 * 1024 * 1024];
        let region = MmapRegionHuge::new(data, 0, HugePagePolicy::Transparent);
        assert!(region.huge_pages_active);
    }

    #[test]
    fn test_mmap_region_huge_required_pages() {
        let data = vec![0u8; 4 * 1024 * 1024]; // 4 MiB = 2 × 2MiB huge pages
        let region = MmapRegionHuge::new(data, 0, HugePagePolicy::Explicit(HugePageSize::TwoMib));
        assert_eq!(region.required_huge_pages(), Some(2));
    }

    #[test]
    fn test_mmap_region_huge_required_pages_none_when_transparent() {
        let data = vec![0u8; 4 * 1024 * 1024];
        let region = MmapRegionHuge::new(data, 0, HugePagePolicy::Transparent);
        assert_eq!(region.required_huge_pages(), None);
    }

    #[test]
    fn test_mmap_region_huge_slice() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let region = MmapRegionHuge::new(data, 0, HugePagePolicy::Disabled);
        assert_eq!(region.slice(2, 3), Some([3u8, 4, 5].as_slice()));
    }

    #[test]
    fn test_mmap_file_huge_map_regions() {
        let mut f = MmapFileHuge::new("big.raw", HugePagePolicy::Transparent);
        // Small region: policy disabled due to size
        let idx0 = f.map_region(vec![0u8; 512], 0);
        // Large region: transparent policy applied
        let large = vec![0u8; 2 * 1024 * 1024];
        let idx1 = f.map_region(large, 512);
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(f.huge_page_region_count(), 1);
    }

    #[test]
    fn test_mmap_file_huge_bytes() {
        let mut f = MmapFileHuge::new("x", HugePagePolicy::Explicit(HugePageSize::TwoMib));
        let large = vec![0u8; 2 * 1024 * 1024];
        f.map_region(large, 0);
        assert_eq!(f.huge_page_bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn test_mmap_file_huge_policy_override() {
        let mut f = MmapFileHuge::new("x", HugePagePolicy::Disabled);
        // Override: force huge pages even on this small region
        f.map_region_with_policy(
            vec![0u8; 16],
            0,
            HugePagePolicy::Explicit(HugePageSize::TwoMib),
        );
        // small: huge_pages_active = false (not big enough)
        assert_eq!(f.huge_page_region_count(), 0);
    }
}
