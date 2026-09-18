#![allow(dead_code)]
//! Still image store for live production graphics and stills.
//!
//! This module provides a managed store of still images that can be loaded
//! onto the switcher buses. Stills can be used for graphics overlays,
//! lower thirds backgrounds, test patterns, or emergency backup frames.

use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

/// Pixel format for stored stills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StillPixelFormat {
    /// 8-bit RGBA (4 bytes per pixel).
    Rgba8,
    /// 8-bit BGRA (4 bytes per pixel).
    Bgra8,
    /// 10-bit YCbCr 4:2:2 packed.
    Ycbcr422_10bit,
    /// 8-bit YCbCr 4:2:2 packed.
    Ycbcr422_8bit,
}

impl StillPixelFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            Self::Rgba8 | Self::Bgra8 => 4,
            Self::Ycbcr422_10bit => 3,
            Self::Ycbcr422_8bit => 2,
        }
    }
}

impl fmt::Display for StillPixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rgba8 => write!(f, "RGBA 8-bit"),
            Self::Bgra8 => write!(f, "BGRA 8-bit"),
            Self::Ycbcr422_10bit => write!(f, "YCbCr 4:2:2 10-bit"),
            Self::Ycbcr422_8bit => write!(f, "YCbCr 4:2:2 8-bit"),
        }
    }
}

/// Resolution of a still image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StillResolution {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl StillResolution {
    /// Create a new resolution.
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Standard 1080p resolution.
    pub fn hd_1080() -> Self {
        Self::new(1920, 1080)
    }

    /// Standard 720p resolution.
    pub fn hd_720() -> Self {
        Self::new(1280, 720)
    }

    /// Standard 4K UHD resolution.
    pub fn uhd_4k() -> Self {
        Self::new(3840, 2160)
    }

    /// Total number of pixels.
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Compute required buffer size for a given pixel format.
    pub fn buffer_size(&self, format: StillPixelFormat) -> usize {
        self.pixel_count() as usize * format.bytes_per_pixel()
    }
}

impl fmt::Display for StillResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Metadata for a stored still image.
#[derive(Debug, Clone)]
pub struct StillMetadata {
    /// User-assigned name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Resolution.
    pub resolution: StillResolution,
    /// Pixel format.
    pub format: StillPixelFormat,
    /// When the still was loaded.
    pub loaded_at: SystemTime,
    /// Whether the still has an alpha channel.
    pub has_alpha: bool,
    /// Whether the still is marked as a favorite.
    pub is_favorite: bool,
}

/// A single still image entry in the store.
#[derive(Debug, Clone)]
pub struct StillEntry {
    /// Unique slot ID.
    pub slot_id: usize,
    /// Metadata.
    pub metadata: StillMetadata,
    /// Pixel data (raw bytes).
    pub data: Vec<u8>,
}

impl StillEntry {
    /// Get the data size in bytes.
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Check if the data size matches the expected size for the resolution and format.
    pub fn is_valid(&self) -> bool {
        let expected = self.metadata.resolution.buffer_size(self.metadata.format);
        self.data.len() == expected
    }
}

/// Configuration for the still store.
#[derive(Debug, Clone)]
pub struct StillStoreConfig {
    /// Maximum number of stills.
    pub max_stills: usize,
    /// Maximum total memory in bytes.
    pub max_memory_bytes: usize,
    /// Default pixel format for new stills.
    pub default_format: StillPixelFormat,
    /// Default resolution for new stills.
    pub default_resolution: StillResolution,
}

impl Default for StillStoreConfig {
    fn default() -> Self {
        Self {
            max_stills: 32,
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            default_format: StillPixelFormat::Rgba8,
            default_resolution: StillResolution::hd_1080(),
        }
    }
}

/// Error type for still store operations.
#[derive(Debug, Clone)]
pub enum StillStoreError {
    /// Store is full (no free slots).
    StoreFull,
    /// Slot not found.
    SlotNotFound(usize),
    /// Data size mismatch.
    DataSizeMismatch {
        /// Expected data size.
        expected: usize,
        /// Actual data size.
        actual: usize,
    },
    /// Memory limit exceeded.
    MemoryLimitExceeded {
        /// Current usage in bytes.
        current: usize,
        /// Requested addition in bytes.
        requested: usize,
        /// Maximum allowed in bytes.
        limit: usize,
    },
    /// Invalid resolution.
    InvalidResolution(String),
}

impl fmt::Display for StillStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StoreFull => write!(f, "Still store is full"),
            Self::SlotNotFound(id) => write!(f, "Slot {id} not found"),
            Self::DataSizeMismatch { expected, actual } => {
                write!(f, "Data size mismatch: expected {expected}, got {actual}")
            }
            Self::MemoryLimitExceeded {
                current,
                requested,
                limit,
            } => {
                write!(
                    f,
                    "Memory limit exceeded: current={current}, requested={requested}, limit={limit}"
                )
            }
            Self::InvalidResolution(msg) => write!(f, "Invalid resolution: {msg}"),
        }
    }
}

/// Still image store for a live production switcher.
#[derive(Debug)]
pub struct StillStore {
    /// Configuration.
    config: StillStoreConfig,
    /// Stored stills indexed by slot ID.
    stills: HashMap<usize, StillEntry>,
    /// Next slot ID.
    next_slot: usize,
    /// Current memory usage in bytes.
    memory_used: usize,
}

impl StillStore {
    /// Create a new still store with the given configuration.
    pub fn new(config: StillStoreConfig) -> Self {
        Self {
            config,
            stills: HashMap::new(),
            next_slot: 0,
            memory_used: 0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(StillStoreConfig::default())
    }

    /// Load a still image into the store.
    pub fn load(
        &mut self,
        name: &str,
        resolution: StillResolution,
        format: StillPixelFormat,
        data: Vec<u8>,
    ) -> Result<usize, StillStoreError> {
        if self.stills.len() >= self.config.max_stills {
            return Err(StillStoreError::StoreFull);
        }

        let expected_size = resolution.buffer_size(format);
        if data.len() != expected_size {
            return Err(StillStoreError::DataSizeMismatch {
                expected: expected_size,
                actual: data.len(),
            });
        }

        if self.memory_used + data.len() > self.config.max_memory_bytes {
            return Err(StillStoreError::MemoryLimitExceeded {
                current: self.memory_used,
                requested: data.len(),
                limit: self.config.max_memory_bytes,
            });
        }

        let slot_id = self.next_slot;
        self.next_slot += 1;

        let has_alpha = matches!(format, StillPixelFormat::Rgba8 | StillPixelFormat::Bgra8);
        let entry = StillEntry {
            slot_id,
            metadata: StillMetadata {
                name: name.to_string(),
                description: None,
                resolution,
                format,
                loaded_at: SystemTime::now(),
                has_alpha,
                is_favorite: false,
            },
            data,
        };

        self.memory_used += entry.data_size();
        self.stills.insert(slot_id, entry);
        Ok(slot_id)
    }

    /// Remove a still from the store.
    pub fn remove(&mut self, slot_id: usize) -> Result<(), StillStoreError> {
        if let Some(entry) = self.stills.remove(&slot_id) {
            self.memory_used = self.memory_used.saturating_sub(entry.data_size());
            Ok(())
        } else {
            Err(StillStoreError::SlotNotFound(slot_id))
        }
    }

    /// Get a reference to a still entry.
    pub fn get(&self, slot_id: usize) -> Option<&StillEntry> {
        self.stills.get(&slot_id)
    }

    /// Get a mutable reference to a still entry.
    pub fn get_mut(&mut self, slot_id: usize) -> Option<&mut StillEntry> {
        self.stills.get_mut(&slot_id)
    }

    /// Set a still as favorite.
    pub fn set_favorite(&mut self, slot_id: usize, favorite: bool) -> Result<(), StillStoreError> {
        if let Some(entry) = self.stills.get_mut(&slot_id) {
            entry.metadata.is_favorite = favorite;
            Ok(())
        } else {
            Err(StillStoreError::SlotNotFound(slot_id))
        }
    }

    /// Get the number of stills stored.
    pub fn count(&self) -> usize {
        self.stills.len()
    }

    /// Get current memory usage in bytes.
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// Get the maximum number of stills.
    pub fn max_stills(&self) -> usize {
        self.config.max_stills
    }

    /// Get available memory in bytes.
    pub fn memory_available(&self) -> usize {
        self.config
            .max_memory_bytes
            .saturating_sub(self.memory_used)
    }

    /// List all slot IDs.
    pub fn slot_ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.stills.keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Clear all stills.
    pub fn clear(&mut self) {
        self.stills.clear();
        self.memory_used = 0;
    }

    /// Get the configuration.
    pub fn config(&self) -> &StillStoreConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data(resolution: StillResolution, format: StillPixelFormat) -> Vec<u8> {
        vec![128u8; resolution.buffer_size(format)]
    }

    #[test]
    fn test_pixel_format_bytes() {
        assert_eq!(StillPixelFormat::Rgba8.bytes_per_pixel(), 4);
        assert_eq!(StillPixelFormat::Bgra8.bytes_per_pixel(), 4);
        assert_eq!(StillPixelFormat::Ycbcr422_10bit.bytes_per_pixel(), 3);
        assert_eq!(StillPixelFormat::Ycbcr422_8bit.bytes_per_pixel(), 2);
    }

    #[test]
    fn test_pixel_format_display() {
        assert_eq!(format!("{}", StillPixelFormat::Rgba8), "RGBA 8-bit");
    }

    #[test]
    fn test_resolution_basics() {
        let r = StillResolution::hd_1080();
        assert_eq!(r.width, 1920);
        assert_eq!(r.height, 1080);
        assert_eq!(r.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_resolution_buffer_size() {
        let r = StillResolution::new(100, 100);
        assert_eq!(r.buffer_size(StillPixelFormat::Rgba8), 100 * 100 * 4);
        assert_eq!(
            r.buffer_size(StillPixelFormat::Ycbcr422_8bit),
            100 * 100 * 2
        );
    }

    #[test]
    fn test_resolution_display() {
        let r = StillResolution::new(1920, 1080);
        assert_eq!(format!("{r}"), "1920x1080");
    }

    #[test]
    fn test_store_creation() {
        let store = StillStore::with_defaults();
        assert_eq!(store.count(), 0);
        assert_eq!(store.memory_used(), 0);
        assert_eq!(store.max_stills(), 32);
    }

    #[test]
    fn test_store_load_still() {
        let mut store = StillStore::with_defaults();
        let res = StillResolution::new(16, 16);
        let data = make_test_data(res, StillPixelFormat::Rgba8);

        let slot = store
            .load("test", res, StillPixelFormat::Rgba8, data.clone())
            .expect("should succeed in test");
        assert_eq!(store.count(), 1);
        assert_eq!(store.memory_used(), data.len());

        let entry = store.get(slot).expect("should succeed in test");
        assert_eq!(entry.metadata.name, "test");
        assert!(entry.is_valid());
    }

    #[test]
    fn test_store_remove_still() {
        let mut store = StillStore::with_defaults();
        let res = StillResolution::new(8, 8);
        let data = make_test_data(res, StillPixelFormat::Rgba8);

        let slot = store
            .load("rem", res, StillPixelFormat::Rgba8, data)
            .expect("should succeed in test");
        assert_eq!(store.count(), 1);

        store.remove(slot).expect("should succeed in test");
        assert_eq!(store.count(), 0);
        assert_eq!(store.memory_used(), 0);
    }

    #[test]
    fn test_store_data_size_mismatch() {
        let mut store = StillStore::with_defaults();
        let res = StillResolution::new(8, 8);
        let bad_data = vec![0u8; 10]; // wrong size

        let result = store.load("bad", res, StillPixelFormat::Rgba8, bad_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_store_full() {
        let config = StillStoreConfig {
            max_stills: 2,
            max_memory_bytes: 1024 * 1024,
            ..Default::default()
        };
        let mut store = StillStore::new(config);
        let res = StillResolution::new(2, 2);
        let data = make_test_data(res, StillPixelFormat::Rgba8);

        store
            .load("a", res, StillPixelFormat::Rgba8, data.clone())
            .expect("should succeed in test");
        store
            .load("b", res, StillPixelFormat::Rgba8, data.clone())
            .expect("should succeed in test");

        let result = store.load("c", res, StillPixelFormat::Rgba8, data);
        assert!(matches!(result, Err(StillStoreError::StoreFull)));
    }

    #[test]
    fn test_store_memory_limit() {
        let config = StillStoreConfig {
            max_stills: 100,
            max_memory_bytes: 100,
            ..Default::default()
        };
        let mut store = StillStore::new(config);
        let res = StillResolution::new(8, 8);
        let data = make_test_data(res, StillPixelFormat::Rgba8); // 8*8*4 = 256 bytes

        let result = store.load("big", res, StillPixelFormat::Rgba8, data);
        assert!(matches!(
            result,
            Err(StillStoreError::MemoryLimitExceeded { .. })
        ));
    }

    #[test]
    fn test_store_favorite() {
        let mut store = StillStore::with_defaults();
        let res = StillResolution::new(4, 4);
        let data = make_test_data(res, StillPixelFormat::Rgba8);

        let slot = store
            .load("fav", res, StillPixelFormat::Rgba8, data)
            .expect("should succeed in test");
        assert!(
            !store
                .get(slot)
                .expect("should succeed in test")
                .metadata
                .is_favorite
        );

        store
            .set_favorite(slot, true)
            .expect("should succeed in test");
        assert!(
            store
                .get(slot)
                .expect("should succeed in test")
                .metadata
                .is_favorite
        );
    }

    #[test]
    fn test_store_clear() {
        let mut store = StillStore::with_defaults();
        let res = StillResolution::new(2, 2);
        let data = make_test_data(res, StillPixelFormat::Rgba8);

        store
            .load("x", res, StillPixelFormat::Rgba8, data.clone())
            .expect("should succeed in test");
        store
            .load("y", res, StillPixelFormat::Rgba8, data)
            .expect("should succeed in test");
        assert_eq!(store.count(), 2);

        store.clear();
        assert_eq!(store.count(), 0);
        assert_eq!(store.memory_used(), 0);
    }

    #[test]
    fn test_store_slot_ids_sorted() {
        let mut store = StillStore::with_defaults();
        let res = StillResolution::new(2, 2);
        let data = make_test_data(res, StillPixelFormat::Rgba8);

        let s0 = store
            .load("a", res, StillPixelFormat::Rgba8, data.clone())
            .expect("should succeed in test");
        let s1 = store
            .load("b", res, StillPixelFormat::Rgba8, data)
            .expect("should succeed in test");

        let ids = store.slot_ids();
        assert_eq!(ids, vec![s0, s1]);
    }
}
