//! Media pool (still store) management for video switchers.
//!
//! Stores still frames, graphics, logos, and other static content for quick recall.

use oximedia_codec::VideoFrame;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur with media pool operations.
#[derive(Error, Debug, Clone)]
pub enum MediaPoolError {
    #[error("Slot {0} not found")]
    SlotNotFound(usize),

    #[error("Slot {0} is empty")]
    SlotEmpty(usize),

    #[error("Invalid slot ID: {0}")]
    InvalidSlotId(usize),

    #[error("Pool is full (capacity: {0})")]
    PoolFull(usize),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Format error: {0}")]
    FormatError(String),
}

/// Media pool slot entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaSlot {
    /// Slot ID
    pub id: usize,
    /// Slot name/label
    pub name: String,
    /// File path (if loaded from file)
    pub file_path: Option<PathBuf>,
    /// Whether the slot contains a frame
    pub occupied: bool,
    /// Frame dimensions (width, height)
    pub dimensions: Option<(u32, u32)>,
    /// Timestamp of when loaded
    pub loaded_timestamp: Option<u64>,
}

impl MediaSlot {
    /// Create a new empty slot.
    pub fn new(id: usize) -> Self {
        Self {
            id,
            name: format!("Slot {id}"),
            file_path: None,
            occupied: false,
            dimensions: None,
            loaded_timestamp: None,
        }
    }

    /// Create a slot with a name.
    pub fn with_name(id: usize, name: String) -> Self {
        Self {
            id,
            name,
            file_path: None,
            occupied: false,
            dimensions: None,
            loaded_timestamp: None,
        }
    }

    /// Check if the slot is occupied.
    pub fn is_occupied(&self) -> bool {
        self.occupied
    }

    /// Mark the slot as occupied.
    pub fn set_occupied(&mut self, occupied: bool) {
        self.occupied = occupied;
    }

    /// Set dimensions.
    pub fn set_dimensions(&mut self, width: u32, height: u32) {
        self.dimensions = Some((width, height));
    }
}

/// Media pool manages still frames.
pub struct MediaPool {
    /// Maximum number of slots
    capacity: usize,
    /// Slots metadata
    slots: HashMap<usize, MediaSlot>,
    /// Stored frames (slot ID -> frame)
    frames: HashMap<usize, VideoFrame>,
}

impl MediaPool {
    /// Create a new media pool.
    pub fn new(capacity: usize) -> Self {
        let mut slots = HashMap::new();
        for i in 0..capacity {
            slots.insert(i, MediaSlot::new(i));
        }

        Self {
            capacity,
            slots,
            frames: HashMap::new(),
        }
    }

    /// Get the pool capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of occupied slots.
    pub fn occupied_count(&self) -> usize {
        self.slots.values().filter(|s| s.is_occupied()).count()
    }

    /// Get the number of available slots.
    pub fn available_count(&self) -> usize {
        self.capacity - self.occupied_count()
    }

    /// Check if a slot exists.
    pub fn has_slot(&self, slot_id: usize) -> bool {
        self.slots.contains_key(&slot_id)
    }

    /// Get a slot's metadata.
    pub fn get_slot(&self, slot_id: usize) -> Result<&MediaSlot, MediaPoolError> {
        self.slots
            .get(&slot_id)
            .ok_or(MediaPoolError::SlotNotFound(slot_id))
    }

    /// Get mutable slot metadata.
    pub fn get_slot_mut(&mut self, slot_id: usize) -> Result<&mut MediaSlot, MediaPoolError> {
        self.slots
            .get_mut(&slot_id)
            .ok_or(MediaPoolError::SlotNotFound(slot_id))
    }

    /// Store a frame in a slot.
    pub fn store_frame(&mut self, slot_id: usize, frame: VideoFrame) -> Result<(), MediaPoolError> {
        if slot_id >= self.capacity {
            return Err(MediaPoolError::InvalidSlotId(slot_id));
        }

        let slot = self.get_slot_mut(slot_id)?;
        slot.set_occupied(true);
        slot.loaded_timestamp = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );

        self.frames.insert(slot_id, frame);
        Ok(())
    }

    /// Get a frame from a slot.
    pub fn get_frame(&self, slot_id: usize) -> Result<&VideoFrame, MediaPoolError> {
        let slot = self.get_slot(slot_id)?;
        if !slot.is_occupied() {
            return Err(MediaPoolError::SlotEmpty(slot_id));
        }

        self.frames
            .get(&slot_id)
            .ok_or(MediaPoolError::SlotEmpty(slot_id))
    }

    /// Clear a slot.
    pub fn clear_slot(&mut self, slot_id: usize) -> Result<(), MediaPoolError> {
        let slot = self.get_slot_mut(slot_id)?;
        slot.set_occupied(false);
        slot.file_path = None;
        slot.dimensions = None;
        slot.loaded_timestamp = None;

        self.frames.remove(&slot_id);
        Ok(())
    }

    /// Clear all slots.
    pub fn clear_all(&mut self) {
        for slot in self.slots.values_mut() {
            slot.set_occupied(false);
            slot.file_path = None;
            slot.dimensions = None;
            slot.loaded_timestamp = None;
        }
        self.frames.clear();
    }

    /// Get all occupied slot IDs.
    pub fn occupied_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .filter(|(_, s)| s.is_occupied())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Get all available slot IDs.
    pub fn available_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .filter(|(_, s)| !s.is_occupied())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Find the next available slot.
    pub fn next_available_slot(&self) -> Option<usize> {
        (0..self.capacity).find(|&id| self.slots.get(&id).is_some_and(|s| !s.is_occupied()))
    }

    /// Set a slot's name.
    pub fn set_slot_name(&mut self, slot_id: usize, name: String) -> Result<(), MediaPoolError> {
        let slot = self.get_slot_mut(slot_id)?;
        slot.name = name;
        Ok(())
    }

    /// Load from file path (metadata only - actual loading would be async).
    pub fn set_file_path(&mut self, slot_id: usize, path: PathBuf) -> Result<(), MediaPoolError> {
        let slot = self.get_slot_mut(slot_id)?;
        slot.file_path = Some(path);
        Ok(())
    }

    /// Get all slots.
    pub fn slots(&self) -> impl Iterator<Item = &MediaSlot> {
        self.slots.values()
    }

    /// Copy a frame from one slot to another.
    pub fn copy_slot(&mut self, from_slot: usize, to_slot: usize) -> Result<(), MediaPoolError> {
        // Get the frame from source slot
        let frame = self.get_frame(from_slot)?.clone();

        // Copy metadata
        let from_slot_meta = self.get_slot(from_slot)?.clone();

        // Store in destination
        self.store_frame(to_slot, frame)?;

        // Copy metadata
        let to_slot_meta = self.get_slot_mut(to_slot)?;
        to_slot_meta.name = format!("{} (copy)", from_slot_meta.name);
        to_slot_meta.file_path = from_slot_meta.file_path;
        to_slot_meta.dimensions = from_slot_meta.dimensions;

        Ok(())
    }
}

/// Media pool preset configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaPoolPreset {
    /// Preset name
    pub name: String,
    /// Slot assignments (slot ID -> file path)
    pub assignments: HashMap<usize, PathBuf>,
}

impl MediaPoolPreset {
    /// Create a new preset.
    pub fn new(name: String) -> Self {
        Self {
            name,
            assignments: HashMap::new(),
        }
    }

    /// Add a slot assignment.
    pub fn add_assignment(&mut self, slot_id: usize, path: PathBuf) {
        self.assignments.insert(slot_id, path);
    }

    /// Remove a slot assignment.
    pub fn remove_assignment(&mut self, slot_id: usize) {
        self.assignments.remove(&slot_id);
    }

    /// Get the number of assignments.
    pub fn count(&self) -> usize {
        self.assignments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_media_slot_creation() {
        let slot = MediaSlot::new(0);
        assert_eq!(slot.id, 0);
        assert!(!slot.is_occupied());
        assert_eq!(slot.name, "Slot 0");
    }

    #[test]
    fn test_media_slot_with_name() {
        let slot = MediaSlot::with_name(1, "Logo".to_string());
        assert_eq!(slot.id, 1);
        assert_eq!(slot.name, "Logo");
    }

    #[test]
    fn test_media_pool_creation() {
        let pool = MediaPool::new(20);
        assert_eq!(pool.capacity(), 20);
        assert_eq!(pool.occupied_count(), 0);
        assert_eq!(pool.available_count(), 20);
    }

    #[test]
    fn test_next_available_slot() {
        let pool = MediaPool::new(5);
        assert_eq!(pool.next_available_slot(), Some(0));
    }

    #[test]
    fn test_occupied_available_slots() {
        let pool = MediaPool::new(5);
        let occupied = pool.occupied_slots();
        let available = pool.available_slots();

        assert_eq!(occupied.len(), 0);
        assert_eq!(available.len(), 5);
    }

    #[test]
    fn test_clear_slot() {
        let mut pool = MediaPool::new(5);
        let slot = pool.get_slot_mut(0).expect("should succeed in test");
        slot.set_occupied(true);

        assert_eq!(pool.occupied_count(), 1);

        pool.clear_slot(0).expect("should succeed in test");
        assert_eq!(pool.occupied_count(), 0);
        assert!(!pool
            .get_slot(0)
            .expect("should succeed in test")
            .is_occupied());
    }

    #[test]
    fn test_clear_all() {
        let mut pool = MediaPool::new(5);
        for i in 0..3 {
            pool.get_slot_mut(i)
                .expect("should succeed in test")
                .set_occupied(true);
        }

        assert_eq!(pool.occupied_count(), 3);

        pool.clear_all();
        assert_eq!(pool.occupied_count(), 0);
    }

    #[test]
    fn test_set_slot_name() {
        let mut pool = MediaPool::new(5);
        pool.set_slot_name(0, "My Logo".to_string())
            .expect("should succeed in test");

        let slot = pool.get_slot(0).expect("should succeed in test");
        assert_eq!(slot.name, "My Logo");
    }

    #[test]
    fn test_set_file_path() {
        let mut pool = MediaPool::new(5);
        let path = PathBuf::from("/path/to/logo.png");

        pool.set_file_path(0, path.clone())
            .expect("should succeed in test");

        let slot = pool.get_slot(0).expect("should succeed in test");
        assert_eq!(slot.file_path, Some(path));
    }

    #[test]
    fn test_slot_not_found() {
        let pool = MediaPool::new(5);
        assert!(pool.get_slot(10).is_err());
    }

    #[test]
    fn test_invalid_slot_id() {
        let pool = MediaPool::new(5);
        // Attempting to store in an invalid slot should fail
        // (Note: store_frame needs a VideoFrame which we can't easily create in tests)
        // So we just test the ID validation through get_slot
        assert!(pool.get_slot(100).is_err());
    }

    #[test]
    fn test_media_pool_preset() {
        let mut preset = MediaPoolPreset::new("Logos".to_string());
        assert_eq!(preset.name, "Logos");
        assert_eq!(preset.count(), 0);

        preset.add_assignment(0, PathBuf::from("/logo1.png"));
        preset.add_assignment(1, PathBuf::from("/logo2.png"));
        assert_eq!(preset.count(), 2);

        preset.remove_assignment(0);
        assert_eq!(preset.count(), 1);
    }

    #[test]
    fn test_has_slot() {
        let pool = MediaPool::new(5);
        assert!(pool.has_slot(0));
        assert!(pool.has_slot(4));
        assert!(!pool.has_slot(5));
        assert!(!pool.has_slot(100));
    }

    #[test]
    fn test_set_dimensions() {
        let mut slot = MediaSlot::new(0);
        assert_eq!(slot.dimensions, None);

        slot.set_dimensions(1920, 1080);
        assert_eq!(slot.dimensions, Some((1920, 1080)));
    }

    #[test]
    fn test_slots_iterator() {
        let pool = MediaPool::new(5);
        let slots: Vec<_> = pool.slots().collect();
        assert_eq!(slots.len(), 5);
    }
}
