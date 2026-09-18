//! Metadata tracking for multi-camera production.

pub mod markers;
pub mod track;

pub use markers::{MarkerManager, MarkerType, SyncMarker};
pub use track::{AngleMetadata, MetadataTracker};

use crate::{AngleId, FrameNumber};

/// Metadata entry
#[derive(Debug, Clone)]
pub struct MetadataEntry {
    /// Angle identifier
    pub angle: AngleId,
    /// Frame number
    pub frame: FrameNumber,
    /// Metadata key
    pub key: String,
    /// Metadata value
    pub value: String,
}

impl MetadataEntry {
    /// Create a new metadata entry
    #[must_use]
    pub fn new(angle: AngleId, frame: FrameNumber, key: String, value: String) -> Self {
        Self {
            angle,
            frame,
            key,
            value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_entry_creation() {
        let entry = MetadataEntry::new(0, 100, "camera".to_string(), "Canon EOS R5".to_string());
        assert_eq!(entry.angle, 0);
        assert_eq!(entry.frame, 100);
        assert_eq!(entry.key, "camera");
        assert_eq!(entry.value, "Canon EOS R5");
    }
}
