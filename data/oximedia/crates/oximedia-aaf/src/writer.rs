//! AAF Writer
//!
//! This module implements AAF file writing functionality.
//! Note: This is a complex implementation and currently provides basic writing capabilities.
//! Full AAF writing requires extensive object model serialization and structured storage writing.

use crate::composition::{CompositionMob, Sequence, Track};
use crate::dictionary::{Auid, Dictionary};
use crate::object_model::{Header, Mob};
use crate::structured_storage::StorageWriter;
use crate::{AafFile, ContentStorage, Result};
use std::fs::File;
use std::io::{Seek, Write};
use std::path::Path;
use uuid::Uuid;

/// AAF Writer
pub struct AafWriter<W: Write + Seek> {
    storage: StorageWriter<W>,
    header: Header,
    dictionary: Dictionary,
    content_storage: ContentStorage,
}

impl<W: Write + Seek> AafWriter<W> {
    /// Create a new AAF writer
    pub fn new(writer: W) -> Result<Self> {
        let storage = StorageWriter::new(writer)?;
        Ok(Self {
            storage,
            header: Header::new(),
            dictionary: Dictionary::new(),
            content_storage: ContentStorage::new(),
        })
    }

    /// Set the header
    pub fn set_header(&mut self, header: Header) {
        self.header = header;
    }

    /// Get mutable reference to content storage
    pub fn content_storage_mut(&mut self) -> &mut ContentStorage {
        &mut self.content_storage
    }

    /// Add a composition mob
    pub fn add_composition_mob(&mut self, comp_mob: CompositionMob) {
        self.content_storage.add_composition_mob(comp_mob);
    }

    /// Add a mob
    pub fn add_mob(&mut self, mob: Mob) {
        self.content_storage.add_mob(mob);
    }

    /// Write the AAF file
    pub fn write(&mut self) -> Result<()> {
        // Write header stream
        self.write_header()?;

        // Write dictionary
        self.write_dictionary()?;

        // Write content storage
        self.write_content_storage()?;

        // Finalize the structured storage
        self.storage.finalize()?;

        Ok(())
    }

    /// Write header to storage
    fn write_header(&mut self) -> Result<()> {
        // Serialize header to bytes
        let header_data = self.serialize_header()?;

        // Write to "Header" stream in root storage
        self.storage.write_stream("Header", &header_data)?;

        Ok(())
    }

    /// Write dictionary to storage
    fn write_dictionary(&mut self) -> Result<()> {
        // Serialize dictionary to bytes
        let dict_data = self.serialize_dictionary()?;

        // Write to "MetaDictionary" stream
        self.storage.write_stream("MetaDictionary", &dict_data)?;

        Ok(())
    }

    /// Write content storage to AAF file
    fn write_content_storage(&mut self) -> Result<()> {
        // Serialize content storage to bytes
        let content_data = self.serialize_content_storage()?;

        // Write to "ContentStorage" stream
        self.storage.write_stream("ContentStorage", &content_data)?;

        Ok(())
    }

    /// Serialize header to bytes
    fn serialize_header(&self) -> Result<Vec<u8>> {
        // This is a simplified serialization
        // Real AAF uses a complex object model serialization format
        let mut data = Vec::new();

        // Write version
        data.extend_from_slice(&self.header.major_version.to_le_bytes());
        data.extend_from_slice(&self.header.minor_version.to_le_bytes());

        // Write byte order
        data.extend_from_slice(&self.header.byte_order.to_le_bytes());

        // Write timestamp
        data.extend_from_slice(&self.header.last_modified.to_le_bytes());

        // Write object model version
        data.extend_from_slice(&self.header.object_model_version.to_le_bytes());

        // Write operational pattern
        data.extend_from_slice(self.header.operational_pattern.as_bytes());

        Ok(data)
    }

    /// Serialize dictionary to bytes
    fn serialize_dictionary(&self) -> Result<Vec<u8>> {
        // Simplified dictionary serialization
        let mut data = Vec::new();

        // Write marker
        data.extend_from_slice(b"AAF_DICT");

        // In a real implementation, we would serialize all class definitions,
        // property definitions, and type definitions

        Ok(data)
    }

    /// Serialize content storage to bytes
    fn serialize_content_storage(&self) -> Result<Vec<u8>> {
        // Simplified content storage serialization
        let mut data = Vec::new();

        // Write marker
        data.extend_from_slice(b"AAF_CONTENT");

        // Write composition mob count
        let comp_mob_count = self.content_storage.composition_mobs().len() as u32;
        data.extend_from_slice(&comp_mob_count.to_le_bytes());

        // Write mob count
        let mob_count = (self.content_storage.master_mobs().len()
            + self.content_storage.source_mobs().len()) as u32;
        data.extend_from_slice(&mob_count.to_le_bytes());

        // In a real implementation, we would serialize all mobs and their slots

        Ok(data)
    }
}

impl AafWriter<File> {
    /// Create an AAF writer for a file path
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        Self::new(file)
    }
}

/// AAF builder for constructing AAF files programmatically
pub struct AafBuilder {
    header: Header,
    dictionary: Dictionary,
    content_storage: ContentStorage,
}

impl AafBuilder {
    /// Create a new AAF builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            header: Header::new(),
            dictionary: Dictionary::new(),
            content_storage: ContentStorage::new(),
        }
    }

    /// Set header
    #[must_use]
    pub fn with_header(mut self, header: Header) -> Self {
        self.header = header;
        self
    }

    /// Add a composition mob
    #[must_use]
    pub fn add_composition_mob(mut self, comp_mob: CompositionMob) -> Self {
        self.content_storage.add_composition_mob(comp_mob);
        self
    }

    /// Add a mob
    #[must_use]
    pub fn add_mob(mut self, mob: Mob) -> Self {
        self.content_storage.add_mob(mob);
        self
    }

    /// Build the AAF file
    #[must_use]
    pub fn build(self) -> AafFile {
        AafFile {
            header: self.header,
            dictionary: self.dictionary,
            content_storage: self.content_storage,
            essence_data: Vec::new(),
        }
    }

    /// Build and write to a file
    pub fn write_to_file<P: AsRef<Path>>(self, path: P) -> Result<()> {
        let aaf_file = self.build();
        let mut writer = AafWriter::create(path)?;

        writer.set_header(aaf_file.header);
        *writer.content_storage_mut() = aaf_file.content_storage;

        writer.write()?;
        Ok(())
    }
}

impl Default for AafBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple composition builder
pub struct CompositionBuilder {
    mob_id: Uuid,
    name: String,
    tracks: Vec<Track>,
}

impl CompositionBuilder {
    /// Create a new composition builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            mob_id: Uuid::new_v4(),
            name: name.into(),
            tracks: Vec::new(),
        }
    }

    /// Set mob ID
    #[must_use]
    pub fn with_mob_id(mut self, mob_id: Uuid) -> Self {
        self.mob_id = mob_id;
        self
    }

    /// Add a track
    #[must_use]
    pub fn add_track(mut self, track: Track) -> Self {
        self.tracks.push(track);
        self
    }

    /// Build the composition mob
    #[must_use]
    pub fn build(self) -> CompositionMob {
        let mut comp_mob = CompositionMob::new(self.mob_id, self.name);
        for track in self.tracks {
            comp_mob.add_track(track);
        }
        comp_mob
    }
}

/// Track builder
pub struct TrackBuilder {
    track_id: u32,
    name: String,
    edit_rate: crate::timeline::EditRate,
    track_type: crate::composition::TrackType,
    sequence: Option<Sequence>,
}

impl TrackBuilder {
    /// Create a new track builder
    pub fn new(
        track_id: u32,
        name: impl Into<String>,
        edit_rate: crate::timeline::EditRate,
        track_type: crate::composition::TrackType,
    ) -> Self {
        Self {
            track_id,
            name: name.into(),
            edit_rate,
            track_type,
            sequence: None,
        }
    }

    /// Set sequence
    #[must_use]
    pub fn with_sequence(mut self, sequence: Sequence) -> Self {
        self.sequence = Some(sequence);
        self
    }

    /// Build the track
    #[must_use]
    pub fn build(self) -> Track {
        let mut track = Track::new(self.track_id, self.name, self.edit_rate, self.track_type);
        if let Some(sequence) = self.sequence {
            track.set_sequence(sequence);
        }
        track
    }
}

/// Sequence builder
pub struct SequenceBuilder {
    data_definition: Auid,
    components: Vec<crate::composition::SequenceComponent>,
}

impl SequenceBuilder {
    /// Create a new sequence builder
    #[must_use]
    pub fn new(data_definition: Auid) -> Self {
        Self {
            data_definition,
            components: Vec::new(),
        }
    }

    /// Create a picture sequence builder
    #[must_use]
    pub fn picture() -> Self {
        Self::new(Auid::PICTURE)
    }

    /// Create a sound sequence builder
    #[must_use]
    pub fn sound() -> Self {
        Self::new(Auid::SOUND)
    }

    /// Add a component
    #[must_use]
    pub fn add_component(mut self, component: crate::composition::SequenceComponent) -> Self {
        self.components.push(component);
        self
    }

    /// Build the sequence
    #[must_use]
    pub fn build(self) -> Sequence {
        let mut sequence = Sequence::new(self.data_definition);
        for component in self.components {
            sequence.add_component(component);
        }
        sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{SequenceComponent, SourceClip, TrackType};
    use crate::timeline::{EditRate, Position};

    #[test]
    fn test_aaf_builder() {
        let builder = AafBuilder::new();
        let aaf_file = builder.build();

        assert!(aaf_file.composition_mobs().is_empty());
    }

    #[test]
    fn test_composition_builder() {
        let comp = CompositionBuilder::new("Test Composition").build();

        assert_eq!(comp.name(), "Test Composition");
    }

    #[test]
    fn test_track_builder() {
        let track = TrackBuilder::new(1, "Video", EditRate::PAL_25, TrackType::Picture).build();

        assert_eq!(track.track_id, 1);
        assert_eq!(track.name, "Video");
    }

    #[test]
    fn test_sequence_builder() {
        let sequence = SequenceBuilder::picture()
            .add_component(SequenceComponent::SourceClip(SourceClip::new(
                100,
                Position::zero(),
                Uuid::new_v4(),
                1,
            )))
            .build();

        assert!(sequence.is_picture());
        assert_eq!(sequence.duration(), Some(100));
    }

    #[test]
    fn test_full_composition_build() {
        let clip = SourceClip::new(100, Position::zero(), Uuid::new_v4(), 1);

        let sequence = SequenceBuilder::picture()
            .add_component(SequenceComponent::SourceClip(clip))
            .build();

        let track = TrackBuilder::new(1, "Video", EditRate::PAL_25, TrackType::Picture)
            .with_sequence(sequence)
            .build();

        let comp = CompositionBuilder::new("My Edit").add_track(track).build();

        assert_eq!(comp.name(), "My Edit");
        assert_eq!(comp.tracks().len(), 1);
    }
}
