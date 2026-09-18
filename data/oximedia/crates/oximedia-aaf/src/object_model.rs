//! AAF Object Model
//!
//! This module implements the AAF object model including:
//! - Header
//! - Mobs (Master Mob, Source Mob, Composition Mob)
//! - Mob Slots (Timeline, Event)
//! - Segments and Components
//! - Transitions and Effects
//! - Parameters
//! - Object reference resolution
//! - Property traversal

use crate::dictionary::{Auid, Dictionary, TypeDefinition};
use crate::structured_storage::StorageReader;
use crate::timeline::{EditRate, Position};
use crate::{ContentStorage, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::io::{Cursor, Read, Seek};
use uuid::Uuid;

/// AAF file header
#[derive(Debug, Clone)]
pub struct Header {
    /// File version
    pub major_version: u16,
    pub minor_version: u16,
    /// Byte order (0xFFFE for little-endian)
    pub byte_order: u16,
    /// Last modified timestamp
    pub last_modified: u64,
    /// Object model version
    pub object_model_version: u32,
    /// Operational pattern
    pub operational_pattern: Auid,
    /// Essence containers
    pub essence_containers: Vec<Auid>,
}

impl Header {
    /// Create a new header with default values
    #[must_use]
    pub fn new() -> Self {
        Self {
            major_version: 1,
            minor_version: 1,
            byte_order: 0xFFFE,
            last_modified: 0,
            object_model_version: 1,
            operational_pattern: Auid::null(),
            essence_containers: Vec::new(),
        }
    }

    /// Get the AAF version as a string
    #[must_use]
    pub fn version_string(&self) -> String {
        format!("{}.{}", self.major_version, self.minor_version)
    }
}

impl Default for Header {
    fn default() -> Self {
        Self::new()
    }
}

/// A locator entry storing an additional file-system path for a mob.
///
/// Source mobs may carry multiple locators (e.g. primary path + NFS mirror).
/// The [`crate::relink`] module uses these to update paths when media is moved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MobLocator {
    /// The file-system or URI path for this locator.
    pub path: String,
}

/// AAF Mob (Master, Source, or Composition)
#[derive(Debug, Clone)]
pub struct Mob {
    /// Mob ID (UUID)
    pub mob_id: Uuid,
    /// Mob name
    pub name: String,
    /// Mob slots
    pub slots: Vec<MobSlot>,
    /// Mob type
    pub mob_type: MobType,
    /// Creation time
    pub creation_time: Option<u64>,
    /// Modified time
    pub modified_time: Option<u64>,
    /// User comments
    pub comments: HashMap<String, String>,
    /// Attributes
    pub attributes: HashMap<String, PropertyValue>,
    /// Primary source file path (used by source mobs)
    source_path: Option<String>,
    /// Additional locator paths
    locators: Vec<MobLocator>,
}

impl Mob {
    /// Create a new mob
    #[must_use]
    pub fn new(mob_id: Uuid, name: String, mob_type: MobType) -> Self {
        Self {
            mob_id,
            name,
            slots: Vec::new(),
            mob_type,
            creation_time: None,
            modified_time: None,
            comments: HashMap::new(),
            attributes: HashMap::new(),
            source_path: None,
            locators: Vec::new(),
        }
    }

    /// Get mob ID
    #[must_use]
    pub fn mob_id(&self) -> Uuid {
        self.mob_id
    }

    /// Get mob name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get a mutable reference to the mob ID.
    ///
    /// Useful for reassigning the ID when deep-cloning a mob.
    pub fn mob_id_mut(&mut self) -> &mut Uuid {
        &mut self.mob_id
    }

    /// Get mob type
    #[must_use]
    pub fn mob_type(&self) -> MobType {
        self.mob_type
    }

    /// Check if this is a master mob
    #[must_use]
    pub fn is_master_mob(&self) -> bool {
        matches!(self.mob_type, MobType::Master)
    }

    /// Check if this is a source mob
    #[must_use]
    pub fn is_source_mob(&self) -> bool {
        matches!(self.mob_type, MobType::Source)
    }

    /// Check if this is a composition mob
    #[must_use]
    pub fn is_composition_mob(&self) -> bool {
        matches!(self.mob_type, MobType::Composition)
    }

    /// Get all slots
    #[must_use]
    pub fn slots(&self) -> &[MobSlot] {
        &self.slots
    }

    /// Get slot by ID
    #[must_use]
    pub fn get_slot(&self, slot_id: u32) -> Option<&MobSlot> {
        self.slots.iter().find(|s| s.slot_id == slot_id)
    }

    /// Add a slot
    pub fn add_slot(&mut self, slot: MobSlot) {
        self.slots.push(slot);
    }

    /// Get the primary source-file path, if set.
    #[must_use]
    pub fn source_path(&self) -> Option<&str> {
        self.source_path.as_deref()
    }

    /// Set the primary source-file path.
    pub fn set_source_path(&mut self, path: &str) {
        self.source_path = Some(path.to_string());
    }

    /// Get all locator entries.
    #[must_use]
    pub fn locators(&self) -> &[MobLocator] {
        &self.locators
    }

    /// Get a mutable reference to the locator list.
    pub fn locators_mut(&mut self) -> &mut Vec<MobLocator> {
        &mut self.locators
    }

    /// Append a locator entry.
    pub fn add_locator(&mut self, locator: MobLocator) {
        self.locators.push(locator);
    }
}

/// Mob type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobType {
    /// Master mob
    Master,
    /// Source mob (file or recording)
    Source,
    /// Composition mob
    Composition,
}

/// Mob slot (timeline or event)
#[derive(Debug, Clone)]
pub struct MobSlot {
    /// Slot ID
    pub slot_id: u32,
    /// Slot name
    pub name: String,
    /// Physical track number
    pub physical_track_number: Option<u32>,
    /// Edit rate
    pub edit_rate: EditRate,
    /// Origin (start position)
    pub origin: Position,
    /// Segment
    pub segment: Option<Box<Segment>>,
    /// Slot type
    pub slot_type: SlotType,
}

impl MobSlot {
    /// Create a new timeline mob slot
    #[must_use]
    pub fn new_timeline(slot_id: u32, name: String, edit_rate: EditRate, origin: Position) -> Self {
        Self {
            slot_id,
            name,
            physical_track_number: None,
            edit_rate,
            origin,
            segment: None,
            slot_type: SlotType::Timeline,
        }
    }

    /// Create a new event mob slot
    #[must_use]
    pub fn new_event(slot_id: u32, name: String, edit_rate: EditRate) -> Self {
        Self {
            slot_id,
            name,
            physical_track_number: None,
            edit_rate,
            origin: Position::zero(),
            segment: None,
            slot_type: SlotType::Event,
        }
    }

    /// Get the segment
    #[must_use]
    pub fn segment(&self) -> Option<&Segment> {
        self.segment.as_deref()
    }

    /// Set the segment
    pub fn set_segment(&mut self, segment: Segment) {
        self.segment = Some(Box::new(segment));
    }
}

/// Slot type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotType {
    /// Timeline slot
    Timeline,
    /// Event slot
    Event,
    /// Static slot
    Static,
}

/// AAF Segment
#[derive(Debug, Clone)]
pub enum Segment {
    /// Sequence of components
    Sequence(SequenceSegment),
    /// Source clip
    SourceClip(SourceClipSegment),
    /// Filler
    Filler(FillerSegment),
    /// Transition
    Transition(TransitionSegment),
    /// Selector
    Selector(SelectorSegment),
    /// Nested scope
    NestedScope(NestedScopeSegment),
    /// Effect invocation
    OperationGroup(OperationGroupSegment),
}

impl Segment {
    /// Get the length of the segment
    #[must_use]
    pub fn length(&self) -> Option<i64> {
        match self {
            Segment::Sequence(s) => s.length,
            Segment::SourceClip(s) => Some(s.length),
            Segment::Filler(s) => Some(s.length),
            Segment::Transition(s) => Some(s.length),
            Segment::Selector(s) => s.length,
            Segment::NestedScope(s) => s.length,
            Segment::OperationGroup(s) => s.length,
        }
    }

    /// Check if this is a sequence
    #[must_use]
    pub fn is_sequence(&self) -> bool {
        matches!(self, Segment::Sequence(_))
    }

    /// Check if this is a source clip
    #[must_use]
    pub fn is_source_clip(&self) -> bool {
        matches!(self, Segment::SourceClip(_))
    }
}

/// Sequence segment
#[derive(Debug, Clone)]
pub struct SequenceSegment {
    /// Components in the sequence
    pub components: Vec<Component>,
    /// Length (optional, may be calculated)
    pub length: Option<i64>,
}

impl SequenceSegment {
    /// Create a new sequence
    #[must_use]
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            length: None,
        }
    }

    /// Add a component
    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }

    /// Calculate total length
    #[must_use]
    pub fn calculate_length(&self) -> Option<i64> {
        let mut total = 0i64;
        for comp in &self.components {
            total += comp.length()?;
        }
        Some(total)
    }
}

impl Default for SequenceSegment {
    fn default() -> Self {
        Self::new()
    }
}

/// Source clip segment
#[derive(Debug, Clone)]
pub struct SourceClipSegment {
    /// Length
    pub length: i64,
    /// Start time in source
    pub start_time: Position,
    /// Source mob ID
    pub source_mob_id: Uuid,
    /// Source mob slot ID
    pub source_mob_slot_id: u32,
}

impl SourceClipSegment {
    /// Create a new source clip
    #[must_use]
    pub fn new(
        length: i64,
        start_time: Position,
        source_mob_id: Uuid,
        source_mob_slot_id: u32,
    ) -> Self {
        Self {
            length,
            start_time,
            source_mob_id,
            source_mob_slot_id,
        }
    }
}

/// Filler segment
#[derive(Debug, Clone)]
pub struct FillerSegment {
    /// Length
    pub length: i64,
}

impl FillerSegment {
    /// Create a new filler
    #[must_use]
    pub fn new(length: i64) -> Self {
        Self { length }
    }
}

/// Transition segment
#[derive(Debug, Clone)]
pub struct TransitionSegment {
    /// Length
    pub length: i64,
    /// Cut point
    pub cut_point: Position,
    /// Effect
    pub effect: Option<Box<OperationGroupSegment>>,
}

impl TransitionSegment {
    /// Create a new transition
    #[must_use]
    pub fn new(length: i64, cut_point: Position) -> Self {
        Self {
            length,
            cut_point,
            effect: None,
        }
    }
}

/// Selector segment
#[derive(Debug, Clone)]
pub struct SelectorSegment {
    /// Selected track
    pub selected: u32,
    /// Alternate tracks
    pub alternates: Vec<Segment>,
    /// Length
    pub length: Option<i64>,
}

impl SelectorSegment {
    /// Create a new selector
    #[must_use]
    pub fn new(selected: u32) -> Self {
        Self {
            selected,
            alternates: Vec::new(),
            length: None,
        }
    }
}

/// Nested scope segment
#[derive(Debug, Clone)]
pub struct NestedScopeSegment {
    /// Slots
    pub slots: Vec<MobSlot>,
    /// Length
    pub length: Option<i64>,
}

impl NestedScopeSegment {
    /// Create a new nested scope
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            length: None,
        }
    }
}

impl Default for NestedScopeSegment {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation group segment (effect invocation)
#[derive(Debug, Clone)]
pub struct OperationGroupSegment {
    /// Operation definition ID
    pub operation_id: Auid,
    /// Input segments
    pub input_segments: Vec<Segment>,
    /// Parameters
    pub parameters: Vec<Parameter>,
    /// Length
    pub length: Option<i64>,
}

impl OperationGroupSegment {
    /// Create a new operation group
    #[must_use]
    pub fn new(operation_id: Auid) -> Self {
        Self {
            operation_id,
            input_segments: Vec::new(),
            parameters: Vec::new(),
            length: None,
        }
    }
}

/// Component (wrapper for segments with data definition)
#[derive(Debug, Clone)]
pub struct Component {
    /// Data definition (picture, sound, timecode, etc.)
    pub data_definition: Auid,
    /// Segment
    pub segment: Segment,
}

impl Component {
    /// Create a new component
    #[must_use]
    pub fn new(data_definition: Auid, segment: Segment) -> Self {
        Self {
            data_definition,
            segment,
        }
    }

    /// Get component length
    #[must_use]
    pub fn length(&self) -> Option<i64> {
        self.segment.length()
    }

    /// Check if this is a picture component
    #[must_use]
    pub fn is_picture(&self) -> bool {
        self.data_definition.is_picture()
    }

    /// Check if this is a sound component
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.data_definition.is_sound()
    }

    /// Check if this is a timecode component
    #[must_use]
    pub fn is_timecode(&self) -> bool {
        self.data_definition.is_timecode()
    }
}

/// Effect parameter
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter definition ID
    pub definition_id: Auid,
    /// Parameter name
    pub name: String,
    /// Parameter value
    pub value: ParameterValue,
}

impl Parameter {
    /// Create a new parameter
    #[must_use]
    pub fn new(definition_id: Auid, name: String, value: ParameterValue) -> Self {
        Self {
            definition_id,
            name,
            value,
        }
    }
}

/// Parameter value types
#[derive(Debug, Clone)]
pub enum ParameterValue {
    /// Constant value
    Constant(PropertyValue),
    /// Varying value (keyframes)
    Varying(Vec<Keyframe>),
}

/// Keyframe for varying parameter
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Time position
    pub time: Position,
    /// Value at this time
    pub value: PropertyValue,
    /// Interpolation type
    pub interpolation: InterpolationType,
}

/// Interpolation type for keyframes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationType {
    /// No interpolation (step)
    None,
    /// Linear interpolation
    Linear,
    /// Bezier interpolation
    Bezier,
    /// Cubic interpolation
    Cubic,
}

/// Property value types
#[derive(Debug, Clone)]
pub enum PropertyValue {
    /// Boolean
    Boolean(bool),
    /// 8-bit signed integer
    Int8(i8),
    /// 8-bit unsigned integer
    UInt8(u8),
    /// 16-bit signed integer
    Int16(i16),
    /// 16-bit unsigned integer
    UInt16(u16),
    /// 32-bit signed integer
    Int32(i32),
    /// 32-bit unsigned integer
    UInt32(u32),
    /// 64-bit signed integer
    Int64(i64),
    /// 64-bit unsigned integer
    UInt64(u64),
    /// 32-bit float
    Float(f32),
    /// 64-bit float
    Double(f64),
    /// String (UTF-16)
    String(String),
    /// Raw bytes
    Bytes(Vec<u8>),
    /// AUID
    Auid(Auid),
    /// UUID
    Uuid(Uuid),
    /// Strong reference
    StrongRef(Uuid),
    /// Weak reference
    WeakRef(Uuid),
    /// Array of values
    Array(Vec<PropertyValue>),
    /// Record (struct)
    Record(HashMap<String, PropertyValue>),
    /// Set
    Set(Vec<PropertyValue>),
}

impl PropertyValue {
    /// Read property value from bytes
    pub fn read_from_bytes(data: &[u8], type_def: &TypeDefinition) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        match type_def.name() {
            "Boolean" | "bool" => {
                let val = cursor.read_u8()? != 0;
                Ok(PropertyValue::Boolean(val))
            }
            "Int8" | "i8" => {
                let val = cursor.read_i8()?;
                Ok(PropertyValue::Int8(val))
            }
            "UInt8" | "u8" => {
                let val = cursor.read_u8()?;
                Ok(PropertyValue::UInt8(val))
            }
            "Int16" | "i16" => {
                let val = cursor.read_i16::<LittleEndian>()?;
                Ok(PropertyValue::Int16(val))
            }
            "UInt16" | "u16" => {
                let val = cursor.read_u16::<LittleEndian>()?;
                Ok(PropertyValue::UInt16(val))
            }
            "Int32" | "i32" => {
                let val = cursor.read_i32::<LittleEndian>()?;
                Ok(PropertyValue::Int32(val))
            }
            "UInt32" | "u32" => {
                let val = cursor.read_u32::<LittleEndian>()?;
                Ok(PropertyValue::UInt32(val))
            }
            "Int64" | "i64" | "Length" | "Position" => {
                let val = cursor.read_i64::<LittleEndian>()?;
                Ok(PropertyValue::Int64(val))
            }
            "UInt64" | "u64" => {
                let val = cursor.read_u64::<LittleEndian>()?;
                Ok(PropertyValue::UInt64(val))
            }
            "Float" | "f32" => {
                let val = cursor.read_f32::<LittleEndian>()?;
                Ok(PropertyValue::Float(val))
            }
            "Double" | "f64" => {
                let val = cursor.read_f64::<LittleEndian>()?;
                Ok(PropertyValue::Double(val))
            }
            "String" | "UTF16String" => {
                let mut utf16_chars = Vec::new();
                while let Ok(ch) = cursor.read_u16::<LittleEndian>() {
                    if ch == 0 {
                        break;
                    }
                    utf16_chars.push(ch);
                }
                let string = String::from_utf16_lossy(&utf16_chars);
                Ok(PropertyValue::String(string))
            }
            "AUID" => {
                let mut auid_bytes = [0u8; 16];
                cursor.read_exact(&mut auid_bytes)?;
                let auid = Auid::from_bytes(&auid_bytes);
                Ok(PropertyValue::Auid(auid))
            }
            "MobID" | "UUID" => {
                let mut uuid_bytes = [0u8; 16];
                cursor.read_exact(&mut uuid_bytes)?;
                let uuid = Uuid::from_bytes(uuid_bytes);
                Ok(PropertyValue::Uuid(uuid))
            }
            _ => {
                // Return raw bytes for unknown types
                Ok(PropertyValue::Bytes(data.to_vec()))
            }
        }
    }

    /// Convert to integer if possible
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            PropertyValue::Int8(v) => Some(i64::from(*v)),
            PropertyValue::UInt8(v) => Some(i64::from(*v)),
            PropertyValue::Int16(v) => Some(i64::from(*v)),
            PropertyValue::UInt16(v) => Some(i64::from(*v)),
            PropertyValue::Int32(v) => Some(i64::from(*v)),
            PropertyValue::UInt32(v) => Some(i64::from(*v)),
            PropertyValue::Int64(v) => Some(*v),
            PropertyValue::UInt64(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Convert to string if possible
    #[must_use]
    pub fn as_string(&self) -> Option<&str> {
        match self {
            PropertyValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Convert to UUID if possible
    #[must_use]
    pub fn as_uuid(&self) -> Option<Uuid> {
        match self {
            PropertyValue::Uuid(u) => Some(*u),
            PropertyValue::StrongRef(u) => Some(*u),
            PropertyValue::WeakRef(u) => Some(*u),
            _ => None,
        }
    }
}

/// Object reference resolver
pub struct ObjectResolver {
    objects: HashMap<Uuid, ObjectData>,
}

impl ObjectResolver {
    /// Create a new resolver
    #[must_use]
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
        }
    }

    /// Register an object
    pub fn register(&mut self, id: Uuid, data: ObjectData) {
        self.objects.insert(id, data);
    }

    /// Resolve a reference
    #[must_use]
    pub fn resolve(&self, id: &Uuid) -> Option<&ObjectData> {
        self.objects.get(id)
    }

    /// Resolve a strong reference
    #[must_use]
    pub fn resolve_strong_ref(&self, value: &PropertyValue) -> Option<&ObjectData> {
        if let PropertyValue::StrongRef(id) = value {
            self.resolve(id)
        } else {
            None
        }
    }

    /// Resolve a weak reference
    #[must_use]
    pub fn resolve_weak_ref(&self, value: &PropertyValue) -> Option<&ObjectData> {
        if let PropertyValue::WeakRef(id) = value {
            self.resolve(id)
        } else {
            None
        }
    }
}

impl Default for ObjectResolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Generic object data
#[derive(Debug, Clone)]
pub struct ObjectData {
    /// Object ID
    pub id: Uuid,
    /// Class ID
    pub class_id: Auid,
    /// Properties
    pub properties: HashMap<String, PropertyValue>,
}

impl ObjectData {
    /// Create a new object data
    #[must_use]
    pub fn new(id: Uuid, class_id: Auid) -> Self {
        Self {
            id,
            class_id,
            properties: HashMap::new(),
        }
    }

    /// Get a property
    #[must_use]
    pub fn get_property(&self, name: &str) -> Option<&PropertyValue> {
        self.properties.get(name)
    }

    /// Set a property
    pub fn set_property(&mut self, name: String, value: PropertyValue) {
        self.properties.insert(name, value);
    }

    /// Get property as integer
    #[must_use]
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.get_property(name).and_then(PropertyValue::as_int)
    }

    /// Get property as string
    #[must_use]
    pub fn get_string(&self, name: &str) -> Option<&str> {
        self.get_property(name).and_then(|v| v.as_string())
    }

    /// Get property as UUID
    #[must_use]
    pub fn get_uuid(&self, name: &str) -> Option<Uuid> {
        self.get_property(name).and_then(PropertyValue::as_uuid)
    }
}

/// Read header from storage
pub fn read_header<R: Read + Seek>(_storage: &mut StorageReader<R>) -> Result<Header> {
    // In a real implementation, we would read from the "Header" stream
    // For now, return a default header
    Ok(Header::new())
}

/// Read content storage from AAF file
pub fn read_content_storage<R: Read + Seek>(
    _storage: &mut StorageReader<R>,
    _dictionary: &Dictionary,
) -> Result<ContentStorage> {
    // In a real implementation, we would:
    // 1. Read the "Content Storage" object
    // 2. Parse all mobs
    // 3. Parse mob slots and segments
    // 4. Resolve references
    // For now, return an empty content storage
    Ok(ContentStorage::new())
}

/// Property visitor trait
pub trait PropertyVisitor {
    /// Visit a property
    fn visit_property(&mut self, name: &str, value: &PropertyValue) -> Result<()>;

    /// Visit an object
    fn visit_object(&mut self, obj: &ObjectData) -> Result<()> {
        for (name, value) in &obj.properties {
            self.visit_property(name, value)?;
        }
        Ok(())
    }
}

/// Property traversal helper
pub struct PropertyTraverser<'a> {
    resolver: &'a ObjectResolver,
}

impl<'a> PropertyTraverser<'a> {
    /// Create a new traverser
    #[must_use]
    pub fn new(resolver: &'a ObjectResolver) -> Self {
        Self { resolver }
    }

    /// Traverse an object and its references
    pub fn traverse<V: PropertyVisitor>(&self, obj: &ObjectData, visitor: &mut V) -> Result<()> {
        visitor.visit_object(obj)?;

        // Traverse strong references
        for value in obj.properties.values() {
            if let Some(ref_obj) = self.resolver.resolve_strong_ref(value) {
                self.traverse(ref_obj, visitor)?;
            }
        }

        Ok(())
    }
}

/// Weak reference map for resolving mob references
pub struct WeakReferenceMap {
    references: HashMap<Uuid, Uuid>,
}

impl WeakReferenceMap {
    /// Create a new weak reference map
    #[must_use]
    pub fn new() -> Self {
        Self {
            references: HashMap::new(),
        }
    }

    /// Add a reference
    pub fn add(&mut self, target_id: Uuid, mob_id: Uuid) {
        self.references.insert(target_id, mob_id);
    }

    /// Resolve a reference
    #[must_use]
    pub fn resolve(&self, target_id: &Uuid) -> Option<Uuid> {
        self.references.get(target_id).copied()
    }
}

impl Default for WeakReferenceMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_creation() {
        let header = Header::new();
        assert_eq!(header.major_version, 1);
        assert_eq!(header.minor_version, 1);
        assert_eq!(header.version_string(), "1.1");
    }

    #[test]
    fn test_mob_creation() {
        let mob_id = Uuid::new_v4();
        let mob = Mob::new(mob_id, "Test Mob".to_string(), MobType::Master);
        assert_eq!(mob.mob_id(), mob_id);
        assert_eq!(mob.name(), "Test Mob");
        assert!(mob.is_master_mob());
        assert!(!mob.is_source_mob());
    }

    #[test]
    fn test_mob_slot_creation() {
        let slot = MobSlot::new_timeline(
            1,
            "Video".to_string(),
            EditRate::new(24, 1),
            Position::zero(),
        );
        assert_eq!(slot.slot_id, 1);
        assert_eq!(slot.name, "Video");
        assert_eq!(slot.slot_type, SlotType::Timeline);
    }

    #[test]
    fn test_sequence_segment() {
        let seq = SequenceSegment::new();
        assert!(seq.components.is_empty());
        assert_eq!(seq.calculate_length(), Some(0));
    }

    #[test]
    fn test_source_clip() {
        let clip = SourceClipSegment::new(100, Position::zero(), Uuid::new_v4(), 1);
        assert_eq!(clip.length, 100);
    }

    #[test]
    fn test_filler_segment() {
        let filler = FillerSegment::new(50);
        assert_eq!(filler.length, 50);
    }

    #[test]
    fn test_property_value_int() {
        let val = PropertyValue::Int32(42);
        assert_eq!(val.as_int(), Some(42));
    }

    #[test]
    fn test_property_value_string() {
        let val = PropertyValue::String("test".to_string());
        assert_eq!(val.as_string(), Some("test"));
    }

    #[test]
    fn test_object_data() {
        let obj = ObjectData::new(Uuid::new_v4(), Auid::null());
        assert!(obj.properties.is_empty());
    }

    #[test]
    fn test_object_resolver() {
        let resolver = ObjectResolver::new();
        let id = Uuid::new_v4();
        assert!(resolver.resolve(&id).is_none());
    }

    #[test]
    fn test_weak_reference_map() {
        let mut map = WeakReferenceMap::new();
        let target_id = Uuid::new_v4();
        let mob_id = Uuid::new_v4();
        map.add(target_id, mob_id);
        assert_eq!(map.resolve(&target_id), Some(mob_id));
    }
}
