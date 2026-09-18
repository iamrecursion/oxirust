//! NMOS IS-08 Audio Channel Mapping API.
//!
//! This module implements the AMWA NMOS IS-08 v1.0 Audio Channel Mapping API,
//! which defines a REST API for controlling audio channel mapping on NMOS devices.
//! It allows controllers to assign audio sources to specific output channels,
//! enabling flexible audio routing in IP-based broadcast infrastructure.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ============================================================================
// Core types
// ============================================================================

/// Audio channel identifier (string label).
pub type ChannelId = String;

/// Output channel descriptor for a device receiver.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutputChannel {
    /// Zero-based index of the output channel on the device.
    pub index: u32,
    /// Optional human-readable label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl OutputChannel {
    /// Create a new output channel with no label.
    pub fn new(index: u32) -> Self {
        Self { index, label: None }
    }

    /// Create a new output channel with a label.
    pub fn with_label(index: u32, label: impl Into<String>) -> Self {
        Self {
            index,
            label: Some(label.into()),
        }
    }
}

/// Reference to a specific channel within a named NMOS input (receiver).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InputReference {
    /// ID of the NMOS input (receiver).
    pub input_id: String,
    /// Zero-based channel index within that input.
    pub channel_index: u32,
}

impl InputReference {
    /// Create a new input reference.
    pub fn new(input_id: impl Into<String>, channel_index: u32) -> Self {
        Self {
            input_id: input_id.into(),
            channel_index,
        }
    }
}

/// A single entry in the channel mapping table.
///
/// Maps one output channel to one input channel (or silence when `input` is `None`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChannelMappingEntry {
    /// Zero-based output channel index on the device.
    pub output_channel: u32,
    /// Source channel reference, or `None` to silence this output.
    pub input: Option<InputReference>,
}

impl ChannelMappingEntry {
    /// Create an entry that routes an output channel to an input channel.
    pub fn routed(output_channel: u32, input_id: impl Into<String>, channel_index: u32) -> Self {
        Self {
            output_channel,
            input: Some(InputReference::new(input_id, channel_index)),
        }
    }

    /// Create an entry that silences an output channel.
    pub fn silence(output_channel: u32) -> Self {
        Self {
            output_channel,
            input: None,
        }
    }
}

// ============================================================================
// ChannelMappingTable
// ============================================================================

/// Complete channel mapping table for a single device.
///
/// Maintains a staged (pending) mapping and an active (committed) mapping,
/// following the IS-08 two-phase commit pattern.
#[derive(Debug, Clone)]
pub struct ChannelMappingTable {
    /// Device identifier this table belongs to.
    pub device_id: String,
    /// Total number of output channels on this device.
    pub output_channels: u32,
    /// All entries (canonical view, merged from active).
    pub entries: Vec<ChannelMappingEntry>,
    /// Currently active mapping (committed and live).
    pub active: Vec<ChannelMappingEntry>,
    /// Pending staged mapping (awaiting activation).
    pub staged: Vec<ChannelMappingEntry>,
}

impl ChannelMappingTable {
    /// Create a new mapping table for a device with `output_channels` outputs.
    ///
    /// All outputs are initialised to silence.
    pub fn new(device_id: impl Into<String>, output_channels: u32) -> Self {
        let silent: Vec<ChannelMappingEntry> = (0..output_channels)
            .map(ChannelMappingEntry::silence)
            .collect();
        Self {
            device_id: device_id.into(),
            output_channels,
            entries: silent.clone(),
            active: silent,
            staged: Vec::new(),
        }
    }

    /// Stage a new set of mapping entries for later activation.
    ///
    /// The entries are validated before being stored.  Any existing staged
    /// mapping is replaced.
    pub fn stage_mapping(
        &mut self,
        entries: Vec<ChannelMappingEntry>,
    ) -> Result<(), ChannelMappingError> {
        self.validate(&entries)?;
        self.staged = entries;
        Ok(())
    }

    /// Activate the currently staged mapping, making it the live mapping.
    ///
    /// Returns `ChannelMappingError::NoStagedMapping` if no mapping has been
    /// staged yet.
    pub fn activate(&mut self) -> Result<(), ChannelMappingError> {
        if self.staged.is_empty() {
            return Err(ChannelMappingError::NoStagedMapping);
        }
        // Merge staged on top of active: start from current active, then apply
        // each staged entry overwriting the corresponding output channel.
        let mut merged = self.active.clone();
        for staged_entry in &self.staged {
            // Find the slot in merged for this output_channel and update it.
            let mut found = false;
            for existing in &mut merged {
                if existing.output_channel == staged_entry.output_channel {
                    existing.input = staged_entry.input.clone();
                    found = true;
                    break;
                }
            }
            if !found {
                merged.push(staged_entry.clone());
            }
        }
        // Sort by output_channel for deterministic ordering.
        merged.sort_by_key(|e| e.output_channel);
        self.active = merged.clone();
        self.entries = merged;
        self.staged.clear();
        Ok(())
    }

    /// Deactivate the current mapping, setting all outputs to silence.
    pub fn deactivate(&mut self) {
        let silent: Vec<ChannelMappingEntry> = (0..self.output_channels)
            .map(ChannelMappingEntry::silence)
            .collect();
        self.active = silent.clone();
        self.entries = silent;
        self.staged.clear();
    }

    /// Immediately route one output channel to a specific input channel.
    ///
    /// This is a convenience helper that stages and activates a single-entry
    /// mapping atomically.
    pub fn route(
        &mut self,
        output: u32,
        input_id: impl Into<String>,
        input_channel: u32,
    ) -> Result<(), ChannelMappingError> {
        if output >= self.output_channels {
            return Err(ChannelMappingError::OutputChannelOutOfRange(
                output,
                self.output_channels,
            ));
        }
        let entry = ChannelMappingEntry::routed(output, input_id, input_channel);
        // Apply directly to active (single-step convenience method).
        self.apply_entry_to_active(entry);
        Ok(())
    }

    /// Immediately silence one output channel.
    pub fn silence_output(&mut self, output: u32) -> Result<(), ChannelMappingError> {
        if output >= self.output_channels {
            return Err(ChannelMappingError::OutputChannelOutOfRange(
                output,
                self.output_channels,
            ));
        }
        let entry = ChannelMappingEntry::silence(output);
        self.apply_entry_to_active(entry);
        Ok(())
    }

    /// Validate a set of mapping entries against this table's constraints.
    ///
    /// Checks:
    /// - No output channel index exceeds the device maximum.
    /// - No duplicate output channel indices.
    pub fn validate(&self, entries: &[ChannelMappingEntry]) -> Result<(), ChannelMappingError> {
        let mut seen = std::collections::HashSet::new();
        for entry in entries {
            if entry.output_channel >= self.output_channels {
                return Err(ChannelMappingError::OutputChannelOutOfRange(
                    entry.output_channel,
                    self.output_channels,
                ));
            }
            if !seen.insert(entry.output_channel) {
                return Err(ChannelMappingError::InvalidMapping(format!(
                    "duplicate output channel index {}",
                    entry.output_channel
                )));
            }
        }
        Ok(())
    }

    // --- private helpers ---

    fn apply_entry_to_active(&mut self, entry: ChannelMappingEntry) {
        let mut found = false;
        for existing in &mut self.active {
            if existing.output_channel == entry.output_channel {
                existing.input = entry.input.clone();
                found = true;
                break;
            }
        }
        if !found {
            self.active.push(entry.clone());
            self.active.sort_by_key(|e| e.output_channel);
        }
        // Keep entries in sync with active.
        self.entries = self.active.clone();
    }
}

// ============================================================================
// ChannelMappingRegistry
// ============================================================================

/// IS-08 channel mapping registry that manages per-device mapping tables.
#[derive(Debug, Default)]
pub struct ChannelMappingRegistry {
    /// Map from device_id to its channel mapping table.
    pub tables: HashMap<String, ChannelMappingTable>,
}

impl ChannelMappingRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new device with the given number of output channels.
    ///
    /// If a table for `device_id` already exists it is replaced.
    pub fn add_device(&mut self, device_id: impl Into<String>, output_channels: u32) {
        let id = device_id.into();
        self.tables
            .insert(id.clone(), ChannelMappingTable::new(id, output_channels));
    }

    /// Return a shared reference to the mapping table for a device.
    pub fn get_table(&self, device_id: &str) -> Option<&ChannelMappingTable> {
        self.tables.get(device_id)
    }

    /// Return a mutable reference to the mapping table for a device.
    pub fn get_table_mut(&mut self, device_id: &str) -> Option<&mut ChannelMappingTable> {
        self.tables.get_mut(device_id)
    }

    /// Return a sorted list of all registered device IDs.
    pub fn list_devices(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.tables.keys().map(|s| s.as_str()).collect();
        ids.sort();
        ids
    }

    /// Stage a mapping for a device, returning an error if the device does not exist.
    pub fn stage_mapping(
        &mut self,
        device_id: &str,
        entries: Vec<ChannelMappingEntry>,
    ) -> Result<(), ChannelMappingError> {
        let table = self
            .tables
            .get_mut(device_id)
            .ok_or_else(|| ChannelMappingError::DeviceNotFound(device_id.to_string()))?;
        table.stage_mapping(entries)
    }

    /// Activate the staged mapping for a device.
    pub fn activate(&mut self, device_id: &str) -> Result<(), ChannelMappingError> {
        let table = self
            .tables
            .get_mut(device_id)
            .ok_or_else(|| ChannelMappingError::DeviceNotFound(device_id.to_string()))?;
        table.activate()
    }

    /// Deactivate (silence all outputs) for a device.
    pub fn deactivate(&mut self, device_id: &str) -> Result<(), ChannelMappingError> {
        let table = self
            .tables
            .get_mut(device_id)
            .ok_or_else(|| ChannelMappingError::DeviceNotFound(device_id.to_string()))?;
        table.deactivate();
        Ok(())
    }
}

// ============================================================================
// Error type
// ============================================================================

/// Errors that can occur during IS-08 channel mapping operations.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ChannelMappingError {
    /// The referenced device was not found in the registry.
    #[error("device not found: {0}")]
    DeviceNotFound(String),

    /// An output channel index exceeded the device's channel count.
    #[error("output channel {0} out of range (max {1})")]
    OutputChannelOutOfRange(u32, u32),

    /// Attempted to activate but no mapping has been staged.
    #[error("no staged mapping to activate")]
    NoStagedMapping,

    /// The mapping was structurally invalid (e.g. duplicate output indices).
    #[error("invalid mapping: {0}")]
    InvalidMapping(String),
}

// ============================================================================
// Serialization helpers (for HTTP handlers)
// ============================================================================

/// Serialize a `ChannelMappingTable`'s active mapping to a serde_json Value.
pub fn active_to_json(table: &ChannelMappingTable) -> serde_json::Value {
    serde_json::to_value(&table.active).unwrap_or(serde_json::Value::Array(vec![]))
}

/// Serialize a `ChannelMappingTable`'s staged mapping to a serde_json Value.
pub fn staged_to_json(table: &ChannelMappingTable) -> serde_json::Value {
    serde_json::to_value(&table.staged).unwrap_or(serde_json::Value::Array(vec![]))
}

/// Serialize the IS-08 IO summary for a device.
pub fn io_summary_to_json(device_id: &str, table: &ChannelMappingTable) -> serde_json::Value {
    let outputs: Vec<serde_json::Value> = (0..table.output_channels)
        .map(|i| {
            serde_json::json!({
                "index": i,
                "routable_inputs": serde_json::Value::Null
            })
        })
        .collect();
    serde_json::json!({
        "device_id": device_id,
        "output_channels": table.output_channels,
        "outputs": outputs
    })
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── OutputChannel ─────────────────────────────────────────────────────

    #[test]
    fn test_output_channel_new_no_label() {
        let ch = OutputChannel::new(3);
        assert_eq!(ch.index, 3);
        assert!(ch.label.is_none());
    }

    #[test]
    fn test_output_channel_with_label() {
        let ch = OutputChannel::with_label(0, "Left");
        assert_eq!(ch.index, 0);
        assert_eq!(ch.label.as_deref(), Some("Left"));
    }

    #[test]
    fn test_output_channel_serde_roundtrip() {
        let ch = OutputChannel::with_label(1, "Right");
        let json = serde_json::to_string(&ch).expect("serialize");
        let back: OutputChannel = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(ch, back);
    }

    // ── InputReference ────────────────────────────────────────────────────

    #[test]
    fn test_input_reference_new() {
        let r = InputReference::new("rx-01", 2);
        assert_eq!(r.input_id, "rx-01");
        assert_eq!(r.channel_index, 2);
    }

    #[test]
    fn test_input_reference_serde_roundtrip() {
        let r = InputReference::new("rx-02", 5);
        let json = serde_json::to_string(&r).expect("serialize");
        let back: InputReference = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(r, back);
    }

    // ── ChannelMappingEntry ───────────────────────────────────────────────

    #[test]
    fn test_mapping_entry_routed() {
        let e = ChannelMappingEntry::routed(0, "rx-a", 1);
        assert_eq!(e.output_channel, 0);
        let inp = e.input.expect("should have input");
        assert_eq!(inp.input_id, "rx-a");
        assert_eq!(inp.channel_index, 1);
    }

    #[test]
    fn test_mapping_entry_silence() {
        let e = ChannelMappingEntry::silence(7);
        assert_eq!(e.output_channel, 7);
        assert!(e.input.is_none());
    }

    #[test]
    fn test_mapping_entry_serde_routed() {
        let e = ChannelMappingEntry::routed(2, "rx-b", 3);
        let json = serde_json::to_string(&e).expect("serialize");
        let back: ChannelMappingEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(e, back);
    }

    #[test]
    fn test_mapping_entry_serde_silence() {
        let e = ChannelMappingEntry::silence(4);
        let json = serde_json::to_string(&e).expect("serialize");
        let back: ChannelMappingEntry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(e, back);
    }

    // ── ChannelMappingTable ───────────────────────────────────────────────

    #[test]
    fn test_table_new_all_silence() {
        let t = ChannelMappingTable::new("dev-1", 4);
        assert_eq!(t.output_channels, 4);
        assert_eq!(t.active.len(), 4);
        for entry in &t.active {
            assert!(entry.input.is_none(), "should start silent");
        }
        assert!(t.staged.is_empty());
    }

    #[test]
    fn test_table_stage_mapping_valid() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        let entries = vec![
            ChannelMappingEntry::routed(0, "rx-1", 0),
            ChannelMappingEntry::routed(1, "rx-1", 1),
        ];
        assert!(t.stage_mapping(entries.clone()).is_ok());
        assert_eq!(t.staged.len(), 2);
        // Active should be unchanged until activate().
        assert!(t.active[0].input.is_none());
    }

    #[test]
    fn test_table_stage_mapping_out_of_range() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        let entries = vec![ChannelMappingEntry::routed(4, "rx-1", 0)]; // index 4 invalid for 4-ch device
        let err = t.stage_mapping(entries);
        assert_eq!(err, Err(ChannelMappingError::OutputChannelOutOfRange(4, 4)));
    }

    #[test]
    fn test_table_stage_mapping_duplicate_output() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        let entries = vec![
            ChannelMappingEntry::routed(0, "rx-1", 0),
            ChannelMappingEntry::routed(0, "rx-1", 1), // duplicate output 0
        ];
        let err = t.stage_mapping(entries);
        assert!(matches!(err, Err(ChannelMappingError::InvalidMapping(_))));
    }

    #[test]
    fn test_table_activate_applies_staged() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        let entries = vec![
            ChannelMappingEntry::routed(0, "rx-1", 0),
            ChannelMappingEntry::routed(1, "rx-1", 1),
        ];
        t.stage_mapping(entries).expect("stage");
        t.activate().expect("activate");

        // Outputs 0 and 1 should now be routed.
        let ch0 = t
            .active
            .iter()
            .find(|e| e.output_channel == 0)
            .expect("ch0");
        assert!(ch0.input.is_some());

        // Staged should be cleared.
        assert!(t.staged.is_empty());
    }

    #[test]
    fn test_table_activate_no_staged_returns_error() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        let err = t.activate();
        assert_eq!(err, Err(ChannelMappingError::NoStagedMapping));
    }

    #[test]
    fn test_table_deactivate_silences_all() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        let entries = vec![ChannelMappingEntry::routed(0, "rx-1", 0)];
        t.stage_mapping(entries).expect("stage");
        t.activate().expect("activate");

        t.deactivate();

        for entry in &t.active {
            assert!(entry.input.is_none(), "should be silent after deactivate");
        }
        assert!(t.staged.is_empty());
    }

    #[test]
    fn test_table_route_immediate() {
        let mut t = ChannelMappingTable::new("dev-1", 8);
        t.route(3, "rx-2", 0).expect("route");

        let ch3 = t
            .active
            .iter()
            .find(|e| e.output_channel == 3)
            .expect("ch3");
        let inp = ch3.input.as_ref().expect("should be routed");
        assert_eq!(inp.input_id, "rx-2");
        assert_eq!(inp.channel_index, 0);
    }

    #[test]
    fn test_table_route_out_of_range() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        let err = t.route(10, "rx-1", 0);
        assert_eq!(
            err,
            Err(ChannelMappingError::OutputChannelOutOfRange(10, 4))
        );
    }

    #[test]
    fn test_table_silence_output_immediate() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        t.route(0, "rx-1", 0).expect("route");
        t.silence_output(0).expect("silence");

        let ch0 = t
            .active
            .iter()
            .find(|e| e.output_channel == 0)
            .expect("ch0");
        assert!(ch0.input.is_none());
    }

    #[test]
    fn test_table_silence_out_of_range() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        let err = t.silence_output(4);
        assert_eq!(err, Err(ChannelMappingError::OutputChannelOutOfRange(4, 4)));
    }

    #[test]
    fn test_table_validate_valid() {
        let t = ChannelMappingTable::new("dev-1", 8);
        let entries = vec![
            ChannelMappingEntry::silence(0),
            ChannelMappingEntry::silence(7),
        ];
        assert!(t.validate(&entries).is_ok());
    }

    #[test]
    fn test_table_entries_sync_with_active_after_route() {
        let mut t = ChannelMappingTable::new("dev-1", 4);
        t.route(2, "rx-1", 0).expect("route");
        // entries should mirror active
        assert_eq!(t.entries, t.active);
    }

    // ── ChannelMappingRegistry ────────────────────────────────────────────

    #[test]
    fn test_registry_new_empty() {
        let reg = ChannelMappingRegistry::new();
        assert!(reg.list_devices().is_empty());
    }

    #[test]
    fn test_registry_add_device() {
        let mut reg = ChannelMappingRegistry::new();
        reg.add_device("dev-a", 8);
        assert_eq!(reg.list_devices(), vec!["dev-a"]);
    }

    #[test]
    fn test_registry_get_table_some() {
        let mut reg = ChannelMappingRegistry::new();
        reg.add_device("dev-b", 4);
        assert!(reg.get_table("dev-b").is_some());
    }

    #[test]
    fn test_registry_get_table_none() {
        let reg = ChannelMappingRegistry::new();
        assert!(reg.get_table("nonexistent").is_none());
    }

    #[test]
    fn test_registry_get_table_mut() {
        let mut reg = ChannelMappingRegistry::new();
        reg.add_device("dev-c", 2);
        let table = reg.get_table_mut("dev-c").expect("should exist");
        table.route(0, "rx-1", 0).expect("route");
        let ch = reg.get_table("dev-c").expect("exists").active[0].clone();
        assert!(ch.input.is_some());
    }

    #[test]
    fn test_registry_list_devices_sorted() {
        let mut reg = ChannelMappingRegistry::new();
        reg.add_device("zzz", 2);
        reg.add_device("aaa", 2);
        reg.add_device("mmm", 2);
        let list = reg.list_devices();
        assert_eq!(list, vec!["aaa", "mmm", "zzz"]);
    }

    #[test]
    fn test_registry_stage_mapping_ok() {
        let mut reg = ChannelMappingRegistry::new();
        reg.add_device("dev-d", 4);
        let entries = vec![ChannelMappingEntry::routed(0, "rx-1", 0)];
        assert!(reg.stage_mapping("dev-d", entries).is_ok());
    }

    #[test]
    fn test_registry_stage_mapping_device_not_found() {
        let mut reg = ChannelMappingRegistry::new();
        let entries = vec![ChannelMappingEntry::routed(0, "rx-1", 0)];
        let err = reg.stage_mapping("ghost", entries);
        assert_eq!(
            err,
            Err(ChannelMappingError::DeviceNotFound("ghost".to_string()))
        );
    }

    #[test]
    fn test_registry_activate_ok() {
        let mut reg = ChannelMappingRegistry::new();
        reg.add_device("dev-e", 4);
        let entries = vec![ChannelMappingEntry::routed(0, "rx-1", 0)];
        reg.stage_mapping("dev-e", entries).expect("stage");
        assert!(reg.activate("dev-e").is_ok());
    }

    #[test]
    fn test_registry_activate_device_not_found() {
        let mut reg = ChannelMappingRegistry::new();
        let err = reg.activate("ghost");
        assert_eq!(
            err,
            Err(ChannelMappingError::DeviceNotFound("ghost".to_string()))
        );
    }

    #[test]
    fn test_registry_deactivate_ok() {
        let mut reg = ChannelMappingRegistry::new();
        reg.add_device("dev-f", 4);
        assert!(reg.deactivate("dev-f").is_ok());
    }

    #[test]
    fn test_registry_deactivate_device_not_found() {
        let mut reg = ChannelMappingRegistry::new();
        let err = reg.deactivate("ghost");
        assert_eq!(
            err,
            Err(ChannelMappingError::DeviceNotFound("ghost".to_string()))
        );
    }

    // ── JSON serialization helpers ────────────────────────────────────────

    #[test]
    fn test_active_to_json() {
        let mut t = ChannelMappingTable::new("dev-1", 2);
        t.route(0, "rx-1", 0).expect("route");
        let v = active_to_json(&t);
        assert!(v.is_array());
        let arr = v.as_array().expect("array");
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_staged_to_json_empty() {
        let t = ChannelMappingTable::new("dev-1", 2);
        let v = staged_to_json(&t);
        assert!(v.is_array());
        let arr = v.as_array().expect("array");
        assert_eq!(arr.len(), 0);
    }

    #[test]
    fn test_io_summary_to_json() {
        let t = ChannelMappingTable::new("dev-1", 3);
        let v = io_summary_to_json("dev-1", &t);
        assert_eq!(v["device_id"], "dev-1");
        assert_eq!(v["output_channels"], 3);
        let outputs = v["outputs"].as_array().expect("array");
        assert_eq!(outputs.len(), 3);
    }

    // ── Error Display ─────────────────────────────────────────────────────

    #[test]
    fn test_error_display_device_not_found() {
        let e = ChannelMappingError::DeviceNotFound("dev-x".to_string());
        assert!(e.to_string().contains("dev-x"));
    }

    #[test]
    fn test_error_display_out_of_range() {
        let e = ChannelMappingError::OutputChannelOutOfRange(5, 4);
        let s = e.to_string();
        assert!(s.contains('5'));
        assert!(s.contains('4'));
    }

    #[test]
    fn test_error_display_no_staged() {
        let e = ChannelMappingError::NoStagedMapping;
        assert!(e.to_string().contains("staged"));
    }

    #[test]
    fn test_error_display_invalid_mapping() {
        let e = ChannelMappingError::InvalidMapping("duplicate".to_string());
        assert!(e.to_string().contains("duplicate"));
    }
}
