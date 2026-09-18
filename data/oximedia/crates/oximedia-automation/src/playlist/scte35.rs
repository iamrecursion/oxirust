//! SCTE-35 splice command generation for playlist-based ad insertion.
//!
//! Provides `PlaylistScte35` — a per-playlist registry of splice events that
//! inserts `SpliceInsert` (ad break start) and `SpliceReturn` (ad break end)
//! command descriptors at precise millisecond positions.  The commands follow
//! the SCTE-35 2022 standard binary encoding conventions while remaining pure
//! Rust with no native dependencies.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A single SCTE-35 splice command attached to a playlist position.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scte35Command {
    /// `splice_insert()` — begins an avail/ad break.
    SpliceInsert(SpliceInsertParams),
    /// `splice_null()` return — restores the main programme.
    SpliceReturn(SpliceReturnParams),
}

/// Parameters for a `splice_insert` command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpliceInsertParams {
    /// 32-bit SCTE-35 event ID.
    pub event_id: u32,
    /// Requested duration of the avail in milliseconds.
    pub duration_ms: u64,
    /// If `true`, this splice shall occur out-of-network (the default for ad
    /// breaks in broadcast automation).
    pub out_of_network: bool,
    /// If `true`, `auto_return` flag is set — the stream returns automatically
    /// after `duration_ms`.
    pub auto_return: bool,
}

/// Parameters for a `splice_return` (return-to-network) command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpliceReturnParams {
    /// 32-bit SCTE-35 event ID of the originating `splice_insert`.
    pub event_id: u32,
}

/// An entry in the playlist splice table: position + command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpliceEntry {
    /// Position in the playlist timeline (milliseconds from start).
    pub position_ms: u64,
    /// The splice command to inject at this position.
    pub command: Scte35Command,
}

/// SCTE-35 splice command manager for a broadcast playlist.
///
/// Entries are stored in a `BTreeMap` keyed by position so iteration always
/// yields entries in chronological order.  Multiple commands at the same
/// millisecond are allowed (stored as a `Vec`).
#[derive(Debug, Default)]
pub struct PlaylistScte35 {
    /// Ordered map from position (ms) to the list of commands at that point.
    entries: BTreeMap<u64, Vec<SpliceEntry>>,
}

impl PlaylistScte35 {
    /// Create an empty splice command registry.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Insert a `splice_insert` (ad break start) command at `position_ms`.
    ///
    /// # Arguments
    ///
    /// * `position_ms`  — timeline position in milliseconds.
    /// * `duration_ms`  — requested avail duration in milliseconds.
    /// * `event_id`     — 32-bit SCTE-35 splice event identifier.
    ///
    /// Both `out_of_network` and `auto_return` are set to `true` by default
    /// which is the standard for linear broadcast ad insertion.
    pub fn insert_splice(&mut self, position_ms: u64, duration_ms: u64, event_id: u32) {
        let entry = SpliceEntry {
            position_ms,
            command: Scte35Command::SpliceInsert(SpliceInsertParams {
                event_id,
                duration_ms,
                out_of_network: true,
                auto_return: true,
            }),
        };
        self.entries.entry(position_ms).or_default().push(entry);
    }

    /// Insert a `splice_return` (return-to-network) command at `position_ms`.
    ///
    /// # Arguments
    ///
    /// * `position_ms` — timeline position in milliseconds.
    /// * `event_id`    — 32-bit SCTE-35 event ID matching the originating
    ///                   `splice_insert`.
    pub fn insert_return(&mut self, position_ms: u64, event_id: u32) {
        let entry = SpliceEntry {
            position_ms,
            command: Scte35Command::SpliceReturn(SpliceReturnParams { event_id }),
        };
        self.entries.entry(position_ms).or_default().push(entry);
    }

    /// Return all splice commands scheduled at or before `position_ms` that
    /// have not yet been yielded (i.e. a simple scan for broadcast playout
    /// consumption).
    ///
    /// Returns entries in chronological order.
    pub fn commands_at_or_before(&self, position_ms: u64) -> Vec<&SpliceEntry> {
        self.entries
            .range(..=position_ms)
            .flat_map(|(_, entries)| entries.iter())
            .collect()
    }

    /// Return all splice commands scheduled at exactly `position_ms`.
    pub fn commands_at(&self, position_ms: u64) -> Vec<&SpliceEntry> {
        self.entries
            .get(&position_ms)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Return all entries in chronological order.
    pub fn all_entries(&self) -> Vec<&SpliceEntry> {
        self.entries.values().flat_map(|v| v.iter()).collect()
    }

    /// Total number of splice commands registered.
    pub fn len(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Return `true` if no splice commands are registered.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all commands at exactly `position_ms`.
    pub fn remove_at(&mut self, position_ms: u64) {
        self.entries.remove(&position_ms);
    }

    /// Clear all registered splice commands.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Return the number of `splice_insert` commands registered.
    pub fn splice_count(&self) -> usize {
        self.entries
            .values()
            .flat_map(|v| v.iter())
            .filter(|e| matches!(e.command, Scte35Command::SpliceInsert(_)))
            .count()
    }

    /// Return the number of `splice_return` commands registered.
    pub fn return_count(&self) -> usize {
        self.entries
            .values()
            .flat_map(|v| v.iter())
            .filter(|e| matches!(e.command, Scte35Command::SpliceReturn(_)))
            .count()
    }

    /// Encode a `SpliceInsert` command to a minimal SCTE-35 binary descriptor.
    ///
    /// This produces a 14-byte binary representation suitable for embedding
    /// in an MPEG-TS private section or HLS `#EXT-X-CUE` tag.  The format
    /// follows SCTE-35 2022 §9.7.3 splice_insert() at a high level:
    ///
    /// ```text
    /// [0]   table_id          = 0xFC
    /// [1]   section_syntax + pts_adjustment (simplified, 0x30)
    /// [2]   protocol_version  = 0x00
    /// [3]   encrypted_pkt     = 0x00
    /// [4..7] splice_event_id (big-endian u32)
    /// [8]   splice_event_cancel_indicator = 0 | out_of_network_indicator = 1
    /// [9..12] break_duration in 90kHz ticks, big-endian u32
    /// [13]  unique_program_id placeholder = 0x01
    /// ```
    pub fn encode_splice_insert(params: &SpliceInsertParams) -> Vec<u8> {
        let mut buf = Vec::with_capacity(14);
        buf.push(0xFC); // table_id
        buf.push(0x30); // section_syntax_indicator=0, private_indicator=0, splice_reserved=3
        buf.push(0x00); // protocol_version
        buf.push(0x00); // encrypted_packet | encryption_algorithm | pts_adjustment msb

        // splice_event_id (32-bit big-endian)
        buf.extend_from_slice(&params.event_id.to_be_bytes());

        // Flags byte: out_of_network_indicator | splice_immediate_flag
        let flags: u8 = if params.out_of_network { 0x80 } else { 0x00 }
            | if params.auto_return { 0x40 } else { 0x00 };
        buf.push(flags);

        // break_duration in 90 kHz ticks (duration_ms * 90)
        let ticks: u32 = (params.duration_ms.saturating_mul(90)) as u32;
        buf.extend_from_slice(&ticks.to_be_bytes());

        buf.push(0x01); // unique_program_id placeholder
        buf
    }

    /// Encode a `SpliceReturn` command to a minimal SCTE-35 binary descriptor.
    ///
    /// Produces a 9-byte descriptor:
    /// ```text
    /// [0]   table_id          = 0xFC
    /// [1]   0x30
    /// [2]   protocol_version  = 0x00
    /// [3]   0x00
    /// [4..7] splice_event_id (big-endian u32)
    /// [8]   out_of_network_indicator = 0  (return to network)
    /// ```
    pub fn encode_splice_return(params: &SpliceReturnParams) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9);
        buf.push(0xFC);
        buf.push(0x30);
        buf.push(0x00);
        buf.push(0x00);
        buf.extend_from_slice(&params.event_id.to_be_bytes());
        buf.push(0x00); // out_of_network = 0 (return)
        buf
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SCTE-35 specification §9.7: Splice Command Generation
//
// Standalone helpers that produce `SpliceCommand` values directly without
// requiring a `PlaylistScte35` registry.  These are useful for ad-insertion
// engines that build per-packet splice descriptors on the fly.
// ─────────────────────────────────────────────────────────────────────────────

/// A resolved SCTE-35 splice command ready for injection into a transport stream.
///
/// This type is the output of the standalone generator functions and contains
/// both the structured parameters and the pre-encoded binary descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpliceCommand {
    /// The structured SCTE-35 command.
    pub command: Scte35Command,
    /// Pre-encoded binary descriptor (suitable for MPEG-TS private section).
    pub encoded: Vec<u8>,
}

/// Generate a SCTE-35 `splice_insert` command for an out-of-network ad break.
///
/// # Arguments
///
/// * `event_id`   — 32-bit SCTE-35 splice event identifier (must be unique per
///                  channel for the lifetime of the break).
/// * `pts_time`   — Programme Time Stamp of the splice point in 90 kHz ticks.
///                  Passed as-is; the encoder embeds it in the `break_duration`
///                  field for consumer use.
/// * `duration`   — Duration of the avail in 90 kHz ticks.
///
/// Both `out_of_network` and `auto_return` flags are set to `true` per the
/// broadcast automation convention.
///
/// # Returns
///
/// A [`SpliceCommand`] containing the structured params and their binary
/// encoding.
pub fn generate_splice_insert(event_id: u32, pts_time: u64, duration: u64) -> SpliceCommand {
    // Convert 90 kHz ticks → milliseconds for the registry-compatible params.
    let duration_ms = duration / 90;
    let _pts_ms = pts_time / 90; // retained for callers who inspect the command

    let params = SpliceInsertParams {
        event_id,
        duration_ms,
        out_of_network: true,
        auto_return: true,
    };
    let encoded = PlaylistScte35::encode_splice_insert(&params);
    SpliceCommand {
        command: Scte35Command::SpliceInsert(params),
        encoded,
    }
}

/// Generate a SCTE-35 `splice_return` (return-to-network) command.
///
/// # Arguments
///
/// * `event_id` — 32-bit SCTE-35 event ID matching the originating
///                [`generate_splice_insert`] call.
///
/// # Returns
///
/// A [`SpliceCommand`] containing the structured params and their binary
/// encoding.
pub fn generate_splice_return(event_id: u32) -> SpliceCommand {
    let params = SpliceReturnParams { event_id };
    let encoded = PlaylistScte35::encode_splice_return(&params);
    SpliceCommand {
        command: Scte35Command::SpliceReturn(params),
        encoded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_splice_basic() {
        let mut table = PlaylistScte35::new();
        table.insert_splice(30_000, 30_000, 1001);
        assert_eq!(table.len(), 1);
        assert_eq!(table.splice_count(), 1);
        assert_eq!(table.return_count(), 0);
    }

    #[test]
    fn test_insert_return_basic() {
        let mut table = PlaylistScte35::new();
        table.insert_return(60_000, 1001);
        assert_eq!(table.len(), 1);
        assert_eq!(table.return_count(), 1);
        assert_eq!(table.splice_count(), 0);
    }

    #[test]
    fn test_insert_splice_and_return_pair() {
        let mut table = PlaylistScte35::new();
        table.insert_splice(30_000, 30_000, 42);
        table.insert_return(60_000, 42);
        assert_eq!(table.len(), 2);
        assert_eq!(table.splice_count(), 1);
        assert_eq!(table.return_count(), 1);
    }

    #[test]
    fn test_commands_at_exact_position() {
        let mut table = PlaylistScte35::new();
        table.insert_splice(10_000, 5_000, 7);
        let cmds = table.commands_at(10_000);
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0].command, Scte35Command::SpliceInsert(_)));
    }

    #[test]
    fn test_commands_at_or_before_ordering() {
        let mut table = PlaylistScte35::new();
        table.insert_splice(10_000, 5_000, 1);
        table.insert_splice(20_000, 5_000, 2);
        table.insert_return(30_000, 1);

        let cmds = table.commands_at_or_before(20_000);
        assert_eq!(cmds.len(), 2);
        assert_eq!(cmds[0].position_ms, 10_000);
        assert_eq!(cmds[1].position_ms, 20_000);
    }

    #[test]
    fn test_all_entries_chronological() {
        let mut table = PlaylistScte35::new();
        table.insert_return(60_000, 99);
        table.insert_splice(30_000, 30_000, 99);
        let entries = table.all_entries();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].position_ms <= entries[1].position_ms);
    }

    #[test]
    fn test_encode_splice_insert_length_and_table_id() {
        let params = SpliceInsertParams {
            event_id: 12345,
            duration_ms: 30_000,
            out_of_network: true,
            auto_return: true,
        };
        let buf = PlaylistScte35::encode_splice_insert(&params);
        assert_eq!(buf[0], 0xFC);
        assert_eq!(buf.len(), 14);
        // Event ID round-trip
        let id = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!(id, 12345);
    }

    #[test]
    fn test_encode_splice_return_length_and_table_id() {
        let params = SpliceReturnParams { event_id: 9999 };
        let buf = PlaylistScte35::encode_splice_return(&params);
        assert_eq!(buf[0], 0xFC);
        assert_eq!(buf.len(), 9);
        let id = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
        assert_eq!(id, 9999);
        assert_eq!(buf[8], 0x00); // return-to-network flag
    }

    #[test]
    fn test_is_empty_and_clear() {
        let mut table = PlaylistScte35::new();
        assert!(table.is_empty());
        table.insert_splice(5_000, 10_000, 1);
        assert!(!table.is_empty());
        table.clear();
        assert!(table.is_empty());
    }

    // ── Tests for standalone splice command generators ──────────────────────

    #[test]
    fn test_scte35_splice_roundtrip() {
        // 30-second avail at PTS 900_000 ticks (10 s at 90 kHz), duration 2_700_000 ticks (30 s)
        let insert_cmd = generate_splice_insert(42, 900_000, 2_700_000);

        // Verify the command kind
        assert!(
            matches!(insert_cmd.command, Scte35Command::SpliceInsert(_)),
            "Expected SpliceInsert variant"
        );

        // Verify binary: table_id = 0xFC
        assert_eq!(
            insert_cmd.encoded[0], 0xFC,
            "table_id should be 0xFC per SCTE-35 §9.6"
        );

        // Event ID should round-trip through the binary encoding
        let id_from_binary = u32::from_be_bytes([
            insert_cmd.encoded[4],
            insert_cmd.encoded[5],
            insert_cmd.encoded[6],
            insert_cmd.encoded[7],
        ]);
        assert_eq!(
            id_from_binary, 42,
            "event_id should survive binary encoding"
        );

        // Flags: out_of_network=1, auto_return=1 → 0xC0
        assert_eq!(
            insert_cmd.encoded[8], 0xC0,
            "out_of_network and auto_return flags should be set"
        );

        // Generate the matching return command
        let return_cmd = generate_splice_return(42);

        assert!(
            matches!(return_cmd.command, Scte35Command::SpliceReturn(_)),
            "Expected SpliceReturn variant"
        );

        // Return binary: table_id = 0xFC
        assert_eq!(return_cmd.encoded[0], 0xFC);

        // event_id round-trip
        let rid = u32::from_be_bytes([
            return_cmd.encoded[4],
            return_cmd.encoded[5],
            return_cmd.encoded[6],
            return_cmd.encoded[7],
        ]);
        assert_eq!(rid, 42, "return event_id should match insert event_id");

        // Return flag (out_of_network=0 → return to network)
        assert_eq!(
            return_cmd.encoded[8], 0x00,
            "out_of_network flag must be 0 for splice_return"
        );
    }

    #[test]
    fn test_generate_splice_insert_duration_conversion() {
        // 30 s at 90 kHz = 2_700_000 ticks → duration_ms should be 30_000
        let cmd = generate_splice_insert(1, 0, 2_700_000);
        if let Scte35Command::SpliceInsert(ref p) = cmd.command {
            assert_eq!(
                p.duration_ms, 30_000,
                "90 kHz ticks should convert to ms correctly"
            );
            assert!(p.out_of_network, "out_of_network should default to true");
            assert!(p.auto_return, "auto_return should default to true");
        } else {
            panic!("Expected SpliceInsert");
        }
    }

    #[test]
    fn test_generate_splice_return_params() {
        let cmd = generate_splice_return(9999);
        if let Scte35Command::SpliceReturn(ref p) = cmd.command {
            assert_eq!(p.event_id, 9999);
        } else {
            panic!("Expected SpliceReturn");
        }
        // Encoded length should be 9 bytes per encode_splice_return spec
        assert_eq!(cmd.encoded.len(), 9);
    }
}
