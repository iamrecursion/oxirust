#![allow(dead_code)]
//! SCTE-35 ad insertion and splice point management for broadcast playout.
//!
//! Provides a splice-event model, scheduling of ad breaks, and a splice
//! decision engine that determines when to cut to/from ad content.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a splice event.
pub type SpliceId = u64;

/// Splice command type (modelled after SCTE-35).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpliceCommandType {
    /// Insert an ad break (splice out).
    SpliceInsert,
    /// Return from ad break (splice in / return).
    SpliceReturn,
    /// Cancel a previously scheduled splice.
    SpliceCancel,
    /// Time signal with segmentation descriptor.
    TimeSignal,
}

/// Status of a splice event in the timeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpliceStatus {
    /// Scheduled but not yet reached.
    Pending,
    /// Currently active (splice-out in progress).
    Active,
    /// Completed (splice-in has happened).
    Completed,
    /// Cancelled before execution.
    Cancelled,
}

/// A splice event representing one ad insertion point.
#[derive(Debug, Clone)]
pub struct SpliceEvent {
    /// Unique splice identifier.
    pub id: SpliceId,
    /// Command type.
    pub command: SpliceCommandType,
    /// Presentation time in microseconds at which the splice occurs.
    pub pts_us: i64,
    /// Duration of the ad break in microseconds (0 if unknown).
    pub duration_us: i64,
    /// Whether an auto-return is expected at pts_us + duration_us.
    pub auto_return: bool,
    /// Current status.
    pub status: SpliceStatus,
    /// Optional descriptive label.
    pub label: String,
}

/// Configuration for the ad insertion engine.
#[derive(Debug, Clone)]
pub struct AdInsertionConfig {
    /// Minimum gap (microseconds) between consecutive splice events.
    pub min_gap_us: i64,
    /// Default ad break duration (microseconds) when unspecified.
    pub default_duration_us: i64,
    /// Whether to enforce auto-return on all splice-inserts.
    pub force_auto_return: bool,
    /// Maximum number of queued splice events.
    pub max_queue_size: usize,
}

impl Default for AdInsertionConfig {
    fn default() -> Self {
        Self {
            min_gap_us: 5_000_000,           // 5 seconds
            default_duration_us: 30_000_000, // 30 seconds
            force_auto_return: true,
            max_queue_size: 256,
        }
    }
}

/// Result of a splice scheduling attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleResult {
    /// Splice was successfully scheduled.
    Scheduled,
    /// Rejected because it violates the minimum gap constraint.
    TooClose,
    /// Rejected because the queue is full.
    QueueFull,
    /// The splice PTS is in the past.
    InThePast,
}

// ---------------------------------------------------------------------------
// Ad Insertion Engine
// ---------------------------------------------------------------------------

/// Manages the lifecycle of SCTE-35-style splice events during playout.
#[derive(Debug)]
pub struct AdInsertionEngine {
    config: AdInsertionConfig,
    /// Splice events ordered by PTS.
    events: BTreeMap<i64, SpliceEvent>,
    next_id: SpliceId,
    /// Current playout PTS (updated externally).
    current_pts_us: i64,
}

impl AdInsertionEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: AdInsertionConfig) -> Self {
        Self {
            config,
            events: BTreeMap::new(),
            next_id: 1,
            current_pts_us: 0,
        }
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &AdInsertionConfig {
        &self.config
    }

    /// Update the current playout position.
    pub fn set_current_pts(&mut self, pts_us: i64) {
        self.current_pts_us = pts_us;
    }

    /// Return the current playout PTS.
    pub fn current_pts(&self) -> i64 {
        self.current_pts_us
    }

    /// Schedule a new splice-insert (ad break).
    pub fn schedule_insert(
        &mut self,
        pts_us: i64,
        duration_us: Option<i64>,
        label: &str,
    ) -> (ScheduleResult, Option<SpliceId>) {
        if pts_us < self.current_pts_us {
            return (ScheduleResult::InThePast, None);
        }
        if self.events.len() >= self.config.max_queue_size {
            return (ScheduleResult::QueueFull, None);
        }
        // Check minimum gap
        if let Some((&prev_pts, _)) = self.events.range(..pts_us).next_back() {
            if pts_us - prev_pts < self.config.min_gap_us {
                return (ScheduleResult::TooClose, None);
            }
        }
        if let Some((&next_pts, _)) = self.events.range(pts_us + 1..).next() {
            if next_pts - pts_us < self.config.min_gap_us {
                return (ScheduleResult::TooClose, None);
            }
        }

        let dur = duration_us.unwrap_or(self.config.default_duration_us);
        let id = self.next_id;
        self.next_id += 1;

        let event = SpliceEvent {
            id,
            command: SpliceCommandType::SpliceInsert,
            pts_us,
            duration_us: dur,
            auto_return: self.config.force_auto_return,
            status: SpliceStatus::Pending,
            label: label.to_string(),
        };

        self.events.insert(pts_us, event);
        (ScheduleResult::Scheduled, Some(id))
    }

    /// Cancel a splice event by its PTS.
    pub fn cancel_at(&mut self, pts_us: i64) -> bool {
        if let Some(ev) = self.events.get_mut(&pts_us) {
            if ev.status == SpliceStatus::Pending {
                ev.status = SpliceStatus::Cancelled;
                return true;
            }
        }
        false
    }

    /// Advance the engine to the given PTS, activating and completing
    /// events as needed. Returns a list of events that changed status.
    pub fn advance_to(&mut self, pts_us: i64) -> Vec<SpliceEvent> {
        self.current_pts_us = pts_us;
        let mut changed = Vec::new();

        for ev in self.events.values_mut() {
            match ev.status {
                SpliceStatus::Pending if pts_us >= ev.pts_us => {
                    ev.status = SpliceStatus::Active;
                    changed.push(ev.clone());
                }
                SpliceStatus::Active if ev.auto_return && pts_us >= ev.pts_us + ev.duration_us => {
                    ev.status = SpliceStatus::Completed;
                    changed.push(ev.clone());
                }
                _ => {}
            }
        }

        changed
    }

    /// Return the number of pending splice events.
    pub fn pending_count(&self) -> usize {
        self.events
            .values()
            .filter(|e| e.status == SpliceStatus::Pending)
            .count()
    }

    /// Return all events (regardless of status).
    pub fn all_events(&self) -> Vec<&SpliceEvent> {
        self.events.values().collect()
    }

    /// Return the next pending splice event (by PTS).
    pub fn next_pending(&self) -> Option<&SpliceEvent> {
        self.events
            .values()
            .find(|e| e.status == SpliceStatus::Pending)
    }

    /// Remove all completed and cancelled events, returning the count removed.
    pub fn purge_finished(&mut self) -> usize {
        let before = self.events.len();
        self.events.retain(|_, ev| {
            ev.status != SpliceStatus::Completed && ev.status != SpliceStatus::Cancelled
        });
        before - self.events.len()
    }
}

// ---------------------------------------------------------------------------
// SCTE-35 TS Injection
// ---------------------------------------------------------------------------

/// Error type for TS injection operations.
#[derive(Debug)]
pub enum AdInsertionError {
    /// PTS value is out of the valid 33-bit range.
    InvalidPts(String),
    /// Serialised section exceeds the maximum MPEG-TS payload size.
    SectionTooLong(usize),
    /// The supplied TS packet buffer is structurally invalid.
    InvalidPackets(String),
}

impl std::fmt::Display for AdInsertionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPts(msg) => write!(f, "invalid PTS: {}", msg),
            Self::SectionTooLong(len) => write!(f, "section too long: {} bytes", len),
            Self::InvalidPackets(msg) => write!(f, "invalid TS packets: {}", msg),
        }
    }
}

impl std::error::Error for AdInsertionError {}

/// A SCTE-35 `splice_insert()` command descriptor.
///
/// Maps to the fields defined in SCTE-35 2022, Section 9.7.3.
#[derive(Debug, Clone)]
pub struct Scte35SpliceInsert {
    /// Unique splice event identifier (32-bit unsigned).
    pub splice_event_id: u32,
    /// When `true`, the splice happens at the next possible opportunity.
    pub splice_immediate: bool,
    /// Presentation timestamp at which the splice occurs (90 kHz clock).
    /// Must be `Some` when `splice_immediate` is `false`.
    pub pts_time: Option<u64>,
    /// Duration of the ad break in 90 kHz ticks.
    pub duration: Option<u64>,
    /// When `true`, playback returns to the network after `duration` ticks.
    pub auto_return: bool,
    /// Program identifier for the break.
    pub unique_program_id: u16,
    /// Avail number within the commercial opportunity.
    pub avail_num: u8,
    /// Total number of avails in the commercial opportunity.
    pub avails_expected: u8,
    /// `true` = splice out (ad start), `false` = splice in (return).
    pub out_of_network: bool,
}

/// A binary MPEG SCTE-35 section header (simplified).
#[derive(Debug, Clone)]
pub struct Scte35Section {
    /// Always `0xFC` for SCTE-35.
    pub table_id: u8,
    /// Byte length of the section following the `section_length` field.
    pub section_length: u16,
    /// Protocol version — currently always `0`.
    pub protocol_version: u8,
    /// Whether the packet payload is encrypted.
    pub encrypted_packet: bool,
    /// 33-bit PTS adjustment value.
    pub pts_adjustment: u64,
    /// Encryption key index.
    pub cw_index: u8,
    /// 12-bit tier value.
    pub tier: u16,
    /// Byte length of the splice command.
    pub splice_command_length: u16,
    /// Splice command type byte (0x05 = SpliceInsert).
    pub splice_command_type: u8,
    /// Raw splice command bytes.
    pub payload: Vec<u8>,
}

/// Injects SCTE-35 splice cues into MPEG-TS packet streams.
#[derive(Debug, Clone)]
pub struct Scte35Injector {
    /// PID to use for the injected SCTE-35 elementary stream.
    pub pid: u16,
    /// Rolling continuity counter (4-bit, wraps at 16).
    pub continuity_counter: u8,
}

impl Scte35Injector {
    /// Creates a new injector for the given PID.
    pub fn new(pid: u16) -> Self {
        Self {
            pid,
            continuity_counter: 0,
        }
    }

    /// Serialises a `Scte35SpliceInsert` into a minimal SCTE-35 binary section.
    ///
    /// The returned bytes are the *entire* SCTE-35 private section (starting
    /// with `table_id` = `0xFC`), ready to be placed in a TS payload.
    pub fn serialize_splice_insert(cue: &Scte35SpliceInsert) -> Vec<u8> {
        let mut cmd: Vec<u8> = Vec::with_capacity(20);

        // Byte 0: splice_event_id (32-bit big-endian)
        cmd.extend_from_slice(&cue.splice_event_id.to_be_bytes());

        // Byte 4: splice_event_cancel_indicator (1 bit) | reserved (7 bits)
        cmd.push(0x00); // event not cancelled

        // Byte 5: out_of_network_indicator (1 bit) | program_splice_flag (1 bit)
        //         | duration_flag (1 bit) | splice_immediate_flag (1 bit) | reserved (4 bits)
        let out_flag: u8 = if cue.out_of_network { 0x80 } else { 0x00 };
        let program_splice: u8 = 0x40; // always use program splice
        let dur_flag: u8 = if cue.duration.is_some() { 0x20 } else { 0x00 };
        let imm_flag: u8 = if cue.splice_immediate { 0x10 } else { 0x00 };
        cmd.push(out_flag | program_splice | dur_flag | imm_flag);

        // splice_time (conditional): 5-byte field if !splice_immediate
        if !cue.splice_immediate {
            if let Some(pts) = cue.pts_time {
                // time_specified_flag (1 bit) | reserved (6 bits) | pts_time[32:30] (1 bit)
                // pts_time[29:0] (30 bits) packed as 4 bytes → total 5 bytes
                let pts_33 = pts & 0x1_FFFF_FFFF; // 33-bit mask
                let b0: u8 = 0x80 | (((pts_33 >> 32) & 0x01) as u8); // time_specified_flag | top bit
                let b1: u8 = ((pts_33 >> 24) & 0xFF) as u8;
                let b2: u8 = ((pts_33 >> 16) & 0xFF) as u8;
                let b3: u8 = ((pts_33 >> 8) & 0xFF) as u8;
                let b4: u8 = (pts_33 & 0xFF) as u8;
                cmd.extend_from_slice(&[b0, b1, b2, b3, b4]);
            } else {
                // time_specified_flag = 0 → only 1 byte
                cmd.push(0x00);
            }
        }

        // break_duration (conditional): 5 bytes if duration_flag set
        if let Some(dur) = cue.duration {
            let auto_ret: u8 = if cue.auto_return { 0x80 } else { 0x00 };
            let dur_33 = dur & 0x1_FFFF_FFFF;
            let b0: u8 = auto_ret | (((dur_33 >> 32) & 0x01) as u8);
            let b1: u8 = ((dur_33 >> 24) & 0xFF) as u8;
            let b2: u8 = ((dur_33 >> 16) & 0xFF) as u8;
            let b3: u8 = ((dur_33 >> 8) & 0xFF) as u8;
            let b4: u8 = (dur_33 & 0xFF) as u8;
            cmd.extend_from_slice(&[b0, b1, b2, b3, b4]);
        }

        // unique_program_id (16-bit)
        cmd.extend_from_slice(&cue.unique_program_id.to_be_bytes());
        // avail_num + avails_expected
        cmd.push(cue.avail_num);
        cmd.push(cue.avails_expected);

        // Build the full section:
        // table_id(1) + section_syntax_indicator+private_indicator+reserved+section_length(3)
        // + protocol_version(1) + encrypted_packet+encryption_algorithm+pts_adjustment(8)
        // + cw_index(1) + tier+splice_command_length(3) + splice_command_type(1)
        // + splice_command(n) + CRC32(4)
        let cmd_len = cmd.len() as u16;
        // section_length = from protocol_version to end (including CRC32)
        // fixed fields after section_length: 8 bytes header + 1 type + cmd_len + 2 (descriptor_loop_length) + 4 CRC
        let section_len: u16 = 8 + 1 + cmd_len + 2 + 4; // minimal

        let mut sec: Vec<u8> = Vec::with_capacity(3 + section_len as usize);
        sec.push(0xFC); // table_id
                        // section_syntax_indicator(0) | private_indicator(0) | reserved(11) | section_length(13)
        let sl_hi = (0xC0 | ((section_len >> 8) & 0x0F)) as u8; // reserved bits set
        let sl_lo = (section_len & 0xFF) as u8;
        sec.push(sl_hi);
        sec.push(sl_lo);

        // protocol_version (8)
        sec.push(0x00);

        // encrypted_packet(1) | encryption_algorithm(6) | pts_adjustment[32:30](1) -- 8 bytes total
        // We use 0 pts_adjustment for simplicity.
        sec.push(0x00); // encrypted=0, algorithm=0, pts_adj top bit = 0
        sec.push(0x00); // pts_adjustment bytes 2-5
        sec.push(0x00);
        sec.push(0x00);
        sec.push(0x00);

        // cw_index (8)
        sec.push(0xFF);

        // tier(12) | splice_command_length(12) packed as 3 bytes
        let tier: u16 = 0xFFF; // all 12 bits set (default)
        let scl: u16 = cmd_len;
        let b0 = ((tier >> 4) & 0xFF) as u8;
        let b1 = (((tier & 0xF) << 4) | ((scl >> 8) & 0xF)) as u8;
        let b2 = (scl & 0xFF) as u8;
        sec.push(b0);
        sec.push(b1);
        sec.push(b2);

        // splice_command_type = 0x05 (splice_insert)
        sec.push(0x05);

        // splice_command bytes
        sec.extend_from_slice(&cmd);

        // descriptor_loop_length = 0 (no descriptors)
        sec.push(0x00);
        sec.push(0x00);

        // CRC32 (MPEG-2 CRC32) — placeholder zeros (not validated by all decoders)
        let crc = mpeg_crc32(&sec);
        sec.extend_from_slice(&crc.to_be_bytes());

        sec
    }

    /// Wraps a binary SCTE-35 section into one or more 188-byte MPEG-TS packets.
    ///
    /// The first packet has `payload_unit_start_indicator` = 1 and a pointer
    /// field of `0x00`. Remaining bytes are split across continuation packets.
    /// Unused payload bytes are padded with `0xFF`.
    pub fn build_ts_packets(&mut self, section: &[u8]) -> Vec<[u8; 188]> {
        let mut packets: Vec<[u8; 188]> = Vec::new();
        let pid = self.pid;
        let mut remaining = section;
        let mut first = true;

        while !remaining.is_empty() || first {
            let mut pkt = [0xFFu8; 188];
            pkt[0] = 0x47; // sync byte

            let pusi: u8 = if first { 0x40 } else { 0x00 };
            let pid_hi = (pusi | 0x00 | ((pid >> 8) as u8 & 0x1F)) as u8;
            let pid_lo = (pid & 0xFF) as u8;
            pkt[1] = pid_hi;
            pkt[2] = pid_lo;
            // adaptation_field_control = 0b01 (payload only) | continuity_counter
            pkt[3] = 0x10 | (self.continuity_counter & 0x0F);
            self.continuity_counter = self.continuity_counter.wrapping_add(1) & 0x0F;

            let payload_start = 4usize;
            let mut offset = payload_start;

            if first {
                pkt[offset] = 0x00; // pointer_field = 0
                offset += 1;
                first = false;
            }

            let space = 188 - offset;
            let take = space.min(remaining.len());
            pkt[offset..offset + take].copy_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            // remainder of packet is already 0xFF padding
            packets.push(pkt);
        }

        packets
    }

    /// Injects SCTE-35 TS packets into the beginning of `ts_packets`.
    ///
    /// Returns the number of SCTE-35 packets prepended.
    ///
    /// # Errors
    ///
    /// - [`AdInsertionError::SectionTooLong`] if the serialised section exceeds
    ///   a practical limit (64 KB).
    pub fn inject_into_ts(
        &mut self,
        ts_packets: &mut Vec<[u8; 188]>,
        cue: &Scte35SpliceInsert,
    ) -> Result<usize, AdInsertionError> {
        let section = Self::serialize_splice_insert(cue);
        if section.len() > 65_535 {
            return Err(AdInsertionError::SectionTooLong(section.len()));
        }
        let scte_packets = self.build_ts_packets(&section);
        let count = scte_packets.len();
        // Prepend: reverse-insert at index 0 to preserve order.
        for pkt in scte_packets.into_iter().rev() {
            ts_packets.insert(0, pkt);
        }
        Ok(count)
    }
}

/// Compute MPEG-2 / DVB CRC-32 over `data`.
///
/// Polynomial: 0x04C11DB7 (normal/MSB-first form).
fn mpeg_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        for i in (0..8).rev() {
            let bit = (byte >> i) & 1;
            let msb = (crc >> 31) & 1;
            crc <<= 1;
            if msb ^ (bit as u32) != 0 {
                crc ^= 0x04C1_1DB7;
            }
        }
    }
    crc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = AdInsertionConfig::default();
        assert_eq!(cfg.min_gap_us, 5_000_000);
        assert!(cfg.force_auto_return);
    }

    #[test]
    fn test_schedule_and_count() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        let (res, id) = eng.schedule_insert(10_000_000, None, "ad1");
        assert_eq!(res, ScheduleResult::Scheduled);
        assert!(id.is_some());
        assert_eq!(eng.pending_count(), 1);
    }

    #[test]
    fn test_reject_past_pts() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        eng.set_current_pts(100_000_000);
        let (res, _) = eng.schedule_insert(50_000_000, None, "old");
        assert_eq!(res, ScheduleResult::InThePast);
    }

    #[test]
    fn test_reject_too_close() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        eng.schedule_insert(10_000_000, None, "a");
        let (res, _) = eng.schedule_insert(11_000_000, None, "b");
        assert_eq!(res, ScheduleResult::TooClose);
    }

    #[test]
    fn test_queue_full() {
        let cfg = AdInsertionConfig {
            max_queue_size: 2,
            min_gap_us: 0,
            ..AdInsertionConfig::default()
        };
        let mut eng = AdInsertionEngine::new(cfg);
        eng.schedule_insert(10_000_000, None, "a");
        eng.schedule_insert(20_000_000, None, "b");
        let (res, _) = eng.schedule_insert(30_000_000, None, "c");
        assert_eq!(res, ScheduleResult::QueueFull);
    }

    #[test]
    fn test_advance_activates() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        eng.schedule_insert(10_000_000, Some(5_000_000), "x");
        let changed = eng.advance_to(10_000_000);
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].status, SpliceStatus::Active);
    }

    #[test]
    fn test_advance_completes_auto_return() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        eng.schedule_insert(10_000_000, Some(5_000_000), "x");
        eng.advance_to(10_000_000); // activate
        let changed = eng.advance_to(15_000_000); // complete
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].status, SpliceStatus::Completed);
    }

    #[test]
    fn test_cancel_pending() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        eng.schedule_insert(10_000_000, None, "c");
        assert!(eng.cancel_at(10_000_000));
        assert_eq!(eng.pending_count(), 0);
    }

    #[test]
    fn test_cancel_non_existent() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        assert!(!eng.cancel_at(99_000_000));
    }

    #[test]
    fn test_purge_finished() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        eng.schedule_insert(10_000_000, Some(5_000_000), "a");
        eng.advance_to(10_000_000);
        eng.advance_to(15_000_000);
        let removed = eng.purge_finished();
        assert_eq!(removed, 1);
        assert!(eng.all_events().is_empty());
    }

    #[test]
    fn test_next_pending() {
        let cfg = AdInsertionConfig {
            min_gap_us: 0,
            ..AdInsertionConfig::default()
        };
        let mut eng = AdInsertionEngine::new(cfg);
        eng.schedule_insert(20_000_000, None, "later");
        eng.schedule_insert(10_000_000, None, "sooner");
        let next = eng.next_pending().expect("should succeed in test");
        assert_eq!(next.pts_us, 10_000_000);
    }

    #[test]
    fn test_all_events_returns_all() {
        let cfg = AdInsertionConfig {
            min_gap_us: 0,
            ..AdInsertionConfig::default()
        };
        let mut eng = AdInsertionEngine::new(cfg);
        eng.schedule_insert(10_000_000, None, "a");
        eng.schedule_insert(20_000_000, None, "b");
        assert_eq!(eng.all_events().len(), 2);
    }

    #[test]
    fn test_current_pts_accessor() {
        let mut eng = AdInsertionEngine::new(AdInsertionConfig::default());
        eng.set_current_pts(42_000);
        assert_eq!(eng.current_pts(), 42_000);
    }
}

// ---------------------------------------------------------------------------
// SCTE-35 Injector tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod scte35_tests {
    use super::*;

    fn sample_cue(out_of_network: bool) -> Scte35SpliceInsert {
        Scte35SpliceInsert {
            splice_event_id: 0x0001_0001,
            splice_immediate: false,
            pts_time: Some(900_000),   // 10 s at 90 kHz
            duration: Some(2_700_000), // 30 s at 90 kHz
            auto_return: true,
            unique_program_id: 1,
            avail_num: 1,
            avails_expected: 2,
            out_of_network,
        }
    }

    fn immediate_cue() -> Scte35SpliceInsert {
        Scte35SpliceInsert {
            splice_event_id: 0x0000_0001,
            splice_immediate: true,
            pts_time: None,
            duration: None,
            auto_return: false,
            unique_program_id: 0,
            avail_num: 0,
            avails_expected: 0,
            out_of_network: true,
        }
    }

    // ── serialize_splice_insert ──────────────────────────────────────────────

    #[test]
    fn test_serialize_returns_nonempty() {
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(true));
        assert!(!section.is_empty());
    }

    #[test]
    fn test_serialize_starts_with_fc_table_id() {
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(true));
        assert_eq!(section[0], 0xFC, "table_id must be 0xFC");
    }

    #[test]
    fn test_serialize_immediate_cue() {
        let section = Scte35Injector::serialize_splice_insert(&immediate_cue());
        assert!(!section.is_empty());
        assert_eq!(section[0], 0xFC);
    }

    #[test]
    fn test_serialize_out_of_network_true() {
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(true));
        // At minimum, the section must be well-formed (non-trivial length)
        assert!(section.len() > 10);
    }

    #[test]
    fn test_serialize_out_of_network_false() {
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(false));
        assert!(section.len() > 10);
    }

    #[test]
    fn test_serialize_splice_insert_has_splice_command_type_05() {
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(true));
        // Layout: table_id(1) + section_length(2) + protocol_version(1)
        // + encrypted+pts_adj(1) + pts_adj(4) + cw_index(1) + tier+cmd_len(3)
        // = 13 bytes → splice_command_type at index 13
        assert_eq!(section[13], 0x05, "splice_command_type should be 0x05");
    }

    // ── build_ts_packets ──────────────────────────────────────────────────────

    #[test]
    fn test_build_ts_packets_each_is_188_bytes() {
        let mut injector = Scte35Injector::new(0x0200);
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(true));
        let packets = injector.build_ts_packets(&section);
        for pkt in &packets {
            assert_eq!(pkt.len(), 188, "each TS packet must be 188 bytes");
        }
    }

    #[test]
    fn test_build_ts_packets_sync_byte_0x47() {
        let mut injector = Scte35Injector::new(0x0200);
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(true));
        let packets = injector.build_ts_packets(&section);
        for pkt in &packets {
            assert_eq!(pkt[0], 0x47, "TS sync byte must be 0x47");
        }
    }

    #[test]
    fn test_build_ts_packets_pid_encoded() {
        let pid: u16 = 0x0200;
        let mut injector = Scte35Injector::new(pid);
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(true));
        let packets = injector.build_ts_packets(&section);
        let first = &packets[0];
        let encoded_pid = ((first[1] as u16 & 0x1F) << 8) | first[2] as u16;
        assert_eq!(
            encoded_pid, pid,
            "PID must be correctly encoded in bytes 1-2"
        );
    }

    #[test]
    fn test_build_ts_packets_first_has_pusi() {
        let mut injector = Scte35Injector::new(0x0200);
        let section = Scte35Injector::serialize_splice_insert(&sample_cue(true));
        let packets = injector.build_ts_packets(&section);
        // PUSI flag is bit 6 of byte 1
        assert!(packets[0][1] & 0x40 != 0, "first packet must have PUSI set");
    }

    #[test]
    fn test_build_ts_packets_continuity_counter_increments() {
        let mut injector = Scte35Injector::new(0x0200);
        // Build enough data to require 2+ packets
        let large_section = vec![0xABu8; 400];
        let packets = injector.build_ts_packets(&large_section);
        assert!(packets.len() >= 2);
        let cc0 = packets[0][3] & 0x0F;
        let cc1 = packets[1][3] & 0x0F;
        assert_eq!(cc1, (cc0 + 1) & 0x0F);
    }

    // ── inject_into_ts ────────────────────────────────────────────────────────

    #[test]
    fn test_inject_into_ts_inserts_at_front() {
        let mut injector = Scte35Injector::new(0x0200);
        // Create a fake TS packet with sync byte 0x47 and PID 0x0100
        let mut fake_pkt = [0u8; 188];
        fake_pkt[0] = 0x47;
        fake_pkt[1] = 0x01;
        fake_pkt[2] = 0x00;
        let mut ts = vec![fake_pkt];

        injector
            .inject_into_ts(&mut ts, &sample_cue(true))
            .expect("inject should succeed");

        // The injected SCTE-35 packets come first; original is at the end.
        let last = ts.last().expect("should have packets");
        assert_eq!(last[1], 0x01); // original PID high byte
    }

    #[test]
    fn test_inject_into_ts_returns_count() {
        let mut injector = Scte35Injector::new(0x0200);
        let mut ts: Vec<[u8; 188]> = Vec::new();
        let count = injector
            .inject_into_ts(&mut ts, &sample_cue(true))
            .expect("inject should succeed");
        assert!(count >= 1, "at least 1 SCTE-35 packet must be injected");
        assert_eq!(
            ts.len(),
            count,
            "ts length must match return count for empty input"
        );
    }

    #[test]
    fn test_inject_into_empty_ts_vector() {
        let mut injector = Scte35Injector::new(0x0200);
        let mut ts: Vec<[u8; 188]> = Vec::new();
        let result = injector.inject_into_ts(&mut ts, &immediate_cue());
        assert!(result.is_ok());
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_inject_scte35_packets_have_correct_pid() {
        let pid: u16 = 0x01F0;
        let mut injector = Scte35Injector::new(pid);
        let mut ts: Vec<[u8; 188]> = Vec::new();
        let count = injector
            .inject_into_ts(&mut ts, &sample_cue(false))
            .expect("inject should succeed");
        for i in 0..count {
            let encoded_pid = ((ts[i][1] as u16 & 0x1F) << 8) | ts[i][2] as u16;
            assert_eq!(encoded_pid, pid);
        }
    }

    #[test]
    fn test_ad_insertion_error_display() {
        let e = AdInsertionError::SectionTooLong(70_000);
        assert!(e.to_string().contains("70000"));
        let e2 = AdInsertionError::InvalidPts("bad pts".to_string());
        assert!(e2.to_string().contains("bad pts"));
        let e3 = AdInsertionError::InvalidPackets("oops".to_string());
        assert!(e3.to_string().contains("oops"));
    }
}
