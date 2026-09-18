//! SCTE-35 marker generation for commercial breaks.
//!
//! Implements the full SCTE-35 2019 binary encoding for `splice_insert`,
//! `time_signal`, and `splice_null` commands, including:
//!
//! - Section header (`table_id = 0xFC`, CRC-32 per MPEG-2)
//! - splice_info_section fields (protocol_version, pts_adjustment, tier)
//! - `splice_command_type` byte and command-specific payload
//! - Descriptor loop serialisation
//! - Trailing CRC-32 (ANSI/SCTE 35 §9.6)
//!
//! # 90 kHz PTS ticks
//!
//! SCTE-35 timestamps use 90 kHz clock ticks (33-bit, masked to `0x1_FFFF_FFFF`).
//! Helper [`duration_to_pts`] converts a [`std::time::Duration`] to ticks.
//!
//! # Binary layout (simplified)
//!
//! ```text
//! splice_info_section() {
//!   table_id                        8   uimsbf  0xFC
//!   section_syntax_indicator        1   bslbf   0
//!   private_indicator               1   bslbf   0
//!   reserved                        2   bslbf   11
//!   section_length                  12  uimsbf  (varies)
//!   protocol_version                8   uimsbf  0x00
//!   encrypted_packet                1   bslbf   0
//!   encryption_algorithm            6   uimsbf  0
//!   pts_adjustment                  33  uimsbf  0
//!   cw_index                        8   uimsbf  0xFF
//!   tier                            12  uimsbf  (0xFFF = unrestricted)
//!   splice_command_length           12  uimsbf  (varies)
//!   splice_command_type             8   uimsbf
//!   [splice_command()]
//!   descriptor_loop_length          16  uimsbf
//!   [splice_descriptor() ...]
//!   E_CRC_32                        32  rpchof  (present when encrypted)
//!   CRC_32                          32  rpchof
//! }
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use serde::{Deserialize, Serialize};
use std::time::Duration;

// ── PTS helpers ───────────────────────────────────────────────────────────────

/// 90 kHz clock ticks per second.
const PTS_HZ: u64 = 90_000;

/// 33-bit PTS mask (SCTE-35 §9).
const PTS_MASK: u64 = 0x1_FFFF_FFFF;

/// Convert a [`Duration`] to 90 kHz PTS ticks (33-bit, wraps at 2^33).
#[must_use]
pub fn duration_to_pts(d: Duration) -> u64 {
    let ticks = d.as_secs() * PTS_HZ + (u64::from(d.subsec_millis()) * PTS_HZ / 1_000);
    ticks & PTS_MASK
}

/// Convert 90 kHz PTS ticks back to a [`Duration`].
#[must_use]
pub fn pts_to_duration(pts: u64) -> Duration {
    let millis = (pts & PTS_MASK) * 1_000 / PTS_HZ;
    Duration::from_millis(millis)
}

// ── CRC-32 (MPEG-2 / ISO 13818-1) ────────────────────────────────────────────

/// MPEG-2 CRC-32 polynomial: `0x04C11DB7`.
const CRC32_POLY: u32 = 0x04C11_DB7;

/// Compute an MPEG-2 CRC-32 over `data`.
#[must_use]
fn crc32_mpeg2(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        for bit in (0..8).rev() {
            let b = u32::from((byte >> bit) & 1);
            if (crc >> 31) ^ b != 0 {
                crc = (crc << 1) ^ CRC32_POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

// ── Break duration encoding ───────────────────────────────────────────────────

/// Encode a break duration into SCTE-35 `break_duration()` structure (6 bytes):
///
/// ```text
/// auto_return      1 bslbf
/// reserved         6 bslbf  111111
/// duration         33 uimsbf  (90 kHz ticks, masked to 33 bits)
/// ```
fn encode_break_duration(d: Duration, auto_return: bool) -> [u8; 5] {
    let pts = duration_to_pts(d);
    // Pack: [auto_return(1) | reserved(6) | pts_bit32(1)] [pts_bits31-24(8)] ... [pts_bits7-0(8)]
    let pts33 = pts & PTS_MASK;
    let byte0: u8 = if auto_return { 0xFE } else { 0x7E } // bit7=auto_return, bits6..1=111111, bit0=pts[32]
        | ((pts33 >> 32) as u8 & 0x01);
    let byte1: u8 = ((pts33 >> 24) & 0xFF) as u8;
    let byte2: u8 = ((pts33 >> 16) & 0xFF) as u8;
    let byte3: u8 = ((pts33 >> 8) & 0xFF) as u8;
    let byte4: u8 = (pts33 & 0xFF) as u8;
    [byte0, byte1, byte2, byte3, byte4]
}

/// Encode a `splice_time()` (conditional, 5 bytes when `time_specified_flag=1`).
///
/// ```text
/// time_specified_flag  1 bslbf
/// if (time_specified_flag == 1) {
///   reserved           6 bslbf  111110
///   pts_time           33 uimsbf
/// } else {
///   reserved           7 bslbf
/// }
/// ```
fn encode_splice_time(pts: u64) -> [u8; 5] {
    let pts33 = pts & PTS_MASK;
    let byte0: u8 = 0xFE | ((pts33 >> 32) as u8 & 0x01); // flag=1, reserved=111110, pts[32]
    let byte1: u8 = ((pts33 >> 24) & 0xFF) as u8;
    let byte2: u8 = ((pts33 >> 16) & 0xFF) as u8;
    let byte3: u8 = ((pts33 >> 8) & 0xFF) as u8;
    let byte4: u8 = (pts33 & 0xFF) as u8;
    [byte0, byte1, byte2, byte3, byte4]
}

// ── SCTE-35 command type ──────────────────────────────────────────────────────

/// SCTE-35 `splice_command_type` byte values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpliceCommandType {
    /// `splice_null()`.
    SpliceNull = 0x00,
    /// `splice_schedule()`.
    SpliceSchedule = 0x04,
    /// `splice_insert()`.
    SpliceInsert = 0x05,
    /// `time_signal()`.
    TimeSignal = 0x06,
    /// `bandwidth_reservation()`.
    BandwidthReservation = 0x07,
    /// Private command.
    PrivateCommand = 0xFF,
}

// ── SCTE-35 command type ──────────────────────────────────────────────────────

/// SCTE-35 command type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Scte35Command {
    /// Splice insert command.
    SpliceInsert {
        /// Unique splice event ID.
        event_id: u32,
        /// Whether this is an immediate splice (no PTS time given).
        immediate: bool,
        /// Pre-roll time before the splice point.  Encoded as a `splice_time()`.
        pre_roll: Option<Duration>,
        /// Break duration.  Encoded as a `break_duration()`.
        duration: Option<Duration>,
        /// If `true`, the break auto-returns at the end of `duration`.
        auto_return: bool,
        /// `out_of_network_indicator` flag.
        out_of_network: bool,
        /// Program-level splice (vs. component-level).
        program_splice: bool,
    },

    /// Time signal command.
    TimeSignal {
        /// PTS time value (90 kHz ticks).
        pts_time: u64,
    },

    /// Splice schedule command.
    SpliceSchedule {
        /// Splice event ID.
        event_id: u32,
        /// Scheduled splice time (wall-clock).
        splice_time: Duration,
    },

    /// Splice null (no-op / heartbeat).
    SpliceNull,
}

impl Scte35Command {
    /// Return the `splice_command_type` byte for this command.
    #[must_use]
    pub fn command_type_byte(&self) -> u8 {
        match self {
            Self::SpliceNull => SpliceCommandType::SpliceNull as u8,
            Self::SpliceInsert { .. } => SpliceCommandType::SpliceInsert as u8,
            Self::TimeSignal { .. } => SpliceCommandType::TimeSignal as u8,
            Self::SpliceSchedule { .. } => SpliceCommandType::SpliceSchedule as u8,
        }
    }

    /// Encode the command-specific payload bytes (everything after the
    /// `splice_command_type` byte and before the descriptor loop).
    #[must_use]
    pub fn encode_payload(&self) -> Vec<u8> {
        match self {
            Self::SpliceNull => Vec::new(),

            Self::TimeSignal { pts_time } => encode_splice_time(*pts_time).to_vec(),

            Self::SpliceSchedule {
                event_id,
                splice_time,
            } => {
                // splice_schedule() payload — simplified single-event form:
                // splice_count(8) event_id(32) cancel_indicator(1) reserved(7)
                // out_of_network(1) program_splice(1) duration(1) … utc_splice_time(32)
                let mut buf = Vec::with_capacity(14);
                buf.push(1u8); // splice_count = 1
                buf.extend_from_slice(&event_id.to_be_bytes());
                buf.push(0x00); // cancel_indicator=0, reserved=0x7F
                                // program_splice=1, duration_flag=0, reserved=0
                buf.push(0x80);
                // utc_splice_time (seconds since GPS epoch approximated as Unix seconds)
                let utc = splice_time.as_secs() as u32;
                buf.extend_from_slice(&utc.to_be_bytes());
                buf.push(0x00); // unique_program_id (2 bytes)
                buf.push(0x00);
                buf.push(0x00); // avail_num
                buf.push(0x00); // avails_expected
                buf
            }

            Self::SpliceInsert {
                event_id,
                immediate,
                pre_roll,
                duration,
                auto_return,
                out_of_network,
                program_splice,
            } => encode_splice_insert(
                *event_id,
                *immediate,
                *pre_roll,
                *duration,
                *auto_return,
                *out_of_network,
                *program_splice,
            ),
        }
    }
}

/// Encode the `splice_insert()` payload according to ANSI/SCTE 35 §9.7.3.
///
/// ```text
/// splice_insert() {
///   splice_event_id                 32  uimsbf
///   splice_event_cancel_indicator   1   bslbf   0
///   reserved                        7   bslbf   0xFF
///   out_of_network_indicator        1   bslbf
///   program_splice_flag             1   bslbf   1
///   duration_flag                   1   bslbf
///   splice_immediate_flag           1   bslbf
///   reserved                        4   bslbf   1111
///   if (splice_immediate_flag == 0 && program_splice_flag == 1) {
///     splice_time()                 5 bytes (when time_specified_flag=1)
///   }
///   if (duration_flag == 1) {
///     break_duration()              5 bytes
///   }
///   unique_program_id               16  uimsbf  0
///   avail_num                       8   uimsbf  0
///   avails_expected                 8   uimsbf  0
/// }
/// ```
#[allow(clippy::fn_params_excessive_bools)]
fn encode_splice_insert(
    event_id: u32,
    immediate: bool,
    pre_roll: Option<Duration>,
    duration: Option<Duration>,
    auto_return: bool,
    out_of_network: bool,
    program_splice: bool,
) -> Vec<u8> {
    let duration_flag = duration.is_some();

    let mut buf = Vec::with_capacity(20);

    // splice_event_id (32 bits)
    buf.extend_from_slice(&event_id.to_be_bytes());

    // splice_event_cancel_indicator(1) | reserved(7)
    buf.push(0x7F); // cancel=0, reserved=all-ones

    // out_of_network_indicator(1) | program_splice_flag(1) |
    // duration_flag(1) | splice_immediate_flag(1) | reserved(4)
    let flags: u8 = (if out_of_network { 0x80 } else { 0x00 })
        | (if program_splice { 0x40 } else { 0x00 })
        | (if duration_flag { 0x20 } else { 0x00 })
        | (if immediate { 0x10 } else { 0x00 })
        | 0x0F; // reserved bits = 1111
    buf.push(flags);

    // splice_time() when not immediate and program_splice
    if !immediate && program_splice {
        let pts = pre_roll.map_or(0, duration_to_pts);
        buf.extend_from_slice(&encode_splice_time(pts));
    }

    // break_duration()
    if let Some(dur) = duration {
        buf.extend_from_slice(&encode_break_duration(dur, auto_return));
    }

    // unique_program_id (16), avail_num (8), avails_expected (8)
    buf.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    buf
}

// ── SCTE-35 descriptor ────────────────────────────────────────────────────────

/// SCTE-35 descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scte35Descriptor {
    /// Descriptor tag.
    pub tag: u8,
    /// Raw descriptor data (after the `identifier` field for splice descriptors).
    pub data: Vec<u8>,
}

impl Scte35Descriptor {
    /// Creates a segmentation descriptor (tag = 0x02).
    ///
    /// Encodes a minimal `segmentation_descriptor()` with `segmentation_event_id`
    /// and `segmentation_type_id`.
    #[must_use]
    pub fn segmentation(segmentation_event_id: u32, type_id: u8) -> Self {
        let mut data = Vec::new();
        // identifier = CUEI (0x43554549) per SCTE-35 §10.3.3
        data.extend_from_slice(b"CUEI");
        // segmentation_event_id (32 bits)
        data.extend_from_slice(&segmentation_event_id.to_be_bytes());
        // segmentation_event_cancel_indicator(1) | reserved(7)
        data.push(0x7F);
        // program_segmentation_flag(1) | segmentation_duration_flag(0) |
        // delivery_not_restricted_flag(1) | reserved(5)
        data.push(0xA0);
        // segmentation_upid_type(8) = 0x00 (no UPID)
        data.push(0x00);
        // segmentation_upid_length(8) = 0
        data.push(0x00);
        // segmentation_type_id(8)
        data.push(type_id);
        // segment_num(8) = 0
        data.push(0x00);
        // segments_expected(8) = 0
        data.push(0x00);

        Self { tag: 0x02, data }
    }

    /// Encode the descriptor as bytes:
    /// `descriptor_tag(8) | descriptor_length(8) | data[..]`
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(2 + self.data.len());
        buf.push(self.tag);
        buf.push(self.data.len() as u8);
        buf.extend_from_slice(&self.data);
        buf
    }
}

// ── Scte35Marker ──────────────────────────────────────────────────────────────

/// SCTE-35 marker for signaling commercial breaks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scte35Marker {
    /// Command to execute.
    pub command: Scte35Command,
    /// Descriptor tags (optional metadata).
    pub descriptors: Vec<Scte35Descriptor>,
    /// Tier value for authorization (`0xFFF` = unrestricted).
    pub tier: u16,
    /// Whether this is a network segmentation.
    pub is_network_segment: bool,
    /// PTS adjustment value (33-bit, 90 kHz ticks).
    pub pts_adjustment: u64,
}

impl Scte35Marker {
    /// Creates a simple splice insert marker.
    #[must_use]
    pub fn splice_insert(event_id: u32, duration: Option<Duration>) -> Self {
        Self {
            command: Scte35Command::SpliceInsert {
                event_id,
                immediate: false,
                pre_roll: Some(Duration::from_secs(2)),
                duration,
                auto_return: true,
                out_of_network: true,
                program_splice: true,
            },
            descriptors: Vec::new(),
            tier: 0xFFF,
            is_network_segment: false,
            pts_adjustment: 0,
        }
    }

    /// Creates a time signal marker.
    #[must_use]
    pub fn time_signal(pts_time: u64) -> Self {
        Self {
            command: Scte35Command::TimeSignal { pts_time },
            descriptors: Vec::new(),
            tier: 0xFFF,
            is_network_segment: false,
            pts_adjustment: 0,
        }
    }

    /// Creates an immediate splice marker.
    #[must_use]
    pub fn immediate_splice(event_id: u32, duration: Option<Duration>) -> Self {
        Self {
            command: Scte35Command::SpliceInsert {
                event_id,
                immediate: true,
                pre_roll: None,
                duration,
                auto_return: true,
                out_of_network: true,
                program_splice: true,
            },
            descriptors: Vec::new(),
            tier: 0xFFF,
            is_network_segment: false,
            pts_adjustment: 0,
        }
    }

    /// Creates a splice null marker (heartbeat / no-op).
    #[must_use]
    pub fn splice_null() -> Self {
        Self {
            command: Scte35Command::SpliceNull,
            descriptors: Vec::new(),
            tier: 0xFFF,
            is_network_segment: false,
            pts_adjustment: 0,
        }
    }

    /// Adds a descriptor to this marker.
    pub fn add_descriptor(&mut self, descriptor: Scte35Descriptor) {
        self.descriptors.push(descriptor);
    }

    /// Sets the tier value.
    #[must_use]
    pub const fn with_tier(mut self, tier: u16) -> Self {
        self.tier = tier & 0xFFF;
        self
    }

    /// Marks this as a network segment.
    #[must_use]
    pub const fn as_network_segment(mut self) -> Self {
        self.is_network_segment = true;
        self
    }

    /// Sets the PTS adjustment (33-bit, 90 kHz ticks).
    #[must_use]
    pub const fn with_pts_adjustment(mut self, pts: u64) -> Self {
        self.pts_adjustment = pts & PTS_MASK;
        self
    }

    /// Encodes this marker to a complete SCTE-35 `splice_info_section` binary
    /// blob, including the MPEG-2 CRC-32 trailer.
    ///
    /// The returned bytes can be base64-encoded for embedding in `#EXT-X-SCTE35`,
    /// `#EXT-OATCLS-SCTE35`, or placed into an MPEG-2 TS private section.
    ///
    /// # Layout
    ///
    /// ```text
    /// [0]    table_id = 0xFC
    /// [1..2] section_syntax_indicator(1)=0, private(1)=0, reserved(2)=11,
    ///        section_length(12)
    /// [3]    protocol_version = 0x00
    /// [4..7] encrypted_packet(1)=0, encryption_algorithm(6)=0,
    ///        pts_adjustment(33 bits across 5 bytes) but packed with cw_index(8)
    /// [8]    cw_index = 0xFF
    /// [9..10] tier(12) | splice_command_length(12) high nibble
    /// ...
    /// [-4..-1] CRC_32 (MPEG-2)
    /// ```
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        // 1. Encode the command payload.
        let cmd_payload = self.command.encode_payload();
        let cmd_type = self.command.command_type_byte();

        // 2. Encode descriptor loop.
        let mut desc_bytes: Vec<u8> = Vec::new();
        for d in &self.descriptors {
            desc_bytes.extend_from_slice(&d.encode());
        }

        // 3. Build the inner section (everything after the 3-byte section header
        //    and before the CRC).  We need the length to fill section_length.
        //
        //    section body:
        //      protocol_version(8)
        //      encrypted_packet(1) | encryption_algorithm(6) | pts_adjustment(33)
        //      = 5 bytes for the pts_adjustment word: byte = [ep(1)|ea(6)|pts32(1)]
        //        [pts31..pts24] [pts23..pts16] [pts15..pts8] [pts7..pts0]
        //      cw_index(8)
        //      tier(12) | splice_command_length(12)  = 3 bytes
        //      splice_command_type(8)
        //      splice_command_payload
        //      descriptor_loop_length(16)
        //      descriptor_loop
        //      CRC_32(32)
        //
        // section_length = everything from protocol_version through CRC_32.

        let splice_cmd_len = cmd_payload.len() as u16;
        let desc_loop_len = desc_bytes.len() as u16;

        // section body length (after section_length field, including CRC):
        //   1 (protocol_version)
        // + 5 (encrypted_packet + pts_adjustment)
        // + 1 (cw_index)
        // + 3 (tier + splice_command_length)
        // + 1 (splice_command_type)
        // + splice_cmd_len
        // + 2 (descriptor_loop_length)
        // + desc_loop_len
        // + 4 (CRC_32)
        let section_length: u16 = 1 + 5 + 1 + 3 + 1 + splice_cmd_len + 2 + desc_loop_len + 4;

        let mut buf: Vec<u8> = Vec::with_capacity(3 + section_length as usize);

        // ── Section header ────────────────────────────────────────────────────

        // table_id = 0xFC
        buf.push(0xFC);

        // section_syntax_indicator(1)=0 | private_indicator(1)=0 |
        // reserved(2)=11 | section_length(12)
        // Byte[1] = 0b00_11_xxxx (top 4), section_length high 8 bits
        // section_length_high_nibble occupies bits [11..8]
        let sl_high = ((section_length >> 8) & 0x0F) as u8;
        buf.push(0b0011_0000 | sl_high);
        buf.push((section_length & 0xFF) as u8);

        // ── Section body ──────────────────────────────────────────────────────

        // protocol_version = 0
        buf.push(0x00);

        // encrypted_packet(1)=0 | encryption_algorithm(6)=0 | pts_adjustment(33)
        // Pack into 5 bytes: bit7 = encrypted_packet, bits6..1 = enc_algo,
        // bit0 = pts_adjustment[32], then pts_adjustment[31..0]
        let pts = self.pts_adjustment & PTS_MASK;
        let pa_byte0: u8 = ((pts >> 32) & 0x01) as u8; // encrypted=0, algo=0
        let pa_byte1: u8 = ((pts >> 24) & 0xFF) as u8;
        let pa_byte2: u8 = ((pts >> 16) & 0xFF) as u8;
        let pa_byte3: u8 = ((pts >> 8) & 0xFF) as u8;
        let pa_byte4: u8 = (pts & 0xFF) as u8;
        buf.extend_from_slice(&[pa_byte0, pa_byte1, pa_byte2, pa_byte3, pa_byte4]);

        // cw_index = 0xFF
        buf.push(0xFF);

        // tier(12) | splice_command_length(12)  (3 bytes, 24 bits total)
        // tier occupies [23..12], splice_command_length occupies [11..0]
        let tier = self.tier & 0xFFF;
        let tier_cmd = ((tier as u32) << 12) | (splice_cmd_len as u32 & 0xFFF);
        buf.push(((tier_cmd >> 16) & 0xFF) as u8);
        buf.push(((tier_cmd >> 8) & 0xFF) as u8);
        buf.push((tier_cmd & 0xFF) as u8);

        // splice_command_type
        buf.push(cmd_type);

        // splice_command_payload
        buf.extend_from_slice(&cmd_payload);

        // descriptor_loop_length (16 bits)
        buf.push((desc_loop_len >> 8) as u8);
        buf.push((desc_loop_len & 0xFF) as u8);

        // descriptors
        buf.extend_from_slice(&desc_bytes);

        // CRC_32 (MPEG-2 over all preceding bytes starting from table_id)
        let crc = crc32_mpeg2(&buf);
        buf.extend_from_slice(&crc.to_be_bytes());

        buf
    }

    /// Encode and return as a base64 string (no padding), suitable for
    /// embedding directly in `#EXT-X-SCTE35:CUE="..."` or `#EXT-OATCLS-SCTE35:`.
    #[must_use]
    pub fn encode_base64(&self) -> String {
        base64_encode(&self.encode())
    }
}

// ── Minimal base64 encoder (no external deps) ─────────────────────────────────

const B64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(input: &[u8]) -> String {
    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;

        out.push(B64_CHARS[((triple >> 18) & 63) as usize] as char);
        out.push(B64_CHARS[((triple >> 12) & 63) as usize] as char);

        if chunk.len() > 1 {
            out.push(B64_CHARS[((triple >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64_CHARS[(triple & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

// ── Scte35Descriptor ──────────────────────────────────────────────────────────

/// Segmentation type IDs for SCTE-35 (ANSI/SCTE 35 §10.3.3).
#[allow(dead_code)]
pub mod segmentation_types {
    /// Program start.
    pub const PROGRAM_START: u8 = 0x10;
    /// Program end.
    pub const PROGRAM_END: u8 = 0x11;
    /// Chapter start.
    pub const CHAPTER_START: u8 = 0x20;
    /// Chapter end.
    pub const CHAPTER_END: u8 = 0x21;
    /// Provider ad start.
    pub const PROVIDER_AD_START: u8 = 0x30;
    /// Provider ad end.
    pub const PROVIDER_AD_END: u8 = 0x31;
    /// Distributor ad start.
    pub const DISTRIBUTOR_AD_START: u8 = 0x32;
    /// Distributor ad end.
    pub const DISTRIBUTOR_AD_END: u8 = 0x33;
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splice_insert_command_type() {
        let marker = Scte35Marker::splice_insert(123, Some(Duration::from_secs(120)));
        assert_eq!(marker.command.command_type_byte(), 0x05);
    }

    #[test]
    fn test_splice_insert_event_id() {
        let marker = Scte35Marker::splice_insert(123, Some(Duration::from_secs(120)));
        match &marker.command {
            Scte35Command::SpliceInsert {
                event_id, duration, ..
            } => {
                assert_eq!(*event_id, 123);
                assert_eq!(*duration, Some(Duration::from_secs(120)));
            }
            _ => panic!("Expected SpliceInsert command"),
        }
    }

    #[test]
    fn test_immediate_splice() {
        let marker = Scte35Marker::immediate_splice(456, None);
        match &marker.command {
            Scte35Command::SpliceInsert { immediate, .. } => {
                assert!(*immediate);
            }
            _ => panic!("Expected SpliceInsert command"),
        }
    }

    #[test]
    fn test_segmentation_descriptor() {
        let descriptor = Scte35Descriptor::segmentation(100, segmentation_types::PROVIDER_AD_START);
        assert_eq!(descriptor.tag, 0x02);
        assert!(!descriptor.data.is_empty());
        // Identifier should be 'CUEI'
        assert_eq!(&descriptor.data[..4], b"CUEI");
    }

    #[test]
    fn test_segmentation_descriptor_encode_has_tag_and_length() {
        let desc = Scte35Descriptor::segmentation(1, 0x30);
        let enc = desc.encode();
        assert_eq!(enc[0], 0x02); // tag
        assert_eq!(enc[1] as usize, desc.data.len()); // length
        assert_eq!(enc.len(), 2 + desc.data.len());
    }

    #[test]
    fn test_marker_encoding_starts_with_fc30() {
        let marker = Scte35Marker::splice_insert(1, Some(Duration::from_secs(30)));
        let encoded = marker.encode();
        assert!(!encoded.is_empty());
        // table_id must be 0xFC
        assert_eq!(encoded[0], 0xFC);
        // Top 4 bits of byte[1] must be 0b0011 (section_syntax=0, private=0, reserved=11)
        assert_eq!(encoded[1] & 0xF0, 0x30);
    }

    #[test]
    fn test_marker_encoding_section_length_consistent() {
        let marker = Scte35Marker::splice_insert(42, Some(Duration::from_secs(60)));
        let encoded = marker.encode();
        // section_length = value at bytes[1..3], lower 12 bits
        let section_length = (((encoded[1] & 0x0F) as usize) << 8) | (encoded[2] as usize);
        // Section body starts at byte[3], total section = 3 + section_length (includes CRC)
        assert_eq!(encoded.len(), 3 + section_length);
    }

    #[test]
    fn test_crc32_appended() {
        let marker = Scte35Marker::splice_null();
        let encoded = marker.encode();
        // Last 4 bytes are CRC; recompute CRC over all but last 4 bytes and check.
        let body = &encoded[..encoded.len() - 4];
        let expected_crc = crc32_mpeg2(body);
        let trailing = &encoded[encoded.len() - 4..];
        let got_crc = u32::from_be_bytes([trailing[0], trailing[1], trailing[2], trailing[3]]);
        assert_eq!(got_crc, expected_crc);
    }

    #[test]
    fn test_splice_null_command_type() {
        let marker = Scte35Marker::splice_null();
        assert_eq!(marker.command.command_type_byte(), 0x00);
    }

    #[test]
    fn test_time_signal_command_type() {
        let marker = Scte35Marker::time_signal(0);
        assert_eq!(marker.command.command_type_byte(), 0x06);
    }

    #[test]
    fn test_duration_to_pts_round_trip() {
        let d = Duration::from_secs(30);
        let pts = duration_to_pts(d);
        let back = pts_to_duration(pts);
        // Should round-trip within ±1 ms.
        let diff_ms = (back.as_millis() as i64 - d.as_millis() as i64).unsigned_abs();
        assert!(diff_ms <= 1, "round-trip error: {diff_ms} ms");
    }

    #[test]
    fn test_base64_encode_basic() {
        // "Man" → "TWFu"
        assert_eq!(base64_encode(b"Man"), "TWFu");
        // "Ma" → "TWE="
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        // "M" → "TQ=="
        assert_eq!(base64_encode(b"M"), "TQ==");
    }

    #[test]
    fn test_encode_base64_non_empty() {
        let marker = Scte35Marker::splice_insert(7, Some(Duration::from_secs(15)));
        let b64 = marker.encode_base64();
        assert!(!b64.is_empty());
        // Base64 length must be a multiple of 4.
        assert_eq!(b64.len() % 4, 0);
    }

    #[test]
    fn test_with_tier() {
        let marker = Scte35Marker::splice_insert(1, None).with_tier(0xABC);
        assert_eq!(marker.tier, 0xABC);
    }

    #[test]
    fn test_splice_insert_with_descriptor() {
        let mut marker = Scte35Marker::splice_insert(99, Some(Duration::from_secs(30)));
        marker.add_descriptor(Scte35Descriptor::segmentation(
            99,
            segmentation_types::PROVIDER_AD_START,
        ));
        let encoded = marker.encode();
        // Should still start with 0xFC and have consistent section_length.
        assert_eq!(encoded[0], 0xFC);
        let section_length = (((encoded[1] & 0x0F) as usize) << 8) | (encoded[2] as usize);
        assert_eq!(encoded.len(), 3 + section_length);
    }

    #[test]
    fn test_crc32_mpeg2_known_value() {
        // CRC-32/MPEG-2 of empty slice should be 0xFFFFFFFF.
        assert_eq!(crc32_mpeg2(&[]), 0xFFFF_FFFF);
    }
}
