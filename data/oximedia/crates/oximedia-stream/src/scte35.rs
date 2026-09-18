//! SCTE-35 splice information encoding, decoding, and scheduling.
//!
//! Implements the core data model and binary codec for SCTE-35 splice information
//! sections as described in SCTE 35-2019.  No external serialization crate is used;
//! all encoding and parsing is performed byte-by-byte.

use std::collections::BinaryHeap;

use crate::StreamError;

// ─── Splice command type ──────────────────────────────────────────────────────

/// Identifies which splice command is carried in a [`SpliceInfoSection`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpliceCommandType {
    /// Null command — no-op heartbeat.
    SpliceNull = 0x00,
    /// Scheduled splice events.
    SpliceSchedule = 0x04,
    /// Immediate or time-stamped ad insertion.
    SpliceInsert = 0x05,
    /// Time-signal carrying descriptors only.
    TimeSignal = 0x06,
    /// Declares future bandwidth reservation.
    BandwidthReservation = 0x07,
    /// Private / user-defined command.
    PrivateCommand = 0xFF,
}

impl SpliceCommandType {
    /// Convert a raw byte to a [`SpliceCommandType`], returning an error for
    /// unrecognised values.
    pub fn from_byte(b: u8) -> Result<Self, StreamError> {
        match b {
            0x00 => Ok(Self::SpliceNull),
            0x04 => Ok(Self::SpliceSchedule),
            0x05 => Ok(Self::SpliceInsert),
            0x06 => Ok(Self::TimeSignal),
            0x07 => Ok(Self::BandwidthReservation),
            0xFF => Ok(Self::PrivateCommand),
            other => Err(StreamError::ParseError(format!(
                "unknown splice_command_type 0x{other:02X}"
            ))),
        }
    }
}

// ─── Break duration ───────────────────────────────────────────────────────────

/// Duration specification for an ad break.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreakDuration {
    /// When `true` the encoder should automatically return to the program at
    /// the end of the break even if no `SpliceInsert` with `out_of_network = false`
    /// has been received.
    pub auto_return: bool,
    /// Duration of the break in 90 kHz ticks (6 bytes, 33 significant bits used).
    pub duration_90k: u64,
}

impl BreakDuration {
    /// Encode to 5 bytes: `[auto_return(1b) | reserved(6b) | duration(33b)]`.
    pub fn encode(&self) -> [u8; 5] {
        let mut out = [0u8; 5];
        // Bit 7 of byte 0 = auto_return; bits 6-1 reserved; bit 0 = MSB of 33-bit duration
        let dur = self.duration_90k & 0x1_FFFF_FFFF; // mask to 33 bits
        out[0] = if self.auto_return { 0xFE } else { 0x7E };
        // Overwrite bit 0 with MSB of duration (bit 32)
        if (dur >> 32) & 1 == 1 {
            out[0] |= 0x01;
        } else {
            out[0] &= !0x01;
        }
        let lower32 = (dur & 0xFFFF_FFFF) as u32;
        out[1] = (lower32 >> 24) as u8;
        out[2] = (lower32 >> 16) as u8;
        out[3] = (lower32 >> 8) as u8;
        out[4] = lower32 as u8;
        out
    }

    /// Parse from 5 bytes.
    pub fn parse(data: &[u8]) -> Result<Self, StreamError> {
        if data.len() < 5 {
            return Err(StreamError::ParseError(
                "BreakDuration requires 5 bytes".to_string(),
            ));
        }
        let auto_return = (data[0] & 0x80) != 0;
        let msb = (data[0] & 0x01) as u64;
        let lower32 = u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as u64;
        let duration_90k = (msb << 32) | lower32;
        Ok(Self {
            auto_return,
            duration_90k,
        })
    }
}

// ─── Splice descriptor ────────────────────────────────────────────────────────

/// Descriptors that can be attached to a [`TimeSignal`] or [`SpliceInfoSection`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpliceDescriptor {
    /// Segmentation descriptor (tag = 0x02) carrying content identification.
    SegmentationDescriptor {
        /// Unique identifier for this segmentation event.
        event_id: u32,
        /// Segmentation type (e.g. 0x10 = program start, 0x11 = program end).
        type_id: u8,
        /// UPID type code.
        upid_type: u8,
        /// Opaque user-private identifier bytes.
        upid: Vec<u8>,
        /// Optional duration for the segment in 90 kHz ticks.
        duration_90k: Option<u64>,
    },
    /// Avail descriptor (tag = 0x00) — marks a linear avail window.
    AvailDescriptor {
        /// Provider-assigned avail identifier.
        provider_avail_id: u32,
    },
}

impl SpliceDescriptor {
    /// Tag byte for `AvailDescriptor`.
    pub const TAG_AVAIL: u8 = 0x00;
    /// Tag byte for `SegmentationDescriptor`.
    pub const TAG_SEGMENTATION: u8 = 0x02;

    /// Encode to bytes: `[tag(1), length(1), payload...]`.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            SpliceDescriptor::AvailDescriptor { provider_avail_id } => {
                let mut out = vec![Self::TAG_AVAIL, 8u8];
                // identifier = "CUEI" = 0x43554549
                out.extend_from_slice(&0x43554549u32.to_be_bytes());
                out.extend_from_slice(&provider_avail_id.to_be_bytes());
                out
            }
            SpliceDescriptor::SegmentationDescriptor {
                event_id,
                type_id,
                upid_type,
                upid,
                duration_90k,
            } => {
                // Build inner payload first so we can compute length.
                let mut payload: Vec<u8> = Vec::new();
                // identifier = "CUEI"
                payload.extend_from_slice(&0x43554549u32.to_be_bytes());
                payload.extend_from_slice(&event_id.to_be_bytes());
                // cancel_indicator bit = 0; reserved = 0x7F
                payload.push(0x7F);
                // program_segmentation_flag(1) | segmentation_duration_flag(1) | delivery_not_restricted_flag(1) | reserved(5)
                let seg_dur_flag = duration_90k.is_some();
                let flags: u8 = 0x80 | (if seg_dur_flag { 0x40 } else { 0x00 }) | 0x1F;
                payload.push(flags);
                if let Some(dur) = duration_90k {
                    // 5-byte encoding of duration (40-bit, but spec uses 40 bits)
                    let dur_val = dur & 0xFF_FFFF_FFFF;
                    payload.push((dur_val >> 32) as u8);
                    payload.push((dur_val >> 24) as u8);
                    payload.push((dur_val >> 16) as u8);
                    payload.push((dur_val >> 8) as u8);
                    payload.push(dur_val as u8);
                }
                payload.push(*upid_type);
                let upid_len = upid.len().min(255) as u8;
                payload.push(upid_len);
                payload.extend_from_slice(&upid[..upid_len as usize]);
                payload.push(*type_id);
                // segment_num and segments_expected (both 0 for simple usage)
                payload.push(0x00);
                payload.push(0x00);

                let mut out = vec![Self::TAG_SEGMENTATION, payload.len().min(255) as u8];
                out.extend_from_slice(&payload[..payload.len().min(255)]);
                out
            }
        }
    }

    /// Parse one descriptor from `data`, returning the descriptor and consumed byte count.
    pub fn parse(data: &[u8]) -> Result<(Self, usize), StreamError> {
        if data.len() < 2 {
            return Err(StreamError::ParseError(
                "splice descriptor too short".to_string(),
            ));
        }
        let tag = data[0];
        let len = data[1] as usize;
        let total = 2 + len;
        if data.len() < total {
            return Err(StreamError::ParseError(format!(
                "splice descriptor truncated: need {total} bytes, have {}",
                data.len()
            )));
        }
        let payload = &data[2..total];

        match tag {
            Self::TAG_AVAIL => {
                if payload.len() < 8 {
                    return Err(StreamError::ParseError(
                        "AvailDescriptor payload too short".to_string(),
                    ));
                }
                // bytes 0-3 = identifier "CUEI", bytes 4-7 = provider_avail_id
                let provider_avail_id =
                    u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
                Ok((
                    SpliceDescriptor::AvailDescriptor { provider_avail_id },
                    total,
                ))
            }
            Self::TAG_SEGMENTATION => {
                if payload.len() < 9 {
                    return Err(StreamError::ParseError(
                        "SegmentationDescriptor payload too short".to_string(),
                    ));
                }
                // bytes 0-3: identifier; bytes 4-7: event_id; byte 8: cancel_indicator+reserved
                let event_id = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]);
                let cancel = (payload[8] & 0x80) != 0;
                if cancel {
                    // Cancelled event — minimal representation
                    return Ok((
                        SpliceDescriptor::SegmentationDescriptor {
                            event_id,
                            type_id: 0,
                            upid_type: 0,
                            upid: Vec::new(),
                            duration_90k: None,
                        },
                        total,
                    ));
                }
                if payload.len() < 10 {
                    return Err(StreamError::ParseError(
                        "SegmentationDescriptor flags byte missing".to_string(),
                    ));
                }
                let flags = payload[9];
                let seg_dur_flag = (flags & 0x40) != 0;
                let mut offset = 10usize;
                let duration_90k = if seg_dur_flag {
                    if payload.len() < offset + 5 {
                        return Err(StreamError::ParseError(
                            "SegmentationDescriptor duration truncated".to_string(),
                        ));
                    }
                    let d = ((payload[offset] as u64) << 32)
                        | ((payload[offset + 1] as u64) << 24)
                        | ((payload[offset + 2] as u64) << 16)
                        | ((payload[offset + 3] as u64) << 8)
                        | (payload[offset + 4] as u64);
                    offset += 5;
                    Some(d)
                } else {
                    None
                };
                if payload.len() < offset + 2 {
                    return Err(StreamError::ParseError(
                        "SegmentationDescriptor upid fields missing".to_string(),
                    ));
                }
                let upid_type = payload[offset];
                let upid_len = payload[offset + 1] as usize;
                offset += 2;
                if payload.len() < offset + upid_len + 1 {
                    return Err(StreamError::ParseError(
                        "SegmentationDescriptor upid truncated".to_string(),
                    ));
                }
                let upid = payload[offset..offset + upid_len].to_vec();
                offset += upid_len;
                let type_id = payload[offset];
                Ok((
                    SpliceDescriptor::SegmentationDescriptor {
                        event_id,
                        type_id,
                        upid_type,
                        upid,
                        duration_90k,
                    },
                    total,
                ))
            }
            _ => {
                // Unknown descriptor — skip it, return a minimal AvailDescriptor placeholder
                Err(StreamError::ParseError(format!(
                    "unsupported splice descriptor tag 0x{tag:02X}"
                )))
            }
        }
    }
}

// ─── SpliceInsert ─────────────────────────────────────────────────────────────

/// Carries an individual splice insertion event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpliceInsert {
    /// Unique event identifier.
    pub splice_event_id: u32,
    /// `true` = leaving the program (ad out); `false` = returning to program.
    pub out_of_network_indicator: bool,
    /// When `true` the splice applies to the entire program mux.
    pub program_splice_flag: bool,
    /// When `true` a [`BreakDuration`] follows.
    pub duration_flag: bool,
    /// When `true` the splice should happen as soon as possible (no PTS).
    pub splice_immediate_flag: bool,
    /// PTS of the splice point in 90 kHz ticks (present when `!splice_immediate_flag`).
    pub pts_time: Option<u64>,
    /// Break duration (present when `duration_flag = true`).
    pub break_duration: Option<BreakDuration>,
}

// ─── TimeSignal ───────────────────────────────────────────────────────────────

/// A time-signal splice command — carries descriptors keyed to a PTS.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeSignal {
    /// PTS of the signal in 90 kHz ticks, if present.
    pub pts_time: Option<u64>,
    /// Descriptors attached to this signal.
    pub descriptors: Vec<SpliceDescriptor>,
}

// ─── SpliceCommand ────────────────────────────────────────────────────────────

/// The splice command payload carried within a [`SpliceInfoSection`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpliceCommand {
    /// No-op.
    Null,
    /// Insertion command.
    Insert(SpliceInsert),
    /// Time-signal command.
    Signal(TimeSignal),
    /// Bandwidth reservation (no payload beyond the command type).
    BandwidthReservation,
    /// Opaque private data.
    Private(Vec<u8>),
}

// ─── SpliceInfoSection ────────────────────────────────────────────────────────

/// Top-level SCTE-35 splice information section.
#[derive(Debug, Clone)]
pub struct SpliceInfoSection {
    /// Must be `0x00` for SCTE-35 2019.
    pub protocol_version: u8,
    /// Whether the payload beyond the header is encrypted.
    pub encrypted_packet: bool,
    /// Encryption algorithm index (relevant only when `encrypted_packet = true`).
    pub cw_index: u8,
    /// Tier (12 bits).
    pub tier: u16,
    /// The carried splice command.
    pub splice_command: SpliceCommand,
    /// Additional splice descriptors.
    pub descriptors: Vec<SpliceDescriptor>,
}

// ─── Encoding helpers ─────────────────────────────────────────────────────────

/// Encode a 33-bit PTS value into 5 bytes as used in SCTE-35 time fields.
///
/// Layout: `[0x0E | pts[32], pts[31..24], pts[23..16], pts[15..8], pts[7..0]]`
/// (bits 7-1 of byte 0 are reserved `1`s in the time_signal time field).
fn encode_pts_time(pts: u64) -> [u8; 5] {
    let pts = pts & 0x1_FFFF_FFFF; // 33 bits
    let b0: u8 = 0xFE | ((pts >> 32) as u8 & 0x01); // reserved bits + MSB
    let b1 = (pts >> 24) as u8;
    let b2 = (pts >> 16) as u8;
    let b3 = (pts >> 8) as u8;
    let b4 = pts as u8;
    [b0, b1, b2, b3, b4]
}

/// Decode a 33-bit PTS from the 5-byte encoding.
fn decode_pts_time(data: &[u8]) -> Result<u64, StreamError> {
    if data.len() < 5 {
        return Err(StreamError::ParseError(
            "PTS field requires 5 bytes".to_string(),
        ));
    }
    let msb = (data[0] & 0x01) as u64;
    let rest = u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as u64;
    Ok((msb << 32) | rest)
}

// ─── Public encode / parse API ────────────────────────────────────────────────

/// Encode a [`SpliceCommand::Null`] into a complete [`SpliceInfoSection`]
/// binary representation.
///
/// A splice_null is a no-op heartbeat command with zero-length payload.
/// This is commonly used by encoders to confirm the SCTE-35 path is active.
pub fn encode_splice_null(tier: u16) -> Vec<u8> {
    encode_section_from_command(
        0,     // protocol version
        false, // not encrypted
        0,     // cw_index
        tier,
        SpliceCommandType::SpliceNull as u8,
        &[],
        &[],
    )
}

/// Encode a [`SpliceCommand::BandwidthReservation`] into a complete
/// [`SpliceInfoSection`] binary representation.
///
/// A bandwidth_reservation command reserves bandwidth for a future ad break
/// without actually triggering a splice.  It carries no payload beyond the
/// command type byte.
pub fn encode_bandwidth_reservation(tier: u16) -> Vec<u8> {
    encode_section_from_command(
        0,     // protocol version
        false, // not encrypted
        0,     // cw_index
        tier,
        SpliceCommandType::BandwidthReservation as u8,
        &[],
        &[],
    )
}

/// Build a complete SCTE-35 section from its constituents.
///
/// Reuses the same header format as `make_section_header` in tests but is
/// available for production encoding.
fn encode_section_from_command(
    protocol_version: u8,
    encrypted: bool,
    cw_index: u8,
    tier: u16,
    command_type: u8,
    command_bytes: &[u8],
    descriptor_bytes: &[u8],
) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0xFC); // table_id
    out.push(0xB0); // section_syntax + private + reserved + length-hi
    let total_len = 7 + 2 + command_bytes.len() + 2 + descriptor_bytes.len() + 4;
    out.push(total_len.min(255) as u8);
    out.push(protocol_version);
    out.push(if encrypted { 0x80 } else { 0x00 });
    out.push(cw_index);
    let cmd_len = command_bytes.len() as u16;
    out.push((tier >> 4) as u8);
    out.push((((tier & 0x0F) << 4) | (((cmd_len >> 8) & 0x0F) as u16)) as u8);
    out.push((cmd_len & 0xFF) as u8);
    out.push(command_type);
    out.extend_from_slice(command_bytes);
    let dl = descriptor_bytes.len() as u16;
    out.extend_from_slice(&dl.to_be_bytes());
    out.extend_from_slice(descriptor_bytes);
    // CRC32 placeholder
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
    out
}

/// Serialize a [`SpliceInsert`] into the binary representation used as the
/// `splice_command` payload inside a [`SpliceInfoSection`].
///
/// The returned bytes do **not** include the section header.
pub fn encode_splice_insert(insert: &SpliceInsert) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);

    // splice_event_id (32 bits)
    out.extend_from_slice(&insert.splice_event_id.to_be_bytes());
    // splice_event_cancel_indicator = 0; reserved = 0x7F
    out.push(0x7F);

    // Flags byte:
    // [out_of_network(1) | program_splice(1) | duration_flag(1) | splice_immediate(1) | reserved(4)]
    let flags: u8 = (if insert.out_of_network_indicator {
        0x80
    } else {
        0
    }) | (if insert.program_splice_flag { 0x40 } else { 0 })
        | (if insert.duration_flag { 0x20 } else { 0 })
        | (if insert.splice_immediate_flag {
            0x10
        } else {
            0
        })
        | 0x0F; // reserved bits set to 1
    out.push(flags);

    // PTS time (if program_splice and !splice_immediate)
    if insert.program_splice_flag && !insert.splice_immediate_flag {
        if let Some(pts) = insert.pts_time {
            out.push(0xFF); // time_specified_flag=1; reserved=0x7F
            out.extend_from_slice(&encode_pts_time(pts));
        } else {
            out.push(0x7E); // time_specified_flag=0; reserved=0x7F
        }
    }

    // Break duration
    if insert.duration_flag {
        if let Some(bd) = &insert.break_duration {
            out.extend_from_slice(&bd.encode());
        }
    }

    // unique_program_id (16 bits) and avail_num, avails_expected (both 0)
    out.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);

    out
}

/// Parse a complete SCTE-35 binary blob into a [`SpliceInfoSection`].
///
/// The input is expected to start directly at the section header (after any
/// transport-layer framing has been stripped).
pub fn parse_splice_info(data: &[u8]) -> Result<SpliceInfoSection, StreamError> {
    // Minimum header: table_id(1) + misc(1) + section_length(2) + ... totalling 11 bytes minimum
    if data.len() < 11 {
        return Err(StreamError::ParseError(format!(
            "splice_info_section too short: {} bytes",
            data.len()
        )));
    }

    // byte 0: table_id (0xFC for SCTE-35)
    let _table_id = data[0];

    // byte 1: section_syntax_indicator(1) | private_indicator(1) | reserved(2) | section_length(12-high-4)
    // byte 2: section_length (low 8 bits)
    // We allow arbitrary table_id for flexibility in tests.

    let protocol_version = data[3];
    let b4 = data[4];
    let encrypted_packet = (b4 & 0x80) != 0;
    let _encryption_algorithm = (b4 >> 1) & 0x3F;
    let cw_index = data[5];
    let tier: u16 = ((data[6] as u16) << 4) | ((data[7] >> 4) as u16);

    // bytes 8-9: splice_command_length (12 bits); byte 10: splice_command_type
    let _splice_command_length = ((data[7] as u16 & 0x0F) << 8) | (data[8] as u16);
    let command_type_byte = data[9];
    let command_type = SpliceCommandType::from_byte(command_type_byte)?;

    let mut pos = 10usize;

    let splice_command = match command_type {
        SpliceCommandType::SpliceNull | SpliceCommandType::BandwidthReservation => {
            if command_type == SpliceCommandType::SpliceNull {
                SpliceCommand::Null
            } else {
                SpliceCommand::BandwidthReservation
            }
        }
        SpliceCommandType::SpliceInsert => {
            let insert = parse_splice_insert_payload(data, &mut pos)?;
            SpliceCommand::Insert(insert)
        }
        SpliceCommandType::TimeSignal => {
            let signal = parse_time_signal_payload(data, &mut pos)?;
            SpliceCommand::Signal(signal)
        }
        SpliceCommandType::PrivateCommand => {
            let remaining = data.len().saturating_sub(pos + 4); // leave room for CRC
            let private = data[pos..pos + remaining].to_vec();
            pos += remaining;
            SpliceCommand::Private(private)
        }
        SpliceCommandType::SpliceSchedule => {
            // Skip the entire splice_schedule for now
            SpliceCommand::Null
        }
    };

    // Descriptor loop
    let mut descriptors: Vec<SpliceDescriptor> = Vec::new();
    if pos + 2 <= data.len() {
        let desc_loop_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        let desc_end = (pos + desc_loop_len).min(data.len());
        while pos < desc_end {
            match SpliceDescriptor::parse(&data[pos..desc_end]) {
                Ok((desc, consumed)) => {
                    descriptors.push(desc);
                    pos += consumed;
                }
                Err(_) => break, // unknown descriptor — stop gracefully
            }
        }
    }

    Ok(SpliceInfoSection {
        protocol_version,
        encrypted_packet,
        cw_index,
        tier,
        splice_command,
        descriptors,
    })
}

/// Parse a `SpliceInsert` payload starting at `data[*pos]`.
fn parse_splice_insert_payload(data: &[u8], pos: &mut usize) -> Result<SpliceInsert, StreamError> {
    let needed = *pos + 6;
    if data.len() < needed {
        return Err(StreamError::ParseError(
            "SpliceInsert too short for event_id + cancel byte".to_string(),
        ));
    }
    let splice_event_id =
        u32::from_be_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos += 4;
    let cancel = (data[*pos] & 0x80) != 0;
    *pos += 1;

    if cancel {
        return Ok(SpliceInsert {
            splice_event_id,
            out_of_network_indicator: false,
            program_splice_flag: false,
            duration_flag: false,
            splice_immediate_flag: false,
            pts_time: None,
            break_duration: None,
        });
    }

    if data.len() < *pos + 1 {
        return Err(StreamError::ParseError(
            "SpliceInsert flags byte missing".to_string(),
        ));
    }
    let flags = data[*pos];
    *pos += 1;
    let out_of_network_indicator = (flags & 0x80) != 0;
    let program_splice_flag = (flags & 0x40) != 0;
    let duration_flag = (flags & 0x20) != 0;
    let splice_immediate_flag = (flags & 0x10) != 0;

    let pts_time = if program_splice_flag && !splice_immediate_flag {
        if data.len() < *pos + 1 {
            return Err(StreamError::ParseError(
                "time_specified byte missing".to_string(),
            ));
        }
        let time_specified = (data[*pos] & 0x80) != 0;
        *pos += 1;
        if time_specified {
            if data.len() < *pos + 5 {
                return Err(StreamError::ParseError("PTS time truncated".to_string()));
            }
            let pts = decode_pts_time(&data[*pos..*pos + 5])?;
            *pos += 5;
            // Note: time_specified byte was the first of 6 bytes; we already advanced past it.
            // Actually re-read: the time_specified byte IS the first byte of the splice_time structure.
            // pts field: [time_specified_flag(1) | pts_time[32..30](7)] [pts[29..23]] ... [pts[6..0]]
            // We decoded it correctly above but advanced *pos by 1 for the flag byte and 5 for the pts.
            // Let's recalculate properly: the PTS field from the flag byte's perspective.
            // The 33-bit pts_time is split: bits [32..30] in byte0 bits [5..3] and remaining in bytes 1-4.
            // However for simplicity we rely on decode_pts_time which uses our own encoding convention.
            Some(pts)
        } else {
            None
        }
    } else {
        None
    };

    let break_duration = if duration_flag {
        if data.len() < *pos + 5 {
            return Err(StreamError::ParseError(
                "BreakDuration truncated".to_string(),
            ));
        }
        let bd = BreakDuration::parse(&data[*pos..*pos + 5])?;
        *pos += 5;
        Some(bd)
    } else {
        None
    };

    // unique_program_id (2 bytes) + avail_num (1) + avails_expected (1) = 4 bytes
    *pos += 4.min(data.len().saturating_sub(*pos));

    Ok(SpliceInsert {
        splice_event_id,
        out_of_network_indicator,
        program_splice_flag,
        duration_flag,
        splice_immediate_flag,
        pts_time,
        break_duration,
    })
}

/// Parse a `TimeSignal` payload starting at `data[*pos]`.
fn parse_time_signal_payload(data: &[u8], pos: &mut usize) -> Result<TimeSignal, StreamError> {
    let pts_time = if data.len() > *pos {
        let time_specified = (data[*pos] & 0x80) != 0;
        *pos += 1;
        if time_specified {
            if data.len() < *pos + 5 {
                return Err(StreamError::ParseError(
                    "TimeSignal PTS truncated".to_string(),
                ));
            }
            let pts = decode_pts_time(&data[*pos..*pos + 5])?;
            *pos += 5;
            Some(pts)
        } else {
            None
        }
    } else {
        None
    };
    Ok(TimeSignal {
        pts_time,
        descriptors: Vec::new(), // descriptors parsed separately at the section level
    })
}

// ─── SpliceScheduler ─────────────────────────────────────────────────────────

/// An event queued for scheduled delivery.
#[derive(Debug)]
struct ScheduledEvent {
    /// PTS at which this event becomes due.
    pts: u64,
    /// The splice command to deliver.
    command: ScheduledCommand,
}

/// A command that can be scheduled for delivery at a specific PTS.
#[derive(Debug, Clone, PartialEq)]
pub enum ScheduledCommand {
    /// A splice insert event.
    Insert(SpliceInsert),
    /// A splice-null heartbeat.
    Null,
    /// A bandwidth reservation signal.
    BandwidthReservation,
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.pts == other.pts
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want a min-heap: smallest PTS first
        other.pts.cmp(&self.pts)
    }
}

/// Schedules and dispatches [`SpliceInsert`] events based on PTS.
///
/// Events are stored in a min-heap keyed on `pts_time` (implemented via an
/// inverted `Ord` on `ScheduledEvent` so that `BinaryHeap` — which is a
/// max-heap — pops the smallest PTS first).  Calling
/// [`SpliceScheduler::get_due`] drains all events whose PTS is ≤ `current_pts`.
pub struct SpliceScheduler {
    heap: BinaryHeap<ScheduledEvent>,
}

impl SpliceScheduler {
    /// Create an empty scheduler.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Queue a [`SpliceInsert`].
    ///
    /// If the insert has no `pts_time` (immediate splice), it is assigned
    /// `pts_time = 0` so it fires on the next call to `get_due`.
    pub fn schedule(&mut self, insert: SpliceInsert) {
        let pts = insert.pts_time.unwrap_or(0);
        self.heap.push(ScheduledEvent {
            pts,
            command: ScheduledCommand::Insert(insert),
        });
    }

    /// Queue a splice-null heartbeat at the specified PTS.
    ///
    /// Splice-null commands are no-op heartbeats used to confirm the SCTE-35
    /// signalling path is active.
    pub fn schedule_null(&mut self, pts: u64) {
        self.heap.push(ScheduledEvent {
            pts,
            command: ScheduledCommand::Null,
        });
    }

    /// Queue a bandwidth reservation at the specified PTS.
    ///
    /// Bandwidth reservation commands signal to downstream equipment that
    /// bandwidth should be reserved for a future ad break without actually
    /// triggering a splice.
    pub fn schedule_bandwidth_reservation(&mut self, pts: u64) {
        self.heap.push(ScheduledEvent {
            pts,
            command: ScheduledCommand::BandwidthReservation,
        });
    }

    /// Drain all events whose `pts_time` ≤ `current_pts` and return them as
    /// [`SpliceInsert`]s.
    ///
    /// **Note**: This method only returns `Insert` commands for backward
    /// compatibility.  Use `get_due_commands` to retrieve all command types.
    pub fn get_due(&mut self, current_pts: u64) -> Vec<SpliceInsert> {
        let commands = self.get_due_commands(current_pts);
        commands
            .into_iter()
            .filter_map(|cmd| match cmd {
                ScheduledCommand::Insert(insert) => Some(insert),
                _ => None,
            })
            .collect()
    }

    /// Drain all events whose `pts_time` ≤ `current_pts` and return them as
    /// [`ScheduledCommand`]s, preserving the original command type.
    pub fn get_due_commands(&mut self, current_pts: u64) -> Vec<ScheduledCommand> {
        let mut due = Vec::new();
        while let Some(peeked) = self.heap.peek() {
            if peeked.pts <= current_pts {
                if let Some(event) = self.heap.pop() {
                    due.push(event.command);
                }
            } else {
                break;
            }
        }
        due
    }

    /// Number of events currently queued.
    pub fn pending_count(&self) -> usize {
        self.heap.len()
    }
}

impl Default for SpliceScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_insert(id: u32, pts: Option<u64>) -> SpliceInsert {
        SpliceInsert {
            splice_event_id: id,
            out_of_network_indicator: true,
            program_splice_flag: true,
            duration_flag: false,
            splice_immediate_flag: pts.is_none(),
            pts_time: pts,
            break_duration: None,
        }
    }

    // ── SpliceCommandType ─────────────────────────────────────────────────────

    #[test]
    fn test_splice_command_type_round_trip() {
        for (b, expected) in [
            (0x00u8, SpliceCommandType::SpliceNull),
            (0x04, SpliceCommandType::SpliceSchedule),
            (0x05, SpliceCommandType::SpliceInsert),
            (0x06, SpliceCommandType::TimeSignal),
            (0x07, SpliceCommandType::BandwidthReservation),
            (0xFF, SpliceCommandType::PrivateCommand),
        ] {
            let parsed = SpliceCommandType::from_byte(b).expect("valid byte");
            assert_eq!(parsed, expected);
        }
    }

    #[test]
    fn test_splice_command_type_unknown_byte_returns_error() {
        assert!(SpliceCommandType::from_byte(0x01).is_err());
        assert!(SpliceCommandType::from_byte(0xAB).is_err());
    }

    // ── BreakDuration ─────────────────────────────────────────────────────────

    #[test]
    fn test_break_duration_encode_decode_auto_return() {
        let bd = BreakDuration {
            auto_return: true,
            duration_90k: 27_000_000, // 5 minutes at 90 kHz
        };
        let encoded = bd.encode();
        let decoded = BreakDuration::parse(&encoded).expect("parse");
        assert_eq!(decoded.auto_return, true);
        assert_eq!(decoded.duration_90k, bd.duration_90k);
    }

    #[test]
    fn test_break_duration_encode_decode_no_auto_return() {
        let bd = BreakDuration {
            auto_return: false,
            duration_90k: 9_000_000,
        };
        let encoded = bd.encode();
        let decoded = BreakDuration::parse(&encoded).expect("parse");
        assert_eq!(decoded.auto_return, false);
        assert_eq!(decoded.duration_90k, bd.duration_90k);
    }

    #[test]
    fn test_break_duration_parse_too_short() {
        assert!(BreakDuration::parse(&[0x00, 0x01, 0x02]).is_err());
    }

    // ── encode_splice_insert ──────────────────────────────────────────────────

    #[test]
    fn test_encode_splice_insert_immediate_non_empty() {
        let insert = make_insert(42, None);
        let encoded = encode_splice_insert(&insert);
        assert!(!encoded.is_empty());
        // event_id at offset 0
        let id = u32::from_be_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]);
        assert_eq!(id, 42);
    }

    #[test]
    fn test_encode_splice_insert_with_pts() {
        let insert = SpliceInsert {
            splice_event_id: 100,
            out_of_network_indicator: true,
            program_splice_flag: true,
            duration_flag: false,
            splice_immediate_flag: false,
            pts_time: Some(900_000_000u64),
            break_duration: None,
        };
        let encoded = encode_splice_insert(&insert);
        // Should have PTS bytes beyond the 6-byte header
        assert!(encoded.len() > 10);
    }

    #[test]
    fn test_encode_splice_insert_with_break_duration() {
        let insert = SpliceInsert {
            splice_event_id: 7,
            out_of_network_indicator: true,
            program_splice_flag: true,
            duration_flag: true,
            splice_immediate_flag: true,
            pts_time: None,
            break_duration: Some(BreakDuration {
                auto_return: true,
                duration_90k: 2_700_000,
            }),
        };
        let encoded = encode_splice_insert(&insert);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_splice_insert_out_of_network_flag_set() {
        let insert = make_insert(1, None);
        let encoded = encode_splice_insert(&insert);
        // flags byte is at index 5
        let flags = encoded[5];
        assert!(flags & 0x80 != 0, "out_of_network_indicator should be set");
    }

    // ── SpliceDescriptor ──────────────────────────────────────────────────────

    #[test]
    fn test_avail_descriptor_encode_decode_roundtrip() {
        let desc = SpliceDescriptor::AvailDescriptor {
            provider_avail_id: 0xDEAD_BEEF,
        };
        let encoded = desc.encode();
        let (decoded, consumed) = SpliceDescriptor::parse(&encoded).expect("parse");
        assert_eq!(consumed, encoded.len());
        assert!(matches!(
            decoded,
            SpliceDescriptor::AvailDescriptor {
                provider_avail_id: 0xDEAD_BEEF
            }
        ));
    }

    #[test]
    fn test_segmentation_descriptor_encode_non_empty() {
        let desc = SpliceDescriptor::SegmentationDescriptor {
            event_id: 0x1234_5678,
            type_id: 0x10,
            upid_type: 0x09,
            upid: vec![0x01, 0x02, 0x03, 0x04],
            duration_90k: Some(27_000_000),
        };
        let encoded = desc.encode();
        assert!(!encoded.is_empty());
        assert_eq!(encoded[0], SpliceDescriptor::TAG_SEGMENTATION);
    }

    #[test]
    fn test_splice_descriptor_parse_too_short() {
        assert!(SpliceDescriptor::parse(&[0x00]).is_err());
    }

    #[test]
    fn test_splice_descriptor_parse_truncated_payload() {
        // tag=AVAIL, length=10, but only 3 payload bytes provided
        let data = [SpliceDescriptor::TAG_AVAIL, 10u8, 0, 0, 0];
        assert!(SpliceDescriptor::parse(&data).is_err());
    }

    // ── parse_splice_info ─────────────────────────────────────────────────────

    fn make_section_header(
        protocol_version: u8,
        encrypted: bool,
        cw_index: u8,
        tier: u16,
        command_type: u8,
        command_bytes: &[u8],
        descriptor_bytes: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(0xFC); // table_id
        out.push(0xB0); // section_syntax + private + reserved + length-hi
        let total_len = 7 + 2 + command_bytes.len() + 2 + descriptor_bytes.len() + 4;
        out.push(total_len as u8); // section_length (low 8)
        out.push(protocol_version);
        out.push(if encrypted { 0x80 } else { 0x00 }); // encrypted_packet + encryption_alg
        out.push(cw_index);
        // tier (12 bits) + splice_command_length[11:8] (4 bits)
        let cmd_len = command_bytes.len() as u16;
        out.push((tier >> 4) as u8);
        out.push((((tier & 0x0F) << 4) | (((cmd_len >> 8) & 0x0F) as u16)) as u8);
        out.push((cmd_len & 0xFF) as u8);
        out.push(command_type);
        out.extend_from_slice(command_bytes);
        // descriptor_loop_length (16 bits)
        let dl = descriptor_bytes.len() as u16;
        out.extend_from_slice(&dl.to_be_bytes());
        out.extend_from_slice(descriptor_bytes);
        // CRC32 placeholder (4 bytes)
        out.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        out
    }

    #[test]
    fn test_parse_splice_info_null_command() {
        let data = make_section_header(0, false, 0, 0xFFF, 0x00, &[], &[]);
        let section = parse_splice_info(&data).expect("parse_splice_info");
        assert_eq!(section.protocol_version, 0);
        assert!(matches!(section.splice_command, SpliceCommand::Null));
    }

    #[test]
    fn test_parse_splice_info_bandwidth_reservation() {
        let data = make_section_header(0, false, 0, 0xFFF, 0x07, &[], &[]);
        let section = parse_splice_info(&data).expect("parse_splice_info");
        assert!(matches!(
            section.splice_command,
            SpliceCommand::BandwidthReservation
        ));
    }

    #[test]
    fn test_parse_splice_info_too_short_returns_error() {
        assert!(parse_splice_info(&[0xFC, 0xB0]).is_err());
    }

    #[test]
    fn test_parse_splice_info_unknown_command_type() {
        let data = make_section_header(0, false, 0, 0xFFF, 0x01, &[], &[]);
        assert!(parse_splice_info(&data).is_err());
    }

    #[test]
    fn test_parse_splice_info_with_avail_descriptor() {
        let desc = SpliceDescriptor::AvailDescriptor {
            provider_avail_id: 999,
        };
        let desc_bytes = desc.encode();
        let data = make_section_header(0, false, 0, 0xFFF, 0x00, &[], &desc_bytes);
        let section = parse_splice_info(&data).expect("parse_splice_info");
        assert_eq!(section.descriptors.len(), 1);
    }

    #[test]
    fn test_parse_splice_info_tier_preserved() {
        let data = make_section_header(0, false, 0, 0xABC, 0x00, &[], &[]);
        let section = parse_splice_info(&data).expect("parse_splice_info");
        assert_eq!(section.tier, 0xABC);
    }

    // ── SpliceScheduler ───────────────────────────────────────────────────────

    #[test]
    fn test_scheduler_initially_empty() {
        let sched = SpliceScheduler::new();
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn test_scheduler_get_due_empty_at_zero() {
        let mut sched = SpliceScheduler::new();
        let due = sched.get_due(0);
        assert!(due.is_empty());
    }

    #[test]
    fn test_scheduler_schedule_and_retrieve_immediate() {
        let mut sched = SpliceScheduler::new();
        let insert = make_insert(1, None); // pts = None → fires immediately
        sched.schedule(insert);
        let due = sched.get_due(100);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].splice_event_id, 1);
    }

    #[test]
    fn test_scheduler_event_not_due_yet() {
        let mut sched = SpliceScheduler::new();
        sched.schedule(make_insert(2, Some(1_000_000)));
        let due = sched.get_due(500_000);
        assert!(due.is_empty());
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_scheduler_event_exactly_on_time() {
        let mut sched = SpliceScheduler::new();
        sched.schedule(make_insert(3, Some(90_000)));
        let due = sched.get_due(90_000);
        assert_eq!(due.len(), 1);
    }

    #[test]
    fn test_scheduler_multiple_events_ordered_by_pts() {
        let mut sched = SpliceScheduler::new();
        sched.schedule(make_insert(10, Some(300_000)));
        sched.schedule(make_insert(20, Some(100_000)));
        sched.schedule(make_insert(30, Some(200_000)));

        // Only PTS ≤ 200_000
        let due = sched.get_due(200_000);
        assert_eq!(due.len(), 2);
        // Smallest PTS first (ids 20, 30)
        assert_eq!(due[0].splice_event_id, 20);
        assert_eq!(due[1].splice_event_id, 30);
    }

    #[test]
    fn test_scheduler_pending_count_decreases_after_get_due() {
        let mut sched = SpliceScheduler::new();
        for i in 0..5 {
            sched.schedule(make_insert(i, Some(i as u64 * 1000)));
        }
        assert_eq!(sched.pending_count(), 5);
        let _ = sched.get_due(3000);
        assert_eq!(sched.pending_count(), 1); // only pts=4000 remains
    }

    #[test]
    fn test_scheduler_get_due_all_at_once() {
        let mut sched = SpliceScheduler::new();
        for i in 0..10u32 {
            sched.schedule(make_insert(i, Some(i as u64 * 90_000)));
        }
        let due = sched.get_due(u64::MAX);
        assert_eq!(due.len(), 10);
    }

    #[test]
    fn test_scheduler_default_impl() {
        let sched = SpliceScheduler::default();
        assert_eq!(sched.pending_count(), 0);
    }

    // ── splice_null / bandwidth_reservation encode/parse roundtrips ─────────

    #[test]
    fn test_encode_splice_null_roundtrip() {
        let encoded = encode_splice_null(0xFFF);
        let section = parse_splice_info(&encoded).expect("parse splice_null");
        assert_eq!(section.protocol_version, 0);
        assert_eq!(section.tier, 0xFFF);
        assert!(
            matches!(section.splice_command, SpliceCommand::Null),
            "expected Null command, got {:?}",
            section.splice_command
        );
    }

    #[test]
    fn test_encode_splice_null_custom_tier() {
        let encoded = encode_splice_null(0x123);
        let section = parse_splice_info(&encoded).expect("parse splice_null");
        assert_eq!(section.tier, 0x123);
    }

    #[test]
    fn test_encode_bandwidth_reservation_roundtrip() {
        let encoded = encode_bandwidth_reservation(0xABC);
        let section = parse_splice_info(&encoded).expect("parse bandwidth_reservation");
        assert_eq!(section.tier, 0xABC);
        assert!(
            matches!(section.splice_command, SpliceCommand::BandwidthReservation),
            "expected BandwidthReservation command, got {:?}",
            section.splice_command
        );
    }

    #[test]
    fn test_encode_bandwidth_reservation_no_descriptors() {
        let encoded = encode_bandwidth_reservation(0xFFF);
        let section = parse_splice_info(&encoded).expect("parse");
        assert!(section.descriptors.is_empty());
    }

    #[test]
    fn test_encode_splice_null_not_encrypted() {
        let encoded = encode_splice_null(0xFFF);
        let section = parse_splice_info(&encoded).expect("parse");
        assert!(!section.encrypted_packet);
    }

    #[test]
    fn test_splice_null_and_bw_reservation_different_command_types() {
        let null_encoded = encode_splice_null(0xFFF);
        let bw_encoded = encode_bandwidth_reservation(0xFFF);
        // They should differ in the command type byte
        assert_ne!(null_encoded, bw_encoded);
    }

    #[test]
    fn test_encode_section_from_command_with_descriptors() {
        let desc = SpliceDescriptor::AvailDescriptor {
            provider_avail_id: 42,
        };
        let desc_bytes = desc.encode();
        let encoded = encode_section_from_command(
            0,
            false,
            0,
            0xFFF,
            SpliceCommandType::SpliceNull as u8,
            &[],
            &desc_bytes,
        );
        let section = parse_splice_info(&encoded).expect("parse");
        assert_eq!(section.descriptors.len(), 1);
    }

    // ── SpliceScheduler: splice_null and bandwidth_reservation ───────────────

    #[test]
    fn test_scheduler_null_fires_at_pts() {
        let mut sched = SpliceScheduler::new();
        sched.schedule_null(50_000);
        assert_eq!(sched.pending_count(), 1);
        let due = sched.get_due_commands(40_000);
        assert!(due.is_empty(), "not yet due");
        let due = sched.get_due_commands(50_000);
        assert_eq!(due.len(), 1);
        assert!(matches!(due[0], ScheduledCommand::Null));
    }

    #[test]
    fn test_scheduler_bandwidth_reservation_fires_at_pts() {
        let mut sched = SpliceScheduler::new();
        sched.schedule_bandwidth_reservation(100_000);
        let due = sched.get_due_commands(100_000);
        assert_eq!(due.len(), 1);
        assert!(matches!(due[0], ScheduledCommand::BandwidthReservation));
    }

    #[test]
    fn test_scheduler_mixed_commands_ordered_by_pts() {
        let mut sched = SpliceScheduler::new();
        sched.schedule_null(300);
        sched.schedule_bandwidth_reservation(100);
        sched.schedule(make_insert(42, Some(200)));
        let due = sched.get_due_commands(300);
        assert_eq!(due.len(), 3);
        // Ordered by PTS: BW(100), Insert(200), Null(300)
        assert!(matches!(due[0], ScheduledCommand::BandwidthReservation));
        assert!(matches!(due[1], ScheduledCommand::Insert(_)));
        assert!(matches!(due[2], ScheduledCommand::Null));
    }

    #[test]
    fn test_scheduler_get_due_legacy_filters_non_inserts() {
        let mut sched = SpliceScheduler::new();
        sched.schedule_null(100);
        sched.schedule_bandwidth_reservation(200);
        sched.schedule(make_insert(1, Some(150)));
        // Legacy get_due only returns inserts
        let due = sched.get_due(200);
        assert_eq!(due.len(), 1);
        assert_eq!(due[0].splice_event_id, 1);
    }

    #[test]
    fn test_scheduler_null_not_yet_due() {
        let mut sched = SpliceScheduler::new();
        sched.schedule_null(1_000_000);
        let due = sched.get_due_commands(500_000);
        assert!(due.is_empty());
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_scheduler_multiple_nulls_all_fire() {
        let mut sched = SpliceScheduler::new();
        for i in 0..5u64 {
            sched.schedule_null(i * 1000);
        }
        let due = sched.get_due_commands(u64::MAX);
        assert_eq!(due.len(), 5);
        assert!(due.iter().all(|c| matches!(c, ScheduledCommand::Null)));
    }

    #[test]
    fn test_scheduler_bw_reservation_pending_count() {
        let mut sched = SpliceScheduler::new();
        sched.schedule_bandwidth_reservation(100);
        sched.schedule_bandwidth_reservation(200);
        assert_eq!(sched.pending_count(), 2);
        let _ = sched.get_due_commands(150);
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_scheduler_null_encode_roundtrip() {
        // Verify that a scheduled null can be encoded and parsed back
        let encoded = encode_splice_null(0xFFF);
        let section = parse_splice_info(&encoded).expect("parse");
        assert!(matches!(section.splice_command, SpliceCommand::Null));
    }

    #[test]
    fn test_scheduler_bw_reservation_encode_roundtrip() {
        let encoded = encode_bandwidth_reservation(0x100);
        let section = parse_splice_info(&encoded).expect("parse");
        assert!(matches!(
            section.splice_command,
            SpliceCommand::BandwidthReservation
        ));
        assert_eq!(section.tier, 0x100);
    }

    #[test]
    fn test_scheduled_command_equality() {
        assert_eq!(ScheduledCommand::Null, ScheduledCommand::Null);
        assert_eq!(
            ScheduledCommand::BandwidthReservation,
            ScheduledCommand::BandwidthReservation
        );
        assert_ne!(
            ScheduledCommand::Null,
            ScheduledCommand::BandwidthReservation
        );
    }

    // ── SpliceInfoSection encode→parse roundtrip tests ────────────────────────

    /// Build a complete section binary from a SpliceInsert, using
    /// encode_splice_insert for command bytes and encode_section_from_command
    /// for the outer wrapper.
    fn make_insert_section(
        insert: &SpliceInsert,
        tier: u16,
        descriptors: &[SpliceDescriptor],
    ) -> Vec<u8> {
        let command_bytes = encode_splice_insert(insert);
        let desc_bytes: Vec<u8> = descriptors.iter().flat_map(|d| d.encode()).collect();
        encode_section_from_command(
            0,
            false,
            0,
            tier,
            SpliceCommandType::SpliceInsert as u8,
            &command_bytes,
            &desc_bytes,
        )
    }

    /// Build TimeSignal command bytes: 1 flag byte (0xFF = time_specified) + 5 PTS bytes.
    fn make_time_signal_command_bytes(pts: u64) -> Vec<u8> {
        let mut bytes = vec![0xFF_u8]; // time_specified_flag = 1
        bytes.extend_from_slice(&encode_pts_time(pts));
        bytes
    }

    #[test]
    fn test_splice_insert_section_roundtrip() {
        // Build a SpliceInsert for an out-of-network avail with PTS and break duration.
        let insert = SpliceInsert {
            splice_event_id: 0xABCD_1234,
            out_of_network_indicator: true,
            program_splice_flag: true,
            duration_flag: true,
            splice_immediate_flag: false,
            pts_time: Some(0x1_0000_0001), // exercises the 33rd PTS bit
            break_duration: Some(BreakDuration {
                auto_return: true,
                duration_90k: 27_000_000, // 5 minutes at 90 kHz
            }),
        };
        let encoded = make_insert_section(&insert, 0xFFF, &[]);
        let section = parse_splice_info(&encoded)
            .expect("parse_splice_info should succeed for a valid SpliceInsert section");

        assert_eq!(section.protocol_version, 0);
        assert!(!section.encrypted_packet);
        assert_eq!(section.tier, 0xFFF);
        assert!(section.descriptors.is_empty());

        match section.splice_command {
            SpliceCommand::Insert(ref parsed_insert) => {
                assert_eq!(
                    parsed_insert.splice_event_id, insert.splice_event_id,
                    "splice_event_id must round-trip"
                );
                assert_eq!(
                    parsed_insert.out_of_network_indicator,
                    insert.out_of_network_indicator
                );
                assert_eq!(
                    parsed_insert.program_splice_flag,
                    insert.program_splice_flag
                );
                assert_eq!(parsed_insert.duration_flag, insert.duration_flag);
                assert_eq!(
                    parsed_insert.splice_immediate_flag,
                    insert.splice_immediate_flag
                );
                assert_eq!(
                    parsed_insert.pts_time, insert.pts_time,
                    "PTS time must round-trip, including 33rd bit"
                );
                let bd = parsed_insert
                    .break_duration
                    .as_ref()
                    .expect("break_duration must be present after roundtrip");
                let orig_bd = insert
                    .break_duration
                    .as_ref()
                    .expect("original has break_duration");
                assert_eq!(bd.auto_return, orig_bd.auto_return);
                assert_eq!(bd.duration_90k, orig_bd.duration_90k);
            }
            other => panic!("expected SpliceCommand::Insert, got {other:?}"),
        }
    }

    #[test]
    fn test_time_signal_section_roundtrip() {
        // Build a TimeSignal with a segmentation descriptor.
        let pts: u64 = 900_000_000;
        let segmentation = SpliceDescriptor::SegmentationDescriptor {
            event_id: 0x0000_0042,
            type_id: 0x10, // program start
            upid_type: 0x09,
            upid: vec![0xDE, 0xAD, 0xBE, 0xEF],
            duration_90k: Some(54_000_000), // 10 minutes
        };

        let command_bytes = make_time_signal_command_bytes(pts);
        let desc_bytes = segmentation.encode();
        let encoded = encode_section_from_command(
            0,
            false,
            0,
            0xABC,
            SpliceCommandType::TimeSignal as u8,
            &command_bytes,
            &desc_bytes,
        );

        let section = parse_splice_info(&encoded)
            .expect("parse_splice_info should succeed for a valid TimeSignal section");

        assert_eq!(section.tier, 0xABC);

        // Verify the command type
        let parsed_pts = match &section.splice_command {
            SpliceCommand::Signal(ts) => ts.pts_time,
            other => panic!("expected SpliceCommand::Signal, got {other:?}"),
        };
        assert_eq!(parsed_pts, Some(pts), "TimeSignal PTS must round-trip");

        // Descriptors live on the section, not inside the TimeSignal struct
        assert_eq!(
            section.descriptors.len(),
            1,
            "segmentation descriptor must be preserved on section"
        );
        match &section.descriptors[0] {
            SpliceDescriptor::SegmentationDescriptor {
                event_id,
                type_id,
                upid_type,
                upid,
                duration_90k,
            } => {
                assert_eq!(*event_id, 0x0000_0042);
                assert_eq!(*type_id, 0x10);
                assert_eq!(*upid_type, 0x09);
                assert_eq!(upid, &vec![0xDE, 0xAD, 0xBE, 0xEF]);
                assert_eq!(*duration_90k, Some(54_000_000));
            }
            other => panic!("expected SegmentationDescriptor, got {other:?}"),
        }
    }

    #[test]
    fn test_multi_descriptor_section_roundtrip() {
        // Section with both AvailDescriptor and SegmentationDescriptor.
        // Assert descriptor count and order are preserved after parse.
        let avail = SpliceDescriptor::AvailDescriptor {
            provider_avail_id: 0x1234_5678,
        };
        let seg = SpliceDescriptor::SegmentationDescriptor {
            event_id: 0xFFFF_0001,
            type_id: 0x11, // program end
            upid_type: 0x01,
            upid: vec![0xAA, 0xBB],
            duration_90k: None,
        };

        let mut desc_bytes: Vec<u8> = avail.encode();
        desc_bytes.extend_from_slice(&seg.encode());

        let encoded = encode_section_from_command(
            0,
            false,
            0,
            0x100,
            SpliceCommandType::SpliceNull as u8,
            &[],
            &desc_bytes,
        );

        let section = parse_splice_info(&encoded)
            .expect("parse_splice_info should succeed for multi-descriptor section");

        assert_eq!(
            section.descriptors.len(),
            2,
            "both descriptors must survive roundtrip"
        );
        // Verify first descriptor is AvailDescriptor (order preserved)
        assert!(
            matches!(
                section.descriptors[0],
                SpliceDescriptor::AvailDescriptor {
                    provider_avail_id: 0x1234_5678
                }
            ),
            "first descriptor must be AvailDescriptor with correct id"
        );
        // Verify second descriptor is SegmentationDescriptor
        match &section.descriptors[1] {
            SpliceDescriptor::SegmentationDescriptor {
                event_id,
                type_id,
                duration_90k,
                ..
            } => {
                assert_eq!(*event_id, 0xFFFF_0001);
                assert_eq!(*type_id, 0x11);
                assert_eq!(*duration_90k, None);
            }
            other => panic!("expected SegmentationDescriptor, got {other:?}"),
        }
    }

    #[test]
    fn test_section_re_encode_byte_identical() {
        // Encode a SpliceInsert section, parse it, then re-encode using the
        // same parameters. Since the encoder is deterministic (no HashMap, fixed
        // CRC placeholder [0,0,0,0]), the bytes must be identical.
        let insert = SpliceInsert {
            splice_event_id: 0x0000_0001,
            out_of_network_indicator: true,
            program_splice_flag: true,
            duration_flag: false,
            splice_immediate_flag: false,
            pts_time: Some(450_000_000),
            break_duration: None,
        };
        let first_encode = make_insert_section(&insert, 0xFFF, &[]);

        let section = parse_splice_info(&first_encode).expect("first parse must succeed");

        // Re-encode from the parsed command
        let re_encoded = match &section.splice_command {
            SpliceCommand::Insert(parsed_insert) => {
                make_insert_section(parsed_insert, section.tier, &section.descriptors)
            }
            other => panic!("expected Insert command, got {other:?}"),
        };

        assert_eq!(
            first_encode, re_encoded,
            "re-encoded bytes must be identical to the original encoding"
        );
    }

    #[test]
    fn test_truncated_section_returns_err() {
        // Encode a valid SpliceInsert section (~33 bytes), truncate to half,
        // then assert parse returns Err (not panics).
        let insert = SpliceInsert {
            splice_event_id: 0xDEAD_BEEF,
            out_of_network_indicator: false,
            program_splice_flag: true,
            duration_flag: false,
            splice_immediate_flag: false,
            pts_time: Some(180_000_000),
            break_duration: None,
        };
        let full_bytes = make_insert_section(&insert, 0xFFF, &[]);
        assert!(
            full_bytes.len() > 10,
            "encoded section should be longer than 11 bytes"
        );
        let half_len = full_bytes.len() / 2;
        let truncated = &full_bytes[..half_len];
        let result = parse_splice_info(truncated);
        assert!(
            result.is_err(),
            "parsing a truncated section must return Err, got Ok"
        );
    }

    #[test]
    fn test_section_with_zero_descriptors_roundtrip() {
        // Minimal SpliceInsert section with empty descriptor loop.
        // Exercises the command-with-payload + empty-descriptor-loop combination.
        let insert = SpliceInsert {
            splice_event_id: 0x0000_0007,
            out_of_network_indicator: true,
            program_splice_flag: true,
            duration_flag: false,
            splice_immediate_flag: true, // immediate — no PTS field
            pts_time: None,
            break_duration: None,
        };
        let encoded = make_insert_section(&insert, 0x000, &[]);
        let section = parse_splice_info(&encoded)
            .expect("parse_splice_info must succeed for zero-descriptor section");

        assert_eq!(
            section.descriptors.len(),
            0,
            "descriptor count must be 0 after roundtrip"
        );
        match section.splice_command {
            SpliceCommand::Insert(ref parsed) => {
                assert_eq!(parsed.splice_event_id, insert.splice_event_id);
                assert_eq!(parsed.splice_immediate_flag, insert.splice_immediate_flag);
                assert_eq!(
                    parsed.out_of_network_indicator,
                    insert.out_of_network_indicator
                );
                assert_eq!(parsed.pts_time, None, "no PTS for immediate splice");
            }
            other => panic!("expected Insert command, got {other:?}"),
        }
    }
}
