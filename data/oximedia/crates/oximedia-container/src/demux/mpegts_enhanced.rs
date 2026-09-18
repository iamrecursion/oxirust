//! Enhanced MPEG-TS demuxer with complete PAT/PMT/PES parsing.
//!
//! This module provides a self-contained, allocation-minimising MPEG-TS parser
//! that can operate on byte slices without an async `MediaSource`.  It is
//! designed for probing, testing, and offline analysis where the full
//! `MpegTsDemuxer` async pipeline is not required.
//!
//! # Overview
//!
//! ```text
//! TsDemuxer::new()
//!   .feed(&raw_bytes)       → Vec<TsPacket>   (parsed 188-byte packets)
//!   .stream_info()          → &TsStreamInfo   (PAT / PMT / per-PID statistics)
//!   .duration_ms()          → Option<u64>     (estimated from first/last PTS)
//! ```
//!
//! Lower-level free functions (`parse_ts_packet`, `parse_pat`, `parse_pmt`)
//! are also exposed for unit-testing and reuse.

#![forbid(unsafe_code)]

use std::collections::HashMap;

// ─── Public types ─────────────────────────────────────────────────────────────

/// A parsed 188-byte MPEG-TS transport packet.
#[derive(Debug, Clone)]
pub struct TsPacket {
    /// 13-bit packet identifier.
    pub pid: u16,
    /// Payload Unit Start Indicator.
    pub payload_unit_start: bool,
    /// 4-bit continuity counter.
    pub continuity_counter: u8,
    /// Whether an adaptation field is present.
    pub has_adaptation: bool,
    /// Whether a payload is present.
    pub has_payload: bool,
    /// Raw payload bytes (empty if `has_payload` is false).
    pub payload: Vec<u8>,
    /// PTS extracted from a PES header, if this packet starts a PES unit.
    pub pts: Option<u64>,
    /// DTS extracted from a PES header, if present.
    pub dts: Option<u64>,
}

/// Program Association Table.
#[derive(Debug, Clone)]
pub struct Pat {
    /// Transport stream ID.
    pub transport_stream_id: u16,
    /// List of (program_number, pmt_pid) entries.
    pub programs: Vec<PatEntry>,
}

/// Single entry in the PAT.
#[derive(Debug, Clone)]
pub struct PatEntry {
    /// Program number (non-zero; 0 is the NIT PID entry).
    pub program_number: u16,
    /// PID of the PMT for this program.
    pub pmt_pid: u16,
}

/// Program Map Table.
#[derive(Debug, Clone)]
pub struct Pmt {
    /// Program number this PMT describes.
    pub program_number: u16,
    /// PID carrying the Program Clock Reference.
    pub pcr_pid: u16,
    /// Elementary streams in this program.
    pub streams: Vec<PmtStream>,
}

/// One elementary stream entry in a PMT.
#[derive(Debug, Clone)]
pub struct PmtStream {
    /// ISO 13818-1 stream type byte.
    ///
    /// Common values: 0x1B = H.264, 0x24 = H.265, 0x06 = PES private (Opus),
    /// 0x81 = AC-3, 0x85 = AV1 (user-defined), 0x84 = VP9 (user-defined).
    pub stream_type: u8,
    /// Elementary PID.
    pub elementary_pid: u16,
    /// Raw descriptor bytes for this stream.
    pub descriptors: Vec<u8>,
}

/// Per-PID statistics accumulated during a `TsDemuxer::feed` pass.
#[derive(Debug, Clone, Default)]
pub struct PidInfo {
    /// PID value.
    pub pid: u16,
    /// ISO 13818-1 stream type, if resolved via PMT.
    pub stream_type: Option<u8>,
    /// Total TS packets observed on this PID.
    pub packet_count: u64,
    /// Total payload bytes observed on this PID.
    pub total_bytes: u64,
    /// Earliest PTS seen on this PID (90 kHz clock).
    pub pts_first: Option<u64>,
    /// Most-recent PTS seen on this PID.
    pub pts_last: Option<u64>,
}

/// Aggregate stream-discovery state maintained by `TsDemuxer`.
#[derive(Debug, Clone, Default)]
pub struct TsStreamInfo {
    /// Most-recently parsed PAT (if seen).
    pub pat: Option<Pat>,
    /// Map of `program_number → Pmt`.
    pub pmts: HashMap<u16, Pmt>,
    /// Map of `pid → PidInfo`.
    pub pids: HashMap<u16, PidInfo>,
}

// ─── Free functions ──────────────────────────────────────────────────────────

/// Parses a single 188-byte TS packet from `data` starting at `offset`.
///
/// Returns `None` if:
/// - `offset + 188 > data.len()`
/// - The sync byte at `data[offset]` is not `0x47`
///
/// PTS/DTS are extracted when `payload_unit_start` is set and the payload
/// begins with a valid PES start code (`00 00 01`).
#[must_use]
pub fn parse_ts_packet(data: &[u8], offset: usize) -> Option<TsPacket> {
    const TS_SIZE: usize = 188;
    const SYNC: u8 = 0x47;

    if offset + TS_SIZE > data.len() {
        return None;
    }
    let pkt = &data[offset..offset + TS_SIZE];
    if pkt[0] != SYNC {
        return None;
    }

    let transport_error = (pkt[1] & 0x80) != 0;
    if transport_error {
        // Discard corrupted packets
        return None;
    }

    let payload_unit_start = (pkt[1] & 0x40) != 0;
    let pid = (u16::from(pkt[1] & 0x1F) << 8) | u16::from(pkt[2]);
    let adaptation_field_control = (pkt[3] >> 4) & 0x03;
    let continuity_counter = pkt[3] & 0x0F;

    let has_adaptation = (adaptation_field_control & 0x02) != 0;
    let has_payload = (adaptation_field_control & 0x01) != 0;

    // Determine payload start offset within the packet
    let mut payload_start = 4usize;
    if has_adaptation {
        let af_len = pkt[4] as usize;
        payload_start = 5 + af_len;
        if payload_start > TS_SIZE {
            payload_start = TS_SIZE;
        }
    }

    let payload = if has_payload && payload_start < TS_SIZE {
        pkt[payload_start..].to_vec()
    } else {
        Vec::new()
    };

    // Extract PTS/DTS from PES header if this is a payload unit start
    let (pts, dts) = if payload_unit_start && payload.len() >= 9 {
        extract_pes_timestamps(&payload)
    } else {
        (None, None)
    };

    Some(TsPacket {
        pid,
        payload_unit_start,
        continuity_counter,
        has_adaptation,
        has_payload,
        payload,
        pts,
        dts,
    })
}

/// Parses a Program Association Table section.
///
/// `payload` should be the raw PSI section bytes starting at the table-id byte
/// (pointer field already consumed).  Returns `None` on any parse failure.
#[must_use]
pub fn parse_pat(payload: &[u8]) -> Option<Pat> {
    if payload.len() < 8 {
        return None;
    }
    if payload[0] != 0x00 {
        return None; // not PAT table_id
    }

    let section_length = (usize::from(payload[1] & 0x0F) << 8) | usize::from(payload[2]);
    if payload.len() < section_length + 3 {
        return None;
    }

    // A malformed PAT may declare `section_length == 0`, making the
    // CRC-excluding end offset `3 + section_length - 4` underflow (usize) and
    // spin the entry loop far out of bounds. `psi.rs` is implicitly guarded by
    // its CRC-length check; this enhanced fast-path has none, so require at
    // least one byte of section body (equivalently `section_length >= 1`).
    if section_length == 0 {
        return None;
    }

    let transport_stream_id = (u16::from(payload[3]) << 8) | u16::from(payload[4]);

    // Entries start at byte 8, end 4 bytes before section end (CRC)
    let entries_end = 3 + section_length - 4;
    let mut programs = Vec::new();
    let mut offset = 8usize;
    while offset + 4 <= entries_end {
        let program_number = (u16::from(payload[offset]) << 8) | u16::from(payload[offset + 1]);
        let pmt_pid = (u16::from(payload[offset + 2] & 0x1F) << 8) | u16::from(payload[offset + 3]);
        if program_number != 0 {
            programs.push(PatEntry {
                program_number,
                pmt_pid,
            });
        }
        offset += 4;
    }

    Some(Pat {
        transport_stream_id,
        programs,
    })
}

/// Parses a Program Map Table section.
///
/// `payload` should start at the table-id byte (pointer field consumed).
/// Returns `None` on any parse failure.
#[must_use]
pub fn parse_pmt(payload: &[u8]) -> Option<Pmt> {
    if payload.len() < 12 {
        return None;
    }
    if payload[0] != 0x02 {
        return None; // not PMT table_id
    }

    let section_length = (usize::from(payload[1] & 0x0F) << 8) | usize::from(payload[2]);
    if payload.len() < section_length + 3 {
        return None;
    }

    // A malformed PMT may declare `section_length == 0`, making the
    // CRC-excluding end offset `3 + section_length - 4` underflow (usize).
    // Require at least one byte of section body (equivalently
    // `section_length >= 1`) before the subtraction below.
    if section_length == 0 {
        return None;
    }

    let program_number = (u16::from(payload[3]) << 8) | u16::from(payload[4]);
    let pcr_pid = (u16::from(payload[8] & 0x1F) << 8) | u16::from(payload[9]);
    let program_info_length = (usize::from(payload[10] & 0x0F) << 8) | usize::from(payload[11]);

    let streams_start = 12 + program_info_length;
    let streams_end = 3 + section_length - 4; // exclude CRC

    if streams_start > streams_end {
        return None;
    }

    let mut streams = Vec::new();
    let mut offset = streams_start;
    while offset + 5 <= streams_end {
        let stream_type = payload[offset];
        let elementary_pid =
            (u16::from(payload[offset + 1] & 0x1F) << 8) | u16::from(payload[offset + 2]);
        let es_info_length =
            (usize::from(payload[offset + 3] & 0x0F) << 8) | usize::from(payload[offset + 4]);

        let desc_end = offset + 5 + es_info_length;
        let descriptors = if desc_end <= streams_end {
            payload[offset + 5..desc_end].to_vec()
        } else {
            Vec::new()
        };

        streams.push(PmtStream {
            stream_type,
            elementary_pid,
            descriptors,
        });

        offset += 5 + es_info_length;
    }

    Some(Pmt {
        program_number,
        pcr_pid,
        streams,
    })
}

// ─── PES timestamp extraction ─────────────────────────────────────────────────

/// Attempts to extract PTS and DTS from the beginning of a PES payload.
///
/// Returns `(pts, dts)` where each is in 90 kHz units.  Both may be `None`.
fn extract_pes_timestamps(payload: &[u8]) -> (Option<u64>, Option<u64>) {
    // PES start code: 00 00 01 xx (stream_id)
    if payload.len() < 9 {
        return (None, None);
    }
    if payload[0] != 0x00 || payload[1] != 0x00 || payload[2] != 0x01 {
        return (None, None);
    }
    // payload[3] = stream_id
    // payload[4..5] = PES_packet_length
    // payload[6] = marker + scrambling + priority + alignment + copyright + original
    // payload[7] = PTS_DTS_flags (bits 7:6), ESCR, ES_rate, DSM, additional_copy, PES_CRC, extension
    // payload[8] = PES_header_data_length

    if payload.len() < 14 {
        return (None, None);
    }

    let pts_dts_flags = (payload[7] >> 6) & 0x03;
    let header_data_length = payload[8] as usize;
    // Guard: need at least 9 + header_data_length bytes
    if payload.len() < 9 + header_data_length {
        return (None, None);
    }

    let mut pts: Option<u64> = None;
    let mut dts: Option<u64> = None;

    if pts_dts_flags >= 0x02 && payload.len() >= 9 + 5 {
        pts = Some(decode_timestamp(&payload[9..14]));
    }
    if pts_dts_flags == 0x03 && payload.len() >= 14 + 5 {
        dts = Some(decode_timestamp(&payload[14..19]));
    }

    (pts, dts)
}

/// Decodes a 33-bit PTS/DTS value from 5 bytes of PES header.
///
/// Bits layout (MPEG-2 spec):
/// ```text
///   byte[0]: XXXX_MMMM (M = bits 32..30)
///   byte[1]: MMMM_MMMM (M = bits 29..22)
///   byte[2]: MMMM_MMMX (M = bits 21..15, X = marker)
///   byte[3]: LLLL_LLLL (L = bits 14..7)
///   byte[4]: LLLL_LLLX (L = bits 6..0, X = marker)
/// ```
fn decode_timestamp(b: &[u8]) -> u64 {
    let hi = (u64::from(b[0]) & 0x0E) << 29;
    let mid = u64::from(b[1]) << 22;
    let lo1 = (u64::from(b[2]) & 0xFE) << 14;
    let lo2 = u64::from(b[3]) << 7;
    let lo3 = u64::from(b[4]) >> 1;
    hi | mid | lo1 | lo2 | lo3
}

// ─── Section assembler ────────────────────────────────────────────────────────

/// Minimal section assembler used internally for multi-packet PSI sections.
#[derive(Debug, Default)]
struct SectionBuf {
    data: Vec<u8>,
    expected: Option<usize>,
}

impl SectionBuf {
    fn push(&mut self, payload: &[u8], unit_start: bool) -> Option<Vec<u8>> {
        if unit_start {
            self.data.clear();
            self.expected = None;
            // Skip pointer field
            let pointer = *payload.first()? as usize;
            let start = 1 + pointer;
            if start >= payload.len() {
                return None;
            }
            self.data.extend_from_slice(&payload[start..]);
        } else {
            self.data.extend_from_slice(payload);
        }

        if self.expected.is_none() && self.data.len() >= 3 {
            let sec_len = (usize::from(self.data[1] & 0x0F) << 8) | usize::from(self.data[2]);
            self.expected = Some(sec_len + 3);
        }

        if let Some(exp) = self.expected {
            if self.data.len() >= exp {
                let section = self.data[..exp].to_vec();
                self.data.clear();
                self.expected = None;
                return Some(section);
            }
        }
        None
    }
}

// ─── TsDemuxer ────────────────────────────────────────────────────────────────

const PAT_PID: u16 = 0x0000;

/// Stateful MPEG-TS demuxer that operates on in-memory byte slices.
///
/// Unlike the async `MpegTsDemuxer`, this type can be used synchronously
/// and is suitable for probing and testing.
#[derive(Debug, Default)]
pub struct TsDemuxer {
    info: TsStreamInfo,
    section_bufs: HashMap<u16, SectionBuf>,
    pmt_pids: Vec<u16>,
}

impl TsDemuxer {
    /// Creates a new `TsDemuxer`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds raw TS bytes into the demuxer and returns all successfully parsed
    /// `TsPacket` values.
    ///
    /// Internally updates PAT/PMT state and per-PID statistics.  Bytes that do
    /// not align to 188-byte boundaries or contain invalid sync bytes are
    /// skipped.
    pub fn feed(&mut self, data: &[u8]) -> Vec<TsPacket> {
        let mut packets = Vec::new();
        let mut offset = 0usize;
        while offset + 188 <= data.len() {
            if data[offset] != 0x47 {
                offset += 1;
                continue;
            }
            if let Some(pkt) = parse_ts_packet(data, offset) {
                self.ingest_packet(&pkt);
                packets.push(pkt);
            }
            offset += 188;
        }
        packets
    }

    /// Returns the current stream discovery state.
    #[must_use]
    pub fn stream_info(&self) -> &TsStreamInfo {
        &self.info
    }

    /// Estimates the stream duration in milliseconds from the first and last
    /// observed PTS across all PIDs.
    ///
    /// Returns `None` if fewer than two PTS values have been seen.
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        let mut global_first: Option<u64> = None;
        let mut global_last: Option<u64> = None;

        for pid_info in self.info.pids.values() {
            if let Some(f) = pid_info.pts_first {
                global_first = Some(match global_first {
                    Some(cur) => cur.min(f),
                    None => f,
                });
            }
            if let Some(l) = pid_info.pts_last {
                global_last = Some(match global_last {
                    Some(cur) => cur.max(l),
                    None => l,
                });
            }
        }

        match (global_first, global_last) {
            (Some(f), Some(l)) if l > f => {
                // PTS is in 90 kHz ticks → ms
                Some((l - f) / 90)
            }
            _ => None,
        }
    }

    // ─── Internal helpers ─────────────────────────────────────────────────

    fn ingest_packet(&mut self, pkt: &TsPacket) {
        // Update PID statistics
        {
            let pid_info = self.info.pids.entry(pkt.pid).or_insert_with(|| PidInfo {
                pid: pkt.pid,
                ..Default::default()
            });
            pid_info.packet_count += 1;
            pid_info.total_bytes += pkt.payload.len() as u64;
            if let Some(pts) = pkt.pts {
                if pid_info.pts_first.is_none() {
                    pid_info.pts_first = Some(pts);
                }
                pid_info.pts_last = Some(pts);
            }
        }

        if pkt.pid == PAT_PID {
            self.handle_pat(pkt);
            return;
        }

        if self.pmt_pids.contains(&pkt.pid) {
            self.handle_pmt(pkt);
        }
    }

    fn handle_pat(&mut self, pkt: &TsPacket) {
        let buf = self.section_bufs.entry(PAT_PID).or_default();
        if let Some(section) = buf.push(&pkt.payload, pkt.payload_unit_start) {
            if let Some(pat) = parse_pat(&section) {
                // Register PMT PIDs
                for entry in &pat.programs {
                    if !self.pmt_pids.contains(&entry.pmt_pid) {
                        self.pmt_pids.push(entry.pmt_pid);
                    }
                }
                self.info.pat = Some(pat);
            }
        }
    }

    fn handle_pmt(&mut self, pkt: &TsPacket) {
        let buf = self.section_bufs.entry(pkt.pid).or_default();
        if let Some(section) = buf.push(&pkt.payload, pkt.payload_unit_start) {
            if let Some(pmt) = parse_pmt(&section) {
                // Record stream_type for each elementary PID
                for stream in &pmt.streams {
                    let pid_info =
                        self.info
                            .pids
                            .entry(stream.elementary_pid)
                            .or_insert_with(|| PidInfo {
                                pid: stream.elementary_pid,
                                ..Default::default()
                            });
                    pid_info.stream_type = Some(stream.stream_type);
                }
                self.info.pmts.insert(pmt.program_number, pmt);
            }
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Security regression tests (P0 #1: section_length underflow) ──────

    #[test]
    fn parse_pat_zero_section_length_is_rejected() {
        // section_length == 0 makes `3 + section_length - 4` underflow (usize)
        // and spin the entry loop out of bounds. Must return None, not panic.
        let payload = [0x00u8, 0x00, 0x00, 0, 0, 0, 0, 0]; // table_id=PAT, len=0
        assert!(parse_pat(&payload).is_none());
    }

    #[test]
    fn parse_pmt_zero_section_length_is_rejected() {
        // Same underflow guard for the PMT fast-path.
        let payload = [0x02u8, 0x00, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // table_id=PMT, len=0
        assert!(parse_pmt(&payload).is_none());
    }

    // ─── Helpers ──────────────────────────────────────────────────────────

    /// Builds a minimal 188-byte TS packet with `payload_only` AFC.
    fn make_ts_packet(pid: u16, pusi: bool, cc: u8, payload: &[u8]) -> Vec<u8> {
        let mut pkt = vec![0u8; 188];
        pkt[0] = 0x47; // sync
        let pusi_bit: u8 = if pusi { 0x40 } else { 0x00 };
        pkt[1] = pusi_bit | ((pid >> 8) as u8 & 0x1F);
        pkt[2] = (pid & 0xFF) as u8;
        pkt[3] = 0x10 | (cc & 0x0F); // AFC=01 (payload only)
        let copy_len = payload.len().min(184);
        pkt[4..4 + copy_len].copy_from_slice(&payload[..copy_len]);
        pkt
    }

    /// Builds a minimal PAT section (one program, no CRC verification needed
    /// because `parse_pat` does not verify CRC — it trusts the section assembler).
    fn make_pat_section(tsid: u16, program_number: u16, pmt_pid: u16) -> Vec<u8> {
        let mut s = Vec::new();
        // table_id=0x00
        s.push(0x00);
        // section_syntax_indicator=1, reserved, section_length (2 entries + 4 crc + 5 fixed = 13)
        let _section_length: u16 = 4 + 4 + 4; // fixed (5) – 3 already counted + 4 program entry + 4 CRC = 13
                                              // Easier: entries=(4), fixed=(5 after length), CRC=4 → section_length = entries + fixed_tail
                                              // section_length = 5 (tsid+version+section) + 4 (program entry) + 4 (CRC) = 13
        let sl: u16 = 13;
        s.push(0xB0 | ((sl >> 8) as u8 & 0x0F));
        s.push((sl & 0xFF) as u8);
        // transport_stream_id
        s.push((tsid >> 8) as u8);
        s.push((tsid & 0xFF) as u8);
        // version_number=0, current_next=1
        s.push(0xC1);
        // section_number, last_section_number
        s.push(0x00);
        s.push(0x00);
        // program entry
        s.push((program_number >> 8) as u8);
        s.push((program_number & 0xFF) as u8);
        s.push(0xE0 | ((pmt_pid >> 8) as u8 & 0x1F));
        s.push((pmt_pid & 0xFF) as u8);
        // CRC (4 bytes) – we use 0 placeholder; parse_pat doesn't verify CRC
        s.extend_from_slice(&[0u8; 4]);
        s
    }

    /// Builds a minimal PMT section for one audio stream.
    fn make_pmt_section(
        program_number: u16,
        pcr_pid: u16,
        stream_type: u8,
        es_pid: u16,
    ) -> Vec<u8> {
        let mut s = Vec::new();
        // table_id=0x02
        s.push(0x02);
        // section_length: 9 (fixed after length) + 5 (stream entry) + 4 (CRC) = 18
        let sl: u16 = 18;
        s.push(0xB0 | ((sl >> 8) as u8 & 0x0F));
        s.push((sl & 0xFF) as u8);
        s.push((program_number >> 8) as u8);
        s.push((program_number & 0xFF) as u8);
        s.push(0xC1); // version=0, current
        s.push(0x00); // section_number
        s.push(0x00); // last_section_number
        s.push(0xE0 | ((pcr_pid >> 8) as u8 & 0x1F));
        s.push((pcr_pid & 0xFF) as u8);
        s.push(0xF0); // program_info_length=0
        s.push(0x00);
        // stream entry
        s.push(stream_type);
        s.push(0xE0 | ((es_pid >> 8) as u8 & 0x1F));
        s.push((es_pid & 0xFF) as u8);
        s.push(0xF0); // ES_info_length=0
        s.push(0x00);
        // CRC placeholder
        s.extend_from_slice(&[0u8; 4]);
        s
    }

    // 1. parse_ts_packet: sync byte check
    #[test]
    fn test_parse_ts_packet_bad_sync() {
        let mut data = vec![0u8; 188];
        data[0] = 0x00; // wrong sync
        assert!(parse_ts_packet(&data, 0).is_none());
    }

    // 2. parse_ts_packet: correct PID extraction
    #[test]
    fn test_parse_ts_packet_pid() {
        let pkt = make_ts_packet(0x0100, false, 0, &[]);
        let parsed = parse_ts_packet(&pkt, 0).expect("valid packet");
        assert_eq!(parsed.pid, 0x0100);
    }

    // 3. parse_ts_packet: PUSI flag
    #[test]
    fn test_parse_ts_packet_pusi() {
        let pkt = make_ts_packet(0x0200, true, 3, &[]);
        let parsed = parse_ts_packet(&pkt, 0).expect("valid packet");
        assert!(parsed.payload_unit_start);
        assert_eq!(parsed.continuity_counter, 3);
    }

    // 4. parse_ts_packet: offset into larger buffer
    #[test]
    fn test_parse_ts_packet_offset() {
        let mut buf = vec![0xFFu8; 200];
        let pkt = make_ts_packet(0x0042, false, 7, &[0xAB; 10]);
        buf[12..200].copy_from_slice(&pkt);
        let parsed = parse_ts_packet(&buf, 12).expect("valid packet");
        assert_eq!(parsed.pid, 0x0042);
        assert_eq!(parsed.continuity_counter, 7);
    }

    // 5. parse_pat: round-trip
    #[test]
    fn test_parse_pat_round_trip() {
        let section = make_pat_section(0x0001, 1, 0x0100);
        let pat = parse_pat(&section).expect("valid PAT");
        assert_eq!(pat.transport_stream_id, 0x0001);
        assert_eq!(pat.programs.len(), 1);
        assert_eq!(pat.programs[0].program_number, 1);
        assert_eq!(pat.programs[0].pmt_pid, 0x0100);
    }

    // 6. parse_pat: bad table_id → None
    #[test]
    fn test_parse_pat_bad_table_id() {
        let mut section = make_pat_section(1, 1, 0x0100);
        section[0] = 0xFF;
        assert!(parse_pat(&section).is_none());
    }

    // 7. parse_pmt: round-trip
    #[test]
    fn test_parse_pmt_round_trip() {
        let section = make_pmt_section(1, 0x01FF, 0x81, 0x0101);
        let pmt = parse_pmt(&section).expect("valid PMT");
        assert_eq!(pmt.program_number, 1);
        assert_eq!(pmt.pcr_pid, 0x01FF);
        assert_eq!(pmt.streams.len(), 1);
        assert_eq!(pmt.streams[0].stream_type, 0x81);
        assert_eq!(pmt.streams[0].elementary_pid, 0x0101);
    }

    // 8. parse_pmt: bad table_id → None
    #[test]
    fn test_parse_pmt_bad_table_id() {
        let mut section = make_pmt_section(1, 0x01FF, 0x81, 0x0101);
        section[0] = 0x00; // PAT id, not PMT
        assert!(parse_pmt(&section).is_none());
    }

    // 9. TsDemuxer::feed returns correct packet count
    #[test]
    fn test_demuxer_feed_count() {
        let mut demux = TsDemuxer::new();
        let mut stream = Vec::new();
        for i in 0..5u16 {
            stream.extend(make_ts_packet(0x0100 + i, false, 0, &[]));
        }
        let pkts = demux.feed(&stream);
        assert_eq!(pkts.len(), 5);
    }

    // 10. TsDemuxer discovers PAT
    #[test]
    fn test_demuxer_discovers_pat() {
        let mut demux = TsDemuxer::new();
        // Wrap PAT section in a TS packet with pointer=0x00
        let section = make_pat_section(1, 1, 0x0100);
        let mut payload = vec![0x00u8]; // pointer field = 0
        payload.extend_from_slice(&section);
        let pkt = make_ts_packet(0x0000, true, 0, &payload);
        demux.feed(&pkt);
        assert!(demux.stream_info().pat.is_some());
    }

    // 11. TsDemuxer duration_ms returns None with single PTS
    #[test]
    fn test_demuxer_duration_none_without_range() {
        let demux = TsDemuxer::new();
        assert!(demux.duration_ms().is_none());
    }

    // 12. decode_timestamp correctness
    #[test]
    fn test_decode_timestamp_known_value() {
        // Encoding: PTS = 0x1_2345_6789 (33-bit value)
        // 0010 1 0010 00100011 01000101 0110011 0 10001001 0
        // Construct the 5 bytes according to spec
        // pts_bits: 0x12345_6789 in 33 bits = binary representation
        let pts_val: u64 = 0x1_2345_6789;
        let b0: u8 = 0x20 | (((pts_val >> 29) & 0x0E) as u8) | 0x01; // marker bit
        let b1: u8 = ((pts_val >> 22) & 0xFF) as u8;
        let b2: u8 = (((pts_val >> 14) & 0xFE) as u8) | 0x01;
        let b3: u8 = ((pts_val >> 7) & 0xFF) as u8;
        let b4: u8 = (((pts_val << 1) & 0xFE) as u8) | 0x01;
        let decoded = decode_timestamp(&[b0, b1, b2, b3, b4]);
        assert_eq!(decoded, pts_val);
    }
}
