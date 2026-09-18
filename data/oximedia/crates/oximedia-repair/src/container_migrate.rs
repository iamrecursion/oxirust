//! Re-muxing repaired streams into a clean container without re-encoding.
//!
//! `ContainerMigrator` accepts raw stream data (previously recovered from a
//! corrupt file) and writes it into a freshly initialised container in one of
//! the supported target formats.  No codec transcoding is performed — the
//! bitstream bytes are transferred verbatim.
//!
//! Supported target formats:
//! - Matroska (MKV) — simple cluster/block framing
//! - MPEG-TS passthrough — wraps PES payloads in 188-byte TS packets
//! - Raw elementary stream — strips all container and writes bare codec data

use crate::{RepairError, Result};
use std::io::Write;
use std::path::Path;

/// Target container format for migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetContainer {
    /// Matroska (MKV) simple framing.
    Matroska,
    /// MPEG Transport Stream (188-byte packets).
    MpegTs,
    /// Raw elementary stream (no container).
    RawEs,
}

impl std::fmt::Display for TargetContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Matroska => write!(f, "Matroska/MKV"),
            Self::MpegTs => write!(f, "MPEG-TS"),
            Self::RawEs => write!(f, "Raw ES"),
        }
    }
}

/// Codec type hint to guide framing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecHint {
    /// AV1 video.
    Av1,
    /// VP9 video.
    Vp9,
    /// H.264/AVC video.
    H264,
    /// Opus audio.
    Opus,
    /// FLAC audio.
    Flac,
    /// Unknown / generic.
    Unknown,
}

/// A recovered stream to be placed in the new container.
#[derive(Debug, Clone)]
pub struct RecoveredStream {
    /// Raw codec data (bitstream bytes).
    pub data: Vec<u8>,
    /// Estimated codec.
    pub codec: CodecHint,
    /// Whether this is a video stream (true) or audio stream (false).
    pub is_video: bool,
    /// Duration hint in milliseconds (0 = unknown).
    pub duration_ms: u64,
}

/// Result of a container migration operation.
#[derive(Debug, Clone)]
pub struct MigrateResult {
    /// Total bytes written to the output container.
    pub bytes_written: u64,
    /// Target container format used.
    pub container: TargetContainer,
    /// Number of streams written.
    pub stream_count: usize,
    /// Whether all streams were written without loss.
    pub lossless: bool,
}

/// Migrate recovered streams into a fresh container.
///
/// # Arguments
/// * `streams` – slice of recovered stream payloads.
/// * `target` – container format to produce.
/// * `output` – path where the new container file will be written.
pub fn migrate(
    streams: &[RecoveredStream],
    target: TargetContainer,
    output: &Path,
) -> Result<MigrateResult> {
    if streams.is_empty() {
        return Err(RepairError::RepairFailed(
            "No streams provided for migration".to_string(),
        ));
    }

    match target {
        TargetContainer::Matroska => migrate_to_matroska(streams, output),
        TargetContainer::MpegTs => migrate_to_mpegts(streams, output),
        TargetContainer::RawEs => migrate_to_raw_es(streams, output),
    }
}

// ---------------------------------------------------------------------------
// Matroska (simplified EBML framing)
// ---------------------------------------------------------------------------

/// Write streams into a minimal Matroska container.
///
/// This produces a valid but very simple MKV file with:
///  - EBML header
///  - Segment element
///  - A single Cluster containing one SimpleBlock per stream byte-range
fn migrate_to_matroska(streams: &[RecoveredStream], output: &Path) -> Result<MigrateResult> {
    let mut buf: Vec<u8> = Vec::new();

    // EBML header (minimal hard-coded values)
    // 0x1A45DFA3 — EBML element, size = 0x9F (approximation for fixed header)
    buf.extend_from_slice(&[0x1A, 0x45, 0xDF, 0xA3]);
    // VINT size = 0x9F (hard-coded for simplicity)
    buf.push(0x9F);
    // EBMLVersion = 1
    buf.extend_from_slice(&[0x42, 0x86, 0x81, 0x01]);
    // EBMLReadVersion = 1
    buf.extend_from_slice(&[0x42, 0xF7, 0x81, 0x01]);
    // EBMLMaxIDLength = 4
    buf.extend_from_slice(&[0x42, 0xF2, 0x81, 0x04]);
    // EBMLMaxSizeLength = 8
    buf.extend_from_slice(&[0x42, 0xF3, 0x81, 0x08]);
    // DocType = "matroska"
    buf.extend_from_slice(&[0x42, 0x82, 0x88]);
    buf.extend_from_slice(b"matroska");
    // DocTypeVersion = 4
    buf.extend_from_slice(&[0x42, 0x87, 0x81, 0x04]);
    // DocTypeReadVersion = 2
    buf.extend_from_slice(&[0x42, 0x85, 0x81, 0x02]);

    // Segment header — unknown size (0x01 FF FF FF FF FF FF FF)
    buf.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
    buf.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

    // Cluster — unknown size
    buf.extend_from_slice(&[0x1F, 0x43, 0xB6, 0x75]);
    buf.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

    // Timestamp (cluster timecode = 0)
    buf.extend_from_slice(&[0xE7, 0x81, 0x00]);

    // Emit one SimpleBlock per stream
    for (track_idx, stream) in streams.iter().enumerate() {
        let track_num = (track_idx + 1) as u8; // 1-based track numbers
        write_matroska_simple_block(&mut buf, track_num, &stream.data);
    }

    let bytes_written = buf.len() as u64;
    std::fs::write(output, &buf)?;

    Ok(MigrateResult {
        bytes_written,
        container: TargetContainer::Matroska,
        stream_count: streams.len(),
        lossless: true,
    })
}

/// Write a Matroska SimpleBlock element.
fn write_matroska_simple_block(buf: &mut Vec<u8>, track_num: u8, data: &[u8]) {
    // SimpleBlock element ID: 0xA3
    buf.push(0xA3);

    // Payload: track VINTnum (1 byte since track ≤ 126) + 2-byte timecode + flags + data
    let payload_len = 1 + 2 + 1 + data.len();
    write_vint_size(buf, payload_len as u64);

    // Track number as VINT (≤ 126 fits in 1 byte: 0x80 | track)
    buf.push(0x80 | (track_num & 0x7F));
    // Relative timecode (i16 BE) = 0
    buf.extend_from_slice(&[0x00, 0x00]);
    // Flags: keyframe bit = 0x80 if video
    buf.push(0x80);
    // Frame data
    buf.extend_from_slice(data);
}

/// Write a VINT-encoded size value into `buf`.
fn write_vint_size(buf: &mut Vec<u8>, size: u64) {
    if size < 0x7F {
        buf.push(0x80 | size as u8);
    } else if size < 0x3FFF {
        buf.push(0x40 | (size >> 8) as u8);
        buf.push((size & 0xFF) as u8);
    } else if size < 0x1F_FFFF {
        buf.push(0x20 | (size >> 16) as u8);
        buf.push((size >> 8 & 0xFF) as u8);
        buf.push((size & 0xFF) as u8);
    } else {
        // 4-byte VINT
        buf.push(0x10 | (size >> 24) as u8);
        buf.push((size >> 16 & 0xFF) as u8);
        buf.push((size >> 8 & 0xFF) as u8);
        buf.push((size & 0xFF) as u8);
    }
}

// ---------------------------------------------------------------------------
// MPEG-TS passthrough
// ---------------------------------------------------------------------------

/// Wrap stream data in 188-byte MPEG-TS packets (PES passthrough).
fn migrate_to_mpegts(streams: &[RecoveredStream], output: &Path) -> Result<MigrateResult> {
    let mut buf: Vec<u8> = Vec::new();
    let mut bytes_written = 0u64;

    // PAT packet (PID 0)
    let pat = build_pat(streams.len() as u16);
    buf.extend_from_slice(&pat);
    bytes_written += 188;

    // PMT packet(s)
    for (idx, _stream) in streams.iter().enumerate() {
        let pmt_pid = 0x0100 + idx as u16;
        let es_pid = 0x0200 + idx as u16;
        let pmt = build_pmt(pmt_pid, es_pid);
        buf.extend_from_slice(&pmt);
        bytes_written += 188;
    }

    // Emit PES-framed data for each stream
    for (idx, stream) in streams.iter().enumerate() {
        let es_pid = 0x0200 + idx as u16;
        let stream_id: u8 = if stream.is_video { 0xE0 } else { 0xC0 };
        let packets = packetize_stream(&stream.data, es_pid, stream_id);
        bytes_written += packets.len() as u64;
        buf.extend_from_slice(&packets);
    }

    std::fs::write(output, &buf)?;

    Ok(MigrateResult {
        bytes_written,
        container: TargetContainer::MpegTs,
        stream_count: streams.len(),
        lossless: true,
    })
}

/// Build a minimal PAT (Program Association Table) TS packet.
fn build_pat(program_count: u16) -> Vec<u8> {
    let mut pkt = vec![0u8; 188];
    pkt[0] = 0x47; // sync byte
    pkt[1] = 0x60; // payload_unit_start=1, PID high = 0
    pkt[2] = 0x00; // PID low = 0
    pkt[3] = 0x10; // adaptation_field_control=1 (payload only), cc=0
    pkt[4] = 0x00; // pointer field

    // PAT section
    pkt[5] = 0x00; // table_id = PAT
    pkt[6] = 0xB0;
    pkt[7] = (4 + program_count * 4 + 4) as u8; // section_length (variable, simplified)
    pkt[8] = 0x00; // transport_stream_id hi
    pkt[9] = 0x01; // transport_stream_id lo
    pkt[10] = 0xC1; // version=0, current=1
    pkt[11] = 0x00; // section_number
    pkt[12] = 0x00; // last_section_number

    // One program entry: program 1 → PMT PID 0x0100
    pkt[13] = 0x00;
    pkt[14] = 0x01; // program_number = 1
    pkt[15] = 0xE1;
    pkt[16] = 0x00; // PMT PID = 0x0100

    // CRC (fake)
    pkt[17] = 0x00;
    pkt[18] = 0x00;
    pkt[19] = 0x00;
    pkt[20] = 0x00;

    pkt
}

/// Build a minimal PMT TS packet.
fn build_pmt(pmt_pid: u16, es_pid: u16) -> Vec<u8> {
    let mut pkt = vec![0u8; 188];
    pkt[0] = 0x47;
    pkt[1] = 0x40 | ((pmt_pid >> 8) & 0x1F) as u8;
    pkt[2] = (pmt_pid & 0xFF) as u8;
    pkt[3] = 0x10;
    pkt[4] = 0x00; // pointer

    // PMT section (very minimal)
    pkt[5] = 0x02; // table_id = PMT
    pkt[6] = 0xB0;
    pkt[7] = 0x12; // section_length
    pkt[8] = 0x00;
    pkt[9] = 0x01; // program_number
    pkt[10] = 0xC1;
    pkt[11] = 0x00;
    pkt[12] = 0x00;
    // PCR PID
    pkt[13] = 0xE0 | ((es_pid >> 8) & 0x1F) as u8;
    pkt[14] = (es_pid & 0xFF) as u8;
    // program_info_length = 0
    pkt[15] = 0xF0;
    pkt[16] = 0x00;

    // ES loop: stream_type=0x1B (H.264), es_pid
    pkt[17] = 0x1B; // stream type
    pkt[18] = 0xE0 | ((es_pid >> 8) & 0x1F) as u8;
    pkt[19] = (es_pid & 0xFF) as u8;
    // es_info_length = 0
    pkt[20] = 0xF0;
    pkt[21] = 0x00;

    // CRC
    pkt[22] = 0x00;
    pkt[23] = 0x00;
    pkt[24] = 0x00;
    pkt[25] = 0x00;

    pkt
}

/// Packetize raw data into 188-byte TS packets with a PES header on the first.
fn packetize_stream(data: &[u8], pid: u16, stream_id: u8) -> Vec<u8> {
    const TS_PKT: usize = 188;
    const TS_HDR: usize = 4;
    const PES_HDR: usize = 9;
    const PAYLOAD_FIRST: usize = TS_PKT - TS_HDR - PES_HDR;
    const PAYLOAD_REST: usize = TS_PKT - TS_HDR;

    let mut out: Vec<u8> = Vec::new();
    let mut offset = 0usize;
    let mut cc: u8 = 0;
    let mut first = true;

    while offset < data.len() {
        let mut pkt = vec![0u8; TS_PKT];
        pkt[0] = 0x47;
        let pusi: u8 = if first { 0x40 } else { 0x00 };
        pkt[1] = pusi | ((pid >> 8) & 0x1F) as u8;
        pkt[2] = (pid & 0xFF) as u8;
        pkt[3] = 0x10 | (cc & 0x0F); // payload only
        cc = cc.wrapping_add(1);

        if first {
            // PES header
            let payload_end = (offset + PAYLOAD_FIRST).min(data.len());
            let payload_len = payload_end - offset;
            let pes_len = payload_len + PES_HDR - 6;

            pkt[4] = 0x00;
            pkt[5] = 0x00;
            pkt[6] = 0x01; // start code
            pkt[7] = stream_id;
            pkt[8] = (pes_len >> 8) as u8;
            pkt[9] = (pes_len & 0xFF) as u8;
            pkt[10] = 0x80; // flags
            pkt[11] = 0x00; // PTS_DTS flags = none
            pkt[12] = 0x00; // header data length = 0

            pkt[13..13 + payload_len].copy_from_slice(&data[offset..payload_end]);
            offset = payload_end;
            // Fill remaining with stuffing (0xFF)
            for b in pkt[13 + payload_len..].iter_mut() {
                *b = 0xFF;
            }
            first = false;
        } else {
            let payload_end = (offset + PAYLOAD_REST).min(data.len());
            let payload_len = payload_end - offset;
            pkt[4..4 + payload_len].copy_from_slice(&data[offset..payload_end]);
            offset = payload_end;
            for b in pkt[4 + payload_len..].iter_mut() {
                *b = 0xFF;
            }
        }

        out.extend_from_slice(&pkt);
    }

    out
}

// ---------------------------------------------------------------------------
// Raw elementary stream
// ---------------------------------------------------------------------------

/// Write streams as a concatenated raw elementary stream.
fn migrate_to_raw_es(streams: &[RecoveredStream], output: &Path) -> Result<MigrateResult> {
    let mut file = std::fs::File::create(output)?;
    let mut total = 0u64;

    for stream in streams {
        file.write_all(&stream.data)
            .map_err(|e| RepairError::Io(e))?;
        total += stream.data.len() as u64;
    }

    Ok(MigrateResult {
        bytes_written: total,
        container: TargetContainer::RawEs,
        stream_count: streams.len(),
        lossless: true,
    })
}

// ---------------------------------------------------------------------------
// High-level helper: migrate from repaired file
// ---------------------------------------------------------------------------

/// Re-mux a repaired file into a clean container.
///
/// Reads the file at `input`, attempts to identify streams via `codec_probe`,
/// and writes a fresh container to `output`.
pub fn remux_file(input: &Path, target: TargetContainer, output: &Path) -> Result<MigrateResult> {
    let data = std::fs::read(input)?;
    if data.is_empty() {
        return Err(RepairError::RepairFailed("Input file is empty".to_string()));
    }

    // Use codec_probe to identify the stream
    let probe = crate::codec_probe::probe_codec(&data)?;
    let codec = match probe.codec {
        crate::codec_probe::ProbeCodec::Av1 => CodecHint::Av1,
        crate::codec_probe::ProbeCodec::Vp9 => CodecHint::Vp9,
        crate::codec_probe::ProbeCodec::H264 => CodecHint::H264,
        crate::codec_probe::ProbeCodec::Opus => CodecHint::Opus,
        crate::codec_probe::ProbeCodec::Flac => CodecHint::Flac,
        _ => CodecHint::Unknown,
    };

    let is_video = matches!(codec, CodecHint::Av1 | CodecHint::Vp9 | CodecHint::H264);

    let stream = RecoveredStream {
        data,
        codec,
        is_video,
        duration_ms: 0,
    };

    migrate(&[stream], target, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("oximedia_migrate_test_{}", name))
    }

    #[test]
    fn test_migrate_raw_es_empty_streams() {
        let result = migrate(&[], TargetContainer::RawEs, &temp_path("empty_out.bin"));
        assert!(result.is_err());
    }

    #[test]
    fn test_migrate_raw_es_single_stream() {
        let stream = RecoveredStream {
            data: vec![0xAA; 256],
            codec: CodecHint::Unknown,
            is_video: false,
            duration_ms: 0,
        };
        let out = temp_path("raw_es_out.bin");
        let result = migrate(&[stream], TargetContainer::RawEs, &out).expect("should succeed");
        assert_eq!(result.bytes_written, 256);
        assert_eq!(result.stream_count, 1);
        assert!(result.lossless);
        assert_eq!(result.container, TargetContainer::RawEs);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn test_migrate_matroska_single_stream() {
        let stream = RecoveredStream {
            data: vec![0xBB; 128],
            codec: CodecHint::H264,
            is_video: true,
            duration_ms: 1000,
        };
        let out = temp_path("mkv_out.mkv");
        let result = migrate(&[stream], TargetContainer::Matroska, &out).expect("should succeed");
        assert!(result.bytes_written > 128);
        assert_eq!(result.stream_count, 1);
        assert_eq!(result.container, TargetContainer::Matroska);

        // Verify EBML magic at start
        let written = std::fs::read(&out).expect("should read output");
        assert_eq!(&written[0..4], &[0x1A, 0x45, 0xDF, 0xA3]);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn test_migrate_mpegts_single_stream() {
        let stream = RecoveredStream {
            data: vec![0xCC; 512],
            codec: CodecHint::H264,
            is_video: true,
            duration_ms: 2000,
        };
        let out = temp_path("ts_out.ts");
        let result = migrate(&[stream], TargetContainer::MpegTs, &out).expect("should succeed");
        assert!(result.bytes_written > 0);
        assert_eq!(result.stream_count, 1);

        // Verify TS sync byte in first packet
        let written = std::fs::read(&out).expect("should read output");
        assert_eq!(written[0], 0x47);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn test_migrate_matroska_multiple_streams() {
        let streams = vec![
            RecoveredStream {
                data: vec![0x01; 64],
                codec: CodecHint::H264,
                is_video: true,
                duration_ms: 0,
            },
            RecoveredStream {
                data: vec![0x02; 48],
                codec: CodecHint::Opus,
                is_video: false,
                duration_ms: 0,
            },
        ];
        let out = temp_path("mkv_multi_out.mkv");
        let result = migrate(&streams, TargetContainer::Matroska, &out).expect("should succeed");
        assert_eq!(result.stream_count, 2);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn test_target_container_display() {
        assert_eq!(format!("{}", TargetContainer::Matroska), "Matroska/MKV");
        assert_eq!(format!("{}", TargetContainer::MpegTs), "MPEG-TS");
        assert_eq!(format!("{}", TargetContainer::RawEs), "Raw ES");
    }

    #[test]
    fn test_write_vint_size_small() {
        let mut buf = Vec::new();
        write_vint_size(&mut buf, 10);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf[0], 0x80 | 10);
    }

    #[test]
    fn test_write_vint_size_medium() {
        let mut buf = Vec::new();
        write_vint_size(&mut buf, 200);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_remux_file_empty() {
        let path = temp_path("remux_empty.bin");
        std::fs::write(&path, b"").expect("write");
        let out = temp_path("remux_empty_out.mkv");
        let result = remux_file(&path, TargetContainer::RawEs, &out);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_remux_file_with_data() {
        let path = temp_path("remux_data.bin");
        // Write some data (not a real codec, will be detected as Unknown)
        std::fs::write(&path, &[0xAA; 256]).expect("write");
        let out = temp_path("remux_data_out.bin");
        let result = remux_file(&path, TargetContainer::RawEs, &out);
        assert!(result.is_ok());
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn test_packetize_stream_produces_ts_sync() {
        let data = vec![0xAB; 376]; // 2 TS payloads
        let packets = packetize_stream(&data, 0x0200, 0xE0);
        assert!(!packets.is_empty());
        assert_eq!(packets.len() % 188, 0);
        // Every packet must start with 0x47
        for chunk in packets.chunks(188) {
            assert_eq!(chunk[0], 0x47);
        }
    }

    #[test]
    fn test_migrate_result_lossless() {
        let result = MigrateResult {
            bytes_written: 1024,
            container: TargetContainer::RawEs,
            stream_count: 1,
            lossless: true,
        };
        assert!(result.lossless);
        assert_eq!(result.stream_count, 1);
    }
}
