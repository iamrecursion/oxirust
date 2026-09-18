//! ISOBMFF box construction helpers for the MP4 muxer.
//!
//! Contains builders for `tkhd`, `mdhd`, `hdlr`, `vmhd`, `smhd`, `dinf`,
//! `stsd` (and its codec-specific sub-boxes), `stco`/`co64`, `stss`, and `trex`.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use oximedia_core::{CodecId, MediaType};

use super::super::av1c::build_av1c_from_extradata;
use super::types::{Mp4TrackState, MOVIE_TIMESCALE};
use crate::mux::cmaf::{write_box, write_full_box, write_u32_be, write_u64_be};

// ─── Constants ───────────────────────────────────────────────────────────────

/// Identity matrix for mvhd/tkhd (9 x u32 in fixed-point 16.16 / 2.30).
pub(super) const IDENTITY_MATRIX: [u8; 36] = [
    0x00, 0x01, 0x00, 0x00, // 1.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x01, 0x00, 0x00, // 1.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x00, 0x00, 0x00, 0x00, // 0.0
    0x40, 0x00, 0x00, 0x00, // 16384.0 (1.0 in 2.30)
];

// ─── Track/media header boxes ─────────────────────────────────────────────────

/// Builds a `tkhd` (track header) box.
pub(super) fn build_tkhd(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();
    // creation_time
    c.extend_from_slice(&write_u32_be(0));
    // modification_time
    c.extend_from_slice(&write_u32_be(0));
    // track_id
    c.extend_from_slice(&write_u32_be(track.track_id));
    // reserved
    c.extend_from_slice(&write_u32_be(0));
    // duration in movie timescale
    let dur_ms = if track.timescale == 0 {
        0
    } else {
        track.total_duration * u64::from(MOVIE_TIMESCALE) / u64::from(track.timescale)
    };
    c.extend_from_slice(&write_u32_be(dur_ms as u32));
    // reserved (8 bytes)
    c.extend_from_slice(&[0u8; 8]);
    // layer
    c.extend_from_slice(&[0u8; 2]);
    // alternate_group
    c.extend_from_slice(&[0u8; 2]);
    // volume (audio=0x0100, video=0)
    if track.stream_info.media_type == MediaType::Audio {
        c.extend_from_slice(&[0x01, 0x00]);
    } else {
        c.extend_from_slice(&[0x00, 0x00]);
    }
    // reserved
    c.extend_from_slice(&[0u8; 2]);
    // matrix
    c.extend_from_slice(&IDENTITY_MATRIX);
    // width (fixed-point 16.16)
    let w = track.stream_info.codec_params.width.unwrap_or(0);
    c.extend_from_slice(&write_u32_be(w << 16));
    // height (fixed-point 16.16)
    let h = track.stream_info.codec_params.height.unwrap_or(0);
    c.extend_from_slice(&write_u32_be(h << 16));

    // flags=3 (track_enabled | track_in_movie)
    write_full_box(b"tkhd", 0, 3, &c)
}

/// Builds a `mdhd` (media header) box.
pub(super) fn build_mdhd(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();
    // creation_time
    c.extend_from_slice(&write_u32_be(0));
    // modification_time
    c.extend_from_slice(&write_u32_be(0));
    // timescale
    c.extend_from_slice(&write_u32_be(track.timescale));
    // duration
    c.extend_from_slice(&write_u32_be(track.total_duration as u32));
    // language (undetermined = 0x55C4)
    c.extend_from_slice(&[0x55, 0xC4]);
    // pre_defined
    c.extend_from_slice(&[0u8; 2]);

    write_full_box(b"mdhd", 0, 0, &c)
}

/// Builds a `hdlr` (handler reference) box.
pub(super) fn build_hdlr(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();
    // pre_defined
    c.extend_from_slice(&write_u32_be(0));
    // handler_type
    c.extend_from_slice(&track.handler_type);
    // reserved (12 bytes)
    c.extend_from_slice(&[0u8; 12]);
    // name (null-terminated)
    c.extend_from_slice(track.handler_name.as_bytes());
    c.push(0);

    write_full_box(b"hdlr", 0, 0, &c)
}

// ─── Media information header boxes ──────────────────────────────────────────

/// Builds a `vmhd` (video media header) box.
pub(super) fn build_vmhd() -> Vec<u8> {
    let mut c = Vec::new();
    // graphicsmode
    c.extend_from_slice(&[0u8; 2]);
    // opcolor (3 x u16)
    c.extend_from_slice(&[0u8; 6]);
    write_full_box(b"vmhd", 0, 1, &c)
}

/// Builds an `smhd` (sound media header) box.
pub(super) fn build_smhd() -> Vec<u8> {
    let mut c = Vec::new();
    // balance
    c.extend_from_slice(&[0u8; 2]);
    // reserved
    c.extend_from_slice(&[0u8; 2]);
    write_full_box(b"smhd", 0, 0, &c)
}

// ─── Data information box ─────────────────────────────────────────────────────

/// Builds a `dinf` + `dref` box indicating a self-contained file.
pub(super) fn build_dinf() -> Vec<u8> {
    // dref with one url entry (self-contained)
    let url_box = write_full_box(b"url ", 0, 1, &[]); // flag=1 means self-contained
    let mut dref_content = Vec::new();
    dref_content.extend_from_slice(&write_u32_be(1)); // entry_count
    dref_content.extend(url_box);
    let dref = write_full_box(b"dref", 0, 0, &dref_content);

    write_box(b"dinf", &dref)
}

// ─── Sample description box ───────────────────────────────────────────────────

/// Builds a `stsd` (sample description) box for the given track.
pub(super) fn build_stsd(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(1)); // entry_count

    match track.stream_info.media_type {
        MediaType::Video => {
            c.extend(build_video_sample_entry(track));
        }
        MediaType::Audio => {
            c.extend(build_audio_sample_entry(track));
        }
        _ => {
            // Generic sample entry
            c.extend(build_generic_sample_entry(track));
        }
    }

    write_full_box(b"stsd", 0, 0, &c)
}

fn build_video_sample_entry(track: &Mp4TrackState) -> Vec<u8> {
    let fourcc = codec_to_fourcc(track.stream_info.codec);
    let mut c = Vec::new();

    // reserved (6 bytes)
    c.extend_from_slice(&[0u8; 6]);
    // data_reference_index
    c.extend_from_slice(&[0x00, 0x01]);
    // pre_defined + reserved (16 bytes)
    c.extend_from_slice(&[0u8; 16]);
    // width
    let w = track.stream_info.codec_params.width.unwrap_or(0) as u16;
    c.extend_from_slice(&w.to_be_bytes());
    // height
    let h = track.stream_info.codec_params.height.unwrap_or(0) as u16;
    c.extend_from_slice(&h.to_be_bytes());
    // horiz_resolution (72 dpi as fixed-point 16.16)
    c.extend_from_slice(&write_u32_be(0x0048_0000));
    // vert_resolution
    c.extend_from_slice(&write_u32_be(0x0048_0000));
    // reserved
    c.extend_from_slice(&write_u32_be(0));
    // frame_count
    c.extend_from_slice(&[0x00, 0x01]);
    // compressor_name (32 bytes, padded)
    let mut comp_name = [0u8; 32];
    let name = b"OxiMedia";
    let len = name.len().min(31);
    comp_name[0] = len as u8;
    comp_name[1..1 + len].copy_from_slice(&name[..len]);
    c.extend_from_slice(&comp_name);
    // depth
    c.extend_from_slice(&[0x00, 0x18]); // 24 bit
                                        // pre_defined
    c.extend_from_slice(&[0xFF, 0xFF]);

    // Codec-specific configuration box
    match track.stream_info.codec {
        CodecId::Av1 => {
            // For AV1, always emit an av1C box.  If the caller supplied a pre-built
            // AV1CodecConfigurationRecord in extradata we use that verbatim; otherwise
            // we attempt to derive one from the first OBU in extradata, falling back
            // to a safe default (Main profile, level 2.0, 4:2:0, 8-bit).
            let av1c_payload = track
                .stream_info
                .codec_params
                .extradata
                .as_deref()
                .and_then(|data| build_av1c_from_extradata(data))
                .unwrap_or_else(|| {
                    // Safe default: marker=1, version=1, Main profile, level 2.0,
                    // tier 0, 8-bit, 4:2:0, no initial presentation delay.
                    vec![0x81u8, 0x00, 0x04, 0x00]
                });
            c.extend(write_box(b"av1C", &av1c_payload));
        }
        _ => {
            if let Some(extradata) = &track.stream_info.codec_params.extradata {
                let config_fourcc = codec_config_fourcc(track.stream_info.codec);
                c.extend(write_box(&config_fourcc, extradata));
            }
        }
    }

    write_box(&fourcc, &c)
}

fn build_audio_sample_entry(track: &Mp4TrackState) -> Vec<u8> {
    let fourcc = codec_to_fourcc(track.stream_info.codec);
    let mut c = Vec::new();

    // reserved (6 bytes)
    c.extend_from_slice(&[0u8; 6]);
    // data_reference_index
    c.extend_from_slice(&[0x00, 0x01]);
    // reserved (8 bytes)
    c.extend_from_slice(&[0u8; 8]);
    // channel_count
    let channels = track.stream_info.codec_params.channels.unwrap_or(2) as u16;
    c.extend_from_slice(&channels.to_be_bytes());
    // sample_size (16 bits)
    c.extend_from_slice(&[0x00, 0x10]);
    // pre_defined + reserved
    c.extend_from_slice(&[0u8; 4]);
    // sample_rate (fixed-point 16.16)
    let sr = track.stream_info.codec_params.sample_rate.unwrap_or(48000);
    c.extend_from_slice(&write_u32_be(sr << 16));

    // Codec-specific configuration box
    if let Some(extradata) = &track.stream_info.codec_params.extradata {
        let config_fourcc = codec_config_fourcc(track.stream_info.codec);
        c.extend(write_box(&config_fourcc, extradata));
    }

    write_box(&fourcc, &c)
}

fn build_generic_sample_entry(track: &Mp4TrackState) -> Vec<u8> {
    let fourcc = codec_to_fourcc(track.stream_info.codec);
    let mut c = Vec::new();
    c.extend_from_slice(&[0u8; 6]); // reserved
    c.extend_from_slice(&[0x00, 0x01]); // data_reference_index
    write_box(&fourcc, &c)
}

// ─── Chunk offset box ─────────────────────────────────────────────────────────

/// Builds a `stco` (32-bit chunk offset) or `co64` (64-bit) box.
///
/// Automatically selects `co64` when any offset exceeds `u32::MAX`.
pub(super) fn build_stco(chunk_offsets: &[u64]) -> Vec<u8> {
    // Use stco (32-bit) if all offsets fit, otherwise co64
    let use_64bit = chunk_offsets.iter().any(|&o| o > u64::from(u32::MAX));

    if use_64bit {
        let mut c = Vec::new();
        c.extend_from_slice(&write_u32_be(chunk_offsets.len() as u32));
        for &offset in chunk_offsets {
            c.extend_from_slice(&write_u64_be(offset));
        }
        write_full_box(b"co64", 0, 0, &c)
    } else {
        let mut c = Vec::new();
        c.extend_from_slice(&write_u32_be(chunk_offsets.len() as u32));
        for &offset in chunk_offsets {
            c.extend_from_slice(&write_u32_be(offset as u32));
        }
        write_full_box(b"stco", 0, 0, &c)
    }
}

// ─── Sync sample table ────────────────────────────────────────────────────────

/// Builds an `stss` (sync sample) box listing all keyframe sample indices.
pub(super) fn build_stss(track: &Mp4TrackState) -> Vec<u8> {
    let sync_samples: Vec<u32> = track
        .samples
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_sync)
        .map(|(i, _)| (i + 1) as u32) // 1-based
        .collect();

    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(sync_samples.len() as u32));
    for &idx in &sync_samples {
        c.extend_from_slice(&write_u32_be(idx));
    }

    write_full_box(b"stss", 0, 0, &c)
}

// ─── Track extension box ─────────────────────────────────────────────────────

/// Builds a `trex` (track extends) box for the `mvex` container.
pub(super) fn build_trex(track_id: u32) -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(track_id));
    c.extend_from_slice(&write_u32_be(1)); // default_sample_description_index
    c.extend_from_slice(&write_u32_be(0)); // default_sample_duration
    c.extend_from_slice(&write_u32_be(0)); // default_sample_size
    c.extend_from_slice(&write_u32_be(0)); // default_sample_flags
    write_full_box(b"trex", 0, 0, &c)
}

// ─── Codec FourCC helpers ────────────────────────────────────────────────────

pub(super) fn codec_to_fourcc(codec: CodecId) -> [u8; 4] {
    match codec {
        CodecId::Av1 => *b"av01",
        CodecId::Vp9 => *b"vp09",
        CodecId::Vp8 => *b"vp08",
        CodecId::Opus => *b"Opus",
        CodecId::Flac => *b"fLaC",
        CodecId::Vorbis => *b"vorb",
        CodecId::Apv => *b"apv1",   // ISO/IEC 23009-13 registered fourcc
        CodecId::Mjpeg => *b"jpeg", // ISOM-registered fourcc for Motion JPEG
        _ => *b"unkn",
    }
}

pub(super) fn codec_config_fourcc(codec: CodecId) -> [u8; 4] {
    match codec {
        CodecId::Av1 => *b"av1C",
        CodecId::Vp9 => *b"vpcC",
        CodecId::Vp8 => *b"vpcC",
        CodecId::Opus => *b"dOps",
        CodecId::Flac => *b"dfLa",
        _ => *b"conf",
    }
}
