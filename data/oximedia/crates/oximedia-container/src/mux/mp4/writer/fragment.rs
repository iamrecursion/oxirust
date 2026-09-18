//! Fragmented MP4 / CMAF helpers for the MP4 muxer.
//!
//! Contains:
//! - Fragment boundary computation (`compute_fragment_boundaries`)
//! - DTS accumulation helpers (`dts_at`, `select_time_range`)
//! - The public `build_sidx` function (DASH `sidx` box emitter)

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use super::types::Mp4TrackState;
use crate::mux::cmaf::{write_box, write_full_box, write_u32_be, write_u64_be};

// ─── Fragment boundary computation ────────────────────────────────────────────

/// Computes `(start_idx, end_idx)` fragment boundaries from a track's sample list.
///
/// A new fragment starts at each keyframe whose accumulated DTS since the previous
/// fragment start exceeds `frag_duration_ticks`.  The last fragment end is always
/// `track.samples.len()`.
pub(super) fn compute_fragment_boundaries(
    track: &Mp4TrackState,
    frag_duration_ticks: u64,
) -> Vec<(usize, usize)> {
    if track.samples.is_empty() {
        return Vec::new();
    }

    let mut boundaries: Vec<(usize, usize)> = Vec::new();
    let mut frag_start = 0usize;
    let mut accumulated: u64 = 0;

    for (i, sample) in track.samples.iter().enumerate() {
        if i > frag_start && sample.is_sync {
            // Decide to cut here if we've accumulated enough or if no duration limit.
            if frag_duration_ticks == 0 || accumulated >= frag_duration_ticks {
                boundaries.push((frag_start, i));
                frag_start = i;
                accumulated = 0;
            }
        }
        accumulated += u64::from(sample.duration);
    }

    // Last fragment
    boundaries.push((frag_start, track.samples.len()));
    boundaries
}

// ─── DTS helpers ──────────────────────────────────────────────────────────────

/// Returns the cumulative DTS (in track timescale ticks) at sample index `idx`.
///
/// Returns 0 when `idx == 0` or the track has no samples.
pub(super) fn dts_at(track: &Mp4TrackState, idx: usize) -> u64 {
    track
        .samples
        .iter()
        .take(idx)
        .map(|s| u64::from(s.duration))
        .sum()
}

/// Selects the slice `[start, end)` of samples from `track` whose accumulated DTS
/// falls within `[base_dts, end_dts)`.
///
/// Returns `(0, 0)` when the track has no samples in range.
pub(super) fn select_time_range(
    track: &Mp4TrackState,
    base_dts: u64,
    end_dts: u64,
) -> (usize, usize) {
    let mut cursor: u64 = 0;
    let mut start: Option<usize> = None;
    let mut end = 0usize;

    for (i, s) in track.samples.iter().enumerate() {
        let sample_dts = cursor;
        cursor += u64::from(s.duration);

        if sample_dts >= end_dts {
            break;
        }
        if sample_dts >= base_dts {
            if start.is_none() {
                start = Some(i);
            }
            end = i + 1;
        }
    }

    match start {
        Some(s) => (s, end),
        None => (0, 0),
    }
}

// ─── sidx box ────────────────────────────────────────────────────────────────

/// Builds a minimal `sidx` (Segment Index Box, ISO 14496-12 §8.16.3).
///
/// Emits exactly one reference entry covering the subsequent `moof`+`mdat`.
/// Build a `sidx` (Segment Index) box for a fragmented MP4 / DASH subsegment.
///
/// The returned `Vec<u8>` is a fully-formed `sidx` full-box (version=1)
/// containing a single reference entry. Useful for downstream packagers
/// emitting DASH-IF live segments where each `moof`+`mdat` is preceded by a
/// `sidx` describing its size, duration, and SAP presence.
///
/// # Parameters
///
/// - `reference_id`: Track ID referenced by this index (must match `tfhd.track_ID`).
/// - `timescale`: Ticks per second; matches the media `mdhd.timescale`.
/// - `earliest_pts`: Earliest presentation time of the subsegment in `timescale` units.
/// - `referenced_size`: Total size in bytes of the subsegment data being indexed.
/// - `subsegment_duration`: Duration of the subsegment in `timescale` units.
/// - `is_sap`: `true` if the subsegment starts with a Stream Access Point.
pub fn build_sidx(
    reference_id: u32,
    timescale: u32,
    earliest_pts: u64,
    referenced_size: u32,
    subsegment_duration: u32,
    is_sap: bool,
) -> Vec<u8> {
    // version=1 → 64-bit earliest_presentation_time + first_offset
    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(reference_id));
    c.extend_from_slice(&write_u32_be(timescale));
    c.extend_from_slice(&write_u64_be(earliest_pts));
    c.extend_from_slice(&write_u64_be(0)); // first_offset (0 = immediately follows)
    c.extend_from_slice(&[0u8; 2]); // reserved
    c.extend_from_slice(&1u16.to_be_bytes()); // reference_count = 1

    // reference entry:
    // bit(1) reference_type = 0 (movie fragment)
    // unsigned int(31) referenced_size
    let ref_type_size: u32 = referenced_size & 0x7FFF_FFFF; // reference_type=0
    c.extend_from_slice(&write_u32_be(ref_type_size));
    c.extend_from_slice(&write_u32_be(subsegment_duration));
    // SAP info: bit(1) starts_with_SAP + bit(3) SAP_type + bit(28) SAP_delta_time
    let sap_word: u32 = if is_sap { 0x9000_0000u32 } else { 0 };
    c.extend_from_slice(&write_u32_be(sap_word));

    write_full_box(b"sidx", 1, 0, &c)
}

// ─── write_box re-export for moof building in mod.rs ─────────────────────────

/// Thin wrapper: forwards to `crate::mux::cmaf::write_box`.
pub(super) fn wrap_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    write_box(fourcc, payload)
}
