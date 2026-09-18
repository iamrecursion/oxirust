//! Sample table box builders for the MP4 muxer.
//!
//! Contains builders for the core ISOBMFF sample-table boxes:
//! - `stts` — time-to-sample (run-length encoded durations)
//! - `ctts` — composition time offset (PTS – DTS)
//! - `stsc` — sample-to-chunk mapping
//! - `stsz` — sample size table

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use super::types::Mp4TrackState;
use crate::mux::cmaf::{write_full_box, write_u32_be};

// ─── stts — time to sample ────────────────────────────────────────────────────

/// Builds an `stts` (time-to-sample) box using run-length encoding.
pub(super) fn build_stts(track: &Mp4TrackState) -> Vec<u8> {
    // Run-length encode sample durations
    let mut entries: Vec<(u32, u32)> = Vec::new();

    for sample in &track.samples {
        if let Some(last) = entries.last_mut() {
            if last.1 == sample.duration {
                last.0 += 1;
                continue;
            }
        }
        entries.push((1, sample.duration));
    }

    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(entries.len() as u32));
    for (count, delta) in &entries {
        c.extend_from_slice(&write_u32_be(*count));
        c.extend_from_slice(&write_u32_be(*delta));
    }

    write_full_box(b"stts", 0, 0, &c)
}

// ─── ctts — composition time offset ──────────────────────────────────────────

/// Builds a `ctts` (composition time offset) box using run-length encoding.
///
/// Uses version=1 for signed offsets (B-frame friendly).
pub(super) fn build_ctts(track: &Mp4TrackState) -> Vec<u8> {
    // Run-length encode composition offsets
    let mut entries: Vec<(u32, i32)> = Vec::new();

    for sample in &track.samples {
        if let Some(last) = entries.last_mut() {
            if last.1 == sample.composition_offset {
                last.0 += 1;
                continue;
            }
        }
        entries.push((1, sample.composition_offset));
    }

    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(entries.len() as u32));
    for (count, offset) in &entries {
        c.extend_from_slice(&write_u32_be(*count));
        c.extend_from_slice(&((*offset) as u32).to_be_bytes());
    }

    // version=1 for signed offsets
    write_full_box(b"ctts", 1, 0, &c)
}

// ─── stsc — sample to chunk ───────────────────────────────────────────────────

/// Builds an `stsc` (sample-to-chunk) box.
pub(super) fn build_stsc(track: &Mp4TrackState) -> Vec<u8> {
    // Build sample-to-chunk table
    let mut entries: Vec<(u32, u32, u32)> = Vec::new(); // (first_chunk, samples_per_chunk, sdi)

    for (chunk_idx, &chunk_start) in track.chunk_boundaries.iter().enumerate() {
        let chunk_end = track
            .chunk_boundaries
            .get(chunk_idx + 1)
            .copied()
            .unwrap_or(track.samples.len() as u32);
        let samples_in_chunk = chunk_end.saturating_sub(chunk_start);

        if samples_in_chunk == 0 {
            continue;
        }

        let first_chunk = (chunk_idx + 1) as u32; // 1-based
        if let Some(last) = entries.last() {
            if last.1 == samples_in_chunk {
                continue; // Same pattern as previous
            }
        }
        entries.push((first_chunk, samples_in_chunk, 1));
    }

    if entries.is_empty() {
        entries.push((1, 1, 1)); // Fallback
    }

    let mut c = Vec::new();
    c.extend_from_slice(&write_u32_be(entries.len() as u32));
    for (first_chunk, spc, sdi) in &entries {
        c.extend_from_slice(&write_u32_be(*first_chunk));
        c.extend_from_slice(&write_u32_be(*spc));
        c.extend_from_slice(&write_u32_be(*sdi));
    }

    write_full_box(b"stsc", 0, 0, &c)
}

// ─── stsz — sample sizes ─────────────────────────────────────────────────────

/// Builds an `stsz` (sample size) box.
///
/// Uses the compact uniform-size form when all samples are the same size.
pub(super) fn build_stsz(track: &Mp4TrackState) -> Vec<u8> {
    let mut c = Vec::new();

    // Check if all samples have the same size
    let first_size = track.samples.first().map_or(0, |s| s.size);
    let all_same = track.samples.iter().all(|s| s.size == first_size);

    if all_same && !track.samples.is_empty() {
        // sample_size (common for all)
        c.extend_from_slice(&write_u32_be(first_size));
        // sample_count
        c.extend_from_slice(&write_u32_be(track.samples.len() as u32));
    } else {
        // sample_size = 0 (variable)
        c.extend_from_slice(&write_u32_be(0));
        // sample_count
        c.extend_from_slice(&write_u32_be(track.samples.len() as u32));
        // entry_size[]
        for sample in &track.samples {
            c.extend_from_slice(&write_u32_be(sample.size));
        }
    }

    write_full_box(b"stsz", 0, 0, &c)
}
