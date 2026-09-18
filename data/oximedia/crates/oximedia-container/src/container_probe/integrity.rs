//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::IntegrityCheckResult;
use crate::container_probe_parsers::read_u32_be;

/// Checks the structural integrity of a container's byte stream.
#[must_use]
pub fn check_container_integrity(data: &[u8]) -> IntegrityCheckResult {
    let mut r = IntegrityCheckResult::ok();
    if data.is_empty() {
        r.add_issue("Container data is empty", 1.0);
        return r;
    }
    if data.len() < 8 {
        r.add_issue("Too short for any known format", 0.8);
        return r;
    }
    if &data[4..8] == b"ftyp" {
        validate_mp4_boxes(data, &mut r);
    } else if &data[..4] == b"fLaC" {
        validate_flac_structure(data, &mut r);
    } else if &data[..4] == b"RIFF" {
        validate_riff_structure(data, &mut r);
    }
    r
}
fn validate_mp4_boxes(data: &[u8], result: &mut IntegrityCheckResult) {
    let (mut offset, mut box_count, mut found_moov) = (0usize, 0u32, false);
    while offset + 8 <= data.len() {
        let size = read_u32_be(data, offset) as usize;
        if size < 8 {
            result.add_issue(format!("Bad MP4 box size at {offset}"), 0.3);
            break;
        }
        if offset + size > data.len() {
            result.add_issue(format!("MP4 box exceeds data at {offset}"), 0.2);
            break;
        }
        if &data[offset + 4..offset + 8] == b"moov" {
            found_moov = true;
        }
        box_count += 1;
        offset += size;
    }
    if box_count == 0 {
        result.add_issue("No valid MP4 boxes", 0.5);
    }
    if !found_moov && data.len() > 1024 {
        result.add_issue("MP4 missing moov", 0.3);
    }
}
fn validate_flac_structure(data: &[u8], result: &mut IntegrityCheckResult) {
    if data.len() < 42 {
        result.add_issue("FLAC too short for STREAMINFO", 0.4);
        return;
    }
    if data[4] & 0x7F != 0 {
        result.add_issue("First FLAC block not STREAMINFO", 0.3);
    }
}
fn validate_riff_structure(data: &[u8], result: &mut IntegrityCheckResult) {
    if data.len() < 12 {
        result.add_issue("RIFF too short", 0.5);
        return;
    }
    if &data[8..12] != b"WAVE" && &data[8..12] != b"AVI " {
        result.add_issue("RIFF form type not WAVE/AVI", 0.2);
    }
    let riff_size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as u64;
    if riff_size + 8 > data.len() as u64 {
        result.add_issue(
            format!("RIFF size mismatch ({} vs {})", riff_size + 8, data.len()),
            0.15,
        );
    }
}
