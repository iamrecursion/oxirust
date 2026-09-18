//! Codec-aware quality scoring for duplicate-group keeper selection.
//!
//! [`quality_score`] assigns a deterministic, comparable `f64` to a media file so
//! that [`crate::dedup_policy::GroupAction::KeepHighestQuality`] keeps the
//! genuinely highest-quality copy in a duplicate group instead of merely the
//! largest file on disk.
//!
//! # What "quality" means here
//!
//! For visual media the dominant quality signal is **spatial resolution** (pixel
//! count): a 3840x2160 master carries far more detail than a 720x480 copy
//! regardless of how aggressively either was compressed, so a *tiny* 4K file must
//! still outrank a *huge* SD file. Within a single resolution tier the
//! **effective bitrate** decides — a larger file at the *same* resolution spent
//! more bits per frame, suffered fewer compression artifacts and is therefore
//! higher quality. Finally **bit depth** (8 vs 10 vs 12 bits per component) breaks
//! any remaining ties.
//!
//! # The formula
//!
//! The three signals are combined additively in the `log2` domain, which is the
//! natural space for quantities that scale multiplicatively (pixels, bytes) and
//! keeps the terms on a comparable footing:
//!
//! ```text
//! score = RESOLUTION_WEIGHT * log2(1 + pixels)
//!       + BITRATE_WEIGHT    * log2(1 + effective_bitrate)
//!       + BIT_DEPTH_WEIGHT  * bit_depth
//!
//! effective_bitrate = size_bytes / duration_secs   (when a positive duration
//!                                                    was read from the container)
//!                   = size_bytes                    (otherwise — still images, or
//!                                                    containers without a probed
//!                                                    duration)
//! ```
//!
//! `RESOLUTION_WEIGHT` (1000) is two orders of magnitude larger than
//! `BITRATE_WEIGHT` (10): a single `log2` step of resolution (a doubling of pixel
//! count) contributes 1000 points, which no plausible bitrate difference can
//! overcome (matching it would require a `2^100` bitrate ratio). Resolution
//! therefore strictly dominates whenever the pixel counts differ by a meaningful
//! tier, while files that share a resolution tier are ordered by their effective
//! bitrate, and `bit_depth` (weight 1) only ever acts as a final tie-breaker.
//!
//! # Honesty / limitations
//!
//! Every signal is read from the **real file header** — nothing is fabricated or
//! defaulted to a plausible-looking constant. The prober understands the common
//! still-image formats (PNG, JPEG, GIF, BMP, WebP) and the ISO base-media
//! container family (MP4 / MOV / M4V), which between them cover the overwhelming
//! majority of deduplicated media. Resolution and bit depth come straight from
//! the image headers; for ISO-BMFF the track display size (`tkhd`) and movie
//! duration (`mvhd`) are walked out of the box tree by seeking — `mdat` payloads
//! are never read, so the probe stays cheap even for multi-gigabyte files.
//!
//! For any format the prober does **not** recognise — or any header it cannot
//! parse — `pixels`/`bit_depth`/`duration_secs` stay `None` and the score degrades
//! to the file-size term alone, which is exactly the previous "largest file" rule.
//! The result is thus never worse than the old behaviour and never reports a
//! resolution, depth or duration it did not actually measure. A consequence worth
//! stating plainly: in a group that mixes a probeable file with an unprobeable one,
//! the probeable file gains a resolution bonus the other cannot, so scoring is
//! biased toward the copy whose quality can be positively verified.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Weight applied to the resolution term (primary signal). Large enough that one
/// `log2` step of pixel count dominates any plausible bitrate/bit-depth delta.
const RESOLUTION_WEIGHT: f64 = 1_000.0;
/// Weight applied to the effective-bitrate term (secondary signal).
const BITRATE_WEIGHT: f64 = 10.0;
/// Weight applied to the bit-depth term (tertiary tie-breaker).
const BIT_DEPTH_WEIGHT: f64 = 1.0;

/// Number of leading bytes sniffed to identify and parse still-image headers.
const HEADER_SNIFF_LEN: usize = 64;
/// Maximum ISO-BMFF box-tree depth walked (guards against pathological nesting).
const MAX_BOX_DEPTH: u32 = 8;
/// Maximum sibling boxes inspected at one level (guards against malformed sizes).
const MAX_BOXES_PER_LEVEL: u32 = 4_096;
/// Maximum JPEG marker segments scanned before giving up on finding an SOF.
const MAX_JPEG_SEGMENTS: u32 = 1_024;

/// The 8-byte PNG signature.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// Compute a deterministic, codec-aware quality score for `path`.
///
/// Higher is better. The score combines header-derived resolution, bit depth and
/// an effective-bitrate proxy as documented at the [module level](self). Files
/// whose format cannot be probed fall back to a pure file-size ranking, so this
/// is always at least as good as the previous "largest file wins" heuristic.
///
/// I/O or metadata errors never panic: an unreadable file scores `0.0`.
#[must_use]
pub fn quality_score(path: &Path) -> f64 {
    score_signals(&probe_quality_signals(path))
}

/// Real, header-derived signals used to score a media file's quality.
#[derive(Debug, Clone, Copy, PartialEq)]
struct QualitySignals {
    /// Spatial resolution in pixels (`width * height`) when read from the header.
    pixels: Option<u64>,
    /// Bits per component/sample when the header exposes it (still images).
    bit_depth: Option<u32>,
    /// Playback duration in seconds when the container header exposes it.
    duration_secs: Option<f64>,
    /// On-disk size in bytes (always available; `0` if metadata is unreadable).
    size_bytes: u64,
}

impl QualitySignals {
    /// Start from just the file size, with every probed signal still unknown.
    const fn with_size(size_bytes: u64) -> Self {
        Self {
            pixels: None,
            bit_depth: None,
            duration_secs: None,
            size_bytes,
        }
    }
}

/// Turn probed [`QualitySignals`] into a single comparable score.
fn score_signals(signals: &QualitySignals) -> f64 {
    // Resolution term (primary). Unknown resolution contributes nothing, leaving
    // the file ranked by its size term alone — the legacy "largest file" rule.
    let resolution_term = match signals.pixels {
        Some(pixels) if pixels > 0 => RESOLUTION_WEIGHT * (1.0 + pixels as f64).log2(),
        _ => 0.0,
    };

    // Effective-bitrate term (secondary): bytes per second when a positive
    // duration was probed, otherwise the raw byte size (the honest fallback).
    let bitrate_proxy = match signals.duration_secs {
        Some(duration) if duration > 0.0 => signals.size_bytes as f64 / duration,
        _ => signals.size_bytes as f64,
    };
    let bitrate_term = BITRATE_WEIGHT * (1.0 + bitrate_proxy.max(0.0)).log2();

    // Bit-depth term (tertiary tie-breaker).
    let depth_term = match signals.bit_depth {
        Some(bits) => BIT_DEPTH_WEIGHT * bits as f64,
        None => 0.0,
    };

    resolution_term + bitrate_term + depth_term
}

/// Probe `path` for the quality signals we can read cheaply from its header.
fn probe_quality_signals(path: &Path) -> QualitySignals {
    let size_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut signals = QualitySignals::with_size(size_bytes);

    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(_) => return signals,
    };

    let mut head = [0u8; HEADER_SNIFF_LEN];
    let filled = fill_buffer(&mut file, &mut head);
    let head = &head[..filled];

    // Recognised still image: resolution (and possibly bit depth) are exact.
    if let Some(dims) = probe_image(&mut file, head) {
        signals.pixels = checked_pixels(dims.width, dims.height);
        signals.bit_depth = dims.bit_depth;
        return signals;
    }

    // Otherwise try the ISO base-media container family for resolution+duration.
    if let Some(iso) = probe_isobmff(&mut file) {
        if iso.pixels > 0 {
            signals.pixels = Some(iso.pixels);
        }
        signals.duration_secs = iso.duration_secs;
    }

    signals
}

/// Multiply width by height, returning `None` for a degenerate (zero) dimension.
fn checked_pixels(width: u32, height: u32) -> Option<u64> {
    if width == 0 || height == 0 {
        None
    } else {
        Some(u64::from(width) * u64::from(height))
    }
}

/// Read up to `buf.len()` bytes from the start of `file`, tolerating short reads.
///
/// Returns the number of bytes actually read.
fn fill_buffer(file: &mut File, buf: &mut [u8]) -> usize {
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(ref err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    filled
}

/// Header-derived spatial dimensions (and bit depth where the format exposes it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ImageDims {
    width: u32,
    height: u32,
    bit_depth: Option<u32>,
}

/// Dispatch on magic bytes to a still-image dimension parser.
fn probe_image(file: &mut File, head: &[u8]) -> Option<ImageDims> {
    if let Some(dims) = probe_png(head) {
        return Some(dims);
    }
    if let Some(dims) = probe_gif(head) {
        return Some(dims);
    }
    if let Some(dims) = probe_bmp(head) {
        return Some(dims);
    }
    if let Some(dims) = probe_webp(head) {
        return Some(dims);
    }
    if head.len() >= 2 && head[0] == 0xFF && head[1] == 0xD8 {
        return probe_jpeg(file);
    }
    None
}

/// Parse a PNG `IHDR` for width, height and per-component bit depth.
fn probe_png(head: &[u8]) -> Option<ImageDims> {
    if head.len() < 26 || !head.starts_with(&PNG_SIGNATURE) {
        return None;
    }
    // After the 8-byte signature: 4-byte chunk length, then the 4-byte chunk type.
    if &head[12..16] != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes([head[16], head[17], head[18], head[19]]);
    let height = u32::from_be_bytes([head[20], head[21], head[22], head[23]]);
    let bit_depth = u32::from(head[24]);
    if width == 0 || height == 0 {
        return None;
    }
    Some(ImageDims {
        width,
        height,
        bit_depth: Some(bit_depth),
    })
}

/// Parse a GIF logical-screen descriptor for the canvas dimensions.
fn probe_gif(head: &[u8]) -> Option<ImageDims> {
    if head.len() < 10 || (&head[..6] != b"GIF87a" && &head[..6] != b"GIF89a") {
        return None;
    }
    let width = u32::from(u16::from_le_bytes([head[6], head[7]]));
    let height = u32::from(u16::from_le_bytes([head[8], head[9]]));
    if width == 0 || height == 0 {
        return None;
    }
    // GIF is palette-based; a clean per-component bit depth does not apply.
    Some(ImageDims {
        width,
        height,
        bit_depth: None,
    })
}

/// Parse a Windows BMP `BITMAPINFOHEADER` for width and height.
fn probe_bmp(head: &[u8]) -> Option<ImageDims> {
    if head.len() < 26 || &head[..2] != b"BM" {
        return None;
    }
    let width = i32::from_le_bytes([head[18], head[19], head[20], head[21]]);
    // Height may be negative for top-down bitmaps; magnitude is the pixel count.
    let height = i32::from_le_bytes([head[22], head[23], head[24], head[25]]);
    let width = width.unsigned_abs();
    let height = height.unsigned_abs();
    if width == 0 || height == 0 {
        return None;
    }
    Some(ImageDims {
        width,
        height,
        bit_depth: None,
    })
}

/// Parse the dimensions out of a WebP `VP8 ` / `VP8L` / `VP8X` chunk header.
fn probe_webp(head: &[u8]) -> Option<ImageDims> {
    if head.len() < 16 || &head[..4] != b"RIFF" || &head[8..12] != b"WEBP" {
        return None;
    }
    let fourcc = &head[12..16];

    if fourcc == b"VP8 " {
        // Lossy: 3-byte frame tag, then start code 0x9D 0x01 0x2A, then dimensions.
        if head.len() < 30 || head[23] != 0x9D || head[24] != 0x01 || head[25] != 0x2A {
            return None;
        }
        let width = u32::from(u16::from_le_bytes([head[26], head[27]]) & 0x3FFF);
        let height = u32::from(u16::from_le_bytes([head[28], head[29]]) & 0x3FFF);
        return non_degenerate(width, height);
    }

    if fourcc == b"VP8L" {
        // Lossless: 0x2F signature, then 14-bit (width-1) and 14-bit (height-1).
        if head.len() < 25 || head[20] != 0x2F {
            return None;
        }
        let bits = u32::from_le_bytes([head[21], head[22], head[23], head[24]]);
        let width = (bits & 0x3FFF) + 1;
        let height = ((bits >> 14) & 0x3FFF) + 1;
        return non_degenerate(width, height);
    }

    if fourcc == b"VP8X" {
        // Extended: 1-byte flags, 3 reserved, then 24-bit (width-1)/(height-1).
        if head.len() < 30 {
            return None;
        }
        let width = read_u24_le(&head[24..27]) + 1;
        let height = read_u24_le(&head[27..30]) + 1;
        return non_degenerate(width, height);
    }

    None
}

/// Build [`ImageDims`] (no bit depth) rejecting any zero dimension.
fn non_degenerate(width: u32, height: u32) -> Option<ImageDims> {
    if width == 0 || height == 0 {
        None
    } else {
        Some(ImageDims {
            width,
            height,
            bit_depth: None,
        })
    }
}

/// Read a little-endian 24-bit unsigned integer from a 3-byte slice.
fn read_u24_le(bytes: &[u8]) -> u32 {
    u32::from(bytes[0]) | (u32::from(bytes[1]) << 8) | (u32::from(bytes[2]) << 16)
}

/// Scan JPEG marker segments for the Start-Of-Frame header (dimensions + precision).
///
/// Only marker headers and segment lengths are read; segment payloads are skipped
/// by seeking, so even JPEGs with large embedded thumbnails stay cheap to probe.
fn probe_jpeg(file: &mut File) -> Option<ImageDims> {
    // Position just past the SOI marker (0xFFD8) that `probe_image` already saw.
    file.seek(SeekFrom::Start(2)).ok()?;

    for _ in 0..MAX_JPEG_SEGMENTS {
        let mut byte = [0u8; 1];
        // Every marker starts with 0xFF; bail if the stream is not marker-aligned.
        if file.read_exact(&mut byte).is_err() || byte[0] != 0xFF {
            return None;
        }
        // Skip any 0xFF fill bytes preceding the marker code.
        let mut marker = 0xFFu8;
        while marker == 0xFF {
            if file.read_exact(&mut byte).is_err() {
                return None;
            }
            marker = byte[0];
        }

        // Standalone markers (TEM, RSTn, padding) carry no length field.
        if marker == 0x00 || marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            continue;
        }
        // Start-of-scan or end-of-image: the frame header would already be behind us.
        if marker == 0xD9 || marker == 0xDA {
            return None;
        }

        let mut len_buf = [0u8; 2];
        if file.read_exact(&mut len_buf).is_err() {
            return None;
        }
        let segment_len = u16::from_be_bytes(len_buf);
        if segment_len < 2 {
            return None;
        }

        // SOF0..SOF15 are frame headers, excluding DHT(C4)/JPG(C8)/DAC(CC).
        let is_sof =
            (0xC0..=0xCF).contains(&marker) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if is_sof {
            let mut sof = [0u8; 5];
            if file.read_exact(&mut sof).is_err() {
                return None;
            }
            let precision = u32::from(sof[0]);
            let height = u32::from(u16::from_be_bytes([sof[1], sof[2]]));
            let width = u32::from(u16::from_be_bytes([sof[3], sof[4]]));
            if width == 0 || height == 0 {
                return None;
            }
            return Some(ImageDims {
                width,
                height,
                bit_depth: Some(precision),
            });
        }

        // Not a frame header: skip the remainder of this segment.
        file.seek(SeekFrom::Current(i64::from(segment_len) - 2))
            .ok()?;
    }

    None
}

/// Resolution and duration extracted from an ISO base-media container.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct IsoInfo {
    /// Set once a `moov` box is located — gates a positive ISO-BMFF detection.
    found_moov: bool,
    /// Largest track display resolution in pixels (`0` until a `tkhd` is read).
    pixels: u64,
    /// Movie duration in seconds, from `mvhd` (timescale + duration).
    duration_secs: Option<f64>,
}

/// Probe an ISO base-media file (MP4 / MOV / M4V) for resolution and duration.
fn probe_isobmff(file: &mut File) -> Option<IsoInfo> {
    let file_len = file.metadata().ok()?.len();
    if file_len < 8 {
        return None;
    }

    // Reject non-ISO-BMFF inputs by sniffing the first top-level box type.
    file.seek(SeekFrom::Start(0)).ok()?;
    let mut header = [0u8; 8];
    if file.read_exact(&mut header).is_err() {
        return None;
    }
    let first_type = [header[4], header[5], header[6], header[7]];
    if !is_known_top_level_box(&first_type) {
        return None;
    }

    let mut info = IsoInfo::default();
    walk_boxes(file, 0, file_len, 0, &mut info);
    if info.found_moov {
        Some(info)
    } else {
        None
    }
}

/// Whether `box_type` is a recognised top-level ISO-BMFF box (detection guard).
fn is_known_top_level_box(box_type: &[u8; 4]) -> bool {
    matches!(
        box_type,
        b"ftyp"
            | b"moov"
            | b"mdat"
            | b"free"
            | b"skip"
            | b"wide"
            | b"styp"
            | b"sidx"
            | b"moof"
            | b"pdin"
            | b"meta"
    )
}

/// Recursively walk the boxes in `[start, end)`, descending only into `moov`/`trak`.
fn walk_boxes(file: &mut File, start: u64, end: u64, depth: u32, info: &mut IsoInfo) {
    if depth > MAX_BOX_DEPTH {
        return;
    }

    let mut pos = start;
    let mut count = 0u32;
    while pos + 8 <= end {
        count += 1;
        if count > MAX_BOXES_PER_LEVEL {
            break;
        }

        if file.seek(SeekFrom::Start(pos)).is_err() {
            break;
        }
        let mut header = [0u8; 8];
        if file.read_exact(&mut header).is_err() {
            break;
        }
        let size32 = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        let box_type = [header[4], header[5], header[6], header[7]];

        let (header_len, box_size) = if size32 == 1 {
            // 64-bit largesize follows the type.
            let mut ext = [0u8; 8];
            if file.read_exact(&mut ext).is_err() {
                break;
            }
            let large = u64::from_be_bytes(ext);
            if large < 16 {
                break;
            }
            (16u64, large)
        } else if size32 == 0 {
            // Box extends to the end of its parent.
            (8u64, end - pos)
        } else if size32 < 8 {
            break;
        } else {
            (8u64, u64::from(size32))
        };

        let box_end = match pos.checked_add(box_size) {
            Some(box_end) if box_end <= end => box_end,
            _ => break,
        };
        let content_start = pos + header_len;
        if content_start > box_end {
            break;
        }
        let content_len = box_end - content_start;

        match &box_type {
            b"moov" => {
                info.found_moov = true;
                walk_boxes(file, content_start, box_end, depth + 1, info);
            }
            b"trak" => walk_boxes(file, content_start, box_end, depth + 1, info),
            b"mvhd" => parse_mvhd(file, content_start, content_len, info),
            b"tkhd" => parse_tkhd(file, content_start, content_len, info),
            _ => {}
        }

        pos = box_end;
    }
}

/// Parse a movie header (`mvhd`) for the timescale and duration.
fn parse_mvhd(file: &mut File, content_start: u64, content_len: u64, info: &mut IsoInfo) {
    if content_len < 4 || file.seek(SeekFrom::Start(content_start)).is_err() {
        return;
    }
    let mut version_flags = [0u8; 4];
    if file.read_exact(&mut version_flags).is_err() {
        return;
    }

    let (timescale, duration) = if version_flags[0] == 1 {
        // version 1: 8-byte timestamps; timescale@+20 (u32), duration@+24 (u64).
        if content_len < 32 || file.seek(SeekFrom::Start(content_start + 20)).is_err() {
            return;
        }
        let mut buf = [0u8; 12];
        if file.read_exact(&mut buf).is_err() {
            return;
        }
        let timescale = u64::from(u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]));
        let duration = u64::from_be_bytes([
            buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
        ]);
        (timescale, duration)
    } else {
        // version 0: 4-byte timestamps; timescale@+12 (u32), duration@+16 (u32).
        if content_len < 20 || file.seek(SeekFrom::Start(content_start + 12)).is_err() {
            return;
        }
        let mut buf = [0u8; 8];
        if file.read_exact(&mut buf).is_err() {
            return;
        }
        let timescale = u64::from(u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]));
        let duration = u64::from(u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]));
        (timescale, duration)
    };

    if timescale > 0 && duration > 0 {
        info.duration_secs = Some(duration as f64 / timescale as f64);
    }
}

/// Parse a track header (`tkhd`) for the display resolution (16.16 fixed point).
fn parse_tkhd(file: &mut File, content_start: u64, content_len: u64, info: &mut IsoInfo) {
    if content_len < 4 || file.seek(SeekFrom::Start(content_start)).is_err() {
        return;
    }
    let mut version_flags = [0u8; 4];
    if file.read_exact(&mut version_flags).is_err() {
        return;
    }

    // width/height sit at the very end of the box; their offset depends on whether
    // the box uses 32-bit (v0) or 64-bit (v1) timestamps.
    let (width_offset, needed) = if version_flags[0] == 1 {
        (88u64, 96u64)
    } else {
        (76u64, 84u64)
    };
    if content_len < needed
        || file
            .seek(SeekFrom::Start(content_start + width_offset))
            .is_err()
    {
        return;
    }

    let mut buf = [0u8; 8];
    if file.read_exact(&mut buf).is_err() {
        return;
    }
    let width_fixed = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let height_fixed = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    // 16.16 fixed point → take the integer part.
    let width = u64::from(width_fixed >> 16);
    let height = u64::from(height_fixed >> 16);
    let pixels = width * height;
    // The video track has the largest dimensions; audio/text tracks are 0x0.
    if pixels > info.pixels {
        info.pixels = pixels;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("oximedia_quality_{}_{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir.join(name)
    }

    fn write_file(path: &Path, bytes: &[u8]) {
        let mut file = File::create(path).expect("create temp file");
        file.write_all(bytes).expect("write temp file");
    }

    /// Build a PNG whose header reports `width`x`height` at `bit_depth`, padded
    /// with zero bytes so the file occupies exactly `total_size` bytes.
    fn build_png(width: u32, height: u32, bit_depth: u8, total_size: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(total_size.max(26));
        bytes.extend_from_slice(&PNG_SIGNATURE);
        bytes.extend_from_slice(&13u32.to_be_bytes()); // IHDR length
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.push(bit_depth);
        bytes.push(2); // colour type (truecolour)
        while bytes.len() < total_size {
            bytes.push(0);
        }
        bytes
    }

    /// Build a JPEG with an optional leading APP0 segment followed by an SOF0
    /// frame header reporting `width`x`height` at `precision` bits.
    fn build_jpeg(width: u16, height: u16, precision: u8, with_app0: bool) -> Vec<u8> {
        let mut bytes = vec![0xFF, 0xD8]; // SOI
        if with_app0 {
            bytes.extend_from_slice(&[0xFF, 0xE0]); // APP0
            bytes.extend_from_slice(&16u16.to_be_bytes());
            bytes.extend_from_slice(b"JFIF\0");
            bytes.extend_from_slice(&[0x01, 0x01, 0x00, 0, 1, 0, 1, 0, 0]);
        }
        bytes.extend_from_slice(&[0xFF, 0xC0]); // SOF0
        bytes.extend_from_slice(&17u16.to_be_bytes()); // length (8 + 3*comp)
        bytes.push(precision);
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.push(3); // component count
        bytes.extend_from_slice(&[1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
        bytes.extend_from_slice(&[0xFF, 0xDA, 0x00, 0x02]); // SOS (truncated)
        bytes
    }

    fn iso_box(box_type: &[u8; 4], content: &[u8]) -> Vec<u8> {
        let size = (8 + content.len()) as u32;
        let mut bytes = Vec::with_capacity(size as usize);
        bytes.extend_from_slice(&size.to_be_bytes());
        bytes.extend_from_slice(box_type);
        bytes.extend_from_slice(content);
        bytes
    }

    /// Build a minimal but structurally valid MP4: `ftyp` + `moov`(`mvhd` +
    /// `trak`(`tkhd`)) + an `mdat` of `payload_len` bytes.
    fn build_mp4(
        width: u32,
        height: u32,
        timescale: u32,
        duration: u32,
        payload_len: usize,
    ) -> Vec<u8> {
        // mvhd (version 0): 20-byte minimal content carrying timescale + duration.
        let mut mvhd = Vec::new();
        mvhd.extend_from_slice(&[0, 0, 0, 0]); // version + flags
        mvhd.extend_from_slice(&0u32.to_be_bytes()); // creation
        mvhd.extend_from_slice(&0u32.to_be_bytes()); // modification
        mvhd.extend_from_slice(&timescale.to_be_bytes());
        mvhd.extend_from_slice(&duration.to_be_bytes());

        // tkhd (version 0): 84-byte content with width/height at offsets 76/80.
        let mut tkhd = vec![0u8; 84];
        tkhd[76..80].copy_from_slice(&(width << 16).to_be_bytes());
        tkhd[80..84].copy_from_slice(&(height << 16).to_be_bytes());

        let trak = iso_box(b"trak", &iso_box(b"tkhd", &tkhd));
        let mut moov_content = iso_box(b"mvhd", &mvhd);
        moov_content.extend_from_slice(&trak);

        let mut ftyp_content = Vec::new();
        ftyp_content.extend_from_slice(b"isom");
        ftyp_content.extend_from_slice(&0x200u32.to_be_bytes());
        ftyp_content.extend_from_slice(b"isom");

        let mut bytes = iso_box(b"ftyp", &ftyp_content);
        bytes.extend_from_slice(&iso_box(b"moov", &moov_content));
        bytes.extend_from_slice(&iso_box(b"mdat", &vec![0u8; payload_len]));
        bytes
    }

    #[test]
    fn test_png_header_parsed() {
        let dims = probe_png(&build_png(1920, 1080, 8, 64)).expect("png dims");
        assert_eq!(dims.width, 1920);
        assert_eq!(dims.height, 1080);
        assert_eq!(dims.bit_depth, Some(8));
    }

    #[test]
    fn test_jpeg_sof_parsed_with_and_without_app0() {
        for with_app0 in [false, true] {
            let path = temp_path(if with_app0 { "app0.jpg" } else { "plain.jpg" });
            write_file(&path, &build_jpeg(800, 600, 8, with_app0));
            let signals = probe_quality_signals(&path);
            assert_eq!(signals.pixels, Some(800 * 600));
            assert_eq!(signals.bit_depth, Some(8));
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn test_gif_header_parsed() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GIF89a");
        bytes.extend_from_slice(&640u16.to_le_bytes());
        bytes.extend_from_slice(&360u16.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let dims = probe_gif(&bytes).expect("gif dims");
        assert_eq!((dims.width, dims.height), (640, 360));
    }

    #[test]
    fn test_bmp_header_parsed() {
        let mut bytes = vec![0u8; 32];
        bytes[0] = b'B';
        bytes[1] = b'M';
        bytes[18..22].copy_from_slice(&1024i32.to_le_bytes());
        bytes[22..26].copy_from_slice(&(-768i32).to_le_bytes()); // top-down
        let dims = probe_bmp(&bytes).expect("bmp dims");
        assert_eq!((dims.width, dims.height), (1024, 768));
    }

    #[test]
    fn test_webp_vp8l_parsed() {
        // canvas 100x80 → (width-1)=99, (height-1)=79 packed 14 bits each.
        let bits: u32 = 99 | (79 << 14);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(b"WEBP");
        bytes.extend_from_slice(b"VP8L");
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.push(0x2F);
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 8]);
        let dims = probe_webp(&bytes).expect("webp dims");
        assert_eq!((dims.width, dims.height), (100, 80));
    }

    #[test]
    fn test_isobmff_resolution_and_duration() {
        let path = temp_path("clip.mp4");
        write_file(&path, &build_mp4(1920, 1080, 1000, 10_000, 4_096));
        let signals = probe_quality_signals(&path);
        assert_eq!(signals.pixels, Some(1920 * 1080));
        let duration = signals.duration_secs.expect("duration");
        assert!((duration - 10.0).abs() < 1e-9, "duration was {duration}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_quality_prefers_higher_resolution_even_when_smaller() {
        // A tiny 4K still vs. a huge SD still: resolution must dominate.
        let small_4k = temp_path("small_4k.png");
        let huge_sd = temp_path("huge_sd.png");
        write_file(&small_4k, &build_png(3840, 2160, 8, 2_048));
        write_file(&huge_sd, &build_png(720, 480, 8, 5_000_000));
        assert!(
            quality_score(&small_4k) > quality_score(&huge_sd),
            "a tiny 4K file must outrank a huge SD file"
        );
        let _ = std::fs::remove_file(&small_4k);
        let _ = std::fs::remove_file(&huge_sd);
    }

    #[test]
    fn test_quality_prefers_larger_file_at_same_resolution() {
        let lean = temp_path("lean.png");
        let rich = temp_path("rich.png");
        write_file(&lean, &build_png(1920, 1080, 8, 50_000));
        write_file(&rich, &build_png(1920, 1080, 8, 500_000));
        assert!(
            quality_score(&rich) > quality_score(&lean),
            "at equal resolution the larger (higher-bitrate) file wins"
        );
        let _ = std::fs::remove_file(&lean);
        let _ = std::fs::remove_file(&rich);
    }

    #[test]
    fn test_quality_bit_depth_breaks_ties() {
        // Identical resolution and size → higher bit depth wins.
        let depth8 = temp_path("depth8.png");
        let depth16 = temp_path("depth16.png");
        write_file(&depth8, &build_png(1280, 720, 8, 100_000));
        write_file(&depth16, &build_png(1280, 720, 16, 100_000));
        assert!(
            quality_score(&depth16) > quality_score(&depth8),
            "deeper bit depth breaks the tie at equal resolution and size"
        );
        let _ = std::fs::remove_file(&depth8);
        let _ = std::fs::remove_file(&depth16);
    }

    #[test]
    fn test_isobmff_higher_bitrate_wins_at_same_resolution() {
        // Same resolution and duration, larger mdat → higher bitrate → higher score.
        let lean = temp_path("lean.mp4");
        let rich = temp_path("rich.mp4");
        write_file(&lean, &build_mp4(1280, 720, 1000, 10_000, 10_000));
        write_file(&rich, &build_mp4(1280, 720, 1000, 10_000, 2_000_000));
        assert!(quality_score(&rich) > quality_score(&lean));
        let _ = std::fs::remove_file(&lean);
        let _ = std::fs::remove_file(&rich);
    }

    #[test]
    fn test_isobmff_shorter_duration_is_higher_bitrate() {
        // Equal file size but a shorter duration means a higher effective bitrate.
        let long_clip = temp_path("long.mp4");
        let short_clip = temp_path("short.mp4");
        write_file(&long_clip, &build_mp4(1280, 720, 1000, 40_000, 500_000));
        write_file(&short_clip, &build_mp4(1280, 720, 1000, 10_000, 500_000));
        assert!(
            quality_score(&short_clip) > quality_score(&long_clip),
            "same bytes over a shorter duration is a higher bitrate"
        );
        let _ = std::fs::remove_file(&long_clip);
        let _ = std::fs::remove_file(&short_clip);
    }

    #[test]
    fn test_unknown_format_falls_back_to_size() {
        // Files with no recognised magic must rank purely by size (legacy rule).
        let small = temp_path("small.bin");
        let large = temp_path("large.bin");
        write_file(&small, &[0u8; 100]);
        write_file(&large, &[0u8; 5_000]);
        let small_signals = probe_quality_signals(&small);
        assert_eq!(
            small_signals.pixels, None,
            "unknown format has no resolution"
        );
        assert_eq!(small_signals.duration_secs, None);
        assert!(quality_score(&large) > quality_score(&small));
        let _ = std::fs::remove_file(&small);
        let _ = std::fs::remove_file(&large);
    }

    #[test]
    fn test_score_is_deterministic() {
        let path = temp_path("stable.png");
        write_file(&path, &build_png(1920, 1080, 8, 4_096));
        assert_eq!(quality_score(&path), quality_score(&path));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_missing_file_scores_zero() {
        let path = temp_path("does_not_exist.bin");
        let _ = std::fs::remove_file(&path);
        assert_eq!(quality_score(&path), 0.0);
    }

    #[test]
    fn test_checked_pixels_rejects_zero() {
        assert_eq!(checked_pixels(0, 1080), None);
        assert_eq!(checked_pixels(1920, 0), None);
        assert_eq!(checked_pixels(1920, 1080), Some(1920 * 1080));
    }
}
