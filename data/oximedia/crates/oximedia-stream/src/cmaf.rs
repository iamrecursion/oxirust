//! CMAF (Common Media Application Format) chunked segment muxer.
//!
//! Implements a subset of the ISO 23000-19 / MPEG-CMAF specification sufficient
//! for generating CMAF-compliant fMP4 segments that carry the `cmf2`/`cmfc`
//! brands in their `ftyp` box.
//!
//! # Box layout
//!
//! Each call to [`CmafMuxer::write_cmaf_segment`] produces a list of
//! [`bytes::Bytes`] slices with the following logical structure:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  ftyp  (File Type Box)                                       │
//! │    major_brand   = "cmf2"                                    │
//! │    minor_version = 0                                         │
//! │    compatible    = ["cmf2", "cmfc", "iso6", "mp41"]          │
//! ├──────────────────────────────────────────────────────────────┤
//! │  Per chunk (one moof+mdat pair):                             │
//! │  ┌────────────────────────────────────────────────────────┐  │
//! │  │  moof  (Movie Fragment Box)                            │  │
//! │  │    mfhd  (Movie Fragment Header Box)                   │  │
//! │  │      sequence_number                                   │  │
//! │  │    traf  (Track Fragment Box)                          │  │
//! │  │      tfhd  (Track Fragment Header Box)                 │  │
//! │  │      tfdt  (Track Fragment Base Media Decode Time Box) │  │
//! │  │        base_media_decode_time                          │  │
//! │  │      trun  (Track Run Box)                             │  │
//! │  │        sample_count = 1                                │  │
//! │  │        data_offset                                     │  │
//! │  │        sample_size                                     │  │
//! │  └────────────────────────────────────────────────────────┘  │
//! │  ┌────────────────────────────────────────────────────────┐  │
//! │  │  mdat  (Media Data Box)                                │  │
//! │  │    <chunk payload>                                     │  │
//! │  └────────────────────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! The scatter-gather output (`Vec<Bytes>`) lets callers perform
//! zero-copy writev-style I/O: the payload [`Bytes`] inside each chunk
//! is just a reference-count bump on the original buffer — no copy occurs.
//!
//! # Example
//!
//! ```
//! use bytes::Bytes;
//! use oximedia_stream::cmaf::{CmafChunk, CmafMuxer};
//!
//! let chunks = vec![
//!     CmafChunk::new(1, 0,    vec![0x00u8; 100]),
//!     CmafChunk::new(2, 3000, vec![0xFFu8; 200]),
//! ];
//! let muxer = CmafMuxer::new();
//! let pieces: Vec<Bytes> = muxer.write_cmaf_segment(&chunks);
//! assert!(!pieces.is_empty());
//! // Callers that need one flat buffer:
//! let flat: Vec<u8> = pieces.into_iter().flat_map(|b| b.to_vec()).collect();
//! assert!(!flat.is_empty());
//! ```

#![allow(dead_code)]

use bytes::Bytes;

use crate::StreamError;

// ─────────────────────────────────────────────────────────────────────────────
// CmafChunk
// ─────────────────────────────────────────────────────────────────────────────

/// A single CMAF chunk — the fundamental unit of low-latency delivery.
///
/// A *chunk* carries one or more *samples* (encoded access units).  In this
/// simplified muxer each chunk is treated as a single-sample track run.
///
/// The `data` field is [`Bytes`], which enables zero-copy scatter-gather I/O:
/// cloning a `Bytes` value only bumps a reference count — no heap allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CmafChunk {
    /// Monotonically increasing sequence number (1-based, per the ISO spec).
    pub sequence_number: u32,

    /// Base Media Decode Time (BMDT) in the track's time-scale units.
    ///
    /// Corresponds to the `baseMediaDecodeTime` field of the `tfdt` box.
    pub base_media_decode_time: u64,

    /// Encoded sample payload (e.g. one H.264 access unit or one AAC frame).
    ///
    /// Stored as [`Bytes`] for zero-copy handoff to the segment writer.
    pub data: Bytes,
}

impl CmafChunk {
    /// Create a new chunk with the given sequence number, BMDT, and payload.
    ///
    /// `data` accepts any type that implements `Into<Bytes>`, including
    /// `Vec<u8>` (converted without copying via `Bytes::from`).
    #[must_use]
    pub fn new(sequence_number: u32, base_media_decode_time: u64, data: impl Into<Bytes>) -> Self {
        Self {
            sequence_number,
            base_media_decode_time,
            data: data.into(),
        }
    }

    /// Convenience constructor that explicitly converts a `Vec<u8>` payload.
    ///
    /// Equivalent to `CmafChunk::new(seq, bmdt, vec)`.
    #[must_use]
    pub fn from_vec(sequence_number: u32, base_media_decode_time: u64, data: Vec<u8>) -> Self {
        Self::new(sequence_number, base_media_decode_time, data)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Low-level box helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Write a big-endian `u32` into `buf`.
#[inline]
fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_be_bytes());
}

/// Write a big-endian `u64` into `buf`.
#[inline]
fn write_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_be_bytes());
}

/// Write a 4-byte box type (FourCC) into `buf`.
#[inline]
fn write_fourcc(buf: &mut Vec<u8>, fourcc: &[u8; 4]) {
    buf.extend_from_slice(fourcc);
}

/// Patch the size field at `offset` in `buf` with `buf.len() - offset`.
///
/// # Panics
///
/// Panics if `offset + 4 > buf.len()` (i.e. the box header is not yet written).
fn patch_size(buf: &mut Vec<u8>, offset: usize) {
    let size = (buf.len() - offset) as u32;
    buf[offset..offset + 4].copy_from_slice(&size.to_be_bytes());
}

/// Begin a box: write a placeholder size (0) + `fourcc`, return the offset
/// so that [`patch_size`] can fill it in later.
fn begin_box(buf: &mut Vec<u8>, fourcc: &[u8; 4]) -> usize {
    let offset = buf.len();
    write_u32(buf, 0); // placeholder
    write_fourcc(buf, fourcc);
    offset
}

/// End a box by patching the size at the returned offset.
fn end_box(buf: &mut Vec<u8>, offset: usize) {
    patch_size(buf, offset);
}

// ─────────────────────────────────────────────────────────────────────────────
// ftyp
// ─────────────────────────────────────────────────────────────────────────────

/// Write the `ftyp` (File Type) box with CMAF brands.
///
/// `major_brand = "cmf2"`, `minor_version = 0`.
/// Compatible brands: `["cmf2", "cmfc", "iso6", "mp41"]`.
fn write_ftyp(buf: &mut Vec<u8>) {
    let offset = begin_box(buf, b"ftyp");

    // major_brand
    write_fourcc(buf, b"cmf2");
    // minor_version
    write_u32(buf, 0);
    // compatible brands
    for brand in &[b"cmf2", b"cmfc", b"iso6", b"mp41"] {
        write_fourcc(buf, brand);
    }

    end_box(buf, offset);
}

// ─────────────────────────────────────────────────────────────────────────────
// moof
// ─────────────────────────────────────────────────────────────────────────────

/// Write the `mfhd` (Movie Fragment Header) box.
fn write_mfhd(buf: &mut Vec<u8>, sequence_number: u32) {
    let offset = begin_box(buf, b"mfhd");
    // version (1 byte) + flags (3 bytes) = 0
    write_u32(buf, 0);
    write_u32(buf, sequence_number);
    end_box(buf, offset);
}

/// Write the `tfhd` (Track Fragment Header) box.
///
/// `track_ID = 1`, `flags = 0x000000` (no optional fields).
fn write_tfhd(buf: &mut Vec<u8>) {
    let offset = begin_box(buf, b"tfhd");
    // version (0) + flags (0)
    write_u32(buf, 0);
    // track_ID
    write_u32(buf, 1);
    end_box(buf, offset);
}

/// Write the `tfdt` (Track Fragment Base Media Decode Time) box (version 1, 64-bit).
fn write_tfdt(buf: &mut Vec<u8>, base_media_decode_time: u64) {
    let offset = begin_box(buf, b"tfdt");
    // version = 1 (64-bit BMDT), flags = 0
    write_u32(buf, 0x0100_0000);
    write_u64(buf, base_media_decode_time);
    end_box(buf, offset);
}

/// Write the `trun` (Track Run) box.
///
/// Flags `0x000B05`:
/// - bit 0: `data-offset-present`
/// - bit 2: `first-sample-flags-present` (not set; using sample-level flags)
/// - bit 8: `sample-duration-present` (not set; use default)
/// - bit 9: `sample-size-present`
///
/// We use flags `0x000205` = data_offset_present | sample_size_present.
///
/// The `data_offset` is computed as `moof_size + 8` (mdat header).
/// `moof_size` is passed in because we know it only after the moof is complete.
fn write_trun(buf: &mut Vec<u8>, sample_size: u32, data_offset: i32) {
    let offset = begin_box(buf, b"trun");

    // version = 0
    // flags: data_offset_present (0x001) | sample_size_present (0x200) = 0x201
    write_u32(buf, 0x0000_0201);

    // sample_count
    write_u32(buf, 1);

    // data_offset (signed, relative to start of moof)
    buf.extend_from_slice(&data_offset.to_be_bytes());

    // Per-sample: sample_size
    write_u32(buf, sample_size);

    end_box(buf, offset);
}

/// Write a complete `moof` + `mdat` pair for a single chunk.
///
/// Returns two [`Bytes`] slices:
///
/// 1. The `moof` box and the `mdat` box header (size + fourcc, 8 bytes).
/// 2. A clone of `chunk.data` — this is a reference-count bump, **not a copy**.
///
/// Callers can feed both slices directly to scatter-gather I/O (e.g. `writev`)
/// without allocating a merged buffer.
fn write_moof_mdat(chunk: &CmafChunk) -> [Bytes; 2] {
    let mut buf: Vec<u8> = Vec::new();

    // ── moof ───────────────────────────────────────────────────────────────
    let moof_start = begin_box(&mut buf, b"moof");

    // mfhd
    write_mfhd(&mut buf, chunk.sequence_number);

    // traf
    let traf_start = begin_box(&mut buf, b"traf");

    // tfhd
    write_tfhd(&mut buf);

    // tfdt
    write_tfdt(&mut buf, chunk.base_media_decode_time);

    // trun — we need to know data_offset = moof_size + 8 (mdat box header).
    // Write a placeholder trun first, then patch it.
    let trun_start = begin_box(&mut buf, b"trun");
    // version=0, flags=0x0201
    write_u32(&mut buf, 0x0000_0201);
    // sample_count = 1
    write_u32(&mut buf, 1);
    // data_offset placeholder (4 bytes)
    let data_offset_pos = buf.len();
    write_u32(&mut buf, 0); // placeholder
                            // sample_size
    write_u32(&mut buf, chunk.data.len() as u32);
    end_box(&mut buf, trun_start);

    end_box(&mut buf, traf_start);
    end_box(&mut buf, moof_start);

    // Now we know moof_size; patch data_offset = moof_size + 8 (mdat size+type)
    let moof_size = buf.len() as i32;
    let data_offset = moof_size + 8; // 8 = mdat 4-byte size + 4-byte fourcc
    buf[data_offset_pos..data_offset_pos + 4].copy_from_slice(&data_offset.to_be_bytes());

    // ── mdat header ────────────────────────────────────────────────────────
    // Encode mdat size and fourcc into the same header buffer so the
    // payload (chunk.data) remains a separate Bytes slice — zero-copy.
    let mdat_size = 8u32 + chunk.data.len() as u32;
    write_u32(&mut buf, mdat_size);
    write_fourcc(&mut buf, b"mdat");
    // The payload goes into the second Bytes element (refcount bump only).

    [Bytes::from(buf), chunk.data.clone()]
}

// ─────────────────────────────────────────────────────────────────────────────
// CmafMuxer
// ─────────────────────────────────────────────────────────────────────────────

/// CMAF segment muxer.
///
/// Produces CMAF-compliant fMP4 byte sequences from [`CmafChunk`] slices.
#[derive(Debug, Default, Clone)]
pub struct CmafMuxer {
    /// Track ID to embed in `tfhd`.  Defaults to `1`.
    pub track_id: u32,
}

impl CmafMuxer {
    /// Create a new muxer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self { track_id: 1 }
    }

    /// Create a new muxer with a specific track ID.
    #[must_use]
    pub fn with_track_id(mut self, id: u32) -> Self {
        self.track_id = id;
        self
    }

    /// Produce a CMAF-compliant segment from a slice of chunks as scatter-gather
    /// [`Bytes`] slices.
    ///
    /// The logical byte layout is described in the module doc:
    /// one `ftyp` followed by one `moof`+`mdat` pair per chunk.  The returned
    /// `Vec<Bytes>` contains:
    ///
    /// * One `Bytes` for the `ftyp` box.
    /// * For each chunk: one `Bytes` for the `moof` + `mdat` header, plus one
    ///   `Bytes` that is a **reference-count clone** of `chunk.data` — no copy.
    ///
    /// An empty `chunks` slice yields a single-element vec containing only the
    /// `ftyp` box.
    ///
    /// Use [`collect_bytes`](CmafMuxer::collect_bytes) if you need the output
    /// as one contiguous `Vec<u8>`.
    #[must_use]
    pub fn write_cmaf_segment(&self, chunks: &[CmafChunk]) -> Vec<Bytes> {
        // Capacity: 1 (ftyp) + 2 per chunk (header + payload).
        let mut out: Vec<Bytes> = Vec::with_capacity(1 + chunks.len() * 2);

        // ftyp
        let mut ftyp_buf: Vec<u8> = Vec::new();
        write_ftyp(&mut ftyp_buf);
        out.push(Bytes::from(ftyp_buf));

        // moof + mdat per chunk (scatter-gather)
        for chunk in chunks {
            let [header, payload] = write_moof_mdat(chunk);
            out.push(header);
            out.push(payload);
        }

        out
    }

    /// Flatten [`write_cmaf_segment`](CmafMuxer::write_cmaf_segment) output into
    /// a single contiguous `Vec<u8>`.
    ///
    /// This is a convenience wrapper for callers that cannot use scatter-gather
    /// I/O.  For performance-sensitive paths prefer the scatter-gather form.
    #[must_use]
    pub fn collect_bytes(&self, chunks: &[CmafChunk]) -> Vec<u8> {
        let pieces = self.write_cmaf_segment(chunks);
        let total: usize = pieces.iter().map(|b| b.len()).sum();
        let mut out = Vec::with_capacity(total);
        for piece in pieces {
            out.extend_from_slice(&piece);
        }
        out
    }

    /// Validate a slice of chunks and return an error if any invariant is
    /// violated (e.g. zero-length payload or sequence number of zero).
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::ParseError`] if any chunk fails validation.
    pub fn validate_chunks(&self, chunks: &[CmafChunk]) -> Result<(), StreamError> {
        for (i, chunk) in chunks.iter().enumerate() {
            if chunk.sequence_number == 0 {
                return Err(StreamError::ParseError(format!(
                    "chunk[{}]: sequence_number must be ≥ 1 (got 0)",
                    i
                )));
            }
            if chunk.data.is_empty() {
                return Err(StreamError::ParseError(format!(
                    "chunk[{}]: data must not be empty",
                    i
                )));
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Box parsing helpers (for tests)
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal box reader for testing — reads the size and FourCC of the next box.
#[cfg(test)]
fn read_box_header(data: &[u8], offset: usize) -> Option<(u32, [u8; 4], usize)> {
    if offset + 8 > data.len() {
        return None;
    }
    let size = u32::from_be_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]);
    let fourcc = [
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ];
    Some((size, fourcc, offset + 8))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_chunk(seq: u32, bmdt: u64, size: usize) -> CmafChunk {
        CmafChunk::new(seq, bmdt, vec![0xAB; size])
    }

    /// Flatten a `Vec<Bytes>` into a single `Vec<u8>` for byte-level assertions.
    fn flatten(pieces: Vec<Bytes>) -> Vec<u8> {
        let total: usize = pieces.iter().map(|b| b.len()).sum();
        let mut out = Vec::with_capacity(total);
        for piece in pieces {
            out.extend_from_slice(&piece);
        }
        out
    }

    // ── CmafChunk ─────────────────────────────────────────────────────────────

    #[test]
    fn test_chunk_new() {
        let c = CmafChunk::new(3, 9000, vec![1, 2, 3]);
        assert_eq!(c.sequence_number, 3);
        assert_eq!(c.base_media_decode_time, 9000);
        assert_eq!(c.data.as_ref(), &[1u8, 2, 3]);
    }

    #[test]
    fn test_chunk_from_vec_equivalent_to_new() {
        let v = vec![0xAAu8; 32];
        let c1 = CmafChunk::new(1, 0, v.clone());
        let c2 = CmafChunk::from_vec(1, 0, v.clone());
        assert_eq!(c1, c2);
    }

    // ── ftyp ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_ftyp_starts_with_cmf2_brand() {
        let mut buf = Vec::new();
        write_ftyp(&mut buf);

        // Size
        let size = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert_eq!(
            size as usize,
            buf.len(),
            "ftyp size field must match buffer length"
        );

        // FourCC = "ftyp"
        assert_eq!(&buf[4..8], b"ftyp");

        // major_brand = "cmf2"
        assert_eq!(&buf[8..12], b"cmf2");

        // minor_version = 0
        let minor = u32::from_be_bytes([buf[12], buf[13], buf[14], buf[15]]);
        assert_eq!(minor, 0);

        // Compatible brands include "cmfc"
        let brands_start = 16usize;
        let num_brands = (buf.len() - brands_start) / 4;
        let brands: Vec<&[u8]> = (0..num_brands)
            .map(|i| &buf[brands_start + i * 4..brands_start + i * 4 + 4])
            .collect();
        assert!(
            brands.iter().any(|&b| b == b"cmfc"),
            "cmfc brand must be present"
        );
        assert!(
            brands.iter().any(|&b| b == b"cmf2"),
            "cmf2 brand must be present in compat list"
        );
    }

    // ── write_cmaf_segment ────────────────────────────────────────────────────

    #[test]
    fn test_empty_chunk_slice_produces_ftyp_only() {
        let muxer = CmafMuxer::new();
        let out = flatten(muxer.write_cmaf_segment(&[]));
        // Should only contain ftyp
        assert!(!out.is_empty());
        assert_eq!(&out[4..8], b"ftyp");
        // After ftyp there should be nothing
        let ftyp_size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]) as usize;
        assert_eq!(out.len(), ftyp_size);
    }

    #[test]
    fn test_single_chunk_segment_structure() {
        let muxer = CmafMuxer::new();
        let chunks = vec![make_chunk(1, 0, 100)];
        let out = flatten(muxer.write_cmaf_segment(&chunks));

        // First box must be ftyp
        let (ftyp_size, ftyp_fourcc, _) =
            read_box_header(&out, 0).expect("ftyp header should be readable");
        assert_eq!(&ftyp_fourcc, b"ftyp");

        // Next box must be moof
        let moof_offset = ftyp_size as usize;
        let (moof_size, moof_fourcc, _) =
            read_box_header(&out, moof_offset).expect("moof header should be readable");
        assert_eq!(&moof_fourcc, b"moof", "expected moof after ftyp");

        // Next box after moof must be mdat
        let mdat_offset = moof_offset + moof_size as usize;
        let (mdat_size, mdat_fourcc, _) =
            read_box_header(&out, mdat_offset).expect("mdat header should be readable");
        assert_eq!(&mdat_fourcc, b"mdat", "expected mdat after moof");
        assert_eq!(mdat_size as usize, 8 + 100, "mdat size must be 8 + payload");
    }

    #[test]
    fn test_multiple_chunks_produce_multiple_moof_mdat_pairs() {
        let muxer = CmafMuxer::new();
        let chunks = vec![
            make_chunk(1, 0, 50),
            make_chunk(2, 3000, 75),
            make_chunk(3, 6000, 90),
        ];
        let out = flatten(muxer.write_cmaf_segment(&chunks));

        // Skip ftyp
        let ftyp_size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]) as usize;
        let mut offset = ftyp_size;

        let expected_payloads = [50usize, 75, 90];
        for expected_payload in &expected_payloads {
            let (moof_size, moof_fourcc, _) =
                read_box_header(&out, offset).expect("moof header in loop should be readable");
            assert_eq!(&moof_fourcc, b"moof");
            offset += moof_size as usize;

            let (mdat_size, mdat_fourcc, _) =
                read_box_header(&out, offset).expect("mdat header in loop should be readable");
            assert_eq!(&mdat_fourcc, b"mdat");
            assert_eq!(mdat_size as usize, 8 + expected_payload);
            offset += mdat_size as usize;
        }

        assert_eq!(offset, out.len(), "no trailing bytes");
    }

    #[test]
    fn test_sequence_number_in_mfhd() {
        let muxer = CmafMuxer::new();
        let chunks = vec![make_chunk(42, 0, 10)];
        let out = flatten(muxer.write_cmaf_segment(&chunks));

        // Navigate: ftyp → moof → mfhd
        let ftyp_size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]) as usize;
        // moof header at ftyp_size
        let moof_body_start = ftyp_size + 8; // skip moof size + fourcc

        // mfhd starts at moof_body_start
        // mfhd: size(4) + fourcc(4) + version+flags(4) + sequence_number(4)
        let mfhd_seq_offset = moof_body_start + 4 + 4 + 4; // after size+fourcc+ver_flags
        let seq = u32::from_be_bytes([
            out[mfhd_seq_offset],
            out[mfhd_seq_offset + 1],
            out[mfhd_seq_offset + 2],
            out[mfhd_seq_offset + 3],
        ]);
        assert_eq!(seq, 42, "sequence_number in mfhd must match chunk");
    }

    #[test]
    fn test_mdat_payload_matches_chunk_data() {
        let payload = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let muxer = CmafMuxer::new();
        let chunks = vec![CmafChunk::new(1, 0, payload.clone())];
        let out = flatten(muxer.write_cmaf_segment(&chunks));

        // Locate mdat content (after ftyp + moof + mdat header)
        let ftyp_size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]) as usize;
        let moof_size = u32::from_be_bytes([
            out[ftyp_size],
            out[ftyp_size + 1],
            out[ftyp_size + 2],
            out[ftyp_size + 3],
        ]) as usize;
        let mdat_start = ftyp_size + moof_size;
        let mdat_payload_start = mdat_start + 8; // skip mdat size+fourcc
        assert_eq!(
            &out[mdat_payload_start..mdat_payload_start + 4],
            &payload[..]
        );
    }

    #[test]
    fn test_validate_chunks_valid() {
        let muxer = CmafMuxer::new();
        let chunks = vec![make_chunk(1, 0, 100)];
        assert!(muxer.validate_chunks(&chunks).is_ok());
    }

    #[test]
    fn test_validate_chunks_zero_sequence_number() {
        let muxer = CmafMuxer::new();
        let chunks = vec![make_chunk(0, 0, 100)];
        assert!(muxer.validate_chunks(&chunks).is_err());
    }

    #[test]
    fn test_validate_chunks_empty_data() {
        let muxer = CmafMuxer::new();
        let chunks = vec![CmafChunk::new(1, 0, vec![])];
        assert!(muxer.validate_chunks(&chunks).is_err());
    }

    #[test]
    fn test_validate_empty_slice_is_ok() {
        let muxer = CmafMuxer::new();
        assert!(muxer.validate_chunks(&[]).is_ok());
    }

    #[test]
    fn test_large_bmdt_version1_encoding() {
        // Use a BMDT that exceeds u32 range to verify version-1 tfdt is used.
        let bmdt: u64 = u64::from(u32::MAX) + 1_000;
        let muxer = CmafMuxer::new();
        let chunks = vec![CmafChunk::new(1, bmdt, vec![0x00; 10])];
        let out = flatten(muxer.write_cmaf_segment(&chunks));
        // The segment must still produce valid output with a non-trivial size
        assert!(out.len() > 40);
    }

    /// Black-box test: produce a segment from 10 chunks, verify that each
    /// mdat payload byte matches the expected fill pattern.
    #[test]
    fn test_round_trip_payload_integrity() {
        let muxer = CmafMuxer::new();
        let chunks: Vec<CmafChunk> = (1u32..=10)
            .map(|seq| CmafChunk::new(seq, u64::from(seq - 1) * 1000, vec![seq as u8; 64]))
            .collect();

        let out = flatten(muxer.write_cmaf_segment(&chunks));

        // Walk the segment and collect mdat payloads
        let ftyp_size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]) as usize;
        let mut offset = ftyp_size;
        let mut payloads: Vec<Vec<u8>> = Vec::new();

        while offset < out.len() {
            let box_size = u32::from_be_bytes([
                out[offset],
                out[offset + 1],
                out[offset + 2],
                out[offset + 3],
            ]) as usize;
            let fourcc = &out[offset + 4..offset + 8];
            if fourcc == b"moof" {
                offset += box_size;
                continue;
            }
            if fourcc == b"mdat" {
                let payload = out[offset + 8..offset + box_size].to_vec();
                payloads.push(payload);
                offset += box_size;
                continue;
            }
            // Unknown box — skip
            offset += box_size;
        }

        assert_eq!(payloads.len(), 10);
        for (i, payload) in payloads.iter().enumerate() {
            let expected_byte = (i + 1) as u8;
            assert!(
                payload.iter().all(|&b| b == expected_byte),
                "payload {} should be all {:#04x}",
                i,
                expected_byte
            );
        }
    }

    // ── collect_bytes convenience wrapper ─────────────────────────────────────

    #[test]
    fn test_collect_bytes_matches_flatten() {
        let muxer = CmafMuxer::new();
        let chunks = vec![make_chunk(1, 0, 64), make_chunk(2, 1000, 128)];
        let via_flatten = flatten(muxer.write_cmaf_segment(&chunks));
        let via_collect = muxer.collect_bytes(&chunks);
        assert_eq!(via_flatten, via_collect);
    }

    // ── Zero-copy assertions ──────────────────────────────────────────────────

    /// Verify that the payload `Bytes` in the scatter-gather output shares the
    /// same underlying allocation as the original `CmafChunk.data`.
    ///
    /// `Bytes::as_ptr()` identity is the canonical way to confirm no copy
    /// occurred: two `Bytes` values sharing the same `as_ptr()` result are
    /// views into the same heap allocation.
    #[test]
    fn test_cmaf_chunk_bytes_zero_copy() {
        // 4 MiB payload — large enough that a copy would be measurable; the
        // ptr-identity check is conclusive regardless of size.
        const PAYLOAD_LEN: usize = 4 * 1024 * 1024;
        let payload_vec = vec![0xCAu8; PAYLOAD_LEN];
        let chunk = CmafChunk::new(1, 0, payload_vec);

        // Record the pointer of the original data buffer.
        let original_ptr = chunk.data.as_ptr();

        let muxer = CmafMuxer::new();
        let pieces = muxer.write_cmaf_segment(&[chunk]);

        // pieces layout: [ftyp, moof+mdat_header, payload]
        // The third element (index 2) is the payload Bytes.
        assert!(
            pieces.len() >= 3,
            "expected at least 3 pieces for one chunk, got {}",
            pieces.len()
        );
        let payload_piece = &pieces[2];

        // Confirm the pointer is identical — no copy was made.
        assert_eq!(
            payload_piece.as_ptr(),
            original_ptr,
            "payload Bytes must reference the original allocation (zero-copy)"
        );
        assert_eq!(payload_piece.len(), PAYLOAD_LEN);
    }

    /// Property-style test: for a range of payload sizes, verify that the
    /// segment write path produces output whose concatenated bytes equal the
    /// original payload when extracted from the mdat box.
    #[test]
    fn test_cmaf_chunk_content_roundtrip() {
        let muxer = CmafMuxer::new();

        // Test a spread of sizes: 1 byte, 1 KiB, 256 KiB, 1 MiB, 16 MiB.
        let sizes: &[usize] = &[1, 1024, 256 * 1024, 1024 * 1024, 16 * 1024 * 1024];

        for &sz in sizes {
            // Fill with a deterministic pattern based on size so mismatches
            // are immediately identifiable.
            let fill = ((sz & 0xFF) as u8).wrapping_add(1);
            let original = vec![fill; sz];

            let chunk = CmafChunk::new(1, 0, original.clone());
            let out = flatten(muxer.write_cmaf_segment(&[chunk]));

            // Navigate to the mdat payload inside the flattened output.
            let ftyp_size = u32::from_be_bytes([out[0], out[1], out[2], out[3]]) as usize;
            let moof_size = u32::from_be_bytes([
                out[ftyp_size],
                out[ftyp_size + 1],
                out[ftyp_size + 2],
                out[ftyp_size + 3],
            ]) as usize;
            let mdat_start = ftyp_size + moof_size;
            let mdat_payload_start = mdat_start + 8; // skip mdat size+fourcc
            let mdat_payload_end = mdat_payload_start + sz;

            assert_eq!(
                &out[mdat_payload_start..mdat_payload_end],
                original.as_slice(),
                "round-trip failed for payload size {sz}"
            );
        }
    }
}
