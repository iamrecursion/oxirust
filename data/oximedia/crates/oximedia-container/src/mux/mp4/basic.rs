//! Basic streaming MP4 muxer.
//!
//! Provides [`BasicMp4Muxer`], a low-level MP4 box writer that operates
//! directly on any [`std::io::Write`] sink.  It exposes three primitive
//! operations matching the ISOBMFF box layout:
//!
//! 1. [`BasicMp4Muxer::write_ftyp`] — writes the `ftyp` (file-type) box.
//! 2. [`BasicMp4Muxer::write_mdat`] — writes an `mdat` (media data) box.
//! 3. [`BasicMp4Muxer::write_moov`] — writes a minimal `moov` box tree
//!    (`moov → mvhd + trak → tkhd + mdia → mdhd + hdlr + minf → ...`).
//!
//! This is intentionally simpler than [`super::Mp4Muxer`]: it performs no
//! packet buffering and is suitable for writing pre-encoded data in a
//! single pass.

#![forbid(unsafe_code)]

use std::io::{self, Write};

use crate::track_info::{TrackInfo, TrackType};

// ─── Error ───────────────────────────────────────────────────────────────────

/// Errors produced by [`BasicMp4Muxer`].
#[derive(Debug, thiserror::Error)]
pub enum BasicMp4Error {
    /// An I/O error occurred while writing to the output sink.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    /// The brand string is not exactly 4 ASCII bytes.
    #[error("invalid brand '{0}': must be exactly 4 ASCII bytes")]
    InvalidBrand(String),
}

// ─── BasicMp4Muxer ───────────────────────────────────────────────────────────

/// Low-level MP4 box writer that streams bytes to any [`Write`] sink.
///
/// # Example
///
/// ```
/// use std::io::Cursor;
/// use oximedia_container::mux::mp4::BasicMp4Muxer;
/// use oximedia_container::track_info::{TrackInfo, TrackType};
///
/// // Cursor<Vec<u8>> is 'static and implements Write.
/// let cursor = Cursor::new(Vec::<u8>::new());
/// let mut muxer = BasicMp4Muxer::new(Box::new(cursor));
///
/// muxer.write_ftyp("isom").expect("ftyp written");
/// muxer.write_mdat(b"encoded-media-data-here").expect("mdat written");
///
/// let tracks = vec![TrackInfo::new(0, TrackType::Video, "av01")];
/// muxer.write_moov(&tracks).expect("moov written");
/// ```
pub struct BasicMp4Muxer {
    output: Box<dyn Write>,
}

impl BasicMp4Muxer {
    /// Creates a new `BasicMp4Muxer` that writes to `output`.
    pub fn new(output: Box<dyn Write>) -> Self {
        Self { output }
    }

    /// Writes an `ftyp` (file type) box.
    ///
    /// `brand` must be exactly 4 ASCII characters (e.g. `"isom"`, `"av01"`).
    /// Compatible brands are set to the same value as the major brand with
    /// minor version 0.
    ///
    /// # Errors
    ///
    /// Returns [`BasicMp4Error::InvalidBrand`] if `brand` is not 4 ASCII bytes,
    /// or [`BasicMp4Error::Io`] on write failure.
    pub fn write_ftyp(&mut self, brand: &str) -> Result<(), BasicMp4Error> {
        let brand_bytes = validate_brand(brand)?;

        // ftyp payload: major_brand (4) + minor_version (4) + compatible_brands (4×n)
        // We write one compatible brand equal to the major brand.
        let payload_size: u32 = 4 + 4 + 4; // major + minor + 1 compatible brand
        let box_size: u32 = 8 + payload_size; // box header (4 size + 4 type) + payload

        write_box_header(&mut self.output, box_size, b"ftyp")?;
        self.output.write_all(&brand_bytes)?; // major brand
        self.output.write_all(&0u32.to_be_bytes())?; // minor version
        self.output.write_all(&brand_bytes)?; // compatible brand
        Ok(())
    }

    /// Writes an `mdat` (media data) box containing `data`.
    ///
    /// For payloads whose total box size (8-byte header + `data`) would not
    /// fit in a 32-bit field, this emits the ISO/IEC 14496-12 §4.2
    /// `largesize` form (`size` field set to the sentinel `1`, followed by
    /// an 8-byte big-endian `largesize` field carrying the true size)
    /// instead of writing a header that lies about its own length. Silently
    /// clamping the 32-bit `size` field for an oversized payload would still
    /// write every byte of `data` after a box header that undersells its own
    /// length, corrupting the file for any conformant demuxer.
    ///
    /// # Errors
    ///
    /// Returns [`BasicMp4Error::Io`] on write failure.
    pub fn write_mdat(&mut self, data: &[u8]) -> Result<(), BasicMp4Error> {
        let header = box_header_for_len(b"mdat", data.len() as u64);
        self.output.write_all(&header)?;
        self.output.write_all(data)?;
        Ok(())
    }

    /// Writes a minimal `moov` box tree for the given track list.
    ///
    /// The generated structure is:
    ///
    /// ```text
    /// moov
    ///   mvhd  (movie header — duration=0, timescale=1000)
    ///   trak  (one per TrackInfo)
    ///     tkhd  (track header)
    ///     mdia
    ///       mdhd  (media header)
    ///       hdlr  (handler — "vide" or "soun")
    ///       minf
    ///         vmhd / smhd  (video / audio media info)
    ///         dinf
    ///           dref (url)
    ///         stbl
    ///           stsd (empty)
    ///           stts (empty)
    ///           stsc (empty)
    ///           stsz (empty)
    ///           stco (empty)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BasicMp4Error::Io`] on write failure.
    pub fn write_moov(&mut self, tracks: &[TrackInfo]) -> Result<(), BasicMp4Error> {
        let moov_payload = build_moov_payload(tracks);
        write_box_header(&mut self.output, 8 + moov_payload.len() as u32, b"moov")?;
        self.output.write_all(&moov_payload)?;
        Ok(())
    }
}

// ─── Box building helpers ────────────────────────────────────────────────────

/// Write the 8-byte box header (4-byte big-endian size + 4-byte type).
fn write_box_header(w: &mut dyn Write, size: u32, box_type: &[u8; 4]) -> io::Result<()> {
    w.write_all(&size.to_be_bytes())?;
    w.write_all(box_type)?;
    Ok(())
}

/// Builds the box header bytes for a box whose payload is `payload_len`
/// bytes, per ISO/IEC 14496-12 §4.2.
///
/// Returns the standard 8-byte header (`size:u32` + `type:4`) when the total
/// box size (header + payload) fits in a 32-bit field. Otherwise returns the
/// 16-byte `largesize` header: a `size` field of `1` (the sentinel meaning
/// "the real size follows as a 64-bit field"), the 4-byte type, and an
/// 8-byte big-endian `largesize` carrying the true total size.
///
/// This is a pure function over the payload length (rather than being
/// inlined into a `data: &[u8]`-taking writer) so the 4 GiB boundary can be
/// exercised in tests without allocating gigabytes of data.
fn box_header_for_len(box_type: &[u8; 4], payload_len: u64) -> Vec<u8> {
    let small_total = 8u64.saturating_add(payload_len);
    if let Ok(size) = u32::try_from(small_total) {
        let mut header = Vec::with_capacity(8);
        header.extend_from_slice(&size.to_be_bytes());
        header.extend_from_slice(box_type);
        header
    } else {
        let large_total = 16u64.saturating_add(payload_len);
        let mut header = Vec::with_capacity(16);
        header.extend_from_slice(&1u32.to_be_bytes());
        header.extend_from_slice(box_type);
        header.extend_from_slice(&large_total.to_be_bytes());
        header
    }
}

/// Write a complete box with `box_type` and `payload`.
fn encode_box(box_type: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let total = 8u32.saturating_add(payload.len() as u32);
    let mut out = Vec::with_capacity(total as usize);
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(box_type);
    out.extend_from_slice(payload);
    out
}

/// Write a FullBox (version + flags) with `box_type` and `payload`.
fn encode_full_box(box_type: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut full_payload = Vec::with_capacity(4 + payload.len());
    full_payload.push(version);
    // 3-byte flags big-endian
    full_payload.push(((flags >> 16) & 0xFF) as u8);
    full_payload.push(((flags >> 8) & 0xFF) as u8);
    full_payload.push((flags & 0xFF) as u8);
    full_payload.extend_from_slice(payload);
    encode_box(box_type, &full_payload)
}

/// Validate and convert a 4-char brand string to bytes.
fn validate_brand(brand: &str) -> Result<[u8; 4], BasicMp4Error> {
    if brand.len() != 4 || !brand.is_ascii() {
        return Err(BasicMp4Error::InvalidBrand(brand.to_owned()));
    }
    let b = brand.as_bytes();
    Ok([b[0], b[1], b[2], b[3]])
}

/// Build the complete moov box payload (all children concatenated).
fn build_moov_payload(tracks: &[TrackInfo]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend(build_mvhd());
    for (i, track) in tracks.iter().enumerate() {
        payload.extend(build_trak(track, (i + 1) as u32));
    }
    payload
}

/// Build the `mvhd` (movie header) box.
///
/// Version 0: creation_time, modification_time, timescale, duration are all u32.
fn build_mvhd() -> Vec<u8> {
    let timescale: u32 = 1000;
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // creation_time
    payload.extend_from_slice(&0u32.to_be_bytes()); // modification_time
    payload.extend_from_slice(&timescale.to_be_bytes()); // timescale
    payload.extend_from_slice(&0u32.to_be_bytes()); // duration
    payload.extend_from_slice(&0x00010000u32.to_be_bytes()); // rate = 1.0 (16.16)
    payload.extend_from_slice(&0x0100u16.to_be_bytes()); // volume = 1.0 (8.8)
    payload.extend_from_slice(&[0u8; 10]); // reserved
                                           // Unity matrix (9 × i32)
    let matrix: [u32; 9] = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
    for v in &matrix {
        payload.extend_from_slice(&v.to_be_bytes());
    }
    payload.extend_from_slice(&[0u8; 24]); // pre_defined
    payload.extend_from_slice(&0xFFFF_FFFFu32.to_be_bytes()); // next_track_id
    encode_full_box(b"mvhd", 0, 0, &payload)
}

/// Build the `trak` box for a single track.
fn build_trak(track: &TrackInfo, track_id: u32) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend(build_tkhd(track_id, &track.track_type));
    payload.extend(build_mdia(track));
    encode_box(b"trak", &payload)
}

/// Build the `tkhd` (track header) box.
fn build_tkhd(track_id: u32, _track_type: &TrackType) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // creation_time
    payload.extend_from_slice(&0u32.to_be_bytes()); // modification_time
    payload.extend_from_slice(&track_id.to_be_bytes()); // track_id
    payload.extend_from_slice(&0u32.to_be_bytes()); // reserved
    payload.extend_from_slice(&0u32.to_be_bytes()); // duration
    payload.extend_from_slice(&[0u8; 8]); // reserved
    payload.extend_from_slice(&0i16.to_be_bytes()); // layer
    payload.extend_from_slice(&0i16.to_be_bytes()); // alternate_group
    payload.extend_from_slice(&0x0100u16.to_be_bytes()); // volume
    payload.extend_from_slice(&0u16.to_be_bytes()); // reserved
                                                    // Unity matrix
    let matrix: [u32; 9] = [0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000];
    for v in &matrix {
        payload.extend_from_slice(&v.to_be_bytes());
    }
    payload.extend_from_slice(&0u32.to_be_bytes()); // width  (16.16)
    payload.extend_from_slice(&0u32.to_be_bytes()); // height (16.16)
                                                    // flags = 3 (track enabled + in movie)
    encode_full_box(b"tkhd", 0, 3, &payload)
}

/// Build the `mdia` (media) box.
fn build_mdia(track: &TrackInfo) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend(build_mdhd());
    payload.extend(build_hdlr(&track.track_type));
    payload.extend(build_minf(&track.track_type));
    encode_box(b"mdia", &payload)
}

/// Build the `mdhd` (media header) box.
fn build_mdhd() -> Vec<u8> {
    let timescale: u32 = 1000;
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // creation_time
    payload.extend_from_slice(&0u32.to_be_bytes()); // modification_time
    payload.extend_from_slice(&timescale.to_be_bytes()); // timescale
    payload.extend_from_slice(&0u32.to_be_bytes()); // duration
                                                    // language = 'und' (packed ISO 639-2/T: 0x55C4 = 'und')
    payload.extend_from_slice(&0x55C4u16.to_be_bytes());
    payload.extend_from_slice(&0u16.to_be_bytes()); // pre_defined
    encode_full_box(b"mdhd", 0, 0, &payload)
}

/// Build the `hdlr` (handler) box.
fn build_hdlr(track_type: &TrackType) -> Vec<u8> {
    let (handler_type, name): (&[u8; 4], &str) = match track_type {
        TrackType::Video => (b"vide", "OxiMedia Video Handler"),
        TrackType::Audio => (b"soun", "OxiMedia Audio Handler"),
        TrackType::Subtitle => (b"text", "OxiMedia Text Handler"),
        TrackType::Data => (b"data", "OxiMedia Data Handler"),
    };
    let mut payload = Vec::new();
    payload.extend_from_slice(&0u32.to_be_bytes()); // pre_defined
    payload.extend_from_slice(handler_type); // handler_type
    payload.extend_from_slice(&[0u8; 12]); // reserved (3 × u32)
    payload.extend_from_slice(name.as_bytes()); // name (null-terminated)
    payload.push(0u8); // null terminator
    encode_full_box(b"hdlr", 0, 0, &payload)
}

/// Build the `minf` (media information) box.
fn build_minf(track_type: &TrackType) -> Vec<u8> {
    let mut payload = Vec::new();

    // media-type specific header box
    match track_type {
        TrackType::Video => {
            // vmhd: graphicsMode=0, opcolor=[0,0,0]
            let vmhd_payload = [0u8; 8]; // graphicsMode (2) + opcolor (6)
            payload.extend(encode_full_box(b"vmhd", 0, 1, &vmhd_payload));
        }
        TrackType::Audio => {
            // smhd: balance=0
            let smhd_payload = [0u8; 4]; // balance (2) + reserved (2)
            payload.extend(encode_full_box(b"smhd", 0, 0, &smhd_payload));
        }
        _ => {
            // nmhd (null media header) for other types
            payload.extend(encode_full_box(b"nmhd", 0, 0, &[]));
        }
    }

    payload.extend(build_dinf());
    payload.extend(build_stbl());
    encode_box(b"minf", &payload)
}

/// Build a minimal `dinf` (data information) box with a self-contained `url ` dref.
fn build_dinf() -> Vec<u8> {
    // dref: entry_count=1, url data entry (self-contained, flags=1)
    let mut dref_payload = Vec::new();
    dref_payload.extend_from_slice(&1u32.to_be_bytes()); // entry_count
                                                         // url entry: FullBox with flags=1 (self-contained), empty location
    let url_entry = encode_full_box(b"url ", 0, 1, &[]);
    dref_payload.extend(url_entry);

    let dref = encode_full_box(b"dref", 0, 0, &dref_payload);
    encode_box(b"dinf", &dref)
}

/// Build a minimal (empty) `stbl` (sample table) box.
fn build_stbl() -> Vec<u8> {
    let mut payload = Vec::new();
    // stsd: empty (entry_count = 0)
    let mut stsd_p = Vec::new();
    stsd_p.extend_from_slice(&0u32.to_be_bytes()); // entry_count
    payload.extend(encode_full_box(b"stsd", 0, 0, &stsd_p));
    // stts: zero entries
    let mut stts_p = Vec::new();
    stts_p.extend_from_slice(&0u32.to_be_bytes()); // entry_count
    payload.extend(encode_full_box(b"stts", 0, 0, &stts_p));
    // stsc: zero entries
    let mut stsc_p = Vec::new();
    stsc_p.extend_from_slice(&0u32.to_be_bytes()); // entry_count
    payload.extend(encode_full_box(b"stsc", 0, 0, &stsc_p));
    // stsz: sample_size=0, sample_count=0
    let mut stsz_p = Vec::new();
    stsz_p.extend_from_slice(&0u32.to_be_bytes()); // sample_size
    stsz_p.extend_from_slice(&0u32.to_be_bytes()); // sample_count
    payload.extend(encode_full_box(b"stsz", 0, 0, &stsz_p));
    // stco: zero chunk offsets
    let mut stco_p = Vec::new();
    stco_p.extend_from_slice(&0u32.to_be_bytes()); // entry_count
    payload.extend(encode_full_box(b"stco", 0, 0, &stco_p));
    encode_box(b"stbl", &payload)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    // ── Shared writer helper ──────────────────────────────────────────────
    //
    // `Box<dyn Write>` defaults to `Box<dyn Write + 'static>`.  We use a
    // reference-counted, mutex-guarded `Vec<u8>` so both the muxer (which owns
    // the `Box`) *and* the test body can access the written bytes after the
    // muxer is dropped.

    /// A `Write` implementor that forwards bytes into a shared `Vec<u8>`.
    #[derive(Clone)]
    struct SharedVec(Arc<Mutex<Vec<u8>>>);

    impl SharedVec {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Vec::new())))
        }

        fn take(&self) -> Vec<u8> {
            self.0.lock().expect("lock ok").clone()
        }
    }

    impl Write for SharedVec {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "lock poisoned"))?
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Run `f` with a fresh muxer and return the bytes it wrote.
    fn capture(f: impl FnOnce(&mut BasicMp4Muxer)) -> Vec<u8> {
        let sv = SharedVec::new();
        let mut muxer = BasicMp4Muxer::new(Box::new(sv.clone()));
        f(&mut muxer);
        sv.take()
    }

    fn read_u32_be(buf: &[u8], offset: usize) -> u32 {
        u32::from_be_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ])
    }

    fn find_box(buf: &[u8], box_type: &[u8; 4]) -> bool {
        buf.windows(4).any(|w| w == box_type)
    }

    // 1. write_ftyp produces a valid ftyp box
    #[test]
    fn test_write_ftyp_valid() {
        let out = capture(|m| {
            m.write_ftyp("isom").expect("should succeed");
        });
        // Total size = 8 (header) + 4 (major) + 4 (minor) + 4 (compat) = 20
        assert_eq!(out.len(), 20);
        assert_eq!(read_u32_be(&out, 0), 20);
        assert_eq!(&out[4..8], b"ftyp");
        assert_eq!(&out[8..12], b"isom"); // major brand
    }

    // 2. write_ftyp rejects invalid brand (short)
    #[test]
    fn test_write_ftyp_invalid_brand_short() {
        let sv = SharedVec::new();
        let mut muxer = BasicMp4Muxer::new(Box::new(sv));
        assert!(muxer.write_ftyp("iso").is_err());
    }

    // 3. write_ftyp rejects brand > 4 chars
    #[test]
    fn test_write_ftyp_invalid_brand_long() {
        let sv = SharedVec::new();
        let mut muxer = BasicMp4Muxer::new(Box::new(sv));
        assert!(muxer.write_ftyp("isom2").is_err());
    }

    // 4. write_mdat wraps data in an mdat box
    #[test]
    fn test_write_mdat_basic() {
        let payload: &[u8] = b"hello world";
        let out = capture(|m| {
            m.write_mdat(payload).expect("should succeed");
        });
        let expected_size = 8 + payload.len() as u32;
        assert_eq!(read_u32_be(&out, 0), expected_size);
        assert_eq!(&out[4..8], b"mdat");
        assert_eq!(&out[8..], payload);
    }

    // 5. write_mdat with empty data
    #[test]
    fn test_write_mdat_empty() {
        let out = capture(|m| {
            m.write_mdat(&[]).expect("should succeed");
        });
        assert_eq!(read_u32_be(&out, 0), 8);
        assert_eq!(&out[4..8], b"mdat");
    }

    // 5a. box_header_for_len: small payload uses the standard 8-byte header
    #[test]
    fn test_box_header_small_uses_32_bit_field() {
        let header = box_header_for_len(b"mdat", 100);
        assert_eq!(header.len(), 8);
        assert_eq!(&header[0..4], &108u32.to_be_bytes());
        assert_eq!(&header[4..8], b"mdat");
    }

    // 5b. box_header_for_len: exactly at the u32 boundary still uses the
    // 8-byte header (total size == u32::MAX must not tip into largesize).
    #[test]
    fn test_box_header_boundary_fits_in_u32() {
        let payload_len = u64::from(u32::MAX) - 8; // header(8) + payload == u32::MAX
        let header = box_header_for_len(b"mdat", payload_len);
        assert_eq!(
            header.len(),
            8,
            "boundary case must still use the 8-byte header"
        );
        assert_eq!(&header[0..4], &u32::MAX.to_be_bytes());
        assert_eq!(&header[4..8], b"mdat");
    }

    // 5c. box_header_for_len: one byte past the boundary must switch to the
    // ISO/IEC 14496-12 largesize form instead of silently truncating.
    #[test]
    fn test_box_header_over_4gib_uses_largesize() {
        let payload_len = u64::from(u32::MAX) - 7; // header(8) + payload == u32::MAX + 1
        let header = box_header_for_len(b"mdat", payload_len);
        assert_eq!(
            header.len(),
            16,
            "must switch to the 16-byte largesize header"
        );
        assert_eq!(
            &header[0..4],
            &1u32.to_be_bytes(),
            "size field must be the largesize sentinel value 1"
        );
        assert_eq!(&header[4..8], b"mdat");
        let largesize = u64::from_be_bytes(
            header[8..16]
                .try_into()
                .expect("largesize field is exactly 8 bytes"),
        );
        assert_eq!(largesize, 16 + payload_len, "largesize must be the TRUE total box size (16-byte header + payload), not the truncated/saturated 32-bit value");
    }

    // 5d. box_header_for_len: well over 4 GiB (~5 GB) also uses largesize
    // and reports the exact size, proving this isn't a boundary-only fix.
    #[test]
    fn test_box_header_well_over_4gib() {
        let payload_len = 5_000_000_000u64;
        let header = box_header_for_len(b"mdat", payload_len);
        assert_eq!(header.len(), 16);
        assert_eq!(&header[0..4], &1u32.to_be_bytes());
        assert_eq!(&header[4..8], b"mdat");
        let largesize = u64::from_be_bytes(
            header[8..16]
                .try_into()
                .expect("largesize field is exactly 8 bytes"),
        );
        assert_eq!(largesize, 16 + payload_len);
    }

    // 5e. write_mdat end-to-end still produces the correct standard header
    // for realistic (small) payloads after the box_header_for_len refactor.
    #[test]
    fn test_write_mdat_still_uses_standard_header_for_small_payload() {
        let payload = vec![0xABu8; 4096];
        let out = capture(|m| {
            m.write_mdat(&payload).expect("should succeed");
        });
        assert_eq!(read_u32_be(&out, 0), 8 + 4096);
        assert_eq!(&out[4..8], b"mdat");
        assert_eq!(&out[8..], payload.as_slice());
    }

    // 6. write_moov produces a moov box with mvhd
    #[test]
    fn test_write_moov_contains_mvhd() {
        let tracks = vec![TrackInfo::new(0, TrackType::Video, "av01")];
        let out = capture(|m| {
            m.write_moov(&tracks).expect("should succeed");
        });
        assert_eq!(&out[4..8], b"moov");
        assert!(find_box(&out, b"mvhd"));
    }

    // 7. write_moov with video track contains trak + vmhd + stbl
    #[test]
    fn test_write_moov_video_track() {
        let tracks = vec![TrackInfo::new(0, TrackType::Video, "av01")];
        let out = capture(|m| {
            m.write_moov(&tracks).expect("should succeed");
        });
        assert!(find_box(&out, b"trak"));
        assert!(find_box(&out, b"tkhd"));
        assert!(find_box(&out, b"mdia"));
        assert!(find_box(&out, b"hdlr"));
        assert!(find_box(&out, b"minf"));
        assert!(find_box(&out, b"vmhd"));
        assert!(find_box(&out, b"stbl"));
    }

    // 8. write_moov with audio track contains smhd
    #[test]
    fn test_write_moov_audio_track() {
        let tracks = vec![TrackInfo::new(0, TrackType::Audio, "Opus")];
        let out = capture(|m| {
            m.write_moov(&tracks).expect("should succeed");
        });
        assert!(find_box(&out, b"smhd"));
    }

    // 9. write_moov with no tracks still produces valid moov+mvhd
    #[test]
    fn test_write_moov_empty_tracks() {
        let out = capture(|m| {
            m.write_moov(&[]).expect("should succeed");
        });
        assert_eq!(&out[4..8], b"moov");
        assert!(find_box(&out, b"mvhd"));
    }

    // 10. moov box size field matches actual output length
    #[test]
    fn test_write_moov_size_consistent() {
        let tracks = vec![
            TrackInfo::new(0, TrackType::Video, "av01"),
            TrackInfo::new(1, TrackType::Audio, "Opus"),
        ];
        let out = capture(|m| {
            m.write_moov(&tracks).expect("should succeed");
        });
        let box_size = read_u32_be(&out, 0) as usize;
        assert_eq!(box_size, out.len());
    }

    // 11. full sequence: ftyp + mdat + moov
    #[test]
    fn test_full_sequence() {
        let tracks = vec![TrackInfo::new(0, TrackType::Video, "av01")];
        let media_data: &[u8] = b"fake encoded video";
        let out = capture(|m| {
            m.write_ftyp("isom").expect("ftyp ok");
            m.write_mdat(media_data).expect("mdat ok");
            m.write_moov(&tracks).expect("moov ok");
        });
        assert!(find_box(&out, b"ftyp"));
        assert!(find_box(&out, b"mdat"));
        assert!(find_box(&out, b"moov"));
    }

    // 12. ftyp with "av01" brand
    #[test]
    fn test_write_ftyp_av01_brand() {
        let out = capture(|m| {
            m.write_ftyp("av01").expect("should succeed");
        });
        assert_eq!(&out[8..12], b"av01");
    }

    // 13. dinf is present in moov
    #[test]
    fn test_write_moov_contains_dinf() {
        let tracks = vec![TrackInfo::new(0, TrackType::Video, "av01")];
        let out = capture(|m| {
            m.write_moov(&tracks).expect("should succeed");
        });
        assert!(find_box(&out, b"dinf"));
        assert!(find_box(&out, b"dref"));
    }

    // 14. subtitle track gets nmhd
    #[test]
    fn test_write_moov_subtitle_track() {
        let tracks = vec![TrackInfo::new(0, TrackType::Subtitle, "wvtt")];
        let out = capture(|m| {
            m.write_moov(&tracks).expect("should succeed");
        });
        assert!(find_box(&out, b"nmhd"));
    }

    // 15. multiple tracks each get their own trak
    #[test]
    fn test_write_moov_multiple_tracks_count() {
        let tracks = vec![
            TrackInfo::new(0, TrackType::Video, "av01"),
            TrackInfo::new(1, TrackType::Audio, "Opus"),
        ];
        let out = capture(|m| {
            m.write_moov(&tracks).expect("should succeed");
        });
        let trak_count = out.windows(4).filter(|w| *w == b"trak").count();
        assert_eq!(trak_count, 2);
    }
}
