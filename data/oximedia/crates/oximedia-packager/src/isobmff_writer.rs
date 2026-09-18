// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! ISOBMFF / CMAF fragmented MP4 writer.
//!
//! Provides low-level box-writing primitives and higher-level helpers that
//! assemble complete `ftyp`, `moov`, `moof`+`mdat` structures suitable for
//! CMAF or DASH fragmented-MP4 delivery.

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

/// Keyframe / sync-sample flag for use in `trun` sample flags.
pub const SAMPLE_FLAG_SYNC: u32 = 0x0200_0000;

/// Non-keyframe flag (depends on a prior sample).
pub const SAMPLE_FLAG_DEPENDS_ON: u32 = 0x0100_0000;

/// A single compressed sample (frame).
#[derive(Debug, Clone)]
pub struct Sample {
    /// Compressed payload bytes.
    pub data: Vec<u8>,
    /// Presentation duration in media timescale ticks.
    pub duration: u32,
    /// `trun` sample flags (use `SAMPLE_FLAG_SYNC` for key-frames).
    pub flags: u32,
    /// Composition time offset (PTS − DTS), in media timescale ticks.
    pub pts_offset: i32,
}

impl Sample {
    /// Construct a new sample.
    #[must_use]
    pub fn new(data: Vec<u8>, duration: u32, flags: u32, pts_offset: i32) -> Self {
        Self {
            data,
            duration,
            flags,
            pts_offset,
        }
    }

    /// Convenience: create a key-frame sample.
    #[must_use]
    pub fn keyframe(data: Vec<u8>, duration: u32) -> Self {
        Self::new(data, duration, SAMPLE_FLAG_SYNC, 0)
    }
}

// ---------------------------------------------------------------------------
// BoxWriter
// ---------------------------------------------------------------------------

/// Low-level helper for writing ISOBMFF boxes into a `Vec<u8>`.
///
/// Use `BoxWriter::write_box` to open a box, then write child content using
/// the `write_*` helpers exposed on `&mut BoxWriter`.  The size field is
/// back-patched once the closure returns.
pub struct BoxWriter {
    buf: Vec<u8>,
}

impl BoxWriter {
    /// Create a new, empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Write a big-endian `u8`.
    pub fn write_u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    /// Write a big-endian `u16`.
    pub fn write_u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Write a big-endian `u32`.
    pub fn write_u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Write a big-endian `u64`.
    pub fn write_u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    /// Write a little-endian `u16`.
    pub fn write_u16_le(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Write a little-endian `u32`.
    pub fn write_u32_le(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    /// Write raw bytes.
    pub fn write_bytes(&mut self, data: &[u8]) {
        self.buf.extend_from_slice(data);
    }

    /// Write a 4-byte fourcc.
    pub fn write_fourcc(&mut self, fourcc: &[u8; 4]) {
        self.buf.extend_from_slice(fourcc);
    }

    /// Append the internal buffer into an external `Vec<u8>` and return the
    /// number of bytes written.
    pub fn flush_into(&self, out: &mut Vec<u8>) -> usize {
        let n = self.buf.len();
        out.extend_from_slice(&self.buf);
        n
    }

    /// Consume the writer and return the buffer.
    #[must_use]
    pub fn into_vec(self) -> Vec<u8> {
        self.buf
    }

    /// Return the current buffer length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Return `true` when no bytes have been written yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    // ------------------------------------------------------------------
    // Box-framing helper
    // ------------------------------------------------------------------

    /// Write a complete ISOBMFF box into `out`.
    ///
    /// The 4-byte size placeholder is appended to `out` first; then `f` is
    /// called with a fresh `BoxWriter`; the child bytes are appended; finally
    /// the size field is back-patched with the real total (header + content).
    pub fn write_box<F>(out: &mut Vec<u8>, fourcc: &[u8; 4], f: F)
    where
        F: FnOnce(&mut BoxWriter),
    {
        // Record start offset so we can back-patch the size.
        let start = out.len();

        // Placeholder for box size (4 bytes).
        out.extend_from_slice(&0u32.to_be_bytes());
        // fourcc (4 bytes).
        out.extend_from_slice(fourcc);

        // Write box contents.
        let mut child = BoxWriter::new();
        f(&mut child);
        child.flush_into(out);

        // Back-patch size.
        let total = out.len() - start;
        out[start..start + 4].copy_from_slice(&(total as u32).to_be_bytes());
    }

    // ------------------------------------------------------------------
    // Convenience: full-box header (version + flags)
    // ------------------------------------------------------------------

    /// Write a FullBox version + flags header (4 bytes).
    pub fn write_full_box_header(out: &mut BoxWriter, version: u8, flags: u32) {
        out.write_u8(version);
        // flags: 3 bytes
        out.write_u8(((flags >> 16) & 0xFF) as u8);
        out.write_u8(((flags >> 8) & 0xFF) as u8);
        out.write_u8((flags & 0xFF) as u8);
    }
}

impl Default for BoxWriter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ftyp box
// ---------------------------------------------------------------------------

/// Build a `ftyp` (File Type) box.
///
/// `brands` should include the major brand (first element) and any compatible
/// brands.  `minor` is the minor version fourcc (typically all-zeros).
#[must_use]
pub fn write_ftyp(brands: &[&[u8; 4]], minor: &[u8; 4]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    BoxWriter::write_box(&mut out, b"ftyp", |w| {
        // major_brand: first entry
        if let Some(major) = brands.first() {
            w.write_fourcc(major);
        } else {
            w.write_bytes(&[0u8; 4]);
        }
        // minor_version: the supplied minor bytes
        w.write_bytes(minor);
        // compatible_brands: all brands (including major)
        for brand in brands {
            w.write_fourcc(brand);
        }
    });

    out
}

// ---------------------------------------------------------------------------
// moov init segment
// ---------------------------------------------------------------------------

/// Build a `moov` initialisation segment for a single video track.
///
/// Produces:
/// ```text
/// moov
///   mvhd
///   trak
///     tkhd
///     mdia
///       mdhd
///       hdlr  (vide)
///       minf
///         vmhd
///         dinf  (dref url)
///         stbl
///           stsd (avc1|hvc1|av01 + ...)
///           stts (empty)
///           stsc (empty)
///           stsz (empty)
///           stco (empty)
/// ```
#[must_use]
pub fn write_moov_init(
    video_width: u32,
    video_height: u32,
    timescale: u32,
    codec_fourcc: &[u8; 4],
    extra_data: &[u8],
) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    BoxWriter::write_box(&mut out, b"moov", |moov| {
        // mvhd (version 0)
        BoxWriter::write_box(&mut moov.buf, b"mvhd", |w| {
            BoxWriter::write_full_box_header(w, 0, 0);
            w.write_u32(0); // creation_time
            w.write_u32(0); // modification_time
            w.write_u32(timescale);
            w.write_u32(0); // duration (unknown at init)
            w.write_u32(0x0001_0000); // rate 1.0
            w.write_u16(0x0100); // volume 1.0
            w.write_bytes(&[0u8; 10]); // reserved
                                       // unity matrix
            w.write_u32(0x0001_0000);
            w.write_u32(0);
            w.write_u32(0);
            w.write_u32(0);
            w.write_u32(0x0001_0000);
            w.write_u32(0);
            w.write_u32(0);
            w.write_u32(0);
            w.write_u32(0x4000_0000);
            w.write_bytes(&[0u8; 24]); // pre_defined
            w.write_u32(2); // next_track_id
        });

        // trak
        BoxWriter::write_box(&mut moov.buf, b"trak", |trak| {
            // tkhd (version 0, flags = enabled|in_movie|in_preview = 0x0f)
            BoxWriter::write_box(&mut trak.buf, b"tkhd", |w| {
                BoxWriter::write_full_box_header(w, 0, 0x0f);
                w.write_u32(0); // creation_time
                w.write_u32(0); // modification_time
                w.write_u32(1); // track_id
                w.write_u32(0); // reserved
                w.write_u32(0); // duration
                w.write_bytes(&[0u8; 8]); // reserved
                w.write_u16(0); // layer
                w.write_u16(0); // alternate_group
                w.write_u16(0); // volume (video = 0)
                w.write_u16(0); // reserved
                                // unity matrix
                w.write_u32(0x0001_0000);
                w.write_u32(0);
                w.write_u32(0);
                w.write_u32(0);
                w.write_u32(0x0001_0000);
                w.write_u32(0);
                w.write_u32(0);
                w.write_u32(0);
                w.write_u32(0x4000_0000);
                // width / height as 16.16 fixed-point
                w.write_u32(video_width << 16);
                w.write_u32(video_height << 16);
            });

            // mdia
            BoxWriter::write_box(&mut trak.buf, b"mdia", |mdia| {
                // mdhd (version 0)
                BoxWriter::write_box(&mut mdia.buf, b"mdhd", |w| {
                    BoxWriter::write_full_box_header(w, 0, 0);
                    w.write_u32(0); // creation_time
                    w.write_u32(0); // modification_time
                    w.write_u32(timescale);
                    w.write_u32(0); // duration
                                    // language = "und" (0x55C4 in packed ISO 639-2/T)
                    w.write_u16(0x55C4);
                    w.write_u16(0); // pre_defined
                });

                // hdlr (vide)
                BoxWriter::write_box(&mut mdia.buf, b"hdlr", |w| {
                    BoxWriter::write_full_box_header(w, 0, 0);
                    w.write_u32(0); // pre_defined
                    w.write_bytes(b"vide"); // handler_type
                    w.write_bytes(&[0u8; 12]); // reserved (3×u32)
                    w.write_bytes(b"Video Track\0"); // name
                });

                // minf
                BoxWriter::write_box(&mut mdia.buf, b"minf", |minf| {
                    // vmhd
                    BoxWriter::write_box(&mut minf.buf, b"vmhd", |w| {
                        BoxWriter::write_full_box_header(w, 0, 1);
                        w.write_u16(0); // graphicsMode
                        w.write_bytes(&[0u8; 6]); // opcolor
                    });

                    // dinf / dref / url
                    BoxWriter::write_box(&mut minf.buf, b"dinf", |dinf| {
                        BoxWriter::write_box(&mut dinf.buf, b"dref", |dref| {
                            BoxWriter::write_full_box_header(dref, 0, 0);
                            dref.write_u32(1); // entry_count = 1
                            BoxWriter::write_box(&mut dref.buf, b"url ", |w| {
                                // flags = 0x000001 means data is in same file
                                BoxWriter::write_full_box_header(w, 0, 1);
                            });
                        });
                    });

                    // stbl
                    BoxWriter::write_box(&mut minf.buf, b"stbl", |stbl| {
                        // stsd
                        BoxWriter::write_box(&mut stbl.buf, b"stsd", |stsd_w| {
                            BoxWriter::write_full_box_header(stsd_w, 0, 0);
                            stsd_w.write_u32(1); // entry_count

                            // Video sample entry (avc1 / hvc1 / av01 / vp09 …)
                            BoxWriter::write_box(&mut stsd_w.buf, codec_fourcc, |ve| {
                                ve.write_bytes(&[0u8; 6]); // reserved
                                ve.write_u16(1); // data_reference_index
                                ve.write_bytes(&[0u8; 16]); // pre_defined / reserved
                                ve.write_u16(video_width as u16);
                                ve.write_u16(video_height as u16);
                                ve.write_u32(0x0048_0000); // horiz resolution 72dpi
                                ve.write_u32(0x0048_0000); // vert resolution 72dpi
                                ve.write_u32(0); // reserved
                                ve.write_u16(1); // frame_count
                                ve.write_bytes(&[0u8; 32]); // compressorname
                                ve.write_u16(0x0018); // depth = 24
                                ve.write_u16(0xFFFF_u16); // pre_defined = -1
                                                          // codec-specific extra data (avcC / hvcC / av1C / vpcC)
                                if !extra_data.is_empty() {
                                    ve.write_bytes(extra_data);
                                }
                            });
                        });

                        // stts: empty time-to-sample table
                        BoxWriter::write_box(&mut stbl.buf, b"stts", |w| {
                            BoxWriter::write_full_box_header(w, 0, 0);
                            w.write_u32(0); // entry_count
                        });

                        // stsc: empty sample-to-chunk table
                        BoxWriter::write_box(&mut stbl.buf, b"stsc", |w| {
                            BoxWriter::write_full_box_header(w, 0, 0);
                            w.write_u32(0); // entry_count
                        });

                        // stsz: empty sample size table
                        BoxWriter::write_box(&mut stbl.buf, b"stsz", |w| {
                            BoxWriter::write_full_box_header(w, 0, 0);
                            w.write_u32(0); // sample_size (0 = variable)
                            w.write_u32(0); // sample_count
                        });

                        // stco: empty chunk offset table
                        BoxWriter::write_box(&mut stbl.buf, b"stco", |w| {
                            BoxWriter::write_full_box_header(w, 0, 0);
                            w.write_u32(0); // entry_count
                        });
                    });
                });
            });
        });

        // mvex with trex
        BoxWriter::write_box(&mut moov.buf, b"mvex", |mvex| {
            BoxWriter::write_box(&mut mvex.buf, b"trex", |w| {
                BoxWriter::write_full_box_header(w, 0, 0);
                w.write_u32(1); // track_id
                w.write_u32(1); // default_sample_description_index
                w.write_u32(0); // default_sample_duration
                w.write_u32(0); // default_sample_size
                w.write_u32(SAMPLE_FLAG_SYNC); // default_sample_flags
            });
        });
    });

    out
}

// ---------------------------------------------------------------------------
// moof + mdat segment
// ---------------------------------------------------------------------------

/// Build a `moof` + `mdat` pair for a sequence of compressed samples.
///
/// The `moof` contains:
/// - `mfhd` (Movie Fragment Header): sequence_number
/// - `traf` (Track Fragment):
///   - `tfhd` (Track Fragment Header): track_id, base data offset flag
///   - `tfdt` (Track Fragment Decode Time): base_media_decode_time
///   - `trun` (Track Run): one row per sample with size + duration + flags + cts_offset
#[must_use]
pub fn write_moof_mdat(
    sequence_number: u32,
    base_media_decode_time: u64,
    samples: &[Sample],
) -> Vec<u8> {
    // We need to know the moof size before we can fill in the data_offset
    // inside trun (which points to the start of mdat payload relative to the
    // start of moof).  We therefore build moof twice: first with a dummy
    // data_offset, measure its size, then rebuild with the real value.

    let moof_size = build_moof_inner(sequence_number, base_media_decode_time, samples, 0).len();
    // mdat header = 8 bytes (size + fourcc), payload starts right after
    let data_offset = (moof_size + 8) as i32;

    let moof = build_moof_inner(
        sequence_number,
        base_media_decode_time,
        samples,
        data_offset,
    );

    // mdat
    let mdat_payload_len: usize = samples.iter().map(|s| s.data.len()).sum();
    let mdat_total = 8 + mdat_payload_len;

    let mut out: Vec<u8> = Vec::with_capacity(moof.len() + mdat_total);
    out.extend_from_slice(&moof);

    // mdat box
    out.extend_from_slice(&(mdat_total as u32).to_be_bytes());
    out.extend_from_slice(b"mdat");
    for s in samples {
        out.extend_from_slice(&s.data);
    }

    out
}

/// Internal: build a moof box with the given data_offset.
fn build_moof_inner(
    sequence_number: u32,
    base_media_decode_time: u64,
    samples: &[Sample],
    data_offset: i32,
) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    BoxWriter::write_box(&mut out, b"moof", |moof| {
        // mfhd
        BoxWriter::write_box(&mut moof.buf, b"mfhd", |w| {
            BoxWriter::write_full_box_header(w, 0, 0);
            w.write_u32(sequence_number);
        });

        // traf
        BoxWriter::write_box(&mut moof.buf, b"traf", |traf| {
            // tfhd: flags 0x020000 = base-data-offset-present (we will put
            // the moof start offset), plus 0x000020 = default-base-is-moof
            // For simplicity use flags=0x020000 (base_data_offset present).
            BoxWriter::write_box(&mut traf.buf, b"tfhd", |w| {
                // flags: 0x020000 = base-data-offset-present
                BoxWriter::write_full_box_header(w, 0, 0x02_0000);
                w.write_u32(1); // track_id
                                // base_data_offset = 0 (start of moof box; will be corrected
                                // by the player using the sequence-level moof offset)
                w.write_u64(0);
            });

            // tfdt (version 1 for 64-bit decode time)
            BoxWriter::write_box(&mut traf.buf, b"tfdt", |w| {
                BoxWriter::write_full_box_header(w, 1, 0);
                w.write_u64(base_media_decode_time);
            });

            // trun: flags
            //   0x000001 = data-offset-present
            //   0x000100 = sample-duration-present
            //   0x000200 = sample-size-present
            //   0x000400 = sample-flags-present
            //   0x000800 = sample-composition-time-offsets-present
            let trun_flags: u32 = 0x0000_0F01;
            BoxWriter::write_box(&mut traf.buf, b"trun", |w| {
                BoxWriter::write_full_box_header(w, 1, trun_flags);
                w.write_u32(samples.len() as u32); // sample_count
                w.write_u32(data_offset as u32); // data_offset (signed; cast is intentional for BE layout)
                for s in samples {
                    w.write_u32(s.duration);
                    w.write_u32(s.data.len() as u32);
                    w.write_u32(s.flags);
                    // composition time offset as i32 big-endian
                    w.write_u32(s.pts_offset as u32);
                }
            });
        });
    });

    out
}

// ---------------------------------------------------------------------------
// InitConfig — high-level init segment builder
// ---------------------------------------------------------------------------

/// Configuration for generating a complete init segment (`ftyp` + `moov`).
#[derive(Debug, Clone)]
pub struct InitConfig {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Timescale (ticks per second).
    pub timescale: u32,
    /// Codec fourcc (e.g. `b"av01"`, `b"vp09"`).
    pub codec_fourcc: [u8; 4],
    /// Optional SPS (Sequence Parameter Set) data for AVC/HEVC codecs.
    pub sps_data: Vec<u8>,
    /// Optional PPS (Picture Parameter Set) data for AVC/HEVC codecs.
    pub pps_data: Vec<u8>,
}

impl InitConfig {
    /// Create a new init configuration.
    #[must_use]
    pub fn new(width: u32, height: u32, timescale: u32, codec_fourcc: [u8; 4]) -> Self {
        Self {
            width,
            height,
            timescale,
            codec_fourcc,
            sps_data: Vec::new(),
            pps_data: Vec::new(),
        }
    }

    /// Set SPS data.
    #[must_use]
    pub fn with_sps(mut self, sps: Vec<u8>) -> Self {
        self.sps_data = sps;
        self
    }

    /// Set PPS data.
    #[must_use]
    pub fn with_pps(mut self, pps: Vec<u8>) -> Self {
        self.pps_data = pps;
        self
    }
}

/// Build a complete init segment (`ftyp` + `moov`) from an [`InitConfig`].
///
/// The `ftyp` box uses CMAF-compatible brands (`cmfc`, `iso6`, `dash`).
#[must_use]
pub fn write_init_segment(config: &InitConfig) -> Vec<u8> {
    let ftyp = write_ftyp(&[b"cmfc", b"iso6", b"dash"], &[0u8; 4]);

    // Merge SPS + PPS into extra_data
    let mut extra_data = Vec::new();
    if !config.sps_data.is_empty() || !config.pps_data.is_empty() {
        extra_data.extend_from_slice(&config.sps_data);
        extra_data.extend_from_slice(&config.pps_data);
    }

    let moov = write_moov_init(
        config.width,
        config.height,
        config.timescale,
        &config.codec_fourcc,
        &extra_data,
    );

    let mut out = Vec::with_capacity(ftyp.len() + moov.len());
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    out
}

// ---------------------------------------------------------------------------
// MediaSample — alias with is_sync convenience
// ---------------------------------------------------------------------------

/// A media sample with explicit sync-frame indicator.
///
/// This is a higher-level wrapper around [`Sample`] that uses a boolean
/// `is_sync` field instead of raw flag bits.
#[derive(Debug, Clone)]
pub struct MediaSample {
    /// Compressed payload bytes.
    pub data: Vec<u8>,
    /// Presentation duration in timescale ticks.
    pub duration: u32,
    /// Raw sample flags (use 0 and set `is_sync` instead for convenience).
    pub flags: u32,
    /// Composition time offset (PTS - DTS).
    pub composition_offset: i32,
    /// Whether this sample is a sync (key) frame.
    pub is_sync: bool,
}

impl MediaSample {
    /// Construct a new media sample.
    #[must_use]
    pub fn new(data: Vec<u8>, duration: u32, is_sync: bool) -> Self {
        Self {
            data,
            duration,
            flags: if is_sync {
                SAMPLE_FLAG_SYNC
            } else {
                SAMPLE_FLAG_DEPENDS_ON
            },
            composition_offset: 0,
            is_sync,
        }
    }

    /// Set the composition time offset.
    #[must_use]
    pub fn with_composition_offset(mut self, offset: i32) -> Self {
        self.composition_offset = offset;
        self
    }

    /// Convert to a low-level [`Sample`].
    #[must_use]
    pub fn to_sample(&self) -> Sample {
        let flags = if self.is_sync {
            SAMPLE_FLAG_SYNC
        } else {
            self.flags | SAMPLE_FLAG_DEPENDS_ON
        };
        Sample {
            data: self.data.clone(),
            duration: self.duration,
            flags,
            pts_offset: self.composition_offset,
        }
    }
}

/// Build a `moof` + `mdat` pair from [`MediaSample`] objects.
///
/// This is a convenience wrapper around [`write_moof_mdat`] that converts
/// `MediaSample` to `Sample` automatically.
#[must_use]
pub fn write_media_segment(
    sequence_number: u32,
    base_media_decode_time: u64,
    samples: &[MediaSample],
) -> Vec<u8> {
    let low_level: Vec<Sample> = samples.iter().map(|ms| ms.to_sample()).collect();
    write_moof_mdat(sequence_number, base_media_decode_time, &low_level)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- BoxWriter basics ----------------------------------------------------

    #[test]
    fn test_box_writer_empty() {
        let w = BoxWriter::new();
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn test_box_writer_write_u8() {
        let mut w = BoxWriter::new();
        w.write_u8(0xAB);
        assert_eq!(w.into_vec(), vec![0xAB]);
    }

    #[test]
    fn test_box_writer_write_u16_big_endian() {
        let mut w = BoxWriter::new();
        w.write_u16(0x1234);
        assert_eq!(w.into_vec(), vec![0x12, 0x34]);
    }

    #[test]
    fn test_box_writer_write_u32_big_endian() {
        let mut w = BoxWriter::new();
        w.write_u32(0xDEAD_BEEF);
        assert_eq!(w.into_vec(), vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_box_writer_write_u64_big_endian() {
        let mut w = BoxWriter::new();
        w.write_u64(0x0102_0304_0506_0708);
        assert_eq!(
            w.into_vec(),
            vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
    }

    #[test]
    fn test_box_writer_write_box_size_patched() {
        let mut out: Vec<u8> = Vec::new();
        BoxWriter::write_box(&mut out, b"ftyp", |w| {
            w.write_bytes(b"cmfc"); // 4 bytes of content
        });
        // Total: 4 (size) + 4 (fourcc) + 4 (content) = 12
        let size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(size, out.len());
        assert_eq!(size, 12);
    }

    #[test]
    fn test_box_writer_write_box_fourcc_present() {
        let mut out: Vec<u8> = Vec::new();
        BoxWriter::write_box(&mut out, b"moof", |_| {});
        assert_eq!(&out[4..8], b"moof");
    }

    // --- ftyp ----------------------------------------------------------------

    #[test]
    fn test_write_ftyp_starts_with_size() {
        let out = write_ftyp(&[b"cmf1", b"iso6", b"dash"], &[0u8; 4]);
        let size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(size, out.len());
    }

    #[test]
    fn test_write_ftyp_fourcc() {
        let out = write_ftyp(&[b"cmf1"], &[0u8; 4]);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn test_write_ftyp_major_brand_is_first() {
        let out = write_ftyp(&[b"cmf1", b"iso6"], &[0u8; 4]);
        // major_brand starts at byte 8
        assert_eq!(&out[8..12], b"cmf1");
    }

    #[test]
    fn test_write_ftyp_compatible_brands_present() {
        let out = write_ftyp(&[b"cmf1", b"iso6", b"dash"], &[0u8; 4]);
        // Compatible brands start at byte 16 (8 header + 4 major + 4 minor)
        let brands_section = &out[16..];
        let brands: Vec<&[u8]> = brands_section.chunks(4).collect();
        assert!(brands.contains(&b"cmf1".as_ref()));
        assert!(brands.contains(&b"iso6".as_ref()));
        assert!(brands.contains(&b"dash".as_ref()));
    }

    // --- moov init -----------------------------------------------------------

    #[test]
    fn test_write_moov_init_starts_with_valid_box() {
        let data = write_moov_init(1920, 1080, 90_000, b"av01", &[]);
        let size = u32::from_be_bytes(data[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(size, data.len());
        assert_eq!(&data[4..8], b"moov");
    }

    #[test]
    fn test_write_moov_init_contains_mvhd() {
        let data = write_moov_init(1280, 720, 90_000, b"vp09", &[]);
        let found = find_box(&data, b"mvhd");
        assert!(found.is_some(), "mvhd not found");
    }

    #[test]
    fn test_write_moov_init_contains_trak() {
        let data = write_moov_init(1280, 720, 90_000, b"vp09", &[]);
        let found = find_box(&data, b"trak");
        assert!(found.is_some(), "trak not found");
    }

    #[test]
    fn test_write_moov_init_contains_mvex() {
        let data = write_moov_init(640, 480, 12_800, b"av01", &[]);
        let found = find_box(&data, b"mvex");
        assert!(found.is_some(), "mvex not found");
    }

    #[test]
    fn test_write_moov_init_with_extra_data() {
        // avcC-style extra data (minimal stub)
        let extra = vec![0x01, 0x64, 0x00, 0x1F, 0xFF];
        let data = write_moov_init(1920, 1080, 90_000, b"avc1", &extra);
        // Just assert it doesn't panic and produces a valid-length box
        let size = u32::from_be_bytes(data[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(size, data.len());
    }

    // --- moof + mdat ---------------------------------------------------------

    #[test]
    fn test_write_moof_mdat_structure() {
        let samples = vec![
            Sample::keyframe(vec![0x00, 0x01, 0x02, 0x03], 3_000),
            Sample::new(vec![0xFF, 0xFE], 3_000, SAMPLE_FLAG_DEPENDS_ON, 0),
        ];
        let out = write_moof_mdat(1, 0, &samples);
        // First box should be moof
        assert_eq!(&out[4..8], b"moof");
        let moof_size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        // Second box should be mdat
        assert_eq!(&out[moof_size + 4..moof_size + 8], b"mdat");
    }

    #[test]
    fn test_write_moof_mdat_mdat_payload() {
        let payload = vec![0xAA, 0xBB, 0xCC, 0xDD];
        let samples = vec![Sample::keyframe(payload.clone(), 3_000)];
        let out = write_moof_mdat(1, 0, &samples);
        let moof_size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        let mdat_offset = moof_size + 8; // skip mdat header
        assert_eq!(&out[mdat_offset..mdat_offset + 4], &payload[..]);
    }

    #[test]
    fn test_write_moof_contains_mfhd() {
        let samples = vec![Sample::keyframe(vec![1, 2, 3], 3_000)];
        let out = write_moof_mdat(42, 0, &samples);
        let found = find_box(&out, b"mfhd");
        assert!(found.is_some(), "mfhd not found in moof");
    }

    #[test]
    fn test_write_moof_contains_trun() {
        let samples = vec![Sample::keyframe(vec![1, 2, 3], 3_000)];
        let out = write_moof_mdat(1, 0, &samples);
        let found = find_box(&out, b"trun");
        assert!(found.is_some(), "trun not found");
    }

    #[test]
    fn test_write_moof_sequence_number() {
        let samples = vec![Sample::keyframe(vec![0u8], 3_000)];
        let out = write_moof_mdat(7, 0, &samples);
        // mfhd content: 4-byte fullbox header + 4-byte sequence_number
        if let Some(mfhd_offset) = find_box_offset(&out, b"mfhd") {
            let seq_offset = mfhd_offset + 8 + 4; // box header + fullbox header
            let seq =
                u32::from_be_bytes(out[seq_offset..seq_offset + 4].try_into().expect("4 bytes"));
            assert_eq!(seq, 7);
        } else {
            panic!("mfhd not found");
        }
    }

    #[test]
    fn test_write_moof_empty_samples() {
        // Should not panic
        let out = write_moof_mdat(1, 0, &[]);
        assert_eq!(&out[4..8], b"moof");
    }

    #[test]
    fn test_sample_flag_sync_value() {
        assert_eq!(SAMPLE_FLAG_SYNC, 0x0200_0000);
    }

    #[test]
    fn test_sample_keyframe_constructor() {
        let s = Sample::keyframe(vec![1, 2], 1_000);
        assert_eq!(s.flags, SAMPLE_FLAG_SYNC);
        assert_eq!(s.duration, 1_000);
        assert_eq!(s.pts_offset, 0);
    }

    // --- InitConfig / write_init_segment ------------------------------------

    #[test]
    fn test_init_config_new() {
        let cfg = InitConfig::new(1920, 1080, 90_000, *b"av01");
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert_eq!(cfg.timescale, 90_000);
        assert_eq!(&cfg.codec_fourcc, b"av01");
        assert!(cfg.sps_data.is_empty());
        assert!(cfg.pps_data.is_empty());
    }

    #[test]
    fn test_init_config_with_sps_pps() {
        let cfg = InitConfig::new(1280, 720, 48_000, *b"vp09")
            .with_sps(vec![0x67, 0x42, 0x00])
            .with_pps(vec![0x68, 0xCE]);
        assert_eq!(cfg.sps_data, vec![0x67, 0x42, 0x00]);
        assert_eq!(cfg.pps_data, vec![0x68, 0xCE]);
    }

    #[test]
    fn test_write_init_segment_starts_with_ftyp() {
        let cfg = InitConfig::new(1920, 1080, 90_000, *b"av01");
        let out = write_init_segment(&cfg);
        assert_eq!(&out[4..8], b"ftyp");
    }

    #[test]
    fn test_write_init_segment_contains_moov() {
        let cfg = InitConfig::new(1920, 1080, 90_000, *b"av01");
        let out = write_init_segment(&cfg);
        // Find moov after ftyp
        let ftyp_size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(&out[ftyp_size + 4..ftyp_size + 8], b"moov");
    }

    #[test]
    fn test_write_init_segment_with_extra_data() {
        let cfg = InitConfig::new(1920, 1080, 90_000, *b"avc1")
            .with_sps(vec![0x67, 0x42, 0x00, 0x1F])
            .with_pps(vec![0x68, 0xCE, 0x38, 0x80]);
        let out = write_init_segment(&cfg);
        // Should produce valid boxes
        let ftyp_size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        let moov_size =
            u32::from_be_bytes(out[ftyp_size..ftyp_size + 4].try_into().expect("4 bytes")) as usize;
        assert_eq!(ftyp_size + moov_size, out.len());
    }

    // --- MediaSample / write_media_segment ----------------------------------

    #[test]
    fn test_media_sample_new_sync() {
        let ms = MediaSample::new(vec![0xAA, 0xBB], 3_000, true);
        assert!(ms.is_sync);
        assert_eq!(ms.flags, SAMPLE_FLAG_SYNC);
        assert_eq!(ms.composition_offset, 0);
    }

    #[test]
    fn test_media_sample_new_non_sync() {
        let ms = MediaSample::new(vec![0xCC], 3_000, false);
        assert!(!ms.is_sync);
        assert_eq!(ms.flags, SAMPLE_FLAG_DEPENDS_ON);
    }

    #[test]
    fn test_media_sample_with_composition_offset() {
        let ms = MediaSample::new(vec![0xDD], 3_000, true).with_composition_offset(1_500);
        assert_eq!(ms.composition_offset, 1_500);
    }

    #[test]
    fn test_media_sample_to_sample() {
        let ms = MediaSample::new(vec![0xEE], 3_000, true);
        let s = ms.to_sample();
        assert_eq!(s.flags, SAMPLE_FLAG_SYNC);
        assert_eq!(s.duration, 3_000);
        assert_eq!(s.data, vec![0xEE]);
    }

    #[test]
    fn test_write_media_segment_produces_moof_mdat() {
        let samples = vec![
            MediaSample::new(vec![0x01, 0x02, 0x03], 3_000, true),
            MediaSample::new(vec![0x04, 0x05], 3_000, false),
        ];
        let out = write_media_segment(1, 0, &samples);
        assert_eq!(&out[4..8], b"moof");
        let moof_size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        assert_eq!(&out[moof_size + 4..moof_size + 8], b"mdat");
    }

    #[test]
    fn test_write_media_segment_mdat_payload_correct() {
        let samples = vec![MediaSample::new(vec![0xAA, 0xBB], 3_000, true)];
        let out = write_media_segment(1, 0, &samples);
        let moof_size = u32::from_be_bytes(out[0..4].try_into().expect("4 bytes")) as usize;
        let mdat_payload_start = moof_size + 8;
        assert_eq!(
            &out[mdat_payload_start..mdat_payload_start + 2],
            &[0xAA, 0xBB]
        );
    }

    // -------------------------------------------------------------------------
    // Helper: naive recursive box search (breadth-first over raw bytes)
    // -------------------------------------------------------------------------

    fn find_box<'a>(data: &'a [u8], fourcc: &[u8; 4]) -> Option<&'a [u8]> {
        find_box_offset(data, fourcc).map(|off| &data[off..])
    }

    fn find_box_offset(data: &[u8], fourcc: &[u8; 4]) -> Option<usize> {
        let mut i = 0;
        while i + 8 <= data.len() {
            let size = u32::from_be_bytes(data[i..i + 4].try_into().ok()?) as usize;
            if size < 8 {
                break;
            }
            if &data[i + 4..i + 8] == fourcc {
                return Some(i);
            }
            // Search inside this box
            if i + size <= data.len() {
                if let Some(inner) = find_box_offset(&data[i + 8..i + size], fourcc) {
                    return Some(i + 8 + inner);
                }
            }
            i += size;
        }
        None
    }
}
