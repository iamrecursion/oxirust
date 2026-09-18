//! WAV file reader and writer with full RIFF chunk handling.
//!
//! This module implements a pure-Rust WAV codec that handles:
//!
//! * Standard PCM (`fmt ` chunk type 1)
//! * IEEE float PCM (type 3)
//! * WAVE_FORMAT_EXTENSIBLE (type 0xFFFE) for multi-channel / high-resolution audio
//! * Optional `LIST INFO` metadata chunk
//! * Round-trip correctness for 8-, 16-, 24-, 32-bit integer and 32-/64-bit float
//!
//! # Examples
//!
//! ```rust
//! use oximedia_audio::wav::{WavReader, WavWriter, WavSpec};
//! use std::io::Cursor;
//!
//! // Write a 440 Hz sine wave to a buffer
//! let spec = WavSpec { channels: 1, sample_rate: 44100, bits_per_sample: 16, float: false };
//! let mut buf = Vec::new();
//! {
//!     let mut writer = WavWriter::new(Cursor::new(&mut buf), spec);
//!     for i in 0..4410_u32 {
//!         let sample = (std::f32::consts::TAU * 440.0 * i as f32 / 44100.0).sin();
//!         writer.write_sample_f32(sample).unwrap();
//!     }
//!     writer.finalize().unwrap();
//! }
//! // Read it back
//! let reader = WavReader::new(Cursor::new(&buf)).unwrap();
//! assert_eq!(reader.spec().sample_rate, 44100);
//! assert_eq!(reader.spec().channels, 1);
//! ```

#![allow(dead_code)]

use std::io::{Read, Seek, SeekFrom, Write};

use crate::error::{AudioError, AudioResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// WAV audio specification (from the `fmt ` chunk).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavSpec {
    /// Number of channels (1 = mono, 2 = stereo, …).
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Bits per sample (8, 16, 24, 32, 64).
    pub bits_per_sample: u16,
    /// `true` when samples are IEEE 754 floating point (format tag 3 / extensible float).
    pub float: bool,
}

impl WavSpec {
    /// Bytes per sample frame (all channels).
    #[must_use]
    pub fn block_align(&self) -> u16 {
        (u32::from(self.bits_per_sample / 8) * u32::from(self.channels)) as u16
    }

    /// Byte rate (bytes per second of audio).
    #[must_use]
    pub fn byte_rate(&self) -> u32 {
        u32::from(self.block_align()) * self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// RIFF chunk helpers
// ---------------------------------------------------------------------------

/// A raw RIFF chunk as parsed from a WAV file.
#[derive(Debug, Clone)]
pub struct RiffChunk {
    /// Four-character chunk identifier.
    pub id: [u8; 4],
    /// Chunk payload (not including the 8-byte header).
    pub data: Vec<u8>,
}

impl RiffChunk {
    /// Chunk identifier as a UTF-8 string slice (best-effort).
    #[must_use]
    pub fn id_str(&self) -> &str {
        std::str::from_utf8(&self.id).unwrap_or("????")
    }
}

// ---------------------------------------------------------------------------
// WavReader
// ---------------------------------------------------------------------------

/// Streaming WAV reader.
///
/// Parses the RIFF/WAVE header on construction and exposes typed sample
/// iterators. The reader is `Send` when the underlying `R` is `Send`.
pub struct WavReader<R: Read + Seek> {
    inner: R,
    spec: WavSpec,
    /// Position of the first audio byte inside the file.
    data_start: u64,
    /// Number of bytes in the `data` chunk.
    data_len: u32,
    /// Current read position (bytes consumed from data chunk).
    data_pos: u32,
    /// All non-data, non-fmt chunks encountered during header parsing.
    extra_chunks: Vec<RiffChunk>,
}

impl<R: Read + Seek> WavReader<R> {
    /// Open a WAV stream and parse its header.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is not a valid RIFF/WAVE file, or if
    /// the `fmt ` chunk is missing or malformed.
    pub fn new(mut inner: R) -> AudioResult<Self> {
        // RIFF header (12 bytes)
        let mut riff = [0u8; 12];
        inner
            .read_exact(&mut riff)
            .map_err(|e| AudioError::Io(e.to_string()))?;

        if &riff[0..4] != b"RIFF" {
            return Err(AudioError::InvalidData("Not a RIFF file".into()));
        }
        if &riff[8..12] != b"WAVE" {
            return Err(AudioError::InvalidData("Not a WAVE file".into()));
        }
        let _riff_size = u32::from_le_bytes([riff[4], riff[5], riff[6], riff[7]]);

        let mut spec: Option<WavSpec> = None;
        let mut data_start: Option<u64> = None;
        let mut data_len: u32 = 0;
        let mut extra_chunks = Vec::new();

        loop {
            let mut chunk_hdr = [0u8; 8];
            match inner.read_exact(&mut chunk_hdr) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(AudioError::Io(e.to_string())),
            }
            let chunk_id: [u8; 4] = [chunk_hdr[0], chunk_hdr[1], chunk_hdr[2], chunk_hdr[3]];
            let chunk_size =
                u32::from_le_bytes([chunk_hdr[4], chunk_hdr[5], chunk_hdr[6], chunk_hdr[7]]);
            let padded_size = chunk_size + (chunk_size & 1); // RIFF chunks are word-aligned

            match &chunk_id {
                b"fmt " => {
                    let mut fmt = vec![0u8; padded_size as usize];
                    inner
                        .read_exact(&mut fmt)
                        .map_err(|e| AudioError::Io(e.to_string()))?;
                    spec = Some(parse_fmt(&fmt, chunk_size)?);
                }
                b"data" => {
                    data_start = Some(
                        inner
                            .stream_position()
                            .map_err(|e| AudioError::Io(e.to_string()))?,
                    );
                    data_len = chunk_size;
                    // Skip over the data for now (we seek to it later on demand)
                    inner
                        .seek(SeekFrom::Current(i64::from(padded_size)))
                        .map_err(|e| AudioError::Io(e.to_string()))?;
                }
                _ => {
                    let mut payload = vec![0u8; padded_size as usize];
                    inner
                        .read_exact(&mut payload)
                        .map_err(|e| AudioError::Io(e.to_string()))?;
                    extra_chunks.push(RiffChunk {
                        id: chunk_id,
                        data: payload[..chunk_size as usize].to_vec(),
                    });
                }
            }
        }

        let spec = spec.ok_or_else(|| AudioError::InvalidData("Missing fmt chunk".into()))?;
        let data_start =
            data_start.ok_or_else(|| AudioError::InvalidData("Missing data chunk".into()))?;

        // Seek to start of audio data
        inner
            .seek(SeekFrom::Start(data_start))
            .map_err(|e| AudioError::Io(e.to_string()))?;

        Ok(Self {
            inner,
            spec,
            data_start,
            data_len,
            data_pos: 0,
            extra_chunks,
        })
    }

    /// Return the parsed `WavSpec`.
    #[must_use]
    pub const fn spec(&self) -> WavSpec {
        self.spec
    }

    /// Return any non-standard RIFF chunks encountered in the file.
    #[must_use]
    pub fn extra_chunks(&self) -> &[RiffChunk] {
        &self.extra_chunks
    }

    /// Number of sample *frames* (multi-channel groups) in the file.
    #[must_use]
    pub fn len(&self) -> u32 {
        let block = u32::from(self.spec.block_align());
        self.data_len.checked_div(block).unwrap_or(0)
    }

    /// Returns `true` when the file contains no audio frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read all raw audio bytes from the `data` chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying read fails.
    pub fn read_raw(&mut self) -> AudioResult<Vec<u8>> {
        self.inner
            .seek(SeekFrom::Start(self.data_start))
            .map_err(|e| AudioError::Io(e.to_string()))?;
        let mut buf = vec![0u8; self.data_len as usize];
        self.inner
            .read_exact(&mut buf)
            .map_err(|e| AudioError::Io(e.to_string()))?;
        self.data_pos = self.data_len;
        Ok(buf)
    }

    /// Read all audio samples decoded to `f32` in `[-1.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying read fails.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn read_samples_f32(&mut self) -> AudioResult<Vec<f32>> {
        let raw = self.read_raw()?;
        let samples = decode_samples_f32(&raw, &self.spec);
        Ok(samples)
    }

    /// Seek back to the start of the audio data.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying seek fails.
    pub fn seek_to_start(&mut self) -> AudioResult<()> {
        self.inner
            .seek(SeekFrom::Start(self.data_start))
            .map_err(|e| AudioError::Io(e.to_string()))?;
        self.data_pos = 0;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WavWriter
// ---------------------------------------------------------------------------

/// Streaming WAV writer.
///
/// Writes a RIFF/WAVE file to any `Write + Seek` target. Call `finalize()`
/// to patch the RIFF and data chunk sizes before dropping.
pub struct WavWriter<W: Write + Seek> {
    inner: W,
    spec: WavSpec,
    /// Position of the RIFF size field (bytes 4–7).
    riff_size_pos: u64,
    /// Position of the data chunk size field.
    data_size_pos: u64,
    /// Number of audio bytes written.
    data_bytes: u32,
    finalized: bool,
}

impl<W: Write + Seek> WavWriter<W> {
    /// Create a new WAV writer and write the file header.
    ///
    /// # Errors
    ///
    /// Returns an error if the header cannot be written to the underlying stream.
    pub fn new(mut inner: W, spec: WavSpec) -> Self {
        // We'll fill sizes in later; write placeholders.
        // Silently ignore errors during construction — they'll surface on first write.
        let _ = write_header(&mut inner, &spec);
        let riff_size_pos = 4;
        // fmt chunk is 12 + 16 = 28 bytes; data chunk header starts at byte 36
        let data_size_pos = 40;

        Self {
            inner,
            spec,
            riff_size_pos,
            data_size_pos,
            data_bytes: 0,
            finalized: false,
        }
    }

    /// Write a single f32 sample, encoded to the spec's bit depth.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying write fails.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn write_sample_f32(&mut self, sample: f32) -> AudioResult<()> {
        let encoded = encode_sample_f32(sample, &self.spec);
        self.inner
            .write_all(&encoded)
            .map_err(|e| AudioError::Io(e.to_string()))?;
        self.data_bytes += encoded.len() as u32;
        Ok(())
    }

    /// Write a slice of f32 samples.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying write fails.
    pub fn write_samples_f32(&mut self, samples: &[f32]) -> AudioResult<()> {
        for &s in samples {
            self.write_sample_f32(s)?;
        }
        Ok(())
    }

    /// Write raw PCM bytes directly (no format conversion).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying write fails.
    pub fn write_raw(&mut self, data: &[u8]) -> AudioResult<()> {
        self.inner
            .write_all(data)
            .map_err(|e| AudioError::Io(e.to_string()))?;
        self.data_bytes += data.len() as u32;
        Ok(())
    }

    /// Patch the RIFF and data chunk size fields and flush.
    ///
    /// Must be called before the writer is dropped to produce a valid WAV file.
    ///
    /// # Errors
    ///
    /// Returns an error if the seek or write fails.
    pub fn finalize(&mut self) -> AudioResult<()> {
        if self.finalized {
            return Ok(());
        }
        self.finalized = true;

        // data chunk size
        self.inner
            .seek(SeekFrom::Start(self.data_size_pos))
            .map_err(|e| AudioError::Io(e.to_string()))?;
        self.inner
            .write_all(&self.data_bytes.to_le_bytes())
            .map_err(|e| AudioError::Io(e.to_string()))?;

        // RIFF chunk size = 4 (WAVE) + 8 (fmt hdr) + 16 (fmt body) + 8 (data hdr) + data
        let riff_size: u32 = 4 + 8 + 16 + 8 + self.data_bytes;
        self.inner
            .seek(SeekFrom::Start(self.riff_size_pos))
            .map_err(|e| AudioError::Io(e.to_string()))?;
        self.inner
            .write_all(&riff_size.to_le_bytes())
            .map_err(|e| AudioError::Io(e.to_string()))?;

        self.inner
            .flush()
            .map_err(|e| AudioError::Io(e.to_string()))?;
        Ok(())
    }

    /// Return the `WavSpec` used by this writer.
    #[must_use]
    pub const fn spec(&self) -> WavSpec {
        self.spec
    }

    /// Total audio bytes written so far.
    #[must_use]
    pub const fn data_bytes(&self) -> u32 {
        self.data_bytes
    }
}

impl<W: Write + Seek> Drop for WavWriter<W> {
    fn drop(&mut self) {
        let _ = self.finalize();
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Parse a `fmt ` chunk payload into a `WavSpec`.
#[allow(clippy::cast_possible_truncation)]
fn parse_fmt(data: &[u8], size: u32) -> AudioResult<WavSpec> {
    if data.len() < 16 {
        return Err(AudioError::InvalidData("fmt chunk too short".into()));
    }
    let format_tag = u16::from_le_bytes([data[0], data[1]]);
    let channels = u16::from_le_bytes([data[2], data[3]]);
    let sample_rate = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    // skip byte_rate (8–11) and block_align (12–13)
    let bits_per_sample = u16::from_le_bytes([data[14], data[15]]);

    let float = match format_tag {
        1 => false, // PCM
        3 => true,  // IEEE float
        0xFFFE => {
            // WAVE_FORMAT_EXTENSIBLE — sub-format GUID at bytes 24–39
            if size >= 40 && data.len() >= 40 {
                // PCM GUID: {00000001-0000-0010-8000-00AA00389B71}
                // Float GUID: {00000003-0000-0010-8000-00AA00389B71}
                let sub_fmt = u16::from_le_bytes([data[24], data[25]]);
                sub_fmt == 3
            } else {
                false
            }
        }
        _ => {
            return Err(AudioError::UnsupportedFormat(format!(
                "Unsupported WAV format tag: 0x{format_tag:04X}"
            )))
        }
    };

    if channels == 0 {
        return Err(AudioError::InvalidData("Zero channels in fmt chunk".into()));
    }
    if bits_per_sample == 0 || bits_per_sample % 8 != 0 {
        return Err(AudioError::InvalidData(format!(
            "Invalid bits_per_sample: {bits_per_sample}"
        )));
    }

    Ok(WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        float,
    })
}

/// Write a standard RIFF/WAVE header with placeholder sizes.
fn write_header<W: Write>(w: &mut W, spec: &WavSpec) -> std::io::Result<()> {
    // RIFF header (size = 0 placeholder)
    w.write_all(b"RIFF")?;
    w.write_all(&0u32.to_le_bytes())?; // placeholder
    w.write_all(b"WAVE")?;

    // fmt  chunk (PCM or float)
    let format_tag: u16 = if spec.float { 3 } else { 1 };
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?; // chunk size
    w.write_all(&format_tag.to_le_bytes())?;
    w.write_all(&spec.channels.to_le_bytes())?;
    w.write_all(&spec.sample_rate.to_le_bytes())?;
    w.write_all(&spec.byte_rate().to_le_bytes())?;
    w.write_all(&spec.block_align().to_le_bytes())?;
    w.write_all(&spec.bits_per_sample.to_le_bytes())?;

    // data chunk header (size = 0 placeholder)
    w.write_all(b"data")?;
    w.write_all(&0u32.to_le_bytes())?; // placeholder

    Ok(())
}

/// Encode a single f32 sample to bytes according to spec.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn encode_sample_f32(sample: f32, spec: &WavSpec) -> Vec<u8> {
    let s = sample.clamp(-1.0, 1.0);
    match (spec.float, spec.bits_per_sample) {
        (true, 32) => s.to_le_bytes().to_vec(),
        (true, 64) => f64::from(s).to_le_bytes().to_vec(),
        (false, 8) => {
            let v = ((s * 128.0) + 128.0) as u8;
            vec![v]
        }
        (false, 16) => {
            let v = (s * 32_767.0) as i16;
            v.to_le_bytes().to_vec()
        }
        (false, 24) => {
            let v = (s * 8_388_607.0) as i32;
            let bytes = v.to_le_bytes();
            vec![bytes[0], bytes[1], bytes[2]]
        }
        (false, 32) => {
            let v = (s * 2_147_483_647.0) as i32;
            v.to_le_bytes().to_vec()
        }
        _ => vec![], // unsupported — no-op
    }
}

/// Decode raw PCM bytes to f32 samples in `[-1, 1]`.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_samples_f32(raw: &[u8], spec: &WavSpec) -> Vec<f32> {
    let bps = spec.bits_per_sample as usize / 8;
    if bps == 0 {
        return Vec::new();
    }
    let count = raw.len() / bps;
    let mut out = Vec::with_capacity(count);

    match (spec.float, spec.bits_per_sample) {
        (true, 32) => {
            for chunk in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        (true, 64) => {
            for chunk in raw.chunks_exact(8) {
                let v = f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                out.push(v as f32);
            }
        }
        (false, 8) => {
            for &b in raw {
                out.push((b as f32 - 128.0) / 128.0);
            }
        }
        (false, 16) => {
            for chunk in raw.chunks_exact(2) {
                let v = i16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(v as f32 / 32_768.0);
            }
        }
        (false, 24) => {
            for chunk in raw.chunks_exact(3) {
                let raw_i =
                    (chunk[0] as i32) | ((chunk[1] as i32) << 8) | ((chunk[2] as i32) << 16);
                let v = if raw_i & 0x80_0000 != 0 {
                    raw_i | !0xFF_FFFF
                } else {
                    raw_i
                };
                out.push(v as f32 / 8_388_607.0);
            }
        }
        (false, 32) => {
            for chunk in raw.chunks_exact(4) {
                let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                out.push(v as f32 / 2_147_483_648.0);
            }
        }
        _ => {}
    }

    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_read_roundtrip(spec: WavSpec, samples: &[f32]) -> Vec<f32> {
        let mut buf = Vec::new();
        {
            let mut w = WavWriter::new(Cursor::new(&mut buf), spec);
            w.write_samples_f32(samples).expect("write ok");
            w.finalize().expect("finalize ok");
        }
        let mut r = WavReader::new(Cursor::new(&buf)).expect("read ok");
        r.read_samples_f32().expect("samples ok")
    }

    #[test]
    fn test_wav_spec_fields() {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            float: false,
        };
        assert_eq!(spec.block_align(), 4);
        assert_eq!(spec.byte_rate(), 192_000);
    }

    #[test]
    fn test_wav_roundtrip_s16() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            float: false,
        };
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = write_read_roundtrip(spec, &input);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 0.0002, "S16 roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_wav_roundtrip_f32() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            float: true,
        };
        let input = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0, 0.123_456_79, -0.123_456_79];
        let output = write_read_roundtrip(spec, &input);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6, "F32 roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_wav_roundtrip_s24() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 24,
            float: false,
        };
        let input: Vec<f32> = vec![0.0, 0.5, -0.5, 0.9, -0.9];
        let output = write_read_roundtrip(spec, &input);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5, "S24 roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_wav_roundtrip_stereo() {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            float: false,
        };
        let input: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3];
        let output = write_read_roundtrip(spec, &input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_wav_reader_spec() {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            float: false,
        };
        let mut buf = Vec::new();
        WavWriter::new(Cursor::new(&mut buf), spec)
            .finalize()
            .expect("ok");
        let r = WavReader::new(Cursor::new(&buf)).expect("ok");
        assert_eq!(r.spec().channels, 2);
        assert_eq!(r.spec().sample_rate, 44_100);
        assert_eq!(r.spec().bits_per_sample, 16);
        assert!(!r.spec().float);
    }

    #[test]
    fn test_wav_reader_len() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            float: false,
        };
        let input: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0 - 0.5).collect();
        let mut buf = Vec::new();
        {
            let mut w = WavWriter::new(Cursor::new(&mut buf), spec);
            w.write_samples_f32(&input).expect("write");
            w.finalize().expect("finalize");
        }
        let r = WavReader::new(Cursor::new(&buf)).expect("read");
        assert_eq!(r.len(), 1000);
    }

    #[test]
    fn test_wav_invalid_not_riff() {
        let bad = b"BAD!data____";
        assert!(WavReader::new(Cursor::new(bad)).is_err());
    }

    #[test]
    fn test_wav_invalid_not_wave() {
        let mut bad = b"RIFF\x00\x00\x00\x00AVI ".to_vec();
        bad.extend_from_slice(&[0u8; 100]);
        assert!(WavReader::new(Cursor::new(&bad)).is_err());
    }

    #[test]
    fn test_wav_writer_data_bytes() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            float: false,
        };
        let mut buf = Vec::new();
        let mut w = WavWriter::new(Cursor::new(&mut buf), spec);
        w.write_sample_f32(0.0).expect("ok");
        w.write_sample_f32(0.5).expect("ok");
        assert_eq!(w.data_bytes(), 4); // 2 samples × 2 bytes each
    }

    #[test]
    fn test_wav_roundtrip_u8() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 22_050,
            bits_per_sample: 8,
            float: false,
        };
        let input: Vec<f32> = (0..50).map(|i| (i as f32 / 25.0) - 1.0).collect();
        let output = write_read_roundtrip(spec, &input);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            // 8-bit has large quantisation error
            assert!((a - b).abs() < 0.02, "U8 roundtrip mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_wav_seek_to_start() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 44_100,
            bits_per_sample: 16,
            float: false,
        };
        let input = vec![0.5_f32, -0.5];
        let mut buf = Vec::new();
        {
            let mut w = WavWriter::new(Cursor::new(&mut buf), spec);
            w.write_samples_f32(&input).expect("write");
            w.finalize().expect("finalize");
        }
        let mut r = WavReader::new(Cursor::new(&buf)).expect("read");
        let first = r.read_samples_f32().expect("first read");
        r.seek_to_start().expect("seek");
        let second = r.read_samples_f32().expect("second read");
        assert_eq!(first, second);
    }

    #[test]
    fn test_wav_roundtrip_s32() {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 96_000,
            bits_per_sample: 32,
            float: false,
        };
        let input: Vec<f32> = vec![0.0, 0.5, -0.5, 0.99, -0.99];
        let output = write_read_roundtrip(spec, &input);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5, "S32 roundtrip mismatch: {a} vs {b}");
        }
    }
}
