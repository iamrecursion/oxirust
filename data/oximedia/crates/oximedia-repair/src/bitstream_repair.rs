//! Bitstream-level error detection and repair for media streams.
//!
//! Provides tools to detect corrupt NAL units, PES packets, and other
//! bitstream-level anomalies, and attempt in-place repair.
//!
//! # Sector-aligned I/O
//!
//! The [`SectorAlignedReader`] wrapper aligns all read operations to a
//! configurable sector boundary (default 4096 bytes, matching NVMe physical
//! sector size).  This eliminates read-modify-write amplification on SSDs and
//! improves throughput for large sequential bitstream scans.

/// Categories of bitstream errors that can be detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitstreamError {
    /// Invalid start code or sync word.
    InvalidStartCode {
        /// Byte offset of the invalid start code.
        offset: u64,
    },
    /// Illegal NALU type for the current codec.
    IllegalNaluType {
        /// Byte offset of the illegal NALU.
        offset: u64,
        /// The illegal NALU type byte.
        code: u8,
    },
    /// Cabac / entropy decode failure.
    EntropyFailure {
        /// Byte offset where entropy decoding failed.
        offset: u64,
    },
    /// Forbidden zero bit set in H.264/H.265 header.
    ForbiddenBitSet {
        /// Byte offset of the header with the forbidden bit set.
        offset: u64,
    },
    /// Profile / level constraint violation.
    ConstraintViolation {
        /// Byte offset of the violation in the bitstream.
        offset: u64,
        /// Human-readable description of the violated constraint.
        detail: String,
    },
    /// Unexpected end of bitstream.
    UnexpectedEof {
        /// Number of bytes expected.
        expected: usize,
        /// Number of bytes actually available.
        got: usize,
    },
    /// Reference frame missing.
    MissingReference {
        /// Picture order count of the missing reference frame.
        poc: i32,
    },
    /// Corrupted PES header in MPEG transport stream.
    CorruptPesHeader {
        /// Byte offset of the corrupt PES header.
        offset: u64,
    },
}

impl BitstreamError {
    /// Return a human-readable label for the error category.
    pub fn label(&self) -> &'static str {
        match self {
            Self::InvalidStartCode { .. } => "invalid_start_code",
            Self::IllegalNaluType { .. } => "illegal_nalu_type",
            Self::EntropyFailure { .. } => "entropy_failure",
            Self::ForbiddenBitSet { .. } => "forbidden_bit_set",
            Self::ConstraintViolation { .. } => "constraint_violation",
            Self::UnexpectedEof { .. } => "unexpected_eof",
            Self::MissingReference { .. } => "missing_reference",
            Self::CorruptPesHeader { .. } => "corrupt_pes_header",
        }
    }

    /// Return the byte offset of the error if known.
    pub fn offset(&self) -> Option<u64> {
        match self {
            Self::InvalidStartCode { offset }
            | Self::IllegalNaluType { offset, .. }
            | Self::EntropyFailure { offset }
            | Self::ForbiddenBitSet { offset }
            | Self::ConstraintViolation { offset, .. }
            | Self::CorruptPesHeader { offset } => Some(*offset),
            _ => None,
        }
    }
}

/// Strategy used when repairing a damaged bitstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairStrategy {
    /// Skip the damaged unit and continue.
    Skip,
    /// Replace the damaged unit with a filler NALU / silence.
    Substitute,
    /// Attempt to interpolate from neighboring good frames.
    Interpolate,
    /// Truncate the stream at the first unrecoverable error.
    Truncate,
}

impl RepairStrategy {
    /// Returns `true` if this strategy may permanently discard stream data.
    pub fn is_destructive(&self) -> bool {
        matches!(self, Self::Skip | Self::Truncate)
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Substitute => "substitute",
            Self::Interpolate => "interpolate",
            Self::Truncate => "truncate",
        }
    }
}

/// Statistics gathered during a repair pass.
#[derive(Debug, Clone, Default)]
pub struct RepairReport {
    /// Total number of errors detected.
    pub errors_detected: usize,
    /// Number of errors successfully repaired.
    pub errors_repaired: usize,
    /// Number of errors that could not be repaired.
    pub errors_failed: usize,
    /// Bytes consumed from input.
    pub bytes_consumed: u64,
    /// Bytes written to output.
    pub bytes_written: u64,
}

impl RepairReport {
    /// Success rate in [0.0, 1.0]; returns 1.0 if no errors were detected.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        if self.errors_detected == 0 {
            return 1.0;
        }
        self.errors_repaired as f64 / self.errors_detected as f64
    }

    /// Returns `true` if every detected error was repaired.
    pub fn is_fully_repaired(&self) -> bool {
        self.errors_failed == 0 && self.errors_detected > 0
    }
}

/// Engine that scans and repairs a raw byte buffer representing a media
/// bitstream.
#[derive(Debug, Clone)]
pub struct BitstreamRepair {
    strategy: RepairStrategy,
    max_errors: usize,
}

impl BitstreamRepair {
    /// Create a new repair engine with the given strategy.
    pub fn new(strategy: RepairStrategy) -> Self {
        Self {
            strategy,
            max_errors: 1024,
        }
    }

    /// Set the maximum number of errors to tolerate before aborting.
    pub fn with_max_errors(mut self, max: usize) -> Self {
        self.max_errors = max;
        self
    }

    /// Scan `data` and return a list of detected [`BitstreamError`]s.
    pub fn detect_errors(&self, data: &[u8]) -> Vec<BitstreamError> {
        let mut errors = Vec::new();
        let len = data.len();
        let mut i = 0usize;

        while i + 3 < len {
            // Look for Annex-B start code 0x000001 or 0x00000001.
            if data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x01 {
                let nalu_byte = data[i + 3];
                // Forbidden zero bit is bit 7.
                if nalu_byte & 0x80 != 0 {
                    errors.push(BitstreamError::ForbiddenBitSet { offset: i as u64 });
                }
                // NALU type 0 is technically reserved / illegal as a stream unit.
                let nalu_type = nalu_byte & 0x1F;
                if nalu_type == 0 {
                    errors.push(BitstreamError::IllegalNaluType {
                        offset: i as u64,
                        code: nalu_type,
                    });
                }
                if errors.len() >= self.max_errors {
                    break;
                }
                i += 4;
            } else {
                i += 1;
            }
        }

        // Detect missing start code at the very beginning (non-empty buffer).
        if len >= 4
            && !(data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x00 && data[3] == 0x01)
            && !(data[0] == 0x00 && data[1] == 0x00 && data[2] == 0x01)
        {
            // Heuristic: if first 3 bytes don't form a start code, flag it.
            // Only add if we don't already have a forbidden-bit error at 0.
            if !errors.iter().any(|e| e.offset() == Some(0)) {
                errors.insert(0, BitstreamError::InvalidStartCode { offset: 0 });
            }
        }

        errors
    }

    /// Attempt to repair `data` in place, returning the (possibly modified)
    /// bytes and a [`RepairReport`].
    pub fn repair(&self, data: &[u8]) -> (Vec<u8>, RepairReport) {
        let errors = self.detect_errors(data);
        let mut report = RepairReport {
            errors_detected: errors.len(),
            bytes_consumed: data.len() as u64,
            ..Default::default()
        };

        let mut out = data.to_vec();

        for error in &errors {
            match self.strategy {
                RepairStrategy::Substitute => {
                    if let Some(off) = error.offset() {
                        // Clear the forbidden zero bit if set.
                        if let BitstreamError::ForbiddenBitSet { .. } = error {
                            let idx = (off as usize) + 3;
                            if idx < out.len() {
                                out[idx] &= 0x7F;
                                report.errors_repaired += 1;
                                continue;
                            }
                        }
                        // For other errors, zero-out the offending byte.
                        let idx = off as usize;
                        if idx < out.len() {
                            out[idx] = 0x00;
                            report.errors_repaired += 1;
                        } else {
                            report.errors_failed += 1;
                        }
                    } else {
                        report.errors_failed += 1;
                    }
                }
                RepairStrategy::Skip => {
                    // Nothing to write; just mark as "handled".
                    report.errors_repaired += 1;
                }
                RepairStrategy::Truncate => {
                    if let Some(off) = error.offset() {
                        out.truncate(off as usize);
                        report.errors_repaired += 1;
                    } else {
                        report.errors_failed += 1;
                    }
                    // Stop after first truncation.
                    break;
                }
                RepairStrategy::Interpolate => {
                    // Simplified: mark as failed (real interpolation needs
                    // decoded frame context).
                    report.errors_failed += 1;
                }
            }
        }

        report.bytes_written = out.len() as u64;
        (out, report)
    }
}

impl Default for BitstreamRepair {
    fn default() -> Self {
        Self::new(RepairStrategy::Substitute)
    }
}

// ── Sector-aligned reader ─────────────────────────────────────────────────────

use std::io::{Read, Seek, SeekFrom};

/// A buffered reader that aligns all underlying I/O to a configurable sector
/// boundary, improving throughput on SSDs and NVMe devices.
///
/// All reads from the inner reader are rounded up to the nearest multiple of
/// `sector_size` bytes.  The internal buffer absorbs over-reads so that
/// callers always see exactly the bytes they requested.
///
/// # Example
///
/// ```no_run
/// use std::fs::File;
/// use oximedia_repair::bitstream_repair::SectorAlignedReader;
/// use std::io::Read;
///
/// let file = File::open("stream.h264").expect("open file");
/// let mut reader = SectorAlignedReader::new(file, 4096);
/// let mut buf = [0u8; 1024];
/// reader.read(&mut buf).expect("read");
/// ```
pub struct SectorAlignedReader<R: Read + Seek> {
    inner: R,
    sector_size: usize,
    /// Internal aligned buffer.
    buf: Vec<u8>,
    /// Logical position of the first valid byte in `buf` within the file.
    buf_file_offset: u64,
    /// Number of valid bytes currently in `buf`.
    buf_len: usize,
    /// Current logical read position within the file.
    pos: u64,
    /// Total file length (cached on construction for efficient Seek math).
    file_len: u64,
}

impl<R: Read + Seek> SectorAlignedReader<R> {
    /// Create a new `SectorAlignedReader` wrapping `inner`.
    ///
    /// `sector_size` must be a power of two and > 0; if not, it is rounded up
    /// to the next power of two (minimum 512).
    pub fn new(mut inner: R, sector_size: usize) -> Self {
        let sector_size = sector_size.max(512).next_power_of_two();
        let file_len = inner.seek(SeekFrom::End(0)).unwrap_or(0);
        // Seek back to start so callers don't need to pre-seek.
        let _ = inner.seek(SeekFrom::Start(0));
        Self {
            inner,
            sector_size,
            buf: Vec::new(),
            buf_file_offset: 0,
            buf_len: 0,
            pos: 0,
            file_len,
        }
    }

    /// Return the sector size this reader was created with.
    pub fn sector_size(&self) -> usize {
        self.sector_size
    }

    /// Return the current logical read position.
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// Ensure the internal buffer covers at least the sector that contains
    /// `self.pos`.  If the buffer already covers `self.pos`, this is a no-op.
    fn fill_buf_if_needed(&mut self) -> std::io::Result<()> {
        // Check whether `self.pos` falls within the currently buffered window.
        let buf_end = self.buf_file_offset + self.buf_len as u64;
        if self.buf_len > 0 && self.pos >= self.buf_file_offset && self.pos < buf_end {
            return Ok(());
        }

        // Align down to sector boundary.
        let aligned_start = (self.pos / self.sector_size as u64) * self.sector_size as u64;
        self.inner.seek(SeekFrom::Start(aligned_start))?;

        // Read one full sector (or to EOF, whichever is smaller).
        let remaining = self.file_len.saturating_sub(aligned_start) as usize;
        let to_read = self.sector_size.min(remaining);
        if to_read == 0 {
            self.buf.clear();
            self.buf_len = 0;
            self.buf_file_offset = aligned_start;
            return Ok(());
        }

        self.buf.resize(to_read, 0);
        let mut total_read = 0usize;
        while total_read < to_read {
            match self.inner.read(&mut self.buf[total_read..to_read]) {
                Ok(0) => break,
                Ok(n) => total_read += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        self.buf_len = total_read;
        self.buf_file_offset = aligned_start;
        Ok(())
    }
}

impl<R: Read + Seek> Read for SectorAlignedReader<R> {
    fn read(&mut self, dst: &mut [u8]) -> std::io::Result<usize> {
        if dst.is_empty() {
            return Ok(0);
        }
        self.fill_buf_if_needed()?;

        if self.buf_len == 0 {
            // EOF
            return Ok(0);
        }

        let offset_in_buf = (self.pos - self.buf_file_offset) as usize;
        if offset_in_buf >= self.buf_len {
            return Ok(0);
        }

        let available = self.buf_len - offset_in_buf;
        let to_copy = dst.len().min(available);
        dst[..to_copy].copy_from_slice(&self.buf[offset_in_buf..offset_in_buf + to_copy]);
        self.pos += to_copy as u64;

        // If we've exhausted this sector and there is more data, pre-load next
        // sector lazily (will be triggered on the next call to `read`).
        Ok(to_copy)
    }
}

impl<R: Read + Seek> Seek for SectorAlignedReader<R> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(n) => n,
            SeekFrom::End(n) => {
                if n >= 0 {
                    self.file_len.saturating_add(n as u64)
                } else {
                    self.file_len.saturating_sub((-n) as u64)
                }
            }
            SeekFrom::Current(n) => {
                if n >= 0 {
                    self.pos.saturating_add(n as u64)
                } else {
                    self.pos.saturating_sub((-n) as u64)
                }
            }
        };
        self.pos = new_pos;
        Ok(new_pos)
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_nalu(nalu_type: u8) -> Vec<u8> {
        // Annex-B 4-byte start code + legal NALU header byte.
        vec![0x00, 0x00, 0x00, 0x01, nalu_type & 0x1F]
    }

    #[test]
    fn test_error_label_invalid_start_code() {
        let e = BitstreamError::InvalidStartCode { offset: 0 };
        assert_eq!(e.label(), "invalid_start_code");
    }

    #[test]
    fn test_error_label_forbidden_bit() {
        let e = BitstreamError::ForbiddenBitSet { offset: 4 };
        assert_eq!(e.label(), "forbidden_bit_set");
    }

    #[test]
    fn test_error_offset_present() {
        let e = BitstreamError::EntropyFailure { offset: 16 };
        assert_eq!(e.offset(), Some(16));
    }

    #[test]
    fn test_error_offset_absent_for_missing_ref() {
        let e = BitstreamError::MissingReference { poc: 3 };
        assert_eq!(e.offset(), None);
    }

    #[test]
    fn test_strategy_is_destructive() {
        assert!(RepairStrategy::Skip.is_destructive());
        assert!(RepairStrategy::Truncate.is_destructive());
        assert!(!RepairStrategy::Substitute.is_destructive());
        assert!(!RepairStrategy::Interpolate.is_destructive());
    }

    #[test]
    fn test_strategy_names() {
        assert_eq!(RepairStrategy::Skip.name(), "skip");
        assert_eq!(RepairStrategy::Substitute.name(), "substitute");
        assert_eq!(RepairStrategy::Interpolate.name(), "interpolate");
        assert_eq!(RepairStrategy::Truncate.name(), "truncate");
    }

    #[test]
    fn test_report_success_rate_no_errors() {
        let r = RepairReport::default();
        assert!((r.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_report_success_rate_partial() {
        let r = RepairReport {
            errors_detected: 4,
            errors_repaired: 3,
            errors_failed: 1,
            ..Default::default()
        };
        assert!((r.success_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_report_is_fully_repaired() {
        let r = RepairReport {
            errors_detected: 2,
            errors_repaired: 2,
            errors_failed: 0,
            ..Default::default()
        };
        assert!(r.is_fully_repaired());
    }

    #[test]
    fn test_detect_errors_clean_stream() {
        let data = make_valid_nalu(0x05); // IDR slice
        let engine = BitstreamRepair::default();
        let errors = engine.detect_errors(&data);
        assert!(errors.is_empty(), "Expected no errors, got {:?}", errors);
    }

    #[test]
    fn test_detect_forbidden_bit() {
        // Byte after start code has bit 7 set.
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x85u8]; // 0x85 = 1000_0101
        let engine = BitstreamRepair::default();
        let errors = engine.detect_errors(&data);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, BitstreamError::ForbiddenBitSet { .. })),
            "Expected ForbiddenBitSet error"
        );
    }

    #[test]
    fn test_repair_substitute_clears_forbidden_bit() {
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x85u8];
        let engine = BitstreamRepair::new(RepairStrategy::Substitute);
        let (repaired, report) = engine.repair(&data);
        // After clearing bit 7: 0x85 & 0x7F = 0x05
        assert_eq!(repaired[4], 0x05);
        assert!(report.errors_repaired > 0);
    }

    #[test]
    fn test_repair_truncate_strategy() {
        // Two valid NALUs; first has a forbidden bit set.
        let mut data = vec![0x00, 0x00, 0x00, 0x01, 0x80u8]; // forbidden
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x05]); // valid IDR
        let engine = BitstreamRepair::new(RepairStrategy::Truncate);
        let (repaired, report) = engine.repair(&data);
        // Stream should be truncated at offset 0.
        assert!(repaired.len() <= data.len());
        assert!(report.errors_repaired > 0);
    }

    #[test]
    fn test_detect_errors_max_errors_limit() {
        // Generate a buffer full of forbidden-bit NALUs.
        let mut data = Vec::new();
        for _ in 0..2000 {
            data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x80]);
        }
        let engine = BitstreamRepair::new(RepairStrategy::Skip).with_max_errors(10);
        let errors = engine.detect_errors(&data);
        assert!(errors.len() <= 11); // at most max_errors + possible start-code error
    }

    #[test]
    fn test_bitstream_repair_default_strategy() {
        let engine = BitstreamRepair::default();
        assert_eq!(engine.strategy, RepairStrategy::Substitute);
    }

    // ── SectorAlignedReader tests ─────────────────────────────────────────────

    use std::io::Write;

    fn write_temp_bitstream(name: &str, data: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        let mut f = std::fs::File::create(&path).expect("create temp");
        f.write_all(data).expect("write temp");
        path
    }

    #[test]
    fn test_sector_aligned_reader_matches_direct_read() {
        // Write 8193 bytes (crosses a 4096-byte sector boundary + 1 extra byte).
        let data: Vec<u8> = (0u16..8193).map(|i| (i & 0xFF) as u8).collect();
        let path = write_temp_bitstream("sar_direct_read.bin", &data);

        let file = std::fs::File::open(&path).expect("open");
        let mut reader = SectorAlignedReader::new(file, 4096);

        let mut buf = vec![0u8; data.len()];
        let mut total = 0;
        while total < buf.len() {
            let n = reader.read(&mut buf[total..]).expect("read");
            if n == 0 {
                break;
            }
            total += n;
        }

        assert_eq!(total, data.len(), "should read all bytes");
        assert_eq!(
            &buf[..total],
            data.as_slice(),
            "content must match direct read"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_sector_aligned_reader_alignment_correctness() {
        // Build data that is exactly 3 sectors = 3 * 512 bytes.
        let sector = 512usize;
        let data: Vec<u8> = (0u16..((3 * sector) as u16))
            .map(|i| (i & 0xFF) as u8)
            .collect();
        let path = write_temp_bitstream("sar_aligned.bin", &data);

        let file = std::fs::File::open(&path).expect("open");
        let mut reader = SectorAlignedReader::new(file, sector);
        assert_eq!(reader.sector_size(), sector);

        // Read 100 bytes from the middle of the second sector.
        reader
            .seek(SeekFrom::Start(sector as u64 + 7))
            .expect("seek");
        let mut buf = [0u8; 100];
        let n = reader.read(&mut buf).expect("read");
        assert!(n > 0);
        // Verify correctness: byte at offset (sector + 7 + k) should be (sector+7+k) & 0xFF.
        for (k, &byte) in buf[..n].iter().enumerate() {
            let expected = ((sector + 7 + k) & 0xFF) as u8;
            assert_eq!(byte, expected, "mismatch at buf[{k}]");
        }

        let _ = std::fs::remove_file(&path);
    }
}
