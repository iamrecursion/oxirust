//! Segment creation for adaptive streaming.

use crate::config::{SegmentConfig, SegmentFormat};
use crate::error::PackagerResult;
use bytes::{BufMut, BytesMut};
use std::time::Duration;
use tracing::{debug, trace};

/// Segment information.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment index.
    pub index: u64,
    /// Segment duration.
    pub duration: Duration,
    /// Segment size in bytes.
    pub size: u64,
    /// Segment file path (relative).
    pub path: String,
    /// Is keyframe segment.
    pub keyframe: bool,
    /// Timestamp.
    pub timestamp: Duration,
}

/// Keyframe information.
#[derive(Debug, Clone)]
pub struct Keyframe {
    /// Frame timestamp.
    pub timestamp: Duration,
    /// Frame position in stream.
    pub position: u64,
    /// Frame size.
    pub size: u32,
}

/// Helper trait to add `put_u24` for `BytesMut`
trait BytesMutExt {
    fn put_u24(&mut self, n: u32);
}

impl BytesMutExt for BytesMut {
    fn put_u24(&mut self, n: u32) {
        let bytes = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        self.put_slice(&bytes);
    }
}

/// MPEG-2 systems CRC-32 polynomial (ISO/IEC 13818-1 Annex A / §2.4.4.10).
const MPEG2_CRC32_POLY: u32 = 0x04C1_1DB7;

/// Compute the MPEG-2 systems CRC-32 over `data`.
///
/// PSI sections (PAT/PMT) carry a `CRC_32` field computed with the MPEG-2
/// parameters (ISO/IEC 13818-1 §2.4.4.10): polynomial [`MPEG2_CRC32_POLY`],
/// initial value `0xFFFF_FFFF`, MSB-first, with no input/output reflection and
/// no final XOR. Strict/broadcast demuxers reject sections whose CRC does not
/// match, so this must be a real computation rather than a zero placeholder.
fn mpeg2_crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= u32::from(byte) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 {
                (crc << 1) ^ MPEG2_CRC32_POLY
            } else {
                crc << 1
            };
        }
    }
    crc
}

/// Segment generator.
pub struct SegmentGenerator {
    config: SegmentConfig,
    segment_index: u64,
    current_duration: Duration,
    current_data: BytesMut,
    keyframes: Vec<Keyframe>,
}

impl SegmentGenerator {
    /// Create a new segment generator.
    #[must_use]
    pub fn new(config: SegmentConfig) -> Self {
        Self {
            config,
            segment_index: 0,
            current_duration: Duration::ZERO,
            current_data: BytesMut::new(),
            keyframes: Vec::new(),
        }
    }

    /// Add a frame to the current segment.
    pub fn add_frame(
        &mut self,
        data: &[u8],
        is_keyframe: bool,
        timestamp: Duration,
    ) -> PackagerResult<Option<SegmentInfo>> {
        trace!(
            "Adding frame: size={}, keyframe={}",
            data.len(),
            is_keyframe
        );

        if is_keyframe {
            self.keyframes.push(Keyframe {
                timestamp,
                position: self.current_data.len() as u64,
                size: data.len() as u32,
            });
        }

        self.current_data.extend_from_slice(data);
        self.current_duration = timestamp;

        // Check if we should finalize the segment
        if self.should_finalize_segment(is_keyframe, timestamp) {
            return self.finalize_segment();
        }

        Ok(None)
    }

    /// Check if segment should be finalized.
    fn should_finalize_segment(&self, is_keyframe: bool, timestamp: Duration) -> bool {
        if self.current_data.is_empty() {
            return false;
        }

        // If keyframe alignment is enabled, only finalize on keyframes
        if self.config.keyframe_alignment && !is_keyframe {
            return false;
        }

        // Check if we've reached target duration
        timestamp >= self.config.duration
    }

    /// Finalize the current segment.
    fn finalize_segment(&mut self) -> PackagerResult<Option<SegmentInfo>> {
        if self.current_data.is_empty() {
            return Ok(None);
        }

        debug!(
            "Finalizing segment {}: {} bytes, duration: {:?}",
            self.segment_index,
            self.current_data.len(),
            self.current_duration
        );

        let segment_data = match self.config.format {
            SegmentFormat::MpegTs => self.create_mpegts_segment()?,
            SegmentFormat::Fmp4 => self.create_fmp4_segment()?,
            SegmentFormat::Cmaf => self.create_cmaf_segment()?,
        };

        let segment_info = SegmentInfo {
            index: self.segment_index,
            duration: self.current_duration,
            size: segment_data.len() as u64,
            path: self.get_segment_path(),
            keyframe: !self.keyframes.is_empty(),
            timestamp: self.current_duration,
        };

        // Reset for next segment
        self.segment_index += 1;
        self.current_data.clear();
        self.current_duration = Duration::ZERO;
        self.keyframes.clear();

        Ok(Some(segment_info))
    }

    /// Create MPEG-TS segment.
    fn create_mpegts_segment(&self) -> PackagerResult<Vec<u8>> {
        debug!("Creating MPEG-TS segment");

        let mut output = BytesMut::new();

        // MPEG-TS packet size is 188 bytes
        const TS_PACKET_SIZE: usize = 188;
        const SYNC_BYTE: u8 = 0x47;

        // PAT (Program Association Table)
        self.write_pat(&mut output)?;

        // PMT (Program Map Table)
        self.write_pmt(&mut output)?;

        // PES (Packetized Elementary Stream) packets
        let payload = &self.current_data[..];
        let mut offset = 0;

        while offset < payload.len() {
            let chunk_size = (payload.len() - offset).min(TS_PACKET_SIZE - 4);

            // TS packet header
            output.put_u8(SYNC_BYTE);
            output.put_u8(0x40); // Payload unit start
            output.put_u8(0x01); // PID (low byte)
            output.put_u8(0x10); // Continuity counter

            // Payload
            output.put_slice(&payload[offset..offset + chunk_size]);

            // Padding
            let padding = TS_PACKET_SIZE - 4 - chunk_size;
            output.put_bytes(0xFF, padding);

            offset += chunk_size;
        }

        Ok(output.to_vec())
    }

    /// Write PAT (Program Association Table).
    fn write_pat(&self, output: &mut BytesMut) -> PackagerResult<()> {
        const TS_PACKET_SIZE: usize = 188;
        const SYNC_BYTE: u8 = 0x47;

        output.put_u8(SYNC_BYTE);
        output.put_u8(0x40); // Payload unit start
        output.put_u8(0x00); // PID 0 for PAT
        output.put_u8(0x10); // Continuity counter

        // PAT table
        output.put_u8(0x00); // Pointer field
        output.put_u8(0x00); // Table ID
        output.put_u8(0xB0); // Section syntax indicator
        output.put_u8(0x0D); // Section length

        // Transport stream ID
        output.put_u16(0x0001);

        // Version number and current/next indicator
        output.put_u8(0xC1);

        // Section number and last section number
        output.put_u8(0x00);
        output.put_u8(0x00);

        // Program number and PID
        output.put_u16(0x0001);
        output.put_u16(0xE100); // PMT PID = 256

        // CRC_32 over the section: from the table_id byte through the end of
        // the section data, i.e. all bytes after the 4-byte TS header and the
        // 1-byte pointer_field (ISO/IEC 13818-1 §2.4.4.10).
        let crc = mpeg2_crc32(&output[5..]);
        output.put_u32(crc);

        // Padding
        let padding = TS_PACKET_SIZE - output.len();
        output.put_bytes(0xFF, padding);

        Ok(())
    }

    /// Write PMT (Program Map Table).
    fn write_pmt(&self, output: &mut BytesMut) -> PackagerResult<()> {
        const TS_PACKET_SIZE: usize = 188;
        const SYNC_BYTE: u8 = 0x47;

        output.put_u8(SYNC_BYTE);
        output.put_u8(0x41); // Payload unit start
        output.put_u8(0x00); // PID 256 for PMT (high)
        output.put_u8(0x10); // Continuity counter

        // PMT table
        output.put_u8(0x00); // Pointer field
        output.put_u8(0x02); // Table ID (PMT)
        output.put_u8(0xB0); // Section syntax indicator
        output.put_u8(0x12); // Section length

        // Program number
        output.put_u16(0x0001);

        // Version number
        output.put_u8(0xC1);

        // Section number and last section number
        output.put_u8(0x00);
        output.put_u8(0x00);

        // PCR PID
        output.put_u16(0xE101);

        // Program info length
        output.put_u16(0xF000);

        // Elementary stream info (video)
        output.put_u8(0x1B); // Stream type (H.264 placeholder)
        output.put_u16(0xE101); // Elementary PID
        output.put_u16(0xF000); // ES info length

        // CRC_32 over the section: from the table_id byte through the end of
        // the section data, i.e. all bytes after the 4-byte TS header and the
        // 1-byte pointer_field (ISO/IEC 13818-1 §2.4.4.10).
        let crc = mpeg2_crc32(&output[5..]);
        output.put_u32(crc);

        // Padding
        let padding = TS_PACKET_SIZE - output.len();
        output.put_bytes(0xFF, padding);

        Ok(())
    }

    /// Create fragmented MP4 segment.
    fn create_fmp4_segment(&self) -> PackagerResult<Vec<u8>> {
        debug!("Creating fMP4 segment");

        let mut output = BytesMut::new();

        if self.config.fast_start {
            // Write moof (movie fragment) before mdat
            self.write_moof(&mut output)?;
            self.write_mdat(&mut output)?;
        } else {
            // Write mdat before moof
            self.write_mdat(&mut output)?;
            self.write_moof(&mut output)?;
        }

        Ok(output.to_vec())
    }

    /// Write moof (Movie Fragment) box.
    fn write_moof(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let moof_start = output.len();

        // Box header (size placeholder + type)
        output.put_u32(0); // Size placeholder
        output.put_slice(b"moof");

        // mfhd (Movie Fragment Header)
        self.write_mfhd(output)?;

        // traf (Track Fragment)
        self.write_traf(output)?;

        // Update moof size
        let moof_size = output.len() - moof_start;
        let size_bytes = (moof_size as u32).to_be_bytes();
        output[moof_start..moof_start + 4].copy_from_slice(&size_bytes);

        Ok(())
    }

    /// Write mfhd (Movie Fragment Header) box.
    fn write_mfhd(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(16); // Box size
        output.put_slice(b"mfhd");
        output.put_u8(0); // Version
        output.put_u24(0); // Flags
        output.put_u32(self.segment_index as u32); // Sequence number

        Ok(())
    }

    /// Write traf (Track Fragment) box.
    fn write_traf(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let traf_start = output.len();

        // Box header
        output.put_u32(0); // Size placeholder
        output.put_slice(b"traf");

        // tfhd (Track Fragment Header)
        self.write_tfhd(output)?;

        // tfdt (Track Fragment Decode Time)
        self.write_tfdt(output)?;

        // trun (Track Fragment Run)
        self.write_trun(output)?;

        // Update traf size
        let traf_size = output.len() - traf_start;
        let size_bytes = (traf_size as u32).to_be_bytes();
        output[traf_start..traf_start + 4].copy_from_slice(&size_bytes);

        Ok(())
    }

    /// Write tfhd (Track Fragment Header) box.
    fn write_tfhd(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(16); // Box size
        output.put_slice(b"tfhd");
        output.put_u8(0); // Version
        output.put_u24(0x020000); // Flags (default-base-is-moof)
        output.put_u32(1); // Track ID

        Ok(())
    }

    /// Write tfdt (Track Fragment Decode Time) box.
    fn write_tfdt(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(20); // Box size
        output.put_slice(b"tfdt");
        output.put_u8(1); // Version (64-bit)
        output.put_u24(0); // Flags
        output.put_u64(self.current_duration.as_millis() as u64); // Base media decode time

        Ok(())
    }

    /// Write trun (Track Fragment Run) box.
    fn write_trun(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let sample_count = 1; // Simplified: one sample per segment

        output.put_u32(20); // Box size
        output.put_slice(b"trun");
        output.put_u8(0); // Version
        output.put_u24(0x000001); // Flags (data-offset-present)
        output.put_u32(sample_count); // Sample count
        output.put_u32(0); // Data offset (updated later)

        Ok(())
    }

    /// Write mdat (Media Data) box.
    fn write_mdat(&self, output: &mut BytesMut) -> PackagerResult<()> {
        let mdat_size = 8 + self.current_data.len();
        output.put_u32(mdat_size as u32); // Box size
        output.put_slice(b"mdat");
        output.put_slice(&self.current_data);

        Ok(())
    }

    /// Create CMAF segment.
    fn create_cmaf_segment(&self) -> PackagerResult<Vec<u8>> {
        debug!("Creating CMAF segment");

        // CMAF is essentially fMP4 with additional constraints
        let mut output = BytesMut::new();

        // Write styp (Segment Type) box
        self.write_styp(&mut output)?;

        // Write moof and mdat
        self.write_moof(&mut output)?;
        self.write_mdat(&mut output)?;

        Ok(output.to_vec())
    }

    /// Write styp (Segment Type) box.
    fn write_styp(&self, output: &mut BytesMut) -> PackagerResult<()> {
        output.put_u32(24); // Box size
        output.put_slice(b"styp");
        output.put_slice(b"cmfc"); // Major brand (CMAF chunk)
        output.put_u32(0); // Minor version
        output.put_slice(b"iso6"); // Compatible brand
        output.put_slice(b"cmfc"); // Compatible brand

        Ok(())
    }

    /// Get segment file path.
    fn get_segment_path(&self) -> String {
        match self.config.format {
            SegmentFormat::MpegTs => format!("segment_{}.ts", self.segment_index),
            SegmentFormat::Fmp4 => format!("segment_{}.m4s", self.segment_index),
            SegmentFormat::Cmaf => format!("chunk_{}.m4s", self.segment_index),
        }
    }

    /// Get current segment index.
    #[must_use]
    pub fn segment_index(&self) -> u64 {
        self.segment_index
    }

    /// Reset the generator.
    pub fn reset(&mut self) {
        self.segment_index = 0;
        self.current_duration = Duration::ZERO;
        self.current_data.clear();
        self.keyframes.clear();
    }
}

/// Segment writer for writing segments to disk.
pub struct SegmentWriter {
    output_dir: std::path::PathBuf,
}

impl SegmentWriter {
    /// Create a new segment writer.
    #[must_use]
    pub fn new(output_dir: std::path::PathBuf) -> Self {
        Self { output_dir }
    }

    /// Write a segment to disk.
    pub async fn write_segment(&self, segment: &SegmentInfo, data: &[u8]) -> PackagerResult<()> {
        let path = self.output_dir.join(&segment.path);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&path, data).await?;

        debug!(
            "Wrote segment {} to {} ({} bytes)",
            segment.index,
            path.display(),
            data.len()
        );

        Ok(())
    }

    /// Delete old segments based on max count.
    pub async fn cleanup_old_segments(
        &self,
        current_index: u64,
        max_segments: usize,
    ) -> PackagerResult<()> {
        if current_index <= max_segments as u64 {
            return Ok(());
        }

        let delete_index = current_index - max_segments as u64;

        // Try to delete old segment files
        for ext in &["ts", "m4s"] {
            let path = self
                .output_dir
                .join(format!("segment_{delete_index}.{ext}"));

            if path.exists() {
                tokio::fs::remove_file(&path).await?;
                debug!("Deleted old segment: {}", path.display());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_generator_creation() {
        let config = SegmentConfig::default();
        let generator = SegmentGenerator::new(config);

        assert_eq!(generator.segment_index(), 0);
    }

    #[test]
    fn test_segment_path_generation() {
        let mut config = SegmentConfig::default();
        config.format = SegmentFormat::MpegTs;

        let generator = SegmentGenerator::new(config);
        let path = generator.get_segment_path();

        assert_eq!(path, "segment_0.ts");
    }

    #[test]
    fn test_mpeg2_crc32_known_answer() {
        // Canonical CRC-32/MPEG-2 check value (CRC RevEng catalogue): the CRC
        // of the ASCII string "123456789" is 0x0376_E6E7. This pins the poly,
        // init value and bit order to the MPEG-2 systems definition.
        assert_eq!(mpeg2_crc32(b"123456789"), 0x0376_E6E7);
    }

    #[test]
    fn test_pat_crc32_computed_and_matches_section() {
        let generator = SegmentGenerator::new(SegmentConfig::default());
        let mut out = BytesMut::new();
        generator
            .write_pat(&mut out)
            .expect("write_pat should succeed");

        // PAT: 4-byte TS header + pointer_field, then a section with
        // section_length = 13 → section body (incl. CRC) spans indices 5..21;
        // the CRC occupies the final 4 bytes at indices 17..21.
        let crc = u32::from_be_bytes([out[17], out[18], out[19], out[20]]);
        assert_ne!(crc, 0, "PAT CRC32 must be computed, not a zero placeholder");
        assert_eq!(
            crc,
            mpeg2_crc32(&out[5..17]),
            "PAT CRC32 must match the section it covers"
        );
    }

    #[test]
    fn test_pmt_crc32_computed_and_matches_section() {
        let generator = SegmentGenerator::new(SegmentConfig::default());
        let mut out = BytesMut::new();
        generator
            .write_pmt(&mut out)
            .expect("write_pmt should succeed");

        // PMT: section_length = 18 → section body (incl. CRC) spans indices
        // 5..26; the CRC occupies the final 4 bytes at indices 22..26.
        let crc = u32::from_be_bytes([out[22], out[23], out[24], out[25]]);
        assert_ne!(crc, 0, "PMT CRC32 must be computed, not a zero placeholder");
        assert_eq!(
            crc,
            mpeg2_crc32(&out[5..22]),
            "PMT CRC32 must match the section it covers"
        );
    }
}
