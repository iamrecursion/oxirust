//! Vertical Interval Timecode (VITC) reading and writing
//!
//! VITC encodes timecode in the vertical blanking interval of a video signal.
//! Typically encoded on lines 10-20 of fields 1 and 2.
//!
//! # VITC Structure
//! - 90 bits total per line
//! - Each bit is 2 pixels wide (for NTSC/PAL)
//! - Contains timecode, user bits, and CRC
//!
//! # Advantages over LTC
//! - Readable at zero speed and in pause
//! - Not affected by audio dropouts
//! - Can be read from both fields
//!
//! # Encoding
//! - Bits encoded as luminance levels (black/white)
//! - Bit 0: Black level
//! - Bit 1: White level
//! - Sync bits mark start and end of data

pub mod decoder;
pub mod encoder;
pub mod smpte309m;

pub use smpte309m::{decode_anc_timecode, encode_anc_timecode, Smpte309mPacket};

use crate::{FrameRate, Timecode, TimecodeError, TimecodeReader, TimecodeWriter};

/// VITC reader configuration
#[derive(Debug, Clone)]
pub struct VitcReaderConfig {
    /// Video standard
    pub video_standard: VideoStandard,
    /// Frame rate
    pub frame_rate: FrameRate,
    /// Scan lines to read (typically 10-20)
    pub scan_lines: Vec<u16>,
    /// Field preference (F1, F2, or both)
    pub field_preference: FieldPreference,
}

impl Default for VitcReaderConfig {
    fn default() -> Self {
        VitcReaderConfig {
            video_standard: VideoStandard::Pal,
            frame_rate: FrameRate::Fps25,
            scan_lines: vec![19, 21], // Common VITC lines for PAL
            field_preference: FieldPreference::Both,
        }
    }
}

/// Video standard
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoStandard {
    /// NTSC (525 lines, 60Hz field rate)
    Ntsc,
    /// PAL (625 lines, 50Hz field rate)
    Pal,
}

impl VideoStandard {
    /// Get total lines per frame
    pub fn total_lines(&self) -> u16 {
        match self {
            VideoStandard::Ntsc => 525,
            VideoStandard::Pal => 625,
        }
    }

    /// Get active video lines
    pub fn active_lines(&self) -> u16 {
        match self {
            VideoStandard::Ntsc => 486,
            VideoStandard::Pal => 576,
        }
    }

    /// Get pixels per line (for digital video)
    pub fn pixels_per_line(&self) -> u16 {
        match self {
            VideoStandard::Ntsc => 720,
            VideoStandard::Pal => 720,
        }
    }
}

/// Field preference for VITC reading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldPreference {
    /// Read from field 1 only
    Field1,
    /// Read from field 2 only
    Field2,
    /// Read from both fields (use first valid)
    Both,
}

/// VITC reader
pub struct VitcReader {
    decoder: decoder::VitcDecoder,
    frame_rate: FrameRate,
}

impl VitcReader {
    /// Create a new VITC reader with configuration
    pub fn new(config: VitcReaderConfig) -> Self {
        let frame_rate = config.frame_rate;
        VitcReader {
            decoder: decoder::VitcDecoder::new(config),
            frame_rate,
        }
    }

    /// Process a video line and attempt to decode VITC
    pub fn process_line(
        &mut self,
        line_number: u16,
        field: u8,
        pixels: &[u8],
    ) -> Result<Option<Timecode>, TimecodeError> {
        self.decoder.process_line(line_number, field, pixels)
    }

    /// Reset the decoder state
    pub fn reset(&mut self) {
        self.decoder.reset();
    }

    /// Get CRC error count
    pub fn crc_errors(&self) -> u32 {
        self.decoder.crc_errors()
    }
}

impl TimecodeReader for VitcReader {
    fn read_timecode(&mut self) -> Result<Option<Timecode>, TimecodeError> {
        // In practice, video lines would be fed externally
        Ok(None)
    }

    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    fn is_synchronized(&self) -> bool {
        self.decoder.is_synchronized()
    }
}

/// VITC writer configuration
#[derive(Debug, Clone)]
pub struct VitcWriterConfig {
    /// Video standard
    pub video_standard: VideoStandard,
    /// Frame rate
    pub frame_rate: FrameRate,
    /// Scan lines to write (typically 19 and 21 for PAL)
    pub scan_lines: Vec<u16>,
    /// Write to both fields
    pub both_fields: bool,
}

impl Default for VitcWriterConfig {
    fn default() -> Self {
        VitcWriterConfig {
            video_standard: VideoStandard::Pal,
            frame_rate: FrameRate::Fps25,
            scan_lines: vec![19, 21],
            both_fields: true,
        }
    }
}

/// VITC writer
pub struct VitcWriter {
    encoder: encoder::VitcEncoder,
    frame_rate: FrameRate,
}

impl VitcWriter {
    /// Create a new VITC writer with configuration
    pub fn new(config: VitcWriterConfig) -> Self {
        let frame_rate = config.frame_rate;
        VitcWriter {
            encoder: encoder::VitcEncoder::new(config),
            frame_rate,
        }
    }

    /// Encode a timecode to VITC pixel data
    pub fn encode_line(
        &mut self,
        timecode: &Timecode,
        field: u8,
    ) -> Result<Vec<u8>, TimecodeError> {
        self.encoder.encode_line(timecode, field)
    }

    /// Reset the encoder state
    pub fn reset(&mut self) {
        self.encoder.reset();
    }
}

impl TimecodeWriter for VitcWriter {
    fn write_timecode(&mut self, timecode: &Timecode) -> Result<(), TimecodeError> {
        // Encode for field 1
        let _pixels_f1 = self.encode_line(timecode, 1)?;
        // Encode for field 2
        let _pixels_f2 = self.encode_line(timecode, 2)?;
        // In a real implementation, pixels would be written to video output
        Ok(())
    }

    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    fn flush(&mut self) -> Result<(), TimecodeError> {
        Ok(())
    }
}

/// VITC bit patterns and constants
pub(crate) mod constants {
    /// Number of bits in a VITC line
    pub const BITS_PER_LINE: usize = 90;

    /// Number of data bits (timecode + user bits + CRC)
    pub const DATA_BITS: usize = 82;

    /// Number of sync bits at start
    pub const SYNC_START_BITS: usize = 2;

    /// Number of sync bits at end
    #[allow(dead_code)]
    pub const SYNC_END_BITS: usize = 6;

    /// Pixels per bit (typically 2)
    pub const PIXELS_PER_BIT: usize = 2;

    /// Black level (bit 0)
    pub const BLACK_LEVEL: u8 = 16;

    /// White level (bit 1)
    pub const WHITE_LEVEL: u8 = 235;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vitc_reader_creation() {
        let config = VitcReaderConfig::default();
        let _reader = VitcReader::new(config);
    }

    #[test]
    fn test_vitc_writer_creation() {
        let config = VitcWriterConfig::default();
        let _writer = VitcWriter::new(config);
    }

    #[test]
    fn test_video_standard() {
        assert_eq!(VideoStandard::Ntsc.total_lines(), 525);
        assert_eq!(VideoStandard::Pal.total_lines(), 625);
        assert_eq!(VideoStandard::Ntsc.pixels_per_line(), 720);
    }

    #[test]
    fn test_constants() {
        assert_eq!(constants::BITS_PER_LINE, 90);
        assert_eq!(constants::PIXELS_PER_BIT, 2);
    }
}
