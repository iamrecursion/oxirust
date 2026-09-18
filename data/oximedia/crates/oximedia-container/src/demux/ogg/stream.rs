//! Logical stream handling for Ogg.
//!
//! An Ogg physical bitstream can contain multiple logical bitstreams,
//! each identified by a unique serial number. This module provides
//! the [`LogicalStream`] type for managing individual streams.

use oximedia_core::{CodecId, Rational};

/// Logical stream within an Ogg container.
///
/// Each logical stream has a unique serial number and contains
/// a single codec's data. The stream tracks header packets,
/// granule positions, and incomplete packet data.
///
/// # Stream Types
///
/// Ogg supports various codecs with different header requirements:
/// - **Opus**: 2 header packets (`OpusHead` + `OpusTags`)
/// - **Vorbis**: 3 header packets (identification, comment, setup)
/// - **FLAC**: 2 header packets (STREAMINFO + optional metadata)
/// - **Theora**: 3 header packets (identification, comment, setup)
#[derive(Clone, Debug)]
pub struct LogicalStream {
    /// Stream serial number.
    ///
    /// Uniquely identifies this logical stream within the container.
    pub serial: u32,

    /// Codec type.
    pub codec: CodecId,

    /// Stream index in demuxer.
    ///
    /// Zero-based index used for packet routing.
    pub stream_index: usize,

    /// Header data for codec initialization.
    ///
    /// Contains all codec-specific header packets needed for
    /// decoder initialization.
    pub headers: Vec<Vec<u8>>,

    /// Expected number of header packets.
    ///
    /// Varies by codec (Opus: 2, Vorbis: 3, etc.).
    pub header_count: usize,

    /// Granule rate (samples per second for audio).
    ///
    /// Used for timestamp calculation from granule positions.
    pub granule_rate: Rational,

    /// Pre-skip for Opus.
    ///
    /// Number of samples to discard at the beginning.
    /// Only meaningful for Opus streams.
    pub pre_skip: u32,

    /// Current granule position.
    ///
    /// Tracks the last known granule position for timestamp
    /// calculation.
    pub last_granule: u64,

    /// Incomplete packet buffer.
    ///
    /// Holds packet data that spans multiple pages.
    pub packet_buffer: Vec<u8>,
}

impl LogicalStream {
    /// Creates a new logical stream.
    ///
    /// Initializes the stream with codec-appropriate defaults for
    /// header count and granule rate.
    ///
    /// # Arguments
    ///
    /// * `serial` - The stream serial number from the BOS page
    /// * `codec` - The detected codec type
    /// * `stream_index` - The stream's index in the demuxer
    ///
    /// # Example
    ///
    /// ```ignore
    /// let stream = LogicalStream::new(0x12345678, CodecId::Opus, 0);
    /// assert_eq!(stream.header_count, 2); // Opus has 2 headers
    /// ```
    #[must_use]
    pub fn new(serial: u32, codec: CodecId, stream_index: usize) -> Self {
        let (header_count, granule_rate) = match codec {
            CodecId::Opus => (2, Rational::new(48000, 1)),
            CodecId::Vorbis | CodecId::Theora => (3, Rational::new(1, 1)), // Set from header
            CodecId::Flac => (2, Rational::new(1, 1)),                     // Set from header
            _ => (1, Rational::new(1, 1)),
        };

        Self {
            serial,
            codec,
            stream_index,
            headers: Vec::with_capacity(header_count),
            header_count,
            granule_rate,
            pre_skip: 0,
            last_granule: 0,
            packet_buffer: Vec::new(),
        }
    }

    /// Checks if all headers have been received.
    ///
    /// Returns `true` when the stream has received all expected
    /// header packets and is ready to produce media packets.
    #[must_use]
    pub fn headers_complete(&self) -> bool {
        self.headers.len() >= self.header_count
    }

    /// Adds a header packet.
    ///
    /// For Opus streams, also parses the pre-skip value from
    /// the first header.
    ///
    /// # Arguments
    ///
    /// * `data` - The header packet data
    pub fn add_header(&mut self, data: Vec<u8>) {
        if self.codec == CodecId::Opus && self.headers.is_empty() {
            // Parse Opus pre-skip from OpusHead
            if data.len() >= 12 {
                self.pre_skip = u16::from_le_bytes([data[10], data[11]]).into();
            }
            // Parse sample rate (though Opus always uses 48kHz internally)
            if data.len() >= 16 {
                let rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
                if rate > 0 {
                    // Store original sample rate for metadata, but granule is always 48kHz
                    self.granule_rate = Rational::new(48000, 1);
                }
            }
        } else if self.codec == CodecId::Vorbis && self.headers.is_empty() {
            // Parse Vorbis identification header
            if data.len() >= 16 {
                // Bytes 12-15: audio sample rate
                let rate = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
                if rate > 0 {
                    self.granule_rate = Rational::new(i64::from(rate), 1);
                }
            }
        } else if self.codec == CodecId::Flac && self.headers.is_empty() {
            // Parse FLAC STREAMINFO
            // Sample rate is at bytes 10-12 (20 bits)
            if data.len() >= 13 {
                #[allow(clippy::cast_lossless)]
                let rate =
                    ((data[10] as u32) << 12) | ((data[11] as u32) << 4) | ((data[12] as u32) >> 4);
                if rate > 0 {
                    self.granule_rate = Rational::new(i64::from(rate), 1);
                }
            }
        }
        self.headers.push(data);
    }

    /// Converts granule position to timestamp in seconds.
    ///
    /// For Opus streams, subtracts the pre-skip value.
    ///
    /// # Arguments
    ///
    /// * `granule` - The granule position to convert
    ///
    /// # Returns
    ///
    /// The timestamp in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn granule_to_seconds(&self, granule: u64) -> f64 {
        if self.granule_rate.den == 0 || self.granule_rate.num == 0 {
            return 0.0;
        }
        let samples = granule.saturating_sub(u64::from(self.pre_skip));
        samples as f64 / self.granule_rate.num as f64
    }

    /// Converts granule position to timestamp in timebase units.
    ///
    /// # Arguments
    ///
    /// * `granule` - The granule position to convert
    /// * `timebase` - The target timebase
    ///
    /// # Returns
    ///
    /// The timestamp in timebase units.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn granule_to_timebase(&self, granule: u64, timebase: Rational) -> i64 {
        if self.granule_rate.num == 0 {
            return 0;
        }

        let samples = granule.saturating_sub(u64::from(self.pre_skip));

        // Convert: samples / sample_rate * timebase.den / timebase.num
        // = samples * timebase.den / (sample_rate * timebase.num)
        let result = (samples as f64 * timebase.den as f64)
            / (self.granule_rate.num as f64 * timebase.num as f64);
        result as i64
    }

    /// Returns the number of received headers.
    #[must_use]
    pub fn header_count_received(&self) -> usize {
        self.headers.len()
    }

    /// Returns the sample rate in Hz.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn sample_rate(&self) -> u32 {
        if self.granule_rate.den == 0 {
            return 0;
        }
        (self.granule_rate.num / self.granule_rate.den) as u32
    }

    /// Appends data to the incomplete packet buffer.
    ///
    /// Used when a packet spans multiple pages.
    pub fn append_to_buffer(&mut self, data: &[u8]) {
        self.packet_buffer.extend_from_slice(data);
    }

    /// Takes the incomplete packet buffer.
    ///
    /// Returns the buffered data and clears the buffer.
    pub fn take_buffer(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.packet_buffer)
    }

    /// Checks if there is incomplete packet data.
    #[must_use]
    pub fn has_incomplete_packet(&self) -> bool {
        !self.packet_buffer.is_empty()
    }
}

/// Identifies codec from Ogg stream header.
///
/// Examines the first packet of a BOS page to determine the codec type.
///
/// # Arguments
///
/// * `header` - The first packet data from a BOS page
///
/// # Returns
///
/// The detected [`CodecId`] if recognized.
///
/// # Errors
///
/// Returns `None` if the codec cannot be identified.
#[must_use]
pub fn identify_codec(header: &[u8]) -> Option<CodecId> {
    // Opus: starts with "OpusHead"
    if header.len() >= 8 && header[..8] == *b"OpusHead" {
        return Some(CodecId::Opus);
    }

    // Vorbis: starts with 0x01 + "vorbis"
    if header.len() >= 7 && header[0] == 0x01 && header[1..7] == *b"vorbis" {
        return Some(CodecId::Vorbis);
    }

    // FLAC: starts with 0x7F + "FLAC"
    if header.len() >= 5 && header[0] == 0x7F && header[1..5] == *b"FLAC" {
        return Some(CodecId::Flac);
    }

    // Theora: starts with 0x80 + "theora"
    if header.len() >= 7 && header[0] == 0x80 && header[1..7] == *b"theora" {
        return Some(CodecId::Theora);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_opus_stream() {
        let stream = LogicalStream::new(0x1234, CodecId::Opus, 0);
        assert_eq!(stream.serial, 0x1234);
        assert_eq!(stream.codec, CodecId::Opus);
        assert_eq!(stream.stream_index, 0);
        assert_eq!(stream.header_count, 2);
        assert_eq!(stream.granule_rate, Rational::new(48000, 1));
        assert!(!stream.headers_complete());
    }

    #[test]
    fn test_new_vorbis_stream() {
        let stream = LogicalStream::new(0x5678, CodecId::Vorbis, 1);
        assert_eq!(stream.header_count, 3);
    }

    #[test]
    fn test_headers_complete() {
        let mut stream = LogicalStream::new(0x1234, CodecId::Opus, 0);
        assert!(!stream.headers_complete());

        stream.headers.push(vec![1, 2, 3]);
        assert!(!stream.headers_complete());

        stream.headers.push(vec![4, 5, 6]);
        assert!(stream.headers_complete());
    }

    #[test]
    fn test_add_opus_header() {
        let mut stream = LogicalStream::new(0x1234, CodecId::Opus, 0);

        // Minimal OpusHead structure
        let opus_head = vec![
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd', // Magic
            1,    // Version
            2,    // Channel count
            0x38, 0x01, // Pre-skip (312 = 0x0138)
            0x80, 0xBB, 0x00, 0x00, // Sample rate (48000)
            0, 0, // Output gain
            0, // Channel mapping family
        ];

        stream.add_header(opus_head);
        assert_eq!(stream.pre_skip, 312);
        assert_eq!(stream.granule_rate, Rational::new(48000, 1));
    }

    #[test]
    fn test_granule_to_seconds() {
        let mut stream = LogicalStream::new(0x1234, CodecId::Opus, 0);
        stream.pre_skip = 312;

        // 48000 samples = 1 second, minus pre-skip
        let seconds = stream.granule_to_seconds(48000 + 312);
        assert!((seconds - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_granule_to_timebase() {
        let stream = LogicalStream::new(0x1234, CodecId::Opus, 0);

        // 48000 samples = 1 second = 1000 ms in 1/1000 timebase
        let timebase = Rational::new(1, 1000);
        let ts = stream.granule_to_timebase(48000, timebase);
        assert_eq!(ts, 1000);
    }

    #[test]
    fn test_identify_codec_opus() {
        let header = b"OpusHead\x01\x02\x00\x00\x00\x00\x00\x00";
        assert_eq!(identify_codec(header), Some(CodecId::Opus));
    }

    #[test]
    fn test_identify_codec_vorbis() {
        let header = b"\x01vorbis\x00\x00\x00\x00\x02\x44\xAC\x00\x00";
        assert_eq!(identify_codec(header), Some(CodecId::Vorbis));
    }

    #[test]
    fn test_identify_codec_flac() {
        let header = b"\x7FFLAC\x01\x00\x00\x03";
        assert_eq!(identify_codec(header), Some(CodecId::Flac));
    }

    #[test]
    fn test_identify_codec_theora() {
        let header = b"\x80theora\x03\x02\x00";
        assert_eq!(identify_codec(header), Some(CodecId::Theora));
    }

    #[test]
    fn test_identify_codec_unknown() {
        let header = b"unknown codec header";
        assert_eq!(identify_codec(header), None);
    }

    #[test]
    fn test_packet_buffer() {
        let mut stream = LogicalStream::new(0x1234, CodecId::Opus, 0);

        assert!(!stream.has_incomplete_packet());

        stream.append_to_buffer(b"part1");
        assert!(stream.has_incomplete_packet());

        stream.append_to_buffer(b"part2");
        let buffer = stream.take_buffer();
        assert_eq!(buffer, b"part1part2");
        assert!(!stream.has_incomplete_packet());
    }

    #[test]
    fn test_sample_rate() {
        let stream = LogicalStream::new(0x1234, CodecId::Opus, 0);
        assert_eq!(stream.sample_rate(), 48000);

        let mut vorbis = LogicalStream::new(0x5678, CodecId::Vorbis, 1);
        vorbis.granule_rate = Rational::new(44100, 1);
        assert_eq!(vorbis.sample_rate(), 44100);
    }
}
