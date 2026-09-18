//! SMPTE ST 2110-20 uncompressed video over RTP.
//!
//! This module implements SMPTE ST 2110-20 which defines the transport of
//! uncompressed active video over RTP for professional broadcast applications.
//! It supports various pixel formats, resolutions from SD to 8K, and both
//! progressive and interlaced scanning.

use crate::error::{NetError, NetResult};
use crate::smpte2110::rtp::{RtpHeader, RtpPacket, MAX_RTP_PAYLOAD};
use crate::smpte2110::timing::{FrameRate, ScanType};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use std::collections::HashMap;

/// RTP payload type for ST 2110-20 video (dynamic range).
pub const RTP_PAYLOAD_TYPE_VIDEO: u8 = 96;

/// Pixel format for video streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// YCbCr 4:2:2 10-bit (most common broadcast format).
    YCbCr422_10bit,
    /// YCbCr 4:2:2 8-bit.
    YCbCr422_8bit,
    /// YCbCr 4:4:4 10-bit.
    YCbCr444_10bit,
    /// YCbCr 4:4:4 12-bit.
    YCbCr444_12bit,
    /// RGB 10-bit.
    Rgb10bit,
    /// RGB 12-bit.
    Rgb12bit,
    /// RGB 8-bit.
    Rgb8bit,
}

impl PixelFormat {
    /// Gets the sampling structure (e.g., "YCbCr-4:2:2").
    #[must_use]
    pub const fn sampling(&self) -> &'static str {
        match self {
            Self::YCbCr422_10bit | Self::YCbCr422_8bit => "YCbCr-4:2:2",
            Self::YCbCr444_10bit | Self::YCbCr444_12bit => "YCbCr-4:4:4",
            Self::Rgb10bit | Self::Rgb12bit | Self::Rgb8bit => "RGB",
        }
    }

    /// Gets the bit depth.
    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        match self {
            Self::YCbCr422_8bit | Self::Rgb8bit => 8,
            Self::YCbCr422_10bit | Self::YCbCr444_10bit | Self::Rgb10bit => 10,
            Self::YCbCr444_12bit | Self::Rgb12bit => 12,
        }
    }

    /// Gets the number of components per pixel.
    #[must_use]
    pub const fn components(&self) -> u8 {
        match self {
            Self::YCbCr422_10bit | Self::YCbCr422_8bit => 2, // Y and Cb/Cr alternating
            Self::YCbCr444_10bit
            | Self::YCbCr444_12bit
            | Self::Rgb10bit
            | Self::Rgb12bit
            | Self::Rgb8bit => 3,
        }
    }

    /// Gets the pixel group size (pixels per group).
    #[must_use]
    pub const fn pixel_group_size(&self) -> usize {
        match self {
            Self::YCbCr422_10bit | Self::YCbCr422_8bit => 2, // Two pixels (4 samples)
            Self::YCbCr444_10bit
            | Self::YCbCr444_12bit
            | Self::Rgb10bit
            | Self::Rgb12bit
            | Self::Rgb8bit => 1,
        }
    }

    /// Gets the bytes per pixel group.
    #[must_use]
    pub const fn bytes_per_pixel_group(&self) -> usize {
        match self {
            Self::YCbCr422_8bit => 4,  // 2 pixels = Y0 Cb Y1 Cr = 4 bytes
            Self::YCbCr422_10bit => 5, // 2 pixels, 10-bit packed = 5 bytes
            Self::YCbCr444_10bit => 4, // 1 pixel, 3 components, 10-bit = 4 bytes (rounded)
            Self::YCbCr444_12bit => 5, // 1 pixel, 3 components, 12-bit = 4.5 -> 5 bytes
            Self::Rgb8bit => 3,        // R G B = 3 bytes
            Self::Rgb10bit => 4,       // R G B 10-bit = 4 bytes (rounded)
            Self::Rgb12bit => 5,       // R G B 12-bit = 4.5 -> 5 bytes
        }
    }

    /// Gets the SDP format string.
    #[must_use]
    pub const fn sdp_format(&self) -> &'static str {
        match self {
            Self::YCbCr422_10bit => "UYVP", // 10-bit 4:2:2
            Self::YCbCr422_8bit => "UYVY",  // 8-bit 4:2:2
            Self::YCbCr444_10bit => "v210", // 10-bit 4:4:4
            Self::YCbCr444_12bit => "v410", // 12-bit 4:4:4
            Self::Rgb10bit => "r210",       // 10-bit RGB
            Self::Rgb12bit => "R12B",       // 12-bit RGB
            Self::Rgb8bit => "RGB8",        // 8-bit RGB
        }
    }
}

/// Video configuration for ST 2110-20.
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in lines.
    pub height: u32,
    /// Frame rate.
    pub frame_rate: FrameRate,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Scan type.
    pub scan_type: ScanType,
    /// Packets per line (for gapped mode).
    pub packets_per_line: u32,
    /// Maximum payload size (bytes).
    pub max_payload_size: usize,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            scan_type: ScanType::Progressive,
            packets_per_line: 1,
            max_payload_size: MAX_RTP_PAYLOAD,
        }
    }
}

impl VideoConfig {
    /// Calculates the bytes per line.
    #[must_use]
    pub fn bytes_per_line(&self) -> usize {
        let pixels_per_line = self.width as usize;
        let pg_size = self.pixel_format.pixel_group_size();
        let bytes_per_pg = self.pixel_format.bytes_per_pixel_group();

        (pixels_per_line / pg_size) * bytes_per_pg
    }

    /// Calculates the number of active lines per frame.
    #[must_use]
    pub fn active_lines(&self) -> u32 {
        match self.scan_type {
            ScanType::Progressive => self.height,
            ScanType::InterlacedField1 | ScanType::InterlacedField2 => self.height / 2,
        }
    }

    /// Calculates the total frame size in bytes.
    #[must_use]
    pub fn frame_size_bytes(&self) -> usize {
        self.bytes_per_line() * self.active_lines() as usize
    }

    /// Calculates the bitrate in bits per second.
    #[must_use]
    pub fn bitrate(&self) -> u64 {
        let frame_size = self.frame_size_bytes() as u64;
        let fps = self.frame_rate.as_f64();
        (frame_size * 8) * (fps as u64)
    }
}

/// RTP header extension for video (RFC 8331).
///
/// Contains scan line information and field identification.
#[derive(Debug, Clone, Copy)]
pub struct VideoHeaderExtension {
    /// Extended sequence number (16 bits).
    pub extended_sequence: u16,
    /// Line number (15 bits) and field ID (1 bit).
    pub line_and_field: u16,
    /// Line offset (12 bits) and continuation (4 bits).
    pub offset_and_continuation: u16,
}

impl VideoHeaderExtension {
    /// Creates a new video header extension.
    #[must_use]
    pub const fn new(line_number: u16, field_id: bool, offset: u16, continuation: bool) -> Self {
        let line_and_field = (line_number & 0x7FFF) | (if field_id { 0x8000 } else { 0 });
        let offset_and_continuation =
            ((offset & 0x0FFF) << 4) | (if continuation { 0x01 } else { 0 });

        Self {
            extended_sequence: 0,
            line_and_field,
            offset_and_continuation,
        }
    }

    /// Gets the line number.
    #[must_use]
    pub const fn line_number(&self) -> u16 {
        self.line_and_field & 0x7FFF
    }

    /// Gets the field ID.
    #[must_use]
    pub const fn field_id(&self) -> bool {
        (self.line_and_field & 0x8000) != 0
    }

    /// Gets the offset.
    #[must_use]
    pub const fn offset(&self) -> u16 {
        (self.offset_and_continuation >> 4) & 0x0FFF
    }

    /// Checks if this is a continuation packet.
    #[must_use]
    pub const fn is_continuation(&self) -> bool {
        (self.offset_and_continuation & 0x01) != 0
    }

    /// Serializes to bytes (6 bytes).
    pub fn serialize(&self, buf: &mut BytesMut) {
        buf.put_u16(self.extended_sequence);
        buf.put_u16(self.line_and_field);
        buf.put_u16(self.offset_and_continuation);
    }

    /// Parses from bytes.
    pub fn parse(data: &[u8]) -> NetResult<Self> {
        if data.len() < 6 {
            return Err(NetError::parse(0, "Video header extension too short"));
        }

        let mut cursor = &data[..];
        let extended_sequence = cursor.get_u16();
        let line_and_field = cursor.get_u16();
        let offset_and_continuation = cursor.get_u16();

        Ok(Self {
            extended_sequence,
            line_and_field,
            offset_and_continuation,
        })
    }
}

/// Video packet containing scan line data.
#[derive(Debug, Clone)]
pub struct VideoPacket {
    /// RTP header.
    pub header: RtpHeader,
    /// Video header extension.
    pub video_extension: VideoHeaderExtension,
    /// Scan line data.
    pub line_data: Bytes,
}

impl VideoPacket {
    /// Creates a new video packet.
    #[must_use]
    pub fn new(header: RtpHeader, video_extension: VideoHeaderExtension, line_data: Bytes) -> Self {
        Self {
            header,
            video_extension,
            line_data,
        }
    }

    /// Parses a video packet from RTP packet.
    pub fn from_rtp(rtp_packet: &RtpPacket) -> NetResult<Self> {
        // Extract video header extension
        let ext_data = rtp_packet
            .header
            .extension_data
            .as_ref()
            .ok_or_else(|| NetError::protocol("Missing video header extension"))?;

        let video_extension = VideoHeaderExtension::parse(&ext_data.data)?;

        Ok(Self {
            header: rtp_packet.header.clone(),
            video_extension,
            line_data: rtp_packet.payload.clone(),
        })
    }

    /// Converts to RTP packet.
    #[must_use]
    pub fn to_rtp(&self) -> RtpPacket {
        let mut ext_data = BytesMut::with_capacity(6);
        self.video_extension.serialize(&mut ext_data);

        let mut header = self.header.clone();
        header.extension = true;
        header.extension_data = Some(crate::smpte2110::rtp::RtpHeaderExtension {
            profile: 0x0100, // ST 2110 extension profile
            data: ext_data.freeze(),
        });

        RtpPacket {
            header,
            payload: self.line_data.clone(),
        }
    }
}

/// Video encoder for ST 2110-20.
pub struct VideoEncoder {
    /// Configuration.
    config: VideoConfig,
    /// Current RTP timestamp.
    current_timestamp: u32,
    /// Current sequence number.
    sequence_number: u16,
    /// SSRC.
    ssrc: u32,
}

impl VideoEncoder {
    /// Creates a new video encoder.
    #[must_use]
    pub fn new(config: VideoConfig, ssrc: u32) -> Self {
        Self {
            config,
            current_timestamp: rand::random(),
            sequence_number: rand::random(),
            ssrc,
        }
    }

    /// Encodes a video frame into RTP packets.
    ///
    /// The frame data should be raw uncompressed video in the configured pixel format.
    pub fn encode_frame(&mut self, frame_data: &[u8]) -> NetResult<Vec<VideoPacket>> {
        let expected_size = self.config.frame_size_bytes();
        if frame_data.len() != expected_size {
            return Err(NetError::protocol(format!(
                "Frame size mismatch: expected {} bytes, got {}",
                expected_size,
                frame_data.len()
            )));
        }

        let mut packets = Vec::new();
        let bytes_per_line = self.config.bytes_per_line();
        let active_lines = self.config.active_lines();

        // Determine field ID for interlaced
        let field_id = matches!(self.config.scan_type, ScanType::InterlacedField2);

        // Encode each line
        for line_num in 0..active_lines {
            let line_start = (line_num as usize) * bytes_per_line;
            let line_end = line_start + bytes_per_line;
            let line_data = &frame_data[line_start..line_end];

            // Split line into packets if needed
            let max_payload = self.config.max_payload_size;
            let packets_this_line = (bytes_per_line + max_payload - 1) / max_payload;

            for packet_idx in 0..packets_this_line {
                let offset = packet_idx * max_payload;
                let remaining = bytes_per_line - offset;
                let payload_size = remaining.min(max_payload);
                let payload = Bytes::copy_from_slice(&line_data[offset..offset + payload_size]);

                let continuation = packet_idx > 0;
                let marker = packet_idx == packets_this_line - 1 && line_num == active_lines - 1;

                let video_ext = VideoHeaderExtension::new(
                    line_num as u16,
                    field_id,
                    offset as u16,
                    continuation,
                );

                let header = RtpHeader {
                    padding: false,
                    extension: true,
                    csrc_count: 0,
                    marker,
                    payload_type: RTP_PAYLOAD_TYPE_VIDEO,
                    sequence_number: self.sequence_number,
                    timestamp: self.current_timestamp,
                    ssrc: self.ssrc,
                    csrcs: Vec::new(),
                    extension_data: None, // Will be set by to_rtp()
                };

                packets.push(VideoPacket::new(header, video_ext, payload));

                self.sequence_number = self.sequence_number.wrapping_add(1);
            }
        }

        // Advance timestamp for next frame
        let timestamp_increment =
            (90000 * self.config.frame_rate.denominator) / self.config.frame_rate.numerator;
        self.current_timestamp = self.current_timestamp.wrapping_add(timestamp_increment);

        Ok(packets)
    }

    /// Gets the current RTP timestamp.
    #[must_use]
    pub const fn current_timestamp(&self) -> u32 {
        self.current_timestamp
    }

    /// Gets the configuration.
    #[must_use]
    pub const fn config(&self) -> &VideoConfig {
        &self.config
    }
}

/// Video decoder for ST 2110-20.
#[derive(Debug)]
pub struct VideoDecoder {
    /// Configuration.
    config: VideoConfig,
    /// Frame assembly buffer.
    frame_buffer: HashMap<u32, FrameAssembler>,
    /// Maximum number of frames to buffer.
    max_buffered_frames: usize,
}

impl VideoDecoder {
    /// Creates a new video decoder.
    #[must_use]
    pub fn new(config: VideoConfig) -> Self {
        Self {
            config,
            frame_buffer: HashMap::new(),
            max_buffered_frames: 4,
        }
    }

    /// Processes an RTP packet.
    pub fn process_rtp_packet(&mut self, rtp_packet: &RtpPacket) -> NetResult<()> {
        let video_packet = VideoPacket::from_rtp(rtp_packet)?;
        let timestamp = video_packet.header.timestamp;

        // Get or create frame assembler
        let assembler = self
            .frame_buffer
            .entry(timestamp)
            .or_insert_with(|| FrameAssembler::new(self.config.clone()));

        assembler.add_packet(video_packet)?;

        // Limit buffer size
        if self.frame_buffer.len() > self.max_buffered_frames {
            // Remove oldest frame (lowest timestamp)
            if let Some(oldest_ts) = self.frame_buffer.keys().min().copied() {
                self.frame_buffer.remove(&oldest_ts);
            }
        }

        Ok(())
    }

    /// Retrieves a completed frame if available.
    pub fn get_frame(&mut self, timestamp: u32) -> Option<Vec<u8>> {
        if let Some(assembler) = self.frame_buffer.get(&timestamp) {
            if assembler.is_complete() {
                return self
                    .frame_buffer
                    .remove(&timestamp)
                    .and_then(|a| a.get_frame());
            }
        }
        None
    }

    /// Gets all completed frames.
    pub fn get_completed_frames(&mut self) -> Vec<(u32, Vec<u8>)> {
        let mut frames = Vec::new();
        let completed: Vec<u32> = self
            .frame_buffer
            .iter()
            .filter(|(_, a)| a.is_complete())
            .map(|(ts, _)| *ts)
            .collect();

        for ts in completed {
            if let Some(assembler) = self.frame_buffer.remove(&ts) {
                if let Some(frame) = assembler.get_frame() {
                    frames.push((ts, frame));
                }
            }
        }

        frames
    }

    /// Gets the configuration.
    #[must_use]
    pub const fn config(&self) -> &VideoConfig {
        &self.config
    }
}

/// Frame assembler for reconstructing frames from RTP packets.
#[derive(Debug)]
struct FrameAssembler {
    /// Configuration.
    config: VideoConfig,
    /// Line buffers.
    lines: HashMap<u16, Vec<u8>>,
    /// Expected number of lines.
    expected_lines: u32,
    /// Marker received flag.
    marker_received: bool,
}

impl FrameAssembler {
    /// Creates a new frame assembler.
    fn new(config: VideoConfig) -> Self {
        let expected_lines = config.active_lines();

        Self {
            config,
            lines: HashMap::new(),
            expected_lines,
            marker_received: false,
        }
    }

    /// Adds a video packet to the frame.
    fn add_packet(&mut self, packet: VideoPacket) -> NetResult<()> {
        let line_num = packet.video_extension.line_number();
        let offset = packet.video_extension.offset() as usize;

        // Get or create line buffer
        let line_buffer = self
            .lines
            .entry(line_num)
            .or_insert_with(|| vec![0u8; self.config.bytes_per_line()]);

        // Copy packet data into line buffer
        let data_len = packet.line_data.len();
        if offset + data_len <= line_buffer.len() {
            line_buffer[offset..offset + data_len].copy_from_slice(&packet.line_data);
        } else {
            return Err(NetError::protocol("Packet data exceeds line buffer"));
        }

        if packet.header.marker {
            self.marker_received = true;
        }

        Ok(())
    }

    /// Checks if the frame is complete.
    fn is_complete(&self) -> bool {
        self.marker_received && self.lines.len() == self.expected_lines as usize
    }

    /// Gets the assembled frame.
    fn get_frame(self) -> Option<Vec<u8>> {
        if !self.is_complete() {
            return None;
        }

        let mut frame = Vec::with_capacity(self.config.frame_size_bytes());

        // Assemble lines in order
        for line_num in 0..self.expected_lines {
            if let Some(line_data) = self.lines.get(&(line_num as u16)) {
                frame.extend_from_slice(line_data);
            } else {
                return None; // Missing line
            }
        }

        Some(frame)
    }
}

/// Pixel group packing utilities.
pub mod packing {
    /// Packs YCbCr 4:2:2 10-bit pixels (2 pixels per group).
    ///
    /// Input: [Y0(10), Cb(10), Y1(10), Cr(10)] as u16 values
    /// Output: 5 bytes packed
    pub fn pack_ycbcr422_10bit(y0: u16, cb: u16, y1: u16, cr: u16) -> [u8; 5] {
        let mut packed = [0u8; 5];

        // Pack 4 10-bit values into 5 bytes
        packed[0] = (cb >> 2) as u8;
        packed[1] = ((cb & 0x03) << 6 | (y0 >> 4)) as u8;
        packed[2] = ((y0 & 0x0F) << 4 | (cr >> 6)) as u8;
        packed[3] = ((cr & 0x3F) << 2 | (y1 >> 8)) as u8;
        packed[4] = (y1 & 0xFF) as u8;

        packed
    }

    /// Unpacks YCbCr 4:2:2 10-bit pixels.
    ///
    /// Input: 5 packed bytes
    /// Output: (Y0, Cb, Y1, Cr) as u16 values
    #[must_use]
    pub fn unpack_ycbcr422_10bit(packed: &[u8; 5]) -> (u16, u16, u16, u16) {
        let cb = (u16::from(packed[0]) << 2) | (u16::from(packed[1]) >> 6);
        let y0 = ((u16::from(packed[1]) & 0x3F) << 4) | (u16::from(packed[2]) >> 4);
        let cr = ((u16::from(packed[2]) & 0x0F) << 6) | (u16::from(packed[3]) >> 2);
        let y1 = ((u16::from(packed[3]) & 0x03) << 8) | u16::from(packed[4]);

        (y0, cb, y1, cr)
    }

    /// Packs RGB 10-bit pixels.
    ///
    /// Input: (R, G, B) as u16 values (10-bit)
    /// Output: 4 bytes packed (30 bits -> 4 bytes with 2 bits padding)
    pub fn pack_rgb_10bit(r: u16, g: u16, b: u16) -> [u8; 4] {
        let mut packed = [0u8; 4];

        packed[0] = (r >> 2) as u8;
        packed[1] = ((r & 0x03) << 6 | (g >> 4)) as u8;
        packed[2] = ((g & 0x0F) << 4 | (b >> 6)) as u8;
        packed[3] = ((b & 0x3F) << 2) as u8;

        packed
    }

    /// Unpacks RGB 10-bit pixels.
    #[must_use]
    pub fn unpack_rgb_10bit(packed: &[u8; 4]) -> (u16, u16, u16) {
        let r = (u16::from(packed[0]) << 2) | (u16::from(packed[1]) >> 6);
        let g = ((u16::from(packed[1]) & 0x3F) << 4) | (u16::from(packed[2]) >> 4);
        let b = ((u16::from(packed[2]) & 0x0F) << 6) | (u16::from(packed[3]) >> 2);

        (r, g, b)
    }
}

/// Timing Reference Signals (TRS) for video synchronization.
pub mod trs {
    /// TRS sequence for Start of Active Video (SAV).
    pub const SAV: [u8; 4] = [0xFF, 0x00, 0x00, 0xAB];

    /// TRS sequence for End of Active Video (EAV).
    pub const EAV: [u8; 4] = [0xFF, 0x00, 0x00, 0x9D];

    /// Checks if data starts with SAV.
    #[must_use]
    pub fn is_sav(data: &[u8]) -> bool {
        data.len() >= 4 && data[0..4] == SAV
    }

    /// Checks if data starts with EAV.
    #[must_use]
    pub fn is_eav(data: &[u8]) -> bool {
        data.len() >= 4 && data[0..4] == EAV
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_properties() {
        let fmt = PixelFormat::YCbCr422_10bit;
        assert_eq!(fmt.sampling(), "YCbCr-4:2:2");
        assert_eq!(fmt.bit_depth(), 10);
        assert_eq!(fmt.pixel_group_size(), 2);
        assert_eq!(fmt.bytes_per_pixel_group(), 5);
    }

    #[test]
    fn test_video_config() {
        let config = VideoConfig {
            width: 1920,
            height: 1080,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            scan_type: ScanType::Progressive,
            packets_per_line: 1,
            max_payload_size: MAX_RTP_PAYLOAD,
        };

        assert_eq!(config.bytes_per_line(), 4800); // 1920 / 2 * 5
        assert_eq!(config.active_lines(), 1080);
        assert!(config.bitrate() > 0);
    }

    #[test]
    fn test_video_header_extension() {
        let ext = VideoHeaderExtension::new(100, true, 256, false);
        assert_eq!(ext.line_number(), 100);
        assert!(ext.field_id());
        assert_eq!(ext.offset(), 256);
        assert!(!ext.is_continuation());

        let mut buf = BytesMut::new();
        ext.serialize(&mut buf);
        assert_eq!(buf.len(), 6);
    }

    #[test]
    fn test_ycbcr422_10bit_packing() {
        let y0 = 512;
        let cb = 256;
        let y1 = 768;
        let cr = 128;

        let packed = packing::pack_ycbcr422_10bit(y0, cb, y1, cr);
        let (y0_out, cb_out, y1_out, cr_out) = packing::unpack_ycbcr422_10bit(&packed);

        assert_eq!(y0, y0_out);
        assert_eq!(cb, cb_out);
        assert_eq!(y1, y1_out);
        assert_eq!(cr, cr_out);
    }

    #[test]
    fn test_rgb_10bit_packing() {
        let r = 512;
        let g = 768;
        let b = 256;

        let packed = packing::pack_rgb_10bit(r, g, b);
        let (r_out, g_out, b_out) = packing::unpack_rgb_10bit(&packed);

        assert_eq!(r, r_out);
        assert_eq!(g, g_out);
        assert_eq!(b, b_out);
    }

    #[test]
    fn test_trs_detection() {
        assert!(trs::is_sav(&trs::SAV));
        assert!(trs::is_eav(&trs::EAV));
        assert!(!trs::is_sav(&[0, 0, 0, 0]));
    }

    #[test]
    fn test_video_encoder() {
        let config = VideoConfig {
            width: 320,
            height: 240,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            scan_type: ScanType::Progressive,
            packets_per_line: 1,
            max_payload_size: MAX_RTP_PAYLOAD,
        };

        let mut encoder = VideoEncoder::new(config.clone(), 12345);
        let frame_size = config.frame_size_bytes();
        let frame_data = vec![0u8; frame_size];

        let packets = encoder
            .encode_frame(&frame_data)
            .expect("should succeed in test");
        assert!(!packets.is_empty());
        assert_eq!(packets.len(), config.active_lines() as usize);
    }

    #[test]
    fn test_frame_assembler() {
        let config = VideoConfig {
            width: 320,
            height: 240,
            frame_rate: FrameRate::FPS_25,
            pixel_format: PixelFormat::YCbCr422_10bit,
            scan_type: ScanType::Progressive,
            packets_per_line: 1,
            max_payload_size: MAX_RTP_PAYLOAD,
        };

        let mut assembler = FrameAssembler::new(config.clone());
        assert!(!assembler.is_complete());

        // Add all lines
        for line_num in 0..config.active_lines() {
            let line_data = vec![0u8; config.bytes_per_line()];
            let ext = VideoHeaderExtension::new(line_num as u16, false, 0, false);
            let header = RtpHeader {
                padding: false,
                extension: true,
                csrc_count: 0,
                marker: line_num == config.active_lines() - 1,
                payload_type: RTP_PAYLOAD_TYPE_VIDEO,
                sequence_number: line_num as u16,
                timestamp: 0,
                ssrc: 12345,
                csrcs: Vec::new(),
                extension_data: None,
            };

            let packet = VideoPacket::new(header, ext, Bytes::from(line_data));
            assembler
                .add_packet(packet)
                .expect("should succeed in test");
        }

        assert!(assembler.is_complete());
        let frame = assembler.get_frame();
        assert!(frame.is_some());
    }
}
