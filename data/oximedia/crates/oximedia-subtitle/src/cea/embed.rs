//! Video frame embedding for CEA-608/708 closed captions.
//!
//! This module provides functions to embed closed caption data into various
//! video formats according to ATSC A/53 and related standards.

use crate::{SubtitleError, SubtitleResult};

/// Video field for CEA-608 line 21 insertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoField {
    /// Field 1 (odd field).
    Field1,
    /// Field 2 (even field).
    Field2,
}

/// Video format for embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoFormat {
    /// NTSC (525 lines, 29.97 fps).
    NTSC,
    /// PAL (625 lines, 25 fps).
    PAL,
    /// 720p HD.
    HD720p,
    /// 1080i HD.
    HD1080i,
    /// 1080p HD.
    HD1080p,
    /// 4K UHD.
    UHD4K,
}

impl VideoFormat {
    /// Get the total line count for this format.
    #[must_use]
    pub const fn line_count(&self) -> u32 {
        match self {
            Self::NTSC => 525,
            Self::PAL => 625,
            Self::HD720p => 720,
            Self::HD1080i | Self::HD1080p => 1080,
            Self::UHD4K => 2160,
        }
    }

    /// Check if this format is interlaced.
    #[must_use]
    pub const fn is_interlaced(&self) -> bool {
        matches!(self, Self::NTSC | Self::PAL | Self::HD1080i)
    }

    /// Get line 21 position for CEA-608 (field 1).
    #[must_use]
    pub const fn line21_field1(&self) -> Option<u32> {
        match self {
            Self::NTSC => Some(21),
            Self::PAL => Some(22),
            _ => None, // Digital formats use SEI/user data
        }
    }

    /// Get line 21 position for CEA-608 (field 2).
    #[must_use]
    pub const fn line21_field2(&self) -> Option<u32> {
        match self {
            Self::NTSC => Some(284), // Line 21 of field 2
            Self::PAL => Some(335),
            _ => None,
        }
    }
}

/// Frame rate information.
#[derive(Clone, Copy, Debug)]
pub struct FrameRate {
    /// Numerator.
    pub num: u32,
    /// Denominator.
    pub den: u32,
}

impl FrameRate {
    /// Create a new frame rate.
    #[must_use]
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// Get frame rate as floating point.
    #[must_use]
    pub fn as_float(&self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// NTSC frame rate (29.97 fps).
    #[must_use]
    pub const fn ntsc() -> Self {
        Self::new(30000, 1001)
    }

    /// PAL frame rate (25 fps).
    #[must_use]
    pub const fn pal() -> Self {
        Self::new(25, 1)
    }

    /// Film frame rate (23.976 fps).
    #[must_use]
    pub const fn film() -> Self {
        Self::new(24000, 1001)
    }

    /// 24 fps.
    #[must_use]
    pub const fn fps24() -> Self {
        Self::new(24, 1)
    }

    /// 30 fps.
    #[must_use]
    pub const fn fps30() -> Self {
        Self::new(30, 1)
    }

    /// 60 fps.
    #[must_use]
    pub const fn fps60() -> Self {
        Self::new(60, 1)
    }
}

/// CEA-608 Line 21 encoder for analog video.
pub struct Line21Encoder {
    format: VideoFormat,
}

impl Line21Encoder {
    /// Create a new Line 21 encoder.
    ///
    /// # Errors
    ///
    /// Returns error if format doesn't support Line 21.
    pub fn new(format: VideoFormat) -> SubtitleResult<Self> {
        if format.line21_field1().is_none() {
            return Err(SubtitleError::InvalidFormat(
                "Format does not support Line 21 encoding".to_string(),
            ));
        }
        Ok(Self { format })
    }

    /// Encode CEA-608 byte pair into Line 21 waveform data.
    ///
    /// Returns the waveform data for one scan line (Line 21).
    #[must_use]
    pub fn encode_line21(&self, byte1: u8, byte2: u8, field: VideoField) -> Vec<u8> {
        // Line 21 format:
        // - Clock run-in (7 cycles of sine wave)
        // - Start bits (3 bits: 001)
        // - Data byte 1 (8 bits with odd parity)
        // - Data byte 2 (8 bits with odd parity)
        //
        // Each bit is encoded as NRZ (Non-Return to Zero)
        // Logic 1 = white level, Logic 0 = black level

        let mut waveform = Vec::with_capacity(512);

        // Clock run-in (about 7 cycles at 0.5 MHz)
        // Simplified: alternating black/white pattern
        for _ in 0..14 {
            waveform.push(0x00); // Black
            waveform.push(0xFF); // White
        }

        // Start bits (001)
        self.encode_bit(&mut waveform, false);
        self.encode_bit(&mut waveform, false);
        self.encode_bit(&mut waveform, true);

        // Data byte 1
        self.encode_byte(&mut waveform, byte1);

        // Data byte 2
        self.encode_byte(&mut waveform, byte2);

        // Pad to full line width
        while waveform.len() < 512 {
            waveform.push(0x10); // Black level
        }

        waveform
    }

    /// Encode a single bit.
    fn encode_bit(&self, waveform: &mut Vec<u8>, bit: bool) {
        // NRZ encoding: 1 = white (0xFF), 0 = black (0x00)
        let value = if bit { 0xFF } else { 0x00 };

        // Each bit spans several samples (depending on pixel clock)
        for _ in 0..4 {
            waveform.push(value);
        }
    }

    /// Encode a byte (LSB first).
    fn encode_byte(&self, waveform: &mut Vec<u8>, byte: u8) {
        for i in 0..8 {
            let bit = (byte & (1 << i)) != 0;
            self.encode_bit(waveform, bit);
        }
    }

    /// Get the line number for embedding.
    #[must_use]
    pub fn get_line_number(&self, field: VideoField) -> Option<u32> {
        match field {
            VideoField::Field1 => self.format.line21_field1(),
            VideoField::Field2 => self.format.line21_field2(),
        }
    }
}

/// SEI (Supplemental Enhancement Information) message type for H.264/H.265.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeiMessageType {
    /// User data registered by ITU-T T.35.
    UserDataRegistered = 4,
    /// User data unregistered.
    UserDataUnregistered = 5,
}

/// H.264/H.265 SEI NAL unit builder for CEA-708.
pub struct SeiNalBuilder {
    is_h265: bool,
}

impl SeiNalBuilder {
    /// Create a new SEI NAL builder.
    #[must_use]
    pub const fn new(is_h265: bool) -> Self {
        Self { is_h265 }
    }

    /// Build SEI NAL unit containing CEA-708 data.
    #[must_use]
    pub fn build_cea708_sei(&self, cea708_data: &[u8]) -> Vec<u8> {
        let mut sei = Vec::new();

        // NAL unit header
        if self.is_h265 {
            // H.265/HEVC NAL header (2 bytes)
            // NAL unit type = 39 (PREFIX_SEI_NUT), nuh_layer_id = 0, nuh_temporal_id_plus1 = 1
            sei.push(0x4E); // (39 << 1) | 0
            sei.push(0x01);
        } else {
            // H.264 NAL header (1 byte)
            // forbidden_zero_bit = 0, nal_ref_idc = 0, nal_unit_type = 6 (SEI)
            sei.push(0x06);
        }

        // SEI message: User Data Registered (type 4)
        self.write_sei_size(&mut sei, 4);

        // Payload size
        let payload_size = 1 + 2 + 1 + cea708_data.len(); // country + provider + user_id + data
        self.write_sei_size(&mut sei, payload_size);

        // ITU-T T.35 country code (United States = 0xB5)
        sei.push(0xB5);

        // Provider code (ATSC = 0x0031)
        sei.push(0x00);
        sei.push(0x31);

        // User identifier (ATSC1_data = 0x47)
        sei.push(0x47);

        // CEA-708 data
        sei.extend_from_slice(cea708_data);

        // RBSP trailing bits (10000000)
        sei.push(0x80);

        // Emulation prevention (insert 0x03 after 0x00 0x00)
        self.add_emulation_prevention(&mut sei);

        sei
    }

    /// Write size in SEI format (variable length).
    fn write_sei_size(&self, buffer: &mut Vec<u8>, mut size: usize) {
        while size >= 255 {
            buffer.push(0xFF);
            size -= 255;
        }
        buffer.push(size as u8);
    }

    /// Add emulation prevention bytes.
    fn add_emulation_prevention(&self, data: &mut Vec<u8>) {
        let mut i = 2;
        while i < data.len() {
            if data[i - 2] == 0x00 && data[i - 1] == 0x00 && data[i] <= 0x03 {
                // Insert emulation prevention byte
                data.insert(i, 0x03);
                i += 1;
            }
            i += 1;
        }
    }
}

/// MPEG-2 user data builder for CEA-608/708.
pub struct Mpeg2UserDataBuilder;

impl Mpeg2UserDataBuilder {
    /// Build ATSC A/53 user data structure for CEA-608.
    #[must_use]
    pub fn build_cea608_user_data(byte1: u8, byte2: u8) -> Vec<u8> {
        vec![
            // User data start code (0x000001B2)
            0x00, 0x00, 0x01, 0xB2, // ATSC identifier (GA94)
            b'G', b'A', b'9', b'4', // User data type code (0x03 = cc_data)
            0x03,
            // Process cc data flag and cc_count (1 pair)
            0xC1, // process_cc_data=1, cc_count=1
            // Reserved bits
            0xFF,
            // CC data (3 bytes per pair)
            0xFC, // cc_valid=1, cc_type=0 (608, field 1)
            byte1, byte2, // Marker bits (end of cc_data)
            0xFF,
        ]
    }

    /// Build ATSC A/53 user data for CEA-708 (from CDP).
    #[must_use]
    pub fn build_cea708_user_data(cdp: &[u8]) -> Vec<u8> {
        let mut user_data = vec![
            // User data start code
            0x00, 0x00, 0x01, 0xB2, // ATSC identifier
            b'G', b'A', b'9', b'4', // User data type code (0x03 = cc_data)
            0x03,
        ];

        // Add CDP data
        user_data.extend_from_slice(cdp);

        user_data
    }
}

/// A/53 compliance validator.
pub struct A53Validator;

impl A53Validator {
    /// Validate CEA-608 byte pair.
    ///
    /// # Errors
    ///
    /// Returns error if bytes don't comply with A/53.
    pub fn validate_cea608_pair(byte1: u8, byte2: u8) -> SubtitleResult<()> {
        // Check parity bits
        if !Self::check_parity(byte1) {
            return Err(SubtitleError::InvalidFormat(
                "CEA-608 byte 1 parity error".to_string(),
            ));
        }

        if byte2 != 0x80 && !Self::check_parity(byte2) {
            return Err(SubtitleError::InvalidFormat(
                "CEA-608 byte 2 parity error".to_string(),
            ));
        }

        Ok(())
    }

    /// Check odd parity.
    fn check_parity(byte: u8) -> bool {
        let mut parity = 0u8;
        let mut b = byte;

        for _ in 0..8 {
            parity ^= b & 1;
            b >>= 1;
        }

        parity == 1
    }

    /// Validate CDP structure.
    ///
    /// # Errors
    ///
    /// Returns error if CDP is malformed.
    pub fn validate_cdp(cdp: &[u8]) -> SubtitleResult<()> {
        if cdp.len() < 6 {
            return Err(SubtitleError::InvalidFormat("CDP too short".to_string()));
        }

        // Check CDP identifier
        if cdp[0] != 0x96 {
            return Err(SubtitleError::InvalidFormat(
                "Invalid CDP identifier".to_string(),
            ));
        }

        // Check length
        let stated_length = cdp[1] as usize;
        if stated_length + 2 != cdp.len() {
            return Err(SubtitleError::InvalidFormat(format!(
                "CDP length mismatch: stated={stated_length}, actual={}",
                cdp.len() - 2
            )));
        }

        // Verify checksum
        let mut sum = 0u8;
        for &byte in cdp {
            sum = sum.wrapping_add(byte);
        }

        if sum != 0 {
            return Err(SubtitleError::InvalidFormat(format!(
                "CDP checksum error: sum={sum:#04x}"
            )));
        }

        Ok(())
    }
}

/// Frame rate adapter for caption timing.
pub struct FrameRateAdapter {
    source_fps: f64,
    target_fps: f64,
    accumulator: f64,
}

impl FrameRateAdapter {
    /// Create a new frame rate adapter.
    #[must_use]
    pub fn new(source_fps: f64, target_fps: f64) -> Self {
        Self {
            source_fps,
            target_fps,
            accumulator: 0.0,
        }
    }

    /// Check if we should output a caption for this target frame.
    #[must_use]
    pub fn should_output(&mut self) -> bool {
        self.accumulator += self.source_fps;

        if self.accumulator >= self.target_fps {
            self.accumulator -= self.target_fps;
            true
        } else {
            false
        }
    }

    /// Reset the adapter.
    pub fn reset(&mut self) {
        self.accumulator = 0.0;
    }

    /// Calculate frame number in target frame rate from source frame.
    #[must_use]
    pub fn convert_frame_number(&self, source_frame: u64) -> u64 {
        ((source_frame as f64) * self.target_fps / self.source_fps).round() as u64
    }

    /// Convert timestamp (milliseconds) to frame number.
    #[must_use]
    pub fn timestamp_to_frame(timestamp_ms: i64, fps: f64) -> u64 {
        ((timestamp_ms as f64 / 1000.0) * fps).round() as u64
    }

    /// Convert frame number to timestamp (milliseconds).
    #[must_use]
    pub fn frame_to_timestamp(frame: u64, fps: f64) -> i64 {
        ((frame as f64 / fps) * 1000.0).round() as i64
    }
}

/// Caption embedding helper.
pub struct CaptionEmbedder {
    format: VideoFormat,
    frame_rate: FrameRate,
}

impl CaptionEmbedder {
    /// Create a new caption embedder.
    #[must_use]
    pub const fn new(format: VideoFormat, frame_rate: FrameRate) -> Self {
        Self { format, frame_rate }
    }

    /// Embed CEA-608 into video frame (Line 21 for analog).
    ///
    /// # Errors
    ///
    /// Returns error if embedding fails.
    pub fn embed_cea608_line21(
        &self,
        frame_data: &mut [u8],
        byte1: u8,
        byte2: u8,
        field: VideoField,
    ) -> SubtitleResult<()> {
        // Validate input
        A53Validator::validate_cea608_pair(byte1, byte2)?;

        // Create Line 21 encoder
        let encoder = Line21Encoder::new(self.format)?;

        // Encode Line 21 waveform
        let waveform = encoder.encode_line21(byte1, byte2, field);

        // Get line number
        let line_num = encoder
            .get_line_number(field)
            .ok_or_else(|| SubtitleError::InvalidFormat("No Line 21 support".to_string()))?;

        // Calculate offset in frame buffer
        let line_offset = (line_num as usize) * self.get_line_stride();

        // Copy waveform to frame
        let end_offset = (line_offset + waveform.len()).min(frame_data.len());
        if line_offset < frame_data.len() {
            let copy_len = end_offset - line_offset;
            frame_data[line_offset..end_offset].copy_from_slice(&waveform[..copy_len]);
        }

        Ok(())
    }

    /// Get line stride (bytes per line) for the video format.
    #[must_use]
    fn get_line_stride(&self) -> usize {
        match self.format {
            VideoFormat::NTSC | VideoFormat::PAL => 720, // SD resolution
            VideoFormat::HD720p => 1280,
            VideoFormat::HD1080i | VideoFormat::HD1080p => 1920,
            VideoFormat::UHD4K => 3840,
        }
    }

    /// Create SEI NAL unit for H.264/H.265 with CEA-708.
    #[must_use]
    pub fn create_sei_nal(&self, cdp: &[u8], is_h265: bool) -> Vec<u8> {
        let builder = SeiNalBuilder::new(is_h265);
        builder.build_cea708_sei(cdp)
    }

    /// Create MPEG-2 user data for CEA-608.
    #[must_use]
    pub fn create_mpeg2_user_data_608(&self, byte1: u8, byte2: u8) -> Vec<u8> {
        Mpeg2UserDataBuilder::build_cea608_user_data(byte1, byte2)
    }

    /// Create MPEG-2 user data for CEA-708.
    #[must_use]
    pub fn create_mpeg2_user_data_708(&self, cdp: &[u8]) -> Vec<u8> {
        Mpeg2UserDataBuilder::build_cea708_user_data(cdp)
    }

    /// Get frame rate as float.
    #[must_use]
    pub fn get_fps(&self) -> f64 {
        self.frame_rate.as_float()
    }
}

/// Utility to calculate SMPTE timecode.
pub struct TimecodeCalculator {
    frame_rate: FrameRate,
    drop_frame: bool,
}

impl TimecodeCalculator {
    /// Create a new timecode calculator.
    #[must_use]
    pub const fn new(frame_rate: FrameRate, drop_frame: bool) -> Self {
        Self {
            frame_rate,
            drop_frame,
        }
    }

    /// Calculate timecode from frame number.
    #[must_use]
    pub fn frame_to_timecode(&self, frame: u64) -> (u8, u8, u8, u8) {
        let fps = self.frame_rate.as_float().round() as u64;

        let frames_per_minute = fps * 60;
        let frames_per_hour = frames_per_minute * 60;

        let mut frame_num = frame;

        // Drop frame adjustment for 29.97 fps
        if self.drop_frame && (self.frame_rate.num == 30000 && self.frame_rate.den == 1001) {
            // Drop 2 frames every minute except every 10th minute
            let frames_per_10min = frames_per_minute * 10 - 18; // 2 * 9 frames dropped
            let ten_min_groups = frame_num / frames_per_10min;
            let remaining = frame_num % frames_per_10min;

            if remaining >= 2 {
                let minutes_in_group = (remaining - 2) / (frames_per_minute - 2);
                frame_num += ten_min_groups * 18 + minutes_in_group * 2;
            }
        }

        let hours = (frame_num / frames_per_hour) as u8;
        frame_num %= frames_per_hour;

        let minutes = (frame_num / frames_per_minute) as u8;
        frame_num %= frames_per_minute;

        let seconds = (frame_num / fps) as u8;
        let frames = (frame_num % fps) as u8;

        (hours, minutes, seconds, frames)
    }

    /// Encode timecode to 32-bit value for CDP.
    #[must_use]
    pub fn encode_timecode(&self, hours: u8, minutes: u8, seconds: u8, frames: u8) -> u32 {
        let mut tc = 0u32;

        // Encode in BCD format
        tc |= u32::from(Self::to_bcd(frames));
        tc |= u32::from(Self::to_bcd(seconds)) << 8;
        tc |= u32::from(Self::to_bcd(minutes)) << 16;
        tc |= u32::from(Self::to_bcd(hours)) << 24;

        // Set drop frame flag if needed
        if self.drop_frame {
            tc |= 0x8000_0000;
        }

        tc
    }

    /// Convert to BCD (Binary Coded Decimal).
    fn to_bcd(value: u8) -> u8 {
        ((value / 10) << 4) | (value % 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line21_encoding() {
        let encoder = Line21Encoder::new(VideoFormat::NTSC).expect("should succeed in test");
        let waveform = encoder.encode_line21(0xB5, 0xA1, VideoField::Field1);
        assert!(!waveform.is_empty());
        assert_eq!(waveform.len(), 512);
    }

    #[test]
    fn test_sei_builder() {
        let builder = SeiNalBuilder::new(false);
        let sei = builder.build_cea708_sei(&[0x96, 0x69]);
        assert!(!sei.is_empty());
        assert_eq!(sei[0], 0x06); // H.264 SEI NAL type
    }

    #[test]
    fn test_parity_check() {
        assert!(A53Validator::check_parity(0xB5));
        assert!(!A53Validator::check_parity(0x00));
    }

    #[test]
    fn test_frame_rate_conversion() {
        let adapter = FrameRateAdapter::new(30.0, 25.0);
        assert_eq!(adapter.convert_frame_number(30), 25);
        assert_eq!(adapter.convert_frame_number(60), 50);
    }

    #[test]
    fn test_timecode_calculation() {
        let calc = TimecodeCalculator::new(FrameRate::ntsc(), false);
        let (h, m, s, f) = calc.frame_to_timecode(108000); // 1 hour at 30fps
        assert_eq!(h, 1);
        assert_eq!(m, 0);
        assert_eq!(s, 0);
        assert_eq!(f, 0);
    }
}
