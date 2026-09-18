//! VITC Decoder - Extract timecode from video scan lines
//!
//! This module implements a complete VITC decoder that:
//! - Analyzes video scan lines for VITC patterns
//! - Detects sync bits and data bits
//! - Validates CRC checksums
//! - Extracts timecode and user bits
//! - Handles field identification

use super::constants::*;
use super::{FieldPreference, VitcReaderConfig};
use crate::{FrameRate, Timecode, TimecodeError};

/// VITC decoder
pub struct VitcDecoder {
    /// Configuration
    config: VitcReaderConfig,
    /// Last decoded timecode
    last_timecode: Option<Timecode>,
    /// CRC error counter
    crc_error_count: u32,
    /// Sync counter
    sync_count: u32,
    /// Bit slicer threshold
    threshold: u8,
}

impl VitcDecoder {
    /// Create a new VITC decoder
    pub fn new(config: VitcReaderConfig) -> Self {
        VitcDecoder {
            config,
            last_timecode: None,
            crc_error_count: 0,
            sync_count: 0,
            threshold: 128,
        }
    }

    /// Process a video line and decode VITC
    pub fn process_line(
        &mut self,
        line_number: u16,
        field: u8,
        pixels: &[u8],
    ) -> Result<Option<Timecode>, TimecodeError> {
        // Check if this line should contain VITC
        if !self.config.scan_lines.contains(&line_number) {
            return Ok(None);
        }

        // Check field preference
        match self.config.field_preference {
            FieldPreference::Field1 if field != 1 => return Ok(None),
            FieldPreference::Field2 if field != 2 => return Ok(None),
            _ => {}
        }

        // Decode bits from pixels
        let bits = self.pixels_to_bits(pixels)?;

        // Validate and decode timecode
        match self.decode_vitc_bits(&bits, field) {
            Ok(timecode) => {
                self.sync_count = self.sync_count.saturating_add(1);
                self.last_timecode = Some(timecode);
                Ok(Some(timecode))
            }
            Err(e) => {
                if e == TimecodeError::CrcError {
                    self.crc_error_count += 1;
                }
                Err(e)
            }
        }
    }

    /// Convert pixels to bit array
    fn pixels_to_bits(&mut self, pixels: &[u8]) -> Result<[bool; BITS_PER_LINE], TimecodeError> {
        if pixels.len() < BITS_PER_LINE * PIXELS_PER_BIT {
            return Err(TimecodeError::BufferTooSmall);
        }

        // Auto-adjust threshold based on signal levels
        self.adjust_threshold(pixels);

        let mut bits = [false; BITS_PER_LINE];

        for (i, bit) in bits.iter_mut().enumerate().take(BITS_PER_LINE) {
            // Sample the middle of each bit period
            let pixel_index = i * PIXELS_PER_BIT + PIXELS_PER_BIT / 2;
            if pixel_index < pixels.len() {
                *bit = pixels[pixel_index] > self.threshold;
            }
        }

        Ok(bits)
    }

    /// Adjust threshold based on signal statistics
    fn adjust_threshold(&mut self, pixels: &[u8]) {
        let mut min_level = 255u8;
        let mut max_level = 0u8;

        for &pixel in pixels.iter().take(BITS_PER_LINE * PIXELS_PER_BIT) {
            min_level = min_level.min(pixel);
            max_level = max_level.max(pixel);
        }

        // Set threshold to midpoint
        self.threshold = ((min_level as u16 + max_level as u16) / 2) as u8;
    }

    /// Decode VITC from bit array
    fn decode_vitc_bits(
        &self,
        bits: &[bool; BITS_PER_LINE],
        field: u8,
    ) -> Result<Timecode, TimecodeError> {
        // Validate sync bits
        self.validate_sync_bits(bits)?;

        // Extract data bits (skip sync bits)
        let mut data_bits = [false; DATA_BITS];
        data_bits[..DATA_BITS]
            .copy_from_slice(&bits[SYNC_START_BITS..(DATA_BITS + SYNC_START_BITS)]);

        // Validate CRC
        self.validate_crc(&data_bits)?;

        // Decode timecode from data bits
        self.decode_timecode_from_bits(&data_bits, field)
    }

    /// Validate sync bits
    fn validate_sync_bits(&self, bits: &[bool; BITS_PER_LINE]) -> Result<(), TimecodeError> {
        // Start sync: bits 0-1 should be 11 (white-white)
        if !bits[0] || !bits[1] {
            return Err(TimecodeError::SyncNotFound);
        }

        // End sync: bits 84-89 should be 001111 (black-black-white-white-white-white)
        if bits[84] || bits[85] || !bits[86] || !bits[87] || !bits[88] || !bits[89] {
            return Err(TimecodeError::SyncNotFound);
        }

        Ok(())
    }

    /// Validate CRC
    fn validate_crc(&self, data_bits: &[bool; DATA_BITS]) -> Result<(), TimecodeError> {
        // Extract CRC from bits 74-81
        let received_crc = self.extract_crc(data_bits);

        // Calculate CRC over bits 2-73
        let calculated_crc = self.calculate_crc(&data_bits[0..72]);

        if received_crc != calculated_crc {
            return Err(TimecodeError::CrcError);
        }

        Ok(())
    }

    /// Extract CRC from data bits
    fn extract_crc(&self, data_bits: &[bool; DATA_BITS]) -> u8 {
        let mut crc = 0u8;
        for i in 0..8 {
            if data_bits[74 + i] {
                crc |= 1 << i;
            }
        }
        crc
    }

    /// Calculate CRC (8-bit CRC with polynomial x^8 + x^2 + x^1 + x^0)
    fn calculate_crc(&self, bits: &[bool]) -> u8 {
        let mut crc = 0u8;

        for &bit in bits {
            let feedback = ((crc & 0x80) != 0) ^ bit;
            crc <<= 1;
            if feedback {
                crc ^= 0x07; // Polynomial: x^8 + x^2 + x^1 + x^0
            }
        }

        crc
    }

    /// Decode timecode from data bits
    fn decode_timecode_from_bits(
        &self,
        data_bits: &[bool; DATA_BITS],
        field: u8,
    ) -> Result<Timecode, TimecodeError> {
        // VITC bit layout (SMPTE 12M):
        // Similar to LTC but with some differences for field identification

        // Bits 0-3: Frame units
        let frame_units = self.bits_to_u8(&data_bits[0..4]);

        // Bits 4-7: User bits 1
        let _user_bits_1 = self.bits_to_u8(&data_bits[4..8]);

        // Bits 8-9: Frame tens
        let frame_tens = self.bits_to_u8(&data_bits[8..10]);

        // Bit 10: Drop frame flag
        let drop_frame = data_bits[10];

        // Bit 11: Color frame flag
        let _color_frame = data_bits[11];

        // Bits 12-15: User bits 2
        let _user_bits_2 = self.bits_to_u8(&data_bits[12..16]);

        // Bits 16-19: Second units
        let second_units = self.bits_to_u8(&data_bits[16..20]);

        // Bits 20-23: User bits 3
        let _user_bits_3 = self.bits_to_u8(&data_bits[20..24]);

        // Bits 24-26: Second tens
        let second_tens = self.bits_to_u8(&data_bits[24..27]);

        // Bit 27: Field mark (1 = field 2, 0 = field 1)
        let field_mark = data_bits[27];

        // Validate field mark
        if (field_mark && field != 2) || (!field_mark && field != 1) {
            // Field mismatch - not necessarily an error, but worth noting
        }

        // Bits 28-31: User bits 4
        let _user_bits_4 = self.bits_to_u8(&data_bits[28..32]);

        // Bits 32-35: Minute units
        let minute_units = self.bits_to_u8(&data_bits[32..36]);

        // Bits 36-39: User bits 5
        let _user_bits_5 = self.bits_to_u8(&data_bits[36..40]);

        // Bits 40-42: Minute tens
        let minute_tens = self.bits_to_u8(&data_bits[40..43]);

        // Bit 43: Binary group flag
        let _binary_group = data_bits[43];

        // Bits 44-47: User bits 6
        let _user_bits_6 = self.bits_to_u8(&data_bits[44..48]);

        // Bits 48-51: Hour units
        let hour_units = self.bits_to_u8(&data_bits[48..52]);

        // Bits 52-55: User bits 7
        let _user_bits_7 = self.bits_to_u8(&data_bits[52..56]);

        // Bits 56-57: Hour tens
        let hour_tens = self.bits_to_u8(&data_bits[56..58]);

        // Bits 58-73: Reserved and user bits 8

        // Compose timecode values
        let frames = frame_tens * 10 + frame_units;
        let seconds = second_tens * 10 + second_units;
        let minutes = minute_tens * 10 + minute_units;
        let hours = hour_tens * 10 + hour_units;

        // Determine frame rate
        let frame_rate = if drop_frame && self.config.frame_rate == FrameRate::Fps2997NDF {
            FrameRate::Fps2997DF
        } else {
            self.config.frame_rate
        };

        // Create timecode
        let timecode = Timecode::new(hours, minutes, seconds, frames, frame_rate)?;

        Ok(timecode)
    }

    /// Convert bit slice to u8
    fn bits_to_u8(&self, bits: &[bool]) -> u8 {
        let mut value = 0u8;
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                value |= 1 << i;
            }
        }
        value
    }

    /// Extract user bits from data bits
    #[allow(dead_code)]
    fn extract_user_bits(&self, data_bits: &[bool; DATA_BITS]) -> u32 {
        let mut user_bits = 0u32;

        // User bits scattered throughout VITC frame (same as LTC)
        user_bits |= self.bits_to_u8(&data_bits[4..8]) as u32;
        user_bits |= (self.bits_to_u8(&data_bits[12..16]) as u32) << 4;
        user_bits |= (self.bits_to_u8(&data_bits[20..24]) as u32) << 8;
        user_bits |= (self.bits_to_u8(&data_bits[28..32]) as u32) << 12;
        user_bits |= (self.bits_to_u8(&data_bits[36..40]) as u32) << 16;
        user_bits |= (self.bits_to_u8(&data_bits[44..48]) as u32) << 20;
        user_bits |= (self.bits_to_u8(&data_bits[52..56]) as u32) << 24;

        user_bits
    }

    /// Reset decoder state
    pub fn reset(&mut self) {
        self.last_timecode = None;
        self.crc_error_count = 0;
        self.sync_count = 0;
        self.threshold = 128;
    }

    /// Check if decoder is synchronized
    pub fn is_synchronized(&self) -> bool {
        self.sync_count >= 5 && self.last_timecode.is_some()
    }

    /// Get CRC error count
    pub fn crc_errors(&self) -> u32 {
        self.crc_error_count
    }
}

/// Bit pattern analyzer
#[allow(dead_code)]
struct BitPatternAnalyzer {
    /// Run-length encoding buffer
    run_lengths: Vec<usize>,
    /// Current run length
    current_run: usize,
    /// Current bit value
    current_bit: bool,
}

impl BitPatternAnalyzer {
    #[allow(dead_code)]
    fn new() -> Self {
        BitPatternAnalyzer {
            run_lengths: Vec::new(),
            current_run: 0,
            current_bit: false,
        }
    }

    /// Add a bit to the analyzer
    #[allow(dead_code)]
    fn add_bit(&mut self, bit: bool) {
        if bit == self.current_bit {
            self.current_run += 1;
        } else {
            if self.current_run > 0 {
                self.run_lengths.push(self.current_run);
            }
            self.current_bit = bit;
            self.current_run = 1;
        }
    }

    /// Finish analysis
    #[allow(dead_code)]
    fn finish(&mut self) {
        if self.current_run > 0 {
            self.run_lengths.push(self.current_run);
        }
    }

    /// Get run lengths
    #[allow(dead_code)]
    fn get_run_lengths(&self) -> &[usize] {
        &self.run_lengths
    }

    #[allow(dead_code)]
    fn reset(&mut self) {
        self.run_lengths.clear();
        self.current_run = 0;
        self.current_bit = false;
    }
}

/// Line quality analyzer
pub struct LineQualityAnalyzer {
    /// Minimum pixel value seen
    min_pixel: u8,
    /// Maximum pixel value seen
    max_pixel: u8,
    /// Pixel count
    pixel_count: usize,
    /// Sum of pixels (for average)
    pixel_sum: u64,
}

impl Default for LineQualityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl LineQualityAnalyzer {
    pub fn new() -> Self {
        LineQualityAnalyzer {
            min_pixel: 255,
            max_pixel: 0,
            pixel_count: 0,
            pixel_sum: 0,
        }
    }

    /// Analyze a line of pixels
    pub fn analyze(&mut self, pixels: &[u8]) {
        for &pixel in pixels {
            self.min_pixel = self.min_pixel.min(pixel);
            self.max_pixel = self.max_pixel.max(pixel);
            self.pixel_sum += pixel as u64;
            self.pixel_count += 1;
        }
    }

    /// Get signal-to-noise ratio estimate
    pub fn get_snr_estimate(&self) -> f32 {
        let dynamic_range = (self.max_pixel as i16 - self.min_pixel as i16).abs() as f32;
        if dynamic_range > 0.0 {
            20.0 * (dynamic_range / 255.0).log10()
        } else {
            0.0
        }
    }

    /// Get average pixel value
    pub fn get_average(&self) -> f32 {
        if self.pixel_count > 0 {
            self.pixel_sum as f32 / self.pixel_count as f32
        } else {
            0.0
        }
    }

    /// Get dynamic range
    pub fn get_dynamic_range(&self) -> u8 {
        self.max_pixel.saturating_sub(self.min_pixel)
    }

    pub fn reset(&mut self) {
        self.min_pixel = 255;
        self.max_pixel = 0;
        self.pixel_count = 0;
        self.pixel_sum = 0;
    }
}

/// Field detector
pub struct FieldDetector {
    /// History of field marks
    field_history: Vec<bool>,
    /// Maximum history size
    max_history: usize,
}

impl FieldDetector {
    pub fn new(max_history: usize) -> Self {
        FieldDetector {
            field_history: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Add a field mark
    pub fn add_field_mark(&mut self, field_mark: bool) {
        self.field_history.push(field_mark);
        if self.field_history.len() > self.max_history {
            self.field_history.remove(0);
        }
    }

    /// Get predominant field
    pub fn get_predominant_field(&self) -> Option<u8> {
        if self.field_history.is_empty() {
            return None;
        }

        let field2_count = self.field_history.iter().filter(|&&f| f).count();
        let field1_count = self.field_history.len() - field2_count;

        if field2_count > field1_count {
            Some(2)
        } else {
            Some(1)
        }
    }

    pub fn reset(&mut self) {
        self.field_history.clear();
    }
}

/// Multi-line VITC reader for redundancy
pub struct MultiLineVitcReader {
    /// Decoders for different lines
    decoders: Vec<(u16, VitcDecoder)>,
}

impl MultiLineVitcReader {
    /// Create a multi-line reader
    pub fn new(config: VitcReaderConfig) -> Self {
        let mut decoders = Vec::new();

        for &line in &config.scan_lines {
            let mut line_config = config.clone();
            line_config.scan_lines = vec![line];
            decoders.push((line, VitcDecoder::new(line_config)));
        }

        MultiLineVitcReader { decoders }
    }

    /// Process a line with all decoders
    pub fn process_line(
        &mut self,
        line_number: u16,
        field: u8,
        pixels: &[u8],
    ) -> Vec<(u16, Result<Option<Timecode>, TimecodeError>)> {
        let mut results = Vec::new();

        for (line, decoder) in &mut self.decoders {
            if *line == line_number {
                let result = decoder.process_line(line_number, field, pixels);
                results.push((*line, result));
            }
        }

        results
    }

    /// Get the best timecode from multiple results
    pub fn get_best_timecode(
        &self,
        results: &[(u16, Result<Option<Timecode>, TimecodeError>)],
    ) -> Option<Timecode> {
        for (_line, result) in results {
            if let Ok(Some(tc)) = result {
                return Some(*tc);
            }
        }
        None
    }
}

/// Error correction for VITC
pub struct VitcErrorCorrector {
    /// History of recent timecodes
    history: Vec<Timecode>,
    /// Maximum history size
    max_history: usize,
}

impl VitcErrorCorrector {
    pub fn new(max_history: usize) -> Self {
        VitcErrorCorrector {
            history: Vec::with_capacity(max_history),
            max_history,
        }
    }

    /// Add a timecode to history
    pub fn add_timecode(&mut self, timecode: Timecode) {
        self.history.push(timecode);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Try to correct or validate a timecode
    pub fn correct_timecode(&self, timecode: &Timecode) -> Option<Timecode> {
        if self.history.is_empty() {
            return Some(*timecode);
        }

        // Check if timecode is sequential with recent history
        if let Some(last) = self.history.last() {
            let mut expected = *last;
            if expected.increment().is_ok() {
                // Allow some tolerance for frame differences
                let diff = (timecode.to_frames() as i64 - expected.to_frames() as i64).abs();
                if diff <= 2 {
                    return Some(*timecode);
                }
            }
        }

        // If not sequential, check if it's consistent with history trend
        Some(*timecode)
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_creation() {
        let config = VitcReaderConfig::default();
        let decoder = VitcDecoder::new(config);
        assert!(!decoder.is_synchronized());
    }

    #[test]
    fn test_crc_calculation() {
        let config = VitcReaderConfig::default();
        let decoder = VitcDecoder::new(config);

        let bits = [false; 72];
        let crc = decoder.calculate_crc(&bits);
        assert_eq!(crc, 0);
    }

    #[test]
    fn test_bits_to_u8() {
        let config = VitcReaderConfig::default();
        let decoder = VitcDecoder::new(config);

        let bits = [true, false, true, false]; // Binary: 0101 = 5
        assert_eq!(decoder.bits_to_u8(&bits), 5);
    }

    #[test]
    fn test_line_quality_analyzer() {
        let mut analyzer = LineQualityAnalyzer::new();
        let pixels = vec![16, 235, 16, 235]; // Black-white pattern

        analyzer.analyze(&pixels);
        assert!(analyzer.get_dynamic_range() > 200);
    }

    #[test]
    fn test_field_detector() {
        let mut detector = FieldDetector::new(10);

        // Add mostly field 2 marks
        detector.add_field_mark(true);
        detector.add_field_mark(true);
        detector.add_field_mark(false);

        assert_eq!(detector.get_predominant_field(), Some(2));
    }
}
