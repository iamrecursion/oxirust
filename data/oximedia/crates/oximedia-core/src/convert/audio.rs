//! Audio sample format conversion functions.
//!
//! This module provides functions to convert between different audio sample formats,
//! including planar/interleaved conversions and sample type conversions.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use crate::types::SampleFormat;

/// Audio format converter with efficient conversion routines.
///
/// This struct provides methods to convert between different audio sample formats.
#[derive(Clone, Debug)]
pub struct AudioConverter {
    /// Source format.
    src_format: SampleFormat,
    /// Destination format.
    dst_format: SampleFormat,
}

impl AudioConverter {
    /// Creates a new audio converter for the specified format pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::convert::audio::AudioConverter;
    /// use oximedia_core::types::SampleFormat;
    ///
    /// let converter = AudioConverter::new(SampleFormat::S16, SampleFormat::F32);
    /// ```
    #[must_use]
    pub const fn new(src_format: SampleFormat, dst_format: SampleFormat) -> Self {
        Self {
            src_format,
            dst_format,
        }
    }

    /// Returns the source format.
    #[must_use]
    pub const fn src_format(&self) -> SampleFormat {
        self.src_format
    }

    /// Returns the destination format.
    #[must_use]
    pub const fn dst_format(&self) -> SampleFormat {
        self.dst_format
    }

    /// Converts audio samples from source to destination format.
    ///
    /// # Arguments
    ///
    /// * `src` - Source audio data
    /// * `channels` - Number of audio channels
    /// * `samples` - Number of samples per channel
    ///
    /// # Returns
    ///
    /// Converted audio data in destination format
    ///
    /// # Panics
    ///
    /// Panics if source data has incorrect size.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::convert::audio::AudioConverter;
    /// use oximedia_core::types::SampleFormat;
    ///
    /// let converter = AudioConverter::new(SampleFormat::S16, SampleFormat::F32);
    /// let src = vec![0i16; 1024 * 2]; // 1024 samples, 2 channels, interleaved
    /// let src_bytes = src.iter()
    ///     .flat_map(|&s| s.to_le_bytes())
    ///     .collect::<Vec<u8>>();
    ///
    /// let dst = converter.convert(&src_bytes, 2, 1024);
    /// ```
    #[must_use]
    pub fn convert(&self, src: &[u8], channels: usize, samples: usize) -> Vec<u8> {
        // Handle planar/interleaved conversions first

        if self.src_format.is_planar() && !self.dst_format.is_planar() {
            // Planar to interleaved, then convert format
            let planar_converted = convert_sample_format(
                src,
                self.src_format,
                self.src_format.to_packed(),
                channels,
                samples,
            );
            planar_to_interleaved(
                &planar_converted,
                channels,
                samples,
                self.src_format.to_packed(),
            )
        } else if !self.src_format.is_planar() && self.dst_format.is_planar() {
            // Convert format, then interleaved to planar
            let format_converted = convert_sample_format(
                src,
                self.src_format,
                self.dst_format.to_packed(),
                channels,
                samples,
            );
            interleaved_to_planar(
                &format_converted,
                channels,
                samples,
                self.dst_format.to_packed(),
            )
        } else {
            // Same layout, just convert format
            convert_sample_format(src, self.src_format, self.dst_format, channels, samples)
        }
    }
}

/// Converts audio samples between different sample formats.
///
/// This function handles conversion between different sample types (u8, s16, s32, f32, f64)
/// while maintaining the same planar/interleaved layout.
///
/// # Arguments
///
/// * `src` - Source audio data
/// * `src_format` - Source sample format
/// * `dst_format` - Destination sample format
/// * `channels` - Number of audio channels
/// * `samples` - Number of samples per channel
///
/// # Returns
///
/// Converted audio data in destination format
///
/// # Panics
///
/// Panics if source data has incorrect size.
///
/// # Examples
///
/// ```
/// use oximedia_core::convert::audio::convert_sample_format;
/// use oximedia_core::types::SampleFormat;
///
/// let src = vec![0i16; 1024 * 2]; // 1024 samples, 2 channels
/// let src_bytes = src.iter()
///     .flat_map(|&s| s.to_le_bytes())
///     .collect::<Vec<u8>>();
///
/// let dst = convert_sample_format(&src_bytes, SampleFormat::S16, SampleFormat::F32, 2, 1024);
/// ```
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn convert_sample_format(
    src: &[u8],
    src_format: SampleFormat,
    dst_format: SampleFormat,
    channels: usize,
    samples: usize,
) -> Vec<u8> {
    let total_samples = channels * samples;

    let expected_size = total_samples * src_format.bytes_per_sample();
    assert_eq!(
        src.len(),
        expected_size,
        "Source data size mismatch: expected {expected_size}, got {}",
        src.len()
    );

    let converter = SampleConverter::new(src_format, dst_format);
    let dst_size = total_samples * dst_format.bytes_per_sample();
    let mut dst = vec![0u8; dst_size];

    // Convert samples
    for i in 0..total_samples {
        let src_offset = i * src_format.bytes_per_sample();
        let dst_offset = i * dst_format.bytes_per_sample();

        let normalized = converter.read_normalized(src, src_offset);
        converter.write_normalized(&mut dst, dst_offset, normalized);
    }

    dst
}

/// Sample format converter with normalization.
///
/// This converts between different sample types by normalizing to/from f64.
#[derive(Clone, Debug)]
pub struct SampleConverter {
    /// Source format.
    src_format: SampleFormat,
    /// Destination format.
    dst_format: SampleFormat,
}

impl SampleConverter {
    /// Creates a new sample converter.
    #[must_use]
    pub const fn new(src_format: SampleFormat, dst_format: SampleFormat) -> Self {
        Self {
            src_format,
            dst_format,
        }
    }

    /// Reads a sample and normalizes it to [-1.0, 1.0] range.
    #[must_use]
    fn read_normalized(&self, data: &[u8], offset: usize) -> f64 {
        match self.src_format {
            SampleFormat::U8 => {
                let val = data[offset];
                f64::from(val) / 128.0 - 1.0
            }
            SampleFormat::S16 | SampleFormat::S16p => {
                let bytes = [data[offset], data[offset + 1]];
                let val = i16::from_le_bytes(bytes);
                f64::from(val) / 32768.0
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let val = i32::from_le_bytes(bytes);
                f64::from(val) / 2_147_483_648.0
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ];
                let val = f32::from_le_bytes(bytes);
                f64::from(val)
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                let bytes = [
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                    data[offset + 4],
                    data[offset + 5],
                    data[offset + 6],
                    data[offset + 7],
                ];
                f64::from_le_bytes(bytes)
            }
            SampleFormat::S24 | SampleFormat::S24p => {
                // 24-bit signed integer stored in 3 bytes, little-endian.
                // Sign-extend from 24-bit to 32-bit.
                let b0 = data[offset];
                let b1 = data[offset + 1];
                let b2 = data[offset + 2];
                let sign_extend = if b2 & 0x80 != 0 { 0xFF } else { 0x00 };
                let val = i32::from_le_bytes([b0, b1, b2, sign_extend]);
                f64::from(val) / 8_388_608.0 // 2^23
            }
        }
    }

    /// Writes a normalized sample to the destination format.
    fn write_normalized(&self, data: &mut [u8], offset: usize, val: f64) {
        let clamped = val.clamp(-1.0, 1.0);

        match self.dst_format {
            SampleFormat::U8 => {
                let scaled = ((clamped + 1.0) * 128.0).round().clamp(0.0, 255.0);
                data[offset] = scaled as u8;
            }
            SampleFormat::S16 | SampleFormat::S16p => {
                let scaled = (clamped * 32767.0).round().clamp(-32768.0, 32767.0);
                let bytes = (scaled as i16).to_le_bytes();
                data[offset] = bytes[0];
                data[offset + 1] = bytes[1];
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                let scaled = (clamped * 2_147_483_647.0)
                    .round()
                    .clamp(-2_147_483_648.0, 2_147_483_647.0);
                let bytes = (scaled as i32).to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                let bytes = (clamped as f32).to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                let bytes = clamped.to_le_bytes();
                data[offset..offset + 8].copy_from_slice(&bytes);
            }
            SampleFormat::S24 | SampleFormat::S24p => {
                // 24-bit signed integer stored in 3 bytes, little-endian.
                let scaled = (clamped * 8_388_607.0)
                    .round()
                    .clamp(-8_388_608.0, 8_388_607.0);
                let val = scaled as i32;
                let bytes = val.to_le_bytes();
                data[offset] = bytes[0];
                data[offset + 1] = bytes[1];
                data[offset + 2] = bytes[2];
            }
        }
    }
}

/// Converts interleaved audio to planar format.
///
/// Interleaved format stores samples as: [L0, R0, L1, R1, ...]
/// Planar format stores samples as: [L0, L1, ...], [R0, R1, ...]
///
/// # Arguments
///
/// * `interleaved` - Interleaved audio data
/// * `channels` - Number of audio channels
/// * `samples` - Number of samples per channel
/// * `format` - Sample format (must be non-planar)
///
/// # Returns
///
/// Planar audio data
///
/// # Panics
///
/// Panics if input format is planar or data has incorrect size.
///
/// # Examples
///
/// ```
/// use oximedia_core::convert::audio::interleaved_to_planar;
/// use oximedia_core::types::SampleFormat;
///
/// let interleaved = vec![0i16, 1, 2, 3, 4, 5]; // 3 samples, 2 channels
/// let interleaved_bytes = interleaved.iter()
///     .flat_map(|&s| s.to_le_bytes())
///     .collect::<Vec<u8>>();
///
/// let planar = interleaved_to_planar(&interleaved_bytes, 2, 3, SampleFormat::S16);
/// ```
#[must_use]
pub fn interleaved_to_planar(
    interleaved: &[u8],
    channels: usize,
    samples: usize,
    format: SampleFormat,
) -> Vec<u8> {
    assert!(!format.is_planar(), "Format must be non-planar");

    let sample_size = format.bytes_per_sample();
    let total_samples = channels * samples;
    let expected_size = total_samples * sample_size;

    assert_eq!(
        interleaved.len(),
        expected_size,
        "Interleaved data size mismatch"
    );

    let mut planar = vec![0u8; expected_size];

    // De-interleave samples
    for ch in 0..channels {
        for s in 0..samples {
            let src_offset = (s * channels + ch) * sample_size;
            let dst_offset = (ch * samples + s) * sample_size;

            planar[dst_offset..dst_offset + sample_size]
                .copy_from_slice(&interleaved[src_offset..src_offset + sample_size]);
        }
    }

    planar
}

/// Converts planar audio to interleaved format.
///
/// Planar format stores samples as: [L0, L1, ...], [R0, R1, ...]
/// Interleaved format stores samples as: [L0, R0, L1, R1, ...]
///
/// # Arguments
///
/// * `planar` - Planar audio data
/// * `channels` - Number of audio channels
/// * `samples` - Number of samples per channel
/// * `format` - Sample format (must be non-planar)
///
/// # Returns
///
/// Interleaved audio data
///
/// # Panics
///
/// Panics if input format is planar or data has incorrect size.
///
/// # Examples
///
/// ```
/// use oximedia_core::convert::audio::planar_to_interleaved;
/// use oximedia_core::types::SampleFormat;
///
/// // Planar: [L0, L1, L2], [R0, R1, R2]
/// let planar = vec![0i16, 2, 4, 1, 3, 5];
/// let planar_bytes = planar.iter()
///     .flat_map(|&s| s.to_le_bytes())
///     .collect::<Vec<u8>>();
///
/// let interleaved = planar_to_interleaved(&planar_bytes, 2, 3, SampleFormat::S16);
/// ```
#[must_use]
pub fn planar_to_interleaved(
    planar: &[u8],
    channels: usize,
    samples: usize,
    format: SampleFormat,
) -> Vec<u8> {
    assert!(!format.is_planar(), "Format must be non-planar");

    let sample_size = format.bytes_per_sample();
    let total_samples = channels * samples;
    let expected_size = total_samples * sample_size;

    assert_eq!(planar.len(), expected_size, "Planar data size mismatch");

    let mut interleaved = vec![0u8; expected_size];

    // Interleave samples
    for ch in 0..channels {
        for s in 0..samples {
            let src_offset = (ch * samples + s) * sample_size;
            let dst_offset = (s * channels + ch) * sample_size;

            interleaved[dst_offset..dst_offset + sample_size]
                .copy_from_slice(&planar[src_offset..src_offset + sample_size]);
        }
    }

    interleaved
}

/// Resamples audio by changing the number of channels.
///
/// # Arguments
///
/// * `src` - Source audio data
/// * `src_channels` - Number of source channels
/// * `dst_channels` - Number of destination channels
/// * `samples` - Number of samples per channel
/// * `format` - Sample format
///
/// # Returns
///
/// Resampled audio data with new channel count
///
/// # Panics
///
/// Panics if source data has incorrect size.
///
/// # Examples
///
/// ```
/// use oximedia_core::convert::audio::change_channel_count;
/// use oximedia_core::types::SampleFormat;
///
/// let stereo = vec![0i16, 1, 2, 3]; // 2 samples, 2 channels
/// let stereo_bytes = stereo.iter()
///     .flat_map(|&s| s.to_le_bytes())
///     .collect::<Vec<u8>>();
///
/// // Convert stereo to mono by averaging
/// let mono = change_channel_count(&stereo_bytes, 2, 1, 2, SampleFormat::S16);
/// ```
#[must_use]
pub fn change_channel_count(
    src: &[u8],
    src_channels: usize,
    dst_channels: usize,
    samples: usize,
    format: SampleFormat,
) -> Vec<u8> {
    let sample_size = format.bytes_per_sample();
    let expected_size = src_channels * samples * sample_size;
    assert_eq!(src.len(), expected_size, "Source data size mismatch");

    let converter = SampleConverter::new(format, format);
    let dst_size = dst_channels * samples * sample_size;
    let mut dst = vec![0u8; dst_size];

    if src_channels == dst_channels {
        // No change, just copy
        dst.copy_from_slice(src);
    } else if dst_channels == 1 {
        // Downmix to mono by averaging all channels
        for s in 0..samples {
            let mut sum = 0.0;
            for ch in 0..src_channels {
                let offset = if format.is_planar() {
                    (ch * samples + s) * sample_size
                } else {
                    (s * src_channels + ch) * sample_size
                };
                sum += converter.read_normalized(src, offset);
            }
            let avg = sum / src_channels as f64;
            let dst_offset = s * sample_size;
            converter.write_normalized(&mut dst, dst_offset, avg);
        }
    } else if src_channels == 1 {
        // Upmix mono to multiple channels by duplicating
        for s in 0..samples {
            let src_offset = s * sample_size;
            let val = converter.read_normalized(src, src_offset);
            for ch in 0..dst_channels {
                let dst_offset = if format.is_planar() {
                    (ch * samples + s) * sample_size
                } else {
                    (s * dst_channels + ch) * sample_size
                };
                converter.write_normalized(&mut dst, dst_offset, val);
            }
        }
    } else {
        // General case: map channels appropriately
        for s in 0..samples {
            for dst_ch in 0..dst_channels {
                let src_ch = if dst_ch < src_channels {
                    dst_ch
                } else {
                    dst_ch % src_channels
                };

                let src_offset = if format.is_planar() {
                    (src_ch * samples + s) * sample_size
                } else {
                    (s * src_channels + src_ch) * sample_size
                };

                let val = converter.read_normalized(src, src_offset);

                let dst_offset = if format.is_planar() {
                    (dst_ch * samples + s) * sample_size
                } else {
                    (s * dst_channels + dst_ch) * sample_size
                };

                converter.write_normalized(&mut dst, dst_offset, val);
            }
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_s16_to_f32() {
        let src = vec![0i16, 16384, -16384, 32767, -32768];
        let src_bytes: Vec<u8> = src.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let dst = convert_sample_format(&src_bytes, SampleFormat::S16, SampleFormat::F32, 1, 5);

        assert_eq!(dst.len(), 5 * 4);

        // Verify first sample (0)
        let val = f32::from_le_bytes([dst[0], dst[1], dst[2], dst[3]]);
        assert!(val.abs() < 0.01);
    }

    #[test]
    fn test_convert_f32_to_s16() {
        let src = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let src_bytes: Vec<u8> = src.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let dst = convert_sample_format(&src_bytes, SampleFormat::F32, SampleFormat::S16, 1, 5);

        assert_eq!(dst.len(), 5 * 2);

        // Verify first sample (0.0)
        let val = i16::from_le_bytes([dst[0], dst[1]]);
        assert!(val.abs() < 10);
    }

    #[test]
    fn test_interleaved_to_planar() {
        // Stereo interleaved: [L0, R0, L1, R1]
        let interleaved = vec![0i16, 1, 2, 3];
        let interleaved_bytes: Vec<u8> =
            interleaved.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let planar = interleaved_to_planar(&interleaved_bytes, 2, 2, SampleFormat::S16);

        // Planar should be: [L0, L1, R0, R1]
        let mut planar_samples = Vec::new();
        for i in 0..(planar.len() / 2) {
            planar_samples.push(i16::from_le_bytes([planar[i * 2], planar[i * 2 + 1]]));
        }

        assert_eq!(planar_samples, vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_planar_to_interleaved() {
        // Stereo planar: [L0, L1, R0, R1]
        let planar = vec![0i16, 2, 1, 3];
        let planar_bytes: Vec<u8> = planar.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let interleaved = planar_to_interleaved(&planar_bytes, 2, 2, SampleFormat::S16);

        // Interleaved should be: [L0, R0, L1, R1]
        let mut interleaved_samples = Vec::new();
        for i in 0..(interleaved.len() / 2) {
            interleaved_samples.push(i16::from_le_bytes([
                interleaved[i * 2],
                interleaved[i * 2 + 1],
            ]));
        }

        assert_eq!(interleaved_samples, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_change_channel_count_stereo_to_mono() {
        // Stereo: [L0, R0, L1, R1]
        let stereo = vec![100i16, 200, 300, 400];
        let stereo_bytes: Vec<u8> = stereo.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let mono = change_channel_count(&stereo_bytes, 2, 1, 2, SampleFormat::S16);

        // Should average: [(100+200)/2, (300+400)/2]
        let mut mono_samples = Vec::new();
        for i in 0..(mono.len() / 2) {
            mono_samples.push(i16::from_le_bytes([mono[i * 2], mono[i * 2 + 1]]));
        }

        // Check that values are approximately correct
        assert!((mono_samples[0] - 150).abs() < 10);
        assert!((mono_samples[1] - 350).abs() < 10);
    }

    #[test]
    fn test_change_channel_count_mono_to_stereo() {
        // Mono: [100, 200]
        let mono = vec![100i16, 200];
        let mono_bytes: Vec<u8> = mono.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let stereo = change_channel_count(&mono_bytes, 1, 2, 2, SampleFormat::S16);

        // Should duplicate: [100, 100, 200, 200]
        let mut stereo_samples = Vec::new();
        for i in 0..(stereo.len() / 2) {
            stereo_samples.push(i16::from_le_bytes([stereo[i * 2], stereo[i * 2 + 1]]));
        }

        assert_eq!(stereo_samples, vec![100, 100, 200, 200]);
    }

    #[test]
    fn test_audio_converter() {
        let converter = AudioConverter::new(SampleFormat::S16, SampleFormat::F32);

        let src = vec![0i16, 16384, -16384];
        let src_bytes: Vec<u8> = src.iter().flat_map(|&s| s.to_le_bytes()).collect();

        let dst = converter.convert(&src_bytes, 1, 3);

        assert_eq!(dst.len(), 3 * 4);
    }

    #[test]
    fn test_sample_converter_u8() {
        let converter = SampleConverter::new(SampleFormat::U8, SampleFormat::S16);

        let src = vec![0u8, 128, 255];
        let mut dst = vec![0u8; 3 * 2];

        for (i, &val) in src.iter().enumerate() {
            let normalized = converter.read_normalized(&[val], 0);
            converter.write_normalized(&mut dst, i * 2, normalized);
        }

        // U8 0 -> normalized -1.0 -> S16 -32768
        let s0 = i16::from_le_bytes([dst[0], dst[1]]);
        assert!(s0 < -30000);

        // U8 128 -> normalized ~0.0 -> S16 ~0
        let s1 = i16::from_le_bytes([dst[2], dst[3]]);
        assert!(s1.abs() < 500);

        // U8 255 -> normalized ~1.0 -> S16 ~32767
        let s2 = i16::from_le_bytes([dst[4], dst[5]]);
        assert!(s2 > 30000);
    }

    #[test]
    fn test_roundtrip_conversion() {
        let original = vec![
            0i16, 1000, -1000, 10000, -10000, 20000, -20000, 32767, -32768,
        ];
        let original_bytes: Vec<u8> = original.iter().flat_map(|&s| s.to_le_bytes()).collect();

        // S16 -> F32 -> S16
        let f32_bytes =
            convert_sample_format(&original_bytes, SampleFormat::S16, SampleFormat::F32, 1, 9);
        let s16_bytes =
            convert_sample_format(&f32_bytes, SampleFormat::F32, SampleFormat::S16, 1, 9);

        let mut result = Vec::new();
        for i in 0..9 {
            result.push(i16::from_le_bytes([s16_bytes[i * 2], s16_bytes[i * 2 + 1]]));
        }

        // Should be very close to original (within rounding error)
        for (i, (&orig, &res)) in original.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig - res).abs() <= 1,
                "Sample {i}: original={orig}, result={res}"
            );
        }
    }
}
