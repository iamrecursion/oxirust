//! RAW camera image format support.
//!
//! Provides Bayer demosaicing, white balance, and RAW-to-linear conversion
//! for common camera RAW formats (CameraRaw, DNG, CR2, NEF, ARW, ORF, RAF).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Supported RAW image formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawFormat {
    /// Generic camera RAW.
    CameraRaw,
    /// Adobe Digital Negative (DNG).
    DngFile,
    /// Canon RAW 2 (CR2).
    Cr2,
    /// Nikon Electronic Format (NEF).
    Nef,
    /// Sony Alpha RAW (ARW).
    Arw,
    /// Olympus RAW Format (ORF).
    Orf,
    /// Fujifilm RAW (RAF).
    Raf,
}

impl RawFormat {
    /// Returns the file extension typically associated with this format.
    #[must_use]
    pub fn extension(&self) -> &str {
        match self {
            Self::CameraRaw => "raw",
            Self::DngFile => "dng",
            Self::Cr2 => "cr2",
            Self::Nef => "nef",
            Self::Arw => "arw",
            Self::Orf => "orf",
            Self::Raf => "raf",
        }
    }

    /// Returns the display name of the format.
    #[must_use]
    pub fn display_name(&self) -> &str {
        match self {
            Self::CameraRaw => "Camera RAW",
            Self::DngFile => "Digital Negative (DNG)",
            Self::Cr2 => "Canon RAW 2 (CR2)",
            Self::Nef => "Nikon Electronic Format (NEF)",
            Self::Arw => "Sony Alpha RAW (ARW)",
            Self::Orf => "Olympus RAW (ORF)",
            Self::Raf => "Fujifilm RAW (RAF)",
        }
    }
}

/// Bayer color filter array pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BayerPattern {
    /// R G / G B
    Rggb,
    /// B G / G R
    Bggr,
    /// G R / B G
    Grbg,
    /// G B / R G
    Gbrg,
}

impl BayerPattern {
    /// Returns the name of this Bayer pattern.
    #[must_use]
    pub fn pattern_name(&self) -> &str {
        match self {
            Self::Rggb => "RGGB",
            Self::Bggr => "BGGR",
            Self::Grbg => "GRBG",
            Self::Gbrg => "GBRG",
        }
    }

    /// Returns the color indices for the 2×2 CFA tile.
    ///
    /// Index meaning: 0 = Red, 1 = Green, 2 = Blue.
    /// Layout: `[[top-left, top-right], [bottom-left, bottom-right]]`
    #[must_use]
    pub fn indices(&self) -> [[u8; 2]; 2] {
        match self {
            Self::Rggb => [[0, 1], [1, 2]], // R G / G B
            Self::Bggr => [[2, 1], [1, 0]], // B G / G R
            Self::Grbg => [[1, 0], [2, 1]], // G R / B G
            Self::Gbrg => [[1, 2], [0, 1]], // G B / R G
        }
    }

    /// Returns the color channel (R=0, G=1, B=2) at pixel (col, row) for this pattern.
    #[must_use]
    pub fn channel_at(&self, col: u32, row: u32) -> u8 {
        self.indices()[(row % 2) as usize][(col % 2) as usize]
    }
}

/// Metadata for a RAW image.
#[derive(Debug, Clone)]
pub struct RawImageInfo {
    /// Source format.
    pub format: RawFormat,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bits per sample (typically 12 or 14).
    pub bits_per_sample: u8,
    /// Color filter array pattern.
    pub color_filter: BayerPattern,
    /// ISO sensitivity.
    pub iso: u32,
    /// Shutter speed in seconds.
    pub exposure_time: f64,
    /// Lens aperture (f-number).
    pub f_number: f64,
}

impl RawImageInfo {
    /// Creates a new `RawImageInfo` with sensible defaults.
    #[must_use]
    pub fn new(format: RawFormat, width: u32, height: u32) -> Self {
        Self {
            format,
            width,
            height,
            bits_per_sample: 12,
            color_filter: BayerPattern::Rggb,
            iso: 100,
            exposure_time: 1.0 / 100.0,
            f_number: 2.8,
        }
    }

    /// Returns total number of raw pixels.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns the maximum value representable at this bit depth.
    #[must_use]
    pub fn max_value(&self) -> u16 {
        if self.bits_per_sample >= 16 {
            u16::MAX
        } else {
            (1u16 << self.bits_per_sample) - 1
        }
    }
}

/// Configuration for the RAW processing pipeline.
#[derive(Debug, Clone)]
pub struct RawProcessConfig {
    /// White balance gains `[R, G, B]`.
    pub white_balance: [f64; 3],
    /// Exposure compensation in EV stops.
    pub exposure_comp_ev: f64,
    /// Noise reduction strength (0.0 = off, 1.0 = maximum).
    pub noise_reduction: f64,
}

impl Default for RawProcessConfig {
    fn default() -> Self {
        Self {
            white_balance: [1.0, 1.0, 1.0],
            exposure_comp_ev: 0.0,
            noise_reduction: 0.0,
        }
    }
}

/// High-level RAW image processor.
pub struct RawProcessor {
    /// Processing configuration.
    pub config: RawProcessConfig,
}

impl RawProcessor {
    /// Creates a new `RawProcessor` with the given configuration.
    #[must_use]
    pub fn new(config: RawProcessConfig) -> Self {
        Self { config }
    }

    /// Creates a `RawProcessor` with default settings.
    #[must_use]
    pub fn default_processor() -> Self {
        Self::new(RawProcessConfig::default())
    }

    /// Processes a RAW buffer: demosaic → white balance → exposure compensation.
    ///
    /// Returns interleaved RGB `f64` values in `[0.0, 1.0]`.
    #[must_use]
    pub fn process(&self, raw: &[u16], width: u32, height: u32, info: &RawImageInfo) -> Vec<f64> {
        let black_level = 0u16;
        let white_level = info.max_value();

        // Demosaic to u16 RGB
        let rgb_u16 = demosaic_bilinear(raw, width, height, info.color_filter);

        // Convert to f64 linear
        let mut rgb_f64: Vec<f64> = rgb_u16
            .iter()
            .map(|&v| raw_to_linear(v, black_level, white_level))
            .collect();

        // Apply white balance
        let [rg, gg, bg] = self.config.white_balance;
        apply_white_balance(&mut rgb_f64, rg, gg, bg);

        // Apply exposure compensation
        let ev_scale = (self.config.exposure_comp_ev).exp2();
        for v in &mut rgb_f64 {
            *v = (*v * ev_scale).min(1.0);
        }

        rgb_f64
    }
}

impl Default for RawProcessor {
    fn default() -> Self {
        Self::default_processor()
    }
}

// ── Public functions ──────────────────────────────────────────────────────────

/// Converts a raw sensor value to a linear `[0.0, 1.0]` value.
///
/// Clamps out-of-range inputs gracefully.
#[must_use]
pub fn raw_to_linear(value: u16, black_level: u16, white_level: u16) -> f64 {
    if white_level <= black_level {
        return 0.0;
    }
    let clamped = value.clamp(black_level, white_level);
    (clamped - black_level) as f64 / (white_level - black_level) as f64
}

/// Applies per-channel white balance gains to an interleaved RGB `f64` slice.
///
/// Values are clamped to `[0.0, 1.0]` after scaling.
pub fn apply_white_balance(rgb: &mut [f64], r_gain: f64, g_gain: f64, b_gain: f64) {
    let gains = [r_gain, g_gain, b_gain];
    for (i, v) in rgb.iter_mut().enumerate() {
        *v = (*v * gains[i % 3]).clamp(0.0, 1.0);
    }
}

/// Bilinear Bayer demosaicing.
///
/// Converts a single-channel RAW sensor buffer into interleaved RGB `u16`.
/// Output length = `width * height * 3`.
#[must_use]
pub fn demosaic_bilinear(raw: &[u16], width: u32, height: u32, pattern: BayerPattern) -> Vec<u16> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u16; w * h * 3];

    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            let ch = pattern.channel_at(col as u32, row as u32) as usize;

            let r = interpolate_channel(raw, w, h, col, row, 0, pattern);
            let g = interpolate_channel(raw, w, h, col, row, 1, pattern);
            let b = interpolate_channel(raw, w, h, col, row, 2, pattern);

            // Suppress unused variable warning for ch (used in optimized path below
            // but we use the interpolation path uniformly here)
            let _ = ch;

            out[idx * 3] = r;
            out[idx * 3 + 1] = g;
            out[idx * 3 + 2] = b;
        }
    }
    out
}

/// Interpolates a single colour channel at pixel (col, row) using bilinear
/// neighbours that actually contain that channel.
fn interpolate_channel(
    raw: &[u16],
    w: usize,
    h: usize,
    col: usize,
    row: usize,
    target_ch: u8,
    pattern: BayerPattern,
) -> u16 {
    // If this pixel already carries the target channel, return it directly.
    if pattern.channel_at(col as u32, row as u32) == target_ch {
        return raw[row * w + col];
    }

    // Collect the nearest neighbours that carry target_ch.
    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    let row_start = row.saturating_sub(1);
    let row_end = (row + 2).min(h);
    let col_start = col.saturating_sub(1);
    let col_end = (col + 2).min(w);

    for nr in row_start..row_end {
        for nc in col_start..col_end {
            if pattern.channel_at(nc as u32, nr as u32) == target_ch {
                sum += u32::from(raw[nr * w + nc]);
                count += 1;
            }
        }
    }

    sum.checked_div(count).unwrap_or(0) as u16
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_format_extension() {
        assert_eq!(RawFormat::DngFile.extension(), "dng");
        assert_eq!(RawFormat::Cr2.extension(), "cr2");
        assert_eq!(RawFormat::Nef.extension(), "nef");
        assert_eq!(RawFormat::Arw.extension(), "arw");
    }

    #[test]
    fn test_raw_format_display_name() {
        assert!(RawFormat::DngFile.display_name().contains("DNG"));
        assert!(RawFormat::Cr2.display_name().contains("Canon"));
        assert!(RawFormat::CameraRaw.display_name().contains("Camera RAW"));
    }

    #[test]
    fn test_bayer_pattern_name() {
        assert_eq!(BayerPattern::Rggb.pattern_name(), "RGGB");
        assert_eq!(BayerPattern::Bggr.pattern_name(), "BGGR");
        assert_eq!(BayerPattern::Grbg.pattern_name(), "GRBG");
        assert_eq!(BayerPattern::Gbrg.pattern_name(), "GBRG");
    }

    #[test]
    fn test_bayer_indices_rggb() {
        let idx = BayerPattern::Rggb.indices();
        assert_eq!(idx[0][0], 0); // top-left = R
        assert_eq!(idx[0][1], 1); // top-right = G
        assert_eq!(idx[1][0], 1); // bottom-left = G
        assert_eq!(idx[1][1], 2); // bottom-right = B
    }

    #[test]
    fn test_bayer_channel_at_rggb() {
        assert_eq!(BayerPattern::Rggb.channel_at(0, 0), 0); // R
        assert_eq!(BayerPattern::Rggb.channel_at(1, 0), 1); // G
        assert_eq!(BayerPattern::Rggb.channel_at(0, 1), 1); // G
        assert_eq!(BayerPattern::Rggb.channel_at(1, 1), 2); // B
    }

    #[test]
    fn test_raw_to_linear_midpoint() {
        let val = raw_to_linear(2048, 0, 4095);
        assert!((val - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_raw_to_linear_clamp_above_white() {
        let val = raw_to_linear(5000, 0, 4095);
        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_raw_to_linear_black_level() {
        let val = raw_to_linear(512, 512, 4095);
        assert!((val - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_raw_to_linear_degenerate_range() {
        // white_level == black_level → should return 0.0
        let val = raw_to_linear(100, 100, 100);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_apply_white_balance_identity() {
        let mut rgb = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        apply_white_balance(&mut rgb, 1.0, 1.0, 1.0);
        for v in &rgb {
            assert!((v - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn test_apply_white_balance_clamp() {
        let mut rgb = vec![0.8, 0.8, 0.8];
        apply_white_balance(&mut rgb, 2.0, 2.0, 2.0);
        for v in &rgb {
            assert_eq!(*v, 1.0);
        }
    }

    #[test]
    fn test_demosaic_bilinear_output_size() {
        let raw = vec![1000u16; 4 * 4];
        let out = demosaic_bilinear(&raw, 4, 4, BayerPattern::Rggb);
        assert_eq!(out.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_raw_image_info_max_value_12bit() {
        let info = RawImageInfo::new(RawFormat::Cr2, 4096, 2732);
        assert_eq!(info.max_value(), 4095);
    }

    #[test]
    fn test_raw_processor_process_output_size() {
        let info = RawImageInfo::new(RawFormat::DngFile, 4, 4);
        let raw = vec![2048u16; 4 * 4];
        let proc = RawProcessor::default_processor();
        let result = proc.process(&raw, 4, 4, &info);
        assert_eq!(result.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_raw_image_info_pixel_count() {
        let info = RawImageInfo::new(RawFormat::Nef, 6000, 4000);
        assert_eq!(info.pixel_count(), 24_000_000);
    }
}
