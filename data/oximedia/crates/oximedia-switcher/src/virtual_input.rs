//! Virtual input generator for video switchers.
//!
//! Generates synthetic video frames that can be used as test sources:
//! colour bars, test patterns, countdown timers, black frames and white frames.
//! All output is in **RGB8** (3 bytes per pixel, row-major).

use thiserror::Error;

/// Errors that can occur during virtual input generation.
#[derive(Error, Debug, Clone)]
pub enum VirtualInputError {
    /// The requested dimensions are zero.
    #[error("Frame dimensions must be non-zero (got {0}x{1})")]
    ZeroDimensions(u32, u32),

    /// The countdown value exceeds the displayable range.
    #[error("Countdown value {0} is too large to display")]
    CountdownOutOfRange(u32),
}

/// SMPTE / EBU colour-bar variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorBarsType {
    /// SMPTE 75% colour bars (reduced saturation).
    Smpte75,
    /// SMPTE 100% colour bars (full saturation).
    Smpte100,
    /// HD-SDI compatible colour bars (ARIB STD-B28 / SMPTE RP219).
    Hd,
    /// EBU colour bars (PAL standard).
    EbuColorbars,
}

/// Test pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    /// Checkerboard with a given square size in pixels.
    CheckerBoard(u32),
    /// Regular grid with a given line spacing in pixels.
    Grid(u32),
    /// Smooth horizontal–vertical gradient.
    Gradient,
}

/// Virtual input source descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VirtualInput {
    /// SMPTE / EBU colour bars.
    ColorBars(ColorBarsType),
    /// Geometric test pattern.
    TestPattern(PatternType),
    /// Countdown overlay showing a number centred on black (seconds remaining).
    Countdown(u32),
    /// Solid black frame.
    BlackFrame,
    /// Solid white frame.
    WhiteFrame,
}

/// Generates RGB8 pixel data for a given `VirtualInput`.
pub struct VirtualInputGenerator;

impl VirtualInputGenerator {
    /// Generate an RGB8 frame of the given dimensions for a `VirtualInput`.
    ///
    /// Returns a `Vec<u8>` of length `width * height * 3` in row-major order.
    pub fn generate(
        input: &VirtualInput,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, VirtualInputError> {
        if width == 0 || height == 0 {
            return Err(VirtualInputError::ZeroDimensions(width, height));
        }

        let pixels = (width as usize) * (height as usize);
        let mut buf = vec![0u8; pixels * 3];

        match input {
            VirtualInput::BlackFrame => {
                // Already zero-initialised.
            }
            VirtualInput::WhiteFrame => {
                buf.fill(255);
            }
            VirtualInput::ColorBars(bars_type) => {
                Self::fill_color_bars(&mut buf, width, height, *bars_type);
            }
            VirtualInput::TestPattern(pattern) => {
                Self::fill_test_pattern(&mut buf, width, height, *pattern);
            }
            VirtualInput::Countdown(value) => {
                if *value > 9999 {
                    return Err(VirtualInputError::CountdownOutOfRange(*value));
                }
                Self::fill_countdown(&mut buf, width, height, *value);
            }
        }

        Ok(buf)
    }

    // ── Colour bars ──────────────────────────────────────────────────────────

    fn fill_color_bars(buf: &mut [u8], width: u32, height: u32, bars_type: ColorBarsType) {
        // Classic 7-bar arrangement: white/yellow/cyan/green/magenta/red/blue
        // For SMPTE 75% the chroma channels are scaled to 75%.
        let bars: &[(u8, u8, u8)] = match bars_type {
            ColorBarsType::Smpte100 | ColorBarsType::Hd => &[
                (235, 235, 235), // White
                (235, 235, 16),  // Yellow
                (16, 235, 235),  // Cyan
                (16, 235, 16),   // Green
                (235, 16, 235),  // Magenta
                (235, 16, 16),   // Red
                (16, 16, 235),   // Blue
            ],
            ColorBarsType::Smpte75 => &[
                (191, 191, 191), // 75% White
                (191, 191, 12),  // 75% Yellow
                (12, 191, 191),  // 75% Cyan
                (12, 191, 12),   // 75% Green
                (191, 12, 191),  // 75% Magenta
                (191, 12, 12),   // 75% Red
                (12, 12, 191),   // 75% Blue
            ],
            ColorBarsType::EbuColorbars => &[
                (180, 180, 180), // EBU white (100IRE)
                (180, 180, 0),   // EBU yellow
                (0, 180, 180),   // EBU cyan
                (0, 180, 0),     // EBU green
                (180, 0, 180),   // EBU magenta
                (180, 0, 0),     // EBU red
                (0, 0, 180),     // EBU blue
            ],
        };

        let num_bars = bars.len() as u32;
        let bar_width = width / num_bars;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let bar_idx = (x as u32 / bar_width.max(1)).min(num_bars - 1) as usize;
                let (r, g, b) = bars[bar_idx];
                let idx = (y * width as usize + x) * 3;
                buf[idx] = r;
                buf[idx + 1] = g;
                buf[idx + 2] = b;
            }
        }
    }

    // ── Test patterns ─────────────────────────────────────────────────────────

    fn fill_test_pattern(buf: &mut [u8], width: u32, height: u32, pattern: PatternType) {
        match pattern {
            PatternType::CheckerBoard(square_size) => {
                let sq = square_size.max(1) as usize;
                for y in 0..height as usize {
                    for x in 0..width as usize {
                        let tile_x = x / sq;
                        let tile_y = y / sq;
                        let white = (tile_x + tile_y) % 2 == 0;
                        let val = if white { 235u8 } else { 16u8 };
                        let idx = (y * width as usize + x) * 3;
                        buf[idx] = val;
                        buf[idx + 1] = val;
                        buf[idx + 2] = val;
                    }
                }
            }
            PatternType::Grid(spacing) => {
                let sp = spacing.max(1) as usize;
                // Fill with mid-grey background first.
                for byte in buf.iter_mut() {
                    *byte = 64;
                }
                // Draw white grid lines.
                for y in 0..height as usize {
                    for x in 0..width as usize {
                        if x % sp == 0 || y % sp == 0 {
                            let idx = (y * width as usize + x) * 3;
                            buf[idx] = 235;
                            buf[idx + 1] = 235;
                            buf[idx + 2] = 235;
                        }
                    }
                }
            }
            PatternType::Gradient => {
                for y in 0..height as usize {
                    for x in 0..width as usize {
                        let r = ((x as f32 / (width as f32 - 1.0).max(1.0)) * 235.0) as u8;
                        let g = ((y as f32 / (height as f32 - 1.0).max(1.0)) * 235.0) as u8;
                        let b = 128u8;
                        let idx = (y * width as usize + x) * 3;
                        buf[idx] = r;
                        buf[idx + 1] = g;
                        buf[idx + 2] = b;
                    }
                }
            }
        }
    }

    // ── Countdown ─────────────────────────────────────────────────────────────

    /// Fills `buf` with a black background and a simple bitmapped digit overlay
    /// centred in the frame.
    ///
    /// We use a built-in 5×7 bitmapped font for digits 0–9 and the colon
    /// separator for readability.  Large frames get upscaled glyphs.
    fn fill_countdown(buf: &mut [u8], width: u32, height: u32, value: u32) {
        // Background is already black (buf zero-initialised before this call).

        // Format the countdown as a string.
        let text = format!("{value}");

        // Determine a pixel scale so the text is clearly visible.
        let scale = ((width / 80).max(1)).min(8) as usize;
        let glyph_w = 5 * scale;
        let glyph_h = 7 * scale;
        let total_w = glyph_w * text.len() + scale * (text.len().saturating_sub(1));
        let start_x = ((width as usize).saturating_sub(total_w)) / 2;
        let start_y = ((height as usize).saturating_sub(glyph_h)) / 2;

        for (char_idx, ch) in text.chars().enumerate() {
            let digit_x = start_x + char_idx * (glyph_w + scale);
            let bitmap = Self::digit_bitmap(ch);

            for row in 0..7_usize {
                for col in 0..5_usize {
                    if bitmap[row] & (1 << (4 - col)) != 0 {
                        // Fill the scaled block.
                        for sy in 0..scale {
                            for sx in 0..scale {
                                let px = digit_x + col * scale + sx;
                                let py = start_y + row * scale + sy;
                                if px < width as usize && py < height as usize {
                                    let idx = (py * width as usize + px) * 3;
                                    buf[idx] = 235;
                                    buf[idx + 1] = 235;
                                    buf[idx + 2] = 235;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Return a 5×7 pixel bitmap for a single character.  Each `u8` encodes
    /// one row; bit 4 = leftmost column, bit 0 = rightmost column.
    fn digit_bitmap(ch: char) -> [u8; 7] {
        match ch {
            '0' => [
                0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
            ],
            '1' => [
                0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            ],
            '2' => [
                0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
            ],
            '3' => [
                0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
            ],
            '4' => [
                0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
            ],
            '5' => [
                0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
            ],
            '6' => [
                0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
            ],
            '7' => [
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
            ],
            '8' => [
                0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
            ],
            '9' => [
                0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
            ],
            ':' => [
                0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000,
            ],
            _ => [0b00000; 7],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_frame_all_zero() {
        let frame = VirtualInputGenerator::generate(&VirtualInput::BlackFrame, 4, 4)
            .expect("should succeed");
        assert_eq!(frame.len(), 4 * 4 * 3);
        assert!(frame.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_white_frame_all_255() {
        let frame = VirtualInputGenerator::generate(&VirtualInput::WhiteFrame, 4, 4)
            .expect("should succeed");
        assert_eq!(frame.len(), 4 * 4 * 3);
        assert!(frame.iter().all(|&b| b == 255));
    }

    #[test]
    fn test_zero_dimensions_error() {
        let result = VirtualInputGenerator::generate(&VirtualInput::BlackFrame, 0, 100);
        assert!(result.is_err());
        let result2 = VirtualInputGenerator::generate(&VirtualInput::BlackFrame, 100, 0);
        assert!(result2.is_err());
    }

    #[test]
    fn test_color_bars_smpte100_correct_length() {
        let frame = VirtualInputGenerator::generate(
            &VirtualInput::ColorBars(ColorBarsType::Smpte100),
            70,
            10,
        )
        .expect("should succeed");
        assert_eq!(frame.len(), 70 * 10 * 3);
    }

    #[test]
    fn test_color_bars_smpte75_not_all_same() {
        let frame = VirtualInputGenerator::generate(
            &VirtualInput::ColorBars(ColorBarsType::Smpte75),
            70,
            10,
        )
        .expect("should succeed");
        // The bars must contain more than one distinct colour.
        let first_pixel = &frame[0..3];
        let has_different = frame.chunks(3).any(|px| px != first_pixel);
        assert!(has_different, "SMPTE 75% bars must have multiple colours");
    }

    #[test]
    fn test_color_bars_hd_correct_length() {
        let frame =
            VirtualInputGenerator::generate(&VirtualInput::ColorBars(ColorBarsType::Hd), 140, 20)
                .expect("should succeed");
        assert_eq!(frame.len(), 140 * 20 * 3);
    }

    #[test]
    fn test_color_bars_ebu_has_variety() {
        let frame = VirtualInputGenerator::generate(
            &VirtualInput::ColorBars(ColorBarsType::EbuColorbars),
            70,
            10,
        )
        .expect("should succeed");
        let first_pixel = &frame[0..3];
        let has_different = frame.chunks(3).any(|px| px != first_pixel);
        assert!(has_different, "EBU bars must have multiple colours");
    }

    #[test]
    fn test_checkerboard_alternating_pixels() {
        let frame = VirtualInputGenerator::generate(
            &VirtualInput::TestPattern(PatternType::CheckerBoard(1)),
            4,
            4,
        )
        .expect("should succeed");
        // With square_size=1 and a 4×4 frame the top-left pixel should be
        // white and the next pixel should be dark.
        let p0 = frame[0];
        let p1 = frame[3]; // second pixel
        assert_ne!(p0, p1, "adjacent checkerboard pixels must differ");
    }

    #[test]
    fn test_grid_has_grid_lines() {
        let frame =
            VirtualInputGenerator::generate(&VirtualInput::TestPattern(PatternType::Grid(4)), 8, 8)
                .expect("should succeed");
        // Pixel at (0,0) should be a grid line (white = 235).
        assert_eq!(frame[0], 235, "grid line pixel should be 235");
        // Pixel at (2,2) should NOT be on a grid line (64).
        let idx = (2 * 8 + 2) * 3;
        assert_eq!(frame[idx], 64, "off-grid pixel should be 64");
    }

    #[test]
    fn test_gradient_not_uniform() {
        let frame = VirtualInputGenerator::generate(
            &VirtualInput::TestPattern(PatternType::Gradient),
            4,
            4,
        )
        .expect("should succeed");
        let first_r = frame[0];
        let last_r = frame[(4 * 4 - 1) * 3];
        // The rightmost pixel should have a higher R value than the leftmost.
        assert!(last_r >= first_r, "gradient R should increase left→right");
    }

    #[test]
    fn test_countdown_produces_correct_length() {
        let frame = VirtualInputGenerator::generate(&VirtualInput::Countdown(10), 64, 32)
            .expect("should succeed");
        assert_eq!(frame.len(), 64 * 32 * 3);
    }

    #[test]
    fn test_countdown_zero_is_valid() {
        let frame =
            VirtualInputGenerator::generate(&VirtualInput::Countdown(0), 32, 32).expect("ok");
        assert_eq!(frame.len(), 32 * 32 * 3);
    }

    #[test]
    fn test_countdown_large_value_out_of_range() {
        let result = VirtualInputGenerator::generate(&VirtualInput::Countdown(10_000), 32, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_countdown_frame_not_all_black() {
        let frame =
            VirtualInputGenerator::generate(&VirtualInput::Countdown(5), 64, 32).expect("ok");
        // The frame should have at least some non-zero pixels (the digit glyph).
        let has_non_zero = frame.iter().any(|&b| b != 0);
        assert!(has_non_zero, "countdown frame must contain digit glyphs");
    }
}
