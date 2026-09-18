//! Zebra stripe overexposure indicator.
//!
//! The zebra display overlays animated diagonal stripes on pixels that exceed
//! a user-defined luminance/IRE threshold, providing a real-time overexposure
//! warning during recording or grading.

#![allow(dead_code)]

/// Zebra analysis mode — which channel to evaluate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZebraMode {
    /// Evaluate luma (Y channel via BT.709 coefficients).
    Luma,
    /// Evaluate red channel only.
    R,
    /// Evaluate green channel only.
    G,
    /// Evaluate blue channel only.
    B,
    /// Evaluate in IRE units (0-100), same as luma but scaled.
    IRE,
}

/// Configuration for the zebra overlay.
#[derive(Debug, Clone)]
pub struct ZebraConfig {
    /// Overexposure threshold in [0, 1] range (default 0.95 = 95%).
    pub threshold: f32,
    /// Width of each diagonal stripe in pixels (default 8).
    pub stripe_width: u32,
    /// Stripe color (RGBA). Defaults to yellow semi-transparent.
    pub color: [u8; 4],
}

impl Default for ZebraConfig {
    fn default() -> Self {
        Self {
            threshold: 0.95,
            stripe_width: 8,
            color: [255, 220, 0, 200],
        }
    }
}

/// Statistics returned alongside the zebra overlay.
#[derive(Debug, Clone)]
pub struct ZebraStats {
    /// Number of pixels flagged as overexposed.
    pub overexposed_pixels: u32,
    /// Percentage of overexposed pixels [0, 100].
    pub overexposed_pct: f32,
    /// Peak IRE value found in the frame (0-100+).
    pub peak_ire: f32,
}

/// Zebra overlay processor.
pub struct ZebraOverlay {
    config: ZebraConfig,
    /// Animation frame counter; increments on each call to `apply`.
    frame_counter: u32,
}

impl ZebraOverlay {
    /// Creates a new `ZebraOverlay` with the given configuration.
    #[must_use]
    pub fn new(config: ZebraConfig) -> Self {
        Self {
            config,
            frame_counter: 0,
        }
    }

    /// Applies the zebra overlay to an RGB24 frame.
    ///
    /// # Arguments
    ///
    /// * `frame`  - RGB24 frame bytes (width * height * 3).
    /// * `width`  - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    /// * `mode`   - Which channel to evaluate for the threshold.
    ///
    /// Returns an RGBA byte buffer (width * height * 4) with stripes painted
    /// over overexposed pixels and the original color elsewhere (alpha = 255).
    ///
    /// Also advances the internal animation counter so that successive calls
    /// produce a moving stripe effect.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply(&mut self, frame: &[u8], width: u32, height: u32, mode: ZebraMode) -> Vec<u8> {
        let n = (width * height) as usize;
        let mut out = vec![0u8; n * 4];

        // Stripe phase offset advances by stripe_width each frame for animation
        let phase = (self.frame_counter as i32 * self.config.stripe_width as i32)
            % (self.config.stripe_width as i32 * 2).max(1);

        let threshold = self.config.threshold;
        let stripe_w = self.config.stripe_width.max(1) as i32;
        let stripe_color = self.config.color;

        let mut overexposed_pixels = 0u32;
        let mut peak_ire: f32 = 0.0;

        for y in 0..height {
            for x in 0..width {
                let pixel_idx = ((y * width + x) * 3) as usize;
                if pixel_idx + 2 >= frame.len() {
                    continue;
                }
                let r = frame[pixel_idx];
                let g = frame[pixel_idx + 1];
                let b = frame[pixel_idx + 2];

                // Compute the value to threshold
                let value = match mode {
                    ZebraMode::Luma | ZebraMode::IRE => {
                        // BT.709 luma
                        0.2126 * (r as f32 / 255.0)
                            + 0.7152 * (g as f32 / 255.0)
                            + 0.0722 * (b as f32 / 255.0)
                    }
                    ZebraMode::R => r as f32 / 255.0,
                    ZebraMode::G => g as f32 / 255.0,
                    ZebraMode::B => b as f32 / 255.0,
                };

                let ire = value * 100.0;
                if ire > peak_ire {
                    peak_ire = ire;
                }

                let out_idx = ((y * width + x) * 4) as usize;

                if value >= threshold {
                    overexposed_pixels += 1;

                    // Diagonal stripe pattern: (x + y + phase) mod (2 * stripe_w)
                    let stripe_val = (x as i32 + y as i32 + phase).rem_euclid(stripe_w * 2);
                    if stripe_val < stripe_w {
                        // Stripe pixel: paint stripe color over original
                        let a = stripe_color[3] as f32 / 255.0;
                        let ia = 1.0 - a;
                        out[out_idx] = (stripe_color[0] as f32 * a + r as f32 * ia) as u8;
                        out[out_idx + 1] = (stripe_color[1] as f32 * a + g as f32 * ia) as u8;
                        out[out_idx + 2] = (stripe_color[2] as f32 * a + b as f32 * ia) as u8;
                        out[out_idx + 3] = 255;
                    } else {
                        // Alternate: show darkened original to contrast stripes
                        out[out_idx] = (r as f32 * 0.4) as u8;
                        out[out_idx + 1] = (g as f32 * 0.4) as u8;
                        out[out_idx + 2] = (b as f32 * 0.4) as u8;
                        out[out_idx + 3] = 255;
                    }
                } else {
                    // Not overexposed: pass through as RGBA
                    out[out_idx] = r;
                    out[out_idx + 1] = g;
                    out[out_idx + 2] = b;
                    out[out_idx + 3] = 255;
                }
            }
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);

        let _ = ZebraStats {
            overexposed_pixels,
            overexposed_pct: if n > 0 {
                (overexposed_pixels as f32 / n as f32) * 100.0
            } else {
                0.0
            },
            peak_ire,
        };

        out
    }

    /// Applies the zebra overlay and also returns statistics.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_with_stats(
        &mut self,
        frame: &[u8],
        width: u32,
        height: u32,
        mode: ZebraMode,
    ) -> (Vec<u8>, ZebraStats) {
        let n = (width * height) as usize;
        let mut out = vec![0u8; n * 4];

        let phase = (self.frame_counter as i32 * self.config.stripe_width as i32)
            % (self.config.stripe_width as i32 * 2).max(1);

        let threshold = self.config.threshold;
        let stripe_w = self.config.stripe_width.max(1) as i32;
        let stripe_color = self.config.color;

        let mut overexposed_pixels = 0u32;
        let mut peak_ire: f32 = 0.0;

        for y in 0..height {
            for x in 0..width {
                let pixel_idx = ((y * width + x) * 3) as usize;
                if pixel_idx + 2 >= frame.len() {
                    continue;
                }
                let r = frame[pixel_idx];
                let g = frame[pixel_idx + 1];
                let b = frame[pixel_idx + 2];

                let value = match mode {
                    ZebraMode::Luma | ZebraMode::IRE => {
                        0.2126 * (r as f32 / 255.0)
                            + 0.7152 * (g as f32 / 255.0)
                            + 0.0722 * (b as f32 / 255.0)
                    }
                    ZebraMode::R => r as f32 / 255.0,
                    ZebraMode::G => g as f32 / 255.0,
                    ZebraMode::B => b as f32 / 255.0,
                };

                let ire = value * 100.0;
                if ire > peak_ire {
                    peak_ire = ire;
                }

                let out_idx = ((y * width + x) * 4) as usize;

                if value >= threshold {
                    overexposed_pixels += 1;
                    let stripe_val = (x as i32 + y as i32 + phase).rem_euclid(stripe_w * 2);
                    if stripe_val < stripe_w {
                        let a = stripe_color[3] as f32 / 255.0;
                        let ia = 1.0 - a;
                        out[out_idx] = (stripe_color[0] as f32 * a + r as f32 * ia) as u8;
                        out[out_idx + 1] = (stripe_color[1] as f32 * a + g as f32 * ia) as u8;
                        out[out_idx + 2] = (stripe_color[2] as f32 * a + b as f32 * ia) as u8;
                        out[out_idx + 3] = 255;
                    } else {
                        out[out_idx] = (r as f32 * 0.4) as u8;
                        out[out_idx + 1] = (g as f32 * 0.4) as u8;
                        out[out_idx + 2] = (b as f32 * 0.4) as u8;
                        out[out_idx + 3] = 255;
                    }
                } else {
                    out[out_idx] = r;
                    out[out_idx + 1] = g;
                    out[out_idx + 2] = b;
                    out[out_idx + 3] = 255;
                }
            }
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);

        let stats = ZebraStats {
            overexposed_pixels,
            overexposed_pct: if n > 0 {
                (overexposed_pixels as f32 / n as f32) * 100.0
            } else {
                0.0
            },
            peak_ire,
        };

        (out, stats)
    }

    /// Returns the current frame counter (for animation state inspection).
    #[must_use]
    pub fn frame_counter(&self) -> u32 {
        self.frame_counter
    }

    /// Returns a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &ZebraConfig {
        &self.config
    }
}

impl Default for ZebraOverlay {
    fn default() -> Self {
        Self::new(ZebraConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a solid-color RGB24 frame.
    fn solid_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let n = (width * height * 3) as usize;
        let mut frame = vec![0u8; n];
        for chunk in frame.chunks_exact_mut(3) {
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
        frame
    }

    #[test]
    fn test_config_default() {
        let cfg = ZebraConfig::default();
        assert!((cfg.threshold - 0.95).abs() < 0.001);
        assert_eq!(cfg.stripe_width, 8);
    }

    #[test]
    fn test_apply_output_size() {
        let mut overlay = ZebraOverlay::default();
        let frame = solid_frame(64, 64, 128, 128, 128);
        let out = overlay.apply(&frame, 64, 64, ZebraMode::Luma);
        assert_eq!(out.len(), 64 * 64 * 4);
    }

    #[test]
    fn test_no_overexposure_passes_through() {
        let mut overlay = ZebraOverlay::default();
        // 50% gray — well below 95% threshold
        let frame = solid_frame(8, 8, 128, 128, 128);
        let out = overlay.apply(&frame, 8, 8, ZebraMode::Luma);
        // Verify pixels are passed through (R channel should be 128)
        for y in 0..8 {
            for x in 0..8 {
                let idx = (y * 8 + x) * 4;
                assert_eq!(out[idx], 128, "Red channel should be passed through");
            }
        }
    }

    #[test]
    fn test_full_white_fully_overexposed() {
        let mut overlay = ZebraOverlay::default();
        let frame = solid_frame(16, 16, 255, 255, 255);
        let (_, stats) = overlay.apply_with_stats(&frame, 16, 16, ZebraMode::Luma);
        assert_eq!(stats.overexposed_pixels, 16 * 16);
        assert!((stats.overexposed_pct - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_peak_ire_white() {
        let mut overlay = ZebraOverlay::default();
        let frame = solid_frame(8, 8, 255, 255, 255);
        let (_, stats) = overlay.apply_with_stats(&frame, 8, 8, ZebraMode::Luma);
        assert!((stats.peak_ire - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_peak_ire_black() {
        let mut overlay = ZebraOverlay::default();
        let frame = solid_frame(8, 8, 0, 0, 0);
        let (_, stats) = overlay.apply_with_stats(&frame, 8, 8, ZebraMode::Luma);
        assert!(stats.peak_ire < 1.0);
    }

    #[test]
    fn test_r_channel_mode() {
        let mut overlay = ZebraOverlay::new(ZebraConfig {
            threshold: 0.9,
            ..Default::default()
        });
        // Pure red at full — R channel should be 100% (overexposed in R mode)
        let frame = solid_frame(8, 8, 255, 0, 0);
        let (_, stats) = overlay.apply_with_stats(&frame, 8, 8, ZebraMode::R);
        assert_eq!(stats.overexposed_pixels, 64);
    }

    #[test]
    fn test_frame_counter_increments() {
        let mut overlay = ZebraOverlay::default();
        let frame = solid_frame(4, 4, 0, 0, 0);
        assert_eq!(overlay.frame_counter(), 0);
        let _ = overlay.apply(&frame, 4, 4, ZebraMode::Luma);
        assert_eq!(overlay.frame_counter(), 1);
        let _ = overlay.apply(&frame, 4, 4, ZebraMode::Luma);
        assert_eq!(overlay.frame_counter(), 2);
    }

    #[test]
    fn test_stripe_animation_changes_output() {
        // Two consecutive calls on an overexposed frame should produce
        // different outputs due to stripe phase shift
        let mut overlay = ZebraOverlay::new(ZebraConfig {
            stripe_width: 4,
            ..Default::default()
        });
        let frame = solid_frame(32, 32, 255, 255, 255);
        let out1 = overlay.apply(&frame, 32, 32, ZebraMode::Luma);
        let out2 = overlay.apply(&frame, 32, 32, ZebraMode::Luma);
        // Outputs should differ (stripe has moved)
        assert_ne!(
            out1, out2,
            "Animation should change stripe pattern each frame"
        );
    }
}
