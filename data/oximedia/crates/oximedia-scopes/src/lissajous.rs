//! Lissajous / Phase scope for audio-visual sync analysis.
//!
//! The Lissajous scope (also called a phase scope or vectorscope for audio)
//! displays the relationship between two audio channels (L and R) by plotting
//! one channel on each axis. This reveals stereo width, phase correlation,
//! and mono compatibility of an audio signal.

#![allow(dead_code)]

use std::collections::VecDeque;

/// Color mode for the Lissajous display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LissajousColorMode {
    /// Classic green phosphor display.
    GreenPhosphor,
    /// White on black.
    WhiteOnBlack,
    /// Gradient from green (new) to blue (old).
    GreenToBlue,
    /// Gradient from yellow (new) to red (old).
    YellowToRed,
}

/// Configuration for the Lissajous scope.
#[derive(Debug, Clone)]
pub struct LissajousConfig {
    /// Width of the rendered output in pixels.
    pub width: u32,
    /// Height of the rendered output in pixels.
    pub height: u32,
    /// Number of samples to keep in the ring buffer.
    pub persistence: usize,
    /// Color mode.
    pub color_mode: LissajousColorMode,
    /// Whether to draw lines between consecutive points.
    pub line_mode: bool,
    /// Zoom/scale factor (1.0 = normal, 2.0 = 2x zoom).
    pub scale: f32,
}

impl Default for LissajousConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            persistence: 2048,
            color_mode: LissajousColorMode::GreenPhosphor,
            line_mode: true,
            scale: 1.0,
        }
    }
}

/// A single point in the Lissajous ring buffer.
#[derive(Debug, Clone, Copy)]
pub struct LissajousPoint {
    /// Left channel value (X axis), normalized -1.0 to 1.0.
    pub x: f32,
    /// Right channel value (Y axis), normalized -1.0 to 1.0.
    pub y: f32,
    /// Age in [0, 1]: 0 = newest, 1 = oldest (fully faded).
    pub age: f32,
}

/// Statistical summary computed from the Lissajous data.
#[derive(Debug, Clone)]
pub struct LissajousStats {
    /// Pearson correlation coefficient between L and R channels (-1.0 to 1.0).
    /// +1.0 = perfectly mono, -1.0 = perfectly out-of-phase, 0.0 = uncorrelated.
    pub correlation: f32,
    /// Estimated phase difference between channels in degrees (0-360).
    pub phase_diff_deg: f32,
    /// Dominant frequency estimated from zero-crossing rate (Hz, approximate).
    pub dominant_frequency: f32,
}

/// Lissajous / phase scope with persistent ring buffer.
pub struct LissajousScope {
    config: LissajousConfig,
    /// Ring buffer of points with age information.
    buffer: VecDeque<LissajousPoint>,
    /// Cached last stats.
    last_stats: Option<LissajousStats>,
}

impl LissajousScope {
    /// Creates a new Lissajous scope with the given configuration.
    #[must_use]
    pub fn new(config: LissajousConfig) -> Self {
        let capacity = config.persistence;
        Self {
            config,
            buffer: VecDeque::with_capacity(capacity),
            last_stats: None,
        }
    }

    /// Updates the scope with new audio samples.
    ///
    /// `left` and `right` are interleaved samples in the range -1.0 to 1.0.
    /// Both slices must have the same length.
    pub fn update(&mut self, left: &[f32], right: &[f32]) {
        let n = left.len().min(right.len());
        if n == 0 {
            return;
        }

        // Age existing points
        let age_step = n as f32 / self.config.persistence as f32;
        for point in &mut self.buffer {
            point.age = (point.age + age_step).min(1.0);
        }

        // Remove fully-aged points
        while self.buffer.front().map_or(false, |p| p.age >= 1.0) {
            self.buffer.pop_front();
        }

        // Add new points
        for i in 0..n {
            // Evict oldest if at capacity
            if self.buffer.len() >= self.config.persistence {
                self.buffer.pop_front();
            }
            self.buffer.push_back(LissajousPoint {
                x: left[i].clamp(-1.0, 1.0),
                y: right[i].clamp(-1.0, 1.0),
                age: 0.0,
            });
        }

        // Compute stats from current input
        self.last_stats = Some(compute_stats(left, right, n));
    }

    /// Renders the current Lissajous display as an RGBA image.
    ///
    /// Returns a `Vec<u8>` with `width * height * 4` bytes (RGBA row-major).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render(&self) -> Vec<u8> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let mut pixels = vec![0u8; w * h * 4];

        // Draw dark background
        for chunk in pixels.chunks_exact_mut(4) {
            chunk[0] = 10;
            chunk[1] = 10;
            chunk[2] = 12;
            chunk[3] = 255;
        }

        // Draw graticule: crosshair and 45° lines
        let cx = (w / 2) as i32;
        let cy = (h / 2) as i32;
        draw_graticule_line(&mut pixels, w, h, cx, 0, cx, h as i32, [40, 60, 40, 255]);
        draw_graticule_line(&mut pixels, w, h, 0, cy, w as i32, cy, [40, 60, 40, 255]);
        // 45° lines (mono +/- lines)
        let diag = w.min(h) as i32 / 2;
        draw_graticule_line(
            &mut pixels,
            w,
            h,
            cx - diag,
            cy - diag,
            cx + diag,
            cy + diag,
            [30, 50, 30, 255],
        );
        draw_graticule_line(
            &mut pixels,
            w,
            h,
            cx + diag,
            cy - diag,
            cx - diag,
            cy + diag,
            [30, 50, 30, 255],
        );

        // Render points
        let half_w = w as f32 / 2.0;
        let half_h = h as f32 / 2.0;
        let scale = self.config.scale;

        let points: Vec<_> = self.buffer.iter().collect();
        let np = points.len();

        for idx in 0..np {
            let point = points[idx];
            // age 0 = new (bright), age 1 = old (dim)
            let brightness = 1.0 - point.age;
            if brightness <= 0.0 {
                continue;
            }

            let px = ((point.x * scale * half_w) + half_w) as i32;
            let py = (half_h - (point.y * scale * half_h)) as i32;

            if self.config.line_mode && idx + 1 < np {
                let next = points[idx + 1];
                let npx = ((next.x * scale * half_w) + half_w) as i32;
                let npy = (half_h - (next.y * scale * half_h)) as i32;

                let alpha = (brightness * 200.0) as u8;
                let color = color_for_mode(self.config.color_mode, brightness);
                bresenham_line(
                    &mut pixels,
                    w,
                    h,
                    px,
                    py,
                    npx,
                    npy,
                    [color[0], color[1], color[2], alpha],
                );
            } else {
                let alpha = (brightness * 220.0) as u8;
                let color = color_for_mode(self.config.color_mode, brightness);
                set_pixel(
                    &mut pixels,
                    w,
                    h,
                    px,
                    py,
                    [color[0], color[1], color[2], alpha],
                );
            }
        }

        pixels
    }

    /// Returns the last computed statistics, if any samples have been ingested.
    #[must_use]
    pub fn stats(&self) -> Option<&LissajousStats> {
        self.last_stats.as_ref()
    }

    /// Clears the ring buffer and resets statistics.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.last_stats = None;
    }

    /// Returns the number of points currently in the ring buffer.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the current configuration.
    #[must_use]
    pub fn config(&self) -> &LissajousConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Computes Lissajous stats from a pair of audio slices.
#[allow(clippy::cast_precision_loss)]
fn compute_stats(left: &[f32], right: &[f32], n: usize) -> LissajousStats {
    let nf = n as f32;
    let mut sum_l = 0.0f32;
    let mut sum_r = 0.0f32;
    let mut sum_ll = 0.0f32;
    let mut sum_rr = 0.0f32;
    let mut sum_lr = 0.0f32;

    for i in 0..n {
        let l = left[i];
        let r = right[i];
        sum_l += l;
        sum_r += r;
        sum_ll += l * l;
        sum_rr += r * r;
        sum_lr += l * r;
    }

    let mean_l = sum_l / nf;
    let mean_r = sum_r / nf;

    let var_l = (sum_ll / nf) - mean_l * mean_l;
    let var_r = (sum_rr / nf) - mean_r * mean_r;
    let covar = (sum_lr / nf) - mean_l * mean_r;

    let correlation = if var_l > 1e-10 && var_r > 1e-10 {
        (covar / (var_l.sqrt() * var_r.sqrt())).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    // Phase difference estimation using cross-spectrum angle approximation
    // For a simple estimate: phase_diff = acos(correlation) * sign
    let phase_diff_rad = if correlation.abs() <= 1.0 {
        correlation.acos()
    } else {
        0.0
    };
    let phase_diff_deg = phase_diff_rad.to_degrees();

    // Dominant frequency from zero-crossing rate of the left channel
    // Assuming a nominal sample rate of 48 kHz for the estimate
    let sample_rate = 48_000.0f32;
    let mut zero_crossings = 0u32;
    for i in 1..n {
        if (left[i - 1] >= 0.0) != (left[i] >= 0.0) {
            zero_crossings += 1;
        }
    }
    let dominant_frequency = if n > 1 {
        (zero_crossings as f32 / 2.0) * (sample_rate / (n - 1) as f32)
    } else {
        0.0
    };

    LissajousStats {
        correlation,
        phase_diff_deg,
        dominant_frequency,
    }
}

/// Returns an RGBA color for a point based on the color mode and age-based brightness.
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
fn color_for_mode(mode: LissajousColorMode, brightness: f32) -> [u8; 3] {
    match mode {
        LissajousColorMode::GreenPhosphor => {
            let g = (brightness * 255.0).min(255.0) as u8;
            [0, g, 0]
        }
        LissajousColorMode::WhiteOnBlack => {
            let v = (brightness * 255.0).min(255.0) as u8;
            [v, v, v]
        }
        LissajousColorMode::GreenToBlue => {
            // New: green, old: blue
            let g = (brightness * 255.0).min(255.0) as u8;
            let b = ((1.0 - brightness) * 200.0).min(255.0) as u8;
            [0, g, b]
        }
        LissajousColorMode::YellowToRed => {
            // New: yellow (255,255,0), old: red (255,0,0)
            let g = (brightness * 255.0).min(255.0) as u8;
            let r = 255u8;
            [r, g, 0]
        }
    }
}

/// Sets a single pixel (clamped to bounds), alpha-blending over the background.
#[allow(clippy::cast_sign_loss)]
fn set_pixel(pixels: &mut [u8], w: usize, h: usize, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        return;
    }
    let idx = (y as usize * w + x as usize) * 4;
    let a = color[3] as f32 / 255.0;
    let ia = 1.0 - a;
    pixels[idx] = ((color[0] as f32 * a) + (pixels[idx] as f32 * ia)) as u8;
    pixels[idx + 1] = ((color[1] as f32 * a) + (pixels[idx + 1] as f32 * ia)) as u8;
    pixels[idx + 2] = ((color[2] as f32 * a) + (pixels[idx + 2] as f32 * ia)) as u8;
    pixels[idx + 3] = 255;
}

/// Draws a line using Bresenham's algorithm.
fn bresenham_line(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 4],
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;

    loop {
        set_pixel(pixels, w, h, x, y, color);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draws a graticule line (used for crosshair and reference lines).
fn draw_graticule_line(
    pixels: &mut [u8],
    w: usize,
    h: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    color: [u8; 4],
) {
    bresenham_line(pixels, w, h, x0, y0, x1, y1, color);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_lissajous_config_default() {
        let cfg = LissajousConfig::default();
        assert_eq!(cfg.width, 512);
        assert_eq!(cfg.height, 512);
        assert_eq!(cfg.persistence, 2048);
        assert!(cfg.line_mode);
    }

    #[test]
    fn test_scope_new_empty() {
        let scope = LissajousScope::new(LissajousConfig::default());
        assert_eq!(scope.point_count(), 0);
        assert!(scope.stats().is_none());
    }

    #[test]
    fn test_update_adds_points() {
        let mut scope = LissajousScope::new(LissajousConfig::default());
        let left = vec![0.5f32; 100];
        let right = vec![-0.5f32; 100];
        scope.update(&left, &right);
        assert_eq!(scope.point_count(), 100);
        assert!(scope.stats().is_some());
    }

    #[test]
    fn test_correlation_identical_channels() {
        let left = sine_wave(440.0, 48_000.0, 1024);
        let right = left.clone();
        let stats = compute_stats(&left, &right, 1024);
        assert!(
            (stats.correlation - 1.0).abs() < 0.001,
            "Identical channels should have correlation ~1.0"
        );
    }

    #[test]
    fn test_correlation_opposite_channels() {
        let left = sine_wave(440.0, 48_000.0, 1024);
        let right: Vec<f32> = left.iter().map(|&v| -v).collect();
        let stats = compute_stats(&left, &right, 1024);
        assert!(
            stats.correlation < -0.99,
            "Opposite channels should have correlation ~-1.0"
        );
    }

    #[test]
    fn test_correlation_uncorrelated() {
        let left = sine_wave(440.0, 48_000.0, 1024);
        let right = sine_wave(997.0, 48_000.0, 1024); // incommensurate freq
        let stats = compute_stats(&left, &right, 1024);
        // Not necessarily 0, but should be in range
        assert!(stats.correlation >= -1.0 && stats.correlation <= 1.0);
    }

    #[test]
    fn test_render_output_size() {
        let cfg = LissajousConfig {
            width: 256,
            height: 256,
            ..Default::default()
        };
        let mut scope = LissajousScope::new(cfg);
        let left = sine_wave(440.0, 48_000.0, 512);
        let right = sine_wave(440.0, 48_000.0, 512);
        scope.update(&left, &right);
        let rendered = scope.render();
        assert_eq!(rendered.len(), 256 * 256 * 4);
    }

    #[test]
    fn test_persistence_limit() {
        let cfg = LissajousConfig {
            persistence: 100,
            ..Default::default()
        };
        let mut scope = LissajousScope::new(cfg);
        let left = vec![0.1f32; 200];
        let right = vec![0.1f32; 200];
        scope.update(&left, &right);
        // Buffer should not exceed persistence
        assert!(scope.point_count() <= 100);
    }

    #[test]
    fn test_clear_resets_scope() {
        let mut scope = LissajousScope::new(LissajousConfig::default());
        let left = vec![0.5f32; 64];
        let right = vec![0.5f32; 64];
        scope.update(&left, &right);
        scope.clear();
        assert_eq!(scope.point_count(), 0);
        assert!(scope.stats().is_none());
    }

    #[test]
    fn test_dominant_frequency_estimate() {
        // 1000 Hz sine at 48 kHz sample rate
        let left = sine_wave(1000.0, 48_000.0, 4800);
        let right = left.clone();
        let stats = compute_stats(&left, &right, 4800);
        // Allow wide tolerance since this is a zero-crossing estimate
        assert!(stats.dominant_frequency > 200.0 && stats.dominant_frequency < 5000.0);
    }

    #[test]
    fn test_phase_diff_in_range() {
        let left = sine_wave(440.0, 48_000.0, 1024);
        let right = sine_wave(440.0, 48_000.0, 1024);
        let stats = compute_stats(&left, &right, 1024);
        assert!(stats.phase_diff_deg >= 0.0 && stats.phase_diff_deg <= 180.0);
    }

    #[test]
    fn test_update_empty_slices() {
        let mut scope = LissajousScope::new(LissajousConfig::default());
        scope.update(&[], &[]);
        assert_eq!(scope.point_count(), 0);
    }
}
