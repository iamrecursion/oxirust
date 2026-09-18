//! Audio waveform and spectrum visualization for broadcast graphics overlays.
//!
//! Provides two complementary visualization types:
//! - **WaveformRenderer**: renders the time-domain audio signal as a waveform.
//! - **SpectrumRenderer**: renders the frequency-domain spectrum as vertical bars.
//!
//! Both produce RGBA pixel buffers and are designed for real-time use at
//! broadcast frame rates.

use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

/// Horizontal direction of waveform rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformOrientation {
    /// Waveform runs left-to-right (standard).
    Horizontal,
    /// Waveform runs top-to-bottom.
    Vertical,
}

impl Default for WaveformOrientation {
    fn default() -> Self {
        Self::Horizontal
    }
}

/// Style in which the waveform is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformStyle {
    /// Thin line connecting sample points.
    Line,
    /// Filled area from center to the sample value.
    Filled,
    /// Mirrored fill (symmetric above and below center).
    MirroredFill,
    /// Individual vertical bars per sample group.
    Bars,
}

impl Default for WaveformStyle {
    fn default() -> Self {
        Self::MirroredFill
    }
}

// ---------------------------------------------------------------------------
// Waveform configuration and renderer
// ---------------------------------------------------------------------------

/// Configuration for waveform rendering.
#[derive(Debug, Clone)]
pub struct WaveformConfig {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Waveform foreground color (RGBA).
    pub waveform_color: [u8; 4],
    /// Background color (RGBA). Use alpha=0 for transparent.
    pub bg_color: [u8; 4],
    /// Drawing style.
    pub style: WaveformStyle,
    /// Orientation.
    pub orientation: WaveformOrientation,
    /// Amplitude gain multiplier (1.0 = no gain).
    pub gain: f32,
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self {
            width: 320,
            height: 80,
            waveform_color: [0, 220, 100, 255],
            bg_color: [0, 0, 0, 192],
            style: WaveformStyle::MirroredFill,
            orientation: WaveformOrientation::Horizontal,
            gain: 1.0,
        }
    }
}

/// Waveform renderer.
pub struct WaveformRenderer;

impl WaveformRenderer {
    /// Render an audio waveform from PCM samples.
    ///
    /// # Parameters
    /// - `samples`: normalized audio samples in [-1.0, 1.0].
    /// - `config`: rendering configuration.
    ///
    /// Returns `Vec<u8>` of length `width * height * 4`.
    pub fn render(samples: &[f32], config: &WaveformConfig) -> Vec<u8> {
        let w = config.width as usize;
        let h = config.height as usize;
        let mut data = vec![0u8; w * h * 4];

        // Fill background.
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = config.bg_color[0];
            chunk[1] = config.bg_color[1];
            chunk[2] = config.bg_color[2];
            chunk[3] = config.bg_color[3];
        }

        if samples.is_empty() || w == 0 || h == 0 {
            return data;
        }

        let center_y = h as f32 * 0.5;

        match config.orientation {
            WaveformOrientation::Horizontal => {
                for col in 0..w {
                    let sample_idx = (col as f32 / w as f32 * (samples.len() - 1) as f32) as usize;
                    let sample = samples[sample_idx.min(samples.len() - 1)] * config.gain;
                    let clamped = sample.clamp(-1.0, 1.0);
                    let amplitude_px = (clamped * center_y).abs();

                    match config.style {
                        WaveformStyle::Line => {
                            let y = (center_y - clamped * center_y) as usize;
                            let y = y.min(h - 1);
                            write_pixel(&mut data, col, y, w, config.waveform_color);
                        }
                        WaveformStyle::Filled => {
                            let y_top = (center_y - clamped * center_y) as usize;
                            let y_bot = center_y as usize;
                            for row in y_top.min(y_bot)..=y_top.max(y_bot).min(h - 1) {
                                write_pixel(&mut data, col, row, w, config.waveform_color);
                            }
                        }
                        WaveformStyle::MirroredFill => {
                            let y_top = (center_y - amplitude_px) as usize;
                            let y_bot = (center_y + amplitude_px) as usize;
                            for row in y_top..=y_bot.min(h - 1) {
                                write_pixel(&mut data, col, row, w, config.waveform_color);
                            }
                        }
                        WaveformStyle::Bars => {
                            let bar_h = amplitude_px as usize;
                            let y_top = (center_y - bar_h as f32) as usize;
                            let y_bot = (center_y + bar_h as f32) as usize;
                            for row in y_top..=y_bot.min(h - 1) {
                                write_pixel(&mut data, col, row, w, config.waveform_color);
                            }
                        }
                    }
                }
            }
            WaveformOrientation::Vertical => {
                let center_x = w as f32 * 0.5;
                for row in 0..h {
                    let sample_idx = (row as f32 / h as f32 * (samples.len() - 1) as f32) as usize;
                    let sample = samples[sample_idx.min(samples.len() - 1)] * config.gain;
                    let clamped = sample.clamp(-1.0, 1.0);
                    let amplitude_px = (clamped * center_x).abs();
                    let x_left = (center_x - amplitude_px) as usize;
                    let x_right = (center_x + amplitude_px) as usize;
                    for col in x_left..=x_right.min(w - 1) {
                        write_pixel(&mut data, col, row, w, config.waveform_color);
                    }
                }
            }
        }

        data
    }
}

// ---------------------------------------------------------------------------
// Spectrum (FFT magnitude bars) renderer
// ---------------------------------------------------------------------------

/// Window function to apply before computing the spectrum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectrumWindow {
    /// Rectangular (no windowing).
    Rectangular,
    /// Hann (Hanning) window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
}

impl Default for SpectrumWindow {
    fn default() -> Self {
        Self::Hann
    }
}

/// Configuration for spectrum rendering.
#[derive(Debug, Clone)]
pub struct SpectrumConfig {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Number of frequency bins to display.
    pub num_bins: usize,
    /// Bar color (RGBA).
    pub bar_color: [u8; 4],
    /// Peak indicator color (RGBA).
    pub peak_color: [u8; 4],
    /// Background color (RGBA).
    pub bg_color: [u8; 4],
    /// Gap between bars in pixels.
    pub bar_gap_px: u32,
    /// Whether to show peak-hold indicators above bars.
    pub show_peaks: bool,
    /// Logarithmic frequency scale (true = log, false = linear).
    pub log_scale: bool,
    /// Floor level in dBFS for the minimum bar height.
    pub floor_db: f32,
    /// Window function applied to samples before DFT.
    pub window: SpectrumWindow,
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self {
            width: 320,
            height: 120,
            num_bins: 32,
            bar_color: [0, 150, 255, 255],
            peak_color: [255, 255, 255, 255],
            bg_color: [0, 0, 0, 192],
            bar_gap_px: 2,
            show_peaks: true,
            log_scale: true,
            floor_db: -60.0,
            window: SpectrumWindow::Hann,
        }
    }
}

/// Peak state per bin for the hold indicator.
#[derive(Debug, Clone)]
pub struct SpectrumPeakState {
    /// Peak hold value per bin (0.0..=1.0).
    pub peaks: Vec<f32>,
    /// Decay rate per second (fraction of peak released per second).
    pub decay_rate: f32,
}

impl SpectrumPeakState {
    /// Create peak state for the given number of bins.
    pub fn new(num_bins: usize) -> Self {
        Self {
            peaks: vec![0.0; num_bins],
            decay_rate: 0.3,
        }
    }

    /// Update peaks with new magnitudes and advance decay.
    pub fn update(&mut self, magnitudes: &[f32], dt_secs: f32) {
        for (i, &mag) in magnitudes.iter().enumerate() {
            if i < self.peaks.len() {
                if mag > self.peaks[i] {
                    self.peaks[i] = mag;
                } else {
                    self.peaks[i] = (self.peaks[i] - self.decay_rate * dt_secs).max(0.0);
                }
            }
        }
    }
}

/// Spectrum renderer.
pub struct SpectrumRenderer;

impl SpectrumRenderer {
    /// Compute a simple DFT-based magnitude spectrum from PCM samples.
    ///
    /// Returns `num_bins` magnitude values in [0.0, 1.0].
    pub fn compute_magnitudes(
        samples: &[f32],
        num_bins: usize,
        window: SpectrumWindow,
    ) -> Vec<f32> {
        if samples.is_empty() || num_bins == 0 {
            return vec![0.0; num_bins];
        }

        let n = samples.len();
        let windowed: Vec<f32> = samples
            .iter()
            .enumerate()
            .map(|(i, &s)| s * window_coeff(i, n, window))
            .collect();

        // Compute DFT magnitudes for the requested bins.
        // Use only the lower half (positive frequencies).
        let half = (n / 2).max(1);
        let bin_step = half.max(1) as f32 / num_bins as f32;

        let mut magnitudes = Vec::with_capacity(num_bins);
        for bin_idx in 0..num_bins {
            let freq_bin = (bin_idx as f32 * bin_step) as usize;
            let freq_bin = freq_bin.min(half - 1);

            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            for (k, &s) in windowed.iter().enumerate() {
                let angle = -2.0 * PI * freq_bin as f32 * k as f32 / n as f32;
                re += s * angle.cos();
                im += s * angle.sin();
            }
            let mag = (re * re + im * im).sqrt() / n as f32;
            magnitudes.push(mag);
        }

        // Normalize to [0, 1].
        let max_mag = magnitudes.iter().cloned().fold(0.0_f32, f32::max);
        if max_mag > 0.0 {
            for m in &mut magnitudes {
                *m /= max_mag;
            }
        }

        magnitudes
    }

    /// Render a spectrum visualization from magnitude values.
    ///
    /// # Parameters
    /// - `magnitudes`: per-bin magnitude values in [0.0, 1.0].
    /// - `peaks`: optional peak-hold state.
    /// - `config`: rendering configuration.
    ///
    /// Returns `Vec<u8>` of length `width * height * 4`.
    pub fn render(
        magnitudes: &[f32],
        peaks: Option<&SpectrumPeakState>,
        config: &SpectrumConfig,
    ) -> Vec<u8> {
        let w = config.width as usize;
        let h = config.height as usize;
        let mut data = vec![0u8; w * h * 4];

        // Fill background.
        for chunk in data.chunks_exact_mut(4) {
            chunk[0] = config.bg_color[0];
            chunk[1] = config.bg_color[1];
            chunk[2] = config.bg_color[2];
            chunk[3] = config.bg_color[3];
        }

        if magnitudes.is_empty() || w == 0 || h == 0 {
            return data;
        }

        let num_bins = magnitudes.len().min(config.num_bins);
        let gap = config.bar_gap_px as usize;
        let total_gap = gap * (num_bins + 1);
        let bar_w = if num_bins > 0 && total_gap < w {
            (w - total_gap) / num_bins
        } else {
            1
        };

        for (bin_idx, &mag) in magnitudes.iter().take(num_bins).enumerate() {
            let bar_h = (mag * h as f32) as usize;
            let x_start = gap + bin_idx * (bar_w + gap);
            let x_end = (x_start + bar_w).min(w);
            let y_top = h.saturating_sub(bar_h);

            // Draw bar.
            for row in y_top..h {
                for col in x_start..x_end {
                    write_pixel(&mut data, col, row, w, config.bar_color);
                }
            }

            // Draw peak indicator.
            if config.show_peaks {
                if let Some(peak_state) = peaks {
                    if let Some(&peak) = peak_state.peaks.get(bin_idx) {
                        let peak_row = h.saturating_sub((peak * h as f32) as usize);
                        if peak_row < h {
                            for col in x_start..x_end {
                                write_pixel(&mut data, col, peak_row, w, config.peak_color);
                            }
                        }
                    }
                }
            }
        }

        data
    }
}

/// Compute a window function coefficient.
fn window_coeff(i: usize, n: usize, window: SpectrumWindow) -> f32 {
    if n <= 1 {
        return 1.0;
    }
    let x = i as f32 / (n - 1) as f32;
    match window {
        SpectrumWindow::Rectangular => 1.0,
        SpectrumWindow::Hann => 0.5 * (1.0 - (2.0 * PI * x).cos()),
        SpectrumWindow::Hamming => 0.54 - 0.46 * (2.0 * PI * x).cos(),
        SpectrumWindow::Blackman => 0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos(),
    }
}

/// Write an RGBA pixel into the buffer at (col, row).
fn write_pixel(data: &mut [u8], col: usize, row: usize, width: usize, color: [u8; 4]) {
    let idx = (row * width + col) * 4;
    if idx + 3 < data.len() {
        let a = color[3] as f32 / 255.0;
        let inv_a = 1.0 - a;
        data[idx] = (color[0] as f32 * a + data[idx] as f32 * inv_a) as u8;
        data[idx + 1] = (color[1] as f32 * a + data[idx + 1] as f32 * inv_a) as u8;
        data[idx + 2] = (color[2] as f32 * a + data[idx + 2] as f32 * inv_a) as u8;
        data[idx + 3] = ((a + data[idx + 3] as f32 / 255.0 * inv_a) * 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_samples(freq_hz: f32, sample_rate: u32, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    // -----------------------------------------------------------------------
    // WaveformRenderer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_waveform_output_size() {
        let samples = sine_samples(440.0, 44100, 0.01);
        let cfg = WaveformConfig::default();
        let data = WaveformRenderer::render(&samples, &cfg);
        assert_eq!(data.len(), (cfg.width * cfg.height * 4) as usize);
    }

    #[test]
    fn test_waveform_empty_samples_fills_bg() {
        let cfg = WaveformConfig {
            bg_color: [50, 0, 0, 255],
            ..WaveformConfig::default()
        };
        let data = WaveformRenderer::render(&[], &cfg);
        assert_eq!(data[0], 50);
    }

    #[test]
    fn test_waveform_renders_nonzero_with_sine() {
        let samples = sine_samples(440.0, 44100, 0.01);
        let cfg = WaveformConfig {
            style: WaveformStyle::MirroredFill,
            ..WaveformConfig::default()
        };
        let data = WaveformRenderer::render(&samples, &cfg);
        let waveform_pixels = data
            .chunks_exact(4)
            .filter(|p| p[0] != cfg.bg_color[0] || p[1] != cfg.bg_color[1])
            .count();
        assert!(waveform_pixels > 0);
    }

    #[test]
    fn test_waveform_style_line() {
        let samples = vec![0.5; 100];
        let cfg = WaveformConfig {
            width: 100,
            height: 40,
            style: WaveformStyle::Line,
            ..WaveformConfig::default()
        };
        let data = WaveformRenderer::render(&samples, &cfg);
        assert_eq!(data.len(), 100 * 40 * 4);
    }

    #[test]
    fn test_waveform_style_filled() {
        let samples = vec![0.5; 100];
        let cfg = WaveformConfig {
            width: 100,
            height: 40,
            style: WaveformStyle::Filled,
            ..WaveformConfig::default()
        };
        let data = WaveformRenderer::render(&samples, &cfg);
        assert_eq!(data.len(), 100 * 40 * 4);
    }

    #[test]
    fn test_waveform_style_bars() {
        let samples = vec![0.8; 100];
        let cfg = WaveformConfig {
            width: 100,
            height: 40,
            style: WaveformStyle::Bars,
            ..WaveformConfig::default()
        };
        let data = WaveformRenderer::render(&samples, &cfg);
        assert_eq!(data.len(), 100 * 40 * 4);
    }

    #[test]
    fn test_waveform_orientation_vertical() {
        let samples = sine_samples(220.0, 44100, 0.01);
        let cfg = WaveformConfig {
            width: 80,
            height: 320,
            orientation: WaveformOrientation::Vertical,
            ..WaveformConfig::default()
        };
        let data = WaveformRenderer::render(&samples, &cfg);
        assert_eq!(data.len(), 80 * 320 * 4);
    }

    #[test]
    fn test_waveform_gain_amplifies() {
        let samples = vec![0.1; 320];
        let cfg_normal = WaveformConfig {
            gain: 1.0,
            style: WaveformStyle::MirroredFill,
            ..WaveformConfig::default()
        };
        let cfg_amplified = WaveformConfig {
            gain: 5.0,
            style: WaveformStyle::MirroredFill,
            ..WaveformConfig::default()
        };
        let data_normal = WaveformRenderer::render(&samples, &cfg_normal);
        let data_amplified = WaveformRenderer::render(&samples, &cfg_amplified);
        // Amplified should have more non-bg pixels.
        let count = |data: &[u8], bg: [u8; 4]| -> usize {
            data.chunks_exact(4).filter(|p| p[0] != bg[0]).count()
        };
        let n = count(&data_normal, cfg_normal.bg_color);
        let a = count(&data_amplified, cfg_amplified.bg_color);
        assert!(a >= n);
    }

    // -----------------------------------------------------------------------
    // SpectrumRenderer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_spectrum_magnitudes_sine() {
        let samples = sine_samples(440.0, 44100, 0.02);
        let mags = SpectrumRenderer::compute_magnitudes(&samples, 16, SpectrumWindow::Hann);
        assert_eq!(mags.len(), 16);
        // Magnitudes should be in [0, 1].
        for &m in &mags {
            assert!(m >= 0.0 && m <= 1.0 + f32::EPSILON);
        }
    }

    #[test]
    fn test_spectrum_magnitudes_empty() {
        let mags = SpectrumRenderer::compute_magnitudes(&[], 8, SpectrumWindow::Rectangular);
        assert_eq!(mags.len(), 8);
        assert!(mags.iter().all(|&m| m == 0.0));
    }

    #[test]
    fn test_spectrum_output_size() {
        let samples = sine_samples(440.0, 44100, 0.02);
        let mags = SpectrumRenderer::compute_magnitudes(&samples, 32, SpectrumWindow::Hann);
        let cfg = SpectrumConfig::default();
        let data = SpectrumRenderer::render(&mags, None, &cfg);
        assert_eq!(data.len(), (cfg.width * cfg.height * 4) as usize);
    }

    #[test]
    fn test_spectrum_renders_nonzero() {
        let samples = sine_samples(440.0, 44100, 0.02);
        let mags = SpectrumRenderer::compute_magnitudes(&samples, 32, SpectrumWindow::Hann);
        let cfg = SpectrumConfig::default();
        let data = SpectrumRenderer::render(&mags, None, &cfg);
        let has_bar_pixels = data.iter().any(|&b| b > 0);
        assert!(has_bar_pixels);
    }

    #[test]
    fn test_spectrum_with_peaks() {
        let samples = sine_samples(440.0, 44100, 0.02);
        let mags = SpectrumRenderer::compute_magnitudes(&samples, 32, SpectrumWindow::Hann);
        let mut peaks = SpectrumPeakState::new(32);
        peaks.update(&mags, 1.0 / 30.0);
        let cfg = SpectrumConfig {
            show_peaks: true,
            ..SpectrumConfig::default()
        };
        let data = SpectrumRenderer::render(&mags, Some(&peaks), &cfg);
        assert_eq!(data.len(), (cfg.width * cfg.height * 4) as usize);
    }

    #[test]
    fn test_spectrum_peak_state_decay() {
        let mut peaks = SpectrumPeakState::new(4);
        peaks.peaks = vec![1.0, 0.5, 0.0, 0.8];
        let mags = vec![0.0; 4]; // No new energy
        peaks.update(&mags, 1.0); // 1 second decay
        assert!(peaks.peaks[0] < 1.0);
        assert!(peaks.peaks[1] < 0.5);
        assert!((peaks.peaks[2]).abs() < f32::EPSILON);
    }

    #[test]
    fn test_spectrum_peak_state_updates_max() {
        let mut peaks = SpectrumPeakState::new(2);
        let mags = vec![0.8, 0.3];
        peaks.update(&mags, 0.016);
        assert!((peaks.peaks[0] - 0.8).abs() < f32::EPSILON);
        assert!((peaks.peaks[1] - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_window_coefficients_sum_nonzero() {
        let coeff: f32 = (0..64)
            .map(|i| window_coeff(i, 64, SpectrumWindow::Hann))
            .sum();
        assert!(coeff > 0.0);
    }

    #[test]
    fn test_window_hamming_range() {
        for i in 0..64 {
            let c = window_coeff(i, 64, SpectrumWindow::Hamming);
            assert!(c >= 0.0 && c <= 1.0 + f32::EPSILON);
        }
    }

    #[test]
    fn test_window_blackman_range() {
        for i in 0..64 {
            let c = window_coeff(i, 64, SpectrumWindow::Blackman);
            assert!(c >= -0.01 && c <= 1.0 + f32::EPSILON);
        }
    }

    #[test]
    fn test_waveform_orientation_default() {
        assert_eq!(
            WaveformOrientation::default(),
            WaveformOrientation::Horizontal
        );
    }

    #[test]
    fn test_waveform_style_default() {
        assert_eq!(WaveformStyle::default(), WaveformStyle::MirroredFill);
    }

    #[test]
    fn test_spectrum_window_default() {
        assert_eq!(SpectrumWindow::default(), SpectrumWindow::Hann);
    }
}
