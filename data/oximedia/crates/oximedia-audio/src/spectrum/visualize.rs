//! Visualization utilities for spectrum analysis.

use super::analyzer::SpectrumData;
use super::features::BandEnergy;

/// VU meter configuration.
#[derive(Clone, Debug)]
pub struct VuMeterConfig {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Minimum dB level.
    pub min_db: f64,
    /// Maximum dB level.
    pub max_db: f64,
    /// Background color.
    pub background_color: [u8; 3],
    /// Normal range color (green).
    pub normal_color: [u8; 3],
    /// Warning range color (yellow).
    pub warning_color: [u8; 3],
    /// Danger range color (red).
    pub danger_color: [u8; 3],
    /// Peak marker color.
    pub peak_color: [u8; 3],
    /// Warning threshold (dB).
    pub warning_threshold: f64,
    /// Danger threshold (dB).
    pub danger_threshold: f64,
}

impl VuMeterConfig {
    /// Create a new VU meter configuration.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            min_db: -60.0,
            max_db: 0.0,
            background_color: [32, 32, 32],
            normal_color: [0, 255, 0],
            warning_color: [255, 255, 0],
            danger_color: [255, 0, 0],
            peak_color: [255, 255, 255],
            warning_threshold: -12.0,
            danger_threshold: -3.0,
        }
    }
}

impl Default for VuMeterConfig {
    fn default() -> Self {
        Self::new(40, 400)
    }
}

/// VU meter image.
#[derive(Clone, Debug)]
pub struct VuMeterImage {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// RGB pixel data.
    pub data: Vec<u8>,
}

impl VuMeterImage {
    /// Create a new VU meter image.
    #[must_use]
    pub fn new(width: usize, height: usize, background_color: [u8; 3]) -> Self {
        let mut data = vec![0; width * height * 3];

        for i in 0..width * height {
            data[i * 3] = background_color[0];
            data[i * 3 + 1] = background_color[1];
            data[i * 3 + 2] = background_color[2];
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Set pixel at (x, y).
    pub fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) * 3;
        self.data[idx] = color[0];
        self.data[idx + 1] = color[1];
        self.data[idx + 2] = color[2];
    }

    /// Fill horizontal line.
    pub fn fill_horizontal(&mut self, y: usize, color: [u8; 3]) {
        for x in 0..self.width {
            self.set_pixel(x, y, color);
        }
    }

    /// Fill rectangle.
    pub fn fill_rect(&mut self, x: usize, y: usize, width: usize, height: usize, color: [u8; 3]) {
        for dy in 0..height {
            for dx in 0..width {
                self.set_pixel(x + dx, y + dy, color);
            }
        }
    }
}

/// VU meter renderer.
pub struct VuMeter {
    config: VuMeterConfig,
    peak_hold: f64,
    peak_decay: f64,
}

impl VuMeter {
    /// Create a new VU meter.
    #[must_use]
    pub fn new(config: VuMeterConfig) -> Self {
        Self {
            config,
            peak_hold: -100.0,
            peak_decay: 0.995,
        }
    }

    /// Update meter with new level (in dB).
    pub fn update(&mut self, level_db: f64) {
        if level_db > self.peak_hold {
            self.peak_hold = level_db;
        } else {
            self.peak_hold *= self.peak_decay;
        }
    }

    /// Render VU meter.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render(&self, level_db: f64) -> VuMeterImage {
        let mut image = VuMeterImage::new(
            self.config.width,
            self.config.height,
            self.config.background_color,
        );

        // Calculate fill height
        let normalized = ((level_db - self.config.min_db)
            / (self.config.max_db - self.config.min_db))
            .clamp(0.0, 1.0);

        let fill_height = (normalized * self.config.height as f64) as usize;

        // Render meter from bottom to top
        for y in 0..fill_height {
            let actual_y = self.config.height - 1 - y;
            let db_at_y = self.config.min_db
                + (y as f64 / self.config.height as f64)
                    * (self.config.max_db - self.config.min_db);

            let color = if db_at_y >= self.config.danger_threshold {
                self.config.danger_color
            } else if db_at_y >= self.config.warning_threshold {
                self.config.warning_color
            } else {
                self.config.normal_color
            };

            image.fill_horizontal(actual_y, color);
        }

        // Draw peak marker
        let peak_normalized = ((self.peak_hold - self.config.min_db)
            / (self.config.max_db - self.config.min_db))
            .clamp(0.0, 1.0);

        let peak_y =
            self.config.height - 1 - (peak_normalized * self.config.height as f64) as usize;
        if peak_y < self.config.height {
            image.fill_horizontal(peak_y, self.config.peak_color);
        }

        image
    }

    /// Reset peak hold.
    pub fn reset(&mut self) {
        self.peak_hold = -100.0;
    }
}

/// Spectrum visualizer configuration.
#[derive(Clone, Debug)]
pub struct SpectrumVisualizerConfig {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Minimum frequency to display (Hz).
    pub min_freq: f64,
    /// Maximum frequency to display (Hz).
    pub max_freq: f64,
    /// Minimum dB level.
    pub min_db: f64,
    /// Maximum dB level.
    pub max_db: f64,
    /// Background color.
    pub background_color: [u8; 3],
    /// Bar color.
    pub bar_color: [u8; 3],
    /// Peak marker color.
    pub peak_color: [u8; 3],
    /// Grid color (None for no grid).
    pub grid_color: Option<[u8; 3]>,
    /// Number of frequency bars.
    pub num_bars: usize,
    /// Bar spacing (pixels).
    pub bar_spacing: usize,
    /// Use logarithmic frequency scale.
    pub log_scale: bool,
}

impl SpectrumVisualizerConfig {
    /// Create a new spectrum visualizer configuration.
    #[must_use]
    pub fn new(width: usize, height: usize, num_bars: usize) -> Self {
        Self {
            width,
            height,
            min_freq: 20.0,
            max_freq: 20000.0,
            min_db: -80.0,
            max_db: 0.0,
            background_color: [0, 0, 0],
            bar_color: [0, 255, 128],
            peak_color: [255, 0, 0],
            grid_color: Some([64, 64, 64]),
            num_bars,
            bar_spacing: 2,
            log_scale: true,
        }
    }
}

impl Default for SpectrumVisualizerConfig {
    fn default() -> Self {
        Self::new(800, 400, 64)
    }
}

/// Spectrum visualizer image.
#[derive(Clone, Debug)]
pub struct SpectrumVisualizerImage {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// RGB pixel data.
    pub data: Vec<u8>,
}

impl SpectrumVisualizerImage {
    /// Create a new spectrum visualizer image.
    #[must_use]
    pub fn new(width: usize, height: usize, background_color: [u8; 3]) -> Self {
        let mut data = vec![0; width * height * 3];

        for i in 0..width * height {
            data[i * 3] = background_color[0];
            data[i * 3 + 1] = background_color[1];
            data[i * 3 + 2] = background_color[2];
        }

        Self {
            width,
            height,
            data,
        }
    }

    /// Set pixel at (x, y).
    pub fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) * 3;
        self.data[idx] = color[0];
        self.data[idx + 1] = color[1];
        self.data[idx + 2] = color[2];
    }

    /// Fill rectangle.
    pub fn fill_rect(&mut self, x: usize, y: usize, width: usize, height: usize, color: [u8; 3]) {
        for dy in 0..height {
            for dx in 0..width {
                if x + dx < self.width && y + dy < self.height {
                    self.set_pixel(x + dx, y + dy, color);
                }
            }
        }
    }

    /// Draw horizontal line.
    pub fn draw_horizontal_line(&mut self, y: usize, color: [u8; 3]) {
        for x in 0..self.width {
            self.set_pixel(x, y, color);
        }
    }
}

/// Spectrum visualizer.
pub struct SpectrumVisualizer {
    config: SpectrumVisualizerConfig,
    peak_holds: Vec<f64>,
    peak_decay: f64,
}

impl SpectrumVisualizer {
    /// Create a new spectrum visualizer.
    #[must_use]
    pub fn new(config: SpectrumVisualizerConfig) -> Self {
        let peak_holds = vec![-100.0; config.num_bars];

        Self {
            config,
            peak_holds,
            peak_decay: 0.95,
        }
    }

    /// Render spectrum visualization.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render(&mut self, spectrum: &SpectrumData) -> SpectrumVisualizerImage {
        let mut image = SpectrumVisualizerImage::new(
            self.config.width,
            self.config.height,
            self.config.background_color,
        );

        // Draw grid
        if let Some(grid_color) = self.config.grid_color {
            let h_step = self.config.height / 4;
            for i in 0..=4 {
                let y = i * h_step;
                if y < self.config.height {
                    image.draw_horizontal_line(y, grid_color);
                }
            }
        }

        // Calculate bar width
        let total_spacing = (self.config.num_bars - 1) * self.config.bar_spacing;
        let bar_width = (self.config.width - total_spacing) / self.config.num_bars;

        // Generate frequency bins for bars
        let freq_bins = self.generate_frequency_bins();

        // Render bars
        for (i, &(min_freq, max_freq)) in freq_bins.iter().enumerate() {
            // Calculate average magnitude in this frequency range
            let avg_db = self.average_magnitude_in_range(spectrum, min_freq, max_freq);

            // Update peak hold
            if avg_db > self.peak_holds[i] {
                self.peak_holds[i] = avg_db;
            } else {
                self.peak_holds[i] *= self.peak_decay;
            }

            // Calculate bar height
            let normalized = ((avg_db - self.config.min_db)
                / (self.config.max_db - self.config.min_db))
                .clamp(0.0, 1.0);

            let bar_height = (normalized * self.config.height as f64) as usize;

            // Calculate bar position
            let x = i * (bar_width + self.config.bar_spacing);
            let y = self.config.height - bar_height;

            // Draw bar
            image.fill_rect(x, y, bar_width, bar_height, self.config.bar_color);

            // Draw peak marker
            let peak_normalized = ((self.peak_holds[i] - self.config.min_db)
                / (self.config.max_db - self.config.min_db))
                .clamp(0.0, 1.0);

            let peak_y =
                self.config.height - (peak_normalized * self.config.height as f64) as usize;
            if peak_y < self.config.height {
                image.fill_rect(x, peak_y, bar_width, 2, self.config.peak_color);
            }
        }

        image
    }

    /// Generate frequency bins for bars.
    #[allow(clippy::cast_precision_loss)]
    fn generate_frequency_bins(&self) -> Vec<(f64, f64)> {
        let mut bins = Vec::new();

        if self.config.log_scale {
            let log_min = self.config.min_freq.ln();
            let log_max = self.config.max_freq.ln();

            for i in 0..self.config.num_bars {
                let log_start =
                    log_min + (log_max - log_min) * i as f64 / self.config.num_bars as f64;
                let log_end =
                    log_min + (log_max - log_min) * (i + 1) as f64 / self.config.num_bars as f64;

                bins.push((log_start.exp(), log_end.exp()));
            }
        } else {
            let freq_range = self.config.max_freq - self.config.min_freq;

            for i in 0..self.config.num_bars {
                let start =
                    self.config.min_freq + freq_range * i as f64 / self.config.num_bars as f64;
                let end = self.config.min_freq
                    + freq_range * (i + 1) as f64 / self.config.num_bars as f64;

                bins.push((start, end));
            }
        }

        bins
    }

    /// Calculate average magnitude in frequency range.
    fn average_magnitude_in_range(
        &self,
        spectrum: &SpectrumData,
        min_freq: f64,
        max_freq: f64,
    ) -> f64 {
        let magnitudes: Vec<f64> = spectrum
            .magnitude_db
            .iter()
            .zip(&spectrum.frequencies)
            .filter(|(_, &f)| f >= min_freq && f <= max_freq)
            .map(|(m, _)| *m)
            .collect();

        if magnitudes.is_empty() {
            self.config.min_db
        } else {
            magnitudes.iter().sum::<f64>() / magnitudes.len() as f64
        }
    }

    /// Reset peak holds.
    pub fn reset(&mut self) {
        self.peak_holds.fill(-100.0);
    }
}

/// Band visualizer for multi-band display.
pub struct BandVisualizer {
    width: usize,
    height: usize,
    background_color: [u8; 3],
    bar_colors: Vec<[u8; 3]>,
}

impl BandVisualizer {
    /// Create a new band visualizer.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            background_color: [0, 0, 0],
            bar_colors: vec![
                [255, 0, 0],   // Red
                [255, 128, 0], // Orange
                [255, 255, 0], // Yellow
                [0, 255, 0],   // Green
                [0, 255, 255], // Cyan
                [0, 128, 255], // Blue
                [128, 0, 255], // Purple
            ],
        }
    }

    /// Render band energy visualization.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render(
        &self,
        band_energies: &[BandEnergy],
        min_db: f64,
        max_db: f64,
    ) -> SpectrumVisualizerImage {
        let mut image =
            SpectrumVisualizerImage::new(self.width, self.height, self.background_color);

        if band_energies.is_empty() {
            return image;
        }

        let bar_width = self.width / band_energies.len();

        for (i, energy) in band_energies.iter().enumerate() {
            let normalized = ((energy.energy_db - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
            let bar_height = (normalized * self.height as f64) as usize;

            let x = i * bar_width;
            let y = self.height - bar_height;

            let color = self.bar_colors[i % self.bar_colors.len()];

            image.fill_rect(x, y, bar_width - 2, bar_height, color);
        }

        image
    }
}

/// Circular spectrum visualizer (radial).
pub struct CircularSpectrumVisualizer {
    radius: usize,
    num_bars: usize,
    background_color: [u8; 3],
    bar_color: [u8; 3],
}

impl CircularSpectrumVisualizer {
    /// Create a new circular spectrum visualizer.
    #[must_use]
    pub const fn new(radius: usize, num_bars: usize) -> Self {
        Self {
            radius,
            num_bars,
            background_color: [0, 0, 0],
            bar_color: [0, 255, 128],
        }
    }

    /// Render circular spectrum.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render(
        &self,
        spectrum: &SpectrumData,
        min_db: f64,
        max_db: f64,
    ) -> SpectrumVisualizerImage {
        let size = self.radius * 2;
        let mut image = SpectrumVisualizerImage::new(size, size, self.background_color);

        let center_x = self.radius as f64;
        let center_y = self.radius as f64;

        let angle_step = 2.0 * std::f64::consts::PI / self.num_bars as f64;

        for i in 0..self.num_bars {
            let angle = i as f64 * angle_step;

            // Get magnitude for this bar
            let spectrum_idx = (i * spectrum.magnitude_db.len()) / self.num_bars;
            let magnitude_db = if spectrum_idx < spectrum.magnitude_db.len() {
                spectrum.magnitude_db[spectrum_idx]
            } else {
                min_db
            };

            let normalized = ((magnitude_db - min_db) / (max_db - min_db)).clamp(0.0, 1.0);
            let bar_length = normalized * self.radius as f64 * 0.8;

            // Draw line from center outward
            let start_x = center_x + (self.radius as f64 * 0.2) * angle.cos();
            let start_y = center_y + (self.radius as f64 * 0.2) * angle.sin();
            let end_x = center_x + (self.radius as f64 * 0.2 + bar_length) * angle.cos();
            let end_y = center_y + (self.radius as f64 * 0.2 + bar_length) * angle.sin();

            self.draw_line(
                &mut image,
                start_x as usize,
                start_y as usize,
                end_x as usize,
                end_y as usize,
            );
        }

        image
    }

    /// Draw a line (simplified Bresenham).
    fn draw_line(
        &self,
        image: &mut SpectrumVisualizerImage,
        x0: usize,
        y0: usize,
        x1: usize,
        y1: usize,
    ) {
        let dx = (x1 as i32 - x0 as i32).abs();
        let dy = -(y1 as i32 - y0 as i32).abs();
        let sx = if x0 < x1 { 1_i32 } else { -1_i32 };
        let sy = if y0 < y1 { 1_i32 } else { -1_i32 };
        let mut err = dx + dy;

        let mut x = x0 as i32;
        let mut y = y0 as i32;

        loop {
            if x >= 0 && y >= 0 && (x as usize) < image.width && (y as usize) < image.height {
                image.set_pixel(x as usize, y as usize, self.bar_color);
            }

            if x == x1 as i32 && y == y1 as i32 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                if x == x1 as i32 {
                    break;
                }
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                if y == y1 as i32 {
                    break;
                }
                err += dx;
                y += sy;
            }
        }
    }
}
