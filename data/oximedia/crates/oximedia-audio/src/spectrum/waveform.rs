//! Waveform display and rendering.

use crate::frame::AudioFrame;
use crate::AudioBuffer;
use oximedia_core::SampleFormat;

/// Waveform rendering mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveformMode {
    /// Line waveform (connect samples).
    Line,
    /// Filled waveform (fill area under curve).
    Filled,
    /// Min/Max bars (show min and max in each column).
    MinMax,
    /// RMS bars (show RMS level in each column).
    Rms,
}

/// Waveform configuration.
#[derive(Clone, Debug)]
pub struct WaveformConfig {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Rendering mode.
    pub mode: WaveformMode,
    /// Background color (RGB).
    pub background_color: [u8; 3],
    /// Waveform color (RGB).
    pub waveform_color: [u8; 3],
    /// Grid color (RGB), None for no grid.
    pub grid_color: Option<[u8; 3]>,
    /// Zero-line color (RGB), None for no zero-line.
    pub zero_line_color: Option<[u8; 3]>,
    /// Number of grid divisions.
    pub grid_divisions: usize,
}

impl WaveformConfig {
    /// Create a new waveform configuration.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            mode: WaveformMode::MinMax,
            background_color: [0, 0, 0],
            waveform_color: [0, 255, 0],
            grid_color: Some([64, 64, 64]),
            zero_line_color: Some([128, 128, 128]),
            grid_divisions: 4,
        }
    }
}

impl Default for WaveformConfig {
    fn default() -> Self {
        Self::new(800, 200)
    }
}

/// Waveform image.
#[derive(Clone, Debug)]
pub struct WaveformImage {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// RGB pixel data (row-major order).
    pub data: Vec<u8>,
}

impl WaveformImage {
    /// Create a new waveform image with background color.
    #[must_use]
    pub fn new(width: usize, height: usize, background_color: [u8; 3]) -> Self {
        let mut data = vec![0; width * height * 3];

        // Fill with background color
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

    /// Set pixel at (x, y) to RGB color.
    pub fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) * 3;
        self.data[idx] = color[0];
        self.data[idx + 1] = color[1];
        self.data[idx + 2] = color[2];
    }

    /// Get pixel at (x, y).
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = (y * self.width + x) * 3;
        Some([self.data[idx], self.data[idx + 1], self.data[idx + 2]])
    }

    /// Draw a horizontal line.
    pub fn draw_horizontal_line(&mut self, y: usize, color: [u8; 3]) {
        for x in 0..self.width {
            self.set_pixel(x, y, color);
        }
    }

    /// Draw a vertical line.
    pub fn draw_vertical_line(&mut self, x: usize, color: [u8; 3]) {
        for y in 0..self.height {
            self.set_pixel(x, y, color);
        }
    }

    /// Draw a line between two points using Bresenham's algorithm.
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_sign_loss)]
    pub fn draw_line(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, color: [u8; 3]) {
        let dx = (x1 as i32 - x0 as i32).abs();
        let dy = -(y1 as i32 - y0 as i32).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x0 as i32;
        let mut y = y0 as i32;

        loop {
            if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
                self.set_pixel(x as usize, y as usize, color);
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

    /// Fill a vertical line (for filled waveform mode).
    pub fn fill_vertical(&mut self, x: usize, y_start: usize, y_end: usize, color: [u8; 3]) {
        let start = y_start.min(y_end);
        let end = y_start.max(y_end);

        for y in start..=end {
            if y < self.height {
                self.set_pixel(x, y, color);
            }
        }
    }

    /// Save as PPM format.
    pub fn save_ppm(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;
        writeln!(file, "P6")?;
        writeln!(file, "{} {}", self.width, self.height)?;
        writeln!(file, "255")?;
        file.write_all(&self.data)?;

        Ok(())
    }
}

/// Waveform renderer.
pub struct WaveformRenderer {
    config: WaveformConfig,
}

impl WaveformRenderer {
    /// Create a new waveform renderer.
    #[must_use]
    pub const fn new(config: WaveformConfig) -> Self {
        Self { config }
    }

    /// Render a waveform from audio frame.
    pub fn render(&self, frame: &AudioFrame) -> Result<WaveformImage, String> {
        let samples = self.extract_samples(frame)?;
        self.render_samples(&samples)
    }

    /// Render a waveform from raw samples.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render_samples(&self, samples: &[f64]) -> Result<WaveformImage, String> {
        let mut image = WaveformImage::new(
            self.config.width,
            self.config.height,
            self.config.background_color,
        );

        if samples.is_empty() {
            return Ok(image);
        }

        // Draw grid
        if let Some(grid_color) = self.config.grid_color {
            self.draw_grid(&mut image, grid_color);
        }

        // Draw zero line
        if let Some(zero_color) = self.config.zero_line_color {
            let center_y = self.config.height / 2;
            image.draw_horizontal_line(center_y, zero_color);
        }

        // Calculate samples per pixel
        let samples_per_pixel = samples.len() as f64 / self.config.width as f64;

        match self.config.mode {
            WaveformMode::Line => {
                self.render_line(&mut image, samples, samples_per_pixel)?;
            }
            WaveformMode::Filled => {
                self.render_filled(&mut image, samples, samples_per_pixel)?;
            }
            WaveformMode::MinMax => {
                self.render_minmax(&mut image, samples, samples_per_pixel)?;
            }
            WaveformMode::Rms => {
                self.render_rms(&mut image, samples, samples_per_pixel)?;
            }
        }

        Ok(image)
    }

    /// Draw grid lines.
    fn draw_grid(&self, image: &mut WaveformImage, color: [u8; 3]) {
        let h_step = self.config.height / self.config.grid_divisions;
        let v_step = self.config.width / self.config.grid_divisions;

        // Horizontal lines
        for i in 0..=self.config.grid_divisions {
            let y = i * h_step;
            if y < self.config.height {
                image.draw_horizontal_line(y, color);
            }
        }

        // Vertical lines
        for i in 0..=self.config.grid_divisions {
            let x = i * v_step;
            if x < self.config.width {
                image.draw_vertical_line(x, color);
            }
        }
    }

    /// Render line waveform.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn render_line(
        &self,
        image: &mut WaveformImage,
        samples: &[f64],
        samples_per_pixel: f64,
    ) -> Result<(), String> {
        let center_y = self.config.height / 2;

        let mut prev_y = center_y;

        for x in 0..self.config.width {
            let sample_idx = (x as f64 * samples_per_pixel) as usize;
            if sample_idx >= samples.len() {
                break;
            }

            let sample = samples[sample_idx].clamp(-1.0, 1.0);
            let y = center_y as f64 - sample * (self.config.height as f64 / 2.0);
            let y = y.clamp(0.0, (self.config.height - 1) as f64) as usize;

            if x > 0 {
                image.draw_line(x - 1, prev_y, x, y, self.config.waveform_color);
            }

            prev_y = y;
        }

        Ok(())
    }

    /// Render filled waveform.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn render_filled(
        &self,
        image: &mut WaveformImage,
        samples: &[f64],
        samples_per_pixel: f64,
    ) -> Result<(), String> {
        let center_y = self.config.height / 2;

        for x in 0..self.config.width {
            let sample_idx = (x as f64 * samples_per_pixel) as usize;
            if sample_idx >= samples.len() {
                break;
            }

            let sample = samples[sample_idx].clamp(-1.0, 1.0);
            let y = center_y as f64 - sample * (self.config.height as f64 / 2.0);
            let y = y.clamp(0.0, (self.config.height - 1) as f64) as usize;

            image.fill_vertical(x, center_y, y, self.config.waveform_color);
        }

        Ok(())
    }

    /// Render min/max waveform.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn render_minmax(
        &self,
        image: &mut WaveformImage,
        samples: &[f64],
        samples_per_pixel: f64,
    ) -> Result<(), String> {
        let center_y = self.config.height / 2;

        for x in 0..self.config.width {
            let start_idx = (x as f64 * samples_per_pixel) as usize;
            let end_idx = ((x + 1) as f64 * samples_per_pixel) as usize;

            if start_idx >= samples.len() {
                break;
            }

            let end_idx = end_idx.min(samples.len());

            // Find min and max in this range
            let mut min_sample = samples[start_idx];
            let mut max_sample = samples[start_idx];

            for &sample in &samples[start_idx..end_idx] {
                min_sample = min_sample.min(sample);
                max_sample = max_sample.max(sample);
            }

            min_sample = min_sample.clamp(-1.0, 1.0);
            max_sample = max_sample.clamp(-1.0, 1.0);

            let min_y = center_y as f64 - min_sample * (self.config.height as f64 / 2.0);
            let max_y = center_y as f64 - max_sample * (self.config.height as f64 / 2.0);

            let min_y = min_y.clamp(0.0, (self.config.height - 1) as f64) as usize;
            let max_y = max_y.clamp(0.0, (self.config.height - 1) as f64) as usize;

            image.fill_vertical(x, min_y, max_y, self.config.waveform_color);
        }

        Ok(())
    }

    /// Render RMS waveform.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    fn render_rms(
        &self,
        image: &mut WaveformImage,
        samples: &[f64],
        samples_per_pixel: f64,
    ) -> Result<(), String> {
        let center_y = self.config.height / 2;

        for x in 0..self.config.width {
            let start_idx = (x as f64 * samples_per_pixel) as usize;
            let end_idx = ((x + 1) as f64 * samples_per_pixel) as usize;

            if start_idx >= samples.len() {
                break;
            }

            let end_idx = end_idx.min(samples.len());

            // Calculate RMS for this range
            let sum_squares: f64 = samples[start_idx..end_idx].iter().map(|&s| s * s).sum();

            let rms = (sum_squares / (end_idx - start_idx) as f64).sqrt();
            let rms = rms.clamp(0.0, 1.0);

            let pos_y = center_y as f64 - rms * (self.config.height as f64 / 2.0);
            let neg_y = center_y as f64 + rms * (self.config.height as f64 / 2.0);

            let pos_y = pos_y.clamp(0.0, (self.config.height - 1) as f64) as usize;
            let neg_y = neg_y.clamp(0.0, (self.config.height - 1) as f64) as usize;

            image.fill_vertical(x, pos_y, neg_y, self.config.waveform_color);
        }

        Ok(())
    }

    /// Extract samples from audio frame.
    fn extract_samples(&self, frame: &AudioFrame) -> Result<Vec<f64>, String> {
        let channel_count = frame.channels.count();
        if channel_count == 0 {
            return Err("No channels in audio frame".to_string());
        }

        match &frame.samples {
            AudioBuffer::Interleaved(data) => {
                self.extract_interleaved_samples(data, frame.format, channel_count)
            }
            AudioBuffer::Planar(planes) => {
                if planes.is_empty() {
                    return Err("Empty planar buffer".to_string());
                }
                self.extract_planar_samples(&planes[0], frame.format)
            }
        }
    }

    /// Extract samples from interleaved buffer.
    #[allow(clippy::cast_precision_loss)]
    fn extract_interleaved_samples(
        &self,
        data: &[u8],
        format: SampleFormat,
        channel_count: usize,
    ) -> Result<Vec<f64>, String> {
        let bytes_per_sample = format.bytes_per_sample();
        let sample_count = data.len() / (bytes_per_sample * channel_count);

        let mut samples = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let offset = i * channel_count * bytes_per_sample;
            let sample = self.bytes_to_f64(&data[offset..offset + bytes_per_sample], format)?;
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Extract samples from planar buffer.
    fn extract_planar_samples(
        &self,
        data: &[u8],
        format: SampleFormat,
    ) -> Result<Vec<f64>, String> {
        let bytes_per_sample = format.bytes_per_sample();
        let sample_count = data.len() / bytes_per_sample;

        let mut samples = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let offset = i * bytes_per_sample;
            let sample = self.bytes_to_f64(&data[offset..offset + bytes_per_sample], format)?;
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Convert bytes to f64 based on sample format.
    #[allow(clippy::cast_precision_loss)]
    fn bytes_to_f64(&self, bytes: &[u8], format: SampleFormat) -> Result<f64, String> {
        match format {
            SampleFormat::U8 => Ok(f64::from(bytes[0]) / 128.0 - 1.0),
            SampleFormat::S16 | SampleFormat::S16p => {
                if bytes.len() < 2 {
                    return Err("Insufficient bytes for S16".to_string());
                }
                let sample = i16::from_le_bytes([bytes[0], bytes[1]]);
                Ok(f64::from(sample) / 32768.0)
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                if bytes.len() < 4 {
                    return Err("Insufficient bytes for S32".to_string());
                }
                let sample = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                Ok(sample as f64 / 2_147_483_648.0)
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                if bytes.len() < 4 {
                    return Err("Insufficient bytes for F32".to_string());
                }
                let sample = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                Ok(f64::from(sample))
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                if bytes.len() < 8 {
                    return Err("Insufficient bytes for F64".to_string());
                }
                Ok(f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]))
            }
            _ => Err("Unsupported sample format".to_string()),
        }
    }
}

/// Phase scope for stereo correlation visualization.
pub struct PhaseScope {
    width: usize,
    height: usize,
    background_color: [u8; 3],
    trace_color: [u8; 3],
    grid_color: Option<[u8; 3]>,
}

impl PhaseScope {
    /// Create a new phase scope.
    #[must_use]
    pub const fn new(
        width: usize,
        height: usize,
        background_color: [u8; 3],
        trace_color: [u8; 3],
    ) -> Self {
        Self {
            width,
            height,
            background_color,
            trace_color,
            grid_color: Some([64, 64, 64]),
        }
    }

    /// Render phase scope from stereo samples.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn render(&self, left: &[f64], right: &[f64]) -> WaveformImage {
        let mut image = WaveformImage::new(self.width, self.height, self.background_color);

        // Draw grid
        if let Some(grid_color) = self.grid_color {
            let center_x = self.width / 2;
            let center_y = self.height / 2;

            image.draw_horizontal_line(center_y, grid_color);
            image.draw_vertical_line(center_x, grid_color);

            // Draw diagonals for correlation reference
            for i in 0..self.width.min(self.height) {
                image.set_pixel(i, i, grid_color);
                if i < self.width && (self.height - 1 - i) < self.height {
                    image.set_pixel(i, self.height - 1 - i, grid_color);
                }
            }
        }

        let center_x = self.width / 2;
        let center_y = self.height / 2;

        let sample_count = left.len().min(right.len());

        for i in 0..sample_count {
            let l = left[i].clamp(-1.0, 1.0);
            let r = right[i].clamp(-1.0, 1.0);

            let x = center_x as f64 + l * (self.width as f64 / 2.0);
            let y = center_y as f64 - r * (self.height as f64 / 2.0);

            let x = x.clamp(0.0, (self.width - 1) as f64) as usize;
            let y = y.clamp(0.0, (self.height - 1) as f64) as usize;

            image.set_pixel(x, y, self.trace_color);
        }

        image
    }
}
