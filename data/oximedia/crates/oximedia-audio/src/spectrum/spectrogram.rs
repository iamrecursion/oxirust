//! Spectrogram generation and visualization.

use super::analyzer::{SpectrumAnalyzer, SpectrumConfig, SpectrumData};
use super::fft::WindowFunction;
use crate::frame::AudioFrame;

/// Color map for spectrogram visualization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMap {
    /// Grayscale (black to white).
    Grayscale,
    /// Viridis (perceptually uniform).
    Viridis,
    /// Plasma (perceptually uniform).
    Plasma,
    /// Inferno (perceptually uniform).
    Inferno,
    /// Magma (perceptually uniform).
    Magma,
    /// Hot (black-red-yellow-white).
    Hot,
    /// Cool (cyan-blue-magenta).
    Cool,
    /// Jet (rainbow, not recommended for scientific use).
    Jet,
}

impl ColorMap {
    /// Map a normalized value (0.0-1.0) to RGB color.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn map(&self, value: f64) -> [u8; 3] {
        let v = value.clamp(0.0, 1.0);

        match self {
            Self::Grayscale => {
                let gray = (v * 255.0) as u8;
                [gray, gray, gray]
            }
            Self::Viridis => Self::viridis(v),
            Self::Plasma => Self::plasma(v),
            Self::Inferno => Self::inferno(v),
            Self::Magma => Self::magma(v),
            Self::Hot => Self::hot(v),
            Self::Cool => Self::cool(v),
            Self::Jet => Self::jet(v),
        }
    }

    fn viridis(t: f64) -> [u8; 3] {
        let r =
            (0.267004 + t * (0.004874 + t * (2.244 + t * (-2.455 + t * 0.904)))).clamp(0.0, 1.0);
        let g = (0.004874 + t * (0.404 + t * (1.88 + t * (-3.075 + t * 1.197)))).clamp(0.0, 1.0);
        let b =
            (0.329415 + t * (1.384 + t * (-1.634 + t * (0.534 + t * (-0.121))))).clamp(0.0, 1.0);

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    fn plasma(t: f64) -> [u8; 3] {
        let r =
            (0.050383 + t * (2.176 + t * (-2.689 + t * (1.677 + t * (-0.496))))).clamp(0.0, 1.0);
        let g = (0.029803 + t * (0.406 + t * (3.81 + t * (-6.99 + t * 3.743)))).clamp(0.0, 1.0);
        let b = (0.527975 + t * (0.779 + t * (-3.426 + t * (4.699 + t * (-2.38))))).clamp(0.0, 1.0);

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    fn inferno(t: f64) -> [u8; 3] {
        let r = (0.001462 + t * (1.384 + t * (1.782 + t * (-1.723 + t * 0.555)))).clamp(0.0, 1.0);
        let g = (0.000466 + t * (-0.111 + t * (3.869 + t * (-6.498 + t * 3.275)))).clamp(0.0, 1.0);
        let b = (0.013866 + t * (2.295 + t * (-5.577 + t * (5.928 + t * (-2.66))))).clamp(0.0, 1.0);

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    fn magma(t: f64) -> [u8; 3] {
        let r = (0.001462 + t * (1.209 + t * (2.192 + t * (-2.313 + t * 0.911)))).clamp(0.0, 1.0);
        let g = (0.000466 + t * (-0.111 + t * (3.512 + t * (-5.739 + t * 2.874)))).clamp(0.0, 1.0);
        let b = (0.013866 + t * (2.295 + t * (-5.577 + t * (5.928 + t * (-2.66))))).clamp(0.0, 1.0);

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    fn hot(t: f64) -> [u8; 3] {
        let r = (t * 2.5).clamp(0.0, 1.0);
        let g = ((t - 0.4) * 2.5).clamp(0.0, 1.0);
        let b = ((t - 0.8) * 5.0).clamp(0.0, 1.0);

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    fn cool(t: f64) -> [u8; 3] {
        let r = t;
        let g = 1.0 - t;
        let b = 1.0;

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    fn jet(t: f64) -> [u8; 3] {
        let r = ((t - 0.25) * 4.0)
            .clamp(0.0, 1.0)
            .min((1.0 - (t - 0.75) * 4.0).clamp(0.0, 1.0));
        let g = ((t - 0.125) * 4.0)
            .clamp(0.0, 1.0)
            .min((1.0 - (t - 0.625) * 4.0).clamp(0.0, 1.0));
        let b = (t * 4.0)
            .clamp(0.0, 1.0)
            .min((1.0 - (t - 0.5) * 4.0).clamp(0.0, 1.0));

        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }
}

/// Spectrogram configuration.
#[derive(Clone, Debug)]
pub struct SpectrogramConfig {
    /// FFT size.
    pub fft_size: usize,
    /// Window function.
    pub window: WindowFunction,
    /// Hop size (overlap).
    pub hop_size: usize,
    /// Minimum frequency to display (Hz).
    pub min_freq: f64,
    /// Maximum frequency to display (Hz).
    pub max_freq: f64,
    /// Minimum dB level to display.
    pub min_db: f64,
    /// Maximum dB level to display.
    pub max_db: f64,
    /// Color map for visualization.
    pub color_map: ColorMap,
    /// Height in pixels (frequency bins).
    pub height: usize,
}

impl SpectrogramConfig {
    /// Create a new spectrogram configuration.
    #[must_use]
    pub fn new(fft_size: usize, _width: usize, height: usize) -> Self {
        Self {
            fft_size,
            window: WindowFunction::Hann,
            hop_size: fft_size / 4,
            min_freq: 20.0,
            max_freq: 20000.0,
            min_db: -80.0,
            max_db: 0.0,
            color_map: ColorMap::Viridis,
            height,
        }
    }
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self::new(2048, 800, 512)
    }
}

/// Spectrogram image data.
#[derive(Clone, Debug)]
pub struct SpectrogramImage {
    /// Image width (time).
    pub width: usize,
    /// Image height (frequency).
    pub height: usize,
    /// RGB pixel data (row-major order).
    pub data: Vec<u8>,
    /// Time stamps for each column (seconds).
    pub time_stamps: Vec<f64>,
    /// Frequency bins for each row (Hz).
    pub frequency_bins: Vec<f64>,
}

impl SpectrogramImage {
    /// Create a new empty spectrogram image.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width * height * 3],
            time_stamps: Vec::new(),
            frequency_bins: Vec::new(),
        }
    }

    /// Get pixel at (x, y) as RGB.
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = (y * self.width + x) * 3;
        Some([self.data[idx], self.data[idx + 1], self.data[idx + 2]])
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

    /// Save as PPM format (simple image format).
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

/// Spectrogram generator.
pub struct SpectrogramGenerator {
    config: SpectrogramConfig,
    analyzer: SpectrumAnalyzer,
    spectra: Vec<SpectrumData>,
}

impl SpectrogramGenerator {
    /// Create a new spectrogram generator.
    pub fn new(config: SpectrogramConfig) -> Result<Self, String> {
        let spectrum_config = SpectrumConfig {
            fft_size: config.fft_size,
            window: config.window,
            hop_size: config.hop_size,
            min_freq: config.min_freq,
            max_freq: config.max_freq,
            smoothing: 0.0,
            use_mel_scale: false,
            mel_bands: 0,
        };

        let analyzer = SpectrumAnalyzer::new(spectrum_config)?;

        Ok(Self {
            config,
            analyzer,
            spectra: Vec::new(),
        })
    }

    /// Process audio frame and accumulate spectrum data.
    pub fn process_frame(&mut self, frame: &AudioFrame) -> Result<(), String> {
        let spectra = self.analyzer.analyze_streaming(frame)?;
        self.spectra.extend(spectra);
        Ok(())
    }

    /// Process multiple audio frames.
    pub fn process_frames(&mut self, frames: &[AudioFrame]) -> Result<(), String> {
        for frame in frames {
            self.process_frame(frame)?;
        }
        Ok(())
    }

    /// Generate spectrogram image from accumulated data.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn generate(&self) -> SpectrogramImage {
        let width = self.spectra.len();
        let height = self.config.height;

        let mut image = SpectrogramImage::new(width, height);

        if self.spectra.is_empty() {
            return image;
        }

        // Generate frequency bins for the image
        let sample_rate = self.spectra[0].sample_rate;
        let nyquist = sample_rate / 2.0;

        let min_freq = self.config.min_freq;
        let max_freq = self.config.max_freq.min(nyquist);

        image.frequency_bins = (0..height)
            .map(|i| {
                // Linear frequency mapping (could be log-scale)
                min_freq + (max_freq - min_freq) * (height - 1 - i) as f64 / (height - 1) as f64
            })
            .collect();

        // Generate time stamps
        image.time_stamps = (0..width)
            .map(|i| i as f64 * self.config.hop_size as f64 / sample_rate)
            .collect();

        // Render spectrogram
        for (x, spectrum) in self.spectra.iter().enumerate() {
            for y in 0..height {
                let freq = image.frequency_bins[y];

                // Find closest frequency bin in spectrum
                let magnitude = self.interpolate_magnitude(spectrum, freq);

                // Convert to dB and normalize
                let db = if magnitude > 0.0 {
                    20.0 * magnitude.log10()
                } else {
                    self.config.min_db
                };

                let normalized = ((db - self.config.min_db)
                    / (self.config.max_db - self.config.min_db))
                    .clamp(0.0, 1.0);

                let color = self.config.color_map.map(normalized);
                image.set_pixel(x, y, color);
            }
        }

        image
    }

    /// Interpolate magnitude at a specific frequency.
    fn interpolate_magnitude(&self, spectrum: &SpectrumData, frequency: f64) -> f64 {
        if spectrum.frequencies.is_empty() {
            return 0.0;
        }

        // Find surrounding frequency bins
        let mut lower_idx = 0;
        for (i, &freq) in spectrum.frequencies.iter().enumerate() {
            if freq <= frequency {
                lower_idx = i;
            } else {
                break;
            }
        }

        let upper_idx = (lower_idx + 1).min(spectrum.frequencies.len() - 1);

        if lower_idx == upper_idx {
            return spectrum.magnitude[lower_idx];
        }

        // Linear interpolation
        let f1 = spectrum.frequencies[lower_idx];
        let f2 = spectrum.frequencies[upper_idx];
        let m1 = spectrum.magnitude[lower_idx];
        let m2 = spectrum.magnitude[upper_idx];

        let t = (frequency - f1) / (f2 - f1);
        m1 + t * (m2 - m1)
    }

    /// Clear accumulated spectra.
    pub fn clear(&mut self) {
        self.spectra.clear();
        self.analyzer.reset();
    }

    /// Get number of accumulated spectra.
    #[must_use]
    pub fn spectrum_count(&self) -> usize {
        self.spectra.len()
    }
}

/// Real-time spectrogram for streaming visualization.
pub struct RealtimeSpectrogram {
    config: SpectrogramConfig,
    analyzer: SpectrumAnalyzer,
    image: SpectrogramImage,
    current_column: usize,
}

impl RealtimeSpectrogram {
    /// Create a new real-time spectrogram.
    pub fn new(config: SpectrogramConfig, width: usize) -> Result<Self, String> {
        let spectrum_config = SpectrumConfig {
            fft_size: config.fft_size,
            window: config.window,
            hop_size: config.hop_size,
            min_freq: config.min_freq,
            max_freq: config.max_freq,
            smoothing: 0.0,
            use_mel_scale: false,
            mel_bands: 0,
        };

        let analyzer = SpectrumAnalyzer::new(spectrum_config)?;
        let image = SpectrogramImage::new(width, config.height);

        Ok(Self {
            config,
            analyzer,
            image,
            current_column: 0,
        })
    }

    /// Process audio frame and update spectrogram.
    #[allow(clippy::cast_precision_loss)]
    pub fn update(&mut self, frame: &AudioFrame) -> Result<(), String> {
        let spectra = self.analyzer.analyze_streaming(frame)?;

        for spectrum in spectra {
            if self.current_column >= self.image.width {
                // Shift image left
                self.shift_left();
                self.current_column = self.image.width - 1;
            }

            // Render new column
            for y in 0..self.config.height {
                let freq = if y < self.image.frequency_bins.len() {
                    self.image.frequency_bins[y]
                } else {
                    let sample_rate = spectrum.sample_rate;
                    let nyquist = sample_rate / 2.0;
                    let min_freq = self.config.min_freq;
                    let max_freq = self.config.max_freq.min(nyquist);
                    min_freq
                        + (max_freq - min_freq) * (self.config.height - 1 - y) as f64
                            / (self.config.height - 1) as f64
                };

                let magnitude = self.interpolate_magnitude(&spectrum, freq);
                let db = if magnitude > 0.0 {
                    20.0 * magnitude.log10()
                } else {
                    self.config.min_db
                };

                let normalized = ((db - self.config.min_db)
                    / (self.config.max_db - self.config.min_db))
                    .clamp(0.0, 1.0);

                let color = self.config.color_map.map(normalized);
                self.image.set_pixel(self.current_column, y, color);
            }

            self.current_column += 1;
        }

        Ok(())
    }

    /// Shift image one column to the left.
    fn shift_left(&mut self) {
        let height = self.image.height;
        let width = self.image.width;

        for y in 0..height {
            for x in 0..width - 1 {
                let src_idx = (y * width + x + 1) * 3;
                let dst_idx = (y * width + x) * 3;
                self.image.data[dst_idx] = self.image.data[src_idx];
                self.image.data[dst_idx + 1] = self.image.data[src_idx + 1];
                self.image.data[dst_idx + 2] = self.image.data[src_idx + 2];
            }
        }
    }

    /// Interpolate magnitude at a specific frequency.
    fn interpolate_magnitude(&self, spectrum: &SpectrumData, frequency: f64) -> f64 {
        if spectrum.frequencies.is_empty() {
            return 0.0;
        }

        let mut lower_idx = 0;
        for (i, &freq) in spectrum.frequencies.iter().enumerate() {
            if freq <= frequency {
                lower_idx = i;
            } else {
                break;
            }
        }

        let upper_idx = (lower_idx + 1).min(spectrum.frequencies.len() - 1);

        if lower_idx == upper_idx {
            return spectrum.magnitude[lower_idx];
        }

        let f1 = spectrum.frequencies[lower_idx];
        let f2 = spectrum.frequencies[upper_idx];
        let m1 = spectrum.magnitude[lower_idx];
        let m2 = spectrum.magnitude[upper_idx];

        let t = (frequency - f1) / (f2 - f1);
        m1 + t * (m2 - m1)
    }

    /// Get current spectrogram image.
    #[must_use]
    pub const fn image(&self) -> &SpectrogramImage {
        &self.image
    }

    /// Reset the spectrogram.
    pub fn reset(&mut self) {
        self.image.data.fill(0);
        self.current_column = 0;
        self.analyzer.reset();
    }
}
