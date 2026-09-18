//! Spectrum analyzer implementation.

use super::fft::{FftProcessor, MelScale, OverlapAdd, WindowFunction};
use crate::frame::AudioFrame;
use crate::AudioBuffer;
use oximedia_core::SampleFormat;

/// Spectrum analyzer configuration.
#[derive(Clone, Debug)]
pub struct SpectrumConfig {
    /// FFT size (must be power of 2).
    pub fft_size: usize,
    /// Window function to use.
    pub window: WindowFunction,
    /// Hop size for overlapping analysis (0 = no overlap, fft_size/2 = 50% overlap).
    pub hop_size: usize,
    /// Minimum frequency for analysis (Hz).
    pub min_freq: f64,
    /// Maximum frequency for analysis (Hz).
    pub max_freq: f64,
    /// Smoothing factor for peak detection (0.0 = no smoothing, 1.0 = max smoothing).
    pub smoothing: f64,
    /// Enable mel-scale frequency mapping.
    pub use_mel_scale: bool,
    /// Number of mel bands (only used if use_mel_scale is true).
    pub mel_bands: usize,
}

impl SpectrumConfig {
    /// Create a new spectrum configuration with default values.
    #[must_use]
    pub fn new(fft_size: usize) -> Self {
        Self {
            fft_size,
            window: WindowFunction::Hann,
            hop_size: fft_size / 2,
            min_freq: 20.0,
            max_freq: 20000.0,
            smoothing: 0.7,
            use_mel_scale: false,
            mel_bands: 40,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if !self.fft_size.is_power_of_two() {
            return Err("FFT size must be a power of 2".to_string());
        }
        if self.fft_size < 64 || self.fft_size > 32768 {
            return Err("FFT size must be between 64 and 32768".to_string());
        }
        if self.hop_size == 0 || self.hop_size > self.fft_size {
            return Err("Invalid hop size".to_string());
        }
        if self.min_freq < 0.0 || self.min_freq >= self.max_freq {
            return Err("Invalid frequency range".to_string());
        }
        if !(0.0..=1.0).contains(&self.smoothing) {
            return Err("Smoothing must be between 0.0 and 1.0".to_string());
        }
        if self.use_mel_scale && self.mel_bands == 0 {
            return Err("Mel bands must be greater than 0".to_string());
        }
        Ok(())
    }
}

impl Default for SpectrumConfig {
    fn default() -> Self {
        Self::new(2048)
    }
}

/// Spectrum data output.
#[derive(Clone, Debug)]
pub struct SpectrumData {
    /// Magnitude spectrum (linear scale).
    pub magnitude: Vec<f64>,
    /// Power spectrum (magnitude squared).
    pub power: Vec<f64>,
    /// Magnitude in decibels.
    pub magnitude_db: Vec<f64>,
    /// Frequency bins (Hz).
    pub frequencies: Vec<f64>,
    /// Peak frequencies detected.
    pub peaks: Vec<Peak>,
    /// Sample rate of the analyzed audio.
    pub sample_rate: f64,
}

impl SpectrumData {
    /// Get the bin with maximum magnitude.
    #[must_use]
    pub fn max_bin(&self) -> Option<usize> {
        self.magnitude
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// Get the frequency with maximum magnitude.
    #[must_use]
    pub fn max_frequency(&self) -> Option<f64> {
        self.max_bin().map(|idx| self.frequencies[idx])
    }

    /// Get magnitude at a specific frequency.
    #[must_use]
    pub fn magnitude_at_frequency(&self, frequency: f64) -> Option<f64> {
        self.frequencies
            .iter()
            .position(|&f| (f - frequency).abs() < 0.1)
            .map(|idx| self.magnitude[idx])
    }

    /// Get average magnitude in a frequency range.
    #[must_use]
    pub fn average_magnitude_in_range(&self, min_freq: f64, max_freq: f64) -> f64 {
        let sum: f64 = self
            .magnitude
            .iter()
            .zip(&self.frequencies)
            .filter(|(_, &f)| f >= min_freq && f <= max_freq)
            .map(|(m, _)| m)
            .sum();

        let count = self
            .frequencies
            .iter()
            .filter(|&&f| f >= min_freq && f <= max_freq)
            .count();

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}

/// Detected spectral peak.
#[derive(Clone, Debug)]
pub struct Peak {
    /// Frequency of the peak (Hz).
    pub frequency: f64,
    /// Magnitude of the peak.
    pub magnitude: f64,
    /// Bin index.
    pub bin: usize,
}

/// Spectrum analyzer.
pub struct SpectrumAnalyzer {
    config: SpectrumConfig,
    fft_processor: FftProcessor,
    overlap_add: OverlapAdd,
    smoothed_spectrum: Vec<f64>,
    mel_filterbank: Option<Vec<Vec<f64>>>,
}

impl SpectrumAnalyzer {
    /// Create a new spectrum analyzer.
    pub fn new(config: SpectrumConfig) -> Result<Self, String> {
        config.validate()?;

        let fft_processor = FftProcessor::new(config.fft_size, config.window);
        let overlap_add = OverlapAdd::new(config.fft_size, config.hop_size);

        let mel_filterbank = if config.use_mel_scale {
            Some(MelScale::create_filterbank(
                config.mel_bands,
                config.fft_size,
                44100.0, // Default sample rate, will be updated during analysis
                config.min_freq,
                config.max_freq,
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            fft_processor,
            overlap_add,
            smoothed_spectrum: Vec::new(),
            mel_filterbank,
        })
    }

    /// Analyze an audio frame and return spectrum data.
    pub fn analyze(&mut self, frame: &AudioFrame) -> Result<SpectrumData, String> {
        // Extract samples from frame
        let samples = self.extract_samples(frame)?;

        // Update mel filterbank if needed
        if self.config.use_mel_scale && self.mel_filterbank.is_some() {
            self.mel_filterbank = Some(MelScale::create_filterbank(
                self.config.mel_bands,
                self.config.fft_size,
                f64::from(frame.sample_rate),
                self.config.min_freq,
                self.config.max_freq,
            ));
        }

        // Compute spectrum
        let magnitude = self.fft_processor.magnitude_spectrum(&samples);
        let power = self.fft_processor.power_spectrum(&samples);

        // Apply mel filterbank if enabled
        let (magnitude, power) = if let Some(ref filterbank) = self.mel_filterbank {
            let mel_mag = MelScale::apply_filterbank(&magnitude[..magnitude.len() / 2], filterbank);
            let mel_pow = MelScale::apply_filterbank(&power[..power.len() / 2], filterbank);
            (mel_mag, mel_pow)
        } else {
            // Use only first half (up to Nyquist)
            (
                magnitude[..magnitude.len() / 2].to_vec(),
                power[..power.len() / 2].to_vec(),
            )
        };

        // Apply smoothing
        let smoothed_magnitude = self.apply_smoothing(&magnitude);

        // Convert to dB
        let magnitude_db: Vec<f64> = smoothed_magnitude
            .iter()
            .map(|&m| FftProcessor::to_db(m, 1.0))
            .collect();

        // Generate frequency bins
        let frequencies = self.generate_frequencies(frame.sample_rate, smoothed_magnitude.len());

        // Detect peaks
        let peaks = self.detect_peaks(&smoothed_magnitude, &frequencies);

        Ok(SpectrumData {
            magnitude: smoothed_magnitude,
            power,
            magnitude_db,
            frequencies,
            peaks,
            sample_rate: f64::from(frame.sample_rate),
        })
    }

    /// Analyze streaming audio (returns multiple spectrum frames).
    pub fn analyze_streaming(&mut self, frame: &AudioFrame) -> Result<Vec<SpectrumData>, String> {
        let samples = self.extract_samples(frame)?;

        let frames = self.overlap_add.push(&samples);
        let mut results = Vec::new();

        for frame_samples in frames {
            let magnitude = self.fft_processor.magnitude_spectrum(&frame_samples);
            let power = self.fft_processor.power_spectrum(&frame_samples);

            let (magnitude, power) = if let Some(ref filterbank) = self.mel_filterbank {
                let mel_mag =
                    MelScale::apply_filterbank(&magnitude[..magnitude.len() / 2], filterbank);
                let mel_pow = MelScale::apply_filterbank(&power[..power.len() / 2], filterbank);
                (mel_mag, mel_pow)
            } else {
                (
                    magnitude[..magnitude.len() / 2].to_vec(),
                    power[..power.len() / 2].to_vec(),
                )
            };

            let smoothed_magnitude = self.apply_smoothing(&magnitude);
            let magnitude_db: Vec<f64> = smoothed_magnitude
                .iter()
                .map(|&m| FftProcessor::to_db(m, 1.0))
                .collect();

            let frequencies =
                self.generate_frequencies(frame.sample_rate, smoothed_magnitude.len());
            let peaks = self.detect_peaks(&smoothed_magnitude, &frequencies);

            results.push(SpectrumData {
                magnitude: smoothed_magnitude,
                power,
                magnitude_db,
                frequencies,
                peaks,
                sample_rate: f64::from(frame.sample_rate),
            });
        }

        Ok(results)
    }

    /// Extract samples from audio frame as f64.
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

    /// Apply temporal smoothing to spectrum.
    fn apply_smoothing(&mut self, spectrum: &[f64]) -> Vec<f64> {
        if self.smoothed_spectrum.is_empty() {
            self.smoothed_spectrum = spectrum.to_vec();
            return spectrum.to_vec();
        }

        if self.smoothed_spectrum.len() != spectrum.len() {
            self.smoothed_spectrum = spectrum.to_vec();
            return spectrum.to_vec();
        }

        let alpha = 1.0 - self.config.smoothing;

        self.smoothed_spectrum
            .iter_mut()
            .zip(spectrum.iter())
            .for_each(|(smoothed, &current)| {
                *smoothed = alpha * current + (1.0 - alpha) * *smoothed;
            });

        self.smoothed_spectrum.clone()
    }

    /// Generate frequency bins.
    #[allow(clippy::cast_precision_loss)]
    fn generate_frequencies(&self, sample_rate: u32, num_bins: usize) -> Vec<f64> {
        if self.config.use_mel_scale {
            // Generate mel-spaced frequencies
            let min_mel = MelScale::hz_to_mel(self.config.min_freq);
            let max_mel = MelScale::hz_to_mel(self.config.max_freq);

            (0..num_bins)
                .map(|i| {
                    let mel = min_mel + (max_mel - min_mel) * i as f64 / (num_bins - 1) as f64;
                    MelScale::mel_to_hz(mel)
                })
                .collect()
        } else {
            // Generate linear frequencies
            let bin_width = sample_rate as f64 / self.config.fft_size as f64;
            (0..num_bins).map(|i| i as f64 * bin_width).collect()
        }
    }

    /// Detect peaks in spectrum.
    fn detect_peaks(&self, spectrum: &[f64], frequencies: &[f64]) -> Vec<Peak> {
        let mut peaks = Vec::new();

        if spectrum.len() < 3 {
            return peaks;
        }

        // Find local maxima
        for i in 1..spectrum.len() - 1 {
            if spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1] {
                // Check if peak is significant
                let avg = (spectrum[i - 1] + spectrum[i + 1]) / 2.0;
                if spectrum[i] > avg * 1.5 {
                    peaks.push(Peak {
                        frequency: frequencies[i],
                        magnitude: spectrum[i],
                        bin: i,
                    });
                }
            }
        }

        // Sort by magnitude (descending)
        peaks.sort_by(|a, b| {
            b.magnitude
                .partial_cmp(&a.magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        peaks
    }

    /// Reset the analyzer state.
    pub fn reset(&mut self) {
        self.smoothed_spectrum.clear();
        self.overlap_add.clear();
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &SpectrumConfig {
        &self.config
    }
}
