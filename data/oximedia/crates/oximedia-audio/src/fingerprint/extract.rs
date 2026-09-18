//! Spectral peak extraction for fingerprinting.

use crate::error::{AudioError, AudioResult};
use crate::spectrum::fft::{FftProcessor, WindowFunction};
use crate::AudioFrame;
use crate::{AudioBuffer, ChannelLayout};
use oximedia_core::SampleFormat;

use super::constellation::{ConstellationMap, Peak};
use super::FingerprintConfig;

/// Spectrogram extractor for fingerprinting.
pub struct SpectrogramExtractor {
    fft_size: usize,
    hop_size: usize,
    freq_range: (f64, f64),
    num_bands: usize,
}

impl SpectrogramExtractor {
    /// Create a new spectrogram extractor.
    #[must_use]
    pub const fn new(
        fft_size: usize,
        hop_size: usize,
        freq_range: (f64, f64),
        num_bands: usize,
    ) -> Self {
        Self {
            fft_size,
            hop_size,
            freq_range,
            num_bands,
        }
    }

    /// Extract constellation map from audio frame.
    pub fn extract(
        &self,
        audio: &AudioFrame,
        config: &FingerprintConfig,
    ) -> AudioResult<ConstellationMap> {
        // Convert to mono and f64 samples
        let samples = self.extract_samples(audio)?;

        // Process with FFT
        let mut fft = FftProcessor::new(self.fft_size, WindowFunction::Hann);

        // Extract time-frequency representation
        let spectrogram = self.compute_spectrogram(&samples, &mut fft, audio.sample_rate);

        // Find spectral peaks
        let peaks = self.find_peaks(&spectrogram, audio.sample_rate, config);

        Ok(ConstellationMap::new(
            peaks,
            audio.sample_rate,
            audio.duration_seconds(),
        ))
    }

    /// Extract mono samples as f64 from audio frame.
    fn extract_samples(&self, audio: &AudioFrame) -> AudioResult<Vec<f64>> {
        match &audio.samples {
            AudioBuffer::Interleaved(data) => {
                self.convert_interleaved_to_mono(data, &audio.format, &audio.channels)
            }
            AudioBuffer::Planar(planes) => self.convert_planar_to_mono(planes, &audio.format),
        }
    }

    /// Convert interleaved samples to mono f64.
    fn convert_interleaved_to_mono(
        &self,
        data: &[u8],
        format: &SampleFormat,
        channels: &ChannelLayout,
    ) -> AudioResult<Vec<f64>> {
        let channel_count = channels.count();
        if channel_count == 0 {
            return Err(AudioError::InvalidParameter("No channels".to_string()));
        }

        let bytes_per_sample = format.bytes_per_sample();
        if bytes_per_sample == 0 {
            return Err(AudioError::InvalidParameter(
                "Invalid sample format".to_string(),
            ));
        }

        let sample_count = data.len() / (bytes_per_sample * channel_count);
        let mut samples = Vec::with_capacity(sample_count);

        match format {
            SampleFormat::U8 => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for ch in 0..channel_count {
                        let idx = i * channel_count + ch;
                        let sample = data[idx];
                        sum += f64::from(sample) / 128.0 - 1.0;
                    }
                    samples.push(sum / channel_count as f64);
                }
            }
            SampleFormat::S16 | SampleFormat::S16p => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for ch in 0..channel_count {
                        let idx = (i * channel_count + ch) * 2;
                        if idx + 1 < data.len() {
                            let sample = i16::from_le_bytes([data[idx], data[idx + 1]]);
                            sum += f64::from(sample) / 32768.0;
                        }
                    }
                    samples.push(sum / channel_count as f64);
                }
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for ch in 0..channel_count {
                        let idx = (i * channel_count + ch) * 4;
                        if idx + 3 < data.len() {
                            let sample = i32::from_le_bytes([
                                data[idx],
                                data[idx + 1],
                                data[idx + 2],
                                data[idx + 3],
                            ]);
                            sum += f64::from(sample) / 2_147_483_648.0;
                        }
                    }
                    samples.push(sum / channel_count as f64);
                }
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for ch in 0..channel_count {
                        let idx = (i * channel_count + ch) * 4;
                        if idx + 3 < data.len() {
                            let sample = f32::from_le_bytes([
                                data[idx],
                                data[idx + 1],
                                data[idx + 2],
                                data[idx + 3],
                            ]);
                            sum += f64::from(sample);
                        }
                    }
                    samples.push(sum / channel_count as f64);
                }
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for ch in 0..channel_count {
                        let idx = (i * channel_count + ch) * 8;
                        if idx + 7 < data.len() {
                            let sample = f64::from_le_bytes([
                                data[idx],
                                data[idx + 1],
                                data[idx + 2],
                                data[idx + 3],
                                data[idx + 4],
                                data[idx + 5],
                                data[idx + 6],
                                data[idx + 7],
                            ]);
                            sum += sample;
                        }
                    }
                    samples.push(sum / channel_count as f64);
                }
            }
            _ => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "Unsupported sample format: {}",
                    format
                )));
            }
        }

        Ok(samples)
    }

    /// Convert planar samples to mono f64.
    fn convert_planar_to_mono(
        &self,
        planes: &[bytes::Bytes],
        format: &SampleFormat,
    ) -> AudioResult<Vec<f64>> {
        if planes.is_empty() {
            return Err(AudioError::InvalidParameter("No planes".to_string()));
        }

        let bytes_per_sample = format.bytes_per_sample();
        if bytes_per_sample == 0 {
            return Err(AudioError::InvalidParameter(
                "Invalid sample format".to_string(),
            ));
        }

        let sample_count = planes[0].len() / bytes_per_sample;
        let mut samples = Vec::with_capacity(sample_count);

        match format {
            SampleFormat::U8 => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for plane in planes {
                        if i < plane.len() {
                            sum += f64::from(plane[i]) / 128.0 - 1.0;
                        }
                    }
                    samples.push(sum / planes.len() as f64);
                }
            }
            SampleFormat::S16 | SampleFormat::S16p => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for plane in planes {
                        let idx = i * 2;
                        if idx + 1 < plane.len() {
                            let sample = i16::from_le_bytes([plane[idx], plane[idx + 1]]);
                            sum += f64::from(sample) / 32768.0;
                        }
                    }
                    samples.push(sum / planes.len() as f64);
                }
            }
            SampleFormat::S32 | SampleFormat::S32p => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for plane in planes {
                        let idx = i * 4;
                        if idx + 3 < plane.len() {
                            let sample = i32::from_le_bytes([
                                plane[idx],
                                plane[idx + 1],
                                plane[idx + 2],
                                plane[idx + 3],
                            ]);
                            sum += f64::from(sample) / 2_147_483_648.0;
                        }
                    }
                    samples.push(sum / planes.len() as f64);
                }
            }
            SampleFormat::F32 | SampleFormat::F32p => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for plane in planes {
                        let idx = i * 4;
                        if idx + 3 < plane.len() {
                            let sample = f32::from_le_bytes([
                                plane[idx],
                                plane[idx + 1],
                                plane[idx + 2],
                                plane[idx + 3],
                            ]);
                            sum += f64::from(sample);
                        }
                    }
                    samples.push(sum / planes.len() as f64);
                }
            }
            SampleFormat::F64 | SampleFormat::F64p => {
                for i in 0..sample_count {
                    let mut sum = 0.0;
                    for plane in planes {
                        let idx = i * 8;
                        if idx + 7 < plane.len() {
                            let sample = f64::from_le_bytes([
                                plane[idx],
                                plane[idx + 1],
                                plane[idx + 2],
                                plane[idx + 3],
                                plane[idx + 4],
                                plane[idx + 5],
                                plane[idx + 6],
                                plane[idx + 7],
                            ]);
                            sum += sample;
                        }
                    }
                    samples.push(sum / planes.len() as f64);
                }
            }
            _ => {
                return Err(AudioError::UnsupportedFormat(format!(
                    "Unsupported sample format: {}",
                    format
                )));
            }
        }

        Ok(samples)
    }

    /// Compute spectrogram using FFT with overlapping windows.
    fn compute_spectrogram(
        &self,
        samples: &[f64],
        fft: &mut FftProcessor,
        _sample_rate: u32,
    ) -> Vec<Vec<f64>> {
        let mut spectrogram = Vec::new();
        let mut pos = 0;

        while pos + self.fft_size <= samples.len() {
            let window = &samples[pos..pos + self.fft_size];
            let magnitude = fft.magnitude_spectrum(window);

            // Keep only up to Nyquist frequency
            let nyquist_bin = self.fft_size / 2;
            spectrogram.push(magnitude[..nyquist_bin].to_vec());

            pos += self.hop_size;
        }

        spectrogram
    }

    /// Find spectral peaks in the spectrogram.
    #[allow(clippy::cast_precision_loss)]
    fn find_peaks(
        &self,
        spectrogram: &[Vec<f64>],
        sample_rate: u32,
        config: &FingerprintConfig,
    ) -> Vec<Peak> {
        let mut all_peaks = Vec::new();

        // Create frequency bands (logarithmic spacing)
        let bands = self.create_frequency_bands(sample_rate);

        for (frame_idx, frame) in spectrogram.iter().enumerate() {
            let time = (frame_idx * self.hop_size) as f64 / f64::from(sample_rate);

            // Find peaks in each frequency band
            let frame_peaks = self.find_peaks_in_frame(
                frame,
                &bands,
                time,
                sample_rate,
                config.peak_threshold,
                config.max_peaks_per_frame,
            );

            all_peaks.extend(frame_peaks);
        }

        all_peaks
    }

    /// Create logarithmically-spaced frequency bands.
    #[allow(clippy::cast_precision_loss)]
    fn create_frequency_bands(&self, sample_rate: u32) -> Vec<(usize, usize)> {
        let nyquist = f64::from(sample_rate) / 2.0;
        let min_freq = self.freq_range.0.max(20.0);
        let max_freq = self.freq_range.1.min(nyquist);

        let mut bands = Vec::new();

        // Logarithmic spacing
        let log_min = min_freq.ln();
        let log_max = max_freq.ln();
        let log_step = (log_max - log_min) / self.num_bands as f64;

        for i in 0..self.num_bands {
            let freq_low = (log_min + i as f64 * log_step).exp();
            let freq_high = (log_min + (i + 1) as f64 * log_step).exp();

            let bin_low = ((freq_low * self.fft_size as f64) / f64::from(sample_rate)) as usize;
            let bin_high = ((freq_high * self.fft_size as f64) / f64::from(sample_rate)) as usize;

            if bin_low < bin_high && bin_high < self.fft_size / 2 {
                bands.push((bin_low, bin_high));
            }
        }

        bands
    }

    /// Find peaks within a single time frame across frequency bands.
    #[allow(clippy::cast_precision_loss)]
    fn find_peaks_in_frame(
        &self,
        frame: &[f64],
        bands: &[(usize, usize)],
        time: f64,
        sample_rate: u32,
        threshold: f64,
        max_peaks: usize,
    ) -> Vec<Peak> {
        let mut peaks = Vec::new();

        for &(bin_low, bin_high) in bands {
            if bin_high >= frame.len() {
                continue;
            }

            // Find local maximum in this band
            let band_data = &frame[bin_low..bin_high];
            if band_data.is_empty() {
                continue;
            }

            let max_magnitude = band_data
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            if max_magnitude <= 0.0 {
                continue;
            }

            // Find peaks above threshold
            for (i, &magnitude) in band_data.iter().enumerate() {
                if magnitude < max_magnitude * threshold {
                    continue;
                }

                // Check if it's a local maximum (simple peak detection)
                let is_peak = if i > 0 && i < band_data.len() - 1 {
                    magnitude >= band_data[i - 1] && magnitude >= band_data[i + 1]
                } else {
                    true
                };

                if is_peak {
                    let bin = bin_low + i;
                    let frequency = (bin as f64 * f64::from(sample_rate)) / self.fft_size as f64;

                    peaks.push(Peak {
                        time,
                        frequency,
                        magnitude,
                        bin,
                    });
                }
            }
        }

        // Sort by magnitude and keep only top peaks
        peaks.sort_by(|a, b| {
            b.magnitude
                .partial_cmp(&a.magnitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        peaks.truncate(max_peaks);

        peaks
    }
}
