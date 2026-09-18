#![allow(dead_code)]
//! Watermark capacity calculation and analysis for different media types.
//!
//! Provides tools to estimate how many bits of watermark payload can be
//! embedded in a given audio or video frame, taking into account codec,
//! sample rate, bit depth, embedding algorithm, and psychoacoustic limits.

/// Supported media types for capacity estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaType {
    /// PCM audio (uncompressed).
    PcmAudio,
    /// Lossy compressed audio (e.g., MP3, AAC).
    CompressedAudio,
    /// Uncompressed video frame (raw pixels).
    RawVideo,
    /// Lossy compressed video (e.g., H.264 I-frame).
    CompressedVideo,
}

/// Embedding algorithm identifier used for capacity estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedAlgorithm {
    /// Spread-spectrum embedding.
    SpreadSpectrum,
    /// Least-significant-bit embedding.
    Lsb,
    /// Quantization index modulation.
    Qim,
    /// Echo hiding.
    EchoHiding,
    /// Phase coding.
    PhaseCoding,
    /// DCT-domain embedding.
    Dct,
}

/// Parameters describing the host media for capacity estimation.
#[derive(Debug, Clone)]
pub struct MediaParams {
    /// Media type.
    pub media_type: MediaType,
    /// Total number of samples (audio) or pixels (video).
    pub sample_count: usize,
    /// Sample rate in Hz (audio) or frame rate (video). Zero means unknown.
    pub sample_rate: u32,
    /// Bits per sample or bits per pixel.
    pub bit_depth: u16,
    /// Number of channels (audio) or colour planes (video).
    pub channels: u16,
}

impl Default for MediaParams {
    fn default() -> Self {
        Self {
            media_type: MediaType::PcmAudio,
            sample_count: 44100,
            sample_rate: 44100,
            bit_depth: 16,
            channels: 2,
        }
    }
}

/// Result of a capacity estimation.
#[derive(Debug, Clone)]
pub struct CapacityResult {
    /// Maximum payload capacity in bits.
    pub capacity_bits: usize,
    /// Maximum payload capacity in bytes (capacity_bits / 8, rounded down).
    pub capacity_bytes: usize,
    /// Effective bit-rate of the watermark channel in bits per second.
    pub bitrate_bps: f64,
    /// Estimated Signal-to-Noise ratio overhead in dB (lower is less perceptible).
    pub estimated_snr_db: f64,
    /// Human-readable summary.
    pub summary: String,
}

/// Calculates watermark embedding capacity for a given media and algorithm.
#[derive(Debug, Clone)]
pub struct CapacityCalculator {
    /// Embedding algorithm to evaluate.
    pub algorithm: EmbedAlgorithm,
    /// Redundancy factor (>= 1.0). Higher values improve robustness but reduce capacity.
    pub redundancy: f64,
    /// Error-correction overhead fraction (0.0 .. 1.0).
    pub ecc_overhead: f64,
}

impl Default for CapacityCalculator {
    fn default() -> Self {
        Self {
            algorithm: EmbedAlgorithm::SpreadSpectrum,
            redundancy: 4.0,
            ecc_overhead: 0.25,
        }
    }
}

impl CapacityCalculator {
    /// Create a new calculator for the given algorithm.
    pub fn new(algorithm: EmbedAlgorithm) -> Self {
        Self {
            algorithm,
            ..Self::default()
        }
    }

    /// Set the redundancy factor.
    pub fn with_redundancy(mut self, r: f64) -> Self {
        self.redundancy = r.max(1.0);
        self
    }

    /// Set the error-correction overhead fraction.
    pub fn with_ecc_overhead(mut self, e: f64) -> Self {
        self.ecc_overhead = e.clamp(0.0, 0.99);
        self
    }

    /// Estimate the raw (before ECC / redundancy) bit capacity for the given media.
    #[allow(clippy::cast_precision_loss)]
    fn raw_capacity(&self, params: &MediaParams) -> usize {
        let total_samples = params.sample_count as f64;
        match self.algorithm {
            EmbedAlgorithm::Lsb => {
                // One bit per sample, per channel
                params.sample_count * params.channels as usize
            }
            EmbedAlgorithm::SpreadSpectrum => {
                // chip_rate = 64 samples per bit typical
                let chip_rate = 64.0;
                (total_samples / chip_rate) as usize
            }
            EmbedAlgorithm::Qim => {
                // One bit per quantization cell; frame_size = 1024
                let frame_size = 1024.0;
                let frames = (total_samples / frame_size).floor();
                // ~4 bits per frame from usable DCT bins
                (frames * 4.0) as usize
            }
            EmbedAlgorithm::EchoHiding => {
                // segment_size = 512 samples per bit
                (total_samples / 512.0) as usize
            }
            EmbedAlgorithm::PhaseCoding => {
                // frame_size = 1024, ~19 usable phase bins per frame
                let frame_size = 1024.0;
                let frames = (total_samples / frame_size).floor();
                (frames * 19.0) as usize
            }
            EmbedAlgorithm::Dct => {
                // 8x8 block, 1 bit per block
                let block_size = 64.0;
                (total_samples / block_size) as usize
            }
        }
    }

    /// Calculate the effective capacity after redundancy and ECC.
    #[allow(clippy::cast_precision_loss)]
    pub fn calculate(&self, params: &MediaParams) -> CapacityResult {
        let raw = self.raw_capacity(params) as f64;
        let after_redundancy = raw / self.redundancy;
        let after_ecc = after_redundancy * (1.0 - self.ecc_overhead);
        let capacity_bits = after_ecc.floor().max(0.0) as usize;
        let capacity_bytes = capacity_bits / 8;

        let duration_secs = if params.sample_rate > 0 {
            params.sample_count as f64 / params.sample_rate as f64
        } else {
            1.0
        };
        let bitrate_bps = if duration_secs > 0.0 {
            capacity_bits as f64 / duration_secs
        } else {
            0.0
        };

        // Rough SNR estimate based on algorithm and bit depth
        let base_snr = match self.algorithm {
            EmbedAlgorithm::Lsb => params.bit_depth as f64 * 6.02,
            EmbedAlgorithm::SpreadSpectrum => 40.0 + params.bit_depth as f64,
            EmbedAlgorithm::Qim => 35.0 + params.bit_depth as f64 * 0.5,
            EmbedAlgorithm::EchoHiding => 30.0,
            EmbedAlgorithm::PhaseCoding => 38.0,
            EmbedAlgorithm::Dct => 36.0 + params.bit_depth as f64 * 0.3,
        };

        let summary = format!(
            "{:?} on {:?}: {} bits ({} bytes) at {:.1} bps, ~{:.1} dB SNR",
            self.algorithm, params.media_type, capacity_bits, capacity_bytes, bitrate_bps, base_snr,
        );

        CapacityResult {
            capacity_bits,
            capacity_bytes,
            bitrate_bps,
            estimated_snr_db: base_snr,
            summary,
        }
    }

    /// Compare capacity across all algorithms for the same media.
    pub fn compare_all(
        params: &MediaParams,
        redundancy: f64,
        ecc_overhead: f64,
    ) -> Vec<CapacityResult> {
        let algorithms = [
            EmbedAlgorithm::SpreadSpectrum,
            EmbedAlgorithm::Lsb,
            EmbedAlgorithm::Qim,
            EmbedAlgorithm::EchoHiding,
            EmbedAlgorithm::PhaseCoding,
            EmbedAlgorithm::Dct,
        ];
        algorithms
            .iter()
            .map(|&algo| {
                CapacityCalculator::new(algo)
                    .with_redundancy(redundancy)
                    .with_ecc_overhead(ecc_overhead)
                    .calculate(params)
            })
            .collect()
    }

    /// Check whether a payload of the given byte size fits in the media.
    pub fn payload_fits(&self, params: &MediaParams, payload_bytes: usize) -> bool {
        let result = self.calculate(params);
        result.capacity_bytes >= payload_bytes
    }

    /// Minimum number of samples required to embed `payload_bytes` bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn min_samples_for_payload(&self, payload_bytes: usize, channels: u16) -> usize {
        let payload_bits = payload_bytes * 8;
        let needed_raw = (payload_bits as f64) * self.redundancy / (1.0 - self.ecc_overhead);
        let per_sample_bits = match self.algorithm {
            EmbedAlgorithm::Lsb => channels as f64,
            EmbedAlgorithm::SpreadSpectrum => 1.0 / 64.0,
            EmbedAlgorithm::Qim => 4.0 / 1024.0,
            EmbedAlgorithm::EchoHiding => 1.0 / 512.0,
            EmbedAlgorithm::PhaseCoding => 19.0 / 1024.0,
            EmbedAlgorithm::Dct => 1.0 / 64.0,
        };
        if per_sample_bits <= 0.0 {
            return usize::MAX;
        }
        (needed_raw / per_sample_bits).ceil() as usize
    }
}

/// Convenience function: estimate capacity for PCM audio.
#[allow(clippy::cast_precision_loss)]
pub fn pcm_audio_capacity(
    sample_count: usize,
    sample_rate: u32,
    bit_depth: u16,
    channels: u16,
    algorithm: EmbedAlgorithm,
) -> CapacityResult {
    let params = MediaParams {
        media_type: MediaType::PcmAudio,
        sample_count,
        sample_rate,
        bit_depth,
        channels,
    };
    CapacityCalculator::new(algorithm).calculate(&params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_media_params() {
        let p = MediaParams::default();
        assert_eq!(p.media_type, MediaType::PcmAudio);
        assert_eq!(p.sample_count, 44100);
        assert_eq!(p.sample_rate, 44100);
        assert_eq!(p.bit_depth, 16);
        assert_eq!(p.channels, 2);
    }

    #[test]
    fn test_default_calculator() {
        let c = CapacityCalculator::default();
        assert_eq!(c.algorithm, EmbedAlgorithm::SpreadSpectrum);
        assert!((c.redundancy - 4.0).abs() < f64::EPSILON);
        assert!((c.ecc_overhead - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lsb_raw_capacity() {
        let calc = CapacityCalculator::new(EmbedAlgorithm::Lsb);
        let params = MediaParams {
            sample_count: 1000,
            channels: 2,
            ..MediaParams::default()
        };
        // raw = 1000 * 2 = 2000
        let raw = calc.raw_capacity(&params);
        assert_eq!(raw, 2000);
    }

    #[test]
    fn test_spread_spectrum_capacity_positive() {
        let calc = CapacityCalculator::new(EmbedAlgorithm::SpreadSpectrum);
        let params = MediaParams {
            sample_count: 44100,
            ..MediaParams::default()
        };
        let result = calc.calculate(&params);
        assert!(result.capacity_bits > 0);
        assert!(result.capacity_bytes > 0);
    }

    #[test]
    fn test_qim_capacity() {
        let calc = CapacityCalculator::new(EmbedAlgorithm::Qim);
        let params = MediaParams {
            sample_count: 44100,
            ..MediaParams::default()
        };
        let result = calc.calculate(&params);
        assert!(result.capacity_bits > 0);
    }

    #[test]
    fn test_echo_hiding_capacity() {
        let calc = CapacityCalculator::new(EmbedAlgorithm::EchoHiding);
        let params = MediaParams {
            sample_count: 44100,
            ..MediaParams::default()
        };
        let result = calc.calculate(&params);
        assert!(result.capacity_bits > 0);
    }

    #[test]
    fn test_phase_coding_capacity() {
        let calc = CapacityCalculator::new(EmbedAlgorithm::PhaseCoding);
        let params = MediaParams {
            sample_count: 44100,
            ..MediaParams::default()
        };
        let result = calc.calculate(&params);
        assert!(result.capacity_bits > 0);
    }

    #[test]
    fn test_dct_capacity() {
        let calc = CapacityCalculator::new(EmbedAlgorithm::Dct);
        let params = MediaParams {
            sample_count: 44100,
            ..MediaParams::default()
        };
        let result = calc.calculate(&params);
        assert!(result.capacity_bits > 0);
    }

    #[test]
    fn test_redundancy_reduces_capacity() {
        let params = MediaParams::default();
        let low = CapacityCalculator::new(EmbedAlgorithm::Lsb)
            .with_redundancy(2.0)
            .calculate(&params);
        let high = CapacityCalculator::new(EmbedAlgorithm::Lsb)
            .with_redundancy(8.0)
            .calculate(&params);
        assert!(low.capacity_bits > high.capacity_bits);
    }

    #[test]
    fn test_ecc_overhead_reduces_capacity() {
        let params = MediaParams::default();
        let low = CapacityCalculator::new(EmbedAlgorithm::Lsb)
            .with_ecc_overhead(0.1)
            .calculate(&params);
        let high = CapacityCalculator::new(EmbedAlgorithm::Lsb)
            .with_ecc_overhead(0.5)
            .calculate(&params);
        assert!(low.capacity_bits > high.capacity_bits);
    }

    #[test]
    fn test_compare_all_returns_six_entries() {
        let params = MediaParams::default();
        let results = CapacityCalculator::compare_all(&params, 4.0, 0.25);
        assert_eq!(results.len(), 6);
        for r in &results {
            assert!(!r.summary.is_empty());
        }
    }

    #[test]
    fn test_payload_fits() {
        let calc = CapacityCalculator::new(EmbedAlgorithm::Lsb)
            .with_redundancy(1.0)
            .with_ecc_overhead(0.0);
        let params = MediaParams {
            sample_count: 1000,
            channels: 1,
            ..MediaParams::default()
        };
        // raw = 1000, capacity_bytes = 125
        assert!(calc.payload_fits(&params, 100));
        assert!(!calc.payload_fits(&params, 200));
    }

    #[test]
    fn test_min_samples_for_payload() {
        let calc = CapacityCalculator::new(EmbedAlgorithm::Lsb)
            .with_redundancy(1.0)
            .with_ecc_overhead(0.0);
        let min = calc.min_samples_for_payload(10, 1);
        // 10 bytes = 80 bits, 1 bit per sample => 80 samples
        assert_eq!(min, 80);
    }

    #[test]
    fn test_pcm_audio_capacity_convenience() {
        let result = pcm_audio_capacity(44100, 44100, 16, 2, EmbedAlgorithm::Lsb);
        assert!(result.capacity_bits > 0);
        assert!(result.bitrate_bps > 0.0);
        assert!(result.estimated_snr_db > 0.0);
    }

    #[test]
    fn test_summary_contains_algorithm_name() {
        let params = MediaParams::default();
        let result = CapacityCalculator::new(EmbedAlgorithm::SpreadSpectrum).calculate(&params);
        assert!(result.summary.contains("SpreadSpectrum"));
    }
}
