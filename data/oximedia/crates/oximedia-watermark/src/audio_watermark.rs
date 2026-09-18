//! Inaudible audio watermarking via frequency-shift keying (FSK).
//!
//! Embeds a 32-bit payload into the near-ultrasonic range (~18 kHz) using FSK
//! so the watermark is imperceptible under normal listening conditions.

use std::f32::consts::PI;

/// Configuration for the audio watermark encoder / decoder.
#[derive(Debug, Clone)]
pub struct AudioWatermarkConfig {
    /// Carrier frequency in Hz (default ~18 kHz).
    pub carrier_hz: f32,
    /// Number of bits in the payload (default 32).
    pub payload_bits: u32,
    /// Duration of each bit in milliseconds (default 100 ms).
    pub bit_duration_ms: u32,
}

impl Default for AudioWatermarkConfig {
    fn default() -> Self {
        Self {
            carrier_hz: 18_000.0,
            payload_bits: 32,
            bit_duration_ms: 100,
        }
    }
}

impl AudioWatermarkConfig {
    /// Number of audio samples that represent one bit at the given sample rate.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn samples_per_bit(&self, sample_rate: u32) -> usize {
        ((self.bit_duration_ms as f32 / 1000.0) * sample_rate as f32) as usize
    }
}

/// Encode a single bit using FSK.
///
/// - `bit == false` → carrier at `carrier_hz`
/// - `bit == true`  → carrier at `carrier_hz + shift_hz`
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn encode_bit_fsk(
    bit: bool,
    carrier_hz: f32,
    shift_hz: f32,
    duration_samples: usize,
    sample_rate: u32,
) -> Vec<f32> {
    let freq = if bit {
        carrier_hz + shift_hz
    } else {
        carrier_hz
    };
    let sr = sample_rate as f32;
    (0..duration_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sr).sin())
        .collect()
}

/// Decode a single bit from a sample buffer using energy comparison at two frequencies.
///
/// Returns `true` if the energy at `carrier_hz + shift_hz` exceeds that at `carrier_hz`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn decode_bit_fsk(samples: &[f32], carrier_hz: f32, shift_hz: f32, sample_rate: u32) -> bool {
    let sr = sample_rate as f32;
    let n = samples.len();
    if n == 0 {
        return false;
    }
    let energy_at = |freq: f32| -> f32 {
        let (mut re, mut im) = (0.0_f32, 0.0_f32);
        for (i, &s) in samples.iter().enumerate() {
            let phase = 2.0 * PI * freq * i as f32 / sr;
            re += s * phase.cos();
            im += s * phase.sin();
        }
        re * re + im * im
    };
    let e0 = energy_at(carrier_hz);
    let e1 = energy_at(carrier_hz + shift_hz);
    e1 > e0
}

/// Encodes a 32-bit payload into an audio watermark using FSK.
#[derive(Debug, Clone)]
pub struct AudioWatermarkEncoder {
    /// Watermark configuration.
    pub config: AudioWatermarkConfig,
}

impl AudioWatermarkEncoder {
    /// Create a new encoder with the given configuration.
    #[must_use]
    pub fn new(config: AudioWatermarkConfig) -> Self {
        Self { config }
    }

    /// Encode a `payload` value into a sequence of FSK samples at `sample_rate`.
    ///
    /// The frequency shift used is 200 Hz above the carrier.
    #[must_use]
    pub fn encode(&self, payload: u32, sample_rate: u32) -> Vec<f32> {
        let shift_hz = 200.0_f32;
        let dur = self.config.samples_per_bit(sample_rate);
        let bits = self.config.payload_bits;
        let mut out = Vec::with_capacity(dur * bits as usize);
        for bit_idx in (0..bits).rev() {
            let bit = (payload >> bit_idx) & 1 == 1;
            let chunk = encode_bit_fsk(bit, self.config.carrier_hz, shift_hz, dur, sample_rate);
            out.extend_from_slice(&chunk);
        }
        out
    }

    /// Embed a watermark into a host audio signal.
    ///
    /// The watermark samples are added to the host with amplitude `gain`.
    /// If the watermark is longer than the host, only the overlapping portion is embedded.
    #[must_use]
    pub fn embed(host: &[f32], watermark: &[f32], gain: f32) -> Vec<f32> {
        let mut out = host.to_vec();
        let overlap = out.len().min(watermark.len());
        for i in 0..overlap {
            out[i] += watermark[i] * gain;
        }
        out
    }
}

/// Decodes a 32-bit payload from an FSK-watermarked audio signal.
#[derive(Debug, Clone)]
pub struct AudioWatermarkDecoder;

impl AudioWatermarkDecoder {
    /// Attempt to decode the watermark payload from `audio`.
    ///
    /// Returns `None` if the audio is too short to contain all bits.
    #[must_use]
    pub fn decode(audio: &[f32], config: &AudioWatermarkConfig, sample_rate: u32) -> Option<u32> {
        let shift_hz = 200.0_f32;
        let dur = config.samples_per_bit(sample_rate);
        let bits = config.payload_bits as usize;

        if audio.len() < dur * bits {
            return None;
        }

        let mut payload: u32 = 0;
        for bit_idx in (0..bits).rev() {
            let start = (bits - 1 - bit_idx) * dur;
            let chunk = &audio[start..start + dur];
            let bit = decode_bit_fsk(chunk, config.carrier_hz, shift_hz, sample_rate);
            if bit {
                payload |= 1 << bit_idx;
            }
        }
        Some(payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_carrier() {
        let cfg = AudioWatermarkConfig::default();
        assert!((cfg.carrier_hz - 18_000.0).abs() < 1.0);
    }

    #[test]
    fn test_config_default_bits() {
        let cfg = AudioWatermarkConfig::default();
        assert_eq!(cfg.payload_bits, 32);
    }

    #[test]
    fn test_config_default_duration() {
        let cfg = AudioWatermarkConfig::default();
        assert_eq!(cfg.bit_duration_ms, 100);
    }

    #[test]
    fn test_samples_per_bit() {
        let cfg = AudioWatermarkConfig::default();
        // 100 ms at 44100 Hz = 4410 samples
        assert_eq!(cfg.samples_per_bit(44100), 4410);
    }

    #[test]
    fn test_encode_bit_fsk_length() {
        let samples = encode_bit_fsk(false, 18000.0, 200.0, 100, 44100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_encode_bit_fsk_amplitude_range() {
        let samples = encode_bit_fsk(true, 18000.0, 200.0, 1000, 44100);
        for s in &samples {
            assert!(s.abs() <= 1.0 + 1e-5, "amplitude out of range: {s}");
        }
    }

    #[test]
    fn test_encode_bit_fsk_differs_by_bit() {
        let s0 = encode_bit_fsk(false, 1000.0, 500.0, 100, 8000);
        let s1 = encode_bit_fsk(true, 1000.0, 500.0, 100, 8000);
        // Different frequencies → different waveforms
        assert_ne!(s0, s1);
    }

    #[test]
    fn test_decode_bit_fsk_zero_bit() {
        // Bit 0: pure carrier
        let samples = encode_bit_fsk(false, 1000.0, 500.0, 2048, 8000);
        let decoded = decode_bit_fsk(&samples, 1000.0, 500.0, 8000);
        assert!(!decoded);
    }

    #[test]
    fn test_decode_bit_fsk_one_bit() {
        // Bit 1: carrier + shift
        let samples = encode_bit_fsk(true, 1000.0, 500.0, 2048, 8000);
        let decoded = decode_bit_fsk(&samples, 1000.0, 500.0, 8000);
        assert!(decoded);
    }

    #[test]
    fn test_decode_bit_fsk_empty() {
        let decoded = decode_bit_fsk(&[], 1000.0, 200.0, 44100);
        assert!(!decoded);
    }

    #[test]
    fn test_encoder_encode_length() {
        let cfg = AudioWatermarkConfig {
            carrier_hz: 1000.0,
            payload_bits: 4,
            bit_duration_ms: 10,
        };
        let enc = AudioWatermarkEncoder::new(cfg.clone());
        let samples = enc.encode(0b1010, 8000);
        // 4 bits × (10ms * 8000/1000) = 4 * 80 = 320
        assert_eq!(samples.len(), 4 * cfg.samples_per_bit(8000));
    }

    #[test]
    fn test_embed_preserves_length() {
        let host = vec![0.5_f32; 1000];
        let wm = vec![0.1_f32; 500];
        let out = AudioWatermarkEncoder::embed(&host, &wm, 0.01);
        assert_eq!(out.len(), host.len());
    }

    #[test]
    fn test_embed_adds_watermark() {
        let host = vec![0.0_f32; 100];
        let wm = vec![1.0_f32; 100];
        let out = AudioWatermarkEncoder::embed(&host, &wm, 0.1);
        // Every sample should be approximately 0.1
        for v in &out {
            assert!((v - 0.1).abs() < 1e-5, "unexpected value: {v}");
        }
    }

    #[test]
    fn test_decoder_returns_none_for_short_audio() {
        let cfg = AudioWatermarkConfig::default();
        let audio = vec![0.0_f32; 10];
        assert!(AudioWatermarkDecoder::decode(&audio, &cfg, 44100).is_none());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Use low carrier to avoid aliasing at the test sample rate.
        let cfg = AudioWatermarkConfig {
            carrier_hz: 1000.0,
            payload_bits: 4,
            bit_duration_ms: 50,
        };
        let sr = 8000_u32;
        let enc = AudioWatermarkEncoder::new(cfg.clone());
        let payload: u32 = 0b1010;
        let wm_samples = enc.encode(payload, sr);
        let decoded = AudioWatermarkDecoder::decode(&wm_samples, &cfg, sr);
        assert!(decoded.is_some());
        assert_eq!(decoded.expect("should succeed in test"), payload);
    }
}
