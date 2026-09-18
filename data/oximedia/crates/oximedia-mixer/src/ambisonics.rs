#![allow(dead_code)]
//! Ambisonics encoding and decoding for spatial audio mixing.
//!
//! Ambisonics represents a sound field using spherical harmonics. This module
//! implements B-format encoding (mono source → multi-channel spatial representation)
//! and decoding (B-format → loudspeaker or binaural signals).
//!
//! # Coordinate Convention
//!
//! - Azimuth 0° = front, 90° = left, 180° = back, −90° = right
//! - Elevation 0° = horizontal plane, 90° = directly above, −90° = directly below
//!
//! # Channel Ordering (ACN)
//!
//! Channels are ordered by Ambisonic Channel Number (ACN):
//! - ACN 0 = W (omnidirectional)
//! - ACN 1 = Y, ACN 2 = Z, ACN 3 = X  (first order)
//! - etc.
//!
//! # Normalisation (SN3D)
//!
//! Schmidt semi-normalised (SN3D) weights are used throughout.

use crate::MixerError;
use std::f32::consts::{FRAC_1_SQRT_2, PI};

// ---------------------------------------------------------------------------
// AmbisonicsOrder
// ---------------------------------------------------------------------------

/// Ambisonics order descriptor.
///
/// | Order | Channels | Common name |
/// |-------|----------|-------------|
/// |   1   |    4     | FOA         |
/// |   2   |    9     | SOA         |
/// |   3   |   16     | TOA         |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AmbisonicsOrder {
    /// Order (1 = FOA, 2 = SOA, 3 = TOA, etc.).
    pub order: u32,
}

impl AmbisonicsOrder {
    /// Create a new `AmbisonicsOrder`.
    #[must_use]
    pub fn new(order: u32) -> Self {
        Self { order }
    }

    /// Return the number of B-format channels for this order: (N+1)².
    #[must_use]
    pub fn channel_count(&self) -> usize {
        (self.order as usize + 1).pow(2)
    }
}

// ---------------------------------------------------------------------------
// Spherical harmonic helpers
// ---------------------------------------------------------------------------

/// Pre-computed SN3D-normalised real spherical harmonics up to order 3.
///
/// Arguments are in **radians**.
///
/// Returns a `Vec<f32>` of length `(order+1)²` in ACN order.
#[must_use]
fn spherical_harmonics_acn(order: u32, az_rad: f32, el_rad: f32) -> Vec<f32> {
    let cos_el = el_rad.cos();
    let sin_el = el_rad.sin();
    let cos_az = az_rad.cos();
    let sin_az = az_rad.sin();

    let n_ch = (order as usize + 1).pow(2);
    let mut sh = vec![0.0_f32; n_ch];

    // ACN 0 — l=0, m=0 — W
    sh[0] = FRAC_1_SQRT_2; // 1/√2

    if order == 0 {
        return sh;
    }

    // ACN 1 — l=1, m=−1 — Y
    sh[1] = sin_az * cos_el;
    // ACN 2 — l=1, m=0  — Z
    sh[2] = sin_el;
    // ACN 3 — l=1, m=+1 — X
    sh[3] = cos_az * cos_el;

    if order == 1 {
        return sh;
    }

    // --- Second order (l=2) --------------------------------------------------
    // ACN 4 — l=2, m=−2
    sh[4] = (3.0_f32).sqrt() / 2.0 * (2.0 * az_rad).sin() * cos_el * cos_el;
    // ACN 5 — l=2, m=−1
    sh[5] = (3.0_f32).sqrt() * sin_az * sin_el * cos_el;
    // ACN 6 — l=2, m=0
    sh[6] = 0.5 * (3.0 * sin_el * sin_el - 1.0);
    // ACN 7 — l=2, m=+1
    sh[7] = (3.0_f32).sqrt() * cos_az * sin_el * cos_el;
    // ACN 8 — l=2, m=+2
    sh[8] = (3.0_f32).sqrt() / 2.0 * (2.0 * az_rad).cos() * cos_el * cos_el;

    if order == 2 {
        return sh;
    }

    // --- Third order (l=3) ---------------------------------------------------
    let cos2_el = cos_el * cos_el;
    let cos3_el = cos2_el * cos_el;
    let sin2_el = sin_el * sin_el;

    // ACN 9 — l=3, m=−3
    sh[9] = (5.0_f32 / 8.0).sqrt() * (3.0 * az_rad).sin() * cos3_el;
    // ACN 10 — l=3, m=−2
    sh[10] = (15.0_f32).sqrt() / 2.0 * (2.0 * az_rad).sin() * sin_el * cos2_el;
    // ACN 11 — l=3, m=−1
    sh[11] = (3.0_f32 / 8.0).sqrt() * sin_az * cos_el * (5.0 * sin2_el - 1.0);
    // ACN 12 — l=3, m=0
    sh[12] = 0.5 * sin_el * (5.0 * sin2_el - 3.0);
    // ACN 13 — l=3, m=+1
    sh[13] = (3.0_f32 / 8.0).sqrt() * cos_az * cos_el * (5.0 * sin2_el - 1.0);
    // ACN 14 — l=3, m=+2
    sh[14] = (15.0_f32).sqrt() / 2.0 * (2.0 * az_rad).cos() * sin_el * cos2_el;
    // ACN 15 — l=3, m=+3
    sh[15] = (5.0_f32 / 8.0).sqrt() * (3.0 * az_rad).cos() * cos3_el;

    sh
}

// ---------------------------------------------------------------------------
// AmbisonicsEncoder
// ---------------------------------------------------------------------------

/// B-format encoder: places mono point-sources into an Ambisonic sound field.
#[derive(Debug, Clone)]
pub struct AmbisonicsEncoder {
    /// Target ambisonics order.
    pub order: AmbisonicsOrder,
}

impl AmbisonicsEncoder {
    /// Create a new encoder for the given order (1 = FOA, 2 = SOA, 3 = TOA).
    #[must_use]
    pub fn new(order: u32) -> Self {
        Self {
            order: AmbisonicsOrder::new(order),
        }
    }

    /// Encode a single mono sample at the given position into B-format.
    ///
    /// - `azimuth_deg`  — 0° = front, 90° = left, 180° = back, −90° = right
    /// - `elevation_deg` — 0° = horizontal, 90° = above, −90° = below
    ///
    /// Returns a `Vec<f32>` of length `(order+1)²` in ACN/SN3D ordering.
    #[must_use]
    pub fn encode_point_source(
        &self,
        sample: f32,
        azimuth_deg: f32,
        elevation_deg: f32,
    ) -> Vec<f32> {
        let az_rad = azimuth_deg * PI / 180.0;
        let el_rad = elevation_deg * PI / 180.0;
        let sh = spherical_harmonics_acn(self.order.order, az_rad, el_rad);
        sh.iter().map(|&h| h * sample).collect()
    }

    /// Encode and sum multiple moving sources into a single B-format buffer.
    ///
    /// Each element of `sources` is `(sample, azimuth_deg, elevation_deg)`.
    /// Returns a `Vec<f32>` of length `(order+1)²`.
    #[must_use]
    pub fn encode_sources(&self, sources: &[(f32, f32, f32)]) -> Vec<f32> {
        let n_ch = self.order.channel_count();
        let mut buf = vec![0.0_f32; n_ch];

        for &(sample, az, el) in sources {
            let encoded = self.encode_point_source(sample, az, el);
            for (b, e) in buf.iter_mut().zip(encoded.iter()) {
                *b += e;
            }
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// Decoding matrices
// ---------------------------------------------------------------------------

/// Compute a basic max-rE / pseudo-inverse decode matrix row for one speaker.
///
/// For each speaker at position `(az, el)` we compute the spherical harmonics
/// and scale them by `2 * (order+1)⁻¹` (basic mode / sampling decoder).
fn decode_row(order: u32, az_rad: f32, el_rad: f32) -> Vec<f32> {
    let sh = spherical_harmonics_acn(order, az_rad, el_rad);
    // Basic sampling decoder coefficient (normalised by channel count)
    let n_ch = (order as f32 + 1.0).powi(2);
    sh.iter().map(|&h| h / n_ch).collect()
}

// ---------------------------------------------------------------------------
// AmbisonicsDecoder
// ---------------------------------------------------------------------------

/// B-format decoder: converts a spatial sound field to loudspeaker / binaural signals.
#[derive(Debug, Clone)]
pub struct AmbisonicsDecoder {
    /// Ambisonics order.
    pub order: AmbisonicsOrder,
    /// Speaker layout as `(azimuth_deg, elevation_deg)` per speaker.
    pub speaker_layout: Vec<(f32, f32)>,
    /// Decode matrix: `[speaker][bformat_channel]`.
    decode_matrix: Vec<Vec<f32>>,
}

impl AmbisonicsDecoder {
    /// Create a decoder for the given order and arbitrary speaker layout.
    #[must_use]
    pub fn new(order: u32, speaker_layout: Vec<(f32, f32)>) -> Self {
        let amb_order = AmbisonicsOrder::new(order);
        let decode_matrix = speaker_layout
            .iter()
            .map(|&(az, el)| {
                let az_rad = az * PI / 180.0;
                let el_rad = el * PI / 180.0;
                decode_row(order, az_rad, el_rad)
            })
            .collect();

        Self {
            order: amb_order,
            speaker_layout,
            decode_matrix,
        }
    }

    /// Create a first-order decoder for a standard stereo pair (±30°).
    #[must_use]
    pub fn new_stereo() -> Self {
        Self::new(1, vec![(30.0, 0.0), (-30.0, 0.0)])
    }

    /// Create a first-order decoder for 5.1 surround.
    ///
    /// Speaker positions (ITU-R BS.775):
    /// - L  +30°, R −30°, C 0°, LS +110°, RS −110°, LFE (subwoofer, same as C for B-format)
    #[must_use]
    pub fn new_51() -> Self {
        Self::new(
            1,
            vec![
                (30.0, 0.0),   // L
                (-30.0, 0.0),  // R
                (0.0, 0.0),    // C
                (110.0, 0.0),  // Ls
                (-110.0, 0.0), // Rs
                (0.0, 0.0),    // LFE (uses centre position)
            ],
        )
    }

    /// Create a first-order binaural decoder (2 channels).
    ///
    /// Uses a simple left-ear / right-ear approximation at ±90°. For production
    /// use, replace with a proper HRTF convolution decoder.
    #[must_use]
    pub fn new_binaural() -> Self {
        Self::new(1, vec![(90.0, 0.0), (-90.0, 0.0)])
    }

    /// Decode a B-format frame to speaker signals.
    ///
    /// `bformat` must have exactly `(order+1)²` elements.
    /// Returns one sample per speaker.
    ///
    /// # Errors
    ///
    /// Returns `MixerError::InvalidParameter` if `bformat` length does not match
    /// the expected channel count.
    pub fn decode(&self, bformat: &[f32]) -> Result<Vec<f32>, MixerError> {
        let expected = self.order.channel_count();
        if bformat.len() != expected {
            return Err(MixerError::InvalidParameter(format!(
                "AmbisonicsDecoder::decode expected {expected} B-format channels, got {}",
                bformat.len()
            )));
        }

        let mut outputs = Vec::with_capacity(self.speaker_layout.len());
        for row in &self.decode_matrix {
            let sample: f32 = row.iter().zip(bformat.iter()).map(|(&d, &b)| d * b).sum();
            outputs.push(sample);
        }
        Ok(outputs)
    }

    /// Number of output (speaker) channels.
    #[must_use]
    pub fn num_speakers(&self) -> usize {
        self.speaker_layout.len()
    }
}

// ---------------------------------------------------------------------------
// Convenience round-trip helper
// ---------------------------------------------------------------------------

/// Encode a mono source and immediately decode to speakers (single-sample round-trip).
///
/// This is a convenience wrapper for small tests and prototypes.
///
/// # Errors
///
/// Propagates any error from [`AmbisonicsDecoder::decode`].
pub fn encode_decode(
    encoder: &AmbisonicsEncoder,
    decoder: &AmbisonicsDecoder,
    sample: f32,
    azimuth_deg: f32,
    elevation_deg: f32,
) -> Result<Vec<f32>, MixerError> {
    if encoder.order.order != decoder.order.order {
        return Err(MixerError::InvalidParameter(
            "encoder and decoder order mismatch".to_string(),
        ));
    }
    let bformat = encoder.encode_point_source(sample, azimuth_deg, elevation_deg);
    decoder.decode(&bformat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_1_SQRT_2;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ------------------------------------------------------------------
    // AmbisonicsOrder
    // ------------------------------------------------------------------

    #[test]
    fn test_foa_channel_count() {
        let o = AmbisonicsOrder::new(1);
        assert_eq!(o.channel_count(), 4);
    }

    #[test]
    fn test_soa_channel_count() {
        let o = AmbisonicsOrder::new(2);
        assert_eq!(o.channel_count(), 9);
    }

    #[test]
    fn test_toa_channel_count() {
        let o = AmbisonicsOrder::new(3);
        assert_eq!(o.channel_count(), 16);
    }

    // ------------------------------------------------------------------
    // Encoder – FOA
    // ------------------------------------------------------------------

    #[test]
    fn test_foa_encode_front() {
        let enc = AmbisonicsEncoder::new(1);
        let bformat = enc.encode_point_source(1.0, 0.0, 0.0);
        assert_eq!(bformat.len(), 4);
        // W = 1/√2
        assert!(
            approx_eq(bformat[0], FRAC_1_SQRT_2, 1e-5),
            "W {}",
            bformat[0]
        );
        // Y = sin(0)*cos(0) = 0
        assert!(approx_eq(bformat[1], 0.0, 1e-5), "Y {}", bformat[1]);
        // Z = sin(0) = 0
        assert!(approx_eq(bformat[2], 0.0, 1e-5), "Z {}", bformat[2]);
        // X = cos(0)*cos(0) = 1
        assert!(approx_eq(bformat[3], 1.0, 1e-5), "X {}", bformat[3]);
    }

    #[test]
    fn test_foa_encode_left() {
        let enc = AmbisonicsEncoder::new(1);
        let bformat = enc.encode_point_source(1.0, 90.0, 0.0);
        assert_eq!(bformat.len(), 4);
        // Y = sin(90°) = 1
        assert!(approx_eq(bformat[1], 1.0, 1e-5), "Y {}", bformat[1]);
        // X = cos(90°) = 0
        assert!(approx_eq(bformat[3], 0.0, 1e-5), "X {}", bformat[3]);
    }

    #[test]
    fn test_foa_encode_above() {
        let enc = AmbisonicsEncoder::new(1);
        let bformat = enc.encode_point_source(1.0, 0.0, 90.0);
        assert_eq!(bformat.len(), 4);
        // Z = sin(90°) = 1
        assert!(approx_eq(bformat[2], 1.0, 1e-5), "Z {}", bformat[2]);
        // X = cos(0°)*cos(90°) ≈ 0
        assert!(approx_eq(bformat[3], 0.0, 1e-5), "X {}", bformat[3]);
    }

    #[test]
    fn test_foa_encode_scales_with_amplitude() {
        let enc = AmbisonicsEncoder::new(1);
        let b1 = enc.encode_point_source(1.0, 45.0, 30.0);
        let b2 = enc.encode_point_source(2.0, 45.0, 30.0);
        for (a, b) in b1.iter().zip(b2.iter()) {
            assert!(approx_eq(*b, 2.0 * a, 1e-5));
        }
    }

    #[test]
    fn test_foa_encode_silence() {
        let enc = AmbisonicsEncoder::new(1);
        let bformat = enc.encode_point_source(0.0, 45.0, 30.0);
        for ch in &bformat {
            assert!(approx_eq(*ch, 0.0, 1e-10));
        }
    }

    // ------------------------------------------------------------------
    // Encoder – multi-order
    // ------------------------------------------------------------------

    #[test]
    fn test_soa_encode_front_channel_count() {
        let enc = AmbisonicsEncoder::new(2);
        let bformat = enc.encode_point_source(1.0, 0.0, 0.0);
        assert_eq!(bformat.len(), 9);
    }

    #[test]
    fn test_toa_encode_front_channel_count() {
        let enc = AmbisonicsEncoder::new(3);
        let bformat = enc.encode_point_source(1.0, 0.0, 0.0);
        assert_eq!(bformat.len(), 16);
    }

    // ------------------------------------------------------------------
    // encode_sources
    // ------------------------------------------------------------------

    #[test]
    fn test_encode_sources_single() {
        let enc = AmbisonicsEncoder::new(1);
        let direct = enc.encode_point_source(1.0, 30.0, 0.0);
        let via_sources = enc.encode_sources(&[(1.0, 30.0, 0.0)]);
        for (a, b) in direct.iter().zip(via_sources.iter()) {
            assert!(approx_eq(*a, *b, 1e-6));
        }
    }

    #[test]
    fn test_encode_sources_sums() {
        let enc = AmbisonicsEncoder::new(1);
        let a = enc.encode_point_source(1.0, 0.0, 0.0);
        let b = enc.encode_point_source(1.0, 90.0, 0.0);
        let combined = enc.encode_sources(&[(1.0, 0.0, 0.0), (1.0, 90.0, 0.0)]);
        for i in 0..4 {
            assert!(approx_eq(combined[i], a[i] + b[i], 1e-6));
        }
    }

    #[test]
    fn test_encode_sources_empty() {
        let enc = AmbisonicsEncoder::new(1);
        let buf = enc.encode_sources(&[]);
        assert_eq!(buf.len(), 4);
        for ch in &buf {
            assert!(approx_eq(*ch, 0.0, 1e-10));
        }
    }

    // ------------------------------------------------------------------
    // Decoder constructors
    // ------------------------------------------------------------------

    #[test]
    fn test_decoder_stereo_layout() {
        let dec = AmbisonicsDecoder::new_stereo();
        assert_eq!(dec.num_speakers(), 2);
        assert_eq!(dec.order.order, 1);
    }

    #[test]
    fn test_decoder_51_layout() {
        let dec = AmbisonicsDecoder::new_51();
        assert_eq!(dec.num_speakers(), 6);
    }

    #[test]
    fn test_decoder_binaural_layout() {
        let dec = AmbisonicsDecoder::new_binaural();
        assert_eq!(dec.num_speakers(), 2);
    }

    // ------------------------------------------------------------------
    // Decoder – decode()
    // ------------------------------------------------------------------

    #[test]
    fn test_decode_wrong_length_returns_error() {
        let dec = AmbisonicsDecoder::new_stereo();
        // FOA expects 4 channels; pass 3
        let err = dec.decode(&[0.0, 0.0, 0.0]);
        assert!(err.is_err());
    }

    #[test]
    fn test_decode_correct_output_count() {
        let dec = AmbisonicsDecoder::new_stereo();
        let enc = AmbisonicsEncoder::new(1);
        let bformat = enc.encode_point_source(1.0, 0.0, 0.0);
        let out = dec.decode(&bformat).expect("decode should succeed");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_decode_silence_is_silence() {
        let dec = AmbisonicsDecoder::new_stereo();
        let silence = vec![0.0_f32; 4];
        let out = dec.decode(&silence).expect("decode should succeed");
        for s in &out {
            assert!(approx_eq(*s, 0.0, 1e-10));
        }
    }

    // ------------------------------------------------------------------
    // Round-trip
    // ------------------------------------------------------------------

    #[test]
    fn test_round_trip_stereo_front() {
        let enc = AmbisonicsEncoder::new(1);
        let dec = AmbisonicsDecoder::new_stereo();
        let out = encode_decode(&enc, &dec, 1.0, 0.0, 0.0).expect("round-trip should succeed");
        // Front source → both speakers should have equal positive output
        assert_eq!(out.len(), 2);
        assert!(out[0] > 0.0, "L should be positive for front source");
        assert!(out[1] > 0.0, "R should be positive for front source");
        assert!(
            approx_eq(out[0], out[1], 1e-4),
            "L and R should be equal for front: L={}, R={}",
            out[0],
            out[1]
        );
    }

    #[test]
    fn test_round_trip_left_source_louder_left() {
        let enc = AmbisonicsEncoder::new(1);
        let dec = AmbisonicsDecoder::new_stereo();
        let out = encode_decode(&enc, &dec, 1.0, 90.0, 0.0).expect("round-trip should succeed");
        // Left source → left speaker louder
        assert!(
            out[0] > out[1],
            "Left speaker should be louder for 90° source: L={}, R={}",
            out[0],
            out[1]
        );
    }

    #[test]
    fn test_round_trip_order_mismatch_error() {
        let enc = AmbisonicsEncoder::new(1);
        let dec = AmbisonicsDecoder::new(2, vec![(0.0, 0.0), (90.0, 0.0)]);
        let result = encode_decode(&enc, &dec, 1.0, 0.0, 0.0);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // Zero-order (W only)
    // ------------------------------------------------------------------

    #[test]
    fn test_zero_order_encodes_single_channel() {
        let enc = AmbisonicsEncoder::new(0);
        let bformat = enc.encode_point_source(1.0, 45.0, 30.0);
        assert_eq!(bformat.len(), 1);
        assert!(approx_eq(bformat[0], FRAC_1_SQRT_2, 1e-5));
    }
}
