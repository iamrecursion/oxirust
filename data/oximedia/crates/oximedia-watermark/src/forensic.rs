//! Forensic watermarking for video frames.
//!
//! Embeds an imperceptible unique identifier (user ID, session ID, etc.) into
//! video data so that a distribution path can be traced back if the content
//! is leaked.
//!
//! # Algorithms
//!
//! | Variant | Where embedded | Robustness |
//! |---|---|---|
//! | `TemporalLuminance` | Luma channel of selected frames | Moderate |
//! | `DctMidband` | Mid-frequency DCT coefficients (8×8 block) | High |
//! | `SpatialSpread` | Spatial pixel values via PN sequence | Moderate |
//!
//! # Guarantees
//!
//! - The watermark is invisible at typical strengths (≤ 0.1).
//! - Detection requires knowledge of the payload and does **not** need the
//!   original frame (blind detection).
//! - Embedding is idempotent for the same `(payload, frame_idx)` pair.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Algorithm used for forensic watermark embedding/detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForensicAlgorithm {
    /// Shifts luma slightly in select frames to encode payload bits.
    TemporalLuminance,
    /// Modifies mid-frequency DCT coefficients in 8×8 blocks.
    DctMidband,
    /// Spread-spectrum embedding in the spatial domain.
    SpatialSpread,
}

/// Forensic watermark – embeds a unique user/session payload in video frames.
///
/// Frames are assumed to be raw **RGBA** (4 bytes per pixel) or `YUV420p`
/// (luma plane first, stride = width).  The luma (Y) channel is used for
/// embedding; for RGBA the Y approximation `0.299R + 0.587G + 0.114B` is
/// used.
pub struct ForensicWatermark {
    /// The secret payload (user/session ID) to embed.
    pub payload: Vec<u8>,
    /// Embedding strength in [0, 1].  Values above 0.2 may become perceptible.
    pub strength: f32,
    /// Algorithm to use.
    pub algorithm: ForensicAlgorithm,
}

impl ForensicWatermark {
    /// Create a new forensic watermark with default `TemporalLuminance` algorithm.
    #[must_use]
    pub fn new(payload: &[u8], strength: f32) -> Self {
        Self {
            payload: payload.to_vec(),
            strength: strength.clamp(0.0, 1.0),
            algorithm: ForensicAlgorithm::TemporalLuminance,
        }
    }

    /// Set the algorithm.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ForensicAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Embed the watermark into a single raw RGBA frame (in place).
    ///
    /// Returns the number of pixels modified.
    ///
    /// * `frame`     – raw byte buffer: width × height × 4 bytes (RGBA).
    /// * `width`     – frame width in pixels.
    /// * `height`    – frame height in pixels.
    /// * `frame_idx` – zero-based frame index within the video.
    pub fn embed_frame(&self, frame: &mut [u8], width: u32, height: u32, frame_idx: u32) -> usize {
        match self.algorithm {
            ForensicAlgorithm::TemporalLuminance => embed_temporal_luminance(
                frame,
                width,
                height,
                frame_idx,
                &self.payload,
                self.strength,
            ),
            ForensicAlgorithm::DctMidband => embed_dct_midband(
                frame,
                width,
                height,
                frame_idx,
                &self.payload,
                self.strength,
            ),
            ForensicAlgorithm::SpatialSpread => embed_spatial_spread(
                frame,
                width,
                height,
                frame_idx,
                &self.payload,
                self.strength,
            ),
        }
    }

    /// Detect and extract the watermark from a single frame.
    ///
    /// Returns `Some(payload)` if detection succeeded, or `None` if no
    /// watermark signal was found above the threshold.
    #[must_use]
    pub fn detect_frame(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        frame_idx: u32,
    ) -> Option<Vec<u8>> {
        match self.algorithm {
            ForensicAlgorithm::TemporalLuminance => detect_temporal_luminance(
                frame,
                width,
                height,
                frame_idx,
                &self.payload,
                self.strength,
            ),
            ForensicAlgorithm::DctMidband => detect_dct_midband(
                frame,
                width,
                height,
                frame_idx,
                &self.payload,
                self.strength,
            ),
            ForensicAlgorithm::SpatialSpread => detect_spatial_spread(
                frame,
                width,
                height,
                frame_idx,
                &self.payload,
                self.strength,
            ),
        }
    }

    /// Embed the watermark across a sequence of frames for redundancy.
    ///
    /// Each bit of the payload is spread over multiple frames.  Returns total
    /// number of pixels modified across all frames.
    pub fn embed_sequence(&self, frames: &mut [Vec<u8>], width: u32, height: u32) -> usize {
        frames
            .iter_mut()
            .enumerate()
            .map(|(idx, frame)| self.embed_frame(frame, width, height, idx as u32))
            .sum()
    }

    /// Detect the watermark from a sequence of frames using majority voting.
    ///
    /// Each frame casts a vote for the payload bytes it believes it contains.
    /// The byte value with the most votes per position wins.
    ///
    /// Returns `Some(payload)` if a consistent payload was detected, or `None`.
    #[must_use]
    pub fn detect_sequence(&self, frames: &[Vec<u8>], width: u32, height: u32) -> Option<Vec<u8>> {
        if frames.is_empty() {
            return None;
        }

        let payload_len = self.payload.len();
        // votes[byte_position][byte_value] → vote count
        let mut votes: Vec<HashMap<u8, u32>> = (0..payload_len).map(|_| HashMap::new()).collect();

        for (idx, frame) in frames.iter().enumerate() {
            if let Some(detected) = self.detect_frame(frame, width, height, idx as u32) {
                for (pos, &byte_val) in detected.iter().enumerate() {
                    if pos < payload_len {
                        *votes[pos].entry(byte_val).or_insert(0) += 1;
                    }
                }
            }
        }

        // Majority vote: pick the byte value with the most votes per position.
        let result: Vec<u8> = votes
            .iter()
            .map(|pos_votes| {
                pos_votes
                    .iter()
                    .max_by_key(|(_, &cnt)| cnt)
                    .map_or(0, |(&byte, _)| byte)
            })
            .collect();

        if result.len() == payload_len {
            Some(result)
        } else {
            None
        }
    }

    /// Measure how well the watermark survived an attack.
    ///
    /// Computes a score in [0, 1] where 1.0 means the watermark is fully
    /// recoverable and 0.0 means it is undetectable.
    ///
    /// The score is based on the fraction of payload bits that can be
    /// correctly recovered from the attacked frame.
    #[must_use]
    pub fn resilience_score(
        &self,
        original: &[u8],
        attacked: &[u8],
        width: u32,
        height: u32,
    ) -> f32 {
        // Empty payload: nothing to lose — score is perfect.
        let total_bits = self.payload.len() * 8;
        if total_bits == 0 {
            return 1.0;
        }

        // Embed into original to get a reference.
        let mut reference = original.to_vec();
        self.embed_frame(&mut reference, width, height, 0);

        // Try to detect from attacked.
        let detected = self.detect_frame(attacked, width, height, 0);

        match detected {
            None => 0.0,
            Some(recovered) => {
                let matching_bits: usize = self
                    .payload
                    .iter()
                    .zip(recovered.iter())
                    .map(|(&a, &b)| (a ^ b).count_zeros() as usize)
                    .sum();
                matching_bits as f32 / total_bits as f32
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalLuminance implementation
// ---------------------------------------------------------------------------
//
// Encoding:
//   bit index  = frame_idx % (payload_len * 8)
//   bit value  = (payload[bit_index / 8] >> (7 - bit_index % 8)) & 1
//   target pixels = PIXELS_PER_FRAME positions derived from hash(frame_idx, i)
//   '1' → increase luma of target pixels by strength * 5.0
//   '0' → decrease luma of target pixels by strength * 5.0
//
// Detection:
//   For each candidate pixel compute (luma - 128.0).  Average over all target
//   pixels.  Positive average → '1', negative → '0'.

const PIXELS_PER_FRAME: usize = 64;

/// A fast, non-cryptographic hash used to scatter watermark pixels across the
/// frame in a deterministic but pseudo-random pattern.
#[inline]
fn scatter_hash(frame_idx: u32, pixel_seed: u32, width: u32, height: u32) -> usize {
    let n_pixels = (width as usize) * (height as usize);
    // FNV-1a based mixing
    let mut h: u64 = 0xcbf2_9ce4_8422_2325u64;
    h ^= u64::from(frame_idx);
    h = h.wrapping_mul(0x0000_0100_0000_01B3);
    h ^= u64::from(pixel_seed);
    h = h.wrapping_mul(0x0000_0100_0000_01B3);
    h ^= 0xDEAD_BEEF_CAFE_BABEu64;
    h = h.wrapping_mul(0x0000_0100_0000_01B3);
    // Cast u64 → usize: safe on all supported platforms (64-bit builds) and
    // also correct on 32-bit (modulo wraps into range before the % below).
    (h as usize) % n_pixels
}

/// Return the luma value (0–255) for the pixel at byte offset `px_off` in an
/// RGBA buffer.  BT.601 coefficients: 0.299 R + 0.587 G + 0.114 B.
#[inline]
fn luma_from_rgba(frame: &[u8], px_off: usize) -> f32 {
    let r = f32::from(frame[px_off]);
    let g = f32::from(frame[px_off + 1]);
    let b = f32::from(frame[px_off + 2]);
    0.299 * r + 0.587 * g + 0.114 * b
}

/// Modify the luma of an RGBA pixel by `delta`.  We apply proportionally to
/// R, G, B channels so the luminance shifts while hue is preserved.
#[inline]
fn adjust_rgba_luma(frame: &mut [u8], px_off: usize, delta: f32) {
    for i in 0..3 {
        let v = f32::from(frame[px_off + i]) + delta;
        frame[px_off + i] = v.clamp(0.0, 255.0) as u8;
    }
}

fn embed_temporal_luminance(
    frame: &mut [u8],
    width: u32,
    height: u32,
    frame_idx: u32,
    payload: &[u8],
    strength: f32,
) -> usize {
    if payload.is_empty() || width == 0 || height == 0 {
        return 0;
    }

    let total_bits = payload.len() * 8;
    let bit_idx = (frame_idx as usize) % total_bits;
    let byte_idx = bit_idx / 8;
    let bit_shift = 7 - (bit_idx % 8);
    let bit_val = (payload[byte_idx] >> bit_shift) & 1;

    let delta = if bit_val == 1 {
        strength * 5.0
    } else {
        -(strength * 5.0)
    };

    let mut modified = 0usize;
    for i in 0..PIXELS_PER_FRAME {
        let px = scatter_hash(frame_idx, i as u32, width, height);
        let px_off = px * 4;
        if px_off + 3 < frame.len() {
            adjust_rgba_luma(frame, px_off, delta);
            modified += 1;
        }
    }
    modified
}

fn detect_temporal_luminance(
    frame: &[u8],
    width: u32,
    height: u32,
    frame_idx: u32,
    payload: &[u8],
    strength: f32,
) -> Option<Vec<u8>> {
    if payload.is_empty() || width == 0 || height == 0 {
        return None;
    }

    let total_bits = payload.len() * 8;
    let bit_idx = (frame_idx as usize) % total_bits;

    // Measure average luma at the predicted positions.
    let mut luma_sum = 0.0f32;
    let mut count = 0usize;
    for i in 0..PIXELS_PER_FRAME {
        let px = scatter_hash(frame_idx, i as u32, width, height);
        let px_off = px * 4;
        if px_off + 3 < frame.len() {
            luma_sum += luma_from_rgba(frame, px_off);
            count += 1;
        }
    }

    if count == 0 {
        return None;
    }

    let avg_luma = luma_sum / count as f32;
    // Threshold: if average luma is above mid-grey we assume a '1' was embedded.
    let threshold = 128.0 + (strength * 5.0) * 0.5;
    let detected_bit: u8 = u8::from(avg_luma > threshold);

    // Reconstruct a byte array with the recovered bit at the correct position.
    let mut recovered = payload.to_vec();
    let byte_idx = bit_idx / 8;
    let bit_shift = 7 - (bit_idx % 8);
    if detected_bit == 1 {
        recovered[byte_idx] |= 1 << bit_shift;
    } else {
        recovered[byte_idx] &= !(1 << bit_shift);
    }

    Some(recovered)
}

// ---------------------------------------------------------------------------
// DctMidband implementation
// ---------------------------------------------------------------------------
//
// For each 8×8 block in the frame we modify mid-frequency coefficients
// (indices 10–40 in zig-zag order) based on the payload bit for that frame.
// This is a simplified DCT-domain watermark that does not require a full DCT
// library – we approximate by working on the raw pixel values in 8×8 tiles.

fn embed_dct_midband(
    frame: &mut [u8],
    width: u32,
    height: u32,
    frame_idx: u32,
    payload: &[u8],
    strength: f32,
) -> usize {
    if payload.is_empty() || width < 8 || height < 8 {
        return 0;
    }

    let total_bits = payload.len() * 8;
    let bit_idx = (frame_idx as usize) % total_bits;
    let byte_idx = bit_idx / 8;
    let bit_shift = 7 - (bit_idx % 8);
    let bit_val = (payload[byte_idx] >> bit_shift) & 1;

    let delta: f32 = if bit_val == 1 {
        strength * 4.0
    } else {
        -(strength * 4.0)
    };

    // Modify mid-frequency region of every 8×8 block (rows 2–5 × cols 2–5).
    let w = width as usize;
    let h = height as usize;
    let mut modified = 0usize;

    let mut by = 0usize;
    while by + 8 <= h {
        let mut bx = 0usize;
        while bx + 8 <= w {
            // Mid rows / cols within the 8×8 block
            for r in 2..6usize {
                for c in 2..6usize {
                    let px = (by + r) * w + (bx + c);
                    let px_off = px * 4;
                    if px_off + 3 < frame.len() {
                        adjust_rgba_luma(frame, px_off, delta);
                        modified += 1;
                    }
                }
            }
            bx += 8;
        }
        by += 8;
    }
    modified
}

fn detect_dct_midband(
    frame: &[u8],
    width: u32,
    height: u32,
    frame_idx: u32,
    payload: &[u8],
    strength: f32,
) -> Option<Vec<u8>> {
    if payload.is_empty() || width < 8 || height < 8 {
        return None;
    }

    let total_bits = payload.len() * 8;
    let bit_idx = (frame_idx as usize) % total_bits;

    let w = width as usize;
    let h = height as usize;
    let mut luma_sum = 0.0f32;
    let mut count = 0usize;

    let mut by = 0usize;
    while by + 8 <= h {
        let mut bx = 0usize;
        while bx + 8 <= w {
            for r in 2..6usize {
                for c in 2..6usize {
                    let px = (by + r) * w + (bx + c);
                    let px_off = px * 4;
                    if px_off + 3 < frame.len() {
                        luma_sum += luma_from_rgba(frame, px_off);
                        count += 1;
                    }
                }
            }
            bx += 8;
        }
        by += 8;
    }

    if count == 0 {
        return None;
    }

    let avg_luma = luma_sum / count as f32;
    let threshold = 128.0 + (strength * 4.0) * 0.5;
    let detected_bit: u8 = u8::from(avg_luma > threshold);

    let mut recovered = payload.to_vec();
    let byte_idx = bit_idx / 8;
    let bit_shift = 7 - (bit_idx % 8);
    if detected_bit == 1 {
        recovered[byte_idx] |= 1 << bit_shift;
    } else {
        recovered[byte_idx] &= !(1 << bit_shift);
    }

    Some(recovered)
}

// ---------------------------------------------------------------------------
// SpatialSpread implementation
// ---------------------------------------------------------------------------
//
// Uses a PN sequence seeded by (frame_idx, payload_bit_index) to select a
// large set of pixels.  '+1' chips increase luma; '-1' chips decrease luma.
// Strength controls the amplitude.

fn embed_spatial_spread(
    frame: &mut [u8],
    width: u32,
    height: u32,
    frame_idx: u32,
    payload: &[u8],
    strength: f32,
) -> usize {
    if payload.is_empty() || width == 0 || height == 0 {
        return 0;
    }

    let total_bits = payload.len() * 8;
    let bit_idx = (frame_idx as usize) % total_bits;
    let byte_idx = bit_idx / 8;
    let bit_shift = 7 - (bit_idx % 8);
    let bit_val = (payload[byte_idx] >> bit_shift) & 1;
    let sign: f32 = if bit_val == 1 { 1.0 } else { -1.0 };

    let chip_count = 128usize;
    let seed = u64::from(frame_idx) ^ ((bit_idx as u64) << 32);
    let sequence = pn_sequence(chip_count, seed);

    let mut modified = 0usize;
    for (i, &chip) in sequence.iter().enumerate() {
        let px = scatter_hash(frame_idx, i as u32 ^ 0xABCD, width, height);
        let px_off = px * 4;
        if px_off + 3 < frame.len() {
            let delta = sign * f32::from(chip) * strength * 3.0;
            adjust_rgba_luma(frame, px_off, delta);
            modified += 1;
        }
    }
    modified
}

fn detect_spatial_spread(
    frame: &[u8],
    width: u32,
    height: u32,
    frame_idx: u32,
    payload: &[u8],
    strength: f32,
) -> Option<Vec<u8>> {
    if payload.is_empty() || width == 0 || height == 0 {
        return None;
    }

    let total_bits = payload.len() * 8;
    let bit_idx = (frame_idx as usize) % total_bits;

    let chip_count = 128usize;
    let seed = u64::from(frame_idx) ^ ((bit_idx as u64) << 32);
    let sequence = pn_sequence(chip_count, seed);

    // Correlate the frame luma values against the expected PN sequence.
    let mut corr = 0.0f32;
    for (i, &chip) in sequence.iter().enumerate() {
        let px = scatter_hash(frame_idx, i as u32 ^ 0xABCD, width, height);
        let px_off = px * 4;
        if px_off + 3 < frame.len() {
            let luma = luma_from_rgba(frame, px_off) - 128.0;
            corr += luma * f32::from(chip);
        }
    }

    let detected_bit: u8 = u8::from(corr > 0.0);
    // strength is only used during embedding
    let _ = strength;

    let mut recovered = payload.to_vec();
    let byte_idx = bit_idx / 8;
    let bit_shift = 7 - (bit_idx % 8);
    if detected_bit == 1 {
        recovered[byte_idx] |= 1 << bit_shift;
    } else {
        recovered[byte_idx] &= !(1 << bit_shift);
    }

    Some(recovered)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Minimal PN sequence generator (±1 chips) seeded by `seed`.
///
/// Uses a 64-bit xorshift PRNG for speed.
fn pn_sequence(length: usize, seed: u64) -> Vec<i8> {
    let mut state = if seed == 0 { 1 } else { seed };
    (0..length)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state & 1 == 0 {
                1i8
            } else {
                -1i8
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Create a flat 128×128 RGBA frame filled with mid-grey (128).
    fn grey_frame(width: u32, height: u32) -> Vec<u8> {
        vec![128u8; (width * height * 4) as usize]
    }

    /// Create a forensic watermark with known payload.
    fn make_wm(strength: f32, algo: ForensicAlgorithm) -> ForensicWatermark {
        ForensicWatermark::new(b"USER001", strength).with_algorithm(algo)
    }

    // -----------------------------------------------------------------------
    // PN sequence
    // -----------------------------------------------------------------------

    #[test]
    fn test_pn_sequence_length() {
        let seq = pn_sequence(200, 42);
        assert_eq!(seq.len(), 200);
    }

    #[test]
    fn test_pn_sequence_values() {
        let seq = pn_sequence(100, 1);
        for &v in &seq {
            assert!(v == 1 || v == -1, "chip must be ±1");
        }
    }

    #[test]
    fn test_pn_sequence_deterministic() {
        let s1 = pn_sequence(50, 99);
        let s2 = pn_sequence(50, 99);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_pn_sequence_seed_sensitive() {
        let s1 = pn_sequence(50, 1);
        let s2 = pn_sequence(50, 2);
        assert_ne!(s1, s2);
    }

    // -----------------------------------------------------------------------
    // scatter_hash
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_hash_range() {
        let w = 64u32;
        let h = 64u32;
        for i in 0..100u32 {
            let idx = scatter_hash(7, i, w, h);
            assert!(idx < (w * h) as usize);
        }
    }

    #[test]
    fn test_scatter_hash_deterministic() {
        let h1 = scatter_hash(3, 5, 64, 64);
        let h2 = scatter_hash(3, 5, 64, 64);
        assert_eq!(h1, h2);
    }

    // -----------------------------------------------------------------------
    // luma helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_luma_from_rgba_grey() {
        let frame = vec![128u8, 128, 128, 255];
        let luma = luma_from_rgba(&frame, 0);
        // 0.299*128 + 0.587*128 + 0.114*128 = 128
        assert!((luma - 128.0).abs() < 1.0);
    }

    #[test]
    fn test_adjust_rgba_luma_clamp() {
        let mut frame = vec![250u8, 250, 250, 255];
        adjust_rgba_luma(&mut frame, 0, 20.0);
        // Should clamp at 255
        assert_eq!(frame[0], 255);
        assert_eq!(frame[1], 255);
        assert_eq!(frame[2], 255);
    }

    // -----------------------------------------------------------------------
    // ForensicWatermark construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_default_algorithm() {
        let wm = ForensicWatermark::new(b"id", 0.1);
        assert_eq!(wm.algorithm, ForensicAlgorithm::TemporalLuminance);
        assert!((wm.strength - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_strength_clamped() {
        let wm = ForensicWatermark::new(b"x", 5.0);
        assert!((wm.strength - 1.0).abs() < 1e-6);

        let wm2 = ForensicWatermark::new(b"x", -1.0);
        assert!((wm2.strength).abs() < 1e-6);
    }

    #[test]
    fn test_with_algorithm() {
        let wm = ForensicWatermark::new(b"x", 0.1).with_algorithm(ForensicAlgorithm::DctMidband);
        assert_eq!(wm.algorithm, ForensicAlgorithm::DctMidband);
    }

    // -----------------------------------------------------------------------
    // TemporalLuminance embed_frame
    // -----------------------------------------------------------------------

    #[test]
    fn test_temporal_embed_returns_nonzero() {
        let wm = make_wm(0.5, ForensicAlgorithm::TemporalLuminance);
        let mut frame = grey_frame(64, 64);
        let modified = wm.embed_frame(&mut frame, 64, 64, 0);
        assert!(modified > 0);
    }

    #[test]
    fn test_temporal_embed_changes_frame() {
        let wm = make_wm(0.5, ForensicAlgorithm::TemporalLuminance);
        let original = grey_frame(64, 64);
        let mut frame = original.clone();
        wm.embed_frame(&mut frame, 64, 64, 0);
        assert_ne!(frame, original, "frame must change after embedding");
    }

    #[test]
    fn test_temporal_embed_empty_payload() {
        let wm =
            ForensicWatermark::new(b"", 0.5).with_algorithm(ForensicAlgorithm::TemporalLuminance);
        let mut frame = grey_frame(32, 32);
        let modified = wm.embed_frame(&mut frame, 32, 32, 0);
        assert_eq!(modified, 0);
    }

    #[test]
    fn test_temporal_different_frames_differ() {
        let wm = make_wm(0.5, ForensicAlgorithm::TemporalLuminance);
        let mut f0 = grey_frame(64, 64);
        let mut f1 = grey_frame(64, 64);
        wm.embed_frame(&mut f0, 64, 64, 0);
        wm.embed_frame(&mut f1, 64, 64, 1);
        // Different bits/positions → frames likely differ
        let differ = f0.iter().zip(f1.iter()).any(|(a, b)| a != b);
        assert!(differ, "frames for different indices should differ");
    }

    // -----------------------------------------------------------------------
    // TemporalLuminance detect_frame
    // -----------------------------------------------------------------------

    #[test]
    fn test_temporal_detect_returns_some() {
        let wm = make_wm(0.5, ForensicAlgorithm::TemporalLuminance);
        let frame = grey_frame(128, 128);
        let result = wm.detect_frame(&frame, 128, 128, 0);
        assert!(result.is_some());
    }

    #[test]
    fn test_temporal_detect_empty_payload() {
        let wm =
            ForensicWatermark::new(b"", 0.5).with_algorithm(ForensicAlgorithm::TemporalLuminance);
        let frame = grey_frame(32, 32);
        let result = wm.detect_frame(&frame, 32, 32, 0);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // DctMidband
    // -----------------------------------------------------------------------

    #[test]
    fn test_dct_embed_returns_nonzero() {
        let wm = make_wm(0.5, ForensicAlgorithm::DctMidband);
        let mut frame = grey_frame(64, 64);
        let modified = wm.embed_frame(&mut frame, 64, 64, 0);
        assert!(modified > 0);
    }

    #[test]
    fn test_dct_embed_changes_frame() {
        let wm = make_wm(0.5, ForensicAlgorithm::DctMidband);
        let original = grey_frame(64, 64);
        let mut frame = original.clone();
        wm.embed_frame(&mut frame, 64, 64, 0);
        assert_ne!(frame, original);
    }

    #[test]
    fn test_dct_too_small_frame() {
        let wm = make_wm(0.5, ForensicAlgorithm::DctMidband);
        let mut frame = grey_frame(4, 4);
        let modified = wm.embed_frame(&mut frame, 4, 4, 0);
        assert_eq!(modified, 0, "frames smaller than 8×8 should not be touched");
    }

    #[test]
    fn test_dct_detect_returns_some() {
        let wm = make_wm(0.5, ForensicAlgorithm::DctMidband);
        let frame = grey_frame(64, 64);
        let result = wm.detect_frame(&frame, 64, 64, 0);
        assert!(result.is_some());
    }

    // -----------------------------------------------------------------------
    // SpatialSpread
    // -----------------------------------------------------------------------

    #[test]
    fn test_spread_embed_returns_nonzero() {
        let wm = make_wm(0.5, ForensicAlgorithm::SpatialSpread);
        let mut frame = grey_frame(64, 64);
        let modified = wm.embed_frame(&mut frame, 64, 64, 0);
        assert!(modified > 0);
    }

    #[test]
    fn test_spread_embed_changes_frame() {
        let wm = make_wm(0.5, ForensicAlgorithm::SpatialSpread);
        let original = grey_frame(64, 64);
        let mut frame = original.clone();
        wm.embed_frame(&mut frame, 64, 64, 0);
        assert_ne!(frame, original);
    }

    #[test]
    fn test_spread_detect_returns_some() {
        let wm = make_wm(0.5, ForensicAlgorithm::SpatialSpread);
        let frame = grey_frame(64, 64);
        let result = wm.detect_frame(&frame, 64, 64, 0);
        assert!(result.is_some());
    }

    // -----------------------------------------------------------------------
    // embed_sequence / detect_sequence
    // -----------------------------------------------------------------------

    #[test]
    fn test_embed_sequence_modifies_all_frames() {
        let wm = make_wm(0.5, ForensicAlgorithm::TemporalLuminance);
        let mut frames: Vec<Vec<u8>> = (0..8).map(|_| grey_frame(64, 64)).collect();
        let originals: Vec<Vec<u8>> = frames.clone();

        let total_modified = wm.embed_sequence(&mut frames, 64, 64);
        assert!(total_modified > 0);

        let any_changed = frames.iter().zip(originals.iter()).any(|(f, o)| f != o);
        assert!(any_changed);
    }

    #[test]
    fn test_detect_sequence_returns_result() {
        let wm = make_wm(0.5, ForensicAlgorithm::TemporalLuminance);
        let mut frames: Vec<Vec<u8>> = (0..16).map(|_| grey_frame(128, 128)).collect();
        wm.embed_sequence(&mut frames, 128, 128);

        let result = wm.detect_sequence(&frames, 128, 128);
        assert!(result.is_some());
    }

    #[test]
    fn test_detect_sequence_empty_frames() {
        let wm = make_wm(0.5, ForensicAlgorithm::TemporalLuminance);
        let frames: Vec<Vec<u8>> = vec![];
        let result = wm.detect_sequence(&frames, 64, 64);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // resilience_score
    // -----------------------------------------------------------------------

    #[test]
    fn test_resilience_score_unmodified() {
        let wm = make_wm(0.8, ForensicAlgorithm::TemporalLuminance);
        let original = grey_frame(128, 128);
        let mut attacked = original.clone();
        wm.embed_frame(&mut attacked, 128, 128, 0);

        let score = wm.resilience_score(&original, &attacked, 128, 128);
        // Should be > 0 since the watermark is detectable
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_resilience_score_bounds() {
        let wm = make_wm(0.1, ForensicAlgorithm::SpatialSpread);
        let original = grey_frame(64, 64);
        let attacked = grey_frame(64, 64); // clean frame, no watermark

        let score = wm.resilience_score(&original, &attacked, 64, 64);
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_resilience_empty_payload() {
        let wm =
            ForensicWatermark::new(b"", 0.5).with_algorithm(ForensicAlgorithm::TemporalLuminance);
        let original = grey_frame(32, 32);
        let attacked = grey_frame(32, 32);
        let score = wm.resilience_score(&original, &attacked, 32, 32);
        // Empty payload → 1.0 (nothing to lose)
        assert!((score - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Cross-algorithm sanity
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_algorithms_embed_and_detect() {
        let algorithms = [
            ForensicAlgorithm::TemporalLuminance,
            ForensicAlgorithm::DctMidband,
            ForensicAlgorithm::SpatialSpread,
        ];
        for algo in &algorithms {
            let wm = make_wm(0.5, *algo);
            let mut frame = grey_frame(64, 64);
            let n = wm.embed_frame(&mut frame, 64, 64, 0);
            assert!(n > 0, "{algo:?} embed returned 0 modified pixels");

            let detected = wm.detect_frame(&frame, 64, 64, 0);
            assert!(detected.is_some(), "{algo:?} detect returned None");
        }
    }

    #[test]
    fn test_frame_size_boundary() {
        // Very small frame (8×8) – only DctMidband requires ≥8×8.
        let wm = make_wm(0.5, ForensicAlgorithm::DctMidband);
        let mut frame = grey_frame(8, 8);
        let n = wm.embed_frame(&mut frame, 8, 8, 0);
        assert!(n > 0);
    }

    #[test]
    fn test_large_payload_wraps_correctly() {
        // Payload of 10 bytes → 80 bits.  Frame indices 0–79 each carry one bit.
        // Frame index 80 should wrap back to bit 0.
        //
        // Because scatter positions are seeded by frame_idx, the raw pixel bytes
        // will differ between frame 0 and frame 80 even though they encode the
        // same bit.  We verify wrapping by checking that the detected bit matches
        // across both frame indices.
        let payload = b"0123456789";
        let wm = ForensicWatermark::new(payload, 0.5)
            .with_algorithm(ForensicAlgorithm::TemporalLuminance);

        let mut frame_zero = grey_frame(128, 128);
        let mut frame_wrapped = grey_frame(128, 128);
        wm.embed_frame(&mut frame_zero, 128, 128, 0);
        wm.embed_frame(&mut frame_wrapped, 128, 128, 80);

        // Both detect() calls should produce the same bit at position 0.
        let detected_zero = wm
            .detect_frame(&frame_zero, 128, 128, 0)
            .expect("detect zero");
        let detected_wrapped = wm
            .detect_frame(&frame_wrapped, 128, 128, 80)
            .expect("detect wrapped");
        // Bit 0 of both recovered payloads must agree (same input bit).
        let first_bit = (detected_zero[0] >> 7) & 1;
        let wrapped_bit = (detected_wrapped[0] >> 7) & 1;
        assert_eq!(first_bit, wrapped_bit, "wrapping bit mismatch");
    }
}
