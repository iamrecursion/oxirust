// Copyright 2024 The OxiMedia Project Developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Theora two-pass encoding support.
//!
//! Two-pass encoding improves quality at a given bitrate by first analysing
//! the entire stream to collect per-frame complexity statistics, then in the
//! second pass allocating bits according to those statistics.
//!
//! # Workflow
//!
//! ```ignore
//! use oximedia_codec::theora::two_pass::{
//!     TwoPassConfig, TheoraFirstPassAnalyzer, TheoraSecondPassEncoder,
//! };
//!
//! let config = TwoPassConfig::default();
//!
//! // --- First pass ---
//! let mut analyzer = TheoraFirstPassAnalyzer::new(config.clone());
//! for (idx, frame_bytes) in frames.iter().enumerate() {
//!     analyzer.analyze_frame(frame_bytes, width, height, idx as u64);
//! }
//! let raw_stats = analyzer.serialize_stats();
//!
//! // --- Second pass ---
//! let stats = TheoraFirstPassAnalyzer::deserialize_stats(&raw_stats)?;
//! let encoder = TheoraSecondPassEncoder::new(config, stats)?;
//! for idx in 0..frame_count {
//!     let is_key = idx % keyframe_interval == 0;
//!     let quality = encoder.get_frame_quality(idx as u64, is_key);
//!     // use quality in TheoraEncoder ...
//! }
//! ```

use crate::error::{CodecError, CodecResult};

// ─────────────────────────────────────────────────────────────────────────────
// Serialization constants
// ─────────────────────────────────────────────────────────────────────────────

/// Magic bytes that prefix a serialised `TwoPassStats` file.
const STATS_MAGIC: &[u8; 4] = b"THRS";

/// Number of bytes needed to serialise one [`TwoPassStats`] entry.
/// Layout: frame_index(8) + dct_energy(8) + motion_magnitude(8) +
///         is_scene_cut(1) + frame_complexity(8) = 33 bytes.
const STATS_ENTRY_SIZE: usize = 33;

/// Size of the file header: magic(4) + count(4) = 8 bytes.
const STATS_HEADER_SIZE: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration shared between the first-pass analyser and second-pass encoder.
#[derive(Debug, Clone)]
pub struct TwoPassConfig {
    /// Target average bitrate in bits per second.
    pub target_bitrate: u64,
    /// Frame rate in frames per second.
    pub framerate: f64,
    /// Distance between key frames (I frames).
    pub keyframe_interval: u32,
    /// Minimum quality value (0 = highest quality, 63 = lowest).
    ///
    /// In Theora the quality scale runs 0–63 where **higher** means *better*.
    /// `quality_min` therefore represents the floor (worst allowed quality).
    pub quality_min: u8,
    /// Maximum quality value (best allowed quality, ≤ 63).
    pub quality_max: u8,
}

impl TwoPassConfig {
    /// Create a `TwoPassConfig` with sensible defaults.
    #[must_use]
    pub fn new(target_bitrate: u64, framerate: f64) -> Self {
        Self {
            target_bitrate,
            framerate,
            keyframe_interval: 64,
            quality_min: 16,
            quality_max: 56,
        }
    }
}

impl Default for TwoPassConfig {
    fn default() -> Self {
        Self::new(2_000_000, 30.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-frame statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-frame statistics collected during the first pass.
///
/// This structure is serialisable to a compact binary format so that the
/// stats can be stored to a file and read back for the second pass.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoPassStats {
    /// Zero-based index of this frame in the stream.
    pub frame_index: u64,
    /// Approximate DCT energy: sum of squared DCT coefficients across all
    /// luma 8×8 blocks (computed via per-block pixel variance).
    pub dct_energy: f64,
    /// Mean absolute pixel difference between this frame and the previous one
    /// (zero for the very first frame).
    pub motion_magnitude: f64,
    /// Whether a scene cut was detected before this frame.
    pub is_scene_cut: bool,
    /// Combined complexity metric derived from `dct_energy` and
    /// `motion_magnitude`.
    pub frame_complexity: f64,
}

impl TwoPassStats {
    /// Serialise this entry to exactly [`STATS_ENTRY_SIZE`] bytes.
    fn to_bytes(&self) -> [u8; STATS_ENTRY_SIZE] {
        let mut buf = [0u8; STATS_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.frame_index.to_le_bytes());
        buf[8..16].copy_from_slice(&self.dct_energy.to_le_bytes());
        buf[16..24].copy_from_slice(&self.motion_magnitude.to_le_bytes());
        buf[24] = if self.is_scene_cut { 1 } else { 0 };
        buf[25..33].copy_from_slice(&self.frame_complexity.to_le_bytes());
        buf
    }

    /// Deserialise one entry from a 33-byte slice.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidBitstream` if `src` is shorter than
    /// [`STATS_ENTRY_SIZE`].
    fn from_bytes(src: &[u8]) -> CodecResult<Self> {
        if src.len() < STATS_ENTRY_SIZE {
            return Err(CodecError::InvalidBitstream(format!(
                "TwoPassStats entry too short: {} < {}",
                src.len(),
                STATS_ENTRY_SIZE
            )));
        }

        let frame_index =
            u64::from_le_bytes(src[0..8].try_into().map_err(|_| {
                CodecError::InvalidBitstream("frame_index slice error".to_string())
            })?);
        let dct_energy = f64::from_le_bytes(
            src[8..16]
                .try_into()
                .map_err(|_| CodecError::InvalidBitstream("dct_energy slice error".to_string()))?,
        );
        let motion_magnitude = f64::from_le_bytes(src[16..24].try_into().map_err(|_| {
            CodecError::InvalidBitstream("motion_magnitude slice error".to_string())
        })?);
        let is_scene_cut = src[24] != 0;
        let frame_complexity = f64::from_le_bytes(src[25..33].try_into().map_err(|_| {
            CodecError::InvalidBitstream("frame_complexity slice error".to_string())
        })?);

        Ok(Self {
            frame_index,
            dct_energy,
            motion_magnitude,
            is_scene_cut,
            frame_complexity,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// First-pass analyser
// ─────────────────────────────────────────────────────────────────────────────

/// First-pass analyser: scans every frame and collects [`TwoPassStats`].
///
/// Feed every frame in order via [`Self::analyze_frame`], then retrieve the
/// statistics with [`Self::collect_stats`] or [`Self::serialize_stats`].
pub struct TheoraFirstPassAnalyzer {
    config: TwoPassConfig,
    stats: Vec<TwoPassStats>,
    /// Luma data of the previous frame for motion estimation.
    prev_luma: Option<Vec<u8>>,
    /// Variance of the previous frame's luma, used for scene-cut detection.
    prev_variance: f64,
}

impl TheoraFirstPassAnalyzer {
    /// Create a new first-pass analyser with the given configuration.
    #[must_use]
    pub fn new(config: TwoPassConfig) -> Self {
        Self {
            config,
            stats: Vec::new(),
            prev_luma: None,
            prev_variance: 0.0,
        }
    }

    /// Analyse a single luma frame and append its statistics.
    ///
    /// `y_plane` must contain `width × height` bytes in row-major order.
    /// Frames should be submitted in display order starting from index 0.
    ///
    /// # Returns
    ///
    /// A reference to the newly appended [`TwoPassStats`] entry.
    pub fn analyze_frame(
        &mut self,
        y_plane: &[u8],
        width: u32,
        height: u32,
        frame_idx: u64,
    ) -> TwoPassStats {
        let dct_energy = compute_dct_energy(y_plane, width, height);
        let curr_variance = compute_frame_variance(y_plane, width, height);

        let (motion_magnitude, is_scene_cut) = match &self.prev_luma {
            Some(prev) => {
                let mad = compute_mad(prev, y_plane);
                let scene_cut = is_scene_cut(mad, self.prev_variance, curr_variance);
                (mad, scene_cut)
            }
            None => (0.0, false),
        };

        // Combine energy and motion into a single complexity score.
        // Normalise roughly so that typical values fall in [0, ~1000].
        let frame_complexity = dct_energy * 0.6 + motion_magnitude * 40.0;

        let entry = TwoPassStats {
            frame_index: frame_idx,
            dct_energy,
            motion_magnitude,
            is_scene_cut,
            frame_complexity,
        };

        // Save luma copy for next frame.
        let luma_len = (width as usize).saturating_mul(height as usize);
        let copy_len = luma_len.min(y_plane.len());
        let mut luma_copy = vec![128u8; luma_len];
        luma_copy[..copy_len].copy_from_slice(&y_plane[..copy_len]);
        self.prev_luma = Some(luma_copy);
        self.prev_variance = curr_variance;

        self.stats.push(entry.clone());
        entry
    }

    /// Return a slice of all collected statistics in order.
    #[must_use]
    pub fn collect_stats(&self) -> &[TwoPassStats] {
        &self.stats
    }

    /// Serialise all collected statistics to a byte vector.
    ///
    /// Format:
    /// - 4 bytes magic `b"THRS"`
    /// - 4 bytes count (u32 LE)
    /// - N × 33 bytes per [`TwoPassStats`] entry
    #[must_use]
    pub fn serialize_stats(&self) -> Vec<u8> {
        serialize_stats_slice(&self.stats)
    }

    /// Deserialise statistics from a byte slice produced by [`Self::serialize_stats`].
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidBitstream` if:
    /// - The magic bytes are wrong.
    /// - The buffer is too short for the declared entry count.
    /// - Any individual entry fails to deserialise.
    pub fn deserialize_stats(data: &[u8]) -> CodecResult<Vec<TwoPassStats>> {
        deserialize_stats_impl(data)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Second-pass encoder
// ─────────────────────────────────────────────────────────────────────────────

/// Second-pass encoder: uses first-pass statistics to assign per-frame quality.
///
/// Bit allocation follows a weighted VBR formula:
///
/// ```text
/// total_bits  = target_bitrate / framerate × N
/// weight[i]   = complexity[i] / avg_complexity
///              × scene_cut_factor   (×2.0 for scene cuts)
///              × keyframe_factor    (×1.5 for key frames)
/// bits[i]     = total_bits × weight[i] / Σweight
/// quality[i]  = map(bits[i] → quality range)
/// ```
///
/// Higher allocated bits correspond to lower quality values (better quality in
/// Theora's 0 = worst / 63 = best scale).
pub struct TheoraSecondPassEncoder {
    config: TwoPassConfig,
    stats: Vec<TwoPassStats>,
    /// Pre-computed per-frame bit allocations.
    frame_bits: Vec<u32>,
    /// Bits per frame at target bitrate (used as the normalisation baseline).
    base_bits_per_frame: f64,
}

impl TheoraSecondPassEncoder {
    /// Create a second-pass encoder from first-pass statistics.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if `stats` is empty or the
    /// configuration is inconsistent.
    pub fn new(config: TwoPassConfig, stats: Vec<TwoPassStats>) -> CodecResult<Self> {
        if stats.is_empty() {
            return Err(CodecError::InvalidParameter(
                "TwoPassStats must not be empty for second-pass encoding".to_string(),
            ));
        }
        if config.framerate <= 0.0 {
            return Err(CodecError::InvalidParameter(
                "framerate must be positive".to_string(),
            ));
        }
        if config.quality_min > config.quality_max {
            return Err(CodecError::InvalidParameter(
                "quality_min must not exceed quality_max".to_string(),
            ));
        }

        let base_bits_per_frame = config.target_bitrate as f64 / config.framerate;
        let frame_bits = allocate_bits(&config, &stats, base_bits_per_frame);

        Ok(Self {
            config,
            stats,
            frame_bits,
            base_bits_per_frame,
        })
    }

    /// Return the Theora quality value (0–63) to use for frame `frame_idx`.
    ///
    /// Frames that are not present in the stats default to `quality_max`.
    #[must_use]
    pub fn get_frame_quality(&self, frame_idx: u64, _is_keyframe: bool) -> u8 {
        let idx = frame_idx as usize;
        if idx >= self.frame_bits.len() {
            return self.config.quality_max;
        }
        bits_to_quality(
            self.frame_bits[idx],
            self.base_bits_per_frame,
            self.config.quality_min,
            self.config.quality_max,
        )
    }

    /// Return the allocated bit budget for frame `frame_idx`.
    ///
    /// Defaults to `base_bits_per_frame` rounded to u32 for out-of-range indices.
    #[must_use]
    pub fn allocate_bits(&self, frame_idx: u64, _is_keyframe: bool) -> u32 {
        let idx = frame_idx as usize;
        if idx < self.frame_bits.len() {
            self.frame_bits[idx]
        } else {
            self.base_bits_per_frame as u32
        }
    }

    /// Return the number of frames for which statistics were provided.
    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.stats.len()
    }

    /// Return a reference to the per-frame statistics.
    #[must_use]
    pub fn stats(&self) -> &[TwoPassStats] {
        &self.stats
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions (module-private)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute approximate DCT energy for a luma plane via per-8×8-block variance.
///
/// We approximate the sum of squared DCT AC coefficients by the spatial
/// variance of each block: `variance ≈ E[x²] − E[x]²`.
fn compute_dct_energy(y_plane: &[u8], width: u32, height: u32) -> f64 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let w = width as usize;
    let h = height as usize;
    let stride = w;

    let mut total_energy = 0.0f64;
    let block = 8usize;

    let bx_count = w.div_ceil(block);
    let by_count = h.div_ceil(block);

    for by in 0..by_count {
        for bx in 0..bx_count {
            let x0 = bx * block;
            let y0 = by * block;
            let x1 = (x0 + block).min(w);
            let y1 = (y0 + block).min(h);

            let mut sum = 0u64;
            let mut sum_sq = 0u64;
            let mut count = 0u64;

            for row in y0..y1 {
                for col in x0..x1 {
                    let off = row * stride + col;
                    if off < y_plane.len() {
                        let v = u64::from(y_plane[off]);
                        sum += v;
                        sum_sq += v * v;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                // variance = E[x²] − E[x]²
                // Compute in f64 to avoid u64 overflow for blocks with large pixel values.
                let mean = sum as f64 / count as f64;
                let mean_sq = mean * mean;
                let ex2 = sum_sq as f64 / count as f64;
                let variance = (ex2 - mean_sq).max(0.0);
                total_energy += variance;
            }
        }
    }

    total_energy
}

/// Compute the overall luma variance of an entire frame (used for scene-cut
/// detection).
fn compute_frame_variance(y_plane: &[u8], width: u32, height: u32) -> f64 {
    if width == 0 || height == 0 || y_plane.is_empty() {
        return 0.0;
    }
    let n = y_plane.len() as u64;
    let sum: u64 = y_plane.iter().map(|&b| u64::from(b)).sum();
    let sum_sq: u64 = y_plane.iter().map(|&b| u64::from(b) * u64::from(b)).sum();
    let mean_sq = (sum * sum) / n;
    sum_sq.saturating_sub(mean_sq / n) as f64
}

/// Compute mean absolute difference between two luma planes of equal length.
fn compute_mad(prev: &[u8], curr: &[u8]) -> f64 {
    let len = prev.len().min(curr.len());
    if len == 0 {
        return 0.0;
    }
    let total: u64 = prev[..len]
        .iter()
        .zip(curr[..len].iter())
        .map(|(&a, &b)| {
            let diff = (i32::from(a) - i32::from(b)).unsigned_abs();
            u64::from(diff)
        })
        .sum();
    total as f64 / len as f64
}

/// Determine whether a scene cut occurred between two frames.
///
/// A cut is declared when:
/// - The mean absolute difference exceeds 30.0, **or**
/// - The ratio of current to previous frame variance exceeds 2.5.
fn is_scene_cut(mad: f64, prev_variance: f64, curr_variance: f64) -> bool {
    if mad > 30.0 {
        return true;
    }
    if prev_variance > 0.0 {
        let ratio = curr_variance / prev_variance;
        if ratio > 2.5 {
            return true;
        }
    }
    false
}

/// Allocate bits across all frames using a weighted VBR formula.
fn allocate_bits(config: &TwoPassConfig, stats: &[TwoPassStats], base_bpf: f64) -> Vec<u32> {
    let n = stats.len();
    if n == 0 {
        return Vec::new();
    }

    // Average complexity across the whole stream.
    let total_complexity: f64 = stats.iter().map(|s| s.frame_complexity).sum();
    let avg_complexity = if total_complexity > 0.0 {
        total_complexity / n as f64
    } else {
        1.0
    };

    // Total available bits for the whole clip.
    let total_bits = base_bpf * n as f64;

    // Build per-frame weights.
    let weights: Vec<f64> = stats
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let base_weight = if avg_complexity > 0.0 {
                s.frame_complexity / avg_complexity
            } else {
                1.0
            };
            // Boost scene cuts and key frames.
            let scene_factor = if s.is_scene_cut { 2.0f64 } else { 1.0 };
            let key_factor = if i % config.keyframe_interval as usize == 0 {
                1.5f64
            } else {
                1.0
            };
            (base_weight * scene_factor * key_factor).max(0.1)
        })
        .collect();

    let weight_sum: f64 = weights.iter().sum();

    weights
        .iter()
        .map(|&w| {
            let bits = if weight_sum > 0.0 {
                total_bits * w / weight_sum
            } else {
                base_bpf
            };
            bits.max(1.0) as u32
        })
        .collect()
}

/// Map an allocated bit budget for a frame to a Theora quality value.
///
/// More bits → lower quality index (better in Theora's scale where 63 = best).
///
/// The mapping is linear: `base_bpf` bits → midpoint quality, up to ±half-range.
fn bits_to_quality(bits: u32, base_bpf: f64, quality_min: u8, quality_max: u8) -> u8 {
    let q_range = quality_max as f64 - quality_min as f64;
    let q_mid = quality_min as f64 + q_range * 0.5;

    if base_bpf <= 0.0 {
        return quality_min;
    }

    // Ratio > 1.0 means more bits than average → higher quality (higher index).
    let ratio = bits as f64 / base_bpf;

    // Scale: ratio 2.0 → quality_max, ratio 0.5 → quality_min.
    let scaled = (ratio - 0.5) / 1.5; // maps [0.5, 2.0] → [0, 1]
    let quality = quality_min as f64 + scaled.clamp(0.0, 1.0) * q_range;

    quality.clamp(quality_min as f64, quality_max as f64) as u8
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialisation helpers (pub(crate) for reuse in tests)
// ─────────────────────────────────────────────────────────────────────────────

fn serialize_stats_slice(stats: &[TwoPassStats]) -> Vec<u8> {
    let count = stats.len() as u32;
    let mut buf = Vec::with_capacity(STATS_HEADER_SIZE + stats.len() * STATS_ENTRY_SIZE);
    buf.extend_from_slice(STATS_MAGIC);
    buf.extend_from_slice(&count.to_le_bytes());
    for s in stats {
        buf.extend_from_slice(&s.to_bytes());
    }
    buf
}

fn deserialize_stats_impl(data: &[u8]) -> CodecResult<Vec<TwoPassStats>> {
    if data.len() < STATS_HEADER_SIZE {
        return Err(CodecError::InvalidBitstream(format!(
            "stats buffer too short for header: {} bytes",
            data.len()
        )));
    }

    if &data[0..4] != STATS_MAGIC {
        return Err(CodecError::InvalidBitstream(
            "invalid TwoPassStats magic bytes".to_string(),
        ));
    }

    let count = u32::from_le_bytes(
        data[4..8]
            .try_into()
            .map_err(|_| CodecError::InvalidBitstream("count slice error".to_string()))?,
    ) as usize;

    let expected_len = STATS_HEADER_SIZE + count * STATS_ENTRY_SIZE;
    if data.len() < expected_len {
        return Err(CodecError::InvalidBitstream(format!(
            "stats buffer too short: have {}, need {} for {} entries",
            data.len(),
            expected_len,
            count
        )));
    }

    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let start = STATS_HEADER_SIZE + i * STATS_ENTRY_SIZE;
        let end = start + STATS_ENTRY_SIZE;
        let entry = TwoPassStats::from_bytes(&data[start..end])?;
        result.push(entry);
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn uniform_plane(width: u32, height: u32, value: u8) -> Vec<u8> {
        vec![value; (width * height) as usize]
    }

    fn checker_plane(width: u32, height: u32) -> Vec<u8> {
        let mut v = vec![0u8; (width * height) as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                v[y * width as usize + x] = if (x + y) % 2 == 0 { 200 } else { 50 };
            }
        }
        v
    }

    // ── TwoPassConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_two_pass_config_defaults() {
        let cfg = TwoPassConfig::default();
        assert_eq!(cfg.target_bitrate, 2_000_000);
        assert!((cfg.framerate - 30.0).abs() < 1e-9);
        assert_eq!(cfg.keyframe_interval, 64);
        assert!(cfg.quality_min < cfg.quality_max);
    }

    #[test]
    fn test_two_pass_config_new() {
        let cfg = TwoPassConfig::new(4_000_000, 60.0);
        assert_eq!(cfg.target_bitrate, 4_000_000);
        assert!((cfg.framerate - 60.0).abs() < 1e-9);
    }

    // ── TheoraFirstPassAnalyzer ───────────────────────────────────────────────

    #[test]
    fn test_first_pass_analyzer_new() {
        let analyzer = TheoraFirstPassAnalyzer::new(TwoPassConfig::default());
        assert!(analyzer.collect_stats().is_empty());
    }

    #[test]
    fn test_analyze_frame_uniform_plane_zero_energy() {
        let mut analyzer = TheoraFirstPassAnalyzer::new(TwoPassConfig::default());
        let plane = uniform_plane(64, 64, 128);
        let stats = analyzer.analyze_frame(&plane, 64, 64, 0);
        // A completely uniform plane has zero block variance → zero DCT energy.
        assert!(
            stats.dct_energy < 1.0,
            "uniform plane should have near-zero DCT energy, got {}",
            stats.dct_energy
        );
    }

    #[test]
    fn test_analyze_frame_varied_plane_nonzero_energy() {
        let mut analyzer = TheoraFirstPassAnalyzer::new(TwoPassConfig::default());
        let plane = checker_plane(64, 64);
        let stats = analyzer.analyze_frame(&plane, 64, 64, 0);
        assert!(
            stats.dct_energy > 0.0,
            "checkered plane must have non-zero DCT energy"
        );
    }

    #[test]
    fn test_analyze_frame_first_frame_zero_motion() {
        let mut analyzer = TheoraFirstPassAnalyzer::new(TwoPassConfig::default());
        let plane = checker_plane(32, 32);
        let stats = analyzer.analyze_frame(&plane, 32, 32, 0);
        assert_eq!(
            stats.motion_magnitude, 0.0,
            "first frame has no prior frame"
        );
        assert!(!stats.is_scene_cut, "first frame cannot be a scene cut");
    }

    #[test]
    fn test_analyze_frame_motion_accumulates() {
        let mut analyzer = TheoraFirstPassAnalyzer::new(TwoPassConfig::default());
        let plane_a = uniform_plane(32, 32, 10);
        let plane_b = uniform_plane(32, 32, 200);
        analyzer.analyze_frame(&plane_a, 32, 32, 0);
        let stats = analyzer.analyze_frame(&plane_b, 32, 32, 1);
        assert!(
            stats.motion_magnitude > 0.0,
            "motion must be positive when frames differ"
        );
        assert!(
            stats.is_scene_cut,
            "large pixel shift should trigger scene cut"
        );
    }

    #[test]
    fn test_scene_cut_detection_high_mad() {
        // MAD > 30 → scene cut
        let result = is_scene_cut(35.0, 100.0, 100.0);
        assert!(result);
    }

    #[test]
    fn test_scene_cut_detection_high_variance_ratio() {
        // ratio > 2.5 → scene cut
        let result = is_scene_cut(5.0, 100.0, 300.0);
        assert!(result);
    }

    #[test]
    fn test_scene_cut_detection_no_cut() {
        // Neither condition met
        let result = is_scene_cut(10.0, 100.0, 120.0);
        assert!(!result);
    }

    #[test]
    fn test_collect_stats_length() {
        let mut analyzer = TheoraFirstPassAnalyzer::new(TwoPassConfig::default());
        for i in 0..10u64 {
            let plane = uniform_plane(16, 16, (i * 25 % 255) as u8);
            analyzer.analyze_frame(&plane, 16, 16, i);
        }
        assert_eq!(analyzer.collect_stats().len(), 10);
    }

    // ── Serialisation ─────────────────────────────────────────────────────────

    #[test]
    fn test_serialize_empty_stats() {
        let analyzer = TheoraFirstPassAnalyzer::new(TwoPassConfig::default());
        let bytes = analyzer.serialize_stats();
        assert_eq!(bytes.len(), STATS_HEADER_SIZE);
        assert_eq!(&bytes[0..4], STATS_MAGIC);
        let count = u32::from_le_bytes(bytes[4..8].try_into().expect("slice"));
        assert_eq!(count, 0);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut analyzer = TheoraFirstPassAnalyzer::new(TwoPassConfig::default());
        let plane_a = uniform_plane(16, 16, 80);
        let plane_b = checker_plane(16, 16);
        let plane_c = uniform_plane(16, 16, 200);
        analyzer.analyze_frame(&plane_a, 16, 16, 0);
        analyzer.analyze_frame(&plane_b, 16, 16, 1);
        analyzer.analyze_frame(&plane_c, 16, 16, 2);

        let bytes = analyzer.serialize_stats();
        let restored =
            TheoraFirstPassAnalyzer::deserialize_stats(&bytes).expect("deserialize should succeed");

        let original = analyzer.collect_stats();
        assert_eq!(restored.len(), original.len());
        for (r, o) in restored.iter().zip(original.iter()) {
            assert_eq!(r.frame_index, o.frame_index);
            assert!((r.dct_energy - o.dct_energy).abs() < 1e-6);
            assert!((r.motion_magnitude - o.motion_magnitude).abs() < 1e-6);
            assert_eq!(r.is_scene_cut, o.is_scene_cut);
            assert!((r.frame_complexity - o.frame_complexity).abs() < 1e-6);
        }
    }

    #[test]
    fn test_deserialize_invalid_magic() {
        let bad: Vec<u8> = b"BADM\x01\x00\x00\x00".to_vec();
        let result = TheoraFirstPassAnalyzer::deserialize_stats(&bad);
        assert!(result.is_err(), "bad magic should return an error");
    }

    #[test]
    fn test_deserialize_too_short() {
        let short: Vec<u8> = vec![0u8; 3];
        let result = TheoraFirstPassAnalyzer::deserialize_stats(&short);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialize_truncated_entries() {
        // Header says 2 entries but buffer only has room for 1.
        let mut buf = Vec::new();
        buf.extend_from_slice(STATS_MAGIC);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; STATS_ENTRY_SIZE]); // only 1 entry
        let result = TheoraFirstPassAnalyzer::deserialize_stats(&buf);
        assert!(result.is_err());
    }

    // ── TheoraSecondPassEncoder ───────────────────────────────────────────────

    fn make_stats(n: usize) -> Vec<TwoPassStats> {
        (0..n)
            .map(|i| TwoPassStats {
                frame_index: i as u64,
                dct_energy: 100.0 + i as f64 * 10.0,
                motion_magnitude: 5.0,
                is_scene_cut: i == 15,
                frame_complexity: 100.0 + i as f64 * 10.0,
            })
            .collect()
    }

    #[test]
    fn test_second_pass_encoder_new() {
        let cfg = TwoPassConfig::default();
        let stats = make_stats(30);
        let enc = TheoraSecondPassEncoder::new(cfg, stats);
        assert!(enc.is_ok());
    }

    #[test]
    fn test_second_pass_encoder_empty_stats_errors() {
        let cfg = TwoPassConfig::default();
        let result = TheoraSecondPassEncoder::new(cfg, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_second_pass_encoder_invalid_framerate_errors() {
        let mut cfg = TwoPassConfig::default();
        cfg.framerate = 0.0;
        let result = TheoraSecondPassEncoder::new(cfg, make_stats(5));
        assert!(result.is_err());
    }

    #[test]
    fn test_second_pass_get_frame_quality_range() {
        let cfg = TwoPassConfig::default();
        let stats = make_stats(30);
        let enc = TheoraSecondPassEncoder::new(cfg.clone(), stats).expect("ok");
        for i in 0..30u64 {
            let q = enc.get_frame_quality(i, i % cfg.keyframe_interval as u64 == 0);
            assert!(
                q >= cfg.quality_min && q <= cfg.quality_max,
                "quality {} out of range [{}, {}] at frame {}",
                q,
                cfg.quality_min,
                cfg.quality_max,
                i
            );
        }
    }

    #[test]
    fn test_second_pass_allocate_bits_keyframe_higher() {
        let cfg = TwoPassConfig {
            target_bitrate: 2_000_000,
            framerate: 30.0,
            keyframe_interval: 10,
            quality_min: 16,
            quality_max: 56,
        };
        // All frames have equal complexity so keyframe boost should dominate.
        let stats: Vec<TwoPassStats> = (0..20)
            .map(|i| TwoPassStats {
                frame_index: i as u64,
                dct_energy: 100.0,
                motion_magnitude: 5.0,
                is_scene_cut: false,
                frame_complexity: 100.0,
            })
            .collect();
        let enc = TheoraSecondPassEncoder::new(cfg.clone(), stats).expect("ok");
        let key_bits = enc.allocate_bits(0, true);
        let inter_bits = enc.allocate_bits(1, false);
        assert!(
            key_bits > inter_bits,
            "keyframe should receive more bits: {} vs {}",
            key_bits,
            inter_bits
        );
    }

    #[test]
    fn test_second_pass_total_frames() {
        let stats = make_stats(42);
        let enc = TheoraSecondPassEncoder::new(TwoPassConfig::default(), stats).expect("ok");
        assert_eq!(enc.total_frames(), 42);
    }

    #[test]
    fn test_second_pass_out_of_range_frame() {
        let cfg = TwoPassConfig::default();
        let enc = TheoraSecondPassEncoder::new(cfg.clone(), make_stats(5)).expect("ok");
        // Frame 999 is beyond stats; should return quality_max.
        let q = enc.get_frame_quality(999, false);
        assert_eq!(q, cfg.quality_max);
    }
}
