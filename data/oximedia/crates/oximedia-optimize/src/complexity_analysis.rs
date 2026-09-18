//! Content complexity analysis for encoder decisions.
//!
//! Provides spatial and temporal complexity analysis to guide encoder
//! mode decisions, bitrate allocation, and preset selection.

#![allow(dead_code)]

/// Spatial complexity of a frame.
#[derive(Debug, Clone)]
pub struct SpatialComplexity {
    /// Sum of squared AC DCT coefficients (energy in high-freq content).
    pub dct_energy: f32,
    /// Fraction of pixels classified as edge pixels via Sobel filter.
    pub edge_density: f32,
    /// Average variance of pixel values within 16x16 blocks.
    pub texture_variance: f32,
    /// Combined complexity score in [0, 1].
    pub complexity_score: f32,
}

/// Analyzes the spatial complexity of a single frame.
pub struct SpatialComplexityAnalyzer;

impl SpatialComplexityAnalyzer {
    /// Analyze spatial complexity of a frame.
    ///
    /// `frame` is a flat f32 array of luma samples, `width` × `height`.
    #[must_use]
    pub fn analyze(frame: &[f32], width: u32, height: u32) -> SpatialComplexity {
        let dct_energy = Self::compute_dct_energy(frame, width, height);
        let edge_density = Self::compute_edge_density(frame, width, height);
        let texture_variance = Self::compute_texture_variance(frame, width, height);

        // Combine into a single score. Each component is normalised then weighted.
        let energy_norm = (dct_energy / 1_000_000.0).min(1.0);
        let complexity_score =
            (energy_norm * 0.4 + edge_density * 0.3 + (texture_variance / 10_000.0).min(1.0) * 0.3)
                .min(1.0);

        SpatialComplexity {
            dct_energy,
            edge_density,
            texture_variance,
            complexity_score,
        }
    }

    /// Compute 8×8 DCT energy: sum of squared AC coefficients over all blocks.
    fn compute_dct_energy(frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;
        if w < 8 || h < 8 {
            return 0.0;
        }

        let mut total_energy = 0.0f32;
        let blocks_x = w / 8;
        let blocks_y = h / 8;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut block = [0.0f32; 64];
                for row in 0..8 {
                    for col in 0..8 {
                        let px = bx * 8 + col;
                        let py = by * 8 + row;
                        block[row * 8 + col] = frame[py * w + px];
                    }
                }

                // 1-D DCT-II approximation per row then per column
                let dct = Self::dct8x8(&block);

                // Sum squared AC coefficients (skip DC at [0][0])
                for k in 0..64usize {
                    if k == 0 {
                        continue;
                    }
                    total_energy += dct[k] * dct[k];
                }
            }
        }

        total_energy
    }

    /// Very fast approximate 8×8 DCT (row/column separable via the AAN algorithm coefficients).
    fn dct8x8(block: &[f32; 64]) -> [f32; 64] {
        // Use a simple loop-based DCT for correctness without SIMD
        let mut out = [0.0f32; 64];
        let pi = std::f32::consts::PI;

        for u in 0..8usize {
            for v in 0..8usize {
                let cu = if u == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                let cv = if v == 0 { 1.0 / 2.0f32.sqrt() } else { 1.0 };
                let mut sum = 0.0f32;
                for x in 0..8usize {
                    for y in 0..8usize {
                        sum += block[y * 8 + x]
                            * ((2 * x + 1) as f32 * u as f32 * pi / 16.0).cos()
                            * ((2 * y + 1) as f32 * v as f32 * pi / 16.0).cos();
                    }
                }
                out[u * 8 + v] = 0.25 * cu * cv * sum;
            }
        }
        out
    }

    /// Compute edge density via Sobel filter + threshold.
    fn compute_edge_density(frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;
        if w < 3 || h < 3 {
            return 0.0;
        }

        let threshold = 30.0f32;
        let mut edge_pixels = 0u32;
        let total = ((w - 2) * (h - 2)) as f32;

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                // Sobel X
                let gx = -frame[(y - 1) * w + (x - 1)]
                    - 2.0 * frame[y * w + (x - 1)]
                    - frame[(y + 1) * w + (x - 1)]
                    + frame[(y - 1) * w + (x + 1)]
                    + 2.0 * frame[y * w + (x + 1)]
                    + frame[(y + 1) * w + (x + 1)];
                // Sobel Y
                let gy = -frame[(y - 1) * w + (x - 1)]
                    - 2.0 * frame[(y - 1) * w + x]
                    - frame[(y - 1) * w + (x + 1)]
                    + frame[(y + 1) * w + (x - 1)]
                    + 2.0 * frame[(y + 1) * w + x]
                    + frame[(y + 1) * w + (x + 1)];
                let mag = (gx * gx + gy * gy).sqrt();
                if mag > threshold {
                    edge_pixels += 1;
                }
            }
        }

        edge_pixels as f32 / total
    }

    /// Compute average pixel variance within 16×16 blocks.
    fn compute_texture_variance(frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;
        let bs = 16usize;
        if w < bs || h < bs {
            return 0.0;
        }

        let blocks_x = w / bs;
        let blocks_y = h / bs;
        let n = (bs * bs) as f32;
        let mut total_var = 0.0f32;
        let block_count = (blocks_x * blocks_y) as f32;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;
                for row in 0..bs {
                    for col in 0..bs {
                        let v = frame[(by * bs + row) * w + bx * bs + col];
                        sum += v;
                        sum_sq += v * v;
                    }
                }
                let mean = sum / n;
                let var = sum_sq / n - mean * mean;
                total_var += var.max(0.0);
            }
        }

        if block_count > 0.0 {
            total_var / block_count
        } else {
            0.0
        }
    }
}

/// Temporal complexity between consecutive frames.
#[derive(Debug, Clone)]
pub struct TemporalComplexity {
    /// Average magnitude of motion vectors (in pixels).
    pub motion_magnitude: f32,
    /// Scene-change score in [0, 1]; values >0.5 suggest a scene cut.
    pub scene_change_score: f32,
    /// Number of non-zero motion vectors found.
    pub motion_vectors_count: u32,
}

/// Analyzes temporal complexity between two frames.
pub struct TemporalComplexityAnalyzer;

impl TemporalComplexityAnalyzer {
    /// Analyze temporal complexity.
    ///
    /// `prev` and `curr` are flat f32 luma arrays of the same `width` × height.
    #[must_use]
    pub fn analyze(prev: &[f32], curr: &[f32], width: u32) -> TemporalComplexity {
        let (magnitudes, sad_sum) = Self::block_match(prev, curr, width);

        let motion_vectors_count = magnitudes.iter().filter(|&&m| m > 0.5).count() as u32;
        let motion_magnitude = if magnitudes.is_empty() {
            0.0
        } else {
            magnitudes.iter().sum::<f32>() / magnitudes.len() as f32
        };

        // Scene change: high average MAD across the whole frame → scene cut
        let total_pixels = prev.len() as f32;
        let scene_change_score = if total_pixels > 0.0 {
            (sad_sum / total_pixels / 255.0).min(1.0)
        } else {
            0.0
        };

        TemporalComplexity {
            motion_magnitude,
            scene_change_score,
            motion_vectors_count,
        }
    }

    /// 16×16 block matching via Mean Absolute Difference (MAD).
    ///
    /// Returns a Vec of per-block motion magnitudes and the total SAD.
    fn block_match(prev: &[f32], curr: &[f32], width: u32) -> (Vec<f32>, f32) {
        let w = width as usize;
        if w == 0 || prev.is_empty() || curr.is_empty() || prev.len() != curr.len() {
            return (Vec::new(), 0.0);
        }
        let h = prev.len() / w;
        let bs = 16usize;
        let search_range = 8isize;
        if w < bs || h < bs {
            return (Vec::new(), 0.0);
        }

        let blocks_x = w / bs;
        let blocks_y = h / bs;
        let mut magnitudes = Vec::with_capacity(blocks_x * blocks_y);
        let mut total_sad = 0.0f32;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let mut best_mx = 0isize;
                let mut best_my = 0isize;

                // Check zero-motion vector first; if MAD is 0 the block is
                // identical and we can skip the full search.
                let mut best_mad = {
                    let mut mad = 0.0f32;
                    for row in 0..bs {
                        for col in 0..bs {
                            let cy = by * bs + row;
                            let cx = bx * bs + col;
                            mad += (curr[cy * w + cx] - prev[cy * w + cx]).abs();
                        }
                    }
                    mad /= (bs * bs) as f32;
                    mad
                    // best_mx / best_my remain 0
                };
                if best_mad > 0.0 {
                    for dy in -search_range..=search_range {
                        for dx in -search_range..=search_range {
                            let ref_x = bx as isize * bs as isize + dx;
                            let ref_y = by as isize * bs as isize + dy;

                            if ref_x < 0
                                || ref_y < 0
                                || ref_x as usize + bs > w
                                || ref_y as usize + bs > h
                            {
                                continue;
                            }

                            let mut mad = 0.0f32;
                            for row in 0..bs {
                                for col in 0..bs {
                                    let cy = by * bs + row;
                                    let cx = bx * bs + col;
                                    let ry = ref_y as usize + row;
                                    let rx = ref_x as usize + col;
                                    mad += (curr[cy * w + cx] - prev[ry * w + rx]).abs();
                                }
                            }
                            mad /= (bs * bs) as f32;

                            if mad < best_mad {
                                best_mad = mad;
                                best_mx = dx;
                                best_my = dy;
                            }
                        }
                    }
                } // end if best_mad > 0.0

                let mag = ((best_mx * best_mx + best_my * best_my) as f32).sqrt();
                magnitudes.push(mag);
                total_sad += best_mad;
            }
        }

        (magnitudes, total_sad)
    }
}

/// Combined spatial + temporal complexity profile.
#[derive(Debug, Clone)]
pub struct ContentComplexityProfile {
    /// Spatial complexity of the current frame.
    pub spatial: SpatialComplexity,
    /// Temporal complexity relative to previous frame.
    pub temporal: TemporalComplexity,
    /// Overall complexity score in [0, 1].
    pub overall: f32,
}

impl ContentComplexityProfile {
    /// Create a new profile.
    #[must_use]
    pub fn new(spatial: SpatialComplexity, temporal: TemporalComplexity) -> Self {
        let temporal_score =
            (temporal.motion_magnitude / 16.0).min(1.0) * 0.5 + temporal.scene_change_score * 0.5;
        let overall = (spatial.complexity_score * 0.6 + temporal_score * 0.4).min(1.0);
        Self {
            spatial,
            temporal,
            overall,
        }
    }

    /// Classify this profile into an encoding difficulty level.
    #[must_use]
    pub fn encoding_difficulty(&self) -> EncodingDifficulty {
        match self.overall {
            s if s < 0.15 => EncodingDifficulty::VeryEasy,
            s if s < 0.35 => EncodingDifficulty::Easy,
            s if s < 0.55 => EncodingDifficulty::Medium,
            s if s < 0.75 => EncodingDifficulty::Hard,
            _ => EncodingDifficulty::VeryHard,
        }
    }
}

/// Qualitative encoding difficulty classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingDifficulty {
    /// Very simple content (solid colours, slides).
    VeryEasy,
    /// Simple content (low motion, flat backgrounds).
    Easy,
    /// Typical mixed content.
    Medium,
    /// Complex content (high motion, heavy texture).
    Hard,
    /// Extremely complex (rapid action, heavy grain, scene cuts).
    VeryHard,
}

impl EncodingDifficulty {
    /// Suggest an encoder preset name for this difficulty level.
    #[must_use]
    pub fn suggested_preset(&self) -> &str {
        match self {
            Self::VeryEasy => "ultrafast",
            Self::Easy => "superfast",
            Self::Medium => "medium",
            Self::Hard => "slow",
            Self::VeryHard => "veryslow",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(w: u32, h: u32, val: f32) -> Vec<f32> {
        vec![val; (w * h) as usize]
    }

    fn ramp_frame(w: u32, h: u32) -> Vec<f32> {
        let n = (w * h) as usize;
        (0..n).map(|i| (i % 256) as f32).collect()
    }

    #[test]
    fn test_spatial_flat_frame_low_energy() {
        let frame = flat_frame(64, 64, 128.0);
        let sc = SpatialComplexityAnalyzer::analyze(&frame, 64, 64);
        // Flat frame → nearly zero DCT AC energy
        assert!(sc.dct_energy < 1.0, "dct_energy={}", sc.dct_energy);
        assert!(sc.edge_density < 0.01);
        assert!(sc.complexity_score >= 0.0 && sc.complexity_score <= 1.0);
    }

    #[test]
    fn test_spatial_ramp_frame_has_energy() {
        let frame = ramp_frame(64, 64);
        let sc = SpatialComplexityAnalyzer::analyze(&frame, 64, 64);
        // Ramp has edges and DCT energy
        assert!(sc.dct_energy > 0.0);
        assert!(sc.edge_density > 0.0);
    }

    #[test]
    fn test_edge_density_range() {
        let frame = ramp_frame(32, 32);
        let sc = SpatialComplexityAnalyzer::analyze(&frame, 32, 32);
        assert!(sc.edge_density >= 0.0 && sc.edge_density <= 1.0);
    }

    #[test]
    fn test_texture_variance_flat() {
        let frame = flat_frame(32, 32, 200.0);
        let sc = SpatialComplexityAnalyzer::analyze(&frame, 32, 32);
        assert!(sc.texture_variance < 1.0);
    }

    #[test]
    fn test_temporal_identical_frames() {
        let frame = ramp_frame(32, 32);
        let tc = TemporalComplexityAnalyzer::analyze(&frame, &frame, 32);
        assert_eq!(tc.motion_magnitude, 0.0);
        assert_eq!(tc.motion_vectors_count, 0);
    }

    #[test]
    fn test_temporal_scene_change() {
        let prev = flat_frame(32, 32, 50.0);
        let curr = flat_frame(32, 32, 200.0);
        let tc = TemporalComplexityAnalyzer::analyze(&prev, &curr, 32);
        assert!(tc.scene_change_score > 0.0);
    }

    #[test]
    fn test_content_complexity_profile_new() {
        let spatial = SpatialComplexityAnalyzer::analyze(&ramp_frame(32, 32), 32, 32);
        let temporal = TemporalComplexityAnalyzer::analyze(
            &flat_frame(32, 32, 100.0),
            &ramp_frame(32, 32),
            32,
        );
        let profile = ContentComplexityProfile::new(spatial, temporal);
        assert!(profile.overall >= 0.0 && profile.overall <= 1.0);
    }

    #[test]
    fn test_encoding_difficulty_very_easy() {
        let d = EncodingDifficulty::VeryEasy;
        assert_eq!(d.suggested_preset(), "ultrafast");
    }

    #[test]
    fn test_encoding_difficulty_very_hard() {
        let d = EncodingDifficulty::VeryHard;
        assert_eq!(d.suggested_preset(), "veryslow");
    }

    #[test]
    fn test_encoding_difficulty_from_profile_flat() {
        // Flat frame, identical temporal → overall near 0 → VeryEasy
        let frame = flat_frame(32, 32, 128.0);
        let spatial = SpatialComplexityAnalyzer::analyze(&frame, 32, 32);
        let temporal = TemporalComplexityAnalyzer::analyze(&frame, &frame, 32);
        let profile = ContentComplexityProfile::new(spatial, temporal);
        assert_eq!(profile.encoding_difficulty(), EncodingDifficulty::VeryEasy);
    }

    #[test]
    fn test_complexity_score_range() {
        let frame = ramp_frame(64, 64);
        let sc = SpatialComplexityAnalyzer::analyze(&frame, 64, 64);
        assert!(sc.complexity_score >= 0.0);
        assert!(sc.complexity_score <= 1.0);
    }

    #[test]
    fn test_encoding_difficulty_all_variants() {
        for (overall, expected) in [
            (0.1f32, EncodingDifficulty::VeryEasy),
            (0.2, EncodingDifficulty::Easy),
            (0.45, EncodingDifficulty::Medium),
            (0.65, EncodingDifficulty::Hard),
            (0.9, EncodingDifficulty::VeryHard),
        ] {
            // Build a profile manually by adjusting `overall`
            let frame = flat_frame(32, 32, 128.0);
            let mut spatial = SpatialComplexityAnalyzer::analyze(&frame, 32, 32);
            spatial.complexity_score = overall;
            let temporal = TemporalComplexityAnalyzer::analyze(&frame, &frame, 32);
            let mut profile = ContentComplexityProfile::new(spatial, temporal);
            profile.overall = overall;
            assert_eq!(profile.encoding_difficulty(), expected, "overall={overall}");
        }
    }

    #[test]
    fn test_small_frame_graceful() {
        let frame = vec![128.0f32; 4];
        let sc = SpatialComplexityAnalyzer::analyze(&frame, 2, 2);
        assert_eq!(sc.dct_energy, 0.0);
    }
}
