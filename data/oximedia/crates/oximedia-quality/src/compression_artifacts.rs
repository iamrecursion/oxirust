//! Compression artifact measurement for video quality assessment.
//!
//! Provides detectors for blockiness (DCT grid artefacts), ringing (Gibbs
//! phenomenon near edges) and banding (false contour / colour banding).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ─── Artifact severity ────────────────────────────────────────────────────────

/// Severity classification for a compression artefact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArtifactSeverity {
    /// No perceptible artefact
    None,
    /// Barely perceptible
    Minor,
    /// Noticeable but not objectionable
    Moderate,
    /// Clearly objectionable
    Severe,
    /// Renders content unwatchable
    Extreme,
}

impl ArtifactSeverity {
    /// Maximum score that still falls within this severity band.
    #[must_use]
    pub fn max_acceptable_score(&self) -> f32 {
        match self {
            Self::None => 0.05,
            Self::Minor => 0.15,
            Self::Moderate => 0.35,
            Self::Severe => 0.60,
            Self::Extreme => f32::MAX,
        }
    }

    /// Classify a raw score into a severity.
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        if score <= Self::None.max_acceptable_score() {
            Self::None
        } else if score <= Self::Minor.max_acceptable_score() {
            Self::Minor
        } else if score <= Self::Moderate.max_acceptable_score() {
            Self::Moderate
        } else if score <= Self::Severe.max_acceptable_score() {
            Self::Severe
        } else {
            Self::Extreme
        }
    }
}

// ─── Blockiness ───────────────────────────────────────────────────────────────

/// Blockiness score for a single frame or a video sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockinessScore {
    /// 0.0 = no blocking, 1.0 = severe blocking
    pub score: f32,
    /// Frame index with the worst score, if available
    pub worst_frame: Option<u32>,
}

impl BlockinessScore {
    /// Classify the score into a severity band.
    #[must_use]
    pub fn severity(&self) -> ArtifactSeverity {
        ArtifactSeverity::from_score(self.score)
    }
}

/// Detector for DCT-grid blockiness.
pub struct BlockinessDetector;

impl BlockinessDetector {
    /// Compute a blockiness score for a single frame.
    ///
    /// The algorithm compares pixel differences *at* 8-pixel DCT block
    /// boundaries with typical *interior* differences.  A higher ratio of
    /// boundary-to-interior difference indicates stronger blocking.
    ///
    /// `frame` is a row-major slice of normalised luma samples (0.0–1.0).
    /// Returns a value in [0.0, 1.0].
    #[must_use]
    pub fn compute(frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;

        if w < 9 || h < 9 || frame.len() < w * h {
            return 0.0;
        }

        let mut boundary_diff = 0.0f64;
        let mut interior_diff = 0.0f64;
        let mut boundary_count = 0u64;
        let mut interior_count = 0u64;

        // Horizontal differences
        for row in 0..h {
            for col in 1..w {
                let diff = (frame[row * w + col] - frame[row * w + col - 1]).abs() as f64;
                if col % 8 == 0 {
                    boundary_diff += diff;
                    boundary_count += 1;
                } else {
                    interior_diff += diff;
                    interior_count += 1;
                }
            }
        }

        // Vertical differences
        for row in 1..h {
            for col in 0..w {
                let diff = (frame[row * w + col] - frame[(row - 1) * w + col]).abs() as f64;
                if row % 8 == 0 {
                    boundary_diff += diff;
                    boundary_count += 1;
                } else {
                    interior_diff += diff;
                    interior_count += 1;
                }
            }
        }

        if boundary_count == 0 || interior_count == 0 {
            return 0.0;
        }

        let mean_boundary = boundary_diff / boundary_count as f64;
        let mean_interior = interior_diff / interior_count as f64;

        if mean_interior < 1e-9 {
            return if mean_boundary > 1e-9 { 1.0 } else { 0.0 };
        }

        // Normalise ratio to [0, 1] via a sigmoid-like mapping
        let ratio = mean_boundary / mean_interior;
        ((ratio - 1.0) / 4.0).clamp(0.0, 1.0) as f32
    }
}

// ─── Ringing detector ─────────────────────────────────────────────────────────

/// Detector for post-edge oscillations (ringing / Gibbs phenomenon).
pub struct RingingDetector;

impl RingingDetector {
    /// Compute a ringing score.
    ///
    /// `frame` is normalised luma, `edges` is a pre-computed edge map
    /// (e.g. Sobel magnitude), both row-major with dimensions `width × height`.
    ///
    /// The method looks for pixels *near* strong edges that oscillate
    /// (alternate above/below the local mean) — a hallmark of ringing.
    #[must_use]
    pub fn compute(frame: &[f32], edges: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;

        if w < 5 || h < 5 || frame.len() < w * h || edges.len() < w * h {
            return 0.0;
        }

        let edge_threshold = 0.1f32; // minimum edge strength to consider
        let search_radius: isize = 4; // pixels around edge to inspect

        let mut ringing_energy = 0.0f64;
        let mut samples = 0u64;

        for row in 0..h as isize {
            for col in 0..w as isize {
                if edges[(row * w as isize + col) as usize] < edge_threshold {
                    continue;
                }

                // Inspect a small neighbourhood for sign oscillations
                let mut neighbourhood = Vec::with_capacity(16);
                for dr in -search_radius..=search_radius {
                    let nr = row + dr;
                    if nr < 0 || nr >= h as isize {
                        continue;
                    }
                    let idx = (nr * w as isize + col) as usize;
                    if idx < frame.len() {
                        neighbourhood.push(frame[idx]);
                    }
                }

                if neighbourhood.len() < 3 {
                    continue;
                }

                let local_mean = neighbourhood.iter().sum::<f32>() / neighbourhood.len() as f32;

                // Count sign changes (oscillations around mean)
                let mut sign_changes = 0u32;
                let mut prev_sign = neighbourhood[0] > local_mean;
                for &v in &neighbourhood[1..] {
                    let cur_sign = v > local_mean;
                    if cur_sign != prev_sign {
                        sign_changes += 1;
                    }
                    prev_sign = cur_sign;
                }

                // Normalise by max possible sign changes
                let max_changes = (neighbourhood.len() - 1) as f32;
                if max_changes > 0.0 {
                    ringing_energy += f64::from(sign_changes as f32 / max_changes);
                    samples += 1;
                }
            }
        }

        if samples == 0 {
            return 0.0;
        }

        (ringing_energy / samples as f64).clamp(0.0, 1.0) as f32
    }
}

// ─── Banding detector ─────────────────────────────────────────────────────────

/// Detector for colour banding (false contours in smooth gradients).
pub struct BandingDetector;

impl BandingDetector {
    /// Compute a banding score.
    ///
    /// Banding manifests as many near-zero gradients inside otherwise smooth
    /// regions.  We build a histogram of absolute gradient magnitudes and
    /// return the fraction of pixels whose gradient is smaller than a small
    /// threshold (indicating a flat step / false contour).
    ///
    /// `frame` is normalised luma, row-major, dimensions `width × height`.
    #[must_use]
    pub fn compute(frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;

        if w < 2 || h < 2 || frame.len() < w * h {
            return 0.0;
        }

        // Collect horizontal gradient magnitudes
        let mut gradients: Vec<f32> = Vec::with_capacity(w * h);

        for row in 0..h {
            for col in 0..w - 1 {
                let g = (frame[row * w + col + 1] - frame[row * w + col]).abs();
                gradients.push(g);
            }
        }
        for row in 0..h - 1 {
            for col in 0..w {
                let g = (frame[(row + 1) * w + col] - frame[row * w + col]).abs();
                gradients.push(g);
            }
        }

        if gradients.is_empty() {
            return 0.0;
        }

        // Fraction of gradients below a "near-zero" threshold
        let near_zero_threshold = 1.0 / 255.0; // one quantisation step
        let near_zero = gradients
            .iter()
            .filter(|&&g| g < near_zero_threshold)
            .count() as f32;
        let total = gradients.len() as f32;

        // In a perfectly smooth gradient, most gradients are near-zero → high banding risk.
        // We scale so that ~70 % near-zero = score 1.0.
        ((near_zero / total - 0.5) / 0.2).clamp(0.0, 1.0)
    }
}

// ─── Compression artifact suite ───────────────────────────────────────────────

/// Combined compression artefact scores for a single frame or video.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionArtifactSuite {
    /// Blockiness score [0, 1]
    pub blocking: f32,
    /// Ringing score [0, 1]
    pub ringing: f32,
    /// Banding score [0, 1]
    pub banding: f32,
    /// Mosquito-noise estimate [0, 1]
    pub mosquito_noise: f32,
}

impl CompressionArtifactSuite {
    /// Compute a weighted overall artefact score [0, 1].
    ///
    /// Weights: blocking 40 %, ringing 30 %, banding 20 %, mosquito 10 %.
    #[must_use]
    pub fn overall_score(&self) -> f32 {
        0.40 * self.blocking
            + 0.30 * self.ringing
            + 0.20 * self.banding
            + 0.10 * self.mosquito_noise
    }

    /// Classify the overall score.
    #[must_use]
    pub fn severity(&self) -> ArtifactSeverity {
        ArtifactSeverity::from_score(self.overall_score())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ArtifactSeverity ───────────────────────────────────────────────────

    #[test]
    fn test_severity_from_score() {
        assert_eq!(ArtifactSeverity::from_score(0.0), ArtifactSeverity::None);
        assert_eq!(ArtifactSeverity::from_score(0.05), ArtifactSeverity::None);
        assert_eq!(ArtifactSeverity::from_score(0.10), ArtifactSeverity::Minor);
        assert_eq!(
            ArtifactSeverity::from_score(0.20),
            ArtifactSeverity::Moderate
        );
        assert_eq!(ArtifactSeverity::from_score(0.40), ArtifactSeverity::Severe);
        assert_eq!(
            ArtifactSeverity::from_score(0.90),
            ArtifactSeverity::Extreme
        );
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ArtifactSeverity::None < ArtifactSeverity::Minor);
        assert!(ArtifactSeverity::Minor < ArtifactSeverity::Moderate);
        assert!(ArtifactSeverity::Moderate < ArtifactSeverity::Severe);
        assert!(ArtifactSeverity::Severe < ArtifactSeverity::Extreme);
    }

    #[test]
    fn test_severity_max_acceptable_score() {
        assert!(
            ArtifactSeverity::None.max_acceptable_score()
                < ArtifactSeverity::Minor.max_acceptable_score()
        );
        assert!(ArtifactSeverity::Extreme.max_acceptable_score() == f32::MAX);
    }

    // ── BlockinessDetector ─────────────────────────────────────────────────

    #[test]
    fn test_blockiness_flat_frame() {
        let frame = vec![0.5f32; 32 * 32];
        let score = BlockinessDetector::compute(&frame, 32, 32);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_blockiness_small_frame() {
        let frame = vec![0.5f32; 4];
        let score = BlockinessDetector::compute(&frame, 2, 2);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_blockiness_range() {
        let w = 32u32;
        let h = 32u32;
        // Create a frame with sharp 8×8 block boundaries
        let frame: Vec<f32> = (0..w * h)
            .map(|i| {
                let row = (i / w) as f32;
                let col = (i % w) as f32;
                ((row / 8.0).floor() + (col / 8.0).floor()) % 2.0 * 0.5
            })
            .collect();
        let score = BlockinessDetector::compute(&frame, w, h);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_blockiness_score_severity() {
        let bs = BlockinessScore {
            score: 0.0,
            worst_frame: None,
        };
        assert_eq!(bs.severity(), ArtifactSeverity::None);

        let bs = BlockinessScore {
            score: 0.5,
            worst_frame: Some(3),
        };
        assert_eq!(bs.severity(), ArtifactSeverity::Severe);
    }

    // ── RingingDetector ────────────────────────────────────────────────────

    #[test]
    fn test_ringing_flat_frame() {
        let frame = vec![0.5f32; 16 * 16];
        let edges = vec![0.0f32; 16 * 16]; // no edges
        let score = RingingDetector::compute(&frame, &edges, 16, 16);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_ringing_small_frame() {
        let frame = vec![0.5f32; 4];
        let edges = vec![0.5f32; 4];
        let score = RingingDetector::compute(&frame, &edges, 2, 2);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_ringing_range() {
        // Alternating stripe pattern near a strong edge
        let w = 16u32;
        let h = 16u32;
        let mut frame = vec![0.5f32; (w * h) as usize];
        let mut edges = vec![0.0f32; (w * h) as usize];

        // Mark the middle column as an edge
        for row in 0..h as usize {
            edges[row * w as usize + w as usize / 2] = 1.0;
        }
        // Alternate pixels around the edge to simulate ringing
        for row in 0..h as usize {
            for col in 0..w as usize {
                frame[row * w as usize + col] = if col % 2 == 0 { 0.2 } else { 0.8 };
            }
        }

        let score = RingingDetector::compute(&frame, &edges, w, h);
        assert!((0.0..=1.0).contains(&score));
    }

    // ── BandingDetector ────────────────────────────────────────────────────

    #[test]
    fn test_banding_flat_frame() {
        let frame = vec![0.5f32; 16 * 16];
        let score = BandingDetector::compute(&frame, 16, 16);
        // Flat frame → all gradients = 0 → very high banding score
        assert!(score > 0.0);
    }

    #[test]
    fn test_banding_gradient_frame() {
        let w = 16u32;
        let h = 16u32;
        let frame: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let score = BandingDetector::compute(&frame, w, h);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_banding_small_frame() {
        let frame = vec![0.5f32; 1];
        let score = BandingDetector::compute(&frame, 1, 1);
        assert_eq!(score, 0.0);
    }

    // ── CompressionArtifactSuite ───────────────────────────────────────────

    #[test]
    fn test_compression_suite_overall_score() {
        let suite = CompressionArtifactSuite {
            blocking: 0.4,
            ringing: 0.2,
            banding: 0.1,
            mosquito_noise: 0.0,
        };
        let expected = 0.40 * 0.4 + 0.30 * 0.2 + 0.20 * 0.1 + 0.10 * 0.0;
        assert!((suite.overall_score() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_compression_suite_zero_score() {
        let suite = CompressionArtifactSuite {
            blocking: 0.0,
            ringing: 0.0,
            banding: 0.0,
            mosquito_noise: 0.0,
        };
        assert_eq!(suite.overall_score(), 0.0);
        assert_eq!(suite.severity(), ArtifactSeverity::None);
    }

    #[test]
    fn test_compression_suite_max_score() {
        let suite = CompressionArtifactSuite {
            blocking: 1.0,
            ringing: 1.0,
            banding: 1.0,
            mosquito_noise: 1.0,
        };
        assert!((suite.overall_score() - 1.0).abs() < 1e-5);
        assert_eq!(suite.severity(), ArtifactSeverity::Extreme);
    }

    #[test]
    fn test_compression_suite_severity_moderate() {
        let suite = CompressionArtifactSuite {
            blocking: 0.3,
            ringing: 0.2,
            banding: 0.1,
            mosquito_noise: 0.0,
        };
        let score = suite.overall_score();
        assert!(score <= ArtifactSeverity::Moderate.max_acceptable_score());
    }

    #[test]
    fn test_compression_suite_weights_sum_to_one() {
        // Verifies the four weights sum to exactly 1.0
        let w: f32 = 0.40 + 0.30 + 0.20 + 0.10;
        assert!((w - 1.0).abs() < 1e-6);
    }
}
