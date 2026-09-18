//! Media tampering detection.
//!
//! Provides block-level consistency analysis to detect splicing, inpainting,
//! copy-move, recompression artifacts, and color manipulation.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// The type of tampering that was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TamperingType {
    /// Image splicing (region pasted from another image).
    Splicing,
    /// Inpainting / content-aware fill.
    Inpainting,
    /// Copy-move (region duplicated within the same image).
    CopyMove,
    /// Artifacts caused by recompression at a different quality level.
    RecompressArtifact,
    /// Suspicious color-channel manipulation.
    ColorManipulation,
}

impl TamperingType {
    /// Return `true` for tampering types that leave spatially-localised traces
    /// (i.e. everything except `RecompressArtifact` which is global).
    #[must_use]
    pub fn is_spatial(&self) -> bool {
        !matches!(self, TamperingType::RecompressArtifact)
    }
}

/// A rectangular image region that has been flagged as suspicious.
#[derive(Debug, Clone)]
pub struct SuspiciousRegion {
    /// Left edge of the region in pixels.
    pub x: u32,
    /// Top edge of the region in pixels.
    pub y: u32,
    /// Width of the region in pixels.
    pub width: u32,
    /// Height of the region in pixels.
    pub height: u32,
    /// Detection confidence in [0, 1].
    pub confidence: f32,
    /// Suspected tampering type.
    pub tampering_type: TamperingType,
}

impl SuspiciousRegion {
    /// Pixel area of this region.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Return `true` if `confidence` is at least `threshold`.
    #[must_use]
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// Noise and DCT statistics measured for a single image block.
#[derive(Debug, Clone, Copy)]
pub struct BlockConsistency {
    /// Block column index.
    pub block_x: u32,
    /// Block row index.
    pub block_y: u32,
    /// Estimated noise variance for this block.
    pub noise_variance: f32,
    /// Mean DCT coefficient magnitude for this block.
    pub dct_mean: f32,
}

impl BlockConsistency {
    /// Return `true` if this block's noise variance deviates significantly from
    /// `global_noise_var` (more than 2× difference in either direction).
    #[must_use]
    pub fn is_inconsistent(&self, global_noise_var: f32) -> bool {
        if global_noise_var <= 0.0 {
            return self.noise_variance > 0.0;
        }
        let ratio = self.noise_variance / global_noise_var;
        // Flag if local variance is less than half or more than twice the global
        ratio < 0.5 || ratio > 2.0
    }
}

/// Analyzes a set of image blocks for internal consistency.
#[derive(Debug, Clone)]
pub struct TamperingAnalyzer {
    /// How sensitive the detection is (0.0 = lenient, 1.0 = strict).
    pub sensitivity: f32,
}

impl TamperingAnalyzer {
    /// Create a new analyzer with the given sensitivity in [0, 1].
    #[must_use]
    pub fn new(sensitivity: f32) -> Self {
        Self {
            sensitivity: sensitivity.clamp(0.0, 1.0),
        }
    }

    /// Compute the global noise variance estimate as the mean of all block
    /// noise variances.
    ///
    /// Returns `0.0` if `blocks` is empty.
    #[must_use]
    pub fn global_noise_estimate(&self, blocks: &[BlockConsistency]) -> f32 {
        if blocks.is_empty() {
            return 0.0;
        }
        let sum: f32 = blocks.iter().map(|b| b.noise_variance).sum();
        sum / blocks.len() as f32
    }

    /// Analyze block consistency and return a list of suspicious regions.
    ///
    /// A block is flagged when its noise variance is inconsistent with the
    /// global average.  Each flagged block becomes a 1×1-block `SuspiciousRegion`.
    /// The confidence is linearly scaled by `sensitivity`.
    #[must_use]
    pub fn analyze_block_consistency(&self, blocks: &[BlockConsistency]) -> Vec<SuspiciousRegion> {
        let global_var = self.global_noise_estimate(blocks);
        let mut regions = Vec::new();

        for block in blocks {
            if block.is_inconsistent(global_var) {
                // Compute raw confidence based on how far the ratio deviates
                let ratio = if global_var > 0.0 {
                    block.noise_variance / global_var
                } else {
                    1.0
                };
                let deviation = (ratio - 1.0).abs();
                let confidence = (deviation * self.sensitivity).min(1.0);

                regions.push(SuspiciousRegion {
                    x: block.block_x,
                    y: block.block_y,
                    width: 1,
                    height: 1,
                    confidence,
                    tampering_type: TamperingType::Splicing,
                });
            }
        }

        regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── TamperingType ──────────────────────────────────────────────────────────

    #[test]
    fn test_splicing_is_spatial() {
        assert!(TamperingType::Splicing.is_spatial());
    }

    #[test]
    fn test_inpainting_is_spatial() {
        assert!(TamperingType::Inpainting.is_spatial());
    }

    #[test]
    fn test_copy_move_is_spatial() {
        assert!(TamperingType::CopyMove.is_spatial());
    }

    #[test]
    fn test_recompress_artifact_is_not_spatial() {
        assert!(!TamperingType::RecompressArtifact.is_spatial());
    }

    #[test]
    fn test_color_manipulation_is_spatial() {
        assert!(TamperingType::ColorManipulation.is_spatial());
    }

    // ── SuspiciousRegion ───────────────────────────────────────────────────────

    #[test]
    fn test_suspicious_region_area() {
        let r = SuspiciousRegion {
            x: 0,
            y: 0,
            width: 10,
            height: 20,
            confidence: 0.8,
            tampering_type: TamperingType::Splicing,
        };
        assert_eq!(r.area(), 200);
    }

    #[test]
    fn test_suspicious_region_is_confident_true() {
        let r = SuspiciousRegion {
            x: 0,
            y: 0,
            width: 5,
            height: 5,
            confidence: 0.9,
            tampering_type: TamperingType::CopyMove,
        };
        assert!(r.is_confident(0.7));
    }

    #[test]
    fn test_suspicious_region_is_confident_false() {
        let r = SuspiciousRegion {
            x: 0,
            y: 0,
            width: 5,
            height: 5,
            confidence: 0.4,
            tampering_type: TamperingType::CopyMove,
        };
        assert!(!r.is_confident(0.7));
    }

    // ── BlockConsistency ───────────────────────────────────────────────────────

    #[test]
    fn test_block_consistency_consistent() {
        let block = BlockConsistency {
            block_x: 0,
            block_y: 0,
            noise_variance: 10.0,
            dct_mean: 5.0,
        };
        // Global is also 10.0, ratio = 1.0 — consistent
        assert!(!block.is_inconsistent(10.0));
    }

    #[test]
    fn test_block_consistency_inconsistent_too_high() {
        let block = BlockConsistency {
            block_x: 1,
            block_y: 0,
            noise_variance: 25.0,
            dct_mean: 3.0,
        };
        // Global is 10.0, ratio = 2.5 — inconsistent
        assert!(block.is_inconsistent(10.0));
    }

    #[test]
    fn test_block_consistency_inconsistent_too_low() {
        let block = BlockConsistency {
            block_x: 0,
            block_y: 1,
            noise_variance: 2.0,
            dct_mean: 1.0,
        };
        // Global is 10.0, ratio = 0.2 — inconsistent
        assert!(block.is_inconsistent(10.0));
    }

    // ── TamperingAnalyzer ──────────────────────────────────────────────────────

    #[test]
    fn test_analyzer_global_noise_empty() {
        let a = TamperingAnalyzer::new(0.8);
        assert_eq!(a.global_noise_estimate(&[]), 0.0);
    }

    #[test]
    fn test_analyzer_global_noise_estimate() {
        let blocks = vec![
            BlockConsistency {
                block_x: 0,
                block_y: 0,
                noise_variance: 8.0,
                dct_mean: 1.0,
            },
            BlockConsistency {
                block_x: 1,
                block_y: 0,
                noise_variance: 12.0,
                dct_mean: 1.0,
            },
        ];
        let a = TamperingAnalyzer::new(1.0);
        let est = a.global_noise_estimate(&blocks);
        assert!((est - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_analyze_block_consistency_no_tamper() {
        // All blocks have the same variance — none should be flagged
        let blocks: Vec<BlockConsistency> = (0..4)
            .map(|i| BlockConsistency {
                block_x: i,
                block_y: 0,
                noise_variance: 10.0,
                dct_mean: 2.0,
            })
            .collect();
        let a = TamperingAnalyzer::new(0.9);
        let regions = a.analyze_block_consistency(&blocks);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_analyze_block_consistency_detects_outlier() {
        let mut blocks: Vec<BlockConsistency> = (0..4)
            .map(|i| BlockConsistency {
                block_x: i,
                block_y: 0,
                noise_variance: 10.0,
                dct_mean: 2.0,
            })
            .collect();
        // Inject one outlier with 5× higher variance
        blocks[2].noise_variance = 50.0;
        let a = TamperingAnalyzer::new(1.0);
        let regions = a.analyze_block_consistency(&blocks);
        assert!(!regions.is_empty());
        assert_eq!(regions[0].block_x(), 2);
    }
}

// Helper for test readability
impl SuspiciousRegion {
    fn block_x(&self) -> u32 {
        self.x
    }
}
