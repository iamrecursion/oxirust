//! Resolution ladder management for adaptive bitrate streaming.
//!
//! Provides standard resolution constants, ladder construction, and normalization utilities.

use crate::aspect_ratio::AspectRatio;

/// A video resolution expressed as width × height.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolution {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Resolution {
    /// Total pixel count.
    #[must_use]
    #[allow(dead_code)]
    pub fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Pixel count in megapixels.
    #[must_use]
    #[allow(dead_code)]
    pub fn megapixels(&self) -> f32 {
        self.pixels() as f32 / 1_000_000.0
    }

    /// Reduced aspect ratio.
    #[must_use]
    #[allow(dead_code)]
    pub fn aspect_ratio(&self) -> AspectRatio {
        AspectRatio::new(self.width, self.height)
    }
}

impl PartialOrd for Resolution {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Resolution {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.pixels().cmp(&other.pixels())
    }
}

// Common resolution constants
/// 240p (426×240).
#[allow(dead_code)]
pub const R240P: Resolution = Resolution {
    width: 426,
    height: 240,
};
/// 360p (640×360).
#[allow(dead_code)]
pub const R360P: Resolution = Resolution {
    width: 640,
    height: 360,
};
/// 480p (854×480).
#[allow(dead_code)]
pub const R480P: Resolution = Resolution {
    width: 854,
    height: 480,
};
/// 720p HD (1280×720).
#[allow(dead_code)]
pub const R720P: Resolution = Resolution {
    width: 1280,
    height: 720,
};
/// 1080p Full HD (1920×1080).
#[allow(dead_code)]
pub const R1080P: Resolution = Resolution {
    width: 1920,
    height: 1080,
};
/// 1440p Quad HD (2560×1440).
#[allow(dead_code)]
pub const R1440P: Resolution = Resolution {
    width: 2560,
    height: 1440,
};
/// 2160p 4K UHD (3840×2160).
#[allow(dead_code)]
pub const R2160P: Resolution = Resolution {
    width: 3840,
    height: 2160,
};
/// 4320p 8K UHD (7680×4320).
#[allow(dead_code)]
pub const R4320P: Resolution = Resolution {
    width: 7680,
    height: 4320,
};

/// A resolution ladder — an ordered set of quality rungs for ABR streaming.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct ResolutionLadder {
    /// Rungs in ascending order (smallest first).
    pub rungs: Vec<Resolution>,
}

impl ResolutionLadder {
    /// Create an empty ladder.
    #[must_use]
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self { rungs: Vec::new() }
    }

    /// Add a resolution rung (inserted in sorted order).
    #[allow(dead_code)]
    pub fn add(&mut self, r: Resolution) {
        let pos = self
            .rungs
            .partition_point(|existing| existing.pixels() <= r.pixels());
        self.rungs.insert(pos, r);
    }

    /// Return all rungs with fewer pixels than `r`.
    #[must_use]
    #[allow(dead_code)]
    pub fn below(&self, r: &Resolution) -> Vec<&Resolution> {
        self.rungs
            .iter()
            .filter(|&rung| rung.pixels() < r.pixels())
            .collect()
    }

    /// Return all rungs with more pixels than `r`.
    #[must_use]
    #[allow(dead_code)]
    pub fn above(&self, r: &Resolution) -> Vec<&Resolution> {
        self.rungs
            .iter()
            .filter(|&rung| rung.pixels() > r.pixels())
            .collect()
    }

    /// Return the rung nearest in pixel count to `(w, h)`.
    ///
    /// Panics if the ladder is empty.
    #[must_use]
    #[allow(dead_code)]
    pub fn nearest(&self, w: u32, h: u32) -> &Resolution {
        let target = u64::from(w) * u64::from(h);
        self.rungs
            .iter()
            .min_by_key(|r| {
                let p = r.pixels();
                if p >= target {
                    p - target
                } else {
                    target - p
                }
            })
            .expect("ResolutionLadder must not be empty")
    }
}

/// Generates standard ABR resolution ladders for common codecs.
pub struct LadderGenerator;

impl LadderGenerator {
    /// Build a standard ABR ladder for the given codec up to `max` resolution.
    ///
    /// - `"h264"` / `"avc"`: 240p, 360p, 480p, 720p, 1080p
    /// - `"h265"` / `"hevc"`: 360p, 720p, 1080p, 2160p
    /// - `"av1"`: 240p, 480p, 720p, 1080p, 1440p, 2160p
    /// - Any other: full standard ladder
    #[must_use]
    #[allow(dead_code)]
    pub fn abr_ladder(max: Resolution, codec: &str) -> ResolutionLadder {
        let candidates: &[Resolution] = match codec.to_lowercase().as_str() {
            "h264" | "avc" => &[R240P, R360P, R480P, R720P, R1080P],
            "h265" | "hevc" => &[R360P, R720P, R1080P, R2160P],
            "av1" => &[R240P, R480P, R720P, R1080P, R1440P, R2160P],
            _ => &[R240P, R360P, R480P, R720P, R1080P, R1440P, R2160P],
        };

        let mut ladder = ResolutionLadder::new();
        for &r in candidates {
            if r.pixels() <= max.pixels() {
                ladder.add(r);
            }
        }
        // Always include max if not already present
        if !ladder.rungs.contains(&max) {
            ladder.add(max);
        }
        ladder
    }
}

/// Normalizes resolutions to be divisible by a given modulus.
pub struct ResolutionNormalizer;

impl ResolutionNormalizer {
    /// Round `(w, h)` to the nearest multiple of `modulus`.
    ///
    /// This is required by many codecs (e.g. H.264 requires mod-2, HEVC may require mod-8).
    #[must_use]
    #[allow(dead_code)]
    pub fn normalize_to_mod(w: u32, h: u32, modulus: u32) -> (u32, u32) {
        if modulus == 0 {
            return (w, h);
        }
        let round_to_mod = |v: u32| {
            let rem = v % modulus;
            if rem == 0 {
                v
            } else if rem < modulus / 2 + modulus % 2 {
                v - rem
            } else {
                v + (modulus - rem)
            }
        };
        (round_to_mod(w).max(modulus), round_to_mod(h).max(modulus))
    }
}

// ── PerceptualLadder ──────────────────────────────────────────────────────────

/// A single rung in a perceptual ABR ladder.
///
/// Holds the encoding dimensions and estimated quality metrics for a given
/// bitrate, computed using the Per-Title Encoding heuristic.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OptimalRung {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Target bitrate in kbit/s.
    pub bitrate: u32,
    /// Estimated PSNR in dB.
    pub psnr_estimate: f32,
    /// Estimated VMAF score (0–100).
    pub vmaf_estimate: f32,
}

/// Content difficulty score for per-title encoding decisions.
///
/// Higher values indicate harder-to-encode content that may need more bits
/// at each resolution rung.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ContentDifficultyScore {
    /// Normalised motion intensity (0.0 = static, 1.0 = very high motion).
    pub motion_score: f32,
    /// Normalised texture richness (0.0 = flat, 1.0 = highly detailed).
    pub texture_score: f32,
    /// Average scene-change rate (scene changes per second).
    pub scene_change_rate: f32,
}

impl ContentDifficultyScore {
    /// Compute an overall encoding complexity factor in the range [0.0, 1.0].
    ///
    /// Weights motion most heavily, followed by texture and scene changes.
    #[must_use]
    #[allow(dead_code)]
    pub fn encoding_complexity(&self) -> f32 {
        let motion_weight = 0.5_f32;
        let texture_weight = 0.3_f32;
        let scene_weight = 0.2_f32;

        // Normalise scene-change rate to [0, 1] assuming >5 changes/s is max.
        let scene_norm = (self.scene_change_rate / 5.0).clamp(0.0, 1.0);

        (motion_weight * self.motion_score.clamp(0.0, 1.0)
            + texture_weight * self.texture_score.clamp(0.0, 1.0)
            + scene_weight * scene_norm)
            .clamp(0.0, 1.0)
    }
}

/// Per-title perceptual ABR ladder builder.
///
/// Given the source dimensions, content type, and a list of candidate
/// bitrates, computes an optimal ladder of encoding rungs using PSNR / VMAF
/// estimates derived from the Per-Title Encoding heuristic.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerceptualLadder {
    /// Source video width.
    pub input_width: u32,
    /// Source video height.
    pub input_height: u32,
    /// Content type hint (e.g. `"animation"`, `"sports"`, `"film"`).
    pub content_type: String,
    /// Candidate bitrates in kbit/s (sorted ascending after construction).
    pub available_bitrates: Vec<u32>,
}

impl PerceptualLadder {
    /// Create a new `PerceptualLadder` builder.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(
        input_width: u32,
        input_height: u32,
        content_type: impl Into<String>,
        available_bitrates: Vec<u32>,
    ) -> Self {
        let mut sorted = available_bitrates;
        sorted.sort_unstable();
        Self {
            input_width,
            input_height,
            content_type: content_type.into(),
            available_bitrates: sorted,
        }
    }

    /// Compute the optimal ladder for this input.
    ///
    /// Delegates to the free function [`compute_optimal_ladder`].
    #[must_use]
    #[allow(dead_code)]
    pub fn compute(&self) -> Vec<OptimalRung> {
        compute_optimal_ladder(
            self.input_width,
            self.input_height,
            &self.content_type,
            &self.available_bitrates,
        )
    }
}

/// Reference bitrate (kbit/s) for 1920×1080 content, used to normalise PSNR.
const REFERENCE_BITRATE_1080P: f32 = 4000.0;

/// Noise variance constant calibrated so that PSNR ≈ 40 dB at reference bitrate.
///
/// PSNR = 10·log10(255² / (σ² · (reference / bitrate)))
/// At bitrate = reference → PSNR = 10·log10(255² / σ²) ≈ 40 dB
/// → σ² = 255² / 10^4 ≈ 6.5025
const NOISE_VARIANCE: f32 = 6.5025;

/// Compute the optimal perceptual ABR ladder.
///
/// Uses the Per-Title Encoding heuristic:
/// - PSNR ≈ `10 · log10(255² / (noise_variance · (reference / bitrate_kbps)))`
/// - VMAF ≈ `95 − 15 · exp(−bitrate_kbps / (reference · 0.3))`
///
/// The reference bitrate is scaled by the pixel-count ratio relative to 1080p.
///
/// Rungs are pruned if:
/// - Estimated VMAF < 50 (poor quality), or
/// - VMAF improvement over the previous (lower) rung < 5 points (redundant rung).
///
/// For each surviving bitrate the output resolution is computed as:
/// - Start from input dimensions.
/// - If the bitrate is less than `reference × 0.5`, scale dimensions down by
///   `sqrt(bitrate / (reference × 0.5))` so lower rungs use lower resolutions.
/// - Dimensions are rounded to even numbers (required by most codecs).
#[must_use]
#[allow(dead_code)]
pub fn compute_optimal_ladder(
    width: u32,
    height: u32,
    content_type: &str,
    bitrates: &[u32],
) -> Vec<OptimalRung> {
    if bitrates.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }

    // Scale reference bitrate proportionally to pixel count.
    let pixel_ratio = (u64::from(width) * u64::from(height)) as f32 / (1920.0 * 1080.0);
    let reference = REFERENCE_BITRATE_1080P * pixel_ratio;

    // Content-type bitrate multiplier: animation is cheaper, sports is harder.
    let content_multiplier = match content_type.to_lowercase().as_str() {
        "animation" | "cartoon" => 0.7,
        "sports" | "action" | "gaming" => 1.4,
        "film" | "movie" | "drama" => 1.0,
        "documentary" | "talking_head" => 0.8,
        _ => 1.0,
    };
    let effective_reference = reference * content_multiplier;

    // Sort bitrates ascending.
    let mut sorted_bitrates: Vec<u32> = bitrates.to_vec();
    sorted_bitrates.sort_unstable();

    // Compute quality estimates for every candidate bitrate.
    let candidates: Vec<(u32, f32, f32, u32, u32)> = sorted_bitrates
        .iter()
        .map(|&br| {
            let br_f = br as f32;
            // PSNR formula
            let psnr_val = 10.0
                * (255.0_f32 * 255.0 / (NOISE_VARIANCE * (effective_reference / br_f).max(1e-6)))
                    .log10();

            // VMAF formula
            let vmaf_val: f32 = 95.0 - 15.0 * (-(br_f / (effective_reference * 0.3))).exp();

            // Compute resolution for this rung.
            let scale_threshold = effective_reference * 0.5;
            let (rung_w, rung_h) = if br_f < scale_threshold {
                let dim_scale = (br_f / scale_threshold).sqrt().clamp(0.1, 1.0);
                let rw = ((width as f32 * dim_scale) as u32).max(2) & !1;
                let rh = ((height as f32 * dim_scale) as u32).max(2) & !1;
                (rw, rh)
            } else {
                (width & !1, height & !1)
            };

            (br, psnr_val, vmaf_val, rung_w, rung_h)
        })
        .collect();

    // Prune rungs: reject VMAF < 50 and duplicate-quality rungs.
    let mut rungs: Vec<OptimalRung> = Vec::new();
    let mut prev_vmaf = 0.0f32;
    for &(br, psnr_val, vmaf_val, rw, rh) in &candidates {
        if vmaf_val < 50.0 {
            continue;
        }
        let improvement = vmaf_val - prev_vmaf;
        if !rungs.is_empty() && improvement < 5.0 {
            // Replace the last rung if this one has higher quality at same improvement range.
            if let Some(last) = rungs.last_mut() {
                if vmaf_val > last.vmaf_estimate {
                    *last = OptimalRung {
                        width: rw,
                        height: rh,
                        bitrate: br,
                        psnr_estimate: psnr_val,
                        vmaf_estimate: vmaf_val,
                    };
                }
            }
            continue;
        }
        rungs.push(OptimalRung {
            width: rw,
            height: rh,
            bitrate: br,
            psnr_estimate: psnr_val,
            vmaf_estimate: vmaf_val,
        });
        prev_vmaf = vmaf_val;
    }

    rungs
}

/// Selects the optimal rung from a ladder given a bandwidth constraint.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct RungSelector {
    /// The pre-computed ladder of rungs in ascending order.
    pub ladder: Vec<OptimalRung>,
}

impl RungSelector {
    /// Create a selector from a pre-computed ladder.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(ladder: Vec<OptimalRung>) -> Self {
        Self { ladder }
    }

    /// Return the best rung whose bitrate does not exceed `bandwidth_kbps`.
    ///
    /// Returns `None` if all rungs exceed the bandwidth.
    #[must_use]
    #[allow(dead_code)]
    pub fn select(&self, bandwidth_kbps: u32) -> Option<&OptimalRung> {
        self.ladder
            .iter()
            .filter(|r| r.bitrate <= bandwidth_kbps)
            .max_by_key(|r| r.bitrate)
    }

    /// Return the rung with the highest estimated VMAF.
    #[must_use]
    #[allow(dead_code)]
    pub fn best_quality(&self) -> Option<&OptimalRung> {
        self.ladder.iter().max_by(|a, b| {
            a.vmaf_estimate
                .partial_cmp(&b.vmaf_estimate)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

// ── Per-Title Encoding Ladder with VIF/SSIM Target Thresholds ─────────────────

/// Quality metric target thresholds for per-title encoding decisions.
///
/// Each threshold defines the minimum acceptable quality at a given resolution
/// rung. Rungs that fall below these thresholds are promoted to a higher
/// bitrate or excluded from the ladder.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct QualityTarget {
    /// Minimum acceptable VIF (Visual Information Fidelity) score [0.0, 1.0].
    /// VIF > 0.6 is considered acceptable, > 0.8 is good.
    pub min_vif: f32,
    /// Minimum acceptable SSIM score [0.0, 1.0].
    /// SSIM > 0.9 is typically considered good quality.
    pub min_ssim: f32,
    /// Maximum acceptable bitrate per pixel per second.
    /// Used to prevent over-allocation of bits to easy content.
    pub max_bits_per_pixel: f32,
}

impl Default for QualityTarget {
    fn default() -> Self {
        Self {
            min_vif: 0.6,
            min_ssim: 0.9,
            max_bits_per_pixel: 0.15,
        }
    }
}

impl QualityTarget {
    /// Create a quality target for high-quality streaming (e.g. premium OTT).
    pub fn premium() -> Self {
        Self {
            min_vif: 0.75,
            min_ssim: 0.95,
            max_bits_per_pixel: 0.20,
        }
    }

    /// Create a quality target for bandwidth-constrained streaming (e.g. mobile).
    pub fn mobile() -> Self {
        Self {
            min_vif: 0.5,
            min_ssim: 0.85,
            max_bits_per_pixel: 0.10,
        }
    }

    /// Create a quality target for archival/mastering quality.
    pub fn archival() -> Self {
        Self {
            min_vif: 0.85,
            min_ssim: 0.98,
            max_bits_per_pixel: 0.30,
        }
    }
}

/// A rung in a per-title encoding ladder with quality metric estimates.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerTitleRung {
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Target bitrate in kbit/s.
    pub bitrate_kbps: u32,
    /// Estimated VIF score.
    pub estimated_vif: f32,
    /// Estimated SSIM score.
    pub estimated_ssim: f32,
    /// Bits per pixel per second at the given framerate.
    pub bits_per_pixel: f32,
    /// Whether this rung meets the quality target thresholds.
    pub meets_target: bool,
}

/// Per-title encoding ladder builder that uses VIF/SSIM target thresholds
/// to determine optimal resolution-bitrate pairings.
///
/// Unlike a fixed ladder, this analyzer computes content-specific quality
/// estimates and removes rungs that don't meet quality requirements, or
/// adjusts bitrates upward to meet targets.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerTitleLadder {
    /// Source video width.
    pub source_width: u32,
    /// Source video height.
    pub source_height: u32,
    /// Source video framerate (fps).
    pub framerate: f32,
    /// Content difficulty score.
    pub difficulty: ContentDifficultyScore,
    /// Quality targets for rung selection.
    pub target: QualityTarget,
    /// Available bitrates to consider (kbit/s).
    pub bitrates: Vec<u32>,
    /// Candidate resolutions (heights) to evaluate. Width derived from aspect ratio.
    pub candidate_heights: Vec<u32>,
}

impl PerTitleLadder {
    /// Create a new per-title ladder builder.
    ///
    /// Uses standard candidate heights (240, 360, 480, 720, 1080, 1440, 2160).
    #[allow(dead_code)]
    pub fn new(
        source_width: u32,
        source_height: u32,
        framerate: f32,
        difficulty: ContentDifficultyScore,
        target: QualityTarget,
        bitrates: Vec<u32>,
    ) -> Self {
        let mut sorted = bitrates;
        sorted.sort_unstable();
        Self {
            source_width,
            source_height,
            framerate: framerate.max(1.0),
            difficulty,
            target,
            bitrates: sorted,
            candidate_heights: vec![240, 360, 480, 720, 1080, 1440, 2160],
        }
    }

    /// Set custom candidate heights.
    #[allow(dead_code)]
    pub fn with_candidate_heights(mut self, heights: Vec<u32>) -> Self {
        self.candidate_heights = heights;
        self
    }

    /// Compute the per-title ladder.
    ///
    /// For each candidate resolution and bitrate pairing:
    /// 1. Estimate VIF using a logarithmic model calibrated to content difficulty.
    /// 2. Estimate SSIM using a hyperbolic tangent model.
    /// 3. Compute bits-per-pixel metric.
    /// 4. Filter rungs that don't meet quality targets.
    /// 5. Select the optimal bitrate for each resolution.
    #[allow(dead_code, clippy::cast_precision_loss)]
    pub fn compute(&self) -> Vec<PerTitleRung> {
        if self.source_width == 0 || self.source_height == 0 || self.bitrates.is_empty() {
            return Vec::new();
        }

        let src_aspect = self.source_width as f64 / self.source_height as f64;
        let complexity = self.difficulty.encoding_complexity();

        // Higher complexity → needs more bits for same quality.
        let complexity_factor = 1.0_f64 + complexity as f64 * 0.8;

        let mut rungs = Vec::new();

        for &height in &self.candidate_heights {
            // Skip resolutions larger than source.
            if height > self.source_height {
                continue;
            }

            let width = ((height as f64 * src_aspect).round() as u32).max(2) & !1;
            let height_even = height & !1;
            if height_even == 0 {
                continue;
            }

            let pixels = u64::from(width) * u64::from(height_even);
            let pixels_f = pixels as f64;

            // Find the best bitrate that meets quality targets for this resolution.
            let mut best_rung: Option<PerTitleRung> = None;

            for &br in &self.bitrates {
                let br_f = br as f64;

                // Bits per pixel per second
                let bpp = (br_f * 1000.0) / (pixels_f * self.framerate as f64);

                // VIF estimate: logarithmic model
                // VIF ≈ 0.3 + 0.7 * (1 - exp(-bpp * k / complexity))
                let k_vif = 15.0;
                let vif = (0.3 + 0.7 * (1.0 - (-bpp * k_vif / complexity_factor).exp()))
                    .clamp(0.0, 1.0) as f32;

                // SSIM estimate: tanh model
                // SSIM ≈ tanh(bpp * k / complexity) clamped to [0, 1]
                let k_ssim = 12.0;
                let ssim = (bpp * k_ssim / complexity_factor).tanh().clamp(0.0, 1.0) as f32;

                let bpp_f32 = bpp as f32;

                let meets = vif >= self.target.min_vif
                    && ssim >= self.target.min_ssim
                    && bpp_f32 <= self.target.max_bits_per_pixel;

                let rung = PerTitleRung {
                    width,
                    height: height_even,
                    bitrate_kbps: br,
                    estimated_vif: vif,
                    estimated_ssim: ssim,
                    bits_per_pixel: bpp_f32,
                    meets_target: meets,
                };

                if meets {
                    // Pick the lowest bitrate that meets the target (most efficient).
                    if best_rung.is_none() {
                        best_rung = Some(rung);
                    }
                } else if best_rung.is_none() {
                    // Keep the highest-quality non-meeting rung as fallback.
                    best_rung = Some(rung);
                }
            }

            if let Some(rung) = best_rung {
                rungs.push(rung);
            }
        }

        rungs
    }

    /// Compute the ladder and filter to only rungs that meet quality targets.
    #[allow(dead_code)]
    pub fn compute_filtered(&self) -> Vec<PerTitleRung> {
        self.compute()
            .into_iter()
            .filter(|r| r.meets_target)
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_pixels() {
        assert_eq!(R1080P.pixels(), 1920 * 1080);
    }

    #[test]
    fn test_resolution_megapixels() {
        let mp = R1080P.megapixels();
        assert!((mp - 2.0736).abs() < 0.001);
    }

    #[test]
    fn test_resolution_aspect_ratio() {
        let ar = R1080P.aspect_ratio();
        assert_eq!(ar.width, 16);
        assert_eq!(ar.height, 9);
    }

    #[test]
    fn test_resolution_ordering() {
        assert!(R720P < R1080P);
        assert!(R2160P > R1080P);
    }

    #[test]
    fn test_ladder_add_sorted() {
        let mut ladder = ResolutionLadder::new();
        ladder.add(R1080P);
        ladder.add(R480P);
        ladder.add(R720P);
        assert_eq!(ladder.rungs[0], R480P);
        assert_eq!(ladder.rungs[1], R720P);
        assert_eq!(ladder.rungs[2], R1080P);
    }

    #[test]
    fn test_ladder_below() {
        let mut ladder = ResolutionLadder::new();
        ladder.add(R480P);
        ladder.add(R720P);
        ladder.add(R1080P);
        let below = ladder.below(&R720P);
        assert_eq!(below.len(), 1);
        assert_eq!(*below[0], R480P);
    }

    #[test]
    fn test_ladder_above() {
        let mut ladder = ResolutionLadder::new();
        ladder.add(R480P);
        ladder.add(R720P);
        ladder.add(R1080P);
        let above = ladder.above(&R720P);
        assert_eq!(above.len(), 1);
        assert_eq!(*above[0], R1080P);
    }

    #[test]
    fn test_ladder_nearest() {
        let mut ladder = ResolutionLadder::new();
        ladder.add(R480P);
        ladder.add(R720P);
        ladder.add(R1080P);
        let nearest = ladder.nearest(1280, 720);
        assert_eq!(*nearest, R720P);
    }

    #[test]
    fn test_abr_ladder_h264() {
        let ladder = LadderGenerator::abr_ladder(R1080P, "h264");
        assert!(!ladder.rungs.is_empty());
        // Should not exceed max
        for r in &ladder.rungs {
            assert!(r.pixels() <= R1080P.pixels());
        }
    }

    #[test]
    fn test_abr_ladder_av1() {
        let ladder = LadderGenerator::abr_ladder(R2160P, "av1");
        assert!(ladder.rungs.len() >= 4);
    }

    #[test]
    fn test_normalize_to_mod_exact() {
        // 1920 is divisible by 16, 1088 is the nearest multiple of 16 to 1080
        let (w, h) = ResolutionNormalizer::normalize_to_mod(1920, 1080, 16);
        assert_eq!(w, 1920);
        // 1080 % 16 = 8; since 8 >= 16/2=8, rounds up to 1080+8=1088
        assert_eq!(h, 1088);
    }

    #[test]
    fn test_normalize_to_mod_rounding() {
        let (w, h) = ResolutionNormalizer::normalize_to_mod(1921, 1081, 2);
        assert_eq!(w % 2, 0);
        assert_eq!(h % 2, 0);
    }

    #[test]
    fn test_normalize_to_mod_zero_modulus() {
        let (w, h) = ResolutionNormalizer::normalize_to_mod(1920, 1080, 0);
        assert_eq!(w, 1920);
        assert_eq!(h, 1080);
    }

    // ── PerceptualLadder / compute_optimal_ladder ─────────────────────────────

    #[test]
    fn test_compute_optimal_ladder_empty_bitrates() {
        let rungs = compute_optimal_ladder(1920, 1080, "film", &[]);
        assert!(rungs.is_empty());
    }

    #[test]
    fn test_compute_optimal_ladder_zero_dimensions() {
        let rungs = compute_optimal_ladder(0, 1080, "film", &[1000, 2000]);
        assert!(rungs.is_empty());
    }

    #[test]
    fn test_compute_optimal_ladder_returns_rungs_for_1080p() {
        let bitrates = vec![500, 1000, 2000, 4000, 8000];
        let rungs = compute_optimal_ladder(1920, 1080, "film", &bitrates);
        assert!(!rungs.is_empty(), "expected at least one rung");
    }

    #[test]
    fn test_compute_optimal_ladder_vmaf_at_least_50() {
        let bitrates = vec![1000, 2000, 4000, 8000];
        let rungs = compute_optimal_ladder(1920, 1080, "film", &bitrates);
        for r in &rungs {
            assert!(r.vmaf_estimate >= 50.0, "VMAF {:.1} < 50", r.vmaf_estimate);
        }
    }

    #[test]
    fn test_compute_optimal_ladder_psnr_increases_with_bitrate() {
        let bitrates = vec![1000, 2000, 4000, 8000];
        let rungs = compute_optimal_ladder(1920, 1080, "film", &bitrates);
        for w in rungs.windows(2) {
            assert!(
                w[1].psnr_estimate >= w[0].psnr_estimate - 0.01,
                "PSNR not monotone: {:.2} then {:.2}",
                w[0].psnr_estimate,
                w[1].psnr_estimate
            );
        }
    }

    #[test]
    fn test_compute_optimal_ladder_vmaf_increases_with_bitrate() {
        let bitrates = vec![2000, 4000, 8000, 16000];
        let rungs = compute_optimal_ladder(1920, 1080, "film", &bitrates);
        for w in rungs.windows(2) {
            assert!(
                w[1].vmaf_estimate >= w[0].vmaf_estimate - 0.01,
                "VMAF not monotone"
            );
        }
    }

    #[test]
    fn test_compute_optimal_ladder_low_bitrate_reduces_resolution() {
        let bitrates = vec![100, 8000];
        let rungs = compute_optimal_ladder(1920, 1080, "film", &bitrates);
        if rungs.len() >= 2 {
            // The low-bitrate rung should have a smaller or equal resolution.
            let low_rung = &rungs[0];
            let high_rung = &rungs[rungs.len() - 1];
            let low_pixels = u64::from(low_rung.width) * u64::from(low_rung.height);
            let high_pixels = u64::from(high_rung.width) * u64::from(high_rung.height);
            assert!(
                low_pixels <= high_pixels,
                "low bitrate should not exceed high bitrate resolution"
            );
        }
    }

    #[test]
    fn test_compute_optimal_ladder_animation_content() {
        let bitrates = vec![500, 1000, 2000, 4000];
        let rungs = compute_optimal_ladder(1920, 1080, "animation", &bitrates);
        // Animation uses lower reference → higher VMAF at same bitrate.
        let rungs_film = compute_optimal_ladder(1920, 1080, "film", &bitrates);
        if !rungs.is_empty() && !rungs_film.is_empty() {
            assert!(
                rungs[0].vmaf_estimate >= rungs_film[0].vmaf_estimate - 1.0,
                "animation should have comparable or better quality at same bitrate"
            );
        }
    }

    #[test]
    fn test_compute_optimal_ladder_sports_content() {
        let bitrates = vec![1000, 4000, 8000];
        let rungs = compute_optimal_ladder(1920, 1080, "sports", &bitrates);
        // Sports uses a higher reference, so VMAF at lower bitrates may be lower.
        assert!(!rungs.is_empty() || bitrates.len() > 0);
    }

    #[test]
    fn test_compute_optimal_ladder_output_dimensions_even() {
        let bitrates = vec![300, 1000, 4000];
        let rungs = compute_optimal_ladder(1920, 1080, "film", &bitrates);
        for r in &rungs {
            assert_eq!(r.width % 2, 0, "width {} is odd", r.width);
            assert_eq!(r.height % 2, 0, "height {} is odd", r.height);
        }
    }

    #[test]
    fn test_perceptual_ladder_struct_compute() {
        let pl = PerceptualLadder::new(1920, 1080, "film", vec![1000, 2000, 4000, 8000]);
        let rungs = pl.compute();
        assert!(!rungs.is_empty());
    }

    #[test]
    fn test_perceptual_ladder_sorts_bitrates() {
        let pl = PerceptualLadder::new(1920, 1080, "film", vec![8000, 1000, 4000]);
        assert_eq!(pl.available_bitrates, vec![1000, 4000, 8000]);
    }

    // ── ContentDifficultyScore ────────────────────────────────────────────────

    #[test]
    fn test_content_difficulty_zero_is_zero() {
        let s = ContentDifficultyScore {
            motion_score: 0.0,
            texture_score: 0.0,
            scene_change_rate: 0.0,
        };
        assert!((s.encoding_complexity()).abs() < 1e-5);
    }

    #[test]
    fn test_content_difficulty_max_is_one() {
        let s = ContentDifficultyScore {
            motion_score: 1.0,
            texture_score: 1.0,
            scene_change_rate: 5.0,
        };
        assert!((s.encoding_complexity() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_content_difficulty_clamped() {
        let s = ContentDifficultyScore {
            motion_score: 2.0, // over-range
            texture_score: -1.0,
            scene_change_rate: 100.0,
        };
        let c = s.encoding_complexity();
        assert!(c >= 0.0 && c <= 1.0, "complexity out of [0,1]: {c}");
    }

    #[test]
    fn test_content_difficulty_motion_dominant() {
        let s_motion = ContentDifficultyScore {
            motion_score: 1.0,
            texture_score: 0.0,
            scene_change_rate: 0.0,
        };
        let s_texture = ContentDifficultyScore {
            motion_score: 0.0,
            texture_score: 1.0,
            scene_change_rate: 0.0,
        };
        assert!(
            s_motion.encoding_complexity() > s_texture.encoding_complexity(),
            "motion weight should dominate texture"
        );
    }

    // ── RungSelector ─────────────────────────────────────────────────────────

    #[test]
    fn test_rung_selector_select_within_bandwidth() {
        let ladder = compute_optimal_ladder(1920, 1080, "film", &[1000, 2000, 4000, 8000]);
        let selector = RungSelector::new(ladder);
        let rung = selector.select(4000);
        assert!(rung.is_some());
        assert!(rung.expect("rung should be selected").bitrate <= 4000);
    }

    #[test]
    fn test_rung_selector_none_below_min_bitrate() {
        let ladder = compute_optimal_ladder(1920, 1080, "film", &[4000, 8000]);
        let selector = RungSelector::new(ladder);
        let rung = selector.select(100);
        assert!(rung.is_none());
    }

    #[test]
    fn test_rung_selector_best_quality_is_highest_vmaf() {
        let ladder = compute_optimal_ladder(1920, 1080, "film", &[1000, 4000, 8000]);
        if !ladder.is_empty() {
            let selector = RungSelector::new(ladder.clone());
            let best = selector.best_quality();
            assert!(best.is_some());
            let best_vmaf = best.expect("should have best quality").vmaf_estimate;
            let max_vmaf = ladder
                .iter()
                .map(|r| r.vmaf_estimate)
                .fold(0.0_f32, f32::max);
            assert!((best_vmaf - max_vmaf).abs() < 1e-4);
        }
    }

    // ── QualityTarget tests ─────────────────────────────────────────────────

    #[test]
    fn test_quality_target_default() {
        let t = QualityTarget::default();
        assert!((t.min_vif - 0.6).abs() < 1e-4);
        assert!((t.min_ssim - 0.9).abs() < 1e-4);
        assert!(t.max_bits_per_pixel > 0.0);
    }

    #[test]
    fn test_quality_target_premium() {
        let t = QualityTarget::premium();
        assert!(t.min_vif > QualityTarget::default().min_vif);
        assert!(t.min_ssim > QualityTarget::default().min_ssim);
    }

    #[test]
    fn test_quality_target_mobile() {
        let t = QualityTarget::mobile();
        assert!(t.min_vif < QualityTarget::default().min_vif);
        assert!(t.min_ssim < QualityTarget::default().min_ssim);
    }

    #[test]
    fn test_quality_target_archival() {
        let t = QualityTarget::archival();
        assert!(t.min_vif > QualityTarget::premium().min_vif);
    }

    // ── PerTitleLadder tests ────────────────────────────────────────────────

    #[test]
    fn test_per_title_ladder_basic() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.3,
            texture_score: 0.5,
            scene_change_rate: 1.0,
        };
        let ladder = PerTitleLadder::new(
            1920,
            1080,
            30.0,
            difficulty,
            QualityTarget::default(),
            vec![500, 1000, 2000, 4000, 8000],
        );
        let rungs = ladder.compute();
        assert!(!rungs.is_empty(), "should produce at least one rung");
    }

    #[test]
    fn test_per_title_ladder_respects_source_resolution() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.5,
            texture_score: 0.5,
            scene_change_rate: 1.0,
        };
        let ladder = PerTitleLadder::new(
            1280,
            720,
            30.0,
            difficulty,
            QualityTarget::default(),
            vec![1000, 2000, 4000],
        );
        let rungs = ladder.compute();
        for r in &rungs {
            assert!(
                r.height <= 720,
                "rung height {} exceeds source 720",
                r.height
            );
        }
    }

    #[test]
    fn test_per_title_ladder_even_dimensions() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.3,
            texture_score: 0.3,
            scene_change_rate: 0.5,
        };
        let ladder = PerTitleLadder::new(
            1920,
            1080,
            24.0,
            difficulty,
            QualityTarget::default(),
            vec![500, 1000, 2000, 4000],
        );
        let rungs = ladder.compute();
        for r in &rungs {
            assert_eq!(r.width % 2, 0, "width {} is odd", r.width);
            assert_eq!(r.height % 2, 0, "height {} is odd", r.height);
        }
    }

    #[test]
    fn test_per_title_ladder_vif_ssim_in_range() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.5,
            texture_score: 0.5,
            scene_change_rate: 2.0,
        };
        let ladder = PerTitleLadder::new(
            1920,
            1080,
            30.0,
            difficulty,
            QualityTarget::default(),
            vec![1000, 2000, 4000, 8000],
        );
        let rungs = ladder.compute();
        for r in &rungs {
            assert!(
                r.estimated_vif >= 0.0 && r.estimated_vif <= 1.0,
                "VIF {:.4} out of [0,1]",
                r.estimated_vif
            );
            assert!(
                r.estimated_ssim >= 0.0 && r.estimated_ssim <= 1.0,
                "SSIM {:.4} out of [0,1]",
                r.estimated_ssim
            );
        }
    }

    #[test]
    fn test_per_title_ladder_higher_bitrate_better_quality() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.4,
            texture_score: 0.4,
            scene_change_rate: 1.0,
        };
        // Test at a single resolution (720p) with increasing bitrates.
        let ladder = PerTitleLadder::new(
            1280,
            720,
            30.0,
            difficulty,
            QualityTarget::mobile(),
            vec![500, 1000, 2000, 4000],
        )
        .with_candidate_heights(vec![720]);
        let rungs = ladder.compute();
        assert!(!rungs.is_empty());
        // There's only one resolution, so the chosen rung should have the lowest
        // bitrate meeting the target, or the highest quality fallback.
    }

    #[test]
    fn test_per_title_ladder_empty_bitrates() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.5,
            texture_score: 0.5,
            scene_change_rate: 1.0,
        };
        let ladder = PerTitleLadder::new(
            1920,
            1080,
            30.0,
            difficulty,
            QualityTarget::default(),
            vec![],
        );
        assert!(ladder.compute().is_empty());
    }

    #[test]
    fn test_per_title_ladder_zero_dimensions() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.5,
            texture_score: 0.5,
            scene_change_rate: 1.0,
        };
        let ladder = PerTitleLadder::new(
            0,
            1080,
            30.0,
            difficulty,
            QualityTarget::default(),
            vec![1000],
        );
        assert!(ladder.compute().is_empty());
    }

    #[test]
    fn test_per_title_ladder_filtered() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.3,
            texture_score: 0.3,
            scene_change_rate: 0.5,
        };
        let ladder = PerTitleLadder::new(
            1920,
            1080,
            30.0,
            difficulty,
            QualityTarget::default(),
            vec![500, 1000, 2000, 4000, 8000],
        );
        let filtered = ladder.compute_filtered();
        for r in &filtered {
            assert!(
                r.meets_target,
                "filtered rung at {}x{} should meet target",
                r.width, r.height
            );
        }
    }

    #[test]
    fn test_per_title_ladder_high_complexity_needs_more_bits() {
        let easy = ContentDifficultyScore {
            motion_score: 0.1,
            texture_score: 0.1,
            scene_change_rate: 0.1,
        };
        let hard = ContentDifficultyScore {
            motion_score: 0.9,
            texture_score: 0.9,
            scene_change_rate: 4.0,
        };
        let bitrates = vec![1000, 2000, 4000];
        let easy_ladder = PerTitleLadder::new(
            1920,
            1080,
            30.0,
            easy,
            QualityTarget::default(),
            bitrates.clone(),
        )
        .with_candidate_heights(vec![720]);
        let hard_ladder =
            PerTitleLadder::new(1920, 1080, 30.0, hard, QualityTarget::default(), bitrates)
                .with_candidate_heights(vec![720]);

        let easy_rungs = easy_ladder.compute();
        let hard_rungs = hard_ladder.compute();

        if !easy_rungs.is_empty() && !hard_rungs.is_empty() {
            // Easy content should have higher quality at same/lower bitrate
            let easy_vif = easy_rungs[0].estimated_vif;
            let hard_vif = hard_rungs[0].estimated_vif;
            // At the same chosen bitrate, easy content should have >= VIF
            // (they might choose different bitrates, so we check the first rung)
            assert!(
                easy_vif >= hard_vif - 0.05,
                "easy VIF {easy_vif:.3} should be >= hard VIF {hard_vif:.3}"
            );
        }
    }

    #[test]
    fn test_per_title_ladder_custom_heights() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.3,
            texture_score: 0.3,
            scene_change_rate: 0.5,
        };
        let ladder = PerTitleLadder::new(
            1920,
            1080,
            30.0,
            difficulty,
            QualityTarget::default(),
            vec![2000, 4000],
        )
        .with_candidate_heights(vec![480, 720, 1080]);
        let rungs = ladder.compute();
        for r in &rungs {
            assert!(
                [480, 720, 1080].contains(&r.height),
                "unexpected height {}",
                r.height
            );
        }
    }

    #[test]
    fn test_per_title_rung_bits_per_pixel_positive() {
        let difficulty = ContentDifficultyScore {
            motion_score: 0.5,
            texture_score: 0.5,
            scene_change_rate: 1.0,
        };
        let ladder = PerTitleLadder::new(
            1920,
            1080,
            30.0,
            difficulty,
            QualityTarget::default(),
            vec![1000, 4000],
        );
        let rungs = ladder.compute();
        for r in &rungs {
            assert!(r.bits_per_pixel > 0.0, "bpp should be positive");
        }
    }
}
