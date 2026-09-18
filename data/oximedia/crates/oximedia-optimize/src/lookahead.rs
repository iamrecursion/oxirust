//! Lookahead analysis for temporal optimization.

use crate::OptimizerConfig;
use oximedia_core::OxiResult;

/// Lookahead frame analysis.
#[derive(Debug, Clone)]
pub struct LookaheadFrame {
    /// Frame index.
    pub frame_idx: usize,
    /// Temporal complexity.
    pub complexity: f64,
    /// Scene change score.
    pub scene_change_score: f64,
    /// Average motion magnitude.
    pub avg_motion: f64,
    /// Whether this is likely a scene change.
    pub is_scene_change: bool,
}

impl LookaheadFrame {
    /// Creates a new lookahead frame.
    #[must_use]
    pub fn new(frame_idx: usize) -> Self {
        Self {
            frame_idx,
            complexity: 0.0,
            scene_change_score: 0.0,
            avg_motion: 0.0,
            is_scene_change: false,
        }
    }
}

/// Lookahead analyzer.
pub struct LookaheadAnalyzer {
    #[allow(dead_code)]
    buffer_size: usize,
    scene_change_threshold: f64,
    enable_complexity_analysis: bool,
}

impl LookaheadAnalyzer {
    /// Creates a new lookahead analyzer.
    pub fn new(config: &OptimizerConfig) -> OxiResult<Self> {
        Ok(Self {
            buffer_size: config.lookahead_frames,
            scene_change_threshold: 0.4,
            enable_complexity_analysis: true,
        })
    }

    /// Analyzes a sequence of frames.
    #[allow(dead_code)]
    #[must_use]
    pub fn analyze(&self, frames: &[&[u8]], width: usize, height: usize) -> Vec<LookaheadFrame> {
        let mut results = Vec::new();

        for (idx, &frame) in frames.iter().enumerate() {
            let mut analysis = LookaheadFrame::new(idx);

            if self.enable_complexity_analysis {
                analysis.complexity = self.calculate_complexity(frame);
            }

            // Scene change detection
            if idx > 0 {
                let prev_frame = frames[idx - 1];
                analysis.scene_change_score =
                    self.calculate_scene_change(prev_frame, frame, width, height);
                analysis.is_scene_change =
                    analysis.scene_change_score > self.scene_change_threshold;
            }

            // Motion estimation (simplified)
            if idx > 0 {
                let prev_frame = frames[idx - 1];
                analysis.avg_motion = self.estimate_motion(prev_frame, frame, width, height);
            }

            results.push(analysis);
        }

        results
    }

    fn calculate_complexity(&self, frame: &[u8]) -> f64 {
        if frame.is_empty() {
            return 0.0;
        }

        // Calculate variance as complexity metric
        let mean = frame.iter().map(|&p| f64::from(p)).sum::<f64>() / frame.len() as f64;
        frame
            .iter()
            .map(|&p| {
                let diff = f64::from(p) - mean;
                diff * diff
            })
            .sum::<f64>()
            / frame.len() as f64
    }

    fn calculate_scene_change(&self, prev: &[u8], curr: &[u8], width: usize, height: usize) -> f64 {
        if prev.len() != curr.len() || prev.is_empty() {
            return 0.0;
        }

        // Calculate SAD between frames
        let sad: u64 = prev
            .iter()
            .zip(curr)
            .map(|(&p, &c)| u64::from(p.abs_diff(c)))
            .sum();

        let pixels = (width * height) as f64;

        sad as f64 / (pixels * 255.0)
    }

    fn estimate_motion(&self, prev: &[u8], curr: &[u8], width: usize, height: usize) -> f64 {
        // Simplified motion estimation using block-based SAD
        const BLOCK_SIZE: usize = 16;
        let blocks_x = width / BLOCK_SIZE;
        let blocks_y = height / BLOCK_SIZE;

        if blocks_x == 0 || blocks_y == 0 {
            return 0.0;
        }

        let mut total_motion = 0.0;
        let mut block_count = 0;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let motion = self.estimate_block_motion(
                    prev,
                    curr,
                    width,
                    bx * BLOCK_SIZE,
                    by * BLOCK_SIZE,
                    BLOCK_SIZE,
                );
                total_motion += motion;
                block_count += 1;
            }
        }

        if block_count > 0 {
            total_motion / f64::from(block_count)
        } else {
            0.0
        }
    }

    fn estimate_block_motion(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: usize,
        block_x: usize,
        block_y: usize,
        block_size: usize,
    ) -> f64 {
        // Simplified: just calculate SAD for the block
        let mut sad = 0u32;

        for y in 0..block_size {
            for x in 0..block_size {
                let px = block_x + x;
                let py = block_y + y;
                let idx = py * width + px;

                if idx < prev.len() {
                    sad += u32::from(prev[idx].abs_diff(curr[idx]));
                }
            }
        }

        f64::from(sad) / (block_size * block_size) as f64
    }

    /// Determines optimal GOP structure based on analysis.
    #[allow(dead_code)]
    #[must_use]
    pub fn determine_gop_structure(&self, analysis: &[LookaheadFrame]) -> GopStructure {
        let mut keyframe_positions = Vec::new();

        for frame in analysis {
            if frame.is_scene_change {
                keyframe_positions.push(frame.frame_idx);
            }
        }

        // Add first frame if not already included
        if keyframe_positions.first() != Some(&0) {
            keyframe_positions.insert(0, 0);
        }

        GopStructure {
            keyframe_positions,
            total_frames: analysis.len(),
        }
    }

    /// Calculates bit allocation for frames.
    #[allow(dead_code)]
    #[must_use]
    pub fn allocate_bits(&self, analysis: &[LookaheadFrame], total_bits: u64) -> Vec<u64> {
        if analysis.is_empty() {
            return Vec::new();
        }

        // Calculate total complexity
        let total_complexity: f64 = analysis.iter().map(|f| f.complexity + 1.0).sum();

        // Allocate bits proportional to complexity
        analysis
            .iter()
            .map(|f| {
                let proportion = (f.complexity + 1.0) / total_complexity;
                (total_bits as f64 * proportion) as u64
            })
            .collect()
    }
}

/// GOP (Group of Pictures) structure.
#[derive(Debug, Clone)]
pub struct GopStructure {
    /// Positions of keyframes.
    pub keyframe_positions: Vec<usize>,
    /// Total number of frames.
    pub total_frames: usize,
}

impl GopStructure {
    /// Gets the distance to next keyframe from a position.
    #[must_use]
    pub fn distance_to_next_keyframe(&self, position: usize) -> usize {
        for &kf_pos in &self.keyframe_positions {
            if kf_pos > position {
                return kf_pos - position;
            }
        }
        self.total_frames - position
    }

    /// Checks if a position is a keyframe.
    #[must_use]
    pub fn is_keyframe(&self, position: usize) -> bool {
        self.keyframe_positions.contains(&position)
    }
}

// ── Scene-cut-aware QP curve (enhanced) ──────────────────────────────────────

/// Configuration for the enhanced scene-cut-aware QP curve.
///
/// Controls how the encoder adjusts quantisation around detected hard cuts.
#[derive(Debug, Clone)]
pub struct SceneCutQpConfig {
    /// QP increase applied to frames immediately before a cut.
    ///
    /// Bits spent at the tail of a dying scene carry no temporal value, so
    /// spending fewer bits there is acceptable. Default: `3`.
    pub pre_cut_boost: i32,

    /// QP decrease applied to the I-frame that opens the new scene.
    ///
    /// The viewer's eye is drawn to fresh content; better quality at the cut
    /// frame pays a perceptual dividend that later P-frames can propagate.
    /// Default: `5`.
    pub post_cut_reduction: i32,

    /// Number of frames over which the QP gradient is smoothed.
    ///
    /// The boost/reduction is linearly interpolated over this many frames so
    /// that the QP curve does not jump discontinuously. Default: `8`.
    pub gradient_frames: usize,
}

impl Default for SceneCutQpConfig {
    fn default() -> Self {
        Self {
            pre_cut_boost: 3,
            post_cut_reduction: 5,
            gradient_frames: 8,
        }
    }
}

/// Apply a scene-cut-aware QP curve to a sequence of base QP values.
///
/// Given the indices of detected scene cuts within the sequence, this function
/// adjusts the QP per frame:
///
/// - **Pre-cut ramp-up**: frames approaching a cut receive a linearly
///   increasing boost toward `cfg.pre_cut_boost`, spread over
///   `cfg.gradient_frames` frames.
/// - **Cut frame (I-frame)**: QP is *reduced* by `cfg.post_cut_reduction`
///   (better quality) because the viewer's attention resets at a cut.
/// - **Post-cut recovery**: QP recovers linearly back to the base value over
///   `cfg.gradient_frames` frames after the cut.
///
/// Overlapping windows from adjacent cuts are combined by taking the maximum
/// pre-cut offset and the minimum post-cut offset, then summing.  All
/// resulting values are clamped to `[base_qp - 12, base_qp + 8]` (a wider
/// asymmetric window to accommodate a large post-cut quality boost).
///
/// # Parameters
///
/// - `base_qp`    — per-frame baseline QP values (typically constant or from
///   a first-pass rate controller).
/// - `scene_cuts` — sorted frame indices at which a hard scene cut begins
///   (the I-frame position).
/// - `cfg`        — tuning parameters.
///
/// # Returns
///
/// A `Vec<i32>` of the same length as `base_qp` with adjusted values.
#[must_use]
pub fn apply_scene_cut_qp_curve(
    base_qp: &[i32],
    scene_cuts: &[usize],
    cfg: &SceneCutQpConfig,
) -> Vec<i32> {
    let n = base_qp.len();
    if n == 0 {
        return Vec::new();
    }

    // Per-frame offset accumulators: split pre- and post-cut so we can
    // combine overlapping windows correctly.
    let mut pre_offsets = vec![0i32; n]; // positive → more bits saved pre-cut
    let mut post_offsets = vec![0i32; n]; // negative → better quality post-cut

    for &cut in scene_cuts {
        if cut >= n {
            continue;
        }

        // ── Pre-cut ramp ────────────────────────────────────────────────────
        // Frames [cut - gradient_frames .. cut) ramp from 0 to pre_cut_boost.
        let pre_len = cfg.gradient_frames.min(cut);
        for k in 0..pre_len {
            // frame index of this pre-cut frame
            let fi = cut - pre_len + k;
            // linear ramp: 0 at k=0, pre_cut_boost at k=pre_len-1 (one before cut)
            let offset = if pre_len > 1 {
                (cfg.pre_cut_boost * (k as i32 + 1)) / pre_len as i32
            } else {
                cfg.pre_cut_boost
            };
            if offset > pre_offsets[fi] {
                pre_offsets[fi] = offset;
            }
        }

        // ── Cut frame (I-frame) ─────────────────────────────────────────────
        // Reduce QP by post_cut_reduction (capped later via clamp).
        let i_offset = -cfg.post_cut_reduction;
        if i_offset < post_offsets[cut] {
            post_offsets[cut] = i_offset;
        }

        // ── Post-cut recovery ramp ──────────────────────────────────────────
        // Frames [cut+1 .. cut+gradient_frames) gradually recover to base.
        let post_len = cfg.gradient_frames.min(n - cut.saturating_add(1));
        for k in 0..post_len {
            let fi = cut + 1 + k;
            if fi >= n {
                break;
            }
            // ramp from -post_cut_reduction+1 → 0 over post_len frames
            let offset = if post_len > 0 {
                -cfg.post_cut_reduction
                    + (cfg.post_cut_reduction * (k as i32 + 1)) / post_len as i32
            } else {
                0
            };
            // Take the most negative (highest quality boost wins)
            if offset < post_offsets[fi] {
                post_offsets[fi] = offset;
            }
        }
    }

    // Combine pre + post offsets and apply to base_qp with clamping.
    // Asymmetric window: large quality boost (post-cut) is valuable.
    base_qp
        .iter()
        .enumerate()
        .map(|(i, &bq)| {
            let combined = pre_offsets[i] + post_offsets[i];
            (bq + combined).clamp(bq - 12, bq + 8)
        })
        .collect()
}

// ── Complexity-based QP adjustment ───────────────────────────────────────────

/// Configuration for the complexity-based QP adjustment algorithm.
#[derive(Debug, Clone)]
pub struct ComplexityQpConfig {
    /// Baseline QP around which adjustments are anchored.
    pub base_qp: i32,

    /// Scale factor controlling how much the complexity score shifts the QP.
    ///
    /// A complexity score of `1.0` produces a shift of `complexity_scale`
    /// fractions of the QP range.  Default: `0.5`.
    pub complexity_scale: f32,

    /// Minimum allowed QP (best quality).  Default: `18`.
    pub min_qp: i32,

    /// Maximum allowed QP (worst quality).  Default: `51`.
    pub max_qp: i32,
}

impl Default for ComplexityQpConfig {
    fn default() -> Self {
        Self {
            base_qp: 28,
            complexity_scale: 0.5,
            min_qp: 18,
            max_qp: 51,
        }
    }
}

/// Estimate combined spatial + temporal complexity of a frame.
///
/// Returns a normalised score in `[0.0, 1.0]` where `0.0` means a completely
/// still and uniform frame and `1.0` means extremely complex / high-motion.
///
/// # Spatial activity
///
/// Edge density is estimated by computing horizontal and vertical first-order
/// gradients (Prewitt-style) over every pixel and counting the fraction that
/// exceed a fixed threshold.
///
/// # Temporal activity
///
/// When `prev` is `Some`, the mean absolute difference (MAD) between `curr`
/// and `prev` (sampled at every fourth pixel for speed) is normalised by the
/// full `[0, 255]` range.
///
/// Both components are combined with equal weight.
///
/// # Parameters
///
/// - `curr` — current frame luma bytes, `w × h` pixels.
/// - `prev` — optional previous frame luma bytes of the same dimensions.
/// - `w`, `h` — frame width and height in pixels.
#[must_use]
pub fn estimate_frame_complexity(curr: &[u8], prev: Option<&[u8]>, w: u32, h: u32) -> f32 {
    let wu = w as usize;
    let hu = h as usize;

    // ── Spatial activity: edge density via first-order gradients ─────────────
    let spatial = if wu < 2 || hu < 2 || curr.len() < wu * hu {
        0.0f32
    } else {
        let edge_threshold = 20u32; // gradient magnitude threshold (0-255 scale)
        let mut edge_count = 0u32;
        let total = ((wu - 1) * (hu - 1)) as f32;
        for y in 0..hu - 1 {
            for x in 0..wu - 1 {
                let idx = y * wu + x;
                // horizontal gradient
                let gx = u32::from(curr[idx + 1].abs_diff(curr[idx]));
                // vertical gradient
                let gy = u32::from(curr[idx + wu].abs_diff(curr[idx]));
                // Use L1 norm as a fast magnitude proxy
                if gx + gy > edge_threshold {
                    edge_count += 1;
                }
            }
        }
        (edge_count as f32 / total).min(1.0)
    };

    // ── Temporal activity: sub-sampled MAD ──────────────────────────────────
    let temporal = match prev {
        None => 0.0f32,
        Some(prev_data) => {
            if prev_data.len() != curr.len() || curr.is_empty() {
                0.0
            } else {
                // Sample every 4th pixel to keep O(n/4) cost
                let step = 4usize;
                let mut sad = 0u64;
                let mut count = 0u64;
                let mut i = 0usize;
                while i < curr.len() {
                    sad += u64::from(curr[i].abs_diff(prev_data[i]));
                    count += 1;
                    i += step;
                }
                if count > 0 {
                    (sad as f32 / (count as f32 * 255.0)).min(1.0)
                } else {
                    0.0
                }
            }
        }
    };

    // Equal blend of spatial and temporal activity.
    ((spatial + temporal) * 0.5).min(1.0)
}

/// Convert a complexity score in `[0.0, 1.0]` to a QP delta.
///
/// The mapping is:
///
/// - complexity `0.0` → QP delta = `min_qp - base_qp` (maximum quality boost
///   for simple content)
/// - complexity `0.5` → QP delta = `0` (no adjustment)
/// - complexity `1.0` → QP delta = `max_qp - base_qp` (maximum quality
///   reduction for complex content)
///
/// The raw delta is further scaled by `cfg.complexity_scale` and clamped to
/// `[cfg.min_qp - cfg.base_qp, cfg.max_qp - cfg.base_qp]`.
#[must_use]
pub fn complexity_to_qp_delta(complexity: f32, cfg: &ComplexityQpConfig) -> i32 {
    // Map [0, 1] → [-1, +1] so that 0.5 complexity → 0 delta
    let normalized = (complexity.clamp(0.0, 1.0) * 2.0 - 1.0) * cfg.complexity_scale;

    // Scale by half the QP range (symmetric around base_qp)
    let half_range = ((cfg.max_qp - cfg.min_qp) as f32 * 0.5).max(1.0);
    let raw_delta = (normalized * half_range).round() as i32;

    // Clamp to the usable window
    let lo = cfg.min_qp - cfg.base_qp;
    let hi = cfg.max_qp - cfg.base_qp;
    raw_delta.clamp(lo, hi)
}

// ── Scene-cut-aware QP curve ──────────────────────────────────────────────────

/// Compute the normalised inter-frame SAD (0.0–1.0) between two Y-plane buffers.
///
/// Returns 0.0 for empty inputs or mismatched lengths.
fn frame_sad_normalised(prev: &[u8], curr: &[u8], width: usize, height: usize) -> f64 {
    if prev.is_empty() || prev.len() != curr.len() {
        return 0.0;
    }
    let sad: u64 = prev
        .iter()
        .zip(curr.iter())
        .map(|(&p, &c)| u64::from(p.abs_diff(c)))
        .sum();
    let pixels = (width * height) as f64;
    if pixels <= 0.0 {
        return 0.0;
    }
    sad as f64 / (pixels * 255.0)
}

/// Scene-cut detection threshold (normalised SAD).
///
/// Frames with an inter-frame SAD exceeding this fraction of the full dynamic
/// range are classified as hard scene cuts.
const SCENE_CUT_THRESHOLD: f64 = 0.30;

/// Generate per-frame QP values with scene-cut-aware adjustment.
///
/// When a scene cut is detected ahead of frame `i` (at frame `i + k`), the
/// frames approaching the cut receive a gradual QP boost (bits at the dying
/// scene's tail have no carry-over value).  The first two frames of the new
/// scene also receive a QP boost (transition masking).
///
/// # Algorithm
///
/// 1. Compute normalised inter-frame SAD between each consecutive pair within
///    the lookahead window.
/// 2. Declare a scene cut when SAD > `SCENE_CUT_THRESHOLD`.
/// 3. **Pre-cut ramp**: up to `min(4, k)` frames before the cut receive a
///    linearly increasing QP boost (max +3 immediately before the cut).
/// 4. **Post-cut frames**: first two frames of the new scene get +2 then +1.
/// 5. All per-frame QP values are clamped to `[base_qp - 5, base_qp + 5]`.
///
/// # Parameters
///
/// - `luma_frames`      — Y-plane data per frame; each must be `width × height` bytes.
/// - `width` / `height` — frame dimensions in pixels.
/// - `base_qp`          — baseline QP; returned for smooth sequences.
/// - `lookahead_frames` — how many frames ahead to consider (typical 8–16).
///
/// Returns `Vec<i32>` of length `luma_frames.len()`.
#[must_use]
pub fn scene_cut_aware_qp_curve(
    luma_frames: &[&[u8]],
    width: usize,
    height: usize,
    base_qp: i32,
    lookahead_frames: usize,
) -> Vec<i32> {
    let n = luma_frames.len();
    if n == 0 {
        return Vec::new();
    }

    // Step 1: detect scene cuts within the lookahead window.
    // is_cut[i] == true means a cut *starts* at frame i (between frame i-1 and i).
    let mut is_cut = vec![false; n];
    let look_limit = n.min(lookahead_frames.saturating_add(1));
    for i in 1..look_limit {
        let score = frame_sad_normalised(luma_frames[i - 1], luma_frames[i], width, height);
        if score > SCENE_CUT_THRESHOLD {
            is_cut[i] = true;
        }
    }

    // Step 2: build per-frame QP offsets.
    let mut qp_offsets = vec![0i32; n];

    for cut_pos in 0..n {
        if !is_cut[cut_pos] {
            continue;
        }

        // Pre-cut ramp: up to min(4, cut_pos) frames immediately before the cut.
        let ramp_len = 4.min(cut_pos);
        for offset in 1..=ramp_len {
            let frame_idx = cut_pos - offset;
            // Frame immediately before the cut gets +3; earlier frames get less.
            let ramp_boost = (3 * (ramp_len + 1 - offset))
                .checked_div(ramp_len)
                .unwrap_or(1)
                .max(1) as i32;
            qp_offsets[frame_idx] = qp_offsets[frame_idx].max(ramp_boost);
        }

        // Post-cut masking: first two frames of the new scene.
        qp_offsets[cut_pos] = qp_offsets[cut_pos].max(2);
        if cut_pos + 1 < n {
            qp_offsets[cut_pos + 1] = qp_offsets[cut_pos + 1].max(1);
        }
    }

    // Step 3: apply base QP and clamp.
    let lo = base_qp - 5;
    let hi = base_qp + 5;
    qp_offsets
        .iter()
        .map(|&offset| (base_qp + offset).clamp(lo, hi))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookahead_frame_creation() {
        let frame = LookaheadFrame::new(42);
        assert_eq!(frame.frame_idx, 42);
        assert_eq!(frame.complexity, 0.0);
        assert!(!frame.is_scene_change);
    }

    #[test]
    fn test_lookahead_analyzer_creation() {
        let config = OptimizerConfig::default();
        let analyzer =
            LookaheadAnalyzer::new(&config).expect("lookahead analyzer creation should succeed");
        assert_eq!(analyzer.buffer_size, config.lookahead_frames);
    }

    #[test]
    fn test_complexity_calculation() {
        let config = OptimizerConfig::default();
        let analyzer =
            LookaheadAnalyzer::new(&config).expect("lookahead analyzer creation should succeed");

        let flat = vec![128u8; 256];
        let complexity_flat = analyzer.calculate_complexity(&flat);
        assert_eq!(complexity_flat, 0.0);

        let varied: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let complexity_varied = analyzer.calculate_complexity(&varied);
        assert!(complexity_varied > 0.0);
    }

    #[test]
    fn test_scene_change_detection() {
        let config = OptimizerConfig::default();
        let analyzer =
            LookaheadAnalyzer::new(&config).expect("lookahead analyzer creation should succeed");

        let frame1 = vec![100u8; 256];
        let frame2 = vec![100u8; 256]; // Same
        let score_same = analyzer.calculate_scene_change(&frame1, &frame2, 16, 16);
        assert_eq!(score_same, 0.0);

        let frame3 = vec![200u8; 256]; // Different
        let score_diff = analyzer.calculate_scene_change(&frame1, &frame3, 16, 16);
        assert!(score_diff > 0.0);
    }

    #[test]
    fn test_gop_structure() {
        let gop = GopStructure {
            keyframe_positions: vec![0, 10, 20],
            total_frames: 30,
        };

        assert!(gop.is_keyframe(0));
        assert!(gop.is_keyframe(10));
        assert!(!gop.is_keyframe(5));

        assert_eq!(gop.distance_to_next_keyframe(0), 10);
        assert_eq!(gop.distance_to_next_keyframe(5), 5);
        assert_eq!(gop.distance_to_next_keyframe(25), 5);
    }

    #[test]
    fn test_bit_allocation() {
        let config = OptimizerConfig::default();
        let analyzer =
            LookaheadAnalyzer::new(&config).expect("lookahead analyzer creation should succeed");

        let analysis = vec![
            LookaheadFrame {
                frame_idx: 0,
                complexity: 100.0,
                scene_change_score: 0.0,
                avg_motion: 0.0,
                is_scene_change: false,
            },
            LookaheadFrame {
                frame_idx: 1,
                complexity: 200.0,
                scene_change_score: 0.0,
                avg_motion: 0.0,
                is_scene_change: false,
            },
        ];

        let allocation = analyzer.allocate_bits(&analysis, 1000);
        assert_eq!(allocation.len(), 2);
        assert!(allocation[1] > allocation[0]); // Higher complexity gets more bits
    }

    // ── scene_cut_aware_qp_curve tests ────────────────────────────────────────

    #[test]
    fn test_scene_cut_qp_curve_empty() {
        let qps = scene_cut_aware_qp_curve(&[], 16, 16, 26, 8);
        assert!(qps.is_empty());
    }

    #[test]
    fn test_scene_cut_no_boost_when_no_cut() {
        // Smooth pan: all frames identical → SAD == 0 → no QP boost.
        let frame = vec![128u8; 64 * 64];
        let frames: Vec<&[u8]> = std::iter::repeat_n(frame.as_slice(), 10).collect();
        let qps = scene_cut_aware_qp_curve(&frames, 64, 64, 26, 8);

        assert_eq!(qps.len(), 10);
        for (i, &qp) in qps.iter().enumerate() {
            assert_eq!(qp, 26, "frame {i}: expected base QP 26, got {qp}");
        }
    }

    #[test]
    fn test_scene_cut_qp_boost_before_cut() {
        // Frames 0-4 = dark scene (value 50), frame 5+ = bright scene (value 200).
        // The SAD at frame 5 should be large enough to trigger a cut.
        let dark = vec![50u8; 64 * 64];
        let bright = vec![200u8; 64 * 64];

        let mut frames_data: Vec<Vec<u8>> = (0..5).map(|_| dark.clone()).collect();
        frames_data.push(bright.clone());
        frames_data.push(bright.clone());
        frames_data.push(bright.clone());

        let frames: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();
        let base_qp = 26;
        let qps = scene_cut_aware_qp_curve(&frames, 64, 64, base_qp, 8);

        assert_eq!(qps.len(), 8);

        // Frames just before the cut (frame 4) should have QP > base.
        assert!(
            qps[4] > base_qp,
            "frame 4 (just before cut) should have QP > {base_qp}, got {}",
            qps[4]
        );

        // The new scene's first frame (frame 5) should have elevated QP.
        assert!(
            qps[5] >= base_qp,
            "frame 5 (start of new scene) should have QP >= {base_qp}, got {}",
            qps[5]
        );
    }

    #[test]
    fn test_scene_cut_qp_clamp_within_five() {
        // Even with large QP boosts, all values must stay within base ±5.
        let dark = vec![0u8; 16 * 16];
        let bright = vec![255u8; 16 * 16];
        let frames_data: Vec<Vec<u8>> = (0..12)
            .map(|i| if i < 6 { dark.clone() } else { bright.clone() })
            .collect();
        let frames: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();
        let base_qp = 30;
        let qps = scene_cut_aware_qp_curve(&frames, 16, 16, base_qp, 10);

        for (i, &qp) in qps.iter().enumerate() {
            assert!(
                qp >= base_qp - 5 && qp <= base_qp + 5,
                "frame {i}: QP {qp} outside [{}, {}]",
                base_qp - 5,
                base_qp + 5
            );
        }
    }

    #[test]
    fn test_scene_cut_false_positive_resilience() {
        // Slow fade: luma changes from 100 to 130 over 8 frames.
        // Per-frame delta ≈ 30/7 ≈ 4.3 luma codes.
        // Normalised SAD ≈ 4.3 / 255 ≈ 0.017 << SCENE_CUT_THRESHOLD (0.30).
        let frames_data: Vec<Vec<u8>> = (0..8)
            .map(|i| {
                let luma = (100 + i * 30 / 7) as u8;
                vec![luma; 64 * 64]
            })
            .collect();
        let frames: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();
        let base_qp = 26;
        let qps = scene_cut_aware_qp_curve(&frames, 64, 64, base_qp, 8);

        for (i, &qp) in qps.iter().enumerate() {
            assert_eq!(
                qp, base_qp,
                "slow fade frame {i}: expected {base_qp}, got {qp}"
            );
        }
    }

    #[test]
    fn test_scene_cut_single_frame_no_panic() {
        let frame = vec![128u8; 32 * 32];
        let frames = [frame.as_slice()];
        let qps = scene_cut_aware_qp_curve(&frames, 32, 32, 26, 4);
        assert_eq!(qps.len(), 1);
        assert_eq!(qps[0], 26);
    }

    // ── apply_scene_cut_qp_curve tests ───────────────────────────────────────

    #[test]
    fn test_qp_curve_no_cuts() {
        // No scene cuts → output must equal input exactly.
        let base: Vec<i32> = (20..30).collect();
        let result = apply_scene_cut_qp_curve(&base, &[], &SceneCutQpConfig::default());
        assert_eq!(result, base, "no cuts: output should equal input");
    }

    #[test]
    fn test_qp_curve_post_cut_reduction() {
        // A cut at frame 5 → QP at frame 5 should be reduced relative to base.
        let base_qp = 28;
        let n = 15;
        let base: Vec<i32> = vec![base_qp; n];
        let cfg = SceneCutQpConfig {
            pre_cut_boost: 3,
            post_cut_reduction: 5,
            gradient_frames: 4,
        };
        let result = apply_scene_cut_qp_curve(&base, &[5], &cfg);

        assert_eq!(result.len(), n);
        assert!(
            result[5] < base_qp,
            "frame 5 (I-frame after cut) should have QP < base {base_qp}, got {}",
            result[5]
        );
        let expected_cut_qp = (base_qp - cfg.post_cut_reduction).max(base_qp - 12);
        assert_eq!(
            result[5], expected_cut_qp,
            "I-frame QP should be base - post_cut_reduction"
        );
    }

    #[test]
    fn test_qp_curve_pre_cut_boost() {
        // A cut at frame 8 → frames in the gradient window before frame 8 should
        // have QP elevated above the base.
        let base_qp = 26;
        let n = 16;
        let base: Vec<i32> = vec![base_qp; n];
        let cfg = SceneCutQpConfig {
            pre_cut_boost: 3,
            post_cut_reduction: 5,
            gradient_frames: 4,
        };
        let result = apply_scene_cut_qp_curve(&base, &[8], &cfg);

        // Frames 4..8 are within the pre-cut window (gradient_frames=4).
        // The frame immediately before the cut (frame 7) should have the highest boost.
        assert!(
            result[7] > base_qp,
            "frame 7 (1 before cut at 8) should have QP > {base_qp}, got {}",
            result[7]
        );
        // At least one frame in [4,7] should be boosted
        let any_boosted = result[4..8].iter().any(|&q| q > base_qp);
        assert!(any_boosted, "some pre-cut frame should have elevated QP");
    }

    #[test]
    fn test_qp_curve_gradient_smooth() {
        // The QP curve should never jump by more than `pre_cut_boost` between
        // adjacent frames in a smooth pre-cut region — enforced by the linear ramp.
        let base_qp = 28;
        let n = 20;
        let base: Vec<i32> = vec![base_qp; n];
        let cfg = SceneCutQpConfig {
            pre_cut_boost: 3,
            post_cut_reduction: 5,
            gradient_frames: 8,
        };
        let result = apply_scene_cut_qp_curve(&base, &[12], &cfg);

        // Check that consecutive frames differ by at most pre_cut_boost (3)
        // in the pre-cut ramp region [4..12].
        for i in 4..11usize {
            let delta = (result[i + 1] - result[i]).abs();
            assert!(
                delta <= cfg.pre_cut_boost,
                "frames {i}→{}: QP jumped by {delta} > {}",
                i + 1,
                cfg.pre_cut_boost
            );
        }
    }

    #[test]
    fn test_qp_curve_clamp_respected() {
        // Even with extreme configs the output must stay within [base-12, base+8].
        let base_qp = 30;
        let n = 10;
        let base = vec![base_qp; n];
        let cfg = SceneCutQpConfig {
            pre_cut_boost: 20,
            post_cut_reduction: 20,
            gradient_frames: 8,
        };
        let result = apply_scene_cut_qp_curve(&base, &[5], &cfg);
        for (i, &qp) in result.iter().enumerate() {
            assert!(
                qp >= base_qp - 12 && qp <= base_qp + 8,
                "frame {i}: QP {qp} outside [{}, {}]",
                base_qp - 12,
                base_qp + 8
            );
        }
    }

    #[test]
    fn test_qp_curve_empty_base() {
        let result = apply_scene_cut_qp_curve(&[], &[0], &SceneCutQpConfig::default());
        assert!(result.is_empty());
    }

    // ── estimate_frame_complexity tests ──────────────────────────────────────

    #[test]
    fn test_complexity_uniform() {
        // A completely uniform (flat) frame has no edges and zero temporal diff.
        let frame = vec![128u8; 64 * 64];
        let complexity = estimate_frame_complexity(&frame, None, 64, 64);
        assert!(
            complexity < 0.05,
            "uniform frame complexity should be near 0, got {complexity}"
        );
    }

    #[test]
    fn test_complexity_high_motion() {
        // Random noise vs. a static frame → high temporal + spatial complexity.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let prev = vec![128u8; 64 * 64];
        // Generate pseudo-random frame
        let noise: Vec<u8> = (0u64..64 * 64)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                (h.finish() % 256) as u8
            })
            .collect();
        let complexity = estimate_frame_complexity(&noise, Some(&prev), 64, 64);
        assert!(
            complexity > 0.5,
            "noise vs static should have complexity > 0.5, got {complexity}"
        );
    }

    #[test]
    fn test_complexity_same_frame() {
        // Comparing a frame to itself: temporal diff = 0, spatial from content.
        let frame = vec![100u8; 32 * 32];
        let c = estimate_frame_complexity(&frame, Some(&frame), 32, 32);
        // A flat frame with no motion → should still be low
        assert!(c < 0.1, "flat frame vs itself complexity={c}");
    }

    #[test]
    fn test_complexity_no_prev() {
        // Without a previous frame only spatial is used; result in [0,1].
        let frame: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let c = estimate_frame_complexity(&frame, None, 16, 16);
        assert!(c >= 0.0 && c <= 1.0, "complexity={c} not in [0,1]");
    }

    // ── complexity_to_qp_delta tests ─────────────────────────────────────────

    #[test]
    fn test_qp_delta_range() {
        // For all complexity values in [0,1] the QP delta must stay within
        // [min_qp - base_qp, max_qp - base_qp].
        let cfg = ComplexityQpConfig::default();
        let lo = cfg.min_qp - cfg.base_qp;
        let hi = cfg.max_qp - cfg.base_qp;

        let steps = 101usize;
        for k in 0..steps {
            let complexity = k as f32 / (steps - 1) as f32;
            let delta = complexity_to_qp_delta(complexity, &cfg);
            assert!(
                delta >= lo && delta <= hi,
                "complexity={complexity:.2}: delta {delta} not in [{lo}, {hi}]"
            );
        }
    }

    #[test]
    fn test_qp_delta_midpoint_near_zero() {
        // complexity = 0.5 → delta should be 0 (or very close after rounding).
        let cfg = ComplexityQpConfig::default();
        let delta = complexity_to_qp_delta(0.5, &cfg);
        assert!(
            delta.abs() <= 1,
            "midpoint complexity should give delta near 0, got {delta}"
        );
    }

    #[test]
    fn test_qp_delta_monotone() {
        // Higher complexity → higher (or equal) QP delta.
        let cfg = ComplexityQpConfig::default();
        let mut prev_delta = i32::MIN;
        for k in 0..=20usize {
            let complexity = k as f32 / 20.0;
            let delta = complexity_to_qp_delta(complexity, &cfg);
            assert!(
                delta >= prev_delta,
                "non-monotone at complexity={:.2}: prev={prev_delta}, cur={delta}",
                complexity
            );
            prev_delta = delta;
        }
    }

    #[test]
    fn test_qp_delta_low_complexity_negative() {
        // complexity near 0 → negative delta (quality boost for simple content).
        let cfg = ComplexityQpConfig::default();
        let delta = complexity_to_qp_delta(0.0, &cfg);
        assert!(
            delta < 0,
            "zero complexity should yield negative QP delta, got {delta}"
        );
    }

    #[test]
    fn test_qp_delta_high_complexity_positive() {
        // complexity near 1 → positive delta (quality reduction for complex content).
        let cfg = ComplexityQpConfig::default();
        let delta = complexity_to_qp_delta(1.0, &cfg);
        assert!(
            delta > 0,
            "max complexity should yield positive QP delta, got {delta}"
        );
    }
}
