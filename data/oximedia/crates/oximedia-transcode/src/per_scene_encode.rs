//! Per-scene adaptive encoding parameters.
//!
//! This module provides scene-aware encoding parameter computation,
//! output size estimation, bitrate solving via binary search, and
//! budget allocation across multiple scenes.

use serde::{Deserialize, Serialize};

// ─── SceneType ────────────────────────────────────────────────────────────────

/// Classification of a scene's content type for encoding optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneType {
    /// Mostly still content — logos, title cards, pause frames.
    Static,
    /// Slow panning or gentle motion.
    SlowMotion,
    /// Fast action, sports, chase sequences.
    ActionFast,
    /// Talking-head content, interviews, news.
    Talking,
    /// End credits or scrolling text.
    Credits,
    /// Animated content — cartoons, CGI.
    Animation,
    /// High spatial and temporal complexity.
    HighComplexity,
}

// ─── SceneSegment ─────────────────────────────────────────────────────────────

/// A contiguous segment of frames identified as a distinct scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSegment {
    /// Index of the first frame (inclusive).
    pub start_frame: u64,
    /// Index of the last frame (inclusive).
    pub end_frame: u64,
    /// Number of frames in this segment.
    pub duration_frames: u32,
    /// Normalised motion score in \[0.0, 1.0\].
    pub motion_score: f32,
    /// Normalised spatial complexity in \[0.0, 1.0\].
    pub complexity: f32,
    /// Whether the scene is predominantly dark (low average luma).
    pub is_dark: bool,
    /// Content-type classification.
    pub scene_type: SceneType,
}

impl SceneSegment {
    /// Constructs a new `SceneSegment` with explicit fields.
    #[must_use]
    pub fn new(
        start_frame: u64,
        end_frame: u64,
        duration_frames: u32,
        motion_score: f32,
        complexity: f32,
        is_dark: bool,
        scene_type: SceneType,
    ) -> Self {
        Self {
            start_frame,
            end_frame,
            duration_frames,
            motion_score,
            complexity,
            is_dark,
            scene_type,
        }
    }

    /// Returns the number of frames (alias for clarity).
    #[must_use]
    pub fn frame_count(&self) -> u32 {
        self.duration_frames
    }

    /// Returns the duration in seconds at the given frame rate.
    #[must_use]
    pub fn duration_secs(&self, fps: f32) -> f32 {
        if fps <= 0.0 {
            return 0.0;
        }
        self.duration_frames as f32 / fps
    }
}

// ─── SceneEncodeParams ────────────────────────────────────────────────────────

/// Per-scene encoding parameters derived from scene analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEncodeParams {
    /// Constant Rate Factor (lower = higher quality, larger file).
    pub crf: u8,
    /// Average bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Peak / maximum bitrate in kbps.
    pub max_bitrate_kbps: u32,
    /// Number of consecutive B-frames.
    pub b_frames: u8,
    /// Number of reference frames.
    pub ref_frames: u8,
    /// Encoder speed/quality preset string (e.g., "medium", "slow").
    pub preset: String,
    /// Group-of-pictures size in frames.
    pub gop_size: u32,
    /// Log2 of tile columns for parallel encoding.
    pub tile_cols: u8,
    /// Log2 of tile rows for parallel encoding.
    pub tile_rows: u8,
}

impl SceneEncodeParams {
    /// Returns true when the params appear valid for submission to an encoder.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.bitrate_kbps > 0
            && self.max_bitrate_kbps >= self.bitrate_kbps
            && !self.preset.is_empty()
            && self.gop_size > 0
    }
}

// ─── PerSceneEncoder ──────────────────────────────────────────────────────────

/// Computes codec-specific encoding parameters tuned for a given scene.
#[derive(Debug, Clone, Default)]
pub struct PerSceneEncoder;

impl PerSceneEncoder {
    /// Creates a new `PerSceneEncoder`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Derives optimal `SceneEncodeParams` for `scene` at the given codec and
    /// target average bitrate.
    ///
    /// # Rules applied
    ///
    /// | Scene characteristic | Adjustment |
    /// |---|---|
    /// | `Static` | CRF −2, fewer B-frames, larger GOP |
    /// | `ActionFast` | CRF +2, B-frames=0, smaller GOP |
    /// | Dark scene | CRF −3 (less noise in shadows) |
    /// | `HighComplexity` | CRF +3, more B-frames |
    /// | AV1 | tile\_cols=4, tile\_rows=2 |
    /// | VP9 | tile\_cols=2, tile\_rows=1 |
    /// | H.265/HEVC | tile\_cols=2, tile\_rows=1 |
    #[must_use]
    pub fn compute_params(
        &self,
        scene: &SceneSegment,
        target_bitrate_kbps: u32,
        codec: &str,
    ) -> SceneEncodeParams {
        // Base values per codec
        let (base_crf, base_b_frames, base_ref_frames, base_preset, base_gop) =
            Self::codec_base_values(codec);

        // Accumulate CRF adjustments
        let crf_adj = self.crf_adjustment(scene);

        // Clamp CRF to [0, 51]
        let raw_crf = base_crf as i32 + crf_adj;
        let crf = raw_crf.clamp(0, 51) as u8;

        // B-frames / GOP adjustments per scene type
        let (b_frames, gop_mult) = self.motion_params(scene, base_b_frames);

        let gop_size = ((base_gop as f32) * gop_mult).round() as u32;
        let gop_size = gop_size.max(1);

        // Bitrate: scale by complexity and dark-scene headroom
        let bitrate_scale = self.bitrate_scale(scene);
        let bitrate_kbps = ((target_bitrate_kbps as f32) * bitrate_scale).round() as u32;
        let bitrate_kbps = bitrate_kbps.max(50);
        let max_bitrate_kbps = (bitrate_kbps as f32 * 2.0).round() as u32;

        // Tile layout per codec
        let (tile_cols, tile_rows) = Self::tile_layout(codec);

        SceneEncodeParams {
            crf,
            bitrate_kbps,
            max_bitrate_kbps,
            b_frames,
            ref_frames: base_ref_frames,
            preset: base_preset.to_string(),
            gop_size,
            tile_cols,
            tile_rows,
        }
    }

    /// Base encoding values per codec.
    fn codec_base_values(codec: &str) -> (u8, u8, u8, &'static str, u32) {
        // (base_crf, base_b_frames, base_ref_frames, preset, gop_size)
        match codec.to_lowercase().as_str() {
            "av1" | "libaom-av1" | "svt-av1" => (35, 0, 3, "5", 240),
            "vp9" | "libvpx-vp9" => (33, 0, 3, "good", 240),
            "h265" | "hevc" | "libx265" => (28, 4, 4, "medium", 250),
            "h264" | "avc" | "libx264" => (23, 3, 3, "medium", 250),
            _ => (28, 2, 3, "medium", 250),
        }
    }

    /// CRF delta based on scene properties.
    fn crf_adjustment(&self, scene: &SceneSegment) -> i32 {
        let mut adj: i32 = 0;

        match scene.scene_type {
            SceneType::Static => adj -= 2,
            SceneType::ActionFast => adj += 2,
            SceneType::HighComplexity => adj += 3,
            SceneType::Animation => adj -= 1,
            SceneType::Credits => adj -= 2,
            SceneType::SlowMotion => adj -= 1,
            SceneType::Talking => {}
        }

        if scene.is_dark {
            adj -= 3;
        }

        // High spatial complexity: slightly raise CRF to keep bitrate sane
        if scene.complexity > 0.8 {
            adj += 1;
        } else if scene.complexity < 0.2 {
            adj -= 1;
        }

        adj
    }

    /// Returns (b_frames, gop_multiplier) based on motion characteristics.
    fn motion_params(&self, scene: &SceneSegment, base_b_frames: u8) -> (u8, f32) {
        match scene.scene_type {
            SceneType::Static => {
                // Static: fewer B-frames is fine, larger GOP to save bits
                let b = base_b_frames.saturating_sub(1);
                (b, 2.0)
            }
            SceneType::ActionFast => {
                // Fast action: no B-frames, small GOP for random-access
                (0, 0.5)
            }
            SceneType::HighComplexity => {
                // Complex: moderate B-frames, normal GOP
                let b = (base_b_frames + 2).min(8);
                (b, 1.0)
            }
            SceneType::Credits => {
                // Scrolling text: larger GOP, fewer B-frames
                let b = base_b_frames.saturating_sub(1);
                (b, 1.5)
            }
            SceneType::Animation => {
                // Animation: larger GOP, normal B-frames
                (base_b_frames, 1.5)
            }
            SceneType::SlowMotion | SceneType::Talking => (base_b_frames, 1.0),
        }
    }

    /// Bitrate scaling factor \[0.5, 2.0\] based on scene complexity.
    fn bitrate_scale(&self, scene: &SceneSegment) -> f32 {
        let mut scale = 0.5 + scene.complexity * 1.5; // [0.5, 2.0]
        if scene.is_dark {
            // Dark scenes need less bitrate for equivalent visual quality
            scale *= 0.85;
        }
        match scene.scene_type {
            SceneType::Static => scale *= 0.6,
            SceneType::ActionFast => scale *= 1.4,
            SceneType::HighComplexity => scale *= 1.5,
            _ => {}
        }
        scale.clamp(0.3, 3.0)
    }

    /// Tile layout (tile_cols, tile_rows) per codec.
    fn tile_layout(codec: &str) -> (u8, u8) {
        match codec.to_lowercase().as_str() {
            "av1" | "libaom-av1" | "svt-av1" => (4, 2),
            "vp9" | "libvpx-vp9" => (2, 1),
            "h265" | "hevc" | "libx265" => (2, 1),
            _ => (1, 1),
        }
    }
}

// ─── Size estimation ──────────────────────────────────────────────────────────

/// Estimates output file size in bytes for a segment encoded with given params.
///
/// Formula: `bitrate_kbps × 1000 / 8 × duration_seconds × fill_factor`
/// where `fill_factor = 0.9` (90% utilisation).
#[must_use]
pub fn estimate_output_size(params: &SceneEncodeParams, duration_frames: u32, fps: f32) -> u64 {
    if fps <= 0.0 || duration_frames == 0 {
        return 0;
    }
    let duration_secs = duration_frames as f64 / fps as f64;
    let bytes_per_sec = params.bitrate_kbps as f64 * 1000.0 / 8.0;
    let raw_bytes = bytes_per_sec * duration_secs;
    (raw_bytes * 0.9) as u64
}

// ─── TargetSizeSolver ─────────────────────────────────────────────────────────

/// Solves for the average bitrate (kbps) that achieves a target file size
/// using binary search over the fill-factor-adjusted bitrate formula.
#[derive(Debug, Clone)]
pub struct TargetSizeSolver {
    /// Target output size in bytes.
    pub target_bytes: u64,
    /// Number of frames in the segment.
    pub duration_frames: u32,
    /// Frame rate.
    pub fps: f32,
}

impl TargetSizeSolver {
    /// Creates a new solver.
    #[must_use]
    pub fn new(target_bytes: u64, duration_frames: u32, fps: f32) -> Self {
        Self {
            target_bytes,
            duration_frames,
            fps,
        }
    }

    /// Performs a binary search to find the bitrate (kbps) that fills
    /// `target_bytes` at the given `complexity_factor` \[0.5, 2.0\].
    ///
    /// The complexity factor scales the effective fill factor (higher
    /// complexity → content is harder to compress → more bits needed per
    /// unit of quality), meaning the solver converges to a slightly higher
    /// bitrate for complex content.
    #[must_use]
    pub fn solve_bitrate(&self, complexity_factor: f32) -> u32 {
        if self.fps <= 0.0 || self.duration_frames == 0 || self.target_bytes == 0 {
            return 0;
        }

        let duration_secs = self.duration_frames as f64 / self.fps as f64;
        // Effective fill factor: 0.9 base, modulated by complexity
        let fill = (0.9 / complexity_factor.max(0.1) as f64).clamp(0.3, 1.5);
        // target_bytes = bitrate_bps / 8 * duration * fill
        // bitrate_bps = target_bytes * 8 / duration / fill
        let bitrate_bps = (self.target_bytes as f64 * 8.0) / (duration_secs * fill);
        let bitrate_kbps = (bitrate_bps / 1000.0).round() as u32;
        bitrate_kbps.max(1)
    }
}

// ─── BudgetAllocator ──────────────────────────────────────────────────────────

/// Allocates a total byte budget proportionally across scenes,
/// weighted by each scene's complexity.
#[derive(Debug, Clone)]
pub struct BudgetAllocator {
    /// Total available byte budget.
    pub total_budget_bytes: u64,
    /// Scenes to allocate budget across.
    pub scenes: Vec<SceneSegment>,
}

impl BudgetAllocator {
    /// Creates a new allocator.
    #[must_use]
    pub fn new(total_budget_bytes: u64, scenes: Vec<SceneSegment>) -> Self {
        Self {
            total_budget_bytes,
            scenes,
        }
    }

    /// Returns per-scene byte budgets, summing to at most `total_budget_bytes`.
    ///
    /// Allocation weight for scene `i`:
    ///   `w_i = duration_i × (0.5 + complexity_i)`
    ///
    /// Each scene receives `budget_i = total × w_i / Σw`.
    #[must_use]
    pub fn allocate(&self) -> Vec<u64> {
        if self.scenes.is_empty() {
            return Vec::new();
        }

        let weights: Vec<f64> = self
            .scenes
            .iter()
            .map(|s| {
                let duration_weight = s.duration_frames as f64;
                let complexity_weight = 0.5 + s.complexity as f64;
                let motion_weight = 1.0 + s.motion_score as f64 * 0.5;
                duration_weight * complexity_weight * motion_weight
            })
            .collect();

        let total_weight: f64 = weights.iter().sum();

        if total_weight <= 0.0 {
            // Uniform allocation fallback
            let per_scene = self.total_budget_bytes / self.scenes.len() as u64;
            return vec![per_scene; self.scenes.len()];
        }

        let mut allocations: Vec<u64> = weights
            .iter()
            .map(|&w| {
                let frac = w / total_weight;
                (self.total_budget_bytes as f64 * frac).round() as u64
            })
            .collect();

        // Correct rounding error: ensure sum <= total_budget_bytes
        let allocated_sum: u64 = allocations.iter().sum();
        if allocated_sum > self.total_budget_bytes {
            // Trim the largest bucket
            if let Some(max_idx) = allocations
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i)
            {
                let excess = allocated_sum - self.total_budget_bytes;
                allocations[max_idx] = allocations[max_idx].saturating_sub(excess);
            }
        }

        allocations
    }

    /// Returns the total bytes allocated (should be ≤ `total_budget_bytes`).
    #[must_use]
    pub fn allocated_total(&self) -> u64 {
        self.allocate().iter().sum()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scene(
        scene_type: SceneType,
        complexity: f32,
        motion: f32,
        is_dark: bool,
    ) -> SceneSegment {
        SceneSegment::new(0, 239, 240, motion, complexity, is_dark, scene_type)
    }

    // ── SceneSegment ─────────────────────────────────────────────────────────

    #[test]
    fn test_scene_segment_new() {
        let seg = SceneSegment::new(0, 299, 300, 0.3, 0.5, false, SceneType::Talking);
        assert_eq!(seg.start_frame, 0);
        assert_eq!(seg.end_frame, 299);
        assert_eq!(seg.duration_frames, 300);
        assert!((seg.motion_score - 0.3).abs() < 1e-6);
        assert!(!seg.is_dark);
    }

    #[test]
    fn test_scene_segment_duration_secs() {
        let seg = make_scene(SceneType::Talking, 0.5, 0.3, false);
        let dur = seg.duration_secs(30.0);
        assert!((dur - 8.0).abs() < 0.01); // 240 / 30 = 8 s
    }

    #[test]
    fn test_scene_segment_duration_secs_zero_fps() {
        let seg = make_scene(SceneType::Talking, 0.5, 0.3, false);
        assert_eq!(seg.duration_secs(0.0), 0.0);
    }

    #[test]
    fn test_frame_count_alias() {
        let seg = make_scene(SceneType::Static, 0.1, 0.0, false);
        assert_eq!(seg.frame_count(), seg.duration_frames);
    }

    // ── PerSceneEncoder ──────────────────────────────────────────────────────

    #[test]
    fn test_static_scene_lower_crf_larger_gop() {
        let enc = PerSceneEncoder::new();
        let static_scene = make_scene(SceneType::Static, 0.3, 0.1, false);
        let action_scene = make_scene(SceneType::ActionFast, 0.3, 0.9, false);
        let p_static = enc.compute_params(&static_scene, 4000, "h264");
        let p_action = enc.compute_params(&action_scene, 4000, "h264");
        assert!(
            p_static.crf < p_action.crf,
            "Static CRF should be lower than ActionFast CRF"
        );
        assert!(
            p_static.gop_size > p_action.gop_size,
            "Static GOP should be larger"
        );
    }

    #[test]
    fn test_action_fast_no_b_frames() {
        let enc = PerSceneEncoder::new();
        let scene = make_scene(SceneType::ActionFast, 0.8, 0.95, false);
        let params = enc.compute_params(&scene, 8000, "h264");
        assert_eq!(params.b_frames, 0, "ActionFast should use no B-frames");
    }

    #[test]
    fn test_dark_scene_lower_crf() {
        let enc = PerSceneEncoder::new();
        let dark = make_scene(SceneType::Talking, 0.5, 0.3, true);
        let bright = make_scene(SceneType::Talking, 0.5, 0.3, false);
        let p_dark = enc.compute_params(&dark, 4000, "h264");
        let p_bright = enc.compute_params(&bright, 4000, "h264");
        assert!(
            p_dark.crf < p_bright.crf,
            "Dark scene should have lower CRF"
        );
    }

    #[test]
    fn test_av1_tile_layout() {
        let enc = PerSceneEncoder::new();
        let scene = make_scene(SceneType::Talking, 0.5, 0.3, false);
        let params = enc.compute_params(&scene, 4000, "av1");
        assert_eq!(params.tile_cols, 4);
        assert_eq!(params.tile_rows, 2);
    }

    #[test]
    fn test_vp9_tile_layout() {
        let enc = PerSceneEncoder::new();
        let scene = make_scene(SceneType::Talking, 0.5, 0.3, false);
        let params = enc.compute_params(&scene, 4000, "vp9");
        assert_eq!(params.tile_cols, 2);
        assert_eq!(params.tile_rows, 1);
    }

    #[test]
    fn test_h265_tile_layout() {
        let enc = PerSceneEncoder::new();
        let scene = make_scene(SceneType::Talking, 0.5, 0.3, false);
        let params = enc.compute_params(&scene, 4000, "h265");
        assert_eq!(params.tile_cols, 2);
        assert_eq!(params.tile_rows, 1);
    }

    #[test]
    fn test_params_are_valid() {
        let enc = PerSceneEncoder::new();
        let scene = make_scene(SceneType::Talking, 0.5, 0.3, false);
        let params = enc.compute_params(&scene, 4000, "vp9");
        assert!(params.is_valid());
    }

    #[test]
    fn test_max_bitrate_gte_avg_bitrate() {
        let enc = PerSceneEncoder::new();
        let scene = make_scene(SceneType::HighComplexity, 0.9, 0.8, false);
        let params = enc.compute_params(&scene, 6000, "av1");
        assert!(params.max_bitrate_kbps >= params.bitrate_kbps);
    }

    #[test]
    fn test_crf_clamped_to_valid_range() {
        let enc = PerSceneEncoder::new();
        // Dark + HighComplexity compensate each other; result must still be [0,51]
        let scene = make_scene(SceneType::HighComplexity, 0.9, 0.9, true);
        let params = enc.compute_params(&scene, 2000, "h264");
        assert!(params.crf <= 51);
    }

    // ── estimate_output_size ──────────────────────────────────────────────────

    #[test]
    fn test_estimate_output_size_basic() {
        let params = SceneEncodeParams {
            crf: 28,
            bitrate_kbps: 1000,
            max_bitrate_kbps: 2000,
            b_frames: 3,
            ref_frames: 3,
            preset: "medium".to_string(),
            gop_size: 250,
            tile_cols: 1,
            tile_rows: 1,
        };
        // 1000 kbps × 1000/8 = 125 000 B/s × 10 s × 0.9 = 1 125 000 bytes
        let size = estimate_output_size(&params, 300, 30.0);
        assert!((size as i64 - 1_125_000).abs() < 1000);
    }

    #[test]
    fn test_estimate_output_size_zero_fps() {
        let params = SceneEncodeParams {
            crf: 28,
            bitrate_kbps: 1000,
            max_bitrate_kbps: 2000,
            b_frames: 3,
            ref_frames: 3,
            preset: "medium".to_string(),
            gop_size: 250,
            tile_cols: 1,
            tile_rows: 1,
        };
        assert_eq!(estimate_output_size(&params, 300, 0.0), 0);
    }

    #[test]
    fn test_estimate_output_size_zero_frames() {
        let params = SceneEncodeParams {
            crf: 28,
            bitrate_kbps: 1000,
            max_bitrate_kbps: 2000,
            b_frames: 3,
            ref_frames: 3,
            preset: "medium".to_string(),
            gop_size: 250,
            tile_cols: 1,
            tile_rows: 1,
        };
        assert_eq!(estimate_output_size(&params, 0, 30.0), 0);
    }

    // ── TargetSizeSolver ──────────────────────────────────────────────────────

    #[test]
    fn test_target_size_solver_basic() {
        // 10 MB target, 300 frames at 30 fps = 10 s
        let solver = TargetSizeSolver::new(10_000_000, 300, 30.0);
        let kbps = solver.solve_bitrate(1.0);
        // Expected: 10_000_000 * 8 / 10 / 0.9 ≈ 8_889_000 bps ≈ 8889 kbps
        assert!(kbps > 7000 && kbps < 10000, "kbps={kbps} out of range");
    }

    #[test]
    fn test_target_size_solver_high_complexity() {
        let solver = TargetSizeSolver::new(10_000_000, 300, 30.0);
        let kbps_normal = solver.solve_bitrate(1.0);
        let kbps_complex = solver.solve_bitrate(2.0);
        // Higher complexity → higher bitrate needed
        assert!(kbps_complex > kbps_normal);
    }

    #[test]
    fn test_target_size_solver_zero_target() {
        let solver = TargetSizeSolver::new(0, 300, 30.0);
        assert_eq!(solver.solve_bitrate(1.0), 0);
    }

    #[test]
    fn test_target_size_solver_zero_frames() {
        let solver = TargetSizeSolver::new(10_000_000, 0, 30.0);
        assert_eq!(solver.solve_bitrate(1.0), 0);
    }

    // ── BudgetAllocator ───────────────────────────────────────────────────────

    #[test]
    fn test_budget_allocator_sums_to_budget() {
        let scenes = vec![
            make_scene(SceneType::Static, 0.2, 0.1, false),
            make_scene(SceneType::Talking, 0.5, 0.4, false),
            make_scene(SceneType::ActionFast, 0.9, 0.95, false),
        ];
        let allocator = BudgetAllocator::new(100_000_000, scenes);
        let allocs = allocator.allocate();
        let total: u64 = allocs.iter().sum();
        assert!(total <= 100_000_000, "total={total} exceeded budget");
    }

    #[test]
    fn test_budget_allocator_complex_scene_gets_more() {
        let scenes = vec![
            make_scene(SceneType::Static, 0.1, 0.1, false),
            make_scene(SceneType::HighComplexity, 0.95, 0.95, false),
        ];
        let allocator = BudgetAllocator::new(100_000_000, scenes);
        let allocs = allocator.allocate();
        assert!(
            allocs[1] > allocs[0],
            "Complex scene should receive more budget"
        );
    }

    #[test]
    fn test_budget_allocator_empty_scenes() {
        let allocator = BudgetAllocator::new(100_000_000, vec![]);
        assert!(allocator.allocate().is_empty());
    }

    #[test]
    fn test_budget_allocator_allocated_total() {
        let scenes = vec![
            make_scene(SceneType::Talking, 0.5, 0.4, false),
            make_scene(SceneType::Animation, 0.6, 0.2, false),
        ];
        let allocator = BudgetAllocator::new(50_000_000, scenes);
        let total = allocator.allocated_total();
        assert!(total <= 50_000_000);
    }

    #[test]
    fn test_budget_allocator_single_scene() {
        let scenes = vec![make_scene(SceneType::Talking, 0.5, 0.3, false)];
        let allocator = BudgetAllocator::new(20_000_000, scenes);
        let allocs = allocator.allocate();
        assert_eq!(allocs.len(), 1);
        assert!(allocs[0] <= 20_000_000);
    }

    #[test]
    fn test_scene_type_equality() {
        assert_eq!(SceneType::Static, SceneType::Static);
        assert_ne!(SceneType::Static, SceneType::ActionFast);
    }

    #[test]
    fn test_encode_params_preset_not_empty() {
        let enc = PerSceneEncoder::new();
        let scene = make_scene(SceneType::Talking, 0.5, 0.3, false);
        for codec in &["av1", "vp9", "h265", "h264"] {
            let p = enc.compute_params(&scene, 4000, codec);
            assert!(!p.preset.is_empty(), "preset empty for codec {codec}");
        }
    }
}
