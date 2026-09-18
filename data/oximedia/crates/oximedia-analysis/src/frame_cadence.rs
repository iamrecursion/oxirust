#![allow(dead_code)]
//! Frame cadence and pulldown detection analysis.
//!
//! Detects telecine patterns (3:2, 2:2, 2:3:3:2), duplicate/dropped frames,
//! and irregular frame cadence in video content. This is essential for
//! inverse telecine (IVTC), standards conversion QC, and identifying
//! encoding issues.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Known cadence patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CadencePattern {
    /// Progressive (no pulldown) — every frame is unique.
    Progressive,
    /// 3:2 pulldown (NTSC telecine from 24fps film).
    Pulldown32,
    /// 2:2 pulldown (PAL telecine, or interlaced from 25fps).
    Pulldown22,
    /// 2:3:3:2 advanced pulldown pattern.
    Pulldown2332,
    /// Irregular / mixed cadence.
    Irregular,
}

impl std::fmt::Display for CadencePattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Progressive => write!(f, "Progressive"),
            Self::Pulldown32 => write!(f, "3:2 Pulldown"),
            Self::Pulldown22 => write!(f, "2:2 Pulldown"),
            Self::Pulldown2332 => write!(f, "2:3:3:2 Pulldown"),
            Self::Irregular => write!(f, "Irregular"),
        }
    }
}

/// Describes a single frame in the cadence analysis.
#[derive(Debug, Clone, Copy)]
pub struct FrameCadenceInfo {
    /// Frame index.
    pub index: usize,
    /// Similarity to the previous frame (0.0 = totally different, 1.0 = identical).
    pub similarity: f64,
    /// Whether this frame is considered a duplicate of the previous.
    pub is_duplicate: bool,
    /// Whether this frame appears to be a drop (abnormal gap).
    pub is_dropped: bool,
}

/// Result of a duplicate-frame run.
#[derive(Debug, Clone, Copy)]
pub struct DuplicateRun {
    /// Start frame index.
    pub start: usize,
    /// End frame index (inclusive).
    pub end: usize,
    /// Number of consecutive duplicate frames.
    pub length: usize,
}

/// Complete cadence analysis result.
#[derive(Debug, Clone)]
pub struct CadenceAnalysisResult {
    /// Total frames analysed.
    pub total_frames: usize,
    /// Detected dominant cadence pattern.
    pub dominant_pattern: CadencePattern,
    /// Confidence in the detected pattern (0.0..1.0).
    pub confidence: f64,
    /// Per-frame information.
    pub frame_info: Vec<FrameCadenceInfo>,
    /// Detected runs of duplicate frames.
    pub duplicate_runs: Vec<DuplicateRun>,
    /// Total duplicate frames.
    pub total_duplicates: usize,
    /// Total detected dropped frames.
    pub total_drops: usize,
    /// Ratio of unique frames to total frames.
    pub unique_ratio: f64,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for cadence analysis.
#[derive(Debug, Clone)]
pub struct CadenceConfig {
    /// Similarity threshold above which two frames are considered duplicates.
    pub duplicate_threshold: f64,
    /// Minimum run length to be reported as a duplicate run.
    pub min_run_length: usize,
    /// Minimum number of frames to attempt pattern detection.
    pub min_pattern_frames: usize,
}

impl Default for CadenceConfig {
    fn default() -> Self {
        Self {
            duplicate_threshold: 0.98,
            min_run_length: 1,
            min_pattern_frames: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// Analyzer
// ---------------------------------------------------------------------------

/// Stateful cadence analyzer.
#[derive(Debug)]
pub struct CadenceAnalyzer {
    /// Configuration.
    config: CadenceConfig,
    /// Per-frame records.
    frames: Vec<FrameCadenceInfo>,
    /// Previous frame Y-plane summary (mean luminance per block).
    prev_block_means: Option<Vec<f64>>,
    /// Grid dimensions for block-based comparison.
    grid_cols: usize,
    /// Grid rows.
    grid_rows: usize,
}

impl CadenceAnalyzer {
    /// Create a new cadence analyzer.
    pub fn new(config: CadenceConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
            prev_block_means: None,
            grid_cols: 8,
            grid_rows: 6,
        }
    }

    /// Feed a Y-plane frame.
    pub fn push_frame(&mut self, y_plane: &[u8], width: usize, height: usize) {
        let block_means =
            compute_block_means(y_plane, width, height, self.grid_cols, self.grid_rows);
        let (similarity, is_dup) = if let Some(ref prev) = self.prev_block_means {
            let sim = block_similarity(prev, &block_means);
            (sim, sim >= self.config.duplicate_threshold)
        } else {
            (0.0, false)
        };

        let index = self.frames.len();
        self.frames.push(FrameCadenceInfo {
            index,
            similarity,
            is_duplicate: is_dup,
            is_dropped: false, // resolved in finalize
        });
        self.prev_block_means = Some(block_means);
    }

    /// Push a pre-computed similarity value (useful when frame data is unavailable).
    pub fn push_similarity(&mut self, similarity: f64) {
        let is_dup = similarity >= self.config.duplicate_threshold;
        let index = self.frames.len();
        self.frames.push(FrameCadenceInfo {
            index,
            similarity,
            is_duplicate: is_dup,
            is_dropped: false,
        });
    }

    /// Finalize and return the analysis result.
    pub fn finalize(mut self) -> CadenceAnalysisResult {
        let total = self.frames.len();
        if total == 0 {
            return CadenceAnalysisResult {
                total_frames: 0,
                dominant_pattern: CadencePattern::Progressive,
                confidence: 0.0,
                frame_info: Vec::new(),
                duplicate_runs: Vec::new(),
                total_duplicates: 0,
                total_drops: 0,
                unique_ratio: 1.0,
            };
        }

        // Mark dropped frames: abnormally low similarity after high-similarity frames
        detect_drops(&mut self.frames);

        let dup_runs = find_duplicate_runs(&self.frames, self.config.min_run_length);
        let total_dups = self.frames.iter().filter(|f| f.is_duplicate).count();
        let total_drops = self.frames.iter().filter(|f| f.is_dropped).count();

        #[allow(clippy::cast_precision_loss)]
        let unique_ratio = if total > 0 {
            (total - total_dups) as f64 / total as f64
        } else {
            1.0
        };

        let (pattern, confidence) = detect_pattern(&self.frames, self.config.min_pattern_frames);

        CadenceAnalysisResult {
            total_frames: total,
            dominant_pattern: pattern,
            confidence,
            frame_info: self.frames,
            duplicate_runs: dup_runs,
            total_duplicates: total_dups,
            total_drops,
            unique_ratio,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute mean luminance for each block in a grid.
fn compute_block_means(
    y_plane: &[u8],
    width: usize,
    height: usize,
    cols: usize,
    rows: usize,
) -> Vec<f64> {
    if width == 0 || height == 0 || cols == 0 || rows == 0 {
        return vec![0.0; cols * rows];
    }
    let bw = width / cols;
    let bh = height / rows;
    let mut means = Vec::with_capacity(cols * rows);
    for r in 0..rows {
        for c in 0..cols {
            let mut sum = 0u64;
            let mut count = 0u64;
            let y0 = r * bh;
            let y1 = if r == rows - 1 { height } else { y0 + bh };
            let x0 = c * bw;
            let x1 = if c == cols - 1 { width } else { x0 + bw };
            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = y * width + x;
                    if idx < y_plane.len() {
                        sum += u64::from(y_plane[idx]);
                        count += 1;
                    }
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let mean = if count > 0 {
                sum as f64 / count as f64
            } else {
                0.0
            };
            means.push(mean);
        }
    }
    means
}

/// Cosine-similarity-like metric between two block-mean vectors.
fn block_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        return if norm_a < 1e-12 && norm_b < 1e-12 {
            1.0
        } else {
            0.0
        };
    }
    (dot / denom).clamp(0.0, 1.0)
}

/// Mark frames that appear to be drops (sudden low similarity surrounded by high).
fn detect_drops(frames: &mut [FrameCadenceInfo]) {
    if frames.len() < 3 {
        return;
    }
    for i in 1..frames.len() - 1 {
        let prev_sim = frames[i - 1].similarity;
        let next_sim = frames[i + 1].similarity;
        let cur_sim = frames[i].similarity;
        // A "drop" manifests as a sudden dip in similarity
        if cur_sim < 0.5 && prev_sim > 0.8 && next_sim > 0.8 {
            frames[i].is_dropped = true;
        }
    }
}

/// Find runs of consecutive duplicate frames.
fn find_duplicate_runs(frames: &[FrameCadenceInfo], min_run: usize) -> Vec<DuplicateRun> {
    let mut runs = Vec::new();
    let mut run_start: Option<usize> = None;
    for (i, f) in frames.iter().enumerate() {
        if f.is_duplicate {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(start) = run_start {
            let len = i - start;
            if len >= min_run {
                runs.push(DuplicateRun {
                    start,
                    end: i - 1,
                    length: len,
                });
            }
            run_start = None;
        }
    }
    if let Some(start) = run_start {
        let len = frames.len() - start;
        if len >= min_run {
            runs.push(DuplicateRun {
                start,
                end: frames.len() - 1,
                length: len,
            });
        }
    }
    runs
}

/// Attempt to match the duplicate/unique pattern to a known cadence.
fn detect_pattern(frames: &[FrameCadenceInfo], min_frames: usize) -> (CadencePattern, f64) {
    if frames.len() < min_frames {
        return (CadencePattern::Irregular, 0.0);
    }

    // Build a binary string: D = duplicate, U = unique
    let pattern_str: Vec<bool> = frames.iter().map(|f| f.is_duplicate).collect();
    let total_dups = pattern_str.iter().filter(|&&d| d).count();

    if total_dups == 0 {
        return (CadencePattern::Progressive, 1.0);
    }

    // Test 3:2 pattern (period 5): DDUUD or similar rotations
    let score_32 = test_periodic_pattern(&pattern_str, &[true, false, false, true, false]);
    // Test 2:2 pattern (period 4): DUDU or similar
    let score_22 = test_periodic_pattern(&pattern_str, &[true, false, true, false]);
    // Test 2:3:3:2 pattern (period 10): DDUUUDDUUU variants
    let score_2332 = test_periodic_pattern(
        &pattern_str,
        &[
            true, false, false, false, true, true, false, false, false, true,
        ],
    );

    let mut best = (CadencePattern::Irregular, 0.0f64);
    if score_32 > best.1 {
        best = (CadencePattern::Pulldown32, score_32);
    }
    if score_22 > best.1 {
        best = (CadencePattern::Pulldown22, score_22);
    }
    if score_2332 > best.1 {
        best = (CadencePattern::Pulldown2332, score_2332);
    }

    if best.1 < 0.5 {
        best = (CadencePattern::Irregular, best.1);
    }

    best
}

/// Score how well the observed pattern matches a reference periodic template.
/// Returns a match ratio 0.0..1.0 (best rotation is used).
fn test_periodic_pattern(observed: &[bool], template: &[bool]) -> f64 {
    let period = template.len();
    if period == 0 || observed.len() < period {
        return 0.0;
    }
    let mut best_score = 0.0f64;
    for rotation in 0..period {
        let mut matches = 0usize;
        for (i, &obs) in observed.iter().enumerate() {
            let tpl = template[(i + rotation) % period];
            if obs == tpl {
                matches += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let score = matches as f64 / observed.len() as f64;
        if score > best_score {
            best_score = score;
        }
    }
    best_score
}

// ---------------------------------------------------------------------------
// Pulldown Pattern Recognition (public API)
// ---------------------------------------------------------------------------

/// Simplified frame cadence descriptor used for pulldown pattern detection.
///
/// Each entry records whether a frame is a duplicate of the previous one and
/// the measured inter-frame similarity, allowing the caller to feed data
/// from any source (live analysis, cached metadata, etc.).
#[derive(Debug, Clone, Copy)]
pub struct FrameCadence {
    /// Frame index.
    pub index: usize,
    /// Whether this frame was detected as a duplicate of the previous frame.
    pub is_duplicate: bool,
    /// Similarity to the previous frame in the range `[0.0, 1.0]`.
    pub similarity: f64,
}

/// Recognised pulldown (telecine) patterns.
///
/// These describe how film frames at one rate are repeated to produce output
/// at a higher frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PulldownPattern {
    /// No pulldown detected — content appears to be native progressive.
    None,
    /// 3:2 pulldown — NTSC telecine converting 24 fps film to ~29.97 fps.
    /// Every 5-frame group contains 2 duplicated frames in a repeating pattern.
    Pulldown32,
    /// 2:2 pulldown — PAL telecine converting 25 fps film to 50 fields/sec (25 fps
    /// interlaced), or converting 24 fps to 48 fps.  Every other frame is duplicated.
    Pulldown22,
    /// Unrecognised or irregular cadence pattern.
    Unknown,
}

impl std::fmt::Display for PulldownPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None (Progressive)"),
            Self::Pulldown32 => write!(f, "3:2 Pulldown (NTSC Telecine)"),
            Self::Pulldown22 => write!(f, "2:2 Pulldown (PAL Telecine)"),
            Self::Unknown => write!(f, "Unknown / Irregular"),
        }
    }
}

/// Detect the dominant pulldown pattern from a slice of cadence history entries.
///
/// The function examines the `is_duplicate` sequence and attempts to match it
/// against the periodic templates for 3:2 and 2:2 pulldown. A confidence
/// threshold of 0.70 is required for a positive match; below that, the pattern
/// is reported as `PulldownPattern::Unknown`.
///
/// # Arguments
///
/// * `cadence_history` - Ordered slice of `FrameCadence` records, one per frame.
///   Must contain at least 10 entries for reliable detection.
///
/// # Returns
///
/// The dominant `PulldownPattern` for the provided history, or
/// `PulldownPattern::None` when fewer than 10 frames are supplied and no
/// duplicates are found, `PulldownPattern::Unknown` for ambiguous short sequences.
#[must_use]
pub fn detect_pulldown_pattern(cadence_history: &[FrameCadence]) -> PulldownPattern {
    const MIN_FRAMES: usize = 10;
    const CONFIDENCE_THRESHOLD: f64 = 0.70;

    if cadence_history.is_empty() {
        return PulldownPattern::None;
    }

    let total_dups = cadence_history.iter().filter(|f| f.is_duplicate).count();

    // No duplicates at all → progressive content.
    if total_dups == 0 {
        return PulldownPattern::None;
    }

    if cadence_history.len() < MIN_FRAMES {
        // Too short to reliably classify, but there are duplicates.
        return PulldownPattern::Unknown;
    }

    let observed: Vec<bool> = cadence_history.iter().map(|f| f.is_duplicate).collect();

    // 3:2 pulldown: period-5 patterns — 2 duplicates in every 5 frames.
    // The canonical pattern (and all its cyclic rotations) are tested.
    // Template: U U D U D  (D = duplicate, U = unique)
    let score_32 = score_periodic_template(&observed, &[false, false, true, false, true]);

    // 2:2 pulldown: period-4 — every other frame is a duplicate.
    // Template: U D U D
    let score_22 = score_periodic_template(&observed, &[false, true, false, true]);

    // Select best match above threshold.
    if score_32 >= CONFIDENCE_THRESHOLD && score_32 >= score_22 {
        return PulldownPattern::Pulldown32;
    }
    if score_22 >= CONFIDENCE_THRESHOLD {
        return PulldownPattern::Pulldown22;
    }

    // Duplicates exist but no recognised pattern.
    PulldownPattern::Unknown
}

/// Score how well an observed bool sequence matches a periodic template.
///
/// All cyclic rotations of the template are tested and the best score is
/// returned. Score is the fraction of positions that agree with the template,
/// in `[0.0, 1.0]`.
fn score_periodic_template(observed: &[bool], template: &[bool]) -> f64 {
    let period = template.len();
    if period == 0 || observed.is_empty() {
        return 0.0;
    }

    let mut best = 0.0f64;
    for rotation in 0..period {
        let mut matches = 0usize;
        for (i, &obs) in observed.iter().enumerate() {
            if obs == template[(i + rotation) % period] {
                matches += 1;
            }
        }
        let score = matches as f64 / observed.len() as f64;
        if score > best {
            best = score;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Interlacing / Combing Detection
// ---------------------------------------------------------------------------

/// Interlacing status classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterlaceStatus {
    /// Frame is progressive (no combing).
    Progressive,
    /// Frame is interlaced with top-field-first (TFF).
    InterlacedTff,
    /// Frame is interlaced with bottom-field-first (BFF).
    InterlacedBff,
    /// Frame is interlaced, field order unknown.
    InterlacedUnknown,
}

impl std::fmt::Display for InterlaceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Progressive => write!(f, "Progressive"),
            Self::InterlacedTff => write!(f, "Interlaced (TFF)"),
            Self::InterlacedBff => write!(f, "Interlaced (BFF)"),
            Self::InterlacedUnknown => write!(f, "Interlaced"),
        }
    }
}

/// Per-frame combing analysis result.
#[derive(Debug, Clone, Copy)]
pub struct CombingInfo {
    /// Frame index.
    pub index: usize,
    /// Combing metric — high values indicate strong combing artifacts.
    /// Normalised to approximately 0.0..1.0 range.
    pub combing_score: f64,
    /// Whether the frame is classified as combed.
    pub is_combed: bool,
}

/// Configuration for interlacing detection.
#[derive(Debug, Clone)]
pub struct InterlaceConfig {
    /// Combing threshold: per-pixel gradient threshold to count as a combing
    /// artifact. Default 15.
    pub gradient_threshold: i32,
    /// Fraction of scanlines exhibiting combing needed to flag a frame.
    /// Range 0.0-1.0. Default 0.10.
    pub line_ratio_threshold: f64,
    /// Minimum number of frames to analyse before determining status.
    pub min_frames: usize,
}

impl Default for InterlaceConfig {
    fn default() -> Self {
        Self {
            gradient_threshold: 15,
            line_ratio_threshold: 0.10,
            min_frames: 5,
        }
    }
}

/// Aggregated interlacing analysis result.
#[derive(Debug, Clone)]
pub struct InterlaceAnalysisResult {
    /// Dominant interlace status.
    pub status: InterlaceStatus,
    /// Confidence in the detected status (0.0-1.0).
    pub confidence: f64,
    /// Per-frame combing info.
    pub frame_info: Vec<CombingInfo>,
    /// Average combing score across all frames.
    pub avg_combing_score: f64,
    /// Total number of combed frames.
    pub combed_frame_count: usize,
    /// Total frames analysed.
    pub total_frames: usize,
}

/// Stateful interlacing detector.
///
/// Uses a vertical combing metric to detect interlacing artifacts.
/// The algorithm examines adjacent-line luminance differences: in
/// interlaced content the odd and even scanlines come from different
/// temporal fields, producing a characteristic "tooth" pattern in the
/// vertical gradient.
#[derive(Debug)]
pub struct InterlaceDetector {
    config: InterlaceConfig,
    frames: Vec<CombingInfo>,
    /// Store last two frames for field-order detection.
    prev_even_mean: Option<f64>,
    prev_odd_mean: Option<f64>,
    tff_votes: usize,
    bff_votes: usize,
}

impl InterlaceDetector {
    /// Create a new interlacing detector.
    #[must_use]
    pub fn new(config: InterlaceConfig) -> Self {
        Self {
            config,
            frames: Vec::new(),
            prev_even_mean: None,
            prev_odd_mean: None,
            tff_votes: 0,
            bff_votes: 0,
        }
    }

    /// Process a Y-plane frame for combing artifacts.
    pub fn push_frame(&mut self, y_plane: &[u8], width: usize, height: usize) {
        if width == 0 || height < 3 || y_plane.len() < width * height {
            let index = self.frames.len();
            self.frames.push(CombingInfo {
                index,
                combing_score: 0.0,
                is_combed: false,
            });
            return;
        }

        // Compute combing metric:
        // For each pixel (x, y) where y ∈ [1, height-2], compute:
        //   combing = |2*L(x,y) - L(x,y-1) - L(x,y+1)|
        // If combing > gradient_threshold, it is a combing hit.
        // Count the fraction of scanlines with significant combing.

        let gt = self.config.gradient_threshold;
        let mut combed_lines = 0usize;
        let interior_lines = height.saturating_sub(2);
        if interior_lines == 0 {
            let index = self.frames.len();
            self.frames.push(CombingInfo {
                index,
                combing_score: 0.0,
                is_combed: false,
            });
            return;
        }

        let mut total_combing_energy = 0.0f64;
        let mut total_pixels = 0usize;

        for y in 1..height - 1 {
            let mut line_combing = 0usize;
            for x in 0..width {
                let above = i32::from(y_plane[(y - 1) * width + x]);
                let center = i32::from(y_plane[y * width + x]);
                let below = i32::from(y_plane[(y + 1) * width + x]);

                let comb = (2 * center - above - below).abs();
                if comb > gt {
                    line_combing += 1;
                }
                total_combing_energy += comb as f64;
                total_pixels += 1;
            }

            // If more than 30% of pixels in this line show combing, count the line
            if width > 0 && (line_combing as f64 / width as f64) > 0.30 {
                combed_lines += 1;
            }
        }

        let combing_score = if total_pixels > 0 {
            // Normalise: typical max energy per pixel is ~510 (extreme case)
            // A practical normalisation factor: divide by (gradient_threshold * 4)
            let norm = (gt as f64) * 4.0;
            (total_combing_energy / total_pixels as f64 / norm).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let line_ratio = combed_lines as f64 / interior_lines as f64;
        let is_combed = line_ratio >= self.config.line_ratio_threshold;

        // Field-order heuristic: compare even-line mean vs odd-line mean to
        // previous frame's fields. In TFF, even lines (0, 2, 4, ...) come
        // first; in BFF, odd lines come first. Whichever field correlates
        // more with the previous same-parity field is likely the "earlier" field.
        let (even_mean, odd_mean) = compute_field_means(y_plane, width, height);

        if let (Some(prev_even), Some(prev_odd)) = (self.prev_even_mean, self.prev_odd_mean) {
            // If even field changed less than odd, even field is temporally first (TFF)
            let even_diff = (even_mean - prev_even).abs();
            let odd_diff = (odd_mean - prev_odd).abs();
            if is_combed {
                if even_diff < odd_diff {
                    self.tff_votes += 1;
                } else {
                    self.bff_votes += 1;
                }
            }
        }
        self.prev_even_mean = Some(even_mean);
        self.prev_odd_mean = Some(odd_mean);

        let index = self.frames.len();
        self.frames.push(CombingInfo {
            index,
            combing_score,
            is_combed,
        });
    }

    /// Finalize and produce the analysis result.
    #[must_use]
    pub fn finalize(self) -> InterlaceAnalysisResult {
        let total = self.frames.len();
        if total == 0 {
            return InterlaceAnalysisResult {
                status: InterlaceStatus::Progressive,
                confidence: 0.0,
                frame_info: Vec::new(),
                avg_combing_score: 0.0,
                combed_frame_count: 0,
                total_frames: 0,
            };
        }

        let combed_count = self.frames.iter().filter(|f| f.is_combed).count();
        let combed_ratio = combed_count as f64 / total as f64;
        let avg_score = self.frames.iter().map(|f| f.combing_score).sum::<f64>() / total as f64;

        let (status, confidence) = if combed_ratio < 0.05 {
            // Very few combed frames => progressive
            (InterlaceStatus::Progressive, 1.0 - combed_ratio)
        } else if combed_ratio > 0.30 {
            // Clearly interlaced
            let field_order = if self.tff_votes > self.bff_votes + 2 {
                InterlaceStatus::InterlacedTff
            } else if self.bff_votes > self.tff_votes + 2 {
                InterlaceStatus::InterlacedBff
            } else {
                InterlaceStatus::InterlacedUnknown
            };
            (field_order, combed_ratio)
        } else {
            // Ambiguous
            (InterlaceStatus::InterlacedUnknown, combed_ratio)
        };

        InterlaceAnalysisResult {
            status,
            confidence,
            frame_info: self.frames,
            avg_combing_score: avg_score,
            combed_frame_count: combed_count,
            total_frames: total,
        }
    }
}

/// Compute mean luminance of even and odd scanlines.
fn compute_field_means(y_plane: &[u8], width: usize, height: usize) -> (f64, f64) {
    let mut even_sum = 0u64;
    let mut even_count = 0u64;
    let mut odd_sum = 0u64;
    let mut odd_count = 0u64;

    for y in 0..height {
        let row_start = y * width;
        let row_end = row_start + width;
        if row_end > y_plane.len() {
            break;
        }
        let row_sum: u64 = y_plane[row_start..row_end]
            .iter()
            .map(|&p| u64::from(p))
            .sum();
        if y % 2 == 0 {
            even_sum += row_sum;
            even_count += width as u64;
        } else {
            odd_sum += row_sum;
            odd_count += width as u64;
        }
    }

    let even_mean = if even_count > 0 {
        even_sum as f64 / even_count as f64
    } else {
        0.0
    };
    let odd_mean = if odd_count > 0 {
        odd_sum as f64 / odd_count as f64
    } else {
        0.0
    };

    (even_mean, odd_mean)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(width: usize, height: usize, value: u8) -> Vec<u8> {
        vec![value; width * height]
    }

    #[test]
    fn test_cadence_config_defaults() {
        let cfg = CadenceConfig::default();
        assert!((cfg.duplicate_threshold - 0.98).abs() < 0.001);
        assert_eq!(cfg.min_run_length, 1);
    }

    #[test]
    fn test_empty_analyzer() {
        let a = CadenceAnalyzer::new(CadenceConfig::default());
        let result = a.finalize();
        assert_eq!(result.total_frames, 0);
        assert_eq!(result.dominant_pattern, CadencePattern::Progressive);
    }

    #[test]
    fn test_progressive_content() {
        let mut a = CadenceAnalyzer::new(CadenceConfig::default());
        // Each frame has a distinct spatial pattern so cosine similarity between
        // consecutive frames is well below the duplicate threshold.
        for i in 0..20u32 {
            let frame: Vec<u8> = (0..16 * 16)
                .map(|p| {
                    // Create a gradient that shifts with each frame index
                    let col = (p % 16) as u32;
                    let row = (p / 16) as u32;
                    ((col
                        .wrapping_mul(17)
                        .wrapping_add(row.wrapping_mul(13))
                        .wrapping_add(i.wrapping_mul(73)))
                        % 256) as u8
                })
                .collect();
            a.push_frame(&frame, 16, 16);
        }
        let result = a.finalize();
        assert_eq!(result.dominant_pattern, CadencePattern::Progressive);
        assert_eq!(result.total_duplicates, 0);
    }

    #[test]
    fn test_all_duplicates() {
        let mut a = CadenceAnalyzer::new(CadenceConfig::default());
        for _ in 0..10 {
            a.push_frame(&make_frame(16, 16, 128), 16, 16);
        }
        let result = a.finalize();
        // First frame has no "previous" so 9 duplicates
        assert!(result.total_duplicates >= 9);
        assert!(result.unique_ratio < 0.2);
    }

    #[test]
    fn test_push_similarity_api() {
        let mut a = CadenceAnalyzer::new(CadenceConfig::default());
        a.push_similarity(0.0);
        a.push_similarity(0.99);
        a.push_similarity(0.50);
        let result = a.finalize();
        assert_eq!(result.total_frames, 3);
        assert_eq!(result.total_duplicates, 1); // only the 0.99 one
    }

    #[test]
    fn test_duplicate_run_detection() {
        let cfg = CadenceConfig {
            min_run_length: 2,
            ..Default::default()
        };
        let mut a = CadenceAnalyzer::new(cfg);
        // Simulate: unique, dup, dup, dup, unique
        a.push_similarity(0.0);
        a.push_similarity(0.99);
        a.push_similarity(0.99);
        a.push_similarity(0.99);
        a.push_similarity(0.5);
        let result = a.finalize();
        assert!(!result.duplicate_runs.is_empty());
        assert_eq!(result.duplicate_runs[0].length, 3);
    }

    #[test]
    fn test_cadence_pattern_display() {
        assert_eq!(CadencePattern::Progressive.to_string(), "Progressive");
        assert_eq!(CadencePattern::Pulldown32.to_string(), "3:2 Pulldown");
        assert_eq!(CadencePattern::Pulldown22.to_string(), "2:2 Pulldown");
        assert_eq!(CadencePattern::Pulldown2332.to_string(), "2:3:3:2 Pulldown");
        assert_eq!(CadencePattern::Irregular.to_string(), "Irregular");
    }

    #[test]
    fn test_block_similarity_identical() {
        let a = vec![100.0, 200.0, 150.0];
        let sim = block_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_block_similarity_zero_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let sim = block_similarity(&a, &a);
        // Both zero => considered identical
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_periodic_pattern_exact_match() {
        // Build exact 2:2 pattern
        let observed: Vec<bool> = (0..20).map(|i| i % 2 == 0).collect();
        let score = test_periodic_pattern(&observed, &[true, false, true, false]);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_frame_cadence_info_fields() {
        let mut a = CadenceAnalyzer::new(CadenceConfig::default());
        a.push_frame(&make_frame(8, 8, 100), 8, 8);
        a.push_frame(&make_frame(8, 8, 100), 8, 8);
        let result = a.finalize();
        assert_eq!(result.frame_info.len(), 2);
        assert_eq!(result.frame_info[0].index, 0);
        assert!(result.frame_info[1].is_duplicate);
    }

    #[test]
    fn test_drop_detection() {
        let mut a = CadenceAnalyzer::new(CadenceConfig::default());
        // High sim, high sim, sudden dip, high sim, high sim
        a.push_similarity(0.0);
        a.push_similarity(0.95);
        a.push_similarity(0.95);
        a.push_similarity(0.2); // should be detected as drop
        a.push_similarity(0.95);
        a.push_similarity(0.95);
        let result = a.finalize();
        assert!(result.total_drops >= 1);
    }

    #[test]
    fn test_unique_ratio_calculation() {
        let mut a = CadenceAnalyzer::new(CadenceConfig::default());
        // 5 unique + 5 duplicate
        for i in 0..10 {
            if i % 2 == 0 {
                a.push_similarity(0.5); // unique
            } else {
                a.push_similarity(0.99); // duplicate
            }
        }
        let result = a.finalize();
        assert!((result.unique_ratio - 0.5).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Interlacing / combing detection tests
    // -----------------------------------------------------------------------

    /// Create a progressive frame (smooth vertical gradients).
    fn make_progressive_frame(width: usize, height: usize) -> Vec<u8> {
        let mut frame = vec![0u8; width * height];
        for y in 0..height {
            let val = ((y as f64 / height as f64) * 255.0) as u8;
            for x in 0..width {
                frame[y * width + x] = val;
            }
        }
        frame
    }

    /// Create a combed/interlaced frame: even lines get one value,
    /// odd lines get a very different value, simulating temporal field mismatch.
    fn make_combed_frame(width: usize, height: usize) -> Vec<u8> {
        let mut frame = vec![0u8; width * height];
        for y in 0..height {
            let val = if y % 2 == 0 { 200u8 } else { 40u8 };
            for x in 0..width {
                frame[y * width + x] = val;
            }
        }
        frame
    }

    #[test]
    fn test_interlace_config_defaults() {
        let cfg = InterlaceConfig::default();
        assert_eq!(cfg.gradient_threshold, 15);
        assert!((cfg.line_ratio_threshold - 0.10).abs() < 0.001);
        assert_eq!(cfg.min_frames, 5);
    }

    #[test]
    fn test_interlace_progressive_content() {
        let mut det = InterlaceDetector::new(InterlaceConfig::default());
        for _ in 0..20 {
            let frame = make_progressive_frame(64, 64);
            det.push_frame(&frame, 64, 64);
        }
        let result = det.finalize();
        assert_eq!(result.status, InterlaceStatus::Progressive);
        assert!(result.combed_frame_count == 0 || result.avg_combing_score < 0.2);
    }

    #[test]
    fn test_interlace_combed_content() {
        let mut det = InterlaceDetector::new(InterlaceConfig::default());
        for _ in 0..20 {
            let frame = make_combed_frame(64, 64);
            det.push_frame(&frame, 64, 64);
        }
        let result = det.finalize();
        assert_ne!(result.status, InterlaceStatus::Progressive);
        assert!(result.combed_frame_count > 10);
        assert!(result.avg_combing_score > 0.1);
    }

    #[test]
    fn test_interlace_empty() {
        let det = InterlaceDetector::new(InterlaceConfig::default());
        let result = det.finalize();
        assert_eq!(result.total_frames, 0);
        assert_eq!(result.status, InterlaceStatus::Progressive);
    }

    #[test]
    fn test_interlace_status_display() {
        assert_eq!(InterlaceStatus::Progressive.to_string(), "Progressive");
        assert_eq!(
            InterlaceStatus::InterlacedTff.to_string(),
            "Interlaced (TFF)"
        );
        assert_eq!(
            InterlaceStatus::InterlacedBff.to_string(),
            "Interlaced (BFF)"
        );
        assert_eq!(InterlaceStatus::InterlacedUnknown.to_string(), "Interlaced");
    }

    #[test]
    fn test_interlace_tiny_frame() {
        let mut det = InterlaceDetector::new(InterlaceConfig::default());
        // Very small frame — should not panic
        let frame = vec![128u8; 4 * 2];
        det.push_frame(&frame, 4, 2);
        let result = det.finalize();
        assert_eq!(result.total_frames, 1);
    }

    #[test]
    fn test_interlace_combing_score_range() {
        let mut det = InterlaceDetector::new(InterlaceConfig::default());
        let frame = make_combed_frame(32, 32);
        det.push_frame(&frame, 32, 32);
        let result = det.finalize();
        assert!(result.frame_info[0].combing_score >= 0.0);
        assert!(result.frame_info[0].combing_score <= 1.0);
    }

    #[test]
    fn test_field_means_symmetric() {
        let frame = vec![100u8; 64 * 64];
        let (even, odd) = compute_field_means(&frame, 64, 64);
        assert!((even - 100.0).abs() < 0.01);
        assert!((odd - 100.0).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // detect_pulldown_pattern tests
    // -----------------------------------------------------------------------

    fn make_cadence(dups: &[bool]) -> Vec<FrameCadence> {
        dups.iter()
            .enumerate()
            .map(|(i, &d)| FrameCadence {
                index: i,
                is_duplicate: d,
                similarity: if d { 0.99 } else { 0.5 },
            })
            .collect()
    }

    #[test]
    fn test_pulldown_empty() {
        let result = detect_pulldown_pattern(&[]);
        assert_eq!(result, PulldownPattern::None);
    }

    #[test]
    fn test_pulldown_no_duplicates() {
        // All unique frames → None (progressive)
        let cadence = make_cadence(&[false; 20]);
        assert_eq!(detect_pulldown_pattern(&cadence), PulldownPattern::None);
    }

    #[test]
    fn test_pulldown_32_detected() {
        // 3:2 pattern (period 5): U U D U D  (repeated 4 times = 20 frames)
        let pattern = [false, false, true, false, true];
        let dups: Vec<bool> = (0..20).map(|i| pattern[i % 5]).collect();
        let cadence = make_cadence(&dups);
        assert_eq!(
            detect_pulldown_pattern(&cadence),
            PulldownPattern::Pulldown32
        );
    }

    #[test]
    fn test_pulldown_22_detected() {
        // 2:2 pattern (period 4): U D U D  (repeated 5 times = 20 frames)
        let pattern = [false, true, false, true];
        let dups: Vec<bool> = (0..20).map(|i| pattern[i % 4]).collect();
        let cadence = make_cadence(&dups);
        assert_eq!(
            detect_pulldown_pattern(&cadence),
            PulldownPattern::Pulldown22
        );
    }

    #[test]
    fn test_pulldown_short_sequence_unknown() {
        // Fewer than MIN_FRAMES with duplicates → Unknown
        let dups = vec![false, true, false, true, false]; // 5 frames
        let cadence = make_cadence(&dups);
        assert_eq!(detect_pulldown_pattern(&cadence), PulldownPattern::Unknown);
    }

    #[test]
    fn test_pulldown_display() {
        assert!(PulldownPattern::None.to_string().contains("Progressive"));
        assert!(PulldownPattern::Pulldown32.to_string().contains("3:2"));
        assert!(PulldownPattern::Pulldown22.to_string().contains("2:2"));
        assert!(PulldownPattern::Unknown.to_string().contains("Unknown"));
    }

    #[test]
    fn test_pulldown_irregular_pattern() {
        // Random duplicates — should not match any clean pattern
        let dups = [
            false, true, false, false, true, true, false, true, false, false, false, true, false,
            true, true, false, true, false, false, true,
        ];
        let cadence = make_cadence(&dups);
        // Result is either Unknown or one of the patterns — just ensure no panic
        let _ = detect_pulldown_pattern(&cadence);
    }
}
