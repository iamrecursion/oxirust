//! 3:2 pulldown and cadence detection for interlaced video streams.
//!
//! Detects the cadence pattern used when 24 fps film is transferred to 29.97 fps
//! video (3:2 pulldown), as well as other common interlace cadences. Provides
//! utilities to reconstruct clean progressive frames from the detected cadence.

use std::collections::VecDeque;

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// The detected field cadence of a video stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cadence {
    /// All frames are already progressive (no interlace artefacts).
    Progressive,
    /// Fully interlaced: every frame is a field pair without pulldown.
    Interlaced,
    /// 2:3 pulldown: groups of 2 progressive + 3 interlaced fields (film @ 24 fps → 29.97 fps).
    Pulldown23,
    /// 3:2 pulldown: reversed pattern (3 interlaced + 2 progressive fields per group).
    Pulldown32,
    /// 2:3:3:2 pulldown: alternate 4-frame cadence variant.
    Pulldown2332,
    /// Pattern could not be determined from the available history.
    Unknown,
}

/// A raw interlaced field pair: top (even) and bottom (odd) scanlines.
#[derive(Debug, Clone)]
pub struct FieldPair {
    /// Even scanlines of the frame (`width × (height/2)` bytes).
    pub top_field: Vec<u8>,
    /// Odd scanlines of the frame (`width × (height/2)` bytes).
    pub bottom_field: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Full frame height in pixels (both fields combined).
    pub height: u32,
}

impl FieldPair {
    /// Reconstruct the interleaved frame (top=even rows, bottom=odd rows).
    ///
    /// Output is `width × height` bytes.
    pub fn interleave(&self) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let field_h = (h + 1) / 2;
        let mut out = vec![0u8; w * h];
        for row in 0..h {
            let field = if row % 2 == 0 {
                &self.top_field
            } else {
                &self.bottom_field
            };
            let field_row = row / 2;
            if field_row >= field_h {
                continue;
            }
            let src_start = field_row * w;
            let dst_start = row * w;
            let src = field.get(src_start..src_start + w).unwrap_or(&[]);
            let dst = out.get_mut(dst_start..dst_start + w).unwrap_or(&mut []);
            let copy_len = src.len().min(dst.len());
            dst[..copy_len].copy_from_slice(&src[..copy_len]);
        }
        out
    }
}

/// Per-field-pair metrics used for cadence analysis.
#[derive(Debug, Clone)]
pub struct FieldMetrics {
    /// Sequential frame index.
    pub frame_number: u64,
    /// Measure of inter-field combing artefacts (0 = clean, 1 = highly combed).
    pub combing_score: f32,
    /// `true` if the top field is temporally first (top-field-first order).
    pub tff: bool,
}

/// A fully reconstructed progressive frame.
#[derive(Debug, Clone)]
pub struct ProgressiveFrame {
    /// Pixel data (`width × height` bytes, luma only).
    pub data: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Index of the originating `FieldPair` in the input slice.
    pub original_index: usize,
}

// -----------------------------------------------------------------------
// CadenceDetector
// -----------------------------------------------------------------------

/// Stateful cadence detector that accumulates `FieldMetrics` over time.
pub struct CadenceDetector {
    /// Rolling history of field metrics.
    pub history: VecDeque<FieldMetrics>,
    /// Number of frames kept in the analysis window.
    pub window_size: usize,
}

impl CadenceDetector {
    /// Create a new `CadenceDetector` with `window_size` frame history.
    ///
    /// A window of at least 10 is recommended; 5 is the minimum for pattern
    /// detection.
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Push a new set of field metrics into the detector.
    pub fn push(&mut self, metrics: FieldMetrics) {
        if self.history.len() >= self.window_size {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
    }

    /// Analyse the current history and return the detected cadence.
    pub fn current_cadence(&self) -> Cadence {
        let slice: Vec<&FieldMetrics> = self.history.iter().collect();
        detect_cadence(&slice)
    }
}

// -----------------------------------------------------------------------
// Public free functions
// -----------------------------------------------------------------------

/// Compute the combing score of a single luma frame.
///
/// The score is defined as:
/// ```text
///   score = sum |row[y][x] - row[y+2][x]| / (width × (height - 2))
/// ```
/// normalised to [0, 1].  A purely progressive frame scores near 0; a
/// heavily combed interlaced frame scores higher.
pub fn combing_score(frame: &[u8], width: u32, height: u32) -> f32 {
    let w = width as usize;
    let h = height as usize;
    if h < 3 || w == 0 {
        return 0.0;
    }

    let mut total_diff = 0u64;
    let compared_rows = h - 2;

    for row in 0..compared_rows {
        for col in 0..w {
            let a = frame.get(row * w + col).copied().unwrap_or(0) as i32;
            let b = frame.get((row + 2) * w + col).copied().unwrap_or(0) as i32;
            total_diff += (a - b).unsigned_abs() as u64;
        }
    }

    let total_pixels = (compared_rows * w) as u64;
    if total_pixels == 0 {
        return 0.0;
    }

    // Normalise: max possible per-pixel diff is 255.
    (total_diff as f64 / (total_pixels as f64 * 255.0)) as f32
}

/// Analyse a slice of `FieldMetrics` and identify the cadence pattern.
///
/// ## Pattern Recognition
/// 2:3 pulldown produces a repeating \[H, L, L, H, L\] pattern of combing
/// scores in a 5-frame cycle.  3:2 produces \[L, L, H, L, H\].  Both variants
/// of \[H,L,L,H,L\] and \[L,L,H,L,H\] are searched within the last 5 frames.
///
/// The thresholds used are:
/// - Score > 0.04 → "High" combing
/// - Score ≤ 0.04 → "Low" combing
pub fn detect_cadence(history: &[&FieldMetrics]) -> Cadence {
    if history.len() < 5 {
        return Cadence::Unknown;
    }

    // Take the last 5 frames.
    let window: Vec<bool> = history
        .iter()
        .rev()
        .take(5)
        .map(|m| m.combing_score > 0.04)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    // H = true, L = false
    // 2:3 pattern (film→NTSC): [H,L,L,H,L]
    let pulldown_23 = [true, false, false, true, false];
    // 3:2 pattern: [L,L,H,L,H]
    let pulldown_32 = [false, false, true, false, true];
    // 2:3:3:2 (less common): all high except positions 0,4
    let pulldown_2332 = [false, true, true, true, false];

    if window == pulldown_23 {
        return Cadence::Pulldown23;
    }
    if window == pulldown_32 {
        return Cadence::Pulldown32;
    }
    if window == pulldown_2332 {
        return Cadence::Pulldown2332;
    }

    // All low → progressive
    if window.iter().all(|&h| !h) {
        return Cadence::Progressive;
    }

    // All high → fully interlaced
    if window.iter().all(|&h| h) {
        return Cadence::Interlaced;
    }

    // Mixed but no recognised pattern.
    Cadence::Unknown
}

/// Remove 3:2 pulldown artefacts and reconstruct clean progressive frames.
///
/// Strategy:
/// - `Progressive` / `Interlaced`: return the interleaved frame as-is.
/// - `Pulldown23` / `Pulldown32`: for every 5-frame group, pick the 3 frames
///   with the lowest combing scores (i.e. the true progressive frames).  The
///   remaining 2 combed frames are reconstructed by blending adjacent fields.
/// - `Pulldown2332` / `Unknown`: fall back to returning all frames unchanged.
pub fn remove_pulldown(frames: &[FieldPair], cadence: Cadence) -> Vec<ProgressiveFrame> {
    match cadence {
        Cadence::Progressive | Cadence::Interlaced | Cadence::Unknown => {
            // No modification; reconstruct each FieldPair as a progressive frame.
            frames
                .iter()
                .enumerate()
                .map(|(idx, fp)| ProgressiveFrame {
                    data: fp.interleave(),
                    width: fp.width,
                    height: fp.height,
                    original_index: idx,
                })
                .collect()
        }
        Cadence::Pulldown23 | Cadence::Pulldown32 => remove_pulldown_23_32(frames),
        Cadence::Pulldown2332 => {
            // Return all frames interleaved; no combed-frame elimination.
            frames
                .iter()
                .enumerate()
                .map(|(idx, fp)| ProgressiveFrame {
                    data: fp.interleave(),
                    width: fp.width,
                    height: fp.height,
                    original_index: idx,
                })
                .collect()
        }
    }
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

/// Reconstruct progressive frames from 2:3 or 3:2 pulldown input.
///
/// Process the input in 5-frame windows.  Within each window:
/// 1. Compute a combing score for each interleaved frame.
/// 2. Identify the 2 most-combed frames (pulldown artefacts).
/// 3. For those frames, blend the top field of the preceding clean frame with
///    the bottom field of the following clean frame to produce a synthetic
///    progressive replacement.
/// 4. The remaining 3 clean frames are passed through unchanged.
///
/// Partial windows at the end are passed through as-is.
fn remove_pulldown_23_32(frames: &[FieldPair]) -> Vec<ProgressiveFrame> {
    let mut out: Vec<ProgressiveFrame> = Vec::with_capacity(frames.len());

    let mut start = 0;
    while start < frames.len() {
        let end = (start + 5).min(frames.len());
        let window = &frames[start..end];

        if window.len() < 5 {
            // Partial window — pass through unchanged.
            for (i, fp) in window.iter().enumerate() {
                out.push(ProgressiveFrame {
                    data: fp.interleave(),
                    width: fp.width,
                    height: fp.height,
                    original_index: start + i,
                });
            }
            start = end;
            continue;
        }

        // Compute combing scores for each interleaved frame.
        let scores: Vec<f32> = window
            .iter()
            .map(|fp| {
                let interleaved = fp.interleave();
                combing_score(&interleaved, fp.width, fp.height)
            })
            .collect();

        // Find the indices of the 2 most-combed frames (artefact frames).
        let mut scored_indices: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        scored_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let artefact_set: std::collections::HashSet<usize> =
            scored_indices.iter().take(2).map(|&(i, _)| i).collect();

        for (local_idx, fp) in window.iter().enumerate() {
            let global_idx = start + local_idx;
            if !artefact_set.contains(&local_idx) {
                out.push(ProgressiveFrame {
                    data: fp.interleave(),
                    width: fp.width,
                    height: fp.height,
                    original_index: global_idx,
                });
            } else {
                // Reconstruct by blending adjacent clean frames' fields.
                let prev = if local_idx > 0 && !artefact_set.contains(&(local_idx - 1)) {
                    Some(&window[local_idx - 1])
                } else if local_idx > 1 && !artefact_set.contains(&(local_idx - 2)) {
                    Some(&window[local_idx - 2])
                } else {
                    None
                };
                let next_idx = local_idx + 1;
                let next = if next_idx < window.len() && !artefact_set.contains(&next_idx) {
                    Some(&window[next_idx])
                } else if next_idx + 1 < window.len() && !artefact_set.contains(&(next_idx + 1)) {
                    Some(&window[next_idx + 1])
                } else {
                    None
                };

                let reconstructed = reconstruct_from_adjacent(fp, prev, next);
                out.push(ProgressiveFrame {
                    data: reconstructed,
                    width: fp.width,
                    height: fp.height,
                    original_index: global_idx,
                });
            }
        }

        start = end;
    }

    out
}

/// Reconstruct a progressive frame for a combed `FieldPair` by using the
/// top field from `prev` (if available) and the bottom field from `next`
/// (if available), blending with the combed frame's own fields where
/// neighbours are absent.
fn reconstruct_from_adjacent(
    combed: &FieldPair,
    prev: Option<&FieldPair>,
    next: Option<&FieldPair>,
) -> Vec<u8> {
    let w = combed.width as usize;
    let h = combed.height as usize;
    let field_h = (h + 1) / 2;
    let mut out = vec![0u8; w * h];

    for row in 0..h {
        let dst_start = row * w;
        if row % 2 == 0 {
            // Even row: prefer top field from `prev`.
            let src = match prev {
                Some(fp) if fp.top_field.len() >= (row / 2 + 1) * w => {
                    let s = (row / 2) * w;
                    fp.top_field.get(s..s + w)
                }
                _ => {
                    let s = (row / 2) * w;
                    combed.top_field.get(s..s + w)
                }
            };
            if let Some(src_slice) = src {
                let dst = out.get_mut(dst_start..dst_start + w).unwrap_or(&mut []);
                let copy_len = src_slice.len().min(dst.len());
                dst[..copy_len].copy_from_slice(&src_slice[..copy_len]);
            }
        } else {
            // Odd row: prefer bottom field from `next`.
            let field_row = row / 2;
            if field_row >= field_h {
                continue;
            }
            let src = match next {
                Some(fp) if fp.bottom_field.len() >= (field_row + 1) * w => {
                    let s = field_row * w;
                    fp.bottom_field.get(s..s + w)
                }
                _ => {
                    let s = field_row * w;
                    combed.bottom_field.get(s..s + w)
                }
            };
            if let Some(src_slice) = src {
                let dst = out.get_mut(dst_start..dst_start + w).unwrap_or(&mut []);
                let copy_len = src_slice.len().min(dst.len());
                dst[..copy_len].copy_from_slice(&src_slice[..copy_len]);
            }
        }
    }

    out
}

/// Split a flat luma frame into a `FieldPair`.
///
/// Top field = even rows; bottom field = odd rows.
pub fn split_into_field_pair(frame: &[u8], width: u32, height: u32, tff: bool) -> FieldPair {
    let w = width as usize;
    let h = height as usize;
    let field_h = (h + 1) / 2;
    let mut top = Vec::with_capacity(w * field_h);
    let mut bottom = Vec::with_capacity(w * field_h);

    for row in 0..h {
        let start = row * w;
        let slice = frame.get(start..start + w).unwrap_or(&[]);
        if row % 2 == 0 {
            top.extend_from_slice(slice);
        } else {
            bottom.extend_from_slice(slice);
        }
    }

    // If bottom-field-first, swap so that top_field is always the first
    // temporal field regardless of display order.
    let (top_field, bottom_field) = if tff { (top, bottom) } else { (bottom, top) };

    FieldPair {
        top_field,
        bottom_field,
        width,
        height,
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -------------------------------------------------------

    /// Flat luma frame of `width × height` all set to `val`.
    fn flat_frame(width: u32, height: u32, val: u8) -> Vec<u8> {
        vec![val; (width * height) as usize]
    }

    /// Build a perfectly progressive flat frame → FieldPair.
    fn progressive_pair(width: u32, height: u32, val: u8) -> FieldPair {
        let frame = flat_frame(width, height, val);
        split_into_field_pair(&frame, width, height, true)
    }

    /// Build a strongly combed frame: alternating rows differ maximally AND
    /// same-parity rows also differ so that the row[y] vs row[y+2] comparison
    /// in `combing_score` produces a nonzero result.
    ///
    /// Pattern: row 0=0, row 1=255, row 2=128, row 3=255, row 4=0, …
    /// Even rows cycle through [0, 128, 0, 128, …]; odd rows are always 255.
    /// Therefore row[0] - row[2] = |0 - 128| = 128 ≠ 0.
    fn combed_frame(width: u32, height: u32) -> Vec<u8> {
        let w = width as usize;
        let h = height as usize;
        let mut v = vec![0u8; w * h];
        for row in 0..h {
            let fill = if row % 2 != 0 {
                255u8
            } else if (row / 2) % 2 == 0 {
                0u8
            } else {
                128u8
            };
            for col in 0..w {
                v[row * w + col] = fill;
            }
        }
        v
    }

    /// Build a `FieldMetrics` with the given combing score.
    fn make_metrics(frame_number: u64, score: f32) -> FieldMetrics {
        FieldMetrics {
            frame_number,
            combing_score: score,
            tff: true,
        }
    }

    // ---- combing_score tests -------------------------------------------

    // 1. Flat frame → combing score 0
    #[test]
    fn test_combing_score_flat_frame_zero() {
        let frame = flat_frame(16, 16, 128);
        let score = combing_score(&frame, 16, 16);
        assert_eq!(score, 0.0);
    }

    // 2. Combed frame → combing score > 0
    #[test]
    fn test_combing_score_combed_frame_nonzero() {
        let frame = combed_frame(16, 16);
        let score = combing_score(&frame, 16, 16);
        assert!(
            score > 0.0,
            "combed frame should have score > 0, got {score}"
        );
    }

    // 3. combing_score: height < 3 → 0
    #[test]
    fn test_combing_score_short_frame_zero() {
        let frame = flat_frame(8, 2, 100);
        let score = combing_score(&frame, 8, 2);
        assert_eq!(score, 0.0);
    }

    // 4. combing_score: result is in [0, 1]
    #[test]
    fn test_combing_score_range() {
        let frame = combed_frame(32, 32);
        let score = combing_score(&frame, 32, 32);
        assert!(score >= 0.0 && score <= 1.0, "score {score} out of range");
    }

    // ---- FieldPair / split tests ---------------------------------------

    // 5. split_into_field_pair: top_field contains even rows
    #[test]
    fn test_split_top_field_even_rows() {
        let mut frame = vec![0u8; 8 * 4]; // 8×4
                                          // row 0 = 10, row 1 = 20, row 2 = 30, row 3 = 40
        for row in 0..4usize {
            let val = (row as u8 + 1) * 10;
            for col in 0..8usize {
                frame[row * 8 + col] = val;
            }
        }
        let fp = split_into_field_pair(&frame, 8, 4, true);
        // top_field row 0 = frame row 0 = 10
        assert_eq!(fp.top_field[0], 10);
        // top_field row 1 = frame row 2 = 30
        assert_eq!(fp.top_field[8], 30);
    }

    // 6. split_into_field_pair: bottom_field contains odd rows
    #[test]
    fn test_split_bottom_field_odd_rows() {
        let mut frame = vec![0u8; 8 * 4];
        for row in 0..4usize {
            let val = (row as u8 + 1) * 10;
            for col in 0..8usize {
                frame[row * 8 + col] = val;
            }
        }
        let fp = split_into_field_pair(&frame, 8, 4, true);
        // bottom_field row 0 = frame row 1 = 20
        assert_eq!(fp.bottom_field[0], 20);
        // bottom_field row 1 = frame row 3 = 40
        assert_eq!(fp.bottom_field[8], 40);
    }

    // 7. FieldPair::interleave round-trips a flat frame
    #[test]
    fn test_field_pair_interleave_roundtrip() {
        let frame = flat_frame(8, 8, 77);
        let fp = split_into_field_pair(&frame, 8, 8, true);
        let reconstructed = fp.interleave();
        assert_eq!(reconstructed, frame);
    }

    // 8. FieldPair::interleave produces correct dimensions
    #[test]
    fn test_field_pair_interleave_dimensions() {
        let frame = flat_frame(16, 12, 0);
        let fp = split_into_field_pair(&frame, 16, 12, true);
        assert_eq!(fp.interleave().len(), 16 * 12);
    }

    // ---- detect_cadence tests ------------------------------------------

    // 9. detect_cadence: fewer than 5 frames → Unknown
    #[test]
    fn test_detect_cadence_too_few_frames() {
        let history: Vec<FieldMetrics> = (0..4).map(|i| make_metrics(i, 0.01)).collect();
        let refs: Vec<&FieldMetrics> = history.iter().collect();
        assert_eq!(detect_cadence(&refs), Cadence::Unknown);
    }

    // 10. detect_cadence: all low → Progressive
    #[test]
    fn test_detect_cadence_all_low_progressive() {
        let history: Vec<FieldMetrics> = (0..5).map(|i| make_metrics(i, 0.01)).collect();
        let refs: Vec<&FieldMetrics> = history.iter().collect();
        assert_eq!(detect_cadence(&refs), Cadence::Progressive);
    }

    // 11. detect_cadence: all high → Interlaced
    #[test]
    fn test_detect_cadence_all_high_interlaced() {
        let history: Vec<FieldMetrics> = (0..5).map(|i| make_metrics(i, 0.2)).collect();
        let refs: Vec<&FieldMetrics> = history.iter().collect();
        assert_eq!(detect_cadence(&refs), Cadence::Interlaced);
    }

    // 12. detect_cadence: [H,L,L,H,L] → Pulldown23
    #[test]
    fn test_detect_cadence_pulldown23_pattern() {
        let scores = [0.2f32, 0.01, 0.01, 0.2, 0.01];
        let history: Vec<FieldMetrics> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| make_metrics(i as u64, s))
            .collect();
        let refs: Vec<&FieldMetrics> = history.iter().collect();
        assert_eq!(detect_cadence(&refs), Cadence::Pulldown23);
    }

    // 13. detect_cadence: [L,L,H,L,H] → Pulldown32
    #[test]
    fn test_detect_cadence_pulldown32_pattern() {
        let scores = [0.01f32, 0.01, 0.2, 0.01, 0.2];
        let history: Vec<FieldMetrics> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| make_metrics(i as u64, s))
            .collect();
        let refs: Vec<&FieldMetrics> = history.iter().collect();
        assert_eq!(detect_cadence(&refs), Cadence::Pulldown32);
    }

    // 14. detect_cadence: [L,H,H,H,L] → Pulldown2332
    #[test]
    fn test_detect_cadence_pulldown2332_pattern() {
        let scores = [0.01f32, 0.2, 0.2, 0.2, 0.01];
        let history: Vec<FieldMetrics> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| make_metrics(i as u64, s))
            .collect();
        let refs: Vec<&FieldMetrics> = history.iter().collect();
        assert_eq!(detect_cadence(&refs), Cadence::Pulldown2332);
    }

    // 15. detect_cadence: irregular pattern → Unknown
    #[test]
    fn test_detect_cadence_irregular_unknown() {
        let scores = [0.01f32, 0.2, 0.01, 0.01, 0.2];
        let history: Vec<FieldMetrics> = scores
            .iter()
            .enumerate()
            .map(|(i, &s)| make_metrics(i as u64, s))
            .collect();
        let refs: Vec<&FieldMetrics> = history.iter().collect();
        assert_eq!(detect_cadence(&refs), Cadence::Unknown);
    }

    // ---- remove_pulldown tests -----------------------------------------

    // 16. remove_pulldown Progressive: returns same number of frames
    #[test]
    fn test_remove_pulldown_progressive_frame_count() {
        let pairs: Vec<FieldPair> = (0..5).map(|_| progressive_pair(8, 8, 128)).collect();
        let out = remove_pulldown(&pairs, Cadence::Progressive);
        assert_eq!(out.len(), pairs.len());
    }

    // 17. remove_pulldown Progressive: data is unchanged
    #[test]
    fn test_remove_pulldown_progressive_data_unchanged() {
        let pairs: Vec<FieldPair> = (0..3)
            .map(|i| progressive_pair(8, 8, i as u8 * 50))
            .collect();
        let out = remove_pulldown(&pairs, Cadence::Progressive);
        for (i, pf) in out.iter().enumerate() {
            assert_eq!(pf.original_index, i);
            assert_eq!(pf.data.len(), (8 * 8) as usize);
        }
    }

    // 18. remove_pulldown Pulldown23: produces 5 output frames from 5 inputs
    #[test]
    fn test_remove_pulldown_23_output_count() {
        let pairs: Vec<FieldPair> = (0..5).map(|_| progressive_pair(8, 8, 100)).collect();
        let out = remove_pulldown(&pairs, Cadence::Pulldown23);
        assert_eq!(out.len(), 5);
    }

    // 19. remove_pulldown output frames have correct dimensions
    #[test]
    fn test_remove_pulldown_output_dimensions() {
        let pairs: Vec<FieldPair> = (0..5).map(|_| progressive_pair(16, 8, 200)).collect();
        let out = remove_pulldown(&pairs, Cadence::Pulldown32);
        for pf in &out {
            assert_eq!(pf.width, 16);
            assert_eq!(pf.height, 8);
            assert_eq!(pf.data.len(), 16 * 8);
        }
    }

    // 20. remove_pulldown Unknown: all frames passed through
    #[test]
    fn test_remove_pulldown_unknown_all_through() {
        let pairs: Vec<FieldPair> = (0..7)
            .map(|i| progressive_pair(8, 8, i as u8 * 30))
            .collect();
        let out = remove_pulldown(&pairs, Cadence::Unknown);
        assert_eq!(out.len(), 7);
    }

    // ---- CadenceDetector tests -----------------------------------------

    // 21. CadenceDetector::push: history is bounded
    #[test]
    fn test_cadence_detector_history_bounded() {
        let mut det = CadenceDetector::new(5);
        for i in 0..20u64 {
            det.push(make_metrics(i, 0.01));
        }
        assert!(det.history.len() <= 5);
    }

    // 22. CadenceDetector::current_cadence: too few frames → Unknown
    #[test]
    fn test_cadence_detector_too_few_unknown() {
        let mut det = CadenceDetector::new(10);
        det.push(make_metrics(0, 0.01));
        assert_eq!(det.current_cadence(), Cadence::Unknown);
    }

    // 23. CadenceDetector: 5 progressive frames → Progressive
    #[test]
    fn test_cadence_detector_progressive() {
        let mut det = CadenceDetector::new(10);
        for i in 0..5u64 {
            det.push(make_metrics(i, 0.01));
        }
        assert_eq!(det.current_cadence(), Cadence::Progressive);
    }

    // 24. Cadence variants are distinguishable
    #[test]
    fn test_cadence_variants_distinct() {
        assert_ne!(Cadence::Progressive, Cadence::Interlaced);
        assert_ne!(Cadence::Pulldown23, Cadence::Pulldown32);
        assert_ne!(Cadence::Pulldown2332, Cadence::Unknown);
    }

    // 25. FieldMetrics fields are accessible
    #[test]
    fn test_field_metrics_fields() {
        let m = FieldMetrics {
            frame_number: 77,
            combing_score: 0.123,
            tff: false,
        };
        assert_eq!(m.frame_number, 77);
        assert!((m.combing_score - 0.123).abs() < 1e-5);
        assert!(!m.tff);
    }
}
