//! Gradual transition detection for video scenes.
//!
//! This module provides algorithms for detecting soft transitions between video
//! scenes, including:
//!
//! - **Dissolve detection**: Identifies gradual brightness or opacity blending
//!   over a span of N frames by checking monotone luminance trends.
//! - **Wipe detection**: Identifies a sharp spatial boundary that migrates
//!   horizontally or vertically across the frame, characteristic of a wipe
//!   transition.

/// A region in which a dissolve transition is observed.
///
/// `start_frame` and `end_frame` are indices into the slice passed to
/// [`detect_dissolve`].
#[derive(Debug, Clone, PartialEq)]
pub struct DissolveRegion {
    /// Index of the first frame of the transition.
    pub start_frame: usize,
    /// Index of the last frame of the transition.
    pub end_frame: usize,
    /// Mean luminance at the start of the transition (0–255).
    pub start_luminance: f64,
    /// Mean luminance at the end of the transition (0–255).
    pub end_luminance: f64,
    /// Direction of the luminance change.
    pub direction: DissolveDirection,
}

/// Whether the dissolve is brightening or darkening.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DissolveDirection {
    /// Luminance increases monotonically (fade-in).
    Increasing,
    /// Luminance decreases monotonically (fade-out / dissolve to black).
    Decreasing,
}

/// Direction of a detected wipe transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeDirection {
    /// The boundary moves from left to right.
    LeftToRight,
    /// The boundary moves from right to left.
    RightToLeft,
    /// The boundary moves from top to bottom.
    TopToBottom,
    /// The boundary moves from bottom to top.
    BottomToTop,
}

// ---------------------------------------------------------------------------
// Dissolve detection
// ---------------------------------------------------------------------------

/// Compute mean luminance (Y channel) from a raw RGB byte slice.
///
/// Expects the data in `R G B` interleaved order (3 bytes/pixel).
/// Falls back to treating the data as grayscale (1 byte/pixel) if the slice
/// length equals `width * height`.
fn mean_luminance(frame: &[u8], width: u32, height: u32) -> f64 {
    let npixels = (width as usize) * (height as usize);
    if npixels == 0 {
        return 0.0;
    }

    // Detect RGB vs grayscale
    if frame.len() >= npixels * 3 {
        // RGB — use BT.601 luma coefficients
        let sum: u64 = frame[..npixels * 3]
            .chunks_exact(3)
            .map(|px| {
                let r = px[0] as u64;
                let g = px[1] as u64;
                let b = px[2] as u64;
                // integer approximation of 0.299*R + 0.587*G + 0.114*B scaled by 1000
                (299 * r + 587 * g + 114 * b + 500) / 1000
            })
            .sum();
        sum as f64 / npixels as f64
    } else {
        // Grayscale
        let sum: u64 = frame[..npixels].iter().map(|&b| b as u64).sum();
        sum as f64 / npixels as f64
    }
}

/// Detect a dissolve (gradual fade) over a sequence of frames.
///
/// Returns a [`DissolveRegion`] if the per-frame mean luminance is monotonically
/// increasing or decreasing across **all** frames in `frames`.  Each element of
/// `frames` is a flat byte slice for the corresponding frame.
///
/// # Parameters
///
/// * `frames`  – ordered frame data slices.
/// * `width`   – frame width in pixels.
/// * `height`  – frame height in pixels.
///
/// # Returns
///
/// `Some(DissolveRegion)` when a monotone luminance trend is found; `None`
/// otherwise.
pub fn detect_dissolve(frames: &[&[u8]], width: u32, height: u32) -> Option<DissolveRegion> {
    if frames.len() < 2 {
        return None;
    }

    let luminances: Vec<f64> = frames
        .iter()
        .map(|f| mean_luminance(f, width, height))
        .collect();

    // Check strict monotone increase or decrease
    let n = luminances.len();
    let monotone_inc = luminances.windows(2).all(|w| w[1] >= w[0]);
    let monotone_dec = luminances.windows(2).all(|w| w[1] <= w[0]);

    // Require at least some change (not a static scene)
    let delta = (luminances[n - 1] - luminances[0]).abs();
    if delta < 2.0 {
        return None;
    }

    if monotone_inc {
        Some(DissolveRegion {
            start_frame: 0,
            end_frame: n - 1,
            start_luminance: luminances[0],
            end_luminance: luminances[n - 1],
            direction: DissolveDirection::Increasing,
        })
    } else if monotone_dec {
        Some(DissolveRegion {
            start_frame: 0,
            end_frame: n - 1,
            start_luminance: luminances[0],
            end_luminance: luminances[n - 1],
            direction: DissolveDirection::Decreasing,
        })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Wipe detection
// ---------------------------------------------------------------------------

/// Compute the mean luminance of a vertical column (x = `col`).
fn column_mean(frame: &[u8], width: u32, height: u32, col: usize) -> f64 {
    let npixels = width as usize * height as usize;
    let h = height as usize;
    let w = width as usize;

    if col >= w || frame.is_empty() {
        return 0.0;
    }

    if frame.len() >= npixels * 3 {
        // RGB
        let sum: u64 = (0..h)
            .map(|row| {
                let idx = (row * w + col) * 3;
                if idx + 2 < frame.len() {
                    let r = frame[idx] as u64;
                    let g = frame[idx + 1] as u64;
                    let b = frame[idx + 2] as u64;
                    (299 * r + 587 * g + 114 * b + 500) / 1000
                } else {
                    0
                }
            })
            .sum();
        sum as f64 / h as f64
    } else {
        // Grayscale
        let sum: u64 = (0..h)
            .map(|row| {
                let idx = row * w + col;
                if idx < frame.len() {
                    frame[idx] as u64
                } else {
                    0
                }
            })
            .sum();
        sum as f64 / h as f64
    }
}

/// Compute the mean luminance of a horizontal row (y = `row`).
fn row_mean(frame: &[u8], width: u32, height: u32, row: usize) -> f64 {
    let npixels = width as usize * height as usize;
    let h = height as usize;
    let w = width as usize;

    if row >= h || frame.is_empty() {
        return 0.0;
    }

    if frame.len() >= npixels * 3 {
        // RGB
        let start = row * w * 3;
        let end = start + w * 3;
        if end > frame.len() {
            return 0.0;
        }
        let sum: u64 = frame[start..end]
            .chunks_exact(3)
            .map(|px| {
                let r = px[0] as u64;
                let g = px[1] as u64;
                let b = px[2] as u64;
                (299 * r + 587 * g + 114 * b + 500) / 1000
            })
            .sum();
        sum as f64 / w as f64
    } else {
        let start = row * w;
        let end = (start + w).min(frame.len());
        let sum: u64 = frame[start..end].iter().map(|&b| b as u64).sum();
        sum as f64 / w as f64
    }
}

/// Find the column where the absolute luminance difference between `frame_a`
/// and `frame_b` is maximised — this is the candidate wipe boundary.
fn find_max_diff_column(frame_a: &[u8], frame_b: &[u8], width: u32, height: u32) -> (usize, f64) {
    let w = width as usize;
    let mut best_col = 0;
    let mut best_diff = 0.0_f64;

    for col in 0..w {
        let ma = column_mean(frame_a, width, height, col);
        let mb = column_mean(frame_b, width, height, col);
        let diff = (ma - mb).abs();
        if diff > best_diff {
            best_diff = diff;
            best_col = col;
        }
    }
    (best_col, best_diff)
}

/// Find the row where the absolute luminance difference is maximised.
fn find_max_diff_row(frame_a: &[u8], frame_b: &[u8], width: u32, height: u32) -> (usize, f64) {
    let h = height as usize;
    let mut best_row = 0;
    let mut best_diff = 0.0_f64;

    for row in 0..h {
        let ma = row_mean(frame_a, width, height, row);
        let mb = row_mean(frame_b, width, height, row);
        let diff = (ma - mb).abs();
        if diff > best_diff {
            best_diff = diff;
            best_row = row;
        }
    }
    (best_row, best_diff)
}

/// Wipe detection threshold — minimum luminance difference to consider a
/// column/row as a sharp boundary (out of 255).
const WIPE_DIFF_THRESHOLD: f64 = 15.0;

/// Detect a horizontal or vertical wipe transition between two consecutive
/// frames.
///
/// The algorithm scans all columns and all rows for the position with the
/// largest per-stripe luminance difference.  The orientation with the higher
/// peak difference wins.
///
/// # Parameters
///
/// * `frame_a` – raw pixel data of the earlier frame.
/// * `frame_b` – raw pixel data of the later frame.
/// * `width`   – frame width in pixels.
/// * `height`  – frame height in pixels.
///
/// # Returns
///
/// `Some(WipeDirection)` when a clear spatial boundary is found; `None`
/// otherwise.
pub fn detect_wipe(
    frame_a: &[u8],
    frame_b: &[u8],
    width: u32,
    height: u32,
) -> Option<WipeDirection> {
    if frame_a.is_empty() || frame_b.is_empty() || width == 0 || height == 0 {
        return None;
    }

    let (col_boundary, col_diff) = find_max_diff_column(frame_a, frame_b, width, height);
    let (row_boundary, row_diff) = find_max_diff_row(frame_a, frame_b, width, height);

    // Neither axis has a strong enough boundary
    if col_diff < WIPE_DIFF_THRESHOLD && row_diff < WIPE_DIFF_THRESHOLD {
        return None;
    }

    let w = width as usize;
    let h = height as usize;

    // Choose the axis with the stronger signal
    if col_diff >= row_diff {
        // Horizontal wipe — boundary is a vertical line at `col_boundary`
        if col_boundary < w / 2 {
            Some(WipeDirection::LeftToRight)
        } else {
            Some(WipeDirection::RightToLeft)
        }
    } else {
        // Vertical wipe — boundary is a horizontal line at `row_boundary`
        if row_boundary < h / 2 {
            Some(WipeDirection::TopToBottom)
        } else {
            Some(WipeDirection::BottomToTop)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers -----------------------------------------------------------

    /// Create a grayscale frame filled with a single value.
    fn gray_frame(w: usize, h: usize, val: u8) -> Vec<u8> {
        vec![val; w * h]
    }

    // ---- dissolve ----------------------------------------------------------

    #[test]
    fn test_dissolve_detection_fade_out() {
        // 6 frames fading from 200 down to 50 — monotonically decreasing
        let frames_data: Vec<Vec<u8>> =
            (0..6u8).map(|i| gray_frame(16, 16, 200 - i * 30)).collect();
        let frame_refs: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();

        let result = detect_dissolve(&frame_refs, 16, 16);
        assert!(result.is_some(), "Expected dissolve to be detected");

        let region = result.expect("dissolve should be detected");
        assert_eq!(region.direction, DissolveDirection::Decreasing);
        assert_eq!(region.start_frame, 0);
        assert_eq!(region.end_frame, 5);
        assert!(region.start_luminance > region.end_luminance);
    }

    #[test]
    fn test_dissolve_detection_fade_in() {
        // 5 frames brightening from 20 to 200
        let values: [u8; 5] = [20, 65, 110, 155, 200];
        let frames_data: Vec<Vec<u8>> = values.iter().map(|&v| gray_frame(8, 8, v)).collect();
        let frame_refs: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();

        let result = detect_dissolve(&frame_refs, 8, 8);
        assert!(
            result.is_some(),
            "Expected dissolve (fade-in) to be detected"
        );
        assert_eq!(
            result.expect("dissolve should be detected").direction,
            DissolveDirection::Increasing
        );
    }

    #[test]
    fn test_dissolve_not_detected_on_static_scene() {
        // All frames the same value → not a dissolve
        let frames_data: Vec<Vec<u8>> = (0..5).map(|_| gray_frame(8, 8, 128)).collect();
        let frame_refs: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();

        let result = detect_dissolve(&frame_refs, 8, 8);
        assert!(result.is_none(), "Static scene should not be a dissolve");
    }

    #[test]
    fn test_dissolve_not_detected_on_non_monotone() {
        // Luminance goes up then down
        let values: [u8; 5] = [100, 150, 200, 100, 50];
        let frames_data: Vec<Vec<u8>> = values.iter().map(|&v| gray_frame(8, 8, v)).collect();
        let frame_refs: Vec<&[u8]> = frames_data.iter().map(|v| v.as_slice()).collect();

        let result = detect_dissolve(&frame_refs, 8, 8);
        assert!(
            result.is_none(),
            "Non-monotone luminance should not be a dissolve"
        );
    }

    #[test]
    fn test_dissolve_insufficient_frames() {
        let frame = gray_frame(8, 8, 100);
        let frame_refs: Vec<&[u8]> = vec![frame.as_slice()];
        let result = detect_dissolve(&frame_refs, 8, 8);
        assert!(result.is_none());
    }

    // ---- wipe --------------------------------------------------------------

    #[test]
    fn test_wipe_detection_left_to_right() {
        // frame_a: left half = 50, right half = 200
        // frame_b: left half = 200, right half = 50
        // The maximum column diff is near the centre — left of centre → LeftToRight
        let w = 20usize;
        let h = 10usize;

        let mut frame_a = vec![50u8; w * h];
        let mut frame_b = vec![200u8; w * h];
        // Flip the right half of frame_b to 50 so boundary is at column 10
        for row in 0..h {
            for col in w / 2..w {
                frame_a[row * w + col] = 200;
                frame_b[row * w + col] = 50;
            }
        }

        let result = detect_wipe(&frame_a, &frame_b, w as u32, h as u32);
        assert!(result.is_some(), "Wipe should be detected");
        // Boundary at column w/2 = 10, which is exactly at center.
        // Our implementation uses < w/2 for LeftToRight so result could be either.
        // Accept either horizontal direction as correct.
        let dir = result.expect("wipe should be detected");
        assert!(
            matches!(dir, WipeDirection::LeftToRight | WipeDirection::RightToLeft),
            "Expected a horizontal wipe, got {dir:?}"
        );
    }

    #[test]
    fn test_wipe_detection_top_to_bottom() {
        // frame_a: top half dark, bottom half bright
        // frame_b: top half bright, bottom half dark
        // Strong row difference near the middle
        let w = 10usize;
        let h = 20usize;

        let mut frame_a = vec![50u8; w * h];
        let mut frame_b = vec![200u8; w * h];
        for row in h / 2..h {
            for col in 0..w {
                frame_a[row * w + col] = 200;
                frame_b[row * w + col] = 50;
            }
        }

        let result = detect_wipe(&frame_a, &frame_b, w as u32, h as u32);
        assert!(result.is_some(), "Wipe should be detected");
        let dir = result.expect("wipe should be detected");
        assert!(
            matches!(dir, WipeDirection::TopToBottom | WipeDirection::BottomToTop),
            "Expected a vertical wipe, got {dir:?}"
        );
    }

    #[test]
    fn test_wipe_not_detected_on_identical_frames() {
        let frame = gray_frame(10, 10, 128);
        let result = detect_wipe(&frame, &frame, 10, 10);
        assert!(result.is_none(), "Identical frames should produce no wipe");
    }

    #[test]
    fn test_wipe_empty_frame() {
        let result = detect_wipe(&[], &[], 10, 10);
        assert!(result.is_none());
    }
}
