#![allow(dead_code)]
//! Visual rhythm extraction and analysis.
//!
//! A *visual rhythm* is a compact 2-D representation of temporal change in a
//! video.  For each frame a single column (or row) of pixels is extracted at a
//! fixed spatial position, and these columns are stacked side-by-side to form a
//! *rhythm image*.  Patterns in this image reveal:
//!
//! - **Scene cuts**: sharp vertical discontinuities.
//! - **Camera motion**: diagonal streaks (pan), vertical streaks (zoom).
//! - **Periodic motion**: repetitive wave-like patterns.
//! - **Flicker / strobe**: high-frequency horizontal banding.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Where in each frame to sample the column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingPosition {
    /// Left edge.
    Left,
    /// Horizontal centre.
    Centre,
    /// Right edge.
    Right,
    /// Custom column index.
    Column(usize),
    /// Average of all columns (projection).
    MeanProjection,
}

/// Configuration for visual rhythm extraction.
#[derive(Debug, Clone)]
pub struct VisualRhythmConfig {
    /// Sampling position for the column extraction.
    pub position: SamplingPosition,
    /// Whether to convert to grayscale before extraction.
    pub grayscale: bool,
    /// Optional temporal sub-sampling factor (1 = every frame).
    pub temporal_stride: usize,
}

/// A visual rhythm image.
#[derive(Debug, Clone)]
pub struct VisualRhythm {
    /// Pixel data of the rhythm image (row-major, grayscale 0..1).
    pub pixels: Vec<f64>,
    /// Width of the rhythm image (= number of frames sampled).
    pub width: usize,
    /// Height of the rhythm image (= frame height).
    pub height: usize,
}

/// A detected event in the visual rhythm.
#[derive(Debug, Clone)]
pub struct RhythmEvent {
    /// Frame index where the event was detected.
    pub frame_index: usize,
    /// Kind of event.
    pub kind: RhythmEventKind,
    /// Strength / magnitude of the event (0..1).
    pub strength: f64,
}

/// Kinds of events detectable in a visual rhythm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhythmEventKind {
    /// Hard scene cut.
    SceneCut,
    /// Gradual transition (dissolve / fade).
    GradualTransition,
    /// Flash or strobe.
    Flash,
    /// Periodic motion detected.
    PeriodicMotion,
}

/// Statistics of a visual rhythm.
#[derive(Debug, Clone)]
pub struct RhythmStats {
    /// Mean pixel intensity.
    pub mean_intensity: f64,
    /// Temporal variance (averaged across rows).
    pub temporal_variance: f64,
    /// Number of detected events.
    pub event_count: usize,
    /// Estimated dominant period (in frames), or `None` if aperiodic.
    pub dominant_period: Option<usize>,
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl Default for VisualRhythmConfig {
    fn default() -> Self {
        Self {
            position: SamplingPosition::Centre,
            grayscale: true,
            temporal_stride: 1,
        }
    }
}

impl VisualRhythm {
    /// Create an empty rhythm image.
    pub fn empty() -> Self {
        Self {
            pixels: Vec::new(),
            width: 0,
            height: 0,
        }
    }

    /// Get the pixel value at (frame, row).
    pub fn get(&self, frame: usize, row: usize) -> Option<f64> {
        if frame < self.width && row < self.height {
            Some(self.pixels[row * self.width + frame])
        } else {
            None
        }
    }

    /// Get the column (all rows) for a given frame.
    pub fn column(&self, frame: usize) -> Option<Vec<f64>> {
        if frame >= self.width {
            return None;
        }
        Some(
            (0..self.height)
                .map(|r| self.pixels[r * self.width + frame])
                .collect(),
        )
    }
}

/// Extract a visual rhythm from a sequence of grayscale frames.
///
/// Each frame is `frame_width x frame_height` pixels, row-major, values in 0..1.
#[allow(clippy::cast_precision_loss)]
pub fn extract_rhythm(
    frames: &[Vec<f64>],
    frame_width: usize,
    frame_height: usize,
    config: &VisualRhythmConfig,
) -> VisualRhythm {
    if frames.is_empty() || frame_width == 0 || frame_height == 0 {
        return VisualRhythm::empty();
    }

    let stride = config.temporal_stride.max(1);
    let sampled_frames: Vec<usize> = (0..frames.len()).step_by(stride).collect();
    let n_frames = sampled_frames.len();

    let mut pixels = vec![0.0f64; n_frames * frame_height];

    for (col, &fi) in sampled_frames.iter().enumerate() {
        let frame = &frames[fi];
        if frame.len() < frame_width * frame_height {
            continue;
        }

        for row in 0..frame_height {
            let value = match config.position {
                SamplingPosition::Left => frame[row * frame_width],
                SamplingPosition::Centre => frame[row * frame_width + frame_width / 2],
                SamplingPosition::Right => frame[row * frame_width + frame_width - 1],
                SamplingPosition::Column(c) => {
                    let c = c.min(frame_width - 1);
                    frame[row * frame_width + c]
                }
                SamplingPosition::MeanProjection => {
                    let start = row * frame_width;
                    let end = start + frame_width;
                    frame[start..end].iter().sum::<f64>() / frame_width as f64
                }
            };
            pixels[row * n_frames + col] = value;
        }
    }

    VisualRhythm {
        pixels,
        width: n_frames,
        height: frame_height,
    }
}

/// Detect events (cuts, flashes, etc.) in a visual rhythm.
#[allow(clippy::cast_precision_loss)]
pub fn detect_events(
    rhythm: &VisualRhythm,
    cut_threshold: f64,
    flash_threshold: f64,
) -> Vec<RhythmEvent> {
    let mut events = Vec::new();
    if rhythm.width < 2 || rhythm.height == 0 {
        return events;
    }

    // Column-difference based detection
    for frame in 1..rhythm.width {
        let mut diff_sum = 0.0f64;
        let mut max_diff = 0.0f64;
        for row in 0..rhythm.height {
            let curr = rhythm.pixels[row * rhythm.width + frame];
            let prev = rhythm.pixels[row * rhythm.width + frame - 1];
            let d = (curr - prev).abs();
            diff_sum += d;
            if d > max_diff {
                max_diff = d;
            }
        }
        let mean_diff = diff_sum / rhythm.height as f64;

        if mean_diff > cut_threshold {
            events.push(RhythmEvent {
                frame_index: frame,
                kind: RhythmEventKind::SceneCut,
                strength: mean_diff.min(1.0),
            });
        } else if max_diff > flash_threshold {
            events.push(RhythmEvent {
                frame_index: frame,
                kind: RhythmEventKind::Flash,
                strength: max_diff.min(1.0),
            });
        }
    }

    events
}

/// Compute statistics of a visual rhythm.
#[allow(clippy::cast_precision_loss)]
pub fn compute_rhythm_stats(rhythm: &VisualRhythm, events: &[RhythmEvent]) -> RhythmStats {
    if rhythm.pixels.is_empty() {
        return RhythmStats {
            mean_intensity: 0.0,
            temporal_variance: 0.0,
            event_count: 0,
            dominant_period: None,
        };
    }

    let mean_intensity = rhythm.pixels.iter().sum::<f64>() / rhythm.pixels.len() as f64;

    // Temporal variance: average across rows
    let mut total_var = 0.0f64;
    for row in 0..rhythm.height {
        let start = row * rhythm.width;
        let end = start + rhythm.width;
        if end > rhythm.pixels.len() {
            break;
        }
        let row_slice = &rhythm.pixels[start..end];
        let row_mean = row_slice.iter().sum::<f64>() / rhythm.width as f64;
        let var = row_slice
            .iter()
            .map(|v| (v - row_mean).powi(2))
            .sum::<f64>()
            / rhythm.width as f64;
        total_var += var;
    }
    let temporal_variance = if rhythm.height > 0 {
        total_var / rhythm.height as f64
    } else {
        0.0
    };

    // Simple periodicity: auto-correlation of column-mean signal
    let dominant_period = estimate_period(rhythm);

    RhythmStats {
        mean_intensity,
        temporal_variance,
        event_count: events.len(),
        dominant_period,
    }
}

/// Estimate dominant period via simple auto-correlation of the column-mean signal.
#[allow(clippy::cast_precision_loss)]
fn estimate_period(rhythm: &VisualRhythm) -> Option<usize> {
    if rhythm.width < 4 {
        return None;
    }

    // Compute column means
    let mut col_means = vec![0.0f64; rhythm.width];
    for frame in 0..rhythm.width {
        let mut sum = 0.0;
        for row in 0..rhythm.height {
            sum += rhythm.pixels[row * rhythm.width + frame];
        }
        col_means[frame] = sum / rhythm.height as f64;
    }

    let global_mean = col_means.iter().sum::<f64>() / rhythm.width as f64;

    // Auto-correlation for lags 2..width/2
    let max_lag = rhythm.width / 2;
    let mut best_lag = 0usize;
    let mut best_corr = f64::NEG_INFINITY;
    for lag in 2..max_lag {
        let mut corr = 0.0;
        let mut count = 0usize;
        for i in 0..rhythm.width - lag {
            corr += (col_means[i] - global_mean) * (col_means[i + lag] - global_mean);
            count += 1;
        }
        if count > 0 {
            corr /= count as f64;
            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }
    }

    if best_corr > 0.01 && best_lag > 0 {
        Some(best_lag)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_frame(w: usize, h: usize, val: f64) -> Vec<f64> {
        vec![val; w * h]
    }

    #[test]
    fn test_visual_rhythm_empty() {
        let r = VisualRhythm::empty();
        assert_eq!(r.width, 0);
        assert_eq!(r.height, 0);
    }

    #[test]
    fn test_extract_rhythm_no_frames() {
        let r = extract_rhythm(&[], 10, 10, &VisualRhythmConfig::default());
        assert_eq!(r.width, 0);
    }

    #[test]
    fn test_extract_rhythm_single_frame() {
        let frames = vec![uniform_frame(4, 4, 0.5)];
        let r = extract_rhythm(&frames, 4, 4, &VisualRhythmConfig::default());
        assert_eq!(r.width, 1);
        assert_eq!(r.height, 4);
        for row in 0..4 {
            assert!((r.get(0, row).expect("should succeed in test") - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn test_extract_rhythm_multiple_frames() {
        let frames = vec![
            uniform_frame(4, 4, 0.2),
            uniform_frame(4, 4, 0.5),
            uniform_frame(4, 4, 0.8),
        ];
        let r = extract_rhythm(&frames, 4, 4, &VisualRhythmConfig::default());
        assert_eq!(r.width, 3);
        assert_eq!(r.height, 4);
    }

    #[test]
    fn test_extract_rhythm_stride() {
        let frames: Vec<Vec<f64>> = (0..6)
            .map(|i| uniform_frame(4, 4, i as f64 / 5.0))
            .collect();
        let cfg = VisualRhythmConfig {
            temporal_stride: 2,
            ..VisualRhythmConfig::default()
        };
        let r = extract_rhythm(&frames, 4, 4, &cfg);
        assert_eq!(r.width, 3); // frames 0, 2, 4
    }

    #[test]
    fn test_extract_rhythm_mean_projection() {
        let frame = vec![0.1, 0.2, 0.3, 0.4]; // 4x1
        let frames = vec![frame];
        let cfg = VisualRhythmConfig {
            position: SamplingPosition::MeanProjection,
            ..VisualRhythmConfig::default()
        };
        let r = extract_rhythm(&frames, 4, 1, &cfg);
        assert!((r.get(0, 0).expect("should succeed in test") - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_column_extraction() {
        let frames = vec![uniform_frame(4, 3, 0.1), uniform_frame(4, 3, 0.9)];
        let r = extract_rhythm(&frames, 4, 3, &VisualRhythmConfig::default());
        let col0 = r.column(0).expect("should succeed in test");
        assert_eq!(col0.len(), 3);
        assert!((col0[0] - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_detect_events_empty() {
        let r = VisualRhythm::empty();
        let events = detect_events(&r, 0.3, 0.5);
        assert!(events.is_empty());
    }

    #[test]
    fn test_detect_scene_cut() {
        let frames = vec![
            uniform_frame(4, 4, 0.1),
            uniform_frame(4, 4, 0.1),
            uniform_frame(4, 4, 0.9), // cut here
            uniform_frame(4, 4, 0.9),
        ];
        let r = extract_rhythm(&frames, 4, 4, &VisualRhythmConfig::default());
        let events = detect_events(&r, 0.3, 0.8);
        assert!(!events.is_empty());
        assert!(events
            .iter()
            .any(|e| e.kind == RhythmEventKind::SceneCut && e.frame_index == 2));
    }

    #[test]
    fn test_detect_no_events_uniform() {
        let frames = vec![uniform_frame(4, 4, 0.5); 10];
        let r = extract_rhythm(&frames, 4, 4, &VisualRhythmConfig::default());
        let events = detect_events(&r, 0.3, 0.8);
        assert!(events.is_empty());
    }

    #[test]
    fn test_compute_stats_empty() {
        let r = VisualRhythm::empty();
        let stats = compute_rhythm_stats(&r, &[]);
        assert_eq!(stats.mean_intensity, 0.0);
        assert_eq!(stats.event_count, 0);
    }

    #[test]
    fn test_compute_stats_values() {
        let frames = vec![uniform_frame(4, 4, 0.5); 5];
        let r = extract_rhythm(&frames, 4, 4, &VisualRhythmConfig::default());
        let events = vec![];
        let stats = compute_rhythm_stats(&r, &events);
        assert!((stats.mean_intensity - 0.5).abs() < 1e-6);
        assert!(stats.temporal_variance < 1e-9);
    }

    #[test]
    fn test_default_config() {
        let c = VisualRhythmConfig::default();
        assert_eq!(c.position, SamplingPosition::Centre);
        assert!(c.grayscale);
        assert_eq!(c.temporal_stride, 1);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let r = VisualRhythm {
            pixels: vec![0.5; 4],
            width: 2,
            height: 2,
        };
        assert!(r.get(5, 0).is_none());
        assert!(r.get(0, 5).is_none());
    }
}
