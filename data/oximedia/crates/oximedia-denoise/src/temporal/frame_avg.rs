//! Frame averaging and motion-adaptive temporal noise filtering.
//!
//! Provides weighted frame averaging with adaptive weighting based on
//! inter-frame motion magnitude to reduce ghosting artifacts.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Weighting strategy for temporal frame averaging.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightingStrategy {
    /// All frames receive equal weight.
    Uniform,
    /// Frames closer to the current frame receive higher weight (linear decay).
    LinearDecay,
    /// Frames closer to the current frame receive higher weight (exponential decay).
    ExponentialDecay,
    /// Weights are computed adaptively based on pixel similarity.
    MotionAdaptive,
}

/// Configuration for temporal frame averaging.
#[derive(Debug, Clone)]
pub struct FrameAvgConfig {
    /// Number of frames to average (must be odd, >= 3).
    pub window_size: usize,
    /// Weighting strategy.
    pub strategy: WeightingStrategy,
    /// Decay rate for exponential weighting (0 < decay < 1).
    pub decay_rate: f32,
    /// Motion sensitivity threshold (0..=255): pixels above this are not averaged.
    pub motion_threshold: u8,
}

impl Default for FrameAvgConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            strategy: WeightingStrategy::ExponentialDecay,
            decay_rate: 0.7,
            motion_threshold: 20,
        }
    }
}

impl FrameAvgConfig {
    /// Create a config for simple uniform averaging.
    pub fn uniform(window_size: usize) -> Self {
        Self {
            window_size,
            strategy: WeightingStrategy::Uniform,
            ..Default::default()
        }
    }

    /// Create a motion-adaptive config.
    pub fn motion_adaptive(motion_threshold: u8) -> Self {
        Self {
            strategy: WeightingStrategy::MotionAdaptive,
            motion_threshold,
            ..Default::default()
        }
    }
}

/// Compute temporal average weights based on the selected strategy.
///
/// Returns a vector of `window_size` weights summing to 1.0, with the
/// centre weight corresponding to the current frame.
pub fn compute_weights(config: &FrameAvgConfig) -> Vec<f32> {
    let n = config.window_size;
    match config.strategy {
        WeightingStrategy::Uniform => {
            let w = 1.0 / n as f32;
            vec![w; n]
        }
        WeightingStrategy::LinearDecay => {
            // Weight[i] proportional to (n - distance_from_centre)
            let centre = n / 2;
            let raw: Vec<f32> = (0..n)
                .map(|i| {
                    let dist = if i <= centre { centre - i } else { i - centre };
                    (n - dist) as f32
                })
                .collect();
            let sum: f32 = raw.iter().sum();
            raw.iter().map(|&w| w / sum).collect()
        }
        WeightingStrategy::ExponentialDecay | WeightingStrategy::MotionAdaptive => {
            let centre = n / 2;
            let raw: Vec<f32> = (0..n)
                .map(|i| {
                    let dist = if i <= centre {
                        (centre - i) as f32
                    } else {
                        (i - centre) as f32
                    };
                    config.decay_rate.powf(dist)
                })
                .collect();
            let sum: f32 = raw.iter().sum();
            raw.iter().map(|&w| w / sum).collect()
        }
    }
}

/// Apply temporal frame averaging over a window of frames.
///
/// Each frame is a flat byte slice of length `width * height`.
/// Returns the averaged frame as a `Vec<u8>`.
///
/// # Panics
/// Panics if `frames` is empty or frames have different lengths.
pub fn temporal_average(frames: &[Vec<u8>], config: &FrameAvgConfig) -> Vec<u8> {
    assert!(!frames.is_empty(), "frames must not be empty");
    let len = frames[0].len();
    let weights = compute_weights(config);
    // Use only as many frames as available
    let n = frames.len().min(config.window_size);
    let used_frames = &frames[frames.len().saturating_sub(n)..];
    let used_weights: Vec<f32> = {
        let base = compute_weights(&FrameAvgConfig {
            window_size: n,
            ..config.clone()
        });
        let sum: f32 = base.iter().sum();
        if sum > 0.0 {
            base.iter().map(|&w| w / sum).collect()
        } else {
            vec![1.0 / n as f32; n]
        }
    };
    let _ = weights; // suppress warning; we re-computed for actual n

    let mut output = vec![0u8; len];
    for (pixel_idx, out) in output.iter_mut().enumerate() {
        let avg: f32 = used_frames
            .iter()
            .zip(used_weights.iter())
            .map(|(frame, &w)| frame[pixel_idx] as f32 * w)
            .sum();
        *out = avg.round().clamp(0.0, 255.0) as u8;
    }
    output
}

/// Apply motion-adaptive temporal averaging.
///
/// For each pixel, if the inter-frame difference exceeds `motion_threshold`,
/// the pixel is taken from the reference (current) frame unchanged.
/// Otherwise, temporal averaging is applied.
pub fn motion_adaptive_average(
    reference: &[u8],
    frames: &[Vec<u8>],
    config: &FrameAvgConfig,
) -> Vec<u8> {
    let averaged = temporal_average(frames, config);
    reference
        .iter()
        .zip(averaged.iter())
        .map(|(&r, &avg)| {
            let diff = (r as i32 - avg as i32).unsigned_abs() as u8;
            if diff > config.motion_threshold {
                r // keep original on motion
            } else {
                avg
            }
        })
        .collect()
}

/// Compute the inter-frame difference magnitude (mean absolute difference).
pub fn inter_frame_mad(frame_a: &[u8], frame_b: &[u8]) -> f32 {
    if frame_a.len() != frame_b.len() || frame_a.is_empty() {
        return 0.0;
    }
    let sum: u32 = frame_a
        .iter()
        .zip(frame_b.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .sum();
    sum as f32 / frame_a.len() as f32
}

/// Detect high-motion pixels between two frames.
///
/// Returns a boolean mask where `true` indicates a motion pixel.
pub fn detect_motion_mask(frame_a: &[u8], frame_b: &[u8], threshold: u8) -> Vec<bool> {
    frame_a
        .iter()
        .zip(frame_b.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u8 > threshold)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(value: u8, len: usize) -> Vec<u8> {
        vec![value; len]
    }

    #[test]
    fn test_uniform_weights_sum_to_one() {
        let config = FrameAvgConfig::uniform(5);
        let weights = compute_weights(&config);
        assert_eq!(weights.len(), 5);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_decay_weights_sum_to_one() {
        let config = FrameAvgConfig {
            window_size: 5,
            strategy: WeightingStrategy::LinearDecay,
            ..Default::default()
        };
        let weights = compute_weights(&config);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_decay_weights_sum_to_one() {
        let config = FrameAvgConfig::default();
        let weights = compute_weights(&config);
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_decay_centre_is_max() {
        let config = FrameAvgConfig::default();
        let weights = compute_weights(&config);
        let centre = config.window_size / 2;
        let centre_w = weights[centre];
        for (i, &w) in weights.iter().enumerate() {
            if i != centre {
                assert!(centre_w >= w);
            }
        }
    }

    #[test]
    fn test_temporal_average_uniform_frames() {
        // All frames the same value: average should be the same value
        let frames = vec![
            make_frame(100, 16),
            make_frame(100, 16),
            make_frame(100, 16),
        ];
        let config = FrameAvgConfig::uniform(3);
        let result = temporal_average(&frames, &config);
        assert_eq!(result, make_frame(100, 16));
    }

    #[test]
    fn test_temporal_average_two_values() {
        // Average of 0 and 200 with uniform weights = 100
        let frames = vec![make_frame(0, 4), make_frame(200, 4)];
        let config = FrameAvgConfig::uniform(2);
        let result = temporal_average(&frames, &config);
        for &v in &result {
            assert!((v as i32 - 100).abs() <= 1);
        }
    }

    #[test]
    fn test_temporal_average_single_frame() {
        let frames = vec![make_frame(77, 8)];
        let config = FrameAvgConfig::uniform(1);
        let result = temporal_average(&frames, &config);
        assert_eq!(result, make_frame(77, 8));
    }

    #[test]
    fn test_motion_adaptive_no_motion() {
        // Reference matches averaged: no motion detected, output = averaged
        let frames = vec![make_frame(128, 16), make_frame(128, 16)];
        let config = FrameAvgConfig::motion_adaptive(20);
        let reference = make_frame(128, 16);
        let result = motion_adaptive_average(&reference, &frames, &config);
        assert_eq!(result, make_frame(128, 16));
    }

    #[test]
    fn test_motion_adaptive_high_motion() {
        // Reference differs greatly from averaged: motion detected, keep reference
        let frames = vec![make_frame(10, 8), make_frame(10, 8)];
        let config = FrameAvgConfig::motion_adaptive(5);
        let reference = make_frame(200, 8);
        let result = motion_adaptive_average(&reference, &frames, &config);
        // All pixels should have been replaced with reference (200)
        assert_eq!(result, make_frame(200, 8));
    }

    #[test]
    fn test_inter_frame_mad_identical() {
        let frame = make_frame(128, 32);
        assert_eq!(inter_frame_mad(&frame, &frame), 0.0);
    }

    #[test]
    fn test_inter_frame_mad_different() {
        let a = make_frame(0, 10);
        let b = make_frame(100, 10);
        let mad = inter_frame_mad(&a, &b);
        assert!((mad - 100.0).abs() < 1e-3);
    }

    #[test]
    fn test_inter_frame_mad_length_mismatch() {
        let a = vec![1u8, 2, 3];
        let b = vec![1u8, 2];
        assert_eq!(inter_frame_mad(&a, &b), 0.0);
    }

    #[test]
    fn test_detect_motion_mask_no_motion() {
        let a = make_frame(100, 8);
        let b = make_frame(100, 8);
        let mask = detect_motion_mask(&a, &b, 10);
        assert!(mask.iter().all(|&m| !m));
    }

    #[test]
    fn test_detect_motion_mask_all_motion() {
        let a = make_frame(0, 8);
        let b = make_frame(200, 8);
        let mask = detect_motion_mask(&a, &b, 10);
        assert!(mask.iter().all(|&m| m));
    }
}
