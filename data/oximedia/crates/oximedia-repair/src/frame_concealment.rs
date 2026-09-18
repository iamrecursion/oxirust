//! Frame concealment strategies.
//!
//! When frames are lost or corrupted beyond repair, concealment is used to
//! substitute plausible data so that playback remains watchable.

/// Strategy used to conceal a missing or damaged frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConcealmentStrategy {
    /// Repeat the last good frame before the gap.
    CopyPrevious,
    /// Use the first good frame after the gap.
    CopyNext,
    /// Blend neighbouring frames temporally.
    Interpolate,
    /// Weighted average of several surrounding frames.
    BlendNeighbors,
    /// Insert a solid black frame.
    InsertBlack,
    /// Insert silence samples (audio gap concealment).
    InsertSilence,
}

impl ConcealmentStrategy {
    /// Return a subjective quality score for the strategy (0.0 = worst, 1.0 = best).
    #[must_use]
    pub fn quality_score(&self) -> f64 {
        match self {
            Self::Interpolate => 0.90,
            Self::BlendNeighbors => 0.85,
            Self::CopyPrevious => 0.70,
            Self::CopyNext => 0.70,
            Self::InsertSilence => 0.50,
            Self::InsertBlack => 0.20,
        }
    }

    /// Return `true` when the strategy uses neighbouring frames temporally.
    #[must_use]
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            Self::CopyPrevious | Self::CopyNext | Self::Interpolate | Self::BlendNeighbors
        )
    }
}

/// The outcome of applying concealment to a set of frames.
#[derive(Debug, Clone)]
pub struct ConcealmentResult {
    /// Strategy that was applied.
    pub strategy: ConcealmentStrategy,
    /// Estimated quality of the concealment (0.0–1.0).
    pub quality: f64,
    /// Indices of frames to which concealment was applied.
    pub applied_to_frames: Vec<u64>,
}

/// Choose the best available concealment strategy given the context.
///
/// - `error_count`: number of consecutive bad frames in the gap.
/// - `consecutive`: whether the bad frames form a single run.
/// - `has_prev`: whether a reference frame before the gap is available.
/// - `has_next`: whether a reference frame after the gap is available.
#[must_use]
pub fn select_strategy(
    error_count: usize,
    consecutive: bool,
    has_prev: bool,
    has_next: bool,
) -> ConcealmentStrategy {
    match (has_prev, has_next, consecutive, error_count) {
        // Both neighbours available and short gap → interpolate
        (true, true, true, 1..=4) => ConcealmentStrategy::Interpolate,
        // Both available, longer gap → blend
        (true, true, _, _) => ConcealmentStrategy::BlendNeighbors,
        // Only previous available
        (true, false, _, _) => ConcealmentStrategy::CopyPrevious,
        // Only next available
        (false, true, _, _) => ConcealmentStrategy::CopyNext,
        // Nothing available → insert black
        _ => ConcealmentStrategy::InsertBlack,
    }
}

/// Blend two raw frame buffers together using a linear weight `alpha`.
///
/// `alpha` = 0.0 → fully `prev`; `alpha` = 1.0 → fully `next`.
/// Both slices must have the same length; if they differ the shorter length is used.
#[must_use]
pub fn interpolate_frame(prev: &[u8], next: &[u8], alpha: f64) -> Vec<u8> {
    let alpha = alpha.clamp(0.0, 1.0);
    let len = prev.len().min(next.len());
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let p = prev[i] as f64;
        let n = next[i] as f64;
        let blended = p + (n - p) * alpha;
        out.push(blended.round().clamp(0.0, 255.0) as u8);
    }
    out
}

/// Generate concealment audio for a silent gap.
///
/// Produces a smooth fade-out from `prev_samples` followed by silence for
/// `gap_frames` frames, then a fade-in from silence, all at `sample_rate`.
/// The number of samples per frame is derived as `sample_rate / 24` (PAL-ish).
///
/// Returns the generated concealment samples.
#[must_use]
pub fn conceal_audio_gap(prev_samples: &[f64], gap_frames: usize, sample_rate: u32) -> Vec<f64> {
    let samples_per_frame = (sample_rate / 24).max(1) as usize;
    let total_samples = gap_frames * samples_per_frame;

    let mut out = Vec::with_capacity(total_samples);

    // Use the last few samples of `prev_samples` for the fade-out envelope
    let tail_len = prev_samples.len().min(samples_per_frame * 2);
    let tail = &prev_samples[prev_samples.len().saturating_sub(tail_len)..];

    let half = total_samples / 2;

    for i in 0..total_samples {
        let sample = if !tail.is_empty() {
            tail[i.min(tail.len() - 1)]
        } else {
            0.0
        };

        let envelope = if total_samples <= 1 {
            0.0
        } else if i < half {
            // Fade out
            1.0 - (i as f64 / half as f64)
        } else {
            // Fade in (silent → small residual)
            (i - half) as f64 / half as f64 * 0.05
        };

        out.push(sample * envelope);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_quality_score_ordering() {
        assert!(
            ConcealmentStrategy::Interpolate.quality_score()
                > ConcealmentStrategy::CopyPrevious.quality_score()
        );
        assert!(
            ConcealmentStrategy::CopyPrevious.quality_score()
                > ConcealmentStrategy::InsertBlack.quality_score()
        );
    }

    #[test]
    fn test_strategy_quality_score_range() {
        let strategies = [
            ConcealmentStrategy::CopyPrevious,
            ConcealmentStrategy::CopyNext,
            ConcealmentStrategy::Interpolate,
            ConcealmentStrategy::BlendNeighbors,
            ConcealmentStrategy::InsertBlack,
            ConcealmentStrategy::InsertSilence,
        ];
        for s in strategies {
            let q = s.quality_score();
            assert!(q >= 0.0 && q <= 1.0, "quality out of range for {:?}", s);
        }
    }

    #[test]
    fn test_is_temporal_true() {
        assert!(ConcealmentStrategy::Interpolate.is_temporal());
        assert!(ConcealmentStrategy::CopyPrevious.is_temporal());
        assert!(ConcealmentStrategy::CopyNext.is_temporal());
        assert!(ConcealmentStrategy::BlendNeighbors.is_temporal());
    }

    #[test]
    fn test_is_temporal_false() {
        assert!(!ConcealmentStrategy::InsertBlack.is_temporal());
        assert!(!ConcealmentStrategy::InsertSilence.is_temporal());
    }

    #[test]
    fn test_select_strategy_both_neighbours_short_gap() {
        assert_eq!(
            select_strategy(2, true, true, true),
            ConcealmentStrategy::Interpolate
        );
    }

    #[test]
    fn test_select_strategy_both_neighbours_long_gap() {
        assert_eq!(
            select_strategy(10, true, true, true),
            ConcealmentStrategy::BlendNeighbors
        );
    }

    #[test]
    fn test_select_strategy_only_prev() {
        assert_eq!(
            select_strategy(3, true, true, false),
            ConcealmentStrategy::CopyPrevious
        );
    }

    #[test]
    fn test_select_strategy_only_next() {
        assert_eq!(
            select_strategy(3, false, false, true),
            ConcealmentStrategy::CopyNext
        );
    }

    #[test]
    fn test_select_strategy_no_neighbours() {
        assert_eq!(
            select_strategy(5, false, false, false),
            ConcealmentStrategy::InsertBlack
        );
    }

    #[test]
    fn test_interpolate_frame_alpha_zero() {
        let prev = vec![100u8, 150, 200];
        let next = vec![0u8, 0, 0];
        let result = interpolate_frame(&prev, &next, 0.0);
        assert_eq!(result, vec![100, 150, 200]);
    }

    #[test]
    fn test_interpolate_frame_alpha_one() {
        let prev = vec![0u8, 0, 0];
        let next = vec![100u8, 150, 200];
        let result = interpolate_frame(&prev, &next, 1.0);
        assert_eq!(result, vec![100, 150, 200]);
    }

    #[test]
    fn test_interpolate_frame_alpha_half() {
        let prev = vec![0u8, 100];
        let next = vec![100u8, 0];
        let result = interpolate_frame(&prev, &next, 0.5);
        assert_eq!(result, vec![50, 50]);
    }

    #[test]
    fn test_interpolate_frame_different_lengths() {
        let prev = vec![10u8, 20, 30, 40];
        let next = vec![0u8, 0];
        let result = interpolate_frame(&prev, &next, 0.0);
        // Uses min length = 2
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_conceal_audio_gap_output_length() {
        let prev = vec![0.5f64; 48];
        let result = conceal_audio_gap(&prev, 4, 48_000);
        // 4 frames * (48_000 / 24) = 4 * 2000 = 8000 samples
        assert_eq!(result.len(), 8_000);
    }

    #[test]
    fn test_conceal_audio_gap_starts_near_amplitude() {
        let prev = vec![1.0f64; 100];
        let result = conceal_audio_gap(&prev, 2, 48_000);
        // First sample should be close to 1.0 (fade starts at full)
        assert!(result[0] > 0.9);
    }

    #[test]
    fn test_conceal_audio_gap_empty_prev() {
        let result = conceal_audio_gap(&[], 2, 24_000);
        assert_eq!(result.len(), 2 * (24_000 / 24) as usize);
        // All zeros since there are no previous samples to draw from
        for s in &result {
            assert!((s - 0.0).abs() < f64::EPSILON);
        }
    }
}
