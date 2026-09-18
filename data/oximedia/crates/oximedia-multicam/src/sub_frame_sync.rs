//! Sub-frame synchronization precision for multi-camera production.
//!
//! Provides [`SubFrameSync`] which computes fractional-frame offsets between
//! camera angles by interpolating between discrete frame positions.  This
//! allows timing accuracy better than a single frame period — important for
//! high-frame-rate material (120 fps, 240 fps) and for audio-visual alignment
//! where a single frame boundary represents ~8 ms at 120 fps.
//!
//! # Algorithm
//!
//! Given two frame-timestamped signals (video or audio), the integer frame
//! offset is determined first (e.g. by [`crate::sync::cross_correlate`]).
//! The residual sub-frame part is then estimated by fitting a parabola through
//! the cross-correlation values at the neighbourhood of the integer peak and
//! reading off its fractional maximum.  This is the same Sinc/parabolic trick
//! used in professional video analysis tools.

// ── SubFrameOffset ────────────────────────────────────────────────────────────

/// The offset between a camera angle and the reference, expressed with
/// sub-frame precision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubFrameOffset {
    /// Angle identifier (0-based index).
    pub angle_id: usize,
    /// Integer part of the offset in frames (can be negative).
    pub integer_frames: i64,
    /// Fractional part of the offset in frames, in \[–0.5, +0.5\].
    pub fractional_frames: f64,
    /// Confidence of the offset estimate in \[0.0, 1.0\].
    pub confidence: f64,
}

impl SubFrameOffset {
    /// Create a `SubFrameOffset`.
    ///
    /// The `fractional_frames` value is clamped to \[–0.5, 0.5\].
    #[must_use]
    pub fn new(
        angle_id: usize,
        integer_frames: i64,
        fractional_frames: f64,
        confidence: f64,
    ) -> Self {
        Self {
            angle_id,
            integer_frames,
            fractional_frames: fractional_frames.clamp(-0.5, 0.5),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Total offset as a floating-point frame count (integer + fractional).
    #[must_use]
    pub fn total_frames(&self) -> f64 {
        self.integer_frames as f64 + self.fractional_frames
    }

    /// Convert to seconds at the given `frame_rate`.
    #[must_use]
    pub fn to_seconds(&self, frame_rate: f64) -> f64 {
        if frame_rate <= 0.0 {
            return 0.0;
        }
        self.total_frames() / frame_rate
    }

    /// Convert to samples at the given `sample_rate` and `frame_rate`.
    #[must_use]
    pub fn to_samples(&self, sample_rate: u32, frame_rate: f64) -> i64 {
        if frame_rate <= 0.0 {
            return 0;
        }
        let samples_per_frame = f64::from(sample_rate) / frame_rate;
        (self.total_frames() * samples_per_frame).round() as i64
    }

    /// `true` when the fractional part is negligibly small (< 0.01 frames).
    #[must_use]
    pub fn is_frame_aligned(&self) -> bool {
        self.fractional_frames.abs() < 0.01
    }
}

// ── SubFrameSync ──────────────────────────────────────────────────────────────

/// Sub-frame synchronization engine.
///
/// Accepts a sequence of correlation samples around the peak of a
/// cross-correlation function and refines the integer lag estimate to
/// sub-frame accuracy using parabolic interpolation.
///
/// # Usage
///
/// ```
/// use oximedia_multicam::sub_frame_sync::SubFrameSync;
///
/// let sync = SubFrameSync::new(24.0);
///
/// // Three correlation values around the integer peak at lag 5:
/// // corr[4]=0.72, corr[5]=0.98, corr[6]=0.75
/// let offset = sync
///     .refine_offset(2, 5, &[0.72, 0.98, 0.75])
///     .expect("should succeed");
///
/// // Fractional component is small (peak is close to integer lag 5).
/// assert!(offset.total_frames().abs() - 5.0 < 0.3);
/// ```
#[derive(Debug, Clone)]
pub struct SubFrameSync {
    /// Frame rate (fps).
    pub frame_rate: f64,
}

impl SubFrameSync {
    /// Create a new `SubFrameSync` for the given frame rate.
    ///
    /// # Panics
    ///
    /// Does not panic; a `frame_rate ≤ 0` simply makes `to_seconds` return 0.
    #[must_use]
    pub fn new(frame_rate: f64) -> Self {
        Self { frame_rate }
    }

    /// Refine an integer frame `lag` to sub-frame accuracy using parabolic
    /// interpolation over `corr_window`.
    ///
    /// `corr_window` must contain **at least three** values ordered as
    /// `[y_before, y_peak, y_after]` corresponding to lags
    /// `[lag-1, lag, lag+1]`.  If more values are provided the middle element
    /// (index `corr_window.len() / 2`) is used as the peak.
    ///
    /// # Errors
    ///
    /// Returns an error string when `corr_window` has fewer than three elements.
    pub fn refine_offset(
        &self,
        angle_id: usize,
        integer_lag: i64,
        corr_window: &[f64],
    ) -> Result<SubFrameOffset, String> {
        if corr_window.len() < 3 {
            return Err(format!(
                "corr_window must have at least 3 elements, got {}",
                corr_window.len()
            ));
        }

        let mid = corr_window.len() / 2;
        let y_m = corr_window[mid.saturating_sub(1)];
        let y_0 = corr_window[mid];
        let y_p = corr_window[mid + 1];

        let fractional = Self::parabolic_peak_offset(y_m, y_0, y_p);
        let confidence = y_0.clamp(0.0, 1.0);

        Ok(SubFrameOffset::new(
            angle_id,
            integer_lag,
            fractional,
            confidence,
        ))
    }

    /// Compute a sub-frame offset from a dense cross-correlation sequence.
    ///
    /// Finds the index of the global maximum in `xcorr`, then applies
    /// parabolic interpolation to obtain the fractional maximum location.
    ///
    /// The result's integer component is `peak_index - center_bias` where
    /// `center_bias = xcorr.len() / 2`.  This convention treats index 0 as
    /// lag `–N/2` (b leads a) and index N–1 as lag `+N/2 – 1` (b is delayed).
    ///
    /// # Errors
    ///
    /// Returns an error when `xcorr` is empty.
    pub fn find_sub_frame_offset(
        &self,
        angle_id: usize,
        xcorr: &[f64],
    ) -> Result<SubFrameOffset, String> {
        if xcorr.is_empty() {
            return Err("xcorr must not be empty".into());
        }

        // Locate the peak.
        let (peak_idx, &peak_val) = xcorr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or("empty xcorr")?;

        // Parabolic interpolation (use neighbours if available).
        let fractional_idx = if peak_idx > 0 && peak_idx + 1 < xcorr.len() {
            let y_m = xcorr[peak_idx - 1];
            let y_0 = peak_val;
            let y_p = xcorr[peak_idx + 1];
            peak_idx as f64 + Self::parabolic_peak_offset(y_m, y_0, y_p)
        } else {
            peak_idx as f64
        };

        let center = xcorr.len() as f64 / 2.0;
        let total_fractional = fractional_idx - center;
        let integer_lag = total_fractional.round() as i64;
        let frac = total_fractional - integer_lag as f64;

        Ok(SubFrameOffset::new(
            angle_id,
            integer_lag,
            frac,
            peak_val.clamp(0.0, 1.0),
        ))
    }

    /// Parabolic peak offset in \[–0.5, +0.5\].
    ///
    /// Given three consecutive correlation values `y_m`, `y_0`, `y_p` at
    /// positions `[lag-1, lag, lag+1]`, returns the fractional shift of the
    /// true maximum from `lag`.
    ///
    /// The formula is: `delta = (y_p - y_m) / (2 * (2*y_0 - y_m - y_p))`
    /// A positive delta means the true peak is to the right (higher lag).
    fn parabolic_peak_offset(y_m: f64, y_0: f64, y_p: f64) -> f64 {
        let denom = 2.0 * (2.0 * y_0 - y_m - y_p);
        if denom.abs() < f64::EPSILON {
            return 0.0;
        }
        ((y_p - y_m) / denom).clamp(-0.5, 0.5)
    }

    /// Apply a sub-frame offset to a sample index.
    ///
    /// Converts the offset to samples (using `frame_rate` and `sample_rate`)
    /// and adds it to `sample_index`.  Saturates rather than wraps on overflow.
    #[must_use]
    pub fn apply_to_sample(
        &self,
        sample_index: i64,
        offset: &SubFrameOffset,
        sample_rate: u32,
    ) -> i64 {
        let delta = offset.to_samples(sample_rate, self.frame_rate);
        sample_index.saturating_add(delta)
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SubFrameOffset ───────────────────────────────────────────────────────

    #[test]
    fn test_total_frames_combines_integer_and_fractional() {
        let o = SubFrameOffset::new(0, 5, 0.25, 0.9);
        let expected = 5.25;
        assert!((o.total_frames() - expected).abs() < 1e-9);
    }

    #[test]
    fn test_fractional_clamped_to_half_frame() {
        let o = SubFrameOffset::new(0, 0, 0.9, 1.0);
        assert!((o.fractional_frames - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_fractional_negative_clamped() {
        let o = SubFrameOffset::new(0, 0, -0.9, 1.0);
        assert!((o.fractional_frames + 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_to_seconds_at_24fps() {
        let o = SubFrameOffset::new(1, 24, 0.0, 1.0);
        let secs = o.to_seconds(24.0);
        assert!((secs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_to_samples_at_48khz_24fps() {
        let o = SubFrameOffset::new(0, 24, 0.0, 1.0);
        let samples = o.to_samples(48_000, 24.0);
        assert_eq!(samples, 48_000);
    }

    #[test]
    fn test_is_frame_aligned_true() {
        let o = SubFrameOffset::new(0, 5, 0.005, 1.0);
        assert!(o.is_frame_aligned());
    }

    #[test]
    fn test_is_frame_aligned_false() {
        let o = SubFrameOffset::new(0, 5, 0.3, 1.0);
        assert!(!o.is_frame_aligned());
    }

    // ── SubFrameSync::refine_offset ─────────────────────────────────────────

    /// Symmetric peak: fractional offset should be ~0.
    #[test]
    fn test_refine_offset_symmetric_peak() {
        let sync = SubFrameSync::new(25.0);
        // Symmetric: y[-1]=y[+1] => fractional = 0
        let corr = [0.70, 0.95, 0.70];
        let o = sync.refine_offset(0, 10, &corr).expect("should succeed");
        assert!(
            o.fractional_frames.abs() < 0.01,
            "Expected ~0 fractional, got {}",
            o.fractional_frames
        );
        assert_eq!(o.integer_frames, 10);
    }

    /// Peak skewed right → fractional should be positive.
    #[test]
    fn test_refine_offset_skewed_right() {
        let sync = SubFrameSync::new(25.0);
        let corr = [0.60, 0.90, 0.82];
        let o = sync.refine_offset(1, 3, &corr).expect("should succeed");
        assert!(
            o.fractional_frames > 0.0,
            "Expected positive fractional, got {}",
            o.fractional_frames
        );
    }

    /// Peak skewed left → fractional should be negative.
    #[test]
    fn test_refine_offset_skewed_left() {
        let sync = SubFrameSync::new(25.0);
        let corr = [0.82, 0.90, 0.60];
        let o = sync.refine_offset(1, 3, &corr).expect("should succeed");
        assert!(
            o.fractional_frames < 0.0,
            "Expected negative fractional, got {}",
            o.fractional_frames
        );
    }

    /// Too-short window returns error.
    #[test]
    fn test_refine_offset_too_short_returns_error() {
        let sync = SubFrameSync::new(25.0);
        assert!(sync.refine_offset(0, 0, &[0.5, 0.8]).is_err());
    }

    // ── SubFrameSync::find_sub_frame_offset ─────────────────────────────────

    /// A cross-correlation array with a clear peak at the centre → near-zero lag.
    #[test]
    fn test_find_sub_frame_offset_centre_peak() {
        let sync = SubFrameSync::new(24.0);
        // 11-element xcorr, peak at index 5 (exactly the centre).
        // center = 11/2.0 = 5.5; fractional_idx ≈ 5; total = 5 - 5.5 = -0.5
        // integer_lag = round(-0.5) = 0 or -1 depending on rounding. Use 11 elements
        // so center lands on an integer: center = 5 when len is odd and we use
        // len/2 integer division.
        // With even-length we get better results: len=10 → center=5.0
        let xcorr = [0.1, 0.2, 0.4, 0.7, 0.85, 0.95, 0.85, 0.7, 0.4, 0.2];
        // len=10, peak at index 5, center = 10/2.0 = 5.0
        // total_fractional = 5.0 - 5.0 = 0.0 → lag=0, frac=0.0
        let o = sync
            .find_sub_frame_offset(0, &xcorr)
            .expect("should succeed");
        assert!(
            o.total_frames().abs() <= 0.5,
            "Expected near-zero lag, got {}",
            o.total_frames()
        );
    }

    /// Peak shifted one position right of centre → lag ≈ +1.
    #[test]
    fn test_find_sub_frame_offset_one_right() {
        let sync = SubFrameSync::new(24.0);
        // 10-element xcorr, peak at index 6 (one right of centre=5)
        let xcorr = [0.1, 0.2, 0.3, 0.5, 0.75, 0.85, 0.95, 0.85, 0.75, 0.5];
        let o = sync
            .find_sub_frame_offset(1, &xcorr)
            .expect("should succeed");
        // total_fractional = 6.0 - 5.0 = 1.0 → lag ≈ +1
        assert!(
            (o.total_frames() - 1.0).abs() <= 0.6,
            "Expected lag ≈ 1, got {}",
            o.total_frames()
        );
    }

    /// Empty xcorr returns error.
    #[test]
    fn test_find_sub_frame_offset_empty_error() {
        let sync = SubFrameSync::new(24.0);
        assert!(sync.find_sub_frame_offset(0, &[]).is_err());
    }

    // ── apply_to_sample ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_to_sample_shifts_correctly() {
        // 24 fps, 48 kHz: 1 frame = 2000 samples
        let sync = SubFrameSync::new(24.0);
        let offset = SubFrameOffset::new(0, 1, 0.0, 1.0); // 1 frame
        let result = sync.apply_to_sample(10_000, &offset, 48_000);
        assert_eq!(result, 12_000); // 10000 + 2000
    }
}
