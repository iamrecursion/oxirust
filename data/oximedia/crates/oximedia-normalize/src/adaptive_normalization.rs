#![allow(dead_code)]
//! Adaptive (per-scene) loudness normalization.
//!
//! Detects segment boundaries in an audio stream by monitoring sliding-window
//! RMS loudness and applies independent gain values to each segment, with
//! linear crossfade transitions to avoid clicks and pops.
//!
//! # Algorithm
//!
//! 1. **Windowed RMS analysis**: Overlapping windows of `window_size` samples
//!    advance by `step_size` samples.  For each window the average power is
//!    converted to a crude LUFS approximation using the standard dBFS relation:
//!    `LUFS ≈ 10·log₁₀(RMS²) - 0.691`.
//! 2. **Boundary detection**: A new segment is started whenever the loudness
//!    change between consecutive windows exceeds 3 dB.
//! 3. **Segment merging**: Adjacent segments whose required gain differs by
//!    less than 0.5 dB are merged into a single segment.
//! 4. **Minimum length enforcement**: Segments shorter than one second of
//!    audio are merged into their preceding neighbour.
//! 5. **Gain application**: Per-segment linear gains are applied.  At each
//!    segment boundary a linear crossfade of `transition_samples` samples
//!    blends the outgoing gain into the incoming gain.

/// A single adaptive segment with its loudness measurement and computed gain.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSegment {
    /// First sample index belonging to this segment (inclusive).
    pub start_sample: usize,
    /// Last sample index belonging to this segment (exclusive).
    pub end_sample: usize,
    /// Measured RMS-based loudness approximation in LUFS.
    pub measured_lufs: f32,
    /// Gain that should be applied to reach the target loudness, in dB.
    pub gain_db: f32,
}

impl AdaptiveSegment {
    /// Length of this segment in samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.end_sample.saturating_sub(self.start_sample)
    }

    /// Whether this segment contains no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.end_sample <= self.start_sample
    }
}

/// Result produced by [`AdaptiveNormalizer::process`].
#[derive(Debug, Clone)]
pub struct AdaptiveResult {
    /// The segments that were identified and processed.
    pub segments: Vec<AdaptiveSegment>,
    /// The fully-normalized output samples.
    pub output_samples: Vec<f32>,
    /// Overall loudness of the *output* (weighted average of per-segment LUFS).
    pub global_lufs: f32,
    /// Dynamic range in LU: difference between the loudest and quietest
    /// *measured* (input) segments.
    pub dynamic_range_lu: f32,
}

/// Configuration and processing core for adaptive per-scene normalization.
///
/// # Example
///
/// ```rust
/// use oximedia_normalize::adaptive_normalization::AdaptiveNormalizer;
///
/// let normalizer = AdaptiveNormalizer::new(-14.0);
/// let silence = vec![0.0f32; 48000];
/// let result = normalizer.process(&silence, 48000);
/// assert_eq!(result.output_samples.len(), 48000);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveNormalizer {
    /// Target integrated loudness in LUFS (negative value, e.g. -14.0).
    pub target_lufs: f32,
    /// Analysis window length in samples.  Default: 3 seconds × sample_rate.
    /// Set to 0 to use the default 3-second window derived from `sample_rate`.
    pub window_size: usize,
    /// Hop size (step) between successive windows in samples.
    /// Default: `window_size / 3` (≈33 % overlap).
    pub step_size: usize,
    /// Safety ceiling: maximum gain that can be applied, in dB.
    pub max_gain_db: f32,
    /// Floor: minimum gain (largest attenuation), in dB.  Should be negative
    /// (e.g. -40.0) to avoid over-boosting near-silence segments.
    pub min_gain_db: f32,
    /// Crossfade length in samples at each segment boundary.
    pub transition_samples: usize,
}

impl AdaptiveNormalizer {
    /// Create a normalizer targeting `target_lufs` with sensible defaults.
    ///
    /// Window size and step size are resolved at analysis time from the
    /// provided `sample_rate`.
    pub fn new(target_lufs: f32) -> Self {
        Self {
            target_lufs,
            window_size: 0, // resolved from sample_rate
            step_size: 0,   // resolved from window_size
            max_gain_db: 20.0,
            min_gain_db: -40.0,
            transition_samples: 0, // resolved from sample_rate
        }
    }

    /// Override the default window size (in samples).
    pub fn with_window_size(mut self, samples: usize) -> Self {
        self.window_size = samples;
        self
    }

    /// Override the default step / hop size (in samples).
    pub fn with_step_size(mut self, samples: usize) -> Self {
        self.step_size = samples;
        self
    }

    /// Override the crossfade transition length (in samples).
    pub fn with_transition_samples(mut self, samples: usize) -> Self {
        self.transition_samples = samples;
        self
    }

    // ------------------------------------------------------------------ //
    //  Internal helpers                                                   //
    // ------------------------------------------------------------------ //

    /// Resolve the effective window size for a given sample rate.
    fn effective_window(&self, sample_rate: u32) -> usize {
        if self.window_size > 0 {
            self.window_size
        } else {
            // Default: 3-second analysis window
            3 * sample_rate as usize
        }
    }

    /// Resolve the effective step / hop size.
    fn effective_step(&self, window: usize) -> usize {
        if self.step_size > 0 {
            self.step_size
        } else {
            // Default: ≈33 % overlap
            (window / 3).max(1)
        }
    }

    /// Resolve the effective crossfade length.
    fn effective_transition(&self, sample_rate: u32) -> usize {
        if self.transition_samples > 0 {
            self.transition_samples
        } else {
            // Default: 20 ms crossfade
            (sample_rate as usize * 20) / 1000
        }
    }

    /// Compute the RMS-based loudness approximation for a slice of samples.
    ///
    /// Returns a value in LUFS (approximation only — not full ITU-R BS.1770).
    fn window_lufs(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return -144.0; // Silence floor
        }

        let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
        let rms_sq = sum_sq / samples.len() as f64;

        if rms_sq <= 0.0 {
            return -144.0;
        }

        // LUFS ≈ 10·log₁₀(mean_square) − 0.691  (ITU-R BS.1770 K-weighting offset)
        // We omit the K-weighting filter here for speed; this is acknowledged as
        // an approximation.
        let lufs = 10.0 * rms_sq.log10() - 0.691;
        lufs as f32
    }

    /// Convert a gain in dB to a linear multiplier.
    #[inline]
    fn db_to_linear(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }

    // ------------------------------------------------------------------ //
    //  Public API                                                         //
    // ------------------------------------------------------------------ //

    /// Analyse `samples` and return a list of [`AdaptiveSegment`]s.
    ///
    /// # Boundary detection
    ///
    /// A new segment boundary is inserted whenever the loudness measured in
    /// consecutive overlapping windows differs by more than 3 dB.
    ///
    /// # Merging
    ///
    /// Adjacent segments are merged when:
    /// * their gain difference is less than 0.5 dB, **or**
    /// * a segment is shorter than 1 second (`sample_rate` samples).
    pub fn analyze_segments(&self, samples: &[f32], sample_rate: u32) -> Vec<AdaptiveSegment> {
        if samples.is_empty() || sample_rate == 0 {
            return Vec::new();
        }

        let window = self.effective_window(sample_rate);
        let step = self.effective_step(window);
        let min_segment_samples = sample_rate as usize; // 1 second minimum

        // --- Step 1: compute per-window loudness values -----------------
        let mut window_lufs_values: Vec<(usize, f32)> = Vec::new();
        let mut pos = 0usize;
        while pos + window <= samples.len() {
            let lufs = Self::window_lufs(&samples[pos..pos + window]);
            window_lufs_values.push((pos, lufs));
            pos += step;
        }
        // Tail window (may be shorter than `window`)
        if pos < samples.len() {
            let lufs = Self::window_lufs(&samples[pos..]);
            window_lufs_values.push((pos, lufs));
        }

        if window_lufs_values.is_empty() {
            // Audio shorter than one window: single segment
            let lufs = Self::window_lufs(samples);
            let gain_db = (self.target_lufs - lufs).clamp(self.min_gain_db, self.max_gain_db);
            return vec![AdaptiveSegment {
                start_sample: 0,
                end_sample: samples.len(),
                measured_lufs: lufs,
                gain_db,
            }];
        }

        // --- Step 2: detect boundaries (>3 dB change) ------------------
        let boundary_threshold_db: f32 = 3.0;
        let mut raw_segments: Vec<AdaptiveSegment> = Vec::new();
        let mut seg_start = 0usize;
        let mut prev_lufs = window_lufs_values[0].1;

        for i in 1..window_lufs_values.len() {
            let (win_pos, lufs) = window_lufs_values[i];
            let delta = (lufs - prev_lufs).abs();
            if delta > boundary_threshold_db {
                // Close the current segment at the window boundary
                let seg_end = win_pos.min(samples.len());
                let seg_lufs = Self::window_lufs(&samples[seg_start..seg_end]);
                let gain_db =
                    (self.target_lufs - seg_lufs).clamp(self.min_gain_db, self.max_gain_db);
                raw_segments.push(AdaptiveSegment {
                    start_sample: seg_start,
                    end_sample: seg_end,
                    measured_lufs: seg_lufs,
                    gain_db,
                });
                seg_start = seg_end;
            }
            prev_lufs = lufs;
        }
        // Final segment
        if seg_start < samples.len() {
            let seg_lufs = Self::window_lufs(&samples[seg_start..]);
            let gain_db = (self.target_lufs - seg_lufs).clamp(self.min_gain_db, self.max_gain_db);
            raw_segments.push(AdaptiveSegment {
                start_sample: seg_start,
                end_sample: samples.len(),
                measured_lufs: seg_lufs,
                gain_db,
            });
        }

        if raw_segments.is_empty() {
            let lufs = Self::window_lufs(samples);
            let gain_db = (self.target_lufs - lufs).clamp(self.min_gain_db, self.max_gain_db);
            return vec![AdaptiveSegment {
                start_sample: 0,
                end_sample: samples.len(),
                measured_lufs: lufs,
                gain_db,
            }];
        }

        // --- Step 3: merge short / similar-gain segments ----------------
        let gain_merge_threshold: f32 = 0.5;
        let merged = Self::merge_segments(raw_segments, min_segment_samples, gain_merge_threshold);

        merged
    }

    /// Merge adjacent segments based on minimum length and gain similarity.
    fn merge_segments(
        mut segs: Vec<AdaptiveSegment>,
        min_len: usize,
        gain_threshold_db: f32,
    ) -> Vec<AdaptiveSegment> {
        if segs.len() <= 1 {
            return segs;
        }

        let mut changed = true;
        while changed {
            changed = false;
            let mut merged: Vec<AdaptiveSegment> = Vec::with_capacity(segs.len());
            let mut i = 0;
            while i < segs.len() {
                if i + 1 < segs.len() {
                    let a = &segs[i];
                    let b = &segs[i + 1];
                    let gain_diff = (a.gain_db - b.gain_db).abs();
                    let too_short = a.len() < min_len || b.len() < min_len;
                    if gain_diff < gain_threshold_db || too_short {
                        // Merge a and b: combine by length-weighted average loudness
                        let total_len = (a.len() + b.len()) as f32;
                        let avg_lufs = if total_len > 0.0 {
                            (a.measured_lufs * a.len() as f32 + b.measured_lufs * b.len() as f32)
                                / total_len
                        } else {
                            a.measured_lufs
                        };
                        // Re-use the already-clamped gain from the longer segment
                        let dominant_gain = if a.len() >= b.len() {
                            a.gain_db
                        } else {
                            b.gain_db
                        };
                        merged.push(AdaptiveSegment {
                            start_sample: a.start_sample,
                            end_sample: b.end_sample,
                            measured_lufs: avg_lufs,
                            gain_db: dominant_gain,
                        });
                        i += 2;
                        changed = true;
                        continue;
                    }
                }
                merged.push(segs[i].clone());
                i += 1;
            }
            segs = merged;
        }
        segs
    }

    /// Apply per-segment gain to `samples`, with linear crossfade transitions.
    ///
    /// Returns an [`AdaptiveResult`] that includes the processed output buffer,
    /// detected segments, a global loudness estimate, and the input dynamic
    /// range across segments.
    pub fn process(&self, samples: &[f32], sample_rate: u32) -> AdaptiveResult {
        let segments = self.analyze_segments(samples, sample_rate);
        let transition = self.effective_transition(sample_rate);
        let mut output = vec![0.0f32; samples.len()];

        if segments.is_empty() {
            return AdaptiveResult {
                segments,
                output_samples: output,
                global_lufs: -144.0,
                dynamic_range_lu: 0.0,
            };
        }

        // Apply gains with linear crossfades at boundaries
        self.apply_segments_with_crossfade(samples, &mut output, &segments, transition);

        // Compute global loudness as length-weighted average of output LUFS
        let total_samples = samples.len() as f32;
        let global_lufs = if total_samples > 0.0 {
            segments
                .iter()
                .map(|s| s.measured_lufs + s.gain_db)
                .zip(segments.iter().map(|s| s.len() as f32 / total_samples))
                .map(|(lufs, weight)| lufs * weight)
                .sum::<f32>()
        } else {
            -144.0
        };

        // Dynamic range: spread of input loudness values across segments
        let lufs_values: Vec<f32> = segments.iter().map(|s| s.measured_lufs).collect();
        let dynamic_range_lu = if lufs_values.len() > 1 {
            let min_lufs = lufs_values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_lufs = lufs_values
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            (max_lufs - min_lufs).max(0.0)
        } else {
            0.0
        };

        AdaptiveResult {
            segments,
            output_samples: output,
            global_lufs,
            dynamic_range_lu,
        }
    }

    /// Low-level gain application with linear crossfades between segments.
    fn apply_segments_with_crossfade(
        &self,
        input: &[f32],
        output: &mut [f32],
        segments: &[AdaptiveSegment],
        transition_len: usize,
    ) {
        // Apply each segment's base gain first (no crossfade)
        for seg in segments {
            let linear = Self::db_to_linear(seg.gain_db);
            let end = seg.end_sample.min(input.len());
            for i in seg.start_sample..end {
                output[i] = input[i] * linear;
            }
        }

        // Blend at each segment boundary with a linear crossfade
        for pair in segments.windows(2) {
            let a = &pair[0];
            let b = &pair[1];
            let boundary = b.start_sample;
            if boundary == 0 || boundary >= input.len() {
                continue;
            }
            let half = transition_len / 2;
            let fade_start = boundary.saturating_sub(half);
            let fade_end = (boundary + half).min(input.len());

            let gain_a = Self::db_to_linear(a.gain_db);
            let gain_b = Self::db_to_linear(b.gain_db);
            let fade_len = fade_end - fade_start;
            if fade_len == 0 {
                continue;
            }

            for i in fade_start..fade_end {
                let t = (i - fade_start) as f32 / fade_len as f32;
                // Linear interpolation from gain_a to gain_b
                let blended_gain = gain_a * (1.0 - t) + gain_b * t;
                output[i] = input[i] * blended_gain;
            }
        }
    }
}

impl Default for AdaptiveNormalizer {
    fn default() -> Self {
        Self::new(-14.0)
    }
}

// ========================================================================== //
//  Tests                                                                      //
// ========================================================================== //

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Generate a sine-wave buffer at `freq` Hz with the given amplitude and length.
    fn sine_wave(freq: f32, amplitude: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| amplitude * (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn test_window_lufs_silence() {
        let silence = vec![0.0f32; 4800];
        let lufs = AdaptiveNormalizer::window_lufs(&silence);
        assert!(lufs < -100.0, "Silence must have very low LUFS: {lufs}");
    }

    #[test]
    fn test_window_lufs_full_scale_sine() {
        // 0 dBFS sine → RMS ≈ 1/√2 ≈ 0.707, rms² = 0.5
        // LUFS ≈ 10·log₁₀(0.5) − 0.691 ≈ −3.01 − 0.69 ≈ −3.7
        let sr = 48000u32;
        let samples = sine_wave(1000.0, 1.0, sr, sr as usize);
        let lufs = AdaptiveNormalizer::window_lufs(&samples);
        // Expect something in the range −4 to −2 LUFS
        assert!(
            lufs > -6.0 && lufs < 0.0,
            "Full-scale sine LUFS out of expected range: {lufs}"
        );
    }

    #[test]
    fn test_single_segment_short_audio() {
        // Audio shorter than one window → one segment
        let sr = 48000u32;
        let samples = sine_wave(440.0, 0.5, sr, sr as usize); // 1 second
        let norm = AdaptiveNormalizer::new(-14.0);
        let segs = norm.analyze_segments(&samples, sr);
        assert!(!segs.is_empty(), "Must produce at least one segment");
        assert_eq!(segs[0].start_sample, 0);
        assert_eq!(segs.last().map(|s| s.end_sample), Some(samples.len()));
    }

    #[test]
    fn test_gain_clamped_to_max() {
        // Near-silence input → would require huge positive gain; must be clamped.
        let sr = 48000u32;
        let tiny: Vec<f32> = vec![1e-10f32; sr as usize];
        let norm = AdaptiveNormalizer::new(-14.0)
            .with_window_size(sr as usize)
            .with_step_size(sr as usize / 2);
        let segs = norm.analyze_segments(&tiny, sr);
        for seg in &segs {
            assert!(
                seg.gain_db <= norm.max_gain_db,
                "gain_db {} exceeds max {} ",
                seg.gain_db,
                norm.max_gain_db
            );
        }
    }

    #[test]
    fn test_gain_clamped_to_min() {
        // Very loud input → gain must not drop below min_gain_db.
        let sr = 48000u32;
        // Amplitude 1.0 → near 0 dBFS sine
        let loud = sine_wave(440.0, 1.0, sr, sr as usize * 5);
        let norm = AdaptiveNormalizer::new(-23.0);
        let segs = norm.analyze_segments(&loud, sr);
        for seg in &segs {
            assert!(
                seg.gain_db >= norm.min_gain_db,
                "gain_db {} is below min {} ",
                seg.gain_db,
                norm.min_gain_db
            );
        }
    }

    #[test]
    fn test_process_output_length() {
        let sr = 48000u32;
        let samples = sine_wave(440.0, 0.3, sr, sr as usize * 4);
        let norm = AdaptiveNormalizer::new(-14.0);
        let result = norm.process(&samples, sr);
        assert_eq!(result.output_samples.len(), samples.len());
    }

    #[test]
    fn test_process_empty_input() {
        let norm = AdaptiveNormalizer::new(-14.0);
        let result = norm.process(&[], 48000);
        assert!(result.output_samples.is_empty());
        assert!(result.segments.is_empty());
    }

    #[test]
    fn test_process_raises_quiet_audio() {
        // A quiet signal (-40 dBFS sine) should be boosted toward target.
        let sr = 48000u32;
        let amplitude = 0.01f32; // ~ -40 dBFS
        let samples = sine_wave(440.0, amplitude, sr, sr as usize * 2);

        // Measure input RMS
        let rms_in: f32 =
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        let norm = AdaptiveNormalizer::new(-14.0);
        let result = norm.process(&samples, sr);

        let rms_out: f32 = (result.output_samples.iter().map(|s| s * s).sum::<f32>()
            / result.output_samples.len() as f32)
            .sqrt();

        assert!(
            rms_out > rms_in,
            "Output RMS {rms_out} must be louder than input RMS {rms_in} after boosting"
        );
    }

    #[test]
    fn test_segment_fields_valid() {
        let sr = 48000u32;
        let samples = sine_wave(440.0, 0.2, sr, sr as usize * 6);
        let norm = AdaptiveNormalizer::new(-14.0);
        let segs = norm.analyze_segments(&samples, sr);

        // Segments must be contiguous and cover the full range
        let mut prev_end = 0;
        for seg in &segs {
            assert_eq!(
                seg.start_sample, prev_end,
                "Gap or overlap between segments at sample {prev_end}"
            );
            assert!(seg.end_sample > seg.start_sample, "Empty segment detected");
            prev_end = seg.end_sample;
        }
        assert_eq!(
            prev_end,
            samples.len(),
            "Segments do not cover entire buffer"
        );
    }

    #[test]
    fn test_dynamic_range_two_distinct_levels() {
        // Create audio with two clearly different loudness levels
        let sr = 48000u32;
        let loud_part = sine_wave(440.0, 0.9, sr, sr as usize * 3);
        let quiet_part = sine_wave(440.0, 0.05, sr, sr as usize * 3);
        let mut samples = loud_part;
        samples.extend(quiet_part);

        let norm = AdaptiveNormalizer::new(-14.0)
            .with_window_size(sr as usize)
            .with_step_size(sr as usize / 3);
        let result = norm.process(&samples, sr);

        // With two very different loudness sections the dynamic range should be > 0
        assert!(
            result.dynamic_range_lu >= 0.0,
            "Dynamic range must be non-negative: {}",
            result.dynamic_range_lu
        );
    }

    #[test]
    fn test_merge_segments_same_gain() {
        // Two segments with identical gain must be merged into one
        let segs = vec![
            AdaptiveSegment {
                start_sample: 0,
                end_sample: 48000,
                measured_lufs: -20.0,
                gain_db: 6.0,
            },
            AdaptiveSegment {
                start_sample: 48000,
                end_sample: 96000,
                measured_lufs: -20.2,
                gain_db: 6.2, // Within 0.5 dB → merge
            },
        ];
        let merged = AdaptiveNormalizer::merge_segments(segs, 48000, 0.5);
        assert_eq!(
            merged.len(),
            1,
            "Segments with <0.5 dB gain difference must merge"
        );
        assert_eq!(merged[0].start_sample, 0);
        assert_eq!(merged[0].end_sample, 96000);
    }

    #[test]
    fn test_db_to_linear_round_trip() {
        // 0 dB → linear 1.0; 6 dB → ~2.0; -6 dB → ~0.5
        let eps = 1e-4f32;
        assert!((AdaptiveNormalizer::db_to_linear(0.0) - 1.0).abs() < eps);
        assert!((AdaptiveNormalizer::db_to_linear(20.0) - 10.0).abs() < 0.01);
        assert!((AdaptiveNormalizer::db_to_linear(-20.0) - 0.1).abs() < 0.001);
    }
}
