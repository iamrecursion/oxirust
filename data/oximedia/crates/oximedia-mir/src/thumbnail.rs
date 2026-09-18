//! Audio thumbnailing — extract the most representative 15–30 second clip.
//!
//! The algorithm scores each candidate window by combining:
//! 1. **Energy** — prefer loud segments (choruses tend to be louder than intros).
//! 2. **Spectral brightness** — prefer bright, harmonically active regions.
//! 3. **Onset density** — prefer rhythmically active regions.
//! 4. **Centrality penalty** — mildly prefer the first half of a track (so
//!    the thumbnail starts before the song has ended).
//!
//! All scoring is done with `Vec<f32>` and manual stride arithmetic — no ndarray.

use crate::utils::{hann_window, magnitude_spectrum};
use crate::MirResult;

/// Result of audio thumbnailing.
#[derive(Debug, Clone)]
pub struct ThumbnailResult {
    /// Start sample index of the best clip.
    pub start_sample: usize,
    /// End sample index (exclusive) of the best clip.
    pub end_sample: usize,
    /// Start time in seconds.
    pub start_secs: f32,
    /// End time in seconds.
    pub end_secs: f32,
    /// Duration of the clip in seconds.
    pub duration_secs: f32,
    /// Score of the selected region (higher is more representative).
    pub score: f32,
}

/// Configuration for the thumbnailing algorithm.
#[derive(Debug, Clone)]
pub struct ThumbnailConfig {
    /// Target thumbnail duration in seconds.
    pub target_duration_secs: f32,
    /// Minimum thumbnail duration in seconds.
    pub min_duration_secs: f32,
    /// Maximum thumbnail duration in seconds.
    pub max_duration_secs: f32,
    /// FFT window size for spectral analysis.
    pub window_size: usize,
    /// Hop size between analysis windows.
    pub hop_size: usize,
    /// Weight for the energy feature (0–1).
    pub energy_weight: f32,
    /// Weight for spectral brightness (0–1).
    pub brightness_weight: f32,
    /// Weight for onset density (0–1).
    pub onset_weight: f32,
    /// Penalty for placing the thumbnail near the very end of the track (0–1).
    pub centrality_weight: f32,
}

impl Default for ThumbnailConfig {
    fn default() -> Self {
        Self {
            target_duration_secs: 20.0,
            min_duration_secs: 15.0,
            max_duration_secs: 30.0,
            window_size: 2048,
            hop_size: 512,
            energy_weight: 0.35,
            brightness_weight: 0.25,
            onset_weight: 0.25,
            centrality_weight: 0.15,
        }
    }
}

/// Audio thumbnailer.
pub struct AudioThumbnail {
    sample_rate: f32,
    config: ThumbnailConfig,
}

impl AudioThumbnail {
    /// Create a new thumbnailer.
    #[must_use]
    pub fn new(sample_rate: f32, config: ThumbnailConfig) -> Self {
        Self {
            sample_rate,
            config,
        }
    }

    /// Create a thumbnailer with defaults for the given sample rate.
    #[must_use]
    pub fn with_sample_rate(sample_rate: f32) -> Self {
        Self::new(sample_rate, ThumbnailConfig::default())
    }

    /// Extract the most representative clip from `signal`.
    ///
    /// Returns an error if the signal is shorter than `min_duration_secs`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::MirError::InsufficientData`] when the signal is too short.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    pub fn extract(&self, signal: &[f32]) -> MirResult<ThumbnailResult> {
        let sr = self.sample_rate;
        let min_samples = (self.config.min_duration_secs * sr) as usize;
        if signal.len() < min_samples {
            return Err(crate::MirError::InsufficientData(format!(
                "Signal too short for thumbnailing: need ≥{:.1}s, got {:.1}s",
                self.config.min_duration_secs,
                signal.len() as f32 / sr
            )));
        }

        let win = self.config.window_size;
        let hop = self.config.hop_size;
        let total_frames = (signal.len().saturating_sub(win)) / hop + 1;

        // ── Frame-level features ──────────────────────────────────────────
        let mut energy_env = Vec::with_capacity(total_frames);
        let mut brightness_env = Vec::with_capacity(total_frames);
        let mut onset_env = Vec::with_capacity(total_frames);
        let window = hann_window(win);

        let mut prev_mag = vec![0.0_f32; win / 2 + 1];

        for frame_idx in 0..total_frames {
            let start = frame_idx * hop;
            let end = (start + win).min(signal.len());
            if end - start < win {
                break;
            }

            // Windowed frame
            let windowed: Vec<oxifft::Complex<f32>> = signal[start..end]
                .iter()
                .zip(window.iter())
                .map(|(&s, &w)| oxifft::Complex::new(s * w, 0.0))
                .collect();

            let spectrum = oxifft::fft(&windowed);
            let mag = magnitude_spectrum(&spectrum);
            let n_bins = mag.len().min(win / 2 + 1);

            // RMS energy
            let frame_energy: f32 =
                signal[start..end].iter().map(|&s| s * s).sum::<f32>() / (end - start) as f32;
            energy_env.push(frame_energy.sqrt());

            // Spectral brightness: weighted centroid normalised to Nyquist
            let freq_per_bin = sr / win as f32;
            let (weighted, total): (f32, f32) = mag[..n_bins]
                .iter()
                .enumerate()
                .fold((0.0, 0.0), |(ws, tm), (k, &m)| {
                    (ws + k as f32 * freq_per_bin * m, tm + m)
                });
            let centroid = if total > 1e-9 {
                weighted / total / (sr * 0.5)
            } else {
                0.0
            };
            brightness_env.push(centroid.clamp(0.0, 1.0));

            // Positive spectral flux (onset strength)
            let flux: f32 = mag[..n_bins]
                .iter()
                .zip(prev_mag.iter())
                .map(|(&m, &p)| (m - p).max(0.0))
                .sum();
            onset_env.push(flux / n_bins as f32);
            prev_mag[..n_bins].copy_from_slice(&mag[..n_bins]);
        }

        let n_frames = energy_env.len();
        if n_frames == 0 {
            return Err(crate::MirError::AnalysisFailed(
                "No frames computed for thumbnailing".to_string(),
            ));
        }

        // Normalise feature envelopes to [0, 1]
        let energy_norm = normalise_vec(&energy_env);
        let bright_norm = normalise_vec(&brightness_env);
        let onset_norm = normalise_vec(&onset_env);

        // ── Score sliding windows ─────────────────────────────────────────
        let target_dur = self.config.target_duration_secs;
        let max_dur = self.config.max_duration_secs;
        let thumb_frames = (target_dur * sr / hop as f32) as usize;

        let min_thumb = ((self.config.min_duration_secs * sr) as usize / hop).max(1);
        let max_thumb = ((max_dur * sr) as usize / hop).min(n_frames);
        let thumb_frames = thumb_frames.clamp(min_thumb, max_thumb);

        if thumb_frames >= n_frames {
            // Entire signal fits within thumbnail window
            let end_sample = signal.len();
            let dur = signal.len() as f32 / sr;
            return Ok(ThumbnailResult {
                start_sample: 0,
                end_sample,
                start_secs: 0.0,
                end_secs: dur,
                duration_secs: dur,
                score: 1.0,
            });
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_start_frame = 0_usize;

        let ew = self.config.energy_weight;
        let bw = self.config.brightness_weight;
        let ow = self.config.onset_weight;
        let cw = self.config.centrality_weight;

        for start_frame in 0..=(n_frames - thumb_frames) {
            let end_frame = start_frame + thumb_frames;

            // Window-sum using pre-computed prefix sums would be O(1), but for
            // typical track lengths (< 20k frames) a direct sum is fast enough.
            let avg_energy: f32 =
                energy_norm[start_frame..end_frame].iter().sum::<f32>() / thumb_frames as f32;
            let avg_bright: f32 =
                bright_norm[start_frame..end_frame].iter().sum::<f32>() / thumb_frames as f32;
            let avg_onset: f32 =
                onset_norm[start_frame..end_frame].iter().sum::<f32>() / thumb_frames as f32;

            // Centrality score: penalise the last 25% of the track
            let frac = start_frame as f32 / n_frames as f32;
            let centrality = if frac > 0.75 {
                1.0 - (frac - 0.75) / 0.25
            } else {
                1.0
            };

            let score = ew * avg_energy + bw * avg_bright + ow * avg_onset + cw * centrality;
            if score > best_score {
                best_score = score;
                best_start_frame = start_frame;
            }
        }

        let start_sample = best_start_frame * hop;
        let end_sample = (best_start_frame + thumb_frames) * hop + win;
        let end_sample = end_sample.min(signal.len());

        let start_secs = start_sample as f32 / sr;
        let end_secs = end_sample as f32 / sr;

        Ok(ThumbnailResult {
            start_sample,
            end_sample,
            start_secs,
            end_secs,
            duration_secs: end_secs - start_secs,
            score: best_score,
        })
    }
}

/// Normalise a slice to [0, 1]; returns a vec of 0.5 if range is zero.
fn normalise_vec(v: &[f32]) -> Vec<f32> {
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    if range < 1e-9 {
        return vec![0.5; v.len()];
    }
    v.iter().map(|&x| (x - min) / range).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn make_signal(seconds: f32, sr: f32) -> Vec<f32> {
        let n = (sr * seconds) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                (TAU * 440.0 * t).sin() * 0.5 + (TAU * 880.0 * t).sin() * 0.3
            })
            .collect()
    }

    #[test]
    fn test_thumbnail_config_default() {
        let cfg = ThumbnailConfig::default();
        assert!((cfg.target_duration_secs - 20.0).abs() < f32::EPSILON);
        assert!(cfg.energy_weight > 0.0);
    }

    #[test]
    fn test_thumbnail_too_short_returns_error() {
        let thumbnail = AudioThumbnail::with_sample_rate(44100.0);
        let short_signal = make_signal(5.0, 44100.0);
        let result = thumbnail.extract(&short_signal);
        assert!(result.is_err(), "Should fail for short signal");
    }

    #[test]
    fn test_thumbnail_full_track_fits_in_window() {
        // When the track is shorter than max_duration the entire signal is used
        let thumbnail = AudioThumbnail::with_sample_rate(44100.0);
        // 20s is exactly at target_duration — should succeed and cover it
        let signal = make_signal(20.0, 44100.0);
        let result = thumbnail.extract(&signal);
        assert!(result.is_ok(), "20s signal should succeed");
    }

    #[test]
    fn test_thumbnail_returns_valid_range() {
        let thumbnail = AudioThumbnail::with_sample_rate(44100.0);
        let signal = make_signal(60.0, 44100.0);
        let result = thumbnail.extract(&signal).expect("should succeed");
        assert!(result.start_sample < result.end_sample);
        assert!(result.end_sample <= signal.len());
        assert!(result.duration_secs > 0.0);
        assert!(result.score.is_finite());
    }

    #[test]
    fn test_thumbnail_start_before_end() {
        let thumbnail = AudioThumbnail::with_sample_rate(22050.0);
        let signal = make_signal(45.0, 22050.0);
        let result = thumbnail.extract(&signal).expect("should succeed");
        assert!(result.start_secs < result.end_secs);
    }

    #[test]
    fn test_thumbnail_clip_duration_within_bounds() {
        let cfg = ThumbnailConfig::default();
        let min_d = cfg.min_duration_secs;
        let max_d = cfg.max_duration_secs;
        let thumbnail = AudioThumbnail::new(44100.0, cfg);
        let signal = make_signal(120.0, 44100.0);
        let result = thumbnail.extract(&signal).expect("should succeed");
        // Duration should be within [min, max + 1 extra window tolerance]
        assert!(
            result.duration_secs >= min_d - 1.0,
            "Duration {:.2}s too short",
            result.duration_secs
        );
        assert!(
            result.duration_secs <= max_d + 2.0,
            "Duration {:.2}s too long",
            result.duration_secs
        );
    }

    #[test]
    fn test_normalise_vec_constant_input() {
        let v = vec![3.0_f32; 10];
        let n = normalise_vec(&v);
        assert!(n.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn test_normalise_vec_range() {
        let v = vec![0.0_f32, 1.0, 0.5];
        let n = normalise_vec(&v);
        assert!((n[0] - 0.0).abs() < 1e-6);
        assert!((n[1] - 1.0).abs() < 1e-6);
        assert!((n[2] - 0.5).abs() < 1e-6);
    }
}
