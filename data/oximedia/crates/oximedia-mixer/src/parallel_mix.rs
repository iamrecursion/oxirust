//! Parallel channel mixing using rayon for high channel counts.
//!
//! When the channel count exceeds `ParallelMixConfig::min_channels_for_parallel`,
//! the per-channel DSP (input gain → phase inversion → fader → pan) is performed
//! concurrently across rayon threads.  The final summing into the master bus and
//! soft-clipping remain sequential because they operate on shared accumulators.
//!
//! This module intentionally implements only the *channel strip* part of the signal
//! chain (input gain, phase, fader, pan) in parallel, omitting aux sends, PDC, bus
//! effects, and VCA group processing.  Those subsystems require mutable access to
//! shared state and are therefore handled sequentially in the existing
//! `ProcessingEngine::process_mix`.  `AudioMixer::process_parallel` is primarily
//! intended as a fast path for summing dry channel outputs when the session has
//! many simple channels.

use rayon::prelude::*;

use crate::processing::{ChannelProcessParams, PanLawType};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the parallel mixing path.
#[derive(Debug, Clone)]
pub struct ParallelMixConfig {
    /// Minimum number of channels required before switching to the parallel path.
    ///
    /// When `channels.len() <= min_channels_for_parallel` the sequential path is
    /// used instead.  Default: 8.
    pub min_channels_for_parallel: usize,
    /// Rayon thread pool size.  `0` means "use the global Rayon pool" (the
    /// default), which typically equals the number of logical CPU cores.
    pub threads: usize,
}

impl Default for ParallelMixConfig {
    fn default() -> Self {
        Self {
            min_channels_for_parallel: 8,
            threads: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-channel parallel DSP helper
// ---------------------------------------------------------------------------

/// Data describing one channel's contribution to be computed in parallel.
#[derive(Clone)]
pub struct ParallelChannelInput {
    /// DSP parameters for this channel.
    pub params: ChannelProcessParams,
    /// Input samples (mono working buffer, length == block_size).
    pub samples: Vec<f32>,
}

/// Output of per-channel parallel DSP: left and right channel samples.
pub struct ParallelChannelOutput {
    /// Left channel samples (length == block_size).
    pub left: Vec<f32>,
    /// Right channel samples (length == block_size).
    pub right: Vec<f32>,
}

/// Process a single channel strip (input gain → phase → fader → pan) without
/// any mutable shared state.  Returns the panned stereo contribution.
pub fn process_channel_strip(
    input: &ParallelChannelInput,
    block_size: usize,
) -> ParallelChannelOutput {
    if input.params.muted {
        return ParallelChannelOutput {
            left: vec![0.0; block_size],
            right: vec![0.0; block_size],
        };
    }

    let input_gain = db_to_linear(input.params.input_gain_db);
    let phase_mult: f32 = if input.params.phase_inverted {
        -1.0
    } else {
        1.0
    };

    let mut working: Vec<f32> = input
        .samples
        .iter()
        .take(block_size)
        .map(|&s| s * input_gain * phase_mult)
        .collect();
    working.resize(block_size, 0.0);

    // Fader gain (no VCA in the parallel path — VCA requires shared state).
    let fader = input.params.fader_gain;
    for s in &mut working {
        *s *= fader;
    }

    // Stereo pan.
    let (left_gain, right_gain) = compute_stereo_pan(input.params.pan, input.params.pan_law);
    let mut left = vec![0.0_f32; block_size];
    let mut right = vec![0.0_f32; block_size];
    for i in 0..block_size {
        left[i] = working[i] * left_gain;
        right[i] = working[i] * right_gain;
    }

    ParallelChannelOutput { left, right }
}

// ---------------------------------------------------------------------------
// Parallel mix function
// ---------------------------------------------------------------------------

/// Mix `channels` in parallel and return interleaved stereo `Vec<f32>`.
///
/// Uses `rayon::par_iter` when `channels.len() > config.min_channels_for_parallel`.
/// Falls back to sequential iteration for smaller sessions.
///
/// Soft clipping (tanh-based) is applied to the final master mix before
/// interleaving.
#[must_use]
pub fn mix_parallel(
    channels: &[ParallelChannelInput],
    block_size: usize,
    config: &ParallelMixConfig,
) -> Vec<f32> {
    // Choose parallel or sequential depending on channel count.
    let per_channel_outputs: Vec<ParallelChannelOutput> =
        if channels.len() > config.min_channels_for_parallel {
            channels
                .par_iter()
                .map(|ch| process_channel_strip(ch, block_size))
                .collect()
        } else {
            channels
                .iter()
                .map(|ch| process_channel_strip(ch, block_size))
                .collect()
        };

    // Sum all channel contributions into master left/right buffers.
    let mut master_left = vec![0.0_f32; block_size];
    let mut master_right = vec![0.0_f32; block_size];

    for ch_out in &per_channel_outputs {
        for i in 0..block_size {
            master_left[i] += ch_out.left[i];
            master_right[i] += ch_out.right[i];
        }
    }

    // Soft clip to prevent digital overs.
    for i in 0..block_size {
        master_left[i] = soft_clip(master_left[i]);
        master_right[i] = soft_clip(master_right[i]);
    }

    // Interleave L/R → output.
    let mut out = Vec::with_capacity(block_size * 2);
    for i in 0..block_size {
        out.push(master_left[i]);
        out.push(master_right[i]);
    }
    out
}

// ---------------------------------------------------------------------------
// Local DSP helpers
// ---------------------------------------------------------------------------

#[inline]
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[must_use]
fn compute_stereo_pan(pan: f32, law: PanLawType) -> (f32, f32) {
    let pan_norm = ((pan + 1.0) * 0.5).clamp(0.0, 1.0);
    match law {
        PanLawType::Linear | PanLawType::Minus6dB => (1.0 - pan_norm, pan_norm),
        PanLawType::Minus3dB => {
            let angle = pan_norm * std::f32::consts::FRAC_PI_2;
            (angle.cos(), angle.sin())
        }
        PanLawType::Minus4Dot5dB => {
            let linear_l = 1.0 - pan_norm;
            let linear_r = pan_norm;
            let angle = pan_norm * std::f32::consts::FRAC_PI_2;
            let power_l = angle.cos();
            let power_r = angle.sin();
            (
                0.5 * linear_l + 0.5 * power_l,
                0.5 * linear_r + 0.5 * power_r,
            )
        }
    }
}

#[inline]
fn soft_clip(x: f32) -> f32 {
    x.tanh()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(gain: f32, pan: f32, muted: bool, samples: Vec<f32>) -> ParallelChannelInput {
        ParallelChannelInput {
            params: ChannelProcessParams {
                fader_gain: gain,
                pan,
                muted,
                input_gain_db: 0.0,
                phase_inverted: false,
                pan_law: PanLawType::Linear,
            },
            samples,
        }
    }

    fn default_config() -> ParallelMixConfig {
        ParallelMixConfig::default()
    }

    // 1. Default config has correct values.
    #[test]
    fn test_parallel_mix_config_defaults() {
        let cfg = ParallelMixConfig::default();
        assert_eq!(cfg.min_channels_for_parallel, 8);
        assert_eq!(cfg.threads, 0);
    }

    // 2. Empty channel list → all-zero interleaved output.
    #[test]
    fn test_mix_parallel_no_channels() {
        let out = mix_parallel(&[], 4, &default_config());
        assert_eq!(out.len(), 8);
        for v in &out {
            assert!(v.abs() < f32::EPSILON, "expected silence, got {v}");
        }
    }

    // 3. Single muted channel produces silence.
    #[test]
    fn test_mix_parallel_single_muted_channel() {
        let ch = make_input(1.0, 0.0, true, vec![1.0; 4]);
        let out = mix_parallel(&[ch], 4, &default_config());
        assert_eq!(out.len(), 8);
        for v in &out {
            assert!(v.abs() < 1e-6, "expected silence, got {v}");
        }
    }

    // 4. Output length is block_size * 2.
    #[test]
    fn test_mix_parallel_output_length() {
        let ch = make_input(1.0, 0.0, false, vec![0.5; 64]);
        let out = mix_parallel(&[ch], 64, &default_config());
        assert_eq!(out.len(), 128);
    }

    // 5. Single center-panned channel: L and R equal.
    #[test]
    fn test_mix_parallel_center_pan_equal_lr() {
        let ch = make_input(1.0, 0.0, false, vec![1.0; 4]);
        let out = mix_parallel(&[ch], 4, &default_config());
        // With Linear pan law, pan=0 → pan_norm=0.5 → L=0.5, R=0.5
        for i in (0..8).step_by(2) {
            let diff = (out[i] - out[i + 1]).abs();
            assert!(diff < 1e-5, "L={} R={} diff={diff}", out[i], out[i + 1]);
        }
    }

    // 6. Hard-left pan puts signal only in left channel.
    #[test]
    fn test_mix_parallel_hard_left_pan() {
        let ch = make_input(1.0, -1.0, false, vec![1.0; 4]);
        let out = mix_parallel(&[ch], 4, &default_config());
        // pan=-1.0 → pan_norm=0.0 → L=1.0, R=0.0
        for i in (0..8).step_by(2) {
            assert!(out[i] > 0.0, "left should be nonzero");
            assert!(out[i + 1].abs() < 1e-6, "right should be zero");
        }
    }

    // 7. Hard-right pan puts signal only in right channel.
    #[test]
    fn test_mix_parallel_hard_right_pan() {
        let ch = make_input(1.0, 1.0, false, vec![1.0; 4]);
        let out = mix_parallel(&[ch], 4, &default_config());
        for i in (0..8).step_by(2) {
            assert!(out[i].abs() < 1e-6, "left should be zero");
            assert!(out[i + 1] > 0.0, "right should be nonzero");
        }
    }

    // 8. Fader gain of 0 produces silence.
    #[test]
    fn test_mix_parallel_zero_fader() {
        let ch = make_input(0.0, 0.0, false, vec![1.0; 4]);
        let out = mix_parallel(&[ch], 4, &default_config());
        for v in &out {
            assert!(v.abs() < 1e-6, "expected silence");
        }
    }

    // 9. Sequential path used when channels <= 8 (no panic).
    #[test]
    fn test_mix_parallel_sequential_path_8_channels() {
        let channels: Vec<_> = (0..8)
            .map(|_| make_input(0.5, 0.0, false, vec![0.1; 16]))
            .collect();
        let out = mix_parallel(&channels, 16, &default_config());
        assert_eq!(out.len(), 32);
    }

    // 10. Parallel path used when channels > 8 (no panic, produces output).
    #[test]
    fn test_mix_parallel_parallel_path_9_channels() {
        let channels: Vec<_> = (0..9)
            .map(|_| make_input(0.5, 0.0, false, vec![0.1; 16]))
            .collect();
        let out = mix_parallel(&channels, 16, &default_config());
        assert_eq!(out.len(), 32);
    }

    // 11. Sequential and parallel paths produce identical output.
    #[test]
    fn test_mix_parallel_sequential_equals_parallel() {
        // Force sequential: min_channels_for_parallel very high
        let seq_config = ParallelMixConfig {
            min_channels_for_parallel: 1000,
            threads: 0,
        };
        // Force parallel: min_channels_for_parallel = 0
        let par_config = ParallelMixConfig {
            min_channels_for_parallel: 0,
            threads: 0,
        };

        let channels: Vec<_> = (0..12)
            .map(|i| make_input(0.8, (i as f32 * 0.1) - 0.5, false, vec![0.3; 32]))
            .collect();

        let seq_out = mix_parallel(&channels, 32, &seq_config);
        let par_out = mix_parallel(&channels, 32, &par_config);

        assert_eq!(seq_out.len(), par_out.len());
        for (a, b) in seq_out.iter().zip(par_out.iter()) {
            assert!((a - b).abs() < 1e-5, "sequential={a} parallel={b} differ");
        }
    }

    // 12. Multiple channels sum correctly.
    #[test]
    fn test_mix_parallel_two_channels_sum() {
        let ch1 = make_input(1.0, -1.0, false, vec![0.2; 4]); // hard left
        let ch2 = make_input(1.0, 1.0, false, vec![0.3; 4]); // hard right
        let out = mix_parallel(&[ch1, ch2], 4, &default_config());
        // Left output comes from ch1 only, right from ch2 only (pre-clip).
        for i in (0..8).step_by(2) {
            assert!(out[i] > 0.0, "left mix should be nonzero");
            assert!(out[i + 1] > 0.0, "right mix should be nonzero");
        }
    }

    // 13. Soft clip prevents values > 1.0 in output.
    #[test]
    fn test_mix_parallel_soft_clip_applied() {
        // Many loud channels summing together should be clipped by tanh.
        let channels: Vec<_> = (0..20)
            .map(|_| make_input(1.0, 0.0, false, vec![10.0; 8]))
            .collect();
        let out = mix_parallel(&channels, 8, &default_config());
        for v in &out {
            assert!(v.abs() <= 1.0 + 1e-5, "soft clip failed: {v}");
        }
    }

    // 14. Phase inversion flips sign.
    #[test]
    fn test_mix_parallel_phase_inversion() {
        let ch_normal = ParallelChannelInput {
            params: ChannelProcessParams {
                fader_gain: 1.0,
                pan: -1.0, // hard left for clarity
                muted: false,
                input_gain_db: 0.0,
                phase_inverted: false,
                pan_law: PanLawType::Linear,
            },
            samples: vec![0.5; 4],
        };
        let ch_inverted = ParallelChannelInput {
            params: ChannelProcessParams {
                fader_gain: 1.0,
                pan: -1.0,
                muted: false,
                input_gain_db: 0.0,
                phase_inverted: true,
                pan_law: PanLawType::Linear,
            },
            samples: vec![0.5; 4],
        };
        let out_normal = mix_parallel(&[ch_normal], 4, &default_config());
        let out_inverted = mix_parallel(&[ch_inverted], 4, &default_config());
        // Before clipping both are small so tanh is nearly linear: signs should differ.
        for i in (0..8).step_by(2) {
            assert!(
                out_normal[i].signum() != out_inverted[i].signum()
                    || (out_normal[i].abs() < 1e-6 && out_inverted[i].abs() < 1e-6),
                "expected opposite signs: normal={} inverted={}",
                out_normal[i],
                out_inverted[i]
            );
        }
    }

    // 15. Input gain in dB scales the output.
    #[test]
    fn test_mix_parallel_input_gain_scales_output() {
        let ch_0db = ParallelChannelInput {
            params: ChannelProcessParams {
                fader_gain: 1.0,
                pan: -1.0,
                muted: false,
                input_gain_db: 0.0,
                phase_inverted: false,
                pan_law: PanLawType::Linear,
            },
            samples: vec![0.1; 4],
        };
        let ch_20db = ParallelChannelInput {
            params: ChannelProcessParams {
                fader_gain: 1.0,
                pan: -1.0,
                muted: false,
                input_gain_db: 20.0,
                phase_inverted: false,
                pan_law: PanLawType::Linear,
            },
            samples: vec![0.1; 4],
        };
        let out_0db = mix_parallel(&[ch_0db], 4, &default_config());
        let out_20db = mix_parallel(&[ch_20db], 4, &default_config());
        // 20 dB should be ~10× louder (before soft clip dominates).
        assert!(
            out_20db[0].abs() > out_0db[0].abs(),
            "20dB input gain should produce louder output: {}  vs {}",
            out_20db[0],
            out_0db[0]
        );
    }

    // 16. Block size 512 works without panic.
    #[test]
    fn test_mix_parallel_block_size_512() {
        let channels: Vec<_> = (0..10)
            .map(|_| make_input(0.5, 0.0, false, vec![0.1; 512]))
            .collect();
        let out = mix_parallel(&channels, 512, &default_config());
        assert_eq!(out.len(), 1024);
    }

    // 17. Custom min_channels_for_parallel respected.
    #[test]
    fn test_mix_parallel_custom_threshold() {
        let cfg = ParallelMixConfig {
            min_channels_for_parallel: 4,
            threads: 0,
        };
        // 5 channels → should use parallel path (no panic / correct length).
        let channels: Vec<_> = (0..5)
            .map(|_| make_input(0.5, 0.0, false, vec![0.2; 8]))
            .collect();
        let out = mix_parallel(&channels, 8, &cfg);
        assert_eq!(out.len(), 16);
    }
}
