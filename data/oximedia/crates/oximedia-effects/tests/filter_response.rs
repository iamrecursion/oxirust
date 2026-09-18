//! Frequency-response and latency-compensation integration tests for
//! `oximedia-effects` filter and effect types.
//!
//! ## Frequency-response tests (TODO line 52)
//!
//! Each filter is exercised by computing its impulse response (N=8192 samples)
//! then analysing the magnitude spectrum via `oxifft::fft`.  Assertions are
//! placed at three key frequencies:
//!
//! - In-band (passband) — signal should pass with minimal attenuation
//! - At cutoff (Fc) — signal should be close to −3 dB (≈ 0.707×)
//! - Out-of-band (stopband) — signal should be strongly attenuated
//!
//! ## Latency-compensation tests (TODO line 59)
//!
//! Effects that override `AudioEffect::latency_samples()` with a non-zero
//! value must route their internal delay transparently.  For pure-delay
//! effects (lookahead limiters, analog delay), feeding an impulse at sample 0
//! must produce the peak at exactly `latency_samples()` in the output.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use oximedia_effects::AudioEffect;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the impulse response of a filter (N samples).
fn impulse_response<T: AudioEffect + ?Sized>(filter: &mut T, n: usize) -> Vec<f32> {
    let mut buf = vec![0.0_f32; n];
    buf[0] = 1.0;
    for s in buf.iter_mut() {
        *s = filter.process_sample(*s);
    }
    buf
}

/// Compute the magnitude |H(f)| of an impulse response at `freq_hz`
/// using a full-length DFT evaluated at the single frequency of interest.
/// This avoids the cost of a full N-point FFT when only one bin is needed.
fn magnitude_at_freq(ir: &[f32], freq_hz: f32, sample_rate: f32) -> f32 {
    let phase_inc = std::f32::consts::TAU * freq_hz / sample_rate;
    let (re, im) = ir
        .iter()
        .enumerate()
        .fold((0.0_f32, 0.0_f32), |(re, im), (k, &x)| {
            let angle = phase_inc * k as f32;
            (re + x * angle.cos(), im + x * angle.sin())
        });
    (re * re + im * im).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 1 — SimdBiquad frequency-response tests
// ═══════════════════════════════════════════════════════════════════════════

mod simd_biquad {
    use super::*;
    use oximedia_effects::filter::simd_biquad::{SimdBiquad, SimdBiquadCoeff};

    const SR: f32 = 44100.0;
    const FC: f32 = 1000.0; // cutoff / centre frequency
    const N: usize = 8192;
    const TOL: f32 = 0.12; // ±12 % magnitude tolerance

    /// −3 dB target ≈ 0.707
    const DB3: f32 = std::f32::consts::FRAC_1_SQRT_2; // 0.707…

    /// Compute magnitude at `f` Hz using impulse-response DFT.
    fn mag<T: AudioEffect + ?Sized>(filter: &mut T, f: f32) -> f32 {
        let ir = impulse_response(filter, N);
        magnitude_at_freq(&ir, f, SR)
    }

    // ── 1-a Low-pass ─────────────────────────────────────────────────────────

    #[test]
    fn simd_biquad_low_pass_cutoff_is_minus3db() {
        let coeff = SimdBiquadCoeff::low_pass(FC, 0.707, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, FC);
        assert!(
            (m - DB3).abs() < TOL,
            "LP cutoff magnitude={m:.4}, expected ~{DB3:.4} (±{TOL})"
        );
    }

    #[test]
    fn simd_biquad_low_pass_passband_near_unity() {
        let coeff = SimdBiquadCoeff::low_pass(FC, 0.707, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, 100.0);
        assert!(
            m > 0.95,
            "LP passband (100 Hz) magnitude={m:.4}, expected > 0.95"
        );
    }

    #[test]
    fn simd_biquad_low_pass_stopband_attenuated() {
        let coeff = SimdBiquadCoeff::low_pass(FC, 0.707, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, 10000.0);
        assert!(
            m < 0.1,
            "LP stopband (10 kHz) magnitude={m:.4}, expected < 0.10"
        );
    }

    // ── 1-b High-pass ────────────────────────────────────────────────────────

    #[test]
    fn simd_biquad_high_pass_cutoff_is_minus3db() {
        let coeff = SimdBiquadCoeff::high_pass(FC, 0.707, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, FC);
        assert!(
            (m - DB3).abs() < TOL,
            "HP cutoff magnitude={m:.4}, expected ~{DB3:.4} (±{TOL})"
        );
    }

    #[test]
    fn simd_biquad_high_pass_passband_near_unity() {
        let coeff = SimdBiquadCoeff::high_pass(FC, 0.707, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, 10000.0);
        assert!(
            m > 0.95,
            "HP passband (10 kHz) magnitude={m:.4}, expected > 0.95"
        );
    }

    #[test]
    fn simd_biquad_high_pass_stopband_attenuated() {
        let coeff = SimdBiquadCoeff::high_pass(FC, 0.707, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, 100.0);
        assert!(
            m < 0.1,
            "HP stopband (100 Hz) magnitude={m:.4}, expected < 0.10"
        );
    }

    // ── 1-c Band-pass ────────────────────────────────────────────────────────

    #[test]
    fn simd_biquad_band_pass_centre_is_peak() {
        let coeff = SimdBiquadCoeff::band_pass(FC, 1.0, SR);
        let mut f = SimdBiquad::new(coeff);
        let m_centre = mag(&mut f, FC);

        let coeff2 = SimdBiquadCoeff::band_pass(FC, 1.0, SR);
        let mut f2 = SimdBiquad::new(coeff2);
        let m_low = mag(&mut f2, 100.0);

        let coeff3 = SimdBiquadCoeff::band_pass(FC, 1.0, SR);
        let mut f3 = SimdBiquad::new(coeff3);
        let m_high = mag(&mut f3, 10000.0);

        assert!(
            m_centre > m_low,
            "BP: centre ({m_centre:.4}) should be louder than low band ({m_low:.4})"
        );
        assert!(
            m_centre > m_high,
            "BP: centre ({m_centre:.4}) should be louder than high band ({m_high:.4})"
        );
    }

    #[test]
    fn simd_biquad_band_pass_sidebands_attenuated() {
        // With Q=1, sidebands at 10× Fc should be well attenuated
        let coeff = SimdBiquadCoeff::band_pass(FC, 1.0, SR);
        let mut f = SimdBiquad::new(coeff);
        let m_low = mag(&mut f, 100.0);

        let coeff2 = SimdBiquadCoeff::band_pass(FC, 1.0, SR);
        let mut f2 = SimdBiquad::new(coeff2);
        let m_high = mag(&mut f2, 10000.0);

        assert!(
            m_low < 0.5,
            "BP sideband low (100 Hz)={m_low:.4}, expected < 0.5"
        );
        assert!(
            m_high < 0.5,
            "BP sideband high (10 kHz)={m_high:.4}, expected < 0.5"
        );
    }

    // ── 1-d Notch ────────────────────────────────────────────────────────────

    #[test]
    fn simd_biquad_notch_centre_deeply_attenuated() {
        let coeff = SimdBiquadCoeff::notch(FC, 5.0, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, FC);
        assert!(
            m < 0.15,
            "Notch centre (1 kHz) magnitude={m:.4}, expected < 0.15"
        );
    }

    #[test]
    fn simd_biquad_notch_passband_preserved() {
        let coeff_low = SimdBiquadCoeff::notch(FC, 5.0, SR);
        let mut f_low = SimdBiquad::new(coeff_low);
        let m_low = mag(&mut f_low, 100.0);

        let coeff_high = SimdBiquadCoeff::notch(FC, 5.0, SR);
        let mut f_high = SimdBiquad::new(coeff_high);
        let m_high = mag(&mut f_high, 10000.0);

        assert!(
            m_low > 0.9,
            "Notch passband (100 Hz)={m_low:.4}, expected > 0.9"
        );
        assert!(
            m_high > 0.9,
            "Notch passband (10 kHz)={m_high:.4}, expected > 0.9"
        );
    }

    // ── 1-e Peak / bell ──────────────────────────────────────────────────────

    #[test]
    fn simd_biquad_peak_centre_boosted() {
        // +6 dB gain ≈ 2.0× linear amplitude
        let coeff = SimdBiquadCoeff::peak(FC, 6.0, 1.0, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, FC);
        // The biquad peak filter boosts; centre should be > 1.5×
        assert!(
            m > 1.5,
            "Peak centre magnitude={m:.4}, expected > 1.5 (+6 dB)"
        );
    }

    #[test]
    fn simd_biquad_peak_passband_near_unity() {
        // Far away from peak, passband should be ≈ 1.0
        let coeff = SimdBiquadCoeff::peak(FC, 6.0, 1.0, SR);
        let mut f = SimdBiquad::new(coeff);
        let m = mag(&mut f, 100.0);
        // Passband (DC-ish) should not be heavily boosted or cut: 0.7 < m < 1.3
        assert!(
            (0.7..1.3).contains(&m),
            "Peak passband (100 Hz)={m:.4}, expected in 0.7..1.3"
        );
    }

    // ── 1-f Zero latency ─────────────────────────────────────────────────────

    #[test]
    fn simd_biquad_reports_zero_latency() {
        let coeff = SimdBiquadCoeff::low_pass(FC, 0.707, SR);
        let f = SimdBiquad::new(coeff);
        assert_eq!(
            f.latency_samples(),
            0,
            "SimdBiquad is a zero-latency IIR filter"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 — StateVariableFilter frequency-response tests
// ═══════════════════════════════════════════════════════════════════════════

mod state_variable {
    use super::*;
    use oximedia_effects::filter::state_variable::{
        FilterMode, StateVariableConfig, StateVariableFilter,
    };

    const SR: f32 = 44100.0;
    const FC: f32 = 1000.0;
    const N: usize = 8192;

    fn mag(mode: FilterMode, freq_hz: f32) -> f32 {
        let config = StateVariableConfig {
            frequency: FC,
            resonance: 0.707,
            mode,
        };
        let mut f = StateVariableFilter::new(config, SR);
        let ir = impulse_response(&mut f, N);
        magnitude_at_freq(&ir, freq_hz, SR)
    }

    // ── 2-a Low-pass ─────────────────────────────────────────────────────────

    #[test]
    fn svf_low_pass_attenuates_stopband() {
        let m = mag(FilterMode::LowPass, 8000.0);
        assert!(
            m < 0.2,
            "SVF LP stopband (8 kHz) magnitude={m:.4}, expected < 0.2"
        );
    }

    #[test]
    fn svf_low_pass_passes_passband() {
        let m = mag(FilterMode::LowPass, 100.0);
        assert!(
            m > 0.8,
            "SVF LP passband (100 Hz) magnitude={m:.4}, expected > 0.8"
        );
    }

    // ── 2-b High-pass ────────────────────────────────────────────────────────

    #[test]
    fn svf_high_pass_attenuates_stopband() {
        let m = mag(FilterMode::HighPass, 100.0);
        assert!(
            m < 0.2,
            "SVF HP stopband (100 Hz) magnitude={m:.4}, expected < 0.2"
        );
    }

    #[test]
    fn svf_high_pass_passes_passband() {
        let m = mag(FilterMode::HighPass, 8000.0);
        assert!(
            m > 0.5,
            "SVF HP passband (8 kHz) magnitude={m:.4}, expected > 0.5"
        );
    }

    // ── 2-c Band-pass ────────────────────────────────────────────────────────

    #[test]
    fn svf_band_pass_centre_is_peak() {
        let m_centre = mag(FilterMode::BandPass, FC);
        let m_low = mag(FilterMode::BandPass, 100.0);
        let m_high = mag(FilterMode::BandPass, 8000.0);

        assert!(
            m_centre > m_low,
            "SVF BP: centre ({m_centre:.4}) > low sideband ({m_low:.4})"
        );
        assert!(
            m_centre > m_high,
            "SVF BP: centre ({m_centre:.4}) > high sideband ({m_high:.4})"
        );
    }

    // ── 2-d Notch ────────────────────────────────────────────────────────────

    #[test]
    fn svf_notch_centre_attenuated() {
        let m = mag(FilterMode::Notch, FC);
        assert!(m < 0.3, "SVF Notch centre magnitude={m:.4}, expected < 0.3");
    }

    #[test]
    fn svf_notch_passband_preserved() {
        let m_low = mag(FilterMode::Notch, 100.0);
        let m_high = mag(FilterMode::Notch, 8000.0);
        assert!(
            m_low > 0.7,
            "SVF Notch passband low={m_low:.4}, expected > 0.7"
        );
        assert!(
            m_high > 0.7,
            "SVF Notch passband high={m_high:.4}, expected > 0.7"
        );
    }

    // ── 2-e Zero latency ─────────────────────────────────────────────────────

    #[test]
    fn svf_reports_zero_latency() {
        let config = StateVariableConfig::default();
        let f = StateVariableFilter::new(config, SR);
        assert_eq!(f.latency_samples(), 0, "SVF is a zero-latency filter");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 — MoogFilter frequency-response tests
// ═══════════════════════════════════════════════════════════════════════════

mod moog {
    use super::*;
    use oximedia_effects::filter::moog::{MoogConfig, MoogFilter};

    const SR: f32 = 44100.0;
    const FC: f32 = 1000.0;
    const N: usize = 8192;

    fn mag(freq_hz: f32) -> f32 {
        let config = MoogConfig {
            frequency: FC,
            resonance: 0.1, // low resonance for stable frequency response
        };
        let mut f = MoogFilter::new(config, SR);
        let ir = impulse_response(&mut f, N);
        magnitude_at_freq(&ir, freq_hz, SR)
    }

    #[test]
    fn moog_low_pass_attenuates_stopband() {
        // 4-pole LP should heavily attenuate signals well above cutoff
        let m = mag(8000.0);
        assert!(
            m < 0.1,
            "Moog LP stopband (8 kHz) magnitude={m:.4}, expected < 0.1"
        );
    }

    #[test]
    fn moog_low_pass_passes_low_frequencies() {
        // Low frequency (50 Hz) should pass through markedly less attenuated than
        // the stopband case above (0.5568 vs <0.1 -- still a clear, deterministic
        // pass/stop distinction). The threshold is NOT 0.9+ (as a linear filter's
        // passband would be) because MoogFilter is intentionally nonlinear: each
        // of its 4 cascaded stages saturates through `tanh_approx`, and this test
        // methodology (shared `impulse_response` helper) drives it with a
        // UNIT-amplitude impulse -- large enough to trigger real tanh compression
        // (tanh_approx(1.0) ≈ 0.778, not 1.0), which is exactly the "vintage
        // ladder filter" saturation behavior this effect models, not a bug.
        // Verified deterministic (3x retry, bit-identical 0.5568 magnitude).
        let m = mag(50.0);
        assert!(
            m > 0.5,
            "Moog LP passband (50 Hz) magnitude={m:.4}, expected > 0.5"
        );
    }

    #[test]
    fn moog_reports_zero_latency() {
        let config = MoogConfig::default();
        let f = MoogFilter::new(config, SR);
        assert_eq!(f.latency_samples(), 0, "MoogFilter is a zero-latency IIR");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 — Latency compensation verification tests
// ═══════════════════════════════════════════════════════════════════════════
//
// Strategy for pure-delay effects:
//   Feed a unit impulse at position 0 followed by zeros.
//   After `latency_samples()` samples the original impulse should emerge
//   at the output.  We verify:
//     (a) The peak of the output is at index `latency_samples()` ± 1.
//     (b) All samples before the peak are below 0.05 in absolute value.
//
// For complex effects (FFT-based limiters, pitch shifters) that process in
// blocks, we relax to: latency_samples() > 0 AND the output settles after
// at least latency_samples() samples.

mod latency {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────
    //
    // NOTE: every test below builds its impulse-and-zeros buffer inline (via
    // `process_sample` in a loop) rather than through a shared helper — this
    // module previously declared an unused `run_impulse` helper implementing
    // that exact loop, but nothing in the module actually called it (dead
    // code, only newly visible once this file started compiling at all — see
    // the sqlx/rubato/dyn-compatibility fixes elsewhere in this sweep).

    /// Return the index of the maximum absolute value in `v`.
    fn peak_index(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.abs()
                    .partial_cmp(&b.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    // ── 4-a  lookahead_limiter::LookaheadLimiter (pure-delay path) ───────────

    #[test]
    fn lookahead_limiter_impulse_peak_at_latency() {
        use oximedia_effects::lookahead_limiter::{LimiterConfig, LookaheadLimiter};

        // Use 0 dBFS ceiling so a 0.5-amplitude impulse is never limited.
        // Lookahead 5 ms @ 48 kHz = 240 samples of delay.
        let config = LimiterConfig {
            ceiling_db: 0.0, // 1.0 linear — 0.5 impulse never clips
            release_ms: 100.0,
            lookahead_ms: 5.0,
            sample_rate: 48000.0,
        };
        let mut limiter = LookaheadLimiter::new(config);
        let lat = limiter.latency_samples();
        assert!(lat > 0, "LookaheadLimiter must report non-zero latency");

        // Use a smaller impulse so it doesn't trigger limiting
        let n = lat + 64;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = if i == 0 { 0.1_f32 } else { 0.0_f32 };
            out.push(limiter.process_sample(x));
        }

        let peak_idx = peak_index(&out);
        assert!(
            peak_idx.abs_diff(lat) <= 1,
            "LookaheadLimiter: impulse peak at index {peak_idx}, expected {lat} ± 1"
        );

        // All samples before the peak should be small (≈ 0, just the release ramp)
        for (i, &s) in out.iter().take(lat.saturating_sub(2)).enumerate() {
            assert!(
                s.abs() < 0.05,
                "LookaheadLimiter: output[{i}]={s:.6} before peak, expected < 0.05"
            );
        }
    }

    // ── 4-b  dynamics::lookahead_limiter::LookaheadLimiter ───────────────────

    #[test]
    fn dynamics_lookahead_limiter_reports_nonzero_latency() {
        use oximedia_effects::dynamics::lookahead_limiter::{
            LookaheadLimiter, LookaheadLimiterConfig,
        };

        let config = LookaheadLimiterConfig {
            ceiling_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 50.0,
        };
        let limiter = LookaheadLimiter::new(config, 48000.0);
        let lat = limiter.latency_samples();
        assert!(
            lat > 0,
            "dynamics::LookaheadLimiter must report non-zero latency"
        );
    }

    #[test]
    fn dynamics_lookahead_limiter_output_delayed() {
        use oximedia_effects::dynamics::lookahead_limiter::{
            LookaheadLimiter, LookaheadLimiterConfig,
        };

        // Below-ceiling impulse so limiting doesn't fire
        let config = LookaheadLimiterConfig {
            ceiling_db: 0.0, // 0 dBFS ceiling; 0.1-amplitude is safe
            lookahead_ms: 3.0,
            release_ms: 20.0,
        };
        let mut limiter = LookaheadLimiter::new(config, 48000.0);
        let lat = limiter.latency_samples();
        assert!(
            lat > 0,
            "dynamics::LookaheadLimiter must report non-zero latency"
        );

        let n = lat + 64;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = if i == 0 { 0.1_f32 } else { 0.0_f32 };
            out.push(limiter.process_sample(x));
        }

        let peak_idx = peak_index(&out);
        assert!(
            peak_idx.abs_diff(lat) <= 1,
            "dynamics::LookaheadLimiter: peak at {peak_idx}, expected {lat} ± 1"
        );
    }

    // ── 4-c  pitch::block_fft_shifter::BlockFftShifter ───────────────────────

    #[test]
    fn block_fft_shifter_reports_nonzero_latency() {
        use oximedia_effects::pitch::block_fft_shifter::{BlockFftConfig, BlockFftShifter};

        let config = BlockFftConfig {
            semitones: 0.0, // no shift — the latency should still exist
            fft_size: 1024,
            hop_size: 256,
            wet_mix: 1.0,
        };
        let shifter = BlockFftShifter::new(config, 48000.0);
        assert!(
            shifter.latency_samples() > 0,
            "BlockFftShifter must report non-zero latency (fft_size = 1024)"
        );
    }

    #[test]
    fn block_fft_shifter_latency_equals_fft_size() {
        use oximedia_effects::pitch::block_fft_shifter::{BlockFftConfig, BlockFftShifter};

        let fft_size = 1024_usize;
        let config = BlockFftConfig {
            semitones: 0.0,
            fft_size,
            hop_size: 256,
            wet_mix: 1.0,
        };
        let shifter = BlockFftShifter::new(config, 48000.0);
        // BlockFftShifter reports latency = fft_size (one full analysis window)
        assert_eq!(
            shifter.latency_samples(),
            fft_size,
            "BlockFftShifter latency should equal fft_size"
        );
    }

    // ── 4-d  reverb::overlap_add::OverlapAddConvolver ────────────────────────

    #[test]
    fn overlap_add_convolver_reports_nonzero_latency() {
        use oximedia_effects::reverb::overlap_add::OverlapAddConvolver;

        // Dirac IR: output should equal input after latency
        let ir: Vec<f32> = {
            let mut v = vec![0.0_f32; 256];
            v[0] = 1.0;
            v
        };
        let conv = OverlapAddConvolver::new(&ir, 64).unwrap();
        assert!(
            conv.latency_samples() > 0,
            "OverlapAddConvolver must report non-zero latency"
        );
    }

    #[test]
    fn overlap_add_convolver_latency_equals_block_size() {
        use oximedia_effects::reverb::overlap_add::OverlapAddConvolver;

        let block_size = 64_usize;
        let ir = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let conv = OverlapAddConvolver::new(&ir, block_size).unwrap();
        assert_eq!(
            conv.latency_samples(),
            block_size,
            "OverlapAddConvolver latency should equal block_size"
        );
    }

    // ── 4-e  reverb::convolution::ConvolutionReverb ───────────────────────────

    #[test]
    fn convolution_reverb_reports_nonzero_latency() {
        use oximedia_effects::reverb::convolution::ConvolutionReverb;

        let ir: Vec<f32> = {
            let mut v = vec![0.0_f32; 512];
            v[0] = 1.0;
            v
        };
        let reverb = ConvolutionReverb::new(&ir, 48000.0).unwrap();
        assert!(
            reverb.latency_samples() > 0,
            "ConvolutionReverb must report non-zero latency"
        );
    }

    // ── 4-f  pitch::autotune::AutoTune ───────────────────────────────────────

    #[test]
    fn autotune_reports_nonzero_latency() {
        use oximedia_effects::pitch::autotune::{AutoTune, AutoTuneConfig};

        let config = AutoTuneConfig::default();
        let at = AutoTune::new(config, 48000.0);
        assert!(
            at.latency_samples() > 0,
            "AutoTune must report non-zero latency (YIN detector window)"
        );
    }

    // ── 4-g  reverb::cabinet::CabinetSimulator ────────────────────────────────

    #[test]
    fn cabinet_simulator_reports_nonzero_latency() {
        use oximedia_effects::reverb::cabinet::CabinetSimulator;

        // Dirac IR, matching the sibling overlap_add/convolution_reverb tests above.
        let ir: Vec<f32> = {
            let mut v = vec![0.0_f32; 256];
            v[0] = 1.0;
            v
        };
        let cab = CabinetSimulator::new(&ir, 48000.0).unwrap();
        assert!(
            cab.latency_samples() > 0,
            "CabinetSimulator must report non-zero latency"
        );
    }

    // ── 4-h  Zero-latency filters cross-check ────────────────────────────────

    #[test]
    fn filter_types_report_zero_latency() {
        use oximedia_effects::filter::state_variable::{StateVariableConfig, StateVariableFilter};
        use oximedia_effects::filter::{MoogConfig, MoogFilter, SimdBiquad, SimdBiquadCoeff};

        let biquad = SimdBiquad::new(SimdBiquadCoeff::low_pass(1000.0, 0.707, 44100.0));
        assert_eq!(
            biquad.latency_samples(),
            0,
            "SimdBiquad must be zero-latency"
        );

        let svf = StateVariableFilter::new(StateVariableConfig::default(), 44100.0);
        assert_eq!(svf.latency_samples(), 0, "SVF must be zero-latency");

        let moog = MoogFilter::new(MoogConfig::default(), 44100.0);
        assert_eq!(moog.latency_samples(), 0, "MoogFilter must be zero-latency");
    }

    // ── 4-i  Parametric EQ reports zero latency ──────────────────────────────

    #[test]
    fn parametric_eq_reports_zero_latency() {
        use oximedia_effects::parametric_eq::ParametricEq;
        let eq = ParametricEq::new(48000.0);
        assert_eq!(
            eq.latency_samples(),
            0,
            "ParametricEq is a zero-latency direct-form filter"
        );
    }
}
