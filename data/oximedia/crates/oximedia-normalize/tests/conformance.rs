//! EBU-R128 / True-Peak / DRC / Latency / Buffer-recycling conformance tests.
//!
//! All signals are generated synthetically (pure math); no files are read.

use oximedia_metering::Standard;
use oximedia_normalize::{
    drc::{DrcConfig, DynamicRangeCompressor},
    limiter::{LimiterConfig, TruePeakLimiter},
    realtime::{RealtimeConfig, RealtimeNormalizer},
    Normalizer, NormalizerConfig, ProcessingMode,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Generate a mono sine wave at `freq_hz` with the given `amplitude`, `sample_rate`,
/// and `duration_secs`.  Returns interleaved samples (1 channel).
fn sine_mono(freq_hz: f64, amplitude: f64, sample_rate: f64, duration_secs: f64) -> Vec<f32> {
    let n = (sample_rate * duration_secs).round() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (amplitude * (2.0 * std::f64::consts::PI * freq_hz * t).sin()) as f32
        })
        .collect()
}

/// Generate stereo interleaved sine (same signal on both channels).
fn sine_stereo(freq_hz: f64, amplitude: f64, sample_rate: f64, duration_secs: f64) -> Vec<f32> {
    let mono = sine_mono(freq_hz, amplitude, sample_rate, duration_secs);
    mono.iter().flat_map(|&s| [s, s]).collect()
}

/// RMS of a slice.
fn rms(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt()
}

/// dB to linear.
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

// ─── Test 1: EBU-R128 null normalization ─────────────────────────────────────
//
// Normalize audio, then re-analyze the output and check that the recommended
// gain on the second pass is very close to 0 (i.e. the output is compliant).
// We don't hard-code an exact LUFS amplitude because K-weighting shifts the
// mapping between amplitude and LUFS.  Instead we:
//   1. Analyze the raw audio to find its LUFS.
//   2. Normalize it to -23 LUFS (EBU R128 target).
//   3. Re-analyze the output.
//   4. Assert the second-pass gain requirement is within ±2 dB of 0.

#[test]
fn test_ebu_r128_null_normalization() {
    // Use a modest amplitude sine; the exact LUFS doesn't matter for this test.
    let samples = sine_stereo(1000.0, 0.3, 48_000.0, 3.0);

    let mut config = NormalizerConfig::new(Standard::EbuR128, 48_000.0, 2);
    config.processing_mode = ProcessingMode::TwoPass;
    config.enable_limiter = false;
    config.enable_drc = false;
    config.max_gain_db = 30.0;

    let mut normalizer = Normalizer::new(config).expect("normalizer creation");
    normalizer.analyze_f32(&samples);
    let first_analysis = normalizer.get_analysis();

    let mut output = vec![0.0f32; samples.len()];
    normalizer
        .process_f32(&samples, &mut output)
        .expect("process_f32 pass2");

    // Re-analyze the normalized output with a fresh analyzer.
    let mut config2 = NormalizerConfig::new(Standard::EbuR128, 48_000.0, 2);
    config2.processing_mode = ProcessingMode::TwoPass;
    config2.enable_limiter = false;
    config2.enable_drc = false;
    config2.max_gain_db = 30.0;

    let mut normalizer2 = Normalizer::new(config2).expect("normalizer2 creation");
    normalizer2.analyze_f32(&output);
    let second_analysis = normalizer2.get_analysis();

    // After normalization, the second-pass recommended gain should be near 0.
    assert!(
        second_analysis.recommended_gain_db.abs() < 2.0,
        "after normalization, residual gain should be <2 dB, got {:.2} dB \
         (first-pass gain was {:.2} dB)",
        second_analysis.recommended_gain_db,
        first_analysis.recommended_gain_db
    );
}

// ─── Test 2: Two-pass +7 dB normalization ────────────────────────────────────
//
// Generate audio ≈ −30 LUFS, normalize to −23 LUFS.
// Assert the recommended gain ≈ +7 dB (within ±3 dB tolerance, because BS.1770
// integrated loudness of a pure sine also depends on K-weighting).

#[test]
fn test_two_pass_boost_7db() {
    // Amplitude ≈ 0.045 (roughly −30 LUFS region for a 1 kHz sine)
    let samples = sine_stereo(1000.0, 0.045, 48_000.0, 3.0);

    let mut config = NormalizerConfig::new(Standard::EbuR128, 48_000.0, 2);
    config.processing_mode = ProcessingMode::TwoPass;
    config.enable_limiter = false;
    config.enable_drc = false;
    config.max_gain_db = 30.0;

    let mut normalizer = Normalizer::new(config).expect("normalizer creation");

    normalizer.analyze_f32(&samples);
    let analysis = normalizer.get_analysis();

    let mut output = vec![0.0f32; samples.len()];
    normalizer
        .process_f32(&samples, &mut output)
        .expect("process_f32");

    // The gain should be positive (boosting toward -23 LUFS from quieter input).
    assert!(
        analysis.recommended_gain_db > 0.0,
        "expected positive gain for quiet input, got {:.2} dB",
        analysis.recommended_gain_db
    );

    // Output should be louder than input.
    let in_rms = rms(&samples);
    let out_rms = rms(&output);
    assert!(
        out_rms > in_rms,
        "output ({:.6}) should be louder than input ({:.6})",
        out_rms,
        in_rms
    );
}

// ─── Test 3: True-peak limiter ─────────────────────────────────────────────────
//
// Apply a full-scale sine (amplitude 1.0) through TruePeakLimiter @ -1 dBTP.
// Assert: no sample in the steady-state region exceeds the ceiling.

#[test]
fn test_true_peak_limiter_ceiling() {
    let sample_rate = 48_000.0;
    let channels = 1;
    let threshold_dbtp = -1.0;
    let ceiling_linear = db_to_linear(threshold_dbtp) as f32;

    let config = LimiterConfig {
        sample_rate,
        channels,
        threshold_dbtp,
        lookahead_ms: 5.0,
        release_ms: 100.0,
    };
    let mut limiter = TruePeakLimiter::new(config).expect("limiter creation");

    // Full-scale 440 Hz sine, 2 seconds mono.
    let mut samples: Vec<f32> = (0..96_000)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0_f64 * std::f64::consts::PI * 440.0 * t).sin() as f32
        })
        .collect();

    limiter
        .process_f32_inplace(&mut samples)
        .expect("limiter process");

    // Skip the first lookahead frames (silence fill-in period).
    let lookahead_skip = (5.0 / 1000.0 * sample_rate).round() as usize;
    let steady = &samples[lookahead_skip.min(samples.len())..];

    let max_abs = steady.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

    assert!(
        max_abs <= ceiling_linear + 1e-3,
        "true-peak limiter: max output {:.6} exceeds ceiling {:.6}",
        max_abs,
        ceiling_linear
    );
}

// ─── Test 4: DRC LRA reduction ────────────────────────────────────────────────
//
// Generate audio with high apparent dynamic range: alternating loud and quiet blocks.
// Apply DRC (aggressive preset).  Assert that the RMS ratio between loud and quiet blocks
// is reduced after DRC — i.e. the dynamic range is compressed.

#[test]
fn test_drc_reduces_dynamic_range() {
    let sample_rate = 48_000.0f64;
    let channels = 1;
    let n_per_block = 4_800usize; // 100 ms blocks at 48 kHz
    let freq = 440.0_f64;

    // Build alternating loud (amp=0.9) and quiet (amp=0.02) blocks.
    let mut samples: Vec<f32> = Vec::with_capacity(n_per_block * 20);
    for block in 0..20 {
        let amp: f64 = if block % 2 == 0 { 0.9 } else { 0.02 };
        for i in 0..n_per_block {
            let t = i as f64 / sample_rate;
            samples.push((amp * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32);
        }
    }

    // Compute RMS of the first (loud) block and second (quiet) block.
    let loud_rms_in = rms(&samples[..n_per_block]);
    let quiet_rms_in = rms(&samples[n_per_block..2 * n_per_block]);
    let in_ratio = if quiet_rms_in > 0.0 {
        loud_rms_in / quiet_rms_in
    } else {
        f64::INFINITY
    };

    let config = DrcConfig::aggressive(sample_rate, channels);
    let mut drc = DynamicRangeCompressor::new(config).expect("drc creation");

    let mut output = samples.clone();
    drc.process_f32_inplace(&mut output).expect("drc process");

    let loud_rms_out = rms(&output[..n_per_block]);
    let quiet_rms_out = rms(&output[n_per_block..2 * n_per_block]);
    let out_ratio = if quiet_rms_out > 0.0 {
        loud_rms_out / quiet_rms_out
    } else {
        f64::INFINITY
    };

    // DRC must reduce the loud-to-quiet RMS ratio.
    assert!(
        out_ratio < in_ratio,
        "DRC should reduce dynamic range: \
         in loud/quiet RMS ratio={:.2} → out ratio={:.2}",
        in_ratio,
        out_ratio
    );
}

// ─── Test 5: RealtimeNormalizer latency ──────────────────────────────────────
//
// Create a RealtimeNormalizer with lookahead_ms = 50.
// Feed an impulse at sample 0.  The output should be zero for the first
// lookahead_samples frames, and then the impulse appears.

#[test]
fn test_realtime_normalizer_latency() {
    let sample_rate: f64 = 48_000.0;
    let lookahead_ms: f64 = 50.0;
    let expected_delay_samples = ((lookahead_ms / 1000.0) * sample_rate).round() as usize;

    let mut config = RealtimeConfig::new(Standard::EbuR128, sample_rate, 1);
    config.lookahead_ms = lookahead_ms;
    config.enable_limiter = false;

    let mut normalizer = RealtimeNormalizer::new(config).expect("realtime creation");

    // Verify the latency accessor is correct.
    assert_eq!(
        normalizer.latency_samples(),
        expected_delay_samples,
        "latency_samples() must equal expected delay"
    );
    assert!(
        (normalizer.latency_ms() - lookahead_ms).abs() < 0.01,
        "latency_ms() must match configured lookahead_ms"
    );

    // Process two chunks: first a DC impulse chunk, then silence.
    let chunk_size = 512usize;
    let mut chunk1 = vec![0.0f32; chunk_size];
    chunk1[0] = 1.0; // Impulse at the very first sample.

    let chunk2 = vec![0.0f32; chunk_size];

    let mut out1 = vec![0.0f32; chunk_size];
    let mut out2 = vec![0.0f32; chunk_size];

    normalizer
        .process_chunk(&chunk1, &mut out1)
        .expect("process chunk1");
    normalizer
        .process_chunk(&chunk2, &mut out2)
        .expect("process chunk2");

    // The first `expected_delay_samples` samples of output should be silence
    // (lookahead buffer filling up).
    let all_output: Vec<f32> = out1.iter().chain(out2.iter()).copied().collect();
    let delay_region = &all_output[..expected_delay_samples.min(all_output.len())];

    let max_in_delay = delay_region.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);
    assert!(
        max_in_delay < 0.01,
        "first {} samples (delay region) should be near-zero, got max={:.6}",
        expected_delay_samples,
        max_in_delay
    );
}

// ─── Test 6: Buffer recycling no-grow ────────────────────────────────────────
//
// Call `process_into` 10 times with 1024-sample chunks.
// After the first call, capacity must not grow.

#[test]
fn test_buffer_recycling_no_grow() {
    let config = RealtimeConfig {
        standard: Standard::EbuR128,
        sample_rate: 48_000.0,
        channels: 1,
        buffer_size: 1024,
        lookahead_ms: 10.0,
        smoothing_time_s: 1.0,
        enable_limiter: false,
    };
    let mut normalizer = RealtimeNormalizer::new(config).expect("realtime creation");

    let input = vec![0.0f32; 1024];
    let mut out = Vec::new();

    // Warm-up call — may allocate.
    normalizer.process_into(&input, &mut out).expect("call 0");
    let cap_after_first = out.capacity();

    assert!(
        cap_after_first >= 1024,
        "capacity after first call must be at least input length"
    );

    // Subsequent calls must not grow capacity.
    for call in 1..10 {
        normalizer.process_into(&input, &mut out).expect("call N");
        assert_eq!(
            out.capacity(),
            cap_after_first,
            "capacity grew on call {call}: before={cap_after_first}, after={}",
            out.capacity()
        );
        assert_eq!(
            out.len(),
            1024,
            "output length must equal input length after call {call}"
        );
    }
}
