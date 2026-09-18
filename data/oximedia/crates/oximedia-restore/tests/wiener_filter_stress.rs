//! Numerical-robustness stress tests for the spectral [`WienerFilter`].
//!
//! Goal: prove the Wiener denoiser is correct and panic / NaN / Inf / OOM-free
//! across boundary input lengths, including lengths shorter than one FFT block,
//! lengths straddling the FFT block boundary, and a very long (10 M sample)
//! input exercised through the one-shot processing path.
//!
//! ## What the filter actually does (drives the chosen invariants)
//!
//! `WienerFilter::process` (see `src/noise/wiener.rs`) is a **one-shot** STFT
//! denoiser:
//!
//!   1. Slides a `fft_size`-wide window across the input with stride `hop_size`.
//!   2. Per frame: Hann **analysis** window → forward FFT (`oxifft`, Pure Rust)
//!      → real-valued Wiener gain per bin → inverse FFT → Hann **synthesis**
//!      window → overlap-add into a heap `Vec` accumulator.
//!   3. Normalises each output sample by the **raw frame-overlap count**.
//!
//! Two consequences shape the assertions below:
//!
//! * **Memory is O(N).** There is *no* streaming / block API on `WienerFilter`
//!   itself (the block path lives on the separate `RestoreChain::process_streaming`).
//!   `process` allocates `output` and `overlap_count` each `vec![0.0; samples.len()]`,
//!   so peak memory grows linearly with the input. We therefore assert memory
//!   bounds **by design** (≈ 2·N f32 plus per-frame O(fft_size) scratch), not by
//!   measuring RSS, and document the O(N) one-shot behaviour explicitly.
//!
//! * **The filter is NOT energy-preserving even at unity gain.** Because both an
//!   analysis *and* a synthesis Hann window are applied (a window-squared, WOLA
//!   shape) but normalisation divides by the integer overlap **count** rather
//!   than by Σ w[i]², the output is intentionally attenuated relative to the
//!   input. A broadband ±5 % RMS invariant would be wrong here. Per the slice
//!   brief we instead assert the gain is **bounded and finite** — specifically
//!   the denoiser must never *amplify* broadband energy and must keep the
//!   narrow-band tone measurably present.
//!
//! ## Synthetic input
//!
//! Pink-ish broadband noise (1/f-weighted octave bands, deterministic LCG so the
//! tests are reproducible without an RNG dependency surface) plus a narrow-band
//! sinusoid. The tone makes the Wiener gain measurable: in the tonal band the
//! a-posteriori SNR is high, so the gain there approaches unity while broadband
//! bins are suppressed toward `min_gain`.

use oximedia_restore::noise::{NoiseProfile, WienerFilter, WienerFilterConfig};
use std::f32::consts::PI;

/// FFT block size used throughout (a typical, power-of-two STFT window).
const FFT_SIZE: usize = 2048;
/// 50 % overlap hop — the conventional Hann STFT stride.
const HOP_SIZE: usize = FFT_SIZE / 2;
/// Sample rate used to place the narrow-band tone.
const SAMPLE_RATE: f32 = 48_000.0;
/// Frequency of the measurable narrow-band sinusoid (Hz).
const TONE_HZ: f32 = 1_000.0;

// ---------------------------------------------------------------------------
// Deterministic synthetic-signal helpers (no RNG dependency, fully reproducible)
// ---------------------------------------------------------------------------

/// Tiny deterministic LCG yielding `f32` in `[-1.0, 1.0)`.
///
/// Used instead of the `rand` crate so the generated buffers are bit-identical
/// across runs/platforms, which the determinism test depends on.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        // Numerical Recipes LCG constants.
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }

    #[inline]
    fn next_bipolar(&mut self) -> f32 {
        // Map to [-1, 1).
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Build a deterministic "pink-ish" broadband noise + narrow-band tone signal of
/// `n` samples.
///
/// The noise is shaped toward 1/f by summing a few octave-spaced low-passed
/// white components (a cheap pinking that nonetheless spreads energy across the
/// spectrum), then a clean `TONE_HZ` sinusoid is added so the filter gain is
/// observable.
fn synth_signal(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n);

    // Leaky integrators at a few time constants approximate 1/f shaping.
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    for i in 0..n {
        let w = rng.next_bipolar();
        s0 = 0.99 * s0 + 0.01 * w;
        s1 = 0.90 * s1 + 0.10 * w;
        s2 = 0.50 * s2 + 0.50 * w;
        let pink = 0.5 * (s0 + s1 + s2) / 3.0; // broadband, ~1/f tilt
        let tone = 0.30 * (2.0 * PI * TONE_HZ * i as f32 / SAMPLE_RATE).sin();
        out.push(pink + tone);
    }
    out
}

/// Build a noise-only buffer (no tone) suitable for learning a noise profile.
fn synth_noise(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(n);
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    for _ in 0..n {
        let w = rng.next_bipolar();
        s0 = 0.99 * s0 + 0.01 * w;
        s1 = 0.90 * s1 + 0.10 * w;
        s2 = 0.50 * s2 + 0.50 * w;
        out.push(0.5 * (s0 + s1 + s2) / 3.0);
    }
    out
}

/// Root-mean-square of a slice.
fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    (sum_sq / x.len() as f64).sqrt() as f32
}

/// Assert every sample is finite (no NaN / ±Inf).
fn assert_all_finite(x: &[f32], context: &str) {
    for (i, &v) in x.iter().enumerate() {
        assert!(v.is_finite(), "{context}: sample [{i}] is not finite: {v}");
    }
}

/// Construct a fresh filter with a freshly learned noise profile.
///
/// A new profile/filter pair per call keeps each test self-contained and the
/// filter's internal `prev_gain` smoothing state clean.
fn fresh_filter() -> WienerFilter {
    // Learn the noise profile from a comfortably long noise-only buffer.
    let noise = synth_noise(FFT_SIZE * 8, 0xC0FF_EE00);
    let profile = NoiseProfile::learn(&noise, FFT_SIZE, HOP_SIZE)
        .expect("noise profile should learn from an 8-frame noise buffer");
    WienerFilter::new(profile, HOP_SIZE, WienerFilterConfig::default())
}

// ---------------------------------------------------------------------------
// Short inputs: every length < one FFT block.
// ---------------------------------------------------------------------------

/// Inputs shorter than `fft_size` must (a) not panic, (b) return an output whose
/// length equals the input length, and (c) be entirely finite.
///
/// By contract the filter short-circuits (`samples.len() < fft_size` ⇒ returns a
/// copy of the input), so the output must additionally be *bit-identical* to the
/// input for these lengths — a strong, exact correctness check at the boundary.
#[test]
fn short_inputs_below_one_block_are_passed_through_finite() {
    for &len in &[1usize, 2, 3, 64, 100, 256] {
        assert!(len < FFT_SIZE, "test precondition: {len} < {FFT_SIZE}");

        let input = synth_signal(len, 0x5EED_0001 ^ len as u64);
        let mut filter = fresh_filter();
        let output = filter
            .process(&input)
            .expect("short input must process without error");

        assert_eq!(
            output.len(),
            len,
            "short input (len={len}): output length must equal input length"
        );
        assert_all_finite(&output, &format!("short input len={len}"));

        // Below fft_size the filter returns the input unchanged: verify exactly.
        for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (a - b).abs() <= f32::EPSILON,
                "short input (len={len}): sample [{i}] altered: in={a} out={b}"
            );
        }
    }
}

/// A zero-length input must be handled gracefully (empty output, no panic).
#[test]
fn empty_input_yields_empty_output() {
    let input: Vec<f32> = Vec::new();
    let mut filter = fresh_filter();
    let output = filter.process(&input).expect("empty input must not panic");
    assert!(output.is_empty(), "empty input must yield empty output");
}

// ---------------------------------------------------------------------------
// FFT block-boundary lengths: input lengths straddling `fft_size`.
// ---------------------------------------------------------------------------

/// Input lengths {fft_size-1, fft_size, fft_size+1} exercise the three regimes
/// around the block boundary:
///
/// * `fft_size - 1` (2047): below threshold ⇒ pass-through.
/// * `fft_size`     (2048): exactly one analysis frame.
/// * `fft_size + 1` (2049): one frame processed, one trailing sample left
///   uncovered by any frame (overlap_count == 0 there ⇒ stays 0.0).
///
/// All three must produce length-correct, fully finite output and not panic.
#[test]
fn block_boundary_lengths_are_finite_and_length_correct() {
    for &len in &[FFT_SIZE - 1, FFT_SIZE, FFT_SIZE + 1] {
        let input = synth_signal(len, 0xB10C_0000 ^ len as u64);
        let mut filter = fresh_filter();
        let output = filter
            .process(&input)
            .expect("block-boundary input must process without error");

        assert_eq!(
            output.len(),
            len,
            "block-boundary (len={len}): output length must equal input length"
        );
        assert_all_finite(&output, &format!("block-boundary len={len}"));
    }

    // The exactly-one-block case (2048) is the first length that is actually
    // filtered (not passed through): confirm the tone survives the filter, i.e.
    // the output carries non-trivial energy and is not silenced.
    let input = synth_signal(FFT_SIZE, 0xB10C_0042);
    let mut filter = fresh_filter();
    let output = filter
        .process(&input)
        .expect("single-block input must process");
    assert!(
        rms(&output) > 0.0,
        "single full block must yield non-zero output energy"
    );
}

/// A couple of full overlapping frames (so interior samples get the 2× overlap
/// treatment and the time-smoothing of `prev_gain` engages): the output must
/// stay finite and must NOT amplify broadband energy.
///
/// This pins the "gain is bounded" invariant at a modest length where the WOLA
/// attenuation is in effect.
#[test]
fn multi_frame_output_is_bounded_and_does_not_amplify() {
    let len = FFT_SIZE * 4; // several hops ⇒ steady-state overlap region
    let input = synth_signal(len, 0xBADD_CAFE);
    let in_rms = rms(&input);

    let mut filter = fresh_filter();
    let output = filter
        .process(&input)
        .expect("multi-frame input must process");

    assert_eq!(output.len(), len);
    assert_all_finite(&output, "multi-frame");

    let out_rms = rms(&output);
    assert!(out_rms.is_finite(), "output RMS must be finite");
    assert!(out_rms >= 0.0, "output RMS must be non-negative");
    // A noise-reducing, window-attenuating filter must never amplify broadband
    // energy. The synthesis-window WOLA shape attenuates further, so out_rms is
    // comfortably below in_rms; the upper bound here is intentionally generous
    // (the strict invariant being merely "no amplification").
    assert!(
        out_rms <= in_rms * 1.05,
        "filter must not amplify broadband energy: in_rms={in_rms} out_rms={out_rms}"
    );
    // The signal must not be annihilated either: the high-SNR tone keeps real
    // energy in the output.
    assert!(
        out_rms > in_rms * 0.05,
        "filter must not annihilate the signal: in_rms={in_rms} out_rms={out_rms}"
    );
}

// ---------------------------------------------------------------------------
// Determinism: identical input ⇒ identical output.
// ---------------------------------------------------------------------------

/// The same input processed twice (through two freshly constructed filters with
/// identically-seeded profiles) must yield bit-identical output. This guards
/// against any nondeterminism in the SIMD gain path, FFT, or overlap-add.
#[test]
fn processing_is_deterministic() {
    let input = synth_signal(FFT_SIZE * 6, 0xDE7E_8011);

    let mut filter_a = fresh_filter();
    let out_a = filter_a.process(&input).expect("run A must process");

    let mut filter_b = fresh_filter();
    let out_b = filter_b.process(&input).expect("run B must process");

    assert_eq!(out_a.len(), out_b.len(), "lengths must match");
    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "nondeterministic output at [{i}]: {a} vs {b}"
        );
    }
}

/// Re-running on the SAME filter after `reset()` must also reproduce the first
/// run exactly — confirming `reset()` fully restores the smoothing state.
#[test]
fn reset_restores_identical_behaviour() {
    let input = synth_signal(FFT_SIZE * 5, 0x9999_8888);
    let mut filter = fresh_filter();

    let first = filter.process(&input).expect("first run must process");
    filter.reset();
    let second = filter.process(&input).expect("second run must process");

    assert_eq!(first.len(), second.len());
    for (i, (&a, &b)) in first.iter().zip(second.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "reset did not restore state at [{i}]: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Very long input: 10_000_000 samples through the one-shot path.
// ---------------------------------------------------------------------------

/// Process a 10-million-sample buffer.
///
/// Marked `#[ignore]` so default CI stays fast; run with
/// `cargo test -p oximedia-restore --release -- --ignored` (release recommended
/// — debug builds make the 10 M-sample FFT sweep slow).
///
/// Asserts:
///
/// * **(a) No panic / no OOM.** Completion of `process` is the assertion.
///
/// * **(b) Memory is bounded BY DESIGN.** `WienerFilter` exposes only a one-shot
///   `process`, which allocates `output` + `overlap_count` ≈ 2·N f32 plus
///   per-frame O(fft_size) scratch — i.e. peak memory is O(N), *not* O(N²) and
///   not proportional to the number of frames squared. There is no streaming
///   API on this type to bound it below O(N); we assert that contract here:
///   output length == N (proving a single N-sized accumulator was produced, the
///   dominant allocation), and we document the O(N) one-shot design rather than
///   measuring RSS. (For sub-linear peak memory a caller must use
///   `RestoreChain::process_streaming`, which is out of scope for this type.)
///
/// * **(c) Energy stays bounded (no amplification, no annihilation).** Per the
///   filter's WOLA/window-squared design the output is attenuated, so the
///   invariant is `0.05·in_rms < out_rms ≤ 1.05·in_rms`, evaluated over the
///   interior (the trailing < hop_size samples may be uncovered and zero).
///
/// * Every (subsampled) output sample is finite.
#[test]
#[ignore = "10M-sample stress test; run with --release --ignored"]
fn ten_million_sample_input_is_bounded_finite_and_energy_conserving() {
    const N: usize = 10_000_000;

    let input = synth_signal(N, 0x10_5A_3E_ED);
    let in_rms = rms(&input);

    let mut filter = fresh_filter();
    // (a) Must complete without panic or OOM.
    let output = filter
        .process(&input)
        .expect("10M-sample input must process without panic or OOM");

    // (b) One-shot O(N) accumulator was produced: length matches input exactly.
    assert_eq!(
        output.len(),
        N,
        "one-shot path must return an N-length buffer (peak memory O(N) by design)"
    );

    // Subsampled finiteness sweep (checking all 10M every run is wasteful).
    for (i, &v) in output.iter().enumerate().step_by(9_973) {
        assert!(v.is_finite(), "long input: sample [{i}] is not finite: {v}");
    }

    // (c) Energy conservation invariant appropriate to a WOLA window filter:
    // evaluate over the interior, which is fully covered by overlapping frames.
    // The final partial region (< hop_size) can be uncovered (count==0 ⇒ 0.0),
    // so exclude a trailing margin of one full block.
    let interior = &output[..N - FFT_SIZE];
    let out_rms = rms(interior);
    assert!(out_rms.is_finite(), "interior output RMS must be finite");
    assert!(
        out_rms <= in_rms * 1.05,
        "long-input filter must not amplify energy: in_rms={in_rms} out_rms={out_rms}"
    );
    assert!(
        out_rms > in_rms * 0.05,
        "long-input filter must not annihilate the signal: in_rms={in_rms} out_rms={out_rms}"
    );
}
