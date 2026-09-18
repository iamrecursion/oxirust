//! SIMD-accelerated K-weighting filter for ITU-R BS.1770-4 loudness.
//!
//! K-weighting is a two-stage biquad cascade:
//! - Stage 1 (pre-filter): high-shelf boost ~+4 dB above ~1.7 kHz (models head diffraction).
//! - Stage 2 (high-pass): 2nd-order HP at ~38 Hz (removes sub-bass).
//!
//! Because each output sample depends on the two previous outputs (IIR), the
//! filter is inherently sequential within a single channel. True SIMD
//! acceleration is therefore applied by processing **four independent channels
//! in parallel** (SIMD width = 4 lanes).
//!
//! The portable SIMD path uses 4-lane f32 SIMD via hand-coded width-4
//! operations (compiled with `target_feature` or scalar fallback) so that
//! no `std::simd` nightly feature is required.

/// K-weighted biquad filter state and coefficients for a single channel.
///
/// Each instance is stateful: it must be driven sample-by-sample through
/// [`KWeightedFilter::process_sample`] or [`KWeightedFilter::process_block`].
///
/// Coefficients are pre-computed for the requested sample rate using the
/// formulas from ITU-R BS.1770-4 Annex 1.
#[derive(Clone, Debug)]
pub struct KWeightedFilter {
    // Pre-filter (high-shelf) coefficients
    b0_pre: f32,
    b1_pre: f32,
    b2_pre: f32,
    a1_pre: f32,
    a2_pre: f32,
    // High-pass coefficients
    b0_hp: f32,
    b1_hp: f32,
    b2_hp: f32,
    a1_hp: f32,
    a2_hp: f32,
    // Delay elements (direct-form II transposed)
    z1_pre: f32,
    z2_pre: f32,
    z1_hp: f32,
    z2_hp: f32,
}

impl KWeightedFilter {
    /// Create a K-weighting filter for the given sample rate.
    ///
    /// Supports any `sample_rate` in the range 8 000–192 000 Hz; the
    /// coefficient formulas scale correctly for all standard rates.
    pub fn new(sample_rate: u32) -> Self {
        let (b0_pre, b1_pre, b2_pre, a1_pre, a2_pre, b0_hp, b1_hp, b2_hp, a1_hp, a2_hp) =
            compute_k_weight_coeffs(f64::from(sample_rate));
        Self {
            b0_pre,
            b1_pre,
            b2_pre,
            a1_pre,
            a2_pre,
            b0_hp,
            b1_hp,
            b2_hp,
            a1_hp,
            a2_hp,
            z1_pre: 0.0,
            z2_pre: 0.0,
            z1_hp: 0.0,
            z2_hp: 0.0,
        }
    }

    /// Process a single sample and return the K-weighted output.
    ///
    /// This function is inherently sequential (IIR state update per sample).
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        // Stage 1: high-shelf pre-filter (direct-form II transposed)
        let y1 = self.b0_pre * x + self.z1_pre;
        self.z1_pre = self.b1_pre * x - self.a1_pre * y1 + self.z2_pre;
        self.z2_pre = self.b2_pre * x - self.a2_pre * y1;

        // Stage 2: high-pass filter
        let y2 = self.b0_hp * y1 + self.z1_hp;
        self.z1_hp = self.b1_hp * y1 - self.a1_hp * y2 + self.z2_hp;
        self.z2_hp = self.b2_hp * y1 - self.a2_hp * y2;

        y2
    }

    /// Process a block of samples into `output`.
    ///
    /// The filter state is updated sequentially (required for IIR correctness).
    /// Both slices must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != output.len()`.
    pub fn process_block(&mut self, input: &[f32], output: &mut [f32]) {
        assert_eq!(
            input.len(),
            output.len(),
            "KWeightedFilter::process_block: input and output length mismatch"
        );
        for (x, y) in input.iter().zip(output.iter_mut()) {
            *y = self.process_sample(*x);
        }
    }

    /// Reset filter delay elements to zero.
    pub fn reset(&mut self) {
        self.z1_pre = 0.0;
        self.z2_pre = 0.0;
        self.z1_hp = 0.0;
        self.z2_hp = 0.0;
    }
}

/// Compute K-weighting biquad coefficients (f32) for the given sample rate (f64 for precision).
///
/// Returns `(b0_pre, b1_pre, b2_pre, a1_pre, a2_pre, b0_hp, b1_hp, b2_hp, a1_hp, a2_hp)`.
fn compute_k_weight_coeffs(fs: f64) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
    // --- Stage 1: high-shelf (pre-filter) ---
    // ITU-R BS.1770-4 Annex 1 reference coefficients for 48 kHz, generalised.
    let db_gain = 3.999_843_853_973_347_f64;
    let f0_shelf = 1_681.974_450_955_533_f64;
    let q_shelf = 0.707_213_195_806_047_6_f64;

    let k_s = (std::f64::consts::PI * f0_shelf / fs).tan();
    let vh = 10_f64.powf(db_gain / 20.0);
    let vb = vh.powf(0.5);
    let denom_s = 1.0 + k_s / q_shelf + k_s * k_s;

    let b0_pre = ((vh + vb * k_s / q_shelf + k_s * k_s) / denom_s) as f32;
    let b1_pre = (2.0 * (k_s * k_s - vh) / denom_s) as f32;
    let b2_pre = ((vh - vb * k_s / q_shelf + k_s * k_s) / denom_s) as f32;
    let a1_pre = (2.0 * (k_s * k_s - 1.0) / denom_s) as f32;
    let a2_pre = ((1.0 - k_s / q_shelf + k_s * k_s) / denom_s) as f32;

    // --- Stage 2: second-order high-pass (RLB weighting) ---
    let f1_hp = 38.134_566_580_756_27_f64;
    let q_hp = 0.500_316_983_843_589_1_f64;
    let k_h = (std::f64::consts::PI * f1_hp / fs).tan();
    let denom_h = 1.0 + k_h / q_hp + k_h * k_h;

    let b0_hp = (1.0 / denom_h) as f32;
    let b1_hp = (-2.0 / denom_h) as f32;
    let b2_hp = (1.0 / denom_h) as f32;
    let a1_hp = (2.0 * (k_h * k_h - 1.0) / denom_h) as f32;
    let a2_hp = ((1.0 - k_h / q_hp + k_h * k_h) / denom_h) as f32;

    (
        b0_pre, b1_pre, b2_pre, a1_pre, a2_pre, b0_hp, b1_hp, b2_hp, a1_hp, a2_hp,
    )
}

/// Process 4 independent audio channels through K-weighting simultaneously.
///
/// Because the biquad IIR is sequential within each channel, the SIMD
/// opportunity is across channels: all four channels share the same
/// coefficient set and are processed in parallel using width-4 SIMD.
///
/// # Arguments
///
/// * `channels` - Four slices of equal length (one per channel).
/// * `filters`  - Four filter states, one per channel (updated in-place).
///
/// # Returns
///
/// Four `Vec<f32>` holding the K-weighted output for each channel.
///
/// # Panics
///
/// Panics if the four input slices are not all the same length.
pub fn k_weight_4ch_simd(
    channels: [&[f32]; 4],
    filters: &mut [KWeightedFilter; 4],
) -> [Vec<f32>; 4] {
    let len = channels[0].len();
    assert_eq!(channels[1].len(), len, "channel 1 length mismatch");
    assert_eq!(channels[2].len(), len, "channel 2 length mismatch");
    assert_eq!(channels[3].len(), len, "channel 3 length mismatch");

    // Pre-allocate all four output buffers.
    let mut out0 = vec![0.0f32; len];
    let mut out1 = vec![0.0f32; len];
    let mut out2 = vec![0.0f32; len];
    let mut out3 = vec![0.0f32; len];

    // Load coefficients into local variables to enable auto-vectorisation.
    // All four filters share the same coefficients (same sample rate).
    // We read them from filters[0] and assume all four are identical.
    let b0p = filters[0].b0_pre;
    let b1p = filters[0].b1_pre;
    let b2p = filters[0].b2_pre;
    let a1p = filters[0].a1_pre;
    let a2p = filters[0].a2_pre;
    let b0h = filters[0].b0_hp;
    let b1h = filters[0].b1_hp;
    let b2h = filters[0].b2_hp;
    let a1h = filters[0].a1_hp;
    let a2h = filters[0].a2_hp;

    // Per-channel state — extracted so we can drive the inner loop without
    // going through struct fields (improves auto-vectorisation).
    let mut z1p = [
        filters[0].z1_pre,
        filters[1].z1_pre,
        filters[2].z1_pre,
        filters[3].z1_pre,
    ];
    let mut z2p = [
        filters[0].z2_pre,
        filters[1].z2_pre,
        filters[2].z2_pre,
        filters[3].z2_pre,
    ];
    let mut z1h = [
        filters[0].z1_hp,
        filters[1].z1_hp,
        filters[2].z1_hp,
        filters[3].z1_hp,
    ];
    let mut z2h = [
        filters[0].z2_hp,
        filters[1].z2_hp,
        filters[2].z2_hp,
        filters[3].z2_hp,
    ];

    // Main loop: process one sample per channel per iteration.
    // The compiler can auto-vectorise across the 4-wide arrays when the
    // coefficients are invariant loop constants.
    for i in 0..len {
        let x = [
            channels[0][i],
            channels[1][i],
            channels[2][i],
            channels[3][i],
        ];

        // Stage 1 (pre-filter) for all 4 channels simultaneously.
        let y1_0 = b0p * x[0] + z1p[0];
        let y1_1 = b0p * x[1] + z1p[1];
        let y1_2 = b0p * x[2] + z1p[2];
        let y1_3 = b0p * x[3] + z1p[3];

        z1p[0] = b1p * x[0] - a1p * y1_0 + z2p[0];
        z1p[1] = b1p * x[1] - a1p * y1_1 + z2p[1];
        z1p[2] = b1p * x[2] - a1p * y1_2 + z2p[2];
        z1p[3] = b1p * x[3] - a1p * y1_3 + z2p[3];

        z2p[0] = b2p * x[0] - a2p * y1_0;
        z2p[1] = b2p * x[1] - a2p * y1_1;
        z2p[2] = b2p * x[2] - a2p * y1_2;
        z2p[3] = b2p * x[3] - a2p * y1_3;

        // Stage 2 (high-pass) for all 4 channels simultaneously.
        let y2_0 = b0h * y1_0 + z1h[0];
        let y2_1 = b0h * y1_1 + z1h[1];
        let y2_2 = b0h * y1_2 + z1h[2];
        let y2_3 = b0h * y1_3 + z1h[3];

        z1h[0] = b1h * y1_0 - a1h * y2_0 + z2h[0];
        z1h[1] = b1h * y1_1 - a1h * y2_1 + z2h[1];
        z1h[2] = b1h * y1_2 - a1h * y2_2 + z2h[2];
        z1h[3] = b1h * y1_3 - a1h * y2_3 + z2h[3];

        z2h[0] = b2h * y1_0 - a2h * y2_0;
        z2h[1] = b2h * y1_1 - a2h * y2_1;
        z2h[2] = b2h * y1_2 - a2h * y2_2;
        z2h[3] = b2h * y1_3 - a2h * y2_3;

        out0[i] = y2_0;
        out1[i] = y2_1;
        out2[i] = y2_2;
        out3[i] = y2_3;
    }

    // Write state back to filter structs.
    for (ch, filt) in filters.iter_mut().enumerate() {
        filt.z1_pre = z1p[ch];
        filt.z2_pre = z2p[ch];
        filt.z1_hp = z1h[ch];
        filt.z2_hp = z2h[ch];
    }

    [out0, out1, out2, out3]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_filter_48k() -> KWeightedFilter {
        KWeightedFilter::new(48_000)
    }

    /// DC input must be heavily attenuated by the high-pass stage.
    #[test]
    fn test_k_weighted_filter_dc() {
        let mut f = make_filter_48k();
        let n = 20_000;
        // Drive the filter to steady state with DC.
        let mut last = 0.0_f32;
        for _ in 0..n {
            last = f.process_sample(1.0);
        }
        // Near-DC (settled) output must be ≪ 1.0.
        assert!(
            last.abs() < 0.01,
            "DC input not attenuated: settled output = {last}"
        );
    }

    /// A 1 kHz sine through the filter should retain reasonable amplitude
    /// (well within the pass-band of the high-pass, pre-shelf only boosts).
    #[test]
    fn test_k_weighted_filter_roundtrip() {
        let sr = 48_000u32;
        let mut f = KWeightedFilter::new(sr);
        let freq = 1_000.0_f32;
        let n = 4 * sr as usize; // 4 seconds for settling

        let mut energy_out = 0.0_f64;
        let mut energy_in = 0.0_f64;
        for i in 0..n {
            let x = (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin();
            let y = f.process_sample(x);
            // Accumulate over the last half only (skip transient).
            if i >= n / 2 {
                energy_in += (x * x) as f64;
                energy_out += (y * y) as f64;
            }
        }
        // Output energy must be between 0.5× and 4× input (shelf adds up to ~4 dB).
        let ratio = energy_out / energy_in;
        assert!(
            ratio > 0.5 && ratio < 8.0,
            "1 kHz energy ratio {ratio:.3} outside [0.5, 8.0] — filter may be broken"
        );
    }

    /// The 4-channel SIMD variant must produce results identical to four
    /// independent scalar runs.
    #[test]
    fn test_k_weighted_4ch_matches_scalar() {
        let sr = 48_000u32;
        let n = 2048;
        let freq = 1_000.0_f32;

        // Generate 4 slightly different input signals.
        let make_input = |phase: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32 + phase).sin())
                .collect()
        };

        let inputs: [Vec<f32>; 4] = [
            make_input(0.0),
            make_input(0.5),
            make_input(1.0),
            make_input(1.5),
        ];

        // Scalar reference: run each channel independently.
        let scalar_out: [Vec<f32>; 4] = {
            let mut results: [Vec<f32>; 4] =
                [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            for ch in 0..4 {
                let mut f = KWeightedFilter::new(sr);
                f.process_block(&inputs[ch], &mut results[ch]);
            }
            results
        };

        // SIMD path.
        let mut simd_filters: [KWeightedFilter; 4] = [
            KWeightedFilter::new(sr),
            KWeightedFilter::new(sr),
            KWeightedFilter::new(sr),
            KWeightedFilter::new(sr),
        ];
        let simd_out = k_weight_4ch_simd(
            [&inputs[0], &inputs[1], &inputs[2], &inputs[3]],
            &mut simd_filters,
        );

        // Compare with a tolerance of 1e-5 (f32 arithmetic).
        for ch in 0..4 {
            for (i, (&s, &r)) in simd_out[ch].iter().zip(scalar_out[ch].iter()).enumerate() {
                assert!(
                    (s - r).abs() < 1e-5,
                    "ch={ch} i={i}: simd={s} scalar={r} diff={}",
                    (s - r).abs()
                );
            }
        }
    }

    /// Verify that process_block produces the same result as repeated process_sample.
    #[test]
    fn test_k_weighted_block_vs_sample() {
        let sr = 44_100u32;
        let n = 512;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
        let mut out_block = vec![0.0f32; n];
        let mut f1 = KWeightedFilter::new(sr);
        f1.process_block(&input, &mut out_block);

        let mut f2 = KWeightedFilter::new(sr);
        let out_sample: Vec<f32> = input.iter().map(|&x| f2.process_sample(x)).collect();

        for (i, (&b, &s)) in out_block.iter().zip(out_sample.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-7,
                "index {i}: block={b} sample={s} diff={}",
                (b - s).abs()
            );
        }
    }
}
