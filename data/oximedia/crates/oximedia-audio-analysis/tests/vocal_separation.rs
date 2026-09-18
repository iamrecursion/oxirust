//! Vocal-separation direction and energy-preservation tests.
//!
//! HPSS (harmonic–percussive source separation) via median filtering is a crude
//! estimator, so these tests pin the DIRECTION of the result (harmonic output
//! correlates more with the harmonic/vocal source than with the broadband
//! instrumental noise) and that the output retains a meaningful fraction of the
//! input energy — they deliberately avoid over-pinning to high correlation values.

use oximedia_audio_analysis::separate::separate_vocals;
use oximedia_audio_analysis::AnalysisConfig;

const SR: f32 = 44_100.0;
const TWO_PI: f32 = 2.0 * std::f32::consts::PI;

/// Tiny seeded LCG producing white noise in `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
    }
}

/// Pearson correlation coefficient between two equal-length slices.
fn corr(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;

    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = (var_a * var_b).sqrt();
    if denom <= f32::EPSILON {
        0.0
    } else {
        cov / denom
    }
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}

/// Build a (vocal harmonic stack, instrumental white noise, mix) triple.
fn build_mix(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Vocal-like source: harmonic stack at 220 Hz (1st–4th harmonics, 1/h roll-off).
    let vocal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / SR;
            (1..=4)
                .map(|h| (0.5 / h as f32) * (TWO_PI * (220.0 * h as f32) * t).sin())
                .sum::<f32>()
        })
        .collect();

    // Instrumental-like source: broadband white noise (percussive/non-harmonic).
    let mut lcg = Lcg(0xBEEF);
    let inst: Vec<f32> = (0..n).map(|_| 0.3 * lcg.next_f32()).collect();

    let mix: Vec<f32> = vocal
        .iter()
        .zip(inst.iter())
        .map(|(&v, &i)| v + i)
        .collect();
    (vocal, inst, mix)
}

#[test]
fn harmonic_output_correlates_more_with_vocal_than_instrumental() {
    let config = AnalysisConfig::default();
    let n = 16_384_usize;
    let (vocal, inst, mix) = build_mix(n);

    let res = separate_vocals(&mix, SR, &config).expect("separation should succeed");

    // Align on the STEADY MIDDLE segment, discarding STFT edge transients.
    let m = res.harmonic.len().min(vocal.len()).min(inst.len());
    assert!(m > 4096, "separated harmonic too short: {m}");
    let lo = 2048_usize;
    let hi = m - 2048;

    let r_voc = corr(&res.harmonic[lo..hi], &vocal[lo..hi]);
    let r_ins = corr(&res.harmonic[lo..hi], &inst[lo..hi]);

    assert!(
        r_voc > r_ins,
        "harmonic output should correlate more with the vocal source (r_voc={r_voc:.3}) than \
         with the instrumental noise (r_ins={r_ins:.3})"
    );
    assert!(
        r_voc > 0.3,
        "harmonic↔vocal correlation should be clearly positive, got r_voc={r_voc:.3} (HPSS is \
         crude; this is a direction-only floor, not a tight bound)"
    );
}

#[test]
fn separation_preserves_energy_order() {
    let config = AnalysisConfig::default();
    let n = 16_384_usize;
    let (_vocal, _inst, mix) = build_mix(n);

    let res = separate_vocals(&mix, SR, &config).expect("separation should succeed");

    let mix_rms = rms(&mix);
    let harm_rms = rms(&res.harmonic);

    // If the harmonic output collapses toward silence, the synthesize step is
    // double-normalizing (oxifft::ifft already divides by N — an extra
    // /window_size attenuates by ~2048x). That is a real bug to fix at the root
    // in src/separate/sources.rs, NOT to mask by lowering this threshold.
    assert!(
        harm_rms > 0.01 * mix_rms,
        "harmonic RMS ({harm_rms:.6}) collapsed relative to mix RMS ({mix_rms:.6}) — \
         likely a synthesize divide-by-window attenuation bug"
    );
}
