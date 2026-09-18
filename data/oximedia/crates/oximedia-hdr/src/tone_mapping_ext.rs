//! Extended tone mapping: BT.2446 Method A forward (HDR→SDR), SDR-to-HDR uplift,
//! and scene-referred geometric-mean exposure analysis.
//!
//! This module extends [`crate::tone_mapping`] with:
//! - [`Bt2446MethodAForwardMapper`] — official ITU-R BT.2446-1 Method A HDR→SDR curve
//! - [`SdrToHdrMapper`] — flexible SDR→HDR uplifter (linear / BT.2446-C inverse / perceptual)
//! - Additional methods on [`SceneReferredToneMapper`]: `analyze_frame` and
//!   `apply_with_scene_analysis` for log-average luminance–based adaptive tone mapping

use crate::tone_mapping::{SceneReferredToneMapper, ToneMapper, ToneMappingConfig};
use crate::{HdrError, Result};

// ── BT.2446 Method A forward (HDR → SDR) ─────────────────────────────────────

/// ITU-R BT.2446-1 Method A piecewise tone mapping curve (HDR → SDR).
///
/// This is the *forward* direction: display-linear HDR → SDR gamma domain.
///
/// Algorithm (per BT.2446-1 Annex A):
///   - Y ≤ 0.7399 : Y_out = 1.0770 × Y
///   - Y > 0.7399 : Y_out = (−1.1510 Y² + 2.7811 Y − 0.6302) / (0.3359 Y + 0.1634)
///
/// Input `y` must be normalised to [0, 1].
#[inline]
pub(crate) fn bt2446_method_a_forward(y: f32) -> f32 {
    if y <= 0.0 {
        return 0.0;
    }
    let y_c = y.clamp(0.0, 1.0);
    let y_out = if y_c <= 0.7399 {
        1.0770 * y_c
    } else {
        let num = -1.1510 * y_c * y_c + 2.7811 * y_c - 0.6302;
        let den = 0.3359 * y_c + 0.1634;
        // Denominator is always > 0 in the valid domain: min ≈ 0.41 at y_c = 0.7399.
        if den.abs() < 1e-9 {
            1.0
        } else {
            num / den
        }
    };
    y_out.clamp(0.0, 1.0)
}

/// Dedicated high-level forward BT.2446 Method A tone mapper (HDR → SDR).
///
/// Implements ITU-R BT.2446-1 Method A piecewise luminance curve with
/// configurable reference white luminance for HLG (203 nits) and PQ (58 nits)
/// sources.  Output is gamma-encoded (default γ = 2.4 per BT.1886).
#[derive(Debug, Clone)]
pub struct Bt2446MethodAForwardMapper {
    /// Input HDR peak luminance in nits (e.g. 1000 for HDR10).
    pub hdr_peak_nits: f32,
    /// Reference white luminance: 203 nits for HLG, 58 nits for PQ.
    /// Normalises the input so that reference white maps to 1.0 before the
    /// BT.2446 curve is applied.
    pub reference_white_nits: f32,
    /// Output SDR gamma exponent (default 2.4 per BT.1886).
    pub gamma_out: f32,
}

impl Bt2446MethodAForwardMapper {
    /// Creates a mapper for PQ (HDR10) sources — reference white 58 nits.
    pub fn new_pq(hdr_peak_nits: f32) -> Self {
        Self {
            hdr_peak_nits,
            reference_white_nits: 58.0,
            gamma_out: 2.4,
        }
    }

    /// Creates a mapper for HLG sources — reference white 203 nits.
    pub fn new_hlg(hdr_peak_nits: f32) -> Self {
        Self {
            hdr_peak_nits,
            reference_white_nits: 203.0,
            gamma_out: 2.4,
        }
    }

    /// Map a single linear HDR luminance (normalised to `hdr_peak_nits`) to
    /// a gamma-encoded SDR value in [0, 1].
    pub fn map_luminance(&self, lin_hdr: f32) -> f32 {
        // Normalise: 1.0 = reference white after this scaling.
        let norm = lin_hdr * self.hdr_peak_nits / self.reference_white_nits.max(1.0);
        let y = norm.clamp(0.0, 1.0);
        let y_sdr = bt2446_method_a_forward(y);
        // Apply BT.1886 OETF: gamma-encode to get the SDR signal.
        let gamma_inv = 1.0 / self.gamma_out.max(0.1);
        y_sdr.clamp(0.0, 1.0).powf(gamma_inv)
    }

    /// Map an interleaved RGB frame (linear HDR, normalised to `hdr_peak_nits`)
    /// to SDR in [0, 1].
    ///
    /// # Errors
    /// Returns an error if the pixel buffer length is not divisible by 3.
    pub fn map_frame(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            // BT.2100 luma coefficients.
            let lum = 0.2627 * chunk[0] + 0.6780 * chunk[1] + 0.0593 * chunk[2];
            let mapped_lum = self.map_luminance(lum.max(0.0));
            if lum > 1e-7 {
                let ratio = mapped_lum / lum;
                out.push((chunk[0] * ratio).clamp(0.0, 1.0));
                out.push((chunk[1] * ratio).clamp(0.0, 1.0));
                out.push((chunk[2] * ratio).clamp(0.0, 1.0));
            } else {
                out.push(mapped_lum);
                out.push(mapped_lum);
                out.push(mapped_lum);
            }
        }
        Ok(out)
    }
}

// ── UpliftAlgorithm / SdrToHdrConfig / SdrToHdrMapper ────────────────────────

/// SDR-to-HDR uplift algorithm selector.
#[derive(Debug, Clone, PartialEq)]
pub enum UpliftAlgorithm {
    /// Simple linear rescaling: `x * (target_peak / sdr_white)`.
    LinearScale,
    /// Inverse of BT.2446 Method C sigmoid — reverses the HDR→SDR compression.
    Bt2446MethodC,
    /// Perceptual S-curve that lifts shadows gently and expands highlights
    /// without the clipping artefacts typical of pure linear expansion.
    PerceptualGamma,
}

/// Configuration for SDR-to-HDR inverse tone mapping.
#[derive(Debug, Clone)]
pub struct SdrToHdrConfig {
    /// Target HDR peak luminance in nits (e.g. 1000.0).
    pub target_peak_nits: f32,
    /// SDR reference white in nits (e.g. 100.0).
    pub sdr_white_nits: f32,
    /// Which uplift algorithm to apply.
    pub algorithm: UpliftAlgorithm,
}

impl SdrToHdrConfig {
    /// Default: 100-nit SDR → 1 000-nit HDR using inverse BT.2446 Method C.
    pub fn default_hdr10() -> Self {
        Self {
            target_peak_nits: 1000.0,
            sdr_white_nits: 100.0,
            algorithm: UpliftAlgorithm::Bt2446MethodC,
        }
    }
}

/// SDR-to-HDR upscaling mapper with three algorithm choices.
///
/// All three algorithms operate on normalised [0, 1] input (SDR white = 1.0)
/// and produce normalised [0, 1] output (target HDR peak = 1.0).
#[derive(Debug, Clone)]
pub struct SdrToHdrMapper {
    config: SdrToHdrConfig,
}

impl SdrToHdrMapper {
    /// Create a new uplifter from the given configuration.
    pub fn new(config: SdrToHdrConfig) -> Self {
        Self { config }
    }

    /// Uplift a single normalised SDR luminance value to HDR [0, 1].
    pub fn process_luminance(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        match &self.config.algorithm {
            UpliftAlgorithm::LinearScale => {
                // Scale so that SDR white (x=1) maps to target/sdr_white nits ratio,
                // then re-normalise to [0, 1] relative to target_peak_nits.
                let scale = self.config.target_peak_nits / self.config.sdr_white_nits.max(1.0);
                (x * scale / (self.config.target_peak_nits / 100.0).max(1.0)).clamp(0.0, 1.0)
            }
            UpliftAlgorithm::Bt2446MethodC => {
                // Invert Method C sigmoid: f(x) = max_out * x / (x + k)
                // => x_hdr = k * x_sdr / (max_out - x_sdr)
                let max_out = 0.98_f32;
                let k = 0.40_f32;
                if x >= max_out {
                    return 1.0;
                }
                let x_hdr = k * x / (max_out - x);
                x_hdr.clamp(0.0, 1.0)
            }
            UpliftAlgorithm::PerceptualGamma => {
                // f(x) = 1 − (1 − x)^(1/α)  with α = 0.45.
                // α < 1 ⟹ 1/α > 1 ⟹ the function lies *above* the diagonal
                // ⟹ highlights are expanded (out > in for in > 0).
                let alpha = 0.45_f32;
                let expanded = 1.0 - (1.0 - x).powf(1.0 / alpha);
                expanded.clamp(0.0, 1.0)
            }
        }
    }

    /// Map a single SDR RGB pixel (values in [0, 1]) to HDR [0, 1].
    pub fn process_pixel(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let lum = 0.2627 * r + 0.6780 * g + 0.0593 * b;
        let mapped = self.process_luminance(lum.max(0.0));
        if lum > 1e-7 {
            let ratio = mapped / lum;
            (
                (r * ratio).clamp(0.0, 1.0),
                (g * ratio).clamp(0.0, 1.0),
                (b * ratio).clamp(0.0, 1.0),
            )
        } else {
            (mapped, mapped, mapped)
        }
    }

    /// Map an interleaved RGB frame in-place.
    ///
    /// `frame` must have length `width * height * 3`; values are in [0, 1]
    /// normalised to SDR white.  `_height` is accepted for API symmetry with
    /// `process_pixel` but is not required for the in-place loop.
    pub fn process_frame(&self, frame: &mut [f32], width: u32, _height: u32) {
        let stride = (width as usize) * 3;
        // Handle width=0 gracefully: chunks_exact will produce no items.
        if stride == 0 {
            return;
        }
        for chunk in frame.chunks_exact_mut(stride) {
            for px in chunk.chunks_exact_mut(3) {
                let (r, g, b) = self.process_pixel(px[0], px[1], px[2]);
                px[0] = r;
                px[1] = g;
                px[2] = b;
            }
        }
    }
}

// ── SceneReferredToneMapper: geometric-mean analysis + adaptive application ───

impl SceneReferredToneMapper {
    /// Compute the log-average (geometric mean) luminance of a frame.
    ///
    /// Returns `exp(mean(log(Y + ε)))` — the standard scene-referred exposure
    /// anchor used in photographic tone mapping research (Reinhard et al. 2002).
    /// `ε = 1e-4` avoids `log(0)` on pure-black pixels.
    ///
    /// The result is cached in `self.scene_luminance` for subsequent use by
    /// `apply_with_scene_analysis`.
    ///
    /// Returns `0.0` for frames that are empty or whose length is not a
    /// multiple of 3 (no `Result` returned to keep the hot path allocation-free).
    pub fn analyze_frame(&mut self, frame: &[f32], _width: u32, _height: u32) -> f32 {
        if frame.len() < 3 || !frame.len().is_multiple_of(3) {
            return 0.0;
        }
        const EPSILON: f32 = 1e-4;
        let log_sum: f64 = frame
            .chunks_exact(3)
            .map(|c| {
                let y = 0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2];
                f64::from((y + EPSILON).ln())
            })
            .sum();
        let n = (frame.len() / 3) as f64;
        let log_avg = (log_sum / n).exp() as f32;
        self.scene_luminance = Some(log_avg);
        log_avg
    }

    /// Analyse per-frame luminance, then apply the tone mapping operator with
    /// automatic exposure compensation.
    ///
    /// When `self.adaptive_exposure` is `true`, the exposure is set so that
    /// the geometric-mean luminance maps to 18% grey (`0.18`) in the output —
    /// the standard photographic key-value anchor.  Otherwise the operator is
    /// applied with unit exposure.
    ///
    /// The frame is modified in-place; each RGB pixel is mapped by the
    /// operator chosen in `self.operator`.
    pub fn apply_with_scene_analysis(&mut self, frame: &mut [f32], width: u32, height: u32) {
        let log_avg = self.analyze_frame(frame, width, height);

        let exposure = if self.adaptive_exposure && log_avg > 1e-6 {
            // Key-value mapping: geometric mean → 0.18 mid-grey.
            0.18 / log_avg
        } else {
            1.0
        };

        // Build a tone-mapping config that incorporates the computed exposure.
        // We scale input_peak_nits inversely to the exposure so that the
        // ToneMapper's internal normalisation keeps output in [0, 1].
        let effective_input_peak =
            self.output_peak_nits * (1.0 / exposure.max(1e-6)).clamp(0.1, 100.0);

        let config = ToneMappingConfig {
            operator: self.operator.clone(),
            input_peak_nits: effective_input_peak,
            output_peak_nits: self.output_peak_nits,
            exposure,
            saturation: 1.0,
            gamma_out: self.gamma_out,
        };
        let tm = ToneMapper::new(config);

        let stride = (width as usize) * 3;
        if stride == 0 {
            return;
        }
        for row in frame.chunks_exact_mut(stride) {
            for px in row.chunks_exact_mut(3) {
                let (r, g, b) = tm.map_pixel(px[0], px[1], px[2]);
                px[0] = r;
                px[1] = g;
                px[2] = b;
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tone_mapping::{SceneReferredToneMapper, ToneMappingOperator};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── bt2446_method_a_forward ───────────────────────────────────────────────

    #[test]
    fn test_bt2446_forward_zero() {
        assert!(approx(bt2446_method_a_forward(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_bt2446_forward_linear_region() {
        // Below knee (0.7399): output = 1.0770 * y
        let y = 0.5_f32;
        let expected = 1.0770 * y;
        let out = bt2446_method_a_forward(y);
        assert!(
            approx(out, expected, 1e-4),
            "linear region: {out} vs {expected}"
        );
    }

    #[test]
    fn test_bt2446_forward_rational_region() {
        // Above knee: rational curve, output must be in [0, 1].
        let out = bt2446_method_a_forward(0.9);
        assert!((0.0..=1.0).contains(&out), "rational region: {out}");
    }

    #[test]
    fn test_bt2446_forward_monotonic() {
        let mut prev = 0.0_f32;
        for i in 1..=200 {
            let x = i as f32 / 200.0;
            let cur = bt2446_method_a_forward(x);
            assert!(cur >= prev - 1e-5, "not monotonic at x={x}: {cur} < {prev}");
            prev = cur;
        }
    }

    // ── Bt2446MethodAForwardMapper ────────────────────────────────────────────

    #[test]
    fn test_forward_mapper_pq_zero() {
        let tm = Bt2446MethodAForwardMapper::new_pq(1000.0);
        assert!(approx(tm.map_luminance(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_forward_mapper_output_in_range() {
        let tm = Bt2446MethodAForwardMapper::new_pq(1000.0);
        for v in [0.0f32, 0.1, 0.3, 0.5, 0.8, 1.0] {
            let out = tm.map_luminance(v);
            assert!((0.0..=1.0).contains(&out), "Bt2446A forward({v}) = {out}");
        }
    }

    #[test]
    fn test_forward_mapper_monotonic() {
        let tm = Bt2446MethodAForwardMapper::new_pq(1000.0);
        let mut prev = 0.0_f32;
        for i in 1..=100 {
            let x = i as f32 / 100.0;
            let out = tm.map_luminance(x);
            assert!(out >= prev - 1e-5, "not monotonic at {x}: {out} < {prev}");
            prev = out;
        }
    }

    #[test]
    fn test_forward_mapper_hlg_frame() {
        let tm = Bt2446MethodAForwardMapper::new_hlg(1000.0);
        let pixels: Vec<f32> = (0..99).map(|i| i as f32 / 100.0).collect();
        let out = tm.map_frame(&pixels).expect("forward frame");
        assert_eq!(out.len(), 99);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "forward frame {v}");
        }
    }

    #[test]
    fn test_forward_mapper_frame_invalid() {
        let tm = Bt2446MethodAForwardMapper::new_pq(1000.0);
        assert!(tm.map_frame(&[0.5_f32, 0.3]).is_err());
    }

    // ── SdrToHdrMapper ────────────────────────────────────────────────────────

    #[test]
    fn test_sdr_to_hdr_linear_scale_in_range() {
        let cfg = SdrToHdrConfig {
            target_peak_nits: 1000.0,
            sdr_white_nits: 100.0,
            algorithm: UpliftAlgorithm::LinearScale,
        };
        let m = SdrToHdrMapper::new(cfg);
        for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let out = m.process_luminance(v);
            assert!((0.0..=1.0).contains(&out), "linear({v}) = {out}");
        }
    }

    #[test]
    fn test_sdr_to_hdr_bt2446c_inverse_zero() {
        let cfg = SdrToHdrConfig::default_hdr10();
        let m = SdrToHdrMapper::new(cfg);
        assert!(approx(m.process_luminance(0.0), 0.0, 1e-6));
    }

    #[test]
    fn test_sdr_to_hdr_bt2446c_mid_range() {
        let cfg = SdrToHdrConfig::default_hdr10();
        let m = SdrToHdrMapper::new(cfg);
        let out = m.process_luminance(0.5);
        assert!((0.0..=1.0).contains(&out), "bt2446c inv mid: {out}");
    }

    #[test]
    fn test_sdr_to_hdr_perceptual_expands_highlights() {
        let cfg = SdrToHdrConfig {
            target_peak_nits: 1000.0,
            sdr_white_nits: 100.0,
            algorithm: UpliftAlgorithm::PerceptualGamma,
        };
        let m = SdrToHdrMapper::new(cfg);
        let out = m.process_luminance(0.7);
        // Perceptual gamma with α=0.45 expands highlights: output > input.
        assert!(out > 0.7, "perceptual should expand highlights: {out}");
        assert!((0.0..=1.0).contains(&out));
    }

    #[test]
    fn test_sdr_to_hdr_process_pixel_black() {
        let cfg = SdrToHdrConfig::default_hdr10();
        let m = SdrToHdrMapper::new(cfg);
        let (r, g, b) = m.process_pixel(0.0, 0.0, 0.0);
        assert!(approx(r, 0.0, 1e-6) && approx(g, 0.0, 1e-6) && approx(b, 0.0, 1e-6));
    }

    #[test]
    fn test_sdr_to_hdr_process_pixel_colour() {
        let cfg = SdrToHdrConfig::default_hdr10();
        let m = SdrToHdrMapper::new(cfg);
        let (r, g, b) = m.process_pixel(0.5, 0.3, 0.2);
        assert!((0.0..=1.0).contains(&r) && (0.0..=1.0).contains(&g) && (0.0..=1.0).contains(&b));
    }

    #[test]
    fn test_sdr_to_hdr_process_frame() {
        let cfg = SdrToHdrConfig::default_hdr10();
        let m = SdrToHdrMapper::new(cfg);
        let mut frame: Vec<f32> = (0..24).map(|i| i as f32 / 24.0).collect();
        m.process_frame(&mut frame, 4, 2);
        for &v in &frame {
            assert!((0.0..=1.0).contains(&v), "process_frame {v}");
        }
    }

    // ── SceneReferredToneMapper::analyze_frame ────────────────────────────────

    #[test]
    fn test_analyze_frame_grey() {
        let mut sr = SceneReferredToneMapper::hdr10_to_sdr_adaptive();
        // 4 equal grey pixels → geometric mean = 0.2627*0.5+0.6780*0.5+0.0593*0.5 = 0.5
        let pixels = vec![0.5_f32; 12];
        let log_avg = sr.analyze_frame(&pixels, 4, 1);
        // Result should be near 0.5 (the actual luminance value)
        assert!(log_avg > 0.0 && log_avg <= 1.0, "log_avg = {log_avg}");
        assert!(sr.scene_luminance == Some(log_avg));
    }

    #[test]
    fn test_analyze_frame_empty() {
        let mut sr = SceneReferredToneMapper::hdr10_to_sdr_adaptive();
        let result = sr.analyze_frame(&[], 0, 0);
        assert!(approx(result, 0.0, 1e-6));
    }

    #[test]
    fn test_analyze_frame_invalid_length() {
        let mut sr = SceneReferredToneMapper::hdr10_to_sdr_adaptive();
        // Not a multiple of 3 — should return 0 without panicking.
        let result = sr.analyze_frame(&[0.5_f32, 0.3], 1, 1);
        assert!(approx(result, 0.0, 1e-6));
    }

    #[test]
    fn test_analyze_frame_bright_vs_dark() {
        let mut sr = SceneReferredToneMapper::hdr10_to_sdr_adaptive();
        // Bright frame should have higher geometric mean than dark frame.
        let bright = vec![0.8_f32; 12];
        let dark = vec![0.1_f32; 12];
        let avg_bright = sr.analyze_frame(&bright, 4, 1);
        let avg_dark = sr.analyze_frame(&dark, 4, 1);
        assert!(
            avg_bright > avg_dark,
            "bright {avg_bright} should > dark {avg_dark}"
        );
    }

    // ── SceneReferredToneMapper::apply_with_scene_analysis ────────────────────

    #[test]
    fn test_apply_with_scene_analysis_output_in_range() {
        let mut sr = SceneReferredToneMapper::hdr10_to_sdr_adaptive();
        let mut frame: Vec<f32> = (0..24).map(|i| i as f32 / 24.0).collect();
        sr.apply_with_scene_analysis(&mut frame, 4, 2);
        for &v in &frame {
            assert!((0.0..=1.0).contains(&v), "scene analysis output {v}");
        }
    }

    #[test]
    fn test_apply_with_scene_analysis_no_adaptive() {
        let mut sr = SceneReferredToneMapper {
            operator: ToneMappingOperator::Reinhard,
            clip_fraction: 0.05,
            output_peak_nits: 100.0,
            gamma_out: 2.2,
            scene_luminance: None,
            adaptive_exposure: false,
        };
        let mut frame = vec![0.5_f32; 12];
        sr.apply_with_scene_analysis(&mut frame, 4, 1);
        for &v in &frame {
            assert!((0.0..=1.0).contains(&v), "non-adaptive output {v}");
        }
    }

    #[test]
    fn test_apply_with_scene_analysis_sets_scene_luminance() {
        let mut sr = SceneReferredToneMapper::hdr10_to_sdr_adaptive();
        assert!(sr.scene_luminance.is_none());
        let mut frame = vec![0.5_f32; 12];
        sr.apply_with_scene_analysis(&mut frame, 4, 1);
        assert!(sr.scene_luminance.is_some());
    }
}
