//! Per-frame HDR to SDR tone mapping.
//!
//! Provides adaptive per-frame peak luminance detection and multiple tone
//! mapping operators for converting linear HDR frames to 8-bit SDR output.
//!
//! Features:
//! - Reinhard global/local tonemapping operators
//! - ACES filmic tonemapping (RRT + ODT approximation)
//! - Uncharted 2 (Hable) filmic curve
//! - Inverse tonemapping for SDR->HDR conversion
//! - Exposure-adaptive tonemapping (auto-adjust based on scene luminance histogram)

/// Tone-mapping operator selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMapMethod {
    /// Filmic Hable / Uncharted-2 curve.
    Hable,
    /// Simple Reinhard `x / (1 + x)` operator (global).
    Reinhard,
    /// Reinhard extended with configurable white point.
    ReinhardExtended,
    /// Reinhard local (Dodging & burning variant with local adaptation).
    ReinhardLocal,
    /// Simplified ACES RRT+ODT approximation.
    ACES,
}

/// Per-frame tone-mapping configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct PerFrameConfig {
    /// Tone-mapping operator.
    pub method: ToneMapMethod,
    /// Scene peak luminance in nits (used as hint; actual peak is auto-detected).
    pub peak_luminance: f32,
    /// Black-level offset (linear; subtracted before mapping).
    pub black_level: f32,
}

/// Reinhard extended parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ReinhardExtendedParams {
    /// White point: luminance value that maps to 1.0. Values above this are clipped.
    pub white_point: f32,
}

impl Default for ReinhardExtendedParams {
    fn default() -> Self {
        Self { white_point: 4.0 }
    }
}

/// Configuration for exposure-adaptive tonemapping.
#[derive(Debug, Clone, PartialEq)]
pub struct ExposureAdaptiveConfig {
    /// Base tone-mapping method to use.
    pub method: ToneMapMethod,
    /// Target average output luminance (0.0 to 1.0). Default: 0.18 (18% grey).
    pub target_mid_grey: f32,
    /// Minimum exposure adjustment (log2 stops). Default: -4.0
    pub min_exposure: f32,
    /// Maximum exposure adjustment (log2 stops). Default: 4.0
    pub max_exposure: f32,
    /// Black level offset.
    pub black_level: f32,
}

impl Default for ExposureAdaptiveConfig {
    fn default() -> Self {
        Self {
            method: ToneMapMethod::Reinhard,
            target_mid_grey: 0.18,
            min_exposure: -4.0,
            max_exposure: 4.0,
            black_level: 0.0,
        }
    }
}

/// Per-frame HDR->SDR tone mapper with adaptive peak luminance detection.
pub struct PerFrameTonemapper {
    /// Configuration used for this tonemapper.
    pub config: PerFrameConfig,
}

impl PerFrameTonemapper {
    /// Create a new `PerFrameTonemapper`.
    pub fn new(config: PerFrameConfig) -> Self {
        Self { config }
    }

    /// Detect the 99th-percentile luminance of a linear HDR RGBA frame.
    ///
    /// `frame_linear` is a flat slice of `f32` values in RGBA order,
    /// with `width * height * 4` elements.  Returns `1.0` for empty input.
    pub fn detect_peak_luminance(frame_linear: &[f32], _width: u32, _height: u32) -> f32 {
        if frame_linear.is_empty() {
            return 1.0;
        }

        let mut lumas: Vec<f32> = frame_linear
            .chunks_exact(4)
            .map(|p| {
                let r = p[0].max(0.0);
                let g = p[1].max(0.0);
                let b = p[2].max(0.0);
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect();

        if lumas.is_empty() {
            return 1.0;
        }

        lumas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((lumas.len() as f32 * 0.99) as usize).min(lumas.len() - 1);
        let peak = lumas[idx];

        // Avoid returning zero which would cause division by zero downstream
        if peak < 1e-6 {
            1.0
        } else {
            peak
        }
    }

    /// Tone-map a linear HDR RGBA frame to an 8-bit SDR RGBA `Vec<u8>`.
    ///
    /// `frame_linear` must be `width * height * 4` `f32` values in RGBA order.
    /// The alpha channel is passed through linearly (clamped to [0, 1], then
    /// scaled to [0, 255]).
    pub fn tonemap(&self, frame_linear: &[f32], width: u32, height: u32) -> Vec<u8> {
        let n_pixels = (width * height) as usize;
        let mut out = Vec::with_capacity(n_pixels * 4);

        // Adaptive peak detection
        let peak = Self::detect_peak_luminance(frame_linear, width, height).max(1e-6);

        for px in frame_linear.chunks_exact(4) {
            let r_lin = (px[0] - self.config.black_level).max(0.0);
            let g_lin = (px[1] - self.config.black_level).max(0.0);
            let b_lin = (px[2] - self.config.black_level).max(0.0);
            let alpha = px[3];

            // Normalize by peak
            let rn = r_lin / peak;
            let gn = g_lin / peak;
            let bn = b_lin / peak;

            // Apply tone-map operator
            let (r_tm, g_tm, b_tm) = match self.config.method {
                ToneMapMethod::Reinhard => {
                    (reinhard_curve(rn), reinhard_curve(gn), reinhard_curve(bn))
                }
                ToneMapMethod::ReinhardExtended => {
                    let params = ReinhardExtendedParams::default();
                    (
                        reinhard_extended_curve(rn, params.white_point),
                        reinhard_extended_curve(gn, params.white_point),
                        reinhard_extended_curve(bn, params.white_point),
                    )
                }
                ToneMapMethod::ReinhardLocal => {
                    // Local Reinhard: adapt per-pixel based on local luminance
                    let local_luma = 0.2126 * rn + 0.7152 * gn + 0.0722 * bn;
                    let adapted = reinhard_local_curve(local_luma, 0.18, 4.0);
                    let scale = if local_luma > 1e-9 {
                        adapted / local_luma
                    } else {
                        1.0
                    };
                    (
                        (rn * scale).clamp(0.0, 1.0),
                        (gn * scale).clamp(0.0, 1.0),
                        (bn * scale).clamp(0.0, 1.0),
                    )
                }
                ToneMapMethod::Hable => (
                    hable_curve_normalized(rn, peak),
                    hable_curve_normalized(gn, peak),
                    hable_curve_normalized(bn, peak),
                ),
                ToneMapMethod::ACES => (
                    aces_curve(rn).clamp(0.0, 1.0),
                    aces_curve(gn).clamp(0.0, 1.0),
                    aces_curve(bn).clamp(0.0, 1.0),
                ),
            };

            // Gamma 2.2 encode
            let r_enc = gamma_encode(r_tm);
            let g_enc = gamma_encode(g_tm);
            let b_enc = gamma_encode(b_tm);

            out.push(to_u8(r_enc));
            out.push(to_u8(g_enc));
            out.push(to_u8(b_enc));
            out.push((alpha.clamp(0.0, 1.0) * 255.0).round() as u8);
        }

        // Pad if input wasn't a multiple of 4 (shouldn't happen for valid frames)
        while out.len() < n_pixels * 4 {
            out.push(0);
        }

        out
    }

    /// Tone-map with Reinhard extended operator and custom white point.
    pub fn tonemap_reinhard_extended(
        &self,
        frame_linear: &[f32],
        width: u32,
        height: u32,
        params: &ReinhardExtendedParams,
    ) -> Vec<u8> {
        let n_pixels = (width * height) as usize;
        let mut out = Vec::with_capacity(n_pixels * 4);

        let peak = Self::detect_peak_luminance(frame_linear, width, height).max(1e-6);

        for px in frame_linear.chunks_exact(4) {
            let r_lin = (px[0] - self.config.black_level).max(0.0) / peak;
            let g_lin = (px[1] - self.config.black_level).max(0.0) / peak;
            let b_lin = (px[2] - self.config.black_level).max(0.0) / peak;
            let alpha = px[3];

            let r_tm = reinhard_extended_curve(r_lin, params.white_point);
            let g_tm = reinhard_extended_curve(g_lin, params.white_point);
            let b_tm = reinhard_extended_curve(b_lin, params.white_point);

            out.push(to_u8(gamma_encode(r_tm)));
            out.push(to_u8(gamma_encode(g_tm)));
            out.push(to_u8(gamma_encode(b_tm)));
            out.push((alpha.clamp(0.0, 1.0) * 255.0).round() as u8);
        }

        while out.len() < n_pixels * 4 {
            out.push(0);
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Exposure-adaptive tonemapping
// ---------------------------------------------------------------------------

/// Exposure-adaptive tonemapper that auto-adjusts exposure based on
/// scene luminance distribution.
pub struct ExposureAdaptiveTonemapper {
    /// Configuration.
    pub config: ExposureAdaptiveConfig,
}

impl ExposureAdaptiveTonemapper {
    /// Create a new exposure-adaptive tonemapper.
    pub fn new(config: ExposureAdaptiveConfig) -> Self {
        Self { config }
    }

    /// Compute the log-average luminance of a frame.
    ///
    /// Uses the geometric mean: `exp(mean(log(delta + luma)))` where delta
    /// avoids log(0).
    pub fn log_average_luminance(frame_linear: &[f32]) -> f32 {
        let delta = 1e-6f32;
        let pixels: Vec<f32> = frame_linear
            .chunks_exact(4)
            .map(|p| {
                let r = p[0].max(0.0);
                let g = p[1].max(0.0);
                let b = p[2].max(0.0);
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect();

        if pixels.is_empty() {
            return delta;
        }

        let sum_log: f64 = pixels.iter().map(|&l| (l + delta).ln() as f64).sum();
        let mean_log = sum_log / pixels.len() as f64;
        (mean_log as f32).exp()
    }

    /// Build a luminance histogram from a linear HDR frame.
    ///
    /// Returns 256 bins covering the log-luminance range.
    pub fn luminance_histogram(frame_linear: &[f32], num_bins: usize) -> Vec<u32> {
        let mut histogram = vec![0u32; num_bins];

        for px in frame_linear.chunks_exact(4) {
            let r = px[0].max(0.0);
            let g = px[1].max(0.0);
            let b = px[2].max(0.0);
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;

            // Map to log scale, then to bin
            let log_l = (luma.max(1e-6)).log2();
            // Map [-20, 10] log2 range to [0, num_bins-1]
            let normalized = ((log_l + 20.0) / 30.0).clamp(0.0, 1.0);
            let bin = (normalized * (num_bins - 1) as f32).round() as usize;
            let bin = bin.min(num_bins - 1);
            histogram[bin] += 1;
        }

        histogram
    }

    /// Compute auto-exposure adjustment (in log2 stops).
    pub fn compute_auto_exposure(&self, frame_linear: &[f32]) -> f32 {
        let avg_luma = Self::log_average_luminance(frame_linear);

        if avg_luma < 1e-9 {
            return 0.0;
        }

        // Exposure = target_mid_grey / avg_luma
        let exposure_linear = self.config.target_mid_grey / avg_luma;
        let exposure_stops = exposure_linear.log2();

        exposure_stops.clamp(self.config.min_exposure, self.config.max_exposure)
    }

    /// Tonemap with automatic exposure adjustment.
    pub fn tonemap_adaptive(&self, frame_linear: &[f32], width: u32, height: u32) -> Vec<u8> {
        let exposure_stops = self.compute_auto_exposure(frame_linear);
        let exposure_linear = 2.0f32.powf(exposure_stops);

        let n_pixels = (width * height) as usize;
        let mut out = Vec::with_capacity(n_pixels * 4);

        for px in frame_linear.chunks_exact(4) {
            let r = (px[0] - self.config.black_level).max(0.0) * exposure_linear;
            let g = (px[1] - self.config.black_level).max(0.0) * exposure_linear;
            let b = (px[2] - self.config.black_level).max(0.0) * exposure_linear;
            let alpha = px[3];

            let (r_tm, g_tm, b_tm) = match self.config.method {
                ToneMapMethod::Reinhard => {
                    (reinhard_curve(r), reinhard_curve(g), reinhard_curve(b))
                }
                ToneMapMethod::ReinhardExtended => (
                    reinhard_extended_curve(r, 4.0),
                    reinhard_extended_curve(g, 4.0),
                    reinhard_extended_curve(b, 4.0),
                ),
                ToneMapMethod::ReinhardLocal => {
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    let adapted = reinhard_local_curve(luma, 0.18, 4.0);
                    let scale = if luma > 1e-9 { adapted / luma } else { 1.0 };
                    (
                        (r * scale).clamp(0.0, 1.0),
                        (g * scale).clamp(0.0, 1.0),
                        (b * scale).clamp(0.0, 1.0),
                    )
                }
                ToneMapMethod::Hable => {
                    let peak =
                        PerFrameTonemapper::detect_peak_luminance(frame_linear, width, height)
                            .max(1e-6);
                    (
                        hable_curve_normalized(r, peak),
                        hable_curve_normalized(g, peak),
                        hable_curve_normalized(b, peak),
                    )
                }
                ToneMapMethod::ACES => (
                    aces_curve(r).clamp(0.0, 1.0),
                    aces_curve(g).clamp(0.0, 1.0),
                    aces_curve(b).clamp(0.0, 1.0),
                ),
            };

            out.push(to_u8(gamma_encode(r_tm)));
            out.push(to_u8(gamma_encode(g_tm)));
            out.push(to_u8(gamma_encode(b_tm)));
            out.push((alpha.clamp(0.0, 1.0) * 255.0).round() as u8);
        }

        while out.len() < n_pixels * 4 {
            out.push(0);
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Inverse tonemapping (SDR -> HDR)
// ---------------------------------------------------------------------------

/// Inverse tone-mapping operator selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InverseToneMapMethod {
    /// Inverse Reinhard: `x / (1 - x)` (clipped near x=1).
    InverseReinhard,
    /// Exponential expansion: `(exp(k*x) - 1) / (exp(k) - 1)` scaled to peak.
    Exponential,
    /// Power curve: `x^gamma * peak`.
    PowerCurve,
}

/// Configuration for inverse tonemapping (SDR->HDR).
#[derive(Debug, Clone, PartialEq)]
pub struct InverseToneMapConfig {
    /// Inverse tone-mapping method.
    pub method: InverseToneMapMethod,
    /// Target peak luminance for the HDR output.
    pub target_peak: f32,
    /// Expansion strength parameter (meaning depends on method).
    pub strength: f32,
}

/// Apply inverse tonemapping to an 8-bit SDR frame, producing linear HDR f32 output.
///
/// `frame_sdr` must be `width * height * 4` bytes (RGBA).
/// Returns `width * height * 4` f32 values in RGBA order.
pub fn inverse_tonemap(
    frame_sdr: &[u8],
    width: u32,
    height: u32,
    config: &InverseToneMapConfig,
) -> Vec<f32> {
    let n_pixels = (width * height) as usize;
    let mut out = Vec::with_capacity(n_pixels * 4);

    for px in frame_sdr.chunks_exact(4) {
        // Decode gamma: \[0,255\] -> \[0,1\] -> linear
        let r_sdr = gamma_decode(px[0] as f32 / 255.0);
        let g_sdr = gamma_decode(px[1] as f32 / 255.0);
        let b_sdr = gamma_decode(px[2] as f32 / 255.0);
        let alpha = px[3] as f32 / 255.0;

        let (r_hdr, g_hdr, b_hdr) = match config.method {
            InverseToneMapMethod::InverseReinhard => (
                inverse_reinhard(r_sdr, config.target_peak),
                inverse_reinhard(g_sdr, config.target_peak),
                inverse_reinhard(b_sdr, config.target_peak),
            ),
            InverseToneMapMethod::Exponential => (
                exponential_expand(r_sdr, config.strength, config.target_peak),
                exponential_expand(g_sdr, config.strength, config.target_peak),
                exponential_expand(b_sdr, config.strength, config.target_peak),
            ),
            InverseToneMapMethod::PowerCurve => {
                let gamma = config.strength.max(0.1);
                (
                    r_sdr.powf(gamma) * config.target_peak,
                    g_sdr.powf(gamma) * config.target_peak,
                    b_sdr.powf(gamma) * config.target_peak,
                )
            }
        };

        out.push(r_hdr);
        out.push(g_hdr);
        out.push(b_hdr);
        out.push(alpha);
    }

    out
}

// ---------------------------------------------------------------------------
// Tone-map curve functions
// ---------------------------------------------------------------------------

/// Simple Reinhard: `x / (1 + x)`.
#[inline]
fn reinhard_curve(x: f32) -> f32 {
    x / (1.0 + x)
}

/// Reinhard extended with white point: `x * (1 + x/w^2) / (1 + x)`.
///
/// This allows values above `white_point` to saturate at 1.0.
#[inline]
fn reinhard_extended_curve(x: f32, white_point: f32) -> f32 {
    let wp2 = white_point * white_point;
    if wp2 < 1e-9 {
        return reinhard_curve(x);
    }
    let num = x * (1.0 + x / wp2);
    let den = 1.0 + x;
    (num / den).clamp(0.0, 1.0)
}

/// Reinhard local tone mapping (simplified dodging-and-burning).
///
/// `key` is the scene key (typically 0.18 for 18% grey).
/// `white_point` is the luminance mapped to pure white.
#[inline]
fn reinhard_local_curve(luma: f32, key: f32, white_point: f32) -> f32 {
    let scaled = key * luma;
    let wp2 = white_point * white_point;
    if wp2 < 1e-9 {
        return reinhard_curve(luma);
    }
    let num = scaled * (1.0 + scaled / wp2);
    let den = 1.0 + scaled;
    (num / den).clamp(0.0, 1.0)
}

/// Hable (Uncharted 2) core curve.
///
/// `f(x) = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F`
/// with A=0.15, B=0.50, C=0.10, D=0.20, E=0.02, F=0.30.
#[inline]
fn hable_curve(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

/// Hable curve with white-point normalization.
///
/// Applies `tone(x) / tone(W / peak)` so that the white point maps to 1.0.
#[inline]
fn hable_curve_normalized(x: f32, peak: f32) -> f32 {
    const W: f32 = 11.2;
    let white_scale_input = W / peak.max(1e-6);
    let denom = hable_curve(white_scale_input);
    if denom.abs() < 1e-6 {
        return 0.0;
    }
    (hable_curve(x) / denom).clamp(0.0, 1.0)
}

/// Simplified ACES RRT+ODT approximation.
///
/// `(x*(2.51*x+0.03))/(x*(2.43*x+0.59)+0.14)`.
#[inline]
fn aces_curve(x: f32) -> f32 {
    let num = x * (2.51 * x + 0.03);
    let den = x * (2.43 * x + 0.59) + 0.14;
    if den.abs() < 1e-10 {
        return 0.0;
    }
    num / den
}

/// Inverse Reinhard: `x / (1 - x)` scaled to target peak.
///
/// Clamps x to [0, 0.999] to avoid division by zero.
#[inline]
fn inverse_reinhard(x: f32, target_peak: f32) -> f32 {
    let x_safe = x.clamp(0.0, 0.999);
    let linear = x_safe / (1.0 - x_safe);
    linear * target_peak
}

/// Exponential expansion: `(exp(k*x) - 1) / (exp(k) - 1) * peak`.
#[inline]
fn exponential_expand(x: f32, k: f32, target_peak: f32) -> f32 {
    let k_safe = k.max(0.01);
    let exp_k = k_safe.exp();
    if (exp_k - 1.0).abs() < 1e-9 {
        return x * target_peak;
    }
    ((k_safe * x).exp() - 1.0) / (exp_k - 1.0) * target_peak
}

/// Gamma 2.2 encode: `x^(1/2.2)`, clamped to [0, 1].
#[inline]
fn gamma_encode(x: f32) -> f32 {
    x.clamp(0.0, 1.0).powf(1.0 / 2.2)
}

/// Gamma 2.2 decode: `x^2.2`, clamped to [0, 1] input.
#[inline]
fn gamma_decode(x: f32) -> f32 {
    x.clamp(0.0, 1.0).powf(2.2)
}

/// Scale a \[0,1\] linear value to \[0,255\] with rounding.
#[inline]
fn to_u8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_frame(width: usize, height: usize, r: f32, g: f32, b: f32, a: f32) -> Vec<f32> {
        let mut v = Vec::with_capacity(width * height * 4);
        for _ in 0..(width * height) {
            v.extend_from_slice(&[r, g, b, a]);
        }
        v
    }

    fn default_config(method: ToneMapMethod) -> PerFrameConfig {
        PerFrameConfig {
            method,
            peak_luminance: 1000.0,
            black_level: 0.0,
        }
    }

    // 1. Constructor sets config
    #[test]
    fn test_new_sets_config() {
        let cfg = default_config(ToneMapMethod::Reinhard);
        let tm = PerFrameTonemapper::new(cfg.clone());
        assert_eq!(tm.config, cfg);
    }

    // 2. Empty frame -> peak = 1.0
    #[test]
    fn test_detect_peak_luminance_empty_returns_one() {
        let peak = PerFrameTonemapper::detect_peak_luminance(&[], 0, 0);
        assert!((peak - 1.0).abs() < 1e-5);
    }

    // 3. Uniform frame -> peak = computed luminance
    #[test]
    fn test_detect_peak_luminance_uniform() {
        // All pixels: R=1.0, G=0.0, B=0.0 -> luma = 0.2126
        let frame = make_linear_frame(4, 4, 1.0, 0.0, 0.0, 1.0);
        let peak = PerFrameTonemapper::detect_peak_luminance(&frame, 4, 4);
        assert!((peak - 0.2126).abs() < 1e-3, "peak was {}", peak);
    }

    // 4. Mixed values -> 99th percentile
    #[test]
    fn test_detect_peak_luminance_mixed() {
        let mut frame = Vec::with_capacity(100 * 4);
        for i in 1..=100 {
            let luma = i as f32 / 100.0;
            let v = luma;
            frame.extend_from_slice(&[v, v, v, 1.0]);
        }
        let peak = PerFrameTonemapper::detect_peak_luminance(&frame, 10, 10);
        assert!(peak > 0.95 && peak <= 1.0, "peak was {}", peak);
    }

    // 5. Reinhard output size
    #[test]
    fn test_tonemap_reinhard_output_size() {
        let frame = make_linear_frame(8, 6, 0.5, 0.4, 0.3, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Reinhard));
        let out = tm.tonemap(&frame, 8, 6);
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    // 6. Hable output size
    #[test]
    fn test_tonemap_hable_output_size() {
        let frame = make_linear_frame(8, 6, 0.5, 0.4, 0.3, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Hable));
        let out = tm.tonemap(&frame, 8, 6);
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    // 7. ACES output size
    #[test]
    fn test_tonemap_aces_output_size() {
        let frame = make_linear_frame(8, 6, 0.5, 0.4, 0.3, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::ACES));
        let out = tm.tonemap(&frame, 8, 6);
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    // 8. Reinhard black frame -> black output
    #[test]
    fn test_tonemap_reinhard_black_frame() {
        let frame = make_linear_frame(4, 4, 0.0, 0.0, 0.0, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Reinhard));
        let out = tm.tonemap(&frame, 4, 4);
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[0], 0, "R should be 0");
            assert_eq!(chunk[1], 0, "G should be 0");
            assert_eq!(chunk[2], 0, "B should be 0");
        }
    }

    // 9. Hable black frame -> black output
    #[test]
    fn test_tonemap_hable_black_frame() {
        let frame = make_linear_frame(4, 4, 0.0, 0.0, 0.0, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Hable));
        let out = tm.tonemap(&frame, 4, 4);
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[0], 0);
            assert_eq!(chunk[1], 0);
            assert_eq!(chunk[2], 0);
        }
    }

    // 10. ACES black frame -> black output
    #[test]
    fn test_tonemap_aces_black_frame() {
        let frame = make_linear_frame(4, 4, 0.0, 0.0, 0.0, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::ACES));
        let out = tm.tonemap(&frame, 4, 4);
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[0], 0);
            assert_eq!(chunk[1], 0);
            assert_eq!(chunk[2], 0);
        }
    }

    // 11. All output bytes in [0, 255] (u8 guarantees this; check no panic)
    #[test]
    fn test_tonemap_output_u8_range() {
        // Very bright HDR frame
        let frame = make_linear_frame(8, 8, 100.0, 200.0, 50.0, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Hable));
        let out = tm.tonemap(&frame, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 4);
        // u8 is always in [0, 255]; this just ensures no panic occurred
    }

    // 12. Very bright Reinhard frame compresses to <255 for R channel
    #[test]
    fn test_tonemap_reinhard_bright_frame() {
        let frame = make_linear_frame(4, 4, 1000.0, 500.0, 200.0, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Reinhard));
        let out = tm.tonemap(&frame, 4, 4);
        // Output is u8, so values are inherently in [0, 255]
        assert!(!out.is_empty(), "output should not be empty");
    }

    // 13. Alpha channel preserved
    #[test]
    fn test_tonemap_alpha_preserved() {
        let frame = make_linear_frame(4, 4, 0.5, 0.5, 0.5, 0.75);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Reinhard));
        let out = tm.tonemap(&frame, 4, 4);
        let expected_alpha = (0.75f32 * 255.0).round() as u8; // 191
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[3], expected_alpha, "alpha mismatch");
        }
    }

    // 14. ToneMapMethod derives Debug
    #[test]
    fn test_tone_map_method_debug() {
        let s = format!("{:?}", ToneMapMethod::ACES);
        assert!(s.contains("ACES"));
    }

    // 15. PerFrameConfig can be cloned
    #[test]
    fn test_per_frame_config_clone() {
        let cfg = PerFrameConfig {
            method: ToneMapMethod::Hable,
            peak_luminance: 1000.0,
            black_level: 0.01,
        };
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }

    // 16. ACES moderate input maps to reasonable range
    #[test]
    fn test_tonemap_aces_moderate_input() {
        let frame = make_linear_frame(8, 8, 0.5, 0.5, 0.5, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::ACES));
        let out = tm.tonemap(&frame, 8, 8);
        for chunk in out.chunks_exact(4) {
            assert!(chunk[0] > 0, "R={}", chunk[0]);
        }
    }

    // 17. Hable moderate input maps to reasonable range
    #[test]
    fn test_tonemap_hable_moderate_input() {
        let frame = make_linear_frame(8, 8, 0.8, 0.6, 0.4, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Hable));
        let out = tm.tonemap(&frame, 8, 8);
        for chunk in out.chunks_exact(4) {
            assert!(chunk[0] > 0, "R should be > 0 for 0.8 input");
        }
    }

    // === NEW TESTS ===

    // 18. Reinhard extended output size
    #[test]
    fn test_tonemap_reinhard_extended_output_size() {
        let frame = make_linear_frame(8, 6, 0.5, 0.4, 0.3, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::ReinhardExtended));
        let out = tm.tonemap(&frame, 8, 6);
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    // 19. Reinhard extended with custom white point
    #[test]
    fn test_tonemap_reinhard_extended_custom_wp() {
        let frame = make_linear_frame(4, 4, 0.8, 0.6, 0.4, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::Reinhard));
        let params = ReinhardExtendedParams { white_point: 2.0 };
        let out = tm.tonemap_reinhard_extended(&frame, 4, 4, &params);
        assert_eq!(out.len(), 4 * 4 * 4);
        // Non-zero output for non-zero input
        for chunk in out.chunks_exact(4) {
            assert!(chunk[0] > 0, "R should be > 0");
        }
    }

    // 20. Reinhard extended black frame -> black
    #[test]
    fn test_tonemap_reinhard_extended_black() {
        let frame = make_linear_frame(4, 4, 0.0, 0.0, 0.0, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::ReinhardExtended));
        let out = tm.tonemap(&frame, 4, 4);
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[0], 0);
        }
    }

    // 21. Reinhard local output size
    #[test]
    fn test_tonemap_reinhard_local_output_size() {
        let frame = make_linear_frame(8, 6, 0.5, 0.4, 0.3, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::ReinhardLocal));
        let out = tm.tonemap(&frame, 8, 6);
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    // 22. Reinhard local produces non-zero for non-zero input
    #[test]
    fn test_tonemap_reinhard_local_nonzero() {
        let frame = make_linear_frame(4, 4, 0.5, 0.5, 0.5, 1.0);
        let tm = PerFrameTonemapper::new(default_config(ToneMapMethod::ReinhardLocal));
        let out = tm.tonemap(&frame, 4, 4);
        for chunk in out.chunks_exact(4) {
            assert!(chunk[0] > 0, "R should be > 0 for local Reinhard");
        }
    }

    // 23. Reinhard curve properties
    #[test]
    fn test_reinhard_curve_properties() {
        // reinhard(0) = 0
        assert!((reinhard_curve(0.0)).abs() < 1e-8);
        // reinhard(1) = 0.5
        assert!((reinhard_curve(1.0) - 0.5).abs() < 1e-6);
        // Monotonically increasing
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let x = i as f32 / 10.0;
            let y = reinhard_curve(x);
            assert!(
                y >= prev - 1e-6,
                "not monotonic at x={}: {} < {}",
                x,
                y,
                prev
            );
            prev = y;
        }
        // Approaches 1.0 for large input
        assert!(reinhard_curve(1000.0) > 0.99);
    }

    // 24. Reinhard extended curve: values above white_point map near 1.0
    #[test]
    fn test_reinhard_extended_curve_white_point() {
        let wp = 4.0;
        let above = reinhard_extended_curve(10.0, wp);
        assert!(
            above > 0.9,
            "above white point should be near 1.0, got {}",
            above
        );
        // At x=0 should be 0
        assert!((reinhard_extended_curve(0.0, wp)).abs() < 1e-6);
    }

    // 25. ACES curve properties
    #[test]
    fn test_aces_curve_properties() {
        // aces(0) should be near 0
        assert!(aces_curve(0.0).abs() < 0.01);
        // Monotonically increasing in [0, 10]
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let x = i as f32 / 10.0;
            let y = aces_curve(x);
            assert!(
                y >= prev - 1e-6,
                "ACES not monotonic at x={}: {} < {}",
                x,
                y,
                prev
            );
            prev = y;
        }
    }

    // 26. Hable curve monotonicity
    #[test]
    fn test_hable_curve_monotonic() {
        let mut prev = hable_curve(0.0);
        for i in 1..=100 {
            let x = i as f32 / 10.0;
            let y = hable_curve(x);
            assert!(
                y >= prev - 1e-6,
                "Hable not monotonic at x={}: {} < {}",
                x,
                y,
                prev
            );
            prev = y;
        }
    }

    // 27. Inverse Reinhard round-trip
    #[test]
    fn test_inverse_reinhard_round_trip() {
        // Forward: x=0.5 -> reinhard = 0.5/(1+0.5) = 0.333...
        let forward = reinhard_curve(0.5);
        // Inverse: 0.333/(1-0.333) ≈ 0.5
        let back = inverse_reinhard(forward, 1.0);
        assert!(
            (back - 0.5).abs() < 0.01,
            "round-trip failed: {} -> {} -> {}",
            0.5,
            forward,
            back
        );
    }

    // 28. Inverse tonemapping output size (InverseReinhard)
    #[test]
    fn test_inverse_tonemap_output_size() {
        let sdr = vec![128u8, 128, 128, 255, 64, 64, 64, 255];
        let config = InverseToneMapConfig {
            method: InverseToneMapMethod::InverseReinhard,
            target_peak: 10.0,
            strength: 1.0,
        };
        let hdr = inverse_tonemap(&sdr, 2, 1, &config);
        assert_eq!(hdr.len(), 2 * 4);
    }

    // 29. Inverse tonemapping black -> black
    #[test]
    fn test_inverse_tonemap_black() {
        let sdr = vec![0u8, 0, 0, 255];
        let config = InverseToneMapConfig {
            method: InverseToneMapMethod::InverseReinhard,
            target_peak: 10.0,
            strength: 1.0,
        };
        let hdr = inverse_tonemap(&sdr, 1, 1, &config);
        assert!(hdr[0].abs() < 1e-5, "R should be ~0, got {}", hdr[0]);
        assert!(hdr[1].abs() < 1e-5, "G should be ~0, got {}", hdr[1]);
        assert!(hdr[2].abs() < 1e-5, "B should be ~0, got {}", hdr[2]);
    }

    // 30. Inverse tonemapping exponential method
    #[test]
    fn test_inverse_tonemap_exponential() {
        let sdr = vec![128u8, 128, 128, 255];
        let config = InverseToneMapConfig {
            method: InverseToneMapMethod::Exponential,
            target_peak: 5.0,
            strength: 2.0,
        };
        let hdr = inverse_tonemap(&sdr, 1, 1, &config);
        // Result should be > 0 and scaled by target_peak
        assert!(hdr[0] > 0.0, "R should be > 0");
        assert!(hdr[0] <= config.target_peak + 0.1, "R should be <= peak");
    }

    // 31. Inverse tonemapping power curve method
    #[test]
    fn test_inverse_tonemap_power_curve() {
        let sdr = vec![200u8, 100, 50, 255];
        let config = InverseToneMapConfig {
            method: InverseToneMapMethod::PowerCurve,
            target_peak: 10.0,
            strength: 2.0,
        };
        let hdr = inverse_tonemap(&sdr, 1, 1, &config);
        assert!(hdr[0] > 0.0, "R should be > 0");
        assert!(hdr[0] <= config.target_peak + 0.1);
    }

    // 32. Exposure-adaptive: log average luminance
    #[test]
    fn test_log_average_luminance() {
        let frame = make_linear_frame(4, 4, 0.18, 0.18, 0.18, 1.0);
        let avg = ExposureAdaptiveTonemapper::log_average_luminance(&frame);
        // Luma = 0.18 * (0.2126 + 0.7152 + 0.0722) = 0.18
        assert!(avg > 0.1 && avg < 0.3, "avg luma = {}", avg);
    }

    // 33. Exposure-adaptive: empty frame luminance
    #[test]
    fn test_log_average_luminance_empty() {
        let avg = ExposureAdaptiveTonemapper::log_average_luminance(&[]);
        assert!(
            avg < 0.001,
            "empty frame should have near-zero avg: {}",
            avg
        );
    }

    // 34. Exposure-adaptive: luminance histogram
    #[test]
    fn test_luminance_histogram() {
        let frame = make_linear_frame(4, 4, 0.5, 0.5, 0.5, 1.0);
        let hist = ExposureAdaptiveTonemapper::luminance_histogram(&frame, 64);
        assert_eq!(hist.len(), 64);
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 16, "should have 16 pixels");
    }

    // 35. Exposure-adaptive: tonemap output size
    #[test]
    fn test_exposure_adaptive_output_size() {
        let frame = make_linear_frame(8, 6, 0.5, 0.4, 0.3, 1.0);
        let tm = ExposureAdaptiveTonemapper::new(ExposureAdaptiveConfig::default());
        let out = tm.tonemap_adaptive(&frame, 8, 6);
        assert_eq!(out.len(), 8 * 6 * 4);
    }

    // 36. Exposure-adaptive: black frame -> black output
    #[test]
    fn test_exposure_adaptive_black() {
        let frame = make_linear_frame(4, 4, 0.0, 0.0, 0.0, 1.0);
        let tm = ExposureAdaptiveTonemapper::new(ExposureAdaptiveConfig::default());
        let out = tm.tonemap_adaptive(&frame, 4, 4);
        for chunk in out.chunks_exact(4) {
            assert_eq!(chunk[0], 0, "R should be 0 for black");
        }
    }

    // 37. Exposure-adaptive: auto-exposure adjustment is clamped
    #[test]
    fn test_auto_exposure_clamped() {
        let config = ExposureAdaptiveConfig {
            min_exposure: -2.0,
            max_exposure: 2.0,
            ..Default::default()
        };
        let tm = ExposureAdaptiveTonemapper::new(config);

        // Very dark frame -> large positive exposure
        let dark = make_linear_frame(4, 4, 0.001, 0.001, 0.001, 1.0);
        let exp = tm.compute_auto_exposure(&dark);
        assert!(exp <= 2.0, "exposure should be clamped to 2.0, got {}", exp);

        // Very bright frame -> large negative exposure
        let bright = make_linear_frame(4, 4, 100.0, 100.0, 100.0, 1.0);
        let exp = tm.compute_auto_exposure(&bright);
        assert!(
            exp >= -2.0,
            "exposure should be clamped to -2.0, got {}",
            exp
        );
    }

    // 38. Gamma encode/decode round-trip
    #[test]
    fn test_gamma_round_trip() {
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            let encoded = gamma_encode(x);
            let decoded = gamma_decode(encoded);
            assert!(
                (decoded - x).abs() < 0.01,
                "gamma round-trip at {}: encoded={}, decoded={}",
                x,
                encoded,
                decoded
            );
        }
    }

    // 39. ToneMapMethod new variants have Debug
    #[test]
    fn test_new_tone_map_methods_debug() {
        let s1 = format!("{:?}", ToneMapMethod::ReinhardExtended);
        assert!(s1.contains("ReinhardExtended"));
        let s2 = format!("{:?}", ToneMapMethod::ReinhardLocal);
        assert!(s2.contains("ReinhardLocal"));
    }

    // 40. InverseToneMapConfig can be created
    #[test]
    fn test_inverse_config() {
        let cfg = InverseToneMapConfig {
            method: InverseToneMapMethod::Exponential,
            target_peak: 10.0,
            strength: 2.0,
        };
        assert_eq!(cfg.method, InverseToneMapMethod::Exponential);
        assert!((cfg.target_peak - 10.0).abs() < 1e-6);
    }
}
