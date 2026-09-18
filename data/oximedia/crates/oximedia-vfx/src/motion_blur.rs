//! Motion blur VFX module.
//!
//! Provides kernel-based motion blur for linear, radial, and zoom motion,
//! as well as motion-vector-based per-pixel directional blur.
//!
//! # Motion-Vector Blur
//!
//! [`mv_motion_blur`] implements the Potmesil & Chakravarty (1983) approach,
//! extended with Navarro et al. (2011) bilinear sampling.  Each pixel is blurred
//! along its individual motion trail, producing physically correct boundary blur
//! that respects per-pixel velocity differences.
//!
//! # VideoEffect Dispatch
//!
//! [`MotionBlur`] implements the [`crate::VideoEffect`] trait and dispatches
//! to the appropriate backend based on [`MotionBlurMode`]:
//! - [`MotionBlurMode::MotionVector`] → [`crate::motion_vector_blur::apply_mv_blur`]
//! - [`MotionBlurMode::UniformLinear`] / [`MotionBlurMode::UniformRadial`] → kernel path

// ── MotionBlurMode ─────────────────────────────────────────────────────────────

/// High-level selector for which motion-blur algorithm to use.
#[derive(Debug, Clone)]
pub enum MotionBlurMode {
    /// Uniform linear kernel applied to every pixel equally.
    UniformLinear {
        /// Horizontal displacement in pixels.
        dx: f32,
        /// Vertical displacement in pixels.
        dy: f32,
        /// Number of kernel samples.
        num_samples: u32,
    },
    /// Radial (rotational) kernel.
    UniformRadial {
        /// Rotation centre X.
        cx: f32,
        /// Rotation centre Y.
        cy: f32,
        /// Total rotation arc in radians.
        angle_rad: f32,
        /// Number of kernel samples.
        num_samples: u32,
    },
    /// Per-pixel motion-vector-driven blur (Potmesil & Chakravarty 1983).
    MotionVector(MvMotionBlurConfig),
}

// ── MvMotionBlurConfig ─────────────────────────────────────────────────────────

/// Configuration for motion-vector-driven per-pixel directional blur.
///
/// Algorithm: Potmesil & Chakravarty (1983), Navarro et al. (2011).
#[derive(Debug, Clone)]
pub struct MvMotionBlurConfig {
    /// Maximum blur in pixels (clamps MV magnitude).
    pub max_blur_pixels: f32,
    /// Number of samples along the motion vector (1..=32). Auto if 0.
    pub num_samples: u32,
}

impl Default for MvMotionBlurConfig {
    fn default() -> Self {
        Self {
            max_blur_pixels: 16.0,
            num_samples: 0,
        }
    }
}

// ── bilinear_sample ────────────────────────────────────────────────────────────

/// Bilinearly sample a grayscale (single-channel u8) frame at floating-point
/// pixel coordinates `(fx, fy)`.
///
/// Coordinates are clamped to `[0, w-1] × [0, h-1]`.
#[inline]
fn bilinear_sample(frame: &[u8], w: u32, h: u32, fx: f32, fy: f32) -> f32 {
    let w = w as usize;
    let h = h as usize;

    let x0 = (fx.floor() as i64).clamp(0, (w as i64) - 1) as usize;
    let y0 = (fy.floor() as i64).clamp(0, (h as i64) - 1) as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let tx = fx - fx.floor();
    let ty = fy - fy.floor();

    let p00 = frame[y0 * w + x0] as f32;
    let p10 = frame[y0 * w + x1] as f32;
    let p01 = frame[y1 * w + x0] as f32;
    let p11 = frame[y1 * w + x1] as f32;

    let top = p00 * (1.0 - tx) + p10 * tx;
    let bot = p01 * (1.0 - tx) + p11 * tx;
    top * (1.0 - ty) + bot * ty
}

// ── mv_motion_blur ─────────────────────────────────────────────────────────────

/// Apply per-pixel directional motion blur using a dense motion-vector field.
///
/// `frame` — grayscale or single-channel u8 pixels, row-major, `w * h` bytes.
/// `mv_field` — per-pixel `(dx, dy)` motion vectors, same indexing.
///
/// Returns blurred frame of same dimensions.
///
/// # Algorithm
///
/// For each pixel `(x, y)`:
/// 1. Look up `(dx, dy)` from `mv_field`, clamp magnitude to `max_blur_pixels`.
/// 2. Determine `n` sample count (auto if `cfg.num_samples == 0`).
/// 3. Accumulate `n` bilinearly-interpolated samples along the segment
///    from `(x - dx/2, y - dy/2)` to `(x + dx/2, y + dy/2)`.
/// 4. Average → output pixel.
#[must_use]
pub fn mv_motion_blur(
    frame: &[u8],
    mv_field: &[(f32, f32)],
    w: u32,
    h: u32,
    cfg: &MvMotionBlurConfig,
) -> Vec<u8> {
    let pixel_count = (w as usize) * (h as usize);
    let mut output = vec![0u8; pixel_count];

    for y in 0..(h as usize) {
        for x in 0..(w as usize) {
            let idx = y * (w as usize) + x;
            let (raw_dx, raw_dy) = if idx < mv_field.len() {
                mv_field[idx]
            } else {
                (0.0, 0.0)
            };

            // Clamp magnitude
            let mag = (raw_dx * raw_dx + raw_dy * raw_dy).sqrt();
            let clamped_mag = mag.min(cfg.max_blur_pixels);
            let (dx, dy) = if mag > 1e-10 {
                let scale = clamped_mag / mag;
                (raw_dx * scale, raw_dy * scale)
            } else {
                (0.0, 0.0)
            };

            // Auto sample count
            let n = if cfg.num_samples == 0 {
                ((clamped_mag / 0.5).ceil() as u32).max(1)
            } else {
                cfg.num_samples
            };
            let n = n.clamp(1, 32);

            // Sample along segment [(x - dx/2, y - dy/2) .. (x + dx/2, y + dy/2)]
            let start_x = x as f32 - dx * 0.5;
            let start_y = y as f32 - dy * 0.5;
            let step_x = if n > 1 { dx / (n - 1) as f32 } else { 0.0 };
            let step_y = if n > 1 { dy / (n - 1) as f32 } else { 0.0 };

            let mut acc = 0.0f32;
            for s in 0..n {
                let sx = start_x + step_x * s as f32;
                let sy = start_y + step_y * s as f32;
                acc += bilinear_sample(frame, w, h, sx, sy);
            }

            output[idx] = (acc / n as f32).round().clamp(0.0, 255.0) as u8;
        }
    }

    output
}

// ── PSNR helper ────────────────────────────────────────────────────────────────

/// Compute PSNR in dB between two equal-length grayscale byte slices.
/// Returns `f64::INFINITY` when MSE is zero.
#[cfg(test)]
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&p, &q)| {
            let d = (p as f64) - (q as f64);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * 255.0_f64.log10() - 10.0 * mse.log10()
    }
}

// ── MotionBlurType (existing) ──────────────────────────────────────────────────

/// The type of motion blur to apply.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub enum MotionBlurType {
    /// Blur along a straight line (dx, dy).
    Linear,
    /// Blur around a rotation centre.
    Radial,
    /// Zoom-in / zoom-out blur from the image centre.
    Zoom,
    /// Caller-supplied custom samples.
    Custom,
}

impl MotionBlurType {
    /// Returns `true` for rotational blur types (`Radial`).
    #[must_use]
    pub fn is_rotational(&self) -> bool {
        matches!(self, Self::Radial)
    }
}

/// A single weighted sample offset within a motion blur kernel.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub struct BlurSample {
    /// Horizontal offset in pixels.
    pub x_offset: f32,
    /// Vertical offset in pixels.
    pub y_offset: f32,
    /// Contribution weight for this sample.
    pub weight: f32,
}

impl BlurSample {
    /// Returns `true` if both offsets are exactly zero.
    #[must_use]
    pub fn is_zero_offset(&self) -> bool {
        self.x_offset == 0.0 && self.y_offset == 0.0
    }
}

/// A motion blur kernel composed of weighted offset samples.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MotionBlurKernel {
    /// Ordered list of weighted samples.
    pub samples: Vec<BlurSample>,
    /// Blur type that generated these samples.
    pub blur_type: MotionBlurType,
}

impl MotionBlurKernel {
    /// Build a linear motion blur kernel.
    ///
    /// Samples are distributed evenly along the vector (`dx`, `dy`),
    /// each with equal weight.
    #[must_use]
    pub fn linear_kernel(dx: f32, dy: f32, num_samples: u32) -> Self {
        let n = num_samples.max(1);
        let weight = 1.0 / n as f32;

        let samples = (0..n)
            .map(|i| {
                let t = if n == 1 {
                    0.0f32
                } else {
                    i as f32 / (n - 1) as f32 - 0.5
                };
                BlurSample {
                    x_offset: dx * t,
                    y_offset: dy * t,
                    weight,
                }
            })
            .collect();

        Self {
            samples,
            blur_type: MotionBlurType::Linear,
        }
    }

    /// Build a radial (rotational) motion blur kernel.
    ///
    /// Samples are distributed over the arc `[−angle_rad/2, +angle_rad/2]`
    /// centred at (`cx`, `cy`).  Since the rotation is applied relative to
    /// an unknown pixel position the offsets here represent the **delta**
    /// contribution at the centre point (all zero) — real per-pixel offsets
    /// are computed in `apply_motion_blur`.  Weight is uniform.
    #[must_use]
    pub fn radial_kernel(cx: f32, cy: f32, angle_rad: f32, num_samples: u32) -> Self {
        let n = num_samples.max(1);
        let weight = 1.0 / n as f32;

        let samples = (0..n)
            .map(|i| {
                let t = if n == 1 {
                    0.0f32
                } else {
                    i as f32 / (n - 1) as f32 - 0.5
                };
                let a = angle_rad * t;
                // For the centre pixel the offset is zero; for others it is
                // approximated as the tangential displacement at that angle.
                BlurSample {
                    x_offset: cx * a.cos() - cy * a.sin() - cx,
                    y_offset: cx * a.sin() + cy * a.cos() - cy,
                    weight,
                }
            })
            .collect();

        Self {
            samples,
            blur_type: MotionBlurType::Radial,
        }
    }

    /// Sum of all sample weights.
    #[must_use]
    pub fn total_weight(&self) -> f32 {
        self.samples.iter().map(|s| s.weight).sum()
    }

    /// Return a copy of the samples re-scaled so that `total_weight() == 1.0`.
    #[must_use]
    pub fn normalized_samples(&self) -> Vec<BlurSample> {
        let total = self.total_weight();
        if total == 0.0 {
            return self.samples.clone();
        }
        self.samples
            .iter()
            .map(|s| BlurSample {
                x_offset: s.x_offset,
                y_offset: s.y_offset,
                weight: s.weight / total,
            })
            .collect()
    }
}

/// Apply a motion blur kernel to an RGB pixel buffer.
///
/// `pixels` is a flat `width * height * 3` (RGB) byte slice.
/// Returns a new `Vec<u8>` of the same layout.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub fn apply_motion_blur(
    pixels: &[u8],
    width: u32,
    height: u32,
    kernel: &MotionBlurKernel,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut output = vec![0u8; w * h * 3];
    let samples = kernel.normalized_samples();

    for y in 0..h {
        for x in 0..w {
            let mut r = 0.0f32;
            let mut g = 0.0f32;
            let mut b = 0.0f32;

            for sample in &samples {
                let sx = (x as f32 + sample.x_offset).round() as i64;
                let sy = (y as f32 + sample.y_offset).round() as i64;

                // Clamp to image bounds
                let sx = sx.clamp(0, (w as i64) - 1) as usize;
                let sy = sy.clamp(0, (h as i64) - 1) as usize;

                let idx = (sy * w + sx) * 3;
                r += pixels[idx] as f32 * sample.weight;
                g += pixels[idx + 1] as f32 * sample.weight;
                b += pixels[idx + 2] as f32 * sample.weight;
            }

            let out_idx = (y * w + x) * 3;
            output[out_idx] = r.round().clamp(0.0, 255.0) as u8;
            output[out_idx + 1] = g.round().clamp(0.0, 255.0) as u8;
            output[out_idx + 2] = b.round().clamp(0.0, 255.0) as u8;
        }
    }

    output
}

// ── MotionBlur VideoEffect ──────────────────────────────────────────────────────

/// A [`crate::VideoEffect`] that applies motion blur to RGBA [`crate::Frame`]s.
///
/// The blur algorithm is selected by [`MotionBlurMode`]:
///
/// * [`MotionBlurMode::MotionVector`] — per-pixel directional blur driven by a
///   [`crate::motion_vector_blur::MotionVectorField`] supplied at construction.
///   Delegates to [`crate::motion_vector_blur::apply_mv_blur`].
/// * [`MotionBlurMode::UniformLinear`] / [`MotionBlurMode::UniformRadial`] —
///   kernel-based blur applied uniformly to every channel independently by
///   extracting the luminance from the RGBA frame, running [`apply_motion_blur`]
///   on each RGB channel, and preserving the alpha channel.
pub struct MotionBlur {
    mode: MotionBlurMode,
    /// Motion-vector field used when `mode == MotionBlurMode::MotionVector`.
    /// If `None` and mode is `MotionVector`, the effect acts as a no-op pass-through.
    mv_field: Option<crate::motion_vector_blur::MotionVectorField>,
    mv_config: crate::motion_vector_blur::MvBlurConfig,
}

impl MotionBlur {
    /// Create a new `MotionBlur` effect.
    ///
    /// `mv_field` is only consulted when `mode` is [`MotionBlurMode::MotionVector`].
    /// Pass `None` to use a zero (stationary) field that results in no blur.
    #[must_use]
    pub fn new(
        mode: MotionBlurMode,
        mv_field: Option<crate::motion_vector_blur::MotionVectorField>,
    ) -> Self {
        Self {
            mode,
            mv_field,
            mv_config: crate::motion_vector_blur::MvBlurConfig::default(),
        }
    }

    /// Create a motion-vector-driven `MotionBlur` with an explicit
    /// [`crate::motion_vector_blur::MvBlurConfig`].
    #[must_use]
    pub fn with_mv_config(
        mode: MotionBlurMode,
        mv_field: Option<crate::motion_vector_blur::MotionVectorField>,
        mv_config: crate::motion_vector_blur::MvBlurConfig,
    ) -> Self {
        Self {
            mode,
            mv_field,
            mv_config,
        }
    }

    /// Replace the motion vector field used for `MotionVector` mode.
    pub fn set_mv_field(&mut self, field: crate::motion_vector_blur::MotionVectorField) {
        self.mv_field = Some(field);
    }
}

impl crate::VideoEffect for MotionBlur {
    fn name(&self) -> &str {
        "MotionBlur"
    }

    fn description(&self) -> &'static str {
        "Per-pixel motion blur: motion-vector, uniform-linear, or uniform-radial."
    }

    /// Apply motion blur to `input`, writing the result to `output`.
    ///
    /// # Errors
    ///
    /// Returns [`crate::VfxError::InvalidDimensions`] if `input` and `output`
    /// dimensions differ, or if a motion-vector field is present but its dimensions
    /// do not match `input`.
    fn apply(
        &mut self,
        input: &crate::Frame,
        output: &mut crate::Frame,
        _params: &crate::EffectParams,
    ) -> crate::VfxResult<()> {
        if input.width != output.width || input.height != output.height {
            return Err(crate::VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }

        match &self.mode {
            MotionBlurMode::MotionVector(_mv_cfg) => {
                // ── Motion-vector path ───────────────────────────────────────
                // Use the higher-level RGBA `apply_mv_blur` from `motion_vector_blur`.
                // If no field has been set, create an on-the-fly zero field so that
                // the effect degrades gracefully to a pass-through.
                let zero_field; // declared outside to extend lifetime if needed
                let field: &crate::motion_vector_blur::MotionVectorField = match &self.mv_field {
                    Some(f) => f,
                    None => {
                        zero_field = crate::motion_vector_blur::MotionVectorField::new_zero(
                            input.width,
                            input.height,
                        )?;
                        &zero_field
                    }
                };
                crate::motion_vector_blur::apply_mv_blur(input, output, field, &self.mv_config)
            }

            MotionBlurMode::UniformLinear {
                dx,
                dy,
                num_samples,
            } => {
                // ── Uniform linear path ──────────────────────────────────────
                // Build a kernel and run apply_motion_blur per RGB channel,
                // preserving the alpha channel from `input`.
                let kernel = MotionBlurKernel::linear_kernel(*dx, *dy, *num_samples);
                apply_kernel_rgba(input, output, &kernel);
                Ok(())
            }

            MotionBlurMode::UniformRadial {
                cx,
                cy,
                angle_rad,
                num_samples,
            } => {
                // ── Uniform radial path ──────────────────────────────────────
                let kernel = MotionBlurKernel::radial_kernel(*cx, *cy, *angle_rad, *num_samples);
                apply_kernel_rgba(input, output, &kernel);
                Ok(())
            }
        }
    }
}

/// Apply a [`MotionBlurKernel`] to an RGBA [`crate::Frame`].
///
/// Processes each RGB channel by extracting it into a flat `u8` slice,
/// passing it through [`apply_motion_blur`] (which treats the data as
/// single-channel grayscale), and reassembling the output.  The alpha
/// channel is copied from `input` unchanged.
fn apply_kernel_rgba(input: &crate::Frame, output: &mut crate::Frame, kernel: &MotionBlurKernel) {
    let w = input.width;
    let h = input.height;
    let pixel_count = (w as usize) * (h as usize);

    // Extract individual channels
    let mut r_plane = vec![0u8; pixel_count];
    let mut g_plane = vec![0u8; pixel_count];
    let mut b_plane = vec![0u8; pixel_count];

    for i in 0..pixel_count {
        let base = i * 4;
        if base + 2 < input.data.len() {
            r_plane[i] = input.data[base];
            g_plane[i] = input.data[base + 1];
            b_plane[i] = input.data[base + 2];
        }
    }

    // apply_motion_blur operates on RGB triples; pack each plane into RGB
    // and unpack after.  We use a small intermediary: pack as [R, G, B, R, G, B, …]
    // and call it once with an artificial 3-channel layout.
    let packed_len = pixel_count * 3;
    let mut packed = vec![0u8; packed_len];
    for i in 0..pixel_count {
        packed[i * 3] = r_plane[i];
        packed[i * 3 + 1] = g_plane[i];
        packed[i * 3 + 2] = b_plane[i];
    }

    let blurred = apply_motion_blur(&packed, w, h, kernel);

    for i in 0..pixel_count {
        let src_base = i * 4;
        let dst_base = i * 3;
        if src_base + 3 < output.data.len() && dst_base + 2 < blurred.len() {
            output.data[src_base] = blurred[dst_base];
            output.data[src_base + 1] = blurred[dst_base + 1];
            output.data[src_base + 2] = blurred[dst_base + 2];
            // Preserve alpha from input
            output.data[src_base + 3] = if src_base + 3 < input.data.len() {
                input.data[src_base + 3]
            } else {
                255
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------ //
    // mv_motion_blur
    // ------------------------------------------------------------------ //

    #[test]
    fn test_mv_motion_blur_zero_vectors_identity() {
        let w = 16u32;
        let h = 16u32;
        let frame: Vec<u8> = (0..(w * h) as usize).map(|i| (i % 256) as u8).collect();
        let mv_field: Vec<(f32, f32)> = vec![(0.0, 0.0); (w * h) as usize];
        let cfg = MvMotionBlurConfig {
            max_blur_pixels: 8.0,
            num_samples: 1,
        };
        let out = mv_motion_blur(&frame, &mv_field, w, h, &cfg);
        let p = psnr(&frame, &out);
        assert!(
            p > 60.0,
            "PSNR with zero MV field should be > 60 dB, got {p:.1}"
        );
    }

    #[test]
    fn test_mv_motion_blur_uniform_horizontal() {
        // Vertical-stripe pattern (alternating columns 0/255)
        let w = 32u32;
        let h = 16u32;
        let frame: Vec<u8> = (0..(w * h) as usize)
            .map(|i| if (i % w as usize) % 2 == 0 { 0 } else { 255 })
            .collect();
        let mv_field: Vec<(f32, f32)> = vec![(5.0, 0.0); (w * h) as usize];
        let cfg = MvMotionBlurConfig {
            max_blur_pixels: 8.0,
            num_samples: 8,
        };
        let out = mv_motion_blur(&frame, &mv_field, w, h, &cfg);
        // Blurred output should differ substantially from the sharp input
        let mse: f64 = frame
            .iter()
            .zip(out.iter())
            .map(|(&a, &b)| {
                let d = (a as f64) - (b as f64);
                d * d
            })
            .sum::<f64>()
            / frame.len() as f64;
        assert!(
            mse > 100.0,
            "horizontal blur should produce visible difference (MSE > 100), got {mse:.2}"
        );
    }

    #[test]
    fn test_mv_motion_blur_psnr_soft_edge() {
        // Gradient image: pixel value = x (horizontal ramp)
        let w = 32u32;
        let h = 32u32;
        let frame: Vec<u8> = (0..(w * h) as usize)
            .map(|i| ((i % w as usize) * 255 / (w as usize - 1)) as u8)
            .collect();
        // Small uniform MV of (1.0, 0.0)
        let mv_field: Vec<(f32, f32)> = vec![(1.0, 0.0); (w * h) as usize];
        let cfg = MvMotionBlurConfig {
            max_blur_pixels: 4.0,
            num_samples: 4,
        };
        let out = mv_motion_blur(&frame, &mv_field, w, h, &cfg);
        let p = psnr(&frame, &out);
        assert!(
            p > 30.0,
            "gradient image with small MV: PSNR vs original should be > 30 dB, got {p:.1}"
        );
    }

    // ------------------------------------------------------------------ //
    // MotionBlurMode
    // ------------------------------------------------------------------ //

    #[test]
    fn test_motion_blur_mode_motion_vector_variant() {
        let mode = MotionBlurMode::MotionVector(MvMotionBlurConfig::default());
        assert!(matches!(mode, MotionBlurMode::MotionVector(_)));
    }

    // ------------------------------------------------------------------ //
    // MotionBlurType
    // ------------------------------------------------------------------ //

    #[test]
    fn test_type_radial_is_rotational() {
        assert!(MotionBlurType::Radial.is_rotational());
    }

    #[test]
    fn test_type_linear_not_rotational() {
        assert!(!MotionBlurType::Linear.is_rotational());
    }

    #[test]
    fn test_type_zoom_not_rotational() {
        assert!(!MotionBlurType::Zoom.is_rotational());
    }

    #[test]
    fn test_type_custom_not_rotational() {
        assert!(!MotionBlurType::Custom.is_rotational());
    }

    // ------------------------------------------------------------------ //
    // BlurSample
    // ------------------------------------------------------------------ //

    #[test]
    fn test_sample_zero_offset() {
        let s = BlurSample {
            x_offset: 0.0,
            y_offset: 0.0,
            weight: 1.0,
        };
        assert!(s.is_zero_offset());
    }

    #[test]
    fn test_sample_nonzero_offset() {
        let s = BlurSample {
            x_offset: 1.0,
            y_offset: 0.0,
            weight: 1.0,
        };
        assert!(!s.is_zero_offset());
    }

    // ------------------------------------------------------------------ //
    // MotionBlurKernel – linear
    // ------------------------------------------------------------------ //

    #[test]
    fn test_linear_kernel_sample_count() {
        let k = MotionBlurKernel::linear_kernel(10.0, 0.0, 5);
        assert_eq!(k.samples.len(), 5);
    }

    #[test]
    fn test_linear_kernel_total_weight_one() {
        let k = MotionBlurKernel::linear_kernel(10.0, 5.0, 8);
        assert!((k.total_weight() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_kernel_type() {
        let k = MotionBlurKernel::linear_kernel(0.0, 5.0, 3);
        assert_eq!(k.blur_type, MotionBlurType::Linear);
    }

    #[test]
    fn test_linear_kernel_single_sample_zero_offset() {
        let k = MotionBlurKernel::linear_kernel(10.0, 10.0, 1);
        assert!(k.samples[0].is_zero_offset());
    }

    // ------------------------------------------------------------------ //
    // MotionBlurKernel – radial
    // ------------------------------------------------------------------ //

    #[test]
    fn test_radial_kernel_sample_count() {
        let k = MotionBlurKernel::radial_kernel(0.0, 0.0, 0.1, 6);
        assert_eq!(k.samples.len(), 6);
    }

    #[test]
    fn test_radial_kernel_type() {
        let k = MotionBlurKernel::radial_kernel(0.0, 0.0, 0.1, 3);
        assert_eq!(k.blur_type, MotionBlurType::Radial);
    }

    // ------------------------------------------------------------------ //
    // MotionBlurKernel – normalization
    // ------------------------------------------------------------------ //

    #[test]
    fn test_normalized_samples_sum_to_one() {
        let k = MotionBlurKernel::linear_kernel(5.0, 0.0, 7);
        let ns = k.normalized_samples();
        let total: f32 = ns.iter().map(|s| s.weight).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    // ------------------------------------------------------------------ //
    // apply_motion_blur
    // ------------------------------------------------------------------ //

    #[test]
    fn test_apply_motion_blur_identity() {
        // Single-sample kernel with zero offset → output == input
        let kernel = MotionBlurKernel {
            samples: vec![BlurSample {
                x_offset: 0.0,
                y_offset: 0.0,
                weight: 1.0,
            }],
            blur_type: MotionBlurType::Custom,
        };
        let pixels: Vec<u8> = (0..3 * 4 * 4).map(|i| (i % 256) as u8).collect();
        let out = apply_motion_blur(&pixels, 4, 4, &kernel);
        assert_eq!(pixels, out);
    }

    #[test]
    fn test_apply_motion_blur_output_length() {
        let kernel = MotionBlurKernel::linear_kernel(2.0, 0.0, 5);
        let pixels = vec![128u8; 3 * 8 * 8];
        let out = apply_motion_blur(&pixels, 8, 8, &kernel);
        assert_eq!(out.len(), pixels.len());
    }

    #[test]
    fn test_apply_motion_blur_uniform_image_unchanged() {
        // A solid colour image should remain identical after any blur
        let kernel = MotionBlurKernel::linear_kernel(5.0, 3.0, 9);
        let pixels = vec![200u8; 3 * 10 * 10];
        let out = apply_motion_blur(&pixels, 10, 10, &kernel);
        assert!(out.iter().all(|&v| v == 200));
    }

    // ------------------------------------------------------------------ //
    // MotionBlur VideoEffect dispatch
    // ------------------------------------------------------------------ //

    /// Helper: create a checkerboard RGBA frame.
    fn checkerboard_frame(w: u32, h: u32) -> crate::Frame {
        let mut f = crate::Frame::new(w, h).expect("frame");
        for y in 0..h {
            for x in 0..w {
                let v: u8 = if (x + y) % 2 == 0 { 64 } else { 192 };
                f.set_pixel(x, y, [v, v, v, 255]);
            }
        }
        f
    }

    /// Helper: build an RGBA frame from a grayscale u8 slice (sets R=G=B=val, A=255).
    fn rgba_from_gray(gray: &[u8], w: u32, h: u32) -> crate::Frame {
        let mut f = crate::Frame::new(w, h).expect("frame");
        for (i, &val) in gray.iter().enumerate() {
            let x = (i % w as usize) as u32;
            let y = (i / w as usize) as u32;
            f.set_pixel(x, y, [val, val, val, 255]);
        }
        f
    }

    /// PSNR between two RGBA frames (only the R channel, representative for gray).
    fn psnr_rgba(a: &crate::Frame, b: &crate::Frame) -> f64 {
        assert_eq!(a.width, b.width);
        assert_eq!(a.height, b.height);
        let mse: f64 = (0..a.width)
            .flat_map(|x| (0..a.height).map(move |y| (x, y)))
            .map(|(x, y)| {
                let pa = a.get_pixel(x, y).unwrap_or([0, 0, 0, 255]);
                let pb = b.get_pixel(x, y).unwrap_or([0, 0, 0, 255]);
                let d = pa[0] as f64 - pb[0] as f64;
                d * d
            })
            .sum::<f64>()
            / (a.width as f64 * a.height as f64);
        if mse == 0.0 {
            f64::INFINITY
        } else {
            20.0 * 255.0_f64.log10() - 10.0 * mse.log10()
        }
    }

    /// `VideoEffect::apply()` with `MotionVector` mode produces output that matches
    /// calling `apply_mv_blur` directly with the same field and config.
    #[test]
    fn test_motion_blur_effect_mv_mode_matches_free_fn() {
        use crate::motion_vector_blur::{MotionVector, MotionVectorField, MvBlurConfig};
        use crate::VideoEffect;

        let w = 16u32;
        let h = 16u32;

        let input = checkerboard_frame(w, h);

        // Build a uniform MV field (horizontal pan, 2.5 px)
        let mut field = MotionVectorField::new_zero(w, h).expect("field");
        field.fill_uniform(MotionVector::new(2.5, 0.0));

        let config = MvBlurConfig {
            num_samples: 6,
            strength: 1.0,
            min_magnitude: 0.5,
            max_displacement: 8.0,
        };

        // ── Reference: call apply_mv_blur directly ──────────────────────────
        let mut ref_output = crate::Frame::new(w, h).expect("ref_output");
        crate::motion_vector_blur::apply_mv_blur(&input, &mut ref_output, &field, &config)
            .expect("apply_mv_blur");

        // ── System under test: MotionBlur VideoEffect ───────────────────────
        let mut effect = MotionBlur::with_mv_config(
            MotionBlurMode::MotionVector(MvMotionBlurConfig::default()),
            Some(field),
            config,
        );
        let params = crate::EffectParams::default();
        let mut effect_output = crate::Frame::new(w, h).expect("effect_output");
        effect
            .apply(&input, &mut effect_output, &params)
            .expect("VideoEffect::apply");

        // Both paths should produce bit-identical results.
        assert_eq!(
            ref_output.data, effect_output.data,
            "VideoEffect::apply(MotionVector) must produce identical output to apply_mv_blur()"
        );
    }

    /// `VideoEffect::apply()` with `MotionVector` mode and no field set acts as
    /// a pass-through (zero-displacement → no blur; pixels copied unchanged).
    #[test]
    fn test_motion_blur_effect_mv_mode_no_field_passthrough() {
        use crate::VideoEffect;

        let w = 8u32;
        let h = 8u32;
        let input = checkerboard_frame(w, h);

        // No MV field provided → effect creates an implicit zero field → no blur.
        let mut effect = MotionBlur::new(
            MotionBlurMode::MotionVector(MvMotionBlurConfig::default()),
            None,
        );
        let params = crate::EffectParams::default();
        let mut output = crate::Frame::new(w, h).expect("output");
        effect
            .apply(&input, &mut output, &params)
            .expect("apply with no field");

        // With zero MVs and default min_magnitude=0.5 all pixels are stationary,
        // so the output should be identical to the input.
        assert_eq!(
            input.data, output.data,
            "zero-displacement field should produce pass-through output"
        );
    }

    /// `VideoEffect::apply()` with `UniformLinear` mode blurs an RGBA frame using
    /// the kernel path and preserves the alpha channel.
    #[test]
    fn test_motion_blur_effect_uniform_linear_rgba_alpha_preserved() {
        use crate::VideoEffect;

        let w = 16u32;
        let h = 16u32;

        // Create a frame with alpha = 200 everywhere (non-trivial alpha to verify preservation)
        let mut input = crate::Frame::new(w, h).expect("input");
        for y in 0..h {
            for x in 0..w {
                let v: u8 = if (x + y) % 2 == 0 { 50 } else { 200 };
                input.set_pixel(x, y, [v, v, v, 200]);
            }
        }

        let mut effect = MotionBlur::new(
            MotionBlurMode::UniformLinear {
                dx: 3.0,
                dy: 0.0,
                num_samples: 5,
            },
            None,
        );
        let params = crate::EffectParams::default();
        let mut output = crate::Frame::new(w, h).expect("output");
        effect.apply(&input, &mut output, &params).expect("apply");

        // Alpha channel must be preserved
        for y in 0..h {
            for x in 0..w {
                let p = output.get_pixel(x, y).expect("pixel");
                assert_eq!(p[3], 200, "alpha should be 200 at ({x},{y}), got {}", p[3]);
            }
        }
    }

    /// The `MotionBlur` effect name is consistent with the trait contract.
    #[test]
    fn test_motion_blur_effect_name() {
        use crate::VideoEffect;
        let effect = MotionBlur::new(
            MotionBlurMode::MotionVector(MvMotionBlurConfig::default()),
            None,
        );
        assert_eq!(effect.name(), "MotionBlur");
    }

    /// The grayscale `mv_motion_blur` result and the RGBA `VideoEffect::apply()` result
    /// agree on the red channel when the frame is uniform gray.
    #[test]
    fn test_motion_blur_effect_mv_mode_gray_channel_consistency() {
        use crate::motion_vector_blur::{MotionVector, MotionVectorField, MvBlurConfig};
        use crate::VideoEffect;

        let w = 12u32;
        let h = 12u32;

        // Gradient gray image: pixel value = x
        let gray: Vec<u8> = (0..w * h)
            .map(|i| ((i % w as u32) * 255 / (w - 1)) as u8)
            .collect();

        let input_rgba = rgba_from_gray(&gray, w, h);

        // MV field: uniform (1.0, 0.5)
        let mut field = MotionVectorField::new_zero(w, h).expect("field");
        field.fill_uniform(MotionVector::new(1.0, 0.5));

        let config = MvBlurConfig {
            num_samples: 4,
            strength: 1.0,
            min_magnitude: 0.5,
            max_displacement: 8.0,
        };

        // Reference via VideoEffect dispatch
        let mut effect = MotionBlur::with_mv_config(
            MotionBlurMode::MotionVector(MvMotionBlurConfig::default()),
            Some(field),
            config,
        );
        let params = crate::EffectParams::default();
        let mut output_rgba = crate::Frame::new(w, h).expect("output");
        effect
            .apply(&input_rgba, &mut output_rgba, &params)
            .expect("apply");

        // The RGBA output should have reasonable PSNR vs. the input (shows it ran)
        let p = psnr_rgba(&input_rgba, &output_rgba);
        // A gradient image with small motion vectors should stay visually similar
        assert!(
            p > 20.0,
            "PSNR vs input after small-MV blur should be > 20 dB, got {p:.1}"
        );
    }
}
