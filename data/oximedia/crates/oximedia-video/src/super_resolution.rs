//! Single-image super resolution (upscaling) for video frames.
//!
//! Provides AI-free upscaling methods based on classical signal processing:
//! bicubic interpolation with optional unsharp-mask sharpening, Lanczos
//! resampling, and edge-directed interpolation for sharper edges.

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur during super-resolution processing.
#[derive(Debug, thiserror::Error)]
pub enum SuperResolutionError {
    /// Input dimensions are invalid (zero width or height).
    #[error("invalid input dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },
    /// Output dimensions are invalid or smaller than input.
    #[error("invalid output dimensions: {out_w}x{out_h} (input: {in_w}x{in_h})")]
    InvalidOutputDimensions {
        /// Input width.
        in_w: u32,
        /// Input height.
        in_h: u32,
        /// Requested output width.
        out_w: u32,
        /// Requested output height.
        out_h: u32,
    },
    /// Buffer size does not match expected dimensions.
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length.
        actual: usize,
    },
    /// Scale factor is out of valid range.
    #[error("scale factor {0} is out of valid range (1.0, 16.0]")]
    InvalidScaleFactor(f64),
}

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Upscaling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleMethod {
    /// Bicubic interpolation (Mitchell-Netravali, B=1/3, C=1/3).
    Bicubic,
    /// Lanczos-3 resampling (windowed sinc).
    Lanczos3,
    /// Nearest-neighbour (pixel-art style, no smoothing).
    NearestNeighbor,
    /// Edge-directed interpolation: bicubic base + gradient-aware correction.
    EdgeDirected,
    /// NEDI-like (New Edge-Directed Interpolation): gradient covariance in a
    /// 4×4 window drives adaptive edge-aligned interpolation.
    NediSr,
}

/// Sharpening parameters applied after upscaling.
#[derive(Debug, Clone)]
pub struct SharpenParams {
    /// Unsharp mask radius in pixels (used for Gaussian blur kernel).
    /// Typical range: 0.5 to 3.0.
    pub radius: f64,
    /// Strength of the sharpening effect. 0.0 = none, 1.0 = strong.
    pub amount: f64,
    /// Threshold below which sharpening is suppressed (reduces noise amplification).
    /// In [0, 255] luma units.
    pub threshold: f64,
}

impl Default for SharpenParams {
    fn default() -> Self {
        Self {
            radius: 1.0,
            amount: 0.5,
            threshold: 2.0,
        }
    }
}

/// Configuration for the super-resolution upscaler.
#[derive(Debug, Clone)]
pub struct UpscaleConfig {
    /// Target output width. If `None`, computed from `scale_factor`.
    pub target_width: Option<u32>,
    /// Target output height. If `None`, computed from `scale_factor`.
    pub target_height: Option<u32>,
    /// Scale factor (e.g. 2.0 for 2x upscale). Ignored if both target
    /// dimensions are specified.
    pub scale_factor: f64,
    /// Upscaling algorithm.
    pub method: UpscaleMethod,
    /// Optional post-upscale sharpening.
    pub sharpen: Option<SharpenParams>,
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self {
            target_width: None,
            target_height: None,
            scale_factor: 2.0,
            method: UpscaleMethod::Bicubic,
            sharpen: Some(SharpenParams::default()),
        }
    }
}

/// Result of an upscale operation containing the output buffer and dimensions.
#[derive(Debug, Clone)]
pub struct UpscaleResult {
    /// Output Y plane (luma).
    pub y_plane: Vec<u8>,
    /// Output U plane (chroma).
    pub u_plane: Vec<u8>,
    /// Output V plane (chroma).
    pub v_plane: Vec<u8>,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
}

impl UpscaleResult {
    /// Return the full YUV420 planar buffer (Y, then U, then V concatenated).
    pub fn to_yuv420(&self) -> Vec<u8> {
        let mut buf =
            Vec::with_capacity(self.y_plane.len() + self.u_plane.len() + self.v_plane.len());
        buf.extend_from_slice(&self.y_plane);
        buf.extend_from_slice(&self.u_plane);
        buf.extend_from_slice(&self.v_plane);
        buf
    }
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// Upscale a YUV420 planar frame according to the given configuration.
///
/// `frame` layout: Y plane `width*height`, then U `(w/2)*(h/2)`, then V.
pub fn upscale(
    frame: &[u8],
    width: u32,
    height: u32,
    config: &UpscaleConfig,
) -> Result<UpscaleResult, SuperResolutionError> {
    validate_input(width, height)?;

    let y_size = (width as usize) * (height as usize);
    let uv_w = ((width + 1) / 2) as usize;
    let uv_h = ((height + 1) / 2) as usize;
    let uv_size = uv_w * uv_h;
    let expected = y_size + 2 * uv_size;

    if frame.len() < expected {
        return Err(SuperResolutionError::BufferSizeMismatch {
            expected,
            actual: frame.len(),
        });
    }

    let (out_w, out_h) = resolve_output_dims(width, height, config)?;

    let y_plane = &frame[..y_size];
    let u_plane = &frame[y_size..y_size + uv_size];
    let v_plane = &frame[y_size + uv_size..y_size + 2 * uv_size];

    // Upscale luma at full resolution.
    let mut y_out = upscale_plane(y_plane, width, height, out_w, out_h, config.method);

    // Upscale chroma planes.
    let out_uv_w = ((out_w + 1) / 2).max(1);
    let out_uv_h = ((out_h + 1) / 2).max(1);
    let u_out = upscale_plane(
        u_plane,
        uv_w as u32,
        uv_h as u32,
        out_uv_w,
        out_uv_h,
        config.method,
    );
    let v_out = upscale_plane(
        v_plane,
        uv_w as u32,
        uv_h as u32,
        out_uv_w,
        out_uv_h,
        config.method,
    );

    // Apply sharpening to luma if configured.
    if let Some(ref sharpen) = config.sharpen {
        apply_unsharp_mask(&mut y_out, out_w, out_h, sharpen);
    }

    Ok(UpscaleResult {
        y_plane: y_out,
        u_plane: u_out,
        v_plane: v_out,
        width: out_w,
        height: out_h,
    })
}

/// Simple 2x upscale with bicubic + default sharpening (convenience wrapper).
pub fn upscale_2x(
    frame: &[u8],
    width: u32,
    height: u32,
) -> Result<UpscaleResult, SuperResolutionError> {
    upscale(frame, width, height, &UpscaleConfig::default())
}

// -----------------------------------------------------------------------
// Plane upscaling
// -----------------------------------------------------------------------

fn upscale_plane(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    method: UpscaleMethod,
) -> Vec<u8> {
    let sw = src_w as usize;
    let sh = src_h as usize;
    let dw = dst_w as usize;
    let dh = dst_h as usize;

    let mut dst = vec![0u8; dw * dh];

    let x_ratio = if dw > 1 { sw as f64 / dw as f64 } else { 0.0 };
    let y_ratio = if dh > 1 { sh as f64 / dh as f64 } else { 0.0 };

    for dy in 0..dh {
        let sy = dy as f64 * y_ratio;
        for dx in 0..dw {
            let sx = dx as f64 * x_ratio;
            let val = match method {
                UpscaleMethod::NearestNeighbor => {
                    let ix = (sx + 0.5) as usize;
                    let iy = (sy + 0.5) as usize;
                    sample_clamped(src, sw, sh, ix as i64, iy as i64)
                }
                UpscaleMethod::Bicubic => bicubic_sample(src, sw, sh, sx, sy),
                UpscaleMethod::Lanczos3 => lanczos3_sample(src, sw, sh, sx, sy),
                UpscaleMethod::EdgeDirected => edge_directed_sample(src, sw, sh, sx, sy),
                UpscaleMethod::NediSr => nedi_sr_sample(src, sw, sh, sx, sy),
            };
            dst[dy * dw + dx] = val.clamp(0.0, 255.0) as u8;
        }
    }

    dst
}

/// Read a pixel with clamped coordinates.
fn sample_clamped(src: &[u8], w: usize, h: usize, x: i64, y: i64) -> f64 {
    let cx = x.clamp(0, w as i64 - 1) as usize;
    let cy = y.clamp(0, h as i64 - 1) as usize;
    src.get(cy * w + cx).copied().unwrap_or(0) as f64
}

// -----------------------------------------------------------------------
// Bicubic interpolation (Mitchell-Netravali B=1/3, C=1/3)
// -----------------------------------------------------------------------

fn mitchell_netravali(t: f64) -> f64 {
    let t = t.abs();
    const B: f64 = 1.0 / 3.0;
    const C: f64 = 1.0 / 3.0;
    if t < 1.0 {
        ((12.0 - 9.0 * B - 6.0 * C) * t * t * t
            + (-18.0 + 12.0 * B + 6.0 * C) * t * t
            + (6.0 - 2.0 * B))
            / 6.0
    } else if t < 2.0 {
        ((-B - 6.0 * C) * t * t * t
            + (6.0 * B + 30.0 * C) * t * t
            + (-12.0 * B - 48.0 * C) * t
            + (8.0 * B + 24.0 * C))
            / 6.0
    } else {
        0.0
    }
}

fn bicubic_sample(src: &[u8], w: usize, h: usize, sx: f64, sy: f64) -> f64 {
    let ix = sx.floor() as i64;
    let iy = sy.floor() as i64;
    let fx = sx - ix as f64;
    let fy = sy - iy as f64;

    let mut sum = 0.0;
    let mut weight_sum = 0.0;

    for j in -1..=2i64 {
        let wy = mitchell_netravali(fy - j as f64);
        for i in -1..=2i64 {
            let wx = mitchell_netravali(fx - i as f64);
            let w_total = wx * wy;
            sum += sample_clamped(src, w, h, ix + i, iy + j) * w_total;
            weight_sum += w_total;
        }
    }

    if weight_sum.abs() > 1e-10 {
        sum / weight_sum
    } else {
        sample_clamped(src, w, h, ix, iy)
    }
}

// -----------------------------------------------------------------------
// Lanczos-3 resampling
// -----------------------------------------------------------------------

fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x.abs() < 1e-10 {
        return 1.0;
    }
    if x.abs() >= a {
        return 0.0;
    }
    let pi_x = std::f64::consts::PI * x;
    let pi_x_a = std::f64::consts::PI * x / a;
    (pi_x.sin() * pi_x_a.sin()) / (pi_x * pi_x_a)
}

fn lanczos3_sample(src: &[u8], w: usize, h: usize, sx: f64, sy: f64) -> f64 {
    let ix = sx.floor() as i64;
    let iy = sy.floor() as i64;
    let fx = sx - ix as f64;
    let fy = sy - iy as f64;

    let a = 3.0;
    let mut sum = 0.0;
    let mut weight_sum = 0.0;

    for j in (-2)..=3i64 {
        let wy = lanczos_kernel(fy - j as f64, a);
        for i in (-2)..=3i64 {
            let wx = lanczos_kernel(fx - i as f64, a);
            let w_total = wx * wy;
            sum += sample_clamped(src, w, h, ix + i, iy + j) * w_total;
            weight_sum += w_total;
        }
    }

    if weight_sum.abs() > 1e-10 {
        sum / weight_sum
    } else {
        sample_clamped(src, w, h, ix, iy)
    }
}

// -----------------------------------------------------------------------
// Edge-directed interpolation
// -----------------------------------------------------------------------

/// Bicubic base + gradient-aware correction.
/// When a strong edge is detected, the interpolation direction follows
/// the edge to avoid blurring across it.
fn edge_directed_sample(src: &[u8], w: usize, h: usize, sx: f64, sy: f64) -> f64 {
    // Start with bicubic as the base.
    let base = bicubic_sample(src, w, h, sx, sy);

    let ix = sx.floor() as i64;
    let iy = sy.floor() as i64;

    // Compute local gradients at the 4 nearest pixels.
    let gx = compute_gradient_x(src, w, h, ix, iy);
    let gy = compute_gradient_y(src, w, h, ix, iy);
    let gradient_mag = (gx * gx + gy * gy).sqrt();

    // If gradient is strong, use directional interpolation.
    const EDGE_THRESHOLD: f64 = 30.0;
    if gradient_mag > EDGE_THRESHOLD {
        // Determine edge direction (perpendicular to gradient).
        let edge_angle = gy.atan2(gx);

        // Sample along the edge direction.
        let fx = sx - ix as f64;
        let fy = sy - iy as f64;

        let edge_dx = -edge_angle.sin();
        let edge_dy = edge_angle.cos();

        // Project sub-pixel offset onto edge direction.
        let proj = fx * edge_dx + fy * edge_dy;

        // Interpolate along edge.
        let p0 = sample_clamped(src, w, h, ix, iy);
        let p1_x = ix as f64 + edge_dx;
        let p1_y = iy as f64 + edge_dy;
        let p1 = bicubic_sample(src, w, h, p1_x, p1_y);

        let edge_val = p0 + proj * (p1 - p0);

        // Blend between base bicubic and edge-directed based on gradient strength.
        let blend = ((gradient_mag - EDGE_THRESHOLD) / 50.0).clamp(0.0, 0.7);
        base * (1.0 - blend) + edge_val * blend
    } else {
        base
    }
}

fn compute_gradient_x(src: &[u8], w: usize, h: usize, x: i64, y: i64) -> f64 {
    let left = sample_clamped(src, w, h, x - 1, y);
    let right = sample_clamped(src, w, h, x + 1, y);
    (right - left) / 2.0
}

fn compute_gradient_y(src: &[u8], w: usize, h: usize, x: i64, y: i64) -> f64 {
    let top = sample_clamped(src, w, h, x, y - 1);
    let bottom = sample_clamped(src, w, h, x, y + 1);
    (bottom - top) / 2.0
}

// -----------------------------------------------------------------------
// Unsharp mask sharpening
// -----------------------------------------------------------------------

fn apply_unsharp_mask(plane: &mut [u8], width: u32, height: u32, params: &SharpenParams) {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || params.amount <= 0.0 {
        return;
    }

    // Generate Gaussian blur of the plane.
    let blurred = gaussian_blur(plane, w, h, params.radius);

    // Unsharp mask: output = original + amount * (original - blurred)
    for i in 0..plane.len().min(w * h) {
        let orig = plane[i] as f64;
        let blur = blurred.get(i).copied().unwrap_or(orig as u8) as f64;
        let diff = orig - blur;

        if diff.abs() < params.threshold {
            continue; // Below threshold: no sharpening.
        }

        let sharpened = orig + params.amount * diff;
        plane[i] = sharpened.clamp(0.0, 255.0) as u8;
    }
}

/// Simple 1D Gaussian kernel generation.
fn make_gaussian_kernel(radius: f64) -> Vec<f64> {
    let sigma = radius.max(0.5);
    let kernel_radius = (sigma * 3.0).ceil() as usize;
    let size = kernel_radius * 2 + 1;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0;

    for i in 0..size {
        let x = i as f64 - kernel_radius as f64;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
        sum += val;
    }

    // Normalize.
    if sum > 0.0 {
        for v in &mut kernel {
            *v /= sum;
        }
    }

    kernel
}

/// Separable Gaussian blur (horizontal then vertical).
fn gaussian_blur(plane: &[u8], w: usize, h: usize, radius: f64) -> Vec<u8> {
    let kernel = make_gaussian_kernel(radius);
    let kr = kernel.len() / 2;

    // Horizontal pass.
    let mut temp = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as i64 + ki as i64 - kr as i64).clamp(0, w as i64 - 1) as usize;
                sum += plane.get(y * w + sx).copied().unwrap_or(0) as f64 * kv;
            }
            temp[y * w + x] = sum;
        }
    }

    // Vertical pass.
    let mut result = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as i64 + ki as i64 - kr as i64).clamp(0, h as i64 - 1) as usize;
                sum += temp[sy * w + x] * kv;
            }
            result[y * w + x] = sum.clamp(0.0, 255.0) as u8;
        }
    }

    result
}

// -----------------------------------------------------------------------
// Validation helpers
// -----------------------------------------------------------------------

fn validate_input(width: u32, height: u32) -> Result<(), SuperResolutionError> {
    if width == 0 || height == 0 {
        return Err(SuperResolutionError::InvalidDimensions { width, height });
    }
    Ok(())
}

fn resolve_output_dims(
    in_w: u32,
    in_h: u32,
    config: &UpscaleConfig,
) -> Result<(u32, u32), SuperResolutionError> {
    let (out_w, out_h) = match (config.target_width, config.target_height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => {
            let scale = w as f64 / in_w as f64;
            (w, (in_h as f64 * scale).round() as u32)
        }
        (None, Some(h)) => {
            let scale = h as f64 / in_h as f64;
            ((in_w as f64 * scale).round() as u32, h)
        }
        (None, None) => {
            if config.scale_factor <= 1.0 || config.scale_factor > 16.0 {
                return Err(SuperResolutionError::InvalidScaleFactor(
                    config.scale_factor,
                ));
            }
            (
                (in_w as f64 * config.scale_factor).round() as u32,
                (in_h as f64 * config.scale_factor).round() as u32,
            )
        }
    };

    if out_w == 0 || out_h == 0 {
        return Err(SuperResolutionError::InvalidOutputDimensions {
            in_w,
            in_h,
            out_w,
            out_h,
        });
    }

    Ok((out_w, out_h))
}

// -----------------------------------------------------------------------
// SuperResolutionEngine — new high-level API
// -----------------------------------------------------------------------

/// Super-resolution processing mode for the [`SuperResolutionEngine`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrMode {
    /// Bicubic interpolation (Mitchell-Netravali) with aggressive unsharp-mask
    /// sharpening applied after upscaling.
    BicubicSharp,
    /// Lanczos-3 resampling with post-upscale unsharp-mask sharpening.
    Lanczos3Sharp,
    /// New Edge-Directed Interpolation (NEDI-like).
    ///
    /// Estimates the gradient covariance in a 4×4 neighbourhood of each
    /// output sample location and uses the dominant eigenvector to determine
    /// the local edge orientation.  Interpolation is blended between the
    /// edge direction and the bicubic base according to the gradient
    /// strength, producing sharper edges without ringing on smooth regions.
    EdgeDirectedSR,
}

/// Configuration for the [`SuperResolutionEngine`].
#[derive(Debug, Clone)]
pub struct SuperResolutionConfig {
    /// Integer scale factor (e.g. `2` for 2×, `4` for 4×).  Must be in `[1, 16]`.
    pub scale: u32,
    /// Algorithm selection.
    pub mode: SrMode,
    /// Post-upscale sharpening strength in `[0.0, 1.0]`.
    ///
    /// `0.0` disables sharpening, `1.0` applies maximum strength.
    pub sharpening_amount: f32,
}

impl Default for SuperResolutionConfig {
    fn default() -> Self {
        Self {
            scale: 2,
            mode: SrMode::BicubicSharp,
            sharpening_amount: 0.5,
        }
    }
}

/// Super-resolution engine for upscaling single-channel image planes.
///
/// Construct with [`SuperResolutionEngine::new`], then call
/// [`upscale_channel`](SuperResolutionEngine::upscale_channel) for each
/// plane (luma / chroma) you wish to upscale.
pub struct SuperResolutionEngine {
    config: SuperResolutionConfig,
}

impl SuperResolutionEngine {
    /// Create a new engine with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SuperResolutionError::InvalidScaleFactor`] if `config.scale`
    /// is 0 or greater than 16.
    pub fn new(config: SuperResolutionConfig) -> Result<Self, SuperResolutionError> {
        if config.scale == 0 || config.scale > 16 {
            return Err(SuperResolutionError::InvalidScaleFactor(
                config.scale as f64,
            ));
        }
        Ok(Self { config })
    }

    /// Returns a reference to this engine's configuration.
    pub fn config(&self) -> &SuperResolutionConfig {
        &self.config
    }

    /// Upscale a raw single-channel (grayscale / luma) frame.
    ///
    /// `frame` must contain exactly `w * h` bytes.  Returns a buffer of
    /// `(w * scale) * (h * scale)` bytes representing the upscaled plane.
    ///
    /// # Errors
    ///
    /// - [`SuperResolutionError::InvalidDimensions`] if `w` or `h` is zero.
    /// - [`SuperResolutionError::BufferSizeMismatch`] if `frame.len() != w * h`.
    pub fn upscale_channel(
        &self,
        frame: &[u8],
        w: u32,
        h: u32,
    ) -> Result<Vec<u8>, SuperResolutionError> {
        if w == 0 || h == 0 {
            return Err(SuperResolutionError::InvalidDimensions {
                width: w,
                height: h,
            });
        }
        let expected = (w as usize) * (h as usize);
        if frame.len() != expected {
            return Err(SuperResolutionError::BufferSizeMismatch {
                expected,
                actual: frame.len(),
            });
        }
        let out_w = w * self.config.scale;
        let out_h = h * self.config.scale;

        let method = match self.config.mode {
            SrMode::BicubicSharp => UpscaleMethod::Bicubic,
            SrMode::Lanczos3Sharp => UpscaleMethod::Lanczos3,
            SrMode::EdgeDirectedSR => UpscaleMethod::NediSr,
        };

        let mut out = upscale_plane(frame, w, h, out_w, out_h, method);

        // Apply sharpening for modes that include it.
        if self.config.sharpening_amount > 0.0 {
            let sharpen_params = SharpenParams {
                radius: 1.0,
                amount: self.config.sharpening_amount as f64,
                threshold: 2.0,
            };
            apply_unsharp_mask(&mut out, out_w, out_h, &sharpen_params);
        }

        Ok(out)
    }
}

/// Convenience function: upscale a raw single-channel frame using a
/// [`SuperResolutionConfig`].
///
/// `frame` must be exactly `w * h` bytes (single plane, e.g. Y luma).
/// Returns `(w * config.scale) * (h * config.scale)` bytes.
///
/// This is a thin wrapper around [`SuperResolutionEngine::upscale_channel`].
pub fn sr_upscale(
    frame: &[u8],
    w: u32,
    h: u32,
    config: &SuperResolutionConfig,
) -> Result<Vec<u8>, SuperResolutionError> {
    let engine = SuperResolutionEngine::new(config.clone())?;
    engine.upscale_channel(frame, w, h)
}

// -----------------------------------------------------------------------
// NEDI-like edge-directed interpolation (NediSr variant)
// -----------------------------------------------------------------------

/// NEDI-like sample: gradient covariance in a 4×4 window drives adaptive
/// edge-directed interpolation.
///
/// Algorithm summary:
/// 1. Build the local 4×4 gradient covariance matrix around (`sx`, `sy`).
/// 2. Compute the dominant eigenvector via the 2×2 covariance determinant.
/// 3. If the edge strength is above a threshold, interpolate along the edge
///    direction (perpendicular to the gradient) and blend with the bicubic base.
fn nedi_sr_sample(src: &[u8], w: usize, h: usize, sx: f64, sy: f64) -> f64 {
    let ix = sx.floor() as i64;
    let iy = sy.floor() as i64;

    // Collect 4×4 window gradients for covariance estimation.
    let mut cxx = 0.0_f64;
    let mut cxy = 0.0_f64;
    let mut cyy = 0.0_f64;
    let mut count = 0usize;

    for wy in -1..=2i64 {
        for wx in -1..=2i64 {
            let gx = sample_clamped(src, w, h, ix + wx + 1, iy + wy)
                - sample_clamped(src, w, h, ix + wx - 1, iy + wy);
            let gy = sample_clamped(src, w, h, ix + wx, iy + wy + 1)
                - sample_clamped(src, w, h, ix + wx, iy + wy - 1);
            cxx += gx * gx;
            cxy += gx * gy;
            cyy += gy * gy;
            count += 1;
        }
    }

    if count > 0 {
        let n = count as f64;
        cxx /= n;
        cxy /= n;
        cyy /= n;
    }

    // Dominant eigenvector of the 2×2 covariance matrix.
    // For a symmetric 2×2 matrix [[a,b],[b,c]]:
    //   eigenvalues: λ = ((a+c) ± sqrt((a-c)^2 + 4b^2)) / 2
    //   dominant eigenvector points along the gradient (max eigenvalue).
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let discriminant = (trace * trace / 4.0 - det).max(0.0);
    let lambda_max = trace / 2.0 + discriminant.sqrt();

    // Gradient magnitude as measure of edge strength.
    let edge_strength = lambda_max;

    const NEDI_EDGE_THRESHOLD: f64 = 20.0;

    if edge_strength <= NEDI_EDGE_THRESHOLD {
        // No strong edge: fall back to Lanczos3 for good quality.
        return lanczos3_sample(src, w, h, sx, sy);
    }

    // Dominant eigenvector direction (perpendicular to gradient = edge direction).
    // For the max eigenvalue λ, the eigenvector satisfies (A - λI)v = 0.
    // v = [cxy, λ - cxx] (normalised).
    let vx = cxy;
    let vy = lambda_max - cxx;
    let vlen = (vx * vx + vy * vy).sqrt();

    let (edge_dx, edge_dy) = if vlen > 1e-10 {
        (vx / vlen, vy / vlen)
    } else {
        // No clear direction: use Lanczos3.
        return lanczos3_sample(src, w, h, sx, sy);
    };

    // Sub-pixel offset from the integer grid point.
    let fx = sx - ix as f64;
    let fy = sy - iy as f64;

    // Project sub-pixel offset onto edge direction.
    let proj = fx * edge_dx + fy * edge_dy;

    // Bilinear interpolation along the edge direction.
    let p0 = sample_clamped(src, w, h, ix, iy);
    let p1 = bicubic_sample(src, w, h, ix as f64 + edge_dx, iy as f64 + edge_dy);

    let edge_val = p0 + proj * (p1 - p0);

    // Blend: stronger edges get more edge-directed weight.
    let bicubic_base = bicubic_sample(src, w, h, sx, sy);
    let blend = ((edge_strength - NEDI_EDGE_THRESHOLD) / 80.0).clamp(0.0, 0.75);
    bicubic_base * (1.0 - blend) + edge_val * blend
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_yuv420(width: u32, height: u32, y_val: u8) -> Vec<u8> {
        let y_size = (width * height) as usize;
        let uv_size = ((width + 1) / 2 * (height + 1) / 2) as usize;
        let mut buf = vec![y_val; y_size];
        buf.extend(vec![128u8; uv_size]); // U
        buf.extend(vec![128u8; uv_size]); // V
        buf
    }

    /// Create a frame with a horizontal gradient (left=0, right=255).
    fn make_gradient_frame(width: u32, height: u32) -> Vec<u8> {
        let y_size = (width * height) as usize;
        let uv_size = ((width + 1) / 2 * (height + 1) / 2) as usize;
        let mut y_plane = Vec::with_capacity(y_size);
        for _row in 0..height {
            for col in 0..width {
                let val = if width > 1 {
                    (col as f64 / (width as f64 - 1.0) * 255.0) as u8
                } else {
                    128
                };
                y_plane.push(val);
            }
        }
        y_plane.extend(vec![128u8; uv_size]);
        y_plane.extend(vec![128u8; uv_size]);
        y_plane
    }

    // ---- Basic upscale: output dimensions correct ----

    #[test]
    fn test_upscale_2x_dimensions() {
        let frame = make_yuv420(16, 16, 128);
        let result = upscale_2x(&frame, 16, 16).expect("ok");
        assert_eq!(result.width, 32);
        assert_eq!(result.height, 32);
        assert_eq!(result.y_plane.len(), 32 * 32);
    }

    #[test]
    fn test_upscale_3x_dimensions() {
        let frame = make_yuv420(8, 8, 128);
        let config = UpscaleConfig {
            scale_factor: 3.0,
            sharpen: None,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 8, 8, &config).expect("ok");
        assert_eq!(result.width, 24);
        assert_eq!(result.height, 24);
    }

    // ---- Target dimensions ----

    #[test]
    fn test_upscale_target_dimensions() {
        let frame = make_yuv420(8, 8, 128);
        let config = UpscaleConfig {
            target_width: Some(20),
            target_height: Some(30),
            sharpen: None,
            method: UpscaleMethod::Bicubic,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 8, 8, &config).expect("ok");
        assert_eq!(result.width, 20);
        assert_eq!(result.height, 30);
    }

    #[test]
    fn test_upscale_target_width_only() {
        let frame = make_yuv420(8, 4, 128);
        let config = UpscaleConfig {
            target_width: Some(16),
            target_height: None,
            sharpen: None,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 8, 4, &config).expect("ok");
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 8); // aspect ratio preserved
    }

    // ---- Flat frame upscale preserves value ----

    #[test]
    fn test_flat_frame_preserves_value_bicubic() {
        let frame = make_yuv420(8, 8, 100);
        let config = UpscaleConfig {
            scale_factor: 2.0,
            sharpen: None,
            method: UpscaleMethod::Bicubic,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 8, 8, &config).expect("ok");
        // Flat frame should remain ~100 everywhere.
        for &val in &result.y_plane {
            assert!((val as i32 - 100).abs() <= 1, "expected ~100, got {val}");
        }
    }

    #[test]
    fn test_flat_frame_preserves_value_lanczos() {
        let frame = make_yuv420(8, 8, 200);
        let config = UpscaleConfig {
            scale_factor: 2.0,
            sharpen: None,
            method: UpscaleMethod::Lanczos3,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 8, 8, &config).expect("ok");
        for &val in &result.y_plane {
            assert!((val as i32 - 200).abs() <= 1, "expected ~200, got {val}");
        }
    }

    // ---- Nearest-neighbor ----

    #[test]
    fn test_nearest_neighbor_2x() {
        let frame = make_yuv420(4, 4, 50);
        let config = UpscaleConfig {
            scale_factor: 2.0,
            sharpen: None,
            method: UpscaleMethod::NearestNeighbor,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 4, 4, &config).expect("ok");
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
        // All pixels should be exactly 50.
        for &val in &result.y_plane {
            assert_eq!(val, 50);
        }
    }

    // ---- Edge-directed ----

    #[test]
    fn test_edge_directed_upscale() {
        let frame = make_gradient_frame(16, 16);
        let config = UpscaleConfig {
            scale_factor: 2.0,
            sharpen: None,
            method: UpscaleMethod::EdgeDirected,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 16, 16, &config).expect("ok");
        assert_eq!(result.width, 32);
        assert_eq!(result.height, 32);
        // Gradient should be approximately preserved.
        let left = result.y_plane[0] as i32;
        let right = result.y_plane[31] as i32;
        assert!(
            right > left,
            "gradient should be preserved: left={left}, right={right}"
        );
    }

    // ---- Gradient frame quality ----

    #[test]
    fn test_gradient_monotonicity_bicubic() {
        let frame = make_gradient_frame(32, 8);
        let config = UpscaleConfig {
            scale_factor: 2.0,
            sharpen: None,
            method: UpscaleMethod::Bicubic,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 32, 8, &config).expect("ok");
        // First row should be approximately non-decreasing.
        let row: Vec<u8> = result.y_plane[..result.width as usize].to_vec();
        let mut violations = 0;
        for w in row.windows(2) {
            if (w[0] as i32) - (w[1] as i32) > 2 {
                violations += 1;
            }
        }
        assert!(
            violations < 3,
            "bicubic should roughly preserve gradient monotonicity, got {violations} violations"
        );
    }

    // ---- Sharpening ----

    #[test]
    fn test_sharpening_increases_contrast() {
        let frame = make_gradient_frame(16, 16);
        let config_no_sharp = UpscaleConfig {
            scale_factor: 2.0,
            sharpen: None,
            method: UpscaleMethod::Bicubic,
            ..UpscaleConfig::default()
        };
        let config_sharp = UpscaleConfig {
            scale_factor: 2.0,
            sharpen: Some(SharpenParams {
                radius: 1.5,
                amount: 1.0,
                threshold: 0.0,
            }),
            method: UpscaleMethod::Bicubic,
            ..UpscaleConfig::default()
        };
        let no_sharp = upscale(&frame, 16, 16, &config_no_sharp).expect("ok");
        let sharp = upscale(&frame, 16, 16, &config_sharp).expect("ok");
        // Sharpened should differ from non-sharpened.
        assert_ne!(
            no_sharp.y_plane, sharp.y_plane,
            "sharpening should change the output"
        );
    }

    #[test]
    fn test_sharpening_threshold_suppresses_noise() {
        let frame = make_yuv420(16, 16, 128);
        let config = UpscaleConfig {
            scale_factor: 2.0,
            sharpen: Some(SharpenParams {
                radius: 1.0,
                amount: 1.0,
                threshold: 255.0, // Very high threshold: suppress all sharpening.
            }),
            method: UpscaleMethod::Bicubic,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 16, 16, &config).expect("ok");
        // With flat input + high threshold, output should be ~128 everywhere.
        for &val in &result.y_plane {
            assert!(
                (val as i32 - 128).abs() <= 1,
                "high threshold should suppress sharpening on flat input, got {val}"
            );
        }
    }

    // ---- YUV420 output format ----

    #[test]
    fn test_to_yuv420_concatenation() {
        let frame = make_yuv420(8, 8, 128);
        let result = upscale_2x(&frame, 8, 8).expect("ok");
        let yuv = result.to_yuv420();
        let y_size = (result.width * result.height) as usize;
        assert_eq!(
            yuv.len(),
            y_size + result.u_plane.len() + result.v_plane.len()
        );
        assert_eq!(&yuv[..y_size], &result.y_plane[..]);
    }

    // ---- Chroma planes are upscaled ----

    #[test]
    fn test_chroma_planes_upscaled() {
        let frame = make_yuv420(8, 8, 128);
        let result = upscale_2x(&frame, 8, 8).expect("ok");
        let out_uv_w = ((result.width + 1) / 2) as usize;
        let out_uv_h = ((result.height + 1) / 2) as usize;
        assert_eq!(result.u_plane.len(), out_uv_w * out_uv_h);
        assert_eq!(result.v_plane.len(), out_uv_w * out_uv_h);
    }

    // ---- Error cases ----

    #[test]
    fn test_zero_width_error() {
        let frame = make_yuv420(1, 1, 128);
        let result = upscale(&frame, 0, 8, &UpscaleConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_height_error() {
        let frame = make_yuv420(1, 1, 128);
        let result = upscale(&frame, 8, 0, &UpscaleConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_too_small_error() {
        let frame = vec![0u8; 10];
        let result = upscale(&frame, 16, 16, &UpscaleConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_scale_factor() {
        let frame = make_yuv420(8, 8, 128);
        let config = UpscaleConfig {
            scale_factor: 0.5,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 8, 8, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_scale_factor_too_large() {
        let frame = make_yuv420(8, 8, 128);
        let config = UpscaleConfig {
            scale_factor: 17.0,
            ..UpscaleConfig::default()
        };
        let result = upscale(&frame, 8, 8, &config);
        assert!(result.is_err());
    }

    // ---- Mitchell-Netravali kernel properties ----

    #[test]
    fn test_mitchell_netravali_at_zero() {
        let val = mitchell_netravali(0.0);
        // Should be close to (6 - 2B) / 6 = (6 - 2/3) / 6 ≈ 0.889
        assert!((val - 0.8889).abs() < 0.01, "MN(0) = {val}");
    }

    #[test]
    fn test_mitchell_netravali_symmetry() {
        for t in [0.3, 0.7, 1.2, 1.8] {
            let a = mitchell_netravali(t);
            let b = mitchell_netravali(-t);
            assert!((a - b).abs() < 1e-10, "kernel should be symmetric");
        }
    }

    #[test]
    fn test_mitchell_netravali_zero_beyond_2() {
        assert_eq!(mitchell_netravali(2.5), 0.0);
        assert_eq!(mitchell_netravali(3.0), 0.0);
    }

    // ---- Lanczos kernel properties ----

    #[test]
    fn test_lanczos_kernel_at_zero() {
        let val = lanczos_kernel(0.0, 3.0);
        assert!(
            (val - 1.0).abs() < 1e-6,
            "Lanczos(0) should be 1.0, got {val}"
        );
    }

    #[test]
    fn test_lanczos_kernel_zero_at_boundary() {
        let val = lanczos_kernel(3.0, 3.0);
        assert!(
            val.abs() < 1e-6,
            "Lanczos at boundary should be ~0, got {val}"
        );
    }

    // ---- Gaussian kernel properties ----

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let kernel = make_gaussian_kernel(2.0);
        let sum: f64 = kernel.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "kernel should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let kernel = make_gaussian_kernel(1.5);
        let n = kernel.len();
        for i in 0..n / 2 {
            assert!(
                (kernel[i] - kernel[n - 1 - i]).abs() < 1e-10,
                "kernel should be symmetric"
            );
        }
    }

    // ---- Gaussian blur preserves flat image ----

    #[test]
    fn test_gaussian_blur_flat_image() {
        let plane = vec![100u8; 16 * 16];
        let blurred = gaussian_blur(&plane, 16, 16, 1.0);
        for &v in &blurred {
            assert!(
                (v as i32 - 100).abs() <= 1,
                "blur of flat image should be ~100, got {v}"
            );
        }
    }

    // ---- UpscaleMethod variants ----

    #[test]
    fn test_upscale_method_eq() {
        assert_eq!(UpscaleMethod::Bicubic, UpscaleMethod::Bicubic);
        assert_ne!(UpscaleMethod::Bicubic, UpscaleMethod::Lanczos3);
        assert_ne!(UpscaleMethod::NearestNeighbor, UpscaleMethod::EdgeDirected);
    }

    // ---- All methods produce valid output ----

    #[test]
    fn test_all_methods_produce_output() {
        let frame = make_gradient_frame(8, 8);
        for method in [
            UpscaleMethod::Bicubic,
            UpscaleMethod::Lanczos3,
            UpscaleMethod::NearestNeighbor,
            UpscaleMethod::EdgeDirected,
            UpscaleMethod::NediSr,
        ] {
            let config = UpscaleConfig {
                scale_factor: 2.0,
                sharpen: None,
                method,
                ..UpscaleConfig::default()
            };
            let result = upscale(&frame, 8, 8, &config)
                .unwrap_or_else(|_| panic!("{method:?} should succeed"));
            assert_eq!(result.width, 16);
            assert_eq!(result.height, 16);
            assert_eq!(result.y_plane.len(), 16 * 16);
        }
    }

    // -----------------------------------------------------------------------
    // SuperResolutionEngine tests
    // -----------------------------------------------------------------------

    fn make_luma_plane(width: u32, height: u32, val: u8) -> Vec<u8> {
        vec![val; (width * height) as usize]
    }

    fn make_gradient_luma(width: u32, height: u32) -> Vec<u8> {
        let mut plane = Vec::with_capacity((width * height) as usize);
        for _row in 0..height {
            for col in 0..width {
                let v = if width > 1 {
                    (col as f64 / (width as f64 - 1.0) * 255.0) as u8
                } else {
                    128
                };
                plane.push(v);
            }
        }
        plane
    }

    #[test]
    fn test_sr_engine_default_config() {
        let cfg = SuperResolutionConfig::default();
        assert_eq!(cfg.scale, 2);
        assert_eq!(cfg.mode, SrMode::BicubicSharp);
        assert!((cfg.sharpening_amount - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sr_engine_new_valid() {
        let cfg = SuperResolutionConfig::default();
        assert!(SuperResolutionEngine::new(cfg).is_ok());
    }

    #[test]
    fn test_sr_engine_zero_scale_error() {
        let cfg = SuperResolutionConfig {
            scale: 0,
            ..SuperResolutionConfig::default()
        };
        assert!(SuperResolutionEngine::new(cfg).is_err());
    }

    #[test]
    fn test_sr_engine_scale_17_error() {
        let cfg = SuperResolutionConfig {
            scale: 17,
            ..SuperResolutionConfig::default()
        };
        assert!(SuperResolutionEngine::new(cfg).is_err());
    }

    #[test]
    fn test_sr_engine_scale_16_valid() {
        let cfg = SuperResolutionConfig {
            scale: 16,
            ..SuperResolutionEngine::new(SuperResolutionConfig::default())
                .expect("default engine creation should succeed")
                .config()
                .clone()
        };
        assert!(SuperResolutionEngine::new(cfg).is_ok());
    }

    #[test]
    fn test_sr_engine_config_accessor() {
        let cfg = SuperResolutionConfig {
            scale: 3,
            mode: SrMode::Lanczos3Sharp,
            sharpening_amount: 0.7,
        };
        let engine =
            SuperResolutionEngine::new(cfg.clone()).expect("engine creation should succeed");
        assert_eq!(engine.config().scale, 3);
        assert_eq!(engine.config().mode, SrMode::Lanczos3Sharp);
    }

    #[test]
    fn test_sr_engine_output_dimensions_bicubic() {
        let plane = make_luma_plane(8, 6, 128);
        let cfg = SuperResolutionConfig {
            scale: 2,
            mode: SrMode::BicubicSharp,
            sharpening_amount: 0.0,
        };
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        let out = engine
            .upscale_channel(&plane, 8, 6)
            .expect("upscale should succeed");
        assert_eq!(out.len(), 16 * 12);
    }

    #[test]
    fn test_sr_engine_output_dimensions_lanczos() {
        let plane = make_luma_plane(4, 4, 100);
        let cfg = SuperResolutionConfig {
            scale: 3,
            mode: SrMode::Lanczos3Sharp,
            sharpening_amount: 0.0,
        };
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        let out = engine
            .upscale_channel(&plane, 4, 4)
            .expect("upscale should succeed");
        assert_eq!(out.len(), 12 * 12);
    }

    #[test]
    fn test_sr_engine_output_dimensions_nedi() {
        let plane = make_luma_plane(8, 8, 100);
        let cfg = SuperResolutionConfig {
            scale: 2,
            mode: SrMode::EdgeDirectedSR,
            sharpening_amount: 0.0,
        };
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        let out = engine
            .upscale_channel(&plane, 8, 8)
            .expect("upscale should succeed");
        assert_eq!(out.len(), 16 * 16);
    }

    #[test]
    fn test_sr_engine_flat_bicubic_preserves_value() {
        let plane = make_luma_plane(8, 8, 150);
        let cfg = SuperResolutionConfig {
            scale: 2,
            mode: SrMode::BicubicSharp,
            sharpening_amount: 0.0,
        };
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        let out = engine
            .upscale_channel(&plane, 8, 8)
            .expect("upscale should succeed");
        for &v in &out {
            assert!((v as i32 - 150).abs() <= 1, "expected ~150, got {v}");
        }
    }

    #[test]
    fn test_sr_engine_flat_lanczos_preserves_value() {
        let plane = make_luma_plane(8, 8, 80);
        let cfg = SuperResolutionConfig {
            scale: 2,
            mode: SrMode::Lanczos3Sharp,
            sharpening_amount: 0.0,
        };
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        let out = engine
            .upscale_channel(&plane, 8, 8)
            .expect("upscale should succeed");
        for &v in &out {
            assert!((v as i32 - 80).abs() <= 1, "expected ~80, got {v}");
        }
    }

    #[test]
    fn test_sr_engine_flat_nedi_preserves_value() {
        let plane = make_luma_plane(8, 8, 200);
        let cfg = SuperResolutionConfig {
            scale: 2,
            mode: SrMode::EdgeDirectedSR,
            sharpening_amount: 0.0,
        };
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        let out = engine
            .upscale_channel(&plane, 8, 8)
            .expect("upscale should succeed");
        for &v in &out {
            assert!((v as i32 - 200).abs() <= 2, "expected ~200, got {v}");
        }
    }

    #[test]
    fn test_sr_engine_gradient_preserved() {
        let plane = make_gradient_luma(16, 8);
        let cfg = SuperResolutionConfig {
            scale: 2,
            mode: SrMode::BicubicSharp,
            sharpening_amount: 0.0,
        };
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        let out = engine
            .upscale_channel(&plane, 16, 8)
            .expect("upscale should succeed");
        // First row of output: left side should be darker than right side.
        let left = out[0] as i32;
        let right = out[31] as i32;
        assert!(
            right > left,
            "gradient should be preserved: left={left}, right={right}"
        );
    }

    #[test]
    fn test_sr_engine_zero_width_error() {
        let plane = make_luma_plane(1, 1, 128);
        let cfg = SuperResolutionConfig::default();
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        assert!(engine.upscale_channel(&plane, 0, 8).is_err());
    }

    #[test]
    fn test_sr_engine_zero_height_error() {
        let plane = make_luma_plane(1, 1, 128);
        let cfg = SuperResolutionConfig::default();
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        assert!(engine.upscale_channel(&plane, 8, 0).is_err());
    }

    #[test]
    fn test_sr_engine_buffer_mismatch_error() {
        let plane = vec![128u8; 4]; // too small for 8x8
        let cfg = SuperResolutionConfig::default();
        let engine = SuperResolutionEngine::new(cfg).expect("engine creation should succeed");
        assert!(engine.upscale_channel(&plane, 8, 8).is_err());
    }

    #[test]
    fn test_sr_upscale_convenience_fn() {
        let plane = make_luma_plane(4, 4, 100);
        let cfg = SuperResolutionConfig {
            scale: 2,
            mode: SrMode::BicubicSharp,
            sharpening_amount: 0.0,
        };
        let out = sr_upscale(&plane, 4, 4, &cfg).expect("sr_upscale should succeed");
        assert_eq!(out.len(), 8 * 8);
    }

    #[test]
    fn test_sr_mode_equality() {
        assert_eq!(SrMode::BicubicSharp, SrMode::BicubicSharp);
        assert_ne!(SrMode::BicubicSharp, SrMode::Lanczos3Sharp);
        assert_ne!(SrMode::Lanczos3Sharp, SrMode::EdgeDirectedSR);
    }

    #[test]
    fn test_sr_config_clone() {
        let cfg = SuperResolutionConfig {
            scale: 4,
            mode: SrMode::EdgeDirectedSR,
            sharpening_amount: 0.3,
        };
        let cfg2 = cfg.clone();
        assert_eq!(cfg2.scale, 4);
        assert_eq!(cfg2.mode, SrMode::EdgeDirectedSR);
    }
}
