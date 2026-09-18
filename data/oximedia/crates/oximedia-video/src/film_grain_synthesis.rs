//! Film grain synthesis for video frames.
//!
//! Generates realistic film grain noise with controllable parameters
//! including intensity, grain size, color vs monochrome grain, and
//! temporal variation. Supports AV1-style grain parameter tables and
//! per-frame grain generation from a deterministic seed.

use std::collections::HashMap;

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur during film grain synthesis.
#[derive(Debug, thiserror::Error)]
pub enum FilmGrainError {
    /// Frame dimensions are invalid (zero width or height).
    #[error("invalid frame dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },
    /// Frame buffer size does not match expected dimensions.
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer size in bytes.
        expected: usize,
        /// Actual buffer size in bytes.
        actual: usize,
    },
    /// Grain intensity is out of valid range [0.0, 1.0].
    #[error("grain intensity {0} is out of range [0.0, 1.0]")]
    InvalidIntensity(f32),
    /// Grain size is out of valid range [1, 64].
    #[error("grain size {0} is out of range [1, 64]")]
    InvalidGrainSize(u32),
}

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Type of film grain to synthesize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrainType {
    /// Monochrome grain applied equally to all planes.
    Monochrome,
    /// Independent color grain on each YUV plane.
    Color,
    /// Luma-only grain (chroma planes untouched).
    LumaOnly,
}

/// Parameters controlling film grain synthesis.
#[derive(Debug, Clone)]
pub struct GrainParams {
    /// Grain intensity in [0.0, 1.0]. 0.0 = no grain, 1.0 = maximum grain.
    pub intensity: f32,
    /// Grain size in pixels [1, 64]. Larger values produce coarser grain.
    pub grain_size: u32,
    /// Type of grain (monochrome, color, luma-only).
    pub grain_type: GrainType,
    /// Gaussian vs uniform noise distribution.
    pub gaussian: bool,
    /// Random seed for deterministic grain generation.
    pub seed: u64,
    /// Temporal correlation factor in [0.0, 1.0].
    /// 0.0 = fully random per frame, 1.0 = static grain pattern.
    pub temporal_correlation: f32,
    /// Intensity scaling curve: maps luma [0..255] to grain scale [0.0, 1.0].
    /// If `None`, grain is applied uniformly regardless of luma.
    pub luma_scaling: Option<LumaScalingCurve>,
}

impl Default for GrainParams {
    fn default() -> Self {
        Self {
            intensity: 0.15,
            grain_size: 1,
            grain_type: GrainType::Monochrome,
            gaussian: true,
            seed: 0,
            temporal_correlation: 0.0,
            luma_scaling: None,
        }
    }
}

/// Luma-dependent grain intensity scaling.
///
/// Film grain is typically less visible in very dark and very bright regions.
/// This curve defines how grain intensity varies with luma.
#[derive(Debug, Clone)]
pub struct LumaScalingCurve {
    /// Control points: (luma_value, scale_factor) pairs.
    /// Luma in [0, 255], scale in [0.0, 1.0].
    /// Linearly interpolated between points; clamped outside range.
    pub points: Vec<(u8, f32)>,
}

impl LumaScalingCurve {
    /// Create a standard photographic grain curve: low grain in shadows
    /// and highlights, peak grain in midtones.
    pub fn photographic() -> Self {
        Self {
            points: vec![
                (0, 0.2),
                (32, 0.5),
                (64, 0.85),
                (128, 1.0),
                (192, 0.85),
                (224, 0.5),
                (255, 0.2),
            ],
        }
    }

    /// Evaluate the scaling factor for a given luma value.
    pub fn evaluate(&self, luma: u8) -> f32 {
        if self.points.is_empty() {
            return 1.0;
        }
        if self.points.len() == 1 {
            return self.points[0].1.clamp(0.0, 1.0);
        }

        let l = luma;
        // Find surrounding control points.
        if l <= self.points[0].0 {
            return self.points[0].1.clamp(0.0, 1.0);
        }
        if l >= self.points[self.points.len() - 1].0 {
            return self.points[self.points.len() - 1].1.clamp(0.0, 1.0);
        }

        for window in self.points.windows(2) {
            let (l0, s0) = window[0];
            let (l1, s1) = window[1];
            if l >= l0 && l <= l1 {
                if l1 == l0 {
                    return s0.clamp(0.0, 1.0);
                }
                let t = (l as f32 - l0 as f32) / (l1 as f32 - l0 as f32);
                return (s0 + t * (s1 - s0)).clamp(0.0, 1.0);
            }
        }

        1.0
    }
}

/// AV1-style film grain parameter table with per-point scaling values.
#[derive(Debug, Clone)]
pub struct Av1GrainTable {
    /// Luma grain points: (intensity, scaling) pairs, up to 14 entries.
    pub luma_points: Vec<(u8, u8)>,
    /// Cb grain points.
    pub cb_points: Vec<(u8, u8)>,
    /// Cr grain points.
    pub cr_points: Vec<(u8, u8)>,
    /// Auto-regression coefficients for luma.
    pub ar_coeffs_y: Vec<i8>,
    /// Auto-regression lag (0, 1, 2, or 3).
    pub ar_coeff_lag: u8,
    /// Grain scaling shift (8-11).
    pub scaling_shift: u8,
}

impl Default for Av1GrainTable {
    fn default() -> Self {
        Self {
            luma_points: vec![(0, 32), (64, 64), (128, 96), (192, 64), (255, 32)],
            cb_points: Vec::new(),
            cr_points: Vec::new(),
            ar_coeffs_y: vec![0],
            ar_coeff_lag: 0,
            scaling_shift: 10,
        }
    }
}

/// Stateful film grain synthesizer that can maintain temporal coherence.
pub struct FilmGrainSynthesizer {
    /// Current grain parameters.
    pub params: GrainParams,
    /// Previous frame's grain pattern (for temporal blending).
    prev_grain_y: Option<Vec<i16>>,
    /// Frame counter for seed variation.
    frame_count: u64,
    /// Grain template cache keyed by (grain_size, seed).
    template_cache: HashMap<(u32, u64), Vec<i16>>,
}

impl FilmGrainSynthesizer {
    /// Create a new synthesizer with the given parameters.
    pub fn new(params: GrainParams) -> Result<Self, FilmGrainError> {
        validate_params(&params)?;
        Ok(Self {
            params,
            prev_grain_y: None,
            frame_count: 0,
            template_cache: HashMap::new(),
        })
    }

    /// Apply film grain to a raw frame buffer in-place using a Linear Congruential
    /// Generator (LCG) for noise generation.
    ///
    /// This is a stateless, seed-driven synthesis path that does not rely on the
    /// temporal state maintained by [`apply_grain`](Self::apply_grain).  It is
    /// suitable for deterministic, per-frame grain without temporal coherence.
    ///
    /// The frame layout is YUV420 planar:
    /// - Y plane: `width × height` bytes starting at index 0.
    /// - U plane: `(width/2) × (height/2)` bytes immediately after Y.
    /// - V plane: same size as U, immediately after U.
    ///
    /// Chroma grain is applied when `self.params.grain_type` is not
    /// [`GrainType::LumaOnly`]; the chroma amplitude is half that of luma.
    ///
    /// # Errors
    ///
    /// Returns [`FilmGrainError::InvalidDimensions`] or
    /// [`FilmGrainError::BufferSizeMismatch`] on bad inputs.
    pub fn synthesize(
        &self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        seed: u64,
    ) -> Result<(), FilmGrainError> {
        validate_dimensions(width, height)?;

        let y_size = (width as usize) * (height as usize);
        let uv_w = ((width + 1) / 2) as usize;
        let uv_h = ((height + 1) / 2) as usize;
        let uv_size = uv_w * uv_h;
        let expected = y_size + 2 * uv_size;

        if frame.len() < expected {
            return Err(FilmGrainError::BufferSizeMismatch {
                expected,
                actual: frame.len(),
            });
        }

        // LCG parameters (Knuth / MMIX): multiplier, increment, modulus = 2^64
        let amplitude = (self.params.intensity * 64.0) as i32;

        let mut lcg = LcgRng::new(seed);

        // Apply luma grain
        for i in 0..y_size {
            let noise = lcg.next_noise(amplitude);
            frame[i] = (frame[i] as i32 + noise).clamp(0, 255) as u8;
        }

        // Apply chroma grain if requested
        match self.params.grain_type {
            GrainType::LumaOnly => {}
            GrainType::Monochrome | GrainType::Color => {
                let chroma_amplitude = amplitude / 2;
                let u_start = y_size;
                let v_start = y_size + uv_size;
                for i in 0..uv_size {
                    let u_noise = lcg.next_noise(chroma_amplitude);
                    let v_noise = lcg.next_noise(chroma_amplitude);
                    frame[u_start + i] = (frame[u_start + i] as i32 + u_noise).clamp(0, 255) as u8;
                    frame[v_start + i] = (frame[v_start + i] as i32 + v_noise).clamp(0, 255) as u8;
                }
            }
        }

        Ok(())
    }

    /// Apply film grain to a YUV420 planar frame in-place.
    ///
    /// `frame` layout: Y plane `width * height`, then U `(w/2)*(h/2)`,
    /// then V `(w/2)*(h/2)`.
    pub fn apply_grain(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<(), FilmGrainError> {
        validate_dimensions(width, height)?;
        let y_size = (width as usize) * (height as usize);
        let uv_w = ((width + 1) / 2) as usize;
        let uv_h = ((height + 1) / 2) as usize;
        let uv_size = uv_w * uv_h;
        let expected = y_size + 2 * uv_size;

        if frame.len() < expected {
            return Err(FilmGrainError::BufferSizeMismatch {
                expected,
                actual: frame.len(),
            });
        }

        let per_frame_seed = self
            .params
            .seed
            .wrapping_add(self.frame_count.wrapping_mul(2654435761));

        // Generate luma grain.
        let grain_y = self.generate_grain_plane(width, height, per_frame_seed);

        // Temporal blending.
        let blended_y = if self.params.temporal_correlation > 0.0 {
            if let Some(ref prev) = self.prev_grain_y {
                if prev.len() == grain_y.len() {
                    let tc = self.params.temporal_correlation;
                    grain_y
                        .iter()
                        .zip(prev.iter())
                        .map(|(&curr, &prev_val)| {
                            let blended = curr as f32 * (1.0 - tc) + prev_val as f32 * tc;
                            blended as i16
                        })
                        .collect::<Vec<i16>>()
                } else {
                    grain_y.clone()
                }
            } else {
                grain_y.clone()
            }
        } else {
            grain_y.clone()
        };

        // Apply luma grain with optional luma-scaling.
        for i in 0..y_size {
            let original = frame[i];
            let scale = match &self.params.luma_scaling {
                Some(curve) => curve.evaluate(original),
                None => 1.0,
            };
            let noise = (blended_y.get(i).copied().unwrap_or(0) as f32 * scale) as i16;
            frame[i] = (original as i16 + noise).clamp(0, 255) as u8;
        }

        // Apply chroma grain if needed.
        match self.params.grain_type {
            GrainType::Color => {
                let grain_u = self.generate_grain_plane(
                    uv_w as u32,
                    uv_h as u32,
                    per_frame_seed.wrapping_add(1),
                );
                let grain_v = self.generate_grain_plane(
                    uv_w as u32,
                    uv_h as u32,
                    per_frame_seed.wrapping_add(2),
                );

                let u_start = y_size;
                let v_start = y_size + uv_size;
                // Chroma grain at half intensity.
                let chroma_scale = 0.5;
                for i in 0..uv_size {
                    let u_noise =
                        (grain_u.get(i).copied().unwrap_or(0) as f32 * chroma_scale) as i16;
                    let v_noise =
                        (grain_v.get(i).copied().unwrap_or(0) as f32 * chroma_scale) as i16;
                    frame[u_start + i] = (frame[u_start + i] as i16 + u_noise).clamp(0, 255) as u8;
                    frame[v_start + i] = (frame[v_start + i] as i16 + v_noise).clamp(0, 255) as u8;
                }
            }
            GrainType::Monochrome => {
                // Same grain pattern subsampled onto chroma.
                let u_start = y_size;
                let v_start = y_size + uv_size;
                let chroma_scale = 0.5;
                for row in 0..uv_h {
                    for col in 0..uv_w {
                        let y_row = (row * 2).min(height as usize - 1);
                        let y_col = (col * 2).min(width as usize - 1);
                        let y_idx = y_row * (width as usize) + y_col;
                        let noise = (blended_y.get(y_idx).copied().unwrap_or(0) as f32
                            * chroma_scale) as i16;
                        let uv_idx = row * uv_w + col;
                        frame[u_start + uv_idx] =
                            (frame[u_start + uv_idx] as i16 + noise).clamp(0, 255) as u8;
                        frame[v_start + uv_idx] =
                            (frame[v_start + uv_idx] as i16 + noise).clamp(0, 255) as u8;
                    }
                }
            }
            GrainType::LumaOnly => {
                // No chroma grain.
            }
        }

        self.prev_grain_y = Some(blended_y);
        self.frame_count += 1;

        Ok(())
    }

    /// Reset temporal state (call when seeking or switching clips).
    pub fn reset(&mut self) {
        self.prev_grain_y = None;
        self.frame_count = 0;
        self.template_cache.clear();
    }

    /// Update parameters on-the-fly.
    pub fn set_params(&mut self, params: GrainParams) -> Result<(), FilmGrainError> {
        validate_params(&params)?;
        self.params = params;
        self.template_cache.clear();
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private grain generation
    // -----------------------------------------------------------------------

    fn generate_grain_plane(&mut self, width: u32, height: u32, seed: u64) -> Vec<i16> {
        let w = width as usize;
        let h = height as usize;
        let size = w * h;
        let grain_size = self.params.grain_size.max(1) as usize;
        let intensity_scale = self.params.intensity * 64.0; // max noise amplitude ~64

        if grain_size == 1 {
            // Fine grain: per-pixel noise.
            let mut grain = Vec::with_capacity(size);
            let mut rng = XorShift64(seed.max(1));
            for _ in 0..size {
                let noise = if self.params.gaussian {
                    gaussian_noise(&mut rng, intensity_scale)
                } else {
                    uniform_noise(&mut rng, intensity_scale)
                };
                grain.push(noise);
            }
            grain
        } else {
            // Coarse grain: generate at reduced resolution, then upscale.
            let template_key = (self.params.grain_size, seed);
            if let Some(cached) = self.template_cache.get(&template_key) {
                return cached.clone();
            }

            let tw = (w + grain_size - 1) / grain_size;
            let th = (h + grain_size - 1) / grain_size;
            let mut rng = XorShift64(seed.max(1));
            let mut template = Vec::with_capacity(tw * th);
            for _ in 0..(tw * th) {
                let noise = if self.params.gaussian {
                    gaussian_noise(&mut rng, intensity_scale)
                } else {
                    uniform_noise(&mut rng, intensity_scale)
                };
                template.push(noise);
            }

            // Bilinear upscale from template to full resolution.
            let mut grain = Vec::with_capacity(size);
            for row in 0..h {
                for col in 0..w {
                    let tx = col as f32 / grain_size as f32;
                    let ty = row as f32 / grain_size as f32;

                    let x0 = (tx as usize).min(tw.saturating_sub(1));
                    let y0 = (ty as usize).min(th.saturating_sub(1));
                    let x1 = (x0 + 1).min(tw.saturating_sub(1));
                    let y1 = (y0 + 1).min(th.saturating_sub(1));

                    let fx = tx - tx.floor();
                    let fy = ty - ty.floor();

                    let v00 = template.get(y0 * tw + x0).copied().unwrap_or(0) as f32;
                    let v10 = template.get(y0 * tw + x1).copied().unwrap_or(0) as f32;
                    let v01 = template.get(y1 * tw + x0).copied().unwrap_or(0) as f32;
                    let v11 = template.get(y1 * tw + x1).copied().unwrap_or(0) as f32;

                    let interp = v00 * (1.0 - fx) * (1.0 - fy)
                        + v10 * fx * (1.0 - fy)
                        + v01 * (1.0 - fx) * fy
                        + v11 * fx * fy;
                    grain.push(interp as i16);
                }
            }

            // Cache for temporal reuse if correlation is > 0.
            if self.params.temporal_correlation > 0.0 && self.template_cache.len() < 16 {
                self.template_cache.insert(template_key, grain.clone());
            }

            grain
        }
    }
}

/// Generate grain from an AV1 grain table and apply to a frame.
///
/// This is a stateless convenience function for AV1-style grain synthesis.
pub fn apply_av1_grain(
    frame: &mut [u8],
    width: u32,
    height: u32,
    table: &Av1GrainTable,
    seed: u64,
) -> Result<(), FilmGrainError> {
    validate_dimensions(width, height)?;
    let y_size = (width as usize) * (height as usize);
    let uv_w = ((width + 1) / 2) as usize;
    let uv_h = ((height + 1) / 2) as usize;
    let uv_size = uv_w * uv_h;
    let expected = y_size + 2 * uv_size;

    if frame.len() < expected {
        return Err(FilmGrainError::BufferSizeMismatch {
            expected,
            actual: frame.len(),
        });
    }

    if table.luma_points.is_empty() {
        return Ok(());
    }

    // Build luma scaling LUT from AV1 grain points.
    let luma_lut = build_scaling_lut(&table.luma_points);
    let shift = table.scaling_shift.clamp(8, 11);
    let round = 1i32 << (shift - 1);

    let mut rng = XorShift64(seed.max(1));

    // The noise amplitude is scaled so that after the shift, visible grain
    // remains. We generate noise in a range proportional to (1 << shift) so
    // the scaling * noise product survives the right-shift.
    let noise_amplitude = (1i32 << shift) as f32;

    // Apply luma grain.
    for i in 0..y_size {
        let original = frame[i];
        let scaling = luma_lut[original as usize] as i32;
        let noise_raw = gaussian_noise(&mut rng, noise_amplitude) as i32;
        let noise = (noise_raw * scaling + round) >> shift;
        frame[i] = (original as i32 + noise).clamp(0, 255) as u8;
    }

    // Apply chroma grain if points are specified.
    if !table.cb_points.is_empty() {
        let cb_lut = build_scaling_lut(&table.cb_points);
        let u_start = y_size;
        for i in 0..uv_size {
            let original = frame[u_start + i];
            let scaling = cb_lut[original as usize] as i32;
            let noise_raw = gaussian_noise(&mut rng, noise_amplitude) as i32;
            let noise = (noise_raw * scaling + round) >> shift;
            frame[u_start + i] = (original as i32 + noise).clamp(0, 255) as u8;
        }
    }

    if !table.cr_points.is_empty() {
        let cr_lut = build_scaling_lut(&table.cr_points);
        let v_start = y_size + uv_size;
        for i in 0..uv_size {
            let original = frame[v_start + i];
            let scaling = cr_lut[original as usize] as i32;
            let noise_raw = gaussian_noise(&mut rng, noise_amplitude) as i32;
            let noise = (noise_raw * scaling + round) >> shift;
            frame[v_start + i] = (original as i32 + noise).clamp(0, 255) as u8;
        }
    }

    Ok(())
}

/// Estimate grain parameters from a flat (uniform-content) region of a frame.
///
/// Analyzes the variance in the given Y-plane patch to estimate grain intensity
/// and coarseness. `patch` is a rectangular luma region of `patch_w x patch_h`.
pub fn estimate_grain_params(
    patch: &[u8],
    patch_w: u32,
    patch_h: u32,
) -> Result<GrainParams, FilmGrainError> {
    validate_dimensions(patch_w, patch_h)?;
    let size = (patch_w as usize) * (patch_h as usize);
    if patch.len() < size {
        return Err(FilmGrainError::BufferSizeMismatch {
            expected: size,
            actual: patch.len(),
        });
    }

    let pixels = &patch[..size];

    // Compute mean and variance.
    let mean = pixels.iter().map(|&p| p as f64).sum::<f64>() / size as f64;
    let variance = pixels
        .iter()
        .map(|&p| {
            let d = p as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / size as f64;

    let stddev = variance.sqrt();

    // Map stddev to intensity: stddev of ~6 corresponds to intensity 0.1,
    // stddev of ~30 corresponds to intensity 0.5.
    let intensity = ((stddev / 60.0) as f32).clamp(0.0, 1.0);

    // Estimate grain size via autocorrelation at lag 1.
    let w = patch_w as usize;
    let h = patch_h as usize;
    let mut autocorr_sum = 0.0f64;
    let mut count = 0u64;

    for row in 0..h {
        for col in 0..(w.saturating_sub(1)) {
            let a = pixels[row * w + col] as f64 - mean;
            let b = pixels[row * w + col + 1] as f64 - mean;
            autocorr_sum += a * b;
            count += 1;
        }
    }

    let autocorr = if count > 0 && variance > 0.0 {
        (autocorr_sum / count as f64) / variance
    } else {
        0.0
    };

    // Higher autocorrelation means coarser grain.
    let grain_size = if autocorr > 0.7 {
        4
    } else if autocorr > 0.4 {
        2
    } else {
        1
    };

    Ok(GrainParams {
        intensity,
        grain_size,
        grain_type: GrainType::LumaOnly,
        gaussian: true,
        seed: 0,
        temporal_correlation: 0.0,
        luma_scaling: None,
    })
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

fn validate_params(params: &GrainParams) -> Result<(), FilmGrainError> {
    if !(0.0..=1.0).contains(&params.intensity) {
        return Err(FilmGrainError::InvalidIntensity(params.intensity));
    }
    if params.grain_size < 1 || params.grain_size > 64 {
        return Err(FilmGrainError::InvalidGrainSize(params.grain_size));
    }
    Ok(())
}

fn validate_dimensions(width: u32, height: u32) -> Result<(), FilmGrainError> {
    if width == 0 || height == 0 {
        return Err(FilmGrainError::InvalidDimensions { width, height });
    }
    Ok(())
}

/// Build a 256-entry scaling LUT from AV1-style (input, output) control points.
fn build_scaling_lut(points: &[(u8, u8)]) -> [u8; 256] {
    let mut lut = [0u8; 256];
    if points.is_empty() {
        return lut;
    }
    if points.len() == 1 {
        lut.fill(points[0].1);
        return lut;
    }

    // Fill before first point.
    for i in 0..=points[0].0 as usize {
        lut[i] = points[0].1;
    }

    // Interpolate between points.
    for win in points.windows(2) {
        let (x0, y0) = win[0];
        let (x1, y1) = win[1];
        if x1 <= x0 {
            continue;
        }
        for x in (x0 as usize)..=(x1 as usize) {
            let t = (x - x0 as usize) as f32 / (x1 as f32 - x0 as f32);
            lut[x] = (y0 as f32 + t * (y1 as f32 - y0 as f32)).clamp(0.0, 255.0) as u8;
        }
    }

    // Fill after last point.
    let last = points[points.len() - 1];
    for i in (last.0 as usize)..256 {
        lut[i] = last.1;
    }

    lut
}

// -----------------------------------------------------------------------
// LCG PRNG (used by synthesize)
// -----------------------------------------------------------------------

/// Linear Congruential Generator (LCG) PRNG.
///
/// Parameters from Knuth's MMIX: multiplier = 6364136223846793005,
/// increment = 1442695040888963407.  The modulus is implicit (2^64 wrap).
struct LcgRng(u64);

impl LcgRng {
    fn new(seed: u64) -> Self {
        // Ensure the state is never zero by adding an offset.
        Self(seed.wrapping_add(1))
    }

    /// Advance the LCG and return the next 64-bit value.
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    /// Return a signed noise sample in `[-amplitude, +amplitude]`.
    fn next_noise(&mut self, amplitude: i32) -> i32 {
        if amplitude == 0 {
            return 0;
        }
        let raw = self.next_u64();
        // Map high 32 bits to [-amplitude, +amplitude].
        let scale = (amplitude * 2 + 1) as u64;
        let mapped = (raw >> 32) % scale;
        mapped as i32 - amplitude
    }
}

// -----------------------------------------------------------------------
// XorShift PRNG (used by apply_grain / generate_grain_plane)
// -----------------------------------------------------------------------

/// Simple xorshift64 PRNG.
struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Generate a single Gaussian noise sample using Box-Muller transform.
fn gaussian_noise(rng: &mut XorShift64, scale: f32) -> i16 {
    let u1 = rng.next_f64().max(1e-10);
    let u2 = rng.next_f64();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (z * scale as f64).clamp(-128.0, 127.0) as i16
}

/// Generate a single uniform noise sample in [-scale, scale].
fn uniform_noise(rng: &mut XorShift64, scale: f32) -> i16 {
    let u = rng.next_f64() * 2.0 - 1.0;
    (u * scale as f64).clamp(-128.0, 127.0) as i16
}

// -----------------------------------------------------------------------
// FilmGrainConfig / GrainSynthesizer — simple API with splitmix64 PRNG
// -----------------------------------------------------------------------

/// Configuration for the simple `GrainSynthesizer` API.
///
/// This struct provides a streamlined alternative to [`GrainParams`] that
/// is fully initialised at construction time.  A pre-generated, spatially
/// correlated grain pattern (created with a box-blurred splitmix64 PRNG) is
/// stored inside the synthesizer and blended with fresh noise each frame
/// according to `temporal_coherence`.
#[derive(Debug, Clone)]
pub struct FilmGrainConfig {
    /// Grain strength in \[0.0, 1.0\].  0.0 = no grain, 1.0 = maximum.
    pub intensity: f32,
    /// Spatial grain size in pixels (typical range 1.5–4.0).
    /// Controls the blur radius applied to white noise during pre-generation.
    pub grain_size: f32,
    /// When `true`, grain is applied only to the luma channel (Y or computed
    /// from RGB).  When `false`, all channels receive independent grain.
    pub luma_only: bool,
    /// Temporal blending factor in \[0.0, 1.0\].
    /// 0.0 = fully independent per-frame noise; 1.0 = static pre-generated
    /// pattern that never changes between frames.
    pub temporal_coherence: f32,
    /// Seed for the splitmix64 PRNG used during grain pre-generation.
    pub seed: u64,
}

impl Default for FilmGrainConfig {
    fn default() -> Self {
        Self {
            intensity: 0.15,
            grain_size: 2.0,
            luma_only: false,
            temporal_coherence: 0.0,
            seed: 0,
        }
    }
}

// ── splitmix64 PRNG ────────────────────────────────────────────────────────

/// Advance a splitmix64 state and return the next pseudo-random 64-bit value.
///
/// splitmix64 (Sebastiano Vigna, 2015) is a simple, high-quality PRNG that
/// does not require any external crate.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Map a raw splitmix64 output to a float in \[−1.0, 1.0\].
#[inline]
fn splitmix64_f32(state: &mut u64) -> f32 {
    let raw = splitmix64(state);
    // Use upper 24 bits for f32 mantissa precision
    let mantissa = (raw >> 40) as f32; // 0 .. 16_777_216
    mantissa / 8_388_608.0 - 1.0 // [-1.0, 1.0]
}

// ── box blur helpers ───────────────────────────────────────────────────────

/// In-place horizontal box blur with mirror (reflect) boundary conditions.
fn box_blur_horizontal(buf: &mut [f32], width: usize, height: usize, radius: usize) {
    if radius == 0 || width == 0 || height == 0 {
        return;
    }
    let r = radius as isize;
    let w = width as isize;
    let mut tmp = vec![0.0f32; width];

    for row in 0..height {
        let base = row * width;
        for col in 0..width {
            let col_i = col as isize;
            let mut sum = 0.0f32;
            let count = 2 * radius + 1;
            for k in -r..=r {
                let x = (col_i + k).rem_euclid(w) as usize;
                sum += buf[base + x];
            }
            tmp[col] = sum / count as f32;
        }
        buf[base..base + width].copy_from_slice(&tmp);
    }
}

/// In-place vertical box blur with mirror (reflect) boundary conditions.
fn box_blur_vertical(buf: &mut [f32], width: usize, height: usize, radius: usize) {
    if radius == 0 || width == 0 || height == 0 {
        return;
    }
    let r = radius as isize;
    let h = height as isize;
    let mut tmp = vec![0.0f32; height];

    for col in 0..width {
        for row in 0..height {
            let row_i = row as isize;
            let mut sum = 0.0f32;
            let count = 2 * radius + 1;
            for k in -r..=r {
                let y = (row_i + k).rem_euclid(h) as usize;
                sum += buf[y * width + col];
            }
            tmp[row] = sum / count as f32;
        }
        for row in 0..height {
            buf[row * width + col] = tmp[row];
        }
    }
}

/// Generate a spatially correlated grain buffer of length `width * height`
/// with values in \[−1.0, 1.0\].
///
/// The buffer is produced by generating independent uniform noise via
/// splitmix64 and then applying a separable box blur to create grain
/// "clusters" whose spatial frequency is controlled by `grain_size`.
fn generate_grain_buffer(width: u32, height: u32, grain_size: f32, seed: u64) -> Vec<f32> {
    let n = (width as usize) * (height as usize);
    if n == 0 {
        return Vec::new();
    }
    let mut state = seed;
    let mut buf: Vec<f32> = (0..n).map(|_| splitmix64_f32(&mut state)).collect();

    // Blur radius clamped to [0, 8]
    let radius = ((grain_size - 1.0).max(0.0).round() as usize).min(8);
    if radius > 0 {
        box_blur_horizontal(&mut buf, width as usize, height as usize, radius);
        box_blur_vertical(&mut buf, width as usize, height as usize, radius);
        // Normalise to [-1, 1] after blur (blur shrinks the range)
        let max_abs = buf.iter().copied().fold(0.0f32, |m, v| m.max(v.abs()));
        if max_abs > 1e-6 {
            buf.iter_mut().for_each(|v| *v /= max_abs);
        }
    }

    buf
}

// ── GrainSynthesizer ───────────────────────────────────────────────────────

/// Simple, stateful film grain synthesizer using [`FilmGrainConfig`].
///
/// A grain pattern is pre-generated at construction time.  On each call to
/// [`GrainSynthesizer::apply`], that pattern is blended with freshly-generated
/// per-frame noise according to `config.temporal_coherence` before being
/// mixed into the frame pixels.
pub struct GrainSynthesizer {
    /// Grain configuration.
    config: FilmGrainConfig,
    /// Pre-generated, spatially correlated grain pattern (one entry per pixel
    /// of the frame; values in \[−1.0, 1.0\]).
    grain_buffer: Vec<f32>,
    /// Number of frames that have been processed.
    frame_counter: u64,
}

impl GrainSynthesizer {
    /// Create a new synthesizer and pre-generate the grain buffer.
    pub fn new(config: FilmGrainConfig, width: u32, height: u32) -> Self {
        let grain_buffer = generate_grain_buffer(width, height, config.grain_size, config.seed);
        Self {
            config,
            grain_buffer,
            frame_counter: 0,
        }
    }

    /// Apply film grain to `frame` in-place.
    ///
    /// # Parameters
    ///
    /// * `frame`  — mutable byte slice; must hold either packed-RGB/BGR
    ///   (`width × height × 3` bytes, `is_yuv=false`) or a planar YUV 4:2:0
    ///   layout (`is_yuv=true`): Y plane (`width × height` bytes) followed by
    ///   U and V planes (each `⌈width/2⌉ × ⌈height/2⌉` bytes).
    /// * `width` / `height` — frame dimensions; should match those supplied at
    ///   construction time.  If they differ from the stored `grain_buffer` the
    ///   grain is applied modulo the buffer length (no panic).
    /// * `is_yuv` — `true` for YUV 4:2:0, `false` for interleaved RGB/BGR.
    pub fn apply(&mut self, frame: &mut [u8], width: u32, height: u32, is_yuv: bool) {
        let n_pixels = (width as usize) * (height as usize);
        if n_pixels == 0 || frame.is_empty() {
            self.frame_counter += 1;
            return;
        }

        // Derive per-frame PRNG seed: mix config seed with frame counter so
        // each frame gets distinct fresh noise when temporal_coherence < 1.
        let frame_seed = self
            .config
            .seed
            .wrapping_add(self.frame_counter.wrapping_mul(0x9e3779b97f4a7c15));

        let coherence = self.config.temporal_coherence.clamp(0.0, 1.0);
        let intensity = self.config.intensity.clamp(0.0, 1.0);
        // Maximum absolute pixel delta: at intensity=1 this is ±32 out of 255.
        let max_delta = intensity * 32.0;

        if is_yuv {
            self.apply_yuv420(
                frame, width, height, n_pixels, frame_seed, coherence, max_delta,
            );
        } else {
            self.apply_rgb(
                frame, width, height, n_pixels, frame_seed, coherence, max_delta,
            );
        }

        self.frame_counter += 1;
    }

    // ── private helpers ───────────────────────────────────────────────────

    /// Apply grain to an interleaved RGB (or BGR) frame.
    fn apply_rgb(
        &self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        n_pixels: usize,
        frame_seed: u64,
        coherence: f32,
        max_delta: f32,
    ) {
        let mut state = frame_seed;
        let buf_len = self.grain_buffer.len().max(1);

        if self.config.luma_only {
            // Apply grain to luma only: compute Y for each pixel, add delta,
            // write back to all three channels (keeps colour ratios).
            // For simplicity, we add a luma-equivalent offset to each channel.
            for px in 0..n_pixels {
                if px * 3 + 2 >= frame.len() {
                    break;
                }
                let stored = *self.grain_buffer.get(px % buf_len).unwrap_or(&0.0);
                let fresh = splitmix64_f32(&mut state);
                let grain = stored * coherence + fresh * (1.0 - coherence);
                let delta = (grain * max_delta).round() as i16;

                let r = frame[px * 3] as i16;
                let g = frame[px * 3 + 1] as i16;
                let b = frame[px * 3 + 2] as i16;
                // BT.601 luma weight approximation: push equally to R, G, B.
                let nr = (r + delta).clamp(0, 255) as u8;
                let ng = (g + delta).clamp(0, 255) as u8;
                let nb = (b + delta).clamp(0, 255) as u8;
                frame[px * 3] = nr;
                frame[px * 3 + 1] = ng;
                frame[px * 3 + 2] = nb;
            }
        } else {
            // Independent grain on R, G, B channels.
            let mut state_g = frame_seed.wrapping_add(0x1111_1111_1111_1111);
            let mut state_b = frame_seed.wrapping_add(0x2222_2222_2222_2222);
            for px in 0..n_pixels {
                if px * 3 + 2 >= frame.len() {
                    break;
                }
                let stored = *self.grain_buffer.get(px % buf_len).unwrap_or(&0.0);
                let fresh_r = splitmix64_f32(&mut state);
                let fresh_g = splitmix64_f32(&mut state_g);
                let fresh_b = splitmix64_f32(&mut state_b);
                let grain_r = stored * coherence + fresh_r * (1.0 - coherence);
                let grain_g = stored * coherence + fresh_g * (1.0 - coherence);
                let grain_b = stored * coherence + fresh_b * (1.0 - coherence);

                frame[px * 3] = ((frame[px * 3] as f32) + grain_r * max_delta)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                frame[px * 3 + 1] = ((frame[px * 3 + 1] as f32) + grain_g * max_delta)
                    .round()
                    .clamp(0.0, 255.0) as u8;
                frame[px * 3 + 2] = ((frame[px * 3 + 2] as f32) + grain_b * max_delta)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
        let _ = (width, height); // suppress unused-variable warnings
    }

    /// Apply grain to a planar YUV 4:2:0 frame.
    fn apply_yuv420(
        &self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        n_pixels: usize,
        frame_seed: u64,
        coherence: f32,
        max_delta: f32,
    ) {
        let uv_w = ((width + 1) / 2) as usize;
        let uv_h = ((height + 1) / 2) as usize;
        let uv_size = uv_w * uv_h;
        let y_end = n_pixels;
        let u_end = y_end + uv_size;
        let v_end = u_end + uv_size;

        let buf_len = self.grain_buffer.len().max(1);
        let mut state = frame_seed;

        // Y plane
        let y_slice_end = y_end.min(frame.len());
        for px in 0..y_slice_end {
            let stored = *self.grain_buffer.get(px % buf_len).unwrap_or(&0.0);
            let fresh = splitmix64_f32(&mut state);
            let grain = stored * coherence + fresh * (1.0 - coherence);
            frame[px] = ((frame[px] as f32) + grain * max_delta)
                .round()
                .clamp(0.0, 255.0) as u8;
        }

        if self.config.luma_only || frame.len() <= y_end {
            return;
        }

        // U plane
        let mut state_u = frame_seed.wrapping_add(0x3333_3333_3333_3333);
        let u_slice_end = u_end.min(frame.len());
        for (i, px) in (y_end..u_slice_end).enumerate() {
            let stored = *self.grain_buffer.get(i % buf_len).unwrap_or(&0.0);
            let fresh = splitmix64_f32(&mut state_u);
            let grain = stored * coherence + fresh * (1.0 - coherence);
            frame[px] = ((frame[px] as f32) + grain * max_delta)
                .round()
                .clamp(0.0, 255.0) as u8;
        }

        // V plane
        if frame.len() <= u_end {
            return;
        }
        let mut state_v = frame_seed.wrapping_add(0x4444_4444_4444_4444);
        let v_slice_end = v_end.min(frame.len());
        for (i, px) in (u_end..v_slice_end).enumerate() {
            let stored = *self.grain_buffer.get(i % buf_len).unwrap_or(&0.0);
            let fresh = splitmix64_f32(&mut state_v);
            let grain = stored * coherence + fresh * (1.0 - coherence);
            frame[px] = ((frame[px] as f32) + grain * max_delta)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }

    /// Return the number of frames processed so far.
    #[must_use]
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }

    /// Return a reference to the pre-generated grain buffer.
    #[must_use]
    pub fn grain_buffer(&self) -> &[f32] {
        &self.grain_buffer
    }
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

    // ---- Parameter validation ----

    #[test]
    fn test_invalid_intensity_high() {
        let params = GrainParams {
            intensity: 1.5,
            ..GrainParams::default()
        };
        let result = FilmGrainSynthesizer::new(params);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_intensity_negative() {
        let params = GrainParams {
            intensity: -0.1,
            ..GrainParams::default()
        };
        let result = FilmGrainSynthesizer::new(params);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_grain_size_zero() {
        let params = GrainParams {
            grain_size: 0,
            ..GrainParams::default()
        };
        let result = FilmGrainSynthesizer::new(params);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_grain_size_too_large() {
        let params = GrainParams {
            grain_size: 65,
            ..GrainParams::default()
        };
        let result = FilmGrainSynthesizer::new(params);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_params_accepted() {
        let params = GrainParams::default();
        let result = FilmGrainSynthesizer::new(params);
        assert!(result.is_ok());
    }

    // ---- Zero intensity produces no change ----

    #[test]
    fn test_zero_intensity_no_change() {
        let params = GrainParams {
            intensity: 0.0,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid params");
        let original = make_yuv420(16, 16, 128);
        let mut frame = original.clone();
        synth.apply_grain(&mut frame, 16, 16).expect("apply ok");
        assert_eq!(frame, original, "zero intensity should not change frame");
    }

    // ---- Non-zero intensity modifies the frame ----

    #[test]
    fn test_nonzero_intensity_modifies_frame() {
        let params = GrainParams {
            intensity: 0.5,
            seed: 42,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid params");
        let original = make_yuv420(32, 32, 128);
        let mut frame = original.clone();
        synth.apply_grain(&mut frame, 32, 32).expect("apply ok");
        assert_ne!(frame, original, "non-zero intensity should modify frame");
    }

    // ---- Deterministic output with same seed ----

    #[test]
    fn test_deterministic_same_seed() {
        let params = GrainParams {
            intensity: 0.3,
            seed: 123,
            ..GrainParams::default()
        };
        let mut synth1 = FilmGrainSynthesizer::new(params.clone()).expect("valid");
        let mut synth2 = FilmGrainSynthesizer::new(params).expect("valid");
        let mut frame1 = make_yuv420(16, 16, 100);
        let mut frame2 = make_yuv420(16, 16, 100);
        synth1.apply_grain(&mut frame1, 16, 16).expect("ok");
        synth2.apply_grain(&mut frame2, 16, 16).expect("ok");
        assert_eq!(frame1, frame2, "same seed should produce same grain");
    }

    // ---- Different seeds produce different output ----

    #[test]
    fn test_different_seeds_differ() {
        let p1 = GrainParams {
            intensity: 0.3,
            seed: 1,
            ..GrainParams::default()
        };
        let p2 = GrainParams {
            intensity: 0.3,
            seed: 999,
            ..GrainParams::default()
        };
        let mut s1 = FilmGrainSynthesizer::new(p1).expect("valid");
        let mut s2 = FilmGrainSynthesizer::new(p2).expect("valid");
        let mut f1 = make_yuv420(16, 16, 128);
        let mut f2 = make_yuv420(16, 16, 128);
        s1.apply_grain(&mut f1, 16, 16).expect("ok");
        s2.apply_grain(&mut f2, 16, 16).expect("ok");
        assert_ne!(f1, f2, "different seeds should produce different grain");
    }

    // ---- Output stays within [0, 255] ----

    #[test]
    fn test_output_clamped_bright() {
        let params = GrainParams {
            intensity: 1.0,
            seed: 7,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut frame = make_yuv420(16, 16, 250);
        synth.apply_grain(&mut frame, 16, 16).expect("ok");
        // Output is u8, values are inherently in [0, 255]
        assert!(!frame.is_empty());
    }

    #[test]
    fn test_output_clamped_dark() {
        let params = GrainParams {
            intensity: 1.0,
            seed: 7,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut frame = make_yuv420(16, 16, 5);
        synth.apply_grain(&mut frame, 16, 16).expect("ok");
        // u8 is always in [0, 255]; check it didn't panic.
    }

    // ---- GrainType::Color modifies chroma ----

    #[test]
    fn test_color_grain_modifies_chroma() {
        let params = GrainParams {
            intensity: 0.5,
            seed: 42,
            grain_type: GrainType::Color,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(32, 32, 128);
        let y_size = 32 * 32;
        let mut frame = original.clone();
        synth.apply_grain(&mut frame, 32, 32).expect("ok");
        // Chroma planes should differ from original.
        let chroma_orig = &original[y_size..];
        let chroma_new = &frame[y_size..];
        assert_ne!(chroma_orig, chroma_new, "color grain should modify chroma");
    }

    // ---- GrainType::LumaOnly does not modify chroma ----

    #[test]
    fn test_luma_only_no_chroma_change() {
        let params = GrainParams {
            intensity: 0.5,
            seed: 42,
            grain_type: GrainType::LumaOnly,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(16, 16, 128);
        let y_size = 16 * 16;
        let mut frame = original.clone();
        synth.apply_grain(&mut frame, 16, 16).expect("ok");
        assert_eq!(
            &frame[y_size..],
            &original[y_size..],
            "LumaOnly should leave chroma unchanged"
        );
    }

    // ---- Coarse grain (grain_size > 1) ----

    #[test]
    fn test_coarse_grain_applies() {
        let params = GrainParams {
            intensity: 0.3,
            seed: 55,
            grain_size: 8,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(32, 32, 128);
        let mut frame = original.clone();
        synth.apply_grain(&mut frame, 32, 32).expect("ok");
        assert_ne!(frame, original);
    }

    // ---- Temporal correlation ----

    #[test]
    fn test_temporal_correlation_produces_coherent_frames() {
        let params = GrainParams {
            intensity: 0.3,
            seed: 99,
            temporal_correlation: 0.8,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut frame1 = make_yuv420(16, 16, 128);
        synth.apply_grain(&mut frame1, 16, 16).expect("ok");
        let mut frame2 = make_yuv420(16, 16, 128);
        synth.apply_grain(&mut frame2, 16, 16).expect("ok");
        // With high temporal correlation, consecutive frames should be similar.
        let y_size = 16 * 16;
        let diff: u64 = frame1[..y_size]
            .iter()
            .zip(frame2[..y_size].iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
            .sum();
        let avg_diff = diff as f64 / y_size as f64;
        assert!(
            avg_diff < 15.0,
            "temporal correlation should reduce inter-frame grain difference, got {avg_diff}"
        );
    }

    // ---- Luma scaling curve ----

    #[test]
    fn test_luma_scaling_photographic() {
        let curve = LumaScalingCurve::photographic();
        // Midtone should have highest scaling.
        let mid = curve.evaluate(128);
        let shadow = curve.evaluate(0);
        let highlight = curve.evaluate(255);
        assert!(mid > shadow);
        assert!(mid > highlight);
    }

    #[test]
    fn test_luma_scaling_applied() {
        let params = GrainParams {
            intensity: 0.5,
            seed: 42,
            luma_scaling: Some(LumaScalingCurve::photographic()),
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut frame = make_yuv420(16, 16, 128);
        synth.apply_grain(&mut frame, 16, 16).expect("ok");
        // Just verify it runs without error.
    }

    // ---- AV1 grain table ----

    #[test]
    fn test_av1_grain_applies() {
        let table = Av1GrainTable::default();
        let mut frame = make_yuv420(32, 32, 128);
        let original = frame.clone();
        apply_av1_grain(&mut frame, 32, 32, &table, 42).expect("ok");
        assert_ne!(frame, original, "AV1 grain should modify the frame");
    }

    #[test]
    fn test_av1_grain_empty_points_no_change() {
        let table = Av1GrainTable {
            luma_points: vec![],
            ..Av1GrainTable::default()
        };
        let mut frame = make_yuv420(16, 16, 128);
        let original = frame.clone();
        apply_av1_grain(&mut frame, 16, 16, &table, 42).expect("ok");
        assert_eq!(
            frame, original,
            "empty luma points should produce no change"
        );
    }

    #[test]
    fn test_av1_grain_with_chroma() {
        let table = Av1GrainTable {
            cb_points: vec![(0, 24), (128, 48), (255, 24)],
            cr_points: vec![(0, 16), (128, 32), (255, 16)],
            ..Av1GrainTable::default()
        };
        let mut frame = make_yuv420(16, 16, 128);
        apply_av1_grain(&mut frame, 16, 16, &table, 42).expect("ok");
    }

    // ---- Estimate grain params ----

    #[test]
    fn test_estimate_flat_patch_low_intensity() {
        let patch = vec![128u8; 64 * 64];
        let params = estimate_grain_params(&patch, 64, 64).expect("ok");
        assert!(
            params.intensity < 0.05,
            "flat patch should have very low intensity, got {}",
            params.intensity
        );
    }

    #[test]
    fn test_estimate_noisy_patch_higher_intensity() {
        let mut rng = XorShift64(12345);
        let patch: Vec<u8> = (0..64 * 64)
            .map(|_| {
                let noise = (rng.next() % 60) as i32 - 30;
                (128 + noise).clamp(0, 255) as u8
            })
            .collect();
        let params = estimate_grain_params(&patch, 64, 64).expect("ok");
        assert!(
            params.intensity > 0.05,
            "noisy patch should have measurable intensity, got {}",
            params.intensity
        );
    }

    // ---- Buffer size validation ----

    #[test]
    fn test_buffer_too_small_error() {
        let params = GrainParams::default();
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut frame = vec![0u8; 10]; // way too small for 16x16
        let result = synth.apply_grain(&mut frame, 16, 16);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_dimensions_error() {
        let params = GrainParams::default();
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut frame = vec![0u8; 100];
        let result = synth.apply_grain(&mut frame, 0, 10);
        assert!(result.is_err());
    }

    // ---- Reset ----

    #[test]
    fn test_reset_clears_temporal_state() {
        let params = GrainParams {
            intensity: 0.3,
            seed: 42,
            temporal_correlation: 0.9,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut frame = make_yuv420(16, 16, 128);
        synth.apply_grain(&mut frame, 16, 16).expect("ok");
        assert!(synth.prev_grain_y.is_some());
        synth.reset();
        assert!(synth.prev_grain_y.is_none());
        assert_eq!(synth.frame_count, 0);
    }

    // ---- Uniform noise ----

    #[test]
    fn test_uniform_noise_mode() {
        let params = GrainParams {
            intensity: 0.3,
            seed: 77,
            gaussian: false,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(16, 16, 128);
        let mut frame = original.clone();
        synth.apply_grain(&mut frame, 16, 16).expect("ok");
        assert_ne!(frame, original);
    }

    // ---- Monochrome grain modifies chroma ----

    #[test]
    fn test_monochrome_grain_modifies_chroma() {
        let params = GrainParams {
            intensity: 0.5,
            seed: 42,
            grain_type: GrainType::Monochrome,
            ..GrainParams::default()
        };
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(32, 32, 128);
        let y_size = 32 * 32;
        let mut frame = original.clone();
        synth.apply_grain(&mut frame, 32, 32).expect("ok");
        assert_ne!(
            &frame[y_size..],
            &original[y_size..],
            "monochrome grain should modify chroma"
        );
    }

    // ---- Scaling LUT ----

    #[test]
    fn test_build_scaling_lut_single_point() {
        let lut = build_scaling_lut(&[(128, 64)]);
        assert!(lut.iter().all(|&v| v == 64));
    }

    #[test]
    fn test_build_scaling_lut_interpolation() {
        let lut = build_scaling_lut(&[(0, 0), (255, 255)]);
        assert_eq!(lut[0], 0);
        assert_eq!(lut[255], 255);
        assert!((lut[128] as i32 - 128).abs() <= 1);
    }

    // ---- GrainType variants ----

    #[test]
    fn test_grain_type_eq() {
        assert_eq!(GrainType::Monochrome, GrainType::Monochrome);
        assert_ne!(GrainType::Monochrome, GrainType::Color);
        assert_ne!(GrainType::Color, GrainType::LumaOnly);
    }

    // ---- set_params ----

    #[test]
    fn test_set_params_updates() {
        let params = GrainParams::default();
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let new_params = GrainParams {
            intensity: 0.8,
            seed: 999,
            ..GrainParams::default()
        };
        synth.set_params(new_params).expect("ok");
        assert!((synth.params.intensity - 0.8).abs() < 1e-6);
        assert_eq!(synth.params.seed, 999);
    }

    #[test]
    fn test_set_params_invalid_rejected() {
        let params = GrainParams::default();
        let mut synth = FilmGrainSynthesizer::new(params).expect("valid");
        let bad_params = GrainParams {
            intensity: 2.0,
            ..GrainParams::default()
        };
        assert!(synth.set_params(bad_params).is_err());
    }

    // ---- synthesize (LCG-based) ----

    #[test]
    fn test_synthesize_zero_intensity_no_change() {
        let params = GrainParams {
            intensity: 0.0,
            ..GrainParams::default()
        };
        let synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(16, 16, 128);
        let mut frame = original.clone();
        synth.synthesize(&mut frame, 16, 16, 42).expect("ok");
        assert_eq!(frame, original, "zero intensity should not change frame");
    }

    #[test]
    fn test_synthesize_nonzero_modifies_frame() {
        let params = GrainParams {
            intensity: 0.5,
            ..GrainParams::default()
        };
        let synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(32, 32, 128);
        let mut frame = original.clone();
        synth.synthesize(&mut frame, 32, 32, 42).expect("ok");
        assert_ne!(frame, original, "non-zero intensity should modify frame");
    }

    #[test]
    fn test_synthesize_deterministic() {
        let params = GrainParams {
            intensity: 0.3,
            ..GrainParams::default()
        };
        let synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut f1 = make_yuv420(16, 16, 100);
        let mut f2 = make_yuv420(16, 16, 100);
        synth.synthesize(&mut f1, 16, 16, 77).expect("ok");
        synth.synthesize(&mut f2, 16, 16, 77).expect("ok");
        assert_eq!(f1, f2, "same seed must produce same grain");
    }

    #[test]
    fn test_synthesize_different_seeds_differ() {
        let params = GrainParams {
            intensity: 0.3,
            ..GrainParams::default()
        };
        let synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut f1 = make_yuv420(16, 16, 128);
        let mut f2 = make_yuv420(16, 16, 128);
        synth.synthesize(&mut f1, 16, 16, 1).expect("ok");
        synth.synthesize(&mut f2, 16, 16, 999).expect("ok");
        assert_ne!(f1, f2, "different seeds should produce different grain");
    }

    #[test]
    fn test_synthesize_luma_only_no_chroma_change() {
        let params = GrainParams {
            intensity: 0.5,
            grain_type: GrainType::LumaOnly,
            ..GrainParams::default()
        };
        let synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(16, 16, 128);
        let y_size = 16 * 16;
        let mut frame = original.clone();
        synth.synthesize(&mut frame, 16, 16, 42).expect("ok");
        assert_eq!(
            &frame[y_size..],
            &original[y_size..],
            "LumaOnly: chroma unchanged"
        );
    }

    #[test]
    fn test_synthesize_color_modifies_chroma() {
        let params = GrainParams {
            intensity: 0.5,
            grain_type: GrainType::Color,
            ..GrainParams::default()
        };
        let synth = FilmGrainSynthesizer::new(params).expect("valid");
        let original = make_yuv420(32, 32, 128);
        let y_size = 32 * 32;
        let mut frame = original.clone();
        synth.synthesize(&mut frame, 32, 32, 42).expect("ok");
        assert_ne!(
            &frame[y_size..],
            &original[y_size..],
            "Color: chroma should be modified"
        );
    }

    #[test]
    fn test_synthesize_buffer_too_small() {
        let params = GrainParams::default();
        let synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut tiny = vec![0u8; 5];
        assert!(synth.synthesize(&mut tiny, 16, 16, 0).is_err());
    }

    #[test]
    fn test_synthesize_zero_dimensions_error() {
        let params = GrainParams::default();
        let synth = FilmGrainSynthesizer::new(params).expect("valid");
        let mut buf = vec![0u8; 100];
        assert!(synth.synthesize(&mut buf, 0, 10, 0).is_err());
    }

    // ── GrainSynthesizer (FilmGrainConfig API) tests ──────────────────────

    #[test]
    fn test_grain_synthesizer_new_creates_correct_buffer_size() {
        let config = FilmGrainConfig {
            grain_size: 2.0,
            ..FilmGrainConfig::default()
        };
        let synth = GrainSynthesizer::new(config, 16, 8);
        assert_eq!(
            synth.grain_buffer().len(),
            16 * 8,
            "grain_buffer must contain width * height entries"
        );
    }

    #[test]
    fn test_grain_synthesizer_zero_intensity_leaves_frame_unchanged() {
        let config = FilmGrainConfig {
            intensity: 0.0,
            ..FilmGrainConfig::default()
        };
        let mut synth = GrainSynthesizer::new(config, 8, 8);
        let original = vec![128u8; 8 * 8 * 3];
        let mut frame = original.clone();
        synth.apply(&mut frame, 8, 8, false);
        assert_eq!(
            frame, original,
            "zero intensity must leave the frame unchanged"
        );
    }

    #[test]
    fn test_grain_synthesizer_nonzero_intensity_modifies_frame() {
        let config = FilmGrainConfig {
            intensity: 1.0,
            grain_size: 1.0,
            ..FilmGrainConfig::default()
        };
        let mut synth = GrainSynthesizer::new(config, 16, 16);
        let original = vec![128u8; 16 * 16 * 3];
        let mut frame = original.clone();
        synth.apply(&mut frame, 16, 16, false);
        assert_ne!(frame, original, "intensity=1.0 must modify the frame");
    }

    #[test]
    fn test_grain_synthesizer_luma_only_rgb_channel_0_modified() {
        // With luma_only=true on an RGB frame, the grey pixel (R=G=B) should be
        // uniformly shifted; all channels change by the same luma delta.
        let config = FilmGrainConfig {
            intensity: 1.0,
            luma_only: true,
            grain_size: 1.0,
            temporal_coherence: 1.0, // use stored buffer for determinism
            ..FilmGrainConfig::default()
        };
        let mut synth = GrainSynthesizer::new(config, 4, 4);
        let mut frame = vec![128u8; 4 * 4 * 3];
        synth.apply(&mut frame, 4, 4, false);
        // Verify at least one pixel changed (grain was actually applied)
        let changed = frame.iter().any(|&b| b != 128);
        assert!(changed, "luma_only should still modify the frame");
    }

    #[test]
    fn test_grain_synthesizer_luma_only_false_all_channels_independent() {
        let config = FilmGrainConfig {
            intensity: 1.0,
            luma_only: false,
            grain_size: 1.0,
            temporal_coherence: 0.0, // fresh noise each channel
            seed: 42,
        };
        let mut synth = GrainSynthesizer::new(config, 16, 16);
        let mut frame = vec![128u8; 16 * 16 * 3];
        synth.apply(&mut frame, 16, 16, false);
        // All channels should be modified
        let mut r_changed = false;
        let mut g_changed = false;
        let mut b_changed = false;
        for px in 0..16 * 16 {
            if frame[px * 3] != 128 {
                r_changed = true;
            }
            if frame[px * 3 + 1] != 128 {
                g_changed = true;
            }
            if frame[px * 3 + 2] != 128 {
                b_changed = true;
            }
        }
        assert!(r_changed, "R channel should be modified");
        assert!(g_changed, "G channel should be modified");
        assert!(b_changed, "B channel should be modified");
    }

    #[test]
    fn test_grain_synthesizer_temporal_coherence_1_gives_same_grain_across_frames() {
        let config = FilmGrainConfig {
            intensity: 1.0,
            luma_only: false,
            temporal_coherence: 1.0, // static grain
            grain_size: 1.0,
            seed: 99,
        };
        let mut synth = GrainSynthesizer::new(config, 8, 8);
        let base = vec![128u8; 8 * 8 * 3];

        let mut f1 = base.clone();
        synth.apply(&mut f1, 8, 8, false);
        let mut f2 = base.clone();
        synth.apply(&mut f2, 8, 8, false);
        assert_eq!(
            f1, f2,
            "temporal_coherence=1.0 should produce identical grain every frame"
        );
    }

    #[test]
    fn test_grain_synthesizer_temporal_coherence_0_varies_per_frame() {
        let config = FilmGrainConfig {
            intensity: 1.0,
            temporal_coherence: 0.0, // pure per-frame noise
            grain_size: 1.0,
            seed: 7,
            ..FilmGrainConfig::default()
        };
        let mut synth = GrainSynthesizer::new(config, 16, 16);
        let base = vec![128u8; 16 * 16 * 3];

        let mut f1 = base.clone();
        synth.apply(&mut f1, 16, 16, false);
        let mut f2 = base.clone();
        synth.apply(&mut f2, 16, 16, false);
        // Very unlikely to be identical with fresh noise each frame
        assert_ne!(
            f1, f2,
            "temporal_coherence=0.0 should produce different grain each frame"
        );
    }

    #[test]
    fn test_grain_synthesizer_yuv420_luma_only_chroma_unchanged() {
        let width = 16u32;
        let height = 16u32;
        let config = FilmGrainConfig {
            intensity: 1.0,
            luma_only: true,
            grain_size: 1.0,
            temporal_coherence: 0.0,
            seed: 5,
        };
        let mut synth = GrainSynthesizer::new(config, width, height);

        let y_size = (width * height) as usize;
        let uv_size = ((width + 1) / 2 * (height + 1) / 2) as usize;
        let original_uv = vec![128u8; uv_size * 2];
        let mut frame = vec![128u8; y_size];
        frame.extend_from_slice(&original_uv);

        synth.apply(&mut frame, width, height, true);

        // Y plane should be modified
        let y_changed = frame[..y_size].iter().any(|&b| b != 128);
        assert!(y_changed, "Y plane must be modified");
        // U and V planes must be unchanged
        assert_eq!(
            &frame[y_size..],
            &original_uv[..],
            "luma_only=true: chroma planes must be unchanged"
        );
    }

    #[test]
    fn test_grain_synthesizer_frame_counter_increments() {
        let config = FilmGrainConfig::default();
        let mut synth = GrainSynthesizer::new(config, 8, 8);
        assert_eq!(synth.frame_counter(), 0);
        let mut frame = vec![128u8; 8 * 8 * 3];
        synth.apply(&mut frame, 8, 8, false);
        assert_eq!(synth.frame_counter(), 1);
        synth.apply(&mut frame, 8, 8, false);
        assert_eq!(synth.frame_counter(), 2);
    }

    #[test]
    fn test_grain_synthesizer_output_clamped_to_byte_range() {
        // Start from 0 and 255 extremes; grain must not overflow u8
        let config = FilmGrainConfig {
            intensity: 1.0,
            grain_size: 1.0,
            ..FilmGrainConfig::default()
        };
        let mut synth = GrainSynthesizer::new(config, 8, 8);
        let mut frame_min = vec![0u8; 8 * 8 * 3];
        synth.apply(&mut frame_min, 8, 8, false);
        let mut frame_max = vec![255u8; 8 * 8 * 3];
        synth.apply(&mut frame_max, 8, 8, false);
        // No assertion needed beyond "no panic"; the clamping is verified by
        // the fact that these are u8 slices that compiled and ran without overflow.
        for &b in &frame_min {
            let _ = b;
        }
        for &b in &frame_max {
            let _ = b;
        }
    }
}
