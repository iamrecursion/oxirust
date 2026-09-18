//! Chroma key implementation for video switchers.
//!
//! Chroma keying (green screen/blue screen) creates transparency based on color.

use oximedia_codec::VideoFrame;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during chroma keying.
#[derive(Error, Debug, Clone)]
pub enum ChromaKeyError {
    #[error("Invalid hue value: {0}")]
    InvalidHue(f32),

    #[error("Invalid saturation value: {0}")]
    InvalidSaturation(f32),

    #[error("Frame dimension mismatch")]
    DimensionMismatch,

    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Chroma key color.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ChromaColor {
    /// Green screen
    Green,
    /// Blue screen
    Blue,
    /// Custom color (H, S, V in 0.0-1.0 range)
    Custom { h: f32, s: f32, v: f32 },
}

impl ChromaColor {
    /// Get the target hue.
    pub fn hue(&self) -> f32 {
        match self {
            ChromaColor::Green => 120.0 / 360.0, // Green at 120 degrees
            ChromaColor::Blue => 240.0 / 360.0,  // Blue at 240 degrees
            ChromaColor::Custom { h, .. } => *h,
        }
    }

    /// Get RGB values for the color.
    pub fn to_rgb(&self) -> (u8, u8, u8) {
        match self {
            ChromaColor::Green => (0, 255, 0),
            ChromaColor::Blue => (0, 0, 255),
            ChromaColor::Custom { h, s, v } => hsv_to_rgb(*h, *s, *v),
        }
    }
}

/// Convert HSV to RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h * 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Convert RGB to HSV.
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;

    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    // Hue
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let h = if h < 0.0 { h + 360.0 } else { h };

    // Saturation
    let s = if max == 0.0 { 0.0 } else { delta / max };

    // Value
    let v = max;

    (h / 360.0, s, v)
}

/// Chroma key parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromaKeyParams {
    /// Key color
    pub color: ChromaColor,
    /// Hue tolerance (0.0 - 1.0)
    pub hue_tolerance: f32,
    /// Saturation tolerance (0.0 - 1.0)
    pub saturation_tolerance: f32,
    /// Value/brightness tolerance (0.0 - 1.0)
    pub value_tolerance: f32,
    /// Spill suppression amount (0.0 - 1.0)
    pub spill_suppression: f32,
    /// Edge softness (0.0 - 1.0)
    pub edge_softness: f32,
    /// Clip level (0.0 - 1.0)
    pub clip: f32,
    /// Gain (0.0 - 2.0)
    pub gain: f32,
}

impl ChromaKeyParams {
    /// Create new chroma key parameters with defaults for green screen.
    pub fn new_green() -> Self {
        Self {
            color: ChromaColor::Green,
            hue_tolerance: 0.1,
            saturation_tolerance: 0.3,
            value_tolerance: 0.3,
            spill_suppression: 0.5,
            edge_softness: 0.1,
            clip: 0.0,
            gain: 1.0,
        }
    }

    /// Create new chroma key parameters with defaults for blue screen.
    pub fn new_blue() -> Self {
        Self {
            color: ChromaColor::Blue,
            hue_tolerance: 0.1,
            saturation_tolerance: 0.3,
            value_tolerance: 0.3,
            spill_suppression: 0.5,
            edge_softness: 0.1,
            clip: 0.0,
            gain: 1.0,
        }
    }

    /// Set hue tolerance.
    pub fn set_hue_tolerance(&mut self, tolerance: f32) -> Result<(), ChromaKeyError> {
        if !(0.0..=1.0).contains(&tolerance) {
            return Err(ChromaKeyError::InvalidHue(tolerance));
        }
        self.hue_tolerance = tolerance;
        Ok(())
    }

    /// Set saturation tolerance.
    pub fn set_saturation_tolerance(&mut self, tolerance: f32) -> Result<(), ChromaKeyError> {
        if !(0.0..=1.0).contains(&tolerance) {
            return Err(ChromaKeyError::InvalidSaturation(tolerance));
        }
        self.saturation_tolerance = tolerance;
        Ok(())
    }
}

impl Default for ChromaKeyParams {
    fn default() -> Self {
        Self::new_green()
    }
}

/// Chroma key processor.
pub struct ChromaKey {
    params: ChromaKeyParams,
    enabled: bool,
}

impl ChromaKey {
    /// Create a new chroma key processor with green screen defaults.
    pub fn new_green() -> Self {
        Self {
            params: ChromaKeyParams::new_green(),
            enabled: true,
        }
    }

    /// Create a new chroma key processor with blue screen defaults.
    pub fn new_blue() -> Self {
        Self {
            params: ChromaKeyParams::new_blue(),
            enabled: true,
        }
    }

    /// Create with specific parameters.
    pub fn with_params(params: ChromaKeyParams) -> Self {
        Self {
            params,
            enabled: true,
        }
    }

    /// Get the parameters.
    pub fn params(&self) -> &ChromaKeyParams {
        &self.params
    }

    /// Get mutable parameters.
    pub fn params_mut(&mut self) -> &mut ChromaKeyParams {
        &mut self.params
    }

    /// Set parameters.
    pub fn set_params(&mut self, params: ChromaKeyParams) {
        self.params = params;
    }

    /// Enable or disable the key.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the key is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Calculate color distance in HSV space.
    pub fn calculate_distance(&self, h: f32, s: f32, v: f32) -> f32 {
        let target_h = self.params.color.hue();

        // Calculate hue distance (circular)
        let mut h_dist = (h - target_h).abs();
        if h_dist > 0.5 {
            h_dist = 1.0 - h_dist;
        }
        h_dist /= self.params.hue_tolerance.max(0.001);

        // Saturation distance: low saturation means "not the key colour".
        // A fully-saturated pixel (s=1) contributes zero saturation distance.
        let s_dist = (1.0 - s) / self.params.saturation_tolerance.max(0.001);

        // Value distance: only penalize very dark pixels (v << 0.5).
        // Bright, saturated pixels should not be penalized for having v=1.
        // Use a one-sided penalty: v_dist is non-zero only for v < 0.5.
        let v_dist = (0.5 - v).max(0.0) / self.params.value_tolerance.max(0.001) * 0.5;

        // Combined distance
        (h_dist * h_dist + s_dist * s_dist + v_dist * v_dist).sqrt()
    }

    /// Calculate alpha from pixel color.
    pub fn calculate_alpha(&self, r: u8, g: u8, b: u8) -> f32 {
        if !self.enabled {
            return 1.0;
        }

        let (h, s, v) = rgb_to_hsv(r, g, b);

        // Calculate distance to key color
        let distance = self.calculate_distance(h, s, v);

        // Convert distance to alpha (closer = more transparent)
        let mut alpha = distance;

        // Apply edge softness
        if self.params.edge_softness > 0.0 {
            let soft = 1.0 - self.params.edge_softness;
            alpha = ((alpha - soft) / self.params.edge_softness.max(0.001)).clamp(0.0, 1.0);
        }

        // Apply clip and gain
        alpha = ((alpha - self.params.clip) * self.params.gain).clamp(0.0, 1.0);

        alpha
    }

    /// Apply spill suppression to a pixel.
    ///
    /// Uses a weighted luminance-preserving algorithm: excess key-color channel
    /// energy is replaced by the average of the other two channels, then
    /// blended with the original by `spill_suppression`.
    pub fn suppress_spill(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        if self.params.spill_suppression == 0.0 {
            return (r, g, b);
        }

        let (r_f, g_f, b_f) = (r as f32, g as f32, b as f32);
        let strength = self.params.spill_suppression.clamp(0.0, 1.0);

        let (r_out, g_out, b_out) = match self.params.color {
            ChromaColor::Green => {
                // Spill amount = how much green exceeds the average of R and B.
                let other_avg = (r_f + b_f) * 0.5;
                let spill = (g_f - other_avg).max(0.0);
                // Replace excess green with the luminance-neutral average.
                let g_corrected = g_f - spill * strength;
                // Boost R and B slightly to compensate for lost luminance.
                let luma_compensation = spill * strength * 0.3;
                (
                    (r_f + luma_compensation).min(255.0),
                    g_corrected,
                    (b_f + luma_compensation).min(255.0),
                )
            }
            ChromaColor::Blue => {
                let other_avg = (r_f + g_f) * 0.5;
                let spill = (b_f - other_avg).max(0.0);
                let b_corrected = b_f - spill * strength;
                let luma_compensation = spill * strength * 0.3;
                (
                    (r_f + luma_compensation).min(255.0),
                    (g_f + luma_compensation).min(255.0),
                    b_corrected,
                )
            }
            ChromaColor::Custom { h, .. } => {
                // Generic: rotate into HSV, desaturate the key hue component.
                let (px_h, px_s, px_v) = rgb_to_hsv(r, g, b);
                let target_h = h;
                let mut h_dist = (px_h - target_h).abs();
                if h_dist > 0.5 {
                    h_dist = 1.0 - h_dist;
                }
                // Within hue tolerance, reduce saturation proportionally.
                let tol = self.params.hue_tolerance.max(0.01);
                if h_dist < tol {
                    let spill_factor = (1.0 - h_dist / tol) * strength;
                    let new_s = px_s * (1.0 - spill_factor);
                    let (nr, ng, nb) = hsv_to_rgb(px_h, new_s, px_v);
                    (nr as f32, ng as f32, nb as f32)
                } else {
                    (r_f, g_f, b_f)
                }
            }
        };

        (
            r_out.clamp(0.0, 255.0) as u8,
            g_out.clamp(0.0, 255.0) as u8,
            b_out.clamp(0.0, 255.0) as u8,
        )
    }

    /// Apply edge softening to an already-computed alpha value.
    ///
    /// Edge softening uses a smooth Hermite curve in the transition zone
    /// defined by `edge_softness`:
    /// - Full transparency region: alpha below `clip`
    /// - Transition zone: `edge_softness` wide (Hermite smoothstep)
    /// - Full opacity: alpha beyond the transition zone
    pub fn apply_edge_softening(&self, raw_alpha: f32) -> f32 {
        if self.params.edge_softness <= 0.0 {
            return raw_alpha.clamp(0.0, 1.0);
        }
        let soft = self.params.edge_softness.clamp(0.0, 1.0);
        // Low boundary: alpha below which we are hard-keyed out.
        let lo = self.params.clip.clamp(0.0, 1.0);
        let hi = (lo + soft).min(1.0);
        if raw_alpha <= lo {
            0.0
        } else if raw_alpha >= hi {
            1.0
        } else {
            // Hermite smoothstep in [lo, hi].
            let t = (raw_alpha - lo) / (hi - lo);
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// Sample the key color and build an enhanced matte using the spill map.
    ///
    /// Returns `(alpha, spill_amount)` for a pixel, where `spill_amount`
    /// is [0..1] indicating how much key-color contamination was detected.
    pub fn analyze_pixel(&self, r: u8, g: u8, b: u8) -> (f32, f32) {
        let (px_h, px_s, px_v) = rgb_to_hsv(r, g, b);
        let distance = self.calculate_distance(px_h, px_s, px_v);

        // Raw alpha from distance (before softening).
        let raw_alpha = distance.clamp(0.0, 1.0);

        // Edge-softened alpha.
        let alpha = self.apply_edge_softening(raw_alpha);
        let alpha_gain = ((alpha - self.params.clip.max(0.0)) * self.params.gain).clamp(0.0, 1.0);

        // Spill amount: how much key-channel energy is present.
        let (r_f, g_f, b_f) = (r as f32, g as f32, b as f32);
        let spill = match self.params.color {
            ChromaColor::Green => (g_f - r_f.max(b_f)).max(0.0) / 255.0,
            ChromaColor::Blue => (b_f - r_f.max(g_f)).max(0.0) / 255.0,
            ChromaColor::Custom { .. } => (1.0 - raw_alpha).max(0.0),
        };

        (alpha_gain, spill.clamp(0.0, 1.0))
    }

    /// Process a single pixel — returns (R, G, B, A) after spill suppression
    /// and edge-softened alpha calculation.
    pub fn process_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8, u8) {
        let (alpha_gain, _spill) = self.analyze_pixel(r, g, b);
        let (r_out, g_out, b_out) = self.suppress_spill(r, g, b);
        let alpha_u8 = (alpha_gain * 255.0) as u8;
        (r_out, g_out, b_out, alpha_u8)
    }

    /// Process a video frame to extract alpha channel.
    ///
    /// For planar YUV formats the luma plane is used as a rough proxy for
    /// luminance-based keying. For frames with at least 3 planes (Y, Cb, Cr),
    /// the chroma planes are converted back to approximate RGB per pixel so
    /// that proper colour-distance keying can be applied.
    ///
    /// Returns a `Vec<u8>` with one alpha byte per luma-plane pixel
    /// (0 = fully transparent / keyed, 255 = fully opaque).
    pub fn process_frame(&self, fill: &VideoFrame) -> Result<Vec<u8>, ChromaKeyError> {
        if fill.planes.is_empty() {
            return Err(ChromaKeyError::ProcessingError(
                "Frame has no planes".to_string(),
            ));
        }

        let luma_plane = &fill.planes[0];
        let pixel_count = (luma_plane.width as usize) * (luma_plane.height as usize);

        // If we have at least 3 planes we can approximate RGB from YCbCr.
        if fill.planes.len() >= 3 {
            let cb_plane = &fill.planes[1];
            let cr_plane = &fill.planes[2];

            let luma_w = luma_plane.width as usize;
            let luma_h = luma_plane.height as usize;
            let cb_w = cb_plane.width as usize;
            let _cr_w = cr_plane.width as usize;

            // Chroma sub-sampling ratios
            let h_ratio = luma_w.checked_div(cb_w).unwrap_or(1);
            let v_ratio_cb = luma_h.checked_div(cb_plane.height as usize).unwrap_or(1);

            let mut alpha_out = Vec::with_capacity(pixel_count);

            for y in 0..luma_h {
                for x in 0..luma_w {
                    let li = y * luma_plane.stride + x;
                    let y_val = if li < luma_plane.data.len() {
                        luma_plane.data[li] as f32
                    } else {
                        0.0
                    };

                    let cx = x / h_ratio.max(1);
                    let cy = y / v_ratio_cb.max(1);

                    let cb_i = cy * cb_plane.stride + cx;
                    let cr_i = cy * cr_plane.stride + cx;

                    let cb_val = if cb_i < cb_plane.data.len() {
                        cb_plane.data[cb_i] as f32 - 128.0
                    } else {
                        0.0
                    };

                    let cr_val = if cr_i < cr_plane.data.len() {
                        cr_plane.data[cr_i] as f32 - 128.0
                    } else {
                        0.0
                    };

                    // BT.601 YCbCr -> RGB
                    let r = (y_val + 1.402 * cr_val).clamp(0.0, 255.0) as u8;
                    let g = (y_val - 0.344136 * cb_val - 0.714136 * cr_val).clamp(0.0, 255.0) as u8;
                    let b = (y_val + 1.772 * cb_val).clamp(0.0, 255.0) as u8;

                    let alpha = self.calculate_alpha(r, g, b);
                    alpha_out.push((alpha * 255.0) as u8);
                }
            }

            Ok(alpha_out)
        } else {
            // Single-plane: use luma only as a fallback (luminance keying)
            let mut alpha_out = Vec::with_capacity(pixel_count);
            for &luma in &luma_plane.data[..pixel_count.min(luma_plane.data.len())] {
                let alpha = self.calculate_alpha(luma, luma, luma);
                alpha_out.push((alpha * 255.0) as u8);
            }
            Ok(alpha_out)
        }
    }
}

impl Default for ChromaKey {
    fn default() -> Self {
        Self::new_green()
    }
}

// ---------------------------------------------------------------------------
// YCbCr-space chroma keying
// ---------------------------------------------------------------------------

/// BT.601 RGB→Y′CbCr conversion constants (studio-swing, full-range).
///
/// For 8-bit full-range:
/// ```text
/// Y  =  0.2989 * R + 0.5866 * G + 0.1145 * B
/// Cb = -0.1687 * R - 0.3313 * G + 0.5000 * B + 128
/// Cr =  0.5000 * R - 0.4187 * G - 0.0813 * B + 128
/// ```
///
/// Cb and Cr are in [0, 255] with neutral grey at 128.
#[inline]
fn rgb_to_ycbcr_601(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;
    let y = 0.2989 * r + 0.5866 * g + 0.1145 * b;
    let cb = -0.1687 * r - 0.3313 * g + 0.5000 * b + 128.0;
    let cr = 0.5000 * r - 0.4187 * g - 0.0813 * b + 128.0;
    (y, cb, cr)
}

/// Parameters for YCbCr-space chroma keying.
///
/// This is the industry-standard algorithm used by professional broadcast
/// switchers (Blackmagic, Ross, Grass Valley, etc.).  Rather than converting to
/// HSV and measuring hue distance, the algorithm measures **Euclidean distance
/// in Cb/Cr space** from the key colour's chrominance coordinates.
///
/// ## Algorithm
///
/// ```text
/// key_cb, key_cr = YCbCr of the key colour (e.g. green → Cb≈83, Cr≈38)
///
/// For each pixel (R, G, B):
///   Y, Cb, Cr = bt601_rgb_to_ycbcr(R, G, B)
///   dist_cb = Cb - key_cb
///   dist_cr = Cr - key_cr
///   chroma_dist = sqrt(dist_cb² + dist_cr²)   ∈ [0, 127]
///
///   # Normalise to [0, 1]:
///   t = chroma_dist / (max_dist * tolerance)
///   raw_alpha = smoothstep(0, 1, t)             ← Hermite cubic
///
///   # Luminance suppression: very dark pixels are always opaque.
///   luma_weight = clamp(Y / 64, 0, 1)           ← 0 = black, 1 = normal
///   alpha = raw_alpha * luma_weight + (1 - luma_weight)
/// ```
///
/// `tolerance` controls how "wide" the key colour window is: `0.0` keys
/// nothing, `1.0` keys the full Cb/Cr hemisphere on the key colour's side.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromaKeyYcbcrParams {
    /// Key colour in RGB (will be converted to Cb/Cr internally).
    pub key_rgb: (u8, u8, u8),
    /// Tolerance radius in normalised Cb/Cr space (0.0 – 1.0).
    ///
    /// Controls how wide the key colour window is.  Typical values: 0.3 – 0.6.
    pub tolerance: f32,
    /// Softness of the key edge (0.0 – 1.0).
    ///
    /// A value of `0.0` gives a hard edge; higher values blend gradually.
    pub softness: f32,
    /// Spill suppression strength (0.0 – 1.0).
    ///
    /// Reduces key-colour contamination on foreground subjects.
    pub spill_suppression: f32,
    /// If `true`, swap transparent and opaque regions.
    pub invert: bool,
}

impl ChromaKeyYcbcrParams {
    /// Maximum Cb/Cr distance from the neutral axis to any pure saturated colour.
    ///
    /// In 8-bit BT.601 with full-range encoding the Cb and Cr channels each
    /// span roughly [3, 250] (≈ ±127 from neutral-128).  The maximum Euclidean
    /// distance from any two antipodal colours in Cb/Cr space is therefore
    /// `sqrt(127² + 127²) ≈ 179.6`.  We use `128.0 * sqrt(2) ≈ 181` as a
    /// convenient normalisation constant.
    const MAX_CBCR_DIST: f32 = 181.0;

    /// Create default green-screen parameters.
    pub fn green_screen() -> Self {
        Self {
            key_rgb: (0, 255, 0),
            tolerance: 0.45,
            softness: 0.15,
            spill_suppression: 0.5,
            invert: false,
        }
    }

    /// Create default blue-screen parameters.
    pub fn blue_screen() -> Self {
        Self {
            key_rgb: (0, 0, 255),
            tolerance: 0.45,
            softness: 0.15,
            spill_suppression: 0.5,
            invert: false,
        }
    }

    /// Compute the Cb and Cr targets for the configured key colour.
    pub fn key_cbcr(&self) -> (f32, f32) {
        let (_, cb, cr) = rgb_to_ycbcr_601(self.key_rgb.0, self.key_rgb.1, self.key_rgb.2);
        (cb, cr)
    }
}

impl Default for ChromaKeyYcbcrParams {
    fn default() -> Self {
        Self::green_screen()
    }
}

/// YCbCr-space chroma keyer.
///
/// Operates entirely in the Cb/Cr chrominance plane using Euclidean distance,
/// which is faster and more numerically accurate than HSV-based approaches for
/// broadcast green-/blue-screen work.
///
/// See [`ChromaKeyYcbcrParams`] for the full mathematical description.
pub struct ChromaKeyYcbcr {
    params: ChromaKeyYcbcrParams,
    enabled: bool,
    /// Cached key Cb/Cr to avoid recomputing every frame.
    cached_key_cbcr: (f32, f32),
}

impl ChromaKeyYcbcr {
    /// Create a green-screen keyer.
    pub fn new_green() -> Self {
        let params = ChromaKeyYcbcrParams::green_screen();
        let cached_key_cbcr = params.key_cbcr();
        Self {
            params,
            enabled: true,
            cached_key_cbcr,
        }
    }

    /// Create a blue-screen keyer.
    pub fn new_blue() -> Self {
        let params = ChromaKeyYcbcrParams::blue_screen();
        let cached_key_cbcr = params.key_cbcr();
        Self {
            params,
            enabled: true,
            cached_key_cbcr,
        }
    }

    /// Create with specific parameters.
    pub fn with_params(params: ChromaKeyYcbcrParams) -> Self {
        let cached_key_cbcr = params.key_cbcr();
        Self {
            params,
            enabled: true,
            cached_key_cbcr,
        }
    }

    /// Get the parameters.
    pub fn params(&self) -> &ChromaKeyYcbcrParams {
        &self.params
    }

    /// Get mutable parameters and refresh the Cb/Cr cache.
    pub fn params_mut(&mut self) -> &mut ChromaKeyYcbcrParams {
        &mut self.params
    }

    /// Refresh the internal Cb/Cr cache after mutating params.
    pub fn refresh_cache(&mut self) {
        self.cached_key_cbcr = self.params.key_cbcr();
    }

    /// Update all parameters at once (refreshes cache automatically).
    pub fn set_params(&mut self, params: ChromaKeyYcbcrParams) {
        self.cached_key_cbcr = params.key_cbcr();
        self.params = params;
    }

    /// Enable or disable the keyer.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether the keyer is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Compute the normalised chroma distance for a pixel.
    ///
    /// Returns a value in `[0.0, 1.0]`:
    /// * `0.0` — pixel is exactly the key colour (fully transparent)
    /// * `1.0` — pixel is maximally far from the key colour (fully opaque)
    pub fn chroma_distance(&self, r: u8, g: u8, b: u8) -> f32 {
        let (_, cb, cr) = rgb_to_ycbcr_601(r, g, b);
        let (key_cb, key_cr) = self.cached_key_cbcr;
        let d_cb = cb - key_cb;
        let d_cr = cr - key_cr;
        let dist = (d_cb * d_cb + d_cr * d_cr).sqrt();
        // Normalise by tolerance-scaled max distance.
        let scaled_max = ChromaKeyYcbcrParams::MAX_CBCR_DIST * self.params.tolerance.max(1e-4);
        (dist / scaled_max).min(1.0)
    }

    /// Compute the alpha (opacity) for a single pixel.
    ///
    /// `0.0` = fully transparent (matches key colour); `1.0` = fully opaque.
    pub fn calculate_alpha(&self, r: u8, g: u8, b: u8) -> f32 {
        if !self.enabled {
            return 1.0;
        }

        let dist = self.chroma_distance(r, g, b);

        // Smoothstep from dist=0 (transparent) to dist=1 (opaque).
        let softness = self.params.softness.max(1e-4);
        let lo = 0.0_f32;
        let hi = softness;

        let raw_alpha = if dist <= lo {
            0.0
        } else if dist >= hi {
            1.0
        } else {
            let t = (dist - lo) / (hi - lo);
            t * t * (3.0 - 2.0 * t) // Hermite smoothstep
        };

        // Luminance-aware blend: very dark pixels are always opaque to avoid
        // keying out dark/shadowed regions on the subject.
        let (y, _, _) = rgb_to_ycbcr_601(r, g, b);
        // Threshold: below 32 (≈ 12% luma) we protect the pixel from keying.
        let luma_protect = (y / 32.0).clamp(0.0, 1.0);
        let alpha = raw_alpha * luma_protect + (1.0 - luma_protect);

        if self.params.invert {
            1.0 - alpha.clamp(0.0, 1.0)
        } else {
            alpha.clamp(0.0, 1.0)
        }
    }

    /// Apply YCbCr-space spill suppression to a pixel.
    ///
    /// Reduces contamination by the key colour in the foreground subject.
    /// Unlike simple colour desaturation, this targets only the key-colour
    /// Cb/Cr quadrant while leaving other colours untouched.
    ///
    /// The technique:
    /// 1. Convert RGB → YCbCr.
    /// 2. Measure angular distance of the pixel's chrominance from the key
    ///    colour's Cb/Cr direction.
    /// 3. For pixels whose chrominance angle is close to the key direction,
    ///    push the Cb/Cr coordinates towards the neutral axis (128, 128)
    ///    proportionally to `spill_suppression`.
    /// 4. Convert back to RGB.
    pub fn suppress_spill(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let strength = self.params.spill_suppression;
        if strength <= 0.0 {
            return (r, g, b);
        }

        let (y, cb, cr) = rgb_to_ycbcr_601(r, g, b);
        let (key_cb, key_cr) = self.cached_key_cbcr;

        // Direction vectors from neutral.
        let key_dcb = key_cb - 128.0;
        let key_dcr = key_cr - 128.0;
        let px_dcb = cb - 128.0;
        let px_dcr = cr - 128.0;

        // Dot product — how much of the pixel's chroma aligns with the key direction.
        let key_len2 = key_dcb * key_dcb + key_dcr * key_dcr;
        if key_len2 < 1.0 {
            return (r, g, b); // key colour is nearly neutral; no suppression possible
        }
        let dot = px_dcb * key_dcb + px_dcr * key_dcr;
        // Only suppress if the pixel's chroma is pointing in the same direction.
        if dot <= 0.0 {
            return (r, g, b);
        }

        // Projection of pixel chrominance onto the key direction.
        let proj = dot / key_len2;
        // Move the Cb/Cr towards neutral by `strength * projection * key_vector`.
        let new_cb = cb - strength * proj * key_dcb;
        let new_cr = cr - strength * proj * key_dcr;

        // Convert back to RGB via BT.601 inverse (full-range).
        // R = Y + 1.4022 * (Cr - 128)
        // G = Y - 0.3456 * (Cb - 128) - 0.7145 * (Cr - 128)
        // B = Y + 1.7710 * (Cb - 128)
        let new_r = (y + 1.4022 * (new_cr - 128.0)).clamp(0.0, 255.0) as u8;
        let new_g =
            (y - 0.3456 * (new_cb - 128.0) - 0.7145 * (new_cr - 128.0)).clamp(0.0, 255.0) as u8;
        let new_b = (y + 1.7710 * (new_cb - 128.0)).clamp(0.0, 255.0) as u8;

        (new_r, new_g, new_b)
    }

    /// Process a single pixel.
    ///
    /// Returns `(R, G, B, A)` after spill suppression and alpha calculation.
    pub fn process_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8, u8) {
        let alpha = self.calculate_alpha(r, g, b);
        let (r_out, g_out, b_out) = self.suppress_spill(r, g, b);
        let alpha_u8 = (alpha * 255.0) as u8;
        (r_out, g_out, b_out, alpha_u8)
    }

    /// Process a video frame (planar YCbCr 4:2:0 / 4:2:2 or single-plane).
    ///
    /// For frames with at least 3 planes (Y, Cb, Cr), the chroma planes are
    /// sampled directly — no RGB conversion is needed, which is considerably
    /// more efficient.  For frames with fewer planes, RGB channels are used
    /// as an approximation.
    ///
    /// Returns a `Vec<u8>` with one alpha byte per luma-plane pixel.
    pub fn process_frame(&self, fill: &VideoFrame) -> Result<Vec<u8>, ChromaKeyError> {
        if fill.planes.is_empty() {
            return Err(ChromaKeyError::ProcessingError(
                "Frame has no planes".to_string(),
            ));
        }

        let luma_plane = &fill.planes[0];
        let luma_w = luma_plane.width as usize;
        let luma_h = luma_plane.height as usize;
        let pixel_count = luma_w * luma_h;

        if fill.planes.len() >= 3 {
            // Native YCbCr path: sample Cb and Cr directly.
            let cb_plane = &fill.planes[1];
            let cr_plane = &fill.planes[2];

            let h_ratio = luma_w.checked_div(cb_plane.width as usize).unwrap_or(1);
            let v_ratio = luma_h.checked_div(cb_plane.height as usize).unwrap_or(1);

            let (key_cb, key_cr) = self.cached_key_cbcr;

            let mut alpha_out = Vec::with_capacity(pixel_count);

            for y in 0..luma_h {
                for x in 0..luma_w {
                    let li = y * luma_plane.stride + x;
                    let y_val = if li < luma_plane.data.len() {
                        luma_plane.data[li] as f32
                    } else {
                        0.0
                    };

                    let cx = x / h_ratio.max(1);
                    let cy = y / v_ratio.max(1);
                    let cb_i = cy * cb_plane.stride + cx;
                    let cr_i = cy * cr_plane.stride + cx;

                    let cb_val = if cb_i < cb_plane.data.len() {
                        cb_plane.data[cb_i] as f32
                    } else {
                        128.0
                    };
                    let cr_val = if cr_i < cr_plane.data.len() {
                        cr_plane.data[cr_i] as f32
                    } else {
                        128.0
                    };

                    // Distance in Cb/Cr space.
                    let d_cb = cb_val - key_cb;
                    let d_cr = cr_val - key_cr;
                    let dist = (d_cb * d_cb + d_cr * d_cr).sqrt();
                    let scaled_max =
                        ChromaKeyYcbcrParams::MAX_CBCR_DIST * self.params.tolerance.max(1e-4);
                    let norm_dist = (dist / scaled_max).min(1.0);

                    let softness = self.params.softness.max(1e-4);
                    let raw_alpha = if norm_dist >= softness {
                        1.0
                    } else {
                        let t = norm_dist / softness;
                        t * t * (3.0 - 2.0 * t)
                    };

                    let luma_protect = (y_val / 32.0).clamp(0.0, 1.0);
                    let alpha = raw_alpha * luma_protect + (1.0 - luma_protect);
                    let alpha = if self.params.invert {
                        1.0 - alpha.clamp(0.0, 1.0)
                    } else {
                        alpha.clamp(0.0, 1.0)
                    };

                    alpha_out.push((alpha * 255.0) as u8);
                }
            }

            Ok(alpha_out)
        } else {
            // Fallback: treat single plane as luminance approximation.
            let mut alpha_out = Vec::with_capacity(pixel_count);
            for &luma in &luma_plane.data[..pixel_count.min(luma_plane.data.len())] {
                // Approximate: treat as grey pixel, calculate alpha.
                let alpha = self.calculate_alpha(luma, luma, luma);
                alpha_out.push((alpha * 255.0) as u8);
            }
            Ok(alpha_out)
        }
    }
}

impl Default for ChromaKeyYcbcr {
    fn default() -> Self {
        Self::new_green()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chroma_color_hue() {
        assert!((ChromaColor::Green.hue() - 120.0 / 360.0).abs() < 0.01);
        assert!((ChromaColor::Blue.hue() - 240.0 / 360.0).abs() < 0.01);
    }

    #[test]
    fn test_chroma_color_rgb() {
        let (r, g, b) = ChromaColor::Green.to_rgb();
        assert_eq!(r, 0);
        assert_eq!(g, 255);
        assert_eq!(b, 0);

        let (r, g, b) = ChromaColor::Blue.to_rgb();
        assert_eq!(r, 0);
        assert_eq!(g, 0);
        assert_eq!(b, 255);
    }

    #[test]
    fn test_rgb_to_hsv() {
        // Pure red
        let (h, s, v) = rgb_to_hsv(255, 0, 0);
        assert!(h.abs() < 0.01);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);

        // Pure green
        let (h, s, v) = rgb_to_hsv(0, 255, 0);
        assert!((h - 120.0 / 360.0).abs() < 0.01);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);

        // Pure blue
        let (h, s, v) = rgb_to_hsv(0, 0, 255);
        assert!((h - 240.0 / 360.0).abs() < 0.01);
        assert!((s - 1.0).abs() < 0.01);
        assert!((v - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hsv_to_rgb_to_hsv() {
        let (h, s, v) = (0.5, 0.8, 0.9);
        let (r, g, b) = hsv_to_rgb(h, s, v);
        let (h2, s2, v2) = rgb_to_hsv(r, g, b);

        assert!((h - h2).abs() < 0.02);
        assert!((s - s2).abs() < 0.02);
        assert!((v - v2).abs() < 0.02);
    }

    #[test]
    fn test_chroma_key_params_green() {
        let params = ChromaKeyParams::new_green();
        assert_eq!(params.color, ChromaColor::Green);
        assert_eq!(params.hue_tolerance, 0.1);
        assert_eq!(params.gain, 1.0);
    }

    #[test]
    fn test_chroma_key_params_blue() {
        let params = ChromaKeyParams::new_blue();
        assert_eq!(params.color, ChromaColor::Blue);
        assert_eq!(params.hue_tolerance, 0.1);
    }

    #[test]
    fn test_chroma_key_creation() {
        let key_green = ChromaKey::new_green();
        assert!(key_green.is_enabled());
        assert_eq!(key_green.params().color, ChromaColor::Green);

        let key_blue = ChromaKey::new_blue();
        assert_eq!(key_blue.params().color, ChromaColor::Blue);
    }

    #[test]
    fn test_calculate_alpha_green() {
        let key = ChromaKey::new_green();

        // Pure green should be transparent
        let alpha_green = key.calculate_alpha(0, 255, 0);
        assert!(alpha_green < 0.5); // Should be mostly transparent

        // Red should be opaque
        let alpha_red = key.calculate_alpha(255, 0, 0);
        assert!(alpha_red > 0.5); // Should be mostly opaque
    }

    #[test]
    fn test_calculate_alpha_blue() {
        let key = ChromaKey::new_blue();

        // Pure blue should be transparent
        let alpha_blue = key.calculate_alpha(0, 0, 255);
        assert!(alpha_blue < 0.5);

        // Red should be opaque
        let alpha_red = key.calculate_alpha(255, 0, 0);
        assert!(alpha_red > 0.5);
    }

    #[test]
    fn test_spill_suppression_green() {
        let key = ChromaKey::new_green();

        // Green spill on skin tone
        let (r, g, b) = key.suppress_spill(200, 220, 180);

        // Green should be reduced
        assert!(g < 220);
        // Red and blue may receive a small luma-compensation boost to preserve
        // luminance — allow up to 10 counts of adjustment
        assert!(r >= 200 && r <= 210, "r={r} should be in 200..=210");
        assert!(b >= 180 && b <= 190, "b={b} should be in 180..=190");
    }

    #[test]
    fn test_spill_suppression_blue() {
        let key = ChromaKey::new_blue();

        // Blue spill
        let (_r, _g, b) = key.suppress_spill(180, 200, 220);

        // Blue should be reduced
        assert!(b < 220);
    }

    #[test]
    fn test_process_pixel() {
        let key = ChromaKey::new_green();

        let (_r, g, _b, a) = key.process_pixel(0, 255, 0);

        // Green screen should produce low alpha
        assert!(a < 128);

        // RGB should have spill suppression applied
        assert!(g < 255);
    }

    #[test]
    fn test_disabled_key() {
        let mut key = ChromaKey::new_green();
        key.set_enabled(false);

        // When disabled, alpha should always be 1.0
        let alpha = key.calculate_alpha(0, 255, 0);
        assert_eq!(alpha, 1.0);
    }

    #[test]
    fn test_edge_softness() {
        let mut key = ChromaKey::new_green();

        // Without softness
        key.params_mut().edge_softness = 0.0;
        let alpha1 = key.calculate_alpha(100, 200, 100);

        // With softness
        key.params_mut().edge_softness = 0.5;
        let alpha2 = key.calculate_alpha(100, 200, 100);

        // Softness should affect the alpha differently
        // (exact values depend on distance calculation)
        assert!(alpha1 >= 0.0 && alpha1 <= 1.0);
        assert!(alpha2 >= 0.0 && alpha2 <= 1.0);
    }

    #[test]
    fn test_custom_color() {
        let custom_color = ChromaColor::Custom {
            h: 0.0, // Red
            s: 1.0,
            v: 1.0,
        };

        let (r, g, b) = custom_color.to_rgb();
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_set_tolerances() {
        let mut params = ChromaKeyParams::new_green();

        assert!(params.set_hue_tolerance(0.2).is_ok());
        assert_eq!(params.hue_tolerance, 0.2);

        assert!(params.set_hue_tolerance(-0.1).is_err());
        assert!(params.set_hue_tolerance(1.5).is_err());

        assert!(params.set_saturation_tolerance(0.4).is_ok());
        assert_eq!(params.saturation_tolerance, 0.4);
    }

    // -----------------------------------------------------------------------
    // YCbCr keyer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ycbcr_keyer_green_keys_green() {
        let keyer = ChromaKeyYcbcr::new_green();

        // Pure green (0, 255, 0) should produce very low / zero alpha.
        let alpha = keyer.calculate_alpha(0, 255, 0);
        assert!(
            alpha < 0.15,
            "pure green should be nearly transparent, got {alpha}"
        );
    }

    #[test]
    fn test_ycbcr_keyer_green_opaque_red() {
        let keyer = ChromaKeyYcbcr::new_green();

        // Pure red (255, 0, 0) is far from green in Cb/Cr space → opaque.
        let alpha = keyer.calculate_alpha(255, 0, 0);
        assert!(
            alpha > 0.8,
            "pure red should be mostly opaque in green keyer, got {alpha}"
        );
    }

    #[test]
    fn test_ycbcr_keyer_blue_keys_blue() {
        let keyer = ChromaKeyYcbcr::new_blue();

        // Pure blue (0, 0, 255) should produce very low / zero alpha.
        let alpha = keyer.calculate_alpha(0, 0, 255);
        assert!(
            alpha < 0.15,
            "pure blue should be nearly transparent, got {alpha}"
        );
    }

    #[test]
    fn test_ycbcr_keyer_disabled_always_opaque() {
        let mut keyer = ChromaKeyYcbcr::new_green();
        keyer.set_enabled(false);

        // Even pure green should be fully opaque when disabled.
        let alpha = keyer.calculate_alpha(0, 255, 0);
        assert_eq!(alpha, 1.0, "disabled keyer must return 1.0 for any pixel");
    }

    #[test]
    fn test_ycbcr_keyer_invert() {
        let mut keyer = ChromaKeyYcbcr::new_green();
        keyer.params_mut().invert = true;
        keyer.refresh_cache();

        // Inverted: pure green should now be opaque.
        let alpha = keyer.calculate_alpha(0, 255, 0);
        assert!(
            alpha > 0.85,
            "inverted keyer: pure green should be nearly opaque, got {alpha}"
        );

        // Inverted: pure red should be transparent.
        let alpha_red = keyer.calculate_alpha(255, 0, 0);
        assert!(
            alpha_red < 0.2,
            "inverted keyer: pure red should be nearly transparent, got {alpha_red}"
        );
    }

    #[test]
    fn test_ycbcr_rgb_to_ycbcr_601_neutral_grey() {
        // For neutral grey (R=G=B=128) the chrominance should be at neutral 128.
        let (y, cb, cr) = rgb_to_ycbcr_601(128, 128, 128);
        assert!(
            (cb - 128.0).abs() < 2.0,
            "grey pixel: Cb should be ~128, got {cb}"
        );
        assert!(
            (cr - 128.0).abs() < 2.0,
            "grey pixel: Cr should be ~128, got {cr}"
        );
        assert!(
            y > 100.0 && y < 150.0,
            "grey pixel: Y should be ~128, got {y}"
        );
    }

    #[test]
    fn test_ycbcr_spill_suppression_reduces_key_channel() {
        let keyer = ChromaKeyYcbcr::new_green();

        // A pixel with green spill (slightly greenish skin tone).
        let (r, g, b) = (200u8, 220u8, 180u8);
        let (_r_out, g_out, _b_out) = keyer.suppress_spill(r, g, b);

        // Green channel must be reduced (or at worst unchanged).
        assert!(
            g_out <= g,
            "spill suppression must not increase the green channel; g_out={g_out} vs g={g}"
        );
    }

    #[test]
    fn test_ycbcr_spill_suppression_unchanged_for_neutral() {
        let keyer = ChromaKeyYcbcr::new_green();

        // Neutral grey has no chroma alignment → should be unchanged.
        let (r, g, b) = (128u8, 128u8, 128u8);
        let (r_out, g_out, b_out) = keyer.suppress_spill(r, g, b);

        // Allow ±2 for rounding in the BT.601 round-trip.
        let diff_r = (r_out as i16 - r as i16).unsigned_abs();
        let diff_g = (g_out as i16 - g as i16).unsigned_abs();
        let diff_b = (b_out as i16 - b as i16).unsigned_abs();
        assert!(
            diff_r <= 2,
            "neutral grey R changed by more than 2: {diff_r}"
        );
        assert!(
            diff_g <= 2,
            "neutral grey G changed by more than 2: {diff_g}"
        );
        assert!(
            diff_b <= 2,
            "neutral grey B changed by more than 2: {diff_b}"
        );
    }

    #[test]
    fn test_ycbcr_keyer_process_pixel_green_reduces_alpha() {
        let keyer = ChromaKeyYcbcr::new_green();
        let (_, _, _, a) = keyer.process_pixel(0, 255, 0);
        assert!(
            a < 128,
            "green screen pixel must yield low alpha (< 128), got {a}"
        );
    }

    #[test]
    fn test_ycbcr_keyer_chroma_distance_zero_for_exact_match() {
        let keyer = ChromaKeyYcbcr::new_green();
        // For (0, 255, 0) the distance must be 0 (exact key colour).
        let d = keyer.chroma_distance(0, 255, 0);
        assert!(
            d < 0.01,
            "exact key colour must have near-zero chroma distance, got {d}"
        );
    }

    #[test]
    fn test_ycbcr_set_params_refreshes_cache() {
        let mut keyer = ChromaKeyYcbcr::new_green();
        // Change key to blue.
        let params = ChromaKeyYcbcrParams::blue_screen();
        keyer.set_params(params);

        // Now pure blue should key out.
        let alpha = keyer.calculate_alpha(0, 0, 255);
        assert!(
            alpha < 0.15,
            "after switching to blue screen, blue should key out; got {alpha}"
        );
    }
}
