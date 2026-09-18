//! Luma key implementation for video switchers.
//!
//! Luma keying uses the brightness (luminance) of a video signal to create transparency.
//!
//! ## Soft-edge Luma Keying
//!
//! Professional broadcast switchers use a smoothstep-based keying curve rather than a
//! hard threshold.  The alpha transition from transparent to opaque is spread over a
//! soft-edge band of width `2 × softness` centred on the `clip` level:
//!
//! ```text
//! lo  = clip - softness
//! hi  = clip + softness
//! t   = clamp((luma - lo) / (hi - lo), 0, 1)
//! raw_alpha = t * t * (3 - 2*t)          // Hermite / smoothstep
//! ```
//!
//! When `softness == 0` this degenerates to the familiar hard-threshold formula.

use oximedia_codec::VideoFrame;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during luma keying.
#[derive(Error, Debug, Clone)]
pub enum LumaKeyError {
    #[error("Invalid clip value: {0}")]
    InvalidClip(f32),

    #[error("Invalid gain value: {0}")]
    InvalidGain(f32),

    #[error("Invalid softness value: {0}")]
    InvalidSoftness(f32),

    #[error("Frame dimension mismatch")]
    DimensionMismatch,

    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Luma key parameters.
///
/// The keying curve is controlled by three interrelated values:
///
/// | Parameter | Range       | Role                                                     |
/// |-----------|-------------|----------------------------------------------------------|
/// | `clip`    | 0.0 – 1.0   | Centre of the transparency/opacity boundary.             |
/// | `softness`| 0.0 – 0.5   | Half-width of the smoothstep transition band.            |
/// | `gain`    | 0.0 – 2.0   | Post-smoothstep scale factor.                            |
///
/// When `softness = 0` the key is a hard threshold: pixels whose luminance is
/// strictly below `clip` are fully transparent; those at or above are fully opaque
/// (before `gain` scaling).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LumaKeyParams {
    /// Clip level (0.0 - 1.0) - centre of the transparency/opacity boundary.
    pub clip: f32,
    /// Edge softness (0.0 - 0.5).
    ///
    /// Defines the half-width of the smoothstep transition zone.  At `0.0` the
    /// boundary is a hard edge; at `0.5` the full luminance range `[0, 1]` is
    /// used as the transition ramp.
    pub softness: f32,
    /// Gain (0.0 - 2.0) - amplifies the post-smoothstep key signal.
    pub gain: f32,
    /// Invert the key — swap transparent and opaque regions.
    pub invert: bool,
    /// Pre-multiply alpha — multiply RGB channels by the computed alpha.
    pub pre_multiply: bool,
}

impl LumaKeyParams {
    /// Create new luma key parameters with defaults.
    ///
    /// Default clip = 0.5, softness = 0.0 (hard edge), gain = 1.0.
    pub fn new() -> Self {
        Self {
            clip: 0.5,
            softness: 0.0,
            gain: 1.0,
            invert: false,
            pre_multiply: false,
        }
    }

    /// Create parameters configured for a soft broadcast-grade luma key.
    ///
    /// Uses `clip = 0.5` with a `softness = 0.1` transition band.
    pub fn soft() -> Self {
        Self {
            clip: 0.5,
            softness: 0.1,
            gain: 1.0,
            invert: false,
            pre_multiply: false,
        }
    }

    /// Set the clip level.
    pub fn set_clip(&mut self, clip: f32) -> Result<(), LumaKeyError> {
        if !(0.0..=1.0).contains(&clip) {
            return Err(LumaKeyError::InvalidClip(clip));
        }
        self.clip = clip;
        Ok(())
    }

    /// Set the edge softness.
    ///
    /// `softness` must be in `[0.0, 0.5]`.  A value of `0.0` produces a hard
    /// threshold; `0.5` makes the entire `[0, 1]` luminance range part of the
    /// transition band (always in soft-edge territory).
    pub fn set_softness(&mut self, softness: f32) -> Result<(), LumaKeyError> {
        if !(0.0..=0.5).contains(&softness) {
            return Err(LumaKeyError::InvalidSoftness(softness));
        }
        self.softness = softness;
        Ok(())
    }

    /// Set the gain.
    pub fn set_gain(&mut self, gain: f32) -> Result<(), LumaKeyError> {
        if !(0.0..=2.0).contains(&gain) {
            return Err(LumaKeyError::InvalidGain(gain));
        }
        self.gain = gain;
        Ok(())
    }

    /// Set invert flag.
    pub fn set_invert(&mut self, invert: bool) {
        self.invert = invert;
    }

    /// Set pre-multiply flag.
    pub fn set_pre_multiply(&mut self, pre_multiply: bool) {
        self.pre_multiply = pre_multiply;
    }
}

impl Default for LumaKeyParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Luma key processor.
pub struct LumaKey {
    params: LumaKeyParams,
    enabled: bool,
}

impl LumaKey {
    /// Create a new luma key processor.
    pub fn new() -> Self {
        Self {
            params: LumaKeyParams::new(),
            enabled: true,
        }
    }

    /// Create with specific parameters.
    pub fn with_params(params: LumaKeyParams) -> Self {
        Self {
            params,
            enabled: true,
        }
    }

    /// Get the parameters.
    pub fn params(&self) -> &LumaKeyParams {
        &self.params
    }

    /// Get mutable parameters.
    pub fn params_mut(&mut self) -> &mut LumaKeyParams {
        &mut self.params
    }

    /// Set parameters.
    pub fn set_params(&mut self, params: LumaKeyParams) {
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

    /// Calculate luminance from RGB values.
    pub fn calculate_luminance(&self, r: u8, g: u8, b: u8) -> f32 {
        // ITU-R BT.709 coefficients
        let r_norm = r as f32 / 255.0;
        let g_norm = g as f32 / 255.0;
        let b_norm = b as f32 / 255.0;

        0.2126 * r_norm + 0.7152 * g_norm + 0.0722 * b_norm
    }

    /// Calculate alpha from luminance using the smoothstep keying curve.
    ///
    /// The algorithm follows the broadcast-standard soft-edge luma key:
    ///
    /// ```text
    /// lo        = clip - softness
    /// hi        = clip + softness
    /// t         = clamp((luma - lo) / (hi - lo), 0, 1)
    /// raw_alpha = t * t * (3 - 2*t)          // Hermite smoothstep
    /// alpha     = clamp(raw_alpha * gain, 0, 1)
    /// ```
    ///
    /// When `softness = 0` the denominator `(hi - lo)` is zero and the formula
    /// degenerates to a hard threshold:
    /// - `luma < clip`  → `alpha = 0`
    /// - `luma >= clip` → `alpha = 1` (before gain)
    ///
    /// The `invert` flag swaps transparent and opaque regions after all other
    /// processing.
    pub fn calculate_alpha(&self, luminance: f32) -> f32 {
        if !self.enabled {
            return 1.0;
        }

        let raw_alpha = self.smoothstep_alpha(luminance);

        // Apply gain and clamp.
        let mut alpha = (raw_alpha * self.params.gain).clamp(0.0, 1.0);

        // Invert if requested.
        if self.params.invert {
            alpha = 1.0 - alpha;
        }

        alpha
    }

    /// Compute the raw smoothstep alpha for a given luminance value.
    ///
    /// This is the core keying curve without gain or invert applied, making it
    /// useful when callers need access to the un-scaled matte value.
    ///
    /// The result is always in `[0.0, 1.0]`.
    ///
    /// ### Mathematical detail
    ///
    /// Let `lo = clip - softness`, `hi = clip + softness`.
    ///
    /// * If `softness == 0`: returns `0.0` for `luma < clip`, `1.0` otherwise.
    /// * Otherwise: computes the cubic Hermite polynomial
    ///   `t²(3 − 2t)` where `t = (luma − lo) / (hi − lo)`, clamped to `[0, 1]`.
    pub fn smoothstep_alpha(&self, luminance: f32) -> f32 {
        let clip = self.params.clip;
        let softness = self.params.softness;

        if softness <= 0.0 {
            // Hard-edge threshold.
            return if luminance >= clip { 1.0 } else { 0.0 };
        }

        let lo = clip - softness;
        let hi = clip + softness;
        let band = hi - lo; // = 2 * softness, always > 0

        // Normalised position within the transition band.
        let t = ((luminance - lo) / band).clamp(0.0, 1.0);

        // Cubic Hermite smoothstep: t²(3 − 2t)
        t * t * (3.0 - 2.0 * t)
    }

    /// Process a single pixel.
    pub fn process_pixel(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8, u8) {
        let luminance = self.calculate_luminance(r, g, b);
        let alpha = self.calculate_alpha(luminance);
        let alpha_u8 = (alpha * 255.0) as u8;

        if self.params.pre_multiply {
            // Pre-multiply RGB by alpha
            let r_out = ((r as f32) * alpha) as u8;
            let g_out = ((g as f32) * alpha) as u8;
            let b_out = ((b as f32) * alpha) as u8;
            (r_out, g_out, b_out, alpha_u8)
        } else {
            (r, g, b, alpha_u8)
        }
    }

    /// Process a video frame to extract alpha channel based on luminance.
    ///
    /// For planar YUV the luma plane is used directly.
    /// For frames with at least 3 planes, approximate RGB is reconstructed
    /// from YCbCr and then BT.709 luminance is computed.
    ///
    /// Returns a `Vec<u8>` with one alpha byte per luma-plane pixel
    /// (0 = fully transparent / keyed, 255 = fully opaque).
    pub fn process_frame(&self, fill: &VideoFrame) -> Result<Vec<u8>, LumaKeyError> {
        if fill.planes.is_empty() {
            return Err(LumaKeyError::ProcessingError(
                "Frame has no planes".to_string(),
            ));
        }

        let luma_plane = &fill.planes[0];
        let luma_w = luma_plane.width as usize;
        let luma_h = luma_plane.height as usize;
        let pixel_count = luma_w * luma_h;

        if fill.planes.len() >= 3 {
            let cb_plane = &fill.planes[1];
            let cr_plane = &fill.planes[2];

            let cb_w = cb_plane.width as usize;
            let h_ratio = luma_w.checked_div(cb_w).unwrap_or(1);
            let v_ratio = luma_h.checked_div(cb_plane.height as usize).unwrap_or(1);

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

                    let luminance = self.calculate_luminance(r, g, b);
                    let alpha = self.calculate_alpha(luminance);
                    alpha_out.push((alpha * 255.0) as u8);
                }
            }

            Ok(alpha_out)
        } else {
            // Single plane: use luma directly as luminance
            let mut alpha_out = Vec::with_capacity(pixel_count);
            for &luma in &luma_plane.data[..pixel_count.min(luma_plane.data.len())] {
                let luminance = luma as f32 / 255.0;
                let alpha = self.calculate_alpha(luminance);
                alpha_out.push((alpha * 255.0) as u8);
            }
            Ok(alpha_out)
        }
    }
}

impl Default for LumaKey {
    fn default() -> Self {
        Self::new()
    }
}

/// Luma key mask for rectangular regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LumaKeyMask {
    /// Top edge (0.0 - 1.0)
    pub top: f32,
    /// Bottom edge (0.0 - 1.0)
    pub bottom: f32,
    /// Left edge (0.0 - 1.0)
    pub left: f32,
    /// Right edge (0.0 - 1.0)
    pub right: f32,
    /// Enabled state
    pub enabled: bool,
}

impl LumaKeyMask {
    /// Create a new mask with default values (full frame).
    pub fn new() -> Self {
        Self {
            top: 0.0,
            bottom: 1.0,
            left: 0.0,
            right: 1.0,
            enabled: false,
        }
    }

    /// Check if a point is inside the mask.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        if !self.enabled {
            return true;
        }

        x >= self.left && x <= self.right && y >= self.top && y <= self.bottom
    }

    /// Set the mask region.
    pub fn set_region(&mut self, top: f32, bottom: f32, left: f32, right: f32) {
        self.top = top.clamp(0.0, 1.0);
        self.bottom = bottom.clamp(0.0, 1.0);
        self.left = left.clamp(0.0, 1.0);
        self.right = right.clamp(0.0, 1.0);
    }

    /// Enable the mask.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the mask.
    pub fn disable(&mut self) {
        self.enabled = false;
    }
}

impl Default for LumaKeyMask {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luma_key_params_defaults() {
        let params = LumaKeyParams::new();
        assert_eq!(params.clip, 0.5);
        assert_eq!(params.gain, 1.0);
        assert!(!params.invert);
        assert!(!params.pre_multiply);
    }

    #[test]
    fn test_set_clip() {
        let mut params = LumaKeyParams::new();

        assert!(params.set_clip(0.3).is_ok());
        assert_eq!(params.clip, 0.3);

        assert!(params.set_clip(-0.1).is_err());
        assert!(params.set_clip(1.5).is_err());
    }

    #[test]
    fn test_set_gain() {
        let mut params = LumaKeyParams::new();

        assert!(params.set_gain(1.5).is_ok());
        assert_eq!(params.gain, 1.5);

        assert!(params.set_gain(-0.1).is_err());
        assert!(params.set_gain(2.5).is_err());
    }

    #[test]
    fn test_luma_key_creation() {
        let key = LumaKey::new();
        assert!(key.is_enabled());
        assert_eq!(key.params().clip, 0.5);
        assert_eq!(key.params().gain, 1.0);
    }

    #[test]
    fn test_calculate_luminance() {
        let key = LumaKey::new();

        // Pure white
        let luma_white = key.calculate_luminance(255, 255, 255);
        assert!((luma_white - 1.0).abs() < 0.01);

        // Pure black
        let luma_black = key.calculate_luminance(0, 0, 0);
        assert!(luma_black.abs() < 0.01);

        // Middle gray
        let luma_gray = key.calculate_luminance(128, 128, 128);
        assert!((luma_gray - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_calculate_alpha_hard_edge() {
        // With softness=0 (default) the key is a hard threshold.
        let mut key = LumaKey::new();
        key.params_mut().clip = 0.5;
        key.params_mut().softness = 0.0;
        key.params_mut().gain = 1.0;

        // Luminance below clip → fully transparent.
        let alpha_below = key.calculate_alpha(0.3);
        assert_eq!(alpha_below, 0.0, "below clip: alpha must be 0.0");

        // Luminance at clip → fully opaque (hard threshold, >=).
        let alpha_at = key.calculate_alpha(0.5);
        assert_eq!(alpha_at, 1.0, "at clip: alpha must be 1.0 (hard edge)");

        // Luminance above clip → fully opaque.
        let alpha_above = key.calculate_alpha(0.8);
        assert_eq!(alpha_above, 1.0, "above clip: alpha must be 1.0");
    }

    #[test]
    fn test_calculate_alpha_soft_edge_smoothstep() {
        // With softness=0.1, clip=0.5 the transition is [0.4, 0.6].
        let mut key = LumaKey::new();
        key.params_mut().clip = 0.5;
        key.params_mut().softness = 0.1;
        key.params_mut().gain = 1.0;

        // Below transition band → fully transparent.
        assert_eq!(key.calculate_alpha(0.3), 0.0);

        // At centre of transition band (luma = 0.5 = clip):
        // t = (0.5 - 0.4) / 0.2 = 0.5
        // smoothstep = 0.5² * (3 - 2*0.5) = 0.25 * 2.0 = 0.5
        let alpha_mid = key.calculate_alpha(0.5);
        assert!(
            (alpha_mid - 0.5).abs() < 0.001,
            "midpoint: expected 0.5, got {alpha_mid}"
        );

        // Above transition band → fully opaque.
        assert_eq!(key.calculate_alpha(0.7), 1.0);
    }

    #[test]
    fn test_smoothstep_monotonically_increasing() {
        let mut key = LumaKey::new();
        key.params_mut().clip = 0.5;
        key.params_mut().softness = 0.2;
        key.params_mut().gain = 1.0;

        // Sample 11 points across the full [0, 1] range.
        let mut prev = key.calculate_alpha(0.0);
        for i in 1..=10 {
            let luma = i as f32 / 10.0;
            let alpha = key.calculate_alpha(luma);
            assert!(
                alpha >= prev - 1e-6,
                "smoothstep must be monotonically non-decreasing: luma={luma}, prev={prev}, alpha={alpha}"
            );
            prev = alpha;
        }
    }

    #[test]
    fn test_invert_key() {
        // With hard edge, invert flips 0→1 and 1→0.
        let mut key = LumaKey::new();
        key.params_mut().clip = 0.5;
        key.params_mut().softness = 0.0;
        key.params_mut().gain = 1.0;
        key.params_mut().invert = true;

        // Below clip normally transparent → inverted to opaque.
        let alpha_below = key.calculate_alpha(0.3);
        assert_eq!(alpha_below, 1.0, "inverted below clip: expected 1.0");

        // Above clip normally opaque → inverted to transparent.
        let alpha_above = key.calculate_alpha(0.7);
        assert_eq!(alpha_above, 0.0, "inverted above clip: expected 0.0");
    }

    #[test]
    fn test_invert_softedge_key() {
        // With softness, the inverted mid-point should be 1 - 0.5 = 0.5.
        let mut key = LumaKey::new();
        key.params_mut().clip = 0.5;
        key.params_mut().softness = 0.1;
        key.params_mut().gain = 1.0;
        key.params_mut().invert = true;

        let alpha_mid = key.calculate_alpha(0.5);
        assert!(
            (alpha_mid - 0.5).abs() < 0.001,
            "inverted soft mid-point should be 0.5, got {alpha_mid}"
        );
    }

    #[test]
    fn test_process_pixel() {
        let key = LumaKey::new();

        let (r, g, b, a) = key.process_pixel(255, 255, 255);
        assert_eq!(r, 255);
        assert_eq!(g, 255);
        assert_eq!(b, 255);
        assert!(a > 0); // White should produce some alpha
    }

    #[test]
    fn test_process_pixel_pre_multiply() {
        let mut key = LumaKey::new();
        key.params_mut().pre_multiply = true;
        key.params_mut().clip = 0.5;

        let (_r, _g, _b, a) = key.process_pixel(255, 255, 255);
        // With pre-multiply, RGB should be scaled by alpha
        assert!(a > 0);
    }

    #[test]
    fn test_luma_key_mask_creation() {
        let mask = LumaKeyMask::new();
        assert_eq!(mask.top, 0.0);
        assert_eq!(mask.bottom, 1.0);
        assert_eq!(mask.left, 0.0);
        assert_eq!(mask.right, 1.0);
        assert!(!mask.enabled);
    }

    #[test]
    fn test_mask_contains() {
        let mut mask = LumaKeyMask::new();
        mask.set_region(0.25, 0.75, 0.25, 0.75);
        mask.enable();

        assert!(mask.contains(0.5, 0.5)); // Center
        assert!(!mask.contains(0.1, 0.1)); // Outside
        assert!(!mask.contains(0.9, 0.9)); // Outside
    }

    #[test]
    fn test_mask_disabled() {
        let mut mask = LumaKeyMask::new();
        mask.set_region(0.25, 0.75, 0.25, 0.75);
        // Don't enable

        // When disabled, all points should be inside
        assert!(mask.contains(0.1, 0.1));
        assert!(mask.contains(0.9, 0.9));
    }

    #[test]
    fn test_enable_disable_key() {
        let mut key = LumaKey::new();
        assert!(key.is_enabled());

        key.set_enabled(false);
        assert!(!key.is_enabled());

        // When disabled, alpha should always be 1.0
        let alpha = key.calculate_alpha(0.0);
        assert_eq!(alpha, 1.0);
    }

    #[test]
    fn test_gain_amplification_with_softness() {
        // Use softness=0.4, clip=0.5, gain=2.0.
        // Transition band is [0.1, 0.9].
        // At luma=0.3: t = (0.3 - 0.1) / 0.8 = 0.25
        //   smoothstep = 0.25² * (3 - 2*0.25) = 0.0625 * 2.5 = 0.15625
        //   with gain=2: 0.15625 * 2 = 0.3125 (not clamped)
        let mut key = LumaKey::new();
        key.params_mut().clip = 0.5;
        key.params_mut().softness = 0.4;
        key.params_mut().gain = 2.0;

        let alpha_low = key.calculate_alpha(0.3);
        let expected_low = {
            let t = (0.3_f32 - 0.1) / 0.8;
            let s = t * t * (3.0 - 2.0 * t);
            (s * 2.0).clamp(0.0, 1.0)
        };
        assert!(
            (alpha_low - expected_low).abs() < 0.001,
            "gain 2x low: expected {expected_low}, got {alpha_low}"
        );

        // At luma=0.8: smoothstep is large enough that gain*2 saturates to 1.0.
        let alpha_high = key.calculate_alpha(0.8);
        assert_eq!(alpha_high, 1.0, "gain 2x high: must saturate at 1.0");
    }

    #[test]
    fn test_softness_params() {
        let mut params = LumaKeyParams::new();

        // Valid values.
        assert!(params.set_softness(0.0).is_ok());
        assert!(params.set_softness(0.25).is_ok());
        assert!(params.set_softness(0.5).is_ok());

        // Out-of-range values.
        assert!(params.set_softness(-0.01).is_err());
        assert!(params.set_softness(0.51).is_err());
        assert_eq!(params.softness, 0.5); // last successful set
    }

    #[test]
    fn test_soft_preset() {
        let params = LumaKeyParams::soft();
        assert_eq!(params.clip, 0.5);
        assert_eq!(params.softness, 0.1);
        assert_eq!(params.gain, 1.0);
        assert!(!params.invert);
    }
}
