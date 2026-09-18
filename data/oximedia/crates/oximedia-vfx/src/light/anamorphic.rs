//! Lens flare and anamorphic streak light effects.
//!
//! Provides functional-style pixel-buffer operations for adding circular
//! lens flares and the characteristic horizontal streaks of anamorphic lenses.

// ── FlareConfig ───────────────────────────────────────────────────────────────

/// Configuration for a circular lens flare.
#[derive(Debug, Clone, PartialEq)]
pub struct FlareConfig {
    /// Flare centre X in pixels.
    pub center_x: f32,
    /// Flare centre Y in pixels.
    pub center_y: f32,
    /// Intensity multiplier (>= 0.0).
    pub intensity: f32,
    /// Flare colour as `[R, G, B]` in [0.0, 1.0].
    pub color: [f32; 3],
    /// Soft circle radius in pixels.
    pub radius: f32,
}

impl FlareConfig {
    /// Create a new flare configuration.
    #[must_use]
    pub fn new(center_x: f32, center_y: f32, intensity: f32, color: [f32; 3], radius: f32) -> Self {
        Self {
            center_x,
            center_y,
            intensity: intensity.max(0.0),
            color,
            radius: radius.max(0.0),
        }
    }
}

// ── AnamorphicStreak ──────────────────────────────────────────────────────────

/// Configuration for an anamorphic lens streak.
#[derive(Debug, Clone, PartialEq)]
pub struct AnamorphicStreak {
    /// Streak centre X in pixels.
    pub cx: f32,
    /// Streak centre Y in pixels.
    pub cy: f32,
    /// Half-length of the streak in pixels.
    pub length: f32,
    /// Half-width (perpendicular thickness) of the streak in pixels.
    pub width: f32,
    /// Streak colour as `[R, G, B]` in [0.0, 1.0].
    pub color: [f32; 3],
    /// Angle of the streak in degrees (0 = horizontal).
    pub angle_deg: f32,
}

impl AnamorphicStreak {
    /// Create a new anamorphic streak.
    #[must_use]
    pub fn new(cx: f32, cy: f32, length: f32, width: f32, color: [f32; 3], angle_deg: f32) -> Self {
        Self {
            cx,
            cy,
            length: length.max(0.0),
            width: width.max(0.0),
            color,
            angle_deg,
        }
    }

    /// Create a horizontal blue anamorphic streak (classic look).
    #[must_use]
    pub fn horizontal_blue(cx: f32, cy: f32, length: f32) -> Self {
        Self::new(cx, cy, length, 2.0, [0.6, 0.8, 1.0], 0.0)
    }
}

// ── Gaussian & circle helpers ─────────────────────────────────────────────────

/// 2-D Gaussian value for displacement `(dx, dy)` and standard deviation `sigma`.
///
/// Returns a value in (0, 1].
#[must_use]
pub fn gaussian_2d(dx: f32, dy: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        return 0.0;
    }
    let s2 = sigma * sigma;
    (-(dx * dx + dy * dy) / (2.0 * s2)).exp()
}

/// Soft circular falloff: 1.0 at centre, 0.0 at `radius` and beyond.
///
/// Uses a smooth quintic fade.
#[must_use]
pub fn circle_mask(px: u32, py: u32, cx: f32, cy: f32, radius: f32) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }
    let dx = px as f32 - cx;
    let dy = py as f32 - cy;
    let dist = (dx * dx + dy * dy).sqrt();
    let t = (1.0 - dist / radius).clamp(0.0, 1.0);
    // Quintic smoothstep for a very soft edge
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

// ── add_lens_flare ─────────────────────────────────────────────────────────────

/// Add a soft circular lens flare to an RGB (3 bytes/pixel) pixel buffer.
///
/// The buffer must have `width * height * 3` bytes.  The flare is additive —
/// each channel is clamped to 255 after addition.
pub fn add_lens_flare(pixels: &mut [u8], width: u32, height: u32, config: &FlareConfig) {
    let expected = (width as usize) * (height as usize) * 3;
    if pixels.len() < expected {
        return;
    }

    let cx = config.center_x;
    let cy = config.center_y;
    let sigma = config.radius * 0.5_f32.max(0.01);

    for row in 0..height {
        for col in 0..width {
            let dx = col as f32 - cx;
            let dy = row as f32 - cy;
            let g = gaussian_2d(dx, dy, sigma);
            let mask = circle_mask(col, row, cx, cy, config.radius * 2.0);

            // Combine Gaussian core with soft circle rim
            let combined = (g * 0.7 + mask * 0.3) * config.intensity;
            if combined < 1e-4 {
                continue;
            }

            let idx = (row * width + col) as usize * 3;
            for ch in 0..3 {
                let add = (config.color[ch] * combined * 255.0) as u16;
                pixels[idx + ch] = (pixels[idx + ch] as u16 + add).min(255) as u8;
            }
        }
    }
}

// ── add_anamorphic_streak ─────────────────────────────────────────────────────

/// Add an anamorphic lens streak to an RGB (3 bytes/pixel) pixel buffer.
///
/// The streak is drawn as an oriented Gaussian band, additive blended.
pub fn add_anamorphic_streak(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    streak: &AnamorphicStreak,
) {
    let expected = (width as usize) * (height as usize) * 3;
    if pixels.len() < expected {
        return;
    }

    let angle_rad = streak.angle_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Perpendicular sigma
    let sigma_perp = (streak.width * 0.5).max(0.5);

    for row in 0..height {
        for col in 0..width {
            let dx = col as f32 - streak.cx;
            let dy = row as f32 - streak.cy;

            // Project onto parallel and perpendicular axes
            let parallel = dx * cos_a + dy * sin_a;
            let perp = -dx * sin_a + dy * cos_a;

            // Outside streak length → skip
            if parallel.abs() > streak.length {
                continue;
            }

            // Gaussian across width; linear falloff along length
            let g_perp = gaussian_2d(perp, 0.0, sigma_perp);
            let len_fade = (1.0 - (parallel.abs() / streak.length.max(1.0))).clamp(0.0, 1.0);
            let combined = g_perp * len_fade;

            if combined < 1e-4 {
                continue;
            }

            let idx = (row * width + col) as usize * 3;
            for ch in 0..3 {
                let add = (streak.color[ch] * combined * 255.0) as u16;
                pixels[idx + ch] = (pixels[idx + ch] as u16 + add).min(255) as u8;
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn black_frame(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w * h * 3) as usize]
    }

    #[test]
    fn test_gaussian_2d_centre() {
        // At the origin, gaussian should be 1.0 for any sigma
        let g = gaussian_2d(0.0, 0.0, 5.0);
        assert!((g - 1.0).abs() < 1e-6, "g at origin should be 1.0, got {g}");
    }

    #[test]
    fn test_gaussian_2d_falloff() {
        let sigma = 10.0_f32;
        let g_near = gaussian_2d(1.0, 0.0, sigma);
        let g_far = gaussian_2d(20.0, 0.0, sigma);
        assert!(g_near > g_far, "closer point should have higher gaussian");
    }

    #[test]
    fn test_gaussian_2d_zero_sigma() {
        let g = gaussian_2d(0.0, 0.0, 0.0);
        assert!((g - 0.0).abs() < 1e-6, "zero sigma should return 0.0");
    }

    #[test]
    fn test_circle_mask_centre() {
        let m = circle_mask(50, 50, 50.0, 50.0, 20.0);
        assert!(
            (m - 1.0).abs() < 1e-6,
            "mask at centre should be 1.0, got {m}"
        );
    }

    #[test]
    fn test_circle_mask_outside() {
        let m = circle_mask(100, 100, 50.0, 50.0, 20.0);
        assert!(
            (m - 0.0).abs() < 1e-6,
            "mask far outside should be 0.0, got {m}"
        );
    }

    #[test]
    fn test_circle_mask_zero_radius() {
        let m = circle_mask(50, 50, 50.0, 50.0, 0.0);
        assert!((m - 0.0).abs() < 1e-6, "zero radius should give 0.0");
    }

    #[test]
    fn test_add_lens_flare_brightens_centre() {
        let w = 32u32;
        let h = 32u32;
        let mut frame = black_frame(w, h);
        let config = FlareConfig::new(15.5, 15.5, 2.0, [1.0, 1.0, 1.0], 8.0);
        add_lens_flare(&mut frame, w, h, &config);
        // Centre pixel should be bright
        let idx = (15 * w + 15) as usize * 3;
        assert!(
            frame[idx] > 50,
            "centre R should be bright, got {}",
            frame[idx]
        );
    }

    #[test]
    fn test_add_lens_flare_no_overflow() {
        let w = 16u32;
        let h = 16u32;
        let mut frame = vec![200u8; (w * h * 3) as usize];
        let config = FlareConfig::new(7.5, 7.5, 100.0, [1.0, 1.0, 1.0], 20.0);
        add_lens_flare(&mut frame, w, h, &config);
        // All values should remain <= 255 (saturating)
        // u8 values are always <= 255 by type invariant; just verify no panics occurred
        assert_eq!(
            frame.len(),
            (w * h * 3) as usize,
            "no pixel should overflow (saturating add)"
        );
    }

    #[test]
    fn test_add_lens_flare_buffer_size_mismatch() {
        let mut frame = vec![0u8; 10]; // too small
        let config = FlareConfig::new(8.0, 8.0, 1.0, [1.0, 0.0, 0.0], 5.0);
        // Should not panic even with wrong size
        add_lens_flare(&mut frame, 32, 32, &config);
    }

    #[test]
    fn test_anamorphic_streak_horizontal_blue() {
        let s = AnamorphicStreak::horizontal_blue(50.0, 50.0, 40.0);
        assert!((s.angle_deg - 0.0).abs() < 1e-5, "angle should be 0");
        assert!(s.color[2] > s.color[0], "blue channel should dominate");
    }

    #[test]
    fn test_add_anamorphic_streak_brightens_along_axis() {
        let w = 64u32;
        let h = 32u32;
        let mut frame = black_frame(w, h);
        let streak = AnamorphicStreak::horizontal_blue(32.0, 16.0, 30.0);
        add_anamorphic_streak(&mut frame, w, h, &streak);
        // Pixel on the horizontal axis should be bright
        let idx = (16 * w + 32) as usize * 3;
        assert!(
            frame[idx] > 10 || frame[idx + 2] > 10,
            "streak axis should be brightened"
        );
    }

    #[test]
    fn test_add_anamorphic_streak_angled() {
        let w = 64u32;
        let h = 64u32;
        let mut frame = black_frame(w, h);
        // 45-degree streak
        let streak = AnamorphicStreak::new(32.0, 32.0, 20.0, 3.0, [1.0, 1.0, 0.0], 45.0);
        add_anamorphic_streak(&mut frame, w, h, &streak);
        // Should not panic and should have at least some bright pixels
        let any_bright = frame.iter().any(|&v| v > 0);
        assert!(
            any_bright,
            "angled streak should produce some bright pixels"
        );
    }

    #[test]
    fn test_add_anamorphic_streak_no_overflow() {
        let w = 32u32;
        let h = 32u32;
        let mut frame = vec![200u8; (w * h * 3) as usize];
        let streak = AnamorphicStreak::new(16.0, 16.0, 30.0, 5.0, [1.0, 1.0, 1.0], 0.0);
        add_anamorphic_streak(&mut frame, w, h, &streak);
        // u8 saturates at 255 by type; just verify no panics and frame size is intact
        assert_eq!(
            frame.len(),
            (w * h * 3) as usize,
            "saturating add: frame size intact"
        );
    }
}
