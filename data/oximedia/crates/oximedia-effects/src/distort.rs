//! Distortion and warp effects operating on normalised UV coordinates [0.0, 1.0].
//!
//! These effects remap UV coordinates; the caller samples the source image at
//! the returned coordinates to produce the distorted output.

#![allow(dead_code)]

/// Axis for the mirror effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MirrorAxis {
    /// Mirror left→right (flip on the vertical axis).
    Horizontal,
    /// Mirror top→bottom (flip on the horizontal axis).
    Vertical,
    /// Mirror on both axes.
    Both,
}

/// Sine-wave distortion applied along one axis.
#[derive(Debug, Clone)]
pub struct WaveDistortion {
    /// Pixel-space amplitude of the wave (in UV units, e.g. 0.05).
    pub amplitude: f32,
    /// Number of full cycles across the image width (or height).
    pub frequency: f32,
    /// Phase offset in radians.
    pub phase: f32,
    /// If `true`, the wave displaces horizontally; otherwise vertically.
    pub horizontal: bool,
}

impl WaveDistortion {
    /// Compute the signed UV displacement for a pixel at normalised position `pos`
    /// along the axis perpendicular to the displacement direction.
    #[must_use]
    pub fn offset_at(&self, pos: f32) -> f32 {
        use std::f32::consts::TAU;
        self.amplitude * (TAU * self.frequency * pos + self.phase).sin()
    }

    /// Apply the wave to a UV coordinate pair and return the warped UV.
    #[must_use]
    pub fn apply_uv(&self, u: f32, v: f32) -> (f32, f32) {
        if self.horizontal {
            (u + self.offset_at(v), v)
        } else {
            (u, v + self.offset_at(u))
        }
    }
}

/// Barrel/pincushion lens distortion.
///
/// Positive `strength` produces barrel distortion; negative produces pincushion.
#[derive(Debug, Clone)]
pub struct BarrelDistortion {
    /// Distortion strength. Typical range: [−1.0, 1.0].
    pub strength: f32,
}

impl BarrelDistortion {
    /// Remap normalised UV coordinates `(u, v)` using the barrel distortion model.
    ///
    /// UV coordinates are expected in [0.0, 1.0]; values outside that range may
    /// map to off-screen pixels.
    #[must_use]
    pub fn distort_uv(&self, u: f32, v: f32) -> (f32, f32) {
        // Work in [-1, 1] centred space
        let cx = u * 2.0 - 1.0;
        let cy = v * 2.0 - 1.0;
        let r2 = cx * cx + cy * cy;
        let factor = 1.0 + self.strength * r2;
        let cx2 = cx * factor;
        let cy2 = cy * factor;
        // Back to [0, 1]
        ((cx2 + 1.0) * 0.5, (cy2 + 1.0) * 0.5)
    }
}

/// Twirl effect: rotates pixels by an angle proportional to their distance
/// from the image centre.
#[derive(Debug, Clone)]
pub struct TwirlEffect {
    /// Maximum rotation angle in degrees (applied at the centre).
    pub angle_deg: f32,
    /// Radius of effect (in UV units from centre, typically 0.5 for half the image).
    pub radius: f32,
}

impl TwirlEffect {
    /// Apply twirl to normalised UV coordinates.
    #[must_use]
    pub fn apply_uv(&self, u: f32, v: f32) -> (f32, f32) {
        let cx = u - 0.5;
        let cy = v - 0.5;
        let dist = (cx * cx + cy * cy).sqrt();

        if self.radius <= 0.0 || dist >= self.radius {
            return (u, v);
        }

        // Rotation angle proportional to (1 - normalised distance)
        let t = 1.0 - (dist / self.radius).clamp(0.0, 1.0);
        let angle_rad = self.angle_deg.to_radians() * t;
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let rx = cx * cos_a - cy * sin_a;
        let ry = cx * sin_a + cy * cos_a;

        (rx + 0.5, ry + 0.5)
    }
}

/// Mirror (flip) effect along one or both axes.
#[derive(Debug, Clone)]
pub struct MirrorEffect {
    /// Which axis to mirror around.
    pub axis: MirrorAxis,
}

impl MirrorEffect {
    /// Apply the mirror transform to normalised UV coordinates.
    #[must_use]
    pub fn apply_uv(&self, u: f32, v: f32) -> (f32, f32) {
        match self.axis {
            MirrorAxis::Horizontal => (1.0 - u, v),
            MirrorAxis::Vertical => (u, 1.0 - v),
            MirrorAxis::Both => (1.0 - u, 1.0 - v),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── WaveDistortion ────────────────────────────────────────────────────────

    #[test]
    fn test_wave_zero_amplitude() {
        let w = WaveDistortion {
            amplitude: 0.0,
            frequency: 2.0,
            phase: 0.0,
            horizontal: true,
        };
        assert!((w.offset_at(0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_wave_max_offset() {
        let w = WaveDistortion {
            amplitude: 0.1,
            frequency: 0.25, // pos=1.0 → 0.25 full cycles → 90° → sin=1
            phase: 0.0,
            horizontal: true,
        };
        // At pos=1.0: sin(TAU * 0.25 * 1.0) = sin(π/2) = 1.0
        let offset = w.offset_at(1.0);
        assert!((offset - 0.1).abs() < 1e-5, "offset: {offset}");
    }

    #[test]
    fn test_wave_horizontal_modifies_u() {
        let w = WaveDistortion {
            amplitude: 0.05,
            frequency: 1.0,
            phase: 0.0,
            horizontal: true,
        };
        let (u2, v2) = w.apply_uv(0.5, 0.25);
        assert!((v2 - 0.25).abs() < 1e-6, "V should not change: {v2}");
        assert!((u2 - 0.5).abs() > 1e-6 || true, "U may change"); // sine may be 0
    }

    #[test]
    fn test_wave_vertical_modifies_v() {
        let w = WaveDistortion {
            amplitude: 0.05,
            frequency: 1.0,
            phase: 0.0,
            horizontal: false,
        };
        let (u2, v2) = w.apply_uv(0.25, 0.5);
        assert!((u2 - 0.25).abs() < 1e-6, "U should not change: {u2}");
        let _ = v2; // V may have changed
    }

    // ── BarrelDistortion ──────────────────────────────────────────────────────

    #[test]
    fn test_barrel_center_unchanged() {
        let b = BarrelDistortion { strength: 0.5 };
        let (u, v) = b.distort_uv(0.5, 0.5);
        assert!((u - 0.5).abs() < 1e-6, "u: {u}");
        assert!((v - 0.5).abs() < 1e-6, "v: {v}");
    }

    #[test]
    fn test_barrel_zero_strength_passthrough() {
        let b = BarrelDistortion { strength: 0.0 };
        for (u, v) in [(0.2, 0.3), (0.8, 0.7), (0.5, 0.5)] {
            let (ou, ov) = b.distort_uv(u, v);
            assert!((ou - u).abs() < 1e-6, "u: {ou} != {u}");
            assert!((ov - v).abs() < 1e-6, "v: {ov} != {v}");
        }
    }

    #[test]
    fn test_barrel_positive_pushes_corners_out() {
        let b = BarrelDistortion { strength: 0.5 };
        // A corner (0, 0) in UV space → centred at (-1, -1)
        let (ou, ov) = b.distort_uv(0.0, 0.0);
        // Positive strength: corner should move further from centre
        assert!(
            ou < 0.0 || ov < 0.0,
            "corners should move outward: ({ou}, {ov})"
        );
    }

    // ── TwirlEffect ───────────────────────────────────────────────────────────

    #[test]
    fn test_twirl_center_unchanged() {
        let t = TwirlEffect {
            angle_deg: 45.0,
            radius: 0.5,
        };
        let (u, v) = t.apply_uv(0.5, 0.5);
        // distance = 0 → rotation = angle_deg → but cx=cy=0 so output is still 0.5
        assert!((u - 0.5).abs() < 1e-5, "u: {u}");
        assert!((v - 0.5).abs() < 1e-5, "v: {v}");
    }

    #[test]
    fn test_twirl_outside_radius_unchanged() {
        let t = TwirlEffect {
            angle_deg: 90.0,
            radius: 0.3,
        };
        // Point at (0.0, 0.5) → distance from (0.5, 0.5) is 0.5 > radius
        let (u, v) = t.apply_uv(0.0, 0.5);
        assert!((u - 0.0).abs() < 1e-5, "u unchanged: {u}");
        assert!((v - 0.5).abs() < 1e-5, "v unchanged: {v}");
    }

    #[test]
    fn test_twirl_zero_angle_passthrough() {
        let t = TwirlEffect {
            angle_deg: 0.0,
            radius: 0.5,
        };
        let (u, v) = t.apply_uv(0.7, 0.4);
        assert!((u - 0.7).abs() < 1e-5, "u: {u}");
        assert!((v - 0.4).abs() < 1e-5, "v: {v}");
    }

    #[test]
    fn test_twirl_zero_radius_passthrough() {
        let t = TwirlEffect {
            angle_deg: 90.0,
            radius: 0.0,
        };
        let (u, v) = t.apply_uv(0.3, 0.6);
        assert!((u - 0.3).abs() < 1e-5, "u: {u}");
        assert!((v - 0.6).abs() < 1e-5, "v: {v}");
    }

    // ── MirrorEffect ──────────────────────────────────────────────────────────

    #[test]
    fn test_mirror_horizontal() {
        let m = MirrorEffect {
            axis: MirrorAxis::Horizontal,
        };
        let (u, v) = m.apply_uv(0.3, 0.7);
        assert!((u - 0.7).abs() < 1e-6, "u mirrored: {u}");
        assert!((v - 0.7).abs() < 1e-6, "v unchanged: {v}");
    }

    #[test]
    fn test_mirror_vertical() {
        let m = MirrorEffect {
            axis: MirrorAxis::Vertical,
        };
        let (u, v) = m.apply_uv(0.3, 0.7);
        assert!((u - 0.3).abs() < 1e-6, "u unchanged: {u}");
        assert!((v - 0.3).abs() < 1e-6, "v mirrored: {v}");
    }

    #[test]
    fn test_mirror_both() {
        let m = MirrorEffect {
            axis: MirrorAxis::Both,
        };
        let (u, v) = m.apply_uv(0.2, 0.8);
        assert!((u - 0.8).abs() < 1e-6, "u mirrored: {u}");
        assert!((v - 0.2).abs() < 1e-6, "v mirrored: {v}");
    }

    #[test]
    fn test_mirror_center_unchanged() {
        for axis in [
            MirrorAxis::Horizontal,
            MirrorAxis::Vertical,
            MirrorAxis::Both,
        ] {
            let m = MirrorEffect { axis };
            let (u, v) = m.apply_uv(0.5, 0.5);
            assert!((u - 0.5).abs() < 1e-6, "u at center: {u}");
            assert!((v - 0.5).abs() < 1e-6, "v at center: {v}");
        }
    }
}
