//! Automated video transition effects between clips.
//!
//! Provides frame-blending, wipes, dissolves, dips, and other transition
//! effects for professional broadcast playout. All operations are performed
//! on RGBA 8-bit frame data (4 bytes per pixel, row-major order).

use crate::{PlayoutError, Result};

/// Direction for wipe transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeDirection {
    /// Wipe from left edge to right
    Left,
    /// Wipe from right edge to left
    Right,
    /// Wipe from top edge to bottom
    Up,
    /// Wipe from bottom edge to top
    Down,
}

/// The kind of transition to apply
#[derive(Debug, Clone)]
pub enum TransitionType {
    /// Hard cut — instantly switch to frame_b (zero frames of overlap)
    Cut,
    /// Linear crossfade between frame_a and frame_b
    Dissolve,
    /// Fade to black, then fade in from black (two-phase)
    Fade,
    /// Directional wipe effect
    Wipe(WipeDirection),
    /// Dip to a solid colour (R, G, B), then fade in from that colour
    Dip(u8, u8, u8),
}

/// A transition specification: kind + total length in frames
#[derive(Debug, Clone)]
pub struct Transition {
    pub kind: TransitionType,
    /// Total number of frames this transition spans (must be ≥ 1)
    pub duration_frames: u32,
}

impl Transition {
    /// Create a hard-cut (zero-duration) transition.
    pub fn cut() -> Self {
        Self {
            kind: TransitionType::Cut,
            duration_frames: 1,
        }
    }

    /// Create a dissolve with the given frame count.
    pub fn dissolve(duration_frames: u32) -> Self {
        Self {
            kind: TransitionType::Dissolve,
            duration_frames: duration_frames.max(1),
        }
    }

    /// Create a fade-to/from-black with the given frame count.
    pub fn fade(duration_frames: u32) -> Self {
        Self {
            kind: TransitionType::Fade,
            duration_frames: duration_frames.max(1),
        }
    }

    /// Create a wipe in the specified direction.
    pub fn wipe(direction: WipeDirection, duration_frames: u32) -> Self {
        Self {
            kind: TransitionType::Wipe(direction),
            duration_frames: duration_frames.max(1),
        }
    }

    /// Create a dip-to-colour transition.
    pub fn dip(r: u8, g: u8, b: u8, duration_frames: u32) -> Self {
        Self {
            kind: TransitionType::Dip(r, g, b),
            duration_frames: duration_frames.max(1),
        }
    }
}

/// Tracks progress through an in-flight transition.
#[derive(Debug, Clone)]
pub struct TransitionEffect {
    pub transition: Transition,
    /// Current frame index within [0, duration_frames)
    pub current_frame: u32,
}

impl TransitionEffect {
    /// Create a new effect at the very first frame.
    pub fn new(transition: Transition) -> Self {
        Self {
            transition,
            current_frame: 0,
        }
    }

    /// Returns `true` if the transition has not yet finished.
    pub fn is_active(&self) -> bool {
        self.current_frame < self.transition.duration_frames
    }

    /// Advance by one frame and return the blended output.
    ///
    /// `frame_a` and `frame_b` must both be RGBA buffers of `width * height * 4` bytes.
    pub fn advance(
        &mut self,
        frame_a: &[u8],
        frame_b: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        let alpha = if self.transition.duration_frames <= 1 {
            1.0_f32
        } else {
            self.current_frame as f32 / (self.transition.duration_frames - 1) as f32
        };

        let output = apply_transition(&self.transition, frame_a, frame_b, width, height, alpha)?;

        if self.current_frame < self.transition.duration_frames {
            self.current_frame += 1;
        }

        Ok(output)
    }
}

/// Apply a transition between two RGBA frames.
///
/// # Arguments
/// * `transition` – The transition specification.
/// * `frame_a`    – Source frame (outgoing clip), RGBA 8-bit.
/// * `frame_b`    – Destination frame (incoming clip), RGBA 8-bit.
/// * `width`      – Frame width in pixels.
/// * `height`     – Frame height in pixels.
/// * `alpha`      – Normalised progress in [0.0, 1.0].
///
/// Returns the blended frame as an RGBA `Vec<u8>`.
pub fn apply_transition(
    transition: &Transition,
    frame_a: &[u8],
    frame_b: &[u8],
    width: u32,
    height: u32,
    alpha: f32,
) -> Result<Vec<u8>> {
    let expected = (width * height * 4) as usize;

    if frame_a.len() != expected {
        return Err(PlayoutError::Playback(format!(
            "frame_a size mismatch: expected {expected}, got {}",
            frame_a.len()
        )));
    }
    if frame_b.len() != expected {
        return Err(PlayoutError::Playback(format!(
            "frame_b size mismatch: expected {expected}, got {}",
            frame_b.len()
        )));
    }

    let alpha = alpha.clamp(0.0, 1.0);

    match &transition.kind {
        TransitionType::Cut => apply_cut(frame_b),
        TransitionType::Dissolve => apply_dissolve(frame_a, frame_b, alpha),
        TransitionType::Fade => apply_fade(frame_a, frame_b, alpha),
        TransitionType::Wipe(direction) => {
            apply_wipe(frame_a, frame_b, width, height, alpha, *direction)
        }
        TransitionType::Dip(r, g, b) => apply_dip(frame_a, frame_b, alpha, *r, *g, *b),
    }
}

// ─── private helpers ─────────────────────────────────────────────────────────

/// Hard cut: return frame_b unchanged.
fn apply_cut(frame_b: &[u8]) -> Result<Vec<u8>> {
    Ok(frame_b.to_vec())
}

/// Linear dissolve: `(1 - alpha) * a + alpha * b` per channel.
fn apply_dissolve(frame_a: &[u8], frame_b: &[u8], alpha: f32) -> Result<Vec<u8>> {
    let inv = 1.0 - alpha;
    let output = frame_a
        .iter()
        .zip(frame_b.iter())
        .map(|(&a, &b)| blend_channel(a, b, inv, alpha))
        .collect();
    Ok(output)
}

/// Two-phase fade: first half fades A to black; second half fades B from black.
fn apply_fade(frame_a: &[u8], frame_b: &[u8], alpha: f32) -> Result<Vec<u8>> {
    if alpha < 0.5 {
        // Phase 1: fade A to black; alpha=0 → full A, alpha=0.5 → black
        let fade_out = 1.0 - alpha * 2.0; // 1.0 → 0.0
        let output = frame_a
            .chunks_exact(4)
            .flat_map(|px| {
                let r = (px[0] as f32 * fade_out) as u8;
                let g = (px[1] as f32 * fade_out) as u8;
                let b2 = (px[2] as f32 * fade_out) as u8;
                let a = px[3];
                [r, g, b2, a]
            })
            .collect();
        Ok(output)
    } else {
        // Phase 2: fade B from black; alpha=0.5 → black, alpha=1.0 → full B
        let fade_in = (alpha - 0.5) * 2.0; // 0.0 → 1.0
        let output = frame_b
            .chunks_exact(4)
            .flat_map(|px| {
                let r = (px[0] as f32 * fade_in) as u8;
                let g = (px[1] as f32 * fade_in) as u8;
                let b2 = (px[2] as f32 * fade_in) as u8;
                let a = px[3];
                [r, g, b2, a]
            })
            .collect();
        Ok(output)
    }
}

/// Directional wipe: one region shows frame_b, the other frame_a.
fn apply_wipe(
    frame_a: &[u8],
    frame_b: &[u8],
    width: u32,
    height: u32,
    alpha: f32,
    direction: WipeDirection,
) -> Result<Vec<u8>> {
    let mut output = vec![0u8; (width * height * 4) as usize];

    for row in 0..height {
        for col in 0..width {
            let pixel_idx = ((row * width + col) * 4) as usize;

            let use_b = match direction {
                // Left: wipe reveals B from the left side
                WipeDirection::Left => col < (alpha * width as f32) as u32,
                // Right: wipe reveals B from the right side
                WipeDirection::Right => col >= width - (alpha * width as f32) as u32,
                // Up: wipe reveals B from the top
                WipeDirection::Up => row < (alpha * height as f32) as u32,
                // Down: wipe reveals B from the bottom
                WipeDirection::Down => row >= height - (alpha * height as f32) as u32,
            };

            let src = if use_b {
                &frame_b[pixel_idx..pixel_idx + 4]
            } else {
                &frame_a[pixel_idx..pixel_idx + 4]
            };

            output[pixel_idx..pixel_idx + 4].copy_from_slice(src);
        }
    }

    Ok(output)
}

/// Dip-to-colour: first half dips A to `(r,g,b)`, second half fades B in from `(r,g,b)`.
fn apply_dip(
    frame_a: &[u8],
    frame_b: &[u8],
    alpha: f32,
    dip_r: u8,
    dip_g: u8,
    dip_b: u8,
) -> Result<Vec<u8>> {
    if alpha < 0.5 {
        // Phase 1: blend A toward the dip colour
        let t = alpha * 2.0; // 0 → 1 over first half
        let inv = 1.0 - t;
        let output = frame_a
            .chunks_exact(4)
            .flat_map(|px| {
                let r = blend_channel(px[0], dip_r, inv, t);
                let g = blend_channel(px[1], dip_g, inv, t);
                let b2 = blend_channel(px[2], dip_b, inv, t);
                let a = px[3];
                [r, g, b2, a]
            })
            .collect();
        Ok(output)
    } else {
        // Phase 2: blend B out of the dip colour
        let t = (alpha - 0.5) * 2.0; // 0 → 1 over second half
        let inv = 1.0 - t;
        let output = frame_b
            .chunks_exact(4)
            .flat_map(|px| {
                let r = blend_channel(dip_r, px[0], inv, t);
                let g = blend_channel(dip_g, px[1], inv, t);
                let b2 = blend_channel(dip_b, px[2], inv, t);
                let a = px[3];
                [r, g, b2, a]
            })
            .collect();
        Ok(output)
    }
}

/// Linear blend of two u8 channel values.
#[inline]
fn blend_channel(a: u8, b: u8, w_a: f32, w_b: f32) -> u8 {
    (a as f32 * w_a + b as f32 * w_b).round().clamp(0.0, 255.0) as u8
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn black_frame(w: u32, h: u32) -> Vec<u8> {
        vec![0u8; (w * h * 4) as usize]
    }

    fn white_frame(w: u32, h: u32) -> Vec<u8> {
        let n = (w * h * 4) as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..w * h {
            v.extend_from_slice(&[255, 255, 255, 255]);
        }
        v
    }

    fn solid_frame(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&[r, g, b, 255]);
        }
        v
    }

    // 1. Cut always returns frame_b verbatim
    #[test]
    fn test_cut_returns_frame_b() {
        let a = black_frame(2, 2);
        let b = white_frame(2, 2);
        let t = Transition::cut();
        let out = apply_transition(&t, &a, &b, 2, 2, 0.0).expect("transition should succeed");
        assert_eq!(out, b);
    }

    // 2. Cut at any alpha still returns frame_b
    #[test]
    fn test_cut_alpha_irrelevant() {
        let a = black_frame(2, 2);
        let b = white_frame(2, 2);
        let t = Transition::cut();
        for alpha in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let out = apply_transition(&t, &a, &b, 2, 2, alpha).expect("transition should succeed");
            assert_eq!(out, b, "alpha={alpha}");
        }
    }

    // 3. Dissolve at alpha=0 gives frame_a
    #[test]
    fn test_dissolve_alpha_zero_is_frame_a() {
        let a = black_frame(2, 2);
        let b = white_frame(2, 2);
        let t = Transition::dissolve(25);
        let out = apply_transition(&t, &a, &b, 2, 2, 0.0).expect("transition should succeed");
        // All channels should be ~0 (black)
        assert!(out.iter().all(|&v| v == 0));
    }

    // 4. Dissolve at alpha=1 gives frame_b
    #[test]
    fn test_dissolve_alpha_one_is_frame_b() {
        let a = black_frame(2, 2);
        let b = white_frame(2, 2);
        let t = Transition::dissolve(25);
        let out = apply_transition(&t, &a, &b, 2, 2, 1.0).expect("transition should succeed");
        assert!(out.iter().all(|&v| v == 255));
    }

    // 5. Dissolve midpoint approximates grey
    #[test]
    fn test_dissolve_midpoint() {
        let a = black_frame(1, 1);
        let b = white_frame(1, 1);
        let t = Transition::dissolve(3);
        let out = apply_transition(&t, &a, &b, 1, 1, 0.5).expect("transition should succeed");
        // Channel 0–2 should be ~128 (rounding)
        assert!((out[0] as i32 - 128).abs() <= 1);
    }

    // 6. Fade at alpha=0 is full frame_a (not black)
    #[test]
    fn test_fade_alpha_zero_is_frame_a() {
        let a = white_frame(1, 1);
        let b = black_frame(1, 1);
        let t = Transition::fade(50);
        let out = apply_transition(&t, &a, &b, 1, 1, 0.0).expect("transition should succeed");
        assert_eq!(out[0], 255);
    }

    // 7. Fade at alpha=0.5 is pure black (midpoint dip)
    #[test]
    fn test_fade_midpoint_is_black() {
        let a = white_frame(1, 1);
        let b = white_frame(1, 1);
        let t = Transition::fade(50);
        let out = apply_transition(&t, &a, &b, 1, 1, 0.5).expect("transition should succeed");
        assert_eq!(out[0], 0, "R should be 0 at midpoint");
        assert_eq!(out[1], 0, "G should be 0 at midpoint");
        assert_eq!(out[2], 0, "B should be 0 at midpoint");
    }

    // 8. Wipe Left at alpha=0 → full frame_a
    #[test]
    fn test_wipe_left_alpha_zero_is_frame_a() {
        let a = solid_frame(4, 1, 100, 0, 0);
        let b = solid_frame(4, 1, 0, 200, 0);
        let t = Transition::wipe(WipeDirection::Left, 25);
        let out = apply_transition(&t, &a, &b, 4, 1, 0.0).expect("transition should succeed");
        // All pixels should be from frame_a
        for i in 0..4 {
            assert_eq!(out[i * 4], 100, "pixel {i} R");
        }
    }

    // 9. Wipe Left at alpha=1 → full frame_b
    #[test]
    fn test_wipe_left_alpha_one_is_frame_b() {
        let a = solid_frame(4, 1, 100, 0, 0);
        let b = solid_frame(4, 1, 0, 200, 0);
        let t = Transition::wipe(WipeDirection::Left, 25);
        let out = apply_transition(&t, &a, &b, 4, 1, 1.0).expect("transition should succeed");
        for i in 0..4 {
            assert_eq!(out[i * 4 + 1], 200, "pixel {i} G");
        }
    }

    // 10. Dip at alpha=0 is full frame_a
    #[test]
    fn test_dip_alpha_zero_is_frame_a() {
        let a = solid_frame(1, 1, 50, 100, 150);
        let b = black_frame(1, 1);
        let t = Transition::dip(255, 0, 0, 25);
        let out = apply_transition(&t, &a, &b, 1, 1, 0.0).expect("transition should succeed");
        assert_eq!(out[0], 50);
        assert_eq!(out[1], 100);
        assert_eq!(out[2], 150);
    }

    // 11. Dip at alpha=0.5 should be near the dip colour
    #[test]
    fn test_dip_midpoint_near_dip_colour() {
        let a = black_frame(1, 1);
        let b = black_frame(1, 1);
        let t = Transition::dip(0, 255, 0, 25); // dip to green
        let out = apply_transition(&t, &a, &b, 1, 1, 0.5).expect("transition should succeed");
        // At exactly 0.5, phase 1 ends with t=1 → pure dip colour
        assert_eq!(out[1], 255, "G should be 255 at dip midpoint");
    }

    // 12. Size mismatch returns Err
    #[test]
    fn test_size_mismatch_error() {
        let a = vec![0u8; 16]; // 2×2
        let b = vec![0u8; 32]; // 2×4 — wrong
        let t = Transition::dissolve(10);
        let result = apply_transition(&t, &a, &b, 2, 2, 0.5);
        assert!(result.is_err());
    }

    // 13. TransitionEffect advances frame counter
    #[test]
    fn test_transition_effect_advance() {
        let t = Transition::dissolve(5);
        let mut fx = TransitionEffect::new(t);
        assert!(fx.is_active());
        let a = black_frame(2, 2);
        let b = white_frame(2, 2);
        for _ in 0..5 {
            assert!(fx.is_active());
            fx.advance(&a, &b, 2, 2).expect("advance should succeed");
        }
        assert!(!fx.is_active());
    }

    // 14. Wipe Right half-way → right half is frame_b
    #[test]
    fn test_wipe_right_half() {
        let a = solid_frame(4, 1, 10, 0, 0);
        let b = solid_frame(4, 1, 0, 20, 0);
        let t = Transition::wipe(WipeDirection::Right, 25);
        let out = apply_transition(&t, &a, &b, 4, 1, 0.5).expect("transition should succeed");
        // Rightmost 2 pixels (col 2,3) should be frame_b (G=20)
        assert_eq!(out[2 * 4 + 1], 20, "col2 G");
        assert_eq!(out[3 * 4 + 1], 20, "col3 G");
        // Leftmost 2 pixels should be frame_a (R=10)
        assert_eq!(out[0 * 4], 10, "col0 R");
        assert_eq!(out[1 * 4], 10, "col1 R");
    }

    // 15. Wipe Up half-way → top rows are frame_b
    #[test]
    fn test_wipe_up_half() {
        let a = solid_frame(2, 4, 10, 0, 0);
        let b = solid_frame(2, 4, 0, 20, 0);
        let t = Transition::wipe(WipeDirection::Up, 25);
        let out = apply_transition(&t, &a, &b, 2, 4, 0.5).expect("transition should succeed");
        // Top 2 rows (row 0,1) → frame_b
        for row in 0_u32..2 {
            for col in 0_u32..2 {
                let idx = ((row * 2 + col) * 4) as usize;
                assert_eq!(out[idx + 1], 20, "row{row} col{col} G");
            }
        }
    }
}
