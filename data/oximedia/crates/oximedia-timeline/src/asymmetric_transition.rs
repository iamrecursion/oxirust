//! Asymmetric transitions: independent easing curves for outgoing and incoming clips.
//!
//! A standard cross-dissolve applies the same linear or eased curve to both the
//! outgoing (fading-out) and incoming (fading-in) clip.  Asymmetric transitions
//! give each clip its own independent [`EasingCurve`], enabling fine-grained
//! control over perceived motion, impact, and rhythm at cut points.
//!
//! # Example
//!
//! ```
//! use oximedia_timeline::asymmetric_transition::{AsymmetricTransition, EasingCurve};
//!
//! // Outgoing clip eases in (starts slow), incoming eases out (starts fast).
//! let tr = AsymmetricTransition::new(24, EasingCurve::EaseIn, EasingCurve::EaseOut);
//! let (out_a, in_a) = tr.alphas_at_frame(12); // midpoint
//! assert!(out_a != in_a, "curves differ at midpoint");
//! ```

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// EasingCurve
// ─────────────────────────────────────────────────────────────────────────────

/// Easing curve applied to a single side of an asymmetric transition.
///
/// All curves map the unit interval [0, 1] → [0, 1] monotonically, with
/// `apply(0.0) == 0.0` and `apply(1.0) == 1.0` guaranteed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EasingCurve {
    /// Constant-rate blend (α = t).
    Linear,
    /// Cubic ease-in: starts slow, ends fast (α = t³).
    EaseIn,
    /// Cubic ease-out: starts fast, ends slow (α = 1 − (1 − t)³).
    EaseOut,
    /// Cubic S-curve: slow-fast-slow (α = 3t² − 2t³).
    EaseInOut,
    /// Instant cut: α = 0 for t < 0.5, α = 1 for t ≥ 0.5.
    Step,
}

impl EasingCurve {
    /// Map normalised progress `t` ∈ [0, 1] to an eased value ∈ [0, 1].
    ///
    /// Values outside [0, 1] are clamped before application.
    #[must_use]
    pub fn apply(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            EasingCurve::Linear => t,
            EasingCurve::EaseIn => t * t * t,
            EasingCurve::EaseOut => {
                let inv = 1.0 - t;
                1.0 - inv * inv * inv
            }
            EasingCurve::EaseInOut => t * t * (3.0 - 2.0 * t),
            EasingCurve::Step => {
                if t < 0.5 {
                    0.0
                } else {
                    1.0
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AsymmetricTransition
// ─────────────────────────────────────────────────────────────────────────────

/// A transition between two clips where the outgoing and incoming sides each
/// follow an independent [`EasingCurve`].
///
/// The two alphas returned by [`alphas_at_frame`](Self::alphas_at_frame) are
/// *independent*: neither is constrained to sum to 1.0.  At frame 0 the
/// outgoing alpha is 1.0 and incoming is 0.0; at the last frame the outgoing
/// alpha is 0.0 and incoming is 1.0.  Between those endpoints each curve
/// evolves on its own trajectory, allowing subtle cross-curves, brief
/// simultaneous over-exposure, or contrast-heavy cuts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymmetricTransition {
    /// Total duration in frames.  Must be ≥ 1.
    pub duration_frames: u32,
    /// Easing curve applied to the outgoing (fading-out) clip.
    pub outgoing_curve: EasingCurve,
    /// Easing curve applied to the incoming (fading-in) clip.
    pub incoming_curve: EasingCurve,
}

impl AsymmetricTransition {
    /// Create a new asymmetric transition.
    ///
    /// `duration_frames` is clamped to a minimum of 1 to avoid division by
    /// zero inside [`alphas_at_frame`](Self::alphas_at_frame).
    #[must_use]
    pub fn new(duration_frames: u32, outgoing: EasingCurve, incoming: EasingCurve) -> Self {
        Self {
            duration_frames: duration_frames.max(1),
            outgoing_curve: outgoing,
            incoming_curve: incoming,
        }
    }

    /// Compute `(outgoing_alpha, incoming_alpha)` at a frame offset within
    /// `[0, duration_frames)`.
    ///
    /// `frame` is clamped to `[0, duration_frames - 1]` before use.
    ///
    /// | frame        | outgoing_alpha | incoming_alpha |
    /// |:-------------|:---------------|:---------------|
    /// | 0            | 1.0            | 0.0            |
    /// | midpoint     | curve-specific | curve-specific |
    /// | duration - 1 | 0.0            | 1.0            |
    ///
    /// The normalised progress `t` is computed as:
    /// `t = frame / (duration_frames - 1)` when `duration_frames > 1`, and
    /// `t = 0.5` when `duration_frames == 1` (a single-frame transition sits
    /// exactly at its midpoint).
    ///
    /// * outgoing alpha = `1.0 − outgoing_curve.apply(t)` — starts at 1, ends at 0.
    /// * incoming alpha = `incoming_curve.apply(t)` — starts at 0, ends at 1.
    #[must_use]
    pub fn alphas_at_frame(&self, frame: u32) -> (f32, f32) {
        let frame = frame.min(self.duration_frames.saturating_sub(1));
        let t = if self.duration_frames <= 1 {
            0.5f32
        } else {
            frame as f32 / (self.duration_frames - 1) as f32
        };
        let outgoing_alpha = 1.0 - self.outgoing_curve.apply(t);
        let incoming_alpha = self.incoming_curve.apply(t);
        (outgoing_alpha, incoming_alpha)
    }

    /// Composite two equal-length RGBA (or raw byte) frame buffers using the
    /// per-frame alphas and return a new blended buffer.
    ///
    /// Each output byte is:
    ///
    /// ```text
    /// out[i] = clamp(out_alpha * outgoing[i] + in_alpha * incoming[i], 0, 255)
    /// ```
    ///
    /// The two buffers must have the same length; if they differ the shorter
    /// one is padded with zeros for the tail.
    ///
    /// # Panics
    ///
    /// Does not panic.  Mismatched lengths produce a gracefully truncated
    /// result (the `zip` iterator stops at the shorter slice).
    #[must_use]
    pub fn composite(&self, outgoing: &[u8], incoming: &[u8], frame: u32) -> Vec<u8> {
        let (out_alpha, in_alpha) = self.alphas_at_frame(frame);
        let len = outgoing.len().max(incoming.len());
        let mut result = Vec::with_capacity(len);

        let out_iter = outgoing.iter().copied().chain(std::iter::repeat(0u8));
        let in_iter = incoming.iter().copied().chain(std::iter::repeat(0u8));

        for (o, i) in out_iter.zip(in_iter).take(len) {
            let blended = (f32::from(o) * out_alpha + f32::from(i) * in_alpha).round() as u16;
            result.push(blended.min(255) as u8);
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration with TransitionType enum extension
// ─────────────────────────────────────────────────────────────────────────────

/// Extension of [`crate::transition::TransitionType`] to carry asymmetric
/// easing data in the type system without modifying the original enum.
///
/// Callers that already match on `TransitionType` variants continue to compile
/// unchanged.  Code that wants asymmetric behaviour creates
/// [`AsymmetricTransitionWrapper`] independently and applies it via
/// [`AsymmetricTransition`] directly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymmetricTransitionWrapper {
    /// The asymmetric easing definition.
    pub params: AsymmetricTransition,
}

impl AsymmetricTransitionWrapper {
    /// Create a wrapper from an [`AsymmetricTransition`].
    #[must_use]
    pub fn new(params: AsymmetricTransition) -> Self {
        Self { params }
    }

    /// Delegate to [`AsymmetricTransition::alphas_at_frame`].
    #[must_use]
    pub fn alphas_at_frame(&self, frame: u32) -> (f32, f32) {
        self.params.alphas_at_frame(frame)
    }

    /// Delegate to [`AsymmetricTransition::composite`].
    #[must_use]
    pub fn composite(&self, outgoing: &[u8], incoming: &[u8], frame: u32) -> Vec<u8> {
        self.params.composite(outgoing, incoming, frame)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EasingCurve boundary values ───────────────────────────────────────────

    #[test]
    fn test_easing_curve_boundary_values() {
        let curves = [
            EasingCurve::Linear,
            EasingCurve::EaseIn,
            EasingCurve::EaseOut,
            EasingCurve::EaseInOut,
            EasingCurve::Step,
        ];
        for curve in curves {
            let v0 = curve.apply(0.0);
            let v1 = curve.apply(1.0);
            assert!(v0.abs() < 1e-6, "{curve:?}.apply(0.0) = {v0}, expected 0.0");
            assert!(
                (v1 - 1.0).abs() < 1e-6,
                "{curve:?}.apply(1.0) = {v1}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_easing_curve_clamping() {
        // Values outside [0,1] should clamp gracefully.
        assert!((EasingCurve::Linear.apply(-0.5) - 0.0).abs() < 1e-6);
        assert!((EasingCurve::Linear.apply(1.5) - 1.0).abs() < 1e-6);
        assert!((EasingCurve::EaseIn.apply(-1.0) - 0.0).abs() < 1e-6);
        assert!((EasingCurve::EaseOut.apply(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ease_in_midpoint_less_than_half() {
        // EaseIn is still accelerating at t=0.5: f(0.5)=0.125 < 0.5
        let v = EasingCurve::EaseIn.apply(0.5);
        assert!(v < 0.5, "EaseIn at 0.5 should be < 0.5, got {v}");
    }

    #[test]
    fn test_ease_out_midpoint_greater_than_half() {
        // EaseOut is decelerating at t=0.5: f(0.5)=0.875 > 0.5
        let v = EasingCurve::EaseOut.apply(0.5);
        assert!(v > 0.5, "EaseOut at 0.5 should be > 0.5, got {v}");
    }

    #[test]
    fn test_ease_in_out_symmetric() {
        // EaseInOut should be symmetric around t=0.5.
        let v_lo = EasingCurve::EaseInOut.apply(0.25);
        let v_hi = EasingCurve::EaseInOut.apply(0.75);
        assert!(
            (v_lo - (1.0 - v_hi)).abs() < 1e-6,
            "EaseInOut not symmetric: f(0.25)={v_lo}, 1-f(0.75)={}",
            1.0 - v_hi
        );
    }

    #[test]
    fn test_step_below_half() {
        assert!((EasingCurve::Step.apply(0.0) - 0.0).abs() < 1e-6);
        assert!((EasingCurve::Step.apply(0.49) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_step_at_and_above_half() {
        assert!((EasingCurve::Step.apply(0.5) - 1.0).abs() < 1e-6);
        assert!((EasingCurve::Step.apply(1.0) - 1.0).abs() < 1e-6);
    }

    // ── AsymmetricTransition::alphas_at_frame ─────────────────────────────────

    #[test]
    fn test_asymmetric_linear_linear() {
        let tr = AsymmetricTransition::new(11, EasingCurve::Linear, EasingCurve::Linear);

        // Frame 0 → outgoing=1.0, incoming=0.0
        let (out0, in0) = tr.alphas_at_frame(0);
        assert!(
            (out0 - 1.0).abs() < 1e-6,
            "Frame 0 outgoing should be 1.0, got {out0}"
        );
        assert!(
            in0.abs() < 1e-6,
            "Frame 0 incoming should be 0.0, got {in0}"
        );

        // Frame 5 = midpoint (t=0.5) → both 0.5
        let (out5, in5) = tr.alphas_at_frame(5);
        assert!(
            (out5 - 0.5).abs() < 1e-5,
            "Midpoint outgoing should be 0.5, got {out5}"
        );
        assert!(
            (in5 - 0.5).abs() < 1e-5,
            "Midpoint incoming should be 0.5, got {in5}"
        );

        // Frame 10 = last → outgoing=0.0, incoming=1.0
        let (out10, in10) = tr.alphas_at_frame(10);
        assert!(
            out10.abs() < 1e-6,
            "Last frame outgoing should be 0.0, got {out10}"
        );
        assert!(
            (in10 - 1.0).abs() < 1e-6,
            "Last frame incoming should be 1.0, got {in10}"
        );
    }

    #[test]
    fn test_asymmetric_ease_in_ease_out() {
        // outgoing=EaseIn (slow start: fades slowly at first → alpha stays HIGH)
        // incoming=EaseOut (fast start: ramps quickly → alpha goes HIGH fast)
        // At midpoint: outgoing_alpha = 1 - EaseIn(0.5) = 1 - 0.125 = 0.875
        //              incoming_alpha = EaseOut(0.5) = 0.875
        // Both are 0.875 here, because EaseIn(0.5)=0.125 and EaseOut(0.5)=0.875
        // and outgoing = 1 - EaseIn(t), incoming = EaseOut(t).
        // 1 - 0.125 = 0.875 == 0.875 → they're actually equal at midpoint!
        //
        // The task spec says: "outgoing_alpha > incoming_alpha (EaseIn is still fast at midpoint)"
        // Let's verify the property that EaseIn lags (outgoing stays higher) compared to linear.
        // With EaseIn outgoing: alpha decays slowly (stays high for longer).
        // With EaseOut incoming: alpha rises quickly (gets high early).
        //
        // So at t=0.3: outgoing = 1 - EaseIn(0.3) = 1 - 0.027 = 0.973
        //              incoming = EaseOut(0.3) = 1 - (0.7)^3 = 1 - 0.343 = 0.657
        // → outgoing > incoming at early frames.
        //
        // The spec says "at midpoint, outgoing_alpha > incoming_alpha (EaseIn is still fast at midpoint)"
        // — this is actually testing a *specific* combination. Let's test the right semantics:
        // EaseIn outgoing means: outgoing alpha = 1 - t^3 (decays slowly, stays high)
        // EaseOut incoming means: incoming alpha = 1 - (1-t)^3 (rises quickly)
        // At t=0.5: outgoing = 1 - 0.125 = 0.875, incoming = 0.875 — equal.
        //
        // Re-reading the spec: "outgoing=EaseIn → at midpoint, outgoing_alpha > incoming_alpha"
        // Let's verify the actual cubic values more carefully:
        // At t=0.4: EaseIn(0.4) = 0.064, so outgoing = 0.936
        //           EaseOut(0.4) = 1 - (0.6)^3 = 1 - 0.216 = 0.784
        // So at t < 0.5, outgoing > incoming, proving EaseIn keeps outgoing high.
        let tr = AsymmetricTransition::new(11, EasingCurve::EaseIn, EasingCurve::EaseOut);
        // Test at frame 4 (t≈0.4): outgoing should dominate
        let (out4, in4) = tr.alphas_at_frame(4);
        assert!(
            out4 > in4,
            "At t≈0.4: outgoing_alpha={out4} should be > incoming_alpha={in4} \
             with EaseIn/EaseOut"
        );
    }

    #[test]
    fn test_asymmetric_clamped_frame() {
        let tr = AsymmetricTransition::new(5, EasingCurve::Linear, EasingCurve::Linear);
        // Frames beyond duration should clamp to last frame.
        let (out_last, in_last) = tr.alphas_at_frame(4);
        let (out_clamped, in_clamped) = tr.alphas_at_frame(100);
        assert!(
            (out_last - out_clamped).abs() < 1e-6,
            "Clamped frame outgoing should equal last frame"
        );
        assert!(
            (in_last - in_clamped).abs() < 1e-6,
            "Clamped frame incoming should equal last frame"
        );
    }

    #[test]
    fn test_asymmetric_single_frame_transition() {
        // duration=1 → single frame sits at t=0.5
        let tr = AsymmetricTransition::new(1, EasingCurve::Linear, EasingCurve::Linear);
        let (out0, in0) = tr.alphas_at_frame(0);
        assert!(
            (out0 - 0.5).abs() < 1e-6,
            "Single-frame outgoing at t=0.5 should be 0.5, got {out0}"
        );
        assert!(
            (in0 - 0.5).abs() < 1e-6,
            "Single-frame incoming at t=0.5 should be 0.5, got {in0}"
        );
    }

    #[test]
    fn test_asymmetric_duration_min_clamped() {
        // duration=0 should be clamped to 1
        let tr = AsymmetricTransition::new(0, EasingCurve::Linear, EasingCurve::Linear);
        assert_eq!(tr.duration_frames, 1);
    }

    // ── AsymmetricTransition::composite ──────────────────────────────────────

    #[test]
    fn test_asymmetric_composite_boundary() {
        let tr = AsymmetricTransition::new(25, EasingCurve::Linear, EasingCurve::Linear);

        let outgoing: Vec<u8> = (0..12).map(|i| (i * 20 + 10) as u8).collect();
        let incoming: Vec<u8> = (0..12).map(|i| (255 - i * 20) as u8).collect();

        // Frame 0: outgoing_alpha=1.0, incoming_alpha=0.0 → output ≈ outgoing
        let result_first = tr.composite(&outgoing, &incoming, 0);
        for (idx, (&o, &r)) in outgoing.iter().zip(result_first.iter()).enumerate() {
            assert_eq!(o, r, "Frame 0 byte {idx}: expected {o}, got {r}");
        }

        // Frame 24 (last): outgoing_alpha=0.0, incoming_alpha=1.0 → output ≈ incoming
        let result_last = tr.composite(&outgoing, &incoming, 24);
        for (idx, (&i, &r)) in incoming.iter().zip(result_last.iter()).enumerate() {
            assert_eq!(i, r, "Frame 24 byte {idx}: expected incoming {i}, got {r}");
        }
    }

    #[test]
    fn test_asymmetric_composite_midpoint_blend() {
        // At midpoint with linear curves, both alphas are 0.5.
        // Output byte should be approximately (o + i) / 2 (with rounding).
        let tr = AsymmetricTransition::new(3, EasingCurve::Linear, EasingCurve::Linear);

        let outgoing = vec![200u8, 0u8];
        let incoming = vec![0u8, 200u8];

        // Frame 1 of 3 → t = 0.5
        let result = tr.composite(&outgoing, &incoming, 1);
        assert_eq!(result.len(), 2);
        // 200*0.5 + 0*0.5 = 100
        assert_eq!(result[0], 100, "Midpoint blend byte 0");
        // 0*0.5 + 200*0.5 = 100
        assert_eq!(result[1], 100, "Midpoint blend byte 1");
    }

    #[test]
    fn test_asymmetric_composite_overflow_clamp() {
        // With step curves and t=0.5: both Step.apply(0.5)=1.0.
        // outgoing_alpha = 1 - 1.0 = 0.0, incoming_alpha = 1.0.
        // To force overflow, we use a custom scenario:
        // Both alphas can simultaneously be > 0 when using curves that don't sum to 1.
        // The composite formula is:  out_alpha * o + in_alpha * i
        // Overflow is triggered when out_alpha + in_alpha > 1.
        //
        // EaseIn at t=0.9: apply(0.9) = 0.729 → outgoing_alpha = 0.271, incoming_alpha = 0.729
        // Sum = 1.0 exactly → no overflow there.
        //
        // EaseOut outgoing at t=0.1: apply(0.1)=0.271 → outgoing_alpha = 0.729
        // EaseOut incoming at t=0.1: apply(0.1)=0.271 → incoming_alpha = 0.271
        // Sum = 1.0 again.
        //
        // The sum always equals 1.0 for complementary curves; overflow can only
        // happen with asymmetric Step curves at the transition point.
        // Step: apply(0.5)=1.0, so outgoing_alpha = 1-1.0=0.0, incoming_alpha=1.0.
        // That doesn't overflow.  But Step(t<0.5)=0.0: outgoing_alpha=1.0, incoming_alpha=0.0.
        //
        // True overflow requires two independent curves where sum > 1:
        // E.g. two EaseOut curves: outgoing_alpha = 1 - EaseOut(t),
        //                          incoming_alpha = EaseOut(t).
        // Again sum = 1.  In practice overflow only happens when the user sets
        // both curves to EaseOut-like shapes that happen to overlap > 1.0.
        //
        // Actually the design is: outgoing = 1 - outgoing_curve(t),
        //                         incoming = incoming_curve(t).
        // These always sum to 1 when both curves are the same.  Sum > 1 only if
        // outgoing_curve(t) + incoming_curve(t) < 1, which can't overflow.
        // Sum < 1 only if the two curves make the sum > 1... let's think:
        // sum = (1 - f_out(t)) + f_in(t).  For sum > 1: f_in(t) > f_out(t).
        // At t=0.1 with EaseOut incoming and EaseIn outgoing:
        //   f_out = EaseIn(0.1) = 0.001, f_in = EaseOut(0.1) = 0.271
        //   sum = 0.999 + 0.271 = 1.270 > 1 → overflow possible!
        //   blend = 0.999*200 + 0.271*200 = 199.8 + 54.2 = 254 → under 255.
        //   With two pixels at 200: same calc.
        //   For value 255: 0.999*255 + 0.271*255 = 254.745 + 69.105 = 323.85 → clamped to 255.
        let tr = AsymmetricTransition::new(11, EasingCurve::EaseIn, EasingCurve::EaseOut);
        let outgoing = vec![255u8];
        let incoming = vec![255u8];
        // Frame 1 (t=0.1): EaseIn(0.1)=0.001, EaseOut(0.1)=0.271
        // outgoing_alpha ≈ 0.999, incoming_alpha ≈ 0.271
        // blend = 0.999*255 + 0.271*255 ≈ 323.85 → clamped to 255
        let result = tr.composite(&outgoing, &incoming, 1);
        assert_eq!(result[0], 255, "Overflow of 323 should clamp to 255");
    }

    // ── AsymmetricTransitionWrapper ───────────────────────────────────────────

    #[test]
    fn test_wrapper_delegates_correctly() {
        let params = AsymmetricTransition::new(11, EasingCurve::EaseInOut, EasingCurve::EaseInOut);
        let wrapper = AsymmetricTransitionWrapper::new(params.clone());

        let (wa, wb) = wrapper.alphas_at_frame(5);
        let (pa, pb) = params.alphas_at_frame(5);
        assert!((wa - pa).abs() < 1e-6);
        assert!((wb - pb).abs() < 1e-6);
    }
}
