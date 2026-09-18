//! Porter-Duff compositing operators for video effects.
//!
//! Implements the full suite of Porter-Duff alpha compositing operations
//! along with convenience functions for RGBA pixel buffers.

#![allow(dead_code)]
#![allow(missing_docs)]

/// Porter-Duff compositing operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeOp {
    /// Source over destination (normal alpha blend).
    Over,
    /// Destination over source.
    Under,
    /// Source where source and destination overlap.
    In,
    /// Source where source and destination do not overlap.
    Out,
    /// Source over destination, only where destination is opaque.
    Atop,
    /// Source and destination combined, excluding overlap.
    Xor,
    /// Clear both source and destination.
    Clear,
}

impl CompositeOp {
    /// Returns the Porter-Duff (Fa, Fb) factor pair for this operator.
    ///
    /// The combined result for a channel `c` is:
    /// `out_c = src_c * alpha_s * Fa + dst_c * alpha_d * Fb`
    /// (pre-multiplied formulation).
    #[must_use]
    pub fn porter_duff_factors(&self, alpha_s: f32, alpha_d: f32) -> (f32, f32) {
        match self {
            CompositeOp::Over => (1.0, 1.0 - alpha_s),
            CompositeOp::Under => (1.0 - alpha_d, 1.0),
            CompositeOp::In => (alpha_d, 0.0),
            CompositeOp::Out => (1.0 - alpha_d, 0.0),
            CompositeOp::Atop => (alpha_d, 1.0 - alpha_s),
            CompositeOp::Xor => (1.0 - alpha_d, 1.0 - alpha_s),
            CompositeOp::Clear => (0.0, 0.0),
        }
    }

    /// Whether the operator results in complete transparency when either input is transparent.
    #[must_use]
    pub fn requires_both_opaque(&self) -> bool {
        matches!(self, CompositeOp::In)
    }
}

/// Composite two pre-multiplied RGBA values using the given operator.
///
/// Inputs and outputs are in `[0.0, 1.0]` pre-multiplied form
/// `(r*a, g*a, b*a, a)`.
#[must_use]
pub fn composite_premul(
    src: (f32, f32, f32, f32),
    dst: (f32, f32, f32, f32),
    op: CompositeOp,
) -> (f32, f32, f32, f32) {
    let (sr, sg, sb, sa) = src;
    let (dr, dg, db, da) = dst;
    let (fa, fb) = op.porter_duff_factors(sa, da);
    let out_r = (sr * fa + dr * fb).clamp(0.0, 1.0);
    let out_g = (sg * fa + dg * fb).clamp(0.0, 1.0);
    let out_b = (sb * fa + db * fb).clamp(0.0, 1.0);
    let out_a = (sa * fa + da * fb).clamp(0.0, 1.0);
    (out_r, out_g, out_b, out_a)
}

/// Composite straight-alpha (un-premultiplied) RGBA float values.
///
/// Converts to pre-multiplied internally, composites, then converts back.
#[must_use]
pub fn composite_straight(
    src: (f32, f32, f32, f32),
    dst: (f32, f32, f32, f32),
    op: CompositeOp,
) -> (f32, f32, f32, f32) {
    let premul = |c: (f32, f32, f32, f32)| (c.0 * c.3, c.1 * c.3, c.2 * c.3, c.3);
    let (pr, pg, pb, pa) = composite_premul(premul(src), premul(dst), op);
    if pa < 1e-8 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    (pr / pa, pg / pa, pb / pa, pa)
}

/// Apply Porter-Duff compositing to RGBA byte buffers (4 bytes per pixel).
///
/// `src_rgba` is composited **over** `dst_rgba` using the specified operator.
/// The result is written back into `dst_rgba`.
///
/// # Panics
///
/// Panics if `src_rgba` and `dst_rgba` do not have the same length.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]
pub fn composite_buffers(src_rgba: &[u8], dst_rgba: &mut [u8], op: CompositeOp) {
    assert_eq!(
        src_rgba.len(),
        dst_rgba.len(),
        "buffers must be the same size"
    );
    for (s, d) in src_rgba.chunks_exact(4).zip(dst_rgba.chunks_exact_mut(4)) {
        let sr = f32::from(s[0]) / 255.0;
        let sg = f32::from(s[1]) / 255.0;
        let sb = f32::from(s[2]) / 255.0;
        let sa = f32::from(s[3]) / 255.0;
        let dr = f32::from(d[0]) / 255.0;
        let dg = f32::from(d[1]) / 255.0;
        let db = f32::from(d[2]) / 255.0;
        let da = f32::from(d[3]) / 255.0;
        // pre-multiply
        let src_pre = (sr * sa, sg * sa, sb * sa, sa);
        let dst_pre = (dr * da, dg * da, db * da, da);
        let (or_p, og_p, ob_p, oa) = composite_premul(src_pre, dst_pre, op);
        let (or_, og_, ob_) = if oa < 1e-8 {
            (0.0, 0.0, 0.0)
        } else {
            (or_p / oa, og_p / oa, ob_p / oa)
        };
        d[0] = (or_ * 255.0).round().clamp(0.0, 255.0) as u8;
        d[1] = (og_ * 255.0).round().clamp(0.0, 255.0) as u8;
        d[2] = (ob_ * 255.0).round().clamp(0.0, 255.0) as u8;
        d[3] = (oa * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(r: f32, g: f32, b: f32, a: f32) -> (f32, f32, f32, f32) {
        (r, g, b, a)
    }

    #[test]
    fn test_composite_over_opaque_src() {
        // Fully opaque red over anything → red
        let src = px(1.0, 0.0, 0.0, 1.0);
        let dst = px(0.0, 1.0, 0.0, 1.0);
        let (r, g, b, a) = composite_straight(src, dst, CompositeOp::Over);
        assert!((r - 1.0).abs() < 1e-5);
        assert!(g.abs() < 1e-5);
        assert!(b.abs() < 1e-5);
        assert!((a - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_composite_over_transparent_src() {
        // Fully transparent src over opaque dst → dst unchanged
        let src = px(1.0, 0.0, 0.0, 0.0);
        let dst = px(0.0, 1.0, 0.0, 1.0);
        let (r, g, b, a) = composite_straight(src, dst, CompositeOp::Over);
        assert!(r.abs() < 1e-5);
        assert!((g - 1.0).abs() < 1e-5);
        assert!(b.abs() < 1e-5);
        assert!((a - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_composite_clear() {
        let src = px(1.0, 1.0, 1.0, 1.0);
        let dst = px(0.5, 0.5, 0.5, 1.0);
        let (_, _, _, a) = composite_premul(
            (src.0 * src.3, src.1 * src.3, src.2 * src.3, src.3),
            (dst.0 * dst.3, dst.1 * dst.3, dst.2 * dst.3, dst.3),
            CompositeOp::Clear,
        );
        assert!(a.abs() < 1e-5);
    }

    #[test]
    fn test_composite_in_transparent_dst() {
        // In with transparent dst → transparent result
        let src = px(1.0, 0.0, 0.0, 1.0);
        let dst = px(0.0, 0.0, 0.0, 0.0);
        let (_, _, _, a) = composite_straight(src, dst, CompositeOp::In);
        assert!(a.abs() < 1e-5);
    }

    #[test]
    fn test_composite_out_opaque_dst() {
        // Out with fully opaque dst → transparent (source hidden by destination)
        let src = px(1.0, 0.0, 0.0, 1.0);
        let dst = px(0.0, 1.0, 0.0, 1.0);
        let (_, _, _, a) = composite_premul(
            (src.0 * src.3, src.1 * src.3, src.2 * src.3, src.3),
            (dst.0 * dst.3, dst.1 * dst.3, dst.2 * dst.3, dst.3),
            CompositeOp::Out,
        );
        assert!(a.abs() < 1e-5);
    }

    #[test]
    fn test_porter_duff_over_factors() {
        let (fa, fb) = CompositeOp::Over.porter_duff_factors(1.0, 1.0);
        assert!((fa - 1.0).abs() < 1e-5);
        assert!(fb.abs() < 1e-5);
    }

    #[test]
    fn test_porter_duff_xor_equal_alpha() {
        // Xor: both factors are 1 - alpha of the other
        let (fa, fb) = CompositeOp::Xor.porter_duff_factors(0.8, 0.6);
        assert!((fa - 0.4).abs() < 1e-5);
        assert!((fb - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_requires_both_opaque_in() {
        assert!(CompositeOp::In.requires_both_opaque());
    }

    #[test]
    fn test_requires_both_opaque_over() {
        assert!(!CompositeOp::Over.requires_both_opaque());
    }

    #[test]
    fn test_composite_buffers_over_opaque() {
        // Red src over green dst → red result
        let src = vec![255u8, 0, 0, 255];
        let mut dst = vec![0u8, 255, 0, 255];
        composite_buffers(&src, &mut dst, CompositeOp::Over);
        assert_eq!(dst[0], 255);
        assert_eq!(dst[1], 0);
        assert_eq!(dst[3], 255);
    }

    #[test]
    fn test_composite_buffers_clear() {
        let src = vec![255u8, 255, 255, 255];
        let mut dst = vec![128u8, 128, 128, 255];
        composite_buffers(&src, &mut dst, CompositeOp::Clear);
        assert_eq!(dst[3], 0);
    }

    #[test]
    fn test_composite_buffers_size_matches() {
        let src = vec![100u8, 100, 100, 200, 50, 50, 50, 128];
        let mut dst = vec![0u8; 8];
        composite_buffers(&src, &mut dst, CompositeOp::Over);
        // should not panic
    }

    #[test]
    fn test_composite_under_reversal() {
        // Under is Over with src and dst swapped
        let src = px(1.0, 0.0, 0.0, 1.0);
        let dst = px(0.0, 1.0, 0.0, 0.5);
        let over = composite_straight(src, dst, CompositeOp::Over);
        let under = composite_straight(dst, src, CompositeOp::Over);
        let under2 = composite_straight(src, dst, CompositeOp::Under);
        // Under(src, dst) should equal Over(dst, src)
        assert!((under.0 - under2.0).abs() < 1e-5);
        assert!((under.1 - under2.1).abs() < 1e-5);
        // Over and Under should differ when dst is semi-transparent
        assert!((over.0 - under2.0).abs() > 1e-3 || (over.1 - under2.1).abs() > 1e-3);
    }
}
