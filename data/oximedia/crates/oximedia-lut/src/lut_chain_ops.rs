//! LUT processing chain with algebraic operations and baking.
//!
//! [`LutChainOps`](crate::lut_chain_ops::LutChainOps) chains [`LutOperation`](crate::lut_chain_ops::LutOperation) steps (gamma, lift, gain,
//! saturation, exposure, or an embedded 3-D LUT) and can *bake* them into a
//! single 33³ [`Lut3DData`](crate::hald_clut::Lut3DData) for fast batch processing.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::hald_clut::Lut3DData;

// ---------------------------------------------------------------------------
// LutOperation
// ---------------------------------------------------------------------------

/// A single colour operation in a [`LutChainOps`] pipeline.
#[derive(Debug, Clone)]
pub enum LutOperation {
    /// Apply a 3-D LUT via trilinear interpolation.
    Apply3D(Lut3DData),
    /// Per-channel power function: `out = in ^ gamma`.
    Gamma(f32),
    /// RGB lift: add a constant to each channel.
    Lift(f32, f32, f32),
    /// RGB gain: multiply each channel.
    Gain(f32, f32, f32),
    /// Adjust saturation: `1.0` = no change, `0.0` = greyscale.
    Saturation(f32),
    /// Exposure adjustment in stops: `out = in * 2 ^ stops`.
    Exposure(f32),
}

impl LutOperation {
    /// Apply this operation to a single `(r, g, b)` pixel.
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        match self {
            Self::Apply3D(lut) => {
                let out = lut.lookup(r, g, b);
                (out[0], out[1], out[2])
            }
            Self::Gamma(gamma) => {
                let g = *gamma;
                (
                    r.clamp(0.0, 1.0).powf(g),
                    b.clamp(0.0, 1.0).powf(g),
                    b.clamp(0.0, 1.0).powf(g),
                )
            }
            Self::Lift(lr, lg, lb) => (r + lr, g + lg, b + lb),
            Self::Gain(gr, gg, gb) => (r * gr, g * gg, b * gb),
            Self::Saturation(sat) => {
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let r2 = luma + sat * (r - luma);
                let g2 = luma + sat * (g - luma);
                let b2 = luma + sat * (b - luma);
                (r2, g2, b2)
            }
            Self::Exposure(stops) => {
                let scale = 2.0_f32.powf(*stops);
                (r * scale, g * scale, b * scale)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LutChainOps
// ---------------------------------------------------------------------------

/// An ordered pipeline of [`LutOperation`] steps, optionally baked into a
/// single 33³ LUT for fast lookup.
#[derive(Debug, Clone)]
pub struct LutChainOps {
    operations: Vec<LutOperation>,
    baked: Option<Lut3DData>,
    /// Size of the baked LUT (default 33).
    pub bake_size: usize,
}

impl Default for LutChainOps {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            baked: None,
            bake_size: 33,
        }
    }
}

impl LutChainOps {
    /// Create an empty chain with the default bake size (33).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append an operation; invalidates any pre-baked LUT.
    ///
    /// Returns `&mut self` to support builder-pattern chaining.
    pub fn push(&mut self, op: LutOperation) -> &mut Self {
        self.baked = None;
        self.operations.push(op);
        self
    }

    /// Apply the full chain (without baking) to one pixel.
    #[must_use]
    fn apply_chain(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let mut out = (r, g, b);
        for op in &self.operations {
            out = op.apply(out.0, out.1, out.2);
        }
        out
    }

    /// Bake all operations into a single [`Lut3DData`] of size `self.bake_size`.
    ///
    /// After calling this, [`apply`](Self::apply) uses a single trilinear
    /// lookup instead of traversing the operation chain.
    pub fn bake(&mut self) {
        let size = self.bake_size;
        let scale = (size - 1) as f32;
        let mut data = Vec::with_capacity(size * size * size);
        for b_idx in 0..size {
            for g_idx in 0..size {
                for r_idx in 0..size {
                    let rf = r_idx as f32 / scale;
                    let gf = g_idx as f32 / scale;
                    let bf = b_idx as f32 / scale;
                    let (ro, go, bo) = self.apply_chain(rf, gf, bf);
                    data.push([ro.clamp(0.0, 1.0), go.clamp(0.0, 1.0), bo.clamp(0.0, 1.0)]);
                }
            }
        }
        self.baked = Some(Lut3DData { size, data });
    }

    /// Returns `true` if the chain has been baked.
    #[must_use]
    pub fn is_baked(&self) -> bool {
        self.baked.is_some()
    }

    /// Apply the chain to a single pixel.
    ///
    /// Uses the baked LUT if available, otherwise traverses the operation chain.
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        if let Some(baked) = &self.baked {
            let out = baked.lookup(r, g, b);
            (out[0], out[1], out[2])
        } else {
            self.apply_chain(r, g, b)
        }
    }

    /// Apply the chain to an interleaved RGB `f32` frame.
    ///
    /// Automatically bakes the chain if it has not been baked yet.
    /// Returns the processed frame as a new `Vec<f32>`.
    pub fn apply_frame(&mut self, pixels: &[f32]) -> Vec<f32> {
        if !self.is_baked() {
            self.bake();
        }
        let mut out = Vec::with_capacity(pixels.len());
        let mut i = 0;
        while i + 2 < pixels.len() {
            let (ro, go, bo) = self.apply(pixels[i], pixels[i + 1], pixels[i + 2]);
            out.push(ro);
            out.push(go);
            out.push(bo);
            i += 3;
        }
        // Passthrough any trailing incomplete triplet
        while i < pixels.len() {
            out.push(pixels[i]);
            i += 1;
        }
        out
    }

    /// Remove all operations and discard any baked LUT.
    pub fn clear(&mut self) {
        self.operations.clear();
        self.baked = None;
    }

    /// Number of operations in the chain.
    #[must_use]
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Returns `true` if the chain has no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_chain_is_empty() {
        let chain = LutChainOps::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(!chain.is_baked());
    }

    #[test]
    fn test_push_and_len() {
        let mut chain = LutChainOps::new();
        chain.push(LutOperation::Gamma(1.0));
        assert_eq!(chain.len(), 1);
        chain.push(LutOperation::Exposure(0.0));
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_push_invalidates_bake() {
        let mut chain = LutChainOps::new();
        chain.push(LutOperation::Gain(1.0, 1.0, 1.0));
        chain.bake();
        assert!(chain.is_baked());
        chain.push(LutOperation::Lift(0.0, 0.0, 0.0));
        assert!(!chain.is_baked(), "bake should be invalidated after push");
    }

    #[test]
    fn test_identity_chain_passthrough() {
        let mut chain = LutChainOps::new();
        chain.push(LutOperation::Gain(1.0, 1.0, 1.0));
        chain.push(LutOperation::Lift(0.0, 0.0, 0.0));
        chain.push(LutOperation::Saturation(1.0));
        chain.push(LutOperation::Exposure(0.0));
        let (r, g, b) = chain.apply(0.5, 0.3, 0.7);
        assert!((r - 0.5).abs() < 1e-4, "r={r}");
        assert!((g - 0.3).abs() < 1e-4, "g={g}");
        assert!((b - 0.7).abs() < 1e-4, "b={b}");
    }

    #[test]
    fn test_bake_and_apply_consistency() {
        let mut chain = LutChainOps::new();
        chain.push(LutOperation::Gain(0.8, 1.1, 0.9));
        let unbaked = chain.apply(0.4, 0.6, 0.2);
        chain.bake();
        let baked = chain.apply(0.4, 0.6, 0.2);
        // Results should be close (trilinear LUT introduces minor quantisation error)
        assert!(
            (unbaked.0 - baked.0).abs() < 0.01,
            "r: {unbaked:?} vs {baked:?}"
        );
        assert!(
            (unbaked.1 - baked.1).abs() < 0.01,
            "g: {unbaked:?} vs {baked:?}"
        );
        assert!(
            (unbaked.2 - baked.2).abs() < 0.01,
            "b: {unbaked:?} vs {baked:?}"
        );
    }

    #[test]
    fn test_exposure_op() {
        let op = LutOperation::Exposure(1.0); // +1 stop = ×2
        let (r, g, b) = op.apply(0.25, 0.5, 0.1);
        assert!((r - 0.5).abs() < 1e-5, "r={r}");
        assert!((g - 1.0).abs() < 1e-5, "g={g}");
        assert!((b - 0.2).abs() < 1e-5, "b={b}");
    }

    #[test]
    fn test_saturation_zero_is_greyscale() {
        let op = LutOperation::Saturation(0.0);
        let (r, g, b) = op.apply(0.8, 0.2, 0.5);
        assert!((r - g).abs() < 1e-5, "r={r} g={g}");
        assert!((g - b).abs() < 1e-5, "g={g} b={b}");
    }

    #[test]
    fn test_lift_op() {
        let op = LutOperation::Lift(0.1, 0.0, -0.05);
        let (r, _g, b) = op.apply(0.5, 0.5, 0.5);
        assert!((r - 0.6).abs() < 1e-5, "r={r}");
        assert!((b - 0.45).abs() < 1e-5, "b={b}");
    }

    #[test]
    fn test_clear_resets_chain() {
        let mut chain = LutChainOps::new();
        chain.push(LutOperation::Gamma(2.2));
        chain.bake();
        chain.clear();
        assert!(chain.is_empty());
        assert!(!chain.is_baked());
    }

    #[test]
    fn test_apply_frame_auto_bakes() {
        let mut chain = LutChainOps::new();
        chain.push(LutOperation::Gain(1.0, 1.0, 1.0));
        assert!(!chain.is_baked());
        let pixels = vec![0.5_f32, 0.3, 0.7];
        let out = chain.apply_frame(&pixels);
        assert!(chain.is_baked());
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.5).abs() < 0.02, "out[0]={}", out[0]);
    }

    #[test]
    fn test_apply_frame_empty() {
        let mut chain = LutChainOps::new();
        chain.push(LutOperation::Saturation(1.0));
        let out = chain.apply_frame(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_builder_pattern_chaining() {
        let mut chain = LutChainOps::new();
        chain
            .push(LutOperation::Exposure(-0.5))
            .push(LutOperation::Saturation(0.8))
            .push(LutOperation::Gain(1.0, 1.0, 1.0));
        assert_eq!(chain.len(), 3);
    }

    #[test]
    fn test_apply3d_identity_lut() {
        let ident = Lut3DData::identity(17);
        let op = LutOperation::Apply3D(ident);
        let (r, g, b) = op.apply(0.4, 0.6, 0.2);
        assert!((r - 0.4).abs() < 0.01, "r={r}");
        assert!((g - 0.6).abs() < 0.01, "g={g}");
        assert!((b - 0.2).abs() < 0.01, "b={b}");
    }
}
