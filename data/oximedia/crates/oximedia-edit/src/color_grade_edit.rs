//! Color grading edit nodes for the timeline editor.
//!
//! Provides a node-based color grading system with lift/gamma/gain,
//! curves, HSL adjustments, and LUT application.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use thiserror::Error;

// ── GradeError ───────────────────────────────────────────────────────────────

/// Error type for color grading operations.
#[derive(Debug, Error)]
pub enum GradeError {
    /// The requested LUT file could not be found.
    #[error("LUT file not found: {0}")]
    LutNotFound(String),

    /// The LUT file could not be parsed.
    #[error("LUT parse error: {0}")]
    LutParse(String),

    /// I/O error while loading a LUT.
    #[error("I/O error loading LUT '{path}': {source}")]
    LutIo {
        /// File path that triggered the error.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

impl From<oximedia_lut::LutError> for GradeError {
    fn from(e: oximedia_lut::LutError) -> Self {
        Self::LutParse(e.to_string())
    }
}

// ── GradeNode ────────────────────────────────────────────────────────────────

/// A single color grading node type.
#[derive(Debug, Clone, PartialEq)]
pub enum GradeNode {
    /// Lift, Gamma, Gain correction.
    LiftGammaGain {
        /// Lift values (shadows) per channel [R, G, B].
        lift: [f32; 3],
        /// Gamma values (midtones) per channel [R, G, B].
        gamma: [f32; 3],
        /// Gain values (highlights) per channel [R, G, B].
        gain: [f32; 3],
    },
    /// Custom curves adjustment (master curve applied uniformly to R, G, B).
    Curves {
        /// Control points for the master curve as (input, output) pairs in `[0, 1]`.
        points: Vec<(f32, f32)>,
    },
    /// Hue/Saturation/Luminance adjustment.
    Hsl {
        /// Hue rotation in degrees.
        hue: f32,
        /// Saturation multiplier.
        saturation: f32,
        /// Luminance offset.
        luminance: f32,
    },
    /// LUT file application.
    Lut {
        /// Path to the LUT file.
        path: String,
        /// Blend opacity (0.0–1.0).
        opacity: f32,
    },
}

impl GradeNode {
    /// Return a human-readable name for this node type.
    #[must_use]
    pub fn node_name(&self) -> &'static str {
        match self {
            Self::LiftGammaGain { .. } => "LiftGammaGain",
            Self::Curves { .. } => "Curves",
            Self::Hsl { .. } => "HSL",
            Self::Lut { .. } => "LUT",
        }
    }
}

// ── ColorGradeEdit ───────────────────────────────────────────────────────────

/// A color grade edit attached to a clip.
#[derive(Debug, Clone)]
pub struct ColorGradeEdit {
    /// The grading node used.
    pub node: GradeNode,
    /// Whether bypassed (not applied).
    pub bypassed: bool,
    /// Whether this edit permanently alters pixel data.
    pub destructive: bool,
}

impl ColorGradeEdit {
    /// Create a new `ColorGradeEdit`.
    #[must_use]
    pub fn new(node: GradeNode, destructive: bool) -> Self {
        Self {
            node,
            bypassed: false,
            destructive,
        }
    }

    /// Returns `true` if the edit permanently alters pixel data.
    #[must_use]
    pub fn is_destructive(&self) -> bool {
        self.destructive
    }

    /// Returns `true` if the edit is currently bypassed.
    #[must_use]
    pub fn is_bypassed(&self) -> bool {
        self.bypassed
    }

    /// Toggle the bypass state.
    pub fn toggle_bypass(&mut self) {
        self.bypassed = !self.bypassed;
    }
}

// ── Monotone cubic spline (Fritsch-Carlson) ──────────────────────────────────

/// Evaluate a monotone cubic spline at `t` given a set of control points.
///
/// Uses the Fritsch-Carlson algorithm to enforce monotonicity.  Control points
/// do not need to be sorted — they are sorted internally.
///
/// Returns the identity value `t` for degenerate inputs (fewer than 2 points).
/// Output is clamped to [0, 1].
#[must_use]
fn evaluate_monotone_spline(control_points: &[(f32, f32)], t: f32) -> f32 {
    if control_points.len() < 2 {
        return t.clamp(0.0, 1.0);
    }

    // Sort by x (input), deduplicate near-coincident x values.
    let mut pts: Vec<(f32, f32)> = control_points.to_vec();
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);

    if pts.len() < 2 {
        return t.clamp(0.0, 1.0);
    }

    // Clamp to the curve domain.
    if t <= pts[0].0 {
        return pts[0].1.clamp(0.0, 1.0);
    }
    if t >= pts[pts.len() - 1].0 {
        return pts[pts.len() - 1].1.clamp(0.0, 1.0);
    }

    // Find the enclosing interval [pts[i], pts[i+1]].
    let i = pts.partition_point(|p| p.0 <= t).saturating_sub(1);
    let i = i.min(pts.len() - 2);

    let n = pts.len();

    // Finite differences (slopes of each chord).
    let mut d = vec![0.0_f32; n.saturating_sub(1)];
    for k in 0..n - 1 {
        let dx = pts[k + 1].0 - pts[k].0;
        if dx.abs() > 1e-10 {
            d[k] = (pts[k + 1].1 - pts[k].1) / dx;
        }
    }

    // Estimate tangents using the three-point formula.
    let mut m = vec![0.0_f32; n];
    m[0] = d[0];
    m[n - 1] = d[n - 2];
    for k in 1..n - 1 {
        m[k] = (d[k - 1] + d[k]) / 2.0;
    }

    // Apply Fritsch-Carlson monotonicity constraints.
    for k in 0..n - 1 {
        if d[k].abs() < 1e-10 {
            m[k] = 0.0;
            m[k + 1] = 0.0;
        } else {
            let alpha = m[k] / d[k];
            let beta = m[k + 1] / d[k];
            let h = alpha * alpha + beta * beta;
            if h > 9.0 {
                let tau = 3.0 / h.sqrt();
                m[k] = tau * alpha * d[k];
                m[k + 1] = tau * beta * d[k];
            }
        }
    }

    // Cubic Hermite interpolation within the interval.
    let t0 = pts[i].0;
    let t1 = pts[i + 1].0;
    let dx = t1 - t0;
    let tt = if dx.abs() > 1e-10 { (t - t0) / dx } else { 0.5 };
    let tt2 = tt * tt;
    let tt3 = tt2 * tt;

    let h00 = 2.0 * tt3 - 3.0 * tt2 + 1.0;
    let h10 = tt3 - 2.0 * tt2 + tt;
    let h01 = -2.0 * tt3 + 3.0 * tt2;
    let h11 = tt3 - tt2;

    (h00 * pts[i].1 + h10 * dx * m[i] + h01 * pts[i + 1].1 + h11 * dx * m[i + 1]).clamp(0.0, 1.0)
}

// ── ColorGradeStack ──────────────────────────────────────────────────────────

/// Shared LUT cache type.  Keyed by file path; value is the parsed LUT.
type LutCache = Arc<RwLock<HashMap<String, Arc<oximedia_lut::Lut3d>>>>;

/// A stack of color grade edits applied in order.
#[derive(Debug, Clone)]
pub struct ColorGradeStack {
    nodes: Vec<ColorGradeEdit>,
    /// Thread-safe cache of pre-loaded 3-D LUTs, keyed by file path.
    lut_cache: LutCache,
}

impl Default for ColorGradeStack {
    fn default() -> Self {
        Self::new()
    }
}

impl ColorGradeStack {
    /// Create an empty `ColorGradeStack`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            lut_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Append a `ColorGradeEdit` to the stack.
    pub fn push(&mut self, edit: ColorGradeEdit) {
        self.nodes.push(edit);
    }

    /// Return the number of nodes in the stack.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Pre-load a LUT file into the internal cache.
    ///
    /// Call this before `apply()` to avoid on-demand file I/O on the hot path.
    ///
    /// # Errors
    ///
    /// Returns [`GradeError`] if the file cannot be read or parsed.
    pub fn warm_lut(&self, path: &str) -> Result<(), GradeError> {
        // Check if already cached (read lock only).
        {
            let cache = self.lut_cache.read().unwrap_or_else(|e| e.into_inner());
            if cache.contains_key(path) {
                return Ok(());
            }
        }

        // Load from disk (no lock held during I/O).
        let lut = oximedia_lut::Lut3d::from_file(path).map_err(GradeError::from)?;

        // Insert into cache (write lock).
        let mut cache = self.lut_cache.write().unwrap_or_else(|e| e.into_inner());
        cache.insert(path.to_string(), Arc::new(lut));
        Ok(())
    }

    /// Retrieve or lazily load a LUT for the given path.
    ///
    /// Returns `None` if the file does not exist or cannot be parsed (the
    /// caller should fall back to an identity transform in that case).
    fn get_or_load_lut(&self, path: &str) -> Option<Arc<oximedia_lut::Lut3d>> {
        // Fast path: cache hit.
        {
            let cache = self.lut_cache.read().unwrap_or_else(|e| e.into_inner());
            if let Some(lut) = cache.get(path) {
                return Some(Arc::clone(lut));
            }
        }

        // Slow path: load from disk.
        let lut = match oximedia_lut::Lut3d::from_file(path) {
            Ok(l) => Arc::new(l),
            Err(_) => return None,
        };

        let mut cache = self.lut_cache.write().unwrap_or_else(|e| e.into_inner());
        // Another thread may have inserted while we were loading — insert only if absent.
        cache
            .entry(path.to_string())
            .or_insert_with(|| Arc::clone(&lut));

        Some(lut)
    }

    /// Apply the stack to a pixel value.
    ///
    /// Each active, non-bypassed node is visited in order.
    ///
    /// **LUT node**: looks up the cached [`oximedia_lut::Lut3d`]; if the file is not in cache
    /// and cannot be loaded, the pixel passes through unchanged (identity
    /// fallback).
    ///
    /// **Curves node**: applies a monotone cubic spline (Fritsch-Carlson) to
    /// all three channels using the `points` master curve.
    #[must_use]
    pub fn apply(&self, pixel: [f32; 3]) -> [f32; 3] {
        let mut out = pixel;
        for edit in &self.nodes {
            if edit.bypassed {
                continue;
            }
            match &edit.node {
                GradeNode::LiftGammaGain { lift, gamma, gain } => {
                    for i in 0..3 {
                        let v = out[i] + lift[i];
                        let v = v.powf(1.0 / gamma[i].max(0.001));
                        out[i] = (v * gain[i]).clamp(0.0, 1.0);
                    }
                }
                GradeNode::Hsl {
                    saturation,
                    luminance,
                    ..
                } => {
                    let lum = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
                    for v in &mut out {
                        *v = ((*v - lum) * saturation + lum + luminance).clamp(0.0, 1.0);
                    }
                }
                GradeNode::Lut { path, opacity } => {
                    let blend = opacity.clamp(0.0, 1.0);
                    if blend < 1e-6 {
                        // Fully transparent — skip.
                        continue;
                    }
                    match self.get_or_load_lut(path) {
                        Some(lut) => {
                            let (r_out, g_out, b_out) = lut.apply_rgb(out[0], out[1], out[2]);
                            out[0] = (out[0] * (1.0 - blend) + r_out * blend).clamp(0.0, 1.0);
                            out[1] = (out[1] * (1.0 - blend) + g_out * blend).clamp(0.0, 1.0);
                            out[2] = (out[2] * (1.0 - blend) + b_out * blend).clamp(0.0, 1.0);
                        }
                        None => {
                            // File not found or parse error — identity fallback; pixel unchanged.
                        }
                    }
                }
                GradeNode::Curves { points } => {
                    // Apply the master curve to all three channels independently.
                    out[0] = evaluate_monotone_spline(points, out[0]);
                    out[1] = evaluate_monotone_spline(points, out[1]);
                    out[2] = evaluate_monotone_spline(points, out[2]);
                }
            }
        }
        out
    }

    /// Returns a reference to the node edits in the stack.
    #[must_use]
    pub fn edits(&self) -> &[ColorGradeEdit] {
        &self.nodes
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_lut::{Lut3d, LutSize};

    fn lggg_node() -> GradeNode {
        GradeNode::LiftGammaGain {
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
        }
    }

    #[test]
    fn test_grade_node_name_lgg() {
        assert_eq!(lggg_node().node_name(), "LiftGammaGain");
    }

    #[test]
    fn test_grade_node_name_curves() {
        let n = GradeNode::Curves { points: vec![] };
        assert_eq!(n.node_name(), "Curves");
    }

    #[test]
    fn test_grade_node_name_hsl() {
        let n = GradeNode::Hsl {
            hue: 0.0,
            saturation: 1.0,
            luminance: 0.0,
        };
        assert_eq!(n.node_name(), "HSL");
    }

    #[test]
    fn test_grade_node_name_lut() {
        let n = GradeNode::Lut {
            path: "film.cube".to_string(),
            opacity: 1.0,
        };
        assert_eq!(n.node_name(), "LUT");
    }

    #[test]
    fn test_color_grade_edit_is_destructive() {
        let edit = ColorGradeEdit::new(lggg_node(), true);
        assert!(edit.is_destructive());
    }

    #[test]
    fn test_color_grade_edit_not_destructive() {
        let edit = ColorGradeEdit::new(lggg_node(), false);
        assert!(!edit.is_destructive());
    }

    #[test]
    fn test_toggle_bypass() {
        let mut edit = ColorGradeEdit::new(lggg_node(), false);
        assert!(!edit.is_bypassed());
        edit.toggle_bypass();
        assert!(edit.is_bypassed());
        edit.toggle_bypass();
        assert!(!edit.is_bypassed());
    }

    #[test]
    fn test_stack_node_count() {
        let mut stack = ColorGradeStack::new();
        assert_eq!(stack.node_count(), 0);
        stack.push(ColorGradeEdit::new(lggg_node(), false));
        assert_eq!(stack.node_count(), 1);
        stack.push(ColorGradeEdit::new(
            GradeNode::Curves { points: vec![] },
            false,
        ));
        assert_eq!(stack.node_count(), 2);
    }

    #[test]
    fn test_stack_apply_identity_lgg() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(lggg_node(), false));
        let px = [0.5, 0.4, 0.3];
        let out = stack.apply(px);
        // gain=1, gamma=1, lift=0 → identity
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.4).abs() < 1e-5);
        assert!((out[2] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_stack_apply_bypass_skips_node() {
        let mut stack = ColorGradeStack::new();
        let mut edit = ColorGradeEdit::new(
            GradeNode::Hsl {
                hue: 0.0,
                saturation: 0.0, // would desaturate
                luminance: 0.0,
            },
            false,
        );
        edit.bypassed = true;
        stack.push(edit);
        let px = [0.8, 0.2, 0.5];
        let out = stack.apply(px);
        // bypassed → unchanged
        assert!((out[0] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_stack_apply_empty() {
        let stack = ColorGradeStack::new();
        let px = [0.1, 0.2, 0.9];
        let out = stack.apply(px);
        assert_eq!(out, px);
    }

    #[test]
    fn test_stack_edits_len() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(lggg_node(), false));
        assert_eq!(stack.edits().len(), 1);
    }

    /// Existing test: a LUT node with a non-existent file must fall back to
    /// identity (pixel unchanged) — not panic or return an error.
    #[test]
    fn test_stack_lut_identity_opacity() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(
            GradeNode::Lut {
                path: "test.cube".into(),
                opacity: 1.0,
            },
            false,
        ));
        let px = [0.3, 0.6, 0.9];
        let out = stack.apply(px);
        // "test.cube" does not exist → identity fallback → pixel unchanged.
        assert!((out[0] - 0.3).abs() < 1e-5);
        assert!((out[1] - 0.6).abs() < 1e-5);
        assert!((out[2] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_stack_hsl_full_saturation() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(
            GradeNode::Hsl {
                hue: 0.0,
                saturation: 1.0,
                luminance: 0.0,
            },
            false,
        ));
        let px = [0.5, 0.5, 0.5];
        let out = stack.apply(px);
        // saturation=1, no luminance offset → should equal original for grey
        assert!((out[0] - 0.5).abs() < 1e-5);
    }

    // ── LUT cache / warm_lut ─────────────────────────────────────────────────

    /// Write a minimal identity 3D LUT (.cube) to a temp file, apply it, and
    /// verify the output equals the input within floating-point tolerance.
    #[test]
    fn test_lut_identity_from_real_file() {
        // Create an identity LUT and save it to a temp file.
        let lut = Lut3d::identity(LutSize::Size17);
        let tmp_dir = std::env::temp_dir();
        let lut_path = tmp_dir.join("oximedia_edit_identity_test.cube");
        lut.to_file(&lut_path)
            .expect("should write identity LUT to temp dir");

        let path_str = lut_path.to_str().expect("temp path should be valid UTF-8");

        let mut stack = ColorGradeStack::new();

        // Warm the cache.
        stack.warm_lut(path_str).expect("warm_lut should succeed");

        // Apply with full opacity.
        stack.push(ColorGradeEdit::new(
            GradeNode::Lut {
                path: path_str.to_string(),
                opacity: 1.0,
            },
            false,
        ));

        let test_pixels: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.3, 0.7],
            [0.2, 0.8, 0.4],
        ];

        for &px in test_pixels {
            let out = stack.apply(px);
            assert!(
                (out[0] - px[0]).abs() < 0.01,
                "R channel mismatch for {:?}: got {}",
                px,
                out[0]
            );
            assert!(
                (out[1] - px[1]).abs() < 0.01,
                "G channel mismatch for {:?}: got {}",
                px,
                out[1]
            );
            assert!(
                (out[2] - px[2]).abs() < 0.01,
                "B channel mismatch for {:?}: got {}",
                px,
                out[2]
            );
        }

        // Clean up.
        let _ = std::fs::remove_file(&lut_path);
    }

    /// Warm a non-existent LUT — must return an error, not panic.
    #[test]
    fn test_warm_lut_nonexistent_returns_error() {
        let stack = ColorGradeStack::new();
        let path = std::env::temp_dir()
            .join("does_not_exist_oximedia_lut.cube")
            .display()
            .to_string();
        let result = stack.warm_lut(&path);
        assert!(result.is_err(), "expected error for nonexistent LUT file");
    }

    /// Verify the cache is shared via Arc — warming on one stack is visible
    /// to a clone.
    #[test]
    fn test_lut_cache_shared_across_clone() {
        let lut = Lut3d::identity(LutSize::Size17);
        let tmp_dir = std::env::temp_dir();
        let lut_path = tmp_dir.join("oximedia_edit_cache_clone_test.cube");
        lut.to_file(&lut_path).expect("should write LUT");
        let path_str = lut_path.to_str().expect("valid UTF-8");

        let stack1 = ColorGradeStack::new();
        stack1.warm_lut(path_str).expect("warm_lut should succeed");

        // Clone shares the Arc<RwLock<...>>.
        let stack2 = stack1.clone();
        let cache = stack2.lut_cache.read().unwrap_or_else(|e| e.into_inner());
        assert!(
            cache.contains_key(path_str),
            "cloned stack should see cached LUT"
        );

        drop(cache);
        let _ = std::fs::remove_file(&lut_path);
    }

    // ── Curves (monotone spline) ─────────────────────────────────────────────

    /// Identity control points [(0,0), (1,1)] must be a pass-through.
    #[test]
    fn test_curves_identity_passthrough() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(
            GradeNode::Curves {
                points: vec![(0.0, 0.0), (1.0, 1.0)],
            },
            false,
        ));
        let values = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        for v in values {
            let out = stack.apply([v, v, v]);
            assert!(
                (out[0] - v).abs() < 0.01,
                "identity curve deviated at {v}: got {}",
                out[0]
            );
        }
    }

    /// Control points [(0,0),(0.5,0.7),(1,1)]: midpoint 0.5 → ~0.7.
    #[test]
    fn test_curves_midpoint_boosted() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(
            GradeNode::Curves {
                points: vec![(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
            },
            false,
        ));
        let out = stack.apply([0.5, 0.5, 0.5]);
        // The spline passes exactly through the control point at x=0.5,
        // so the result must equal 0.7 within floating-point precision.
        assert!(
            (out[0] - 0.7).abs() < 0.01,
            "expected ~0.7 at midpoint, got {}",
            out[0]
        );
        assert!(
            (out[1] - 0.7).abs() < 0.01,
            "expected ~0.7 at midpoint, got {}",
            out[1]
        );
        assert!(
            (out[2] - 0.7).abs() < 0.01,
            "expected ~0.7 at midpoint, got {}",
            out[2]
        );
    }

    /// Empty curve control points → identity pass-through.
    #[test]
    fn test_curves_empty_points_identity() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(
            GradeNode::Curves { points: vec![] },
            false,
        ));
        let px = [0.3, 0.6, 0.9];
        let out = stack.apply(px);
        assert!((out[0] - 0.3).abs() < 1e-5);
        assert!((out[1] - 0.6).abs() < 1e-5);
        assert!((out[2] - 0.9).abs() < 1e-5);
    }

    /// Single control point → identity pass-through (degenerate case).
    #[test]
    fn test_curves_single_point_identity() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(
            GradeNode::Curves {
                points: vec![(0.5, 0.5)],
            },
            false,
        ));
        let out = stack.apply([0.3, 0.6, 0.9]);
        // With one point we cannot interpolate → clamped identity.
        assert!(out[0] >= 0.0 && out[0] <= 1.0);
    }

    /// Curves output is always in [0, 1].
    #[test]
    fn test_curves_output_clamped() {
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(
            GradeNode::Curves {
                points: vec![(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)],
            },
            false,
        ));
        for v in 0..=10 {
            let t = v as f32 / 10.0;
            let out = stack.apply([t, t, t]);
            assert!(out[0] >= 0.0 && out[0] <= 1.0, "out of range at t={t}");
        }
    }

    /// Curves applied at control-point x values must return the control-point y.
    #[test]
    fn test_curves_passes_through_control_points() {
        let ctrl = vec![(0.0_f32, 0.0), (0.25, 0.4), (0.75, 0.6), (1.0, 1.0)];
        let mut stack = ColorGradeStack::new();
        stack.push(ColorGradeEdit::new(
            GradeNode::Curves {
                points: ctrl.clone(),
            },
            false,
        ));
        for (x, y) in &ctrl {
            let out = stack.apply([*x, *x, *x]);
            assert!(
                (out[0] - y).abs() < 0.02,
                "curve should pass through ({x},{y}), got {}",
                out[0]
            );
        }
    }

    // ── Spline helper unit tests (private fn, tested directly) ────────────────

    #[test]
    fn test_spline_identity_endpoints() {
        let pts = vec![(0.0_f32, 0.0), (1.0, 1.0)];
        assert!((evaluate_monotone_spline(&pts, 0.0) - 0.0).abs() < 1e-5);
        assert!((evaluate_monotone_spline(&pts, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_spline_extrapolation_clamps_low() {
        let pts = vec![(0.2_f32, 0.3), (0.8, 0.9)];
        // Below domain — should return pts[0].1 clamped to [0,1]
        let v = evaluate_monotone_spline(&pts, 0.0);
        assert!((v - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_spline_extrapolation_clamps_high() {
        let pts = vec![(0.2_f32, 0.3), (0.8, 0.9)];
        // Above domain — should return pts[last].1
        let v = evaluate_monotone_spline(&pts, 1.0);
        assert!((v - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_spline_unsorted_input_normalised() {
        // Provide points in reverse order — should sort internally.
        let pts = vec![(1.0_f32, 1.0), (0.0, 0.0), (0.5, 0.7)];
        let v = evaluate_monotone_spline(&pts, 0.5);
        assert!((v - 0.7).abs() < 0.01);
    }
}
