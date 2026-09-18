//! Motion mask: exclude moving foreground objects from camera-motion estimation.
//!
//! When a scene contains independently moving objects (people walking, vehicles,
//! etc.) the features tracked on those objects introduce outliers that corrupt
//! the global motion estimate.  This module provides a pixel-level binary mask
//! that marks foreground pixels as *excluded* so the tracker operates only on
//! the background.
//!
//! # Algorithm
//!
//! A [`MotionMaskBuilder`] compares each pixel's frame-to-frame difference
//! against an adaptive per-pixel background model (running mean + variance).
//! Pixels whose difference exceeds a configurable number of standard deviations
//! are marked as foreground (moving).  The raw binary mask is then morphologically
//! dilated to cover object boundaries and blurred to create a soft exclusion
//! weight map used during feature scoring.
//!
//! For simple sequences a global-threshold variant [`ThresholdMask`] is also
//! provided as a lower-cost alternative.

use crate::Frame;

// ---------------------------------------------------------------------------
// Binary mask type
// ---------------------------------------------------------------------------

/// A single-channel binary mask aligned with a video frame.
///
/// A value of `true` marks a pixel as belonging to a *foreground* (moving)
/// object and therefore **excluded** from camera-motion estimation.
#[derive(Debug, Clone)]
pub struct MotionMask {
    /// Width of the mask.
    pub width: usize,
    /// Height of the mask.
    pub height: usize,
    /// Row-major boolean map: `true` = excluded (foreground).
    pub data: Vec<bool>,
}

impl MotionMask {
    /// Create an all-background (nothing excluded) mask.
    #[must_use]
    pub fn background(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![false; width * height],
        }
    }

    /// Create an all-foreground (everything excluded) mask.
    #[must_use]
    pub fn foreground(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![true; width * height],
        }
    }

    /// Get the mask value at `(x, y)`.
    ///
    /// Returns `false` (background) for out-of-bounds coordinates.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> bool {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            false
        }
    }

    /// Set the mask value at `(x, y)`.
    pub fn set(&mut self, x: usize, y: usize, value: bool) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }

    /// Return the fraction of pixels marked as foreground.
    #[must_use]
    pub fn foreground_fraction(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let fg = self.data.iter().filter(|&&b| b).count();
        fg as f64 / self.data.len() as f64
    }

    /// Morphological dilation with a square structuring element of radius `r`.
    ///
    /// Any background pixel within radius `r` of a foreground pixel is also
    /// marked foreground.  This expands object boundaries to reduce leakage
    /// of edge features into the estimation.
    #[must_use]
    pub fn dilated(&self, radius: usize) -> Self {
        let mut out = Self::background(self.width, self.height);
        let w = self.width;
        let h = self.height;

        for y in 0..h {
            for x in 0..w {
                if self.get(x, y) {
                    // Mark neighbourhood.
                    let y0 = y.saturating_sub(radius);
                    let y1 = (y + radius + 1).min(h);
                    let x0 = x.saturating_sub(radius);
                    let x1 = (x + radius + 1).min(w);
                    for ny in y0..y1 {
                        for nx in x0..x1 {
                            out.set(nx, ny, true);
                        }
                    }
                }
            }
        }

        out
    }

    /// Create a soft weight map where foreground = 0.0 and background = 1.0,
    /// with a Gaussian-like transition in the dilation border.
    ///
    /// `dilation_radius` controls the transition width.
    #[must_use]
    pub fn to_weight_map(&self, dilation_radius: usize) -> Vec<f32> {
        let dilated = self.dilated(dilation_radius);
        // Hard weights for now; a future version can apply a distance transform.
        dilated
            .data
            .iter()
            .map(|&fg| if fg { 0.0f32 } else { 1.0f32 })
            .collect()
    }

    /// Merge two masks with OR (union of foreground regions).
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let n = self.data.len().min(other.data.len());
        let mut data = self.data.clone();
        for i in 0..n {
            data[i] = data[i] || other.data[i];
        }
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Merge two masks with AND (intersection of foreground regions).
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let n = self.data.len().min(other.data.len());
        let mut data = self.data.clone();
        for i in 0..n {
            data[i] = data[i] && other.data[i];
        }
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// Invert the mask (swap foreground and background).
    #[must_use]
    pub fn inverted(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|&b| !b).collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Threshold-based mask builder
// ---------------------------------------------------------------------------

/// Simple frame-differencing mask: marks pixels whose absolute difference
/// between consecutive frames exceeds a fixed threshold.
///
/// This is the cheapest mask variant — O(W×H) per frame pair — and works well
/// for scenes with large, clearly defined moving objects.
#[derive(Debug, Clone)]
pub struct ThresholdMask {
    /// Pixel difference threshold (0–255).
    pub threshold: u8,
    /// Dilation radius applied after thresholding.
    pub dilation_radius: usize,
}

impl ThresholdMask {
    /// Create a threshold mask builder.
    #[must_use]
    pub const fn new(threshold: u8, dilation_radius: usize) -> Self {
        Self {
            threshold,
            dilation_radius,
        }
    }

    /// Build the mask from two consecutive frames.
    ///
    /// Pixels where `|prev[y,x] - curr[y,x]| > threshold` are marked
    /// foreground.
    #[must_use]
    pub fn build(&self, prev: &Frame, curr: &Frame) -> MotionMask {
        let w = prev.width.min(curr.width);
        let h = prev.height.min(curr.height);
        let mut mask = MotionMask::background(w, h);

        for y in 0..h {
            for x in 0..w {
                let p = prev.data[[y, x]];
                let c = curr.data[[y, x]];
                let diff = (p as i32 - c as i32).unsigned_abs() as u8;
                if diff > self.threshold {
                    mask.set(x, y, true);
                }
            }
        }

        if self.dilation_radius > 0 {
            mask.dilated(self.dilation_radius)
        } else {
            mask
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive background model
// ---------------------------------------------------------------------------

/// Per-pixel running statistics used by [`MotionMaskBuilder`].
#[derive(Debug, Clone)]
struct PixelModel {
    mean: f64,
    variance: f64,
    count: u32,
}

impl PixelModel {
    const fn new(initial: f64) -> Self {
        Self {
            mean: initial,
            variance: 100.0, // start with high uncertainty
            count: 1,
        }
    }

    /// Welford online update.
    fn update(&mut self, value: f64, learning_rate: f64) {
        self.mean = self.mean * (1.0 - learning_rate) + value * learning_rate;
        let diff = value - self.mean;
        self.variance = self.variance * (1.0 - learning_rate) + diff * diff * learning_rate;
        self.variance = self.variance.max(1.0); // floor to avoid div-by-zero
        self.count = self.count.saturating_add(1);
    }

    /// Returns true if `value` is a foreground (outlier) observation.
    fn is_foreground(&self, value: f64, sigma_threshold: f64) -> bool {
        let sigma = self.variance.sqrt();
        (value - self.mean).abs() > sigma_threshold * sigma
    }
}

/// Adaptive background-subtraction mask builder.
///
/// Maintains a per-pixel running mean and variance; foreground is declared when
/// a pixel deviates beyond `sigma_threshold` standard deviations.
pub struct MotionMaskBuilder {
    /// Number of standard deviations required to declare foreground.
    pub sigma_threshold: f64,
    /// Exponential learning rate for the background model (0 < α ≤ 1).
    pub learning_rate: f64,
    /// Dilation radius applied after adaptive thresholding.
    pub dilation_radius: usize,
    /// Only update background model with background pixels (prevents
    /// foreground objects from "eating" the background model).
    pub selective_update: bool,

    model: Option<Vec<PixelModel>>,
    model_width: usize,
}

impl MotionMaskBuilder {
    /// Create a new adaptive mask builder with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sigma_threshold: 3.0,
            learning_rate: 0.05,
            dilation_radius: 3,
            selective_update: true,
            model: None,
            model_width: 0,
        }
    }

    /// Process a frame and return the foreground mask.
    ///
    /// The first call initialises the background model and returns an
    /// all-background mask.  Subsequent calls compute the foreground mask
    /// and optionally update the model.
    #[must_use]
    pub fn process(&mut self, frame: &Frame) -> MotionMask {
        let w = frame.width;
        let h = frame.height;

        // Initialise model on first frame.
        if self.model.is_none() || self.model_width != w {
            let pixels: Vec<PixelModel> = (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| {
                    let val = if y < frame.data.dim().0 && x < frame.data.dim().1 {
                        frame.data[[y, x]] as f64
                    } else {
                        0.0
                    };
                    PixelModel::new(val)
                })
                .collect();
            self.model = Some(pixels);
            self.model_width = w;
            return MotionMask::background(w, h);
        }

        let model = self.model.as_mut().expect("model initialised above");
        let mut mask = MotionMask::background(w, h);

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if idx >= model.len() {
                    continue;
                }
                let val = if y < frame.data.dim().0 && x < frame.data.dim().1 {
                    frame.data[[y, x]] as f64
                } else {
                    0.0
                };

                let is_fg = model[idx].is_foreground(val, self.sigma_threshold);
                mask.set(x, y, is_fg);

                // Update model only for background pixels when selective_update is on.
                if !is_fg || !self.selective_update {
                    model[idx].update(val, self.learning_rate);
                }
            }
        }

        if self.dilation_radius > 0 {
            mask.dilated(self.dilation_radius)
        } else {
            mask
        }
    }

    /// Reset the background model (e.g., after a scene cut).
    pub fn reset(&mut self) {
        self.model = None;
        self.model_width = 0;
    }
}

impl Default for MotionMaskBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Integration helper: filter features by mask
// ---------------------------------------------------------------------------

/// Filter a list of feature coordinate pairs `(x, y)` using a [`MotionMask`],
/// returning only background features.
///
/// This is the primary integration point with the feature tracker: pass the
/// weight map (or raw mask) here before running motion estimation.
#[must_use]
pub fn filter_features_by_mask<'a>(
    features: &'a [(f64, f64)],
    mask: &MotionMask,
) -> Vec<&'a (f64, f64)> {
    features
        .iter()
        .filter(|(x, y)| {
            let ix = x.round() as usize;
            let iy = y.round() as usize;
            !mask.get(ix, iy)
        })
        .collect()
}

/// Build a sequence of motion masks for an entire frame sequence.
///
/// Uses the adaptive [`MotionMaskBuilder`].  The first element is always an
/// all-background mask.
#[must_use]
pub fn build_sequence(frames: &[Frame]) -> Vec<MotionMask> {
    let mut builder = MotionMaskBuilder::new();
    frames.iter().map(|f| builder.process(f)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn solid_frame(w: usize, h: usize, fill: u8) -> Frame {
        Frame::new(w, h, 0.0, Array2::from_elem((h, w), fill))
    }

    fn frame_with_blob(w: usize, h: usize, bx: usize, by: usize, val: u8) -> Frame {
        let mut data = Array2::from_elem((h, w), 64u8);
        for r in by..(by + 20).min(h) {
            for c in bx..(bx + 20).min(w) {
                data[[r, c]] = val;
            }
        }
        Frame::new(w, h, 0.0, data)
    }

    #[test]
    fn test_motion_mask_background_all_false() {
        let mask = MotionMask::background(10, 10);
        assert_eq!(mask.foreground_fraction(), 0.0);
    }

    #[test]
    fn test_motion_mask_foreground_all_true() {
        let mask = MotionMask::foreground(10, 10);
        assert!((mask.foreground_fraction() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_motion_mask_get_set() {
        let mut mask = MotionMask::background(10, 10);
        mask.set(3, 4, true);
        assert!(mask.get(3, 4));
        assert!(!mask.get(4, 3));
    }

    #[test]
    fn test_motion_mask_out_of_bounds_returns_false() {
        let mask = MotionMask::foreground(5, 5);
        assert!(!mask.get(100, 100));
    }

    #[test]
    fn test_motion_mask_dilation_expands() {
        let mut mask = MotionMask::background(20, 20);
        mask.set(10, 10, true);
        let dilated = mask.dilated(2);
        // All pixels within radius 2 should be foreground.
        assert!(dilated.get(10, 10));
        assert!(dilated.get(8, 10));
        assert!(dilated.get(12, 10));
    }

    #[test]
    fn test_motion_mask_union() {
        let mut a = MotionMask::background(5, 5);
        let mut b = MotionMask::background(5, 5);
        a.set(1, 1, true);
        b.set(3, 3, true);
        let u = a.union(&b);
        assert!(u.get(1, 1));
        assert!(u.get(3, 3));
        assert!(!u.get(0, 0));
    }

    #[test]
    fn test_motion_mask_intersection() {
        let mut a = MotionMask::background(5, 5);
        let mut b = MotionMask::background(5, 5);
        a.set(1, 1, true);
        a.set(2, 2, true);
        b.set(2, 2, true);
        b.set(3, 3, true);
        let i = a.intersection(&b);
        assert!(i.get(2, 2));
        assert!(!i.get(1, 1));
        assert!(!i.get(3, 3));
    }

    #[test]
    fn test_motion_mask_invert() {
        let mask = MotionMask::background(4, 4);
        let inv = mask.inverted();
        assert!((inv.foreground_fraction() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_motion_mask_weight_map_background_is_one() {
        let mask = MotionMask::background(10, 10);
        let weights = mask.to_weight_map(0);
        assert!(weights.iter().all(|&w| (w - 1.0f32).abs() < 1e-6));
    }

    #[test]
    fn test_threshold_mask_detects_bright_blob() {
        let prev = solid_frame(100, 100, 100);
        let curr = frame_with_blob(100, 100, 40, 40, 200); // large difference
        let builder = ThresholdMask::new(50, 0);
        let mask = builder.build(&prev, &curr);
        // Centre of blob should be foreground.
        assert!(mask.get(50, 50));
    }

    #[test]
    fn test_threshold_mask_no_motion_is_background() {
        let frame = solid_frame(100, 100, 100);
        let builder = ThresholdMask::new(20, 0);
        let mask = builder.build(&frame, &frame);
        assert!((mask.foreground_fraction()).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_builder_first_frame_is_background() {
        let mut builder = MotionMaskBuilder::new();
        let frame = solid_frame(50, 50, 128);
        let mask = builder.process(&frame);
        assert_eq!(mask.foreground_fraction(), 0.0);
    }

    #[test]
    fn test_adaptive_builder_detects_sudden_change() {
        let mut builder = MotionMaskBuilder::new();
        let bg = solid_frame(50, 50, 100);
        // Warm up model.
        for _ in 0..5 {
            builder.process(&bg);
        }
        // Now introduce a very different frame.
        let fg = solid_frame(50, 50, 250);
        let mask = builder.process(&fg);
        // Most pixels should be flagged as foreground.
        assert!(mask.foreground_fraction() > 0.5);
    }

    #[test]
    fn test_adaptive_builder_reset() {
        let mut builder = MotionMaskBuilder::new();
        let frame = solid_frame(50, 50, 128);
        builder.process(&frame);
        builder.reset();
        // After reset, next frame should produce background mask.
        let mask = builder.process(&frame);
        assert_eq!(mask.foreground_fraction(), 0.0);
    }

    #[test]
    fn test_filter_features_by_mask() {
        let mut mask = MotionMask::background(100, 100);
        mask.set(10, 10, true); // foreground
        let features = vec![(10.0f64, 10.0f64), (50.0, 50.0)];
        let filtered = filter_features_by_mask(&features, &mask);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].0 - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_build_sequence_length() {
        let frames: Vec<Frame> = (0..6)
            .map(|i| solid_frame(40, 40, (i * 30) as u8))
            .collect();
        let masks = build_sequence(&frames);
        assert_eq!(masks.len(), 6);
    }
}
