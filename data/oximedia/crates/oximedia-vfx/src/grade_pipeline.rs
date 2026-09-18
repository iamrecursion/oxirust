//! Non-destructive colour-grading pipeline for `OxiMedia` VFX.
//!
//! Chains [`ColorGradeNode`] operations into a pipeline that can be applied
//! to RGBA pixel buffers.

#![allow(dead_code)]

/// A single colour-grading operation node.
#[derive(Debug, Clone)]
pub struct ColorGradeNode {
    /// Human-readable label for this node.
    pub label: String,
    /// LUT as 256 output values per channel (R, G, B).  None means identity.
    lut_r: Option<[u8; 256]>,
    lut_g: Option<[u8; 256]>,
    lut_b: Option<[u8; 256]>,
    /// Per-channel lift/gamma/gain as (lift, gamma, gain) tuples.
    lift: [f32; 3],
    gamma: [f32; 3],
    gain: [f32; 3],
    /// Whether this node is bypassed.
    pub bypass: bool,
}

impl ColorGradeNode {
    /// Create an identity grading node.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            lut_r: None,
            lut_g: None,
            lut_b: None,
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            bypass: false,
        }
    }

    /// Load a 1-D LUT (256 entries each for R, G, B).
    pub fn apply_lut(&mut self, r: [u8; 256], g: [u8; 256], b: [u8; 256]) {
        self.lut_r = Some(r);
        self.lut_g = Some(g);
        self.lut_b = Some(b);
    }

    /// Set S-curve style curves via lift / gamma / gain per channel.
    ///
    /// `lift` shifts the blacks, `gamma` controls midtones, `gain` scales whites.
    pub fn apply_curves(&mut self, lift: [f32; 3], gamma: [f32; 3], gain: [f32; 3]) {
        self.lift = lift;
        for i in 0..3 {
            self.gamma[i] = gamma[i].max(0.001);
            self.gain[i] = gain[i].max(0.0);
        }
    }

    /// Process a single normalised float channel value [0, 1].
    #[allow(clippy::cast_precision_loss)]
    fn process_channel(&self, value: f32, ch: usize) -> f32 {
        // 1. LUT (applied on 8-bit domain)
        let mut v = value;
        if let (Some(lr), Some(lg), Some(lb)) = (&self.lut_r, &self.lut_g, &self.lut_b) {
            let lut = match ch {
                0 => lr.as_ref(),
                1 => lg.as_ref(),
                _ => lb.as_ref(),
            };
            let idx = (v * 255.0).clamp(0.0, 255.0) as usize;
            v = lut[idx] as f32 / 255.0;
        }

        // 2. Lift / Gamma / Gain
        let lifted = v + self.lift[ch] * (1.0 - v);
        let gained = lifted * self.gain[ch];
        let gamma_corrected = gained.clamp(0.0, 1.0).powf(1.0 / self.gamma[ch]);
        gamma_corrected.clamp(0.0, 1.0)
    }

    /// Apply this node to an RGBA pixel returning a modified pixel.
    #[allow(clippy::cast_precision_loss)]
    pub fn process_pixel(&self, pixel: [u8; 4]) -> [u8; 4] {
        if self.bypass {
            return pixel;
        }
        let r = self.process_channel(pixel[0] as f32 / 255.0, 0);
        let g = self.process_channel(pixel[1] as f32 / 255.0, 1);
        let b = self.process_channel(pixel[2] as f32 / 255.0, 2);
        [
            (r * 255.0).round() as u8,
            (g * 255.0).round() as u8,
            (b * 255.0).round() as u8,
            pixel[3],
        ]
    }
}

/// A sequential pipeline of [`ColorGradeNode`] operations.
#[derive(Debug, Default)]
pub struct ColorGradePipeline {
    nodes: Vec<ColorGradeNode>,
}

impl ColorGradePipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a node to the end of the pipeline.
    pub fn push_node(&mut self, node: ColorGradeNode) {
        self.nodes.push(node);
    }

    /// Number of nodes in the pipeline.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Apply the full pipeline to an RGBA pixel.
    #[must_use]
    pub fn apply_pixel(&self, mut pixel: [u8; 4]) -> [u8; 4] {
        for node in &self.nodes {
            pixel = node.process_pixel(pixel);
        }
        pixel
    }

    /// Apply the pipeline to an entire RGBA buffer in-place.
    ///
    /// `data` must be a flat `width * height * 4` byte buffer.
    pub fn apply_all(&self, data: &mut [u8]) {
        assert_eq!(data.len() % 4, 0, "buffer length must be a multiple of 4");
        for chunk in data.chunks_exact_mut(4) {
            let pixel = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let out = self.apply_pixel(pixel);
            chunk.copy_from_slice(&out);
        }
    }

    /// Borrow a node by index.
    #[must_use]
    pub fn get_node(&self, idx: usize) -> Option<&ColorGradeNode> {
        self.nodes.get(idx)
    }

    /// Mutably borrow a node by index.
    pub fn get_node_mut(&mut self, idx: usize) -> Option<&mut ColorGradeNode> {
        self.nodes.get_mut(idx)
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_lut() -> [u8; 256] {
        let mut lut = [0u8; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
        lut
    }

    #[test]
    fn test_node_identity_pixel() {
        let node = ColorGradeNode::new("identity");
        let pixel = [100u8, 150, 200, 255];
        let out = node.process_pixel(pixel);
        assert_eq!(out, pixel);
    }

    #[test]
    fn test_node_bypass() {
        let mut node = ColorGradeNode::new("bypass");
        node.apply_curves([0.2, 0.2, 0.2], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
        node.bypass = true;
        let pixel = [100u8, 150, 200, 255];
        let out = node.process_pixel(pixel);
        assert_eq!(out, pixel);
    }

    #[test]
    fn test_node_gain_brightens() {
        let mut node = ColorGradeNode::new("gain");
        node.apply_curves([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
        let pixel = [50u8, 50, 50, 255];
        let out = node.process_pixel(pixel);
        // doubling gain should brighten
        assert!(out[0] > 50);
    }

    #[test]
    fn test_node_lift_raises_blacks() {
        let mut node = ColorGradeNode::new("lift");
        node.apply_curves([0.2, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
        let pixel = [0u8, 0, 0, 255];
        let out = node.process_pixel(pixel);
        assert!(out[0] > 0);
    }

    #[test]
    fn test_node_lut_identity() {
        let mut node = ColorGradeNode::new("lut");
        let lut = identity_lut();
        node.apply_lut(lut, lut, lut);
        let pixel = [120u8, 180, 220, 255];
        let out = node.process_pixel(pixel);
        assert_eq!(out[0], pixel[0]);
        assert_eq!(out[1], pixel[1]);
        assert_eq!(out[2], pixel[2]);
    }

    #[test]
    fn test_node_lut_invert() {
        let mut node = ColorGradeNode::new("invert");
        let mut inv_lut = [0u8; 256];
        for i in 0..256usize {
            inv_lut[i] = (255 - i) as u8;
        }
        node.apply_lut(inv_lut, inv_lut, inv_lut);
        let pixel = [0u8, 0, 0, 255];
        let out = node.process_pixel(pixel);
        assert_eq!(out[0], 255);
    }

    #[test]
    fn test_pipeline_empty() {
        let pipeline = ColorGradePipeline::new();
        assert_eq!(pipeline.node_count(), 0);
    }

    #[test]
    fn test_pipeline_push_and_count() {
        let mut pipeline = ColorGradePipeline::new();
        pipeline.push_node(ColorGradeNode::new("a"));
        pipeline.push_node(ColorGradeNode::new("b"));
        assert_eq!(pipeline.node_count(), 2);
    }

    #[test]
    fn test_pipeline_apply_all_identity() {
        let mut pipeline = ColorGradePipeline::new();
        pipeline.push_node(ColorGradeNode::new("identity"));
        let mut buf = vec![128u8, 64, 32, 255, 10, 20, 30, 200];
        let original = buf.clone();
        pipeline.apply_all(&mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn test_pipeline_alpha_preserved() {
        let mut pipeline = ColorGradePipeline::new();
        let mut node = ColorGradeNode::new("gain");
        node.apply_curves([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
        pipeline.push_node(node);
        let pixel = [50u8, 50, 50, 128];
        let out = pipeline.apply_pixel(pixel);
        assert_eq!(out[3], 128); // alpha unchanged
    }

    #[test]
    fn test_pipeline_get_node() {
        let mut pipeline = ColorGradePipeline::new();
        pipeline.push_node(ColorGradeNode::new("first"));
        let node = pipeline.get_node(0);
        assert!(node.is_some());
        assert_eq!(node.expect("should succeed in test").label, "first");
    }

    #[test]
    fn test_pipeline_get_node_out_of_bounds() {
        let pipeline = ColorGradePipeline::new();
        assert!(pipeline.get_node(99).is_none());
    }
}
