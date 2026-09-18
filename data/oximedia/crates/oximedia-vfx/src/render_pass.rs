#![allow(dead_code)]

//! Render pass sequencing for multi-pass VFX compositing.
//!
//! Defines an ordered chain of render passes that are applied one after
//! another, each reading from the previous output. Useful for building
//! complex post-processing stacks (e.g. blur -> grade -> grain -> output).

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a render pass within a chain.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PassId(String);

impl PassId {
    /// Create a new pass identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the inner string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PassId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Blend mode used when compositing a pass output onto the running buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Replace the buffer with the pass output.
    Replace,
    /// Alpha-over composite.
    Over,
    /// Additive blend.
    Add,
    /// Multiply blend.
    Multiply,
    /// Screen blend.
    Screen,
}

impl BlendMode {
    /// Blend two scalar values (0.0 – 1.0) according to the mode.
    #[allow(clippy::cast_precision_loss)]
    pub fn blend(self, base: f64, layer: f64) -> f64 {
        match self {
            Self::Replace => layer,
            Self::Over => layer + base * (1.0 - layer),
            Self::Add => (base + layer).min(1.0),
            Self::Multiply => base * layer,
            Self::Screen => 1.0 - (1.0 - base) * (1.0 - layer),
        }
    }

    /// Return a human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Replace => "Replace",
            Self::Over => "Over",
            Self::Add => "Add",
            Self::Multiply => "Multiply",
            Self::Screen => "Screen",
        }
    }
}

/// Resolution mode for a render pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResolutionMode {
    /// Same as the input.
    Same,
    /// Fixed dimensions.
    Fixed {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },
    /// Scaled relative to input (e.g. 0.5 = half).
    Scaled(f64),
}

impl ResolutionMode {
    /// Compute the output dimensions given input (w, h).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn resolve(&self, input_w: u32, input_h: u32) -> (u32, u32) {
        match self {
            Self::Same => (input_w, input_h),
            Self::Fixed { width, height } => (*width, *height),
            Self::Scaled(factor) => {
                let w = ((input_w as f64) * factor).round().max(1.0) as u32;
                let h = ((input_h as f64) * factor).round().max(1.0) as u32;
                (w, h)
            }
        }
    }
}

/// A single render pass descriptor.
#[derive(Debug, Clone)]
pub struct RenderPass {
    /// Unique identifier.
    pub id: PassId,
    /// Human-readable name.
    pub name: String,
    /// Whether the pass is currently enabled.
    pub enabled: bool,
    /// Blend mode for compositing the result.
    pub blend_mode: BlendMode,
    /// Opacity (0.0 – 1.0).
    pub opacity: f64,
    /// Resolution mode for this pass.
    pub resolution: ResolutionMode,
    /// Arbitrary string parameters passed to the effect.
    pub parameters: HashMap<String, String>,
}

impl RenderPass {
    /// Create a new enabled pass with default settings.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let id_str: String = id.into();
        Self {
            id: PassId::new(id_str),
            name: name.into(),
            enabled: true,
            blend_mode: BlendMode::Replace,
            opacity: 1.0,
            resolution: ResolutionMode::Same,
            parameters: HashMap::new(),
        }
    }

    /// Set blend mode.
    pub fn with_blend(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Set opacity.
    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Set resolution mode.
    pub fn with_resolution(mut self, res: ResolutionMode) -> Self {
        self.resolution = res;
        self
    }

    /// Add a parameter.
    pub fn with_param(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), val.into());
        self
    }

    /// Enable or disable the pass.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Effective opacity considering enabled state.
    pub fn effective_opacity(&self) -> f64 {
        if self.enabled {
            self.opacity
        } else {
            0.0
        }
    }
}

/// An ordered chain of render passes.
#[derive(Debug, Default)]
pub struct RenderChain {
    passes: Vec<RenderPass>,
}

impl RenderChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Append a pass to the end of the chain.
    pub fn push(&mut self, pass: RenderPass) {
        self.passes.push(pass);
    }

    /// Insert a pass at a given index (clamped to bounds).
    pub fn insert(&mut self, index: usize, pass: RenderPass) {
        let idx = index.min(self.passes.len());
        self.passes.insert(idx, pass);
    }

    /// Remove the pass at the given index. Returns it if in bounds.
    pub fn remove(&mut self, index: usize) -> Option<RenderPass> {
        if index < self.passes.len() {
            Some(self.passes.remove(index))
        } else {
            None
        }
    }

    /// Remove a pass by id. Returns `true` if found.
    pub fn remove_by_id(&mut self, id: &str) -> bool {
        let before = self.passes.len();
        self.passes.retain(|p| p.id.as_str() != id);
        self.passes.len() < before
    }

    /// Swap two passes by index. Returns `false` if either is out of range.
    pub fn swap(&mut self, a: usize, b: usize) -> bool {
        if a < self.passes.len() && b < self.passes.len() {
            self.passes.swap(a, b);
            true
        } else {
            false
        }
    }

    /// Number of passes.
    pub fn len(&self) -> usize {
        self.passes.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    /// Get a pass by index.
    pub fn get(&self, index: usize) -> Option<&RenderPass> {
        self.passes.get(index)
    }

    /// Get a mutable pass by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut RenderPass> {
        self.passes.get_mut(index)
    }

    /// Find a pass by id.
    pub fn find(&self, id: &str) -> Option<&RenderPass> {
        self.passes.iter().find(|p| p.id.as_str() == id)
    }

    /// Return how many passes are currently enabled.
    pub fn enabled_count(&self) -> usize {
        self.passes.iter().filter(|p| p.enabled).count()
    }

    /// Disable all passes.
    pub fn disable_all(&mut self) {
        for p in &mut self.passes {
            p.enabled = false;
        }
    }

    /// Enable all passes.
    pub fn enable_all(&mut self) {
        for p in &mut self.passes {
            p.enabled = true;
        }
    }

    /// Iterate over enabled passes in order.
    pub fn enabled_passes(&self) -> Vec<&RenderPass> {
        self.passes.iter().filter(|p| p.enabled).collect()
    }

    /// Return all pass ids in chain order.
    pub fn pass_ids(&self) -> Vec<String> {
        self.passes
            .iter()
            .map(|p| p.id.as_str().to_owned())
            .collect()
    }

    /// Apply a blend chain on a single scalar pixel value (for testing / preview).
    pub fn blend_scalar(&self, base: f64) -> f64 {
        let mut result = base;
        for pass in &self.passes {
            if !pass.enabled {
                continue;
            }
            let layer = pass.blend_mode.blend(result, pass.opacity);
            result = layer;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_id_display() {
        let id = PassId::new("blur_pass");
        assert_eq!(id.to_string(), "blur_pass");
    }

    #[test]
    fn test_blend_mode_replace() {
        assert!((BlendMode::Replace.blend(0.3, 0.7) - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_blend_mode_add_clamp() {
        let r = BlendMode::Add.blend(0.8, 0.5);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_blend_mode_multiply() {
        let r = BlendMode::Multiply.blend(0.5, 0.5);
        assert!((r - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_blend_mode_screen() {
        let r = BlendMode::Screen.blend(0.5, 0.5);
        assert!((r - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_blend_mode_label() {
        assert_eq!(BlendMode::Over.label(), "Over");
        assert_eq!(BlendMode::Screen.label(), "Screen");
    }

    #[test]
    fn test_resolution_mode_same() {
        assert_eq!(ResolutionMode::Same.resolve(1920, 1080), (1920, 1080));
    }

    #[test]
    fn test_resolution_mode_fixed() {
        let m = ResolutionMode::Fixed {
            width: 640,
            height: 480,
        };
        assert_eq!(m.resolve(1920, 1080), (640, 480));
    }

    #[test]
    fn test_resolution_mode_scaled() {
        let (w, h) = ResolutionMode::Scaled(0.5).resolve(1920, 1080);
        assert_eq!(w, 960);
        assert_eq!(h, 540);
    }

    #[test]
    fn test_render_pass_defaults() {
        let p = RenderPass::new("p1", "Pass 1");
        assert!(p.enabled);
        assert_eq!(p.blend_mode, BlendMode::Replace);
        assert!((p.opacity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_render_pass_effective_opacity() {
        let mut p = RenderPass::new("p", "P").with_opacity(0.8);
        assert!((p.effective_opacity() - 0.8).abs() < 1e-9);
        p.set_enabled(false);
        assert!((p.effective_opacity()).abs() < 1e-9);
    }

    #[test]
    fn test_chain_push_len() {
        let mut chain = RenderChain::new();
        chain.push(RenderPass::new("a", "A"));
        chain.push(RenderPass::new("b", "B"));
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn test_chain_insert() {
        let mut chain = RenderChain::new();
        chain.push(RenderPass::new("a", "A"));
        chain.push(RenderPass::new("c", "C"));
        chain.insert(1, RenderPass::new("b", "B"));
        assert_eq!(
            chain.get(1).expect("should succeed in test").id.as_str(),
            "b"
        );
    }

    #[test]
    fn test_chain_remove_by_id() {
        let mut chain = RenderChain::new();
        chain.push(RenderPass::new("x", "X"));
        assert!(chain.remove_by_id("x"));
        assert!(chain.is_empty());
    }

    #[test]
    fn test_chain_swap() {
        let mut chain = RenderChain::new();
        chain.push(RenderPass::new("first", "First"));
        chain.push(RenderPass::new("second", "Second"));
        assert!(chain.swap(0, 1));
        assert_eq!(
            chain.get(0).expect("should succeed in test").id.as_str(),
            "second"
        );
    }

    #[test]
    fn test_chain_enabled_count() {
        let mut chain = RenderChain::new();
        chain.push(RenderPass::new("a", "A"));
        let mut disabled = RenderPass::new("b", "B");
        disabled.set_enabled(false);
        chain.push(disabled);
        assert_eq!(chain.enabled_count(), 1);
    }

    #[test]
    fn test_chain_disable_enable_all() {
        let mut chain = RenderChain::new();
        chain.push(RenderPass::new("a", "A"));
        chain.push(RenderPass::new("b", "B"));
        chain.disable_all();
        assert_eq!(chain.enabled_count(), 0);
        chain.enable_all();
        assert_eq!(chain.enabled_count(), 2);
    }

    #[test]
    fn test_chain_pass_ids() {
        let mut chain = RenderChain::new();
        chain.push(RenderPass::new("p1", "P1"));
        chain.push(RenderPass::new("p2", "P2"));
        let ids = chain.pass_ids();
        assert_eq!(ids, vec!["p1", "p2"]);
    }

    #[test]
    fn test_chain_blend_scalar() {
        let mut chain = RenderChain::new();
        chain.push(
            RenderPass::new("a", "A")
                .with_blend(BlendMode::Replace)
                .with_opacity(0.5),
        );
        let result = chain.blend_scalar(1.0);
        // Replace mode: result = 0.5
        assert!((result - 0.5).abs() < 1e-9);
    }
}
