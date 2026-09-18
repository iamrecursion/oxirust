#![allow(dead_code)]
//! Render output management for virtual production pipelines.
//!
//! Provides render layer enumeration, per-output configuration,
//! and a manager that coordinates multiple simultaneous render outputs.

/// Render layer type used to classify an output's content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderLayer {
    /// Physical LED wall background plate.
    Background,
    /// Real-time foreground compositing layer.
    Foreground,
    /// Lighting estimation and injection layer.
    Lighting,
    /// Depth-buffer / Z-composite layer.
    Depth,
    /// Final composited output.
    Final,
}

impl RenderLayer {
    /// Returns true for layers that contribute to the foreground subject.
    #[must_use]
    pub fn is_foreground(&self) -> bool {
        matches!(self, RenderLayer::Foreground | RenderLayer::Final)
    }

    /// Returns true for infrastructure / support layers.
    #[must_use]
    pub fn is_support_layer(&self) -> bool {
        matches!(self, RenderLayer::Lighting | RenderLayer::Depth)
    }
}

impl std::fmt::Display for RenderLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderLayer::Background => write!(f, "Background"),
            RenderLayer::Foreground => write!(f, "Foreground"),
            RenderLayer::Lighting => write!(f, "Lighting"),
            RenderLayer::Depth => write!(f, "Depth"),
            RenderLayer::Final => write!(f, "Final"),
        }
    }
}

/// Status of a render output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderOutputStatus {
    /// Output is initializing its render pipeline.
    Initializing,
    /// Output is actively rendering frames.
    Active,
    /// Output is suspended but retains its configuration.
    Suspended,
    /// Output has been shut down.
    Shutdown,
}

impl std::fmt::Display for RenderOutputStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderOutputStatus::Initializing => write!(f, "Initializing"),
            RenderOutputStatus::Active => write!(f, "Active"),
            RenderOutputStatus::Suspended => write!(f, "Suspended"),
            RenderOutputStatus::Shutdown => write!(f, "Shutdown"),
        }
    }
}

/// Configuration for a single render output channel.
#[derive(Debug, Clone)]
pub struct RenderOutputConfig {
    /// Output pixel width.
    pub width: u32,
    /// Output pixel height.
    pub height: u32,
    /// Target frames per second.
    pub fps: f32,
    /// Whether HDR rendering is enabled.
    pub hdr_enabled: bool,
    /// Bit depth per channel.
    pub bit_depth: u8,
}

impl Default for RenderOutputConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 60.0,
            hdr_enabled: false,
            bit_depth: 8,
        }
    }
}

impl RenderOutputConfig {
    /// Total pixel count for this configuration.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn total_pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Estimated data rate in bytes per second at the given bit depth.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn data_rate_bps(&self) -> f64 {
        let pixels_per_frame = self.total_pixels();
        let bytes_per_pixel = (u64::from(self.bit_depth) * 3).div_ceil(8); // 3 channels
        pixels_per_frame as f64 * bytes_per_pixel as f64 * f64::from(self.fps)
    }
}

/// A named render output with an associated layer type, config, and runtime status.
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// Human-readable name for this output.
    pub name: String,
    /// Which layer this output renders.
    pub layer: RenderLayer,
    /// Output configuration.
    pub config: RenderOutputConfig,
    /// Current operational status.
    pub status: RenderOutputStatus,
    /// Number of frames rendered so far.
    pub frames_rendered: u64,
}

impl RenderOutput {
    /// Create a new render output in `Initializing` state.
    pub fn new(name: impl Into<String>, layer: RenderLayer, config: RenderOutputConfig) -> Self {
        Self {
            name: name.into(),
            layer,
            config,
            status: RenderOutputStatus::Initializing,
            frames_rendered: 0,
        }
    }

    /// Transition the output to `Active`.
    pub fn activate(&mut self) {
        if self.status == RenderOutputStatus::Initializing
            || self.status == RenderOutputStatus::Suspended
        {
            self.status = RenderOutputStatus::Active;
        }
    }

    /// Suspend the output.
    pub fn suspend(&mut self) {
        if self.status == RenderOutputStatus::Active {
            self.status = RenderOutputStatus::Suspended;
        }
    }

    /// Shut down the output.
    pub fn shutdown(&mut self) {
        self.status = RenderOutputStatus::Shutdown;
    }

    /// Simulate rendering a frame.  Increments the counter only when active.
    pub fn render_frame(&mut self) -> bool {
        if self.status == RenderOutputStatus::Active {
            self.frames_rendered += 1;
            return true;
        }
        false
    }

    /// Returns true if the output is actively producing frames.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.status == RenderOutputStatus::Active
    }
}

/// Manages multiple render outputs across a virtual production stage.
#[derive(Debug, Default)]
pub struct RenderOutputManager {
    outputs: Vec<RenderOutput>,
}

impl RenderOutputManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a render output.
    pub fn add_output(&mut self, output: RenderOutput) {
        self.outputs.push(output);
    }

    /// Return an immutable slice of all outputs.
    #[must_use]
    pub fn outputs(&self) -> &[RenderOutput] {
        &self.outputs
    }

    /// Find the first output with the given layer type.
    #[must_use]
    pub fn find_by_layer(&self, layer: RenderLayer) -> Option<&RenderOutput> {
        self.outputs.iter().find(|o| o.layer == layer)
    }

    /// Find the first output with the given name.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&RenderOutput> {
        self.outputs.iter().find(|o| o.name == name)
    }

    /// Find a mutable reference to an output by name.
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut RenderOutput> {
        self.outputs.iter_mut().find(|o| o.name == name)
    }

    /// Activate all outputs that are in `Initializing` state.
    pub fn activate_all(&mut self) {
        for output in &mut self.outputs {
            output.activate();
        }
    }

    /// Returns the count of currently active outputs.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.outputs.iter().filter(|o| o.is_active()).count()
    }

    /// Returns total frames rendered across all outputs.
    #[must_use]
    pub fn total_frames_rendered(&self) -> u64 {
        self.outputs.iter().map(|o| o.frames_rendered).sum()
    }

    /// Remove all outputs that have been shut down.
    pub fn remove_shutdown(&mut self) {
        self.outputs
            .retain(|o| o.status != RenderOutputStatus::Shutdown);
    }

    /// Returns the number of registered outputs.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_output(name: &str, layer: RenderLayer) -> RenderOutput {
        RenderOutput::new(name, layer, RenderOutputConfig::default())
    }

    #[test]
    fn test_render_layer_is_foreground() {
        assert!(RenderLayer::Foreground.is_foreground());
        assert!(RenderLayer::Final.is_foreground());
        assert!(!RenderLayer::Background.is_foreground());
        assert!(!RenderLayer::Lighting.is_foreground());
    }

    #[test]
    fn test_render_layer_is_support() {
        assert!(RenderLayer::Lighting.is_support_layer());
        assert!(RenderLayer::Depth.is_support_layer());
        assert!(!RenderLayer::Final.is_support_layer());
    }

    #[test]
    fn test_render_layer_display() {
        assert_eq!(RenderLayer::Background.to_string(), "Background");
        assert_eq!(RenderLayer::Final.to_string(), "Final");
        assert_eq!(RenderLayer::Depth.to_string(), "Depth");
    }

    #[test]
    fn test_render_output_status_display() {
        assert_eq!(RenderOutputStatus::Active.to_string(), "Active");
        assert_eq!(RenderOutputStatus::Suspended.to_string(), "Suspended");
        assert_eq!(RenderOutputStatus::Shutdown.to_string(), "Shutdown");
    }

    #[test]
    fn test_output_config_total_pixels() {
        let cfg = RenderOutputConfig {
            width: 1920,
            height: 1080,
            ..Default::default()
        };
        assert_eq!(cfg.total_pixels(), 1920 * 1080);
    }

    #[test]
    fn test_output_config_data_rate() {
        let cfg = RenderOutputConfig::default(); // 1920x1080, 60fps, 8-bit
        let rate = cfg.data_rate_bps();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_render_output_activate() {
        let mut o = make_output("bg", RenderLayer::Background);
        assert_eq!(o.status, RenderOutputStatus::Initializing);
        o.activate();
        assert_eq!(o.status, RenderOutputStatus::Active);
        assert!(o.is_active());
    }

    #[test]
    fn test_render_output_suspend_resume() {
        let mut o = make_output("fg", RenderLayer::Foreground);
        o.activate();
        o.suspend();
        assert_eq!(o.status, RenderOutputStatus::Suspended);
        o.activate(); // resume from suspended
        assert_eq!(o.status, RenderOutputStatus::Active);
    }

    #[test]
    fn test_render_output_shutdown() {
        let mut o = make_output("final", RenderLayer::Final);
        o.activate();
        o.shutdown();
        assert_eq!(o.status, RenderOutputStatus::Shutdown);
        assert!(!o.is_active());
    }

    #[test]
    fn test_render_frame_increments_only_when_active() {
        let mut o = make_output("depth", RenderLayer::Depth);
        assert!(!o.render_frame()); // not active yet
        assert_eq!(o.frames_rendered, 0);
        o.activate();
        assert!(o.render_frame());
        assert!(o.render_frame());
        assert_eq!(o.frames_rendered, 2);
    }

    #[test]
    fn test_manager_activate_all() {
        let mut mgr = RenderOutputManager::new();
        mgr.add_output(make_output("bg", RenderLayer::Background));
        mgr.add_output(make_output("fg", RenderLayer::Foreground));
        mgr.activate_all();
        assert_eq!(mgr.active_count(), 2);
    }

    #[test]
    fn test_manager_find_by_layer() {
        let mut mgr = RenderOutputManager::new();
        mgr.add_output(make_output("lighting", RenderLayer::Lighting));
        assert!(mgr.find_by_layer(RenderLayer::Lighting).is_some());
        assert!(mgr.find_by_layer(RenderLayer::Final).is_none());
    }

    #[test]
    fn test_manager_find_by_name_mut() {
        let mut mgr = RenderOutputManager::new();
        mgr.add_output(make_output("myout", RenderLayer::Final));
        let out = mgr
            .find_by_name_mut("myout")
            .expect("should succeed in test");
        out.activate();
        assert!(mgr
            .find_by_name("myout")
            .expect("should succeed in test")
            .is_active());
    }

    #[test]
    fn test_manager_total_frames_rendered() {
        let mut mgr = RenderOutputManager::new();
        mgr.add_output(make_output("a", RenderLayer::Background));
        mgr.add_output(make_output("b", RenderLayer::Foreground));
        mgr.activate_all();
        {
            let a = mgr.find_by_name_mut("a").expect("should succeed in test");
            a.render_frame();
            a.render_frame();
        }
        {
            let b = mgr.find_by_name_mut("b").expect("should succeed in test");
            b.render_frame();
        }
        assert_eq!(mgr.total_frames_rendered(), 3);
    }

    #[test]
    fn test_manager_remove_shutdown() {
        let mut mgr = RenderOutputManager::new();
        mgr.add_output(make_output("keep", RenderLayer::Background));
        mgr.add_output(make_output("drop", RenderLayer::Depth));
        {
            let drop = mgr
                .find_by_name_mut("drop")
                .expect("should succeed in test");
            drop.shutdown();
        }
        mgr.remove_shutdown();
        assert_eq!(mgr.output_count(), 1);
        assert!(mgr.find_by_name("keep").is_some());
    }

    #[test]
    fn test_manager_output_count() {
        let mut mgr = RenderOutputManager::new();
        assert_eq!(mgr.output_count(), 0);
        mgr.add_output(make_output("x", RenderLayer::Final));
        assert_eq!(mgr.output_count(), 1);
    }
}
