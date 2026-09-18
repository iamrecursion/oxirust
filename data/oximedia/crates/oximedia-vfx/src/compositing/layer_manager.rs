//! Layer management for compositing with id-based addressing.
//!
//! Provides `ManagedLayer` (with unique id, locked flag, and `z_order`),
//! `ManagedLayerStack` for id-based operations, and a richer `ExtBlendMode`
//! enum used specifically for managed layers.

use serde::{Deserialize, Serialize};

// ── ExtBlendMode ──────────────────────────────────────────────────────────────

/// Blend mode for managed layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerBlendMode {
    /// Normal alpha compositing.
    Normal,
    /// Additive blending.
    Add,
    /// Multiply blend.
    Multiply,
    /// Screen blend.
    Screen,
    /// Overlay blend.
    Overlay,
    /// Difference blend.
    Difference,
    /// Exclusion blend.
    Exclusion,
    /// Hard Light blend.
    HardLight,
    /// Soft Light blend.
    SoftLight,
}

impl Default for LayerBlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

impl LayerBlendMode {
    /// Human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Add => "Add",
            Self::Multiply => "Multiply",
            Self::Screen => "Screen",
            Self::Overlay => "Overlay",
            Self::Difference => "Difference",
            Self::Exclusion => "Exclusion",
            Self::HardLight => "Hard Light",
            Self::SoftLight => "Soft Light",
        }
    }
}

// ── ManagedLayer ──────────────────────────────────────────────────────────────

/// A compositing layer with unique id, locking, and z-ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagedLayer {
    /// Unique layer identifier.
    pub id: u64,
    /// Human-readable layer name.
    pub name: String,
    /// Blend mode for this layer.
    pub blend_mode: LayerBlendMode,
    /// Layer opacity in [0.0, 1.0].
    pub opacity: f32,
    /// Whether the layer is visible.
    pub visible: bool,
    /// Whether the layer is locked (prevents editing).
    pub locked: bool,
    /// Z-order (higher = closer to viewer).
    pub z_order: i32,
}

impl ManagedLayer {
    /// Create a new layer with default settings (fully opaque, visible, unlocked).
    #[must_use]
    pub fn new(id: u64, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            blend_mode: LayerBlendMode::Normal,
            opacity: 1.0,
            visible: true,
            locked: false,
            z_order: 0,
        }
    }

    /// Builder: set opacity.
    #[must_use]
    pub fn with_opacity(mut self, opacity: f32) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        self
    }

    /// Builder: set blend mode.
    #[must_use]
    pub const fn with_blend(mut self, mode: LayerBlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Builder: set visibility.
    #[must_use]
    pub const fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Builder: set locked state.
    #[must_use]
    pub const fn with_locked(mut self, locked: bool) -> Self {
        self.locked = locked;
        self
    }

    /// Builder: set z-order.
    #[must_use]
    pub const fn with_z_order(mut self, z_order: i32) -> Self {
        self.z_order = z_order;
        self
    }

    /// Effective opacity: 0.0 if hidden, otherwise the opacity value.
    #[must_use]
    pub fn effective_opacity(&self) -> f32 {
        if self.visible {
            self.opacity
        } else {
            0.0
        }
    }
}

// ── ManagedLayerStack ─────────────────────────────────────────────────────────

/// A stack of `ManagedLayer` instances with id-based addressing.
#[derive(Debug, Clone, Default)]
pub struct ManagedLayerStack {
    layers: Vec<ManagedLayer>,
}

impl ManagedLayerStack {
    /// Create a new empty layer stack.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a layer.  The stack does **not** enforce unique ids.
    pub fn add(&mut self, layer: ManagedLayer) {
        self.layers.push(layer);
        self.sort_by_z();
    }

    /// Remove a layer by id.  Returns `true` if a layer was removed.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.layers.len();
        self.layers.retain(|l| l.id != id);
        self.layers.len() < before
    }

    /// Move a layer to a new z-order and re-sort the stack.
    pub fn reorder(&mut self, id: u64, new_z: i32) {
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == id) {
            layer.z_order = new_z;
        }
        self.sort_by_z();
    }

    /// Return all visible layers in z-order (bottom → top).
    #[must_use]
    pub fn visible_layers(&self) -> Vec<&ManagedLayer> {
        self.layers.iter().filter(|l| l.visible).collect()
    }

    /// Find a layer by id.
    #[must_use]
    pub fn find(&self, id: u64) -> Option<&ManagedLayer> {
        self.layers.iter().find(|l| l.id == id)
    }

    /// Find a layer by id (mutable).
    pub fn find_mut(&mut self, id: u64) -> Option<&mut ManagedLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }

    /// Total number of layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the stack has no layers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Iterate all layers in z-order.
    pub fn iter(&self) -> impl Iterator<Item = &ManagedLayer> {
        self.layers.iter()
    }

    // Internal: keep the vec sorted by z_order (stable).
    fn sort_by_z(&mut self) {
        self.layers.sort_by_key(|l| l.z_order);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_new_defaults() {
        let layer = ManagedLayer::new(1, "Background");
        assert_eq!(layer.id, 1);
        assert_eq!(layer.name, "Background");
        assert!((layer.opacity - 1.0).abs() < f32::EPSILON);
        assert!(layer.visible);
        assert!(!layer.locked);
        assert_eq!(layer.z_order, 0);
        assert_eq!(layer.blend_mode, LayerBlendMode::Normal);
    }

    #[test]
    fn test_layer_with_opacity() {
        let layer = ManagedLayer::new(2, "Mid").with_opacity(0.5);
        assert!((layer.opacity - 0.5).abs() < 1e-6, "opacity should be 0.5");
    }

    #[test]
    fn test_layer_opacity_clamped() {
        let l1 = ManagedLayer::new(3, "a").with_opacity(2.0);
        assert!((l1.opacity - 1.0).abs() < 1e-6, "opacity clamped to 1.0");
        let l2 = ManagedLayer::new(4, "b").with_opacity(-0.5);
        assert!((l2.opacity - 0.0).abs() < 1e-6, "opacity clamped to 0.0");
    }

    #[test]
    fn test_layer_effective_opacity_hidden() {
        let layer = ManagedLayer::new(5, "Hidden")
            .with_opacity(0.8)
            .with_visible(false);
        assert!(
            (layer.effective_opacity() - 0.0).abs() < 1e-6,
            "hidden layer effective opacity should be 0"
        );
    }

    #[test]
    fn test_layer_effective_opacity_visible() {
        let layer = ManagedLayer::new(6, "Visible").with_opacity(0.6);
        assert!(
            (layer.effective_opacity() - 0.6).abs() < 1e-6,
            "visible layer effective opacity should equal opacity"
        );
    }

    #[test]
    fn test_layer_with_blend() {
        let layer = ManagedLayer::new(7, "Multiply").with_blend(LayerBlendMode::Multiply);
        assert_eq!(layer.blend_mode, LayerBlendMode::Multiply);
    }

    #[test]
    fn test_stack_add_and_find() {
        let mut stack = ManagedLayerStack::new();
        stack.add(ManagedLayer::new(10, "Layer A").with_z_order(2));
        stack.add(ManagedLayer::new(11, "Layer B").with_z_order(1));
        assert_eq!(stack.len(), 2);
        assert!(stack.find(10).is_some(), "should find layer 10");
        assert!(stack.find(99).is_none(), "should not find layer 99");
    }

    #[test]
    fn test_stack_remove() {
        let mut stack = ManagedLayerStack::new();
        stack.add(ManagedLayer::new(20, "A"));
        stack.add(ManagedLayer::new(21, "B"));
        let removed = stack.remove(20);
        assert!(removed, "remove should return true");
        assert_eq!(stack.len(), 1);
        assert!(stack.find(20).is_none(), "removed layer should be gone");
    }

    #[test]
    fn test_stack_remove_nonexistent() {
        let mut stack = ManagedLayerStack::new();
        stack.add(ManagedLayer::new(30, "A"));
        let removed = stack.remove(999);
        assert!(!removed, "removing nonexistent id should return false");
        assert_eq!(stack.len(), 1, "stack size unchanged");
    }

    #[test]
    fn test_stack_reorder() {
        let mut stack = ManagedLayerStack::new();
        stack.add(ManagedLayer::new(40, "Low").with_z_order(0));
        stack.add(ManagedLayer::new(41, "High").with_z_order(1));
        // Promote "Low" to z=10 (should now be above "High")
        stack.reorder(40, 10);
        let ids: Vec<u64> = stack.iter().map(|l| l.id).collect();
        assert_eq!(ids[1], 40, "layer 40 should be on top after reorder");
    }

    #[test]
    fn test_stack_visible_layers() {
        let mut stack = ManagedLayerStack::new();
        stack.add(ManagedLayer::new(50, "Visible").with_visible(true));
        stack.add(ManagedLayer::new(51, "Hidden").with_visible(false));
        stack.add(ManagedLayer::new(52, "Visible2").with_visible(true));
        let visible = stack.visible_layers();
        assert_eq!(visible.len(), 2, "should have 2 visible layers");
        assert!(
            visible.iter().all(|l| l.visible),
            "all returned layers should be visible"
        );
    }

    #[test]
    fn test_stack_z_order_sorted_after_add() {
        let mut stack = ManagedLayerStack::new();
        stack.add(ManagedLayer::new(60, "Z3").with_z_order(3));
        stack.add(ManagedLayer::new(61, "Z1").with_z_order(1));
        stack.add(ManagedLayer::new(62, "Z2").with_z_order(2));
        let orders: Vec<i32> = stack.iter().map(|l| l.z_order).collect();
        assert_eq!(orders, vec![1, 2, 3], "layers should be sorted by z_order");
    }

    #[test]
    fn test_blend_mode_names() {
        assert_eq!(LayerBlendMode::Normal.name(), "Normal");
        assert_eq!(LayerBlendMode::Multiply.name(), "Multiply");
        assert_eq!(LayerBlendMode::Screen.name(), "Screen");
    }
}
