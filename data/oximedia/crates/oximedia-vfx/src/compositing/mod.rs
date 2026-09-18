//! Compositing operations for layering and blending.

pub mod alpha;
pub mod blend;
pub mod blend_modes;
pub mod compositor;
pub mod layer;
pub mod layer_manager;
pub mod matte;

pub use alpha::{composite_atop, composite_over, composite_under, AlphaMode};
pub use blend::{blend_pixels, BlendMode};
pub use blend_modes::BlendMode as ExtBlendMode;
pub use compositor::Compositor;
pub use layer::{Layer, LayerStack, Transform};
pub use layer_manager::{LayerBlendMode, ManagedLayer, ManagedLayerStack};
pub use matte::{apply_matte, MatteType};
