//! Stylization effects.

pub mod cartoon;
pub mod halftone;
pub mod mosaic;
pub mod paint;
pub mod sketch;

pub use cartoon::{Cartoon, CartoonStyle};
pub use halftone::{Halftone, HalftonePattern};
pub use mosaic::{Mosaic, MosaicMode};
pub use paint::{OilPaint, PaintStyle};
pub use sketch::{Sketch, SketchStyle};
