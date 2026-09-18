//! Zoom optimization and dynamic effects.
//!
//! Calculates optimal zoom levels to minimize black borders and supports
//! dynamic zoom effects like Ken Burns.

pub mod calculate;
pub mod dynamic;

pub use calculate::ZoomOptimizer;
pub use dynamic::DynamicZoom;
