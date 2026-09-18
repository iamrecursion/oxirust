//! Two-dimensional geometry: matrices, curves, bounding boxes, and viewports.

pub mod arc;
pub mod bezier;
pub mod matrix;
pub mod path_ops;
pub mod primitives;
pub mod viewport;

// Convenience re-exports of the core geometry types. The free functions that
// operate on a whole [`crate::values::path::PathData`]
// ([`path_ops::bounding_box`], [`path_ops::total_length`], ...) and the
// `viewBox`-to-viewport transforms stay reachable through their own
// submodule, since a bare `bounding_box`/`normalize` at this level would
// read as if it belonged to `Matrix`/`Rect` rather than to a path.
pub use arc::{ArcOutcome, ArcParams, CenterParams};
pub use bezier::{CubicBezier, QuadBezier};
pub use matrix::Matrix;
pub use primitives::{Point, Rect};
