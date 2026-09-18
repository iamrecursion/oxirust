//! Programmatic construction and serialization of SVG documents.

pub mod builder;
pub mod fmtnum;
pub mod paint_defs;
pub mod pathbuilder;
pub mod shapes;

// Convenience re-exports of the main writer types. `shapes` only adds
// methods to [`builder::SvgBuilder`] itself (no new types of its own), and
// the number-formatting free functions in [`fmtnum`] stay reachable through
// that submodule.
pub use builder::{BuildFlaw, SvgBuilder};
pub use paint_defs::{CoordinateUnits, GradientStop};
pub use pathbuilder::PathBuilder;
