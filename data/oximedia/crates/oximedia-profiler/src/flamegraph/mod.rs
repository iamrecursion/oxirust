//! Flame graph modules.

pub mod generate;
pub mod interactive;
pub mod svg;

pub use generate::{FlameGraphData, FlameGraphGenerator};
pub use interactive::InteractiveRenderer;
pub use svg::SvgRenderer;
