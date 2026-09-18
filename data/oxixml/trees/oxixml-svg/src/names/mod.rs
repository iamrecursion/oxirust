//! The SVG vocabulary: element and attribute catalogs and content models.

pub mod attrs;
pub mod content;
pub mod elements;

// Convenience re-exports of the main vocabulary types and content-model
// predicates.
pub use attrs::AttrId;
pub use content::{allows_character_data, allows_child, attribute_applies};
pub use elements::{ElementId, ElementStatus};
