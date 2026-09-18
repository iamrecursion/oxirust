//! The typed SVG document model over an `oxixml-dom` tree.

pub mod cascade;
pub mod document;
pub mod refs;
pub mod shapes;
pub mod typed;

// Convenience re-exports of the main typed-model types. Reference-following
// error handling ([`refs::RefError`], [`refs::HrefChain`],
// [`refs::follow_href_chain`], [`refs::resolve_use`]) and the shape-to-path
// helpers ([`shapes::equivalent_path`], [`shapes::bounding_box`]) stay
// reachable through their own submodule — the latter, in particular, shares
// its name with [`crate::geometry::path_ops::bounding_box`], so re-exporting
// both at this level would shadow one of them.
pub use cascade::{StyleOrigin, StyleResolver};
pub use document::{SvgDocument, SvgOptions};
pub use refs::IdIndex;
pub use typed::SvgElement;
