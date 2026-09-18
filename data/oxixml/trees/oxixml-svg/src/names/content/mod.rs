//! Content-model tables: which children each SVG element admits, whether an
//! element admits character data, and which attributes apply to which
//! elements.
//!
//! This module is split across two files purely for size: `children`
//! implements [`allows_child`] and [`allows_character_data`];
//! `applies` implements [`attribute_applies`]. Both are driven by the
//! specification requirements catalog's per-element content-model text
//! (catalog §1) and per-attribute "Elements"/"Applies to" columns (catalog
//! §2), combined with the category predicates [`crate::names::elements`]
//! and [`crate::names::attrs`] already expose — this module adds the
//! remaining element-specific and attribute-specific narrowing the coarse
//! predicates alone cannot express (for example that `linearGradient`
//! accepts `stop` plus only three of the seven animation elements, or that
//! `rx`/`ry` apply to `rect`/`ellipse` but not `circle`/`line`).
//!
//! Every rule here traces back to a specific catalog table or prose
//! statement; where the catalog's own §8 "open confirmation items" flags a
//! value as unconfirmed, the permissive answer is chosen and the choice is
//! commented at its point of use rather than silently guessed into a
//! stricter shape.

mod applies;
mod children;

pub use applies::attribute_applies;
pub use children::{allows_character_data, allows_child};
