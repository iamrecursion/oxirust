//! The in-memory reference tree: [`Tree`], its [`TreeBuilder`], and the XML front end.
//!
//! This module is what makes the rest of the crate usable on its own. Everything a query engine
//! needs is expressed against the [`XdmStore`](crate::XdmStore) trait, so a DOM, a database
//! cursor, or a streaming window can be plugged in instead; [`Tree`] is simply the
//! implementation that ships here, and the one the crate's own tests and examples navigate.

mod builder;
mod dtd;
mod entity;
pub mod external;
mod parse;
mod position;
mod store;
mod tree_impl;

pub use builder::{Annotation, TreeBuilder};
pub use dtd::{Notation, UnparsedEntity};
pub use external::{
    DEFAULT_MAX_ENTITY_BYTES, EXTERNAL_SUBSET_NAME, ExternalEntity, ExternalEntityError,
    ExternalEntityResolver, FileEntityResolver,
};
pub use parse::ParseOptions;
pub use position::SourcePosition;
pub use store::next_tree_id;
pub use tree_impl::Tree;
