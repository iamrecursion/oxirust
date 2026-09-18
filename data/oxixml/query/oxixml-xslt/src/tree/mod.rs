//! The result tree: a writable XDM store that a transformation builds its output in.
//!
//! A source document is read through [`oxixml_infoset::Tree`]; the *result* of a transformation
//! is a different animal. It can hold text at the top level, it needs namespace fixup applied as
//! it is built, and its text nodes remember whether output escaping was disabled on them. This
//! module supplies that store: [`ResultTree`] is a full
//! [`XdmStore`](oxixml_infoset::XdmStore), so a variable bound to a temporary tree is navigable
//! by XPath exactly like a source document.

mod builder;
mod store;

pub(crate) use builder::ResultBuilder;
pub(crate) use store::ResultNode;
pub use store::ResultTree;
