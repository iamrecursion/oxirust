//! Constants and small name-matching helpers for the XInclude namespace.

use oxixml_dom::{Document, NodeId, NodeKind};

/// The namespace name of the XInclude vocabulary, fixed by
/// [XInclude 1.0 §3.1](https://www.w3.org/TR/xinclude/#include_element).
pub const XINCLUDE_NAMESPACE: &str = "http://www.w3.org/2001/XInclude";

/// The local name of the `xi:include` element.
pub(crate) const INCLUDE_LOCAL_NAME: &str = "include";

/// The local name of the `xi:fallback` element.
pub(crate) const FALLBACK_LOCAL_NAME: &str = "fallback";

/// Returns the local name of `id` when it is an element in the [`XINCLUDE_NAMESPACE`], or
/// `None` when `id` is not an element or is an element in some other (or no) namespace.
pub(crate) fn xinclude_local_name(doc: &Document, id: NodeId) -> Option<&str> {
    if doc.kind(id) != Some(NodeKind::Element) {
        return None;
    }
    let element = doc.element(id)?;
    if element.namespace_uri() == Some(XINCLUDE_NAMESPACE) {
        Some(element.local_name())
    } else {
        None
    }
}
