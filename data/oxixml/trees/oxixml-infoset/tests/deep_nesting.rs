//! Regression: an ordinary, well-formed but deeply nested document must not
//! overflow the native stack.
//!
//! `TreeBuilder::materialize`, the `descendant` axis and the string-value walk
//! all used to recurse once per tree level, so a document nested a few thousand
//! elements deep aborted the *process* (an uncatchable stack overflow, not a
//! panic) inside `Tree::parse_str` — taking down everything layered on the XDM
//! tree with it.

use oxixml_infoset::{Tree, XdmError, XdmNode, descendants};

/// Deep enough to blow a default 8 MiB stack many times over with one frame per
/// level, but still a perfectly ordinary document.
const DEPTH: usize = 100_000;

fn deep_document(depth: usize) -> String {
    let mut xml = String::with_capacity(depth * 7 + 1);
    for _ in 0..depth {
        xml.push_str("<a>");
    }
    xml.push('x');
    for _ in 0..depth {
        xml.push_str("</a>");
    }
    xml
}

#[test]
fn a_deeply_nested_document_parses_without_overflowing_the_stack() -> Result<(), XdmError> {
    let tree = Tree::parse_str(&deep_document(DEPTH))?;
    let root = tree.root();

    // The string value walks the whole subtree.
    assert_eq!(root.string_value(), "x");

    // So does the `descendant` axis: one element per level, plus the text node.
    assert_eq!(descendants(&root).len(), DEPTH + 1);
    Ok(())
}

#[test]
fn a_shallow_document_is_unaffected() -> Result<(), XdmError> {
    let tree = Tree::parse_str(&deep_document(3))?;
    assert_eq!(tree.root().string_value(), "x");
    assert_eq!(descendants(&tree.root()).len(), 4);
    Ok(())
}
