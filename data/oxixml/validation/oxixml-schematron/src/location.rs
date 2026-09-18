//! [`location_path`] — the SVRL-style absolute path a firing check's `location` attribute
//! carries.

use oxixml_infoset::{NodeKind, NodeRef, XdmNode};
use oxixml_qname::ExpandedName;

/// The absolute, indexed path from the document root down to `node`, in the abbreviated form
/// [SVRL](https://schematron.com/document/321.html) reports use for `@location`: each step is
/// `name[position]` among like-named, like-kind siblings, e.g. `/catalog[1]/book[3]/@id`.
///
/// This is diagnostic text, not a re-parseable XPath expression — a namespace step is rendered
/// with a prefix drawn from the node's own in-scope bindings when one resolves the URI, and
/// falls back to a `Q{uri}local` clark-name form otherwise.
#[must_use]
pub fn location_path(node: &NodeRef) -> String {
    let mut segments = Vec::new();
    let mut current = node.clone();
    loop {
        match current.node_kind() {
            NodeKind::Document => break,
            NodeKind::Attribute => {
                let scope = current.parent().unwrap_or_else(|| current.clone());
                let name = current
                    .node_name()
                    .map(|name| prefixed_name(&scope, &name))
                    .unwrap_or_default();
                segments.push(format!("@{name}"));
                match current.parent() {
                    Some(parent) => current = parent,
                    None => break,
                }
                continue;
            }
            NodeKind::Namespace => {
                segments.push("namespace::*".to_owned());
                match current.parent() {
                    Some(parent) => current = parent,
                    None => break,
                }
                continue;
            }
            NodeKind::Element => {
                let name = current
                    .node_name()
                    .map(|name| prefixed_name(&current, &name))
                    .unwrap_or_default();
                segments.push(format!("{name}[{}]", same_name_index(&current)));
            }
            NodeKind::Text => segments.push(format!("text()[{}]", same_kind_index(&current))),
            NodeKind::Comment => segments.push(format!("comment()[{}]", same_kind_index(&current))),
            NodeKind::ProcessingInstruction => {
                segments.push(format!(
                    "processing-instruction()[{}]",
                    same_kind_index(&current)
                ));
            }
        }
        match current.parent() {
            Some(parent) => current = parent,
            None => break,
        }
    }
    segments.reverse();
    format!("/{}", segments.join("/"))
}

/// `node`'s 1-based position among its parent's element children sharing its expanded name.
fn same_name_index(node: &NodeRef) -> usize {
    let Some(parent) = node.parent() else {
        return 1;
    };
    let name = node.node_name();
    let mut count = 0usize;
    for sibling in parent.children() {
        if sibling.node_kind() != NodeKind::Element || sibling.node_name() != name {
            continue;
        }
        count += 1;
        if &sibling == node {
            return count;
        }
    }
    count.max(1)
}

/// `node`'s 1-based position among its parent's children of the same node kind (used for the
/// kinds that have no name: text, comment, processing instruction).
fn same_kind_index(node: &NodeRef) -> usize {
    let Some(parent) = node.parent() else {
        return 1;
    };
    let kind = node.node_kind();
    let mut count = 0usize;
    for sibling in parent.children() {
        if sibling.node_kind() != kind {
            continue;
        }
        count += 1;
        if &sibling == node {
            return count;
        }
    }
    count.max(1)
}

/// Renders `name` as it should appear textually, resolving its namespace to a prefix bound
/// in-scope at `scope` when possible.
pub(crate) fn prefixed_name(scope: &NodeRef, name: &ExpandedName) -> String {
    let Some(uri) = name.namespace() else {
        return name.local_name().to_owned();
    };
    let prefix = scope
        .namespaces()
        .into_iter()
        .find(|ns| ns.string_value() == uri)
        .and_then(|ns| ns.node_name())
        .map(|ns| ns.local_name().to_owned());
    match prefix {
        Some(prefix) if !prefix.is_empty() => format!("{prefix}:{}", name.local_name()),
        _ => format!("Q{{{uri}}}{}", name.local_name()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxixml_infoset::Tree;

    #[test]
    fn indexes_like_named_siblings() -> Result<(), oxixml_infoset::XdmError> {
        let tree = Tree::parse_str("<catalog><book/><book/><cd/></catalog>")?;
        let catalog = tree.document_element().expect("a document element");
        let children = catalog.children();
        assert_eq!(location_path(&children[0]), "/catalog[1]/book[1]");
        assert_eq!(location_path(&children[1]), "/catalog[1]/book[2]");
        assert_eq!(location_path(&children[2]), "/catalog[1]/cd[1]");
        Ok(())
    }

    #[test]
    fn indexes_attributes_and_resolves_prefixes() -> Result<(), oxixml_infoset::XdmError> {
        let tree = Tree::parse_str(r#"<a xmlns:x="urn:x" x:k="v"/>"#)?;
        let element = tree.document_element().expect("a document element");
        let attribute = element
            .attributes()
            .into_iter()
            .next()
            .expect("one attribute");
        assert_eq!(location_path(&attribute), "/a[1]/@x:k");
        Ok(())
    }
}
