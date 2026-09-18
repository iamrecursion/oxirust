//! Structural comparison of trees.
//!
//! This implements DOM Level 3 `Node.isEqualNode` — two nodes are equal when
//! their kinds, names, values, attributes and children match — plus a diagnostic
//! variant that reports *where* two trees start to differ, which is what makes a
//! parse/serialize/re-parse round-trip test readable when it fails.

use crate::document::Document;
use crate::node::{NodeContent, NodeId};

impl Document {
    /// Returns `true` when the two documents have structurally identical trees.
    ///
    /// The XML declaration metadata ([`Document::xml_version`],
    /// [`Document::encoding`], [`Document::standalone`]) is *not* part of the
    /// tree and is therefore not compared.
    #[must_use]
    pub fn structural_eq(&self, other: &Self) -> bool {
        self.structural_diff(other).is_none()
    }

    /// Returns a human-readable description of the first structural difference
    /// between the two documents, or `None` when they are equal.
    #[must_use]
    pub fn structural_diff(&self, other: &Self) -> Option<String> {
        self.node_diff(self.root(), other, other.root())
    }

    /// Returns `true` when the two subtrees are structurally identical, i.e. DOM
    /// Level 3 `Node.isEqualNode`.
    #[must_use]
    pub fn is_equal_node(&self, a: NodeId, other: &Self, b: NodeId) -> bool {
        self.node_diff(a, other, b).is_none()
    }

    /// Returns a human-readable description of the first structural difference
    /// between two subtrees, or `None` when they are equal.
    #[must_use]
    pub fn node_diff(&self, a: NodeId, other: &Self, b: NodeId) -> Option<String> {
        let mut stack = vec![(a, b, String::from("/"))];
        while let Some((left, right, path)) = stack.pop() {
            if let Some(difference) = self.shallow_diff(left, other, right, &path) {
                return Some(difference);
            }

            let left_children: Vec<NodeId> = self.children(left).collect();
            let right_children: Vec<NodeId> = other.children(right).collect();
            if left_children.len() != right_children.len() {
                return Some(format!(
                    "{path}: child count differs ({} vs {})",
                    left_children.len(),
                    right_children.len()
                ));
            }
            for (index, (left_child, right_child)) in left_children
                .into_iter()
                .zip(right_children)
                .enumerate()
                .rev()
            {
                let name = self.node_name(left_child).unwrap_or_else(|| "?".to_owned());
                stack.push((left_child, right_child, format!("{path}{name}[{index}]/")));
            }
        }
        None
    }

    fn shallow_diff(&self, a: NodeId, other: &Self, b: NodeId, path: &str) -> Option<String> {
        let left = self.get(a)?;
        let right = other.get(b)?;
        if left.kind() != right.kind() {
            return Some(format!(
                "{path}: node kind differs ({} vs {})",
                left.kind(),
                right.kind()
            ));
        }

        match (&left.content, &right.content) {
            (NodeContent::Element(left_data), NodeContent::Element(right_data)) => {
                if left_data.name != right_data.name || left_data.prefix != right_data.prefix {
                    return Some(format!(
                        "{path}: element name differs ({} vs {})",
                        left_data.qualified_name(),
                        right_data.qualified_name()
                    ));
                }
                self.attributes_diff(a, other, b, path)
            }
            (NodeContent::Attr(left_data), NodeContent::Attr(right_data)) => {
                if left_data.name != right_data.name
                    || left_data.prefix != right_data.prefix
                    || left_data.value != right_data.value
                {
                    return Some(format!(
                        "{path}: attribute differs ({}={:?} vs {}={:?})",
                        left_data.qualified_name(),
                        left_data.value,
                        right_data.qualified_name(),
                        right_data.value
                    ));
                }
                None
            }
            (NodeContent::Text(left_text), NodeContent::Text(right_text))
            | (NodeContent::CdataSection(left_text), NodeContent::CdataSection(right_text))
            | (NodeContent::Comment(left_text), NodeContent::Comment(right_text))
            | (NodeContent::EntityReference(left_text), NodeContent::EntityReference(right_text)) => {
                if left_text == right_text {
                    None
                } else {
                    Some(format!(
                        "{path}: character data differs ({left_text:?} vs {right_text:?})"
                    ))
                }
            }
            (
                NodeContent::ProcessingInstruction(left_data),
                NodeContent::ProcessingInstruction(right_data),
            ) => {
                if left_data == right_data {
                    None
                } else {
                    Some(format!(
                        "{path}: processing instruction differs ({:?} {:?} vs {:?} {:?})",
                        left_data.target(),
                        left_data.data(),
                        right_data.target(),
                        right_data.data()
                    ))
                }
            }
            (NodeContent::DocumentType(left_data), NodeContent::DocumentType(right_data)) => {
                if left_data == right_data {
                    None
                } else {
                    Some(format!(
                        "{path}: document type differs ({:?} vs {:?})",
                        left_data.name(),
                        right_data.name()
                    ))
                }
            }
            _ => None,
        }
    }

    fn attributes_diff(&self, a: NodeId, other: &Self, b: NodeId, path: &str) -> Option<String> {
        let mut left: Vec<(String, String, String)> = self
            .attributes(a)
            .filter_map(|id| self.attr(id))
            .map(|data| {
                (
                    data.namespace_uri().unwrap_or_default().to_owned(),
                    data.qualified_name(),
                    data.value().to_owned(),
                )
            })
            .collect();
        let mut right: Vec<(String, String, String)> = other
            .attributes(b)
            .filter_map(|id| other.attr(id))
            .map(|data| {
                (
                    data.namespace_uri().unwrap_or_default().to_owned(),
                    data.qualified_name(),
                    data.value().to_owned(),
                )
            })
            .collect();
        left.sort();
        right.sort();
        if left == right {
            return None;
        }
        Some(format!("{path}: attributes differ ({left:?} vs {right:?})"))
    }
}
