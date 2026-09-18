//! Shared helpers for the conformance harness: suite discovery, tree navigation and the
//! canonical form results are compared in.

use oxixml_infoset::{NodeKind, NodeRef, Tree, XdmNode};
use std::path::PathBuf;

/// How one test case ended.
#[derive(Debug)]
pub enum Outcome {
    /// The assertions held.
    Passed,
    /// The case exercises something outside this processor's scope.
    Skipped,
    /// The assertions did not hold, with a description of the difference.
    Failed(String),
}

/// Running counts for a suite.
#[derive(Debug, Default)]
pub struct Tally {
    /// Cases whose assertions held.
    pub passed: usize,
    /// Cases whose assertions did not hold.
    pub failed: usize,
    /// Cases outside this processor's scope.
    pub skipped: usize,
}

impl Tally {
    /// The number of cases actually judged.
    pub fn attempted(&self) -> usize {
        self.passed + self.failed
    }

    /// The percentage of judged cases that passed.
    pub fn percentage(&self) -> f64 {
        let attempted = self.attempted();
        if attempted == 0 {
            return 0.0;
        }
        (self.passed as f64) * 100.0 / (attempted as f64)
    }
}

/// The directory holding `name` under `OXIXML_XML_SUITES_DIR`, discovered rather than
/// hard-coded so that a dated checkout such as `xmlts20130923` is found too.
pub fn suite_dir(name: &str) -> Option<PathBuf> {
    let root = PathBuf::from(std::env::var_os("OXIXML_XML_SUITES_DIR")?);
    let direct = root.join(name);
    if direct.is_dir() {
        return Some(direct);
    }
    let prefix = name.trim_end_matches(|c: char| c.is_ascii_digit());
    for entry in std::fs::read_dir(&root).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|found| found.starts_with(prefix))
        {
            return Some(path);
        }
    }
    None
}

/// The element children of `element`.
pub fn child_elements(element: &NodeRef) -> Vec<NodeRef> {
    element
        .children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Element)
        .collect()
}

/// The first element child named `name`, ignoring namespaces.
pub fn find_child(element: &NodeRef, name: &str) -> Option<NodeRef> {
    child_elements(element).into_iter().find(|child| {
        child
            .node_name()
            .is_some_and(|found| found.local_name() == name)
    })
}

/// The concatenated text children of `element` — not its whole string value, so that a nested
/// element's text does not leak into an assertion's expected value.
pub fn text_of(element: &NodeRef) -> String {
    element
        .children()
        .into_iter()
        .filter(|child| child.node_kind() == NodeKind::Text)
        .map(|child| child.string_value())
        .collect()
}

/// A canonical form of an XML fragment, for comparing a result with an expected value.
///
/// The fragment is parsed and written back with attributes in a fixed order and without the
/// XML declaration, so that two serializations that differ only in attribute order or in the
/// declaration compare equal. Text that does not parse is compared with its whitespace
/// collapsed, which is the best a lexical comparison can do.
pub fn canonical(text: &str) -> String {
    let trimmed = strip_declaration(text);
    let wrapped = format!("<canonical-wrapper>{trimmed}</canonical-wrapper>");
    let Ok(tree) = Tree::parse_str(&wrapped) else {
        return trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
    };
    let Some(root) = tree.document_element() else {
        return trimmed.to_owned();
    };
    let mut out = String::new();
    for child in root.children() {
        write_canonical(&child, &mut out);
    }
    out
}

fn strip_declaration(text: &str) -> &str {
    let trimmed = text.trim();
    match trimmed.strip_prefix("<?xml") {
        Some(rest) => match rest.find("?>") {
            Some(end) => trimmed[end + 7..].trim(),
            None => trimmed,
        },
        None => trimmed,
    }
}

fn write_canonical(node: &NodeRef, out: &mut String) {
    match node.node_kind() {
        NodeKind::Element => {
            let name = node
                .node_name()
                .map(|name| name.to_string())
                .unwrap_or_default();
            out.push('<');
            out.push_str(&name);
            let mut attributes: Vec<(String, String)> = node
                .attributes()
                .into_iter()
                .map(|attribute| {
                    (
                        attribute
                            .node_name()
                            .map(|name| name.to_string())
                            .unwrap_or_default(),
                        attribute.string_value(),
                    )
                })
                .collect();
            attributes.sort();
            for (name, value) in attributes {
                out.push_str(&format!(" {name}=\"{}\"", escape(&value)));
            }
            out.push('>');
            for child in node.children() {
                write_canonical(&child, out);
            }
            out.push_str(&format!("</{name}>"));
        }
        NodeKind::Text => out.push_str(&escape(&node.string_value())),
        NodeKind::Comment => out.push_str(&format!("<!--{}-->", node.string_value())),
        NodeKind::ProcessingInstruction => {
            let target = node
                .node_name()
                .map(|name| name.local_name().to_owned())
                .unwrap_or_default();
            out.push_str(&format!("<?{target} {}?>", node.string_value()));
        }
        _ => {}
    }
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
