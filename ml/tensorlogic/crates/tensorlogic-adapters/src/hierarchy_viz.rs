//! Domain hierarchy visualization and analysis.
//!
//! Renders the domain subtype hierarchy as ASCII trees, DOT (Graphviz) format,
//! and computes structural metrics (depth, breadth, root count).
//!
//! # Example
//!
//! ```rust
//! use tensorlogic_adapters::{SymbolTable, DomainInfo, DomainHierarchy};
//! use tensorlogic_adapters::hierarchy_viz::{render_hierarchy_ascii, hierarchy_stats};
//!
//! let mut table = SymbolTable::new();
//! table.add_domain(DomainInfo::new("Entity", 500)).unwrap();
//! table.add_domain(DomainInfo::new("Person", 100)).unwrap();
//! table.add_domain(DomainInfo::new("Student", 50)).unwrap();
//!
//! let mut hierarchy = DomainHierarchy::new();
//! hierarchy.add_subtype("Person", "Entity");
//! hierarchy.add_subtype("Student", "Person");
//!
//! let ascii = render_hierarchy_ascii(&table, &hierarchy);
//! assert!(ascii.contains("Entity"));
//! assert!(ascii.contains("Person"));
//!
//! let stats = hierarchy_stats(&table, &hierarchy);
//! assert_eq!(stats.total_domains, 3);
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::{DomainHierarchy, SymbolTable};

/// A node in the domain hierarchy tree.
#[derive(Debug, Clone)]
pub struct HierarchyNode {
    /// Domain name.
    pub name: String,
    /// Child nodes in the hierarchy.
    pub children: Vec<HierarchyNode>,
    /// Depth in the tree (root = 0).
    pub depth: usize,
    /// Cardinality of the domain from the symbol table.
    pub domain_size: usize,
}

/// Statistics about the domain hierarchy.
#[derive(Debug, Clone, Default)]
pub struct HierarchyStats {
    /// Number of root domains (no parent).
    pub root_count: usize,
    /// Total number of domains.
    pub total_domains: usize,
    /// Maximum depth of the hierarchy tree.
    pub max_depth: usize,
    /// Maximum branching factor (max children of any single node).
    pub max_breadth: usize,
    /// Number of leaf domains (no children).
    pub leaf_count: usize,
}

impl HierarchyStats {
    /// Returns `true` if the hierarchy is flat (max depth <= 1).
    pub fn is_flat(&self) -> bool {
        self.max_depth <= 1
    }

    /// Returns a one-line summary string.
    pub fn summary(&self) -> String {
        format!(
            "{} domains, depth {}, {} roots, {} leaves",
            self.total_domains, self.max_depth, self.root_count, self.leaf_count
        )
    }
}

/// Build the hierarchy tree from a [`SymbolTable`] and [`DomainHierarchy`].
///
/// Domains present in the symbol table but absent from the hierarchy are treated
/// as roots. The returned vector contains one [`HierarchyNode`] per root,
/// sorted alphabetically by name for deterministic output.
pub fn build_hierarchy(table: &SymbolTable, hierarchy: &DomainHierarchy) -> Vec<HierarchyNode> {
    // Collect all domain names from the symbol table.
    let domain_names: BTreeSet<String> = table.domains.keys().cloned().collect();

    if domain_names.is_empty() {
        return Vec::new();
    }

    // Build a map of parent -> sorted children (only for domains in the table).
    let mut children_map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut has_parent: BTreeSet<String> = BTreeSet::new();

    for name in &domain_names {
        if let Some(parent) = hierarchy.get_parent(name) {
            // Only track the relationship if the parent is also in the table.
            if domain_names.contains(parent) {
                children_map
                    .entry(parent.to_string())
                    .or_default()
                    .insert(name.clone());
                has_parent.insert(name.clone());
            }
        }
    }

    // Root domains: present in table but not a child of any other domain in the table.
    let roots: Vec<String> = domain_names
        .iter()
        .filter(|name| !has_parent.contains(*name))
        .cloned()
        .collect();

    // Recursively build nodes.
    roots
        .into_iter()
        .map(|name| build_node(&name, 0, table, &children_map))
        .collect()
}

/// Recursively construct a [`HierarchyNode`].
fn build_node(
    name: &str,
    depth: usize,
    table: &SymbolTable,
    children_map: &BTreeMap<String, BTreeSet<String>>,
) -> HierarchyNode {
    let domain_size = table.get_domain(name).map(|d| d.cardinality).unwrap_or(0);

    let children: Vec<HierarchyNode> = children_map
        .get(name)
        .map(|kids| {
            kids.iter()
                .map(|child| build_node(child, depth + 1, table, children_map))
                .collect()
        })
        .unwrap_or_default();

    HierarchyNode {
        name: name.to_string(),
        children,
        depth,
        domain_size,
    }
}

/// Compute hierarchy statistics from a [`SymbolTable`] and [`DomainHierarchy`].
pub fn hierarchy_stats(table: &SymbolTable, hierarchy: &DomainHierarchy) -> HierarchyStats {
    let roots = build_hierarchy(table, hierarchy);

    if roots.is_empty() {
        return HierarchyStats::default();
    }

    let mut stats = HierarchyStats {
        root_count: roots.len(),
        total_domains: 0,
        max_depth: 0,
        max_breadth: 0,
        leaf_count: 0,
    };

    for root in &roots {
        collect_stats(root, &mut stats);
    }

    stats
}

/// Walk the tree collecting statistics.
fn collect_stats(node: &HierarchyNode, stats: &mut HierarchyStats) {
    stats.total_domains += 1;

    let effective_depth = node.depth + 1; // depth is 0-based, we want max level count
    if effective_depth > stats.max_depth {
        stats.max_depth = effective_depth;
    }

    let breadth = node.children.len();
    if breadth > stats.max_breadth {
        stats.max_breadth = breadth;
    }

    if node.children.is_empty() {
        stats.leaf_count += 1;
    }

    for child in &node.children {
        collect_stats(child, stats);
    }
}

/// Render the domain hierarchy as an ASCII tree.
///
/// # Example output
///
/// ```text
/// Entity (500)
/// ├── Organization (200)
/// │   └── University (20)
/// └── Person (100)
///     ├── Student (50)
///     └── Teacher (30)
/// ```
pub fn render_hierarchy_ascii(table: &SymbolTable, hierarchy: &DomainHierarchy) -> String {
    let roots = build_hierarchy(table, hierarchy);
    let mut out = String::new();

    if roots.is_empty() {
        out.push_str("(empty hierarchy)\n");
        return out;
    }

    for (i, root) in roots.iter().enumerate() {
        let is_last = i == roots.len() - 1;
        render_node_ascii(&mut out, root, "", is_last, true);
    }

    out
}

/// Recursive ASCII renderer for a single node.
fn render_node_ascii(
    out: &mut String,
    node: &HierarchyNode,
    prefix: &str,
    is_last: bool,
    is_root: bool,
) {
    let connector = if is_root {
        ""
    } else if is_last {
        "└── "
    } else {
        "├── "
    };

    let _ = writeln!(
        out,
        "{}{}{} ({})",
        prefix, connector, node.name, node.domain_size
    );

    let child_prefix = if is_root {
        String::new()
    } else if is_last {
        format!("{}    ", prefix)
    } else {
        format!("{}│   ", prefix)
    };

    for (i, child) in node.children.iter().enumerate() {
        let child_is_last = i == node.children.len() - 1;
        render_node_ascii(out, child, &child_prefix, child_is_last, false);
    }
}

/// Render hierarchy as DOT (Graphviz) format.
///
/// The output can be piped to `dot -Tpng` to produce an image.
pub fn render_hierarchy_dot(table: &SymbolTable, hierarchy: &DomainHierarchy) -> String {
    let roots = build_hierarchy(table, hierarchy);
    let mut dot = String::new();
    let _ = writeln!(dot, "digraph DomainHierarchy {{");
    let _ = writeln!(dot, "  rankdir=TB;");
    let _ = writeln!(dot, "  node [shape=box];");

    for root in &roots {
        render_dot_node(&mut dot, root);
    }

    let _ = writeln!(dot, "}}");
    dot
}

/// Recursive DOT renderer for a single node and its children.
fn render_dot_node(dot: &mut String, node: &HierarchyNode) {
    let _ = writeln!(
        dot,
        "  \"{}\" [label=\"{}\\n(size={})\"];",
        node.name, node.name, node.domain_size
    );
    for child in &node.children {
        let _ = writeln!(dot, "  \"{}\" -> \"{}\";", node.name, child.name);
        render_dot_node(dot, child);
    }
}

/// Find all ancestors of a domain (parent chain up to a root).
///
/// Returns an empty vector if the domain has no parent in the hierarchy.
/// The result order is parent-first (immediate parent, then grandparent, etc.).
pub fn ancestors(hierarchy: &DomainHierarchy, domain: &str) -> Vec<String> {
    hierarchy.get_ancestors(domain)
}

/// Find all descendants of a domain (all children recursively).
///
/// Returns an empty vector if the domain has no children.
pub fn descendants(hierarchy: &DomainHierarchy, domain: &str) -> Vec<String> {
    hierarchy.get_descendants(domain)
}

/// Render a compact indented listing (simpler than ASCII tree).
///
/// Each level is indented by two spaces. Useful for logging.
pub fn render_hierarchy_indented(table: &SymbolTable, hierarchy: &DomainHierarchy) -> String {
    let roots = build_hierarchy(table, hierarchy);
    let mut out = String::new();

    if roots.is_empty() {
        out.push_str("(empty hierarchy)\n");
        return out;
    }

    for root in &roots {
        render_indented_node(&mut out, root);
    }

    out
}

/// Recursive indented renderer.
fn render_indented_node(out: &mut String, node: &HierarchyNode) {
    let indent = "  ".repeat(node.depth);
    let _ = writeln!(out, "{}{} ({})", indent, node.name, node.domain_size);
    for child in &node.children {
        render_indented_node(out, child);
    }
}

/// Find the path from one domain to another through the hierarchy.
///
/// Returns `None` if no path exists (i.e., neither is an ancestor of the other).
pub fn path_between(hierarchy: &DomainHierarchy, from: &str, to: &str) -> Option<Vec<String>> {
    if from == to {
        return Some(vec![from.to_string()]);
    }

    // Check if `to` is an ancestor of `from`.
    let from_ancestors = hierarchy.get_ancestors(from);
    if let Some(pos) = from_ancestors.iter().position(|a| a == to) {
        let mut path = vec![from.to_string()];
        path.extend(from_ancestors[..=pos].to_vec());
        return Some(path);
    }

    // Check if `from` is an ancestor of `to`.
    let to_ancestors = hierarchy.get_ancestors(to);
    if let Some(pos) = to_ancestors.iter().position(|a| a == from) {
        let mut path: Vec<String> = to_ancestors[..=pos].iter().rev().cloned().collect();
        path.push(to.to_string());
        // Reverse so it goes from `from` downward.
        // Actually, to_ancestors is [parent, grandparent, ...], position found means
        // from is at index `pos`. Build path from->...->to.
        let mut result = vec![from.to_string()];
        for ancestor in to_ancestors[..pos].iter().rev() {
            result.push(ancestor.clone());
        }
        result.push(to.to_string());
        return Some(result);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DomainInfo;

    /// Helper: create a symbol table with the given (name, cardinality) pairs.
    fn make_table(domains: &[(&str, usize)]) -> SymbolTable {
        let mut table = SymbolTable::new();
        for &(name, card) in domains {
            table
                .add_domain(DomainInfo::new(name, card))
                .expect("add_domain should succeed");
        }
        table
    }

    #[test]
    fn test_hierarchy_empty_table() {
        let table = SymbolTable::new();
        let hierarchy = DomainHierarchy::new();
        let ascii = render_hierarchy_ascii(&table, &hierarchy);
        assert_eq!(ascii, "(empty hierarchy)\n");
    }

    #[test]
    fn test_hierarchy_flat_domains() {
        let table = make_table(&[("Alpha", 10), ("Beta", 20), ("Gamma", 30)]);
        let hierarchy = DomainHierarchy::new();
        let ascii = render_hierarchy_ascii(&table, &hierarchy);
        // All three should appear as roots, each on its own line.
        assert!(ascii.contains("Alpha"));
        assert!(ascii.contains("Beta"));
        assert!(ascii.contains("Gamma"));
        // No tree connectors for roots.
        assert!(!ascii.contains("├"));
        assert!(!ascii.contains("└"));
    }

    #[test]
    fn test_hierarchy_stats_flat() {
        let table = make_table(&[("A", 1), ("B", 2), ("C", 3)]);
        let hierarchy = DomainHierarchy::new();
        let stats = hierarchy_stats(&table, &hierarchy);
        assert_eq!(stats.max_depth, 1);
        assert!(stats.is_flat());
        assert_eq!(stats.root_count, 3);
        assert_eq!(stats.leaf_count, 3);
    }

    #[test]
    fn test_hierarchy_stats_summary() {
        let table = make_table(&[("A", 1), ("B", 2)]);
        let hierarchy = DomainHierarchy::new();
        let summary = hierarchy_stats(&table, &hierarchy).summary();
        assert!(summary.contains("2 domains"));
        assert!(summary.contains("2 roots"));
    }

    #[test]
    fn test_hierarchy_node_depth() {
        let table = make_table(&[("Entity", 500), ("Person", 100), ("Student", 50)]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Person", "Entity");
        hierarchy.add_subtype("Student", "Person");

        let roots = build_hierarchy(&table, &hierarchy);
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].depth, 0);
        assert_eq!(roots[0].name, "Entity");

        // Person is child of Entity.
        assert_eq!(roots[0].children.len(), 1);
        assert_eq!(roots[0].children[0].depth, 1);

        // Student is child of Person.
        assert_eq!(roots[0].children[0].children.len(), 1);
        assert_eq!(roots[0].children[0].children[0].depth, 2);
    }

    #[test]
    fn test_hierarchy_ascii_contains_names() {
        let table = make_table(&[("Entity", 500), ("Person", 100)]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Person", "Entity");

        let ascii = render_hierarchy_ascii(&table, &hierarchy);
        assert!(ascii.contains("Entity"));
        assert!(ascii.contains("Person"));
        assert!(ascii.contains("500"));
        assert!(ascii.contains("100"));
    }

    #[test]
    fn test_hierarchy_ascii_tree_connectors() {
        let table = make_table(&[("Entity", 500), ("Person", 100), ("Org", 200)]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Person", "Entity");
        hierarchy.add_subtype("Org", "Entity");

        let ascii = render_hierarchy_ascii(&table, &hierarchy);
        // With two children we expect both connectors.
        let has_branch = ascii.contains('├') || ascii.contains('└');
        assert!(has_branch, "Expected tree connectors in:\n{}", ascii);
    }

    #[test]
    fn test_hierarchy_dot_contains_digraph() {
        let table = make_table(&[("A", 1)]);
        let hierarchy = DomainHierarchy::new();
        let dot = render_hierarchy_dot(&table, &hierarchy);
        assert!(dot.starts_with("digraph DomainHierarchy {"));
    }

    #[test]
    fn test_hierarchy_dot_contains_edges() {
        let table = make_table(&[("Parent", 10), ("Child", 5)]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Child", "Parent");

        let dot = render_hierarchy_dot(&table, &hierarchy);
        assert!(dot.contains("->"), "Expected edges in DOT output:\n{}", dot);
        assert!(dot.contains("\"Parent\" -> \"Child\""));
    }

    #[test]
    fn test_hierarchy_stats_default() {
        let stats = HierarchyStats::default();
        assert_eq!(stats.root_count, 0);
        assert_eq!(stats.total_domains, 0);
        assert_eq!(stats.max_depth, 0);
        assert_eq!(stats.max_breadth, 0);
        assert_eq!(stats.leaf_count, 0);
    }

    #[test]
    fn test_hierarchy_stats_leaf_count() {
        let table = make_table(&[
            ("Entity", 500),
            ("Person", 100),
            ("Student", 50),
            ("Teacher", 30),
            ("Org", 200),
        ]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Person", "Entity");
        hierarchy.add_subtype("Student", "Person");
        hierarchy.add_subtype("Teacher", "Person");
        hierarchy.add_subtype("Org", "Entity");

        let stats = hierarchy_stats(&table, &hierarchy);
        // Leaves: Student, Teacher, Org
        assert_eq!(stats.leaf_count, 3);
    }

    #[test]
    fn test_hierarchy_stats_root_count() {
        let table = make_table(&[("A", 1), ("B", 2), ("C", 3), ("D", 4)]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("B", "A");
        // C and D are roots, A is a root.
        let stats = hierarchy_stats(&table, &hierarchy);
        assert_eq!(stats.root_count, 3); // A, C, D
    }

    #[test]
    fn test_ancestors_empty() {
        let hierarchy = DomainHierarchy::new();
        let result = ancestors(&hierarchy, "Root");
        assert!(result.is_empty());
    }

    #[test]
    fn test_descendants_leaf() {
        let hierarchy = DomainHierarchy::new();
        // A domain not in the hierarchy at all has no descendants.
        let result = descendants(&hierarchy, "Leaf");
        assert!(result.is_empty());
    }

    #[test]
    fn test_hierarchy_render_deterministic() {
        let table = make_table(&[
            ("Entity", 500),
            ("Person", 100),
            ("Student", 50),
            ("Org", 200),
        ]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Person", "Entity");
        hierarchy.add_subtype("Student", "Person");
        hierarchy.add_subtype("Org", "Entity");

        let ascii1 = render_hierarchy_ascii(&table, &hierarchy);
        let ascii2 = render_hierarchy_ascii(&table, &hierarchy);
        assert_eq!(ascii1, ascii2, "Rendering should be deterministic");
    }

    #[test]
    fn test_hierarchy_multiple_roots() {
        let table = make_table(&[("TreeA", 10), ("TreeB", 20), ("ChildA", 5)]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("ChildA", "TreeA");

        let roots = build_hierarchy(&table, &hierarchy);
        assert_eq!(roots.len(), 2); // TreeA and TreeB
        let names: Vec<&str> = roots.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"TreeA"));
        assert!(names.contains(&"TreeB"));
    }

    #[test]
    fn test_hierarchy_stats_max_breadth() {
        let table = make_table(&[("Root", 100), ("A", 10), ("B", 20), ("C", 30)]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("A", "Root");
        hierarchy.add_subtype("B", "Root");
        hierarchy.add_subtype("C", "Root");

        let stats = hierarchy_stats(&table, &hierarchy);
        assert_eq!(stats.max_breadth, 3);
    }

    #[test]
    fn test_hierarchy_node_domain_size() {
        let table = make_table(&[("Entity", 500), ("Person", 100)]);
        let mut hierarchy = DomainHierarchy::new();
        hierarchy.add_subtype("Person", "Entity");

        let roots = build_hierarchy(&table, &hierarchy);
        assert_eq!(roots[0].domain_size, 500);
        assert_eq!(roots[0].children[0].domain_size, 100);
    }
}
