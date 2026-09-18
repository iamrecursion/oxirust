//! Validation of filter/pipeline graphs.
#![allow(dead_code)]

/// A single issue found during graph validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphIssue {
    /// A directed cycle exists; the graph cannot be processed in order.
    CycleDetected {
        /// Human-readable description of where the cycle was detected.
        description: String,
    },
    /// A node has no incoming connections when at least one is required.
    DisconnectedInput {
        /// Label of the node that is missing an incoming connection.
        node_label: String,
    },
    /// A node has no outgoing connections when at least one is required.
    DisconnectedOutput {
        /// Label of the node that is missing an outgoing connection.
        node_label: String,
    },
    /// Two adjacent nodes have incompatible formats / types.
    IncompatibleFormat {
        /// Label of the upstream (source) node.
        from_label: String,
        /// Label of the downstream (destination) node.
        to_label: String,
    },
    /// A node references a resource that does not exist.
    MissingResource {
        /// Label of the node with the missing resource reference.
        node_label: String,
        /// Name / identifier of the missing resource.
        resource: String,
    },
    /// Informational warning; does not block execution.
    Warning {
        /// Human-readable warning message.
        message: String,
    },
}

impl GraphIssue {
    /// Returns `true` if this issue must be resolved before the graph can run.
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        !matches!(self, Self::Warning { .. })
    }

    /// Short category tag for the issue.
    #[must_use]
    pub fn category(&self) -> &'static str {
        match self {
            Self::CycleDetected { .. } => "cycle",
            Self::DisconnectedInput { .. } => "disconnected_input",
            Self::DisconnectedOutput { .. } => "disconnected_output",
            Self::IncompatibleFormat { .. } => "format_mismatch",
            Self::MissingResource { .. } => "missing_resource",
            Self::Warning { .. } => "warning",
        }
    }
}

/// The aggregated result of a validation pass.
#[derive(Debug, Clone, Default)]
pub struct GraphValidationResult {
    issues: Vec<GraphIssue>,
}

impl GraphValidationResult {
    /// Create an empty (pass) result.
    #[must_use]
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Add an issue to the result.
    pub fn push(&mut self, issue: GraphIssue) {
        self.issues.push(issue);
    }

    /// Returns `true` if any fatal issue is present.
    #[must_use]
    pub fn has_fatal(&self) -> bool {
        self.issues.iter().any(|i| i.is_fatal())
    }

    /// Returns `true` if there are zero issues (no warnings either).
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// All issues in this result.
    #[must_use]
    pub fn issues(&self) -> &[GraphIssue] {
        &self.issues
    }

    /// Count of fatal issues.
    #[must_use]
    pub fn fatal_count(&self) -> usize {
        self.issues.iter().filter(|i| i.is_fatal()).count()
    }

    /// Count of warnings (non-fatal issues).
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues.iter().filter(|i| !i.is_fatal()).count()
    }
}

/// A graph description suitable for validation – minimal representation.
#[derive(Debug, Clone)]
pub struct GraphDescriptor {
    /// `(from_index, to_index)` directed edges.
    pub edges: Vec<(usize, usize)>,
    /// Node labels (index-aligned with nodes).
    pub node_labels: Vec<String>,
    /// Whether each node requires at least one incoming edge.
    pub requires_input: Vec<bool>,
    /// Whether each node requires at least one outgoing edge.
    pub requires_output: Vec<bool>,
}

impl GraphDescriptor {
    /// Create a descriptor for `n` nodes with no edges.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            edges: Vec::new(),
            node_labels: (0..n).map(|i| format!("node_{i}")).collect(),
            requires_input: vec![true; n],
            requires_output: vec![true; n],
        }
    }

    /// Add a directed edge.
    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
    }
}

/// Validates a [`GraphDescriptor`] and produces a [`GraphValidationResult`].
pub struct GraphValidator;

impl GraphValidator {
    /// Run all checks on `desc` and return the combined result.
    #[must_use]
    pub fn validate(desc: &GraphDescriptor) -> GraphValidationResult {
        let mut result = GraphValidationResult::new();
        Self::check_connectivity(desc, &mut result);
        Self::check_cycles(desc, &mut result);
        result
    }

    /// Count the number of directed cycles found in `desc`.
    ///
    /// Uses DFS to count strongly-connected back-edges.
    #[must_use]
    pub fn cycle_count(desc: &GraphDescriptor) -> usize {
        let n = desc.node_labels.len();
        if n == 0 {
            return 0;
        }

        // Build adjacency list
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(f, t) in &desc.edges {
            if f < n && t < n {
                adj[f].push(t);
            }
        }

        // 0 = unvisited, 1 = in-stack, 2 = done
        let mut color = vec![0u8; n];
        let mut back_edges = 0usize;

        fn dfs(node: usize, adj: &[Vec<usize>], color: &mut Vec<u8>, back_edges: &mut usize) {
            color[node] = 1;
            for &nb in &adj[node] {
                if color[nb] == 1 {
                    *back_edges += 1;
                } else if color[nb] == 0 {
                    dfs(nb, adj, color, back_edges);
                }
            }
            color[node] = 2;
        }

        for i in 0..n {
            if color[i] == 0 {
                dfs(i, &adj, &mut color, &mut back_edges);
            }
        }
        back_edges
    }

    // -- private helpers --

    fn check_connectivity(desc: &GraphDescriptor, result: &mut GraphValidationResult) {
        let n = desc.node_labels.len();
        let mut has_incoming = vec![false; n];
        let mut has_outgoing = vec![false; n];

        for &(f, t) in &desc.edges {
            if f < n {
                has_outgoing[f] = true;
            }
            if t < n {
                has_incoming[t] = true;
            }
        }

        for i in 0..n {
            if desc.requires_input[i] && !has_incoming[i] {
                result.push(GraphIssue::DisconnectedInput {
                    node_label: desc.node_labels[i].clone(),
                });
            }
            if desc.requires_output[i] && !has_outgoing[i] {
                result.push(GraphIssue::DisconnectedOutput {
                    node_label: desc.node_labels[i].clone(),
                });
            }
        }
    }

    fn check_cycles(desc: &GraphDescriptor, result: &mut GraphValidationResult) {
        if Self::cycle_count(desc) > 0 {
            result.push(GraphIssue::CycleDetected {
                description: "Directed cycle detected in graph".to_string(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_desc() -> GraphDescriptor {
        // 3 nodes: 0 → 1 → 2
        // node 0: source (no input required)
        // node 2: sink (no output required)
        let mut d = GraphDescriptor::new(3);
        d.requires_input[0] = false;
        d.requires_output[2] = false;
        d.add_edge(0, 1);
        d.add_edge(1, 2);
        d
    }

    // --- GraphIssue tests ---

    #[test]
    fn test_cycle_is_fatal() {
        let issue = GraphIssue::CycleDetected {
            description: "cycle".to_string(),
        };
        assert!(issue.is_fatal());
    }

    #[test]
    fn test_warning_not_fatal() {
        let issue = GraphIssue::Warning {
            message: "note".to_string(),
        };
        assert!(!issue.is_fatal());
    }

    #[test]
    fn test_disconnected_input_is_fatal() {
        let issue = GraphIssue::DisconnectedInput {
            node_label: "n".to_string(),
        };
        assert!(issue.is_fatal());
    }

    #[test]
    fn test_category_cycle() {
        let issue = GraphIssue::CycleDetected {
            description: String::new(),
        };
        assert_eq!(issue.category(), "cycle");
    }

    #[test]
    fn test_category_warning() {
        let issue = GraphIssue::Warning {
            message: String::new(),
        };
        assert_eq!(issue.category(), "warning");
    }

    // --- GraphValidationResult tests ---

    #[test]
    fn test_empty_result_is_clean() {
        let r = GraphValidationResult::new();
        assert!(r.is_clean());
        assert!(!r.has_fatal());
    }

    #[test]
    fn test_has_fatal_after_push_fatal() {
        let mut r = GraphValidationResult::new();
        r.push(GraphIssue::CycleDetected {
            description: String::new(),
        });
        assert!(r.has_fatal());
    }

    #[test]
    fn test_fatal_count() {
        let mut r = GraphValidationResult::new();
        r.push(GraphIssue::CycleDetected {
            description: String::new(),
        });
        r.push(GraphIssue::Warning {
            message: String::new(),
        });
        assert_eq!(r.fatal_count(), 1);
    }

    #[test]
    fn test_warning_count() {
        let mut r = GraphValidationResult::new();
        r.push(GraphIssue::Warning {
            message: "w".to_string(),
        });
        r.push(GraphIssue::Warning {
            message: "w2".to_string(),
        });
        assert_eq!(r.warning_count(), 2);
    }

    // --- GraphValidator tests ---

    #[test]
    fn test_validate_linear_is_clean() {
        let d = linear_desc();
        let r = GraphValidator::validate(&d);
        assert!(r.is_clean());
    }

    #[test]
    fn test_validate_cycle_produces_fatal() {
        let mut d = GraphDescriptor::new(2);
        d.requires_input[0] = false;
        d.requires_output[1] = false;
        d.add_edge(0, 1);
        d.add_edge(1, 0); // cycle
        let r = GraphValidator::validate(&d);
        assert!(r.has_fatal());
    }

    #[test]
    fn test_validate_disconnected_input_detected() {
        let mut d = GraphDescriptor::new(2);
        d.requires_input[0] = false;
        d.requires_input[1] = true;
        d.requires_output[0] = false;
        d.requires_output[1] = false;
        // node 1 has no incoming edge despite requires_input = true
        let r = GraphValidator::validate(&d);
        assert!(r.has_fatal());
    }

    #[test]
    fn test_cycle_count_no_cycle() {
        let d = linear_desc();
        assert_eq!(GraphValidator::cycle_count(&d), 0);
    }

    #[test]
    fn test_cycle_count_single_cycle() {
        let mut d = GraphDescriptor::new(3);
        d.add_edge(0, 1);
        d.add_edge(1, 2);
        d.add_edge(2, 0);
        assert_eq!(GraphValidator::cycle_count(&d), 1);
    }

    #[test]
    fn test_cycle_count_empty_graph() {
        let d = GraphDescriptor::new(0);
        assert_eq!(GraphValidator::cycle_count(&d), 0);
    }
}
