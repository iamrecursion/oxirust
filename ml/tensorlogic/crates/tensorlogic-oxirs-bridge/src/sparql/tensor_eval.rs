//! Tensor-based BGP SPARQL evaluator.
//!
//! # Overview
//!
//! [`TensorBgpEvaluator`] evaluates conjunctive SELECT queries (basic graph patterns
//! only — no FILTER/OPTIONAL/UNION) by encoding each triple pattern as a dense
//! Boolean adjacency matrix and chaining binary einsum contractions to compute the
//! join.
//!
//! ## Alphabet limit
//!
//! Each distinct query variable is mapped to a unique axis letter `'a'..'z'`.  A
//! query may therefore reference at most **26 distinct variables**.  Queries
//! exceeding this limit are rejected with [`BridgeError::ValidationError`].
//!
//! ## Constant-subject / constant-object handling
//!
//! When a triple pattern has a `PatternElement::Constant` subject or object the
//! corresponding adjacency matrix is sliced before building the einsum graph,
//! collapsing that dimension to a 1-D vector.  This avoids the need for a
//! one-hot encoding while keeping the einsum dimension accounting correct.
//!
//! ## Execution pipeline
//!
//! 1. Validate query type and pattern.
//! 2. Collect all `Triple` patterns (flattening `Group`).
//! 3. Assign axis letters to variables.
//! 4. For each pattern materialize an `ArrayD<f64>` and register it with
//!    `Scirs2Exec`.
//! 5. Build an `EinsumGraph` of chained binary einsum nodes.
//! 6. Execute via `Scirs2Exec::forward(graph)` (calls `TlAutodiff::forward`).
//! 7. Decode non-zero entries of the output tensor into result rows.

use std::collections::HashMap;

use scirs2_core::ndarray::{Array2, ArrayD, IxDyn};
use tensorlogic_infer::TlAutodiff;
use tensorlogic_ir::{EinsumGraph, EinsumNode};
use tensorlogic_scirs_backend::Scirs2Exec;

use crate::error::BridgeError;
use crate::interned_graph::InternedGraph;
use crate::sparql::types::{GraphPattern, PatternElement, QueryType, SparqlQuery, TriplePattern};

// ─── Index helpers ────────────────────────────────────────────────────────────

/// Decompose a flat (row-major/C-order) index into per-axis indices.
fn compute_multi_index(flat: usize, shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut result = vec![0usize; ndim];
    let mut remaining = flat;
    for i in (0..ndim).rev() {
        result[i] = remaining % shape[i];
        remaining /= shape[i];
    }
    result
}

// ─── axis alphabet ────────────────────────────────────────────────────────────

const AXIS_LETTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

fn axis_letter(index: usize) -> Result<char, BridgeError> {
    AXIS_LETTERS.get(index).map(|&b| b as char).ok_or_else(|| {
        BridgeError::ValidationError(format!(
            "Too many variables: maximum 26 distinct variables supported, index {}",
            index
        ))
    })
}

// ─── Pattern collection ───────────────────────────────────────────────────────

/// Recursively collect all `Triple` patterns from a `GraphPattern`.
///
/// Returns `Err` if any nested operator (Filter, Optional, Union, Bind, Values)
/// is encountered, because those require the non-tensor path.
fn collect_triples(pattern: &GraphPattern) -> Result<Vec<&TriplePattern>, BridgeError> {
    match pattern {
        GraphPattern::Triple(tp) => Ok(vec![tp]),
        GraphPattern::Group(children) => {
            let mut result = Vec::new();
            for child in children {
                match child {
                    GraphPattern::Filter(_)
                    | GraphPattern::Optional(_)
                    | GraphPattern::Union(_, _)
                    | GraphPattern::Bind(_, _)
                    | GraphPattern::Values(_, _) => {
                        return Err(BridgeError::ValidationError(
                            "FILTER/OPTIONAL/UNION/BIND/VALUES not supported in tensor path; \
                             use OxirsSparqlExecutor"
                                .to_string(),
                        ));
                    }
                    other => {
                        let sub = collect_triples(other)?;
                        result.extend(sub);
                    }
                }
            }
            Ok(result)
        }
        GraphPattern::Filter(_)
        | GraphPattern::Optional(_)
        | GraphPattern::Union(_, _)
        | GraphPattern::Bind(_, _)
        | GraphPattern::Values(_, _) => Err(BridgeError::ValidationError(
            "FILTER/OPTIONAL/UNION/BIND/VALUES not supported in tensor path; \
             use OxirsSparqlExecutor"
                .to_string(),
        )),
    }
}

// ─── TensorBgpEvaluator ───────────────────────────────────────────────────────

/// Evaluates a conjunctive SELECT SPARQL query using tensor einsum contractions.
///
/// Only basic graph patterns (BGP) with constant predicates are supported.
/// For queries involving FILTER, OPTIONAL, UNION, BIND, or VALUES, use
/// [`crate::oxirs_executor::OxirsSparqlExecutor`] instead.
pub struct TensorBgpEvaluator<'a> {
    graph: &'a InternedGraph,
}

impl<'a> TensorBgpEvaluator<'a> {
    /// Create a new evaluator bound to the given interned graph.
    pub fn new(graph: &'a InternedGraph) -> Self {
        Self { graph }
    }

    /// Evaluate a SELECT query and return result rows as variable→term mappings.
    ///
    /// # Errors
    ///
    /// - [`BridgeError::ValidationError`] for non-SELECT queries, unsupported
    ///   graph patterns, variable predicates, or too many variables (>26).
    /// - [`BridgeError::IndexError`] if internal tensor indexing fails.
    pub fn evaluate(
        &self,
        query: &SparqlQuery,
    ) -> Result<Vec<HashMap<String, String>>, BridgeError> {
        // ── 1. Validate query type ────────────────────────────────────────────
        let projected_var_names: Vec<String> = match &query.query_type {
            QueryType::Select { select_vars, .. } => select_vars.clone(),
            _ => {
                return Err(BridgeError::ValidationError(
                    "TensorBgpEvaluator only supports SELECT queries".to_string(),
                ))
            }
        };

        // ── 2. Validate top-level pattern ─────────────────────────────────────
        match &query.where_pattern {
            GraphPattern::Filter(_)
            | GraphPattern::Optional(_)
            | GraphPattern::Union(_, _)
            | GraphPattern::Bind(_, _)
            | GraphPattern::Values(_, _) => {
                return Err(BridgeError::ValidationError(
                    "FILTER/OPTIONAL/UNION/BIND not supported in tensor path; \
                     use OxirsSparqlExecutor"
                        .to_string(),
                ));
            }
            _ => {}
        }

        // ── 3. Collect triple patterns ────────────────────────────────────────
        let triple_patterns = collect_triples(&query.where_pattern)?;

        // Validate: no variable predicates
        for tp in &triple_patterns {
            if matches!(&tp.predicate, PatternElement::Variable(_)) {
                return Err(BridgeError::ValidationError(
                    "Variable predicates not supported in tensor path".to_string(),
                ));
            }
        }

        // ── 4. Empty graph ────────────────────────────────────────────────────
        let n = self.graph.num_entities();
        if n == 0 {
            return Ok(vec![]);
        }

        // ── 5. Assign axis letters to variables ───────────────────────────────
        let mut var_to_axis: HashMap<String, char> = HashMap::new();

        let assign_axis = |var_name: &str,
                           var_to_axis: &mut HashMap<String, char>|
         -> Result<char, BridgeError> {
            if let Some(&ch) = var_to_axis.get(var_name) {
                return Ok(ch);
            }
            let idx = var_to_axis.len();
            let ch = axis_letter(idx)?;
            var_to_axis.insert(var_name.to_string(), ch);
            Ok(ch)
        };

        // Pre-scan all patterns to register all variable names in order
        for tp in &triple_patterns {
            if let PatternElement::Variable(v) = &tp.subject {
                assign_axis(v, &mut var_to_axis)?;
            }
            if let PatternElement::Variable(v) = &tp.object {
                assign_axis(v, &mut var_to_axis)?;
            }
        }

        // ── 6. Materialize pattern tensors ────────────────────────────────────
        // Each triple pattern produces a named tensor and a pair of axis chars
        // describing its row (subject) and column (object) dimensions.

        struct PatternTensor {
            /// Name registered in Scirs2Exec
            exec_name: String,
            /// Axis char for the row dimension (subject variable or None for constant)
            row_axis: Option<char>,
            /// Axis char for the col dimension (object variable or None for constant)
            col_axis: Option<char>,
            /// The materialized tensor
            tensor: ArrayD<f64>,
        }

        let mut pattern_tensors: Vec<PatternTensor> = Vec::with_capacity(triple_patterns.len());

        for (pat_idx, tp) in triple_patterns.iter().enumerate() {
            // Resolve predicate ID — if unknown, the pattern matches nothing
            let pred_str = match &tp.predicate {
                PatternElement::Constant(p) => p.as_str(),
                PatternElement::Variable(_) => unreachable!("validated above"),
            };
            let pred_id = match self.graph.intern_or_none(pred_str) {
                Some(id) => id,
                None => {
                    // Predicate not in graph → empty result immediately
                    return Ok(vec![]);
                }
            };

            let pairs = self.graph.predicate_pairs(pred_id);

            // Determine axis assignments
            let row_axis: Option<char> = match &tp.subject {
                PatternElement::Variable(v) => {
                    Some(*var_to_axis.get(v.as_str()).ok_or_else(|| {
                        BridgeError::IndexError(format!("variable '{}' not in axis map", v))
                    })?)
                }
                PatternElement::Constant(_) => None,
            };
            let col_axis: Option<char> = match &tp.object {
                PatternElement::Variable(v) => {
                    Some(*var_to_axis.get(v.as_str()).ok_or_else(|| {
                        BridgeError::IndexError(format!("variable '{}' not in axis map", v))
                    })?)
                }
                PatternElement::Constant(_) => None,
            };

            // Resolve constant IDs (if applicable)
            let const_subj_id: Option<u32> = if row_axis.is_none() {
                match &tp.subject {
                    PatternElement::Constant(s) => match self.graph.intern_or_none(s.as_str()) {
                        Some(id) => Some(id),
                        None => return Ok(vec![]), // constant not in graph → no results
                    },
                    _ => None,
                }
            } else {
                None
            };

            let const_obj_id: Option<u32> = if col_axis.is_none() {
                match &tp.object {
                    PatternElement::Constant(o) => match self.graph.intern_or_none(o.as_str()) {
                        Some(id) => Some(id),
                        None => return Ok(vec![]), // constant not in graph → no results
                    },
                    _ => None,
                }
            } else {
                None
            };

            // Build the materialized tensor
            let tensor: ArrayD<f64> = match (row_axis.is_none(), col_axis.is_none()) {
                (false, false) => {
                    // Full adjacency matrix [N x N]
                    let mut mat = Array2::<f64>::zeros((n, n));
                    for &(s_id, o_id) in pairs {
                        mat[(s_id as usize, o_id as usize)] = 1.0;
                    }
                    mat.into_dyn()
                }
                (true, false) => {
                    // Subject is constant → project to a 1-D vector [N] indexed by object
                    let s_id = const_subj_id.ok_or_else(|| {
                        BridgeError::IndexError("const_subj_id missing".to_string())
                    })?;
                    let mut vec_data = vec![0.0f64; n];
                    for &(subj, obj_id) in pairs {
                        if subj == s_id {
                            vec_data[obj_id as usize] = 1.0;
                        }
                    }
                    ArrayD::from_shape_vec(IxDyn(&[n]), vec_data).map_err(|e| {
                        BridgeError::IndexError(format!("shape error (const subj): {}", e))
                    })?
                }
                (false, true) => {
                    // Object is constant → project to a 1-D vector [N] indexed by subject
                    let o_id = const_obj_id.ok_or_else(|| {
                        BridgeError::IndexError("const_obj_id missing".to_string())
                    })?;
                    let mut vec_data = vec![0.0f64; n];
                    for &(subj, obj) in pairs {
                        if obj == o_id {
                            vec_data[subj as usize] = 1.0;
                        }
                    }
                    ArrayD::from_shape_vec(IxDyn(&[n]), vec_data).map_err(|e| {
                        BridgeError::IndexError(format!("shape error (const obj): {}", e))
                    })?
                }
                (true, true) => {
                    // Both are constants → scalar 0.0 or 1.0
                    let s_id = const_subj_id.ok_or_else(|| {
                        BridgeError::IndexError("const_subj_id missing".to_string())
                    })?;
                    let o_id = const_obj_id.ok_or_else(|| {
                        BridgeError::IndexError("const_obj_id missing".to_string())
                    })?;
                    let val = if pairs.contains(&(s_id, o_id)) {
                        1.0f64
                    } else {
                        0.0f64
                    };
                    ArrayD::from_shape_vec(IxDyn(&[]), vec![val]).map_err(|e| {
                        BridgeError::IndexError(format!("shape error (both const): {}", e))
                    })?
                }
            };

            let exec_name = format!("pat_{}", pat_idx);
            pattern_tensors.push(PatternTensor {
                exec_name,
                row_axis,
                col_axis,
                tensor,
            });
        }

        // ── 7. Build the EinsumGraph ──────────────────────────────────────────
        // We chain binary einsum contractions.  The running subscript tracks
        // which axis letters are currently "active" in the accumulated result.

        // Determine the full set of projected variable axes (in order of
        // select_vars for consistent output index ordering).
        let projected_axes: Vec<char> = projected_var_names
            .iter()
            .filter_map(|v| var_to_axis.get(v.as_str()).copied())
            .collect();

        if pattern_tensors.is_empty() {
            // No patterns → no results
            return Ok(vec![]);
        }

        // Helper: collect axis chars for a pattern tensor
        let pattern_axes = |pt: &PatternTensor| -> Vec<char> {
            let mut axes = Vec::with_capacity(2);
            if let Some(ra) = pt.row_axis {
                axes.push(ra);
            }
            if let Some(ca) = pt.col_axis {
                axes.push(ca);
            }
            axes
        };

        let mut exec = Scirs2Exec::new();
        let mut eg = EinsumGraph::new();

        // Register all pattern tensors with the executor
        for pt in &pattern_tensors {
            exec.add_tensor(pt.exec_name.clone(), pt.tensor.clone());
        }

        // Step 0: first pattern tensor
        let first_pt = &pattern_tensors[0];
        let first_axes = pattern_axes(first_pt);

        // The "running" tensor index in the EinsumGraph and the running subscript
        let first_tidx = eg.add_tensor(first_pt.exec_name.clone());
        eg.add_input(first_tidx)
            .map_err(|e| BridgeError::IndexError(e.to_string()))?;

        let mut running_tidx = first_tidx;
        let mut running_axes: Vec<char> = first_axes.clone();

        // Step 1..N: pairwise contractions
        for pt in &pattern_tensors[1..] {
            let next_axes = pattern_axes(pt);
            let next_tidx = eg.add_tensor(pt.exec_name.clone());
            eg.add_input(next_tidx)
                .map_err(|e| BridgeError::IndexError(e.to_string()))?;

            // Compute the union of running_axes + next_axes (preserving order,
            // de-duplicating for shared join axes)
            let mut output_axes: Vec<char> = running_axes.clone();
            for &ch in &next_axes {
                if !output_axes.contains(&ch) {
                    output_axes.push(ch);
                }
            }

            // Build einsum spec: "running_subs,next_subs->output_subs"
            let running_subs: String = running_axes.iter().collect();
            let next_subs: String = next_axes.iter().collect();
            let output_subs: String = output_axes.iter().collect();
            let spec = format!("{},{}->{}", running_subs, next_subs, output_subs);

            let result_tidx = eg.add_tensor(format!("join_{}", next_tidx));
            eg.add_node(EinsumNode::einsum(
                spec,
                vec![running_tidx, next_tidx],
                vec![result_tidx],
            ))
            .map_err(|e| BridgeError::IndexError(e.to_string()))?;

            running_tidx = result_tidx;
            running_axes = output_axes;
        }

        // Step 2: reduce (sum-out) axes that are NOT in projected_axes
        // We emit a final einsum that sums over non-projected axes.
        let current_subs: String = running_axes.iter().collect();
        let output_subs: String = projected_axes.iter().collect();

        // Only add a final projection node if the subscripts differ
        let final_tidx = if current_subs != output_subs && !output_subs.is_empty() {
            let proj_subs = format!("{}->{}", current_subs, output_subs);
            let result_tidx = eg.add_tensor("result".to_string());
            eg.add_node(EinsumNode::einsum(
                proj_subs,
                vec![running_tidx],
                vec![result_tidx],
            ))
            .map_err(|e| BridgeError::IndexError(e.to_string()))?;
            result_tidx
        } else {
            running_tidx
        };

        eg.add_output(final_tidx)
            .map_err(|e| BridgeError::IndexError(e.to_string()))?;

        // ── 8. Execute ────────────────────────────────────────────────────────
        let output_tensor = exec.forward(&eg).map_err(|e| {
            BridgeError::ValidationError(format!("Tensor forward pass failed: {}", e))
        })?;

        // ── 9. Decode result tensor ───────────────────────────────────────────
        let shape = output_tensor.shape().to_vec();
        let num_proj = projected_axes.len();

        // Handle edge case: no projected axes (COUNT(*) or all constants)
        if num_proj == 0 {
            let val = output_tensor.iter().next().copied().unwrap_or(0.0);
            if val > 0.5 {
                return Ok(vec![HashMap::new()]);
            } else {
                return Ok(vec![]);
            }
        }

        // Decode the result tensor by iterating flat values and computing
        // multi-dimensional indices manually from the tensor shape.
        let flat_values: Vec<f64> = output_tensor.iter().copied().collect();
        let mut results: Vec<HashMap<String, String>> = Vec::new();

        // Compute total number of elements
        let total_elements: usize = shape.iter().product();

        for flat_idx in 0..total_elements {
            let val = flat_values.get(flat_idx).copied().unwrap_or(0.0);
            if val < 0.5 {
                continue;
            }

            // Decompose flat_idx into per-axis indices using row-major (C) order
            let multi_idx: Vec<usize> = compute_multi_index(flat_idx, &shape);

            if multi_idx.len() != num_proj {
                continue;
            }

            let mut row: HashMap<String, String> = HashMap::new();
            let mut valid = true;
            for (axis_pos, var_name) in projected_var_names.iter().enumerate() {
                let entity_id = multi_idx[axis_pos] as u32;
                match self.graph.term(entity_id) {
                    Some(term_str) => {
                        row.insert(var_name.clone(), term_str.to_string());
                    }
                    None => {
                        valid = false;
                        break;
                    }
                }
            }
            if valid {
                results.push(row);
            }
        }

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparql::types::{
        GraphPattern, PatternElement, QueryType, SelectElement, SparqlQuery, TriplePattern,
    };

    fn make_select_query(vars: Vec<&str>, pattern: GraphPattern) -> SparqlQuery {
        SparqlQuery {
            query_type: QueryType::Select {
                projections: vars
                    .iter()
                    .map(|v| SelectElement::Variable(v.to_string()))
                    .collect(),
                select_vars: vars.iter().map(|v| v.to_string()).collect(),
                distinct: false,
            },
            where_pattern: pattern,
            group_by: vec![],
            having: vec![],
            limit: None,
            offset: None,
            order_by: vec![],
        }
    }

    fn var(name: &str) -> PatternElement {
        PatternElement::Variable(name.to_string())
    }

    fn cst(s: &str) -> PatternElement {
        PatternElement::Constant(s.to_string())
    }

    fn triple(s: PatternElement, p: PatternElement, o: PatternElement) -> GraphPattern {
        GraphPattern::Triple(TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        })
    }

    // ── basic SELECT ?x ?y WHERE { ?x <knows> ?y } ────────────────────────────

    #[test]
    fn test_interned_graph_single_pattern() {
        let mut g = InternedGraph::new();
        g.add_triple("Alice", "knows", "Bob");
        g.add_triple("Bob", "knows", "Carol");

        let evaluator = TensorBgpEvaluator::new(&g);
        let query = make_select_query(vec!["x", "y"], triple(var("x"), cst("knows"), var("y")));

        let results = evaluator.evaluate(&query).expect("evaluate must succeed");
        assert_eq!(results.len(), 2, "should find 2 knows pairs");

        let pairs: Vec<(&str, &str)> = results
            .iter()
            .map(|r| (r["x"].as_str(), r["y"].as_str()))
            .collect();
        assert!(pairs.contains(&("Alice", "Bob")));
        assert!(pairs.contains(&("Bob", "Carol")));
    }

    // ── join: SELECT ?x ?z WHERE { ?x <knows> ?y . ?y <knows> ?z } ───────────

    #[test]
    fn test_interned_graph_join() {
        let mut g = InternedGraph::new();
        g.add_triple("Alice", "knows", "Bob");
        g.add_triple("Bob", "knows", "Carol");

        let evaluator = TensorBgpEvaluator::new(&g);
        let pattern = GraphPattern::Group(vec![
            triple(var("x"), cst("knows"), var("y")),
            triple(var("y"), cst("knows"), var("z")),
        ]);
        let query = make_select_query(vec!["x", "z"], pattern);

        let results = evaluator.evaluate(&query).expect("evaluate must succeed");
        // Only path: Alice → Bob → Carol
        assert_eq!(results.len(), 1, "expected one path (Alice → Carol)");
        assert_eq!(results[0]["x"], "Alice");
        assert_eq!(results[0]["z"], "Carol");
    }

    // ── constant subject: SELECT ?y WHERE { <Alice> <knows> ?y } ─────────────

    #[test]
    fn test_constant_subject() {
        let mut g = InternedGraph::new();
        g.add_triple("Alice", "knows", "Bob");
        g.add_triple("Bob", "knows", "Carol");

        let evaluator = TensorBgpEvaluator::new(&g);
        let query = make_select_query(vec!["y"], triple(cst("Alice"), cst("knows"), var("y")));

        let results = evaluator.evaluate(&query).expect("evaluate must succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["y"], "Bob");
    }

    // ── empty graph → empty results ───────────────────────────────────────────

    #[test]
    fn test_empty_graph() {
        let g = InternedGraph::new();
        let evaluator = TensorBgpEvaluator::new(&g);
        let query = make_select_query(vec!["x"], triple(var("x"), cst("knows"), var("y")));
        let results = evaluator.evaluate(&query).expect("evaluate must succeed");
        assert!(results.is_empty());
    }

    // ── unknown predicate → empty results ────────────────────────────────────

    #[test]
    fn test_unknown_predicate() {
        let mut g = InternedGraph::new();
        g.add_triple("Alice", "knows", "Bob");

        let evaluator = TensorBgpEvaluator::new(&g);
        let query = make_select_query(vec!["x"], triple(var("x"), cst("likes"), var("y")));
        let results = evaluator.evaluate(&query).expect("evaluate must succeed");
        assert!(results.is_empty());
    }

    // ── unsupported FILTER → ValidationError ────────────────────────────────

    #[test]
    fn test_unsupported_filter_error() {
        use crate::sparql::types::FilterCondition;
        let g = InternedGraph::new();
        let evaluator = TensorBgpEvaluator::new(&g);
        let query = make_select_query(
            vec!["x"],
            GraphPattern::Filter(FilterCondition::Bound("x".to_string())),
        );
        match evaluator.evaluate(&query) {
            Err(BridgeError::ValidationError(_)) => {}
            other => panic!("expected ValidationError, got: {:?}", other),
        }
    }

    // ── variable predicate → ValidationError ─────────────────────────────────

    #[test]
    fn test_variable_predicate_error() {
        let g = InternedGraph::new();
        let evaluator = TensorBgpEvaluator::new(&g);
        let query = make_select_query(vec!["x"], triple(var("x"), var("p"), var("y")));
        match evaluator.evaluate(&query) {
            Err(BridgeError::ValidationError(_)) => {}
            other => panic!("expected ValidationError, got: {:?}", other),
        }
    }

    // ── non-SELECT query → ValidationError ───────────────────────────────────

    #[test]
    fn test_non_select_query_error() {
        let g = InternedGraph::new();
        let evaluator = TensorBgpEvaluator::new(&g);
        let query = SparqlQuery {
            query_type: QueryType::Ask,
            where_pattern: triple(var("x"), cst("p"), var("y")),
            group_by: vec![],
            having: vec![],
            limit: None,
            offset: None,
            order_by: vec![],
        };
        match evaluator.evaluate(&query) {
            Err(BridgeError::ValidationError(_)) => {}
            other => panic!("expected ValidationError, got: {:?}", other),
        }
    }
}
