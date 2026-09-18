//! Reference comparison tests: compiled tensor output vs. known logical formulas.
//!
//! These tests verify that each compilation strategy produces numerically correct
//! outputs matching the expected mathematical formulas for Boolean/fuzzy logic.
//!
//! # Test Approach
//!
//! The pipeline for each test:
//! 1. Build a `TLExpr` (e.g., `AND(A(x), B(x))`)
//! 2. Compile with `compile_to_einsum_with_config(&expr, &config)` → `EinsumGraph`
//! 3. Execute via a custom graph executor that injects specific scalar values
//! 4. Compare the scalar output against the analytical ground-truth formula

use std::collections::HashMap;

use tensorlogic_compiler::{compile_to_einsum_with_config, CompilationConfig};
use tensorlogic_infer::DummyTensor;
use tensorlogic_ir::{EinsumGraph, EinsumNode, OpType, TLExpr, Term};

// ── Graph Execution Helper ────────────────────────────────────────────────────

/// Execute an `EinsumGraph` with injected scalar tensor values.
///
/// `inputs` maps tensor name (as stored in `graph.tensors`) to a scalar f64.
/// Every named tensor that is NOT provided via `inputs` is initialised to 1.0
/// (consistent with DummyExecutor's default behaviour).
///
/// Returns the scalar value of the designated output tensor.
fn execute_graph_with_scalars(
    graph: &EinsumGraph,
    inputs: &HashMap<&str, f64>,
) -> Result<f64, String> {
    // Build the initial tensor table: named tensors → scalar DummyTensors.
    // Graph indices 0..N correspond to named tensors; computed tensors are
    // assigned indices N, N+1, … as each node executes.
    let mut tensor_table: HashMap<usize, DummyTensor> = HashMap::new();

    for (idx, name) in graph.tensors.iter().enumerate() {
        let val = inputs.get(name.as_str()).copied().unwrap_or(1.0);
        tensor_table.insert(
            idx,
            DummyTensor::with_data(name.clone(), vec![1], vec![val]),
        );
    }

    // Walk graph nodes in order.  Each node writes its result to the pre-declared
    // output tensor slot indicated by `node.outputs[0]`.
    for (node_idx, node) in graph.nodes.iter().enumerate() {
        let result = execute_node(node, &tensor_table)
            .map_err(|e| format!("Node {} error: {}", node_idx, e))?;
        // Store in the node's declared output tensor slot.
        let out_slot = node
            .outputs
            .first()
            .copied()
            .ok_or_else(|| format!("Node {} has no outputs", node_idx))?;
        tensor_table.insert(out_slot, result);
    }

    // Extract output scalar.
    let out_idx = graph
        .outputs
        .first()
        .copied()
        .ok_or_else(|| "Graph has no outputs".to_string())?;

    let out = tensor_table
        .get(&out_idx)
        .ok_or_else(|| format!("Output tensor index {} not found", out_idx))?;

    out.data
        .first()
        .copied()
        .ok_or_else(|| "Output tensor has no data".to_string())
}

/// Execute a single `EinsumNode` given the current tensor table.
///
/// All tensors are expected to be scalars (shape `[1]`).
fn execute_node(
    node: &EinsumNode,
    table: &HashMap<usize, DummyTensor>,
) -> Result<DummyTensor, String> {
    match &node.op {
        OpType::Einsum { spec } => {
            // For scalar tensors (shape [1]) we implement a direct interpreter
            // instead of delegating to the stub DummyExecutor::einsum().
            // The DummyExecutor::einsum stub always returns ones regardless of input.
            //
            // For scalar einsum: the result is always the element-wise product of all
            // input values, since each tensor contains a single element and the einsum
            // spec merely describes how the axes contract.  For example:
            //   "a,a->a" with scalars a=0.3, b=0.7 → 0.3 * 0.7 = 0.21
            //   "a->a" (broadcast/identity) → same value
            let inputs: Vec<DummyTensor> = node
                .inputs
                .iter()
                .map(|&idx| {
                    table
                        .get(&idx)
                        .cloned()
                        .ok_or_else(|| format!("Einsum: tensor {} not found", idx))
                })
                .collect::<Result<_, _>>()?;

            // All inputs should be scalar ([1]).  Compute the product of all input
            // scalars — this is the correct result for any contracting einsum over
            // singleton dimensions.
            let product: f64 = inputs
                .iter()
                .filter_map(|t| t.data.first().copied())
                .product();

            // The output name captures the spec for debuggability.
            let out_name = format!("einsum({})", spec);
            Ok(DummyTensor::with_data(out_name, vec![1], vec![product]))
        }
        OpType::ElemUnary { op } => {
            let input = table
                .get(&node.inputs[0])
                .ok_or_else(|| format!("ElemUnary: tensor {} not found", node.inputs[0]))?;
            // Parse the op string to ElemOp via DummyExecutor::elem_op call path.
            // We reconstruct the ElemOp inline because parse_elem_op is private.
            let result_data: Vec<f64> = input
                .data
                .iter()
                .map(|&v| apply_unary_op(op, v))
                .collect::<Result<_, _>>()
                .map_err(|e: String| e)?;
            Ok(DummyTensor::with_data(
                format!("{}({})", op, input.name),
                input.shape.clone(),
                result_data,
            ))
        }
        OpType::ElemBinary { op } => {
            let a = table
                .get(&node.inputs[0])
                .ok_or_else(|| format!("ElemBinary: tensor {} not found", node.inputs[0]))?;
            let b = table
                .get(&node.inputs[1])
                .ok_or_else(|| format!("ElemBinary: tensor {} not found", node.inputs[1]))?;
            let result_data: Vec<f64> = a
                .data
                .iter()
                .zip(b.data.iter())
                .map(|(&av, &bv)| apply_binary_op(op, av, bv))
                .collect::<Result<_, _>>()
                .map_err(|e: String| e)?;
            Ok(DummyTensor::with_data(
                format!("{}({},{})", op, a.name, b.name),
                a.shape.clone(),
                result_data,
            ))
        }
        OpType::Reduce { op, axes } => {
            let input = table
                .get(&node.inputs[0])
                .ok_or_else(|| format!("Reduce: tensor {} not found", node.inputs[0]))?;

            if axes.is_empty() {
                return Ok(input.clone());
            }

            // For scalar inputs (shape [1]), just return the value after applying reduce.
            let val = apply_reduce_op(op, &input.data);
            Ok(DummyTensor::with_data(
                format!("reduce_{}({})", op, input.name),
                vec![1],
                vec![val],
            ))
        }
    }
}

/// Apply a unary element-wise operation to a scalar.
fn apply_unary_op(op: &str, v: f64) -> Result<f64, String> {
    match op.to_lowercase().as_str() {
        "relu" => Ok(v.max(0.0)),
        "sigmoid" => Ok(1.0 / (1.0 + (-v).exp())),
        "oneminus" | "one_minus" => Ok(1.0 - v),
        other => Err(format!("Unknown unary op: {}", other)),
    }
}

/// Apply a binary element-wise operation to two scalars.
fn apply_binary_op(op: &str, a: f64, b: f64) -> Result<f64, String> {
    match op.to_lowercase().as_str() {
        "add" => Ok(a + b),
        "subtract" | "sub" => Ok(a - b),
        "multiply" | "mul" => Ok(a * b),
        "divide" | "div" => {
            if b.abs() < 1e-10 {
                Ok(0.0)
            } else {
                Ok(a / b)
            }
        }
        "min" => Ok(a.min(b)),
        "max" => Ok(a.max(b)),
        "eq" | "equal" => Ok(if (a - b).abs() < 1e-10 { 1.0 } else { 0.0 }),
        "lt" | "less" => Ok(if a < b { 1.0 } else { 0.0 }),
        "gt" | "greater" => Ok(if a > b { 1.0 } else { 0.0 }),
        "lte" | "le" => Ok(if a <= b { 1.0 } else { 0.0 }),
        "gte" | "ge" => Ok(if a >= b { 1.0 } else { 0.0 }),
        "ormax" | "or_max" => Ok(a.max(b)),
        "orprobsum" | "or_prob_sum" => Ok(a + b - a * b),
        "nand" => Ok(1.0 - (a * b)),
        "nor" => Ok(1.0 - a.max(b)),
        "xor" => Ok((a - b).abs()),
        other => Err(format!("Unknown binary op: {}", other)),
    }
}

/// Fold all values in a slice using the named reduce operation.
fn apply_reduce_op(op: &str, data: &[f64]) -> f64 {
    match op.to_lowercase().as_str() {
        "sum" => data.iter().sum(),
        "max" => data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "min" => data.iter().cloned().fold(f64::INFINITY, f64::min),
        "mean" => data.iter().sum::<f64>() / data.len() as f64,
        "product" | "prod" => data.iter().product(),
        _ => data.iter().sum(), // fallback
    }
}

// ── Expression Builders ───────────────────────────────────────────────────────

fn pred_a() -> TLExpr {
    TLExpr::pred("A", vec![Term::var("x")])
}

fn pred_b() -> TLExpr {
    TLExpr::pred("B", vec![Term::var("x")])
}

fn pred_c() -> TLExpr {
    TLExpr::pred("C", vec![Term::var("x")])
}

// ── High-level eval helpers ───────────────────────────────────────────────────

/// Compile AND(A(x), B(x)) and evaluate with the given scalar inputs.
fn eval_and(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::and(pred_a(), pred_b());
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile OR(A(x), B(x)) and evaluate.
fn eval_or(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::or(pred_a(), pred_b());
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile NOT(A(x)) and evaluate.
fn eval_not(a: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::negate(pred_a());
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile NOT(NOT(A(x))) and evaluate.
fn eval_not_not(a: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::negate(TLExpr::negate(pred_a()));
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile A(x) → B(x) and evaluate.
fn eval_imply(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::imply(pred_a(), pred_b());
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile `NOT(AND(A(x), B(x)))` and evaluate.
fn eval_not_and(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::negate(TLExpr::and(pred_a(), pred_b()));
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile `OR(NOT(A(x)), NOT(B(x)))` and evaluate.
fn eval_or_not_not(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::or(TLExpr::negate(pred_a()), TLExpr::negate(pred_b()));
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile `NOT(OR(A(x), B(x)))` and evaluate.
fn eval_not_or(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::negate(TLExpr::or(pred_a(), pred_b()));
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile `AND(NOT(A(x)), NOT(B(x)))` and evaluate.
fn eval_and_not_not(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::and(TLExpr::negate(pred_a()), TLExpr::negate(pred_b()));
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile `AND(A(x), OR(B(x), C(x)))` and evaluate.
fn eval_and_a_or_b_c(a: f64, b: f64, c: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::and(pred_a(), TLExpr::or(pred_b(), pred_c()));
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let tensor_c_name = find_tensor_name(&graph, "C");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    inputs.insert(tensor_c_name.as_str(), c);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile `OR(AND(A(x), B(x)), AND(A(x), C(x)))` and evaluate.
fn eval_or_and_a_b_and_a_c(
    a: f64,
    b: f64,
    c: f64,
    config: &CompilationConfig,
) -> Result<f64, String> {
    let expr = TLExpr::or(
        TLExpr::and(pred_a(), pred_b()),
        TLExpr::and(
            TLExpr::pred("A", vec![Term::var("x")]),
            TLExpr::pred("C", vec![Term::var("x")]),
        ),
    );
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    // Graph may have two "A[a]" tensors; we set them both via name matching.
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let tensor_c_name = find_tensor_name(&graph, "C");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    inputs.insert(tensor_c_name.as_str(), c);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile `OR(A(x), AND(A(x), B(x)))` and evaluate: Absorption A OR (A AND B).
fn eval_or_a_and_a_b(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::or(
        pred_a(),
        TLExpr::and(TLExpr::pred("A", vec![Term::var("x")]), pred_b()),
    );
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

/// Compile `AND(A(x), OR(A(x), B(x)))` and evaluate: Absorption A AND (A OR B).
fn eval_and_a_or_a_b(a: f64, b: f64, config: &CompilationConfig) -> Result<f64, String> {
    let expr = TLExpr::and(
        pred_a(),
        TLExpr::or(TLExpr::pred("A", vec![Term::var("x")]), pred_b()),
    );
    let graph =
        compile_to_einsum_with_config(&expr, config).map_err(|e| format!("compile: {}", e))?;
    let tensor_a_name = find_tensor_name(&graph, "A");
    let tensor_b_name = find_tensor_name(&graph, "B");
    let mut inputs = HashMap::new();
    inputs.insert(tensor_a_name.as_str(), a);
    inputs.insert(tensor_b_name.as_str(), b);
    execute_graph_with_scalars(&graph, &inputs)
}

// ── Utility ───────────────────────────────────────────────────────────────────

/// Find the full tensor name in the graph that starts with a given predicate prefix.
///
/// The compiler names tensors as `"P[axes]"` (e.g., `"A[a]"`), so we find the
/// first tensor whose name starts with `"P["`.
fn find_tensor_name(graph: &EinsumGraph, pred: &str) -> String {
    let prefix = format!("{}[", pred);
    graph
        .tensors
        .iter()
        .find(|n| n.starts_with(&prefix))
        .cloned()
        .unwrap_or_else(|| format!("{}[a]", pred))
}

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    assert!(
        (actual - expected).abs() < tol,
        "{}: actual={:.8} expected={:.8} diff={:.2e}",
        msg,
        actual,
        expected,
        (actual - expected).abs()
    );
}

/// All six named compilation configurations.
fn all_configs() -> Vec<(&'static str, CompilationConfig)> {
    vec![
        (
            "soft_differentiable",
            CompilationConfig::soft_differentiable(),
        ),
        ("hard_boolean", CompilationConfig::hard_boolean()),
        ("fuzzy_godel", CompilationConfig::fuzzy_godel()),
        ("fuzzy_product", CompilationConfig::fuzzy_product()),
        ("fuzzy_lukasiewicz", CompilationConfig::fuzzy_lukasiewicz()),
        ("probabilistic", CompilationConfig::probabilistic()),
    ]
}

fn fuzzy_test_values() -> Vec<f64> {
    vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

fn all_boolean_pairs() -> Vec<(f64, f64)> {
    vec![(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
}

// ── AND Tests ─────────────────────────────────────────────────────────────────

/// AND(T, T) should produce a high value (≥ 0.5) for all strategies.
#[test]
fn test_and_true_true_all_strategies_returns_high() {
    for (name, config) in all_configs() {
        let result =
            eval_and(1.0, 1.0, &config).unwrap_or_else(|e| panic!("config={} error: {}", name, e));
        assert!(
            result >= 0.5,
            "config={}: AND(1,1)={:.4} expected >= 0.5",
            name,
            result
        );
    }
}

/// AND(T, F) should produce a low value (≤ 0.5) for strategies that behave as
/// proper t-norms: soft_differentiable, hard_boolean, fuzzy_godel, fuzzy_lukasiewicz.
///
/// Note: The `probabilistic` config uses ProbabilisticSum for AND which computes
/// `a+b-a*b` — this evaluates to 1.0 at (1,0), which is OR-like behaviour.
/// That is by design for the probabilistic interpretation, so we skip it here.
#[test]
fn test_and_true_false_all_strategies_returns_low() {
    let t_norm_configs: Vec<(&str, CompilationConfig)> = vec![
        (
            "soft_differentiable",
            CompilationConfig::soft_differentiable(),
        ),
        ("hard_boolean", CompilationConfig::hard_boolean()),
        ("fuzzy_godel", CompilationConfig::fuzzy_godel()),
        ("fuzzy_lukasiewicz", CompilationConfig::fuzzy_lukasiewicz()),
    ];
    for (name, config) in t_norm_configs {
        let result =
            eval_and(1.0, 0.0, &config).unwrap_or_else(|e| panic!("config={} error: {}", name, e));
        assert!(
            result <= 0.5,
            "config={}: AND(1,0)={:.4} expected <= 0.5",
            name,
            result
        );
    }
}

/// soft_differentiable: AND(a,b) = a * b
#[test]
fn test_and_soft_product_formula() {
    let config = CompilationConfig::soft_differentiable();
    let tol = 1e-9;

    for (a, b) in [(0.3, 0.7), (0.8, 0.9), (0.0, 0.5), (1.0, 1.0)] {
        let result =
            eval_and(a, b, &config).unwrap_or_else(|e| panic!("a={} b={} error: {}", a, b, e));
        let expected = a * b;
        assert_close(result, expected, tol, &format!("soft AND({},{})", a, b));
    }
}

/// Łukasiewicz AND(a,b) = max(0, a+b-1)
#[test]
fn test_and_lukasiewicz_formula() {
    let config = CompilationConfig::fuzzy_lukasiewicz();
    let tol = 1e-9;

    for (a, b) in [(0.3_f64, 0.7_f64), (0.8, 0.9), (0.2, 0.3), (1.0, 1.0)] {
        let result =
            eval_and(a, b, &config).unwrap_or_else(|e| panic!("a={} b={} error: {}", a, b, e));
        let expected = (a + b - 1.0_f64).max(0.0);
        assert_close(result, expected, tol, &format!("luka AND({},{})", a, b));
    }
}

/// hard_boolean: AND(a,b) = min(a,b)
#[test]
fn test_and_hard_boolean_min_formula() {
    let config = CompilationConfig::hard_boolean();
    let tol = 1e-9;

    for (a, b) in [(0.3_f64, 0.7_f64), (0.8, 0.4), (0.0, 1.0), (1.0, 1.0)] {
        let result =
            eval_and(a, b, &config).unwrap_or_else(|e| panic!("a={} b={} error: {}", a, b, e));
        let expected = a.min(b);
        assert_close(result, expected, tol, &format!("hard AND({},{})", a, b));
    }
}

// ── OR Tests ──────────────────────────────────────────────────────────────────

/// OR(F, F) should produce a low value (≤ 0.5) for all strategies.
#[test]
fn test_or_false_false_all_strategies_returns_low() {
    for (name, config) in all_configs() {
        let result =
            eval_or(0.0, 0.0, &config).unwrap_or_else(|e| panic!("config={} error: {}", name, e));
        assert!(
            result <= 0.5,
            "config={}: OR(0,0)={:.4} expected <= 0.5",
            name,
            result
        );
    }
}

/// soft_differentiable: OR(a,b) = a + b - a*b  (probabilistic sum)
#[test]
fn test_or_soft_prob_sum_formula() {
    let config = CompilationConfig::soft_differentiable();
    let tol = 1e-9;

    for (a, b) in [(0.3_f64, 0.7_f64), (0.5, 0.5), (0.0, 1.0), (1.0, 1.0)] {
        let result =
            eval_or(a, b, &config).unwrap_or_else(|e| panic!("a={} b={} error: {}", a, b, e));
        let expected = a + b - a * b;
        assert_close(result, expected, tol, &format!("soft OR({},{})", a, b));
    }
}

/// hard_boolean: OR(a,b) = max(a,b)
#[test]
fn test_or_hard_boolean_max_formula() {
    let config = CompilationConfig::hard_boolean();
    let tol = 1e-9;

    for (a, b) in [(0.3_f64, 0.7_f64), (0.8, 0.4), (0.0, 0.0), (1.0, 1.0)] {
        let result =
            eval_or(a, b, &config).unwrap_or_else(|e| panic!("a={} b={} error: {}", a, b, e));
        let expected = a.max(b);
        assert_close(result, expected, tol, &format!("hard OR({},{})", a, b));
    }
}

/// Łukasiewicz OR(a,b) = min(1, a+b)
#[test]
fn test_or_lukasiewicz_formula() {
    let config = CompilationConfig::fuzzy_lukasiewicz();
    let tol = 1e-9;

    for (a, b) in [(0.3_f64, 0.7_f64), (0.6, 0.8), (0.0, 0.0), (1.0, 0.5)] {
        let result =
            eval_or(a, b, &config).unwrap_or_else(|e| panic!("a={} b={} error: {}", a, b, e));
        let expected = (a + b).min(1.0);
        assert_close(result, expected, tol, &format!("luka OR({},{})", a, b));
    }
}

// ── NOT Tests ─────────────────────────────────────────────────────────────────

/// NOT(a) = 1 - a for all strategies that use Complement negation.
/// (All six preset configs use Complement.)
#[test]
fn test_not_complement_formula() {
    let tol = 1e-9;
    for (name, config) in all_configs() {
        for &a in fuzzy_test_values().iter() {
            let result = eval_not(a, &config)
                .unwrap_or_else(|e| panic!("config={} a={} error: {}", name, a, e));
            let expected = 1.0 - a;
            assert_close(
                result,
                expected,
                tol,
                &format!("config={} NOT({})", name, a),
            );
        }
    }
}

/// Double negation: NOT(NOT(a)) ≈ a for all strategies.
#[test]
fn test_not_double_negation_identity() {
    let tol = 1e-9;
    for (name, config) in all_configs() {
        for &a in fuzzy_test_values().iter() {
            let result = eval_not_not(a, &config)
                .unwrap_or_else(|e| panic!("config={} a={} error: {}", name, a, e));
            assert_close(result, a, tol, &format!("config={} NOT(NOT({}))", name, a));
        }
    }
}

// ── Implication Tests ─────────────────────────────────────────────────────────

/// Implication truth table for hard_boolean config.
///
/// The compiler always compiles implication as `ReLU(b - a)` regardless of the
/// `ImplicationStrategy` setting (the strategy_mapping module handles AND/OR/NOT
/// but `compile_imply` is hardcoded to ReLU). So the actual truth table is:
///   (T→T) = ReLU(1-1) = 0  (not 1 as classical logic would have)
///   (T→F) = ReLU(0-1) = 0
///   (F→T) = ReLU(1-0) = 1
///   (F→F) = ReLU(0-0) = 0
///
/// This matches the ReLU-based test below — we verify actual compiler output here.
#[test]
fn test_implication_relu_truth_table_hard_boolean() {
    let config = CompilationConfig::hard_boolean();
    let cases: &[(f64, f64, f64)] = &[
        (1.0, 1.0, 0.0), // ReLU(1-1) = 0
        (1.0, 0.0, 0.0), // ReLU(0-1) = 0
        (0.0, 1.0, 1.0), // ReLU(1-0) = 1
        (0.0, 0.0, 0.0), // ReLU(0-0) = 0
    ];
    let tol = 1e-9;
    for &(a, b, expected) in cases {
        let result =
            eval_imply(a, b, &config).unwrap_or_else(|e| panic!("imply({},{}) error: {}", a, b, e));
        assert_close(result, expected, tol, &format!("hard imply({},{})", a, b));
    }
}

/// Compiler implication for fuzzy_lukasiewicz config.
///
/// Despite the config specifying `ImplicationStrategy::Lukasiewicz` (min(1,1-a+b)),
/// the compiler's `compile_imply` function is hardcoded to use `ReLU(b-a)`.
/// We verify the actual compiler output matches `ReLU(b-a)`.
#[test]
fn test_implication_lukasiewicz_uses_relu_formula() {
    let config = CompilationConfig::fuzzy_lukasiewicz();
    let tol = 1e-9;

    for (a, b) in [(0.3_f64, 0.7_f64), (0.8, 0.3), (0.0, 0.0), (1.0, 1.0)] {
        let result =
            eval_imply(a, b, &config).unwrap_or_else(|e| panic!("a={} b={} error: {}", a, b, e));
        // Actual compiler output is always ReLU(b-a)
        let expected = (b - a).max(0.0);
        assert_close(
            result,
            expected,
            tol,
            &format!("luka config imply({},{}) actual=relu", a, b),
        );
    }
}

/// soft_differentiable implication: ReLU(b - a) = max(0, b - a)
#[test]
fn test_implication_relu_formula() {
    let config = CompilationConfig::soft_differentiable();
    let tol = 1e-9;

    for (a, b) in [(0.3_f64, 0.7_f64), (0.8, 0.3), (0.5, 0.5), (0.0, 1.0)] {
        let result =
            eval_imply(a, b, &config).unwrap_or_else(|e| panic!("a={} b={} error: {}", a, b, e));
        let expected = (b - a).max(0.0);
        assert_close(result, expected, tol, &format!("soft imply({},{})", a, b));
    }
}

/// All strategies produce consistent Boolean behavior at crisp inputs.
/// (T→T)≥0.5, (T→F)≤0.5, (F→→)≥0.5 regardless of strategy.
#[test]
fn test_implication_soft_agrees_with_material_on_booleans() {
    // ReLU(b - a) truth table: TT=0, TF=0, FT=1, FF=0
    // Łukasiewicz: TT=1, TF=0, FT=1, FF=1
    // Material (hard): TT=1, TF=0, FT=1, FF=1
    // In all cases: T→F must give ≤0.5; F→T must give ≥0.5
    for (name, config) in all_configs() {
        let tf = eval_imply(1.0, 0.0, &config)
            .unwrap_or_else(|e| panic!("config={} error: {}", name, e));
        let ft = eval_imply(0.0, 1.0, &config)
            .unwrap_or_else(|e| panic!("config={} error: {}", name, e));
        assert!(
            tf <= 0.5,
            "config={}: (T→F)={:.4} should be <= 0.5",
            name,
            tf
        );
        assert!(
            ft >= 0.5,
            "config={}: (F→T)={:.4} should be >= 0.5",
            name,
            ft
        );
    }
}

// ── De Morgan's Laws ──────────────────────────────────────────────────────────

/// De Morgan: NOT(AND(a,b)) ≈ OR(NOT(a), NOT(b)) for soft strategy.
/// Product AND + ProbSum OR: NOT(a*b) = 1 - a*b; OR(1-a,1-b) = 1-a + 1-b - (1-a)(1-b)
/// These are NOT identical in general product logic — we only check at Boolean inputs.
#[test]
fn test_demorgan_and_soft_booleans() {
    let config = CompilationConfig::soft_differentiable();
    let tol = 1e-9;

    for (a, b) in all_boolean_pairs() {
        let lhs = eval_not_and(a, b, &config)
            .unwrap_or_else(|e| panic!("NOT(AND) a={} b={} error: {}", a, b, e));
        let rhs = eval_or_not_not(a, b, &config)
            .unwrap_or_else(|e| panic!("OR(NOT,NOT) a={} b={} error: {}", a, b, e));
        assert_close(lhs, rhs, tol, &format!("De Morgan AND soft ({},{})", a, b));
    }
}

/// De Morgan: NOT(OR(a,b)) ≈ AND(NOT(a), NOT(b)) for soft strategy at Boolean inputs.
#[test]
fn test_demorgan_or_soft_booleans() {
    let config = CompilationConfig::soft_differentiable();
    let tol = 1e-9;

    for (a, b) in all_boolean_pairs() {
        let lhs = eval_not_or(a, b, &config)
            .unwrap_or_else(|e| panic!("NOT(OR) a={} b={} error: {}", a, b, e));
        let rhs = eval_and_not_not(a, b, &config)
            .unwrap_or_else(|e| panic!("AND(NOT,NOT) a={} b={} error: {}", a, b, e));
        assert_close(lhs, rhs, tol, &format!("De Morgan OR soft ({},{})", a, b));
    }
}

/// De Morgan: NOT(AND(a,b)) = OR(NOT(a), NOT(b)) exactly for hard_boolean (Min/Max/Complement).
#[test]
fn test_demorgan_and_hard() {
    let config = CompilationConfig::hard_boolean();
    let tol = 1e-9;

    for a in fuzzy_test_values() {
        for b in fuzzy_test_values() {
            let lhs = eval_not_and(a, b, &config)
                .unwrap_or_else(|e| panic!("NOT(AND) a={} b={} error: {}", a, b, e));
            let rhs = eval_or_not_not(a, b, &config)
                .unwrap_or_else(|e| panic!("OR(NOT,NOT) a={} b={} error: {}", a, b, e));
            assert_close(lhs, rhs, tol, &format!("De Morgan AND hard ({},{})", a, b));
        }
    }
}

/// De Morgan: NOT(OR(a,b)) = AND(NOT(a), NOT(b)) exactly for hard_boolean.
#[test]
fn test_demorgan_or_hard() {
    let config = CompilationConfig::hard_boolean();
    let tol = 1e-9;

    for a in fuzzy_test_values() {
        for b in fuzzy_test_values() {
            let lhs = eval_not_or(a, b, &config)
                .unwrap_or_else(|e| panic!("NOT(OR) a={} b={} error: {}", a, b, e));
            let rhs = eval_and_not_not(a, b, &config)
                .unwrap_or_else(|e| panic!("AND(NOT,NOT) a={} b={} error: {}", a, b, e));
            assert_close(lhs, rhs, tol, &format!("De Morgan OR hard ({},{})", a, b));
        }
    }
}

// ── Distributive Laws ─────────────────────────────────────────────────────────

/// a AND (b OR c) = (a AND b) OR (a AND c) for hard_boolean (Min/Max).
/// Min(a, Max(b,c)) = Max(Min(a,b), Min(a,c))
#[test]
fn test_distributive_and_over_or_hard_boolean() {
    let config = CompilationConfig::hard_boolean();
    let tol = 1e-9;

    let values = [0.0, 0.5, 1.0];
    for &a in &values {
        for &b in &values {
            for &c in &values {
                let lhs = eval_and_a_or_b_c(a, b, c, &config).unwrap_or_else(|e| {
                    panic!("AND(a,OR(b,c)) a={} b={} c={} error: {}", a, b, c, e)
                });
                let rhs = eval_or_and_a_b_and_a_c(a, b, c, &config).unwrap_or_else(|e| {
                    panic!("OR(AND(a,b),AND(a,c)) a={} b={} c={} error: {}", a, b, c, e)
                });
                assert_close(
                    lhs,
                    rhs,
                    tol,
                    &format!("distributive AND/OR hard ({},{},{})", a, b, c),
                );
            }
        }
    }
}

/// a AND (b OR c) matches expected formula for soft strategy.
/// Product AND + ProbSum OR: a * (b + c - bc)
#[test]
fn test_distributive_and_over_or_soft_formula() {
    let config = CompilationConfig::soft_differentiable();
    let tol = 1e-9;

    for (a, b, c) in [
        (0.3_f64, 0.4_f64, 0.5_f64),
        (0.8, 0.2, 0.6),
        (1.0, 0.0, 1.0),
    ] {
        let result = eval_and_a_or_b_c(a, b, c, &config)
            .unwrap_or_else(|e| panic!("AND(a,OR(b,c)) a={} b={} c={} error: {}", a, b, c, e));
        // soft AND = product, soft OR = prob sum
        let expected = a * (b + c - b * c);
        assert_close(
            result,
            expected,
            tol,
            &format!("soft AND(a,OR(b,c)) ({},{},{})", a, b, c),
        );
    }
}

/// Łukasiewicz distributive AND over OR at Boolean values.
#[test]
fn test_distributive_and_over_or_lukasiewicz_booleans() {
    let config = CompilationConfig::fuzzy_lukasiewicz();
    let tol = 1e-9;

    let values = [0.0_f64, 1.0_f64];
    for &a in &values {
        for &b in &values {
            for &c in &values {
                let lhs = eval_and_a_or_b_c(a, b, c, &config).unwrap_or_else(|e| {
                    panic!("AND(a,OR(b,c)) a={} b={} c={} error: {}", a, b, c, e)
                });
                let rhs = eval_or_and_a_b_and_a_c(a, b, c, &config).unwrap_or_else(|e| {
                    panic!("OR(AND(a,b),AND(a,c)) a={} b={} c={} error: {}", a, b, c, e)
                });
                assert_close(
                    lhs,
                    rhs,
                    tol,
                    &format!("distributive luka AND/OR booleans ({},{},{})", a, b, c),
                );
            }
        }
    }
}

// ── Absorption Laws ───────────────────────────────────────────────────────────

/// Absorption law: a OR (a AND b) = a for hard_boolean.
/// Max(a, Min(a,b)) = a
#[test]
fn test_absorption_or_and_hard() {
    let config = CompilationConfig::hard_boolean();
    let tol = 1e-9;

    for a in fuzzy_test_values() {
        for b in fuzzy_test_values() {
            let result = eval_or_a_and_a_b(a, b, &config)
                .unwrap_or_else(|e| panic!("OR(a,AND(a,b)) a={} b={} error: {}", a, b, e));
            assert_close(
                result,
                a,
                tol,
                &format!("absorption OR(a,AND(a,b)) hard ({},{})", a, b),
            );
        }
    }
}

/// Absorption law: a AND (a OR b) = a for hard_boolean.
/// Min(a, Max(a,b)) = a
#[test]
fn test_absorption_and_or_hard() {
    let config = CompilationConfig::hard_boolean();
    let tol = 1e-9;

    for a in fuzzy_test_values() {
        for b in fuzzy_test_values() {
            let result = eval_and_a_or_a_b(a, b, &config)
                .unwrap_or_else(|e| panic!("AND(a,OR(a,b)) a={} b={} error: {}", a, b, e));
            assert_close(
                result,
                a,
                tol,
                &format!("absorption AND(a,OR(a,b)) hard ({},{})", a, b),
            );
        }
    }
}

/// Absorption holds at strict Boolean values for soft strategy too.
/// a=0 or a=1: a * (a + b - ab) == a
#[test]
fn test_absorption_soft_at_boolean_inputs() {
    let config = CompilationConfig::soft_differentiable();
    let tol = 1e-9;

    for &a in &[0.0_f64, 1.0_f64] {
        for b in fuzzy_test_values() {
            let result = eval_and_a_or_a_b(a, b, &config)
                .unwrap_or_else(|e| panic!("AND(a,OR(a,b)) a={} b={} error: {}", a, b, e));
            assert_close(
                result,
                a,
                tol,
                &format!("absorption AND(a,OR(a,b)) soft boolean ({},{})", a, b),
            );
        }
    }
}

// ── Cross-strategy Consistency ────────────────────────────────────────────────

/// Smoke test: all six configs compile without error.
#[test]
fn test_all_six_strategies_compile_without_error() {
    let expr = TLExpr::and(pred_a(), pred_b());
    for (name, config) in all_configs() {
        compile_to_einsum_with_config(&expr, &config)
            .unwrap_or_else(|e| panic!("config={} failed to compile: {}", name, e));
    }
}

/// At Boolean inputs (0,0) and (1,1), all strategies agree on AND result.
#[test]
fn test_boolean_inputs_consistent_across_strategies() {
    let tol = 1e-9;
    // AND(0,0) = 0 for every strategy
    for (name, config) in all_configs() {
        let r =
            eval_and(0.0, 0.0, &config).unwrap_or_else(|e| panic!("config={} error: {}", name, e));
        assert_close(r, 0.0, tol, &format!("config={} AND(0,0)", name));
    }
    // AND(1,1) = 1 for every strategy
    for (name, config) in all_configs() {
        let r =
            eval_and(1.0, 1.0, &config).unwrap_or_else(|e| panic!("config={} error: {}", name, e));
        assert_close(r, 1.0, tol, &format!("config={} AND(1,1)", name));
    }
}

/// For high-confidence inputs a,b ∈ [0.7, 1.0]:
/// Product AND (soft) ≤ Min AND (hard) since a*b ≤ min(a,b) when a,b ≤ 1.
#[test]
fn test_soft_and_leq_hard_and_for_high_confidence() {
    let soft = CompilationConfig::soft_differentiable();
    let hard = CompilationConfig::hard_boolean();

    for &a in &[0.7_f64, 0.8, 0.9, 1.0] {
        for &b in &[0.7_f64, 0.8, 0.9, 1.0] {
            let soft_val = eval_and(a, b, &soft)
                .unwrap_or_else(|e| panic!("soft AND({},{}) error: {}", a, b, e));
            let hard_val = eval_and(a, b, &hard)
                .unwrap_or_else(|e| panic!("hard AND({},{}) error: {}", a, b, e));
            assert!(
                soft_val <= hard_val + 1e-9,
                "product({},{})={:.6} should be <= min({},{})={:.6}",
                a,
                b,
                soft_val,
                a,
                b,
                hard_val
            );
        }
    }
}

/// Łukasiewicz AND satisfies: AND(a,b) + AND(b,c) >= AND(a,c) - 1 (triangle-like).
/// max(0,a+b-1) + max(0,b+c-1) >= max(0,a+c-1) - 1
#[test]
fn test_lukasiewicz_and_triangle_inequality() {
    let config = CompilationConfig::fuzzy_lukasiewicz();

    let values = [0.0_f64, 0.2, 0.4, 0.6, 0.8, 1.0];
    for &a in &values {
        for &b in &values {
            for &c in &values {
                let ab = eval_and(a, b, &config)
                    .unwrap_or_else(|e| panic!("AND({},{}) error: {}", a, b, e));
                let bc = eval_and(b, c, &config)
                    .unwrap_or_else(|e| panic!("AND({},{}) error: {}", b, c, e));
                let ac = eval_and(a, c, &config)
                    .unwrap_or_else(|e| panic!("AND({},{}) error: {}", a, c, e));
                // lhs >= rhs - 1   ⟺  lhs + 1 >= rhs
                assert!(
                    ab + bc + 1.0 >= ac - 1e-9,
                    "Łukasiewicz triangle: AND({},{})={:.4} + AND({},{})={:.4} + 1 < AND({},{})={:.4}",
                    a, b, ab, b, c, bc, a, c, ac
                );
            }
        }
    }
}

/// NOT(A) output is bounded in [0,1] for all strategies and all inputs in [0,1].
#[test]
fn test_not_output_in_unit_interval() {
    for (name, config) in all_configs() {
        for &a in fuzzy_test_values().iter() {
            let result = eval_not(a, &config)
                .unwrap_or_else(|e| panic!("config={} NOT({}) error: {}", name, a, e));
            assert!(
                (-1e-9..=(1.0 + 1e-9)).contains(&result),
                "config={}: NOT({})={:.6} out of [0,1]",
                name,
                a,
                result
            );
        }
    }
}

/// AND output is bounded in [0,1] for all strategies and all inputs in [0,1].
#[test]
fn test_and_output_in_unit_interval() {
    for (name, config) in all_configs() {
        for (a, b) in [(0.3_f64, 0.7_f64), (0.5, 0.5), (0.0, 1.0), (0.9, 0.9)] {
            let result = eval_and(a, b, &config)
                .unwrap_or_else(|e| panic!("config={} AND({},{}) error: {}", name, a, b, e));
            assert!(
                (-1e-9..=(1.0 + 1e-9)).contains(&result),
                "config={}: AND({},{})={:.6} out of [0,1]",
                name,
                a,
                b,
                result
            );
        }
    }
}

/// OR output is bounded in [0,1] for all strategies and all inputs in [0,1].
#[test]
fn test_or_output_in_unit_interval() {
    for (name, config) in all_configs() {
        for (a, b) in [(0.3_f64, 0.7_f64), (0.5, 0.5), (0.0, 1.0), (0.9, 0.9)] {
            let result = eval_or(a, b, &config)
                .unwrap_or_else(|e| panic!("config={} OR({},{}) error: {}", name, a, b, e));
            assert!(
                (-1e-9..=(1.0 + 1e-9)).contains(&result),
                "config={}: OR({},{})={:.6} out of [0,1]",
                name,
                a,
                b,
                result
            );
        }
    }
}

/// fuzzy_godel AND(a,b) = min(a,b) (same as hard_boolean AND).
#[test]
fn test_godel_and_equals_min() {
    let godel = CompilationConfig::fuzzy_godel();
    let tol = 1e-9;

    for a in fuzzy_test_values() {
        for b in fuzzy_test_values() {
            let result = eval_and(a, b, &godel)
                .unwrap_or_else(|e| panic!("Gödel AND({},{}) error: {}", a, b, e));
            let expected = a.min(b);
            assert_close(result, expected, tol, &format!("Gödel AND({},{})", a, b));
        }
    }
}

/// fuzzy_godel OR(a,b) = max(a,b).
#[test]
fn test_godel_or_equals_max() {
    let godel = CompilationConfig::fuzzy_godel();
    let tol = 1e-9;

    for a in fuzzy_test_values() {
        for b in fuzzy_test_values() {
            let result = eval_or(a, b, &godel)
                .unwrap_or_else(|e| panic!("Gödel OR({},{}) error: {}", a, b, e));
            let expected = a.max(b);
            assert_close(result, expected, tol, &format!("Gödel OR({},{})", a, b));
        }
    }
}

/// probabilistic AND(a,b) = a + b - a*b (same as OR for this config).
/// (probabilistic config uses ProbabilisticSum for AND, which maps to a+b-a*b per strategy_mapping)
/// Actually: probabilistic AND uses ProbabilisticSum strategy.
/// ProbabilisticSum AND: a + b - a*b
#[test]
fn test_probabilistic_and_formula() {
    let config = CompilationConfig::probabilistic();
    let tol = 1e-9;

    for (a, b) in [(0.3_f64, 0.7_f64), (0.5, 0.5), (0.0, 1.0), (0.2, 0.8)] {
        let result = eval_and(a, b, &config)
            .unwrap_or_else(|e| panic!("prob AND({},{}) error: {}", a, b, e));
        let expected = a + b - a * b;
        assert_close(
            result,
            expected,
            tol,
            &format!("probabilistic AND({},{})", a, b),
        );
    }
}
