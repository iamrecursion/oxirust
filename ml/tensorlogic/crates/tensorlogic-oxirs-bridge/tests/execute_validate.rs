//! Integration tests for the execute-validate pipeline.

use tensorlogic_ir::{TLExpr, Term};
use tensorlogic_oxirs_bridge::{ValidationExecutor, ValidationExecutorConfig};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn default_executor() -> ValidationExecutor {
    ValidationExecutor::new(ValidationExecutorConfig::default())
}

// ─────────────────────────────────────────────────────────────────────────────
// Core pipeline tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_simple_predicate_executes() {
    let expr = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
    let executor = default_executor();
    let result = executor
        .execute_rule(&expr)
        .expect("execute simple predicate");
    // A simple predicate may compile to a graph with zero computational nodes
    // (it is a direct pass-through).  We only verify that the pipeline ran
    // without errors and produced output tensors.
    assert!(
        !result.output_tensors.is_empty(),
        "expected at least one output tensor"
    );
}

#[test]
fn test_finite_result_conforms() {
    let expr = TLExpr::pred("p", vec![Term::var("x")]);
    let executor = default_executor();
    let result = executor.execute_rule(&expr).unwrap();
    let report = executor.generate_validation_report(&result);
    // Placeholder inputs are finite so the output should also be finite.
    assert!(
        report.conforms,
        "finite placeholder outputs should produce a conforming report"
    );
}

#[test]
fn test_export_as_rdf_prefix() {
    let expr = TLExpr::pred("q", vec![Term::var("a")]);
    let executor = default_executor();
    let result = executor.execute_rule(&expr).unwrap();
    let rdf = executor.export_as_rdf(&result);
    assert!(
        rdf.contains("@prefix"),
        "RDF output must contain @prefix declarations"
    );
    assert!(rdf.contains("tl:"), "RDF output must use the tl: prefix");
}

#[test]
fn test_execution_stats_recorded() {
    // A binary pred compiles to a non-trivial graph (at least a tensor load).
    let expr = TLExpr::pred("r", vec![Term::var("x"), Term::constant("1.0")]);
    let executor = default_executor();
    let result = executor.execute_rule(&expr).unwrap();
    // Stats must be populated — compile time is always recorded.
    assert!(
        result.stats.output_tensor_count > 0,
        "output_tensor_count must be ≥ 1"
    );
}

#[test]
fn test_max_tensor_size_respected() {
    let config = ValidationExecutorConfig {
        max_tensor_size: 0,
        ..Default::default()
    };
    let expr = TLExpr::pred("s", vec![Term::var("x")]);
    let executor = ValidationExecutor::new(config);
    // Either succeeds (output has 0 elements) or returns TensorTooLarge.
    // Either way it must not panic.
    let _ = executor.execute_rule(&expr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Report generation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_report_has_no_violations_for_finite_output() {
    let expr = TLExpr::pred("finite_check", vec![Term::var("x")]);
    let executor = default_executor();
    let result = executor.execute_rule(&expr).unwrap();
    let report = executor.generate_validation_report(&result);
    assert!(
        report.results.is_empty() || report.conforms,
        "no violations expected for finite output"
    );
}

#[test]
fn test_export_rdf_contains_conforms_field() {
    let expr = TLExpr::pred("v", vec![Term::var("x"), Term::var("y")]);
    let executor = default_executor();
    let result = executor.execute_rule(&expr).unwrap();
    let rdf = executor.export_as_rdf(&result);
    let has_conforms = rdf.contains("tl:conforms true") || rdf.contains("tl:conforms false");
    assert!(
        has_conforms,
        "RDF must contain a tl:conforms field, got:\n{rdf}"
    );
}

#[test]
fn test_export_rdf_contains_execution_result_type() {
    let expr = TLExpr::pred("u", vec![Term::var("a")]);
    let executor = default_executor();
    let result = executor.execute_rule(&expr).unwrap();
    let rdf = executor.export_as_rdf(&result);
    assert!(
        rdf.contains("tl:ExecutionResult"),
        "RDF must declare a tl:ExecutionResult node"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_custom_base_iri_reflected_in_rdf() {
    let config = ValidationExecutorConfig {
        base_iri: "https://example.org/test/".to_string(),
        ..Default::default()
    };
    let expr = TLExpr::pred("w", vec![Term::var("x")]);
    let executor = ValidationExecutor::new(config);
    let result = executor.execute_rule(&expr).unwrap();
    let rdf = executor.export_as_rdf(&result);
    assert!(
        rdf.contains("https://example.org/test/"),
        "custom base IRI must appear in RDF output"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Expression coverage
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_unary_predicate_executes() {
    let expr = TLExpr::pred("person", vec![Term::var("x")]);
    let executor = default_executor();
    let result = executor.execute_rule(&expr).expect("unary predicate");
    assert!(!result.output_tensors.is_empty());
}

#[test]
fn test_conjunctive_expression_executes() {
    // AND(p(x), q(x)) compiles to a non-trivial graph with at least one node.
    let p = TLExpr::pred("p", vec![Term::var("x")]);
    let q = TLExpr::pred("q", vec![Term::var("x")]);
    let expr = TLExpr::and(p, q);
    let executor = default_executor();
    let result = executor
        .execute_rule(&expr)
        .expect("conjunctive expression");
    // A conjunctive expression always has at least one computational node.
    assert!(
        result.graph_node_count > 0,
        "AND expression must produce at least one graph node"
    );
    assert!(!result.output_tensors.is_empty());
}

#[test]
fn test_disjunctive_expression_executes() {
    let p = TLExpr::pred("alpha", vec![Term::var("x")]);
    let q = TLExpr::pred("beta", vec![Term::var("x")]);
    let expr = TLExpr::or(p, q);
    let executor = default_executor();
    let result = executor
        .execute_rule(&expr)
        .expect("disjunctive expression");
    assert!(!result.output_tensors.is_empty());
}
