//! Integration tests: SPARQL BIND/VALUES parse → AST assertions and builder roundtrip.
//!
//! Note on "roundtrip": `sparql::types::GraphPattern` and `sparql_gen::types::GraphPattern`
//! are distinct types with no serializer from the former, so the roundtrip is tested as:
//!   parse query string → assert correct `sparql::GraphPattern` AST shape
//! plus builder tests that produce the equivalent SPARQL via the fluent API.

use tensorlogic_oxirs_bridge::{
    BindExpr, GraphPattern, PatternElement, SparqlCompiler, SparqlTerm, WhereClause,
};

// ---------------------------------------------------------------------------
// Parse → AST tests
// ---------------------------------------------------------------------------

#[test]
fn roundtrip_bind_constant() {
    let compiler = SparqlCompiler::new();
    let query = r#"SELECT ?greeting WHERE { BIND ( "hello" AS ?greeting ) }"#;
    let parsed = compiler.parse_query(query).expect("parse_query failed");

    match &parsed.where_pattern {
        GraphPattern::Bind(BindExpr::Term(PatternElement::Constant(c)), var) => {
            assert_eq!(c, "hello", "constant value mismatch");
            assert_eq!(var, "greeting", "variable name mismatch");
        }
        other => panic!("Expected Bind(Term(Constant)), got: {other:?}"),
    }
}

#[test]
fn roundtrip_values_single_var() {
    let compiler = SparqlCompiler::new();
    let query = r#"SELECT ?x WHERE { VALUES ?x { 1 2 3 } }"#;
    let parsed = compiler.parse_query(query).expect("parse_query failed");

    match &parsed.where_pattern {
        GraphPattern::Values(vars, rows) => {
            assert_eq!(vars, &["x"], "variable list mismatch");
            assert_eq!(rows.len(), 3, "expected 3 rows");
            for (i, expected) in ["1", "2", "3"].iter().enumerate() {
                match &rows[i][0] {
                    PatternElement::Constant(c) => assert_eq!(c, expected),
                    other => panic!("Expected Constant, got: {other:?}"),
                }
            }
        }
        other => panic!("Expected Values, got: {other:?}"),
    }
}

#[test]
fn roundtrip_values_multi_var() {
    let compiler = SparqlCompiler::new();
    let query = r#"SELECT ?x ?y WHERE { VALUES (?x ?y) { (1 "a") (2 "b") } }"#;
    let parsed = compiler.parse_query(query).expect("parse_query failed");

    match &parsed.where_pattern {
        GraphPattern::Values(vars, rows) => {
            assert_eq!(vars, &["x", "y"], "variable list mismatch");
            assert_eq!(rows.len(), 2, "expected 2 rows");

            // Row 0: (Constant("1"), Constant("a"))
            assert_eq!(rows[0].len(), 2);
            match &rows[0][0] {
                PatternElement::Constant(c) => assert_eq!(c, "1"),
                other => panic!("row 0 elem 0: expected Constant, got {other:?}"),
            }
            match &rows[0][1] {
                PatternElement::Constant(c) => assert_eq!(c, "a"),
                other => panic!("row 0 elem 1: expected Constant, got {other:?}"),
            }

            // Row 1: (Constant("2"), Constant("b"))
            assert_eq!(rows[1].len(), 2);
            match &rows[1][0] {
                PatternElement::Constant(c) => assert_eq!(c, "2"),
                other => panic!("row 1 elem 0: expected Constant, got {other:?}"),
            }
            match &rows[1][1] {
                PatternElement::Constant(c) => assert_eq!(c, "b"),
                other => panic!("row 1 elem 1: expected Constant, got {other:?}"),
            }
        }
        other => panic!("Expected Values, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Builder roundtrip — produce SPARQL via fluent API and check it contains
// the expected VALUES clause syntax.
// ---------------------------------------------------------------------------

#[test]
fn builder_values_single_var_renders_correctly() {
    use tensorlogic_oxirs_bridge::SelectQuery;
    let query = SelectQuery::new()
        .select("x")
        .where_clause(WhereClause::new().values(
            "x",
            vec![SparqlTerm::literal("1"), SparqlTerm::literal("2")],
        ))
        .build()
        .expect("build should succeed");

    assert!(
        query.contains("VALUES ?x"),
        "rendered query should contain VALUES ?x — got:\n{query}"
    );
    assert!(
        query.contains("\"1\""),
        "rendered query should contain \"1\" — got:\n{query}"
    );
}

#[test]
fn builder_values_multi_var_renders_correctly() {
    use tensorlogic_oxirs_bridge::SelectQuery;
    let rows = vec![
        vec![SparqlTerm::literal("1"), SparqlTerm::literal("a")],
        vec![SparqlTerm::literal("2"), SparqlTerm::literal("b")],
    ];
    let query = SelectQuery::new()
        .select("x")
        .select("y")
        .where_clause(WhereClause::new().values_multi(vec!["x", "y"], rows))
        .build()
        .expect("build should succeed");

    assert!(
        query.contains("VALUES (?x ?y)"),
        "rendered query should contain VALUES (?x ?y) — got:\n{query}"
    );
    assert!(
        query.contains("(\"1\" \"a\")"),
        "rendered query should contain (\"1\" \"a\") — got:\n{query}"
    );
}
