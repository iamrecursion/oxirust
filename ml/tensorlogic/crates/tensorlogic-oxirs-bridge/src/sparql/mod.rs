//! Advanced SPARQL query compilation to TensorLogic operations
//!
//! This module provides comprehensive support for compiling SPARQL 1.1 queries
//! into TensorLogic expressions. Supports:
//! - SELECT queries (basic and complex)
//! - ASK queries (boolean existence checks)
//! - DESCRIBE queries (resource descriptions)
//! - CONSTRUCT queries (RDF graph construction)
//! - Triple patterns with variables and constants
//! - Filter constraints (comparison, regex)
//! - OPTIONAL patterns (left-outer join semantics)
//! - UNION patterns (disjunction)
//!
//! For production SPARQL federation and advanced features, consider using a dedicated SPARQL engine.

mod compiler;
mod tensor_eval;
mod types;

pub use compiler::SparqlCompiler;
pub use tensor_eval::TensorBgpEvaluator;
pub use types::{
    AggregateFunction, BindExpr, FilterCondition, GraphPattern, PatternElement, QueryType,
    SelectElement, SparqlQuery, TriplePattern,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ====== Basic SELECT Query Tests ======

    #[test]
    fn test_parse_simple_query() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x ?y WHERE {
              ?x <http://example.org/knows> ?y .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        // Check query type
        match &parsed.query_type {
            QueryType::Select {
                select_vars,
                distinct,
                ..
            } => {
                assert_eq!(select_vars, &vec!["x", "y"]);
                assert!(!distinct);
            }
            _ => panic!("Expected SELECT query"),
        }

        // Check WHERE pattern
        match &parsed.where_pattern {
            GraphPattern::Triple(pattern) => {
                assert_eq!(pattern.subject, PatternElement::Variable("x".to_string()));
                assert_eq!(
                    pattern.predicate,
                    PatternElement::Constant("http://example.org/knows".to_string())
                );
                assert_eq!(pattern.object, PatternElement::Variable("y".to_string()));
            }
            _ => panic!("Expected Triple pattern"),
        }
    }

    #[test]
    fn test_parse_select_distinct() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT DISTINCT ?x WHERE {
              ?x <http://example.org/type> ?t .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Select {
                select_vars,
                distinct,
                ..
            } => {
                assert_eq!(select_vars, &vec!["x"]);
                assert!(distinct);
            }
            _ => panic!("Expected SELECT DISTINCT query"),
        }
    }

    #[test]
    fn test_parse_query_with_filter() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x ?age WHERE {
              ?x <http://example.org/age> ?age .
              FILTER(?age > 18)
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Select { select_vars, .. } => {
                assert_eq!(select_vars, &vec!["x", "age"]);
            }
            _ => panic!("Expected SELECT query"),
        }

        // Check WHERE pattern contains filter
        match &parsed.where_pattern {
            GraphPattern::Group(patterns) => {
                assert_eq!(patterns.len(), 2);
                // One Triple, one Filter
                assert!(matches!(patterns[0], GraphPattern::Triple(_)));
                assert!(matches!(patterns[1], GraphPattern::Filter(_)));
            }
            _ => panic!("Expected Group pattern with filter"),
        }
    }

    #[test]
    fn test_parse_query_with_limit_offset() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x WHERE {
              ?x <http://example.org/type> ?t .
            } LIMIT 10 OFFSET 20
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        assert_eq!(parsed.limit, Some(10));
        assert_eq!(parsed.offset, Some(20));
    }

    #[test]
    fn test_parse_query_with_order_by() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x ?name WHERE {
              ?x <http://example.org/name> ?name .
            } ORDER BY ?name
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        assert_eq!(parsed.order_by, vec!["name"]);
    }

    // ====== ASK Query Tests ======

    #[test]
    fn test_parse_ask_query() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            ASK WHERE {
              ?x <http://example.org/knows> ?y .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Ask => {
                // Success
            }
            _ => panic!("Expected ASK query"),
        }
    }

    #[test]
    fn test_compile_ask_query() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());

        let query = r#"
            ASK WHERE {
              ?x <http://example.org/knows> ?y .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");

        // Should generate existence check
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("knows"));
    }

    // ====== DESCRIBE Query Tests ======

    #[test]
    fn test_parse_describe_query() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            DESCRIBE ?x WHERE {
              ?x <http://example.org/type> <http://example.org/Person> .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Describe { resources } => {
                assert_eq!(resources, &vec!["x"]);
            }
            _ => panic!("Expected DESCRIBE query"),
        }
    }

    #[test]
    fn test_compile_describe_query() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/type".to_string(), "type".to_string());

        let query = r#"
            DESCRIBE ?x WHERE {
              ?x <http://example.org/type> ?t .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");

        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("type"));
    }

    // ====== CONSTRUCT Query Tests ======

    #[test]
    fn test_parse_construct_query() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            CONSTRUCT { ?x <http://example.org/friend> ?y }
            WHERE {
              ?x <http://example.org/knows> ?y .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Construct { template } => {
                assert_eq!(template.len(), 1);
                let pattern = &template[0];
                assert_eq!(pattern.subject, PatternElement::Variable("x".to_string()));
                assert_eq!(
                    pattern.predicate,
                    PatternElement::Constant("http://example.org/friend".to_string())
                );
                assert_eq!(pattern.object, PatternElement::Variable("y".to_string()));
            }
            _ => panic!("Expected CONSTRUCT query"),
        }
    }

    #[test]
    fn test_compile_construct_query() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());

        let query = r#"
            CONSTRUCT { ?x <http://example.org/friend> ?y }
            WHERE {
              ?x <http://example.org/knows> ?y .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");

        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("knows"));
    }

    // ====== OPTIONAL Pattern Tests ======

    #[test]
    fn test_parse_optional_pattern() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x ?name ?age WHERE {
              ?x <http://example.org/name> ?name .
              OPTIONAL { ?x <http://example.org/age> ?age }
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.where_pattern {
            GraphPattern::Group(patterns) => {
                assert_eq!(patterns.len(), 2);
                assert!(matches!(patterns[0], GraphPattern::Triple(_)));
                assert!(matches!(patterns[1], GraphPattern::Optional(_)));
            }
            _ => panic!("Expected Group with OPTIONAL"),
        }
    }

    #[test]
    fn test_compile_optional_pattern() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/name".to_string(), "name".to_string());
        compiler.add_predicate_mapping("http://example.org/age".to_string(), "age".to_string());

        let query = r#"
            SELECT ?x ?name WHERE {
              ?x <http://example.org/name> ?name .
              OPTIONAL { ?x <http://example.org/age> ?age }
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");

        // Should have OR for optional semantics
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("name"));
        assert!(expr_str.contains("Or"));
    }

    // ====== UNION Pattern Tests ======

    #[test]
    fn test_parse_union_pattern() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x ?y WHERE {
              { ?x <http://example.org/knows> ?y }
              UNION
              { ?x <http://example.org/likes> ?y }
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.where_pattern {
            GraphPattern::Union(_, _) => {
                // Success - found UNION pattern
            }
            _ => panic!("Expected UNION pattern"),
        }
    }

    #[test]
    fn test_compile_union_pattern() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());
        compiler.add_predicate_mapping("http://example.org/likes".to_string(), "likes".to_string());

        let query = r#"
            SELECT ?x ?y WHERE {
              { ?x <http://example.org/knows> ?y }
              UNION
              { ?x <http://example.org/likes> ?y }
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");

        // Should have OR for union
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("knows") || expr_str.contains("likes"));
        assert!(expr_str.contains("Or"));
    }

    // ====== Filter Conditions Tests ======

    #[test]
    fn test_filter_greater_or_equal() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?x WHERE {
              ?x <http://example.org/age> ?age .
              FILTER(?age >= 18)
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.where_pattern {
            GraphPattern::Group(patterns) => {
                if let Some(GraphPattern::Filter(FilterCondition::GreaterOrEqual(var, val))) =
                    patterns.get(1)
                {
                    assert_eq!(var, "age");
                    assert_eq!(val, "18");
                } else {
                    panic!("Expected GreaterOrEqual filter");
                }
            }
            _ => panic!("Expected Group pattern"),
        }
    }

    // ====== Compilation Tests ======

    #[test]
    fn test_compile_simple_query() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());

        let query = r#"
            SELECT ?x ?y WHERE {
              ?x <http://example.org/knows> ?y .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");

        // Should generate a predicate expression
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("knows"));
    }

    #[test]
    fn test_compile_query_with_multiple_patterns() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/knows".to_string(), "knows".to_string());

        let query = r#"
            SELECT ?x ?y ?z WHERE {
              ?x <http://example.org/knows> ?y .
              ?y <http://example.org/knows> ?z .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");

        // Should generate AND of predicates
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("knows"));
        assert!(expr_str.contains("And"));
    }

    #[test]
    fn test_compile_query_with_filter() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/age".to_string(), "age".to_string());

        let query = r#"
            SELECT ?x ?a WHERE {
              ?x <http://example.org/age> ?a .
              FILTER(?a > 18)
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");

        // Should include both predicate and filter
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("age"));
        assert!(expr_str.contains("greaterThan"));
    }

    // ====== Complex Integration Tests ======

    #[test]
    fn test_complex_query_with_optional_and_filter() {
        let mut compiler = SparqlCompiler::new();
        compiler.add_predicate_mapping("http://example.org/name".to_string(), "name".to_string());
        compiler.add_predicate_mapping("http://example.org/age".to_string(), "age".to_string());

        let query = r#"
            SELECT DISTINCT ?x ?name WHERE {
              ?x <http://example.org/name> ?name .
              OPTIONAL {
                ?x <http://example.org/age> ?age .
                FILTER(?age >= 21)
              }
            } LIMIT 100 ORDER BY ?name
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        // Check all components
        match &parsed.query_type {
            QueryType::Select {
                select_vars,
                distinct,
                ..
            } => {
                assert_eq!(select_vars, &vec!["x", "name"]);
                assert!(distinct);
            }
            _ => panic!("Expected SELECT DISTINCT"),
        }

        assert_eq!(parsed.limit, Some(100));
        assert_eq!(parsed.order_by, vec!["name"]);

        // Check WHERE pattern structure - should be a Group with at least 2 patterns
        match &parsed.where_pattern {
            GraphPattern::Group(patterns) => {
                assert!(patterns.len() >= 2, "Expected at least 2 patterns in group");
                // First should be a Triple (name predicate)
                assert!(matches!(patterns[0], GraphPattern::Triple(_)));
            }
            _ => panic!("Expected Group pattern"),
        }

        // Compile and check basic predicates are present
        let tl_expr = compiler.compile_to_tensorlogic(&parsed).expect("unwrap");
        let expr_str = format!("{:?}", tl_expr);
        assert!(expr_str.contains("name"));
        // Should have logical operators combining the patterns
        assert!(expr_str.contains("And") || expr_str.contains("Or"));
    }

    // ====== Aggregate Function Tests ======

    #[test]
    fn test_parse_count_aggregate() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT (COUNT(?x) AS ?count) WHERE {
              ?x <http://example.org/type> <http://example.org/Person> .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Select { projections, .. } => {
                assert_eq!(projections.len(), 1);
                match &projections[0] {
                    SelectElement::Aggregate { function, alias } => {
                        assert!(matches!(function, AggregateFunction::Count { .. }));
                        assert_eq!(alias, &Some("count".to_string()));
                    }
                    _ => panic!("Expected Aggregate element"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_sum_aggregate() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT (SUM(?amount) AS ?total) WHERE {
              ?x <http://example.org/amount> ?amount .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Select { projections, .. } => {
                assert_eq!(projections.len(), 1);
                match &projections[0] {
                    SelectElement::Aggregate { function, .. } => {
                        if let AggregateFunction::Sum { variable, .. } = function {
                            assert_eq!(variable, "amount");
                        } else {
                            panic!("Expected SUM aggregate");
                        }
                    }
                    _ => panic!("Expected Aggregate element"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_avg_min_max() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT (AVG(?age) AS ?avg_age) (MIN(?age) AS ?min_age) (MAX(?age) AS ?max_age) WHERE {
              ?x <http://example.org/age> ?age .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Select { projections, .. } => {
                assert_eq!(projections.len(), 3);
                // Check AVG
                match &projections[0] {
                    SelectElement::Aggregate { function, .. } => {
                        assert!(matches!(function, AggregateFunction::Avg { .. }));
                    }
                    _ => panic!("Expected Aggregate element"),
                }
                // Check MIN
                match &projections[1] {
                    SelectElement::Aggregate { function, .. } => {
                        assert!(matches!(function, AggregateFunction::Min { .. }));
                    }
                    _ => panic!("Expected Aggregate element"),
                }
                // Check MAX
                match &projections[2] {
                    SelectElement::Aggregate { function, .. } => {
                        assert!(matches!(function, AggregateFunction::Max { .. }));
                    }
                    _ => panic!("Expected Aggregate element"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_group_by() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?dept (COUNT(?person) AS ?count) WHERE {
              ?person <http://example.org/department> ?dept .
            } GROUP BY ?dept
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        assert_eq!(parsed.group_by, vec!["dept"]);

        match &parsed.query_type {
            QueryType::Select { projections, .. } => {
                assert_eq!(projections.len(), 2);
                // First should be variable
                match &projections[0] {
                    SelectElement::Variable(name) => assert_eq!(name, "dept"),
                    _ => panic!("Expected Variable element"),
                }
                // Second should be aggregate
                match &projections[1] {
                    SelectElement::Aggregate { function, .. } => {
                        assert!(matches!(function, AggregateFunction::Count { .. }));
                    }
                    _ => panic!("Expected Aggregate element"),
                }
            }
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_having() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?dept (COUNT(?person) AS ?count) WHERE {
              ?person <http://example.org/department> ?dept .
            } GROUP BY ?dept HAVING(?count > 10)
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        assert_eq!(parsed.group_by, vec!["dept"]);
        assert_eq!(parsed.having.len(), 1);

        match &parsed.having[0] {
            FilterCondition::GreaterThan(var, val) => {
                assert_eq!(var, "count");
                assert_eq!(val, "10");
            }
            _ => panic!("Expected GreaterThan condition"),
        }
    }

    #[test]
    fn test_parse_count_distinct() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT (COUNT(DISTINCT ?person) AS ?unique) WHERE {
              ?person <http://example.org/type> <http://example.org/Person> .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Select { projections, .. } => match &projections[0] {
                SelectElement::Aggregate { function, .. } => {
                    if let AggregateFunction::Count { distinct, .. } = function {
                        assert!(distinct);
                    } else {
                        panic!("Expected COUNT aggregate");
                    }
                }
                _ => panic!("Expected Aggregate element"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_parse_count_star() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT (COUNT(*) AS ?total) WHERE {
              ?x <http://example.org/type> ?type .
            }
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        match &parsed.query_type {
            QueryType::Select { projections, .. } => match &projections[0] {
                SelectElement::Aggregate { function, .. } => {
                    if let AggregateFunction::Count { variable, .. } = function {
                        assert!(variable.is_none());
                    } else {
                        panic!("Expected COUNT aggregate");
                    }
                }
                _ => panic!("Expected Aggregate element"),
            },
            _ => panic!("Expected SELECT"),
        }
    }

    #[test]
    fn test_compile_bind_constant() {
        let compiler = SparqlCompiler::new();
        let query = r#"SELECT ?greeting WHERE { BIND ( "hello" AS ?greeting ) }"#;
        let parsed = compiler.parse_query(query).expect("parse_query failed");
        let tl_expr = compiler
            .compile_to_tensorlogic(&parsed)
            .expect("compile failed");
        let expr_str = format!("{tl_expr:?}");
        assert!(
            expr_str.contains("equals"),
            "compiled Bind should contain 'equals': {expr_str}"
        );
        assert!(
            expr_str.contains("greeting"),
            "compiled Bind should contain 'greeting': {expr_str}"
        );
    }

    #[test]
    fn test_compile_values_single_var() {
        let compiler = SparqlCompiler::new();
        let query = r#"SELECT ?x WHERE { VALUES ?x { 1 2 } }"#;
        let parsed = compiler.parse_query(query).expect("parse_query failed");
        let tl_expr = compiler
            .compile_to_tensorlogic(&parsed)
            .expect("compile failed");
        let expr_str = format!("{tl_expr:?}");
        assert!(
            expr_str.contains("Or"),
            "compiled Values should contain 'Or' for row disjunction: {expr_str}"
        );
        assert!(
            expr_str.contains("equals"),
            "compiled Values should contain 'equals': {expr_str}"
        );
    }

    #[test]
    fn test_combined_variables_and_aggregates() {
        let compiler = SparqlCompiler::new();
        let query = r#"
            SELECT ?category (SUM(?price) AS ?total) (AVG(?price) AS ?average) WHERE {
              ?item <http://example.org/category> ?category .
              ?item <http://example.org/price> ?price .
            } GROUP BY ?category ORDER BY ?total LIMIT 10
        "#;

        let parsed = compiler.parse_query(query).expect("unwrap");

        // Check projections
        match &parsed.query_type {
            QueryType::Select {
                projections,
                select_vars,
                ..
            } => {
                assert_eq!(projections.len(), 3);
                assert_eq!(select_vars, &vec!["category", "total", "average"]);
            }
            _ => panic!("Expected SELECT"),
        }

        // Check modifiers
        assert_eq!(parsed.group_by, vec!["category"]);
        assert_eq!(parsed.order_by, vec!["total"]);
        assert_eq!(parsed.limit, Some(10));
    }
}
