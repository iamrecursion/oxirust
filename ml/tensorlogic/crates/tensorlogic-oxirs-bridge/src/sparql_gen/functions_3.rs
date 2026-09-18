//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::functions::{render_filter, render_pattern};
    use super::super::functions_2::expr_to_sparql;
    use super::super::types::{
        GraphPattern, SparqlExpr, SparqlFilter, SparqlGenConfig, SparqlGenError, SparqlQuery,
        SparqlTerm, TriplePattern,
    };
    use tensorlogic_ir::{TLExpr, Term};
    fn cfg() -> SparqlGenConfig {
        SparqlGenConfig::default()
    }
    #[test]
    fn test_term_var_rendering() {
        assert_eq!(SparqlTerm::var("x").as_sparql_string(), "?x");
    }
    #[test]
    fn test_term_iri_rendering() {
        assert_eq!(
            SparqlTerm::iri("http://ex.org/foo").as_sparql_string(),
            "<http://ex.org/foo>"
        );
    }
    #[test]
    fn test_term_literal_rendering() {
        assert_eq!(SparqlTerm::literal("hello").as_sparql_string(), "\"hello\"");
    }
    #[test]
    fn test_term_numeric_literal() {
        let s = SparqlTerm::NumericLiteral(std::f64::consts::PI).as_sparql_string();
        assert!(s.contains('3'));
    }
    #[test]
    fn test_empty_query_contains_select() {
        let q = SparqlQuery::new();
        let s = q.to_sparql();
        assert!(s.contains("SELECT"));
    }
    #[test]
    fn test_query_prefix_rendering() {
        let q = SparqlQuery::new().with_prefix("ex", "http://example.org/");
        let s = q.to_sparql();
        assert!(s.contains("PREFIX ex:"));
    }
    #[test]
    fn test_query_distinct_rendering() {
        let q = SparqlQuery::new().distinct();
        let s = q.to_sparql();
        assert!(s.contains("DISTINCT"));
    }
    #[test]
    fn test_query_limit_rendering() {
        let q = SparqlQuery::new().limit(42);
        let s = q.to_sparql();
        assert!(s.contains("LIMIT 42"));
    }
    #[test]
    fn test_triple_pattern_rendering() {
        let tp = GraphPattern::Triple(TriplePattern::new(
            SparqlTerm::var("s"),
            SparqlTerm::iri("http://ex.org/p"),
            SparqlTerm::var("o"),
        ));
        let s = render_pattern(&tp, "  ", 0);
        assert!(s.contains("?s"));
        assert!(s.contains("?o"));
    }
    #[test]
    fn test_optional_pattern_rendering() {
        let tp = GraphPattern::Triple(TriplePattern::new(
            SparqlTerm::var("s"),
            SparqlTerm::iri("http://ex.org/p"),
            SparqlTerm::var("o"),
        ));
        let opt = GraphPattern::Optional(vec![tp]);
        let s = render_pattern(&opt, "  ", 0);
        assert!(s.contains("OPTIONAL {"));
    }
    #[test]
    fn test_union_pattern_rendering() {
        let make_tp = || {
            GraphPattern::Triple(TriplePattern::new(
                SparqlTerm::var("s"),
                SparqlTerm::iri("http://ex.org/p"),
                SparqlTerm::var("o"),
            ))
        };
        let union = GraphPattern::Union(vec![make_tp()], vec![make_tp()]);
        let s = render_pattern(&union, "  ", 0);
        assert!(s.contains("UNION"));
    }
    #[test]
    fn test_filter_pattern_rendering() {
        let f = GraphPattern::Filter(SparqlFilter::BoolValue(true));
        let s = render_pattern(&f, "  ", 0);
        assert!(s.contains("FILTER ("));
    }
    #[test]
    fn test_filter_equals_rendering() {
        let f = SparqlFilter::Equals(SparqlTerm::var("x"), SparqlTerm::var("y"));
        assert!(render_filter(&f).contains('='));
    }
    #[test]
    fn test_filter_not_equals_rendering() {
        let inner = SparqlFilter::Equals(SparqlTerm::var("x"), SparqlTerm::var("y"));
        let f = SparqlFilter::Not(Box::new(inner));
        let s = render_filter(&f);
        assert!(s.contains("!("));
    }
    #[test]
    fn test_filter_and_rendering() {
        let f = SparqlFilter::And(
            Box::new(SparqlFilter::BoolValue(true)),
            Box::new(SparqlFilter::BoolValue(false)),
        );
        assert!(render_filter(&f).contains("&&"));
    }
    #[test]
    fn test_filter_or_rendering() {
        let f = SparqlFilter::Or(
            Box::new(SparqlFilter::BoolValue(true)),
            Box::new(SparqlFilter::BoolValue(false)),
        );
        assert!(render_filter(&f).contains("||"));
    }
    #[test]
    fn test_pred_generates_triple_or_type_pattern() {
        let pred = TLExpr::pred("Person", vec![Term::var("x")]);
        let q = expr_to_sparql(&pred, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("WHERE {"));
        assert!(!s.trim().is_empty());
    }
    #[test]
    fn test_and_generates_both_patterns() {
        let p = TLExpr::pred("Person", vec![Term::var("x")]);
        let q_pred = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let expr = TLExpr::and(p, q_pred);
        let q = expr_to_sparql(&expr, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("v_x") || s.contains("x"));
    }
    #[test]
    fn test_or_generates_union() {
        let p = TLExpr::pred("Person", vec![Term::var("x")]);
        let q_pred = TLExpr::pred("Agent", vec![Term::var("x")]);
        let expr = TLExpr::or(p, q_pred);
        let q = expr_to_sparql(&expr, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("UNION"));
    }
    #[test]
    fn test_constant_zero_generates_filter_false() {
        let expr = TLExpr::Constant(0.0);
        let q = expr_to_sparql(&expr, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("false"));
    }
    #[test]
    fn test_not_generates_filter_not_exists() {
        let p = TLExpr::pred("Person", vec![Term::var("x")]);
        let expr = TLExpr::negate(p);
        let q = expr_to_sparql(&expr, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("NOT EXISTS"));
    }
    #[test]
    fn test_error_display() {
        let e = SparqlGenError::UnsupportedExpr("Lambda".to_owned());
        let s = format!("{e}");
        assert!(s.contains("Lambda"));
        let e2 = SparqlGenError::AmbiguousVariable("x".to_owned());
        let s2 = format!("{e2}");
        assert!(s2.contains('x'));
        let e3 = SparqlGenError::EmptyQuery;
        let s3 = format!("{e3}");
        assert!(!s3.is_empty());
    }
    #[test]
    fn test_select_projects_vars() {
        let q = SparqlQuery::new().select(&["foo", "bar"]);
        let s = q.to_sparql();
        assert!(s.contains("?foo"));
        assert!(s.contains("?bar"));
    }
    #[test]
    fn test_rendered_sparql_contains_where() {
        let pred = TLExpr::pred("knows", vec![Term::var("a"), Term::var("b")]);
        let q = expr_to_sparql(&pred, &cfg()).expect("query");
        let s = q.to_sparql();
        assert!(s.contains("WHERE {"));
    }
    #[test]
    fn test_zero_arity_pred_generates_type_pattern() {
        let pred = TLExpr::pred("Person", vec![]);
        let q = expr_to_sparql(&pred, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("rdf:type") || s.contains("type"));
    }
    #[test]
    fn test_binary_pred_generates_triple() {
        let pred = TLExpr::pred("knows", vec![Term::var("a"), Term::var("b")]);
        let q = expr_to_sparql(&pred, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("knows") || s.contains("v_a"));
    }
    #[test]
    fn test_offset_rendering() {
        let q = SparqlQuery::new().offset(10);
        let s = q.to_sparql();
        assert!(s.contains("OFFSET 10"));
    }
    #[test]
    fn test_order_by_rendering() {
        let q = SparqlQuery::new().order_by("name");
        let s = q.to_sparql();
        assert!(s.contains("ORDER BY ?name"));
    }
    #[test]
    fn test_bound_filter_rendering() {
        let f = SparqlFilter::Bound("x".to_owned());
        assert!(render_filter(&f).contains("BOUND(?x)"));
    }
    #[test]
    fn test_isiri_filter_rendering() {
        let f = SparqlFilter::IsIri(SparqlTerm::var("x"));
        assert!(render_filter(&f).contains("isIRI(?x)"));
    }
    #[test]
    fn test_regex_filter_rendering() {
        let f = SparqlFilter::Regex(SparqlTerm::var("name"), "^Alice".to_owned());
        let s = render_filter(&f);
        assert!(s.contains("REGEX("));
        assert!(s.contains("^Alice"));
    }
    #[test]
    fn test_let_generates_bind() {
        let expr = TLExpr::Let {
            var: "result".to_owned(),
            value: Box::new(TLExpr::Constant(42.0)),
            body: Box::new(TLExpr::pred("Check", vec![Term::var("result")])),
        };
        let q = expr_to_sparql(&expr, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("BIND"));
    }
    #[test]
    fn test_select_star_when_no_vars() {
        let q = SparqlQuery::new();
        let s = q.to_sparql();
        assert!(s.contains("SELECT *"));
    }
    #[test]
    fn test_bool_literal_rendering() {
        assert_eq!(SparqlTerm::BoolLiteral(true).as_sparql_string(), "true");
        assert_eq!(SparqlTerm::BoolLiteral(false).as_sparql_string(), "false");
    }
    #[test]
    fn test_imply_generates_union() {
        let p = TLExpr::pred("Person", vec![Term::var("x")]);
        let q_pred = TLExpr::pred("Mortal", vec![Term::var("x")]);
        let expr = TLExpr::imply(p, q_pred);
        let q = expr_to_sparql(&expr, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("UNION") || s.contains("NOT EXISTS"));
    }
    #[test]
    fn test_exists_introduces_variable() {
        let body = TLExpr::pred("knows", vec![Term::var("x"), Term::var("y")]);
        let expr = TLExpr::exists("y", "Person", body);
        let q = expr_to_sparql(&expr, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("WHERE {"));
    }
    #[test]
    fn test_config_variable_prefix() {
        let config = SparqlGenConfig {
            variable_prefix: "my_".to_owned(),
            ..SparqlGenConfig::default()
        };
        let pred = TLExpr::pred("P", vec![Term::var("x")]);
        let q = expr_to_sparql(&pred, &config).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("my_x"));
    }
    #[test]
    fn test_add_pattern_builder() {
        let tp = TriplePattern::new(
            SparqlTerm::var("s"),
            SparqlTerm::iri("http://ex.org/p"),
            SparqlTerm::var("o"),
        );
        let q = SparqlQuery::new().add_pattern(GraphPattern::Triple(tp));
        assert_eq!(q.patterns.len(), 1);
    }
    #[test]
    fn test_group_pattern_rendering() {
        let inner = GraphPattern::Triple(TriplePattern::new(
            SparqlTerm::var("s"),
            SparqlTerm::iri("http://ex.org/p"),
            SparqlTerm::var("o"),
        ));
        let group = GraphPattern::Group(vec![inner]);
        let s = render_pattern(&group, "  ", 0);
        assert!(s.contains('{'));
    }
    #[test]
    fn test_bind_pattern_rendering() {
        let bind = GraphPattern::Bind(
            SparqlExpr::Term(SparqlTerm::NumericLiteral(1.0)),
            "counter".to_owned(),
        );
        let s = render_pattern(&bind, "  ", 0);
        assert!(s.contains("BIND"));
        assert!(s.contains("?counter"));
    }
    #[test]
    fn test_nested_and_or() {
        let p = TLExpr::pred("A", vec![Term::var("x")]);
        let q_pred = TLExpr::pred("B", vec![Term::var("x")]);
        let r = TLExpr::pred("C", vec![Term::var("y")]);
        let expr = TLExpr::and(TLExpr::or(p, q_pred), r);
        let q = expr_to_sparql(&expr, &cfg()).expect("generates query");
        let s = q.to_sparql();
        assert!(s.contains("UNION"));
        assert!(s.contains("WHERE {"));
    }
}
