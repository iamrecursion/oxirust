//! Integration tests for the tensor-based SPARQL evaluator and InternedGraph.

use tensorlogic_oxirs_bridge::{
    rdf_bulk_importer_into_interned, BridgeError, FilterCondition, GraphPattern, InternedGraph,
    PatternElement, QueryType, RdfBulkImporter, RdfTriple, SelectElement, SparqlQuery,
    TensorBgpEvaluator, TriplePattern,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn var(name: &str) -> PatternElement {
    PatternElement::Variable(name.to_string())
}

fn cst(s: &str) -> PatternElement {
    PatternElement::Constant(s.to_string())
}

fn triple_pat(s: PatternElement, p: PatternElement, o: PatternElement) -> GraphPattern {
    GraphPattern::Triple(TriplePattern {
        subject: s,
        predicate: p,
        object: o,
    })
}

fn make_select(vars: Vec<&str>, pattern: GraphPattern) -> SparqlQuery {
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

// ── Test: single pattern ───────────────────────────────────────────────────────

#[test]
fn test_interned_graph_single_pattern() {
    let mut g = InternedGraph::new();
    g.add_triple("Alice", "knows", "Bob");
    g.add_triple("Bob", "knows", "Carol");

    let evaluator = TensorBgpEvaluator::new(&g);
    let query = make_select(vec!["x", "y"], triple_pat(var("x"), cst("knows"), var("y")));

    let results = evaluator.evaluate(&query).expect("evaluate must succeed");
    assert_eq!(results.len(), 2, "expected 2 result rows");

    let pairs: Vec<(&str, &str)> = results
        .iter()
        .map(|r| (r["x"].as_str(), r["y"].as_str()))
        .collect();
    assert!(
        pairs.contains(&("Alice", "Bob")),
        "Alice knows Bob expected"
    );
    assert!(
        pairs.contains(&("Bob", "Carol")),
        "Bob knows Carol expected"
    );
}

// ── Test: join (path composition) ─────────────────────────────────────────────

#[test]
fn test_interned_graph_join() {
    let mut g = InternedGraph::new();
    g.add_triple("Alice", "knows", "Bob");
    g.add_triple("Bob", "knows", "Carol");

    let evaluator = TensorBgpEvaluator::new(&g);
    let pattern = GraphPattern::Group(vec![
        triple_pat(var("x"), cst("knows"), var("y")),
        triple_pat(var("y"), cst("knows"), var("z")),
    ]);
    let query = make_select(vec!["x", "z"], pattern);

    let results = evaluator.evaluate(&query).expect("evaluate must succeed");
    // Only Alice → Bob → Carol exists
    assert_eq!(results.len(), 1, "expected one transitive path");
    assert_eq!(results[0]["x"], "Alice");
    assert_eq!(results[0]["z"], "Carol");
}

// ── Test: constant subject ─────────────────────────────────────────────────────

#[test]
fn test_constant_subject() {
    let mut g = InternedGraph::new();
    g.add_triple("Alice", "knows", "Bob");
    g.add_triple("Bob", "knows", "Carol");

    let evaluator = TensorBgpEvaluator::new(&g);
    let query = make_select(vec!["y"], triple_pat(cst("Alice"), cst("knows"), var("y")));

    let results = evaluator.evaluate(&query).expect("evaluate must succeed");
    assert_eq!(results.len(), 1, "Alice knows only Bob");
    assert_eq!(results[0]["y"], "Bob");
}

// ── Test: empty graph → empty results ─────────────────────────────────────────

#[test]
fn test_empty_graph() {
    let g = InternedGraph::new();
    let evaluator = TensorBgpEvaluator::new(&g);
    let query = make_select(vec!["x"], triple_pat(var("x"), cst("knows"), var("y")));
    let results = evaluator
        .evaluate(&query)
        .expect("evaluate should not error on empty graph");
    assert!(results.is_empty(), "empty graph must produce empty results");
}

// ── Test: unknown predicate → empty results ────────────────────────────────────

#[test]
fn test_unknown_predicate() {
    let mut g = InternedGraph::new();
    g.add_triple("Alice", "knows", "Bob");

    let evaluator = TensorBgpEvaluator::new(&g);
    let query = make_select(vec!["x", "y"], triple_pat(var("x"), cst("likes"), var("y")));
    let results = evaluator
        .evaluate(&query)
        .expect("evaluate should not error");
    assert!(results.is_empty(), "unknown predicate → no results");
}

// ── Test: FILTER → ValidationError ─────────────────────────────────────────────

#[test]
fn test_unsupported_filter_error() {
    let g = InternedGraph::new();
    let evaluator = TensorBgpEvaluator::new(&g);
    let query = make_select(
        vec!["x"],
        GraphPattern::Filter(FilterCondition::Bound("x".to_string())),
    );
    match evaluator.evaluate(&query) {
        Err(BridgeError::ValidationError(msg)) => {
            assert!(
                msg.contains("FILTER") || msg.contains("tensor path"),
                "error should mention FILTER: {msg}"
            );
        }
        other => panic!("expected ValidationError for FILTER, got: {:?}", other),
    }
}

// ── Test: variable predicate → ValidationError ────────────────────────────────

#[test]
fn test_variable_predicate_error() {
    let g = InternedGraph::new();
    let evaluator = TensorBgpEvaluator::new(&g);
    let query = make_select(vec!["x"], triple_pat(var("x"), var("p"), var("y")));
    match evaluator.evaluate(&query) {
        Err(BridgeError::ValidationError(msg)) => {
            assert!(
                msg.contains("Variable predicates") || msg.contains("predicate"),
                "error should mention variable predicate: {msg}"
            );
        }
        other => panic!(
            "expected ValidationError for variable predicate, got: {:?}",
            other
        ),
    }
}

// ── Test: parallel vs serial bulk load ────────────────────────────────────────

#[test]
fn test_parallel_vs_serial_bulk_load() {
    let nt = "<Alice> <knows> <Bob> .\n\
              <Bob> <knows> <Carol> .\n\
              <Carol> <knows> <Dave> .\n";
    let importer = RdfBulkImporter::new();
    let g = rdf_bulk_importer_into_interned(&importer, nt).expect("parallel import must succeed");
    assert_eq!(g.num_triples(), 3, "parallel load should find 3 triples");

    // Compare with direct sequential build
    let triples = vec![
        RdfTriple::new("<Alice>", "<knows>", "<Bob>"),
        RdfTriple::new("<Bob>", "<knows>", "<Carol>"),
        RdfTriple::new("<Carol>", "<knows>", "<Dave>"),
    ];
    // force sequential path by calling from_rdf_triples on larger slice
    // (in tests PARALLEL_THRESHOLD=4, so len=3 uses parallel path, but sequential is always available)
    // We test the counts match
    let g2 = InternedGraph::from_rdf_triples(triples);
    assert_eq!(
        g2.num_triples(),
        g.num_triples(),
        "triple counts must match"
    );
    assert_eq!(
        g2.num_entities(),
        g.num_entities(),
        "entity counts must match"
    );
}

// ── Test: bulk load into QuadStore ────────────────────────────────────────────

#[test]
fn test_bulk_load_into_quad_store() {
    let nt = "<Alice> <knows> <Bob> .\n\
              <Bob> <likes> <Carol> .\n";
    let importer = RdfBulkImporter::new();
    let g = rdf_bulk_importer_into_interned(&importer, nt).expect("bulk import must succeed");

    let qs = g.into_quad_store();
    assert_eq!(qs.total_quads(), 2, "QuadStore should contain 2 triples");

    let knows_pairs = qs.query_predicate(None, "<knows>");
    assert_eq!(knows_pairs.len(), 1, "one 'knows' triple");

    let likes_pairs = qs.query_predicate(None, "<likes>");
    assert_eq!(likes_pairs.len(), 1, "one 'likes' triple");
}
