//! Representative parsing / round-trip tests, ported from `spargebra`.

use oxixml_sparql_syntax::{Query, SparqlParser, Update};

fn q(s: &str) -> Query {
    SparqlParser::new()
        .parse_query(s)
        .unwrap_or_else(|e| panic!("failed to parse {s:?}: {e}"))
}

fn u(s: &str) -> Update {
    SparqlParser::new()
        .parse_update(s)
        .unwrap_or_else(|e| panic!("failed to parse update {s:?}: {e}"))
}

#[test]
fn select_sse_and_display() {
    let query = q("SELECT ?s ?p ?o WHERE { ?s ?p ?o . }");
    assert_eq!(query.to_string(), "SELECT ?s ?p ?o WHERE { ?s ?p ?o . }");
    assert_eq!(
        query.to_sse(),
        "(project (?s ?p ?o) (bgp (triple ?s ?p ?o)))"
    );
}

#[test]
fn select_star() {
    let query = q("SELECT * WHERE { ?s ?p ?o }");
    assert_eq!(
        query.to_sse(),
        "(project (?o ?p ?s) (bgp (triple ?s ?p ?o)))"
    );
}

#[test]
fn ask() {
    let query = q("ASK { ?s ?p ?o }");
    assert_eq!(query.to_sse(), "(ask (bgp (triple ?s ?p ?o)))");
}

#[test]
fn filter() {
    let query = q("SELECT ?s WHERE { ?s ?p ?o FILTER(?o > 3) }");
    assert_eq!(
        query.to_sse(),
        "(project (?s) (filter (<http://www.w3.org/ns/sparql#greater-than> ?o \"3\"^^<http://www.w3.org/2001/XMLSchema#integer>) (bgp (triple ?s ?p ?o))))"
    );
}

#[test]
fn optional() {
    let query = q("SELECT * WHERE { ?s ?p ?o OPTIONAL { ?s ?p2 ?o2 } }");
    assert_eq!(
        query.to_sse(),
        "(project (?o ?o2 ?p ?p2 ?s) (leftjoin (bgp (triple ?s ?p ?o)) (bgp (triple ?s ?p2 ?o2))))"
    );
}

#[test]
fn union() {
    let query = q("SELECT * WHERE { { ?s ?p ?o } UNION { ?a ?b ?c } }");
    assert_eq!(
        query.to_sse(),
        "(project (?a ?b ?c ?o ?p ?s) (union (bgp (triple ?s ?p ?o)) (bgp (triple ?a ?b ?c))))"
    );
}

#[test]
fn prefixed_and_a() {
    let query = q("PREFIX ex: <http://example.com/> SELECT * WHERE { ?s a ex:C }");
    assert_eq!(
        query.to_sse(),
        "(project (?s) (bgp (triple ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.com/C>)))"
    );
}

#[test]
fn property_path() {
    let query = q("SELECT * WHERE { ?s <http://p>+ ?o }");
    assert_eq!(
        query.to_sse(),
        "(project (?o ?s) (path ?s (path+ <http://p>) ?o))"
    );
}

#[test]
fn aggregate_group() {
    let query = q("SELECT (COUNT(?o) AS ?c) WHERE { ?s ?p ?o } GROUP BY ?s");
    // The aggregate is projected, so it round-trips through Display back to itself.
    assert_eq!(q(&query.to_string()).to_sse(), query.to_sse());
}

#[test]
fn values_clause() {
    let query = q("SELECT * WHERE { VALUES ?x { 1 2 } }");
    assert_eq!(
        query.to_sse(),
        "(project (?x) (table (vars ?x) (row (?x \"1\"^^<http://www.w3.org/2001/XMLSchema#integer>)) (row (?x \"2\"^^<http://www.w3.org/2001/XMLSchema#integer>))))"
    );
}

#[test]
fn construct() {
    let query = q("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }");
    assert!(query.to_sse().starts_with("(construct"));
}

#[test]
fn describe() {
    let query = q("DESCRIBE <http://example.com/>");
    assert!(matches!(query, Query::Describe(_)));
}

#[test]
fn update_clear() {
    let update = u("CLEAR ALL");
    assert_eq!(update.to_sse(), "(update (clear all))");
    assert_eq!(update.to_string().trim(), "CLEAR ALL ;");
}

#[test]
fn update_insert_data() {
    let update = u("INSERT DATA { <http://s> <http://p> <http://o> }");
    assert_eq!(
        update.to_sse(),
        "(update (insertData ((triple <http://s> <http://p> <http://o>))))"
    );
}

#[test]
fn update_delete_where() {
    let update = u("DELETE WHERE { ?s ?p ?o }");
    assert!(update.to_sse().starts_with("(update (modify"));
}

#[test]
fn base_iri_resolution() {
    let query = SparqlParser::new()
        .with_base_iri("http://example.com/")
        .expect("valid base")
        .parse_query("SELECT * WHERE { <s> <p> <o> }")
        .expect("parse");
    assert_eq!(
        query.to_string(),
        "BASE <http://example.com/>\nSELECT * WHERE { <http://example.com/s> <http://example.com/p> <http://example.com/o> . }"
    );
}

#[test]
fn negative_syntax_is_rejected() {
    let bad = [
        // `SELECT` needs either `*` or at least one projection member.
        "SELECT WHERE { }",
        // A variable may not be projected twice.
        "SELECT ?s ?s { }",
        // `AS` may not override a variable that is already in scope.
        "SELECT (?s AS ?s) WHERE { ?s ?p ?o }",
        // Aggregates do not nest.
        "SELECT (COUNT(SUM(?o)) AS ?c) WHERE { ?s ?p ?o }",
        // `BIND` may not override a variable that is already in scope.
        "SELECT * WHERE { ?s ?p ?o BIND(1 AS ?o) }",
        // The longest-token rule turns `<?a&&?b>` into an IRI, so this is not a comparison.
        "SELECT * WHERE { FILTER (?x<?a&&?b>?y) }",
        // `SELECT *` is incompatible with grouping.
        "SELECT * WHERE { ?s ?p ?o } GROUP BY ?s",
        // Undeclared prefix.
        "SELECT * WHERE { ?s ex:p ?o }",
        // Not a query at all.
        "not a query",
        "",
    ];
    for query in bad {
        assert!(
            SparqlParser::new().parse_query(query).is_err(),
            "{query:?} should have been rejected"
        );
    }
    let bad_updates = [
        // Blank node labels are scoped to the whole update request.
        "INSERT DATA { _:b <http://p> <http://o> } ; INSERT DATA { _:b <http://p> <http://o> }",
        // `DELETE DATA` accepts neither variables nor blank nodes.
        "DELETE DATA { ?s <http://p> <http://o> }",
        "DELETE DATA { _:b <http://p> <http://o> }",
        // `INSERT DATA` accepts no variables.
        "INSERT DATA { ?s <http://p> <http://o> }",
        // `DELETE WHERE` accepts no blank nodes.
        "DELETE WHERE { _:b <http://p> <http://o> }",
    ];
    for update in bad_updates {
        assert!(
            SparqlParser::new().parse_update(update).is_err(),
            "{update:?} should have been rejected"
        );
    }
}

#[test]
fn a_keyword_respects_the_longest_token_rule() {
    // `a:` is the `PNAME_NS` of the prefix `a`, never `rdf:type` followed by `:`.
    let query =
        q("PREFIX a: <http://example.com/> PREFIX : <http://x/> SELECT * WHERE { : a: :a }");
    assert_eq!(
        query.to_sse(),
        "(project () (bgp (triple <http://x/> <http://example.com/> <http://x/a>)))"
    );
}

#[test]
fn with_clause_round_trips() {
    let update = u("PREFIX : <http://example/> WITH :g DELETE { ?s ?p ?o } WHERE { ?s ?p ?o }");
    let again = u(&update.to_string());
    assert_eq!(again, update);
    assert!(update.to_string().starts_with("WITH <http://example/g>"));
}

#[test]
fn service_and_subquery() {
    let query = q(
        "SELECT * WHERE { SERVICE SILENT <http://example.com/sparql> { { SELECT ?s WHERE { ?s ?p ?o } LIMIT 1 } } }",
    );
    assert_eq!(q(&query.to_string()).to_sse(), query.to_sse());
    assert!(query.to_sse().contains("(service silent"));
}

#[test]
fn all_update_forms_round_trip() {
    for update in [
        "LOAD SILENT <http://example.com/g> INTO GRAPH <http://example.com/h>",
        "CLEAR NAMED",
        "DROP SILENT GRAPH <http://example.com/g>",
        "CREATE GRAPH <http://example.com/g>",
        "ADD GRAPH <http://example.com/g> TO DEFAULT",
        "MOVE DEFAULT TO GRAPH <http://example.com/g>",
        "COPY GRAPH <http://example.com/g> TO GRAPH <http://example.com/h>",
        "INSERT DATA { GRAPH <http://example.com/g> { <http://s> <http://p> 1 } }",
        "DELETE DATA { <http://s> <http://p> \"x\"@en }",
        "DELETE WHERE { ?s <http://p> ?o }",
        "DELETE { ?s <http://p> ?o } INSERT { ?s <http://q> ?o } USING <http://example.com/g> USING NAMED <http://example.com/h> WHERE { ?s ?p ?o }",
    ] {
        let parsed = u(update);
        assert_eq!(u(&parsed.to_string()), parsed, "{update}");
    }
}

#[test]
fn property_path_precedence() {
    let query = q("SELECT * WHERE { ?s <http://a>/<http://b>|^<http://c>* ?o }");
    assert_eq!(
        query.to_sse(),
        "(project (?o ?s) (path ?s (alt (seq <http://a> <http://b>) (reverse (path* <http://c>))) ?o))"
    );
    let negated = q("SELECT * WHERE { ?s !(<http://a>|^<http://b>) ?o }");
    assert_eq!(
        negated.to_sse(),
        "(project (?o ?s) (path ?s (alt (notoneof <http://a>) (reverse (notoneof <http://b>))) ?o))"
    );
}

#[test]
fn escapes_and_comments() -> Result<(), Box<dyn std::error::Error>> {
    let query = SparqlParser::new()
        .parse_query("PREFIX ex: <http://example.com/> # a comment\nSELECT * WHERE { ex:a\\-b ex:p \"tab:\\tend\" }")?;
    let sse = query.to_sse();
    assert!(sse.contains("<http://example.com/a-b>"), "{sse}");
    assert!(sse.contains(r"tab:\tend"), "{sse}");
    Ok(())
}

#[cfg(feature = "sparql-12")]
mod sparql_12 {
    use super::{q, u};

    #[test]
    fn triple_term() {
        let query = q("SELECT * WHERE { ?r <http://p> <<( ?s <http://q> ?o )>> }");
        assert_eq!(
            query.to_sse(),
            "(project (?o ?r ?s) (bgp (triple ?r <http://p> (triple ?s <http://q> ?o))))"
        );
        assert_eq!(q(&query.to_string()), query);
    }

    #[test]
    fn reified_triple_uses_rdf_reifies() {
        let query = q("SELECT * WHERE { << <http://s> <http://p> <http://o> ~ ?r >> }");
        assert_eq!(
            query.to_sse(),
            "(project (?r) (bgp (triple ?r <http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies> (triple <http://s> <http://p> <http://o>))))"
        );
        assert_eq!(q(&query.to_string()), query);
    }

    #[test]
    fn annotation_block() {
        let query =
            q("SELECT * WHERE { <http://s> <http://p> <http://o> ~ ?r {| <http://q> ?v |} }");
        let sse = query.to_sse();
        assert!(sse.contains("(triple ?r <http://q> ?v)"), "{sse}");
        assert!(
            sse.contains(
                "(triple ?r <http://www.w3.org/1999/02/22-rdf-syntax-ns#reifies> (triple <http://s> <http://p> <http://o>))"
            ),
            "{sse}"
        );
        assert_eq!(q(&query.to_string()), query);
    }

    #[test]
    fn version_declaration_is_accepted() {
        let query = q("VERSION \"1.2\" SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(
            query.to_sse(),
            "(project (?o ?p ?s) (bgp (triple ?s ?p ?o)))"
        );
    }

    #[test]
    fn triple_term_builtins() {
        let query =
            q("SELECT * WHERE { ?s ?p ?o FILTER(isTRIPLE(?o) && SUBJECT(?o) = TRIPLE(1, 2, 3)) }");
        assert_eq!(q(&query.to_string()).to_sse(), query.to_sse());
        assert!(query.to_sse().contains("sparql#isTriple"));
        assert!(query.to_sse().contains("sparql#subject"));
        assert!(query.to_sse().contains("sparql#triple"));
    }

    #[test]
    fn directional_language_tagged_string() {
        let query = q("SELECT * WHERE { ?s ?p \"hello\"@en--ltr }");
        assert!(query.to_string().contains("@en--ltr"), "{query}");
        assert_eq!(q(&query.to_string()), query);
    }

    #[test]
    fn codepoint_escapes_are_expanded_before_tokenizing() {
        // The whole query written with codepoint escapes.
        let query = q("\\u0041\\u0053\\u004B\\u0020\\u007B\\u007D");
        assert_eq!(query.to_sse(), "(ask (bgp))");
        // `\\u0041` decodes to the invalid string escape `\\A`.
        assert!(
            super::SparqlParser::new()
                .parse_query("SELECT * WHERE { ?s ?p \"\\\\u0041\" }")
                .is_err()
        );
    }

    #[test]
    fn triple_term_in_insert_data() {
        let update = u("INSERT DATA { <http://r> <http://p> <<( <http://s> <http://q> 1 )>> }");
        assert_eq!(u(&update.to_string()), update);
        // Ground quads keep the N-Triples rendering of the embedded triple term, exactly like
        // `spargebra` does.
        assert!(
            update
                .to_sse()
                .contains("(triple <http://r> <http://p> <<( <http://s> <http://q> "),
            "{}",
            update.to_sse()
        );
    }
}
