//! Regression tests for audited parser and serializer defects.
//!
//! Every test here reproduces a concrete input that used to be mis-parsed, wrongly accepted,
//! wrongly rejected, mis-printed or that used to abort the process.

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

fn count_occurrences(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

// --- generated blank nodes never collide with user-written labels -----------

#[test]
fn anonymous_blank_node_does_not_merge_with_a_user_label() {
    // `[]` used to be named `_:g0`, silently merging with the user's `_:g0`.
    let query = q("SELECT * WHERE { _:g0 <http://e/p> 1 . [] <http://e/q> 2 }");
    let sse = query.to_sse();
    assert_eq!(count_occurrences(&sse, "_:g0 "), 1, "{sse}");
    assert!(sse.contains("(triple _:g0 <http://e/p>"), "{sse}");
    assert!(!sse.contains("(triple _:g0 <http://e/q>"), "{sse}");
}

#[test]
fn insert_data_keeps_the_anonymous_node_distinct() {
    let update = u("INSERT DATA { _:g0 <http://e/p> 1 . [] <http://e/q> 2 }");
    let printed = update.to_string();
    assert_eq!(count_occurrences(&printed, "_:g0 "), 1, "{printed}");
    assert!(printed.contains("_:g0 <http://e/p>"), "{printed}");
    assert!(!printed.contains("_:g0 <http://e/q>"), "{printed}");
}

#[test]
fn collection_cells_do_not_collide_with_a_user_label() {
    let update = u("INSERT DATA { _:g0 <http://e/p> 1 . <http://e/s> <http://e/q> (2 3) }");
    let sse = update.to_sse();
    assert_eq!(count_occurrences(&sse, "_:g0 "), 1, "{sse}");
}

#[test]
fn property_path_expansion_does_not_collide_with_a_user_label() {
    let query = q("SELECT * WHERE { _:g0 <http://e/p> 1 . ?x <http://e/a>/<http://e/b> ?y }");
    let sse = query.to_sse();
    assert_eq!(count_occurrences(&sse, "_:g0 "), 1, "{sse}");
    assert!(sse.contains("(triple _:g0 <http://e/p>"), "{sse}");
}

#[test]
fn generated_labels_stay_clear_of_every_user_label_shape() {
    // Whatever the user writes, the generated label is never one of theirs.
    for label in ["g0", "gg0", "b0", "gb0", "ggb0"] {
        let query = q(&format!(
            "SELECT * WHERE {{ _:{label} <http://e/p> 1 . [] <http://e/q> 2 }}"
        ));
        let sse = query.to_sse();
        assert_eq!(count_occurrences(&sse, &format!("_:{label} ")), 1, "{sse}");
    }
}

// --- generated variables never collide with user-written variables ----------

#[test]
fn aggregate_variable_does_not_clobber_the_user_variable() {
    let query = q("SELECT ?0 (COUNT(*) AS ?c) WHERE { ?0 ?p ?o } GROUP BY ?0");
    let sse = query.to_sse();
    // The group key stays `?0`, the count gets its own variable.
    assert!(sse.contains("(group (?0) "), "{sse}");
    assert!(!sse.contains("((count) ?0)"), "{sse}");
    assert!(!sse.contains("(extend ((?c ?0))"), "{sse}");
}

#[test]
fn user_may_name_an_aggregate_output_zero() {
    let query = q("SELECT (COUNT(*) AS ?0) WHERE { ?s ?p ?o }");
    assert!(
        query.to_sse().contains("(extend ((?0 ?"),
        "{}",
        query.to_sse()
    );
}

#[test]
fn describe_target_does_not_clash_with_a_user_variable() {
    let query = q("DESCRIBE <http://e/a> WHERE { ?0 ?p ?o }");
    assert!(matches!(query, Query::Describe(_)));
}

#[test]
fn aggregate_with_having_round_trips_through_display() {
    let query = q("SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } HAVING(COUNT(*) > 1)");
    let printed = query.to_string();
    let again = q(&printed);
    assert!(again.to_sse().contains("(count)"), "{printed}");
}

// --- GROUP BY (?x AS ?y) ----------------------------------------------------

#[test]
fn group_by_rename_of_a_bare_variable_binds_the_variable() {
    let query = q("SELECT ?y WHERE { ?x ?p ?o } GROUP BY (?x AS ?y)");
    let sse = query.to_sse();
    assert!(sse.contains("(extend ((?y ?x))"), "{sse}");
    assert!(sse.contains("(group (?y) ()"), "{sse}");

    let query = q("SELECT ?y (COUNT(*) AS ?c) WHERE { ?x ?p ?o } GROUP BY (?x AS ?y)");
    let sse = query.to_sse();
    assert!(sse.contains("(extend ((?y ?x))"), "{sse}");
    assert!(sse.contains("(group (?y) "), "{sse}");

    // A non-variable expression keeps working.
    let query = q("SELECT ?y WHERE { ?x ?p ?o } GROUP BY (?x + 1 AS ?y)");
    assert!(
        query.to_sse().contains("(group (?y) ()"),
        "{}",
        query.to_sse()
    );
}

// --- aggregates are not allowed in FILTER / BIND / GROUP BY -----------------

#[test]
fn aggregates_are_rejected_outside_of_aggregation_contexts() {
    for query in [
        "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o FILTER(SUM(?o) > 1) }",
        "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o BIND(SUM(?o) AS ?x) }",
        "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } GROUP BY (COUNT(?o) AS ?z)",
        "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } GROUP BY (COUNT(?o))",
        "SELECT * WHERE { ?s ?p ?o FILTER(COUNT(*) > 1) }",
    ] {
        assert!(
            SparqlParser::new().parse_query(query).is_err(),
            "{query:?} should have been rejected"
        );
    }
}

#[test]
fn aggregates_stay_allowed_where_the_specification_permits_them() {
    // HAVING and ORDER BY are evaluated above the Group node.
    q("SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } HAVING(COUNT(*) > 1)");
    q("SELECT ?s WHERE { ?s ?p ?o } GROUP BY ?s ORDER BY DESC(COUNT(*))");
    // A FILTER inside a nested sub-SELECT opens a new aggregation scope.
    q("SELECT * WHERE { ?s ?p ?o FILTER EXISTS { SELECT (COUNT(*) AS ?c) WHERE { ?a ?b ?d } } }");
    q("SELECT * WHERE { { SELECT (COUNT(*) AS ?c) WHERE { ?a ?b ?d } HAVING(COUNT(*) > 1) } }");
}

// --- ORDER BY on a non-projected aggregate ---------------------------------

#[test]
fn order_by_a_non_projected_aggregate_survives_serialization() {
    let query = q("SELECT ?s WHERE { ?s ?p ?o } GROUP BY ?s ORDER BY DESC(COUNT(*))");
    assert!(
        query.to_sse().contains("(order ((desc ?"),
        "{}",
        query.to_sse()
    );
    let printed = query.to_string();
    assert!(printed.contains("COUNT(*)"), "{printed}");
    // The aggregate must still be defined in the printed query, and the ordering must still be on
    // it rather than on an unbound variable.
    let again = q(&printed);
    let sse = again.to_sse();
    assert!(sse.contains("(count)"), "{printed} -> {sse}");
    assert!(sse.contains("(order ((desc ?"), "{printed} -> {sse}");
}

// --- VALUES with a repeated variable ---------------------------------------

#[test]
fn values_rejects_a_repeated_variable() {
    for query in [
        "SELECT * WHERE { VALUES (?a ?a) { (1 2) } }",
        "SELECT * WHERE { VALUES (?a ?b ?a) { (1 2 3) } }",
    ] {
        assert!(
            SparqlParser::new().parse_query(query).is_err(),
            "{query:?} should have been rejected"
        );
    }
    // Distinct variables keep working.
    let query = q("SELECT * WHERE { VALUES (?a ?b) { (1 2) } }");
    assert!(
        query.to_sse().contains("(vars ?a ?b)"),
        "{}",
        query.to_sse()
    );
}

// --- keywords follow the longest-token rule --------------------------------

#[test]
fn keyword_pairs_need_a_token_boundary() {
    for query in [
        "SELECT ?s WHERE { ?s ?p ?o } GROUPBY ?s",
        "SELECT ?s WHERE { ?s ?p ?o } ORDERBY ?s",
        "SELECT * WHERE { ?s ?p ?o FILTER(NOTEXISTS { ?s ?p ?o }) }",
        "SELECTDISTINCT ?s WHERE { ?s ?p ?o }",
        "SELECT ?s WHERE { ?s ?p ?o } ORDER BY ASC?s",
    ] {
        assert!(
            SparqlParser::new().parse_query(query).is_err(),
            "{query:?} should have been rejected"
        );
    }
    for update in [
        "INSERTDATA { <http://e/a> <http://e/b> <http://e/c> }",
        "DELETEWHERE { ?s <http://e/p> ?o }",
    ] {
        assert!(
            SparqlParser::new().parse_update(update).is_err(),
            "{update:?} should have been rejected"
        );
    }
    // The properly spaced forms still parse.
    q("SELECT ?s WHERE { ?s ?p ?o } GROUP BY ?s");
    q("SELECT ?s WHERE { ?s ?p ?o } ORDER BY ?s");
    q("SELECT * WHERE { ?s ?p ?o FILTER(NOT EXISTS { ?s ?p ?o }) }");
    q("SELECT DISTINCT ?s WHERE { ?s ?p ?o }");
    u("INSERT DATA { <http://e/a> <http://e/b> <http://e/c> }");
    u("DELETE WHERE { ?s <http://e/p> ?o }");
}

#[test]
fn a_keyword_prefix_is_still_a_prefixed_name() {
    // `VALUES:` is the `PNAME_NS` of the prefix `VALUES`, not the `VALUES` keyword.
    let query =
        q("PREFIX VALUES: <http://e/> SELECT * WHERE { <http://e/s> <http://e/p> VALUES:o }");
    assert!(
        query.to_sse().contains("<http://e/o>"),
        "{}",
        query.to_sse()
    );
}

// --- unbounded recursion ----------------------------------------------------

#[test]
fn deeply_nested_expressions_are_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let query = format!(
        "SELECT * WHERE {{ ?s ?p ?o . FILTER({}1{}) }}",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn deeply_nested_group_patterns_are_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let query = format!(
        "SELECT * WHERE {}?s ?p ?o .{}",
        "{".repeat(depth),
        "}".repeat(depth)
    );
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn deeply_nested_collections_and_paths_are_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let collections = format!(
        "SELECT * WHERE {{ <http://e/s> <http://e/p> {}1{} }}",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    assert!(SparqlParser::new().parse_query(&collections).is_err());

    let paths = format!(
        "SELECT * WHERE {{ ?s {}<http://e/p>{} ?o }}",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    assert!(SparqlParser::new().parse_query(&paths).is_err());

    let property_lists = format!(
        "SELECT * WHERE {{ {}<http://e/p> 1{} }}",
        "[ <http://e/p> ".repeat(depth),
        "]".repeat(depth)
    );
    assert!(SparqlParser::new().parse_query(&property_lists).is_err());

    let functions = format!(
        "SELECT * WHERE {{ ?s ?p ?o FILTER({}1{}) }}",
        "ABS(".repeat(depth),
        ")".repeat(depth)
    );
    assert!(SparqlParser::new().parse_query(&functions).is_err());
}

#[test]
fn reasonable_nesting_is_still_accepted() {
    let depth = 24;
    let query = format!(
        "SELECT * WHERE {{ ?s ?p ?o . FILTER({}1{} = 1) }}",
        "(".repeat(depth),
        ")".repeat(depth)
    );
    q(&query);
    let query = format!(
        "SELECT * WHERE {}?s ?p ?o .{}",
        "{".repeat(depth),
        "}".repeat(depth)
    );
    q(&query);
}

// --- unbounded recursion via *flat* (non-nested) operator chains -----------
//
// The recursion-depth guard above only charged one unit per syntactic *nesting* level. A flat
// chain of N operator-joined elements (no parentheses/braces anywhere) folds into a tree N-1
// levels deep without ever nesting syntactically, so every one of these used to escape the guard
// entirely and overflow the stack while parsing, while building the fold, or later while
// printing or dropping the resulting (equally native-recursive) tree -- regardless of
// `MAX_RECURSION_DEPTH`. Each construct below now charges the same shared guard once per fold
// level, so a chain long enough to matter is rejected with the ordinary "nested too deeply"
// syntax error instead.

#[test]
fn flat_property_path_sequence_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let path = vec!["<http://e/p>"; depth].join("/");
    let query = format!("SELECT * WHERE {{ ?s {path} ?o }}");
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn flat_property_path_alternative_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let path = vec!["<http://e/p>"; depth].join("|");
    let query = format!("SELECT * WHERE {{ ?s {path} ?o }}");
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn flat_conditional_or_expression_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let expr = vec!["1=1"; depth].join(" || ");
    let query = format!("SELECT * WHERE {{ ?s ?p ?o FILTER({expr}) }}");
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn flat_conditional_and_expression_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let expr = vec!["1=1"; depth].join(" && ");
    let query = format!("SELECT * WHERE {{ ?s ?p ?o FILTER({expr}) }}");
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn flat_additive_expression_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let expr = vec!["1"; depth].join(" + ");
    let query = format!("SELECT * WHERE {{ ?s ?p ?o FILTER({expr} > 0) }}");
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn flat_union_chain_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let arms = vec!["{ ?s ?p ?o }"; depth].join(" UNION ");
    let query = format!("SELECT * WHERE {{ {arms} }}");
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn flat_having_chain_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let conds = vec!["(1=1)"; depth].join(" ");
    let query = format!("SELECT (COUNT(*) AS ?c) WHERE {{ ?s ?p ?o }} GROUP BY ?s HAVING {conds}");
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn flat_triples_block_of_property_paths_is_rejected_without_overflowing_the_stack() {
    // A run of one-or-more-typed paths, each its own `GraphPattern::Path`, exercises
    // `build_bgp`'s/`fold_graph_patterns`' per-statement charge. A plain sequence (`<a>/<b>`)
    // would *not* reach it: `add_to_triple_or_path` expands a `Sequence` into ordinary triple
    // patterns joined by a fresh blank node, so a run of those collapses into a single `Bgp`
    // block regardless of how many statements there are.
    let depth = 100_000;
    let mut query = String::from("SELECT * WHERE { ");
    for i in 0..depth {
        query.push_str(&format!("?s{i} <http://e/p>+ ?o{i} . "));
    }
    query.push('}');
    assert!(SparqlParser::new().parse_query(&query).is_err());
}

#[test]
fn flat_delete_where_quad_chain_is_rejected_without_overflowing_the_stack() {
    // The same `fold_graph_patterns` fold as the previous test, reached through `parse_update`
    // rather than `parse_query`, proving both public parsing entry points are guarded.
    let depth = 100_000;
    let mut update = String::from("DELETE WHERE { ");
    for i in 0..depth {
        update.push_str(&format!("?s{i} <http://e/p{i}> ?o{i} . "));
    }
    update.push('}');
    assert!(SparqlParser::new().parse_update(&update).is_err());
}

#[test]
fn flat_chain_of_functions_in_insert_data_is_rejected_without_overflowing_the_stack() {
    // `parse_update`'s other data-modifying form (`INSERT DATA`, ground triples only) does not
    // itself fold a flat chain, but a `DELETE { ... } WHERE { ... FILTER(...) }` general update
    // reaches the very same `conditional_or_expression` fold as `parse_query` does.
    let depth = 100_000;
    let expr = vec!["1=1"; depth].join(" || ");
    let update = format!("DELETE {{ ?s ?p ?o }} WHERE {{ ?s ?p ?o . FILTER({expr}) }}");
    assert!(SparqlParser::new().parse_update(&update).is_err());
}

/// A parsed value that is never printed and never even bound to a bare `Result` must still not
/// abort the process: a stack overflow happens on `Drop`, not on `Display`, so merely letting a
/// value go out of scope has to be checked on its own (`#[should_panic]` cannot express an abort,
/// see the crate's contribution notes). Since the fix rejects an over-long chain at parse time,
/// there is no oversized tree to drop in the first place — this exercises exactly that: parsing
/// runs to completion inside the block below, and only the small `SparqlSyntaxError` (never a
/// multi-million-node `Query`) is dropped when it ends.
#[test]
fn flat_chain_past_the_limit_never_builds_a_value_to_drop() {
    let depth = 100_000;
    {
        let arms = vec!["{ ?s ?p ?o }"; depth].join(" UNION ");
        let query = format!("SELECT * WHERE {{ {arms} }}");
        let result = SparqlParser::new().parse_query(&query);
        assert!(result.is_err());
    }
}

// --- the shared 48-deep recursion guard's exact cost on flat chains --------
//
// These pin the concrete boundary a few representative constructs hit today, rather than just
// "some sufficiently long chain fails": `MAX_RECURSION_DEPTH` (48) is shared with every other
// recursive production (see `parser/mod.rs`), so the exact number depends on how much depth is
// already spent entering the surrounding productions (`group_graph_pattern`, `expression`, ...)
// before the fold itself starts charging. A real, auto-generated query with a flat chain longer
// than these boundaries — a `FILTER` of many OR'd ids, a `UNION` of many alternatives — is
// rejected today where it would have parsed (and then likely crashed) before this fix.

#[test]
fn flat_property_path_sequence_boundary() {
    let boundary = 47;
    q(&format!(
        "SELECT * WHERE {{ ?s {} ?o }}",
        vec!["<http://e/p>"; boundary].join("/")
    ));
    let query = format!(
        "SELECT * WHERE {{ ?s {} ?o }}",
        vec!["<http://e/p>"; boundary + 1].join("/")
    );
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("one step past the boundary must be rejected");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

#[test]
fn flat_union_chain_boundary() {
    let boundary = 46;
    q(&format!(
        "SELECT * WHERE {{ {} }}",
        vec!["{ ?s ?p ?o }"; boundary].join(" UNION ")
    ));
    let query = format!(
        "SELECT * WHERE {{ {} }}",
        vec!["{ ?s ?p ?o }"; boundary + 1].join(" UNION ")
    );
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("one arm past the boundary must be rejected");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

#[test]
fn flat_having_chain_boundary() {
    let boundary = 48;
    q(&format!(
        "SELECT (COUNT(*) AS ?c) WHERE {{ ?s ?p ?o }} GROUP BY ?s HAVING {}",
        vec!["(1=1)"; boundary].join(" ")
    ));
    let query = format!(
        "SELECT (COUNT(*) AS ?c) WHERE {{ ?s ?p ?o }} GROUP BY ?s HAVING {}",
        vec!["(1=1)"; boundary + 1].join(" ")
    );
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("one condition past the boundary must be rejected");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

#[test]
fn flat_chain_depth_composes_with_surrounding_syntactic_nesting() {
    // The guard is one shared counter: a flat chain's charge and genuine syntactic nesting add
    // up rather than being tracked separately, so a group with many clauses leaves correspondingly
    // less budget for nesting inside the next one. Each `OPTIONAL` clause and each parenthesis
    // level costs exactly one charge, and (empirically) 46 total is the shared budget here.
    //
    // 20 `OPTIONAL` clauses (20 charges) plus 27 levels of parenthesized nesting (27 more) is 47,
    // one past that budget, and is rejected -- even though 27 levels of parenthesized nesting on
    // its own parses just fine (it is comfortably under 46 by itself). 26 levels instead of 27
    // brings the same query back under the shared budget.
    let optionals = "OPTIONAL { ?s <http://e/p> ?o } ".repeat(20);

    let lone_nesting = format!(
        "SELECT * WHERE {{ ?s ?p ?o FILTER({}1{} = 1) }}",
        "(".repeat(27),
        ")".repeat(27)
    );
    q(&lone_nesting);

    let combined_shallow = format!(
        "SELECT * WHERE {{ ?s ?p ?o . {optionals}FILTER({}1{} = 1) }}",
        "(".repeat(26),
        ")".repeat(26)
    );
    q(&combined_shallow);

    let combined_deep = format!(
        "SELECT * WHERE {{ ?s ?p ?o . {optionals}FILTER({}1{} = 1) }}",
        "(".repeat(27),
        ")".repeat(27)
    );
    assert!(
        SparqlParser::new().parse_query(&combined_deep).is_err(),
        "27 levels of nesting alone parses, but not after 20 preceding OPTIONAL clauses"
    );
}

// --- unbounded recursion via nested RDF-1.2 triple terms (`sparql-12`) -----
//
// `<<( ... )>>` triple terms, their ground (`INSERT DATA`/`DELETE DATA`/`VALUES`) and
// expression-context counterparts, and `<< ... >>` reified triples can each nest inside their
// own subject or object position and used to recurse straight back into the same production with
// no depth accounting at all -- a different shape of the same underlying defect as the flat
// chains above (this one is genuine *syntactic* nesting, just never wired up to the shared
// guard). A query built from tens of thousands of nested levels overflowed the stack while
// parsing, well before any tree was ever printed or dropped.

#[cfg(feature = "sparql-12")]
#[test]
fn nested_triple_term_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let mut term = String::new();
    for _ in 0..depth {
        term.push_str("<<(<http://e/s> <http://e/p> ");
    }
    term.push_str("<http://e/o>");
    term.push_str(&")>>".repeat(depth));
    let query = format!("SELECT * WHERE {{ ?s ?p {term} }}");
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("a triple term nested this deeply must be rejected, not crash");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

#[cfg(feature = "sparql-12")]
#[test]
fn nested_triple_term_data_in_values_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let mut term = String::new();
    for _ in 0..depth {
        term.push_str("<<(<http://e/s> <http://e/p> ");
    }
    term.push_str("<http://e/o>");
    term.push_str(&")>>".repeat(depth));
    let query = format!("SELECT * WHERE {{ VALUES ?x {{ {term} }} }}");
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("a ground triple term nested this deeply must be rejected, not crash");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

#[cfg(feature = "sparql-12")]
#[test]
fn nested_triple_term_data_in_insert_data_is_rejected_without_overflowing_the_stack() {
    // The same `TripleTermData` production as the `VALUES` case above, reached through
    // `parse_update` rather than `parse_query`.
    let depth = 100_000;
    let mut term = String::new();
    for _ in 0..depth {
        term.push_str("<<(<http://e/s> <http://e/p> ");
    }
    term.push_str("<http://e/o>");
    term.push_str(&")>>".repeat(depth));
    let update = format!("INSERT DATA {{ <http://e/s> <http://e/p> {term} }}");
    let err = SparqlParser::new()
        .parse_update(&update)
        .expect_err("a ground triple term nested this deeply must be rejected, not crash");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

#[cfg(feature = "sparql-12")]
#[test]
fn nested_expr_triple_term_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let mut term = String::new();
    for _ in 0..depth {
        term.push_str("<<(<http://e/s> <http://e/p> ");
    }
    term.push_str("<http://e/o>");
    term.push_str(&")>>".repeat(depth));
    let query = format!("SELECT * WHERE {{ ?s ?p ?o FILTER(?s = {term}) }}");
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("an expression triple term nested this deeply must be rejected, not crash");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

#[cfg(feature = "sparql-12")]
#[test]
fn nested_reified_triple_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let mut pattern = "<< ".repeat(depth);
    pattern.push_str("<http://e/s> <http://e/p> <http://e/o>");
    pattern.push_str(&" >>".repeat(depth));
    let query = format!("SELECT * WHERE {{ {pattern} ?p2 ?o2 }}");
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("a reified triple nested this deeply must be rejected, not crash");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

/// An annotation block (`{| ... |}`) nested inside another recurses back into `annotation_path`
/// through `property_list_path_not_empty`'s own object list -- the fifth self-recursive
/// `sparql-12` production, alongside the four above, and the only one that is not itself a triple
/// term or reified triple.
#[cfg(feature = "sparql-12")]
#[test]
fn nested_annotation_block_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let mut pattern = String::from("?s <http://e/p> ?o");
    for _ in 0..depth {
        pattern.push_str(" {| <http://e/p2> ?o2");
    }
    pattern.push_str(&" |}".repeat(depth));
    let query = format!("SELECT * WHERE {{ {pattern} }}");
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("an annotation block nested this deeply must be rejected, not crash");
    assert!(err.to_string().contains("nested too deeply"), "{err}");
}

/// The same construct as [`nested_annotation_block_is_rejected_without_overflowing_the_stack`],
/// through a `CONSTRUCT` template's *non-path* `annotation`/`property_list_not_empty` route
/// instead of a `WHERE` clause's path-aware one.
///
/// `construct_template`'s own loop used to check `.is_err()` without distinguishing a fatal
/// error from an ordinary, recoverable one: a fatal "nested too deeply" error from deep inside
/// the template was silently reinterpreted as "this template happens to be empty", backtracking
/// all the way to the start and going on to report a confusing "expected token" failure at the
/// (unrelated) next token instead of the real, actionable error -- itself never a crash, but a
/// second, independent bug this same construct exposed.
#[cfg(feature = "sparql-12")]
#[test]
fn nested_annotation_block_in_construct_template_is_rejected_without_overflowing_the_stack() {
    let depth = 100_000;
    let mut pattern = String::from("?s <http://e/p> ?o");
    for _ in 0..depth {
        pattern.push_str(" {| <http://e/p2> ?o2");
    }
    pattern.push_str(&" |}".repeat(depth));
    let query = format!("CONSTRUCT {{ {pattern} }} WHERE {{ ?s ?p ?o }}");
    let err = SparqlParser::new()
        .parse_query(&query)
        .expect_err("an annotation block nested this deeply must be rejected, not crash");
    assert!(
        err.to_string().contains("nested too deeply"),
        "must report the real, fatal cause rather than a confusing backtracked one: {err}"
    );
}

// --- documented divergences from `spargebra`'s printed form -----------------

#[test]
fn ask_is_printed_without_an_extra_projection() {
    // Documented divergence from `spargebra`, which wraps the ASK pattern in a projection of the
    // in-scope variables. Both forms answer the same boolean.
    let query = q("ASK { ?s ?p ?o }");
    assert_eq!(query.to_sse(), "(ask (bgp (triple ?s ?p ?o)))");
    assert_eq!(query.to_string(), "ASK WHERE { ?s ?p ?o . }");
    assert_eq!(q(&query.to_string()), query);
}

#[test]
fn projected_aggregates_are_printed_flat() {
    // Documented divergence: `spargebra` prints a nested sub-SELECT with a random variable name.
    let query = q("SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o } GROUP BY ?p");
    assert_eq!(
        query.to_string(),
        "SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o . } GROUP BY ?p"
    );
    assert_eq!(q(&query.to_string()).to_sse(), query.to_sse());
}
