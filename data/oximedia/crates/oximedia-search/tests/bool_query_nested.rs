//! Integration tests for nested boolean-query evaluation.
//!
//! These exercise the real [`oximedia_search::bool_query`] evaluator
//! ([`BoolQuery::matches`]) against a fixed corpus of documents represented as
//! term sets, asserting EXACT matching counts for complex nested
//! `AND` / `OR` / `NOT` / phrase expressions.

use oximedia_search::bool_query::{terms_from_text, BoolQuery};
use std::collections::HashSet;

/// A small, fixed corpus of documents (each lowered to its set of terms via the
/// crate's real [`terms_from_text`] tokenizer).
///
/// Index → contents:
/// 0. "cat dog fish"
/// 1. "cat bird"
/// 2. "dog fish bird"
/// 3. "cat dog bird snake"
/// 4. "snake lizard"
/// 5. "fish only"
/// 6. "bird snake"
/// 7. "cat fish snake"
/// 8. "lizard frog"
/// 9. "dog frog cat"
fn corpus() -> Vec<HashSet<String>> {
    [
        "cat dog fish",
        "cat bird",
        "dog fish bird",
        "cat dog bird snake",
        "snake lizard",
        "fish only",
        "bird snake",
        "cat fish snake",
        "lizard frog",
        "dog frog cat",
    ]
    .iter()
    .map(|text| terms_from_text(text))
    .collect()
}

/// Count how many documents in the corpus match `query`.
fn match_count(query: &BoolQuery, docs: &[HashSet<String>]) -> usize {
    docs.iter().filter(|d| query.matches(d)).count()
}

#[test]
fn case_01_simple_or_two_terms() {
    let docs = corpus();
    // cat OR dog → docs {0,1,2,3,7,9} = 6.
    let q = BoolQuery::or(vec![BoolQuery::term("cat"), BoolQuery::term("dog")]);
    assert_eq!(match_count(&q, &docs), 6);
}

#[test]
fn case_02_simple_and_two_terms() {
    let docs = corpus();
    // cat AND dog → docs {0,3,9} = 3.
    let q = BoolQuery::and(vec![BoolQuery::term("cat"), BoolQuery::term("dog")]);
    assert_eq!(match_count(&q, &docs), 3);
}

#[test]
fn case_03_nested_or_and_or() {
    let docs = corpus();
    // (cat OR bird) AND (fish OR snake)
    //   cat OR bird  → {0,1,2,3,6,7,9}
    //   fish OR snake→ {0,2,3,4,5,6,7}
    //   intersection → {0,2,3,6,7} = 5.
    let q = BoolQuery::and(vec![
        BoolQuery::or(vec![BoolQuery::term("cat"), BoolQuery::term("bird")]),
        BoolQuery::or(vec![BoolQuery::term("fish"), BoolQuery::term("snake")]),
    ]);
    assert_eq!(match_count(&q, &docs), 5);
}

#[test]
fn case_04_nested_and_with_not() {
    let docs = corpus();
    // (cat OR bird) AND (fish OR snake) AND NOT dog
    //   from case_03 the base is {0,2,3,6,7}
    //   remove docs containing dog ({0,2,3,9}) → {6,7} = 2.
    let q = BoolQuery::and(vec![
        BoolQuery::or(vec![BoolQuery::term("cat"), BoolQuery::term("bird")]),
        BoolQuery::or(vec![BoolQuery::term("fish"), BoolQuery::term("snake")]),
        BoolQuery::not(BoolQuery::term("dog")),
    ]);
    assert_eq!(match_count(&q, &docs), 2);
}

#[test]
fn case_05_all_not_conjunction() {
    let docs = corpus();
    // NOT cat AND NOT dog → docs with neither cat nor dog.
    //   has cat or dog → {0,1,2,3,7,9}; complement over 10 docs → {4,5,6,8} = 4.
    let q = BoolQuery::and(vec![
        BoolQuery::not(BoolQuery::term("cat")),
        BoolQuery::not(BoolQuery::term("dog")),
    ]);
    assert_eq!(match_count(&q, &docs), 4);
}

#[test]
fn case_06_phrase_and_term() {
    let docs = corpus();
    // phrase[cat, dog] AND bird
    //   phrase requires both cat & dog present → {0,3,9}
    //   AND bird → only doc 3 has bird → {3} = 1.
    let q = BoolQuery::and(vec![
        BoolQuery::phrase(&["cat", "dog"]),
        BoolQuery::term("bird"),
    ]);
    assert_eq!(match_count(&q, &docs), 1);
}

#[test]
fn case_07_deeply_nested_three_levels() {
    let docs = corpus();
    // ((cat AND fish) OR (bird AND snake)) AND NOT frog
    //   cat AND fish  → {0,7}
    //   bird AND snake→ {3,6}
    //   union         → {0,3,6,7}
    //   NOT frog (frog ∈ {8,9}) removes nothing here → {0,3,6,7} = 4.
    let q = BoolQuery::and(vec![
        BoolQuery::or(vec![
            BoolQuery::and(vec![BoolQuery::term("cat"), BoolQuery::term("fish")]),
            BoolQuery::and(vec![BoolQuery::term("bird"), BoolQuery::term("snake")]),
        ]),
        BoolQuery::not(BoolQuery::term("frog")),
    ]);
    assert_eq!(match_count(&q, &docs), 4);
}

#[test]
fn case_08_or_of_nots() {
    let docs = corpus();
    // NOT cat OR NOT fish → everything except docs that have BOTH cat and fish.
    //   cat AND fish → {0,7}; complement over 10 → 8 docs.
    let q = BoolQuery::or(vec![
        BoolQuery::not(BoolQuery::term("cat")),
        BoolQuery::not(BoolQuery::term("fish")),
    ]);
    assert_eq!(match_count(&q, &docs), 8);
}

#[test]
fn case_09_empty_result_query() {
    let docs = corpus();
    // (cat AND dog) AND NOT cat → contradiction, matches nothing.
    let q = BoolQuery::and(vec![
        BoolQuery::and(vec![BoolQuery::term("cat"), BoolQuery::term("dog")]),
        BoolQuery::not(BoolQuery::term("cat")),
    ]);
    assert_eq!(match_count(&q, &docs), 0);
}

#[test]
fn case_10_empty_result_absent_term() {
    let docs = corpus();
    // A term that appears in no document.
    let q = BoolQuery::term("dragon");
    assert_eq!(match_count(&q, &docs), 0);
}

#[test]
fn case_11_wide_or_four_terms() {
    let docs = corpus();
    // lizard OR frog OR only OR dragon
    //   lizard → {4,8}; frog → {8,9}; only → {5}; dragon → {}.
    //   union → {4,5,8,9} = 4.
    let q = BoolQuery::or(vec![
        BoolQuery::term("lizard"),
        BoolQuery::term("frog"),
        BoolQuery::term("only"),
        BoolQuery::term("dragon"),
    ]);
    assert_eq!(match_count(&q, &docs), 4);
}

#[test]
fn case_12_double_negation_nested() {
    let docs = corpus();
    // NOT (NOT snake) ≡ snake → {3,4,6,7} = 4.
    let q = BoolQuery::not(BoolQuery::not(BoolQuery::term("snake")));
    assert_eq!(match_count(&q, &docs), 4);
}

#[test]
fn case_13_and_or_not_combined() {
    let docs = corpus();
    // (snake OR bird) AND NOT (cat OR dog)
    //   snake OR bird → {1,2,3,4,6,7}
    //   cat OR dog    → {0,1,2,3,7,9}
    //   NOT(cat OR dog) removes those → {4,6} = 2.
    let q = BoolQuery::and(vec![
        BoolQuery::or(vec![BoolQuery::term("snake"), BoolQuery::term("bird")]),
        BoolQuery::not(BoolQuery::or(vec![
            BoolQuery::term("cat"),
            BoolQuery::term("dog"),
        ])),
    ]);
    assert_eq!(match_count(&q, &docs), 2);
}
