//! Wave 15 integration tests for incremental search index and index warming.
//!
//! These tests use `TantivySearchIndex::new_ram()` so they require no
//! file-system I/O, no PostgreSQL connection, and no external services.

use oximedia_mam::search_index::{AssetSearchFields, TantivySearchIndex};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fields(title: &str, description: &str, tags: &str) -> AssetSearchFields {
    AssetSearchFields {
        title: title.to_string(),
        description: description.to_string(),
        tags: tags.to_string(),
        mime_type: "video/mp4".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Test 1: add → search (found) → delete → search (not found)
// ---------------------------------------------------------------------------

/// Verify that `add_document` makes a document findable and `delete_document`
/// removes it so it no longer appears in search results.
#[test]
fn test_incremental_index_add_search_delete() {
    let idx = TantivySearchIndex::new_ram().expect("RAM index should initialise");

    // Initially the index is empty.
    assert!(!idx.contains("id1").expect("contains check"));
    assert_eq!(idx.num_docs(), 0);

    // Add a document.
    idx.add_document(
        "id1",
        &fields("Breaking News", "top story of the day", "news uk"),
    )
    .expect("add_document");

    // The document should now be visible.
    assert!(idx.contains("id1").expect("contains after add"));

    // A full-text search on a term from the title should return the doc.
    let results = idx.search("Breaking", 10).expect("search after add");
    assert!(
        results.iter().any(|r| r == "id1"),
        "id1 should appear in search results after add; got: {results:?}"
    );

    // Delete the document.
    idx.delete_document("id1").expect("delete_document");

    // The document must no longer be in the index.
    assert!(
        !idx.contains("id1").expect("contains after delete"),
        "id1 should be absent after delete"
    );

    // A search should no longer return it.
    let results_after = idx.search("Breaking", 10).expect("search after delete");
    assert!(
        !results_after.iter().any(|r| r == "id1"),
        "id1 should not appear after delete; got: {results_after:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 2: add "foo" → update to "bar" → search for old / new term
// ---------------------------------------------------------------------------

/// Verify that `update_document` replaces the old content so that a search for
/// the old title no longer matches and a search for the new title does.
#[test]
fn test_incremental_index_update() {
    let idx = TantivySearchIndex::new_ram().expect("RAM index should initialise");

    // Add initial document with title "foo".
    idx.add_document("id1", &fields("foo document", "original content", "alpha"))
        .expect("initial add");

    // Confirm "foo" is findable.
    let initial = idx.search("foo", 10).expect("search for foo");
    assert!(
        initial.iter().any(|r| r == "id1"),
        "id1 should be found by 'foo' before update; got: {initial:?}"
    );

    // Update the document: new title "bar".
    idx.update_document("id1", &fields("bar document", "updated content", "beta"))
        .expect("update");

    // Search for old title "foo" — should NOT find id1.
    let after_foo = idx.search("foo", 10).expect("search for foo after update");
    assert!(
        !after_foo.iter().any(|r| r == "id1"),
        "id1 should NOT be found by 'foo' after update; got: {after_foo:?}"
    );

    // Search for new title "bar" — MUST find id1.
    let after_bar = idx.search("bar", 10).expect("search for bar after update");
    assert!(
        after_bar.iter().any(|r| r == "id1"),
        "id1 MUST be found by 'bar' after update; got: {after_bar:?}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: warm() makes reader current; immediate search returns without error
// ---------------------------------------------------------------------------

/// Verify that calling `warm()` after construction succeeds and that a
/// subsequent search completes without error (warming primed the reader).
#[test]
fn test_index_warming() {
    use std::time::Instant;

    let idx = TantivySearchIndex::new_ram().expect("RAM index should initialise");

    // Explicit warm — should not fail even on an empty index.
    idx.warm().expect("warm should succeed");

    // Pre-populate a document so there is something to search.
    idx.add_document(
        "warm-test-id",
        &fields("WarmTest document", "warming primes the reader", "cache"),
    )
    .expect("add document for warming test");

    // Warm again after the add.
    idx.warm().expect("second warm should succeed");

    // Search must complete quickly and without error.
    let t0 = Instant::now();
    let results = idx
        .search("WarmTest", 10)
        .expect("search after warm should not fail");
    let elapsed = t0.elapsed();

    assert!(
        results.iter().any(|r| r == "warm-test-id"),
        "warm-test-id should be returned after warming; got: {results:?}"
    );

    // The search should complete well within 1 second on any reasonable machine.
    assert!(
        elapsed.as_secs() < 1,
        "search after warm took too long: {elapsed:?}"
    );
}
