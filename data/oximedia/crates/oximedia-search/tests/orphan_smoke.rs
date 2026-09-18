//! Smoke tests for newly-wired orphan modules in oximedia-search.

// ─── batch_index ─────────────────────────────────────────────────────────────
#[test]
fn test_batch_index_flush() {
    use oximedia_search::batch_index::{BatchDocument, BatchIndexer, InMemoryBackend};

    let backend = InMemoryBackend::new();
    let mut indexer = BatchIndexer::with_capacity(backend, 3);

    for i in 0..7u32 {
        let doc = BatchDocument::new(format!("doc-{i}"), format!("content about item {i}"));
        indexer.push(doc).expect("push should succeed");
    }
    indexer.flush().expect("flush should succeed");
    assert_eq!(
        indexer.buffered_count(),
        0,
        "all docs should be committed after flush"
    );
}

// ─── facet_multi_value ───────────────────────────────────────────────────────
#[test]
fn test_facet_multi_value_distinct_docs() {
    use oximedia_search::facet_multi_value::{FacetField, MultiValueFacetIndex};

    let mut index = MultiValueFacetIndex::new();
    index.add_document("doc-1", FacetField::Tags, &["sport", "outdoor"]);
    index.add_document("doc-2", FacetField::Tags, &["sport", "music"]);
    index.add_document("doc-3", FacetField::Tags, &["music"]);

    let counts = index.counts(FacetField::Tags);
    // counts() returns Vec<MultiFacetCount>; find by value
    let sport_count = counts
        .iter()
        .find(|c| c.value == "sport")
        .map(|c| c.count)
        .unwrap_or(0);
    assert_eq!(sport_count, 2, "sport should appear in 2 distinct docs");
    let music_count = counts
        .iter()
        .find(|c| c.value == "music")
        .map(|c| c.count)
        .unwrap_or(0);
    assert_eq!(music_count, 2, "music should appear in 2 distinct docs");
}

// ─── ir_evaluation ───────────────────────────────────────────────────────────
#[test]
fn test_ir_evaluation_precision_recall() {
    use oximedia_search::ir_evaluation::{evaluate_query, RelevanceJudgements};

    let mut qrels = RelevanceJudgements::new();
    qrels.add("q1", "doc-a", 2);
    qrels.add("q1", "doc-b", 1);
    qrels.add("q1", "doc-c", 0);

    let ranked = vec!["doc-b", "doc-a", "doc-c"];
    let metrics = evaluate_query(&qrels, "q1", &ranked, 3);
    assert!(metrics.precision_at_k > 0.0, "precision should be positive");
    assert!(metrics.recall_at_k > 0.0, "recall should be positive");
}

// ─── metrics ─────────────────────────────────────────────────────────────────
#[test]
fn test_metrics_precision_at_k() {
    use oximedia_search::metrics::compute_precision_at_k;
    use std::collections::HashSet;

    let retrieved = vec![1usize, 3, 5, 7, 9];
    let relevant: HashSet<usize> = [1, 3, 7].iter().copied().collect();
    let p3 = compute_precision_at_k(&retrieved, &relevant, 3);
    assert!((p3 - 2.0 / 3.0).abs() < 1e-5, "precision@3 should be 2/3");
}

#[test]
fn test_metrics_recall_at_k() {
    use oximedia_search::metrics::compute_recall_at_k;
    use std::collections::HashSet;

    let retrieved = vec![1usize, 3, 5, 7, 9];
    let relevant: HashSet<usize> = [1, 3, 7].iter().copied().collect();
    let r5 = compute_recall_at_k(&retrieved, &relevant, 5);
    assert!((r5 - 1.0).abs() < 1e-5, "recall@5 should be 1.0");
}

// ─── scene_search ────────────────────────────────────────────────────────────
#[test]
fn test_scene_search_basic() {
    use oximedia_search::scene_search::{SceneSearchFilter, SceneSearchIndex};

    let mut index = SceneSearchIndex::new();
    index.add_scene(
        "doc1".to_string(),
        vec!["car".to_string(), "road".to_string()],
    );
    index.add_scene(
        "doc2".to_string(),
        vec!["cat".to_string(), "sofa".to_string()],
    );

    let filter = SceneSearchFilter {
        detected_objects: vec!["car".to_string()],
        scene_types: vec![],
        min_confidence: 0.0,
    };
    let results = index.search(&filter);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].document_id, "doc1");
}

// ─── scene_search_integration ────────────────────────────────────────────────
#[test]
fn test_scene_search_integration_bridge() {
    use oximedia_search::scene_search::SceneSearchFilter;
    use oximedia_search::scene_search_integration::SceneSearchBridge;

    let mut bridge = SceneSearchBridge::new();
    bridge.add_scene(
        "doc1".to_string(),
        vec!["car".to_string(), "outdoor".to_string()],
    );

    let filter = SceneSearchFilter {
        detected_objects: vec!["car".to_string()],
        scene_types: vec![],
        min_confidence: 0.0,
    };
    let items = bridge.search(&filter);
    assert!(!items.is_empty(), "bridge should return at least one item");
}

// ─── search_history ──────────────────────────────────────────────────────────
#[test]
fn test_search_history_record_and_retrieve() {
    use oximedia_search::search_history::{HistoryEntry, SearchHistory};

    let mut history = SearchHistory::new();
    history.record(HistoryEntry {
        query: "nature documentary".to_string(),
        timestamp: 1000,
        result_count: 42,
        user_id: 1,
    });
    history.record(HistoryEntry {
        query: "underwater footage".to_string(),
        timestamp: 1010,
        result_count: 7,
        user_id: 1,
    });

    let recent = history.recent(1, 5);
    assert_eq!(recent.len(), 2);
}

// ─── search_shard ────────────────────────────────────────────────────────────
#[test]
fn test_search_shard_basic_insert_query() {
    use oximedia_search::search_shard::ShardedIndex;
    use uuid::Uuid;

    let mut idx = ShardedIndex::new(4, 100).expect("create sharded index");
    let id1 = Uuid::nil();
    let id2 = Uuid::from_u128(1);
    let id3 = Uuid::from_u128(2);
    idx.add_document(id1, "alpha beta gamma");
    idx.add_document(id2, "delta epsilon");
    idx.add_document(id3, "alpha epsilon");

    let results = idx.search("alpha");
    let ids: Vec<Uuid> = results.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&id1), "should find doc1 with 'alpha'");
    assert!(ids.contains(&id3), "should find doc3 with 'alpha'");
}

// ─── search_snapshot ─────────────────────────────────────────────────────────
#[test]
fn test_search_snapshot_create_and_compare() {
    use oximedia_search::search_snapshot::SnapshotStore;

    let mut store = SnapshotStore::new();
    let id1 = store.take_snapshot(1000, 100, vec![]);
    let id2 = store.take_snapshot(2000, 110, vec![]);

    assert_eq!(store.len(), 2);
    let latest = store.latest().expect("should have a latest snapshot");
    assert_eq!(latest.id, id2);
    assert!(id1 < id2, "ids should be monotonically increasing");
}

// ─── search_throttle ─────────────────────────────────────────────────────────
#[test]
fn test_search_throttle_allow_under_budget() {
    use oximedia_search::search_throttle::{SearchThrottle, ThrottleConfig, ThrottleDecision};

    let cfg = ThrottleConfig {
        capacity: 10,
        refill_rate: 0.001, // negligible refill so tokens drain
        initial_tokens: 10,
    };
    let mut throttle = SearchThrottle::new(cfg);
    // First 10 requests at t=0 should be allowed
    for i in 0..10u64 {
        assert_eq!(
            throttle.try_acquire(42, 0.0),
            ThrottleDecision::Allowed,
            "request {} should be allowed within budget",
            i
        );
    }
    // 11th request should be denied (budget exhausted)
    assert_eq!(
        throttle.try_acquire(42, 0.0),
        ThrottleDecision::Denied,
        "should be denied when budget exhausted"
    );
}

// ─── suggest ─────────────────────────────────────────────────────────────────
#[test]
fn test_suggest_prefix_matching() {
    use oximedia_search::suggest::QuerySuggester;

    let index = vec![
        "audio".to_string(),
        "audio mixing".to_string(),
        "audition".to_string(),
        "video".to_string(),
    ];
    let suggester = QuerySuggester::new(&index);
    let suggestions = suggester.suggest("audi");
    assert!(
        !suggestions.is_empty(),
        "should return suggestions for 'audi'"
    );
    for s in &suggestions {
        assert!(
            s.to_lowercase().starts_with("audi"),
            "suggestion '{s}' must start with 'audi'"
        );
    }
}

// ─── transcript_search ───────────────────────────────────────────────────────
#[test]
fn test_transcript_search_index_and_query() {
    use oximedia_search::transcript_search::{TranscriptIndex, TranscriptSegment};
    use uuid::Uuid;

    let asset_id = Uuid::nil();
    let mut index = TranscriptIndex::new();
    index.index_asset(
        asset_id,
        vec![
            TranscriptSegment::new(0, 5000, "the quick brown fox"),
            TranscriptSegment::new(5000, 10000, "jumped over the lazy dog"),
        ],
    );

    let results = index.search("fox");
    assert!(!results.is_empty(), "should find segment containing 'fox'");
    assert_eq!(results[0].segment.start_ms, 0);
    assert_eq!(results[0].segment.end_ms, 5000);
}
