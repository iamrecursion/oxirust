//! Golden Information-Retrieval evaluation tests.
//!
//! Two layers:
//!
//! 1. **Hand-computed known answers** for the set-based metrics in
//!    [`oximedia_search::eval`] (`precision_at_k`, `recall_at_k`,
//!    `average_precision`, `mean_average_precision`). Every expected value is
//!    derived by hand in the comments and asserted to within `1e-9`.
//!
//! 2. **End-to-end relevance** over a 20-document in-test corpus run through the
//!    crate's *real* full-text search ([`oximedia_search::SearchEngine`], backed
//!    by Tantivy). This layer is gated behind the `search-engine` feature and
//!    only built when that feature is enabled (it is under `--all-features`).
//!    Thresholds are calibrated to what the real ranker actually returns on this
//!    fixed corpus, keeping the test deterministic.

use oximedia_search::eval::{
    average_precision, mean_average_precision, precision_at_k, recall_at_k,
};
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// Layer 1 — golden hand-computed metric values
// ─────────────────────────────────────────────────────────────────────────────

fn rel<'a>(ids: &[&'a str]) -> HashSet<&'a str> {
    ids.iter().copied().collect()
}

#[test]
fn golden_average_precision_hits_at_1_and_3() {
    // Length-5 ranking with relevant hits at ranks {1, 3}, |relevant| = 2.
    // AP = (1/1 + 2/3) / 2 = (1.0 + 0.6666…) / 2 = 0.83333…
    let ranked = ["d1", "d2", "d3", "d4", "d5"];
    let relevant = rel(&["d1", "d3"]);
    let expected = (1.0_f64 / 1.0 + 2.0 / 3.0) / 2.0;
    let ap = average_precision(&ranked, &relevant);
    assert!(
        (ap - expected).abs() < 1e-9,
        "AP expected {expected} got {ap}"
    );
    assert!(
        (ap - 0.833_333_333_333).abs() < 1e-9,
        "AP ≈ 0.8333 got {ap}"
    );
}

#[test]
fn golden_average_precision_hits_at_2_and_4() {
    // Relevant at ranks {2, 4}, |relevant| = 2.
    // AP = (1/2 + 2/4) / 2 = (0.5 + 0.5)/2 = 0.5
    let ranked = ["x", "r1", "y", "r2", "z"];
    let relevant = rel(&["r1", "r2"]);
    let ap = average_precision(&ranked, &relevant);
    assert!((ap - 0.5).abs() < 1e-9, "AP expected 0.5 got {ap}");
}

#[test]
fn golden_precision_at_k_known() {
    // top-4 of a 6-list; relevant ids hit at ranks 1 and 4 inside the cut-off.
    let ranked = ["a", "b", "c", "d", "e", "f"];
    let relevant = rel(&["a", "d", "z"]); // "z" not present
                                          // P@4 = 2 relevant in top-4 / 4 = 0.5
    assert!((precision_at_k(&ranked, &relevant, 4) - 0.5).abs() < 1e-9);
    // P@2 = 1 relevant ("a") in top-2 / 2 = 0.5
    assert!((precision_at_k(&ranked, &relevant, 2) - 0.5).abs() < 1e-9);
    // P@1 = "a" relevant / 1 = 1.0
    assert!((precision_at_k(&ranked, &relevant, 1) - 1.0).abs() < 1e-9);
}

#[test]
fn golden_recall_at_k_known() {
    let ranked = ["a", "b", "c", "d", "e", "f"];
    let relevant = rel(&["a", "d", "z"]); // 3 relevant total, "z" never retrieved
                                          // R@4 = 2 found / 3 = 0.6666…
    assert!((recall_at_k(&ranked, &relevant, 4) - 2.0 / 3.0).abs() < 1e-9);
    // R@6 (full list) = 2 found / 3 (z still missing) = 0.6666…
    assert!((recall_at_k(&ranked, &relevant, 6) - 2.0 / 3.0).abs() < 1e-9);
    // R@1 = 1 found / 3
    assert!((recall_at_k(&ranked, &relevant, 1) - 1.0 / 3.0).abs() < 1e-9);
}

#[test]
fn golden_map_three_queries() {
    // q1: relevant at {1}    → AP = 1/1 / 1 = 1.0
    // q2: relevant at {2}    → AP = 1/2 / 1 = 0.5
    // q3: relevant at {1,3}  → AP = (1/1 + 2/3)/2 = 0.83333…
    // MAP = (1.0 + 0.5 + 0.83333…)/3 = 0.777_77…
    let queries = [
        (vec!["a", "b", "c"], rel(&["a"])),
        (vec!["x", "b", "c"], rel(&["b"])),
        (vec!["a", "x", "c"], rel(&["a", "c"])),
    ];
    let ap3 = (1.0_f64 + 2.0 / 3.0) / 2.0;
    let expected = (1.0 + 0.5 + ap3) / 3.0;
    let map = mean_average_precision(&queries);
    assert!(
        (map - expected).abs() < 1e-9,
        "MAP expected {expected} got {map}"
    );
}

#[test]
fn golden_edge_cases_no_panic() {
    let ranked = ["a", "b"];
    let relevant = rel(&["a"]);
    // k=0
    assert_eq!(precision_at_k(&ranked, &relevant, 0), 0.0);
    assert_eq!(recall_at_k(&ranked, &relevant, 0), 0.0);
    // k > len clamps
    assert!((precision_at_k(&ranked, &relevant, 99) - 0.5).abs() < 1e-9);
    assert!((recall_at_k(&ranked, &relevant, 99) - 1.0).abs() < 1e-9);
    // empty relevant
    let empty: HashSet<&str> = HashSet::new();
    assert_eq!(precision_at_k(&ranked, &empty, 2), 0.0);
    assert_eq!(recall_at_k(&ranked, &empty, 2), 0.0);
    assert_eq!(average_precision(&ranked, &empty), 0.0);
    // empty query set
    let no_q: [(Vec<&str>, HashSet<&str>); 0] = [];
    assert_eq!(mean_average_precision(&no_q), 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Layer 2 — end-to-end relevance over the real text search engine
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "search-engine")]
mod end_to_end {
    use super::{average_precision, precision_at_k, recall_at_k};
    use oximedia_search::eval::mean_average_precision;
    use oximedia_search::index::builder::IndexDocument;
    use oximedia_search::{SearchEngine, SearchFilters, SearchQuery, SortOptions};
    use std::collections::HashSet;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use uuid::Uuid;

    /// Allocate a unique temp directory for an isolated engine instance.
    fn unique_index_dir(tag: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("oximedia-search-ireval-{tag}-{pid}-{n}"))
    }

    /// One corpus document: a deterministic UUID (derived from `idx`) plus the
    /// title/description text that the real ranker will tokenize and score.
    fn doc(idx: u8, title: &str, description: &str, keywords: &[&str]) -> IndexDocument {
        // Deterministic UUID so the relevant-set membership is stable across runs.
        let mut bytes = [0u8; 16];
        bytes[0] = idx;
        let asset_id = Uuid::from_bytes(bytes);
        IndexDocument {
            asset_id,
            file_path: format!("/media/{idx}.mp4"),
            title: Some(title.to_string()),
            description: Some(description.to_string()),
            keywords: keywords.iter().map(|s| (*s).to_string()).collect(),
            categories: vec![],
            mime_type: Some("video/mp4".to_string()),
            format: Some("mp4".to_string()),
            codec: Some("h264".to_string()),
            resolution: Some("1920x1080".to_string()),
            duration_ms: Some(60_000),
            file_size: Some(10_000_000),
            bitrate: Some(5_000_000),
            framerate: Some(30.0),
            created_at: 1_700_000_000,
            modified_at: 1_700_000_000,
            transcript: None,
            ocr_text: None,
            visual_features: None,
            audio_fingerprint: None,
            faces: None,
            dominant_colors: None,
            scene_tags: vec![],
            detected_objects: vec![],
            metadata: serde_json::json!({}),
        }
    }

    /// UUID for corpus document `idx` (mirrors [`doc`]).
    fn id(idx: u8) -> Uuid {
        let mut bytes = [0u8; 16];
        bytes[0] = idx;
        Uuid::from_bytes(bytes)
    }

    /// The 20-document corpus. Topics are deliberately separable so that the
    /// real ranker produces meaningful precision/recall.
    fn corpus() -> Vec<IndexDocument> {
        vec![
            // --- ocean / marine cluster (ids 1..=5) ---
            doc(
                1,
                "Ocean Depths",
                "deep blue ocean marine life documentary",
                &["ocean", "sea"],
            ),
            doc(
                2,
                "Coral Reef",
                "vibrant coral reef ocean marine fish",
                &["ocean", "coral"],
            ),
            doc(
                3,
                "Whale Song",
                "ocean whales marine mammals singing",
                &["ocean", "whale"],
            ),
            doc(
                4,
                "Tide Pools",
                "shallow ocean tide pools marine creatures",
                &["ocean", "tide"],
            ),
            doc(
                5,
                "Submarine",
                "ocean submarine exploring deep marine trench",
                &["ocean", "submarine"],
            ),
            // --- mountain cluster (ids 6..=10) ---
            doc(
                6,
                "Alpine Peak",
                "snowy mountain alpine peak climbing",
                &["mountain", "alpine"],
            ),
            doc(
                7,
                "Rocky Trail",
                "mountain rocky hiking trail summit",
                &["mountain", "hiking"],
            ),
            doc(
                8,
                "Glacier View",
                "mountain glacier ice frozen alpine",
                &["mountain", "glacier"],
            ),
            doc(
                9,
                "Volcano",
                "mountain volcano eruption lava peak",
                &["mountain", "volcano"],
            ),
            doc(
                10,
                "Ridge Walk",
                "mountain ridge walking high altitude",
                &["mountain", "ridge"],
            ),
            // --- city / urban cluster (ids 11..=15) ---
            doc(
                11,
                "Night City",
                "urban city skyline night lights",
                &["city", "urban"],
            ),
            doc(
                12,
                "Subway",
                "city subway underground transit urban",
                &["city", "subway"],
            ),
            doc(
                13,
                "Skyscraper",
                "city skyscraper architecture urban tower",
                &["city", "tower"],
            ),
            doc(
                14,
                "Street Market",
                "city street market urban vendors crowd",
                &["city", "market"],
            ),
            doc(
                15,
                "Bridge",
                "city bridge river urban crossing span",
                &["city", "bridge"],
            ),
            // --- wildlife / forest cluster (ids 16..=20) ---
            doc(
                16,
                "Jungle Cats",
                "forest jungle wildlife tiger predator",
                &["forest", "wildlife"],
            ),
            doc(
                17,
                "Bird Watching",
                "forest birds wildlife feathers nesting",
                &["forest", "birds"],
            ),
            doc(
                18,
                "Deer Herd",
                "forest deer wildlife grazing meadow",
                &["forest", "deer"],
            ),
            doc(
                19,
                "Wolf Pack",
                "forest wolves wildlife hunting pack",
                &["forest", "wolf"],
            ),
            doc(
                20,
                "Rainforest",
                "tropical forest rainforest wildlife canopy",
                &["forest", "rain"],
            ),
        ]
    }

    /// Build + commit an engine populated with the whole corpus.
    fn build_engine(tag: &str) -> SearchEngine {
        let mut engine = SearchEngine::new(&unique_index_dir(tag)).expect("create search engine");
        let docs = corpus();
        engine
            .index_documents_batch(&docs)
            .expect("index corpus batch");
        engine
    }

    fn text_query(text: &str, limit: usize) -> SearchQuery {
        SearchQuery {
            text: Some(text.to_string()),
            visual: None,
            audio: None,
            filters: SearchFilters::default(),
            limit,
            offset: 0,
            sort: SortOptions::default(),
        }
    }

    /// Run the real engine for `query_text` and return the ranked asset-id list.
    fn ranked_ids(engine: &SearchEngine, query_text: &str, limit: usize) -> Vec<Uuid> {
        let results = engine
            .search(&text_query(query_text, limit))
            .expect("search ok");
        results.results.iter().map(|r| r.asset_id).collect()
    }

    /// The five evaluation queries paired with their ground-truth relevant sets
    /// (the cluster a topic word belongs to).
    fn queries() -> Vec<(&'static str, HashSet<Uuid>)> {
        vec![
            ("ocean marine", (1..=5).map(id).collect()),
            ("mountain alpine", (6..=10).map(id).collect()),
            ("city urban", (11..=15).map(id).collect()),
            ("forest wildlife", (16..=20).map(id).collect()),
            ("ocean OR mountain", (1..=10).map(id).collect()),
        ]
    }

    #[test]
    fn corpus_is_fully_indexed() {
        let engine = build_engine("count");
        // Every doc shares no single universal token, so probe per cluster and
        // confirm the engine indexed all 20 documents by unioning hits.
        let total: usize = ["ocean", "mountain", "city", "forest"]
            .iter()
            .map(|t| engine.search(&text_query(t, 100)).expect("search ok").total)
            .sum();
        assert_eq!(total, 20, "all 20 docs should be reachable across clusters");
    }

    #[test]
    fn per_query_precision_recall_meet_thresholds() {
        let engine = build_engine("pr");
        for (qtext, relevant) in queries() {
            let ranked = ranked_ids(&engine, qtext, 50);

            // The cluster queries have 5 relevant docs; the union query has 10.
            // The real ranker places the topical cluster at the very top, so
            // precision@k (k = |relevant|) and recall@20 should be perfect on
            // this cleanly-separated corpus.
            let k = relevant.len();
            let p_at_k = precision_at_k(&ranked, &relevant, k);
            let r_at_20 = recall_at_k(&ranked, &relevant, 20);
            let ap = average_precision(&ranked, &relevant);

            assert!(
                p_at_k >= 0.99,
                "query {qtext:?}: P@{k} = {p_at_k} (< 0.99); ranked len {}",
                ranked.len()
            );
            assert!(
                r_at_20 >= 0.99,
                "query {qtext:?}: R@20 = {r_at_20} (< 0.99)"
            );
            assert!(ap >= 0.99, "query {qtext:?}: AP = {ap} (< 0.99)");
        }
    }

    #[test]
    fn precision_at_10_threshold() {
        let engine = build_engine("p10");

        // Calibration note: the real Tantivy ranker only returns documents that
        // *match* the query. On this cleanly-separated corpus the topic words
        // ("ocean"/"marine") appear in exactly the 5 cluster docs and nowhere
        // else, so the cluster query's ranked list has length 5 — never 10. Thus
        // P@10 clamps its cut-off to min(10, 5) = 5, and with all 5 hits
        // relevant the value is exactly 1.0 (NOT 5/10 = 0.5, which would only
        // arise if the engine padded the window with non-matching docs).
        let cluster_ranked = ranked_ids(&engine, "ocean marine", 50);
        assert_eq!(
            cluster_ranked.len(),
            5,
            "cluster query should return exactly its 5 matching docs"
        );
        let cluster_p10 = precision_at_k(&cluster_ranked, &queries()[0].1, 10);
        assert!(
            (cluster_p10 - 1.0).abs() < 1e-9,
            "cluster P@10 should be 1.0 (cut-off clamps to 5 matches) got {cluster_p10}"
        );

        // The union query "ocean OR mountain" matches exactly its 10 relevant
        // docs, so P@10 = 10/10 = 1.0.
        let union_q = &queries()[4];
        let union_ranked = ranked_ids(&engine, union_q.0, 50);
        assert_eq!(
            union_ranked.len(),
            10,
            "union query should return exactly its 10 matching docs"
        );
        let union_p10 = precision_at_k(&union_ranked, &union_q.1, 10);
        assert!(
            (union_p10 - 1.0).abs() < 1e-9,
            "union P@10 should be exactly 1.0 got {union_p10}"
        );
    }

    #[test]
    fn map_over_query_set_meets_threshold() {
        let engine = build_engine("map");
        // Build (ranked, relevant) pairs and compute MAP via the real ranker.
        let pairs: Vec<(Vec<Uuid>, HashSet<Uuid>)> = queries()
            .into_iter()
            .map(|(qtext, relevant)| (ranked_ids(&engine, qtext, 50), relevant))
            .collect();

        let map = mean_average_precision(&pairs);
        // The corpus is cleanly separated; the real ranker should retrieve each
        // cluster contiguously at the top → MAP ≈ 1.0. Use a conservative 0.95
        // lower bound so minor scoring ties never make this flaky.
        assert!(map >= 0.95, "MAP over 5 queries = {map} (< 0.95)");
    }
}
