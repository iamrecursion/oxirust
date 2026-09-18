//! Smoke tests for newly-wired orphan modules in oximedia-dedup.

#[test]
fn audio_fingerprint_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::audio_fingerprint));
}

#[test]
fn bloom_prescreen_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::bloom_prescreen));
}

#[test]
fn chromagram_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::chromagram));
}

#[test]
fn dedup_queue_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::dedup_queue));
}

#[test]
fn dedup_report_detailed_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::dedup_report_detailed));
}

#[test]
fn exact_match_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::exact_match));
}

#[test]
fn hierarchical_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::hierarchical));
}

#[test]
fn minhash_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::minhash));
}

#[test]
fn near_duplicate_cluster_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::near_duplicate_cluster));
}

#[test]
fn network_dedup_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::network_dedup));
}

#[test]
fn parallel_indexer_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::parallel_indexer));
}

#[test]
fn persistent_cache_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::persistent_cache));
}

#[test]
fn signature_store_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::signature_store));
}

#[test]
fn space_savings_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::space_savings));
}

#[test]
fn stream_dedup_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::stream_dedup));
}

#[test]
fn video_dedup_pipeline_module_accessible() {
    let _ = std::hint::black_box(stringify!(oximedia_dedup::video_dedup_pipeline));
}
