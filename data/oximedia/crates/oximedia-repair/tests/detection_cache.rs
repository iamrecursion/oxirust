//! Integration tests for the mtime-keyed detection cache in `RepairEngine`.
//!
//! We verify that:
//! 1. Calling `analyze()` twice on the same file returns the same result.
//! 2. After `invalidate_cache()` a second scan still returns the same result.
//! 3. `clear_cache()` does not cause subsequent `analyze()` calls to error.
//! 4. No panics occur under normal usage.

use oximedia_repair::RepairEngine;
use std::io::Write;
use std::path::PathBuf;

fn temp_file(name: &str, data: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "oximedia_cache_test_{}_{}",
        std::process::id(),
        name
    ));
    let mut f = std::fs::File::create(&path).expect("create temp file");
    f.write_all(data).expect("write temp file");
    path
}

/// Synthetic file data: some non-zero entropy to trigger analyze()
fn synthetic_data() -> Vec<u8> {
    let mut data = Vec::with_capacity(4096);
    for i in 0u8..=255 {
        data.push(i);
        data.push(i.wrapping_mul(3));
    }
    // Pad to 4096
    data.resize(4096, 0x42);
    data
}

// ---------------------------------------------------------------------------
// Test 1: same path → same result (cache hit)
// ---------------------------------------------------------------------------

#[test]
fn test_analyze_twice_returns_same_result() {
    let engine = RepairEngine::new();
    let path = temp_file("cache_double.bin", &synthetic_data());

    let first = engine.analyze(&path).expect("first analyze failed");
    let second = engine.analyze(&path).expect("second analyze failed");

    assert_eq!(
        first.len(),
        second.len(),
        "analyze() twice should return the same number of issues"
    );

    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!(
            format!("{:?}", a.issue_type),
            format!("{:?}", b.issue_type),
            "issue types should match between cached and fresh calls"
        );
    }

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test 2: invalidate_cache → subsequent call still succeeds
// ---------------------------------------------------------------------------

#[test]
fn test_analyze_after_invalidate_succeeds() {
    let engine = RepairEngine::new();
    let path = temp_file("cache_invalidate.bin", &synthetic_data());

    let first = engine.analyze(&path).expect("first analyze failed");
    engine.invalidate_cache(&path);
    let second = engine
        .analyze(&path)
        .expect("second analyze (post-invalidate) failed");

    // Both should return the same set of issue types
    let mut first_types: Vec<String> = first
        .iter()
        .map(|i| format!("{:?}", i.issue_type))
        .collect();
    let mut second_types: Vec<String> = second
        .iter()
        .map(|i| format!("{:?}", i.issue_type))
        .collect();
    first_types.sort();
    second_types.sort();
    first_types.dedup();
    second_types.dedup();

    assert_eq!(
        first_types, second_types,
        "issue types should be consistent before and after invalidate_cache()"
    );

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test 3: clear_cache → subsequent call still succeeds
// ---------------------------------------------------------------------------

#[test]
fn test_analyze_after_clear_cache_succeeds() {
    let engine = RepairEngine::new();
    let path = temp_file("cache_clear.bin", &synthetic_data());

    let _ = engine.analyze(&path).expect("initial analyze");
    engine.clear_cache();
    let after_clear = engine
        .analyze(&path)
        .expect("analyze after clear_cache failed");

    // Merely verify it returns successfully — no panic
    let _ = after_clear;

    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Test 4: analyze on multiple distinct files — no cross-contamination
// ---------------------------------------------------------------------------

#[test]
fn test_cache_no_cross_contamination() {
    let engine = RepairEngine::new();

    let path_a = temp_file("cache_multi_a.bin", &synthetic_data());
    let mut data_b = synthetic_data();
    // Make data_b different by flipping some bytes
    for b in data_b.iter_mut().take(16) {
        *b = b.wrapping_add(1);
    }
    let path_b = temp_file("cache_multi_b.bin", &data_b);

    let issues_a1 = engine.analyze(&path_a).expect("a1");
    let issues_b1 = engine.analyze(&path_b).expect("b1");

    // Second call should still hit cache for each
    let issues_a2 = engine.analyze(&path_a).expect("a2");
    let issues_b2 = engine.analyze(&path_b).expect("b2");

    assert_eq!(
        issues_a1.len(),
        issues_a2.len(),
        "a: cache should be stable"
    );
    assert_eq!(
        issues_b1.len(),
        issues_b2.len(),
        "b: cache should be stable"
    );

    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);
}

// ---------------------------------------------------------------------------
// Test 5: non-existent file → analyze returns Err, does not panic
// ---------------------------------------------------------------------------

#[test]
fn test_analyze_nonexistent_file_returns_err() {
    let engine = RepairEngine::new();
    let path = std::env::temp_dir().join("oximedia_cache_this_file_does_not_exist_1234.bin");

    let result = engine.analyze(&path);
    // We expect an Err — importantly, no panic
    assert!(result.is_err(), "analyze on missing file should return Err");
}
