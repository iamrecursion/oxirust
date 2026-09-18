//! End-to-end deduplication integration test.
//!
//! Creates a temporary directory with 3 identical and 2 unique files,
//! indexes them through `DuplicateDetector`, and verifies that the three
//! identical files end up in the same duplicate group.

#![cfg(feature = "sqlite")]

use oximedia_dedup::{DedupConfig, DetectionStrategy, DuplicateDetector};
use std::path::PathBuf;

fn temp_db_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("dedup_e2e_{tag}_{nanos}.db"))
}

#[tokio::test]
async fn test_e2e_exact_hash_finds_identical_group() {
    let dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(42);

    // Three identical files (same content → same hash)
    let identical_content = b"the same media content for deduplication testing";
    let dup_a = dir.join(format!("e2e_dup_a_{nanos}.bin"));
    let dup_b = dir.join(format!("e2e_dup_b_{nanos}.bin"));
    let dup_c = dir.join(format!("e2e_dup_c_{nanos}.bin"));
    // Two unique files with distinct content
    let unique_x = dir.join(format!("e2e_uniq_x_{nanos}.bin"));
    let unique_y = dir.join(format!("e2e_uniq_y_{nanos}.bin"));

    std::fs::write(&dup_a, identical_content).expect("write dup_a");
    std::fs::write(&dup_b, identical_content).expect("write dup_b");
    std::fs::write(&dup_c, identical_content).expect("write dup_c");
    std::fs::write(&unique_x, b"unique content X - nothing else matches this")
        .expect("write unique_x");
    std::fs::write(
        &unique_y,
        b"unique content Y - also distinctive and different",
    )
    .expect("write unique_y");

    let db_path = temp_db_path("e2e");
    let config = DedupConfig {
        database_path: db_path.clone(),
        ..Default::default()
    };

    let mut detector = DuplicateDetector::new(config).await.expect("new detector");

    for p in &[&dup_a, &dup_b, &dup_c, &unique_x, &unique_y] {
        detector.add_file(p).await.expect("add_file");
    }

    let report = detector
        .find_duplicates(DetectionStrategy::ExactHash)
        .await
        .expect("find_duplicates");

    // Collect paths as strings for comparison
    let dup_a_str = dup_a.to_string_lossy().to_string();
    let dup_b_str = dup_b.to_string_lossy().to_string();
    let dup_c_str = dup_c.to_string_lossy().to_string();

    // At least one group must contain all three identical files
    let has_triple = report.groups.iter().any(|group| {
        group.files.contains(&dup_a_str)
            && group.files.contains(&dup_b_str)
            && group.files.contains(&dup_c_str)
    });

    assert!(
        has_triple,
        "expected the three identical files to be grouped together; \
         found {} groups: {:?}",
        report.groups.len(),
        report.groups.iter().map(|g| &g.files).collect::<Vec<_>>()
    );

    // Cleanup
    for p in &[&dup_a, &dup_b, &dup_c, &unique_x, &unique_y] {
        std::fs::remove_file(p).ok();
    }
    std::fs::remove_file(&db_path).ok();
}
