//! Full-workflow SQLite integration test for `oximedia-dedup`.
//!
//! Exercises `DuplicateDetector` with a real on-disk SQLite database:
//! creates 5 files (3 duplicates + 2 unique), indexes them, runs
//! `DetectionStrategy::ExactHash`, and verifies the grouping.

#[cfg(feature = "sqlite")]
mod sqlite_tests {
    use oximedia_dedup::{DedupConfig, DetectionStrategy, DuplicateDetector};
    use std::path::PathBuf;

    fn unique_db_path(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(1);
        std::env::temp_dir().join(format!("dedup_sqlite_{tag}_{nanos}.db"))
    }

    #[tokio::test]
    async fn test_full_workflow_exact_hash() {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(99);

        // Three files with identical content
        let identical = b"oximedia-dedup-sqlite-integration-test-identical-content";
        let dup1 = dir.join(format!("sqlit_dup1_{nanos}.bin"));
        let dup2 = dir.join(format!("sqlit_dup2_{nanos}.bin"));
        let dup3 = dir.join(format!("sqlit_dup3_{nanos}.bin"));
        // Two unique files
        let uniq1 = dir.join(format!("sqlit_uniq1_{nanos}.bin"));
        let uniq2 = dir.join(format!("sqlit_uniq2_{nanos}.bin"));

        std::fs::write(&dup1, identical).expect("write dup1");
        std::fs::write(&dup2, identical).expect("write dup2");
        std::fs::write(&dup3, identical).expect("write dup3");
        std::fs::write(&uniq1, b"unique file one content A").expect("write uniq1");
        std::fs::write(&uniq2, b"unique file two content B").expect("write uniq2");

        let db_path = unique_db_path("workflow");
        let config = DedupConfig {
            database_path: db_path.clone(),
            ..Default::default()
        };

        let mut detector = DuplicateDetector::new(config)
            .await
            .expect("create detector");

        // Index all five files
        let errors = detector
            .add_files(&[&dup1, &dup2, &dup3, &uniq1, &uniq2])
            .await
            .expect("add_files");
        assert!(errors.is_empty(), "add_files reported errors: {errors:?}");

        // Run exact-hash detection
        let report = detector
            .find_duplicates(DetectionStrategy::ExactHash)
            .await
            .expect("find_duplicates");

        // Must have at least one group with ≥ 3 members
        let large_group = report.groups.iter().any(|g| g.files.len() >= 3);

        assert!(
            large_group,
            "expected at least one group with ≥ 3 duplicate files; \
             found {} groups: {:?}",
            report.groups.len(),
            report.groups.iter().map(|g| &g.files).collect::<Vec<_>>()
        );

        // Cleanup
        for p in &[&dup1, &dup2, &dup3, &uniq1, &uniq2] {
            std::fs::remove_file(p).ok();
        }
        std::fs::remove_file(&db_path).ok();
    }

    #[tokio::test]
    async fn test_batch_insert_via_database_api() {
        use oximedia_dedup::database::{BatchFileEntry, DedupDatabase};

        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(77);

        let db_path = unique_db_path("batch");
        let db = DedupDatabase::open(&db_path).await.expect("open database");

        // Create 30 temp files and batch-insert them
        let mut created = Vec::with_capacity(30);
        let mut entries = Vec::with_capacity(30);
        for i in 0..30usize {
            let p = dir.join(format!("sqlit_batch_{nanos}_{i}.tmp"));
            std::fs::write(&p, format!("batch_content_{i}")).expect("write file");
            entries.push(BatchFileEntry {
                path: p.to_string_lossy().to_string(),
                hash: format!("batchhash_{i:04x}"),
            });
            created.push(p);
        }

        let count = db.insert_batch(&entries).await.expect("insert_batch");
        assert_eq!(count, 30, "insert_batch must return 30");

        let total = db.count_files().await.expect("count_files");
        assert_eq!(total, 30, "database must contain exactly 30 files");

        // Cleanup
        db.close().await.ok();
        for p in &created {
            std::fs::remove_file(p).ok();
        }
        std::fs::remove_file(&db_path).ok();
    }

    #[tokio::test]
    async fn test_incremental_add_and_query() {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(55);

        let db_path = unique_db_path("incremental");
        let config = DedupConfig {
            database_path: db_path.clone(),
            ..Default::default()
        };

        let mut detector = DuplicateDetector::new(config).await.expect("detector");

        // First wave: 4 files (2 pairs of duplicates)
        let pair_a1 = dir.join(format!("incr_a1_{nanos}.bin"));
        let pair_a2 = dir.join(format!("incr_a2_{nanos}.bin"));
        let pair_b1 = dir.join(format!("incr_b1_{nanos}.bin"));
        let pair_b2 = dir.join(format!("incr_b2_{nanos}.bin"));

        std::fs::write(&pair_a1, b"pair A content").expect("write a1");
        std::fs::write(&pair_a2, b"pair A content").expect("write a2");
        std::fs::write(&pair_b1, b"pair B content distinct").expect("write b1");
        std::fs::write(&pair_b2, b"pair B content distinct").expect("write b2");

        detector.add_file(&pair_a1).await.expect("add a1");
        detector.add_file(&pair_a2).await.expect("add a2");
        detector.add_file(&pair_b1).await.expect("add b1");
        detector.add_file(&pair_b2).await.expect("add b2");

        let report = detector
            .find_duplicates(DetectionStrategy::ExactHash)
            .await
            .expect("find");

        // Must find at least 2 groups (pair A and pair B)
        assert!(
            report.groups.len() >= 2,
            "expected ≥ 2 duplicate groups from 2 pairs; found {}",
            report.groups.len()
        );

        // Cleanup
        for p in &[&pair_a1, &pair_a2, &pair_b1, &pair_b2] {
            std::fs::remove_file(p).ok();
        }
        std::fs::remove_file(&db_path).ok();
    }
}
