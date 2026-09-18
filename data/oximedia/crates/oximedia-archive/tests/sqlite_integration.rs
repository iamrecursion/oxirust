//! End-to-end integration tests for the Pure-Rust SQLite (`oxisql-sqlite-compat`)
//! backend introduced when migrating off `sqlx`/`libsqlite3-sys`.
//!
//! These tests exercise the actual public API against a real (file-backed and
//! in-memory-equivalent) database, since the migration replaced a mature C
//! SQLite engine with a young Pure-Rust one (`oxisqlite` 0.3.x) that has real
//! functional gaps (e.g. `RETURNING` is parsed but not executed; `MAX`/`MIN`
//! over a column whose first-scanned value is `NULL` used to panic). Plain
//! `cargo check`/`clippy` cannot catch these — only exercising the runtime
//! query paths can.
#![cfg(feature = "sqlite")]

use oximedia_archive::checksum::ChecksumRecord;
use oximedia_archive::quarantine::QuarantineRecord;
use oximedia_archive::report::{self, ReportFormat};
use oximedia_archive::{ArchiveVerifier, ChecksumSet, VerificationConfig};
use std::path::PathBuf;

fn temp_db_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("oximedia_archive_sqlite_it_{tag}_{nanos}.db"))
}

/// Round-trips a [`ChecksumRecord`] through `save`/`load`, including a `NULL`
/// `last_verified_at` — the column whose `MAX()` used to crash the engine
/// when it was the first row scanned.
#[tokio::test]
async fn checksum_record_save_and_load_roundtrip() {
    let db_path = temp_db_path("checksum");
    let mut verifier = ArchiveVerifier::new();
    verifier.config_mut().database_path = db_path.clone();
    verifier.initialize().await.expect("initialize");
    let pool = verifier.db_pool().expect("pool present after initialize");

    let checksums = ChecksumSet {
        blake3: Some("blakehash".to_string()),
        md5: None,
        sha256: Some("shahash".to_string()),
        crc32: Some("deadbeef".to_string()),
    };
    let record = ChecksumRecord::new(&PathBuf::from("a.mp4"), 1000, checksums);
    let id = record.save(pool).await.expect("save checksum");
    assert!(id > 0, "insert should yield a positive rowid");

    let loaded = ChecksumRecord::load(pool, "a.mp4")
        .await
        .expect("load should not error")
        .expect("record should exist");
    assert_eq!(loaded.blake3.as_deref(), Some("blakehash"));
    assert_eq!(loaded.md5, None);
    assert!(loaded.last_verified_at.is_none());

    std::fs::remove_file(&db_path).ok();
}

/// Round-trips a [`QuarantineRecord`] through `save`/`load`/`mark_restored`,
/// verifying the `BOOLEAN` columns (`auto_quarantine`, `restored`) — which
/// SQLite (and this engine) store as `INTEGER 0/1`, not a native bool.
#[tokio::test]
async fn quarantine_record_save_load_and_restore_roundtrip() {
    let db_path = temp_db_path("quarantine");
    let mut verifier = ArchiveVerifier::new();
    verifier.config_mut().database_path = db_path.clone();
    verifier.initialize().await.expect("initialize");
    let pool = verifier.db_pool().expect("pool present after initialize");

    let record = QuarantineRecord::new(
        PathBuf::from("orig.mp4"),
        PathBuf::from("quarantine/orig.mp4"),
        "test reason".to_string(),
        Some("checksum-before".to_string()),
        true,
    );
    let id = record.save(pool).await.expect("save quarantine record");
    assert!(id > 0);

    let mut loaded = QuarantineRecord::load(pool, id)
        .await
        .expect("load should not error")
        .expect("record should exist");
    assert!(loaded.auto_quarantine, "auto_quarantine bool round-trip");
    assert!(!loaded.restored, "restored should start false");

    loaded.mark_restored(pool).await.expect("mark restored");
    assert!(loaded.restored);

    let all = QuarantineRecord::load_all(pool)
        .await
        .expect("load_all should not error");
    assert_eq!(all.len(), 1);
    assert!(all[0].restored, "restored flag must persist after reload");

    std::fs::remove_file(&db_path).ok();
}

/// Exercises every aggregate query in `report::generate_report` against an
/// EMPTY database — the scenario most likely to hit engine edge cases
/// (`SUM`/`MAX`/`MIN` over zero or all-`NULL` rows).
#[tokio::test]
async fn generate_report_on_empty_database_does_not_panic() {
    let db_path = temp_db_path("report_empty");
    let mut verifier = ArchiveVerifier::new();
    verifier.config_mut().database_path = db_path.clone();
    verifier.initialize().await.expect("initialize");
    let pool = verifier.db_pool().expect("pool present after initialize");

    let out_path = std::env::temp_dir().join(format!(
        "oximedia_archive_report_empty_{}.json",
        std::process::id()
    ));
    report::generate_report(pool, ReportFormat::Json, &out_path)
        .await
        .expect("generate_report should succeed on an empty database");
    assert!(out_path.exists());

    std::fs::remove_file(&db_path).ok();
    std::fs::remove_file(&out_path).ok();
}

/// Exercises `generate_report` against a database containing checksum
/// records with a mix of `NULL` and non-`NULL` `last_verified_at` — the
/// exact shape that used to panic the Pure-Rust engine's `MAX()`
/// implementation when the first-scanned row's value was `NULL`.
#[tokio::test]
async fn generate_report_with_mixed_null_last_verified_at() {
    let db_path = temp_db_path("report_mixed");
    let mut verifier = ArchiveVerifier::new();
    verifier.config_mut().database_path = db_path.clone();
    verifier.initialize().await.expect("initialize");
    let pool = verifier.db_pool().expect("pool present after initialize");

    // First-inserted record has NULL last_verified_at (never verified).
    let unverified = ChecksumRecord::new(
        &PathBuf::from("unverified.mp4"),
        500,
        ChecksumSet {
            blake3: Some("h1".to_string()),
            ..Default::default()
        },
    );
    unverified.save(pool).await.expect("save unverified");

    // Second record has been verified (non-NULL last_verified_at).
    let mut verified = ChecksumRecord::new(
        &PathBuf::from("verified.mp4"),
        700,
        ChecksumSet {
            blake3: Some("h2".to_string()),
            sha256: Some("h2sha".to_string()),
            ..Default::default()
        },
    );
    verified.save(pool).await.expect("save verified");
    verified
        .update_verified(pool)
        .await
        .expect("update_verified");

    let out_path = std::env::temp_dir().join(format!(
        "oximedia_archive_report_mixed_{}.json",
        std::process::id()
    ));
    report::generate_report(pool, ReportFormat::Json, &out_path)
        .await
        .expect("generate_report must not panic on mixed-NULL last_verified_at");

    let json = std::fs::read_to_string(&out_path).expect("read report json");
    assert!(json.contains("\"total_checksums\": 2"));

    std::fs::remove_file(&db_path).ok();
    std::fs::remove_file(&out_path).ok();
}

/// Verifies `VerificationConfig::default()` builds and `initialize()`
/// creates all four tables (checksums, fixity_checks, premis_events,
/// quarantine_records) without error against a fresh file-backed database.
#[tokio::test]
async fn archive_verifier_initialize_creates_schema() {
    let db_path = temp_db_path("schema");
    let mut verifier = ArchiveVerifier::with_config(VerificationConfig {
        database_path: db_path.clone(),
        ..Default::default()
    });
    verifier.initialize().await.expect("initialize");
    // Re-initializing (idempotent CREATE TABLE IF NOT EXISTS) must also succeed.
    verifier.initialize().await.expect("re-initialize");

    std::fs::remove_file(&db_path).ok();
}
