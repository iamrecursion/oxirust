//! Tests for `FolderSync` — directory walk → fingerprint → upload → event.

use oximedia_mam::event_bus::MamEvent;
use oximedia_mam::folders::FolderSync;
use oximedia_mam::integration::MamEventBus;
use oximedia_mam::storage::MamStorage;
use std::env;
use std::sync::Arc;

// ── sync_once uploads files found in the watch directory ─────────────────────

#[tokio::test]
async fn test_sync_once_uploads_new_file() {
    // Set up a temporary watched directory
    let watch_dir = env::temp_dir().join(format!("folder_sync_test_{}", unique()));
    tokio::fs::create_dir_all(&watch_dir)
        .await
        .expect("create watch dir");

    // Write a test file into it
    let test_file = watch_dir.join("clip.mp4");
    tokio::fs::write(&test_file, b"fake video data for ingest test")
        .await
        .expect("write test file");

    // Set up storage in a separate temp dir
    let store_root = env::temp_dir().join(format!("folder_sync_store_{}", unique()));
    let storage = Arc::new(
        MamStorage::local(&store_root)
            .await
            .expect("create storage"),
    );

    let bus = MamEventBus::new(32);
    let mut rx = bus.subscribe();
    let tx = bus.sender();

    let sync = FolderSync::new(watch_dir.clone(), Arc::clone(&storage), tx);
    let count = sync.sync_once().await.expect("sync_once");

    // We should have ingested exactly 1 file
    assert_eq!(count, 1, "expected 1 file ingested, got {count}");

    // Verify we received an AssetIngested event
    let event = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
        .await
        .expect("event should arrive within timeout")
        .expect("no recv error");

    assert!(
        matches!(event, MamEvent::AssetIngested { .. }),
        "expected AssetIngested, got {event:?}"
    );
}

// ── sync_once is idempotent: second call returns 0 ───────────────────────────

#[tokio::test]
async fn test_sync_once_idempotent() {
    let watch_dir = env::temp_dir().join(format!("folder_sync_idem_{}", unique()));
    tokio::fs::create_dir_all(&watch_dir).await.expect("mkdir");

    let test_file = watch_dir.join("audio.flac");
    tokio::fs::write(&test_file, b"audio content here")
        .await
        .expect("write");

    let store_root = env::temp_dir().join(format!("folder_sync_store_idem_{}", unique()));
    let storage = Arc::new(MamStorage::local(&store_root).await.expect("storage"));

    let bus = MamEventBus::new(16);
    let tx = bus.sender();

    let sync = FolderSync::new(watch_dir.clone(), Arc::clone(&storage), tx.clone());

    // First run: uploads 1 file
    let first = sync.sync_once().await.expect("first sync");
    assert_eq!(first, 1);

    // Second run with a new FolderSync over the same watch dir + store:
    // the file already exists in storage so the count should be 0.
    let sync2 = FolderSync::new(watch_dir, Arc::clone(&storage), tx);
    let second = sync2.sync_once().await.expect("second sync");
    assert_eq!(second, 0, "second sync should skip already-uploaded file");
}

// ── sync_once with empty directory returns 0 ─────────────────────────────────

#[tokio::test]
async fn test_sync_once_empty_directory() {
    let watch_dir = env::temp_dir().join(format!("folder_sync_empty_{}", unique()));
    tokio::fs::create_dir_all(&watch_dir).await.expect("mkdir");

    let store_root = env::temp_dir().join(format!("folder_sync_store_empty_{}", unique()));
    let storage = Arc::new(MamStorage::local(&store_root).await.expect("storage"));

    let bus = MamEventBus::new(16);
    let tx = bus.sender();

    let sync = FolderSync::new(watch_dir, storage, tx);
    let count = sync.sync_once().await.expect("sync empty dir");
    assert_eq!(count, 0);
}

// ── sync_once with multiple files uploads all of them ────────────────────────

#[tokio::test]
async fn test_sync_once_multiple_files() {
    let watch_dir = env::temp_dir().join(format!("folder_sync_multi_{}", unique()));
    tokio::fs::create_dir_all(&watch_dir).await.expect("mkdir");

    for (name, content) in [
        ("video.mp4", "video content"),
        ("audio.wav", "audio content"),
        ("image.png", "image content"),
    ] {
        tokio::fs::write(watch_dir.join(name), content)
            .await
            .expect("write file");
    }

    let store_root = env::temp_dir().join(format!("folder_sync_store_multi_{}", unique()));
    let storage = Arc::new(MamStorage::local(&store_root).await.expect("storage"));

    let bus = MamEventBus::new(32);
    let mut rx = bus.subscribe();
    let tx = bus.sender();

    let sync = FolderSync::new(watch_dir, Arc::clone(&storage), tx);
    let count = sync.sync_once().await.expect("sync");
    assert_eq!(count, 3, "should upload 3 files");

    // All 3 events should be AssetIngested
    for _ in 0..3 {
        let event = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
            .await
            .expect("event timeout")
            .expect("recv error");
        assert!(
            matches!(event, MamEvent::AssetIngested { .. }),
            "expected AssetIngested"
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn unique() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    format!("{nanos:x}")
}
