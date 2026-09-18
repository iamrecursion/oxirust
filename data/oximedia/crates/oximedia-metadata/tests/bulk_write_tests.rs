//! Integration tests for BulkUpdateEngine batched on-disk write (Wave 14, Slice C).

use oximedia_metadata::bulk_update::{BulkUpdateEngine, BulkWriteMode, FieldUpdate};
use std::collections::HashMap;

/// Helper: create a BulkUpdateEngine pre-loaded with a small set of key-value pairs.
fn make_engine() -> BulkUpdateEngine {
    let mut data = HashMap::new();
    data.insert("title".to_string(), "My Test Media".to_string());
    data.insert("artist".to_string(), "Test Artist".to_string());
    data.insert("year".to_string(), "2026".to_string());
    BulkUpdateEngine::with_data(data)
}

/// Roundtrip test: write three sidecar files and verify they are non-empty and
/// contain the expected key.
#[test]
fn test_bulk_write_sidecar_roundtrip() {
    let tmp = std::env::temp_dir().join(format!(
        "oximedia_bulk_write_sidecar_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    // Create 3 dummy media files
    let paths: Vec<std::path::PathBuf> = (0..3)
        .map(|i| {
            let p = tmp.join(format!("clip{i}.mp4"));
            std::fs::write(&p, b"dummy media content").expect("write dummy file");
            p
        })
        .collect();

    let engine = make_engine();
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
    let results = engine.write_batch(&path_refs, BulkWriteMode::Sidecar);

    assert_eq!(results.len(), 3, "one result per input path");

    for (i, result) in results.iter().enumerate() {
        assert!(
            result.success,
            "sidecar write {i} failed: {:?}",
            result.error
        );
        assert!(
            result.bytes_written > 0,
            "sidecar {i} must have non-zero bytes"
        );
        assert!(
            result.error.is_none(),
            "sidecar {i} has unexpected error: {:?}",
            result.error
        );

        // The sidecar file must exist and be non-empty
        let content =
            std::fs::read_to_string(&result.path).expect("sidecar file should be readable");
        assert!(!content.is_empty(), "sidecar {i} content must not be empty");

        // Must contain the "title" key we set
        assert!(
            content.contains("title"),
            "sidecar {i} must contain 'title' key, got: {content}"
        );
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp);
}

/// For an unsupported format (plain .txt) Embed mode must return success: false
/// with a non-None error — it must not panic.
#[test]
fn test_bulk_write_embed_unsupported() {
    let tmp = std::env::temp_dir().join(format!(
        "oximedia_bulk_write_embed_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let txt_path = tmp.join("notes.txt");
    std::fs::write(&txt_path, b"plain text content, not a media file").expect("write txt file");

    let engine = make_engine();
    let results = engine.write_batch(&[txt_path.as_path()], BulkWriteMode::Embed);

    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(!r.success, "embed of unsupported format must not succeed");
    assert!(
        r.error.is_some(),
        "embed of unsupported format must set error field"
    );
    assert_eq!(
        r.bytes_written, 0,
        "no bytes must be written for unsupported format"
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Verify that apply() + write_batch() integration: updates applied via `apply()`
/// are reflected in the sidecar output.
#[test]
fn test_bulk_write_sidecar_reflects_apply() {
    let tmp = std::env::temp_dir().join(format!(
        "oximedia_bulk_write_apply_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let media = tmp.join("video.mkv");
    std::fs::write(&media, b"dummy mkv").expect("write dummy file");

    let mut engine = BulkUpdateEngine::new();
    let _ = engine.apply(&[
        FieldUpdate::set("title", "Applied Title"),
        FieldUpdate::set("genre", "Documentary"),
    ]);

    let results = engine.write_batch(&[media.as_path()], BulkWriteMode::Sidecar);
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(r.success, "write must succeed: {:?}", r.error);

    let content = std::fs::read_to_string(&r.path).expect("read sidecar");
    assert!(
        content.contains("Applied Title"),
        "sidecar must include applied title"
    );
    assert!(content.contains("genre"), "sidecar must include genre key");

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Empty engine produces a valid (but empty-ish) sidecar, not an error.
#[test]
fn test_bulk_write_sidecar_empty_engine() {
    let tmp = std::env::temp_dir().join(format!(
        "oximedia_bulk_write_empty_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    let f = tmp.join("clip.mp4");
    std::fs::write(&f, b"dummy").expect("write file");

    let engine = BulkUpdateEngine::new();
    let results = engine.write_batch(&[f.as_path()], BulkWriteMode::Sidecar);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].success,
        "empty engine sidecar write must succeed"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Embed mode with a valid ID3v2 header must succeed (magic bytes `ID3`).
#[test]
fn test_bulk_write_embed_id3v2() {
    let tmp = std::env::temp_dir().join(format!(
        "oximedia_bulk_write_id3_test_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&tmp).expect("create tmp dir");

    // Minimal ID3v2 header (10-byte header + 0 payload)
    let mut id3_data = vec![b'I', b'D', b'3', 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    id3_data.extend_from_slice(b"audio payload");
    let mp3 = tmp.join("song.mp3");
    std::fs::write(&mp3, &id3_data).expect("write mp3");

    let engine = make_engine();
    let results = engine.write_batch(&[mp3.as_path()], BulkWriteMode::Embed);
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(r.success, "ID3v2 embed must succeed: {:?}", r.error);
    assert!(r.bytes_written > 0);

    let _ = std::fs::remove_dir_all(&tmp);
}
