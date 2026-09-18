//! Integration tests for `deep_scan_mmap`.
//!
//! Writes a synthetic file of ≥ 5 MiB with a broken header pattern, then:
//!
//! 1. Calls `deep_scan_mmap` — which should take the mmap path.
//! 2. Calls `deep_scan_streaming` on the same data — the streaming path.
//! 3. Asserts that both paths return the same `IssueType` set (order may
//!    differ since we sort by offset before comparison).

use oximedia_repair::detect::scan::{deep_scan_mmap, deep_scan_streaming};
use oximedia_repair::IssueType;
use std::io::Write;
use std::path::PathBuf;

/// Create a temp file containing `data`; return its path.
///
/// The filename embeds the current process ID so that concurrent test runs
/// (e.g. under `cargo nextest`) cannot collide on the same path even when the
/// test is momentarily not in the `serial-latency` group.
fn temp_file(name: &str, data: &[u8]) -> PathBuf {
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("oximedia_mmap_test_{}_{}", name, pid));
    let mut f = std::fs::File::create(&path).expect("create temp file");
    f.write_all(data).expect("write temp file");
    path
}

/// Build synthetic broken-media data of at least `min_bytes`.
///
/// The buffer:
/// - starts with a recognisable-but-invalid "header" sentinel,
/// - contains runs of zeros longer than 4096 bytes (triggers `CorruptPackets`),
/// - has scattered MPEG-TS sync bytes,
/// - ends with 16 zero bytes (triggers `Truncated`).
fn make_synthetic_broken(min_bytes: usize) -> Vec<u8> {
    let mut data = vec![0u8; min_bytes];

    // Broken header sentinel (not any real format magic)
    data[0] = 0xDE;
    data[1] = 0xAD;
    data[2] = 0xBE;
    data[3] = 0xEF;

    // Scatter some MPEG-TS sync bytes so there is non-zero content
    for i in (1000..min_bytes - 1000).step_by(500) {
        if i + 188 < min_bytes {
            data[i] = 0x47; // TS sync
        }
    }

    // Large zero run in the middle — should trigger CorruptPackets detection
    let mid = min_bytes / 2;
    let run_len = 8192.min(mid);
    for b in data[mid..mid + run_len].iter_mut() {
        *b = 0;
    }

    // Last 16 bytes are zeros to trigger truncation detection
    let last = min_bytes - 16;
    for b in data[last..].iter_mut() {
        *b = 0;
    }

    data
}

/// Collect the sorted set of `IssueType`s from a list of `Issue`s.
fn issue_type_set(issues: &[oximedia_repair::Issue]) -> Vec<IssueType> {
    let mut types: Vec<IssueType> = issues.iter().map(|i| i.issue_type).collect();
    // Sort by discriminant (Debug string) for stable comparison
    types.sort_by_key(|t| format!("{:?}", t));
    types.dedup();
    types
}

#[test]
fn test_mmap_scan_file_larger_than_threshold() {
    // 5 MiB — well above the 4 MiB threshold
    let data = make_synthetic_broken(5 * 1024 * 1024);
    let path = temp_file("broken_5mib.bin", &data);

    let result = deep_scan_mmap(&path);
    assert!(
        result.is_ok(),
        "deep_scan_mmap should not error: {:?}",
        result.err()
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_mmap_and_streaming_agree_on_issue_types() {
    // 5 MiB file — mmap path is taken
    let data = make_synthetic_broken(5 * 1024 * 1024);
    let mmap_path = temp_file("compare_mmap.bin", &data);
    let stream_path = temp_file("compare_stream.bin", &data);

    let mmap_issues = deep_scan_mmap(&mmap_path).expect("mmap scan failed");
    let stream_issues = deep_scan_streaming(&stream_path).expect("streaming scan failed");

    let mmap_types = issue_type_set(&mmap_issues);
    let stream_types = issue_type_set(&stream_issues);

    assert_eq!(
        mmap_types, stream_types,
        "mmap and streaming should detect the same issue types.\n\
         mmap:     {:?}\n\
         streaming: {:?}",
        mmap_types, stream_types
    );

    let _ = std::fs::remove_file(&mmap_path);
    let _ = std::fs::remove_file(&stream_path);
}

#[test]
fn test_mmap_falls_back_for_small_file() {
    // 1 MiB — below threshold; deep_scan_mmap should delegate to streaming
    let data = make_synthetic_broken(1024 * 1024);
    let path = temp_file("small_file.bin", &data);

    // Both paths should return Ok
    let mmap_result = deep_scan_mmap(&path);
    let stream_result = deep_scan_streaming(&path);

    assert!(mmap_result.is_ok(), "mmap fallback should succeed");
    assert!(stream_result.is_ok(), "streaming should succeed");

    // Same issue types in both paths
    let mmap_types = issue_type_set(&mmap_result.expect("mmap scan failed"));
    let stream_types = issue_type_set(&stream_result.expect("streaming scan failed"));
    assert_eq!(mmap_types, stream_types);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_mmap_on_empty_file_returns_no_issues() {
    let path = temp_file("empty.bin", &[]);
    // File is 0 bytes — below threshold, streaming path is taken
    let result = deep_scan_mmap(&path);
    assert!(result.is_ok());
    // An empty file produces no corruption issues
    let issues = result.unwrap();
    assert!(
        issues.is_empty(),
        "empty file should produce no issues, got: {:?}",
        issues
    );
    let _ = std::fs::remove_file(&path);
}
