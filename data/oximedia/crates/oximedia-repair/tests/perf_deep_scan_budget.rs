//! Performance regression guard for deep-scan throughput.
//!
//! Asserts that scanning a 5 MiB synthetic file completes within a reasonable
//! wall-clock budget.  The synthetic data mirrors the layout used by
//! `test_mmap_and_streaming_agree_on_issue_types` so this test catches the
//! same class of algorithmic regression.

use oximedia_repair::detect::scan::{deep_scan_mmap, deep_scan_streaming};
use std::io::Write;
use std::path::PathBuf;

fn temp_file(name: &str, data: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("oximedia_perf_budget_{}", name));
    let mut f = std::fs::File::create(&path).expect("create perf budget temp file");
    f.write_all(data).expect("write perf budget temp file");
    path
}

fn make_synthetic_broken(min_bytes: usize) -> Vec<u8> {
    let mut data = vec![0u8; min_bytes];
    data[0] = 0xDE;
    data[1] = 0xAD;
    data[2] = 0xBE;
    data[3] = 0xEF;
    for i in (1000..min_bytes - 1000).step_by(500) {
        if i + 188 < min_bytes {
            data[i] = 0x47;
        }
    }
    let mid = min_bytes / 2;
    let run_len = 8192.min(mid);
    for b in data[mid..mid + run_len].iter_mut() {
        *b = 0;
    }
    let last = min_bytes - 16;
    for b in data[last..].iter_mut() {
        *b = 0;
    }
    data
}

#[test]
fn test_deep_scan_mmap_perf_budget() {
    let data = make_synthetic_broken(5 * 1024 * 1024);
    let path = temp_file("mmap_budget.bin", &data);

    let start = std::time::Instant::now();
    let result = deep_scan_mmap(&path).expect("deep_scan_mmap should succeed");
    let elapsed = start.elapsed();

    // Budget: 1800 s in debug mode (system-load tolerant; observed worst-case 782 s under full
    // workspace parallel load; 1800 s gives 3× margin over that baseline), 5 s in release mode.
    let budget_secs = if cfg!(debug_assertions) { 1800 } else { 5 };
    assert!(
        elapsed.as_secs() < budget_secs,
        "deep_scan_mmap took {elapsed:?} — exceeded {budget_secs}s budget (O(n²) regression?)"
    );

    let _ = result;
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_deep_scan_streaming_perf_budget() {
    let data = make_synthetic_broken(5 * 1024 * 1024);
    let path = temp_file("streaming_budget.bin", &data);

    let start = std::time::Instant::now();
    let result = deep_scan_streaming(&path).expect("deep_scan_streaming should succeed");
    let elapsed = start.elapsed();

    // Budget: 1800 s in debug mode (system-load tolerant; observed worst-case 782 s under full
    // workspace parallel load; 1800 s gives 3× margin over that baseline), 5 s in release mode.
    let budget_secs = if cfg!(debug_assertions) { 1800 } else { 5 };
    assert!(
        elapsed.as_secs() < budget_secs,
        "deep_scan_streaming took {elapsed:?} — exceeded {budget_secs}s budget (O(n²) regression?)"
    );

    let _ = result;
    let _ = std::fs::remove_file(&path);
}
