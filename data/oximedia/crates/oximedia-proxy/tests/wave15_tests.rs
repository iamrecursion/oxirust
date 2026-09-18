//! Wave 15 integration tests for oximedia-proxy.
//!
//! Covers `RangeProxyIndex` path-prefix queries and timecode-range queries.

use oximedia_proxy::proxy_index::{ProxyEntry, RangeProxyIndex};

// ── helpers ───────────────────────────────────────────────────────────────────

fn entry_pts(path: &str, pts: u64, proxy_suffix: &str) -> ProxyEntry {
    ProxyEntry::with_timecode(
        path,
        &format!("/proxy/{proxy_suffix}.mp4"),
        640,
        360,
        500,
        pts,
    )
}

// ── Test 1: path-prefix query returns only matching entries ───────────────────

#[test]
fn test_btree_range_query_path_prefix() {
    let mut idx = RangeProxyIndex::new();

    // Path A — 3 entries
    idx.insert(entry_pts("/media/projectA/reel1.mov", 0, "a1"));
    idx.insert(entry_pts("/media/projectA/reel2.mov", 0, "a2"));
    idx.insert(entry_pts("/media/projectA/reel3.mov", 0, "a3"));

    // Path B — 2 entries
    idx.insert(entry_pts("/media/projectB/reel1.mov", 0, "b1"));
    idx.insert(entry_pts("/media/projectB/reel2.mov", 0, "b2"));

    // Query by prefix for projectB only
    let found = idx.find_by_path_prefix("/media/projectB/");
    assert_eq!(
        found.len(),
        2,
        "expected exactly 2 projectB entries, got {}: {:?}",
        found.len(),
        found.iter().map(|e| &e.original_path).collect::<Vec<_>>()
    );

    for e in &found {
        assert!(
            e.original_path.starts_with("/media/projectB/"),
            "unexpected path: {}",
            e.original_path
        );
    }
}

// ── Test 2: timecode range query returns inclusive PTS span entries ────────────

#[test]
fn test_btree_range_query_timecode() {
    let path = "/media/interview.mov";
    let mut idx = RangeProxyIndex::new();

    idx.insert(entry_pts(path, 0, "tc0"));
    idx.insert(entry_pts(path, 100, "tc100"));
    idx.insert(entry_pts(path, 200, "tc200"));
    idx.insert(entry_pts(path, 300, "tc300"));

    // Query [50, 250] — should return only pts=100 and pts=200
    let found = idx.find_in_timecode_range(path, 50, 250);
    assert_eq!(
        found.len(),
        2,
        "expected 2 entries in range [50,250], got {}: {:?}",
        found.len(),
        found.iter().map(|e| e.timecode_pts).collect::<Vec<_>>()
    );

    let pts_set: std::collections::HashSet<u64> = found.iter().map(|e| e.timecode_pts).collect();
    assert!(pts_set.contains(&100), "missing pts=100; got: {pts_set:?}");
    assert!(pts_set.contains(&200), "missing pts=200; got: {pts_set:?}");
    assert!(
        !pts_set.contains(&0),
        "pts=0 should not be in [50,250]; got: {pts_set:?}"
    );
    assert!(
        !pts_set.contains(&300),
        "pts=300 should not be in [50,250]; got: {pts_set:?}"
    );
}
