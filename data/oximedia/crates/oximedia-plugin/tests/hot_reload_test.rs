//! Integration tests for hot-reload with simulated file-modification events.
//!
//! These tests verify the complete hot-reload pipeline:
//! 1. Register a plugin and start watching its file.
//! 2. Simulate a file modification by changing the content.
//! 3. Detect the change via `check_for_changes`.
//! 4. Reload the plugin and update the hash.
//! 5. Verify seamless reload — no change detected after hash update.

use oximedia_plugin::hot_reload::{
    compute_hash, compute_hash_file, GracefulReload, HotReloadManager, PluginVersion, ReloadPolicy,
};
use oximedia_plugin::version_resolver::SemVer;
use std::collections::HashMap;

fn make_version(id: &str, major: u32, minor: u32, patch: u32) -> PluginVersion {
    PluginVersion::new(id, SemVer::new(major, minor, patch), 0)
}

// ── Simulated file-modification workflow ──────────────────────────────────────

/// Simulated file content for a plugin binary.
const V1_CONTENT: &[u8] = b"plugin binary v1.0.0 -- lots of code here (simulated)";
const V2_CONTENT: &[u8] = b"plugin binary v1.1.0 -- updated code with bug fixes";
const V3_CONTENT: &[u8] = b"plugin binary v1.2.0 -- new features";

/// Full hot-reload lifecycle: watch → detect change → reload → no further change.
#[test]
fn test_hot_reload_seamless_lifecycle() {
    let mut mgr = HotReloadManager::new(ReloadPolicy::OnChange);

    // 1. Load plugin v1 and register it.
    let v1 = make_version("my-codec", 1, 0, 0);
    mgr.register_loaded(v1);

    // 2. Start watching with v1 content.
    mgr.watch("my-codec", "/lib/my-codec.so", V1_CONTENT);
    assert_eq!(mgr.watchers.len(), 1);

    // 3. No change initially.
    let mut current = HashMap::new();
    current.insert("my-codec".to_string(), V1_CONTENT.to_vec());
    let changed = mgr.check_for_changes(&current);
    assert!(changed.is_empty(), "No change expected initially");

    // 4. Simulate file modification: content changes to v2.
    current.insert("my-codec".to_string(), V2_CONTENT.to_vec());
    let changed = mgr.check_for_changes(&current);
    assert_eq!(changed.len(), 1);
    assert_eq!(changed[0], "my-codec");

    // 5. Reload the plugin.
    let v2 = make_version("my-codec", 1, 1, 0);
    mgr.reload_plugin("my-codec", v2).expect("reload v2");
    assert_eq!(mgr.loaded_plugins["my-codec"].version, SemVer::new(1, 1, 0));

    // 6. Update the watch hash to reflect the new content.
    mgr.update_hash("my-codec", V2_CONTENT);

    // 7. No change after hash update.
    let changed = mgr.check_for_changes(&current);
    assert!(changed.is_empty(), "No change expected after hash update");
}

/// Multiple watchers — only the modified one is flagged.
#[test]
fn test_multiple_watchers_only_changed_flagged() {
    let mut mgr = HotReloadManager::new(ReloadPolicy::OnChange);
    mgr.register_loaded(make_version("codec-a", 1, 0, 0));
    mgr.register_loaded(make_version("codec-b", 2, 0, 0));

    mgr.watch("codec-a", "/lib/a.so", V1_CONTENT);
    mgr.watch("codec-b", "/lib/b.so", V1_CONTENT);

    let mut current = HashMap::new();
    current.insert("codec-a".to_string(), V2_CONTENT.to_vec()); // a changed
    current.insert("codec-b".to_string(), V1_CONTENT.to_vec()); // b unchanged

    let changed = mgr.check_for_changes(&current);
    assert_eq!(changed.len(), 1);
    assert_eq!(changed[0], "codec-a");
}

/// Two successive modifications to the same plugin.
#[test]
fn test_two_successive_modifications() {
    let mut mgr = HotReloadManager::new(ReloadPolicy::OnChange);
    mgr.register_loaded(make_version("p", 1, 0, 0));
    mgr.watch("p", "/lib/p.so", V1_CONTENT);

    // First modification: v1 → v2.
    {
        let mut current = HashMap::new();
        current.insert("p".to_string(), V2_CONTENT.to_vec());
        let changed = mgr.check_for_changes(&current);
        assert_eq!(changed.len(), 1);

        mgr.reload_plugin("p", make_version("p", 1, 1, 0))
            .expect("reload");
        mgr.update_hash("p", V2_CONTENT);
    }

    // Second modification: v2 → v3.
    {
        let mut current = HashMap::new();
        current.insert("p".to_string(), V3_CONTENT.to_vec());
        let changed = mgr.check_for_changes(&current);
        assert_eq!(changed.len(), 1);

        mgr.reload_plugin("p", make_version("p", 1, 2, 0))
            .expect("reload");
        mgr.update_hash("p", V3_CONTENT);

        let changed = mgr.check_for_changes(&current);
        assert!(changed.is_empty());
    }

    assert_eq!(mgr.loaded_plugins["p"].version, SemVer::new(1, 2, 0));
}

/// GracefulReload with simulated file modification.
#[test]
fn test_graceful_reload_with_file_change() {
    let mut mgr = HotReloadManager::new(ReloadPolicy::OnChange);
    mgr.register_loaded(make_version("g", 1, 0, 0));
    mgr.watch("g", "/lib/g.so", V1_CONTENT);

    // Detect change.
    let mut current = HashMap::new();
    current.insert("g".to_string(), V2_CONTENT.to_vec());
    let changed = mgr.check_for_changes(&current);
    assert_eq!(changed.len(), 1);

    // Graceful drain-and-reload.
    let gr = GracefulReload::new(50);
    gr.drain_and_reload("g", &mut mgr, make_version("g", 1, 1, 0))
        .expect("graceful reload");

    mgr.update_hash("g", V2_CONTENT);

    // No further change.
    let still_changed = mgr.check_for_changes(&current);
    assert!(still_changed.is_empty());
}

/// Reload with Disabled policy should still work (policy only affects auto-check).
#[test]
fn test_reload_disabled_policy_still_allows_manual_reload() {
    let mut mgr = HotReloadManager::new(ReloadPolicy::Disabled);
    mgr.register_loaded(make_version("p", 1, 0, 0));
    assert!(!mgr.auto_reload_enabled());

    // Manual reload is still possible.
    mgr.reload_plugin("p", make_version("p", 2, 0, 0))
        .expect("manual reload");
    assert_eq!(mgr.loaded_plugins["p"].version, SemVer::new(2, 0, 0));
}

/// Reload-then-unload lifecycle.
#[test]
fn test_reload_then_unload() {
    let mut mgr = HotReloadManager::new(ReloadPolicy::OnChange);
    mgr.register_loaded(make_version("p", 1, 0, 0));
    mgr.watch("p", "/lib/p.so", V1_CONTENT);

    let v2 = make_version("p", 1, 1, 0);
    mgr.reload_plugin("p", v2).expect("reload");
    let removed = mgr.unload_plugin("p").expect("unload");
    assert_eq!(removed.version, SemVer::new(1, 1, 0));
    assert!(!mgr.loaded_plugins.contains_key("p"));
}

/// compute_hash_file round-trip with a temporary file.
#[test]
fn test_compute_hash_file_matches_compute_hash() {
    let dir = std::env::temp_dir().join("oximedia-plugin-hot-reload-test");
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("test_plugin.so");
    std::fs::write(&path, V1_CONTENT).expect("write");

    let hash_file = compute_hash_file(&path).expect("hash_file");
    let hash_mem = compute_hash(V1_CONTENT);
    assert_eq!(
        hash_file, hash_mem,
        "memmap hash should match in-memory hash"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// compute_hash_file detects change after file overwrite.
#[test]
fn test_compute_hash_file_detects_change() {
    let dir = std::env::temp_dir().join("oximedia-plugin-hot-reload-test2");
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("plugin.so");

    std::fs::write(&path, V1_CONTENT).expect("write v1");
    let h1 = compute_hash_file(&path).expect("hash v1");

    std::fs::write(&path, V2_CONTENT).expect("write v2");
    let h2 = compute_hash_file(&path).expect("hash v2");

    assert_ne!(h1, h2);

    let _ = std::fs::remove_dir_all(&dir);
}

/// compute_hash_file on empty file returns FNV offset basis.
#[test]
fn test_compute_hash_file_empty() {
    let dir = std::env::temp_dir().join("oximedia-plugin-hot-reload-test3");
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("empty.so");
    std::fs::write(&path, b"").expect("write empty");

    let h = compute_hash_file(&path).expect("hash empty");
    assert_eq!(h, compute_hash(&[]));

    let _ = std::fs::remove_dir_all(&dir);
}
