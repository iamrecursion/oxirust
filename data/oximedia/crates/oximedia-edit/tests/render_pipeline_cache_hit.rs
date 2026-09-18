//! Integration tests for `RawFrameCache` hit/miss behaviour in the rendering
//! pipeline and the standalone `RawFrameCache` API.

use oximedia_edit::{RawFrameCache, RAW_FRAME_CACHE_CAPACITY};

// ─── RawFrameCache unit tests via the pipeline ───────────────────────────────

#[test]
fn test_cache_hit_does_not_re_render() {
    let mut cache = RawFrameCache::new(4);
    let mut render_count = 0usize;

    cache.get_or_render(42, || {
        render_count += 1;
        vec![0u8; 64]
    });
    cache.get_or_render(42, || {
        render_count += 1; // must NOT be called on second access
        vec![0u8; 64]
    });

    assert_eq!(render_count, 1, "render_fn must only be called once");
}

#[test]
fn test_cache_miss_on_different_key() {
    let mut cache = RawFrameCache::new(4);
    let mut render_count = 0usize;

    cache.get_or_render(1, || {
        render_count += 1;
        vec![1u8; 32]
    });
    cache.get_or_render(2, || {
        render_count += 1; // different key → must render
        vec![2u8; 32]
    });

    assert_eq!(render_count, 2, "each distinct key must render once");
}

#[test]
fn test_lru_eviction_at_capacity() {
    let mut cache = RawFrameCache::new(4);

    for i in 0u64..4 {
        cache.get_or_render(i, || vec![i as u8]);
    }
    // Insert one more → evicts frame 0.
    cache.get_or_render(4, || vec![4u8]);

    assert!(!cache.contains(0), "oldest frame (0) must be evicted");
    assert!(
        cache.contains(4),
        "newly inserted frame (4) must be present"
    );
}

#[test]
fn test_insert_and_retrieve() {
    let mut cache = RawFrameCache::new(8);
    let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
    cache.insert(99, data.clone());
    assert_eq!(cache.get(99), Some(data.as_slice()));
}

#[test]
fn test_invalidate_removes_entry() {
    let mut cache = RawFrameCache::new(4);
    cache.insert(7, vec![7u8]);
    cache.invalidate(7);
    assert!(!cache.contains(7));
    assert!(cache.get(7).is_none());
}

#[test]
fn test_clear_empties_cache() {
    let mut cache = RawFrameCache::new(8);
    for i in 0u64..8 {
        cache.insert(i, vec![i as u8]);
    }
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_default_capacity_constant() {
    let cache = RawFrameCache::new(RAW_FRAME_CACHE_CAPACITY);
    assert_eq!(cache.capacity(), RAW_FRAME_CACHE_CAPACITY);
}

#[test]
fn test_capacity_clamped_minimum() {
    let cache = RawFrameCache::new(0);
    assert!(cache.capacity() >= 1, "capacity must be at least 1");
}

#[test]
fn test_get_missing_returns_none() {
    let cache = RawFrameCache::new(4);
    assert!(cache.get(999).is_none());
}

#[test]
fn test_eviction_order_is_insertion_order() {
    let mut cache = RawFrameCache::new(3);
    cache.insert(10, vec![10]);
    cache.insert(20, vec![20]);
    cache.insert(30, vec![30]);

    // Frame 40 → evicts 10 (oldest).
    cache.insert(40, vec![40]);
    assert!(!cache.contains(10));
    assert!(cache.contains(20));
    assert!(cache.contains(30));
    assert!(cache.contains(40));
}
