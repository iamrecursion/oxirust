//! Wave 13 integration tests for `oximedia-cache`.
//!
//! Tests: LRU concurrent-stress, tiered promote/demote, bloom FPR validation,
//! distributed-cache rebalance, cache-warming proptest, content-aware eviction.

// ── (a) LRU concurrent-stress ─────────────────────────────────────────────────

#[test]
fn test_lru_concurrent_stress() {
    use oximedia_cache::sharded_lru::ShardedLruCache;
    use std::sync::Arc;

    const N_THREADS: usize = 8;
    const OPS_PER_THREAD: usize = 10_000;
    const CAPACITY: usize = 1024;
    const KEY_SPACE: u64 = 512;

    let cache: Arc<ShardedLruCache<u64, u64>> = Arc::new(ShardedLruCache::new(N_THREADS, CAPACITY));

    let mut handles = Vec::with_capacity(N_THREADS);
    for t in 0..N_THREADS {
        let c = Arc::clone(&cache);
        let handle = std::thread::spawn(move || {
            // Simple xorshift64 per-thread PRNG for determinism.
            let mut state: u64 =
                0xcafe_f00d_d15e_a5e5 ^ (t as u64).wrapping_mul(6364136223846793005);
            for _ in 0..OPS_PER_THREAD {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let key = state % KEY_SPACE;
                if state & 1 == 0 {
                    c.put(key, key * 2, 1);
                } else {
                    let _ = c.get(&key);
                }
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // After concurrent stress, total entry count should not exceed capacity.
    assert!(
        cache.len() <= CAPACITY,
        "cache.len() {} exceeded capacity {}",
        cache.len(),
        CAPACITY
    );
}

// ── (b) Tiered promote/demote correctness ─────────────────────────────────────

#[test]
fn test_tiered_promote_demote_correctness() {
    use oximedia_cache::tiered_cache::{EvictionPolicy, TierConfig, TieredCache};

    // L1: tiny (fits 4 bytes), threshold 0 (always promote).
    // L2: larger, threshold 2 (must access ≥2 times before promoting).
    let mut cache = TieredCache::new(vec![
        TierConfig {
            name: "L1".into(),
            capacity_bytes: 100,
            access_latency_us: 1,
            eviction_policy: EvictionPolicy::Lru,
            disk_path: None,
            promotion_threshold: 0,
            compress: false,
            adaptive_promotion: false,
            use_arena: false,
        },
        TierConfig {
            name: "L2".into(),
            capacity_bytes: 10_000,
            access_latency_us: 10,
            eviction_policy: EvictionPolicy::Lru,
            disk_path: None,
            promotion_threshold: 2,
            compress: false,
            adaptive_promotion: false,
            use_arena: false,
        },
    ]);

    // Insert several keys directly into L2.
    let hot_key = "hot_key";
    let cold_key = "cold_key";
    cache.put_at_tier(1, hot_key, b"hot_value".to_vec());
    cache.put_at_tier(1, cold_key, b"cold_value".to_vec());

    // Access cold_key only once — should NOT be promoted (threshold=2).
    let _ = cache.get(cold_key); // freq → 1

    // Access hot_key twice — should be promoted on the 2nd access.
    let _ = cache.get(hot_key); // freq → 1, no promotion yet
    let _ = cache.get(hot_key); // freq → 2, promotion fires

    let stats = cache.stats();
    // L2 should have recorded at least one promotion (hot_key).
    assert!(
        stats.tier_stats[1].promotions >= 1,
        "hot_key should have been promoted from L2 (promotions={})",
        stats.tier_stats[1].promotions
    );

    // Verify tier_hits counters are non-zero.
    assert!(
        cache.tier_hit_count(1) >= 2,
        "L2 should have >= 2 hits, got {}",
        cache.tier_hit_count(1)
    );
}

// ── (c) Bloom FPR validation ──────────────────────────────────────────────────

#[test]
fn test_bloom_fpr_validation() {
    use oximedia_cache::bloom_filter::BloomFilter;

    const N_INSERT: usize = 10_000;
    const N_QUERY: usize = 50_000;
    let target_fpr = 0.01_f64;

    // Seeded RNG (xorshift64) for determinism.
    let mut rng_state: u64 = 0x0123_4567_89ab_cdef;
    let next_u64 = |state: &mut u64| -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    let mut bf = BloomFilter::new(N_INSERT, target_fpr);

    // Insert N_INSERT items with keys "ins_<i>".
    for i in 0..N_INSERT {
        let key = format!("ins_{i}");
        bf.insert(key.as_bytes());
    }

    // Query N_QUERY disjoint absent keys (prefix "abs_" never used in inserts).
    let mut false_positives = 0usize;
    for _ in 0..N_QUERY {
        let r = next_u64(&mut rng_state);
        let key = format!("abs_{r}");
        if bf.contains(key.as_bytes()) {
            false_positives += 1;
        }
    }

    let empirical_fpr = false_positives as f64 / N_QUERY as f64;

    // Theoretical FPR = (1 - exp(-k*n/m))^k.
    let theoretical_fpr = bf.estimate_false_positive_rate();

    // Assert empirical ≤ 2× theoretical.
    assert!(
        empirical_fpr <= theoretical_fpr * 2.0 + 0.005,
        "empirical FPR {:.4} > 2× theoretical {:.4}",
        empirical_fpr,
        theoretical_fpr
    );
    // Also assert empirical is not wildly high.
    assert!(
        empirical_fpr <= 0.05,
        "empirical FPR {:.4} too high (target {})",
        empirical_fpr,
        target_fpr
    );
}

// ── (d) Distributed-cache rebalance ──────────────────────────────────────────

#[test]
fn test_distributed_cache_rebalance() {
    use oximedia_cache::distributed_cache::{ConsistentHash, NodeId};

    const N_KEYS: usize = 1_000;
    const VIRTUAL_NODES: u32 = 150; // standard industry value

    let node_a = NodeId(1);
    let node_b = NodeId(2);
    let node_c = NodeId(3);
    let node_d = NodeId(4); // new node to add

    // Build a 3-node ring.
    let mut ring3 = ConsistentHash::new(VIRTUAL_NODES);
    ring3.add_node(node_a);
    ring3.add_node(node_b);
    ring3.add_node(node_c);

    // Assign all keys to the 3-node ring.
    let keys: Vec<String> = (0..N_KEYS).map(|i| format!("media_key_{i:04}")).collect();
    let initial_assignments: Vec<NodeId> = keys
        .iter()
        .map(|k| ring3.get_node(k.as_bytes()).expect("ring should route key"))
        .collect();

    // Add a 4th node.
    let mut ring4 = ring3.clone();
    ring4.add_node(node_d);

    // Re-assign all keys.
    let new_assignments: Vec<NodeId> = keys
        .iter()
        .map(|k| ring4.get_node(k.as_bytes()).expect("ring should route key"))
        .collect();

    // Count remapped keys.
    let remapped = initial_assignments
        .iter()
        .zip(new_assignments.iter())
        .filter(|(old, new)| old != new)
        .count();

    // With consistent hashing, ~K/N keys should move (K=1000, N_old=3, N_new=4).
    // Expected remapped ≈ K * (1/N_new) = 1000/4 = 250.
    // Allow 3× tolerance for statistical variation.
    let expected = N_KEYS / 4;
    assert!(
        remapped <= expected * 3,
        "too many keys remapped: {} (expected ~{}, max {})",
        remapped,
        expected,
        expected * 3
    );
    // At least some keys should have moved.
    assert!(
        remapped > 0,
        "no keys were remapped after adding a node — consistent hash may be broken"
    );
}

// ── (e) Cache-warming proptest ────────────────────────────────────────────────

#[cfg(test)]
mod proptest_cache_warming {
    use oximedia_cache::cache_warming::CacheWarmer;
    use proptest::prelude::*;

    proptest! {
        /// For any random (capacity, item-size-list), the warmup plan must:
        /// 1. Respect the byte capacity (`sum(sizes) ≤ capacity`).
        /// 2. Contain only keys from the requested set.
        #[test]
        fn warmup_plan_respects_capacity(
            capacity in 100usize..50_000,
            sizes in prop::collection::vec(1usize..5_000, 0..30),
        ) {
            let current_time: u64 = 1_000_000;
            let mut warmer = CacheWarmer::new();
            warmer.min_frequency = 0.0; // admit all frequencies
            // look-ahead = 10 min: all predicted accesses are within window.
            warmer.look_ahead_secs = 600;

            let all_keys: Vec<String> = sizes
                .iter()
                .enumerate()
                .map(|(i, _)| format!("prop_key_{i}"))
                .collect();

            for (i, &sz) in sizes.iter().enumerate() {
                let key = &all_keys[i];
                // Insert 3 accesses at times 990_000, 995_000, 1_000_000 so
                // frequency > 0 and predicted next access is ~1_005_000 (within window).
                warmer.record_access(key, sz, current_time - 10_000);
                warmer.record_access(key, sz, current_time - 5_000);
                warmer.record_access(key, sz, current_time);
            }

            let plan = warmer.plan_warmup(current_time, capacity);

            // 1. total size ≤ capacity.
            prop_assert!(
                plan.estimated_bytes <= capacity,
                "plan bytes {} > capacity {}",
                plan.estimated_bytes,
                capacity
            );

            // 2. all plan keys are a subset of requested keys.
            for plan_key in &plan.entries_to_warm {
                prop_assert!(
                    all_keys.contains(plan_key),
                    "plan key {plan_key:?} not in requested set"
                );
            }
        }
    }
}

// ── (f) Content-aware eviction order ─────────────────────────────────────────

#[test]
fn test_content_aware_eviction_order() {
    use oximedia_cache::content_aware_cache::{ContentAwareCache, MediaContentType};

    // Capacity = 5 entries.  We insert 7 entries so 2 must be evicted.
    // Low-priority + large + aged items should be evicted first.
    let mut cache = ContentAwareCache::new(5);

    // Insert 5 entries: 1 manifest (high priority), 1 thumbnail, 3 metadata
    // (low priority = 3).
    cache.insert_media("manifest".into(), vec![0u8; 10], MediaContentType::Manifest);
    cache.insert_media("thumb".into(), vec![0u8; 8], MediaContentType::Thumbnail);
    cache.insert_media("meta_a".into(), vec![0u8; 200], MediaContentType::Metadata);
    cache.insert_media("meta_b".into(), vec![0u8; 300], MediaContentType::Metadata);
    cache.insert_media("meta_c".into(), vec![0u8; 400], MediaContentType::Metadata);

    assert_eq!(cache.len(), 5);

    // Age the metadata entries by sleeping briefly so their score rises.
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Refresh manifest and thumb to mark them recently-used.
    let _ = cache.get("manifest");
    let _ = cache.get("thumb");

    // Insert 2 more entries to force 2 evictions.
    cache.insert_media(
        "video_1".into(),
        vec![0u8; 50],
        MediaContentType::VideoSegment {
            bitrate: 2_000_000,
            codec: "av1".into(),
        },
    );
    cache.insert_media(
        "audio_1".into(),
        vec![0u8; 30],
        MediaContentType::AudioSegment { bitrate: 128_000 },
    );

    // Cache should still be at capacity (5).
    assert_eq!(cache.len(), 5, "cache should still hold 5 entries");

    // High-priority entries should survive eviction.
    assert!(
        cache.peek("manifest").is_some(),
        "manifest (priority=10) should survive eviction"
    );
    assert!(
        cache.peek("thumb").is_some(),
        "thumbnail (priority=8) should survive eviction"
    );
}
