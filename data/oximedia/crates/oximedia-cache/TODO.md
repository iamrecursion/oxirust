# oximedia-cache TODO

## Current Status
- 7 modules: `lru_cache`, `tiered_cache`, `cache_warming`, `bloom_filter`, `distributed_cache`, `eviction_policies`, `content_aware_cache`
- Zero external dependencies beyond `thiserror` (pure Rust, minimal footprint)
- LRU cache with O(1) operations, tiered multi-level cache with pluggable eviction (LRU/LFU/FIFO/Random/TinyLFU)
- Distributed cache with consistent-hash ring and quorum replication

## Enhancements
- [x] Add TTL (time-to-live) expiration support to `lru_cache::LruCache` entries
- [x] Extend `tiered_cache` with async disk tier using file-backed storage (not just simulated) (verified: tiered_cache.rs:52-259)
- [x] Add cache entry pinning in `lru_cache` to prevent eviction of critical items
- [x] Implement adaptive tier promotion thresholds in `tiered_cache` based on access frequency (Wave 13: P² quantile estimator, tiered_cache.rs)
- [x] Extend `bloom_filter` with scalable Bloom filter variant that grows as elements are added
- [x] Add `distributed_cache` virtual node support for better hash ring distribution (verified 2026-05-16; src/distributed_cache.rs:50 virtual_nodes_per_node, ConsistentHash ring)
- [x] Improve `content_aware_cache` scoring with configurable weight factors per media type (Wave 13: ScoringWeights, content_aware_cache.rs)
- [x] Add cache entry compression option for `tiered_cache` L2+ tiers to reduce memory footprint (verified 2026-05-16; src/tiered_cache.rs:65 TierConfig.compress, run-length encoding:133)

## New Features
- [x] Add `write_behind_cache` module with write-back to origin on eviction
- [x] Implement `cache_metrics` module with hit rate, miss rate, latency percentiles, and eviction counters (verified 2026-05-16; src/cache_metrics.rs)
- [x] Add `prefetch` module that pre-loads sequential media segments based on access pattern (verified 2026-05-16; src/prefetch.rs)
- [x] Implement `two_queue` (2Q) eviction policy for scan-resistant caching
- [x] Add `slab_allocator` module for cache entries to reduce fragmentation in long-running processes (verified 2026-05-16; src/slab_allocator.rs)
- [x] Implement `cache_partitioning` module for isolating cache space per tenant or stream (verified 2026-05-16; src/cache_partitioning.rs)
- [x] Add `cache_serialization` module for persisting cache state to disk on shutdown and restoring on startup (verified 2026-05-16; src/cache_serialization.rs)

## Performance
- [x] Replace HashMap with a more cache-friendly data structure in `lru_cache` (e.g., Swiss table) (Wave 28: swapped `std::collections::HashMap` → `hashbrown::HashMap` (Swiss table, foldhash default hasher) in lru_cache.rs:8 — byte-identical public API + list-driven LRU semantics, all 529 tests pass unchanged; added benches/lru_bench.rs put/get_hit/mixed at 100K)
- [x] Add SIMD-accelerated hash computation for `bloom_filter` FNV-1a double hashing (Wave 13: hash_batch_fnv1a, AVX2+NEON+scalar, bloom_filter.rs)
- [x] Implement sharded LRU cache for concurrent access without global lock contention (verified 2026-05-16; src/sharded_lru.rs)
- [x] Use arena allocation for `tiered_cache` tier entries to reduce allocator pressure (Wave 13: BumpArena, TierEntry::Arena, tiered_cache.rs)
- [x] Benchmark `eviction_policies::TinyLFU` admission gate overhead and optimize hot path (Wave 13: benches/tinylfu.rs, inlined CMS, eviction_policies.rs)

## Testing
- [x] Add concurrent access stress test for `lru_cache` with multiple reader/writer threads (Wave 13: test_lru_concurrent_stress)
- [x] Test `tiered_cache` promotion/demotion correctness under mixed workload patterns (Wave 13: test_tiered_promote_demote_correctness)
- [x] Add false positive rate validation test for `bloom_filter` against theoretical bounds (Wave 13: test_bloom_fpr_validation)
- [x] Test `distributed_cache` rebalancing when nodes join/leave the consistent hash ring (Wave 13: test_distributed_cache_rebalance)
- [x] Add property-based test for `cache_warming` ensuring warmup plan respects capacity limits (Wave 13: warmup_plan_respects_capacity proptest)
- [x] Test `content_aware_cache` eviction ordering with heterogeneous entry sizes and priorities (Wave 13: test_content_aware_eviction_order)

## Documentation
- [ ] Add capacity planning guide: how to size LRU/tiered caches for different media workloads
- [ ] Document eviction policy tradeoffs (LRU vs LFU vs TinyLFU vs ARC) with use-case recommendations
- [ ] Add examples showing `tiered_cache` configuration for L1 (memory) + L2 (disk) setup
