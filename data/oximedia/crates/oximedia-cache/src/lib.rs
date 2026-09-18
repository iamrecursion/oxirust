//! High-performance caching infrastructure for OxiMedia.
//!
//! `oximedia-cache` provides sixteen complementary caching primitives:
//!
//! - [`lru_cache`] — arena-backed O(1) LRU cache with TTL expiration,
//!   entry pinning, hit/miss/eviction stats, and capacity resize
//! - [`tiered_cache`] — multi-tier (L1 memory → L2 memory → disk) cache with
//!   file-backed disk tier, adaptive promotion thresholds, entry compression
//!   for L2+ tiers, pluggable eviction (LRU/LFU/FIFO/Random/TinyLFU), and
//!   automatic promotion across tiers
//! - [`cache_warming`] — predictive warming via access-pattern analysis,
//!   exponential inter-arrival EMA, auto-correlation periodicity detection, and
//!   score-ranked warmup plans
//! - [`bloom_filter`] — probabilistic membership filter (standard, counting
//!   with deletion, and scalable auto-growing variant) using FNV-1a double
//!   hashing
//! - [`distributed_cache`] — consistent-hash ring with configurable virtual
//!   nodes, per-node client, quorum replication factor, and cluster coordinator
//! - [`eviction_policies`] — standalone LFU tracker, frequency counter with
//!   decay, TinyLFU admission gate (optimized hot path), and ARC ghost-list
//!   tracker
//! - [`content_aware_cache`] — media-type-aware cache with configurable weight
//!   factors per media type; scores eviction candidates by recency × priority ×
//!   size
//! - [`write_behind_cache`] — write-back cache with dirty tracking,
//!   flush-by-age, mark-clean, and backing-store abstraction
//! - [`two_queue`] — 2Q scan-resistant eviction policy (A1in FIFO + Am LRU
//!   + A1out ghost list) as alternative to LRU
//! - [`cache_metrics`] — atomic hit/miss/eviction counters with latency
//!   percentile tracking and `Arc`-shareable snapshots
//! - [`prefetch`] — sequential media segment pre-loading based on access
//!   patterns with pluggable loader and pending queue
//! - [`sharded_lru`] — concurrent LRU cache sharded across N independent
//!   `Mutex<LruCache>` instances to reduce lock contention
//! - [`cache_partitioning`] — isolate cache space per tenant, stream, or
//!   workload with independent byte-level budgets and LRU eviction
//! - [`cache_serialization`] — persist the cache state to disk on shutdown and
//!   restore on startup using a zero-copy binary format
//! - [`slab_allocator`] — fixed-size slab allocator for cache entries to
//!   reduce heap fragmentation in long-running processes
//!
//! # Quick start
//!
//! ```rust
//! use oximedia_cache::lru_cache::LruCache;
//!
//! let mut cache: LruCache<&str, Vec<u8>> = LruCache::new(128);
//! cache.insert("frame-001", vec![0u8; 4096], 4096);
//! assert!(cache.get(&"frame-001").is_some());
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

pub mod adaptive;
pub mod admission_filter;
pub mod bloom_filter;
pub mod cache_metrics;
pub mod cache_partitioning;
pub mod cache_serialization;
pub mod cache_warming;
pub mod content_aware_cache;
pub mod distributed_cache;
pub mod eviction;
pub mod eviction_policies;
pub mod key_norm;
pub mod lru_cache;
pub mod negative;
pub mod prefetch;
pub mod segment_cache;
pub mod sharded_lru;
pub mod slab_allocator;
pub mod stats;
pub mod tier_compressor;
pub mod tiered_cache;
pub mod ttl_cache;
pub mod two_queue;
pub mod weighted_cache;
pub mod write_behind_cache;
pub mod write_through;
