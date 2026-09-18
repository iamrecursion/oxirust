//! Cache management module.
//!
//! # Cache management guide: sizing by project scale
//!
//! [`CacheManager`] tracks proxy files against a `max_size` byte budget and
//! evicts entries (real, in-memory bookkeeping — see
//! [`manager::CacheManager::add`] / `evict`) under one of three strategies
//! ([`CacheStrategy::Lru`], [`CacheStrategy::Lfu`], [`CacheStrategy::Fifo`])
//! once that budget would be exceeded. Sizing `max_size` correctly for your
//! project scale avoids two failure modes: too small and the cache
//! thrashes (constantly evicting and re-generating proxies editors are
//! actively using); too large and it never reclaims space until the disk
//! actually fills up.
//!
//! Rough starting points (assuming quarter-resolution H.264 proxies at
//! roughly 2 Mbps — see [`crate::Quality::bitrate_1080p`] for the bitrate
//! this crate recommends per quality tier):
//!
//! | Project scale | Example | Suggested `max_size` | Strategy |
//! |---|---|---|---|
//! | Solo editor / short-form | a few dozen clips, < 2 hours of footage | 5-10 GB | [`CacheStrategy::Lru`] — most recently cut scenes stay resident |
//! | Small team / doc project | hundreds of clips, multi-day shoot | 50-100 GB | [`CacheStrategy::Lru`] |
//! | Episodic / multi-editor | thousands of clips, ongoing series | 250 GB-1 TB | [`CacheStrategy::Lfu`] — protects frequently-reused b-roll/inserts from eviction even if not recently touched |
//! | Archive / rarely-revisited footage | tens of thousands of clips | small cache, aggressive eviction | [`CacheStrategy::Fifo`] combined with a short [`crate::proxy_aging::AgingPolicy`] TTL |
//!
//! For lifecycle policy *beyond* the in-memory cache budget — i.e. how long
//! an on-disk proxy should live before being archived or deleted regardless
//! of cache pressure — see [`crate::proxy_aging::AgingPolicy`] and
//! [`crate::proxy_aging::AgingManager`], which implement real
//! active/idle/stale/expired/archived stage transitions driven by
//! [`crate::proxy_aging::AgingPolicy::idle_after_days`] /
//! `stale_after_days` / `expire_after_days`. `AgingPolicy::strict()` (3/14/30
//! days, auto-archive + auto-delete) suits space-constrained shared storage;
//! `AgingPolicy::relaxed()` (30/180/365 days, no auto-delete) suits archival
//! workflows where footage may be revisited long after a project wraps.
//!
//! Note: [`cleanup::CacheCleanup`] (disk-level cleanup by policy) is
//! currently a placeholder that reports zero files removed / zero bytes
//! freed — actual on-disk reclamation today happens through
//! [`crate::proxy_aging::AgingManager::sweep`]'s `bytes_reclaimed` accounting
//! plus the caller deleting files in stages reported by
//! [`crate::proxy_aging::AgingManager::records_in_stage`].

pub mod cleanup;
pub mod manager;
pub mod strategy;
pub mod warmer;

pub use cleanup::{CacheCleanup, CacheStats, CleanupPolicy, CleanupResult};
pub use manager::CacheManager;
pub use strategy::CacheStrategy;
pub use warmer::{ProxyCacheWarmer, WarmCandidate, WarmingConfig, WarmingResult};
