//! Key-Value cache for efficient autoregressive inference.
//!
//! During autoregressive generation (e.g., text generation), transformers repeatedly
//! compute attention over the same prefix tokens. KV-caching stores the key and value
//! projections from previous steps, avoiding redundant computation.
//!
//! ## Performance Impact
//!
//! Without KV-cache:
//! ```text
//! Step 1: Compute attention for token 1
//! Step 2: Compute attention for tokens 1,2    (redundant!)
//! Step 3: Compute attention for tokens 1,2,3  (redundant!)
//! ```
//!
//! With KV-cache:
//! ```text
//! Step 1: Compute K,V for token 1, cache them
//! Step 2: Compute K,V for token 2, append to cache
//! Step 3: Compute K,V for token 3, append to cache
//! ```
//!
//! **Speedup**: ~10-100x for long sequences!
//!
//! ## Usage
//!
//! ```rust,no_run
//! use tensorlogic_trustformers::KVCache;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create cache for 12-layer model with 12 heads
//! let mut cache = KVCache::new(12, 12, 64);
//!
//! // During generation, update cache for each layer
//! # let (new_keys, new_values) = (vec![], vec![]);
//! cache.update_layer(0, new_keys, new_values)?;
//!
//! // Retrieve cached keys/values for attention
//! let (cached_keys, cached_values) = cache.get_layer(0)?;
//! # Ok(())
//! # }
//! ```

mod cached_attention;
mod config;
mod position;
mod simple_cache;
mod stats;

pub use cached_attention::{CachedAttention, CachedAttentionError};
pub use config::{CacheEntry, CacheStats, KVCache, KVCacheConfig};
pub use position::{PositionError, RelativePositionBias, RotaryPositionEmbedding};
pub use simple_cache::{KvCache, KvCacheError};
pub use stats::InferenceStats;
