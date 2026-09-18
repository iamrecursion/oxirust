//! Cache for parsed CPL (Composition Playlist) structures.
//!
//! Avoids redundant XML parses by keying on the file path and using
//! (mtime, file-size) as the cache-validity sentinel. The cache is
//! thread-safe: a single `CplCache` instance may be shared across threads
//! via `Arc<CplCache>`.
//!
//! # Example
//! ```ignore
//! use std::sync::Arc;
//! use oximedia_imf::cpl_cache::CplCache;
//!
//! let cache = Arc::new(CplCache::new());
//! let cpl = cache.get_or_parse(path, || parse_cpl_from_disk(path))?;
//! ```

use crate::cpl_parser::CompositionPlaylist;
use crate::ImfError;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

// ── Internal cache entry ──────────────────────────────────────────────────────

struct CacheEntry {
    value: Arc<CompositionPlaylist>,
    /// File modification time at the time of caching.
    mtime: SystemTime,
    /// File size (bytes) at the time of caching.
    size: u64,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// A parse cache for [`CompositionPlaylist`] values, keyed by filesystem path.
///
/// Each entry is considered valid as long as the file's `(mtime, size)` tuple
/// matches the cached snapshot. A mismatch triggers a full re-parse via the
/// caller-supplied `parser` closure.
pub struct CplCache {
    inner: Mutex<HashMap<PathBuf, CacheEntry>>,
}

impl Default for CplCache {
    fn default() -> Self {
        Self::new()
    }
}

impl CplCache {
    /// Create a new, empty cache.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    /// Return a cached [`CompositionPlaylist`] for `path`, or invoke `parser`
    /// to produce one, cache it, and return it.
    ///
    /// The cache entry is considered *stale* if either the file's modification
    /// time or its size has changed since the last parse.
    ///
    /// # Errors
    ///
    /// Returns an `ImfError` if `parser` returns an error, or if the file
    /// metadata cannot be read.
    pub fn get_or_parse<F>(
        &self,
        path: &Path,
        parser: F,
    ) -> Result<Arc<CompositionPlaylist>, ImfError>
    where
        F: FnOnce() -> Result<CompositionPlaylist, ImfError>,
    {
        // Snapshot the current on-disk metadata first (outside the lock so we
        // don't hold the mutex during I/O).
        let meta = std::fs::metadata(path)?;
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let size = meta.len();

        let map = self
            .inner
            .lock()
            .map_err(|_| ImfError::Other("CplCache mutex poisoned".to_string()))?;

        if let Some(entry) = map.get(path) {
            if entry.mtime == mtime && entry.size == size {
                return Ok(Arc::clone(&entry.value));
            }
        }

        // Cache miss or stale — invoke the parser (drop lock first to avoid
        // holding it during a potentially expensive parse).
        drop(map);

        let cpl = parser()?;
        let value = Arc::new(cpl);

        let mut map = self
            .inner
            .lock()
            .map_err(|_| ImfError::Other("CplCache mutex poisoned".to_string()))?;

        map.insert(
            path.to_path_buf(),
            CacheEntry {
                value: Arc::clone(&value),
                mtime,
                size,
            },
        );

        Ok(value)
    }

    /// Remove any cached entry for `path`, forcing the next `get_or_parse`
    /// call to invoke the parser again.
    pub fn invalidate(&self, path: &Path) {
        if let Ok(mut map) = self.inner.lock() {
            map.remove(path);
        }
    }

    /// Return the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.inner.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// Return `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpl_parser::{CompositionPlaylist, CplResource, CplSegment, CplSequence};
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Write a minimal CPL XML to a temp file and return the path.
    fn write_temp_cpl(id: &str, title: &str) -> tempfile::NamedTempFile {
        let cpl = make_cpl(id, title);
        let xml = cpl.to_xml();
        let mut tmp = tempfile::NamedTempFile::new().expect("temp file");
        tmp.write_all(xml.as_bytes()).expect("write");
        tmp.flush().expect("flush");
        tmp
    }

    fn make_cpl(id: &str, title: &str) -> CompositionPlaylist {
        let mut cpl = CompositionPlaylist::new(id, title, (24, 1));
        let mut seg = CplSegment::new("seg-cache-001");
        let mut seq = CplSequence::new("seq-cache-001", "track-cache-001");
        seq.add_resource(CplResource::simple("tf-cache-001", 480));
        seg.add_sequence(seq);
        cpl.add_segment(seg);
        cpl
    }

    #[test]
    fn test_cache_hit_avoids_reparse() {
        let tmp = write_temp_cpl("cpl-cache-hit-001", "Cache Hit Test");
        let path = tmp.path();

        let cache = CplCache::new();
        let call_count = Arc::new(AtomicUsize::new(0));

        let cc = Arc::clone(&call_count);
        let xml = std::fs::read_to_string(path).expect("read");
        let first = cache
            .get_or_parse(path, || {
                cc.fetch_add(1, Ordering::SeqCst);
                CompositionPlaylist::from_xml(&xml).map_err(|e| ImfError::XmlError(e))
            })
            .expect("first get");

        let cc2 = Arc::clone(&call_count);
        let xml2 = std::fs::read_to_string(path).expect("read");
        let second = cache
            .get_or_parse(path, || {
                cc2.fetch_add(1, Ordering::SeqCst);
                CompositionPlaylist::from_xml(&xml2).map_err(|e| ImfError::XmlError(e))
            })
            .expect("second get");

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            1,
            "parser must be called exactly once on a cache hit"
        );
        assert!(
            Arc::ptr_eq(&first, &second),
            "both calls must return the same Arc"
        );
    }

    #[test]
    fn test_cache_invalidation() {
        let tmp = write_temp_cpl("cpl-cache-inv-001", "Cache Invalidation Test");
        let path = tmp.path();

        let cache = CplCache::new();
        let call_count = Arc::new(AtomicUsize::new(0));

        // First call — populates cache.
        {
            let cc = Arc::clone(&call_count);
            let xml = std::fs::read_to_string(path).expect("read");
            cache
                .get_or_parse(path, || {
                    cc.fetch_add(1, Ordering::SeqCst);
                    CompositionPlaylist::from_xml(&xml).map_err(|e| ImfError::XmlError(e))
                })
                .expect("first get");
        }

        // Invalidate.
        cache.invalidate(path);
        assert!(cache.is_empty(), "cache must be empty after invalidation");

        // Second call after invalidation — must re-invoke parser.
        {
            let cc = Arc::clone(&call_count);
            let xml = std::fs::read_to_string(path).expect("read");
            cache
                .get_or_parse(path, || {
                    cc.fetch_add(1, Ordering::SeqCst);
                    CompositionPlaylist::from_xml(&xml).map_err(|e| ImfError::XmlError(e))
                })
                .expect("second get");
        }

        assert_eq!(
            call_count.load(Ordering::SeqCst),
            2,
            "parser must be called again after invalidation"
        );
    }
}
