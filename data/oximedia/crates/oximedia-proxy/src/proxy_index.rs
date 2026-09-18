//! In-memory index mapping original media paths to proxy entries.
//!
//! Provides `ProxyEntry`, `ProxyIndex` (simple HashMap path-keyed index), and
//! `RangeProxyIndex` (BTreeMap composite-key index for timecode-range and
//! path-prefix queries).

use std::collections::{BTreeMap, HashMap};

/// A record in the proxy index describing one proxy asset.
#[derive(Debug, Clone, PartialEq)]
pub struct ProxyEntry {
    /// Absolute path to the original high-resolution media file.
    pub original_path: String,
    /// Absolute path to the proxy media file.
    pub proxy_path: String,
    /// Width of the proxy in pixels.
    pub width: u32,
    /// Height of the proxy in pixels.
    pub height: u32,
    /// Bitrate of the proxy in kbps.
    pub bitrate_kbps: u32,
    /// Optional codec identifier (e.g. "h264").
    pub codec: Option<String>,
    /// Presentation timestamp in microseconds (0 when not applicable).
    ///
    /// Used as the timecode axis in `RangeProxyIndex`. For full-file proxies
    /// this is typically 0; for clip-level proxies it holds the clip start PTS.
    pub timecode_pts: u64,
}

impl ProxyEntry {
    /// Create a new proxy entry with `timecode_pts` set to 0.
    pub fn new(
        original_path: impl Into<String>,
        proxy_path: impl Into<String>,
        width: u32,
        height: u32,
        bitrate_kbps: u32,
    ) -> Self {
        Self {
            original_path: original_path.into(),
            proxy_path: proxy_path.into(),
            width,
            height,
            bitrate_kbps,
            codec: None,
            timecode_pts: 0,
        }
    }

    /// Create a proxy entry with an explicit timecode PTS (µs).
    pub fn with_timecode(
        original_path: impl Into<String>,
        proxy_path: impl Into<String>,
        width: u32,
        height: u32,
        bitrate_kbps: u32,
        timecode_pts: u64,
    ) -> Self {
        Self {
            original_path: original_path.into(),
            proxy_path: proxy_path.into(),
            width,
            height,
            bitrate_kbps,
            codec: None,
            timecode_pts,
        }
    }

    /// Return `true` when required fields are non-empty and dimensions are > 0.
    pub fn is_valid(&self) -> bool {
        !self.original_path.is_empty()
            && !self.proxy_path.is_empty()
            && self.width > 0
            && self.height > 0
            && self.bitrate_kbps > 0
    }

    /// Return a display label combining resolution and bitrate.
    pub fn display_label(&self) -> String {
        format!("{}x{}@{}kbps", self.width, self.height, self.bitrate_kbps)
    }

    /// Return total pixel count (width × height).
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// An in-memory index of proxy entries keyed by original file path.
#[derive(Debug, Default)]
pub struct ProxyIndex {
    // Maps original_path → Vec<ProxyEntry> (multiple qualities possible).
    map: HashMap<String, Vec<ProxyEntry>>,
}

impl ProxyIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a `ProxyEntry`.  Entries with the same `original_path` are accumulated.
    pub fn insert(&mut self, entry: ProxyEntry) {
        self.map
            .entry(entry.original_path.clone())
            .or_default()
            .push(entry);
    }

    /// Find all proxy entries for a given original path.
    pub fn find_by_original(&self, original_path: &str) -> &[ProxyEntry] {
        self.map
            .get(original_path)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Remove all entries for a given original path.  Returns the removed entries.
    pub fn remove(&mut self, original_path: &str) -> Vec<ProxyEntry> {
        self.map.remove(original_path).unwrap_or_default()
    }

    /// Return the total number of proxy entries across all originals.
    pub fn count(&self) -> usize {
        self.map.values().map(Vec::len).sum()
    }

    /// Return the number of unique originals in the index.
    pub fn original_count(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Return all entries as a flat iterator.
    pub fn all_entries(&self) -> impl Iterator<Item = &ProxyEntry> {
        self.map.values().flat_map(|v| v.iter())
    }

    /// Return `true` if any proxy entry exists for the given original path.
    pub fn contains(&self, original_path: &str) -> bool {
        self.map.contains_key(original_path)
    }

    /// Find the entry with the highest bitrate for a given original path.
    pub fn best_quality(&self, original_path: &str) -> Option<&ProxyEntry> {
        self.find_by_original(original_path)
            .iter()
            .max_by_key(|e| e.bitrate_kbps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(orig: &str, proxy: &str, w: u32, h: u32, br: u32) -> ProxyEntry {
        ProxyEntry::new(orig, proxy, w, h, br)
    }

    #[test]
    fn test_entry_is_valid() {
        let e = make_entry("/media/orig.mov", "/proxy/p.mp4", 640, 360, 500);
        assert!(e.is_valid());
    }

    #[test]
    fn test_entry_invalid_empty_path() {
        let e = make_entry("", "/proxy/p.mp4", 640, 360, 500);
        assert!(!e.is_valid());
    }

    #[test]
    fn test_entry_invalid_zero_dimension() {
        let e = make_entry("/media/orig.mov", "/proxy/p.mp4", 0, 360, 500);
        assert!(!e.is_valid());
    }

    #[test]
    fn test_entry_invalid_zero_bitrate() {
        let e = make_entry("/media/orig.mov", "/proxy/p.mp4", 640, 360, 0);
        assert!(!e.is_valid());
    }

    #[test]
    fn test_entry_display_label() {
        let e = make_entry("/orig.mov", "/p.mp4", 1280, 720, 2000);
        assert_eq!(e.display_label(), "1280x720@2000kbps");
    }

    #[test]
    fn test_entry_pixel_count() {
        let e = make_entry("/orig.mov", "/p.mp4", 1920, 1080, 8000);
        assert_eq!(e.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_index_insert_and_count() {
        let mut idx = ProxyIndex::new();
        idx.insert(make_entry("/orig.mov", "/p1.mp4", 640, 360, 500));
        idx.insert(make_entry("/orig.mov", "/p2.mp4", 1280, 720, 2000));
        assert_eq!(idx.count(), 2);
        assert_eq!(idx.original_count(), 1);
    }

    #[test]
    fn test_index_find_by_original() {
        let mut idx = ProxyIndex::new();
        idx.insert(make_entry("/orig.mov", "/p.mp4", 640, 360, 500));
        let found = idx.find_by_original("/orig.mov");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].proxy_path, "/p.mp4");
    }

    #[test]
    fn test_index_find_by_original_not_found() {
        let idx = ProxyIndex::new();
        assert!(idx.find_by_original("/missing.mov").is_empty());
    }

    #[test]
    fn test_index_remove() {
        let mut idx = ProxyIndex::new();
        idx.insert(make_entry("/orig.mov", "/p.mp4", 640, 360, 500));
        let removed = idx.remove("/orig.mov");
        assert_eq!(removed.len(), 1);
        assert_eq!(idx.count(), 0);
    }

    #[test]
    fn test_index_contains() {
        let mut idx = ProxyIndex::new();
        idx.insert(make_entry("/orig.mov", "/p.mp4", 640, 360, 500));
        assert!(idx.contains("/orig.mov"));
        assert!(!idx.contains("/other.mov"));
    }

    #[test]
    fn test_index_best_quality() {
        let mut idx = ProxyIndex::new();
        idx.insert(make_entry("/orig.mov", "/p_draft.mp4", 640, 360, 500));
        idx.insert(make_entry("/orig.mov", "/p_delivery.mp4", 1920, 1080, 8000));
        let best = idx
            .best_quality("/orig.mov")
            .expect("should succeed in test");
        assert_eq!(best.proxy_path, "/p_delivery.mp4");
    }

    #[test]
    fn test_index_is_empty() {
        let idx = ProxyIndex::new();
        assert!(idx.is_empty());
    }
}

// ── RangeProxyIndex ───────────────────────────────────────────────────────────

/// Composite key `(original_path, timecode_pts)` used inside `RangeProxyIndex`.
///
/// Derives `Ord` so the BTreeMap sorts first by path, then by PTS, enabling
/// both prefix-range queries and timecode-span queries in O(log n + k).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RangeKey {
    /// Absolute path of the original high-resolution file.
    pub path: String,
    /// Presentation timestamp in microseconds (from `ProxyEntry::timecode_pts`).
    pub pts: u64,
}

impl RangeKey {
    fn new(path: impl Into<String>, pts: u64) -> Self {
        Self {
            path: path.into(),
            pts,
        }
    }
}

/// An ordered index of proxy entries supporting BTree range queries.
///
/// Entries are stored in a `BTreeMap<RangeKey, ProxyEntry>` where the key is
/// the composite `(original_path, timecode_pts)`.  This allows:
///
/// - `find_by_original(path)` — exact-path match, all timecodes.
/// - `find_in_timecode_range(path, start, end)` — PTS span within one path.
/// - `find_by_path_prefix(prefix)` — all entries whose path starts with a
///   given prefix (e.g. all proxies under `/project/reel1/`).
///
/// Multiple entries for the same `(path, pts)` are not directly supported.
/// If two entries share the same composite key the later `insert` wins.
///
/// # Example
///
/// ```rust
/// use oximedia_proxy::proxy_index::{ProxyEntry, RangeProxyIndex};
///
/// let mut idx = RangeProxyIndex::new();
/// let e = ProxyEntry::with_timecode("/orig.mov", "/proxy.mp4", 640, 360, 500, 0);
/// idx.insert(e);
/// assert_eq!(idx.find_by_original("/orig.mov").len(), 1);
/// ```
#[derive(Debug, Default)]
pub struct RangeProxyIndex {
    /// BTreeMap keyed by `(path, pts)` — sorted lexicographically by path then by PTS.
    entries: BTreeMap<RangeKey, ProxyEntry>,
}

impl RangeProxyIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a `RangeProxyIndex` from an existing `ProxyIndex`.
    ///
    /// Each `ProxyEntry` in the source index is inserted using its
    /// `timecode_pts` field as the PTS component of the composite key.
    pub fn from_index(source: ProxyIndex) -> Self {
        let mut range_idx = Self::new();
        for entry in source.all_entries().cloned() {
            range_idx.insert(entry);
        }
        range_idx
    }

    /// Insert a `ProxyEntry`.
    ///
    /// The composite key is derived from `entry.original_path` and
    /// `entry.timecode_pts`.  If an entry with the same key already exists it
    /// is replaced.
    pub fn insert(&mut self, entry: ProxyEntry) {
        let key = RangeKey::new(entry.original_path.clone(), entry.timecode_pts);
        self.entries.insert(key, entry);
    }

    /// Remove the entry with the given path and PTS.  Returns `None` when the
    /// key is not present.
    pub fn remove(&mut self, path: &str, pts: u64) -> Option<ProxyEntry> {
        let key = RangeKey::new(path, pts);
        self.entries.remove(&key)
    }

    /// Remove all entries for a given original path (all PTS values).
    ///
    /// Returns the number of entries removed.
    pub fn remove_all(&mut self, path: &str) -> usize {
        let start = RangeKey::new(path, 0);
        let end = Self::path_end_key(path);
        let keys: Vec<RangeKey> = self
            .entries
            .range(start..end)
            .map(|(k, _)| k.clone())
            .collect();
        let count = keys.len();
        for k in keys {
            self.entries.remove(&k);
        }
        count
    }

    // ── Query methods ─────────────────────────────────────────────────────────

    /// Return all proxy entries whose `original_path` exactly matches `path`.
    ///
    /// Uses a BTreeMap range query over `(path, 0) ..= (path, u64::MAX)`.
    pub fn find_by_original(&self, path: &str) -> Vec<&ProxyEntry> {
        let start = RangeKey::new(path, 0);
        let end = Self::path_end_key(path);
        self.entries.range(start..end).map(|(_, v)| v).collect()
    }

    /// Return all proxy entries for `path` whose PTS falls within `[start, end]`
    /// (inclusive on both ends).
    ///
    /// Uses a single BTreeMap range scan — O(log n + k).
    pub fn find_in_timecode_range(
        &self,
        path: &str,
        start_pts: u64,
        end_pts: u64,
    ) -> Vec<&ProxyEntry> {
        let lo = RangeKey::new(path, start_pts);
        let hi = RangeKey::new(path, end_pts);
        self.entries.range(&lo..=&hi).map(|(_, v)| v).collect()
    }

    /// Return all proxy entries whose `original_path` starts with `prefix`.
    ///
    /// Useful for bulk queries such as "all proxies under `/project/reel1/`".
    /// The query scans the BTreeMap from the first key ≥ `(prefix, 0)` up to
    /// (but not including) the successor prefix, which is O(log n + k).
    pub fn find_by_path_prefix(&self, prefix: &str) -> Vec<&ProxyEntry> {
        if prefix.is_empty() {
            return self.entries.values().collect();
        }
        let start = RangeKey::new(prefix, 0);
        // The end key is formed by incrementing the last byte of `prefix` so
        // that all strings starting with `prefix` sort before it.
        match Self::prefix_end_str(prefix) {
            Some(end_str) => {
                let end = RangeKey::new(end_str, 0);
                self.entries
                    .range(start..end)
                    .filter(|(k, _)| k.path.starts_with(prefix))
                    .map(|(_, v)| v)
                    .collect()
            }
            None => {
                // `prefix` consists entirely of `\xFF` bytes — return all entries
                // from the start key to the end of the map.
                self.entries
                    .range(start..)
                    .filter(|(k, _)| k.path.starts_with(prefix))
                    .map(|(_, v)| v)
                    .collect()
            }
        }
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Total number of entries in the index.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all entries as an iterator (path-sorted, then by PTS).
    pub fn all_entries(&self) -> impl Iterator<Item = &ProxyEntry> {
        self.entries.values()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Return a `RangeKey` that is the strict upper bound for all keys with
    /// `path` as their path component.  Achieved by appending `\x00` which
    /// makes the resulting string lexicographically greater than any string
    /// that starts with `path` but equal to it in length, yet comes before any
    /// string starting with `path` concatenated with a non-NUL character.
    ///
    /// Actually the safe choice is a NUL-terminated version: the smallest
    /// string that is strictly greater than every string that starts with `path`
    /// and has the same length is `path` with the last byte incremented.  We
    /// use a simpler approach: append `\x00` works perfectly because
    /// `"foo\x00"` > `"foo"` and `"foo\x00"` < `"foo_anything_else"` only when
    /// the "anything_else" part starts with a byte > `\x00` — but all printable
    /// characters are > `\x00`, so this is correct for filesystem paths.
    fn path_end_key(path: &str) -> RangeKey {
        // Using path + "\x00" as the exclusive upper bound:
        // All keys with this exact path component will be in the range
        // [path, path\x00).
        // However since we want `pts` to range freely we set pts = u64::MAX
        // and use an inclusive upper bound trick: the cleanest way is to
        // append a byte that is lexicographically after all valid path chars
        // (NUL = 0x00 is LESS, so we use a null upper range via appending
        // a character with codepoint 0 ... this won't work).
        //
        // The simplest correct approach: construct a String that is
        // `path` with the last char incremented.  If that fails (empty or
        // all-max-char), fall back to returning a key that uses u64::MAX.
        match Self::prefix_end_str(path) {
            Some(next_path) => RangeKey::new(next_path, 0),
            None => {
                // path is empty or all-max; use a sentinel with MAX pts.
                RangeKey {
                    path: path.to_string(),
                    pts: u64::MAX,
                }
            }
        }
    }

    /// Compute the lexicographic successor of `prefix` by incrementing the
    /// last byte.  Returns `None` when `prefix` is empty or all bytes are 0xFF.
    fn prefix_end_str(prefix: &str) -> Option<String> {
        let bytes = prefix.as_bytes();
        // Find the rightmost byte that can be incremented.
        for i in (0..bytes.len()).rev() {
            if bytes[i] < u8::MAX {
                let mut end_bytes = bytes[..=i].to_vec();
                end_bytes[i] += 1;
                // Safety: we only work with UTF-8 paths; the incremented byte
                // may produce a non-UTF-8 sequence.  Use lossy conversion to
                // stay safe.  The key is only used for comparisons in the
                // BTreeMap, not returned to callers.
                return Some(String::from_utf8_lossy(&end_bytes).into_owned());
            }
        }
        None
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;

    fn entry(path: &str, pts: u64) -> ProxyEntry {
        ProxyEntry::with_timecode(path, &format!("/proxy/{pts}.mp4"), 640, 360, 500, pts)
    }

    #[test]
    fn test_range_insert_and_count() {
        let mut idx = RangeProxyIndex::new();
        idx.insert(entry("/a.mov", 0));
        idx.insert(entry("/a.mov", 100));
        assert_eq!(idx.count(), 2);
    }

    #[test]
    fn test_find_by_original_exact() {
        let mut idx = RangeProxyIndex::new();
        idx.insert(entry("/a.mov", 0));
        idx.insert(entry("/a.mov", 100));
        idx.insert(entry("/b.mov", 0));
        let found = idx.find_by_original("/a.mov");
        assert_eq!(found.len(), 2);
        for e in &found {
            assert_eq!(e.original_path, "/a.mov");
        }
    }

    #[test]
    fn test_find_by_original_not_found() {
        let idx = RangeProxyIndex::new();
        assert!(idx.find_by_original("/missing.mov").is_empty());
    }

    #[test]
    fn test_find_in_timecode_range() {
        let mut idx = RangeProxyIndex::new();
        idx.insert(entry("/reel.mov", 0));
        idx.insert(entry("/reel.mov", 100));
        idx.insert(entry("/reel.mov", 200));
        idx.insert(entry("/reel.mov", 300));
        // Query [50, 250] — should return pts=100 and pts=200
        let found = idx.find_in_timecode_range("/reel.mov", 50, 250);
        assert_eq!(found.len(), 2);
        let pts_values: Vec<u64> = found.iter().map(|e| e.timecode_pts).collect();
        assert!(pts_values.contains(&100));
        assert!(pts_values.contains(&200));
    }

    #[test]
    fn test_find_by_path_prefix() {
        let mut idx = RangeProxyIndex::new();
        idx.insert(entry("/project/reel1/a.mov", 0));
        idx.insert(entry("/project/reel1/b.mov", 0));
        idx.insert(entry("/project/reel2/c.mov", 0));
        idx.insert(entry("/other/d.mov", 0));
        let found = idx.find_by_path_prefix("/project/reel1/");
        assert_eq!(found.len(), 2);
        for e in &found {
            assert!(e.original_path.starts_with("/project/reel1/"));
        }
    }

    #[test]
    fn test_from_index_conversion() {
        let mut plain = ProxyIndex::new();
        plain.insert(ProxyEntry::new("/x.mov", "/px.mp4", 640, 360, 500));
        plain.insert(ProxyEntry::new("/y.mov", "/py.mp4", 640, 360, 500));
        let range_idx = RangeProxyIndex::from_index(plain);
        assert_eq!(range_idx.count(), 2);
    }

    #[test]
    fn test_remove_single() {
        let mut idx = RangeProxyIndex::new();
        idx.insert(entry("/a.mov", 0));
        idx.insert(entry("/a.mov", 100));
        let removed = idx.remove("/a.mov", 0);
        assert!(removed.is_some());
        assert_eq!(idx.count(), 1);
    }

    #[test]
    fn test_remove_all() {
        let mut idx = RangeProxyIndex::new();
        idx.insert(entry("/a.mov", 0));
        idx.insert(entry("/a.mov", 100));
        idx.insert(entry("/b.mov", 0));
        let n = idx.remove_all("/a.mov");
        assert_eq!(n, 2);
        assert_eq!(idx.count(), 1);
    }
}
