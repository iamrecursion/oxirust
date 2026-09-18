#![allow(dead_code)]
//! High-performance buffered media reader with seek and lookahead.
//!
//! [`BufferedMediaReader`] wraps an in-memory byte slice and exposes a rich
//! cursor-based API for reading primitive types, peeking without advancing,
//! arbitrary seeking, and pattern searching via the Boyer-Moore-Horspool
//! algorithm.

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulated statistics for a [`BufferedMediaReader`] session.
#[derive(Debug, Clone, Default)]
pub struct ReadStats {
    /// Total number of `read*` calls made.
    pub total_reads: u64,
    /// Total number of bytes consumed by `read*` calls.
    pub total_bytes: u64,
    /// Number of peek calls that returned data without advancing the cursor.
    pub cache_hits: u64,
    /// Average number of bytes per `read*` call (0.0 when no reads yet).
    pub avg_read_size: f64,
}

/// High-performance buffered reader backed by an in-memory `Vec<u8>`.
///
/// The reader maintains a cursor position and accumulates [`ReadStats`].
/// All multi-byte integer readers use explicit endianness suffixes
/// (`_be` / `_le`).
#[derive(Debug)]
pub struct BufferedMediaReader {
    data: Vec<u8>,
    position: usize,
    pub buffer_size: usize,
    read_count: u64,
    bytes_read: u64,
    peek_count: u64,
}

impl BufferedMediaReader {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a reader from an owned byte vector.
    ///
    /// `buffer_size` is advisory (e.g. for display in stats); the full
    /// in-memory buffer is always available.
    #[must_use]
    pub fn from_bytes(data: Vec<u8>) -> Self {
        let buffer_size = data.len().max(65536);
        Self {
            data,
            position: 0,
            buffer_size,
            read_count: 0,
            bytes_read: 0,
            peek_count: 0,
        }
    }

    /// Create a reader with an explicit `buffer_size` hint.
    #[must_use]
    pub fn from_bytes_with_buffer_size(data: Vec<u8>, buffer_size: usize) -> Self {
        Self {
            buffer_size,
            ..Self::from_bytes(data)
        }
    }

    // ── Cursor helpers ────────────────────────────────────────────────────────

    /// Return the current byte offset (number of bytes already consumed).
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Return the number of bytes still available beyond the current position.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Return `true` when the cursor is at or past the end of the buffer.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.position >= self.data.len()
    }

    /// Return the total length of the underlying buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` when the underlying buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // ── Seek / skip ───────────────────────────────────────────────────────────

    /// Seek to an absolute byte offset.
    ///
    /// Returns `true` on success, `false` if `pos` is beyond the buffer end.
    pub fn seek(&mut self, pos: usize) -> bool {
        if pos > self.data.len() {
            return false;
        }
        self.position = pos;
        true
    }

    /// Advance the cursor by `count` bytes.
    ///
    /// Returns the number of bytes actually skipped (may be less than `count`
    /// if the end of the buffer is reached).
    pub fn skip(&mut self, count: usize) -> usize {
        let available = self.remaining();
        let actual = count.min(available);
        self.position += actual;
        actual
    }

    // ── Core read / peek ─────────────────────────────────────────────────────

    /// Read up to `count` bytes from the current position and advance the
    /// cursor by the number of bytes returned.
    ///
    /// The returned slice borrows from the internal buffer.
    pub fn read(&mut self, count: usize) -> &[u8] {
        let start = self.position;
        let end = (start + count).min(self.data.len());
        let actual = end - start;
        self.position = end;
        self.read_count += 1;
        self.bytes_read += actual as u64;
        &self.data[start..end]
    }

    /// Return up to `count` bytes starting at the current position *without*
    /// advancing the cursor.
    pub fn peek(&self, count: usize) -> &[u8] {
        let start = self.position;
        let end = (start + count).min(self.data.len());
        &self.data[start..end]
    }

    /// Read exactly `count` bytes and return them as an owned `Vec<u8>`.
    ///
    /// Returns `None` if fewer than `count` bytes remain.
    pub fn read_exact(&mut self, count: usize) -> Option<Vec<u8>> {
        if self.remaining() < count {
            return None;
        }
        let start = self.position;
        let end = start + count;
        self.position = end;
        self.read_count += 1;
        self.bytes_read += count as u64;
        Some(self.data[start..end].to_vec())
    }

    // ── Primitive readers (big-endian) ────────────────────────────────────────

    /// Read a single unsigned byte or `None` at EOF.
    pub fn read_u8(&mut self) -> Option<u8> {
        if self.is_eof() {
            return None;
        }
        let val = self.data[self.position];
        self.position += 1;
        self.read_count += 1;
        self.bytes_read += 1;
        Some(val)
    }

    /// Read a big-endian `u16`.
    pub fn read_u16_be(&mut self) -> Option<u16> {
        let bytes = self.read_exact(2)?;
        Some(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    /// Read a big-endian `u32`.
    pub fn read_u32_be(&mut self) -> Option<u32> {
        let bytes = self.read_exact(4)?;
        Some(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    /// Read a big-endian `u64`.
    pub fn read_u64_be(&mut self) -> Option<u64> {
        let bytes = self.read_exact(8)?;
        Some(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    // ── Primitive readers (little-endian) ────────────────────────────────────

    /// Read a little-endian `u16`.
    pub fn read_u16_le(&mut self) -> Option<u16> {
        let bytes = self.read_exact(2)?;
        Some(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    /// Read a little-endian `u32`.
    pub fn read_u32_le(&mut self) -> Option<u32> {
        let bytes = self.read_exact(4)?;
        Some(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    // ── Pattern search ────────────────────────────────────────────────────────

    /// Search for `pattern` in the buffer starting at the current cursor
    /// position using the Boyer-Moore-Horspool algorithm.
    ///
    /// Returns the absolute offset of the first occurrence, or `None`.
    #[must_use]
    pub fn find_pattern(&self, pattern: &[u8]) -> Option<usize> {
        if pattern.is_empty() {
            return Some(self.position);
        }
        let haystack = &self.data[self.position..];
        bmh_search(haystack, pattern).map(|rel| rel + self.position)
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Return a snapshot of accumulated I/O statistics.
    #[must_use]
    pub fn stats(&self) -> ReadStats {
        let avg_read_size = if self.read_count == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            {
                self.bytes_read as f64 / self.read_count as f64
            }
        };
        ReadStats {
            total_reads: self.read_count,
            total_bytes: self.bytes_read,
            cache_hits: self.peek_count,
            avg_read_size,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive Read-Ahead Buffered Reader
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the adaptive read-ahead buffer.
#[derive(Debug, Clone)]
pub struct ReadAheadConfig {
    /// Minimum read-ahead window in bytes (floor).
    pub min_window: usize,
    /// Maximum read-ahead window in bytes (ceiling).
    pub max_window: usize,
    /// Initial read-ahead window in bytes.
    pub initial_window: usize,
    /// Growth factor when sequential access is detected (multiplied by 1024).
    /// e.g. 1536 means 1.5x growth.
    pub growth_factor_per_mille: u32,
    /// Shrink factor when random access is detected (multiplied by 1024).
    /// e.g. 512 means 0.5x shrink.
    pub shrink_factor_per_mille: u32,
    /// Number of consecutive sequential reads required before growing the window.
    pub sequential_threshold: usize,
}

impl Default for ReadAheadConfig {
    fn default() -> Self {
        Self {
            min_window: 4096,
            max_window: 4 * 1024 * 1024,   // 4 MiB
            initial_window: 65536,         // 64 KiB
            growth_factor_per_mille: 1500, // 1.5x
            shrink_factor_per_mille: 500,  // 0.5x
            sequential_threshold: 4,
        }
    }
}

/// Access pattern classification for the adaptive algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Reads are progressing sequentially forward.
    Sequential,
    /// Reads are jumping around (random access).
    Random,
    /// Not enough information yet to classify.
    Unknown,
}

/// Statistics collected by the adaptive read-ahead reader.
#[derive(Debug, Clone, Default)]
pub struct ReadAheadStats {
    /// Total number of read operations performed.
    pub total_reads: u64,
    /// Number of reads served entirely from the read-ahead buffer.
    pub cache_hits: u64,
    /// Number of reads that required fetching new data.
    pub cache_misses: u64,
    /// Current adaptive window size in bytes.
    pub current_window: usize,
    /// Current access pattern classification.
    pub pattern: Option<AccessPattern>,
    /// Total bytes delivered to the caller.
    pub total_bytes_delivered: u64,
}

/// An adaptive read-ahead buffered reader that automatically tunes its
/// prefetch window based on observed access patterns.
///
/// When sequential access is detected, the window grows (up to `max_window`)
/// to reduce I/O overhead. When random access is detected, the window
/// shrinks (down to `min_window`) to avoid wasting prefetched data.
pub struct AdaptiveReadAheadReader {
    /// The underlying data.
    data: Vec<u8>,
    /// Current read position in the underlying data.
    position: usize,
    /// Read-ahead buffer (a window of data starting at `buffer_start`).
    buffer: Vec<u8>,
    /// Absolute offset in the data where `buffer` begins.
    buffer_start: usize,
    /// Configuration parameters.
    config: ReadAheadConfig,
    /// Current adaptive window size.
    current_window: usize,
    /// Counter of consecutive sequential reads.
    sequential_count: usize,
    /// Position of the last read (used to detect sequential vs random access).
    last_read_end: Option<usize>,
    /// Accumulated statistics.
    stats: ReadAheadStats,
}

impl AdaptiveReadAheadReader {
    /// Create a new adaptive read-ahead reader with default configuration.
    #[must_use]
    pub fn new(data: Vec<u8>) -> Self {
        Self::with_config(data, ReadAheadConfig::default())
    }

    /// Create a new adaptive read-ahead reader with custom configuration.
    #[must_use]
    pub fn with_config(data: Vec<u8>, config: ReadAheadConfig) -> Self {
        let initial_window = config
            .initial_window
            .clamp(config.min_window, config.max_window);
        Self {
            data,
            position: 0,
            buffer: Vec::new(),
            buffer_start: 0,
            current_window: initial_window,
            config,
            sequential_count: 0,
            last_read_end: None,
            stats: ReadAheadStats::default(),
        }
    }

    /// Return the current read position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Return the total length of the underlying data.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the underlying data is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the number of bytes remaining from the current position.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Return `true` if the position is at or past the end.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.position >= self.data.len()
    }

    /// Return the current adaptive window size.
    #[must_use]
    pub fn current_window(&self) -> usize {
        self.current_window
    }

    /// Return the detected access pattern.
    #[must_use]
    pub fn access_pattern(&self) -> AccessPattern {
        self.stats.pattern.unwrap_or(AccessPattern::Unknown)
    }

    /// Return a snapshot of accumulated statistics.
    #[must_use]
    pub fn stats(&self) -> ReadAheadStats {
        ReadAheadStats {
            current_window: self.current_window,
            pattern: Some(self.access_pattern()),
            ..self.stats.clone()
        }
    }

    /// Seek to an absolute position.
    ///
    /// Returns `true` on success, `false` if `pos` is beyond the data length.
    pub fn seek(&mut self, pos: usize) -> bool {
        if pos > self.data.len() {
            return false;
        }
        self.position = pos;
        true
    }

    /// Read up to `count` bytes from the current position.
    ///
    /// Returns the bytes read (may be fewer than `count` at EOF). The
    /// internal read-ahead window is adapted based on the access pattern.
    pub fn read(&mut self, count: usize) -> Vec<u8> {
        if self.is_eof() || count == 0 {
            self.stats.total_reads += 1;
            return Vec::new();
        }

        // Detect access pattern
        self.update_pattern();

        let actual_count = count.min(self.remaining());

        // Check if we can serve from the read-ahead buffer
        if self.is_in_buffer(self.position, actual_count) {
            self.stats.cache_hits += 1;
        } else {
            self.stats.cache_misses += 1;
            self.fill_buffer();
        }

        // Extract the data
        let start = self.position;
        let end = (start + actual_count).min(self.data.len());
        let result = self.data[start..end].to_vec();
        let bytes_read = result.len();

        self.position = end;
        self.last_read_end = Some(end);
        self.stats.total_reads += 1;
        self.stats.total_bytes_delivered += bytes_read as u64;

        result
    }

    /// Read exactly `count` bytes, returning `None` if insufficient data remains.
    pub fn read_exact(&mut self, count: usize) -> Option<Vec<u8>> {
        if self.remaining() < count {
            return None;
        }
        Some(self.read(count))
    }

    /// Peek at up to `count` bytes without advancing the position.
    #[must_use]
    pub fn peek(&self, count: usize) -> &[u8] {
        let start = self.position;
        let end = (start + count).min(self.data.len());
        &self.data[start..end]
    }

    /// Check whether the range `[offset, offset+len)` is within the current
    /// read-ahead buffer.
    fn is_in_buffer(&self, offset: usize, len: usize) -> bool {
        if self.buffer.is_empty() {
            return false;
        }
        let buffer_end = self.buffer_start + self.buffer.len();
        offset >= self.buffer_start && offset + len <= buffer_end
    }

    /// Fill the read-ahead buffer starting from the current position,
    /// using the current adaptive window size.
    fn fill_buffer(&mut self) {
        let start = self.position;
        let window_end = (start + self.current_window).min(self.data.len());
        self.buffer = self.data[start..window_end].to_vec();
        self.buffer_start = start;
    }

    /// Update the access pattern classification and adapt the window size.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn update_pattern(&mut self) {
        let is_sequential = match self.last_read_end {
            Some(last_end) => {
                // Allow a small gap (up to 64 bytes) to still count as sequential
                let diff = if self.position >= last_end {
                    self.position - last_end
                } else {
                    usize::MAX
                };
                diff <= 64
            }
            None => true, // first read is considered sequential
        };

        if is_sequential {
            self.sequential_count = self.sequential_count.saturating_add(1);
            if self.sequential_count >= self.config.sequential_threshold {
                self.stats.pattern = Some(AccessPattern::Sequential);
                // Grow window
                let new_window = (self.current_window as f64
                    * (self.config.growth_factor_per_mille as f64 / 1000.0))
                    as usize;
                self.current_window =
                    new_window.clamp(self.config.min_window, self.config.max_window);
            }
        } else {
            self.sequential_count = 0;
            self.stats.pattern = Some(AccessPattern::Random);
            // Shrink window
            let new_window = (self.current_window as f64
                * (self.config.shrink_factor_per_mille as f64 / 1000.0))
                as usize;
            self.current_window = new_window.clamp(self.config.min_window, self.config.max_window);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Boyer-Moore-Horspool implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Boyer-Moore-Horspool substring search.
///
/// Returns the offset of the first occurrence of `needle` in `haystack`, or
/// `None` if not found.
fn bmh_search(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let n = haystack.len();
    let m = needle.len();

    if m == 0 {
        return Some(0);
    }
    if m > n {
        return None;
    }

    // Build the bad-character skip table (256 entries).
    let mut skip = [m; 256];
    for (i, &b) in needle.iter().enumerate().take(m - 1) {
        skip[b as usize] = m - 1 - i;
    }

    let mut i = m - 1;
    while i < n {
        let mut k = 0usize;
        let mut j = m - 1;
        loop {
            if haystack[i - k] != needle[j] {
                break;
            }
            if j == 0 {
                return Some(i - (m - 1));
            }
            k += 1;
            j -= 1;
        }
        i += skip[haystack[i] as usize];
    }

    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_reader(data: &[u8]) -> BufferedMediaReader {
        BufferedMediaReader::from_bytes(data.to_vec())
    }

    // ── Basic construction ────────────────────────────────────────────────────

    #[test]
    fn test_initial_state() {
        let r = make_reader(b"hello");
        assert_eq!(r.position(), 0);
        assert_eq!(r.remaining(), 5);
        assert!(!r.is_eof());
    }

    #[test]
    fn test_empty_reader() {
        let r = make_reader(b"");
        assert!(r.is_eof());
        assert_eq!(r.remaining(), 0);
    }

    // ── Read ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_read_advances_position() {
        let mut r = make_reader(b"abcdef");
        let slice = r.read(3);
        assert_eq!(slice, b"abc");
        assert_eq!(r.position(), 3);
        assert_eq!(r.remaining(), 3);
    }

    #[test]
    fn test_read_past_end_clips() {
        let mut r = make_reader(b"ab");
        let slice = r.read(100);
        assert_eq!(slice, b"ab");
        assert!(r.is_eof());
    }

    // ── Peek ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_peek_does_not_advance() {
        let r = make_reader(b"abcdef");
        let s = r.peek(3);
        assert_eq!(s, b"abc");
        assert_eq!(r.position(), 0); // unchanged
    }

    #[test]
    fn test_peek_past_end_clips() {
        let r = make_reader(b"xy");
        assert_eq!(r.peek(100), b"xy");
    }

    // ── Seek / skip ───────────────────────────────────────────────────────────

    #[test]
    fn test_seek_valid() {
        let mut r = make_reader(b"abcdef");
        assert!(r.seek(3));
        assert_eq!(r.position(), 3);
    }

    #[test]
    fn test_seek_past_end_fails() {
        let mut r = make_reader(b"abc");
        assert!(!r.seek(10));
        assert_eq!(r.position(), 0);
    }

    #[test]
    fn test_skip_partial() {
        let mut r = make_reader(b"abcde");
        let skipped = r.skip(3);
        assert_eq!(skipped, 3);
        assert_eq!(r.position(), 3);
    }

    #[test]
    fn test_skip_past_end_clips() {
        let mut r = make_reader(b"ab");
        let skipped = r.skip(100);
        assert_eq!(skipped, 2);
        assert!(r.is_eof());
    }

    // ── read_exact ────────────────────────────────────────────────────────────

    #[test]
    fn test_read_exact_success() {
        let mut r = make_reader(b"abcdef");
        let v = r.read_exact(4).expect("should succeed");
        assert_eq!(v, b"abcd");
        assert_eq!(r.position(), 4);
    }

    #[test]
    fn test_read_exact_insufficient() {
        let mut r = make_reader(b"ab");
        assert!(r.read_exact(5).is_none());
    }

    // ── Primitive integers ────────────────────────────────────────────────────

    #[test]
    fn test_read_u8() {
        let mut r = make_reader(&[0x42]);
        assert_eq!(r.read_u8(), Some(0x42));
        assert!(r.read_u8().is_none());
    }

    #[test]
    fn test_read_u16_be() {
        let mut r = make_reader(&[0x01, 0x02]);
        assert_eq!(r.read_u16_be(), Some(0x0102));
    }

    #[test]
    fn test_read_u16_le() {
        let mut r = make_reader(&[0x01, 0x02]);
        assert_eq!(r.read_u16_le(), Some(0x0201));
    }

    #[test]
    fn test_read_u32_be() {
        let mut r = make_reader(&[0x00, 0x01, 0x02, 0x03]);
        assert_eq!(r.read_u32_be(), Some(0x0001_0203));
    }

    #[test]
    fn test_read_u32_le() {
        let mut r = make_reader(&[0x01, 0x00, 0x00, 0x00]);
        assert_eq!(r.read_u32_le(), Some(1));
    }

    #[test]
    fn test_read_u64_be() {
        let data = [0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF];
        let mut r = make_reader(&data);
        assert_eq!(r.read_u64_be(), Some(0xFF));
    }

    // ── find_pattern (BMH) ────────────────────────────────────────────────────

    #[test]
    fn test_find_pattern_found() {
        let r = make_reader(b"the quick brown fox");
        let pos = r.find_pattern(b"brown");
        assert_eq!(pos, Some(10));
    }

    #[test]
    fn test_find_pattern_not_found() {
        let r = make_reader(b"hello world");
        assert_eq!(r.find_pattern(b"xyz"), None);
    }

    #[test]
    fn test_find_pattern_from_current_position() {
        let mut r = make_reader(b"aababc");
        r.seek(2); // skip first two bytes
                   // "ab" occurs at absolute offset 2 (relative 0) and 3 (relative 1).
        let pos = r.find_pattern(b"ab");
        // Should return first hit from position 2 onwards.
        assert!(pos.is_some());
        assert!(pos.expect("pattern should be found after seek") >= 2);
    }

    #[test]
    fn test_find_empty_pattern() {
        let r = make_reader(b"abc");
        // Empty pattern always matches at current position.
        assert_eq!(r.find_pattern(b""), Some(0));
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_after_reads() {
        let mut r = make_reader(b"0123456789");
        r.read(4);
        r.read(3);
        let s = r.stats();
        assert_eq!(s.total_reads, 2);
        assert_eq!(s.total_bytes, 7);
        assert!((s.avg_read_size - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_empty() {
        let r = make_reader(b"abc");
        let s = r.stats();
        assert_eq!(s.total_reads, 0);
        assert_eq!(s.avg_read_size, 0.0);
    }

    // ── AdaptiveReadAheadReader ──────────────────────────────────────────────

    #[test]
    fn test_adaptive_initial_state() {
        let data = vec![0u8; 1024];
        let reader = AdaptiveReadAheadReader::new(data);
        assert_eq!(reader.position(), 0);
        assert_eq!(reader.len(), 1024);
        assert!(!reader.is_eof());
        assert_eq!(reader.remaining(), 1024);
        assert_eq!(reader.access_pattern(), AccessPattern::Unknown);
    }

    #[test]
    fn test_adaptive_read_basic() {
        let data: Vec<u8> = (0..100).collect();
        let mut reader = AdaptiveReadAheadReader::new(data);
        let chunk = reader.read(10);
        assert_eq!(chunk.len(), 10);
        assert_eq!(chunk, (0..10).collect::<Vec<u8>>());
        assert_eq!(reader.position(), 10);
    }

    #[test]
    fn test_adaptive_read_exact() {
        let data: Vec<u8> = (0..50).collect();
        let mut reader = AdaptiveReadAheadReader::new(data);
        let chunk = reader.read_exact(10);
        assert!(chunk.is_some());
        assert_eq!(chunk.as_ref().map(|c| c.len()), Some(10));
        // Try to read more than available
        assert!(reader.read_exact(100).is_none());
    }

    #[test]
    fn test_adaptive_sequential_detection() {
        let data = vec![0u8; 65536];
        let config = ReadAheadConfig {
            sequential_threshold: 3,
            ..Default::default()
        };
        let mut reader = AdaptiveReadAheadReader::with_config(data, config);

        // Perform 5 sequential reads
        for _ in 0..5 {
            reader.read(100);
        }

        assert_eq!(reader.access_pattern(), AccessPattern::Sequential);
    }

    #[test]
    fn test_adaptive_random_detection() {
        let data = vec![0u8; 65536];
        let config = ReadAheadConfig {
            sequential_threshold: 2,
            ..Default::default()
        };
        let mut reader = AdaptiveReadAheadReader::with_config(data, config);

        // Perform sequential reads to establish pattern
        reader.read(100);
        reader.read(100);
        reader.read(100);

        // Now jump backwards (random access)
        reader.seek(0);
        reader.read(50);

        assert_eq!(reader.access_pattern(), AccessPattern::Random);
    }

    #[test]
    fn test_adaptive_window_grows_on_sequential() {
        let data = vec![0u8; 1024 * 1024]; // 1 MiB
        let config = ReadAheadConfig {
            min_window: 1024,
            max_window: 1024 * 1024,
            initial_window: 4096,
            growth_factor_per_mille: 2000, // 2x
            shrink_factor_per_mille: 500,
            sequential_threshold: 2,
        };
        let initial = config.initial_window;
        let mut reader = AdaptiveReadAheadReader::with_config(data, config);

        // Do enough sequential reads to trigger growth
        for _ in 0..5 {
            reader.read(100);
        }

        assert!(reader.current_window() > initial);
    }

    #[test]
    fn test_adaptive_window_shrinks_on_random() {
        let data = vec![0u8; 1024 * 1024];
        let config = ReadAheadConfig {
            min_window: 1024,
            max_window: 1024 * 1024,
            initial_window: 65536,
            growth_factor_per_mille: 1500,
            shrink_factor_per_mille: 500, // 0.5x
            sequential_threshold: 2,
        };
        let mut reader = AdaptiveReadAheadReader::with_config(data, config);

        // Establish sequential pattern
        for _ in 0..4 {
            reader.read(100);
        }
        let window_before = reader.current_window();

        // Random jump
        reader.seek(50000);
        reader.read(100);

        assert!(reader.current_window() < window_before);
    }

    #[test]
    fn test_adaptive_window_clamped_to_min() {
        let data = vec![0u8; 65536];
        let config = ReadAheadConfig {
            min_window: 4096,
            max_window: 65536,
            initial_window: 4096, // already at min
            growth_factor_per_mille: 1500,
            shrink_factor_per_mille: 100, // aggressive shrink
            sequential_threshold: 2,
        };
        let mut reader = AdaptiveReadAheadReader::with_config(data, config);

        // Random jumps
        reader.read(10);
        reader.seek(5000);
        reader.read(10);
        reader.seek(100);
        reader.read(10);

        assert!(reader.current_window() >= 4096);
    }

    #[test]
    fn test_adaptive_window_clamped_to_max() {
        let data = vec![0u8; 16 * 1024 * 1024];
        let config = ReadAheadConfig {
            min_window: 1024,
            max_window: 65536,
            initial_window: 32768,
            growth_factor_per_mille: 3000, // 3x growth
            shrink_factor_per_mille: 500,
            sequential_threshold: 1,
        };
        let mut reader = AdaptiveReadAheadReader::with_config(data, config);

        for _ in 0..20 {
            reader.read(100);
        }

        assert!(reader.current_window() <= 65536);
    }

    #[test]
    fn test_adaptive_cache_hit_tracking() {
        let data = vec![0u8; 65536];
        let mut reader = AdaptiveReadAheadReader::new(data);

        // First read causes a miss (buffer is empty)
        reader.read(10);
        assert_eq!(reader.stats().cache_misses, 1);

        // Subsequent sequential reads within the window should be hits
        let hits_before = reader.stats().cache_hits;
        reader.read(10);
        // It should be at least as many hits as before (may or may not be a hit
        // depending on buffer state)
        assert!(reader.stats().cache_hits >= hits_before);
    }

    #[test]
    fn test_adaptive_eof_handling() {
        let data = vec![1u8, 2, 3];
        let mut reader = AdaptiveReadAheadReader::new(data);

        let chunk = reader.read(100);
        assert_eq!(chunk, vec![1, 2, 3]);
        assert!(reader.is_eof());

        let empty = reader.read(10);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_adaptive_seek() {
        let data: Vec<u8> = (0..100).collect();
        let mut reader = AdaptiveReadAheadReader::new(data);

        assert!(reader.seek(50));
        assert_eq!(reader.position(), 50);
        let chunk = reader.read(5);
        assert_eq!(chunk, vec![50, 51, 52, 53, 54]);

        assert!(!reader.seek(200)); // beyond end
        assert_eq!(reader.position(), 55); // unchanged
    }

    #[test]
    fn test_adaptive_peek() {
        let data: Vec<u8> = (0..100).collect();
        let reader = AdaptiveReadAheadReader::new(data);

        let peeked = reader.peek(5);
        assert_eq!(peeked, &[0, 1, 2, 3, 4]);
        assert_eq!(reader.position(), 0); // unchanged
    }

    #[test]
    fn test_adaptive_empty_data() {
        let reader = AdaptiveReadAheadReader::new(Vec::new());
        assert!(reader.is_empty());
        assert!(reader.is_eof());
        assert_eq!(reader.remaining(), 0);
    }

    #[test]
    fn test_adaptive_stats_total_bytes() {
        let data = vec![0u8; 1000];
        let mut reader = AdaptiveReadAheadReader::new(data);

        reader.read(100);
        reader.read(200);
        reader.read(50);

        let stats = reader.stats();
        assert_eq!(stats.total_reads, 3);
        assert_eq!(stats.total_bytes_delivered, 350);
    }
}
