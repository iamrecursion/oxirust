#![allow(dead_code)]

//! Rolling hash for content-defined chunking in media deduplication.
//!
//! This module implements the Buzhash and Rabin-style rolling hash algorithms
//! used for content-defined chunking (CDC). CDC splits a byte stream at
//! boundaries determined by the content itself, which ensures that
//! insertions or deletions in one part of the stream do not shift all
//! subsequent chunk boundaries.
//!
//! # Key Types
//!
//! - [`BuzHash`] - Buzhash rolling hash with configurable window
//! - [`ChunkBoundary`] - A detected chunk boundary with offset and hash
//! - [`ChunkerConfig`] - Configuration for content-defined chunking
//! - [`ContentChunker`] - Splits a byte stream into content-defined chunks
//! - [`RollingHashStream`] - Streaming Rabin-fingerprint iterator over `Read` sources

use std::collections::VecDeque;
use std::io::{self, Read};

/// Configuration for the content-defined chunker.
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Minimum chunk size in bytes.
    pub min_chunk: usize,
    /// Maximum chunk size in bytes.
    pub max_chunk: usize,
    /// Target (average) chunk size in bytes.
    pub target_chunk: usize,
    /// Rolling hash window size.
    pub window_size: usize,
    /// Mask bits used to detect chunk boundaries.
    /// A boundary is declared when `(hash & mask) == 0`.
    pub mask_bits: u32,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            min_chunk: 2048,
            max_chunk: 65536,
            target_chunk: 8192,
            window_size: 48,
            mask_bits: 13, // 2^13 = 8192 average
        }
    }
}

impl ChunkerConfig {
    /// Create a config for small media chunks (e.g. subtitle segments).
    #[must_use]
    pub fn small() -> Self {
        Self {
            min_chunk: 512,
            max_chunk: 8192,
            target_chunk: 2048,
            window_size: 32,
            mask_bits: 11,
        }
    }

    /// Create a config for large media chunks (e.g. video segments).
    #[must_use]
    pub fn large() -> Self {
        Self {
            min_chunk: 16384,
            max_chunk: 524_288,
            target_chunk: 65536,
            window_size: 64,
            mask_bits: 16,
        }
    }

    /// Compute the chunk boundary mask from `mask_bits`.
    #[must_use]
    pub fn boundary_mask(&self) -> u64 {
        (1u64 << self.mask_bits) - 1
    }

    /// Validate the configuration.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.min_chunk > 0
            && self.max_chunk >= self.min_chunk
            && self.target_chunk >= self.min_chunk
            && self.target_chunk <= self.max_chunk
            && self.window_size > 0
            && self.mask_bits > 0
            && self.mask_bits <= 32
    }
}

/// Pre-computed byte hash table for Buzhash (256 random entries).
const BUZHASH_TABLE: [u64; 256] = {
    let mut table = [0u64; 256];
    // Use a simple deterministic PRNG to fill the table at compile time.
    let mut state: u64 = 0x5555_5555_5555_5555;
    let mut i = 0;
    while i < 256 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        table[i] = state;
        i += 1;
    }
    table
};

/// Buzhash rolling hash.
///
/// Maintains a sliding window over the input and computes a rolling hash
/// that can be updated in O(1) as bytes enter and leave the window.
#[derive(Clone)]
pub struct BuzHash {
    /// Current hash value.
    hash: u64,
    /// The sliding window buffer.
    window: Vec<u8>,
    /// Current position in the circular window buffer.
    window_pos: usize,
    /// Window size.
    window_size: usize,
    /// Number of bytes fed so far (capped at window_size for warm-up).
    count: usize,
}

impl BuzHash {
    /// Create a new Buzhash with the given window size.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            hash: 0,
            window: vec![0u8; window_size],
            window_pos: 0,
            window_size,
            count: 0,
        }
    }

    /// Feed a single byte and return the updated hash.
    pub fn update(&mut self, byte: u8) -> u64 {
        let out_byte = self.window[self.window_pos];
        self.window[self.window_pos] = byte;
        self.window_pos = (self.window_pos + 1) % self.window_size;

        // Rotate left by 1
        self.hash = self.hash.rotate_left(1);
        // XOR in the new byte
        self.hash ^= BUZHASH_TABLE[byte as usize];

        if self.count >= self.window_size {
            // XOR out the old byte (rotated by window_size)
            self.hash ^= BUZHASH_TABLE[out_byte as usize].rotate_left(self.window_size as u32);
        } else {
            self.count += 1;
        }

        self.hash
    }

    /// Return the current hash value.
    #[must_use]
    pub fn value(&self) -> u64 {
        self.hash
    }

    /// Return how many bytes have been fed.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset the hash state.
    pub fn reset(&mut self) {
        self.hash = 0;
        self.window.fill(0);
        self.window_pos = 0;
        self.count = 0;
    }
}

impl std::fmt::Debug for BuzHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuzHash")
            .field("hash", &format_args!("0x{:016x}", self.hash))
            .field("window_size", &self.window_size)
            .field("count", &self.count)
            .finish()
    }
}

/// A detected chunk boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkBoundary {
    /// Byte offset of the boundary in the input stream.
    pub offset: usize,
    /// Rolling hash value at the boundary.
    pub hash: u64,
    /// Length of the chunk ending at this boundary.
    pub chunk_len: usize,
}

/// Splits a byte stream into content-defined chunks.
pub struct ContentChunker {
    /// Configuration.
    config: ChunkerConfig,
    /// Rolling hash.
    hasher: BuzHash,
    /// Current position in the overall stream.
    position: usize,
    /// Position of the last boundary.
    last_boundary: usize,
    /// Detected boundaries.
    boundaries: Vec<ChunkBoundary>,
}

impl ContentChunker {
    /// Create a new chunker with the given configuration.
    #[must_use]
    pub fn new(config: ChunkerConfig) -> Self {
        let hasher = BuzHash::new(config.window_size);
        Self {
            config,
            hasher,
            position: 0,
            last_boundary: 0,
            boundaries: Vec::new(),
        }
    }

    /// Create a chunker with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ChunkerConfig::default())
    }

    /// Feed a chunk of data and detect boundaries within it.
    ///
    /// Returns the boundaries found in this batch.
    pub fn feed(&mut self, data: &[u8]) -> Vec<ChunkBoundary> {
        let mask = self.config.boundary_mask();
        let mut found = Vec::new();

        for &byte in data {
            let h = self.hasher.update(byte);
            self.position += 1;

            let chunk_len = self.position - self.last_boundary;

            // Enforce minimum chunk size
            if chunk_len < self.config.min_chunk {
                continue;
            }

            // Check for boundary or max chunk size reached
            let is_boundary = (h & mask) == 0 || chunk_len >= self.config.max_chunk;

            if is_boundary {
                let boundary = ChunkBoundary {
                    offset: self.position,
                    hash: h,
                    chunk_len,
                };
                found.push(boundary.clone());
                self.boundaries.push(boundary);
                self.last_boundary = self.position;
            }
        }

        found
    }

    /// Finalise the chunker, emitting a boundary for any trailing data.
    pub fn finish(&mut self) -> Option<ChunkBoundary> {
        let chunk_len = self.position - self.last_boundary;
        if chunk_len > 0 {
            let boundary = ChunkBoundary {
                offset: self.position,
                hash: self.hasher.value(),
                chunk_len,
            };
            self.boundaries.push(boundary.clone());
            self.last_boundary = self.position;
            Some(boundary)
        } else {
            None
        }
    }

    /// Return all detected boundaries so far.
    #[must_use]
    pub fn boundaries(&self) -> &[ChunkBoundary] {
        &self.boundaries
    }

    /// Return the current stream position.
    #[must_use]
    pub fn position(&self) -> usize {
        self.position
    }

    /// Reset the chunker state.
    pub fn reset(&mut self) {
        self.hasher.reset();
        self.position = 0;
        self.last_boundary = 0;
        self.boundaries.clear();
    }
}

/// Convenience function: chunk a complete byte slice.
#[must_use]
pub fn chunk_bytes(data: &[u8], config: ChunkerConfig) -> Vec<ChunkBoundary> {
    let mut chunker = ContentChunker::new(config);
    let mut all = chunker.feed(data);
    if let Some(last) = chunker.finish() {
        all.push(last);
    }
    all
}

// ── Rabin-fingerprint streaming iterator ─────────────────────────────────────

/// Internal read buffer size used by [`RollingHashStream`].
const CHUNK_SIZE: usize = 65_536;

/// Rabin-fingerprint rolling hash multiplier (odd prime for good diffusion).
const RABIN_BASE: u64 = 0x08D3_B1B9_ADFA_BC4D;

/// A streaming Rabin-fingerprint rolling hash over an arbitrary [`Read`] source.
///
/// Each call to `next()` advances the window by one byte and yields
/// `(byte_offset, hash)` where `byte_offset` is the 0-based index of the
/// leading byte of the current window.
///
/// The hash is computed as:
/// ```text
/// hash = (hash * BASE + byte_in) XOR (BASE^window_size * byte_out)
/// ```
/// with `BASE^window_size` pre-computed at construction time.
pub struct RollingHashStream<R: Read> {
    /// Wrapped reader.
    inner: R,
    /// Sliding window of the last `window_size` bytes.
    window: VecDeque<u8>,
    /// Window size in bytes.
    window_size: usize,
    /// Current hash value.
    hash: u64,
    /// Current byte offset (index of leading byte of window).
    pos: u64,
    /// Pre-computed `BASE^window_size` for O(1) removal of oldest byte.
    pow_table: u64,
    /// Read buffer (heap-allocated to avoid large stack arrays).
    buf: Box<[u8]>,
    /// Valid bytes in `buf`.
    buf_len: usize,
    /// Cursor inside `buf`.
    buf_pos: usize,
    /// Whether the underlying reader is exhausted.
    eof: bool,
    /// Whether the iterator has been fully consumed (including EOF state).
    done: bool,
}

impl<R: Read> RollingHashStream<R> {
    /// Create a new streaming rolling hash with the given window size.
    ///
    /// The window is "warmed up" by the first `window_size` bytes fed, after
    /// which each subsequent byte produces a yielded `(offset, hash)` pair.
    #[must_use]
    pub fn new(inner: R, window_size: usize) -> Self {
        let window_size = window_size.max(1);
        // Pre-compute BASE^window_size mod 2^64.
        let pow_table = (0..window_size).fold(1u64, |acc, _| acc.wrapping_mul(RABIN_BASE));
        Self {
            inner,
            window: VecDeque::with_capacity(window_size),
            window_size,
            hash: 0,
            pos: 0,
            pow_table,
            buf: vec![0u8; CHUNK_SIZE].into_boxed_slice(),
            buf_len: 0,
            buf_pos: 0,
            eof: false,
            done: false,
        }
    }

    /// Read the next byte from the buffered inner reader.
    ///
    /// Returns `Ok(Some(byte))`, `Ok(None)` at EOF, or `Err(io::Error)`.
    fn read_byte(&mut self) -> io::Result<Option<u8>> {
        if self.buf_pos >= self.buf_len {
            if self.eof {
                return Ok(None);
            }
            let n = self.inner.read(&mut self.buf[..])?;
            if n == 0 {
                self.eof = true;
                return Ok(None);
            }
            self.buf_len = n;
            self.buf_pos = 0;
        }
        let byte = self.buf[self.buf_pos];
        self.buf_pos += 1;
        Ok(Some(byte))
    }
}

impl<R: Read> Iterator for RollingHashStream<R> {
    /// `(byte_offset_of_window_start, rolling_hash)`.
    type Item = io::Result<(u64, u64)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        // Feed bytes until the window is full and we can yield.
        loop {
            let byte = match self.read_byte() {
                Ok(Some(b)) => b,
                Ok(None) => {
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };

            // Remove the oldest byte from the window (if full).
            let byte_out = if self.window.len() == self.window_size {
                self.window.pop_front()
            } else {
                None
            };

            self.window.push_back(byte);

            // Update Rabin hash:
            //   hash = hash * BASE + byte_in
            //   hash ^= pow(BASE, window_size) * byte_out  (if window was full)
            self.hash = self
                .hash
                .wrapping_mul(RABIN_BASE)
                .wrapping_add(u64::from(byte));
            if let Some(out) = byte_out {
                self.hash ^= self.pow_table.wrapping_mul(u64::from(out));
            }

            self.pos += 1;

            // Only yield once the window is fully filled.
            if self.window.len() == self.window_size {
                let window_start = self.pos - self.window_size as u64;
                return Some(Ok((window_start, self.hash)));
            }
        }
    }
}

/// Compute rolling hashes over a byte slice using the same Rabin formula as
/// [`RollingHashStream`], for comparison and testing.
///
/// Returns one `(offset, hash)` per position once the window is filled.
#[must_use]
pub fn rolling_hash_slice(data: &[u8], window_size: usize) -> Vec<(u64, u64)> {
    let window_size = window_size.max(1);
    let pow_table = (0..window_size).fold(1u64, |acc, _| acc.wrapping_mul(RABIN_BASE));
    let mut window: VecDeque<u8> = VecDeque::with_capacity(window_size);
    let mut hash: u64 = 0;
    let mut results = Vec::with_capacity(data.len().saturating_sub(window_size) + 1);

    for (i, &byte) in data.iter().enumerate() {
        let byte_out = if window.len() == window_size {
            window.pop_front()
        } else {
            None
        };
        window.push_back(byte);
        hash = hash.wrapping_mul(RABIN_BASE).wrapping_add(u64::from(byte));
        if let Some(out) = byte_out {
            hash ^= pow_table.wrapping_mul(u64::from(out));
        }
        if window.len() == window_size {
            let offset = (i + 1 - window_size) as u64;
            results.push((offset, hash));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_config_default() {
        let cfg = ChunkerConfig::default();
        assert_eq!(cfg.min_chunk, 2048);
        assert_eq!(cfg.max_chunk, 65536);
        assert_eq!(cfg.target_chunk, 8192);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_chunker_config_small() {
        let cfg = ChunkerConfig::small();
        assert_eq!(cfg.min_chunk, 512);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_chunker_config_large() {
        let cfg = ChunkerConfig::large();
        assert_eq!(cfg.min_chunk, 16384);
        assert!(cfg.is_valid());
    }

    #[test]
    fn test_chunker_config_boundary_mask() {
        let cfg = ChunkerConfig::default(); // mask_bits = 13
        assert_eq!(cfg.boundary_mask(), (1 << 13) - 1);
    }

    #[test]
    fn test_buzhash_new() {
        let h = BuzHash::new(32);
        assert_eq!(h.value(), 0);
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn test_buzhash_deterministic() {
        let mut h1 = BuzHash::new(16);
        let mut h2 = BuzHash::new(16);
        for b in b"identical input" {
            h1.update(*b);
            h2.update(*b);
        }
        assert_eq!(h1.value(), h2.value());
    }

    #[test]
    fn test_buzhash_different_input() {
        let mut h1 = BuzHash::new(16);
        let mut h2 = BuzHash::new(16);
        for b in b"input A" {
            h1.update(*b);
        }
        for b in b"input B" {
            h2.update(*b);
        }
        assert_ne!(h1.value(), h2.value());
    }

    #[test]
    fn test_buzhash_reset() {
        let mut h = BuzHash::new(8);
        for b in b"some data" {
            h.update(*b);
        }
        assert_ne!(h.value(), 0);
        h.reset();
        assert_eq!(h.value(), 0);
        assert_eq!(h.count(), 0);
    }

    #[test]
    fn test_content_chunker_small_input() {
        // Input smaller than min_chunk => only finish() produces boundary
        let config = ChunkerConfig {
            min_chunk: 100,
            max_chunk: 1000,
            target_chunk: 500,
            window_size: 8,
            mask_bits: 3,
        };
        let mut chunker = ContentChunker::new(config);
        let data = vec![0x42u8; 50];
        let during = chunker.feed(&data);
        assert!(during.is_empty()); // too small for any boundary
        let last = chunker.finish();
        assert!(last.is_some());
        assert_eq!(last.expect("operation should succeed").chunk_len, 50);
    }

    #[test]
    fn test_content_chunker_max_chunk() {
        // Ensure max_chunk is enforced even if hash never triggers
        let config = ChunkerConfig {
            min_chunk: 4,
            max_chunk: 16,
            target_chunk: 8,
            window_size: 4,
            mask_bits: 30, // extremely unlikely to trigger via hash
        };
        let mut chunker = ContentChunker::new(config);
        let data = vec![0u8; 100];
        let boundaries = chunker.feed(&data);
        // Should get boundaries at multiples of 16 (max_chunk)
        assert!(!boundaries.is_empty());
        for b in &boundaries {
            assert!(b.chunk_len <= 16);
        }
    }

    #[test]
    fn test_chunk_bytes_convenience() {
        let data = vec![0xABu8; 200];
        let config = ChunkerConfig {
            min_chunk: 10,
            max_chunk: 50,
            target_chunk: 30,
            window_size: 4,
            mask_bits: 30,
        };
        let boundaries = chunk_bytes(&data, config);
        assert!(!boundaries.is_empty());

        // Sum of chunk lengths should equal data length
        let total: usize = boundaries.iter().map(|b| b.chunk_len).sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn test_content_chunker_reset() {
        let mut chunker = ContentChunker::with_defaults();
        chunker.feed(&vec![1u8; 100_000]);
        assert!(chunker.position() > 0);
        chunker.reset();
        assert_eq!(chunker.position(), 0);
        assert!(chunker.boundaries().is_empty());
    }

    #[test]
    fn test_chunk_boundary_equality() {
        let a = ChunkBoundary {
            offset: 100,
            hash: 42,
            chunk_len: 50,
        };
        let b = ChunkBoundary {
            offset: 100,
            hash: 42,
            chunk_len: 50,
        };
        assert_eq!(a, b);
    }

    // ── RollingHashStream tests ───────────────────────────────────────────────

    #[test]
    fn test_rolling_hash_stream_matches_slice() {
        // Feed the same data through both APIs and assert identical results.
        let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
        let window_size = 32;

        // Slice-based reference.
        let expected = rolling_hash_slice(&data, window_size);

        // Stream-based.
        let cursor = std::io::Cursor::new(&data);
        let stream = RollingHashStream::new(cursor, window_size);
        let actual: Vec<(u64, u64)> = stream
            .map(|r| r.expect("stream should not error"))
            .collect();

        assert_eq!(
            expected.len(),
            actual.len(),
            "number of hash pairs must match"
        );
        for (i, (exp, got)) in expected.iter().zip(actual.iter()).enumerate() {
            assert_eq!(
                exp, got,
                "hash mismatch at position {i}: expected {exp:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn test_rolling_hash_stream_large_data() {
        // Stream 4 MB of synthetic data and verify no panic and reasonable count.
        const MB4: usize = 4 * 1024 * 1024;
        let data: Vec<u8> = (0u8..=255).cycle().take(MB4).collect();
        let window_size = 64;

        let cursor = std::io::Cursor::new(&data);
        let stream = RollingHashStream::new(cursor, window_size);
        let mut count = 0usize;
        for item in stream {
            let _ = item.expect("stream should not error");
            count += 1;
        }

        // Expect exactly MB4 - window_size + 1 hash pairs.
        let expected_count = MB4 - window_size + 1;
        assert_eq!(
            count, expected_count,
            "expected {expected_count} hash pairs, got {count}"
        );
    }
}
